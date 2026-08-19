//! Sinker impl for the packed UYVA source format — 8-bit packed YUV 4:4:4
//! with real source alpha, chroma-first channel order.
//!
//! UYVA (FFmpeg `AV_PIX_FMT_UYVA`) packs **four u8 bytes per pixel**
//! (`[U, Y, V, A]`). The A byte (offset 3) is **real source alpha** — not
//! padding. It is the chroma-first channel re-ordering of
//! [`Vuya`](crate::source::Vuya); only the byte positions differ, so this
//! sink mirrors the `Vuya` sink exactly with the UYVA row kernels.
//!
//! Outputs map to the sink's standard channels:
//! - `with_rgb` — packed YUV → RGB 8-bit pipeline; alpha discarded.
//! - `with_rgba` — packed YUV → RGBA 8-bit pipeline; **source α byte
//!   is passed through** verbatim from byte 3 of each pixel.
//! - `with_luma` — extracts the Y byte at offset 1 of each pixel directly.
//! - `with_luma_u16` — zero-extends the Y byte to u16.
//! - `with_hsv` — stages u8 RGB into the user's RGB buffer (if attached)
//!   or a scratch buffer, then runs `rgb_to_hsv_row`.
//!
//! Alpha semantics are identical to `Vuya` (§ 7.2 / § 7.3): standalone
//! RGBA runs `uyva_to_rgba_row` directly (source α through the kernel);
//! the RGB + RGBA combo derives RGBA from the computed RGB row then
//! overwrites the α slot from packed source byte 3 via
//! `alpha_extract::copy_alpha_packed_u8x4_at_0`.

use super::{
  GeometryOverflow, InsufficientBuffer, MixedSinker, MixedSinkerError, RowIndexOutOfRange,
  RowShapeMismatch, RowSlice, check_dimensions_match, check_frozen_alpha_mode,
  packed_yuva444_filter_resample, packed_yuva444_resample, rgb_row_buf_or_scratch,
  rgba_plane_row_slice,
};
use crate::{
  PixelSink,
  row::{
    expand_rgb_to_rgba_row, rgb_to_hsv_row, uyva_to_hsv_row, uyva_to_luma_row,
    uyva_to_luma_u16_row, uyva_to_rgb_row, uyva_to_rgba_row,
  },
  source::{Uyva, UyvaRow, UyvaSink},
};

impl<'a, R> MixedSinker<'a, Uyva, R> {
  /// Attaches a **`u16`** luma output buffer. Y bytes from the packed UYVA
  /// `[U, Y, V, A]` layout are zero-extended to u16 (`out[x] = Y_byte as
  /// u16`). Length in u16 **elements** (`width x height`).
  ///
  /// Returns `Err(InsufficientLumaU16Buffer)` if `buf.len() < width x height`,
  /// or `Err(GeometryOverflow)` on 32-bit targets.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_luma_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_luma_u16(buf)?;
    Ok(self)
  }
  /// In-place variant of [`with_luma_u16`](Self::with_luma_u16).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_luma_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_pixels()?;
    if buf.len() < expected {
      return Err(MixedSinkerError::InsufficientLumaU16Buffer(
        InsufficientBuffer::new(expected, buf.len()),
      ));
    }
    self.luma_u16 = Some(buf);
    Ok(self)
  }

  /// Attaches a packed **8-bit** RGBA output buffer. When UYVA is the
  /// source, the per-pixel alpha byte is **sourced from the A byte of each
  /// pixel quadruple** (offset 3) — not forced to `0xFF`.
  ///
  /// Returns `Err(InsufficientRgbaBuffer)` if `buf.len() < width x height x 4`,
  /// or `Err(GeometryOverflow)` on 32‑bit targets when the product overflows.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgba(mut self, buf: &'a mut [u8]) -> Result<Self, MixedSinkerError> {
    self.set_rgba(buf)?;
    Ok(self)
  }

  /// In-place variant of [`with_rgba`](Self::with_rgba).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgba(&mut self, buf: &'a mut [u8]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_elems(4)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::InsufficientRgbaBuffer(
        InsufficientBuffer::new(expected, buf.len()),
      ));
    }
    self.rgba = Some(buf);
    Ok(self)
  }
}

impl<R> UyvaSink for MixedSinker<'_, Uyva, R> {}

impl<R> PixelSink for MixedSinker<'_, Uyva, R> {
  type Input<'r> = UyvaRow<'r>;
  type Error = MixedSinkerError;

  #[cfg_attr(not(tarpaulin), inline(always))]
  fn kernel_matrix(&self) -> crate::KernelMatrix {
    self.kernel_matrix
  }
  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    // New frame: restart the 4-channel u8 RGBA colour stream and the
    // independent native-Y u16 luma stream, re-arm the alpha-mode snapshot.
    // `Uyva` exposes no u16 colour outputs, so its u16 RGBA streams are
    // never created.
    if let Some(stream) = self.rgba_stream.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.luma_stream_u16.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.rgba_filter_stream.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.luma_filter_stream_u16.as_mut() {
      stream.reset();
    }
    self.resample_outputs = None;
    self.frozen_alpha_mode = Some(self.alpha_mode);
    Ok(())
  }

  fn process(&mut self, row: UyvaRow<'_>) -> Result<(), Self::Error> {
    const BITS: u32 = 8;
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    // UYVA row = `width x 4` bytes (one quadruple per pixel).
    let packed_expected =
      w.checked_mul(4)
        .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
          w, h, 4,
        )))?;
    if row.packed().len() != packed_expected {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::UyvaPacked,
        idx,
        packed_expected,
        row.packed().len(),
      )));
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange(
        RowIndexOutOfRange::new(idx, self.height),
      ));
    }

    // Non-identity plan: `Uyva` is packed 4:4:4 YUV with real source alpha
    // (the A byte at offset 3). Route through the packed-YUVA tail at
    // `SRC_BITS = 8`, exactly like `Vuya` but with the UYVA channel order.
    if self.plan.is_some() {
      let alpha_mode = self.alpha_mode;
      let matrix = row.matrix();
      let full_range = row.full_range();
      let packed = row.packed();
      let Self {
        rgb,
        rgba,
        rgb_u16,
        rgba_u16,
        luma,
        luma_u16,
        hsv,
        rgba_scratch,
        rgb_scratch,
        rgba_scratch_u16,
        rgba_color_scratch_u16,
        luma_scratch_u16,
        plan,
        rgba_stream,
        rgba_stream_u16,
        luma_stream_u16,
        rgba_filter_stream,
        rgba_filter_stream_u16,
        luma_filter_stream_u16,
        resample_outputs,
        frozen_alpha_mode,
        ..
      } = self;
      let plan = plan.as_ref().expect("plan.is_some() checked above");
      check_frozen_alpha_mode(*frozen_alpha_mode, alpha_mode, idx)?;
      return match plan.kind() {
        crate::resample::SpanKind::Area => packed_yuva444_resample::<BITS>(
          rgba_stream,
          rgba_stream_u16,
          luma_stream_u16,
          resample_outputs,
          rgb,
          rgba,
          rgb_u16,
          rgba_u16,
          luma,
          luma_u16,
          hsv,
          rgba_scratch,
          rgb_scratch,
          rgba_scratch_u16,
          rgba_color_scratch_u16,
          luma_scratch_u16,
          w,
          plan,
          idx,
          use_simd,
          alpha_mode,
          |dst| uyva_to_rgba_row(packed, dst, w, matrix, full_range, use_simd),
          // `Uyva` has no u16 colour outputs, so this closure is never called.
          |_dst: &mut [u16]| {},
          |dst| uyva_to_luma_u16_row(packed, dst, w, use_simd),
        ),
        crate::resample::SpanKind::Filter if alpha_mode.is_premultiplied() => {
          // Premultiplied + filter has no analogue: route to the area tail
          // with the filter plan so it returns the typed `UnsupportedFilter`.
          packed_yuva444_resample::<BITS>(
            rgba_stream,
            rgba_stream_u16,
            luma_stream_u16,
            resample_outputs,
            rgb,
            rgba,
            rgb_u16,
            rgba_u16,
            luma,
            luma_u16,
            hsv,
            rgba_scratch,
            rgb_scratch,
            rgba_scratch_u16,
            rgba_color_scratch_u16,
            luma_scratch_u16,
            w,
            plan,
            idx,
            use_simd,
            alpha_mode,
            |dst| uyva_to_rgba_row(packed, dst, w, matrix, full_range, use_simd),
            |_dst: &mut [u16]| {},
            |dst| uyva_to_luma_u16_row(packed, dst, w, use_simd),
          )
        }
        crate::resample::SpanKind::Filter => packed_yuva444_filter_resample::<BITS, false, false>(
          rgba_filter_stream,
          rgba_filter_stream_u16,
          // Packed `Uyva` never uses the u8 native-Y luma stream
          // (`NATIVE_LUMA_U8 = false`); pass an inert slot.
          &mut None,
          luma_filter_stream_u16,
          resample_outputs,
          rgb,
          rgba,
          rgb_u16,
          rgba_u16,
          luma,
          luma_u16,
          hsv,
          rgba_scratch,
          rgb_scratch,
          rgba_scratch_u16,
          rgba_color_scratch_u16,
          luma_scratch_u16,
          w,
          plan,
          idx,
          use_simd,
          // Packed `Uyva` routes luma through `deinterleave_y` + the u16
          // stream (no contiguous native-Y plane), so the u8-luma input and
          // its de-interleave scratch are unused.
          &[],
          None,
          |dst| uyva_to_rgba_row(packed, dst, w, matrix, full_range, use_simd),
          // `Uyva` has no u16 colour outputs, so this closure is never called.
          |_dst: &mut [u16]| {},
          |dst| uyva_to_luma_u16_row(packed, dst, w, use_simd),
          // u16-luma path, so this u8 de-interleave is never called.
          |_dst: &mut [u8]| {},
        ),
      };
    }

    let Self {
      rgb,
      rgba,
      luma,
      luma_u16,
      hsv,
      rgb_scratch,
      ..
    } = self;
    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;
    let packed = row.packed();

    // Output mode resolution — hoisted ABOVE the luma writes so the atomicity
    // preflight runs ahead of any output mutation.
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();

    // Atomicity preflight (#308, cf. the crate's #180 resample fix and the
    // merged packed 4:2:2 / planar / semi-planar siblings): reserve the only
    // fallible row scratch this identity row can grow BEFORE any output row
    // (luma / luma_u16 included) is written, so an allocator refusal returns a
    // typed `AllocationFailed` leaving the output frame untouched rather than
    // partially mutated. The sole growable scratch is the RGB row buffer, taken
    // exactly when the colour decode needs an RGB row but no caller RGB buffer
    // is borrowable — `want_hsv && want_rgba && !want_rgb` (the
    // `rgb_row_buf_or_scratch` scratch arm; an attached RGB buffer is borrowed
    // and never allocates, and standalone `with_rgba` uses the dedicated
    // `uyva_to_rgba_row` kernel). The later `need_rgb_kernel` decode reuses the
    // already-sized buffer.
    if want_hsv && want_rgba && !want_rgb {
      rgb_row_buf_or_scratch(
        rgb.as_deref_mut(),
        rgb_scratch,
        one_plane_start,
        one_plane_end,
        w,
        h,
      )?;
    }

    // Luma u8 — extract Y byte (offset 1 in each UYVA quadruple) directly.
    if let Some(buf) = luma.as_deref_mut() {
      uyva_to_luma_row(
        packed,
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    // Luma u16 — extract Y bytes and zero-extend to u16.
    if let Some(buf) = luma_u16.as_deref_mut() {
      uyva_to_luma_u16_row(
        packed,
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    // ===== u8 RGB / RGBA / HSV path =====
    let want_hsv_direct = want_hsv && !want_rgb && !want_rgba;
    let need_rgb_kernel = want_rgb || (want_hsv && want_rgba);

    if want_hsv_direct {
      let hsv = hsv.as_mut().expect("want_hsv_direct implies hsv attached");
      let (h, s, v) = hsv.hsv();
      uyva_to_hsv_row(
        packed,
        &mut h[one_plane_start..one_plane_end],
        &mut s[one_plane_start..one_plane_end],
        &mut v[one_plane_start..one_plane_end],
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
      return Ok(());
    }

    // RGB kernel — write into the user's RGB buffer (if attached) or the
    // internal scratch buffer. Required when with_rgb or with_hsv is set.
    if need_rgb_kernel {
      let rgb_row = rgb_row_buf_or_scratch(
        rgb.as_deref_mut(),
        rgb_scratch,
        one_plane_start,
        one_plane_end,
        w,
        h,
      )?;
      uyva_to_rgb_row(packed, rgb_row, w, row.matrix(), row.full_range(), use_simd);

      if let Some(hsv) = hsv.as_mut() {
        let (h, s, v) = hsv.hsv();
        rgb_to_hsv_row(
          rgb_row,
          &mut h[one_plane_start..one_plane_end],
          &mut s[one_plane_start..one_plane_end],
          &mut v[one_plane_start..one_plane_end],
          w,
          use_simd,
        );
      }

      // Strategy A+ combo: RGBA both attached — UYVA carries source α at
      // offset 3 (same slot as `Vuya`'s α), so derive RGBA from the
      // just-computed RGB row (writes α=0xFF) then overwrite the α slot from
      // packed source byte 3 via the shared offset-3 α-extract helper.
      // Output is byte-identical to `uyva_to_rgba_row` directly (spec § 3.2).
      if want_rgba {
        let rgba_buf = rgba.as_deref_mut().unwrap();
        let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
        expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
        crate::row::alpha_extract::copy_alpha_packed_u8x4_at_3(packed, rgba_row, w, use_simd);
      }
    }

    // Standalone RGBA path — no RGB/HSV requested. Run uyva_to_rgba_row
    // directly from the packed source; source α passes through in the kernel.
    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      uyva_to_rgba_row(
        packed,
        rgba_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
    }

    Ok(())
  }
}
