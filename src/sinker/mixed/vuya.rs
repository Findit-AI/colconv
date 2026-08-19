//! Sinker impl for the packed VUYA source format — Ship 12c (Tier 5
//! 8-bit packed YUV 4:4:4 with real source alpha).
//!
//! VUYA (FFmpeg `AV_PIX_FMT_VUYA`) packs **four u8 bytes per pixel**
//! (`[V, U, Y, A]`). The A byte is **real source alpha** — not padding.
//! The packed slice type is `&[u8]`, with `4 x width` byte elements per
//! row. There is no chroma subsampling — every pixel carries its own
//! independent V / U / Y triplet (4:4:4).
//!
//! Outputs map to the sink's standard channels:
//! - `with_rgb` — packed YUV → RGB 8-bit pipeline; alpha discarded.
//! - `with_rgba` — packed YUV → RGBA 8-bit pipeline; **source α byte
//!   is passed through** verbatim from byte 3 of each pixel (not
//!   substituted with `0xFF`).
//! - `with_luma` — extracts the Y byte at offset 2 of each pixel
//!   directly (no YUV→RGB pipeline).
//! - `with_luma_u16` — zero-extends the Y byte to u16
//!   (`out[x] = Y_byte as u16`).
//! - `with_hsv` — stages u8 RGB into the user's RGB buffer (if
//!   attached) or a scratch buffer, then runs `rgb_to_hsv_row`.
//!
//! ## Alpha semantics (`§ 7.2` / `§ 7.3` rules)
//!
//! - **Standalone RGBA** (`with_rgba` attached, no `with_rgb`, no
//!   `with_hsv`): `vuya_to_rgba_row` runs directly — source α passes
//!   through via the kernel.
//! - **RGB + RGBA** (both attached, with or without HSV): Strategy A+
//!   combo — `with_rgb` calls `vuya_to_rgb_row` (chroma kernel runs
//!   ONCE); `with_rgba` is derived by `expand_rgb_to_rgba_row` (writes
//!   α=`0xFF`) followed by
//!   `alpha_extract::copy_alpha_packed_u8x4_at_3` to overwrite
//!   the α slot from the packed source (slot 3). Output is byte-identical
//!   to calling `vuya_to_rgba_row` directly (spec § 3.2 / § 7.2).

use super::{
  GeometryOverflow, InsufficientBuffer, MixedSinker, MixedSinkerError, RowIndexOutOfRange,
  RowShapeMismatch, RowSlice, check_dimensions_match, check_frozen_alpha_mode,
  packed_yuva444_filter_resample, packed_yuva444_resample, rgb_row_buf_or_scratch,
  rgba_plane_row_slice,
};
use crate::{
  PixelSink,
  row::{
    expand_rgb_to_rgba_row, rgb_to_hsv_row, vuya_to_hsv_row, vuya_to_luma_row,
    vuya_to_luma_u16_row, vuya_to_rgb_row, vuya_to_rgba_row,
  },
  source::{Vuya, VuyaRow, VuyaSink},
};

impl<'a, R> MixedSinker<'a, Vuya, R> {
  /// Attaches a **`u16`** luma output buffer. Y bytes from the packed VUYA
  /// `[V, U, Y, A]` layout are zero-extended to u16
  /// (`out[x] = Y_byte as u16`). Length in u16 **elements**
  /// (`width x height`).
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

  /// Attaches a packed **8-bit** RGBA output buffer. When VUYA is the
  /// source, the per-pixel alpha byte is **sourced from the A byte of
  /// each pixel quadruple** — not forced to `0xFF`.
  ///
  /// Returns `Err(InsufficientRgbaBuffer)` if
  /// `buf.len() < width x height x 4`, or `Err(GeometryOverflow)` on
  /// 32‑bit targets when the product overflows.
  ///
  /// ## Strategy note
  ///
  /// Source-α pass-through is guaranteed in **all** paths (standalone or
  /// combined with `with_rgb` / `with_hsv`). When standalone (no
  /// `with_rgb` / `with_hsv`), `vuya_to_rgba_row` runs directly from the
  /// packed source. When combined with `with_rgb`, Strategy A+ applies:
  /// `expand_rgb_to_rgba_row` fans out the RGB row (α=`0xFF`) and
  /// `alpha_extract::copy_alpha_packed_u8x4_at_3` overwrites
  /// the α slot — output is byte-identical to the standalone path (spec
  /// § 3.2 / § 7.2).
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

impl<R> VuyaSink for MixedSinker<'_, Vuya, R> {}

impl<R> PixelSink for MixedSinker<'_, Vuya, R> {
  type Input<'r> = VuyaRow<'r>;
  type Error = MixedSinkerError;

  #[cfg_attr(not(tarpaulin), inline(always))]
  fn kernel_matrix(&self) -> crate::KernelMatrix {
    self.kernel_matrix
  }
  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    // New frame: restart the 4-channel u8 RGBA colour stream and the
    // independent native-Y u16 luma stream (both lazily created in
    // `process`, area or filter kind) and re-arm the alpha-mode snapshot,
    // mirroring the alpha-aware packed-RGBA / `Ya` sinks. `Vuya` exposes no
    // u16 colour outputs, so its u16 RGBA streams are never created.
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

  fn process(&mut self, row: VuyaRow<'_>) -> Result<(), Self::Error> {
    const BITS: u32 = 8;
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    // VUYA row = `width x 4` bytes (one quadruple per pixel).
    let packed_expected =
      w.checked_mul(4)
        .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
          w, h, 4,
        )))?;
    if row.packed().len() != packed_expected {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::VuyaPacked,
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

    // Non-identity plan: `Vuya` is packed 4:4:4 YUV **with real source
    // alpha** (the A byte of each `[V, U, Y, A]` quadruple). Route through
    // the packed-YUVA tail at `SRC_BITS = 8`: the u8 colour stream resamples
    // the converted u8 RGBA row (`vuya_to_rgba_row` — real source α, NOT
    // forced opaque) so resampled alpha is a real mean; the native-Y luma
    // stream resamples the Y bytes (`vuya_to_luma_u16_row` — zero-extended
    // Y) so luma / luma_u16 are the downscaled native Y, alpha- and
    // range-independent (never derived from the colour). `Vuya` exposes no
    // u16 colour outputs, so the tail's u16 colour resampling is never
    // active (`rgb_u16` / `rgba_u16` stay `None`) and its `convert_rgba_u16`
    // closure is never invoked.
    //
    // The span kind picks the engine: `Area` bins (the alpha-aware tail —
    // premultiplied colour is binned premultiplied then un-premultiplied);
    // `Filter` runs the signed-coefficient filter on the same converted
    // RGBA (PIL RGBA semantics — all four channels filtered independently,
    // straight alpha only). Premultiplied alpha has no filter analogue (the
    // engine cannot un-premultiply), so a premultiplied `Filter` plan is
    // routed to the area tail, which surfaces the typed `UnsupportedFilter`
    // rather than emitting straight-filtered premultiplied colour.
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
          |dst| vuya_to_rgba_row(packed, dst, w, matrix, full_range, use_simd),
          // `Vuya` has no u16 colour outputs, so this closure is never called.
          |_dst: &mut [u16]| {},
          |dst| vuya_to_luma_u16_row(packed, dst, w, use_simd),
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
            |dst| vuya_to_rgba_row(packed, dst, w, matrix, full_range, use_simd),
            |_dst: &mut [u16]| {},
            |dst| vuya_to_luma_u16_row(packed, dst, w, use_simd),
          )
        }
        crate::resample::SpanKind::Filter => packed_yuva444_filter_resample::<BITS, false, false>(
          rgba_filter_stream,
          rgba_filter_stream_u16,
          // Packed `Vuya` never uses the u8 native-Y luma stream
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
          // Packed `Vuya` routes luma through `deinterleave_y` + the u16
          // stream (no contiguous native-Y plane), so the u8-luma input and
          // its de-interleave scratch are unused.
          &[],
          None,
          |dst| vuya_to_rgba_row(packed, dst, w, matrix, full_range, use_simd),
          // `Vuya` has no u16 colour outputs, so this closure is never called.
          |_dst: &mut [u16]| {},
          |dst| vuya_to_luma_u16_row(packed, dst, w, use_simd),
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
    // `vuya_to_rgba_row` kernel). The later `need_rgb_kernel` decode reuses the
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

    // Luma u8 — extract Y byte (offset 2 in each VUYA quadruple) directly.
    if let Some(buf) = luma.as_deref_mut() {
      vuya_to_luma_row(
        packed,
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    // Luma u16 — extract Y bytes and zero-extend to u16.
    if let Some(buf) = luma_u16.as_deref_mut() {
      vuya_to_luma_u16_row(
        packed,
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    // ===== u8 RGB / RGBA / HSV path =====
    // HSV-without-RGB-or-RGBA goes through the direct `vuya_to_hsv_row`
    // kernel — no source-width RGB scratch (the SIMD path stages a fixed
    // 64-pixel 8-bit RGB chunk internally). When RGB or RGBA is also
    // attached the RGB kernel runs anyway, so HSV derives off that buffer
    // for free and `need_rgb_kernel` keeps it alive. Resample row-stage
    // HSV-only is a #263 follow-up — HSV stays correct via the
    // convert-once path.
    let want_hsv_direct = want_hsv && !want_rgb && !want_rgba;
    let need_rgb_kernel = want_rgb || (want_hsv && want_rgba);

    if want_hsv_direct {
      let hsv = hsv.as_mut().expect("want_hsv_direct implies hsv attached");
      let (h, s, v) = hsv.hsv();
      vuya_to_hsv_row(
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
      vuya_to_rgb_row(packed, rgb_row, w, row.matrix(), row.full_range(), use_simd);

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

      // Strategy A+ combo: RGBA both attached — derive from the just-computed
      // RGB row (writes α=0xFF), then overwrite α slot from packed source.
      // Output is byte-identical to vuya_to_rgba_row directly (spec § 3.2).
      // See spec docs/superpowers/specs/2026-05-04-pr4-strategy-a-plus-design.md.
      if want_rgba {
        let rgba_buf = rgba.as_deref_mut().unwrap();
        let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
        expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
        crate::row::alpha_extract::copy_alpha_packed_u8x4_at_3(packed, rgba_row, w, use_simd);
      }
    }

    // Standalone RGBA path — no RGB/HSV requested. Run vuya_to_rgba_row
    // directly from the packed source; source α passes through in the
    // kernel. This path is already optimal — unchanged (spec § 7.2).
    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      vuya_to_rgba_row(
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
