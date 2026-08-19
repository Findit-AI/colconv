//! Sinker impls for source-side YUVA 4:4:4 formats —
//! [`Yuva444p9`](crate::source::Yuva444p9),
//! [`Yuva444p10`](crate::source::Yuva444p10),
//! [`Yuva444p12`](crate::source::Yuva444p12), and
//! [`Yuva444p14`](crate::source::Yuva444p14). The 16-bit sibling
//! (`Yuva444p16`) needs its own SIMD work and lands in Ship 8b‑5.
//!
//! For each format:
//! - **RGBA output paths (u8 + native-depth u16)** — alpha is sourced
//!   from the source alpha plane via the
//!   `yuv_444p_n_to_rgba*_with_alpha_src_row::<BITS>` SIMD/scalar
//!   kernels (BITS-generic across {9, 10, 12, 14}).
//! - **RGB / RGB-u16 / luma / HSV alpha-drop paths** — these reuse the
//!   existing non-alpha 4:4:4 row dispatchers verbatim. The alpha
//!   plane is simply ignored; output bytes / elements match what the
//!   corresponding `Yuv444p<BITS>` source would produce given the
//!   same Y/U/V data. Without these paths
//!   `MixedSinker::with_rgb` / `with_luma` / `with_hsv` (declared on
//!   the generic impl) would silently accept a buffer and never write
//!   it.
//!
//! The 9 / 10 / 12 / 14-bit `process` bodies are structurally
//! identical — only the depth-named row primitives, the `RowSlice`
//! variants used in error reports, and the depth-conversion shift
//! (`BITS - 8`) for luma differ. They share the
//! [`yuva444p_high_bit_process`] helper to avoid 4x of the same ~70
//! lines.

use super::{
  GeometryOverflow, InsufficientBuffer, MixedSinker, MixedSinkerError, RowIndexOutOfRange,
  RowShapeMismatch, RowSlice, check_dimensions_match, check_frozen_alpha_mode,
  deinterleave_y_high_bit_masked, packed_yuva444_filter_resample, packed_yuva444_resample,
  reset_high_bit_yuva_streams, rgb_row_buf_or_scratch, rgba_plane_row_slice,
  rgba_u16_plane_row_slice,
};
use crate::{PixelSink, row::*, source::*};

// ---- Yuva444p impl (8-bit) ---------------------------------------------

impl<'a, R> MixedSinker<'a, Yuva444p, R> {
  /// Attaches a packed **8-bit** RGBA output buffer. The 8-bit YUVA
  /// 4:4:4 source is converted to 8-bit RGBA via the same Q15 i32
  /// 8-bit kernel that backs [`MixedSinker<Yuv444p>::with_rgba`]; the
  /// per-pixel alpha byte is **sourced from the alpha plane** — not
  /// constant `0xFF`.
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

  /// Attaches a **`u16`** luma output buffer. The 8-bit Y plane is
  /// zero-extended to u16 (`out[x] = Y_byte as u16`). Length in u16
  /// **elements** (`width x height`).
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
}

impl<R> Yuva444pSink for MixedSinker<'_, Yuva444p, R> {}

impl<R> PixelSink for MixedSinker<'_, Yuva444p, R> {
  type Input<'r> = Yuva444pRow<'r>;
  type Error = MixedSinkerError;

  #[cfg_attr(not(tarpaulin), inline(always))]
  fn kernel_matrix(&self) -> crate::KernelMatrix {
    self.kernel_matrix
  }
  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    // New frame: restart the 4-channel u8 RGBA colour stream and the
    // native-Y luma stream (both lazily created in `process`) and re-arm the
    // alpha-mode snapshot, mirroring the alpha-aware packed-YUVA (`Vuya`)
    // sink. The luma stream kind depends on the plan: the area path bins the
    // native Y at u16 (`luma_stream_u16`), the filter path resamples the
    // contiguous native Y at u8 (`luma_filter_stream`, parity with the
    // no-alpha `Yuv444p`). The 8-bit `Yuva444p` exposes no u16 colour
    // outputs, so its u16 RGBA streams are never created.
    if let Some(stream) = self.rgba_stream.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.luma_stream_u16.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.rgba_filter_stream.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.luma_filter_stream.as_mut() {
      stream.reset();
    }
    self.resample_outputs = None;
    self.frozen_alpha_mode = Some(self.alpha_mode);
    Ok(())
  }

  fn process(&mut self, row: Yuva444pRow<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Y,
        idx,
        w,
        row.y().len(),
      )));
    }
    if row.u().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::UFull,
        idx,
        w,
        row.u().len(),
      )));
    }
    if row.v().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::VFull,
        idx,
        w,
        row.v().len(),
      )));
    }
    if row.a().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::AFull,
        idx,
        w,
        row.a().len(),
      )));
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange(
        RowIndexOutOfRange::new(idx, self.height),
      ));
    }

    // Non-identity plan: `Yuva444p` is 8-bit planar 4:4:4 YUV **with a
    // real full-resolution source alpha plane** (no chroma subsampling —
    // every pixel carries its own U / V). Route through the packed-YUVA
    // tail at `SRC_BITS = 8`: the u8 colour stream resamples the converted
    // u8 RGBA row (`yuva444p_to_rgba_row` — full-width chroma, real source
    // α, NOT forced opaque), the native-Y luma stream resamples the
    // (zero-extended) Y plane directly so luma / luma_u16 are the
    // downscaled native Y, alpha- and range-independent. The 8-bit
    // `Yuva444p` exposes no u16 colour outputs, so the tail's u16 colour
    // resampling is never active and its `convert_rgba_u16` closure is never
    // invoked.
    //
    // The span kind picks the engine (mirrors `Yuva420p`): `Area` bins (the
    // alpha-aware tail — premultiplied colour binned premultiplied then
    // un-premultiplied); `Filter` runs the signed-coefficient filter on the
    // same converted RGBA (straight alpha only). A premultiplied `Filter`
    // plan is routed to the area tail so it surfaces the typed
    // `UnsupportedFilter` rather than straight-filtering premultiplied colour.
    if self.plan.is_some() {
      let alpha_mode = self.alpha_mode;
      let matrix = row.matrix();
      let full_range = row.full_range();
      let y = row.y();
      let u = row.u();
      let v = row.v();
      let a = row.a();
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
        luma_filter_stream,
        luma_filter_stream_u16,
        resample_outputs,
        frozen_alpha_mode,
        ..
      } = self;
      let plan = plan.as_ref().expect("plan.is_some() checked above");
      check_frozen_alpha_mode(*frozen_alpha_mode, alpha_mode, idx)?;
      return match plan.kind() {
        crate::resample::SpanKind::Area => packed_yuva444_resample::<8>(
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
          |dst| yuva444p_to_rgba_row(y, u, v, a, dst, w, matrix, full_range, use_simd),
          // `Yuva444p` has no u16 colour outputs, so this closure is never called.
          |_dst: &mut [u16]| {},
          |dst| {
            for (d, &s) in dst.iter_mut().zip(y) {
              *d = s as u16;
            }
          },
        ),
        crate::resample::SpanKind::Filter if alpha_mode.is_premultiplied() => {
          // Premultiplied + filter has no analogue: route to the area tail
          // with the filter plan so it returns the typed `UnsupportedFilter`.
          packed_yuva444_resample::<8>(
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
            |dst| yuva444p_to_rgba_row(y, u, v, a, dst, w, matrix, full_range, use_simd),
            |_dst: &mut [u16]| {},
            |dst| {
              for (d, &s) in dst.iter_mut().zip(y) {
                *d = s as u16;
              }
            },
          )
        }
        crate::resample::SpanKind::Filter => packed_yuva444_filter_resample::<8, true, false>(
          rgba_filter_stream,
          rgba_filter_stream_u16,
          luma_filter_stream,
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
          // 8-bit native-Y luma rides the u8 stream (parity with `Yuv444p`):
          // the contiguous Y plane is fed directly, so no de-interleave scratch.
          y,
          None,
          |dst| yuva444p_to_rgba_row(y, u, v, a, dst, w, matrix, full_range, use_simd),
          // `Yuva444p` has no u16 colour outputs, so this closure is never called.
          |_dst: &mut [u16]| {},
          // u8-luma path: the u16 luma stream is detached, so this is never
          // called.
          |_dst: &mut [u16]| {},
          // Contiguous Y plane fed directly, so this u8 de-interleave is unused.
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

    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();
    // HSV-without-RGB-or-RGBA goes through the direct `yuv_444_to_hsv_row`
    // kernel (no source-width RGB scratch) — the same 4:4:4 kernel the RGB
    // alpha-drop path uses. HSV is colour-only — the source alpha plane is
    // dropped. RGB or RGBA also attached keeps the
    // convert-once-then-derive path alive via `need_rgb_kernel` (an RGBA
    // sink still needs the alpha plane, so it stays on the RGB path).
    let want_hsv_direct = want_hsv && !want_rgb && !want_rgba;
    let need_rgb_kernel = want_rgb || (want_hsv && want_rgba);

    // Acquire the u8 RGB row buffer up front — before any caller-output
    // write below — so an allocator refusal in the HSV-only scratch path
    // returns a recoverable error rather than leaving a partially-written
    // caller buffer (luma / luma_u16). `rgb_row_buf_or_scratch` is the
    // only fallible allocation on this direct path.
    let rgb_row = if need_rgb_kernel {
      Some(rgb_row_buf_or_scratch(
        rgb.as_deref_mut(),
        rgb_scratch,
        one_plane_start,
        one_plane_end,
        w,
        h,
      )?)
    } else {
      None
    };

    // Luma — Y plane is already u8.
    if let Some(luma) = luma.as_deref_mut() {
      luma[one_plane_start..one_plane_end].copy_from_slice(&row.y()[..w]);
    }

    // Luma u16 — zero-extend the Y plane (`out[x] = Y_byte as u16`).
    if let Some(luma_u16) = luma_u16.as_deref_mut() {
      for (d, &s) in luma_u16[one_plane_start..one_plane_end]
        .iter_mut()
        .zip(&row.y()[..w])
      {
        *d = s as u16;
      }
    }

    // HSV-only (no RGB / RGBA): convert the source Y/U/V straight to HSV
    // via the alpha-drop 4:4:4 planar kernel — no source-width RGB scratch.
    if want_hsv_direct {
      let hsv = hsv.as_mut().expect("want_hsv_direct implies hsv attached");
      let (h, s, v) = hsv.hsv();
      yuv_444_to_hsv_row(
        row.y(),
        row.u(),
        row.v(),
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

    // `need_rgb_kernel` / `rgb_row` were computed and (when needed)
    // allocated at the top, before any caller-output write.
    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      yuva444p_to_rgba_row(
        row.y(),
        row.u(),
        row.v(),
        row.a(),
        rgba_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
      return Ok(());
    }

    // RGB kernel — alpha-drop reuses the existing 4:4:4 row dispatcher.
    let Some(rgb_row) = rgb_row else {
      return Ok(());
    };
    yuv_444_to_rgb_row(
      row.y(),
      row.u(),
      row.v(),
      rgb_row,
      w,
      row.matrix(),
      row.full_range(),
      use_simd,
    );

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

    if want_rgba {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      // Strategy A+: RGB output already computed in rgb_row above.
      // Expand RGB → RGBA (fills α with 0xFF), then overwrite α slot
      // from the source alpha plane — avoids a second chroma kernel.
      expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
      crate::row::alpha_extract::copy_alpha_plane_u8(row.a(), rgba_row, w, use_simd);
    }

    Ok(())
  }
}

// ---- Yuva444p9 impl ---------------------------------------------------

impl<'a, R, const BE: bool> MixedSinker<'a, Yuva444p9<BE>, R> {
  /// Attaches a packed **8-bit** RGBA output buffer. The 9-bit YUVA
  /// source is converted to 8-bit RGBA via the same `BITS = 9` Q15
  /// kernel family used by [`MixedSinker<Yuv444p9>::with_rgba`]; the
  /// per-pixel alpha byte is **sourced from the alpha plane**
  /// (depth-converted via `a >> 1` to fit `u8`) — not constant `0xFF`.
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

  /// Attaches a packed **`u16`** RGBA output buffer. 9-bit
  /// low-packed (`[0, 511]`); the per-pixel alpha element is
  /// **sourced from the alpha plane** at native depth.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgba_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_rgba_u16(buf)?;
    Ok(self)
  }
  /// In-place variant of [`with_rgba_u16`](Self::with_rgba_u16).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgba_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_elems(4)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::InsufficientRgbaU16Buffer(
        InsufficientBuffer::new(expected, buf.len()),
      ));
    }
    self.rgba_u16 = Some(buf);
    Ok(self)
  }

  /// Attaches a packed **`u16`** RGB output buffer (alpha-drop).
  /// Output is identical to [`MixedSinker<Yuv444p9>::with_rgb_u16`] —
  /// the alpha plane is ignored.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgb_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_rgb_u16(buf)?;
    Ok(self)
  }
  /// In-place variant of [`with_rgb_u16`](Self::with_rgb_u16).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgb_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_elems(3)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::InsufficientRgbU16Buffer(
        InsufficientBuffer::new(expected, buf.len()),
      ));
    }
    self.rgb_u16 = Some(buf);
    Ok(self)
  }

  /// Attaches a **`u16`** luma output buffer. Luma is the **native Y**
  /// (the binned Y plane at native depth under a non-identity plan; the
  /// Y plane verbatim otherwise). Length in u16 **elements**
  /// (`width x height`).
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
}

impl<R, const BE: bool> Yuva444p9Sink<BE> for MixedSinker<'_, Yuva444p9<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, Yuva444p9<BE>, R> {
  type Input<'r> = Yuva444p9Row<'r>;
  type Error = MixedSinkerError;

  #[cfg_attr(not(tarpaulin), inline(always))]
  fn kernel_matrix(&self) -> crate::KernelMatrix {
    self.kernel_matrix
  }
  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    reset_high_bit_yuva_streams(self);
    Ok(())
  }

  fn process(&mut self, row: Yuva444p9Row<'_>) -> Result<(), Self::Error> {
    if self.plan.is_some() {
      return yuva444p_high_bit_resample::<9, BE>(
        self,
        row.row(),
        row.y(),
        row.u(),
        row.v(),
        row.a(),
        row.matrix(),
        row.full_range(),
        RowSlice::Y9,
        RowSlice::UFull9,
        RowSlice::VFull9,
        RowSlice::AFull9,
        yuva444p9_to_rgba_row_endian,
        yuva444p9_to_rgba_u16_row_endian,
      );
    }
    yuva444p_high_bit_process::<9, BE, _, _, _, _, _, _>(
      self,
      row.row(),
      row.y(),
      row.u(),
      row.v(),
      row.a(),
      row.matrix(),
      row.full_range(),
      RowSlice::Y9,
      RowSlice::UFull9,
      RowSlice::VFull9,
      RowSlice::AFull9,
      yuv444p9_to_rgb_row_endian,
      yuv444p9_to_rgb_u16_row_endian,
      yuva444p9_to_rgba_row_endian,
      yuv444p9_to_hsv_row_endian,
      yuva444p9_to_rgba_u16_row_endian,
    )
  }
}

// ---- Yuva444p10 impl --------------------------------------------------

impl<'a, R, const BE: bool> MixedSinker<'a, Yuva444p10<BE>, R> {
  /// Attaches a packed **8-bit** RGBA output buffer. The 10-bit YUVA
  /// source is converted to 8-bit RGBA via the same `BITS = 10` Q15
  /// kernel family used by [`MixedSinker<Yuv444p10>::with_rgba`]; the
  /// per-pixel alpha byte is **sourced from the alpha plane**
  /// (depth-converted via `a >> 2` to fit `u8`) — not constant `0xFF`.
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

  /// Attaches a packed **`u16`** RGBA output buffer. 10-bit
  /// low-packed (`[0, 1023]`); the per-pixel alpha element is
  /// **sourced from the alpha plane** at native depth.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgba_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_rgba_u16(buf)?;
    Ok(self)
  }
  /// In-place variant of [`with_rgba_u16`](Self::with_rgba_u16).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgba_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_elems(4)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::InsufficientRgbaU16Buffer(
        InsufficientBuffer::new(expected, buf.len()),
      ));
    }
    self.rgba_u16 = Some(buf);
    Ok(self)
  }

  /// Attaches a packed **`u16`** RGB output buffer (alpha-drop).
  /// Output is identical to [`MixedSinker<Yuv444p10>::with_rgb_u16`] —
  /// the alpha plane is ignored.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgb_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_rgb_u16(buf)?;
    Ok(self)
  }
  /// In-place variant of [`with_rgb_u16`](Self::with_rgb_u16).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgb_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_elems(3)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::InsufficientRgbU16Buffer(
        InsufficientBuffer::new(expected, buf.len()),
      ));
    }
    self.rgb_u16 = Some(buf);
    Ok(self)
  }

  /// Attaches a **`u16`** luma output buffer. Luma is the **native Y**
  /// (the binned Y plane at native depth under a non-identity plan; the
  /// Y plane verbatim otherwise). Length in u16 **elements**
  /// (`width x height`).
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
}

impl<R, const BE: bool> Yuva444p10Sink<BE> for MixedSinker<'_, Yuva444p10<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, Yuva444p10<BE>, R> {
  type Input<'r> = Yuva444p10Row<'r>;
  type Error = MixedSinkerError;

  #[cfg_attr(not(tarpaulin), inline(always))]
  fn kernel_matrix(&self) -> crate::KernelMatrix {
    self.kernel_matrix
  }
  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    reset_high_bit_yuva_streams(self);
    Ok(())
  }

  fn process(&mut self, row: Yuva444p10Row<'_>) -> Result<(), Self::Error> {
    if self.plan.is_some() {
      return yuva444p_high_bit_resample::<10, BE>(
        self,
        row.row(),
        row.y(),
        row.u(),
        row.v(),
        row.a(),
        row.matrix(),
        row.full_range(),
        RowSlice::Y10,
        RowSlice::UFull10,
        RowSlice::VFull10,
        RowSlice::AFull10,
        yuva444p10_to_rgba_row_endian,
        yuva444p10_to_rgba_u16_row_endian,
      );
    }
    yuva444p_high_bit_process::<10, BE, _, _, _, _, _, _>(
      self,
      row.row(),
      row.y(),
      row.u(),
      row.v(),
      row.a(),
      row.matrix(),
      row.full_range(),
      RowSlice::Y10,
      RowSlice::UFull10,
      RowSlice::VFull10,
      RowSlice::AFull10,
      yuv444p10_to_rgb_row_endian,
      yuv444p10_to_rgb_u16_row_endian,
      yuva444p10_to_rgba_row_endian,
      yuv444p10_to_hsv_row_endian,
      yuva444p10_to_rgba_u16_row_endian,
    )
  }
}

// ---- Yuva444p12 impl --------------------------------------------------

impl<'a, R, const BE: bool> MixedSinker<'a, Yuva444p12<BE>, R> {
  /// Attaches a packed **8-bit** RGBA output buffer. The 12-bit YUVA
  /// source is converted to 8-bit RGBA via the same `BITS = 12` Q15
  /// kernel family used by [`MixedSinker<Yuv444p12>::with_rgba`]; the
  /// per-pixel alpha byte is **sourced from the alpha plane**
  /// (depth-converted via `a >> 4` to fit `u8`) — not constant `0xFF`.
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

  /// Attaches a packed **`u16`** RGBA output buffer. 12-bit
  /// low-packed (`[0, 4095]`); the per-pixel alpha element is
  /// **sourced from the alpha plane** at native depth.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgba_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_rgba_u16(buf)?;
    Ok(self)
  }
  /// In-place variant of [`with_rgba_u16`](Self::with_rgba_u16).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgba_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_elems(4)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::InsufficientRgbaU16Buffer(
        InsufficientBuffer::new(expected, buf.len()),
      ));
    }
    self.rgba_u16 = Some(buf);
    Ok(self)
  }

  /// Attaches a packed **`u16`** RGB output buffer (alpha-drop).
  /// Output is identical to [`MixedSinker<Yuv444p12>::with_rgb_u16`] —
  /// the alpha plane is ignored.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgb_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_rgb_u16(buf)?;
    Ok(self)
  }
  /// In-place variant of [`with_rgb_u16`](Self::with_rgb_u16).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgb_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_elems(3)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::InsufficientRgbU16Buffer(
        InsufficientBuffer::new(expected, buf.len()),
      ));
    }
    self.rgb_u16 = Some(buf);
    Ok(self)
  }

  /// Attaches a **`u16`** luma output buffer. Luma is the **native Y**
  /// (the binned Y plane at native depth under a non-identity plan; the
  /// Y plane verbatim otherwise). Length in u16 **elements**
  /// (`width x height`).
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
}

impl<R, const BE: bool> Yuva444p12Sink<BE> for MixedSinker<'_, Yuva444p12<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, Yuva444p12<BE>, R> {
  type Input<'r> = Yuva444p12Row<'r>;
  type Error = MixedSinkerError;

  #[cfg_attr(not(tarpaulin), inline(always))]
  fn kernel_matrix(&self) -> crate::KernelMatrix {
    self.kernel_matrix
  }
  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    reset_high_bit_yuva_streams(self);
    Ok(())
  }

  fn process(&mut self, row: Yuva444p12Row<'_>) -> Result<(), Self::Error> {
    if self.plan.is_some() {
      return yuva444p_high_bit_resample::<12, BE>(
        self,
        row.row(),
        row.y(),
        row.u(),
        row.v(),
        row.a(),
        row.matrix(),
        row.full_range(),
        RowSlice::Y12,
        RowSlice::UFull12,
        RowSlice::VFull12,
        RowSlice::AFull12,
        yuva444p12_to_rgba_row_endian,
        yuva444p12_to_rgba_u16_row_endian,
      );
    }
    yuva444p_high_bit_process::<12, BE, _, _, _, _, _, _>(
      self,
      row.row(),
      row.y(),
      row.u(),
      row.v(),
      row.a(),
      row.matrix(),
      row.full_range(),
      RowSlice::Y12,
      RowSlice::UFull12,
      RowSlice::VFull12,
      RowSlice::AFull12,
      yuv444p12_to_rgb_row_endian,
      yuv444p12_to_rgb_u16_row_endian,
      yuva444p12_to_rgba_row_endian,
      yuv444p12_to_hsv_row_endian,
      yuva444p12_to_rgba_u16_row_endian,
    )
  }
}

// ---- Yuva444p14 impl --------------------------------------------------

impl<'a, R, const BE: bool> MixedSinker<'a, Yuva444p14<BE>, R> {
  /// Attaches a packed **8-bit** RGBA output buffer. The 14-bit YUVA
  /// source is converted to 8-bit RGBA via the same `BITS = 14` Q15
  /// kernel family used by [`MixedSinker<Yuv444p14>::with_rgba`]; the
  /// per-pixel alpha byte is **sourced from the alpha plane**
  /// (depth-converted via `a >> 6` to fit `u8`) — not constant `0xFF`.
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

  /// Attaches a packed **`u16`** RGBA output buffer. 14-bit
  /// low-packed (`[0, 16383]`); the per-pixel alpha element is
  /// **sourced from the alpha plane** at native depth.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgba_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_rgba_u16(buf)?;
    Ok(self)
  }
  /// In-place variant of [`with_rgba_u16`](Self::with_rgba_u16).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgba_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_elems(4)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::InsufficientRgbaU16Buffer(
        InsufficientBuffer::new(expected, buf.len()),
      ));
    }
    self.rgba_u16 = Some(buf);
    Ok(self)
  }

  /// Attaches a packed **`u16`** RGB output buffer (alpha-drop).
  /// Output is identical to [`MixedSinker<Yuv444p14>::with_rgb_u16`] —
  /// the alpha plane is ignored.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgb_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_rgb_u16(buf)?;
    Ok(self)
  }
  /// In-place variant of [`with_rgb_u16`](Self::with_rgb_u16).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgb_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_elems(3)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::InsufficientRgbU16Buffer(
        InsufficientBuffer::new(expected, buf.len()),
      ));
    }
    self.rgb_u16 = Some(buf);
    Ok(self)
  }

  /// Attaches a **`u16`** luma output buffer. Luma is the **native Y**
  /// (the binned Y plane at native depth under a non-identity plan; the
  /// Y plane verbatim otherwise). Length in u16 **elements**
  /// (`width x height`).
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
}

impl<R, const BE: bool> Yuva444p14Sink<BE> for MixedSinker<'_, Yuva444p14<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, Yuva444p14<BE>, R> {
  type Input<'r> = Yuva444p14Row<'r>;
  type Error = MixedSinkerError;

  #[cfg_attr(not(tarpaulin), inline(always))]
  fn kernel_matrix(&self) -> crate::KernelMatrix {
    self.kernel_matrix
  }
  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    reset_high_bit_yuva_streams(self);
    Ok(())
  }

  fn process(&mut self, row: Yuva444p14Row<'_>) -> Result<(), Self::Error> {
    if self.plan.is_some() {
      return yuva444p_high_bit_resample::<14, BE>(
        self,
        row.row(),
        row.y(),
        row.u(),
        row.v(),
        row.a(),
        row.matrix(),
        row.full_range(),
        RowSlice::Y14,
        RowSlice::UFull14,
        RowSlice::VFull14,
        RowSlice::AFull14,
        yuva444p14_to_rgba_row_endian,
        yuva444p14_to_rgba_u16_row_endian,
      );
    }
    yuva444p_high_bit_process::<14, BE, _, _, _, _, _, _>(
      self,
      row.row(),
      row.y(),
      row.u(),
      row.v(),
      row.a(),
      row.matrix(),
      row.full_range(),
      RowSlice::Y14,
      RowSlice::UFull14,
      RowSlice::VFull14,
      RowSlice::AFull14,
      yuv444p14_to_rgb_row_endian,
      yuv444p14_to_rgb_u16_row_endian,
      yuva444p14_to_rgba_row_endian,
      yuv444p14_to_hsv_row_endian,
      yuva444p14_to_rgba_u16_row_endian,
    )
  }
}

// ---- Yuva444p16 impl --------------------------------------------------

impl<'a, R, const BE: bool> MixedSinker<'a, Yuva444p16<BE>, R> {
  /// Attaches a packed **8-bit** RGBA output buffer. The 16-bit YUVA
  /// source is converted to 8-bit RGBA via the same `BITS = 16` Q15
  /// kernel family used by [`MixedSinker<Yuv444p16>::with_rgba`]; the
  /// per-pixel alpha byte is **sourced from the alpha plane**
  /// (depth-converted via `a >> 8` to fit `u8`) — not constant `0xFF`.
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

  /// Attaches a packed **`u16`** RGBA output buffer. 16-bit
  /// low-packed (`[0, 65535]`); the per-pixel alpha element is
  /// **sourced from the alpha plane** at native depth.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgba_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_rgba_u16(buf)?;
    Ok(self)
  }
  /// In-place variant of [`with_rgba_u16`](Self::with_rgba_u16).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgba_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_elems(4)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::InsufficientRgbaU16Buffer(
        InsufficientBuffer::new(expected, buf.len()),
      ));
    }
    self.rgba_u16 = Some(buf);
    Ok(self)
  }

  /// Attaches a packed **`u16`** RGB output buffer (alpha-drop).
  /// Output is identical to [`MixedSinker<Yuv444p16>::with_rgb_u16`] —
  /// the alpha plane is ignored.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgb_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_rgb_u16(buf)?;
    Ok(self)
  }
  /// In-place variant of [`with_rgb_u16`](Self::with_rgb_u16).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgb_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_elems(3)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::InsufficientRgbU16Buffer(
        InsufficientBuffer::new(expected, buf.len()),
      ));
    }
    self.rgb_u16 = Some(buf);
    Ok(self)
  }

  /// Attaches a **`u16`** luma output buffer. Luma is the **native Y**
  /// (the binned Y plane at native depth under a non-identity plan; the
  /// Y plane verbatim otherwise). Length in u16 **elements**
  /// (`width x height`).
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
}

impl<R, const BE: bool> Yuva444p16Sink<BE> for MixedSinker<'_, Yuva444p16<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, Yuva444p16<BE>, R> {
  type Input<'r> = Yuva444p16Row<'r>;
  type Error = MixedSinkerError;

  #[cfg_attr(not(tarpaulin), inline(always))]
  fn kernel_matrix(&self) -> crate::KernelMatrix {
    self.kernel_matrix
  }
  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    reset_high_bit_yuva_streams(self);
    Ok(())
  }

  fn process(&mut self, row: Yuva444p16Row<'_>) -> Result<(), Self::Error> {
    if self.plan.is_some() {
      return yuva444p_high_bit_resample::<16, BE>(
        self,
        row.row(),
        row.y(),
        row.u(),
        row.v(),
        row.a(),
        row.matrix(),
        row.full_range(),
        RowSlice::Y16,
        RowSlice::UFull16,
        RowSlice::VFull16,
        RowSlice::AFull16,
        yuva444p16_to_rgba_row_endian,
        yuva444p16_to_rgba_u16_row_endian,
      );
    }
    yuva444p_high_bit_process::<16, BE, _, _, _, _, _, _>(
      self,
      row.row(),
      row.y(),
      row.u(),
      row.v(),
      row.a(),
      row.matrix(),
      row.full_range(),
      RowSlice::Y16,
      RowSlice::UFull16,
      RowSlice::VFull16,
      RowSlice::AFull16,
      yuv444p16_to_rgb_row_endian,
      yuv444p16_to_rgb_u16_row_endian,
      yuva444p16_to_rgba_row_endian,
      yuv444p16_to_hsv_row_endian,
      yuva444p16_to_rgba_u16_row_endian,
    )
  }
}

// ---- Shared high-bit YUVA 4:4:4 process body --------------------------
//
// The 9 / 10-bit YUVA 4:4:4 sinker `process` bodies are structurally
// identical — only the depth-named row primitives, the `RowSlice`
// variants used in error reports, and the depth-conversion shift
// (`BITS - 8`) for luma differ. Factor into a generic helper to avoid
// 2x of the same ~70 lines.
//
// 4:4:4 chroma is full-width (one U / V sample per Y pixel — no
// chroma duplication step), unlike the 4:2:0 sibling helper which
// takes half-width chroma rows.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
#[cfg_attr(not(tarpaulin), inline(always))]
fn yuva444p_high_bit_process<
  const BITS: u32,
  const BE: bool,
  F: crate::SourceFormat,
  R,
  RgbRowFn: Fn(&[u16], &[u16], &[u16], &mut [u8], usize, crate::KernelMatrix, bool, bool, bool),
  RgbU16RowFn: Fn(&[u16], &[u16], &[u16], &mut [u16], usize, crate::KernelMatrix, bool, bool, bool),
  RgbaRowFn: Fn(&[u16], &[u16], &[u16], &[u16], &mut [u8], usize, crate::KernelMatrix, bool, bool, bool),
  // The matching alpha-drop YUV→HSV `_endian` kernel — same Y/U/V inputs
  // as `rgb_dispatch`, three `&mut [u8]` H/S/V outputs.
  HsvRowFn: Fn(
    &[u16],
    &[u16],
    &[u16],
    &mut [u8],
    &mut [u8],
    &mut [u8],
    usize,
    crate::KernelMatrix,
    bool,
    bool,
    bool,
  ),
>(
  sinker: &mut MixedSinker<'_, F, R>,
  idx: usize,
  y_row: &[u16],
  u_row: &[u16],
  v_row: &[u16],
  a_row: &[u16],
  matrix: crate::KernelMatrix,
  full_range: bool,
  y_slice: RowSlice,
  u_slice: RowSlice,
  v_slice: RowSlice,
  a_slice: RowSlice,
  rgb_dispatch: RgbRowFn,
  rgb_u16_dispatch: RgbU16RowFn,
  rgba_dispatch: RgbaRowFn,
  hsv_dispatch: HsvRowFn,
  rgba_u16_dispatch: fn(
    &[u16],
    &[u16],
    &[u16],
    &[u16],
    &mut [u16],
    usize,
    crate::KernelMatrix,
    bool,
    bool,
    bool,
  ),
) -> Result<(), MixedSinkerError> {
  let w = sinker.width;
  let h = sinker.height;
  let use_simd = sinker.simd;

  if y_row.len() != w {
    return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
      y_slice,
      idx,
      w,
      y_row.len(),
    )));
  }
  if u_row.len() != w {
    return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
      u_slice,
      idx,
      w,
      u_row.len(),
    )));
  }
  if v_row.len() != w {
    return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
      v_slice,
      idx,
      w,
      v_row.len(),
    )));
  }
  if a_row.len() != w {
    return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
      a_slice,
      idx,
      w,
      a_row.len(),
    )));
  }
  if idx >= sinker.height {
    return Err(MixedSinkerError::RowIndexOutOfRange(
      RowIndexOutOfRange::new(idx, sinker.height),
    ));
  }

  let MixedSinker {
    rgb,
    rgb_u16,
    rgba,
    rgba_u16,
    luma,
    luma_u16,
    hsv,
    rgb_scratch,
    ..
  } = sinker;
  let one_plane_start = idx * w;
  let one_plane_end = one_plane_start + w;

  let want_rgb = rgb.is_some();
  let want_rgba = rgba.is_some();
  let want_hsv = hsv.is_some();
  // HSV-without-RGB-or-RGBA goes through the direct alpha-drop
  // `hsv_dispatch` kernel (no source-width RGB scratch). HSV is
  // colour-only — the source alpha plane is dropped. See
  // `yuva420p_high_bit_process` for the full routing rationale.
  let want_hsv_direct = want_hsv && !want_rgb && !want_rgba;
  let need_rgb_kernel = want_rgb || (want_hsv && want_rgba);

  // Acquire the u8 RGB row buffer up front — before any caller-output
  // write below — so an allocator refusal in the HSV-only scratch path
  // returns a recoverable error rather than a partially-written caller
  // buffer. See `yuva420p_high_bit_process` for the full rationale.
  let rgb_row = if need_rgb_kernel {
    Some(rgb_row_buf_or_scratch(
      rgb.as_deref_mut(),
      rgb_scratch,
      one_plane_start,
      one_plane_end,
      w,
      h,
    )?)
  } else {
    None
  };

  // ---- luma (native Y; luma narrows `>> (BITS - 8)`, luma_u16 is the
  // host-native logical Y) — see `yuva420p_high_bit_process`. -----------
  if luma.is_some() || luma_u16.is_some() {
    let mut luma_row = luma
      .as_deref_mut()
      .map(|b| &mut b[one_plane_start..one_plane_end]);
    let mut luma_u16_row = luma_u16
      .as_deref_mut()
      .map(|b| &mut b[one_plane_start..one_plane_end]);
    // Mask each decoded Y to the source's native depth `(1 << BITS) - 1` (a
    // no-op at `BITS = 16`). `Yuva444p*Frame::try_new` is geometry-only, so a
    // malformed-but-accepted frame can carry out-of-range Y (e.g. `0x1000` at
    // 12-bit); without this mask `luma_u16` would publish that raw value (and the
    // 8-bit luma shift the wrapped value), inconsistent with the
    // `(1 << BITS) - 1`-masked Y the RGB/RGBA row kernels decode from the same
    // row. Mirrors the high-bit `Yuva420p` / `Yuva422p` siblings — dirty-upper-bit
    // sanitization covers EVERY high-bit output (luma_u16 + 8-bit luma + chroma).
    let sample_mask = ((1u32 << BITS) - 1) as u16;
    for (i, &s) in y_row.iter().enumerate().take(w) {
      let logical = (if BE { u16::from_be(s) } else { u16::from_le(s) }) & sample_mask;
      if let Some(row) = luma_row.as_deref_mut() {
        row[i] = (logical >> (BITS - 8)) as u8;
      }
      if let Some(row) = luma_u16_row.as_deref_mut() {
        row[i] = logical;
      }
    }
  }

  // ---- u16 RGB / RGBA path --------------------------------------
  // rgb_u16 (alpha-drop) reuses the non-alpha dispatcher.
  // rgba_u16 — when rgb_u16 is also present (combo case), Strategy A+:
  // expand the already-computed rgb_u16_row → rgba_u16_row then
  // overwrite α slot from the source plane (avoids a second chroma
  // kernel). When rgba_u16 is alone, delegate to the alpha-source-aware
  // dispatcher directly (standalone path — already optimal).
  let want_rgb_u16 = rgb_u16.is_some();
  let want_rgba_u16 = rgba_u16.is_some();
  if want_rgb_u16 {
    let buf = rgb_u16.as_deref_mut().unwrap();
    let rgb_plane_end = one_plane_end
      .checked_mul(3)
      .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
        w, h, 3,
      )))?;
    let rgb_plane_start = one_plane_start * 3;
    let rgb_u16_row = &mut buf[rgb_plane_start..rgb_plane_end];
    rgb_u16_dispatch(
      y_row,
      u_row,
      v_row,
      rgb_u16_row,
      w,
      matrix,
      full_range,
      use_simd,
      BE,
    );
    if want_rgba_u16 {
      // Combo: expand rgb_u16_row → rgba_u16_row, then overwrite α slot.
      let rgba_buf = rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row = rgba_u16_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      expand_rgb_u16_to_rgba_u16_row::<BITS>(rgb_u16_row, rgba_u16_row, w);
      // `BE` is plumbed from the row type so the alpha-plane copy honors
      // the same wire endianness as the caller's high-bit Yuva*p input.
      crate::row::alpha_extract::copy_alpha_plane_u16::<BITS, BE>(a_row, rgba_u16_row, w, use_simd);
    }
  } else if want_rgba_u16 {
    // Standalone rgba_u16: delegate to the alpha-source-aware dispatcher.
    let rgba_buf = rgba_u16.as_deref_mut().unwrap();
    let rgba_u16_row = rgba_u16_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
    rgba_u16_dispatch(
      y_row,
      u_row,
      v_row,
      a_row,
      rgba_u16_row,
      w,
      matrix,
      full_range,
      use_simd,
      BE,
    );
  }

  // ---- u8 RGB / RGBA / HSV path ----------------------------------
  // `need_rgb_kernel` / `rgb_row` were computed and (when needed)
  // allocated at the top, before any caller-output write.
  //
  // HSV-only (no u8 RGB / RGBA): convert the source Y/U/V straight to HSV
  // via the alpha-drop kernel — no source-width RGB scratch. Any u16
  // RGB/RGBA outputs already ran above.
  if want_hsv_direct {
    let hsv = hsv.as_mut().expect("want_hsv_direct implies hsv attached");
    let (h, s, v) = hsv.hsv();
    hsv_dispatch(
      y_row,
      u_row,
      v_row,
      &mut h[one_plane_start..one_plane_end],
      &mut s[one_plane_start..one_plane_end],
      &mut v[one_plane_start..one_plane_end],
      w,
      matrix,
      full_range,
      use_simd,
      BE,
    );
    return Ok(());
  }

  if want_rgba && !need_rgb_kernel {
    let rgba_buf = rgba.as_deref_mut().unwrap();
    let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
    rgba_dispatch(
      y_row, u_row, v_row, a_row, rgba_row, w, matrix, full_range, use_simd, BE,
    );
    return Ok(());
  }

  // RGB kernel (alpha-drop reuses the non-alpha dispatcher verbatim).
  let Some(rgb_row) = rgb_row else {
    return Ok(());
  };
  rgb_dispatch(
    y_row, u_row, v_row, rgb_row, w, matrix, full_range, use_simd, BE,
  );

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

  // Both rgb and rgba attached (combo case): Strategy A+ — expand the
  // already-computed rgb_row → rgba_row (fills α = 0xFF), then
  // overwrite α slot from the source alpha plane with depth-conv
  // `>> (BITS - 8)`. Avoids a second chroma kernel.
  if want_rgba {
    let rgba_buf = rgba.as_deref_mut().unwrap();
    let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
    expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
    // BE = false: see the rgba_u16 branch above for rationale.
    crate::row::alpha_extract::copy_alpha_plane_u16_to_u8::<BITS, BE>(a_row, rgba_row, w, use_simd);
  }

  Ok(())
}

// ---- Shared high-bit YUVA 4:4:4 resample-routing body -----------------
//
// The non-identity plan branch for the 9 / 10 / 12 / 14 / 16-bit YUVA 4:4:4
// sinks. 4:4:4 chroma is full-width (one U / V per Y pixel, no upsampling),
// so the decode closures take full-width chroma rows. The `Area` arm routes
// through the shared packed-YUVA area tail with THREE independent binnings:
// u8 colour, the **independent** native u16 colour (never a narrowing of the
// u8 bin — the u8 / u16 `YUV→RGB` kernels round independently), and the
// **low-packed** native-Y luma (`deinterleave_y_high_bit_masked`, a depth-masked host-native
// copy — planar YUVA Y stores logical values directly, so luma is
// `binned_Y >> (BITS - 8)`, NOT the semi-planar `>> (16 - BITS)` de-pack).
// The `Filter` arm routes the SAME converted RGBA / native-Y through the
// signed-coefficient filter tail (`NATIVE_LUMA_U8 = false`, the u16-luma
// branch — native Y is u16), straight alpha only.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
#[cfg_attr(not(tarpaulin), inline(always))]
fn yuva444p_high_bit_resample<const BITS: u32, const BE: bool>(
  sinker: &mut MixedSinker<'_, impl crate::SourceFormat, impl Sized>,
  idx: usize,
  y_row: &[u16],
  u_row: &[u16],
  v_row: &[u16],
  a_row: &[u16],
  matrix: crate::KernelMatrix,
  full_range: bool,
  y_slice: RowSlice,
  u_slice: RowSlice,
  v_slice: RowSlice,
  a_slice: RowSlice,
  rgba_dispatch: fn(
    &[u16],
    &[u16],
    &[u16],
    &[u16],
    &mut [u8],
    usize,
    crate::KernelMatrix,
    bool,
    bool,
    bool,
  ),
  rgba_u16_dispatch: fn(
    &[u16],
    &[u16],
    &[u16],
    &[u16],
    &mut [u16],
    usize,
    crate::KernelMatrix,
    bool,
    bool,
    bool,
  ),
) -> Result<(), MixedSinkerError> {
  let w = sinker.width;
  let use_simd = sinker.simd;

  if y_row.len() != w {
    return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
      y_slice,
      idx,
      w,
      y_row.len(),
    )));
  }
  if u_row.len() != w {
    return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
      u_slice,
      idx,
      w,
      u_row.len(),
    )));
  }
  if v_row.len() != w {
    return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
      v_slice,
      idx,
      w,
      v_row.len(),
    )));
  }
  if a_row.len() != w {
    return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
      a_slice,
      idx,
      w,
      a_row.len(),
    )));
  }
  if idx >= sinker.height {
    return Err(MixedSinkerError::RowIndexOutOfRange(
      RowIndexOutOfRange::new(idx, sinker.height),
    ));
  }

  let alpha_mode = sinker.alpha_mode;
  let MixedSinker {
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
  } = sinker;
  let plan = plan.as_ref().expect("plan.is_some() checked by the caller");
  check_frozen_alpha_mode(*frozen_alpha_mode, alpha_mode, idx)?;
  // The span kind picks the engine (mirrors the 8-bit `Yuva444p` and packed
  // `Vuya`): `Area` bins (the alpha-aware tail — premultiplied colour binned
  // premultiplied then un-premultiplied) at native precision; `Filter` runs
  // the signed-coefficient filter on the same converted RGBA (straight alpha
  // only). The high-bit native Y is genuinely `u16`, so the filter route uses
  // `NATIVE_LUMA_U8 = false` — the same u16-luma branch `Vuya` / `Vuyx` use:
  // luma rides the `u16` filter stream over the de-interleaved native Y. The
  // filter tail clamps every sub-16-bit colour AND native-Y overshoot to the
  // source's native max `(1 << BITS) - 1` (a value no-op at `BITS = 16`),
  // matching the in-range area path. A premultiplied `Filter` plan has no
  // analogue (the engine cannot un-premultiply), so it is routed to the area
  // tail, which surfaces the typed `UnsupportedFilter`.
  match plan.kind() {
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
      |dst| {
        rgba_dispatch(
          y_row, u_row, v_row, a_row, dst, w, matrix, full_range, use_simd, BE,
        )
      },
      |dst| {
        rgba_u16_dispatch(
          y_row, u_row, v_row, a_row, dst, w, matrix, full_range, use_simd, BE,
        )
      },
      |dst| deinterleave_y_high_bit_masked::<BITS, BE>(y_row, dst, w),
    ),
    crate::resample::SpanKind::Filter if alpha_mode.is_premultiplied() => {
      // Premultiplied + filter has no analogue: route to the area tail with
      // the filter plan so it returns the typed `UnsupportedFilter`.
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
        |dst| {
          rgba_dispatch(
            y_row, u_row, v_row, a_row, dst, w, matrix, full_range, use_simd, BE,
          )
        },
        |dst| {
          rgba_u16_dispatch(
            y_row, u_row, v_row, a_row, dst, w, matrix, full_range, use_simd, BE,
          )
        },
        |dst| deinterleave_y_high_bit_masked::<BITS, BE>(y_row, dst, w),
      )
    }
    crate::resample::SpanKind::Filter => packed_yuva444_filter_resample::<BITS, false, false>(
      rgba_filter_stream,
      rgba_filter_stream_u16,
      // High-bit planar YUVA never uses the u8 native-Y luma stream
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
      // Luma rides `deinterleave_y` + the u16 stream (native Y is u16), so the
      // u8-luma input is unused.
      &[],
      None,
      |dst| {
        rgba_dispatch(
          y_row, u_row, v_row, a_row, dst, w, matrix, full_range, use_simd, BE,
        )
      },
      |dst| {
        rgba_u16_dispatch(
          y_row, u_row, v_row, a_row, dst, w, matrix, full_range, use_simd, BE,
        )
      },
      |dst| deinterleave_y_high_bit_masked::<BITS, BE>(y_row, dst, w),
      |_dst: &mut [u8]| {},
    ),
  }
}
