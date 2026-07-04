//! Sinker impls for 16-bit packed-RGB **source** formats (Tier 8 finish).
//!
//! Sources:
//! - [`Rgb48`]  — `R, G, B` u16 per pixel (`AV_PIX_FMT_RGB48LE`).
//! - [`Bgr48`]  — `B, G, R` u16 per pixel (`AV_PIX_FMT_BGR48LE`).
//! - [`Rgba64`] — `R, G, B, A` u16 per pixel (`AV_PIX_FMT_RGBA64LE`).
//! - [`Bgra64`] — `B, G, R, A` u16 per pixel (`AV_PIX_FMT_BGRA64LE`).
//!
//! All 7 output paths per format:
//! - `with_rgb`      — packed 16-bit → packed u8 RGB (narrow `>> 8` per channel).
//! - `with_rgba`     — same narrow; for Rgb48/Bgr48 alpha = `0xFF` (no source α);
//!   for Rgba64/Bgra64 source α is passed through (`>> 8`).
//! - `with_rgb_u16`  — native u16 passthrough (3 elements per pixel, R/G/B order).
//! - `with_rgba_u16` — native u16; for Rgb48/Bgr48 alpha = `0xFFFF`; for Rgba64/
//!   Bgra64 source α is copied verbatim (no shift).
//! - `with_luma`     — Y' derived from narrowed u8 RGB via `rgb_to_luma_row`.
//! - `with_luma_u16` — Y' derived from narrowed u8 RGB, zero-extended to u16.
//! - `with_hsv`      — HSV derived from narrowed u8 RGB via `rgb_to_hsv_row`.
//!
//! ## Alpha semantics — Rgba64 / Bgra64 (Strategy A+)
//!
//! When both `with_rgb` and `with_rgba` are attached, the u8 RGB kernel runs
//! **once** and RGBA is derived via `expand_rgb_to_rgba_row` (writes α=`0xFF`)
//! followed by `copy_alpha_packed_u16x4_to_u8_at_3` to overwrite slot 3 from
//! the source α. Output is byte-identical to calling `rgba64_to_rgba_row`
//! (standalone path) directly — spec § 3.2 / § 7.2.
//!
//! The same Strategy A+ applies on the u16 path: `expand_rgb_u16_to_rgba_u16_row`
//! fans out, then `copy_alpha_packed_u16x4_at_3` overwrites α from slot 3.

use super::{
  GeometryOverflow, InsufficientBuffer, MixedSinker, MixedSinkerError, RowIndexOutOfRange,
  RowShapeMismatch, RowSlice, check_dimensions_match, check_frozen_alpha_mode,
  packed_rgb_u16_resample, packed_rgb_u16_resample_filter, packed_rgba_u16_filter_resample,
  packed_rgba_u16_resample, rgb_row_buf_or_scratch, rgba_plane_row_slice, rgba_u16_plane_row_slice,
};
use crate::{
  PixelSink,
  row::{
    bgr48_to_rgb_row_endian, bgr48_to_rgb_u16_row_endian, bgr48_to_rgba_row_endian,
    bgr48_to_rgba_u16_row_endian, bgra64_to_rgb_row_endian, bgra64_to_rgb_u16_row_endian,
    bgra64_to_rgba_row_endian, bgra64_to_rgba_u16_row_endian, expand_rgb_to_rgba_row,
    expand_rgb_u16_to_rgba_u16_row, rgb_to_hsv_row, rgb_to_luma_row, rgb_to_luma_u16_row,
    rgb48_to_rgb_row_endian, rgb48_to_rgb_u16_row_endian, rgb48_to_rgba_row_endian,
    rgb48_to_rgba_u16_row_endian, rgba64_to_rgb_row_endian, rgba64_to_rgb_u16_row_endian,
    rgba64_to_rgba_row_endian, rgba64_to_rgba_u16_row_endian,
  },
  source::{
    Bgr48, Bgr48Row, Bgr48Sink, Bgra64, Bgra64Row, Bgra64Sink, Rgb48, Rgb48Row, Rgb48Sink, Rgba64,
    Rgba64Row, Rgba64Sink,
  },
};

// ---- Rgb48 -----------------------------------------------------------------

impl<'a, R, const BE: bool> MixedSinker<'a, Rgb48<BE>, R> {
  /// Attaches a packed **8-bit** RGBA output buffer. Each 16-bit channel is
  /// narrowed `>> 8` and alpha is forced to `0xFF` (no source alpha in Rgb48).
  ///
  /// Returns `Err(InsufficientRgbaBuffer)` if `buf.len() < width x height x 4`,
  /// or `Err(GeometryOverflow)` on 32-bit targets when the product overflows.
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

  /// Attaches a native **`u16`** RGB output buffer. Length in `u16` **elements**
  /// (`width x height x 3`). Channels are passed through verbatim (no shift).
  ///
  /// Returns `Err(InsufficientRgbU16Buffer)` if the buffer is too short.
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

  /// Attaches a native **`u16`** RGBA output buffer. Length in `u16` **elements**
  /// (`width x height x 4`). Alpha is forced to `0xFFFF` (no source alpha).
  ///
  /// Returns `Err(InsufficientRgbaU16Buffer)` if the buffer is too short.
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

  /// Attaches a native **`u16`** luma output buffer. Length in `u16` **elements**
  /// (`width x height`). Y' is computed at 8-bit precision and zero-extended.
  ///
  /// Returns `Err(InsufficientLumaU16Buffer)` if the buffer is too short.
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

impl<R, const BE: bool> Rgb48Sink<BE> for MixedSinker<'_, Rgb48<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, Rgb48<BE>, R> {
  type Input<'r> = Rgb48Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    if let Some(stream) = self.rgb_stream_u16.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.rgb_filter_stream_u16.as_mut() {
      stream.reset();
    }
    self.resample_outputs = None;
    self.frozen_alpha_mode = Some(self.alpha_mode);
    Ok(())
  }

  fn process(&mut self, row: Rgb48Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    let packed_expected =
      w.checked_mul(3)
        .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
          w, h, 3,
        )))?;
    if row.rgb48().len() != packed_expected {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Rgb48Packed,
        idx,
        packed_expected,
        row.rgb48().len(),
      )));
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange(
        RowIndexOutOfRange::new(idx, self.height),
      ));
    }

    let Self {
      rgb,
      rgb_u16,
      rgba,
      rgba_u16,
      luma,
      luma_u16,
      hsv,
      rgb_scratch,
      rgb_scratch_u16,
      rgb_stream_u16,
      rgb_filter_stream_u16,
      resample_outputs,
      plan,
      ..
    } = self;

    // Non-identity plan: convert the wire row to source-width host u16
    // RGB, bin it at native 16-bit depth, then derive every attached
    // output from each finalized output row (native-depth u16 outputs
    // copy the binned row; u8 / luma_u16 outputs narrow it `>> 8`). The
    // span kind picks the engine — integer area or signed-coefficient
    // filter (both bin the same staged native-u16 row).
    if let Some(plan) = plan.as_ref() {
      // Transactional resample (RFC #238 tail-collapse): stage the wire row into the
      // source-width native-u16 RGB scratch and feed the shared 3-channel u16 helper
      // (`SRC_BITS = 16`, narrowed `luma_u16`); the span kind picks the engine
      // (integer area / signed-coefficient filter, both binning the staged row).
      return match plan.kind() {
        crate::resample::SpanKind::Area => packed_rgb_u16_resample::<16, false>(
          rgb_stream_u16,
          resample_outputs,
          rgb,
          rgba,
          luma,
          rgb_u16,
          rgba_u16,
          luma_u16,
          hsv,
          rgb_scratch_u16,
          rgb_scratch,
          w,
          plan,
          idx,
          use_simd,
          row.matrix(),
          row.full_range(),
          |src_u16| rgb48_to_rgb_u16_row_endian::<BE>(row.rgb48(), src_u16, w, use_simd),
        ),
        crate::resample::SpanKind::Filter => packed_rgb_u16_resample_filter::<16, false>(
          rgb_filter_stream_u16,
          resample_outputs,
          rgb,
          rgba,
          luma,
          rgb_u16,
          rgba_u16,
          luma_u16,
          hsv,
          rgb_scratch_u16,
          rgb_scratch,
          w,
          plan,
          idx,
          use_simd,
          row.matrix(),
          row.full_range(),
          |src_u16| rgb48_to_rgb_u16_row_endian::<BE>(row.rgb48(), src_u16, w, use_simd),
        ),
      };
    }

    let ps = idx * w;
    let pe = ps + w;
    let in48 = row.rgb48();

    let want_rgb = rgb.is_some();
    let want_luma = luma.is_some();
    let want_luma_u16 = luma_u16.is_some();
    let want_hsv = hsv.is_some();
    let need_u8_rgb = want_rgb || want_luma || want_luma_u16 || want_hsv;

    // u8 RGB staging — required when any of: with_rgb, with_luma,
    // with_luma_u16, or with_hsv is attached.
    if need_u8_rgb {
      let rgb_row = rgb_row_buf_or_scratch(rgb.as_deref_mut(), rgb_scratch, ps, pe, w, h)?;
      rgb48_to_rgb_row_endian::<BE>(in48, rgb_row, w, use_simd);

      if let Some(luma_buf) = luma.as_deref_mut() {
        rgb_to_luma_row(
          rgb_row,
          &mut luma_buf[ps..pe],
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
        );
      }

      if let Some(luma_u16_buf) = luma_u16.as_deref_mut() {
        rgb_to_luma_u16_row(
          rgb_row,
          &mut luma_u16_buf[ps..pe],
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
        );
      }

      if let Some(hsv_bufs) = hsv.as_mut() {
        let (h, s, v) = hsv_bufs.hsv();
        rgb_to_hsv_row(
          rgb_row,
          &mut h[ps..pe],
          &mut s[ps..pe],
          &mut v[ps..pe],
          w,
          use_simd,
        );
      }
    }

    // u8 RGBA — single-pass kernel, alpha forced to 0xFF.
    if let Some(buf) = rgba.as_deref_mut() {
      let rgba_row = rgba_plane_row_slice(buf, ps, pe, w, h)?;
      rgb48_to_rgba_row_endian::<BE>(in48, rgba_row, w, use_simd);
    }

    // u16 RGB — native passthrough.
    if let Some(buf) = rgb_u16.as_deref_mut() {
      let end =
        pe.checked_mul(3)
          .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
            w, h, 3,
          )))?;
      rgb48_to_rgb_u16_row_endian::<BE>(in48, &mut buf[ps * 3..end], w, use_simd);
    }

    // u16 RGBA — native passthrough, alpha forced to 0xFFFF.
    if let Some(buf) = rgba_u16.as_deref_mut() {
      let rgba_u16_row = rgba_u16_plane_row_slice(buf, ps, pe, w, h)?;
      rgb48_to_rgba_u16_row_endian::<BE>(in48, rgba_u16_row, w, use_simd);
    }

    Ok(())
  }
}

// ---- Bgr48 -----------------------------------------------------------------

impl<'a, R, const BE: bool> MixedSinker<'a, Bgr48<BE>, R> {
  /// Attaches a packed **8-bit** RGBA output buffer. B/R channels are swapped
  /// on output; each 16-bit channel is narrowed `>> 8`; alpha is forced to
  /// `0xFF` (no source alpha in Bgr48).
  ///
  /// Returns `Err(InsufficientRgbaBuffer)` if `buf.len() < width x height x 4`.
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

  /// Attaches a native **`u16`** RGB output buffer. Length in `u16` **elements**
  /// (`width x height x 3`). B/R channels are swapped on output; no shift.
  ///
  /// Returns `Err(InsufficientRgbU16Buffer)` if the buffer is too short.
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

  /// Attaches a native **`u16`** RGBA output buffer. Length in `u16` **elements**
  /// (`width x height x 4`). B/R channels swapped; alpha forced to `0xFFFF`.
  ///
  /// Returns `Err(InsufficientRgbaU16Buffer)` if the buffer is too short.
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

  /// Attaches a native **`u16`** luma output buffer (`width x height` elements).
  /// Y' is computed at 8-bit precision and zero-extended to u16.
  ///
  /// Returns `Err(InsufficientLumaU16Buffer)` if the buffer is too short.
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

impl<R, const BE: bool> Bgr48Sink<BE> for MixedSinker<'_, Bgr48<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, Bgr48<BE>, R> {
  type Input<'r> = Bgr48Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    if let Some(stream) = self.rgb_stream_u16.as_mut() {
      stream.reset();
    }
    self.resample_outputs = None;
    self.frozen_alpha_mode = Some(self.alpha_mode);
    Ok(())
  }

  fn process(&mut self, row: Bgr48Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    let packed_expected =
      w.checked_mul(3)
        .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
          w, h, 3,
        )))?;
    if row.bgr48().len() != packed_expected {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Bgr48Packed,
        idx,
        packed_expected,
        row.bgr48().len(),
      )));
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange(
        RowIndexOutOfRange::new(idx, self.height),
      ));
    }

    let Self {
      rgb,
      rgb_u16,
      rgba,
      rgba_u16,
      luma,
      luma_u16,
      hsv,
      rgb_scratch,
      rgb_scratch_u16,
      rgb_stream_u16,
      resample_outputs,
      plan,
      ..
    } = self;

    // Non-identity plan: convert the wire row to source-width host u16
    // RGB, bin it at native 16-bit depth, then derive every attached
    // output from each finalized output row.
    if let Some(plan) = plan.as_ref() {
      // Transactional area resample (RFC #238 tail-collapse) — the Bgr48 twin of the
      // Rgb48 path (`SRC_BITS = 16`, narrowed `luma_u16`, area-only).
      return packed_rgb_u16_resample::<16, false>(
        rgb_stream_u16,
        resample_outputs,
        rgb,
        rgba,
        luma,
        rgb_u16,
        rgba_u16,
        luma_u16,
        hsv,
        rgb_scratch_u16,
        rgb_scratch,
        w,
        plan,
        idx,
        use_simd,
        row.matrix(),
        row.full_range(),
        |src_u16| bgr48_to_rgb_u16_row_endian::<BE>(row.bgr48(), src_u16, w, use_simd),
      );
    }

    let ps = idx * w;
    let pe = ps + w;
    let in48 = row.bgr48();

    let want_rgb = rgb.is_some();
    let want_luma = luma.is_some();
    let want_luma_u16 = luma_u16.is_some();
    let want_hsv = hsv.is_some();
    let need_u8_rgb = want_rgb || want_luma || want_luma_u16 || want_hsv;

    if need_u8_rgb {
      let rgb_row = rgb_row_buf_or_scratch(rgb.as_deref_mut(), rgb_scratch, ps, pe, w, h)?;
      bgr48_to_rgb_row_endian::<BE>(in48, rgb_row, w, use_simd);

      if let Some(luma_buf) = luma.as_deref_mut() {
        rgb_to_luma_row(
          rgb_row,
          &mut luma_buf[ps..pe],
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
        );
      }

      if let Some(luma_u16_buf) = luma_u16.as_deref_mut() {
        rgb_to_luma_u16_row(
          rgb_row,
          &mut luma_u16_buf[ps..pe],
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
        );
      }

      if let Some(hsv_bufs) = hsv.as_mut() {
        let (h, s, v) = hsv_bufs.hsv();
        rgb_to_hsv_row(
          rgb_row,
          &mut h[ps..pe],
          &mut s[ps..pe],
          &mut v[ps..pe],
          w,
          use_simd,
        );
      }
    }

    if let Some(buf) = rgba.as_deref_mut() {
      let rgba_row = rgba_plane_row_slice(buf, ps, pe, w, h)?;
      bgr48_to_rgba_row_endian::<BE>(in48, rgba_row, w, use_simd);
    }

    if let Some(buf) = rgb_u16.as_deref_mut() {
      let end =
        pe.checked_mul(3)
          .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
            w, h, 3,
          )))?;
      bgr48_to_rgb_u16_row_endian::<BE>(in48, &mut buf[ps * 3..end], w, use_simd);
    }

    if let Some(buf) = rgba_u16.as_deref_mut() {
      let rgba_u16_row = rgba_u16_plane_row_slice(buf, ps, pe, w, h)?;
      bgr48_to_rgba_u16_row_endian::<BE>(in48, rgba_u16_row, w, use_simd);
    }

    Ok(())
  }
}

// ---- Rgba64 ----------------------------------------------------------------

impl<'a, R, const BE: bool> MixedSinker<'a, Rgba64<BE>, R> {
  /// Attaches a packed **8-bit** RGBA output buffer. Each 16-bit channel is
  /// narrowed `>> 8`; the **source alpha** at slot 3 of each pixel is
  /// depth-converted and passed through (not forced to `0xFF`).
  ///
  /// ## Strategy note
  ///
  /// Source-α pass-through is guaranteed in **all** paths. When standalone
  /// (no `with_rgb` / `with_hsv`), `rgba64_to_rgba_row` runs directly.
  /// When combined with `with_rgb`, Strategy A+ applies:
  /// `expand_rgb_to_rgba_row` fans out the RGB row (α=`0xFF`) and
  /// `copy_alpha_packed_u16x4_to_u8_at_3` overwrites the α slot from the
  /// source — output is byte-identical to the standalone path.
  ///
  /// Returns `Err(InsufficientRgbaBuffer)` if `buf.len() < width x height x 4`.
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

  /// Attaches a native **`u16`** RGB output buffer. Length in `u16` **elements**
  /// (`width x height x 3`). Alpha slot dropped (RGB only, 3 channels).
  ///
  /// Returns `Err(InsufficientRgbU16Buffer)` if the buffer is too short.
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

  /// Attaches a native **`u16`** RGBA output buffer. Length in `u16` **elements**
  /// (`width x height x 4`). Source α at slot 3 is copied verbatim (no shift).
  ///
  /// ## Strategy note
  ///
  /// When `with_rgb_u16` is also attached, Strategy A+ applies on the u16
  /// path: `expand_rgb_u16_to_rgba_u16_row` fans out and
  /// `copy_alpha_packed_u16x4_at_3` overwrites α — output byte-identical
  /// to the standalone `rgba64_to_rgba_u16_row` path.
  ///
  /// Returns `Err(InsufficientRgbaU16Buffer)` if the buffer is too short.
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

  /// Attaches a native **`u16`** luma output buffer (`width x height` elements).
  /// Y' is derived from narrowed u8 RGB and zero-extended to u16.
  ///
  /// Returns `Err(InsufficientLumaU16Buffer)` if the buffer is too short.
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

impl<R, const BE: bool> Rgba64Sink<BE> for MixedSinker<'_, Rgba64<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, Rgba64<BE>, R> {
  type Input<'r> = Rgba64Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    if let Some(stream) = self.rgb_stream_u16.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.rgba_stream_u16.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.rgb_filter_stream_u16.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.rgba_filter_stream_u16.as_mut() {
      stream.reset();
    }
    self.resample_outputs = None;
    self.frozen_alpha_mode = Some(self.alpha_mode);
    Ok(())
  }

  fn process(&mut self, row: Rgba64Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    let packed_expected =
      w.checked_mul(4)
        .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
          w, h, 4,
        )))?;
    if row.rgba64().len() != packed_expected {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Rgba64Packed,
        idx,
        packed_expected,
        row.rgba64().len(),
      )));
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange(
        RowIndexOutOfRange::new(idx, self.height),
      ));
    }

    // Non-identity plan. Route the alpha-aware 4-channel u16 tail when
    // resampled alpha would be dropped (rgba / rgba_u16 attached) or the
    // color must be alpha-weighted (premultiplied); otherwise the
    // rgb-only straight outputs keep the 3-channel u16 RGB path. Rgba64
    // stages canonical host-native RGBA via `rgba64_to_rgba_u16_row_endian`
    // (identity, α pass-through) and drop-alpha RGB via
    // `rgba64_to_rgb_u16_row_endian`.
    if self.plan.is_some() {
      let alpha_mode = self.alpha_mode;
      let in64 = row.rgba64();
      let matrix = row.matrix();
      let full_range = row.full_range();
      let Self {
        rgb,
        rgb_u16,
        rgba,
        rgba_u16,
        luma,
        luma_u16,
        hsv,
        rgb_scratch,
        rgb_scratch_u16,
        rgba_scratch_u16,
        rgba_color_scratch_u16,
        rgb_stream_u16,
        rgba_stream_u16,
        rgb_filter_stream_u16,
        rgba_filter_stream_u16,
        resample_outputs,
        frozen_alpha_mode,
        plan,
        ..
      } = self;
      let plan = plan.as_ref().expect("plan.is_some() checked above");
      // The alpha mode is snapshotted at begin_frame; reject a mid-frame
      // change here, before route selection (it picks the 4-channel vs
      // 3-channel route), so a flip can neither reroute nor mix modes. The
      // span kind then picks the engine: the integer area stream or the
      // signed-coefficient filter stream (PIL parity, straight alpha only).
      check_frozen_alpha_mode(*frozen_alpha_mode, alpha_mode, idx)?;
      match plan.kind() {
        crate::resample::SpanKind::Area => {
          if rgba.is_some() || rgba_u16.is_some() || alpha_mode.is_premultiplied() {
            return packed_rgba_u16_resample::<16, false, false>(
              rgba_stream_u16,
              // No native-Y luma stream: `Rgba64` / `Bgra64` are chromatic, so
              // luma is color-derived (`NATIVE_Y_LUMA = false`) and the Y
              // stream / scratch / de-interleave are inert placeholders.
              &mut None,
              resample_outputs,
              rgb,
              rgba,
              rgb_u16,
              rgba_u16,
              luma,
              luma_u16,
              hsv,
              rgba_scratch_u16,
              rgba_color_scratch_u16,
              rgb_scratch,
              rgb_scratch_u16,
              &mut std::vec::Vec::new(),
              w,
              plan,
              idx,
              use_simd,
              alpha_mode,
              matrix,
              full_range,
              |dst| crate::row::rgba64_to_rgba_u16_row_endian::<BE>(in64, dst, w, use_simd),
              |_| {},
            );
          }
          // Straight rgb-only (alpha dropped): drop-alpha RGB via the u16 kernel and
          // feed the shared 3-channel u16 area helper (RFC #238 tail-collapse).
          return packed_rgb_u16_resample::<16, false>(
            rgb_stream_u16,
            resample_outputs,
            rgb,
            rgba,
            luma,
            rgb_u16,
            rgba_u16,
            luma_u16,
            hsv,
            rgb_scratch_u16,
            rgb_scratch,
            w,
            plan,
            idx,
            use_simd,
            matrix,
            full_range,
            |src_u16| crate::row::rgba64_to_rgb_u16_row_endian::<BE>(in64, src_u16, w, use_simd),
          );
        }
        crate::resample::SpanKind::Filter => {
          // Premultiplied alpha has no filter analogue (the filter engine
          // cannot un-premultiply, so it cannot derive a straight color);
          // surface the typed `UnsupportedFilter` via the area tail's reject
          // rather than emit straight-filtered premultiplied color. Straight
          // alpha: filter all four native u16 channels independently (PIL RGBA
          // semantics) when alpha survives, else the 3-channel u16 filter for
          // rgb-only outputs (forced-opaque alpha, like the direct path).
          if alpha_mode.is_premultiplied() {
            return packed_rgba_u16_resample::<16, false, false>(
              rgba_stream_u16,
              &mut None,
              resample_outputs,
              rgb,
              rgba,
              rgb_u16,
              rgba_u16,
              luma,
              luma_u16,
              hsv,
              rgba_scratch_u16,
              rgba_color_scratch_u16,
              rgb_scratch,
              rgb_scratch_u16,
              &mut std::vec::Vec::new(),
              w,
              plan,
              idx,
              use_simd,
              alpha_mode,
              matrix,
              full_range,
              |dst| crate::row::rgba64_to_rgba_u16_row_endian::<BE>(in64, dst, w, use_simd),
              |_| {},
            );
          }
          if rgba.is_some() || rgba_u16.is_some() {
            return packed_rgba_u16_filter_resample::<16, false>(
              rgba_filter_stream_u16,
              resample_outputs,
              rgb,
              rgba,
              rgb_u16,
              rgba_u16,
              luma,
              luma_u16,
              hsv,
              rgba_scratch_u16,
              rgba_color_scratch_u16,
              rgb_scratch,
              w,
              plan,
              idx,
              use_simd,
              matrix,
              full_range,
              |dst| crate::row::rgba64_to_rgba_u16_row_endian::<BE>(in64, dst, w, use_simd),
            );
          }
          // Straight rgb-only (alpha dropped): the 3-channel u16 filter twin.
          return packed_rgb_u16_resample_filter::<16, false>(
            rgb_filter_stream_u16,
            resample_outputs,
            rgb,
            rgba,
            luma,
            rgb_u16,
            rgba_u16,
            luma_u16,
            hsv,
            rgb_scratch_u16,
            rgb_scratch,
            w,
            plan,
            idx,
            use_simd,
            matrix,
            full_range,
            |src_u16| crate::row::rgba64_to_rgb_u16_row_endian::<BE>(in64, src_u16, w, use_simd),
          );
        }
      }
    }

    let Self {
      rgb,
      rgb_u16,
      rgba,
      rgba_u16,
      luma,
      luma_u16,
      hsv,
      rgb_scratch,
      ..
    } = self;
    let ps = idx * w;
    let pe = ps + w;
    let in64 = row.rgba64();

    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_luma = luma.is_some();
    let want_luma_u16 = luma_u16.is_some();
    let want_hsv = hsv.is_some();
    let want_rgb_u16 = rgb_u16.is_some();
    let want_rgba_u16 = rgba_u16.is_some();
    let need_u8_rgb = want_rgb || want_luma || want_luma_u16 || want_hsv;

    // ===== u8 path =====

    // Standalone RGBA u8 fast path — only rgba attached (no u8 RGB or u16
    // work). Source α passes through via the kernel.
    if want_rgba && !need_u8_rgb && !want_rgb_u16 && !want_rgba_u16 {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, ps, pe, w, h)?;
      rgba64_to_rgba_row_endian::<BE>(in64, rgba_row, w, use_simd);
      return Ok(());
    }

    // Standalone RGBA u16 fast path — only rgba_u16 attached, no u8 work.
    if want_rgba_u16 && !want_rgb_u16 && !need_u8_rgb && !want_rgba {
      let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row = rgba_u16_plane_row_slice(rgba_u16_buf, ps, pe, w, h)?;
      rgba64_to_rgba_u16_row_endian::<BE>(in64, rgba_u16_row, w, use_simd);
      return Ok(());
    }

    // u8 RGB staging — drives with_rgb / with_luma / with_luma_u16 / with_hsv,
    // and Strategy A+ RGBA fan-out.
    if need_u8_rgb {
      let rgb_row = rgb_row_buf_or_scratch(rgb.as_deref_mut(), rgb_scratch, ps, pe, w, h)?;
      rgba64_to_rgb_row_endian::<BE>(in64, rgb_row, w, use_simd);

      if let Some(luma_buf) = luma.as_deref_mut() {
        rgb_to_luma_row(
          rgb_row,
          &mut luma_buf[ps..pe],
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
        );
      }

      if let Some(luma_u16_buf) = luma_u16.as_deref_mut() {
        rgb_to_luma_u16_row(
          rgb_row,
          &mut luma_u16_buf[ps..pe],
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
        );
      }

      if let Some(hsv_bufs) = hsv.as_mut() {
        let (h, s, v) = hsv_bufs.hsv();
        rgb_to_hsv_row(
          rgb_row,
          &mut h[ps..pe],
          &mut s[ps..pe],
          &mut v[ps..pe],
          w,
          use_simd,
        );
      }

      // Strategy A+ u8: RGBA also attached — derive from the just-computed
      // RGB row (writes α=0xFF), then overwrite α slot from packed source
      // (slot 3, depth-conv >> 8). Output is byte-identical to calling
      // rgba64_to_rgba_row directly (spec § 3.2 / § 7.2).
      if want_rgba {
        let rgba_buf = rgba.as_deref_mut().unwrap();
        let rgba_row = rgba_plane_row_slice(rgba_buf, ps, pe, w, h)?;
        expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
        // BE propagated from the parent `Rgba64Frame<'_, BE>` via the
        // sinker's `MixedSinker<Rgba64<BE>>` monomorphization.
        crate::row::scalar::alpha_extract::copy_alpha_packed_u16x4_to_u8_at_3::<BE>(
          in64, rgba_row, w,
        );
      }
    }

    // Standalone RGBA u8 path — want_rgba without need_u8_rgb (combined with
    // u16 work only). Run rgba64_to_rgba_row directly; source α depth-conv.
    if want_rgba && !need_u8_rgb {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, ps, pe, w, h)?;
      rgba64_to_rgba_row_endian::<BE>(in64, rgba_row, w, use_simd);
    }

    // ===== u16 path =====

    if want_rgb_u16 {
      let rgb_u16_buf = rgb_u16.as_deref_mut().unwrap();
      let end =
        pe.checked_mul(3)
          .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
            w, h, 3,
          )))?;
      let rgb_u16_row = &mut rgb_u16_buf[ps * 3..end];
      rgba64_to_rgb_u16_row_endian::<BE>(in64, rgb_u16_row, w, use_simd);

      // Strategy A+ u16: RGBA u16 also attached — derive from the
      // just-computed u16 RGB row (writes α=0xFFFF), then overwrite α
      // slot from packed source (slot 3, u16 verbatim).
      if want_rgba_u16 {
        let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
        let rgba_u16_row = rgba_u16_plane_row_slice(rgba_u16_buf, ps, pe, w, h)?;
        expand_rgb_u16_to_rgba_u16_row::<16>(rgb_u16_row, rgba_u16_row, w);
        // BE propagated from the parent `Rgba64Frame<'_, BE>` via the
        // sinker's `MixedSinker<Rgba64<BE>>` monomorphization.
        crate::row::scalar::alpha_extract::copy_alpha_packed_u16x4_at_3::<BE>(
          in64,
          rgba_u16_row,
          w,
        );
      }
    }

    // Standalone RGBA u16 path — want_rgba_u16 without want_rgb_u16 (combined
    // with u8 work). Run rgba64_to_rgba_u16_row directly.
    if want_rgba_u16 && !want_rgb_u16 {
      let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row = rgba_u16_plane_row_slice(rgba_u16_buf, ps, pe, w, h)?;
      rgba64_to_rgba_u16_row_endian::<BE>(in64, rgba_u16_row, w, use_simd);
    }

    Ok(())
  }
}

// ---- Bgra64 ----------------------------------------------------------------

impl<'a, R, const BE: bool> MixedSinker<'a, Bgra64<BE>, R> {
  /// Attaches a packed **8-bit** RGBA output buffer. B/R channels swapped on
  /// output; each 16-bit channel narrowed `>> 8`; the **source alpha** at slot
  /// 3 of each pixel is depth-converted and passed through (not forced).
  ///
  /// Same Strategy A+ semantics as [`MixedSinker::<Rgba64>::with_rgba`] —
  /// see that method's doc for the standalone vs combo behaviour.
  ///
  /// Returns `Err(InsufficientRgbaBuffer)` if `buf.len() < width x height x 4`.
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

  /// Attaches a native **`u16`** RGB output buffer. B/R channels swapped on
  /// output; length in `u16` **elements** (`width x height x 3`).
  ///
  /// Returns `Err(InsufficientRgbU16Buffer)` if the buffer is too short.
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

  /// Attaches a native **`u16`** RGBA output buffer. B/R channels swapped;
  /// source α at slot 3 copied verbatim. Length in `u16` **elements**
  /// (`width x height x 4`).
  ///
  /// Returns `Err(InsufficientRgbaU16Buffer)` if the buffer is too short.
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

  /// Attaches a native **`u16`** luma output buffer (`width x height` elements).
  /// Y' is derived from narrowed u8 RGB and zero-extended to u16.
  ///
  /// Returns `Err(InsufficientLumaU16Buffer)` if the buffer is too short.
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

impl<R, const BE: bool> Bgra64Sink<BE> for MixedSinker<'_, Bgra64<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, Bgra64<BE>, R> {
  type Input<'r> = Bgra64Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    if let Some(stream) = self.rgb_stream_u16.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.rgba_stream_u16.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.rgb_filter_stream_u16.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.rgba_filter_stream_u16.as_mut() {
      stream.reset();
    }
    self.resample_outputs = None;
    self.frozen_alpha_mode = Some(self.alpha_mode);
    Ok(())
  }

  fn process(&mut self, row: Bgra64Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    let packed_expected =
      w.checked_mul(4)
        .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
          w, h, 4,
        )))?;
    if row.bgra64().len() != packed_expected {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Bgra64Packed,
        idx,
        packed_expected,
        row.bgra64().len(),
      )));
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange(
        RowIndexOutOfRange::new(idx, self.height),
      ));
    }

    // Non-identity plan — see the `Rgba64` impl for the routing
    // rationale. Bgra64 stages canonical host-native RGBA via
    // `bgra64_to_rgba_u16_row_endian` (swap R↔B, α pass-through) and
    // drop-alpha RGB via `bgra64_to_rgb_u16_row_endian`.
    if self.plan.is_some() {
      let alpha_mode = self.alpha_mode;
      let in64 = row.bgra64();
      let matrix = row.matrix();
      let full_range = row.full_range();
      let Self {
        rgb,
        rgb_u16,
        rgba,
        rgba_u16,
        luma,
        luma_u16,
        hsv,
        rgb_scratch,
        rgb_scratch_u16,
        rgba_scratch_u16,
        rgba_color_scratch_u16,
        rgb_stream_u16,
        rgba_stream_u16,
        rgb_filter_stream_u16,
        rgba_filter_stream_u16,
        resample_outputs,
        frozen_alpha_mode,
        plan,
        ..
      } = self;
      let plan = plan.as_ref().expect("plan.is_some() checked above");
      // The alpha mode is snapshotted at begin_frame; reject a mid-frame
      // change here, before route selection (it picks the 4-channel vs
      // 3-channel route), so a flip can neither reroute nor mix modes. The
      // span kind then picks the engine — see the `Rgba64` impl for the
      // routing rationale.
      check_frozen_alpha_mode(*frozen_alpha_mode, alpha_mode, idx)?;
      match plan.kind() {
        crate::resample::SpanKind::Area => {
          if rgba.is_some() || rgba_u16.is_some() || alpha_mode.is_premultiplied() {
            return packed_rgba_u16_resample::<16, false, false>(
              rgba_stream_u16,
              // No native-Y luma stream: `Rgba64` / `Bgra64` are chromatic, so
              // luma is color-derived (`NATIVE_Y_LUMA = false`) and the Y
              // stream / scratch / de-interleave are inert placeholders.
              &mut None,
              resample_outputs,
              rgb,
              rgba,
              rgb_u16,
              rgba_u16,
              luma,
              luma_u16,
              hsv,
              rgba_scratch_u16,
              rgba_color_scratch_u16,
              rgb_scratch,
              rgb_scratch_u16,
              &mut std::vec::Vec::new(),
              w,
              plan,
              idx,
              use_simd,
              alpha_mode,
              matrix,
              full_range,
              |dst| crate::row::bgra64_to_rgba_u16_row_endian::<BE>(in64, dst, w, use_simd),
              |_| {},
            );
          }
          // Straight rgb-only (alpha dropped): the Bgra64 twin of the Rgba64
          // 3-channel u16 area path (RFC #238 tail-collapse).
          return packed_rgb_u16_resample::<16, false>(
            rgb_stream_u16,
            resample_outputs,
            rgb,
            rgba,
            luma,
            rgb_u16,
            rgba_u16,
            luma_u16,
            hsv,
            rgb_scratch_u16,
            rgb_scratch,
            w,
            plan,
            idx,
            use_simd,
            matrix,
            full_range,
            |src_u16| crate::row::bgra64_to_rgb_u16_row_endian::<BE>(in64, src_u16, w, use_simd),
          );
        }
        crate::resample::SpanKind::Filter => {
          // Premultiplied: no filter analogue — surface `UnsupportedFilter`
          // via the area tail's reject. Straight: filter all four native u16
          // channels independently when alpha survives, else the 3-channel
          // u16 filter for rgb-only outputs. See the `Rgba64` impl.
          if alpha_mode.is_premultiplied() {
            return packed_rgba_u16_resample::<16, false, false>(
              rgba_stream_u16,
              &mut None,
              resample_outputs,
              rgb,
              rgba,
              rgb_u16,
              rgba_u16,
              luma,
              luma_u16,
              hsv,
              rgba_scratch_u16,
              rgba_color_scratch_u16,
              rgb_scratch,
              rgb_scratch_u16,
              &mut std::vec::Vec::new(),
              w,
              plan,
              idx,
              use_simd,
              alpha_mode,
              matrix,
              full_range,
              |dst| crate::row::bgra64_to_rgba_u16_row_endian::<BE>(in64, dst, w, use_simd),
              |_| {},
            );
          }
          if rgba.is_some() || rgba_u16.is_some() {
            return packed_rgba_u16_filter_resample::<16, false>(
              rgba_filter_stream_u16,
              resample_outputs,
              rgb,
              rgba,
              rgb_u16,
              rgba_u16,
              luma,
              luma_u16,
              hsv,
              rgba_scratch_u16,
              rgba_color_scratch_u16,
              rgb_scratch,
              w,
              plan,
              idx,
              use_simd,
              matrix,
              full_range,
              |dst| crate::row::bgra64_to_rgba_u16_row_endian::<BE>(in64, dst, w, use_simd),
            );
          }
          // Straight rgb-only (alpha dropped): the Bgra64 twin of the Rgba64
          // 3-channel u16 filter path.
          return packed_rgb_u16_resample_filter::<16, false>(
            rgb_filter_stream_u16,
            resample_outputs,
            rgb,
            rgba,
            luma,
            rgb_u16,
            rgba_u16,
            luma_u16,
            hsv,
            rgb_scratch_u16,
            rgb_scratch,
            w,
            plan,
            idx,
            use_simd,
            matrix,
            full_range,
            |src_u16| crate::row::bgra64_to_rgb_u16_row_endian::<BE>(in64, src_u16, w, use_simd),
          );
        }
      }
    }

    let Self {
      rgb,
      rgb_u16,
      rgba,
      rgba_u16,
      luma,
      luma_u16,
      hsv,
      rgb_scratch,
      ..
    } = self;
    let ps = idx * w;
    let pe = ps + w;
    let in64 = row.bgra64();

    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_luma = luma.is_some();
    let want_luma_u16 = luma_u16.is_some();
    let want_hsv = hsv.is_some();
    let want_rgb_u16 = rgb_u16.is_some();
    let want_rgba_u16 = rgba_u16.is_some();
    let need_u8_rgb = want_rgb || want_luma || want_luma_u16 || want_hsv;

    // Standalone RGBA u8 fast path.
    if want_rgba && !need_u8_rgb && !want_rgb_u16 && !want_rgba_u16 {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, ps, pe, w, h)?;
      bgra64_to_rgba_row_endian::<BE>(in64, rgba_row, w, use_simd);
      return Ok(());
    }

    // Standalone RGBA u16 fast path.
    if want_rgba_u16 && !want_rgb_u16 && !need_u8_rgb && !want_rgba {
      let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row = rgba_u16_plane_row_slice(rgba_u16_buf, ps, pe, w, h)?;
      bgra64_to_rgba_u16_row_endian::<BE>(in64, rgba_u16_row, w, use_simd);
      return Ok(());
    }

    // u8 RGB staging path.
    if need_u8_rgb {
      let rgb_row = rgb_row_buf_or_scratch(rgb.as_deref_mut(), rgb_scratch, ps, pe, w, h)?;
      bgra64_to_rgb_row_endian::<BE>(in64, rgb_row, w, use_simd);

      if let Some(luma_buf) = luma.as_deref_mut() {
        rgb_to_luma_row(
          rgb_row,
          &mut luma_buf[ps..pe],
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
        );
      }

      if let Some(luma_u16_buf) = luma_u16.as_deref_mut() {
        rgb_to_luma_u16_row(
          rgb_row,
          &mut luma_u16_buf[ps..pe],
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
        );
      }

      if let Some(hsv_bufs) = hsv.as_mut() {
        let (h, s, v) = hsv_bufs.hsv();
        rgb_to_hsv_row(
          rgb_row,
          &mut h[ps..pe],
          &mut s[ps..pe],
          &mut v[ps..pe],
          w,
          use_simd,
        );
      }

      // Strategy A+ u8: RGBA also attached.
      if want_rgba {
        let rgba_buf = rgba.as_deref_mut().unwrap();
        let rgba_row = rgba_plane_row_slice(rgba_buf, ps, pe, w, h)?;
        expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
        // BE propagated from the parent `Bgra64Frame<'_, BE>` via the
        // sinker's `MixedSinker<Bgra64<BE>>` monomorphization.
        crate::row::scalar::alpha_extract::copy_alpha_packed_u16x4_to_u8_at_3::<BE>(
          in64, rgba_row, w,
        );
      }
    }

    // Standalone RGBA u8 path — combined with u16 work only.
    if want_rgba && !need_u8_rgb {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, ps, pe, w, h)?;
      bgra64_to_rgba_row_endian::<BE>(in64, rgba_row, w, use_simd);
    }

    // u16 RGB path.
    if want_rgb_u16 {
      let rgb_u16_buf = rgb_u16.as_deref_mut().unwrap();
      let end =
        pe.checked_mul(3)
          .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
            w, h, 3,
          )))?;
      let rgb_u16_row = &mut rgb_u16_buf[ps * 3..end];
      bgra64_to_rgb_u16_row_endian::<BE>(in64, rgb_u16_row, w, use_simd);

      // Strategy A+ u16: RGBA u16 also attached.
      if want_rgba_u16 {
        let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
        let rgba_u16_row = rgba_u16_plane_row_slice(rgba_u16_buf, ps, pe, w, h)?;
        expand_rgb_u16_to_rgba_u16_row::<16>(rgb_u16_row, rgba_u16_row, w);
        // BE propagated from the parent `Bgra64Frame<'_, BE>` via the
        // sinker's `MixedSinker<Bgra64<BE>>` monomorphization.
        crate::row::scalar::alpha_extract::copy_alpha_packed_u16x4_at_3::<BE>(
          in64,
          rgba_u16_row,
          w,
        );
      }
    }

    // Standalone RGBA u16 path — combined with u8 work.
    if want_rgba_u16 && !want_rgb_u16 {
      let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row = rgba_u16_plane_row_slice(rgba_u16_buf, ps, pe, w, h)?;
      bgra64_to_rgba_u16_row_endian::<BE>(in64, rgba_u16_row, w, use_simd);
    }

    Ok(())
  }
}
