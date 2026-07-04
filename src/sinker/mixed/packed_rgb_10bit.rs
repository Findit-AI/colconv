//! Sinker impls for 10-bit packed-RGB **source** formats (Tier 6 —
//! Ship 9e). Each source pixel is a 32-bit little-endian word with
//! `(MSB) 2X | 10c2 | 10c1 | 10c0 (LSB)` packing — the 2 leading
//! bits are ignored padding.
//!
//! Sources:
//! - [`X2Rgb10`] — c2/c1/c0 = R/G/B (FFmpeg `AV_PIX_FMT_X2RGB10LE`).
//! - [`X2Bgr10`] — c2/c1/c0 = B/G/R (FFmpeg `AV_PIX_FMT_X2BGR10LE`).
//!
//! Outputs (per source):
//! - `with_rgb` — `x2rgb10_to_rgb_row` / `x2bgr10_to_rgb_row`
//!   (extract 10-bit channels, down-shift to 8 bits, pack as
//!   `R, G, B`).
//! - `with_rgba` — `x2rgb10_to_rgba_row` / `x2bgr10_to_rgba_row`
//!   (same down-shift + force alpha to `0xFF`; the source has no
//!   real alpha).
//! - `with_rgb_u16` — `x2rgb10_to_rgb_u16_row` /
//!   `x2bgr10_to_rgb_u16_row` (native 10-bit precision, low-bit
//!   aligned in `u16`, max value `1023`).
//! - `with_rgba_u16` — the native 10-bit `rgb_u16` row expanded to
//!   RGBA with opaque alpha `(1 << 10) - 1 = 1023` via
//!   `expand_rgb_u16_to_rgba_u16_row::<10>` (no native α in the
//!   source — the 2-bit field is padding). Matches `Rgb48`'s
//!   `rgba_u16`, scaled to this source's 10-bit depth.
//! - `with_luma` — drop padding into the u8 RGB scratch via
//!   `x2*_to_rgb_row`, then `rgb_to_luma_row`.
//! - `with_luma_u16` — Y' derived from the same narrowed u8 RGB via
//!   `rgb_to_luma_u16_row`, zero-extended to u16. The 2-bit field is
//!   padding (no native-depth luma kernel for this source), so luma is
//!   8-bit-precise — matching the `Rgb48` narrowed-`luma_u16` contract.
//! - `with_hsv` — same scratch path, then `rgb_to_hsv_row`.
//!
//! This is the full `u16` output set (`rgb_u16` + `rgba_u16` +
//! `luma_u16`) that `Rgb48` exposes, scaled to the source's 10-bit
//! depth. The 2-bit field is padding (no real alpha at native
//! precision), so `rgba_u16` carries a synthesized opaque alpha
//! (`1023`), not a passed-through source channel.
//!
//! # Fused area-resample (`with_resampler`)
//!
//! On a non-identity plan each source unpacks its packed 10-bit row into
//! a source-width packed `u16` RGB row (`x2*_to_rgb_u16_row`, channels in
//! `0..=1023`) and feeds the shared packed-RGB resample tail
//! ([`packed_rgb_u16_resample_emit`](super::packed_rgb_u16_resample_emit)
//! and siblings) — the same tail the `Rgb48` / `Bgr48` and high-bit GBR
//! sources take, with `SRC_BITS = 10` driving the u8 narrowing
//! (`>> 2`). Binning runs at native 10-bit depth: `rgb_u16` copies the
//! binned row and `rgba_u16` expands it with opaque alpha `1023`, while
//! `rgb` / `rgba` / `luma` / `luma_u16` / `hsv` derive from a single
//! narrowing of it. `luma_u16` stays the tail's narrowed
//! variant (`NATIVE_LUMA16 = false`) — byte-identical to this source's
//! own narrowed direct path.

use super::{
  GeometryOverflow, InsufficientBuffer, MixedSinker, MixedSinkerError, RowIndexOutOfRange,
  RowShapeMismatch, RowSlice, check_dimensions_match, direct_rgb_u16_scratch,
  packed_rgb_u16_resample, rgb_row_buf_or_scratch, rgba_plane_row_slice, rgba_u16_plane_row_slice,
};
use crate::{
  PixelSink,
  row::{
    expand_rgb_u16_to_rgba_u16_row, rgb_to_hsv_row, rgb_to_luma_row, rgb_to_luma_u16_row,
    x2bgr10_to_rgb_row_endian, x2bgr10_to_rgb_u16_row_endian, x2bgr10_to_rgba_row_endian,
    x2rgb10_to_rgb_row_endian, x2rgb10_to_rgb_u16_row_endian, x2rgb10_to_rgba_row_endian,
  },
  source::{X2Bgr10, X2Bgr10Row, X2Bgr10Sink, X2Rgb10, X2Rgb10Row, X2Rgb10Sink},
};

// ---- X2Rgb10 -----------------------------------------------------------

impl<'a, R, const BE: bool> MixedSinker<'a, X2Rgb10<BE>, R> {
  /// Attaches a packed **8-bit** RGBA output buffer. Each 10-bit
  /// channel is down-shifted to 8 bits and alpha is forced to
  /// `0xFF` (the source has no real alpha — the 2-bit field is
  /// padding).
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

  /// Attaches a native-depth `u16` RGB output buffer. Length is
  /// measured in `u16` **elements** (not bytes): minimum
  /// `width x height x 3`. Each 10-bit channel value is preserved
  /// at full precision in the low 10 bits of its `u16` element
  /// (range `[0, 1023]`).
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

  /// Attaches a native-depth `u16` RGBA output buffer. Length is
  /// measured in `u16` **elements**: minimum `width x height x 4`. Each
  /// 10-bit channel is preserved at full precision (range `[0, 1023]`)
  /// and alpha is forced to `(1 << 10) - 1 = 1023` — the source has no
  /// real alpha (the 2-bit field is padding).
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

  /// Attaches a native **`u16`** luma output buffer. Length in `u16`
  /// **elements** (`width x height`). Y' is computed at 8-bit precision
  /// from the narrowed u8 RGB and zero-extended — the source's 2-bit
  /// field is padding, so there is no native-depth luma kernel.
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

impl<R, const BE: bool> X2Rgb10Sink<BE> for MixedSinker<'_, X2Rgb10<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, X2Rgb10<BE>, R> {
  type Input<'r> = X2Rgb10Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    if let Some(stream) = self.rgb_stream_u16.as_mut() {
      stream.reset();
    }
    self.resample_outputs = None;
    Ok(())
  }

  fn process(&mut self, row: X2Rgb10Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if row.x2rgb10().len() != w * 4 {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::X2Rgb10Packed,
        idx,
        w * 4,
        row.x2rgb10().len(),
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

    // Non-identity plan: unpack the packed 10-bit wire row into a
    // source-width host u16 RGB row (channels 0..=1023), bin it at
    // native 10-bit depth, then derive every attached output from each
    // finalized output row. `rgb_u16` copies the binned row, `rgba_u16`
    // expands it with opaque alpha 1023; the u8 / luma_u16 outputs narrow
    // it `>> 2` (SRC_BITS = 10). luma_u16 takes the tail's narrowed
    // variant — byte-identical to this source's own narrowed direct path
    // (no native 10-bit luma kernel exists).
    if let Some(plan) = plan.as_ref() {
      // Transactional area resample (RFC #238 tail-collapse): unpack the packed
      // 10-bit wire row into the source-width native-u16 RGB staging scratch and
      // feed the shared 3-channel u16 helper (`SRC_BITS = 10`, narrowed
      // `luma_u16`). X2Rgb10 is area-only — a filter plan is rejected inside.
      return packed_rgb_u16_resample::<10, false>(
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
        |src_u16| x2rgb10_to_rgb_u16_row_endian::<BE>(row.x2rgb10(), src_u16, w, use_simd),
      );
    }

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;
    let x2rgb10_in = row.x2rgb10();

    let want_rgb = rgb.is_some();
    let want_luma = luma.is_some();
    let want_luma_u16 = luma_u16.is_some();
    let want_hsv = hsv.is_some();
    let want_rgb_u16 = rgb_u16.is_some();
    let want_rgba_u16 = rgba_u16.is_some();
    let need_u8_rgb = want_rgb || want_luma || want_luma_u16 || want_hsv;

    // Allocate the rgba_u16 staging scratch up front — before any output
    // buffer is written — so an allocator refusal returns a recoverable
    // error rather than leaving a partially-written caller buffer.
    let rgba_u16_staging = if want_rgba_u16 {
      Some(direct_rgb_u16_scratch(rgb_scratch_u16, w, h)?)
    } else {
      None
    };

    // u8 RGB staging path (drives with_rgb / with_luma / with_luma_u16 /
    // with_hsv).
    if need_u8_rgb {
      let rgb_row = rgb_row_buf_or_scratch(
        rgb.as_deref_mut(),
        rgb_scratch,
        one_plane_start,
        one_plane_end,
        w,
        h,
      )?;
      x2rgb10_to_rgb_row_endian::<BE>(x2rgb10_in, rgb_row, w, use_simd);

      if let Some(luma) = luma.as_deref_mut() {
        rgb_to_luma_row(
          rgb_row,
          &mut luma[one_plane_start..one_plane_end],
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
        );
      }

      if let Some(luma_u16) = luma_u16.as_deref_mut() {
        rgb_to_luma_u16_row(
          rgb_row,
          &mut luma_u16[one_plane_start..one_plane_end],
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
        );
      }

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
    }

    // u8 RGBA output (single-pass, dedicated kernel forces alpha).
    if let Some(buf) = rgba.as_deref_mut() {
      let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      x2rgb10_to_rgba_row_endian::<BE>(x2rgb10_in, rgba_row, w, use_simd);
    }

    // u16 native RGB output (10-bit precision preserved).
    if want_rgb_u16 {
      let rgb_u16_buf = rgb_u16.as_deref_mut().unwrap();
      let rgb_plane_end =
        one_plane_end
          .checked_mul(3)
          .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
            w, h, 3,
          )))?;
      let rgb_plane_start = one_plane_start * 3;
      let rgb_u16_row = &mut rgb_u16_buf[rgb_plane_start..rgb_plane_end];
      x2rgb10_to_rgb_u16_row_endian::<BE>(x2rgb10_in, rgb_u16_row, w, use_simd);
    }

    // u16 native RGBA output: build the native 10-bit RGB row into the
    // up-front-allocated staging scratch, then expand to RGBA with opaque
    // alpha 1023. The source has no native α kernel, so this mirrors
    // `Rgb48`'s `expand_rgb_u16_to_rgba_u16_row` fan-out at 10-bit depth.
    if let Some(src_u16) = rgba_u16_staging {
      x2rgb10_to_rgb_u16_row_endian::<BE>(x2rgb10_in, src_u16, w, use_simd);
      let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row =
        rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
      expand_rgb_u16_to_rgba_u16_row::<10>(src_u16, rgba_u16_row, w);
    }

    Ok(())
  }
}

// ---- X2Bgr10 -----------------------------------------------------------

impl<'a, R, const BE: bool> MixedSinker<'a, X2Bgr10<BE>, R> {
  /// Attaches a packed **8-bit** RGBA output buffer. Channel order
  /// is reversed on output (input bit positions: `R` at low, `B` at
  /// high) and alpha is forced to `0xFF`.
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

  /// Attaches a native-depth `u16` RGB output buffer. See
  /// [`MixedSinker::<X2Rgb10>::with_rgb_u16`] for the same layout
  /// contract.
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

  /// Attaches a native-depth `u16` RGBA output buffer. See
  /// [`MixedSinker::<X2Rgb10>::with_rgba_u16`] for the same layout and
  /// opaque-alpha (`1023`) contract; B/R channel order is resolved on
  /// unpack.
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

  /// Attaches a native **`u16`** luma output buffer. See
  /// [`MixedSinker::<X2Rgb10>::with_luma_u16`] for the narrowed-luma
  /// contract.
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

impl<R, const BE: bool> X2Bgr10Sink<BE> for MixedSinker<'_, X2Bgr10<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, X2Bgr10<BE>, R> {
  type Input<'r> = X2Bgr10Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    if let Some(stream) = self.rgb_stream_u16.as_mut() {
      stream.reset();
    }
    self.resample_outputs = None;
    Ok(())
  }

  fn process(&mut self, row: X2Bgr10Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if row.x2bgr10().len() != w * 4 {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::X2Bgr10Packed,
        idx,
        w * 4,
        row.x2bgr10().len(),
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

    // Non-identity plan: unpack the packed 10-bit wire row into a
    // source-width host u16 RGB row (channel order resolved on unpack;
    // values 0..=1023), bin at native 10-bit depth, then derive every
    // attached output. See the X2Rgb10 path for the SRC_BITS = 10 /
    // narrowed-luma_u16 rationale; `rgba_u16` expands the binned row with
    // opaque alpha 1023.
    if let Some(plan) = plan.as_ref() {
      // Transactional area resample (RFC #238 tail-collapse) — the X2Bgr10 twin of
      // the X2Rgb10 path (`SRC_BITS = 10`, narrowed `luma_u16`, area-only).
      return packed_rgb_u16_resample::<10, false>(
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
        |src_u16| x2bgr10_to_rgb_u16_row_endian::<BE>(row.x2bgr10(), src_u16, w, use_simd),
      );
    }

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;
    let x2bgr10_in = row.x2bgr10();

    let want_rgb = rgb.is_some();
    let want_luma = luma.is_some();
    let want_luma_u16 = luma_u16.is_some();
    let want_hsv = hsv.is_some();
    let want_rgb_u16 = rgb_u16.is_some();
    let want_rgba_u16 = rgba_u16.is_some();
    let need_u8_rgb = want_rgb || want_luma || want_luma_u16 || want_hsv;

    // Allocate the rgba_u16 staging scratch up front — before any output
    // buffer is written — so an allocator refusal returns a recoverable
    // error rather than leaving a partially-written caller buffer.
    let rgba_u16_staging = if want_rgba_u16 {
      Some(direct_rgb_u16_scratch(rgb_scratch_u16, w, h)?)
    } else {
      None
    };

    if need_u8_rgb {
      let rgb_row = rgb_row_buf_or_scratch(
        rgb.as_deref_mut(),
        rgb_scratch,
        one_plane_start,
        one_plane_end,
        w,
        h,
      )?;
      x2bgr10_to_rgb_row_endian::<BE>(x2bgr10_in, rgb_row, w, use_simd);

      if let Some(luma) = luma.as_deref_mut() {
        rgb_to_luma_row(
          rgb_row,
          &mut luma[one_plane_start..one_plane_end],
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
        );
      }

      if let Some(luma_u16) = luma_u16.as_deref_mut() {
        rgb_to_luma_u16_row(
          rgb_row,
          &mut luma_u16[one_plane_start..one_plane_end],
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
        );
      }

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
    }

    if let Some(buf) = rgba.as_deref_mut() {
      let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      x2bgr10_to_rgba_row_endian::<BE>(x2bgr10_in, rgba_row, w, use_simd);
    }

    if want_rgb_u16 {
      let rgb_u16_buf = rgb_u16.as_deref_mut().unwrap();
      let rgb_plane_end =
        one_plane_end
          .checked_mul(3)
          .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
            w, h, 3,
          )))?;
      let rgb_plane_start = one_plane_start * 3;
      let rgb_u16_row = &mut rgb_u16_buf[rgb_plane_start..rgb_plane_end];
      x2bgr10_to_rgb_u16_row_endian::<BE>(x2bgr10_in, rgb_u16_row, w, use_simd);
    }

    // u16 native RGBA: stage the native 10-bit RGB row (B/R resolved on
    // unpack) into the up-front-allocated scratch, then expand with opaque
    // alpha 1023. See the X2Rgb10 path for the no-native-α-kernel rationale.
    if let Some(src_u16) = rgba_u16_staging {
      x2bgr10_to_rgb_u16_row_endian::<BE>(x2bgr10_in, src_u16, w, use_simd);
      let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row =
        rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
      expand_rgb_u16_to_rgba_u16_row::<10>(src_u16, rgba_u16_row, w);
    }

    Ok(())
  }
}
