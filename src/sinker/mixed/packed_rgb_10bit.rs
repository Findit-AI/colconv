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
//! - `with_luma` — drop padding into the u8 RGB scratch via
//!   `x2*_to_rgb_row`, then `rgb_to_luma_row`.
//! - `with_hsv` — same scratch path, then `rgb_to_hsv_row`.
//!
//! `with_rgba_u16` is **not** declared on these source impls — the
//! 2-bit field is padding (no real alpha at native precision), so
//! padding-source sinkers don't fan out a `u16` RGBA output.

use super::{
  MixedSinker, MixedSinkerError, RowSlice, check_dimensions_match, rgb_row_buf_or_scratch,
  rgba_plane_row_slice,
};
use crate::{
  PixelSink,
  row::{
    rgb_to_hsv_row, rgb_to_luma_row, x2bgr10_to_rgb_row, x2bgr10_to_rgb_u16_row,
    x2bgr10_to_rgba_row, x2rgb10_to_rgb_row, x2rgb10_to_rgb_u16_row, x2rgb10_to_rgba_row,
  },
  yuv::{X2Bgr10, X2Bgr10Row, X2Bgr10Sink, X2Rgb10, X2Rgb10Row, X2Rgb10Sink},
};

// ---- X2Rgb10 -----------------------------------------------------------

impl<'a> MixedSinker<'a, X2Rgb10> {
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
    let expected = self.frame_bytes(4)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::RgbaBufferTooShort {
        expected,
        actual: buf.len(),
      });
    }
    self.rgba = Some(buf);
    Ok(self)
  }

  /// Attaches a native-depth `u16` RGB output buffer. Length is
  /// measured in `u16` **elements** (not bytes): minimum
  /// `width × height × 3`. Each 10-bit channel value is preserved
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
    let expected = self.frame_bytes(3)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::RgbU16BufferTooShort {
        expected,
        actual: buf.len(),
      });
    }
    self.rgb_u16 = Some(buf);
    Ok(self)
  }
}

impl X2Rgb10Sink for MixedSinker<'_, X2Rgb10> {}

impl PixelSink for MixedSinker<'_, X2Rgb10> {
  type Input<'r> = X2Rgb10Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: X2Rgb10Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if row.x2rgb10().len() != w * 4 {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::X2Rgb10Packed,
        row: idx,
        expected: w * 4,
        actual: row.x2rgb10().len(),
      });
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange {
        row: idx,
        configured_height: self.height,
      });
    }

    let Self {
      rgb,
      rgb_u16,
      rgba,
      luma,
      hsv,
      rgb_scratch,
      ..
    } = self;
    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;
    let x2rgb10_in = row.x2rgb10();

    let want_rgb = rgb.is_some();
    let want_luma = luma.is_some();
    let want_hsv = hsv.is_some();
    let want_rgb_u16 = rgb_u16.is_some();
    let need_u8_rgb = want_rgb || want_luma || want_hsv;

    // u8 RGB staging path (drives with_rgb / with_luma / with_hsv).
    if need_u8_rgb {
      let rgb_row = rgb_row_buf_or_scratch(
        rgb.as_deref_mut(),
        rgb_scratch,
        one_plane_start,
        one_plane_end,
        w,
        h,
      )?;
      x2rgb10_to_rgb_row(x2rgb10_in, rgb_row, w, use_simd);

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

      if let Some(hsv) = hsv.as_mut() {
        rgb_to_hsv_row(
          rgb_row,
          &mut hsv.h[one_plane_start..one_plane_end],
          &mut hsv.s[one_plane_start..one_plane_end],
          &mut hsv.v[one_plane_start..one_plane_end],
          w,
          use_simd,
        );
      }
    }

    // u8 RGBA output (single-pass, dedicated kernel forces alpha).
    if let Some(buf) = rgba.as_deref_mut() {
      let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      x2rgb10_to_rgba_row(x2rgb10_in, rgba_row, w, use_simd);
    }

    // u16 native RGB output (10-bit precision preserved).
    if want_rgb_u16 {
      let rgb_u16_buf = rgb_u16.as_deref_mut().unwrap();
      let rgb_plane_end =
        one_plane_end
          .checked_mul(3)
          .ok_or(MixedSinkerError::GeometryOverflow {
            width: w,
            height: h,
            channels: 3,
          })?;
      let rgb_plane_start = one_plane_start * 3;
      let rgb_u16_row = &mut rgb_u16_buf[rgb_plane_start..rgb_plane_end];
      x2rgb10_to_rgb_u16_row(x2rgb10_in, rgb_u16_row, w, use_simd);
    }

    Ok(())
  }
}

// ---- X2Bgr10 -----------------------------------------------------------

impl<'a> MixedSinker<'a, X2Bgr10> {
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
    let expected = self.frame_bytes(4)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::RgbaBufferTooShort {
        expected,
        actual: buf.len(),
      });
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
    let expected = self.frame_bytes(3)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::RgbU16BufferTooShort {
        expected,
        actual: buf.len(),
      });
    }
    self.rgb_u16 = Some(buf);
    Ok(self)
  }
}

impl X2Bgr10Sink for MixedSinker<'_, X2Bgr10> {}

impl PixelSink for MixedSinker<'_, X2Bgr10> {
  type Input<'r> = X2Bgr10Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: X2Bgr10Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if row.x2bgr10().len() != w * 4 {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::X2Bgr10Packed,
        row: idx,
        expected: w * 4,
        actual: row.x2bgr10().len(),
      });
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange {
        row: idx,
        configured_height: self.height,
      });
    }

    let Self {
      rgb,
      rgb_u16,
      rgba,
      luma,
      hsv,
      rgb_scratch,
      ..
    } = self;
    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;
    let x2bgr10_in = row.x2bgr10();

    let want_rgb = rgb.is_some();
    let want_luma = luma.is_some();
    let want_hsv = hsv.is_some();
    let want_rgb_u16 = rgb_u16.is_some();
    let need_u8_rgb = want_rgb || want_luma || want_hsv;

    if need_u8_rgb {
      let rgb_row = rgb_row_buf_or_scratch(
        rgb.as_deref_mut(),
        rgb_scratch,
        one_plane_start,
        one_plane_end,
        w,
        h,
      )?;
      x2bgr10_to_rgb_row(x2bgr10_in, rgb_row, w, use_simd);

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

      if let Some(hsv) = hsv.as_mut() {
        rgb_to_hsv_row(
          rgb_row,
          &mut hsv.h[one_plane_start..one_plane_end],
          &mut hsv.s[one_plane_start..one_plane_end],
          &mut hsv.v[one_plane_start..one_plane_end],
          w,
          use_simd,
        );
      }
    }

    if let Some(buf) = rgba.as_deref_mut() {
      let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      x2bgr10_to_rgba_row(x2bgr10_in, rgba_row, w, use_simd);
    }

    if want_rgb_u16 {
      let rgb_u16_buf = rgb_u16.as_deref_mut().unwrap();
      let rgb_plane_end =
        one_plane_end
          .checked_mul(3)
          .ok_or(MixedSinkerError::GeometryOverflow {
            width: w,
            height: h,
            channels: 3,
          })?;
      let rgb_plane_start = one_plane_start * 3;
      let rgb_u16_row = &mut rgb_u16_buf[rgb_plane_start..rgb_plane_end];
      x2bgr10_to_rgb_u16_row(x2bgr10_in, rgb_u16_row, w, use_simd);
    }

    Ok(())
  }
}
