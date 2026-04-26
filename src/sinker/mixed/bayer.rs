//! Bayer / Bayer16 RAW `MixedSinker` impls.

#![allow(unused_imports)]

use super::{
  HsvBuffers, HsvPlane, LumaChannel, LumaCoefficients, MixedSinker, MixedSinkerError, RowSlice,
  check_dimensions_match, rgb_row_to_luma_row,
};
use crate::{ColorMatrix, PixelSink, SourceFormat, raw::*, row::*, yuv::*};

// ---- Bayer (8-bit) impl --------------------------------------------------

impl MixedSinker<'_, Bayer> {
  /// Sets the luma coefficient set used to derive the luma plane
  /// from demosaiced RGB. Only matters when `with_luma` is also
  /// attached. Default: [`LumaCoefficients::Bt709`].
  ///
  /// Pick the set that matches the gamut your
  /// [`crate::raw::ColorCorrectionMatrix`] targets — see
  /// [`LumaCoefficients`] for guidance. Choosing the wrong set
  /// still produces a valid `u8` luma plane, but its numeric
  /// values won't match what a downstream luma-driven analysis
  /// (scene-cut detection, brightness thresholding, perceptual
  /// diff) expects for non-grayscale content.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_luma_coefficients(mut self, coeffs: LumaCoefficients) -> Self {
    self.set_luma_coefficients(coeffs);
    self
  }

  /// In-place variant of
  /// [`with_luma_coefficients`](Self::with_luma_coefficients).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_luma_coefficients(&mut self, coeffs: LumaCoefficients) -> &mut Self {
    self.luma_coefficients_q8 = coeffs.to_q8();
    self
  }
}

impl BayerSink for MixedSinker<'_, Bayer> {}

impl PixelSink for MixedSinker<'_, Bayer> {
  type Input<'r> = BayerRow<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    // Bayer accepts odd dimensions — see `BayerFrame::try_new` for
    // the rationale (cropped Bayer is a real workflow).
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: BayerRow<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    // Defense-in-depth row-shape checks. The walker always hands
    // matching slices, but a caller bypassing the walker (or one of
    // the future unsafe SIMD backends being wired up) needs the
    // no-panic contract: bad lengths surface as `RowShapeMismatch`,
    // not as a kernel-level `assert!` panic.
    if row.mid().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::BayerMid,
        row: idx,
        expected: w,
        actual: row.mid().len(),
      });
    }
    if row.above().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::BayerAbove,
        row: idx,
        expected: w,
        actual: row.above().len(),
      });
    }
    if row.below().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::BayerBelow,
        row: idx,
        expected: w,
        actual: row.below().len(),
      });
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange {
        row: idx,
        configured_height: self.height,
      });
    }

    // `Copy`, captured before the `Self { .. }` destructure so the
    // luma path doesn't have to re-borrow `self`.
    let luma_coeffs_q8 = self.luma_coefficients_q8;

    let Self {
      rgb,
      luma,
      hsv,
      rgb_scratch,
      ..
    } = self;

    let want_rgb = rgb.is_some();
    let want_luma = luma.is_some();
    let want_hsv = hsv.is_some();
    if !want_rgb && !want_luma && !want_hsv {
      return Ok(());
    }

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // 8-bit RGB scratch / output buffer. Bayer always derives every
    // output channel from the demosaiced RGB, so the RGB row exists
    // unconditionally when any of `rgb` / `luma` / `hsv` is set.
    let rgb_row: &mut [u8] = match rgb.as_deref_mut() {
      Some(buf) => {
        let rgb_plane_end =
          one_plane_end
            .checked_mul(3)
            .ok_or(MixedSinkerError::GeometryOverflow {
              width: w,
              height: h,
              channels: 3,
            })?;
        let rgb_plane_start = one_plane_start * 3;
        &mut buf[rgb_plane_start..rgb_plane_end]
      }
      None => {
        let rgb_row_bytes = w.checked_mul(3).ok_or(MixedSinkerError::GeometryOverflow {
          width: w,
          height: h,
          channels: 3,
        })?;
        if rgb_scratch.len() < rgb_row_bytes {
          rgb_scratch.resize(rgb_row_bytes, 0);
        }
        &mut rgb_scratch[..rgb_row_bytes]
      }
    };

    bayer_to_rgb_row(
      row.above(),
      row.mid(),
      row.below(),
      row.row_parity(),
      row.pattern(),
      row.demosaic(),
      row.m(),
      rgb_row,
      use_simd,
    );

    if let Some(luma) = luma.as_deref_mut() {
      rgb_row_to_luma_row(
        rgb_row,
        &mut luma[one_plane_start..one_plane_end],
        luma_coeffs_q8,
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
    Ok(())
  }
}

// ---- Bayer16<BITS> impl --------------------------------------------------

impl<'a, const BITS: u32> MixedSinker<'a, Bayer16<BITS>> {
  /// Attaches a packed **`u16`** RGB output buffer.
  ///
  /// Length is measured in `u16` **elements** (not bytes): minimum
  /// `width × height × 3`. Output is **low-packed** at `BITS`
  /// (10-bit white = 1023, 12-bit = 4095, 14-bit = 16383, 16-bit =
  /// 65535) — matches the rest of the high-bit-depth crate.
  ///
  /// Returns `Err(RgbU16BufferTooShort)` if
  /// `buf.len() < width × height × 3`, or `Err(GeometryOverflow)`
  /// on 32-bit overflow.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgb_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_rgb_u16(buf)?;
    Ok(self)
  }

  /// In-place variant of [`with_rgb_u16`](Self::with_rgb_u16). The
  /// required length is measured in `u16` **elements**, not bytes.
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

  /// Sets the luma coefficient set used to derive the (8-bit)
  /// luma plane from demosaiced RGB. Only matters when `with_luma`
  /// is also attached. Default: [`LumaCoefficients::Bt709`].
  ///
  /// Pick the set that matches the gamut your
  /// [`crate::raw::ColorCorrectionMatrix`] targets — see
  /// [`LumaCoefficients`] for guidance. Choosing the wrong set
  /// still produces a valid `u8` luma plane, but its numeric
  /// values won't match what a downstream luma-driven analysis
  /// (scene-cut detection, brightness thresholding, perceptual
  /// diff) expects for non-grayscale content.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_luma_coefficients(mut self, coeffs: LumaCoefficients) -> Self {
    self.set_luma_coefficients(coeffs);
    self
  }

  /// In-place variant of
  /// [`with_luma_coefficients`](Self::with_luma_coefficients).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_luma_coefficients(&mut self, coeffs: LumaCoefficients) -> &mut Self {
    self.luma_coefficients_q8 = coeffs.to_q8();
    self
  }
}

impl<const BITS: u32> BayerSink16<BITS> for MixedSinker<'_, Bayer16<BITS>> {}

impl<const BITS: u32> PixelSink for MixedSinker<'_, Bayer16<BITS>> {
  type Input<'r> = BayerRow16<'r, BITS>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    // Bayer accepts odd dimensions — see `BayerFrame::try_new` for
    // the rationale (cropped Bayer is a real workflow).
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: BayerRow16<'_, BITS>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    // See the 8-bit Bayer impl for the row-shape rationale.
    if row.mid().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::Bayer16Mid,
        row: idx,
        expected: w,
        actual: row.mid().len(),
      });
    }
    if row.above().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::Bayer16Above,
        row: idx,
        expected: w,
        actual: row.above().len(),
      });
    }
    if row.below().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::Bayer16Below,
        row: idx,
        expected: w,
        actual: row.below().len(),
      });
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange {
        row: idx,
        configured_height: self.height,
      });
    }

    // `Copy`, captured before the `Self { .. }` destructure so the
    // luma path doesn't have to re-borrow `self`.
    let luma_coeffs_q8 = self.luma_coefficients_q8;

    let Self {
      rgb,
      rgb_u16,
      luma,
      hsv,
      rgb_scratch,
      ..
    } = self;

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // u16 RGB output runs the native-depth kernel directly. Output
    // is low-packed at `BITS` per the `*_to_rgb_u16_row` convention.
    if let Some(buf) = rgb_u16.as_deref_mut() {
      let rgb_plane_end =
        one_plane_end
          .checked_mul(3)
          .ok_or(MixedSinkerError::GeometryOverflow {
            width: w,
            height: h,
            channels: 3,
          })?;
      let rgb_plane_start = one_plane_start * 3;
      bayer16_to_rgb_u16_row::<BITS>(
        row.above(),
        row.mid(),
        row.below(),
        row.row_parity(),
        row.pattern(),
        row.demosaic(),
        row.m(),
        &mut buf[rgb_plane_start..rgb_plane_end],
        use_simd,
      );
    }

    let want_rgb = rgb.is_some();
    let want_luma = luma.is_some();
    let want_hsv = hsv.is_some();
    if !want_rgb && !want_luma && !want_hsv {
      return Ok(());
    }

    // 8-bit RGB scratch / output. Same lazy-grow pattern as the
    // 8-bit Bayer impl above.
    let rgb_row: &mut [u8] = match rgb.as_deref_mut() {
      Some(buf) => {
        let rgb_plane_end =
          one_plane_end
            .checked_mul(3)
            .ok_or(MixedSinkerError::GeometryOverflow {
              width: w,
              height: h,
              channels: 3,
            })?;
        let rgb_plane_start = one_plane_start * 3;
        &mut buf[rgb_plane_start..rgb_plane_end]
      }
      None => {
        let rgb_row_bytes = w.checked_mul(3).ok_or(MixedSinkerError::GeometryOverflow {
          width: w,
          height: h,
          channels: 3,
        })?;
        if rgb_scratch.len() < rgb_row_bytes {
          rgb_scratch.resize(rgb_row_bytes, 0);
        }
        &mut rgb_scratch[..rgb_row_bytes]
      }
    };

    bayer16_to_rgb_row::<BITS>(
      row.above(),
      row.mid(),
      row.below(),
      row.row_parity(),
      row.pattern(),
      row.demosaic(),
      row.m(),
      rgb_row,
      use_simd,
    );

    if let Some(luma) = luma.as_deref_mut() {
      rgb_row_to_luma_row(
        rgb_row,
        &mut luma[one_plane_start..one_plane_end],
        luma_coeffs_q8,
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
    Ok(())
  }
}
