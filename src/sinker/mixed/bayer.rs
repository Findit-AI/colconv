//! Bayer / Bayer16 RAW `MixedSinker` impls.

use super::{
  GeometryOverflow, InsufficientBuffer, LumaCoefficients, MixedSinker, MixedSinkerError,
  RowIndexOutOfRange, RowShapeMismatch, RowSlice, check_dimensions_match, frozen_outputs_check,
  rgb_row_buf_or_scratch, rgb_row_to_luma_row, source_rgb_scratch,
};
use crate::{PixelSink, raw::*, row::*};

// ---- Bayer (8-bit) impl --------------------------------------------------

impl<R> MixedSinker<'_, Bayer, R> {
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

impl<R> BayerSink for MixedSinker<'_, Bayer, R> {}

impl<R> PixelSink for MixedSinker<'_, Bayer, R> {
  type Input<'r> = BayerRow<'r>;
  type Error = MixedSinkerError;

  #[cfg_attr(not(tarpaulin), inline(always))]
  fn kernel_matrix(&self) -> crate::KernelMatrix {
    self.kernel_matrix
  }
  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    // Bayer accepts odd dimensions — see `BayerFrame::try_new` for
    // the rationale (cropped Bayer is a real workflow).
    check_dimensions_match(self.width, self.height, width, height)?;
    if let Some(stream) = self.rgb_stream.as_mut() {
      stream.reset();
    }
    self.resample_outputs = None;
    Ok(())
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
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::BayerMid,
        idx,
        w,
        row.mid().len(),
      )));
    }
    if row.above().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::BayerAbove,
        idx,
        w,
        row.above().len(),
      )));
    }
    if row.below().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::BayerBelow,
        idx,
        w,
        row.below().len(),
      )));
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange(
        RowIndexOutOfRange::new(idx, self.height),
      ));
    }

    // `Copy`, captured before the `Self { .. }` destructure so the
    // luma path doesn't have to re-borrow `self`.
    let luma_coeffs_q8 = self.luma_coefficients_q8;

    let Self {
      rgb,
      luma,
      hsv,
      rgb_scratch,
      plan,
      rgb_stream,
      resample_outputs,
      ..
    } = self;

    // Non-identity plan: run the shared L1 compare-only preflight (no-output
    // no-op + out-of-sequence + mid-frame output-set-change rejection), build the
    // area stream and grow the scratch, then COMMIT the output-set freeze only
    // after both allocations succeed (RFC #238 tail-collapse: a first-row
    // allocation failure leaves `resample_outputs` uncommitted, retryable with a
    // changed output set). Only then demosaic the CFA row into the source-width
    // RGB scratch and feed the 3-channel area stream. Every output derives from
    // each finalized (binned) RGB row via the *same Bayer derivations the
    // identity path uses* — Q8-coefficient luma (`rgb_row_to_luma_row`) and the
    // OpenCV HSV kernel — so a resampled frame matches a direct demosaic of the
    // source followed by an area-bin of that RGB. Bayer carries no
    // `KernelMatrix`/`full_range`, so it CANNOT share the packed-RGB tail's
    // `KernelMatrix`-based emit; it keeps its BESPOKE emit inline below and shares
    // only the transactional preflight/stream/scratch/freeze plumbing.
    if let Some(plan) = plan.as_ref() {
      // `rgba` and `luma_u16` are never attached on a Bayer sink (RGB-only
      // family), so those shared-preflight slots are `&None`.
      let need_any = rgb.is_some() || luma.is_some() || hsv.is_some();
      let expected = if need_any {
        Some(rgb_stream.as_ref().map_or(0, |s| s.next_y()))
      } else {
        None
      };
      if let core::ops::ControlFlow::Break(()) = super::resample_preflight_check_only(
        resample_outputs,
        luma,
        &None,
        rgb,
        &None,
        &None,
        &None,
        &None,
        &None,
        &None,
        &None,
        &None,
        hsv,
        &None,
        expected,
        idx,
      )? {
        return Ok(());
      }
      // Area-only: reject a filter plan before any allocation (Bayer is not routed
      // to the filter engine).
      if plan.kind().is_filter() {
        return Err(plan.unsupported_filter().into());
      }
      // Build the missing area stream into a LOCAL and grow the scratch — every
      // fallible pre-feed step — with NO field mutation, so the stream field insert
      // and the freeze commit TOGETHER only after both succeed (the
      // `planar_dual_resample` atomic shape: a scratch OOM leaves `rgb_stream`
      // `None` AND `resample_outputs` uncommitted, never a partial commit).
      let new_stream = if rgb_stream.is_none() {
        let stream =
          crate::resample::AreaStream::new(plan.h(), plan.v(), plan.src_w(), plan.src_h(), 3)?;
        Some(crate::resample::try_box(stream).map_err(|_| {
          MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
            crate::resample::PlanGeometry::new(
              plan.src_w(),
              plan.src_h(),
              plan.out_w(),
              plan.out_h(),
            ),
          ))
        })?)
      } else {
        None
      };
      source_rgb_scratch(rgb_scratch, w, plan)?;
      // Every pre-feed allocation succeeded — COMMIT atomically: insert the stream
      // and freeze the output set together (the compare-only preflight above
      // already caught any later output change).
      if let Some(s) = new_stream {
        *rgb_stream = Some(s);
      }
      frozen_outputs_check(
        resample_outputs,
        luma,
        &None,
        rgb,
        &None,
        &None,
        &None,
        &None,
        &None,
        &None,
        &None,
        &None,
        hsv,
        &None,
        idx,
      )?;
      // Demosaic the CFA row into the (grown) scratch and feed the BESPOKE emit
      // (Q8 luma + OpenCV HSV, no `KernelMatrix`); this scratch grow re-runs as a
      // no-op after the pre-commit grow above.
      let stream = rgb_stream.as_mut().expect("created above");
      let scratch = source_rgb_scratch(rgb_scratch, w, plan)?;
      bayer_to_rgb_row(
        row.above(),
        row.mid(),
        row.below(),
        row.row_parity(),
        row.pattern(),
        row.demosaic(),
        row.m(),
        scratch,
        use_simd,
      );
      let ow = plan.out_w();
      stream.feed_row(idx, scratch, use_simd, |oy, out_row| {
        if let Some(buf) = rgb.as_deref_mut() {
          buf[oy * 3 * ow..(oy + 1) * 3 * ow].copy_from_slice(out_row);
        }
        if let Some(buf) = luma.as_deref_mut() {
          rgb_row_to_luma_row(out_row, &mut buf[oy * ow..(oy + 1) * ow], luma_coeffs_q8);
        }
        if let Some(hsv) = hsv.as_mut() {
          let (h, s, v) = hsv.hsv();
          rgb_to_hsv_row(
            out_row,
            &mut h[oy * ow..(oy + 1) * ow],
            &mut s[oy * ow..(oy + 1) * ow],
            &mut v[oy * ow..(oy + 1) * ow],
            ow,
            use_simd,
          );
        }
      })?;
      return Ok(());
    }

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
    let rgb_row = rgb_row_buf_or_scratch(
      rgb.as_deref_mut(),
      rgb_scratch,
      one_plane_start,
      one_plane_end,
      w,
      h,
    )?;

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
    Ok(())
  }
}

// ---- Bayer16<BITS, BE> impl ----------------------------------------------

impl<'a, R, const BITS: u32, const BE: bool> MixedSinker<'a, Bayer16<BITS, BE>, R> {
  /// Attaches a packed **`u16`** RGB output buffer.
  ///
  /// Length is measured in `u16` **elements** (not bytes): minimum
  /// `width x height x 3`. Output is **low-packed** at `BITS`
  /// (10-bit white = 1023, 12-bit = 4095, 14-bit = 16383, 16-bit =
  /// 65535) — matches the rest of the high-bit-depth crate.
  ///
  /// Returns `Err(InsufficientRgbU16Buffer)` if
  /// `buf.len() < width x height x 3`, or `Err(GeometryOverflow)`
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
    let expected = self.frame_elems(3)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::InsufficientRgbU16Buffer(
        InsufficientBuffer::new(expected, buf.len()),
      ));
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

impl<const BITS: u32, const BE: bool> BayerSink16<BITS, BE> for MixedSinker<'_, Bayer16<BITS, BE>> {}

impl<const BITS: u32, const BE: bool> PixelSink for MixedSinker<'_, Bayer16<BITS, BE>> {
  type Input<'r> = BayerRow16<'r, BITS>;
  type Error = MixedSinkerError;

  #[cfg_attr(not(tarpaulin), inline(always))]
  fn kernel_matrix(&self) -> crate::KernelMatrix {
    self.kernel_matrix
  }
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
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Bayer16Mid,
        idx,
        w,
        row.mid().len(),
      )));
    }
    if row.above().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Bayer16Above,
        idx,
        w,
        row.above().len(),
      )));
    }
    if row.below().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Bayer16Below,
        idx,
        w,
        row.below().len(),
      )));
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange(
        RowIndexOutOfRange::new(idx, self.height),
      ));
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
          .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
            w, h, 3,
          )))?;
      let rgb_plane_start = one_plane_start * 3;
      bayer16_to_rgb_u16_row_endian::<BITS, BE>(
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
    let rgb_row = rgb_row_buf_or_scratch(
      rgb.as_deref_mut(),
      rgb_scratch,
      one_plane_start,
      one_plane_end,
      w,
      h,
    )?;

    bayer16_to_rgb_row_endian::<BITS, BE>(
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
    Ok(())
  }
}
