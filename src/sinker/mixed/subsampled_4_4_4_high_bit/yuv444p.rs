use super::super::{
  GeometryOverflow, InsufficientBuffer, MixedSinker, MixedSinkerError, NativeRouteChanged,
  RowIndexOutOfRange, RowShapeMismatch, RowSlice, UnsupportedMatrixResample,
  check_dimensions_match, deinterleave_y_high_bit_masked, packed_yuv444_triple_filter_resample,
  packed_yuv444_triple_resample, reset_high_bit_yuv_streams, rgb_row_buf_or_scratch,
  rgba_plane_row_slice, rgba_u16_plane_row_slice, yuv_planar16_process_native,
};
use crate::{
  PixelSink,
  resample::{
    AveragingDomain, InsertionContext, InsertionPoint, ResamplePlan, select_insertion_point,
  },
  row::*,
  source::*,
};

/// The high-bit 4:4:4 planar formats (`Yuv444p9` … `Yuv444p16`) ship the
/// non-4:2:0 native planar fast tier ([`yuv_planar16_process_native`]), so
/// each is statically eligible to splice an [`AveragingDomain::Encoded`] area
/// downscale at the native codes.
const YUV444P_HIGH_BIT_NATIVE_ELIGIBLE: bool = true;

// ---- Yuv444p9 impl -----------------------------------------------------

impl<'a, R, const BE: bool> MixedSinker<'a, Yuv444p9<BE>, R> {
  /// Attaches a packed **`u16`** RGB output buffer. 9-bit low-packed.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgb_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_rgb_u16(buf)?;
    Ok(self)
  }
  /// In-place variant.
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

  /// Attaches a packed **8-bit** RGBA output buffer. The 9-bit YUV
  /// source is converted to 8-bit RGBA via the same `BITS = 9` Q15
  /// kernel family used by [`Self::with_rgb`]; the fourth byte per
  /// pixel is alpha = `0xFF` (Yuv444p9 has no alpha plane).
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

  /// Attaches a packed **`u16`** RGBA output buffer. 9-bit low-packed
  /// (`(1 << 9) - 1 = 511` max). Length is measured in `u16`
  /// **elements** (`width x height x 4`). Alpha element is `(1 << 9) - 1`.
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
}

impl<R, const BE: bool> Yuv444p9Sink<BE> for MixedSinker<'_, Yuv444p9<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, Yuv444p9<BE>, R> {
  type Input<'r> = Yuv444p9Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    reset_high_bit_yuv_streams(self);
    Ok(())
  }

  fn process(&mut self, row: Yuv444p9Row<'_>) -> Result<(), Self::Error> {
    const BITS: u32 = 9;
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Y9,
        idx,
        w,
        row.y().len(),
      )));
    }
    if row.u().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::UFull9,
        idx,
        w,
        row.u().len(),
      )));
    }
    if row.v().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::VFull9,
        idx,
        w,
        row.v().len(),
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
      hsv,
      rgb_scratch,
      rgb_scratch_u16,
      luma_scratch_u16,
      rgb_stream,
      rgb_stream_u16,
      luma_stream_u16,
      rgb_filter_stream,
      rgb_filter_stream_u16,
      luma_filter_stream_u16,
      resample_outputs,
      plan,
      native,
      native_planar_u16,
      frozen_native_route,
      ..
    } = self;

    // Non-identity plan: feed the shared high-bit 4:4:4 triple-resample
    // tail (u8 color, independent native-u16 color, native Y). The planar
    // decode closures stage source-width rows; chroma is full-width (no
    // upsampling). Yuv444p exposes no `luma_u16` output, so it is `&mut
    // None` and only `luma` (binned native Y `>> (BITS - 8)`) is emitted.
    // The span kind picks the engine: area binning, or the signed-coefficient
    // filter twin (both convert the YUV to RGB with the same closures and
    // resample in RGB space, so filter colour equals the RGB filter of the
    // converted pixels and matches area up to the kernel). The filter tail
    // clamps every sub-16-bit colour sample AND the native Y to
    // `(1 << BITS) - 1` before publishing.
    if let Some(plan) = plan.as_ref() {
      let matrix = row.matrix();
      let full_range = row.full_range();
      let (y, u, v) = (row.y(), row.u(), row.v());
      if plan.kind().is_filter() {
        return packed_yuv444_triple_filter_resample::<BITS>(
          rgb_filter_stream,
          rgb_filter_stream_u16,
          luma_filter_stream_u16,
          resample_outputs,
          rgb,
          rgba,
          rgb_u16,
          rgba_u16,
          luma,
          &mut None,
          hsv,
          rgb_scratch,
          rgb_scratch_u16,
          luma_scratch_u16,
          w,
          plan,
          idx,
          use_simd,
          matrix,
          full_range,
          |scratch| {
            yuv444p9_to_rgb_row_endian(y, u, v, scratch, w, matrix, full_range, use_simd, BE)
          },
          |scratch| {
            yuv444p9_to_rgb_u16_row_endian(y, u, v, scratch, w, matrix, full_range, use_simd, BE)
          },
          |scratch| deinterleave_y_high_bit_masked::<BITS, BE>(y, scratch, w),
        );
      }
      // Native / row-stage route split — see the high-bit 4:2:0 Yuv420p impl
      // for the CHECK-before / SET-after `frozen_native_route` contract.
      let need_output = luma.is_some()
        || rgb.is_some()
        || rgba.is_some()
        || hsv.is_some()
        || rgb_u16.is_some()
        || rgba_u16.is_some();
      if need_output
        && let Some(frozen) = *frozen_native_route
        && frozen != *native
      {
        return Err(MixedSinkerError::NativeRouteChanged(
          NativeRouteChanged::new(idx),
        ));
      }
      // RFC #238 splice-stage selection — see the Yuv420p impl for the
      // selector contract; reproduces the former `if *native` boolean
      // bit-for-bit (a filter plan already returned above, so `area_plan` is
      // always true here).
      let insertion = select_insertion_point(
        AveragingDomain::Encoded,
        InsertionContext {
          native_eligible: YUV444P_HIGH_BIT_NATIVE_ELIGIBLE,
          with_native: *native,
          area_plan: true,
        },
      );
      match insertion {
        InsertionPoint::NativeCodes => {
          // 4:4:4: chroma `w x h` — identical to Y; a chroma row per Y row
          // (`chroma_vsub = 1`, `chroma_w = w`), chroma plan equals the luma
          // plan.
          yuv_planar16_process_native::<BITS, BE>(
            plan,
            native_planar_u16,
            resample_outputs,
            rgb,
            rgba,
            rgb_u16,
            rgba_u16,
            luma,
            // The high-bit planar 4:4:4 family exposes no `luma_u16` output.
            &mut None,
            hsv,
            rgb_scratch,
            rgb_scratch_u16,
            y,
            u,
            v,
            matrix,
            full_range,
            idx,
            w,
            h,
            1,
            w,
            || ResamplePlan::area(w, h, plan.out_w(), plan.out_h()),
            use_simd,
          )?;
          if frozen_native_route.is_none() && need_output {
            *frozen_native_route = Some(true);
          }
          return Ok(());
        }
        InsertionPoint::EncodedOutput => {
          packed_yuv444_triple_resample::<BITS>(
            rgb_stream,
            rgb_stream_u16,
            luma_stream_u16,
            resample_outputs,
            rgb,
            rgba,
            rgb_u16,
            rgba_u16,
            luma,
            &mut None,
            hsv,
            rgb_scratch,
            rgb_scratch_u16,
            luma_scratch_u16,
            w,
            plan,
            idx,
            use_simd,
            matrix,
            full_range,
            |scratch| {
              yuv444p9_to_rgb_row_endian(y, u, v, scratch, w, matrix, full_range, use_simd, BE)
            },
            |scratch| {
              yuv444p9_to_rgb_u16_row_endian(y, u, v, scratch, w, matrix, full_range, use_simd, BE)
            },
            |scratch| deinterleave_y_high_bit_masked::<BITS, BE>(y, scratch, w),
          )?;
          if frozen_native_route.is_none() && need_output {
            *frozen_native_route = Some(false);
          }
          return Ok(());
        }
        // The encoded domain only resolves to the native-codes or
        // encoded-output splice; the linear-light splice is reached via the
        // sink's Linear averaging domain, dispatched before this match.
        InsertionPoint::LinearLight => {
          unreachable!("encoded domain never selects the linear-light splice")
        }
      }
    }

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // Resolve the output set up front so the atomicity preflight below runs
    // before any output row is written.
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();

    // Atomicity preflight (#308, cf. the crate's #180 resample fix and the
    // high-bit 4:2:0 p0xx / yuv420p siblings): reserve the only growable row
    // scratch this identity row can touch — the u8 RGB row buffer — BEFORE any
    // output row is written (the luma plane below, then the u16 RGB / RGBA
    // fan-out), so an allocator refusal returns a typed `AllocationFailed` and
    // leaves the output frame untouched rather than partially mutated. At 4:4:4
    // the chroma is full-resolution (no upsampling), but the output staging
    // matches 4:2:0: the u16 RGB / RGBA outputs write straight into their caller
    // buffers (the rgb_u16 plane itself stages the rgba_u16 expand) and never
    // grow a scratch, and these formats expose no luma_u16. The allocating
    // (rgb=None) arm of `rgb_row_buf_or_scratch` is reached exactly when a colour
    // decode needs an RGB row but no caller RGB buffer is borrowable —
    // `want_hsv && want_rgba && !want_rgb`. The later decode reuses the
    // already-sized buffer, so the default path is byte-identical; only the
    // failure-path ordering changes.
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

    if let Some(luma) = luma.as_deref_mut() {
      let dst = &mut luma[one_plane_start..one_plane_end];
      for (d, &s) in dst.iter_mut().zip(row.y().iter()) {
        // Normalize BE-encoded wire bytes to host-native before the
        // luma downshift — see Yuv420p9 luma path for rationale.
        let logical = if BE { u16::from_be(s) } else { u16::from_le(s) };
        *d = (logical >> (BITS - 8)) as u8;
      }
    }

    // ===== u16 RGB / RGBA path (Strategy A) =====
    let want_rgb_u16 = rgb_u16.is_some();
    let want_rgba_u16 = rgba_u16.is_some();

    if want_rgba_u16 && !want_rgb_u16 {
      let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row =
        rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
      yuv444p9_to_rgba_u16_row_endian(
        row.y(),
        row.u(),
        row.v(),
        rgba_u16_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
        BE,
      );
    } else if want_rgb_u16 {
      let rgb_u16_buf = rgb_u16.as_deref_mut().unwrap();
      let rgb_plane_end =
        one_plane_end
          .checked_mul(3)
          .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
            w, h, 3,
          )))?;
      let rgb_plane_start = one_plane_start * 3;
      let rgb_u16_row = &mut rgb_u16_buf[rgb_plane_start..rgb_plane_end];
      yuv444p9_to_rgb_u16_row_endian(
        row.y(),
        row.u(),
        row.v(),
        rgb_u16_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
        BE,
      );
      if want_rgba_u16 {
        let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
        let rgba_u16_row =
          rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
        expand_rgb_u16_to_rgba_u16_row::<BITS>(rgb_u16_row, rgba_u16_row, w);
      }
    }

    // ===== u8 RGB / RGBA / HSV path (Strategy A) =====
    // HSV-without-RGB-or-RGBA goes through the direct `yuv444p9_to_hsv_row_endian`
    // kernel (no source-width RGB scratch — the SIMD path stages a fixed
    // 8-bit RGB chunk internally). RGB or RGBA also attached keeps the
    // convert-once-then-derive path alive via `need_rgb_kernel`.
    let want_hsv_direct = want_hsv && rgb.is_none() && !want_rgba;
    let need_rgb_kernel = rgb.is_some() || (want_hsv && want_rgba);

    if want_hsv_direct {
      let hsv = hsv.as_mut().expect("want_hsv_direct implies hsv attached");
      let (h, s, v) = hsv.hsv();
      yuv444p9_to_hsv_row_endian(
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
        BE,
      );
      return Ok(());
    }

    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      yuv444p9_to_rgba_row_endian(
        row.y(),
        row.u(),
        row.v(),
        rgba_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
        BE,
      );
      return Ok(());
    }

    if !need_rgb_kernel {
      return Ok(());
    }

    let rgb_row = rgb_row_buf_or_scratch(
      rgb.as_deref_mut(),
      rgb_scratch,
      one_plane_start,
      one_plane_end,
      w,
      h,
    )?;

    yuv444p9_to_rgb_row_endian(
      row.y(),
      row.u(),
      row.v(),
      rgb_row,
      w,
      row.matrix(),
      row.full_range(),
      use_simd,
      BE,
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

    if let Some(buf) = rgba.as_deref_mut() {
      let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
    }

    Ok(())
  }
}

// ---- Yuv444p10 / 12 / 14 impl ------------------------------------------

impl<'a, R, const BE: bool> MixedSinker<'a, Yuv444p10<BE>, R> {
  /// Attaches a packed **`u16`** RGB output buffer. 10-bit low-packed.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgb_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_rgb_u16(buf)?;
    Ok(self)
  }
  /// In-place variant.
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

  /// Attaches a packed **8-bit** RGBA output buffer. The 10-bit YUV
  /// source is converted to 8-bit RGBA via the same `BITS = 10` Q15
  /// kernel family used by [`Self::with_rgb`]; alpha = `0xFF`.
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

  /// Attaches a packed **`u16`** RGBA output buffer. 10-bit low-packed
  /// (`[0, 1023]`); alpha element is `1023`.
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
}

impl<R, const BE: bool> Yuv444p10Sink<BE> for MixedSinker<'_, Yuv444p10<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, Yuv444p10<BE>, R> {
  type Input<'r> = Yuv444p10Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    reset_high_bit_yuv_streams(self);
    Ok(())
  }

  fn process(&mut self, row: Yuv444p10Row<'_>) -> Result<(), Self::Error> {
    const BITS: u32 = 10;
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Y10,
        idx,
        w,
        row.y().len(),
      )));
    }
    if row.u().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::UFull10,
        idx,
        w,
        row.u().len(),
      )));
    }
    if row.v().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::VFull10,
        idx,
        w,
        row.v().len(),
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
      hsv,
      rgb_scratch,
      rgb_scratch_u16,
      luma_scratch_u16,
      rgb_stream,
      rgb_stream_u16,
      luma_stream_u16,
      rgb_filter_stream,
      rgb_filter_stream_u16,
      luma_filter_stream_u16,
      resample_outputs,
      plan,
      native,
      native_planar_u16,
      frozen_native_route,
      ..
    } = self;

    // Non-identity plan: feed the shared high-bit 4:4:4 triple-resample
    // tail (u8 color, independent native-u16 color, native Y). The planar
    // decode closures stage source-width rows; chroma is full-width (no
    // upsampling). Yuv444p exposes no `luma_u16` output, so it is `&mut
    // None` and only `luma` (binned native Y `>> (BITS - 8)`) is emitted.
    // The span kind picks the engine: area binning, or the signed-coefficient
    // filter twin (both convert the YUV to RGB with the same closures and
    // resample in RGB space, so filter colour equals the RGB filter of the
    // converted pixels and matches area up to the kernel). The filter tail
    // clamps every sub-16-bit colour sample AND the native Y to
    // `(1 << BITS) - 1` before publishing.
    if let Some(plan) = plan.as_ref() {
      let matrix = row.matrix();
      let full_range = row.full_range();
      let (y, u, v) = (row.y(), row.u(), row.v());
      if plan.kind().is_filter() {
        return packed_yuv444_triple_filter_resample::<BITS>(
          rgb_filter_stream,
          rgb_filter_stream_u16,
          luma_filter_stream_u16,
          resample_outputs,
          rgb,
          rgba,
          rgb_u16,
          rgba_u16,
          luma,
          &mut None,
          hsv,
          rgb_scratch,
          rgb_scratch_u16,
          luma_scratch_u16,
          w,
          plan,
          idx,
          use_simd,
          matrix,
          full_range,
          |scratch| {
            yuv444p10_to_rgb_row_endian(y, u, v, scratch, w, matrix, full_range, use_simd, BE)
          },
          |scratch| {
            yuv444p10_to_rgb_u16_row_endian(y, u, v, scratch, w, matrix, full_range, use_simd, BE)
          },
          |scratch| deinterleave_y_high_bit_masked::<BITS, BE>(y, scratch, w),
        );
      }
      // Native / row-stage route split — see the high-bit 4:2:0 Yuv420p impl
      // for the CHECK-before / SET-after `frozen_native_route` contract.
      let need_output = luma.is_some()
        || rgb.is_some()
        || rgba.is_some()
        || hsv.is_some()
        || rgb_u16.is_some()
        || rgba_u16.is_some();
      if need_output
        && let Some(frozen) = *frozen_native_route
        && frozen != *native
      {
        return Err(MixedSinkerError::NativeRouteChanged(
          NativeRouteChanged::new(idx),
        ));
      }
      // RFC #238 splice-stage selection — see the Yuv420p impl for the
      // selector contract; reproduces the former `if *native` boolean
      // bit-for-bit (a filter plan already returned above, so `area_plan` is
      // always true here).
      let insertion = select_insertion_point(
        AveragingDomain::Encoded,
        InsertionContext {
          native_eligible: YUV444P_HIGH_BIT_NATIVE_ELIGIBLE,
          with_native: *native,
          area_plan: true,
        },
      );
      match insertion {
        InsertionPoint::NativeCodes => {
          // 4:4:4: chroma `w x h` — identical to Y; a chroma row per Y row
          // (`chroma_vsub = 1`, `chroma_w = w`), chroma plan equals the luma
          // plan.
          yuv_planar16_process_native::<BITS, BE>(
            plan,
            native_planar_u16,
            resample_outputs,
            rgb,
            rgba,
            rgb_u16,
            rgba_u16,
            luma,
            // The high-bit planar 4:4:4 family exposes no `luma_u16` output.
            &mut None,
            hsv,
            rgb_scratch,
            rgb_scratch_u16,
            y,
            u,
            v,
            matrix,
            full_range,
            idx,
            w,
            h,
            1,
            w,
            || ResamplePlan::area(w, h, plan.out_w(), plan.out_h()),
            use_simd,
          )?;
          if frozen_native_route.is_none() && need_output {
            *frozen_native_route = Some(true);
          }
          return Ok(());
        }
        InsertionPoint::EncodedOutput => {
          packed_yuv444_triple_resample::<BITS>(
            rgb_stream,
            rgb_stream_u16,
            luma_stream_u16,
            resample_outputs,
            rgb,
            rgba,
            rgb_u16,
            rgba_u16,
            luma,
            &mut None,
            hsv,
            rgb_scratch,
            rgb_scratch_u16,
            luma_scratch_u16,
            w,
            plan,
            idx,
            use_simd,
            matrix,
            full_range,
            |scratch| {
              yuv444p10_to_rgb_row_endian(y, u, v, scratch, w, matrix, full_range, use_simd, BE)
            },
            |scratch| {
              yuv444p10_to_rgb_u16_row_endian(y, u, v, scratch, w, matrix, full_range, use_simd, BE)
            },
            |scratch| deinterleave_y_high_bit_masked::<BITS, BE>(y, scratch, w),
          )?;
          if frozen_native_route.is_none() && need_output {
            *frozen_native_route = Some(false);
          }
          return Ok(());
        }
        // The encoded domain only resolves to the native-codes or
        // encoded-output splice; the linear-light splice is reached via the
        // sink's Linear averaging domain, dispatched before this match.
        InsertionPoint::LinearLight => {
          unreachable!("encoded domain never selects the linear-light splice")
        }
      }
    }

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // Resolve the output set up front so the atomicity preflight below runs
    // before any output row is written.
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();

    // Atomicity preflight (#308, cf. the crate's #180 resample fix and the
    // high-bit 4:2:0 p0xx / yuv420p siblings): reserve the only growable row
    // scratch this identity row can touch — the u8 RGB row buffer — BEFORE any
    // output row is written (the luma plane below, then the u16 RGB / RGBA
    // fan-out), so an allocator refusal returns a typed `AllocationFailed` and
    // leaves the output frame untouched rather than partially mutated. At 4:4:4
    // the chroma is full-resolution (no upsampling), but the output staging
    // matches 4:2:0: the u16 RGB / RGBA outputs write straight into their caller
    // buffers (the rgb_u16 plane itself stages the rgba_u16 expand) and never
    // grow a scratch, and these formats expose no luma_u16. The allocating
    // (rgb=None) arm of `rgb_row_buf_or_scratch` is reached exactly when a colour
    // decode needs an RGB row but no caller RGB buffer is borrowable —
    // `want_hsv && want_rgba && !want_rgb`. The later decode reuses the
    // already-sized buffer, so the default path is byte-identical; only the
    // failure-path ordering changes.
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

    if let Some(luma) = luma.as_deref_mut() {
      let dst = &mut luma[one_plane_start..one_plane_end];
      for (d, &s) in dst.iter_mut().zip(row.y().iter()) {
        // Normalize BE-encoded wire bytes to host-native before the
        // luma downshift — see Yuv420p9 luma path for rationale.
        let logical = if BE { u16::from_be(s) } else { u16::from_le(s) };
        *d = (logical >> (BITS - 8)) as u8;
      }
    }

    // ===== u16 RGB / RGBA path (Strategy A) =====
    let want_rgb_u16 = rgb_u16.is_some();
    let want_rgba_u16 = rgba_u16.is_some();

    if want_rgba_u16 && !want_rgb_u16 {
      let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row =
        rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
      yuv444p10_to_rgba_u16_row_endian(
        row.y(),
        row.u(),
        row.v(),
        rgba_u16_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
        BE,
      );
    } else if want_rgb_u16 {
      let rgb_u16_buf = rgb_u16.as_deref_mut().unwrap();
      let rgb_plane_end =
        one_plane_end
          .checked_mul(3)
          .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
            w, h, 3,
          )))?;
      let rgb_plane_start = one_plane_start * 3;
      let rgb_u16_row = &mut rgb_u16_buf[rgb_plane_start..rgb_plane_end];
      yuv444p10_to_rgb_u16_row_endian(
        row.y(),
        row.u(),
        row.v(),
        rgb_u16_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
        BE,
      );
      if want_rgba_u16 {
        let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
        let rgba_u16_row =
          rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
        expand_rgb_u16_to_rgba_u16_row::<BITS>(rgb_u16_row, rgba_u16_row, w);
      }
    }

    // ===== u8 RGB / RGBA / HSV path (Strategy A) =====
    // HSV-without-RGB-or-RGBA goes through the direct `yuv444p10_to_hsv_row_endian`
    // kernel (no source-width RGB scratch — the SIMD path stages a fixed
    // 8-bit RGB chunk internally). RGB or RGBA also attached keeps the
    // convert-once-then-derive path alive via `need_rgb_kernel`.
    let want_hsv_direct = want_hsv && rgb.is_none() && !want_rgba;
    let need_rgb_kernel = rgb.is_some() || (want_hsv && want_rgba);

    if want_hsv_direct {
      let hsv = hsv.as_mut().expect("want_hsv_direct implies hsv attached");
      let (h, s, v) = hsv.hsv();
      yuv444p10_to_hsv_row_endian(
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
        BE,
      );
      return Ok(());
    }

    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      yuv444p10_to_rgba_row_endian(
        row.y(),
        row.u(),
        row.v(),
        rgba_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
        BE,
      );
      return Ok(());
    }

    if !need_rgb_kernel {
      return Ok(());
    }

    let rgb_row = rgb_row_buf_or_scratch(
      rgb.as_deref_mut(),
      rgb_scratch,
      one_plane_start,
      one_plane_end,
      w,
      h,
    )?;

    yuv444p10_to_rgb_row_endian(
      row.y(),
      row.u(),
      row.v(),
      rgb_row,
      w,
      row.matrix(),
      row.full_range(),
      use_simd,
      BE,
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

    if let Some(buf) = rgba.as_deref_mut() {
      let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
    }

    Ok(())
  }
}

impl<'a, R, const BE: bool> MixedSinker<'a, Yuv444p12<BE>, R> {
  /// Attaches a packed **`u16`** RGB output buffer. 12-bit low-packed.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgb_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_rgb_u16(buf)?;
    Ok(self)
  }
  /// In-place variant.
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

  /// Attaches a packed **8-bit** RGBA output buffer. The 12-bit YUV
  /// source is converted to 8-bit RGBA via the same `BITS = 12` Q15
  /// kernel family used by [`Self::with_rgb`]; alpha = `0xFF`.
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

  /// Attaches a packed **`u16`** RGBA output buffer. 12-bit low-packed
  /// (`[0, 4095]`); alpha element is `4095`.
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
}

impl<R, const BE: bool> Yuv444p12Sink<BE> for MixedSinker<'_, Yuv444p12<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, Yuv444p12<BE>, R> {
  type Input<'r> = Yuv444p12Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    reset_high_bit_yuv_streams(self);
    Ok(())
  }

  fn process(&mut self, row: Yuv444p12Row<'_>) -> Result<(), Self::Error> {
    const BITS: u32 = 12;
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;
    // The non-affine decodes (#303) select their variant from the sink's
    // signalled colour spec (the YUV row carries only matrix + range);
    // captured before the `self` destructure below. ICtCp selects PQ vs HLG
    // from the transfer; constant-luminance (CL) selects the BT.2020 OETF
    // bit-depth from the transfer and gates on BT.2020 primaries; IPT-C2 gates
    // on a PQ transfer (its only defined variant). The defaults (`Unspecified`)
    // route a tagged source back to the affine fallback.
    let transfer = self.transfer;
    let primaries = self.primaries;
    // Whether this row decodes as ICtCp (BT.2100, #303): the `Ictcp` matrix
    // with a resolvable PQ/HLG transfer.
    let ictcp_active = matches!(row.matrix(), crate::ColorMatrix::Ictcp)
      && crate::row::scalar::ictcp::IctcpTransfer::for_transfer(transfer).is_some();
    // Whether this row decodes as constant-luminance `YcCbcCrc` (BT.2020 CL,
    // H.273 MatrixCoefficients = 13, #303): the `ChromaDerivedCl` matrix with
    // BT.2020 primaries. Mutually exclusive with `ictcp_active` (distinct
    // matrices). Both drive the non-affine HSV routing and the RGB-scratch
    // atomicity preflight below; `false` leaves every affine fast path
    // byte-identical.
    let cl_active = matches!(row.matrix(), crate::ColorMatrix::ChromaDerivedCl)
      && crate::row::scalar::cl::ClSystem::resolve(primaries, transfer).is_some();
    // Whether this row decodes as IPT-C2 (H.273 MatrixCoefficients = 15, the
    // Dolby Vision Profile 5 base colour space, #303): the `IptC2` matrix with
    // a resolvable PQ transfer. Mutually exclusive with `ictcp_active` /
    // `cl_active` (distinct matrices).
    let iptc2_active = matches!(row.matrix(), crate::ColorMatrix::IptC2)
      && crate::row::scalar::iptc2::IptC2Transfer::for_transfer(transfer).is_some();
    // Whether this row decodes as SMPTE ST 2085 (H.273 MatrixCoefficients = 11,
    // "Y'D'zD'x", the PQ-only non-affine X'Y'Z' colour-difference model, #303):
    // the `Smpte2085` matrix with a resolvable PQ transfer. Mutually exclusive
    // with `ictcp_active` / `cl_active` / `iptc2_active` (distinct matrices).
    let smpte2085_active = matches!(row.matrix(), crate::ColorMatrix::Smpte2085)
      && crate::row::scalar::smpte2085::Smpte2085Transfer::for_transfer(transfer).is_some();
    // Any of the non-affine decodes needs the convert-once-then-derive path
    // (its RGB cannot be reconstructed from a single Q15 matrix), so they share
    // the HSV-routing and atomicity-preflight predicates below.
    let non_affine_active = ictcp_active || cl_active || iptc2_active || smpte2085_active;

    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Y12,
        idx,
        w,
        row.y().len(),
      )));
    }
    if row.u().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::UFull12,
        idx,
        w,
        row.u().len(),
      )));
    }
    if row.v().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::VFull12,
        idx,
        w,
        row.v().len(),
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
      hsv,
      rgb_scratch,
      rgb_scratch_u16,
      luma_scratch_u16,
      rgb_stream,
      rgb_stream_u16,
      luma_stream_u16,
      rgb_filter_stream,
      rgb_filter_stream_u16,
      luma_filter_stream_u16,
      resample_outputs,
      plan,
      native,
      native_planar_u16,
      frozen_native_route,
      ..
    } = self;

    // Non-identity plan: feed the shared high-bit 4:4:4 triple-resample
    // tail (u8 color, independent native-u16 color, native Y). The planar
    // decode closures stage source-width rows; chroma is full-width (no
    // upsampling). Yuv444p exposes no `luma_u16` output, so it is `&mut
    // None` and only `luma` (binned native Y `>> (BITS - 8)`) is emitted.
    // The span kind picks the engine (area bin or signed-coefficient filter
    // twin) — see the Yuv444p10 impl for the full rationale; the filter tail
    // clamps every sub-16-bit colour sample AND the native Y to
    // `(1 << BITS) - 1`.
    if let Some(plan) = plan.as_ref() {
      // Non-affine matrices (constant-luminance `ChromaDerivedCl`, BT.2100
      // `Ictcp`, IPT-C2 `IptC2`) are decoded only on the identity path: the
      // resample tail below — the filter twin, the native-code binner, and the
      // encoded average — all stage colour through the *affine*
      // `yuv444p12_to_rgb*` kernels (which see only matrix + range, not the
      // primaries / transfer the non-affine decode needs), so resampling such a
      // source would emit silent, route-dependent wrong colour. Reject it with
      // a typed error instead (#303). Only the *resolved* non-affine decode is
      // rejected; an unresolved tag (`non_affine_active == false`) already
      // falls through to the affine path on the identity route, so it resamples
      // affinely, unchanged.
      if non_affine_active {
        let name = if cl_active {
          "ChromaDerivedCl"
        } else if iptc2_active {
          "IptC2"
        } else if smpte2085_active {
          "Smpte2085"
        } else {
          "Ictcp"
        };
        return Err(MixedSinkerError::UnsupportedMatrixResample(
          UnsupportedMatrixResample::new(idx, name),
        ));
      }
      let matrix = row.matrix();
      let full_range = row.full_range();
      let (y, u, v) = (row.y(), row.u(), row.v());
      if plan.kind().is_filter() {
        return packed_yuv444_triple_filter_resample::<BITS>(
          rgb_filter_stream,
          rgb_filter_stream_u16,
          luma_filter_stream_u16,
          resample_outputs,
          rgb,
          rgba,
          rgb_u16,
          rgba_u16,
          luma,
          &mut None,
          hsv,
          rgb_scratch,
          rgb_scratch_u16,
          luma_scratch_u16,
          w,
          plan,
          idx,
          use_simd,
          matrix,
          full_range,
          |scratch| {
            yuv444p12_to_rgb_row_endian(y, u, v, scratch, w, matrix, full_range, use_simd, BE)
          },
          |scratch| {
            yuv444p12_to_rgb_u16_row_endian(y, u, v, scratch, w, matrix, full_range, use_simd, BE)
          },
          |scratch| deinterleave_y_high_bit_masked::<BITS, BE>(y, scratch, w),
        );
      }
      // Native / row-stage route split — see the high-bit 4:2:0 Yuv420p impl
      // for the CHECK-before / SET-after `frozen_native_route` contract.
      let need_output = luma.is_some()
        || rgb.is_some()
        || rgba.is_some()
        || hsv.is_some()
        || rgb_u16.is_some()
        || rgba_u16.is_some();
      if need_output
        && let Some(frozen) = *frozen_native_route
        && frozen != *native
      {
        return Err(MixedSinkerError::NativeRouteChanged(
          NativeRouteChanged::new(idx),
        ));
      }
      // RFC #238 splice-stage selection — see the Yuv420p impl for the
      // selector contract; reproduces the former `if *native` boolean
      // bit-for-bit (a filter plan already returned above, so `area_plan` is
      // always true here).
      let insertion = select_insertion_point(
        AveragingDomain::Encoded,
        InsertionContext {
          native_eligible: YUV444P_HIGH_BIT_NATIVE_ELIGIBLE,
          with_native: *native,
          area_plan: true,
        },
      );
      match insertion {
        InsertionPoint::NativeCodes => {
          // 4:4:4: chroma `w x h` — identical to Y; a chroma row per Y row
          // (`chroma_vsub = 1`, `chroma_w = w`), chroma plan equals the luma
          // plan.
          yuv_planar16_process_native::<BITS, BE>(
            plan,
            native_planar_u16,
            resample_outputs,
            rgb,
            rgba,
            rgb_u16,
            rgba_u16,
            luma,
            // The high-bit planar 4:4:4 family exposes no `luma_u16` output.
            &mut None,
            hsv,
            rgb_scratch,
            rgb_scratch_u16,
            y,
            u,
            v,
            matrix,
            full_range,
            idx,
            w,
            h,
            1,
            w,
            || ResamplePlan::area(w, h, plan.out_w(), plan.out_h()),
            use_simd,
          )?;
          if frozen_native_route.is_none() && need_output {
            *frozen_native_route = Some(true);
          }
          return Ok(());
        }
        InsertionPoint::EncodedOutput => {
          packed_yuv444_triple_resample::<BITS>(
            rgb_stream,
            rgb_stream_u16,
            luma_stream_u16,
            resample_outputs,
            rgb,
            rgba,
            rgb_u16,
            rgba_u16,
            luma,
            &mut None,
            hsv,
            rgb_scratch,
            rgb_scratch_u16,
            luma_scratch_u16,
            w,
            plan,
            idx,
            use_simd,
            matrix,
            full_range,
            |scratch| {
              yuv444p12_to_rgb_row_endian(y, u, v, scratch, w, matrix, full_range, use_simd, BE)
            },
            |scratch| {
              yuv444p12_to_rgb_u16_row_endian(y, u, v, scratch, w, matrix, full_range, use_simd, BE)
            },
            |scratch| deinterleave_y_high_bit_masked::<BITS, BE>(y, scratch, w),
          )?;
          if frozen_native_route.is_none() && need_output {
            *frozen_native_route = Some(false);
          }
          return Ok(());
        }
        // The encoded domain only resolves to the native-codes or
        // encoded-output splice; the linear-light splice is reached via the
        // sink's Linear averaging domain, dispatched before this match.
        InsertionPoint::LinearLight => {
          unreachable!("encoded domain never selects the linear-light splice")
        }
      }
    }

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // Resolve the output set up front so the atomicity preflight below runs
    // before any output row is written.
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();

    // Atomicity preflight (#308, cf. the crate's #180 resample fix and the
    // high-bit 4:2:0 p0xx / yuv420p siblings): reserve the only growable row
    // scratch this identity row can touch — the u8 RGB row buffer — BEFORE any
    // output row is written (the luma plane below, then the u16 RGB / RGBA
    // fan-out), so an allocator refusal returns a typed `AllocationFailed` and
    // leaves the output frame untouched rather than partially mutated. At 4:4:4
    // the chroma is full-resolution (no upsampling), but the output staging
    // matches 4:2:0: the u16 RGB / RGBA outputs write straight into their caller
    // buffers (the rgb_u16 plane itself stages the rgba_u16 expand) and never
    // grow a scratch, and these formats expose no luma_u16. The allocating
    // (rgb=None) arm of `rgb_row_buf_or_scratch` is reached exactly when a colour
    // decode needs an RGB row but no caller RGB buffer is borrowable —
    // `want_hsv && !want_rgb && (want_rgba || non_affine_active)`: HSV staged from
    // an RGB row, either the affine `want_rgba` case or the ICtCp / CL HSV path
    // (which derives HSV from the non-affine RGB even for HSV-only / HSV+rgb_u16).
    // The later decode reuses the already-sized buffer, so the default path is
    // byte-identical; only the failure-path ordering changes.
    if want_hsv && !want_rgb && (want_rgba || non_affine_active) {
      rgb_row_buf_or_scratch(
        rgb.as_deref_mut(),
        rgb_scratch,
        one_plane_start,
        one_plane_end,
        w,
        h,
      )?;
    }

    if let Some(luma) = luma.as_deref_mut() {
      let dst = &mut luma[one_plane_start..one_plane_end];
      for (d, &s) in dst.iter_mut().zip(row.y().iter()) {
        // Normalize BE-encoded wire bytes to host-native before the
        // luma downshift — see Yuv420p9 luma path for rationale.
        let logical = if BE { u16::from_be(s) } else { u16::from_le(s) };
        *d = (logical >> (BITS - 8)) as u8;
      }
    }

    // ===== u16 RGB / RGBA path (Strategy A) =====
    let want_rgb_u16 = rgb_u16.is_some();
    let want_rgba_u16 = rgba_u16.is_some();

    if want_rgba_u16 && !want_rgb_u16 {
      let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row =
        rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
      yuv444p12_to_rgba_u16_row_smpte2085_endian(
        row.y(),
        row.u(),
        row.v(),
        rgba_u16_row,
        w,
        row.matrix(),
        primaries,
        row.full_range(),
        transfer,
        use_simd,
        BE,
      );
    } else if want_rgb_u16 {
      let rgb_u16_buf = rgb_u16.as_deref_mut().unwrap();
      let rgb_plane_end =
        one_plane_end
          .checked_mul(3)
          .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
            w, h, 3,
          )))?;
      let rgb_plane_start = one_plane_start * 3;
      let rgb_u16_row = &mut rgb_u16_buf[rgb_plane_start..rgb_plane_end];
      yuv444p12_to_rgb_u16_row_smpte2085_endian(
        row.y(),
        row.u(),
        row.v(),
        rgb_u16_row,
        w,
        row.matrix(),
        primaries,
        row.full_range(),
        transfer,
        use_simd,
        BE,
      );
      if want_rgba_u16 {
        let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
        let rgba_u16_row =
          rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
        expand_rgb_u16_to_rgba_u16_row::<BITS>(rgb_u16_row, rgba_u16_row, w);
      }
    }

    // ===== u8 RGB / RGBA / HSV path (Strategy A) =====
    // HSV-without-RGB-or-RGBA goes through the direct `yuv444p12_to_hsv_row_endian`
    // kernel (no source-width RGB scratch — the SIMD path stages a fixed
    // 8-bit RGB chunk internally) for NON-ICtCp matrices. RGB or RGBA also
    // attached — or ICtCp, which must derive HSV from its non-affine RGB —
    // keeps the convert-once-then-derive path alive via `need_rgb_kernel`.
    let want_hsv_direct = want_hsv && rgb.is_none() && !want_rgba && !non_affine_active;
    let need_rgb_kernel = rgb.is_some() || (want_hsv && (want_rgba || non_affine_active));

    if want_hsv_direct {
      let hsv = hsv.as_mut().expect("want_hsv_direct implies hsv attached");
      let (h, s, v) = hsv.hsv();
      yuv444p12_to_hsv_row_endian(
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
        BE,
      );
      return Ok(());
    }

    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      yuv444p12_to_rgba_row_smpte2085_endian(
        row.y(),
        row.u(),
        row.v(),
        rgba_row,
        w,
        row.matrix(),
        primaries,
        row.full_range(),
        transfer,
        use_simd,
        BE,
      );
      return Ok(());
    }

    if !need_rgb_kernel {
      return Ok(());
    }

    let rgb_row = rgb_row_buf_or_scratch(
      rgb.as_deref_mut(),
      rgb_scratch,
      one_plane_start,
      one_plane_end,
      w,
      h,
    )?;

    yuv444p12_to_rgb_row_smpte2085_endian(
      row.y(),
      row.u(),
      row.v(),
      rgb_row,
      w,
      row.matrix(),
      primaries,
      row.full_range(),
      transfer,
      use_simd,
      BE,
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

    if let Some(buf) = rgba.as_deref_mut() {
      let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
    }

    Ok(())
  }
}

impl<'a, R, const BE: bool> MixedSinker<'a, Yuv444p14<BE>, R> {
  /// Attaches a packed **`u16`** RGB output buffer. 14-bit low-packed.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgb_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_rgb_u16(buf)?;
    Ok(self)
  }
  /// In-place variant.
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

  /// Attaches a packed **8-bit** RGBA output buffer. The 14-bit YUV
  /// source is converted to 8-bit RGBA via the same `BITS = 14` Q15
  /// kernel family used by [`Self::with_rgb`]; alpha = `0xFF`.
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

  /// Attaches a packed **`u16`** RGBA output buffer. 14-bit low-packed
  /// (`[0, 16383]`); alpha element is `16383`.
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
}

impl<R, const BE: bool> Yuv444p14Sink<BE> for MixedSinker<'_, Yuv444p14<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, Yuv444p14<BE>, R> {
  type Input<'r> = Yuv444p14Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    reset_high_bit_yuv_streams(self);
    Ok(())
  }

  fn process(&mut self, row: Yuv444p14Row<'_>) -> Result<(), Self::Error> {
    const BITS: u32 = 14;
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Y14,
        idx,
        w,
        row.y().len(),
      )));
    }
    if row.u().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::UFull14,
        idx,
        w,
        row.u().len(),
      )));
    }
    if row.v().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::VFull14,
        idx,
        w,
        row.v().len(),
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
      hsv,
      rgb_scratch,
      rgb_scratch_u16,
      luma_scratch_u16,
      rgb_stream,
      rgb_stream_u16,
      luma_stream_u16,
      rgb_filter_stream,
      rgb_filter_stream_u16,
      luma_filter_stream_u16,
      resample_outputs,
      plan,
      native,
      native_planar_u16,
      frozen_native_route,
      ..
    } = self;

    // Non-identity plan: feed the shared high-bit 4:4:4 triple-resample
    // tail (u8 color, independent native-u16 color, native Y). The planar
    // decode closures stage source-width rows; chroma is full-width (no
    // upsampling). Yuv444p exposes no `luma_u16` output, so it is `&mut
    // None` and only `luma` (binned native Y `>> (BITS - 8)`) is emitted.
    // The span kind picks the engine (area bin or signed-coefficient filter
    // twin) — see the Yuv444p10 impl for the full rationale; the filter tail
    // clamps every sub-16-bit colour sample AND the native Y to
    // `(1 << BITS) - 1`.
    if let Some(plan) = plan.as_ref() {
      let matrix = row.matrix();
      let full_range = row.full_range();
      let (y, u, v) = (row.y(), row.u(), row.v());
      if plan.kind().is_filter() {
        return packed_yuv444_triple_filter_resample::<BITS>(
          rgb_filter_stream,
          rgb_filter_stream_u16,
          luma_filter_stream_u16,
          resample_outputs,
          rgb,
          rgba,
          rgb_u16,
          rgba_u16,
          luma,
          &mut None,
          hsv,
          rgb_scratch,
          rgb_scratch_u16,
          luma_scratch_u16,
          w,
          plan,
          idx,
          use_simd,
          matrix,
          full_range,
          |scratch| {
            yuv444p14_to_rgb_row_endian(y, u, v, scratch, w, matrix, full_range, use_simd, BE)
          },
          |scratch| {
            yuv444p14_to_rgb_u16_row_endian(y, u, v, scratch, w, matrix, full_range, use_simd, BE)
          },
          |scratch| deinterleave_y_high_bit_masked::<BITS, BE>(y, scratch, w),
        );
      }
      // Native / row-stage route split — see the high-bit 4:2:0 Yuv420p impl
      // for the CHECK-before / SET-after `frozen_native_route` contract.
      let need_output = luma.is_some()
        || rgb.is_some()
        || rgba.is_some()
        || hsv.is_some()
        || rgb_u16.is_some()
        || rgba_u16.is_some();
      if need_output
        && let Some(frozen) = *frozen_native_route
        && frozen != *native
      {
        return Err(MixedSinkerError::NativeRouteChanged(
          NativeRouteChanged::new(idx),
        ));
      }
      // RFC #238 splice-stage selection — see the Yuv420p impl for the
      // selector contract; reproduces the former `if *native` boolean
      // bit-for-bit (a filter plan already returned above, so `area_plan` is
      // always true here).
      let insertion = select_insertion_point(
        AveragingDomain::Encoded,
        InsertionContext {
          native_eligible: YUV444P_HIGH_BIT_NATIVE_ELIGIBLE,
          with_native: *native,
          area_plan: true,
        },
      );
      match insertion {
        InsertionPoint::NativeCodes => {
          // 4:4:4: chroma `w x h` — identical to Y; a chroma row per Y row
          // (`chroma_vsub = 1`, `chroma_w = w`), chroma plan equals the luma
          // plan.
          yuv_planar16_process_native::<BITS, BE>(
            plan,
            native_planar_u16,
            resample_outputs,
            rgb,
            rgba,
            rgb_u16,
            rgba_u16,
            luma,
            // The high-bit planar 4:4:4 family exposes no `luma_u16` output.
            &mut None,
            hsv,
            rgb_scratch,
            rgb_scratch_u16,
            y,
            u,
            v,
            matrix,
            full_range,
            idx,
            w,
            h,
            1,
            w,
            || ResamplePlan::area(w, h, plan.out_w(), plan.out_h()),
            use_simd,
          )?;
          if frozen_native_route.is_none() && need_output {
            *frozen_native_route = Some(true);
          }
          return Ok(());
        }
        InsertionPoint::EncodedOutput => {
          packed_yuv444_triple_resample::<BITS>(
            rgb_stream,
            rgb_stream_u16,
            luma_stream_u16,
            resample_outputs,
            rgb,
            rgba,
            rgb_u16,
            rgba_u16,
            luma,
            &mut None,
            hsv,
            rgb_scratch,
            rgb_scratch_u16,
            luma_scratch_u16,
            w,
            plan,
            idx,
            use_simd,
            matrix,
            full_range,
            |scratch| {
              yuv444p14_to_rgb_row_endian(y, u, v, scratch, w, matrix, full_range, use_simd, BE)
            },
            |scratch| {
              yuv444p14_to_rgb_u16_row_endian(y, u, v, scratch, w, matrix, full_range, use_simd, BE)
            },
            |scratch| deinterleave_y_high_bit_masked::<BITS, BE>(y, scratch, w),
          )?;
          if frozen_native_route.is_none() && need_output {
            *frozen_native_route = Some(false);
          }
          return Ok(());
        }
        // The encoded domain only resolves to the native-codes or
        // encoded-output splice; the linear-light splice is reached via the
        // sink's Linear averaging domain, dispatched before this match.
        InsertionPoint::LinearLight => {
          unreachable!("encoded domain never selects the linear-light splice")
        }
      }
    }

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // Resolve the output set up front so the atomicity preflight below runs
    // before any output row is written.
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();

    // Atomicity preflight (#308, cf. the crate's #180 resample fix and the
    // high-bit 4:2:0 p0xx / yuv420p siblings): reserve the only growable row
    // scratch this identity row can touch — the u8 RGB row buffer — BEFORE any
    // output row is written (the luma plane below, then the u16 RGB / RGBA
    // fan-out), so an allocator refusal returns a typed `AllocationFailed` and
    // leaves the output frame untouched rather than partially mutated. At 4:4:4
    // the chroma is full-resolution (no upsampling), but the output staging
    // matches 4:2:0: the u16 RGB / RGBA outputs write straight into their caller
    // buffers (the rgb_u16 plane itself stages the rgba_u16 expand) and never
    // grow a scratch, and these formats expose no luma_u16. The allocating
    // (rgb=None) arm of `rgb_row_buf_or_scratch` is reached exactly when a colour
    // decode needs an RGB row but no caller RGB buffer is borrowable —
    // `want_hsv && want_rgba && !want_rgb`. The later decode reuses the
    // already-sized buffer, so the default path is byte-identical; only the
    // failure-path ordering changes.
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

    if let Some(luma) = luma.as_deref_mut() {
      let dst = &mut luma[one_plane_start..one_plane_end];
      for (d, &s) in dst.iter_mut().zip(row.y().iter()) {
        // Normalize BE-encoded wire bytes to host-native before the
        // luma downshift — see Yuv420p9 luma path for rationale.
        let logical = if BE { u16::from_be(s) } else { u16::from_le(s) };
        *d = (logical >> (BITS - 8)) as u8;
      }
    }

    // ===== u16 RGB / RGBA path (Strategy A) =====
    let want_rgb_u16 = rgb_u16.is_some();
    let want_rgba_u16 = rgba_u16.is_some();

    if want_rgba_u16 && !want_rgb_u16 {
      let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row =
        rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
      yuv444p14_to_rgba_u16_row_endian(
        row.y(),
        row.u(),
        row.v(),
        rgba_u16_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
        BE,
      );
    } else if want_rgb_u16 {
      let rgb_u16_buf = rgb_u16.as_deref_mut().unwrap();
      let rgb_plane_end =
        one_plane_end
          .checked_mul(3)
          .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
            w, h, 3,
          )))?;
      let rgb_plane_start = one_plane_start * 3;
      let rgb_u16_row = &mut rgb_u16_buf[rgb_plane_start..rgb_plane_end];
      yuv444p14_to_rgb_u16_row_endian(
        row.y(),
        row.u(),
        row.v(),
        rgb_u16_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
        BE,
      );
      if want_rgba_u16 {
        let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
        let rgba_u16_row =
          rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
        expand_rgb_u16_to_rgba_u16_row::<BITS>(rgb_u16_row, rgba_u16_row, w);
      }
    }

    // ===== u8 RGB / RGBA / HSV path (Strategy A) =====
    // HSV-without-RGB-or-RGBA goes through the direct `yuv444p14_to_hsv_row_endian`
    // kernel (no source-width RGB scratch — the SIMD path stages a fixed
    // 8-bit RGB chunk internally). RGB or RGBA also attached keeps the
    // convert-once-then-derive path alive via `need_rgb_kernel`.
    let want_hsv_direct = want_hsv && rgb.is_none() && !want_rgba;
    let need_rgb_kernel = rgb.is_some() || (want_hsv && want_rgba);

    if want_hsv_direct {
      let hsv = hsv.as_mut().expect("want_hsv_direct implies hsv attached");
      let (h, s, v) = hsv.hsv();
      yuv444p14_to_hsv_row_endian(
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
        BE,
      );
      return Ok(());
    }

    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      yuv444p14_to_rgba_row_endian(
        row.y(),
        row.u(),
        row.v(),
        rgba_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
        BE,
      );
      return Ok(());
    }

    if !need_rgb_kernel {
      return Ok(());
    }

    let rgb_row = rgb_row_buf_or_scratch(
      rgb.as_deref_mut(),
      rgb_scratch,
      one_plane_start,
      one_plane_end,
      w,
      h,
    )?;

    yuv444p14_to_rgb_row_endian(
      row.y(),
      row.u(),
      row.v(),
      rgb_row,
      w,
      row.matrix(),
      row.full_range(),
      use_simd,
      BE,
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

    if let Some(buf) = rgba.as_deref_mut() {
      let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
    }

    Ok(())
  }
}

impl<'a, R, const BE: bool> MixedSinker<'a, Yuv444p16<BE>, R> {
  /// Attaches a packed **`u16`** RGB output buffer. Output covers
  /// full `u16` range `[0, 65535]`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgb_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_rgb_u16(buf)?;
    Ok(self)
  }
  /// In-place variant.
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

  /// Attaches a packed **8-bit** RGBA output buffer. The 16-bit YUV
  /// source is converted to 8-bit RGBA via the dedicated 16-bit
  /// kernel; alpha = `0xFF`.
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

  /// Attaches a packed **`u16`** RGBA output buffer. Full-range output
  /// `[0, 65535]`; alpha element is `0xFFFF`.
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
}

impl<R, const BE: bool> Yuv444p16Sink<BE> for MixedSinker<'_, Yuv444p16<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, Yuv444p16<BE>, R> {
  type Input<'r> = Yuv444p16Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    reset_high_bit_yuv_streams(self);
    Ok(())
  }

  fn process(&mut self, row: Yuv444p16Row<'_>) -> Result<(), Self::Error> {
    const BITS: u32 = 16;
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Y16,
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
      hsv,
      rgb_scratch,
      rgb_scratch_u16,
      luma_scratch_u16,
      rgb_stream,
      rgb_stream_u16,
      luma_stream_u16,
      rgb_filter_stream,
      rgb_filter_stream_u16,
      luma_filter_stream_u16,
      resample_outputs,
      plan,
      native,
      native_planar_u16,
      frozen_native_route,
      ..
    } = self;

    // Non-identity plan: feed the shared high-bit 4:4:4 triple-resample
    // tail (u8 color, independent native-u16 color, native Y). The planar
    // decode closures stage source-width rows; chroma is full-width (no
    // upsampling). Yuv444p exposes no `luma_u16` output, so it is `&mut
    // None` and only `luma` (binned native Y `>> (BITS - 8)`) is emitted.
    // The span kind picks the engine (area bin or signed-coefficient filter
    // twin) — see the Yuv444p10 impl for the full rationale. At `BITS = 16`
    // the native max is the u16 max, so the filter tail's sub-16-bit clamp is
    // a value no-op.
    if let Some(plan) = plan.as_ref() {
      let matrix = row.matrix();
      let full_range = row.full_range();
      let (y, u, v) = (row.y(), row.u(), row.v());
      if plan.kind().is_filter() {
        return packed_yuv444_triple_filter_resample::<BITS>(
          rgb_filter_stream,
          rgb_filter_stream_u16,
          luma_filter_stream_u16,
          resample_outputs,
          rgb,
          rgba,
          rgb_u16,
          rgba_u16,
          luma,
          &mut None,
          hsv,
          rgb_scratch,
          rgb_scratch_u16,
          luma_scratch_u16,
          w,
          plan,
          idx,
          use_simd,
          matrix,
          full_range,
          |scratch| {
            yuv444p16_to_rgb_row_endian(y, u, v, scratch, w, matrix, full_range, use_simd, BE)
          },
          |scratch| {
            yuv444p16_to_rgb_u16_row_endian(y, u, v, scratch, w, matrix, full_range, use_simd, BE)
          },
          |scratch| deinterleave_y_high_bit_masked::<BITS, BE>(y, scratch, w),
        );
      }
      // Native / row-stage route split — see the high-bit 4:2:0 Yuv420p impl
      // for the CHECK-before / SET-after `frozen_native_route` contract.
      let need_output = luma.is_some()
        || rgb.is_some()
        || rgba.is_some()
        || hsv.is_some()
        || rgb_u16.is_some()
        || rgba_u16.is_some();
      if need_output
        && let Some(frozen) = *frozen_native_route
        && frozen != *native
      {
        return Err(MixedSinkerError::NativeRouteChanged(
          NativeRouteChanged::new(idx),
        ));
      }
      // RFC #238 splice-stage selection — see the Yuv420p impl for the
      // selector contract; reproduces the former `if *native` boolean
      // bit-for-bit (a filter plan already returned above, so `area_plan` is
      // always true here).
      let insertion = select_insertion_point(
        AveragingDomain::Encoded,
        InsertionContext {
          native_eligible: YUV444P_HIGH_BIT_NATIVE_ELIGIBLE,
          with_native: *native,
          area_plan: true,
        },
      );
      match insertion {
        InsertionPoint::NativeCodes => {
          // 4:4:4: chroma `w x h` — identical to Y; a chroma row per Y row
          // (`chroma_vsub = 1`, `chroma_w = w`), chroma plan equals the luma
          // plan.
          yuv_planar16_process_native::<BITS, BE>(
            plan,
            native_planar_u16,
            resample_outputs,
            rgb,
            rgba,
            rgb_u16,
            rgba_u16,
            luma,
            // The high-bit planar 4:4:4 family exposes no `luma_u16` output.
            &mut None,
            hsv,
            rgb_scratch,
            rgb_scratch_u16,
            y,
            u,
            v,
            matrix,
            full_range,
            idx,
            w,
            h,
            1,
            w,
            || ResamplePlan::area(w, h, plan.out_w(), plan.out_h()),
            use_simd,
          )?;
          if frozen_native_route.is_none() && need_output {
            *frozen_native_route = Some(true);
          }
          return Ok(());
        }
        InsertionPoint::EncodedOutput => {
          packed_yuv444_triple_resample::<BITS>(
            rgb_stream,
            rgb_stream_u16,
            luma_stream_u16,
            resample_outputs,
            rgb,
            rgba,
            rgb_u16,
            rgba_u16,
            luma,
            &mut None,
            hsv,
            rgb_scratch,
            rgb_scratch_u16,
            luma_scratch_u16,
            w,
            plan,
            idx,
            use_simd,
            matrix,
            full_range,
            |scratch| {
              yuv444p16_to_rgb_row_endian(y, u, v, scratch, w, matrix, full_range, use_simd, BE)
            },
            |scratch| {
              yuv444p16_to_rgb_u16_row_endian(y, u, v, scratch, w, matrix, full_range, use_simd, BE)
            },
            |scratch| deinterleave_y_high_bit_masked::<BITS, BE>(y, scratch, w),
          )?;
          if frozen_native_route.is_none() && need_output {
            *frozen_native_route = Some(false);
          }
          return Ok(());
        }
        // The encoded domain only resolves to the native-codes or
        // encoded-output splice; the linear-light splice is reached via the
        // sink's Linear averaging domain, dispatched before this match.
        InsertionPoint::LinearLight => {
          unreachable!("encoded domain never selects the linear-light splice")
        }
      }
    }

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // Resolve the output set up front so the atomicity preflight below runs
    // before any output row is written.
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();

    // Atomicity preflight (#308, cf. the crate's #180 resample fix and the
    // high-bit 4:2:0 p0xx / yuv420p siblings): reserve the only growable row
    // scratch this identity row can touch — the u8 RGB row buffer — BEFORE any
    // output row is written (the luma plane below, then the u16 RGB / RGBA
    // fan-out), so an allocator refusal returns a typed `AllocationFailed` and
    // leaves the output frame untouched rather than partially mutated. At 4:4:4
    // the chroma is full-resolution (no upsampling), but the output staging
    // matches 4:2:0: the u16 RGB / RGBA outputs write straight into their caller
    // buffers (the rgb_u16 plane itself stages the rgba_u16 expand) and never
    // grow a scratch, and these formats expose no luma_u16. The allocating
    // (rgb=None) arm of `rgb_row_buf_or_scratch` is reached exactly when a colour
    // decode needs an RGB row but no caller RGB buffer is borrowable —
    // `want_hsv && want_rgba && !want_rgb`. The later decode reuses the
    // already-sized buffer, so the default path is byte-identical; only the
    // failure-path ordering changes.
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

    if let Some(luma) = luma.as_deref_mut() {
      let dst = &mut luma[one_plane_start..one_plane_end];
      for (d, &s) in dst.iter_mut().zip(row.y().iter()) {
        // Normalize BE-encoded wire bytes to host-native before the
        // luma downshift — see Yuv420p9 luma path for rationale.
        let logical = if BE { u16::from_be(s) } else { u16::from_le(s) };
        *d = (logical >> (BITS - 8)) as u8;
      }
    }

    // ===== u16 RGB / RGBA path (Strategy A) =====
    let want_rgb_u16 = rgb_u16.is_some();
    let want_rgba_u16 = rgba_u16.is_some();

    if want_rgba_u16 && !want_rgb_u16 {
      let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row =
        rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
      yuv444p16_to_rgba_u16_row_endian(
        row.y(),
        row.u(),
        row.v(),
        rgba_u16_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
        BE,
      );
    } else if want_rgb_u16 {
      let rgb_u16_buf = rgb_u16.as_deref_mut().unwrap();
      let rgb_plane_end =
        one_plane_end
          .checked_mul(3)
          .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
            w, h, 3,
          )))?;
      let rgb_plane_start = one_plane_start * 3;
      let rgb_u16_row = &mut rgb_u16_buf[rgb_plane_start..rgb_plane_end];
      yuv444p16_to_rgb_u16_row_endian(
        row.y(),
        row.u(),
        row.v(),
        rgb_u16_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
        BE,
      );
      if want_rgba_u16 {
        let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
        let rgba_u16_row =
          rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
        expand_rgb_u16_to_rgba_u16_row::<BITS>(rgb_u16_row, rgba_u16_row, w);
      }
    }

    // ===== u8 RGB / RGBA / HSV path (Strategy A) =====
    // HSV-without-RGB-or-RGBA goes through the direct `yuv444p16_to_hsv_row_endian`
    // kernel (no source-width RGB scratch — the SIMD path stages a fixed
    // 8-bit RGB chunk internally). RGB or RGBA also attached keeps the
    // convert-once-then-derive path alive via `need_rgb_kernel`.
    let want_hsv_direct = want_hsv && rgb.is_none() && !want_rgba;
    let need_rgb_kernel = rgb.is_some() || (want_hsv && want_rgba);

    if want_hsv_direct {
      let hsv = hsv.as_mut().expect("want_hsv_direct implies hsv attached");
      let (h, s, v) = hsv.hsv();
      yuv444p16_to_hsv_row_endian(
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
        BE,
      );
      return Ok(());
    }

    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      yuv444p16_to_rgba_row_endian(
        row.y(),
        row.u(),
        row.v(),
        rgba_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
        BE,
      );
      return Ok(());
    }

    if !need_rgb_kernel {
      return Ok(());
    }

    let rgb_row = rgb_row_buf_or_scratch(
      rgb.as_deref_mut(),
      rgb_scratch,
      one_plane_start,
      one_plane_end,
      w,
      h,
    )?;

    yuv444p16_to_rgb_row_endian(
      row.y(),
      row.u(),
      row.v(),
      rgb_row,
      w,
      row.matrix(),
      row.full_range(),
      use_simd,
      BE,
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

    if let Some(buf) = rgba.as_deref_mut() {
      let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
    }

    Ok(())
  }
}
