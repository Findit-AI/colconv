//! MSB-aligned high-bit planar YUV 4:4:4 sinker impls (`Yuv444p10Msb` /
//! `Yuv444p12Msb`).
//!
//! The recovery-shift twins of the low-bit [`yuv444p`](super::yuv444p)
//! `Yuv444p10` / `Yuv444p12` impls: samples live in the **high** `BITS` bits of
//! each `u16` (FFmpeg `shift = 16 - BITS`), recovered via `>> (16 - BITS)`
//! instead of a low-bit mask. Once recovered the sample lands in
//! `[0, (1 << BITS) - 1]`, so the entire YUV→RGB pipeline, the HSV / luma
//! derivations, and the fused area / filter / native resample tails are reused
//! from the low-bit family unchanged. The only deltas are the per-sample
//! extraction (the `_msb_` row kernels) and, for the native fast tier, a
//! per-row de-pack into host-native LOGICAL scratch before the shared planar
//! join (mirroring the high-bit-packed P4xx `p4xx_process_native`).

use super::super::{
  GeometryOverflow, InsufficientBuffer, MixedSinker, MixedSinkerError, NativeRouteChanged,
  RowIndexOutOfRange, RowShapeMismatch, RowSlice, check_dimensions_match,
  packed_yuv444_triple_filter_resample, packed_yuv444_triple_resample, reset_high_bit_yuv_streams,
  rgb_row_buf_or_scratch, rgba_plane_row_slice, rgba_u16_plane_row_slice,
};
use crate::{
  KernelMatrix, PixelSink,
  resample::{
    AveragingDomain, InsertionContext, InsertionPoint, ResamplePlan, select_insertion_point,
  },
  row::*,
  source::*,
};

#[cfg(feature = "yuv-planar")]
use super::super::{
  FrozenOutputs, HsvFrameMut, NativePlanarYuvU16, native_planar_hb_preflight_check_only,
  yuv_planar16_process_native,
};
#[cfg(feature = "yuv-planar")]
use crate::resample::{PlanGeometry, ResampleError};

/// The MSB-aligned high-bit 4:4:4 planar formats ship the non-4:2:0 native
/// planar fast tier (via [`yuv444p_msb_process_native`]), so each is statically
/// eligible to splice an [`AveragingDomain::Encoded`] area downscale at the
/// native codes.
const YUV444P_MSB_NATIVE_ELIGIBLE: bool = true;

// The native fast tier de-packs each wire plane into wrapper-owned host-native
// LOGICAL u16 scratch BEFORE handing it to the planar delegate, so the
// delegate's own `from_le` / `from_be` decode must be a no-op load on every
// host: pass `BE = HOST_NATIVE_BE` (= `from_ne`). Mirrors the high-bit-packed
// semi-planar P4xx `p4xx_process_native`.
#[cfg(feature = "yuv-planar")]
const HOST_NATIVE_BE: bool = cfg!(target_endian = "big");

/// Grows a wrapper-owned de-pack scratch to `len` `u16` under the planner's
/// recoverable-allocation contract. Runs after the join's preflight clears, so
/// a rejected row never reaches it.
#[cfg(feature = "yuv-planar")]
#[cfg_attr(not(tarpaulin), inline(always))]
fn grow_msb_depack(
  scratch: &mut std::vec::Vec<u16>,
  len: usize,
  w: usize,
  h: usize,
  plan: &ResamplePlan,
) -> Result<(), MixedSinkerError> {
  if scratch.len() < len {
    scratch
      .try_reserve_exact(len - scratch.len())
      .map_err(|_| {
        MixedSinkerError::Resample(ResampleError::AllocationFailed(PlanGeometry::new(
          w,
          h,
          plan.out_w(),
          plan.out_h(),
        )))
      })?;
    scratch.resize(len, 0);
  }
  Ok(())
}

/// Native fast-tier decimator for the **MSB-aligned high-bit planar 4:4:4**
/// family ([`Yuv444p10Msb`] / [`Yuv444p12Msb`]): de-packs the wire Y / U / V
/// planes into wrapper-owned host-native LOGICAL u16 scratch (`>> (16 - BITS)`),
/// then reuses the high-bit non-4:2:0 PLANAR join verbatim
/// ([`yuv_planar16_process_native`]) at `BE = HOST_NATIVE_BE`. The planar twin
/// of the high-bit-packed semi-planar `p4xx_process_native`; the only
/// difference is the planar layout (three separate planes, no UV de-interleave).
///
/// 4:4:4: chroma is full-resolution (`chroma_w = w`, `chroma_vsub = 1`), so a
/// chroma row feeds EVERY colour Y row and the chroma de-pack runs at full
/// width. The join's COMPLETE pre-feed preflight runs FIRST (deterministic typed
/// rejection, never `AllocationFailed`); a luma-only sink skips the chroma
/// de-pack / scratch.
#[cfg(feature = "yuv-planar")]
#[allow(clippy::too_many_arguments)]
fn yuv444p_msb_process_native<const BITS: u32, const BE: bool>(
  plan: &ResamplePlan,
  native_planar_u16: &mut Option<std::boxed::Box<NativePlanarYuvU16>>,
  y_scratch: &mut std::vec::Vec<u16>,
  u_scratch: &mut std::vec::Vec<u16>,
  v_scratch: &mut std::vec::Vec<u16>,
  resample_outputs: &mut Option<FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  rgb_u16: &mut Option<&mut [u16]>,
  rgba_u16: &mut Option<&mut [u16]>,
  luma: &mut Option<&mut [u8]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  rgb_scratch: &mut std::vec::Vec<u8>,
  rgb_scratch_u16: &mut std::vec::Vec<u16>,
  y_row: &[u16],
  u_row: &[u16],
  v_row: &[u16],
  matrix: KernelMatrix,
  full_range: bool,
  idx: usize,
  w: usize,
  h: usize,
  use_simd: bool,
) -> Result<(), MixedSinkerError> {
  const { assert!(BITS == 10 || BITS == 12, "BITS must be 10 or 12") };
  let need_luma = luma.is_some();
  let need_color =
    rgb.is_some() || rgba.is_some() || hsv.is_some() || rgb_u16.is_some() || rgba_u16.is_some();

  // The planar join's COMPLETE compare-only pre-feed rejection preflight runs
  // FIRST — BEFORE any fallible scratch grow below — so every rejection returns
  // its deterministic typed error and leaves the wrapper scratch untouched. The
  // MSB planar 4:4:4 family exposes no `luma_u16` output (`&None`). Compare-only
  // (no output-set freeze): the de-pack grows below AND the delegate's own
  // colour / source scratch grows all stay pre-commit, and the delegate performs
  // the SINGLE freeze once every pre-feed allocation has succeeded — so a LATER
  // grow (here or in the delegate) failing leaves `resample_outputs` untouched
  // and the row retryable.
  if !native_planar_hb_preflight_check_only(
    native_planar_u16,
    resample_outputs,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    luma,
    &None,
    hsv,
    idx,
    need_luma,
    need_color,
  )? {
    return Ok(());
  }

  // Grow the wrapper de-pack scratch — Y always, U / V only on a colour row
  // (4:4:4: every Y row is a chroma row when colour is wanted). All grows
  // precede the infallible de-pack and the delegate call.
  grow_msb_depack(y_scratch, w, w, h, plan)?;
  if need_color {
    grow_msb_depack(u_scratch, w, w, h, plan)?;
    grow_msb_depack(v_scratch, w, w, h, plan)?;
  }

  // De-pack the wire planes into host-native LOGICAL scratch: decode the wire
  // endianness, then shift the active high `BITS` down to the low `BITS`
  // (`>> (16 - BITS)`). Everything past here is infallible.
  for (d, &s) in y_scratch[..w].iter_mut().zip(y_row.iter()) {
    let logical = if BE { u16::from_be(s) } else { u16::from_le(s) };
    *d = logical >> (16 - BITS);
  }
  if need_color {
    for (d, &s) in u_scratch[..w].iter_mut().zip(u_row.iter()) {
      let logical = if BE { u16::from_be(s) } else { u16::from_le(s) };
      *d = logical >> (16 - BITS);
    }
    for (d, &s) in v_scratch[..w].iter_mut().zip(v_row.iter()) {
      let logical = if BE { u16::from_be(s) } else { u16::from_le(s) };
      *d = logical >> (16 - BITS);
    }
  }

  // Delegate to the planar high-bit non-4:2:0 join with `BE = HOST_NATIVE_BE`
  // at 4:4:4 chroma geometry (`chroma_vsub = 1`, `chroma_w = w`). Empty U / V on
  // luma-only rows (the join reads chroma only under colour).
  let (u_plane, v_plane): (&[u16], &[u16]) = if need_color {
    (&u_scratch[..w], &v_scratch[..w])
  } else {
    (&[], &[])
  };
  yuv_planar16_process_native::<BITS, HOST_NATIVE_BE>(
    plan,
    native_planar_u16,
    resample_outputs,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    luma,
    // The MSB planar 4:4:4 family exposes no `luma_u16` output.
    &mut None,
    hsv,
    rgb_scratch,
    rgb_scratch_u16,
    &y_scratch[..w],
    u_plane,
    v_plane,
    matrix,
    full_range,
    idx,
    w,
    h,
    1,
    w,
    || ResamplePlan::area(w, h, plan.out_w(), plan.out_h()),
    use_simd,
  )
}

// ---- Yuv444p10Msb impl --------------------------------------------------

impl<'a, R, const BE: bool> MixedSinker<'a, Yuv444p10Msb<BE>, R> {
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

impl<R, const BE: bool> Yuv444p10MsbSink<BE> for MixedSinker<'_, Yuv444p10Msb<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, Yuv444p10Msb<BE>, R> {
  type Input<'r> = Yuv444p10MsbRow<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    reset_high_bit_yuv_streams(self);
    Ok(())
  }

  fn process(&mut self, row: Yuv444p10MsbRow<'_>) -> Result<(), Self::Error> {
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
      yuv444_msb_y_depack,
      yuv444_msb_u_depack,
      yuv444_msb_v_depack,
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
            yuv444p10_msb_to_rgb_row_endian(y, u, v, scratch, w, matrix, full_range, use_simd, BE)
          },
          |scratch| {
            yuv444p10_msb_to_rgb_u16_row_endian(
              y, u, v, scratch, w, matrix, full_range, use_simd, BE,
            )
          },
          |scratch| {
            for (d, &s) in scratch[..w].iter_mut().zip(y.iter()) {
              let logical = if BE { u16::from_be(s) } else { u16::from_le(s) };
              *d = logical >> (16 - BITS);
            }
          },
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
          native_eligible: YUV444P_MSB_NATIVE_ELIGIBLE,
          with_native: *native,
          area_plan: true,
        },
      );
      match insertion {
        InsertionPoint::NativeCodes => {
          // 4:4:4: chroma `w x h` — identical to Y; a chroma row per Y row
          // (`chroma_vsub = 1`, `chroma_w = w`), chroma plan equals the luma
          // plan.
          yuv444p_msb_process_native::<BITS, BE>(
            plan,
            native_planar_u16,
            yuv444_msb_y_depack,
            yuv444_msb_u_depack,
            yuv444_msb_v_depack,
            resample_outputs,
            rgb,
            rgba,
            rgb_u16,
            rgba_u16,
            luma,
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
              yuv444p10_msb_to_rgb_row_endian(y, u, v, scratch, w, matrix, full_range, use_simd, BE)
            },
            |scratch| {
              yuv444p10_msb_to_rgb_u16_row_endian(
                y, u, v, scratch, w, matrix, full_range, use_simd, BE,
              )
            },
            |scratch| {
              for (d, &s) in scratch[..w].iter_mut().zip(y.iter()) {
                let logical = if BE { u16::from_be(s) } else { u16::from_le(s) };
                *d = logical >> (16 - BITS);
              }
            },
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
    // leaves the output frame untouched rather than partially mutated. The
    // MSB-aligned recovery doesn't change the output staging: the u16 RGB / RGBA
    // outputs write straight into their caller buffers (the rgb_u16 plane itself
    // stages the rgba_u16 expand) and never grow a scratch, and these formats
    // expose no luma_u16. The allocating (rgb=None) arm of
    // `rgb_row_buf_or_scratch` is reached exactly when a colour decode needs an
    // RGB row but no caller RGB buffer is borrowable —
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
        *d = (logical >> 8) as u8;
      }
    }

    // ===== u16 RGB / RGBA path (Strategy A) =====
    let want_rgb_u16 = rgb_u16.is_some();
    let want_rgba_u16 = rgba_u16.is_some();

    if want_rgba_u16 && !want_rgb_u16 {
      let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row =
        rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
      yuv444p10_msb_to_rgba_u16_row_endian(
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
      yuv444p10_msb_to_rgb_u16_row_endian(
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
    // HSV-without-RGB-or-RGBA goes through the direct `yuv444p10_msb_to_hsv_row_endian`
    // kernel (no source-width RGB scratch — the SIMD path stages a fixed
    // 8-bit RGB chunk internally). RGB or RGBA also attached keeps the
    // convert-once-then-derive path alive via `need_rgb_kernel`.
    let want_hsv_direct = want_hsv && rgb.is_none() && !want_rgba;
    let need_rgb_kernel = rgb.is_some() || (want_hsv && want_rgba);

    if want_hsv_direct {
      let hsv = hsv.as_mut().expect("want_hsv_direct implies hsv attached");
      let (h, s, v) = hsv.hsv();
      yuv444p10_msb_to_hsv_row_endian(
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
      yuv444p10_msb_to_rgba_row_endian(
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

    yuv444p10_msb_to_rgb_row_endian(
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

// ---- Yuv444p12Msb impl --------------------------------------------------

impl<'a, R, const BE: bool> MixedSinker<'a, Yuv444p12Msb<BE>, R> {
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

impl<R, const BE: bool> Yuv444p12MsbSink<BE> for MixedSinker<'_, Yuv444p12Msb<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, Yuv444p12Msb<BE>, R> {
  type Input<'r> = Yuv444p12MsbRow<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    reset_high_bit_yuv_streams(self);
    Ok(())
  }

  fn process(&mut self, row: Yuv444p12MsbRow<'_>) -> Result<(), Self::Error> {
    const BITS: u32 = 12;
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

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
      yuv444_msb_y_depack,
      yuv444_msb_u_depack,
      yuv444_msb_v_depack,
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
            yuv444p12_msb_to_rgb_row_endian(y, u, v, scratch, w, matrix, full_range, use_simd, BE)
          },
          |scratch| {
            yuv444p12_msb_to_rgb_u16_row_endian(
              y, u, v, scratch, w, matrix, full_range, use_simd, BE,
            )
          },
          |scratch| {
            for (d, &s) in scratch[..w].iter_mut().zip(y.iter()) {
              let logical = if BE { u16::from_be(s) } else { u16::from_le(s) };
              *d = logical >> (16 - BITS);
            }
          },
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
          native_eligible: YUV444P_MSB_NATIVE_ELIGIBLE,
          with_native: *native,
          area_plan: true,
        },
      );
      match insertion {
        InsertionPoint::NativeCodes => {
          // 4:4:4: chroma `w x h` — identical to Y; a chroma row per Y row
          // (`chroma_vsub = 1`, `chroma_w = w`), chroma plan equals the luma
          // plan.
          yuv444p_msb_process_native::<BITS, BE>(
            plan,
            native_planar_u16,
            yuv444_msb_y_depack,
            yuv444_msb_u_depack,
            yuv444_msb_v_depack,
            resample_outputs,
            rgb,
            rgba,
            rgb_u16,
            rgba_u16,
            luma,
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
              yuv444p12_msb_to_rgb_row_endian(y, u, v, scratch, w, matrix, full_range, use_simd, BE)
            },
            |scratch| {
              yuv444p12_msb_to_rgb_u16_row_endian(
                y, u, v, scratch, w, matrix, full_range, use_simd, BE,
              )
            },
            |scratch| {
              for (d, &s) in scratch[..w].iter_mut().zip(y.iter()) {
                let logical = if BE { u16::from_be(s) } else { u16::from_le(s) };
                *d = logical >> (16 - BITS);
              }
            },
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
    // leaves the output frame untouched rather than partially mutated. The
    // MSB-aligned recovery doesn't change the output staging: the u16 RGB / RGBA
    // outputs write straight into their caller buffers (the rgb_u16 plane itself
    // stages the rgba_u16 expand) and never grow a scratch, and these formats
    // expose no luma_u16. The allocating (rgb=None) arm of
    // `rgb_row_buf_or_scratch` is reached exactly when a colour decode needs an
    // RGB row but no caller RGB buffer is borrowable —
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
        *d = (logical >> 8) as u8;
      }
    }

    // ===== u16 RGB / RGBA path (Strategy A) =====
    let want_rgb_u16 = rgb_u16.is_some();
    let want_rgba_u16 = rgba_u16.is_some();

    if want_rgba_u16 && !want_rgb_u16 {
      let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row =
        rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
      yuv444p12_msb_to_rgba_u16_row_endian(
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
      yuv444p12_msb_to_rgb_u16_row_endian(
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
    // HSV-without-RGB-or-RGBA goes through the direct `yuv444p12_msb_to_hsv_row_endian`
    // kernel (no source-width RGB scratch — the SIMD path stages a fixed
    // 8-bit RGB chunk internally). RGB or RGBA also attached keeps the
    // convert-once-then-derive path alive via `need_rgb_kernel`.
    let want_hsv_direct = want_hsv && rgb.is_none() && !want_rgba;
    let need_rgb_kernel = rgb.is_some() || (want_hsv && want_rgba);

    if want_hsv_direct {
      let hsv = hsv.as_mut().expect("want_hsv_direct implies hsv attached");
      let (h, s, v) = hsv.hsv();
      yuv444p12_msb_to_hsv_row_endian(
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
      yuv444p12_msb_to_rgba_row_endian(
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

    yuv444p12_msb_to_rgb_row_endian(
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
