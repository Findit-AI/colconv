use super::super::{
  ChromaSitingChanged, ChromaU16, GeometryOverflow, InsufficientBuffer, MixedSinker,
  MixedSinkerError, NativeRouteChanged, RowIndexOutOfRange, RowShapeMismatch, RowSlice,
  WidthAlignment, check_dimensions_match, chroma_422_center_sited_h,
  deinterleave_y_high_bit_masked, packed_yuv422_triple_filter_resample,
  packed_yuv422_triple_resample, planar_8bit::YUV422P_CENTERED_H_PHASE,
  planar_resample::resample_preflight_check_only, reconstruct_chroma, reset_high_bit_yuv_streams,
  rgb_row_buf_or_scratch, rgba_plane_row_slice, rgba_u16_plane_row_slice,
  subsampled_4_2_0_high_bit::reserve_420_chroma_full_u16, yuv_planar16_process_native,
};
use crate::{
  PixelSink,
  resample::{
    AveragingDomain, InsertionContext, InsertionPoint, ResamplePlan, select_insertion_point,
  },
  row::*,
  source::*,
};

/// The high-bit 4:2:2 planar formats (`Yuv422p9` … `Yuv422p16`) ship the
/// non-4:2:0 native planar fast tier ([`yuv_planar16_process_native`]), so
/// each is statically eligible to splice an [`AveragingDomain::Encoded`] area
/// downscale at the native codes.
const YUV422P_HIGH_BIT_NATIVE_ELIGIBLE: bool = true;

// ---- Yuv422p9 impl -----------------------------------------------------
//
// 4:2:2 planar 9‑bit — same per-row chroma shape as 4:2:0 (half-width
// U / V), one chroma row per Y row instead of one per two. Reuses
// `yuv420p9_to_rgb_*` row primitives verbatim.

impl<'a, R, const BE: bool> MixedSinker<'a, Yuv422p9<BE>, R> {
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

  /// Attaches a packed **8‑bit** RGBA output buffer. The 9‑bit YUV
  /// source is converted to 8‑bit RGBA via the same `BITS = 9` Q15
  /// kernel family used by [`Self::with_rgb`]; the fourth byte per
  /// pixel is alpha = `0xFF` (Yuv422p9 has no alpha plane).
  ///
  /// Returns `Err(InsufficientRgbaBuffer)` if
  /// `buf.len() < width x height x 4`, or `Err(GeometryOverflow)` on
  /// 32‑bit targets when the product overflows.
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

  /// Attaches a packed **`u16`** RGBA output buffer. 9‑bit low‑packed
  /// (`(1 << 9) - 1 = 511` max). Length is measured in `u16`
  /// **elements** (`width x height x 4`). Alpha element is
  /// `(1 << 9) - 1`.
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

impl<R, const BE: bool> Yuv422p9Sink<BE> for MixedSinker<'_, Yuv422p9<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, Yuv422p9<BE>, R> {
  type Input<'r> = Yuv422p9Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::WidthAlignment(WidthAlignment::odd(
        self.width,
      )));
    }
    check_dimensions_match(self.width, self.height, width, height)?;
    reset_high_bit_yuv_streams(self);
    Ok(())
  }

  fn process(&mut self, row: Yuv422p9Row<'_>) -> Result<(), Self::Error> {
    const BITS: u32 = 9;
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if w & 1 != 0 {
      return Err(MixedSinkerError::WidthAlignment(WidthAlignment::odd(w)));
    }
    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Y9,
        idx,
        w,
        row.y().len(),
      )));
    }
    if row.u_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::UHalf9,
        idx,
        w / 2,
        row.u_half().len(),
      )));
    }
    if row.v_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::VHalf9,
        idx,
        w / 2,
        row.v_half().len(),
      )));
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange(
        RowIndexOutOfRange::new(idx, self.height),
      ));
    }

    // Chroma siting (#302): drives the identity-plan horizontal chroma phase.
    // `Copy`, so read it out before the field split-borrow below.
    let chroma_location = self.chroma_location;

    let Self {
      rgb,
      rgb_u16,
      rgba,
      rgba_u16,
      luma,
      hsv,
      rgb_scratch,
      rgb_scratch_u16,
      chroma_full_u16,
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
      frozen_chroma_centered,
      ..
    } = self;

    // Non-identity plan: feed the shared high-bit 4:2:2 triple-resample
    // tail (u8 color, independent native-u16 color, native Y). The half-
    // width U / V planes are horizontally upsampled in-register by the
    // shared 4:2:0 row kernels (4:2:0 and 4:2:2 have the identical per-row
    // chroma contract). Yuv422p exposes no `luma_u16` output, so it is
    // `&mut None` and only `luma` (binned native Y `>> (BITS - 8)`) is
    // emitted. The span kind picks the engine (area bin or signed-coefficient
    // filter twin) — see the Yuv422p10 impl for the full rationale; the
    // filter tail clamps every sub-16-bit colour sample AND the native Y to
    // `(1 << BITS) - 1`.
    if let Some(plan) = plan.as_ref() {
      let matrix = row.matrix();
      let full_range = row.full_range();
      let (y, u_half, v_half) = (row.y(), row.u_half(), row.v_half());
      // RFC #238 S5a — 4:2:2 horizontal chroma siting. The centered group
      // (`Center` / `Top` / `Bottom`, `chroma_422_center_sited_h`) samples
      // chroma at `+0.5` luma = `+0.25` chroma-sample; the co-sited /
      // unspecified group is phase 0 (today's byte-identical decode). Siting
      // enters the chroma RECONSTRUCTION only — the averaging tier is still
      // chosen by `select_insertion_point` below — so the native fast tier
      // folds the phase into the `area_chroma_422` chroma weights while the
      // row-stage and filter tiers reconstruct full-width `u16` chroma and
      // decode 4:4:4.
      let center_sited = chroma_422_center_sited_h(chroma_location);
      let chroma_h_phase = if center_sited {
        YUV422P_CENTERED_H_PHASE
      } else {
        0.0
      };
      let need_output = luma.is_some()
        || rgb.is_some()
        || rgba.is_some()
        || hsv.is_some()
        || rgb_u16.is_some()
        || rgba_u16.is_some();
      // Only the colour tiers reconstruct full-width chroma for the centered
      // decode; a luma-only centered row bins native Y unchanged (siting is a
      // chroma-only property).
      let want_color =
        rgb.is_some() || rgba.is_some() || hsv.is_some() || rgb_u16.is_some() || rgba_u16.is_some();
      // Freeze the effective 4:2:2 chroma siting on the first output-bearing
      // row (mirrors the `frozen_native_route` freeze below). This CHECK is at
      // the always-compiled choke point every tier passes through; the matching
      // SET rides each tier's accept path (never before dispatch, so a rejected
      // row leaves it unset for a corrected retry). A later row observing a
      // different phase would bin a mixture of co-sited and centered chroma, so
      // it is rejected here before any reconstruction.
      if need_output
        && let Some(frozen) = *frozen_chroma_centered
        && frozen != center_sited
      {
        return Err(MixedSinkerError::ChromaSitingChanged(
          ChromaSitingChanged::new(idx),
        ));
      }
      if plan.kind().is_filter() {
        // Reject a multi-kernel (BICUBLIN) plan BEFORE the centered reserve
        // below — the delegate's first act is this same check, so hoisting it
        // keeps a rejected filter plan from reserving / reconstructing chroma
        // first (the #180 reject-before-allocation invariant). Idempotent — the
        // delegate re-runs it.
        plan.ensure_single_kernel_filter()?;
        if center_sited && want_color {
          // Centered filter: reconstruct full-width `u16` chroma, but ONLY after
          // the resample preflight (frozen-output + sequence), so an
          // out-of-sequence / rejected row is caught before the chroma
          // reservation (#180). `packed_yuv422_triple_filter_resample` re-runs
          // the idempotent preflight and owns the transactional commit.
          let expected = if luma.is_some() {
            luma_filter_stream_u16.as_ref().map_or(0, |s| s.next_y())
          } else if rgb.is_some() || rgba.is_some() || hsv.is_some() {
            rgb_filter_stream.as_ref().map_or(0, |s| s.next_y())
          } else {
            rgb_filter_stream_u16.as_ref().map_or(0, |s| s.next_y())
          };
          if let core::ops::ControlFlow::Break(()) = resample_preflight_check_only(
            resample_outputs,
            luma,
            &None,
            rgb,
            rgba,
            rgb_u16,
            rgba_u16,
            &None,
            &None,
            &None,
            &None,
            &None,
            hsv,
            &None,
            Some(expected),
            idx,
          )? {
            return Ok(());
          }
          reserve_420_chroma_full_u16(chroma_full_u16, w, h)?;
          let (u_full, v_full) = reconstruct_chroma(
            ChromaU16::<BITS> { big_endian: BE },
            chroma_full_u16,
            u_half,
            v_half,
            w,
          );
          let r = packed_yuv422_triple_filter_resample::<BITS>(
            luma_filter_stream_u16,
            rgb_filter_stream,
            rgb_filter_stream_u16,
            resample_outputs,
            rgb,
            rgba,
            rgb_u16,
            rgba_u16,
            luma,
            &mut None,
            hsv,
            luma_scratch_u16,
            rgb_scratch,
            rgb_scratch_u16,
            w,
            plan,
            idx,
            use_simd,
            matrix,
            full_range,
            |scratch| deinterleave_y_high_bit_masked::<BITS, BE>(y, scratch, w),
            |scratch| {
              yuv444p9_to_rgb_row_endian(
                y, u_full, v_full, scratch, w, matrix, full_range, use_simd, BE,
              )
            },
            |scratch| {
              yuv444p9_to_rgb_u16_row_endian(
                y, u_full, v_full, scratch, w, matrix, full_range, use_simd, BE,
              )
            },
          );
          if r.is_ok() && need_output && frozen_chroma_centered.is_none() {
            *frozen_chroma_centered = Some(center_sited);
          }
          return r;
        }
        let r = packed_yuv422_triple_filter_resample::<BITS>(
          luma_filter_stream_u16,
          rgb_filter_stream,
          rgb_filter_stream_u16,
          resample_outputs,
          rgb,
          rgba,
          rgb_u16,
          rgba_u16,
          luma,
          &mut None,
          hsv,
          luma_scratch_u16,
          rgb_scratch,
          rgb_scratch_u16,
          w,
          plan,
          idx,
          use_simd,
          matrix,
          full_range,
          |scratch| deinterleave_y_high_bit_masked::<BITS, BE>(y, scratch, w),
          |scratch| {
            yuv420p9_to_rgb_row_endian(
              y, u_half, v_half, scratch, w, matrix, full_range, use_simd, BE,
            )
          },
          |scratch| {
            yuv420p9_to_rgb_u16_row_endian(
              y, u_half, v_half, scratch, w, matrix, full_range, use_simd, BE,
            )
          },
        );
        if r.is_ok() && need_output && frozen_chroma_centered.is_none() {
          *frozen_chroma_centered = Some(center_sited);
        }
        return r;
      }
      // Native / row-stage route split — see the high-bit 4:2:0 Yuv420p impl
      // for the CHECK-before / SET-after `frozen_native_route` contract.
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
          native_eligible: YUV422P_HIGH_BIT_NATIVE_ELIGIBLE,
          with_native: *native,
          area_plan: true,
        },
      );
      match insertion {
        InsertionPoint::NativeCodes => {
          // 4:2:2: chroma `w/2 x h` — half width, full height; a chroma row per
          // Y row (`chroma_vsub = 1`, `chroma_w = w/2`), chroma plan the phased
          // `area_chroma_422` (h-phase `0.25` chroma-sample when centered, else
          // the plain box). RFC #238 point-of-use siting invalidation: a reused
          // sink's cached join is only `reset` between frames, so a frame whose
          // `chroma_location` moved to a different phase must REBUILD it. Drop
          // the stale-phase join ONLY on the in-sequence first row of a fresh
          // frame (`idx == 0`, `next_y() == 0`) so a mid-frame / out-of-sequence
          // row rejects against the INTACT join and a corrected retry rebuilds
          // cleanly; a luma-only join carries no chroma phase and is never
          // dropped. Move it OUT (the delegate builds the replacement into the
          // field, keeping it untouched until every pre-feed allocation
          // succeeds) and restore the intact prior-phase join on a rejected
          // rebuild so the row mutates no join state.
          let stale_native = idx == 0
            && native_planar_u16.as_ref().is_some_and(|join| {
              join.chroma_phase_centered() == Some(!center_sited) && join.next_y() == 0
            });
          let prev_native = if stale_native {
            native_planar_u16.take()
          } else {
            None
          };
          let native_result = yuv_planar16_process_native::<BITS, BE>(
            plan,
            native_planar_u16,
            resample_outputs,
            rgb,
            rgba,
            rgb_u16,
            rgba_u16,
            luma,
            // The high-bit planar 4:2:2 family exposes no `luma_u16` output.
            &mut None,
            hsv,
            rgb_scratch,
            rgb_scratch_u16,
            y,
            u_half,
            v_half,
            matrix,
            full_range,
            idx,
            w,
            h,
            1,
            w / 2,
            || {
              ResamplePlan::area_chroma_422(
                w / 2,
                h,
                plan.out_w(),
                plan.out_h(),
                chroma_h_phase,
                0.0,
              )
            },
            use_simd,
          );
          // Restore the taken stale-phase join if the delegate's rebuild was
          // rejected at any pre-feed step: it leaves the field `None` on such a
          // failure, so restoring the intact prior-phase join leaves the
          // rejected row mutating no join state. A non-stale row took nothing.
          if stale_native && native_result.is_err() {
            *native_planar_u16 = prev_native;
          }
          native_result?;
          if frozen_native_route.is_none() && need_output {
            *frozen_native_route = Some(true);
          }
          // RFC #238 S5a: freeze the siting on the same accepted output row.
          if frozen_chroma_centered.is_none() && need_output {
            *frozen_chroma_centered = Some(center_sited);
          }
          return Ok(());
        }
        InsertionPoint::EncodedOutput => {
          if center_sited && want_color {
            // Centered row-stage: reconstruct full-width `u16` chroma AFTER the
            // resample preflight (frozen-output + sequence), so an
            // out-of-sequence / rejected row is caught before the chroma
            // reservation (#180). `packed_yuv422_triple_resample` re-runs the
            // idempotent preflight and owns the transactional commit.
            let expected = if luma.is_some() {
              luma_stream_u16.as_ref().map_or(0, |s| s.next_y())
            } else if rgb.is_some() || rgba.is_some() || hsv.is_some() {
              rgb_stream.as_ref().map_or(0, |s| s.next_y())
            } else {
              rgb_stream_u16.as_ref().map_or(0, |s| s.next_y())
            };
            if let core::ops::ControlFlow::Break(()) = resample_preflight_check_only(
              resample_outputs,
              luma,
              &None,
              rgb,
              rgba,
              rgb_u16,
              rgba_u16,
              &None,
              &None,
              &None,
              &None,
              &None,
              hsv,
              &None,
              Some(expected),
              idx,
            )? {
              return Ok(());
            }
            reserve_420_chroma_full_u16(chroma_full_u16, w, h)?;
            let (u_full, v_full) = reconstruct_chroma(
              ChromaU16::<BITS> { big_endian: BE },
              chroma_full_u16,
              u_half,
              v_half,
              w,
            );
            packed_yuv422_triple_resample::<BITS>(
              luma_stream_u16,
              rgb_stream,
              rgb_stream_u16,
              resample_outputs,
              rgb,
              rgba,
              rgb_u16,
              rgba_u16,
              luma,
              &mut None,
              hsv,
              luma_scratch_u16,
              rgb_scratch,
              rgb_scratch_u16,
              w,
              plan,
              idx,
              use_simd,
              matrix,
              full_range,
              |scratch| deinterleave_y_high_bit_masked::<BITS, BE>(y, scratch, w),
              |scratch| {
                yuv444p9_to_rgb_row_endian(
                  y, u_full, v_full, scratch, w, matrix, full_range, use_simd, BE,
                )
              },
              |scratch| {
                yuv444p9_to_rgb_u16_row_endian(
                  y, u_full, v_full, scratch, w, matrix, full_range, use_simd, BE,
                )
              },
            )?;
          } else {
            packed_yuv422_triple_resample::<BITS>(
              luma_stream_u16,
              rgb_stream,
              rgb_stream_u16,
              resample_outputs,
              rgb,
              rgba,
              rgb_u16,
              rgba_u16,
              luma,
              &mut None,
              hsv,
              luma_scratch_u16,
              rgb_scratch,
              rgb_scratch_u16,
              w,
              plan,
              idx,
              use_simd,
              matrix,
              full_range,
              |scratch| deinterleave_y_high_bit_masked::<BITS, BE>(y, scratch, w),
              |scratch| {
                yuv420p9_to_rgb_row_endian(
                  y, u_half, v_half, scratch, w, matrix, full_range, use_simd, BE,
                )
              },
              |scratch| {
                yuv420p9_to_rgb_u16_row_endian(
                  y, u_half, v_half, scratch, w, matrix, full_range, use_simd, BE,
                )
              },
            )?;
          }
          if frozen_native_route.is_none() && need_output {
            *frozen_native_route = Some(false);
          }
          // RFC #238 S5a: freeze the siting on the same accepted output row.
          if frozen_chroma_centered.is_none() && need_output {
            *frozen_chroma_centered = Some(center_sited);
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

    // Resolve the FULL output set up front so BOTH the no-output guard and the
    // atomicity preflight below run before any row-offset arithmetic.
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();
    let want_rgb_u16 = rgb_u16.is_some();
    let want_rgba_u16 = rgba_u16.is_some();

    // Repo-wide no-output invariant: a `process` call carrying NO output runs
    // NOTHING — no per-row offset arithmetic, no allocation, no state mutation.
    // Returning HERE, before the `idx * w` offsets below, also keeps the invariant
    // overflow-safe: a no-output call never ran an attach-time `w x h x 1`
    // validation, so `idx * w` could overflow `usize` on a 32-bit target with
    // absurd geometry; the guard skips that math (and the centered chroma
    // reservation) entirely. Mirrors the 8-bit Yuv422p sibling. (The high-bit
    // planar 4:2:2 family exposes no `luma_u16` output.)
    let need_output =
      want_rgb || want_rgba || want_hsv || want_rgb_u16 || want_rgba_u16 || luma.is_some();
    if !need_output {
      return Ok(());
    }

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // Chroma siting (#302): the centered horizontal sitings reconstruct chroma at
    // the phase-0.5 position then decode via the 4:4:4 kernels; the default /
    // co-sited path keeps the byte-identical decode (the fused high-bit 4:2:2
    // kernels upsample chroma in-register). 4:2:2 is subsampled horizontally only
    // — there is no vertical blend or chroma lookback (cf. the 4:2:0 sibling).
    let center_sited = chroma_422_center_sited_h(chroma_location);

    // Atomicity preflight (#302 / #308 / #314, cf. the crate's #180 resample fix
    // and the high-bit 4:2:0 sibling): reserve EVERY fallible row scratch this
    // identity row can touch BEFORE any output row is written (the luma plane
    // below, then the u16 / u8 RGB / RGBA / HSV fan-out), so an allocator refusal
    // returns a typed `AllocationFailed` leaving the output frame untouched rather
    // than partially mutated. Two scratches can grow:
    //  1. the centered-siting full-width `u16` chroma (`chroma_full_u16`), needed
    //     by ANY colour output (u8 OR u16 RGB / RGBA / HSV); and
    //  2. the u8 RGB row buffer, reached exactly when a colour decode needs an RGB
    //     row but no caller RGB buffer is borrowable — `want_hsv && want_rgba &&
    //     !want_rgb` (`rgb_row_buf_or_scratch`'s own scratch arm). The u16 RGB /
    //     RGBA outputs write straight into their caller buffers (the rgb_u16 plane
    //     itself stages the rgba_u16 expand) and never grow a scratch; this format
    //     exposes no luma_u16 output.
    // The later `reconstruct_chroma` / `rgb_row_buf_or_scratch` calls
    // reuse the already-sized buffers, so the default path is byte-identical; only
    // the failure-path ordering changes.
    let need_centered_chroma =
      center_sited && (want_rgb || want_rgba || want_hsv || want_rgb_u16 || want_rgba_u16);
    if need_centered_chroma {
      reserve_420_chroma_full_u16(chroma_full_u16, w, h)?;
    }
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

    // Centered full-width chroma (phase-0.5), reconstructed ONCE per row from the
    // wire-format half-width U / V and reused by every colour decode (u16 and u8).
    // Infallible — the scratch was reserved above. The default / co-sited siting
    // leaves it `None`, so the fused 4:2:2 kernels upsample chroma in-register and
    // the output stays byte-identical.
    let centered = if need_centered_chroma {
      Some(reconstruct_chroma(
        ChromaU16::<BITS> { big_endian: BE },
        chroma_full_u16,
        row.u_half(),
        row.v_half(),
        w,
      ))
    } else {
      None
    };

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
    if want_rgba_u16 && !want_rgb_u16 {
      let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row =
        rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
      if let Some((u_full, v_full)) = centered {
        yuv444p9_to_rgba_u16_row_endian(
          row.y(),
          u_full,
          v_full,
          rgba_u16_row,
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
          BE,
        );
      } else {
        yuv420p9_to_rgba_u16_row_endian(
          row.y(),
          row.u_half(),
          row.v_half(),
          rgba_u16_row,
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
          BE,
        );
      }
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
      if let Some((u_full, v_full)) = centered {
        yuv444p9_to_rgb_u16_row_endian(
          row.y(),
          u_full,
          v_full,
          rgb_u16_row,
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
          BE,
        );
      } else {
        yuv420p9_to_rgb_u16_row_endian(
          row.y(),
          row.u_half(),
          row.v_half(),
          rgb_u16_row,
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
          BE,
        );
      }
      if want_rgba_u16 {
        let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
        let rgba_u16_row =
          rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
        expand_rgb_u16_to_rgba_u16_row::<BITS>(rgb_u16_row, rgba_u16_row, w);
      }
    }

    // ===== u8 RGB / RGBA / HSV path (Strategy A) =====
    // HSV-without-RGB-or-RGBA goes through the direct `yuv420p9_to_hsv_row_endian`
    // kernel (no source-width RGB scratch — the SIMD path stages a fixed
    // 8-bit RGB chunk internally). RGB or RGBA also attached keeps the
    // convert-once-then-derive path alive via `need_rgb_kernel`.
    let want_hsv_direct = want_hsv && !want_rgb && !want_rgba;
    let need_rgb_kernel = want_rgb || (want_hsv && want_rgba);

    if want_hsv_direct {
      let hsv = hsv.as_mut().expect("want_hsv_direct implies hsv attached");
      let (h, s, v) = hsv.hsv();
      if let Some((u_full, v_full)) = centered {
        yuv444p9_to_hsv_row_endian(
          row.y(),
          u_full,
          v_full,
          &mut h[one_plane_start..one_plane_end],
          &mut s[one_plane_start..one_plane_end],
          &mut v[one_plane_start..one_plane_end],
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
          BE,
        );
      } else {
        yuv420p9_to_hsv_row_endian(
          row.y(),
          row.u_half(),
          row.v_half(),
          &mut h[one_plane_start..one_plane_end],
          &mut s[one_plane_start..one_plane_end],
          &mut v[one_plane_start..one_plane_end],
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
          BE,
        );
      }
      return Ok(());
    }

    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      if let Some((u_full, v_full)) = centered {
        yuv444p9_to_rgba_row_endian(
          row.y(),
          u_full,
          v_full,
          rgba_row,
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
          BE,
        );
      } else {
        yuv420p9_to_rgba_row_endian(
          row.y(),
          row.u_half(),
          row.v_half(),
          rgba_row,
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
          BE,
        );
      }
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

    if let Some((u_full, v_full)) = centered {
      yuv444p9_to_rgb_row_endian(
        row.y(),
        u_full,
        v_full,
        rgb_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
        BE,
      );
    } else {
      yuv420p9_to_rgb_row_endian(
        row.y(),
        row.u_half(),
        row.v_half(),
        rgb_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
        BE,
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

    if let Some(buf) = rgba.as_deref_mut() {
      let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
    }

    Ok(())
  }
}

// ---- Yuv422p10 / 12 / 14 impl ------------------------------------------
//
// 4:2:2 is 4:2:0's vertical-axis twin at each bit depth: same per-row
// chroma shape (half-width U / V samples, one pair per Y pair), just
// one chroma row per Y row instead of one per two. These impls reuse
// `yuv420p10_to_rgb_*` / `yuv420p12_to_rgb_*` / `yuv420p14_to_rgb_*`
// verbatim — no new row kernels.

impl<'a, R, const BE: bool> MixedSinker<'a, Yuv422p10<BE>, R> {
  /// Attaches a packed **`u16`** RGB output buffer. 10-bit low-packed
  /// values (`(1 << 10) - 1 = 1023` max).
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

  /// Attaches a packed **8‑bit** RGBA output buffer. The 10‑bit YUV
  /// source is converted to 8‑bit RGBA via the `BITS = 10` Q15 kernel
  /// family; alpha = `0xFF` (Yuv422p10 has no alpha plane).
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

  /// Attaches a packed **`u16`** RGBA output buffer. 10‑bit
  /// low‑packed (`(1 << 10) - 1 = 1023` max). Length is measured in
  /// `u16` **elements** (`width x height x 4`). Alpha element is
  /// `(1 << 10) - 1`.
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

impl<R, const BE: bool> Yuv422p10Sink<BE> for MixedSinker<'_, Yuv422p10<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, Yuv422p10<BE>, R> {
  type Input<'r> = Yuv422p10Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::WidthAlignment(WidthAlignment::odd(
        self.width,
      )));
    }
    check_dimensions_match(self.width, self.height, width, height)?;
    reset_high_bit_yuv_streams(self);
    Ok(())
  }

  fn process(&mut self, row: Yuv422p10Row<'_>) -> Result<(), Self::Error> {
    const BITS: u32 = 10;
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if w & 1 != 0 {
      return Err(MixedSinkerError::WidthAlignment(WidthAlignment::odd(w)));
    }
    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Y10,
        idx,
        w,
        row.y().len(),
      )));
    }
    if row.u_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::UHalf10,
        idx,
        w / 2,
        row.u_half().len(),
      )));
    }
    if row.v_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::VHalf10,
        idx,
        w / 2,
        row.v_half().len(),
      )));
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange(
        RowIndexOutOfRange::new(idx, self.height),
      ));
    }

    // Chroma siting (#302): drives the identity-plan horizontal chroma phase.
    // `Copy`, so read it out before the field split-borrow below.
    let chroma_location = self.chroma_location;

    let Self {
      rgb,
      rgb_u16,
      rgba,
      rgba_u16,
      luma,
      hsv,
      rgb_scratch,
      rgb_scratch_u16,
      chroma_full_u16,
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
      frozen_chroma_centered,
      ..
    } = self;

    // Non-identity plan. A `Filter` plan routes to the signed-coefficient
    // filter twin of the row-stage tier BEFORE the native/row-stage route
    // machinery (the native fast tier is an area-specific optimization that
    // never sees a filter plan). For an `Area` plan the native tier bins the
    // host-native Y / U / V planes at output resolution and converts ONCE per
    // output row at output width (4:4:4 kernels); the row-stage tier
    // ([`packed_yuv422_triple_resample`]) converts each source row at source
    // width then bins (u8 color, independent native-u16 color, native Y).
    // `with_native(false)` forces the latter; the route is frozen per frame.
    // The half-width U / V planes are horizontally upsampled in-register by the
    // shared 4:2:0 row kernels (4:2:0 and 4:2:2 have the identical per-row
    // chroma contract) for the row-stage / filter tails. Yuv422p exposes no
    // `luma_u16` output, so it is `&mut None` and only `luma` (binned native Y
    // `>> (BITS - 8)`) is emitted. The filter tail clamps every sub-16-bit
    // colour sample AND the native Y to `(1 << BITS) - 1`; the native tier's
    // colour clamp lives in its convert kernels.
    if let Some(plan) = plan.as_ref() {
      let matrix = row.matrix();
      let full_range = row.full_range();
      let (y, u_half, v_half) = (row.y(), row.u_half(), row.v_half());
      // RFC #238 S5a — 4:2:2 horizontal chroma siting. The centered group
      // (`Center` / `Top` / `Bottom`, `chroma_422_center_sited_h`) samples
      // chroma at `+0.5` luma = `+0.25` chroma-sample; the co-sited /
      // unspecified group is phase 0 (today's byte-identical decode). Siting
      // enters the chroma RECONSTRUCTION only — the averaging tier is still
      // chosen by `select_insertion_point` below — so the native fast tier
      // folds the phase into the `area_chroma_422` chroma weights while the
      // row-stage and filter tiers reconstruct full-width `u16` chroma and
      // decode 4:4:4.
      let center_sited = chroma_422_center_sited_h(chroma_location);
      let chroma_h_phase = if center_sited {
        YUV422P_CENTERED_H_PHASE
      } else {
        0.0
      };
      let need_output = luma.is_some()
        || rgb.is_some()
        || rgba.is_some()
        || hsv.is_some()
        || rgb_u16.is_some()
        || rgba_u16.is_some();
      // Only the colour tiers reconstruct full-width chroma for the centered
      // decode; a luma-only centered row bins native Y unchanged (siting is a
      // chroma-only property).
      let want_color =
        rgb.is_some() || rgba.is_some() || hsv.is_some() || rgb_u16.is_some() || rgba_u16.is_some();
      // Freeze the effective 4:2:2 chroma siting on the first output-bearing
      // row (mirrors the `frozen_native_route` freeze below). This CHECK is at
      // the always-compiled choke point every tier passes through; the matching
      // SET rides each tier's accept path (never before dispatch, so a rejected
      // row leaves it unset for a corrected retry). A later row observing a
      // different phase would bin a mixture of co-sited and centered chroma, so
      // it is rejected here before any reconstruction.
      if need_output
        && let Some(frozen) = *frozen_chroma_centered
        && frozen != center_sited
      {
        return Err(MixedSinkerError::ChromaSitingChanged(
          ChromaSitingChanged::new(idx),
        ));
      }
      if plan.kind().is_filter() {
        // Reject a multi-kernel (BICUBLIN) plan BEFORE the centered reserve
        // below — the delegate's first act is this same check, so hoisting it
        // keeps a rejected filter plan from reserving / reconstructing chroma
        // first (the #180 reject-before-allocation invariant). Idempotent — the
        // delegate re-runs it.
        plan.ensure_single_kernel_filter()?;
        if center_sited && want_color {
          // Centered filter: reconstruct full-width `u16` chroma, but ONLY after
          // the resample preflight (frozen-output + sequence), so an
          // out-of-sequence / rejected row is caught before the chroma
          // reservation (#180). `packed_yuv422_triple_filter_resample` re-runs
          // the idempotent preflight and owns the transactional commit.
          let expected = if luma.is_some() {
            luma_filter_stream_u16.as_ref().map_or(0, |s| s.next_y())
          } else if rgb.is_some() || rgba.is_some() || hsv.is_some() {
            rgb_filter_stream.as_ref().map_or(0, |s| s.next_y())
          } else {
            rgb_filter_stream_u16.as_ref().map_or(0, |s| s.next_y())
          };
          if let core::ops::ControlFlow::Break(()) = resample_preflight_check_only(
            resample_outputs,
            luma,
            &None,
            rgb,
            rgba,
            rgb_u16,
            rgba_u16,
            &None,
            &None,
            &None,
            &None,
            &None,
            hsv,
            &None,
            Some(expected),
            idx,
          )? {
            return Ok(());
          }
          reserve_420_chroma_full_u16(chroma_full_u16, w, h)?;
          let (u_full, v_full) = reconstruct_chroma(
            ChromaU16::<BITS> { big_endian: BE },
            chroma_full_u16,
            u_half,
            v_half,
            w,
          );
          let r = packed_yuv422_triple_filter_resample::<BITS>(
            luma_filter_stream_u16,
            rgb_filter_stream,
            rgb_filter_stream_u16,
            resample_outputs,
            rgb,
            rgba,
            rgb_u16,
            rgba_u16,
            luma,
            &mut None,
            hsv,
            luma_scratch_u16,
            rgb_scratch,
            rgb_scratch_u16,
            w,
            plan,
            idx,
            use_simd,
            matrix,
            full_range,
            |scratch| deinterleave_y_high_bit_masked::<BITS, BE>(y, scratch, w),
            |scratch| {
              yuv444p10_to_rgb_row_endian(
                y, u_full, v_full, scratch, w, matrix, full_range, use_simd, BE,
              )
            },
            |scratch| {
              yuv444p10_to_rgb_u16_row_endian(
                y, u_full, v_full, scratch, w, matrix, full_range, use_simd, BE,
              )
            },
          );
          if r.is_ok() && need_output && frozen_chroma_centered.is_none() {
            *frozen_chroma_centered = Some(center_sited);
          }
          return r;
        }
        let r = packed_yuv422_triple_filter_resample::<BITS>(
          luma_filter_stream_u16,
          rgb_filter_stream,
          rgb_filter_stream_u16,
          resample_outputs,
          rgb,
          rgba,
          rgb_u16,
          rgba_u16,
          luma,
          &mut None,
          hsv,
          luma_scratch_u16,
          rgb_scratch,
          rgb_scratch_u16,
          w,
          plan,
          idx,
          use_simd,
          matrix,
          full_range,
          |scratch| deinterleave_y_high_bit_masked::<BITS, BE>(y, scratch, w),
          |scratch| {
            yuv420p10_to_rgb_row_endian(
              y, u_half, v_half, scratch, w, matrix, full_range, use_simd, BE,
            )
          },
          |scratch| {
            yuv420p10_to_rgb_u16_row_endian(
              y, u_half, v_half, scratch, w, matrix, full_range, use_simd, BE,
            )
          },
        );
        if r.is_ok() && need_output && frozen_chroma_centered.is_none() {
          *frozen_chroma_centered = Some(center_sited);
        }
        return r;
      }
      // Native / row-stage route split — see the high-bit 4:2:0 Yuv420p impl
      // for the CHECK-before / SET-after `frozen_native_route` contract.
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
          native_eligible: YUV422P_HIGH_BIT_NATIVE_ELIGIBLE,
          with_native: *native,
          area_plan: true,
        },
      );
      match insertion {
        InsertionPoint::NativeCodes => {
          // 4:2:2: chroma `w/2 x h` — half width, full height; a chroma row per
          // Y row (`chroma_vsub = 1`, `chroma_w = w/2`), chroma plan the phased
          // `area_chroma_422` (h-phase `0.25` chroma-sample when centered, else
          // the plain box). RFC #238 point-of-use siting invalidation: a reused
          // sink's cached join is only `reset` between frames, so a frame whose
          // `chroma_location` moved to a different phase must REBUILD it. Drop
          // the stale-phase join ONLY on the in-sequence first row of a fresh
          // frame (`idx == 0`, `next_y() == 0`) so a mid-frame / out-of-sequence
          // row rejects against the INTACT join and a corrected retry rebuilds
          // cleanly; a luma-only join carries no chroma phase and is never
          // dropped. Move it OUT (the delegate builds the replacement into the
          // field, keeping it untouched until every pre-feed allocation
          // succeeds) and restore the intact prior-phase join on a rejected
          // rebuild so the row mutates no join state.
          let stale_native = idx == 0
            && native_planar_u16.as_ref().is_some_and(|join| {
              join.chroma_phase_centered() == Some(!center_sited) && join.next_y() == 0
            });
          let prev_native = if stale_native {
            native_planar_u16.take()
          } else {
            None
          };
          let native_result = yuv_planar16_process_native::<BITS, BE>(
            plan,
            native_planar_u16,
            resample_outputs,
            rgb,
            rgba,
            rgb_u16,
            rgba_u16,
            luma,
            // The high-bit planar 4:2:2 family exposes no `luma_u16` output.
            &mut None,
            hsv,
            rgb_scratch,
            rgb_scratch_u16,
            y,
            u_half,
            v_half,
            matrix,
            full_range,
            idx,
            w,
            h,
            1,
            w / 2,
            || {
              ResamplePlan::area_chroma_422(
                w / 2,
                h,
                plan.out_w(),
                plan.out_h(),
                chroma_h_phase,
                0.0,
              )
            },
            use_simd,
          );
          // Restore the taken stale-phase join if the delegate's rebuild was
          // rejected at any pre-feed step: it leaves the field `None` on such a
          // failure, so restoring the intact prior-phase join leaves the
          // rejected row mutating no join state. A non-stale row took nothing.
          if stale_native && native_result.is_err() {
            *native_planar_u16 = prev_native;
          }
          native_result?;
          if frozen_native_route.is_none() && need_output {
            *frozen_native_route = Some(true);
          }
          // RFC #238 S5a: freeze the siting on the same accepted output row.
          if frozen_chroma_centered.is_none() && need_output {
            *frozen_chroma_centered = Some(center_sited);
          }
          return Ok(());
        }
        InsertionPoint::EncodedOutput => {
          if center_sited && want_color {
            // Centered row-stage: reconstruct full-width `u16` chroma AFTER the
            // resample preflight (frozen-output + sequence), so an
            // out-of-sequence / rejected row is caught before the chroma
            // reservation (#180). `packed_yuv422_triple_resample` re-runs the
            // idempotent preflight and owns the transactional commit.
            let expected = if luma.is_some() {
              luma_stream_u16.as_ref().map_or(0, |s| s.next_y())
            } else if rgb.is_some() || rgba.is_some() || hsv.is_some() {
              rgb_stream.as_ref().map_or(0, |s| s.next_y())
            } else {
              rgb_stream_u16.as_ref().map_or(0, |s| s.next_y())
            };
            if let core::ops::ControlFlow::Break(()) = resample_preflight_check_only(
              resample_outputs,
              luma,
              &None,
              rgb,
              rgba,
              rgb_u16,
              rgba_u16,
              &None,
              &None,
              &None,
              &None,
              &None,
              hsv,
              &None,
              Some(expected),
              idx,
            )? {
              return Ok(());
            }
            reserve_420_chroma_full_u16(chroma_full_u16, w, h)?;
            let (u_full, v_full) = reconstruct_chroma(
              ChromaU16::<BITS> { big_endian: BE },
              chroma_full_u16,
              u_half,
              v_half,
              w,
            );
            packed_yuv422_triple_resample::<BITS>(
              luma_stream_u16,
              rgb_stream,
              rgb_stream_u16,
              resample_outputs,
              rgb,
              rgba,
              rgb_u16,
              rgba_u16,
              luma,
              &mut None,
              hsv,
              luma_scratch_u16,
              rgb_scratch,
              rgb_scratch_u16,
              w,
              plan,
              idx,
              use_simd,
              matrix,
              full_range,
              |scratch| deinterleave_y_high_bit_masked::<BITS, BE>(y, scratch, w),
              |scratch| {
                yuv444p10_to_rgb_row_endian(
                  y, u_full, v_full, scratch, w, matrix, full_range, use_simd, BE,
                )
              },
              |scratch| {
                yuv444p10_to_rgb_u16_row_endian(
                  y, u_full, v_full, scratch, w, matrix, full_range, use_simd, BE,
                )
              },
            )?;
          } else {
            packed_yuv422_triple_resample::<BITS>(
              luma_stream_u16,
              rgb_stream,
              rgb_stream_u16,
              resample_outputs,
              rgb,
              rgba,
              rgb_u16,
              rgba_u16,
              luma,
              &mut None,
              hsv,
              luma_scratch_u16,
              rgb_scratch,
              rgb_scratch_u16,
              w,
              plan,
              idx,
              use_simd,
              matrix,
              full_range,
              |scratch| deinterleave_y_high_bit_masked::<BITS, BE>(y, scratch, w),
              |scratch| {
                yuv420p10_to_rgb_row_endian(
                  y, u_half, v_half, scratch, w, matrix, full_range, use_simd, BE,
                )
              },
              |scratch| {
                yuv420p10_to_rgb_u16_row_endian(
                  y, u_half, v_half, scratch, w, matrix, full_range, use_simd, BE,
                )
              },
            )?;
          }
          if frozen_native_route.is_none() && need_output {
            *frozen_native_route = Some(false);
          }
          // RFC #238 S5a: freeze the siting on the same accepted output row.
          if frozen_chroma_centered.is_none() && need_output {
            *frozen_chroma_centered = Some(center_sited);
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

    // Resolve the FULL output set up front so BOTH the no-output guard and the
    // atomicity preflight below run before any row-offset arithmetic.
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();
    let want_rgb_u16 = rgb_u16.is_some();
    let want_rgba_u16 = rgba_u16.is_some();

    // Repo-wide no-output invariant: a `process` call carrying NO output runs
    // NOTHING — no per-row offset arithmetic, no allocation, no state mutation.
    // Returning HERE, before the `idx * w` offsets below, also keeps the invariant
    // overflow-safe: a no-output call never ran an attach-time `w x h x 1`
    // validation, so `idx * w` could overflow `usize` on a 32-bit target with
    // absurd geometry; the guard skips that math (and the centered chroma
    // reservation) entirely. Mirrors the 8-bit Yuv422p sibling. (The high-bit
    // planar 4:2:2 family exposes no `luma_u16` output.)
    let need_output =
      want_rgb || want_rgba || want_hsv || want_rgb_u16 || want_rgba_u16 || luma.is_some();
    if !need_output {
      return Ok(());
    }

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // Chroma siting (#302): the centered horizontal sitings reconstruct chroma at
    // the phase-0.5 position then decode via the 4:4:4 kernels; the default /
    // co-sited path keeps the byte-identical decode (the fused high-bit 4:2:2
    // kernels upsample chroma in-register). 4:2:2 is subsampled horizontally only
    // — there is no vertical blend or chroma lookback (cf. the 4:2:0 sibling).
    let center_sited = chroma_422_center_sited_h(chroma_location);

    // Atomicity preflight (#302 / #308 / #314, cf. the crate's #180 resample fix
    // and the high-bit 4:2:0 sibling): reserve EVERY fallible row scratch this
    // identity row can touch BEFORE any output row is written (the luma plane
    // below, then the u16 / u8 RGB / RGBA / HSV fan-out), so an allocator refusal
    // returns a typed `AllocationFailed` leaving the output frame untouched rather
    // than partially mutated. Two scratches can grow:
    //  1. the centered-siting full-width `u16` chroma (`chroma_full_u16`), needed
    //     by ANY colour output (u8 OR u16 RGB / RGBA / HSV); and
    //  2. the u8 RGB row buffer, reached exactly when a colour decode needs an RGB
    //     row but no caller RGB buffer is borrowable — `want_hsv && want_rgba &&
    //     !want_rgb` (`rgb_row_buf_or_scratch`'s own scratch arm). The u16 RGB /
    //     RGBA outputs write straight into their caller buffers (the rgb_u16 plane
    //     itself stages the rgba_u16 expand) and never grow a scratch; this format
    //     exposes no luma_u16 output.
    // The later `reconstruct_chroma` / `rgb_row_buf_or_scratch` calls
    // reuse the already-sized buffers, so the default path is byte-identical; only
    // the failure-path ordering changes.
    let need_centered_chroma =
      center_sited && (want_rgb || want_rgba || want_hsv || want_rgb_u16 || want_rgba_u16);
    if need_centered_chroma {
      reserve_420_chroma_full_u16(chroma_full_u16, w, h)?;
    }
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

    // Centered full-width chroma (phase-0.5), reconstructed ONCE per row from the
    // wire-format half-width U / V and reused by every colour decode (u16 and u8).
    // Infallible — the scratch was reserved above. The default / co-sited siting
    // leaves it `None`, so the fused 4:2:2 kernels upsample chroma in-register and
    // the output stays byte-identical.
    let centered = if need_centered_chroma {
      Some(reconstruct_chroma(
        ChromaU16::<BITS> { big_endian: BE },
        chroma_full_u16,
        row.u_half(),
        row.v_half(),
        w,
      ))
    } else {
      None
    };

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
    if want_rgba_u16 && !want_rgb_u16 {
      let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row =
        rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
      if let Some((u_full, v_full)) = centered {
        yuv444p10_to_rgba_u16_row_endian(
          row.y(),
          u_full,
          v_full,
          rgba_u16_row,
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
          BE,
        );
      } else {
        yuv420p10_to_rgba_u16_row_endian(
          row.y(),
          row.u_half(),
          row.v_half(),
          rgba_u16_row,
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
          BE,
        );
      }
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
      if let Some((u_full, v_full)) = centered {
        yuv444p10_to_rgb_u16_row_endian(
          row.y(),
          u_full,
          v_full,
          rgb_u16_row,
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
          BE,
        );
      } else {
        yuv420p10_to_rgb_u16_row_endian(
          row.y(),
          row.u_half(),
          row.v_half(),
          rgb_u16_row,
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
          BE,
        );
      }
      if want_rgba_u16 {
        let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
        let rgba_u16_row =
          rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
        expand_rgb_u16_to_rgba_u16_row::<BITS>(rgb_u16_row, rgba_u16_row, w);
      }
    }

    // ===== u8 RGB / RGBA / HSV path (Strategy A) =====
    // HSV-without-RGB-or-RGBA goes through the direct `yuv420p10_to_hsv_row_endian`
    // kernel (no source-width RGB scratch — the SIMD path stages a fixed
    // 8-bit RGB chunk internally). RGB or RGBA also attached keeps the
    // convert-once-then-derive path alive via `need_rgb_kernel`.
    let want_hsv_direct = want_hsv && !want_rgb && !want_rgba;
    let need_rgb_kernel = want_rgb || (want_hsv && want_rgba);

    if want_hsv_direct {
      let hsv = hsv.as_mut().expect("want_hsv_direct implies hsv attached");
      let (h, s, v) = hsv.hsv();
      if let Some((u_full, v_full)) = centered {
        yuv444p10_to_hsv_row_endian(
          row.y(),
          u_full,
          v_full,
          &mut h[one_plane_start..one_plane_end],
          &mut s[one_plane_start..one_plane_end],
          &mut v[one_plane_start..one_plane_end],
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
          BE,
        );
      } else {
        yuv420p10_to_hsv_row_endian(
          row.y(),
          row.u_half(),
          row.v_half(),
          &mut h[one_plane_start..one_plane_end],
          &mut s[one_plane_start..one_plane_end],
          &mut v[one_plane_start..one_plane_end],
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
          BE,
        );
      }
      return Ok(());
    }

    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      if let Some((u_full, v_full)) = centered {
        yuv444p10_to_rgba_row_endian(
          row.y(),
          u_full,
          v_full,
          rgba_row,
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
          BE,
        );
      } else {
        yuv420p10_to_rgba_row_endian(
          row.y(),
          row.u_half(),
          row.v_half(),
          rgba_row,
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
          BE,
        );
      }
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

    if let Some((u_full, v_full)) = centered {
      yuv444p10_to_rgb_row_endian(
        row.y(),
        u_full,
        v_full,
        rgb_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
        BE,
      );
    } else {
      yuv420p10_to_rgb_row_endian(
        row.y(),
        row.u_half(),
        row.v_half(),
        rgb_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
        BE,
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

    if let Some(buf) = rgba.as_deref_mut() {
      let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
    }

    Ok(())
  }
}

impl<'a, R, const BE: bool> MixedSinker<'a, Yuv422p12<BE>, R> {
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

  /// Attaches a packed **8‑bit** RGBA output buffer. The 12‑bit YUV
  /// source is converted to 8‑bit RGBA via the `BITS = 12` Q15 kernel
  /// family; alpha = `0xFF` (Yuv422p12 has no alpha plane).
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

  /// Attaches a packed **`u16`** RGBA output buffer. 12‑bit
  /// low‑packed (`(1 << 12) - 1 = 4095` max). Length is measured in
  /// `u16` **elements** (`width x height x 4`). Alpha element is
  /// `(1 << 12) - 1`.
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

impl<R, const BE: bool> Yuv422p12Sink<BE> for MixedSinker<'_, Yuv422p12<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, Yuv422p12<BE>, R> {
  type Input<'r> = Yuv422p12Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::WidthAlignment(WidthAlignment::odd(
        self.width,
      )));
    }
    check_dimensions_match(self.width, self.height, width, height)?;
    reset_high_bit_yuv_streams(self);
    Ok(())
  }

  fn process(&mut self, row: Yuv422p12Row<'_>) -> Result<(), Self::Error> {
    const BITS: u32 = 12;
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if w & 1 != 0 {
      return Err(MixedSinkerError::WidthAlignment(WidthAlignment::odd(w)));
    }
    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Y12,
        idx,
        w,
        row.y().len(),
      )));
    }
    if row.u_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::UHalf12,
        idx,
        w / 2,
        row.u_half().len(),
      )));
    }
    if row.v_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::VHalf12,
        idx,
        w / 2,
        row.v_half().len(),
      )));
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange(
        RowIndexOutOfRange::new(idx, self.height),
      ));
    }

    // Chroma siting (#302): drives the identity-plan horizontal chroma phase.
    // `Copy`, so read it out before the field split-borrow below.
    let chroma_location = self.chroma_location;

    let Self {
      rgb,
      rgb_u16,
      rgba,
      rgba_u16,
      luma,
      hsv,
      rgb_scratch,
      rgb_scratch_u16,
      chroma_full_u16,
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
      frozen_chroma_centered,
      ..
    } = self;

    // Non-identity plan: feed the shared high-bit 4:2:2 triple-resample
    // tail (u8 color, independent native-u16 color, native Y). The half-
    // width U / V planes are horizontally upsampled in-register by the
    // shared 4:2:0 row kernels (4:2:0 and 4:2:2 have the identical per-row
    // chroma contract). Yuv422p exposes no `luma_u16` output, so it is
    // `&mut None` and only `luma` (binned native Y `>> (BITS - 8)`) is
    // emitted. The span kind picks the engine (area bin or signed-coefficient
    // filter twin) — see the Yuv422p10 impl for the full rationale; the
    // filter tail clamps every sub-16-bit colour sample AND the native Y to
    // `(1 << BITS) - 1`.
    if let Some(plan) = plan.as_ref() {
      let matrix = row.matrix();
      let full_range = row.full_range();
      let (y, u_half, v_half) = (row.y(), row.u_half(), row.v_half());
      // RFC #238 S5a — 4:2:2 horizontal chroma siting. The centered group
      // (`Center` / `Top` / `Bottom`, `chroma_422_center_sited_h`) samples
      // chroma at `+0.5` luma = `+0.25` chroma-sample; the co-sited /
      // unspecified group is phase 0 (today's byte-identical decode). Siting
      // enters the chroma RECONSTRUCTION only — the averaging tier is still
      // chosen by `select_insertion_point` below — so the native fast tier
      // folds the phase into the `area_chroma_422` chroma weights while the
      // row-stage and filter tiers reconstruct full-width `u16` chroma and
      // decode 4:4:4.
      let center_sited = chroma_422_center_sited_h(chroma_location);
      let chroma_h_phase = if center_sited {
        YUV422P_CENTERED_H_PHASE
      } else {
        0.0
      };
      let need_output = luma.is_some()
        || rgb.is_some()
        || rgba.is_some()
        || hsv.is_some()
        || rgb_u16.is_some()
        || rgba_u16.is_some();
      // Only the colour tiers reconstruct full-width chroma for the centered
      // decode; a luma-only centered row bins native Y unchanged (siting is a
      // chroma-only property).
      let want_color =
        rgb.is_some() || rgba.is_some() || hsv.is_some() || rgb_u16.is_some() || rgba_u16.is_some();
      // Freeze the effective 4:2:2 chroma siting on the first output-bearing
      // row (mirrors the `frozen_native_route` freeze below). This CHECK is at
      // the always-compiled choke point every tier passes through; the matching
      // SET rides each tier's accept path (never before dispatch, so a rejected
      // row leaves it unset for a corrected retry). A later row observing a
      // different phase would bin a mixture of co-sited and centered chroma, so
      // it is rejected here before any reconstruction.
      if need_output
        && let Some(frozen) = *frozen_chroma_centered
        && frozen != center_sited
      {
        return Err(MixedSinkerError::ChromaSitingChanged(
          ChromaSitingChanged::new(idx),
        ));
      }
      if plan.kind().is_filter() {
        // Reject a multi-kernel (BICUBLIN) plan BEFORE the centered reserve
        // below — the delegate's first act is this same check, so hoisting it
        // keeps a rejected filter plan from reserving / reconstructing chroma
        // first (the #180 reject-before-allocation invariant). Idempotent — the
        // delegate re-runs it.
        plan.ensure_single_kernel_filter()?;
        if center_sited && want_color {
          // Centered filter: reconstruct full-width `u16` chroma, but ONLY after
          // the resample preflight (frozen-output + sequence), so an
          // out-of-sequence / rejected row is caught before the chroma
          // reservation (#180). `packed_yuv422_triple_filter_resample` re-runs
          // the idempotent preflight and owns the transactional commit.
          let expected = if luma.is_some() {
            luma_filter_stream_u16.as_ref().map_or(0, |s| s.next_y())
          } else if rgb.is_some() || rgba.is_some() || hsv.is_some() {
            rgb_filter_stream.as_ref().map_or(0, |s| s.next_y())
          } else {
            rgb_filter_stream_u16.as_ref().map_or(0, |s| s.next_y())
          };
          if let core::ops::ControlFlow::Break(()) = resample_preflight_check_only(
            resample_outputs,
            luma,
            &None,
            rgb,
            rgba,
            rgb_u16,
            rgba_u16,
            &None,
            &None,
            &None,
            &None,
            &None,
            hsv,
            &None,
            Some(expected),
            idx,
          )? {
            return Ok(());
          }
          reserve_420_chroma_full_u16(chroma_full_u16, w, h)?;
          let (u_full, v_full) = reconstruct_chroma(
            ChromaU16::<BITS> { big_endian: BE },
            chroma_full_u16,
            u_half,
            v_half,
            w,
          );
          let r = packed_yuv422_triple_filter_resample::<BITS>(
            luma_filter_stream_u16,
            rgb_filter_stream,
            rgb_filter_stream_u16,
            resample_outputs,
            rgb,
            rgba,
            rgb_u16,
            rgba_u16,
            luma,
            &mut None,
            hsv,
            luma_scratch_u16,
            rgb_scratch,
            rgb_scratch_u16,
            w,
            plan,
            idx,
            use_simd,
            matrix,
            full_range,
            |scratch| deinterleave_y_high_bit_masked::<BITS, BE>(y, scratch, w),
            |scratch| {
              yuv444p12_to_rgb_row_endian(
                y, u_full, v_full, scratch, w, matrix, full_range, use_simd, BE,
              )
            },
            |scratch| {
              yuv444p12_to_rgb_u16_row_endian(
                y, u_full, v_full, scratch, w, matrix, full_range, use_simd, BE,
              )
            },
          );
          if r.is_ok() && need_output && frozen_chroma_centered.is_none() {
            *frozen_chroma_centered = Some(center_sited);
          }
          return r;
        }
        let r = packed_yuv422_triple_filter_resample::<BITS>(
          luma_filter_stream_u16,
          rgb_filter_stream,
          rgb_filter_stream_u16,
          resample_outputs,
          rgb,
          rgba,
          rgb_u16,
          rgba_u16,
          luma,
          &mut None,
          hsv,
          luma_scratch_u16,
          rgb_scratch,
          rgb_scratch_u16,
          w,
          plan,
          idx,
          use_simd,
          matrix,
          full_range,
          |scratch| deinterleave_y_high_bit_masked::<BITS, BE>(y, scratch, w),
          |scratch| {
            yuv420p12_to_rgb_row_endian(
              y, u_half, v_half, scratch, w, matrix, full_range, use_simd, BE,
            )
          },
          |scratch| {
            yuv420p12_to_rgb_u16_row_endian(
              y, u_half, v_half, scratch, w, matrix, full_range, use_simd, BE,
            )
          },
        );
        if r.is_ok() && need_output && frozen_chroma_centered.is_none() {
          *frozen_chroma_centered = Some(center_sited);
        }
        return r;
      }
      // Native / row-stage route split — see the high-bit 4:2:0 Yuv420p impl
      // for the CHECK-before / SET-after `frozen_native_route` contract.
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
          native_eligible: YUV422P_HIGH_BIT_NATIVE_ELIGIBLE,
          with_native: *native,
          area_plan: true,
        },
      );
      match insertion {
        InsertionPoint::NativeCodes => {
          // 4:2:2: chroma `w/2 x h` — half width, full height; a chroma row per
          // Y row (`chroma_vsub = 1`, `chroma_w = w/2`), chroma plan the phased
          // `area_chroma_422` (h-phase `0.25` chroma-sample when centered, else
          // the plain box). RFC #238 point-of-use siting invalidation: a reused
          // sink's cached join is only `reset` between frames, so a frame whose
          // `chroma_location` moved to a different phase must REBUILD it. Drop
          // the stale-phase join ONLY on the in-sequence first row of a fresh
          // frame (`idx == 0`, `next_y() == 0`) so a mid-frame / out-of-sequence
          // row rejects against the INTACT join and a corrected retry rebuilds
          // cleanly; a luma-only join carries no chroma phase and is never
          // dropped. Move it OUT (the delegate builds the replacement into the
          // field, keeping it untouched until every pre-feed allocation
          // succeeds) and restore the intact prior-phase join on a rejected
          // rebuild so the row mutates no join state.
          let stale_native = idx == 0
            && native_planar_u16.as_ref().is_some_and(|join| {
              join.chroma_phase_centered() == Some(!center_sited) && join.next_y() == 0
            });
          let prev_native = if stale_native {
            native_planar_u16.take()
          } else {
            None
          };
          let native_result = yuv_planar16_process_native::<BITS, BE>(
            plan,
            native_planar_u16,
            resample_outputs,
            rgb,
            rgba,
            rgb_u16,
            rgba_u16,
            luma,
            // The high-bit planar 4:2:2 family exposes no `luma_u16` output.
            &mut None,
            hsv,
            rgb_scratch,
            rgb_scratch_u16,
            y,
            u_half,
            v_half,
            matrix,
            full_range,
            idx,
            w,
            h,
            1,
            w / 2,
            || {
              ResamplePlan::area_chroma_422(
                w / 2,
                h,
                plan.out_w(),
                plan.out_h(),
                chroma_h_phase,
                0.0,
              )
            },
            use_simd,
          );
          // Restore the taken stale-phase join if the delegate's rebuild was
          // rejected at any pre-feed step: it leaves the field `None` on such a
          // failure, so restoring the intact prior-phase join leaves the
          // rejected row mutating no join state. A non-stale row took nothing.
          if stale_native && native_result.is_err() {
            *native_planar_u16 = prev_native;
          }
          native_result?;
          if frozen_native_route.is_none() && need_output {
            *frozen_native_route = Some(true);
          }
          // RFC #238 S5a: freeze the siting on the same accepted output row.
          if frozen_chroma_centered.is_none() && need_output {
            *frozen_chroma_centered = Some(center_sited);
          }
          return Ok(());
        }
        InsertionPoint::EncodedOutput => {
          if center_sited && want_color {
            // Centered row-stage: reconstruct full-width `u16` chroma AFTER the
            // resample preflight (frozen-output + sequence), so an
            // out-of-sequence / rejected row is caught before the chroma
            // reservation (#180). `packed_yuv422_triple_resample` re-runs the
            // idempotent preflight and owns the transactional commit.
            let expected = if luma.is_some() {
              luma_stream_u16.as_ref().map_or(0, |s| s.next_y())
            } else if rgb.is_some() || rgba.is_some() || hsv.is_some() {
              rgb_stream.as_ref().map_or(0, |s| s.next_y())
            } else {
              rgb_stream_u16.as_ref().map_or(0, |s| s.next_y())
            };
            if let core::ops::ControlFlow::Break(()) = resample_preflight_check_only(
              resample_outputs,
              luma,
              &None,
              rgb,
              rgba,
              rgb_u16,
              rgba_u16,
              &None,
              &None,
              &None,
              &None,
              &None,
              hsv,
              &None,
              Some(expected),
              idx,
            )? {
              return Ok(());
            }
            reserve_420_chroma_full_u16(chroma_full_u16, w, h)?;
            let (u_full, v_full) = reconstruct_chroma(
              ChromaU16::<BITS> { big_endian: BE },
              chroma_full_u16,
              u_half,
              v_half,
              w,
            );
            packed_yuv422_triple_resample::<BITS>(
              luma_stream_u16,
              rgb_stream,
              rgb_stream_u16,
              resample_outputs,
              rgb,
              rgba,
              rgb_u16,
              rgba_u16,
              luma,
              &mut None,
              hsv,
              luma_scratch_u16,
              rgb_scratch,
              rgb_scratch_u16,
              w,
              plan,
              idx,
              use_simd,
              matrix,
              full_range,
              |scratch| deinterleave_y_high_bit_masked::<BITS, BE>(y, scratch, w),
              |scratch| {
                yuv444p12_to_rgb_row_endian(
                  y, u_full, v_full, scratch, w, matrix, full_range, use_simd, BE,
                )
              },
              |scratch| {
                yuv444p12_to_rgb_u16_row_endian(
                  y, u_full, v_full, scratch, w, matrix, full_range, use_simd, BE,
                )
              },
            )?;
          } else {
            packed_yuv422_triple_resample::<BITS>(
              luma_stream_u16,
              rgb_stream,
              rgb_stream_u16,
              resample_outputs,
              rgb,
              rgba,
              rgb_u16,
              rgba_u16,
              luma,
              &mut None,
              hsv,
              luma_scratch_u16,
              rgb_scratch,
              rgb_scratch_u16,
              w,
              plan,
              idx,
              use_simd,
              matrix,
              full_range,
              |scratch| deinterleave_y_high_bit_masked::<BITS, BE>(y, scratch, w),
              |scratch| {
                yuv420p12_to_rgb_row_endian(
                  y, u_half, v_half, scratch, w, matrix, full_range, use_simd, BE,
                )
              },
              |scratch| {
                yuv420p12_to_rgb_u16_row_endian(
                  y, u_half, v_half, scratch, w, matrix, full_range, use_simd, BE,
                )
              },
            )?;
          }
          if frozen_native_route.is_none() && need_output {
            *frozen_native_route = Some(false);
          }
          // RFC #238 S5a: freeze the siting on the same accepted output row.
          if frozen_chroma_centered.is_none() && need_output {
            *frozen_chroma_centered = Some(center_sited);
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

    // Resolve the FULL output set up front so BOTH the no-output guard and the
    // atomicity preflight below run before any row-offset arithmetic.
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();
    let want_rgb_u16 = rgb_u16.is_some();
    let want_rgba_u16 = rgba_u16.is_some();

    // Repo-wide no-output invariant: a `process` call carrying NO output runs
    // NOTHING — no per-row offset arithmetic, no allocation, no state mutation.
    // Returning HERE, before the `idx * w` offsets below, also keeps the invariant
    // overflow-safe: a no-output call never ran an attach-time `w x h x 1`
    // validation, so `idx * w` could overflow `usize` on a 32-bit target with
    // absurd geometry; the guard skips that math (and the centered chroma
    // reservation) entirely. Mirrors the 8-bit Yuv422p sibling. (The high-bit
    // planar 4:2:2 family exposes no `luma_u16` output.)
    let need_output =
      want_rgb || want_rgba || want_hsv || want_rgb_u16 || want_rgba_u16 || luma.is_some();
    if !need_output {
      return Ok(());
    }

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // Chroma siting (#302): the centered horizontal sitings reconstruct chroma at
    // the phase-0.5 position then decode via the 4:4:4 kernels; the default /
    // co-sited path keeps the byte-identical decode (the fused high-bit 4:2:2
    // kernels upsample chroma in-register). 4:2:2 is subsampled horizontally only
    // — there is no vertical blend or chroma lookback (cf. the 4:2:0 sibling).
    let center_sited = chroma_422_center_sited_h(chroma_location);

    // Atomicity preflight (#302 / #308 / #314, cf. the crate's #180 resample fix
    // and the high-bit 4:2:0 sibling): reserve EVERY fallible row scratch this
    // identity row can touch BEFORE any output row is written (the luma plane
    // below, then the u16 / u8 RGB / RGBA / HSV fan-out), so an allocator refusal
    // returns a typed `AllocationFailed` leaving the output frame untouched rather
    // than partially mutated. Two scratches can grow:
    //  1. the centered-siting full-width `u16` chroma (`chroma_full_u16`), needed
    //     by ANY colour output (u8 OR u16 RGB / RGBA / HSV); and
    //  2. the u8 RGB row buffer, reached exactly when a colour decode needs an RGB
    //     row but no caller RGB buffer is borrowable — `want_hsv && want_rgba &&
    //     !want_rgb` (`rgb_row_buf_or_scratch`'s own scratch arm). The u16 RGB /
    //     RGBA outputs write straight into their caller buffers (the rgb_u16 plane
    //     itself stages the rgba_u16 expand) and never grow a scratch; this format
    //     exposes no luma_u16 output.
    // The later `reconstruct_chroma` / `rgb_row_buf_or_scratch` calls
    // reuse the already-sized buffers, so the default path is byte-identical; only
    // the failure-path ordering changes.
    let need_centered_chroma =
      center_sited && (want_rgb || want_rgba || want_hsv || want_rgb_u16 || want_rgba_u16);
    if need_centered_chroma {
      reserve_420_chroma_full_u16(chroma_full_u16, w, h)?;
    }
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

    // Centered full-width chroma (phase-0.5), reconstructed ONCE per row from the
    // wire-format half-width U / V and reused by every colour decode (u16 and u8).
    // Infallible — the scratch was reserved above. The default / co-sited siting
    // leaves it `None`, so the fused 4:2:2 kernels upsample chroma in-register and
    // the output stays byte-identical.
    let centered = if need_centered_chroma {
      Some(reconstruct_chroma(
        ChromaU16::<BITS> { big_endian: BE },
        chroma_full_u16,
        row.u_half(),
        row.v_half(),
        w,
      ))
    } else {
      None
    };

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
    if want_rgba_u16 && !want_rgb_u16 {
      let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row =
        rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
      if let Some((u_full, v_full)) = centered {
        yuv444p12_to_rgba_u16_row_endian(
          row.y(),
          u_full,
          v_full,
          rgba_u16_row,
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
          BE,
        );
      } else {
        yuv420p12_to_rgba_u16_row_endian(
          row.y(),
          row.u_half(),
          row.v_half(),
          rgba_u16_row,
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
          BE,
        );
      }
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
      if let Some((u_full, v_full)) = centered {
        yuv444p12_to_rgb_u16_row_endian(
          row.y(),
          u_full,
          v_full,
          rgb_u16_row,
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
          BE,
        );
      } else {
        yuv420p12_to_rgb_u16_row_endian(
          row.y(),
          row.u_half(),
          row.v_half(),
          rgb_u16_row,
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
          BE,
        );
      }
      if want_rgba_u16 {
        let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
        let rgba_u16_row =
          rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
        expand_rgb_u16_to_rgba_u16_row::<BITS>(rgb_u16_row, rgba_u16_row, w);
      }
    }

    // ===== u8 RGB / RGBA / HSV path (Strategy A) =====
    // HSV-without-RGB-or-RGBA goes through the direct `yuv420p12_to_hsv_row_endian`
    // kernel (no source-width RGB scratch — the SIMD path stages a fixed
    // 8-bit RGB chunk internally). RGB or RGBA also attached keeps the
    // convert-once-then-derive path alive via `need_rgb_kernel`.
    let want_hsv_direct = want_hsv && !want_rgb && !want_rgba;
    let need_rgb_kernel = want_rgb || (want_hsv && want_rgba);

    if want_hsv_direct {
      let hsv = hsv.as_mut().expect("want_hsv_direct implies hsv attached");
      let (h, s, v) = hsv.hsv();
      if let Some((u_full, v_full)) = centered {
        yuv444p12_to_hsv_row_endian(
          row.y(),
          u_full,
          v_full,
          &mut h[one_plane_start..one_plane_end],
          &mut s[one_plane_start..one_plane_end],
          &mut v[one_plane_start..one_plane_end],
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
          BE,
        );
      } else {
        yuv420p12_to_hsv_row_endian(
          row.y(),
          row.u_half(),
          row.v_half(),
          &mut h[one_plane_start..one_plane_end],
          &mut s[one_plane_start..one_plane_end],
          &mut v[one_plane_start..one_plane_end],
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
          BE,
        );
      }
      return Ok(());
    }

    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      if let Some((u_full, v_full)) = centered {
        yuv444p12_to_rgba_row_endian(
          row.y(),
          u_full,
          v_full,
          rgba_row,
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
          BE,
        );
      } else {
        yuv420p12_to_rgba_row_endian(
          row.y(),
          row.u_half(),
          row.v_half(),
          rgba_row,
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
          BE,
        );
      }
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

    if let Some((u_full, v_full)) = centered {
      yuv444p12_to_rgb_row_endian(
        row.y(),
        u_full,
        v_full,
        rgb_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
        BE,
      );
    } else {
      yuv420p12_to_rgb_row_endian(
        row.y(),
        row.u_half(),
        row.v_half(),
        rgb_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
        BE,
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

    if let Some(buf) = rgba.as_deref_mut() {
      let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
    }

    Ok(())
  }
}

impl<'a, R, const BE: bool> MixedSinker<'a, Yuv422p14<BE>, R> {
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

  /// Attaches a packed **8‑bit** RGBA output buffer. The 14‑bit YUV
  /// source is converted to 8‑bit RGBA via the `BITS = 14` Q15 kernel
  /// family; alpha = `0xFF` (Yuv422p14 has no alpha plane).
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

  /// Attaches a packed **`u16`** RGBA output buffer. 14‑bit
  /// low‑packed (`(1 << 14) - 1 = 16383` max). Length is measured in
  /// `u16` **elements** (`width x height x 4`). Alpha element is
  /// `(1 << 14) - 1`.
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

impl<R, const BE: bool> Yuv422p14Sink<BE> for MixedSinker<'_, Yuv422p14<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, Yuv422p14<BE>, R> {
  type Input<'r> = Yuv422p14Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::WidthAlignment(WidthAlignment::odd(
        self.width,
      )));
    }
    check_dimensions_match(self.width, self.height, width, height)?;
    reset_high_bit_yuv_streams(self);
    Ok(())
  }

  fn process(&mut self, row: Yuv422p14Row<'_>) -> Result<(), Self::Error> {
    const BITS: u32 = 14;
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if w & 1 != 0 {
      return Err(MixedSinkerError::WidthAlignment(WidthAlignment::odd(w)));
    }
    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Y14,
        idx,
        w,
        row.y().len(),
      )));
    }
    if row.u_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::UHalf14,
        idx,
        w / 2,
        row.u_half().len(),
      )));
    }
    if row.v_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::VHalf14,
        idx,
        w / 2,
        row.v_half().len(),
      )));
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange(
        RowIndexOutOfRange::new(idx, self.height),
      ));
    }

    // Chroma siting (#302): drives the identity-plan horizontal chroma phase.
    // `Copy`, so read it out before the field split-borrow below.
    let chroma_location = self.chroma_location;

    let Self {
      rgb,
      rgb_u16,
      rgba,
      rgba_u16,
      luma,
      hsv,
      rgb_scratch,
      rgb_scratch_u16,
      chroma_full_u16,
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
      frozen_chroma_centered,
      ..
    } = self;

    // Non-identity plan: feed the shared high-bit 4:2:2 triple-resample
    // tail (u8 color, independent native-u16 color, native Y). The half-
    // width U / V planes are horizontally upsampled in-register by the
    // shared 4:2:0 row kernels (4:2:0 and 4:2:2 have the identical per-row
    // chroma contract). Yuv422p exposes no `luma_u16` output, so it is
    // `&mut None` and only `luma` (binned native Y `>> (BITS - 8)`) is
    // emitted. The span kind picks the engine (area bin or signed-coefficient
    // filter twin) — see the Yuv422p10 impl for the full rationale; the
    // filter tail clamps every sub-16-bit colour sample AND the native Y to
    // `(1 << BITS) - 1`.
    if let Some(plan) = plan.as_ref() {
      let matrix = row.matrix();
      let full_range = row.full_range();
      let (y, u_half, v_half) = (row.y(), row.u_half(), row.v_half());
      // RFC #238 S5a — 4:2:2 horizontal chroma siting. The centered group
      // (`Center` / `Top` / `Bottom`, `chroma_422_center_sited_h`) samples
      // chroma at `+0.5` luma = `+0.25` chroma-sample; the co-sited /
      // unspecified group is phase 0 (today's byte-identical decode). Siting
      // enters the chroma RECONSTRUCTION only — the averaging tier is still
      // chosen by `select_insertion_point` below — so the native fast tier
      // folds the phase into the `area_chroma_422` chroma weights while the
      // row-stage and filter tiers reconstruct full-width `u16` chroma and
      // decode 4:4:4.
      let center_sited = chroma_422_center_sited_h(chroma_location);
      let chroma_h_phase = if center_sited {
        YUV422P_CENTERED_H_PHASE
      } else {
        0.0
      };
      let need_output = luma.is_some()
        || rgb.is_some()
        || rgba.is_some()
        || hsv.is_some()
        || rgb_u16.is_some()
        || rgba_u16.is_some();
      // Only the colour tiers reconstruct full-width chroma for the centered
      // decode; a luma-only centered row bins native Y unchanged (siting is a
      // chroma-only property).
      let want_color =
        rgb.is_some() || rgba.is_some() || hsv.is_some() || rgb_u16.is_some() || rgba_u16.is_some();
      // Freeze the effective 4:2:2 chroma siting on the first output-bearing
      // row (mirrors the `frozen_native_route` freeze below). This CHECK is at
      // the always-compiled choke point every tier passes through; the matching
      // SET rides each tier's accept path (never before dispatch, so a rejected
      // row leaves it unset for a corrected retry). A later row observing a
      // different phase would bin a mixture of co-sited and centered chroma, so
      // it is rejected here before any reconstruction.
      if need_output
        && let Some(frozen) = *frozen_chroma_centered
        && frozen != center_sited
      {
        return Err(MixedSinkerError::ChromaSitingChanged(
          ChromaSitingChanged::new(idx),
        ));
      }
      if plan.kind().is_filter() {
        // Reject a multi-kernel (BICUBLIN) plan BEFORE the centered reserve
        // below — the delegate's first act is this same check, so hoisting it
        // keeps a rejected filter plan from reserving / reconstructing chroma
        // first (the #180 reject-before-allocation invariant). Idempotent — the
        // delegate re-runs it.
        plan.ensure_single_kernel_filter()?;
        if center_sited && want_color {
          // Centered filter: reconstruct full-width `u16` chroma, but ONLY after
          // the resample preflight (frozen-output + sequence), so an
          // out-of-sequence / rejected row is caught before the chroma
          // reservation (#180). `packed_yuv422_triple_filter_resample` re-runs
          // the idempotent preflight and owns the transactional commit.
          let expected = if luma.is_some() {
            luma_filter_stream_u16.as_ref().map_or(0, |s| s.next_y())
          } else if rgb.is_some() || rgba.is_some() || hsv.is_some() {
            rgb_filter_stream.as_ref().map_or(0, |s| s.next_y())
          } else {
            rgb_filter_stream_u16.as_ref().map_or(0, |s| s.next_y())
          };
          if let core::ops::ControlFlow::Break(()) = resample_preflight_check_only(
            resample_outputs,
            luma,
            &None,
            rgb,
            rgba,
            rgb_u16,
            rgba_u16,
            &None,
            &None,
            &None,
            &None,
            &None,
            hsv,
            &None,
            Some(expected),
            idx,
          )? {
            return Ok(());
          }
          reserve_420_chroma_full_u16(chroma_full_u16, w, h)?;
          let (u_full, v_full) = reconstruct_chroma(
            ChromaU16::<BITS> { big_endian: BE },
            chroma_full_u16,
            u_half,
            v_half,
            w,
          );
          let r = packed_yuv422_triple_filter_resample::<BITS>(
            luma_filter_stream_u16,
            rgb_filter_stream,
            rgb_filter_stream_u16,
            resample_outputs,
            rgb,
            rgba,
            rgb_u16,
            rgba_u16,
            luma,
            &mut None,
            hsv,
            luma_scratch_u16,
            rgb_scratch,
            rgb_scratch_u16,
            w,
            plan,
            idx,
            use_simd,
            matrix,
            full_range,
            |scratch| deinterleave_y_high_bit_masked::<BITS, BE>(y, scratch, w),
            |scratch| {
              yuv444p14_to_rgb_row_endian(
                y, u_full, v_full, scratch, w, matrix, full_range, use_simd, BE,
              )
            },
            |scratch| {
              yuv444p14_to_rgb_u16_row_endian(
                y, u_full, v_full, scratch, w, matrix, full_range, use_simd, BE,
              )
            },
          );
          if r.is_ok() && need_output && frozen_chroma_centered.is_none() {
            *frozen_chroma_centered = Some(center_sited);
          }
          return r;
        }
        let r = packed_yuv422_triple_filter_resample::<BITS>(
          luma_filter_stream_u16,
          rgb_filter_stream,
          rgb_filter_stream_u16,
          resample_outputs,
          rgb,
          rgba,
          rgb_u16,
          rgba_u16,
          luma,
          &mut None,
          hsv,
          luma_scratch_u16,
          rgb_scratch,
          rgb_scratch_u16,
          w,
          plan,
          idx,
          use_simd,
          matrix,
          full_range,
          |scratch| deinterleave_y_high_bit_masked::<BITS, BE>(y, scratch, w),
          |scratch| {
            yuv420p14_to_rgb_row_endian(
              y, u_half, v_half, scratch, w, matrix, full_range, use_simd, BE,
            )
          },
          |scratch| {
            yuv420p14_to_rgb_u16_row_endian(
              y, u_half, v_half, scratch, w, matrix, full_range, use_simd, BE,
            )
          },
        );
        if r.is_ok() && need_output && frozen_chroma_centered.is_none() {
          *frozen_chroma_centered = Some(center_sited);
        }
        return r;
      }
      // Native / row-stage route split — see the high-bit 4:2:0 Yuv420p impl
      // for the CHECK-before / SET-after `frozen_native_route` contract.
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
          native_eligible: YUV422P_HIGH_BIT_NATIVE_ELIGIBLE,
          with_native: *native,
          area_plan: true,
        },
      );
      match insertion {
        InsertionPoint::NativeCodes => {
          // 4:2:2: chroma `w/2 x h` — half width, full height; a chroma row per
          // Y row (`chroma_vsub = 1`, `chroma_w = w/2`), chroma plan the phased
          // `area_chroma_422` (h-phase `0.25` chroma-sample when centered, else
          // the plain box). RFC #238 point-of-use siting invalidation: a reused
          // sink's cached join is only `reset` between frames, so a frame whose
          // `chroma_location` moved to a different phase must REBUILD it. Drop
          // the stale-phase join ONLY on the in-sequence first row of a fresh
          // frame (`idx == 0`, `next_y() == 0`) so a mid-frame / out-of-sequence
          // row rejects against the INTACT join and a corrected retry rebuilds
          // cleanly; a luma-only join carries no chroma phase and is never
          // dropped. Move it OUT (the delegate builds the replacement into the
          // field, keeping it untouched until every pre-feed allocation
          // succeeds) and restore the intact prior-phase join on a rejected
          // rebuild so the row mutates no join state.
          let stale_native = idx == 0
            && native_planar_u16.as_ref().is_some_and(|join| {
              join.chroma_phase_centered() == Some(!center_sited) && join.next_y() == 0
            });
          let prev_native = if stale_native {
            native_planar_u16.take()
          } else {
            None
          };
          let native_result = yuv_planar16_process_native::<BITS, BE>(
            plan,
            native_planar_u16,
            resample_outputs,
            rgb,
            rgba,
            rgb_u16,
            rgba_u16,
            luma,
            // The high-bit planar 4:2:2 family exposes no `luma_u16` output.
            &mut None,
            hsv,
            rgb_scratch,
            rgb_scratch_u16,
            y,
            u_half,
            v_half,
            matrix,
            full_range,
            idx,
            w,
            h,
            1,
            w / 2,
            || {
              ResamplePlan::area_chroma_422(
                w / 2,
                h,
                plan.out_w(),
                plan.out_h(),
                chroma_h_phase,
                0.0,
              )
            },
            use_simd,
          );
          // Restore the taken stale-phase join if the delegate's rebuild was
          // rejected at any pre-feed step: it leaves the field `None` on such a
          // failure, so restoring the intact prior-phase join leaves the
          // rejected row mutating no join state. A non-stale row took nothing.
          if stale_native && native_result.is_err() {
            *native_planar_u16 = prev_native;
          }
          native_result?;
          if frozen_native_route.is_none() && need_output {
            *frozen_native_route = Some(true);
          }
          // RFC #238 S5a: freeze the siting on the same accepted output row.
          if frozen_chroma_centered.is_none() && need_output {
            *frozen_chroma_centered = Some(center_sited);
          }
          return Ok(());
        }
        InsertionPoint::EncodedOutput => {
          if center_sited && want_color {
            // Centered row-stage: reconstruct full-width `u16` chroma AFTER the
            // resample preflight (frozen-output + sequence), so an
            // out-of-sequence / rejected row is caught before the chroma
            // reservation (#180). `packed_yuv422_triple_resample` re-runs the
            // idempotent preflight and owns the transactional commit.
            let expected = if luma.is_some() {
              luma_stream_u16.as_ref().map_or(0, |s| s.next_y())
            } else if rgb.is_some() || rgba.is_some() || hsv.is_some() {
              rgb_stream.as_ref().map_or(0, |s| s.next_y())
            } else {
              rgb_stream_u16.as_ref().map_or(0, |s| s.next_y())
            };
            if let core::ops::ControlFlow::Break(()) = resample_preflight_check_only(
              resample_outputs,
              luma,
              &None,
              rgb,
              rgba,
              rgb_u16,
              rgba_u16,
              &None,
              &None,
              &None,
              &None,
              &None,
              hsv,
              &None,
              Some(expected),
              idx,
            )? {
              return Ok(());
            }
            reserve_420_chroma_full_u16(chroma_full_u16, w, h)?;
            let (u_full, v_full) = reconstruct_chroma(
              ChromaU16::<BITS> { big_endian: BE },
              chroma_full_u16,
              u_half,
              v_half,
              w,
            );
            packed_yuv422_triple_resample::<BITS>(
              luma_stream_u16,
              rgb_stream,
              rgb_stream_u16,
              resample_outputs,
              rgb,
              rgba,
              rgb_u16,
              rgba_u16,
              luma,
              &mut None,
              hsv,
              luma_scratch_u16,
              rgb_scratch,
              rgb_scratch_u16,
              w,
              plan,
              idx,
              use_simd,
              matrix,
              full_range,
              |scratch| deinterleave_y_high_bit_masked::<BITS, BE>(y, scratch, w),
              |scratch| {
                yuv444p14_to_rgb_row_endian(
                  y, u_full, v_full, scratch, w, matrix, full_range, use_simd, BE,
                )
              },
              |scratch| {
                yuv444p14_to_rgb_u16_row_endian(
                  y, u_full, v_full, scratch, w, matrix, full_range, use_simd, BE,
                )
              },
            )?;
          } else {
            packed_yuv422_triple_resample::<BITS>(
              luma_stream_u16,
              rgb_stream,
              rgb_stream_u16,
              resample_outputs,
              rgb,
              rgba,
              rgb_u16,
              rgba_u16,
              luma,
              &mut None,
              hsv,
              luma_scratch_u16,
              rgb_scratch,
              rgb_scratch_u16,
              w,
              plan,
              idx,
              use_simd,
              matrix,
              full_range,
              |scratch| deinterleave_y_high_bit_masked::<BITS, BE>(y, scratch, w),
              |scratch| {
                yuv420p14_to_rgb_row_endian(
                  y, u_half, v_half, scratch, w, matrix, full_range, use_simd, BE,
                )
              },
              |scratch| {
                yuv420p14_to_rgb_u16_row_endian(
                  y, u_half, v_half, scratch, w, matrix, full_range, use_simd, BE,
                )
              },
            )?;
          }
          if frozen_native_route.is_none() && need_output {
            *frozen_native_route = Some(false);
          }
          // RFC #238 S5a: freeze the siting on the same accepted output row.
          if frozen_chroma_centered.is_none() && need_output {
            *frozen_chroma_centered = Some(center_sited);
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

    // Resolve the FULL output set up front so BOTH the no-output guard and the
    // atomicity preflight below run before any row-offset arithmetic.
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();
    let want_rgb_u16 = rgb_u16.is_some();
    let want_rgba_u16 = rgba_u16.is_some();

    // Repo-wide no-output invariant: a `process` call carrying NO output runs
    // NOTHING — no per-row offset arithmetic, no allocation, no state mutation.
    // Returning HERE, before the `idx * w` offsets below, also keeps the invariant
    // overflow-safe: a no-output call never ran an attach-time `w x h x 1`
    // validation, so `idx * w` could overflow `usize` on a 32-bit target with
    // absurd geometry; the guard skips that math (and the centered chroma
    // reservation) entirely. Mirrors the 8-bit Yuv422p sibling. (The high-bit
    // planar 4:2:2 family exposes no `luma_u16` output.)
    let need_output =
      want_rgb || want_rgba || want_hsv || want_rgb_u16 || want_rgba_u16 || luma.is_some();
    if !need_output {
      return Ok(());
    }

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // Chroma siting (#302): the centered horizontal sitings reconstruct chroma at
    // the phase-0.5 position then decode via the 4:4:4 kernels; the default /
    // co-sited path keeps the byte-identical decode (the fused high-bit 4:2:2
    // kernels upsample chroma in-register). 4:2:2 is subsampled horizontally only
    // — there is no vertical blend or chroma lookback (cf. the 4:2:0 sibling).
    let center_sited = chroma_422_center_sited_h(chroma_location);

    // Atomicity preflight (#302 / #308 / #314, cf. the crate's #180 resample fix
    // and the high-bit 4:2:0 sibling): reserve EVERY fallible row scratch this
    // identity row can touch BEFORE any output row is written (the luma plane
    // below, then the u16 / u8 RGB / RGBA / HSV fan-out), so an allocator refusal
    // returns a typed `AllocationFailed` leaving the output frame untouched rather
    // than partially mutated. Two scratches can grow:
    //  1. the centered-siting full-width `u16` chroma (`chroma_full_u16`), needed
    //     by ANY colour output (u8 OR u16 RGB / RGBA / HSV); and
    //  2. the u8 RGB row buffer, reached exactly when a colour decode needs an RGB
    //     row but no caller RGB buffer is borrowable — `want_hsv && want_rgba &&
    //     !want_rgb` (`rgb_row_buf_or_scratch`'s own scratch arm). The u16 RGB /
    //     RGBA outputs write straight into their caller buffers (the rgb_u16 plane
    //     itself stages the rgba_u16 expand) and never grow a scratch; this format
    //     exposes no luma_u16 output.
    // The later `reconstruct_chroma` / `rgb_row_buf_or_scratch` calls
    // reuse the already-sized buffers, so the default path is byte-identical; only
    // the failure-path ordering changes.
    let need_centered_chroma =
      center_sited && (want_rgb || want_rgba || want_hsv || want_rgb_u16 || want_rgba_u16);
    if need_centered_chroma {
      reserve_420_chroma_full_u16(chroma_full_u16, w, h)?;
    }
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

    // Centered full-width chroma (phase-0.5), reconstructed ONCE per row from the
    // wire-format half-width U / V and reused by every colour decode (u16 and u8).
    // Infallible — the scratch was reserved above. The default / co-sited siting
    // leaves it `None`, so the fused 4:2:2 kernels upsample chroma in-register and
    // the output stays byte-identical.
    let centered = if need_centered_chroma {
      Some(reconstruct_chroma(
        ChromaU16::<BITS> { big_endian: BE },
        chroma_full_u16,
        row.u_half(),
        row.v_half(),
        w,
      ))
    } else {
      None
    };

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
    if want_rgba_u16 && !want_rgb_u16 {
      let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row =
        rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
      if let Some((u_full, v_full)) = centered {
        yuv444p14_to_rgba_u16_row_endian(
          row.y(),
          u_full,
          v_full,
          rgba_u16_row,
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
          BE,
        );
      } else {
        yuv420p14_to_rgba_u16_row_endian(
          row.y(),
          row.u_half(),
          row.v_half(),
          rgba_u16_row,
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
          BE,
        );
      }
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
      if let Some((u_full, v_full)) = centered {
        yuv444p14_to_rgb_u16_row_endian(
          row.y(),
          u_full,
          v_full,
          rgb_u16_row,
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
          BE,
        );
      } else {
        yuv420p14_to_rgb_u16_row_endian(
          row.y(),
          row.u_half(),
          row.v_half(),
          rgb_u16_row,
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
          BE,
        );
      }
      if want_rgba_u16 {
        let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
        let rgba_u16_row =
          rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
        expand_rgb_u16_to_rgba_u16_row::<BITS>(rgb_u16_row, rgba_u16_row, w);
      }
    }

    // ===== u8 RGB / RGBA / HSV path (Strategy A) =====
    // HSV-without-RGB-or-RGBA goes through the direct `yuv420p14_to_hsv_row_endian`
    // kernel (no source-width RGB scratch — the SIMD path stages a fixed
    // 8-bit RGB chunk internally). RGB or RGBA also attached keeps the
    // convert-once-then-derive path alive via `need_rgb_kernel`.
    let want_hsv_direct = want_hsv && !want_rgb && !want_rgba;
    let need_rgb_kernel = want_rgb || (want_hsv && want_rgba);

    if want_hsv_direct {
      let hsv = hsv.as_mut().expect("want_hsv_direct implies hsv attached");
      let (h, s, v) = hsv.hsv();
      if let Some((u_full, v_full)) = centered {
        yuv444p14_to_hsv_row_endian(
          row.y(),
          u_full,
          v_full,
          &mut h[one_plane_start..one_plane_end],
          &mut s[one_plane_start..one_plane_end],
          &mut v[one_plane_start..one_plane_end],
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
          BE,
        );
      } else {
        yuv420p14_to_hsv_row_endian(
          row.y(),
          row.u_half(),
          row.v_half(),
          &mut h[one_plane_start..one_plane_end],
          &mut s[one_plane_start..one_plane_end],
          &mut v[one_plane_start..one_plane_end],
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
          BE,
        );
      }
      return Ok(());
    }

    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      if let Some((u_full, v_full)) = centered {
        yuv444p14_to_rgba_row_endian(
          row.y(),
          u_full,
          v_full,
          rgba_row,
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
          BE,
        );
      } else {
        yuv420p14_to_rgba_row_endian(
          row.y(),
          row.u_half(),
          row.v_half(),
          rgba_row,
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
          BE,
        );
      }
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

    if let Some((u_full, v_full)) = centered {
      yuv444p14_to_rgb_row_endian(
        row.y(),
        u_full,
        v_full,
        rgb_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
        BE,
      );
    } else {
      yuv420p14_to_rgb_row_endian(
        row.y(),
        row.u_half(),
        row.v_half(),
        rgb_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
        BE,
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

    if let Some(buf) = rgba.as_deref_mut() {
      let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
    }

    Ok(())
  }
}

// ---- Yuv422p16 / Yuv444p16 impl ----------------------------------------
//
// 16-bit family. Yuv422p16 reuses the 4:2:0 16-bit kernel family
// (identical per-row shape); Yuv444p16 has its own kernels.

impl<'a, R, const BE: bool> MixedSinker<'a, Yuv422p16<BE>, R> {
  /// Attaches a packed **`u16`** RGB output buffer. Output covers
  /// full `u16` range `[0, 65535]` (16 active bits, no packing).
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

  /// Attaches a packed **8‑bit** RGBA output buffer. The 16‑bit YUV
  /// source is converted to 8‑bit RGBA via the dedicated `BITS = 16`
  /// kernel family (i64 chroma multiply — not the BITS-generic Q15
  /// pipeline); alpha = `0xFF` (Yuv422p16 has no alpha plane).
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

  /// Attaches a packed **`u16`** RGBA output buffer. Output covers the
  /// full `u16` range `[0, 65535]` (16 active bits, no packing). Length
  /// is measured in `u16` **elements** (`width x height x 4`). Alpha
  /// element is `0xFFFF`.
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

impl<R, const BE: bool> Yuv422p16Sink<BE> for MixedSinker<'_, Yuv422p16<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, Yuv422p16<BE>, R> {
  type Input<'r> = Yuv422p16Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::WidthAlignment(WidthAlignment::odd(
        self.width,
      )));
    }
    check_dimensions_match(self.width, self.height, width, height)?;
    reset_high_bit_yuv_streams(self);
    Ok(())
  }

  fn process(&mut self, row: Yuv422p16Row<'_>) -> Result<(), Self::Error> {
    const BITS: u32 = 16;
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if w & 1 != 0 {
      return Err(MixedSinkerError::WidthAlignment(WidthAlignment::odd(w)));
    }
    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Y16,
        idx,
        w,
        row.y().len(),
      )));
    }
    if row.u_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::UHalf16,
        idx,
        w / 2,
        row.u_half().len(),
      )));
    }
    if row.v_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::VHalf16,
        idx,
        w / 2,
        row.v_half().len(),
      )));
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange(
        RowIndexOutOfRange::new(idx, self.height),
      ));
    }

    // Chroma siting (#302): drives the identity-plan horizontal chroma phase.
    // `Copy`, so read it out before the field split-borrow below.
    let chroma_location = self.chroma_location;

    let Self {
      rgb,
      rgb_u16,
      rgba,
      rgba_u16,
      luma,
      hsv,
      rgb_scratch,
      rgb_scratch_u16,
      chroma_full_u16,
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
      frozen_chroma_centered,
      ..
    } = self;

    // Non-identity plan: feed the shared high-bit 4:2:2 triple-resample
    // tail (u8 color, independent native-u16 color, native Y). The half-
    // width U / V planes are horizontally upsampled in-register by the
    // shared 4:2:0 row kernels (4:2:0 and 4:2:2 have the identical per-row
    // chroma contract). Yuv422p exposes no `luma_u16` output, so it is
    // `&mut None` and only `luma` (binned native Y `>> (BITS - 8)`) is
    // emitted. The span kind picks the engine (area bin or signed-coefficient
    // filter twin) — see the Yuv422p10 impl for the full rationale. At
    // `BITS = 16` the native max is the u16 max, so the filter tail's
    // sub-16-bit clamp is a value no-op.
    if let Some(plan) = plan.as_ref() {
      let matrix = row.matrix();
      let full_range = row.full_range();
      let (y, u_half, v_half) = (row.y(), row.u_half(), row.v_half());
      // RFC #238 S5a — 4:2:2 horizontal chroma siting. The centered group
      // (`Center` / `Top` / `Bottom`, `chroma_422_center_sited_h`) samples
      // chroma at `+0.5` luma = `+0.25` chroma-sample; the co-sited /
      // unspecified group is phase 0 (today's byte-identical decode). Siting
      // enters the chroma RECONSTRUCTION only — the averaging tier is still
      // chosen by `select_insertion_point` below — so the native fast tier
      // folds the phase into the `area_chroma_422` chroma weights while the
      // row-stage and filter tiers reconstruct full-width `u16` chroma and
      // decode 4:4:4.
      let center_sited = chroma_422_center_sited_h(chroma_location);
      let chroma_h_phase = if center_sited {
        YUV422P_CENTERED_H_PHASE
      } else {
        0.0
      };
      let need_output = luma.is_some()
        || rgb.is_some()
        || rgba.is_some()
        || hsv.is_some()
        || rgb_u16.is_some()
        || rgba_u16.is_some();
      // Only the colour tiers reconstruct full-width chroma for the centered
      // decode; a luma-only centered row bins native Y unchanged (siting is a
      // chroma-only property).
      let want_color =
        rgb.is_some() || rgba.is_some() || hsv.is_some() || rgb_u16.is_some() || rgba_u16.is_some();
      // Freeze the effective 4:2:2 chroma siting on the first output-bearing
      // row (mirrors the `frozen_native_route` freeze below). This CHECK is at
      // the always-compiled choke point every tier passes through; the matching
      // SET rides each tier's accept path (never before dispatch, so a rejected
      // row leaves it unset for a corrected retry). A later row observing a
      // different phase would bin a mixture of co-sited and centered chroma, so
      // it is rejected here before any reconstruction.
      if need_output
        && let Some(frozen) = *frozen_chroma_centered
        && frozen != center_sited
      {
        return Err(MixedSinkerError::ChromaSitingChanged(
          ChromaSitingChanged::new(idx),
        ));
      }
      if plan.kind().is_filter() {
        // Reject a multi-kernel (BICUBLIN) plan BEFORE the centered reserve
        // below — the delegate's first act is this same check, so hoisting it
        // keeps a rejected filter plan from reserving / reconstructing chroma
        // first (the #180 reject-before-allocation invariant). Idempotent — the
        // delegate re-runs it.
        plan.ensure_single_kernel_filter()?;
        if center_sited && want_color {
          // Centered filter: reconstruct full-width `u16` chroma, but ONLY after
          // the resample preflight (frozen-output + sequence), so an
          // out-of-sequence / rejected row is caught before the chroma
          // reservation (#180). `packed_yuv422_triple_filter_resample` re-runs
          // the idempotent preflight and owns the transactional commit.
          let expected = if luma.is_some() {
            luma_filter_stream_u16.as_ref().map_or(0, |s| s.next_y())
          } else if rgb.is_some() || rgba.is_some() || hsv.is_some() {
            rgb_filter_stream.as_ref().map_or(0, |s| s.next_y())
          } else {
            rgb_filter_stream_u16.as_ref().map_or(0, |s| s.next_y())
          };
          if let core::ops::ControlFlow::Break(()) = resample_preflight_check_only(
            resample_outputs,
            luma,
            &None,
            rgb,
            rgba,
            rgb_u16,
            rgba_u16,
            &None,
            &None,
            &None,
            &None,
            &None,
            hsv,
            &None,
            Some(expected),
            idx,
          )? {
            return Ok(());
          }
          reserve_420_chroma_full_u16(chroma_full_u16, w, h)?;
          let (u_full, v_full) = reconstruct_chroma(
            ChromaU16::<BITS> { big_endian: BE },
            chroma_full_u16,
            u_half,
            v_half,
            w,
          );
          let r = packed_yuv422_triple_filter_resample::<BITS>(
            luma_filter_stream_u16,
            rgb_filter_stream,
            rgb_filter_stream_u16,
            resample_outputs,
            rgb,
            rgba,
            rgb_u16,
            rgba_u16,
            luma,
            &mut None,
            hsv,
            luma_scratch_u16,
            rgb_scratch,
            rgb_scratch_u16,
            w,
            plan,
            idx,
            use_simd,
            matrix,
            full_range,
            |scratch| deinterleave_y_high_bit_masked::<BITS, BE>(y, scratch, w),
            |scratch| {
              yuv444p16_to_rgb_row_endian(
                y, u_full, v_full, scratch, w, matrix, full_range, use_simd, BE,
              )
            },
            |scratch| {
              yuv444p16_to_rgb_u16_row_endian(
                y, u_full, v_full, scratch, w, matrix, full_range, use_simd, BE,
              )
            },
          );
          if r.is_ok() && need_output && frozen_chroma_centered.is_none() {
            *frozen_chroma_centered = Some(center_sited);
          }
          return r;
        }
        let r = packed_yuv422_triple_filter_resample::<BITS>(
          luma_filter_stream_u16,
          rgb_filter_stream,
          rgb_filter_stream_u16,
          resample_outputs,
          rgb,
          rgba,
          rgb_u16,
          rgba_u16,
          luma,
          &mut None,
          hsv,
          luma_scratch_u16,
          rgb_scratch,
          rgb_scratch_u16,
          w,
          plan,
          idx,
          use_simd,
          matrix,
          full_range,
          |scratch| deinterleave_y_high_bit_masked::<BITS, BE>(y, scratch, w),
          |scratch| {
            yuv420p16_to_rgb_row_endian(
              y, u_half, v_half, scratch, w, matrix, full_range, use_simd, BE,
            )
          },
          |scratch| {
            yuv420p16_to_rgb_u16_row_endian(
              y, u_half, v_half, scratch, w, matrix, full_range, use_simd, BE,
            )
          },
        );
        if r.is_ok() && need_output && frozen_chroma_centered.is_none() {
          *frozen_chroma_centered = Some(center_sited);
        }
        return r;
      }
      // Native / row-stage route split — see the high-bit 4:2:0 Yuv420p impl
      // for the CHECK-before / SET-after `frozen_native_route` contract.
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
          native_eligible: YUV422P_HIGH_BIT_NATIVE_ELIGIBLE,
          with_native: *native,
          area_plan: true,
        },
      );
      match insertion {
        InsertionPoint::NativeCodes => {
          // 4:2:2: chroma `w/2 x h` — half width, full height; a chroma row per
          // Y row (`chroma_vsub = 1`, `chroma_w = w/2`), chroma plan the phased
          // `area_chroma_422` (h-phase `0.25` chroma-sample when centered, else
          // the plain box). RFC #238 point-of-use siting invalidation: a reused
          // sink's cached join is only `reset` between frames, so a frame whose
          // `chroma_location` moved to a different phase must REBUILD it. Drop
          // the stale-phase join ONLY on the in-sequence first row of a fresh
          // frame (`idx == 0`, `next_y() == 0`) so a mid-frame / out-of-sequence
          // row rejects against the INTACT join and a corrected retry rebuilds
          // cleanly; a luma-only join carries no chroma phase and is never
          // dropped. Move it OUT (the delegate builds the replacement into the
          // field, keeping it untouched until every pre-feed allocation
          // succeeds) and restore the intact prior-phase join on a rejected
          // rebuild so the row mutates no join state.
          let stale_native = idx == 0
            && native_planar_u16.as_ref().is_some_and(|join| {
              join.chroma_phase_centered() == Some(!center_sited) && join.next_y() == 0
            });
          let prev_native = if stale_native {
            native_planar_u16.take()
          } else {
            None
          };
          let native_result = yuv_planar16_process_native::<BITS, BE>(
            plan,
            native_planar_u16,
            resample_outputs,
            rgb,
            rgba,
            rgb_u16,
            rgba_u16,
            luma,
            // The high-bit planar 4:2:2 family exposes no `luma_u16` output.
            &mut None,
            hsv,
            rgb_scratch,
            rgb_scratch_u16,
            y,
            u_half,
            v_half,
            matrix,
            full_range,
            idx,
            w,
            h,
            1,
            w / 2,
            || {
              ResamplePlan::area_chroma_422(
                w / 2,
                h,
                plan.out_w(),
                plan.out_h(),
                chroma_h_phase,
                0.0,
              )
            },
            use_simd,
          );
          // Restore the taken stale-phase join if the delegate's rebuild was
          // rejected at any pre-feed step: it leaves the field `None` on such a
          // failure, so restoring the intact prior-phase join leaves the
          // rejected row mutating no join state. A non-stale row took nothing.
          if stale_native && native_result.is_err() {
            *native_planar_u16 = prev_native;
          }
          native_result?;
          if frozen_native_route.is_none() && need_output {
            *frozen_native_route = Some(true);
          }
          // RFC #238 S5a: freeze the siting on the same accepted output row.
          if frozen_chroma_centered.is_none() && need_output {
            *frozen_chroma_centered = Some(center_sited);
          }
          return Ok(());
        }
        InsertionPoint::EncodedOutput => {
          if center_sited && want_color {
            // Centered row-stage: reconstruct full-width `u16` chroma AFTER the
            // resample preflight (frozen-output + sequence), so an
            // out-of-sequence / rejected row is caught before the chroma
            // reservation (#180). `packed_yuv422_triple_resample` re-runs the
            // idempotent preflight and owns the transactional commit.
            let expected = if luma.is_some() {
              luma_stream_u16.as_ref().map_or(0, |s| s.next_y())
            } else if rgb.is_some() || rgba.is_some() || hsv.is_some() {
              rgb_stream.as_ref().map_or(0, |s| s.next_y())
            } else {
              rgb_stream_u16.as_ref().map_or(0, |s| s.next_y())
            };
            if let core::ops::ControlFlow::Break(()) = resample_preflight_check_only(
              resample_outputs,
              luma,
              &None,
              rgb,
              rgba,
              rgb_u16,
              rgba_u16,
              &None,
              &None,
              &None,
              &None,
              &None,
              hsv,
              &None,
              Some(expected),
              idx,
            )? {
              return Ok(());
            }
            reserve_420_chroma_full_u16(chroma_full_u16, w, h)?;
            let (u_full, v_full) = reconstruct_chroma(
              ChromaU16::<BITS> { big_endian: BE },
              chroma_full_u16,
              u_half,
              v_half,
              w,
            );
            packed_yuv422_triple_resample::<BITS>(
              luma_stream_u16,
              rgb_stream,
              rgb_stream_u16,
              resample_outputs,
              rgb,
              rgba,
              rgb_u16,
              rgba_u16,
              luma,
              &mut None,
              hsv,
              luma_scratch_u16,
              rgb_scratch,
              rgb_scratch_u16,
              w,
              plan,
              idx,
              use_simd,
              matrix,
              full_range,
              |scratch| deinterleave_y_high_bit_masked::<BITS, BE>(y, scratch, w),
              |scratch| {
                yuv444p16_to_rgb_row_endian(
                  y, u_full, v_full, scratch, w, matrix, full_range, use_simd, BE,
                )
              },
              |scratch| {
                yuv444p16_to_rgb_u16_row_endian(
                  y, u_full, v_full, scratch, w, matrix, full_range, use_simd, BE,
                )
              },
            )?;
          } else {
            packed_yuv422_triple_resample::<BITS>(
              luma_stream_u16,
              rgb_stream,
              rgb_stream_u16,
              resample_outputs,
              rgb,
              rgba,
              rgb_u16,
              rgba_u16,
              luma,
              &mut None,
              hsv,
              luma_scratch_u16,
              rgb_scratch,
              rgb_scratch_u16,
              w,
              plan,
              idx,
              use_simd,
              matrix,
              full_range,
              |scratch| deinterleave_y_high_bit_masked::<BITS, BE>(y, scratch, w),
              |scratch| {
                yuv420p16_to_rgb_row_endian(
                  y, u_half, v_half, scratch, w, matrix, full_range, use_simd, BE,
                )
              },
              |scratch| {
                yuv420p16_to_rgb_u16_row_endian(
                  y, u_half, v_half, scratch, w, matrix, full_range, use_simd, BE,
                )
              },
            )?;
          }
          if frozen_native_route.is_none() && need_output {
            *frozen_native_route = Some(false);
          }
          // RFC #238 S5a: freeze the siting on the same accepted output row.
          if frozen_chroma_centered.is_none() && need_output {
            *frozen_chroma_centered = Some(center_sited);
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

    // Resolve the FULL output set up front so BOTH the no-output guard and the
    // atomicity preflight below run before any row-offset arithmetic.
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();
    let want_rgb_u16 = rgb_u16.is_some();
    let want_rgba_u16 = rgba_u16.is_some();

    // Repo-wide no-output invariant: a `process` call carrying NO output runs
    // NOTHING — no per-row offset arithmetic, no allocation, no state mutation.
    // Returning HERE, before the `idx * w` offsets below, also keeps the invariant
    // overflow-safe: a no-output call never ran an attach-time `w x h x 1`
    // validation, so `idx * w` could overflow `usize` on a 32-bit target with
    // absurd geometry; the guard skips that math (and the centered chroma
    // reservation) entirely. Mirrors the 8-bit Yuv422p sibling. (The high-bit
    // planar 4:2:2 family exposes no `luma_u16` output.)
    let need_output =
      want_rgb || want_rgba || want_hsv || want_rgb_u16 || want_rgba_u16 || luma.is_some();
    if !need_output {
      return Ok(());
    }

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // Chroma siting (#302): the centered horizontal sitings reconstruct chroma at
    // the phase-0.5 position then decode via the 4:4:4 kernels; the default /
    // co-sited path keeps the byte-identical decode (the fused high-bit 4:2:2
    // kernels upsample chroma in-register). 4:2:2 is subsampled horizontally only
    // — there is no vertical blend or chroma lookback (cf. the 4:2:0 sibling).
    let center_sited = chroma_422_center_sited_h(chroma_location);

    // Atomicity preflight (#302 / #308 / #314, cf. the crate's #180 resample fix
    // and the high-bit 4:2:0 sibling): reserve EVERY fallible row scratch this
    // identity row can touch BEFORE any output row is written (the luma plane
    // below, then the u16 / u8 RGB / RGBA / HSV fan-out), so an allocator refusal
    // returns a typed `AllocationFailed` leaving the output frame untouched rather
    // than partially mutated. Two scratches can grow:
    //  1. the centered-siting full-width `u16` chroma (`chroma_full_u16`), needed
    //     by ANY colour output (u8 OR u16 RGB / RGBA / HSV); and
    //  2. the u8 RGB row buffer, reached exactly when a colour decode needs an RGB
    //     row but no caller RGB buffer is borrowable — `want_hsv && want_rgba &&
    //     !want_rgb` (`rgb_row_buf_or_scratch`'s own scratch arm). The u16 RGB /
    //     RGBA outputs write straight into their caller buffers (the rgb_u16 plane
    //     itself stages the rgba_u16 expand) and never grow a scratch; this format
    //     exposes no luma_u16 output.
    // The later `reconstruct_chroma` / `rgb_row_buf_or_scratch` calls
    // reuse the already-sized buffers, so the default path is byte-identical; only
    // the failure-path ordering changes.
    let need_centered_chroma =
      center_sited && (want_rgb || want_rgba || want_hsv || want_rgb_u16 || want_rgba_u16);
    if need_centered_chroma {
      reserve_420_chroma_full_u16(chroma_full_u16, w, h)?;
    }
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

    // Centered full-width chroma (phase-0.5), reconstructed ONCE per row from the
    // wire-format half-width U / V and reused by every colour decode (u16 and u8).
    // Infallible — the scratch was reserved above. The default / co-sited siting
    // leaves it `None`, so the fused 4:2:2 kernels upsample chroma in-register and
    // the output stays byte-identical.
    let centered = if need_centered_chroma {
      Some(reconstruct_chroma(
        ChromaU16::<BITS> { big_endian: BE },
        chroma_full_u16,
        row.u_half(),
        row.v_half(),
        w,
      ))
    } else {
      None
    };

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
    // Reuses Yuv420p16's u16-output kernel — 4:2:2 per-row shape
    // matches 4:2:0's (half-width UV, one pair per Y pair).
    if want_rgba_u16 && !want_rgb_u16 {
      let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row =
        rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
      if let Some((u_full, v_full)) = centered {
        yuv444p16_to_rgba_u16_row_endian(
          row.y(),
          u_full,
          v_full,
          rgba_u16_row,
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
          BE,
        );
      } else {
        yuv420p16_to_rgba_u16_row_endian(
          row.y(),
          row.u_half(),
          row.v_half(),
          rgba_u16_row,
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
          BE,
        );
      }
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
      if let Some((u_full, v_full)) = centered {
        yuv444p16_to_rgb_u16_row_endian(
          row.y(),
          u_full,
          v_full,
          rgb_u16_row,
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
          BE,
        );
      } else {
        yuv420p16_to_rgb_u16_row_endian(
          row.y(),
          row.u_half(),
          row.v_half(),
          rgb_u16_row,
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
          BE,
        );
      }
      if want_rgba_u16 {
        let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
        let rgba_u16_row =
          rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
        expand_rgb_u16_to_rgba_u16_row::<BITS>(rgb_u16_row, rgba_u16_row, w);
      }
    }

    // ===== u8 RGB / RGBA / HSV path (Strategy A) =====
    // HSV-without-RGB-or-RGBA goes through the direct `yuv420p16_to_hsv_row_endian`
    // kernel (no source-width RGB scratch — the SIMD path stages a fixed
    // 8-bit RGB chunk internally). RGB or RGBA also attached keeps the
    // convert-once-then-derive path alive via `need_rgb_kernel`.
    let want_hsv_direct = want_hsv && !want_rgb && !want_rgba;
    let need_rgb_kernel = want_rgb || (want_hsv && want_rgba);

    if want_hsv_direct {
      let hsv = hsv.as_mut().expect("want_hsv_direct implies hsv attached");
      let (h, s, v) = hsv.hsv();
      if let Some((u_full, v_full)) = centered {
        yuv444p16_to_hsv_row_endian(
          row.y(),
          u_full,
          v_full,
          &mut h[one_plane_start..one_plane_end],
          &mut s[one_plane_start..one_plane_end],
          &mut v[one_plane_start..one_plane_end],
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
          BE,
        );
      } else {
        yuv420p16_to_hsv_row_endian(
          row.y(),
          row.u_half(),
          row.v_half(),
          &mut h[one_plane_start..one_plane_end],
          &mut s[one_plane_start..one_plane_end],
          &mut v[one_plane_start..one_plane_end],
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
          BE,
        );
      }
      return Ok(());
    }

    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      if let Some((u_full, v_full)) = centered {
        yuv444p16_to_rgba_row_endian(
          row.y(),
          u_full,
          v_full,
          rgba_row,
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
          BE,
        );
      } else {
        yuv420p16_to_rgba_row_endian(
          row.y(),
          row.u_half(),
          row.v_half(),
          rgba_row,
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
          BE,
        );
      }
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

    if let Some((u_full, v_full)) = centered {
      yuv444p16_to_rgb_row_endian(
        row.y(),
        u_full,
        v_full,
        rgb_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
        BE,
      );
    } else {
      yuv420p16_to_rgb_row_endian(
        row.y(),
        row.u_half(),
        row.v_half(),
        rgb_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
        BE,
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

    if let Some(buf) = rgba.as_deref_mut() {
      let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
    }

    Ok(())
  }
}
