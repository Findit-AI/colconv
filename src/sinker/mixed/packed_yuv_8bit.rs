//! Sinker impls for packed YUV 4:2:2 (8-bit) source formats — Tier 3,
//! Ship 10.
//!
//! Source family covered here:
//! - [`Yuyv422`] — `Y0, U0, Y1, V0, …` (FFmpeg `yuyv422` / YUY2).
//! - [`Uyvy422`] — `U0, Y0, V0, Y1, …` (FFmpeg `uyvy422` / UYVY).
//! - [`Yvyu422`] — `Y0, V0, Y1, U0, …` (FFmpeg `yvyu422` / YVYU).
//!
//! All three formats carry one packed plane of `2 * width` bytes per
//! row. The differences are pure byte permutation within each
//! 4-byte / 2-pixel block; the three dispatchers
//! ([`yuyv422_to_rgb_row`], [`uyvy422_to_rgb_row`],
//! [`yvyu422_to_rgb_row`] and the matching `_to_rgba_row` /
//! `_to_luma_row` siblings) hide that permutation behind a single
//! const-generic kernel template.
//!
//! Outputs map to the sink's standard channels:
//! - `with_rgb` / `with_rgba` — packed YUV → RGB Q15 pipeline (full
//!   `ColorMatrix` + range support inherited from the row); RGBA
//!   alpha is forced to `0xFF` (the source has no alpha channel).
//! - `with_luma` — extracts the Y bytes from the packed plane via
//!   the dedicated luma kernel (much cheaper than a full YUV→RGB
//!   pass).
//! - `with_hsv` — stages an internal RGB scratch (or the user's RGB
//!   buffer if attached) and runs the existing `rgb_to_hsv_row`
//!   kernel.
//!
//! When both RGB and RGBA outputs are requested, the RGBA plane is
//! derived from the just-computed RGB row via
//! [`expand_rgb_to_rgba_row`] (Strategy A — memory-bound copy + 0xFF
//! alpha pad) instead of running a second YUV→RGB kernel. When only
//! RGBA is wanted, the dedicated `_to_rgba_row` kernel writes the
//! RGBA buffer directly without staging RGB.
//!
//! ## Fused downscale
//!
//! Under a non-identity [`ResamplePlan`] the source rows feed the
//! shared area-resample engine through [`packed_yuv422_dual_resample`],
//! mirroring the planar YUV row-stage tier:
//! - **luma / luma_u16** de-interleave the **Y bytes** into a
//!   source-width row (the format's own `*_to_luma_row` kernel — the
//!   exact Y→luma derivation the direct path uses) and area-bin them
//!   through a 1-channel stream. Luma is taken from Y, *not* re-derived
//!   from converted RGB: under saturated / clamped chroma the two
//!   diverge, and the direct path takes luma from Y.
//! - **rgb / rgba / hsv** convert each packed row to canonical RGB at
//!   source width (the format's own fused `*_to_rgb_row` kernel does the
//!   chroma de-interleave + 4:2:2 horizontal upsample in-register,
//!   exactly as the identity path) and area-bin that RGB row through the
//!   3-channel stream, deriving every colour output from each finalized
//!   output row.
//!
//! RGB is byte-identical to an `Rgb24` area-resample of the
//! identity-converted frame; luma equals the area-downscaled Y plane.

use super::{
  FrozenOutputs, GeometryOverflow, HsvFrameMut, InsufficientBuffer, MixedSinker, MixedSinkerError,
  RowIndexOutOfRange, RowShapeMismatch, RowSlice, WidthAlignment, check_dimensions_match,
  frozen_outputs_check, rgb_row_buf_or_scratch, rgba_plane_row_slice, source_luma_scratch,
  source_rgb_scratch,
};
use crate::{
  PixelSink,
  resample::{AreaStream, FilterStream, ResamplePlan, RowResampler, SpanKind},
  row::{
    expand_rgb_to_rgba_row, rgb_to_hsv_row, uyvy422_to_hsv_row, uyvy422_to_luma_row,
    uyvy422_to_luma_u16_row, uyvy422_to_rgb_row, uyvy422_to_rgba_row, yuyv422_to_hsv_row,
    yuyv422_to_luma_row, yuyv422_to_luma_u16_row, yuyv422_to_rgb_row, yuyv422_to_rgba_row,
    yvyu422_to_hsv_row, yvyu422_to_luma_row, yvyu422_to_luma_u16_row, yvyu422_to_rgb_row,
    yvyu422_to_rgba_row,
  },
  source::{
    Uyvy422, Uyvy422Row, Uyvy422Sink, Yuyv422, Yuyv422Row, Yuyv422Sink, Yvyu422, Yvyu422Row,
    Yvyu422Sink,
  },
};

#[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
use super::{
  ChromaSitingChanged, NativeRouteChanged, chroma_422_center_sited_h,
  planar_8bit::{
    NativePlanarYuv, YUV422P_CENTERED_H_PHASE, native_planar_preflight_check_only,
    reserve_420_chroma_full, upsample_420_chroma_center_h, yuv_planar_process_native,
  },
};
#[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
use crate::{
  ColorMatrix,
  resample::{
    AveragingDomain, InsertionContext, InsertionPoint, PlanGeometry, ResampleError,
    select_insertion_point,
  },
  row::{yuv_444_to_hsv_row, yuv_444_to_rgb_row, yuv_444_to_rgba_row},
};

/// Shared stage-then-feed tail for the 8-bit packed-YUV dual-stream
/// resample, used by both [`packed_yuv422_dual_resample`] (area) and
/// [`packed_yuv422_dual_filter_resample`] (filter). The two paths differ
/// only in the resampler kind built by the caller — the convert-then-bin
/// staging and per-output emit are identical, so they live here behind the
/// [`RowResampler<u8>`] trait (which both
/// [`AreaStream<u8>`](crate::resample::AreaStream) and
/// [`FilterStream<u8>`](crate::resample::FilterStream) implement). Keeping
/// the emit byte-identical between the arms is what makes the filter output
/// equal the equivalent RGB filter of the converted pixels, and match the
/// area output up to the kernel weights.
///
/// `deinterleave_y` fills a source-width scratch with the Y samples pulled
/// from the packed row (the format's own `*_to_luma_row` kernel) and runs
/// only when a luma output is attached; `convert_rgb` fills a source-width
/// RGB scratch from the packed row (the format's own `*_to_rgb_row` kernel,
/// which de-interleaves + horizontally upsamples the chroma) and runs only
/// when a colour output is attached. The two scratches are distinct fields
/// and never alias.
///
/// No native-depth clamp: these are 8-bit sources, so the source's native
/// range *is* the full `u8` range and the stream's own finalize keeps every
/// binned sample in range (`AreaStream` averages within `[0, 255]`;
/// `FilterStream<u8>`'s `clip8` clamps a signed-kernel overshoot to
/// `[0, 255]`). Both streams are created by the caller post-sequence-check;
/// staging runs every fallible growth + conversion before the first feed so
/// a failure mutates no caller output.
#[cfg(feature = "yuv-packed")]
#[allow(clippy::too_many_arguments)]
fn packed_yuv422_dual_feed_emit<LS, CS>(
  luma_stream: Option<&mut LS>,
  rgb_stream: Option<&mut CS>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  luma_scratch: &mut std::vec::Vec<u8>,
  rgb_scratch: &mut std::vec::Vec<u8>,
  w: usize,
  plan: &ResamplePlan,
  idx: usize,
  use_simd: bool,
  deinterleave_y: impl FnOnce(&mut [u8]),
  convert_rgb: impl FnOnce(&mut [u8]),
) -> Result<(), MixedSinkerError>
where
  LS: RowResampler<u8>,
  CS: RowResampler<u8>,
{
  let ow = plan.out_w();
  // Stage the source-width rows (both fallible growths run before the
  // feeds, keeping the call atomic). The Y row uses its own scratch so
  // it does not collide with the colour stream's RGB scratch.
  let luma_row = if luma_stream.is_some() {
    let scratch = source_luma_scratch(luma_scratch, w, plan)?;
    deinterleave_y(scratch);
    Some(scratch)
  } else {
    None
  };
  let color_row = if rgb_stream.is_some() {
    let scratch = source_rgb_scratch(rgb_scratch, w, plan)?;
    convert_rgb(scratch);
    Some(scratch)
  } else {
    None
  };

  if let Some(y_row) = luma_row {
    let stream = luma_stream.expect("staged only when present");
    stream.feed_row(idx, y_row, use_simd, |oy, out_row| {
      if let Some(buf) = luma.as_deref_mut() {
        buf[oy * ow..(oy + 1) * ow].copy_from_slice(out_row);
      }
      if let Some(buf) = luma_u16.as_deref_mut() {
        for (dst, &src) in buf[oy * ow..(oy + 1) * ow].iter_mut().zip(out_row) {
          *dst = src as u16;
        }
      }
    })?;
  }

  if let Some(scratch) = color_row {
    let stream = rgb_stream.expect("staged only when present");
    stream.feed_row(idx, scratch, use_simd, |oy, out_row| {
      if let Some(buf) = rgb.as_deref_mut() {
        buf[oy * 3 * ow..(oy + 1) * 3 * ow].copy_from_slice(out_row);
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
      if let Some(buf) = rgba.as_deref_mut() {
        expand_rgb_to_rgba_row(out_row, &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow], ow);
      }
    })?;
  }

  Ok(())
}

/// Row-stage **area** fused downscale shared by the three packed YUV 4:2:2
/// formats and (for its filter twin) the packed YUV 4:1:1 `Uyyvyy411`.
/// Mirrors the planar YUV dual-stream path: **luma / luma_u16 area-resample
/// the de-interleaved Y bytes directly** (the YUV luma contract — luma is
/// *not* re-derived from converted RGB), while RGB / RGBA / HSV bin a
/// converted source-width RGB row.
///
/// `deinterleave_y` fills a source-width scratch with the Y samples
/// pulled from the packed row (the format's own `*_to_luma_row`
/// kernel), and runs only when a luma output is attached. `convert_rgb`
/// fills a source-width RGB scratch from the packed row (the format's
/// own `*_to_rgb_row` kernel), and runs only when a colour output is
/// attached.
///
/// Atomic preflight (mirrors [`planar_dual_resample`](super::planar_resample::planar_dual_resample)):
/// a compare-only preflight validates the output set + row sequence WITHOUT
/// freezing, then both streams and both source-width scratches are created, and
/// only after every pre-feed allocation succeeds do the output-set freeze and
/// the stream inserts commit together. So an out-of-sequence first row is
/// rejected before the freeze (storing no snapshot to poison a retry), a
/// first-row allocation failure leaves `resample_outputs` and both streams
/// untouched, `AllocationFailed` never masks `OutOfSequenceRow`, and a no-output
/// call is a true no-op regardless of the row index.
#[cfg(feature = "yuv-packed")]
#[allow(clippy::too_many_arguments)]
fn packed_yuv422_dual_resample(
  luma_stream: &mut Option<std::boxed::Box<AreaStream<u8>>>,
  rgb_stream: &mut Option<std::boxed::Box<AreaStream<u8>>>,
  resample_outputs: &mut Option<FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  luma_scratch: &mut std::vec::Vec<u8>,
  rgb_scratch: &mut std::vec::Vec<u8>,
  w: usize,
  plan: &ResamplePlan,
  idx: usize,
  use_simd: bool,
  deinterleave_y: impl FnOnce(&mut [u8]),
  convert_rgb: impl FnOnce(&mut [u8]),
) -> Result<(), MixedSinkerError> {
  let need_luma = luma.is_some() || luma_u16.is_some();
  let need_color = rgb.is_some() || hsv.is_some() || rgba.is_some();

  // Sequence-counter row for the shared L1 preflight: whichever stream is fed
  // every row (all attached streams advance in lockstep), or `None` when no
  // output is attached so the preflight short-circuits to a no-op.
  let expected = if need_luma {
    Some(luma_stream.as_ref().map_or(0, |stream| stream.next_y()))
  } else if need_color {
    Some(rgb_stream.as_ref().map_or(0, |stream| stream.next_y()))
  } else {
    None
  };
  // Compare-only preflight (NO commit): validate the output set + row sequence
  // WITHOUT freezing, so the output-set freeze and the stream inserts commit
  // TOGETHER only after every pre-feed allocation below has succeeded. A
  // first-row allocation failure therefore leaves `resample_outputs` and both
  // streams untouched — state-atomic, retryable even with a changed output
  // attachment. (The centered siting in Part B adds a fallible chroma reserve
  // before this delegate, so the freeze must not commit until every alloc has.)
  if let core::ops::ControlFlow::Break(()) = super::resample_preflight_check_only(
    resample_outputs,
    luma,
    luma_u16,
    rgb,
    rgba,
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
  // Build any missing streams into LOCALS first (fallible; NO field mutation
  // and NO output-set commit yet).
  let new_luma = if need_luma && luma_stream.is_none() {
    let stream = AreaStream::new(plan.h(), plan.v(), plan.src_w(), plan.src_h(), 1)?;
    Some(crate::resample::try_box(stream).map_err(|_| {
      MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
        crate::resample::PlanGeometry::new(plan.src_w(), plan.src_h(), plan.out_w(), plan.out_h()),
      ))
    })?)
  } else {
    None
  };
  let new_rgb = if need_color && rgb_stream.is_none() {
    let stream = AreaStream::new(plan.h(), plan.v(), plan.src_w(), plan.src_h(), 3)?;
    Some(crate::resample::try_box(stream).map_err(|_| {
      MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
        crate::resample::PlanGeometry::new(plan.src_w(), plan.src_h(), plan.out_w(), plan.out_h()),
      ))
    })?)
  } else {
    None
  };
  // Grow BOTH source-width scratches (fallible) BEFORE the commit: the shared
  // feed tail grows them on the first fed row, so hoisting them here keeps EVERY
  // pre-feed allocation (streams + luma scratch + RGB scratch) ahead of the
  // freeze and stream inserts; `packed_yuv422_dual_feed_emit`'s own grows are
  // then no-ops.
  if need_luma {
    source_luma_scratch(luma_scratch, w, plan)?;
  }
  if need_color {
    source_rgb_scratch(rgb_scratch, w, plan)?;
  }
  // Every pre-feed allocation succeeded — COMMIT atomically: insert the streams
  // and freeze the output set (the compare above already passed, so this only
  // stores the snapshot on the first row).
  if let Some(l) = new_luma {
    *luma_stream = Some(l);
  }
  if let Some(r) = new_rgb {
    *rgb_stream = Some(r);
  }
  frozen_outputs_check(
    resample_outputs,
    luma,
    luma_u16,
    rgb,
    rgba,
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

  packed_yuv422_dual_feed_emit(
    luma_stream.as_mut(),
    rgb_stream.as_mut(),
    rgb,
    rgba,
    luma,
    luma_u16,
    hsv,
    luma_scratch,
    rgb_scratch,
    w,
    plan,
    idx,
    use_simd,
    deinterleave_y,
    convert_rgb,
  )
}

/// Row-stage **separable-filter** fused resample — the
/// [`SpanKind::Filter`](crate::resample::SpanKind) twin of
/// [`packed_yuv422_dual_resample`], shared by the three packed YUV 4:2:2
/// formats and the packed YUV 4:1:1 `Uyyvyy411` (their convert closures
/// have the same shape — de-interleave Y for luma, convert the packed row
/// to a source-width RGB row for colour). Luma stays **native Y**: the
/// de-interleaved Y bytes feed a 1-channel
/// [`FilterStream<u8>`](crate::resample::FilterStream)
/// (`luma_filter_stream`), so luma is byte-exact to the direct `*_to_luma*`
/// kernels' filter resample, never colour-derived. Colour feeds the
/// 3-channel [`FilterStream<u8>`](crate::resample::FilterStream)
/// (`rgb_filter_stream`), so `rgb` / `rgba` / `hsv` equal the equivalent
/// `Rgb24` filter of the converted pixels.
///
/// No native-depth clamp: 8-bit source, so the stream's `clip8` (a
/// signed-kernel overshoot clamped to `[0, 255]`) already keeps every
/// sample in the native range — the colour and luma emit are identical to
/// the area path's.
///
/// Atomic preflight mirrors [`packed_yuv422_dual_resample`]: a BICUBLIN plan is
/// rejected first, then a compare-only preflight validates the output set + row
/// sequence WITHOUT freezing; only after both streams and both source-width
/// scratches are created does the output-set freeze + stream inserts commit
/// together. So an out-of-sequence first row is rejected before the freeze
/// (storing no snapshot to poison a retry), a first-row allocation failure
/// leaves `resample_outputs` and both streams untouched (`AllocationFailed`
/// never masks `OutOfSequenceRow`), and a no-output call is a true no-op.
#[cfg(feature = "yuv-packed")]
#[allow(clippy::too_many_arguments)]
pub(super) fn packed_yuv422_dual_filter_resample(
  luma_filter_stream: &mut Option<std::boxed::Box<FilterStream<u8>>>,
  rgb_filter_stream: &mut Option<std::boxed::Box<FilterStream<u8>>>,
  resample_outputs: &mut Option<FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  luma_scratch: &mut std::vec::Vec<u8>,
  rgb_scratch: &mut std::vec::Vec<u8>,
  w: usize,
  plan: &ResamplePlan,
  idx: usize,
  use_simd: bool,
  deinterleave_y: impl FnOnce(&mut [u8]),
  convert_rgb: impl FnOnce(&mut [u8]),
) -> Result<(), MixedSinkerError> {
  // Single-kernel filter tail — reject a BICUBLIN plan (its chroma windows are
  // read only by the `Yuv420p` per-plane route) before any state change.
  plan.ensure_single_kernel_filter()?;
  let need_luma = luma.is_some() || luma_u16.is_some();
  let need_color = rgb.is_some() || hsv.is_some() || rgba.is_some();

  let (fh, fv) = (
    plan
      .filter_h()
      .expect("filter plan carries horizontal windows"),
    plan
      .filter_v()
      .expect("filter plan carries vertical windows"),
  );

  // Sequence-counter row for the shared L1 preflight: whichever stream is fed
  // every row (all attached streams advance in lockstep), or `None` when no
  // output is attached so the preflight short-circuits to a no-op.
  let expected = if need_luma {
    Some(luma_filter_stream.as_ref().map_or(0, |s| s.next_y()))
  } else if need_color {
    Some(rgb_filter_stream.as_ref().map_or(0, |s| s.next_y()))
  } else {
    None
  };
  // Compare-only preflight (NO commit) — see `packed_yuv422_dual_resample`: the
  // output-set freeze and the stream inserts commit TOGETHER only after every
  // pre-feed allocation below has succeeded, so a first-row allocation failure
  // is state-atomic (retryable with a changed output attachment).
  if let core::ops::ControlFlow::Break(()) = super::resample_preflight_check_only(
    resample_outputs,
    luma,
    luma_u16,
    rgb,
    rgba,
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
  let new_luma = if need_luma && luma_filter_stream.is_none() {
    let stream = FilterStream::new(fh, fv, plan.src_w(), plan.src_h(), 1)?;
    Some(crate::resample::try_box(stream).map_err(|_| {
      MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
        crate::resample::PlanGeometry::new(plan.src_w(), plan.src_h(), plan.out_w(), plan.out_h()),
      ))
    })?)
  } else {
    None
  };
  let new_rgb = if need_color && rgb_filter_stream.is_none() {
    let stream = FilterStream::new(fh, fv, plan.src_w(), plan.src_h(), 3)?;
    Some(crate::resample::try_box(stream).map_err(|_| {
      MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
        crate::resample::PlanGeometry::new(plan.src_w(), plan.src_h(), plan.out_w(), plan.out_h()),
      ))
    })?)
  } else {
    None
  };
  // Grow BOTH source-width scratches (fallible) BEFORE the commit too (see
  // `packed_yuv422_dual_resample`): every pre-feed allocation ahead of the
  // freeze/insert, so `packed_yuv422_dual_feed_emit`'s own grows are no-ops.
  if need_luma {
    source_luma_scratch(luma_scratch, w, plan)?;
  }
  if need_color {
    source_rgb_scratch(rgb_scratch, w, plan)?;
  }
  if let Some(l) = new_luma {
    *luma_filter_stream = Some(l);
  }
  if let Some(r) = new_rgb {
    *rgb_filter_stream = Some(r);
  }
  frozen_outputs_check(
    resample_outputs,
    luma,
    luma_u16,
    rgb,
    rgba,
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

  packed_yuv422_dual_feed_emit(
    luma_filter_stream.as_mut(),
    rgb_filter_stream.as_mut(),
    rgb,
    rgba,
    luma,
    luma_u16,
    hsv,
    luma_scratch,
    rgb_scratch,
    w,
    plan,
    idx,
    use_simd,
    deinterleave_y,
    convert_rgb,
  )
}

/// Native fast-tier decimator for the 8-bit PACKED 4:2:2 YUV family
/// ([`Yuyv422`](crate::source::Yuyv422) `Y U Y V …` /
/// [`Uyvy422`](crate::source::Uyvy422) `U Y V Y …` /
/// [`Yvyu422`](crate::source::Yvyu422) `Y V Y U …`): the PACKED analog of
/// the semi-planar non-4:2:0 native wrapper
/// ([`super::semi_planar_8bit`]'s `semi_planar_process_native_non420`).
/// De-PACKS the fully-interleaved source row into the sink's separate
/// Y (`w`) / U (`w / 2`) / V (`w / 2`) scratch planes at the per-format byte
/// offsets, then reuses the planar twin's non-4:2:0 join verbatim
/// ([`yuv_planar_process_native`]) at [`Yuv422p`](crate::source::Yuv422p)
/// geometry (chroma `w / 2 x h`, `chroma_vsub = 1`) — so every output is
/// byte-identical to a [`Yuv422p`](crate::source::Yuv422p) native conversion
/// of those de-packed planes, and within ±1 LSB of the packed row-stage tier
/// (the conversion-order rounding caveat the planar tiers already carry).
/// Luma is bit-identical (both bin the same native Y).
///
/// `y0_off` / `y1_off` are the two Y byte positions within each 4-byte /
/// 2-pixel group; `u_off` / `v_off` the chroma positions. `chroma_h_phase` is
/// the RFC #238 horizontal chroma sampling phase folded into the chroma area
/// weights ([`ResamplePlan::area_chroma_422`]): `0.25` for the centered 4:2:2
/// group (`Center` / `Top` / `Bottom`), `0.0` for co-sited / unspecified. At
/// phase `0.0` the folded plan is byte-identical to the plain `area` plan, so
/// the co-sited output is untouched. Like the
/// semi-planar non-4:2:0 wrapper the chroma cadence is one row per Y row, so
/// the U / V de-pack runs on EVERY colour row. On luma-only / no-colour rows
/// only Y is de-packed — the join never reads chroma there, and the chroma
/// scratch (left as-is) is handed to the join as empty slices, so a
/// luma-only sink never plans or allocates chroma state (the lazy-chroma
/// contract `NativePlanarYuv::new` upholds under `need_color`).
///
/// 8-bit source, so no native-depth clamp is needed (the source's native
/// range is the full `u8` range and the join's averaging keeps every sample
/// in range).
///
/// Atomicity mirrors the semi-planar non-4:2:0 wrapper: the join's COMPLETE
/// pre-feed rejection preflight runs FIRST (via
/// [`native_planar_preflight_check_only`]), before the fallible Y / U / V
/// scratch grow, so a rejected row returns its deterministic typed error
/// (`OutOfSequenceRow` / `ResampleOutputsChanged`), never `AllocationFailed`,
/// and grows no sink state. It is compare-only (no output-set freeze), so the
/// scratch grow stays a pre-feed step ahead of the delegate's commit; the
/// de-pack writes only the private scratch, so no caller output is touched
/// until the delegate clears.
#[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
#[allow(clippy::too_many_arguments)]
fn packed_yuv422_process_native(
  plan: &ResamplePlan,
  native_planar: &mut Option<std::boxed::Box<NativePlanarYuv>>,
  y_scratch: &mut std::vec::Vec<u8>,
  u_scratch: &mut std::vec::Vec<u8>,
  v_scratch: &mut std::vec::Vec<u8>,
  resample_outputs: &mut Option<FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  rgb_scratch: &mut std::vec::Vec<u8>,
  packed: &[u8],
  y0_off: usize,
  y1_off: usize,
  u_off: usize,
  v_off: usize,
  chroma_h_phase: f64,
  matrix: ColorMatrix,
  full_range: bool,
  idx: usize,
  w: usize,
  h: usize,
  use_simd: bool,
) -> Result<(), MixedSinkerError> {
  let need_luma = luma.is_some() || luma_u16.is_some();
  let need_color = rgb.is_some() || rgba.is_some() || hsv.is_some();
  let cw = w / 2;

  // Run the join's COMPLETE pre-feed rejection preflight FIRST — before the
  // fallible Y / U / V de-pack scratch grow — so EVERY rejection case
  // (out-of-sequence first row OR mid-frame output change) returns its
  // deterministic typed error, never AllocationFailed under allocation
  // pressure, and leaves the scratch untouched (the crate's
  // preflight-atomicity contract). Compare-only (no output-set freeze), so the
  // scratch grow below stays a pre-feed step ahead of the delegate's commit.
  // `Ok(false)` is the no-output no-op: return without reserving.
  // `yuv_planar_process_native` re-runs this identical compare harmlessly and
  // owns the commit, keeping a single source of truth.
  if !native_planar_preflight_check_only(
    native_planar,
    resample_outputs,
    rgb,
    rgba,
    luma,
    luma_u16,
    hsv,
    idx,
    need_luma,
    need_color,
  )? {
    return Ok(());
  }

  // De-pack the interleaved row into the private Y / U / V scratch. Y is
  // always de-packed (the join bins Y for both luma and colour); U / V are
  // de-packed only on a colour row (chroma_vsub == 1: a chroma row per Y
  // row). The de-pack writes only this private scratch, so no caller output
  // is touched until the join's own preflight (re-run inside the delegate
  // below) clears. On luma-only / no-colour rows the join never reads chroma,
  // so the scratch is left as-is and the join gets empty U / V slices —
  // keeping a luma-only sink from planning or allocating chroma state.
  let grow = |scratch: &mut std::vec::Vec<u8>, len: usize| -> Result<(), MixedSinkerError> {
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
  };
  grow(y_scratch, w)?;
  for (i, group) in packed.chunks_exact(4).enumerate() {
    y_scratch[i * 2] = group[y0_off];
    y_scratch[i * 2 + 1] = group[y1_off];
  }
  if need_color {
    grow(u_scratch, cw)?;
    grow(v_scratch, cw)?;
    for (i, group) in packed.chunks_exact(4).enumerate() {
      u_scratch[i] = group[u_off];
      v_scratch[i] = group[v_off];
    }
  }

  let (u_plane, v_plane): (&[u8], &[u8]) = if need_color {
    (&u_scratch[..cw], &v_scratch[..cw])
  } else {
    (&[], &[])
  };
  yuv_planar_process_native(
    plan,
    native_planar,
    resample_outputs,
    rgb,
    rgba,
    luma,
    luma_u16,
    hsv,
    rgb_scratch,
    &y_scratch[..w],
    u_plane,
    v_plane,
    matrix,
    full_range,
    idx,
    w,
    h,
    1,
    || ResamplePlan::area_chroma_422(cw, h, plan.out_w(), plan.out_h(), chroma_h_phase, 0.0),
    use_simd,
  )
}

// The packed siblings of the planar [`reserve_420_chroma_full`] /
// [`upsample_420_chroma_center_h`] centered-siting staging (#302). A packed
// 4:2:2 row interleaves chroma 2:1 horizontally inline (`[Y0 U Y1 V]` for YUYV,
// `[U Y0 V Y1]` for UYVY), so the centered horizontal upsample first
// de-interleaves the row's Y into a full-width plane and its U / V into
// half-width planes, then reuses the planar twin's exact phase-0.5 kernel + the
// plain (non-primaries) 4:4:4 kernels — making a centered packed decode
// bit-identical to a [`Yuv422p`](crate::source::Yuv422p) decode of those
// de-interleaved planes on the shared matrix-tag path. (`ChromaDerivedNcl` is
// the lone exception: packed — like every format except `Yuv420p` — resolves it
// via the BT.709 matrix-tag fallback `Coefficients::for_matrix`, not
// `Yuv420p`'s primaries-derived path; the default and centered packed paths
// agree on that fallback, so they stay internally consistent.) Only the centered
// sitings (`Center` / `Top` / `Bottom`) reach here; every co-sited /
// unspecified siting keeps the default fused `*_to_*_row` decode, byte-identical
// to the pre-#302 output.

/// **Fallible preflight** for the packed 4:2:2 centered-siting de-interleave
/// scratch (#302): grows the full-width Y buffer (`width`) and the half-width
/// U / V buffers (`width / 2` each) so the later infallible de-interleave +
/// 4:4:4 decode reuse already-sized buffers. Split from the de-interleave (like
/// [`reserve_420_chroma_full`], called alongside it) so it runs **before any
/// output row is written** — the crate's preflight-ordering atomicity contract
/// (cf. the #180 resample fix): an allocator refusal must leave the output frame
/// *untouched*, never partially mutated. Reuses the native tier's de-pack
/// scratch (`width` Y + `width / 2` U / V), so a centered colour row and an area
/// native row never both run in one `process` call. A grow refusal is the typed,
/// recoverable [`ResampleError::AllocationFailed`]; `height` feeds the payload.
#[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
fn reserve_packed_center_chroma(
  y_full: &mut std::vec::Vec<u8>,
  u_half: &mut std::vec::Vec<u8>,
  v_half: &mut std::vec::Vec<u8>,
  width: usize,
  height: usize,
) -> Result<(), MixedSinkerError> {
  let grow = |scratch: &mut std::vec::Vec<u8>, len: usize| -> Result<(), MixedSinkerError> {
    if scratch.len() < len {
      scratch
        .try_reserve_exact(len - scratch.len())
        .map_err(|_| {
          MixedSinkerError::Resample(ResampleError::AllocationFailed(PlanGeometry::new(
            width, height, width, height,
          )))
        })?;
      scratch.resize(len, 0);
    }
    Ok(())
  };
  grow(y_full, width)?;
  let cw = width / 2;
  grow(u_half, cw)?;
  grow(v_half, cw)?;
  Ok(())
}

/// De-interleaves a packed 4:2:2 row's chroma into the already-reserved
/// half-width U / V scratch, then phase-0.5 upsamples each plane to full width in
/// `chroma_full`, returning the full-width `(u_full, v_full)` the 4:4:4 decode
/// kernels consume (#302). `u_off` / `v_off` are the chroma byte positions within
/// each 4-byte / 2-pixel group (YUYV: U at 1, V at 3; UYVY: U at 0, V at 2) — the
/// SAME offsets the native tier de-packs, so a U/V swap here would diverge from
/// the planar twin. The row's Y is de-interleaved separately by the format's own
/// `*_to_luma_row` kernel (the exact Y→luma derivation the default path uses).
///
/// **Infallible**: the caller must have run [`reserve_420_chroma_full`] and
/// [`reserve_packed_center_chroma`] up front (every centered output path does,
/// before any output write), so all three buffers are guaranteed long enough.
#[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
#[allow(clippy::too_many_arguments)]
fn packed_center_upsample_chroma<'s>(
  chroma_full: &'s mut [u8],
  u_half: &mut [u8],
  v_half: &mut [u8],
  packed: &[u8],
  width: usize,
  u_off: usize,
  v_off: usize,
  use_simd: bool,
) -> (&'s [u8], &'s [u8]) {
  let cw = width / 2;
  debug_assert!(
    u_half.len() >= cw && v_half.len() >= cw,
    "half-width chroma scratch must be reserved via reserve_packed_center_chroma first"
  );
  for (i, group) in packed.chunks_exact(4).take(cw).enumerate() {
    u_half[i] = group[u_off];
    v_half[i] = group[v_off];
  }
  upsample_420_chroma_center_h(chroma_full, &u_half[..cw], &v_half[..cw], width, use_simd)
}

// ---- Yuyv422 impl ------------------------------------------------------

impl<'a, R> MixedSinker<'a, Yuyv422, R> {
  /// Attaches a packed **8-bit** RGBA output buffer. Alpha is filled
  /// with constant `0xFF` (the source has no alpha channel).
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

  /// Attaches a **`u16`** luma output buffer. Y bytes are zero-extended
  /// to u16 (`out[x] = Y_byte as u16`). Length in u16 **elements**
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

impl<R> Yuyv422Sink for MixedSinker<'_, Yuyv422, R> {}

impl<R> PixelSink for MixedSinker<'_, Yuyv422, R> {
  type Input<'r> = Yuyv422Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::WidthAlignment(WidthAlignment::odd(
        self.width,
      )));
    }
    // New frame: restart the row-stage streams (lazily created in
    // `process`, so a direct-`process` caller that skips `begin_frame`
    // still gets a correctly initialized first frame) and drop the
    // frozen output set.
    if let Some(stream) = self.rgb_stream.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.luma_stream.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.rgb_filter_stream.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.luma_filter_stream.as_mut() {
      stream.reset();
    }
    // New frame: restart the native join and clear the per-frame frozen
    // native/row-stage route so the next frame may pick either tier; a
    // mid-frame flip stays rejected. Gated to the native tier's feature
    // intersection (the planar join the native tier reuses is compiled only
    // under `yuv-planar`).
    #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
    if let Some(native) = self.native_planar.as_mut() {
      native.reset();
    }
    #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
    {
      self.frozen_native_route = None;
      // RFC #238 S2b: clear the per-frame frozen 4:2:2 chroma siting so the next
      // frame may pick either phase; a mid-frame flip stays rejected.
      self.frozen_chroma_centered = None;
    }
    self.resample_outputs = None;
    Ok(())
  }

  fn process(&mut self, row: Yuyv422Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if w & 1 != 0 {
      return Err(MixedSinkerError::WidthAlignment(WidthAlignment::odd(w)));
    }

    let packed_expected =
      w.checked_mul(2)
        .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
          w, h, 2,
        )))?;
    if row.yuyv().len() != packed_expected {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Yuyv422Packed,
        idx,
        packed_expected,
        row.yuyv().len(),
      )));
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange(
        RowIndexOutOfRange::new(idx, self.height),
      ));
    }

    // Chroma siting (#302): drives the identity-plan horizontal chroma phase.
    // `Copy`, so read it before the field split-borrow below. Gated like its
    // only consumer (`chroma_422_center_sited_h` + the 4:4:4 kernels need
    // `yuv-planar`); a `yuv-packed`-only build keeps the default nearest decode.
    #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
    let chroma_location = self.chroma_location;

    let Self {
      rgb,
      rgba,
      luma,
      luma_u16,
      hsv,
      luma_scratch,
      rgb_scratch,
      plan,
      rgb_stream,
      luma_stream,
      rgb_filter_stream,
      luma_filter_stream,
      resample_outputs,
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      native,
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      native_planar,
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      packed_yuv_y_full,
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      packed_yuv_u_half,
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      packed_yuv_v_half,
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      frozen_native_route,
      // RFC #238 S2b: the 4:2:2 chroma phase frozen on the first output row.
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      frozen_chroma_centered,
      // Centered chroma-siting (#302) stages full-width U + V here.
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      chroma_full,
      ..
    } = self;
    let packed = row.yuyv();

    // Non-identity plan: feed the shared packed-YUV dual-stream tail —
    // luma de-interleaves the Y bytes and bins them; colour converts the
    // packed row to RGB and bins that. The span kind picks the engine —
    // area binning or signed-coefficient filter (both stage the same
    // de-interleaved Y / converted RGB and share the emit, so filter colour
    // equals the RGB filter of the converted pixels and luma stays native
    // Y). Freeze + sequence-check before staging, so a no-output sink stays
    // a no-op and an out-of-sequence row is rejected without allocating.
    if let Some(plan) = plan.as_ref() {
      let matrix = row.matrix();
      let full_range = row.full_range();
      // RFC #238 S2b — 4:2:2 horizontal chroma siting for Yuyv422, mirroring the
      // planar Yuv422p / semi-planar Nv16 twins. The centered group (`Center` /
      // `Top` / `Bottom`, [`chroma_422_center_sited_h`]) samples chroma at
      // `+0.25` chroma-sample; the co-sited / unspecified group is phase 0
      // (byte-identical to the pre-siting resample). The native fast tier folds
      // the phase into the chroma area weights (`area_chroma_422`); the filter
      // and row-stage tiers reconstruct full-width chroma (de-interleave +
      // phase-0.5 upsample) and decode 4:4:4. The co-sited path keeps the fused
      // `yuyv422_to_rgb_row` decode.
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      let center_sited = chroma_422_center_sited_h(chroma_location);
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      let chroma_h_phase = if center_sited {
        YUV422P_CENTERED_H_PHASE
      } else {
        0.0
      };
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      let want_color = rgb.is_some() || rgba.is_some() || hsv.is_some();
      // Whether this call carries any output — the EXACT set both tiers'
      // preflight tests. The route (and the siting phase) freezes only on an
      // output-bearing row a tier ACCEPTS; a no-output call must not freeze.
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      let need_output =
        luma.is_some() || luma_u16.is_some() || rgb.is_some() || rgba.is_some() || hsv.is_some();
      // RFC #238 S2b: freeze the effective 4:2:2 chroma siting on the first
      // output-bearing row (mirroring the Yuv422p / Nv16 twins' always-compiled
      // choke point). A later row observing a different phase — in sequence or
      // not — would bin a mixture of co-sited and centered chroma, so it is
      // rejected HERE before any reconstruction or dispatch; the matching SET
      // rides each tier's accept-time freeze below (never on a reject, so a
      // corrected retry is not falsely rejected).
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      if need_output
        && let Some(frozen) = *frozen_chroma_centered
        && frozen != center_sited
      {
        return Err(MixedSinkerError::ChromaSitingChanged(
          ChromaSitingChanged::new(idx),
        ));
      }
      // A `Filter` plan routes to the filter resampler (the native fast tier
      // is area-only and never sees a filter plan; the per-sink plan kind is
      // fixed at construction, so a filter sink bypasses the native/row-stage
      // route machinery entirely). Branched FIRST, before the native-route
      // guard below.
      if let SpanKind::Filter = plan.kind() {
        // Centered filter reconstructs full-width chroma (de-interleave +
        // phase-0.5 upsample) and decodes 4:4:4 on the de-interleaved Y, but
        // ONLY after the resample preflight (frozen-output + sequence), so an
        // out-of-sequence / rejected row is caught before the chroma reservation
        // (#180). `packed_yuv422_dual_filter_resample` re-runs the idempotent
        // preflight. A luma-only centered row never calls the RGB converter, so
        // it stays on the co-sited arm (which only bins luma). Co-sited keeps
        // the fused `yuyv422_to_rgb_row` decode.
        #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
        {
          // Reject a multi-kernel (BICUBLIN) filter plan BEFORE the centered
          // reserve below, mirroring the delegate's own first act (idempotent).
          plan.ensure_single_kernel_filter()?;
          if center_sited && want_color {
            let need_luma = luma.is_some() || luma_u16.is_some();
            let expected = if need_luma {
              luma_filter_stream.as_ref().map_or(0, |s| s.next_y())
            } else {
              rgb_filter_stream.as_ref().map_or(0, |s| s.next_y())
            };
            if let core::ops::ControlFlow::Break(()) = super::resample_preflight_check_only(
              resample_outputs,
              luma,
              luma_u16,
              rgb,
              rgba,
              &None,
              &None,
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
            reserve_420_chroma_full(chroma_full, w, h)?;
            reserve_packed_center_chroma(
              packed_yuv_y_full,
              packed_yuv_u_half,
              packed_yuv_v_half,
              w,
              h,
            )?;
            yuyv422_to_luma_row(packed, &mut packed_yuv_y_full[..w], w, use_simd);
            let (u_full, v_full) = packed_center_upsample_chroma(
              chroma_full,
              packed_yuv_u_half,
              packed_yuv_v_half,
              packed,
              w,
              1,
              3,
              use_simd,
            );
            let r = packed_yuv422_dual_filter_resample(
              luma_filter_stream,
              rgb_filter_stream,
              resample_outputs,
              rgb,
              rgba,
              luma,
              luma_u16,
              hsv,
              luma_scratch,
              rgb_scratch,
              w,
              plan,
              idx,
              use_simd,
              |scratch| yuyv422_to_luma_row(packed, scratch, w, use_simd),
              |scratch| {
                yuv_444_to_rgb_row(
                  &packed_yuv_y_full[..w],
                  u_full,
                  v_full,
                  scratch,
                  w,
                  matrix,
                  full_range,
                  use_simd,
                );
              },
            );
            if r.is_ok() && need_output && frozen_chroma_centered.is_none() {
              *frozen_chroma_centered = Some(center_sited);
            }
            return r;
          }
        }
        packed_yuv422_dual_filter_resample(
          luma_filter_stream,
          rgb_filter_stream,
          resample_outputs,
          rgb,
          rgba,
          luma,
          luma_u16,
          hsv,
          luma_scratch,
          rgb_scratch,
          w,
          plan,
          idx,
          use_simd,
          |scratch| yuyv422_to_luma_row(packed, scratch, w, use_simd),
          |scratch| yuyv422_to_rgb_row(packed, scratch, w, matrix, full_range, use_simd),
        )?;
        #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
        if need_output && frozen_chroma_centered.is_none() {
          *frozen_chroma_centered = Some(center_sited);
        }
        return Ok(());
      }
      // Area plan. When the native tier is enabled (and the planar join it
      // reuses is compiled in), de-pack the interleaved row into Y / U / V
      // scratch and bin those planes at output resolution, converting once
      // per output row (YUYV: Y at 0,2 / U at 1 / V at 3). Otherwise (or under
      // `with_native(false)`) take the row-stage tier: bin the de-interleaved
      // Y for luma, convert the packed row to a source-width RGB row and bin
      // that.
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      {
        // The RFC #238 splice stage. A filter plan already returned above, so
        // `area_plan` is true and the selector reproduces the former `*native`
        // boolean bit-for-bit (`cfg!` is true wherever this block compiles).
        let take_native = matches!(
          select_insertion_point(
            AveragingDomain::Encoded,
            InsertionContext {
              native_eligible: cfg!(all(feature = "yuv-packed", feature = "yuv-planar")),
              with_native: *native,
              area_plan: true,
            },
          ),
          InsertionPoint::NativeCodes
        );
        // Reject a mid-frame native/row-stage route flip BEFORE either tier's
        // dispatch (the #186 CHECK-before / SET-after template).
        if need_output
          && let Some(frozen) = *frozen_native_route
          && frozen != take_native
        {
          return Err(MixedSinkerError::NativeRouteChanged(
            NativeRouteChanged::new(idx),
          ));
        }
        if take_native {
          // RFC #238 S2b point-of-use siting invalidation, mirroring the Nv16
          // native arm: `chroma_location` can change at ANY point before this row
          // (including AFTER `begin_frame`, before row 0), so re-check the cached
          // join HERE and drop it when its folded chroma plan was built for a
          // different phase; `packed_yuv422_process_native` (via
          // `yuv_planar_process_native`) then rebuilds it with the current siting.
          // Retry-atomic: drop ONLY on the IN-SEQUENCE fresh-frame first row
          // (`idx == 0`, `next_y() == 0`); an out-of-sequence first row is left
          // for the delegate to reject against the INTACT join. A luma-only join
          // carries no chroma plan (siting-independent), a no-output sink built no
          // join. Transactional: move the stale join OUT, let the delegate build
          // the replacement into `native_planar` (its build runs BEFORE it
          // inserts), and restore the intact prior-phase join on a rejected
          // rebuild so the REJECTED row mutates nothing.
          let stale_native = idx == 0
            && native_planar.as_ref().is_some_and(|join| {
              join.has_chroma() && join.chroma_centered() != center_sited && join.next_y() == 0
            });
          let prev_native = if stale_native {
            native_planar.take()
          } else {
            None
          };
          // Dispatch first; freeze the route + siting ONLY after the call returns
          // Ok on an output-bearing row.
          let native_result = packed_yuv422_process_native(
            plan,
            native_planar,
            packed_yuv_y_full,
            packed_yuv_u_half,
            packed_yuv_v_half,
            resample_outputs,
            rgb,
            rgba,
            luma,
            luma_u16,
            hsv,
            rgb_scratch,
            packed,
            0,
            2,
            1,
            3,
            chroma_h_phase,
            matrix,
            full_range,
            idx,
            w,
            h,
            use_simd,
          );
          // Restore the taken stale-phase join if the delegate's rebuild was
          // rejected at any pre-feed step: it leaves the field `None` and
          // `resample_outputs` uncommitted on such a failure, so restoring the
          // intact prior-phase join leaves the rejected row mutating nothing. A
          // non-stale row (first-ever / colour-capability rebuild) took nothing.
          if stale_native && native_result.is_err() {
            *native_planar = prev_native;
          }
          native_result?;
          if frozen_native_route.is_none() && need_output {
            *frozen_native_route = Some(true);
          }
          // RFC #238 S2b: freeze the siting on the same accepted output row.
          if frozen_chroma_centered.is_none() && need_output {
            *frozen_chroma_centered = Some(center_sited);
          }
          return Ok(());
        }
      }
      // Row-stage tail (the only route under a `yuv-packed`-solo build).
      // Centered colour reconstructs full-width chroma (de-interleave +
      // phase-0.5 upsample) and decodes 4:4:4 — but ONLY after the resample
      // preflight (frozen-output + sequence), so an out-of-sequence / rejected
      // row is caught before the chroma reservation (#180). A luma-only centered
      // row never calls the RGB converter, so it stays on the co-sited arm
      // (which only bins luma). `packed_yuv422_dual_resample` re-runs the
      // idempotent preflight. Dispatch, then under the native tier freeze the
      // route + siting only when the call accepts an output-bearing row.
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      if center_sited && want_color {
        let need_luma = luma.is_some() || luma_u16.is_some();
        let expected = if need_luma {
          luma_stream.as_ref().map_or(0, |s| s.next_y())
        } else {
          rgb_stream.as_ref().map_or(0, |s| s.next_y())
        };
        if let core::ops::ControlFlow::Break(()) = super::resample_preflight_check_only(
          resample_outputs,
          luma,
          luma_u16,
          rgb,
          rgba,
          &None,
          &None,
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
        reserve_420_chroma_full(chroma_full, w, h)?;
        reserve_packed_center_chroma(
          packed_yuv_y_full,
          packed_yuv_u_half,
          packed_yuv_v_half,
          w,
          h,
        )?;
        yuyv422_to_luma_row(packed, &mut packed_yuv_y_full[..w], w, use_simd);
        let (u_full, v_full) = packed_center_upsample_chroma(
          chroma_full,
          packed_yuv_u_half,
          packed_yuv_v_half,
          packed,
          w,
          1,
          3,
          use_simd,
        );
        packed_yuv422_dual_resample(
          luma_stream,
          rgb_stream,
          resample_outputs,
          rgb,
          rgba,
          luma,
          luma_u16,
          hsv,
          luma_scratch,
          rgb_scratch,
          w,
          plan,
          idx,
          use_simd,
          |scratch| yuyv422_to_luma_row(packed, scratch, w, use_simd),
          |scratch| {
            yuv_444_to_rgb_row(
              &packed_yuv_y_full[..w],
              u_full,
              v_full,
              scratch,
              w,
              matrix,
              full_range,
              use_simd,
            );
          },
        )?;
        if frozen_native_route.is_none() && need_output {
          *frozen_native_route = Some(false);
        }
        if frozen_chroma_centered.is_none() && need_output {
          *frozen_chroma_centered = Some(center_sited);
        }
        return Ok(());
      }
      packed_yuv422_dual_resample(
        luma_stream,
        rgb_stream,
        resample_outputs,
        rgb,
        rgba,
        luma,
        luma_u16,
        hsv,
        luma_scratch,
        rgb_scratch,
        w,
        plan,
        idx,
        use_simd,
        |scratch| yuyv422_to_luma_row(packed, scratch, w, use_simd),
        |scratch| yuyv422_to_rgb_row(packed, scratch, w, matrix, full_range, use_simd),
      )?;
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      {
        if frozen_native_route.is_none() && need_output {
          *frozen_native_route = Some(false);
        }
        if frozen_chroma_centered.is_none() && need_output {
          *frozen_chroma_centered = Some(center_sited);
        }
      }
      return Ok(());
    }

    // Strategy A output mode resolution — resolved BEFORE any output write so
    // the atomicity preflight below runs ahead of the luma writes.
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();

    // No-output guard (#302, cf. the planar / semi-planar siblings): a `process`
    // call with NO output attached must run NOTHING — no `idx * w` offset
    // arithmetic (a no-output call never ran an attach-time `w x h` validation,
    // so on a 32-bit target an absurd geometry would overflow that offset), no
    // allocation, no state mutation. Returning HERE — before the offsets AND the
    // centered chroma preflight — keeps a no-output row panic-free and
    // allocation-free.
    let need_output = want_rgb || want_rgba || want_hsv || luma.is_some() || luma_u16.is_some();
    if !need_output {
      return Ok(());
    }

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // Chroma siting (#302): the centered horizontal sitings reconstruct chroma at
    // the phase-0.5 position then decode via the 4:4:4 kernels on the
    // de-interleaved Y; the default / co-sited path keeps the byte-identical
    // nearest-neighbor decode. 4:2:2 is horizontally subsampled only — there is no
    // vertical blend or chroma lookback (cf. the Yuv420p `Bottom` path).
    #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
    let center_sited = chroma_422_center_sited_h(chroma_location);

    // Per-frame chroma-siting freeze (RFC #238, mirroring the resample-path guard
    // above): the first output-bearing row pins the phase; a later row whose siting
    // flipped would decode a mixture of centered and co-sited chroma into ONE
    // frame, so reject it here BEFORE any scratch reserve or output write.
    // `begin_frame` clears the freeze so the next frame may pick either phase.
    #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
    if need_output
      && let Some(frozen) = *frozen_chroma_centered
      && frozen != center_sited
    {
      return Err(MixedSinkerError::ChromaSitingChanged(
        ChromaSitingChanged::new(idx),
      ));
    }

    // Atomicity preflight (#302 / #308, cf. the crate's #180 resample fix and the
    // planar / semi-planar siblings): reserve EVERY fallible row scratch this row
    // needs BEFORE any output row (luma / luma_u16 included) is written, so an
    // allocator refusal returns a typed `AllocationFailed` leaving the output
    // frame untouched rather than partially mutated. Two groups can grow:
    //  1. the centered-siting full-width chroma (`chroma_full`) plus the packed
    //     de-interleave scratch (full-width Y + half-width U / V); and
    //  2. the RGB row buffer, reserved exactly when a colour decode needs an RGB
    //     row but no caller RGB buffer is borrowable — `want_hsv && want_rgba &&
    //     !want_rgb` (`rgb_row_buf_or_scratch`'s own scratch arm; an attached RGB
    //     buffer is borrowed instead and never allocates).
    // The later de-interleave / decode calls reuse the already-sized buffers, so
    // the default path is byte-identical; only the failure-path ordering changes.
    #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
    if center_sited && (want_rgb || want_rgba || want_hsv) {
      reserve_420_chroma_full(chroma_full, w, h)?;
      reserve_packed_center_chroma(
        packed_yuv_y_full,
        packed_yuv_u_half,
        packed_yuv_v_half,
        w,
        h,
      )?;
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

    // Freeze the phase on the first output-bearing row — AFTER the fallible scratch
    // reserves above have succeeded, so an `AllocationFailed` row stays retryable
    // (frozen stays unset); later rows are checked against it up top.
    #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
    if need_output && frozen_chroma_centered.is_none() {
      *frozen_chroma_centered = Some(center_sited);
    }

    // Luma u8 — extract Y bytes from packed plane via dedicated kernel.
    if let Some(luma) = luma.as_deref_mut() {
      yuyv422_to_luma_row(
        packed,
        &mut luma[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }
    // Luma u16 — zero-extend Y bytes to u16.
    if let Some(buf) = luma_u16.as_deref_mut() {
      yuyv422_to_luma_u16_row(
        packed,
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    // HSV-without-RGB-or-RGBA goes through the direct `yuyv422_to_hsv_row`
    // kernel (no source-width RGB scratch). When RGB or RGBA is *also*
    // attached the RGB kernel runs anyway, so HSV derives off that buffer
    // for free (the cheap path) and `need_rgb_kernel` keeps it alive.
    let want_hsv_direct = want_hsv && !want_rgb && !want_rgba;
    let need_rgb_kernel = want_rgb || (want_hsv && want_rgba);

    if want_hsv_direct {
      let hsv = hsv.as_mut().expect("want_hsv_direct implies hsv attached");
      let (h_out, s_out, v_out) = hsv.hsv();
      // Centered siting (#302): de-interleave Y + phase-0.5 upsample chroma to
      // full width, then run the 4:4:4 HSV kernel (scratch reserved above).
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      if center_sited {
        yuyv422_to_luma_row(packed, &mut packed_yuv_y_full[..w], w, use_simd);
        let (u_full, v_full) = packed_center_upsample_chroma(
          chroma_full,
          packed_yuv_u_half,
          packed_yuv_v_half,
          packed,
          w,
          1,
          3,
          use_simd,
        );
        yuv_444_to_hsv_row(
          &packed_yuv_y_full[..w],
          u_full,
          v_full,
          &mut h_out[one_plane_start..one_plane_end],
          &mut s_out[one_plane_start..one_plane_end],
          &mut v_out[one_plane_start..one_plane_end],
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
        );
        return Ok(());
      }
      yuyv422_to_hsv_row(
        packed,
        &mut h_out[one_plane_start..one_plane_end],
        &mut s_out[one_plane_start..one_plane_end],
        &mut v_out[one_plane_start..one_plane_end],
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
      return Ok(());
    }

    // Standalone RGBA fast path — no RGB / HSV requested. Run the
    // dedicated RGBA kernel directly into the output buffer; avoids
    // both the scratch allocation and the RGB→RGBA expand pass.
    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      // Centered siting (#302): full-width phase-0.5 chroma + the 4:4:4 RGBA
      // kernel on the de-interleaved Y; default keeps `yuyv422_to_rgba_row`.
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      if center_sited {
        yuyv422_to_luma_row(packed, &mut packed_yuv_y_full[..w], w, use_simd);
        let (u_full, v_full) = packed_center_upsample_chroma(
          chroma_full,
          packed_yuv_u_half,
          packed_yuv_v_half,
          packed,
          w,
          1,
          3,
          use_simd,
        );
        yuv_444_to_rgba_row(
          &packed_yuv_y_full[..w],
          u_full,
          v_full,
          rgba_row,
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
        );
        return Ok(());
      }
      yuyv422_to_rgba_row(
        packed,
        rgba_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
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
    // Centered siting (#302) decodes via the 4:4:4 RGB kernel on the
    // de-interleaved Y + phase-0.5 upsampled chroma; the HSV / RGBA follow-ups
    // below derive off the produced RGB row either way.
    #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
    let centered = if center_sited {
      yuyv422_to_luma_row(packed, &mut packed_yuv_y_full[..w], w, use_simd);
      let (u_full, v_full) = packed_center_upsample_chroma(
        chroma_full,
        packed_yuv_u_half,
        packed_yuv_v_half,
        packed,
        w,
        1,
        3,
        use_simd,
      );
      yuv_444_to_rgb_row(
        &packed_yuv_y_full[..w],
        u_full,
        v_full,
        rgb_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
      true
    } else {
      false
    };
    #[cfg(not(all(feature = "yuv-packed", feature = "yuv-planar")))]
    let centered = false;
    if !centered {
      yuyv422_to_rgb_row(packed, rgb_row, w, row.matrix(), row.full_range(), use_simd);
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

    // Strategy A: when both RGB and RGBA are requested, derive RGBA
    // from the just-computed RGB row instead of running a second
    // YUV→RGB kernel.
    if let Some(buf) = rgba.as_deref_mut() {
      let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
    }

    Ok(())
  }
}

// ---- Uyvy422 impl ------------------------------------------------------

impl<'a, R> MixedSinker<'a, Uyvy422, R> {
  /// Attaches a packed **8-bit** RGBA output buffer. Alpha is filled
  /// with constant `0xFF` (the source has no alpha channel).
  ///
  /// See [`MixedSinker::<Yuyv422>::with_rgba`] for the same rationale
  /// and constraints; UYVY differs only in byte position (Y in odd
  /// vs even slots).
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

  /// Attaches a **`u16`** luma output buffer. Y bytes (at offset 1 of
  /// each UYVY pair) are zero-extended to u16. Length in u16 **elements**
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

impl<R> Uyvy422Sink for MixedSinker<'_, Uyvy422, R> {}

impl<R> PixelSink for MixedSinker<'_, Uyvy422, R> {
  type Input<'r> = Uyvy422Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::WidthAlignment(WidthAlignment::odd(
        self.width,
      )));
    }
    if let Some(stream) = self.rgb_stream.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.luma_stream.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.rgb_filter_stream.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.luma_filter_stream.as_mut() {
      stream.reset();
    }
    #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
    if let Some(native) = self.native_planar.as_mut() {
      native.reset();
    }
    #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
    {
      self.frozen_native_route = None;
      // RFC #238 S2b: clear the per-frame frozen 4:2:2 chroma siting so the next
      // frame may pick either phase; a mid-frame flip stays rejected.
      self.frozen_chroma_centered = None;
    }
    self.resample_outputs = None;
    Ok(())
  }

  fn process(&mut self, row: Uyvy422Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if w & 1 != 0 {
      return Err(MixedSinkerError::WidthAlignment(WidthAlignment::odd(w)));
    }

    let packed_expected =
      w.checked_mul(2)
        .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
          w, h, 2,
        )))?;
    if row.uyvy().len() != packed_expected {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Uyvy422Packed,
        idx,
        packed_expected,
        row.uyvy().len(),
      )));
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange(
        RowIndexOutOfRange::new(idx, self.height),
      ));
    }

    // Chroma siting (#302): drives the identity-plan horizontal chroma phase.
    // `Copy`, so read it before the field split-borrow below. Gated like its
    // only consumer (`chroma_422_center_sited_h` + the 4:4:4 kernels need
    // `yuv-planar`); a `yuv-packed`-only build keeps the default nearest decode.
    #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
    let chroma_location = self.chroma_location;

    let Self {
      rgb,
      rgba,
      luma,
      luma_u16,
      hsv,
      luma_scratch,
      rgb_scratch,
      plan,
      rgb_stream,
      luma_stream,
      rgb_filter_stream,
      luma_filter_stream,
      resample_outputs,
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      native,
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      native_planar,
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      packed_yuv_y_full,
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      packed_yuv_u_half,
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      packed_yuv_v_half,
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      frozen_native_route,
      // RFC #238 S2b: the 4:2:2 chroma phase frozen on the first output row.
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      frozen_chroma_centered,
      // Centered chroma-siting (#302) stages full-width U + V here.
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      chroma_full,
      ..
    } = self;
    let packed = row.uyvy();

    if let Some(plan) = plan.as_ref() {
      let matrix = row.matrix();
      let full_range = row.full_range();
      // RFC #238 S2b — 4:2:2 horizontal chroma siting for Uyvy422, mirroring the
      // Yuyv422 twin above (UYVY: U at 0, V at 2). The centered group samples
      // chroma at `+0.25` chroma-sample; the co-sited / unspecified group is
      // phase 0 (byte-identical to the pre-siting resample). See the Yuyv422 impl
      // for the full per-tier rationale.
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      let center_sited = chroma_422_center_sited_h(chroma_location);
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      let chroma_h_phase = if center_sited {
        YUV422P_CENTERED_H_PHASE
      } else {
        0.0
      };
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      let want_color = rgb.is_some() || rgba.is_some() || hsv.is_some();
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      let need_output =
        luma.is_some() || luma_u16.is_some() || rgb.is_some() || rgba.is_some() || hsv.is_some();
      // RFC #238 S2b: freeze the effective 4:2:2 chroma siting on the first
      // output-bearing row; a later row observing a different phase is rejected
      // HERE before any reconstruction or dispatch (see the Yuyv422 twin).
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      if need_output
        && let Some(frozen) = *frozen_chroma_centered
        && frozen != center_sited
      {
        return Err(MixedSinkerError::ChromaSitingChanged(
          ChromaSitingChanged::new(idx),
        ));
      }
      // Filter plan: native is area-only, so route straight to the filter
      // resampler (see the Yuyv422 impl above for the full rationale).
      if let SpanKind::Filter = plan.kind() {
        #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
        {
          plan.ensure_single_kernel_filter()?;
          if center_sited && want_color {
            let need_luma = luma.is_some() || luma_u16.is_some();
            let expected = if need_luma {
              luma_filter_stream.as_ref().map_or(0, |s| s.next_y())
            } else {
              rgb_filter_stream.as_ref().map_or(0, |s| s.next_y())
            };
            if let core::ops::ControlFlow::Break(()) = super::resample_preflight_check_only(
              resample_outputs,
              luma,
              luma_u16,
              rgb,
              rgba,
              &None,
              &None,
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
            reserve_420_chroma_full(chroma_full, w, h)?;
            reserve_packed_center_chroma(
              packed_yuv_y_full,
              packed_yuv_u_half,
              packed_yuv_v_half,
              w,
              h,
            )?;
            uyvy422_to_luma_row(packed, &mut packed_yuv_y_full[..w], w, use_simd);
            let (u_full, v_full) = packed_center_upsample_chroma(
              chroma_full,
              packed_yuv_u_half,
              packed_yuv_v_half,
              packed,
              w,
              0,
              2,
              use_simd,
            );
            let r = packed_yuv422_dual_filter_resample(
              luma_filter_stream,
              rgb_filter_stream,
              resample_outputs,
              rgb,
              rgba,
              luma,
              luma_u16,
              hsv,
              luma_scratch,
              rgb_scratch,
              w,
              plan,
              idx,
              use_simd,
              |scratch| uyvy422_to_luma_row(packed, scratch, w, use_simd),
              |scratch| {
                yuv_444_to_rgb_row(
                  &packed_yuv_y_full[..w],
                  u_full,
                  v_full,
                  scratch,
                  w,
                  matrix,
                  full_range,
                  use_simd,
                );
              },
            );
            if r.is_ok() && need_output && frozen_chroma_centered.is_none() {
              *frozen_chroma_centered = Some(center_sited);
            }
            return r;
          }
        }
        packed_yuv422_dual_filter_resample(
          luma_filter_stream,
          rgb_filter_stream,
          resample_outputs,
          rgb,
          rgba,
          luma,
          luma_u16,
          hsv,
          luma_scratch,
          rgb_scratch,
          w,
          plan,
          idx,
          use_simd,
          |scratch| uyvy422_to_luma_row(packed, scratch, w, use_simd),
          |scratch| uyvy422_to_rgb_row(packed, scratch, w, matrix, full_range, use_simd),
        )?;
        #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
        if need_output && frozen_chroma_centered.is_none() {
          *frozen_chroma_centered = Some(center_sited);
        }
        return Ok(());
      }
      // Area plan — UYVY: Y at 1,3 / U at 0 / V at 2.
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      {
        // The RFC #238 splice stage. A filter plan already returned above, so
        // `area_plan` is true and the selector reproduces the former `*native`
        // boolean bit-for-bit (`cfg!` is true wherever this block compiles).
        let take_native = matches!(
          select_insertion_point(
            AveragingDomain::Encoded,
            InsertionContext {
              native_eligible: cfg!(all(feature = "yuv-packed", feature = "yuv-planar")),
              with_native: *native,
              area_plan: true,
            },
          ),
          InsertionPoint::NativeCodes
        );
        if need_output
          && let Some(frozen) = *frozen_native_route
          && frozen != take_native
        {
          return Err(MixedSinkerError::NativeRouteChanged(
            NativeRouteChanged::new(idx),
          ));
        }
        if take_native {
          // RFC #238 S2b point-of-use siting invalidation (see the Yuyv422 twin):
          // drop the cached join at the in-sequence fresh-frame first row when its
          // folded chroma plan was built for a different phase, transactionally.
          let stale_native = idx == 0
            && native_planar.as_ref().is_some_and(|join| {
              join.has_chroma() && join.chroma_centered() != center_sited && join.next_y() == 0
            });
          let prev_native = if stale_native {
            native_planar.take()
          } else {
            None
          };
          let native_result = packed_yuv422_process_native(
            plan,
            native_planar,
            packed_yuv_y_full,
            packed_yuv_u_half,
            packed_yuv_v_half,
            resample_outputs,
            rgb,
            rgba,
            luma,
            luma_u16,
            hsv,
            rgb_scratch,
            packed,
            1,
            3,
            0,
            2,
            chroma_h_phase,
            matrix,
            full_range,
            idx,
            w,
            h,
            use_simd,
          );
          if stale_native && native_result.is_err() {
            *native_planar = prev_native;
          }
          native_result?;
          if frozen_native_route.is_none() && need_output {
            *frozen_native_route = Some(true);
          }
          if frozen_chroma_centered.is_none() && need_output {
            *frozen_chroma_centered = Some(center_sited);
          }
          return Ok(());
        }
      }
      // Row-stage tail (see the Yuyv422 twin): centered colour reconstructs
      // full-width chroma after the resample preflight, then decodes 4:4:4;
      // a luma-only centered row stays on the co-sited arm.
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      if center_sited && want_color {
        let need_luma = luma.is_some() || luma_u16.is_some();
        let expected = if need_luma {
          luma_stream.as_ref().map_or(0, |s| s.next_y())
        } else {
          rgb_stream.as_ref().map_or(0, |s| s.next_y())
        };
        if let core::ops::ControlFlow::Break(()) = super::resample_preflight_check_only(
          resample_outputs,
          luma,
          luma_u16,
          rgb,
          rgba,
          &None,
          &None,
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
        reserve_420_chroma_full(chroma_full, w, h)?;
        reserve_packed_center_chroma(
          packed_yuv_y_full,
          packed_yuv_u_half,
          packed_yuv_v_half,
          w,
          h,
        )?;
        uyvy422_to_luma_row(packed, &mut packed_yuv_y_full[..w], w, use_simd);
        let (u_full, v_full) = packed_center_upsample_chroma(
          chroma_full,
          packed_yuv_u_half,
          packed_yuv_v_half,
          packed,
          w,
          0,
          2,
          use_simd,
        );
        packed_yuv422_dual_resample(
          luma_stream,
          rgb_stream,
          resample_outputs,
          rgb,
          rgba,
          luma,
          luma_u16,
          hsv,
          luma_scratch,
          rgb_scratch,
          w,
          plan,
          idx,
          use_simd,
          |scratch| uyvy422_to_luma_row(packed, scratch, w, use_simd),
          |scratch| {
            yuv_444_to_rgb_row(
              &packed_yuv_y_full[..w],
              u_full,
              v_full,
              scratch,
              w,
              matrix,
              full_range,
              use_simd,
            );
          },
        )?;
        if frozen_native_route.is_none() && need_output {
          *frozen_native_route = Some(false);
        }
        if frozen_chroma_centered.is_none() && need_output {
          *frozen_chroma_centered = Some(center_sited);
        }
        return Ok(());
      }
      packed_yuv422_dual_resample(
        luma_stream,
        rgb_stream,
        resample_outputs,
        rgb,
        rgba,
        luma,
        luma_u16,
        hsv,
        luma_scratch,
        rgb_scratch,
        w,
        plan,
        idx,
        use_simd,
        |scratch| uyvy422_to_luma_row(packed, scratch, w, use_simd),
        |scratch| uyvy422_to_rgb_row(packed, scratch, w, matrix, full_range, use_simd),
      )?;
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      {
        if frozen_native_route.is_none() && need_output {
          *frozen_native_route = Some(false);
        }
        if frozen_chroma_centered.is_none() && need_output {
          *frozen_chroma_centered = Some(center_sited);
        }
      }
      return Ok(());
    }

    // Strategy A output mode resolution — resolved BEFORE any output write so
    // the atomicity preflight below runs ahead of the luma writes.
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();

    // No-output guard (#302, cf. the planar / semi-planar siblings): a `process`
    // call with NO output attached must run NOTHING — no `idx * w` offset
    // arithmetic (a no-output call never ran an attach-time `w x h` validation,
    // so on a 32-bit target an absurd geometry would overflow that offset), no
    // allocation, no state mutation. Returning HERE — before the offsets AND the
    // centered chroma preflight — keeps a no-output row panic-free and
    // allocation-free.
    let need_output = want_rgb || want_rgba || want_hsv || luma.is_some() || luma_u16.is_some();
    if !need_output {
      return Ok(());
    }

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // Chroma siting (#302): the centered horizontal sitings reconstruct chroma at
    // the phase-0.5 position then decode via the 4:4:4 kernels on the
    // de-interleaved Y; the default / co-sited path keeps the byte-identical
    // nearest-neighbor decode. 4:2:2 is horizontally subsampled only — there is no
    // vertical blend or chroma lookback (cf. the Yuv420p `Bottom` path).
    #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
    let center_sited = chroma_422_center_sited_h(chroma_location);

    // Per-frame chroma-siting freeze (RFC #238, mirroring the resample-path guard
    // above): the first output-bearing row pins the phase; a later row whose siting
    // flipped would decode a mixture of centered and co-sited chroma into ONE
    // frame, so reject it here BEFORE any scratch reserve or output write.
    // `begin_frame` clears the freeze so the next frame may pick either phase.
    #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
    if need_output
      && let Some(frozen) = *frozen_chroma_centered
      && frozen != center_sited
    {
      return Err(MixedSinkerError::ChromaSitingChanged(
        ChromaSitingChanged::new(idx),
      ));
    }

    // Atomicity preflight (#302 / #308, cf. the crate's #180 resample fix and the
    // planar / semi-planar siblings): reserve EVERY fallible row scratch this row
    // needs BEFORE any output row (luma / luma_u16 included) is written, so an
    // allocator refusal returns a typed `AllocationFailed` leaving the output
    // frame untouched rather than partially mutated. Two groups can grow:
    //  1. the centered-siting full-width chroma (`chroma_full`) plus the packed
    //     de-interleave scratch (full-width Y + half-width U / V); and
    //  2. the RGB row buffer, reserved exactly when a colour decode needs an RGB
    //     row but no caller RGB buffer is borrowable — `want_hsv && want_rgba &&
    //     !want_rgb` (`rgb_row_buf_or_scratch`'s own scratch arm; an attached RGB
    //     buffer is borrowed instead and never allocates).
    // The later de-interleave / decode calls reuse the already-sized buffers, so
    // the default path is byte-identical; only the failure-path ordering changes.
    #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
    if center_sited && (want_rgb || want_rgba || want_hsv) {
      reserve_420_chroma_full(chroma_full, w, h)?;
      reserve_packed_center_chroma(
        packed_yuv_y_full,
        packed_yuv_u_half,
        packed_yuv_v_half,
        w,
        h,
      )?;
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

    // Freeze the phase on the first output-bearing row — AFTER the fallible scratch
    // reserves above have succeeded, so an `AllocationFailed` row stays retryable
    // (frozen stays unset); later rows are checked against it up top.
    #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
    if need_output && frozen_chroma_centered.is_none() {
      *frozen_chroma_centered = Some(center_sited);
    }

    if let Some(luma) = luma.as_deref_mut() {
      uyvy422_to_luma_row(
        packed,
        &mut luma[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }
    // Luma u16 — zero-extend Y bytes to u16.
    if let Some(buf) = luma_u16.as_deref_mut() {
      uyvy422_to_luma_u16_row(
        packed,
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    // HSV-without-RGB-or-RGBA goes through the direct `uyvy422_to_hsv_row`
    // kernel (no source-width RGB scratch). When RGB or RGBA is *also*
    // attached the RGB kernel runs anyway, so HSV derives off that buffer
    // for free (the cheap path) and `need_rgb_kernel` keeps it alive.
    let want_hsv_direct = want_hsv && !want_rgb && !want_rgba;
    let need_rgb_kernel = want_rgb || (want_hsv && want_rgba);

    if want_hsv_direct {
      let hsv = hsv.as_mut().expect("want_hsv_direct implies hsv attached");
      let (h_out, s_out, v_out) = hsv.hsv();
      // Centered siting (#302): de-interleave Y + phase-0.5 upsample chroma to
      // full width, then run the 4:4:4 HSV kernel (scratch reserved above).
      // UYVY: Y at 1,3 / U at 0 / V at 2.
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      if center_sited {
        uyvy422_to_luma_row(packed, &mut packed_yuv_y_full[..w], w, use_simd);
        let (u_full, v_full) = packed_center_upsample_chroma(
          chroma_full,
          packed_yuv_u_half,
          packed_yuv_v_half,
          packed,
          w,
          0,
          2,
          use_simd,
        );
        yuv_444_to_hsv_row(
          &packed_yuv_y_full[..w],
          u_full,
          v_full,
          &mut h_out[one_plane_start..one_plane_end],
          &mut s_out[one_plane_start..one_plane_end],
          &mut v_out[one_plane_start..one_plane_end],
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
        );
        return Ok(());
      }
      uyvy422_to_hsv_row(
        packed,
        &mut h_out[one_plane_start..one_plane_end],
        &mut s_out[one_plane_start..one_plane_end],
        &mut v_out[one_plane_start..one_plane_end],
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
      return Ok(());
    }

    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      // Centered siting (#302): full-width phase-0.5 chroma + the 4:4:4 RGBA
      // kernel on the de-interleaved Y; default keeps `uyvy422_to_rgba_row`.
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      if center_sited {
        uyvy422_to_luma_row(packed, &mut packed_yuv_y_full[..w], w, use_simd);
        let (u_full, v_full) = packed_center_upsample_chroma(
          chroma_full,
          packed_yuv_u_half,
          packed_yuv_v_half,
          packed,
          w,
          0,
          2,
          use_simd,
        );
        yuv_444_to_rgba_row(
          &packed_yuv_y_full[..w],
          u_full,
          v_full,
          rgba_row,
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
        );
        return Ok(());
      }
      uyvy422_to_rgba_row(
        packed,
        rgba_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
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
    // Centered siting (#302) decodes via the 4:4:4 RGB kernel on the
    // de-interleaved Y + phase-0.5 upsampled chroma; the HSV / RGBA follow-ups
    // below derive off the produced RGB row either way. UYVY: Y at 1,3 / U at 0 /
    // V at 2.
    #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
    let centered = if center_sited {
      uyvy422_to_luma_row(packed, &mut packed_yuv_y_full[..w], w, use_simd);
      let (u_full, v_full) = packed_center_upsample_chroma(
        chroma_full,
        packed_yuv_u_half,
        packed_yuv_v_half,
        packed,
        w,
        0,
        2,
        use_simd,
      );
      yuv_444_to_rgb_row(
        &packed_yuv_y_full[..w],
        u_full,
        v_full,
        rgb_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
      true
    } else {
      false
    };
    #[cfg(not(all(feature = "yuv-packed", feature = "yuv-planar")))]
    let centered = false;
    if !centered {
      uyvy422_to_rgb_row(packed, rgb_row, w, row.matrix(), row.full_range(), use_simd);
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

// ---- Yvyu422 impl ------------------------------------------------------

impl<'a, R> MixedSinker<'a, Yvyu422, R> {
  /// Attaches a packed **8-bit** RGBA output buffer. Alpha is filled
  /// with constant `0xFF` (the source has no alpha channel).
  ///
  /// See [`MixedSinker::<Yuyv422>::with_rgba`] for the same rationale
  /// and constraints; YVYU differs only in chroma byte order (V
  /// before U).
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

  /// Attaches a **`u16`** luma output buffer. Y bytes are zero-extended
  /// to u16 (`out[x] = Y_byte as u16`). Length in u16 **elements**
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

impl<R> Yvyu422Sink for MixedSinker<'_, Yvyu422, R> {}

impl<R> PixelSink for MixedSinker<'_, Yvyu422, R> {
  type Input<'r> = Yvyu422Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::WidthAlignment(WidthAlignment::odd(
        self.width,
      )));
    }
    if let Some(stream) = self.rgb_stream.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.luma_stream.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.rgb_filter_stream.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.luma_filter_stream.as_mut() {
      stream.reset();
    }
    #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
    if let Some(native) = self.native_planar.as_mut() {
      native.reset();
    }
    #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
    {
      self.frozen_native_route = None;
      // RFC #238: clear the per-frame frozen 4:2:2 chroma siting so the next
      // frame may pick either phase; a mid-frame flip stays rejected.
      self.frozen_chroma_centered = None;
    }
    self.resample_outputs = None;
    Ok(())
  }

  fn process(&mut self, row: Yvyu422Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if w & 1 != 0 {
      return Err(MixedSinkerError::WidthAlignment(WidthAlignment::odd(w)));
    }

    let packed_expected =
      w.checked_mul(2)
        .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
          w, h, 2,
        )))?;
    if row.yvyu().len() != packed_expected {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Yvyu422Packed,
        idx,
        packed_expected,
        row.yvyu().len(),
      )));
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange(
        RowIndexOutOfRange::new(idx, self.height),
      ));
    }

    // Chroma siting (#302): drives the identity-plan horizontal chroma phase.
    // `Copy`, so read it before the field split-borrow below. Gated like its
    // only consumer (`chroma_422_center_sited_h` + the 4:4:4 kernels need
    // `yuv-planar`); a `yuv-packed`-only build keeps the default nearest decode.
    #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
    let chroma_location = self.chroma_location;

    let Self {
      rgb,
      rgba,
      luma,
      luma_u16,
      hsv,
      luma_scratch,
      rgb_scratch,
      plan,
      rgb_stream,
      luma_stream,
      rgb_filter_stream,
      luma_filter_stream,
      resample_outputs,
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      native,
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      native_planar,
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      packed_yuv_y_full,
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      packed_yuv_u_half,
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      packed_yuv_v_half,
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      frozen_native_route,
      // RFC #238: the 4:2:2 chroma phase frozen on the first output row.
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      frozen_chroma_centered,
      // Centered chroma-siting (#302) stages full-width U + V here.
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      chroma_full,
      ..
    } = self;
    let packed = row.yvyu();

    if let Some(plan) = plan.as_ref() {
      let matrix = row.matrix();
      let full_range = row.full_range();
      // RFC #238 — 4:2:2 horizontal chroma siting for Yvyu422, mirroring the
      // Yuyv422 / Uyvy422 twins above (YVYU: V at 1, U at 3). The centered group
      // samples chroma at `+0.25` chroma-sample; the co-sited / unspecified group
      // is phase 0 (byte-identical to the pre-siting resample). See the Yuyv422
      // impl for the full per-tier rationale.
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      let center_sited = chroma_422_center_sited_h(chroma_location);
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      let chroma_h_phase = if center_sited {
        YUV422P_CENTERED_H_PHASE
      } else {
        0.0
      };
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      let want_color = rgb.is_some() || rgba.is_some() || hsv.is_some();
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      let need_output =
        luma.is_some() || luma_u16.is_some() || rgb.is_some() || rgba.is_some() || hsv.is_some();
      // RFC #238: freeze the effective 4:2:2 chroma siting on the first
      // output-bearing row; a later row observing a different phase is rejected
      // HERE before any reconstruction or dispatch (see the Yuyv422 twin).
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      if need_output
        && let Some(frozen) = *frozen_chroma_centered
        && frozen != center_sited
      {
        return Err(MixedSinkerError::ChromaSitingChanged(
          ChromaSitingChanged::new(idx),
        ));
      }
      // Filter plan: native is area-only, so route straight to the filter
      // resampler (see the Yuyv422 impl above for the full rationale).
      if let SpanKind::Filter = plan.kind() {
        #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
        {
          plan.ensure_single_kernel_filter()?;
          if center_sited && want_color {
            let need_luma = luma.is_some() || luma_u16.is_some();
            let expected = if need_luma {
              luma_filter_stream.as_ref().map_or(0, |s| s.next_y())
            } else {
              rgb_filter_stream.as_ref().map_or(0, |s| s.next_y())
            };
            if let core::ops::ControlFlow::Break(()) = super::resample_preflight_check_only(
              resample_outputs,
              luma,
              luma_u16,
              rgb,
              rgba,
              &None,
              &None,
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
            reserve_420_chroma_full(chroma_full, w, h)?;
            reserve_packed_center_chroma(
              packed_yuv_y_full,
              packed_yuv_u_half,
              packed_yuv_v_half,
              w,
              h,
            )?;
            yvyu422_to_luma_row(packed, &mut packed_yuv_y_full[..w], w, use_simd);
            let (u_full, v_full) = packed_center_upsample_chroma(
              chroma_full,
              packed_yuv_u_half,
              packed_yuv_v_half,
              packed,
              w,
              3,
              1,
              use_simd,
            );
            let r = packed_yuv422_dual_filter_resample(
              luma_filter_stream,
              rgb_filter_stream,
              resample_outputs,
              rgb,
              rgba,
              luma,
              luma_u16,
              hsv,
              luma_scratch,
              rgb_scratch,
              w,
              plan,
              idx,
              use_simd,
              |scratch| yvyu422_to_luma_row(packed, scratch, w, use_simd),
              |scratch| {
                yuv_444_to_rgb_row(
                  &packed_yuv_y_full[..w],
                  u_full,
                  v_full,
                  scratch,
                  w,
                  matrix,
                  full_range,
                  use_simd,
                );
              },
            );
            if r.is_ok() && need_output && frozen_chroma_centered.is_none() {
              *frozen_chroma_centered = Some(center_sited);
            }
            return r;
          }
        }
        packed_yuv422_dual_filter_resample(
          luma_filter_stream,
          rgb_filter_stream,
          resample_outputs,
          rgb,
          rgba,
          luma,
          luma_u16,
          hsv,
          luma_scratch,
          rgb_scratch,
          w,
          plan,
          idx,
          use_simd,
          |scratch| yvyu422_to_luma_row(packed, scratch, w, use_simd),
          |scratch| yvyu422_to_rgb_row(packed, scratch, w, matrix, full_range, use_simd),
        )?;
        #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
        if need_output && frozen_chroma_centered.is_none() {
          *frozen_chroma_centered = Some(center_sited);
        }
        return Ok(());
      }
      // Area plan — YVYU: Y at 0,2 / V at 1 / U at 3.
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      {
        // The RFC #238 splice stage. A filter plan already returned above, so
        // `area_plan` is true and the selector reproduces the former `*native`
        // boolean bit-for-bit (`cfg!` is true wherever this block compiles).
        let take_native = matches!(
          select_insertion_point(
            AveragingDomain::Encoded,
            InsertionContext {
              native_eligible: cfg!(all(feature = "yuv-packed", feature = "yuv-planar")),
              with_native: *native,
              area_plan: true,
            },
          ),
          InsertionPoint::NativeCodes
        );
        if need_output
          && let Some(frozen) = *frozen_native_route
          && frozen != take_native
        {
          return Err(MixedSinkerError::NativeRouteChanged(
            NativeRouteChanged::new(idx),
          ));
        }
        if take_native {
          // RFC #238 point-of-use siting invalidation (see the Yuyv422 twin):
          // drop the cached join at the in-sequence fresh-frame first row when its
          // folded chroma plan was built for a different phase, transactionally.
          let stale_native = idx == 0
            && native_planar.as_ref().is_some_and(|join| {
              join.has_chroma() && join.chroma_centered() != center_sited && join.next_y() == 0
            });
          let prev_native = if stale_native {
            native_planar.take()
          } else {
            None
          };
          let native_result = packed_yuv422_process_native(
            plan,
            native_planar,
            packed_yuv_y_full,
            packed_yuv_u_half,
            packed_yuv_v_half,
            resample_outputs,
            rgb,
            rgba,
            luma,
            luma_u16,
            hsv,
            rgb_scratch,
            packed,
            0,
            2,
            3,
            1,
            chroma_h_phase,
            matrix,
            full_range,
            idx,
            w,
            h,
            use_simd,
          );
          if stale_native && native_result.is_err() {
            *native_planar = prev_native;
          }
          native_result?;
          if frozen_native_route.is_none() && need_output {
            *frozen_native_route = Some(true);
          }
          if frozen_chroma_centered.is_none() && need_output {
            *frozen_chroma_centered = Some(center_sited);
          }
          return Ok(());
        }
      }
      // Row-stage tail (see the Yuyv422 twin): centered colour reconstructs
      // full-width chroma after the resample preflight, then decodes 4:4:4;
      // a luma-only centered row stays on the co-sited arm.
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      if center_sited && want_color {
        let need_luma = luma.is_some() || luma_u16.is_some();
        let expected = if need_luma {
          luma_stream.as_ref().map_or(0, |s| s.next_y())
        } else {
          rgb_stream.as_ref().map_or(0, |s| s.next_y())
        };
        if let core::ops::ControlFlow::Break(()) = super::resample_preflight_check_only(
          resample_outputs,
          luma,
          luma_u16,
          rgb,
          rgba,
          &None,
          &None,
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
        reserve_420_chroma_full(chroma_full, w, h)?;
        reserve_packed_center_chroma(
          packed_yuv_y_full,
          packed_yuv_u_half,
          packed_yuv_v_half,
          w,
          h,
        )?;
        yvyu422_to_luma_row(packed, &mut packed_yuv_y_full[..w], w, use_simd);
        let (u_full, v_full) = packed_center_upsample_chroma(
          chroma_full,
          packed_yuv_u_half,
          packed_yuv_v_half,
          packed,
          w,
          3,
          1,
          use_simd,
        );
        packed_yuv422_dual_resample(
          luma_stream,
          rgb_stream,
          resample_outputs,
          rgb,
          rgba,
          luma,
          luma_u16,
          hsv,
          luma_scratch,
          rgb_scratch,
          w,
          plan,
          idx,
          use_simd,
          |scratch| yvyu422_to_luma_row(packed, scratch, w, use_simd),
          |scratch| {
            yuv_444_to_rgb_row(
              &packed_yuv_y_full[..w],
              u_full,
              v_full,
              scratch,
              w,
              matrix,
              full_range,
              use_simd,
            );
          },
        )?;
        if frozen_native_route.is_none() && need_output {
          *frozen_native_route = Some(false);
        }
        if frozen_chroma_centered.is_none() && need_output {
          *frozen_chroma_centered = Some(center_sited);
        }
        return Ok(());
      }
      packed_yuv422_dual_resample(
        luma_stream,
        rgb_stream,
        resample_outputs,
        rgb,
        rgba,
        luma,
        luma_u16,
        hsv,
        luma_scratch,
        rgb_scratch,
        w,
        plan,
        idx,
        use_simd,
        |scratch| yvyu422_to_luma_row(packed, scratch, w, use_simd),
        |scratch| yvyu422_to_rgb_row(packed, scratch, w, matrix, full_range, use_simd),
      )?;
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      {
        if frozen_native_route.is_none() && need_output {
          *frozen_native_route = Some(false);
        }
        if frozen_chroma_centered.is_none() && need_output {
          *frozen_chroma_centered = Some(center_sited);
        }
      }
      return Ok(());
    }

    // Strategy A output mode resolution — resolved BEFORE any output write so
    // the atomicity preflight below runs ahead of the luma writes.
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();

    // No-output guard (#302, cf. the planar / semi-planar siblings): a `process`
    // call with NO output attached must run NOTHING — no `idx * w` offset
    // arithmetic (a no-output call never ran an attach-time `w x h` validation,
    // so on a 32-bit target an absurd geometry would overflow that offset), no
    // allocation, no state mutation. Returning HERE — before the offsets AND the
    // centered chroma preflight — keeps a no-output row panic-free and
    // allocation-free.
    let need_output = want_rgb || want_rgba || want_hsv || luma.is_some() || luma_u16.is_some();
    if !need_output {
      return Ok(());
    }

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // Chroma siting (#302): the centered horizontal sitings reconstruct chroma at
    // the phase-0.5 position then decode via the 4:4:4 kernels on the
    // de-interleaved Y; the default / co-sited path keeps the byte-identical
    // nearest-neighbor decode. 4:2:2 is horizontally subsampled only — there is no
    // vertical blend or chroma lookback (cf. the Yuv420p `Bottom` path).
    #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
    let center_sited = chroma_422_center_sited_h(chroma_location);

    // Per-frame chroma-siting freeze (RFC #238, mirroring the resample-path guard
    // above): the first output-bearing row pins the phase; a later row whose siting
    // flipped would decode a mixture of centered and co-sited chroma into ONE
    // frame, so reject it here BEFORE any scratch reserve or output write.
    // `begin_frame` clears the freeze so the next frame may pick either phase.
    #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
    if need_output
      && let Some(frozen) = *frozen_chroma_centered
      && frozen != center_sited
    {
      return Err(MixedSinkerError::ChromaSitingChanged(
        ChromaSitingChanged::new(idx),
      ));
    }

    // Atomicity preflight (#302 / #308, cf. the crate's #180 resample fix and the
    // planar / semi-planar siblings): reserve EVERY fallible row scratch this row
    // needs BEFORE any output row (luma / luma_u16 included) is written, so an
    // allocator refusal returns a typed `AllocationFailed` leaving the output
    // frame untouched rather than partially mutated. Two groups can grow:
    //  1. the centered-siting full-width chroma (`chroma_full`) plus the packed
    //     de-interleave scratch (full-width Y + half-width U / V); and
    //  2. the RGB row buffer, reserved exactly when a colour decode needs an RGB
    //     row but no caller RGB buffer is borrowable — `want_hsv && want_rgba &&
    //     !want_rgb` (`rgb_row_buf_or_scratch`'s own scratch arm; an attached RGB
    //     buffer is borrowed instead and never allocates).
    // The later de-interleave / decode calls reuse the already-sized buffers, so
    // the default path is byte-identical; only the failure-path ordering changes.
    #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
    if center_sited && (want_rgb || want_rgba || want_hsv) {
      reserve_420_chroma_full(chroma_full, w, h)?;
      reserve_packed_center_chroma(
        packed_yuv_y_full,
        packed_yuv_u_half,
        packed_yuv_v_half,
        w,
        h,
      )?;
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

    // Freeze the phase on the first output-bearing row — AFTER the fallible scratch
    // reserves above have succeeded, so an `AllocationFailed` row stays retryable
    // (frozen stays unset); later rows are checked against it up top.
    #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
    if need_output && frozen_chroma_centered.is_none() {
      *frozen_chroma_centered = Some(center_sited);
    }

    if let Some(luma) = luma.as_deref_mut() {
      yvyu422_to_luma_row(
        packed,
        &mut luma[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }
    // Luma u16 — zero-extend Y bytes to u16.
    if let Some(buf) = luma_u16.as_deref_mut() {
      yvyu422_to_luma_u16_row(
        packed,
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    // HSV-without-RGB-or-RGBA goes through the direct `yvyu422_to_hsv_row`
    // kernel (no source-width RGB scratch). When RGB or RGBA is *also*
    // attached the RGB kernel runs anyway, so HSV derives off that buffer
    // for free (the cheap path) and `need_rgb_kernel` keeps it alive.
    let want_hsv_direct = want_hsv && !want_rgb && !want_rgba;
    let need_rgb_kernel = want_rgb || (want_hsv && want_rgba);

    if want_hsv_direct {
      let hsv = hsv.as_mut().expect("want_hsv_direct implies hsv attached");
      let (h_out, s_out, v_out) = hsv.hsv();
      // Centered siting (#302): de-interleave Y + phase-0.5 upsample chroma to
      // full width, then run the 4:4:4 HSV kernel (scratch reserved above).
      // YVYU: Y at 0,2 / V at 1 / U at 3.
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      if center_sited {
        yvyu422_to_luma_row(packed, &mut packed_yuv_y_full[..w], w, use_simd);
        let (u_full, v_full) = packed_center_upsample_chroma(
          chroma_full,
          packed_yuv_u_half,
          packed_yuv_v_half,
          packed,
          w,
          3,
          1,
          use_simd,
        );
        yuv_444_to_hsv_row(
          &packed_yuv_y_full[..w],
          u_full,
          v_full,
          &mut h_out[one_plane_start..one_plane_end],
          &mut s_out[one_plane_start..one_plane_end],
          &mut v_out[one_plane_start..one_plane_end],
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
        );
        return Ok(());
      }
      yvyu422_to_hsv_row(
        packed,
        &mut h_out[one_plane_start..one_plane_end],
        &mut s_out[one_plane_start..one_plane_end],
        &mut v_out[one_plane_start..one_plane_end],
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
      return Ok(());
    }

    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      // Centered siting (#302): full-width phase-0.5 chroma + the 4:4:4 RGBA
      // kernel on the de-interleaved Y; default keeps `yvyu422_to_rgba_row`.
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      if center_sited {
        yvyu422_to_luma_row(packed, &mut packed_yuv_y_full[..w], w, use_simd);
        let (u_full, v_full) = packed_center_upsample_chroma(
          chroma_full,
          packed_yuv_u_half,
          packed_yuv_v_half,
          packed,
          w,
          3,
          1,
          use_simd,
        );
        yuv_444_to_rgba_row(
          &packed_yuv_y_full[..w],
          u_full,
          v_full,
          rgba_row,
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
        );
        return Ok(());
      }
      yvyu422_to_rgba_row(
        packed,
        rgba_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
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
    // Centered siting (#302) decodes via the 4:4:4 RGB kernel on the
    // de-interleaved Y + phase-0.5 upsampled chroma; the HSV / RGBA follow-ups
    // below derive off the produced RGB row either way. YVYU: Y at 0,2 / V at 1 /
    // U at 3.
    #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
    let centered = if center_sited {
      yvyu422_to_luma_row(packed, &mut packed_yuv_y_full[..w], w, use_simd);
      let (u_full, v_full) = packed_center_upsample_chroma(
        chroma_full,
        packed_yuv_u_half,
        packed_yuv_v_half,
        packed,
        w,
        3,
        1,
        use_simd,
      );
      yuv_444_to_rgb_row(
        &packed_yuv_y_full[..w],
        u_full,
        v_full,
        rgb_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
      true
    } else {
      false
    };
    #[cfg(not(all(feature = "yuv-packed", feature = "yuv-planar")))]
    let centered = false;
    if !centered {
      yvyu422_to_rgb_row(packed, rgb_row, w, row.matrix(), row.full_range(), use_simd);
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
