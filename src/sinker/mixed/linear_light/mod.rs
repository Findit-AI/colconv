//! RFC #238 Phase 2 — the [`AveragingDomain::Linear`] linear-light tail for
//! the planar 8-bit YUV family (`Yuv420p` / `Yuv422p` / `Yuv444p` /
//! `Yuv440p`).
//!
//! The encoded average bins the YUV codes (or the gamma-encoded RGB) and is
//! the default; the linear-light average instead mixes light. The convert
//! `YUV→RGB` is affine (a Q15 matrix + offset + clamp, no transfer), so the
//! two land at materially different RGB — the quality trade the Linear
//! domain offers. The pipeline is, per the RFC:
//!
//! ```text
//! YUV row  →[convert]→  encoded RGB  →[EOTF]→  linear RGB
//!          →[area bin (AreaStream<f32>)]→  linear RGB (out res)
//!          →[OETF]→  encoded RGB  →  RGB / RGBA
//! ```
//!
//! The decode is the format's own existing `YUV→RGB` row kernel (chroma
//! upsampled in-register), supplied by the caller as a closure so all four
//! formats share the linearise → bin → re-encode tail. The
//! [`TransferFunction`] is the sink's caller override, else the
//! per-[`KernelMatrix`] default.
//!
//! # What "linear" means here: display-referred vs scene-referred
//!
//! Which `YUV→RGB` decode fills the buffer the EOTF lifts is selected by the
//! sink's [`LinearMode`] (RFC #238 #244) — the only mode-dependent step:
//!
//! - [`LinearMode::DisplayReferred`] (the default) decodes through the
//!   production `yuv_*_to_rgb_row` family, whose Q15 convert **clamps and
//!   quantizes** the result to 8-bit `[0, 255]` BEFORE this tail sees it. So
//!   the average is taken over the *display-referred* converted 8-bit RGB
//!   lifted to linear light — a gamma-correct resize of the in-gamut RGB.
//!   Out-of-gamut YUV excursions (super-black / super-white, or chroma that
//!   drives a channel past the cube) are clipped at the convert, so values
//!   that would average back into gamut are lost. The clamped 8-bit RGB
//!   mapped to `[0, 1]` is never negative, so the odd-symmetric negative-side
//!   extrapolation of [`TransferFunction::eotf`] / [`oetf`] is dormant on this
//!   path.
//! - [`LinearMode::SceneReferred`] decodes the **same affine matrix** — the
//!   same `Coefficients` and `range_params_n::<8, 8>` the Q15 kernel uses —
//!   in unclamped real-valued `f32` (`yuv_*_to_rgb_f32_unclamped_row`),
//!   differing from the display decode ONLY by the absent intermediate Q15
//!   rounding and the absent final clamp+round. The out-of-gamut excursions
//!   are preserved (a channel may go `< 0` or `> 1`), lifted to linear light
//!   by the SAME EOTF — whose odd-symmetric extrapolation, retained in
//!   [`TransferFunction`] as public API for exactly this consumer, now
//!   activates — averaged in linear light, and clamped **only** at the
//!   re-encoded output. The two modes coincide (modulo `f32` rounding) on
//!   content that stays in gamut through the decode.
//!
//! [`LinearMode`]: crate::resample::LinearMode
//! [`LinearMode::DisplayReferred`]: crate::resample::LinearMode::DisplayReferred
//! [`LinearMode::SceneReferred`]: crate::resample::LinearMode::SceneReferred
//!
//! # Frame-buffered
//!
//! Like the validated PoC, the tail accumulates the linearised RGB for the
//! whole frame and bins once the last source row lands — the area bin is 2D
//! and the chroma upsample is per-row, so a frame buffer keeps the splice
//! self-contained without threading the row-stream bookkeeping the encoded
//! tiers carry. The frame buffer lives on the sink and resets per
//! `begin_frame`. Only `Yuv420p` / `Yuv422p` / `Yuv444p` / `Yuv440p` are
//! wired; every other format and the encoded domain are untouched.
//!
//! [`AveragingDomain::Linear`]: crate::resample::AveragingDomain::Linear
//! [`KernelMatrix`]: crate::KernelMatrix
//! [`TransferFunction::eotf`]: crate::resample::TransferFunction::eotf
//! [`oetf`]: crate::resample::TransferFunction::oetf

use super::{
  FrozenOutputs, GeometryOverflow, HsvFrameMut, LinearModeChanged, MixedSinkerError,
  ResampleOutputsChanged, TransferFunctionChanged,
};
use crate::{
  resample::{AreaStream, LinearMode, ResamplePlan, TransferFunction, try_zeroed},
  row::{expand_rgb_to_rgba_row, rgb_row_bytes, rgb_to_hsv_row, y_plane_to_luma_u16_row},
};

#[cfg(all(test, feature = "std"))]
mod tests;

#[cfg(all(test, feature = "std", feature = "yuv-planar", feature = "rgb"))]
std::thread_local! {
  static FORCE_LINEAR_TAIL_ALLOC_FAILURE: core::cell::Cell<bool> =
    const { core::cell::Cell::new(false) };
  static FORCE_LINEAR_SCRATCH_FAILURE: core::cell::Cell<bool> =
    const { core::cell::Cell::new(false) };
}

/// Arms a one-shot failpoint that fires on the next final-row tail allocation
/// of the linear-light resample (the f32 bin stream / `binned` accumulator /
/// luma stream / re-encode row — the first fallible step of the bin tail). The
/// flag is taken (cleared) when the tail reaches it, so it fires exactly once.
/// Test-only; mirrors `arm_planar_native_chroma_failure`. Used to prove the
/// tail is transactional w.r.t. the persistent frame accumulator: a failed
/// final row must not advance `next_y` or consume the buffered frame.
#[cfg(all(test, feature = "std", feature = "yuv-planar", feature = "rgb"))]
pub(crate) fn arm_linear_tail_alloc_failure() {
  FORCE_LINEAR_TAIL_ALLOC_FAILURE.with(|f| f.set(true));
}

/// Takes the test failpoint: `true` exactly once after [`arm_linear_tail_alloc_failure`].
/// Always `false` in non-test builds (the failpoint compiles out).
#[cfg_attr(not(tarpaulin), inline(always))]
fn take_linear_tail_alloc_failure() -> bool {
  #[cfg(all(test, feature = "std", feature = "yuv-planar", feature = "rgb"))]
  {
    FORCE_LINEAR_TAIL_ALLOC_FAILURE.with(|f| f.take())
  }
  #[cfg(not(all(test, feature = "std", feature = "yuv-planar", feature = "rgb")))]
  {
    false
  }
}

/// Arms a one-shot failpoint that fires on the next per-row decode-scratch
/// reserve of the linear-light resample (the very first fallible allocation a
/// row reaches AFTER the frame build, but — crucially — BEFORE the `*frame`
/// commit). One-shot: taken (cleared) when reached. Test-only; mirrors
/// [`arm_linear_tail_alloc_failure`]. Used to prove the FIRST-row frame commit
/// is transactional: a scratch refusal on the row that *would* create the frame
/// must leave `*frame` `None` (no frozen transfer), so a corrected retry of the
/// SAME row is not mis-rejected as [`TransferFunctionChanged`].
#[cfg(all(test, feature = "std", feature = "yuv-planar", feature = "rgb"))]
pub(crate) fn arm_linear_scratch_failure() {
  FORCE_LINEAR_SCRATCH_FAILURE.with(|f| f.set(true));
}

/// Takes the scratch failpoint: `true` exactly once after [`arm_linear_scratch_failure`].
/// Always `false` in non-test builds (the failpoint compiles out).
#[cfg_attr(not(tarpaulin), inline(always))]
fn take_linear_scratch_failure() -> bool {
  #[cfg(all(test, feature = "std", feature = "yuv-planar", feature = "rgb"))]
  {
    FORCE_LINEAR_SCRATCH_FAILURE.with(|f| f.take())
  }
  #[cfg(not(all(test, feature = "std", feature = "yuv-planar", feature = "rgb")))]
  {
    false
  }
}

/// The output RGB element count `out_w * out_h * 3` for the final-row `binned`
/// frame, computed with checked arithmetic so a size overflow surfaces as a
/// typed [`MixedSinkerError::GeometryOverflow`] instead of wrapping into an
/// undersized allocation.
///
/// The bin tail allocates a full RGB `binned` frame even for a one-channel-only
/// Linear run (luma / luma_u16 / HSV), whose plan validates only the
/// one-channel output size — so on a 32-bit target `out_w * out_h * 3` is not
/// otherwise bounded. Mirrors the size guard in [`LinearLightFrame::new`].
fn linear_tail_rgb_len(out_w: usize, out_h: usize) -> Result<usize, MixedSinkerError> {
  out_w
    .checked_mul(out_h)
    .and_then(|n| n.checked_mul(3))
    .ok_or_else(|| MixedSinkerError::GeometryOverflow(GeometryOverflow::new(out_w, out_h, 3)))
}

/// Per-frame linear-light accumulator: the source frame's RGB decoded and
/// linearised (and the Y plane buffered for the domain-independent luma
/// outputs), held until the last row arrives and the area bin runs.
///
/// `Vec` resolves to `alloc::vec::Vec` under `no_std` (the crate aliases
/// `alloc` to `std`), so this is `no_std + alloc` clean.
/// The final-row bin tail — the `f32` RGB bin stream, the `binned` output
/// accumulator, the optional luma `u8` bin stream, and the per-output re-encode
/// row. All four are dimension-only allocations (independent of any row's
/// data), so they can be pre-built ahead of the final row and consumed there.
/// Pre-building lets the RFC #238 Top odd-height two-row FINAL flush reserve
/// them BEFORE either of its feeds, keeping the flush retry-atomic (a refusal
/// returns before any feed advances the frame cursor).
type LinearTail = (
  AreaStream<f32>,
  std::vec::Vec<f32>,
  Option<AreaStream<u8>>,
  std::vec::Vec<u8>,
);

#[derive(Debug)]
pub(super) struct LinearLightFrame {
  /// `src_w * src_h * 3` interleaved linear-light RGB (`f32`), filled row
  /// by row as source rows arrive.
  linear_rgb: std::vec::Vec<f32>,
  /// `src_w * src_h` Y-plane bytes, buffered only when a luma output is
  /// attached (empty otherwise). Luma is domain-independent, so binning the
  /// Y plane here matches the encoded tiers byte-for-byte.
  y_plane: std::vec::Vec<u8>,
  /// The next source row index this frame expects — enforces the same
  /// strict-sequencing contract the encoded streams keep.
  next_y: usize,
  /// Source width (luma grid).
  src_w: usize,
  /// Source height (luma grid).
  src_h: usize,
  /// The transfer function resolved on the first output-bearing row, frozen
  /// for the life of the frame. Every buffered row is linearised under this
  /// curve, so a later row resolving a different transfer is rejected (a
  /// mid-frame flip would bin rows linearised under inconsistent curves).
  frozen_transfer: TransferFunction,
  /// The [`LinearMode`] resolved on the first output-bearing row, frozen for
  /// the life of the frame alongside `frozen_transfer`. The mode selects which
  /// `YUV→RGB` decode (display-referred Q15-clamped vs scene-referred
  /// unclamped `f32`) fills the buffer the EOTF lifts, so a later row resolving
  /// a different mode is rejected (a mid-frame flip would bin display- and
  /// scene-decoded rows in the same frame).
  frozen_linear_mode: LinearMode,
  /// RFC #238 Top: an optionally pre-built final-row bin tail. `None` on the
  /// ordinary path — the final row builds the tail inline. The Top odd-height
  /// two-row FINAL flush pre-reserves it here via
  /// [`linear_light_reserve_final_tail`] BEFORE either of its two feeds, so the
  /// final feed consumes it instead of allocating and the flush stays
  /// retry-atomic. Cleared (taken) when the final row consumes it.
  reserved_tail: Option<LinearTail>,
}

impl LinearLightFrame {
  /// Allocates the per-frame linear-RGB buffer (and, when `want_luma`, the
  /// Y-plane buffer) for an `src_w x src_h` frame. Follows the planner's
  /// recoverable-allocation contract: a caller-proportional buffer refusal
  /// surfaces as an error, not an abort.
  ///
  /// The two failure modes are *distinct typed errors*: a size **overflow**
  /// (the `src_w * src_h * 3` element count cannot be represented) is a
  /// [`MixedSinkerError::GeometryOverflow`], whereas the allocator **refusing**
  /// a representable size is a [`ResampleError::AllocationFailed`] carrying the
  /// [`PlanGeometry`] — the same split the row-stage / native frame tails use,
  /// so an out-of-memory condition is never mislabelled as a geometry overflow.
  /// `out_w` / `out_h` are threaded in only to populate the `AllocationFailed`
  /// geometry payload (the buffer itself is sized by the source grid).
  ///
  /// [`ResampleError::AllocationFailed`]: crate::resample::ResampleError::AllocationFailed
  /// [`PlanGeometry`]: crate::resample::PlanGeometry
  fn new(
    src_w: usize,
    src_h: usize,
    out_w: usize,
    out_h: usize,
    want_luma: bool,
    frozen_transfer: TransferFunction,
    frozen_linear_mode: LinearMode,
  ) -> Result<Self, MixedSinkerError> {
    let overflow = || MixedSinkerError::GeometryOverflow(GeometryOverflow::new(src_w, src_h, 3));
    let alloc_failed = || {
      MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
        crate::resample::PlanGeometry::new(src_w, src_h, out_w, out_h),
      ))
    };
    let luma = src_w.checked_mul(src_h).ok_or_else(overflow)?;
    let n = luma.checked_mul(3).ok_or_else(overflow)?;
    let y_plane = if want_luma {
      try_zeroed(luma).map_err(|_| alloc_failed())?
    } else {
      std::vec::Vec::new()
    };
    Ok(Self {
      linear_rgb: try_zeroed(n).map_err(|_| alloc_failed())?,
      y_plane,
      next_y: 0,
      src_w,
      src_h,
      frozen_transfer,
      frozen_linear_mode,
      reserved_tail: None,
    })
  }

  /// The next source row this frame expects (0 at a fresh frame). Read by the
  /// `Yuv420p` Top forward-delay dispatch to derive the next expected SOURCE row
  /// (`next_y + buffered-held`), for the out-of-sequence walker guard.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(super) const fn next_y(&self) -> usize {
    self.next_y
  }
}

/// Pre-builds the final-row bin tail into an already-committed
/// [`LinearLightFrame`] so the RFC #238 Top odd-height two-row FINAL flush can
/// reserve it BEFORE either of its feeds. The tail is the ONLY fallible
/// allocation the final row runs, and in the two-row flush the final feed (the
/// second) would otherwise allocate it AFTER the first feed already advanced the
/// frame cursor — un-retryable. Hoisting it here (via the SAME failpoint and the
/// SAME dimension-only allocations [`linear_light_resample`] uses) makes both
/// feeds infallible, so a refusal returns before any feed and the same row
/// retries cleanly (#180). Idempotent (a second call is a no-op once reserved);
/// a `None` frame (the first row, never a two-row flush) reserves nothing.
pub(super) fn linear_light_reserve_final_tail(
  frame: &mut Option<LinearLightFrame>,
  plan: &ResamplePlan,
  w: usize,
  h: usize,
  want_luma: bool,
) -> Result<(), MixedSinkerError> {
  let Some(buf) = frame.as_mut() else {
    return Ok(());
  };
  if buf.reserved_tail.is_some() {
    return Ok(());
  }
  let ow = plan.out_w();
  let oh = plan.out_h();
  if take_linear_tail_alloc_failure() {
    return Err(MixedSinkerError::Resample(
      crate::resample::ResampleError::AllocationFailed(crate::resample::PlanGeometry::new(
        w, h, ow, oh,
      )),
    ));
  }
  let rgb_len = linear_tail_rgb_len(ow, oh)?;
  let alloc_failed = || {
    MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
      crate::resample::PlanGeometry::new(w, h, ow, oh),
    ))
  };
  let stream = AreaStream::<f32>::new(plan.h(), plan.v(), w, h, 3)?;
  let binned = try_zeroed::<f32>(rgb_len).map_err(|_| alloc_failed())?;
  let y_stream = if want_luma {
    Some(AreaStream::<u8>::new(plan.h(), plan.v(), w, h, 1)?)
  } else {
    None
  };
  let enc_out = try_zeroed::<u8>(rgb_row_bytes(ow)).map_err(|_| alloc_failed())?;
  buf.reserved_tail = Some((stream, binned, y_stream, enc_out));
  Ok(())
}

/// Check-only Linear preflight: the exact rejects [`linear_light_resample`]
/// runs (unsupported filter plan, mid-frame output-set / transfer / mode
/// change, out-of-sequence row) with **no** state mutation and **no** commit.
///
/// RFC #238's centered-Linear path must reserve and reconstruct full-width
/// chroma before it can decode, so it runs this check FIRST and only reserves
/// once every rejection has passed — a rejected row therefore allocates
/// nothing (#180). [`linear_light_resample`] re-runs the full preflight and
/// owns the transactional output/transfer/mode commit, so calling this ahead
/// of it is a pure, idempotent gate.
#[allow(clippy::too_many_arguments)]
pub(super) fn linear_light_preflight(
  frame: &Option<LinearLightFrame>,
  resample_outputs: &Option<FrozenOutputs>,
  luma: &Option<&mut [u8]>,
  luma_u16: &Option<&mut [u16]>,
  rgb: &Option<&mut [u8]>,
  rgba: &Option<&mut [u8]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  tf: TransferFunction,
  mode: LinearMode,
  plan: &ResamplePlan,
  idx: usize,
) -> Result<(), MixedSinkerError> {
  if plan.kind().is_filter() {
    return Err(plan.unsupported_filter().into());
  }
  let snapshot = FrozenOutputs::snapshot(
    luma.as_deref(),
    luma_u16.as_deref(),
    rgb.as_deref(),
    rgba.as_deref(),
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    hsv.as_mut().map(|f| {
      let (h, s, v) = f.hsv();
      (&h[..], &s[..], &v[..])
    }),
    None,
  );
  if let Some(frozen) = resample_outputs
    && *frozen != snapshot
  {
    return Err(MixedSinkerError::ResampleOutputsChanged(
      ResampleOutputsChanged::new(idx),
    ));
  }
  let expected = frame.as_ref().map_or(0, |b| b.next_y);
  if idx != expected {
    return Err(MixedSinkerError::Resample(
      crate::resample::ResampleError::OutOfSequenceRow(crate::resample::OutOfSequenceRow::new(
        expected, idx,
      )),
    ));
  }
  if let Some(b) = frame.as_ref()
    && b.frozen_transfer != tf
  {
    return Err(MixedSinkerError::TransferFunctionChanged(
      TransferFunctionChanged::new(idx),
    ));
  }
  if let Some(b) = frame.as_ref()
    && b.frozen_linear_mode != mode
  {
    return Err(MixedSinkerError::LinearModeChanged(LinearModeChanged::new(
      idx,
    )));
  }
  Ok(())
}

/// Runs the [`AveragingDomain::Linear`](crate::resample::AveragingDomain::Linear)
/// linear-light resample for one source row of a planar 8-bit YUV frame.
///
/// The decode that fills the per-row encoded RGB depends on `mode` (RFC #238
/// #244), the only mode-dependent step — the rest of the pipeline (EOTF →
/// area bin → OETF → clamp) is shared:
///
/// - [`LinearMode::DisplayReferred`] (the default): `decode_rgb_row(idx, dst)`
///   converts source row `idx` to a source-width **encoded 8-bit** RGB row
///   (`3 * w` bytes) via the format's production `yuv_*_to_rgb_row` kernel
///   (which clamps + quantizes to `[0, 255]`), and each byte is normalized
///   `byte / 255.0` before the EOTF. **Byte-identical to RFC #238 Phase 2.**
/// - [`LinearMode::SceneReferred`]: `decode_unclamped_f32(idx, dst)` decodes
///   the SAME affine matrix in unclamped `f32` into `scene_scratch` (`3 * w`
///   `f32`, a `[0, 1]` scale that MAY leave `[0, 1]` where the source is out
///   of gamut), and that value feeds the EOTF directly — preserving the
///   out-of-gamut excursions the clamped decode discards.
///
/// Either decode is then lifted to linear light through `tf`'s EOTF (whose
/// odd-symmetric extrapolation handles out-of-`[0, 1]` scene-referred inputs)
/// into the frame buffer; on the final row it area-bins the linear RGB
/// through [`AreaStream<f32>`], re-encodes through `tf`'s OETF, and **clamps**
/// the result to the output range, writing the RGB / RGBA / luma outputs at
/// output geometry.
///
/// Full state atomicity and strict row sequencing mirror the encoded
/// row-stage tail. Every fallible step — the filter-plan reject, the
/// output-set compare, the out-of-sequence-row check, AND every fallible
/// allocation (the per-row decode-scratch reserve and the final row's bin
/// streams / accumulator / re-encode row) — runs BEFORE any **persistent
/// accumulator mutation** and before the first caller-output byte. The
/// persistent mutations are the per-row `next_y` advance and frame-buffer
/// writes AND, on the **first** output-bearing row, the commit of the frame
/// itself AND the freeze of the output set: the new [`LinearLightFrame`] is
/// built into a local and only assigned to `*frame` after the rest of the
/// fallible phase succeeds, so `*frame` is `Some` iff at least one output-bearing
/// row was fully accepted. The output-set freeze is split COMPARE-from-COMMIT
/// the same way — the snapshot is *compared* before the fallible phase (an
/// already-frozen mismatch rejects with no mutation) but, when `*resample_outputs`
/// is still `None`, *committed* only at the `*frame` commit point — so a first-row
/// failure leaves `*resample_outputs` `None` alongside `*frame`, fully retryable
/// with a changed output attachment. A failure on the row that *would* create the
/// frame therefore leaves `*frame` AND `*resample_outputs` `None` (no
/// `frozen_transfer`, no `frozen_linear_mode`, no frozen output set, no
/// `next_y == 0`), and a failure on a
/// later row leaves the already-committed frame with its `next_y` unadvanced and
/// this row's slot unwritten. Either way the same sink retries the row cleanly (no
/// `begin_frame`) once the pressure clears — with a *corrected* transfer / matrix
/// / output attachment on the first-row case — surfacing the typed error rather
/// than a poisoned accumulator (a stale `frozen_transfer` mis-rejecting the retry
/// as [`TransferFunctionChanged`], or a stale frozen output set mis-rejecting it
/// as [`ResampleOutputsChanged`]) or a downstream `AllocationFailed`. A no-output
/// call is a route-invisible no-op.
///
/// The Linear domain is **area-only**: a [`SpanKind::Filter`] plan is
/// rejected with [`ResampleError::UnsupportedFilter`] at preflight (the
/// dispatch sites also reject before their filter path, so the encoded
/// filter result is never silently returned for a Linear sink). The bin is
/// the integer-area [`AreaStream<f32>`]; there is no signed-coefficient
/// filter twin here.
///
/// [`SpanKind::Filter`]: crate::resample::SpanKind::Filter
/// [`ResampleError::UnsupportedFilter`]: crate::resample::ResampleError::UnsupportedFilter
#[allow(clippy::too_many_arguments)]
pub(super) fn linear_light_resample(
  frame: &mut Option<LinearLightFrame>,
  resample_outputs: &mut Option<FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  rgb_scratch: &mut std::vec::Vec<u8>,
  scene_scratch: &mut std::vec::Vec<f32>,
  tf: TransferFunction,
  mode: LinearMode,
  plan: &ResamplePlan,
  y_row: &[u8],
  idx: usize,
  w: usize,
  h: usize,
  use_simd: bool,
  mut decode_rgb_row: impl FnMut(usize, &mut [u8]),
  mut decode_unclamped_f32: impl FnMut(usize, &mut [f32]),
) -> Result<(), MixedSinkerError> {
  // Whether this call carries any output — the same set the encoded tiers'
  // preflight uses. A no-output call consumes no frame state and must not
  // freeze the outputs or sequence.
  let need_output =
    luma.is_some() || luma_u16.is_some() || rgb.is_some() || rgba.is_some() || hsv.is_some();
  if !need_output {
    return Ok(());
  }

  // The Linear domain bins in the integer-area engine only; a filter plan
  // has empty area spans and no signed-coefficient twin here. Reject it
  // BEFORE the frozen check / sequence check / any allocation, so a Linear
  // sink handed a filter plan fails with the typed `UnsupportedFilter`
  // rather than silently mis-binning (the dispatch sites reject ahead of
  // their filter path too — this is the in-tail backstop).
  if plan.kind().is_filter() {
    return Err(plan.unsupported_filter().into());
  }

  // Mid-frame output-set change → typed rejection (matches the row-stage
  // tail). Split COMPARE from COMMIT so the output freeze is transactional with
  // the first-row fallible phase (unlike the shared `frozen_outputs_check`,
  // which commits the snapshot the instant it is `None` — used unchanged by the
  // encoded tails, whose freeze runs ahead of an infallible row). Here the
  // snapshot is an owned, `Copy` `FrozenOutputs` (no borrow of the output
  // slices, which are only WRITTEN in the commit/bin phase below), so it is held
  // across the fallible phase and committed at the SAME point as `*frame`:
  //  - `*resample_outputs` already `Some` and differs → reject now, no mutation
  //    (a genuine mid-frame output change, first or later row);
  //  - already `Some` and equal → proceed, nothing to commit (a later row,
  //    already frozen on the accepted first row);
  //  - still `None` → do NOT commit yet; hold the snapshot in `pending_outputs`
  //    and assign it only after the fallible phase succeeds, so a first-row
  //    failure leaves `*resample_outputs` `None` (alongside `*frame`) and the
  //    same sink retries the row with a changed output attachment.
  let snapshot = FrozenOutputs::snapshot(
    luma.as_deref(),
    luma_u16.as_deref(),
    rgb.as_deref(),
    rgba.as_deref(),
    None, // rgb_u16
    None, // rgba_u16
    None, // rgb_f32
    None, // rgba_f32
    None, // xyz_f32
    None, // rgb_f16
    None, // rgba_f16
    hsv.as_mut().map(|f| {
      let (h, s, v) = f.hsv();
      (&h[..], &s[..], &v[..])
    }),
    None, // luma_f32
  );
  let pending_outputs = match resample_outputs {
    Some(frozen) if *frozen != snapshot => {
      return Err(MixedSinkerError::ResampleOutputsChanged(
        ResampleOutputsChanged::new(idx),
      ));
    }
    Some(_) => None,
    None => Some(snapshot),
  };

  // Strict sequencing: reject an out-of-sequence row BEFORE the frame
  // buffer is allocated, so a rejected row allocates nothing and the
  // accumulator stays resumable on the expected row. The expected index is
  // the existing accumulator's `next_y` (0 before the first row's buffer
  // exists), mirroring the encoded native preflight's pre-build check.
  let expected = frame.as_ref().map_or(0, |b| b.next_y);
  if idx != expected {
    return Err(MixedSinkerError::Resample(
      crate::resample::ResampleError::OutOfSequenceRow(crate::resample::OutOfSequenceRow::new(
        expected, idx,
      )),
    ));
  }

  // Mid-frame transfer-function change → typed rejection BEFORE any state
  // mutation, mirroring the `frozen_native_route` / `frozen_outputs` freeze.
  // The transfer is captured when the frame is created (first output-bearing
  // row) and must hold until the frame completes; every buffered row is
  // already linearised under it, so a later row resolving a different curve
  // would bin inconsistent data. The frame is `None` again after the final
  // row consumes it (and `begin_frame` clears it), so the freeze is per-frame.
  if let Some(b) = frame.as_ref()
    && b.frozen_transfer != tf
  {
    return Err(MixedSinkerError::TransferFunctionChanged(
      TransferFunctionChanged::new(idx),
    ));
  }

  // Mid-frame `LinearMode` change → typed rejection BEFORE any state mutation,
  // mirroring the `frozen_transfer` freeze directly above. The mode is captured
  // when the frame is created (first output-bearing row) and selects which
  // decode (display-referred Q15-clamped vs scene-referred unclamped `f32`)
  // fills the buffer the EOTF lifts; every buffered row is already decoded
  // under it, so a later row resolving a different mode would bin display- and
  // scene-decoded rows in one frame. The frame is `None` again after the final
  // row consumes it (and `begin_frame` clears it), so the freeze is per-frame.
  if let Some(b) = frame.as_ref()
    && b.frozen_linear_mode != mode
  {
    return Err(MixedSinkerError::LinearModeChanged(LinearModeChanged::new(
      idx,
    )));
  }

  let want_luma = luma.is_some() || luma_u16.is_some();

  // State-atomicity contract (stronger than reject-before-emit): EVERY fallible
  // operation of this call completes in this phase, BEFORE any persistent
  // accumulator mutation (the `*frame` commit on the first row, the `next_y`
  // advance, and the per-row frame-buffer writes) and before the first output
  // byte. The fallible operations are: the per-row scratch reserve here, the
  // decode (writes scratch only), and — on the final row — the whole bin tail's
  // allocations. On the FIRST output-bearing row the new frame is built into a
  // LOCAL (`new_frame`), NOT `*frame`, so a failure in this phase that precedes
  // the commit leaves `*frame` `None` (no `frozen_transfer`, no `next_y == 0`):
  // the call is fully retryable with a corrected transfer / matrix. The
  // bin-tail and decode read only plane DIMENSIONS (`w` / `h` / `plan`), never
  // persistent frame state, so they need no `*frame` borrow and run identically
  // whether or not the frame already existed. A failure here therefore leaves
  // the accumulator EXACTLY as it was on entry — `*frame` unchanged (`None` on
  // the first row, the prior buffer with its `next_y` unadvanced on later rows)
  // — so the same sink retries the row cleanly with no `begin_frame`.

  // On the first output-bearing row, build the frame into a LOCAL; it is only
  // committed to `*frame` after the rest of this fallible phase succeeds. On a
  // later row `*frame` already holds the (legitimately committed) frame.
  let mut new_frame = if frame.is_none() {
    Some(LinearLightFrame::new(
      w,
      h,
      plan.out_w(),
      plan.out_h(),
      want_luma,
      tf,
      mode,
    )?)
  } else {
    None
  };

  // The per-row decode scratch. The decode writes only the scratch (not
  // persistent state), so it is part of this fallible phase. A refusal here on
  // the first output-bearing row must leave `*frame` `None` (the frame above is
  // still only a local), so the failpoint fires before the scratch reserve and
  // therefore before the `*frame` commit. Both modes reserve a source-width
  // scratch under the same failpoint and the same recoverable contract; only
  // the element type and the decode differ (`u8` clamped vs `f32` unclamped).
  if take_linear_scratch_failure() {
    return Err(MixedSinkerError::Resample(
      crate::resample::ResampleError::AllocationFailed(crate::resample::PlanGeometry::new(
        w,
        h,
        plan.out_w(),
        plan.out_h(),
      )),
    ));
  }
  let scratch_alloc_failed = || {
    MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
      crate::resample::PlanGeometry::new(w, h, plan.out_w(), plan.out_h()),
    ))
  };
  match mode {
    // Display-referred: decode the clamped 8-bit RGB into the `u8` scratch.
    // This branch is BYTE-IDENTICAL to RFC #238 Phase 2 (same reserve, same
    // decode, same `byte / 255.0` lift below).
    LinearMode::DisplayReferred => {
      let rgb_bytes = rgb_row_bytes(w);
      if rgb_scratch.len() < rgb_bytes {
        rgb_scratch
          .try_reserve(rgb_bytes - rgb_scratch.len())
          .map_err(|_| scratch_alloc_failed())?;
        rgb_scratch.resize(rgb_bytes, 0);
      }
      decode_rgb_row(idx, &mut rgb_scratch[..rgb_bytes]);
    }
    // Scene-referred: decode the SAME affine matrix unclamped into the `f32`
    // scratch — out-of-gamut excursions preserved, no clamp / round.
    LinearMode::SceneReferred => {
      let n = w * 3;
      if scene_scratch.len() < n {
        scene_scratch
          .try_reserve(n - scene_scratch.len())
          .map_err(|_| scratch_alloc_failed())?;
        scene_scratch.resize(n, 0.0);
      }
      decode_unclamped_f32(idx, &mut scene_scratch[..n]);
    }
  }

  // On the final row, pre-build the entire bin tail — the f32 bin stream, the
  // `binned` accumulator, the (optional) luma u8 bin stream, AND the `enc_out`
  // re-encode row — here, still inside the fallible phase, so a refusal aborts
  // with the accumulator untouched (and, on a single-row frame, `*frame` still
  // `None`). The tail reads only plane DIMENSIONS — `w` / `h` (the frame's
  // `src_w` / `src_h`) and `plan` — never persistent frame state. Held in
  // `tail` and consumed after the commit below.
  let is_final = idx + 1 == h;
  let tail = if is_final && let Some(pre) = frame.as_mut().and_then(|b| b.reserved_tail.take()) {
    // RFC #238 Top: the two-row FINAL flush pre-reserved this tail (via
    // `linear_light_reserve_final_tail`) BEFORE its first feed, so the final feed
    // consumes it here with no allocation — the failpoint already fired at the
    // reserve, keeping the flush retry-atomic.
    let (stream, binned, y_stream, enc_out) = pre;
    Some((
      plan.out_w(),
      plan.out_h(),
      stream,
      binned,
      y_stream,
      enc_out,
    ))
  } else if is_final {
    let ow = plan.out_w();
    let oh = plan.out_h();
    if take_linear_tail_alloc_failure() {
      return Err(MixedSinkerError::Resample(
        crate::resample::ResampleError::AllocationFailed(crate::resample::PlanGeometry::new(
          w, h, ow, oh,
        )),
      ));
    }
    // The full-RGB `binned` element count, checked FIRST (before any tail
    // allocation): a one-channel-only Linear run validates only its
    // one-channel output size, so `ow * oh * 3` is otherwise unbounded on a
    // 32-bit target. A size overflow is a typed `GeometryOverflow`, never a
    // wrap into an undersized allocation.
    let rgb_len = linear_tail_rgb_len(ow, oh)?;
    // Allocation refusals in the bin tail are `AllocationFailed` (an
    // out-of-memory condition), NOT `GeometryOverflow` — the size was checked
    // just above, so a failure here is the allocator refusing a representable
    // size. Mirrors the failpoint above and the `AreaStream` `?` allocations,
    // which surface the same variant.
    let alloc_failed = || {
      MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
        crate::resample::PlanGeometry::new(w, h, ow, oh),
      ))
    };
    let stream = AreaStream::<f32>::new(plan.h(), plan.v(), w, h, 3)?;
    let binned = try_zeroed::<f32>(rgb_len).map_err(|_| alloc_failed())?;
    let y_stream = if want_luma {
      Some(AreaStream::<u8>::new(plan.h(), plan.v(), w, h, 1)?)
    } else {
      None
    };
    let enc_out = try_zeroed::<u8>(rgb_row_bytes(ow)).map_err(|_| alloc_failed())?;
    Some((ow, oh, stream, binned, y_stream, enc_out))
  } else {
    None
  };

  // ---- Commit phase: infallible from here. ----
  // Every fallible first-row operation has succeeded, so it is now safe to
  // install the newly-built frame AND freeze the output set together: both
  // `*frame` and `*resample_outputs` become `Some` only here, upholding the
  // invariant that each is `Some` iff at least one output-bearing row was fully
  // accepted. On a later row `new_frame` and `pending_outputs` are both `None`
  // (the frame and the freeze were committed on the accepted first row), so the
  // pre-existing committed state stands. Committing the output freeze at the same
  // point as `*frame` is what makes the first-row failure fully retryable with a
  // changed output attachment (a pre-commit failure left `*resample_outputs`
  // `None`, so the compare above does not mis-reject the corrected retry).
  if let Some(new_frame) = new_frame.take() {
    *frame = Some(new_frame);
  }
  if let Some(pending_outputs) = pending_outputs {
    *resample_outputs = Some(pending_outputs);
  }
  let buf = frame
    .as_mut()
    .expect("frame committed above (or pre-existing)");

  // Linearise this source row's decoded RGB into the frame buffer at `idx` and
  // advance `next_y`. Both run only now that every fallible step above has
  // succeeded. The EOTF is the SAME for both modes — only the normalized
  // encoded value it lifts differs:
  //  - display-referred: the clamped 8-bit code / 255 (in `[0, 1]`);
  //  - scene-referred: the unclamped real decode (a `[0, 1]` scale that MAY
  //    leave `[0, 1]`; the EOTF's odd-symmetric extrapolation handles it).
  let lin = &mut buf.linear_rgb[idx * w * 3..(idx + 1) * w * 3];
  match mode {
    LinearMode::DisplayReferred => {
      let enc_row = &rgb_scratch[..w * 3];
      for (l, &e) in lin.iter_mut().zip(enc_row.iter()) {
        *l = tf.eotf(e as f32 / 255.0);
      }
    }
    LinearMode::SceneReferred => {
      let enc_row = &scene_scratch[..w * 3];
      for (l, &e) in lin.iter_mut().zip(enc_row.iter()) {
        *l = tf.eotf(e);
      }
    }
  }
  if want_luma {
    buf.y_plane[idx * w..(idx + 1) * w].copy_from_slice(&y_row[..w]);
  }
  buf.next_y += 1;

  // Non-final rows buffer and return; only the final row runs the bin tail.
  let Some((ow, oh, mut stream, mut binned, mut y_stream, mut enc_out)) = tail else {
    return Ok(());
  };

  // Past this point the only fallible call is `feed_row`, which errors only
  // on a non-sequential row; every feed below walks `0..src_h` in order on a
  // stream freshly built at `next_y == 0`, so the `?` is unreachable here and
  // no output write can be followed by a real failure. The frame's output is
  // therefore all-or-nothing.
  let stride = buf.src_w * 3;
  for sy in 0..buf.src_h {
    stream.feed_row(
      sy,
      &buf.linear_rgb[sy * stride..(sy + 1) * stride],
      use_simd,
      |oy, finalized| {
        binned[oy * ow * 3..(oy + 1) * ow * 3].copy_from_slice(finalized);
      },
    )?;
  }

  if let Some(y_stream) = y_stream.as_mut() {
    for sy in 0..buf.src_h {
      y_stream.feed_row(
        sy,
        &buf.y_plane[sy * buf.src_w..(sy + 1) * buf.src_w],
        use_simd,
        |oy, finalized| {
          if let Some(luma) = luma.as_deref_mut() {
            luma[oy * ow..(oy + 1) * ow].copy_from_slice(finalized);
          }
          if let Some(luma_u16) = luma_u16.as_deref_mut() {
            y_plane_to_luma_u16_row(
              finalized,
              &mut luma_u16[oy * ow..(oy + 1) * ow],
              ow,
              use_simd,
            );
          }
        },
      )?;
    }
  }

  // Re-encode the binned linear RGB into a per-output encoded RGB row, then
  // fan it out to the requested colour outputs.
  for oy in 0..oh {
    let lin_row = &binned[oy * ow * 3..(oy + 1) * ow * 3];
    for (o, &l) in enc_out[..ow * 3].iter_mut().zip(lin_row.iter()) {
      *o = (tf.oetf(l) * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
    }
    if let Some(rgb) = rgb.as_deref_mut() {
      rgb[oy * ow * 3..(oy + 1) * ow * 3].copy_from_slice(&enc_out[..ow * 3]);
    }
    if let Some(rgba) = rgba.as_deref_mut() {
      expand_rgb_to_rgba_row(
        &enc_out[..ow * 3],
        &mut rgba[oy * 4 * ow..(oy + 1) * 4 * ow],
        ow,
      );
    }
    if let Some(hsv) = hsv.as_mut() {
      // HSV derives from the re-encoded RGB, exactly as the encoded tiers
      // derive HSV from their (encoded) RGB row.
      let (hh, hs, hv) = hsv.hsv();
      rgb_to_hsv_row(
        &enc_out[..ow * 3],
        &mut hh[oy * ow..(oy + 1) * ow],
        &mut hs[oy * ow..(oy + 1) * ow],
        &mut hv[oy * ow..(oy + 1) * ow],
        ow,
        use_simd,
      );
    }
  }
  Ok(())
}
