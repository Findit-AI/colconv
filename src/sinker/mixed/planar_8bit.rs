//! 8-bit planar YUV `MixedSinker` impls: Yuv410p / Yuv420p / Yuv422p / Yuv444p / Yuv440p.

use super::{
  AveragingDomainChanged, Chroma411U8, ChromaSitingChanged, ChromaU8, GeometryOverflow,
  HsvFrameMut, InsufficientBuffer, MixedSinker, MixedSinkerError, NativeRouteChanged,
  RowIndexOutOfRange, RowShapeMismatch, RowSlice, WidthAlignment, check_dimensions_match,
  chroma_410_center_sited_h, chroma_411_center_sited_h, chroma_420_bottom_sited_v,
  chroma_420_center_sited_h, chroma_420_top_sited_v, chroma_422_center_sited_h,
  chroma_440_bottom_sited_v, frozen_outputs_check,
  planar_resample::{planar_dual_filter_resample, planar_dual_resample},
  reconstruct_chroma, rgb_row_buf_or_scratch, rgba_plane_row_slice,
};
use crate::{
  ColorMatrix, PixelSink,
  resample::{
    AreaStream, AveragingDomain, FilterStream, InsertionContext, InsertionPoint, OutOfSequenceRow,
    PlanGeometry, ResampleError, ResamplePlan, select_insertion_point, try_zeroed,
  },
  row::*,
  source::*,
};
// The RFC #238 linear-light tail + its caller-configurable transfer curve
// drive the `rgb`-gated `AveragingDomain::Linear` dispatch only.
#[cfg(feature = "rgb")]
use super::linear_light;
#[cfg(feature = "rgb")]
use crate::resample::TransferFunction;

/// `Yuv420p` ships the native 4:2:0 fast tier ([`yuv420p_process_native`]),
/// so it is statically eligible to splice an [`AveragingDomain::Encoded`]
/// area downscale at the native codes.
const YUV420P_NATIVE_ELIGIBLE: bool = true;

/// `Yuv422p` / `Yuv444p` / `Yuv440p` ship the non-4:2:0 native planar fast
/// tier ([`yuv_planar_process_native`]), so each is statically eligible to
/// splice an [`AveragingDomain::Encoded`] area downscale at the native codes.
const YUV_PLANAR_8BIT_NATIVE_ELIGIBLE: bool = true;

/// RFC #238 S1 — the `Yuv422p` horizontal **centered** chroma phase, in
/// chroma-sample units. FFmpeg's centered group (`Center` / `Top` / `Bottom`,
/// [`chroma_422_center_sited_h`]) places chroma at `+0.5` luma-sample =
/// `+0.25` chroma-sample (subsample factor 2) relative to the left-co-sited
/// default. S1 realizes this phase through the #302 `1/4`–`3/4` triangle
/// **directly** — the native folded chroma weights
/// ([`ResamplePlan::area_chroma_422`]) and the `convert_rgb` full-width
/// reconstruction ([`reconstruct_chroma`]) — so the centered *direction* is
/// anchored to the #302 reconstruction by construction (and pinned by the
/// centered-vs-oracle tests). The numeric value is carried on the plan only as
/// the non-zero discriminant today; the phase-additive *filter* axes a later
/// PR (BICUBLIN) consumes calibrate the sign against this same #302 anchor.
pub(super) const YUV422P_CENTERED_H_PHASE: f64 = 0.25;

/// The centered (`Center` / `Top` / `Bottom`) horizontal chroma phase for the
/// `1→4`-subsampled 4:1:1 / 4:1:0 layouts (#302), the factor-4 sibling of
/// [`YUV422P_CENTERED_H_PHASE`]. The centered sample sits at the CENTER of its
/// four luma columns ([`chroma_411_center_sited_h`] /
/// [`chroma_410_center_sited_h`]) — luma `+1.5` = `+0.375` chroma-sample
/// (subsample factor 4) relative to the left-co-sited default — realized through
/// the #302 `1→4` triangle (the native folded weights
/// [`ResamplePlan::area_chroma_411`] / [`ResamplePlan::area_chroma_410`] and the
/// full-width [`reconstruct_chroma`] reconstruction). Carried on the plan only as
/// the non-zero discriminant today (the folded weights are geometry-fixed); the
/// numeric value anchors the phase-additive filter axes a later PR consumes.
pub(super) const YUV41X_CENTERED_H_PHASE: f64 = 0.375;

/// **Fallible preflight** for the centered-siting 4:2:0 chroma scratch
/// (#302): grows `chroma_full` to the checked `2 * width` so the later
/// infallible [`upsample_420_chroma_center_h`] reuses an already-sized
/// buffer. The allocation is split out from the upsample so it can run
/// **before any output row is written** (luma included) — the crate's
/// preflight-ordering atomicity contract (cf. the #180 resample fix): an
/// allocator refusal must leave the output frame *untouched*, never
/// partially mutated.
///
/// Mirrors [`rgb_row_buf_or_scratch`](super::rgb_row_buf_or_scratch)'s
/// recoverable grow: the `2 * width` length is `checked_mul`'d (→
/// [`GeometryOverflow`]) and `try_reserve_exact` precedes the resize (→
/// [`ResampleError::AllocationFailed`]), so the failure is a typed,
/// recoverable error rather than an abort. `height` feeds the error payloads.
///
/// `pub(super)` because the semi-planar `Nv12` / `Nv21` centered-siting path
/// (#302) reserves the SAME full-width chroma scratch — it de-interleaves the
/// interleaved UV into half-width U / V, then reuses this staging buffer.
pub(super) fn reserve_420_chroma_full(
  chroma_full: &mut std::vec::Vec<u8>,
  width: usize,
  height: usize,
) -> Result<(), MixedSinkerError> {
  // Test-only failpoint: simulate a recoverable allocator refusal of the
  // chroma-scratch grow WITHOUT exhausting memory, so the atomicity
  // regression test can prove no output row (esp. luma) is written before
  // this preflight. `take()` fires the armed flag exactly once. Mirrors
  // `arm_rgb_scratch_alloc_failure`; the non-test build compiles this away.
  #[cfg(all(test, feature = "std", feature = "yuv-planar"))]
  if super::FORCE_CHROMA_FULL_ALLOC_FAILURE.with(|f| f.take()) {
    return Err(MixedSinkerError::Resample(ResampleError::AllocationFailed(
      PlanGeometry::new(width, height, width, height),
    )));
  }
  let needed = width
    .checked_mul(2)
    .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
      width, height, 2,
    )))?;
  if chroma_full.len() < needed {
    chroma_full
      .try_reserve_exact(needed - chroma_full.len())
      .map_err(|_| {
        MixedSinkerError::Resample(ResampleError::AllocationFailed(PlanGeometry::new(
          width, height, width, height,
        )))
      })?;
    chroma_full.resize(needed, 0);
  }
  Ok(())
}

/// **Fallible preflight** for the bottom-sited (`AVCHROMA_LOC_BOTTOM`)
/// vertical-phase 4:2:0 chroma lookback (#302): grows `chroma_prev` to `width`
/// `u8` (half-width U then V) so the later infallible
/// [`upsample_420_chroma_sited`] can read the previous chroma row for the even
/// output row's vertical box blend. Split out from the upsample, like
/// [`reserve_420_chroma_full`], so it runs **before any output row is written**
/// — the crate's preflight-ordering atomicity contract (an allocator refusal
/// must leave the output frame untouched). `try_reserve_exact` precedes the
/// resize (→ [`ResampleError::AllocationFailed`]); `width` already fits `usize`
/// (the Y plane is `width * height`), so no `checked_mul` is needed. `height`
/// feeds the error payload.
///
/// `pub(super)` so the sibling-module `Yuva420p` 4:2:0 sink (RFC #238 S4-D)
/// reuses this shared bottom-sited lookback reservation for its resample
/// reconstruction arms and direct decode.
#[cfg(feature = "yuv-planar")]
pub(super) fn reserve_420_chroma_prev(
  chroma_prev: &mut std::vec::Vec<u8>,
  width: usize,
  height: usize,
) -> Result<(), MixedSinkerError> {
  // Test-only failpoint: simulate a recoverable allocator refusal of the
  // lookback grow WITHOUT exhausting memory, so the atomicity regression test
  // can prove no output row is written before this preflight. Mirrors
  // `reserve_420_chroma_full`'s failpoint; compiled away in non-test builds.
  #[cfg(all(test, feature = "std", feature = "yuv-planar"))]
  if super::FORCE_CHROMA_PREV_ALLOC_FAILURE.with(|f| f.take()) {
    return Err(MixedSinkerError::Resample(ResampleError::AllocationFailed(
      PlanGeometry::new(width, height, width, height),
    )));
  }
  if chroma_prev.len() < width {
    chroma_prev
      .try_reserve_exact(width - chroma_prev.len())
      .map_err(|_| {
        MixedSinkerError::Resample(ResampleError::AllocationFailed(PlanGeometry::new(
          width, height, width, height,
        )))
      })?;
    chroma_prev.resize(width, 0);
  }
  Ok(())
}

/// Stages the current half-width chroma row into the bottom-sited
/// vertical-phase lookback (#302): copies `u_half` then `v_half` into
/// `chroma_prev` (`[0..width/2]` = U, `[width/2..width]` = V) and tags it with
/// the chroma row it now holds (`idx / 2`), so a *later* even output row can
/// validate (`chroma_prev_row == Some(its_chroma_row - 1)`) and box-blend it.
///
/// Called on EVERY accepted bottom-sited row — both the colour-decode path
/// ([`upsample_420_chroma_sited`], after it has read the *previous* lookback)
/// and the luma-only / `luma_u16`-only path (which never reaches that helper) —
/// so the lookback is always current regardless of which outputs are attached.
/// **Infallible**: the caller must have run [`reserve_420_chroma_prev`] up front
/// (the luma-only path does so before its luma write), so `chroma_prev` is
/// guaranteed `>= width` here.
///
/// `pub(super)` so the sibling-module `Yuva420p` 4:2:0 sink (RFC #238 S4-D)
/// stages the same lookback after its deferred, commit-accepted bottom rows.
#[cfg(feature = "yuv-planar")]
#[inline]
pub(super) fn stage_420_chroma_prev(
  chroma_prev: &mut [u8],
  chroma_prev_row: &mut Option<usize>,
  u_half: &[u8],
  v_half: &[u8],
  idx: usize,
  width: usize,
) {
  debug_assert!(
    chroma_prev.len() >= width,
    "chroma_prev must be reserved via reserve_420_chroma_prev first"
  );
  let half = width / 2;
  chroma_prev[..half].copy_from_slice(&u_half[..half]);
  chroma_prev[half..width].copy_from_slice(&v_half[..half]);
  *chroma_prev_row = Some(idx / 2);
}

/// Horizontally upsamples the half-width U / V rows of a centered-siting
/// 4:2:0 source to full width into the **already-reserved** `chroma_full`
/// (#302), returning the two full-width chroma slices `(u_full, v_full)`.
/// The buffer is split `[0..width]` = U, `[width..2*width]` = V; each half is
/// filled by [`chroma_upsample_2to1_center_h`](crate::row::scalar::chroma_upsample_2to1_center_h)
/// (the MPEG-1 / JPEG phase-0.5 reconstruction), then fed to the 4:4:4 decode
/// kernels — so the centered path reuses their SIMD backends and stays
/// bit-identical per tier.
///
/// **Infallible**: the caller must have run [`reserve_420_chroma_full`] up
/// front (every centered-siting output path does, before any output write),
/// so `chroma_full` is guaranteed `>= 2 * width` here and `2 * width` cannot
/// overflow. Only the centered sitings reach here; the default
/// left/unspecified path never touches this scratch.
///
/// `pub(super)` because the semi-planar `Nv12` / `Nv21` centered path (#302)
/// feeds it the U / V it de-interleaved from the source's interleaved UV — so
/// a centered NV decode is bit-identical to the planar twin of those planes on
/// the shared matrix-tag path (NV resolves `ChromaDerivedNcl` via the BT.709
/// fallback, not `Yuv420p`'s #316 primaries-derived path — a #302/#303
/// follow-up).
pub(super) fn upsample_420_chroma_center_h<'s>(
  chroma_full: &'s mut [u8],
  u_half: &[u8],
  v_half: &[u8],
  width: usize,
) -> (&'s [u8], &'s [u8]) {
  debug_assert!(
    chroma_full.len() >= 2 * width,
    "chroma_full must be reserved via reserve_420_chroma_full first"
  );
  let (u_full, v_full) = chroma_full[..2 * width].split_at_mut(width);
  crate::row::scalar::chroma_upsample_2to1_center_h(u_half, u_full, width);
  crate::row::scalar::chroma_upsample_2to1_center_h(v_half, v_full, width);
  (u_full, v_full)
}

/// Co-sited (`h = 0`) full-width 4:2:0 chroma reconstruction — the co-sited twin
/// of [`upsample_420_chroma_center_h`] used by the `BottomLeft` siting, whose
/// horizontal axis is nearest-neighbor rather than the `h = 0.5` centered phase.
/// Each half of the already-reserved `chroma_full` is filled by
/// [`chroma_upsample_2to1_cosited_h`](crate::row::scalar::chroma_upsample_2to1_cosited_h)
/// (a plain 2× replicate), then fed to the same 4:4:4 decode kernels — so a
/// `BottomLeft` decode reuses their SIMD backends and reconstructs its `v = 1`
/// vertical fold ([`upsample_420_chroma_sited`]) atop a consistent full-width
/// horizontal staging.
///
/// **Infallible**: the caller must have run [`reserve_420_chroma_full`] up front,
/// so `chroma_full` is guaranteed `>= 2 * width` here.
///
/// `pub(super)` for parity with [`upsample_420_chroma_center_h`], so the
/// sibling `Nv12` / `Nv21` and `Yuva420p` 4:2:0 sinks reach the same co-sited
/// staging on the shared `super` path.
#[cfg(feature = "yuv-planar")]
pub(super) fn upsample_420_chroma_cosited_h<'s>(
  chroma_full: &'s mut [u8],
  u_half: &[u8],
  v_half: &[u8],
  width: usize,
) -> (&'s [u8], &'s [u8]) {
  debug_assert!(
    chroma_full.len() >= 2 * width,
    "chroma_full must be reserved via reserve_420_chroma_full first"
  );
  let (u_full, v_full) = chroma_full[..2 * width].split_at_mut(width);
  crate::row::scalar::chroma_upsample_2to1_cosited_h(u_half, u_full, width);
  crate::row::scalar::chroma_upsample_2to1_cosited_h(v_half, v_full, width);
  (u_full, v_full)
}

/// Siting-aware full-width 4:2:0 chroma reconstruction for the `Yuv420p`
/// identity path (#302), folding the optional **bottom-sited vertical** phase
/// into the horizontal centered upsample and maintaining the one-row chroma
/// lookback. Returns the full-width `(u_full, v_full)` slices the 4:4:4 decode
/// reads, staged in the already-reserved `chroma_full`.
///
/// `bottom_v` is [`chroma_420_bottom_sited_v`](super::chroma_420_bottom_sited_v)
/// (true for `ChromaLocation::Bottom` and `BottomLeft`, both `v = 1`) and
/// `center_h` is [`chroma_420_center_sited_h`](super::chroma_420_center_sited_h)
/// (true for `Bottom`'s `h = 0.5`, false for `BottomLeft`'s co-sited `h = 0`),
/// so the horizontal reconstruction is centered for `Bottom` and
/// nearest-neighbor for `BottomLeft`:
///
/// - On an **even** luma row (`idx & 1 == 0`) with `bottom_v`, each chroma sample
///   is the vertical box average of the previous chroma row (`chroma_prev`) and
///   the current row (`u_half` / `v_half`), then horizontally reconstructed — one
///   fused pass via
///   [`chroma_upsample_420_bottom_even_h`](crate::row::scalar::chroma_upsample_420_bottom_even_h)
///   (centered) or
///   [`chroma_upsample_420_bottomleft_even_h`](crate::row::scalar::chroma_upsample_420_bottomleft_even_h)
///   (co-sited) — **but only when `chroma_prev` provably holds the wanted
///   predecessor** chroma row `idx/2 - 1` (`chroma_prev_row == Some(idx/2 - 1)`).
///   When it does not — the top edge (`idx == 0`), or a direct `process` caller
///   that replayed / skipped / reordered rows or attached colour late, or a fresh
///   frame whose lookback was reset in `begin_frame` — it falls back to the plain
///   horizontal upsample of the *current* chroma row (the top-edge clamp: current
///   row as its own predecessor). So the blend NEVER mixes stale chroma from an
///   older pair or a previous frame; the worst a broken sequence does is omit the
///   vertical average for that row.
/// - Otherwise (odd luma row, or a non-bottom centered siting) the current chroma
///   row is horizontally reconstructed with the plain
///   [`chroma_upsample_2to1_center_h`] (centered) or
///   [`chroma_upsample_2to1_cosited_h`](crate::row::scalar::chroma_upsample_2to1_cosited_h)
///   (co-sited) — byte-identical to the horizontal-only path.
///
/// When `stage` is set, on every bottom-sited row the current half-width chroma
/// row is copied into `chroma_prev` and `chroma_prev_row` is set to `idx / 2`,
/// so the next pair's even row can validate and read it (the in-order walker
/// path stages chroma row `i` from pair `i`'s rows as the predecessor pair
/// `i+1`'s even row wants). The direct decode passes `stage = true` (its
/// post-reconstruction work is infallible, so advancing the lookback here is
/// safe); the resample reconstruction arms pass `stage = false` and defer
/// [`stage_420_chroma_prev`] until AFTER their fallible resample commit accepts
/// the row, so a rejected row leaves the lookback pointing at the predecessor
/// and a retry of the same even row still box-blends it (state-atomic, #180).
/// The caller must have run [`reserve_420_chroma_full`] (always) and, when
/// `bottom_v`, [`reserve_420_chroma_prev`] up front, so both buffers are sized
/// here and this is infallible.
///
/// `pub(super)` so the sibling-module `Yuva420p` 4:2:0 sink (RFC #238 S4-D)
/// reconstructs the same bottom-sited full-width chroma before its alpha-aware
/// 4:4:4 convert (the α plane is siting-independent and never routed here).
#[cfg(feature = "yuv-planar")]
#[allow(clippy::too_many_arguments)]
pub(super) fn upsample_420_chroma_sited<'s>(
  chroma_full: &'s mut [u8],
  chroma_prev: &mut [u8],
  chroma_prev_row: &mut Option<usize>,
  u_half: &[u8],
  v_half: &[u8],
  idx: usize,
  bottom_v: bool,
  center_h: bool,
  stage: bool,
  width: usize,
) -> (&'s [u8], &'s [u8]) {
  debug_assert!(
    chroma_full.len() >= 2 * width,
    "chroma_full must be reserved via reserve_420_chroma_full first"
  );
  let half = width / 2;
  let chroma_row = idx / 2;
  // The bottom-sited EVEN row box-blends only when the lookback PROVABLY holds
  // the wanted predecessor `chroma_row - 1`; otherwise it clamps to the current
  // row (never stale). Every other case (non-bottom siting, the bottom-sited odd
  // row, or an unvalidated even row) is the plain horizontal upsample of the
  // current chroma row — centered or co-sited per `center_h` — delegated to the
  // shared wrappers, which keep those helpers referenced in feature subsets
  // without the semi-planar / YUVA callers.
  let do_vblend =
    bottom_v && idx & 1 == 0 && chroma_row > 0 && *chroma_prev_row == Some(chroma_row - 1);

  let result = if do_vblend {
    debug_assert!(
      chroma_prev.len() >= width,
      "chroma_prev must be reserved via reserve_420_chroma_prev first"
    );
    let (u_full, v_full) = chroma_full[..2 * width].split_at_mut(width);
    let (u_prev, v_prev) = chroma_prev[..width].split_at(half);
    if center_h {
      crate::row::scalar::chroma_upsample_420_bottom_even_h(u_prev, u_half, u_full, width);
      crate::row::scalar::chroma_upsample_420_bottom_even_h(v_prev, v_half, v_full, width);
    } else {
      crate::row::scalar::chroma_upsample_420_bottomleft_even_h(u_prev, u_half, u_full, width);
      crate::row::scalar::chroma_upsample_420_bottomleft_even_h(v_prev, v_half, v_full, width);
    }
    (&*u_full, &*v_full)
  } else if center_h {
    upsample_420_chroma_center_h(chroma_full, u_half, v_half, width)
  } else {
    upsample_420_chroma_cosited_h(chroma_full, u_half, v_half, width)
  };

  // Refresh the lookback with the current chroma row + its validity tag so the
  // next pair's even row can trust it (after the read above). Gated on `stage`:
  // the direct decode refreshes it here (its later work is infallible), while
  // the resample reconstruction arms pass `stage = false` and defer the refresh
  // to AFTER their fallible commit accepts the row, so a rejected row does not
  // advance the lookback. Only the bottom-sited path maintains it; every other
  // siting leaves `chroma_prev` empty and its tag `None`.
  if bottom_v && stage {
    stage_420_chroma_prev(chroma_prev, chroma_prev_row, u_half, v_half, idx, width);
  }
  result
}

/// **Fallible preflight** for the bottom-sited (`AVCHROMA_LOC_BOTTOM`)
/// vertical-phase 4:4:0 chroma lookback: grows `chroma_prev` to `2 * width`
/// `u8` (full-width U then full-width V) so the later infallible
/// [`upsample_440_chroma_sited`] can read the previous chroma row for the even
/// output row's vertical box blend. 4:4:0 keeps FULL-width chroma (no
/// horizontal subsampling), so the lookback is twice the half-width 4:2:0
/// [`reserve_420_chroma_prev`]. Split out from the upsample so it runs **before
/// any output row is written** — the crate's preflight-ordering atomicity
/// contract (an allocator refusal must leave the output frame untouched).
/// `2 * width` is `checked_mul`'d (→ [`GeometryOverflow`]); `try_reserve_exact`
/// precedes the resize (→ [`ResampleError::AllocationFailed`]). `height` feeds
/// the error payload.
#[cfg(feature = "yuv-planar")]
pub(super) fn reserve_440_chroma_prev(
  chroma_prev: &mut std::vec::Vec<u8>,
  width: usize,
  height: usize,
) -> Result<(), MixedSinkerError> {
  // Test-only failpoint (shared with the 4:2:0 lookback): simulate a recoverable
  // allocator refusal WITHOUT exhausting memory, so the atomicity regression can
  // prove no output row is written before this preflight. Compiled away in
  // non-test builds.
  #[cfg(all(test, feature = "std", feature = "yuv-planar"))]
  if super::FORCE_CHROMA_PREV_ALLOC_FAILURE.with(|f| f.take()) {
    return Err(MixedSinkerError::Resample(ResampleError::AllocationFailed(
      PlanGeometry::new(width, height, width, height),
    )));
  }
  let needed = width
    .checked_mul(2)
    .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
      width, height, 2,
    )))?;
  if chroma_prev.len() < needed {
    chroma_prev
      .try_reserve_exact(needed - chroma_prev.len())
      .map_err(|_| {
        MixedSinkerError::Resample(ResampleError::AllocationFailed(PlanGeometry::new(
          width, height, width, height,
        )))
      })?;
    chroma_prev.resize(needed, 0);
  }
  Ok(())
}

/// Stages the current full-width chroma row into the bottom-sited
/// vertical-phase 4:4:0 lookback: copies `u` then `v` into `chroma_prev`
/// (`[0..width]` = U, `[width..2*width]` = V) and tags it with the chroma row it
/// now holds (`idx / 2`), so a *later* even output row can validate
/// (`chroma_prev_row == Some(its_chroma_row - 1)`) and box-blend it. 4:4:0
/// chroma is full-width, so both halves are `width` wide (twice the half-width
/// 4:2:0 [`stage_420_chroma_prev`]).
///
/// Called on EVERY accepted bottom-sited row — the colour-decode path
/// ([`upsample_440_chroma_sited`], after it has read the *previous* lookback)
/// and the luma-only path alike — so the lookback is always current regardless
/// of which outputs are attached. **Infallible**: the caller must have run
/// [`reserve_440_chroma_prev`] up front, so `chroma_prev` is guaranteed
/// `>= 2 * width` here.
#[cfg(feature = "yuv-planar")]
#[inline]
pub(super) fn stage_440_chroma_prev(
  chroma_prev: &mut [u8],
  chroma_prev_row: &mut Option<usize>,
  u: &[u8],
  v: &[u8],
  idx: usize,
  width: usize,
) {
  debug_assert!(
    chroma_prev.len() >= 2 * width,
    "chroma_prev must be reserved via reserve_440_chroma_prev first"
  );
  chroma_prev[..width].copy_from_slice(&u[..width]);
  chroma_prev[width..2 * width].copy_from_slice(&v[..width]);
  *chroma_prev_row = Some(idx / 2);
}

/// Siting-aware full-width 4:4:0 chroma reconstruction for the `Yuv440p`
/// `Bottom` (`v = 1`) vertical phase, maintaining the one-row chroma lookback.
/// Returns the full-width `(u_full, v_full)` slices the 4:4:4 decode reads,
/// staged in the already-reserved `chroma_full`. 4:4:0 keeps FULL-width chroma,
/// so — unlike the 4:2:0 [`upsample_420_chroma_sited`] — there is NO horizontal
/// reconstruction: the even output row is a pure vertical box blend and every
/// other case is a straight copy of the current chroma row.
///
/// - On an **even** luma row (`idx & 1 == 0`) with `bottom_v`, each output
///   sample is the vertical box average of the previous chroma row
///   (`chroma_prev`) and the current row (`u` / `v`) via
///   [`chroma_upsample_440_bottom_v`](crate::row::scalar::chroma_upsample_440_bottom_v)
///   — **but only when `chroma_prev` provably holds the wanted predecessor**
///   chroma row `idx/2 - 1` (`chroma_prev_row == Some(idx/2 - 1)`). When it does
///   not — the top edge (`idx == 0`), or a caller that replayed / skipped /
///   reordered rows or attached colour late, or a fresh frame whose lookback
///   `begin_frame` reset — it clamps to a straight copy of the *current* chroma
///   row. So the blend NEVER mixes stale chroma from an older pair or a previous
///   frame.
/// - Otherwise (the bottom-sited odd row `2i+1`, co-sited with chroma row `i`)
///   the current chroma row is copied straight through.
///
/// When `stage` is set the current chroma row is copied into `chroma_prev`
/// (tagged `idx / 2`) so the next pair's even row can validate and read it. The
/// direct decode passes `stage = true` (its post-reconstruction work is
/// infallible, so advancing the lookback here is safe); the resample
/// reconstruction arms pass `stage = false` and defer [`stage_440_chroma_prev`]
/// until AFTER their fallible resample commit accepts the row, so a rejected row
/// leaves the lookback pointing at the predecessor and a retry of the same even
/// row still box-blends it (state-atomic). The caller must have run
/// [`reserve_420_chroma_full`] (always) and [`reserve_440_chroma_prev`] up
/// front, so both buffers are sized here and this is infallible.
#[cfg(feature = "yuv-planar")]
#[allow(clippy::too_many_arguments)]
pub(super) fn upsample_440_chroma_sited<'s>(
  chroma_full: &'s mut [u8],
  chroma_prev: &mut [u8],
  chroma_prev_row: &mut Option<usize>,
  u: &[u8],
  v: &[u8],
  idx: usize,
  bottom_v: bool,
  stage: bool,
  width: usize,
) -> (&'s [u8], &'s [u8]) {
  debug_assert!(
    chroma_full.len() >= 2 * width,
    "chroma_full must be reserved via reserve_420_chroma_full first"
  );
  let chroma_row = idx / 2;
  // The bottom-sited EVEN row box-blends only when the lookback PROVABLY holds
  // the wanted predecessor `chroma_row - 1`; otherwise it clamps to a straight
  // copy of the current row (never stale). Every other case (the bottom-sited
  // odd row, or an unvalidated even row) copies the current full-width chroma
  // row straight through — 4:4:0 needs no horizontal reconstruction.
  let do_vblend =
    bottom_v && idx & 1 == 0 && chroma_row > 0 && *chroma_prev_row == Some(chroma_row - 1);
  let (u_full, v_full) = chroma_full[..2 * width].split_at_mut(width);
  if do_vblend {
    debug_assert!(
      chroma_prev.len() >= 2 * width,
      "chroma_prev must be reserved via reserve_440_chroma_prev first"
    );
    let (u_prev, v_prev) = chroma_prev[..2 * width].split_at(width);
    crate::row::scalar::chroma_upsample_440_bottom_v(u_prev, u, u_full, width);
    crate::row::scalar::chroma_upsample_440_bottom_v(v_prev, v, v_full, width);
  } else {
    u_full.copy_from_slice(&u[..width]);
    v_full.copy_from_slice(&v[..width]);
  }

  // Refresh the lookback with the current chroma row + its validity tag (after
  // the read above). Gated on `stage`: the direct decode refreshes here (its
  // later work is infallible); the resample arms pass `stage = false` and defer
  // the refresh to AFTER their fallible commit accepts the row.
  if bottom_v && stage {
    stage_440_chroma_prev(chroma_prev, chroma_prev_row, u, v, idx, width);
  }
  (&*u_full, &*v_full)
}

/// The `Yuv420p` identity 4:4:4 colour decode from an ALREADY-reconstructed
/// full-width chroma pair `(u_full, v_full)`, writing the requested colour
/// outputs (`rgb` / `rgba` / `hsv`) for output row `[one_plane_start,
/// one_plane_end)`. Factored out of the inline identity tail so the RFC #238
/// **Top** (`v = 0`) forward one-row delay can invoke it for BOTH the deferred
/// odd row (buffered Y + forward-blended chroma) and the current even row
/// (co-sited chroma) — the two output rows a single delayed `process` call
/// emits — with the SAME kernel sequence the inline centered / bottom-sited path
/// runs, so a delayed decode is bit-identical to an in-order one.
///
/// The three output modes mirror the inline tail: HSV-only (direct
/// [`yuv_444_to_hsv_row`]), RGBA-only ([`yuv_444_to_rgba_row_primaries`]), and
/// the RGB path ([`yuv_444_to_rgb_row_primaries`], then HSV-from-RGB and the
/// RGBA fan-out). `rgb_scratch` must already be grown by the caller's atomicity
/// preflight, so the only fallible steps are the checked `x 3` / `x 4`
/// output-range computations (deterministic, never allocation).
#[cfg(feature = "yuv-planar")]
#[allow(clippy::too_many_arguments)]
fn yuv444_identity_color_row(
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  rgb_scratch: &mut std::vec::Vec<u8>,
  y: &[u8],
  u_full: &[u8],
  v_full: &[u8],
  one_plane_start: usize,
  one_plane_end: usize,
  w: usize,
  h: usize,
  matrix: crate::ColorMatrix,
  primaries: crate::Primaries,
  full_range: bool,
  use_simd: bool,
) -> Result<(), MixedSinkerError> {
  let want_rgb = rgb.is_some();
  let want_rgba = rgba.is_some();
  let want_hsv = hsv.is_some();
  let want_hsv_direct = want_hsv && !want_rgb && !want_rgba;
  let need_rgb_kernel = want_rgb || (want_hsv && want_rgba);

  if want_hsv_direct {
    let hsv = hsv.as_mut().expect("want_hsv_direct implies hsv attached");
    let (hh, ss, vv) = hsv.hsv();
    yuv_444_to_hsv_row(
      y,
      u_full,
      v_full,
      &mut hh[one_plane_start..one_plane_end],
      &mut ss[one_plane_start..one_plane_end],
      &mut vv[one_plane_start..one_plane_end],
      w,
      matrix,
      full_range,
      use_simd,
    );
    return Ok(());
  }

  if want_rgba && !need_rgb_kernel {
    let rgba_buf = rgba.as_deref_mut().unwrap();
    let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
    yuv_444_to_rgba_row_primaries(
      y, u_full, v_full, rgba_row, w, matrix, primaries, full_range, use_simd,
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
  yuv_444_to_rgb_row_primaries(
    y, u_full, v_full, rgb_row, w, matrix, primaries, full_range, use_simd,
  );
  if let Some(hsv) = hsv.as_mut() {
    let (hh, ss, vv) = hsv.hsv();
    rgb_to_hsv_row(
      rgb_row,
      &mut hh[one_plane_start..one_plane_end],
      &mut ss[one_plane_start..one_plane_end],
      &mut vv[one_plane_start..one_plane_end],
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

// ---- Yuv420p impl --------------------------------------------------------

impl<'a, R> MixedSinker<'a, Yuv420p, R> {
  /// Attaches a packed 32‑bit RGBA output buffer.
  ///
  /// Only available on sinker types whose `PixelSink` impl writes
  /// RGBA — calling `with_rgba` on a sink that doesn't (e.g.
  /// [`MixedSinker<Nv12>`] today) is a compile error rather than a
  /// silent no‑op that would leave the caller's buffer stale while
  /// [`Self::produces_rgba`] returned `true`. The compile-time
  /// scoping is load-bearing: if a future format adds RGBA, it must
  /// add its own impl block here, which both wires the new path and
  /// prevents accidental cross-format leakage.
  ///
  /// The fourth byte per pixel is alpha. [`Yuv420p`] has no alpha
  /// plane, so every alpha byte is filled with `0xFF` (opaque).
  /// Future YUVA source impls will copy alpha through from the
  /// source plane.
  ///
  /// Returns `Err(InsufficientRgbaBuffer)` if
  /// `buf.len() < width x height x 4`, or `Err(GeometryOverflow)` on
  /// 32‑bit targets when the product overflows.
  ///
  /// ```compile_fail
  /// // Attaching RGBA to a sink that doesn't write it is rejected
  /// // at compile time. `Bayer` (RAW Bayer-mosaic) has no RGBA path —
  /// // there's no inherent alpha channel and the format demosaics to
  /// // RGB only. Once / if a future PR adds RGBA, the negative example
  /// // here moves to the next not‑yet‑wired format.
  /// use colconv::{sinker::MixedSinker, raw::Bayer};
  /// let mut buf = vec![0u8; 16 * 8 * 4];
  /// let _ = MixedSinker::<Bayer>::new(16, 8).with_rgba(&mut buf);
  /// ```
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

  /// Attaches a `u16` luma output buffer. The 8-bit Y plane samples
  /// are zero-extended into `u16` (i.e. each output element equals
  /// `y_byte as u16`). Length is measured in `u16` elements
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

impl<R> PixelSink for MixedSinker<'_, Yuv420p, R> {
  type Input<'r> = Yuv420pRow<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    // Reject odd-width sinkers up front — the underlying row
    // primitives assume `width & 1 == 0` and would panic on the
    // first `process` call otherwise (`MixedSinker::new` is
    // infallible and accepts any width).
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::WidthAlignment(WidthAlignment::odd(
        self.width,
      )));
    }
    check_dimensions_match(self.width, self.height, width, height)?;
    // New frame: restart the row-stage streams (the streams are
    // lazily created in `process`, so a direct-`process` caller that
    // skips `begin_frame` still gets a correctly initialized first
    // frame).
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
    if let Some(native) = self.native_420.as_mut() {
      native.reset();
    }
    if let Some(bicublin) = self.bicublin_420.as_mut() {
      bicublin.reset();
    }
    // New frame: restart the RGB-free HSV-only row-stage join (#263 follow-up).
    if let Some(hsv) = self.hsv_planar.as_mut() {
      hsv.reset();
    }
    // New frame: clear the per-frame frozen native/row-stage route and
    // averaging domain so the next frame may pick either tier / any domain; a
    // mid-frame flip stays rejected.
    self.frozen_native_route = None;
    self.frozen_domain = None;
    self.resample_outputs = None;
    // New frame: clear the RFC #238 frozen 4:2:0 chroma siting so the next frame
    // may pick either phase; a mid-frame flip stays rejected. (The siting
    // invalidation of the cached native / HSV-only join is done at point-of-use
    // in `process`, not here — `chroma_location` may change AFTER `begin_frame`
    // but before row 0, so the join's phase is re-checked immediately before it
    // is reused.)
    self.frozen_chroma_centered = None;
    // S4-B companion: clear the frozen vertical phase too, so the next frame may
    // pick Center or Bottom and a mid-frame Center ⇆ Bottom flip stays rejected.
    self.frozen_chroma_bottom_v = None;
    // RFC #238 Top companion: clear the frozen Top vertical phase so the next
    // frame may pick Top and a mid-frame flip to / from Top stays rejected.
    self.frozen_chroma_top_v = None;
    // New frame: invalidate the bottom-sited vertical-phase chroma lookback
    // (#302) so frame N+1's first even row can never box-blend frame N's last
    // chroma row. The buffer bytes are left as-is (re-overwritten before any
    // trusted read); only the validity tag is cleared.
    self.chroma_prev_row = None;
    // New frame: drop any held Top (`v = 0`) forward-delay odd row so frame N's
    // last deferred colour row never emits into frame N+1. Its Y buffer bytes
    // are left as-is (re-overwritten before any trusted read).
    self.chroma_top_pending = None;
    // New frame: drop the RFC #238 linear-light accumulator (if any) so the
    // next frame re-seeds it from row 0.
    #[cfg(feature = "rgb")]
    {
      self.linear_light_frame = None;
    }
    Ok(())
  }

  fn process(&mut self, row: Yuv420pRow<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;
    // Chroma siting (#302): drives the identity-plan horizontal chroma
    // phase. `Copy`, so read it out before the field split-borrow below.
    let chroma_location = self.chroma_location;
    // Source primaries (#303): the `ChromaDerivedNcl` decode derives its
    // Kr/Kb from these. `Copy`, read before the split-borrow; consumed by the
    // identity-plan RGB / RGBA path (non-`ChromaDerivedNcl` matrices ignore it).
    let primaries = self.primaries;

    // Defense in depth: `begin_frame` already validated frame‑level
    // dimensions, so these checks are unreachable from the walker.
    // They guard direct `process` callers (hand-crafted rows, row
    // replay) from handing a wrong-shaped row or out-of-range index
    // to unsafe SIMD kernels. Report the offending slice length and
    // row index directly — don't reuse `DimensionMismatch`, whose
    // `frame_w` / `frame_h` fields would be meaningless here.
    //
    // Odd-width check first: the row primitives assume
    // `width & 1 == 0` and would panic past this point. Keeping the
    // check here (and in `begin_frame`) preserves the no-panic
    // contract for direct `process` callers that skip `begin_frame`.
    if w & 1 != 0 {
      return Err(MixedSinkerError::WidthAlignment(WidthAlignment::odd(w)));
    }
    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Y,
        idx,
        w,
        row.y().len(),
      )));
    }
    if row.u_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::UHalf,
        idx,
        w / 2,
        row.u_half().len(),
      )));
    }
    if row.v_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::VHalf,
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

    // Split-borrow so the `rgb_scratch` path and the `hsv` write don't
    // collide with the `rgb` read-after-write chain below.
    let Self {
      rgb,
      rgba,
      luma,
      luma_u16,
      hsv,
      rgb_scratch,
      chroma_full,
      chroma_prev,
      chroma_prev_row,
      plan,
      rgb_stream,
      luma_stream,
      rgb_filter_stream,
      luma_filter_stream,
      resample_outputs,
      native,
      native_420,
      hsv_planar,
      bicublin_420,
      frozen_native_route,
      frozen_domain,
      averaging_domain,
      #[cfg(feature = "rgb")]
      linear_mode,
      #[cfg(feature = "rgb")]
      linear_light_frame,
      #[cfg(feature = "rgb")]
      linear_scene_scratch,
      #[cfg(feature = "rgb")]
      transfer_function,
      // RFC #238 S3a: the 4:2:0 chroma phase frozen on the first output row.
      frozen_chroma_centered,
      // RFC #238 S4-B: the vertical (`Bottom`) chroma phase frozen alongside it.
      frozen_chroma_bottom_v,
      // RFC #238 Top: the `Top` (`v = 0`) vertical phase frozen alongside it.
      frozen_chroma_top_v,
      // RFC #238 Top forward one-row delay line (identity decode + resample tiers).
      chroma_top_pending,
      chroma_top_y,
      ..
    } = self;

    // Non-identity plan: the native tier bins the Y/U/V planes at
    // output resolution and converts once per output row; the
    // row-stage tier converts this source row at source width, then
    // area-streams it. `with_native(false)` forces the latter.
    if let Some(plan) = plan.as_ref() {
      let matrix = row.matrix();
      let full_range = row.full_range();
      // RFC #238 S3a — 4:2:0 horizontal chroma siting. The centered group
      // (`Center` / `Top` / `Bottom`, [`chroma_420_center_sited_h`]) samples
      // chroma at `+0.5` luma horizontally; the co-sited / unspecified group is
      // phase 0 (today's nearest decode, byte-identical). Siting enters the
      // chroma RECONSTRUCTION only — the averaging DOMAIN is still chosen by
      // `select_insertion_point` below — so the native fast tier and the encoded
      // HSV-only arm fold the phase into the chroma area weights
      // (`area_chroma_420`), while the RGB-domain converters (row-stage RGB/RGBA,
      // filter, linear-light) reconstruct full-width chroma. VERTICAL stays
      // co-sited (`v_phase = 0`): S3a routes the horizontal Top/Center phase
      // only; `Bottom`'s vertical blend is a later stage. The centered arms build
      // their OWN 4:4:4 converter over freshly reconstructed chroma AFTER the
      // resample preflight, so a rejected / out-of-sequence row is caught before
      // that chroma reservation (the #180 reserve-after-preflight invariant).
      //
      // RFC #238 S4-B extends this to the VERTICAL axis for `Bottom`
      // ([`chroma_420_bottom_sited_v`], a strict sub-case of `center_sited` — it
      // is `h = 0.5, v = 1`): the binning tiers (native / HSV-only) fold the
      // `v = 1` triangle into `area_chroma_420`'s vertical weights, and the
      // RGB-domain reconstruction tiers box-blend the even output row's chroma
      // with the previous chroma row via the `chroma_prev` lookback. `Center` /
      // `Top` keep `v_phase = 0` (co-sited vertical, byte-identical to S3a).
      let center_sited = chroma_420_center_sited_h(chroma_location);
      let bottom_v = chroma_420_bottom_sited_v(chroma_location);
      // RFC #238 Top (`v = 0`, FORWARD fold). `Top` shares `Center`'s centered
      // horizontal phase, `TopLeft` the co-sited one; both fold the forward
      // vertical triangle. Disjoint from `bottom_v` (a chroma sample cannot be
      // both top- and bottom-sited). The binning tiers (native / HSV-only) fold
      // it into `area_chroma_420`'s vertical weights via `v_top`; the RGB-domain
      // reconstruction tiers reconstruct each row's chroma through a FORWARD
      // one-row delay (mirror of `Bottom`'s backward `chroma_prev` lookback).
      let top_v = chroma_420_top_sited_v(chroma_location);
      let chroma_h_phase = if center_sited {
        YUV422P_CENTERED_H_PHASE
      } else {
        0.0
      };
      let chroma_v_phase = if bottom_v { 1.0 } else { 0.0 };
      let want_color = rgb.is_some() || rgba.is_some() || hsv.is_some();
      // RFC #238 Phase 2 — single always-compiled choke point for the averaging
      // domain, BEFORE any filter / native / row-stage branching, so a
      // non-encoded sink can NEVER fall through to the Encoded path under ANY
      // feature combination. The match is EXHAUSTIVE with no wildcard arm: a
      // future `AveragingDomain` variant fails to compile here until it is
      // explicitly handled, so the silent-fallback class is structurally
      // impossible rather than merely audited. The `Encoded` arm is empty —
      // control continues into the encoded dispatch below; `Linear` and
      // `Premultiplied` both return.
      //
      // `need_output` — whether this call carries any output — is the EXACT set
      // both tiers' preflight tests (`need_luma || need_color` =
      // `luma || luma_u16 || rgb || rgba || hsv`); it gates BOTH the domain
      // freeze here and the native/row-stage route freeze below, so a no-output
      // call (which consumes no stream state) freezes neither.
      let need_output =
        luma.is_some() || luma_u16.is_some() || rgb.is_some() || rgba.is_some() || hsv.is_some();
      // CHECK the averaging-domain freeze BEFORE the choke-point match (so the
      // freeze guards the domain choice itself), parallel to the frozen native
      // route. This is CHECK-ONLY — the matching SET happens AFTER the selected
      // path ACCEPTS an output-bearing row (mirroring `frozen_native_route`'s
      // timing below), never before dispatch. Committing the freeze before the
      // row is accepted would poison a retry: a row the selected path rejects
      // (an unsupported domain / filter plan, an out-of-sequence or
      // output-changed row, an alloc failure) must leave `frozen_domain`
      // UNCHANGED so the caller can correct the config and retry the SAME row.
      // A no-output row neither checks nor sets (a true route-invisible no-op).
      if need_output
        && let Some(frozen) = *frozen_domain
        && frozen != *averaging_domain
      {
        return Err(MixedSinkerError::AveragingDomainChanged(
          AveragingDomainChanged::new(idx),
        ));
      }
      // RFC #238 S3a: freeze the effective 4:2:0 chroma siting on the first
      // output-bearing row, mirroring the averaging-domain freeze above. This
      // CHECK is at the always-compiled choke point (every domain / tier passes
      // through it); the matching SET rides each domain's accept-time freeze
      // below (never before dispatch, so a rejected row leaves it unset for a
      // corrected retry). A later row observing a different phase — in sequence
      // or not — would bin a mixture of co-sited and centered chroma, so it is
      // rejected here before any reconstruction.
      if need_output
        && let Some(frozen) = *frozen_chroma_centered
        && (frozen != center_sited
          || *frozen_chroma_bottom_v != Some(bottom_v)
          || *frozen_chroma_top_v != Some(top_v))
      {
        return Err(MixedSinkerError::ChromaSitingChanged(
          ChromaSitingChanged::new(idx),
        ));
      }
      match *averaging_domain {
        AveragingDomain::Encoded => {}
        // `with_averaging_domain` is gated on `yuv-planar` alone, but the
        // linear-light tail decodes to RGB and so compiles only under `rgb`.
        // Under `rgb` it runs the linear tail (which itself rejects a filter
        // plan with the typed `UnsupportedFilter` — the Linear domain is
        // area-only); without `rgb` it returns the typed
        // `LinearDomainUnsupported` rather than silently downgrading to the
        // encoded average.
        AveragingDomain::Linear => {
          #[cfg(feature = "rgb")]
          {
            let tf = transfer_function.unwrap_or_else(|| TransferFunction::for_matrix(matrix));
            // RFC #238 Top (`v = 0`) FORWARD one-row delay for the linear-light
            // reconstruction tier — the structure of the row-stage / filter delays,
            // adapted to the linear accumulator (which decodes each source row into
            // the frame buffer and area-bins in linear light on the FINAL row). The
            // whole odd row is HELD and the following even row decodes TWO source
            // rows into the buffer (held forward-blend + current co-sited); the
            // trailing odd row clamps to a co-sited decode. The one fallible
            // final-row allocation is the bin tail, which the ODD-height two-row
            // FINAL flush pre-reserves BEFORE either feed (so the final feed does no
            // allocation) — making the flush retry-atomic (#180).
            if top_v && want_color {
              let is_last = idx + 1 == h;
              let want_luma_tail = luma.is_some() || luma_u16.is_some();
              // The frame cursor lags the source by the buffered (unfed) held odd
              // row, so the next expected SOURCE row is `next_y + pending`. Reject
              // an out-of-sequence walker row BEFORE any reserve (#180); the
              // output-set / transfer / mode gate below runs against the FIRST
              // feed's frame position (== `next_y`) so it does not spuriously trip
              // the frame's own in-order sequence check.
              let cursor = linear_light_frame.as_ref().map_or(0, |b| b.next_y());
              let expected_source = cursor + usize::from(chroma_top_pending.is_some());
              if idx != expected_source {
                return Err(MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(
                  crate::resample::OutOfSequenceRow::new(expected_source, idx),
                )));
              }
              linear_light::linear_light_preflight(
                linear_light_frame,
                resample_outputs,
                luma,
                luma_u16,
                rgb,
                rgba,
                hsv,
                tf,
                *linear_mode,
                plan,
                cursor,
              )?;
              // Reserve every buffer BOTH feeds may touch, up front. The final
              // even flush's SECOND feed IS the final row, so pre-reserve its bin
              // tail here too — then both feeds are infallible (#180).
              reserve_420_chroma_full(chroma_full, w, h)?;
              reserve_420_chroma_prev(chroma_prev, w, h)?;
              if chroma_top_y.len() < w {
                chroma_top_y
                  .try_reserve_exact(w - chroma_top_y.len())
                  .map_err(|_| {
                    MixedSinkerError::Resample(ResampleError::AllocationFailed(PlanGeometry::new(
                      w, h, w, h,
                    )))
                  })?;
                chroma_top_y.resize(w, 0);
              }
              if idx & 1 == 0 && is_last && chroma_top_pending.is_some() {
                linear_light::linear_light_reserve_final_tail(
                  linear_light_frame,
                  plan,
                  w,
                  h,
                  want_luma_tail,
                )?;
              }
              if idx & 1 == 0 {
                if let Some((p_idx, p_matrix, p_full_range)) = chroma_top_pending.take() {
                  let (u_full, v_full) = upsample_420_chroma_sited(
                    chroma_full,
                    chroma_prev,
                    chroma_prev_row,
                    row.u_half(),
                    row.v_half(),
                    idx,
                    true,
                    center_sited,
                    false,
                    w,
                  );
                  linear_light::linear_light_resample(
                    linear_light_frame,
                    resample_outputs,
                    rgb,
                    rgba,
                    luma,
                    luma_u16,
                    hsv,
                    rgb_scratch,
                    linear_scene_scratch,
                    tf,
                    *linear_mode,
                    plan,
                    &chroma_top_y[..w],
                    p_idx,
                    w,
                    h,
                    use_simd,
                    |_idx, dst| {
                      yuv_444_to_rgb_row(
                        &chroma_top_y[..w],
                        u_full,
                        v_full,
                        dst,
                        w,
                        p_matrix,
                        p_full_range,
                        use_simd,
                      );
                    },
                    |_idx, dst| {
                      crate::row::scalar::yuv_444_to_rgb_f32_unclamped_row(
                        &chroma_top_y[..w],
                        u_full,
                        v_full,
                        dst,
                        w,
                        p_matrix,
                        p_full_range,
                      );
                    },
                  )?;
                }
                let (u_full, v_full) = upsample_420_chroma_sited(
                  chroma_full,
                  chroma_prev,
                  chroma_prev_row,
                  row.u_half(),
                  row.v_half(),
                  idx,
                  false,
                  center_sited,
                  false,
                  w,
                );
                linear_light::linear_light_resample(
                  linear_light_frame,
                  resample_outputs,
                  rgb,
                  rgba,
                  luma,
                  luma_u16,
                  hsv,
                  rgb_scratch,
                  linear_scene_scratch,
                  tf,
                  *linear_mode,
                  plan,
                  row.y(),
                  idx,
                  w,
                  h,
                  use_simd,
                  |_idx, dst| {
                    yuv_444_to_rgb_row(
                      row.y(),
                      u_full,
                      v_full,
                      dst,
                      w,
                      matrix,
                      full_range,
                      use_simd,
                    );
                  },
                  |_idx, dst| {
                    crate::row::scalar::yuv_444_to_rgb_f32_unclamped_row(
                      row.y(),
                      u_full,
                      v_full,
                      dst,
                      w,
                      matrix,
                      full_range,
                    );
                  },
                )?;
              } else if !is_last {
                chroma_top_y[..w].copy_from_slice(&row.y()[..w]);
                *chroma_top_pending = Some((idx, matrix, full_range));
              } else {
                let (u_full, v_full) = upsample_420_chroma_sited(
                  chroma_full,
                  chroma_prev,
                  chroma_prev_row,
                  row.u_half(),
                  row.v_half(),
                  idx,
                  false,
                  center_sited,
                  false,
                  w,
                );
                linear_light::linear_light_resample(
                  linear_light_frame,
                  resample_outputs,
                  rgb,
                  rgba,
                  luma,
                  luma_u16,
                  hsv,
                  rgb_scratch,
                  linear_scene_scratch,
                  tf,
                  *linear_mode,
                  plan,
                  row.y(),
                  idx,
                  w,
                  h,
                  use_simd,
                  |_idx, dst| {
                    yuv_444_to_rgb_row(
                      row.y(),
                      u_full,
                      v_full,
                      dst,
                      w,
                      matrix,
                      full_range,
                      use_simd,
                    );
                  },
                  |_idx, dst| {
                    crate::row::scalar::yuv_444_to_rgb_f32_unclamped_row(
                      row.y(),
                      u_full,
                      v_full,
                      dst,
                      w,
                      matrix,
                      full_range,
                    );
                  },
                )?;
              }
              stage_420_chroma_prev(
                chroma_prev,
                chroma_prev_row,
                row.u_half(),
                row.v_half(),
                idx,
                w,
              );
              if need_output && frozen_domain.is_none() {
                *frozen_domain = Some(AveragingDomain::Linear);
              }
              if need_output && frozen_chroma_centered.is_none() {
                *frozen_chroma_centered = Some(center_sited);
                *frozen_chroma_bottom_v = Some(bottom_v);
                *frozen_chroma_top_v = Some(top_v);
              }
              return Ok(());
            }
            // Dispatch first; commit the domain freeze to Linear ONLY when the
            // tail ACCEPTS an output-bearing row. `linear_light_resample`
            // returns Ok(()) without consuming for a no-output call and Err for
            // a filter / out-of-sequence / output-changed / alloc reject, so
            // `r.is_ok() && need_output` is exactly an accepted output-bearing
            // row — a rejected row leaves `frozen_domain` unset so a
            // corrected-domain retry of the SAME row is not falsely rejected.
            //
            // RFC #238 S3a — sited Linear must fold the correct chroma PHASE, not
            // silently co-site. For the reconstruct group (centered `Center` /
            // `Top` / `Bottom`, plus the co-sited-H + `v = 1` `BottomLeft`),
            // reconstruct full-width chroma (CODE domain) and feed the 4:4:4
            // decode (both the clamped u8 and the scene-referred f32 twin); the
            // fully co-sited group keeps today's byte-identical decode. The chroma
            // reserve runs AFTER the Linear preflight's first-row rejects (a filter
            // plan or an out-of-sequence first row) — so a rejected row allocates
            // nothing (#180); `linear_light_resample` re-runs the full preflight
            // before it commits. Gated on `want_color`: a luma-only Linear row
            // decodes no chroma, so it must not reserve / reconstruct.
            let r = if (center_sited || bottom_v) && want_color {
              linear_light::linear_light_preflight(
                linear_light_frame,
                resample_outputs,
                luma,
                luma_u16,
                rgb,
                rgba,
                hsv,
                tf,
                *linear_mode,
                plan,
                idx,
              )?;
              reserve_420_chroma_full(chroma_full, w, h)?;
              if bottom_v {
                reserve_420_chroma_prev(chroma_prev, w, h)?;
              }
              let (u_full, v_full) = upsample_420_chroma_sited(
                chroma_full,
                chroma_prev,
                chroma_prev_row,
                row.u_half(),
                row.v_half(),
                idx,
                bottom_v,
                center_sited,
                false,
                w,
              );
              linear_light::linear_light_resample(
                linear_light_frame,
                resample_outputs,
                rgb,
                rgba,
                luma,
                luma_u16,
                hsv,
                rgb_scratch,
                linear_scene_scratch,
                tf,
                *linear_mode,
                plan,
                row.y(),
                idx,
                w,
                h,
                use_simd,
                |_idx, dst| {
                  yuv_444_to_rgb_row(
                    row.y(),
                    u_full,
                    v_full,
                    dst,
                    w,
                    matrix,
                    full_range,
                    use_simd,
                  );
                },
                |_idx, dst| {
                  crate::row::scalar::yuv_444_to_rgb_f32_unclamped_row(
                    row.y(),
                    u_full,
                    v_full,
                    dst,
                    w,
                    matrix,
                    full_range,
                  );
                },
              )
            } else {
              linear_light::linear_light_resample(
                linear_light_frame,
                resample_outputs,
                rgb,
                rgba,
                luma,
                luma_u16,
                hsv,
                rgb_scratch,
                linear_scene_scratch,
                tf,
                *linear_mode,
                plan,
                row.y(),
                idx,
                w,
                h,
                use_simd,
                |_idx, dst| {
                  yuv_420_to_rgb_row(
                    row.y(),
                    row.u_half(),
                    row.v_half(),
                    dst,
                    w,
                    matrix,
                    full_range,
                    use_simd,
                  );
                },
                |_idx, dst| {
                  // Scene-referred: the SAME affine matrix, real-valued and
                  // unclamped (the 4:2:0 horizontal chroma upsample, as the
                  // Q15 `yuv_420_to_rgb_row` above does).
                  crate::row::scalar::yuv_420_to_rgb_f32_unclamped_row(
                    row.y(),
                    row.u_half(),
                    row.v_half(),
                    dst,
                    w,
                    matrix,
                    full_range,
                  );
                },
              )
            };
            // Bottom lookback: advance it only AFTER the Linear resample
            // accepts the row (mirroring the #180 chroma reserve ordering), so a
            // rejected final row leaves the predecessor in place for a clean
            // retry — the sited reconstruction read the lookback above but did
            // not stage. A luma-only Bottom row took the else arm (no
            // reconstruction), so gate on `want_color`.
            if r.is_ok() && bottom_v && want_color {
              stage_420_chroma_prev(
                chroma_prev,
                chroma_prev_row,
                row.u_half(),
                row.v_half(),
                idx,
                w,
              );
            }
            if r.is_ok() && need_output && frozen_domain.is_none() {
              *frozen_domain = Some(AveragingDomain::Linear);
            }
            // RFC #238 S3a: freeze the siting on the same accepted output-bearing
            // row as the domain (never on a reject, so a corrected retry is not
            // falsely rejected as a mid-frame change).
            if r.is_ok() && need_output && frozen_chroma_centered.is_none() {
              *frozen_chroma_centered = Some(center_sited);
              *frozen_chroma_bottom_v = Some(bottom_v);
              *frozen_chroma_top_v = Some(top_v);
            }
            return r;
          }
          #[cfg(not(feature = "rgb"))]
          {
            return Err(plan.linear_domain_unsupported().into());
          }
        }
        // `Yuv420p` has no alpha plane, so premultiplied weighting is a category
        // error here — reject with a typed error rather than silently running
        // the encoded average. Phase 5 wires Premultiplied for actual alpha
        // formats; on these non-alpha formats rejecting is correct.
        AveragingDomain::Premultiplied => {
          return Err(plan.premultiplied_domain_unsupported().into());
        }
      }
      // A `Filter` plan routes to the filter resampler, which converts the
      // separate Y/U/V planes to a source-width RGB row (the same
      // `yuv_420_to_rgb_row` kernel the row-stage tier uses) and
      // filter-resamples it plus the native Y. The native fast tier is an
      // area-specific optimization, so it never sees a filter plan; the
      // per-sink plan kind is fixed at construction, so a filter sink
      // bypasses the native/row-stage route machinery entirely (no
      // `frozen_native_route` interaction).
      if plan.kind().is_filter() {
        // A BICUBLIN plan ([`Bicublin`](crate::resample::Bicublin)) carries a
        // SECOND (chroma) window set, so it cannot route through the
        // single-kernel `planar_dual_filter_resample` (which filters one
        // converted RGB row). It instead filters the Y / U / V planes
        // SEPARATELY in plane space (cubic luma, linear chroma) and converts
        // the filtered planes — the filter twin of the native area tier. It does
        // not carry the RFC #238 chroma phase (folding the centered siting into
        // its separate chroma window is a later PR), so a bicublin sink stays
        // siting-neutral: it neither reconstructs centered chroma nor freezes the
        // phase, and its output is byte-unchanged.
        if plan.is_bicublin() {
          return bicublin_yuv420p_resample(
            plan,
            bicublin_420,
            resample_outputs,
            rgb,
            rgba,
            luma,
            luma_u16,
            hsv,
            rgb_scratch,
            row.y(),
            row.u_half(),
            row.v_half(),
            matrix,
            full_range,
            idx,
            w,
            h,
            use_simd,
          );
        }
        if top_v && want_color {
          // RFC #238 Top (`v = 0`) FORWARD one-row delay for the single-kernel
          // filter tier — the exact structure of the row-stage delay below, only
          // the resampler kind differs (`FilterStream`, driven by
          // `planar_dual_filter_resample`). The whole odd row is HELD and the
          // following even row filter-feeds TWO source rows (held forward-blend +
          // current co-sited); the trailing odd row clamps to a co-sited decode.
          // `FilterStream::feed_row` allocates nothing, so once row 0 has built
          // the streams + scratch the two feeds are infallible; reserving every
          // buffer up front makes the two-row flush retry-atomic (#180).
          let is_last = idx + 1 == h;
          let need_luma = luma.is_some() || luma_u16.is_some();
          let cursor = if need_luma {
            luma_filter_stream.as_ref().map_or(0, |s| s.next_y())
          } else {
            rgb_filter_stream.as_ref().map_or(0, |s| s.next_y())
          };
          let expected_source = cursor + usize::from(chroma_top_pending.is_some());
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
            Some(expected_source),
            idx,
          )? {
            return Ok(());
          }
          reserve_420_chroma_full(chroma_full, w, h)?;
          reserve_420_chroma_prev(chroma_prev, w, h)?;
          if chroma_top_y.len() < w {
            chroma_top_y
              .try_reserve_exact(w - chroma_top_y.len())
              .map_err(|_| {
                MixedSinkerError::Resample(ResampleError::AllocationFailed(PlanGeometry::new(
                  w, h, w, h,
                )))
              })?;
            chroma_top_y.resize(w, 0);
          }
          if idx & 1 == 0 {
            if let Some((p_idx, p_matrix, p_full_range)) = chroma_top_pending.take() {
              let (u_full, v_full) = upsample_420_chroma_sited(
                chroma_full,
                chroma_prev,
                chroma_prev_row,
                row.u_half(),
                row.v_half(),
                idx,
                true,
                center_sited,
                false,
                w,
              );
              planar_dual_filter_resample(
                luma_filter_stream,
                rgb_filter_stream,
                resample_outputs,
                rgb,
                rgba,
                luma,
                luma_u16,
                hsv,
                rgb_scratch,
                &chroma_top_y[..w],
                w,
                plan,
                p_idx,
                use_simd,
                |scratch| {
                  yuv_444_to_rgb_row(
                    &chroma_top_y[..w],
                    u_full,
                    v_full,
                    scratch,
                    w,
                    p_matrix,
                    p_full_range,
                    use_simd,
                  );
                },
              )?;
            }
            let (u_full, v_full) = upsample_420_chroma_sited(
              chroma_full,
              chroma_prev,
              chroma_prev_row,
              row.u_half(),
              row.v_half(),
              idx,
              false,
              center_sited,
              false,
              w,
            );
            planar_dual_filter_resample(
              luma_filter_stream,
              rgb_filter_stream,
              resample_outputs,
              rgb,
              rgba,
              luma,
              luma_u16,
              hsv,
              rgb_scratch,
              row.y(),
              w,
              plan,
              idx,
              use_simd,
              |scratch| {
                yuv_444_to_rgb_row(
                  row.y(),
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
          } else if !is_last {
            chroma_top_y[..w].copy_from_slice(&row.y()[..w]);
            *chroma_top_pending = Some((idx, matrix, full_range));
          } else {
            let (u_full, v_full) = upsample_420_chroma_sited(
              chroma_full,
              chroma_prev,
              chroma_prev_row,
              row.u_half(),
              row.v_half(),
              idx,
              false,
              center_sited,
              false,
              w,
            );
            planar_dual_filter_resample(
              luma_filter_stream,
              rgb_filter_stream,
              resample_outputs,
              rgb,
              rgba,
              luma,
              luma_u16,
              hsv,
              rgb_scratch,
              row.y(),
              w,
              plan,
              idx,
              use_simd,
              |scratch| {
                yuv_444_to_rgb_row(
                  row.y(),
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
          }
          stage_420_chroma_prev(
            chroma_prev,
            chroma_prev_row,
            row.u_half(),
            row.v_half(),
            idx,
            w,
          );
          if need_output && frozen_chroma_centered.is_none() {
            *frozen_chroma_centered = Some(center_sited);
            *frozen_chroma_bottom_v = Some(bottom_v);
            *frozen_chroma_top_v = Some(top_v);
          }
          return Ok(());
        }
        if (center_sited || bottom_v) && want_color {
          // Sited single-kernel filter: reconstruct full-width chroma (centered
          // `Center` / `Top` / `Bottom`, or co-sited-H + `v = 1` `BottomLeft`),
          // but ONLY after the resample preflight (frozen-output + sequence), so
          // an out-of-sequence / rejected row is caught before the chroma
          // reservation (#180). `planar_dual_filter_resample` re-runs the
          // idempotent preflight. Gated by `want_color` — a luma-only sited row
          // never calls the RGB converter, so it stays on the co-sited arm.
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
          if bottom_v {
            reserve_420_chroma_prev(chroma_prev, w, h)?;
          }
          let (u_full, v_full) = upsample_420_chroma_sited(
            chroma_full,
            chroma_prev,
            chroma_prev_row,
            row.u_half(),
            row.v_half(),
            idx,
            bottom_v,
            center_sited,
            false,
            w,
          );
          let r = planar_dual_filter_resample(
            luma_filter_stream,
            rgb_filter_stream,
            resample_outputs,
            rgb,
            rgba,
            luma,
            luma_u16,
            hsv,
            rgb_scratch,
            row.y(),
            w,
            plan,
            idx,
            use_simd,
            |scratch| {
              yuv_444_to_rgb_row(
                row.y(),
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
          // Bottom lookback: advance only AFTER the filter resample accepts the
          // row, so a rejected row leaves the predecessor for a clean retry (the
          // sited reconstruction read it above but did not stage).
          if r.is_ok() && bottom_v {
            stage_420_chroma_prev(
              chroma_prev,
              chroma_prev_row,
              row.u_half(),
              row.v_half(),
              idx,
              w,
            );
          }
          // Filter tier freezes the siting on the same accepted output-bearing
          // row as the native / row-stage arms (a rejected row leaves it unset
          // for a corrected retry).
          if r.is_ok() && need_output && frozen_chroma_centered.is_none() {
            *frozen_chroma_centered = Some(center_sited);
            *frozen_chroma_bottom_v = Some(bottom_v);
            *frozen_chroma_top_v = Some(top_v);
          }
          return r;
        }
        let r = planar_dual_filter_resample(
          luma_filter_stream,
          rgb_filter_stream,
          resample_outputs,
          rgb,
          rgba,
          luma,
          luma_u16,
          hsv,
          rgb_scratch,
          row.y(),
          w,
          plan,
          idx,
          use_simd,
          |scratch| {
            yuv_420_to_rgb_row(
              row.y(),
              row.u_half(),
              row.v_half(),
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
          *frozen_chroma_bottom_v = Some(bottom_v);
          *frozen_chroma_top_v = Some(top_v);
        }
        return r;
      }
      // Reject a mid-frame native/row-stage route flip BEFORE either tier's
      // dispatch. The two tiers carry independent, in-order, once-only
      // stream state, so splitting a frame across them yields a
      // mixed/partial frame rather than a deterministic rejection. The route
      // is both CHECKED here and frozen below (the SET) ONLY on an
      // output-bearing row a tier ACCEPTS — both gate on `need_output`. A
      // no-output call therefore neither checks nor freezes the route: it is
      // a true no-op, route-invisible regardless of row index, so it can
      // never spuriously trip `NativeRouteChanged` after the route is
      // frozen. A preflight-rejected (out-of-sequence / frozen)
      // output-bearing call returns Err before the SET, so it leaves
      // `frozen_native_route` untouched and a later same-or-other-route
      // retry is not falsely rejected.
      if need_output
        && let Some(frozen) = *frozen_native_route
        && frozen != *native
      {
        return Err(MixedSinkerError::NativeRouteChanged(
          NativeRouteChanged::new(idx),
        ));
      }
      // RFC #238 splice-stage selection. The native-vs-row-stage choice is
      // re-expressed through the framework selector: for the encoded domain
      // it returns `NativeCodes` exactly when this format is native-eligible,
      // the sink enabled the native tier (`*native`), and the plan is an area
      // downscale (a filter plan already returned above, so `area_plan` is
      // always true here). That reproduces the former `if *native` boolean
      // bit-for-bit — the route-freeze and rejection above are unchanged, so
      // the dispatch stays byte-identical. The native tier is
      // [`InsertionPoint::NativeCodes`] (bin codes, then convert); the
      // row-stage tier is [`InsertionPoint::EncodedOutput`] (convert, then
      // area-stream the output).
      let insertion = select_insertion_point(
        AveragingDomain::Encoded,
        InsertionContext {
          native_eligible: YUV420P_NATIVE_ELIGIBLE,
          with_native: *native,
          area_plan: true,
        },
      );
      match insertion {
        InsertionPoint::NativeCodes => {
          // Dispatch first; freeze the route to native ONLY after the call
          // returns Ok on an output-bearing row. A no-output call returns
          // Ok(()) with `need_output` false (no freeze); an out-of-sequence /
          // frozen row returns Err via `?` (no freeze) — so only an accepted
          // output-bearing row commits the route.
          //
          // RFC #238 S3a point-of-use siting invalidation, mirroring the Yuv422p
          // native arm: `chroma_location` can change at ANY point before this row
          // (including AFTER `begin_frame`, before row 0), so re-check the cached
          // join HERE and drop it when its folded chroma plan was built for a
          // different phase; `yuv420p_process_native` then rebuilds it with the
          // current siting. Retry-atomic: drop ONLY on the IN-SEQUENCE fresh-frame
          // first row (`idx == 0`, `next_y() == 0`); an out-of-sequence first row
          // after a siting change is left for the delegate to reject against the
          // INTACT join, so a corrected retry of row 0 rebuilds cleanly. A
          // luma-only join carries no chroma plan (siting-independent), a
          // no-output sink built no join. Transactional: move the stale join OUT
          // and let the delegate build the replacement into `native_420` (its
          // build runs BEFORE it inserts, so a build failure leaves the field
          // `None`); on such a failure, restore the intact prior-phase join so the
          // REJECTED row mutates nothing — state-atomic under allocation failure.
          let stale_native = idx == 0
            && native_420.as_ref().is_some_and(|join| {
              join.chroma.is_some()
                && (join.chroma_centered != center_sited
                  || join.chroma_bottom != bottom_v
                  || join.chroma_top != top_v)
                && join.y.next_y() == 0
            });
          let prev_native = if stale_native {
            native_420.take()
          } else {
            None
          };
          let native_result = yuv420p_process_native(
            plan,
            native_420,
            resample_outputs,
            rgb,
            rgba,
            luma,
            luma_u16,
            hsv,
            rgb_scratch,
            row.y(),
            row.u_half(),
            row.v_half(),
            matrix,
            full_range,
            idx,
            w,
            h,
            || {
              ResamplePlan::area_chroma_420(
                w / 2,
                h,
                plan.out_w(),
                plan.out_h(),
                chroma_h_phase,
                chroma_v_phase,
                top_v,
              )
            },
            use_simd,
          );
          // Restore the taken stale-phase join if the delegate's rebuild was
          // rejected at any pre-feed step: it leaves the field `None` and
          // `resample_outputs` uncommitted on such a failure, so restoring the
          // intact prior-phase join leaves the rejected row mutating nothing. A
          // non-stale row (first-ever / colour-capability rebuild) took nothing.
          if stale_native && native_result.is_err() {
            *native_420 = prev_native;
          }
          native_result?;
          if frozen_native_route.is_none() && need_output {
            *frozen_native_route = Some(true);
          }
          // Encoded domain committed alongside the route, on the same accepted
          // output-bearing row (the `?` above already returned any reject).
          if frozen_domain.is_none() && need_output {
            *frozen_domain = Some(AveragingDomain::Encoded);
          }
          // RFC #238 S3a: freeze the siting on the same accepted output row.
          if frozen_chroma_centered.is_none() && need_output {
            *frozen_chroma_centered = Some(center_sited);
            *frozen_chroma_bottom_v = Some(bottom_v);
            *frozen_chroma_top_v = Some(top_v);
          }
          return Ok(());
        }
        InsertionPoint::EncodedOutput => {
          // Row-stage tail. Same CHECK-before / SET-after split: dispatch,
          // then freeze the route to row-stage only when the call accepts an
          // output-bearing row (a no-output call returns Ok with
          // `need_output` false; an out-of-sequence / frozen row returns Err
          // via `?`).
          if hsv.is_some() && rgb.is_none() && rgba.is_none() {
            // HSV-only (no RGB / RGBA): bin Y / U / V and convert per output row,
            // RGB-free — bit-identical to the native fast tier. 4:2:0 chroma is
            // half-width, ceil-half-height (`chroma_vsub = 2`); the folded chroma
            // plan carries the phase.
            //
            // RFC #238 S3a point-of-use siting invalidation (see the native arm):
            // drop the HSV-only join if its cached chroma plan was built for a
            // different phase than the current `chroma_location`, so
            // `hsv_direct_resample` rebuilds it with the current siting.
            // Retry-atomic (drop only on the in-sequence fresh-frame first row)
            // and transactional (move the stale join OUT, let the delegate build
            // the replacement, and restore the intact prior-phase join on a
            // rejected rebuild so the REJECTED row mutates nothing).
            let stale_hsv = idx == 0
              && hsv_planar.as_ref().is_some_and(|join| {
                (join.chroma_centered() != center_sited
                  || join.chroma_bottom() != bottom_v
                  || join.chroma_top() != top_v)
                  && join.next_y() == 0
              });
            let prev_hsv = if stale_hsv { hsv_planar.take() } else { None };
            let hsv_result = super::planar_resample::hsv_direct_resample(
              hsv_planar,
              resample_outputs,
              luma,
              luma_u16,
              hsv,
              row.y(),
              row.u_half(),
              row.v_half(),
              matrix,
              full_range,
              2,
              || {
                ResamplePlan::area_chroma_420(
                  w / 2,
                  h,
                  plan.out_w(),
                  plan.out_h(),
                  chroma_h_phase,
                  chroma_v_phase,
                  top_v,
                )
              },
              w,
              plan,
              idx,
              use_simd,
            );
            if stale_hsv && hsv_result.is_err() {
              *hsv_planar = prev_hsv;
            }
            hsv_result?;
          } else if top_v && want_color {
            // RFC #238 Top (`v = 0`) FORWARD one-row delay — the mirror of
            // `Bottom`'s backward `chroma_prev` lookback. A `Top` chroma sample is
            // co-sited with the TOP luma row of its pair, so an EVEN source row
            // decodes co-sited chroma while an ODD source row needs the vertical
            // box-average of its chroma row and the NEXT one, which a
            // row-at-a-time stream has not been fed when the odd row arrives. Luma
            // rides the SAME dual-stream feed (siting-independent), so the whole
            // odd row is HELD (`chroma_top_pending` + the buffered `chroma_top_y`)
            // and the following even row feeds TWO source rows: the held odd row
            // (forward-blended through the `Bottom`-EVEN `(prev + cur) >> 1`
            // kernels at THIS even index) then the current even row (co-sited).
            // The trailing odd row of the frame has no next chroma, so it clamps
            // to a co-sited decode. `feed_row` allocates nothing, so once row 0
            // has built the streams + scratch the two feeds are infallible;
            // reserving every buffer up front therefore makes the two-row flush
            // retry-atomic — a reservation refusal returns before any feed,
            // leaving the stream cursors and the held row untouched (#180).
            let is_last = idx + 1 == h;
            let need_luma = luma.is_some() || luma_u16.is_some();
            // The stream cursor lags the source by the buffered (unfed) held odd
            // row, so the next expected SOURCE row is `cursor + pending`. Reject an
            // out-of-sequence walker row (and a mid-frame output change) BEFORE
            // reserving any chroma (#180).
            let cursor = if need_luma {
              luma_stream.as_ref().map_or(0, |s| s.next_y())
            } else {
              rgb_stream.as_ref().map_or(0, |s| s.next_y())
            };
            let expected_source = cursor + usize::from(chroma_top_pending.is_some());
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
              Some(expected_source),
              idx,
            )? {
              return Ok(());
            }
            // Reserve every buffer BOTH feeds may touch, up front (#180). After
            // row 0 these are no-ops, so the even-row double feed does no
            // allocation and both feeds are infallible.
            reserve_420_chroma_full(chroma_full, w, h)?;
            reserve_420_chroma_prev(chroma_prev, w, h)?;
            if chroma_top_y.len() < w {
              chroma_top_y
                .try_reserve_exact(w - chroma_top_y.len())
                .map_err(|_| {
                  MixedSinkerError::Resample(ResampleError::AllocationFailed(PlanGeometry::new(
                    w, h, w, h,
                  )))
                })?;
              chroma_top_y.resize(w, 0);
            }
            if idx & 1 == 0 {
              // Flush the held odd predecessor (`p_idx = idx - 1`) FIRST,
              // forward-blending the previous chroma row (`chroma_prev`) with the
              // current one at this even index.
              if let Some((p_idx, p_matrix, p_full_range)) = chroma_top_pending.take() {
                let (u_full, v_full) = upsample_420_chroma_sited(
                  chroma_full,
                  chroma_prev,
                  chroma_prev_row,
                  row.u_half(),
                  row.v_half(),
                  idx,
                  true,
                  center_sited,
                  false,
                  w,
                );
                planar_dual_resample(
                  luma_stream,
                  rgb_stream,
                  resample_outputs,
                  rgb,
                  rgba,
                  luma,
                  luma_u16,
                  hsv,
                  rgb_scratch,
                  &chroma_top_y[..w],
                  w,
                  plan,
                  p_idx,
                  use_simd,
                  |scratch| {
                    yuv_444_to_rgb_row(
                      &chroma_top_y[..w],
                      u_full,
                      v_full,
                      scratch,
                      w,
                      p_matrix,
                      p_full_range,
                      use_simd,
                    );
                  },
                )?;
              }
              // Then feed the current EVEN row (co-sited chroma).
              let (u_full, v_full) = upsample_420_chroma_sited(
                chroma_full,
                chroma_prev,
                chroma_prev_row,
                row.u_half(),
                row.v_half(),
                idx,
                false,
                center_sited,
                false,
                w,
              );
              planar_dual_resample(
                luma_stream,
                rgb_stream,
                resample_outputs,
                rgb,
                rgba,
                luma,
                luma_u16,
                hsv,
                rgb_scratch,
                row.y(),
                w,
                plan,
                idx,
                use_simd,
                |scratch| {
                  yuv_444_to_rgb_row(
                    row.y(),
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
            } else if !is_last {
              // Defer this odd row: copy its Y (the borrow expires after
              // `process`) + record its decode params; it emits at the next even
              // row.
              chroma_top_y[..w].copy_from_slice(&row.y()[..w]);
              *chroma_top_pending = Some((idx, matrix, full_range));
            } else {
              // Trailing odd row (bottom-edge clamp → co-sited): feed it now.
              let (u_full, v_full) = upsample_420_chroma_sited(
                chroma_full,
                chroma_prev,
                chroma_prev_row,
                row.u_half(),
                row.v_half(),
                idx,
                false,
                center_sited,
                false,
                w,
              );
              planar_dual_resample(
                luma_stream,
                rgb_stream,
                resample_outputs,
                rgb,
                rgba,
                luma,
                luma_u16,
                hsv,
                rgb_scratch,
                row.y(),
                w,
                plan,
                idx,
                use_simd,
                |scratch| {
                  yuv_444_to_rgb_row(
                    row.y(),
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
            }
            // Refresh the lookback with the current chroma row (after the reads
            // above) so the NEXT even row's flush forward-blends this row as its
            // predecessor.
            stage_420_chroma_prev(
              chroma_prev,
              chroma_prev_row,
              row.u_half(),
              row.v_half(),
              idx,
              w,
            );
          } else if (center_sited || bottom_v) && want_color {
            // Sited RGB/RGBA: reconstruct full-width chroma (centered `Center` /
            // `Top` / `Bottom`, or co-sited-H + `v = 1` `BottomLeft`) AFTER the
            // resample preflight (frozen-output + sequence), so an out-of-sequence
            // / rejected row is caught before the chroma reservation (#180). Gated
            // by `want_color` — a luma-only sited row never calls the RGB
            // converter, so it falls to the co-sited arm (which only bins luma).
            // `planar_dual_resample` re-runs the idempotent preflight.
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
            if bottom_v {
              reserve_420_chroma_prev(chroma_prev, w, h)?;
            }
            let (u_full, v_full) = upsample_420_chroma_sited(
              chroma_full,
              chroma_prev,
              chroma_prev_row,
              row.u_half(),
              row.v_half(),
              idx,
              bottom_v,
              center_sited,
              false,
              w,
            );
            planar_dual_resample(
              luma_stream,
              rgb_stream,
              resample_outputs,
              rgb,
              rgba,
              luma,
              luma_u16,
              hsv,
              rgb_scratch,
              row.y(),
              w,
              plan,
              idx,
              use_simd,
              |scratch| {
                yuv_444_to_rgb_row(
                  row.y(),
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
            // Bottom lookback: advance only after the row-stage resample
            // accepts the row (the `?` above already returned any reject), so a
            // rejected row leaves the predecessor for a clean retry — the sited
            // reconstruction read it above but did not stage.
            if bottom_v {
              stage_420_chroma_prev(
                chroma_prev,
                chroma_prev_row,
                row.u_half(),
                row.v_half(),
                idx,
                w,
              );
            }
          } else {
            // Co-sited RGB / RGBA / luma: the byte-identical row-stage tail.
            yuv420p_process_resampled(
              plan,
              rgb_stream,
              luma_stream,
              resample_outputs,
              rgb,
              rgba,
              luma,
              luma_u16,
              hsv,
              rgb_scratch,
              row.y(),
              row.u_half(),
              row.v_half(),
              matrix,
              full_range,
              idx,
              w,
              use_simd,
            )?;
          }
          if frozen_native_route.is_none() && need_output {
            *frozen_native_route = Some(false);
          }
          // Encoded domain committed alongside the route, on the same accepted
          // output-bearing row (the `?` above already returned any reject).
          if frozen_domain.is_none() && need_output {
            *frozen_domain = Some(AveragingDomain::Encoded);
          }
          // RFC #238 S3a: freeze the siting on the same accepted output row.
          if frozen_chroma_centered.is_none() && need_output {
            *frozen_chroma_centered = Some(center_sited);
            *frozen_chroma_bottom_v = Some(bottom_v);
            *frozen_chroma_top_v = Some(top_v);
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

    // Output mode resolution (Strategy A):
    // - RGBA-only: run dedicated `yuv_420_to_rgba_row` (4 bpp store).
    // - RGB / HSV (with or without RGBA): run RGB kernel once, then if
    //   RGBA is also requested, fan it out via `expand_rgb_to_rgba_row`
    //   (memory-bound copy + 0xFF alpha pad). Saves the second YUV→RGB
    //   per-pixel math when both buffers are attached.
    // - None of the above: nothing to do beyond luma above.
    //
    // Computed BEFORE any per-row offset arithmetic so the no-output guard
    // below short-circuits a no-output `process` call before it runs ANY row
    // math (`need_output` depends only on which output buffers are attached,
    // never on `idx` / `w`).
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();
    // Whether this row produces any colour output (and so runs the centered /
    // bottom-sited chroma upsample). A bottom-sited row maintains the vertical
    // lookback even when it produces only luma (luma-only / `luma_u16`-only) —
    // see the staging below — so `want_color` gates only the colour scratches.
    let want_color = want_rgb || want_rgba || want_hsv;

    // Repo-wide no-output invariant: a `process` call carrying NO output — no
    // colour, no luma, no luma_u16 — must run NOTHING: no per-row offset
    // arithmetic, no allocation, no state mutation (the bottom-sited vertical
    // lookback included). Returning HERE, before the `idx * w` offsets below,
    // is also what makes the invariant overflow-safe: a no-output call never ran
    // an attach-time `w x h x 1` validation (nothing was attached), so `idx * w`
    // could overflow `usize` on a 32-bit target with absurd geometry — the guard
    // skips that math entirely. It also means such a row never reserves
    // `chroma_prev` (no spurious `AllocationFailed`) nor primes the lookback (so
    // it can never make a later colour even row box-blend through an invisible,
    // never-output row). The identity path's siting freeze and Top forward-delay
    // state are all output-gated below (a no-output row neither checks nor sets
    // them), so this early return is a pure no-op that just makes the invariant
    // explicit — mirroring the resample path's `need_output` short-circuit.
    let need_output = want_color || luma.is_some() || luma_u16.is_some();
    if !need_output {
      return Ok(());
    }

    // Single-plane row ranges are guaranteed not to overflow: the no-output
    // guard above ensures ≥ 1 output (luma / luma_u16 / colour) is attached, so
    // its attach-time `w x h x 1` validation ran and `w x h` fits `usize`; with
    // `idx < h` that makes `(idx + 1) * w ≤ h * w` fit too. (A no-output call —
    // which never ran that validation — returned above before reaching here.)
    // The `x 3` RGB ranges are only needed when RGB output is requested —
    // computed lazily below with overflow checking.
    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // Chroma siting (#302): the centered horizontal sitings reconstruct
    // chroma at the phase-0.5 position; the default / co-sited path keeps the
    // byte-identical nearest-neighbor decode.
    let center_sited = chroma_420_center_sited_h(chroma_location);
    // Bottom-sited vertical phase (#302): `ChromaLocation::Bottom` (a strict
    // sub-case of `center_sited`, so it always rides the centered staging below)
    // additionally box-blends the EVEN output row's chroma with the PREVIOUS
    // chroma row (a backward one-row lookback).
    let bottom_v = chroma_420_bottom_sited_v(chroma_location);
    // Top-sited vertical phase (RFC #238): `Top` / `TopLeft` (`v = 0`) box-blend
    // the ODD output row's chroma with the NEXT chroma row — which a row-at-a-time
    // stream has not been fed when the odd row arrives — so the identity colour
    // decode holds the odd row's output in a FORWARD one-row delay
    // (`chroma_top_pending` / `chroma_top_y`) and emits it once the following even
    // row supplies the next chroma. Handled in its own branch below; the luma
    // planes are siting-independent and written in order regardless.
    let top_v = chroma_420_top_sited_v(chroma_location);

    // RFC #238 Top: freeze the effective 4:2:0 chroma siting on the direct
    // (identity) path too, mirroring the resample tiers' choke-point CHECK above
    // and the 4:4:0 sibling. `set_chroma_location` is public, so without this a
    // caller could hold an odd `Top` row (buffered in `chroma_top_pending` /
    // `chroma_top_y`), flip to `Center` / `Left` — whose `top_v == false` skips
    // the flush/defer branch below, so the held row is neither emitted nor
    // cleared — then flip back to `Top` and flush that STALE row against the
    // wrong chroma row (silent RGB / RGBA / HSV corruption). Reject any mid-frame
    // flip TO or FROM `Top` / `TopLeft` (or any horizontal / bottom phase flip)
    // with the typed `ChromaSitingChanged`. CHECK here — BEFORE the atomicity
    // preflight and any output write or delay-state mutation — so a rejected flip
    // leaves `chroma_top_pending` / `chroma_top_y` UNTOUCHED and the row is
    // retry-atomic (#180); the matching SET rides the row's accept point below,
    // after the fallible preflight. `need_output` is already established (the
    // no-output guard returned above).
    if let Some(frozen) = *frozen_chroma_centered
      && (frozen != center_sited
        || *frozen_chroma_bottom_v != Some(bottom_v)
        || *frozen_chroma_top_v != Some(top_v))
    {
      return Err(MixedSinkerError::ChromaSitingChanged(
        ChromaSitingChanged::new(idx),
      ));
    }

    // Atomicity preflight (#302, cf. the crate's #180 resample fix): reserve
    // EVERY fallible row scratch this `Yuv420p` row needs BEFORE any output
    // row (luma / luma_u16 included) is written, so an allocator refusal
    // returns a typed error leaving the output frame untouched, never
    // partially mutated. Three scratches can grow:
    //  1. the centered-siting full-width chroma (`chroma_full`), only on a
    //     colour decode;
    //  2. the bottom-sited vertical-phase one-row chroma lookback
    //     (`chroma_prev`), on every bottom-sited OUTPUT row — colour OR
    //     luma-only — because the lookback is maintained so a later colour row
    //     can box-blend it (the luma-only staging runs below, before any luma
    //     write). A no-output row returned early above, so it never reaches here
    //     and never primes the lookback; and
    //  3. the RGB row scratch, when a colour decode needs an RGB row but no
    //     caller RGB buffer is borrowable — exactly `want_hsv && want_rgba &&
    //     !want_rgb` (`rgb_row_buf_or_scratch`'s own scratch arm; with an
    //     attached RGB buffer it borrows that instead and never allocates).
    // The later `upsample_420_chroma_sited` / `rgb_row_buf_or_scratch`
    // calls then reuse the already-sized buffers. NOTE: the identical
    // pre-existing RGB-scratch gap in the sibling planar formats
    // (Yuv410p / Yuv422p / Yuv444p / Yuv440p / Yuv411p) is NOT a #302
    // regression and is tracked for a dedicated crate-wide fix in #308; this
    // PR fixes only the `Yuv420p` instance it owns.
    if (center_sited || bottom_v || top_v) && want_color {
      reserve_420_chroma_full(chroma_full, w, h)?;
    }
    if bottom_v || top_v {
      reserve_420_chroma_prev(chroma_prev, w, h)?;
    }
    // Top forward-delay Y buffer (RFC #238): reserved up front — BEFORE any luma
    // write — so the later deferred-row copy is infallible and an allocator
    // refusal leaves the output frame untouched (the #180 atomicity contract).
    if top_v && want_color && chroma_top_y.len() < w {
      chroma_top_y
        .try_reserve_exact(w - chroma_top_y.len())
        .map_err(|_| {
          MixedSinkerError::Resample(ResampleError::AllocationFailed(PlanGeometry::new(
            w, h, w, h,
          )))
        })?;
      chroma_top_y.resize(w, 0);
    }
    if want_hsv && want_rgba && !want_rgb {
      // Grow the RGB scratch up front via the same helper the decode uses, so
      // its (test-injectable) allocation runs here — before luma — and the
      // later call is a no-op reserve. `rgb` is `None` in this arm, so this
      // takes the scratch branch and returns a slice we discard.
      rgb_row_buf_or_scratch(
        rgb.as_deref_mut(),
        rgb_scratch,
        one_plane_start,
        one_plane_end,
        w,
        h,
      )?;
    }

    // Bottom / Top sited LUMA-ONLY row (#302 / RFC #238): the colour upsample
    // helper — which normally refreshes the vertical lookback — won't run, so
    // stage the current chroma row into the lookback HERE, after its preflight
    // reservation above and BEFORE any luma write, so a later colour row in the
    // same frame can box-blend it (a luma-only-then-colour in-order sequence then
    // reconstructs the same vertical phase as the all-output walk). A colour row
    // instead stages after reading the previous lookback, so this is skipped for
    // it; a NO-output row returned early above, so `!want_color` here means a
    // genuine luma-only / luma_u16 row. The validity tag (`chroma_prev_row`)
    // still guards out-of-sequence / cross-frame reads (→ clamp, never stale).
    if (bottom_v || top_v) && !want_color {
      stage_420_chroma_prev(
        chroma_prev,
        chroma_prev_row,
        row.u_half(),
        row.v_half(),
        idx,
        w,
      );
    }

    // RFC #238 Top: the row is now past its fallible preflight — every scratch is
    // reserved, so the remaining writes and the Top forward-delay bookkeeping
    // cannot fail. Commit the siting freeze here (mirroring the resample arms'
    // accept-time SET and the 4:4:0 sibling), at the single point all output
    // sub-paths pass through before diverging. Only the frame's first
    // output-bearing row sets it; the CHECK above rejects any later phase flip.
    if frozen_chroma_centered.is_none() {
      *frozen_chroma_centered = Some(center_sited);
      *frozen_chroma_bottom_v = Some(bottom_v);
      *frozen_chroma_top_v = Some(top_v);
    }

    // Luma — YUV420p luma *is* the Y plane. Just copy.
    if let Some(luma) = luma.as_deref_mut() {
      luma[one_plane_start..one_plane_end].copy_from_slice(&row.y()[..w]);
    }

    // Luma u16 — zero-extend the 8-bit Y plane into u16.
    if let Some(buf) = luma_u16.as_deref_mut() {
      crate::row::y_plane_to_luma_u16_row(
        row.y(),
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    // RFC #238 Top (`v = 0`) forward one-row delay for the identity colour
    // decode. A `Top` chroma sample is co-sited with the TOP luma row of its
    // pair, so an EVEN output row is a plain co-sited decode while an ODD output
    // row needs the vertical box-average of the current chroma row and the NEXT
    // one — unavailable until the following even row arrives. Each odd colour row
    // is therefore HELD (`chroma_top_pending` + the buffered `chroma_top_y`) and
    // emitted at the following even row, box-blended through the SAME
    // `(prev + cur) >> 1` fold + fused-`h` kernels a `Bottom` EVEN row uses —
    // reused verbatim via `upsample_420_chroma_sited(bottom_v = true)` at the
    // even row's index, whose lookback validity check (`chroma_prev` holds
    // `chroma_row - 1`) is exactly what the held odd row wants. The trailing odd
    // row of the frame has no next chroma, so it clamps to a co-sited decode.
    // Luma was already written above (siting-independent); only colour is delayed.
    if top_v && want_color {
      let is_last = idx + 1 == h;
      // Flush the held odd predecessor (`idx - 1`) at this even row, box-blending
      // the previous chroma row (`chroma_prev`) with the current one.
      if idx & 1 == 0
        && let Some((p_idx, p_matrix, p_full_range)) = chroma_top_pending.take()
      {
        let (u_full, v_full) = upsample_420_chroma_sited(
          chroma_full,
          chroma_prev,
          chroma_prev_row,
          row.u_half(),
          row.v_half(),
          idx,
          true,
          center_sited,
          false,
          w,
        );
        yuv444_identity_color_row(
          rgb,
          rgba,
          hsv,
          rgb_scratch,
          &chroma_top_y[..w],
          u_full,
          v_full,
          p_idx * w,
          p_idx * w + w,
          w,
          h,
          p_matrix,
          primaries,
          p_full_range,
          use_simd,
        )?;
      }
      if idx & 1 == 1 && !is_last {
        // Defer this odd row: copy its Y (the row borrow expires after `process`)
        // and record its decode parameters; its colour emits at the next even row.
        chroma_top_y[..w].copy_from_slice(&row.y()[..w]);
        *chroma_top_pending = Some((idx, row.matrix(), row.full_range()));
      } else {
        // Even row (co-sited `c[i]`) or the trailing odd row (bottom-edge clamp →
        // co-sited): a plain co-sited decode of the current chroma row.
        let (u_full, v_full) = upsample_420_chroma_sited(
          chroma_full,
          chroma_prev,
          chroma_prev_row,
          row.u_half(),
          row.v_half(),
          idx,
          false,
          center_sited,
          false,
          w,
        );
        yuv444_identity_color_row(
          rgb,
          rgba,
          hsv,
          rgb_scratch,
          row.y(),
          u_full,
          v_full,
          one_plane_start,
          one_plane_end,
          w,
          h,
          row.matrix(),
          primaries,
          row.full_range(),
          use_simd,
        )?;
      }
      // Refresh the lookback with the current chroma row so the NEXT even row's
      // odd-flush box-blends this row as its predecessor (after the read above).
      stage_420_chroma_prev(
        chroma_prev,
        chroma_prev_row,
        row.u_half(),
        row.v_half(),
        idx,
        w,
      );
      return Ok(());
    }

    // HSV-without-RGB-or-RGBA goes through the direct `yuv_420_to_hsv_row`
    // kernel (no source-width RGB scratch). When RGB or RGBA is *also*
    // attached the RGB kernel runs anyway, so HSV derives off that buffer
    // for free — the cheap path — and `need_rgb_kernel` keeps it alive.
    let want_hsv_direct = want_hsv && !want_rgb && !want_rgba;
    let need_rgb_kernel = want_rgb || (want_hsv && want_rgba);

    if want_hsv_direct {
      let hsv = hsv.as_mut().expect("want_hsv_direct implies hsv attached");
      let (h, s, v) = hsv.hsv();
      if center_sited || bottom_v {
        // Scratch already reserved by the atomicity preflight above; this
        // upsample is infallible.
        let (u_full, v_full) = upsample_420_chroma_sited(
          chroma_full,
          chroma_prev,
          chroma_prev_row,
          row.u_half(),
          row.v_half(),
          idx,
          bottom_v,
          center_sited,
          true,
          w,
        );
        yuv_444_to_hsv_row(
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
        );
      } else {
        yuv_420_to_hsv_row(
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
        );
      }
      return Ok(());
    }

    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      if center_sited || bottom_v {
        let (u_full, v_full) = upsample_420_chroma_sited(
          chroma_full,
          chroma_prev,
          chroma_prev_row,
          row.u_half(),
          row.v_half(),
          idx,
          bottom_v,
          center_sited,
          true,
          w,
        );
        yuv_444_to_rgba_row_primaries(
          row.y(),
          u_full,
          v_full,
          rgba_row,
          w,
          row.matrix(),
          primaries,
          row.full_range(),
          use_simd,
        );
      } else {
        yuv_420_to_rgba_row_primaries(
          row.y(),
          row.u_half(),
          row.v_half(),
          rgba_row,
          w,
          row.matrix(),
          primaries,
          row.full_range(),
          use_simd,
        );
      }
      return Ok(());
    }

    if !need_rgb_kernel {
      return Ok(());
    }

    // Pick where the RGB row lands. If the caller wants RGB in their
    // own buffer, write directly there; otherwise use the scratch.
    // Either way, the slice we hold is `&mut [u8]` that we then
    // reborrow as `&[u8]` for the HSV step.
    //
    // RGB byte ranges use `checked_mul` because `w x 3` (and
    // `(idx + 1) x w x 3`) can wrap 32-bit `usize` for large widths
    // even when the single-plane ranges fit — a caller can attach
    // only `with_hsv` (which validates `w x h x 1`) and never go
    // through the `x 3` check at buffer attachment. Overflow here
    // returns `GeometryOverflow` instead of panicking inside the row
    // dispatcher's own checked multiplication.
    let rgb_row = rgb_row_buf_or_scratch(
      rgb.as_deref_mut(),
      rgb_scratch,
      one_plane_start,
      one_plane_end,
      w,
      h,
    )?;

    // Fused YUV→RGB: upsample chroma in registers inside the row
    // primitive, no intermediate memory. Sited siting (#302) instead
    // upsamples chroma to full width (centered `h = 0.5` or co-sited `h = 0`,
    // plus `BottomLeft`'s `v = 1` vertical fold) and runs the 4:4:4 kernel.
    if center_sited || bottom_v {
      let (u_full, v_full) = upsample_420_chroma_sited(
        chroma_full,
        chroma_prev,
        chroma_prev_row,
        row.u_half(),
        row.v_half(),
        idx,
        bottom_v,
        center_sited,
        true,
        w,
      );
      yuv_444_to_rgb_row_primaries(
        row.y(),
        u_full,
        v_full,
        rgb_row,
        w,
        row.matrix(),
        primaries,
        row.full_range(),
        use_simd,
      );
    } else {
      yuv_420_to_rgb_row_primaries(
        row.y(),
        row.u_half(),
        row.v_half(),
        rgb_row,
        w,
        row.matrix(),
        primaries,
        row.full_range(),
        use_simd,
      );
    }

    // HSV from the RGB row we just wrote.
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

impl<R> Yuv420pSink for MixedSinker<'_, Yuv420p, R> {}

/// Native decimation join for 4:2:0 planar sources: Y streams on the
/// frame grid, U and V on the chroma grid (half width, ceil-half
/// height — possibly the upsample direction), every plane binned to
/// FULL output resolution. Each plane's in-order emissions stage into
/// a two-slot ring; the moment all three planes hold an output row it
/// finalizes through the 4:4:4 kernels at output width — so no
/// alignment constraint ever applies to the output geometry. A plane
/// may lead another by at most one source row (the grids are within a
/// factor of two), which two slots absorb.
pub(super) struct NativeYuv420 {
  y: AreaStream<u8>,
  /// Two-slot staging ring, `2 * out_w` (slot = `out_y & 1`).
  y_stage: std::vec::Vec<u8>,
  /// Chroma half of the join — absent for luma-only sinks, which
  /// therefore never read the chroma planes (the documented fast
  /// path). Safe to decide at creation: the frozen-output contract
  /// makes the attached set frame-constant.
  chroma: Option<NativeChroma>,
  /// `staged[plane][slot]` — plane 0 = Y, 1 = U, 2 = V.
  staged: [[bool; 2]; 3],
  /// Next output row to finalize.
  next_emit: usize,
  /// Whether the cached chroma plan was built with a non-zero chroma phase
  /// (RFC #238 centered siting). The join is built once and only `reset`
  /// between frames, so a reused sink whose `chroma_location` moved to a
  /// different phase must REBUILD it — the native arm compares this against the
  /// current siting and drops the join on a mismatch. `false` for a luma-only
  /// join (no chroma plan) and for every co-sited layout.
  chroma_centered: bool,
  /// RFC #238 S4-B: whether the cached chroma plan folded the `Bottom`
  /// vertical phase (`v = 1`). `Center` and `Bottom` share
  /// `chroma_centered = true`, so the horizontal flag alone cannot tell them
  /// apart; the native arm compares this too and rebuilds the join when the
  /// vertical siting moves. `false` for a luma-only join and every
  /// vertical-co-sited layout.
  chroma_bottom: bool,
  /// RFC #238 Top: whether the cached chroma plan folded the `Top` vertical
  /// phase (`v = 0`, the FORWARD triangle). `Center` and `Top` share
  /// `chroma_centered = true` and both keep `chroma_bottom = false`, so neither
  /// the horizontal flag nor the `Bottom` flag can tell them apart; the native
  /// arm compares this too and rebuilds the join when the vertical siting moves
  /// to / from Top. `false` for a luma-only join and every non-Top layout.
  chroma_top: bool,
}

/// Chroma-grid streams and staging of [`NativeYuv420`].
pub(super) struct NativeChroma {
  u: AreaStream<u8>,
  v: AreaStream<u8>,
  u_stage: std::vec::Vec<u8>,
  v_stage: std::vec::Vec<u8>,
}

impl NativeYuv420 {
  /// `build_chroma_plan` lazily builds the 4:2:0 chroma grid against the SAME
  /// output geometry as `plan` (the luma plan) — invoked ONLY when colour is
  /// needed, so a luma-only sink never plans or allocates chroma state. The
  /// per-frame caller folds the RFC #238 horizontal chroma phase into it
  /// ([`ResamplePlan::area_chroma_420`]); at phase 0 the plan is byte-identical
  /// to the pre-siting co-sited grid.
  fn new(
    plan: &ResamplePlan,
    build_chroma_plan: impl FnOnce() -> Result<ResamplePlan, ResampleError>,
    w: usize,
    h: usize,
    need_color: bool,
  ) -> Result<Self, ResampleError> {
    // The native 4:2:0 join is integer area-only; a filter plan reaches it
    // when a FilteredResampler is attached with the native tier enabled, so
    // reject it before building any plane's area stream.
    if plan.kind().is_filter() {
      return Err(plan.unsupported_filter());
    }
    let y = AreaStream::new(plan.h(), plan.v(), w, h, 1)?;
    let alloc =
      |_| ResampleError::AllocationFailed(PlanGeometry::new(w, h, plan.out_w(), plan.out_h()));
    let stage_len = plan.out_w().checked_mul(2).ok_or_else(|| {
      ResampleError::Overflow(PlanGeometry::new(w, h, plan.out_w(), plan.out_h()))
    })?;
    // Recorded so a reused sink rebuilds (not merely resets) the join when the
    // frame's chroma siting phase changes; stays `false` for a luma-only join.
    let mut chroma_centered = false;
    let mut chroma_bottom = false;
    let mut chroma_top = false;
    let chroma = if need_color {
      // Vertical chroma weighting runs in the LUMA domain so an odd trailing
      // luma row weights its chroma row by half; the plan's stored dims are the
      // per-plane denominators (scaled on the H axis for the centered fold, and
      // on the V axis for the RFC #238 S4-B `Bottom` fold). `chroma_centered`
      // keys on the HORIZONTAL phase alone so co-sited-H + `v = 1` `BottomLeft`
      // (which carries a vertical but no horizontal phase) is NOT misread as
      // centered against the sink's `center_sited`.
      let cplan = build_chroma_plan()?;
      chroma_centered = cplan.has_chroma_h_phase();
      chroma_bottom = cplan.has_chroma_v_phase();
      chroma_top = cplan.has_chroma_v_top();
      Some(NativeChroma {
        u: AreaStream::new(cplan.h(), cplan.v(), cplan.src_w(), cplan.src_h(), 1)?,
        v: AreaStream::new(cplan.h(), cplan.v(), cplan.src_w(), cplan.src_h(), 1)?,
        u_stage: try_zeroed(stage_len).map_err(alloc)?,
        v_stage: try_zeroed(stage_len).map_err(alloc)?,
      })
    } else {
      None
    };
    Ok(Self {
      y,
      y_stage: try_zeroed(stage_len).map_err(alloc)?,
      chroma,
      staged: [[false; 2]; 3],
      next_emit: 0,
      chroma_centered,
      chroma_bottom,
      chroma_top,
    })
  }

  pub(super) fn reset(&mut self) {
    self.y.reset();
    if let Some(chroma) = self.chroma.as_mut() {
      chroma.u.reset();
      chroma.v.reset();
    }
    self.staged = [[false; 2]; 3];
    self.next_emit = 0;
  }

  /// Whether this join carries a chroma half — a luma-only join has none, so its
  /// cached plan is siting-agnostic. Read by the semi-planar 4:2:0 (`Nv12` /
  /// `Nv21`) point-of-use siting invalidation, whose joins live in a sibling
  /// module and so cannot touch the private `chroma` field directly.
  #[cfg(feature = "yuv-semi-planar")]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(super) const fn has_chroma(&self) -> bool {
    self.chroma.is_some()
  }

  /// Whether the cached chroma plan was built with a non-zero chroma phase
  /// (RFC #238 centered siting) — see
  /// [`chroma_centered`](Self#structfield.chroma_centered). Read by the
  /// semi-planar 4:2:0 point-of-use siting invalidation to rebuild the join on a
  /// phase change.
  #[cfg(feature = "yuv-semi-planar")]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(super) const fn chroma_centered(&self) -> bool {
    self.chroma_centered
  }

  /// Whether the cached chroma plan folded the RFC #238 S4-C `Bottom` vertical
  /// phase (`v = 1`) — see [`chroma_bottom`](Self#structfield.chroma_bottom).
  /// `Center` and `Bottom` share `chroma_centered = true`, so the semi-planar
  /// 4:2:0 (`Nv12` / `Nv21`) point-of-use siting invalidation compares this too
  /// (its joins live in a sibling module and cannot touch the private field
  /// directly) to rebuild the join on a vertical-phase change.
  #[cfg(feature = "yuv-semi-planar")]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(super) const fn chroma_bottom(&self) -> bool {
    self.chroma_bottom
  }

  /// Whether the cached chroma plan folded the RFC #238 `Top` vertical phase
  /// (`v = 0`, the FORWARD triangle) — see
  /// [`chroma_top`](Self#structfield.chroma_top). `Center` and `Top` share
  /// `chroma_centered = true` and both keep `chroma_bottom = false`, so the
  /// semi-planar 4:2:0 (`Nv12` / `Nv21`) point-of-use siting invalidation
  /// compares this too (its joins live in a sibling module and cannot touch the
  /// private field directly) to rebuild the join on a Top vertical-phase change.
  #[cfg(feature = "yuv-semi-planar")]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(super) const fn chroma_top(&self) -> bool {
    self.chroma_top
  }

  /// Next source row this join expects (0 at a fresh frame). Read by the
  /// semi-planar 4:2:0 point-of-use siting invalidation to fire the
  /// phase-rebuild drop ONLY at a fresh-frame boundary (`next_y() == 0`), never
  /// mid-frame.
  #[cfg(feature = "yuv-semi-planar")]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(super) const fn next_y(&self) -> usize {
    self.y.next_y()
  }

  /// Sequencing preflight across all three plane streams — checked
  /// before any plane is fed so a violating call mutates nothing.
  /// Chroma rows advance once per source-row pair, so their expected
  /// counter is the ceiling half of the source row.
  fn check_sequence(&self, idx: usize) -> Result<(), MixedSinkerError> {
    if self.y.next_y() != idx {
      return Err(MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(
        OutOfSequenceRow::new(self.y.next_y(), idx),
      )));
    }
    if let Some(chroma) = self.chroma.as_ref() {
      let chroma_expected = idx.div_ceil(2);
      for stream in [&chroma.u, &chroma.v] {
        if stream.next_y() != chroma_expected {
          return Err(MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(
            OutOfSequenceRow::new(stream.next_y().saturating_mul(2), idx),
          )));
        }
      }
    }
    Ok(())
  }
}

/// The COMPLETE compare-only pre-feed rejection preflight shared by the native
/// 4:2:0 path ([`yuv420p_process_native`]) and its semi-planar reuse: the
/// no-output short-circuit, the first-row out-of-sequence check, the output-set
/// compare, AND the post-compare sequence check — every rejection point ahead of
/// the delegate's first fallible allocation (the join build / colour scratch /
/// feed). It wraps [`native_preflight_core_check_only`], keyed on the join's Y
/// stream `next_y`, and does **NO commit**: it COMPARES the frozen output set
/// against a fresh snapshot instead of storing it. The delegate builds its join
/// and grows its colour scratch into locals AFTER this passes, then commits the
/// output-set freeze only once every pre-feed allocation has succeeded — so a
/// first-row allocation failure leaves `resample_outputs` (and the cached join)
/// untouched, state-atomic and retryable even with a changed output attachment.
///
/// Returns `Ok(false)` for a no-output call (the caller should no-op),
/// `Ok(true)` to proceed into the join, `Err(OutOfSequenceRow)` for a rejected
/// out-of-sequence first OR post-compare row, or `Err(ResampleOutputsChanged)`
/// for a mid-frame output-set change. The conditional ordering is load-bearing —
/// see [`native_preflight_core_check_only`]: on the first row nothing is frozen
/// yet, so an out-of-sequence row is rejected before the output-set compare
/// (storing no snapshot that would poison a retry); a later row runs the compare
/// first, so a mid-frame output-set change is reported as ResampleOutputsChanged
/// rather than masked, and the post-compare sequence check then rejects an
/// out-of-sequence later row (including the failure-retry case where the join
/// was never built, so `expected == 0`).
///
/// The semi-planar wrapper runs this FIRST, before reserving / filling its
/// U / V de-interleave scratch, so NO rejection case — no-output, OOS first row,
/// OR mid-frame output change — can reach a wrapper allocation (which under
/// allocation pressure would surface AllocationFailed instead of the
/// deterministic typed error). `yuv420p_process_native` re-runs this identical
/// compare and owns the single commit.
///
/// A no-output call has nothing to sequence and stays a no-op regardless of the
/// row index — returned before the compare, the Y-stream sequence check, and the
/// join allocation so it stores no frozen-output snapshot that a later
/// attach-then-retry would trip on. The Y stream is bound to has-output here, not
/// always fed: a no-output row must not advance it (otherwise the snapshot taken
/// on the first output-bearing row of a retry mismatches the rejected no-output
/// row's frozen-as-absent set). The native 4:2:0 path bins the Y plane on every
/// output-bearing row (luma is implicit), so the Y stream is the canonical
/// per-row sequence counter.
#[allow(clippy::too_many_arguments)]
pub(super) fn yuv420p_native_preflight_check_only(
  native_420: &Option<std::boxed::Box<NativeYuv420>>,
  resample_outputs: &Option<super::FrozenOutputs>,
  rgb: &Option<&mut [u8]>,
  rgba: &Option<&mut [u8]>,
  luma: &Option<&mut [u8]>,
  luma_u16: &Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  idx: usize,
  need_luma: bool,
  need_color: bool,
) -> Result<bool, MixedSinkerError> {
  native_preflight_core_check_only(
    native_420.as_ref().map_or(0, |join| join.y.next_y()),
    resample_outputs,
    rgb,
    rgba,
    &None,
    &None,
    luma,
    luma_u16,
    hsv,
    idx,
    need_luma,
    need_color,
  )
}

/// The COMPLETE compare-only 4-point pre-feed rejection logic for the native fast
/// tier: the no-output short-circuit, first-row out-of-sequence, the output-set
/// compare, and post-compare sequence check — with **NO commit**. It takes
/// `resample_outputs` by shared reference and COMPARES the frozen set against a
/// fresh snapshot instead of storing it. The native delegate builds its join and
/// grows its colour scratch into locals AFTER this passes, then commits the
/// output-set freeze via [`frozen_outputs_check`] only once every pre-feed
/// allocation has succeeded — so a first-row allocation failure leaves
/// `resample_outputs` (and the cached join) untouched, state-atomic and retryable
/// even with a changed output attachment. Mirrors
/// [`resample_preflight_check_only`](super::resample_preflight_check_only)
/// for the native tier's element-agnostic (`join.y.next_y()`-keyed) shape.
///
/// Returns `Ok(false)` for a no-output call, `Ok(true)` to proceed,
/// `Err(OutOfSequenceRow)` for a rejected first OR post-compare row, or
/// `Err(ResampleOutputsChanged)` for a mid-frame output-set change.
#[allow(clippy::too_many_arguments)]
pub(super) fn native_preflight_core_check_only(
  expected: usize,
  resample_outputs: &Option<super::FrozenOutputs>,
  rgb: &Option<&mut [u8]>,
  rgba: &Option<&mut [u8]>,
  rgb_u16: &Option<&mut [u16]>,
  rgba_u16: &Option<&mut [u16]>,
  luma: &Option<&mut [u8]>,
  luma_u16: &Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  idx: usize,
  need_luma: bool,
  need_color: bool,
) -> Result<bool, MixedSinkerError> {
  if !need_luma && !need_color {
    return Ok(false);
  }
  // First row only: reject an out-of-sequence row BEFORE the output-set compare,
  // mirroring the committing path's ordering. On a later row the compare runs
  // first, so a genuine mid-frame output change surfaces as ResampleOutputsChanged
  // rather than OutOfSequenceRow.
  if resample_outputs.is_none() && expected != idx {
    return Err(MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(
      OutOfSequenceRow::new(expected, idx),
    )));
  }
  // Mid-frame output-set change → reject WITHOUT committing (a later row only;
  // the first row's `None` stores no snapshot).
  if let Some(frozen) = resample_outputs {
    let snapshot = super::FrozenOutputs::snapshot(
      luma.as_deref(),
      luma_u16.as_deref(),
      rgb.as_deref(),
      rgba.as_deref(),
      rgb_u16.as_deref(),
      rgba_u16.as_deref(),
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
    if *frozen != snapshot {
      return Err(MixedSinkerError::ResampleOutputsChanged(
        super::ResampleOutputsChanged::new(idx),
      ));
    }
  }
  // Post-compare: reject an out-of-sequence later row (mirrors the committing
  // path's post-freeze sequence check).
  if expected != idx {
    return Err(MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(
      OutOfSequenceRow::new(expected, idx),
    )));
  }
  Ok(true)
}

/// Native-tier path for [`MixedSinker<Yuv420p, R>`]: see
/// [`NativeYuv420`]. Phasing mirrors the row-stage tier — frozen
/// configuration, join creation, sequencing, color scratch sizing,
/// then the feeds, with nothing fallible after the first feed.
///
/// Takes the Y plane and the **separate** half-width U / V planes
/// (rather than a `Yuv420pRow`) so the 4:2:0 semi-planar family
/// ([`Nv12`](crate::source::Nv12) / [`Nv21`](crate::source::Nv21)) can
/// reuse it verbatim after de-interleaving its interleaved chroma plane
/// into U / V scratch — every 4:2:0 source then bins Y + U + V through
/// the same per-plane join, byte-identical to a `Yuv420p` conversion of
/// the pre-binned planes.
#[allow(clippy::too_many_arguments)]
pub(super) fn yuv420p_process_native(
  plan: &ResamplePlan,
  native_420: &mut Option<std::boxed::Box<NativeYuv420>>,
  resample_outputs: &mut Option<super::FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  rgb_scratch: &mut std::vec::Vec<u8>,
  y_row: &[u8],
  u_half: &[u8],
  v_half: &[u8],
  matrix: ColorMatrix,
  full_range: bool,
  idx: usize,
  w: usize,
  h: usize,
  build_chroma_plan: impl FnOnce() -> Result<ResamplePlan, ResampleError>,
  use_simd: bool,
) -> Result<(), MixedSinkerError> {
  let ow = plan.out_w();
  let need_luma = luma.is_some() || luma_u16.is_some();
  let need_color = rgb.is_some() || hsv.is_some() || rgba.is_some();
  // HSV-without-RGB-or-RGBA emits directly from the binned output-width
  // Y/U/V via `yuv_444_to_hsv_row` — chroma must still be decimated
  // (`need_color`), but no output-width RGB scratch is staged. RGB or
  // RGBA attached keeps the convert-once-then-derive path.
  let want_hsv_direct = hsv.is_some() && rgb.is_none() && rgba.is_none();

  // Compare-only pre-feed rejection preflight ahead of any fallible allocation
  // (no-output short-circuit, first-row out-of-sequence, output-set compare,
  // post-compare sequence) — see [`yuv420p_native_preflight_check_only`]. It
  // does NOT commit the output-set freeze; that commit is deferred below until
  // every pre-feed allocation has succeeded, so a first-row failure leaves the
  // cached join and `resample_outputs` untouched. Extracted in full so the
  // semi-planar caller can run this identical compare BEFORE it reserves and
  // fills its U / V de-interleave scratch — otherwise a rejected row would grow
  // sink state under allocation pressure and surface AllocationFailed instead of
  // the deterministic typed error. `Ok(false)` is the no-output no-op.
  if !yuv420p_native_preflight_check_only(
    native_420,
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
  // A fresh join is needed on a first-ever build (`None`) or a colour-capability
  // change (the cached join's chroma half no longer matches `need_color` — the
  // frozen check pins the set WITHIN a frame, not across frames). Build the
  // replacement into a LOCAL and do NOT clear the field yet: the cached join
  // stays intact until every pre-feed allocation has succeeded, so a build / box
  // refusal leaves it (and `resample_outputs`) untouched — state-atomic and
  // retryable.
  let rebuild = native_420.is_none()
    || native_420
      .as_ref()
      .is_some_and(|join| join.chroma.is_some() != need_color);
  let new_join = if rebuild {
    // Build the join (its U / V de-interleave scratch allocates recoverably) and
    // box it recoverably (`try_box`, not `Box::new` — the latter aborts on OOM).
    let join = NativeYuv420::new(plan, build_chroma_plan, w, h, need_color)?;
    Some(crate::resample::try_box(join).map_err(|_| {
      MixedSinkerError::Resample(ResampleError::AllocationFailed(PlanGeometry::new(
        w,
        h,
        plan.out_w(),
        plan.out_h(),
      )))
    })?)
  } else {
    None
  };
  // Grow the output-width RGB scratch (fallible) BEFORE the commit — no scratch
  // for the HSV-direct case (the emit loop converts the binned Y/U/V straight to
  // HSV). Hoisting it ahead of the freeze keeps EVERY pre-feed allocation (the
  // join above + this scratch) ahead of the output-set commit, so any refusal is
  // state-atomic.
  if need_color && !want_hsv_direct {
    let row_bytes =
      ow.checked_mul(3)
        .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
          ow,
          plan.out_h(),
          3,
        )))?;
    if rgb_scratch.len() < row_bytes {
      #[cfg(all(test, feature = "std", feature = "yuv-planar"))]
      if FORCE_NATIVE_RGB_SCRATCH_FAILURE.with(|f| f.take()) {
        return Err(MixedSinkerError::Resample(ResampleError::AllocationFailed(
          PlanGeometry::new(w, h, plan.out_w(), plan.out_h()),
        )));
      }
      rgb_scratch
        .try_reserve_exact(row_bytes - rgb_scratch.len())
        .map_err(|_| {
          MixedSinkerError::Resample(ResampleError::AllocationFailed(PlanGeometry::new(
            w,
            h,
            plan.out_w(),
            plan.out_h(),
          )))
        })?;
      rgb_scratch.resize(row_bytes, 0);
    }
  }
  // Every pre-feed allocation succeeded — COMMIT: install the rebuilt join
  // (dropping the old) then freeze the output set. A build only reaches here on
  // the first output-bearing row (its `resample_outputs` is still `None`), so the
  // freeze just stores the snapshot; a reuse row builds nothing and re-freezes
  // nothing new.
  if let Some(boxed) = new_join {
    *native_420 = Some(boxed);
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
  let join = native_420
    .as_mut()
    .expect("built or reused in the preflight");
  join.check_sequence(idx)?;

  // Feed the planes; everything past this point is infallible.
  let NativeYuv420 {
    y,
    y_stage,
    chroma,
    staged,
    next_emit,
    chroma_centered: _,
    chroma_bottom: _,
    chroma_top: _,
  } = &mut **join;
  y.feed_row(idx, y_row, use_simd, |oy, out_row| {
    let slot = oy & 1;
    y_stage[slot * ow..slot * ow + ow].copy_from_slice(out_row);
    staged[0][slot] = true;
  })?;
  if let Some(c) = chroma.as_mut()
    && idx.is_multiple_of(2)
  {
    let cidx = idx / 2;
    let NativeChroma {
      u,
      v,
      u_stage,
      v_stage,
    } = c;
    u.feed_row(cidx, u_half, use_simd, |oy, out_row| {
      let slot = oy & 1;
      u_stage[slot * ow..slot * ow + ow].copy_from_slice(out_row);
      staged[1][slot] = true;
    })?;
    v.feed_row(cidx, v_half, use_simd, |oy, out_row| {
      let slot = oy & 1;
      v_stage[slot * ow..slot * ow + ow].copy_from_slice(out_row);
      staged[2][slot] = true;
    })?;
  }

  // Drain every output row whose participating planes are staged.
  while *next_emit < plan.out_h() {
    let slot = *next_emit & 1;
    let chroma_ready = match chroma.as_ref() {
      Some(_) => staged[1][slot] && staged[2][slot],
      None => true,
    };
    if !(staged[0][slot] && chroma_ready) {
      break;
    }
    let oy = *next_emit;
    let y_out = &y_stage[slot * ow..slot * ow + ow];

    if let Some(buf) = luma.as_deref_mut() {
      buf[oy * ow..(oy + 1) * ow].copy_from_slice(y_out);
    }
    if let Some(buf) = luma_u16.as_deref_mut() {
      for (dst, &src) in buf[oy * ow..(oy + 1) * ow].iter_mut().zip(y_out) {
        *dst = src as u16;
      }
    }
    if let Some(c) = chroma.as_ref() {
      let u_row = &c.u_stage[slot * ow..slot * ow + ow];
      let v_row = &c.v_stage[slot * ow..slot * ow + ow];
      if want_hsv_direct {
        // RGB-free: convert the binned output-width Y/U/V straight to HSV.
        let hsv = hsv.as_mut().expect("want_hsv_direct implies hsv attached");
        let (hp, sp, vp) = hsv.hsv();
        yuv_444_to_hsv_row(
          y_out,
          u_row,
          v_row,
          &mut hp[oy * ow..(oy + 1) * ow],
          &mut sp[oy * ow..(oy + 1) * ow],
          &mut vp[oy * ow..(oy + 1) * ow],
          ow,
          matrix,
          full_range,
          use_simd,
        );
      } else {
        let out_rgb = &mut rgb_scratch[..ow * 3];
        yuv_444_to_rgb_row(
          y_out, u_row, v_row, out_rgb, ow, matrix, full_range, use_simd,
        );
        if let Some(buf) = rgb.as_deref_mut() {
          buf[oy * 3 * ow..(oy + 1) * 3 * ow].copy_from_slice(out_rgb);
        }
        if let Some(hsv) = hsv.as_mut() {
          let (hp, sp, vp) = hsv.hsv();
          rgb_to_hsv_row(
            out_rgb,
            &mut hp[oy * ow..(oy + 1) * ow],
            &mut sp[oy * ow..(oy + 1) * ow],
            &mut vp[oy * ow..(oy + 1) * ow],
            ow,
            use_simd,
          );
        }
        if let Some(buf) = rgba.as_deref_mut() {
          expand_rgb_to_rgba_row(out_rgb, &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow], ow);
        }
      }
    }
    staged[0][slot] = false;
    staged[1][slot] = false;
    staged[2][slot] = false;
    *next_emit += 1;
  }
  Ok(())
}

/// BICUBLIN per-plane filter join for 4:2:0 — the signed-coefficient twin of
/// [`NativeYuv420`]. Y streams on the frame grid through the **cubic** luma
/// filter, U and V on the chroma grid (half width, ceil-half height) through
/// the **linear** chroma filter, every plane resampled to FULL output
/// resolution. Each plane's in-order emissions land in a full-output-plane
/// buffer; the moment all participating planes have emitted an output row it
/// is converted through the 4:4:4 kernel at output width.
///
/// Unlike the area join, a filter window is wide, so a plane can lead another
/// by **more than one** output row (the cubic luma vertical filter and the
/// linear chroma vertical filter complete output rows at different cadences
/// relative to the interleaved source feed). A fixed two-slot ring cannot
/// absorb that, so each plane lands in its own full-output-resolution buffer
/// and a per-plane "rows emitted" cursor tracks how far each has progressed;
/// output rows are converted up to the minimum cursor across the participating
/// planes. The buffers are output-proportional (caller geometry), reserved
/// fallibly at join creation.
pub(super) struct BicublinYuv420 {
  /// Luma plane filter stream (cubic kernel), `src_w x src_h -> out`.
  y: FilterStream<u8>,
  /// Full-output-resolution landing buffer for the filtered Y, `out_w x
  /// out_h`. Always present — colour reads it for the convert and luma copies
  /// from it.
  y_plane: std::vec::Vec<u8>,
  /// Output rows the Y stream has emitted so far (the stream emits in order,
  /// so this is a dense `0..` count).
  y_emitted: usize,
  /// Chroma half of the join — absent for luma-only sinks (which never touch
  /// the chroma planes). Frame-constant by the frozen-output contract.
  chroma: Option<BicublinChroma>,
  /// Next output row to convert / emit.
  next_emit: usize,
}

/// Chroma-grid filter streams and landing buffers of [`BicublinYuv420`].
struct BicublinChroma {
  /// U plane filter stream (linear kernel), `chroma_w x chroma_h -> out`.
  u: FilterStream<u8>,
  /// V plane filter stream (linear kernel), `chroma_w x chroma_h -> out`.
  v: FilterStream<u8>,
  u_plane: std::vec::Vec<u8>,
  v_plane: std::vec::Vec<u8>,
  /// Output rows the U / V streams have emitted so far (both advance in
  /// lockstep — same geometry, same chroma feed — but tracked independently so
  /// a future asymmetry cannot silently desync the drain).
  u_emitted: usize,
  v_emitted: usize,
}

impl BicublinYuv420 {
  /// Builds the per-plane BICUBLIN join from a [`ResamplePlan::is_bicublin`]
  /// plan: the luma stream from the cubic windows
  /// ([`ResamplePlan::filter_h`]/[`ResamplePlan::filter_v`]) over the frame
  /// grid, and (when colour is attached) the U / V streams from the linear
  /// chroma windows
  /// ([`ResamplePlan::filter_h_chroma`]/[`ResamplePlan::filter_v_chroma`])
  /// over the chroma grid. Each plane lands in a full-output-resolution
  /// buffer.
  fn new(plan: &ResamplePlan, w: usize, h: usize, need_color: bool) -> Result<Self, ResampleError> {
    let (fh, fv) = (
      plan
        .filter_h()
        .expect("bicublin plan carries luma horizontal windows"),
      plan
        .filter_v()
        .expect("bicublin plan carries luma vertical windows"),
    );
    let y = FilterStream::<u8>::new(fh, fv, w, h, 1)?;
    let alloc =
      |_| ResampleError::AllocationFailed(PlanGeometry::new(w, h, plan.out_w(), plan.out_h()));
    let plane_len = plan.out_w().checked_mul(plan.out_h()).ok_or_else(|| {
      ResampleError::Overflow(PlanGeometry::new(w, h, plan.out_w(), plan.out_h()))
    })?;
    let chroma = if need_color {
      let (fhc, fvc) = (
        plan
          .filter_h_chroma()
          .expect("bicublin plan carries chroma horizontal windows"),
        plan
          .filter_v_chroma()
          .expect("bicublin plan carries chroma vertical windows"),
      );
      // 4:2:0 chroma grid: half width, ceil-half height (matches the plan's
      // chroma window source dims).
      let (cw, ch) = (w / 2, h.div_ceil(2));
      Some(BicublinChroma {
        u: FilterStream::<u8>::new(fhc, fvc, cw, ch, 1)?,
        v: FilterStream::<u8>::new(fhc, fvc, cw, ch, 1)?,
        u_plane: try_zeroed(plane_len).map_err(alloc)?,
        v_plane: try_zeroed(plane_len).map_err(alloc)?,
        u_emitted: 0,
        v_emitted: 0,
      })
    } else {
      None
    };
    Ok(Self {
      y,
      y_plane: try_zeroed(plane_len).map_err(alloc)?,
      y_emitted: 0,
      chroma,
      next_emit: 0,
    })
  }

  pub(super) fn reset(&mut self) {
    self.y.reset();
    self.y_emitted = 0;
    if let Some(chroma) = self.chroma.as_mut() {
      chroma.u.reset();
      chroma.v.reset();
      chroma.u_emitted = 0;
      chroma.v_emitted = 0;
    }
    self.next_emit = 0;
  }

  /// Sequencing preflight across all three plane streams — checked before any
  /// plane is fed so a violating call mutates nothing. Chroma rows advance
  /// once per source-row pair, so their expected counter is the ceiling half
  /// of the source row. Mirrors [`NativeYuv420::check_sequence`].
  fn check_sequence(&self, idx: usize) -> Result<(), MixedSinkerError> {
    if self.y.next_y() != idx {
      return Err(MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(
        OutOfSequenceRow::new(self.y.next_y(), idx),
      )));
    }
    if let Some(chroma) = self.chroma.as_ref() {
      let chroma_expected = idx.div_ceil(2);
      for stream in [&chroma.u, &chroma.v] {
        if stream.next_y() != chroma_expected {
          return Err(MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(
            OutOfSequenceRow::new(stream.next_y().saturating_mul(2), idx),
          )));
        }
      }
    }
    Ok(())
  }
}

/// BICUBLIN per-plane filter path for [`MixedSinker<Yuv420p, R>`]: filters
/// the Y plane with the cubic luma kernel and the U / V planes with the linear
/// chroma kernel — each in YUV-plane space at its own resolution — then
/// converts the filtered planes to RGB at output resolution through the 4:4:4
/// kernel (the **same** convert the native area tier
/// [`yuv420p_process_native`] runs on its binned planes). It is the filter
/// analog of that area tier; the only differences are the signed-coefficient
/// [`FilterStream`]s in place of the integer [`AreaStream`]s and the
/// full-output-plane landing in place of the two-slot ring (a filter window is
/// wide, so a plane can lead by more than one output row).
///
/// Atomic preflight, mirroring [`yuv420p_process_native`] /
/// [`planar_dual_filter_resample`]: the complete compare-only pre-feed rejection
/// ([`yuv420p_native_preflight_check_only`] — no-output short-circuit, first-row
/// out-of-sequence, output-set compare, post-compare sequence) runs before any
/// fallible allocation; the join is then built into a LOCAL and the colour
/// scratch grown, and only once both succeed does the delegate COMMIT — install
/// the join, freeze the output set via [`frozen_outputs_check`], then run
/// [`Self::check_sequence`] across all three streams — before the first
/// `feed_row`. So a rejected or failed pre-feed row mutates no caller output and
/// leaves the cached join + `resample_outputs` untouched (the deterministic
/// typed error, never `AllocationFailed`-on-abort), and a corrected retry of the
/// SAME row is accepted. Luma is the native Y
/// filter-resampled (the YUV luma contract — never colour-derived); these are
/// 8-bit sources, so the `u8` stream finalises to the full `u8` range, which
/// IS the native range (no sub-16-bit clamp).
#[allow(clippy::too_many_arguments)]
pub(super) fn bicublin_yuv420p_resample(
  plan: &ResamplePlan,
  bicublin_420: &mut Option<std::boxed::Box<BicublinYuv420>>,
  resample_outputs: &mut Option<super::FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  rgb_scratch: &mut std::vec::Vec<u8>,
  y_row: &[u8],
  u_half: &[u8],
  v_half: &[u8],
  matrix: ColorMatrix,
  full_range: bool,
  idx: usize,
  w: usize,
  h: usize,
  use_simd: bool,
) -> Result<(), MixedSinkerError> {
  let ow = plan.out_w();
  let need_luma = luma.is_some() || luma_u16.is_some();
  let need_color = rgb.is_some() || hsv.is_some() || rgba.is_some();
  // HSV-without-RGB-or-RGBA emits HSV straight from the filtered
  // output-width Y/U/V (no output-width RGB scratch); chroma is still
  // filtered (`need_color`). See [`yuv420p_process_native`].
  let want_hsv_direct = hsv.is_some() && rgb.is_none() && rgba.is_none();

  // Compare-only pre-feed rejection preflight ahead of any fallible allocation
  // (no-output short-circuit, first-row out-of-sequence, output-set compare, AND
  // post-compare sequence) — the SAME gate the native tier runs, keyed on the
  // Y stream's `next_y` (luma is implicit on every output-bearing row). It does
  // NOT commit the output-set freeze; that commit is deferred below until every
  // pre-feed allocation has succeeded. A rejected row therefore allocates
  // nothing and a no-output call is a true route-invisible no-op.
  if !yuv420p_native_preflight_bicublin_check_only(
    bicublin_420,
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
  // A fresh join is needed on a first-ever build (`None`) or a colour-capability
  // change (the cached join's chroma half no longer matches `need_color` — the
  // frozen check pins the set WITHIN a frame, not across frames). Build the
  // replacement into a LOCAL and do NOT clear the field yet: the cached join
  // stays intact until every pre-feed allocation has succeeded, so a build / box
  // refusal leaves it (and `resample_outputs`) untouched — state-atomic and
  // retryable.
  let rebuild = bicublin_420.is_none()
    || bicublin_420
      .as_ref()
      .is_some_and(|join| join.chroma.is_some() != need_color);
  let new_join = if rebuild {
    // Build the join (its Y / U / V filter streams allocate recoverably) and box
    // it recoverably (`try_box`, not `Box::new` — the latter aborts on OOM).
    let join = BicublinYuv420::new(plan, w, h, need_color)?;
    Some(crate::resample::try_box(join).map_err(|_| {
      MixedSinkerError::Resample(ResampleError::AllocationFailed(PlanGeometry::new(
        w,
        h,
        plan.out_w(),
        plan.out_h(),
      )))
    })?)
  } else {
    None
  };
  // Grow the output-width RGB scratch (fallible) BEFORE the commit — no scratch
  // for the HSV-direct case (the emit loop converts the filtered Y/U/V straight
  // to HSV). Hoisting it ahead of the freeze keeps EVERY pre-feed allocation
  // (the join above + this scratch) ahead of the output-set commit, so any
  // refusal is state-atomic.
  if need_color && !want_hsv_direct {
    let row_bytes =
      ow.checked_mul(3)
        .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
          ow,
          plan.out_h(),
          3,
        )))?;
    if rgb_scratch.len() < row_bytes {
      #[cfg(all(test, feature = "std", feature = "yuv-planar"))]
      if FORCE_NATIVE_RGB_SCRATCH_FAILURE.with(|f| f.take()) {
        return Err(MixedSinkerError::Resample(ResampleError::AllocationFailed(
          PlanGeometry::new(w, h, plan.out_w(), plan.out_h()),
        )));
      }
      rgb_scratch
        .try_reserve_exact(row_bytes - rgb_scratch.len())
        .map_err(|_| {
          MixedSinkerError::Resample(ResampleError::AllocationFailed(PlanGeometry::new(
            w,
            h,
            plan.out_w(),
            plan.out_h(),
          )))
        })?;
      rgb_scratch.resize(row_bytes, 0);
    }
  }
  // Every pre-feed allocation succeeded — COMMIT: install the rebuilt join
  // (dropping the old) then freeze the output set. A build only reaches here on
  // the first output-bearing row (its `resample_outputs` is still `None`), so the
  // freeze just stores the snapshot; a reuse row builds nothing and re-freezes
  // nothing new.
  if let Some(boxed) = new_join {
    *bicublin_420 = Some(boxed);
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
  let join = bicublin_420
    .as_mut()
    .expect("built or reused in the preflight");
  join.check_sequence(idx)?;

  // Feed the planes; everything past this point is infallible. Each stream
  // lands every emitted output row in its full-output-resolution plane buffer
  // and advances that plane's emitted cursor.
  let BicublinYuv420 {
    y,
    y_plane,
    y_emitted,
    chroma,
    next_emit,
  } = &mut **join;
  y.feed_row(idx, y_row, use_simd, |oy, out_row| {
    y_plane[oy * ow..oy * ow + ow].copy_from_slice(out_row);
    *y_emitted = oy + 1;
  })?;
  if let Some(c) = chroma.as_mut()
    && idx.is_multiple_of(2)
  {
    let cidx = idx / 2;
    let BicublinChroma {
      u,
      v,
      u_plane,
      v_plane,
      u_emitted,
      v_emitted,
    } = c;
    u.feed_row(cidx, u_half, use_simd, |oy, out_row| {
      u_plane[oy * ow..oy * ow + ow].copy_from_slice(out_row);
      *u_emitted = oy + 1;
    })?;
    v.feed_row(cidx, v_half, use_simd, |oy, out_row| {
      v_plane[oy * ow..oy * ow + ow].copy_from_slice(out_row);
      *v_emitted = oy + 1;
    })?;
  }

  // Convert every output row all participating planes have now emitted — up to
  // the minimum emitted cursor across Y (and U / V when colour is attached).
  let ready = match chroma.as_ref() {
    Some(c) => (*y_emitted).min(c.u_emitted).min(c.v_emitted),
    None => *y_emitted,
  };
  while *next_emit < ready {
    let oy = *next_emit;
    let y_out = &y_plane[oy * ow..oy * ow + ow];

    if let Some(buf) = luma.as_deref_mut() {
      buf[oy * ow..(oy + 1) * ow].copy_from_slice(y_out);
    }
    if let Some(buf) = luma_u16.as_deref_mut() {
      for (dst, &src) in buf[oy * ow..(oy + 1) * ow].iter_mut().zip(y_out) {
        *dst = src as u16;
      }
    }
    if let Some(c) = chroma.as_ref() {
      let u_out = &c.u_plane[oy * ow..oy * ow + ow];
      let v_out = &c.v_plane[oy * ow..oy * ow + ow];
      if want_hsv_direct {
        // RGB-free: convert the filtered output-width Y/U/V straight to HSV.
        let hsv = hsv.as_mut().expect("want_hsv_direct implies hsv attached");
        let (hp, sp, vp) = hsv.hsv();
        yuv_444_to_hsv_row(
          y_out,
          u_out,
          v_out,
          &mut hp[oy * ow..(oy + 1) * ow],
          &mut sp[oy * ow..(oy + 1) * ow],
          &mut vp[oy * ow..(oy + 1) * ow],
          ow,
          matrix,
          full_range,
          use_simd,
        );
      } else {
        let out_rgb = &mut rgb_scratch[..ow * 3];
        yuv_444_to_rgb_row(
          y_out, u_out, v_out, out_rgb, ow, matrix, full_range, use_simd,
        );
        if let Some(buf) = rgb.as_deref_mut() {
          buf[oy * 3 * ow..(oy + 1) * 3 * ow].copy_from_slice(out_rgb);
        }
        if let Some(hsv) = hsv.as_mut() {
          let (hp, sp, vp) = hsv.hsv();
          rgb_to_hsv_row(
            out_rgb,
            &mut hp[oy * ow..(oy + 1) * ow],
            &mut sp[oy * ow..(oy + 1) * ow],
            &mut vp[oy * ow..(oy + 1) * ow],
            ow,
            use_simd,
          );
        }
        if let Some(buf) = rgba.as_deref_mut() {
          expand_rgb_to_rgba_row(out_rgb, &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow], ow);
        }
      }
    }
    *next_emit += 1;
  }
  Ok(())
}

/// Compare-only BICUBLIN preflight — the per-plane filter twin of
/// [`yuv420p_native_preflight_check_only`], keyed on the BICUBLIN join's Y
/// stream `next_y` (the canonical per-row counter; luma is implicit on every
/// output-bearing row). Runs [`native_preflight_core_check_only`] so the
/// no-output short-circuit, first-row out-of-sequence, output-set compare, and
/// post-compare sequence checks all precede any fallible allocation, exactly as
/// the native tier does — WITHOUT committing the output-set freeze, which
/// [`bicublin_yuv420p_resample`] defers until every pre-feed allocation
/// (join + colour scratch) has succeeded, so a first-row failure is
/// state-atomic and retryable.
#[allow(clippy::too_many_arguments)]
fn yuv420p_native_preflight_bicublin_check_only(
  bicublin_420: &Option<std::boxed::Box<BicublinYuv420>>,
  resample_outputs: &Option<super::FrozenOutputs>,
  rgb: &Option<&mut [u8]>,
  rgba: &Option<&mut [u8]>,
  luma: &Option<&mut [u8]>,
  luma_u16: &Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  idx: usize,
  need_luma: bool,
  need_color: bool,
) -> Result<bool, MixedSinkerError> {
  native_preflight_core_check_only(
    bicublin_420.as_ref().map_or(0, |join| join.y.next_y()),
    resample_outputs,
    rgb,
    rgba,
    &None,
    &None,
    luma,
    luma_u16,
    hsv,
    idx,
    need_luma,
    need_color,
  )
}

/// Row-stage tier for the 4:2:0 planar family (the `with_native(false)`
/// path), for the RGB / RGBA / luma output sets. Takes the Y plane and the
/// **separate** half-width U / V planes so the 4:2:0 semi-planar family reuses
/// it after de-interleave.
///
/// The **HSV-only** (no RGB / RGBA) and **centered-siting** decodes are handled
/// at the `process` call site (RFC #238): HSV-only routes through the shared
/// [`HsvDirectPlanarYuv`](super::planar_resample::HsvDirectPlanarYuv) join with
/// the frame's chroma phase folded into its plan, and centered RGB / RGBA
/// reconstructs full-width chroma before the 4:4:4 decode — both need the
/// per-frame phase this co-sited tail does not carry. So this path always
/// decodes the co-sited nearest chroma via `yuv_420_to_rgb_row`.
#[allow(clippy::too_many_arguments)]
pub(super) fn yuv420p_process_resampled(
  plan: &ResamplePlan,
  rgb_stream: &mut Option<std::boxed::Box<AreaStream<u8>>>,
  luma_stream: &mut Option<std::boxed::Box<AreaStream<u8>>>,
  resample_outputs: &mut Option<super::FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  rgb_scratch: &mut std::vec::Vec<u8>,
  y_row: &[u8],
  u_half: &[u8],
  v_half: &[u8],
  matrix: ColorMatrix,
  full_range: bool,
  idx: usize,
  w: usize,
  use_simd: bool,
) -> Result<(), MixedSinkerError> {
  // Row-stage 4:2:0 tail is integer area-only: reject a filter plan before
  // any work, so the plan's empty area spans never reach an area stream.
  if plan.kind().is_filter() {
    return Err(plan.unsupported_filter().into());
  }
  let ow = plan.out_w();
  let need_luma = luma.is_some() || luma_u16.is_some();
  let need_color = rgb.is_some() || hsv.is_some() || rgba.is_some();

  // Atomic preflight: every fallible step runs before any stream is
  // fed, so a failed call mutates no caller output and the frame can
  // restart via begin_frame.
  //
  // Single sequence check, on whichever stream is fed every row (all
  // attached streams advance in lockstep). A no-output call has no stream
  // to sequence and stays a no-op regardless of the row index — returned
  // before the freeze so it stores no snapshot a later attach-then-retry
  // would trip on.
  let expected = if need_luma {
    luma_stream.as_ref().map_or(0, |stream| stream.next_y())
  } else if need_color {
    rgb_stream.as_ref().map_or(0, |stream| stream.next_y())
  } else {
    return Ok(());
  };
  // First row: reject an out-of-sequence row BEFORE the freeze, so a
  // rejected first row stores no snapshot that would poison a retry. On a
  // later row the freeze runs first (below), so a mid-frame output-set
  // change is reported as ResampleOutputsChanged rather than masked by a
  // freshly-attached stream's row-0 sequence mismatch (a stream attached
  // mid-frame starts at row 0).
  if resample_outputs.is_none() && expected != idx {
    return Err(MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(
      OutOfSequenceRow::new(expected, idx),
    )));
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
  if expected != idx {
    return Err(MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(
      OutOfSequenceRow::new(expected, idx),
    )));
  }
  if need_luma && luma_stream.is_none() {
    *luma_stream = Some({
      let stream = AreaStream::new(plan.h(), plan.v(), plan.src_w(), plan.src_h(), 1)?;
      crate::resample::try_box(stream).map_err(|_| {
        MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
          crate::resample::PlanGeometry::new(
            plan.src_w(),
            plan.src_h(),
            plan.out_w(),
            plan.out_h(),
          ),
        ))
      })?
    });
  }
  if need_color && rgb_stream.is_none() {
    *rgb_stream = Some({
      let stream = AreaStream::new(plan.h(), plan.v(), plan.src_w(), plan.src_h(), 3)?;
      crate::resample::try_box(stream).map_err(|_| {
        MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
          crate::resample::PlanGeometry::new(
            plan.src_w(),
            plan.src_h(),
            plan.out_w(),
            plan.out_h(),
          ),
        ))
      })?
    });
  }
  // (3) Color-group preparation is also fallible (scratch sizing) and
  // scratch-mutating, so it runs before the luma feed too. The user
  // RGB buffer is output-sized; the source-width row always lands in
  // the scratch. (The overflow arm is defense in depth: any geometry
  // large enough to wrap w * 3 cannot plan — its span arena alloc is
  // out of reach first.)
  //
  // NOTE (#263 follow-up): the HSV-without-RGB case still stages a
  // SOURCE-WIDTH RGB row here. The direct + native fast tiers go
  // RGB-free, but the row-stage resample tail bins an RGB stream
  // (the `AreaStream` is keyed on the 3-channel RGB row) and derives
  // HSV per OUTPUT row off that stream. Eliminating the RGB scratch
  // for resample+HSV-only needs a dedicated HSV-plane resample (resample
  // the HSV planes, or resample Y/U/V then convert per output row) and
  // is deferred to a follow-up PR to keep this one scoped. `need_color`
  // therefore still triggers the source-width grow on the resample path.
  let color_row = if need_color {
    let row_bytes =
      w.checked_mul(3)
        .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
          w,
          plan.src_h(),
          3,
        )))?;
    if rgb_scratch.len() < row_bytes {
      // Same recoverable-allocation contract as the planner and the
      // stream buffers: the scratch is source-width-proportional, so
      // refusal surfaces as an error in the preflight phase instead
      // of aborting inside infallible growth. The exact reserve makes
      // the resize below incapable of reallocating.
      rgb_scratch
        .try_reserve_exact(row_bytes - rgb_scratch.len())
        .map_err(|_| {
          MixedSinkerError::Resample(ResampleError::AllocationFailed(PlanGeometry::new(
            plan.src_w(),
            plan.src_h(),
            plan.out_w(),
            plan.out_h(),
          )))
        })?;
      rgb_scratch.resize(row_bytes, 0);
    }
    let scratch = &mut rgb_scratch[..row_bytes];
    yuv_420_to_rgb_row(
      y_row, u_half, v_half, scratch, w, matrix, full_range, use_simd,
    );
    Some(scratch)
  } else {
    None
  };

  if need_luma {
    let stream = luma_stream.as_mut().expect("created in the preflight");
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
    let stream = rgb_stream.as_mut().expect("created in the preflight");
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

// ---- Native fast tier: STRAIGHT-alpha 4:2:0 planar (Yuva420p) -----------
//
// Gated on `yuva` (which implies `yuv-planar`): this join, its α helper, and
// its entry point reference the alpha helpers (`row::alpha_extract`) that exist
// only under `yuva`, and their sole consumer is the `yuva`-gated
// `Yuva420p` sink. Without the gate a `yuv-planar`-without-`yuva` build pulls
// these items in but not their `yuva` dependencies, so it fails to compile.

/// Native decimation join for the **straight-alpha** 4:2:0 planar source
/// `Yuva420p` (the RFC #238 Phase 5 / #235 alpha resolution) — the
/// alpha-bearing sibling of [`NativeYuv420`].
///
/// The native campaign excluded every alpha format because PREMULTIPLIED
/// alpha is bin-then-convert-incompatible (colour has been scaled by α, so
/// a correct average must convert-at-source, premultiply, bin, then
/// un-premultiply late). STRAIGHT alpha is the exception: colour and α are
/// independent, so binning the native Y / U / V / A codes independently and
/// converting Y / U / V → RGB once per output pixel — with the binned α
/// attached straight — reproduces a `Yuva444p` convert of the pre-binned
/// planes. That is exactly what this join does:
/// - Y / U / V stream and stage exactly as [`NativeYuv420`] (so the rgb /
///   luma / hsv outputs are byte-identical to the no-alpha native tier),
///   embedded here verbatim so the proven 4:2:0 join is reused, not forked.
/// - α streams on the **luma grid** (α is full-resolution in `Yuva420p`,
///   like Y) through its own [`AreaStream<u8>`], staged into a parallel
///   two-slot ring. Per finalized output row the colour RGBA is emitted
///   opaque (the [`NativeYuv420`] path) and the binned α is then scattered
///   into its α slot via [`copy_alpha_plane_u8`](crate::row::alpha_extract::copy_alpha_plane_u8).
///
/// PREMULTIPLIED `Yuva420p` never reaches here — it is routed to the
/// existing packed-YUVA area tail
/// ([`packed_yuva444_resample`](super::packed_yuva444_resample)) BYTE-IDENTICALLY,
/// gated by `AlphaMode::Straight` at the sink.
#[cfg(feature = "yuva")]
pub(super) struct NativeYuva420 {
  /// The embedded no-alpha 4:2:0 join — Y / U / V binning, staging, and the
  /// rgb / rgba(opaque) / luma / hsv emit, reused verbatim.
  inner: NativeYuv420,
  /// Straight α plane, binned on the LUMA grid (full-resolution in
  /// `Yuva420p`). Present only when a colour output that carries α (rgba)
  /// is attached — a luma / rgb / hsv-only sink drops α, so it never bins
  /// the α plane (the documented fast path).
  alpha: Option<NativeAlpha>,
}

/// Luma-grid α stream and staging of [`NativeYuva420`].
#[cfg(feature = "yuva")]
struct NativeAlpha {
  a: AreaStream<u8>,
  /// Two-slot staging ring, `out_w` per slot (slot = `out_y & 1`), parallel
  /// to the embedded [`NativeYuv420`]'s Y / U / V slots.
  a_stage: std::vec::Vec<u8>,
  /// `staged[slot]` — whether this slot holds a finalized α output row.
  staged: [bool; 2],
}

#[cfg(feature = "yuva")]
impl NativeYuva420 {
  /// `need_alpha` is true exactly when an α-carrying colour output (rgba) is
  /// attached AND the alpha mode is straight; the caller resolves it. The
  /// embedded [`NativeYuv420`] is built with `need_color` for the Y / U / V
  /// chroma half (rgb / rgba / hsv all need chroma); α rides its own stream.
  fn new(
    plan: &ResamplePlan,
    w: usize,
    h: usize,
    need_color: bool,
    need_alpha: bool,
    chroma_h_phase: f64,
    chroma_v_phase: f64,
  ) -> Result<Self, ResampleError> {
    // The embedded no-alpha 4:2:0 join folds the RFC #238 horizontal chroma
    // sampling phase (`chroma_h_phase`: `0.25` centered, `0.0` co-sited) into the
    // chroma area weights, exactly as the no-alpha `Yuv420p` / semi-planar `Nv12`
    // native arms do, and (RFC #238 S4-D) the vertical `Bottom` phase
    // (`chroma_v_phase`: `1.0` for `v = 1`, `0.0` co-sited vertical) into the V
    // weights. The full-resolution α plane is siting-independent (never
    // subsampled), so it is untouched. At phase 0 on both axes the folded plan is
    // byte-identical to the plain co-sited grid, so co-sited output is unchanged.
    let inner = NativeYuv420::new(
      plan,
      || {
        ResamplePlan::area_chroma_420(
          w / 2,
          h,
          plan.out_w(),
          plan.out_h(),
          chroma_h_phase,
          chroma_v_phase,
          false,
        )
      },
      w,
      h,
      need_color,
    )?;
    let alpha = if need_alpha {
      // Two-slot staging ring (`slot = out_y & 1`), `2 * out_w` like the
      // embedded Y / U / V stages, so an output row may lead its sibling by
      // one slot without clobbering it.
      let stage_len = plan.out_w().checked_mul(2).ok_or_else(|| {
        ResampleError::Overflow(PlanGeometry::new(w, h, plan.out_w(), plan.out_h()))
      })?;
      let alloc =
        |_| ResampleError::AllocationFailed(PlanGeometry::new(w, h, plan.out_w(), plan.out_h()));
      Some(NativeAlpha {
        // α bins on the luma grid (`w x h`), same plan as Y.
        a: AreaStream::new(plan.h(), plan.v(), w, h, 1)?,
        a_stage: try_zeroed(stage_len).map_err(alloc)?,
        staged: [false; 2],
      })
    } else {
      None
    };
    Ok(Self { inner, alpha })
  }

  pub(super) fn reset(&mut self) {
    self.inner.reset();
    if let Some(alpha) = self.alpha.as_mut() {
      alpha.a.reset();
      alpha.staged = [false; 2];
    }
  }

  /// Whether the embedded 4:2:0 join carries a chroma half — a luma-only join has
  /// none, so its cached plan is siting-agnostic. Read by the `Yuva420p` native
  /// arm's point-of-use siting invalidation, whose sink lives in a sibling module
  /// and so cannot reach the embedded join's private fields directly.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(super) const fn has_chroma(&self) -> bool {
    self.inner.chroma.is_some()
  }

  /// Whether the embedded join's cached chroma plan was built with a non-zero
  /// chroma phase (RFC #238 centered siting). Read by the `Yuva420p` native arm to
  /// rebuild the join on a phase change.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(super) const fn chroma_centered(&self) -> bool {
    self.inner.chroma_centered
  }

  /// Whether the embedded join's cached chroma plan folded the `Bottom` vertical
  /// phase (`v = 1`, RFC #238 S4-D). `Center` and `Bottom` share
  /// `chroma_centered = true`, so the horizontal flag alone cannot tell them
  /// apart; the `Yuva420p` native arm compares this too to rebuild the join when
  /// the vertical siting moves.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(super) const fn chroma_bottom(&self) -> bool {
    self.inner.chroma_bottom
  }

  /// Next source row the embedded join expects (0 at a fresh frame). Read by the
  /// `Yuva420p` native arm to fire the phase-rebuild drop ONLY at a fresh-frame
  /// boundary (`next_y() == 0`), never mid-frame.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(super) const fn next_y(&self) -> usize {
    self.inner.y.next_y()
  }
}

/// Native-tier path for the **straight-alpha** [`MixedSinker<Yuva420p, R>`]:
/// see [`NativeYuva420`]. Mirrors [`yuv420p_process_native`] for the
/// Y / U / V → rgb / luma / hsv outputs and the opaque RGBA fan-out, and
/// additionally bins the straight α plane on the luma grid and substitutes
/// it into the RGBA α slot.
///
/// Atomicity (the Phase 2 lesson, applied from the start): the α stream's
/// sequence is checked and its [`AreaStream`] / staging buffers are
/// allocated in the SAME fallible-before-feed phase as the embedded
/// Y / U / V join, so a rejected row (out-of-sequence, mid-frame output
/// change, or an allocation refusal) mutates no caller output and the frame
/// stays retryable. The α feed only runs after every fallible step, in
/// lockstep with the Y feed (same `idx`, same plan).
///
/// `AlphaMode` is frozen at `begin_frame` and checked by the caller (via
/// [`check_frozen_alpha_mode`](super::check_frozen_alpha_mode)) BEFORE this
/// runs, and the sink only routes here under `AlphaMode::Straight`, so a
/// mid-frame flip to Premultiplied is rejected before any native feed.
///
/// `chroma_h_phase` / `chroma_v_phase` are the RFC #238 horizontal and vertical
/// chroma sampling phases folded into the embedded Y / U / V join's chroma area
/// weights (`chroma_h_phase` `0.25` centered / `0.0` co-sited; `chroma_v_phase`
/// `1.0` for the S4-D `Bottom` `v = 1` fold / `0.0` co-sited vertical); the
/// full-resolution α plane is siting-independent, so neither phase touches it. At
/// phase 0 on both axes the folded plan is byte-identical to the plain co-sited
/// grid.
#[cfg(feature = "yuva")]
#[allow(clippy::too_many_arguments)]
pub(super) fn yuva420p_process_native(
  plan: &ResamplePlan,
  native: &mut Option<std::boxed::Box<NativeYuva420>>,
  resample_outputs: &mut Option<super::FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  rgb_scratch: &mut std::vec::Vec<u8>,
  y_row: &[u8],
  u_half: &[u8],
  v_half: &[u8],
  a_row: &[u8],
  matrix: ColorMatrix,
  full_range: bool,
  idx: usize,
  w: usize,
  h: usize,
  chroma_h_phase: f64,
  chroma_v_phase: f64,
  use_simd: bool,
) -> Result<(), MixedSinkerError> {
  let ow = plan.out_w();
  let need_luma = luma.is_some() || luma_u16.is_some();
  let need_color = rgb.is_some() || hsv.is_some() || rgba.is_some();
  // α is materialised only into the RGBA output (rgb / luma / hsv drop it).
  let need_alpha = rgba.is_some();

  // Compare-only pre-feed rejection preflight — no-output short-circuit,
  // first-row out-of-sequence rejection, the output-set compare, AND the
  // post-compare sequence check, all ahead of any fallible allocation (the
  // compare-only twin of the gate the no-alpha 4:2:0 native tier runs). It does
  // NOT commit the output-set freeze; that commit is deferred below until every
  // pre-feed allocation has succeeded. `Ok(false)` is the no-output no-op. The
  // Y-stream expected row is read from the embedded join; the α stream advances
  // in lockstep with Y (same grid), so the Y sequence governs both.
  if !native_preflight_core_check_only(
    native.as_ref().map_or(0, |n| n.inner.y.next_y()),
    resample_outputs,
    rgb,
    rgba,
    &None,
    &None,
    luma,
    luma_u16,
    hsv,
    idx,
    need_luma,
    need_color,
  )? {
    return Ok(());
  }
  // A fresh join is needed on a first-ever build (`None`) or a
  // colour-or-alpha-capability change (the cached join's chroma / α half no
  // longer matches — the frozen check pins the set WITHIN a frame, not across
  // frames). Build the replacement into a LOCAL and do NOT clear the field yet:
  // the cached join stays intact until every pre-feed allocation has succeeded,
  // so a build / box refusal leaves it (and `resample_outputs`) untouched —
  // state-atomic and retryable.
  let rebuild = native.is_none()
    || native
      .as_ref()
      .is_some_and(|n| n.inner.chroma.is_some() != need_color || n.alpha.is_some() != need_alpha);
  let new_join = if rebuild {
    // Build the join (its inner Y / U / V / α streams allocate recoverably) and
    // box it recoverably (`try_box`, not `Box::new` — the latter aborts on OOM).
    let join = NativeYuva420::new(
      plan,
      w,
      h,
      need_color,
      need_alpha,
      chroma_h_phase,
      chroma_v_phase,
    )?;
    Some(crate::resample::try_box(join).map_err(|_| {
      MixedSinkerError::Resample(ResampleError::AllocationFailed(PlanGeometry::new(
        w,
        h,
        plan.out_w(),
        plan.out_h(),
      )))
    })?)
  } else {
    None
  };
  // Grow the output-width RGB scratch (fallible) BEFORE the commit. Hoisting it
  // ahead of the freeze keeps EVERY pre-feed allocation (the join above + this
  // scratch) ahead of the output-set commit, so any refusal is state-atomic.
  if need_color {
    let row_bytes =
      ow.checked_mul(3)
        .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
          ow,
          plan.out_h(),
          3,
        )))?;
    if rgb_scratch.len() < row_bytes {
      #[cfg(all(test, feature = "std", feature = "yuv-planar"))]
      if FORCE_NATIVE_RGB_SCRATCH_FAILURE.with(|f| f.take()) {
        return Err(MixedSinkerError::Resample(ResampleError::AllocationFailed(
          PlanGeometry::new(w, h, plan.out_w(), plan.out_h()),
        )));
      }
      rgb_scratch
        .try_reserve_exact(row_bytes - rgb_scratch.len())
        .map_err(|_| {
          MixedSinkerError::Resample(ResampleError::AllocationFailed(PlanGeometry::new(
            w,
            h,
            plan.out_w(),
            plan.out_h(),
          )))
        })?;
      rgb_scratch.resize(row_bytes, 0);
    }
  }
  // Every pre-feed allocation succeeded — COMMIT: install the rebuilt join
  // (dropping the old) then freeze the output set. A build only reaches here on
  // the first output-bearing row (its `resample_outputs` is still `None`), so the
  // freeze just stores the snapshot; a reuse row builds nothing and re-freezes
  // nothing new.
  if let Some(boxed) = new_join {
    *native = Some(boxed);
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
  let join = native.as_mut().expect("built or reused in the preflight");
  // Post-commit sequence checks (pure reads — no state mutation on rejection).
  // The Y-keyed preflight above already rejected an out-of-sequence row before
  // any allocation; these re-check the embedded Y stream and, defensively, the
  // lockstep α stream (which shares Y's grid, so `α.next_y == inner.y.next_y`).
  join.inner.check_sequence(idx)?;
  if let Some(alpha) = join.alpha.as_ref()
    && alpha.a.next_y() != idx
  {
    return Err(MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(
      OutOfSequenceRow::new(alpha.a.next_y(), idx),
    )));
  }

  // Feed the planes; everything past this point is infallible. Feed α FIRST
  // (into its staging ring) so its finalized rows are present when the
  // Y / U / V drain emits the matching output rows below.
  if let Some(alpha) = join.alpha.as_mut() {
    let NativeAlpha { a, a_stage, staged } = alpha;
    a.feed_row(idx, a_row, use_simd, |oy, out_row| {
      let slot = oy & 1;
      a_stage[slot * ow..slot * ow + ow].copy_from_slice(out_row);
      staged[slot] = true;
    })?;
  }

  // Feed Y / U / V through the embedded join and emit. The embedded
  // `NativeYuv420` drains rgb / rgba(opaque) / luma / hsv; we re-bind its
  // fields here (rather than calling `yuv420p_process_native`, whose RGBA
  // emit is opaque-only) so the RGBA α slot can be overwritten with the
  // binned α inside the SAME drain — keeping the colour and α outputs
  // finalized together, byte-identical to the no-alpha native tier except
  // for the α channel. `alpha` is destructured alongside `inner` (disjoint
  // fields) so the drain can read the staged α without re-borrowing `join`.
  let NativeYuva420 { inner, alpha } = &mut **join;
  let NativeYuv420 {
    y,
    y_stage,
    chroma,
    staged,
    next_emit,
    chroma_centered: _,
    chroma_bottom: _,
    chroma_top: _,
  } = inner;
  y.feed_row(idx, y_row, use_simd, |oy, out_row| {
    let slot = oy & 1;
    y_stage[slot * ow..slot * ow + ow].copy_from_slice(out_row);
    staged[0][slot] = true;
  })?;
  if let Some(c) = chroma.as_mut()
    && idx.is_multiple_of(2)
  {
    let cidx = idx / 2;
    let NativeChroma {
      u,
      v,
      u_stage,
      v_stage,
    } = c;
    u.feed_row(cidx, u_half, use_simd, |oy, out_row| {
      let slot = oy & 1;
      u_stage[slot * ow..slot * ow + ow].copy_from_slice(out_row);
      staged[1][slot] = true;
    })?;
    v.feed_row(cidx, v_half, use_simd, |oy, out_row| {
      let slot = oy & 1;
      v_stage[slot * ow..slot * ow + ow].copy_from_slice(out_row);
      staged[2][slot] = true;
    })?;
  }

  // Drain every output row whose participating planes are staged. Mirrors
  // the no-alpha `yuv420p_process_native` drain, plus the binned-α
  // substitution into the RGBA α slot. The α slot (when attached) advances
  // in lockstep with Y, so an output row whose Y is staged also has its α
  // staged.
  while *next_emit < plan.out_h() {
    let slot = *next_emit & 1;
    let chroma_ready = match chroma.as_ref() {
      Some(_) => staged[1][slot] && staged[2][slot],
      None => true,
    };
    if !(staged[0][slot] && chroma_ready) {
      break;
    }
    let oy = *next_emit;
    let y_out = &y_stage[slot * ow..slot * ow + ow];

    if let Some(buf) = luma.as_deref_mut() {
      buf[oy * ow..(oy + 1) * ow].copy_from_slice(y_out);
    }
    if let Some(buf) = luma_u16.as_deref_mut() {
      for (dst, &src) in buf[oy * ow..(oy + 1) * ow].iter_mut().zip(y_out) {
        *dst = src as u16;
      }
    }
    if let Some(c) = chroma.as_ref() {
      let u_row = &c.u_stage[slot * ow..slot * ow + ow];
      let v_row = &c.v_stage[slot * ow..slot * ow + ow];
      let out_rgb = &mut rgb_scratch[..ow * 3];
      yuv_444_to_rgb_row(
        y_out, u_row, v_row, out_rgb, ow, matrix, full_range, use_simd,
      );
      if let Some(buf) = rgb.as_deref_mut() {
        buf[oy * 3 * ow..(oy + 1) * 3 * ow].copy_from_slice(out_rgb);
      }
      if let Some(hsv) = hsv.as_mut() {
        let (hp, sp, vp) = hsv.hsv();
        rgb_to_hsv_row(
          out_rgb,
          &mut hp[oy * ow..(oy + 1) * ow],
          &mut sp[oy * ow..(oy + 1) * ow],
          &mut vp[oy * ow..(oy + 1) * ow],
          ow,
          use_simd,
        );
      }
      if let Some(buf) = rgba.as_deref_mut() {
        let rgba_row = &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow];
        // Fan the binned RGB to RGBA (opaque α), then overwrite the α slot
        // with the independently-binned straight α — the #235 straight-alpha
        // resolution: RGB from the YUV convert, α from the native α bin.
        expand_rgb_to_rgba_row(out_rgb, rgba_row, ow);
        if let Some(alpha) = alpha.as_ref() {
          let a_out = &alpha.a_stage[slot * ow..slot * ow + ow];
          crate::row::alpha_extract::copy_alpha_plane_u8(a_out, rgba_row, ow, use_simd);
        }
      }
    }
    staged[0][slot] = false;
    staged[1][slot] = false;
    staged[2][slot] = false;
    if let Some(alpha) = alpha.as_mut() {
      alpha.staged[slot] = false;
    }
    *next_emit += 1;
  }
  Ok(())
}

// ---- Native fast tier: 4:2:2 / 4:4:4 / 4:4:0 planar 8-bit ---------------

/// Native decimation join for the non-4:2:0 8-bit planar families
/// (`Yuv422p` / `Yuv444p` / `Yuv440p`) — the sibling of [`NativeYuv420`]
/// for chroma layouts that are NOT half-resolution in both axes. Y streams
/// on the frame grid; U / V stream on the format's chroma grid, every plane
/// binned to FULL output resolution and converted ONCE per output row at
/// output width through the 4:4:4 kernel (the binned chroma is full-width
/// at output resolution, so the convert is always 4:4:4 — identical to the
/// [`NativeYuv420`] finalize).
///
/// The three formats differ only in the chroma grid and its vertical
/// cadence, both captured here:
/// - `Yuv422p` (4:2:2): chroma `w/2 x h` — half width, FULL height. The
///   chroma plane advances one row per Y row (`chroma_vsub == 1`), so Y and
///   chroma stage in lockstep; the chroma plan is a plain
///   [`ResamplePlan::area`] over `(w/2, h)`.
/// - `Yuv444p` (4:4:4): chroma `w x h` — identical to Y. Same lockstep
///   cadence (`chroma_vsub == 1`); the chroma plan equals the luma plan
///   ([`ResamplePlan::area`] over `(w, h)`).
/// - `Yuv440p` (4:4:0): chroma `w x h/2` — FULL width, half height. The
///   chroma plane advances one row per TWO Y rows (`chroma_vsub == 2`,
///   exactly like 4:2:0 vertically); the chroma plan is
///   [`ResamplePlan::area_chroma_440`] (full-width horizontal,
///   luma-domain `area_halved` vertical so an odd trailing luma row weights
///   its chroma row by half).
///
/// Each plane's in-order emissions stage into a two-slot ring
/// (`out_y & 1`); the moment all participating planes hold an output row it
/// finalizes — so no output-geometry alignment constraint ever applies. A
/// plane may lead another by at most one source row (the chroma grid is
/// within a factor of two of the luma grid vertically), which the two slots
/// absorb. For the lockstep formats (`chroma_vsub == 1`) the planes never
/// lead at all, but the two-slot machinery is harmless and shared.
pub(super) struct NativePlanarYuv {
  y: AreaStream<u8>,
  /// Two-slot staging ring, `2 * out_w` (slot = `out_y & 1`).
  y_stage: std::vec::Vec<u8>,
  /// Chroma half of the join — absent for luma-only sinks, which therefore
  /// never read the chroma planes (the documented fast path). Decided at
  /// creation: the frozen-output contract makes the attached set
  /// frame-constant.
  chroma: Option<NativePlanarChroma>,
  /// Vertical chroma subsample factor: 1 for 4:2:2 / 4:4:4 (a chroma row
  /// per luma row), 2 for 4:4:0 (a chroma row per two luma rows). The
  /// chroma stream is fed when `idx % chroma_vsub == 0`, at
  /// `cidx = idx / chroma_vsub`, and the sequence check expects
  /// `chroma.next_y() == idx.div_ceil(chroma_vsub)`.
  chroma_vsub: usize,
  /// `staged[plane][slot]` — plane 0 = Y, 1 = U, 2 = V.
  staged: [[bool; 2]; 3],
  /// Next output row to finalize.
  next_emit: usize,
  /// Whether the cached chroma plan was built with a non-zero chroma phase
  /// (RFC #238 centered siting). The join is built once and only `reset`
  /// between frames, so a reused sink whose `chroma_location` moved to a
  /// different phase must REBUILD it — `begin_frame` compares this against the
  /// current siting and drops the join on a mismatch. `false` for a luma-only
  /// join (no chroma plan) and for every co-sited / non-4:2:2 layout.
  chroma_centered: bool,
  /// Whether the cached chroma plan folded the `Bottom` vertical chroma phase
  /// (`v = 1`, RFC #238). A vertical-co-sited plan and a `Bottom` one leave
  /// the horizontal `chroma_centered` flag identical, so the vertical axis is
  /// recorded separately here — letting a reused sink tell the two apart and
  /// rebuild the join when the vertical siting moves. Only the
  /// vertically-subsampled 4:4:0 layout folds it; `false` for a luma-only join
  /// and every vertical-co-sited layout.
  #[cfg_attr(
    not(any(feature = "yuv-semi-planar", feature = "yuv-packed")),
    allow(dead_code)
  )]
  chroma_bottom: bool,
}

/// Chroma-grid streams and staging of [`NativePlanarYuv`].
struct NativePlanarChroma {
  u: AreaStream<u8>,
  v: AreaStream<u8>,
  u_stage: std::vec::Vec<u8>,
  v_stage: std::vec::Vec<u8>,
}

#[cfg(all(test, feature = "std", feature = "yuv-planar"))]
std::thread_local! {
  static FORCE_PLANAR_NATIVE_CHROMA_FAILURE: core::cell::Cell<bool> =
    const { core::cell::Cell::new(false) };
  /// Fires at the native tier's output-width RGB-scratch grow, which runs as a
  /// pre-feed step BEFORE the delegate commits (installs the rebuilt join +
  /// freezes the output set) — the pre-commit failure path the transactional
  /// delegates must leave state-atomic (cached join + `resample_outputs`
  /// untouched, retryable).
  static FORCE_NATIVE_RGB_SCRATCH_FAILURE: core::cell::Cell<bool> =
    const { core::cell::Cell::new(false) };
}

/// Arms a failpoint that fires when (and only when) the non-4:2:0 planar
/// native join PLANS its chroma grid — which happens exactly when colour
/// output is requested. A luma-only sink must never reach it, so an armed
/// flag survives a luma-only row unconsumed (the regression assertion) and is
/// taken by the first colour row. Test-only. Mirrors the high-bit
/// `arm_planar_hb_native_chroma_failure`.
///
/// Only the non-4:2:0 native suites that own a colour oracle arm it, and each
/// reuses the planar join so each also pulls in `yuv-planar`: the packed-8-bit /
/// packed-4:4:4 / 4:1:1 natives, plus the semi-planar 8-bit suite's
/// `yuv-planar`-gated `native_tier` (its colour oracle adds `rgb`). So the
/// setter is dead in a `yuv-planar`-without-any-consumer build even though the
/// thread-local it sets is read by the (always-`yuv-planar`) native join.
#[cfg(all(
  test,
  feature = "std",
  feature = "yuv-planar",
  any(
    feature = "yuv-packed",
    feature = "yuv-444-packed",
    all(feature = "yuv-semi-planar", feature = "rgb")
  )
))]
pub(crate) fn arm_planar_native_chroma_failure() {
  FORCE_PLANAR_NATIVE_CHROMA_FAILURE.with(|f| f.set(true));
}

/// Arms [`FORCE_NATIVE_RGB_SCRATCH_FAILURE`] for the NEXT native output-width
/// RGB-scratch grow on this thread (a pre-feed step that runs BEFORE the
/// delegate commits the rebuilt join + output-set freeze), consumed on read so
/// it fires exactly once. Test-only.
#[cfg(all(test, feature = "std", feature = "yuv-planar", feature = "rgb"))]
pub(crate) fn arm_native_rgb_scratch_failure() {
  FORCE_NATIVE_RGB_SCRATCH_FAILURE.with(|f| f.set(true));
}

impl NativePlanarYuv {
  /// `build_chroma_plan` lazily builds the format's chroma grid against the
  /// SAME output geometry as `plan` (the luma plan) — invoked ONLY when colour
  /// is needed, so a luma-only sink never plans or allocates chroma state.
  /// `chroma_vsub` is its vertical cadence. Both are supplied by the
  /// per-format caller so this body stays layout-agnostic.
  fn new(
    plan: &ResamplePlan,
    build_chroma_plan: impl FnOnce() -> Result<ResamplePlan, ResampleError>,
    chroma_vsub: usize,
    w: usize,
    h: usize,
    need_color: bool,
  ) -> Result<Self, ResampleError> {
    // The native planar join is integer area-only; reject a filter plan
    // before building any plane's area stream.
    if plan.kind().is_filter() {
      return Err(plan.unsupported_filter());
    }
    let y = AreaStream::new(plan.h(), plan.v(), w, h, 1)?;
    let alloc =
      |_| ResampleError::AllocationFailed(PlanGeometry::new(w, h, plan.out_w(), plan.out_h()));
    let stage_len = plan.out_w().checked_mul(2).ok_or_else(|| {
      ResampleError::Overflow(PlanGeometry::new(w, h, plan.out_w(), plan.out_h()))
    })?;
    // Recorded so a reused sink rebuilds (not merely resets) the join when the
    // frame's chroma siting phase changes; stays `false` for a luma-only join.
    let mut chroma_centered = false;
    // RFC #238 S8b: the vertical (`Bottom`) companion — set when the chroma plan
    // folds the `v = 1` phase (the 4:4:0 native tier). `Center` and `Bottom`
    // share `chroma_centered`, so the vertical axis is recorded separately.
    let mut chroma_bottom = false;
    let chroma = if need_color {
      #[cfg(all(test, feature = "std", feature = "yuv-planar"))]
      if FORCE_PLANAR_NATIVE_CHROMA_FAILURE.with(|f| f.take()) {
        return Err(ResampleError::AllocationFailed(PlanGeometry::new(
          w,
          h,
          plan.out_w(),
          plan.out_h(),
        )));
      }
      let chroma_plan = build_chroma_plan()?;
      chroma_centered = chroma_plan.has_chroma_phase();
      chroma_bottom = chroma_plan.has_chroma_v_phase();
      Some(NativePlanarChroma {
        u: AreaStream::new(
          chroma_plan.h(),
          chroma_plan.v(),
          chroma_plan.src_w(),
          chroma_plan.src_h(),
          1,
        )?,
        v: AreaStream::new(
          chroma_plan.h(),
          chroma_plan.v(),
          chroma_plan.src_w(),
          chroma_plan.src_h(),
          1,
        )?,
        u_stage: try_zeroed(stage_len).map_err(alloc)?,
        v_stage: try_zeroed(stage_len).map_err(alloc)?,
      })
    } else {
      None
    };
    Ok(Self {
      y,
      y_stage: try_zeroed(stage_len).map_err(alloc)?,
      chroma,
      chroma_vsub,
      staged: [[false; 2]; 3],
      next_emit: 0,
      chroma_centered,
      chroma_bottom,
    })
  }

  pub(super) fn reset(&mut self) {
    self.y.reset();
    if let Some(chroma) = self.chroma.as_mut() {
      chroma.u.reset();
      chroma.v.reset();
    }
    self.staged = [[false; 2]; 3];
    self.next_emit = 0;
  }

  /// Whether this join carries a chroma half — a luma-only join has none, so
  /// its cached plan is siting-agnostic. Read by the semi-planar 4:2:2
  /// (`Nv16`) and packed 4:2:2 (`Yuyv422` / `Uyvy422`) point-of-use siting
  /// invalidation, whose joins live in sibling modules and so cannot touch the
  /// private `chroma` field directly.
  #[cfg(any(feature = "yuv-semi-planar", feature = "yuv-packed"))]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(super) const fn has_chroma(&self) -> bool {
    self.chroma.is_some()
  }

  /// Whether the cached chroma plan was built with a non-zero chroma phase
  /// (RFC #238 centered siting) — see
  /// [`chroma_centered`](Self#structfield.chroma_centered). Read by the
  /// semi-planar 4:2:2 and packed 4:2:2 point-of-use siting invalidation to
  /// rebuild the join on a phase change.
  #[cfg(any(feature = "yuv-semi-planar", feature = "yuv-packed"))]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(super) const fn chroma_centered(&self) -> bool {
    self.chroma_centered
  }

  /// Whether the cached chroma plan folded the `Bottom` vertical chroma phase
  /// (`v = 1`) — see [`chroma_bottom`](Self#structfield.chroma_bottom).
  /// `Center` and `Bottom` leave [`chroma_centered`](Self::chroma_centered)
  /// identical, so a caller distinguishing them on the vertical axis compares
  /// this too.
  #[cfg(any(feature = "yuv-semi-planar", feature = "yuv-packed"))]
  #[cfg_attr(not(tarpaulin), inline(always))]
  #[allow(dead_code)]
  pub(super) const fn chroma_bottom(&self) -> bool {
    self.chroma_bottom
  }

  /// Next source row this join expects (0 at a fresh frame). Read by the
  /// semi-planar 4:2:2 and packed 4:2:2 point-of-use siting invalidation to
  /// fire the phase-rebuild drop ONLY at a fresh-frame boundary
  /// (`next_y() == 0`), never mid-frame.
  #[cfg(any(feature = "yuv-semi-planar", feature = "yuv-packed"))]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(super) const fn next_y(&self) -> usize {
    self.y.next_y()
  }

  /// Sequencing preflight across all three plane streams — checked before
  /// any plane is fed so a violating call mutates nothing. Chroma rows
  /// advance once per `chroma_vsub` source rows, so their expected counter
  /// is `idx.div_ceil(chroma_vsub)`.
  fn check_sequence(&self, idx: usize) -> Result<(), MixedSinkerError> {
    if self.y.next_y() != idx {
      return Err(MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(
        OutOfSequenceRow::new(self.y.next_y(), idx),
      )));
    }
    if let Some(chroma) = self.chroma.as_ref() {
      let chroma_expected = idx.div_ceil(self.chroma_vsub);
      for stream in [&chroma.u, &chroma.v] {
        if stream.next_y() != chroma_expected {
          return Err(MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(
            OutOfSequenceRow::new(stream.next_y().saturating_mul(self.chroma_vsub), idx),
          )));
        }
      }
    }
    Ok(())
  }
}

/// Compare-only preflight wrapper over [`native_preflight_core_check_only`] for
/// the [`NativePlanarYuv`] join — supplies the join-typed expected row and
/// freezes the 8-bit-absent native-depth u16 colour outputs. The 4:2:0 sibling
/// of [`yuv420p_native_preflight_check_only`]: like it, this does **not** commit
/// the output-set freeze; the delegate that calls it commits only after its
/// pre-feed allocations (the join build, the boxing, the output RGB scratch
/// grow) have all succeeded. See `native_preflight_core_check_only` for the
/// 4-point rejection logic and its ordering contract.
#[allow(clippy::too_many_arguments)]
pub(super) fn native_planar_preflight_check_only(
  join: &Option<std::boxed::Box<NativePlanarYuv>>,
  resample_outputs: &Option<super::FrozenOutputs>,
  rgb: &Option<&mut [u8]>,
  rgba: &Option<&mut [u8]>,
  luma: &Option<&mut [u8]>,
  luma_u16: &Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  idx: usize,
  need_luma: bool,
  need_color: bool,
) -> Result<bool, MixedSinkerError> {
  native_preflight_core_check_only(
    join.as_ref().map_or(0, |join| join.y.next_y()),
    resample_outputs,
    rgb,
    rgba,
    &None,
    &None,
    luma,
    luma_u16,
    hsv,
    idx,
    need_luma,
    need_color,
  )
}

/// Native-tier path for the non-4:2:0 8-bit planar families
/// (`Yuv422p` / `Yuv444p` / `Yuv440p`): see [`NativePlanarYuv`]. Bins the
/// Y / U / V planes to output resolution and converts ONCE per output row
/// at output width through the 4:4:4 kernel — vs the row-stage tier
/// ([`planar_dual_resample`](super::planar_resample::planar_dual_resample)),
/// which converts each source row at source width then bins. Phasing
/// mirrors [`yuv420p_process_native`]: the complete pre-feed preflight, the
/// join build, sequencing, colour scratch sizing, then the feeds — with
/// nothing fallible after the first feed.
///
/// `chroma_vsub` is the format's vertical chroma cadence (1 for 4:2:2 /
/// 4:4:4, 2 for 4:4:0) and `build_chroma_plan` builds its chroma grid
/// against the same output geometry; both are supplied by the per-format
/// caller so this body is layout-agnostic.
#[allow(clippy::too_many_arguments)]
pub(super) fn yuv_planar_process_native(
  plan: &ResamplePlan,
  native: &mut Option<std::boxed::Box<NativePlanarYuv>>,
  resample_outputs: &mut Option<super::FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  rgb_scratch: &mut std::vec::Vec<u8>,
  y_row: &[u8],
  u_row: &[u8],
  v_row: &[u8],
  matrix: ColorMatrix,
  full_range: bool,
  idx: usize,
  w: usize,
  h: usize,
  chroma_vsub: usize,
  build_chroma_plan: impl FnOnce() -> Result<ResamplePlan, ResampleError>,
  use_simd: bool,
) -> Result<(), MixedSinkerError> {
  let ow = plan.out_w();
  let need_luma = luma.is_some() || luma_u16.is_some();
  let need_color = rgb.is_some() || hsv.is_some() || rgba.is_some();
  // HSV-without-RGB-or-RGBA emits HSV straight from the binned
  // output-width Y/U/V (no output-width RGB scratch); chroma is still
  // decimated (`need_color`). See [`yuv420p_process_native`].
  let want_hsv_direct = hsv.is_some() && rgb.is_none() && rgba.is_none();

  // Compare-only pre-feed rejection preflight ahead of any fallible allocation
  // (no-output short-circuit, first-row out-of-sequence, output-set compare,
  // post-compare sequence) — see [`native_preflight_core_check_only`]. It does
  // NOT commit the output-set freeze; that commit is deferred below until every
  // pre-feed allocation has succeeded. `Ok(false)` is the no-output no-op.
  if !native_planar_preflight_check_only(
    native,
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
  // A fresh join is needed on a first-ever build (`None`) or a colour-capability
  // change (the cached join's chroma half no longer matches `need_color` — colour
  // attached / detached since the previous frame; the frozen check pins the set
  // WITHIN a frame, not across frames). Build the replacement into a LOCAL and do
  // NOT clear the field yet: the cached join stays intact until every pre-feed
  // allocation has succeeded, so a build / box refusal leaves it (and
  // `resample_outputs`) untouched — state-atomic and retryable.
  let rebuild = native.is_none()
    || native
      .as_ref()
      .is_some_and(|join| join.chroma.is_some() != need_color);
  let new_join = if rebuild {
    // Build the join (its de-interleave scratch allocates recoverably) and box it
    // recoverably (`try_box`, not `Box::new` — the latter aborts on OOM).
    let join = NativePlanarYuv::new(plan, build_chroma_plan, chroma_vsub, w, h, need_color)?;
    Some(crate::resample::try_box(join).map_err(|_| {
      MixedSinkerError::Resample(ResampleError::AllocationFailed(PlanGeometry::new(
        w,
        h,
        plan.out_w(),
        plan.out_h(),
      )))
    })?)
  } else {
    None
  };
  // Grow the output-width RGB scratch (fallible) BEFORE the commit — no scratch
  // for the HSV-direct case (the emit loop converts the binned Y/U/V straight to
  // HSV). Hoisting it ahead of the freeze keeps EVERY pre-feed allocation (the
  // join above + this scratch) ahead of the output-set commit, so any refusal is
  // state-atomic.
  if need_color && !want_hsv_direct {
    let row_bytes =
      ow.checked_mul(3)
        .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
          ow,
          plan.out_h(),
          3,
        )))?;
    if rgb_scratch.len() < row_bytes {
      #[cfg(all(test, feature = "std", feature = "yuv-planar"))]
      if FORCE_NATIVE_RGB_SCRATCH_FAILURE.with(|f| f.take()) {
        return Err(MixedSinkerError::Resample(ResampleError::AllocationFailed(
          PlanGeometry::new(w, h, plan.out_w(), plan.out_h()),
        )));
      }
      rgb_scratch
        .try_reserve_exact(row_bytes - rgb_scratch.len())
        .map_err(|_| {
          MixedSinkerError::Resample(ResampleError::AllocationFailed(PlanGeometry::new(
            w,
            h,
            plan.out_w(),
            plan.out_h(),
          )))
        })?;
      rgb_scratch.resize(row_bytes, 0);
    }
  }
  // Every pre-feed allocation succeeded — COMMIT: install the rebuilt join
  // (dropping the old) then freeze the output set. A build only reaches here on
  // the first output-bearing row (its `resample_outputs` is still `None`), so the
  // freeze just stores the snapshot; a reuse row builds nothing and re-freezes
  // nothing new.
  if let Some(boxed) = new_join {
    *native = Some(boxed);
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
  let join = native.as_mut().expect("built or reused in the preflight");
  join.check_sequence(idx)?;

  // Feed the planes; everything past this point is infallible.
  let NativePlanarYuv {
    y,
    y_stage,
    chroma,
    chroma_vsub,
    staged,
    next_emit,
    // Cache keys read only at build / begin_frame; the feed loop ignores them.
    chroma_centered: _,
    chroma_bottom: _,
  } = &mut **join;
  y.feed_row(idx, y_row, use_simd, |oy, out_row| {
    let slot = oy & 1;
    y_stage[slot * ow..slot * ow + ow].copy_from_slice(out_row);
    staged[0][slot] = true;
  })?;
  if let Some(c) = chroma.as_mut()
    && idx.is_multiple_of(*chroma_vsub)
  {
    let cidx = idx / *chroma_vsub;
    let NativePlanarChroma {
      u,
      v,
      u_stage,
      v_stage,
    } = c;
    u.feed_row(cidx, u_row, use_simd, |oy, out_row| {
      let slot = oy & 1;
      u_stage[slot * ow..slot * ow + ow].copy_from_slice(out_row);
      staged[1][slot] = true;
    })?;
    v.feed_row(cidx, v_row, use_simd, |oy, out_row| {
      let slot = oy & 1;
      v_stage[slot * ow..slot * ow + ow].copy_from_slice(out_row);
      staged[2][slot] = true;
    })?;
  }

  // Drain every output row whose participating planes are staged.
  while *next_emit < plan.out_h() {
    let slot = *next_emit & 1;
    let chroma_ready = match chroma.as_ref() {
      Some(_) => staged[1][slot] && staged[2][slot],
      None => true,
    };
    if !(staged[0][slot] && chroma_ready) {
      break;
    }
    let oy = *next_emit;
    let y_out = &y_stage[slot * ow..slot * ow + ow];

    if let Some(buf) = luma.as_deref_mut() {
      buf[oy * ow..(oy + 1) * ow].copy_from_slice(y_out);
    }
    if let Some(buf) = luma_u16.as_deref_mut() {
      for (dst, &src) in buf[oy * ow..(oy + 1) * ow].iter_mut().zip(y_out) {
        *dst = src as u16;
      }
    }
    if let Some(c) = chroma.as_ref() {
      let u_out = &c.u_stage[slot * ow..slot * ow + ow];
      let v_out = &c.v_stage[slot * ow..slot * ow + ow];
      if want_hsv_direct {
        // RGB-free: convert the binned output-width Y/U/V straight to HSV.
        let hsv = hsv.as_mut().expect("want_hsv_direct implies hsv attached");
        let (hp, sp, vp) = hsv.hsv();
        yuv_444_to_hsv_row(
          y_out,
          u_out,
          v_out,
          &mut hp[oy * ow..(oy + 1) * ow],
          &mut sp[oy * ow..(oy + 1) * ow],
          &mut vp[oy * ow..(oy + 1) * ow],
          ow,
          matrix,
          full_range,
          use_simd,
        );
      } else {
        let out_rgb = &mut rgb_scratch[..ow * 3];
        yuv_444_to_rgb_row(
          y_out, u_out, v_out, out_rgb, ow, matrix, full_range, use_simd,
        );
        if let Some(buf) = rgb.as_deref_mut() {
          buf[oy * 3 * ow..(oy + 1) * 3 * ow].copy_from_slice(out_rgb);
        }
        if let Some(hsv) = hsv.as_mut() {
          let (hp, sp, vp) = hsv.hsv();
          rgb_to_hsv_row(
            out_rgb,
            &mut hp[oy * ow..(oy + 1) * ow],
            &mut sp[oy * ow..(oy + 1) * ow],
            &mut vp[oy * ow..(oy + 1) * ow],
            ow,
            use_simd,
          );
        }
        if let Some(buf) = rgba.as_deref_mut() {
          expand_rgb_to_rgba_row(out_rgb, &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow], ow);
        }
      }
    }
    staged[0][slot] = false;
    staged[1][slot] = false;
    staged[2][slot] = false;
    *next_emit += 1;
  }
  Ok(())
}

// ---- Yuv410p impl -------------------------------------------------------
//
// 4:1:0 planar 8-bit (Cinepak / Sorenson legacy, Tier 1 P3). Chroma
// is subsampled 4:1 in **both** axes — `width / 4` chroma bytes per
// row, `height / 4` chroma rows. Per-row math reuses the dedicated
// `yuv_410_to_rgb_row` / `yuv_410_to_rgba_row` kernels (4:2:0's
// half-width chroma layout doesn't apply — each chroma sample
// covers four Y columns instead of two).
//
// Output set matches the Tier 1 contract:
// `with_rgb` / `with_rgba` / `with_luma` / `with_luma_u16` / `with_hsv`.
// No source alpha (no YUVA 4:1:0 format). Strategy A applies for the
// RGB+RGBA combo: run the 3-channel kernel once, fan out via
// `expand_rgb_to_rgba_row`.

impl<'a, R> MixedSinker<'a, Yuv410p, R> {
  /// Attaches a packed 32-bit RGBA output buffer.
  ///
  /// See [`MixedSinker::<Yuv420p>::with_rgba`] for the rationale and
  /// constraints. Yuv410p has no alpha plane, so every alpha byte is
  /// filled with `0xFF` (opaque).
  ///
  /// Returns `Err(InsufficientRgbaBuffer)` if
  /// `buf.len() < width x height x 4`, or `Err(GeometryOverflow)` on
  /// 32-bit targets when the product overflows.
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

  /// Attaches a `u16` luma output buffer. The 8-bit Y plane samples
  /// are zero-extended into `u16`.
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

impl<R> Yuv410pSink for MixedSinker<'_, Yuv410p, R> {}

impl<R> PixelSink for MixedSinker<'_, Yuv410p, R> {
  type Input<'r> = Yuv410pRow<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    // Yuv410p requires width to be a multiple of 4 (the row kernels
    // operate on 4-pixel chroma groups). Height is unconstrained — the
    // walker reuses the trailing chroma row for the final 1..=3 Y rows.
    // The frame `try_new` already enforces width alignment, but a
    // hand-crafted row walker (driving `process` directly) might bypass
    // it — guard at the sinker boundary so unsafe SIMD dispatchers
    // never see a non-multiple-of-4 width.
    if self.width & 3 != 0 {
      return Err(MixedSinkerError::WidthAlignment(
        WidthAlignment::multiple_of_four(self.width),
      ));
    }
    check_dimensions_match(self.width, self.height, width, height)?;
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
    // New frame: restart the RGB-free HSV-only row-stage join (#263 follow-up).
    if let Some(hsv) = self.hsv_planar.as_mut() {
      hsv.reset();
    }
    // RFC #238 #302: clear the per-frame frozen chroma siting so the next frame
    // may pick either phase; a mid-frame flip stays rejected within a frame.
    self.frozen_chroma_centered = None;
    self.resample_outputs = None;
    Ok(())
  }

  fn process(&mut self, row: Yuv410pRow<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;
    // Chroma siting (#302): drives the horizontal chroma phase. `Copy`, so read
    // it out before the field split-borrow below.
    let chroma_location = self.chroma_location;

    // Defense in depth — see Yuv420p impl.
    if w & 3 != 0 {
      return Err(MixedSinkerError::WidthAlignment(
        WidthAlignment::multiple_of_four(w),
      ));
    }
    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Y,
        idx,
        w,
        row.y().len(),
      )));
    }
    if row.u_quarter().len() != w / 4 {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::UQuarter,
        idx,
        w / 4,
        row.u_quarter().len(),
      )));
    }
    if row.v_quarter().len() != w / 4 {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::VQuarter,
        idx,
        w / 4,
        row.v_quarter().len(),
      )));
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange(
        RowIndexOutOfRange::new(idx, self.height),
      ));
    }

    let Self {
      rgb,
      rgba,
      luma,
      luma_u16,
      hsv,
      rgb_scratch,
      plan,
      rgb_stream,
      luma_stream,
      rgb_filter_stream,
      luma_filter_stream,
      hsv_planar,
      resample_outputs,
      // Centered chroma-siting (#302) stages full-width U + V here.
      chroma_full,
      // RFC #238: the 4:1:0 chroma phase frozen on the first output row.
      frozen_chroma_centered,
      ..
    } = self;

    // Non-identity plan: convert the source row to canonical RGB at source width
    // in the shared scratch (the same `yuv_410_to_rgb_row` kernel the identity
    // path uses — 4:1:0's 1→4 chroma upsample), then feed the shared planar
    // resample tail. Row-stage only — converting each source row to RGB and
    // binning is the whole job. The `Area` arm bins the converted RGB / native Y;
    // the `Filter` arm filter-resamples them through the merged engine. Both
    // freeze the output set and check stream sequencing before staging, so a
    // no-output sink stays a no-op and an out-of-sequence row is rejected without
    // allocating. RFC #238 #302: the centered group
    // (`chroma_410_center_sited_h`) samples chroma at the CENTER of its four luma
    // columns — the RGB converters reconstruct full-width chroma here (`1→4` #302
    // horizontal phase), the HSV-only arm folds the phase into the chroma area
    // weights; the co-sited / unspecified group keeps the byte-identical nearest
    // decode. This PR sites 4:1:0 horizontally only (the vertical chroma phase is
    // unchanged).
    if let Some(plan) = plan.as_ref() {
      let matrix = row.matrix();
      let full_range = row.full_range();
      let center_sited = chroma_410_center_sited_h(chroma_location);
      let chroma_h_phase = if center_sited {
        YUV41X_CENTERED_H_PHASE
      } else {
        0.0
      };
      let want_color = rgb.is_some() || rgba.is_some() || hsv.is_some();
      let need_output =
        luma.is_some() || luma_u16.is_some() || rgb.is_some() || rgba.is_some() || hsv.is_some();
      // RFC #238 #302: freeze the effective 4:1:0 chroma siting on the first
      // output-bearing row; a later row observing a different phase would bin a
      // mixture of co-sited and centered chroma, so reject it here before any
      // reconstruction. The matching SET rides each path's accept-time freeze.
      if need_output
        && let Some(frozen) = *frozen_chroma_centered
        && frozen != center_sited
      {
        return Err(MixedSinkerError::ChromaSitingChanged(
          ChromaSitingChanged::new(idx),
        ));
      }
      // Co-sited RGB decode (nearest `1→4` upsample). The centered arms build
      // their own 4:4:4 converter over freshly reconstructed chroma below.
      let convert_rgb = |scratch: &mut [u8]| {
        yuv_410_to_rgb_row(
          row.y(),
          row.u_quarter(),
          row.v_quarter(),
          scratch,
          w,
          matrix,
          full_range,
          use_simd,
        );
      };
      // HSV-only (no RGB / RGBA) area plan: bin Y / U / V and convert per
      // output row, RGB-free. 4:1:0 chroma is quarter-width AND quarter-height
      // (`chroma_vsub = 4`); the folded `area_chroma_410` plan carries the
      // centered phase. Bit-identical to a YUV-domain bin-then-convert reference.
      // A filter plan keeps the RGB-staged HSV (filter twin), and RGB / RGBA
      // attached keeps the RGB-staged path.
      if plan.kind().is_area() && hsv.is_some() && rgb.is_none() && rgba.is_none() {
        // RFC #238 point-of-use siting invalidation (see the Yuv422p HSV-only
        // arm): drop the cached HSV-only join if its chroma plan was built for a
        // different phase; retry-atomic (fires only on the IN-SEQUENCE fresh-frame
        // first row) and the taken join is restored on a rejected rebuild.
        let stale_hsv = idx == 0
          && hsv_planar
            .as_ref()
            .is_some_and(|join| join.chroma_centered() != center_sited && join.next_y() == 0);
        let prev_hsv = if stale_hsv { hsv_planar.take() } else { None };
        let hsv_result = super::planar_resample::hsv_direct_resample(
          hsv_planar,
          resample_outputs,
          luma,
          luma_u16,
          hsv,
          row.y(),
          row.u_quarter(),
          row.v_quarter(),
          matrix,
          full_range,
          4,
          || ResamplePlan::area_chroma_410(w, h, plan.out_w(), plan.out_h(), chroma_h_phase, 0.0),
          w,
          plan,
          idx,
          use_simd,
        );
        if stale_hsv && hsv_result.is_err() {
          *hsv_planar = prev_hsv;
        }
        hsv_result?;
        if frozen_chroma_centered.is_none() && need_output {
          *frozen_chroma_centered = Some(center_sited);
        }
        return Ok(());
      }
      // RGB / RGBA area or filter. The co-sited path bins the nearest
      // `yuv_410_to_rgb_row` decode; the centered path reconstructs full-width
      // chroma (`1→4` #302 phase) AFTER the resample preflight and feeds the
      // 4:4:4 decode, so an out-of-sequence / rejected row is caught before the
      // chroma reservation (#180). `want_color` gates the reservation: a
      // luma-only row bins only luma and must not reserve / reconstruct chroma.
      let r = if center_sited && want_color {
        // Reject a multi-kernel (BICUBLIN) filter plan BEFORE the centered
        // preflight / reserve (the filter delegate's first act; hoisted so a
        // rejected BICUBLIN plan reserves nothing).
        if plan.kind().is_filter() {
          plan.ensure_single_kernel_filter()?;
        }
        let need_luma = luma.is_some() || luma_u16.is_some();
        let expected = if plan.kind().is_filter() {
          if need_luma {
            luma_filter_stream.as_ref().map_or(0, |s| s.next_y())
          } else {
            rgb_filter_stream.as_ref().map_or(0, |s| s.next_y())
          }
        } else if need_luma {
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
        let (u_full, v_full) = reconstruct_chroma(
          Chroma411U8,
          chroma_full,
          row.u_quarter(),
          row.v_quarter(),
          w,
        );
        let convert_444 = |scratch: &mut [u8]| {
          yuv_444_to_rgb_row(
            row.y(),
            u_full,
            v_full,
            scratch,
            w,
            matrix,
            full_range,
            use_simd,
          );
        };
        match plan.kind() {
          crate::resample::SpanKind::Area => planar_dual_resample(
            luma_stream,
            rgb_stream,
            resample_outputs,
            rgb,
            rgba,
            luma,
            luma_u16,
            hsv,
            rgb_scratch,
            row.y(),
            w,
            plan,
            idx,
            use_simd,
            convert_444,
          ),
          crate::resample::SpanKind::Filter => planar_dual_filter_resample(
            luma_filter_stream,
            rgb_filter_stream,
            resample_outputs,
            rgb,
            rgba,
            luma,
            luma_u16,
            hsv,
            rgb_scratch,
            row.y(),
            w,
            plan,
            idx,
            use_simd,
            convert_444,
          ),
        }
      } else {
        match plan.kind() {
          crate::resample::SpanKind::Area => planar_dual_resample(
            luma_stream,
            rgb_stream,
            resample_outputs,
            rgb,
            rgba,
            luma,
            luma_u16,
            hsv,
            rgb_scratch,
            row.y(),
            w,
            plan,
            idx,
            use_simd,
            convert_rgb,
          ),
          crate::resample::SpanKind::Filter => planar_dual_filter_resample(
            luma_filter_stream,
            rgb_filter_stream,
            resample_outputs,
            rgb,
            rgba,
            luma,
            luma_u16,
            hsv,
            rgb_scratch,
            row.y(),
            w,
            plan,
            idx,
            use_simd,
            convert_rgb,
          ),
        }
      };
      if r.is_ok() && need_output && frozen_chroma_centered.is_none() {
        *frozen_chroma_centered = Some(center_sited);
      }
      return r;
    }

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // Strategy A output mode resolution — see Yuv420p impl above. The centered
    // horizontal sitings (#302) reconstruct chroma at the `1→4` phase-0.375
    // position then reuse the 4:4:4 dispatchers.
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();
    let want_color = want_rgb || want_rgba || want_hsv;

    // Chroma siting (#302): the centered horizontal sitings reconstruct chroma at
    // the `1→4` phase-0.375 position then decode via the 4:4:4 kernels; the
    // default / co-sited path keeps the byte-identical nearest decode. This PR
    // sites 4:1:0 horizontally only — the vertical chroma replication is
    // unchanged.
    let center_sited = chroma_410_center_sited_h(chroma_location);

    // RFC #238 #302: mirror the resample branch's per-frame siting freeze on the
    // identity path. `chroma_location` is public and `Copy`, so a caller can flip
    // the phase mid-frame; without this a single frame could mix centered (`1→4`)
    // and co-sited nearest reconstruction. Freeze the effective phase on the first
    // output-bearing row and reject a later differing phase BEFORE any centered
    // reservation or output write, so a rejected row mutates nothing (#180). The
    // matching SET rides the accept-time freeze after the reservations below.
    let need_output =
      luma.is_some() || luma_u16.is_some() || rgb.is_some() || rgba.is_some() || hsv.is_some();
    if need_output
      && let Some(frozen) = *frozen_chroma_centered
      && frozen != center_sited
    {
      return Err(MixedSinkerError::ChromaSitingChanged(
        ChromaSitingChanged::new(idx),
      ));
    }

    // Atomicity preflight (#302 / #308, cf. the crate's #180 resample fix and the
    // Yuv420p sibling): reserve EVERY fallible row scratch this row needs BEFORE
    // any output row (luma / luma_u16 included) is written, so an allocator
    // refusal returns a typed `AllocationFailed` leaving the output frame
    // untouched. Two scratches can grow: the centered-siting full-width chroma
    // (`chroma_full`, only on a colour decode) and the RGB row buffer (`want_hsv
    // && want_rgba && !want_rgb`). The later `reconstruct_chroma` /
    // `rgb_row_buf_or_scratch` calls reuse the already-sized buffers, so the
    // default path is byte-identical; only the failure-path ordering changes.
    if center_sited && want_color {
      reserve_420_chroma_full(chroma_full, w, h)?;
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

    // RFC #238 #302: every fallible per-row reservation above has succeeded, so
    // freeze the accepted phase for the frame (mirrors the resample branch's
    // accept-time SET). Never before the reservations, so a rejected or failed row
    // leaves the freeze clear for a corrected retry.
    if need_output && frozen_chroma_centered.is_none() {
      *frozen_chroma_centered = Some(center_sited);
    }

    // Luma — Yuv410p luma *is* the Y plane. Just copy.
    if let Some(luma) = luma.as_deref_mut() {
      luma[one_plane_start..one_plane_end].copy_from_slice(&row.y()[..w]);
    }

    // Luma u16 — zero-extend the 8-bit Y plane into u16.
    if let Some(buf) = luma_u16.as_deref_mut() {
      crate::row::y_plane_to_luma_u16_row(
        row.y(),
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    // HSV-only (no RGB / RGBA) goes direct through `yuv_410_to_hsv_row` (centered:
    // `yuv_444_to_hsv_row` on the reconstructed chroma) — see the Yuv420p impl for
    // the routing rationale.
    let want_hsv_direct = want_hsv && !want_rgb && !want_rgba;
    let need_rgb_kernel = want_rgb || (want_hsv && want_rgba);

    if want_hsv_direct {
      let hsv = hsv.as_mut().expect("want_hsv_direct implies hsv attached");
      let (h, s, v) = hsv.hsv();
      if center_sited {
        let (u_full, v_full) = reconstruct_chroma(
          Chroma411U8,
          chroma_full,
          row.u_quarter(),
          row.v_quarter(),
          w,
        );
        yuv_444_to_hsv_row(
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
        );
      } else {
        yuv_410_to_hsv_row(
          row.y(),
          row.u_quarter(),
          row.v_quarter(),
          &mut h[one_plane_start..one_plane_end],
          &mut s[one_plane_start..one_plane_end],
          &mut v[one_plane_start..one_plane_end],
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
        );
      }
      return Ok(());
    }

    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      if center_sited {
        let (u_full, v_full) = reconstruct_chroma(
          Chroma411U8,
          chroma_full,
          row.u_quarter(),
          row.v_quarter(),
          w,
        );
        yuv_444_to_rgba_row(
          row.y(),
          u_full,
          v_full,
          rgba_row,
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
        );
      } else {
        yuv_410_to_rgba_row(
          row.y(),
          row.u_quarter(),
          row.v_quarter(),
          rgba_row,
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
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

    if center_sited {
      let (u_full, v_full) = reconstruct_chroma(
        Chroma411U8,
        chroma_full,
        row.u_quarter(),
        row.v_quarter(),
        w,
      );
      yuv_444_to_rgb_row(
        row.y(),
        u_full,
        v_full,
        rgb_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
    } else {
      yuv_410_to_rgb_row(
        row.y(),
        row.u_quarter(),
        row.v_quarter(),
        rgb_row,
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

    if let Some(buf) = rgba.as_deref_mut() {
      let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
    }

    Ok(())
  }
}

// ---- Yuv422p impl -------------------------------------------------------
//
// 4:2:2 is 4:2:0's vertical-axis twin: same per-row chroma shape
// (half-width U / V, one pair per Y pair), just one chroma row per Y
// row instead of one per two. This impl reuses `yuv_420_to_rgb_row`
// (and `yuv_420_to_rgba_row` for the RGBA path) — no new kernels
// needed.

impl<'a, R> MixedSinker<'a, Yuv422p, R> {
  /// Attaches a packed 32‑bit RGBA output buffer.
  ///
  /// Only available on sinker types whose `PixelSink` impl writes
  /// RGBA — see [`MixedSinker::<Yuv420p>::with_rgba`] for the same
  /// rationale and constraints. Yuv422p has no alpha plane, so every
  /// alpha byte is filled with `0xFF` (opaque).
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

  /// Attaches a `u16` luma output buffer. The 8-bit Y plane samples
  /// are zero-extended into `u16`. Length is measured in `u16`
  /// elements (`width x height`).
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

impl<R> Yuv422pSink for MixedSinker<'_, Yuv422p, R> {}

impl<R> PixelSink for MixedSinker<'_, Yuv422p, R> {
  type Input<'r> = Yuv422pRow<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::WidthAlignment(WidthAlignment::odd(
        self.width,
      )));
    }
    check_dimensions_match(self.width, self.height, width, height)?;
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
    if let Some(native) = self.native_planar.as_mut() {
      native.reset();
    }
    // New frame: restart the RGB-free HSV-only row-stage join (#263 follow-up).
    // (RFC #238 siting invalidation is done at point-of-use in `process`, not
    // here — `chroma_location` may change AFTER `begin_frame` but before row 0,
    // so the cached join's phase is re-checked immediately before it is reused.)
    if let Some(hsv) = self.hsv_planar.as_mut() {
      hsv.reset();
    }
    self.frozen_native_route = None;
    self.frozen_domain = None;
    self.resample_outputs = None;
    // New frame: clear the RFC #238 frozen 4:2:2 chroma siting so the next
    // frame may pick either phase; a mid-frame flip stays rejected.
    self.frozen_chroma_centered = None;
    // New frame: drop the RFC #238 linear-light accumulator (if any).
    #[cfg(feature = "rgb")]
    {
      self.linear_light_frame = None;
    }
    Ok(())
  }

  fn process(&mut self, row: Yuv422pRow<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;
    // Chroma siting (#302): drives the identity-plan horizontal chroma phase.
    // `Copy`, so read it out before the field split-borrow below.
    let chroma_location = self.chroma_location;

    if w & 1 != 0 {
      return Err(MixedSinkerError::WidthAlignment(WidthAlignment::odd(w)));
    }
    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Y,
        idx,
        w,
        row.y().len(),
      )));
    }
    if row.u_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::UHalf,
        idx,
        w / 2,
        row.u_half().len(),
      )));
    }
    if row.v_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::VHalf,
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

    let Self {
      rgb,
      rgba,
      luma,
      luma_u16,
      hsv,
      rgb_scratch,
      plan,
      rgb_stream,
      luma_stream,
      rgb_filter_stream,
      luma_filter_stream,
      resample_outputs,
      native,
      native_planar,
      hsv_planar,
      frozen_native_route,
      frozen_domain,
      averaging_domain,
      #[cfg(feature = "rgb")]
      linear_mode,
      #[cfg(feature = "rgb")]
      linear_light_frame,
      #[cfg(feature = "rgb")]
      linear_scene_scratch,
      #[cfg(feature = "rgb")]
      transfer_function,
      // Centered chroma-siting (#302) stages full-width U + V here.
      chroma_full,
      // RFC #238 S1: the 4:2:2 chroma phase frozen on the first output row.
      frozen_chroma_centered,
      ..
    } = self;

    // Non-identity plan. A `Filter` plan converts each source row to a
    // source-width RGB row (the 4:2:0 per-row dispatcher 4:2:2 reuses) and
    // filter-resamples it plus the native Y. An `Area` plan picks the
    // native fast tier (bin Y / U / V to output res, convert once at output
    // width) or the row-stage tier (convert each source row, bin RGB) per
    // `with_native`, frozen per frame. Both tiers freeze the output set and
    // check stream sequencing before staging, so a no-output sink stays a
    // no-op and an out-of-sequence row is rejected without allocating.
    if let Some(plan) = plan.as_ref() {
      let matrix = row.matrix();
      let full_range = row.full_range();
      // RFC #238 S1 — 4:2:2 horizontal chroma siting. The centered group
      // (`Center` / `Top` / `Bottom`) samples chroma at `+0.5` luma; the
      // co-sited / unspecified group is phase 0 (today's nearest decode,
      // byte-identical). Siting enters the chroma RECONSTRUCTION only — the
      // averaging DOMAIN is still chosen by `select_insertion_point` below —
      // so the native fast tier and the encoded HSV-only arm fold the phase
      // into the chroma area weights (`area_chroma_422`), while the RGB-domain
      // converters reconstruct full-width chroma here.
      let center_sited = chroma_422_center_sited_h(chroma_location);
      let chroma_h_phase = if center_sited {
        YUV422P_CENTERED_H_PHASE
      } else {
        0.0
      };
      // Only the RGB-domain converters reconstruct full-width chroma for the
      // centered decode: the encoded row-stage RGB/RGBA bin and the filter bin.
      // The native fast tier and the encoded HSV-only arm bin the SUBSAMPLED
      // chroma through the folded `area_chroma_422` plan, and the linear-light
      // tail keeps the co-sited decode. `convert_rgb` here is the co-sited
      // nearest decode (linear + every co-sited RGB path); the centered RGB/RGBA
      // and filter arms build their OWN 4:4:4 converter over freshly
      // reconstructed chroma AFTER the resample preflight, so a rejected /
      // out-of-sequence row is caught before that chroma reservation (the #180
      // reserve-after-preflight invariant).
      let want_color = rgb.is_some() || rgba.is_some() || hsv.is_some();
      let convert_rgb = |scratch: &mut [u8]| {
        yuv_420_to_rgb_row(
          row.y(),
          row.u_half(),
          row.v_half(),
          scratch,
          w,
          matrix,
          full_range,
          use_simd,
        );
      };
      // The scene-referred ([`LinearMode::SceneReferred`]) twin of
      // `convert_rgb`: the SAME affine matrix, real-valued and unclamped (4:2:2
      // reuses the 4:2:0 horizontal chroma upsample, as `convert_rgb` reuses
      // `yuv_420_to_rgb_row`). Consumed only by the `rgb`-gated Linear arm.
      #[cfg(feature = "rgb")]
      let convert_rgb_unclamped = |dst: &mut [f32]| {
        crate::row::scalar::yuv_420_to_rgb_f32_unclamped_row(
          row.y(),
          row.u_half(),
          row.v_half(),
          dst,
          w,
          matrix,
          full_range,
        );
      };
      // RFC #238 Phase 2 — single always-compiled choke point for the averaging
      // domain, BEFORE any filter / native / row-stage branching, so a
      // non-encoded sink never falls through to the Encoded path under any
      // feature combination. The match is EXHAUSTIVE with no wildcard arm: a
      // future `AveragingDomain` variant fails to compile until handled here.
      // The `Encoded` arm is empty (control continues into the encoded dispatch
      // below); `Linear` and `Premultiplied` return. See the Yuv420p impl.
      //
      // `need_output` gates BOTH the averaging-domain freeze here and the
      // native/row-stage route freeze below. The domain freeze is CHECK-ONLY
      // here — the matching SET happens AFTER the selected path accepts an
      // output-bearing row (mirroring `frozen_native_route` below), never
      // before dispatch, so a rejected row leaves the freeze unchanged and a
      // corrected-domain retry of the SAME row is not falsely rejected. See the
      // Yuv420p impl for the full CHECK-before / SET-after rationale.
      let need_output =
        luma.is_some() || luma_u16.is_some() || rgb.is_some() || rgba.is_some() || hsv.is_some();
      if need_output
        && let Some(frozen) = *frozen_domain
        && frozen != *averaging_domain
      {
        return Err(MixedSinkerError::AveragingDomainChanged(
          AveragingDomainChanged::new(idx),
        ));
      }
      // RFC #238 S1: freeze the effective 4:2:2 chroma siting on the first
      // output-bearing row, mirroring the averaging-domain freeze above. This
      // CHECK is at the always-compiled choke point (every domain / tier passes
      // through it); the matching SET rides each domain's accept-time freeze
      // below (never before dispatch, so a rejected row leaves it unset for a
      // corrected retry). A later row observing a different phase — in sequence
      // or not — would bin a mixture of co-sited and centered chroma, so it is
      // rejected here before any reconstruction.
      if need_output
        && let Some(frozen) = *frozen_chroma_centered
        && frozen != center_sited
      {
        return Err(MixedSinkerError::ChromaSitingChanged(
          ChromaSitingChanged::new(idx),
        ));
      }
      match *averaging_domain {
        AveragingDomain::Encoded => {}
        // Under `rgb` it runs the linear tail (which itself rejects a filter
        // plan); without `rgb` it returns the typed `LinearDomainUnsupported`.
        AveragingDomain::Linear => {
          #[cfg(feature = "rgb")]
          {
            let tf = transfer_function.unwrap_or_else(|| TransferFunction::for_matrix(matrix));
            // Dispatch first; commit the domain freeze to Linear ONLY when the
            // tail accepts an output-bearing row (`r.is_ok() && need_output`).
            // A no-output call returns Ok without consuming; a filter /
            // out-of-sequence / output-changed / alloc reject returns Err — so
            // a rejected row leaves `frozen_domain` unset for a corrected retry.
            // See the Yuv420p impl.
            // RFC #238 S1 — centered Linear must fold the correct chroma
            // PHASE, not silently co-site. For the centered group, reconstruct
            // full-width chroma (CODE domain) and feed the 4:4:4 decode (both the
            // clamped u8 and the scene-referred f32 twin); co-sited Linear keeps
            // today's byte-identical decode. The chroma reserve runs AFTER the
            // Linear preflight's first-row rejects — a filter plan or an
            // out-of-sequence FIRST row, the only rejects that can fire on the
            // frame's one allocating row (frozen-output/transfer/mode never
            // reject a first row; later rows find `chroma_full` sized → a no-op
            // reserve) — so a rejected row allocates nothing (#180).
            // `linear_light_resample` re-runs the full preflight before it
            // commits. Gated on `want_color`: a luma-only Linear row decodes no
            // chroma, so it must not reserve/reconstruct.
            let r = if center_sited && want_color {
              // Run the FULL Linear preflight as a check BEFORE reserving chroma:
              // an unsupported filter plan, a mid-frame output-set change (e.g.
              // attaching RGB/HSV to a previously luma-only frame, flipping
              // `want_color` true while `chroma_full` is empty), a transfer /
              // mode change, or an out-of-sequence row must all be rejected
              // before the first allocation (#180). `linear_light_resample`
              // re-runs the same preflight and owns the transactional commit.
              linear_light::linear_light_preflight(
                linear_light_frame,
                resample_outputs,
                luma,
                luma_u16,
                rgb,
                rgba,
                hsv,
                tf,
                *linear_mode,
                plan,
                idx,
              )?;
              reserve_420_chroma_full(chroma_full, w, h)?;
              let (u_full, v_full) =
                reconstruct_chroma(ChromaU8, chroma_full, row.u_half(), row.v_half(), w);
              linear_light::linear_light_resample(
                linear_light_frame,
                resample_outputs,
                rgb,
                rgba,
                luma,
                luma_u16,
                hsv,
                rgb_scratch,
                linear_scene_scratch,
                tf,
                *linear_mode,
                plan,
                row.y(),
                idx,
                w,
                h,
                use_simd,
                |_idx, dst| {
                  yuv_444_to_rgb_row(
                    row.y(),
                    u_full,
                    v_full,
                    dst,
                    w,
                    matrix,
                    full_range,
                    use_simd,
                  );
                },
                |_idx, dst| {
                  crate::row::scalar::yuv_444_to_rgb_f32_unclamped_row(
                    row.y(),
                    u_full,
                    v_full,
                    dst,
                    w,
                    matrix,
                    full_range,
                  );
                },
              )
            } else {
              linear_light::linear_light_resample(
                linear_light_frame,
                resample_outputs,
                rgb,
                rgba,
                luma,
                luma_u16,
                hsv,
                rgb_scratch,
                linear_scene_scratch,
                tf,
                *linear_mode,
                plan,
                row.y(),
                idx,
                w,
                h,
                use_simd,
                |_idx, dst| convert_rgb(dst),
                |_idx, dst| convert_rgb_unclamped(dst),
              )
            };
            if r.is_ok() && need_output && frozen_domain.is_none() {
              *frozen_domain = Some(AveragingDomain::Linear);
            }
            // RFC #238 S1: freeze the siting on the same accepted output-bearing
            // row as the domain (never on a reject, so a corrected retry is not
            // falsely rejected as a mid-frame change).
            if r.is_ok() && need_output && frozen_chroma_centered.is_none() {
              *frozen_chroma_centered = Some(center_sited);
            }
            return r;
          }
          #[cfg(not(feature = "rgb"))]
          {
            return Err(plan.linear_domain_unsupported().into());
          }
        }
        // This format has no alpha plane, so premultiplied weighting is a
        // category error — reject with a typed error rather than silently
        // running the encoded average. Phase 5 wires Premultiplied for alpha
        // formats; on these non-alpha formats rejecting is correct.
        AveragingDomain::Premultiplied => {
          return Err(plan.premultiplied_domain_unsupported().into());
        }
      }
      if plan.kind().is_filter() {
        // Reject a multi-kernel (BICUBLIN) filter plan BEFORE the centered
        // preflight/reserve below: `planar_dual_filter_resample`'s FIRST act is
        // this same check, so without hoisting it a BICUBLIN plan would freeze
        // the output set and reserve/reconstruct chroma before being rejected as
        // `UnsupportedFilter` (the #180 reject-before-allocation invariant).
        // Idempotent — the delegate re-runs it.
        plan.ensure_single_kernel_filter()?;
        if center_sited && want_color {
          // Centered filter: reconstruct full-width chroma, but ONLY after the
          // resample preflight (frozen-output + sequence), so an out-of-sequence
          // / rejected row is caught before the chroma reservation (#180).
          // `planar_dual_filter_resample` re-runs the idempotent preflight.
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
          let (u_full, v_full) =
            reconstruct_chroma(ChromaU8, chroma_full, row.u_half(), row.v_half(), w);
          let r = planar_dual_filter_resample(
            luma_filter_stream,
            rgb_filter_stream,
            resample_outputs,
            rgb,
            rgba,
            luma,
            luma_u16,
            hsv,
            rgb_scratch,
            row.y(),
            w,
            plan,
            idx,
            use_simd,
            |scratch| {
              yuv_444_to_rgb_row(
                row.y(),
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
          // Filter tier freezes the domain + siting on the same accepted
          // output-bearing row as the native / row-stage arms (a rejected row
          // leaves both unset for a corrected retry).
          if r.is_ok() && need_output {
            if frozen_domain.is_none() {
              *frozen_domain = Some(AveragingDomain::Encoded);
            }
            if frozen_chroma_centered.is_none() {
              *frozen_chroma_centered = Some(center_sited);
            }
          }
          return r;
        }
        let r = planar_dual_filter_resample(
          luma_filter_stream,
          rgb_filter_stream,
          resample_outputs,
          rgb,
          rgba,
          luma,
          luma_u16,
          hsv,
          rgb_scratch,
          row.y(),
          w,
          plan,
          idx,
          use_simd,
          convert_rgb,
        );
        if r.is_ok() && need_output {
          if frozen_domain.is_none() {
            *frozen_domain = Some(AveragingDomain::Encoded);
          }
          if frozen_chroma_centered.is_none() {
            *frozen_chroma_centered = Some(center_sited);
          }
        }
        return r;
      }
      // Native / row-stage route split — see the Yuv420p impl for the
      // CHECK-before / SET-after `frozen_native_route` contract. Reuses the
      // `need_output` computed for the domain freeze above.
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
          native_eligible: YUV_PLANAR_8BIT_NATIVE_ELIGIBLE,
          with_native: *native,
          area_plan: true,
        },
      );
      match insertion {
        InsertionPoint::NativeCodes => {
          // 4:2:2: chroma `w/2 x h` — half width, full height; a chroma row
          // per Y row (`chroma_vsub = 1`), chroma plan a phased `area_chroma_422`.
          // RFC #238 point-of-use siting invalidation: `chroma_location` can
          // change at ANY point before this row — including AFTER `begin_frame`,
          // before row 0 — so re-check the cached join HERE (its single reuse
          // site) and drop it when its folded chroma plan was built for a
          // different phase; `yuv_planar_process_native` then rebuilds it with
          // the current siting. A no-output sink never built the join
          // (`native_planar` is `None` → no-op) and a luma-only join carries no
          // chroma plan (`chroma.is_none()` → siting-independent).
          //
          // Gated on `y.next_y() == 0` — the drop fires ONLY at a fresh-frame
          // boundary (`begin_frame` having reset the streams). A phase change is
          // frame-constant, so mid-frame (`next_y > 0`) NEVER drops the cached
          // stream: a rejected mid-frame row (incl. a mid-frame phase change)
          // therefore mutates no stream state and stays retryable — the
          // `yuv_planar_process_native` preflight below rejects it against the
          // intact cached join, so a corrected retry resumes on the right row.
          // RFC #238 point-of-use siting invalidation — RETRY-ATOMIC: drop the
          // stale-phase join ONLY on the IN-SEQUENCE first row of a fresh frame
          // (`idx == 0`, `next_y() == 0`). An out-of-sequence first row after a
          // siting change is left for `yuv_planar_process_native` to reject
          // against the INTACT join, so a corrected retry of row 0 rebuilds
          // cleanly (never a needless drop → AllocationFailed / lost reuse). A
          // mid-frame row (`next_y > 0`) never drops (the phase is
          // frame-constant); `resample_outputs` is `None` on a fresh frame, so
          // there is no committed output set to validate before the drop.
          // Transactional phase-rebuild: if the cached join is the wrong phase
          // (in-sequence fresh-frame row), move it OUT and let the delegate build
          // the replacement into `native_planar` (its `NativePlanarYuv::new` /
          // `try_box` run BEFORE it inserts, so a build failure leaves the field
          // `None`). On such a failure, restore the intact prior-phase join so the
          // REJECTED row mutates nothing — state-atomic under allocation failure.
          let stale_native = idx == 0
            && native_planar.as_ref().is_some_and(|join| {
              join.chroma.is_some() && join.chroma_centered != center_sited && join.y.next_y() == 0
            });
          let prev_native = if stale_native {
            native_planar.take()
          } else {
            None
          };
          // With a stale-phase join taken above, `native_planar` is `None` and the
          // delegate builds the replacement; it keeps the field untouched and
          // `resample_outputs` uncommitted until every pre-feed allocation
          // succeeds, so the arm restores ONLY this taken join on a rejected
          // rebuild. A first-ever build or a colour-capability rebuild is handled
          // entirely inside the delegate — the arm took nothing and restores
          // nothing for those.
          let native_result = yuv_planar_process_native(
            plan,
            native_planar,
            resample_outputs,
            rgb,
            rgba,
            luma,
            luma_u16,
            hsv,
            rgb_scratch,
            row.y(),
            row.u_half(),
            row.v_half(),
            matrix,
            full_range,
            idx,
            w,
            h,
            1,
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
          // rejected at any pre-feed step (build / box / output-scratch reserve):
          // the delegate leaves the field `None` and `resample_outputs`
          // uncommitted on such a failure, so restoring the intact prior-phase join
          // leaves the rejected row mutating nothing — state-atomic under
          // allocation failure. A non-stale row (first-ever or colour-capability
          // rebuild) took nothing, so there is nothing to restore.
          if stale_native && native_result.is_err() {
            *native_planar = prev_native;
          }
          native_result?;
          if frozen_native_route.is_none() && need_output {
            *frozen_native_route = Some(true);
          }
          // Encoded domain committed alongside the route, on the same accepted
          // output-bearing row (the `?` above already returned any reject).
          if frozen_domain.is_none() && need_output {
            *frozen_domain = Some(AveragingDomain::Encoded);
          }
          // RFC #238 S1: freeze the siting on the same accepted output row.
          if frozen_chroma_centered.is_none() && need_output {
            *frozen_chroma_centered = Some(center_sited);
          }
          return Ok(());
        }
        InsertionPoint::EncodedOutput => {
          // HSV-only (no RGB / RGBA): bin Y / U / V and convert per output
          // row, RGB-free — bit-identical to the native fast tier. 4:2:2
          // chroma is half-width, full height (`chroma_vsub = 1`); the
          // chroma plan matches the native tier's. Still the ROW-STAGE route
          // (`frozen_native_route = false`).
          if hsv.is_some() && rgb.is_none() && rgba.is_none() {
            // RFC #238 point-of-use siting invalidation (see the native arm):
            // drop the HSV-only join if its cached chroma plan was built for a
            // different phase than the current `chroma_location`, so
            // `hsv_direct_resample` rebuilds it with the current siting. Gated on
            // `next_y() == 0` — fires ONLY at a fresh-frame boundary, so a
            // mid-frame phase change never drops the cached stream (the
            // `hsv_direct_resample` preflight rejects the row against the intact
            // join → a corrected retry still resumes).
            // Retry-atomic (see the native arm): drop the stale-phase join ONLY
            // on the IN-SEQUENCE fresh-frame first row (`idx == 0`). An
            // out-of-sequence first row after a siting change is left for
            // `hsv_direct_resample` to reject against the INTACT join, so a
            // corrected retry of row 0 rebuilds cleanly.
            // Transactional phase-rebuild (see the native arm): move the stale
            // join OUT, let the delegate build the replacement, and restore the
            // intact prior-phase join on a build failure so the REJECTED row
            // mutates nothing (state-atomic under allocation failure).
            let stale_hsv = idx == 0
              && hsv_planar
                .as_ref()
                .is_some_and(|join| join.chroma_centered() != center_sited && join.next_y() == 0);
            // With a stale-phase join taken above, the delegate rebuilds it; it owns
            // the freeze + build atomicity (field untouched, `resample_outputs`
            // uncommitted on a pre-feed failure), so the arm restores ONLY this
            // taken join on a rejected rebuild (see the native arm). A first-ever
            // build is handled entirely inside the delegate.
            let prev_hsv = if stale_hsv { hsv_planar.take() } else { None };
            let hsv_result = super::planar_resample::hsv_direct_resample(
              hsv_planar,
              resample_outputs,
              luma,
              luma_u16,
              hsv,
              row.y(),
              row.u_half(),
              row.v_half(),
              matrix,
              full_range,
              1,
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
              w,
              plan,
              idx,
              use_simd,
            );
            // Restore the taken stale-phase join if the delegate's rebuild was
            // rejected (see the native arm): the delegate leaves the field `None`
            // and `resample_outputs` uncommitted on a pre-feed failure, so this is
            // the only state the arm must undo. A non-stale first-ever build took
            // nothing.
            if stale_hsv && hsv_result.is_err() {
              *hsv_planar = prev_hsv;
            }
            hsv_result?;
          } else if center_sited && want_color {
            // Centered RGB/RGBA: reconstruct full-width chroma AFTER the
            // resample preflight (frozen-output + sequence), so an
            // out-of-sequence / rejected row is caught before the chroma
            // reservation (#180). Gated by `want_color` — a luma-only centered
            // row never calls the RGB converter, so it must NOT reserve /
            // reconstruct chroma (it falls to the co-sited arm, which only bins
            // luma). `planar_dual_resample` re-runs the idempotent preflight.
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
            let (u_full, v_full) =
              reconstruct_chroma(ChromaU8, chroma_full, row.u_half(), row.v_half(), w);
            planar_dual_resample(
              luma_stream,
              rgb_stream,
              resample_outputs,
              rgb,
              rgba,
              luma,
              luma_u16,
              hsv,
              rgb_scratch,
              row.y(),
              w,
              plan,
              idx,
              use_simd,
              |scratch| {
                yuv_444_to_rgb_row(
                  row.y(),
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
          } else {
            planar_dual_resample(
              luma_stream,
              rgb_stream,
              resample_outputs,
              rgb,
              rgba,
              luma,
              luma_u16,
              hsv,
              rgb_scratch,
              row.y(),
              w,
              plan,
              idx,
              use_simd,
              convert_rgb,
            )?;
          }
          if frozen_native_route.is_none() && need_output {
            *frozen_native_route = Some(false);
          }
          // Encoded domain committed alongside the route, on the same accepted
          // output-bearing row (the `?` above already returned any reject).
          if frozen_domain.is_none() && need_output {
            *frozen_domain = Some(AveragingDomain::Encoded);
          }
          // RFC #238 S1: freeze the siting on the same accepted output row.
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

    // Strategy A output mode resolution — see Yuv420p impl above. The default /
    // co-sited path reuses the Yuv420p RGB / RGBA / HSV dispatchers (4:2:2's
    // per-row chroma contract is identical — half-width chroma, one pair per Y
    // pair); the centered horizontal sitings (#302) reuse the 4:4:4 dispatchers
    // after a phase-0.5 chroma upsample.
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();
    let want_color = want_rgb || want_rgba || want_hsv;

    // Repo-wide no-output invariant: a `process` call carrying NO output — no
    // colour, no luma, no luma_u16 — must run NOTHING: no per-row offset
    // arithmetic, no allocation, no state mutation. Returning HERE, before the
    // `idx * w` offsets below, also keeps the invariant overflow-safe — a
    // no-output call never ran an attach-time `w x h x 1` validation, so
    // `idx * w` could overflow `usize` on a 32-bit target with absurd geometry;
    // the guard skips that math (and the centered chroma reservation) entirely.
    // Mirrors the Yuv420p sibling.
    let need_output = want_color || luma.is_some() || luma_u16.is_some();
    if !need_output {
      return Ok(());
    }

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // Chroma siting (#302): the centered horizontal sitings reconstruct chroma at
    // the phase-0.5 position then decode via the 4:4:4 kernels; the default /
    // co-sited path keeps the byte-identical nearest-neighbor decode. 4:2:2 is
    // subsampled horizontally only — there is no vertical blend or chroma
    // lookback (cf. the Yuv420p `Bottom` path).
    let center_sited = chroma_422_center_sited_h(chroma_location);

    // Per-frame chroma-siting freeze (RFC #238, mirroring the resample-path guard
    // above): the first output-bearing row pins the phase; a later row whose siting
    // flipped would decode a mixture of centered and co-sited chroma into ONE
    // frame, so reject it here BEFORE any scratch reserve or output write.
    // `begin_frame` clears the freeze so the next frame may pick either phase.
    if need_output
      && let Some(frozen) = *frozen_chroma_centered
      && frozen != center_sited
    {
      return Err(MixedSinkerError::ChromaSitingChanged(
        ChromaSitingChanged::new(idx),
      ));
    }

    // Atomicity preflight (#302 / #308, cf. the crate's #180 resample fix and the
    // Yuv420p sibling): reserve EVERY fallible row scratch this row needs BEFORE
    // any output row (luma / luma_u16 included) is written, so an allocator
    // refusal returns a typed `AllocationFailed` leaving the output frame
    // untouched rather than partially mutated. Two scratches can grow:
    //  1. the centered-siting full-width chroma (`chroma_full`), only on a colour
    //     decode; and
    //  2. the RGB row buffer, reserved exactly when a colour decode needs an RGB
    //     row but no caller RGB buffer is borrowable — `want_hsv && want_rgba &&
    //     !want_rgb` (`rgb_row_buf_or_scratch`'s own scratch arm; an attached RGB
    //     buffer is borrowed instead and never allocates).
    // The later `reconstruct_chroma` / `rgb_row_buf_or_scratch` calls
    // reuse the already-sized buffers, so the default path is byte-identical;
    // only the failure-path ordering changes.
    if center_sited && want_color {
      reserve_420_chroma_full(chroma_full, w, h)?;
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
    if need_output && frozen_chroma_centered.is_none() {
      *frozen_chroma_centered = Some(center_sited);
    }

    if let Some(luma) = luma.as_deref_mut() {
      luma[one_plane_start..one_plane_end].copy_from_slice(&row.y()[..w]);
    }

    // Luma u16 — zero-extend the 8-bit Y plane into u16.
    if let Some(buf) = luma_u16.as_deref_mut() {
      crate::row::y_plane_to_luma_u16_row(
        row.y(),
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    // HSV-only (no RGB / RGBA) goes direct through `yuv_420_to_hsv_row` (centered:
    // `yuv_444_to_hsv_row` on the upsampled chroma) — see the Yuv420p impl for the
    // routing rationale.
    let want_hsv_direct = want_hsv && !want_rgb && !want_rgba;
    let need_rgb_kernel = want_rgb || (want_hsv && want_rgba);

    if want_hsv_direct {
      let hsv = hsv.as_mut().expect("want_hsv_direct implies hsv attached");
      let (h, s, v) = hsv.hsv();
      if center_sited {
        let (u_full, v_full) =
          reconstruct_chroma(ChromaU8, chroma_full, row.u_half(), row.v_half(), w);
        yuv_444_to_hsv_row(
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
        );
      } else {
        yuv_420_to_hsv_row(
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
        );
      }
      return Ok(());
    }

    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      if center_sited {
        let (u_full, v_full) =
          reconstruct_chroma(ChromaU8, chroma_full, row.u_half(), row.v_half(), w);
        yuv_444_to_rgba_row(
          row.y(),
          u_full,
          v_full,
          rgba_row,
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
        );
      } else {
        yuv_420_to_rgba_row(
          row.y(),
          row.u_half(),
          row.v_half(),
          rgba_row,
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
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

    // Centered siting (#302) upsamples chroma to full width (phase-0.5) then runs
    // the 4:4:4 kernel; otherwise the fused 4:2:0 dispatcher upsamples chroma
    // in-register (4:2:2's per-row contract is identical — half-width chroma, one
    // pair per Y pair).
    if center_sited {
      let (u_full, v_full) =
        reconstruct_chroma(ChromaU8, chroma_full, row.u_half(), row.v_half(), w);
      yuv_444_to_rgb_row(
        row.y(),
        u_full,
        v_full,
        rgb_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
    } else {
      yuv_420_to_rgb_row(
        row.y(),
        row.u_half(),
        row.v_half(),
        rgb_row,
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

    if let Some(buf) = rgba.as_deref_mut() {
      let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
    }

    Ok(())
  }
}

// ---- Yuv444p impl -------------------------------------------------------
//
// 4:4:4 planar: U and V are full-width, full-height. No width parity
// constraint. Uses the `yuv_444_to_rgb_row` / `yuv_444_to_rgba_row`
// kernel family.

impl<'a, R> MixedSinker<'a, Yuv444p, R> {
  /// Attaches a packed 32‑bit RGBA output buffer.
  ///
  /// Only available on sinker types whose `PixelSink` impl writes
  /// RGBA — see [`MixedSinker::<Yuv420p>::with_rgba`] for the same
  /// rationale and constraints. Yuv444p has no alpha plane, so every
  /// alpha byte is filled with `0xFF` (opaque).
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

  /// Attaches a `u16` luma output buffer. The 8-bit Y plane samples
  /// are zero-extended into `u16`. Length is measured in `u16`
  /// elements (`width x height`).
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

impl<R> Yuv444pSink for MixedSinker<'_, Yuv444p, R> {}

impl<R> PixelSink for MixedSinker<'_, Yuv444p, R> {
  type Input<'r> = Yuv444pRow<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
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
    if let Some(native) = self.native_planar.as_mut() {
      native.reset();
    }
    // New frame: restart the RGB-free HSV-only row-stage join (#263 follow-up).
    if let Some(hsv) = self.hsv_planar.as_mut() {
      hsv.reset();
    }
    self.frozen_native_route = None;
    self.frozen_domain = None;
    self.resample_outputs = None;
    // New frame: drop the RFC #238 linear-light accumulator (if any).
    #[cfg(feature = "rgb")]
    {
      self.linear_light_frame = None;
    }
    Ok(())
  }

  fn process(&mut self, row: Yuv444pRow<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Y,
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
      rgba,
      luma,
      luma_u16,
      hsv,
      rgb_scratch,
      plan,
      rgb_stream,
      luma_stream,
      rgb_filter_stream,
      luma_filter_stream,
      resample_outputs,
      native,
      native_planar,
      hsv_planar,
      frozen_native_route,
      frozen_domain,
      averaging_domain,
      #[cfg(feature = "rgb")]
      linear_mode,
      #[cfg(feature = "rgb")]
      linear_light_frame,
      #[cfg(feature = "rgb")]
      linear_scene_scratch,
      #[cfg(feature = "rgb")]
      transfer_function,
      ..
    } = self;

    // Non-identity plan. A `Filter` plan converts each source row to a
    // source-width RGB row (the same 4:4:4 kernel the identity path uses)
    // and filter-resamples it plus the native Y. An `Area` plan picks the
    // native fast tier (bin Y / U / V to output res, convert once at output
    // width) or the row-stage tier (convert each source row, bin RGB) per
    // `with_native`, frozen per frame. Both tiers freeze the output set and
    // check stream sequencing before staging, so a no-output sink stays a
    // no-op and an out-of-sequence row is rejected without allocating.
    if let Some(plan) = plan.as_ref() {
      let matrix = row.matrix();
      let full_range = row.full_range();
      let convert_rgb = |scratch: &mut [u8]| {
        yuv_444_to_rgb_row(
          row.y(),
          row.u(),
          row.v(),
          scratch,
          w,
          matrix,
          full_range,
          use_simd,
        );
      };
      // The scene-referred ([`LinearMode::SceneReferred`]) twin of
      // `convert_rgb`: the SAME affine matrix, real-valued and unclamped.
      // Consumed only by the `rgb`-gated Linear arm.
      #[cfg(feature = "rgb")]
      let convert_rgb_unclamped = |dst: &mut [f32]| {
        crate::row::scalar::yuv_444_to_rgb_f32_unclamped_row(
          row.y(),
          row.u(),
          row.v(),
          dst,
          w,
          matrix,
          full_range,
        );
      };
      // RFC #238 Phase 2 — single always-compiled choke point for the averaging
      // domain, BEFORE any filter / native / row-stage branching, so a
      // non-encoded sink never falls through to the Encoded path under any
      // feature combination. The match is EXHAUSTIVE with no wildcard arm: a
      // future `AveragingDomain` variant fails to compile until handled here.
      // The `Encoded` arm is empty (control continues into the encoded dispatch
      // below); `Linear` and `Premultiplied` return. See the Yuv420p impl.
      //
      // `need_output` gates BOTH the averaging-domain freeze here and the
      // native/row-stage route freeze below. The domain freeze is CHECK-ONLY
      // here — the matching SET happens AFTER the selected path accepts an
      // output-bearing row (mirroring `frozen_native_route` below), never
      // before dispatch, so a rejected row leaves the freeze unchanged and a
      // corrected-domain retry of the SAME row is not falsely rejected. See the
      // Yuv420p impl for the full CHECK-before / SET-after rationale.
      let need_output =
        luma.is_some() || luma_u16.is_some() || rgb.is_some() || rgba.is_some() || hsv.is_some();
      if need_output
        && let Some(frozen) = *frozen_domain
        && frozen != *averaging_domain
      {
        return Err(MixedSinkerError::AveragingDomainChanged(
          AveragingDomainChanged::new(idx),
        ));
      }
      match *averaging_domain {
        AveragingDomain::Encoded => {}
        // Under `rgb` it runs the linear tail (which itself rejects a filter
        // plan); without `rgb` it returns the typed `LinearDomainUnsupported`.
        AveragingDomain::Linear => {
          #[cfg(feature = "rgb")]
          {
            let tf = transfer_function.unwrap_or_else(|| TransferFunction::for_matrix(matrix));
            // Dispatch first; commit the domain freeze to Linear ONLY when the
            // tail accepts an output-bearing row (`r.is_ok() && need_output`).
            // A no-output call returns Ok without consuming; a filter /
            // out-of-sequence / output-changed / alloc reject returns Err — so
            // a rejected row leaves `frozen_domain` unset for a corrected retry.
            // See the Yuv420p impl.
            let r = linear_light::linear_light_resample(
              linear_light_frame,
              resample_outputs,
              rgb,
              rgba,
              luma,
              luma_u16,
              hsv,
              rgb_scratch,
              linear_scene_scratch,
              tf,
              *linear_mode,
              plan,
              row.y(),
              idx,
              w,
              h,
              use_simd,
              |_idx, dst| convert_rgb(dst),
              |_idx, dst| convert_rgb_unclamped(dst),
            );
            if r.is_ok() && need_output && frozen_domain.is_none() {
              *frozen_domain = Some(AveragingDomain::Linear);
            }
            return r;
          }
          #[cfg(not(feature = "rgb"))]
          {
            return Err(plan.linear_domain_unsupported().into());
          }
        }
        // This format has no alpha plane, so premultiplied weighting is a
        // category error — reject with a typed error rather than silently
        // running the encoded average. Phase 5 wires Premultiplied for alpha
        // formats; on these non-alpha formats rejecting is correct.
        AveragingDomain::Premultiplied => {
          return Err(plan.premultiplied_domain_unsupported().into());
        }
      }
      if plan.kind().is_filter() {
        return planar_dual_filter_resample(
          luma_filter_stream,
          rgb_filter_stream,
          resample_outputs,
          rgb,
          rgba,
          luma,
          luma_u16,
          hsv,
          rgb_scratch,
          row.y(),
          w,
          plan,
          idx,
          use_simd,
          convert_rgb,
        );
      }
      // Native / row-stage route split — see the Yuv420p impl for the
      // CHECK-before / SET-after `frozen_native_route` contract. Reuses the
      // `need_output` computed for the domain freeze above.
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
          native_eligible: YUV_PLANAR_8BIT_NATIVE_ELIGIBLE,
          with_native: *native,
          area_plan: true,
        },
      );
      match insertion {
        InsertionPoint::NativeCodes => {
          // 4:4:4: chroma `w x h` — identical to Y; a chroma row per Y row
          // (`chroma_vsub = 1`), chroma plan equals the luma plan.
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
            row.y(),
            row.u(),
            row.v(),
            matrix,
            full_range,
            idx,
            w,
            h,
            1,
            || ResamplePlan::area(w, h, plan.out_w(), plan.out_h()),
            use_simd,
          )?;
          if frozen_native_route.is_none() && need_output {
            *frozen_native_route = Some(true);
          }
          // Encoded domain committed alongside the route, on the same accepted
          // output-bearing row (the `?` above already returned any reject).
          if frozen_domain.is_none() && need_output {
            *frozen_domain = Some(AveragingDomain::Encoded);
          }
          return Ok(());
        }
        InsertionPoint::EncodedOutput => {
          // HSV-only (no RGB / RGBA): bin Y / U / V and convert per output
          // row, RGB-free — bit-identical to the native fast tier. 4:4:4
          // chroma is full-width, full height (`chroma_vsub = 1`); the chroma
          // plan matches the native tier's. Still the ROW-STAGE route.
          if hsv.is_some() && rgb.is_none() && rgba.is_none() {
            super::planar_resample::hsv_direct_resample(
              hsv_planar,
              resample_outputs,
              luma,
              luma_u16,
              hsv,
              row.y(),
              row.u(),
              row.v(),
              matrix,
              full_range,
              1,
              || ResamplePlan::area(w, h, plan.out_w(), plan.out_h()),
              w,
              plan,
              idx,
              use_simd,
            )?;
          } else {
            planar_dual_resample(
              luma_stream,
              rgb_stream,
              resample_outputs,
              rgb,
              rgba,
              luma,
              luma_u16,
              hsv,
              rgb_scratch,
              row.y(),
              w,
              plan,
              idx,
              use_simd,
              convert_rgb,
            )?;
          }
          if frozen_native_route.is_none() && need_output {
            *frozen_native_route = Some(false);
          }
          // Encoded domain committed alongside the route, on the same accepted
          // output-bearing row (the `?` above already returned any reject).
          if frozen_domain.is_none() && need_output {
            *frozen_domain = Some(AveragingDomain::Encoded);
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

    // Strategy A output mode resolution — see Yuv420p impl above.
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();

    // Atomicity preflight (#308, cf. the crate's #180 resample fix and the
    // Yuv420p sibling): reserve EVERY fallible row scratch this row needs
    // BEFORE any output row (luma / luma_u16 included) is written, so an
    // allocator refusal returns a typed `AllocationFailed` leaving the output
    // frame untouched rather than partially mutated. The only growable scratch
    // here is the RGB row buffer, reserved exactly when a colour decode needs
    // an RGB row but no caller RGB buffer is borrowable — `want_hsv &&
    // want_rgba && !want_rgb` (`rgb_row_buf_or_scratch`'s own scratch arm; an
    // attached RGB buffer is borrowed instead and never allocates). The later
    // decode call then reuses the already-sized buffer.
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
      luma[one_plane_start..one_plane_end].copy_from_slice(&row.y()[..w]);
    }

    // Luma u16 — zero-extend the 8-bit Y plane into u16.
    if let Some(buf) = luma_u16.as_deref_mut() {
      crate::row::y_plane_to_luma_u16_row(
        row.y(),
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    // HSV-only (no RGB / RGBA) goes direct through `yuv_444_to_hsv_row`
    // — see the Yuv420p impl for the routing rationale.
    let want_hsv_direct = want_hsv && !want_rgb && !want_rgba;
    let need_rgb_kernel = want_rgb || (want_hsv && want_rgba);

    if want_hsv_direct {
      let hsv = hsv.as_mut().expect("want_hsv_direct implies hsv attached");
      let (h, s, v) = hsv.hsv();
      yuv_444_to_hsv_row(
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
      );
      return Ok(());
    }

    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      yuv_444_to_rgba_row(
        row.y(),
        row.u(),
        row.v(),
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

    yuv_444_to_rgb_row(
      row.y(),
      row.u(),
      row.v(),
      rgb_row,
      w,
      row.matrix(),
      row.full_range(),
      use_simd,
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

// ---- Yuv440p impl -------------------------------------------------------
//
// 4:4:0 planar 8‑bit — full-width chroma, half-height. Per-row math
// matches 4:4:4 (full-width U / V); only the walker reads chroma row
// `r / 2`. Reuses `yuv_444_to_rgb_row` and `yuv_444_to_rgba_row`
// verbatim.

impl<'a, R> MixedSinker<'a, Yuv440p, R> {
  /// Attaches a packed 32‑bit RGBA output buffer.
  ///
  /// See [`MixedSinker::<Yuv420p>::with_rgba`] for the rationale and
  /// constraints. Yuv440p has no alpha plane, so every alpha byte is
  /// filled with `0xFF` (opaque).
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

  /// Attaches a `u16` luma output buffer. The 8-bit Y plane samples
  /// are zero-extended into `u16`. Length is measured in `u16`
  /// elements (`width x height`).
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

impl<R> Yuv440pSink for MixedSinker<'_, Yuv440p, R> {}

impl<R> PixelSink for MixedSinker<'_, Yuv440p, R> {
  type Input<'r> = Yuv440pRow<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
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
    if let Some(native) = self.native_planar.as_mut() {
      native.reset();
    }
    // New frame: restart the RGB-free HSV-only row-stage join (#263 follow-up).
    if let Some(hsv) = self.hsv_planar.as_mut() {
      hsv.reset();
    }
    self.frozen_native_route = None;
    self.frozen_domain = None;
    self.resample_outputs = None;
    // New frame: clear the RFC #238 S8b frozen 4:4:0 chroma siting so the next
    // frame may pick either vertical phase; a mid-frame co-sited ⇆ `Bottom`
    // flip stays rejected. 4:4:0 folds only the VERTICAL axis (full-width
    // chroma → no horizontal phase), so `frozen_chroma_bottom_v` carries the
    // freeze; `frozen_chroma_centered` is cleared for parity (never set here).
    // (The cached native / HSV-only join's siting invalidation is done at
    // point-of-use in `process`, not here — `chroma_location` may change AFTER
    // `begin_frame` but before row 0.)
    self.frozen_chroma_centered = None;
    self.frozen_chroma_bottom_v = None;
    // New frame: invalidate the bottom-sited vertical-phase chroma lookback so
    // frame N+1's first even row can never box-blend frame N's last chroma row.
    // The buffer bytes are left as-is (re-overwritten before any trusted read);
    // only the validity tag is cleared.
    self.chroma_prev_row = None;
    // New frame: drop the RFC #238 linear-light accumulator (if any).
    #[cfg(feature = "rgb")]
    {
      self.linear_light_frame = None;
    }
    Ok(())
  }

  fn process(&mut self, row: Yuv440pRow<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;
    // Chroma siting (RFC #238 S8b): 4:4:0 carries only a VERTICAL phase
    // (full-width chroma → no horizontal siting). `Copy`, so read it out before
    // the field split-borrow below.
    let chroma_location = self.chroma_location;

    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Y,
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
      rgba,
      luma,
      luma_u16,
      hsv,
      rgb_scratch,
      chroma_full,
      chroma_prev,
      chroma_prev_row,
      plan,
      rgb_stream,
      luma_stream,
      rgb_filter_stream,
      luma_filter_stream,
      resample_outputs,
      native,
      native_planar,
      hsv_planar,
      frozen_native_route,
      frozen_domain,
      averaging_domain,
      #[cfg(feature = "rgb")]
      linear_mode,
      #[cfg(feature = "rgb")]
      linear_light_frame,
      #[cfg(feature = "rgb")]
      linear_scene_scratch,
      #[cfg(feature = "rgb")]
      transfer_function,
      // RFC #238 S8b: the 4:4:0 vertical (`Bottom`) chroma phase frozen on the
      // first output-bearing row. `frozen_chroma_centered` is destructured for
      // the shared clear in `begin_frame` but stays `None` here — 4:4:0 has no
      // horizontal phase.
      frozen_chroma_bottom_v,
      ..
    } = self;

    // Non-identity plan. A `Filter` plan converts each source row to a
    // source-width RGB row (the same `yuv_444_to_rgb_row` kernel the
    // identity path uses — 4:4:0's per-row math is identical to 4:4:4,
    // full-width chroma) and filter-resamples it plus the native Y. An
    // `Area` plan picks the native fast tier (bin Y / U / V to output res,
    // convert once at output width) or the row-stage tier (convert each
    // source row, bin RGB) per `with_native`, frozen per frame. Both tiers
    // freeze the output set and check stream sequencing before staging, so
    // a no-output sink stays a no-op and an out-of-sequence row is rejected
    // without allocating.
    if let Some(plan) = plan.as_ref() {
      let matrix = row.matrix();
      let full_range = row.full_range();
      // RFC #238 S8b — 4:4:0 VERTICAL chroma siting. `Bottom` / `BottomLeft`
      // ([`chroma_440_bottom_sited_v`], `v = 1`) box-blends the even output
      // row's full-width chroma with the previous chroma row; every other siting
      // keeps the co-sited vertical decode (byte-identical). 4:4:0 has FULL-width
      // chroma, so there is NO horizontal phase — only the vertical axis is ever
      // folded (`h_phase` stays 0). The binning tiers (native / HSV-only) fold
      // `v = 1` into `area_chroma_440`'s vertical weights; the RGB-domain
      // reconstruction tiers (row-stage RGB/RGBA, filter, linear-light) box-blend
      // full-width chroma via the `chroma_prev` lookback. Those reconstruction
      // arms build their OWN 4:4:4 converter over freshly reconstructed chroma
      // AFTER the resample preflight, so a rejected / out-of-sequence row is
      // caught before that chroma reservation (the #180 reserve-after-preflight
      // invariant).
      let bottom_v = chroma_440_bottom_sited_v(chroma_location);
      let chroma_v_phase = if bottom_v { 1.0 } else { 0.0 };
      let want_color = rgb.is_some() || rgba.is_some() || hsv.is_some();
      let convert_rgb = |scratch: &mut [u8]| {
        yuv_444_to_rgb_row(
          row.y(),
          row.u(),
          row.v(),
          scratch,
          w,
          matrix,
          full_range,
          use_simd,
        );
      };
      // The scene-referred ([`LinearMode::SceneReferred`]) twin of
      // `convert_rgb`: the SAME affine matrix, real-valued and unclamped.
      // Consumed only by the `rgb`-gated Linear arm.
      #[cfg(feature = "rgb")]
      let convert_rgb_unclamped = |dst: &mut [f32]| {
        crate::row::scalar::yuv_444_to_rgb_f32_unclamped_row(
          row.y(),
          row.u(),
          row.v(),
          dst,
          w,
          matrix,
          full_range,
        );
      };
      // RFC #238 Phase 2 — single always-compiled choke point for the averaging
      // domain, BEFORE any filter / native / row-stage branching, so a
      // non-encoded sink never falls through to the Encoded path under any
      // feature combination. The match is EXHAUSTIVE with no wildcard arm: a
      // future `AveragingDomain` variant fails to compile until handled here.
      // The `Encoded` arm is empty (control continues into the encoded dispatch
      // below); `Linear` and `Premultiplied` return. See the Yuv420p impl.
      //
      // `need_output` gates BOTH the averaging-domain freeze here and the
      // native/row-stage route freeze below. The domain freeze is CHECK-ONLY
      // here — the matching SET happens AFTER the selected path accepts an
      // output-bearing row (mirroring `frozen_native_route` below), never
      // before dispatch, so a rejected row leaves the freeze unchanged and a
      // corrected-domain retry of the SAME row is not falsely rejected. See the
      // Yuv420p impl for the full CHECK-before / SET-after rationale.
      let need_output =
        luma.is_some() || luma_u16.is_some() || rgb.is_some() || rgba.is_some() || hsv.is_some();
      if need_output
        && let Some(frozen) = *frozen_domain
        && frozen != *averaging_domain
      {
        return Err(MixedSinkerError::AveragingDomainChanged(
          AveragingDomainChanged::new(idx),
        ));
      }
      // RFC #238 S8b: freeze the effective 4:4:0 vertical chroma siting on the
      // first output-bearing row, mirroring the averaging-domain freeze above.
      // 4:4:0 folds only the vertical axis, so `frozen_chroma_bottom_v` alone is
      // the sentinel (co-sited freezes `Some(false)`, `Bottom` freezes
      // `Some(true)`). This CHECK is at the always-compiled choke point; the
      // matching SET rides each tier's accept-time freeze below (never before
      // dispatch, so a rejected row leaves it unset for a corrected retry). A
      // later row observing a different vertical phase would bin a mixture of
      // co-sited and bottom-folded chroma, so it is rejected here before any
      // reconstruction.
      if need_output
        && let Some(frozen) = *frozen_chroma_bottom_v
        && frozen != bottom_v
      {
        return Err(MixedSinkerError::ChromaSitingChanged(
          ChromaSitingChanged::new(idx),
        ));
      }
      match *averaging_domain {
        AveragingDomain::Encoded => {}
        // Under `rgb` it runs the linear tail (which itself rejects a filter
        // plan); without `rgb` it returns the typed `LinearDomainUnsupported`.
        AveragingDomain::Linear => {
          #[cfg(feature = "rgb")]
          {
            let tf = transfer_function.unwrap_or_else(|| TransferFunction::for_matrix(matrix));
            // Dispatch first; commit the domain freeze to Linear ONLY when the
            // tail accepts an output-bearing row (`r.is_ok() && need_output`).
            // A no-output call returns Ok without consuming; a filter /
            // out-of-sequence / output-changed / alloc reject returns Err — so
            // a rejected row leaves `frozen_domain` unset for a corrected retry.
            // See the Yuv420p impl.
            //
            // RFC #238 S8b — bottom-sited Linear must fold the correct VERTICAL
            // chroma phase, not silently co-site. For `Bottom` reconstruct
            // full-width chroma (CODE domain) and feed the 4:4:4 decode (both the
            // clamped u8 and scene-referred f32 twin); co-sited Linear keeps
            // today's byte-identical decode. The chroma reserve runs AFTER the
            // Linear preflight's first-row rejects (#180). Gated on `want_color`
            // — a luma-only Linear row decodes no chroma, so it must not reserve
            // / reconstruct.
            let r = if bottom_v && want_color {
              linear_light::linear_light_preflight(
                linear_light_frame,
                resample_outputs,
                luma,
                luma_u16,
                rgb,
                rgba,
                hsv,
                tf,
                *linear_mode,
                plan,
                idx,
              )?;
              reserve_420_chroma_full(chroma_full, w, h)?;
              reserve_440_chroma_prev(chroma_prev, w, h)?;
              let (u_full, v_full) = upsample_440_chroma_sited(
                chroma_full,
                chroma_prev,
                chroma_prev_row,
                row.u(),
                row.v(),
                idx,
                bottom_v,
                false,
                w,
              );
              linear_light::linear_light_resample(
                linear_light_frame,
                resample_outputs,
                rgb,
                rgba,
                luma,
                luma_u16,
                hsv,
                rgb_scratch,
                linear_scene_scratch,
                tf,
                *linear_mode,
                plan,
                row.y(),
                idx,
                w,
                h,
                use_simd,
                |_idx, dst| {
                  yuv_444_to_rgb_row(
                    row.y(),
                    u_full,
                    v_full,
                    dst,
                    w,
                    matrix,
                    full_range,
                    use_simd,
                  );
                },
                |_idx, dst| {
                  crate::row::scalar::yuv_444_to_rgb_f32_unclamped_row(
                    row.y(),
                    u_full,
                    v_full,
                    dst,
                    w,
                    matrix,
                    full_range,
                  );
                },
              )
            } else {
              linear_light::linear_light_resample(
                linear_light_frame,
                resample_outputs,
                rgb,
                rgba,
                luma,
                luma_u16,
                hsv,
                rgb_scratch,
                linear_scene_scratch,
                tf,
                *linear_mode,
                plan,
                row.y(),
                idx,
                w,
                h,
                use_simd,
                |_idx, dst| convert_rgb(dst),
                |_idx, dst| convert_rgb_unclamped(dst),
              )
            };
            // Bottom lookback: advance it only AFTER the Linear resample accepts
            // the row (mirroring the #180 chroma reserve ordering), so a rejected
            // final row leaves the predecessor in place for a clean retry — the
            // sited reconstruction read the lookback above but did not stage. A
            // luma-only Bottom row took the else arm (no reconstruction), so gate
            // on `want_color`.
            if r.is_ok() && bottom_v && want_color {
              stage_440_chroma_prev(chroma_prev, chroma_prev_row, row.u(), row.v(), idx, w);
            }
            if r.is_ok() && need_output && frozen_domain.is_none() {
              *frozen_domain = Some(AveragingDomain::Linear);
            }
            // RFC #238 S8b: freeze the siting on the same accepted output-bearing
            // row as the domain (never on a reject, so a corrected retry is not
            // falsely rejected as a mid-frame change).
            if r.is_ok() && need_output && frozen_chroma_bottom_v.is_none() {
              *frozen_chroma_bottom_v = Some(bottom_v);
            }
            return r;
          }
          #[cfg(not(feature = "rgb"))]
          {
            return Err(plan.linear_domain_unsupported().into());
          }
        }
        // This format has no alpha plane, so premultiplied weighting is a
        // category error — reject with a typed error rather than silently
        // running the encoded average. Phase 5 wires Premultiplied for alpha
        // formats; on these non-alpha formats rejecting is correct.
        AveragingDomain::Premultiplied => {
          return Err(plan.premultiplied_domain_unsupported().into());
        }
      }
      if plan.kind().is_filter() {
        if bottom_v && want_color {
          // Bottom single-kernel filter: reconstruct full-width chroma, but ONLY
          // after the resample preflight (frozen-output + sequence), so an
          // out-of-sequence / rejected row is caught before the chroma
          // reservation (#180). `planar_dual_filter_resample` re-runs the
          // idempotent preflight. Gated by `want_color` — a luma-only Bottom row
          // never calls the RGB converter, so it stays on the co-sited arm.
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
          reserve_440_chroma_prev(chroma_prev, w, h)?;
          let (u_full, v_full) = upsample_440_chroma_sited(
            chroma_full,
            chroma_prev,
            chroma_prev_row,
            row.u(),
            row.v(),
            idx,
            bottom_v,
            false,
            w,
          );
          let r = planar_dual_filter_resample(
            luma_filter_stream,
            rgb_filter_stream,
            resample_outputs,
            rgb,
            rgba,
            luma,
            luma_u16,
            hsv,
            rgb_scratch,
            row.y(),
            w,
            plan,
            idx,
            use_simd,
            |scratch| {
              yuv_444_to_rgb_row(
                row.y(),
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
          // Bottom lookback: advance only AFTER the filter resample accepts the
          // row, so a rejected row leaves the predecessor for a clean retry (the
          // sited reconstruction read it above but did not stage).
          if r.is_ok() && bottom_v {
            stage_440_chroma_prev(chroma_prev, chroma_prev_row, row.u(), row.v(), idx, w);
          }
          // Filter tier freezes the siting on the same accepted output-bearing
          // row as the native / row-stage arms (a rejected row leaves it unset
          // for a corrected retry).
          if r.is_ok() && need_output && frozen_chroma_bottom_v.is_none() {
            *frozen_chroma_bottom_v = Some(bottom_v);
          }
          return r;
        }
        let r = planar_dual_filter_resample(
          luma_filter_stream,
          rgb_filter_stream,
          resample_outputs,
          rgb,
          rgba,
          luma,
          luma_u16,
          hsv,
          rgb_scratch,
          row.y(),
          w,
          plan,
          idx,
          use_simd,
          convert_rgb,
        );
        if r.is_ok() && need_output && frozen_chroma_bottom_v.is_none() {
          *frozen_chroma_bottom_v = Some(bottom_v);
        }
        return r;
      }
      // Native / row-stage route split — see the Yuv420p impl for the
      // CHECK-before / SET-after `frozen_native_route` contract. Reuses the
      // `need_output` computed for the domain freeze above.
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
          native_eligible: YUV_PLANAR_8BIT_NATIVE_ELIGIBLE,
          with_native: *native,
          area_plan: true,
        },
      );
      match insertion {
        InsertionPoint::NativeCodes => {
          // 4:4:0: chroma `w x h/2` — full width, half height; a chroma row
          // per TWO Y rows (`chroma_vsub = 2`, like 4:2:0 vertically), chroma
          // plan full-width horizontal + luma-domain `area_halved` vertical
          // (or the RFC #238 S8b `Bottom` `v = 1` phased-V fold).
          //
          // RFC #238 S8b point-of-use siting invalidation (see the Yuv422p native
          // arm): `chroma_location` can change at ANY point before this row
          // (including AFTER `begin_frame`, before row 0), so re-check the cached
          // join HERE and drop it when its folded chroma plan was built for a
          // different vertical phase; `yuv_planar_process_native` rebuilds it with
          // the current siting. Retry-atomic: drop ONLY on the IN-SEQUENCE
          // fresh-frame first row (`idx == 0`, `next_y() == 0`); an out-of-sequence
          // first row after a siting change is left for the delegate to reject
          // against the INTACT join. Transactional: move the stale join OUT and let
          // the delegate build the replacement into `native_planar` (its build runs
          // BEFORE it inserts, so a build failure leaves the field `None`); on such
          // a failure, restore the intact prior-phase join so the REJECTED row
          // mutates nothing. A luma-only join carries no chroma plan
          // (siting-independent), a no-output sink built no join.
          let stale_native = idx == 0
            && native_planar.as_ref().is_some_and(|join| {
              join.chroma.is_some() && join.chroma_bottom != bottom_v && join.y.next_y() == 0
            });
          let prev_native = if stale_native {
            native_planar.take()
          } else {
            None
          };
          let native_result = yuv_planar_process_native(
            plan,
            native_planar,
            resample_outputs,
            rgb,
            rgba,
            luma,
            luma_u16,
            hsv,
            rgb_scratch,
            row.y(),
            row.u(),
            row.v(),
            matrix,
            full_range,
            idx,
            w,
            h,
            2,
            || ResamplePlan::area_chroma_440(w, h, plan.out_w(), plan.out_h(), 0.0, chroma_v_phase),
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
          // Encoded domain committed alongside the route, on the same accepted
          // output-bearing row (the `?` above already returned any reject).
          if frozen_domain.is_none() && need_output {
            *frozen_domain = Some(AveragingDomain::Encoded);
          }
          // RFC #238 S8b: freeze the siting on the same accepted output row.
          if frozen_chroma_bottom_v.is_none() && need_output {
            *frozen_chroma_bottom_v = Some(bottom_v);
          }
          return Ok(());
        }
        InsertionPoint::EncodedOutput => {
          // HSV-only (no RGB / RGBA): bin Y / U / V and convert per output
          // row, RGB-free — bit-identical to the native fast tier. 4:4:0
          // chroma is full-width, half height (`chroma_vsub = 2`); the chroma
          // plan matches the native tier's `area_chroma_440` (luma-domain
          // vertical weighting, plus the S8b `Bottom` phased-V fold). Still the
          // ROW-STAGE route.
          if hsv.is_some() && rgb.is_none() && rgba.is_none() {
            // RFC #238 S8b point-of-use siting invalidation (see the native arm):
            // drop the HSV-only join if its cached chroma plan was built for a
            // different vertical phase, so `hsv_direct_resample` rebuilds it with
            // the current siting. Retry-atomic (drop only on the in-sequence
            // fresh-frame first row) and transactional (move the stale join OUT,
            // let the delegate build the replacement, restore the intact join on a
            // rejected rebuild so the REJECTED row mutates nothing).
            let stale_hsv = idx == 0
              && hsv_planar
                .as_ref()
                .is_some_and(|join| join.chroma_bottom() != bottom_v && join.next_y() == 0);
            let prev_hsv = if stale_hsv { hsv_planar.take() } else { None };
            let hsv_result = super::planar_resample::hsv_direct_resample(
              hsv_planar,
              resample_outputs,
              luma,
              luma_u16,
              hsv,
              row.y(),
              row.u(),
              row.v(),
              matrix,
              full_range,
              2,
              || {
                ResamplePlan::area_chroma_440(w, h, plan.out_w(), plan.out_h(), 0.0, chroma_v_phase)
              },
              w,
              plan,
              idx,
              use_simd,
            );
            if stale_hsv && hsv_result.is_err() {
              *hsv_planar = prev_hsv;
            }
            hsv_result?;
          } else if bottom_v && want_color {
            // Bottom RGB/RGBA: reconstruct full-width chroma AFTER the resample
            // preflight (frozen-output + sequence), so an out-of-sequence /
            // rejected row is caught before the chroma reservation (#180). Gated
            // by `want_color` — a luma-only Bottom row never calls the RGB
            // converter, so it falls to the co-sited arm (which only bins luma).
            // `planar_dual_resample` re-runs the idempotent preflight.
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
            reserve_440_chroma_prev(chroma_prev, w, h)?;
            let (u_full, v_full) = upsample_440_chroma_sited(
              chroma_full,
              chroma_prev,
              chroma_prev_row,
              row.u(),
              row.v(),
              idx,
              bottom_v,
              false,
              w,
            );
            planar_dual_resample(
              luma_stream,
              rgb_stream,
              resample_outputs,
              rgb,
              rgba,
              luma,
              luma_u16,
              hsv,
              rgb_scratch,
              row.y(),
              w,
              plan,
              idx,
              use_simd,
              |scratch| {
                yuv_444_to_rgb_row(
                  row.y(),
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
            // Bottom lookback: advance only after the row-stage resample accepts
            // the row (the `?` above already returned any reject), so a rejected
            // row leaves the predecessor for a clean retry — the sited
            // reconstruction read it above but did not stage.
            stage_440_chroma_prev(chroma_prev, chroma_prev_row, row.u(), row.v(), idx, w);
          } else {
            // Co-sited RGB / RGBA / luma: the byte-identical row-stage tail.
            planar_dual_resample(
              luma_stream,
              rgb_stream,
              resample_outputs,
              rgb,
              rgba,
              luma,
              luma_u16,
              hsv,
              rgb_scratch,
              row.y(),
              w,
              plan,
              idx,
              use_simd,
              convert_rgb,
            )?;
          }
          if frozen_native_route.is_none() && need_output {
            *frozen_native_route = Some(false);
          }
          // Encoded domain committed alongside the route, on the same accepted
          // output-bearing row (the `?` above already returned any reject).
          if frozen_domain.is_none() && need_output {
            *frozen_domain = Some(AveragingDomain::Encoded);
          }
          // RFC #238 S8b: freeze the siting on the same accepted output row.
          if frozen_chroma_bottom_v.is_none() && need_output {
            *frozen_chroma_bottom_v = Some(bottom_v);
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

    // Strategy A output mode resolution — see Yuv420p impl above. Reuses the
    // Yuv444p RGBA / RGB / HSV dispatchers since 4:4:0's per-row math is
    // identical (full-width chroma). Computed BEFORE any per-row offset
    // arithmetic so the no-output guard below short-circuits a no-output call
    // before any row math.
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();
    // Whether this row produces any colour output (and so runs the bottom-sited
    // chroma reconstruction). A bottom-sited row maintains the vertical lookback
    // even when it produces only luma (see the staging below), so `want_color`
    // gates only the colour scratches / reconstruction.
    let want_color = want_rgb || want_rgba || want_hsv;

    // Repo-wide no-output invariant: a `process` call carrying NO output runs
    // NOTHING — no per-row offset arithmetic, no allocation, no state mutation
    // (the bottom-sited vertical lookback included). Returning HERE, before the
    // `idx * w` offsets below, keeps the invariant overflow-safe (a no-output
    // call never ran an attach-time `w x h x 1` validation, so `idx * w` could
    // overflow `usize` on a 32-bit target) and means such a row never reserves
    // `chroma_prev` nor primes the lookback (so it can never make a later colour
    // even row box-blend through an invisible, never-output row).
    let need_output = want_color || luma.is_some() || luma_u16.is_some();
    if !need_output {
      return Ok(());
    }

    // Single-plane row ranges cannot overflow: the no-output guard above ensures
    // >= 1 output is attached, so its attach-time `w x h x 1` validation ran and
    // `w x h` fits `usize`; with `idx < h` that makes `(idx + 1) * w <= h * w`
    // fit too.
    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // Chroma siting (RFC #238 S8b): `Bottom` (v = 1) box-blends the even output
    // row's full-width chroma with the previous chroma row; the default / Center
    // / Top sitings keep the byte-identical co-sited decode (their odd output row
    // needs the *next* chroma row, deferred). 4:4:0 has full-width chroma, so
    // there is no horizontal phase.
    let bottom_v = chroma_440_bottom_sited_v(chroma_location);

    // RFC #238 S8b: freeze the vertical chroma siting on the direct (identity)
    // path too. The resample branch above rejects a mid-frame vertical-phase flip
    // via `frozen_chroma_bottom_v`, and this path must honour the SAME per-frame
    // invariant: `set_chroma_location` is public, so without this a caller could
    // emit rows `Bottom`, switch to a co-sited location, then back, binning a
    // mixture of co-sited and bottom-folded chroma (and, after a co-sited gap,
    // box-blend a later even row against a stale lookback). CHECK here — before any
    // reservation or lookback mutation; the matching SET rides the row's accept
    // point below (after the fallible preflight). `need_output` is already
    // established (the no-output guard returned above).
    if let Some(frozen) = *frozen_chroma_bottom_v
      && frozen != bottom_v
    {
      return Err(MixedSinkerError::ChromaSitingChanged(
        ChromaSitingChanged::new(idx),
      ));
    }

    // Atomicity preflight (#308, cf. the crate's #180 resample fix and the
    // Yuv420p sibling): reserve EVERY fallible row scratch this row needs BEFORE
    // any output row (luma / luma_u16 included) is written, so an allocator
    // refusal returns a typed `AllocationFailed` leaving the output frame
    // untouched. Growable here:
    //  1. the bottom-sited full-width blend scratch (`chroma_full`, `2 * w`),
    //     only on a colour Bottom decode;
    //  2. the bottom-sited vertical-phase full-width lookback (`chroma_prev`), on
    //     every Bottom OUTPUT row — colour OR luma-only — because the lookback is
    //     maintained so a later colour row can box-blend it (the luma-only
    //     staging runs below, before any luma write); and
    //  3. the RGB row scratch, exactly `want_hsv && want_rgba && !want_rgb`.
    // A no-output row returned early above, so it never reaches here.
    if bottom_v && want_color {
      reserve_420_chroma_full(chroma_full, w, h)?;
    }
    if bottom_v {
      reserve_440_chroma_prev(chroma_prev, w, h)?;
    }
    if want_hsv && want_rgba && !want_rgb {
      // Grow the RGB scratch up front via the same helper the decode uses, so its
      // (test-injectable) allocation runs here — before luma — and the later call
      // is a no-op reserve. `rgb` is `None` in this arm, so this takes the scratch
      // branch and returns a slice we discard.
      rgb_row_buf_or_scratch(
        rgb.as_deref_mut(),
        rgb_scratch,
        one_plane_start,
        one_plane_end,
        w,
        h,
      )?;
    }

    // Bottom-sited LUMA-ONLY row: the colour upsample helper — which normally
    // refreshes the vertical lookback — won't run, so stage the current
    // full-width chroma row into the lookback HERE, after its preflight
    // reservation above and BEFORE any luma write, so a later colour row in the
    // same frame can box-blend it. A colour row instead stages inside
    // `upsample_440_chroma_sited` after reading the previous lookback, so this is
    // skipped for it; a no-output row returned early above.
    if bottom_v && !want_color {
      stage_440_chroma_prev(chroma_prev, chroma_prev_row, row.u(), row.v(), idx, w);
    }

    // RFC #238 S8b: the row is now past its fallible preflight — every scratch is
    // reserved and every remaining write targets an attached output buffer or the
    // pre-grown RGB scratch, so it cannot fail. Commit the vertical-siting freeze
    // here (mirroring the resample arms' accept-time SET), at the single point all
    // four output sub-paths pass through before diverging. Only the frame's first
    // output-bearing row sets it; the CHECK above rejects any later phase flip.
    if frozen_chroma_bottom_v.is_none() {
      *frozen_chroma_bottom_v = Some(bottom_v);
    }

    if let Some(luma) = luma.as_deref_mut() {
      luma[one_plane_start..one_plane_end].copy_from_slice(&row.y()[..w]);
    }

    // Luma u16 — zero-extend the 8-bit Y plane into u16.
    if let Some(buf) = luma_u16.as_deref_mut() {
      crate::row::y_plane_to_luma_u16_row(
        row.y(),
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    // HSV-only (no RGB / RGBA) goes direct through `yuv_444_to_hsv_row` (4:4:0
    // reuses the 4:4:4 kernel) — see the Yuv420p impl for the routing rationale.
    // `Bottom` first reconstructs the full-width vertically-blended chroma.
    let want_hsv_direct = want_hsv && !want_rgb && !want_rgba;
    let need_rgb_kernel = want_rgb || (want_hsv && want_rgba);

    if want_hsv_direct {
      let hsv = hsv.as_mut().expect("want_hsv_direct implies hsv attached");
      let (h, s, v) = hsv.hsv();
      if bottom_v {
        // Scratch already reserved by the atomicity preflight above; this
        // upsample is infallible and stages the lookback (`stage = true`).
        let (u_full, v_full) = upsample_440_chroma_sited(
          chroma_full,
          chroma_prev,
          chroma_prev_row,
          row.u(),
          row.v(),
          idx,
          bottom_v,
          true,
          w,
        );
        yuv_444_to_hsv_row(
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
        );
      } else {
        yuv_444_to_hsv_row(
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
        );
      }
      return Ok(());
    }

    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      if bottom_v {
        let (u_full, v_full) = upsample_440_chroma_sited(
          chroma_full,
          chroma_prev,
          chroma_prev_row,
          row.u(),
          row.v(),
          idx,
          bottom_v,
          true,
          w,
        );
        yuv_444_to_rgba_row(
          row.y(),
          u_full,
          v_full,
          rgba_row,
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
        );
      } else {
        yuv_444_to_rgba_row(
          row.y(),
          row.u(),
          row.v(),
          rgba_row,
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
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

    // Fused YUV→RGB (4:4:0 reuses the 4:4:4 kernel); `Bottom` first reconstructs
    // the full-width vertically-blended chroma and runs the 4:4:4 kernel on it.
    if bottom_v {
      let (u_full, v_full) = upsample_440_chroma_sited(
        chroma_full,
        chroma_prev,
        chroma_prev_row,
        row.u(),
        row.v(),
        idx,
        bottom_v,
        true,
        w,
      );
      yuv_444_to_rgb_row(
        row.y(),
        u_full,
        v_full,
        rgb_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
    } else {
      yuv_444_to_rgb_row(
        row.y(),
        row.u(),
        row.v(),
        rgb_row,
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

    if let Some(buf) = rgba.as_deref_mut() {
      let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
    }

    Ok(())
  }
}

// ---- Yuv411p impl -------------------------------------------------------
//
// 4:1:1 planar 8-bit — quarter-width chroma, full-height. DV-NTSC
// legacy. Per-row math reuses the dedicated `yuv_411_to_rgb_row` /
// `yuv_411_to_rgba_row` family (1→4 chroma upsample). Following
// FFmpeg's `AV_PIX_FMT_YUV411P` semantics, chroma row width is
// `width.div_ceil(4)`: non-4-aligned widths get a partial 1..3-pixel
// final chroma group, handled by the scalar tail.

impl<'a, R> MixedSinker<'a, Yuv411p, R> {
  /// Attaches a packed 32-bit RGBA output buffer.
  ///
  /// See [`MixedSinker::<Yuv420p>::with_rgba`] for the rationale and
  /// constraints. Yuv411p has no alpha plane, so every alpha byte is
  /// filled with `0xFF` (opaque).
  ///
  /// Returns `Err(InsufficientRgbaBuffer)` if
  /// `buf.len() < width x height x 4`, or `Err(GeometryOverflow)` on
  /// 32-bit targets when the product overflows.
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

  /// Attaches a `u16` luma output buffer. The 8-bit Y plane samples
  /// are zero-extended into `u16`. Length is measured in `u16`
  /// elements (`width x height`).
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

impl<R> Yuv411pSink for MixedSinker<'_, Yuv411p, R> {}

impl<R> PixelSink for MixedSinker<'_, Yuv411p, R> {
  type Input<'r> = Yuv411pRow<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    // FFmpeg-compatible: arbitrary widths accepted (chroma row is
    // `width.div_ceil(4)` samples; the scalar kernel handles a
    // partial 1..3-pixel final chroma group). No width-parity
    // restriction here.
    check_dimensions_match(self.width, self.height, width, height)?;
    // Resampling carries frame progress in the stream; reset it and
    // re-freeze the output set so each frame starts clean.
    if let Some(stream) = self.rgb_stream.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.luma_stream.as_mut() {
      stream.reset();
    }
    // New frame: restart the RGB-free HSV-only row-stage join (#263 follow-up).
    if let Some(hsv) = self.hsv_planar.as_mut() {
      hsv.reset();
    }
    // RFC #238 #302: clear the per-frame frozen chroma siting so the next frame
    // may pick either phase; a mid-frame flip stays rejected within a frame.
    self.frozen_chroma_centered = None;
    self.resample_outputs = None;
    Ok(())
  }

  fn process(&mut self, row: Yuv411pRow<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;
    // Chroma siting (#302): drives the horizontal chroma phase. `Copy`, so read
    // it out before the field split-borrow below.
    let chroma_location = self.chroma_location;

    // Chroma row shape: `width.div_ceil(4)` samples (FFmpeg
    // `AV_PIX_FMT_YUV411P`). For widths divisible by 4 this matches
    // `w / 4`; for non-aligned widths the trailing 1..3 Y pixels
    // share the last (partial) chroma sample.
    let chroma_w = w.div_ceil(4);
    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Y,
        idx,
        w,
        row.y().len(),
      )));
    }
    if row.u_quarter().len() != chroma_w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::UQuarter,
        idx,
        chroma_w,
        row.u_quarter().len(),
      )));
    }
    if row.v_quarter().len() != chroma_w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::VQuarter,
        idx,
        chroma_w,
        row.v_quarter().len(),
      )));
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange(
        RowIndexOutOfRange::new(idx, self.height),
      ));
    }

    let Self {
      rgb,
      rgba,
      luma,
      luma_u16,
      hsv,
      rgb_scratch,
      plan,
      rgb_stream,
      luma_stream,
      hsv_planar,
      resample_outputs,
      // Centered chroma-siting (#302) stages full-width U + V here.
      chroma_full,
      // RFC #238: the 4:1:1 chroma phase frozen on the first output row.
      frozen_chroma_centered,
      ..
    } = self;

    // Non-identity plan: freeze the output set, then check stream
    // sequencing — both before touching the scratch — so a no-output
    // sink stays a no-op and an out-of-sequence row is rejected
    // without the source-width allocation/conversion. Yuv411p is row-stage
    // only — every RGB output derives from the binned RGB rows, and the
    // HSV-only arm bins the SUBSAMPLED chroma through the folded
    // `area_chroma_411` plan. RFC #238 #302: the centered group
    // (`chroma_411_center_sited_h`) samples chroma at the CENTER of its four
    // luma columns — the RGB / HSV-with-RGBA converters reconstruct full-width
    // chroma here (`1→4` #302 phase), the HSV-only arm folds the phase into the
    // chroma area weights; the co-sited / unspecified group keeps the
    // byte-identical nearest decode.
    if let Some(plan) = plan.as_ref() {
      let matrix = row.matrix();
      let full_range = row.full_range();
      let center_sited = chroma_411_center_sited_h(chroma_location);
      let chroma_h_phase = if center_sited {
        YUV41X_CENTERED_H_PHASE
      } else {
        0.0
      };
      let want_color = rgb.is_some() || rgba.is_some() || hsv.is_some();
      // RFC #238 #302: freeze the effective 4:1:1 chroma siting on the first
      // output-bearing row. A later row observing a different phase — in sequence
      // or not — would bin a mixture of co-sited and centered chroma, so reject
      // it here before any reconstruction. The matching SET rides each path's
      // accept-time freeze below (never before dispatch, so a rejected row leaves
      // it unset for a corrected retry).
      let need_output =
        luma.is_some() || luma_u16.is_some() || rgb.is_some() || rgba.is_some() || hsv.is_some();
      if need_output
        && let Some(frozen) = *frozen_chroma_centered
        && frozen != center_sited
      {
        return Err(MixedSinkerError::ChromaSitingChanged(
          ChromaSitingChanged::new(idx),
        ));
      }
      // HSV-only (no RGB / RGBA): bin Y / U / V and convert per output row,
      // RGB-free. 4:1:1 chroma is quarter-width (`chroma_w = ceil(w/4)`),
      // full height (`chroma_vsub = 1`); the folded `area_chroma_411` plan
      // carries the centered phase. Bit-identical to a YUV-domain
      // bin-then-convert reference.
      if plan.kind().is_area() && hsv.is_some() && rgb.is_none() && rgba.is_none() {
        // RFC #238 point-of-use siting invalidation (see the Yuv422p HSV-only
        // arm): drop the cached HSV-only join if its chroma plan was built for a
        // different phase, so `hsv_direct_resample` rebuilds it with the current
        // siting. Retry-atomic — fires ONLY on the IN-SEQUENCE fresh-frame first
        // row (`idx == 0`, `next_y() == 0`); an out-of-sequence first row is left
        // for the delegate to reject against the INTACT join, and a mid-frame row
        // never drops (the phase is frame-constant). The taken join is restored on
        // a rejected rebuild so the row mutates nothing.
        let stale_hsv = idx == 0
          && hsv_planar
            .as_ref()
            .is_some_and(|join| join.chroma_centered() != center_sited && join.next_y() == 0);
        let prev_hsv = if stale_hsv { hsv_planar.take() } else { None };
        let hsv_result = super::planar_resample::hsv_direct_resample(
          hsv_planar,
          resample_outputs,
          luma,
          luma_u16,
          hsv,
          row.y(),
          row.u_quarter(),
          row.v_quarter(),
          matrix,
          full_range,
          1,
          || ResamplePlan::area_chroma_411(w, h, plan.out_w(), plan.out_h(), chroma_h_phase, 0.0),
          w,
          plan,
          idx,
          use_simd,
        );
        if stale_hsv && hsv_result.is_err() {
          *hsv_planar = prev_hsv;
        }
        hsv_result?;
        if frozen_chroma_centered.is_none() && need_output {
          *frozen_chroma_centered = Some(center_sited);
        }
        return Ok(());
      }
      // RGB / RGBA (row-stage): the co-sited path bins the nearest
      // `yuv_411_to_rgb_row` decode; the centered path reconstructs full-width
      // chroma (`1→4` #302 phase) AFTER the resample preflight and feeds the
      // 4:4:4 decode, so an out-of-sequence / rejected row is caught before the
      // chroma reservation (#180). `planar_dual_resample` re-runs the idempotent
      // preflight. `want_color` gates the reservation: a luma-only row bins only
      // luma and must not reserve / reconstruct chroma.
      let r = if center_sited && want_color {
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
        let (u_full, v_full) = reconstruct_chroma(
          Chroma411U8,
          chroma_full,
          row.u_quarter(),
          row.v_quarter(),
          w,
        );
        planar_dual_resample(
          luma_stream,
          rgb_stream,
          resample_outputs,
          rgb,
          rgba,
          luma,
          luma_u16,
          hsv,
          rgb_scratch,
          row.y(),
          w,
          plan,
          idx,
          use_simd,
          |scratch| {
            yuv_444_to_rgb_row(
              row.y(),
              u_full,
              v_full,
              scratch,
              w,
              matrix,
              full_range,
              use_simd,
            );
          },
        )
      } else {
        planar_dual_resample(
          luma_stream,
          rgb_stream,
          resample_outputs,
          rgb,
          rgba,
          luma,
          luma_u16,
          hsv,
          rgb_scratch,
          row.y(),
          w,
          plan,
          idx,
          use_simd,
          |scratch| {
            yuv_411_to_rgb_row(
              row.y(),
              row.u_quarter(),
              row.v_quarter(),
              scratch,
              w,
              matrix,
              full_range,
              use_simd,
            );
          },
        )
      };
      if r.is_ok() && need_output && frozen_chroma_centered.is_none() {
        *frozen_chroma_centered = Some(center_sited);
      }
      return r;
    }

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // Strategy A output mode resolution — see Yuv420p impl above.
    // 4:1:1 has its own dedicated `yuv_411_to_rgb_row` /
    // `yuv_411_to_rgba_row` kernels (1→4 chroma upsample) for the co-sited
    // decode; the centered horizontal sitings (#302) reconstruct chroma at the
    // `1→4` phase-0.375 position then reuse the 4:4:4 dispatchers.
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();
    let want_color = want_rgb || want_rgba || want_hsv;

    // Chroma siting (#302): the centered horizontal sitings reconstruct chroma at
    // the `1→4` phase-0.375 position then decode via the 4:4:4 kernels; the
    // default / co-sited path keeps the byte-identical nearest decode. 4:1:1 is
    // subsampled horizontally only — no vertical blend or chroma lookback.
    let center_sited = chroma_411_center_sited_h(chroma_location);

    // RFC #238 #302: mirror the resample branch's per-frame siting freeze on the
    // identity path. `chroma_location` is public and `Copy`, so a caller can flip
    // the phase mid-frame; without this a single frame could mix centered (`1→4`)
    // and co-sited nearest reconstruction. Freeze the effective phase on the first
    // output-bearing row and reject a later differing phase BEFORE any centered
    // reservation or output write, so a rejected row mutates nothing (#180). The
    // matching SET rides the accept-time freeze after the reservations below.
    let need_output =
      luma.is_some() || luma_u16.is_some() || rgb.is_some() || rgba.is_some() || hsv.is_some();
    if need_output
      && let Some(frozen) = *frozen_chroma_centered
      && frozen != center_sited
    {
      return Err(MixedSinkerError::ChromaSitingChanged(
        ChromaSitingChanged::new(idx),
      ));
    }

    // Atomicity preflight (#302 / #308, cf. the crate's #180 resample fix and the
    // Yuv420p sibling): reserve EVERY fallible row scratch this row needs BEFORE
    // any output row (luma / luma_u16 included) is written, so an allocator
    // refusal returns a typed `AllocationFailed` leaving the output frame
    // untouched rather than partially mutated. Two scratches can grow:
    //  1. the centered-siting full-width chroma (`chroma_full`), only on a colour
    //     decode; and
    //  2. the RGB row buffer, reserved exactly when a colour decode needs an RGB
    //     row but no caller RGB buffer is borrowable — `want_hsv && want_rgba &&
    //     !want_rgb` (`rgb_row_buf_or_scratch`'s own scratch arm; an attached RGB
    //     buffer is borrowed instead and never allocates).
    // The later `reconstruct_chroma` / `rgb_row_buf_or_scratch` calls reuse the
    // already-sized buffers, so the default path is byte-identical; only the
    // failure-path ordering changes.
    if center_sited && want_color {
      reserve_420_chroma_full(chroma_full, w, h)?;
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

    // RFC #238 #302: every fallible per-row reservation above has succeeded, so
    // freeze the accepted phase for the frame (mirrors the resample branch's
    // accept-time SET). Never before the reservations, so a rejected or failed row
    // leaves the freeze clear for a corrected retry.
    if need_output && frozen_chroma_centered.is_none() {
      *frozen_chroma_centered = Some(center_sited);
    }

    // Luma — Yuv411p luma *is* the Y plane. Just copy.
    if let Some(luma) = luma.as_deref_mut() {
      luma[one_plane_start..one_plane_end].copy_from_slice(&row.y()[..w]);
    }

    // Luma u16 — zero-extend the 8-bit Y plane into u16.
    if let Some(buf) = luma_u16.as_deref_mut() {
      crate::row::y_plane_to_luma_u16_row(
        row.y(),
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    // HSV-only (no RGB / RGBA) goes direct through `yuv_411_to_hsv_row`
    // — see the Yuv420p impl for the routing rationale.
    let want_hsv_direct = want_hsv && !want_rgb && !want_rgba;
    let need_rgb_kernel = want_rgb || (want_hsv && want_rgba);

    if want_hsv_direct {
      let hsv = hsv.as_mut().expect("want_hsv_direct implies hsv attached");
      let (h, s, v) = hsv.hsv();
      if center_sited {
        let (u_full, v_full) = reconstruct_chroma(
          Chroma411U8,
          chroma_full,
          row.u_quarter(),
          row.v_quarter(),
          w,
        );
        yuv_444_to_hsv_row(
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
        );
      } else {
        yuv_411_to_hsv_row(
          row.y(),
          row.u_quarter(),
          row.v_quarter(),
          &mut h[one_plane_start..one_plane_end],
          &mut s[one_plane_start..one_plane_end],
          &mut v[one_plane_start..one_plane_end],
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
        );
      }
      return Ok(());
    }

    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      if center_sited {
        let (u_full, v_full) = reconstruct_chroma(
          Chroma411U8,
          chroma_full,
          row.u_quarter(),
          row.v_quarter(),
          w,
        );
        yuv_444_to_rgba_row(
          row.y(),
          u_full,
          v_full,
          rgba_row,
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
        );
      } else {
        yuv_411_to_rgba_row(
          row.y(),
          row.u_quarter(),
          row.v_quarter(),
          rgba_row,
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
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

    if center_sited {
      let (u_full, v_full) = reconstruct_chroma(
        Chroma411U8,
        chroma_full,
        row.u_quarter(),
        row.v_quarter(),
        w,
      );
      yuv_444_to_rgb_row(
        row.y(),
        u_full,
        v_full,
        rgb_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
    } else {
      yuv_411_to_rgb_row(
        row.y(),
        row.u_quarter(),
        row.v_quarter(),
        rgb_row,
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

    if let Some(buf) = rgba.as_deref_mut() {
      let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
    }

    Ok(())
  }
}
