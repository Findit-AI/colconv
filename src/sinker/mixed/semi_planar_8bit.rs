//! 8-bit semi-planar YUV `MixedSinker` impls: Nv12 / Nv16 / Nv21 / Nv24 / Nv42.
//!
//! On a non-identity **area** plan every member routes through the shared
//! row-stage planar resample ([`super::planar_resample::planar_dual_resample`]):
//! the Y plane area-resamples directly for luma (the YUV luma contract),
//! while RGB / RGBA / HSV bin a source-width RGB row converted with the
//! format's own fused `nv*_to_rgb_row` kernel (chroma de-interleave +
//! upsample happen in registers inside that kernel, exactly as on the
//! identity path). RGB therefore equals an `Rgb24` area-resample of the
//! identity-converted frame — byte-identical to the matching
//! [`Yuv420p`] (row-stage) / [`Yuv422p`] / [`Yuv444p`] resample of the
//! de-interleaved planes. The 4:2:0 native decimation tier is a
//! planar-only optimization and does not apply here.
//!
//! A non-identity **filter** plan routes through the filter twin
//! ([`super::planar_resample::planar_dual_filter_resample`]) with the SAME
//! `nv*_to_rgb_row` convert closure: the Y plane is filter-resampled as a
//! 1-channel `u8` stream (luma stays native Y, never colour-derived) and the
//! converted source-width RGB is filter-resampled by the signed-coefficient
//! filter stream — so a filter colour output equals the equivalent `Rgb24`
//! filter resample of those exact converted pixels (byte-identical to the
//! planar twins). The 4:2:0 members (Nv12 / Nv21) branch the filter plan
//! BEFORE their native/row-stage route machinery, which is area-only.

use super::{
  GeometryOverflow, InsufficientBuffer, MixedSinker, MixedSinkerError, RowIndexOutOfRange,
  RowShapeMismatch, RowSlice, WidthAlignment, check_dimensions_match,
  planar_resample::{planar_dual_filter_resample, planar_dual_resample},
  rgb_row_buf_or_scratch, rgba_plane_row_slice,
};
use crate::{PixelSink, row::*, source::*};

#[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
use super::{
  ChromaSitingChanged, HsvFrameMut, NativeRouteChanged, chroma_420_bottom_sited_v,
  chroma_420_center_sited_h, chroma_422_center_sited_h,
  planar_8bit::{
    NativePlanarYuv, YUV422P_CENTERED_H_PHASE, native_planar_preflight_check_only,
    reserve_420_chroma_full, reserve_420_chroma_prev, stage_420_chroma_prev,
    upsample_420_chroma_sited, yuv_planar_process_native, yuv420p_process_native,
  },
};
#[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
use crate::{
  ColorMatrix,
  resample::{
    AveragingDomain, InsertionContext, InsertionPoint, PlanGeometry, ResampleError, ResamplePlan,
    select_insertion_point,
  },
};

// Test-only allocation failpoint for the U / V de-interleave scratch grow
// in `semi_planar_process_native`. When armed, the next chroma-bearing
// reserve returns the crate's recoverable `AllocationFailed` WITHOUT
// growing — letting the atomicity regression test prove the first-row
// out-of-sequence preflight runs BEFORE this fallible grow (so a rejected
// even-colour first row returns OutOfSequenceRow, never AllocationFailed).
// `Cell<bool>` is plenty (single-threaded, take-on-read). Strictly
// test-only — the non-test build is byte-identical (this hook compiles
// away entirely).
#[cfg(all(
  test,
  feature = "std",
  feature = "yuv-semi-planar",
  feature = "yuv-planar"
))]
std::thread_local! {
  static FORCE_DEINTERLEAVE_ALLOC_FAILURE: core::cell::Cell<bool> =
    const { core::cell::Cell::new(false) };
}

/// Arms the de-interleave scratch allocation failpoint for the **next**
/// chroma-bearing native semi-planar row on the current thread. The flag
/// is consumed (take-on-read) by that reserve, so it fires exactly once
/// and cannot leak into a later test. Test-only. The sole caller is the
/// native-tier atomicity coverage in `tests/resample_semi_planar.rs`, which
/// is `rgb`-gated (it drives colour rows); the thread-local and its
/// production take-on-read stay broad.
#[cfg(all(
  test,
  feature = "std",
  feature = "yuv-semi-planar",
  feature = "yuv-planar",
  feature = "rgb"
))]
pub(super) fn arm_deinterleave_alloc_failure() {
  FORCE_DEINTERLEAVE_ALLOC_FAILURE.with(|f| f.set(true));
}

/// Native fast-tier 4:2:0 decimator for the semi-planar family
/// ([`Nv12`](crate::source::Nv12) / [`Nv21`](crate::source::Nv21)): bins
/// the native Y / U / V planes straight to the output grid and converts
/// once per output row at output resolution. Reuses the planar twin's
/// join verbatim ([`yuv420p_process_native`]) after de-interleaving the
/// interleaved chroma row into the sink's U / V scratch — so every output
/// is byte-identical to a [`Yuv420p`](crate::source::Yuv420p) native
/// conversion of the de-interleaved planes, and within ±1 LSB of the
/// semi-planar row-stage tier (the conversion-order rounding caveat the
/// planar tiers already carry).
///
/// `chroma_uv` is the interleaved chroma half-row; `swap_uv = false`
/// reads `U0 V0 U1 V1 …` (NV12), `swap_uv = true` reads `V0 U0 …`
/// (NV21). The chroma row is consumed only on even source rows; the
/// caller passes the full interleaved row regardless and this splits it.
///
/// `chroma_h_phase` / `chroma_v_phase` are the RFC #238 chroma sampling phases
/// folded into the 4:2:0 chroma area weights ([`ResamplePlan::area_chroma_420`]):
/// `chroma_h_phase` is `0.25` for the centered group (`Nv12` / `Nv21` `Center` /
/// `Top` / `Bottom`) and `0.0` for co-sited / unspecified; `chroma_v_phase` is
/// `1.0` for the S4-C `Bottom` vertical siting and `0.0` for every
/// vertical-co-sited layout. At both phases `0.0` the folded plan is
/// byte-identical to the plain co-sited plan, so co-sited / horizontal-only
/// output is untouched.
///
/// The U / V scratch is reserved (fallibly) before the call into the
/// planar join, and the de-interleave writes only into that private
/// scratch — so the recoverable-allocation / atomicity contract the join
/// enforces (no caller-output write before the preflight completes) holds
/// across the de-interleave too.
#[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
#[allow(clippy::too_many_arguments)]
fn semi_planar_process_native(
  plan: &ResamplePlan,
  native_420: &mut Option<std::boxed::Box<super::planar_8bit::NativeYuv420>>,
  u_scratch: &mut std::vec::Vec<u8>,
  v_scratch: &mut std::vec::Vec<u8>,
  resample_outputs: &mut Option<super::FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  rgb_scratch: &mut std::vec::Vec<u8>,
  y_row: &[u8],
  chroma_uv: &[u8],
  swap_uv: bool,
  chroma_h_phase: f64,
  chroma_v_phase: f64,
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

  // Run the join's COMPLETE pre-feed rejection preflight FIRST — no-output
  // short-circuit, first-row out-of-sequence check, AND the frozen-output
  // (mid-frame output-set change) check — before touching the U / V
  // de-interleave scratch. Compare-only (no output-set freeze), so the
  // de-interleave reserve below stays a genuine pre-commit step ahead of the
  // delegate's own commit. The reserve is fallible and grows sink state;
  // deferring it until the full compare clears keeps EVERY rejection case
  // (out-of-sequence first colour row OR mid-frame output change) returning its
  // deterministic typed error (OutOfSequenceRow / ResampleOutputsChanged),
  // never AllocationFailed under allocation pressure, and leaves the scratch
  // untouched — the crate's preflight-atomicity contract. `Ok(false)` is the
  // no-output no-op: return without reserving. `yuv420p_process_native` re-runs
  // this identical compare and owns the commit, keeping a single source of
  // truth.
  if !super::planar_8bit::yuv420p_native_preflight_check_only(
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

  // De-interleave the chroma half-row into the U / V scratch — only on
  // chroma-bearing rows (even source index) when colour is wanted, which
  // is exactly where the planar join reads chroma and nowhere else. The
  // split writes only this private scratch, so no caller output is touched
  // until the join's own preflight (re-run inside the call below) clears.
  // On odd / luma-only / no-colour rows the join never reads chroma, so the
  // scratch is left as-is and the join gets empty U / V slices — which also
  // keeps a direct caller's out-of-sequence odd first row (no
  // `begin_frame`, empty scratch) from indexing past the scratch before the
  // join rejects it.
  let chroma_row = need_color && idx.is_multiple_of(2);
  if chroma_row {
    for scratch in [&mut *u_scratch, &mut *v_scratch] {
      if scratch.len() < cw {
        // Test-only failpoint: simulate a recoverable allocator refusal of
        // the de-interleave scratch grow WITHOUT exhausting memory, so the
        // regression test can prove the first-row preflight already
        // rejected an out-of-sequence colour row (returning
        // OutOfSequenceRow) before this fallible grow is ever reached. With
        // the preflight ordered AFTER this grow (the bug) an armed failure
        // would surface as AllocationFailed instead.
        #[cfg(all(
          test,
          feature = "std",
          feature = "yuv-semi-planar",
          feature = "yuv-planar"
        ))]
        if FORCE_DEINTERLEAVE_ALLOC_FAILURE.with(|f| f.take()) {
          return Err(MixedSinkerError::Resample(ResampleError::AllocationFailed(
            PlanGeometry::new(w, h, plan.out_w(), plan.out_h()),
          )));
        }
        scratch.try_reserve_exact(cw - scratch.len()).map_err(|_| {
          MixedSinkerError::Resample(ResampleError::AllocationFailed(PlanGeometry::new(
            w,
            h,
            plan.out_w(),
            plan.out_h(),
          )))
        })?;
        scratch.resize(cw, 0);
      }
    }
    // NV12 chroma is `U V U V …` (U at even byte), NV21 is `V U V U …`.
    let (u_off, v_off) = if swap_uv { (1, 0) } else { (0, 1) };
    for (i, pair) in chroma_uv.chunks_exact(2).enumerate() {
      u_scratch[i] = pair[u_off];
      v_scratch[i] = pair[v_off];
    }
  }

  let (u_half, v_half): (&[u8], &[u8]) = if chroma_row {
    (&u_scratch[..cw], &v_scratch[..cw])
  } else {
    (&[], &[])
  };
  yuv420p_process_native(
    plan,
    native_420,
    resample_outputs,
    rgb,
    rgba,
    luma,
    luma_u16,
    hsv,
    rgb_scratch,
    y_row,
    u_half,
    v_half,
    matrix,
    full_range,
    idx,
    w,
    h,
    // 4:2:0 semi-planar chroma folds the RFC #238 horizontal chroma phase
    // (`chroma_h_phase`: `0.25` centered, `0.0` co-sited) AND the S4-C vertical
    // phase (`chroma_v_phase`: `1.0` for `Bottom`, `0.0` co-sited) into the
    // chroma area weights, exactly as the planar Yuv420p native arm does. At both
    // phases `0` the folded plan is byte-identical to the pre-closure
    // `NativeYuv420::new` internal co-sited plan, so co-sited / unspecified /
    // horizontal-only output is untouched.
    || {
      ResamplePlan::area_chroma_420(
        w / 2,
        h,
        plan.out_w(),
        plan.out_h(),
        chroma_h_phase,
        chroma_v_phase,
      )
    },
    use_simd,
  )
}

/// Native fast-tier decimator for the NON-4:2:0 semi-planar 8-bit family
/// ([`Nv16`](crate::source::Nv16) 4:2:2 / [`Nv24`](crate::source::Nv24)
/// 4:4:4 UV / [`Nv42`](crate::source::Nv42) 4:4:4 VU): the non-4:2:0
/// sibling of [`semi_planar_process_native`]. De-interleaves the packed
/// chroma row into the sink's U / V scratch, then reuses the planar twin's
/// non-4:2:0 join verbatim ([`yuv_planar_process_native`]) — so every output
/// is byte-identical to a [`Yuv422p`](crate::source::Yuv422p) /
/// [`Yuv444p`](crate::source::Yuv444p) native conversion of those planes,
/// and within ±1 LSB of the semi-planar row-stage tier (the conversion-order
/// rounding caveat the planar tiers already carry).
///
/// `chroma_w` is the chroma plane width per row (`w / 2` for 4:2:2, `w` for
/// 4:4:4); the packed chroma row is `2 * chroma_w` bytes. Unlike the 4:2:0
/// wrapper the chroma cadence is one row per Y row (`chroma_vsub = 1`), so
/// the de-interleave runs on EVERY colour row. `swap_uv = false` reads
/// `U0 V0 U1 V1 …` (NV16 / NV24); `swap_uv = true` reads `V0 U0 …` (NV42) —
/// the SAME swapped-order split the NV21 4:2:0 path uses.
///
/// `chroma_h_phase` is the RFC #238 horizontal chroma sampling phase folded
/// into the chroma area weights ([`ResamplePlan::area_chroma_422`]): `0.25`
/// for the centered 4:2:2 group (`Nv16`), `0.0` for co-sited 4:2:2 and every
/// 4:4:4 (`Nv24` / `Nv42`) caller. At phase `0.0` the folded plan is
/// byte-identical to the plain `area` plan, so the co-sited / 4:4:4 output is
/// untouched.
///
/// Atomicity mirrors [`semi_planar_process_native`]: the join's COMPLETE
/// pre-feed rejection preflight runs FIRST (via
/// [`native_planar_preflight_check_only`]), before the fallible U / V scratch
/// grow, so a rejected row returns its deterministic typed error
/// (`OutOfSequenceRow` / `ResampleOutputsChanged`), never `AllocationFailed`,
/// and grows no sink state. It is compare-only (no output-set freeze), so the
/// scratch grow stays a pre-feed step: the delegate commits the freeze only
/// after its own build + scratch succeed. The de-interleave writes only the
/// private scratch, so no caller output is touched until the delegate clears.
#[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
#[allow(clippy::too_many_arguments)]
fn semi_planar_process_native_non420(
  plan: &ResamplePlan,
  native_planar: &mut Option<std::boxed::Box<NativePlanarYuv>>,
  u_scratch: &mut std::vec::Vec<u8>,
  v_scratch: &mut std::vec::Vec<u8>,
  resample_outputs: &mut Option<super::FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  rgb_scratch: &mut std::vec::Vec<u8>,
  y_row: &[u8],
  chroma_uv: &[u8],
  chroma_w: usize,
  swap_uv: bool,
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

  // Run the join's COMPLETE pre-feed rejection preflight FIRST — before the
  // U / V de-interleave scratch grow — so EVERY rejection case (out-of-
  // sequence first colour row OR mid-frame output change) returns its
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

  // De-interleave the packed chroma row into the U / V scratch on every
  // colour row (chroma_vsub == 1: a chroma row per Y row, vs the 4:2:0
  // even-only cadence). The split writes only this private scratch, so no
  // caller output is touched until the join's own preflight (re-run inside
  // the delegate below) clears. On luma-only / no-colour rows the join never
  // reads chroma, so the scratch is left as-is and the join gets empty
  // U / V slices.
  if need_color {
    for scratch in [&mut *u_scratch, &mut *v_scratch] {
      if scratch.len() < chroma_w {
        scratch
          .try_reserve_exact(chroma_w - scratch.len())
          .map_err(|_| {
            MixedSinkerError::Resample(ResampleError::AllocationFailed(PlanGeometry::new(
              w,
              h,
              plan.out_w(),
              plan.out_h(),
            )))
          })?;
        scratch.resize(chroma_w, 0);
      }
    }
    // NV16 / NV24 chroma is `U V U V …` (U at even byte), NV42 is
    // `V U V U …` — mirror the NV21 swapped-order split.
    let (u_off, v_off) = if swap_uv { (1, 0) } else { (0, 1) };
    for (i, pair) in chroma_uv.chunks_exact(2).enumerate() {
      u_scratch[i] = pair[u_off];
      v_scratch[i] = pair[v_off];
    }
  }

  let (u_plane, v_plane): (&[u8], &[u8]) = if need_color {
    (&u_scratch[..chroma_w], &v_scratch[..chroma_w])
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
    y_row,
    u_plane,
    v_plane,
    matrix,
    full_range,
    idx,
    w,
    h,
    1,
    || ResamplePlan::area_chroma_422(chroma_w, h, plan.out_w(), plan.out_h(), chroma_h_phase, 0.0),
    use_simd,
  )
}

// ---- Chroma-siting-aware 4:2:0 upsample (#302) --------------------------
//
// The semi-planar siblings of the planar [`reserve_420_chroma_full`] /
// [`upsample_420_chroma_center_h`] staging. NV chroma is INTERLEAVED in one
// plane (Nv12 `U V U V …`, Nv21 `V U V U …`), so the centered horizontal
// upsample first de-interleaves the half-row into half-width U / V scratch,
// then reuses the planar twin's exact phase-0.5 kernel + the plain (non-
// primaries) 4:4:4 kernels — making a centered NV decode bit-identical to a
// [`Yuv420p`](crate::source::Yuv420p) decode of the de-interleaved planes on
// the shared matrix-tag path. (`ChromaDerivedNcl` is the lone exception: NV —
// like every format except `Yuv420p` — resolves it via the BT.709 matrix-tag
// fallback `Coefficients::for_matrix`, NOT `Yuv420p`'s #316 primaries-derived
// path; the default and centered NV paths agree on that fallback, so they stay
// internally consistent. Primaries-derived NV is a #302/#303 follow-up.) Only
// the centered sitings (`Center` / `Top` / `Bottom`) reach here; every co-sited
// / unspecified siting keeps the default fused `nv*_to_*_row` decode,
// byte-identical to the pre-#302 output.

/// **Fallible preflight** for the centered-siting de-interleave scratch: grows
/// the half-width U and V buffers to `width / 2` so the later infallible
/// [`nv_center_upsample_chroma`] de-interleave reuses already-sized buffers.
/// Split from the de-interleave so it runs **before any output row is written**
/// — the crate's preflight-ordering atomicity contract (cf. #180 / #308). Sized
/// like the native tier's de-interleave scratch (`width / 2` each). A grow
/// refusal is the typed, recoverable [`ResampleError::AllocationFailed`], never
/// an abort; `height` feeds the payload geometry.
#[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
fn reserve_nv_chroma_half(
  u_half: &mut std::vec::Vec<u8>,
  v_half: &mut std::vec::Vec<u8>,
  width: usize,
  height: usize,
) -> Result<(), MixedSinkerError> {
  let cw = width / 2;
  for scratch in [u_half, v_half] {
    if scratch.len() < cw {
      scratch.try_reserve_exact(cw - scratch.len()).map_err(|_| {
        MixedSinkerError::Resample(ResampleError::AllocationFailed(PlanGeometry::new(
          width, height, width, height,
        )))
      })?;
      scratch.resize(cw, 0);
    }
  }
  Ok(())
}

/// De-interleaves the NV packed chroma half-row into the already-reserved
/// half-width U / V scratch. `swap_uv = false` reads Nv12 (`U` at the even
/// byte), `true` reads Nv21 (`V` at the even byte) — the SAME split the native
/// tier uses. Split out so the `Bottom`-sited luma-only path can stage the
/// de-interleaved chroma into the vertical-phase lookback without a colour
/// upsample.
///
/// The caller must have run [`reserve_nv_chroma_half`] up front, so both
/// scratches are guaranteed `>= width / 2` here.
#[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
fn nv_deinterleave_chroma_half(
  u_half: &mut [u8],
  v_half: &mut [u8],
  uv: &[u8],
  width: usize,
  swap_uv: bool,
) {
  let cw = width / 2;
  debug_assert!(
    u_half.len() >= cw && v_half.len() >= cw,
    "half-width chroma scratch must be reserved via reserve_nv_chroma_half first"
  );
  let (u_off, v_off) = if swap_uv { (1, 0) } else { (0, 1) };
  for (i, pair) in uv.chunks_exact(2).take(cw).enumerate() {
    u_half[i] = pair[u_off];
    v_half[i] = pair[v_off];
  }
}

/// De-interleaves the NV packed chroma half-row into the already-reserved
/// half-width U / V scratch ([`nv_deinterleave_chroma_half`]), then reconstructs
/// full-width chroma into `chroma_full` via the siting-aware
/// [`upsample_420_chroma_sited`] — the `center_h` horizontal upsample (centered
/// `1/4`–`3/4` when true, co-sited nearest-neighbor when false), plus (for
/// `bottom_v`, the RFC #238 `Bottom` / `BottomLeft` vertical phase) the `v = 1`
/// box blend of the current chroma row with the previous one through the
/// `chroma_prev` lookback. Returns the full-width `(u_full, v_full)` the 4:4:4
/// decode kernels consume. The lookback stores the DE-INTERLEAVED chroma, so a
/// semi-planar `Bottom` / `BottomLeft` even row box-blends the de-interleaved
/// prev + cur rows — identical math to the planar `Yuv420p` twin, just after the
/// de-interleave. When `bottom_v == false` (`Center` / `Top`, and every 4:2:2
/// caller) it is byte-identical to the pre-vertical horizontal-only upsample: the
/// sited helper delegates to the plain horizontal upsample and stages nothing.
///
/// `stage` follows the [`upsample_420_chroma_sited`] contract — the direct
/// decode passes `true` (its post-reconstruction work is infallible), the
/// resample reconstruction arms pass `false` and defer [`stage_420_chroma_prev`]
/// until AFTER their fallible commit accepts the row (#180).
///
/// **Infallible**: the caller must have run [`reserve_420_chroma_full`],
/// [`reserve_nv_chroma_half`], and — when `bottom_v` — [`reserve_420_chroma_prev`]
/// up front, so all buffers are guaranteed long enough here.
#[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
#[allow(clippy::too_many_arguments)]
fn nv_center_upsample_chroma<'s>(
  chroma_full: &'s mut [u8],
  chroma_prev: &mut [u8],
  chroma_prev_row: &mut Option<usize>,
  u_half: &mut [u8],
  v_half: &mut [u8],
  uv: &[u8],
  idx: usize,
  bottom_v: bool,
  center_h: bool,
  stage: bool,
  width: usize,
  swap_uv: bool,
) -> (&'s [u8], &'s [u8]) {
  let cw = width / 2;
  nv_deinterleave_chroma_half(u_half, v_half, uv, width, swap_uv);
  upsample_420_chroma_sited(
    chroma_full,
    chroma_prev,
    chroma_prev_row,
    &u_half[..cw],
    &v_half[..cw],
    idx,
    bottom_v,
    center_h,
    stage,
    width,
  )
}

// ---- Nv12 impl ----------------------------------------------------------

impl<'a, R> MixedSinker<'a, Nv12, R> {
  /// Attaches a packed 32‑bit RGBA output buffer.
  ///
  /// Only available on sinker types whose `PixelSink` impl writes
  /// RGBA — calling `with_rgba` on a sink that doesn't (e.g. a
  /// not‑yet‑wired `MixedSinker<Nv16>` today) is a compile error
  /// rather than a silent no‑op. Each format that adds RGBA support
  /// adds its own impl block here.
  ///
  /// The fourth byte per pixel is alpha. NV12 has no alpha plane,
  /// so every alpha byte is filled with `0xFF` (opaque). Future
  /// YUVA source impls will copy alpha through from the source
  /// plane.
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

impl<R> Nv12Sink for MixedSinker<'_, Nv12, R> {}

impl<R> PixelSink for MixedSinker<'_, Nv12, R> {
  type Input<'r> = Nv12Row<'r>;
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
    // New frame: restart the row-stage resample streams (area + filter) so a
    // reused sink starts each frame clean.
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
    #[cfg(feature = "yuv-planar")]
    if let Some(native) = self.native_420.as_mut() {
      native.reset();
    }
    // New frame: clear the per-frame frozen native/row-stage route so the
    // next frame may pick either tier; a mid-frame flip stays rejected.
    // Gated to the native tier's feature (the route guard only exists when
    // the planar join the native tier reuses is compiled in). The RFC #238 S3b
    // frozen chroma-siting phase is cleared alongside it, so the next frame may
    // pick either siting while a mid-frame phase flip stays rejected.
    #[cfg(feature = "yuv-planar")]
    {
      self.frozen_native_route = None;
      self.frozen_chroma_centered = None;
      // RFC #238 S4-C companion: clear the frozen vertical phase too, so the
      // next frame may pick Center or Bottom and a mid-frame Center ⇆ Bottom
      // flip stays rejected.
      self.frozen_chroma_bottom_v = None;
      // New frame: invalidate the bottom-sited vertical-phase chroma lookback
      // (#302) so frame N+1's first even row can never box-blend frame N's last
      // chroma row. The buffer bytes are left as-is (re-overwritten before any
      // trusted read); only the validity tag is cleared.
      self.chroma_prev_row = None;
    }
    self.resample_outputs = None;
    Ok(())
  }

  fn process(&mut self, row: Nv12Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;
    // Chroma siting (#302): drives the identity-plan horizontal chroma phase.
    // `Copy`, so read it before the field split-borrow below. Gated like its
    // only consumer (`chroma_420_center_sited_h` + the 4:4:4 kernels need
    // `yuv-planar`); a semi-planar-only build keeps the default decode.
    #[cfg(feature = "yuv-planar")]
    let chroma_location = self.chroma_location;

    // Defense-in-depth shape check (see Yuv420p impl above). An NV12
    // UV row is `width` bytes of interleaved U / V payload — same
    // length as Y — so both slices must equal `self.width`. Odd-width
    // check comes first since the row primitive would panic on it.
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
    if row.uv_half().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::UvHalf,
        idx,
        w,
        row.uv_half().len(),
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
      #[cfg(feature = "yuv-planar")]
      native,
      #[cfg(feature = "yuv-planar")]
      native_420,
      #[cfg(feature = "yuv-planar")]
      semi_planar_u_half,
      #[cfg(feature = "yuv-planar")]
      semi_planar_v_half,
      #[cfg(feature = "yuv-planar")]
      frozen_native_route,
      // RFC #238 S3b: the 4:2:0 chroma phase frozen on the first output row.
      #[cfg(feature = "yuv-planar")]
      frozen_chroma_centered,
      // RFC #238 S4-C: the vertical (`Bottom`) chroma phase frozen alongside it.
      #[cfg(feature = "yuv-planar")]
      frozen_chroma_bottom_v,
      // Full-width chroma staging for the centered-siting (#302) identity
      // decode; reuses the half-width de-interleave scratch above.
      #[cfg(feature = "yuv-planar")]
      chroma_full,
      // RFC #238 S4-C: the one-row bottom-sited vertical-phase chroma lookback
      // (de-interleaved U then V) + its validity tag, so a `Bottom` even output
      // row box-blends the previous chroma row.
      #[cfg(feature = "yuv-planar")]
      chroma_prev,
      #[cfg(feature = "yuv-planar")]
      chroma_prev_row,
      ..
    } = self;

    // Non-identity plan. A `Filter` plan routes to the filter resampler
    // (branched first, below). Otherwise, when the native tier is enabled
    // (and the planar join it reuses is compiled in), bin the native
    // Y / U / V planes at output resolution and convert once per output
    // row, de-interleaving the NV12 chroma row into U / V scratch first.
    // Otherwise (or under `with_native(false)`) take the row-stage tier:
    // bin the Y plane for luma directly (the YUV luma contract); for
    // colour, convert the interleaved source row to RGB with the same
    // fused `nv12_to_rgb_row`
    // kernel the identity path uses, then bin the RGB row. RGB therefore
    // equals an `Rgb24` area-resample of the identity-converted frame —
    // byte-identical to the `Yuv420p` row-stage twin.
    if let Some(plan) = plan.as_ref() {
      let matrix = row.matrix();
      let full_range = row.full_range();
      let convert_rgb = |scratch: &mut [u8]| {
        nv12_to_rgb_row(
          row.y(),
          row.uv_half(),
          scratch,
          w,
          matrix,
          full_range,
          use_simd,
        );
      };
      // RFC #238 S3b/S4-C — 4:2:0 chroma siting for Nv12, mirroring the planar
      // Yuv420p twin (S3a/S4-B) and the 4:2:2 Nv16 sibling (S2a). The centered
      // group (`Center` / `Top` / `Bottom`, [`chroma_420_center_sited_h`]) samples
      // chroma at `+0.25` chroma-sample horizontally; S4-C additionally folds the
      // VERTICAL `v = 1` triangle for `Bottom` ([`chroma_420_bottom_sited_v`], a
      // strict sub-case of `center_sited`), while `Center` / `Top` keep
      // `v_phase = 0` (co-sited vertical, byte-identical to S3b). The co-sited /
      // unspecified group is phase 0 (byte-identical to the pre-siting resample).
      // The native fast tier folds both phases into the chroma area weights
      // (`area_chroma_420`); the filter and row-stage tiers reconstruct full-width
      // chroma (de-interleave + phase-0.5 upsample, plus the `Bottom` even-row
      // vertical box blend) and decode 4:4:4. `convert_rgb` above is the co-sited
      // fused decode.
      #[cfg(feature = "yuv-planar")]
      let center_sited = chroma_420_center_sited_h(chroma_location);
      #[cfg(feature = "yuv-planar")]
      let bottom_v = chroma_420_bottom_sited_v(chroma_location);
      #[cfg(feature = "yuv-planar")]
      let chroma_h_phase = if center_sited {
        YUV422P_CENTERED_H_PHASE
      } else {
        0.0
      };
      #[cfg(feature = "yuv-planar")]
      let chroma_v_phase = if bottom_v { 1.0 } else { 0.0 };
      #[cfg(feature = "yuv-planar")]
      let want_color = rgb.is_some() || rgba.is_some() || hsv.is_some();
      // Whether this call carries any output — the EXACT set both tiers' preflight
      // tests. The route (and the siting phase) freezes only on an output-bearing
      // row a tier ACCEPTS; a no-output call must not freeze.
      #[cfg(feature = "yuv-planar")]
      let need_output =
        luma.is_some() || luma_u16.is_some() || rgb.is_some() || rgba.is_some() || hsv.is_some();
      // RFC #238 S3b/S4-C: freeze the effective 4:2:0 chroma siting on the first
      // output-bearing row (mirroring the Yuv420p twin's always-compiled choke
      // point). A later row observing a different phase — in sequence or not —
      // would bin a mixture of co-sited and centered chroma, so it is rejected
      // HERE before any reconstruction or dispatch; the matching SET rides each
      // tier's accept-time freeze below (never on a reject, so a corrected retry
      // is not falsely rejected). `Center` and `Bottom` share the horizontal
      // center, so the vertical phase is frozen too — a mid-frame Center ⇆ Bottom
      // flip is caught only by comparing `frozen_chroma_bottom_v`.
      #[cfg(feature = "yuv-planar")]
      if need_output
        && let Some(frozen) = *frozen_chroma_centered
        && (frozen != center_sited || *frozen_chroma_bottom_v != Some(bottom_v))
      {
        return Err(MixedSinkerError::ChromaSitingChanged(
          ChromaSitingChanged::new(idx),
        ));
      }
      // A `Filter` plan routes to the filter resampler, which converts the
      // de-interleaved chroma to a source-width RGB row (the same
      // `nv12_to_rgb_row` kernel the row-stage tier uses) and filter-resamples it
      // plus the native Y. The native fast tier is area-only and never sees a
      // filter plan; the per-sink plan kind is fixed at construction, so a filter
      // sink bypasses the native/row-stage route machinery entirely. Branched
      // FIRST, before the native-route guard below.
      if plan.kind().is_filter() {
        // Centered filter reconstructs full-width chroma and decodes 4:4:4, but
        // ONLY after the resample preflight (frozen-output + sequence), so an
        // out-of-sequence / rejected row is caught before the chroma reservation
        // (#180). `planar_dual_filter_resample` re-runs the idempotent preflight.
        // Co-sited keeps the fused `convert_rgb` decode.
        #[cfg(feature = "yuv-planar")]
        {
          // Reject a multi-kernel (BICUBLIN) filter plan BEFORE the centered
          // reserve below, mirroring the delegate's own first act (idempotent).
          plan.ensure_single_kernel_filter()?;
          if (center_sited || bottom_v) && want_color {
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
            reserve_nv_chroma_half(semi_planar_u_half, semi_planar_v_half, w, h)?;
            // RFC #238 S4-C: reserve the bottom-sited vertical-phase
            // lookback AFTER the preflight, BEFORE the commit (#180); only
            // for v=1. The sited call reads it but defers staging.
            if bottom_v {
              reserve_420_chroma_prev(chroma_prev, w, h)?;
            }
            let (u_full, v_full) = nv_center_upsample_chroma(
              chroma_full,
              chroma_prev,
              chroma_prev_row,
              semi_planar_u_half,
              semi_planar_v_half,
              row.uv_half(),
              idx,
              bottom_v,
              center_sited,
              false,
              w,
              false,
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
            // RFC #238 S4-C: advance the bottom lookback only AFTER the
            // filter resample accepts the row (#180 ordering) — the sited
            // reconstruction read it above but did not stage, so a rejected
            // row keeps the predecessor for a clean retry. The de-interleaved
            // chroma is in the half scratch from the upsample.
            if r.is_ok() && bottom_v {
              stage_420_chroma_prev(
                chroma_prev,
                chroma_prev_row,
                semi_planar_u_half,
                semi_planar_v_half,
                idx,
                w,
              );
            }
            if r.is_ok() && need_output && frozen_chroma_centered.is_none() {
              *frozen_chroma_centered = Some(center_sited);
              *frozen_chroma_bottom_v = Some(bottom_v);
            }
            return r;
          }
        }
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
          convert_rgb,
        )?;
        #[cfg(feature = "yuv-planar")]
        if need_output && frozen_chroma_centered.is_none() {
          *frozen_chroma_centered = Some(center_sited);
          *frozen_chroma_bottom_v = Some(bottom_v);
        }
        return Ok(());
      }
      // Reject a mid-frame native/row-stage route flip BEFORE either tier's
      // dispatch. The two tiers carry independent, in-order, once-only stream
      // state, so splitting a frame across them yields a mixed/partial frame
      // rather than a deterministic rejection. The route is both CHECKED here and
      // frozen below (the SET) ONLY on an output-bearing row a tier ACCEPTS — both
      // gate on `need_output`. A no-output call therefore neither checks nor
      // freezes the route: it is a true no-op, route-invisible regardless of row
      // index. A preflight-rejected (out-of-sequence / frozen) output-bearing call
      // returns Err before the SET, so it leaves `frozen_native_route` untouched
      // and a later same-or-other-route retry is not falsely rejected. (The native
      // tier — hence the route guard — only exists under `yuv-planar`.)
      // The RFC #238 splice stage. A filter plan already returned above, so
      // `area_plan` is true and the selector reproduces the former `*native`
      // boolean bit-for-bit (`cfg!` is true wherever this block compiles).
      #[cfg(feature = "yuv-planar")]
      let take_native = matches!(
        select_insertion_point(
          AveragingDomain::Encoded,
          InsertionContext {
            native_eligible: cfg!(feature = "yuv-planar"),
            with_native: *native,
            area_plan: true,
          },
        ),
        InsertionPoint::NativeCodes
      );
      #[cfg(feature = "yuv-planar")]
      if need_output
        && let Some(frozen) = *frozen_native_route
        && frozen != take_native
      {
        return Err(MixedSinkerError::NativeRouteChanged(
          NativeRouteChanged::new(idx),
        ));
      }
      #[cfg(feature = "yuv-planar")]
      if take_native {
        // RFC #238 S3b point-of-use siting invalidation, mirroring the Yuv420p
        // native arm: `chroma_location` can change at ANY point before this row
        // (including AFTER `begin_frame`, before row 0), so re-check the cached
        // join HERE and drop it when its folded chroma plan was built for a
        // different phase; `semi_planar_process_native` (via
        // `yuv420p_process_native`) then rebuilds it with the current siting.
        // Retry-atomic: drop ONLY on the IN-SEQUENCE fresh-frame first row
        // (`idx == 0`, `next_y() == 0`); an out-of-sequence first row is left for
        // the delegate to reject against the INTACT join. A luma-only join carries
        // no chroma plan (siting-independent), a no-output sink built no join.
        // Transactional: move the stale join OUT, let the delegate build the
        // replacement into `native_420` (its build runs BEFORE it inserts), and
        // restore the intact prior-phase join on a rejected rebuild so the
        // REJECTED row mutates nothing.
        let stale_native = idx == 0
          && native_420.as_ref().is_some_and(|join| {
            join.has_chroma()
              && (join.chroma_centered() != center_sited || join.chroma_bottom() != bottom_v)
              && join.next_y() == 0
          });
        let prev_native = if stale_native {
          native_420.take()
        } else {
          None
        };
        // Dispatch first; freeze the route + siting ONLY after the call returns Ok
        // on an output-bearing row. A no-output call returns Ok(()) with
        // `need_output` false (no freeze); an out-of-sequence / frozen row returns
        // Err via `?` (no freeze) — so only an accepted output-bearing row commits.
        let native_result = semi_planar_process_native(
          plan,
          native_420,
          semi_planar_u_half,
          semi_planar_v_half,
          resample_outputs,
          rgb,
          rgba,
          luma,
          luma_u16,
          hsv,
          rgb_scratch,
          row.y(),
          row.uv_half(),
          false,
          chroma_h_phase,
          chroma_v_phase,
          matrix,
          full_range,
          idx,
          w,
          h,
          use_simd,
        );
        // Restore the taken stale-phase join if the delegate's rebuild was rejected
        // at any pre-feed step: it leaves the field `None` and `resample_outputs`
        // uncommitted on such a failure, so restoring the intact prior-phase join
        // leaves the rejected row mutating nothing. A non-stale row (first-ever /
        // colour-capability rebuild) took nothing.
        if stale_native && native_result.is_err() {
          *native_420 = prev_native;
        }
        native_result?;
        if frozen_native_route.is_none() && need_output {
          *frozen_native_route = Some(true);
        }
        // RFC #238 S3b: freeze the siting on the same accepted output row.
        if frozen_chroma_centered.is_none() && need_output {
          *frozen_chroma_centered = Some(center_sited);
          *frozen_chroma_bottom_v = Some(bottom_v);
        }
        return Ok(());
      }
      // Row-stage tail. Same CHECK-before / SET-after split: dispatch, then freeze
      // the route + siting only when the call accepts an output-bearing row.
      // Centered colour reconstructs full-width chroma (de-interleave + phase-0.5
      // upsample) and decodes 4:4:4 — but ONLY after the resample preflight
      // (frozen-output + sequence), so an out-of-sequence / rejected row is caught
      // before the chroma reservation (#180). A luma-only centered row never calls
      // the RGB converter, so it stays on the co-sited arm (which only bins luma).
      // `planar_dual_resample` re-runs the idempotent preflight.
      #[cfg(feature = "yuv-planar")]
      if (center_sited || bottom_v) && want_color {
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
        reserve_nv_chroma_half(semi_planar_u_half, semi_planar_v_half, w, h)?;
        // RFC #238 S4-C: reserve the bottom-sited vertical-phase lookback
        // AFTER the preflight, BEFORE the commit (#180); only for v=1. The
        // sited call reads it but defers staging until the commit accepts.
        if bottom_v {
          reserve_420_chroma_prev(chroma_prev, w, h)?;
        }
        let (u_full, v_full) = nv_center_upsample_chroma(
          chroma_full,
          chroma_prev,
          chroma_prev_row,
          semi_planar_u_half,
          semi_planar_v_half,
          row.uv_half(),
          idx,
          bottom_v,
          center_sited,
          false,
          w,
          false,
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
        // RFC #238 S4-C: advance the bottom lookback only AFTER the
        // row-stage resample accepts the row (the `?` above already returned
        // any reject); the sited reconstruction read it but did not stage,
        // so a rejected row keeps the predecessor for a clean retry.
        if bottom_v {
          stage_420_chroma_prev(
            chroma_prev,
            chroma_prev_row,
            semi_planar_u_half,
            semi_planar_v_half,
            idx,
            w,
          );
        }
        if frozen_native_route.is_none() && need_output {
          *frozen_native_route = Some(false);
        }
        if frozen_chroma_centered.is_none() && need_output {
          *frozen_chroma_centered = Some(center_sited);
          *frozen_chroma_bottom_v = Some(bottom_v);
        }
        return Ok(());
      }
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
      #[cfg(feature = "yuv-planar")]
      if frozen_native_route.is_none() && need_output {
        *frozen_native_route = Some(false);
      }
      #[cfg(feature = "yuv-planar")]
      if frozen_chroma_centered.is_none() && need_output {
        *frozen_chroma_centered = Some(center_sited);
        *frozen_chroma_bottom_v = Some(bottom_v);
      }
      return Ok(());
    }

    // Strategy A output mode resolution — see Yuv420p impl above. Computed
    // BEFORE any per-row offset arithmetic so the no-output guard below
    // short-circuits before it runs ANY row math (`need_output` depends only on
    // the attached buffers, never on `idx` / `w`).
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();
    let want_color = want_rgb || want_rgba || want_hsv;

    // Repo-wide no-output invariant (mirrors the Yuv420p S4-B direct twin): a
    // `process` call carrying NO output — no colour, no luma, no luma_u16 — must
    // run NOTHING: no per-row offset arithmetic, no allocation, no state mutation
    // (the bottom-sited vertical lookback included). Returning HERE, before the
    // `idx * w` offsets below, makes those single-plane ranges overflow-safe (a
    // no-output call never ran an attach-time `w x h x 1` validation, so `idx * w`
    // could overflow `usize` on a 32-bit target) and guarantees such an invisible
    // row never reserves `chroma_prev` nor primes the lookback.
    let need_output = want_color || luma.is_some() || luma_u16.is_some();
    if !need_output {
      return Ok(());
    }

    // Single-plane row ranges are guaranteed to fit: the no-output guard above
    // ensures ≥ 1 output is attached, so its attach-time `w x h x 1` validation
    // ran and `w x h` fits `usize`; with `idx < h` that makes `(idx + 1) * w` fit
    // too. RGB / RGBA (x3 / x4) ranges use checked arithmetic below.
    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // Chroma siting (#302): the centered horizontal sitings reconstruct chroma
    // at the phase-0.5 position; the default / co-sited path keeps the
    // byte-identical fused nearest-neighbor decode.
    #[cfg(feature = "yuv-planar")]
    let center_sited = chroma_420_center_sited_h(chroma_location);
    #[cfg(feature = "yuv-planar")]
    let bottom_v = chroma_420_bottom_sited_v(chroma_location);

    // Atomicity preflight (#302 / #308, cf. the crate's #180 resample fix and
    // the planar_8bit / Yuv420p siblings): reserve EVERY fallible row scratch
    // this row needs BEFORE any output row (luma / luma_u16 included) is
    // written, so an allocator refusal returns a typed `AllocationFailed`
    // leaving the output frame untouched rather than partially mutated. Two
    // scratches can grow:
    //  1. the centered-siting full-width chroma (`chroma_full`) plus the
    //     half-width de-interleave scratch (`semi_planar_u_half/v_half`); and
    //  2. the RGB row buffer, reserved exactly when a colour decode needs an
    //     RGB row but no caller RGB buffer is borrowable — `want_hsv &&
    //     want_rgba && !want_rgb` (`rgb_row_buf_or_scratch`'s own scratch arm;
    //     an attached RGB buffer is borrowed instead and never allocates).
    // The later `nv_center_upsample_chroma` / `rgb_row_buf_or_scratch` calls
    // then reuse the already-sized buffers.
    #[cfg(feature = "yuv-planar")]
    if (center_sited || bottom_v) && (want_rgb || want_rgba || want_hsv) {
      reserve_420_chroma_full(chroma_full, w, h)?;
      reserve_nv_chroma_half(semi_planar_u_half, semi_planar_v_half, w, h)?;
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
    // RFC #238 S4-C: on every bottom output row reserve the vertical-phase
    // lookback + de-interleave half scratch BEFORE any luma write (#180/#302),
    // so a later colour row can box-blend the previous chroma row. A colour row
    // stages inside `nv_center_upsample_chroma` after reading the lookback; a
    // LUMA-ONLY row has no colour upsample, so it de-interleaves + stages here.
    // (A no-output row already returned at the guard above, so it never reaches
    // this reserve / stage.)
    #[cfg(feature = "yuv-planar")]
    if bottom_v {
      reserve_420_chroma_prev(chroma_prev, w, h)?;
      reserve_nv_chroma_half(semi_planar_u_half, semi_planar_v_half, w, h)?;
      if !want_color {
        nv_deinterleave_chroma_half(
          semi_planar_u_half,
          semi_planar_v_half,
          row.uv_half(),
          w,
          false,
        );
        stage_420_chroma_prev(
          chroma_prev,
          chroma_prev_row,
          semi_planar_u_half,
          semi_planar_v_half,
          idx,
          w,
        );
      }
    }

    // Luma — NV12 luma is the Y plane, copied verbatim by the native-Y
    // kernel (bit-identical to the former inline `copy_from_slice`).
    if let Some(luma) = luma.as_deref_mut() {
      nv_to_luma_row(row.y(), &mut luma[one_plane_start..one_plane_end], w);
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

    // HSV-without-RGB-or-RGBA goes through the direct `nv12_to_hsv_row`
    // kernel (no source-width RGB scratch). When RGB or RGBA is *also*
    // attached the RGB kernel runs anyway, so HSV derives off that buffer
    // for free (the cheap path) and `need_rgb_kernel` keeps it alive.
    let want_hsv_direct = want_hsv && !want_rgb && !want_rgba;
    let need_rgb_kernel = want_rgb || (want_hsv && want_rgba);

    if want_hsv_direct {
      let hsv = hsv.as_mut().expect("want_hsv_direct implies hsv attached");
      let (h, s, v) = hsv.hsv();
      // Sited siting (#302): de-interleave + full-width upsample chroma
      // (centered phase-0.5 or co-sited nearest-neighbor + `BottomLeft`'s `v = 1`
      // fold), then run the 4:4:4 HSV kernel (scratch reserved above).
      // `center_sited` / `bottom_v` are only ever true under `yuv-planar`.
      #[cfg(feature = "yuv-planar")]
      if center_sited || bottom_v {
        let (u_full, v_full) = nv_center_upsample_chroma(
          chroma_full,
          chroma_prev,
          chroma_prev_row,
          semi_planar_u_half,
          semi_planar_v_half,
          row.uv_half(),
          idx,
          bottom_v,
          center_sited,
          true,
          w,
          false,
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
        return Ok(());
      }
      nv12_to_hsv_row(
        row.y(),
        row.uv_half(),
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
      // Sited siting (#302): full-width chroma (centered phase-0.5 or co-sited
      // nearest-neighbor + `BottomLeft`'s `v = 1` fold) + the 4:4:4 RGBA kernel;
      // the default co-sited path keeps the fused `nv12_to_rgba_row`.
      #[cfg(feature = "yuv-planar")]
      if center_sited || bottom_v {
        let (u_full, v_full) = nv_center_upsample_chroma(
          chroma_full,
          chroma_prev,
          chroma_prev_row,
          semi_planar_u_half,
          semi_planar_v_half,
          row.uv_half(),
          idx,
          bottom_v,
          center_sited,
          true,
          w,
          false,
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
        return Ok(());
      }
      nv12_to_rgba_row(
        row.y(),
        row.uv_half(),
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

    // Fused NV12 → RGB: UV deinterleave + chroma upsample both happen
    // in registers inside the row primitive, no intermediate memory.
    // Sited siting (#302) instead de-interleaves + upsamples chroma to full
    // width (centered phase-0.5 or co-sited nearest-neighbor + `BottomLeft`'s
    // `v = 1` fold) and runs the 4:4:4 kernel; HSV / RGBA follow-ups below derive
    // off the produced RGB row either way.
    #[cfg(feature = "yuv-planar")]
    let centered = if center_sited || bottom_v {
      let (u_full, v_full) = nv_center_upsample_chroma(
        chroma_full,
        chroma_prev,
        chroma_prev_row,
        semi_planar_u_half,
        semi_planar_v_half,
        row.uv_half(),
        idx,
        bottom_v,
        center_sited,
        true,
        w,
        false,
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
      true
    } else {
      false
    };
    #[cfg(not(feature = "yuv-planar"))]
    let centered = false;
    if !centered {
      nv12_to_rgb_row(
        row.y(),
        row.uv_half(),
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

// ---- Nv16 impl ----------------------------------------------------------
//
// 4:2:2 is 4:2:0's vertical‑axis twin: one UV row per Y row instead of
// one per two. Per‑row math is identical, so this impl calls the same
// `nv12_to_rgb_row` / `nv12_to_rgba_row` dispatchers — no new kernels
// needed.

impl<'a, R> MixedSinker<'a, Nv16, R> {
  /// Attaches a packed 32‑bit RGBA output buffer.
  ///
  /// Only available on sinker types whose `PixelSink` impl writes
  /// RGBA — see [`MixedSinker::<Yuv420p>::with_rgba`] for the same
  /// rationale and constraints. NV16 has no alpha plane, so every
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

impl<R> Nv16Sink for MixedSinker<'_, Nv16, R> {}

impl<R> PixelSink for MixedSinker<'_, Nv16, R> {
  type Input<'r> = Nv16Row<'r>;
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
    #[cfg(feature = "yuv-planar")]
    if let Some(native) = self.native_planar.as_mut() {
      native.reset();
    }
    // New frame: clear the per-frame frozen native/row-stage route so the
    // next frame may pick either tier; a mid-frame flip stays rejected.
    // Gated to the native tier's feature (the planar join the native tier
    // reuses is compiled only under `yuv-planar`). The RFC #238 S2a frozen
    // chroma-siting phase is cleared alongside it, so the next frame may pick
    // either siting while a mid-frame phase flip stays rejected.
    #[cfg(feature = "yuv-planar")]
    {
      self.frozen_native_route = None;
      self.frozen_chroma_centered = None;
    }
    self.resample_outputs = None;
    Ok(())
  }

  fn process(&mut self, row: Nv16Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

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
    // NV16 UV row is `width` bytes of interleaved U/V — identical shape
    // to NV12's `uv_half`.
    if row.uv().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::UvHalf,
        idx,
        w,
        row.uv().len(),
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
    // `yuv-planar`); a semi-planar-only build keeps the default decode.
    #[cfg(feature = "yuv-planar")]
    let chroma_location = self.chroma_location;

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
      #[cfg(feature = "yuv-planar")]
      native,
      #[cfg(feature = "yuv-planar")]
      native_planar,
      #[cfg(feature = "yuv-planar")]
      semi_planar_u_half,
      #[cfg(feature = "yuv-planar")]
      semi_planar_v_half,
      #[cfg(feature = "yuv-planar")]
      frozen_native_route,
      // RFC #238 S2a: the 4:2:2 chroma phase frozen on the first output row.
      #[cfg(feature = "yuv-planar")]
      frozen_chroma_centered,
      // Full-width chroma staging for the centered-siting (#302) identity
      // decode; reuses the half-width de-interleave scratch above.
      #[cfg(feature = "yuv-planar")]
      chroma_full,
      // RFC #238 S4-C: the shared bottom-sited vertical-phase lookback,
      // threaded (inert) so the 4:2:2 sited call matches the shared signature.
      // 4:2:2 never folds a vertical phase, so the sited call always passes a
      // literal `false` and this lookback is never read or written.
      #[cfg(feature = "yuv-planar")]
      chroma_prev,
      #[cfg(feature = "yuv-planar")]
      chroma_prev_row,
      ..
    } = self;

    // Non-identity plan. A `Filter` plan routes to the filter resampler
    // (branched first, below). Otherwise, when the native tier is enabled
    // (and the planar join it reuses is compiled in), bin the native
    // Y / U / V planes at output resolution and convert once per output row,
    // de-interleaving the NV16 chroma row into U / V scratch first (4:2:2 ->
    // chroma `w/2 x h`). Otherwise (or under `with_native(false)`) take the
    // row-stage tier (matches the Yuv422p twin): bin Y for luma; for colour,
    // convert the interleaved source row to RGB with the fused
    // `nv12_to_rgb_row` kernel the identity path reuses for 4:2:2, then bin
    // the RGB row.
    if let Some(plan) = plan.as_ref() {
      let matrix = row.matrix();
      let full_range = row.full_range();
      let convert_rgb = |scratch: &mut [u8]| {
        nv12_to_rgb_row(row.y(), row.uv(), scratch, w, matrix, full_range, use_simd);
      };
      // RFC #238 S2a — 4:2:2 horizontal chroma siting for Nv16, mirroring the
      // planar Yuv422p twin. The centered group (`Center` / `Top` / `Bottom`,
      // [`chroma_422_center_sited_h`]) samples chroma at `+0.25` chroma-sample;
      // the co-sited / unspecified group is phase 0 (byte-identical to the
      // pre-siting resample). The native fast tier folds the phase into the
      // chroma area weights (`area_chroma_422`); the filter and row-stage tiers
      // reconstruct full-width chroma (de-interleave + phase-0.5 upsample) and
      // decode 4:4:4. `convert_rgb` above is the co-sited fused decode.
      #[cfg(feature = "yuv-planar")]
      let center_sited = chroma_422_center_sited_h(chroma_location);
      #[cfg(feature = "yuv-planar")]
      let chroma_h_phase = if center_sited {
        YUV422P_CENTERED_H_PHASE
      } else {
        0.0
      };
      #[cfg(feature = "yuv-planar")]
      let want_color = rgb.is_some() || rgba.is_some() || hsv.is_some();
      // Whether this call carries any output — the EXACT set both tiers'
      // preflight tests. The route (and the siting phase) freezes only on an
      // output-bearing row a tier ACCEPTS; a no-output call must not freeze.
      #[cfg(feature = "yuv-planar")]
      let need_output =
        luma.is_some() || luma_u16.is_some() || rgb.is_some() || rgba.is_some() || hsv.is_some();
      // RFC #238 S2a: freeze the effective 4:2:2 chroma siting on the first
      // output-bearing row (mirroring the Yuv422p twin's always-compiled choke
      // point). A later row observing a different phase — in sequence or not —
      // would bin a mixture of co-sited and centered chroma, so it is rejected
      // HERE before any reconstruction or dispatch; the matching SET rides each
      // tier's accept-time freeze below (never on a reject, so a corrected
      // retry is not falsely rejected).
      #[cfg(feature = "yuv-planar")]
      if need_output
        && let Some(frozen) = *frozen_chroma_centered
        && frozen != center_sited
      {
        return Err(MixedSinkerError::ChromaSitingChanged(
          ChromaSitingChanged::new(idx),
        ));
      }
      // A `Filter` plan routes to the filter resampler. The native fast tier
      // is area-only and never sees a filter plan; the per-sink plan kind is
      // fixed at construction, so a filter sink bypasses the native/row-stage
      // route machinery entirely. Branched FIRST, before the native-route
      // guard below.
      if plan.kind().is_filter() {
        // Centered filter reconstructs full-width chroma and decodes 4:4:4, but
        // ONLY after the resample preflight (frozen-output + sequence), so an
        // out-of-sequence / rejected row is caught before the chroma
        // reservation (#180). `planar_dual_filter_resample` re-runs the
        // idempotent preflight. Co-sited keeps the fused `convert_rgb` decode.
        #[cfg(feature = "yuv-planar")]
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
            reserve_nv_chroma_half(semi_planar_u_half, semi_planar_v_half, w, h)?;
            let (u_full, v_full) = nv_center_upsample_chroma(
              chroma_full,
              chroma_prev,
              chroma_prev_row,
              semi_planar_u_half,
              semi_planar_v_half,
              row.uv(),
              idx,
              false,
              center_sited,
              false,
              w,
              false,
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
            if r.is_ok() && need_output && frozen_chroma_centered.is_none() {
              *frozen_chroma_centered = Some(center_sited);
            }
            return r;
          }
        }
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
          convert_rgb,
        )?;
        #[cfg(feature = "yuv-planar")]
        if need_output && frozen_chroma_centered.is_none() {
          *frozen_chroma_centered = Some(center_sited);
        }
        return Ok(());
      }
      // Reject a mid-frame native/row-stage route flip BEFORE either tier's
      // dispatch (see the Nv12 impl above for the full CHECK-before /
      // SET-after rationale; both gate on `need_output`).
      // The RFC #238 splice stage. A filter plan already returned above, so
      // `area_plan` is true and the selector reproduces the former `*native`
      // boolean bit-for-bit (`cfg!` is true wherever this block compiles).
      #[cfg(feature = "yuv-planar")]
      let take_native = matches!(
        select_insertion_point(
          AveragingDomain::Encoded,
          InsertionContext {
            native_eligible: cfg!(feature = "yuv-planar"),
            with_native: *native,
            area_plan: true,
          },
        ),
        InsertionPoint::NativeCodes
      );
      #[cfg(feature = "yuv-planar")]
      if need_output
        && let Some(frozen) = *frozen_native_route
        && frozen != take_native
      {
        return Err(MixedSinkerError::NativeRouteChanged(
          NativeRouteChanged::new(idx),
        ));
      }
      #[cfg(feature = "yuv-planar")]
      if take_native {
        // RFC #238 S2a point-of-use siting invalidation, mirroring the Yuv422p
        // native arm: `chroma_location` can change at ANY point before this row
        // (including AFTER `begin_frame`, before row 0), so re-check the cached
        // join HERE and drop it when its folded chroma plan was built for a
        // different phase; `semi_planar_process_native_non420` (via
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
        let native_result = semi_planar_process_native_non420(
          plan,
          native_planar,
          semi_planar_u_half,
          semi_planar_v_half,
          resample_outputs,
          rgb,
          rgba,
          luma,
          luma_u16,
          hsv,
          rgb_scratch,
          row.y(),
          row.uv(),
          w / 2,
          false,
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
        // RFC #238 S2a: freeze the siting on the same accepted output row.
        if frozen_chroma_centered.is_none() && need_output {
          *frozen_chroma_centered = Some(center_sited);
        }
        return Ok(());
      }
      // Row-stage tail. Same CHECK-before / SET-after split: dispatch, then
      // freeze the route + siting only when the call accepts an output-bearing
      // row. Centered colour reconstructs full-width chroma (de-interleave +
      // phase-0.5 upsample) and decodes 4:4:4 — but ONLY after the resample
      // preflight (frozen-output + sequence), so an out-of-sequence / rejected
      // row is caught before the chroma reservation (#180). A luma-only centered
      // row never calls the RGB converter, so it stays on the co-sited arm
      // (which only bins luma). `planar_dual_resample` re-runs the idempotent
      // preflight.
      #[cfg(feature = "yuv-planar")]
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
        reserve_nv_chroma_half(semi_planar_u_half, semi_planar_v_half, w, h)?;
        let (u_full, v_full) = nv_center_upsample_chroma(
          chroma_full,
          chroma_prev,
          chroma_prev_row,
          semi_planar_u_half,
          semi_planar_v_half,
          row.uv(),
          idx,
          false,
          center_sited,
          false,
          w,
          false,
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
        if frozen_native_route.is_none() && need_output {
          *frozen_native_route = Some(false);
        }
        if frozen_chroma_centered.is_none() && need_output {
          *frozen_chroma_centered = Some(center_sited);
        }
        return Ok(());
      }
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
      #[cfg(feature = "yuv-planar")]
      if frozen_native_route.is_none() && need_output {
        *frozen_native_route = Some(false);
      }
      #[cfg(feature = "yuv-planar")]
      if frozen_chroma_centered.is_none() && need_output {
        *frozen_chroma_centered = Some(center_sited);
      }
      return Ok(());
    }

    // Strategy A output mode resolution — see Yuv420p impl above.
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();

    // No-output guard (#302): a `process` call with NO output attached never ran
    // an attach-time `w x h` validation, so on a 32-bit target an absurd geometry
    // could overflow the `idx * w` offset below. Returning HERE — before that
    // arithmetic AND before the centered chroma preflight — keeps a no-output row
    // panic-free and allocation-free.
    let need_output = want_rgb || want_rgba || want_hsv || luma.is_some() || luma_u16.is_some();
    if !need_output {
      return Ok(());
    }

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // Chroma siting (#302): the centered horizontal sitings reconstruct chroma at
    // the phase-0.5 position; the default / co-sited path keeps the byte-identical
    // fused nearest-neighbor decode.
    #[cfg(feature = "yuv-planar")]
    let center_sited = chroma_422_center_sited_h(chroma_location);

    // Atomicity preflight (#302 / #308, cf. the crate's #180 resample fix and the
    // planar_8bit / Yuv420p / Nv12 siblings): reserve EVERY fallible row scratch
    // this row needs BEFORE any output row (luma / luma_u16 included) is written,
    // so an allocator refusal returns a typed `AllocationFailed` leaving the
    // output frame untouched rather than partially mutated. Two scratches can
    // grow:
    //  1. the centered-siting full-width chroma (`chroma_full`) plus the
    //     half-width de-interleave scratch (`semi_planar_u_half/v_half`); and
    //  2. the RGB row buffer, reserved exactly when a colour decode needs an RGB
    //     row but no caller RGB buffer is borrowable — `want_hsv && want_rgba &&
    //     !want_rgb` (`rgb_row_buf_or_scratch`'s own scratch arm; an attached RGB
    //     buffer is borrowed instead and never allocates).
    // The later `nv_center_upsample_chroma` / `rgb_row_buf_or_scratch` calls then
    // reuse the already-sized buffers.
    #[cfg(feature = "yuv-planar")]
    if center_sited && (want_rgb || want_rgba || want_hsv) {
      reserve_420_chroma_full(chroma_full, w, h)?;
      reserve_nv_chroma_half(semi_planar_u_half, semi_planar_v_half, w, h)?;
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

    if let Some(luma) = luma.as_deref_mut() {
      nv_to_luma_row(row.y(), &mut luma[one_plane_start..one_plane_end], w);
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

    // Reuses NV12 dispatchers (RGB, RGBA, and the direct HSV kernel)
    // since 4:2:2's row contract is identical to 4:2:0's. HSV-only (no
    // RGB / RGBA) goes direct through `nv12_to_hsv_row` (no source-width
    // RGB scratch); see the Nv12 impl above for the routing rationale.
    let want_hsv_direct = want_hsv && !want_rgb && !want_rgba;
    let need_rgb_kernel = want_rgb || (want_hsv && want_rgba);

    if want_hsv_direct {
      let hsv = hsv.as_mut().expect("want_hsv_direct implies hsv attached");
      let (h, s, v) = hsv.hsv();
      // Centered siting (#302): de-interleave + phase-0.5 upsample chroma to full
      // width, then run the 4:4:4 HSV kernel (scratch reserved above).
      // `center_sited` is only ever true under `yuv-planar`.
      #[cfg(feature = "yuv-planar")]
      if center_sited {
        let (u_full, v_full) = nv_center_upsample_chroma(
          chroma_full,
          chroma_prev,
          chroma_prev_row,
          semi_planar_u_half,
          semi_planar_v_half,
          row.uv(),
          idx,
          false,
          center_sited,
          false,
          w,
          false,
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
        return Ok(());
      }
      nv12_to_hsv_row(
        row.y(),
        row.uv(),
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
      // Centered siting (#302): full-width phase-0.5 chroma + the 4:4:4 RGBA
      // kernel; the default co-sited path keeps the fused `nv12_to_rgba_row`.
      #[cfg(feature = "yuv-planar")]
      if center_sited {
        let (u_full, v_full) = nv_center_upsample_chroma(
          chroma_full,
          chroma_prev,
          chroma_prev_row,
          semi_planar_u_half,
          semi_planar_v_half,
          row.uv(),
          idx,
          false,
          center_sited,
          false,
          w,
          false,
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
        return Ok(());
      }
      nv12_to_rgba_row(
        row.y(),
        row.uv(),
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

    // Fused NV16 → RGB reuses the NV12 dispatcher — 4:2:2's row contract is
    // identical. Centered siting (#302) instead de-interleaves + upsamples chroma
    // to full width (phase-0.5) and runs the 4:4:4 kernel; HSV / RGBA follow-ups
    // below derive off the produced RGB row either way.
    #[cfg(feature = "yuv-planar")]
    let centered = if center_sited {
      let (u_full, v_full) = nv_center_upsample_chroma(
        chroma_full,
        chroma_prev,
        chroma_prev_row,
        semi_planar_u_half,
        semi_planar_v_half,
        row.uv(),
        idx,
        false,
        center_sited,
        false,
        w,
        false,
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
      true
    } else {
      false
    };
    #[cfg(not(feature = "yuv-planar"))]
    let centered = false;
    if !centered {
      nv12_to_rgb_row(
        row.y(),
        row.uv(),
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

// ---- Nv21 impl ----------------------------------------------------------
//
// Structurally identical to the Nv12 impl — the row primitives hide
// the U/V byte-order difference. Only the trait `Input<'r>` and the
// primitive name change.

impl<'a, R> MixedSinker<'a, Nv21, R> {
  /// Attaches a packed 32‑bit RGBA output buffer.
  ///
  /// Only available on sinker types whose `PixelSink` impl writes
  /// RGBA — see [`MixedSinker::<Nv12>::with_rgba`] for the same
  /// rationale and constraints. NV21 has no alpha plane, so every
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

impl<R> Nv21Sink for MixedSinker<'_, Nv21, R> {}

impl<R> PixelSink for MixedSinker<'_, Nv21, R> {
  type Input<'r> = Nv21Row<'r>;
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
    #[cfg(feature = "yuv-planar")]
    if let Some(native) = self.native_420.as_mut() {
      native.reset();
    }
    // New frame: clear the per-frame frozen native/row-stage route so the
    // next frame may pick either tier; a mid-frame flip stays rejected.
    // Gated to the native tier's feature (the route guard only exists when
    // the planar join the native tier reuses is compiled in). The RFC #238 S3b
    // frozen chroma-siting phase is cleared alongside it, so the next frame may
    // pick either siting while a mid-frame phase flip stays rejected.
    #[cfg(feature = "yuv-planar")]
    {
      self.frozen_native_route = None;
      self.frozen_chroma_centered = None;
      // RFC #238 S4-C companion: clear the frozen vertical phase too, so the
      // next frame may pick Center or Bottom and a mid-frame Center ⇆ Bottom
      // flip stays rejected.
      self.frozen_chroma_bottom_v = None;
      // New frame: invalidate the bottom-sited vertical-phase chroma lookback
      // (#302) so frame N+1's first even row can never box-blend frame N's last
      // chroma row. The buffer bytes are left as-is (re-overwritten before any
      // trusted read); only the validity tag is cleared.
      self.chroma_prev_row = None;
    }
    self.resample_outputs = None;
    Ok(())
  }

  fn process(&mut self, row: Nv21Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;
    // Chroma siting (#302) — see the Nv12 impl. `Copy`; read before the
    // split-borrow, gated like its `yuv-planar` consumer.
    #[cfg(feature = "yuv-planar")]
    let chroma_location = self.chroma_location;

    // Defense in depth: same shape check as the Nv12 impl. A VU row
    // has `width` bytes of interleaved V / U payload — same length
    // as Y — so both slices must equal `self.width`.
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
    if row.vu_half().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::VuHalf,
        idx,
        w,
        row.vu_half().len(),
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
      #[cfg(feature = "yuv-planar")]
      native,
      #[cfg(feature = "yuv-planar")]
      native_420,
      #[cfg(feature = "yuv-planar")]
      semi_planar_u_half,
      #[cfg(feature = "yuv-planar")]
      semi_planar_v_half,
      #[cfg(feature = "yuv-planar")]
      frozen_native_route,
      // RFC #238 S3b: the 4:2:0 chroma phase frozen on the first output row.
      #[cfg(feature = "yuv-planar")]
      frozen_chroma_centered,
      // RFC #238 S4-C: the vertical (`Bottom`) chroma phase frozen alongside it.
      #[cfg(feature = "yuv-planar")]
      frozen_chroma_bottom_v,
      // Full-width chroma staging for the centered-siting (#302) identity
      // decode; reuses the half-width de-interleave scratch above.
      #[cfg(feature = "yuv-planar")]
      chroma_full,
      // RFC #238 S4-C: the one-row bottom-sited vertical-phase chroma lookback
      // (de-interleaved U then V) + its validity tag, so a `Bottom` even output
      // row box-blends the previous chroma row.
      #[cfg(feature = "yuv-planar")]
      chroma_prev,
      #[cfg(feature = "yuv-planar")]
      chroma_prev_row,
      ..
    } = self;

    // Non-identity plan. A `Filter` plan routes to the filter resampler
    // (branched first, below). Otherwise, when the native tier is enabled
    // (and the planar join it reuses is compiled in), bin the native
    // Y / U / V planes at output resolution and convert once per output
    // row, de-interleaving the NV21 VU chroma row into U / V scratch first.
    // Otherwise (or under `with_native(false)`) take the row-stage tier
    // (matches the Yuv420p row-stage twin): bin Y for luma; for colour,
    // convert the interleaved VU source row to RGB with the fused
    // `nv21_to_rgb_row` kernel the identity path uses, then bin the RGB row.
    if let Some(plan) = plan.as_ref() {
      let matrix = row.matrix();
      let full_range = row.full_range();
      let convert_rgb = |scratch: &mut [u8]| {
        nv21_to_rgb_row(
          row.y(),
          row.vu_half(),
          scratch,
          w,
          matrix,
          full_range,
          use_simd,
        );
      };
      // RFC #238 S3b — 4:2:0 horizontal chroma siting for Nv21, mirroring the
      // planar Yuv420p twin (S3a) and the 4:2:2 Nv16 sibling (S2a). The centered
      // group (`Center` / `Top` / `Bottom`, [`chroma_420_center_sited_h`]) samples
      // chroma at `+0.25` chroma-sample horizontally; the vertical pairing stays
      // co-sited (`v_phase = 0`, `Bottom`'s vertical blend is a later stage). The
      // co-sited / unspecified group is phase 0 (byte-identical to the pre-siting
      // resample). The native fast tier folds the phase into the chroma area
      // weights (`area_chroma_420`); the filter and row-stage tiers reconstruct
      // full-width chroma (de-interleave + phase-0.5 upsample) and decode 4:4:4.
      // `convert_rgb` above is the co-sited fused decode.
      #[cfg(feature = "yuv-planar")]
      let center_sited = chroma_420_center_sited_h(chroma_location);
      #[cfg(feature = "yuv-planar")]
      let bottom_v = chroma_420_bottom_sited_v(chroma_location);
      #[cfg(feature = "yuv-planar")]
      let chroma_h_phase = if center_sited {
        YUV422P_CENTERED_H_PHASE
      } else {
        0.0
      };
      #[cfg(feature = "yuv-planar")]
      let chroma_v_phase = if bottom_v { 1.0 } else { 0.0 };
      #[cfg(feature = "yuv-planar")]
      let want_color = rgb.is_some() || rgba.is_some() || hsv.is_some();
      // Whether this call carries any output — the EXACT set both tiers' preflight
      // tests. The route (and the siting phase) freezes only on an output-bearing
      // row a tier ACCEPTS; a no-output call must not freeze.
      #[cfg(feature = "yuv-planar")]
      let need_output =
        luma.is_some() || luma_u16.is_some() || rgb.is_some() || rgba.is_some() || hsv.is_some();
      // RFC #238 S3b: freeze the effective 4:2:0 chroma siting on the first
      // output-bearing row (see the Nv12 impl above for the full choke-point
      // rationale). A later row observing a different phase — in sequence or not —
      // would bin a mixture of co-sited and centered chroma, so it is rejected
      // HERE before any reconstruction or dispatch; the matching SET rides each
      // tier's accept-time freeze below.
      #[cfg(feature = "yuv-planar")]
      if need_output
        && let Some(frozen) = *frozen_chroma_centered
        && (frozen != center_sited || *frozen_chroma_bottom_v != Some(bottom_v))
      {
        return Err(MixedSinkerError::ChromaSitingChanged(
          ChromaSitingChanged::new(idx),
        ));
      }
      // A `Filter` plan routes to the filter resampler (the same `nv21_to_rgb_row`
      // kernel feeds the RGB-space filter, plus the native Y). The native fast tier
      // is area-only and never sees a filter plan; the per-sink plan kind is fixed
      // at construction, so a filter sink bypasses the native/row-stage route
      // machinery entirely. Branched FIRST, before the native-route guard below.
      if plan.kind().is_filter() {
        // Centered filter reconstructs full-width chroma and decodes 4:4:4, but
        // ONLY after the resample preflight (frozen-output + sequence), so an
        // out-of-sequence / rejected row is caught before the chroma reservation
        // (#180). `planar_dual_filter_resample` re-runs the idempotent preflight.
        // Co-sited keeps the fused `convert_rgb` decode.
        #[cfg(feature = "yuv-planar")]
        {
          plan.ensure_single_kernel_filter()?;
          if (center_sited || bottom_v) && want_color {
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
            reserve_nv_chroma_half(semi_planar_u_half, semi_planar_v_half, w, h)?;
            // RFC #238 S4-C: reserve the bottom-sited vertical-phase
            // lookback AFTER the preflight, BEFORE the commit (#180); only
            // for v=1. The sited call reads it but defers staging.
            if bottom_v {
              reserve_420_chroma_prev(chroma_prev, w, h)?;
            }
            let (u_full, v_full) = nv_center_upsample_chroma(
              chroma_full,
              chroma_prev,
              chroma_prev_row,
              semi_planar_u_half,
              semi_planar_v_half,
              row.vu_half(),
              idx,
              bottom_v,
              center_sited,
              false,
              w,
              true,
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
            // RFC #238 S4-C: advance the bottom lookback only AFTER the
            // filter resample accepts the row (#180 ordering) — the sited
            // reconstruction read it above but did not stage, so a rejected
            // row keeps the predecessor for a clean retry. The de-interleaved
            // chroma is in the half scratch from the upsample.
            if r.is_ok() && bottom_v {
              stage_420_chroma_prev(
                chroma_prev,
                chroma_prev_row,
                semi_planar_u_half,
                semi_planar_v_half,
                idx,
                w,
              );
            }
            if r.is_ok() && need_output && frozen_chroma_centered.is_none() {
              *frozen_chroma_centered = Some(center_sited);
              *frozen_chroma_bottom_v = Some(bottom_v);
            }
            return r;
          }
        }
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
          convert_rgb,
        )?;
        #[cfg(feature = "yuv-planar")]
        if need_output && frozen_chroma_centered.is_none() {
          *frozen_chroma_centered = Some(center_sited);
          *frozen_chroma_bottom_v = Some(bottom_v);
        }
        return Ok(());
      }
      // Reject a mid-frame native/row-stage route flip BEFORE either tier's
      // dispatch (see the Nv12 impl above for the full rationale; both the CHECK
      // here and the SET below gate on `need_output`, so a no-output call is a true
      // route-invisible no-op and a preflight-rejected row leaves the route
      // unfrozen).
      // The RFC #238 splice stage. A filter plan already returned above, so
      // `area_plan` is true and the selector reproduces the former `*native`
      // boolean bit-for-bit (`cfg!` is true wherever this block compiles).
      #[cfg(feature = "yuv-planar")]
      let take_native = matches!(
        select_insertion_point(
          AveragingDomain::Encoded,
          InsertionContext {
            native_eligible: cfg!(feature = "yuv-planar"),
            with_native: *native,
            area_plan: true,
          },
        ),
        InsertionPoint::NativeCodes
      );
      #[cfg(feature = "yuv-planar")]
      if need_output
        && let Some(frozen) = *frozen_native_route
        && frozen != take_native
      {
        return Err(MixedSinkerError::NativeRouteChanged(
          NativeRouteChanged::new(idx),
        ));
      }
      #[cfg(feature = "yuv-planar")]
      if take_native {
        // RFC #238 S3b point-of-use siting invalidation, mirroring the Yuv420p
        // native arm: `chroma_location` can change at ANY point before this row
        // (including AFTER `begin_frame`, before row 0), so re-check the cached
        // join HERE and drop it when its folded chroma plan was built for a
        // different phase; `semi_planar_process_native` (via
        // `yuv420p_process_native`) then rebuilds it with the current siting.
        // Retry-atomic: drop ONLY on the IN-SEQUENCE fresh-frame first row
        // (`idx == 0`, `next_y() == 0`); an out-of-sequence first row is left for
        // the delegate to reject against the INTACT join. A luma-only join carries
        // no chroma plan (siting-independent), a no-output sink built no join.
        // Transactional: move the stale join OUT, let the delegate build the
        // replacement into `native_420` (its build runs BEFORE it inserts), and
        // restore the intact prior-phase join on a rejected rebuild so the
        // REJECTED row mutates nothing.
        let stale_native = idx == 0
          && native_420.as_ref().is_some_and(|join| {
            join.has_chroma()
              && (join.chroma_centered() != center_sited || join.chroma_bottom() != bottom_v)
              && join.next_y() == 0
          });
        let prev_native = if stale_native {
          native_420.take()
        } else {
          None
        };
        // Dispatch first; freeze the route + siting ONLY after the call returns Ok
        // on an output-bearing row. A no-output call returns Ok(()) with
        // `need_output` false (no freeze); an out-of-sequence / frozen row returns
        // Err via `?` (no freeze) — so only an accepted output-bearing row commits.
        let native_result = semi_planar_process_native(
          plan,
          native_420,
          semi_planar_u_half,
          semi_planar_v_half,
          resample_outputs,
          rgb,
          rgba,
          luma,
          luma_u16,
          hsv,
          rgb_scratch,
          row.y(),
          row.vu_half(),
          true,
          chroma_h_phase,
          chroma_v_phase,
          matrix,
          full_range,
          idx,
          w,
          h,
          use_simd,
        );
        // Restore the taken stale-phase join if the delegate's rebuild was rejected
        // at any pre-feed step: it leaves the field `None` and `resample_outputs`
        // uncommitted on such a failure, so restoring the intact prior-phase join
        // leaves the rejected row mutating nothing. A non-stale row (first-ever /
        // colour-capability rebuild) took nothing.
        if stale_native && native_result.is_err() {
          *native_420 = prev_native;
        }
        native_result?;
        if frozen_native_route.is_none() && need_output {
          *frozen_native_route = Some(true);
        }
        // RFC #238 S3b: freeze the siting on the same accepted output row.
        if frozen_chroma_centered.is_none() && need_output {
          *frozen_chroma_centered = Some(center_sited);
          *frozen_chroma_bottom_v = Some(bottom_v);
        }
        return Ok(());
      }
      // Row-stage tail. Same CHECK-before / SET-after split: dispatch, then freeze
      // the route + siting only when the call accepts an output-bearing row.
      // Centered colour reconstructs full-width chroma (de-interleave + phase-0.5
      // upsample) and decodes 4:4:4 — but ONLY after the resample preflight
      // (frozen-output + sequence), so an out-of-sequence / rejected row is caught
      // before the chroma reservation (#180). A luma-only centered row never calls
      // the RGB converter, so it stays on the co-sited arm (which only bins luma).
      // `planar_dual_resample` re-runs the idempotent preflight.
      #[cfg(feature = "yuv-planar")]
      if (center_sited || bottom_v) && want_color {
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
        reserve_nv_chroma_half(semi_planar_u_half, semi_planar_v_half, w, h)?;
        // RFC #238 S4-C: reserve the bottom-sited vertical-phase lookback
        // AFTER the preflight, BEFORE the commit (#180); only for v=1. The
        // sited call reads it but defers staging until the commit accepts.
        if bottom_v {
          reserve_420_chroma_prev(chroma_prev, w, h)?;
        }
        let (u_full, v_full) = nv_center_upsample_chroma(
          chroma_full,
          chroma_prev,
          chroma_prev_row,
          semi_planar_u_half,
          semi_planar_v_half,
          row.vu_half(),
          idx,
          bottom_v,
          center_sited,
          false,
          w,
          true,
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
        // RFC #238 S4-C: advance the bottom lookback only AFTER the
        // row-stage resample accepts the row (the `?` above already returned
        // any reject); the sited reconstruction read it but did not stage,
        // so a rejected row keeps the predecessor for a clean retry.
        if bottom_v {
          stage_420_chroma_prev(
            chroma_prev,
            chroma_prev_row,
            semi_planar_u_half,
            semi_planar_v_half,
            idx,
            w,
          );
        }
        if frozen_native_route.is_none() && need_output {
          *frozen_native_route = Some(false);
        }
        if frozen_chroma_centered.is_none() && need_output {
          *frozen_chroma_centered = Some(center_sited);
          *frozen_chroma_bottom_v = Some(bottom_v);
        }
        return Ok(());
      }
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
      #[cfg(feature = "yuv-planar")]
      if frozen_native_route.is_none() && need_output {
        *frozen_native_route = Some(false);
      }
      #[cfg(feature = "yuv-planar")]
      if frozen_chroma_centered.is_none() && need_output {
        *frozen_chroma_centered = Some(center_sited);
        *frozen_chroma_bottom_v = Some(bottom_v);
      }
      return Ok(());
    }

    // Strategy A output mode resolution — see Yuv420p impl above. Computed
    // BEFORE any per-row offset arithmetic so the no-output guard below
    // short-circuits before it runs ANY row math (`need_output` depends only on
    // the attached buffers, never on `idx` / `w`).
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();
    let want_color = want_rgb || want_rgba || want_hsv;

    // Repo-wide no-output invariant (mirrors the Yuv420p S4-B direct twin): a
    // `process` call carrying NO output — no colour, no luma, no luma_u16 — must
    // run NOTHING: no per-row offset arithmetic, no allocation, no state mutation
    // (the bottom-sited vertical lookback included). Returning HERE, before the
    // `idx * w` offsets below, makes those single-plane ranges overflow-safe (a
    // no-output call never ran an attach-time `w x h x 1` validation, so `idx * w`
    // could overflow `usize` on a 32-bit target) and guarantees such an invisible
    // row never reserves `chroma_prev` nor primes the lookback.
    let need_output = want_color || luma.is_some() || luma_u16.is_some();
    if !need_output {
      return Ok(());
    }

    // Single-plane row ranges are guaranteed to fit: the no-output guard above
    // ensures ≥ 1 output is attached, so its attach-time `w x h x 1` validation
    // ran and `w x h` fits `usize`; with `idx < h` that makes `(idx + 1) * w` fit
    // too. RGB / RGBA (x3 / x4) ranges use checked arithmetic below.
    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // Chroma siting (#302) — the centered horizontal sitings reconstruct
    // chroma at phase-0.5; the default / co-sited path stays byte-identical.
    #[cfg(feature = "yuv-planar")]
    let center_sited = chroma_420_center_sited_h(chroma_location);
    #[cfg(feature = "yuv-planar")]
    let bottom_v = chroma_420_bottom_sited_v(chroma_location);

    // Atomicity preflight (#302 / #308) — see the Nv12 impl for the full
    // rationale: reserve EVERY fallible row scratch this row needs BEFORE any
    // output row (luma / luma_u16 included) is written, so an allocator refusal
    // returns a typed `AllocationFailed` leaving the output frame untouched.
    // Centered siting grows the full-width chroma (`chroma_full`) + half-width
    // de-interleave scratch; the colour decode grows the RGB row buffer exactly
    // when `want_hsv && want_rgba && !want_rgb`.
    #[cfg(feature = "yuv-planar")]
    if (center_sited || bottom_v) && (want_rgb || want_rgba || want_hsv) {
      reserve_420_chroma_full(chroma_full, w, h)?;
      reserve_nv_chroma_half(semi_planar_u_half, semi_planar_v_half, w, h)?;
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
    // RFC #238 S4-C: on every bottom output row reserve the vertical-phase
    // lookback + de-interleave half scratch BEFORE any luma write (#180/#302),
    // so a later colour row can box-blend the previous chroma row. A colour row
    // stages inside `nv_center_upsample_chroma` after reading the lookback; a
    // LUMA-ONLY row has no colour upsample, so it de-interleaves + stages here.
    // (A no-output row already returned at the guard above, so it never reaches
    // this reserve / stage.)
    #[cfg(feature = "yuv-planar")]
    if bottom_v {
      reserve_420_chroma_prev(chroma_prev, w, h)?;
      reserve_nv_chroma_half(semi_planar_u_half, semi_planar_v_half, w, h)?;
      if !want_color {
        nv_deinterleave_chroma_half(
          semi_planar_u_half,
          semi_planar_v_half,
          row.vu_half(),
          w,
          true,
        );
        stage_420_chroma_prev(
          chroma_prev,
          chroma_prev_row,
          semi_planar_u_half,
          semi_planar_v_half,
          idx,
          w,
        );
      }
    }

    if let Some(luma) = luma.as_deref_mut() {
      nv_to_luma_row(row.y(), &mut luma[one_plane_start..one_plane_end], w);
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

    // HSV-only (no RGB / RGBA) goes direct through `nv21_to_hsv_row`
    // (no source-width RGB scratch); see the Nv12 impl for the rationale.
    let want_hsv_direct = want_hsv && !want_rgb && !want_rgba;
    let need_rgb_kernel = want_rgb || (want_hsv && want_rgba);

    if want_hsv_direct {
      let hsv = hsv.as_mut().expect("want_hsv_direct implies hsv attached");
      let (h, s, v) = hsv.hsv();
      // Sited siting (#302): de-interleave (VU) + full-width upsample chroma
      // (centered phase-0.5 or co-sited nearest-neighbor + `BottomLeft`'s `v = 1`
      // fold), then the 4:4:4 HSV kernel. `swap_uv = true` for NV21.
      #[cfg(feature = "yuv-planar")]
      if center_sited || bottom_v {
        let (u_full, v_full) = nv_center_upsample_chroma(
          chroma_full,
          chroma_prev,
          chroma_prev_row,
          semi_planar_u_half,
          semi_planar_v_half,
          row.vu_half(),
          idx,
          bottom_v,
          center_sited,
          true,
          w,
          true,
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
        return Ok(());
      }
      nv21_to_hsv_row(
        row.y(),
        row.vu_half(),
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
      // Sited siting (#302): full-width chroma (centered phase-0.5 or co-sited
      // nearest-neighbor + `BottomLeft`'s `v = 1` fold) + the 4:4:4 RGBA kernel;
      // the default co-sited path keeps the fused `nv21_to_rgba_row`.
      #[cfg(feature = "yuv-planar")]
      if center_sited || bottom_v {
        let (u_full, v_full) = nv_center_upsample_chroma(
          chroma_full,
          chroma_prev,
          chroma_prev_row,
          semi_planar_u_half,
          semi_planar_v_half,
          row.vu_half(),
          idx,
          bottom_v,
          center_sited,
          true,
          w,
          true,
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
        return Ok(());
      }
      nv21_to_rgba_row(
        row.y(),
        row.vu_half(),
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

    // Fused NV21 → RGB: VU deinterleave + chroma upsample both happen
    // in registers inside the row primitive, no intermediate memory.
    // Sited siting (#302) instead de-interleaves (VU) + upsamples chroma to full
    // width (centered phase-0.5 or co-sited nearest-neighbor + `BottomLeft`'s
    // `v = 1` fold) and runs the 4:4:4 kernel; the HSV / RGBA follow-ups below
    // derive off the produced RGB row either way.
    #[cfg(feature = "yuv-planar")]
    let centered = if center_sited || bottom_v {
      let (u_full, v_full) = nv_center_upsample_chroma(
        chroma_full,
        chroma_prev,
        chroma_prev_row,
        semi_planar_u_half,
        semi_planar_v_half,
        row.vu_half(),
        idx,
        bottom_v,
        center_sited,
        true,
        w,
        true,
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
      true
    } else {
      false
    };
    #[cfg(not(feature = "yuv-planar"))]
    let centered = false;
    if !centered {
      nv21_to_rgb_row(
        row.y(),
        row.vu_half(),
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

// ---- Nv24 impl ----------------------------------------------------------
//
// 4:4:4 semi-planar: UV plane is full-width (`2 * width` bytes per
// row), one UV pair per Y pixel. No width parity constraint. Kernel
// is its own family (`nv24_to_rgb_row`) since chroma is no longer
// duplicated across columns.

impl<'a, R> MixedSinker<'a, Nv24, R> {
  /// Attaches a packed 32‑bit RGBA output buffer.
  ///
  /// Only available on sinker types whose `PixelSink` impl writes
  /// RGBA — see [`MixedSinker::<Yuv420p>::with_rgba`] for the same
  /// rationale and constraints. Nv24 has no alpha plane, so every
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

impl<R> Nv24Sink for MixedSinker<'_, Nv24, R> {}

impl<R> PixelSink for MixedSinker<'_, Nv24, R> {
  type Input<'r> = Nv24Row<'r>;
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
    #[cfg(feature = "yuv-planar")]
    if let Some(native) = self.native_planar.as_mut() {
      native.reset();
    }
    // New frame: clear the per-frame frozen native/row-stage route (see the
    // Nv16 impl); gated to the native tier's feature.
    #[cfg(feature = "yuv-planar")]
    {
      self.frozen_native_route = None;
    }
    self.resample_outputs = None;
    Ok(())
  }

  fn process(&mut self, row: Nv24Row<'_>) -> Result<(), Self::Error> {
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
    // NV24 UV row is `2 * width` bytes. `checked_mul` covers the
    // boundary where `2 * width` could overflow `usize` on 32-bit
    // targets with very large widths.
    let uv_expected =
      w.checked_mul(2)
        .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
          w, h, 2,
        )))?;
    if row.uv().len() != uv_expected {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::UvFull,
        idx,
        uv_expected,
        row.uv().len(),
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
      #[cfg(feature = "yuv-planar")]
      native,
      #[cfg(feature = "yuv-planar")]
      native_planar,
      #[cfg(feature = "yuv-planar")]
      semi_planar_u_half,
      #[cfg(feature = "yuv-planar")]
      semi_planar_v_half,
      #[cfg(feature = "yuv-planar")]
      frozen_native_route,
      ..
    } = self;

    // Non-identity plan. A `Filter` plan routes to the filter resampler
    // (branched first, below). Otherwise, when the native tier is enabled
    // (and the planar join it reuses is compiled in), bin the native
    // Y / U / V planes at output resolution and convert once per output row,
    // de-interleaving the NV24 full-width UV chroma row into U / V scratch
    // first (4:4:4 -> chroma `w x h`). Otherwise (or under
    // `with_native(false)`) take the row-stage tier (matches the Yuv444p
    // twin): bin Y for luma; for colour, convert the interleaved full-width
    // UV source row to RGB with the fused `nv24_to_rgb_row` kernel the
    // identity path uses, then bin the RGB row.
    if let Some(plan) = plan.as_ref() {
      let matrix = row.matrix();
      let full_range = row.full_range();
      let convert_rgb = |scratch: &mut [u8]| {
        nv24_to_rgb_row(row.y(), row.uv(), scratch, w, matrix, full_range, use_simd);
      };
      // A `Filter` plan routes to the filter resampler; the native fast tier
      // is area-only (see the Nv16 impl). Branched FIRST, before the
      // native-route guard below.
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
      #[cfg(feature = "yuv-planar")]
      let need_output =
        luma.is_some() || luma_u16.is_some() || rgb.is_some() || rgba.is_some() || hsv.is_some();
      // The RFC #238 splice stage. A filter plan already returned above, so
      // `area_plan` is true and the selector reproduces the former `*native`
      // boolean bit-for-bit (`cfg!` is true wherever this block compiles).
      #[cfg(feature = "yuv-planar")]
      let take_native = matches!(
        select_insertion_point(
          AveragingDomain::Encoded,
          InsertionContext {
            native_eligible: cfg!(feature = "yuv-planar"),
            with_native: *native,
            area_plan: true,
          },
        ),
        InsertionPoint::NativeCodes
      );
      #[cfg(feature = "yuv-planar")]
      if need_output
        && let Some(frozen) = *frozen_native_route
        && frozen != take_native
      {
        return Err(MixedSinkerError::NativeRouteChanged(
          NativeRouteChanged::new(idx),
        ));
      }
      #[cfg(feature = "yuv-planar")]
      if take_native {
        semi_planar_process_native_non420(
          plan,
          native_planar,
          semi_planar_u_half,
          semi_planar_v_half,
          resample_outputs,
          rgb,
          rgba,
          luma,
          luma_u16,
          hsv,
          rgb_scratch,
          row.y(),
          row.uv(),
          w,
          false,
          // 4:4:4 has no horizontal chroma subsampling, so no sampling phase.
          0.0,
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
      #[cfg(feature = "yuv-planar")]
      if frozen_native_route.is_none() && need_output {
        *frozen_native_route = Some(false);
      }
      return Ok(());
    }

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // Strategy A output mode resolution — see Yuv420p impl above.
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();

    // Atomicity preflight (#308, cf. the crate's #180 resample fix and the
    // planar_8bit / Yuv420p siblings): reserve EVERY fallible row scratch this
    // row needs BEFORE any output row (luma / luma_u16 included) is written, so
    // an allocator refusal returns a typed `AllocationFailed` leaving the output
    // frame untouched rather than partially mutated. The only growable scratch
    // here is the RGB row buffer, reserved exactly when a colour decode needs an
    // RGB row but no caller RGB buffer is borrowable — `want_hsv && want_rgba &&
    // !want_rgb` (`rgb_row_buf_or_scratch`'s own scratch arm; an attached RGB
    // buffer is borrowed instead and never allocates). The later decode call
    // then reuses the already-sized buffer.
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
      nv_to_luma_row(row.y(), &mut luma[one_plane_start..one_plane_end], w);
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

    // HSV-only (no RGB / RGBA) goes direct through `nv24_to_hsv_row`
    // (no source-width RGB scratch); see the Nv12 impl for the rationale.
    let want_hsv_direct = want_hsv && !want_rgb && !want_rgba;
    let need_rgb_kernel = want_rgb || (want_hsv && want_rgba);

    if want_hsv_direct {
      let hsv = hsv.as_mut().expect("want_hsv_direct implies hsv attached");
      let (h, s, v) = hsv.hsv();
      nv24_to_hsv_row(
        row.y(),
        row.uv(),
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

    // Standalone RGBA path: the caller wants only RGBA (no RGB / HSV),
    // so run the dedicated RGBA kernel directly into the output buffer.
    // Avoids both the scratch allocation and the expand-pad pass.
    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      nv24_to_rgba_row(
        row.y(),
        row.uv(),
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

    nv24_to_rgb_row(
      row.y(),
      row.uv(),
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

    // Strategy A: when both RGB-side and RGBA outputs are requested,
    // derive RGBA from the just-computed RGB row (memory-bound copy +
    // 0xFF alpha pad) instead of running a second YUV→RGB kernel.
    if let Some(buf) = rgba.as_deref_mut() {
      let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
    }

    Ok(())
  }
}

// ---- Nv42 impl ----------------------------------------------------------
//
// Structurally identical to the Nv24 impl — the row primitive hides
// the V/U byte-order difference.

impl<'a, R> MixedSinker<'a, Nv42, R> {
  /// Attaches a packed 32‑bit RGBA output buffer.
  ///
  /// See [`MixedSinker::<Nv24>::with_rgba`] for the same rationale and
  /// constraints; Nv42 differs only in chroma byte order (V before U).
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

impl<R> Nv42Sink for MixedSinker<'_, Nv42, R> {}

impl<R> PixelSink for MixedSinker<'_, Nv42, R> {
  type Input<'r> = Nv42Row<'r>;
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
    #[cfg(feature = "yuv-planar")]
    if let Some(native) = self.native_planar.as_mut() {
      native.reset();
    }
    // New frame: clear the per-frame frozen native/row-stage route (see the
    // Nv16 impl); gated to the native tier's feature.
    #[cfg(feature = "yuv-planar")]
    {
      self.frozen_native_route = None;
    }
    self.resample_outputs = None;
    Ok(())
  }

  fn process(&mut self, row: Nv42Row<'_>) -> Result<(), Self::Error> {
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
    let vu_expected =
      w.checked_mul(2)
        .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
          w, h, 2,
        )))?;
    if row.vu().len() != vu_expected {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::VuFull,
        idx,
        vu_expected,
        row.vu().len(),
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
      #[cfg(feature = "yuv-planar")]
      native,
      #[cfg(feature = "yuv-planar")]
      native_planar,
      #[cfg(feature = "yuv-planar")]
      semi_planar_u_half,
      #[cfg(feature = "yuv-planar")]
      semi_planar_v_half,
      #[cfg(feature = "yuv-planar")]
      frozen_native_route,
      ..
    } = self;

    // Non-identity plan. A `Filter` plan routes to the filter resampler
    // (branched first, below). Otherwise, when the native tier is enabled
    // (and the planar join it reuses is compiled in), bin the native
    // Y / U / V planes at output resolution and convert once per output row,
    // de-interleaving the NV42 full-width VU chroma row into U / V scratch
    // first with U / V SWAPPED (4:4:4 VU-order -> chroma `w x h`). Otherwise
    // (or under `with_native(false)`) take the row-stage tier (matches the
    // Yuv444p twin): convert the interleaved VU source row to RGB with the
    // fused `nv42_to_rgb_row` kernel the identity path uses, then bin it.
    if let Some(plan) = plan.as_ref() {
      let matrix = row.matrix();
      let full_range = row.full_range();
      let convert_rgb = |scratch: &mut [u8]| {
        nv42_to_rgb_row(row.y(), row.vu(), scratch, w, matrix, full_range, use_simd);
      };
      // A `Filter` plan routes to the filter resampler; the native fast tier
      // is area-only (see the Nv16 impl). Branched FIRST, before the
      // native-route guard below.
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
      #[cfg(feature = "yuv-planar")]
      let need_output =
        luma.is_some() || luma_u16.is_some() || rgb.is_some() || rgba.is_some() || hsv.is_some();
      // The RFC #238 splice stage. A filter plan already returned above, so
      // `area_plan` is true and the selector reproduces the former `*native`
      // boolean bit-for-bit (`cfg!` is true wherever this block compiles).
      #[cfg(feature = "yuv-planar")]
      let take_native = matches!(
        select_insertion_point(
          AveragingDomain::Encoded,
          InsertionContext {
            native_eligible: cfg!(feature = "yuv-planar"),
            with_native: *native,
            area_plan: true,
          },
        ),
        InsertionPoint::NativeCodes
      );
      #[cfg(feature = "yuv-planar")]
      if need_output
        && let Some(frozen) = *frozen_native_route
        && frozen != take_native
      {
        return Err(MixedSinkerError::NativeRouteChanged(
          NativeRouteChanged::new(idx),
        ));
      }
      #[cfg(feature = "yuv-planar")]
      if take_native {
        // NV42 chroma is VU-order: de-interleave with U / V swapped
        // (`swap_uv = true`), mirroring the NV21 4:2:0 path.
        semi_planar_process_native_non420(
          plan,
          native_planar,
          semi_planar_u_half,
          semi_planar_v_half,
          resample_outputs,
          rgb,
          rgba,
          luma,
          luma_u16,
          hsv,
          rgb_scratch,
          row.y(),
          row.vu(),
          w,
          true,
          // 4:4:4 has no horizontal chroma subsampling, so no sampling phase.
          0.0,
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
      #[cfg(feature = "yuv-planar")]
      if frozen_native_route.is_none() && need_output {
        *frozen_native_route = Some(false);
      }
      return Ok(());
    }

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // Strategy A output mode resolution — see Yuv420p impl above.
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();

    // Atomicity preflight (#308, cf. the crate's #180 resample fix and the
    // planar_8bit / Yuv420p siblings): reserve EVERY fallible row scratch this
    // row needs BEFORE any output row (luma / luma_u16 included) is written, so
    // an allocator refusal returns a typed `AllocationFailed` leaving the output
    // frame untouched rather than partially mutated. The only growable scratch
    // here is the RGB row buffer, reserved exactly when a colour decode needs an
    // RGB row but no caller RGB buffer is borrowable — `want_hsv && want_rgba &&
    // !want_rgb` (`rgb_row_buf_or_scratch`'s own scratch arm; an attached RGB
    // buffer is borrowed instead and never allocates). The later decode call
    // then reuses the already-sized buffer.
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
      nv_to_luma_row(row.y(), &mut luma[one_plane_start..one_plane_end], w);
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

    // HSV-only (no RGB / RGBA) goes direct through `nv42_to_hsv_row`
    // (no source-width RGB scratch); see the Nv12 impl for the rationale.
    let want_hsv_direct = want_hsv && !want_rgb && !want_rgba;
    let need_rgb_kernel = want_rgb || (want_hsv && want_rgba);

    if want_hsv_direct {
      let hsv = hsv.as_mut().expect("want_hsv_direct implies hsv attached");
      let (h, s, v) = hsv.hsv();
      nv42_to_hsv_row(
        row.y(),
        row.vu(),
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
      nv42_to_rgba_row(
        row.y(),
        row.vu(),
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

    nv42_to_rgb_row(
      row.y(),
      row.vu(),
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
