use super::super::{
  GeometryOverflow, InsufficientBuffer, MixedSinker, MixedSinkerError, RowIndexOutOfRange,
  RowShapeMismatch, RowSlice, WidthAlignment, check_dimensions_match,
  packed_yuv422_triple_filter_resample, packed_yuv422_triple_resample, reset_high_bit_yuv_streams,
  rgb_row_buf_or_scratch, rgba_plane_row_slice, rgba_u16_plane_row_slice,
};
use crate::{PixelSink, row::*, source::*};

// `NativeRouteChanged` is raised only by the native fast tier's route-flip
// guard, which exists only when the reused planar join is compiled in.
// `ChromaSitingChanged` rejects a mid-frame horizontal siting flip; the shared
// `0.25` centered phase and the check-only resample preflight back the centered
// filter / row-stage arms (RFC #238 S6b).
#[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
use super::super::{
  ChromaSitingChanged, FrozenOutputs, HsvFrameMut, NativeRouteChanged,
  planar_8bit::YUV422P_CENTERED_H_PHASE, resample_preflight_check_only,
};
#[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
use super::native::{
  NativeYuv420U16, yuv420p16_native_preflight_check_only, yuv420p16_process_native,
};
// The RFC #238 insertion-point selector decides the native-vs-row-stage
// splice; it is consulted in the `yuv-planar` dispatch block, so its import
// matches that gate exactly.
#[cfg(feature = "yuv-planar")]
use crate::resample::{AveragingDomain, InsertionContext, InsertionPoint, select_insertion_point};
#[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
use crate::{
  ColorMatrix,
  resample::{PlanGeometry, ResampleError, ResamplePlan},
};

// The centered-siting (#302) horizontal chroma upsample needs BOTH features:
// the predicate + the full-width `u16` chroma scratch + the 4:4:4 P-format
// decode kernels are reachable only when the `yuv-planar` half (which gates
// `chroma_420_center_sited_h` and the centered staging machinery) is compiled
// alongside the `yuv-semi-planar` P-format sinks. A semi-planar-only build
// keeps the byte-identical default decode.
#[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
use super::super::{chroma_420_bottom_sited_v, chroma_420_center_sited_h};
// RFC #238 S6e — the bottom-sited vertical-phase chroma lookback reserve is
// shared verbatim with the planar high-bit path (same `width` `u16` grow, same
// `FORCE_CHROMA_PREV_ALLOC_FAILURE` failpoint): only the STAGE + reconstruction
// layout (interleaved `U V …` here vs the planar's separate U-then-V) differs,
// so those get P-format siblings below while the reserve reuses the twin.
#[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
use super::yuv420p::reserve_420_chroma_prev_u16;

// The native fast tier de-interleaves + DE-PACKS each wire plane into
// wrapper-owned host-native LOGICAL u16 scratch BEFORE handing it to the
// planar delegate, so the delegate's own `from_le` / `from_be` decode must
// be a no-op load on every host: pass `BE = HOST_NATIVE_BE` (= `from_ne`).
// Passing the source wire `BE` here would byte-swap the already-native
// scratch on a big-endian target — the exact bug `native.rs` warns about.
// Mirrors `native.rs` / `planar_gbr_f16`.
#[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
const HOST_NATIVE_BE: bool = cfg!(target_endian = "big");

// Test-only allocation failpoint for the wrapper-owned Y / U / V de-pack
// scratch grow in `p0xx_process_native`. Armed, the FIRST (Y) scratch grow
// of an output-bearing row returns the crate's recoverable
// `AllocationFailed` WITHOUT growing — so the atomicity regressions can
// prove the join's pre-feed preflight (out-of-sequence / frozen-output)
// runs BEFORE this fallible grow. Mirrors `native.rs`'s
// `FORCE_NATIVE_U16_ALLOC_FAILURE` and the semi-planar 8-bit
// `FORCE_DEINTERLEAVE_ALLOC_FAILURE`. Strictly test-only — the non-test
// build compiles this away entirely.
#[cfg(all(
  test,
  feature = "std",
  feature = "yuv-semi-planar",
  feature = "yuv-planar"
))]
std::thread_local! {
  static FORCE_P0XX_ALLOC_FAILURE: core::cell::Cell<bool> =
    const { core::cell::Cell::new(false) };
}

/// Arms the wrapper de-pack scratch allocation failpoint for the **next**
/// output-bearing high-bit semi-planar native row on the current thread.
/// The flag is consumed (take-on-read) by the first fallible scratch grow
/// that row reaches, so it fires exactly once and cannot leak into a later
/// test. Test-only.
#[cfg(all(
  test,
  feature = "std",
  feature = "yuv-semi-planar",
  feature = "yuv-planar"
))]
pub(crate) fn arm_p0xx_alloc_failure() {
  FORCE_P0XX_ALLOC_FAILURE.with(|f| f.set(true));
}

/// Grows a wrapper-owned de-pack scratch to `len` `u16` under the planner's
/// recoverable-allocation contract, optionally firing the test-only
/// failpoint (`fail = true` only on the FIRST grow of an output-bearing
/// row). Runs after the join's preflight clears, so a rejected row never
/// reaches it.
#[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
#[cfg_attr(not(tarpaulin), inline(always))]
fn grow_depack_scratch(
  scratch: &mut std::vec::Vec<u16>,
  len: usize,
  fail: bool,
  w: usize,
  h: usize,
  plan: &ResamplePlan,
) -> Result<(), MixedSinkerError> {
  // `fail` is consumed by the caller; on the non-test build it is `false`
  // and the whole branch compiles away.
  let _ = fail;
  if scratch.len() < len {
    #[cfg(all(
      test,
      feature = "std",
      feature = "yuv-semi-planar",
      feature = "yuv-planar"
    ))]
    if fail && FORCE_P0XX_ALLOC_FAILURE.with(|f| f.take()) {
      return Err(MixedSinkerError::Resample(ResampleError::AllocationFailed(
        PlanGeometry::new(w, h, plan.out_w(), plan.out_h()),
      )));
    }
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

/// Native fast-tier 4:2:0 decimator for the **high-bit semi-planar**
/// P-format family ([`P010`](crate::source::P010) /
/// [`P012`](crate::source::P012) / [`P016`](crate::source::P016)): bins the
/// native Y / U / V planes straight to the output grid and converts once
/// per output row at output resolution. The `u16` semi-planar twin of the
/// 8-bit [`semi_planar_process_native`](crate::sinker::mixed::semi_planar_8bit),
/// reusing the high-bit PLANAR join verbatim
/// ([`yuv420p16_process_native`]) after de-interleaving + DE-PACKING the
/// wire row into wrapper-owned host-native LOGICAL u16 scratch.
///
/// THE SEAM: [`yuv420p16_process_native`] treats its `y_row` / `u_half` /
/// `v_half` input as **low-packed LOGICAL** u16 — it decodes the wire
/// endianness via `from_le` / `from_be` but applies **no** high-bit shift.
/// P-format Y is HIGH-BIT-PACKED (`logical << (16 - BITS)`) and the UV
/// plane is INTERLEAVED + high-packed. So this wrapper must, per row,
/// decode the wire AND de-pack (`>> (16 - BITS)`) the Y, and decode +
/// de-interleave (`U,V` order) + de-pack EACH of U and V, into the
/// wrapper's host-native logical scratch — then delegate with
/// `BE = HOST_NATIVE_BE` so the delegate's internal decode is a no-op load
/// on every host. The de-pack MUST hit Y AND U AND V; at `BITS = 16` the
/// shift is `>> 0` (a harmless no-op — the per-format tests guard the
/// 10/12 shift the 16-bit no-op would mask).
///
/// `chroma_h_phase` / `chroma_v_phase` are the RFC #238 chroma sampling phases
/// folded into the chroma area weights ([`ResamplePlan::area_chroma_420`]):
/// `chroma_h_phase` is `0.25` for the centered 4:2:0 group
/// ([`chroma_420_center_sited_h`]) and `0.0` for co-sited / unspecified;
/// `chroma_v_phase` is `1.0` for the RFC #238 S6e `Bottom` VERTICAL siting
/// ([`chroma_420_bottom_sited_v`]) and `0.0` for every vertical-co-sited layout
/// (`Center` / `Top` / co-sited). At both phases `0.0` the folded plan is
/// byte-identical to the plain `area` plan, so co-sited / horizontal-only output
/// is untouched.
///
/// Atomicity (the nv12 / high-bit lesson): the join's COMPLETE pre-feed
/// preflight runs FIRST — `Ok(false)` no-op short-circuit, first-row
/// out-of-sequence, frozen-output — BEFORE any fallible scratch grow, so a
/// rejected row returns its deterministic typed error
/// (OutOfSequenceRow / ResampleOutputsChanged), never AllocationFailed,
/// and touches no caller output. The de-pack into scratch is infallible
/// and happens only after the preflight clears; the delegate re-runs the
/// identical preflight (idempotent) and owns the binning + conversion.
#[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
#[allow(clippy::too_many_arguments)]
fn p0xx_process_native<const BITS: u32, const BE: bool>(
  plan: &ResamplePlan,
  native_420_u16: &mut Option<std::boxed::Box<NativeYuv420U16>>,
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
  uv_half: &[u16],
  chroma_h_phase: f64,
  chroma_v_phase: f64,
  matrix: ColorMatrix,
  full_range: bool,
  idx: usize,
  w: usize,
  h: usize,
  use_simd: bool,
) -> Result<(), MixedSinkerError> {
  const {
    assert!(
      BITS > 8 && BITS <= 16,
      "BITS must be in (8, 16] for high-bit semi-planar 4:2:0 P-format"
    )
  };
  let need_luma = luma.is_some();
  let need_color =
    rgb.is_some() || rgba.is_some() || hsv.is_some() || rgb_u16.is_some() || rgba_u16.is_some();
  let cw = w / 2;

  // Run the planar join's COMPLETE compare-only pre-feed rejection preflight
  // FIRST — no-output short-circuit, first-row out-of-sequence, AND output-set
  // compare (mid-frame output change) — BEFORE any fallible scratch grow below,
  // so every rejection returns its deterministic typed error and leaves the
  // wrapper scratch untouched (the crate's preflight-atomicity contract).
  // Compare-only (no output-set freeze), so the de-pack reserve below stays a
  // genuine pre-commit step; the delegate re-runs the committing preflight and
  // owns the single commit.
  if !yuv420p16_native_preflight_check_only(
    native_420_u16,
    resample_outputs,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    luma,
    hsv,
    idx,
    need_luma,
    need_color,
  )? {
    return Ok(());
  }

  // Grow the wrapper de-pack scratch under the planner's recoverable
  // contract — Y always, U / V only on a chroma-bearing row (exactly where
  // the delegate reads chroma). All grows precede the infallible de-pack
  // and the delegate call. The failpoint fires on the FIRST (Y) grow only.
  grow_depack_scratch(y_scratch, w, true, w, h, plan)?;
  let chroma_row = need_color && idx.is_multiple_of(2);
  if chroma_row {
    grow_depack_scratch(u_scratch, cw, false, w, h, plan)?;
    grow_depack_scratch(v_scratch, cw, false, w, h, plan)?;
  }

  // De-pack the wire planes into host-native LOGICAL scratch. Decode the
  // wire endianness, then shift the active high `BITS` down to the low
  // `BITS` (`>> (16 - BITS)`; `>> 0` at BITS = 16). Everything past here is
  // infallible.
  for (d, &s) in y_scratch[..w].iter_mut().zip(y_row.iter()) {
    let logical = if BE { u16::from_be(s) } else { u16::from_le(s) };
    *d = logical >> (16 - BITS);
  }
  if chroma_row {
    // P-format chroma is interleaved `U,V,U,V…` (U at even element); each
    // of U and V is independently high-bit-packed and must be de-packed.
    for (i, pair) in uv_half.chunks_exact(2).enumerate() {
      let u = if BE {
        u16::from_be(pair[0])
      } else {
        u16::from_le(pair[0])
      };
      let v = if BE {
        u16::from_be(pair[1])
      } else {
        u16::from_le(pair[1])
      };
      u_scratch[i] = u >> (16 - BITS);
      v_scratch[i] = v >> (16 - BITS);
    }
  }

  // Delegate to the planar high-bit join with `BE = HOST_NATIVE_BE` so its
  // internal decode is a no-op on the already-native scratch. Empty U / V
  // on non-chroma rows (the join reads chroma only on even rows).
  let (u_half, v_half): (&[u16], &[u16]) = if chroma_row {
    (&u_scratch[..cw], &v_scratch[..cw])
  } else {
    (&[], &[])
  };
  yuv420p16_process_native::<BITS, HOST_NATIVE_BE>(
    plan,
    native_420_u16,
    resample_outputs,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    luma,
    hsv,
    rgb_scratch,
    rgb_scratch_u16,
    &y_scratch[..w],
    u_half,
    v_half,
    matrix,
    full_range,
    idx,
    w,
    h,
    // The RFC #238 chroma phases fold into the 4:2:0 chroma area weights: the
    // horizontal `chroma_h_phase` (`0.25` centered, `0.0` co-sited) AND the S6e
    // vertical `chroma_v_phase` (`1.0` for `Bottom`, `0.0` co-sited). At both
    // phases `0` this is byte-identical to the plain area grid, so the co-sited /
    // unspecified / horizontal-only output is untouched.
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

// ---- Chroma-siting-aware 4:2:0 upsample (#302) --------------------------
//
// The high-bit semi-planar siblings of the planar
// [`reserve_420_chroma_full_u16`](super::yuv420p) /
// [`upsample_420_chroma_center_h_u16`](super::yuv420p) staging. P010 / P012 /
// P016 chroma is INTERLEAVED in one MSB-aligned plane (`U V U V …`, active
// value in the high `BITS`), so the centered horizontal upsample
// de-interleaves, phase-0.5 upsamples each of U / V, and RE-INTERLEAVES into a
// full-width `U V …` plane the existing P-format 4:4:4 kernels consume — all
// fused in [`chroma_upsample_2to1_center_h_p0xx`](crate::row::scalar::chroma_upsample_2to1_center_h_p0xx),
// reusing their full SIMD backends so a centered decode stays bit-identical per
// tier. (`ChromaDerivedNcl` is the lone exception: P010/012/016 — like every
// format except `Yuv420p` — resolve it via the BT.709 matrix-tag fallback
// `Coefficients::for_matrix`, NOT `Yuv420p`'s #316 primaries-derived path; the
// default and centered P-format paths agree on that fallback, so they stay
// internally consistent. Primaries-derived P-format is a #302/#303 follow-up.)
// Only the centered sitings (`Center` / `Top` / `Bottom`) reach here; every
// co-sited / unspecified siting keeps the default fused `p0xx_to_*_row` decode,
// byte-identical to the pre-#302 output.

/// **Fallible preflight** for the centered-siting high-bit semi-planar 4:2:0
/// chroma scratch (#302) — the P-format twin of the planar
/// `reserve_420_chroma_full_u16`. Grows `chroma_full` to the checked
/// `2 * width` `u16` (a full-width INTERLEAVED `U V …` plane) so the later
/// infallible [`upsample_420_chroma_center_h_p0xx`] reuses an already-sized
/// buffer. Split out so it runs **before any output row is written** (luma
/// included) — the crate's preflight-ordering atomicity contract (cf. the #180
/// resample fix and the planar / NV centered paths): an allocator refusal must
/// leave the output frame *untouched*, never partially mutated.
///
/// Mirrors the siblings' recoverable grow: the `2 * width` length is
/// `checked_mul`'d (→ [`GeometryOverflow`]) and `try_reserve_exact` precedes the
/// resize (→ [`ResampleError::AllocationFailed`]). `height` feeds the error
/// payloads. Shares the planar/NV centered failpoint
/// (`FORCE_CHROMA_FULL_ALLOC_FAILURE`) so the atomicity regression reuses
/// `arm_chroma_full_alloc_failure`.
#[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
pub(crate) fn reserve_420_chroma_full_u16(
  chroma_full: &mut std::vec::Vec<u16>,
  width: usize,
  height: usize,
) -> Result<(), MixedSinkerError> {
  #[cfg(all(test, feature = "std", feature = "yuv-planar"))]
  if super::super::FORCE_CHROMA_FULL_ALLOC_FAILURE.with(|f| f.take()) {
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

/// Horizontally upsamples the half-width INTERLEAVED MSB-aligned `u16` chroma
/// row of a centered-siting high-bit semi-planar 4:2:0 source to a full-width
/// INTERLEAVED `U V …` plane in the **already-reserved** `chroma_full` (#302),
/// returning the `2 * width` `u16` slice the P-format 4:4:4 decode kernels
/// consume. The de-interleave + phase-0.5 blend + re-interleave are fused in
/// [`chroma_upsample_2to1_center_h_p0xx`](crate::row::scalar::chroma_upsample_2to1_center_h_p0xx)
/// (operating in the source's MSB-aligned wire convention + `big_endian`
/// order), so the output feeds straight to `p0xx_to_*_row_endian` with the same
/// `big_endian` flag — reusing their SIMD backends, bit-identical per tier.
/// `BITS` is threaded so the de-pack / re-pack shift matches the decode kernels'
/// `>> (16 - BITS)` (a `>> 0` / `<< 0` no-op at `BITS = 16`).
///
/// **Infallible**: the caller must have run [`reserve_420_chroma_full_u16`] up
/// front (every centered-siting output path does, before any output write), so
/// `chroma_full` is `>= 2 * width` here and `2 * width` cannot overflow. Only
/// the centered sitings reach here; the default left/unspecified path never
/// touches this scratch.
#[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
pub(crate) fn upsample_420_chroma_center_h_p0xx<'s, const BITS: u32>(
  chroma_full: &'s mut [u16],
  uv_half: &[u16],
  width: usize,
  big_endian: bool,
) -> &'s [u16] {
  debug_assert!(
    chroma_full.len() >= 2 * width,
    "chroma_full must be reserved via reserve_420_chroma_full_u16 first"
  );
  crate::row::scalar::chroma_upsample_2to1_center_h_p0xx::<BITS>(
    uv_half,
    &mut chroma_full[..2 * width],
    width,
    big_endian,
  );
  &chroma_full[..2 * width]
}

/// Stages the current half-width INTERLEAVED `u16` chroma row into the
/// bottom-sited vertical-phase lookback (RFC #238 S6e) — the P-format sibling of
/// the planar [`stage_420_chroma_prev_u16`](super::yuv420p::stage_420_chroma_prev_u16):
/// copies the whole interleaved `uv_half` (`width` `u16`, `U V …`, MSB-aligned
/// wire order preserved) into `chroma_prev` and tags it with the chroma row it
/// now holds (`idx / 2`), so a *later* even output row can validate
/// (`chroma_prev_row == Some(its_chroma_row - 1)`) and box-blend it. The layout
/// is INTERLEAVED (not the planar twin's separate U-then-V) because the P-format
/// reconstruction / decode kernels stay interleaved; the two never co-monomorphize
/// on one sink, so the shared `chroma_prev_u16` buffer carries whichever layout
/// its owning sink writes.
///
/// Called on EVERY accepted bottom-sited row — the colour path
/// ([`upsample_420_chroma_sited_p0xx`], after it has read the *previous* lookback)
/// and the luma-only path — so the lookback is always current. **Infallible**:
/// the caller must have run [`reserve_420_chroma_prev_u16`] up front.
#[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
#[inline]
pub(crate) fn stage_420_chroma_prev_p0xx(
  chroma_prev: &mut [u16],
  chroma_prev_row: &mut Option<usize>,
  uv_half: &[u16],
  idx: usize,
  width: usize,
) {
  debug_assert!(
    chroma_prev.len() >= width,
    "chroma_prev must be reserved via reserve_420_chroma_prev_u16 first"
  );
  chroma_prev[..width].copy_from_slice(&uv_half[..width]);
  *chroma_prev_row = Some(idx / 2);
}

/// Siting-aware full-width INTERLEAVED high-bit semi-planar 4:2:0 chroma
/// reconstruction (RFC #238 S6e) — the P-format sibling of the planar
/// [`upsample_420_chroma_sited_u16`](super::yuv420p::upsample_420_chroma_sited_u16),
/// folding the optional **bottom-sited vertical** phase into the horizontal
/// centered `p0xx` upsample and maintaining the one-row chroma lookback. Returns
/// the full-width interleaved `2 * width` `u16` slice the P-format 4:4:4 decode
/// kernels read, staged in the already-reserved `chroma_full` in the source's
/// MSB-aligned wire byte order.
///
/// `bottom_v` is [`chroma_420_bottom_sited_v`] (true only for
/// `ChromaLocation::Bottom`, which is also horizontally centered):
///
/// - On an **even** luma row (`idx & 1 == 0`) with `bottom_v`, each chroma sample
///   is the vertical box average of the previous chroma row (`chroma_prev`) and
///   the current row, then horizontally centered — one fused pass via
///   [`chroma_upsample_420_bottom_even_h_p0xx`](crate::row::scalar::chroma_upsample_420_bottom_even_h_p0xx)
///   — **but only when `chroma_prev` provably holds the wanted predecessor**
///   chroma row `idx/2 - 1` (`chroma_prev_row == Some(idx/2 - 1)`). Otherwise —
///   the top edge (`idx == 0`), a fresh frame, or a direct `process` caller that
///   replayed / skipped / reordered rows or attached colour late — it falls back
///   to the plain centered upsample of the *current* chroma row (the top-edge
///   clamp), so the blend NEVER mixes stale chroma from an older pair or frame.
/// - Otherwise (odd luma row, or any non-bottom centered siting) the current
///   chroma row is horizontally centered with the plain
///   [`upsample_420_chroma_center_h_p0xx`] — bit-identical to the horizontal-only
///   (S6b) path, so `Center` / `Top` output is untouched.
///
/// When `stage` is set, on every bottom-sited row the current interleaved chroma
/// row is copied into `chroma_prev` (tagged `idx / 2`). The direct decode passes
/// `stage = true` (its post-reconstruction work is infallible); the resample
/// reconstruction arms pass `stage = false` and defer [`stage_420_chroma_prev_p0xx`]
/// until AFTER their fallible resample commit accepts the row, so a rejected row
/// leaves the lookback pointing at the predecessor and a retry still box-blends it
/// (state-atomic, #180). The caller must have run [`reserve_420_chroma_full_u16`]
/// (always) and, when `bottom_v`, [`reserve_420_chroma_prev_u16`] up front.
#[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn upsample_420_chroma_sited_p0xx<'s, const BITS: u32>(
  chroma_full: &'s mut [u16],
  chroma_prev: &mut [u16],
  chroma_prev_row: &mut Option<usize>,
  uv_half: &[u16],
  idx: usize,
  bottom_v: bool,
  stage: bool,
  width: usize,
  big_endian: bool,
) -> &'s [u16] {
  debug_assert!(
    chroma_full.len() >= 2 * width,
    "chroma_full must be reserved via reserve_420_chroma_full_u16 first"
  );
  let chroma_row = idx / 2;
  // The bottom-sited EVEN row box-blends only when the lookback PROVABLY holds
  // the wanted predecessor `chroma_row - 1`; otherwise it clamps to the current
  // row (never stale). Every other case (non-bottom siting, the bottom-sited odd
  // row, or an unvalidated even row) is the plain centered upsample.
  let do_vblend =
    bottom_v && idx & 1 == 0 && chroma_row > 0 && *chroma_prev_row == Some(chroma_row - 1);

  let result = if do_vblend {
    debug_assert!(
      chroma_prev.len() >= width,
      "chroma_prev must be reserved via reserve_420_chroma_prev_u16 first"
    );
    crate::row::scalar::chroma_upsample_420_bottom_even_h_p0xx::<BITS>(
      &chroma_prev[..width],
      uv_half,
      &mut chroma_full[..2 * width],
      width,
      big_endian,
    );
    &chroma_full[..2 * width]
  } else {
    upsample_420_chroma_center_h_p0xx::<BITS>(chroma_full, uv_half, width, big_endian)
  };

  // Refresh the lookback with the current interleaved chroma row + its validity
  // tag (after the read above). Gated on `stage`: the direct decode refreshes it
  // here (its later work is infallible), while the resample reconstruction arms
  // pass `stage = false` and defer the refresh to AFTER their fallible commit
  // accepts the row. Only the bottom-sited path maintains it.
  if bottom_v && stage {
    stage_420_chroma_prev_p0xx(chroma_prev, chroma_prev_row, uv_half, idx, width);
  }
  result
}

// ---- P010 impl ---------------------------------------------------------

impl<'a, R, const BE: bool> MixedSinker<'a, P010<BE>, R> {
  /// Attaches a packed **`u16`** RGB output buffer. Mirrors
  /// [`MixedSinker<Yuv420p10>::with_rgb_u16`] — compile‑time gated to
  /// sinkers whose source format populates native‑depth RGB.
  ///
  /// Length is measured in `u16` **elements** (not bytes): minimum
  /// `width x height x 3`. Output is **low‑bit‑packed** (10‑bit
  /// values in the low 10 of each `u16`, upper 6 zero) — matches
  /// FFmpeg `yuv420p10le` convention. This is **not** P010 packing
  /// (which puts the 10 bits in the high 10); callers feeding a P010
  /// consumer must shift the output left by 6.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgb_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_rgb_u16(buf)?;
    Ok(self)
  }

  /// In-place variant of [`with_rgb_u16`](Self::with_rgb_u16). The
  /// required length is measured in `u16` **elements**, not bytes.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgb_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
    let expected_elements = self.frame_elems(3)?;
    if buf.len() < expected_elements {
      return Err(MixedSinkerError::InsufficientRgbU16Buffer(
        InsufficientBuffer::new(expected_elements, buf.len()),
      ));
    }
    self.rgb_u16 = Some(buf);
    Ok(self)
  }

  /// Attaches a packed **8‑bit** RGBA output buffer. The 10‑bit P010
  /// source (semi‑planar, high‑bit‑packed) is converted to 8‑bit RGBA
  /// via the `BITS = 10` Q15 kernel family; alpha = `0xFF` (P010 has
  /// no alpha plane).
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

  /// Attaches a packed **`u16`** RGBA output buffer. Output is
  /// **low‑bit‑packed** 10‑bit values (`yuv420p10le` convention) — not
  /// P010 high‑bit packing. Length is measured in `u16` **elements**
  /// (`width x height x 4`). Alpha element is `(1 << 10) - 1`.
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

impl<R, const BE: bool> P010Sink<BE> for MixedSinker<'_, P010<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, P010<BE>, R> {
  type Input<'r> = P010Row<'r>;
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

  fn process(&mut self, row: P010Row<'_>) -> Result<(), Self::Error> {
    // P010 stores 10‑bit samples high‑bit‑packed; bit depth is fixed
    // by the format. Used for the u16 RGBA expand path's alpha pad.
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
    // Semi-planar UV: `width` u16 elements total (`width / 2` pairs).
    if row.uv_half().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::UvHalf10,
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

    // Chroma siting (#302): drives the identity-plan horizontal chroma phase.
    // `Copy`, so read it out before the field split-borrow below. Gated like
    // its only consumer (`chroma_420_center_sited_h` + the 4:4:4 P-format
    // kernels need `yuv-planar`); a semi-planar-only build keeps the default
    // decode.
    #[cfg(feature = "yuv-planar")]
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
      luma_scratch_u16,
      rgb_stream,
      rgb_stream_u16,
      luma_stream_u16,
      rgb_filter_stream,
      rgb_filter_stream_u16,
      luma_filter_stream_u16,
      resample_outputs,
      plan,
      #[cfg(feature = "yuv-planar")]
      native,
      #[cfg(feature = "yuv-planar")]
      native_420_u16,
      #[cfg(feature = "yuv-planar")]
      p0xx_y_half,
      #[cfg(feature = "yuv-planar")]
      p0xx_u_half,
      #[cfg(feature = "yuv-planar")]
      p0xx_v_half,
      #[cfg(feature = "yuv-planar")]
      frozen_native_route,
      // Full-width interleaved chroma staging for the centered-siting (#302)
      // identity decode.
      #[cfg(feature = "yuv-planar")]
      chroma_full_u16,
      // RFC #238 S6b: the 4:2:0 horizontal chroma siting phase frozen on the
      // first output row (the `u16` semi-planar twin of the planar Yuv420p
      // `frozen_chroma_centered`).
      #[cfg(feature = "yuv-planar")]
      frozen_chroma_centered,
      // RFC #238 S6e: the vertical (`Bottom`) chroma phase frozen alongside it,
      // plus the one-row interleaved chroma lookback + its validity tag, so a
      // `Bottom` even output row box-blends the previous chroma row.
      #[cfg(feature = "yuv-planar")]
      frozen_chroma_bottom_v,
      #[cfg(feature = "yuv-planar")]
      chroma_prev_u16,
      #[cfg(feature = "yuv-planar")]
      chroma_prev_row,
      ..
    } = self;

    // Non-identity plan. When the native tier is enabled (and the planar
    // join it reuses is compiled in), bin the native Y / U / V planes at
    // output resolution and convert once per output row, de-interleaving +
    // de-packing the P010 chroma + Y into wrapper-owned logical scratch
    // first. Otherwise (or under `with_native(false)`) feed the shared
    // high-bit 4:2:2 triple-resample tail (u8 color, independent native-u16
    // color, native Y). P010 is semi-planar 4:2:0: the interleaved
    // half-width UV plane is de-interleaved + horizontally upsampled
    // in-register by the `p010_to_rgb*` row kernels, and 4:2:0's vertical
    // chroma sharing is resolved by the walker (each luma row gets its
    // shared `uv_half`), so the per-row chroma contract matches 4:2:2's and
    // the same tail binds. The Y de-pack closure shifts each
    // high-bit-packed sample right by `16 - BITS` so the binned native Y is
    // the logical value (matching the low-packed planar Yuv420p10 luma
    // contract): `luma = binned_Y >> (BITS - 8)`. P010 exposes no
    // `luma_u16`, so it is `&mut None`.
    if let Some(plan) = plan.as_ref() {
      let matrix = row.matrix();
      let full_range = row.full_range();
      let (y, uv_half) = (row.y(), row.uv_half());
      // RFC #238 S6b/S6e — 4:2:0 chroma siting for the high-bit semi-planar
      // P010, mirroring the planar Yuv420p10 twin (S6a/S6d) on the semi-planar
      // de-interleave (S3b/S4-C). The centered group
      // (`chroma_420_center_sited_h`) samples chroma at `+0.5` luma = `+0.25`
      // chroma-sample horizontally; S6e additionally folds the VERTICAL `v = 1`
      // triangle for `Bottom` ([`chroma_420_bottom_sited_v`], a strict sub-case
      // of `center_sited`), while `Center` / `Top` keep `v_phase = 0` (co-sited
      // vertical, byte-identical to S6b). The co-sited / unspecified group is
      // phase 0 (byte-identical). The native fast tier folds both phases into the
      // `area_chroma_420` chroma weights; the filter and row-stage tiers
      // reconstruct full-width interleaved `u16` chroma (de-interleave + phase-0.5
      // upsample into `chroma_full_u16`, plus the `Bottom` even-row vertical box
      // blend against the `chroma_prev_u16` lookback) and decode 4:4:4 via the
      // `p410` full-chroma kernels — the co-sited arms keep the fused `p010`
      // half-chroma decode.
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      let center_sited = chroma_420_center_sited_h(chroma_location);
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      let bottom_v = chroma_420_bottom_sited_v(chroma_location);
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      let chroma_h_phase = if center_sited {
        YUV422P_CENTERED_H_PHASE
      } else {
        0.0
      };
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      let chroma_v_phase = if bottom_v { 1.0 } else { 0.0 };
      // Only the colour tiers reconstruct full-width chroma for the centered
      // decode; a luma-only centered row bins native Y unchanged (siting is a
      // chroma-only property).
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      let want_color =
        rgb.is_some() || rgba.is_some() || hsv.is_some() || rgb_u16.is_some() || rgba_u16.is_some();
      // Whether this call carries any output — the EXACT set the tier preflight
      // (`yuv420p16_native_preflight`'s `need_luma || need_color`) tests. The
      // route (and the siting phase) freezes only on an output-bearing row a tier
      // ACCEPTS; a no-output call consumes no stream state, so it must not freeze.
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      let need_output = luma.is_some()
        || rgb.is_some()
        || rgba.is_some()
        || rgb_u16.is_some()
        || rgba_u16.is_some()
        || hsv.is_some();
      // Freeze the effective 4:2:0 chroma siting on the first output-bearing row
      // (mirrors the `frozen_native_route` freeze below). This CHECK is at the
      // always-compiled choke point every tier passes through; the matching SET
      // rides each tier's accept path (never before dispatch, so a rejected row
      // leaves it unset for a corrected retry). A later row observing a different
      // phase would bin a mixture of co-sited and centered chroma, so it is
      // rejected here before any reconstruction. `Center` and `Bottom` share the
      // horizontal center, so the vertical phase is frozen too (RFC #238 S6e) — a
      // mid-frame Center ⇆ Bottom flip is caught only by comparing
      // `frozen_chroma_bottom_v`.
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      if need_output
        && let Some(frozen) = *frozen_chroma_centered
        && (frozen != center_sited || *frozen_chroma_bottom_v != Some(bottom_v))
      {
        return Err(MixedSinkerError::ChromaSitingChanged(
          ChromaSitingChanged::new(idx),
        ));
      }

      // FILTER FIRST. A `Filter` plan routes to the signed-coefficient
      // filter resampler (the same `p010_to_rgb*` convert closures the area
      // path uses, then resampled in RGB space) — and there is NO native
      // fast tier for the filter path, so it must branch BEFORE the
      // native-route machinery (`frozen_native_route` / `*native`) below,
      // which is area-only. The filter tail clamps a signed-kernel
      // overshoot to the native max for this sub-16-bit source (both colour
      // and native-Y luma), matching the in-range area path.
      if plan.kind().is_filter() {
        // Centered filter reconstructs full-width chroma and decodes 4:4:4, but
        // ONLY after the resample preflight (frozen-output + sequence), so an
        // out-of-sequence / rejected row is caught before the chroma reservation
        // (#180). `packed_yuv422_triple_filter_resample` re-runs the idempotent
        // preflight. Co-sited keeps the fused `p010` half-chroma decode.
        #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
        {
          // Reject a multi-kernel (BICUBLIN) filter plan BEFORE the centered
          // reserve below, mirroring the delegate's own first act (idempotent).
          plan.ensure_single_kernel_filter()?;
          if center_sited && want_color {
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
            // RFC #238 S6e: reserve the bottom-sited vertical-phase lookback
            // AFTER the preflight, BEFORE the commit (#180); only for v=1. The
            // sited call reads it but defers staging until the commit accepts.
            if bottom_v {
              reserve_420_chroma_prev_u16(chroma_prev_u16, w, h)?;
            }
            // `stage = false`: DEFER the lookback advance until AFTER the fallible
            // filter commit accepts the row (below), so a rejected row leaves the
            // predecessor for a clean retry (#180 state-atomicity).
            let uv_full = upsample_420_chroma_sited_p0xx::<BITS>(
              chroma_full_u16,
              chroma_prev_u16,
              chroma_prev_row,
              uv_half,
              idx,
              bottom_v,
              false,
              w,
              BE,
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
              |scratch| {
                for (dst, &s) in scratch[..w].iter_mut().zip(y.iter()) {
                  let logical = if BE { u16::from_be(s) } else { u16::from_le(s) };
                  *dst = logical >> (16 - BITS);
                }
              },
              |scratch| {
                p410_to_rgb_row_endian(y, uv_full, scratch, w, matrix, full_range, use_simd, BE)
              },
              |scratch| {
                p410_to_rgb_u16_row_endian(y, uv_full, scratch, w, matrix, full_range, use_simd, BE)
              },
            );
            // RFC #238 S6e: advance the bottom lookback only AFTER the filter
            // resample accepts the row (`r.is_ok()`) — the sited reconstruction
            // read it above but did not stage, so a rejected row keeps the
            // predecessor for a clean retry.
            if r.is_ok() && bottom_v {
              stage_420_chroma_prev_p0xx(chroma_prev_u16, chroma_prev_row, uv_half, idx, w);
            }
            if r.is_ok() && need_output && frozen_chroma_centered.is_none() {
              *frozen_chroma_centered = Some(center_sited);
              *frozen_chroma_bottom_v = Some(bottom_v);
            }
            return r;
          }
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
          |scratch| {
            for (dst, &s) in scratch[..w].iter_mut().zip(y.iter()) {
              let logical = if BE { u16::from_be(s) } else { u16::from_le(s) };
              *dst = logical >> (16 - BITS);
            }
          },
          |scratch| {
            p010_to_rgb_row_endian(y, uv_half, scratch, w, matrix, full_range, use_simd, BE)
          },
          |scratch| {
            p010_to_rgb_u16_row_endian(y, uv_half, scratch, w, matrix, full_range, use_simd, BE)
          },
        );
        #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
        if r.is_ok() && need_output && frozen_chroma_centered.is_none() {
          *frozen_chroma_centered = Some(center_sited);
          *frozen_chroma_bottom_v = Some(bottom_v);
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
      // a true no-op, fully route-invisible regardless of row index, so it
      // can never spuriously trip `NativeRouteChanged` after the route is
      // frozen. A preflight-rejected (out-of-sequence / frozen)
      // output-bearing call returns Err before the SET, so it leaves
      // `frozen_native_route` untouched and a later same-or-other-route
      // retry is not falsely rejected. (Issue #186 tracks the same gap in
      // the other native families.)
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
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
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
        // Dispatch first; freeze the route to native ONLY after the call
        // returns Ok on an output-bearing row. A no-output call returns
        // Ok(()) with `need_output` false (no freeze); an out-of-sequence /
        // frozen row returns Err via `?` (no freeze) — so only an accepted
        // output-bearing row commits the route.
        //
        // RFC #238 S6b point-of-use siting invalidation, mirroring the planar
        // Yuv420p native arm: a reused sink's cached join is only `reset` between
        // frames, so a frame whose `chroma_location` moved to a different
        // horizontal phase must REBUILD it (`area_chroma_420` folds the phase into
        // the chroma weights). Drop the stale-phase join ONLY on the in-sequence
        // first row of a fresh frame (`idx == 0`, `next_y() == 0`) so a mid-frame
        // / out-of-sequence row rejects against the INTACT join and a corrected
        // retry rebuilds cleanly; a luma-only join carries no chroma phase and is
        // never dropped. Move it OUT (the delegate builds the replacement into the
        // field, untouched until every pre-feed allocation succeeds) and restore
        // the intact prior-phase join on a rejected rebuild so the row mutates no
        // join state.
        let stale_native = idx == 0
          && native_420_u16.as_ref().is_some_and(|join| {
            (join.chroma_phase_centered() == Some(!center_sited)
              || join.chroma_bottom() == Some(!bottom_v))
              && join.next_y() == 0
          });
        let prev_native = if stale_native {
          native_420_u16.take()
        } else {
          None
        };
        let native_result = p0xx_process_native::<BITS, BE>(
          plan,
          native_420_u16,
          p0xx_y_half,
          p0xx_u_half,
          p0xx_v_half,
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
          uv_half,
          chroma_h_phase,
          chroma_v_phase,
          matrix,
          full_range,
          idx,
          w,
          h,
          use_simd,
        );
        // Restore the taken stale-phase join if the delegate's rebuild was
        // rejected at any pre-feed step: it leaves the field `None` on such a
        // failure, so restoring the intact prior-phase join leaves the rejected
        // row mutating no join state. A non-stale row took nothing.
        if stale_native && native_result.is_err() {
          *native_420_u16 = prev_native;
        }
        native_result?;
        #[cfg(feature = "yuv-semi-planar")]
        if frozen_native_route.is_none() && need_output {
          *frozen_native_route = Some(true);
        }
        #[cfg(feature = "yuv-semi-planar")]
        if frozen_chroma_centered.is_none() && need_output {
          *frozen_chroma_centered = Some(center_sited);
          *frozen_chroma_bottom_v = Some(bottom_v);
        }
        return Ok(());
      }
      // Row-stage area tail. Same CHECK-before / SET-after split. Centered colour
      // reconstructs full-width interleaved `u16` chroma (de-interleave +
      // phase-0.5 upsample, plus the `Bottom` even-row vertical box blend) and
      // decodes 4:4:4 — but ONLY after the resample preflight (frozen-output +
      // sequence), so an out-of-sequence / rejected row is caught before the
      // chroma reservation (#180). A luma-only centered row stays on the co-sited
      // arm (which only bins luma).
      // `packed_yuv422_triple_resample` re-runs the idempotent preflight.
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      if center_sited && want_color {
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
        // RFC #238 S6e: reserve the bottom-sited vertical-phase lookback AFTER
        // the preflight, BEFORE the commit (#180); only for v=1. The sited call
        // reads it but defers staging until the commit accepts.
        if bottom_v {
          reserve_420_chroma_prev_u16(chroma_prev_u16, w, h)?;
        }
        // `stage = false`: DEFER the lookback advance until AFTER the fallible
        // row-stage commit accepts the row (below), so a rejected row leaves the
        // predecessor for a clean retry (#180 state-atomicity).
        let uv_full = upsample_420_chroma_sited_p0xx::<BITS>(
          chroma_full_u16,
          chroma_prev_u16,
          chroma_prev_row,
          uv_half,
          idx,
          bottom_v,
          false,
          w,
          BE,
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
          |scratch| {
            for (dst, &s) in scratch[..w].iter_mut().zip(y.iter()) {
              let logical = if BE { u16::from_be(s) } else { u16::from_le(s) };
              *dst = logical >> (16 - BITS);
            }
          },
          |scratch| {
            p410_to_rgb_row_endian(y, uv_full, scratch, w, matrix, full_range, use_simd, BE)
          },
          |scratch| {
            p410_to_rgb_u16_row_endian(y, uv_full, scratch, w, matrix, full_range, use_simd, BE)
          },
        )?;
        // RFC #238 S6e: advance the bottom lookback only AFTER the row-stage
        // resample accepts the row (the `?` above already returned any reject);
        // the sited reconstruction read it but did not stage, so a rejected row
        // keeps the predecessor for a clean retry.
        if bottom_v {
          stage_420_chroma_prev_p0xx(chroma_prev_u16, chroma_prev_row, uv_half, idx, w);
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
      // Row-stage tail. Same CHECK-before / SET-after split: dispatch, then
      // freeze the route to row-stage only when the call accepts an
      // output-bearing row (a no-output call returns Ok with `need_output`
      // false; an out-of-sequence / frozen row returns Err via `?`).
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
        |scratch| {
          for (dst, &s) in scratch[..w].iter_mut().zip(y.iter()) {
            let logical = if BE { u16::from_be(s) } else { u16::from_le(s) };
            *dst = logical >> (16 - BITS);
          }
        },
        |scratch| p010_to_rgb_row_endian(y, uv_half, scratch, w, matrix, full_range, use_simd, BE),
        |scratch| {
          p010_to_rgb_u16_row_endian(y, uv_half, scratch, w, matrix, full_range, use_simd, BE)
        },
      )?;
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      if frozen_native_route.is_none() && need_output {
        *frozen_native_route = Some(false);
      }
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      if frozen_chroma_centered.is_none() && need_output {
        *frozen_chroma_centered = Some(center_sited);
        *frozen_chroma_bottom_v = Some(bottom_v);
      }
      return Ok(());
    }

    // Resolve the output set up front so the no-output guard short-circuits
    // before ANY per-row offset arithmetic and the atomicity preflight runs
    // before any output row is written.
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();
    let want_rgb_u16 = rgb_u16.is_some();
    let want_rgba_u16 = rgba_u16.is_some();
    let want_color = want_rgb || want_rgba || want_hsv || want_rgb_u16 || want_rgba_u16;
    // Repo-wide no-output invariant (RFC #238 S6e, mirrors the planar Yuv420p /
    // NV S4-C direct twins): a `process` call carrying NO output — no colour, no
    // luma — must run NOTHING: no per-row offset arithmetic, no allocation, no
    // state mutation (the bottom-sited vertical lookback included). Returning
    // HERE, before the `idx * w` offsets below, keeps the invariant overflow-safe
    // (a no-output call never ran an attach-time `w x h` validation, so `idx * w`
    // could overflow on a 32-bit target) AND means such a row never reserves
    // `chroma_prev_u16` nor primes the lookback (so it can never make a later
    // colour even row box-blend through an invisible, never-output row). P010
    // exposes no `luma_u16`.
    let need_output = want_color || luma.is_some();
    if !need_output {
      return Ok(());
    }

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // Chroma siting (#302): the centered horizontal sitings reconstruct chroma
    // at the phase-0.5 position; the default / co-sited path keeps the
    // byte-identical fused decode (the fused P-format 4:2:0 kernels
    // de-interleave + upsample chroma in-register, exactly as before).
    #[cfg(feature = "yuv-planar")]
    let center_sited = chroma_420_center_sited_h(chroma_location);
    // RFC #238 S6e: `Bottom` (a strict sub-case of `center_sited`) additionally
    // box-blends the even output row's chroma with the previous chroma row via
    // the `chroma_prev_u16` lookback maintained below; `Center` / `Top` keep the
    // vertical-replicate (co-sited) decode, byte-identical to S6b.
    #[cfg(feature = "yuv-planar")]
    let bottom_v = chroma_420_bottom_sited_v(chroma_location);

    // Atomicity preflight (#302 / #308, cf. the crate's #180 resample fix and
    // the planar_8bit / semi_planar_8bit 8-bit siblings): reserve EVERY fallible
    // row scratch this identity row can touch BEFORE any output row is written
    // (the luma plane below, then the u16 / u8 RGB / RGBA / HSV fan-out), so an
    // allocator refusal returns a typed `AllocationFailed` leaving the output
    // frame untouched rather than partially mutated. Two scratches can grow:
    //  1. the centered-siting full-width interleaved `u16` chroma
    //     (`chroma_full_u16`), needed by ANY colour output (u8 OR u16 RGB / RGBA
    //     / HSV); and
    //  2. the u8 RGB row buffer, reached exactly when a colour decode needs an
    //     RGB row but no caller RGB buffer is borrowable — `want_hsv &&
    //     want_rgba && !want_rgb` (`rgb_row_buf_or_scratch`'s own scratch arm).
    // The later `upsample_420_chroma_sited_p0xx` / `rgb_row_buf_or_scratch`
    // calls reuse the already-sized buffers, so the default path is
    // byte-identical; only the failure-path ordering changes. The u16 RGB / RGBA
    // outputs write straight into their caller buffers (the rgb_u16 plane itself
    // stages the rgba_u16 expand) and never grow a scratch of their own. A
    // luma-only row consumes no centered chroma, so it neither reserves nor
    // upsamples it (the reserve makes the later upsample infallible).
    #[cfg(feature = "yuv-planar")]
    let need_centered_chroma = center_sited && want_color;
    #[cfg(feature = "yuv-planar")]
    if need_centered_chroma {
      reserve_420_chroma_full_u16(chroma_full_u16, w, h)?;
    }
    // Bottom-sited vertical-phase one-row chroma lookback (RFC #238 S6e): reserve
    // it on EVERY bottom-sited OUTPUT row — colour OR luma-only — because the
    // lookback is maintained so a LATER colour row can box-blend it (the
    // luma-only staging runs below, before any luma write). A no-output row
    // returned early above, so it never reaches here and never primes the
    // lookback. Reserved BEFORE any output write (the #180 preflight ordering).
    #[cfg(feature = "yuv-planar")]
    if bottom_v {
      reserve_420_chroma_prev_u16(chroma_prev_u16, w, h)?;
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

    // Bottom-sited LUMA-ONLY row (RFC #238 S6e): the colour upsample below —
    // which normally refreshes the vertical lookback — won't run, so stage the
    // current interleaved chroma row HERE, after its reservation above and BEFORE
    // any luma write, so a later colour row in the same frame can box-blend it (a
    // luma-only-then-colour in-order sequence reconstructs the same Bottom
    // vertical phase as the all-output walk). A colour row instead stages inside
    // `upsample_420_chroma_sited_p0xx` after reading the previous lookback, so
    // this is skipped for it; a no-output row returned early above. The validity
    // tag (`chroma_prev_row`) still guards out-of-sequence / cross-frame reads.
    #[cfg(feature = "yuv-planar")]
    if bottom_v && !want_color {
      stage_420_chroma_prev_p0xx(chroma_prev_u16, chroma_prev_row, row.uv_half(), idx, w);
    }

    // Centered full-width INTERLEAVED chroma (phase-0.5), reconstructed ONCE per
    // row from the wire-format half-width interleaved UV and reused by every
    // colour decode (u16 and u8). Infallible — the scratch was reserved above.
    // The default left/unspecified siting leaves it `None`, so the fused
    // P-format 4:2:0 kernels de-interleave + upsample chroma in-register and the
    // output stays byte-identical. `Center` / `Top` take the plain horizontal
    // centered phase-0.5 fold; `Bottom` (RFC #238 S6e) additionally box-blends
    // the even output row with the previous chroma row (`stage = true` — the
    // direct decode's post-reconstruction work is infallible).
    #[cfg(feature = "yuv-planar")]
    let centered: Option<&[u16]> = if need_centered_chroma {
      Some(upsample_420_chroma_sited_p0xx::<BITS>(
        chroma_full_u16,
        chroma_prev_u16,
        chroma_prev_row,
        row.uv_half(),
        idx,
        bottom_v,
        true,
        w,
        BE,
      ))
    } else {
      None
    };
    #[cfg(not(feature = "yuv-planar"))]
    let centered: Option<&[u16]> = None;

    // Luma: P010 samples are high-bit-packed (`value << 6`). Taking the
    // high byte via `>> 8` gives the top 8 bits of the 10-bit value —
    // functionally equivalent to `(value >> 2)` for the yuv420p10 path.
    // Routed through the native-Y kernel (bit-identical to the former
    // inline `>> 8` loop, including the BE-wire normalization).
    if let Some(luma) = luma.as_deref_mut() {
      p010_to_luma_row_endian(row.y(), &mut luma[one_plane_start..one_plane_end], w, BE);
    }

    // ===== u16 RGB / RGBA path (Strategy A) — see Yuv420p10 for rationale.
    // u16 outputs are low-bit-packed (yuv420p10le convention), not
    // P010's high-bit packing.
    if want_rgba_u16 && !want_rgb_u16 {
      let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row =
        rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
      if let Some(uv_full) = centered {
        p410_to_rgba_u16_row_endian(
          row.y(),
          uv_full,
          rgba_u16_row,
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
          BE,
        );
      } else {
        p010_to_rgba_u16_row_endian(
          row.y(),
          row.uv_half(),
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
      if let Some(uv_full) = centered {
        p410_to_rgb_u16_row_endian(
          row.y(),
          uv_full,
          rgb_u16_row,
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
          BE,
        );
      } else {
        p010_to_rgb_u16_row_endian(
          row.y(),
          row.uv_half(),
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
    // HSV-without-RGB-or-RGBA goes through the direct `p010_to_hsv_row`
    // kernel (no source-width RGB scratch). When RGB or RGBA is *also*
    // attached the RGB kernel runs anyway, so HSV derives off that buffer
    // for free (the cheap path) and `need_rgb_kernel` keeps it alive.
    // (Resample row-stage HSV-only is a #263 follow-up — see the row-
    // stage block above; HSV stays correct via the convert-once path.)
    let want_hsv_direct = want_hsv && !want_rgb && !want_rgba;
    let need_rgb_kernel = want_rgb || (want_hsv && want_rgba);

    if want_hsv_direct {
      let hsv = hsv.as_mut().expect("want_hsv_direct implies hsv attached");
      let (h, s, v) = hsv.hsv();
      if let Some(uv_full) = centered {
        p410_to_hsv_row_endian(
          row.y(),
          uv_full,
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
        p010_to_hsv_row_endian(
          row.y(),
          row.uv_half(),
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
      if let Some(uv_full) = centered {
        p410_to_rgba_row_endian(
          row.y(),
          uv_full,
          rgba_row,
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
          BE,
        );
      } else {
        p010_to_rgba_row_endian(
          row.y(),
          row.uv_half(),
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

    if let Some(uv_full) = centered {
      p410_to_rgb_row_endian(
        row.y(),
        uv_full,
        rgb_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
        BE,
      );
    } else {
      p010_to_rgb_row_endian(
        row.y(),
        row.uv_half(),
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

// ---- P012 impl ---------------------------------------------------------

impl<'a, R, const BE: bool> MixedSinker<'a, P012<BE>, R> {
  /// Attaches a packed **`u16`** RGB output buffer. Produces 12‑bit
  /// output in **low‑bit‑packed** `yuv420p12le` convention (values in
  /// `[0, 4095]` in the low 12 of each `u16`, upper 4 zero) —
  /// **not** P012's high‑bit packing. Callers feeding a P012 consumer
  /// must shift the output left by 4.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgb_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_rgb_u16(buf)?;
    Ok(self)
  }

  /// In-place variant of [`with_rgb_u16`](Self::with_rgb_u16).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgb_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
    let expected_elements = self.frame_elems(3)?;
    if buf.len() < expected_elements {
      return Err(MixedSinkerError::InsufficientRgbU16Buffer(
        InsufficientBuffer::new(expected_elements, buf.len()),
      ));
    }
    self.rgb_u16 = Some(buf);
    Ok(self)
  }

  /// Attaches a packed **8‑bit** RGBA output buffer. The 12‑bit P012
  /// source (semi‑planar, high‑bit‑packed) is converted to 8‑bit RGBA
  /// via the `BITS = 12` Q15 kernel family; alpha = `0xFF`.
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

  /// Attaches a packed **`u16`** RGBA output buffer. Output is
  /// **low‑bit‑packed** 12‑bit values (`yuv420p12le` convention) —
  /// not P012 high‑bit packing. Length is measured in `u16`
  /// **elements** (`width x height x 4`). Alpha element is
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

impl<R, const BE: bool> P012Sink<BE> for MixedSinker<'_, P012<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, P012<BE>, R> {
  type Input<'r> = P012Row<'r>;
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

  fn process(&mut self, row: P012Row<'_>) -> Result<(), Self::Error> {
    // P012 stores 12‑bit samples high‑bit‑packed; bit depth is fixed
    // by the format. Used for the u16 RGBA expand path's alpha pad.
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
    if row.uv_half().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::UvHalf12,
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

    // Chroma siting (#302): drives the identity-plan horizontal chroma phase.
    // `Copy`, so read it out before the field split-borrow below. Gated like
    // its only consumer (`chroma_420_center_sited_h` + the 4:4:4 P-format
    // kernels need `yuv-planar`); a semi-planar-only build keeps the default
    // decode.
    #[cfg(feature = "yuv-planar")]
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
      luma_scratch_u16,
      rgb_stream,
      rgb_stream_u16,
      luma_stream_u16,
      rgb_filter_stream,
      rgb_filter_stream_u16,
      luma_filter_stream_u16,
      resample_outputs,
      plan,
      #[cfg(feature = "yuv-planar")]
      native,
      #[cfg(feature = "yuv-planar")]
      native_420_u16,
      #[cfg(feature = "yuv-planar")]
      p0xx_y_half,
      #[cfg(feature = "yuv-planar")]
      p0xx_u_half,
      #[cfg(feature = "yuv-planar")]
      p0xx_v_half,
      #[cfg(feature = "yuv-planar")]
      frozen_native_route,
      // Full-width interleaved chroma staging for the centered-siting (#302)
      // identity decode.
      #[cfg(feature = "yuv-planar")]
      chroma_full_u16,
      // RFC #238 S6b: the 4:2:0 horizontal chroma siting phase frozen on the
      // first output row (the `u16` semi-planar twin of the planar Yuv420p
      // `frozen_chroma_centered`).
      #[cfg(feature = "yuv-planar")]
      frozen_chroma_centered,
      // RFC #238 S6e: the vertical (`Bottom`) chroma phase frozen alongside it,
      // plus the one-row interleaved chroma lookback + its validity tag, so a
      // `Bottom` even output row box-blends the previous chroma row.
      #[cfg(feature = "yuv-planar")]
      frozen_chroma_bottom_v,
      #[cfg(feature = "yuv-planar")]
      chroma_prev_u16,
      #[cfg(feature = "yuv-planar")]
      chroma_prev_row,
      ..
    } = self;

    // Non-identity plan. When the native tier is enabled, bin the native
    // planes at output resolution and convert once per output row (de-pack
    // into wrapper scratch first); otherwise feed the shared high-bit 4:2:2
    // triple-resample tail. See the P010 impl for the full rationale —
    // P012 is identical bar the 12-bit kernel family and the `16 - BITS` Y
    // de-pack shift.
    if let Some(plan) = plan.as_ref() {
      let matrix = row.matrix();
      let full_range = row.full_range();
      let (y, uv_half) = (row.y(), row.uv_half());
      // RFC #238 S6b/S6e — 4:2:0 chroma siting for the high-bit semi-planar
      // P012, mirroring the planar Yuv420p12 twin (S6a/S6d) on the semi-planar
      // de-interleave (S3b/S4-C). The centered group
      // (`chroma_420_center_sited_h`) samples chroma at `+0.5` luma = `+0.25`
      // chroma-sample horizontally; S6e additionally folds the VERTICAL `v = 1`
      // triangle for `Bottom` ([`chroma_420_bottom_sited_v`], a strict sub-case
      // of `center_sited`), while `Center` / `Top` keep `v_phase = 0` (co-sited
      // vertical, byte-identical to S6b). The co-sited / unspecified group is
      // phase 0 (byte-identical). The native fast tier folds both phases into the
      // `area_chroma_420` chroma weights; the filter and row-stage tiers
      // reconstruct full-width interleaved `u16` chroma (de-interleave + phase-0.5
      // upsample into `chroma_full_u16`, plus the `Bottom` even-row vertical box
      // blend against the `chroma_prev_u16` lookback) and decode 4:4:4 via the
      // `p412` full-chroma kernels — the co-sited arms keep the fused `p012`
      // half-chroma decode.
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      let center_sited = chroma_420_center_sited_h(chroma_location);
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      let bottom_v = chroma_420_bottom_sited_v(chroma_location);
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      let chroma_h_phase = if center_sited {
        YUV422P_CENTERED_H_PHASE
      } else {
        0.0
      };
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      let chroma_v_phase = if bottom_v { 1.0 } else { 0.0 };
      // Only the colour tiers reconstruct full-width chroma for the centered
      // decode; a luma-only centered row bins native Y unchanged (siting is a
      // chroma-only property).
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      let want_color =
        rgb.is_some() || rgba.is_some() || hsv.is_some() || rgb_u16.is_some() || rgba_u16.is_some();
      // Whether this call carries any output — the EXACT set the tier preflight
      // (`yuv420p16_native_preflight`'s `need_luma || need_color`) tests. The
      // route (and the siting phase) freezes only on an output-bearing row a tier
      // ACCEPTS; a no-output call consumes no stream state, so it must not freeze.
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      let need_output = luma.is_some()
        || rgb.is_some()
        || rgba.is_some()
        || rgb_u16.is_some()
        || rgba_u16.is_some()
        || hsv.is_some();
      // Freeze the effective 4:2:0 chroma siting on the first output-bearing row
      // (mirrors the `frozen_native_route` freeze below). This CHECK is at the
      // always-compiled choke point every tier passes through; the matching SET
      // rides each tier's accept path (never before dispatch, so a rejected row
      // leaves it unset for a corrected retry). A later row observing a different
      // phase would bin a mixture of co-sited and centered chroma, so it is
      // rejected here before any reconstruction. `Center` and `Bottom` share the
      // horizontal center, so the vertical phase is frozen too (RFC #238 S6e) — a
      // mid-frame Center ⇆ Bottom flip is caught only by comparing
      // `frozen_chroma_bottom_v`.
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      if need_output
        && let Some(frozen) = *frozen_chroma_centered
        && (frozen != center_sited || *frozen_chroma_bottom_v != Some(bottom_v))
      {
        return Err(MixedSinkerError::ChromaSitingChanged(
          ChromaSitingChanged::new(idx),
        ));
      }

      // FILTER FIRST — the filter path has no native fast tier, so it must
      // branch before the area-only native-route machinery below. See the
      // P010 impl for the full rationale; P012 differs only in the 12-bit
      // kernel family.
      if plan.kind().is_filter() {
        // Centered filter reconstructs full-width chroma and decodes 4:4:4, but
        // ONLY after the resample preflight (frozen-output + sequence), so an
        // out-of-sequence / rejected row is caught before the chroma reservation
        // (#180). `packed_yuv422_triple_filter_resample` re-runs the idempotent
        // preflight. Co-sited keeps the fused `p012` half-chroma decode.
        #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
        {
          // Reject a multi-kernel (BICUBLIN) filter plan BEFORE the centered
          // reserve below, mirroring the delegate's own first act (idempotent).
          plan.ensure_single_kernel_filter()?;
          if center_sited && want_color {
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
            // RFC #238 S6e: reserve the bottom-sited vertical-phase lookback
            // AFTER the preflight, BEFORE the commit (#180); only for v=1. The
            // sited call reads it but defers staging until the commit accepts.
            if bottom_v {
              reserve_420_chroma_prev_u16(chroma_prev_u16, w, h)?;
            }
            // `stage = false`: DEFER the lookback advance until AFTER the fallible
            // filter commit accepts the row (below), so a rejected row leaves the
            // predecessor for a clean retry (#180 state-atomicity).
            let uv_full = upsample_420_chroma_sited_p0xx::<BITS>(
              chroma_full_u16,
              chroma_prev_u16,
              chroma_prev_row,
              uv_half,
              idx,
              bottom_v,
              false,
              w,
              BE,
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
              |scratch| {
                for (dst, &s) in scratch[..w].iter_mut().zip(y.iter()) {
                  let logical = if BE { u16::from_be(s) } else { u16::from_le(s) };
                  *dst = logical >> (16 - BITS);
                }
              },
              |scratch| {
                p412_to_rgb_row_endian(y, uv_full, scratch, w, matrix, full_range, use_simd, BE)
              },
              |scratch| {
                p412_to_rgb_u16_row_endian(y, uv_full, scratch, w, matrix, full_range, use_simd, BE)
              },
            );
            // RFC #238 S6e: advance the bottom lookback only AFTER the filter
            // resample accepts the row (`r.is_ok()`) — the sited reconstruction
            // read it above but did not stage, so a rejected row keeps the
            // predecessor for a clean retry.
            if r.is_ok() && bottom_v {
              stage_420_chroma_prev_p0xx(chroma_prev_u16, chroma_prev_row, uv_half, idx, w);
            }
            if r.is_ok() && need_output && frozen_chroma_centered.is_none() {
              *frozen_chroma_centered = Some(center_sited);
              *frozen_chroma_bottom_v = Some(bottom_v);
            }
            return r;
          }
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
          |scratch| {
            for (dst, &s) in scratch[..w].iter_mut().zip(y.iter()) {
              let logical = if BE { u16::from_be(s) } else { u16::from_le(s) };
              *dst = logical >> (16 - BITS);
            }
          },
          |scratch| {
            p012_to_rgb_row_endian(y, uv_half, scratch, w, matrix, full_range, use_simd, BE)
          },
          |scratch| {
            p012_to_rgb_u16_row_endian(y, uv_half, scratch, w, matrix, full_range, use_simd, BE)
          },
        );
        #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
        if r.is_ok() && need_output && frozen_chroma_centered.is_none() {
          *frozen_chroma_centered = Some(center_sited);
          *frozen_chroma_bottom_v = Some(bottom_v);
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
      // a true no-op, fully route-invisible regardless of row index, so it
      // can never spuriously trip `NativeRouteChanged` after the route is
      // frozen. A preflight-rejected (out-of-sequence / frozen)
      // output-bearing call returns Err before the SET, so it leaves
      // `frozen_native_route` untouched and a later same-or-other-route
      // retry is not falsely rejected. (Issue #186 tracks the same gap in
      // the other native families.)
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
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
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
        // Dispatch first; freeze the route to native ONLY after the call
        // returns Ok on an output-bearing row. A no-output call returns
        // Ok(()) with `need_output` false (no freeze); an out-of-sequence /
        // frozen row returns Err via `?` (no freeze) — so only an accepted
        // output-bearing row commits the route.
        //
        // RFC #238 S6b point-of-use siting invalidation, mirroring the planar
        // Yuv420p native arm: a reused sink's cached join is only `reset` between
        // frames, so a frame whose `chroma_location` moved to a different
        // horizontal phase must REBUILD it (`area_chroma_420` folds the phase into
        // the chroma weights). Drop the stale-phase join ONLY on the in-sequence
        // first row of a fresh frame (`idx == 0`, `next_y() == 0`) so a mid-frame
        // / out-of-sequence row rejects against the INTACT join and a corrected
        // retry rebuilds cleanly; a luma-only join carries no chroma phase and is
        // never dropped. Move it OUT (the delegate builds the replacement into the
        // field, untouched until every pre-feed allocation succeeds) and restore
        // the intact prior-phase join on a rejected rebuild so the row mutates no
        // join state.
        let stale_native = idx == 0
          && native_420_u16.as_ref().is_some_and(|join| {
            (join.chroma_phase_centered() == Some(!center_sited)
              || join.chroma_bottom() == Some(!bottom_v))
              && join.next_y() == 0
          });
        let prev_native = if stale_native {
          native_420_u16.take()
        } else {
          None
        };
        let native_result = p0xx_process_native::<BITS, BE>(
          plan,
          native_420_u16,
          p0xx_y_half,
          p0xx_u_half,
          p0xx_v_half,
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
          uv_half,
          chroma_h_phase,
          chroma_v_phase,
          matrix,
          full_range,
          idx,
          w,
          h,
          use_simd,
        );
        // Restore the taken stale-phase join if the delegate's rebuild was
        // rejected at any pre-feed step: it leaves the field `None` on such a
        // failure, so restoring the intact prior-phase join leaves the rejected
        // row mutating no join state. A non-stale row took nothing.
        if stale_native && native_result.is_err() {
          *native_420_u16 = prev_native;
        }
        native_result?;
        #[cfg(feature = "yuv-semi-planar")]
        if frozen_native_route.is_none() && need_output {
          *frozen_native_route = Some(true);
        }
        #[cfg(feature = "yuv-semi-planar")]
        if frozen_chroma_centered.is_none() && need_output {
          *frozen_chroma_centered = Some(center_sited);
          *frozen_chroma_bottom_v = Some(bottom_v);
        }
        return Ok(());
      }
      // Row-stage area tail. Same CHECK-before / SET-after split. Centered colour
      // reconstructs full-width interleaved `u16` chroma (de-interleave +
      // phase-0.5 upsample, plus the `Bottom` even-row vertical box blend) and
      // decodes 4:4:4 — but ONLY after the resample preflight (frozen-output +
      // sequence), so an out-of-sequence / rejected row is caught before the
      // chroma reservation (#180). A luma-only centered row stays on the co-sited
      // arm (which only bins luma).
      // `packed_yuv422_triple_resample` re-runs the idempotent preflight.
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      if center_sited && want_color {
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
        // RFC #238 S6e: reserve the bottom-sited vertical-phase lookback AFTER
        // the preflight, BEFORE the commit (#180); only for v=1. The sited call
        // reads it but defers staging until the commit accepts.
        if bottom_v {
          reserve_420_chroma_prev_u16(chroma_prev_u16, w, h)?;
        }
        // `stage = false`: DEFER the lookback advance until AFTER the fallible
        // row-stage commit accepts the row (below), so a rejected row leaves the
        // predecessor for a clean retry (#180 state-atomicity).
        let uv_full = upsample_420_chroma_sited_p0xx::<BITS>(
          chroma_full_u16,
          chroma_prev_u16,
          chroma_prev_row,
          uv_half,
          idx,
          bottom_v,
          false,
          w,
          BE,
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
          |scratch| {
            for (dst, &s) in scratch[..w].iter_mut().zip(y.iter()) {
              let logical = if BE { u16::from_be(s) } else { u16::from_le(s) };
              *dst = logical >> (16 - BITS);
            }
          },
          |scratch| {
            p412_to_rgb_row_endian(y, uv_full, scratch, w, matrix, full_range, use_simd, BE)
          },
          |scratch| {
            p412_to_rgb_u16_row_endian(y, uv_full, scratch, w, matrix, full_range, use_simd, BE)
          },
        )?;
        // RFC #238 S6e: advance the bottom lookback only AFTER the row-stage
        // resample accepts the row (the `?` above already returned any reject);
        // the sited reconstruction read it but did not stage, so a rejected row
        // keeps the predecessor for a clean retry.
        if bottom_v {
          stage_420_chroma_prev_p0xx(chroma_prev_u16, chroma_prev_row, uv_half, idx, w);
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
      // Row-stage tail. Same CHECK-before / SET-after split: dispatch, then
      // freeze the route to row-stage only when the call accepts an
      // output-bearing row (a no-output call returns Ok with `need_output`
      // false; an out-of-sequence / frozen row returns Err via `?`).
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
        |scratch| {
          for (dst, &s) in scratch[..w].iter_mut().zip(y.iter()) {
            let logical = if BE { u16::from_be(s) } else { u16::from_le(s) };
            *dst = logical >> (16 - BITS);
          }
        },
        |scratch| p012_to_rgb_row_endian(y, uv_half, scratch, w, matrix, full_range, use_simd, BE),
        |scratch| {
          p012_to_rgb_u16_row_endian(y, uv_half, scratch, w, matrix, full_range, use_simd, BE)
        },
      )?;
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      if frozen_native_route.is_none() && need_output {
        *frozen_native_route = Some(false);
      }
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      if frozen_chroma_centered.is_none() && need_output {
        *frozen_chroma_centered = Some(center_sited);
        *frozen_chroma_bottom_v = Some(bottom_v);
      }
      return Ok(());
    }

    // Resolve the output set up front so the no-output guard short-circuits
    // before ANY per-row offset arithmetic and the atomicity preflight runs
    // before any output row is written.
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();
    let want_rgb_u16 = rgb_u16.is_some();
    let want_rgba_u16 = rgba_u16.is_some();
    let want_color = want_rgb || want_rgba || want_hsv || want_rgb_u16 || want_rgba_u16;
    // Repo-wide no-output invariant (RFC #238 S6e, mirrors the planar Yuv420p /
    // NV S4-C direct twins): a `process` call carrying NO output — no colour, no
    // luma — must run NOTHING: no per-row offset arithmetic, no allocation, no
    // state mutation (the bottom-sited vertical lookback included). Returning
    // HERE, before the `idx * w` offsets below, keeps the invariant overflow-safe
    // (a no-output call never ran an attach-time `w x h` validation, so `idx * w`
    // could overflow on a 32-bit target) AND means such a row never reserves
    // `chroma_prev_u16` nor primes the lookback (so it can never make a later
    // colour even row box-blend through an invisible, never-output row). P012
    // exposes no `luma_u16`.
    let need_output = want_color || luma.is_some();
    if !need_output {
      return Ok(());
    }

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // Chroma siting (#302): the centered horizontal sitings reconstruct chroma
    // at the phase-0.5 position; the default / co-sited path keeps the
    // byte-identical fused decode (the fused P-format 4:2:0 kernels
    // de-interleave + upsample chroma in-register, exactly as before).
    #[cfg(feature = "yuv-planar")]
    let center_sited = chroma_420_center_sited_h(chroma_location);
    // RFC #238 S6e: `Bottom` (a strict sub-case of `center_sited`) additionally
    // box-blends the even output row's chroma with the previous chroma row via
    // the `chroma_prev_u16` lookback maintained below; `Center` / `Top` keep the
    // vertical-replicate (co-sited) decode, byte-identical to S6b.
    #[cfg(feature = "yuv-planar")]
    let bottom_v = chroma_420_bottom_sited_v(chroma_location);

    // Atomicity preflight (#302 / #308, cf. the crate's #180 resample fix and
    // the planar_8bit / semi_planar_8bit 8-bit siblings): reserve EVERY fallible
    // row scratch this identity row can touch BEFORE any output row is written
    // (the luma plane below, then the u16 / u8 RGB / RGBA / HSV fan-out), so an
    // allocator refusal returns a typed `AllocationFailed` leaving the output
    // frame untouched rather than partially mutated. Two scratches can grow:
    //  1. the centered-siting full-width interleaved `u16` chroma
    //     (`chroma_full_u16`), needed by ANY colour output (u8 OR u16 RGB / RGBA
    //     / HSV); and
    //  2. the u8 RGB row buffer, reached exactly when a colour decode needs an
    //     RGB row but no caller RGB buffer is borrowable — `want_hsv &&
    //     want_rgba && !want_rgb` (`rgb_row_buf_or_scratch`'s own scratch arm).
    // The later `upsample_420_chroma_sited_p0xx` / `rgb_row_buf_or_scratch`
    // calls reuse the already-sized buffers, so the default path is
    // byte-identical; only the failure-path ordering changes. The u16 RGB / RGBA
    // outputs write straight into their caller buffers (the rgb_u16 plane itself
    // stages the rgba_u16 expand) and never grow a scratch of their own. A
    // luma-only row consumes no centered chroma, so it neither reserves nor
    // upsamples it (the reserve makes the later upsample infallible).
    #[cfg(feature = "yuv-planar")]
    let need_centered_chroma = center_sited && want_color;
    #[cfg(feature = "yuv-planar")]
    if need_centered_chroma {
      reserve_420_chroma_full_u16(chroma_full_u16, w, h)?;
    }
    // Bottom-sited vertical-phase one-row chroma lookback (RFC #238 S6e): reserve
    // it on EVERY bottom-sited OUTPUT row — colour OR luma-only — because the
    // lookback is maintained so a LATER colour row can box-blend it (the
    // luma-only staging runs below, before any luma write). A no-output row
    // returned early above, so it never reaches here and never primes the
    // lookback. Reserved BEFORE any output write (the #180 preflight ordering).
    #[cfg(feature = "yuv-planar")]
    if bottom_v {
      reserve_420_chroma_prev_u16(chroma_prev_u16, w, h)?;
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

    // Bottom-sited LUMA-ONLY row (RFC #238 S6e): the colour upsample below —
    // which normally refreshes the vertical lookback — won't run, so stage the
    // current interleaved chroma row HERE, after its reservation above and BEFORE
    // any luma write, so a later colour row in the same frame can box-blend it (a
    // luma-only-then-colour in-order sequence reconstructs the same Bottom
    // vertical phase as the all-output walk). A colour row instead stages inside
    // `upsample_420_chroma_sited_p0xx` after reading the previous lookback, so
    // this is skipped for it; a no-output row returned early above. The validity
    // tag (`chroma_prev_row`) still guards out-of-sequence / cross-frame reads.
    #[cfg(feature = "yuv-planar")]
    if bottom_v && !want_color {
      stage_420_chroma_prev_p0xx(chroma_prev_u16, chroma_prev_row, row.uv_half(), idx, w);
    }

    // Centered full-width INTERLEAVED chroma (phase-0.5), reconstructed ONCE per
    // row from the wire-format half-width interleaved UV and reused by every
    // colour decode (u16 and u8). Infallible — the scratch was reserved above.
    // The default left/unspecified siting leaves it `None`, so the fused
    // P-format 4:2:0 kernels de-interleave + upsample chroma in-register and the
    // output stays byte-identical. `Center` / `Top` take the plain horizontal
    // centered phase-0.5 fold; `Bottom` (RFC #238 S6e) additionally box-blends
    // the even output row with the previous chroma row (`stage = true` — the
    // direct decode's post-reconstruction work is infallible).
    #[cfg(feature = "yuv-planar")]
    let centered: Option<&[u16]> = if need_centered_chroma {
      Some(upsample_420_chroma_sited_p0xx::<BITS>(
        chroma_full_u16,
        chroma_prev_u16,
        chroma_prev_row,
        row.uv_half(),
        idx,
        bottom_v,
        true,
        w,
        BE,
      ))
    } else {
      None
    };
    #[cfg(not(feature = "yuv-planar"))]
    let centered: Option<&[u16]> = None;

    // Luma: P012 samples are high‑bit‑packed (`value << 4`). Taking the
    // high byte via `>> 8` gives the top 8 bits of the 12‑bit value —
    // identical accessor to P010 (both put active bits in the high
    // `BITS` positions of the `u16`). Routed through the native-Y kernel
    // (bit-identical to the former inline `>> 8` loop, including the
    // BE-wire normalization).
    if let Some(luma) = luma.as_deref_mut() {
      p012_to_luma_row_endian(row.y(), &mut luma[one_plane_start..one_plane_end], w, BE);
    }

    // ===== u16 RGB / RGBA path (Strategy A) — see Yuv420p10 for rationale.
    // u16 outputs are low-bit-packed (yuv420p12le convention), not
    // P012's high-bit packing.
    if want_rgba_u16 && !want_rgb_u16 {
      let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row =
        rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
      if let Some(uv_full) = centered {
        p412_to_rgba_u16_row_endian(
          row.y(),
          uv_full,
          rgba_u16_row,
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
          BE,
        );
      } else {
        p012_to_rgba_u16_row_endian(
          row.y(),
          row.uv_half(),
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
      if let Some(uv_full) = centered {
        p412_to_rgb_u16_row_endian(
          row.y(),
          uv_full,
          rgb_u16_row,
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
          BE,
        );
      } else {
        p012_to_rgb_u16_row_endian(
          row.y(),
          row.uv_half(),
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
    // HSV-only (no RGB / RGBA) goes direct through `p012_to_hsv_row` (no
    // source-width RGB scratch); see the P010 impl for the routing
    // rationale and the #263 row-stage deferral.
    let want_hsv_direct = want_hsv && !want_rgb && !want_rgba;
    let need_rgb_kernel = want_rgb || (want_hsv && want_rgba);

    if want_hsv_direct {
      let hsv = hsv.as_mut().expect("want_hsv_direct implies hsv attached");
      let (h, s, v) = hsv.hsv();
      if let Some(uv_full) = centered {
        p412_to_hsv_row_endian(
          row.y(),
          uv_full,
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
        p012_to_hsv_row_endian(
          row.y(),
          row.uv_half(),
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
      if let Some(uv_full) = centered {
        p412_to_rgba_row_endian(
          row.y(),
          uv_full,
          rgba_row,
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
          BE,
        );
      } else {
        p012_to_rgba_row_endian(
          row.y(),
          row.uv_half(),
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

    if let Some(uv_full) = centered {
      p412_to_rgb_row_endian(
        row.y(),
        uv_full,
        rgb_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
        BE,
      );
    } else {
      p012_to_rgb_row_endian(
        row.y(),
        row.uv_half(),
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

// ---- P016 impl ---------------------------------------------------------

impl<'a, R, const BE: bool> MixedSinker<'a, P016<BE>, R> {
  /// Attaches a packed **`u16`** RGB output buffer. Produces 16‑bit
  /// output in `[0, 65535]` — at 16 bits there is no high‑ vs
  /// low‑packing distinction, so the output matches
  /// [`MixedSinker<Yuv420p16>::with_rgb_u16`] numerically.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgb_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_rgb_u16(buf)?;
    Ok(self)
  }

  /// In-place variant of [`with_rgb_u16`](Self::with_rgb_u16).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgb_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
    let expected_elements = self.frame_elems(3)?;
    if buf.len() < expected_elements {
      return Err(MixedSinkerError::InsufficientRgbU16Buffer(
        InsufficientBuffer::new(expected_elements, buf.len()),
      ));
    }
    self.rgb_u16 = Some(buf);
    Ok(self)
  }

  /// Attaches a packed **8‑bit** RGBA output buffer. The 16‑bit P016
  /// source (semi‑planar) is converted to 8‑bit RGBA via the dedicated
  /// `BITS = 16` kernel family; alpha = `0xFF`.
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

  /// Attaches a packed **`u16`** RGBA output buffer. 16‑bit output
  /// (full `u16` range). Length is measured in `u16` **elements**
  /// (`width x height x 4`). Alpha element is `u16::MAX`.
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

impl<R, const BE: bool> P016Sink<BE> for MixedSinker<'_, P016<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, P016<BE>, R> {
  type Input<'r> = P016Row<'r>;
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

  fn process(&mut self, row: P016Row<'_>) -> Result<(), Self::Error> {
    // Bit depth is fixed by the format (16). Used for the u16 RGBA
    // expand path's alpha pad (`alpha = u16::MAX` at this depth).
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
    if row.uv_half().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::UvHalf16,
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

    // Chroma siting (#302): drives the identity-plan horizontal chroma phase.
    // `Copy`, so read it out before the field split-borrow below. Gated like
    // its only consumer (`chroma_420_center_sited_h` + the 4:4:4 P-format
    // kernels need `yuv-planar`); a semi-planar-only build keeps the default
    // decode.
    #[cfg(feature = "yuv-planar")]
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
      luma_scratch_u16,
      rgb_stream,
      rgb_stream_u16,
      luma_stream_u16,
      rgb_filter_stream,
      rgb_filter_stream_u16,
      luma_filter_stream_u16,
      resample_outputs,
      plan,
      #[cfg(feature = "yuv-planar")]
      native,
      #[cfg(feature = "yuv-planar")]
      native_420_u16,
      #[cfg(feature = "yuv-planar")]
      p0xx_y_half,
      #[cfg(feature = "yuv-planar")]
      p0xx_u_half,
      #[cfg(feature = "yuv-planar")]
      p0xx_v_half,
      #[cfg(feature = "yuv-planar")]
      frozen_native_route,
      // Full-width interleaved chroma staging for the centered-siting (#302)
      // identity decode.
      #[cfg(feature = "yuv-planar")]
      chroma_full_u16,
      // RFC #238 S6b: the 4:2:0 horizontal chroma siting phase frozen on the
      // first output row (the `u16` semi-planar twin of the planar Yuv420p
      // `frozen_chroma_centered`).
      #[cfg(feature = "yuv-planar")]
      frozen_chroma_centered,
      // RFC #238 S6e: the vertical (`Bottom`) chroma phase frozen alongside it,
      // plus the one-row interleaved chroma lookback + its validity tag, so a
      // `Bottom` even output row box-blends the previous chroma row.
      #[cfg(feature = "yuv-planar")]
      frozen_chroma_bottom_v,
      #[cfg(feature = "yuv-planar")]
      chroma_prev_u16,
      #[cfg(feature = "yuv-planar")]
      chroma_prev_row,
      ..
    } = self;

    // Non-identity plan. When the native tier is enabled, bin the native
    // planes at output resolution and convert once per output row (de-pack
    // into wrapper scratch first); otherwise feed the shared high-bit 4:2:2
    // triple-resample tail. See the P010 impl for the full rationale. At 16
    // bits the Y de-pack shift `>> (16 - BITS)` is `>> 0` (the 16-bit
    // sample already is the logical value), and the 16-bit kernel family is
    // dedicated (i32/i64 chroma, no Q15 downshift).
    if let Some(plan) = plan.as_ref() {
      let matrix = row.matrix();
      let full_range = row.full_range();
      let (y, uv_half) = (row.y(), row.uv_half());
      // RFC #238 S6b/S6e — 4:2:0 chroma siting for the high-bit semi-planar
      // P016, mirroring the planar Yuv420p16 twin (S6a/S6d) on the semi-planar
      // de-interleave (S3b/S4-C). The centered group
      // (`chroma_420_center_sited_h`) samples chroma at `+0.5` luma = `+0.25`
      // chroma-sample horizontally; S6e additionally folds the VERTICAL `v = 1`
      // triangle for `Bottom` ([`chroma_420_bottom_sited_v`], a strict sub-case
      // of `center_sited`), while `Center` / `Top` keep `v_phase = 0` (co-sited
      // vertical, byte-identical to S6b). The co-sited / unspecified group is
      // phase 0 (byte-identical). The native fast tier folds both phases into the
      // `area_chroma_420` chroma weights; the filter and row-stage tiers
      // reconstruct full-width interleaved `u16` chroma (de-interleave + phase-0.5
      // upsample into `chroma_full_u16`, plus the `Bottom` even-row vertical box
      // blend against the `chroma_prev_u16` lookback) and decode 4:4:4 via the
      // `p416` full-chroma kernels — the co-sited arms keep the fused `p016`
      // half-chroma decode.
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      let center_sited = chroma_420_center_sited_h(chroma_location);
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      let bottom_v = chroma_420_bottom_sited_v(chroma_location);
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      let chroma_h_phase = if center_sited {
        YUV422P_CENTERED_H_PHASE
      } else {
        0.0
      };
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      let chroma_v_phase = if bottom_v { 1.0 } else { 0.0 };
      // Only the colour tiers reconstruct full-width chroma for the centered
      // decode; a luma-only centered row bins native Y unchanged (siting is a
      // chroma-only property).
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      let want_color =
        rgb.is_some() || rgba.is_some() || hsv.is_some() || rgb_u16.is_some() || rgba_u16.is_some();
      // Whether this call carries any output — the EXACT set the tier preflight
      // (`yuv420p16_native_preflight`'s `need_luma || need_color`) tests. The
      // route (and the siting phase) freezes only on an output-bearing row a tier
      // ACCEPTS; a no-output call consumes no stream state, so it must not freeze.
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      let need_output = luma.is_some()
        || rgb.is_some()
        || rgba.is_some()
        || rgb_u16.is_some()
        || rgba_u16.is_some()
        || hsv.is_some();
      // Freeze the effective 4:2:0 chroma siting on the first output-bearing row
      // (mirrors the `frozen_native_route` freeze below). This CHECK is at the
      // always-compiled choke point every tier passes through; the matching SET
      // rides each tier's accept path (never before dispatch, so a rejected row
      // leaves it unset for a corrected retry). A later row observing a different
      // phase would bin a mixture of co-sited and centered chroma, so it is
      // rejected here before any reconstruction. `Center` and `Bottom` share the
      // horizontal center, so the vertical phase is frozen too (RFC #238 S6e) — a
      // mid-frame Center ⇆ Bottom flip is caught only by comparing
      // `frozen_chroma_bottom_v`.
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      if need_output
        && let Some(frozen) = *frozen_chroma_centered
        && (frozen != center_sited || *frozen_chroma_bottom_v != Some(bottom_v))
      {
        return Err(MixedSinkerError::ChromaSitingChanged(
          ChromaSitingChanged::new(idx),
        ));
      }

      // FILTER FIRST — the filter path has no native fast tier, so it must
      // branch before the area-only native-route machinery below. See the
      // P010 impl for the full rationale. At 16 bits the native max is
      // `u16::MAX`, so the filter tail's clamp is a value no-op.
      if plan.kind().is_filter() {
        // Centered filter reconstructs full-width chroma and decodes 4:4:4, but
        // ONLY after the resample preflight (frozen-output + sequence), so an
        // out-of-sequence / rejected row is caught before the chroma reservation
        // (#180). `packed_yuv422_triple_filter_resample` re-runs the idempotent
        // preflight. Co-sited keeps the fused `p016` half-chroma decode.
        #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
        {
          // Reject a multi-kernel (BICUBLIN) filter plan BEFORE the centered
          // reserve below, mirroring the delegate's own first act (idempotent).
          plan.ensure_single_kernel_filter()?;
          if center_sited && want_color {
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
            // RFC #238 S6e: reserve the bottom-sited vertical-phase lookback
            // AFTER the preflight, BEFORE the commit (#180); only for v=1. The
            // sited call reads it but defers staging until the commit accepts.
            if bottom_v {
              reserve_420_chroma_prev_u16(chroma_prev_u16, w, h)?;
            }
            // `stage = false`: DEFER the lookback advance until AFTER the fallible
            // filter commit accepts the row (below), so a rejected row leaves the
            // predecessor for a clean retry (#180 state-atomicity).
            let uv_full = upsample_420_chroma_sited_p0xx::<BITS>(
              chroma_full_u16,
              chroma_prev_u16,
              chroma_prev_row,
              uv_half,
              idx,
              bottom_v,
              false,
              w,
              BE,
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
              |scratch| {
                for (dst, &s) in scratch[..w].iter_mut().zip(y.iter()) {
                  let logical = if BE { u16::from_be(s) } else { u16::from_le(s) };
                  *dst = logical >> (16 - BITS);
                }
              },
              |scratch| {
                p416_to_rgb_row_endian(y, uv_full, scratch, w, matrix, full_range, use_simd, BE)
              },
              |scratch| {
                p416_to_rgb_u16_row_endian(y, uv_full, scratch, w, matrix, full_range, use_simd, BE)
              },
            );
            // RFC #238 S6e: advance the bottom lookback only AFTER the filter
            // resample accepts the row (`r.is_ok()`) — the sited reconstruction
            // read it above but did not stage, so a rejected row keeps the
            // predecessor for a clean retry.
            if r.is_ok() && bottom_v {
              stage_420_chroma_prev_p0xx(chroma_prev_u16, chroma_prev_row, uv_half, idx, w);
            }
            if r.is_ok() && need_output && frozen_chroma_centered.is_none() {
              *frozen_chroma_centered = Some(center_sited);
              *frozen_chroma_bottom_v = Some(bottom_v);
            }
            return r;
          }
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
          |scratch| {
            for (dst, &s) in scratch[..w].iter_mut().zip(y.iter()) {
              let logical = if BE { u16::from_be(s) } else { u16::from_le(s) };
              *dst = logical >> (16 - BITS);
            }
          },
          |scratch| {
            p016_to_rgb_row_endian(y, uv_half, scratch, w, matrix, full_range, use_simd, BE)
          },
          |scratch| {
            p016_to_rgb_u16_row_endian(y, uv_half, scratch, w, matrix, full_range, use_simd, BE)
          },
        );
        #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
        if r.is_ok() && need_output && frozen_chroma_centered.is_none() {
          *frozen_chroma_centered = Some(center_sited);
          *frozen_chroma_bottom_v = Some(bottom_v);
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
      // a true no-op, fully route-invisible regardless of row index, so it
      // can never spuriously trip `NativeRouteChanged` after the route is
      // frozen. A preflight-rejected (out-of-sequence / frozen)
      // output-bearing call returns Err before the SET, so it leaves
      // `frozen_native_route` untouched and a later same-or-other-route
      // retry is not falsely rejected. (Issue #186 tracks the same gap in
      // the other native families.)
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
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
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
        // Dispatch first; freeze the route to native ONLY after the call
        // returns Ok on an output-bearing row. A no-output call returns
        // Ok(()) with `need_output` false (no freeze); an out-of-sequence /
        // frozen row returns Err via `?` (no freeze) — so only an accepted
        // output-bearing row commits the route.
        //
        // RFC #238 S6b point-of-use siting invalidation, mirroring the planar
        // Yuv420p native arm: a reused sink's cached join is only `reset` between
        // frames, so a frame whose `chroma_location` moved to a different
        // horizontal phase must REBUILD it (`area_chroma_420` folds the phase into
        // the chroma weights). Drop the stale-phase join ONLY on the in-sequence
        // first row of a fresh frame (`idx == 0`, `next_y() == 0`) so a mid-frame
        // / out-of-sequence row rejects against the INTACT join and a corrected
        // retry rebuilds cleanly; a luma-only join carries no chroma phase and is
        // never dropped. Move it OUT (the delegate builds the replacement into the
        // field, untouched until every pre-feed allocation succeeds) and restore
        // the intact prior-phase join on a rejected rebuild so the row mutates no
        // join state.
        let stale_native = idx == 0
          && native_420_u16.as_ref().is_some_and(|join| {
            (join.chroma_phase_centered() == Some(!center_sited)
              || join.chroma_bottom() == Some(!bottom_v))
              && join.next_y() == 0
          });
        let prev_native = if stale_native {
          native_420_u16.take()
        } else {
          None
        };
        let native_result = p0xx_process_native::<BITS, BE>(
          plan,
          native_420_u16,
          p0xx_y_half,
          p0xx_u_half,
          p0xx_v_half,
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
          uv_half,
          chroma_h_phase,
          chroma_v_phase,
          matrix,
          full_range,
          idx,
          w,
          h,
          use_simd,
        );
        // Restore the taken stale-phase join if the delegate's rebuild was
        // rejected at any pre-feed step: it leaves the field `None` on such a
        // failure, so restoring the intact prior-phase join leaves the rejected
        // row mutating no join state. A non-stale row took nothing.
        if stale_native && native_result.is_err() {
          *native_420_u16 = prev_native;
        }
        native_result?;
        #[cfg(feature = "yuv-semi-planar")]
        if frozen_native_route.is_none() && need_output {
          *frozen_native_route = Some(true);
        }
        #[cfg(feature = "yuv-semi-planar")]
        if frozen_chroma_centered.is_none() && need_output {
          *frozen_chroma_centered = Some(center_sited);
          *frozen_chroma_bottom_v = Some(bottom_v);
        }
        return Ok(());
      }
      // Row-stage area tail. Same CHECK-before / SET-after split. Centered colour
      // reconstructs full-width interleaved `u16` chroma (de-interleave +
      // phase-0.5 upsample, plus the `Bottom` even-row vertical box blend) and
      // decodes 4:4:4 — but ONLY after the resample preflight (frozen-output +
      // sequence), so an out-of-sequence / rejected row is caught before the
      // chroma reservation (#180). A luma-only centered row stays on the co-sited
      // arm (which only bins luma).
      // `packed_yuv422_triple_resample` re-runs the idempotent preflight.
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      if center_sited && want_color {
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
        // RFC #238 S6e: reserve the bottom-sited vertical-phase lookback AFTER
        // the preflight, BEFORE the commit (#180); only for v=1. The sited call
        // reads it but defers staging until the commit accepts.
        if bottom_v {
          reserve_420_chroma_prev_u16(chroma_prev_u16, w, h)?;
        }
        // `stage = false`: DEFER the lookback advance until AFTER the fallible
        // row-stage commit accepts the row (below), so a rejected row leaves the
        // predecessor for a clean retry (#180 state-atomicity).
        let uv_full = upsample_420_chroma_sited_p0xx::<BITS>(
          chroma_full_u16,
          chroma_prev_u16,
          chroma_prev_row,
          uv_half,
          idx,
          bottom_v,
          false,
          w,
          BE,
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
          |scratch| {
            for (dst, &s) in scratch[..w].iter_mut().zip(y.iter()) {
              let logical = if BE { u16::from_be(s) } else { u16::from_le(s) };
              *dst = logical >> (16 - BITS);
            }
          },
          |scratch| {
            p416_to_rgb_row_endian(y, uv_full, scratch, w, matrix, full_range, use_simd, BE)
          },
          |scratch| {
            p416_to_rgb_u16_row_endian(y, uv_full, scratch, w, matrix, full_range, use_simd, BE)
          },
        )?;
        // RFC #238 S6e: advance the bottom lookback only AFTER the row-stage
        // resample accepts the row (the `?` above already returned any reject);
        // the sited reconstruction read it but did not stage, so a rejected row
        // keeps the predecessor for a clean retry.
        if bottom_v {
          stage_420_chroma_prev_p0xx(chroma_prev_u16, chroma_prev_row, uv_half, idx, w);
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
      // Row-stage tail. Same CHECK-before / SET-after split: dispatch, then
      // freeze the route to row-stage only when the call accepts an
      // output-bearing row (a no-output call returns Ok with `need_output`
      // false; an out-of-sequence / frozen row returns Err via `?`).
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
        |scratch| {
          for (dst, &s) in scratch[..w].iter_mut().zip(y.iter()) {
            let logical = if BE { u16::from_be(s) } else { u16::from_le(s) };
            *dst = logical >> (16 - BITS);
          }
        },
        |scratch| p016_to_rgb_row_endian(y, uv_half, scratch, w, matrix, full_range, use_simd, BE),
        |scratch| {
          p016_to_rgb_u16_row_endian(y, uv_half, scratch, w, matrix, full_range, use_simd, BE)
        },
      )?;
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      if frozen_native_route.is_none() && need_output {
        *frozen_native_route = Some(false);
      }
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      if frozen_chroma_centered.is_none() && need_output {
        *frozen_chroma_centered = Some(center_sited);
        *frozen_chroma_bottom_v = Some(bottom_v);
      }
      return Ok(());
    }

    // Resolve the output set up front so the no-output guard short-circuits
    // before ANY per-row offset arithmetic and the atomicity preflight runs
    // before any output row is written.
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();
    let want_rgb_u16 = rgb_u16.is_some();
    let want_rgba_u16 = rgba_u16.is_some();
    let want_color = want_rgb || want_rgba || want_hsv || want_rgb_u16 || want_rgba_u16;
    // Repo-wide no-output invariant (RFC #238 S6e, mirrors the planar Yuv420p /
    // NV S4-C direct twins): a `process` call carrying NO output — no colour, no
    // luma — must run NOTHING: no per-row offset arithmetic, no allocation, no
    // state mutation (the bottom-sited vertical lookback included). Returning
    // HERE, before the `idx * w` offsets below, keeps the invariant overflow-safe
    // (a no-output call never ran an attach-time `w x h` validation, so `idx * w`
    // could overflow on a 32-bit target) AND means such a row never reserves
    // `chroma_prev_u16` nor primes the lookback (so it can never make a later
    // colour even row box-blend through an invisible, never-output row). P016
    // exposes no `luma_u16`.
    let need_output = want_color || luma.is_some();
    if !need_output {
      return Ok(());
    }

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // Chroma siting (#302): the centered horizontal sitings reconstruct chroma
    // at the phase-0.5 position; the default / co-sited path keeps the
    // byte-identical fused decode (the fused P-format 4:2:0 kernels
    // de-interleave + upsample chroma in-register, exactly as before).
    #[cfg(feature = "yuv-planar")]
    let center_sited = chroma_420_center_sited_h(chroma_location);
    // RFC #238 S6e: `Bottom` (a strict sub-case of `center_sited`) additionally
    // box-blends the even output row's chroma with the previous chroma row via
    // the `chroma_prev_u16` lookback maintained below; `Center` / `Top` keep the
    // vertical-replicate (co-sited) decode, byte-identical to S6b.
    #[cfg(feature = "yuv-planar")]
    let bottom_v = chroma_420_bottom_sited_v(chroma_location);

    // Atomicity preflight (#302 / #308, cf. the crate's #180 resample fix and
    // the planar_8bit / semi_planar_8bit 8-bit siblings): reserve EVERY fallible
    // row scratch this identity row can touch BEFORE any output row is written
    // (the luma plane below, then the u16 / u8 RGB / RGBA / HSV fan-out), so an
    // allocator refusal returns a typed `AllocationFailed` leaving the output
    // frame untouched rather than partially mutated. Two scratches can grow:
    //  1. the centered-siting full-width interleaved `u16` chroma
    //     (`chroma_full_u16`), needed by ANY colour output (u8 OR u16 RGB / RGBA
    //     / HSV); and
    //  2. the u8 RGB row buffer, reached exactly when a colour decode needs an
    //     RGB row but no caller RGB buffer is borrowable — `want_hsv &&
    //     want_rgba && !want_rgb` (`rgb_row_buf_or_scratch`'s own scratch arm).
    // The later `upsample_420_chroma_sited_p0xx` / `rgb_row_buf_or_scratch`
    // calls reuse the already-sized buffers, so the default path is
    // byte-identical; only the failure-path ordering changes. The u16 RGB / RGBA
    // outputs write straight into their caller buffers (the rgb_u16 plane itself
    // stages the rgba_u16 expand) and never grow a scratch of their own. A
    // luma-only row consumes no centered chroma, so it neither reserves nor
    // upsamples it (the reserve makes the later upsample infallible).
    #[cfg(feature = "yuv-planar")]
    let need_centered_chroma = center_sited && want_color;
    #[cfg(feature = "yuv-planar")]
    if need_centered_chroma {
      reserve_420_chroma_full_u16(chroma_full_u16, w, h)?;
    }
    // Bottom-sited vertical-phase one-row chroma lookback (RFC #238 S6e): reserve
    // it on EVERY bottom-sited OUTPUT row — colour OR luma-only — because the
    // lookback is maintained so a LATER colour row can box-blend it (the
    // luma-only staging runs below, before any luma write). A no-output row
    // returned early above, so it never reaches here and never primes the
    // lookback. Reserved BEFORE any output write (the #180 preflight ordering).
    #[cfg(feature = "yuv-planar")]
    if bottom_v {
      reserve_420_chroma_prev_u16(chroma_prev_u16, w, h)?;
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

    // Bottom-sited LUMA-ONLY row (RFC #238 S6e): the colour upsample below —
    // which normally refreshes the vertical lookback — won't run, so stage the
    // current interleaved chroma row HERE, after its reservation above and BEFORE
    // any luma write, so a later colour row in the same frame can box-blend it (a
    // luma-only-then-colour in-order sequence reconstructs the same Bottom
    // vertical phase as the all-output walk). A colour row instead stages inside
    // `upsample_420_chroma_sited_p0xx` after reading the previous lookback, so
    // this is skipped for it; a no-output row returned early above. The validity
    // tag (`chroma_prev_row`) still guards out-of-sequence / cross-frame reads.
    #[cfg(feature = "yuv-planar")]
    if bottom_v && !want_color {
      stage_420_chroma_prev_p0xx(chroma_prev_u16, chroma_prev_row, row.uv_half(), idx, w);
    }

    // Centered full-width INTERLEAVED chroma (phase-0.5), reconstructed ONCE per
    // row from the wire-format half-width interleaved UV and reused by every
    // colour decode (u16 and u8). Infallible — the scratch was reserved above.
    // The default left/unspecified siting leaves it `None`, so the fused
    // P-format 4:2:0 kernels de-interleave + upsample chroma in-register and the
    // output stays byte-identical. `Center` / `Top` take the plain horizontal
    // centered phase-0.5 fold; `Bottom` (RFC #238 S6e) additionally box-blends
    // the even output row with the previous chroma row (`stage = true` — the
    // direct decode's post-reconstruction work is infallible).
    #[cfg(feature = "yuv-planar")]
    let centered: Option<&[u16]> = if need_centered_chroma {
      Some(upsample_420_chroma_sited_p0xx::<BITS>(
        chroma_full_u16,
        chroma_prev_u16,
        chroma_prev_row,
        row.uv_half(),
        idx,
        bottom_v,
        true,
        w,
        BE,
      ))
    } else {
      None
    };
    #[cfg(not(feature = "yuv-planar"))]
    let centered: Option<&[u16]> = None;

    // Luma: 16‑bit Y value >> 8 is the top byte. Routed through the
    // native-Y kernel (bit-identical to the former inline `>> 8` loop,
    // including the BE-wire normalization).
    if let Some(luma) = luma.as_deref_mut() {
      p016_to_luma_row_endian(row.y(), &mut luma[one_plane_start..one_plane_end], w, BE);
    }

    // ===== u16 RGB / RGBA path (Strategy A) — see Yuv420p10 for rationale.
    if want_rgba_u16 && !want_rgb_u16 {
      let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row =
        rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
      if let Some(uv_full) = centered {
        p416_to_rgba_u16_row_endian(
          row.y(),
          uv_full,
          rgba_u16_row,
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
          BE,
        );
      } else {
        p016_to_rgba_u16_row_endian(
          row.y(),
          row.uv_half(),
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
      if let Some(uv_full) = centered {
        p416_to_rgb_u16_row_endian(
          row.y(),
          uv_full,
          rgb_u16_row,
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
          BE,
        );
      } else {
        p016_to_rgb_u16_row_endian(
          row.y(),
          row.uv_half(),
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
    // HSV-only (no RGB / RGBA) goes direct through `p016_to_hsv_row` (no
    // source-width RGB scratch); see the P010 impl for the routing
    // rationale and the #263 row-stage deferral.
    let want_hsv_direct = want_hsv && !want_rgb && !want_rgba;
    let need_rgb_kernel = want_rgb || (want_hsv && want_rgba);

    if want_hsv_direct {
      let hsv = hsv.as_mut().expect("want_hsv_direct implies hsv attached");
      let (h, s, v) = hsv.hsv();
      if let Some(uv_full) = centered {
        p416_to_hsv_row_endian(
          row.y(),
          uv_full,
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
        p016_to_hsv_row_endian(
          row.y(),
          row.uv_half(),
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
      if let Some(uv_full) = centered {
        p416_to_rgba_row_endian(
          row.y(),
          uv_full,
          rgba_row,
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
          BE,
        );
      } else {
        p016_to_rgba_row_endian(
          row.y(),
          row.uv_half(),
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

    if let Some(uv_full) = centered {
      p416_to_rgb_row_endian(
        row.y(),
        uv_full,
        rgb_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
        BE,
      );
    } else {
      p016_to_rgb_row_endian(
        row.y(),
        row.uv_half(),
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
