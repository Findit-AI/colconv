//! Native fast-tier 4:2:0 decimator for the **high-bit planar** YUV family
//! (`Yuv420p10` / `Yuv420p12` / `Yuv420p14` / `Yuv420p16`, LE + BE wire).
//!
//! The parallel `u16` twin of the 8-bit
//! [`yuv420p_process_native`](crate::sinker::mixed::planar_8bit::yuv420p_process_native):
//! bin the native Y / U / V planes straight to the output grid, then
//! convert ONCE per output row at output resolution — vs the row-stage
//! tier ([`packed_yuv422_triple_resample`](crate::sinker::mixed::packed_yuv422_triple_resample)),
//! which converts each source row at source width then bins. This is the
//! demand-driven P2 fast tier: default-on, within a small tolerance of the
//! row-stage tier (NOT byte-identical to the direct path), bench-gated.
//!
//! Built as a SEPARATE `u16` path rather than making the 8-bit path
//! element-generic — the nv12 lesson: reuse / parallel the proven path,
//! never refactor it. Const-generic over `BITS` only (one body, four call
//! sites), mirroring
//! [`packed_yuv422_triple_resample`](crate::sinker::mixed::packed_yuv422_triple_resample).
//!
//! The triple-bin + convert-kernel independence is the highest risk (the
//! #37 lesson: the u8 and u16 `YUV→RGB` kernels round INDEPENDENTLY, and a
//! u16-output kernel must never be narrowed from a u8 one or vice versa).
//! Per output row the staged host-native u16 Y / U / V feed:
//! - `luma` (u8): `dst = (Y >> (BITS - 8)) as u8`.
//! - `rgb_u16` / `rgba_u16`: the native-depth 4:4:4 u16 kernel
//!   ([`yuv_444p_n_to_rgb_u16_row`](crate::row::yuv_444p_n_to_rgb_u16_row)
//!   for BITS ∈ {9, 10, 12, 14}, the dedicated `yuv444p16_to_rgb_u16_row`
//!   for BITS = 16), then expand to rgba_u16.
//! - `rgb` / `rgba` / `hsv` (u8): the u16-INPUT → u8-OUTPUT 4:4:4 kernel
//!   ([`yuv_444p_n_to_rgb_row`](crate::row::yuv_444p_n_to_rgb_row) /
//!   `yuv444p16_to_rgb_row`), the SAME one the row-stage 4:4:4 high-bit
//!   path uses, then fan to rgba / hsv.
//!
//! The staged Y / U / V are HOST-NATIVE (the wire row is de-interleaved to
//! host-native u16 via [`deinterleave_y_high_bit_masked`] into private
//! source-width scratch BEFORE the [`AreaStream`] bins it — matching the
//! row-stage path), so every convert kernel runs with
//! `BE = HOST_NATIVE_BE` (= `from_ne`, a no-op load on every host)
//! regardless of the source wire endianness.

use super::super::{
  GeometryOverflow, HsvFrameMut, MixedSinkerError, deinterleave_y_high_bit_masked,
  frozen_outputs_check,
};
use crate::{
  ColorMatrix,
  resample::{AreaStream, PlanGeometry, ResampleError, ResamplePlan, try_box, try_zeroed},
  row::{
    expand_rgb_to_rgba_row, expand_rgb_u16_to_rgba_u16_row, rgb_to_hsv_row,
    yuv444p9_to_hsv_row_endian, yuv444p9_to_rgb_row_endian, yuv444p9_to_rgb_u16_row_endian,
    yuv444p10_to_hsv_row_endian, yuv444p10_to_rgb_row_endian, yuv444p10_to_rgb_u16_row_endian,
    yuv444p12_to_hsv_row_endian, yuv444p12_to_rgb_row_endian, yuv444p12_to_rgb_u16_row_endian,
    yuv444p14_to_hsv_row_endian, yuv444p14_to_rgb_row_endian, yuv444p14_to_rgb_u16_row_endian,
    yuv444p16_to_hsv_row_endian, yuv444p16_to_rgb_row_endian, yuv444p16_to_rgb_u16_row_endian,
  },
};

// The staged Y / U / V the `AreaStream` produces are HOST-NATIVE u16 — the
// wire was decoded to native by `deinterleave_y_high_bit_masked` BEFORE binning —
// so every `*_to_*_row_endian` convert below must read them with
// `BE = HOST_NATIVE_BE` to keep the kernel's `from_le` / `from_be` load a
// no-op on EVERY host. A hard-coded `false` (`from_le`) would byte-swap the
// already-native value on a big-endian target, corrupting default-on native
// color. Mirrors `planar_gbr_f16`'s `HOST_NATIVE_BE`.
const HOST_NATIVE_BE: bool = cfg!(target_endian = "big");

#[cfg(all(test, feature = "std", feature = "yuv-planar"))]
std::thread_local! {
  static FORCE_NATIVE_U16_ALLOC_FAILURE: core::cell::Cell<bool> =
    const { core::cell::Cell::new(false) };
}

/// Arms the source-scratch allocation failpoint for the **next**
/// output-bearing high-bit native row on the current thread. The flag is
/// consumed (take-on-read) by the first fallible source-scratch grow that
/// row reaches, so it fires exactly once and cannot leak into a later
/// test. Test-only — mirrors `arm_deinterleave_alloc_failure` for the
/// semi-planar 8-bit native tier.
#[cfg(all(test, feature = "std", feature = "yuv-planar"))]
pub(crate) fn arm_native_u16_alloc_failure() {
  FORCE_NATIVE_U16_ALLOC_FAILURE.with(|f| f.set(true));
}

#[cfg(all(test, feature = "std", feature = "yuv-planar"))]
std::thread_local! {
  static FORCE_NATIVE_U16_CHROMA_FAILURE: core::cell::Cell<bool> =
    const { core::cell::Cell::new(false) };
  /// Fires at the u8 output-width RGB-scratch grow, a pre-feed step that runs
  /// BEFORE the delegate commits (installs the rebuilt join + freezes the output
  /// set) — the pre-commit failure path the transactional delegate must leave
  /// state-atomic (cached join + `resample_outputs` untouched, retryable).
  static FORCE_NATIVE_U16_RGB_SCRATCH_FAILURE: core::cell::Cell<bool> =
    const { core::cell::Cell::new(false) };
  /// Fires at the native-depth u16 colour-scratch (`rgb_scratch_u16`) grow, which
  /// runs AFTER the u8 colour scratch — a LATER pre-feed grow, so arming it proves
  /// a late failure still leaves the cached join, the `resample_outputs` freeze,
  /// and the output frame untouched and the row retryable.
  static FORCE_NATIVE_U16_RGB_U16_SCRATCH_FAILURE: core::cell::Cell<bool> =
    const { core::cell::Cell::new(false) };
}

/// Arms a failpoint that fires when (and only when) the native join PLANS its
/// chroma grid — which happens exactly when colour output is requested, so a
/// colour-capability rebuild reaches it. A luma-only sink never does, so an armed
/// flag survives a luma-only row unconsumed and is taken by the first colour row.
/// Test-only — the 4:2:0 high-bit sibling of `arm_planar_hb_native_chroma_failure`.
#[cfg(all(test, feature = "std", feature = "yuv-planar"))]
pub(crate) fn arm_native_u16_chroma_failure() {
  FORCE_NATIVE_U16_CHROMA_FAILURE.with(|f| f.set(true));
}

/// Arms [`FORCE_NATIVE_U16_RGB_SCRATCH_FAILURE`] for the NEXT native output-width
/// u8 RGB-scratch grow on this thread (a pre-feed step that runs BEFORE the
/// delegate commits the rebuilt join + output-set freeze), consumed on read so it
/// fires exactly once. Test-only.
#[cfg(all(test, feature = "std", feature = "yuv-planar"))]
pub(crate) fn arm_native_u16_rgb_scratch_failure() {
  FORCE_NATIVE_U16_RGB_SCRATCH_FAILURE.with(|f| f.set(true));
}

/// Arms [`FORCE_NATIVE_U16_RGB_U16_SCRATCH_FAILURE`] for the NEXT native-depth
/// u16 colour-scratch grow on this thread — a LATER pre-feed grow than the u8
/// scratch, so a test arming it proves a late pre-feed failure still leaves the
/// cached join, the `resample_outputs` freeze, and the output frame untouched and
/// the row retryable. Consumed on read so it fires exactly once. Test-only.
#[cfg(all(test, feature = "std", feature = "yuv-planar"))]
pub(crate) fn arm_native_u16_rgb_u16_scratch_failure() {
  FORCE_NATIVE_U16_RGB_U16_SCRATCH_FAILURE.with(|f| f.set(true));
}

/// Native decimation join for the high-bit planar 4:2:0 family — the
/// `u16` twin of [`NativeYuv420`](crate::sinker::mixed::planar_8bit::NativeYuv420).
/// Y streams on the frame grid, U / V on the chroma grid (half width,
/// ceil-half height) via the SAME [`area_chroma_420`] plan (element-
/// agnostic geometry — the 4:2:0 chroma double-count trap is already
/// solved there), every plane binned to FULL output resolution. Each
/// plane's in-order emissions stage into a two-slot ring (`out_y & 1`);
/// the moment all participating planes hold an output row it finalizes
/// through the 4:4:4 kernels at output width — so no output-geometry
/// alignment constraint ever applies. A plane may lead another by at most
/// one source row (the grids are within a factor of two), which the two
/// slots absorb.
pub(crate) struct NativeYuv420U16 {
  y: AreaStream<u16>,
  /// Source-width host-native Y de-interleave scratch (the wire Y plane
  /// normalized via [`deinterleave_y_high_bit_masked`] before [`Self::y`] bins
  /// it). Lazily grown to `src_w` `u16` on the first output-bearing row;
  /// empty otherwise.
  y_src: std::vec::Vec<u16>,
  /// Two-slot staging ring, `2 * out_w` (slot = `out_y & 1`).
  y_stage: std::vec::Vec<u16>,
  /// Chroma half of the join — absent for luma-only sinks, which
  /// therefore never read the chroma planes (the documented fast path).
  /// Decided at creation: the frozen-output contract makes the attached
  /// set frame-constant.
  chroma: Option<NativeChromaU16>,
  /// `staged[plane][slot]` — plane 0 = Y, 1 = U, 2 = V.
  staged: [[bool; 2]; 3],
  /// Next output row to finalize.
  next_emit: usize,
  /// Whether the cached chroma plan was built with a non-zero chroma phase
  /// (RFC #238 centered siting). The join is built once and only `reset`
  /// between frames, so a reused sink whose `chroma_location` moved to a
  /// different phase must REBUILD it — the caller compares this against the
  /// current siting and drops the join on a mismatch. `false` for a luma-only
  /// join (no chroma plan) and for every co-sited / unphased layout.
  chroma_centered: bool,
  /// RFC #238 S6d: whether the cached chroma plan folded the `Bottom` VERTICAL
  /// phase (`v = 1`). `Center` and `Bottom` share `chroma_centered = true`, so
  /// the horizontal flag alone cannot tell them apart; the native call site
  /// compares this too and rebuilds the join when the vertical siting moves.
  /// `false` for a luma-only join and every vertical-co-sited layout.
  chroma_bottom: bool,
}

/// Chroma-grid streams, source de-interleave scratch, and staging of
/// [`NativeYuv420U16`].
struct NativeChromaU16 {
  u: AreaStream<u16>,
  v: AreaStream<u16>,
  /// Source-width (chroma-width) host-native U / V de-interleave scratch.
  /// Lazily grown to `chroma_w` `u16` each on the first chroma-bearing
  /// output row; empty otherwise.
  u_src: std::vec::Vec<u16>,
  v_src: std::vec::Vec<u16>,
  u_stage: std::vec::Vec<u16>,
  v_stage: std::vec::Vec<u16>,
}

impl NativeYuv420U16 {
  fn new(
    plan: &ResamplePlan,
    build_chroma_plan: impl FnOnce() -> Result<ResamplePlan, ResampleError>,
    w: usize,
    h: usize,
    need_color: bool,
  ) -> Result<Self, ResampleError> {
    // The native high-bit 4:2:0 join is integer area-only; reject a filter
    // plan before building any plane's area stream.
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
    let chroma = if need_color {
      #[cfg(all(test, feature = "std", feature = "yuv-planar"))]
      if FORCE_NATIVE_U16_CHROMA_FAILURE.with(|f| f.take()) {
        return Err(ResampleError::AllocationFailed(PlanGeometry::new(
          w,
          h,
          plan.out_w(),
          plan.out_h(),
        )));
      }
      // Vertical chroma weighting runs in the LUMA domain so an odd trailing
      // luma row weights its chroma row by half; the plan's stored dims are the
      // per-plane denominators (scaled on the H axis for the centered fold, and
      // on the V axis for the RFC #238 S6d `Bottom` fold).
      let cplan = build_chroma_plan()?;
      chroma_centered = cplan.has_chroma_phase();
      chroma_bottom = cplan.has_chroma_v_phase();
      Some(NativeChromaU16 {
        u: AreaStream::new(cplan.h(), cplan.v(), cplan.src_w(), cplan.src_h(), 1)?,
        v: AreaStream::new(cplan.h(), cplan.v(), cplan.src_w(), cplan.src_h(), 1)?,
        u_src: std::vec::Vec::new(),
        v_src: std::vec::Vec::new(),
        u_stage: try_zeroed(stage_len).map_err(alloc)?,
        v_stage: try_zeroed(stage_len).map_err(alloc)?,
      })
    } else {
      None
    };
    Ok(Self {
      y,
      y_src: std::vec::Vec::new(),
      y_stage: try_zeroed(stage_len).map_err(alloc)?,
      chroma,
      staged: [[false; 2]; 3],
      next_emit: 0,
      chroma_centered,
      chroma_bottom,
    })
  }

  pub(crate) fn reset(&mut self) {
    self.y.reset();
    if let Some(chroma) = self.chroma.as_mut() {
      chroma.u.reset();
      chroma.v.reset();
    }
    self.staged = [[false; 2]; 3];
    self.next_emit = 0;
  }

  /// Next Y source row this join expects — the per-row sequence counter
  /// (the native path bins Y on every output-bearing row, luma implicit).
  pub(crate) fn next_y(&self) -> usize {
    self.y.next_y()
  }

  /// The cached chroma plan's centered-phase flag (RFC #238), `Some` only when
  /// a chroma half is present — a luma-only join carries no phase, so its
  /// siting is irrelevant. The native call site drops-and-rebuilds a reused
  /// join whose horizontal chroma phase no longer matches the frame's siting.
  pub(crate) fn chroma_phase_centered(&self) -> Option<bool> {
    self.chroma.as_ref().map(|_| self.chroma_centered)
  }

  /// The cached chroma plan's `Bottom` VERTICAL-phase flag (RFC #238 S6d),
  /// `Some` only when a chroma half is present (like
  /// [`Self::chroma_phase_centered`]). The native call site drops-and-rebuilds a
  /// reused join whose vertical chroma phase no longer matches the frame's
  /// siting — `Center` and `Bottom` share the centered flag, so this
  /// distinguishes them.
  pub(crate) fn chroma_bottom(&self) -> Option<bool> {
    self.chroma.as_ref().map(|_| self.chroma_bottom)
  }

  /// Sequencing preflight across all three plane streams — checked before
  /// any plane is fed so a violating call mutates nothing. Chroma rows
  /// advance once per source-row pair, so their expected counter is the
  /// ceiling half of the source row.
  fn check_sequence(&self, idx: usize) -> Result<(), MixedSinkerError> {
    if self.y.next_y() != idx {
      return Err(MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(
        crate::resample::OutOfSequenceRow::new(self.y.next_y(), idx),
      )));
    }
    if let Some(chroma) = self.chroma.as_ref() {
      let chroma_expected = idx.div_ceil(2);
      for stream in [&chroma.u, &chroma.v] {
        if stream.next_y() != chroma_expected {
          return Err(MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(
            crate::resample::OutOfSequenceRow::new(stream.next_y().saturating_mul(2), idx),
          )));
        }
      }
    }
    Ok(())
  }
}

/// The COMPLETE compare-only 4-point pre-feed rejection preflight for the high-bit
/// planar 4:2:0 native fast tier — the no-output short-circuit, first-row
/// out-of-sequence check, output-set compare, and post-compare sequence check via
/// [`native_preflight_core_check_only`](crate::sinker::mixed::planar_8bit::native_preflight_core_check_only),
/// keyed on the u16-join-typed expected row ([`NativeYuv420U16::next_y`]) and
/// threading the native-depth u16 colour outputs (`rgb_u16` / `rgba_u16`) into the
/// compare (the 8-bit twin treats those as absent).
///
/// **NO commit** — it COMPARES the frozen output set against a fresh snapshot
/// instead of storing it. Every native 4:2:0 caller runs it: the semi-planar 4:2:0
/// P-format wrapper ([`p0xx_process_native`](super::p0xx)) runs it ahead of its own
/// fallible de-pack scratch grow, and [`yuv420p16_process_native`] itself runs it
/// (the double-run is idempotent). [`yuv420p16_process_native`] builds its join into
/// a LOCAL and grows its colour / source scratch AFTER this passes, then performs the
/// SINGLE output-set freeze once every pre-feed allocation (the wrapper's de-pack
/// grows AND the delegate's colour / source grows) has succeeded — so a first-row /
/// rebuild-row / later-grow failure leaves the cached join, the `resample_outputs`
/// freeze, and the output frame untouched and the row retryable (the persistent
/// scratch may keep grown capacity — join / freeze / output-atomic, not
/// memory-exact). The 4:2:0 high-bit sibling of
/// [`native_planar_hb_preflight_check_only`](crate::sinker::mixed::planar_high_bit_native::native_planar_hb_preflight_check_only).
#[allow(clippy::too_many_arguments)]
pub(crate) fn yuv420p16_native_preflight_check_only(
  native_420_u16: &Option<std::boxed::Box<NativeYuv420U16>>,
  resample_outputs: &Option<super::super::FrozenOutputs>,
  rgb: &Option<&mut [u8]>,
  rgba: &Option<&mut [u8]>,
  rgb_u16: &Option<&mut [u16]>,
  rgba_u16: &Option<&mut [u16]>,
  luma: &Option<&mut [u8]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  idx: usize,
  need_luma: bool,
  need_color: bool,
) -> Result<bool, MixedSinkerError> {
  super::super::planar_8bit::native_preflight_core_check_only(
    native_420_u16.as_ref().map_or(0, |j| j.next_y()),
    resample_outputs,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    luma,
    // The high-bit planar 4:2:0 family exposes no `luma_u16` output.
    &None,
    hsv,
    idx,
    need_luma,
    need_color,
  )
}

/// Grows a source-de-interleave scratch to `len` `u16` under the planner's
/// recoverable-allocation contract, with the test-only allocation
/// failpoint on the FIRST such grow of an output-bearing row (mirrors the
/// semi-planar 8-bit de-interleave failpoint). Runs after the preflight
/// and after the join is built, so a rejected row never reaches it.
#[cfg_attr(not(tarpaulin), inline(always))]
fn grow_src_scratch(
  scratch: &mut std::vec::Vec<u16>,
  len: usize,
  w: usize,
  h: usize,
  plan: &ResamplePlan,
) -> Result<(), MixedSinkerError> {
  if scratch.len() < len {
    #[cfg(all(test, feature = "std", feature = "yuv-planar"))]
    if FORCE_NATIVE_U16_ALLOC_FAILURE.with(|f| f.take()) {
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

/// Native-tier path for the high-bit planar 4:2:0 family. Const-generic
/// over `BITS` (one body, the call sites 9 / 10 / 12 / 14 / 16) and `BE` (the
/// source wire endianness). Preflight-transactional (mirrors the 8-bit 4:2:0
/// twin [`yuv420p_process_native`](crate::sinker::mixed::planar_8bit::yuv420p_process_native)
/// and the non-4:2:0 high-bit twin
/// [`yuv_planar16_process_native`](crate::sinker::mixed::planar_high_bit_native::yuv_planar16_process_native)):
/// the compare-only pre-feed preflight, the join built into a LOCAL, then the
/// colour and source scratch sizing — and only once every pre-feed allocation
/// has succeeded does it COMMIT (install the rebuilt join, freeze the output
/// set), sequence-check, and feed. So on ANY pre-feed allocation failure the
/// cached join, the `resample_outputs` freeze, and the output frame stay
/// untouched and the row is retryable (a deterministic typed error, never a
/// committed-then-corrupt state); the persistent reused scratch buffers
/// (`rgb_scratch` / `rgb_scratch_u16` / the join's source-plane scratch) MAY keep
/// already-grown capacity, which a retry reuses correctly and the sink frees on
/// drop — join / freeze / output-atomic, not memory-exact. Nothing is fallible
/// after the first feed.
#[allow(clippy::too_many_arguments)]
pub(crate) fn yuv420p16_process_native<const BITS: u32, const BE: bool>(
  plan: &ResamplePlan,
  native_420_u16: &mut Option<std::boxed::Box<NativeYuv420U16>>,
  resample_outputs: &mut Option<super::super::FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  rgb_u16: &mut Option<&mut [u16]>,
  rgba_u16: &mut Option<&mut [u16]>,
  luma: &mut Option<&mut [u8]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  rgb_scratch: &mut std::vec::Vec<u8>,
  rgb_scratch_u16: &mut std::vec::Vec<u16>,
  y_row: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  matrix: ColorMatrix,
  full_range: bool,
  idx: usize,
  w: usize,
  h: usize,
  // Lazily builds the 4:2:0 chroma grid against the SAME output geometry as
  // `plan`; the per-frame caller folds the RFC #238 horizontal chroma phase into
  // it ([`ResamplePlan::area_chroma_420`]). At phase 0 it is byte-identical to
  // the pre-closure co-sited grid, and it is invoked ONLY when colour is needed.
  build_chroma_plan: impl FnOnce() -> Result<ResamplePlan, ResampleError>,
  use_simd: bool,
) -> Result<(), MixedSinkerError> {
  const {
    assert!(
      BITS > 8 && BITS <= 16,
      "BITS must be in (8, 16] for high-bit planar 4:2:0 YUV"
    )
  };
  let ow = plan.out_w();
  let need_luma = luma.is_some();
  let need_color_u8 = rgb.is_some() || rgba.is_some() || hsv.is_some();
  let need_color_u16 = rgb_u16.is_some() || rgba_u16.is_some();
  let need_color = need_color_u8 || need_color_u16;
  // HSV-without-RGB-or-RGBA emits HSV straight from the binned
  // output-width host-native Y/U/V via `yuv444p{BITS}_to_hsv_row_endian`
  // (the binned chroma is full-width at output resolution → the 4:4:4
  // kernel family) — no output-width u8 RGB scratch is staged. Any u8
  // RGB / RGBA output (or any u16 colour) keeps the convert-once-then-
  // derive path. The u16 colour scratch is independent and unaffected.
  let want_hsv_direct = hsv.is_some() && rgb.is_none() && rgba.is_none();

  // Compare-only pre-feed rejection preflight (no-output short-circuit,
  // first-row out-of-sequence, output-set compare, post-compare sequence)
  // ahead of any fallible allocation — the check-only preflight
  // [`yuv420p16_native_preflight_check_only`]. It does NOT commit the output-set
  // freeze; that commit is deferred below until every pre-feed allocation has
  // succeeded, so a first-row / rebuild-row failure leaves the cached join and
  // `resample_outputs` untouched. The direct planar callers reach this as their
  // only preflight; the semi-planar P-format wrapper runs the identical compare
  // first and the double-run is idempotent. `Ok(false)` is the no-output no-op.
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

  // A fresh join is needed on a first-ever build (`None`) or a colour-capability
  // change (the cached join's chroma half no longer matches `need_color` —
  // outputs attached / detached since the previous frame; the frozen check pins
  // them WITHIN a frame, not across frames). Build the replacement into a LOCAL
  // and do NOT clear the field yet: the cached join stays intact until every
  // pre-feed allocation has succeeded, so a build / box refusal leaves it, the
  // `resample_outputs` freeze, and the output frame untouched and the row
  // retryable.
  let rebuild = native_420_u16.is_none()
    || native_420_u16
      .as_ref()
      .is_some_and(|join| join.chroma.is_some() != need_color);
  let mut new_join = if rebuild {
    // Build the join (its de-interleave / stage scratch allocates recoverably)
    // and box it recoverably (`try_box`, not `Box::new` — the latter aborts on
    // OOM).
    let join = NativeYuv420U16::new(plan, build_chroma_plan, w, h, need_color)?;
    Some(try_box(join).map_err(|_| {
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

  // Colour OUTPUT scratch at output width (one binned row converts here
  // before fanning to the caller buffers). Both grows are fallible and
  // precede the commit, so a failure here (e.g. the u16 grow after the u8 grow
  // already succeeded) leaves the join / freeze / output untouched and the row
  // retryable; the already-grown persistent scratch simply keeps its capacity for
  // the retry. The u8 RGB scratch is skipped entirely for the HSV-direct case —
  // the emit loop converts the binned Y/U/V straight to HSV.
  if need_color_u8 && !want_hsv_direct {
    let row_bytes =
      ow.checked_mul(3)
        .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
          ow,
          plan.out_h(),
          3,
        )))?;
    if rgb_scratch.len() < row_bytes {
      #[cfg(all(test, feature = "std", feature = "yuv-planar"))]
      if FORCE_NATIVE_U16_RGB_SCRATCH_FAILURE.with(|f| f.take()) {
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
  if need_color_u16 {
    let row_elems =
      ow.checked_mul(3)
        .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
          ow,
          plan.out_h(),
          3,
        )))?;
    if rgb_scratch_u16.len() < row_elems {
      #[cfg(all(test, feature = "std", feature = "yuv-planar"))]
      if FORCE_NATIVE_U16_RGB_U16_SCRATCH_FAILURE.with(|f| f.take()) {
        return Err(MixedSinkerError::Resample(ResampleError::AllocationFailed(
          PlanGeometry::new(w, h, plan.out_w(), plan.out_h()),
        )));
      }
      rgb_scratch_u16
        .try_reserve_exact(row_elems - rgb_scratch_u16.len())
        .map_err(|_| {
          MixedSinkerError::Resample(ResampleError::AllocationFailed(PlanGeometry::new(
            w,
            h,
            plan.out_w(),
            plan.out_h(),
          )))
        })?;
      rgb_scratch_u16.resize(row_elems, 0);
    }
  }

  // Source-width de-interleave scratch (wire → host-native), grown after the
  // colour-output scratch so the FIRST grow here carries the test failpoint.
  // Grown on the rebuilt LOCAL join if rebuilding, else the cached join — all
  // fallible, all BEFORE the commit, so a failure at any of them leaves the join /
  // freeze / output untouched and the row retryable (an earlier source scratch
  // keeps its grown capacity). The chroma scratch is grown only on chroma-bearing
  // rows, exactly where the join reads chroma.
  let cw = w / 2;
  let feed_chroma;
  {
    let target: &mut NativeYuv420U16 = match new_join.as_deref_mut() {
      Some(join) => join,
      None => native_420_u16
        .as_deref_mut()
        .expect("reuse implies a cached join"),
    };
    grow_src_scratch(&mut target.y_src, w, w, h, plan)?;
    feed_chroma = target.chroma.is_some() && idx.is_multiple_of(2);
    if feed_chroma {
      // Split-borrow: grow the two chroma scratches without holding `target`
      // immutably across the call.
      let chroma = target.chroma.as_mut().expect("feed_chroma implies Some");
      grow_src_scratch(&mut chroma.u_src, cw, w, h, plan)?;
      grow_src_scratch(&mut chroma.v_src, cw, w, h, plan)?;
    }
  }

  // Every pre-feed allocation succeeded — COMMIT: install the rebuilt join
  // (dropping the old) then freeze the output set. A build only reaches here on
  // the first output-bearing row (its `resample_outputs` is still `None`), so the
  // freeze just stores the snapshot; a reuse row builds nothing and re-freezes
  // nothing new. The native-depth u16 colour outputs are frozen as present (the
  // 8-bit twin freezes them absent).
  if let Some(boxed) = new_join {
    *native_420_u16 = Some(boxed);
  }
  frozen_outputs_check(
    resample_outputs,
    luma,
    // The high-bit planar 4:2:0 family exposes no `luma_u16` output.
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
    idx,
  )?;
  let join = native_420_u16
    .as_mut()
    .expect("built or reused in the preflight");
  join.check_sequence(idx)?;

  // De-interleave the wire planes into host-native scratch. Everything
  // past this point is infallible.
  deinterleave_y_high_bit_masked::<BITS, BE>(y_row, &mut join.y_src, w);
  if feed_chroma {
    let chroma = join.chroma.as_mut().expect("feed_chroma implies Some");
    // Mask BEFORE the AreaStream bins these: the native tier averages the
    // source chroma, so the colour kernel's per-pixel extract_hb would mask
    // only the binned result. `avg(dirty) & mask` can differ from
    // `avg(dirty & mask)`, so dirty bits must be cleared per source sample.
    deinterleave_y_high_bit_masked::<BITS, BE>(u_half, &mut chroma.u_src, cw);
    deinterleave_y_high_bit_masked::<BITS, BE>(v_half, &mut chroma.v_src, cw);
  }

  // Feed the planes into their streams. The Y plane bins every row; the
  // chroma planes only on even source rows (`cidx = idx / 2`).
  let NativeYuv420U16 {
    y,
    y_src,
    y_stage,
    chroma,
    staged,
    next_emit,
    // Cache keys read only at build / the native call site; the feed loop
    // ignores them.
    chroma_centered: _,
    chroma_bottom: _,
  } = &mut **join;
  y.feed_row(idx, &y_src[..w], use_simd, |oy, out_row| {
    let slot = oy & 1;
    y_stage[slot * ow..slot * ow + ow].copy_from_slice(out_row);
    staged[0][slot] = true;
  })?;
  if let Some(c) = chroma.as_mut()
    && idx.is_multiple_of(2)
  {
    let cidx = idx / 2;
    let NativeChromaU16 {
      u,
      v,
      u_src,
      v_src,
      u_stage,
      v_stage,
    } = c;
    u.feed_row(cidx, &u_src[..cw], use_simd, |oy, out_row| {
      let slot = oy & 1;
      u_stage[slot * ow..slot * ow + ow].copy_from_slice(out_row);
      staged[1][slot] = true;
    })?;
    v.feed_row(cidx, &v_src[..cw], use_simd, |oy, out_row| {
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
      // Clamp to the native max before narrowing: an overrange binned Y (from
      // out-of-gamut input whose high bits exceed BITS) must saturate, not
      // wrap modulo 256 — the row-stage luma path narrows the clamped u16.
      let native_max: u16 = ((1u32 << BITS) - 1) as u16;
      for (dst, &src) in buf[oy * ow..(oy + 1) * ow].iter_mut().zip(y_out) {
        *dst = (src.min(native_max) >> (BITS - 8)) as u8;
      }
    }

    if let Some(c) = chroma.as_ref() {
      let u_row = &c.u_stage[slot * ow..slot * ow + ow];
      let v_row = &c.v_stage[slot * ow..slot * ow + ow];

      // Native-depth u16 colour — its OWN independent kernel, never a
      // narrowing of the u8 colour (the #37 contract). The staged Y / U / V
      // are host-native, so `BE = HOST_NATIVE_BE`.
      if need_color_u16 {
        let out_rgb = &mut rgb_scratch_u16[..ow * 3];
        emit_rgb_u16::<BITS>(
          y_out, u_row, v_row, out_rgb, ow, matrix, full_range, use_simd,
        );
        if let Some(buf) = rgb_u16.as_deref_mut() {
          buf[oy * 3 * ow..(oy + 1) * 3 * ow].copy_from_slice(out_rgb);
        }
        if let Some(buf) = rgba_u16.as_deref_mut() {
          expand_rgb_u16_to_rgba_u16_row::<BITS>(
            out_rgb,
            &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow],
            ow,
          );
        }
      }

      // u8 colour — the u16-INPUT → u8-OUTPUT 4:4:4 kernel (the same one
      // the row-stage 4:4:4 high-bit path uses), independent of the u16
      // colour above.
      if need_color_u8 {
        if want_hsv_direct {
          // RGB-free: convert the binned output-width host-native Y/U/V
          // straight to HSV via the 4:4:4 HSV kernel (the binned chroma
          // is full-width), no output-width RGB scratch.
          let hsv = hsv.as_mut().expect("want_hsv_direct implies hsv attached");
          let (hp, sp, vp) = hsv.hsv();
          emit_hsv_u8::<BITS>(
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
          emit_rgb_u8::<BITS>(
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
    }
    staged[0][slot] = false;
    staged[1][slot] = false;
    staged[2][slot] = false;
    *next_emit += 1;
  }
  Ok(())
}

/// u16-output 4:4:4 conversion at output width — the staged Y / U / V are
/// host-native, so `big_endian = HOST_NATIVE_BE`. Dispatches on `BITS` to the
/// per-format wrapper, which pins the generic 4:4:4 u16 kernel to a
/// supported `BITS` (9/10/12/14 → the `i32` `yuv_444p_n_to_rgb_u16_row`
/// family; 16 → the dedicated `i64`-chroma `yuv444p16_to_rgb_u16_row`).
///
/// A runtime `if BITS == 16 { dedicated } else { generic::<BITS> }` will
/// not do: a const-generic `if` still MONOMORPHIZES the `else` arm at
/// `BITS = 16`, where the generic kernel const-asserts against 16 (the
/// `(1 << 16) - 1 as i16` wrap footgun). A `match` over the four CONCRETE
/// (non-generic) per-format wrappers calls only the live one and never
/// instantiates the invalid `::<16>` form.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
fn emit_rgb_u16<const BITS: u32>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  match BITS {
    9 => yuv444p9_to_rgb_u16_row_endian(
      y,
      u,
      v,
      rgb_out,
      width,
      matrix,
      full_range,
      use_simd,
      HOST_NATIVE_BE,
    ),
    10 => yuv444p10_to_rgb_u16_row_endian(
      y,
      u,
      v,
      rgb_out,
      width,
      matrix,
      full_range,
      use_simd,
      HOST_NATIVE_BE,
    ),
    12 => yuv444p12_to_rgb_u16_row_endian(
      y,
      u,
      v,
      rgb_out,
      width,
      matrix,
      full_range,
      use_simd,
      HOST_NATIVE_BE,
    ),
    14 => yuv444p14_to_rgb_u16_row_endian(
      y,
      u,
      v,
      rgb_out,
      width,
      matrix,
      full_range,
      use_simd,
      HOST_NATIVE_BE,
    ),
    16 => yuv444p16_to_rgb_u16_row_endian(
      y,
      u,
      v,
      rgb_out,
      width,
      matrix,
      full_range,
      use_simd,
      HOST_NATIVE_BE,
    ),
    _ => unreachable!("BITS pinned to 9/10/12/14/16 by the call sites"),
  }
}

/// u8-output 4:4:4 conversion at output width (the u16-INPUT → u8-OUTPUT
/// kernel — the SAME one the row-stage 4:4:4 high-bit path uses). Same
/// per-format dispatch + `big_endian = HOST_NATIVE_BE` rationale as
/// [`emit_rgb_u16`].
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
fn emit_rgb_u8<const BITS: u32>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  match BITS {
    9 => yuv444p9_to_rgb_row_endian(
      y,
      u,
      v,
      rgb_out,
      width,
      matrix,
      full_range,
      use_simd,
      HOST_NATIVE_BE,
    ),
    10 => yuv444p10_to_rgb_row_endian(
      y,
      u,
      v,
      rgb_out,
      width,
      matrix,
      full_range,
      use_simd,
      HOST_NATIVE_BE,
    ),
    12 => yuv444p12_to_rgb_row_endian(
      y,
      u,
      v,
      rgb_out,
      width,
      matrix,
      full_range,
      use_simd,
      HOST_NATIVE_BE,
    ),
    14 => yuv444p14_to_rgb_row_endian(
      y,
      u,
      v,
      rgb_out,
      width,
      matrix,
      full_range,
      use_simd,
      HOST_NATIVE_BE,
    ),
    16 => yuv444p16_to_rgb_row_endian(
      y,
      u,
      v,
      rgb_out,
      width,
      matrix,
      full_range,
      use_simd,
      HOST_NATIVE_BE,
    ),
    _ => unreachable!("BITS pinned to 9/10/12/14/16 by the call sites"),
  }
}

/// Direct u16-INPUT → **u8 HSV**-OUTPUT 4:4:4 conversion at output width
/// for the HSV-without-RGB native fast tier — the staged Y / U / V are
/// host-native, so `big_endian = HOST_NATIVE_BE`. The same kernel family
/// as [`emit_rgb_u8`], fused with the HSV quantizer (no output-width RGB
/// scratch). Dispatches on `BITS` to the per-format
/// `yuv444p{BITS}_to_hsv_row_endian` wrapper.
#[allow(clippy::too_many_arguments)]
fn emit_hsv_u8<const BITS: u32>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  match BITS {
    9 => yuv444p9_to_hsv_row_endian(
      y,
      u,
      v,
      h_out,
      s_out,
      v_out,
      width,
      matrix,
      full_range,
      use_simd,
      HOST_NATIVE_BE,
    ),
    10 => yuv444p10_to_hsv_row_endian(
      y,
      u,
      v,
      h_out,
      s_out,
      v_out,
      width,
      matrix,
      full_range,
      use_simd,
      HOST_NATIVE_BE,
    ),
    12 => yuv444p12_to_hsv_row_endian(
      y,
      u,
      v,
      h_out,
      s_out,
      v_out,
      width,
      matrix,
      full_range,
      use_simd,
      HOST_NATIVE_BE,
    ),
    14 => yuv444p14_to_hsv_row_endian(
      y,
      u,
      v,
      h_out,
      s_out,
      v_out,
      width,
      matrix,
      full_range,
      use_simd,
      HOST_NATIVE_BE,
    ),
    16 => yuv444p16_to_hsv_row_endian(
      y,
      u,
      v,
      h_out,
      s_out,
      v_out,
      width,
      matrix,
      full_range,
      use_simd,
      HOST_NATIVE_BE,
    ),
    _ => unreachable!("BITS pinned to 9/10/12/14/16 by the call sites"),
  }
}
