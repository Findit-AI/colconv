//! Format-agnostic row-stage planar-YUV resample helper shared by the
//! 8-bit planar family ([`super::planar_8bit`]) and the semi-planar
//! family ([`super::semi_planar_8bit`]). [`planar_dual_resample`] bins the
//! Y plane for luma and bins a caller-converted source-width RGB row for
//! colour, so it references no source-format kernel — the caller supplies
//! the conversion closure (the planar formats convert their separate
//! planes, the semi-planar formats convert their interleaved chroma row
//! with the matching `nv*` kernel). Both routes are byte-identical to an
//! `Rgb24` area-resample of the identity-converted frame.
//!
//! #263 follow-up — RGB-free YUV-domain HSV-only **area** resample: when
//! ONLY `with_hsv()` is attached (no RGB / RGBA) and the plan is an area
//! plan, [`planar_dual_resample`] no longer stages a source-width RGB row.
//! Instead [`HsvDirectPlanarYuv`] bins Y / U / V on their own grids (Y on
//! the luma grid, U / V decimated per the format's chroma subsampling)
//! straight to OUTPUT resolution, then converts each finalized output row
//! through [`yuv_444_to_hsv_row`] at output width — exactly the native fast
//! tier's HSV-only path
//! ([`yuv_planar_process_native`](super::planar_8bit::yuv_planar_process_native)).
//! The binning and the conversion kernel are therefore identical to the
//! native tier, so the row-stage HSV-only output is **bit-identical** to
//! the native tier's HSV-only output for the same (format, area-resample) —
//! a deliberate behaviour change from the former RGB-domain averaging (the
//! HSV planes used to derive off the binned RGB row). When RGB or RGBA is
//! ALSO attached the cheap RGB-staged path is unchanged (it bins one RGB
//! row and derives HSV off it). The 3-stream lockstep mirrors
//! [`NativePlanarYuv`](super::planar_8bit::NativePlanarYuv) verbatim — Y and
//! the two chroma streams finalising the same output row through a two-slot
//! ring.
//!
//! SCOPE: this RGB-free HSV-only area path is wired for the **planar** 8-bit
//! families ([`super::planar_8bit`], `yuv-planar`-gated — the join's chroma
//! plans use the `yuv-planar`-gated `area_chroma_*` builders). Two cases stay
//! RGB-staged for now (correct, but not yet RGB-free), each a documented
//! follow-up:
//!
//! - The **filter** twin ([`planar_dual_filter_resample`]): a filter window
//!   leads by more than one output row, so a filter HSV-direct join needs the
//!   full-output-plane buffers + per-plane cursors the
//!   [`BicublinYuv420`](super::planar_8bit::BicublinYuv420) join uses (not the
//!   area join's two-slot ring) — a materially different stream shape.
//! - The **semi-planar** family (`Nv12` etc.) row-stage, which reaches
//!   [`planar_dual_resample`] with interleaved chroma: it would need a
//!   per-row de-interleave into U / V scratch at each call site plus the
//!   `yuv-planar`-gated 4:2:0 / 4:4:0 chroma-plan builders under
//!   `yuv-semi-planar`.

use core::ops::ControlFlow;

use super::{HsvFrameMut, MixedSinkerError, frozen_outputs_check};
// `ColorMatrix`, `PlanGeometry`, and `try_zeroed` back the `yuv-planar`-only
// RGB-free HSV-only area join below; the rest serve the always-present
// (`yuv-planar` / `yuv-semi-planar`) RGB-staged dual-resample helpers.
#[cfg(feature = "yuv-planar")]
use crate::{
  ColorMatrix,
  resample::{PlanGeometry, try_zeroed},
};
use crate::{
  resample::{AreaStream, ResamplePlan, RowResampler},
  row::*,
};
// `OutOfSequenceRow` / `ResampleError` back the sequence checks + typed resample
// errors of the `yuv-planar` HSV-direct join and the `yuv-packed` dual-resample;
// a semi-planar-solo build reaches neither, so they carry the union gate.
#[cfg(any(feature = "yuv-planar", feature = "yuv-packed"))]
use crate::resample::{OutOfSequenceRow, ResampleError};

/// RGB-free YUV-domain HSV-only **area** join for the shared planar /
/// semi-planar row-stage resample — the colour twin of the native fast
/// tier's HSV-only path, structurally a near-verbatim copy of
/// [`NativePlanarYuv`](super::planar_8bit::NativePlanarYuv). Y streams on
/// the frame grid; U and V on the chroma grid (their per-plane denominators
/// supplied by the caller's chroma plan, fed once per `chroma_vsub` source
/// rows), every plane binned to FULL output resolution. Each plane's
/// in-order emissions land in a `chroma_vsub`-slot ring
/// (`slot = out_y % chroma_vsub`); the moment all three planes hold an
/// output row it finalises through [`yuv_444_to_hsv_row`] at output width —
/// so no alignment constraint ever applies to the output geometry and the
/// result is bit-identical to the native tier's HSV-only output for the same
/// area-resample. When the chroma grid is upsampled vertically (a
/// height-preserving downscale of a `chroma_vsub`-subsampled plane), one
/// chroma feed finalises up to `chroma_vsub` output rows before the matching
/// luma rows arrive, so the ring holds `chroma_vsub` rows (4 for 4:1:0); the
/// lockstep formats (`chroma_vsub == 1`) never lead at all.
///
/// HSV-only by construction: it is built ONLY when `with_hsv()` is the sole
/// colour output, so it never carries an RGB stream or source-width RGB
/// scratch — the structural RGB-free guarantee.
#[cfg(feature = "yuv-planar")]
pub(super) struct HsvDirectPlanarYuv {
  y: AreaStream<u8>,
  /// Staging ring, `chroma_vsub * out_w` (slot = `out_y % chroma_vsub`).
  y_stage: std::vec::Vec<u8>,
  u: AreaStream<u8>,
  v: AreaStream<u8>,
  u_stage: std::vec::Vec<u8>,
  v_stage: std::vec::Vec<u8>,
  /// Vertical chroma subsample factor: 1 for 4:2:2 / 4:4:4 / 4:1:1, 2 for
  /// 4:2:0 / 4:4:0, 4 for 4:1:0. The chroma streams are fed when
  /// `idx % chroma_vsub == 0` at `cidx = idx / chroma_vsub`, and the
  /// sequence check expects `next_y() == idx.div_ceil(chroma_vsub)`.
  chroma_vsub: usize,
  /// `staged[plane][slot]` — plane 0 = Y, 1 = U, 2 = V; the low `chroma_vsub`
  /// slots are live (4 = the 4:1:0 maximum).
  staged: [[bool; 4]; 3],
  /// Next output row to finalise.
  next_emit: usize,
  /// Whether the cached chroma plan was built with a non-zero chroma phase
  /// (RFC #238 centered siting) — the [`NativePlanarYuv`](super::planar_8bit)
  /// `chroma_centered` twin. The join is built once and only `reset` between
  /// frames, so a reused sink whose siting phase changed must REBUILD it;
  /// `begin_frame` reads [`Self::chroma_centered`] to drop a stale join.
  chroma_centered: bool,
  /// RFC #238 S4-B: whether the cached chroma plan folded the `Bottom` vertical
  /// phase (`v = 1`) — the [`NativeYuv420`](super::planar_8bit) `chroma_bottom`
  /// twin. `Center` and `Bottom` share `chroma_centered = true`, so the point-
  /// of-use siting invalidation compares this too to rebuild the join on a
  /// vertical-phase change. `false` for every horizontal-only siting.
  chroma_bottom: bool,
  /// RFC #238 Top: whether the cached chroma plan folded the `Top` vertical
  /// phase (`v = 0`, the FORWARD triangle) — the [`NativeYuv420`](super::planar_8bit)
  /// `chroma_top` twin. `Center` and `Top` share `chroma_centered = true` and
  /// both keep `chroma_bottom = false`, so the point-of-use siting invalidation
  /// compares this too to rebuild the join on a Top ⇄ Center / Top ⇄ Bottom
  /// vertical-phase change. `false` for every non-Top siting.
  chroma_top: bool,
}

#[cfg(feature = "yuv-planar")]
impl HsvDirectPlanarYuv {
  /// `build_chroma_plan` builds the format's chroma grid against the SAME
  /// output geometry as `plan` (the luma plan); `chroma_vsub` is its
  /// vertical cadence. Both are supplied by the per-format caller so this
  /// body stays layout-agnostic — exactly the
  /// [`NativePlanarYuv::new`](super::planar_8bit::NativePlanarYuv) contract,
  /// minus the luma-only fast path (this join exists only to serve colour).
  fn new(
    plan: &ResamplePlan,
    build_chroma_plan: impl FnOnce() -> Result<ResamplePlan, ResampleError>,
    chroma_vsub: usize,
    w: usize,
    h: usize,
  ) -> Result<Self, ResampleError> {
    // Area-only join; a filter plan never reaches it (the filter twin keeps
    // its RGB-staged HSV path), but guard for parity with the native join.
    if plan.kind().is_filter() {
      return Err(plan.unsupported_filter());
    }
    let alloc =
      |_| ResampleError::AllocationFailed(PlanGeometry::new(w, h, plan.out_w(), plan.out_h()));
    let stage_len = plan.out_w().checked_mul(chroma_vsub).ok_or_else(|| {
      ResampleError::Overflow(PlanGeometry::new(w, h, plan.out_w(), plan.out_h()))
    })?;
    let chroma_plan = build_chroma_plan()?;
    // Keys on the HORIZONTAL phase alone so a co-sited-H + `v = 1` siting
    // (`BottomLeft`) is not misread as centered against a reader's `center_sited`
    // — the vertical axis is tracked separately by `chroma_bottom`. Identical to
    // `has_chroma_phase()` for every siting whose only phase is horizontal (the
    // 4:2:2 readers) or whose reader checks `chroma_bottom` alone (4:4:0).
    let chroma_centered = chroma_plan.has_chroma_h_phase();
    let chroma_bottom = chroma_plan.has_chroma_v_phase();
    let chroma_top = chroma_plan.has_chroma_v_top();
    Ok(Self {
      y: AreaStream::new(plan.h(), plan.v(), w, h, 1)?,
      y_stage: try_zeroed(stage_len).map_err(alloc)?,
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
      chroma_vsub,
      staged: [[false; 4]; 3],
      next_emit: 0,
      chroma_centered,
      chroma_bottom,
      chroma_top,
    })
  }

  pub(super) fn reset(&mut self) {
    self.y.reset();
    self.u.reset();
    self.v.reset();
    self.staged = [[false; 4]; 3];
    self.next_emit = 0;
  }

  /// Whether the cached chroma plan carries a non-zero chroma phase — see
  /// [`Self::chroma_centered`](Self#structfield.chroma_centered). Read by the
  /// sink's `begin_frame` to rebuild the join on a siting-phase change.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(super) const fn chroma_centered(&self) -> bool {
    self.chroma_centered
  }

  /// Whether the cached chroma plan folded the `Bottom` vertical phase — see
  /// [`Self::chroma_bottom`](Self#structfield.chroma_bottom). Read by the
  /// `Yuv420p` HSV-only point-of-use siting invalidation, alongside
  /// [`Self::chroma_centered`], to rebuild the join on a vertical-phase change.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(super) const fn chroma_bottom(&self) -> bool {
    self.chroma_bottom
  }

  /// Whether the cached chroma plan folded the `Top` vertical phase — see
  /// [`Self::chroma_top`](Self#structfield.chroma_top). Read by the `Yuv420p`
  /// HSV-only point-of-use siting invalidation, alongside
  /// [`Self::chroma_centered`] / [`Self::chroma_bottom`], to rebuild the join on
  /// a Top vertical-phase change.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(super) const fn chroma_top(&self) -> bool {
    self.chroma_top
  }

  /// Next source row this join expects (0 at a fresh frame). Read by the sink's
  /// point-of-use siting invalidation to fire the phase-rebuild drop ONLY at a
  /// fresh-frame boundary (`next_y() == 0`), never mid-frame.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(super) const fn next_y(&self) -> usize {
    self.y.next_y()
  }

  /// Sequencing preflight across all three plane streams — checked before
  /// any plane is fed so a violating call mutates nothing (the same
  /// contract as [`NativePlanarYuv::check_sequence`](super::planar_8bit::NativePlanarYuv)).
  fn check_sequence(&self, idx: usize) -> Result<(), MixedSinkerError> {
    if self.y.next_y() != idx {
      return Err(MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(
        OutOfSequenceRow::new(self.y.next_y(), idx),
      )));
    }
    let chroma_expected = idx.div_ceil(self.chroma_vsub);
    for stream in [&self.u, &self.v] {
      if stream.next_y() != chroma_expected {
        return Err(MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(
          OutOfSequenceRow::new(stream.next_y().saturating_mul(self.chroma_vsub), idx),
        )));
      }
    }
    Ok(())
  }
}

/// Drives the RGB-free YUV-domain HSV-only area resample: feeds Y / U / V
/// into [`HsvDirectPlanarYuv`] and drains every output row whose three
/// planes are staged, emitting luma (the binned Y, the YUV luma contract)
/// and HSV (through [`yuv_444_to_hsv_row`] at output width) — reproducing
/// the native tier's HSV-only emit
/// ([`yuv_planar_process_native`](super::planar_8bit::yuv_planar_process_native)).
/// Like the native tier, luma derives from the SAME binned Y as the HSV
/// conversion (Y is binned once), so a luma + HSV-only sink stays a single
/// 3-stream join rather than re-binning Y. The join is built (its three
/// area streams + ring scratch allocate recoverably) BEFORE the first feed,
/// so a failure mutates no caller output; the sequence check runs first so
/// an out-of-sequence row is rejected without consuming any stream.
/// Everything past the first feed is infallible.
#[cfg(feature = "yuv-planar")]
#[allow(clippy::too_many_arguments)]
fn hsv_direct_feed_emit(
  join: &mut HsvDirectPlanarYuv,
  hsv: &mut HsvFrameMut<'_>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  y_row: &[u8],
  u_row: &[u8],
  v_row: &[u8],
  matrix: ColorMatrix,
  full_range: bool,
  plan: &ResamplePlan,
  idx: usize,
  use_simd: bool,
) -> Result<(), MixedSinkerError> {
  let ow = plan.out_w();
  join.check_sequence(idx)?;
  let HsvDirectPlanarYuv {
    y,
    y_stage,
    u,
    v,
    u_stage,
    v_stage,
    chroma_vsub,
    staged,
    next_emit,
    // Cache keys read only at build / begin_frame; the feed loop ignores them.
    chroma_centered: _,
    chroma_bottom: _,
    chroma_top: _,
  } = join;
  let cv = *chroma_vsub;
  y.feed_row(idx, y_row, use_simd, |oy, out_row| {
    let slot = oy % cv;
    y_stage[slot * ow..slot * ow + ow].copy_from_slice(out_row);
    staged[0][slot] = true;
  })?;
  if idx.is_multiple_of(cv) {
    let cidx = idx / cv;
    u.feed_row(cidx, u_row, use_simd, |oy, out_row| {
      let slot = oy % cv;
      u_stage[slot * ow..slot * ow + ow].copy_from_slice(out_row);
      staged[1][slot] = true;
    })?;
    v.feed_row(cidx, v_row, use_simd, |oy, out_row| {
      let slot = oy % cv;
      v_stage[slot * ow..slot * ow + ow].copy_from_slice(out_row);
      staged[2][slot] = true;
    })?;
  }

  let (hp, sp, vp) = hsv.hsv();
  while *next_emit < plan.out_h() {
    let slot = *next_emit % cv;
    if !(staged[0][slot] && staged[1][slot] && staged[2][slot]) {
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
    yuv_444_to_hsv_row(
      y_out,
      &u_stage[slot * ow..slot * ow + ow],
      &v_stage[slot * ow..slot * ow + ow],
      &mut hp[oy * ow..(oy + 1) * ow],
      &mut sp[oy * ow..(oy + 1) * ow],
      &mut vp[oy * ow..(oy + 1) * ow],
      ow,
      matrix,
      full_range,
      use_simd,
    );
    staged[0][slot] = false;
    staged[1][slot] = false;
    staged[2][slot] = false;
    *next_emit += 1;
  }
  Ok(())
}

/// The RGB-free YUV-domain HSV-only area path: the complete atomic
/// preflight + the 3-stream feed/drain. Phasing mirrors
/// [`planar_dual_resample`]'s compare-before / commit-after discipline (and
/// the native tier's
/// [`yuv_planar_process_native`](super::planar_8bit::yuv_planar_process_native)):
///
/// 1. The HSV join's Y stream is the canonical per-row sequence counter
///    (Y is binned on every output-bearing row). On the FIRST row nothing
///    is frozen yet, so an out-of-sequence row is rejected BEFORE the
///    output-set compare — a rejected first row stores no snapshot to poison
///    a retry.
/// 2. The compare-only preflight ([`super::resample_preflight_check_only`]) verifies
///    the output set (HSV + optional luma) WITHOUT committing; on a later row
///    it runs first, so a mid-frame output change surfaces as
///    `ResampleOutputsChanged` rather than being masked by a freshly-built
///    join's row-0 sequence mismatch, and a post-compare sequence check
///    rejects an out-of-sequence later row (including the failure-retry case
///    where the join was never built, so `expected == 0`).
/// 3. The join is built (its three area streams + ring scratch allocate
///    recoverably) and boxed recoverably — the sole pre-feed allocation — and
///    only THEN is the output-set freeze committed ([`frozen_outputs_check`]).
///    So a first-row build refusal leaves the field `None` AND
///    `resample_outputs` uncommitted (retryable with a changed output
///    attachment), and no caller output is touched.
///
/// A no-output call cannot reach here (`want_hsv_direct` implies HSV is
/// attached). The result is byte-identical to the native tier's HSV-only
/// output for the same area-resample.
#[cfg(feature = "yuv-planar")]
#[allow(clippy::too_many_arguments)]
pub(super) fn hsv_direct_resample(
  hsv_planar: &mut Option<std::boxed::Box<HsvDirectPlanarYuv>>,
  resample_outputs: &mut Option<super::FrozenOutputs>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  y_row: &[u8],
  u_row: &[u8],
  v_row: &[u8],
  matrix: ColorMatrix,
  full_range: bool,
  chroma_vsub: usize,
  build_chroma_plan: impl FnOnce() -> Result<ResamplePlan, ResampleError>,
  w: usize,
  plan: &ResamplePlan,
  idx: usize,
  use_simd: bool,
) -> Result<(), MixedSinkerError> {
  let expected = hsv_planar.as_ref().map_or(0, |join| join.y.next_y());
  // Compare-only preflight (NO commit): validate the output set + row sequence
  // WITHOUT freezing, so the join build below (the sole pre-feed allocation) runs
  // BEFORE the output-set freeze. A first-row build failure therefore leaves
  // `resample_outputs` uncommitted — state-atomic, retryable even with a changed
  // output attachment (mirrors `planar_dual_resample`). `want_hsv_direct` implies
  // HSV is attached, so `expected` is always `Some` and the no-output Break never
  // fires.
  if let ControlFlow::Break(()) = super::resample_preflight_check_only(
    resample_outputs,
    luma,
    luma_u16,
    &None,
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
    Some(expected),
    idx,
  )? {
    return Ok(());
  }
  // Build the join (its three area streams + ring scratch allocate recoverably)
  // and box it recoverably BEFORE the freeze commit; both fallible steps run
  // ahead of any field mutation, so a refusal leaves `hsv_planar` and
  // `resample_outputs` untouched.
  if hsv_planar.is_none() {
    let join = HsvDirectPlanarYuv::new(plan, build_chroma_plan, chroma_vsub, w, plan.src_h())?;
    let boxed = crate::resample::try_box(join).map_err(|_| {
      MixedSinkerError::Resample(ResampleError::AllocationFailed(PlanGeometry::new(
        plan.src_w(),
        plan.src_h(),
        plan.out_w(),
        plan.out_h(),
      )))
    })?;
    *hsv_planar = Some(boxed);
  }
  // Every pre-feed allocation succeeded — COMMIT the output-set freeze. On the
  // building (first) row this only stores the snapshot; the compare above already
  // rejected any mid-frame output change on a later row.
  frozen_outputs_check(
    resample_outputs,
    luma,
    luma_u16,
    &None,
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
  let join = hsv_planar.as_mut().expect("created in the preflight");
  let hsv = hsv.as_mut().expect("want_hsv_direct implies hsv attached");
  hsv_direct_feed_emit(
    join, hsv, luma, luma_u16, y_row, u_row, v_row, matrix, full_range, plan, idx, use_simd,
  )
}

/// Row-stage fused downscale shared by the planar formats with no
/// native tier (Yuv411p / Yuv422p / Yuv444p). Mirrors the Yuv420p
/// row-stage path: **luma / luma_u16 area-resample the Y plane
/// directly** (a 1-channel stream over `y_row`, the YUV luma
/// contract — luma is *not* re-derived from converted RGB), while RGB
/// / RGBA / HSV bin a converted source-width RGB row (the 3-channel
/// stream). `convert_rgb` fills the source-width scratch with RGB
/// using the format's own conversion kernel, and runs only when a
/// colour output is attached. Atomic preflight: every fallible step
/// (freeze, stream creation, sequence check, scratch growth +
/// conversion) precedes the first feed, so a failure mutates no
/// caller output.
///
/// #263 follow-up — **HSV-only** (no RGB / RGBA) is handled by the caller
/// BEFORE this helper: the planar dispatch sites call
/// [`hsv_direct_resample`] directly (binning Y / U / V on their own grids,
/// RGB-free) for an HSV-only area plan and reach `planar_dual_resample` only
/// for RGB / RGBA (with or without HSV — those keep the cheap RGB-staged
/// HSV). The semi-planar callers, whose chroma arrives interleaved and whose
/// 4:2:0 / 4:4:0 chroma-plan builders are `yuv-planar`-gated, keep the
/// RGB-staged HSV here unchanged (their RGB-free HSV-only resample is a
/// documented follow-up). So whenever `hsv` is set here, RGB or RGBA is also
/// set and the RGB-staged colour path runs.
#[cfg_attr(
  not(any(feature = "yuv-planar", feature = "yuv-semi-planar")),
  allow(dead_code)
)]
#[allow(clippy::too_many_arguments)]
pub(super) fn planar_dual_resample(
  luma_stream: &mut Option<std::boxed::Box<AreaStream<u8>>>,
  rgb_stream: &mut Option<std::boxed::Box<AreaStream<u8>>>,
  resample_outputs: &mut Option<super::FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  rgb_scratch: &mut std::vec::Vec<u8>,
  y_row: &[u8],
  w: usize,
  plan: &ResamplePlan,
  idx: usize,
  use_simd: bool,
  convert_rgb: impl FnOnce(&mut [u8]),
) -> Result<(), MixedSinkerError> {
  // Area-only sink (these planar YUV families are not routed to the filter
  // path): reject a filter plan before any work, so the plan's empty area
  // spans never reach an area stream.
  if plan.kind().is_filter() {
    return Err(plan.unsupported_filter().into());
  }
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
  // TOGETHER only after every pre-feed allocation below has succeeded (mirrors
  // `linear_light_resample`'s compare-before / commit-after). A first-row
  // allocation failure therefore leaves `resample_outputs` and both streams
  // untouched — state-atomic, retryable even with a changed output attachment.
  if let ControlFlow::Break(()) = super::resample_preflight_check_only(
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
  // Build any missing streams into LOCALS first (fallible; NO field mutation and
  // NO output-set commit yet).
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
  // Also grow the source-width colour scratch (fallible) BEFORE the commit: the
  // shared feed tail grows it on the first coloured row, so hoisting it here
  // keeps EVERY pre-feed allocation (streams + scratch) ahead of the freeze and
  // stream inserts; `planar_dual_feed_emit`'s own grow is then a no-op.
  if need_color {
    super::source_rgb_scratch(rgb_scratch, w, plan)?;
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

  // Stage + feed + emit. Shared with the filter path
  // ([`planar_dual_filter_resample`]) so the area and filter arms run the
  // identical convert-then-resample tail — the only difference is the
  // stream kind built above.
  planar_dual_feed_emit(
    luma_stream.as_mut(),
    rgb_stream.as_mut(),
    rgb,
    rgba,
    luma,
    luma_u16,
    hsv,
    rgb_scratch,
    y_row,
    w,
    plan,
    idx,
    use_simd,
    convert_rgb,
  )
}

/// Shared stage-then-feed tail for the 8-bit planar YUV family, used by
/// both [`planar_dual_resample`] (area) and [`planar_dual_filter_resample`]
/// (filter). The two paths differ only in the resampler kind built by the
/// caller — the convert-then-resample staging and per-output emit are
/// identical, so they live here behind the
/// [`RowResampler`](crate::resample::RowResampler) trait (which both
/// [`AreaStream`](crate::resample::AreaStream) and
/// [`FilterStream`](crate::resample::FilterStream) implement). Keeping the
/// emit byte-identical between the arms is what makes the filter output
/// match the area output up to the kernel weights.
///
/// Luma is the native Y resampled directly (the YUV luma contract — `luma`
/// copies each finalized `u8` Y row, `luma_u16` zero-extends it); colour
/// bins a caller-converted source-width RGB row (`convert_rgb` fills the
/// scratch only when a colour output is attached). The scratch grows via
/// the recoverable-allocation helper before the first feed, so a failure
/// mutates no caller output. These sources are 8-bit, so no native-depth
/// clamp applies — the `u8` stream finalizes to the full `u8` range, which
/// is the native range.
#[cfg_attr(
  not(any(feature = "yuv-planar", feature = "yuv-semi-planar")),
  allow(dead_code)
)]
#[allow(clippy::too_many_arguments)]
fn planar_dual_feed_emit<LS, CS>(
  luma_stream: Option<&mut LS>,
  rgb_stream: Option<&mut CS>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  rgb_scratch: &mut std::vec::Vec<u8>,
  y_row: &[u8],
  w: usize,
  plan: &ResamplePlan,
  idx: usize,
  use_simd: bool,
  convert_rgb: impl FnOnce(&mut [u8]),
) -> Result<(), MixedSinkerError>
where
  LS: RowResampler<u8>,
  CS: RowResampler<u8>,
{
  let ow = plan.out_w();
  // Stage the source-width colour scratch (the fallible growth runs before
  // the first feed, keeping the call atomic) only when a colour output is
  // attached.
  let color_row = if rgb_stream.is_some() {
    let scratch = super::source_rgb_scratch(rgb_scratch, w, plan)?;
    convert_rgb(scratch);
    Some(scratch)
  } else {
    None
  };

  if let Some(stream) = luma_stream {
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

/// Separable-filter fused resize for the 8-bit planar YUV family — the
/// [`SpanKind::Filter`](crate::resample::SpanKind) twin of
/// [`planar_dual_resample`]. It mirrors the area path exactly: the separate
/// Y/U/V planes are converted to a source-width RGB row by the **same**
/// `convert_rgb` closure (which upsamples 4:2:0 / 4:2:2 / 4:1:0 / 4:4:0
/// chroma), then the RGB is resampled by the signed-coefficient
/// [`FilterStream`] (the filter twin of the area bin) and the **same** emit
/// ([`planar_dual_feed_emit`]) is run — so the resampled colour output
/// equals the equivalent packed-RGB filter resample of the converted
/// pixels, and (because the area path converts-then-bins the same RGB)
/// matches the area output up to the kernel weights.
///
/// Luma stays the native Y filter-resampled (the filter twin of the area
/// path's native-Y bin): a 1-channel [`FilterStream<u8>`] resamples the Y
/// plane directly, so luma is taken from Y, never colour-derived. These
/// sources are 8-bit, so the `u8` stream finalizes to the full `u8` range,
/// which *is* the native range — no sub-16-bit native-depth clamp applies
/// (unlike the high-bit planar / packed YUV filter routes). `luma_u16`
/// zero-extends each resampled Y byte.
///
/// Atomic preflight (mirrors [`planar_dual_resample`]): a single
/// [`frozen_outputs_check`] over the output set, then a single sequence
/// check on whichever stream is fed every row **before any allocation** (an
/// out-of-sequence first row is rejected before the freeze, storing no
/// snapshot to poison a retry; on a later row the freeze runs first so a
/// mid-frame output change trips `ResampleOutputsChanged`), then every
/// stream and the source-width scratch is created before the first feed —
/// so a failure mutates no caller output. A no-output call has no stream to
/// sequence and stays a no-op.
#[cfg(any(feature = "yuv-planar", feature = "yuv-semi-planar"))]
#[allow(clippy::too_many_arguments)]
pub(super) fn planar_dual_filter_resample(
  luma_filter_stream: &mut Option<std::boxed::Box<crate::resample::FilterStream<u8>>>,
  rgb_filter_stream: &mut Option<std::boxed::Box<crate::resample::FilterStream<u8>>>,
  resample_outputs: &mut Option<super::FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  rgb_scratch: &mut std::vec::Vec<u8>,
  y_row: &[u8],
  w: usize,
  plan: &ResamplePlan,
  idx: usize,
  use_simd: bool,
  convert_rgb: impl FnOnce(&mut [u8]),
) -> Result<(), MixedSinkerError> {
  // This single-kernel tail filters ONE converted RGB row; a BICUBLIN plan
  // ([`Bicublin`](crate::resample::Bicublin)) carries a second (chroma) window
  // set that only the `Yuv420p` per-plane route reads, so reject it here rather
  // than silently filtering every plane with the luma kernel. Every non-4:2:0
  // planar / semi-planar format routes its filter dispatch through this tail,
  // so the one guard fences the whole family.
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
    Some(
      luma_filter_stream
        .as_ref()
        .map_or(0, |stream| stream.next_y()),
    )
  } else if need_color {
    Some(
      rgb_filter_stream
        .as_ref()
        .map_or(0, |stream| stream.next_y()),
    )
  } else {
    None
  };
  // Compare-only preflight (NO commit) — see `planar_dual_resample`: the
  // output-set freeze and the stream inserts commit TOGETHER only after every
  // pre-feed allocation below has succeeded, so a first-row allocation failure
  // is state-atomic (retryable with a changed output attachment).
  if let ControlFlow::Break(()) = super::resample_preflight_check_only(
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
    let stream = crate::resample::FilterStream::new(fh, fv, plan.src_w(), plan.src_h(), 1)?;
    Some(crate::resample::try_box(stream).map_err(|_| {
      MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
        crate::resample::PlanGeometry::new(plan.src_w(), plan.src_h(), plan.out_w(), plan.out_h()),
      ))
    })?)
  } else {
    None
  };
  let new_rgb = if need_color && rgb_filter_stream.is_none() {
    let stream = crate::resample::FilterStream::new(fh, fv, plan.src_w(), plan.src_h(), 3)?;
    Some(crate::resample::try_box(stream).map_err(|_| {
      MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
        crate::resample::PlanGeometry::new(plan.src_w(), plan.src_h(), plan.out_w(), plan.out_h()),
      ))
    })?)
  } else {
    None
  };
  // Grow the source-width colour scratch (fallible) BEFORE the commit too (see
  // `planar_dual_resample`): every pre-feed allocation ahead of the freeze/insert.
  if need_color {
    super::source_rgb_scratch(rgb_scratch, w, plan)?;
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

  // Stage + feed + emit. Shared with the area path
  // ([`planar_dual_resample`]) so the area and filter arms run the
  // identical convert-then-resample tail — the only difference is the
  // stream kind built above.
  planar_dual_feed_emit(
    luma_filter_stream.as_mut(),
    rgb_filter_stream.as_mut(),
    rgb,
    rgba,
    luma,
    luma_u16,
    hsv,
    rgb_scratch,
    y_row,
    w,
    plan,
    idx,
    use_simd,
    convert_rgb,
  )
}

/// Row-stage fused downscale for the **packed** YUV formats whose Y
/// samples are interleaved in the source plane (packed 4:1:1 —
/// [`Uyyvyy411`](crate::source::Uyyvyy411)). The planar twin
/// [`planar_dual_resample`] takes a ready source-width Y plane; here Y
/// must first be de-interleaved out of the packed plane, so this helper
/// owns a **second** scratch (`luma_scratch`, distinct from the colour
/// `rgb_scratch`) and the caller supplies a `convert_luma` closure that
/// fills it (the format's own `*_to_luma_row` kernel — the YUV luma
/// contract, *not* RGB-derived luma). Colour binning is identical to
/// the planar twin: `convert_rgb` fills the colour scratch with a
/// source-width RGB row via the format's fused `*_to_rgb_row` kernel
/// (chroma de-interleave + horizontal upsample in registers) and the
/// 3-channel stream bins it, so RGB equals an `Rgb24` area-resample of
/// the identity-converted frame.
///
/// Atomic preflight (matching [`planar_dual_resample`]): the output set
/// is frozen, then stream sequencing is checked, **both before any
/// allocation** — so a no-output sink stays a no-op, an out-of-sequence
/// row is rejected without staging a buffer, and `AllocationFailed`
/// can never mask `OutOfSequenceRow`. Only then are the (separate) luma
/// and colour scratches grown via the recoverable-allocation helpers
/// and their conversions run, before the first feed; a failure at any
/// step mutates no caller output.
#[cfg(feature = "yuv-packed")]
#[allow(clippy::too_many_arguments)]
pub(super) fn packed_yuv_dual_resample(
  luma_stream: &mut Option<std::boxed::Box<AreaStream<u8>>>,
  rgb_stream: &mut Option<std::boxed::Box<AreaStream<u8>>>,
  resample_outputs: &mut Option<super::FrozenOutputs>,
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
  convert_luma: impl FnOnce(&mut [u8]),
  convert_rgb: impl FnOnce(&mut [u8]),
) -> Result<(), MixedSinkerError> {
  // Area-only sink (packed YUV 4:2:2 is not routed to the filter path):
  // reject a filter plan before any work, so the plan's empty area spans
  // never reach an area stream.
  if plan.kind().is_filter() {
    return Err(plan.unsupported_filter().into());
  }
  let ow = plan.out_w();
  let need_luma = luma.is_some() || luma_u16.is_some();
  let need_color = rgb.is_some() || hsv.is_some() || rgba.is_some();

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
  // freshly-attached stream's row-0 sequence mismatch (attaching a luma or
  // colour output mid-frame spins that stream fresh at row 0).
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
  // Stage both source-width scratches (each via its own recoverable
  // grow) and run the conversions before any feed, keeping the call
  // atomic. The luma scratch is the de-interleaved Y plane; the colour
  // scratch is the source-width RGB row.
  let luma_row = if need_luma {
    let scratch = super::source_luma_scratch(luma_scratch, w, plan)?;
    convert_luma(scratch);
    Some(scratch)
  } else {
    None
  };
  let color_row = if need_color {
    let scratch = super::source_rgb_scratch(rgb_scratch, w, plan)?;
    convert_rgb(scratch);
    Some(scratch)
  } else {
    None
  };

  if let Some(y_row) = luma_row {
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
