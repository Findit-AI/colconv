//! Native fast-tier decimator for the **high-bit planar non-4:2:0** YUV
//! families — `Yuv422p10/12/14/16` (4:2:2), `Yuv444p10/12/14/16` (4:4:4),
//! `Yuv440p10/12` (4:4:0), LE + BE wire.
//!
//! The `u16` twin of the 8-bit non-4:2:0 planar
//! [`yuv_planar_process_native`](super::planar_8bit::yuv_planar_process_native),
//! and the non-4:2:0 sibling of the high-bit 4:2:0
//! [`yuv420p16_process_native`](super::subsampled_4_2_0_high_bit::yuv420p16_process_native):
//! bin the host-native Y / U / V planes straight to the output grid, then
//! convert ONCE per output row at output width through the 4:4:4 kernel (the
//! binned chroma is full-width at output resolution, so the convert is always
//! 4:4:4) — vs the row-stage tier
//! ([`packed_yuv422_triple_resample`](super::packed_yuv422_triple_resample) /
//! [`packed_yuv444_triple_resample`](super::packed_yuv444_triple_resample)),
//! which converts each source row at source width then bins. This is the
//! demand-driven P2 fast tier: default-on, within a small tolerance of the
//! row-stage tier (NOT byte-identical to the direct path), bench-gated.
//!
//! The three formats differ only in the chroma grid and its vertical cadence,
//! both supplied by the per-format caller (`chroma_vsub` + a `build_chroma_plan`
//! closure) so this body stays layout-agnostic — identical to the 8-bit twin:
//! - `Yuv422p` (4:2:2): chroma `w/2 x h` — half width, FULL height; a chroma
//!   row per Y row (`chroma_vsub == 1`); chroma plan a plain
//!   [`ResamplePlan::area`](crate::resample::ResamplePlan::area) over
//!   `(w/2, h)`; de-interleave width `w/2`.
//! - `Yuv444p` (4:4:4): chroma `w x h` — identical to Y; same lockstep cadence
//!   (`chroma_vsub == 1`); chroma plan equals the luma plan; de-interleave
//!   width `w`.
//! - `Yuv440p` (4:4:0): chroma `w x h/2` — FULL width, half height; a chroma
//!   row per TWO Y rows (`chroma_vsub == 2`, like 4:2:0 vertically); chroma
//!   plan [`ResamplePlan::area_chroma_440`](crate::resample::ResamplePlan::area_chroma_440)
//!   (full-width horizontal, luma-domain `area_halved` vertical); de-interleave
//!   width `w`.
//!
//! The triple-bin + convert-kernel independence is the highest risk (the #37
//! lesson: the u8 and u16 `YUV→RGB` kernels round INDEPENDENTLY, and a
//! u16-output kernel must never be narrowed from a u8 one or vice versa). Per
//! output row the staged host-native u16 Y / U / V feed:
//! - `luma` (u8): `dst = (Y >> (BITS - 8)) as u8`.
//! - `rgb_u16` / `rgba_u16`: the native-depth 4:4:4 u16 kernel (the generic
//!   `yuv444pN_to_rgb_u16_row_endian` for BITS ∈ {9, 10, 12, 14}, the dedicated
//!   `yuv444p16_to_rgb_u16_row_endian` for BITS = 16) — its OWN i32/i64 kernel,
//!   never a narrowing of the u8 colour — then expand to rgba_u16.
//! - `rgb` / `rgba` / `hsv` (u8): the u16-INPUT → u8-OUTPUT 4:4:4 kernel (the
//!   SAME one the row-stage 4:4:4 high-bit path uses), then fan to rgba / hsv.
//!
//! ★ THE NATIVE-DEPTH CLAMP. For sub-16-bit BITS (9/10/12/14) every colour
//! sample is clamped to `(1 << BITS) - 1` — done INSIDE the convert kernels
//! (`yuv_444p_n_to_rgb_or_rgba_u16_row` masks the inputs `& (1<<BITS)-1` and
//! clamps each output to `(1<<BITS)-1`; the i64 BITS=16 kernel covers the full
//! u16 range). The binned Y / U / V are area means of in-range host-native
//! samples, so they stay in range; the native-Y `luma` output is `>> (BITS-8)`
//! into u8, always in range. The optional native-depth `luma_u16` output is the
//! binned Y clamped to `(1 << BITS) - 1` (host-native u16, NOT narrowed) — the
//! formats with no `luma_u16` channel (planar / semi-planar) pass `&mut None`;
//! the packed Y2xx family threads its real buffer so attaching `luma_u16` no
//! longer falls the pipeline back to the row-stage tier. The bin-then-convert
//! test oracle MUST run the SAME clamping kernels (via an identity-resolution
//! high-bit 4:4:4 sink) so the comparison is clamp-for-clamp exact, never
//! against an unclamped value.
//!
//! The staged Y / U / V are HOST-NATIVE (the wire row is de-interleaved to
//! host-native u16 via [`deinterleave_y_high_bit_masked`](super::deinterleave_y_high_bit_masked)
//! into private source-width scratch BEFORE the [`AreaStream`] bins it —
//! matching the row-stage path), so every convert kernel runs with
//! `BE = HOST_NATIVE_BE` (= `from_ne`, a no-op load on every host) regardless
//! of the source wire endianness. A hard-coded `false` (`from_le`) would
//! byte-swap the already-native value on a big-endian target, corrupting
//! default-on native colour.

use super::{
  GeometryOverflow, HsvFrameMut, MixedSinkerError, deinterleave_y_high_bit_masked,
  planar_8bit::native_preflight_core,
};
use crate::{
  ColorMatrix,
  resample::{AreaStream, PlanGeometry, ResampleError, ResamplePlan, try_box, try_zeroed},
  row::{
    expand_rgb_to_rgba_row, expand_rgb_u16_to_rgba_u16_row, rgb_to_hsv_row,
    yuv444p9_to_rgb_row_endian, yuv444p9_to_rgb_u16_row_endian, yuv444p10_to_rgb_row_endian,
    yuv444p10_to_rgb_u16_row_endian, yuv444p12_to_rgb_row_endian, yuv444p12_to_rgb_u16_row_endian,
    yuv444p14_to_rgb_row_endian, yuv444p14_to_rgb_u16_row_endian, yuv444p16_to_rgb_row_endian,
    yuv444p16_to_rgb_u16_row_endian,
  },
};

// The staged Y / U / V the `AreaStream` produces are HOST-NATIVE u16 (the wire
// was decoded to native by `deinterleave_y_high_bit_masked` BEFORE binning), so every
// `*_to_*_row_endian` convert below must read them with `BE = HOST_NATIVE_BE`
// to keep the kernel's `from_le` / `from_be` load a no-op on EVERY host.
// Mirrors `subsampled_4_2_0_high_bit::native`'s `HOST_NATIVE_BE`.
const HOST_NATIVE_BE: bool = cfg!(target_endian = "big");

#[cfg(all(test, feature = "std", feature = "yuv-planar"))]
std::thread_local! {
  static FORCE_PLANAR_HB_NATIVE_ALLOC_FAILURE: core::cell::Cell<bool> =
    const { core::cell::Cell::new(false) };
}

/// Arms the source-scratch allocation failpoint for the **next**
/// output-bearing high-bit non-4:2:0 planar native row on the current thread.
/// The flag is consumed (take-on-read) by the first fallible source-scratch
/// grow that row reaches, so it fires exactly once and cannot leak into a
/// later test. Test-only — mirrors `arm_native_u16_alloc_failure` for the
/// 4:2:0 high-bit planar native tier. Only the high-bit planar native suite
/// arms it, and that suite is gated on `rgb` (its colour oracle), so the
/// setter is dead in a `yuv-planar`-without-`rgb` build even though the
/// thread-local it sets is read by the (always-`yuv-planar`) native join.
#[cfg(all(test, feature = "std", feature = "yuv-planar", feature = "rgb"))]
pub(crate) fn arm_planar_hb_native_alloc_failure() {
  FORCE_PLANAR_HB_NATIVE_ALLOC_FAILURE.with(|f| f.set(true));
}

#[cfg(all(test, feature = "std", feature = "yuv-planar"))]
std::thread_local! {
  static FORCE_PLANAR_HB_NATIVE_CHROMA_FAILURE: core::cell::Cell<bool> =
    const { core::cell::Cell::new(false) };
}

/// Arms a failpoint that fires when (and only when) the native join PLANS its
/// chroma grid — which happens exactly when colour output is requested. A
/// luma-only sink must never reach it, so an armed flag survives a luma-only
/// row unconsumed (the regression assertion) and is taken by the first colour
/// row. Test-only.
///
/// Only the high-bit non-4:2:0 native suites that own a colour oracle arm it —
/// V210 / Y2xx / P2xx / P4xx / packed-4:4:4 (each pulling in `yuv-planar` for
/// the join) plus the `rgb`-gated planar high-bit native suite — so the setter
/// is dead in a `yuv-planar`-without-any-consumer build even though the
/// thread-local it sets is read by the (always-`yuv-planar`) native join.
#[cfg(all(
  test,
  feature = "std",
  feature = "yuv-planar",
  any(
    feature = "rgb",
    feature = "v210",
    feature = "y2xx",
    feature = "yuv-444-packed",
    feature = "yuv-semi-planar"
  )
))]
pub(crate) fn arm_planar_hb_native_chroma_failure() {
  FORCE_PLANAR_HB_NATIVE_CHROMA_FAILURE.with(|f| f.set(true));
}

/// Native decimation join for the high-bit non-4:2:0 planar families — the
/// `u16` twin of [`NativePlanarYuv`](super::planar_8bit::NativePlanarYuv) and
/// the non-4:2:0 sibling of
/// [`NativeYuv420U16`](super::subsampled_4_2_0_high_bit::NativeYuv420U16). Y
/// streams on the frame grid; U / V on the format's chroma grid (its vertical
/// cadence captured by [`Self::chroma_vsub`]), every plane binned to FULL
/// output resolution and converted ONCE per output row at output width through
/// the 4:4:4 kernels. Each plane's in-order emissions stage into a two-slot
/// ring (`out_y & 1`); the moment all participating planes hold an output row
/// it finalizes — so no output-geometry alignment constraint ever applies. A
/// plane may lead another by at most one source row (the chroma grid is within
/// a factor of two of the luma grid vertically), which the two slots absorb.
/// For the lockstep formats (`chroma_vsub == 1`) the planes never lead, but
/// the two-slot machinery is harmless and shared.
pub(crate) struct NativePlanarYuvU16 {
  y: AreaStream<u16>,
  /// Source-width host-native Y de-interleave scratch (the wire Y plane
  /// normalized via [`deinterleave_y_high_bit_masked`] before [`Self::y`] bins it).
  /// Lazily grown to `src_w` `u16` on the first output-bearing row; empty
  /// otherwise.
  y_src: std::vec::Vec<u16>,
  /// Two-slot staging ring, `2 * out_w` (slot = `out_y & 1`).
  y_stage: std::vec::Vec<u16>,
  /// Chroma half of the join — absent for luma-only sinks, which therefore
  /// never read the chroma planes (the documented fast path). Decided at
  /// creation: the frozen-output contract makes the attached set
  /// frame-constant.
  chroma: Option<NativePlanarChromaU16>,
  /// Source-row width of the chroma planes (`w/2` for 4:2:2, `w` for 4:4:4 /
  /// 4:4:0) — the de-interleave length the chroma scratch is grown to and
  /// fed at. Distinct from the luma `src_w`.
  chroma_w: usize,
  /// Vertical chroma subsample factor: 1 for 4:2:2 / 4:4:4 (a chroma row per
  /// luma row), 2 for 4:4:0 (a chroma row per two luma rows). The chroma
  /// stream is fed when `idx % chroma_vsub == 0`, at `cidx = idx / chroma_vsub`,
  /// and the sequence check expects `chroma.next_y() == idx.div_ceil(chroma_vsub)`.
  chroma_vsub: usize,
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
}

/// Chroma-grid streams, source de-interleave scratch, and staging of
/// [`NativePlanarYuvU16`].
struct NativePlanarChromaU16 {
  u: AreaStream<u16>,
  v: AreaStream<u16>,
  /// Source-width (chroma-width) host-native U / V de-interleave scratch.
  /// Lazily grown to `chroma_w` `u16` each on the first chroma-bearing output
  /// row; empty otherwise.
  u_src: std::vec::Vec<u16>,
  v_src: std::vec::Vec<u16>,
  u_stage: std::vec::Vec<u16>,
  v_stage: std::vec::Vec<u16>,
}

impl NativePlanarYuvU16 {
  /// `build_chroma_plan` lazily builds the format's chroma grid against the
  /// SAME output geometry as `plan` (the luma plan) — invoked ONLY when colour
  /// is needed, so a luma-only sink never plans or allocates chroma state.
  /// `chroma_vsub` is its vertical cadence; `chroma_w` its source-row width.
  /// All are supplied by the per-format caller so this body stays
  /// layout-agnostic.
  fn new(
    plan: &ResamplePlan,
    build_chroma_plan: impl FnOnce() -> Result<ResamplePlan, ResampleError>,
    chroma_vsub: usize,
    chroma_w: usize,
    w: usize,
    h: usize,
    need_color: bool,
  ) -> Result<Self, ResampleError> {
    // The native high-bit planar join is integer area-only; reject a filter
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
    let chroma = if need_color {
      #[cfg(all(test, feature = "std", feature = "yuv-planar"))]
      if FORCE_PLANAR_HB_NATIVE_CHROMA_FAILURE.with(|f| f.take()) {
        return Err(ResampleError::AllocationFailed(PlanGeometry::new(
          w,
          h,
          plan.out_w(),
          plan.out_h(),
        )));
      }
      let chroma_plan = build_chroma_plan()?;
      chroma_centered = chroma_plan.has_chroma_phase();
      Some(NativePlanarChromaU16 {
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
      chroma_w,
      chroma_vsub,
      staged: [[false; 2]; 3],
      next_emit: 0,
      chroma_centered,
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

  /// Next Y source row this join expects — the per-row sequence counter (the
  /// native path bins Y on every output-bearing row, luma implicit).
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

  /// Sequencing preflight across all three plane streams — checked before any
  /// plane is fed so a violating call mutates nothing. Chroma rows advance
  /// once per `chroma_vsub` source rows, so their expected counter is
  /// `idx.div_ceil(chroma_vsub)`.
  fn check_sequence(&self, idx: usize) -> Result<(), MixedSinkerError> {
    if self.y.next_y() != idx {
      return Err(MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(
        crate::resample::OutOfSequenceRow::new(self.y.next_y(), idx),
      )));
    }
    if let Some(chroma) = self.chroma.as_ref() {
      let chroma_expected = idx.div_ceil(self.chroma_vsub);
      for stream in [&chroma.u, &chroma.v] {
        if stream.next_y() != chroma_expected {
          return Err(MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(
            crate::resample::OutOfSequenceRow::new(
              stream.next_y().saturating_mul(self.chroma_vsub),
              idx,
            ),
          )));
        }
      }
    }
    Ok(())
  }
}

/// Thin high-bit wrapper over
/// [`native_preflight_core`](super::planar_8bit::native_preflight_core) for
/// the [`NativePlanarYuvU16`] join — supplies the u16-join-typed expected row
/// and threads the native-depth u16 colour outputs (`rgb_u16` / `rgba_u16`)
/// plus the optional native-depth `luma_u16` into the frozen-output check. The
/// planar / semi-planar families expose no `luma_u16` channel and pass `&None`;
/// the packed Y2xx family threads its real buffer so a mid-frame `luma_u16`
/// attach is classified by the frozen-output check (`ResampleOutputsChanged`),
/// not the route guard. See `native_preflight_core` for the 4-point rejection
/// logic and its ordering contract.
///
/// `pub(crate)` so the high-bit **semi-planar** non-4:2:0 P-format wrapper
/// (`subsampled_4_2_2_high_bit::p2xx` / `subsampled_4_4_4_high_bit::p4xx`),
/// which reuses [`yuv_planar16_process_native`] after de-interleaving + de-packing
/// the packed UV plane, can run this COMPLETE pre-feed preflight ahead of its own
/// fallible de-pack scratch grow — the same ordering the 8-bit semi-planar
/// non-4:2:0 wrapper uses with `native_planar_preflight_check_only`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn native_planar_hb_preflight(
  join: &Option<std::boxed::Box<NativePlanarYuvU16>>,
  resample_outputs: &mut Option<super::FrozenOutputs>,
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
  native_preflight_core(
    join.as_ref().map_or(0, |j| j.next_y()),
    resample_outputs,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    luma,
    luma_u16,
    hsv,
    idx,
    need_luma,
    need_color,
  )
}

/// Grows a source-de-interleave scratch to `len` `u16` under the planner's
/// recoverable-allocation contract, with the test-only allocation failpoint on
/// the FIRST such grow of an output-bearing row (mirrors the 4:2:0 high-bit
/// `grow_src_scratch`). Runs after the preflight and after the join is built,
/// so a rejected row never reaches it.
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
    if FORCE_PLANAR_HB_NATIVE_ALLOC_FAILURE.with(|f| f.take()) {
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

/// Native-tier path for the high-bit non-4:2:0 planar families. Const-generic
/// over `BITS` (one body, the live call sites 9 / 10 / 12 / 14 / 16) and `BE`
/// (the source wire endianness). `chroma_vsub` is the format's vertical chroma
/// cadence (1 for 4:2:2 / 4:4:4, 2 for 4:4:0), `chroma_w` its source-row width
/// (`w/2` for 4:2:2, `w` for 4:4:4 / 4:4:0), and `build_chroma_plan` builds its
/// chroma grid against the same output geometry; all three are supplied by the
/// per-format caller so this body is layout-agnostic. Phasing mirrors the
/// 4:2:0 high-bit twin and the row-stage tier: the COMPLETE pre-feed preflight
/// (idempotent double-run vs the routing wrapper), the join build, sequencing,
/// source / colour scratch sizing, then the feeds — with nothing fallible
/// after the first feed.
#[allow(clippy::too_many_arguments)]
pub(crate) fn yuv_planar16_process_native<const BITS: u32, const BE: bool>(
  plan: &ResamplePlan,
  native: &mut Option<std::boxed::Box<NativePlanarYuvU16>>,
  resample_outputs: &mut Option<super::FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  rgb_u16: &mut Option<&mut [u16]>,
  rgba_u16: &mut Option<&mut [u16]>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  rgb_scratch: &mut std::vec::Vec<u8>,
  rgb_scratch_u16: &mut std::vec::Vec<u16>,
  y_row: &[u16],
  u_row: &[u16],
  v_row: &[u16],
  matrix: ColorMatrix,
  full_range: bool,
  idx: usize,
  w: usize,
  h: usize,
  chroma_vsub: usize,
  chroma_w: usize,
  build_chroma_plan: impl FnOnce() -> Result<ResamplePlan, ResampleError>,
  use_simd: bool,
) -> Result<(), MixedSinkerError> {
  const {
    assert!(
      BITS > 8 && BITS <= 16,
      "BITS must be in (8, 16] for high-bit non-4:2:0 planar YUV"
    )
  };
  let ow = plan.out_w();
  let need_luma = luma.is_some() || luma_u16.is_some();
  let need_color_u8 = rgb.is_some() || rgba.is_some() || hsv.is_some();
  let need_color_u16 = rgb_u16.is_some() || rgba_u16.is_some();
  let need_color = need_color_u8 || need_color_u16;

  // Complete pre-feed rejection preflight (no-output short-circuit, first-row
  // out-of-sequence, frozen-output, post-freeze sequence) ahead of any
  // fallible allocation — re-run in place of an inline block, as the 4:2:0
  // high-bit native does; the double-run vs the routing wrapper is idempotent
  // (the freeze stores on the first output-bearing row, the second run is a
  // matching check, the OOS-first-row branch is `is_none()`-guarded so it is
  // skipped once frozen). `Ok(false)` is the no-output no-op.
  if !native_planar_hb_preflight(
    native,
    resample_outputs,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    luma,
    luma_u16,
    hsv,
    idx,
    need_luma,
    need_color,
  )? {
    return Ok(());
  }

  // The join's chroma half is fixed at creation; if the frame's colour
  // capability differs (outputs attached since the previous frame — the
  // frozen check pins them WITHIN a frame, not across frames), rebuild it.
  if native
    .as_ref()
    .is_some_and(|join| join.chroma.is_some() != need_color)
  {
    *native = None;
  }
  let join = match native {
    Some(join) => join,
    None => {
      // Build the join (its de-interleave scratch allocates recoverably) and box
      // it recoverably (`try_box`, not `Box::new` — the latter aborts on OOM).
      // Both fallible steps run BEFORE `native.insert`, so a refusal at either
      // returns `Err` with the field still `None` — no caller output is touched
      // and the next row retries (the first-row-transactional contract).
      let join = NativePlanarYuvU16::new(
        plan,
        build_chroma_plan,
        chroma_vsub,
        chroma_w,
        w,
        h,
        need_color,
      )?;
      let boxed = try_box(join).map_err(|_| {
        MixedSinkerError::Resample(ResampleError::AllocationFailed(PlanGeometry::new(
          w,
          h,
          plan.out_w(),
          plan.out_h(),
        )))
      })?;
      native.insert(boxed)
    }
  };
  join.check_sequence(idx)?;

  // Colour OUTPUT scratch at output width (one binned row converts here before
  // fanning to the caller buffers). Both grows are fallible and precede the
  // first feed, keeping the call atomic.
  if need_color_u8 {
    let row_bytes =
      ow.checked_mul(3)
        .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
          ow,
          plan.out_h(),
          3,
        )))?;
    if rgb_scratch.len() < row_bytes {
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
  // The chroma scratch is grown only on chroma-bearing rows, exactly where the
  // join reads chroma.
  grow_src_scratch(&mut join.y_src, w, w, h, plan)?;
  let feed_chroma = join.chroma.is_some() && idx.is_multiple_of(join.chroma_vsub);
  if feed_chroma {
    // Split-borrow: grow the two chroma scratches without holding `join`
    // immutably across the call.
    let cw = join.chroma_w;
    let chroma = join.chroma.as_mut().expect("feed_chroma implies Some");
    grow_src_scratch(&mut chroma.u_src, cw, w, h, plan)?;
    grow_src_scratch(&mut chroma.v_src, cw, w, h, plan)?;
  }

  // De-interleave the wire planes into host-native scratch. Everything past
  // this point is infallible.
  deinterleave_y_high_bit_masked::<BITS, BE>(y_row, &mut join.y_src, w);
  if feed_chroma {
    let cw = join.chroma_w;
    let chroma = join.chroma.as_mut().expect("feed_chroma implies Some");
    // Mask BEFORE the AreaStream bins these: the native tier averages the
    // source chroma, and the colour kernel's per-pixel extract_hb would then
    // mask only the binned result. `avg(dirty) & mask` can differ from
    // `avg(dirty & mask)` (e.g. 10-bit 0 and 0x400 average to 0x200, which
    // survives the mask, vs per-sample 0), so dirty bits must be cleared per
    // source sample here, not after binning.
    deinterleave_y_high_bit_masked::<BITS, BE>(u_row, &mut chroma.u_src, cw);
    deinterleave_y_high_bit_masked::<BITS, BE>(v_row, &mut chroma.v_src, cw);
  }

  // Feed the planes into their streams. The Y plane bins every row; the chroma
  // planes only on `idx % chroma_vsub == 0` rows (`cidx = idx / chroma_vsub`).
  let NativePlanarYuvU16 {
    y,
    y_src,
    y_stage,
    chroma,
    chroma_w,
    chroma_vsub,
    staged,
    next_emit,
    // Cache key read only at build / the native call site; the feed loop
    // ignores it.
    chroma_centered: _,
  } = &mut **join;
  y.feed_row(idx, &y_src[..w], use_simd, |oy, out_row| {
    let slot = oy & 1;
    y_stage[slot * ow..slot * ow + ow].copy_from_slice(out_row);
    staged[0][slot] = true;
  })?;
  if let Some(c) = chroma.as_mut()
    && idx.is_multiple_of(*chroma_vsub)
  {
    let cidx = idx / *chroma_vsub;
    let cw = *chroma_w;
    let NativePlanarChromaU16 {
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

    if need_luma {
      // Clamp to the native max before EITHER luma emit: an overrange binned Y
      // (from out-of-gamut input whose high bits exceed BITS) must saturate, not
      // wrap — the row-stage luma path narrows / passes through the clamped u16.
      let native_max: u16 = ((1u32 << BITS) - 1) as u16;
      // u8 luma: the clamped binned Y narrowed `>> (BITS - 8)`.
      if let Some(buf) = luma.as_deref_mut() {
        for (dst, &src) in buf[oy * ow..(oy + 1) * ow].iter_mut().zip(y_out) {
          *dst = (src.min(native_max) >> (BITS - 8)) as u8;
        }
      }
      // Native-depth luma_u16: the SAME clamped binned Y, host-native u16, NOT
      // narrowed — keeping `luma_u16 <= (1 << BITS) - 1` (the native-depth
      // contract). The packed Y2xx family threads its real buffer here; the
      // planar / semi-planar families pass `&mut None` and skip this.
      if let Some(buf) = luma_u16.as_deref_mut() {
        for (dst, &src) in buf[oy * ow..(oy + 1) * ow].iter_mut().zip(y_out) {
          *dst = src.min(native_max);
        }
      }
    }

    if let Some(c) = chroma.as_ref() {
      let u_out = &c.u_stage[slot * ow..slot * ow + ow];
      let v_out = &c.v_stage[slot * ow..slot * ow + ow];

      // Native-depth u16 colour — its OWN independent kernel, never a
      // narrowing of the u8 colour (the #37 contract). The staged Y / U / V
      // are host-native, so `BE = HOST_NATIVE_BE`.
      if need_color_u16 {
        let out_rgb = &mut rgb_scratch_u16[..ow * 3];
        emit_rgb_u16::<BITS>(
          y_out, u_out, v_out, out_rgb, ow, matrix, full_range, use_simd,
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

      // u8 colour — the u16-INPUT → u8-OUTPUT 4:4:4 kernel (the same one the
      // row-stage 4:4:4 high-bit path uses), independent of the u16 colour.
      if need_color_u8 {
        let out_rgb = &mut rgb_scratch[..ow * 3];
        emit_rgb_u8::<BITS>(
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

/// u16-output 4:4:4 conversion at output width — the staged Y / U / V are
/// host-native, so `big_endian = HOST_NATIVE_BE`. Dispatches on `BITS` to the
/// per-format wrapper, which pins the generic 4:4:4 u16 kernel to a supported
/// `BITS` (9/10/12/14 → the `i32` `yuv_444p_n_to_rgb_u16_row` family; 16 → the
/// dedicated `i64`-chroma `yuv444p16_to_rgb_u16_row`).
///
/// A runtime `if BITS == 16 { dedicated } else { generic::<BITS> }` will not
/// do: a const-generic `if` still MONOMORPHIZES the `else` arm at `BITS = 16`,
/// where the generic kernel const-asserts against 16. A `match` over the four
/// CONCRETE (non-generic) per-format wrappers calls only the live one and
/// never instantiates the invalid `::<16>` form. Identical to the 4:2:0
/// high-bit native's `emit_rgb_u16`.
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

/// u8-output 4:4:4 conversion at output width (the u16-INPUT → u8-OUTPUT kernel
/// — the SAME one the row-stage 4:4:4 high-bit path uses). Same per-format
/// dispatch + `big_endian = HOST_NATIVE_BE` rationale as [`emit_rgb_u16`].
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
