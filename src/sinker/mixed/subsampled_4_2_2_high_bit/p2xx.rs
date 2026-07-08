use super::super::{
  GeometryOverflow, InsufficientBuffer, MixedSinker, MixedSinkerError, RowIndexOutOfRange,
  RowShapeMismatch, RowSlice, WidthAlignment, check_dimensions_match,
  packed_yuv422_triple_filter_resample, packed_yuv422_triple_resample, reset_high_bit_yuv_streams,
  rgb_row_buf_or_scratch, rgba_plane_row_slice, rgba_u16_plane_row_slice,
};
use crate::{PixelSink, row::*, source::*};

// `NativeRouteChanged` is raised only by the native fast tier's route-flip
// guard, which exists only when the reused planar join is compiled in.
// `chroma_422_center_sited_h` + the `_pn` chroma-staging wrappers drive the
// centered-siting (#302) horizontal upsample; both halves of the feature are
// required (the predicate + the 4:4:4 P-format decode kernels live behind
// `yuv-planar`). 4:2:2 reuses the 4:2:0 `p0xx` interleaved-chroma staging
// verbatim — the horizontal 2:1 upsample is identical; only the vertical
// subsample differs (handled by the walker, not the sink).
#[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
use super::super::{
  ChromaSitingChanged, FrozenOutputs, HsvFrameMut, NativePlanarYuvU16, NativeRouteChanged,
  chroma_422_center_sited_h, native_planar_hb_preflight_check_only,
  planar_8bit::YUV422P_CENTERED_H_PHASE,
  resample_preflight_check_only,
  subsampled_4_2_0_high_bit::{
    reserve_pn_chroma_full_u16, upsample_pn_chroma_center_h, upsample_pn_chroma_center_h_low_packed,
  },
  yuv_planar16_process_native,
};
#[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
use crate::{
  ColorMatrix,
  resample::{
    AveragingDomain, InsertionContext, InsertionPoint, PlanGeometry, ResampleError, ResamplePlan,
    select_insertion_point,
  },
};

// The native fast tier de-interleaves + DE-PACKS each wire plane into
// wrapper-owned host-native LOGICAL u16 scratch BEFORE handing it to the
// planar delegate, so the delegate's own `from_le` / `from_be` decode must be
// a no-op load on every host: pass `BE = HOST_NATIVE_BE` (= `from_ne`).
// Passing the source wire `BE` here would byte-swap the already-native scratch
// on a big-endian target. Mirrors the 4:2:0 high-bit semi-planar `p0xx`.
#[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
const HOST_NATIVE_BE: bool = cfg!(target_endian = "big");

// Test-only allocation failpoint for the wrapper-owned Y / U / V de-pack
// scratch grow in `p2xx_process_native`. Armed, the FIRST (Y) scratch grow of
// an output-bearing row returns the crate's recoverable `AllocationFailed`
// WITHOUT growing — so the atomicity regressions can prove the join's pre-feed
// preflight (out-of-sequence / frozen-output) runs BEFORE this fallible grow.
// Mirrors the 4:2:0 high-bit semi-planar `FORCE_P0XX_ALLOC_FAILURE`. Strictly
// test-only — the non-test build compiles this away entirely.
#[cfg(all(
  test,
  feature = "std",
  feature = "yuv-semi-planar",
  feature = "yuv-planar"
))]
std::thread_local! {
  static FORCE_P2XX_ALLOC_FAILURE: core::cell::Cell<bool> =
    const { core::cell::Cell::new(false) };
}

/// Arms the wrapper de-pack scratch allocation failpoint for the **next**
/// output-bearing high-bit semi-planar 4:2:2 native row on the current thread.
/// The flag is consumed (take-on-read) by the first fallible scratch grow that
/// row reaches, so it fires exactly once and cannot leak into a later test.
/// Test-only.
#[cfg(all(
  test,
  feature = "std",
  feature = "yuv-semi-planar",
  feature = "yuv-planar"
))]
pub(crate) fn arm_p2xx_alloc_failure() {
  FORCE_P2XX_ALLOC_FAILURE.with(|f| f.set(true));
}

/// Grows a wrapper-owned de-pack scratch to `len` `u16` under the planner's
/// recoverable-allocation contract, optionally firing the test-only failpoint
/// (`fail = true` only on the FIRST grow of an output-bearing row). Runs after
/// the join's preflight clears, so a rejected row never reaches it.
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
  // `fail` is consumed by the caller; on the non-test build it is `false` and
  // the whole branch compiles away.
  let _ = fail;
  if scratch.len() < len {
    #[cfg(all(
      test,
      feature = "std",
      feature = "yuv-semi-planar",
      feature = "yuv-planar"
    ))]
    if fail && FORCE_P2XX_ALLOC_FAILURE.with(|f| f.take()) {
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

/// Native fast-tier decimator for the **high-bit semi-planar 4:2:2** P-format
/// family ([`P210`](crate::source::P210) / [`P212`](crate::source::P212) /
/// [`P216`](crate::source::P216)): bins the native Y / U / V planes straight to
/// the output grid and converts once per output row at output resolution. The
/// 4:2:2 sibling of the 4:2:0 high-bit semi-planar
/// [`p0xx_process_native`](crate::sinker::mixed::subsampled_4_2_0_high_bit) and
/// the `u16` twin of the 8-bit semi-planar non-4:2:0
/// [`semi_planar_process_native_non420`](crate::sinker::mixed::semi_planar_8bit),
/// reusing the high-bit non-4:2:0 PLANAR join verbatim
/// ([`yuv_planar16_process_native`]) after de-interleaving + DE-PACKING the
/// wire row into wrapper-owned host-native LOGICAL u16 scratch.
///
/// THE SEAM: [`yuv_planar16_process_native`] wire-decodes its `y_row` / `u_row`
/// / `v_row` input (`from_le` / `from_be`) but applies **no** high-bit shift —
/// it treats them as **low-packed LOGICAL** u16. P-format Y is HIGH-BIT-PACKED
/// (`logical << (16 - BITS)`) and the UV plane is INTERLEAVED + high-packed. So
/// this wrapper must, per row, decode the wire AND de-pack (`>> (16 - BITS)`)
/// the Y, and de-interleave (`U,V` order — every P-format is UV-order, no VU
/// variant) + de-pack EACH of U and V, into host-native logical scratch — then
/// delegate with `BE = HOST_NATIVE_BE` so the delegate's internal decode is a
/// no-op load on every host. The de-pack hits Y AND U AND V; at `BITS = 16` the
/// shift is `>> 0` (a harmless no-op — the 10/12 tests guard the live shift).
///
/// 4:2:2 layout vs 4:2:0: the chroma plane is `w/2 × h` (horizontal-only
/// subsample, vertical cadence `chroma_vsub = 1`), so a chroma row feeds EVERY
/// colour Y row — vs the 4:2:0 even-only cadence. `chroma_w = w / 2`; the packed
/// UV row is `w` u16 (`w/2` interleaved pairs). The delegate builds its chroma
/// grid against the same output geometry via the `build_chroma_plan` closure.
///
/// `chroma_h_phase` is the RFC #238 horizontal chroma sampling phase folded into
/// the chroma area weights ([`ResamplePlan::area_chroma_422`]): `0.25` for the
/// centered 4:2:2 group ([`chroma_422_center_sited_h`]), `0.0` for co-sited /
/// unspecified. At phase `0.0` the folded plan is byte-identical to the plain
/// `area` plan, so the co-sited output is untouched.
///
/// Atomicity (the nv12 / high-bit lesson): the join's COMPLETE pre-feed
/// preflight runs FIRST — `Ok(false)` no-op short-circuit, first-row
/// out-of-sequence, frozen-output — BEFORE any fallible scratch grow, so a
/// rejected row returns its deterministic typed error
/// (`OutOfSequenceRow` / `ResampleOutputsChanged`), never `AllocationFailed`,
/// and touches no caller output. The de-pack into scratch is infallible and
/// happens only after the preflight clears; the delegate re-runs the identical
/// preflight (idempotent) and owns the binning + conversion.
///
/// Lazy chroma: a luma-only sink skips the chroma de-interleave/scratch
/// entirely (`need_color` guard), matching the delegate's lazy chroma plan —
/// luma-only resampling never depends on an unused chroma allocation.
#[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
#[allow(clippy::too_many_arguments)]
fn p2xx_process_native<const BITS: u32, const BE: bool, const LOW_PACKED: bool>(
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
  uv_half: &[u16],
  chroma_h_phase: f64,
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
      "BITS must be in (8, 16] for high-bit semi-planar 4:2:2 P-format"
    )
  };
  let need_luma = luma.is_some();
  let need_color =
    rgb.is_some() || rgba.is_some() || hsv.is_some() || rgb_u16.is_some() || rgba_u16.is_some();
  // 4:2:2 chroma is half-width, full-height: `chroma_w = w / 2`, a chroma row
  // per Y row (`chroma_vsub = 1`).
  let cw = w / 2;

  // Run the planar join's COMPLETE compare-only pre-feed rejection preflight
  // FIRST — no-output short-circuit, first-row out-of-sequence, AND output-set
  // compare (mid-frame output change) — BEFORE any fallible scratch grow below,
  // so every rejection returns its deterministic typed error and leaves the
  // wrapper scratch untouched (the crate's preflight-atomicity contract).
  // Compare-only (no output-set freeze), so the de-pack reserve below stays a
  // genuine pre-commit step ahead of the delegate's own commit; the delegate
  // re-runs this identical compare and owns the single commit.
  if !native_planar_hb_preflight_check_only(
    native_planar_u16,
    resample_outputs,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    luma,
    // The high-bit semi-planar 4:2:2 P-format exposes no `luma_u16` output.
    &None,
    hsv,
    idx,
    need_luma,
    need_color,
  )? {
    return Ok(());
  }

  // Grow the wrapper de-pack scratch under the planner's recoverable contract —
  // Y always, U / V only on a colour row (4:2:2: every Y row is a chroma row
  // when colour is wanted). All grows precede the infallible de-pack and the
  // delegate call. The failpoint fires on the FIRST (Y) grow only.
  grow_depack_scratch(y_scratch, w, true, w, h, plan)?;
  if need_color {
    grow_depack_scratch(u_scratch, cw, false, w, h, plan)?;
    grow_depack_scratch(v_scratch, cw, false, w, h, plan)?;
  }

  // De-pack the wire planes into host-native LOGICAL scratch. Decode the wire
  // endianness, then extract the active `BITS`: for the high-bit-packed
  // P-formats (`LOW_PACKED = false`) shift the high `BITS` down
  // (`>> (16 - BITS)`; `>> 0` at BITS = 16); for low-bit-packed NV20
  // (`LOW_PACKED = true`) mask the low `BITS` (`& ((1 << BITS) - 1)`).
  // Everything past here is infallible.
  let depack = |logical: u16| -> u16 {
    if LOW_PACKED {
      logical & ((1u16 << BITS) - 1)
    } else {
      logical >> (16 - BITS)
    }
  };
  for (d, &s) in y_scratch[..w].iter_mut().zip(y_row.iter()) {
    let logical = if BE { u16::from_be(s) } else { u16::from_le(s) };
    *d = depack(logical);
  }
  if need_color {
    // P-format / NV20 chroma is interleaved `U,V,U,V…` (U at even element);
    // each of U and V is independently packed and must be de-packed.
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
      u_scratch[i] = depack(u);
      v_scratch[i] = depack(v);
    }
  }

  // Delegate to the planar high-bit non-4:2:0 join with `BE = HOST_NATIVE_BE`
  // so its internal decode is a no-op on the already-native scratch, at the
  // 4:2:2 chroma geometry (`chroma_vsub = 1`, `chroma_w = w / 2`). Empty U / V
  // on luma-only rows (the join reads chroma only under colour).
  let (u_plane, v_plane): (&[u16], &[u16]) = if need_color {
    (&u_scratch[..cw], &v_scratch[..cw])
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
    // The high-bit semi-planar 4:2:2 P-format exposes no `luma_u16` output.
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
    cw,
    || ResamplePlan::area_chroma_422(cw, h, plan.out_w(), plan.out_h(), chroma_h_phase, 0.0),
    use_simd,
  )
}

// ---- P210 impl ----------------------------------------------------------
//
// 4:2:2 high-bit-packed semi-planar (10-bit). Per-row UV layout is
// identical to P010 (`width` u16 elements, half-width interleaved);
// only the walker reads chroma row `r` instead of `r / 2`. Reuses the
// `p010_to_rgb_*` row primitives verbatim.

impl<'a, R, const BE: bool> MixedSinker<'a, P210<BE>, R> {
  /// Attaches a packed **`u16`** RGB output buffer. 10-bit
  /// **low-bit-packed** output (yuv420p10le convention, not P210
  /// packing). Length is in `u16` elements: `width x height x 3`.
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

  /// Attaches a packed **8‑bit** RGBA output buffer. The 10‑bit P210
  /// source (semi‑planar, high‑bit‑packed) is converted to 8‑bit RGBA
  /// via the `BITS = 10` Q15 kernel family; alpha = `0xFF` (P210 has
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
  /// P210 high‑bit packing. Length is measured in `u16` **elements**
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

impl<R, const BE: bool> P210Sink<BE> for MixedSinker<'_, P210<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, P210<BE>, R> {
  type Input<'r> = P210Row<'r>;
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

  fn process(&mut self, row: P210Row<'_>) -> Result<(), Self::Error> {
    // P210 stores 10‑bit samples high‑bit‑packed; bit depth is fixed
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
    // `Copy`, so read it before the field split-borrow below. Gated like its
    // only consumer (`chroma_422_center_sited_h` + the 4:4:4 P-format kernels
    // need `yuv-planar`); a semi-planar-only build keeps the default decode.
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
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      native,
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      native_planar_u16,
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      p0xx_y_half,
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      p0xx_u_half,
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      p0xx_v_half,
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      frozen_native_route,
      // Full-width interleaved chroma staging for the centered-siting (#302)
      // identity decode; reuses the 4:2:0 `p0xx` scratch + wrappers verbatim.
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      chroma_full_u16,
      // RFC #238 S5b: the 4:2:2 chroma siting phase frozen on the first output
      // row (the `u16` semi-planar twin of the planar Yuv422p `frozen_chroma_centered`).
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      frozen_chroma_centered,
      ..
    } = self;

    // Non-identity plan. A `Filter` plan routes to the shared high-bit 4:2:2
    // signed-coefficient filter tail (there is NO native fast tier for the
    // filter path), so it branches FIRST, before the area native-route
    // machinery below. For an `Area` plan: when the native tier is enabled
    // (and the planar join it reuses is compiled in), bin the native Y / U / V
    // planes at output resolution and convert once per output row,
    // de-interleaving + de-packing the P210 chroma + Y into wrapper-owned
    // logical scratch first; otherwise (or under `with_native(false)`) feed
    // the shared area triple-resample tail. P210 is semi-planar 4:2:2: the
    // interleaved half-width UV is de-interleaved + horizontally upsampled
    // in-register by the (P010-shared) `p010_to_rgb*` kernels, and 4:2:2
    // chroma is full-height (the walker hands each luma row its own
    // `uv_half`). The Y de-pack shift `>> (16 - BITS)` yields the logical
    // native Y; `luma = binned_Y >> (BITS - 8)`. P210 exposes no `luma_u16`,
    // so it is `&mut None`. The filter tail clamps a signed-kernel overshoot
    // to the native max for this sub-16-bit source (both colour and native-Y
    // luma), matching the in-range area path.
    if let Some(plan) = plan.as_ref() {
      let matrix = row.matrix();
      let full_range = row.full_range();
      let (y, uv_half) = (row.y(), row.uv_half());
      // RFC #238 S5b — 4:2:2 horizontal chroma siting for the high-bit
      // semi-planar P210, mirroring the planar Yuv422p twin (S5a) on the
      // semi-planar de-interleave (S2a). The centered group
      // (`chroma_422_center_sited_h`) samples chroma at `+0.25` chroma-sample;
      // the co-sited / unspecified group is phase 0 (byte-identical). The native
      // fast tier folds the phase into the `area_chroma_422` chroma weights; the
      // filter and row-stage tiers reconstruct full-width interleaved `u16`
      // chroma (de-interleave + phase-0.5 upsample into `chroma_full_u16`) and
      // decode 4:4:4 via the `p410` full-chroma kernels — the co-sited arms keep
      // the fused `p010` half-chroma decode.
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      let center_sited = chroma_422_center_sited_h(chroma_location);
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      let chroma_h_phase = if center_sited {
        YUV422P_CENTERED_H_PHASE
      } else {
        0.0
      };
      // Only the colour tiers reconstruct full-width chroma for the centered
      // decode; a luma-only centered row bins native Y unchanged (siting is a
      // chroma-only property).
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      let want_color =
        rgb.is_some() || rgba.is_some() || hsv.is_some() || rgb_u16.is_some() || rgba_u16.is_some();
      // Whether this call carries any output — the EXACT set the tier preflight
      // tests. The route (and the siting phase) freezes only on an output-bearing
      // row a tier ACCEPTS; a no-output call consumes no stream state, so it must
      // not freeze.
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      let need_output = luma.is_some()
        || rgb.is_some()
        || rgba.is_some()
        || rgb_u16.is_some()
        || rgba_u16.is_some()
        || hsv.is_some();
      // Freeze the effective 4:2:2 chroma siting on the first output-bearing row
      // (mirrors the `frozen_native_route` freeze below). This CHECK is at the
      // always-compiled choke point every tier passes through; the matching SET
      // rides each tier's accept path (never before dispatch, so a rejected row
      // leaves it unset for a corrected retry). A later row observing a different
      // phase would bin a mixture of co-sited and centered chroma, so it is
      // rejected here before any reconstruction.
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      if need_output
        && let Some(frozen) = *frozen_chroma_centered
        && frozen != center_sited
      {
        return Err(MixedSinkerError::ChromaSitingChanged(
          ChromaSitingChanged::new(idx),
        ));
      }
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
            reserve_pn_chroma_full_u16(chroma_full_u16, w, h)?;
            let uv_full = upsample_pn_chroma_center_h::<BITS>(chroma_full_u16, uv_half, w, BE);
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
            if r.is_ok() && need_output && frozen_chroma_centered.is_none() {
              *frozen_chroma_centered = Some(center_sited);
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
        }
        return r;
      }
      // Reject a mid-frame native/row-stage route flip BEFORE either tier's
      // dispatch (the two tiers carry independent, in-order, once-only stream
      // state). CHECKED here and frozen below ONLY on an output-bearing row a
      // tier ACCEPTS — both gate on `need_output`. (Mirrors the 4:2:0 high-bit
      // semi-planar `p0xx`.)
      // The RFC #238 splice stage. A filter plan already returned above, so
      // `area_plan` is true and the selector reproduces the former `*native`
      // boolean bit-for-bit (`cfg!` is true wherever this block compiles).
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      let take_native = matches!(
        select_insertion_point(
          AveragingDomain::Encoded,
          InsertionContext {
            native_eligible: cfg!(all(feature = "yuv-semi-planar", feature = "yuv-planar")),
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
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      if take_native {
        // RFC #238 S5b point-of-use siting invalidation, mirroring the planar
        // Yuv422p native arm: a reused sink's cached join is only `reset` between
        // frames, so a frame whose `chroma_location` moved to a different phase
        // must REBUILD it. Drop the stale-phase join ONLY on the in-sequence
        // first row of a fresh frame (`idx == 0`, `next_y() == 0`) so a mid-frame
        // / out-of-sequence row rejects against the INTACT join and a corrected
        // retry rebuilds cleanly; a luma-only join carries no chroma phase and is
        // never dropped. Move it OUT (the delegate builds the replacement into
        // the field, untouched until every pre-feed allocation succeeds) and
        // restore the intact prior-phase join on a rejected rebuild so the row
        // mutates no join state.
        let stale_native = idx == 0
          && native_planar_u16.as_ref().is_some_and(|join| {
            join.chroma_phase_centered() == Some(!center_sited) && join.next_y() == 0
          });
        let prev_native = if stale_native {
          native_planar_u16.take()
        } else {
          None
        };
        // Dispatch first; freeze the route + siting ONLY after the call returns
        // Ok on an output-bearing row.
        let native_result = p2xx_process_native::<BITS, BE, false>(
          plan,
          native_planar_u16,
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
          *native_planar_u16 = prev_native;
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
      // Row-stage area tail. Same CHECK-before / SET-after split. Centered colour
      // reconstructs full-width interleaved `u16` chroma (de-interleave +
      // phase-0.5 upsample) and decodes 4:4:4 — but ONLY after the resample
      // preflight (frozen-output + sequence), so an out-of-sequence / rejected
      // row is caught before the chroma reservation (#180). A luma-only centered
      // row stays on the co-sited arm (which only bins luma).
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
        reserve_pn_chroma_full_u16(chroma_full_u16, w, h)?;
        let uv_full = upsample_pn_chroma_center_h::<BITS>(chroma_full_u16, uv_half, w, BE);
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
        if frozen_native_route.is_none() && need_output {
          *frozen_native_route = Some(false);
        }
        if frozen_chroma_centered.is_none() && need_output {
          *frozen_chroma_centered = Some(center_sited);
        }
        return Ok(());
      }
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
      }
      return Ok(());
    }

    // Resolve the output set up front: the no-output guard below must return
    // before the `idx * w` row-offset math (and before the centered preflight),
    // and the atomicity preflight needs the full want_* set anyway.
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();
    let want_rgb_u16 = rgb_u16.is_some();
    let want_rgba_u16 = rgba_u16.is_some();

    // No-output guard (#302): a `process` call with NO output attached never ran
    // an attach-time `w x h` validation, so on a 32-bit target an absurd geometry
    // could overflow the `idx * w` offset below. Returning HERE — before that
    // arithmetic AND before the centered chroma preflight — keeps a no-output row
    // panic-free and allocation-free.
    let need_output =
      luma.is_some() || want_rgb || want_rgba || want_hsv || want_rgb_u16 || want_rgba_u16;
    if !need_output {
      return Ok(());
    }

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // Chroma siting (#302): the centered horizontal sitings reconstruct chroma at
    // the phase-0.5 position; the default / co-sited path keeps the byte-identical
    // fused decode (the fused P-format kernels de-interleave + upsample chroma
    // in-register, exactly as before).
    #[cfg(feature = "yuv-planar")]
    let center_sited = chroma_422_center_sited_h(chroma_location);

    // Per-frame chroma-siting freeze (RFC #238, mirroring the resample-path guard
    // + the Nv20 sibling's direct-path freeze in this file): the first
    // output-bearing row pins the phase; a later row whose siting flipped would
    // decode a mixture of centered and co-sited chroma into ONE frame, so reject
    // it here BEFORE any scratch reserve or output write. `begin_frame`'s
    // `reset_high_bit_yuv_streams` clears the freeze so the next frame may pick
    // either phase.
    #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
    if need_output
      && let Some(frozen) = *frozen_chroma_centered
      && frozen != center_sited
    {
      return Err(MixedSinkerError::ChromaSitingChanged(
        ChromaSitingChanged::new(idx),
      ));
    }

    // Atomicity preflight (#302 / #308, cf. the crate's #180 resample fix and the
    // high-bit 4:2:0 `p0xx` sibling): reserve EVERY fallible row scratch this
    // identity row can touch BEFORE any output row is written (the luma plane
    // below, then the u16 / u8 RGB / RGBA / HSV fan-out), so an allocator refusal
    // returns a typed `AllocationFailed` leaving the output frame untouched
    // rather than partially mutated. Two scratches can grow:
    //  1. the centered-siting full-width interleaved `u16` chroma
    //     (`chroma_full_u16`), needed by ANY colour output (u8 OR u16 RGB / RGBA
    //     / HSV); and
    //  2. the u8 RGB row buffer, reached exactly when a colour decode needs an
    //     RGB row but no caller RGB buffer is borrowable — `want_hsv && want_rgba
    //     && !want_rgb` (`rgb_row_buf_or_scratch`'s own scratch arm; HSV-only
    //     routes through the direct `p010_to_hsv_row` kernel, no RGB scratch).
    // The later `upsample_pn_chroma_center_h` / `rgb_row_buf_or_scratch` calls
    // reuse the already-sized buffers, so the default path is byte-identical; only
    // the failure-path ordering changes. The u16 RGB / RGBA outputs write straight
    // into their caller buffers (the rgb_u16 plane stages the rgba_u16 expand) and
    // never grow a scratch; P210 exposes no luma_u16 output. A luma-only row
    // consumes no centered chroma, so it neither reserves nor upsamples it.
    #[cfg(feature = "yuv-planar")]
    let need_centered_chroma =
      center_sited && (want_rgb || want_rgba || want_hsv || want_rgb_u16 || want_rgba_u16);
    #[cfg(feature = "yuv-planar")]
    if need_centered_chroma {
      reserve_pn_chroma_full_u16(chroma_full_u16, w, h)?;
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

    // Centered full-width INTERLEAVED chroma (phase-0.5), reconstructed ONCE per
    // row from the wire-format half-width interleaved UV and reused by every
    // colour decode (u16 and u8). Infallible — the scratch was reserved above.
    // 4:2:2 reuses the 4:2:0 `p0xx` staging verbatim: the horizontal 2:1 upsample
    // is identical; only the vertical subsample differs (the walker hands each
    // luma row its own chroma row). The default left/unspecified siting leaves it
    // `None`, so the fused P-format kernels de-interleave + upsample in-register
    // and the output stays byte-identical.
    #[cfg(feature = "yuv-planar")]
    let centered: Option<&[u16]> = if need_centered_chroma {
      Some(upsample_pn_chroma_center_h::<BITS>(
        chroma_full_u16,
        row.uv_half(),
        w,
        BE,
      ))
    } else {
      None
    };
    #[cfg(not(feature = "yuv-planar"))]
    let centered: Option<&[u16]> = None;

    // Freeze the phase on the first output-bearing row — AFTER the fallible
    // scratch reserves above have succeeded, so an `AllocationFailed` row stays
    // retryable (frozen stays unset); later rows are checked against it up top.
    // The remaining fallible ops below are geometry / bounds checks, deterministic
    // regardless of siting.
    #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
    if need_output && frozen_chroma_centered.is_none() {
      *frozen_chroma_centered = Some(center_sited);
    }

    if let Some(luma) = luma.as_deref_mut() {
      let dst = &mut luma[one_plane_start..one_plane_end];
      for (d, &s) in dst.iter_mut().zip(row.y().iter()) {
        // Normalize BE-encoded wire bytes to host-native before the
        // luma downshift — see P010 luma path for rationale.
        let logical = if BE { u16::from_be(s) } else { u16::from_le(s) };
        *d = (logical >> 8) as u8;
      }
    }

    // ===== u16 RGB / RGBA path (Strategy A) =====
    // u16 outputs are low-bit-packed (yuv420p10le convention), not
    // P210's high-bit packing.
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
    // kernel (no source-width RGB scratch). The per-row chroma contract of
    // 4:2:2 matches 4:2:0's half-width interleaved UV, so the P210 row
    // kernel IS the P010 one; the 4:2:0-vs-4:2:2 difference is purely
    // vertical, resolved by the walker. When RGB or RGBA is also attached
    // the RGB kernel runs anyway, so HSV derives off that buffer for free
    // and `need_rgb_kernel` keeps it alive.
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

// ---- P212 impl ----------------------------------------------------------
//
// 4:2:2 high-bit-packed semi-planar (12-bit). Reuses `p012_to_rgb_*`
// row primitives — only the walker reads chroma row `r` not `r / 2`.

impl<'a, R, const BE: bool> MixedSinker<'a, P212<BE>, R> {
  /// Attaches a packed **`u16`** RGB output buffer. 12-bit
  /// **low-bit-packed** output.
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

  /// Attaches a packed **8‑bit** RGBA output buffer. The 12‑bit P212
  /// source (semi‑planar, high‑bit‑packed) is converted to 8‑bit RGBA
  /// via the `BITS = 12` Q15 kernel family; alpha = `0xFF` (P212 has
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
  /// **low‑bit‑packed** 12‑bit values (`yuv420p12le` convention) — not
  /// P212 high‑bit packing. Length is measured in `u16` **elements**
  /// (`width x height x 4`). Alpha element is `(1 << 12) - 1`.
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

impl<R, const BE: bool> P212Sink<BE> for MixedSinker<'_, P212<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, P212<BE>, R> {
  type Input<'r> = P212Row<'r>;
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

  fn process(&mut self, row: P212Row<'_>) -> Result<(), Self::Error> {
    // P212 stores 12‑bit samples high‑bit‑packed; bit depth is fixed
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
    // `Copy`, so read it before the field split-borrow below. Gated like its
    // only consumer (`chroma_422_center_sited_h` + the 4:4:4 P-format kernels
    // need `yuv-planar`); a semi-planar-only build keeps the default decode.
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
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      native,
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      native_planar_u16,
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      p0xx_y_half,
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      p0xx_u_half,
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      p0xx_v_half,
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      frozen_native_route,
      // Full-width interleaved chroma staging for the centered-siting (#302)
      // identity decode; reuses the 4:2:0 `p0xx` scratch + wrappers verbatim.
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      chroma_full_u16,
      // RFC #238 S5b: the 4:2:2 chroma siting phase frozen on the first output
      // row (the `u16` semi-planar twin of the planar Yuv422p `frozen_chroma_centered`).
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      frozen_chroma_centered,
      ..
    } = self;

    // Non-identity plan: filter branches first (no native fast tier); an area
    // plan routes native-or-row-stage. See the P210 impl for the full
    // rationale — P212 is identical bar the 12-bit kernel family
    // (`p012_to_rgb*`).
    if let Some(plan) = plan.as_ref() {
      let matrix = row.matrix();
      let full_range = row.full_range();
      let (y, uv_half) = (row.y(), row.uv_half());
      // RFC #238 S5b — 4:2:2 horizontal chroma siting for the high-bit
      // semi-planar P212, mirroring the planar Yuv422p twin (S5a) on the
      // semi-planar de-interleave (S2a). The centered group
      // (`chroma_422_center_sited_h`) samples chroma at `+0.25` chroma-sample;
      // the co-sited / unspecified group is phase 0 (byte-identical). The native
      // fast tier folds the phase into the `area_chroma_422` chroma weights; the
      // filter and row-stage tiers reconstruct full-width interleaved `u16`
      // chroma (de-interleave + phase-0.5 upsample into `chroma_full_u16`) and
      // decode 4:4:4 via the `p412` full-chroma kernels — the co-sited arms keep
      // the fused `p012` half-chroma decode.
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      let center_sited = chroma_422_center_sited_h(chroma_location);
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      let chroma_h_phase = if center_sited {
        YUV422P_CENTERED_H_PHASE
      } else {
        0.0
      };
      // Only the colour tiers reconstruct full-width chroma for the centered
      // decode; a luma-only centered row bins native Y unchanged (siting is a
      // chroma-only property).
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      let want_color =
        rgb.is_some() || rgba.is_some() || hsv.is_some() || rgb_u16.is_some() || rgba_u16.is_some();
      // Whether this call carries any output — the EXACT set the tier preflight
      // tests. The route (and the siting phase) freezes only on an output-bearing
      // row a tier ACCEPTS; a no-output call consumes no stream state, so it must
      // not freeze.
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      let need_output = luma.is_some()
        || rgb.is_some()
        || rgba.is_some()
        || rgb_u16.is_some()
        || rgba_u16.is_some()
        || hsv.is_some();
      // Freeze the effective 4:2:2 chroma siting on the first output-bearing row
      // (mirrors the `frozen_native_route` freeze below). This CHECK is at the
      // always-compiled choke point every tier passes through; the matching SET
      // rides each tier's accept path (never before dispatch, so a rejected row
      // leaves it unset for a corrected retry). A later row observing a different
      // phase would bin a mixture of co-sited and centered chroma, so it is
      // rejected here before any reconstruction.
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      if need_output
        && let Some(frozen) = *frozen_chroma_centered
        && frozen != center_sited
      {
        return Err(MixedSinkerError::ChromaSitingChanged(
          ChromaSitingChanged::new(idx),
        ));
      }
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
            reserve_pn_chroma_full_u16(chroma_full_u16, w, h)?;
            let uv_full = upsample_pn_chroma_center_h::<BITS>(chroma_full_u16, uv_half, w, BE);
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
            if r.is_ok() && need_output && frozen_chroma_centered.is_none() {
              *frozen_chroma_centered = Some(center_sited);
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
        }
        return r;
      }
      // The RFC #238 splice stage. A filter plan already returned above, so
      // `area_plan` is true and the selector reproduces the former `*native`
      // boolean bit-for-bit (`cfg!` is true wherever this block compiles).
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      let take_native = matches!(
        select_insertion_point(
          AveragingDomain::Encoded,
          InsertionContext {
            native_eligible: cfg!(all(feature = "yuv-semi-planar", feature = "yuv-planar")),
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
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      if take_native {
        // RFC #238 S5b point-of-use siting invalidation, mirroring the planar
        // Yuv422p native arm: a reused sink's cached join is only `reset` between
        // frames, so a frame whose `chroma_location` moved to a different phase
        // must REBUILD it. Drop the stale-phase join ONLY on the in-sequence
        // first row of a fresh frame (`idx == 0`, `next_y() == 0`) so a mid-frame
        // / out-of-sequence row rejects against the INTACT join and a corrected
        // retry rebuilds cleanly; a luma-only join carries no chroma phase and is
        // never dropped. Move it OUT (the delegate builds the replacement into
        // the field, untouched until every pre-feed allocation succeeds) and
        // restore the intact prior-phase join on a rejected rebuild so the row
        // mutates no join state.
        let stale_native = idx == 0
          && native_planar_u16.as_ref().is_some_and(|join| {
            join.chroma_phase_centered() == Some(!center_sited) && join.next_y() == 0
          });
        let prev_native = if stale_native {
          native_planar_u16.take()
        } else {
          None
        };
        // Dispatch first; freeze the route + siting ONLY after the call returns
        // Ok on an output-bearing row.
        let native_result = p2xx_process_native::<BITS, BE, false>(
          plan,
          native_planar_u16,
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
          *native_planar_u16 = prev_native;
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
      // Row-stage area tail. Same CHECK-before / SET-after split. Centered colour
      // reconstructs full-width interleaved `u16` chroma (de-interleave +
      // phase-0.5 upsample) and decodes 4:4:4 — but ONLY after the resample
      // preflight (frozen-output + sequence), so an out-of-sequence / rejected
      // row is caught before the chroma reservation (#180). A luma-only centered
      // row stays on the co-sited arm (which only bins luma).
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
        reserve_pn_chroma_full_u16(chroma_full_u16, w, h)?;
        let uv_full = upsample_pn_chroma_center_h::<BITS>(chroma_full_u16, uv_half, w, BE);
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
        if frozen_native_route.is_none() && need_output {
          *frozen_native_route = Some(false);
        }
        if frozen_chroma_centered.is_none() && need_output {
          *frozen_chroma_centered = Some(center_sited);
        }
        return Ok(());
      }
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
      }
      return Ok(());
    }

    // Resolve the output set up front: the no-output guard below must return
    // before the `idx * w` row-offset math (and before the centered preflight),
    // and the atomicity preflight needs the full want_* set anyway.
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();
    let want_rgb_u16 = rgb_u16.is_some();
    let want_rgba_u16 = rgba_u16.is_some();

    // No-output guard (#302): a `process` call with NO output attached never ran
    // an attach-time `w x h` validation, so on a 32-bit target an absurd geometry
    // could overflow the `idx * w` offset below. Returning HERE — before that
    // arithmetic AND before the centered chroma preflight — keeps a no-output row
    // panic-free and allocation-free.
    let need_output =
      luma.is_some() || want_rgb || want_rgba || want_hsv || want_rgb_u16 || want_rgba_u16;
    if !need_output {
      return Ok(());
    }

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // Chroma siting (#302): the centered horizontal sitings reconstruct chroma at
    // the phase-0.5 position; the default / co-sited path keeps the byte-identical
    // fused decode (the fused P-format kernels de-interleave + upsample chroma
    // in-register, exactly as before).
    #[cfg(feature = "yuv-planar")]
    let center_sited = chroma_422_center_sited_h(chroma_location);

    // Per-frame chroma-siting freeze (RFC #238, mirroring the resample-path guard
    // + the Nv20 sibling's direct-path freeze in this file): the first
    // output-bearing row pins the phase; a later row whose siting flipped would
    // decode a mixture of centered and co-sited chroma into ONE frame, so reject
    // it here BEFORE any scratch reserve or output write. `begin_frame`'s
    // `reset_high_bit_yuv_streams` clears the freeze so the next frame may pick
    // either phase.
    #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
    if need_output
      && let Some(frozen) = *frozen_chroma_centered
      && frozen != center_sited
    {
      return Err(MixedSinkerError::ChromaSitingChanged(
        ChromaSitingChanged::new(idx),
      ));
    }

    // Atomicity preflight (#302 / #308, cf. the crate's #180 resample fix and the
    // high-bit 4:2:0 `p0xx` sibling): reserve EVERY fallible row scratch this
    // identity row can touch BEFORE any output row is written (the luma plane
    // below, then the u16 / u8 RGB / RGBA / HSV fan-out), so an allocator refusal
    // returns a typed `AllocationFailed` leaving the output frame untouched
    // rather than partially mutated. Two scratches can grow:
    //  1. the centered-siting full-width interleaved `u16` chroma
    //     (`chroma_full_u16`), needed by ANY colour output (u8 OR u16 RGB / RGBA
    //     / HSV); and
    //  2. the u8 RGB row buffer, reached exactly when a colour decode needs an
    //     RGB row but no caller RGB buffer is borrowable — `want_hsv && want_rgba
    //     && !want_rgb` (`rgb_row_buf_or_scratch`'s own scratch arm; HSV-only
    //     routes through the direct `p012_to_hsv_row` kernel, no RGB scratch).
    // The later `upsample_pn_chroma_center_h` / `rgb_row_buf_or_scratch` calls
    // reuse the already-sized buffers, so the default path is byte-identical; only
    // the failure-path ordering changes. The u16 RGB / RGBA outputs write straight
    // into their caller buffers (the rgb_u16 plane stages the rgba_u16 expand) and
    // never grow a scratch; P212 exposes no luma_u16 output. A luma-only row
    // consumes no centered chroma, so it neither reserves nor upsamples it.
    #[cfg(feature = "yuv-planar")]
    let need_centered_chroma =
      center_sited && (want_rgb || want_rgba || want_hsv || want_rgb_u16 || want_rgba_u16);
    #[cfg(feature = "yuv-planar")]
    if need_centered_chroma {
      reserve_pn_chroma_full_u16(chroma_full_u16, w, h)?;
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

    // Centered full-width INTERLEAVED chroma (phase-0.5), reconstructed ONCE per
    // row from the wire-format half-width interleaved UV and reused by every
    // colour decode (u16 and u8). Infallible — the scratch was reserved above.
    // 4:2:2 reuses the 4:2:0 `p0xx` staging verbatim: the horizontal 2:1 upsample
    // is identical; only the vertical subsample differs (the walker hands each
    // luma row its own chroma row). The default left/unspecified siting leaves it
    // `None`, so the fused P-format kernels de-interleave + upsample in-register
    // and the output stays byte-identical.
    #[cfg(feature = "yuv-planar")]
    let centered: Option<&[u16]> = if need_centered_chroma {
      Some(upsample_pn_chroma_center_h::<BITS>(
        chroma_full_u16,
        row.uv_half(),
        w,
        BE,
      ))
    } else {
      None
    };
    #[cfg(not(feature = "yuv-planar"))]
    let centered: Option<&[u16]> = None;

    // Freeze the phase on the first output-bearing row — AFTER the fallible
    // scratch reserves above have succeeded, so an `AllocationFailed` row stays
    // retryable (frozen stays unset); later rows are checked against it up top.
    // The remaining fallible ops below are geometry / bounds checks, deterministic
    // regardless of siting.
    #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
    if need_output && frozen_chroma_centered.is_none() {
      *frozen_chroma_centered = Some(center_sited);
    }

    if let Some(luma) = luma.as_deref_mut() {
      let dst = &mut luma[one_plane_start..one_plane_end];
      for (d, &s) in dst.iter_mut().zip(row.y().iter()) {
        // Normalize BE-encoded wire bytes to host-native before the
        // luma downshift — see P010 luma path for rationale.
        let logical = if BE { u16::from_be(s) } else { u16::from_le(s) };
        *d = (logical >> 8) as u8;
      }
    }

    // ===== u16 RGB / RGBA path (Strategy A) =====
    // u16 outputs are low-bit-packed (yuv420p12le convention), not
    // P212's high-bit packing.
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
    // HSV-without-RGB-or-RGBA goes through the direct `p012_to_hsv_row`
    // kernel (no source-width RGB scratch). The per-row chroma contract of
    // 4:2:2 matches 4:2:0's half-width interleaved UV, so the P212 row
    // kernel IS the P012 one; the 4:2:0-vs-4:2:2 difference is purely
    // vertical, resolved by the walker. When RGB or RGBA is also attached
    // the RGB kernel runs anyway, so HSV derives off that buffer for free
    // and `need_rgb_kernel` keeps it alive.
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

// ---- P216 impl ----------------------------------------------------------
//
// 4:2:2 16-bit semi-planar. Reuses `p016_to_rgb_*` row primitives.

impl<'a, R, const BE: bool> MixedSinker<'a, P216<BE>, R> {
  /// Attaches a packed **`u16`** RGB output buffer. 16-bit output
  /// (full `[0, 65535]` range, every bit active).
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

  /// Attaches a packed **8‑bit** RGBA output buffer. The 16‑bit P216
  /// source (semi‑planar, 16 active bits) is converted to 8‑bit RGBA
  /// via the dedicated `BITS = 16` kernel family (i64 chroma multiply
  /// — not the BITS-generic Q15 pipeline); alpha = `0xFF` (P216 has
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

  /// Attaches a packed **`u16`** RGBA output buffer. Output covers the
  /// full `u16` range `[0, 65535]` (16 active bits). Length is
  /// measured in `u16` **elements** (`width x height x 4`). Alpha
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

impl<R, const BE: bool> P216Sink<BE> for MixedSinker<'_, P216<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, P216<BE>, R> {
  type Input<'r> = P216Row<'r>;
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

  fn process(&mut self, row: P216Row<'_>) -> Result<(), Self::Error> {
    // P216 is 16-bit semi-planar (every bit active); used for the u16
    // RGBA expand path's alpha pad (alpha = 0xFFFF).
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
    // `Copy`, so read it before the field split-borrow below. Gated like its
    // only consumer (`chroma_422_center_sited_h` + the 4:4:4 P-format kernels
    // need `yuv-planar`); a semi-planar-only build keeps the default decode.
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
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      native,
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      native_planar_u16,
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      p0xx_y_half,
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      p0xx_u_half,
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      p0xx_v_half,
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      frozen_native_route,
      // Full-width interleaved chroma staging for the centered-siting (#302)
      // identity decode; reuses the 4:2:0 `p0xx` scratch + wrappers verbatim.
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      chroma_full_u16,
      // RFC #238 S5b: the 4:2:2 chroma siting phase frozen on the first output
      // row (the `u16` semi-planar twin of the planar Yuv422p `frozen_chroma_centered`).
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      frozen_chroma_centered,
      ..
    } = self;

    // Non-identity plan: filter branches first (no native fast tier); an area
    // plan routes native-or-row-stage. See the P210 impl for the full
    // rationale. At 16 bits the Y de-pack shift `>> (16 - BITS)` is `>> 0`, and
    // the dedicated 16-bit kernel family (`p016_to_rgb*`) is used; the native
    // max is `u16::MAX`, so the native-depth clamp is a value no-op.
    if let Some(plan) = plan.as_ref() {
      let matrix = row.matrix();
      let full_range = row.full_range();
      let (y, uv_half) = (row.y(), row.uv_half());
      // RFC #238 S5b — 4:2:2 horizontal chroma siting for the high-bit
      // semi-planar P216, mirroring the planar Yuv422p twin (S5a) on the
      // semi-planar de-interleave (S2a). The centered group
      // (`chroma_422_center_sited_h`) samples chroma at `+0.25` chroma-sample;
      // the co-sited / unspecified group is phase 0 (byte-identical). The native
      // fast tier folds the phase into the `area_chroma_422` chroma weights; the
      // filter and row-stage tiers reconstruct full-width interleaved `u16`
      // chroma (de-interleave + phase-0.5 upsample into `chroma_full_u16`) and
      // decode 4:4:4 via the `p416` full-chroma kernels — the co-sited arms keep
      // the fused `p016` half-chroma decode.
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      let center_sited = chroma_422_center_sited_h(chroma_location);
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      let chroma_h_phase = if center_sited {
        YUV422P_CENTERED_H_PHASE
      } else {
        0.0
      };
      // Only the colour tiers reconstruct full-width chroma for the centered
      // decode; a luma-only centered row bins native Y unchanged (siting is a
      // chroma-only property).
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      let want_color =
        rgb.is_some() || rgba.is_some() || hsv.is_some() || rgb_u16.is_some() || rgba_u16.is_some();
      // Whether this call carries any output — the EXACT set the tier preflight
      // tests. The route (and the siting phase) freezes only on an output-bearing
      // row a tier ACCEPTS; a no-output call consumes no stream state, so it must
      // not freeze.
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      let need_output = luma.is_some()
        || rgb.is_some()
        || rgba.is_some()
        || rgb_u16.is_some()
        || rgba_u16.is_some()
        || hsv.is_some();
      // Freeze the effective 4:2:2 chroma siting on the first output-bearing row
      // (mirrors the `frozen_native_route` freeze below). This CHECK is at the
      // always-compiled choke point every tier passes through; the matching SET
      // rides each tier's accept path (never before dispatch, so a rejected row
      // leaves it unset for a corrected retry). A later row observing a different
      // phase would bin a mixture of co-sited and centered chroma, so it is
      // rejected here before any reconstruction.
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      if need_output
        && let Some(frozen) = *frozen_chroma_centered
        && frozen != center_sited
      {
        return Err(MixedSinkerError::ChromaSitingChanged(
          ChromaSitingChanged::new(idx),
        ));
      }
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
            reserve_pn_chroma_full_u16(chroma_full_u16, w, h)?;
            let uv_full = upsample_pn_chroma_center_h::<BITS>(chroma_full_u16, uv_half, w, BE);
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
            if r.is_ok() && need_output && frozen_chroma_centered.is_none() {
              *frozen_chroma_centered = Some(center_sited);
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
        }
        return r;
      }
      // The RFC #238 splice stage. A filter plan already returned above, so
      // `area_plan` is true and the selector reproduces the former `*native`
      // boolean bit-for-bit (`cfg!` is true wherever this block compiles).
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      let take_native = matches!(
        select_insertion_point(
          AveragingDomain::Encoded,
          InsertionContext {
            native_eligible: cfg!(all(feature = "yuv-semi-planar", feature = "yuv-planar")),
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
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      if take_native {
        // RFC #238 S5b point-of-use siting invalidation, mirroring the planar
        // Yuv422p native arm: a reused sink's cached join is only `reset` between
        // frames, so a frame whose `chroma_location` moved to a different phase
        // must REBUILD it. Drop the stale-phase join ONLY on the in-sequence
        // first row of a fresh frame (`idx == 0`, `next_y() == 0`) so a mid-frame
        // / out-of-sequence row rejects against the INTACT join and a corrected
        // retry rebuilds cleanly; a luma-only join carries no chroma phase and is
        // never dropped. Move it OUT (the delegate builds the replacement into
        // the field, untouched until every pre-feed allocation succeeds) and
        // restore the intact prior-phase join on a rejected rebuild so the row
        // mutates no join state.
        let stale_native = idx == 0
          && native_planar_u16.as_ref().is_some_and(|join| {
            join.chroma_phase_centered() == Some(!center_sited) && join.next_y() == 0
          });
        let prev_native = if stale_native {
          native_planar_u16.take()
        } else {
          None
        };
        // Dispatch first; freeze the route + siting ONLY after the call returns
        // Ok on an output-bearing row.
        let native_result = p2xx_process_native::<BITS, BE, false>(
          plan,
          native_planar_u16,
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
          *native_planar_u16 = prev_native;
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
      // Row-stage area tail. Same CHECK-before / SET-after split. Centered colour
      // reconstructs full-width interleaved `u16` chroma (de-interleave +
      // phase-0.5 upsample) and decodes 4:4:4 — but ONLY after the resample
      // preflight (frozen-output + sequence), so an out-of-sequence / rejected
      // row is caught before the chroma reservation (#180). A luma-only centered
      // row stays on the co-sited arm (which only bins luma).
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
        reserve_pn_chroma_full_u16(chroma_full_u16, w, h)?;
        let uv_full = upsample_pn_chroma_center_h::<BITS>(chroma_full_u16, uv_half, w, BE);
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
        if frozen_native_route.is_none() && need_output {
          *frozen_native_route = Some(false);
        }
        if frozen_chroma_centered.is_none() && need_output {
          *frozen_chroma_centered = Some(center_sited);
        }
        return Ok(());
      }
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
      }
      return Ok(());
    }

    // Resolve the output set up front: the no-output guard below must return
    // before the `idx * w` row-offset math (and before the centered preflight),
    // and the atomicity preflight needs the full want_* set anyway.
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();
    let want_rgb_u16 = rgb_u16.is_some();
    let want_rgba_u16 = rgba_u16.is_some();

    // No-output guard (#302): a `process` call with NO output attached never ran
    // an attach-time `w x h` validation, so on a 32-bit target an absurd geometry
    // could overflow the `idx * w` offset below. Returning HERE — before that
    // arithmetic AND before the centered chroma preflight — keeps a no-output row
    // panic-free and allocation-free.
    let need_output =
      luma.is_some() || want_rgb || want_rgba || want_hsv || want_rgb_u16 || want_rgba_u16;
    if !need_output {
      return Ok(());
    }

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // Chroma siting (#302): the centered horizontal sitings reconstruct chroma at
    // the phase-0.5 position; the default / co-sited path keeps the byte-identical
    // fused decode (the fused P-format kernels de-interleave + upsample chroma
    // in-register, exactly as before).
    #[cfg(feature = "yuv-planar")]
    let center_sited = chroma_422_center_sited_h(chroma_location);

    // Per-frame chroma-siting freeze (RFC #238, mirroring the resample-path guard
    // + the Nv20 sibling's direct-path freeze in this file): the first
    // output-bearing row pins the phase; a later row whose siting flipped would
    // decode a mixture of centered and co-sited chroma into ONE frame, so reject
    // it here BEFORE any scratch reserve or output write. `begin_frame`'s
    // `reset_high_bit_yuv_streams` clears the freeze so the next frame may pick
    // either phase.
    #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
    if need_output
      && let Some(frozen) = *frozen_chroma_centered
      && frozen != center_sited
    {
      return Err(MixedSinkerError::ChromaSitingChanged(
        ChromaSitingChanged::new(idx),
      ));
    }

    // Atomicity preflight (#302 / #308, cf. the crate's #180 resample fix and the
    // high-bit 4:2:0 `p0xx` sibling): reserve EVERY fallible row scratch this
    // identity row can touch BEFORE any output row is written (the luma plane
    // below, then the u16 / u8 RGB / RGBA / HSV fan-out), so an allocator refusal
    // returns a typed `AllocationFailed` leaving the output frame untouched
    // rather than partially mutated. Two scratches can grow:
    //  1. the centered-siting full-width interleaved `u16` chroma
    //     (`chroma_full_u16`), needed by ANY colour output (u8 OR u16 RGB / RGBA
    //     / HSV); and
    //  2. the u8 RGB row buffer, reached exactly when a colour decode needs an
    //     RGB row but no caller RGB buffer is borrowable — `want_hsv && want_rgba
    //     && !want_rgb` (`rgb_row_buf_or_scratch`'s own scratch arm; HSV-only
    //     routes through the direct `p016_to_hsv_row` kernel, no RGB scratch).
    // The later `upsample_pn_chroma_center_h` / `rgb_row_buf_or_scratch` calls
    // reuse the already-sized buffers, so the default path is byte-identical; only
    // the failure-path ordering changes. The u16 RGB / RGBA outputs write straight
    // into their caller buffers (the rgb_u16 plane stages the rgba_u16 expand) and
    // never grow a scratch; P216 exposes no luma_u16 output. A luma-only row
    // consumes no centered chroma, so it neither reserves nor upsamples it.
    #[cfg(feature = "yuv-planar")]
    let need_centered_chroma =
      center_sited && (want_rgb || want_rgba || want_hsv || want_rgb_u16 || want_rgba_u16);
    #[cfg(feature = "yuv-planar")]
    if need_centered_chroma {
      reserve_pn_chroma_full_u16(chroma_full_u16, w, h)?;
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

    // Centered full-width INTERLEAVED chroma (phase-0.5), reconstructed ONCE per
    // row from the wire-format half-width interleaved UV and reused by every
    // colour decode (u16 and u8). Infallible — the scratch was reserved above.
    // 4:2:2 reuses the 4:2:0 `p0xx` staging verbatim: the horizontal 2:1 upsample
    // is identical; only the vertical subsample differs (the walker hands each
    // luma row its own chroma row). The default left/unspecified siting leaves it
    // `None`, so the fused P-format kernels de-interleave + upsample in-register
    // and the output stays byte-identical.
    #[cfg(feature = "yuv-planar")]
    let centered: Option<&[u16]> = if need_centered_chroma {
      Some(upsample_pn_chroma_center_h::<BITS>(
        chroma_full_u16,
        row.uv_half(),
        w,
        BE,
      ))
    } else {
      None
    };
    #[cfg(not(feature = "yuv-planar"))]
    let centered: Option<&[u16]> = None;

    // Freeze the phase on the first output-bearing row — AFTER the fallible
    // scratch reserves above have succeeded, so an `AllocationFailed` row stays
    // retryable (frozen stays unset); later rows are checked against it up top.
    // The remaining fallible ops below are geometry / bounds checks, deterministic
    // regardless of siting.
    #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
    if need_output && frozen_chroma_centered.is_none() {
      *frozen_chroma_centered = Some(center_sited);
    }

    // 16-bit Y >> 8 is the top byte (all bits active).
    if let Some(luma) = luma.as_deref_mut() {
      let dst = &mut luma[one_plane_start..one_plane_end];
      for (d, &s) in dst.iter_mut().zip(row.y().iter()) {
        // Normalize BE-encoded wire bytes to host-native before the
        // luma downshift — see P010 luma path for rationale.
        let logical = if BE { u16::from_be(s) } else { u16::from_le(s) };
        *d = (logical >> 8) as u8;
      }
    }

    // ===== u16 RGB / RGBA path (Strategy A) =====
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
    // HSV-without-RGB-or-RGBA goes through the direct `p016_to_hsv_row`
    // kernel (no source-width RGB scratch). The per-row chroma contract of
    // 4:2:2 matches 4:2:0's half-width interleaved UV, so the P216 row
    // kernel IS the P016 one; the 4:2:0-vs-4:2:2 difference is purely
    // vertical, resolved by the walker. When RGB or RGBA is also attached
    // the RGB kernel runs anyway, so HSV derives off that buffer for free
    // and `need_rgb_kernel` keeps it alive.
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

// ---- NV20 impl ----------------------------------------------------------
//
// 4:2:2 LOW-bit-packed semi-planar (10-bit) — the low-bit twin of P210.
// Same per-row plane shape (full-width Y + half-width interleaved UV at
// full height) and same `u16` element type; the ONLY difference is the
// per-`u16` bit alignment: NV20's 10 active bits live in the LOW 10
// (`& 0x03FF`) vs P210's high 10 (`>> 6`). Routes through the dedicated
// low-bit `nv20_to_rgb_*` row primitives (the `LOW_PACKED = true`
// monomorphization of the shared Pn kernel family) and the native fast
// tier with `LOW_PACKED = true`; everything else mirrors the P210 impl
// (chroma row `r` per Y row `r`, native luma = de-packed Y narrowed to
// 8 bits, no `luma_u16` output).

impl<'a, R, const BE: bool> MixedSinker<'a, Nv20<BE>, R> {
  /// Attaches a packed **`u16`** RGB output buffer. 10-bit
  /// **low-bit-packed** output (yuv420p10le convention — matching NV20's
  /// own low-bit packing). Length is in `u16` elements: `width x height x 3`.
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

  /// Attaches a packed **8‑bit** RGBA output buffer. The 10‑bit NV20
  /// source (semi‑planar, low‑bit‑packed) is converted to 8‑bit RGBA
  /// via the `BITS = 10` Q15 kernel family; alpha = `0xFF` (NV20 has
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
  /// **low‑bit‑packed** 10‑bit values (`yuv420p10le` convention).
  /// Length is measured in `u16` **elements** (`width x height x 4`).
  /// Alpha element is `(1 << 10) - 1`.
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

impl<R, const BE: bool> Nv20Sink<BE> for MixedSinker<'_, Nv20<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, Nv20<BE>, R> {
  type Input<'r> = Nv20Row<'r>;
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

  fn process(&mut self, row: Nv20Row<'_>) -> Result<(), Self::Error> {
    // NV20 stores 10‑bit samples LOW‑bit‑packed; bit depth is fixed by
    // the format. Used for the u16 RGBA expand path's alpha pad.
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
    if row.uv().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::UvHalf10,
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

    // Chroma siting (RFC #238): drives the identity-plan centered-vs-cosited
    // horizontal chroma decode. Read before the field destructure (which
    // reborrows `self`). `yuv-planar` gates the reconstruction (the predicate
    // + the low-packed 4:4:4 kernels); a semi-planar-only build keeps the
    // default fused decode.
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
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      native,
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      native_planar_u16,
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      p0xx_y_half,
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      p0xx_u_half,
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      p0xx_v_half,
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      frozen_native_route,
      // Full-width interleaved chroma staging for the centered-siting (RFC
      // #238) identity decode; reuses the 4:2:0 `p0xx` scratch + the
      // low-packed upsample wrapper.
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      chroma_full_u16,
      // The identity-plan chroma siting phase frozen on the first output row,
      // so a mid-frame `set_chroma_location` flip is rejected instead of
      // producing a mixed-phase frame (the `u16` semi-planar twin of the
      // planar Yuv422p `frozen_chroma_centered`). Reset each frame by
      // `reset_high_bit_yuv_streams`.
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      frozen_chroma_centered,
      ..
    } = self;

    // Non-identity plan. Identical structure to the P210 impl (see there for
    // the full rationale); the only differences are the LOW-bit de-pack
    // (`& 0x03FF` instead of `>> 6`), the low-bit `nv20_to_rgb*` kernels, and
    // the native tier's `LOW_PACKED = true`.
    if let Some(plan) = plan.as_ref() {
      let matrix = row.matrix();
      let full_range = row.full_range();
      let (y, uv_half) = (row.y(), row.uv());
      // RFC #238 — 4:2:2 horizontal chroma siting for the low-packed
      // semi-planar Nv20, mirroring the high-packed P210 twin on the low-packed
      // de-interleave. The centered group (`chroma_422_center_sited_h`) samples
      // chroma at `+0.25` chroma-sample; the co-sited / unspecified group is
      // phase 0 (byte-identical). The native fast tier folds the phase into the
      // `area_chroma_422` chroma weights; the filter and row-stage tiers
      // reconstruct full-width interleaved `u16` chroma (de-interleave +
      // phase-0.5 upsample into `chroma_full_u16`) and decode 4:4:4 via the
      // low-packed `nv20_444_*` full-chroma kernels — the co-sited arms keep the
      // fused `nv20_*` half-chroma decode.
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      let center_sited = chroma_422_center_sited_h(chroma_location);
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      let chroma_h_phase = if center_sited {
        YUV422P_CENTERED_H_PHASE
      } else {
        0.0
      };
      // Only the colour tiers reconstruct full-width chroma for the centered
      // decode; a luma-only centered row bins native Y unchanged (siting is a
      // chroma-only property).
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      let want_color =
        rgb.is_some() || rgba.is_some() || hsv.is_some() || rgb_u16.is_some() || rgba_u16.is_some();
      // Whether this call carries any output — the EXACT set the tier preflight
      // tests. The route (and the siting phase) freezes only on an output-bearing
      // row a tier ACCEPTS; a no-output call consumes no stream state, so it must
      // not freeze.
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      let need_output = luma.is_some()
        || rgb.is_some()
        || rgba.is_some()
        || rgb_u16.is_some()
        || rgba_u16.is_some()
        || hsv.is_some();
      // Freeze the effective 4:2:2 chroma siting on the first output-bearing row
      // (mirrors the `frozen_native_route` freeze below). This CHECK is at the
      // always-compiled choke point every tier passes through; the matching SET
      // rides each tier's accept path (never before dispatch, so a rejected row
      // leaves it unset for a corrected retry). A later row observing a different
      // phase would bin a mixture of co-sited and centered chroma, so it is
      // rejected here before any reconstruction.
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      if need_output
        && let Some(frozen) = *frozen_chroma_centered
        && frozen != center_sited
      {
        return Err(MixedSinkerError::ChromaSitingChanged(
          ChromaSitingChanged::new(idx),
        ));
      }
      if plan.kind().is_filter() {
        // Centered filter reconstructs full-width chroma and decodes 4:4:4, but
        // ONLY after the resample preflight (frozen-output + sequence), so an
        // out-of-sequence / rejected row is caught before the chroma reservation
        // (#180). `packed_yuv422_triple_filter_resample` re-runs the idempotent
        // preflight. Co-sited keeps the fused `nv20_*` half-chroma decode.
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
            reserve_pn_chroma_full_u16(chroma_full_u16, w, h)?;
            let uv_full =
              upsample_pn_chroma_center_h_low_packed::<BITS>(chroma_full_u16, uv_half, w, BE);
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
                  *dst = logical & ((1u16 << BITS) - 1);
                }
              },
              |scratch| {
                nv20_444_to_rgb_row_endian(y, uv_full, scratch, w, matrix, full_range, use_simd, BE)
              },
              |scratch| {
                nv20_444_to_rgb_u16_row_endian(
                  y, uv_full, scratch, w, matrix, full_range, use_simd, BE,
                )
              },
            );
            if r.is_ok() && need_output && frozen_chroma_centered.is_none() {
              *frozen_chroma_centered = Some(center_sited);
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
              *dst = logical & ((1u16 << BITS) - 1);
            }
          },
          |scratch| {
            nv20_to_rgb_row_endian(y, uv_half, scratch, w, matrix, full_range, use_simd, BE)
          },
          |scratch| {
            nv20_to_rgb_u16_row_endian(y, uv_half, scratch, w, matrix, full_range, use_simd, BE)
          },
        );
        #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
        if r.is_ok() && need_output && frozen_chroma_centered.is_none() {
          *frozen_chroma_centered = Some(center_sited);
        }
        return r;
      }
      // The RFC #238 splice stage — see the P210 impl.
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      let take_native = matches!(
        select_insertion_point(
          AveragingDomain::Encoded,
          InsertionContext {
            native_eligible: cfg!(all(feature = "yuv-semi-planar", feature = "yuv-planar")),
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
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      if take_native {
        // NV20's native fast tier de-packs with `LOW_PACKED = true` and folds the
        // horizontal chroma-siting `chroma_h_phase` into the `area_chroma_422`
        // weights (co-sited phase 0 stays byte-identical to the plain `area`
        // plan). A reused sink's cached join is only `reset` between frames, so a
        // frame whose `chroma_location` moved to a different phase must REBUILD
        // it. Drop the stale-phase join ONLY on the in-sequence first row of a
        // fresh frame (`idx == 0`, `next_y() == 0`) so a mid-frame /
        // out-of-sequence row rejects against the INTACT join and a corrected
        // retry rebuilds cleanly; a luma-only join carries no chroma phase and is
        // never dropped. Move it OUT (the delegate builds the replacement into
        // the field, untouched until every pre-feed allocation succeeds) and
        // restore the intact prior-phase join on a rejected rebuild so the row
        // mutates no join state.
        let stale_native = idx == 0
          && native_planar_u16.as_ref().is_some_and(|join| {
            join.chroma_phase_centered() == Some(!center_sited) && join.next_y() == 0
          });
        let prev_native = if stale_native {
          native_planar_u16.take()
        } else {
          None
        };
        // Dispatch first; freeze the route + siting ONLY after the call returns
        // Ok on an output-bearing row.
        let native_result = p2xx_process_native::<BITS, BE, true>(
          plan,
          native_planar_u16,
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
          *native_planar_u16 = prev_native;
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
      // Row-stage area tail. Same CHECK-before / SET-after split. Centered colour
      // reconstructs full-width interleaved `u16` chroma (de-interleave +
      // phase-0.5 upsample) and decodes 4:4:4 — but ONLY after the resample
      // preflight (frozen-output + sequence), so an out-of-sequence / rejected
      // row is caught before the chroma reservation (#180). A luma-only centered
      // row stays on the co-sited arm (which only bins luma).
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
        reserve_pn_chroma_full_u16(chroma_full_u16, w, h)?;
        let uv_full =
          upsample_pn_chroma_center_h_low_packed::<BITS>(chroma_full_u16, uv_half, w, BE);
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
              *dst = logical & ((1u16 << BITS) - 1);
            }
          },
          |scratch| {
            nv20_444_to_rgb_row_endian(y, uv_full, scratch, w, matrix, full_range, use_simd, BE)
          },
          |scratch| {
            nv20_444_to_rgb_u16_row_endian(y, uv_full, scratch, w, matrix, full_range, use_simd, BE)
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
            *dst = logical & ((1u16 << BITS) - 1);
          }
        },
        |scratch| nv20_to_rgb_row_endian(y, uv_half, scratch, w, matrix, full_range, use_simd, BE),
        |scratch| {
          nv20_to_rgb_u16_row_endian(y, uv_half, scratch, w, matrix, full_range, use_simd, BE)
        },
      )?;
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      if frozen_native_route.is_none() && need_output {
        *frozen_native_route = Some(false);
      }
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      if frozen_chroma_centered.is_none() && need_output {
        *frozen_chroma_centered = Some(center_sited);
      }
      return Ok(());
    }

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // Resolve the output set up front so the atomicity preflight below runs
    // before any output row is written.
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();
    // u16 outputs are low-bit-packed (yuv420p10le convention).
    let want_rgb_u16 = rgb_u16.is_some();
    let want_rgba_u16 = rgba_u16.is_some();
    // The full output set (matching the P210 identity `need_output`, INCLUDING
    // the u16 twins): gates the chroma-siting freeze so a luma-only / no-output
    // row neither freezes nor rejects. Only the centered path consumes it, so
    // it carries the same `yuv-planar` gate (a semi-planar-solo build has no
    // centered decode and never reads it).
    #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
    let need_output =
      luma.is_some() || want_rgb || want_rgba || want_hsv || want_rgb_u16 || want_rgba_u16;

    // Centered chroma siting (RFC #238): when the horizontal siting is
    // centered, reconstruct a full-width interleaved chroma plane ONCE (the
    // low-bit-packed 4:2:2 twin of the P210 identity path) and decode via the
    // `nv20_444_*` kernels; the default / co-sited path keeps the
    // byte-identical fused `nv20_*` decode. Reserve the scratch up front
    // (atomicity, before any output write); 4:2:2 is horizontal-only (no
    // vertical / `Bottom` phase). `yuv-planar` gates both halves (the
    // predicate + the low-packed 4:4:4 kernels).
    #[cfg(feature = "yuv-planar")]
    let center_sited = chroma_422_center_sited_h(chroma_location);

    // Per-frame chroma-siting freeze (RFC #238, mirroring the resample-path
    // guard + the planar Yuv422p direct-path freeze): the first output-bearing
    // row pins the phase; a later row whose siting flipped would decode a
    // mixture of centered and co-sited chroma into ONE frame, so reject it here
    // BEFORE any scratch reserve or output write. `begin_frame`'s
    // `reset_high_bit_yuv_streams` clears the freeze so the next frame may pick
    // either phase.
    #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
    if need_output
      && let Some(frozen) = *frozen_chroma_centered
      && frozen != center_sited
    {
      return Err(MixedSinkerError::ChromaSitingChanged(
        ChromaSitingChanged::new(idx),
      ));
    }

    #[cfg(feature = "yuv-planar")]
    let need_centered_chroma =
      center_sited && (want_rgb || want_rgba || want_hsv || want_rgb_u16 || want_rgba_u16);
    #[cfg(feature = "yuv-planar")]
    if need_centered_chroma {
      reserve_pn_chroma_full_u16(chroma_full_u16, w, h)?;
    }

    // Atomicity preflight (#308, cf. the crate's #180 resample fix and the
    // high-bit 4:2:0 sibling): reserve the only growable row scratch this
    // identity row can touch — the u8 RGB row buffer — BEFORE any output row
    // is written (the luma plane below, then the u16 RGB / RGBA fan-out), so an
    // allocator refusal returns a typed `AllocationFailed` leaving the output
    // frame untouched rather than partially mutated. The u16 RGB / RGBA outputs
    // need no preflight: they write straight into their caller buffers (the
    // rgb_u16 plane itself stages the rgba_u16 expand) and never grow a scratch;
    // NV20 exposes no luma_u16 output. UNLIKE the high-bit P-formats NV20 has NO
    // direct YUV→HSV kernel, so EVERY HSV-wanting row without a caller RGB
    // buffer grows `rgb_row_buf_or_scratch`'s allocating (rgb=None) arm — it is
    // reached at `want_hsv && !want_rgb`, independent of rgba (whereas the
    // P-formats route HSV-only through a direct kernel and gate on
    // `want_hsv && want_rgba && !want_rgb`). The later decode reuses the
    // already-sized buffer, so the default path is byte-identical; only the
    // failure-path ordering changes.
    if want_hsv && !want_rgb {
      rgb_row_buf_or_scratch(
        rgb.as_deref_mut(),
        rgb_scratch,
        one_plane_start,
        one_plane_end,
        w,
        h,
      )?;
    }

    // Centered full-width INTERLEAVED chroma (phase-0.5, low-bit-packed),
    // reconstructed ONCE per row from the wire half-width interleaved UV and
    // reused by every colour decode (u16 + u8). Infallible — the scratch was
    // reserved above. The default left/unspecified siting leaves it `None`, so
    // the fused `nv20_*` kernels de-interleave + upsample in-register and the
    // output stays byte-identical to the co-sited decode.
    #[cfg(feature = "yuv-planar")]
    let centered: Option<&[u16]> = if need_centered_chroma {
      Some(upsample_pn_chroma_center_h_low_packed::<BITS>(
        chroma_full_u16,
        row.uv(),
        w,
        BE,
      ))
    } else {
      None
    };
    #[cfg(not(feature = "yuv-planar"))]
    let centered: Option<&[u16]> = None;

    // Freeze the phase on the first output-bearing row — AFTER the fallible
    // scratch reserves above have succeeded, so an `AllocationFailed` row stays
    // retryable (frozen stays unset); later rows are checked against it up top.
    // The remaining fallible ops below are geometry / bounds checks,
    // deterministic regardless of siting.
    #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
    if need_output && frozen_chroma_centered.is_none() {
      *frozen_chroma_centered = Some(center_sited);
    }

    if let Some(luma) = luma.as_deref_mut() {
      let dst = &mut luma[one_plane_start..one_plane_end];
      // NV20 native luma: de-pack the low 10 then narrow to 8 bits
      // (`(logical & 0x03FF) >> 2`) — the `>> 8` top-byte shortcut the
      // high-bit P-formats use does NOT apply here.
      for (d, &s) in dst.iter_mut().zip(row.y().iter()) {
        let logical = if BE { u16::from_be(s) } else { u16::from_le(s) };
        *d = ((logical & ((1u16 << BITS) - 1)) >> (BITS - 8)) as u8;
      }
    }

    // ===== u16 RGB / RGBA path (Strategy A) =====
    if want_rgba_u16 && !want_rgb_u16 {
      let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row =
        rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
      if let Some(uv_full) = centered {
        nv20_444_to_rgba_u16_row_endian(
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
        nv20_to_rgba_u16_row_endian(
          row.y(),
          row.uv(),
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
        nv20_444_to_rgb_u16_row_endian(
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
        nv20_to_rgb_u16_row_endian(
          row.y(),
          row.uv(),
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
    let need_rgb_kernel = want_rgb || want_hsv;

    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      if let Some(uv_full) = centered {
        nv20_444_to_rgba_row_endian(
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
        nv20_to_rgba_row_endian(
          row.y(),
          row.uv(),
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

    // Centered → decode from the reconstructed full-width chroma; HSV then
    // derives off this RGB row (NV20 has no direct low-packed YUV→HSV kernel).
    if let Some(uv_full) = centered {
      nv20_444_to_rgb_row_endian(
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
      nv20_to_rgb_row_endian(
        row.y(),
        row.uv(),
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
