use super::super::{
  GeometryOverflow, InsufficientBuffer, MixedSinker, MixedSinkerError, RowIndexOutOfRange,
  RowShapeMismatch, RowSlice, check_dimensions_match, packed_yuv444_triple_filter_resample,
  packed_yuv444_triple_resample, reset_high_bit_yuv_streams, rgb_row_buf_or_scratch,
  rgba_plane_row_slice, rgba_u16_plane_row_slice,
};
use crate::{PixelSink, row::*, source::*};

// `NativeRouteChanged` is raised only by the native fast tier's route-flip
// guard, which exists only when the reused planar join is compiled in.
#[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
use super::super::{
  FrozenOutputs, HsvFrameMut, NativePlanarYuvU16, NativeRouteChanged,
  native_planar_hb_preflight_check_only, yuv_planar16_process_native,
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
// scratch grow in `p4xx_process_native`. Armed, the FIRST (Y) scratch grow of
// an output-bearing row returns the crate's recoverable `AllocationFailed`
// WITHOUT growing — so the atomicity regressions can prove the join's pre-feed
// preflight runs BEFORE this fallible grow. Mirrors the 4:2:0 high-bit
// semi-planar `FORCE_P0XX_ALLOC_FAILURE`. Strictly test-only.
#[cfg(all(
  test,
  feature = "std",
  feature = "yuv-semi-planar",
  feature = "yuv-planar"
))]
std::thread_local! {
  static FORCE_P4XX_ALLOC_FAILURE: core::cell::Cell<bool> =
    const { core::cell::Cell::new(false) };
}

/// Arms the wrapper de-pack scratch allocation failpoint for the **next**
/// output-bearing high-bit semi-planar 4:4:4 native row on the current thread.
/// The flag is consumed (take-on-read) by the first fallible scratch grow that
/// row reaches, so it fires exactly once and cannot leak into a later test.
/// Test-only.
#[cfg(all(
  test,
  feature = "std",
  feature = "yuv-semi-planar",
  feature = "yuv-planar"
))]
pub(crate) fn arm_p4xx_alloc_failure() {
  FORCE_P4XX_ALLOC_FAILURE.with(|f| f.set(true));
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
    if fail && FORCE_P4XX_ALLOC_FAILURE.with(|f| f.take()) {
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

/// Native fast-tier decimator for the **high-bit semi-planar 4:4:4** P-format
/// family ([`P410`](crate::source::P410) / [`P412`](crate::source::P412) /
/// [`P416`](crate::source::P416)): bins the native Y / U / V planes straight to
/// the output grid and converts once per output row at output resolution. The
/// 4:4:4 sibling of the 4:2:0 high-bit semi-planar `p0xx_process_native` and
/// the `u16` twin of the 8-bit semi-planar non-4:2:0
/// `semi_planar_process_native_non420`, reusing the high-bit non-4:2:0 PLANAR
/// join verbatim ([`yuv_planar16_process_native`]) after de-interleaving +
/// DE-PACKING the wire row into wrapper-owned host-native LOGICAL u16 scratch.
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
/// 4:4:4 layout: the chroma plane is full-resolution (`chroma_w = w`, vertical
/// cadence `chroma_vsub = 1`), so a chroma row feeds EVERY colour Y row and the
/// chroma de-pack runs at full width. The packed UV row is `2 * w` u16 (`w`
/// interleaved pairs). The delegate builds its chroma grid against the same
/// output geometry via the `build_chroma_plan` closure.
///
/// Atomicity + lazy chroma mirror the 4:2:2 `p2xx_process_native`: the join's
/// COMPLETE pre-feed preflight runs FIRST (deterministic typed rejection, never
/// `AllocationFailed`); a luma-only sink skips the chroma de-interleave/scratch.
#[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
#[allow(clippy::too_many_arguments)]
fn p4xx_process_native<const BITS: u32, const BE: bool>(
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
  uv_full: &[u16],
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
      "BITS must be in (8, 16] for high-bit semi-planar 4:4:4 P-format"
    )
  };
  let need_luma = luma.is_some();
  let need_color =
    rgb.is_some() || rgba.is_some() || hsv.is_some() || rgb_u16.is_some() || rgba_u16.is_some();
  // 4:4:4 chroma is full-resolution: `chroma_w = w`, a chroma row per Y row.
  let cw = w;

  // Run the planar join's COMPLETE compare-only pre-feed rejection preflight
  // FIRST — BEFORE any fallible scratch grow below — so every rejection returns
  // its deterministic typed error and leaves the wrapper scratch untouched.
  // Compare-only (no output-set freeze): the de-pack grows below AND the
  // delegate's own colour / source scratch grows all stay pre-commit, and the
  // delegate performs the SINGLE freeze once every pre-feed allocation has
  // succeeded — so a LATER grow (here or in the delegate) failing leaves
  // `resample_outputs` untouched and the row retryable.
  if !native_planar_hb_preflight_check_only(
    native_planar_u16,
    resample_outputs,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    luma,
    // The high-bit semi-planar 4:4:4 P-format exposes no `luma_u16` output.
    &None,
    hsv,
    idx,
    need_luma,
    need_color,
  )? {
    return Ok(());
  }

  // Grow the wrapper de-pack scratch under the planner's recoverable contract —
  // Y always, U / V only on a colour row (4:4:4: every Y row is a chroma row
  // when colour is wanted). All grows precede the infallible de-pack and the
  // delegate call. The failpoint fires on the FIRST (Y) grow only.
  grow_depack_scratch(y_scratch, w, true, w, h, plan)?;
  if need_color {
    grow_depack_scratch(u_scratch, cw, false, w, h, plan)?;
    grow_depack_scratch(v_scratch, cw, false, w, h, plan)?;
  }

  // De-pack the wire planes into host-native LOGICAL scratch. Decode the wire
  // endianness, then shift the active high `BITS` down to the low `BITS`
  // (`>> (16 - BITS)`; `>> 0` at BITS = 16). Everything past here is infallible.
  for (d, &s) in y_scratch[..w].iter_mut().zip(y_row.iter()) {
    let logical = if BE { u16::from_be(s) } else { u16::from_le(s) };
    *d = logical >> (16 - BITS);
  }
  if need_color {
    // P-format chroma is interleaved `U,V,U,V…` (U at even element); each of U
    // and V is independently high-bit-packed and must be de-packed. 4:4:4 is
    // full-width: `w` interleaved pairs (`2 * w` u16).
    for (i, pair) in uv_full.chunks_exact(2).enumerate() {
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

  // Delegate to the planar high-bit non-4:2:0 join with `BE = HOST_NATIVE_BE`
  // at the 4:4:4 chroma geometry (`chroma_vsub = 1`, `chroma_w = w`). Empty
  // U / V on luma-only rows (the join reads chroma only under colour).
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
    // The high-bit semi-planar 4:4:4 P-format exposes no `luma_u16` output.
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
    || ResamplePlan::area(cw, h, plan.out_w(), plan.out_h()),
    use_simd,
  )
}

// ---- P410 impl ----------------------------------------------------------
//
// 4:4:4 high-bit-packed semi-planar (10-bit). Full-width interleaved
// UV (`2 * width` u16 elements per row). Uses the new
// `p410_to_rgb_*` row primitives (which dispatch to the
// `p_n_444_to_rgb_*<10>` family).

impl<'a, R, const BE: bool> MixedSinker<'a, P410<BE>, R> {
  /// Attaches a packed **`u16`** RGB output buffer. 10-bit low-packed.
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

  /// Attaches a packed **8-bit** RGBA output buffer. P410 has no alpha
  /// plane, so alpha = `0xFF` (opaque).
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

  /// Attaches a packed **`u16`** RGBA output buffer. 10-bit low-packed
  /// (`[0, 1023]`); alpha element is `1023`.
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

impl<R, const BE: bool> P410Sink<BE> for MixedSinker<'_, P410<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, P410<BE>, R> {
  type Input<'r> = P410Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    reset_high_bit_yuv_streams(self);
    Ok(())
  }

  fn process(&mut self, row: P410Row<'_>) -> Result<(), Self::Error> {
    const BITS: u32 = 10;
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Y10,
        idx,
        w,
        row.y().len(),
      )));
    }
    // 4:4:4 semi-planar: full-width x 2 elements per pair.
    if row.uv_full().len() != 2 * w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::UvFull10,
        idx,
        2 * w,
        row.uv_full().len(),
      )));
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange(
        RowIndexOutOfRange::new(idx, self.height),
      ));
    }

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
      ..
    } = self;

    // Non-identity plan: a `Filter` plan routes to the shared high-bit 4:4:4
    // signed-coefficient filter tail (no native fast tier), so it branches
    // FIRST. An `Area` plan routes native-or-row-stage: when the native tier
    // is enabled (and the planar join it reuses is compiled in), bin the
    // native Y / U / V planes at output resolution and convert once per output
    // row, de-interleaving + de-packing the P410 chroma + Y into wrapper-owned
    // logical scratch first; otherwise (or under `with_native(false)`) feed
    // the shared area triple-resample tail. P410 is semi-planar 4:4:4: the
    // full-width interleaved UV plane (`2 * w` u16) is de-interleaved per-pixel
    // (no chroma upsample) by the `p410_to_rgb*` kernels. The Y de-pack shift
    // `>> (16 - BITS)` yields the logical native Y; `luma = binned_Y >>
    // (BITS - 8)`. P410 exposes no `luma_u16`, so it is `&mut None`. The filter
    // tail clamps a signed-kernel overshoot to the native max for this
    // sub-16-bit source (both colour and native-Y luma), matching the in-range
    // area path.
    if let Some(plan) = plan.as_ref() {
      let matrix = row.matrix();
      let full_range = row.full_range();
      let (y, uv_full) = (row.y(), row.uv_full());
      if plan.kind().is_filter() {
        return packed_yuv444_triple_filter_resample::<BITS>(
          rgb_filter_stream,
          rgb_filter_stream_u16,
          luma_filter_stream_u16,
          resample_outputs,
          rgb,
          rgba,
          rgb_u16,
          rgba_u16,
          luma,
          &mut None,
          hsv,
          rgb_scratch,
          rgb_scratch_u16,
          luma_scratch_u16,
          w,
          plan,
          idx,
          use_simd,
          matrix,
          full_range,
          |scratch| {
            p410_to_rgb_row_endian(y, uv_full, scratch, w, matrix, full_range, use_simd, BE)
          },
          |scratch| {
            p410_to_rgb_u16_row_endian(y, uv_full, scratch, w, matrix, full_range, use_simd, BE)
          },
          |scratch| {
            for (dst, &s) in scratch[..w].iter_mut().zip(y.iter()) {
              let logical = if BE { u16::from_be(s) } else { u16::from_le(s) };
              *dst = logical >> (16 - BITS);
            }
          },
        );
      }
      // Whether this call carries any output — the EXACT set the tier
      // preflight tests. The route freezes only on an output-bearing row a
      // tier ACCEPTS.
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      let need_output = luma.is_some()
        || rgb.is_some()
        || rgba.is_some()
        || rgb_u16.is_some()
        || rgba_u16.is_some()
        || hsv.is_some();
      // Reject a mid-frame native/row-stage route flip BEFORE either tier's
      // dispatch (mirrors the 4:2:0 high-bit semi-planar `p0xx`).
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
        p4xx_process_native::<BITS, BE>(
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
          uv_full,
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
      packed_yuv444_triple_resample::<BITS>(
        rgb_stream,
        rgb_stream_u16,
        luma_stream_u16,
        resample_outputs,
        rgb,
        rgba,
        rgb_u16,
        rgba_u16,
        luma,
        &mut None,
        hsv,
        rgb_scratch,
        rgb_scratch_u16,
        luma_scratch_u16,
        w,
        plan,
        idx,
        use_simd,
        matrix,
        full_range,
        |scratch| p410_to_rgb_row_endian(y, uv_full, scratch, w, matrix, full_range, use_simd, BE),
        |scratch| {
          p410_to_rgb_u16_row_endian(y, uv_full, scratch, w, matrix, full_range, use_simd, BE)
        },
        |scratch| {
          for (dst, &s) in scratch[..w].iter_mut().zip(y.iter()) {
            let logical = if BE { u16::from_be(s) } else { u16::from_le(s) };
            *dst = logical >> (16 - BITS);
          }
        },
      )?;
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      if frozen_native_route.is_none() && need_output {
        *frozen_native_route = Some(false);
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

    // Atomicity preflight (#308, cf. the crate's #180 resample fix and the
    // high-bit 4:2:0 p0xx / yuv420p siblings): reserve the only growable row
    // scratch this identity row can touch — the u8 RGB row buffer — BEFORE any
    // output row is written (the luma plane below, then the u16 RGB / RGBA
    // fan-out), so an allocator refusal returns a typed `AllocationFailed` and
    // leaves the output frame untouched rather than partially mutated. At 4:4:4
    // the chroma is full-resolution (no upsampling), but the output staging
    // matches 4:2:0: the u16 RGB / RGBA outputs write straight into their caller
    // buffers (the rgb_u16 plane itself stages the rgba_u16 expand) and never
    // grow a scratch, and these formats expose no luma_u16. The allocating
    // (rgb=None) arm of `rgb_row_buf_or_scratch` is reached exactly when a colour
    // decode needs an RGB row but no caller RGB buffer is borrowable —
    // `want_hsv && want_rgba && !want_rgb`. The later decode reuses the
    // already-sized buffer, so the default path is byte-identical; only the
    // failure-path ordering changes.
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
      let dst = &mut luma[one_plane_start..one_plane_end];
      for (d, &s) in dst.iter_mut().zip(row.y().iter()) {
        // Normalize BE-encoded wire bytes to host-native before the
        // luma downshift — see P010 luma path for rationale.
        let logical = if BE { u16::from_be(s) } else { u16::from_le(s) };
        *d = (logical >> 8) as u8;
      }
    }

    // ===== u16 RGB / RGBA path (Strategy A) =====
    let want_rgb_u16 = rgb_u16.is_some();
    let want_rgba_u16 = rgba_u16.is_some();

    if want_rgba_u16 && !want_rgb_u16 {
      let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row =
        rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
      p410_to_rgba_u16_row_endian(
        row.y(),
        row.uv_full(),
        rgba_u16_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
        BE,
      );
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
      p410_to_rgb_u16_row_endian(
        row.y(),
        row.uv_full(),
        rgb_u16_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
        BE,
      );
      if want_rgba_u16 {
        let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
        let rgba_u16_row =
          rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
        expand_rgb_u16_to_rgba_u16_row::<BITS>(rgb_u16_row, rgba_u16_row, w);
      }
    }

    // ===== u8 RGB / RGBA / HSV path (Strategy A) =====
    // HSV-without-RGB-or-RGBA goes through the direct `p410_to_hsv_row`
    // kernel (no source-width RGB scratch) — the 4:4:4 full-width-UV twin
    // of the P010 path. When RGB or RGBA is also attached the RGB kernel
    // runs anyway, so HSV derives off that buffer for free and
    // `need_rgb_kernel` keeps it alive.
    let want_hsv_direct = want_hsv && !want_rgb && !want_rgba;
    let need_rgb_kernel = want_rgb || (want_hsv && want_rgba);

    if want_hsv_direct {
      let hsv = hsv.as_mut().expect("want_hsv_direct implies hsv attached");
      let (h, s, v) = hsv.hsv();
      p410_to_hsv_row_endian(
        row.y(),
        row.uv_full(),
        &mut h[one_plane_start..one_plane_end],
        &mut s[one_plane_start..one_plane_end],
        &mut v[one_plane_start..one_plane_end],
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
        BE,
      );
      return Ok(());
    }

    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      p410_to_rgba_row_endian(
        row.y(),
        row.uv_full(),
        rgba_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
        BE,
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

    p410_to_rgb_row_endian(
      row.y(),
      row.uv_full(),
      rgb_row,
      w,
      row.matrix(),
      row.full_range(),
      use_simd,
      BE,
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

// ---- P412 impl ----------------------------------------------------------

impl<'a, R, const BE: bool> MixedSinker<'a, P412<BE>, R> {
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

  /// Attaches a packed **8-bit** RGBA output buffer. P412 has no alpha
  /// plane, so alpha = `0xFF` (opaque).
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

  /// Attaches a packed **`u16`** RGBA output buffer. 12-bit low-packed
  /// (`[0, 4095]`); alpha element is `4095`.
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

impl<R, const BE: bool> P412Sink<BE> for MixedSinker<'_, P412<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, P412<BE>, R> {
  type Input<'r> = P412Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    reset_high_bit_yuv_streams(self);
    Ok(())
  }

  fn process(&mut self, row: P412Row<'_>) -> Result<(), Self::Error> {
    const BITS: u32 = 12;
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Y12,
        idx,
        w,
        row.y().len(),
      )));
    }
    if row.uv_full().len() != 2 * w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::UvFull12,
        idx,
        2 * w,
        row.uv_full().len(),
      )));
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange(
        RowIndexOutOfRange::new(idx, self.height),
      ));
    }

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
      ..
    } = self;

    // Non-identity plan: filter branches first (no native fast tier); an area
    // plan routes native-or-row-stage. See the P410 impl for the full
    // rationale — P412 is identical bar the 12-bit kernel family
    // (`p412_to_rgb*`).
    if let Some(plan) = plan.as_ref() {
      let matrix = row.matrix();
      let full_range = row.full_range();
      let (y, uv_full) = (row.y(), row.uv_full());
      if plan.kind().is_filter() {
        return packed_yuv444_triple_filter_resample::<BITS>(
          rgb_filter_stream,
          rgb_filter_stream_u16,
          luma_filter_stream_u16,
          resample_outputs,
          rgb,
          rgba,
          rgb_u16,
          rgba_u16,
          luma,
          &mut None,
          hsv,
          rgb_scratch,
          rgb_scratch_u16,
          luma_scratch_u16,
          w,
          plan,
          idx,
          use_simd,
          matrix,
          full_range,
          |scratch| {
            p412_to_rgb_row_endian(y, uv_full, scratch, w, matrix, full_range, use_simd, BE)
          },
          |scratch| {
            p412_to_rgb_u16_row_endian(y, uv_full, scratch, w, matrix, full_range, use_simd, BE)
          },
          |scratch| {
            for (dst, &s) in scratch[..w].iter_mut().zip(y.iter()) {
              let logical = if BE { u16::from_be(s) } else { u16::from_le(s) };
              *dst = logical >> (16 - BITS);
            }
          },
        );
      }
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      let need_output = luma.is_some()
        || rgb.is_some()
        || rgba.is_some()
        || rgb_u16.is_some()
        || rgba_u16.is_some()
        || hsv.is_some();
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
        p4xx_process_native::<BITS, BE>(
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
          uv_full,
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
      packed_yuv444_triple_resample::<BITS>(
        rgb_stream,
        rgb_stream_u16,
        luma_stream_u16,
        resample_outputs,
        rgb,
        rgba,
        rgb_u16,
        rgba_u16,
        luma,
        &mut None,
        hsv,
        rgb_scratch,
        rgb_scratch_u16,
        luma_scratch_u16,
        w,
        plan,
        idx,
        use_simd,
        matrix,
        full_range,
        |scratch| p412_to_rgb_row_endian(y, uv_full, scratch, w, matrix, full_range, use_simd, BE),
        |scratch| {
          p412_to_rgb_u16_row_endian(y, uv_full, scratch, w, matrix, full_range, use_simd, BE)
        },
        |scratch| {
          for (dst, &s) in scratch[..w].iter_mut().zip(y.iter()) {
            let logical = if BE { u16::from_be(s) } else { u16::from_le(s) };
            *dst = logical >> (16 - BITS);
          }
        },
      )?;
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      if frozen_native_route.is_none() && need_output {
        *frozen_native_route = Some(false);
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

    // Atomicity preflight (#308, cf. the crate's #180 resample fix and the
    // high-bit 4:2:0 p0xx / yuv420p siblings): reserve the only growable row
    // scratch this identity row can touch — the u8 RGB row buffer — BEFORE any
    // output row is written (the luma plane below, then the u16 RGB / RGBA
    // fan-out), so an allocator refusal returns a typed `AllocationFailed` and
    // leaves the output frame untouched rather than partially mutated. At 4:4:4
    // the chroma is full-resolution (no upsampling), but the output staging
    // matches 4:2:0: the u16 RGB / RGBA outputs write straight into their caller
    // buffers (the rgb_u16 plane itself stages the rgba_u16 expand) and never
    // grow a scratch, and these formats expose no luma_u16. The allocating
    // (rgb=None) arm of `rgb_row_buf_or_scratch` is reached exactly when a colour
    // decode needs an RGB row but no caller RGB buffer is borrowable —
    // `want_hsv && want_rgba && !want_rgb`. The later decode reuses the
    // already-sized buffer, so the default path is byte-identical; only the
    // failure-path ordering changes.
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
      let dst = &mut luma[one_plane_start..one_plane_end];
      for (d, &s) in dst.iter_mut().zip(row.y().iter()) {
        // Normalize BE-encoded wire bytes to host-native before the
        // luma downshift — see P010 luma path for rationale.
        let logical = if BE { u16::from_be(s) } else { u16::from_le(s) };
        *d = (logical >> 8) as u8;
      }
    }

    // ===== u16 RGB / RGBA path (Strategy A) =====
    let want_rgb_u16 = rgb_u16.is_some();
    let want_rgba_u16 = rgba_u16.is_some();

    if want_rgba_u16 && !want_rgb_u16 {
      let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row =
        rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
      p412_to_rgba_u16_row_endian(
        row.y(),
        row.uv_full(),
        rgba_u16_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
        BE,
      );
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
      p412_to_rgb_u16_row_endian(
        row.y(),
        row.uv_full(),
        rgb_u16_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
        BE,
      );
      if want_rgba_u16 {
        let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
        let rgba_u16_row =
          rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
        expand_rgb_u16_to_rgba_u16_row::<BITS>(rgb_u16_row, rgba_u16_row, w);
      }
    }

    // ===== u8 RGB / RGBA / HSV path (Strategy A) =====
    // HSV-without-RGB-or-RGBA goes through the direct `p412_to_hsv_row`
    // kernel (no source-width RGB scratch) — the 4:4:4 full-width-UV twin
    // of the P012 path. When RGB or RGBA is also attached the RGB kernel
    // runs anyway, so HSV derives off that buffer for free and
    // `need_rgb_kernel` keeps it alive.
    let want_hsv_direct = want_hsv && !want_rgb && !want_rgba;
    let need_rgb_kernel = want_rgb || (want_hsv && want_rgba);

    if want_hsv_direct {
      let hsv = hsv.as_mut().expect("want_hsv_direct implies hsv attached");
      let (h, s, v) = hsv.hsv();
      p412_to_hsv_row_endian(
        row.y(),
        row.uv_full(),
        &mut h[one_plane_start..one_plane_end],
        &mut s[one_plane_start..one_plane_end],
        &mut v[one_plane_start..one_plane_end],
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
        BE,
      );
      return Ok(());
    }

    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      p412_to_rgba_row_endian(
        row.y(),
        row.uv_full(),
        rgba_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
        BE,
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

    p412_to_rgb_row_endian(
      row.y(),
      row.uv_full(),
      rgb_row,
      w,
      row.matrix(),
      row.full_range(),
      use_simd,
      BE,
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

// ---- P416 impl ----------------------------------------------------------
//
// 4:4:4 16-bit semi-planar. Uses `p416_to_rgb_*` (parallel i64-chroma
// family for u16 output, i32 for u8).

impl<'a, R, const BE: bool> MixedSinker<'a, P416<BE>, R> {
  /// Attaches a packed **`u16`** RGB output buffer. 16-bit output
  /// (full `[0, 65535]` range).
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

  /// Attaches a packed **8-bit** RGBA output buffer. P416 has no alpha
  /// plane, so alpha = `0xFF` (opaque).
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

  /// Attaches a packed **`u16`** RGBA output buffer. Full-range output
  /// `[0, 65535]`; alpha element is `0xFFFF`.
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

impl<R, const BE: bool> P416Sink<BE> for MixedSinker<'_, P416<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, P416<BE>, R> {
  type Input<'r> = P416Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    reset_high_bit_yuv_streams(self);
    Ok(())
  }

  fn process(&mut self, row: P416Row<'_>) -> Result<(), Self::Error> {
    const BITS: u32 = 16;
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Y16,
        idx,
        w,
        row.y().len(),
      )));
    }
    if row.uv_full().len() != 2 * w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::UvFull16,
        idx,
        2 * w,
        row.uv_full().len(),
      )));
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange(
        RowIndexOutOfRange::new(idx, self.height),
      ));
    }

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
      ..
    } = self;

    // Non-identity plan: filter branches first (no native fast tier); an area
    // plan routes native-or-row-stage. See the P410 impl for the full
    // rationale. At 16 bits the Y de-pack shift `>> (16 - BITS)` is `>> 0`, and
    // the dedicated 16-bit kernel family (`p416_to_rgb*`) is used; the native
    // max is `u16::MAX`, so the native-depth clamp is a value no-op.
    if let Some(plan) = plan.as_ref() {
      let matrix = row.matrix();
      let full_range = row.full_range();
      let (y, uv_full) = (row.y(), row.uv_full());
      if plan.kind().is_filter() {
        return packed_yuv444_triple_filter_resample::<BITS>(
          rgb_filter_stream,
          rgb_filter_stream_u16,
          luma_filter_stream_u16,
          resample_outputs,
          rgb,
          rgba,
          rgb_u16,
          rgba_u16,
          luma,
          &mut None,
          hsv,
          rgb_scratch,
          rgb_scratch_u16,
          luma_scratch_u16,
          w,
          plan,
          idx,
          use_simd,
          matrix,
          full_range,
          |scratch| {
            p416_to_rgb_row_endian(y, uv_full, scratch, w, matrix, full_range, use_simd, BE)
          },
          |scratch| {
            p416_to_rgb_u16_row_endian(y, uv_full, scratch, w, matrix, full_range, use_simd, BE)
          },
          |scratch| {
            for (dst, &s) in scratch[..w].iter_mut().zip(y.iter()) {
              let logical = if BE { u16::from_be(s) } else { u16::from_le(s) };
              *dst = logical >> (16 - BITS);
            }
          },
        );
      }
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      let need_output = luma.is_some()
        || rgb.is_some()
        || rgba.is_some()
        || rgb_u16.is_some()
        || rgba_u16.is_some()
        || hsv.is_some();
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
        p4xx_process_native::<BITS, BE>(
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
          uv_full,
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
      packed_yuv444_triple_resample::<BITS>(
        rgb_stream,
        rgb_stream_u16,
        luma_stream_u16,
        resample_outputs,
        rgb,
        rgba,
        rgb_u16,
        rgba_u16,
        luma,
        &mut None,
        hsv,
        rgb_scratch,
        rgb_scratch_u16,
        luma_scratch_u16,
        w,
        plan,
        idx,
        use_simd,
        matrix,
        full_range,
        |scratch| p416_to_rgb_row_endian(y, uv_full, scratch, w, matrix, full_range, use_simd, BE),
        |scratch| {
          p416_to_rgb_u16_row_endian(y, uv_full, scratch, w, matrix, full_range, use_simd, BE)
        },
        |scratch| {
          for (dst, &s) in scratch[..w].iter_mut().zip(y.iter()) {
            let logical = if BE { u16::from_be(s) } else { u16::from_le(s) };
            *dst = logical >> (16 - BITS);
          }
        },
      )?;
      #[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
      if frozen_native_route.is_none() && need_output {
        *frozen_native_route = Some(false);
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

    // Atomicity preflight (#308, cf. the crate's #180 resample fix and the
    // high-bit 4:2:0 p0xx / yuv420p siblings): reserve the only growable row
    // scratch this identity row can touch — the u8 RGB row buffer — BEFORE any
    // output row is written (the luma plane below, then the u16 RGB / RGBA
    // fan-out), so an allocator refusal returns a typed `AllocationFailed` and
    // leaves the output frame untouched rather than partially mutated. At 4:4:4
    // the chroma is full-resolution (no upsampling), but the output staging
    // matches 4:2:0: the u16 RGB / RGBA outputs write straight into their caller
    // buffers (the rgb_u16 plane itself stages the rgba_u16 expand) and never
    // grow a scratch, and these formats expose no luma_u16. The allocating
    // (rgb=None) arm of `rgb_row_buf_or_scratch` is reached exactly when a colour
    // decode needs an RGB row but no caller RGB buffer is borrowable —
    // `want_hsv && want_rgba && !want_rgb`. The later decode reuses the
    // already-sized buffer, so the default path is byte-identical; only the
    // failure-path ordering changes.
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
      let dst = &mut luma[one_plane_start..one_plane_end];
      for (d, &s) in dst.iter_mut().zip(row.y().iter()) {
        // Normalize BE-encoded wire bytes to host-native before the
        // luma downshift — see P010 luma path for rationale.
        let logical = if BE { u16::from_be(s) } else { u16::from_le(s) };
        *d = (logical >> 8) as u8;
      }
    }

    // ===== u16 RGB / RGBA path (Strategy A) =====
    let want_rgb_u16 = rgb_u16.is_some();
    let want_rgba_u16 = rgba_u16.is_some();

    if want_rgba_u16 && !want_rgb_u16 {
      let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row =
        rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
      p416_to_rgba_u16_row_endian(
        row.y(),
        row.uv_full(),
        rgba_u16_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
        BE,
      );
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
      p416_to_rgb_u16_row_endian(
        row.y(),
        row.uv_full(),
        rgb_u16_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
        BE,
      );
      if want_rgba_u16 {
        let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
        let rgba_u16_row =
          rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
        expand_rgb_u16_to_rgba_u16_row::<BITS>(rgb_u16_row, rgba_u16_row, w);
      }
    }

    // ===== u8 RGB / RGBA / HSV path (Strategy A) =====
    // HSV-without-RGB-or-RGBA goes through the direct `p416_to_hsv_row`
    // kernel (no source-width RGB scratch) — the 4:4:4 full-width-UV twin
    // of the P016 path. When RGB or RGBA is also attached the RGB kernel
    // runs anyway, so HSV derives off that buffer for free and
    // `need_rgb_kernel` keeps it alive.
    let want_hsv_direct = want_hsv && !want_rgb && !want_rgba;
    let need_rgb_kernel = want_rgb || (want_hsv && want_rgba);

    if want_hsv_direct {
      let hsv = hsv.as_mut().expect("want_hsv_direct implies hsv attached");
      let (h, s, v) = hsv.hsv();
      p416_to_hsv_row_endian(
        row.y(),
        row.uv_full(),
        &mut h[one_plane_start..one_plane_end],
        &mut s[one_plane_start..one_plane_end],
        &mut v[one_plane_start..one_plane_end],
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
        BE,
      );
      return Ok(());
    }

    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      p416_to_rgba_row_endian(
        row.y(),
        row.uv_full(),
        rgba_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
        BE,
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

    p416_to_rgb_row_endian(
      row.y(),
      row.uv_full(),
      rgb_row,
      w,
      row.matrix(),
      row.full_range(),
      use_simd,
      BE,
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
