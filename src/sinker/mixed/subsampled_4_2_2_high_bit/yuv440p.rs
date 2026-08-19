use super::super::{
  ChromaSitingChanged, FrozenOutputs, GeometryOverflow, HsvFrameMut, InsufficientBuffer,
  MixedSinker, MixedSinkerError, NativeRouteChanged, RowIndexOutOfRange, RowShapeMismatch,
  RowSlice, check_dimensions_match, chroma_440_bottom_sited_v, chroma_440_top_sited_v,
  deinterleave_y_high_bit_masked, packed_yuv444_triple_filter_resample,
  packed_yuv444_triple_resample, resample_preflight_check_only, reset_high_bit_yuv_streams,
  rgb_row_buf_or_scratch, rgba_plane_row_slice, rgba_u16_plane_row_slice,
  subsampled_4_2_0_high_bit::{
    emit_rgb_u8_wire, emit_rgb_u16_wire, reserve_420_chroma_full_u16, reserve_420_chroma_top_y_u16,
    yuv444p_top_identity_color_row,
  },
  yuv_planar16_process_native,
};
use crate::{
  PixelSink,
  resample::{
    AveragingDomain, InsertionContext, InsertionPoint, PlanGeometry, ResampleError, ResamplePlan,
    select_insertion_point,
  },
  row::*,
  source::*,
};

/// The high-bit 4:4:0 planar formats (`Yuv440p10` / `Yuv440p12`) ship the
/// non-4:2:0 native planar fast tier ([`yuv_planar16_process_native`]), so
/// each is statically eligible to splice an [`AveragingDomain::Encoded`] area
/// downscale at the native codes.
const YUV440P_HIGH_BIT_NATIVE_ELIGIBLE: bool = true;

/// **Fallible preflight** for the bottom-sited (`AVCHROMA_LOC_BOTTOM`)
/// vertical-phase HIGH-BIT 4:4:0 chroma lookback (RFC #238 S8c) — the `u16` twin
/// of the 8-bit [`reserve_440_chroma_prev`](super::super::reserve_440_chroma_prev)
/// and the FULL-width sibling of the half-width 4:2:0
/// [`reserve_420_chroma_prev_u16`]. Grows `chroma_prev` to `2 * width` `u16`
/// (full-width U then full-width V, wire byte order) so the later infallible
/// [`upsample_440_chroma_sited_u16`] can read the previous chroma row for the even
/// output row's vertical box blend. 4:4:0 keeps FULL-width chroma (no horizontal
/// subsampling), so the lookback is twice the half-width 4:2:0 one. Split out from
/// the upsample so it runs **before any output row is written** — the crate's
/// preflight-ordering atomicity contract (an allocator refusal must leave the
/// output frame untouched). `2 * width` is `checked_mul`'d (→ [`GeometryOverflow`]);
/// `try_reserve_exact` precedes the resize (→ [`ResampleError::AllocationFailed`]).
/// `height` feeds the error payload.
pub(crate) fn reserve_440_chroma_prev_u16(
  chroma_prev: &mut std::vec::Vec<u16>,
  width: usize,
  height: usize,
) -> Result<(), MixedSinkerError> {
  // Test-only failpoint (shared with the 4:2:0 / 8-bit lookbacks): simulate a
  // recoverable allocator refusal WITHOUT exhausting memory, so the atomicity
  // regression can prove no output row is written before this preflight. Compiled
  // away in non-test builds.
  #[cfg(all(test, feature = "std", feature = "yuv-planar"))]
  if super::super::FORCE_CHROMA_PREV_ALLOC_FAILURE.with(|f| f.take()) {
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

/// Stages the current full-width `u16` chroma row into the bottom-sited
/// vertical-phase HIGH-BIT 4:4:0 lookback (RFC #238 S8c) — the `u16` twin of the
/// 8-bit [`stage_440_chroma_prev`](super::super::stage_440_chroma_prev): copies
/// `u` then `v` into `chroma_prev` (`[0..width]` = U, `[width..2*width]` = V, wire
/// byte order preserved) and tags it with the chroma row it now holds (`idx / 2`),
/// so a *later* even output row can validate (`chroma_prev_row ==
/// Some(its_chroma_row - 1)`) and box-blend it. 4:4:0 chroma is full-width, so both
/// halves are `width` wide (twice the half-width 4:2:0 [`stage_420_chroma_prev_u16`]).
///
/// Called on EVERY accepted bottom-sited row — the colour-decode path
/// ([`upsample_440_chroma_sited_u16`], after it has read the *previous* lookback)
/// and the luma-only path alike — so the lookback is always current regardless of
/// which outputs are attached. **Infallible**: the caller must have run
/// [`reserve_440_chroma_prev_u16`] up front, so `chroma_prev` is guaranteed
/// `>= 2 * width` here.
#[inline]
pub(crate) fn stage_440_chroma_prev_u16(
  chroma_prev: &mut [u16],
  chroma_prev_row: &mut Option<usize>,
  u: &[u16],
  v: &[u16],
  idx: usize,
  width: usize,
) {
  debug_assert!(
    chroma_prev.len() >= 2 * width,
    "chroma_prev must be reserved via reserve_440_chroma_prev_u16 first"
  );
  chroma_prev[..width].copy_from_slice(&u[..width]);
  chroma_prev[width..2 * width].copy_from_slice(&v[..width]);
  *chroma_prev_row = Some(idx / 2);
}

/// Siting-aware full-width HIGH-BIT 4:4:0 chroma reconstruction for the
/// `Yuv440p{10,12}` `Bottom` (`v = 1`) vertical phase (RFC #238 S8c) — the `u16`
/// twin of the 8-bit
/// [`upsample_440_chroma_sited`](super::super::upsample_440_chroma_sited),
/// maintaining the one-row chroma lookback. Returns the full-width
/// `(u_full, v_full)` `u16` slices the high-bit 4:4:4 decode reads, staged in the
/// already-reserved `chroma_full` in the source's wire byte order. 4:4:0 keeps
/// FULL-width chroma, so — unlike the 4:2:0 [`upsample_420_chroma_sited_u16`] —
/// there is NO horizontal reconstruction: the even output row is a pure vertical
/// box blend and every other case is a straight copy of the current chroma row.
///
/// - On an **even** luma row (`idx & 1 == 0`) with `bottom_v`, each output sample
///   is the vertical box average of the previous chroma row (`chroma_prev`) and the
///   current row via the SIMD-dispatched
///   [`chroma_upsample_440_bottom_v_u16_row`](crate::row::chroma_upsample_440_bottom_v_u16_row)
///   (the depth-/endian-aware realization of
///   [`chroma_upsample_440_bottom_v_u16_wire`](crate::row::scalar::chroma_upsample_440_bottom_v_u16_wire))
///   — **but only when `chroma_prev` provably holds the wanted predecessor** chroma
///   row `idx/2 - 1` (`chroma_prev_row == Some(idx/2 - 1)`). When it does not — the
///   top edge (`idx == 0`), a caller that replayed / skipped / reordered rows or
///   attached colour late, or a fresh frame whose lookback `begin_frame` reset — it
///   clamps to a straight copy of the *current* chroma row. So the blend NEVER mixes
///   stale chroma from an older pair or a previous frame.
/// - Otherwise (the bottom-sited odd row `2i+1`, co-sited with chroma row `i`, or an
///   unvalidated even row) the current full-width chroma row is copied straight
///   through — 4:4:0 needs no horizontal reconstruction.
///
/// When `stage` is set the current chroma row is copied into `chroma_prev` (tagged
/// `idx / 2`) so the next pair's even row can validate and read it. The direct
/// decode passes `stage = true` (its post-reconstruction work is infallible); the
/// resample reconstruction arms pass `stage = false` and defer
/// [`stage_440_chroma_prev_u16`] until AFTER their fallible resample commit accepts
/// the row, so a rejected row leaves the lookback pointing at the predecessor and a
/// retry still box-blends it (state-atomic, #180). The caller must have run
/// [`reserve_420_chroma_full_u16`] (always) and, when `bottom_v`,
/// [`reserve_440_chroma_prev_u16`] up front, so both buffers are sized here and this
/// is infallible.
#[allow(clippy::too_many_arguments)]
pub(crate) fn upsample_440_chroma_sited_u16<'s, const BITS: u32>(
  chroma_full: &'s mut [u16],
  chroma_prev: &mut [u16],
  chroma_prev_row: &mut Option<usize>,
  u: &[u16],
  v: &[u16],
  idx: usize,
  bottom_v: bool,
  stage: bool,
  width: usize,
  big_endian: bool,
  use_simd: bool,
) -> (&'s [u16], &'s [u16]) {
  debug_assert!(
    chroma_full.len() >= 2 * width,
    "chroma_full must be reserved via reserve_420_chroma_full_u16 first"
  );
  let chroma_row = idx / 2;
  // The bottom-sited EVEN row box-blends only when the lookback PROVABLY holds the
  // wanted predecessor `chroma_row - 1`; otherwise it clamps to a straight copy of
  // the current row (never stale). Every other case (the bottom-sited odd row, or
  // an unvalidated even row) copies the current full-width chroma row straight
  // through — 4:4:0 needs no horizontal reconstruction.
  let do_vblend =
    bottom_v && idx & 1 == 0 && chroma_row > 0 && *chroma_prev_row == Some(chroma_row - 1);
  let (u_full, v_full) = chroma_full[..2 * width].split_at_mut(width);
  if do_vblend {
    debug_assert!(
      chroma_prev.len() >= 2 * width,
      "chroma_prev must be reserved via reserve_440_chroma_prev_u16 first"
    );
    let (u_prev, v_prev) = chroma_prev[..2 * width].split_at(width);
    crate::row::chroma_upsample_440_bottom_v_u16_row::<BITS>(
      u_prev, u, u_full, width, big_endian, use_simd,
    );
    crate::row::chroma_upsample_440_bottom_v_u16_row::<BITS>(
      v_prev, v, v_full, width, big_endian, use_simd,
    );
  } else {
    u_full.copy_from_slice(&u[..width]);
    v_full.copy_from_slice(&v[..width]);
  }

  // Refresh the lookback with the current chroma row + its validity tag (after the
  // read above). Gated on `stage`: the direct decode refreshes here (its later work
  // is infallible); the resample arms pass `stage = false` and defer the refresh to
  // AFTER their fallible commit accepts the row.
  if bottom_v && stage {
    stage_440_chroma_prev_u16(chroma_prev, chroma_prev_row, u, v, idx, width);
  }
  (&*u_full, &*v_full)
}

/// The RFC #238 **Top** (`v = 0`) FORWARD one-row delay for the high-bit
/// `Yuv440pN` row-stage (area) reconstruction tier — the full-width `u16` 4:4:0
/// twin of the 4:2:0 [`yuv420p_top_reconstruct_area`](super::super::subsampled_4_2_0_high_bit)
/// (no horizontal fold, so no `center_sited`). A `Top` chroma sample is co-sited
/// with the TOP luma row of its vertical pair, so an EVEN source row decodes
/// co-sited chroma while an ODD source row needs the vertical box-average of its
/// chroma row and the NEXT one, which a row-at-a-time stream has not been fed
/// when the odd row arrives. The whole odd row is HELD (`chroma_top_pending` +
/// the buffered `chroma_top_y_u16`) and the following even row feeds TWO source
/// rows: the held odd row (forward-blended through the SAME `Bottom`-EVEN
/// [`upsample_440_chroma_sited_u16`]`(bottom_v = true)` kernel at THIS even index)
/// then the current even row (co-sited). The trailing odd row of an even-height
/// frame clamps to a co-sited decode; an ODD height's final even row is the
/// two-row flush. `packed_yuv444_triple_resample`'s stream `feed_row` allocates
/// nothing once row 0 has sized the streams + scratches, so reserving every
/// buffer up front (#180) makes the two-row flush retry-atomic — a refusal
/// returns before any feed. Returns [`ControlFlow::Break`](core::ops::ControlFlow)
/// when the preflight declined the row (no output; caller returns without
/// freezing).
#[cfg(feature = "yuv-planar")]
#[allow(clippy::too_many_arguments)]
fn yuv440p_top_reconstruct_area<const BITS: u32, const BE: bool>(
  rgb_stream: &mut Option<std::boxed::Box<crate::resample::AreaStream<u8>>>,
  rgb_stream_u16: &mut Option<std::boxed::Box<crate::resample::AreaStream<u16>>>,
  luma_stream_u16: &mut Option<std::boxed::Box<crate::resample::AreaStream<u16>>>,
  resample_outputs: &mut Option<FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  rgb_u16: &mut Option<&mut [u16]>,
  rgba_u16: &mut Option<&mut [u16]>,
  luma: &mut Option<&mut [u8]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  rgb_scratch: &mut std::vec::Vec<u8>,
  rgb_scratch_u16: &mut std::vec::Vec<u16>,
  luma_scratch_u16: &mut std::vec::Vec<u16>,
  chroma_full_u16: &mut std::vec::Vec<u16>,
  chroma_prev_u16: &mut std::vec::Vec<u16>,
  chroma_prev_row: &mut Option<usize>,
  chroma_top_pending: &mut Option<(usize, crate::KernelMatrix, bool)>,
  chroma_top_y_u16: &mut std::vec::Vec<u16>,
  y_row: &[u16],
  u: &[u16],
  v: &[u16],
  w: usize,
  h: usize,
  plan: &ResamplePlan,
  idx: usize,
  use_simd: bool,
  matrix: crate::KernelMatrix,
  full_range: bool,
) -> Result<core::ops::ControlFlow<()>, MixedSinkerError> {
  // The stream cursor lags the source by the buffered (unfed) held odd row, so
  // the next expected SOURCE row is `cursor + pending`. Reject an out-of-sequence
  // walker row (and a mid-frame output change) BEFORE reserving any chroma (#180).
  let cursor = if luma.is_some() {
    luma_stream_u16.as_ref().map_or(0, |s| s.next_y())
  } else if rgb.is_some() || rgba.is_some() || hsv.is_some() {
    rgb_stream.as_ref().map_or(0, |s| s.next_y())
  } else {
    rgb_stream_u16.as_ref().map_or(0, |s| s.next_y())
  };
  let expected_source = cursor + usize::from(chroma_top_pending.is_some());
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
    Some(expected_source),
    idx,
  )? {
    return Ok(core::ops::ControlFlow::Break(()));
  }
  // Reserve every buffer BOTH feeds may touch, up front (#180). After row 0 these
  // are no-ops, so the even-row double feed does no allocation.
  reserve_420_chroma_full_u16(chroma_full_u16, w, h)?;
  reserve_440_chroma_prev_u16(chroma_prev_u16, w, h)?;
  reserve_420_chroma_top_y_u16(chroma_top_y_u16, w, h)?;
  let is_last = idx + 1 == h;
  if idx & 1 == 0 {
    // Flush the held odd predecessor FIRST, forward-blending the previous chroma
    // row (`chroma_prev_u16`) with the current one at this even index.
    if let Some((p_idx, p_matrix, p_full_range)) = chroma_top_pending.take() {
      let (u_full, v_full) = upsample_440_chroma_sited_u16::<BITS>(
        chroma_full_u16,
        chroma_prev_u16,
        chroma_prev_row,
        u,
        v,
        idx,
        true,
        false,
        w,
        BE,
        use_simd,
      );
      let held_y: &[u16] = &chroma_top_y_u16[..w];
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
        p_idx,
        use_simd,
        p_matrix,
        p_full_range,
        |scratch| {
          emit_rgb_u8_wire::<BITS>(
            held_y,
            u_full,
            v_full,
            scratch,
            w,
            p_matrix,
            p_full_range,
            use_simd,
            BE,
          )
        },
        |scratch| {
          emit_rgb_u16_wire::<BITS>(
            held_y,
            u_full,
            v_full,
            scratch,
            w,
            p_matrix,
            p_full_range,
            use_simd,
            BE,
          )
        },
        |scratch| deinterleave_y_high_bit_masked::<BITS, BE>(held_y, scratch, w),
      )?;
    }
    // Then feed the current EVEN row (co-sited chroma).
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
      |scratch| emit_rgb_u8_wire::<BITS>(y_row, u, v, scratch, w, matrix, full_range, use_simd, BE),
      |scratch| {
        emit_rgb_u16_wire::<BITS>(y_row, u, v, scratch, w, matrix, full_range, use_simd, BE)
      },
      |scratch| deinterleave_y_high_bit_masked::<BITS, BE>(y_row, scratch, w),
    )?;
  } else if !is_last {
    // Defer this odd row: copy its wire Y (the borrow expires after `process`) +
    // record its decode params; it emits at the next even row.
    chroma_top_y_u16[..w].copy_from_slice(&y_row[..w]);
    *chroma_top_pending = Some((idx, matrix, full_range));
  } else {
    // Trailing odd row (bottom-edge clamp → co-sited): feed it now.
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
      |scratch| emit_rgb_u8_wire::<BITS>(y_row, u, v, scratch, w, matrix, full_range, use_simd, BE),
      |scratch| {
        emit_rgb_u16_wire::<BITS>(y_row, u, v, scratch, w, matrix, full_range, use_simd, BE)
      },
      |scratch| deinterleave_y_high_bit_masked::<BITS, BE>(y_row, scratch, w),
    )?;
  }
  // Refresh the lookback with the current chroma row (after the reads above) so
  // the NEXT even row's flush forward-blends this row as its predecessor.
  stage_440_chroma_prev_u16(chroma_prev_u16, chroma_prev_row, u, v, idx, w);
  Ok(core::ops::ControlFlow::Continue(()))
}

/// The RFC #238 **Top** (`v = 0`) FORWARD one-row delay for the high-bit
/// `Yuv440pN` single-kernel FILTER reconstruction tier — the exact structure of
/// [`yuv440p_top_reconstruct_area`], only the resampler kind differs
/// (`packed_yuv444_triple_filter_resample`, driven by `FilterStream`). Returns
/// [`ControlFlow::Break`](core::ops::ControlFlow) when the preflight declined the
/// row.
#[cfg(feature = "yuv-planar")]
#[allow(clippy::too_many_arguments)]
fn yuv440p_top_reconstruct_filter<const BITS: u32, const BE: bool>(
  rgb_filter_stream: &mut Option<std::boxed::Box<crate::resample::FilterStream<u8>>>,
  rgb_filter_stream_u16: &mut Option<std::boxed::Box<crate::resample::FilterStream<u16>>>,
  luma_filter_stream_u16: &mut Option<std::boxed::Box<crate::resample::FilterStream<u16>>>,
  resample_outputs: &mut Option<FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  rgb_u16: &mut Option<&mut [u16]>,
  rgba_u16: &mut Option<&mut [u16]>,
  luma: &mut Option<&mut [u8]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  rgb_scratch: &mut std::vec::Vec<u8>,
  rgb_scratch_u16: &mut std::vec::Vec<u16>,
  luma_scratch_u16: &mut std::vec::Vec<u16>,
  chroma_full_u16: &mut std::vec::Vec<u16>,
  chroma_prev_u16: &mut std::vec::Vec<u16>,
  chroma_prev_row: &mut Option<usize>,
  chroma_top_pending: &mut Option<(usize, crate::KernelMatrix, bool)>,
  chroma_top_y_u16: &mut std::vec::Vec<u16>,
  y_row: &[u16],
  u: &[u16],
  v: &[u16],
  w: usize,
  h: usize,
  plan: &ResamplePlan,
  idx: usize,
  use_simd: bool,
  matrix: crate::KernelMatrix,
  full_range: bool,
) -> Result<core::ops::ControlFlow<()>, MixedSinkerError> {
  let cursor = if luma.is_some() {
    luma_filter_stream_u16.as_ref().map_or(0, |s| s.next_y())
  } else if rgb.is_some() || rgba.is_some() || hsv.is_some() {
    rgb_filter_stream.as_ref().map_or(0, |s| s.next_y())
  } else {
    rgb_filter_stream_u16.as_ref().map_or(0, |s| s.next_y())
  };
  let expected_source = cursor + usize::from(chroma_top_pending.is_some());
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
    Some(expected_source),
    idx,
  )? {
    return Ok(core::ops::ControlFlow::Break(()));
  }
  reserve_420_chroma_full_u16(chroma_full_u16, w, h)?;
  reserve_440_chroma_prev_u16(chroma_prev_u16, w, h)?;
  reserve_420_chroma_top_y_u16(chroma_top_y_u16, w, h)?;
  let is_last = idx + 1 == h;
  if idx & 1 == 0 {
    if let Some((p_idx, p_matrix, p_full_range)) = chroma_top_pending.take() {
      let (u_full, v_full) = upsample_440_chroma_sited_u16::<BITS>(
        chroma_full_u16,
        chroma_prev_u16,
        chroma_prev_row,
        u,
        v,
        idx,
        true,
        false,
        w,
        BE,
        use_simd,
      );
      let held_y: &[u16] = &chroma_top_y_u16[..w];
      packed_yuv444_triple_filter_resample::<BITS>(
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
        p_idx,
        use_simd,
        p_matrix,
        p_full_range,
        |scratch| {
          emit_rgb_u8_wire::<BITS>(
            held_y,
            u_full,
            v_full,
            scratch,
            w,
            p_matrix,
            p_full_range,
            use_simd,
            BE,
          )
        },
        |scratch| {
          emit_rgb_u16_wire::<BITS>(
            held_y,
            u_full,
            v_full,
            scratch,
            w,
            p_matrix,
            p_full_range,
            use_simd,
            BE,
          )
        },
        |scratch| deinterleave_y_high_bit_masked::<BITS, BE>(held_y, scratch, w),
      )?;
    }
    packed_yuv444_triple_filter_resample::<BITS>(
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
      |scratch| emit_rgb_u8_wire::<BITS>(y_row, u, v, scratch, w, matrix, full_range, use_simd, BE),
      |scratch| {
        emit_rgb_u16_wire::<BITS>(y_row, u, v, scratch, w, matrix, full_range, use_simd, BE)
      },
      |scratch| deinterleave_y_high_bit_masked::<BITS, BE>(y_row, scratch, w),
    )?;
  } else if !is_last {
    chroma_top_y_u16[..w].copy_from_slice(&y_row[..w]);
    *chroma_top_pending = Some((idx, matrix, full_range));
  } else {
    packed_yuv444_triple_filter_resample::<BITS>(
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
      |scratch| emit_rgb_u8_wire::<BITS>(y_row, u, v, scratch, w, matrix, full_range, use_simd, BE),
      |scratch| {
        emit_rgb_u16_wire::<BITS>(y_row, u, v, scratch, w, matrix, full_range, use_simd, BE)
      },
      |scratch| deinterleave_y_high_bit_masked::<BITS, BE>(y_row, scratch, w),
    )?;
  }
  stage_440_chroma_prev_u16(chroma_prev_u16, chroma_prev_row, u, v, idx, w);
  Ok(core::ops::ControlFlow::Continue(()))
}

// ---- Yuv440p10 impl -----------------------------------------------------
//
// 4:4:0 planar 10‑bit. Same row math as 4:4:4 10-bit; reuses
// `yuv444p10_to_rgb_*`. Walker handles the half-height chroma.

impl<'a, R, const BE: bool> MixedSinker<'a, Yuv440p10<BE>, R> {
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

  /// Attaches a packed **8-bit** RGBA output buffer. Yuv440p10 reuses
  /// the `BITS = 10` 4:4:4 RGBA kernel; alpha = `0xFF`.
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

impl<R, const BE: bool> Yuv440p10Sink<BE> for MixedSinker<'_, Yuv440p10<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, Yuv440p10<BE>, R> {
  type Input<'r> = Yuv440p10Row<'r>;
  type Error = MixedSinkerError;

  #[cfg_attr(not(tarpaulin), inline(always))]
  fn kernel_matrix(&self) -> crate::KernelMatrix {
    self.kernel_matrix
  }
  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    reset_high_bit_yuv_streams(self);
    Ok(())
  }

  fn process(&mut self, row: Yuv440p10Row<'_>) -> Result<(), Self::Error> {
    const BITS: u32 = 10;
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;
    // Chroma siting (RFC #238 S8c): 4:4:0 carries only a VERTICAL phase
    // (full-width chroma → no horizontal siting). `Copy`, so read it out before
    // the field split-borrow below.
    let chroma_location = self.chroma_location.clone();

    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Y10,
        idx,
        w,
        row.y().len(),
      )));
    }
    if row.u().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::UFull10,
        idx,
        w,
        row.u().len(),
      )));
    }
    if row.v().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::VFull10,
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
      native,
      native_planar_u16,
      frozen_native_route,
      chroma_full_u16,
      chroma_prev_u16,
      chroma_prev_row,
      // RFC #238 S8c: the 4:4:0 vertical (`Bottom`) chroma phase frozen on the
      // first output-bearing row. `frozen_chroma_centered` is cleared alongside it
      // in `begin_frame` (via `reset_high_bit_yuv_streams`) but never set here —
      // 4:4:0 has no horizontal phase.
      frozen_chroma_bottom_v,
      // RFC #238 Top: the `Top` (`v = 0`) vertical phase frozen alongside it, and
      // the forward one-row delay line (row-stage / filter / identity tiers).
      frozen_chroma_top_v,
      chroma_top_pending,
      chroma_top_y_u16,
      ..
    } = self;

    // Non-identity plan: feed the shared high-bit 4:4:4 triple-resample
    // tail (u8 color, independent native-u16 color, native Y). 4:4:0 is
    // full-width chroma (no horizontal upsampling, same per-row contract
    // as 4:4:4) — its vertical chroma sharing is already resolved by the
    // walker, which hands this luma row the (vertically-shared) full-width
    // `u` / `v`, so the converted RGB is full-res and the 4:4:4 tail binds
    // via the shared `yuv444pN_to_rgb_*` kernels. Yuv440p exposes no
    // `luma_u16` output, so it is `&mut None` and only `luma` (binned
    // native Y `>> (BITS - 8)`) is emitted. The span kind picks the engine:
    // area binning, or the signed-coefficient filter twin (both convert the
    // YUV to RGB with the same closures and resample in RGB space, so filter
    // colour equals the RGB filter of the converted pixels and matches area
    // up to the kernel). The filter tail clamps every sub-16-bit colour
    // sample AND the native Y to `(1 << BITS) - 1` before publishing.
    if let Some(plan) = plan.as_ref() {
      let matrix = row.matrix();
      let full_range = row.full_range();
      let (y, u, v) = (row.y(), row.u(), row.v());
      // RFC #238 S8c — 4:4:0 VERTICAL chroma siting. `Bottom` / `BottomLeft`
      // ([`chroma_440_bottom_sited_v`], `v = 1`) box-blends the even output row's
      // full-width chroma with the previous chroma row; every other siting keeps
      // the co-sited vertical decode (byte-identical). 4:4:0 has FULL-width chroma,
      // so there is NO horizontal phase — only the vertical axis is ever folded
      // (`h_phase` stays 0). The native fast tier folds `v = 1` into
      // `area_chroma_440`'s vertical weights; the RGB-domain reconstruction tiers
      // (row-stage RGB/RGBA/HSV, filter) box-blend full-width `u16` chroma via the
      // `chroma_prev_u16` lookback and decode 4:4:4. Those reconstruction arms
      // reserve chroma AFTER the resample preflight, so a rejected / out-of-sequence
      // row is caught first (the #180 reserve-after-preflight invariant).
      let bottom_v = chroma_440_bottom_sited_v(&chroma_location);
      let chroma_v_phase = if bottom_v { 1.0 } else { 0.0 };
      // RFC #238 Top (`v = 0`, FORWARD fold) — `Top` / `TopLeft`. 4:4:0 has no
      // horizontal phase, so both share the identical full-width vertical box
      // average. Disjoint from `bottom_v`. The native fast tier folds it into
      // `area_chroma_440`'s vertical weights via `top_v`; the reconstruction tiers
      // reconstruct each row's chroma through a FORWARD one-row delay (mirror of
      // `Bottom`'s backward `chroma_prev_u16` lookback).
      let top_v = chroma_440_top_sited_v(&chroma_location);
      // Whether this row produces any colour output (and so runs the bottom-sited
      // chroma reconstruction). A luma-only row bins native Y unchanged (siting is a
      // chroma-only property).
      let want_color =
        rgb.is_some() || rgba.is_some() || hsv.is_some() || rgb_u16.is_some() || rgba_u16.is_some();
      // Whether this call carries any output — the EXACT set both tiers' preflights
      // test. The route / siting freezes only on an output-bearing row a tier
      // ACCEPTS; a no-output call consumes no stream state, so it must not freeze.
      let need_output = luma.is_some() || want_color;
      // Freeze the effective 4:4:0 vertical chroma siting on the first
      // output-bearing row (mirrors the `frozen_native_route` freeze below). This
      // CHECK is at the always-compiled choke point every tier passes through; the
      // matching SET rides each tier's accept path (never before dispatch, so a
      // rejected row leaves it unset for a corrected retry). A later row observing a
      // different vertical phase would bin a mixture of co-sited and
      // bottom-/top-folded chroma, so it is rejected here before any reconstruction.
      // 4:4:0 folds only the vertical axis, so `frozen_chroma_bottom_v` is the
      // set/unset sentinel; the Top flag is compared alongside because co-sited and
      // `Top` share `frozen_chroma_bottom_v = Some(false)`.
      if need_output
        && let Some(frozen) = *frozen_chroma_bottom_v
        && (frozen != bottom_v || *frozen_chroma_top_v != Some(top_v))
      {
        return Err(MixedSinkerError::ChromaSitingChanged(
          ChromaSitingChanged::new(idx),
        ));
      }
      if plan.kind().is_filter() {
        if bottom_v && want_color {
          // Bottom single-kernel filter: reconstruct full-width `u16` chroma, but
          // ONLY after the resample preflight (frozen-output + sequence), so an
          // out-of-sequence / rejected row is caught before the chroma reservation
          // (#180). Reject a multi-kernel (BICUBLIN) plan BEFORE the reserve too —
          // hoisted from the delegate's first act (idempotent, it re-runs it).
          // `packed_yuv444_triple_filter_resample` owns the transactional commit.
          plan.ensure_single_kernel_filter()?;
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
          reserve_440_chroma_prev_u16(chroma_prev_u16, w, h)?;
          // `stage = false`: DEFER the lookback advance until AFTER the fallible
          // filter commit accepts the row (below), so a rejected row leaves the
          // predecessor in place for a clean retry (#180 state-atomicity).
          let (u_full, v_full) = upsample_440_chroma_sited_u16::<BITS>(
            chroma_full_u16,
            chroma_prev_u16,
            chroma_prev_row,
            u,
            v,
            idx,
            bottom_v,
            false,
            w,
            BE,
            use_simd,
          );
          let r = packed_yuv444_triple_filter_resample::<BITS>(
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
              yuv444p10_to_rgb_row_endian(
                y, u_full, v_full, scratch, w, matrix, full_range, use_simd, BE,
              )
            },
            |scratch| {
              yuv444p10_to_rgb_u16_row_endian(
                y, u_full, v_full, scratch, w, matrix, full_range, use_simd, BE,
              )
            },
            |scratch| deinterleave_y_high_bit_masked::<BITS, BE>(y, scratch, w),
          );
          // Bottom lookback: advance only AFTER the filter resample accepts the row,
          // so a rejected row leaves the predecessor for a clean retry — the sited
          // reconstruction read it above but did not stage.
          if r.is_ok() {
            stage_440_chroma_prev_u16(chroma_prev_u16, chroma_prev_row, u, v, idx, w);
          }
          if r.is_ok() && need_output && frozen_chroma_bottom_v.is_none() {
            *frozen_chroma_bottom_v = Some(bottom_v);
            *frozen_chroma_top_v = Some(top_v);
          }
          return r;
        }
        if top_v && want_color {
          // RFC #238 Top (`v = 0`) FORWARD one-row delay for the single-kernel
          // filter tier — the whole odd row is HELD and the following even row
          // filter-feeds TWO source rows (held forward-blend + current co-sited);
          // the trailing odd row clamps to a co-sited decode. Reject a multi-kernel
          // (BICUBLIN) plan BEFORE the reserves (idempotent, the delegate re-runs
          // it). Retry-atomic (#180) — the reserves precede either feed.
          plan.ensure_single_kernel_filter()?;
          match yuv440p_top_reconstruct_filter::<BITS, BE>(
            rgb_filter_stream,
            rgb_filter_stream_u16,
            luma_filter_stream_u16,
            resample_outputs,
            rgb,
            rgba,
            rgb_u16,
            rgba_u16,
            luma,
            hsv,
            rgb_scratch,
            rgb_scratch_u16,
            luma_scratch_u16,
            chroma_full_u16,
            chroma_prev_u16,
            chroma_prev_row,
            chroma_top_pending,
            chroma_top_y_u16,
            y,
            u,
            v,
            w,
            h,
            plan,
            idx,
            use_simd,
            matrix,
            full_range,
          )? {
            core::ops::ControlFlow::Break(()) => return Ok(()),
            core::ops::ControlFlow::Continue(()) => {}
          }
          if frozen_chroma_bottom_v.is_none() && need_output {
            *frozen_chroma_bottom_v = Some(bottom_v);
            *frozen_chroma_top_v = Some(top_v);
          }
          return Ok(());
        }
        let r = packed_yuv444_triple_filter_resample::<BITS>(
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
            yuv444p10_to_rgb_row_endian(y, u, v, scratch, w, matrix, full_range, use_simd, BE)
          },
          |scratch| {
            yuv444p10_to_rgb_u16_row_endian(y, u, v, scratch, w, matrix, full_range, use_simd, BE)
          },
          |scratch| deinterleave_y_high_bit_masked::<BITS, BE>(y, scratch, w),
        );
        if r.is_ok() && need_output && frozen_chroma_bottom_v.is_none() {
          *frozen_chroma_bottom_v = Some(bottom_v);
          *frozen_chroma_top_v = Some(top_v);
        }
        return r;
      }
      // Native / row-stage route split — see the high-bit 4:2:0 Yuv420p impl
      // for the CHECK-before / SET-after `frozen_native_route` contract. Reuses
      // the `need_output` computed for the siting freeze above.
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
          native_eligible: YUV440P_HIGH_BIT_NATIVE_ELIGIBLE,
          with_native: *native,
          area_plan: true,
        },
      );
      match insertion {
        InsertionPoint::NativeCodes => {
          // 4:4:0: chroma `w x h/2` — full width, half height; a chroma row per
          // TWO Y rows (`chroma_vsub = 2`, like 4:2:0 vertically; `chroma_w = w`),
          // chroma plan full-width horizontal + luma-domain `area_halved` vertical
          // (or the RFC #238 S8c `Bottom` `v = 1` phased-V fold).
          //
          // Point-of-use siting invalidation (see the high-bit 4:2:0 native arm):
          // `chroma_location` can change at ANY point before this row (including
          // AFTER `begin_frame`, before row 0), so re-check the cached join HERE and
          // drop it when its folded chroma plan was built for a different vertical
          // phase; `yuv_planar16_process_native` rebuilds it with the current siting.
          // Retry-atomic: drop ONLY on the IN-SEQUENCE fresh-frame first row
          // (`idx == 0`, `next_y() == 0`); an out-of-sequence first row after a
          // siting change is left for the delegate to reject against the INTACT join.
          // Transactional: move the stale join OUT and let the delegate build the
          // replacement into `native_planar_u16` (its build runs BEFORE it inserts,
          // so a build failure leaves the field `None`); on such a failure, restore
          // the intact prior-phase join so the REJECTED row mutates nothing. A
          // luma-only join carries no chroma plan (siting-independent).
          let stale_native = idx == 0
            && native_planar_u16.as_ref().is_some_and(|join| {
              (join.chroma_bottom() == Some(!bottom_v) || join.chroma_top() == Some(!top_v))
                && join.next_y() == 0
            });
          let prev_native = if stale_native {
            native_planar_u16.take()
          } else {
            None
          };
          let native_result = yuv_planar16_process_native::<BITS, BE>(
            plan,
            native_planar_u16,
            resample_outputs,
            rgb,
            rgba,
            rgb_u16,
            rgba_u16,
            luma,
            // The high-bit planar 4:4:0 family exposes no `luma_u16` output.
            &mut None,
            hsv,
            rgb_scratch,
            rgb_scratch_u16,
            y,
            u,
            v,
            matrix,
            full_range,
            idx,
            w,
            h,
            2,
            w,
            || {
              ResamplePlan::area_chroma_440(
                w,
                h,
                plan.out_w(),
                plan.out_h(),
                0.0,
                chroma_v_phase,
                top_v,
              )
            },
            use_simd,
          );
          // Restore the taken stale-phase join if the delegate's rebuild was rejected
          // at any pre-feed step: it leaves the field `None` on such a failure, so
          // restoring the intact prior-phase join leaves the rejected row mutating no
          // join state. A non-stale row took nothing.
          if stale_native && native_result.is_err() {
            *native_planar_u16 = prev_native;
          }
          native_result?;
          if frozen_native_route.is_none() && need_output {
            *frozen_native_route = Some(true);
          }
          // RFC #238 S8c: freeze the siting on the same accepted output row.
          if frozen_chroma_bottom_v.is_none() && need_output {
            *frozen_chroma_bottom_v = Some(bottom_v);
            *frozen_chroma_top_v = Some(top_v);
          }
          return Ok(());
        }
        InsertionPoint::EncodedOutput => {
          if bottom_v && want_color {
            // Bottom row-stage: reconstruct full-width `u16` chroma AFTER the
            // resample preflight (frozen-output + sequence), so an out-of-sequence /
            // rejected row is caught before the chroma reservation (#180).
            // `packed_yuv444_triple_resample` re-runs the idempotent preflight and
            // owns the transactional commit. Gated by `want_color` — a luma-only
            // Bottom row never calls the RGB converter, so it stays on the co-sited
            // arm (which only bins native Y).
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
            reserve_440_chroma_prev_u16(chroma_prev_u16, w, h)?;
            // `stage = false`: DEFER the lookback advance until AFTER the fallible
            // row-stage commit accepts the row (below), so a rejected row leaves the
            // predecessor for a clean retry (#180 state-atomicity).
            let (u_full, v_full) = upsample_440_chroma_sited_u16::<BITS>(
              chroma_full_u16,
              chroma_prev_u16,
              chroma_prev_row,
              u,
              v,
              idx,
              bottom_v,
              false,
              w,
              BE,
              use_simd,
            );
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
              |scratch| {
                yuv444p10_to_rgb_row_endian(
                  y, u_full, v_full, scratch, w, matrix, full_range, use_simd, BE,
                )
              },
              |scratch| {
                yuv444p10_to_rgb_u16_row_endian(
                  y, u_full, v_full, scratch, w, matrix, full_range, use_simd, BE,
                )
              },
              |scratch| deinterleave_y_high_bit_masked::<BITS, BE>(y, scratch, w),
            )?;
            // Bottom lookback: advance only AFTER the row-stage resample accepts the
            // row (the `?` above already returned any reject), so a rejected row
            // leaves the predecessor for a clean retry — the sited reconstruction
            // read it above but did not stage.
            stage_440_chroma_prev_u16(chroma_prev_u16, chroma_prev_row, u, v, idx, w);
          } else if top_v && want_color {
            // RFC #238 Top (`v = 0`) FORWARD one-row delay for the row-stage tier —
            // the whole odd row is HELD and the following even row feeds TWO source
            // rows (held forward-blend + current co-sited); the trailing odd row
            // clamps to a co-sited decode. Retry-atomic (#180) — the reserves
            // precede either feed. A preflight decline returns without freezing.
            match yuv440p_top_reconstruct_area::<BITS, BE>(
              rgb_stream,
              rgb_stream_u16,
              luma_stream_u16,
              resample_outputs,
              rgb,
              rgba,
              rgb_u16,
              rgba_u16,
              luma,
              hsv,
              rgb_scratch,
              rgb_scratch_u16,
              luma_scratch_u16,
              chroma_full_u16,
              chroma_prev_u16,
              chroma_prev_row,
              chroma_top_pending,
              chroma_top_y_u16,
              y,
              u,
              v,
              w,
              h,
              plan,
              idx,
              use_simd,
              matrix,
              full_range,
            )? {
              core::ops::ControlFlow::Break(()) => return Ok(()),
              core::ops::ControlFlow::Continue(()) => {}
            }
          } else {
            // Co-sited RGB / RGBA / HSV / luma: the byte-identical row-stage tail.
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
              |scratch| {
                yuv444p10_to_rgb_row_endian(y, u, v, scratch, w, matrix, full_range, use_simd, BE)
              },
              |scratch| {
                yuv444p10_to_rgb_u16_row_endian(
                  y, u, v, scratch, w, matrix, full_range, use_simd, BE,
                )
              },
              |scratch| deinterleave_y_high_bit_masked::<BITS, BE>(y, scratch, w),
            )?;
          }
          if frozen_native_route.is_none() && need_output {
            *frozen_native_route = Some(false);
          }
          // RFC #238 S8c: freeze the siting on the same accepted output row.
          if frozen_chroma_bottom_v.is_none() && need_output {
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

    // Resolve the output set + the no-output guard up front, BEFORE any per-row
    // offset arithmetic, so a no-output call runs NOTHING — no `idx * w` (which
    // could overflow on a 32-bit target for a call that never ran an attach-time
    // `w x h` validation), no allocation, no lookback priming. `need_output` MUST
    // include the u16 twins (`rgb_u16` / `rgba_u16`), else a u16-only sink would
    // skip the siting freeze / lookback.
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();
    let want_rgb_u16 = rgb_u16.is_some();
    let want_rgba_u16 = rgba_u16.is_some();
    // A bottom-sited row maintains the vertical lookback even when it produces only
    // luma (the luma-only staging below), so `want_color` gates only the colour
    // scratches / reconstruction.
    let want_color = want_rgb || want_rgba || want_hsv || want_rgb_u16 || want_rgba_u16;
    let need_output = want_color || luma.is_some();
    if !need_output {
      return Ok(());
    }

    // Single-plane row ranges cannot overflow: the no-output guard above ensures
    // >= 1 output is attached, so its attach-time `w x h x 1` validation ran and
    // `w x h` fits `usize`; with `idx < h` that makes `(idx + 1) * w <= h * w` fit.
    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // Chroma siting (RFC #238 S8c): `Bottom` (v = 1) box-blends the even output
    // row's full-width chroma with the previous chroma row; the default / Center /
    // Top sitings keep the byte-identical co-sited decode (their odd output row
    // needs the *next* chroma row, deferred). 4:4:0 has full-width chroma, so there
    // is no horizontal phase.
    let bottom_v = chroma_440_bottom_sited_v(&chroma_location);
    // RFC #238 Top (`v = 0`, FORWARD one-row delay) — `Top` / `TopLeft`. An ODD
    // output row needs the NEXT chroma row (unavailable when it arrives), so its
    // colour output is HELD (`chroma_top_pending` / `chroma_top_y_u16`) and emitted
    // at the following even row; handled in its own branch below. Disjoint from
    // `bottom_v`. Luma is siting-independent and written in order.
    let top_v = chroma_440_top_sited_v(&chroma_location);

    // RFC #238 S8c: freeze the vertical chroma siting on the direct (identity) path
    // too. The resample branch above rejects a mid-frame vertical-phase flip via
    // `frozen_chroma_bottom_v`, and this path must honour the SAME per-frame
    // invariant: `set_chroma_location` is public, so without this a caller could
    // emit rows `Bottom`, switch to a co-sited location, then back, binning a
    // mixture of co-sited and bottom-folded chroma (and, after a co-sited gap,
    // box-blend a later even row against a stale lookback). CHECK here — before any
    // reservation or lookback mutation; the matching SET rides the row's accept
    // point below (after the fallible preflight). `need_output` is already
    // established (the no-output guard returned above).
    if let Some(frozen) = *frozen_chroma_bottom_v
      && (frozen != bottom_v || *frozen_chroma_top_v != Some(top_v))
    {
      return Err(MixedSinkerError::ChromaSitingChanged(
        ChromaSitingChanged::new(idx),
      ));
    }

    // Atomicity preflight (#308, cf. the crate's #180 resample fix and the high-bit
    // 4:2:0 sibling): reserve EVERY fallible row scratch this row needs BEFORE any
    // output row (luma included) is written, so an allocator refusal returns a typed
    // `AllocationFailed` leaving the output frame untouched. Growable here:
    //  1. the bottom-sited full-width blend scratch (`chroma_full_u16`, `2 * w`),
    //     only on a colour Bottom decode;
    //  2. the bottom-sited vertical-phase full-width lookback (`chroma_prev_u16`), on
    //     every Bottom OUTPUT row — colour OR luma-only — because the lookback is
    //     maintained so a later colour row can box-blend it (the luma-only staging
    //     runs below, before any luma write); and
    //  3. the u8 RGB row scratch, exactly `want_hsv && want_rgba && !want_rgb`.
    // The u16 RGB / RGBA outputs write straight into their caller buffers and never
    // grow a scratch; this format exposes no luma_u16. A no-output row returned early
    // above, so it never reaches here.
    if (bottom_v || top_v) && want_color {
      reserve_420_chroma_full_u16(chroma_full_u16, w, h)?;
    }
    if bottom_v || top_v {
      reserve_440_chroma_prev_u16(chroma_prev_u16, w, h)?;
    }
    // Top forward-delay held-Y buffer: reserved up front — BEFORE any luma write —
    // so the later deferred-row copy + the two-row flush are infallible (#180).
    if top_v && want_color {
      reserve_420_chroma_top_y_u16(chroma_top_y_u16, w, h)?;
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

    // Bottom / Top sited LUMA-ONLY row: the colour upsample helper — which normally
    // refreshes the vertical lookback — won't run, so stage the current full-width
    // chroma row into the lookback HERE, after its preflight reservation above and
    // BEFORE any luma write, so a later colour row in the same frame can box-blend
    // it. A colour row instead stages inside `upsample_440_chroma_sited_u16` (Bottom)
    // or its own Top branch after reading the previous lookback, so this is skipped
    // for it; a no-output row returned early above.
    if (bottom_v || top_v) && !want_color {
      stage_440_chroma_prev_u16(chroma_prev_u16, chroma_prev_row, row.u(), row.v(), idx, w);
    }

    // RFC #238 S8c: the row is now past its fallible preflight — every scratch is
    // reserved and every remaining write targets an attached output buffer or the
    // pre-grown RGB scratch, so it cannot fail. Commit the vertical-siting freeze
    // here (mirroring the resample arms' accept-time SET), at the single point all
    // output sub-paths pass through before diverging. Only the frame's first
    // output-bearing row sets it; the CHECK above rejects any later phase flip.
    if frozen_chroma_bottom_v.is_none() {
      *frozen_chroma_bottom_v = Some(bottom_v);
      *frozen_chroma_top_v = Some(top_v);
    }

    // Full-width chroma, reconstructed ONCE per row for the bottom-sited decode and
    // reused by every colour output (u16 and u8). Infallible — the scratches were
    // reserved above (`stage = true`: the direct decode's post-reconstruction work
    // is infallible, so advancing the lookback here is safe). The co-sited / Center
    // / Top sitings leave it `None`, so the decodes read `row.u()` / `row.v()`
    // straight and the output stays byte-identical.
    let centered = if bottom_v && want_color {
      Some(upsample_440_chroma_sited_u16::<BITS>(
        chroma_full_u16,
        chroma_prev_u16,
        chroma_prev_row,
        row.u(),
        row.v(),
        idx,
        bottom_v,
        true,
        w,
        BE,
        use_simd,
      ))
    } else {
      None
    };
    let matrix = row.matrix();
    let full_range = row.full_range();

    if let Some(luma) = luma.as_deref_mut() {
      let dst = &mut luma[one_plane_start..one_plane_end];
      for (d, &s) in dst.iter_mut().zip(row.y().iter()) {
        // Normalize BE-encoded wire bytes to host-native before the
        // luma downshift — see Yuv420p9 luma path for rationale.
        let logical = if BE { u16::from_be(s) } else { u16::from_le(s) };
        *d = (logical >> (BITS - 8)) as u8;
      }
    }

    // RFC #238 Top (`v = 0`) FORWARD one-row delay for the identity colour decode.
    // An EVEN output row is a plain co-sited decode; an ODD output row needs the
    // vertical box-average of its chroma row and the NEXT one, so it is HELD
    // (`chroma_top_pending` + the buffered `chroma_top_y_u16`) and emitted at the
    // following even row, forward-blended through the SAME `Bottom`-EVEN
    // `upsample_440_chroma_sited_u16(bottom_v = true)` kernel at the even index. The
    // trailing odd row of the frame clamps to a co-sited decode. Luma was written
    // above (siting-independent); only colour is delayed. Reuses the wire 4:4:4
    // identity colour helper for both feeds so a delayed decode is bit-identical to
    // an in-order one.
    if top_v && want_color {
      let is_last = idx + 1 == h;
      // Flush the held odd predecessor at this even row, forward-blending the
      // previous chroma row (`chroma_prev_u16`) with the current one.
      if idx & 1 == 0
        && let Some((p_idx, p_matrix, p_full_range)) = chroma_top_pending.take()
      {
        let (u_full, v_full) = upsample_440_chroma_sited_u16::<BITS>(
          chroma_full_u16,
          chroma_prev_u16,
          chroma_prev_row,
          row.u(),
          row.v(),
          idx,
          true,
          false,
          w,
          BE,
          use_simd,
        );
        yuv444p_top_identity_color_row::<BITS, BE>(
          rgb,
          rgba,
          rgb_u16,
          rgba_u16,
          hsv,
          rgb_scratch,
          &chroma_top_y_u16[..w],
          u_full,
          v_full,
          p_idx * w,
          p_idx * w + w,
          w,
          h,
          p_matrix,
          p_full_range,
          use_simd,
        )?;
      }
      if idx & 1 == 1 && !is_last {
        // Defer this odd row: copy its wire Y (the borrow expires after `process`)
        // and record its decode params; its colour emits at the next even row.
        chroma_top_y_u16[..w].copy_from_slice(&row.y()[..w]);
        *chroma_top_pending = Some((idx, matrix, full_range));
      } else {
        // Even row (co-sited `c[i]`) or the trailing odd row (bottom-edge clamp →
        // co-sited): a plain co-sited decode of the current chroma row.
        yuv444p_top_identity_color_row::<BITS, BE>(
          rgb,
          rgba,
          rgb_u16,
          rgba_u16,
          hsv,
          rgb_scratch,
          row.y(),
          row.u(),
          row.v(),
          one_plane_start,
          one_plane_end,
          w,
          h,
          matrix,
          full_range,
          use_simd,
        )?;
      }
      // Refresh the lookback with the current chroma row so the NEXT even row's
      // odd-flush box-blends this row as its predecessor (after the read above).
      stage_440_chroma_prev_u16(chroma_prev_u16, chroma_prev_row, row.u(), row.v(), idx, w);
      return Ok(());
    }

    // ===== u16 RGB / RGBA path (Strategy A) =====
    // `Bottom` (`centered` is `Some`) feeds the full-width vertically-blended chroma
    // reconstructed above; every co-sited siting reads `row.u()` / `row.v()`.
    if want_rgba_u16 && !want_rgb_u16 {
      let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row =
        rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
      if let Some((u_full, v_full)) = centered {
        yuv444p10_to_rgba_u16_row_endian(
          row.y(),
          u_full,
          v_full,
          rgba_u16_row,
          w,
          matrix,
          full_range,
          use_simd,
          BE,
        );
      } else {
        yuv444p10_to_rgba_u16_row_endian(
          row.y(),
          row.u(),
          row.v(),
          rgba_u16_row,
          w,
          matrix,
          full_range,
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
      if let Some((u_full, v_full)) = centered {
        yuv444p10_to_rgb_u16_row_endian(
          row.y(),
          u_full,
          v_full,
          rgb_u16_row,
          w,
          matrix,
          full_range,
          use_simd,
          BE,
        );
      } else {
        yuv444p10_to_rgb_u16_row_endian(
          row.y(),
          row.u(),
          row.v(),
          rgb_u16_row,
          w,
          matrix,
          full_range,
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
    // HSV-without-RGB-or-RGBA goes through the direct `yuv444p10_to_hsv_row_endian`
    // kernel (no source-width RGB scratch — the SIMD path stages a fixed 8-bit RGB
    // chunk internally). RGB or RGBA also attached keeps the convert-once-then-derive
    // path alive via `need_rgb_kernel`. `Bottom` first reconstructs the full-width
    // vertically-blended chroma (`centered`).
    let want_hsv_direct = want_hsv && !want_rgb && !want_rgba;
    let need_rgb_kernel = want_rgb || (want_hsv && want_rgba);

    if want_hsv_direct {
      let hsv = hsv.as_mut().expect("want_hsv_direct implies hsv attached");
      let (h, s, v) = hsv.hsv();
      let (yu, uu, vu) = match centered {
        Some((u_full, v_full)) => (row.y(), u_full, v_full),
        None => (row.y(), row.u(), row.v()),
      };
      yuv444p10_to_hsv_row_endian(
        yu,
        uu,
        vu,
        &mut h[one_plane_start..one_plane_end],
        &mut s[one_plane_start..one_plane_end],
        &mut v[one_plane_start..one_plane_end],
        w,
        matrix,
        full_range,
        use_simd,
        BE,
      );
      return Ok(());
    }

    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      let (yu, uu, vu) = match centered {
        Some((u_full, v_full)) => (row.y(), u_full, v_full),
        None => (row.y(), row.u(), row.v()),
      };
      yuv444p10_to_rgba_row_endian(yu, uu, vu, rgba_row, w, matrix, full_range, use_simd, BE);
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

    let (yu, uu, vu) = match centered {
      Some((u_full, v_full)) => (row.y(), u_full, v_full),
      None => (row.y(), row.u(), row.v()),
    };
    yuv444p10_to_rgb_row_endian(yu, uu, vu, rgb_row, w, matrix, full_range, use_simd, BE);

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

// ---- Yuv440p12 impl -----------------------------------------------------

impl<'a, R, const BE: bool> MixedSinker<'a, Yuv440p12<BE>, R> {
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

  /// Attaches a packed **8-bit** RGBA output buffer. Yuv440p12 reuses
  /// the `BITS = 12` 4:4:4 RGBA kernel; alpha = `0xFF`.
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

impl<R, const BE: bool> Yuv440p12Sink<BE> for MixedSinker<'_, Yuv440p12<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, Yuv440p12<BE>, R> {
  type Input<'r> = Yuv440p12Row<'r>;
  type Error = MixedSinkerError;

  #[cfg_attr(not(tarpaulin), inline(always))]
  fn kernel_matrix(&self) -> crate::KernelMatrix {
    self.kernel_matrix
  }
  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    reset_high_bit_yuv_streams(self);
    Ok(())
  }

  fn process(&mut self, row: Yuv440p12Row<'_>) -> Result<(), Self::Error> {
    const BITS: u32 = 12;
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;
    // Chroma siting (RFC #238 S8c): 4:4:0 carries only a VERTICAL phase
    // (full-width chroma → no horizontal siting). `Copy`, so read it out before
    // the field split-borrow below.
    let chroma_location = self.chroma_location.clone();

    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Y12,
        idx,
        w,
        row.y().len(),
      )));
    }
    if row.u().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::UFull12,
        idx,
        w,
        row.u().len(),
      )));
    }
    if row.v().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::VFull12,
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
      native,
      native_planar_u16,
      frozen_native_route,
      chroma_full_u16,
      chroma_prev_u16,
      chroma_prev_row,
      // RFC #238 S8c: the 4:4:0 vertical (`Bottom`) chroma phase frozen on the
      // first output-bearing row. `frozen_chroma_centered` is cleared alongside it
      // in `begin_frame` (via `reset_high_bit_yuv_streams`) but never set here —
      // 4:4:0 has no horizontal phase.
      frozen_chroma_bottom_v,
      // RFC #238 Top: the `Top` (`v = 0`) vertical phase frozen alongside it, and
      // the forward one-row delay line (row-stage / filter / identity tiers).
      frozen_chroma_top_v,
      chroma_top_pending,
      chroma_top_y_u16,
      ..
    } = self;

    // Non-identity plan: feed the shared high-bit 4:4:4 triple-resample
    // tail (u8 color, independent native-u16 color, native Y). 4:4:0 is
    // full-width chroma (no horizontal upsampling, same per-row contract
    // as 4:4:4) — its vertical chroma sharing is already resolved by the
    // walker, which hands this luma row the (vertically-shared) full-width
    // `u` / `v`, so the converted RGB is full-res and the 4:4:4 tail binds
    // via the shared `yuv444pN_to_rgb_*` kernels. Yuv440p exposes no
    // `luma_u16` output, so it is `&mut None` and only `luma` (binned
    // native Y `>> (BITS - 8)`) is emitted. The span kind picks the engine
    // (area bin or signed-coefficient filter twin) — see the Yuv440p10 impl
    // for the full rationale; the filter tail clamps every sub-16-bit colour
    // sample AND the native Y to `(1 << BITS) - 1`.
    if let Some(plan) = plan.as_ref() {
      let matrix = row.matrix();
      let full_range = row.full_range();
      let (y, u, v) = (row.y(), row.u(), row.v());
      // RFC #238 S8c — 4:4:0 VERTICAL chroma siting. `Bottom` / `BottomLeft`
      // ([`chroma_440_bottom_sited_v`], `v = 1`) box-blends the even output row's
      // full-width chroma with the previous chroma row; every other siting keeps
      // the co-sited vertical decode (byte-identical). 4:4:0 has FULL-width chroma,
      // so there is NO horizontal phase — only the vertical axis is ever folded
      // (`h_phase` stays 0). The native fast tier folds `v = 1` into
      // `area_chroma_440`'s vertical weights; the RGB-domain reconstruction tiers
      // (row-stage RGB/RGBA/HSV, filter) box-blend full-width `u16` chroma via the
      // `chroma_prev_u16` lookback and decode 4:4:4. Those reconstruction arms
      // reserve chroma AFTER the resample preflight, so a rejected / out-of-sequence
      // row is caught first (the #180 reserve-after-preflight invariant).
      let bottom_v = chroma_440_bottom_sited_v(&chroma_location);
      let chroma_v_phase = if bottom_v { 1.0 } else { 0.0 };
      // RFC #238 Top (`v = 0`, FORWARD fold) — `Top` / `TopLeft`. 4:4:0 has no
      // horizontal phase, so both share the identical full-width vertical box
      // average. Disjoint from `bottom_v`. The native fast tier folds it into
      // `area_chroma_440`'s vertical weights via `top_v`; the reconstruction tiers
      // reconstruct each row's chroma through a FORWARD one-row delay (mirror of
      // `Bottom`'s backward `chroma_prev_u16` lookback).
      let top_v = chroma_440_top_sited_v(&chroma_location);
      // Whether this row produces any colour output (and so runs the bottom-sited
      // chroma reconstruction). A luma-only row bins native Y unchanged (siting is a
      // chroma-only property).
      let want_color =
        rgb.is_some() || rgba.is_some() || hsv.is_some() || rgb_u16.is_some() || rgba_u16.is_some();
      // Whether this call carries any output — the EXACT set both tiers' preflights
      // test. The route / siting freezes only on an output-bearing row a tier
      // ACCEPTS; a no-output call consumes no stream state, so it must not freeze.
      let need_output = luma.is_some() || want_color;
      // Freeze the effective 4:4:0 vertical chroma siting on the first
      // output-bearing row (mirrors the `frozen_native_route` freeze below). This
      // CHECK is at the always-compiled choke point every tier passes through; the
      // matching SET rides each tier's accept path (never before dispatch, so a
      // rejected row leaves it unset for a corrected retry). A later row observing a
      // different vertical phase would bin a mixture of co-sited and
      // bottom-/top-folded chroma, so it is rejected here before any reconstruction.
      // 4:4:0 folds only the vertical axis, so `frozen_chroma_bottom_v` is the
      // set/unset sentinel; the Top flag is compared alongside because co-sited and
      // `Top` share `frozen_chroma_bottom_v = Some(false)`.
      if need_output
        && let Some(frozen) = *frozen_chroma_bottom_v
        && (frozen != bottom_v || *frozen_chroma_top_v != Some(top_v))
      {
        return Err(MixedSinkerError::ChromaSitingChanged(
          ChromaSitingChanged::new(idx),
        ));
      }
      if plan.kind().is_filter() {
        if bottom_v && want_color {
          // Bottom single-kernel filter: reconstruct full-width `u16` chroma, but
          // ONLY after the resample preflight (frozen-output + sequence), so an
          // out-of-sequence / rejected row is caught before the chroma reservation
          // (#180). Reject a multi-kernel (BICUBLIN) plan BEFORE the reserve too —
          // hoisted from the delegate's first act (idempotent, it re-runs it).
          // `packed_yuv444_triple_filter_resample` owns the transactional commit.
          plan.ensure_single_kernel_filter()?;
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
          reserve_440_chroma_prev_u16(chroma_prev_u16, w, h)?;
          // `stage = false`: DEFER the lookback advance until AFTER the fallible
          // filter commit accepts the row (below), so a rejected row leaves the
          // predecessor in place for a clean retry (#180 state-atomicity).
          let (u_full, v_full) = upsample_440_chroma_sited_u16::<BITS>(
            chroma_full_u16,
            chroma_prev_u16,
            chroma_prev_row,
            u,
            v,
            idx,
            bottom_v,
            false,
            w,
            BE,
            use_simd,
          );
          let r = packed_yuv444_triple_filter_resample::<BITS>(
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
              yuv444p12_to_rgb_row_endian(
                y, u_full, v_full, scratch, w, matrix, full_range, use_simd, BE,
              )
            },
            |scratch| {
              yuv444p12_to_rgb_u16_row_endian(
                y, u_full, v_full, scratch, w, matrix, full_range, use_simd, BE,
              )
            },
            |scratch| deinterleave_y_high_bit_masked::<BITS, BE>(y, scratch, w),
          );
          // Bottom lookback: advance only AFTER the filter resample accepts the row,
          // so a rejected row leaves the predecessor for a clean retry — the sited
          // reconstruction read it above but did not stage.
          if r.is_ok() {
            stage_440_chroma_prev_u16(chroma_prev_u16, chroma_prev_row, u, v, idx, w);
          }
          if r.is_ok() && need_output && frozen_chroma_bottom_v.is_none() {
            *frozen_chroma_bottom_v = Some(bottom_v);
            *frozen_chroma_top_v = Some(top_v);
          }
          return r;
        }
        if top_v && want_color {
          // RFC #238 Top (`v = 0`) FORWARD one-row delay for the single-kernel
          // filter tier — the whole odd row is HELD and the following even row
          // filter-feeds TWO source rows (held forward-blend + current co-sited);
          // the trailing odd row clamps to a co-sited decode. Reject a multi-kernel
          // (BICUBLIN) plan BEFORE the reserves (idempotent, the delegate re-runs
          // it). Retry-atomic (#180) — the reserves precede either feed.
          plan.ensure_single_kernel_filter()?;
          match yuv440p_top_reconstruct_filter::<BITS, BE>(
            rgb_filter_stream,
            rgb_filter_stream_u16,
            luma_filter_stream_u16,
            resample_outputs,
            rgb,
            rgba,
            rgb_u16,
            rgba_u16,
            luma,
            hsv,
            rgb_scratch,
            rgb_scratch_u16,
            luma_scratch_u16,
            chroma_full_u16,
            chroma_prev_u16,
            chroma_prev_row,
            chroma_top_pending,
            chroma_top_y_u16,
            y,
            u,
            v,
            w,
            h,
            plan,
            idx,
            use_simd,
            matrix,
            full_range,
          )? {
            core::ops::ControlFlow::Break(()) => return Ok(()),
            core::ops::ControlFlow::Continue(()) => {}
          }
          if frozen_chroma_bottom_v.is_none() && need_output {
            *frozen_chroma_bottom_v = Some(bottom_v);
            *frozen_chroma_top_v = Some(top_v);
          }
          return Ok(());
        }
        let r = packed_yuv444_triple_filter_resample::<BITS>(
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
            yuv444p12_to_rgb_row_endian(y, u, v, scratch, w, matrix, full_range, use_simd, BE)
          },
          |scratch| {
            yuv444p12_to_rgb_u16_row_endian(y, u, v, scratch, w, matrix, full_range, use_simd, BE)
          },
          |scratch| deinterleave_y_high_bit_masked::<BITS, BE>(y, scratch, w),
        );
        if r.is_ok() && need_output && frozen_chroma_bottom_v.is_none() {
          *frozen_chroma_bottom_v = Some(bottom_v);
          *frozen_chroma_top_v = Some(top_v);
        }
        return r;
      }
      // Native / row-stage route split — see the high-bit 4:2:0 Yuv420p impl
      // for the CHECK-before / SET-after `frozen_native_route` contract. Reuses
      // the `need_output` computed for the siting freeze above.
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
          native_eligible: YUV440P_HIGH_BIT_NATIVE_ELIGIBLE,
          with_native: *native,
          area_plan: true,
        },
      );
      match insertion {
        InsertionPoint::NativeCodes => {
          // 4:4:0: chroma `w x h/2` — full width, half height; a chroma row per
          // TWO Y rows (`chroma_vsub = 2`, like 4:2:0 vertically; `chroma_w = w`),
          // chroma plan full-width horizontal + luma-domain `area_halved` vertical
          // (or the RFC #238 S8c `Bottom` `v = 1` phased-V fold).
          //
          // Point-of-use siting invalidation (see the high-bit 4:2:0 native arm):
          // `chroma_location` can change at ANY point before this row (including
          // AFTER `begin_frame`, before row 0), so re-check the cached join HERE and
          // drop it when its folded chroma plan was built for a different vertical
          // phase; `yuv_planar16_process_native` rebuilds it with the current siting.
          // Retry-atomic: drop ONLY on the IN-SEQUENCE fresh-frame first row
          // (`idx == 0`, `next_y() == 0`); an out-of-sequence first row after a
          // siting change is left for the delegate to reject against the INTACT join.
          // Transactional: move the stale join OUT and let the delegate build the
          // replacement into `native_planar_u16` (its build runs BEFORE it inserts,
          // so a build failure leaves the field `None`); on such a failure, restore
          // the intact prior-phase join so the REJECTED row mutates nothing. A
          // luma-only join carries no chroma plan (siting-independent).
          let stale_native = idx == 0
            && native_planar_u16.as_ref().is_some_and(|join| {
              (join.chroma_bottom() == Some(!bottom_v) || join.chroma_top() == Some(!top_v))
                && join.next_y() == 0
            });
          let prev_native = if stale_native {
            native_planar_u16.take()
          } else {
            None
          };
          let native_result = yuv_planar16_process_native::<BITS, BE>(
            plan,
            native_planar_u16,
            resample_outputs,
            rgb,
            rgba,
            rgb_u16,
            rgba_u16,
            luma,
            // The high-bit planar 4:4:0 family exposes no `luma_u16` output.
            &mut None,
            hsv,
            rgb_scratch,
            rgb_scratch_u16,
            y,
            u,
            v,
            matrix,
            full_range,
            idx,
            w,
            h,
            2,
            w,
            || {
              ResamplePlan::area_chroma_440(
                w,
                h,
                plan.out_w(),
                plan.out_h(),
                0.0,
                chroma_v_phase,
                top_v,
              )
            },
            use_simd,
          );
          // Restore the taken stale-phase join if the delegate's rebuild was rejected
          // at any pre-feed step: it leaves the field `None` on such a failure, so
          // restoring the intact prior-phase join leaves the rejected row mutating no
          // join state. A non-stale row took nothing.
          if stale_native && native_result.is_err() {
            *native_planar_u16 = prev_native;
          }
          native_result?;
          if frozen_native_route.is_none() && need_output {
            *frozen_native_route = Some(true);
          }
          // RFC #238 S8c: freeze the siting on the same accepted output row.
          if frozen_chroma_bottom_v.is_none() && need_output {
            *frozen_chroma_bottom_v = Some(bottom_v);
            *frozen_chroma_top_v = Some(top_v);
          }
          return Ok(());
        }
        InsertionPoint::EncodedOutput => {
          if bottom_v && want_color {
            // Bottom row-stage: reconstruct full-width `u16` chroma AFTER the
            // resample preflight (frozen-output + sequence), so an out-of-sequence /
            // rejected row is caught before the chroma reservation (#180).
            // `packed_yuv444_triple_resample` re-runs the idempotent preflight and
            // owns the transactional commit. Gated by `want_color` — a luma-only
            // Bottom row never calls the RGB converter, so it stays on the co-sited
            // arm (which only bins native Y).
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
            reserve_440_chroma_prev_u16(chroma_prev_u16, w, h)?;
            // `stage = false`: DEFER the lookback advance until AFTER the fallible
            // row-stage commit accepts the row (below), so a rejected row leaves the
            // predecessor for a clean retry (#180 state-atomicity).
            let (u_full, v_full) = upsample_440_chroma_sited_u16::<BITS>(
              chroma_full_u16,
              chroma_prev_u16,
              chroma_prev_row,
              u,
              v,
              idx,
              bottom_v,
              false,
              w,
              BE,
              use_simd,
            );
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
              |scratch| {
                yuv444p12_to_rgb_row_endian(
                  y, u_full, v_full, scratch, w, matrix, full_range, use_simd, BE,
                )
              },
              |scratch| {
                yuv444p12_to_rgb_u16_row_endian(
                  y, u_full, v_full, scratch, w, matrix, full_range, use_simd, BE,
                )
              },
              |scratch| deinterleave_y_high_bit_masked::<BITS, BE>(y, scratch, w),
            )?;
            // Bottom lookback: advance only AFTER the row-stage resample accepts the
            // row (the `?` above already returned any reject), so a rejected row
            // leaves the predecessor for a clean retry — the sited reconstruction
            // read it above but did not stage.
            stage_440_chroma_prev_u16(chroma_prev_u16, chroma_prev_row, u, v, idx, w);
          } else if top_v && want_color {
            // RFC #238 Top (`v = 0`) FORWARD one-row delay for the row-stage tier —
            // the whole odd row is HELD and the following even row feeds TWO source
            // rows (held forward-blend + current co-sited); the trailing odd row
            // clamps to a co-sited decode. Retry-atomic (#180) — the reserves
            // precede either feed. A preflight decline returns without freezing.
            match yuv440p_top_reconstruct_area::<BITS, BE>(
              rgb_stream,
              rgb_stream_u16,
              luma_stream_u16,
              resample_outputs,
              rgb,
              rgba,
              rgb_u16,
              rgba_u16,
              luma,
              hsv,
              rgb_scratch,
              rgb_scratch_u16,
              luma_scratch_u16,
              chroma_full_u16,
              chroma_prev_u16,
              chroma_prev_row,
              chroma_top_pending,
              chroma_top_y_u16,
              y,
              u,
              v,
              w,
              h,
              plan,
              idx,
              use_simd,
              matrix,
              full_range,
            )? {
              core::ops::ControlFlow::Break(()) => return Ok(()),
              core::ops::ControlFlow::Continue(()) => {}
            }
          } else {
            // Co-sited RGB / RGBA / HSV / luma: the byte-identical row-stage tail.
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
              |scratch| {
                yuv444p12_to_rgb_row_endian(y, u, v, scratch, w, matrix, full_range, use_simd, BE)
              },
              |scratch| {
                yuv444p12_to_rgb_u16_row_endian(
                  y, u, v, scratch, w, matrix, full_range, use_simd, BE,
                )
              },
              |scratch| deinterleave_y_high_bit_masked::<BITS, BE>(y, scratch, w),
            )?;
          }
          if frozen_native_route.is_none() && need_output {
            *frozen_native_route = Some(false);
          }
          // RFC #238 S8c: freeze the siting on the same accepted output row.
          if frozen_chroma_bottom_v.is_none() && need_output {
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

    // Resolve the output set + the no-output guard up front, BEFORE any per-row
    // offset arithmetic, so a no-output call runs NOTHING — no `idx * w` (which
    // could overflow on a 32-bit target for a call that never ran an attach-time
    // `w x h` validation), no allocation, no lookback priming. `need_output` MUST
    // include the u16 twins (`rgb_u16` / `rgba_u16`), else a u16-only sink would
    // skip the siting freeze / lookback.
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();
    let want_rgb_u16 = rgb_u16.is_some();
    let want_rgba_u16 = rgba_u16.is_some();
    // A bottom-sited row maintains the vertical lookback even when it produces only
    // luma (the luma-only staging below), so `want_color` gates only the colour
    // scratches / reconstruction.
    let want_color = want_rgb || want_rgba || want_hsv || want_rgb_u16 || want_rgba_u16;
    let need_output = want_color || luma.is_some();
    if !need_output {
      return Ok(());
    }

    // Single-plane row ranges cannot overflow: the no-output guard above ensures
    // >= 1 output is attached, so its attach-time `w x h x 1` validation ran and
    // `w x h` fits `usize`; with `idx < h` that makes `(idx + 1) * w <= h * w` fit.
    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // Chroma siting (RFC #238 S8c): `Bottom` (v = 1) box-blends the even output
    // row's full-width chroma with the previous chroma row; the default / Center /
    // Top sitings keep the byte-identical co-sited decode (their odd output row
    // needs the *next* chroma row, deferred). 4:4:0 has full-width chroma, so there
    // is no horizontal phase.
    let bottom_v = chroma_440_bottom_sited_v(&chroma_location);
    // RFC #238 Top (`v = 0`, FORWARD one-row delay) — `Top` / `TopLeft`. An ODD
    // output row needs the NEXT chroma row (unavailable when it arrives), so its
    // colour output is HELD (`chroma_top_pending` / `chroma_top_y_u16`) and emitted
    // at the following even row; handled in its own branch below. Disjoint from
    // `bottom_v`. Luma is siting-independent and written in order.
    let top_v = chroma_440_top_sited_v(&chroma_location);

    // RFC #238 S8c: freeze the vertical chroma siting on the direct (identity) path
    // too. The resample branch above rejects a mid-frame vertical-phase flip via
    // `frozen_chroma_bottom_v`, and this path must honour the SAME per-frame
    // invariant: `set_chroma_location` is public, so without this a caller could
    // emit rows `Bottom`, switch to a co-sited location, then back, binning a
    // mixture of co-sited and bottom-folded chroma (and, after a co-sited gap,
    // box-blend a later even row against a stale lookback). CHECK here — before any
    // reservation or lookback mutation; the matching SET rides the row's accept
    // point below (after the fallible preflight). `need_output` is already
    // established (the no-output guard returned above).
    if let Some(frozen) = *frozen_chroma_bottom_v
      && (frozen != bottom_v || *frozen_chroma_top_v != Some(top_v))
    {
      return Err(MixedSinkerError::ChromaSitingChanged(
        ChromaSitingChanged::new(idx),
      ));
    }

    // Atomicity preflight (#308, cf. the crate's #180 resample fix and the high-bit
    // 4:2:0 sibling): reserve EVERY fallible row scratch this row needs BEFORE any
    // output row (luma included) is written, so an allocator refusal returns a typed
    // `AllocationFailed` leaving the output frame untouched. Growable here:
    //  1. the bottom-sited full-width blend scratch (`chroma_full_u16`, `2 * w`),
    //     only on a colour Bottom decode;
    //  2. the bottom-sited vertical-phase full-width lookback (`chroma_prev_u16`), on
    //     every Bottom OUTPUT row — colour OR luma-only — because the lookback is
    //     maintained so a later colour row can box-blend it (the luma-only staging
    //     runs below, before any luma write); and
    //  3. the u8 RGB row scratch, exactly `want_hsv && want_rgba && !want_rgb`.
    // The u16 RGB / RGBA outputs write straight into their caller buffers and never
    // grow a scratch; this format exposes no luma_u16. A no-output row returned early
    // above, so it never reaches here.
    if (bottom_v || top_v) && want_color {
      reserve_420_chroma_full_u16(chroma_full_u16, w, h)?;
    }
    if bottom_v || top_v {
      reserve_440_chroma_prev_u16(chroma_prev_u16, w, h)?;
    }
    // Top forward-delay held-Y buffer: reserved up front — BEFORE any luma write —
    // so the later deferred-row copy + the two-row flush are infallible (#180).
    if top_v && want_color {
      reserve_420_chroma_top_y_u16(chroma_top_y_u16, w, h)?;
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

    // Bottom / Top sited LUMA-ONLY row: the colour upsample helper — which normally
    // refreshes the vertical lookback — won't run, so stage the current full-width
    // chroma row into the lookback HERE, after its preflight reservation above and
    // BEFORE any luma write, so a later colour row in the same frame can box-blend
    // it. A colour row instead stages inside `upsample_440_chroma_sited_u16` (Bottom)
    // or its own Top branch after reading the previous lookback, so this is skipped
    // for it; a no-output row returned early above.
    if (bottom_v || top_v) && !want_color {
      stage_440_chroma_prev_u16(chroma_prev_u16, chroma_prev_row, row.u(), row.v(), idx, w);
    }

    // RFC #238 S8c: the row is now past its fallible preflight — every scratch is
    // reserved and every remaining write targets an attached output buffer or the
    // pre-grown RGB scratch, so it cannot fail. Commit the vertical-siting freeze
    // here (mirroring the resample arms' accept-time SET), at the single point all
    // output sub-paths pass through before diverging. Only the frame's first
    // output-bearing row sets it; the CHECK above rejects any later phase flip.
    if frozen_chroma_bottom_v.is_none() {
      *frozen_chroma_bottom_v = Some(bottom_v);
      *frozen_chroma_top_v = Some(top_v);
    }

    // Full-width chroma, reconstructed ONCE per row for the bottom-sited decode and
    // reused by every colour output (u16 and u8). Infallible — the scratches were
    // reserved above (`stage = true`: the direct decode's post-reconstruction work
    // is infallible, so advancing the lookback here is safe). The co-sited / Center
    // / Top sitings leave it `None`, so the decodes read `row.u()` / `row.v()`
    // straight and the output stays byte-identical.
    let centered = if bottom_v && want_color {
      Some(upsample_440_chroma_sited_u16::<BITS>(
        chroma_full_u16,
        chroma_prev_u16,
        chroma_prev_row,
        row.u(),
        row.v(),
        idx,
        bottom_v,
        true,
        w,
        BE,
        use_simd,
      ))
    } else {
      None
    };
    let matrix = row.matrix();
    let full_range = row.full_range();

    if let Some(luma) = luma.as_deref_mut() {
      let dst = &mut luma[one_plane_start..one_plane_end];
      for (d, &s) in dst.iter_mut().zip(row.y().iter()) {
        // Normalize BE-encoded wire bytes to host-native before the
        // luma downshift — see Yuv420p9 luma path for rationale.
        let logical = if BE { u16::from_be(s) } else { u16::from_le(s) };
        *d = (logical >> (BITS - 8)) as u8;
      }
    }

    // RFC #238 Top (`v = 0`) FORWARD one-row delay for the identity colour decode.
    // An EVEN output row is a plain co-sited decode; an ODD output row needs the
    // vertical box-average of its chroma row and the NEXT one, so it is HELD
    // (`chroma_top_pending` + the buffered `chroma_top_y_u16`) and emitted at the
    // following even row, forward-blended through the SAME `Bottom`-EVEN
    // `upsample_440_chroma_sited_u16(bottom_v = true)` kernel at the even index. The
    // trailing odd row of the frame clamps to a co-sited decode. Luma was written
    // above (siting-independent); only colour is delayed. Reuses the wire 4:4:4
    // identity colour helper for both feeds so a delayed decode is bit-identical to
    // an in-order one.
    if top_v && want_color {
      let is_last = idx + 1 == h;
      // Flush the held odd predecessor at this even row, forward-blending the
      // previous chroma row (`chroma_prev_u16`) with the current one.
      if idx & 1 == 0
        && let Some((p_idx, p_matrix, p_full_range)) = chroma_top_pending.take()
      {
        let (u_full, v_full) = upsample_440_chroma_sited_u16::<BITS>(
          chroma_full_u16,
          chroma_prev_u16,
          chroma_prev_row,
          row.u(),
          row.v(),
          idx,
          true,
          false,
          w,
          BE,
          use_simd,
        );
        yuv444p_top_identity_color_row::<BITS, BE>(
          rgb,
          rgba,
          rgb_u16,
          rgba_u16,
          hsv,
          rgb_scratch,
          &chroma_top_y_u16[..w],
          u_full,
          v_full,
          p_idx * w,
          p_idx * w + w,
          w,
          h,
          p_matrix,
          p_full_range,
          use_simd,
        )?;
      }
      if idx & 1 == 1 && !is_last {
        // Defer this odd row: copy its wire Y (the borrow expires after `process`)
        // and record its decode params; its colour emits at the next even row.
        chroma_top_y_u16[..w].copy_from_slice(&row.y()[..w]);
        *chroma_top_pending = Some((idx, matrix, full_range));
      } else {
        // Even row (co-sited `c[i]`) or the trailing odd row (bottom-edge clamp →
        // co-sited): a plain co-sited decode of the current chroma row.
        yuv444p_top_identity_color_row::<BITS, BE>(
          rgb,
          rgba,
          rgb_u16,
          rgba_u16,
          hsv,
          rgb_scratch,
          row.y(),
          row.u(),
          row.v(),
          one_plane_start,
          one_plane_end,
          w,
          h,
          matrix,
          full_range,
          use_simd,
        )?;
      }
      // Refresh the lookback with the current chroma row so the NEXT even row's
      // odd-flush box-blends this row as its predecessor (after the read above).
      stage_440_chroma_prev_u16(chroma_prev_u16, chroma_prev_row, row.u(), row.v(), idx, w);
      return Ok(());
    }

    // ===== u16 RGB / RGBA path (Strategy A) =====
    // `Bottom` (`centered` is `Some`) feeds the full-width vertically-blended chroma
    // reconstructed above; every co-sited siting reads `row.u()` / `row.v()`.
    if want_rgba_u16 && !want_rgb_u16 {
      let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row =
        rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
      if let Some((u_full, v_full)) = centered {
        yuv444p12_to_rgba_u16_row_endian(
          row.y(),
          u_full,
          v_full,
          rgba_u16_row,
          w,
          matrix,
          full_range,
          use_simd,
          BE,
        );
      } else {
        yuv444p12_to_rgba_u16_row_endian(
          row.y(),
          row.u(),
          row.v(),
          rgba_u16_row,
          w,
          matrix,
          full_range,
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
      if let Some((u_full, v_full)) = centered {
        yuv444p12_to_rgb_u16_row_endian(
          row.y(),
          u_full,
          v_full,
          rgb_u16_row,
          w,
          matrix,
          full_range,
          use_simd,
          BE,
        );
      } else {
        yuv444p12_to_rgb_u16_row_endian(
          row.y(),
          row.u(),
          row.v(),
          rgb_u16_row,
          w,
          matrix,
          full_range,
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
    // HSV-without-RGB-or-RGBA goes through the direct `yuv444p12_to_hsv_row_endian`
    // kernel (no source-width RGB scratch — the SIMD path stages a fixed 8-bit RGB
    // chunk internally). RGB or RGBA also attached keeps the convert-once-then-derive
    // path alive via `need_rgb_kernel`. `Bottom` first reconstructs the full-width
    // vertically-blended chroma (`centered`).
    let want_hsv_direct = want_hsv && !want_rgb && !want_rgba;
    let need_rgb_kernel = want_rgb || (want_hsv && want_rgba);

    if want_hsv_direct {
      let hsv = hsv.as_mut().expect("want_hsv_direct implies hsv attached");
      let (h, s, v) = hsv.hsv();
      let (yu, uu, vu) = match centered {
        Some((u_full, v_full)) => (row.y(), u_full, v_full),
        None => (row.y(), row.u(), row.v()),
      };
      yuv444p12_to_hsv_row_endian(
        yu,
        uu,
        vu,
        &mut h[one_plane_start..one_plane_end],
        &mut s[one_plane_start..one_plane_end],
        &mut v[one_plane_start..one_plane_end],
        w,
        matrix,
        full_range,
        use_simd,
        BE,
      );
      return Ok(());
    }

    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      let (yu, uu, vu) = match centered {
        Some((u_full, v_full)) => (row.y(), u_full, v_full),
        None => (row.y(), row.u(), row.v()),
      };
      yuv444p12_to_rgba_row_endian(yu, uu, vu, rgba_row, w, matrix, full_range, use_simd, BE);
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

    let (yu, uu, vu) = match centered {
      Some((u_full, v_full)) => (row.y(), u_full, v_full),
      None => (row.y(), row.u(), row.v()),
    };
    yuv444p12_to_rgb_row_endian(yu, uu, vu, rgb_row, w, matrix, full_range, use_simd, BE);

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
