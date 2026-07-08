use super::{
  super::{
    ChromaSitingChanged, GeometryOverflow, InsufficientBuffer, MixedSinker, MixedSinkerError,
    NativeRouteChanged, RowIndexOutOfRange, RowShapeMismatch, RowSlice, WidthAlignment,
    check_dimensions_match, chroma_420_bottom_sited_v, chroma_420_center_sited_h,
    deinterleave_y_high_bit_masked, packed_yuv422_triple_filter_resample,
    packed_yuv422_triple_resample, planar_8bit::YUV422P_CENTERED_H_PHASE,
    resample_preflight_check_only, reset_high_bit_yuv_streams, rgb_row_buf_or_scratch,
    rgba_plane_row_slice, rgba_u16_plane_row_slice,
  },
  yuv420p16_process_native,
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

/// The high-bit 4:2:0 planar formats (`Yuv420p9` … `Yuv420p16`) ship the
/// native 4:2:0 fast tier ([`yuv420p16_process_native`]), so each is
/// statically eligible to splice an [`AveragingDomain::Encoded`] area
/// downscale at the native codes.
const YUV420P_HIGH_BIT_NATIVE_ELIGIBLE: bool = true;

/// **Fallible preflight** for the centered-siting high-bit 4:2:0 chroma scratch
/// (#302) — the `u16` twin of [`reserve_420_chroma_full`](super::super::reserve_420_chroma_full).
/// Grows `chroma_full` to the checked `2 * width` `u16` so the later infallible
/// [`upsample_420_chroma_center_h_u16`] reuses an already-sized buffer. Split
/// out from the upsample so it can run **before any output row is written**
/// (luma included) — the crate's preflight-ordering atomicity contract (cf. the
/// #180 resample fix and the #314 high-bit atomicity pass): an allocator refusal
/// must leave the output frame *untouched*, never partially mutated.
///
/// Mirrors the 8-bit sibling's recoverable grow: the `2 * width` length is
/// `checked_mul`'d (→ [`GeometryOverflow`]) and `try_reserve_exact` precedes the
/// resize (→ [`ResampleError::AllocationFailed`]), so the failure is a typed,
/// recoverable error rather than an abort. `height` feeds the error payloads.
pub(crate) fn reserve_420_chroma_full_u16(
  chroma_full: &mut std::vec::Vec<u16>,
  width: usize,
  height: usize,
) -> Result<(), MixedSinkerError> {
  // Test-only failpoint: simulate a recoverable allocator refusal of the
  // chroma-scratch grow WITHOUT exhausting memory, so the atomicity regression
  // test can prove no output row (esp. luma) is written before this preflight.
  // Reuses the planar/semi-planar centered path's shared failpoint (`take()`
  // fires the armed flag exactly once); the non-test build compiles it away.
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

/// Horizontally upsamples the half-width `u16` U / V rows of a centered-siting
/// high-bit 4:2:0 source to full width into the **already-reserved**
/// `chroma_full` (#302), returning the two full-width chroma slices
/// `(u_full, v_full)`. The buffer is split `[0..width]` = U,
/// `[width..2*width]` = V; each half is filled by
/// [`chroma_upsample_2to1_center_h_u16`](crate::row::scalar::chroma_upsample_2to1_center_h_u16)
/// (the MPEG-1 / JPEG phase-0.5 reconstruction — masking each sample to the low
/// `BITS` and operating in the source's wire byte order), then fed to the
/// high-bit 4:4:4 decode kernels with the same `big_endian` flag — so the
/// centered path reuses their SIMD backends and stays bit-identical per tier.
/// `BITS` is threaded through so the per-sample mask matches the decode kernels'
/// `bits_mask::<BITS>()` (it is `u16::MAX` / a no-op at `BITS = 16`).
///
/// **Infallible**: the caller must have run [`reserve_420_chroma_full_u16`] up
/// front (every centered-siting output path does, before any output write), so
/// `chroma_full` is guaranteed `>= 2 * width` here and `2 * width` cannot
/// overflow. Only the centered sitings reach here; the default
/// left/unspecified path never touches this scratch.
pub(crate) fn upsample_420_chroma_center_h_u16<'s, const BITS: u32>(
  chroma_full: &'s mut [u16],
  u_half: &[u16],
  v_half: &[u16],
  width: usize,
  big_endian: bool,
) -> (&'s [u16], &'s [u16]) {
  debug_assert!(
    chroma_full.len() >= 2 * width,
    "chroma_full must be reserved via reserve_420_chroma_full_u16 first"
  );
  let (u_full, v_full) = chroma_full[..2 * width].split_at_mut(width);
  crate::row::scalar::chroma_upsample_2to1_center_h_u16::<BITS>(u_half, u_full, width, big_endian);
  crate::row::scalar::chroma_upsample_2to1_center_h_u16::<BITS>(v_half, v_full, width, big_endian);
  (u_full, v_full)
}

/// Co-sited (`h = 0`) full-width HIGH-BIT 4:2:0 chroma reconstruction — the
/// co-sited twin of [`upsample_420_chroma_center_h_u16`] used by the `BottomLeft`
/// siting, whose horizontal axis is nearest-neighbor rather than the `h = 0.5`
/// centered phase. Each half is filled by
/// [`chroma_upsample_2to1_cosited_h_u16`](crate::row::scalar::chroma_upsample_2to1_cosited_h_u16)
/// (a plain 2× replicate, masking each sample to the low `BITS` in wire byte
/// order), then fed to the same high-bit 4:4:4 decode kernels — so a `BottomLeft`
/// decode reconstructs its `v = 1` fold ([`upsample_420_chroma_sited_u16`]) atop a
/// consistent full-width horizontal staging and reuses those SIMD backends.
///
/// **Infallible**: the caller must have run [`reserve_420_chroma_full_u16`] up
/// front, so `chroma_full` is `>= 2 * width` here.
pub(crate) fn upsample_420_chroma_cosited_h_u16<'s, const BITS: u32>(
  chroma_full: &'s mut [u16],
  u_half: &[u16],
  v_half: &[u16],
  width: usize,
  big_endian: bool,
) -> (&'s [u16], &'s [u16]) {
  debug_assert!(
    chroma_full.len() >= 2 * width,
    "chroma_full must be reserved via reserve_420_chroma_full_u16 first"
  );
  let (u_full, v_full) = chroma_full[..2 * width].split_at_mut(width);
  crate::row::scalar::chroma_upsample_2to1_cosited_h_u16::<BITS>(u_half, u_full, width, big_endian);
  crate::row::scalar::chroma_upsample_2to1_cosited_h_u16::<BITS>(v_half, v_full, width, big_endian);
  (u_full, v_full)
}

/// **Fallible preflight** for the bottom-sited (`AVCHROMA_LOC_BOTTOM`)
/// vertical-phase HIGH-BIT 4:2:0 chroma lookback (RFC #238 S6d) — the `u16` twin
/// of [`reserve_420_chroma_prev`](super::super::reserve_420_chroma_prev). Grows
/// `chroma_prev` to `width` `u16` (half-width U then V, wire byte order) so the
/// later infallible [`upsample_420_chroma_sited_u16`] can read the previous
/// chroma row for the even output row's vertical box blend. Split out from the
/// upsample, like [`reserve_420_chroma_full_u16`], so it runs **before any
/// output row is written** — the crate's preflight-ordering atomicity contract
/// (an allocator refusal must leave the output frame untouched).
/// `try_reserve_exact` precedes the resize (→ [`ResampleError::AllocationFailed`]);
/// `width` already fits `usize` (the Y plane is `width * height`), so no
/// `checked_mul` is needed. `height` feeds the error payload.
pub(crate) fn reserve_420_chroma_prev_u16(
  chroma_prev: &mut std::vec::Vec<u16>,
  width: usize,
  height: usize,
) -> Result<(), MixedSinkerError> {
  // Test-only failpoint: reuse the 8-bit path's shared bottom-sited lookback
  // failpoint (`take()` fires the armed flag exactly once), so the atomicity
  // regression test can prove no output row is written before this preflight.
  // The non-test build compiles it away.
  #[cfg(all(test, feature = "std", feature = "yuv-planar"))]
  if super::super::FORCE_CHROMA_PREV_ALLOC_FAILURE.with(|f| f.take()) {
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

/// Stages the current half-width `u16` chroma row into the bottom-sited
/// vertical-phase lookback (RFC #238 S6d) — the `u16` twin of
/// [`stage_420_chroma_prev`](super::super::stage_420_chroma_prev): copies
/// `u_half` then `v_half` into `chroma_prev` (`[0..width/2]` = U,
/// `[width/2..width]` = V, wire byte order preserved) and tags it with the chroma
/// row it now holds (`idx / 2`), so a *later* even output row can validate
/// (`chroma_prev_row == Some(its_chroma_row - 1)`) and box-blend it.
///
/// Called on EVERY accepted bottom-sited row — the colour-decode path
/// ([`upsample_420_chroma_sited_u16`], after it has read the *previous* lookback)
/// and the luma-only path (which never reaches that helper) — so the lookback is
/// always current regardless of which outputs are attached. **Infallible**: the
/// caller must have run [`reserve_420_chroma_prev_u16`] up front.
#[inline]
pub(crate) fn stage_420_chroma_prev_u16(
  chroma_prev: &mut [u16],
  chroma_prev_row: &mut Option<usize>,
  u_half: &[u16],
  v_half: &[u16],
  idx: usize,
  width: usize,
) {
  debug_assert!(
    chroma_prev.len() >= width,
    "chroma_prev must be reserved via reserve_420_chroma_prev_u16 first"
  );
  let half = width / 2;
  chroma_prev[..half].copy_from_slice(&u_half[..half]);
  chroma_prev[half..width].copy_from_slice(&v_half[..half]);
  *chroma_prev_row = Some(idx / 2);
}

/// Siting-aware full-width HIGH-BIT 4:2:0 chroma reconstruction (RFC #238 S6d) —
/// the `u16` twin of [`upsample_420_chroma_sited`](super::super::planar_8bit),
/// folding the optional **bottom-sited vertical** phase into the horizontal
/// centered `u16` upsample and maintaining the one-row chroma lookback. Returns
/// the full-width `(u_full, v_full)` `u16` slices the high-bit 4:4:4 decode reads,
/// staged in the already-reserved `chroma_full` in the source's wire byte order.
///
/// `bottom_v` is [`chroma_420_bottom_sited_v`](super::super::chroma_420_bottom_sited_v)
/// (true only for `ChromaLocation::Bottom`, which is also horizontally centered):
///
/// - On an **even** luma row (`idx & 1 == 0`) with `bottom_v`, each chroma sample
///   is the vertical box average of the previous chroma row (`chroma_prev`) and
///   the current row, then horizontally centered — one fused pass via
///   [`chroma_upsample_420_bottom_even_h_u16`](crate::row::scalar::chroma_upsample_420_bottom_even_h_u16)
///   — **but only when `chroma_prev` provably holds the wanted predecessor**
///   chroma row `idx/2 - 1` (`chroma_prev_row == Some(idx/2 - 1)`). Otherwise —
///   the top edge (`idx == 0`), a fresh frame, or a direct `process` caller that
///   replayed / skipped / reordered rows or attached colour late — it falls back
///   to the plain centered upsample of the *current* chroma row (the top-edge
///   clamp), so the blend NEVER mixes stale chroma from an older pair or frame.
/// - Otherwise (odd luma row, or any non-bottom centered siting) the current
///   chroma row is horizontally centered with the plain
///   [`upsample_420_chroma_center_h_u16`] — bit-identical to the horizontal-only
///   path.
///
/// When `stage` is set, on every bottom-sited row the current half-width chroma
/// row is copied into `chroma_prev` (tagged `idx / 2`), so the next pair's even
/// row can validate and read it. The direct decode passes `stage = true` (its
/// post-reconstruction work is infallible); the resample reconstruction arms pass
/// `stage = false` and defer [`stage_420_chroma_prev_u16`] until AFTER their
/// fallible resample commit accepts the row, so a rejected row leaves the lookback
/// pointing at the predecessor and a retry still box-blends it (state-atomic,
/// #180). The caller must have run [`reserve_420_chroma_full_u16`] (always) and,
/// when `bottom_v`, [`reserve_420_chroma_prev_u16`] up front, so both buffers are
/// sized here and this is infallible.
#[allow(clippy::too_many_arguments)]
pub(crate) fn upsample_420_chroma_sited_u16<'s, const BITS: u32>(
  chroma_full: &'s mut [u16],
  chroma_prev: &mut [u16],
  chroma_prev_row: &mut Option<usize>,
  u_half: &[u16],
  v_half: &[u16],
  idx: usize,
  bottom_v: bool,
  center_h: bool,
  stage: bool,
  width: usize,
  big_endian: bool,
) -> (&'s [u16], &'s [u16]) {
  debug_assert!(
    chroma_full.len() >= 2 * width,
    "chroma_full must be reserved via reserve_420_chroma_full_u16 first"
  );
  let half = width / 2;
  let chroma_row = idx / 2;
  // The bottom-sited EVEN row box-blends only when the lookback PROVABLY holds
  // the wanted predecessor `chroma_row - 1`; otherwise it clamps to the current
  // row (never stale). Every other case (non-bottom siting, the bottom-sited odd
  // row, or an unvalidated even row) is the plain horizontal upsample — centered
  // or co-sited per `center_h`.
  let do_vblend =
    bottom_v && idx & 1 == 0 && chroma_row > 0 && *chroma_prev_row == Some(chroma_row - 1);

  let result = if do_vblend {
    debug_assert!(
      chroma_prev.len() >= width,
      "chroma_prev must be reserved via reserve_420_chroma_prev_u16 first"
    );
    let (u_full, v_full) = chroma_full[..2 * width].split_at_mut(width);
    let (u_prev, v_prev) = chroma_prev[..width].split_at(half);
    if center_h {
      crate::row::scalar::chroma_upsample_420_bottom_even_h_u16::<BITS>(
        u_prev, u_half, u_full, width, big_endian,
      );
      crate::row::scalar::chroma_upsample_420_bottom_even_h_u16::<BITS>(
        v_prev, v_half, v_full, width, big_endian,
      );
    } else {
      crate::row::scalar::chroma_upsample_420_bottomleft_even_h_u16::<BITS>(
        u_prev, u_half, u_full, width, big_endian,
      );
      crate::row::scalar::chroma_upsample_420_bottomleft_even_h_u16::<BITS>(
        v_prev, v_half, v_full, width, big_endian,
      );
    }
    (&*u_full, &*v_full)
  } else if center_h {
    upsample_420_chroma_center_h_u16::<BITS>(chroma_full, u_half, v_half, width, big_endian)
  } else {
    upsample_420_chroma_cosited_h_u16::<BITS>(chroma_full, u_half, v_half, width, big_endian)
  };

  // Refresh the lookback with the current chroma row + its validity tag (after
  // the read above). Gated on `stage`: the direct decode refreshes it here (its
  // later work is infallible), while the resample reconstruction arms pass
  // `stage = false` and defer the refresh to AFTER their fallible commit accepts
  // the row. Only the bottom-sited path maintains it.
  if bottom_v && stage {
    stage_420_chroma_prev_u16(chroma_prev, chroma_prev_row, u_half, v_half, idx, width);
  }
  result
}

// ---- Yuv420p9 impl -----------------------------------------------------
//
// 9-bit 4:2:0 planar. AV_PIX_FMT_YUV420P9LE — niche AVC High 9 only.
// Reuses the Q15 i32 kernel family at `BITS = 9` via the
// `yuv420p9_to_rgb_*` row primitives (which dispatch to
// `yuv_420p_n_to_rgb_*<9>` internally).

impl<'a, R, const BE: bool> MixedSinker<'a, Yuv420p9<BE>, R> {
  /// Attaches a packed **`u16`** RGB output buffer. 9‑bit low‑packed
  /// (`(1 << 9) - 1 = 511` max).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgb_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_rgb_u16(buf)?;
    Ok(self)
  }
  /// In-place variant of [`with_rgb_u16`](Self::with_rgb_u16).
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

  /// Attaches a packed **8‑bit** RGBA output buffer. The 9‑bit YUV
  /// source is converted to 8‑bit RGBA via the same `BITS = 9` Q15
  /// kernel family used by [`Self::with_rgb`]; the fourth byte per
  /// pixel is alpha = `0xFF` (Yuv420p9 has no alpha plane).
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

  /// Attaches a packed **`u16`** RGBA output buffer. 9‑bit low‑packed
  /// (`(1 << 9) - 1 = 511` max). Length is measured in `u16`
  /// **elements** (`width x height x 4`). Alpha element is
  /// `(1 << 9) - 1`.
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

impl<R, const BE: bool> Yuv420p9Sink<BE> for MixedSinker<'_, Yuv420p9<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, Yuv420p9<BE>, R> {
  type Input<'r> = Yuv420p9Row<'r>;
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

  #[allow(clippy::too_many_lines)]
  fn process(&mut self, row: Yuv420p9Row<'_>) -> Result<(), Self::Error> {
    const BITS: u32 = 9;
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if w & 1 != 0 {
      return Err(MixedSinkerError::WidthAlignment(WidthAlignment::odd(w)));
    }
    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Y9,
        idx,
        w,
        row.y().len(),
      )));
    }
    if row.u_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::UHalf9,
        idx,
        w / 2,
        row.u_half().len(),
      )));
    }
    if row.v_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::VHalf9,
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

    // Chroma siting (#302): drives the identity-plan horizontal chroma phase.
    // `Copy`, so read it out before the field split-borrow below.
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
      chroma_full_u16,
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
      native_420_u16,
      frozen_native_route,
      frozen_chroma_centered,
      frozen_chroma_bottom_v,
      chroma_prev_u16,
      chroma_prev_row,
      ..
    } = self;

    // Non-identity plan: the native tier bins the host-native Y / U / V
    // planes at output resolution and converts ONCE per output row at
    // output width (4:4:4 kernels); the row-stage tier
    // ([`packed_yuv422_triple_resample`]) converts each source row at
    // source width then area-streams it (u8 color, independent native-u16
    // color, native Y). `with_native(false)` forces the latter. The half-
    // width U / V planes are horizontally upsampled in-register by the
    // shared 4:2:0 row kernels — 4:2:0's vertical chroma sharing is
    // already resolved by the walker, which hands this luma row its
    // (vertically-shared) `u_half` / `v_half`, so the per-row chroma
    // contract is identical to 4:2:2's and the same tail binds. Yuv420p
    // exposes no `luma_u16` output, so it is `&mut None` and only `luma`
    // (binned native Y `>> (BITS - 8)`) is emitted.
    if let Some(plan) = plan.as_ref() {
      let matrix = row.matrix();
      let full_range = row.full_range();
      let (y, u_half, v_half) = (row.y(), row.u_half(), row.v_half());
      // RFC #238 S6a — 4:2:0 HORIZONTAL chroma siting. The centered group
      // (`Center` / `Top` / `Bottom`, [`chroma_420_center_sited_h`]) samples
      // chroma at `+0.5` luma = `+0.25` chroma-sample horizontally; the co-sited
      // / unspecified group is phase 0 (today's byte-identical decode). Siting
      // enters the chroma RECONSTRUCTION only — the averaging tier is still
      // chosen by `select_insertion_point` below — so the native fast tier folds
      // the horizontal phase into the `area_chroma_420` chroma weights while the
      // row-stage and filter tiers reconstruct full-width `u16` chroma and decode
      // 4:4:4. VERTICAL stays co-sited (`v_phase = 0`): S6a routes the horizontal
      // Top / Center phase only; `Bottom`'s vertical blend is a later stage.
      let center_sited = chroma_420_center_sited_h(chroma_location);
      // RFC #238 S6d — 4:2:0 VERTICAL `Bottom` (`v = 1`) siting on top of the
      // S6a horizontal fold. `Bottom` ([`chroma_420_bottom_sited_v`]) is a strict
      // sub-case of `center_sited` (it is `h = 0.5, v = 1`), so it rides the
      // centered reconstruction below; the binning tiers additionally fold the
      // `v = 1` triangle into `area_chroma_420`'s vertical weights, and the
      // RGB-domain reconstruction tiers box-blend the even output row's chroma
      // with the previous chroma row via the `chroma_prev_u16` lookback. `Center`
      // / `Top` keep `v_phase = 0` (co-sited vertical, byte-identical to S6a).
      let bottom_v = chroma_420_bottom_sited_v(chroma_location);
      let chroma_h_phase = if center_sited {
        YUV422P_CENTERED_H_PHASE
      } else {
        0.0
      };
      let chroma_v_phase = if bottom_v { 1.0 } else { 0.0 };
      // Whether this call carries any output — the EXACT set both tiers'
      // preflight tests (`luma || rgb || rgba || hsv || rgb_u16 || rgba_u16`).
      // The route / siting freezes only on an output-bearing row a tier ACCEPTS;
      // a no-output call consumes no stream state, so it must not freeze.
      let need_output = luma.is_some()
        || rgb.is_some()
        || rgba.is_some()
        || hsv.is_some()
        || rgb_u16.is_some()
        || rgba_u16.is_some();
      // Only the colour tiers reconstruct full-width chroma for the centered
      // decode; a luma-only centered row bins native Y unchanged (siting is a
      // chroma-only property).
      let want_color =
        rgb.is_some() || rgba.is_some() || hsv.is_some() || rgb_u16.is_some() || rgba_u16.is_some();
      // Freeze the effective 4:2:0 chroma siting on the first output-bearing row
      // (mirrors the `frozen_native_route` freeze below). This CHECK is at the
      // always-compiled choke point every tier passes through; the matching SET
      // rides each tier's accept path (never before dispatch, so a rejected row
      // leaves it unset for a corrected retry). A later row observing a different
      // phase would bin a mixture of co-sited and centered chroma, so it is
      // rejected here before any reconstruction.
      if need_output
        && let Some(frozen) = *frozen_chroma_centered
        && (frozen != center_sited || *frozen_chroma_bottom_v != Some(bottom_v))
      {
        return Err(MixedSinkerError::ChromaSitingChanged(
          ChromaSitingChanged::new(idx),
        ));
      }
      // A `Filter` plan routes to the filter resampler BEFORE the
      // native/row-stage route machinery: the native fast tier is an
      // area-specific optimization that never sees a filter plan, and the
      // per-sink plan kind is fixed at construction, so a filter sink bypasses
      // the `frozen_native_route` interaction entirely. It converts the
      // separate Y/U/V planes to a source-width u8 + native-u16 RGB row (the
      // SAME closures the row-stage tier uses) and filter-resamples them plus
      // the native Y — the filter twin of the row-stage tier. The shared tail
      // clamps every sub-16-bit colour sample AND the native Y to
      // `(1 << BITS) - 1`. Yuv420p exposes no `luma_u16`, so it is `&mut None`.
      if plan.kind().is_filter() {
        // Reject a multi-kernel (BICUBLIN) plan BEFORE the centered reserve
        // below — the delegate's first act is this same check, so hoisting it
        // keeps a rejected filter plan from reserving / reconstructing chroma
        // first (the #180 reject-before-allocation invariant). Idempotent — the
        // delegate re-runs it.
        plan.ensure_single_kernel_filter()?;
        if (center_sited || bottom_v) && want_color {
          // Centered filter: reconstruct full-width `u16` chroma, but ONLY after
          // the resample preflight (frozen-output + sequence), so an
          // out-of-sequence / rejected row is caught before the chroma
          // reservation (#180). `packed_yuv422_triple_filter_resample` re-runs
          // the idempotent preflight and owns the transactional commit. The
          // HORIZONTAL centered reconstruction is all the row-stage / filter
          // tiers need — the walker already handed this luma row its
          // (vertically co-sited) chroma row.
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
          if bottom_v {
            reserve_420_chroma_prev_u16(chroma_prev_u16, w, h)?;
          }
          // `stage = false`: DEFER the lookback advance until AFTER the fallible
          // filter commit accepts the row (below), so a rejected row leaves the
          // predecessor in place for a clean retry (#180 state-atomicity).
          let (u_full, v_full) = upsample_420_chroma_sited_u16::<BITS>(
            chroma_full_u16,
            chroma_prev_u16,
            chroma_prev_row,
            u_half,
            v_half,
            idx,
            bottom_v,
            center_sited,
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
            |scratch| deinterleave_y_high_bit_masked::<BITS, BE>(y, scratch, w),
            |scratch| {
              yuv444p9_to_rgb_row_endian(
                y, u_full, v_full, scratch, w, matrix, full_range, use_simd, BE,
              )
            },
            |scratch| {
              yuv444p9_to_rgb_u16_row_endian(
                y, u_full, v_full, scratch, w, matrix, full_range, use_simd, BE,
              )
            },
          );
          // Bottom lookback: advance only AFTER the filter resample accepts the
          // row (`r.is_ok()`), so a rejected row leaves the predecessor for a
          // clean retry — the sited reconstruction read it above but did not
          // stage. Inside the centered && want_color arm, so gate on `bottom_v`.
          if r.is_ok() && bottom_v {
            stage_420_chroma_prev_u16(chroma_prev_u16, chroma_prev_row, u_half, v_half, idx, w);
          }
          if r.is_ok() && need_output && frozen_chroma_centered.is_none() {
            *frozen_chroma_centered = Some(center_sited);
            *frozen_chroma_bottom_v = Some(bottom_v);
          }
          return r;
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
          |scratch| deinterleave_y_high_bit_masked::<BITS, BE>(y, scratch, w),
          |scratch| {
            yuv420p9_to_rgb_row_endian(
              y, u_half, v_half, scratch, w, matrix, full_range, use_simd, BE,
            )
          },
          |scratch| {
            yuv420p9_to_rgb_u16_row_endian(
              y, u_half, v_half, scratch, w, matrix, full_range, use_simd, BE,
            )
          },
        );
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
      // a true no-op, route-invisible regardless of row index. A
      // preflight-rejected (out-of-sequence / frozen) output-bearing call
      // returns Err before the SET, so it leaves `frozen_native_route`
      // untouched and a later same-or-other-route retry is not falsely
      // rejected.
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
          native_eligible: YUV420P_HIGH_BIT_NATIVE_ELIGIBLE,
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
          // RFC #238 S6a point-of-use siting invalidation: a reused sink's
          // cached join is only `reset` between frames, so a frame whose
          // `chroma_location` moved to a different horizontal phase must REBUILD
          // it (`area_chroma_420` folds the phase into the chroma weights). Drop
          // the stale-phase join ONLY on the in-sequence first row of a fresh
          // frame (`idx == 0`, `next_y() == 0`) so a mid-frame / out-of-sequence
          // row rejects against the INTACT join and a corrected retry rebuilds
          // cleanly; a luma-only join carries no chroma phase and is never
          // dropped. Move it OUT (the delegate builds the replacement into the
          // field, keeping it untouched until every pre-feed allocation
          // succeeds) and restore the intact prior-phase join on a rejected
          // rebuild so the row mutates no join state.
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
          let native_result = yuv420p16_process_native::<BITS, BE>(
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
            y,
            u_half,
            v_half,
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
                false,
              )
            },
            use_simd,
          );
          // Restore the taken stale-phase join if the delegate's rebuild was
          // rejected at any pre-feed step: it leaves the field `None` on such a
          // failure, so restoring the intact prior-phase join leaves the
          // rejected row mutating no join state. A non-stale row took nothing.
          if stale_native && native_result.is_err() {
            *native_420_u16 = prev_native;
          }
          native_result?;
          if frozen_native_route.is_none() && need_output {
            *frozen_native_route = Some(true);
          }
          // RFC #238 S6a: freeze the siting on the same accepted output row.
          if frozen_chroma_centered.is_none() && need_output {
            *frozen_chroma_centered = Some(center_sited);
            *frozen_chroma_bottom_v = Some(bottom_v);
          }
          return Ok(());
        }
        InsertionPoint::EncodedOutput => {
          // Row-stage tail. Same CHECK-before / SET-after split: dispatch, then
          // freeze the route to row-stage only when the call accepts an
          // output-bearing row (a no-output call returns Ok with `need_output`
          // false; an out-of-sequence / frozen row returns Err via `?`).
          if (center_sited || bottom_v) && want_color {
            // Centered row-stage: reconstruct full-width `u16` chroma AFTER the
            // resample preflight (frozen-output + sequence), so an
            // out-of-sequence / rejected row is caught before the chroma
            // reservation (#180). `packed_yuv422_triple_resample` re-runs the
            // idempotent preflight and owns the transactional commit. HORIZONTAL
            // reconstruction only — the walker handed this luma row its
            // (vertically co-sited) chroma row.
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
            if bottom_v {
              reserve_420_chroma_prev_u16(chroma_prev_u16, w, h)?;
            }
            // `stage = false`: DEFER the lookback advance until AFTER the fallible
            // row-stage commit accepts the row (below), so a rejected row leaves
            // the predecessor for a clean retry (#180 state-atomicity).
            let (u_full, v_full) = upsample_420_chroma_sited_u16::<BITS>(
              chroma_full_u16,
              chroma_prev_u16,
              chroma_prev_row,
              u_half,
              v_half,
              idx,
              bottom_v,
              center_sited,
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
              |scratch| deinterleave_y_high_bit_masked::<BITS, BE>(y, scratch, w),
              |scratch| {
                yuv444p9_to_rgb_row_endian(
                  y, u_full, v_full, scratch, w, matrix, full_range, use_simd, BE,
                )
              },
              |scratch| {
                yuv444p9_to_rgb_u16_row_endian(
                  y, u_full, v_full, scratch, w, matrix, full_range, use_simd, BE,
                )
              },
            )?;
            // Bottom lookback: advance only AFTER the row-stage resample accepts
            // the row (the `?` above already returned any reject), so a rejected
            // row leaves the predecessor for a clean retry — the sited
            // reconstruction read it above but did not stage.
            if bottom_v {
              stage_420_chroma_prev_u16(chroma_prev_u16, chroma_prev_row, u_half, v_half, idx, w);
            }
          } else {
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
              |scratch| deinterleave_y_high_bit_masked::<BITS, BE>(y, scratch, w),
              |scratch| {
                yuv420p9_to_rgb_row_endian(
                  y, u_half, v_half, scratch, w, matrix, full_range, use_simd, BE,
                )
              },
              |scratch| {
                yuv420p9_to_rgb_u16_row_endian(
                  y, u_half, v_half, scratch, w, matrix, full_range, use_simd, BE,
                )
              },
            )?;
          }
          if frozen_native_route.is_none() && need_output {
            *frozen_native_route = Some(false);
          }
          // RFC #238 S6a: freeze the siting on the same accepted output row.
          if frozen_chroma_centered.is_none() && need_output {
            *frozen_chroma_centered = Some(center_sited);
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

    // Resolve the full output set up front so the no-output guard below
    // short-circuits before ANY per-row offset arithmetic and the atomicity
    // preflight runs before any output row (luma included) is written.
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();
    let want_rgb_u16 = rgb_u16.is_some();
    let want_rgba_u16 = rgba_u16.is_some();
    // Whether this row produces any colour output (and so runs the centered /
    // bottom-sited chroma upsample). A bottom-sited row maintains the vertical
    // lookback even when it produces only luma — see the luma-only staging below
    // — so `want_color` gates only the colour scratches.
    let want_color = want_rgb || want_rgba || want_hsv || want_rgb_u16 || want_rgba_u16;
    // Repo-wide no-output invariant (RFC #238 S6d): a `process` call carrying NO
    // output — no colour, no luma — must run NOTHING: no per-row offset
    // arithmetic, no allocation, no state mutation (the bottom-sited vertical
    // lookback included). Returning HERE, before the `idx * w` offsets below,
    // keeps the invariant overflow-safe (a no-output call never ran an
    // attach-time `w x h` validation, so `idx * w` could overflow on a 32-bit
    // target) AND means such a row never reserves `chroma_prev_u16` nor primes
    // the lookback (so it can never make a later colour even row box-blend
    // through an invisible, never-output row). Yuv420p exposes no `luma_u16`.
    let need_output = want_color || luma.is_some();
    if !need_output {
      return Ok(());
    }

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // Chroma siting (#302): the centered horizontal sitings reconstruct chroma
    // at the phase-0.5 position; the default / co-sited path keeps the
    // byte-identical decode (the fused high-bit 4:2:0 kernels upsample chroma
    // in-register, exactly as before).
    let center_sited = chroma_420_center_sited_h(chroma_location);
    // RFC #238 S6d: `Bottom` (a strict sub-case of `center_sited`) additionally
    // box-blends the even output row's chroma with the previous chroma row via
    // the `chroma_prev_u16` lookback maintained below; `Center` / `Top` keep the
    // vertical-replicate (co-sited) decode, byte-identical to S6a.
    let bottom_v = chroma_420_bottom_sited_v(chroma_location);

    // Per-frame chroma-siting freeze (RFC #238, mirroring the resample-path guard
    // above): the first output-bearing row pins the effective 4:2:0 phase — BOTH
    // the horizontal centered flag and the vertical `Bottom` flag. A later row
    // whose siting flipped would decode a mixture of phases into ONE frame, or box-
    // blend against a STALE `chroma_prev_u16` lookback, so reject it here BEFORE
    // any scratch reserve, lookback priming, or output write. This CHECK precedes
    // the `stage_420_chroma_prev_u16` / reconstruct staging below so a rejected
    // flip leaves `chroma_prev_u16` / `chroma_prev_row` untouched (retry-atomic,
    // #180). `begin_frame`'s `reset_high_bit_yuv_streams` clears the freeze so the
    // next frame may pick either phase.
    if need_output
      && let Some(frozen) = *frozen_chroma_centered
      && (frozen != center_sited || *frozen_chroma_bottom_v != Some(bottom_v))
    {
      return Err(MixedSinkerError::ChromaSitingChanged(
        ChromaSitingChanged::new(idx),
      ));
    }

    // Atomicity preflight (#302 / #308 / #314, cf. the crate's #180 resample
    // fix): reserve EVERY fallible row scratch this identity row can touch
    // BEFORE any output row is written (the luma plane below, then the u16 / u8
    // RGB / RGBA / HSV fan-out), so an allocator refusal returns a typed
    // `AllocationFailed` leaving the output frame untouched rather than
    // partially mutated. Two scratches can grow:
    //  1. the centered-siting full-width `u16` chroma (`chroma_full_u16`),
    //     needed by ANY colour output (u8 OR u16 RGB / RGBA / HSV); and
    //  2. the u8 RGB row buffer, reached exactly when a colour decode needs an
    //     RGB row but no caller RGB buffer is borrowable — `want_hsv &&
    //     want_rgba && !want_rgb` (`rgb_row_buf_or_scratch`'s own scratch arm).
    // The later `upsample_420_chroma_center_h_u16` / `rgb_row_buf_or_scratch`
    // calls then reuse the already-sized buffers, so the default path is
    // byte-identical; only the failure-path ordering changes. The u16 RGB /
    // RGBA outputs write straight into their caller buffers (the rgb_u16 plane
    // itself stages the rgba_u16 expand) and never grow a scratch of their own.
    // Any colour output (u8 or u16 RGB / RGBA / HSV) consumes the centered
    // chroma; a luma-only row never does, so it neither reserves nor upsamples
    // it (and the reserve below is what makes the later upsample infallible).
    let need_centered_chroma = (center_sited || bottom_v) && want_color;
    if need_centered_chroma {
      reserve_420_chroma_full_u16(chroma_full_u16, w, h)?;
    }
    // Bottom-sited vertical-phase one-row chroma lookback (RFC #238 S6d): reserve
    // it on EVERY bottom-sited OUTPUT row — colour OR luma-only — because the
    // lookback is maintained so a LATER colour row can box-blend it (the
    // luma-only staging runs below, before any luma write). A no-output row
    // returned early above, so it never reaches here and never primes the
    // lookback. Reserved BEFORE any output write (the #180 preflight ordering).
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

    // Bottom-sited LUMA-ONLY row (RFC #238 S6d): the colour upsample below —
    // which normally refreshes the vertical lookback — won't run, so stage the
    // current chroma row HERE, after its reservation above and BEFORE any luma
    // write, so a later colour row in the same frame can box-blend it (a
    // luma-only-then-colour in-order sequence reconstructs the same Bottom
    // vertical phase as the all-output walk). A colour row instead stages inside
    // `upsample_420_chroma_sited_u16` after reading the previous lookback, so
    // this is skipped for it; a no-output row returned early above, so
    // `!want_color` here is a genuine luma-only row. The validity tag
    // (`chroma_prev_row`) still guards out-of-sequence / cross-frame reads.
    if bottom_v && !want_color {
      stage_420_chroma_prev_u16(
        chroma_prev_u16,
        chroma_prev_row,
        row.u_half(),
        row.v_half(),
        idx,
        w,
      );
    }

    // Centered full-width chroma, reconstructed ONCE per row from the wire-format
    // half-width U / V and reused by every colour decode (u16 and u8). Infallible
    // — the scratches were reserved above. The default left/unspecified siting
    // leaves it `None`, so the fused 4:2:0 kernels upsample chroma in-register
    // instead and the output stays byte-identical. `Center` / `Top` take the
    // plain horizontal centered phase-0.5 fold; `Bottom` (RFC #238 S6d)
    // additionally box-blends the even output row with the previous chroma row
    // (`stage = true` — the direct decode's post-reconstruction work is
    // infallible, so advancing the lookback here is safe).
    let centered = if need_centered_chroma {
      Some(upsample_420_chroma_sited_u16::<BITS>(
        chroma_full_u16,
        chroma_prev_u16,
        chroma_prev_row,
        row.u_half(),
        row.v_half(),
        idx,
        bottom_v,
        center_sited,
        true,
        w,
        BE,
      ))
    } else {
      None
    };

    // Freeze the effective 4:2:0 phase on the first output-bearing row — AFTER the
    // fallible scratch reserves above have succeeded, so an `AllocationFailed` row
    // stays retryable (frozen stays unset); later rows are checked against it up
    // top. Both the horizontal centered flag and the vertical `Bottom` flag are
    // pinned together.
    if need_output && frozen_chroma_centered.is_none() {
      *frozen_chroma_centered = Some(center_sited);
      *frozen_chroma_bottom_v = Some(bottom_v);
    }

    let matrix = row.matrix();
    let full_range = row.full_range();

    if let Some(luma) = luma.as_deref_mut() {
      let dst = &mut luma[one_plane_start..one_plane_end];
      for (d, &s) in dst.iter_mut().zip(row.y().iter()) {
        // Normalize BE-encoded wire bytes to host-native before the
        // luma downshift — without this, a valid BE mid-gray sample
        // (`1 << (BITS - 1)`, e.g. `0x0100` for 9-bit, `0x0200` for
        // 10-bit, `0x0800` for 12-bit) would be byte-swapped on a LE
        // host and the `>> (BITS - 8)` would write 0 instead of 128.
        let logical = if BE { u16::from_be(s) } else { u16::from_le(s) };
        *d = (logical >> (BITS - 8)) as u8;
      }
    }

    // ===== u16 RGB / RGBA path (Strategy A) =====
    // Compute u16 RGB once (to caller's buffer when attached) and fan
    // out to u16 RGBA via the cheap per-pixel pad. RGBA-only avoids the
    // RGB kernel entirely and writes RGBA directly.
    if want_rgba_u16 && !want_rgb_u16 {
      let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row =
        rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
      if let Some((u_full, v_full)) = centered {
        yuv444p9_to_rgba_u16_row_endian(
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
        yuv420p9_to_rgba_u16_row_endian(
          row.y(),
          row.u_half(),
          row.v_half(),
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
        yuv444p9_to_rgb_u16_row_endian(
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
        yuv420p9_to_rgb_u16_row_endian(
          row.y(),
          row.u_half(),
          row.v_half(),
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
    // HSV-without-RGB-or-RGBA goes through the direct `*_to_hsv_row_endian`
    // kernel (no source-width RGB scratch — the SIMD path stages a fixed
    // 8-bit RGB chunk internally). When RGB or RGBA is *also* attached the
    // RGB kernel runs anyway, so HSV derives off that 8-bit buffer for free
    // — the cheap path — and `need_rgb_kernel` keeps it alive. Centered siting
    // (#302) routes each colour kernel through its 4:4:4 twin, fed the
    // full-width phase-0.5 chroma reconstructed above.
    let want_hsv_direct = want_hsv && !want_rgb && !want_rgba;
    let need_rgb_kernel = want_rgb || (want_hsv && want_rgba);

    if want_hsv_direct {
      let hsv = hsv.as_mut().expect("want_hsv_direct implies hsv attached");
      let (h, s, v) = hsv.hsv();
      if let Some((u_full, v_full)) = centered {
        yuv444p9_to_hsv_row_endian(
          row.y(),
          u_full,
          v_full,
          &mut h[one_plane_start..one_plane_end],
          &mut s[one_plane_start..one_plane_end],
          &mut v[one_plane_start..one_plane_end],
          w,
          matrix,
          full_range,
          use_simd,
          BE,
        );
      } else {
        yuv420p9_to_hsv_row_endian(
          row.y(),
          row.u_half(),
          row.v_half(),
          &mut h[one_plane_start..one_plane_end],
          &mut s[one_plane_start..one_plane_end],
          &mut v[one_plane_start..one_plane_end],
          w,
          matrix,
          full_range,
          use_simd,
          BE,
        );
      }
      return Ok(());
    }

    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      if let Some((u_full, v_full)) = centered {
        yuv444p9_to_rgba_row_endian(
          row.y(),
          u_full,
          v_full,
          rgba_row,
          w,
          matrix,
          full_range,
          use_simd,
          BE,
        );
      } else {
        yuv420p9_to_rgba_row_endian(
          row.y(),
          row.u_half(),
          row.v_half(),
          rgba_row,
          w,
          matrix,
          full_range,
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

    if let Some((u_full, v_full)) = centered {
      yuv444p9_to_rgb_row_endian(
        row.y(),
        u_full,
        v_full,
        rgb_row,
        w,
        matrix,
        full_range,
        use_simd,
        BE,
      );
    } else {
      yuv420p9_to_rgb_row_endian(
        row.y(),
        row.u_half(),
        row.v_half(),
        rgb_row,
        w,
        matrix,
        full_range,
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

// ---- Yuv420p10 impl -----------------------------------------------------

impl<'a, R, const BE: bool> MixedSinker<'a, Yuv420p10<BE>, R> {
  /// Attaches a packed **`u16`** RGB output buffer. Only available on
  /// sinkers whose source format populates native‑depth `u16` RGB —
  /// calling `with_rgb_u16` on an 8‑bit source sinker (e.g.
  /// [`MixedSinker<Yuv420p>`]) is a compile error rather than a
  /// silent no‑op that would leave the caller's buffer stale.
  ///
  /// Length is measured in `u16` **elements** (not bytes): minimum
  /// `width x height x 3`. Each element carries a 10‑bit value in
  /// the **low** 10 bits (upper 6 bits zero), matching FFmpeg's
  /// `yuv420p10le` convention. This is **not** the `p010` layout
  /// (which stores samples in the high 10 bits); callers feeding a
  /// p010 consumer must shift the output left by 6.
  ///
  /// Returns `Err(InsufficientRgbU16Buffer)` if
  /// `buf.len() < width x height x 3`, or `Err(GeometryOverflow)`
  /// on 32‑bit targets when the product overflows.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgb_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_rgb_u16(buf)?;
    Ok(self)
  }

  /// In-place variant of [`with_rgb_u16`](Self::with_rgb_u16). The
  /// required length is measured in `u16` **elements**, not bytes.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgb_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
    // Packed RGB requires `width x height x 3` channel values —
    // that's the same count whether the element type is `u8` or
    // `u16`, so the [`Self::frame_elems`] helper (named for the u8
    // RGB path's byte count) gives the element count here too. No
    // size conversion needed.
    let expected_elements = self.frame_elems(3)?;
    if buf.len() < expected_elements {
      return Err(MixedSinkerError::InsufficientRgbU16Buffer(
        InsufficientBuffer::new(expected_elements, buf.len()),
      ));
    }
    self.rgb_u16 = Some(buf);
    Ok(self)
  }

  /// Attaches a packed **8‑bit** RGBA output buffer. The 10‑bit YUV
  /// source is converted to 8‑bit RGBA via the `BITS = 10` Q15 kernel
  /// family; the fourth byte per pixel is alpha = `0xFF` (Yuv420p10
  /// has no alpha plane).
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

  /// Attaches a packed **`u16`** RGBA output buffer. 10‑bit
  /// low‑packed (`(1 << 10) - 1 = 1023` max). Length is measured in
  /// `u16` **elements** (`width x height x 4`). Alpha element is
  /// `(1 << 10) - 1`.
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

impl<R, const BE: bool> Yuv420p10Sink<BE> for MixedSinker<'_, Yuv420p10<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, Yuv420p10<BE>, R> {
  type Input<'r> = Yuv420p10Row<'r>;
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

  #[allow(clippy::too_many_lines)]
  fn process(&mut self, row: Yuv420p10Row<'_>) -> Result<(), Self::Error> {
    // Bit depth is fixed by the format (10) — declared as a const so
    // the downshift for u8 luma stays obvious at the call site.
    const BITS: u32 = 10;

    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    // Defense in depth — see the [`Yuv420p`] impl for the rationale.
    // Row slice checks use the 10‑bit variants of [`RowSlice`] so
    // downstream log output disambiguates from the 8‑bit source impls.
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
    if row.u_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::UHalf10,
        idx,
        w / 2,
        row.u_half().len(),
      )));
    }
    if row.v_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::VHalf10,
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

    // Chroma siting (#302): drives the identity-plan horizontal chroma phase.
    // `Copy`, so read it out before the field split-borrow below.
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
      chroma_full_u16,
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
      native_420_u16,
      frozen_native_route,
      frozen_chroma_centered,
      frozen_chroma_bottom_v,
      chroma_prev_u16,
      chroma_prev_row,
      ..
    } = self;

    // Non-identity plan: the native tier bins the host-native Y / U / V
    // planes at output resolution and converts ONCE per output row at
    // output width (4:4:4 kernels); the row-stage tier
    // ([`packed_yuv422_triple_resample`]) converts each source row at
    // source width then area-streams it (u8 color, independent native-u16
    // color, native Y). `with_native(false)` forces the latter. The half-
    // width U / V planes are horizontally upsampled in-register by the
    // shared 4:2:0 row kernels — 4:2:0's vertical chroma sharing is
    // already resolved by the walker, which hands this luma row its
    // (vertically-shared) `u_half` / `v_half`, so the per-row chroma
    // contract is identical to 4:2:2's and the same tail binds. Yuv420p
    // exposes no `luma_u16` output, so it is `&mut None` and only `luma`
    // (binned native Y `>> (BITS - 8)`) is emitted.
    if let Some(plan) = plan.as_ref() {
      let matrix = row.matrix();
      let full_range = row.full_range();
      let (y, u_half, v_half) = (row.y(), row.u_half(), row.v_half());
      // RFC #238 S6a — 4:2:0 HORIZONTAL chroma siting. The centered group
      // (`Center` / `Top` / `Bottom`, [`chroma_420_center_sited_h`]) samples
      // chroma at `+0.5` luma = `+0.25` chroma-sample horizontally; the co-sited
      // / unspecified group is phase 0 (today's byte-identical decode). Siting
      // enters the chroma RECONSTRUCTION only — the averaging tier is still
      // chosen by `select_insertion_point` below — so the native fast tier folds
      // the horizontal phase into the `area_chroma_420` chroma weights while the
      // row-stage and filter tiers reconstruct full-width `u16` chroma and decode
      // 4:4:4. VERTICAL stays co-sited (`v_phase = 0`): S6a routes the horizontal
      // Top / Center phase only; `Bottom`'s vertical blend is a later stage.
      let center_sited = chroma_420_center_sited_h(chroma_location);
      // RFC #238 S6d — 4:2:0 VERTICAL `Bottom` (`v = 1`) siting on top of the
      // S6a horizontal fold. `Bottom` ([`chroma_420_bottom_sited_v`]) is a strict
      // sub-case of `center_sited` (it is `h = 0.5, v = 1`), so it rides the
      // centered reconstruction below; the binning tiers additionally fold the
      // `v = 1` triangle into `area_chroma_420`'s vertical weights, and the
      // RGB-domain reconstruction tiers box-blend the even output row's chroma
      // with the previous chroma row via the `chroma_prev_u16` lookback. `Center`
      // / `Top` keep `v_phase = 0` (co-sited vertical, byte-identical to S6a).
      let bottom_v = chroma_420_bottom_sited_v(chroma_location);
      let chroma_h_phase = if center_sited {
        YUV422P_CENTERED_H_PHASE
      } else {
        0.0
      };
      let chroma_v_phase = if bottom_v { 1.0 } else { 0.0 };
      // Whether this call carries any output — the EXACT set both tiers'
      // preflight tests (`luma || rgb || rgba || hsv || rgb_u16 || rgba_u16`).
      // The route / siting freezes only on an output-bearing row a tier ACCEPTS;
      // a no-output call consumes no stream state, so it must not freeze.
      let need_output = luma.is_some()
        || rgb.is_some()
        || rgba.is_some()
        || hsv.is_some()
        || rgb_u16.is_some()
        || rgba_u16.is_some();
      // Only the colour tiers reconstruct full-width chroma for the centered
      // decode; a luma-only centered row bins native Y unchanged (siting is a
      // chroma-only property).
      let want_color =
        rgb.is_some() || rgba.is_some() || hsv.is_some() || rgb_u16.is_some() || rgba_u16.is_some();
      // Freeze the effective 4:2:0 chroma siting on the first output-bearing row
      // (mirrors the `frozen_native_route` freeze below). This CHECK is at the
      // always-compiled choke point every tier passes through; the matching SET
      // rides each tier's accept path (never before dispatch, so a rejected row
      // leaves it unset for a corrected retry). A later row observing a different
      // phase would bin a mixture of co-sited and centered chroma, so it is
      // rejected here before any reconstruction.
      if need_output
        && let Some(frozen) = *frozen_chroma_centered
        && (frozen != center_sited || *frozen_chroma_bottom_v != Some(bottom_v))
      {
        return Err(MixedSinkerError::ChromaSitingChanged(
          ChromaSitingChanged::new(idx),
        ));
      }
      // A `Filter` plan routes to the filter resampler BEFORE the
      // native/row-stage route machinery: the native fast tier is an
      // area-specific optimization that never sees a filter plan, and the
      // per-sink plan kind is fixed at construction, so a filter sink bypasses
      // the `frozen_native_route` interaction entirely. It converts the
      // separate Y/U/V planes to a source-width u8 + native-u16 RGB row (the
      // SAME closures the row-stage tier uses) and filter-resamples them plus
      // the native Y — the filter twin of the row-stage tier. The shared tail
      // clamps every sub-16-bit colour sample AND the native Y to
      // `(1 << BITS) - 1`. Yuv420p exposes no `luma_u16`, so it is `&mut None`.
      if plan.kind().is_filter() {
        // Reject a multi-kernel (BICUBLIN) plan BEFORE the centered reserve
        // below — the delegate's first act is this same check, so hoisting it
        // keeps a rejected filter plan from reserving / reconstructing chroma
        // first (the #180 reject-before-allocation invariant). Idempotent — the
        // delegate re-runs it.
        plan.ensure_single_kernel_filter()?;
        if (center_sited || bottom_v) && want_color {
          // Centered filter: reconstruct full-width `u16` chroma, but ONLY after
          // the resample preflight (frozen-output + sequence), so an
          // out-of-sequence / rejected row is caught before the chroma
          // reservation (#180). `packed_yuv422_triple_filter_resample` re-runs
          // the idempotent preflight and owns the transactional commit. The
          // HORIZONTAL centered reconstruction is all the row-stage / filter
          // tiers need — the walker already handed this luma row its
          // (vertically co-sited) chroma row.
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
          if bottom_v {
            reserve_420_chroma_prev_u16(chroma_prev_u16, w, h)?;
          }
          // `stage = false`: DEFER the lookback advance until AFTER the fallible
          // filter commit accepts the row (below), so a rejected row leaves the
          // predecessor in place for a clean retry (#180 state-atomicity).
          let (u_full, v_full) = upsample_420_chroma_sited_u16::<BITS>(
            chroma_full_u16,
            chroma_prev_u16,
            chroma_prev_row,
            u_half,
            v_half,
            idx,
            bottom_v,
            center_sited,
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
            |scratch| deinterleave_y_high_bit_masked::<BITS, BE>(y, scratch, w),
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
          );
          // Bottom lookback: advance only AFTER the filter resample accepts the
          // row (`r.is_ok()`), so a rejected row leaves the predecessor for a
          // clean retry — the sited reconstruction read it above but did not
          // stage. Inside the centered && want_color arm, so gate on `bottom_v`.
          if r.is_ok() && bottom_v {
            stage_420_chroma_prev_u16(chroma_prev_u16, chroma_prev_row, u_half, v_half, idx, w);
          }
          if r.is_ok() && need_output && frozen_chroma_centered.is_none() {
            *frozen_chroma_centered = Some(center_sited);
            *frozen_chroma_bottom_v = Some(bottom_v);
          }
          return r;
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
          |scratch| deinterleave_y_high_bit_masked::<BITS, BE>(y, scratch, w),
          |scratch| {
            yuv420p10_to_rgb_row_endian(
              y, u_half, v_half, scratch, w, matrix, full_range, use_simd, BE,
            )
          },
          |scratch| {
            yuv420p10_to_rgb_u16_row_endian(
              y, u_half, v_half, scratch, w, matrix, full_range, use_simd, BE,
            )
          },
        );
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
      // a true no-op, route-invisible regardless of row index. A
      // preflight-rejected (out-of-sequence / frozen) output-bearing call
      // returns Err before the SET, so it leaves `frozen_native_route`
      // untouched and a later same-or-other-route retry is not falsely
      // rejected.
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
          native_eligible: YUV420P_HIGH_BIT_NATIVE_ELIGIBLE,
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
          // RFC #238 S6a point-of-use siting invalidation: a reused sink's
          // cached join is only `reset` between frames, so a frame whose
          // `chroma_location` moved to a different horizontal phase must REBUILD
          // it (`area_chroma_420` folds the phase into the chroma weights). Drop
          // the stale-phase join ONLY on the in-sequence first row of a fresh
          // frame (`idx == 0`, `next_y() == 0`) so a mid-frame / out-of-sequence
          // row rejects against the INTACT join and a corrected retry rebuilds
          // cleanly; a luma-only join carries no chroma phase and is never
          // dropped. Move it OUT (the delegate builds the replacement into the
          // field, keeping it untouched until every pre-feed allocation
          // succeeds) and restore the intact prior-phase join on a rejected
          // rebuild so the row mutates no join state.
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
          let native_result = yuv420p16_process_native::<BITS, BE>(
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
            y,
            u_half,
            v_half,
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
                false,
              )
            },
            use_simd,
          );
          // Restore the taken stale-phase join if the delegate's rebuild was
          // rejected at any pre-feed step: it leaves the field `None` on such a
          // failure, so restoring the intact prior-phase join leaves the
          // rejected row mutating no join state. A non-stale row took nothing.
          if stale_native && native_result.is_err() {
            *native_420_u16 = prev_native;
          }
          native_result?;
          if frozen_native_route.is_none() && need_output {
            *frozen_native_route = Some(true);
          }
          // RFC #238 S6a: freeze the siting on the same accepted output row.
          if frozen_chroma_centered.is_none() && need_output {
            *frozen_chroma_centered = Some(center_sited);
            *frozen_chroma_bottom_v = Some(bottom_v);
          }
          return Ok(());
        }
        InsertionPoint::EncodedOutput => {
          // Row-stage tail. Same CHECK-before / SET-after split: dispatch, then
          // freeze the route to row-stage only when the call accepts an
          // output-bearing row (a no-output call returns Ok with `need_output`
          // false; an out-of-sequence / frozen row returns Err via `?`).
          if (center_sited || bottom_v) && want_color {
            // Centered row-stage: reconstruct full-width `u16` chroma AFTER the
            // resample preflight (frozen-output + sequence), so an
            // out-of-sequence / rejected row is caught before the chroma
            // reservation (#180). `packed_yuv422_triple_resample` re-runs the
            // idempotent preflight and owns the transactional commit. HORIZONTAL
            // reconstruction only — the walker handed this luma row its
            // (vertically co-sited) chroma row.
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
            if bottom_v {
              reserve_420_chroma_prev_u16(chroma_prev_u16, w, h)?;
            }
            // `stage = false`: DEFER the lookback advance until AFTER the fallible
            // row-stage commit accepts the row (below), so a rejected row leaves
            // the predecessor for a clean retry (#180 state-atomicity).
            let (u_full, v_full) = upsample_420_chroma_sited_u16::<BITS>(
              chroma_full_u16,
              chroma_prev_u16,
              chroma_prev_row,
              u_half,
              v_half,
              idx,
              bottom_v,
              center_sited,
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
              |scratch| deinterleave_y_high_bit_masked::<BITS, BE>(y, scratch, w),
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
            )?;
            // Bottom lookback: advance only AFTER the row-stage resample accepts
            // the row (the `?` above already returned any reject), so a rejected
            // row leaves the predecessor for a clean retry — the sited
            // reconstruction read it above but did not stage.
            if bottom_v {
              stage_420_chroma_prev_u16(chroma_prev_u16, chroma_prev_row, u_half, v_half, idx, w);
            }
          } else {
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
              |scratch| deinterleave_y_high_bit_masked::<BITS, BE>(y, scratch, w),
              |scratch| {
                yuv420p10_to_rgb_row_endian(
                  y, u_half, v_half, scratch, w, matrix, full_range, use_simd, BE,
                )
              },
              |scratch| {
                yuv420p10_to_rgb_u16_row_endian(
                  y, u_half, v_half, scratch, w, matrix, full_range, use_simd, BE,
                )
              },
            )?;
          }
          if frozen_native_route.is_none() && need_output {
            *frozen_native_route = Some(false);
          }
          // RFC #238 S6a: freeze the siting on the same accepted output row.
          if frozen_chroma_centered.is_none() && need_output {
            *frozen_chroma_centered = Some(center_sited);
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

    // Resolve the full output set up front so the no-output guard below
    // short-circuits before ANY per-row offset arithmetic and the atomicity
    // preflight runs before any output row (luma included) is written.
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();
    let want_rgb_u16 = rgb_u16.is_some();
    let want_rgba_u16 = rgba_u16.is_some();
    // Whether this row produces any colour output (and so runs the centered /
    // bottom-sited chroma upsample). A bottom-sited row maintains the vertical
    // lookback even when it produces only luma — see the luma-only staging below
    // — so `want_color` gates only the colour scratches.
    let want_color = want_rgb || want_rgba || want_hsv || want_rgb_u16 || want_rgba_u16;
    // Repo-wide no-output invariant (RFC #238 S6d): a `process` call carrying NO
    // output — no colour, no luma — must run NOTHING: no per-row offset
    // arithmetic, no allocation, no state mutation (the bottom-sited vertical
    // lookback included). Returning HERE, before the `idx * w` offsets below,
    // keeps the invariant overflow-safe (a no-output call never ran an
    // attach-time `w x h` validation, so `idx * w` could overflow on a 32-bit
    // target) AND means such a row never reserves `chroma_prev_u16` nor primes
    // the lookback (so it can never make a later colour even row box-blend
    // through an invisible, never-output row). Yuv420p exposes no `luma_u16`.
    let need_output = want_color || luma.is_some();
    if !need_output {
      return Ok(());
    }

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // Chroma siting (#302): the centered horizontal sitings reconstruct chroma
    // at the phase-0.5 position; the default / co-sited path keeps the
    // byte-identical decode (the fused high-bit 4:2:0 kernels upsample chroma
    // in-register, exactly as before).
    let center_sited = chroma_420_center_sited_h(chroma_location);
    // RFC #238 S6d: `Bottom` (a strict sub-case of `center_sited`) additionally
    // box-blends the even output row's chroma with the previous chroma row via
    // the `chroma_prev_u16` lookback maintained below; `Center` / `Top` keep the
    // vertical-replicate (co-sited) decode, byte-identical to S6a.
    let bottom_v = chroma_420_bottom_sited_v(chroma_location);

    // Per-frame chroma-siting freeze (RFC #238, mirroring the resample-path guard
    // above): the first output-bearing row pins the effective 4:2:0 phase — BOTH
    // the horizontal centered flag and the vertical `Bottom` flag. A later row
    // whose siting flipped would decode a mixture of phases into ONE frame, or box-
    // blend against a STALE `chroma_prev_u16` lookback, so reject it here BEFORE
    // any scratch reserve, lookback priming, or output write. This CHECK precedes
    // the `stage_420_chroma_prev_u16` / reconstruct staging below so a rejected
    // flip leaves `chroma_prev_u16` / `chroma_prev_row` untouched (retry-atomic,
    // #180). `begin_frame`'s `reset_high_bit_yuv_streams` clears the freeze so the
    // next frame may pick either phase.
    if need_output
      && let Some(frozen) = *frozen_chroma_centered
      && (frozen != center_sited || *frozen_chroma_bottom_v != Some(bottom_v))
    {
      return Err(MixedSinkerError::ChromaSitingChanged(
        ChromaSitingChanged::new(idx),
      ));
    }

    // Atomicity preflight (#302 / #308 / #314, cf. the crate's #180 resample
    // fix): reserve EVERY fallible row scratch this identity row can touch
    // BEFORE any output row is written (the luma plane below, then the u16 / u8
    // RGB / RGBA / HSV fan-out), so an allocator refusal returns a typed
    // `AllocationFailed` leaving the output frame untouched rather than
    // partially mutated. Two scratches can grow:
    //  1. the centered-siting full-width `u16` chroma (`chroma_full_u16`),
    //     needed by ANY colour output (u8 OR u16 RGB / RGBA / HSV); and
    //  2. the u8 RGB row buffer, reached exactly when a colour decode needs an
    //     RGB row but no caller RGB buffer is borrowable — `want_hsv &&
    //     want_rgba && !want_rgb` (`rgb_row_buf_or_scratch`'s own scratch arm).
    // The later `upsample_420_chroma_center_h_u16` / `rgb_row_buf_or_scratch`
    // calls then reuse the already-sized buffers, so the default path is
    // byte-identical; only the failure-path ordering changes. The u16 RGB /
    // RGBA outputs write straight into their caller buffers (the rgb_u16 plane
    // itself stages the rgba_u16 expand) and never grow a scratch of their own.
    // Any colour output (u8 or u16 RGB / RGBA / HSV) consumes the centered
    // chroma; a luma-only row never does, so it neither reserves nor upsamples
    // it (and the reserve below is what makes the later upsample infallible).
    let need_centered_chroma = (center_sited || bottom_v) && want_color;
    if need_centered_chroma {
      reserve_420_chroma_full_u16(chroma_full_u16, w, h)?;
    }
    // Bottom-sited vertical-phase one-row chroma lookback (RFC #238 S6d): reserve
    // it on EVERY bottom-sited OUTPUT row — colour OR luma-only — because the
    // lookback is maintained so a LATER colour row can box-blend it (the
    // luma-only staging runs below, before any luma write). A no-output row
    // returned early above, so it never reaches here and never primes the
    // lookback. Reserved BEFORE any output write (the #180 preflight ordering).
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

    // Bottom-sited LUMA-ONLY row (RFC #238 S6d): the colour upsample below —
    // which normally refreshes the vertical lookback — won't run, so stage the
    // current chroma row HERE, after its reservation above and BEFORE any luma
    // write, so a later colour row in the same frame can box-blend it (a
    // luma-only-then-colour in-order sequence reconstructs the same Bottom
    // vertical phase as the all-output walk). A colour row instead stages inside
    // `upsample_420_chroma_sited_u16` after reading the previous lookback, so
    // this is skipped for it; a no-output row returned early above, so
    // `!want_color` here is a genuine luma-only row. The validity tag
    // (`chroma_prev_row`) still guards out-of-sequence / cross-frame reads.
    if bottom_v && !want_color {
      stage_420_chroma_prev_u16(
        chroma_prev_u16,
        chroma_prev_row,
        row.u_half(),
        row.v_half(),
        idx,
        w,
      );
    }

    // Centered full-width chroma, reconstructed ONCE per row from the wire-format
    // half-width U / V and reused by every colour decode (u16 and u8). Infallible
    // — the scratches were reserved above. The default left/unspecified siting
    // leaves it `None`, so the fused 4:2:0 kernels upsample chroma in-register
    // instead and the output stays byte-identical. `Center` / `Top` take the
    // plain horizontal centered phase-0.5 fold; `Bottom` (RFC #238 S6d)
    // additionally box-blends the even output row with the previous chroma row
    // (`stage = true` — the direct decode's post-reconstruction work is
    // infallible, so advancing the lookback here is safe).
    let centered = if need_centered_chroma {
      Some(upsample_420_chroma_sited_u16::<BITS>(
        chroma_full_u16,
        chroma_prev_u16,
        chroma_prev_row,
        row.u_half(),
        row.v_half(),
        idx,
        bottom_v,
        center_sited,
        true,
        w,
        BE,
      ))
    } else {
      None
    };

    // Freeze the effective 4:2:0 phase on the first output-bearing row — AFTER the
    // fallible scratch reserves above have succeeded, so an `AllocationFailed` row
    // stays retryable (frozen stays unset); later rows are checked against it up
    // top. Both the horizontal centered flag and the vertical `Bottom` flag are
    // pinned together.
    if need_output && frozen_chroma_centered.is_none() {
      *frozen_chroma_centered = Some(center_sited);
      *frozen_chroma_bottom_v = Some(bottom_v);
    }

    let matrix = row.matrix();
    let full_range = row.full_range();

    // Luma: downshift 10‑bit Y to 8‑bit for the existing u8 luma
    // buffer contract. Bit‑extension by `(BITS - 8)` preserves the
    // most significant bits — functionally equivalent to FFmpeg's
    // `>> (BITS - 8)` conversion used by many downstream analyses.
    if let Some(luma) = luma.as_deref_mut() {
      let dst = &mut luma[one_plane_start..one_plane_end];
      for (d, &s) in dst.iter_mut().zip(row.y().iter()) {
        // Normalize BE-encoded wire bytes to host-native before the
        // luma downshift — without this, a valid BE mid-gray sample
        // (`1 << (BITS - 1)`, e.g. `0x0100` for 9-bit, `0x0200` for
        // 10-bit, `0x0800` for 12-bit) would be byte-swapped on a LE
        // host and the `>> (BITS - 8)` would write 0 instead of 128.
        let logical = if BE { u16::from_be(s) } else { u16::from_le(s) };
        *d = (logical >> (BITS - 8)) as u8;
      }
    }

    // ===== u16 RGB / RGBA path (Strategy A) =====
    // u16 outputs are written via the native-depth row primitive, kept
    // independent of the u8 path: the two have different scale params
    // inside `range_params_n` and can't share an intermediate without
    // losing precision. Within the u16 family, however, the RGB row
    // and RGBA row are bit-identical for R/G/B, so we run the RGB
    // kernel once and fan out to RGBA via the cheap pad.
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
        yuv420p10_to_rgba_u16_row_endian(
          row.y(),
          row.u_half(),
          row.v_half(),
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
        yuv420p10_to_rgb_u16_row_endian(
          row.y(),
          row.u_half(),
          row.v_half(),
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
    // HSV-without-RGB-or-RGBA goes through the direct `*_to_hsv_row_endian`
    // kernel (no source-width RGB scratch — the SIMD path stages a fixed
    // 8-bit RGB chunk internally). RGB or RGBA also attached keeps the
    // convert-once-then-derive path alive via `need_rgb_kernel`. Centered
    // siting (#302) routes each colour kernel through its 4:4:4 twin, fed the
    // full-width phase-0.5 chroma reconstructed above.
    let want_hsv_direct = want_hsv && !want_rgb && !want_rgba;
    let need_rgb_kernel = want_rgb || (want_hsv && want_rgba);

    if want_hsv_direct {
      let hsv = hsv.as_mut().expect("want_hsv_direct implies hsv attached");
      let (h, s, v) = hsv.hsv();
      if let Some((u_full, v_full)) = centered {
        yuv444p10_to_hsv_row_endian(
          row.y(),
          u_full,
          v_full,
          &mut h[one_plane_start..one_plane_end],
          &mut s[one_plane_start..one_plane_end],
          &mut v[one_plane_start..one_plane_end],
          w,
          matrix,
          full_range,
          use_simd,
          BE,
        );
      } else {
        yuv420p10_to_hsv_row_endian(
          row.y(),
          row.u_half(),
          row.v_half(),
          &mut h[one_plane_start..one_plane_end],
          &mut s[one_plane_start..one_plane_end],
          &mut v[one_plane_start..one_plane_end],
          w,
          matrix,
          full_range,
          use_simd,
          BE,
        );
      }
      return Ok(());
    }

    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      if let Some((u_full, v_full)) = centered {
        yuv444p10_to_rgba_row_endian(
          row.y(),
          u_full,
          v_full,
          rgba_row,
          w,
          matrix,
          full_range,
          use_simd,
          BE,
        );
      } else {
        yuv420p10_to_rgba_row_endian(
          row.y(),
          row.u_half(),
          row.v_half(),
          rgba_row,
          w,
          matrix,
          full_range,
          use_simd,
          BE,
        );
      }
      return Ok(());
    }

    if !need_rgb_kernel {
      return Ok(());
    }

    // 8‑bit RGB path — either writes to the caller's buffer (when
    // `with_rgb` is set) or to the lazily‑grown scratch (when HSV is
    // requested without RGB). Mirrors the 8‑bit source impls' layout.
    let rgb_row = rgb_row_buf_or_scratch(
      rgb.as_deref_mut(),
      rgb_scratch,
      one_plane_start,
      one_plane_end,
      w,
      h,
    )?;

    if let Some((u_full, v_full)) = centered {
      yuv444p10_to_rgb_row_endian(
        row.y(),
        u_full,
        v_full,
        rgb_row,
        w,
        matrix,
        full_range,
        use_simd,
        BE,
      );
    } else {
      yuv420p10_to_rgb_row_endian(
        row.y(),
        row.u_half(),
        row.v_half(),
        rgb_row,
        w,
        matrix,
        full_range,
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

// ---- Yuv420p12 impl ----------------------------------------------------

impl<'a, R, const BE: bool> MixedSinker<'a, Yuv420p12<BE>, R> {
  /// Attaches a packed **`u16`** RGB output buffer. Mirrors
  /// [`MixedSinker<Yuv420p10>::with_rgb_u16`] but produces 12‑bit
  /// output (values in `[0, 4095]` in the low 12 of each `u16`, upper
  /// 4 zero). Length is measured in `u16` **elements** (`width x
  /// height x 3`).
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

  /// Attaches a packed **8‑bit** RGBA output buffer. The 12‑bit YUV
  /// source is converted to 8‑bit RGBA via the `BITS = 12` Q15 kernel
  /// family; alpha = `0xFF` (Yuv420p12 has no alpha plane).
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

  /// Attaches a packed **`u16`** RGBA output buffer. 12‑bit
  /// low‑packed (`(1 << 12) - 1 = 4095` max). Length is measured in
  /// `u16` **elements** (`width x height x 4`). Alpha element is
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

impl<R, const BE: bool> Yuv420p12Sink<BE> for MixedSinker<'_, Yuv420p12<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, Yuv420p12<BE>, R> {
  type Input<'r> = Yuv420p12Row<'r>;
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

  #[allow(clippy::too_many_lines)]
  fn process(&mut self, row: Yuv420p12Row<'_>) -> Result<(), Self::Error> {
    // Bit depth is fixed by the format (12) — declared as a const so
    // the downshift for u8 luma stays obvious at the call site.
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
    if row.u_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::UHalf12,
        idx,
        w / 2,
        row.u_half().len(),
      )));
    }
    if row.v_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::VHalf12,
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

    // Chroma siting (#302): drives the identity-plan horizontal chroma phase.
    // `Copy`, so read it out before the field split-borrow below.
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
      chroma_full_u16,
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
      native_420_u16,
      frozen_native_route,
      frozen_chroma_centered,
      frozen_chroma_bottom_v,
      chroma_prev_u16,
      chroma_prev_row,
      ..
    } = self;

    // Non-identity plan: native tier (bin native planes, convert once at
    // output width via 4:4:4 kernels) vs row-stage tier (convert each
    // source row then bin); `with_native(false)` forces the latter. See
    // the Yuv420p10 impl for the full chroma-contract rationale.
    if let Some(plan) = plan.as_ref() {
      let matrix = row.matrix();
      let full_range = row.full_range();
      let (y, u_half, v_half) = (row.y(), row.u_half(), row.v_half());
      // RFC #238 S6a — 4:2:0 HORIZONTAL chroma siting. The centered group
      // (`Center` / `Top` / `Bottom`, [`chroma_420_center_sited_h`]) samples
      // chroma at `+0.5` luma = `+0.25` chroma-sample horizontally; the co-sited
      // / unspecified group is phase 0 (today's byte-identical decode). Siting
      // enters the chroma RECONSTRUCTION only — the averaging tier is still
      // chosen by `select_insertion_point` below — so the native fast tier folds
      // the horizontal phase into the `area_chroma_420` chroma weights while the
      // row-stage and filter tiers reconstruct full-width `u16` chroma and decode
      // 4:4:4. VERTICAL stays co-sited (`v_phase = 0`): S6a routes the horizontal
      // Top / Center phase only; `Bottom`'s vertical blend is a later stage.
      let center_sited = chroma_420_center_sited_h(chroma_location);
      // RFC #238 S6d — 4:2:0 VERTICAL `Bottom` (`v = 1`) siting on top of the
      // S6a horizontal fold. `Bottom` ([`chroma_420_bottom_sited_v`]) is a strict
      // sub-case of `center_sited` (it is `h = 0.5, v = 1`), so it rides the
      // centered reconstruction below; the binning tiers additionally fold the
      // `v = 1` triangle into `area_chroma_420`'s vertical weights, and the
      // RGB-domain reconstruction tiers box-blend the even output row's chroma
      // with the previous chroma row via the `chroma_prev_u16` lookback. `Center`
      // / `Top` keep `v_phase = 0` (co-sited vertical, byte-identical to S6a).
      let bottom_v = chroma_420_bottom_sited_v(chroma_location);
      let chroma_h_phase = if center_sited {
        YUV422P_CENTERED_H_PHASE
      } else {
        0.0
      };
      let chroma_v_phase = if bottom_v { 1.0 } else { 0.0 };
      // Whether this call carries any output — the EXACT set both tiers'
      // preflight tests (`luma || rgb || rgba || hsv || rgb_u16 || rgba_u16`).
      // The route / siting freezes only on an output-bearing row a tier ACCEPTS;
      // a no-output call consumes no stream state, so it must not freeze.
      let need_output = luma.is_some()
        || rgb.is_some()
        || rgba.is_some()
        || hsv.is_some()
        || rgb_u16.is_some()
        || rgba_u16.is_some();
      // Only the colour tiers reconstruct full-width chroma for the centered
      // decode; a luma-only centered row bins native Y unchanged (siting is a
      // chroma-only property).
      let want_color =
        rgb.is_some() || rgba.is_some() || hsv.is_some() || rgb_u16.is_some() || rgba_u16.is_some();
      // Freeze the effective 4:2:0 chroma siting on the first output-bearing row
      // (mirrors the `frozen_native_route` freeze below). This CHECK is at the
      // always-compiled choke point every tier passes through; the matching SET
      // rides each tier's accept path (never before dispatch, so a rejected row
      // leaves it unset for a corrected retry). A later row observing a different
      // phase would bin a mixture of co-sited and centered chroma, so it is
      // rejected here before any reconstruction.
      if need_output
        && let Some(frozen) = *frozen_chroma_centered
        && (frozen != center_sited || *frozen_chroma_bottom_v != Some(bottom_v))
      {
        return Err(MixedSinkerError::ChromaSitingChanged(
          ChromaSitingChanged::new(idx),
        ));
      }
      // A `Filter` plan routes to the filter resampler BEFORE the
      // native/row-stage route machinery: the native fast tier is an
      // area-specific optimization that never sees a filter plan, and the
      // per-sink plan kind is fixed at construction, so a filter sink bypasses
      // the `frozen_native_route` interaction entirely. It converts the
      // separate Y/U/V planes to a source-width u8 + native-u16 RGB row (the
      // SAME closures the row-stage tier uses) and filter-resamples them plus
      // the native Y — the filter twin of the row-stage tier. The shared tail
      // clamps every sub-16-bit colour sample AND the native Y to
      // `(1 << BITS) - 1`. Yuv420p exposes no `luma_u16`, so it is `&mut None`.
      if plan.kind().is_filter() {
        // Reject a multi-kernel (BICUBLIN) plan BEFORE the centered reserve
        // below — the delegate's first act is this same check, so hoisting it
        // keeps a rejected filter plan from reserving / reconstructing chroma
        // first (the #180 reject-before-allocation invariant). Idempotent — the
        // delegate re-runs it.
        plan.ensure_single_kernel_filter()?;
        if (center_sited || bottom_v) && want_color {
          // Centered filter: reconstruct full-width `u16` chroma, but ONLY after
          // the resample preflight (frozen-output + sequence), so an
          // out-of-sequence / rejected row is caught before the chroma
          // reservation (#180). `packed_yuv422_triple_filter_resample` re-runs
          // the idempotent preflight and owns the transactional commit. The
          // HORIZONTAL centered reconstruction is all the row-stage / filter
          // tiers need — the walker already handed this luma row its
          // (vertically co-sited) chroma row.
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
          if bottom_v {
            reserve_420_chroma_prev_u16(chroma_prev_u16, w, h)?;
          }
          // `stage = false`: DEFER the lookback advance until AFTER the fallible
          // filter commit accepts the row (below), so a rejected row leaves the
          // predecessor in place for a clean retry (#180 state-atomicity).
          let (u_full, v_full) = upsample_420_chroma_sited_u16::<BITS>(
            chroma_full_u16,
            chroma_prev_u16,
            chroma_prev_row,
            u_half,
            v_half,
            idx,
            bottom_v,
            center_sited,
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
            |scratch| deinterleave_y_high_bit_masked::<BITS, BE>(y, scratch, w),
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
          );
          // Bottom lookback: advance only AFTER the filter resample accepts the
          // row (`r.is_ok()`), so a rejected row leaves the predecessor for a
          // clean retry — the sited reconstruction read it above but did not
          // stage. Inside the centered && want_color arm, so gate on `bottom_v`.
          if r.is_ok() && bottom_v {
            stage_420_chroma_prev_u16(chroma_prev_u16, chroma_prev_row, u_half, v_half, idx, w);
          }
          if r.is_ok() && need_output && frozen_chroma_centered.is_none() {
            *frozen_chroma_centered = Some(center_sited);
            *frozen_chroma_bottom_v = Some(bottom_v);
          }
          return r;
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
          |scratch| deinterleave_y_high_bit_masked::<BITS, BE>(y, scratch, w),
          |scratch| {
            yuv420p12_to_rgb_row_endian(
              y, u_half, v_half, scratch, w, matrix, full_range, use_simd, BE,
            )
          },
          |scratch| {
            yuv420p12_to_rgb_u16_row_endian(
              y, u_half, v_half, scratch, w, matrix, full_range, use_simd, BE,
            )
          },
        );
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
      // a true no-op, route-invisible regardless of row index. A
      // preflight-rejected (out-of-sequence / frozen) output-bearing call
      // returns Err before the SET, so it leaves `frozen_native_route`
      // untouched and a later same-or-other-route retry is not falsely
      // rejected.
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
          native_eligible: YUV420P_HIGH_BIT_NATIVE_ELIGIBLE,
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
          // RFC #238 S6a point-of-use siting invalidation: a reused sink's
          // cached join is only `reset` between frames, so a frame whose
          // `chroma_location` moved to a different horizontal phase must REBUILD
          // it (`area_chroma_420` folds the phase into the chroma weights). Drop
          // the stale-phase join ONLY on the in-sequence first row of a fresh
          // frame (`idx == 0`, `next_y() == 0`) so a mid-frame / out-of-sequence
          // row rejects against the INTACT join and a corrected retry rebuilds
          // cleanly; a luma-only join carries no chroma phase and is never
          // dropped. Move it OUT (the delegate builds the replacement into the
          // field, keeping it untouched until every pre-feed allocation
          // succeeds) and restore the intact prior-phase join on a rejected
          // rebuild so the row mutates no join state.
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
          let native_result = yuv420p16_process_native::<BITS, BE>(
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
            y,
            u_half,
            v_half,
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
                false,
              )
            },
            use_simd,
          );
          // Restore the taken stale-phase join if the delegate's rebuild was
          // rejected at any pre-feed step: it leaves the field `None` on such a
          // failure, so restoring the intact prior-phase join leaves the
          // rejected row mutating no join state. A non-stale row took nothing.
          if stale_native && native_result.is_err() {
            *native_420_u16 = prev_native;
          }
          native_result?;
          if frozen_native_route.is_none() && need_output {
            *frozen_native_route = Some(true);
          }
          // RFC #238 S6a: freeze the siting on the same accepted output row.
          if frozen_chroma_centered.is_none() && need_output {
            *frozen_chroma_centered = Some(center_sited);
            *frozen_chroma_bottom_v = Some(bottom_v);
          }
          return Ok(());
        }
        InsertionPoint::EncodedOutput => {
          // Row-stage tail. Same CHECK-before / SET-after split: dispatch, then
          // freeze the route to row-stage only when the call accepts an
          // output-bearing row (a no-output call returns Ok with `need_output`
          // false; an out-of-sequence / frozen row returns Err via `?`).
          if (center_sited || bottom_v) && want_color {
            // Centered row-stage: reconstruct full-width `u16` chroma AFTER the
            // resample preflight (frozen-output + sequence), so an
            // out-of-sequence / rejected row is caught before the chroma
            // reservation (#180). `packed_yuv422_triple_resample` re-runs the
            // idempotent preflight and owns the transactional commit. HORIZONTAL
            // reconstruction only — the walker handed this luma row its
            // (vertically co-sited) chroma row.
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
            if bottom_v {
              reserve_420_chroma_prev_u16(chroma_prev_u16, w, h)?;
            }
            // `stage = false`: DEFER the lookback advance until AFTER the fallible
            // row-stage commit accepts the row (below), so a rejected row leaves
            // the predecessor for a clean retry (#180 state-atomicity).
            let (u_full, v_full) = upsample_420_chroma_sited_u16::<BITS>(
              chroma_full_u16,
              chroma_prev_u16,
              chroma_prev_row,
              u_half,
              v_half,
              idx,
              bottom_v,
              center_sited,
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
              |scratch| deinterleave_y_high_bit_masked::<BITS, BE>(y, scratch, w),
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
            )?;
            // Bottom lookback: advance only AFTER the row-stage resample accepts
            // the row (the `?` above already returned any reject), so a rejected
            // row leaves the predecessor for a clean retry — the sited
            // reconstruction read it above but did not stage.
            if bottom_v {
              stage_420_chroma_prev_u16(chroma_prev_u16, chroma_prev_row, u_half, v_half, idx, w);
            }
          } else {
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
              |scratch| deinterleave_y_high_bit_masked::<BITS, BE>(y, scratch, w),
              |scratch| {
                yuv420p12_to_rgb_row_endian(
                  y, u_half, v_half, scratch, w, matrix, full_range, use_simd, BE,
                )
              },
              |scratch| {
                yuv420p12_to_rgb_u16_row_endian(
                  y, u_half, v_half, scratch, w, matrix, full_range, use_simd, BE,
                )
              },
            )?;
          }
          if frozen_native_route.is_none() && need_output {
            *frozen_native_route = Some(false);
          }
          // RFC #238 S6a: freeze the siting on the same accepted output row.
          if frozen_chroma_centered.is_none() && need_output {
            *frozen_chroma_centered = Some(center_sited);
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

    // Resolve the full output set up front so the no-output guard below
    // short-circuits before ANY per-row offset arithmetic and the atomicity
    // preflight runs before any output row (luma included) is written.
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();
    let want_rgb_u16 = rgb_u16.is_some();
    let want_rgba_u16 = rgba_u16.is_some();
    // Whether this row produces any colour output (and so runs the centered /
    // bottom-sited chroma upsample). A bottom-sited row maintains the vertical
    // lookback even when it produces only luma — see the luma-only staging below
    // — so `want_color` gates only the colour scratches.
    let want_color = want_rgb || want_rgba || want_hsv || want_rgb_u16 || want_rgba_u16;
    // Repo-wide no-output invariant (RFC #238 S6d): a `process` call carrying NO
    // output — no colour, no luma — must run NOTHING: no per-row offset
    // arithmetic, no allocation, no state mutation (the bottom-sited vertical
    // lookback included). Returning HERE, before the `idx * w` offsets below,
    // keeps the invariant overflow-safe (a no-output call never ran an
    // attach-time `w x h` validation, so `idx * w` could overflow on a 32-bit
    // target) AND means such a row never reserves `chroma_prev_u16` nor primes
    // the lookback (so it can never make a later colour even row box-blend
    // through an invisible, never-output row). Yuv420p exposes no `luma_u16`.
    let need_output = want_color || luma.is_some();
    if !need_output {
      return Ok(());
    }

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // Chroma siting (#302): the centered horizontal sitings reconstruct chroma
    // at the phase-0.5 position; the default / co-sited path keeps the
    // byte-identical decode (the fused high-bit 4:2:0 kernels upsample chroma
    // in-register, exactly as before).
    let center_sited = chroma_420_center_sited_h(chroma_location);
    // RFC #238 S6d: `Bottom` (a strict sub-case of `center_sited`) additionally
    // box-blends the even output row's chroma with the previous chroma row via
    // the `chroma_prev_u16` lookback maintained below; `Center` / `Top` keep the
    // vertical-replicate (co-sited) decode, byte-identical to S6a.
    let bottom_v = chroma_420_bottom_sited_v(chroma_location);

    // Per-frame chroma-siting freeze (RFC #238, mirroring the resample-path guard
    // above): the first output-bearing row pins the effective 4:2:0 phase — BOTH
    // the horizontal centered flag and the vertical `Bottom` flag. A later row
    // whose siting flipped would decode a mixture of phases into ONE frame, or box-
    // blend against a STALE `chroma_prev_u16` lookback, so reject it here BEFORE
    // any scratch reserve, lookback priming, or output write. This CHECK precedes
    // the `stage_420_chroma_prev_u16` / reconstruct staging below so a rejected
    // flip leaves `chroma_prev_u16` / `chroma_prev_row` untouched (retry-atomic,
    // #180). `begin_frame`'s `reset_high_bit_yuv_streams` clears the freeze so the
    // next frame may pick either phase.
    if need_output
      && let Some(frozen) = *frozen_chroma_centered
      && (frozen != center_sited || *frozen_chroma_bottom_v != Some(bottom_v))
    {
      return Err(MixedSinkerError::ChromaSitingChanged(
        ChromaSitingChanged::new(idx),
      ));
    }

    // Atomicity preflight (#302 / #308 / #314, cf. the crate's #180 resample
    // fix): reserve EVERY fallible row scratch this identity row can touch
    // BEFORE any output row is written (the luma plane below, then the u16 / u8
    // RGB / RGBA / HSV fan-out), so an allocator refusal returns a typed
    // `AllocationFailed` leaving the output frame untouched rather than
    // partially mutated. Two scratches can grow:
    //  1. the centered-siting full-width `u16` chroma (`chroma_full_u16`),
    //     needed by ANY colour output (u8 OR u16 RGB / RGBA / HSV); and
    //  2. the u8 RGB row buffer, reached exactly when a colour decode needs an
    //     RGB row but no caller RGB buffer is borrowable — `want_hsv &&
    //     want_rgba && !want_rgb` (`rgb_row_buf_or_scratch`'s own scratch arm).
    // The later `upsample_420_chroma_center_h_u16` / `rgb_row_buf_or_scratch`
    // calls then reuse the already-sized buffers, so the default path is
    // byte-identical; only the failure-path ordering changes. The u16 RGB /
    // RGBA outputs write straight into their caller buffers (the rgb_u16 plane
    // itself stages the rgba_u16 expand) and never grow a scratch of their own.
    // Any colour output (u8 or u16 RGB / RGBA / HSV) consumes the centered
    // chroma; a luma-only row never does, so it neither reserves nor upsamples
    // it (and the reserve below is what makes the later upsample infallible).
    let need_centered_chroma = (center_sited || bottom_v) && want_color;
    if need_centered_chroma {
      reserve_420_chroma_full_u16(chroma_full_u16, w, h)?;
    }
    // Bottom-sited vertical-phase one-row chroma lookback (RFC #238 S6d): reserve
    // it on EVERY bottom-sited OUTPUT row — colour OR luma-only — because the
    // lookback is maintained so a LATER colour row can box-blend it (the
    // luma-only staging runs below, before any luma write). A no-output row
    // returned early above, so it never reaches here and never primes the
    // lookback. Reserved BEFORE any output write (the #180 preflight ordering).
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

    // Bottom-sited LUMA-ONLY row (RFC #238 S6d): the colour upsample below —
    // which normally refreshes the vertical lookback — won't run, so stage the
    // current chroma row HERE, after its reservation above and BEFORE any luma
    // write, so a later colour row in the same frame can box-blend it (a
    // luma-only-then-colour in-order sequence reconstructs the same Bottom
    // vertical phase as the all-output walk). A colour row instead stages inside
    // `upsample_420_chroma_sited_u16` after reading the previous lookback, so
    // this is skipped for it; a no-output row returned early above, so
    // `!want_color` here is a genuine luma-only row. The validity tag
    // (`chroma_prev_row`) still guards out-of-sequence / cross-frame reads.
    if bottom_v && !want_color {
      stage_420_chroma_prev_u16(
        chroma_prev_u16,
        chroma_prev_row,
        row.u_half(),
        row.v_half(),
        idx,
        w,
      );
    }

    // Centered full-width chroma, reconstructed ONCE per row from the wire-format
    // half-width U / V and reused by every colour decode (u16 and u8). Infallible
    // — the scratches were reserved above. The default left/unspecified siting
    // leaves it `None`, so the fused 4:2:0 kernels upsample chroma in-register
    // instead and the output stays byte-identical. `Center` / `Top` take the
    // plain horizontal centered phase-0.5 fold; `Bottom` (RFC #238 S6d)
    // additionally box-blends the even output row with the previous chroma row
    // (`stage = true` — the direct decode's post-reconstruction work is
    // infallible, so advancing the lookback here is safe).
    let centered = if need_centered_chroma {
      Some(upsample_420_chroma_sited_u16::<BITS>(
        chroma_full_u16,
        chroma_prev_u16,
        chroma_prev_row,
        row.u_half(),
        row.v_half(),
        idx,
        bottom_v,
        center_sited,
        true,
        w,
        BE,
      ))
    } else {
      None
    };

    // Freeze the effective 4:2:0 phase on the first output-bearing row — AFTER the
    // fallible scratch reserves above have succeeded, so an `AllocationFailed` row
    // stays retryable (frozen stays unset); later rows are checked against it up
    // top. Both the horizontal centered flag and the vertical `Bottom` flag are
    // pinned together.
    if need_output && frozen_chroma_centered.is_none() {
      *frozen_chroma_centered = Some(center_sited);
      *frozen_chroma_bottom_v = Some(bottom_v);
    }

    let matrix = row.matrix();
    let full_range = row.full_range();

    if let Some(luma) = luma.as_deref_mut() {
      let dst = &mut luma[one_plane_start..one_plane_end];
      for (d, &s) in dst.iter_mut().zip(row.y().iter()) {
        // Normalize BE-encoded wire bytes to host-native before the
        // luma downshift — without this, a valid BE mid-gray sample
        // (`1 << (BITS - 1)`, e.g. `0x0100` for 9-bit, `0x0200` for
        // 10-bit, `0x0800` for 12-bit) would be byte-swapped on a LE
        // host and the `>> (BITS - 8)` would write 0 instead of 128.
        let logical = if BE { u16::from_be(s) } else { u16::from_le(s) };
        *d = (logical >> (BITS - 8)) as u8;
      }
    }

    // ===== u16 RGB / RGBA path (Strategy A) — see Yuv420p10 for rationale.
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
        yuv420p12_to_rgba_u16_row_endian(
          row.y(),
          row.u_half(),
          row.v_half(),
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
        yuv420p12_to_rgb_u16_row_endian(
          row.y(),
          row.u_half(),
          row.v_half(),
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
    // HSV-without-RGB-or-RGBA goes through the direct `*_to_hsv_row_endian`
    // kernel (no source-width RGB scratch — the SIMD path stages a fixed
    // 8-bit RGB chunk internally). RGB or RGBA also attached keeps the
    // convert-once-then-derive path alive via `need_rgb_kernel`. Centered
    // siting (#302) routes each colour kernel through its 4:4:4 twin, fed the
    // full-width phase-0.5 chroma reconstructed above.
    let want_hsv_direct = want_hsv && !want_rgb && !want_rgba;
    let need_rgb_kernel = want_rgb || (want_hsv && want_rgba);

    if want_hsv_direct {
      let hsv = hsv.as_mut().expect("want_hsv_direct implies hsv attached");
      let (h, s, v) = hsv.hsv();
      if let Some((u_full, v_full)) = centered {
        yuv444p12_to_hsv_row_endian(
          row.y(),
          u_full,
          v_full,
          &mut h[one_plane_start..one_plane_end],
          &mut s[one_plane_start..one_plane_end],
          &mut v[one_plane_start..one_plane_end],
          w,
          matrix,
          full_range,
          use_simd,
          BE,
        );
      } else {
        yuv420p12_to_hsv_row_endian(
          row.y(),
          row.u_half(),
          row.v_half(),
          &mut h[one_plane_start..one_plane_end],
          &mut s[one_plane_start..one_plane_end],
          &mut v[one_plane_start..one_plane_end],
          w,
          matrix,
          full_range,
          use_simd,
          BE,
        );
      }
      return Ok(());
    }

    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      if let Some((u_full, v_full)) = centered {
        yuv444p12_to_rgba_row_endian(
          row.y(),
          u_full,
          v_full,
          rgba_row,
          w,
          matrix,
          full_range,
          use_simd,
          BE,
        );
      } else {
        yuv420p12_to_rgba_row_endian(
          row.y(),
          row.u_half(),
          row.v_half(),
          rgba_row,
          w,
          matrix,
          full_range,
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

    if let Some((u_full, v_full)) = centered {
      yuv444p12_to_rgb_row_endian(
        row.y(),
        u_full,
        v_full,
        rgb_row,
        w,
        matrix,
        full_range,
        use_simd,
        BE,
      );
    } else {
      yuv420p12_to_rgb_row_endian(
        row.y(),
        row.u_half(),
        row.v_half(),
        rgb_row,
        w,
        matrix,
        full_range,
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

// ---- Yuv420p14 impl ----------------------------------------------------

impl<'a, R, const BE: bool> MixedSinker<'a, Yuv420p14<BE>, R> {
  /// Attaches a packed **`u16`** RGB output buffer. Produces 14‑bit
  /// output (values in `[0, 16383]` in the low 14 of each `u16`, upper
  /// 2 zero). Length is measured in `u16` **elements** (`width x
  /// height x 3`).
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

  /// Attaches a packed **8‑bit** RGBA output buffer. The 14‑bit YUV
  /// source is converted to 8‑bit RGBA via the `BITS = 14` Q15 kernel
  /// family; alpha = `0xFF` (Yuv420p14 has no alpha plane).
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

  /// Attaches a packed **`u16`** RGBA output buffer. 14‑bit
  /// low‑packed (`(1 << 14) - 1 = 16383` max). Length is measured in
  /// `u16` **elements** (`width x height x 4`). Alpha element is
  /// `(1 << 14) - 1`.
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

impl<R, const BE: bool> Yuv420p14Sink<BE> for MixedSinker<'_, Yuv420p14<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, Yuv420p14<BE>, R> {
  type Input<'r> = Yuv420p14Row<'r>;
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

  #[allow(clippy::too_many_lines)]
  fn process(&mut self, row: Yuv420p14Row<'_>) -> Result<(), Self::Error> {
    const BITS: u32 = 14;

    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if w & 1 != 0 {
      return Err(MixedSinkerError::WidthAlignment(WidthAlignment::odd(w)));
    }
    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Y14,
        idx,
        w,
        row.y().len(),
      )));
    }
    if row.u_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::UHalf14,
        idx,
        w / 2,
        row.u_half().len(),
      )));
    }
    if row.v_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::VHalf14,
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

    // Chroma siting (#302): drives the identity-plan horizontal chroma phase.
    // `Copy`, so read it out before the field split-borrow below.
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
      chroma_full_u16,
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
      native_420_u16,
      frozen_native_route,
      frozen_chroma_centered,
      frozen_chroma_bottom_v,
      chroma_prev_u16,
      chroma_prev_row,
      ..
    } = self;

    // Non-identity plan: native tier (bin native planes, convert once at
    // output width via 4:4:4 kernels) vs row-stage tier (convert each
    // source row then bin); `with_native(false)` forces the latter. See
    // the Yuv420p10 impl for the full chroma-contract rationale.
    if let Some(plan) = plan.as_ref() {
      let matrix = row.matrix();
      let full_range = row.full_range();
      let (y, u_half, v_half) = (row.y(), row.u_half(), row.v_half());
      // RFC #238 S6a — 4:2:0 HORIZONTAL chroma siting. The centered group
      // (`Center` / `Top` / `Bottom`, [`chroma_420_center_sited_h`]) samples
      // chroma at `+0.5` luma = `+0.25` chroma-sample horizontally; the co-sited
      // / unspecified group is phase 0 (today's byte-identical decode). Siting
      // enters the chroma RECONSTRUCTION only — the averaging tier is still
      // chosen by `select_insertion_point` below — so the native fast tier folds
      // the horizontal phase into the `area_chroma_420` chroma weights while the
      // row-stage and filter tiers reconstruct full-width `u16` chroma and decode
      // 4:4:4. VERTICAL stays co-sited (`v_phase = 0`): S6a routes the horizontal
      // Top / Center phase only; `Bottom`'s vertical blend is a later stage.
      let center_sited = chroma_420_center_sited_h(chroma_location);
      // RFC #238 S6d — 4:2:0 VERTICAL `Bottom` (`v = 1`) siting on top of the
      // S6a horizontal fold. `Bottom` ([`chroma_420_bottom_sited_v`]) is a strict
      // sub-case of `center_sited` (it is `h = 0.5, v = 1`), so it rides the
      // centered reconstruction below; the binning tiers additionally fold the
      // `v = 1` triangle into `area_chroma_420`'s vertical weights, and the
      // RGB-domain reconstruction tiers box-blend the even output row's chroma
      // with the previous chroma row via the `chroma_prev_u16` lookback. `Center`
      // / `Top` keep `v_phase = 0` (co-sited vertical, byte-identical to S6a).
      let bottom_v = chroma_420_bottom_sited_v(chroma_location);
      let chroma_h_phase = if center_sited {
        YUV422P_CENTERED_H_PHASE
      } else {
        0.0
      };
      let chroma_v_phase = if bottom_v { 1.0 } else { 0.0 };
      // Whether this call carries any output — the EXACT set both tiers'
      // preflight tests (`luma || rgb || rgba || hsv || rgb_u16 || rgba_u16`).
      // The route / siting freezes only on an output-bearing row a tier ACCEPTS;
      // a no-output call consumes no stream state, so it must not freeze.
      let need_output = luma.is_some()
        || rgb.is_some()
        || rgba.is_some()
        || hsv.is_some()
        || rgb_u16.is_some()
        || rgba_u16.is_some();
      // Only the colour tiers reconstruct full-width chroma for the centered
      // decode; a luma-only centered row bins native Y unchanged (siting is a
      // chroma-only property).
      let want_color =
        rgb.is_some() || rgba.is_some() || hsv.is_some() || rgb_u16.is_some() || rgba_u16.is_some();
      // Freeze the effective 4:2:0 chroma siting on the first output-bearing row
      // (mirrors the `frozen_native_route` freeze below). This CHECK is at the
      // always-compiled choke point every tier passes through; the matching SET
      // rides each tier's accept path (never before dispatch, so a rejected row
      // leaves it unset for a corrected retry). A later row observing a different
      // phase would bin a mixture of co-sited and centered chroma, so it is
      // rejected here before any reconstruction.
      if need_output
        && let Some(frozen) = *frozen_chroma_centered
        && (frozen != center_sited || *frozen_chroma_bottom_v != Some(bottom_v))
      {
        return Err(MixedSinkerError::ChromaSitingChanged(
          ChromaSitingChanged::new(idx),
        ));
      }
      // A `Filter` plan routes to the filter resampler BEFORE the
      // native/row-stage route machinery: the native fast tier is an
      // area-specific optimization that never sees a filter plan, and the
      // per-sink plan kind is fixed at construction, so a filter sink bypasses
      // the `frozen_native_route` interaction entirely. It converts the
      // separate Y/U/V planes to a source-width u8 + native-u16 RGB row (the
      // SAME closures the row-stage tier uses) and filter-resamples them plus
      // the native Y — the filter twin of the row-stage tier. The shared tail
      // clamps every sub-16-bit colour sample AND the native Y to
      // `(1 << BITS) - 1`. Yuv420p exposes no `luma_u16`, so it is `&mut None`.
      if plan.kind().is_filter() {
        // Reject a multi-kernel (BICUBLIN) plan BEFORE the centered reserve
        // below — the delegate's first act is this same check, so hoisting it
        // keeps a rejected filter plan from reserving / reconstructing chroma
        // first (the #180 reject-before-allocation invariant). Idempotent — the
        // delegate re-runs it.
        plan.ensure_single_kernel_filter()?;
        if (center_sited || bottom_v) && want_color {
          // Centered filter: reconstruct full-width `u16` chroma, but ONLY after
          // the resample preflight (frozen-output + sequence), so an
          // out-of-sequence / rejected row is caught before the chroma
          // reservation (#180). `packed_yuv422_triple_filter_resample` re-runs
          // the idempotent preflight and owns the transactional commit. The
          // HORIZONTAL centered reconstruction is all the row-stage / filter
          // tiers need — the walker already handed this luma row its
          // (vertically co-sited) chroma row.
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
          if bottom_v {
            reserve_420_chroma_prev_u16(chroma_prev_u16, w, h)?;
          }
          // `stage = false`: DEFER the lookback advance until AFTER the fallible
          // filter commit accepts the row (below), so a rejected row leaves the
          // predecessor in place for a clean retry (#180 state-atomicity).
          let (u_full, v_full) = upsample_420_chroma_sited_u16::<BITS>(
            chroma_full_u16,
            chroma_prev_u16,
            chroma_prev_row,
            u_half,
            v_half,
            idx,
            bottom_v,
            center_sited,
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
            |scratch| deinterleave_y_high_bit_masked::<BITS, BE>(y, scratch, w),
            |scratch| {
              yuv444p14_to_rgb_row_endian(
                y, u_full, v_full, scratch, w, matrix, full_range, use_simd, BE,
              )
            },
            |scratch| {
              yuv444p14_to_rgb_u16_row_endian(
                y, u_full, v_full, scratch, w, matrix, full_range, use_simd, BE,
              )
            },
          );
          // Bottom lookback: advance only AFTER the filter resample accepts the
          // row (`r.is_ok()`), so a rejected row leaves the predecessor for a
          // clean retry — the sited reconstruction read it above but did not
          // stage. Inside the centered && want_color arm, so gate on `bottom_v`.
          if r.is_ok() && bottom_v {
            stage_420_chroma_prev_u16(chroma_prev_u16, chroma_prev_row, u_half, v_half, idx, w);
          }
          if r.is_ok() && need_output && frozen_chroma_centered.is_none() {
            *frozen_chroma_centered = Some(center_sited);
            *frozen_chroma_bottom_v = Some(bottom_v);
          }
          return r;
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
          |scratch| deinterleave_y_high_bit_masked::<BITS, BE>(y, scratch, w),
          |scratch| {
            yuv420p14_to_rgb_row_endian(
              y, u_half, v_half, scratch, w, matrix, full_range, use_simd, BE,
            )
          },
          |scratch| {
            yuv420p14_to_rgb_u16_row_endian(
              y, u_half, v_half, scratch, w, matrix, full_range, use_simd, BE,
            )
          },
        );
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
      // a true no-op, route-invisible regardless of row index. A
      // preflight-rejected (out-of-sequence / frozen) output-bearing call
      // returns Err before the SET, so it leaves `frozen_native_route`
      // untouched and a later same-or-other-route retry is not falsely
      // rejected.
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
          native_eligible: YUV420P_HIGH_BIT_NATIVE_ELIGIBLE,
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
          // RFC #238 S6a point-of-use siting invalidation: a reused sink's
          // cached join is only `reset` between frames, so a frame whose
          // `chroma_location` moved to a different horizontal phase must REBUILD
          // it (`area_chroma_420` folds the phase into the chroma weights). Drop
          // the stale-phase join ONLY on the in-sequence first row of a fresh
          // frame (`idx == 0`, `next_y() == 0`) so a mid-frame / out-of-sequence
          // row rejects against the INTACT join and a corrected retry rebuilds
          // cleanly; a luma-only join carries no chroma phase and is never
          // dropped. Move it OUT (the delegate builds the replacement into the
          // field, keeping it untouched until every pre-feed allocation
          // succeeds) and restore the intact prior-phase join on a rejected
          // rebuild so the row mutates no join state.
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
          let native_result = yuv420p16_process_native::<BITS, BE>(
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
            y,
            u_half,
            v_half,
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
                false,
              )
            },
            use_simd,
          );
          // Restore the taken stale-phase join if the delegate's rebuild was
          // rejected at any pre-feed step: it leaves the field `None` on such a
          // failure, so restoring the intact prior-phase join leaves the
          // rejected row mutating no join state. A non-stale row took nothing.
          if stale_native && native_result.is_err() {
            *native_420_u16 = prev_native;
          }
          native_result?;
          if frozen_native_route.is_none() && need_output {
            *frozen_native_route = Some(true);
          }
          // RFC #238 S6a: freeze the siting on the same accepted output row.
          if frozen_chroma_centered.is_none() && need_output {
            *frozen_chroma_centered = Some(center_sited);
            *frozen_chroma_bottom_v = Some(bottom_v);
          }
          return Ok(());
        }
        InsertionPoint::EncodedOutput => {
          // Row-stage tail. Same CHECK-before / SET-after split: dispatch, then
          // freeze the route to row-stage only when the call accepts an
          // output-bearing row (a no-output call returns Ok with `need_output`
          // false; an out-of-sequence / frozen row returns Err via `?`).
          if (center_sited || bottom_v) && want_color {
            // Centered row-stage: reconstruct full-width `u16` chroma AFTER the
            // resample preflight (frozen-output + sequence), so an
            // out-of-sequence / rejected row is caught before the chroma
            // reservation (#180). `packed_yuv422_triple_resample` re-runs the
            // idempotent preflight and owns the transactional commit. HORIZONTAL
            // reconstruction only — the walker handed this luma row its
            // (vertically co-sited) chroma row.
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
            if bottom_v {
              reserve_420_chroma_prev_u16(chroma_prev_u16, w, h)?;
            }
            // `stage = false`: DEFER the lookback advance until AFTER the fallible
            // row-stage commit accepts the row (below), so a rejected row leaves
            // the predecessor for a clean retry (#180 state-atomicity).
            let (u_full, v_full) = upsample_420_chroma_sited_u16::<BITS>(
              chroma_full_u16,
              chroma_prev_u16,
              chroma_prev_row,
              u_half,
              v_half,
              idx,
              bottom_v,
              center_sited,
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
              |scratch| deinterleave_y_high_bit_masked::<BITS, BE>(y, scratch, w),
              |scratch| {
                yuv444p14_to_rgb_row_endian(
                  y, u_full, v_full, scratch, w, matrix, full_range, use_simd, BE,
                )
              },
              |scratch| {
                yuv444p14_to_rgb_u16_row_endian(
                  y, u_full, v_full, scratch, w, matrix, full_range, use_simd, BE,
                )
              },
            )?;
            // Bottom lookback: advance only AFTER the row-stage resample accepts
            // the row (the `?` above already returned any reject), so a rejected
            // row leaves the predecessor for a clean retry — the sited
            // reconstruction read it above but did not stage.
            if bottom_v {
              stage_420_chroma_prev_u16(chroma_prev_u16, chroma_prev_row, u_half, v_half, idx, w);
            }
          } else {
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
              |scratch| deinterleave_y_high_bit_masked::<BITS, BE>(y, scratch, w),
              |scratch| {
                yuv420p14_to_rgb_row_endian(
                  y, u_half, v_half, scratch, w, matrix, full_range, use_simd, BE,
                )
              },
              |scratch| {
                yuv420p14_to_rgb_u16_row_endian(
                  y, u_half, v_half, scratch, w, matrix, full_range, use_simd, BE,
                )
              },
            )?;
          }
          if frozen_native_route.is_none() && need_output {
            *frozen_native_route = Some(false);
          }
          // RFC #238 S6a: freeze the siting on the same accepted output row.
          if frozen_chroma_centered.is_none() && need_output {
            *frozen_chroma_centered = Some(center_sited);
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

    // Resolve the full output set up front so the no-output guard below
    // short-circuits before ANY per-row offset arithmetic and the atomicity
    // preflight runs before any output row (luma included) is written.
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();
    let want_rgb_u16 = rgb_u16.is_some();
    let want_rgba_u16 = rgba_u16.is_some();
    // Whether this row produces any colour output (and so runs the centered /
    // bottom-sited chroma upsample). A bottom-sited row maintains the vertical
    // lookback even when it produces only luma — see the luma-only staging below
    // — so `want_color` gates only the colour scratches.
    let want_color = want_rgb || want_rgba || want_hsv || want_rgb_u16 || want_rgba_u16;
    // Repo-wide no-output invariant (RFC #238 S6d): a `process` call carrying NO
    // output — no colour, no luma — must run NOTHING: no per-row offset
    // arithmetic, no allocation, no state mutation (the bottom-sited vertical
    // lookback included). Returning HERE, before the `idx * w` offsets below,
    // keeps the invariant overflow-safe (a no-output call never ran an
    // attach-time `w x h` validation, so `idx * w` could overflow on a 32-bit
    // target) AND means such a row never reserves `chroma_prev_u16` nor primes
    // the lookback (so it can never make a later colour even row box-blend
    // through an invisible, never-output row). Yuv420p exposes no `luma_u16`.
    let need_output = want_color || luma.is_some();
    if !need_output {
      return Ok(());
    }

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // Chroma siting (#302): the centered horizontal sitings reconstruct chroma
    // at the phase-0.5 position; the default / co-sited path keeps the
    // byte-identical decode (the fused high-bit 4:2:0 kernels upsample chroma
    // in-register, exactly as before).
    let center_sited = chroma_420_center_sited_h(chroma_location);
    // RFC #238 S6d: `Bottom` (a strict sub-case of `center_sited`) additionally
    // box-blends the even output row's chroma with the previous chroma row via
    // the `chroma_prev_u16` lookback maintained below; `Center` / `Top` keep the
    // vertical-replicate (co-sited) decode, byte-identical to S6a.
    let bottom_v = chroma_420_bottom_sited_v(chroma_location);

    // Per-frame chroma-siting freeze (RFC #238, mirroring the resample-path guard
    // above): the first output-bearing row pins the effective 4:2:0 phase — BOTH
    // the horizontal centered flag and the vertical `Bottom` flag. A later row
    // whose siting flipped would decode a mixture of phases into ONE frame, or box-
    // blend against a STALE `chroma_prev_u16` lookback, so reject it here BEFORE
    // any scratch reserve, lookback priming, or output write. This CHECK precedes
    // the `stage_420_chroma_prev_u16` / reconstruct staging below so a rejected
    // flip leaves `chroma_prev_u16` / `chroma_prev_row` untouched (retry-atomic,
    // #180). `begin_frame`'s `reset_high_bit_yuv_streams` clears the freeze so the
    // next frame may pick either phase.
    if need_output
      && let Some(frozen) = *frozen_chroma_centered
      && (frozen != center_sited || *frozen_chroma_bottom_v != Some(bottom_v))
    {
      return Err(MixedSinkerError::ChromaSitingChanged(
        ChromaSitingChanged::new(idx),
      ));
    }

    // Atomicity preflight (#302 / #308 / #314, cf. the crate's #180 resample
    // fix): reserve EVERY fallible row scratch this identity row can touch
    // BEFORE any output row is written (the luma plane below, then the u16 / u8
    // RGB / RGBA / HSV fan-out), so an allocator refusal returns a typed
    // `AllocationFailed` leaving the output frame untouched rather than
    // partially mutated. Two scratches can grow:
    //  1. the centered-siting full-width `u16` chroma (`chroma_full_u16`),
    //     needed by ANY colour output (u8 OR u16 RGB / RGBA / HSV); and
    //  2. the u8 RGB row buffer, reached exactly when a colour decode needs an
    //     RGB row but no caller RGB buffer is borrowable — `want_hsv &&
    //     want_rgba && !want_rgb` (`rgb_row_buf_or_scratch`'s own scratch arm).
    // The later `upsample_420_chroma_center_h_u16` / `rgb_row_buf_or_scratch`
    // calls then reuse the already-sized buffers, so the default path is
    // byte-identical; only the failure-path ordering changes. The u16 RGB /
    // RGBA outputs write straight into their caller buffers (the rgb_u16 plane
    // itself stages the rgba_u16 expand) and never grow a scratch of their own.
    // Any colour output (u8 or u16 RGB / RGBA / HSV) consumes the centered
    // chroma; a luma-only row never does, so it neither reserves nor upsamples
    // it (and the reserve below is what makes the later upsample infallible).
    let need_centered_chroma = (center_sited || bottom_v) && want_color;
    if need_centered_chroma {
      reserve_420_chroma_full_u16(chroma_full_u16, w, h)?;
    }
    // Bottom-sited vertical-phase one-row chroma lookback (RFC #238 S6d): reserve
    // it on EVERY bottom-sited OUTPUT row — colour OR luma-only — because the
    // lookback is maintained so a LATER colour row can box-blend it (the
    // luma-only staging runs below, before any luma write). A no-output row
    // returned early above, so it never reaches here and never primes the
    // lookback. Reserved BEFORE any output write (the #180 preflight ordering).
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

    // Bottom-sited LUMA-ONLY row (RFC #238 S6d): the colour upsample below —
    // which normally refreshes the vertical lookback — won't run, so stage the
    // current chroma row HERE, after its reservation above and BEFORE any luma
    // write, so a later colour row in the same frame can box-blend it (a
    // luma-only-then-colour in-order sequence reconstructs the same Bottom
    // vertical phase as the all-output walk). A colour row instead stages inside
    // `upsample_420_chroma_sited_u16` after reading the previous lookback, so
    // this is skipped for it; a no-output row returned early above, so
    // `!want_color` here is a genuine luma-only row. The validity tag
    // (`chroma_prev_row`) still guards out-of-sequence / cross-frame reads.
    if bottom_v && !want_color {
      stage_420_chroma_prev_u16(
        chroma_prev_u16,
        chroma_prev_row,
        row.u_half(),
        row.v_half(),
        idx,
        w,
      );
    }

    // Centered full-width chroma, reconstructed ONCE per row from the wire-format
    // half-width U / V and reused by every colour decode (u16 and u8). Infallible
    // — the scratches were reserved above. The default left/unspecified siting
    // leaves it `None`, so the fused 4:2:0 kernels upsample chroma in-register
    // instead and the output stays byte-identical. `Center` / `Top` take the
    // plain horizontal centered phase-0.5 fold; `Bottom` (RFC #238 S6d)
    // additionally box-blends the even output row with the previous chroma row
    // (`stage = true` — the direct decode's post-reconstruction work is
    // infallible, so advancing the lookback here is safe).
    let centered = if need_centered_chroma {
      Some(upsample_420_chroma_sited_u16::<BITS>(
        chroma_full_u16,
        chroma_prev_u16,
        chroma_prev_row,
        row.u_half(),
        row.v_half(),
        idx,
        bottom_v,
        center_sited,
        true,
        w,
        BE,
      ))
    } else {
      None
    };

    // Freeze the effective 4:2:0 phase on the first output-bearing row — AFTER the
    // fallible scratch reserves above have succeeded, so an `AllocationFailed` row
    // stays retryable (frozen stays unset); later rows are checked against it up
    // top. Both the horizontal centered flag and the vertical `Bottom` flag are
    // pinned together.
    if need_output && frozen_chroma_centered.is_none() {
      *frozen_chroma_centered = Some(center_sited);
      *frozen_chroma_bottom_v = Some(bottom_v);
    }

    let matrix = row.matrix();
    let full_range = row.full_range();

    if let Some(luma) = luma.as_deref_mut() {
      let dst = &mut luma[one_plane_start..one_plane_end];
      for (d, &s) in dst.iter_mut().zip(row.y().iter()) {
        // Normalize BE-encoded wire bytes to host-native before the
        // luma downshift — without this, a valid BE mid-gray sample
        // (`1 << (BITS - 1)`, e.g. `0x0100` for 9-bit, `0x0200` for
        // 10-bit, `0x0800` for 12-bit) would be byte-swapped on a LE
        // host and the `>> (BITS - 8)` would write 0 instead of 128.
        let logical = if BE { u16::from_be(s) } else { u16::from_le(s) };
        *d = (logical >> (BITS - 8)) as u8;
      }
    }

    // ===== u16 RGB / RGBA path (Strategy A) — see Yuv420p10 for rationale.
    if want_rgba_u16 && !want_rgb_u16 {
      let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row =
        rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
      if let Some((u_full, v_full)) = centered {
        yuv444p14_to_rgba_u16_row_endian(
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
        yuv420p14_to_rgba_u16_row_endian(
          row.y(),
          row.u_half(),
          row.v_half(),
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
        yuv444p14_to_rgb_u16_row_endian(
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
        yuv420p14_to_rgb_u16_row_endian(
          row.y(),
          row.u_half(),
          row.v_half(),
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
    // HSV-without-RGB-or-RGBA goes through the direct `*_to_hsv_row_endian`
    // kernel (no source-width RGB scratch — the SIMD path stages a fixed
    // 8-bit RGB chunk internally). RGB or RGBA also attached keeps the
    // convert-once-then-derive path alive via `need_rgb_kernel`. Centered
    // siting (#302) routes each colour kernel through its 4:4:4 twin, fed the
    // full-width phase-0.5 chroma reconstructed above.
    let want_hsv_direct = want_hsv && !want_rgb && !want_rgba;
    let need_rgb_kernel = want_rgb || (want_hsv && want_rgba);

    if want_hsv_direct {
      let hsv = hsv.as_mut().expect("want_hsv_direct implies hsv attached");
      let (h, s, v) = hsv.hsv();
      if let Some((u_full, v_full)) = centered {
        yuv444p14_to_hsv_row_endian(
          row.y(),
          u_full,
          v_full,
          &mut h[one_plane_start..one_plane_end],
          &mut s[one_plane_start..one_plane_end],
          &mut v[one_plane_start..one_plane_end],
          w,
          matrix,
          full_range,
          use_simd,
          BE,
        );
      } else {
        yuv420p14_to_hsv_row_endian(
          row.y(),
          row.u_half(),
          row.v_half(),
          &mut h[one_plane_start..one_plane_end],
          &mut s[one_plane_start..one_plane_end],
          &mut v[one_plane_start..one_plane_end],
          w,
          matrix,
          full_range,
          use_simd,
          BE,
        );
      }
      return Ok(());
    }

    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      if let Some((u_full, v_full)) = centered {
        yuv444p14_to_rgba_row_endian(
          row.y(),
          u_full,
          v_full,
          rgba_row,
          w,
          matrix,
          full_range,
          use_simd,
          BE,
        );
      } else {
        yuv420p14_to_rgba_row_endian(
          row.y(),
          row.u_half(),
          row.v_half(),
          rgba_row,
          w,
          matrix,
          full_range,
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

    if let Some((u_full, v_full)) = centered {
      yuv444p14_to_rgb_row_endian(
        row.y(),
        u_full,
        v_full,
        rgb_row,
        w,
        matrix,
        full_range,
        use_simd,
        BE,
      );
    } else {
      yuv420p14_to_rgb_row_endian(
        row.y(),
        row.u_half(),
        row.v_half(),
        rgb_row,
        w,
        matrix,
        full_range,
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

// ---- Yuv420p16 impl ----------------------------------------------------

impl<'a, R, const BE: bool> MixedSinker<'a, Yuv420p16<BE>, R> {
  /// Attaches a packed **`u16`** RGB output buffer. Produces 16‑bit
  /// output (values in `[0, 65535]` — full `u16` range). Length is
  /// measured in `u16` **elements** (`width x height x 3`).
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

  /// Attaches a packed **8‑bit** RGBA output buffer. The 16‑bit YUV
  /// source is converted to 8‑bit RGBA via the dedicated `BITS = 16`
  /// kernel family; alpha = `0xFF` (Yuv420p16 has no alpha plane).
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

impl<R, const BE: bool> Yuv420p16Sink<BE> for MixedSinker<'_, Yuv420p16<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, Yuv420p16<BE>, R> {
  type Input<'r> = Yuv420p16Row<'r>;
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

  #[allow(clippy::too_many_lines)]
  fn process(&mut self, row: Yuv420p16Row<'_>) -> Result<(), Self::Error> {
    // Luma downshift is `>> 8` — top 8 bits of the 16-bit Y value.
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
    if row.u_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::UHalf16,
        idx,
        w / 2,
        row.u_half().len(),
      )));
    }
    if row.v_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::VHalf16,
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

    // Chroma siting (#302): drives the identity-plan horizontal chroma phase.
    // `Copy`, so read it out before the field split-borrow below.
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
      chroma_full_u16,
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
      native_420_u16,
      frozen_native_route,
      frozen_chroma_centered,
      frozen_chroma_bottom_v,
      chroma_prev_u16,
      chroma_prev_row,
      ..
    } = self;

    // Non-identity plan: native tier (bin native planes, convert once at
    // output width via 4:4:4 kernels — the dedicated 16-bit i64-chroma
    // family for BITS = 16) vs row-stage tier (convert each source row
    // then bin); `with_native(false)` forces the latter. See the Yuv420p10
    // impl for the full chroma-contract rationale.
    if let Some(plan) = plan.as_ref() {
      let matrix = row.matrix();
      let full_range = row.full_range();
      let (y, u_half, v_half) = (row.y(), row.u_half(), row.v_half());
      // RFC #238 S6a — 4:2:0 HORIZONTAL chroma siting. The centered group
      // (`Center` / `Top` / `Bottom`, [`chroma_420_center_sited_h`]) samples
      // chroma at `+0.5` luma = `+0.25` chroma-sample horizontally; the co-sited
      // / unspecified group is phase 0 (today's byte-identical decode). Siting
      // enters the chroma RECONSTRUCTION only — the averaging tier is still
      // chosen by `select_insertion_point` below — so the native fast tier folds
      // the horizontal phase into the `area_chroma_420` chroma weights while the
      // row-stage and filter tiers reconstruct full-width `u16` chroma and decode
      // 4:4:4. VERTICAL stays co-sited (`v_phase = 0`): S6a routes the horizontal
      // Top / Center phase only; `Bottom`'s vertical blend is a later stage.
      let center_sited = chroma_420_center_sited_h(chroma_location);
      // RFC #238 S6d — 4:2:0 VERTICAL `Bottom` (`v = 1`) siting on top of the
      // S6a horizontal fold. `Bottom` ([`chroma_420_bottom_sited_v`]) is a strict
      // sub-case of `center_sited` (it is `h = 0.5, v = 1`), so it rides the
      // centered reconstruction below; the binning tiers additionally fold the
      // `v = 1` triangle into `area_chroma_420`'s vertical weights, and the
      // RGB-domain reconstruction tiers box-blend the even output row's chroma
      // with the previous chroma row via the `chroma_prev_u16` lookback. `Center`
      // / `Top` keep `v_phase = 0` (co-sited vertical, byte-identical to S6a).
      let bottom_v = chroma_420_bottom_sited_v(chroma_location);
      let chroma_h_phase = if center_sited {
        YUV422P_CENTERED_H_PHASE
      } else {
        0.0
      };
      let chroma_v_phase = if bottom_v { 1.0 } else { 0.0 };
      // Whether this call carries any output — the EXACT set both tiers'
      // preflight tests (`luma || rgb || rgba || hsv || rgb_u16 || rgba_u16`).
      // The route / siting freezes only on an output-bearing row a tier ACCEPTS;
      // a no-output call consumes no stream state, so it must not freeze.
      let need_output = luma.is_some()
        || rgb.is_some()
        || rgba.is_some()
        || hsv.is_some()
        || rgb_u16.is_some()
        || rgba_u16.is_some();
      // Only the colour tiers reconstruct full-width chroma for the centered
      // decode; a luma-only centered row bins native Y unchanged (siting is a
      // chroma-only property).
      let want_color =
        rgb.is_some() || rgba.is_some() || hsv.is_some() || rgb_u16.is_some() || rgba_u16.is_some();
      // Freeze the effective 4:2:0 chroma siting on the first output-bearing row
      // (mirrors the `frozen_native_route` freeze below). This CHECK is at the
      // always-compiled choke point every tier passes through; the matching SET
      // rides each tier's accept path (never before dispatch, so a rejected row
      // leaves it unset for a corrected retry). A later row observing a different
      // phase would bin a mixture of co-sited and centered chroma, so it is
      // rejected here before any reconstruction.
      if need_output
        && let Some(frozen) = *frozen_chroma_centered
        && (frozen != center_sited || *frozen_chroma_bottom_v != Some(bottom_v))
      {
        return Err(MixedSinkerError::ChromaSitingChanged(
          ChromaSitingChanged::new(idx),
        ));
      }
      // A `Filter` plan routes to the filter resampler BEFORE the
      // native/row-stage route machinery: the native fast tier is an
      // area-specific optimization that never sees a filter plan, and the
      // per-sink plan kind is fixed at construction, so a filter sink bypasses
      // the `frozen_native_route` interaction entirely. It converts the
      // separate Y/U/V planes to a source-width u8 + native-u16 RGB row (the
      // SAME closures the row-stage tier uses) and filter-resamples them plus
      // the native Y — the filter twin of the row-stage tier. The shared tail
      // clamps every sub-16-bit colour sample AND the native Y to
      // `(1 << BITS) - 1`. Yuv420p exposes no `luma_u16`, so it is `&mut None`.
      if plan.kind().is_filter() {
        // Reject a multi-kernel (BICUBLIN) plan BEFORE the centered reserve
        // below — the delegate's first act is this same check, so hoisting it
        // keeps a rejected filter plan from reserving / reconstructing chroma
        // first (the #180 reject-before-allocation invariant). Idempotent — the
        // delegate re-runs it.
        plan.ensure_single_kernel_filter()?;
        if (center_sited || bottom_v) && want_color {
          // Centered filter: reconstruct full-width `u16` chroma, but ONLY after
          // the resample preflight (frozen-output + sequence), so an
          // out-of-sequence / rejected row is caught before the chroma
          // reservation (#180). `packed_yuv422_triple_filter_resample` re-runs
          // the idempotent preflight and owns the transactional commit. The
          // HORIZONTAL centered reconstruction is all the row-stage / filter
          // tiers need — the walker already handed this luma row its
          // (vertically co-sited) chroma row.
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
          if bottom_v {
            reserve_420_chroma_prev_u16(chroma_prev_u16, w, h)?;
          }
          // `stage = false`: DEFER the lookback advance until AFTER the fallible
          // filter commit accepts the row (below), so a rejected row leaves the
          // predecessor in place for a clean retry (#180 state-atomicity).
          let (u_full, v_full) = upsample_420_chroma_sited_u16::<BITS>(
            chroma_full_u16,
            chroma_prev_u16,
            chroma_prev_row,
            u_half,
            v_half,
            idx,
            bottom_v,
            center_sited,
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
            |scratch| deinterleave_y_high_bit_masked::<BITS, BE>(y, scratch, w),
            |scratch| {
              yuv444p16_to_rgb_row_endian(
                y, u_full, v_full, scratch, w, matrix, full_range, use_simd, BE,
              )
            },
            |scratch| {
              yuv444p16_to_rgb_u16_row_endian(
                y, u_full, v_full, scratch, w, matrix, full_range, use_simd, BE,
              )
            },
          );
          // Bottom lookback: advance only AFTER the filter resample accepts the
          // row (`r.is_ok()`), so a rejected row leaves the predecessor for a
          // clean retry — the sited reconstruction read it above but did not
          // stage. Inside the centered && want_color arm, so gate on `bottom_v`.
          if r.is_ok() && bottom_v {
            stage_420_chroma_prev_u16(chroma_prev_u16, chroma_prev_row, u_half, v_half, idx, w);
          }
          if r.is_ok() && need_output && frozen_chroma_centered.is_none() {
            *frozen_chroma_centered = Some(center_sited);
            *frozen_chroma_bottom_v = Some(bottom_v);
          }
          return r;
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
          |scratch| deinterleave_y_high_bit_masked::<BITS, BE>(y, scratch, w),
          |scratch| {
            yuv420p16_to_rgb_row_endian(
              y, u_half, v_half, scratch, w, matrix, full_range, use_simd, BE,
            )
          },
          |scratch| {
            yuv420p16_to_rgb_u16_row_endian(
              y, u_half, v_half, scratch, w, matrix, full_range, use_simd, BE,
            )
          },
        );
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
      // a true no-op, route-invisible regardless of row index. A
      // preflight-rejected (out-of-sequence / frozen) output-bearing call
      // returns Err before the SET, so it leaves `frozen_native_route`
      // untouched and a later same-or-other-route retry is not falsely
      // rejected.
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
          native_eligible: YUV420P_HIGH_BIT_NATIVE_ELIGIBLE,
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
          // RFC #238 S6a point-of-use siting invalidation: a reused sink's
          // cached join is only `reset` between frames, so a frame whose
          // `chroma_location` moved to a different horizontal phase must REBUILD
          // it (`area_chroma_420` folds the phase into the chroma weights). Drop
          // the stale-phase join ONLY on the in-sequence first row of a fresh
          // frame (`idx == 0`, `next_y() == 0`) so a mid-frame / out-of-sequence
          // row rejects against the INTACT join and a corrected retry rebuilds
          // cleanly; a luma-only join carries no chroma phase and is never
          // dropped. Move it OUT (the delegate builds the replacement into the
          // field, keeping it untouched until every pre-feed allocation
          // succeeds) and restore the intact prior-phase join on a rejected
          // rebuild so the row mutates no join state.
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
          let native_result = yuv420p16_process_native::<BITS, BE>(
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
            y,
            u_half,
            v_half,
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
                false,
              )
            },
            use_simd,
          );
          // Restore the taken stale-phase join if the delegate's rebuild was
          // rejected at any pre-feed step: it leaves the field `None` on such a
          // failure, so restoring the intact prior-phase join leaves the
          // rejected row mutating no join state. A non-stale row took nothing.
          if stale_native && native_result.is_err() {
            *native_420_u16 = prev_native;
          }
          native_result?;
          if frozen_native_route.is_none() && need_output {
            *frozen_native_route = Some(true);
          }
          // RFC #238 S6a: freeze the siting on the same accepted output row.
          if frozen_chroma_centered.is_none() && need_output {
            *frozen_chroma_centered = Some(center_sited);
            *frozen_chroma_bottom_v = Some(bottom_v);
          }
          return Ok(());
        }
        InsertionPoint::EncodedOutput => {
          // Row-stage tail. Same CHECK-before / SET-after split: dispatch, then
          // freeze the route to row-stage only when the call accepts an
          // output-bearing row (a no-output call returns Ok with `need_output`
          // false; an out-of-sequence / frozen row returns Err via `?`).
          if (center_sited || bottom_v) && want_color {
            // Centered row-stage: reconstruct full-width `u16` chroma AFTER the
            // resample preflight (frozen-output + sequence), so an
            // out-of-sequence / rejected row is caught before the chroma
            // reservation (#180). `packed_yuv422_triple_resample` re-runs the
            // idempotent preflight and owns the transactional commit. HORIZONTAL
            // reconstruction only — the walker handed this luma row its
            // (vertically co-sited) chroma row.
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
            if bottom_v {
              reserve_420_chroma_prev_u16(chroma_prev_u16, w, h)?;
            }
            // `stage = false`: DEFER the lookback advance until AFTER the fallible
            // row-stage commit accepts the row (below), so a rejected row leaves
            // the predecessor for a clean retry (#180 state-atomicity).
            let (u_full, v_full) = upsample_420_chroma_sited_u16::<BITS>(
              chroma_full_u16,
              chroma_prev_u16,
              chroma_prev_row,
              u_half,
              v_half,
              idx,
              bottom_v,
              center_sited,
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
              |scratch| deinterleave_y_high_bit_masked::<BITS, BE>(y, scratch, w),
              |scratch| {
                yuv444p16_to_rgb_row_endian(
                  y, u_full, v_full, scratch, w, matrix, full_range, use_simd, BE,
                )
              },
              |scratch| {
                yuv444p16_to_rgb_u16_row_endian(
                  y, u_full, v_full, scratch, w, matrix, full_range, use_simd, BE,
                )
              },
            )?;
            // Bottom lookback: advance only AFTER the row-stage resample accepts
            // the row (the `?` above already returned any reject), so a rejected
            // row leaves the predecessor for a clean retry — the sited
            // reconstruction read it above but did not stage.
            if bottom_v {
              stage_420_chroma_prev_u16(chroma_prev_u16, chroma_prev_row, u_half, v_half, idx, w);
            }
          } else {
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
              |scratch| deinterleave_y_high_bit_masked::<BITS, BE>(y, scratch, w),
              |scratch| {
                yuv420p16_to_rgb_row_endian(
                  y, u_half, v_half, scratch, w, matrix, full_range, use_simd, BE,
                )
              },
              |scratch| {
                yuv420p16_to_rgb_u16_row_endian(
                  y, u_half, v_half, scratch, w, matrix, full_range, use_simd, BE,
                )
              },
            )?;
          }
          if frozen_native_route.is_none() && need_output {
            *frozen_native_route = Some(false);
          }
          // RFC #238 S6a: freeze the siting on the same accepted output row.
          if frozen_chroma_centered.is_none() && need_output {
            *frozen_chroma_centered = Some(center_sited);
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

    // Resolve the full output set up front so the no-output guard below
    // short-circuits before ANY per-row offset arithmetic and the atomicity
    // preflight runs before any output row (luma included) is written.
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();
    let want_rgb_u16 = rgb_u16.is_some();
    let want_rgba_u16 = rgba_u16.is_some();
    // Whether this row produces any colour output (and so runs the centered /
    // bottom-sited chroma upsample). A bottom-sited row maintains the vertical
    // lookback even when it produces only luma — see the luma-only staging below
    // — so `want_color` gates only the colour scratches.
    let want_color = want_rgb || want_rgba || want_hsv || want_rgb_u16 || want_rgba_u16;
    // Repo-wide no-output invariant (RFC #238 S6d): a `process` call carrying NO
    // output — no colour, no luma — must run NOTHING: no per-row offset
    // arithmetic, no allocation, no state mutation (the bottom-sited vertical
    // lookback included). Returning HERE, before the `idx * w` offsets below,
    // keeps the invariant overflow-safe (a no-output call never ran an
    // attach-time `w x h` validation, so `idx * w` could overflow on a 32-bit
    // target) AND means such a row never reserves `chroma_prev_u16` nor primes
    // the lookback (so it can never make a later colour even row box-blend
    // through an invisible, never-output row). Yuv420p exposes no `luma_u16`.
    let need_output = want_color || luma.is_some();
    if !need_output {
      return Ok(());
    }

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // Chroma siting (#302): the centered horizontal sitings reconstruct chroma
    // at the phase-0.5 position; the default / co-sited path keeps the
    // byte-identical decode (the fused high-bit 4:2:0 kernels upsample chroma
    // in-register, exactly as before).
    let center_sited = chroma_420_center_sited_h(chroma_location);
    // RFC #238 S6d: `Bottom` (a strict sub-case of `center_sited`) additionally
    // box-blends the even output row's chroma with the previous chroma row via
    // the `chroma_prev_u16` lookback maintained below; `Center` / `Top` keep the
    // vertical-replicate (co-sited) decode, byte-identical to S6a.
    let bottom_v = chroma_420_bottom_sited_v(chroma_location);

    // Per-frame chroma-siting freeze (RFC #238, mirroring the resample-path guard
    // above): the first output-bearing row pins the effective 4:2:0 phase — BOTH
    // the horizontal centered flag and the vertical `Bottom` flag. A later row
    // whose siting flipped would decode a mixture of phases into ONE frame, or box-
    // blend against a STALE `chroma_prev_u16` lookback, so reject it here BEFORE
    // any scratch reserve, lookback priming, or output write. This CHECK precedes
    // the `stage_420_chroma_prev_u16` / reconstruct staging below so a rejected
    // flip leaves `chroma_prev_u16` / `chroma_prev_row` untouched (retry-atomic,
    // #180). `begin_frame`'s `reset_high_bit_yuv_streams` clears the freeze so the
    // next frame may pick either phase.
    if need_output
      && let Some(frozen) = *frozen_chroma_centered
      && (frozen != center_sited || *frozen_chroma_bottom_v != Some(bottom_v))
    {
      return Err(MixedSinkerError::ChromaSitingChanged(
        ChromaSitingChanged::new(idx),
      ));
    }

    // Atomicity preflight (#302 / #308 / #314, cf. the crate's #180 resample
    // fix): reserve EVERY fallible row scratch this identity row can touch
    // BEFORE any output row is written (the luma plane below, then the u16 / u8
    // RGB / RGBA / HSV fan-out), so an allocator refusal returns a typed
    // `AllocationFailed` leaving the output frame untouched rather than
    // partially mutated. Two scratches can grow:
    //  1. the centered-siting full-width `u16` chroma (`chroma_full_u16`),
    //     needed by ANY colour output (u8 OR u16 RGB / RGBA / HSV); and
    //  2. the u8 RGB row buffer, reached exactly when a colour decode needs an
    //     RGB row but no caller RGB buffer is borrowable — `want_hsv &&
    //     want_rgba && !want_rgb` (`rgb_row_buf_or_scratch`'s own scratch arm).
    // The later `upsample_420_chroma_center_h_u16` / `rgb_row_buf_or_scratch`
    // calls then reuse the already-sized buffers, so the default path is
    // byte-identical; only the failure-path ordering changes. The u16 RGB /
    // RGBA outputs write straight into their caller buffers (the rgb_u16 plane
    // itself stages the rgba_u16 expand) and never grow a scratch of their own.
    // Any colour output (u8 or u16 RGB / RGBA / HSV) consumes the centered
    // chroma; a luma-only row never does, so it neither reserves nor upsamples
    // it (and the reserve below is what makes the later upsample infallible).
    let need_centered_chroma = (center_sited || bottom_v) && want_color;
    if need_centered_chroma {
      reserve_420_chroma_full_u16(chroma_full_u16, w, h)?;
    }
    // Bottom-sited vertical-phase one-row chroma lookback (RFC #238 S6d): reserve
    // it on EVERY bottom-sited OUTPUT row — colour OR luma-only — because the
    // lookback is maintained so a LATER colour row can box-blend it (the
    // luma-only staging runs below, before any luma write). A no-output row
    // returned early above, so it never reaches here and never primes the
    // lookback. Reserved BEFORE any output write (the #180 preflight ordering).
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

    // Bottom-sited LUMA-ONLY row (RFC #238 S6d): the colour upsample below —
    // which normally refreshes the vertical lookback — won't run, so stage the
    // current chroma row HERE, after its reservation above and BEFORE any luma
    // write, so a later colour row in the same frame can box-blend it (a
    // luma-only-then-colour in-order sequence reconstructs the same Bottom
    // vertical phase as the all-output walk). A colour row instead stages inside
    // `upsample_420_chroma_sited_u16` after reading the previous lookback, so
    // this is skipped for it; a no-output row returned early above, so
    // `!want_color` here is a genuine luma-only row. The validity tag
    // (`chroma_prev_row`) still guards out-of-sequence / cross-frame reads.
    if bottom_v && !want_color {
      stage_420_chroma_prev_u16(
        chroma_prev_u16,
        chroma_prev_row,
        row.u_half(),
        row.v_half(),
        idx,
        w,
      );
    }

    // Centered full-width chroma, reconstructed ONCE per row from the wire-format
    // half-width U / V and reused by every colour decode (u16 and u8). Infallible
    // — the scratches were reserved above. The default left/unspecified siting
    // leaves it `None`, so the fused 4:2:0 kernels upsample chroma in-register
    // instead and the output stays byte-identical. `Center` / `Top` take the
    // plain horizontal centered phase-0.5 fold; `Bottom` (RFC #238 S6d)
    // additionally box-blends the even output row with the previous chroma row
    // (`stage = true` — the direct decode's post-reconstruction work is
    // infallible, so advancing the lookback here is safe).
    let centered = if need_centered_chroma {
      Some(upsample_420_chroma_sited_u16::<BITS>(
        chroma_full_u16,
        chroma_prev_u16,
        chroma_prev_row,
        row.u_half(),
        row.v_half(),
        idx,
        bottom_v,
        center_sited,
        true,
        w,
        BE,
      ))
    } else {
      None
    };

    // Freeze the effective 4:2:0 phase on the first output-bearing row — AFTER the
    // fallible scratch reserves above have succeeded, so an `AllocationFailed` row
    // stays retryable (frozen stays unset); later rows are checked against it up
    // top. Both the horizontal centered flag and the vertical `Bottom` flag are
    // pinned together.
    if need_output && frozen_chroma_centered.is_none() {
      *frozen_chroma_centered = Some(center_sited);
      *frozen_chroma_bottom_v = Some(bottom_v);
    }

    let matrix = row.matrix();
    let full_range = row.full_range();

    if let Some(luma) = luma.as_deref_mut() {
      let dst = &mut luma[one_plane_start..one_plane_end];
      for (d, &s) in dst.iter_mut().zip(row.y().iter()) {
        // Normalize BE-encoded wire bytes to host-native before the
        // luma downshift — without this, a valid BE mid-gray sample
        // (`1 << (BITS - 1)`, e.g. `0x0100` for 9-bit, `0x0200` for
        // 10-bit, `0x0800` for 12-bit) would be byte-swapped on a LE
        // host and the `>> (BITS - 8)` would write 0 instead of 128.
        let logical = if BE { u16::from_be(s) } else { u16::from_le(s) };
        *d = (logical >> (BITS - 8)) as u8;
      }
    }

    // ===== u16 RGB / RGBA path (Strategy A) — see Yuv420p10 for rationale.
    if want_rgba_u16 && !want_rgb_u16 {
      let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row =
        rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
      if let Some((u_full, v_full)) = centered {
        yuv444p16_to_rgba_u16_row_endian(
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
        yuv420p16_to_rgba_u16_row_endian(
          row.y(),
          row.u_half(),
          row.v_half(),
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
        yuv444p16_to_rgb_u16_row_endian(
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
        yuv420p16_to_rgb_u16_row_endian(
          row.y(),
          row.u_half(),
          row.v_half(),
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
    // HSV-without-RGB-or-RGBA goes through the direct `*_to_hsv_row_endian`
    // kernel (no source-width RGB scratch — the SIMD path stages a fixed
    // 8-bit RGB chunk internally). RGB or RGBA also attached keeps the
    // convert-once-then-derive path alive via `need_rgb_kernel`. Centered
    // siting (#302) routes each colour kernel through its 4:4:4 twin, fed the
    // full-width phase-0.5 chroma reconstructed above.
    let want_hsv_direct = want_hsv && !want_rgb && !want_rgba;
    let need_rgb_kernel = want_rgb || (want_hsv && want_rgba);

    if want_hsv_direct {
      let hsv = hsv.as_mut().expect("want_hsv_direct implies hsv attached");
      let (h, s, v) = hsv.hsv();
      if let Some((u_full, v_full)) = centered {
        yuv444p16_to_hsv_row_endian(
          row.y(),
          u_full,
          v_full,
          &mut h[one_plane_start..one_plane_end],
          &mut s[one_plane_start..one_plane_end],
          &mut v[one_plane_start..one_plane_end],
          w,
          matrix,
          full_range,
          use_simd,
          BE,
        );
      } else {
        yuv420p16_to_hsv_row_endian(
          row.y(),
          row.u_half(),
          row.v_half(),
          &mut h[one_plane_start..one_plane_end],
          &mut s[one_plane_start..one_plane_end],
          &mut v[one_plane_start..one_plane_end],
          w,
          matrix,
          full_range,
          use_simd,
          BE,
        );
      }
      return Ok(());
    }

    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      if let Some((u_full, v_full)) = centered {
        yuv444p16_to_rgba_row_endian(
          row.y(),
          u_full,
          v_full,
          rgba_row,
          w,
          matrix,
          full_range,
          use_simd,
          BE,
        );
      } else {
        yuv420p16_to_rgba_row_endian(
          row.y(),
          row.u_half(),
          row.v_half(),
          rgba_row,
          w,
          matrix,
          full_range,
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

    if let Some((u_full, v_full)) = centered {
      yuv444p16_to_rgb_row_endian(
        row.y(),
        u_full,
        v_full,
        rgb_row,
        w,
        matrix,
        full_range,
        use_simd,
        BE,
      );
    } else {
      yuv420p16_to_rgb_row_endian(
        row.y(),
        row.u_half(),
        row.v_half(),
        rgb_row,
        w,
        matrix,
        full_range,
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
