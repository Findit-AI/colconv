//! Sinker impls for source-side YUVA 4:2:0 formats — Yuva420p (8-bit),
//! Yuva420p9, Yuva420p10, Yuva420p16.
//!
//! Tranche 8b‑2a wires:
//! - **RGBA output paths (u8 + native-depth u16)** — alpha is sourced
//!   from the source alpha plane via the new
//!   `yuv_420*_to_rgba*_with_alpha_src_row` scalar kernels. SIMD lands
//!   in 8b‑2b (u8) / 8b‑2c (u16). The 8-bit `Yuva420p` has no native
//!   `u16` RGBA output.
//! - **RGB / RGB-u16 / luma / HSV alpha-drop paths** — these reuse the
//!   existing non-alpha 4:2:0 row dispatchers verbatim. The alpha plane
//!   is simply ignored; output bytes/elements match what the
//!   corresponding `Yuv420p*` source would produce given the same
//!   Y/U/V data. Without these paths `MixedSinker::with_rgb` /
//!   `with_luma` / `with_hsv` (declared on the generic impl) would
//!   silently accept a buffer and never write it.

use super::{
  ChromaSitingChanged, GeometryOverflow, InsufficientBuffer, MixedSinker, MixedSinkerError,
  NativeRouteChanged, RowIndexOutOfRange, RowShapeMismatch, RowSlice, WidthAlignment,
  check_dimensions_match, check_frozen_alpha_mode, chroma_420_bottom_sited_v,
  chroma_420_center_sited_h, deinterleave_y_high_bit_masked, packed_yuva444_filter_resample,
  packed_yuva444_resample,
  planar_8bit::{
    YUV422P_CENTERED_H_PHASE, reserve_420_chroma_full, reserve_420_chroma_prev,
    stage_420_chroma_prev, upsample_420_chroma_sited, yuva420p_process_native,
  },
  reset_high_bit_yuva_streams, rgb_row_buf_or_scratch, rgba_plane_row_slice,
  rgba_u16_plane_row_slice,
};
use crate::{
  PixelSink,
  resample::{
    AveragingDomain, InsertionContext, InsertionPoint, PlanGeometry, ResampleError,
    select_insertion_point,
  },
  row::*,
  source::*,
};

/// `Yuva420p` ships the RFC #238 Phase 5 STRAIGHT-alpha native fast tier
/// ([`yuva420p_process_native`]), so it is statically eligible to splice an
/// area downscale at the native Y / U / V / A codes — but ONLY under
/// [`AlphaMode::Straight`](super::AlphaMode::Straight). Premultiplied alpha
/// is bin-then-convert-incompatible and stays on the packed-YUVA area tail.
const YUVA420P_STRAIGHT_NATIVE_ELIGIBLE: bool = true;

/// **Fallible preflight** for the centered-siting high-bit YUVA 4:2:0 chroma
/// scratch (#302) — the `u16` twin of
/// [`reserve_420_chroma_full`](super::reserve_420_chroma_full), identical to
/// the high-bit planar `Yuv420p9` … `Yuv420p16` sinks' private helper. Grows
/// `chroma_full` to the checked `2 * width` `u16` so the later infallible
/// [`upsample_420_chroma_center_h_u16`] reuses an already-sized buffer. Split
/// out from the upsample so it can run **before any output row is written**
/// (luma included) — the crate's preflight-ordering atomicity contract (cf.
/// the #180 resample fix): an allocator refusal must leave the output frame
/// *untouched*, never partially mutated. The full-resolution source alpha
/// plane is never subsampled, so it is siting-independent and consumes no
/// scratch. `height` feeds the error payloads.
fn reserve_420_chroma_full_u16(
  chroma_full: &mut std::vec::Vec<u16>,
  width: usize,
  height: usize,
) -> Result<(), MixedSinkerError> {
  // Test-only failpoint shared with the planar / semi-planar centered paths
  // (`take()` fires the armed flag exactly once); the non-test build compiles
  // it away. Reuses the parent module's thread-local so the YUVA atomicity
  // test arms it the same way the planar suite does.
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

/// Horizontally upsamples the half-width `u16` U / V rows of a centered-siting
/// high-bit YUVA 4:2:0 source to full width into the **already-reserved**
/// `chroma_full` (#302), returning the two full-width chroma slices
/// `(u_full, v_full)`. The buffer is split `[0..width]` = U,
/// `[width..2*width]` = V; each half is filled by
/// [`chroma_upsample_2to1_center_h_u16`](crate::row::scalar::chroma_upsample_2to1_center_h_u16)
/// (the MPEG-1 / JPEG phase-0.5 reconstruction — masking each sample to the low
/// `BITS` and operating in the source's wire byte order), then fed to the
/// high-bit 4:4:4 (+ source-alpha) decode kernels with the same `big_endian`
/// flag — so the centered path reuses their SIMD backends and stays
/// bit-identical per tier. Planar YUVA stores Y / U / V / A as **low-packed**
/// logical `u16` (the `& bits_mask::<BITS>()` convention, like `Yuv420p9` …
/// `Yuv420p16`), NOT the MSB-aligned p0xx convention; `BITS` threads the
/// per-sample mask so it matches the decode kernels (a no-op at `BITS = 16`).
///
/// **Infallible**: the caller must have run [`reserve_420_chroma_full_u16`] up
/// front (every centered-siting output path does, before any output write).
fn upsample_420_chroma_center_h_u16<'s, const BITS: u32>(
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

// ---- Yuva420p impl (8-bit) ---------------------------------------------

impl<'a, R> MixedSinker<'a, Yuva420p, R> {
  /// Attaches a packed **8‑bit** RGBA output buffer. The 8‑bit YUVA
  /// source is converted to 8‑bit RGBA via the same Q15 i32 8‑bit
  /// kernel that backs [`MixedSinker<Yuv420p>::with_rgba`]; the
  /// per-pixel alpha byte is **sourced from the alpha plane** — not
  /// constant `0xFF`.
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

  /// Attaches a **`u16`** luma output buffer. The 8-bit Y plane is
  /// zero-extended to u16 (`out[x] = Y_byte as u16`). Length in u16
  /// **elements** (`width x height`).
  ///
  /// Returns `Err(InsufficientLumaU16Buffer)` if `buf.len() < width x height`,
  /// or `Err(GeometryOverflow)` on 32-bit targets.
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

impl<R> Yuva420pSink for MixedSinker<'_, Yuva420p, R> {}

impl<R> PixelSink for MixedSinker<'_, Yuva420p, R> {
  type Input<'r> = Yuva420pRow<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::WidthAlignment(WidthAlignment::odd(
        self.width,
      )));
    }
    check_dimensions_match(self.width, self.height, width, height)?;
    // New frame: restart the 4-channel u8 RGBA colour stream and the
    // native-Y luma stream (both lazily created in `process`) and re-arm the
    // alpha-mode snapshot, mirroring the alpha-aware packed-YUVA (`Vuya`)
    // sink. The luma stream kind depends on the plan: the area path bins the
    // native Y at u16 (`luma_stream_u16`), the filter path resamples the
    // contiguous native Y at u8 (`luma_filter_stream`, parity with the
    // no-alpha `Yuv420p`). The 8-bit `Yuva420p` exposes no u16 colour
    // outputs, so its u16 RGBA streams are never created.
    if let Some(stream) = self.rgba_stream.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.luma_stream_u16.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.rgba_filter_stream.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.luma_filter_stream.as_mut() {
      stream.reset();
    }
    // Restart the RFC #238 Phase 5 straight-alpha native join (Y / U / V + α
    // streams), lazily created in `process`, so the next frame reuses it.
    if let Some(native) = self.native_yuva_420.as_mut() {
      native.reset();
    }
    // Clear the per-frame native/row-stage route freeze so the next frame may
    // pick either tier (the dispatch re-freezes on its first output-bearing
    // resampled row); a flip WITHIN a frame stays rejected. Mirrors the
    // high-bit families' `reset_high_bit_yuv_streams`.
    self.frozen_native_route = None;
    self.resample_outputs = None;
    // Clear the RFC #238 frozen 4:2:0 chroma siting so the next frame may pick
    // either phase; a mid-frame flip stays rejected. (The siting invalidation of
    // the cached native join is done at point-of-use in `process`, since
    // `chroma_location` may change AFTER `begin_frame` but before row 0.)
    self.frozen_chroma_centered = None;
    // RFC #238 S4-D companion: clear the frozen vertical phase so the next frame
    // may pick Center or Bottom, and invalidate the bottom-sited vertical-phase
    // chroma lookback (#302) so frame N+1's first even row can never box-blend
    // frame N's last chroma row (only the validity tag is cleared; the buffer
    // bytes are re-overwritten before any trusted read). A mid-frame Center ⇆
    // Bottom flip stays rejected.
    self.frozen_chroma_bottom_v = None;
    self.chroma_prev_row = None;
    self.frozen_alpha_mode = Some(self.alpha_mode);
    Ok(())
  }

  fn process(&mut self, row: Yuva420pRow<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;
    // Chroma siting (#302): drives the identity-plan horizontal chroma phase
    // for the Y/U/V colour decode. The full-resolution source alpha plane is
    // siting-independent (it is never subsampled), so it passes through
    // unchanged on every path. `Copy`, so read it out before the field
    // split-borrow below.
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
    if row.a().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::AFull,
        idx,
        w,
        row.a().len(),
      )));
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange(
        RowIndexOutOfRange::new(idx, self.height),
      ));
    }

    // Non-identity plan: `Yuva420p` is 8-bit planar 4:2:0 YUV **with a
    // real full-resolution source alpha plane**. Route through the
    // packed-YUVA tail at `SRC_BITS = 8`: the u8 colour stream resamples the
    // converted u8 RGBA row (`yuva420p_to_rgba_row` — chroma upsampled per
    // 4:2:0, real source α from the alpha plane, NOT forced opaque) so
    // resampled alpha is a real mean and, under `AlphaMode::Premultiplied`,
    // colour is binned premultiplied; the native-Y luma stream resamples the
    // (zero-extended) Y plane directly so luma / luma_u16 are the
    // downscaled native Y, alpha- and range-independent (never derived from
    // the colour). The 8-bit `Yuva420p` exposes no u16 colour outputs, so
    // the tail's u16 colour resampling is never active (`rgb_u16` /
    // `rgba_u16` stay `None`) and its `convert_rgba_u16` closure is never
    // invoked.
    //
    // The span kind picks the engine: `Area` bins (the alpha-aware tail —
    // premultiplied colour is binned premultiplied then un-premultiplied);
    // `Filter` runs the signed-coefficient filter on the same converted RGBA
    // (PIL RGBA semantics — all four channels filtered independently,
    // straight alpha only). Premultiplied alpha has no filter analogue (the
    // engine cannot un-premultiply), so a premultiplied `Filter` plan is
    // routed to the area tail, which surfaces the typed `UnsupportedFilter`
    // rather than emitting straight-filtered premultiplied colour.
    if self.plan.is_some() {
      let alpha_mode = self.alpha_mode;
      let matrix = row.matrix();
      let full_range = row.full_range();
      let y = row.y();
      let u_half = row.u_half();
      let v_half = row.v_half();
      let a = row.a();
      let Self {
        rgb,
        rgba,
        rgb_u16,
        rgba_u16,
        luma,
        luma_u16,
        hsv,
        rgba_scratch,
        rgb_scratch,
        rgba_scratch_u16,
        rgba_color_scratch_u16,
        luma_scratch_u16,
        chroma_full,
        plan,
        rgba_stream,
        rgba_stream_u16,
        luma_stream_u16,
        rgba_filter_stream,
        rgba_filter_stream_u16,
        luma_filter_stream,
        luma_filter_stream_u16,
        resample_outputs,
        frozen_alpha_mode,
        native,
        native_yuva_420,
        frozen_native_route,
        frozen_chroma_centered,
        frozen_chroma_bottom_v,
        chroma_prev,
        chroma_prev_row,
        ..
      } = self;
      let plan = plan.as_ref().expect("plan.is_some() checked above");
      check_frozen_alpha_mode(*frozen_alpha_mode, alpha_mode, idx)?;
      // RFC #238 S3c — 4:2:0 horizontal chroma siting for the alpha-bearing
      // source. The centered group (`Center` / `Top` / `Bottom`,
      // [`chroma_420_center_sited_h`]) samples chroma at `+0.5` luma horizontally;
      // the co-sited / unspecified group is phase 0 (today's decode,
      // byte-identical). Siting enters the chroma RECONSTRUCTION only: the native
      // fast tier folds the phase into the chroma area weights (`area_chroma_420`),
      // while the row-stage / filter converters reconstruct full-width chroma. The
      // full-resolution α plane is NEVER subsampled, so it is siting-independent
      // and passes through UNCHANGED on every tier. RFC #238 S4-D adds the
      // VERTICAL `Bottom` (`v = 1`) fold on top ([`chroma_420_bottom_sited_v`], a
      // strict sub-case of `center_sited` — `Bottom` is `h = 0.5, v = 1`): the
      // binning tiers (native / HSV-via-packed) fold the `v = 1` triangle into
      // `area_chroma_420`'s vertical weights, and the RGB-domain reconstruction
      // tiers box-blend the even output row's chroma with the previous chroma row
      // via the `chroma_prev` lookback. `Center` / `Top` keep `v_phase = 0`
      // (co-sited vertical, byte-identical to S3c). α is untouched on every path.
      let center_sited = chroma_420_center_sited_h(chroma_location);
      let bottom_v = chroma_420_bottom_sited_v(chroma_location);
      let chroma_h_phase = if center_sited {
        YUV422P_CENTERED_H_PHASE
      } else {
        0.0
      };
      let chroma_v_phase = if bottom_v { 1.0 } else { 0.0 };
      // Whether this call carries any output — the EXACT set both tiers' no-output
      // short-circuit tests. Gates BOTH the siting freeze here and the
      // native/row-stage route freeze below, so a no-output call (which consumes
      // no stream state) freezes neither.
      let need_output =
        rgb.is_some() || rgba.is_some() || hsv.is_some() || luma.is_some() || luma_u16.is_some();
      // Colour outputs drive the centered chroma reconstruction; a luma-only
      // centered row bins the native Y (siting-independent) and never
      // reconstructs chroma.
      let want_color = rgb.is_some() || rgba.is_some() || hsv.is_some();
      // Freeze the effective 4:2:0 chroma siting on the first output-bearing row,
      // mirroring the no-alpha `Yuv420p` arm. This always-compiled CHECK sits at
      // the choke point every tier passes through; the matching SET rides each
      // tier's accept-time freeze below (never before dispatch, so a rejected row
      // leaves it unset for a corrected retry). A later row observing a different
      // phase — in sequence or not — would bin a mixture of co-sited and centered
      // chroma, so it is rejected here before any reconstruction. RFC #238 S4-D
      // also freezes the VERTICAL phase: `Center` and `Bottom` are both
      // horizontally centered, so a mid-frame Center ⇆ Bottom flip is invisible to
      // `center_sited` alone and is caught by the `frozen_chroma_bottom_v`
      // companion.
      if need_output
        && let Some(frozen) = *frozen_chroma_centered
        && (frozen != center_sited || *frozen_chroma_bottom_v != Some(bottom_v))
      {
        return Err(MixedSinkerError::ChromaSitingChanged(
          ChromaSitingChanged::new(idx),
        ));
      }
      return match plan.kind() {
        crate::resample::SpanKind::Area => {
          // RFC #238 Phase 5 splice-stage selection for the area downscale.
          // STRAIGHT alpha is the #235 native resolution: bin Y / U / V / A
          // codes, convert once, attach the binned α → straight RGBA. The
          // framework selector reproduces the native-vs-row-stage choice — it
          // returns `NativeCodes` exactly when this format is straight-native
          // eligible, the sink enabled the native tier, AND the alpha mode is
          // straight (a premultiplied frame must convert-at-source then
          // premultiply then bin then un-premultiply late, so it is NOT
          // native-eligible and keeps the packed-YUVA area tail BYTE-IDENTICAL).
          // `area_plan` is always true on this arm (the filter plan is handled
          // by the arms below).
          let insertion = select_insertion_point(
            AveragingDomain::Encoded,
            InsertionContext {
              native_eligible: YUVA420P_STRAIGHT_NATIVE_ELIGIBLE && !alpha_mode.is_premultiplied(),
              with_native: *native,
              area_plan: true,
            },
          );
          let take_native = matches!(insertion, InsertionPoint::NativeCodes);
          // Reject a mid-frame native/row-stage route flip (a caller toggling
          // `set_native` between rows — native row 0, row-stage row 1)
          // BEFORE either tier's dispatch: the two tiers carry independent,
          // in-order, once-only stream state, so splitting a frame across them
          // would yield a mixed/partial frame rather than a deterministic
          // rejection. CHECKED here and frozen below (the SET), both gated on
          // `need_output`, so a no-output call is route-invisible. A
          // preflight-rejected output-bearing call returns Err before the SET,
          // leaving `frozen_native_route` untouched so a later retry on the
          // same (or other) route is not falsely rejected.
          if need_output
            && let Some(frozen) = *frozen_native_route
            && frozen != take_native
          {
            return Err(MixedSinkerError::NativeRouteChanged(
              NativeRouteChanged::new(idx),
            ));
          }
          match insertion {
            // Straight-alpha native tier. The AlphaMode freeze
            // (`check_frozen_alpha_mode` above) plus the inherited native
            // preflight (out-of-sequence / frozen-output / alloc, all before
            // any feed) keep the call atomic. Dispatch first; freeze the route
            // to native ONLY after the call returns Ok on an output-bearing
            // row — a no-output call returns Ok(()) with `need_output` false
            // (no freeze); an out-of-sequence / frozen / alloc-refused row
            // returns Err via `?` (no freeze), so only an accepted
            // output-bearing row commits the route (the Phase 2 set-after-accept
            // lesson — never freeze before the row is accepted).
            InsertionPoint::NativeCodes => {
              // RFC #238 S3c point-of-use siting invalidation, mirroring the
              // no-alpha `Yuv420p` native arm: `chroma_location` can change at ANY
              // point before this row (including AFTER `begin_frame`, before row
              // 0), so re-check the cached join HERE and drop it when its folded
              // chroma plan was built for a different phase; `yuva420p_process_native`
              // then rebuilds it with the current siting. Retry-atomic: drop ONLY on
              // the IN-SEQUENCE fresh-frame first row (`idx == 0`, `next_y() == 0`);
              // an out-of-sequence first row is left for the delegate to reject
              // against the INTACT join. A luma-only join carries no chroma plan
              // (siting-independent), a no-output sink built no join. Transactional:
              // move the stale join OUT, let the delegate build the replacement into
              // `native_yuva_420`, and restore the intact prior-phase join on a
              // rejected rebuild so the REJECTED row mutates nothing. The α stream is
              // siting-independent, rebuilt in lockstep and carrying no phase.
              let stale_native = idx == 0
                && native_yuva_420.as_ref().is_some_and(|join| {
                  join.has_chroma()
                    && (join.chroma_centered() != center_sited || join.chroma_bottom() != bottom_v)
                    && join.next_y() == 0
                });
              let prev_native = if stale_native {
                native_yuva_420.take()
              } else {
                None
              };
              let native_result = yuva420p_process_native(
                plan,
                native_yuva_420,
                resample_outputs,
                rgb,
                rgba,
                luma,
                luma_u16,
                hsv,
                rgb_scratch,
                y,
                u_half,
                v_half,
                a,
                matrix,
                full_range,
                idx,
                w,
                h,
                chroma_h_phase,
                chroma_v_phase,
                use_simd,
              );
              if stale_native && native_result.is_err() {
                *native_yuva_420 = prev_native;
              }
              native_result?;
              if frozen_native_route.is_none() && need_output {
                *frozen_native_route = Some(true);
              }
              // RFC #238 S3c: freeze the horizontal siting on the same accepted
              // output row; S4-D freezes the vertical `Bottom` phase alongside it.
              if frozen_chroma_centered.is_none() && need_output {
                *frozen_chroma_centered = Some(center_sited);
                *frozen_chroma_bottom_v = Some(bottom_v);
              }
              Ok(())
            }
            // Row-stage tier (the `with_native(false)` path, and EVERY
            // premultiplied frame): convert each source row to RGBA then bin —
            // the packed-YUVA area tail. Premultiplied colour is binned
            // premultiplied then un-premultiplied, byte-identical to today.
            // Same CHECK-before / SET-after split: freeze to row-stage only
            // when the call accepts an output-bearing row.
            InsertionPoint::EncodedOutput => {
              if center_sited && want_color {
                // Centered row-stage: reconstruct full-width chroma and decode
                // 4:4:4 — but ONLY after the resample preflight (frozen-output +
                // sequence), so an out-of-sequence / rejected row is caught before
                // the chroma reservation (#180). `packed_yuva444_resample` re-runs
                // the idempotent preflight. Gated by `want_color` — a luma-only
                // centered row bins the native Y and never reconstructs chroma, so
                // it stays on the co-sited arm below. The α plane is
                // full-resolution and siting-independent: it flows into the 4:4:4
                // converter UNCHANGED (never chroma-upsampled).
                let expected = rgba_stream.as_ref().map_or(0, |s| s.next_y());
                if let core::ops::ControlFlow::Break(()) =
                  super::planar_resample::resample_preflight_check_only(
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
                  )?
                {
                  return Ok(());
                }
                reserve_420_chroma_full(chroma_full, w, h)?;
                // RFC #238 S4-D: the `Bottom` even output row box-blends its
                // chroma with the previous chroma row, so reserve the lookback
                // up front — AFTER the preflight, BEFORE the commit (#180).
                if bottom_v {
                  reserve_420_chroma_prev(chroma_prev, w, h)?;
                }
                let (u_full, v_full) = upsample_420_chroma_sited(
                  chroma_full,
                  chroma_prev,
                  chroma_prev_row,
                  u_half,
                  v_half,
                  idx,
                  bottom_v,
                  false,
                  w,
                );
                packed_yuva444_resample::<8>(
                  rgba_stream,
                  rgba_stream_u16,
                  luma_stream_u16,
                  resample_outputs,
                  rgb,
                  rgba,
                  rgb_u16,
                  rgba_u16,
                  luma,
                  luma_u16,
                  hsv,
                  rgba_scratch,
                  rgb_scratch,
                  rgba_scratch_u16,
                  rgba_color_scratch_u16,
                  luma_scratch_u16,
                  w,
                  plan,
                  idx,
                  use_simd,
                  alpha_mode,
                  |dst| {
                    yuva444p_to_rgba_row(y, u_full, v_full, a, dst, w, matrix, full_range, use_simd)
                  },
                  // `Yuva420p` has no u16 colour outputs, so this is never called.
                  |_dst: &mut [u16]| {},
                  |dst| {
                    for (d, &s) in dst.iter_mut().zip(y) {
                      *d = s as u16;
                    }
                  },
                )?;
                // RFC #238 S4-D bottom lookback: advance ONLY after the row-stage
                // packed resample accepts the row (its `?` above already returned
                // any reject), so a rejected row leaves the predecessor for a clean
                // retry — the sited reconstruction read the lookback but did not
                // stage (deferred, #180).
                if bottom_v {
                  stage_420_chroma_prev(chroma_prev, chroma_prev_row, u_half, v_half, idx, w);
                }
              } else {
                packed_yuva444_resample::<8>(
                  rgba_stream,
                  rgba_stream_u16,
                  luma_stream_u16,
                  resample_outputs,
                  rgb,
                  rgba,
                  rgb_u16,
                  rgba_u16,
                  luma,
                  luma_u16,
                  hsv,
                  rgba_scratch,
                  rgb_scratch,
                  rgba_scratch_u16,
                  rgba_color_scratch_u16,
                  luma_scratch_u16,
                  w,
                  plan,
                  idx,
                  use_simd,
                  alpha_mode,
                  |dst| {
                    yuva420p_to_rgba_row(y, u_half, v_half, a, dst, w, matrix, full_range, use_simd)
                  },
                  // `Yuva420p` has no u16 colour outputs, so this is never called.
                  |_dst: &mut [u16]| {},
                  |dst| {
                    for (d, &s) in dst.iter_mut().zip(y) {
                      *d = s as u16;
                    }
                  },
                )?;
              }
              if frozen_native_route.is_none() && need_output {
                *frozen_native_route = Some(false);
              }
              // RFC #238 S3c: freeze the horizontal siting on the same accepted
              // output row; S4-D freezes the vertical `Bottom` phase alongside it.
              if frozen_chroma_centered.is_none() && need_output {
                *frozen_chroma_centered = Some(center_sited);
                *frozen_chroma_bottom_v = Some(bottom_v);
              }
              Ok(())
            }
            // The encoded domain only resolves to native-codes / encoded-output;
            // the linear-light splice is reached via the Linear averaging domain,
            // which this 4:2:0 YUVA sink does not expose.
            InsertionPoint::LinearLight => {
              unreachable!("Yuva420p area downscale never selects the linear-light splice")
            }
          }
        }
        crate::resample::SpanKind::Filter if alpha_mode.is_premultiplied() => {
          // Premultiplied + filter has no analogue: route to the area tail
          // with the filter plan so it returns the typed `UnsupportedFilter`.
          packed_yuva444_resample::<8>(
            rgba_stream,
            rgba_stream_u16,
            luma_stream_u16,
            resample_outputs,
            rgb,
            rgba,
            rgb_u16,
            rgba_u16,
            luma,
            luma_u16,
            hsv,
            rgba_scratch,
            rgb_scratch,
            rgba_scratch_u16,
            rgba_color_scratch_u16,
            luma_scratch_u16,
            w,
            plan,
            idx,
            use_simd,
            alpha_mode,
            |dst| yuva420p_to_rgba_row(y, u_half, v_half, a, dst, w, matrix, full_range, use_simd),
            |_dst: &mut [u16]| {},
            |dst| {
              for (d, &s) in dst.iter_mut().zip(y) {
                *d = s as u16;
              }
            },
          )
        }
        crate::resample::SpanKind::Filter => {
          // Reject a multi-kernel (BICUBLIN) filter plan BEFORE the centered
          // preflight/reserve below: `packed_yuva444_filter_resample`'s FIRST act
          // is this same check, so without hoisting it a BICUBLIN plan would
          // freeze the output set and reserve/reconstruct chroma before being
          // rejected as `UnsupportedFilter` (the #180 reject-before-allocation
          // invariant). Idempotent — the delegate re-runs it.
          plan.ensure_single_kernel_filter()?;
          let r = if center_sited && want_color {
            // Centered filter: reconstruct full-width chroma and decode 4:4:4 —
            // but ONLY after the filter preflight (frozen-output + sequence), so
            // an out-of-sequence / rejected row is caught before the chroma
            // reservation (#180). `packed_yuva444_filter_resample` re-runs the
            // idempotent preflight. The α plane is full-resolution and
            // siting-independent — it flows into the 4:4:4 converter UNCHANGED.
            let expected = rgba_filter_stream.as_ref().map_or(0, |s| s.next_y());
            if let core::ops::ControlFlow::Break(()) =
              super::planar_resample::resample_preflight_check_only(
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
              )?
            {
              return Ok(());
            }
            reserve_420_chroma_full(chroma_full, w, h)?;
            // RFC #238 S4-D: reserve the bottom-sited lookback the even output row
            // box-blends against — AFTER the preflight, BEFORE the commit (#180).
            if bottom_v {
              reserve_420_chroma_prev(chroma_prev, w, h)?;
            }
            let (u_full, v_full) = upsample_420_chroma_sited(
              chroma_full,
              chroma_prev,
              chroma_prev_row,
              u_half,
              v_half,
              idx,
              bottom_v,
              false,
              w,
            );
            packed_yuva444_filter_resample::<8, true, false>(
              rgba_filter_stream,
              rgba_filter_stream_u16,
              luma_filter_stream,
              luma_filter_stream_u16,
              resample_outputs,
              rgb,
              rgba,
              rgb_u16,
              rgba_u16,
              luma,
              luma_u16,
              hsv,
              rgba_scratch,
              rgb_scratch,
              rgba_scratch_u16,
              rgba_color_scratch_u16,
              luma_scratch_u16,
              w,
              plan,
              idx,
              use_simd,
              y,
              None,
              |dst| {
                yuva444p_to_rgba_row(y, u_full, v_full, a, dst, w, matrix, full_range, use_simd)
              },
              |_dst: &mut [u16]| {},
              |_dst: &mut [u16]| {},
              |_dst: &mut [u8]| {},
            )
          } else {
            packed_yuva444_filter_resample::<8, true, false>(
              rgba_filter_stream,
              rgba_filter_stream_u16,
              luma_filter_stream,
              luma_filter_stream_u16,
              resample_outputs,
              rgb,
              rgba,
              rgb_u16,
              rgba_u16,
              luma,
              luma_u16,
              hsv,
              rgba_scratch,
              rgb_scratch,
              rgba_scratch_u16,
              rgba_color_scratch_u16,
              luma_scratch_u16,
              w,
              plan,
              idx,
              use_simd,
              // 8-bit native-Y luma rides the u8 stream (parity with `Yuv420p`):
              // the contiguous Y plane is fed directly, so no de-interleave scratch.
              y,
              None,
              |dst| {
                yuva420p_to_rgba_row(y, u_half, v_half, a, dst, w, matrix, full_range, use_simd)
              },
              // `Yuva420p` has no u16 colour outputs, so this closure is never called.
              |_dst: &mut [u16]| {},
              // u8-luma path: the u16 luma stream is detached, so this is never
              // called.
              |_dst: &mut [u16]| {},
              // Contiguous Y plane fed directly, so this u8 de-interleave is unused.
              |_dst: &mut [u8]| {},
            )
          };
          // RFC #238 S4-D bottom lookback: advance ONLY after the filter resample
          // accepts the row, so a rejected row leaves the predecessor for a clean
          // retry — the sited reconstruction read it but did not stage (deferred,
          // #180). Gated on `want_color`: a luma-only Bottom row took the co-sited
          // else arm (no reconstruction, no lookback read/stage).
          if r.is_ok() && bottom_v && want_color {
            stage_420_chroma_prev(chroma_prev, chroma_prev_row, u_half, v_half, idx, w);
          }
          // RFC #238 S3c: freeze the horizontal siting on the same accepted output
          // row (a rejected row leaves it unset for a corrected retry); S4-D
          // freezes the vertical `Bottom` phase alongside it. The filter plan kind
          // is fixed at construction, so there is no native/row-stage route here.
          if r.is_ok() && need_output && frozen_chroma_centered.is_none() {
            *frozen_chroma_centered = Some(center_sited);
            *frozen_chroma_bottom_v = Some(bottom_v);
          }
          r
        }
      };
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
      ..
    } = self;

    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();
    // A colour output that drives the centered / bottom chroma reconstruction;
    // the full-resolution α plane is siting-independent and never gates it.
    let want_color = want_rgb || want_rgba || want_hsv;
    // A row with no attached output is a no-op — return BEFORE the bottom-sited
    // lookback is primed (reserved / staged below), so a genuinely no-output row
    // never advances the vertical-phase lookback (mirrors the no-alpha `Yuv420p`
    // direct decode); it also guarantees ≥ 1 output was attached, so the
    // single-plane row ranges below cannot overflow.
    let need_output = want_color || luma.is_some() || luma_u16.is_some();
    if !need_output {
      return Ok(());
    }
    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // Chroma siting (#302): the centered horizontal sitings reconstruct chroma
    // at the phase-0.5 position then feed the 4:4:4 (+ source-alpha) kernels;
    // the default / co-sited path keeps the byte-identical fused 4:2:0 decode.
    // RFC #238 S4-D: `Bottom` (`v = 1`) additionally box-blends the even output
    // row's chroma with the previous chroma row via the `chroma_prev` lookback.
    let center_sited = chroma_420_center_sited_h(chroma_location);
    let bottom_v = chroma_420_bottom_sited_v(chroma_location);
    // HSV-without-RGB-or-RGBA goes through the direct `yuv_420_to_hsv_row`
    // kernel (no source-width RGB scratch). HSV is colour-only — the
    // source alpha plane is dropped — so a YUVA HSV is byte-identical to
    // the no-alpha `Yuv420p` HSV on the same Y/U/V; the alpha-drop RGB
    // path already reuses `yuv_420_to_rgb_row`, so HSV reuses the matching
    // planar `yuv_420_to_hsv_row`. RGB or RGBA also attached keeps the
    // convert-once-then-derive path alive via `need_rgb_kernel` (the RGB
    // kernel runs anyway, so HSV derives off that buffer; an RGBA sink
    // still needs the alpha plane, so it stays on the RGB path).
    let want_hsv_direct = want_hsv && !want_rgb && !want_rgba;
    let need_rgb_kernel = want_rgb || (want_hsv && want_rgba);

    // Atomicity preflight (#302, cf. the crate's #180 resample fix): reserve
    // EVERY fallible row scratch this identity row needs BEFORE any output row
    // (luma / luma_u16 included) is written, so an allocator refusal returns a
    // typed error leaving the output frame untouched, never partially mutated.
    // Three scratches can grow:
    //  1. the centered-siting full-width chroma (`chroma_full`), only on a colour
    //     decode (RGB / RGBA / HSV — the alpha plane never subsamples, so it is
    //     siting-independent);
    //  2. (RFC #238 S4-D) the bottom-sited (`Bottom`, `v = 1`) one-row chroma
    //     lookback (`chroma_prev`), on EVERY Bottom OUTPUT row — colour OR
    //     luma-only — because it is maintained so a LATER colour row (after a
    //     luma-only-then-colour attach) can box-blend it; the luma-only staging
    //     runs below, before any luma write. A no-output row returned early above
    //     and never primes it; and
    //  3. the u8 RGB row buffer, reached exactly when a colour decode needs an
    //     RGB row but no caller RGB buffer is borrowable.
    // The later `upsample_420_chroma_sited` reuses the already-sized buffers.
    if center_sited && want_color {
      reserve_420_chroma_full(chroma_full, w, h)?;
    }
    if bottom_v {
      reserve_420_chroma_prev(chroma_prev, w, h)?;
    }

    // Acquire the u8 RGB row buffer up front — before any caller-output
    // write below — so an allocator refusal in the HSV-only scratch path
    // returns a recoverable error rather than leaving a partially-written
    // caller buffer (luma / luma_u16). `rgb_row_buf_or_scratch` is the
    // only other fallible allocation on this direct path.
    let rgb_row = if need_rgb_kernel {
      Some(rgb_row_buf_or_scratch(
        rgb.as_deref_mut(),
        rgb_scratch,
        one_plane_start,
        one_plane_end,
        w,
        h,
      )?)
    } else {
      None
    };

    // Centered full-width chroma (phase-0.5), reconstructed ONCE per row from the
    // wire half-width U / V and reused by every colour decode below. For `Bottom`
    // (`v = 1`, RFC #238 S4-D) the even output row additionally box-blends with the
    // previous chroma row (the `chroma_prev` lookback), and this call stages that
    // lookback for the next pair (`stage = true`: the direct decode's remaining
    // work is effectively infallible, so advancing here is safe). Infallible — both
    // scratches were reserved above. The default left/unspecified siting leaves it
    // `None`, so the fused 4:2:0 kernels upsample chroma in-register and the output
    // stays byte-identical.
    let centered = if center_sited && want_color {
      Some(upsample_420_chroma_sited(
        chroma_full,
        chroma_prev,
        chroma_prev_row,
        row.u_half(),
        row.v_half(),
        idx,
        bottom_v,
        true,
        w,
      ))
    } else {
      None
    };

    // Bottom-sited LUMA-ONLY row (RFC #238 S4-D): the colour upsample above —
    // which refreshes the lookback via `stage = true` — did not run (`centered`
    // is `None`), so stage the current chroma row HERE, after its preflight
    // reservation and BEFORE any luma write, so a LATER colour row in the same
    // frame box-blends it (a luma-only-then-colour in-order walk then
    // reconstructs the same Bottom vertical phase as the all-output walk). A
    // colour row instead staged inside `upsample_420_chroma_sited`, so gate on
    // `!want_color`; the `chroma_prev_row` tag still guards out-of-sequence /
    // cross-frame reads (→ clamp, never stale).
    if bottom_v && !want_color {
      stage_420_chroma_prev(
        chroma_prev,
        chroma_prev_row,
        row.u_half(),
        row.v_half(),
        idx,
        w,
      );
    }

    // Luma — copy the Y plane verbatim (8-bit YUVA's Y is already u8).
    if let Some(luma) = luma.as_deref_mut() {
      luma[one_plane_start..one_plane_end].copy_from_slice(&row.y()[..w]);
    }

    // Luma u16 — zero-extend the Y plane (`out[x] = Y_byte as u16`).
    if let Some(luma_u16) = luma_u16.as_deref_mut() {
      for (d, &s) in luma_u16[one_plane_start..one_plane_end]
        .iter_mut()
        .zip(&row.y()[..w])
      {
        *d = s as u16;
      }
    }

    // HSV-only (no RGB / RGBA): convert the source Y/U/V straight to HSV
    // via the alpha-drop kernel — no source-width RGB scratch. Centered siting
    // (#302) routes through the 4:4:4 twin fed the full-width phase-0.5 chroma.
    if want_hsv_direct {
      let hsv = hsv.as_mut().expect("want_hsv_direct implies hsv attached");
      let (h, s, v) = hsv.hsv();
      if let Some((u_full, v_full)) = centered {
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

    // ---- u8 RGB / RGBA / HSV path ----------------------------------
    // `need_rgb_kernel` / `rgb_row` were computed and (when needed)
    // allocated at the top, before any caller-output write.
    //
    // Direct-RGBA-only fast path routes through the alpha-source-aware
    // dispatcher (real source α, NOT forced opaque). Centered siting feeds the
    // 4:4:4 + source-alpha kernel the full-width phase-0.5 chroma; the alpha
    // plane is identical on both paths.
    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      if let Some((u_full, v_full)) = centered {
        yuva444p_to_rgba_row(
          row.y(),
          u_full,
          v_full,
          row.a(),
          rgba_row,
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
        );
      } else {
        yuva420p_to_rgba_row(
          row.y(),
          row.u_half(),
          row.v_half(),
          row.a(),
          rgba_row,
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
        );
      }
      return Ok(());
    }

    // RGB kernel — alpha-drop reuses the planar dispatcher verbatim (the 4:4:4
    // twin for centered siting).
    let Some(rgb_row) = rgb_row else {
      return Ok(());
    };
    if let Some((u_full, v_full)) = centered {
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

    // Both rgb and rgba attached (combo case): Strategy A+ — expand the
    // already-computed rgb_row → rgba_row (fills α = 0xFF), then
    // overwrite α slot from the source alpha plane. Avoids a second
    // chroma kernel — so the centered RGB row above is reused as-is, and only
    // the real source alpha is layered on, identical to the default path.
    if want_rgba {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
      crate::row::alpha_extract::copy_alpha_plane_u8(row.a(), rgba_row, w, use_simd);
    }

    Ok(())
  }
}

// ---- Yuva420p9 impl ---------------------------------------------------

impl<'a, R, const BE: bool> MixedSinker<'a, Yuva420p9<BE>, R> {
  /// Attaches a packed **8-bit** RGBA output buffer. Source-derived
  /// alpha (depth-converted via `>> 1`).
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

  /// Attaches a packed **`u16`** RGBA output buffer (9‑bit
  /// low‑packed, `[0, 511]`). Alpha is sourced at native depth.
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

  /// Attaches a packed **`u16`** RGB output buffer (alpha-drop).
  /// Output is identical to [`MixedSinker<Yuv420p9>::with_rgb_u16`].
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

  /// Attaches a **`u16`** luma output buffer. Luma is the **native Y**
  /// (the binned Y plane at native depth under a non-identity plan; the
  /// Y plane verbatim otherwise). Length in u16 **elements**
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

impl<R, const BE: bool> Yuva420p9Sink<BE> for MixedSinker<'_, Yuva420p9<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, Yuva420p9<BE>, R> {
  type Input<'r> = Yuva420p9Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::WidthAlignment(WidthAlignment::odd(
        self.width,
      )));
    }
    check_dimensions_match(self.width, self.height, width, height)?;
    reset_high_bit_yuva_streams(self);
    Ok(())
  }

  fn process(&mut self, row: Yuva420p9Row<'_>) -> Result<(), Self::Error> {
    if self.plan.is_some() {
      return yuva420p_high_bit_resample::<9, BE>(
        self,
        row.row(),
        row.y(),
        row.u_half(),
        row.v_half(),
        row.a(),
        row.matrix(),
        row.full_range(),
        RowSlice::Y9,
        RowSlice::UHalf9,
        RowSlice::VHalf9,
        RowSlice::AFull9,
        yuva420p9_to_rgba_row_endian,
        yuva420p9_to_rgba_u16_row_endian,
        // Centered-siting (#302, RFC #238 S6c) 4:4:4 twins fed the horizontally
        // reconstructed full-width chroma; α passed through unchanged.
        yuva444p9_to_rgba_row_endian,
        yuva444p9_to_rgba_u16_row_endian,
      );
    }
    yuva420p_high_bit_process::<9, BE, _, _>(
      self,
      row.row(),
      row.y(),
      row.u_half(),
      row.v_half(),
      row.a(),
      row.matrix(),
      row.full_range(),
      RowSlice::Y9,
      RowSlice::UHalf9,
      RowSlice::VHalf9,
      RowSlice::AFull9,
      yuv420p9_to_rgb_row_endian,
      yuv420p9_to_rgb_u16_row_endian,
      yuva420p9_to_rgba_row_endian,
      yuv420p9_to_hsv_row_endian,
      yuva420p9_to_rgba_u16_row_endian,
      // Centered-siting (#302) 4:4:4 twins.
      yuv444p9_to_rgb_row_endian,
      yuv444p9_to_rgb_u16_row_endian,
      yuva444p9_to_rgba_row_endian,
      yuv444p9_to_hsv_row_endian,
      yuva444p9_to_rgba_u16_row_endian,
    )
  }
}

// ---- Yuva420p10 impl --------------------------------------------------

impl<'a, R, const BE: bool> MixedSinker<'a, Yuva420p10<BE>, R> {
  /// Attaches a packed **8-bit** RGBA output buffer. Source-derived
  /// alpha (depth-converted via `>> 2`).
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

  /// Attaches a packed **`u16`** RGBA output buffer (10‑bit
  /// low‑packed, `[0, 1023]`). Alpha is sourced at native depth.
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

  /// Attaches a packed **`u16`** RGB output buffer (alpha-drop).
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

  /// Attaches a **`u16`** luma output buffer. Luma is the **native Y**
  /// (the binned Y plane at native depth under a non-identity plan; the
  /// Y plane verbatim otherwise). Length in u16 **elements**
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

impl<R, const BE: bool> Yuva420p10Sink<BE> for MixedSinker<'_, Yuva420p10<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, Yuva420p10<BE>, R> {
  type Input<'r> = Yuva420p10Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::WidthAlignment(WidthAlignment::odd(
        self.width,
      )));
    }
    check_dimensions_match(self.width, self.height, width, height)?;
    reset_high_bit_yuva_streams(self);
    Ok(())
  }

  fn process(&mut self, row: Yuva420p10Row<'_>) -> Result<(), Self::Error> {
    if self.plan.is_some() {
      return yuva420p_high_bit_resample::<10, BE>(
        self,
        row.row(),
        row.y(),
        row.u_half(),
        row.v_half(),
        row.a(),
        row.matrix(),
        row.full_range(),
        RowSlice::Y10,
        RowSlice::UHalf10,
        RowSlice::VHalf10,
        RowSlice::AFull10,
        yuva420p10_to_rgba_row_endian,
        yuva420p10_to_rgba_u16_row_endian,
        // Centered-siting (#302, RFC #238 S6c) 4:4:4 twins fed the horizontally
        // reconstructed full-width chroma; α passed through unchanged.
        yuva444p10_to_rgba_row_endian,
        yuva444p10_to_rgba_u16_row_endian,
      );
    }
    yuva420p_high_bit_process::<10, BE, _, _>(
      self,
      row.row(),
      row.y(),
      row.u_half(),
      row.v_half(),
      row.a(),
      row.matrix(),
      row.full_range(),
      RowSlice::Y10,
      RowSlice::UHalf10,
      RowSlice::VHalf10,
      RowSlice::AFull10,
      yuv420p10_to_rgb_row_endian,
      yuv420p10_to_rgb_u16_row_endian,
      yuva420p10_to_rgba_row_endian,
      yuv420p10_to_hsv_row_endian,
      yuva420p10_to_rgba_u16_row_endian,
      // Centered-siting (#302) 4:4:4 twins.
      yuv444p10_to_rgb_row_endian,
      yuv444p10_to_rgb_u16_row_endian,
      yuva444p10_to_rgba_row_endian,
      yuv444p10_to_hsv_row_endian,
      yuva444p10_to_rgba_u16_row_endian,
    )
  }
}

// ---- Yuva420p12 impl --------------------------------------------------

impl<'a, R, const BE: bool> MixedSinker<'a, Yuva420p12<BE>, R> {
  /// Attaches a packed **8-bit** RGBA output buffer. Source-derived
  /// alpha (depth-converted via `>> 4`).
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

  /// Attaches a packed **`u16`** RGBA output buffer (12‑bit
  /// low‑packed, `[0, 4095]`). Alpha is sourced at native depth.
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

  /// Attaches a packed **`u16`** RGB output buffer (alpha-drop).
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

  /// Attaches a **`u16`** luma output buffer. Luma is the **native Y**
  /// (the binned Y plane at native depth under a non-identity plan; the
  /// Y plane verbatim otherwise). Length in u16 **elements**
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

impl<R, const BE: bool> Yuva420p12Sink<BE> for MixedSinker<'_, Yuva420p12<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, Yuva420p12<BE>, R> {
  type Input<'r> = Yuva420p12Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::WidthAlignment(WidthAlignment::odd(
        self.width,
      )));
    }
    check_dimensions_match(self.width, self.height, width, height)?;
    reset_high_bit_yuva_streams(self);
    Ok(())
  }

  fn process(&mut self, row: Yuva420p12Row<'_>) -> Result<(), Self::Error> {
    if self.plan.is_some() {
      return yuva420p_high_bit_resample::<12, BE>(
        self,
        row.row(),
        row.y(),
        row.u_half(),
        row.v_half(),
        row.a(),
        row.matrix(),
        row.full_range(),
        RowSlice::Y12,
        RowSlice::UHalf12,
        RowSlice::VHalf12,
        RowSlice::AFull12,
        yuva420p12_to_rgba_row_endian,
        yuva420p12_to_rgba_u16_row_endian,
        // Centered-siting (#302, RFC #238 S6c) 4:4:4 twins fed the horizontally
        // reconstructed full-width chroma; α passed through unchanged.
        yuva444p12_to_rgba_row_endian,
        yuva444p12_to_rgba_u16_row_endian,
      );
    }
    yuva420p_high_bit_process::<12, BE, _, _>(
      self,
      row.row(),
      row.y(),
      row.u_half(),
      row.v_half(),
      row.a(),
      row.matrix(),
      row.full_range(),
      RowSlice::Y12,
      RowSlice::UHalf12,
      RowSlice::VHalf12,
      RowSlice::AFull12,
      yuv420p12_to_rgb_row_endian,
      yuv420p12_to_rgb_u16_row_endian,
      yuva420p12_to_rgba_row_endian,
      yuv420p12_to_hsv_row_endian,
      yuva420p12_to_rgba_u16_row_endian,
      // Centered-siting (#302) 4:4:4 twins.
      yuv444p12_to_rgb_row_endian,
      yuv444p12_to_rgb_u16_row_endian,
      yuva444p12_to_rgba_row_endian,
      yuv444p12_to_hsv_row_endian,
      yuva444p12_to_rgba_u16_row_endian,
    )
  }
}

// ---- Yuva420p16 impl --------------------------------------------------

impl<'a, R, const BE: bool> MixedSinker<'a, Yuva420p16<BE>, R> {
  /// Attaches a packed **8-bit** RGBA output buffer. Source-derived
  /// alpha (depth-converted via `>> 8`).
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

  /// Attaches a packed **`u16`** RGBA output buffer (16‑bit, full
  /// `u16` range `[0, 65535]`). Alpha is sourced at native depth.
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

  /// Attaches a packed **`u16`** RGB output buffer (alpha-drop).
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

  /// Attaches a **`u16`** luma output buffer. Luma is the **native Y**
  /// (the binned Y plane at native depth under a non-identity plan; the
  /// Y plane verbatim otherwise). Length in u16 **elements**
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

impl<R, const BE: bool> Yuva420p16Sink<BE> for MixedSinker<'_, Yuva420p16<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, Yuva420p16<BE>, R> {
  type Input<'r> = Yuva420p16Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::WidthAlignment(WidthAlignment::odd(
        self.width,
      )));
    }
    check_dimensions_match(self.width, self.height, width, height)?;
    reset_high_bit_yuva_streams(self);
    Ok(())
  }

  fn process(&mut self, row: Yuva420p16Row<'_>) -> Result<(), Self::Error> {
    if self.plan.is_some() {
      return yuva420p_high_bit_resample::<16, BE>(
        self,
        row.row(),
        row.y(),
        row.u_half(),
        row.v_half(),
        row.a(),
        row.matrix(),
        row.full_range(),
        RowSlice::Y16,
        RowSlice::UHalf16,
        RowSlice::VHalf16,
        RowSlice::AFull16,
        yuva420p16_to_rgba_row_endian,
        yuva420p16_to_rgba_u16_row_endian,
        // Centered-siting (#302, RFC #238 S6c) 4:4:4 twins fed the horizontally
        // reconstructed full-width chroma; α passed through unchanged.
        yuva444p16_to_rgba_row_endian,
        yuva444p16_to_rgba_u16_row_endian,
      );
    }
    yuva420p_high_bit_process::<16, BE, _, _>(
      self,
      row.row(),
      row.y(),
      row.u_half(),
      row.v_half(),
      row.a(),
      row.matrix(),
      row.full_range(),
      RowSlice::Y16,
      RowSlice::UHalf16,
      RowSlice::VHalf16,
      RowSlice::AFull16,
      yuv420p16_to_rgb_row_endian,
      yuv420p16_to_rgb_u16_row_endian,
      yuva420p16_to_rgba_row_endian,
      yuv420p16_to_hsv_row_endian,
      yuva420p16_to_rgba_u16_row_endian,
      // Centered-siting (#302) 4:4:4 twins.
      yuv444p16_to_rgb_row_endian,
      yuv444p16_to_rgb_u16_row_endian,
      yuva444p16_to_rgba_row_endian,
      yuv444p16_to_hsv_row_endian,
      yuva444p16_to_rgba_u16_row_endian,
    )
  }
}

// ---- Shared high-bit YUVA 4:2:0 process body --------------------------
//
// The 9 / 10 / 16-bit YUVA 4:2:0 sinker `process` bodies are
// structurally identical — only the depth-named row primitives, the
// `RowSlice` variants used in error reports, and the depth-conversion
// shift (`BITS - 8`) for luma differ. Factor the shared body into a
// generic helper to avoid 3x of the same ~70 lines.
//
// Strategy A combine for the alpha-drop paths (rgb_u16 alpha-drop +
// rgba_u16 source alpha): runs the rgb_u16 kernel into the caller
// buffer + the rgba_u16 kernel separately. Cannot reuse rgb_u16's
// output because alpha comes from the source plane; the expand
// helper would splat opaque alpha.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
#[cfg_attr(not(tarpaulin), inline(always))]
fn yuva420p_high_bit_process<const BITS: u32, const BE: bool, F: crate::SourceFormat, R>(
  sinker: &mut MixedSinker<'_, F, R>,
  idx: usize,
  y_row: &[u16],
  u_half_row: &[u16],
  v_half_row: &[u16],
  a_row: &[u16],
  matrix: crate::ColorMatrix,
  full_range: bool,
  y_slice: RowSlice,
  u_slice: RowSlice,
  v_slice: RowSlice,
  a_slice: RowSlice,
  // Each dispatch fn is the `_endian` variant — last `bool` is the runtime
  // `big_endian` flag the helper passes from `BE`. Function POINTERS (not
  // generic `Fn` bounds) so the 4:2:0 dispatcher and its 4:4:4 twin below —
  // distinct fn items of the SAME signature — coerce to one parameter type.
  rgb_dispatch: fn(&[u16], &[u16], &[u16], &mut [u8], usize, crate::ColorMatrix, bool, bool, bool),
  rgb_u16_dispatch: fn(
    &[u16],
    &[u16],
    &[u16],
    &mut [u16],
    usize,
    crate::ColorMatrix,
    bool,
    bool,
    bool,
  ),
  rgba_dispatch: fn(
    &[u16],
    &[u16],
    &[u16],
    &[u16],
    &mut [u8],
    usize,
    crate::ColorMatrix,
    bool,
    bool,
    bool,
  ),
  // The matching alpha-drop YUV→HSV `_endian` kernel — same Y/U/V inputs as
  // `rgb_dispatch`, three `&mut [u8]` H/S/V outputs.
  hsv_dispatch: fn(
    &[u16],
    &[u16],
    &[u16],
    &mut [u8],
    &mut [u8],
    &mut [u8],
    usize,
    crate::ColorMatrix,
    bool,
    bool,
    bool,
  ),
  rgba_u16_dispatch: fn(
    &[u16],
    &[u16],
    &[u16],
    &[u16],
    &mut [u16],
    usize,
    crate::ColorMatrix,
    bool,
    bool,
    bool,
  ),
  // Chroma siting (#302): the centered (phase-0.5) 4:4:4 twins of the five
  // dispatchers above. `*_444` rgb / rgb_u16 / hsv are the alpha-drop 4:4:4
  // kernels (`yuv444pN_to_rgb*` / `_to_hsv`); `*_444` rgba / rgba_u16 are the
  // source-alpha 4:4:4 kernels (`yuva444pN_to_rgba*`) — the alpha plane is
  // full-resolution, so it is fed unchanged. Each takes FULL-width U / V (the
  // reconstructed chroma); their signatures match the matching 4:2:0
  // dispatcher. They are invoked only on the centered sitings; the default
  // path never touches them.
  rgb_444_dispatch: fn(
    &[u16],
    &[u16],
    &[u16],
    &mut [u8],
    usize,
    crate::ColorMatrix,
    bool,
    bool,
    bool,
  ),
  rgb_u16_444_dispatch: fn(
    &[u16],
    &[u16],
    &[u16],
    &mut [u16],
    usize,
    crate::ColorMatrix,
    bool,
    bool,
    bool,
  ),
  rgba_444_dispatch: fn(
    &[u16],
    &[u16],
    &[u16],
    &[u16],
    &mut [u8],
    usize,
    crate::ColorMatrix,
    bool,
    bool,
    bool,
  ),
  hsv_444_dispatch: fn(
    &[u16],
    &[u16],
    &[u16],
    &mut [u8],
    &mut [u8],
    &mut [u8],
    usize,
    crate::ColorMatrix,
    bool,
    bool,
    bool,
  ),
  rgba_u16_444_dispatch: fn(
    &[u16],
    &[u16],
    &[u16],
    &[u16],
    &mut [u16],
    usize,
    crate::ColorMatrix,
    bool,
    bool,
    bool,
  ),
) -> Result<(), MixedSinkerError> {
  let w = sinker.width;
  let h = sinker.height;
  let use_simd = sinker.simd;
  // Chroma siting (#302): `Copy`, read before the field split-borrow below.
  let chroma_location = sinker.chroma_location;

  if w & 1 != 0 {
    return Err(MixedSinkerError::WidthAlignment(WidthAlignment::odd(w)));
  }
  if y_row.len() != w {
    return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
      y_slice,
      idx,
      w,
      y_row.len(),
    )));
  }
  if u_half_row.len() != w / 2 {
    return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
      u_slice,
      idx,
      w / 2,
      u_half_row.len(),
    )));
  }
  if v_half_row.len() != w / 2 {
    return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
      v_slice,
      idx,
      w / 2,
      v_half_row.len(),
    )));
  }
  if a_row.len() != w {
    return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
      a_slice,
      idx,
      w,
      a_row.len(),
    )));
  }
  if idx >= sinker.height {
    return Err(MixedSinkerError::RowIndexOutOfRange(
      RowIndexOutOfRange::new(idx, sinker.height),
    ));
  }

  let MixedSinker {
    rgb,
    rgb_u16,
    rgba,
    rgba_u16,
    luma,
    luma_u16,
    hsv,
    rgb_scratch,
    chroma_full_u16,
    ..
  } = sinker;
  let one_plane_start = idx * w;
  let one_plane_end = one_plane_start + w;

  let want_rgb = rgb.is_some();
  let want_rgba = rgba.is_some();
  let want_hsv = hsv.is_some();
  let want_rgb_u16 = rgb_u16.is_some();
  let want_rgba_u16 = rgba_u16.is_some();
  // Chroma siting (#302): the centered horizontal sitings reconstruct chroma
  // at the phase-0.5 position then feed the 4:4:4 (+ source-alpha) kernels; the
  // default / co-sited path keeps the byte-identical fused 4:2:0 decode.
  let center_sited = chroma_420_center_sited_h(chroma_location);
  // HSV-without-RGB-or-RGBA goes through the direct alpha-drop
  // `hsv_dispatch` kernel (no source-width RGB scratch). HSV is
  // colour-only — the source alpha plane is dropped — so a high-bit YUVA
  // HSV is byte-identical to the no-alpha planar HSV on the same Y/U/V
  // (the alpha-drop `rgb_dispatch` and `hsv_dispatch` share the same
  // depth-named kernel family). RGB or RGBA also attached keeps the
  // convert-once-then-derive path alive via `need_rgb_kernel` (an RGBA
  // sink still needs the alpha plane, so it stays on the RGB path).
  let want_hsv_direct = want_hsv && !want_rgb && !want_rgba;
  let need_rgb_kernel = want_rgb || (want_hsv && want_rgba);

  // Atomicity preflight (#302, cf. the crate's #180 resample fix): reserve
  // EVERY fallible row scratch this identity row needs BEFORE any output row
  // (luma included) is written, so an allocator refusal returns a typed error
  // leaving the output frame untouched, never partially mutated. Two scratches
  // can grow: the centered-siting full-width `u16` chroma (`chroma_full_u16`),
  // needed by ANY colour output (u8 OR u16 RGB / RGBA / HSV — the alpha plane
  // is full-resolution, so siting-independent); and the u8 RGB row scratch
  // (below). The later `upsample_420_chroma_center_h_u16` reuses the
  // already-sized buffer.
  let need_centered_chroma =
    center_sited && (want_rgb || want_rgba || want_hsv || want_rgb_u16 || want_rgba_u16);
  if need_centered_chroma {
    reserve_420_chroma_full_u16(chroma_full_u16, w, h)?;
  }

  // Acquire the u8 RGB row buffer (caller plane or growing scratch) up
  // front — before any caller-output write below — so an allocator
  // refusal in the HSV-only scratch path returns a recoverable error
  // rather than leaving a partially-written caller buffer (luma_u16 /
  // rgb_u16 / rgba_u16). `rgb_row_buf_or_scratch` is the only other fallible
  // allocation on this direct path; the per-row writes that follow are
  // infallible. Mirrors the `packed_rgb_10bit` direct path.
  let rgb_row = if need_rgb_kernel {
    Some(rgb_row_buf_or_scratch(
      rgb.as_deref_mut(),
      rgb_scratch,
      one_plane_start,
      one_plane_end,
      w,
      h,
    )?)
  } else {
    None
  };

  // Centered full-width chroma (phase-0.5), reconstructed ONCE per row from the
  // wire-format half-width U / V and reused by every colour decode (u16 and
  // u8). Infallible — the scratch was reserved above. The default
  // left/unspecified siting leaves it `None`, so the fused 4:2:0 kernels
  // upsample chroma in-register and the output stays byte-identical.
  let centered = if need_centered_chroma {
    Some(upsample_420_chroma_center_h_u16::<BITS>(
      chroma_full_u16,
      u_half_row,
      v_half_row,
      w,
      BE,
    ))
  } else {
    None
  };

  // ---- luma (native Y; luma narrows `>> (BITS - 8)`, luma_u16 is the
  // host-native logical Y) ---------------------------------------------
  if luma.is_some() || luma_u16.is_some() {
    let luma_dst = luma.as_deref_mut();
    let luma_u16_dst = luma_u16.as_deref_mut();
    // Borrow each output's row slice up front so the per-pixel loop just
    // writes (the two outputs are disjoint fields).
    let mut luma_row = luma_dst.map(|b| &mut b[one_plane_start..one_plane_end]);
    let mut luma_u16_row = luma_u16_dst.map(|b| &mut b[one_plane_start..one_plane_end]);
    // Mask each decoded Y to the source's native depth `(1 << BITS) - 1`
    // (a no-op at `BITS = 16`). `Yuva420p*Frame::try_new` is geometry-only,
    // so a malformed-but-accepted frame can carry out-of-range Y (e.g.
    // `0x1000` at 12-bit); without this mask `luma_u16` would publish that
    // raw value, inconsistent with the `(1 << BITS) - 1`-masked Y the
    // RGB/RGBA row kernels decode from the same row.
    let sample_mask = ((1u32 << BITS) - 1) as u16;
    for (i, &s) in y_row.iter().enumerate().take(w) {
      // Normalize BE-encoded wire bytes to host-native before use —
      // without this, a valid BE sample like mid-gray `0x0200` (10-bit)
      // would be read as `0x0002` on a LE host and the `>> (BITS - 8)`
      // would write 0 instead of 128.
      let logical = (if BE { u16::from_be(s) } else { u16::from_le(s) }) & sample_mask;
      if let Some(row) = luma_row.as_deref_mut() {
        row[i] = (logical >> (BITS - 8)) as u8;
      }
      if let Some(row) = luma_u16_row.as_deref_mut() {
        row[i] = logical;
      }
    }
  }

  // ---- u16 RGB / RGBA path --------------------------------------
  // rgb_u16 (alpha-drop) reuses the non-alpha dispatcher.
  // rgba_u16 — when rgb_u16 is also present (combo case), Strategy A+:
  // expand the already-computed rgb_u16_row → rgba_u16_row then
  // overwrite α slot from the source plane (avoids a second chroma
  // kernel). When rgba_u16 is alone, delegate to the alpha-source-aware
  // dispatcher directly (standalone path — already optimal). Centered siting
  // (#302) routes each kernel through its 4:4:4 twin fed the full-width
  // phase-0.5 chroma reconstructed above.
  if want_rgb_u16 {
    let buf = rgb_u16.as_deref_mut().unwrap();
    let rgb_plane_end = one_plane_end
      .checked_mul(3)
      .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
        w, h, 3,
      )))?;
    let rgb_plane_start = one_plane_start * 3;
    let rgb_u16_row = &mut buf[rgb_plane_start..rgb_plane_end];
    if let Some((u_full, v_full)) = centered {
      rgb_u16_444_dispatch(
        y_row,
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
      rgb_u16_dispatch(
        y_row,
        u_half_row,
        v_half_row,
        rgb_u16_row,
        w,
        matrix,
        full_range,
        use_simd,
        BE,
      );
    }
    if want_rgba_u16 {
      // Combo: expand rgb_u16_row → rgba_u16_row, then overwrite α slot.
      let rgba_buf = rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row = rgba_u16_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      expand_rgb_u16_to_rgba_u16_row::<BITS>(rgb_u16_row, rgba_u16_row, w);
      // `BE` is plumbed from the row type so the alpha-plane copy honors
      // the same wire endianness as the caller's high-bit Yuva*p input.
      crate::row::alpha_extract::copy_alpha_plane_u16::<BITS, BE>(a_row, rgba_u16_row, w, use_simd);
    }
  } else if want_rgba_u16 {
    // Standalone rgba_u16: delegate to the alpha-source-aware dispatcher.
    let rgba_buf = rgba_u16.as_deref_mut().unwrap();
    let rgba_u16_row = rgba_u16_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
    if let Some((u_full, v_full)) = centered {
      rgba_u16_444_dispatch(
        y_row,
        u_full,
        v_full,
        a_row,
        rgba_u16_row,
        w,
        matrix,
        full_range,
        use_simd,
        BE,
      );
    } else {
      rgba_u16_dispatch(
        y_row,
        u_half_row,
        v_half_row,
        a_row,
        rgba_u16_row,
        w,
        matrix,
        full_range,
        use_simd,
        BE,
      );
    }
  }

  // ---- u8 RGB / RGBA / HSV path ----------------------------------
  // `need_rgb_kernel` / `rgb_row` were computed and (when needed)
  // allocated at the top, before any caller-output write.
  //
  // HSV-only (no u8 RGB / RGBA): convert the source Y/U/V straight to HSV
  // via the alpha-drop kernel — no source-width RGB scratch. Any u16
  // RGB/RGBA outputs already ran above. Centered siting (#302) routes through
  // the 4:4:4 twin fed the full-width phase-0.5 chroma.
  if want_hsv_direct {
    let hsv = hsv.as_mut().expect("want_hsv_direct implies hsv attached");
    let (h, s, v) = hsv.hsv();
    if let Some((u_full, v_full)) = centered {
      hsv_444_dispatch(
        y_row,
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
      hsv_dispatch(
        y_row,
        u_half_row,
        v_half_row,
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

  // Direct-RGBA-only (real source α, NOT forced opaque). Centered siting feeds
  // the 4:4:4 + source-alpha kernel the full-width phase-0.5 chroma; the alpha
  // plane is identical on both paths.
  if want_rgba && !need_rgb_kernel {
    let rgba_buf = rgba.as_deref_mut().unwrap();
    let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
    if let Some((u_full, v_full)) = centered {
      rgba_444_dispatch(
        y_row, u_full, v_full, a_row, rgba_row, w, matrix, full_range, use_simd, BE,
      );
    } else {
      rgba_dispatch(
        y_row, u_half_row, v_half_row, a_row, rgba_row, w, matrix, full_range, use_simd, BE,
      );
    }
    return Ok(());
  }

  // RGB kernel (alpha-drop reuses the non-alpha dispatcher verbatim; the 4:4:4
  // twin for centered siting).
  let Some(rgb_row) = rgb_row else {
    return Ok(());
  };
  if let Some((u_full, v_full)) = centered {
    rgb_444_dispatch(
      y_row, u_full, v_full, rgb_row, w, matrix, full_range, use_simd, BE,
    );
  } else {
    rgb_dispatch(
      y_row, u_half_row, v_half_row, rgb_row, w, matrix, full_range, use_simd, BE,
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

  // Both rgb and rgba attached (combo case): Strategy A+ — expand the
  // already-computed rgb_row → rgba_row (fills α = 0xFF), then
  // overwrite α slot from the source alpha plane with depth-conv
  // `>> (BITS - 8)`. Avoids a second chroma kernel.
  if want_rgba {
    let rgba_buf = rgba.as_deref_mut().unwrap();
    let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
    expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
    // BE = false: see the rgba_u16 branch above for rationale.
    crate::row::alpha_extract::copy_alpha_plane_u16_to_u8::<BITS, BE>(a_row, rgba_row, w, use_simd);
  }

  Ok(())
}

// ---- Shared high-bit YUVA 4:2:0 resample-routing body -----------------
//
// The non-identity plan branch for the 9 / 10 / 16-bit YUVA 4:2:0 sinks.
// Mirrors the 8-bit `Yuva420p` plan branch but at native depth `BITS`. The
// `Area` (downscale) arm routes through the shared packed-YUVA area tail with
// THREE independent binnings — the u8 colour stream (the `_to_rgba` u8 kernel
// with real source α, chroma upsampled 4:2:0), the **independent** native u16
// colour stream (the `_to_rgba_u16` kernel — never a narrowing of the u8
// bin), and the **low-packed** native-Y luma stream
// (`deinterleave_y_high_bit_masked::<BITS, BE>`, a host-native copy masked to
// the native depth — planar YUVA Y stores logical values directly, unlike the
// high-bit-packed semi-planar P-formats, so luma is `binned_Y >> (BITS - 8)`).
// The `Filter` arm routes
// the SAME converted RGBA / native-Y through the signed-coefficient filter
// tail (`NATIVE_LUMA_U8 = false`, the u16-luma branch — native Y is u16),
// straight alpha only. The 4:2:0-vs-4:2:2 difference (chroma row index `r / 2`
// vs `r`) lives in the walker, so the 4:2:2 sinks reuse this body verbatim
// with the same half-chroma kernels.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
#[cfg_attr(not(tarpaulin), inline(always))]
fn yuva420p_high_bit_resample<const BITS: u32, const BE: bool>(
  sinker: &mut MixedSinker<'_, impl crate::SourceFormat, impl Sized>,
  idx: usize,
  y_row: &[u16],
  u_half_row: &[u16],
  v_half_row: &[u16],
  a_row: &[u16],
  matrix: crate::ColorMatrix,
  full_range: bool,
  y_slice: RowSlice,
  u_slice: RowSlice,
  v_slice: RowSlice,
  a_slice: RowSlice,
  rgba_dispatch: fn(
    &[u16],
    &[u16],
    &[u16],
    &[u16],
    &mut [u8],
    usize,
    crate::ColorMatrix,
    bool,
    bool,
    bool,
  ),
  rgba_u16_dispatch: fn(
    &[u16],
    &[u16],
    &[u16],
    &[u16],
    &mut [u16],
    usize,
    crate::ColorMatrix,
    bool,
    bool,
    bool,
  ),
  // Chroma siting (#302, RFC #238 S6c): the centered (phase-0.5) 4:4:4 twins of
  // the two source-alpha dispatchers above (`yuva444pN_to_rgba*`). Each takes
  // FULL-width U / V (the horizontally reconstructed chroma) with the α plane
  // fed unchanged; their signatures match the matching 4:2:0 dispatcher. They
  // are invoked ONLY on the centered horizontal sitings; the co-sited / default
  // path keeps the byte-identical fused 4:2:0 decode.
  rgba_444_dispatch: fn(
    &[u16],
    &[u16],
    &[u16],
    &[u16],
    &mut [u8],
    usize,
    crate::ColorMatrix,
    bool,
    bool,
    bool,
  ),
  rgba_u16_444_dispatch: fn(
    &[u16],
    &[u16],
    &[u16],
    &[u16],
    &mut [u16],
    usize,
    crate::ColorMatrix,
    bool,
    bool,
    bool,
  ),
) -> Result<(), MixedSinkerError> {
  let w = sinker.width;
  let h = sinker.height;
  let use_simd = sinker.simd;
  // Chroma siting (#302): `Copy`, read before the field split-borrow below.
  let chroma_location = sinker.chroma_location;

  if w & 1 != 0 {
    return Err(MixedSinkerError::WidthAlignment(WidthAlignment::odd(w)));
  }
  if y_row.len() != w {
    return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
      y_slice,
      idx,
      w,
      y_row.len(),
    )));
  }
  if u_half_row.len() != w / 2 {
    return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
      u_slice,
      idx,
      w / 2,
      u_half_row.len(),
    )));
  }
  if v_half_row.len() != w / 2 {
    return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
      v_slice,
      idx,
      w / 2,
      v_half_row.len(),
    )));
  }
  if a_row.len() != w {
    return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
      a_slice,
      idx,
      w,
      a_row.len(),
    )));
  }
  if idx >= sinker.height {
    return Err(MixedSinkerError::RowIndexOutOfRange(
      RowIndexOutOfRange::new(idx, sinker.height),
    ));
  }

  let alpha_mode = sinker.alpha_mode;
  let MixedSinker {
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    luma,
    luma_u16,
    hsv,
    rgba_scratch,
    rgb_scratch,
    rgba_scratch_u16,
    rgba_color_scratch_u16,
    luma_scratch_u16,
    chroma_full_u16,
    plan,
    rgba_stream,
    rgba_stream_u16,
    luma_stream_u16,
    rgba_filter_stream,
    rgba_filter_stream_u16,
    luma_filter_stream_u16,
    resample_outputs,
    frozen_alpha_mode,
    frozen_chroma_centered,
    ..
  } = sinker;
  let plan = plan.as_ref().expect("plan.is_some() checked by the caller");
  check_frozen_alpha_mode(*frozen_alpha_mode, alpha_mode, idx)?;
  // RFC #238 S6c — 4:2:0 HORIZONTAL chroma siting for the high-bit alpha-bearing
  // source. The centered group (`chroma_420_center_sited_h` — chroma at `+0.5`
  // luma horizontally) reconstructs full-width chroma then feeds the 4:4:4
  // (+ source-alpha) twins; the co-sited / unspecified group is phase 0 (today's
  // fused 4:2:0 decode, byte-identical). The full-resolution α plane is never
  // subsampled, so it is siting-independent and passes through the 4:4:4 twins
  // UNCHANGED. Vertical stays co-sited (the `Bottom` v-fold is a later PR).
  let center_sited = chroma_420_center_sited_h(chroma_location);
  // A colour output drives the centered chroma reconstruction (u8 OR native
  // u16); a luma-only row bins the native Y (siting-independent) and never
  // reconstructs chroma, so it stays on the co-sited fused arm.
  let want_color =
    rgb.is_some() || rgba.is_some() || hsv.is_some() || rgb_u16.is_some() || rgba_u16.is_some();
  // The EXACT set the packed tails treat as output-bearing (colour u8 / u16 or
  // luma); gates the siting freeze so a no-output call (which consumes no stream
  // state) freezes nothing. MUST include the u16 colour outputs — a u16-only
  // sink still advances a stream and must freeze the siting.
  let need_output = want_color || luma.is_some() || luma_u16.is_some();
  // Freeze the effective horizontal siting on the first output-bearing row: a
  // later row observing a different phase — in sequence or not — would bin a
  // mixture of co-sited and centered chroma, so it is rejected here before any
  // reconstruction. The matching SET rides each accepted output row below.
  if need_output
    && let Some(frozen) = *frozen_chroma_centered
    && frozen != center_sited
  {
    return Err(MixedSinkerError::ChromaSitingChanged(
      ChromaSitingChanged::new(idx),
    ));
  }
  // The span kind picks the engine (mirrors the 8-bit `Yuva420p` and packed
  // `Vuya`): `Area` bins (the alpha-aware tail — premultiplied colour binned
  // premultiplied then un-premultiplied) at native precision; `Filter` runs
  // the signed-coefficient filter on the same converted RGBA (straight alpha
  // only). Unlike the 8-bit planar sinks (whose native Y is `u8`), the
  // high-bit native Y is genuinely `u16`, so the filter route uses
  // `NATIVE_LUMA_U8 = false` — the same u16-luma branch `Vuya` / `Vuyx` use:
  // luma rides the `u16` filter stream over the de-interleaved native Y. The
  // filter tail clamps every sub-16-bit colour AND native-Y overshoot to the
  // source's native max `(1 << BITS) - 1` (a value no-op at `BITS = 16`),
  // matching the in-range area path. A premultiplied `Filter` plan has no
  // analogue (the engine cannot un-premultiply), so it is routed to the area
  // tail, which surfaces the typed `UnsupportedFilter`.
  let r = match plan.kind() {
    crate::resample::SpanKind::Area => {
      if center_sited && want_color {
        // Centered area: reconstruct full-width chroma and bin the 4:4:4-decoded
        // RGBA — but ONLY after the resample preflight (frozen-output +
        // sequence), so an out-of-sequence / rejected row is caught before the
        // chroma reservation (#180). `packed_yuva444_resample` re-runs the
        // idempotent preflight. The α plane is full-resolution and
        // siting-independent: it flows into the 4:4:4 converters UNCHANGED.
        let expected = if rgb.is_some() || rgba.is_some() || hsv.is_some() {
          rgba_stream.as_ref().map_or(0, |s| s.next_y())
        } else {
          rgba_stream_u16.as_ref().map_or(0, |s| s.next_y())
        };
        match super::planar_resample::resample_preflight_check_only(
          resample_outputs,
          luma,
          luma_u16,
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
          core::ops::ControlFlow::Break(()) => Ok(()),
          core::ops::ControlFlow::Continue(()) => {
            reserve_420_chroma_full_u16(chroma_full_u16, w, h)?;
            let (u_full, v_full) = upsample_420_chroma_center_h_u16::<BITS>(
              chroma_full_u16,
              u_half_row,
              v_half_row,
              w,
              BE,
            );
            packed_yuva444_resample::<BITS>(
              rgba_stream,
              rgba_stream_u16,
              luma_stream_u16,
              resample_outputs,
              rgb,
              rgba,
              rgb_u16,
              rgba_u16,
              luma,
              luma_u16,
              hsv,
              rgba_scratch,
              rgb_scratch,
              rgba_scratch_u16,
              rgba_color_scratch_u16,
              luma_scratch_u16,
              w,
              plan,
              idx,
              use_simd,
              alpha_mode,
              |dst| {
                rgba_444_dispatch(
                  y_row, u_full, v_full, a_row, dst, w, matrix, full_range, use_simd, BE,
                )
              },
              |dst| {
                rgba_u16_444_dispatch(
                  y_row, u_full, v_full, a_row, dst, w, matrix, full_range, use_simd, BE,
                )
              },
              |dst| deinterleave_y_high_bit_masked::<BITS, BE>(y_row, dst, w),
            )
          }
        }
      } else {
        packed_yuva444_resample::<BITS>(
          rgba_stream,
          rgba_stream_u16,
          luma_stream_u16,
          resample_outputs,
          rgb,
          rgba,
          rgb_u16,
          rgba_u16,
          luma,
          luma_u16,
          hsv,
          rgba_scratch,
          rgb_scratch,
          rgba_scratch_u16,
          rgba_color_scratch_u16,
          luma_scratch_u16,
          w,
          plan,
          idx,
          use_simd,
          alpha_mode,
          |dst| {
            rgba_dispatch(
              y_row, u_half_row, v_half_row, a_row, dst, w, matrix, full_range, use_simd, BE,
            )
          },
          |dst| {
            rgba_u16_dispatch(
              y_row, u_half_row, v_half_row, a_row, dst, w, matrix, full_range, use_simd, BE,
            )
          },
          |dst| deinterleave_y_high_bit_masked::<BITS, BE>(y_row, dst, w),
        )
      }
    }
    crate::resample::SpanKind::Filter if alpha_mode.is_premultiplied() => {
      // Premultiplied + filter has no analogue: route to the area tail with the
      // filter plan so it returns the typed `UnsupportedFilter`. Siting is
      // irrelevant — the row is rejected before any chroma reconstruction, so it
      // stays on the co-sited kernels (no wasted reservation, #180).
      packed_yuva444_resample::<BITS>(
        rgba_stream,
        rgba_stream_u16,
        luma_stream_u16,
        resample_outputs,
        rgb,
        rgba,
        rgb_u16,
        rgba_u16,
        luma,
        luma_u16,
        hsv,
        rgba_scratch,
        rgb_scratch,
        rgba_scratch_u16,
        rgba_color_scratch_u16,
        luma_scratch_u16,
        w,
        plan,
        idx,
        use_simd,
        alpha_mode,
        |dst| {
          rgba_dispatch(
            y_row, u_half_row, v_half_row, a_row, dst, w, matrix, full_range, use_simd, BE,
          )
        },
        |dst| {
          rgba_u16_dispatch(
            y_row, u_half_row, v_half_row, a_row, dst, w, matrix, full_range, use_simd, BE,
          )
        },
        |dst| deinterleave_y_high_bit_masked::<BITS, BE>(y_row, dst, w),
      )
    }
    crate::resample::SpanKind::Filter => {
      // Reject a multi-kernel (BICUBLIN) filter plan BEFORE the centered
      // preflight/reserve below: `packed_yuva444_filter_resample`'s first act is
      // this same check, so without hoisting it a BICUBLIN plan would freeze the
      // output set and reserve/reconstruct chroma before being rejected as
      // `UnsupportedFilter` (#180 reject-before-allocation). Idempotent — the
      // delegate re-runs it.
      plan.ensure_single_kernel_filter()?;
      if center_sited && want_color {
        // Centered filter: reconstruct full-width chroma and filter the
        // 4:4:4-decoded RGBA — but ONLY after the filter preflight (frozen-output
        // + sequence), so an out-of-sequence / rejected row is caught before the
        // chroma reservation (#180). The α plane is siting-independent and flows
        // into the 4:4:4 converters UNCHANGED.
        let expected = if rgb.is_some() || rgba.is_some() || hsv.is_some() {
          rgba_filter_stream.as_ref().map_or(0, |s| s.next_y())
        } else {
          rgba_filter_stream_u16.as_ref().map_or(0, |s| s.next_y())
        };
        match super::planar_resample::resample_preflight_check_only(
          resample_outputs,
          luma,
          luma_u16,
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
          core::ops::ControlFlow::Break(()) => Ok(()),
          core::ops::ControlFlow::Continue(()) => {
            reserve_420_chroma_full_u16(chroma_full_u16, w, h)?;
            let (u_full, v_full) = upsample_420_chroma_center_h_u16::<BITS>(
              chroma_full_u16,
              u_half_row,
              v_half_row,
              w,
              BE,
            );
            packed_yuva444_filter_resample::<BITS, false, false>(
              rgba_filter_stream,
              rgba_filter_stream_u16,
              &mut None,
              luma_filter_stream_u16,
              resample_outputs,
              rgb,
              rgba,
              rgb_u16,
              rgba_u16,
              luma,
              luma_u16,
              hsv,
              rgba_scratch,
              rgb_scratch,
              rgba_scratch_u16,
              rgba_color_scratch_u16,
              luma_scratch_u16,
              w,
              plan,
              idx,
              use_simd,
              &[],
              None,
              |dst| {
                rgba_444_dispatch(
                  y_row, u_full, v_full, a_row, dst, w, matrix, full_range, use_simd, BE,
                )
              },
              |dst| {
                rgba_u16_444_dispatch(
                  y_row, u_full, v_full, a_row, dst, w, matrix, full_range, use_simd, BE,
                )
              },
              |dst| deinterleave_y_high_bit_masked::<BITS, BE>(y_row, dst, w),
              |_dst: &mut [u8]| {},
            )
          }
        }
      } else {
        packed_yuva444_filter_resample::<BITS, false, false>(
          rgba_filter_stream,
          rgba_filter_stream_u16,
          // High-bit planar YUVA never uses the u8 native-Y luma stream
          // (`NATIVE_LUMA_U8 = false`); pass an inert slot.
          &mut None,
          luma_filter_stream_u16,
          resample_outputs,
          rgb,
          rgba,
          rgb_u16,
          rgba_u16,
          luma,
          luma_u16,
          hsv,
          rgba_scratch,
          rgb_scratch,
          rgba_scratch_u16,
          rgba_color_scratch_u16,
          luma_scratch_u16,
          w,
          plan,
          idx,
          use_simd,
          // Luma rides `deinterleave_y` + the u16 stream (native Y is u16), so the
          // u8-luma input is unused.
          &[],
          None,
          |dst| {
            rgba_dispatch(
              y_row, u_half_row, v_half_row, a_row, dst, w, matrix, full_range, use_simd, BE,
            )
          },
          |dst| {
            rgba_u16_dispatch(
              y_row, u_half_row, v_half_row, a_row, dst, w, matrix, full_range, use_simd, BE,
            )
          },
          |dst| deinterleave_y_high_bit_masked::<BITS, BE>(y_row, dst, w),
          |_dst: &mut [u8]| {},
        )
      }
    }
  };
  // RFC #238 S6c: freeze the horizontal 4:2:0 chroma siting on the first
  // ACCEPTED output-bearing resampled row (a rejected row returns Err above and
  // leaves it unset for a corrected retry). Both packed tails carry one plan
  // kind for the frame, so there is no native/row-stage route to co-freeze here.
  if r.is_ok() && need_output && frozen_chroma_centered.is_none() {
    *frozen_chroma_centered = Some(center_sited);
  }
  r
}
