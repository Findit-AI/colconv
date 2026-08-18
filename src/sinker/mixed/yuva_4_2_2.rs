//! Sinker impls for source-side YUVA 4:2:2 formats — Yuva422p (8-bit),
//! Yuva422p9, Yuva422p10, Yuva422p12, Yuva422p16. Wiring-only: per-row chroma
//! layout is identical to YUVA 4:2:0 (half-width U / V), so this file
//! delegates row-level work to the existing
//! `yuva420p*_to_rgba*_with_alpha_src_row` dispatchers from Ship 8b‑2.
//! The 4:2:0 vs 4:2:2 difference is purely in the vertical walker
//! (chroma row index `r / 2` vs `r`) and is handled in the walker
//! / sinker layer.
//!
//! Each format gets:
//! - **RGBA output paths (u8 + native-depth u16, where applicable)** —
//!   alpha is sourced from the source alpha plane via the
//!   `yuva420p*_to_rgba*_with_alpha_src_row` dispatchers. The 8-bit
//!   `Yuva422p` has no native `u16` RGBA output (mirrors `Yuva420p`).
//! - **RGB / RGB-u16 / luma / HSV alpha-drop paths** — these reuse the
//!   existing 4:2:0 row dispatchers
//!   (`yuv420p*_to_rgb*` / `yuv420p_to_rgb_row`) directly, since the
//!   per-row chroma layout is identical between 4:2:0 and 4:2:2 (the
//!   YUV 4:2:2 sinkers do the same). The alpha plane is ignored;
//!   output bytes / elements match what the corresponding
//!   `Yuv422p<BITS>` source would produce given the same Y/U/V data.
//!
//! **Chroma siting (#302):** the centered horizontal sitings
//! (`chroma_422_center_sited_h` — `Center` / `Top` / `Bottom`) reconstruct
//! full-width chroma at the phase-0.5 position then decode via the 4:4:4 (+
//! source-alpha) kernels, identical to the just-merged `Yuv422p` path; the
//! co-sited / `Left` / unspecified sitings keep the byte-identical
//! nearest-neighbor decode. 4:2:2 is subsampled horizontally only (no vertical
//! phase). The full-resolution source alpha plane is never subsampled, so it is
//! siting-independent and passes through unchanged on every path (RGBA), exactly
//! as the YUVA 4:2:0 sink does.

use super::{
  ChromaSitingChanged, GeometryOverflow, InsufficientBuffer, MixedSinker, MixedSinkerError,
  RowIndexOutOfRange, RowShapeMismatch, RowSlice, WidthAlignment, check_dimensions_match,
  check_frozen_alpha_mode, chroma_422_center_sited_h, deinterleave_y_high_bit_masked,
  packed_yuva444_filter_resample, packed_yuva444_resample,
  planar_8bit::{reserve_420_chroma_full, upsample_420_chroma_center_h},
  reset_high_bit_yuva_streams, rgb_row_buf_or_scratch, rgba_plane_row_slice,
  rgba_u16_plane_row_slice,
  subsampled_4_2_0_high_bit::{reserve_420_chroma_full_u16, upsample_420_chroma_center_h_u16},
};
use crate::{PixelSink, row::*, source::*};

// ---- Yuva422p impl (8-bit) ---------------------------------------------

impl<'a, R> MixedSinker<'a, Yuva422p, R> {
  /// Attaches a packed **8‑bit** RGBA output buffer. The 8‑bit YUVA
  /// 4:2:2 source is converted to 8‑bit RGBA via the same Q15 i32
  /// 8‑bit kernel that backs [`MixedSinker<Yuv422p>::with_rgba`] (the
  /// 4:2:0 row kernel — `yuva420p_to_rgba_row` — applies verbatim
  /// since per-row chroma layout is identical between 4:2:0 and
  /// 4:2:2); the per-pixel alpha byte is **sourced from the alpha
  /// plane** — not constant `0xFF`.
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

impl<R> Yuva422pSink for MixedSinker<'_, Yuva422p, R> {}

impl<R> PixelSink for MixedSinker<'_, Yuva422p, R> {
  type Input<'r> = Yuva422pRow<'r>;
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
    // no-alpha `Yuv422p`). The 8-bit `Yuva422p` exposes no u16 colour
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
    self.resample_outputs = None;
    // RFC #238 S7: clear the per-frame frozen 4:2:2 horizontal chroma siting so
    // the next frame may pick either phase while a mid-frame flip stays rejected.
    self.frozen_chroma_centered = None;
    self.frozen_alpha_mode = Some(self.alpha_mode);
    Ok(())
  }

  fn process(&mut self, row: Yuva422pRow<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;
    // Chroma siting (#302): drives the identity-plan horizontal chroma phase
    // for the Y/U/V colour decode. The full-resolution source alpha plane is
    // siting-independent (it is never subsampled), so it passes through
    // unchanged on every path. `Copy`, so read it out before the field
    // split-borrow below.
    let chroma_location = self.chroma_location.clone();

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

    // Non-identity plan: `Yuva422p` is 8-bit planar 4:2:2 YUV **with a
    // real full-resolution source alpha plane**. Route through the
    // packed-YUVA tail at `SRC_BITS = 8` exactly like `Yuva420p`: the u8
    // colour stream resamples the converted u8 RGBA row, the native-Y luma
    // stream resamples the (zero-extended) Y plane directly. The per-row
    // chroma layout (half-width U / V) is identical to 4:2:0, so the colour
    // closure uses the same `yuva420p_to_rgba_row` kernel the direct
    // `Yuva422p` path already reuses (the 4:2:0-vs-4:2:2 difference is the
    // vertical chroma-row index, owned by the walker). The 8-bit
    // `Yuva422p` exposes no u16 colour outputs, so the tail's u16 colour
    // resampling is never active and its `convert_rgba_u16` closure is never
    // invoked.
    //
    // The span kind picks the engine (mirrors `Yuva420p`): `Area` bins (the
    // alpha-aware tail — premultiplied colour binned premultiplied then
    // un-premultiplied); `Filter` runs the signed-coefficient filter on the
    // same converted RGBA (straight alpha only). A premultiplied `Filter`
    // plan is routed to the area tail so it surfaces the typed
    // `UnsupportedFilter` rather than straight-filtering premultiplied colour.
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
        frozen_chroma_centered,
        ..
      } = self;
      let plan = plan.as_ref().expect("plan.is_some() checked above");
      check_frozen_alpha_mode(*frozen_alpha_mode, alpha_mode, idx)?;
      // RFC #238 S7 — 4:2:2 HORIZONTAL chroma siting for the 8-bit alpha-bearing
      // source. The centered group (`chroma_422_center_sited_h` — chroma at `+0.5`
      // luma horizontally) reconstructs full-width chroma then feeds the 4:4:4
      // (+ source-alpha) kernels; the co-sited / unspecified group is phase 0
      // (today's fused 4:2:2 decode, byte-identical). 4:2:2 is subsampled
      // horizontally only — no vertical phase. The full-resolution α plane is never
      // subsampled, so it is siting-independent and passes through the 4:4:4 kernels
      // UNCHANGED.
      let center_sited = chroma_422_center_sited_h(&chroma_location);
      // A colour output drives the centered chroma reconstruction; a luma-only row
      // bins the native Y (siting-independent) and never reconstructs chroma, so it
      // stays on the co-sited fused arm.
      let want_color = rgb.is_some() || rgba.is_some() || hsv.is_some();
      // The EXACT set the packed tail treats as output-bearing; gates the siting
      // freeze so a no-output call (which consumes no stream state) freezes nothing.
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
      let r = match plan.kind() {
        crate::resample::SpanKind::Area if center_sited && want_color => {
          // Centered area: reconstruct full-width chroma and bin the 4:4:4-decoded
          // RGBA — but ONLY after the resample preflight (frozen-output +
          // sequence), so an out-of-sequence / rejected row is caught before the
          // chroma reservation (#180). `packed_yuva444_resample` re-runs the
          // idempotent preflight. The α plane is full-resolution and
          // siting-independent: it flows into the 4:4:4 converter UNCHANGED.
          let expected = rgba_stream.as_ref().map_or(0, |s| s.next_y());
          match super::resample_preflight_check_only(
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
            core::ops::ControlFlow::Break(()) => Ok(()),
            core::ops::ControlFlow::Continue(()) => {
              reserve_420_chroma_full(chroma_full, w, h)?;
              let (u_full, v_full) =
                upsample_420_chroma_center_h(chroma_full, u_half, v_half, w, use_simd);
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
                // `Yuva422p` has no u16 colour outputs, so this is never called.
                |_dst: &mut [u16]| {},
                |dst| {
                  for (d, &s) in dst.iter_mut().zip(y) {
                    *d = s as u16;
                  }
                },
              )
            }
          }
        }
        crate::resample::SpanKind::Area => packed_yuva444_resample::<8>(
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
          // `Yuva422p` has no u16 colour outputs, so this closure is never called.
          |_dst: &mut [u16]| {},
          |dst| {
            for (d, &s) in dst.iter_mut().zip(y) {
              *d = s as u16;
            }
          },
        ),
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
        crate::resample::SpanKind::Filter if center_sited && want_color => {
          // Reject a multi-kernel (BICUBLIN) filter plan BEFORE the centered
          // preflight/reserve below: `packed_yuva444_filter_resample`'s first act
          // is this same check, so without hoisting it a BICUBLIN plan would freeze
          // the output set and reserve/reconstruct chroma before being rejected as
          // `UnsupportedFilter` (the #180 reject-before-allocation invariant).
          // Idempotent — the delegate re-runs it.
          plan.ensure_single_kernel_filter()?;
          // Centered filter: reconstruct full-width chroma and filter the
          // 4:4:4-decoded RGBA — but ONLY after the filter preflight (frozen-output
          // + sequence), so an out-of-sequence / rejected row is caught before the
          // chroma reservation (#180). The α plane is siting-independent and flows
          // into the 4:4:4 converter UNCHANGED.
          let expected = rgba_filter_stream.as_ref().map_or(0, |s| s.next_y());
          match super::resample_preflight_check_only(
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
            core::ops::ControlFlow::Break(()) => Ok(()),
            core::ops::ControlFlow::Continue(()) => {
              reserve_420_chroma_full(chroma_full, w, h)?;
              let (u_full, v_full) =
                upsample_420_chroma_center_h(chroma_full, u_half, v_half, w, use_simd);
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
            }
          }
        }
        crate::resample::SpanKind::Filter => packed_yuva444_filter_resample::<8, true, false>(
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
          // 8-bit native-Y luma rides the u8 stream (parity with `Yuv422p`):
          // the contiguous Y plane is fed directly, so no de-interleave scratch.
          y,
          None,
          |dst| yuva420p_to_rgba_row(y, u_half, v_half, a, dst, w, matrix, full_range, use_simd),
          // `Yuva422p` has no u16 colour outputs, so this closure is never called.
          |_dst: &mut [u16]| {},
          // u8-luma path: the u16 luma stream is detached, so this is never
          // called.
          |_dst: &mut [u16]| {},
          // Contiguous Y plane fed directly, so this u8 de-interleave is unused.
          |_dst: &mut [u8]| {},
        ),
      };
      // RFC #238 S7: freeze the horizontal 4:2:2 chroma siting on the first
      // ACCEPTED output-bearing resampled row (a rejected row returns Err above and
      // leaves it unset for a corrected retry). The single packed tail carries one
      // plan kind for the frame, so there is no native/row-stage route to co-freeze.
      if r.is_ok() && need_output && frozen_chroma_centered.is_none() {
        *frozen_chroma_centered = Some(center_sited);
      }
      return r;
    }

    let Self {
      rgb,
      rgba,
      luma,
      luma_u16,
      hsv,
      rgb_scratch,
      chroma_full,
      // RFC #238 S7: the 4:2:2 chroma phase frozen on the first output row.
      frozen_chroma_centered,
      ..
    } = self;

    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();

    // Repo-wide no-output invariant: a `process` call carrying NO output — no
    // colour, no luma, no luma_u16 — runs NOTHING: no per-row offset arithmetic,
    // no allocation, no state mutation. Returning HERE, before the `idx * w`
    // offsets below, also keeps the invariant overflow-safe — a no-output call
    // never ran an attach-time `w x h x 1` validation, so `idx * w` could overflow
    // `usize` on a 32-bit target with absurd geometry; the guard skips that math
    // (and the centered chroma reservation) entirely. Mirrors the no-alpha
    // `Yuv422p` sibling.
    let need_output = want_rgb || want_rgba || want_hsv || luma.is_some() || luma_u16.is_some();
    if !need_output {
      return Ok(());
    }

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // Chroma siting (#302): the centered horizontal sitings reconstruct chroma at
    // the phase-0.5 position then feed the 4:4:4 (+ source-alpha) kernels; the
    // default / co-sited path keeps the byte-identical fused 4:2:0 decode (4:2:2's
    // per-row chroma contract is identical — half-width chroma, one pair per Y
    // pair). 4:2:2 is subsampled horizontally only — no vertical blend or chroma
    // lookback (cf. the 4:2:0 sibling).
    let center_sited = chroma_422_center_sited_h(&chroma_location);

    // Per-frame chroma-siting freeze (RFC #238, mirroring the resample-path guard
    // above): the first output-bearing row pins the phase; a later row whose siting
    // flipped would decode a mixture of centered and co-sited chroma into ONE
    // frame, so reject it here BEFORE any scratch reserve or output write.
    // `begin_frame` clears the freeze so the next frame may pick either phase.
    if need_output
      && let Some(frozen) = *frozen_chroma_centered
      && frozen != center_sited
    {
      return Err(MixedSinkerError::ChromaSitingChanged(
        ChromaSitingChanged::new(idx),
      ));
    }

    // HSV-without-RGB-or-RGBA goes through the direct `yuv_*_to_hsv_row` kernel
    // (no source-width RGB scratch). HSV is colour-only — the source alpha plane
    // is dropped — so a YUVA HSV is byte-identical to the no-alpha `Yuv422p` HSV
    // on the same Y/U/V. RGB or RGBA also attached keeps the
    // convert-once-then-derive path alive via `need_rgb_kernel` (an RGBA sink
    // still needs the alpha plane, so it stays on the RGB path).
    let want_hsv_direct = want_hsv && !want_rgb && !want_rgba;
    let need_rgb_kernel = want_rgb || (want_hsv && want_rgba);

    // Atomicity preflight (#302, cf. the crate's #180 resample fix): reserve EVERY
    // fallible row scratch this identity row needs BEFORE any output row (luma /
    // luma_u16 included) is written, so an allocator refusal returns a typed error
    // leaving the output frame untouched, never partially mutated. Two scratches
    // can grow:
    //  1. the centered-siting full-width chroma (`chroma_full`), needed by ANY
    //     colour output (RGB / RGBA / HSV — the alpha plane never subsamples, so it
    //     is siting-independent); and
    //  2. the u8 RGB row buffer, reached exactly when a colour decode needs an RGB
    //     row but no caller RGB buffer is borrowable.
    // The later `upsample_420_chroma_center_h` reuses the already-sized buffer.
    if center_sited && (want_rgb || want_rgba || want_hsv) {
      reserve_420_chroma_full(chroma_full, w, h)?;
    }

    // Acquire the u8 RGB row buffer up front — before any caller-output write
    // below — so an allocator refusal in the HSV-only scratch path returns a
    // recoverable error rather than leaving a partially-written caller buffer
    // (luma / luma_u16). `rgb_row_buf_or_scratch` is the only other fallible
    // allocation on this direct path.
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
    // wire half-width U / V and reused by every colour decode below. Infallible —
    // the scratch was reserved above. The default / co-sited siting leaves it
    // `None`, so the fused 4:2:0 kernels upsample chroma in-register and the output
    // stays byte-identical.
    let centered = if center_sited && (want_rgb || want_rgba || want_hsv) {
      Some(upsample_420_chroma_center_h(
        chroma_full,
        row.u_half(),
        row.v_half(),
        w,
        use_simd,
      ))
    } else {
      None
    };

    // Freeze the phase on the first output-bearing row — AFTER the fallible scratch
    // reserves above have succeeded, so an `AllocationFailed` row stays retryable
    // (frozen stays unset); later rows are checked against it up top.
    if need_output && frozen_chroma_centered.is_none() {
      *frozen_chroma_centered = Some(center_sited);
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

    // HSV-only (no RGB / RGBA): convert the source Y/U/V straight to HSV via the
    // alpha-drop kernel — no source-width RGB scratch. Centered siting (#302)
    // routes through the 4:4:4 twin fed the full-width phase-0.5 chroma.
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

    // `need_rgb_kernel` / `rgb_row` were computed and (when needed) allocated at
    // the top, before any caller-output write.
    //
    // Direct-RGBA-only routes through the alpha-source-aware dispatcher (real
    // source α, NOT forced opaque). Centered siting feeds the 4:4:4 + source-alpha
    // kernel the full-width phase-0.5 chroma; the alpha plane is identical on both
    // paths.
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

    if want_rgba {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      // Strategy A+: RGB output already computed in rgb_row above (centered or
      // fused). Expand RGB → RGBA (fills α with 0xFF), then overwrite α slot from
      // the source alpha plane — avoids a second chroma kernel, so the centered RGB
      // row is reused as-is and only the real source alpha is layered on.
      expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
      crate::row::alpha_extract::copy_alpha_plane_u8(row.a(), rgba_row, w, use_simd);
    }

    Ok(())
  }
}

// ---- Yuva422p9 impl ---------------------------------------------------

impl<'a, R, const BE: bool> MixedSinker<'a, Yuva422p9<BE>, R> {
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
  /// Output is identical to [`MixedSinker<Yuv422p9>::with_rgb_u16`].
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

impl<R, const BE: bool> Yuva422p9Sink<BE> for MixedSinker<'_, Yuva422p9<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, Yuva422p9<BE>, R> {
  type Input<'r> = Yuva422p9Row<'r>;
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

  fn process(&mut self, row: Yuva422p9Row<'_>) -> Result<(), Self::Error> {
    if self.plan.is_some() {
      return yuva422p_high_bit_resample::<9, BE>(
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
        // Centered-siting (#302, RFC #238 S7) 4:4:4 twins fed the horizontally
        // reconstructed full-width chroma; α passed through unchanged.
        yuva444p9_to_rgba_row_endian,
        yuva444p9_to_rgba_u16_row_endian,
      );
    }
    yuva422p_high_bit_process::<9, BE, _, _>(
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

// ---- Yuva422p10 impl --------------------------------------------------

impl<'a, R, const BE: bool> MixedSinker<'a, Yuva422p10<BE>, R> {
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

impl<R, const BE: bool> Yuva422p10Sink<BE> for MixedSinker<'_, Yuva422p10<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, Yuva422p10<BE>, R> {
  type Input<'r> = Yuva422p10Row<'r>;
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

  fn process(&mut self, row: Yuva422p10Row<'_>) -> Result<(), Self::Error> {
    if self.plan.is_some() {
      return yuva422p_high_bit_resample::<10, BE>(
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
        // Centered-siting (#302, RFC #238 S7) 4:4:4 twins fed the horizontally
        // reconstructed full-width chroma; α passed through unchanged.
        yuva444p10_to_rgba_row_endian,
        yuva444p10_to_rgba_u16_row_endian,
      );
    }
    yuva422p_high_bit_process::<10, BE, _, _>(
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

// ---- Yuva422p12 impl --------------------------------------------------

impl<'a, R, const BE: bool> MixedSinker<'a, Yuva422p12<BE>, R> {
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

impl<R, const BE: bool> Yuva422p12Sink<BE> for MixedSinker<'_, Yuva422p12<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, Yuva422p12<BE>, R> {
  type Input<'r> = Yuva422p12Row<'r>;
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

  fn process(&mut self, row: Yuva422p12Row<'_>) -> Result<(), Self::Error> {
    if self.plan.is_some() {
      return yuva422p_high_bit_resample::<12, BE>(
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
        // Centered-siting (#302, RFC #238 S7) 4:4:4 twins fed the horizontally
        // reconstructed full-width chroma; α passed through unchanged.
        yuva444p12_to_rgba_row_endian,
        yuva444p12_to_rgba_u16_row_endian,
      );
    }
    yuva422p_high_bit_process::<12, BE, _, _>(
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

// ---- Yuva422p16 impl --------------------------------------------------

impl<'a, R, const BE: bool> MixedSinker<'a, Yuva422p16<BE>, R> {
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

impl<R, const BE: bool> Yuva422p16Sink<BE> for MixedSinker<'_, Yuva422p16<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, Yuva422p16<BE>, R> {
  type Input<'r> = Yuva422p16Row<'r>;
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

  fn process(&mut self, row: Yuva422p16Row<'_>) -> Result<(), Self::Error> {
    if self.plan.is_some() {
      return yuva422p_high_bit_resample::<16, BE>(
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
        // Centered-siting (#302, RFC #238 S7) 4:4:4 twins fed the horizontally
        // reconstructed full-width chroma; α passed through unchanged.
        yuva444p16_to_rgba_row_endian,
        yuva444p16_to_rgba_u16_row_endian,
      );
    }
    yuva422p_high_bit_process::<16, BE, _, _>(
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

// ---- Shared high-bit YUVA 4:2:2 process body --------------------------
//
// Same shape as the 4:2:0 sibling helper (`yuva420p_high_bit_process`)
// — half-width chroma per row — except this one is called from the
// 4:2:2 walker (vertical chroma index `r` rather than `r / 2`). The
// 4:2:0 helper would work too if we threaded the row dispatcher
// through, but Rust's monomorphization gives the same code either way
// and this keeps the symbol table organized by source family.
//
// Chroma siting (#302): the centered (phase-0.5) horizontal sitings reconstruct
// full-width chroma then decode via the FIVE 4:4:4 twin dispatchers; the default
// / co-sited path keeps the byte-identical fused 4:2:2 decode. 4:2:2 gates on
// `chroma_422_center_sited_h` and has no vertical phase (cf. the 4:2:0 sibling's
// bottom-sited lookback). The full-resolution source alpha plane is never
// subsampled, so it is siting-independent and passes through unchanged.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
#[cfg_attr(not(tarpaulin), inline(always))]
fn yuva422p_high_bit_process<const BITS: u32, const BE: bool, F: crate::SourceFormat, R>(
  sinker: &mut MixedSinker<'_, F, R>,
  idx: usize,
  y_row: &[u16],
  u_half_row: &[u16],
  v_half_row: &[u16],
  a_row: &[u16],
  matrix: crate::KernelMatrix,
  full_range: bool,
  y_slice: RowSlice,
  u_slice: RowSlice,
  v_slice: RowSlice,
  a_slice: RowSlice,
  // Each dispatch fn is the `_endian` variant — last `bool` is the runtime
  // `big_endian` flag the helper passes from `BE`. Function POINTERS (not generic
  // `Fn` bounds) so the 4:2:0 dispatcher and its 4:4:4 twin below — distinct fn
  // items of the SAME signature — coerce to one parameter type.
  rgb_dispatch: fn(&[u16], &[u16], &[u16], &mut [u8], usize, crate::KernelMatrix, bool, bool, bool),
  rgb_u16_dispatch: fn(
    &[u16],
    &[u16],
    &[u16],
    &mut [u16],
    usize,
    crate::KernelMatrix,
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
    crate::KernelMatrix,
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
    crate::KernelMatrix,
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
    crate::KernelMatrix,
    bool,
    bool,
    bool,
  ),
  // Chroma siting (#302): the centered (phase-0.5) 4:4:4 twins of the five
  // dispatchers above. `*_444` rgb / rgb_u16 / hsv are the alpha-drop 4:4:4
  // kernels (`yuv444pN_to_rgb*` / `_to_hsv`); `*_444` rgba / rgba_u16 are the
  // source-alpha 4:4:4 kernels (`yuva444pN_to_rgba*`) — the alpha plane is
  // full-resolution, so it is fed unchanged. Each takes FULL-width U / V (the
  // reconstructed chroma); their signatures match the matching half-chroma
  // dispatcher. They are invoked only on the centered sitings; the default path
  // never touches them.
  rgb_444_dispatch: fn(
    &[u16],
    &[u16],
    &[u16],
    &mut [u8],
    usize,
    crate::KernelMatrix,
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
    crate::KernelMatrix,
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
    crate::KernelMatrix,
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
    crate::KernelMatrix,
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
    crate::KernelMatrix,
    bool,
    bool,
    bool,
  ),
) -> Result<(), MixedSinkerError> {
  let w = sinker.width;
  let h = sinker.height;
  let use_simd = sinker.simd;
  // Chroma siting (#302): `Copy`, read before the field split-borrow below.
  let chroma_location = sinker.chroma_location.clone();

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
    // RFC #238 S7: the 4:2:2 chroma phase frozen on the first output row.
    frozen_chroma_centered,
    ..
  } = sinker;

  let want_rgb = rgb.is_some();
  let want_rgba = rgba.is_some();
  let want_hsv = hsv.is_some();
  let want_rgb_u16 = rgb_u16.is_some();
  let want_rgba_u16 = rgba_u16.is_some();

  // Repo-wide no-output invariant: a `process` call carrying NO output runs
  // NOTHING — no per-row offset arithmetic, no allocation, no state mutation.
  // Returning HERE, before the `idx * w` offsets below, also keeps the invariant
  // overflow-safe: a no-output call never ran an attach-time `w x h x 1`
  // validation, so `idx * w` could overflow `usize` on a 32-bit target with absurd
  // geometry; the guard skips that math (and the centered chroma reservation)
  // entirely. Mirrors the no-alpha `Yuv422p` sibling.
  let need_output = want_rgb
    || want_rgba
    || want_hsv
    || want_rgb_u16
    || want_rgba_u16
    || luma.is_some()
    || luma_u16.is_some();
  if !need_output {
    return Ok(());
  }

  let one_plane_start = idx * w;
  let one_plane_end = one_plane_start + w;

  // Chroma siting (#302): the centered horizontal sitings reconstruct chroma at
  // the phase-0.5 position then feed the 4:4:4 (+ source-alpha) kernels; the
  // default / co-sited path keeps the byte-identical fused 4:2:0 decode (4:2:2's
  // per-row chroma contract is identical — half-width chroma, one pair per Y
  // pair). 4:2:2 is subsampled horizontally only — no vertical blend or chroma
  // lookback (cf. the 4:2:0 sibling).
  let center_sited = chroma_422_center_sited_h(&chroma_location);

  // Per-frame chroma-siting freeze (RFC #238, mirroring the resample-path guard):
  // the first output-bearing row pins the phase; a later row whose siting flipped
  // would decode a mixture of centered and co-sited chroma into ONE frame, so
  // reject it here BEFORE any scratch reserve or output write. `begin_frame`'s
  // `reset_high_bit_yuva_streams` clears the freeze so the next frame may pick
  // either phase.
  if need_output
    && let Some(frozen) = *frozen_chroma_centered
    && frozen != center_sited
  {
    return Err(MixedSinkerError::ChromaSitingChanged(
      ChromaSitingChanged::new(idx),
    ));
  }

  // HSV-without-RGB-or-RGBA goes through the direct alpha-drop `hsv_dispatch`
  // kernel (no source-width RGB scratch). HSV is colour-only — the source alpha
  // plane is dropped. See `yuva420p_high_bit_process` for the full routing
  // rationale.
  let want_hsv_direct = want_hsv && !want_rgb && !want_rgba;
  let need_rgb_kernel = want_rgb || (want_hsv && want_rgba);

  // Atomicity preflight (#302, cf. the crate's #180 resample fix): reserve EVERY
  // fallible row scratch this identity row needs BEFORE any output row (luma /
  // luma_u16 included) is written, so an allocator refusal returns a typed error
  // leaving the output frame untouched, never partially mutated. Two scratches can
  // grow: the centered-siting full-width `u16` chroma (`chroma_full_u16`), needed
  // by ANY colour output (u8 OR u16 RGB / RGBA / HSV — the alpha plane is
  // full-resolution, so siting-independent); and the u8 RGB row scratch (below).
  // The later `upsample_420_chroma_center_h_u16` reuses the already-sized buffer.
  let need_centered_chroma =
    center_sited && (want_rgb || want_rgba || want_hsv || want_rgb_u16 || want_rgba_u16);
  if need_centered_chroma {
    reserve_420_chroma_full_u16(chroma_full_u16, w, h)?;
  }

  // Acquire the u8 RGB row buffer up front — before any caller-output write below
  // — so an allocator refusal in the HSV-only scratch path returns a recoverable
  // error rather than a partially-written caller buffer. See
  // `yuva420p_high_bit_process` for the full rationale.
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
  // wire-format half-width U / V and reused by every colour decode (u16 and u8).
  // Infallible — the scratch was reserved above. The default / co-sited siting
  // leaves it `None`, so the fused 4:2:2 kernels upsample chroma in-register and
  // the output stays byte-identical.
  let centered = if need_centered_chroma {
    Some(upsample_420_chroma_center_h_u16::<BITS>(
      chroma_full_u16,
      u_half_row,
      v_half_row,
      w,
      BE,
      use_simd,
    ))
  } else {
    None
  };

  // Freeze the phase on the first output-bearing row — AFTER the fallible scratch
  // reserves above have succeeded, so an `AllocationFailed` row stays retryable
  // (frozen stays unset); later rows are checked against it up top.
  if need_output && frozen_chroma_centered.is_none() {
    *frozen_chroma_centered = Some(center_sited);
  }

  // ---- luma (native Y; luma narrows `>> (BITS - 8)`, luma_u16 is the
  // host-native logical Y) — see `yuva420p_high_bit_process`. -----------
  if luma.is_some() || luma_u16.is_some() {
    let mut luma_row = luma
      .as_deref_mut()
      .map(|b| &mut b[one_plane_start..one_plane_end]);
    let mut luma_u16_row = luma_u16
      .as_deref_mut()
      .map(|b| &mut b[one_plane_start..one_plane_end]);
    // Mask each decoded Y to the source's native depth `(1 << BITS) - 1` (a
    // no-op at `BITS = 16`). `Yuva422p*Frame::try_new` is geometry-only, so a
    // malformed-but-accepted frame can carry out-of-range Y (e.g. `0x1000` at
    // 12-bit); without this mask `luma_u16` would publish that raw value (and the
    // 8-bit luma shift the wrapped value), inconsistent with the
    // `(1 << BITS) - 1`-masked Y the RGB/RGBA row kernels decode from the same
    // row. Mirrors the high-bit `Yuva420p` sibling — dirty-upper-bit
    // sanitization covers EVERY high-bit output (luma_u16 + 8-bit luma + chroma).
    let sample_mask = ((1u32 << BITS) - 1) as u16;
    for (i, &s) in y_row.iter().enumerate().take(w) {
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
  // RGB/RGBA outputs already ran above. Centered siting (#302) routes through the
  // 4:4:4 twin fed the full-width phase-0.5 chroma.
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

  // Direct-RGBA-only (real source α, NOT forced opaque). Centered siting feeds the
  // 4:4:4 + source-alpha kernel the full-width phase-0.5 chroma; the alpha plane is
  // identical on both paths.
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

// ---- Shared high-bit YUVA 4:2:2 resample-routing body -----------------
//
// The non-identity plan branch for the 9 / 10 / 12 / 16-bit YUVA 4:2:2
// sinks — the 4:2:2 sibling of the 4:2:0 routing helper. Per-row chroma is
// half-width (identical to 4:2:0), so the same half-chroma `_to_rgba` /
// `_to_rgba_u16` kernels apply; the 4:2:0-vs-4:2:2 difference (vertical
// chroma index `r / 2` vs `r`) is owned by the walker. The `Area` arm routes
// through the shared packed-YUVA area tail with THREE independent binnings:
// u8 colour, the **independent** native u16 colour (never a narrowing of the
// u8 bin), and the **low-packed** native-Y luma (`deinterleave_y_high_bit_masked`
// — host-native copy masked to `(1 << BITS) - 1` so dirty upper bits in a
// malformed-but-accepted Y are sanitized BEFORE binning, matching the masked Y
// the colour kernels decode and the identity luma path; planar YUVA Y is
// low-packed, so luma is `binned_Y >> (BITS - 8)`, NOT the semi-planar
// `>> (16 - BITS)` de-pack). The `Filter` arm routes the SAME converted RGBA /
// native-Y through the signed-coefficient filter tail (`NATIVE_LUMA_U8 = false`,
// the u16-luma branch — native Y is u16), straight alpha only.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
#[cfg_attr(not(tarpaulin), inline(always))]
fn yuva422p_high_bit_resample<const BITS: u32, const BE: bool>(
  sinker: &mut MixedSinker<'_, impl crate::SourceFormat, impl Sized>,
  idx: usize,
  y_row: &[u16],
  u_half_row: &[u16],
  v_half_row: &[u16],
  a_row: &[u16],
  matrix: crate::KernelMatrix,
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
    crate::KernelMatrix,
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
    crate::KernelMatrix,
    bool,
    bool,
    bool,
  ),
  // Chroma siting (#302, RFC #238 S7): the centered (phase-0.5) 4:4:4 twins of the
  // two source-alpha dispatchers above (`yuva444pN_to_rgba*`). Each takes
  // FULL-width U / V (the horizontally reconstructed chroma) with the α plane fed
  // unchanged; their signatures match the matching 4:2:2 dispatcher. They are
  // invoked ONLY on the centered horizontal sitings; the co-sited / default path
  // keeps the byte-identical fused 4:2:2 decode.
  rgba_444_dispatch: fn(
    &[u16],
    &[u16],
    &[u16],
    &[u16],
    &mut [u8],
    usize,
    crate::KernelMatrix,
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
    crate::KernelMatrix,
    bool,
    bool,
    bool,
  ),
) -> Result<(), MixedSinkerError> {
  let w = sinker.width;
  let h = sinker.height;
  let use_simd = sinker.simd;
  // Chroma siting (#302): `Copy`, read before the field split-borrow below.
  let chroma_location = sinker.chroma_location.clone();

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
  // RFC #238 S7 — 4:2:2 HORIZONTAL chroma siting for the high-bit alpha-bearing
  // source. The centered group (`chroma_422_center_sited_h` — chroma at `+0.5`
  // luma horizontally) reconstructs full-width chroma then feeds the 4:4:4
  // (+ source-alpha) twins; the co-sited / unspecified group is phase 0 (today's
  // fused 4:2:2 decode, byte-identical). 4:2:2 is subsampled horizontally only —
  // no vertical phase. The full-resolution α plane is never subsampled, so it is
  // siting-independent and passes through the 4:4:4 twins UNCHANGED.
  let center_sited = chroma_422_center_sited_h(&chroma_location);
  // A colour output drives the centered chroma reconstruction (u8 OR native u16);
  // a luma-only row bins the native Y (siting-independent) and never reconstructs
  // chroma, so it stays on the co-sited fused arm.
  let want_color =
    rgb.is_some() || rgba.is_some() || hsv.is_some() || rgb_u16.is_some() || rgba_u16.is_some();
  // The EXACT set the packed tails treat as output-bearing (colour u8 / u16 or
  // luma); gates the siting freeze so a no-output call (which consumes no stream
  // state) freezes nothing. MUST include the u16 colour outputs — a u16-only sink
  // still advances a stream and must freeze the siting.
  let need_output = want_color || luma.is_some() || luma_u16.is_some();
  // Freeze the effective horizontal siting on the first output-bearing row: a later
  // row observing a different phase — in sequence or not — would bin a mixture of
  // co-sited and centered chroma, so it is rejected here before any reconstruction.
  // The matching SET rides each accepted output row below.
  if need_output
    && let Some(frozen) = *frozen_chroma_centered
    && frozen != center_sited
  {
    return Err(MixedSinkerError::ChromaSitingChanged(
      ChromaSitingChanged::new(idx),
    ));
  }
  // The span kind picks the engine (mirrors the 8-bit `Yuva422p` and packed
  // `Vuya`): `Area` bins (the alpha-aware tail — premultiplied colour binned
  // premultiplied then un-premultiplied) at native precision; `Filter` runs
  // the signed-coefficient filter on the same converted RGBA (straight alpha
  // only). The high-bit native Y is genuinely `u16`, so the filter route uses
  // `NATIVE_LUMA_U8 = false` — the same u16-luma branch `Vuya` / `Vuyx` use:
  // luma rides the `u16` filter stream over the de-interleaved native Y. The
  // filter tail clamps every sub-16-bit colour AND native-Y overshoot to the
  // source's native max `(1 << BITS) - 1` (a value no-op at `BITS = 16`),
  // matching the in-range area path. A premultiplied `Filter` plan has no
  // analogue (the engine cannot un-premultiply), so it is routed to the area
  // tail, which surfaces the typed `UnsupportedFilter`.
  let r = match plan.kind() {
    crate::resample::SpanKind::Area if center_sited && want_color => {
      // Centered area: reconstruct full-width chroma and bin the 4:4:4-decoded
      // RGBA — but ONLY after the resample preflight (frozen-output + sequence), so
      // an out-of-sequence / rejected row is caught before the chroma reservation
      // (#180). `packed_yuva444_resample` re-runs the idempotent preflight. The α
      // plane is full-resolution and siting-independent: it flows into the 4:4:4
      // converters UNCHANGED.
      let expected = if rgb.is_some() || rgba.is_some() || hsv.is_some() {
        rgba_stream.as_ref().map_or(0, |s| s.next_y())
      } else {
        rgba_stream_u16.as_ref().map_or(0, |s| s.next_y())
      };
      match super::resample_preflight_check_only(
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
            use_simd,
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
    }
    crate::resample::SpanKind::Area => packed_yuva444_resample::<BITS>(
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
    ),
    crate::resample::SpanKind::Filter if alpha_mode.is_premultiplied() => {
      // Premultiplied + filter has no analogue: route to the area tail with
      // the filter plan so it returns the typed `UnsupportedFilter`.
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
    crate::resample::SpanKind::Filter if center_sited && want_color => {
      // Reject a multi-kernel (BICUBLIN) filter plan BEFORE the centered
      // preflight/reserve below: `packed_yuva444_filter_resample`'s first act is
      // this same check, so without hoisting it a BICUBLIN plan would freeze the
      // output set and reserve/reconstruct chroma before being rejected as
      // `UnsupportedFilter` (#180 reject-before-allocation). Idempotent — the
      // delegate re-runs it.
      plan.ensure_single_kernel_filter()?;
      // Centered filter: reconstruct full-width chroma and filter the
      // 4:4:4-decoded RGBA — but ONLY after the filter preflight (frozen-output +
      // sequence), so an out-of-sequence / rejected row is caught before the chroma
      // reservation (#180). The α plane is siting-independent and flows into the
      // 4:4:4 converters UNCHANGED.
      let expected = if rgb.is_some() || rgba.is_some() || hsv.is_some() {
        rgba_filter_stream.as_ref().map_or(0, |s| s.next_y())
      } else {
        rgba_filter_stream_u16.as_ref().map_or(0, |s| s.next_y())
      };
      match super::resample_preflight_check_only(
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
            use_simd,
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
    }
    crate::resample::SpanKind::Filter => packed_yuva444_filter_resample::<BITS, false, false>(
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
    ),
  };
  // RFC #238 S7: freeze the horizontal 4:2:2 chroma siting on the first ACCEPTED
  // output-bearing resampled row (a rejected row returns Err above and leaves it
  // unset for a corrected retry). Both packed tails carry one plan kind for the
  // frame, so there is no native/row-stage route to co-freeze here.
  if r.is_ok() && need_output && frozen_chroma_centered.is_none() {
    *frozen_chroma_centered = Some(center_sited);
  }
  r
}
