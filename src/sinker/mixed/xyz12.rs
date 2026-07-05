//! Sinker impl for the Tier 12 (DCP / `Xyz12`) **source** format.
//!
//! Each pixel is `3 x u16` in `X, Y, Z` order, **high-bit-packed** per
//! FFmpeg `AV_PIX_FMT_XYZ12LE/BE` (active 12 bits in `[15:4]`, low 4
//! bits zero). The conversion chain is the heaviest in colconv: SMPTE
//! ST 428-1 §8 inverse OETF → 3x3 matrix to one of three target gamuts
//! → sRGB-shape OETF → integer narrow.
//!
//! Output paths:
//!
//! - `with_rgb` / `with_rgba` — full pipeline → packed u8 RGB / RGBA
//!   (alpha = `0xFF`).
//! - `with_rgb_u16` / `with_rgba_u16` — full pipeline, full-range
//!   `[0, 1] x 65535` scaling. RGBA alpha = `0xFFFF`.
//! - `with_rgb_f32` — **lossless** linear-RGB f32 (matrix applied,
//!   OETF skipped, no clamp; out-of-gamut negative R/G/B and HDR > 1
//!   values preserved bit-exact).
//! - `with_xyz_f32` — lossless **linear XYZ** f32 (only step-1 inverse
//!   OETF applied; no matrix, no gamma, no clamp).
//! - `with_rgb_f16` / `with_rgba_f16` — full pipeline + clamp `[0, 1]`
//!   + IEEE-754 RNE narrow to f16; alpha = `1.0` for the rgba variant.
//! - `with_luma` / `with_luma_u16` — staged through u8 RGB scratch,
//!   then `xyz12_rgb_to_luma_row` / `xyz12_rgb_to_luma_u16_row` with
//!   the gamut-derived Q15 weights (BT.709 for Rec709,
//!   `(6865, 23645, 2258)` for DciP3 theatrical, BT.2020Ncl for
//!   Rec2020 — see [`crate::source::luma_weights_q15_for_gamut`]). The
//!   DciP3 triple is not the BT.709 one: re-using BT.709 would bias luma
//!   for saturated content under the theatrical DCI-white target.
//! - `with_hsv` — same staging, then `rgb_to_hsv_row`.

use super::{
  InsufficientBuffer, MixedSinker, MixedSinkerError, RowIndexOutOfRange, RowShapeMismatch,
  RowSlice, check_dimensions_match, frozen_outputs_check, resample_preflight_check_only,
  rgb_row_buf_or_scratch, rgba_plane_row_slice, rgba_u16_plane_row_slice, source_rgb_scratch,
  source_xyz_f32_scratch, xyz12_resample_emit,
};
use crate::{
  PixelSink,
  row::{
    rgb_to_hsv_row, xyz12_rgb_to_luma_row, xyz12_rgb_to_luma_u16_row, xyz12_to_rgb_f16_row,
    xyz12_to_rgb_f32_row, xyz12_to_rgb_row, xyz12_to_rgb_u16_row, xyz12_to_rgba_f16_row,
    xyz12_to_rgba_row, xyz12_to_rgba_u16_row, xyz12_to_xyz_f32_row,
  },
  source::{Xyz12, Xyz12Row, Xyz12Sink},
};

// ---- Xyz12<BE> impl ----------------------------------------------------

impl<'a, R, const BE: bool> MixedSinker<'a, Xyz12<BE>, R> {
  /// Attaches a packed **8-bit** RGBA output buffer. Alpha forced to
  /// `0xFF` (no source alpha plane).
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

  /// Attaches a `u16` RGB output buffer (`width x height x 3`
  /// elements). Source XYZ flows through the full pipeline; the
  /// resulting linear RGB is clamped to `[0, 1]` and **scaled to the
  /// full u16 range** (x65535).
  ///
  /// # Naming consistency note
  ///
  /// Other source families' `with_rgb_u16` accessor preserves the
  /// source's *native integer precision* in a u16 carrier. The `Xyz12`
  /// variant has no native RGB precision — it is always derived
  /// through the gamut matrix — so it instead applies full-range
  /// scaling, matching the [`Rgbf32`](crate::source::Rgbf32) and
  /// [`Rgbf16`](crate::source::Rgbf16) divergence.
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

  /// Attaches a `u16` RGBA output buffer. Same `[0, 1]` x 65535 scaling
  /// as [`with_rgb_u16`](Self::with_rgb_u16); alpha forced to `0xFFFF`.
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

  /// Attaches a `u16` luma output buffer. Y' is computed at u8
  /// precision (matching `with_luma`'s output) and zero-extended to
  /// `u16` — same convention as the packed-YUV `with_luma_u16` family.
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

  /// Attaches a packed f32 RGB output buffer for **lossless linear
  /// RGB** output. The XYZ → RGB matrix is applied; the OETF and clamp
  /// are skipped. Out-of-gamut negative R/G/B and HDR > 1 values are
  /// preserved bit-exact.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgb_f32(mut self, buf: &'a mut [f32]) -> Result<Self, MixedSinkerError> {
    self.set_rgb_f32(buf)?;
    Ok(self)
  }
  /// In-place variant of [`with_rgb_f32`](Self::with_rgb_f32).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgb_f32(&mut self, buf: &'a mut [f32]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_elems(3)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::InsufficientRgbF32Buffer(
        InsufficientBuffer::new(expected, buf.len()),
      ));
    }
    self.rgb_f32 = Some(buf);
    Ok(self)
  }

  /// Attaches a packed f32 XYZ output buffer for **lossless linear
  /// XYZ** pass-through. Only the SMPTE ST 428-1 §8 inverse OETF
  /// (step 1) is applied; no gamut matrix, no gamma, no clamp.
  /// Useful for callers that want to do their own gamut conversion
  /// downstream.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_xyz_f32(mut self, buf: &'a mut [f32]) -> Result<Self, MixedSinkerError> {
    self.set_xyz_f32(buf)?;
    Ok(self)
  }
  /// In-place variant of [`with_xyz_f32`](Self::with_xyz_f32).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_xyz_f32(&mut self, buf: &'a mut [f32]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_elems(3)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::InsufficientXyzF32Buffer(
        InsufficientBuffer::new(expected, buf.len()),
      ));
    }
    self.xyz_f32 = Some(buf);
    Ok(self)
  }

  /// Attaches a packed f16 RGB output buffer. Full pipeline; output
  /// values clamped to `[0, 1]` before the IEEE-754 RNE narrow to f16.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgb_f16(mut self, buf: &'a mut [half::f16]) -> Result<Self, MixedSinkerError> {
    self.set_rgb_f16(buf)?;
    Ok(self)
  }
  /// In-place variant of [`with_rgb_f16`](Self::with_rgb_f16).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgb_f16(&mut self, buf: &'a mut [half::f16]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_elems(3)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::InsufficientRgbF16Buffer(
        InsufficientBuffer::new(expected, buf.len()),
      ));
    }
    self.rgb_f16 = Some(buf);
    Ok(self)
  }

  /// Attaches a packed f16 RGBA output buffer (alpha = `1.0`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgba_f16(mut self, buf: &'a mut [half::f16]) -> Result<Self, MixedSinkerError> {
    self.set_rgba_f16(buf)?;
    Ok(self)
  }
  /// In-place variant of [`with_rgba_f16`](Self::with_rgba_f16).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgba_f16(&mut self, buf: &'a mut [half::f16]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_elems(4)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::InsufficientRgbaF16Buffer(
        InsufficientBuffer::new(expected, buf.len()),
      ));
    }
    self.rgba_f16 = Some(buf);
    Ok(self)
  }
}

// `Xyz12` is in the engine gate (the `#145`/`#146` cascade widened the
// shared `AreaStream` / `FrozenOutputs` cfg to `xyz`), so the streaming
// engine, the linear-XYZ `f32` area + filter streams, and the staging
// scratch are always compiled in whenever this `xyz`-gated module is. No
// engine fence is needed: the sink accepts any resampler `R`, takes the
// direct conversion under the identity plan, and routes a non-identity
// plan through the linear-light tail — the area engine for an `Area`
// plan, the separable filter engine for a `Filter` plan
// (`xyz12_resample_*`).

impl<R, const BE: bool> Xyz12Sink<BE> for MixedSinker<'_, Xyz12<BE>, R> {}

impl<R, const BE: bool> MixedSinker<'_, Xyz12<BE>, R> {
  /// Runs the identity (no-resample) output derivation for one source
  /// row over every attached output buffer.
  ///
  /// This is the parity oracle: the byte-exact output the area-resample
  /// tail must reproduce for an identity plan. It assumes the row shape
  /// and index are already validated by the caller.
  fn xyz12_process_direct(
    &mut self,
    row: Xyz12Row<'_, BE>,
    use_simd: bool,
  ) -> Result<(), MixedSinkerError> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let target_gamut = row.target_gamut();

    let Self {
      rgb,
      rgb_u16,
      rgb_f32,
      rgb_f16,
      rgba,
      rgba_u16,
      rgba_f16,
      luma,
      luma_u16,
      hsv,
      xyz_f32,
      rgb_scratch,
      ..
    } = self;
    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;
    let xyz_in = row.xyz();

    // 1. Lossless XYZ pass-through (independent of all other paths).
    if let Some(buf) = xyz_f32.as_deref_mut() {
      let f32_start = one_plane_start * 3;
      let f32_end = one_plane_end * 3;
      xyz12_to_xyz_f32_row::<BE>(xyz_in, &mut buf[f32_start..f32_end], w, use_simd);
    }

    // 2. Lossless linear RGB f32 (skips OETF, skips clamp).
    if let Some(buf) = rgb_f32.as_deref_mut() {
      let f32_start = one_plane_start * 3;
      let f32_end = one_plane_end * 3;
      xyz12_to_rgb_f32_row::<BE>(
        xyz_in,
        &mut buf[f32_start..f32_end],
        w,
        target_gamut,
        use_simd,
      );
    }

    // 3. f16 RGB (gamma-encoded, clamped).
    if let Some(buf) = rgb_f16.as_deref_mut() {
      let f16_start = one_plane_start * 3;
      let f16_end = one_plane_end * 3;
      xyz12_to_rgb_f16_row::<BE>(
        xyz_in,
        &mut buf[f16_start..f16_end],
        w,
        target_gamut,
        use_simd,
      );
    }

    // 4. f16 RGBA.
    if let Some(buf) = rgba_f16.as_deref_mut() {
      let f16_start = one_plane_start * 4;
      let f16_end = one_plane_end * 4;
      xyz12_to_rgba_f16_row::<BE>(
        xyz_in,
        &mut buf[f16_start..f16_end],
        w,
        target_gamut,
        use_simd,
      );
    }

    // 5. u16 RGB.
    if let Some(buf) = rgb_u16.as_deref_mut() {
      let u16_start = one_plane_start * 3;
      let u16_end = one_plane_end * 3;
      xyz12_to_rgb_u16_row::<BE>(
        xyz_in,
        &mut buf[u16_start..u16_end],
        w,
        target_gamut,
        use_simd,
      );
    }

    // 6. u16 RGBA.
    if let Some(buf) = rgba_u16.as_deref_mut() {
      let rgba_row = rgba_u16_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      xyz12_to_rgba_u16_row::<BE>(xyz_in, rgba_row, w, target_gamut, use_simd);
    }

    // 7. u8 RGBA standalone fast path — direct from XYZ when nothing
    // else needs the staged u8 RGB row.
    let want_rgba_u8 = rgba.is_some();
    let want_rgb_u8 = rgb.is_some();
    let want_luma_u8 = luma.is_some();
    let want_luma_u16 = luma_u16.is_some();
    let want_hsv = hsv.is_some();
    let need_u8_rgb = want_rgb_u8 || want_luma_u8 || want_luma_u16 || want_hsv;

    if want_rgba_u8 && !need_u8_rgb {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      xyz12_to_rgba_row::<BE>(xyz_in, rgba_row, w, target_gamut, use_simd);
      return Ok(());
    }

    if !need_u8_rgb && !want_rgba_u8 {
      return Ok(());
    }

    // 8. Stage the u8 RGB scratch row (own buffer or shared with
    // user's RGB output), feed downstream luma / HSV.
    let rgb_row = rgb_row_buf_or_scratch(
      rgb.as_deref_mut(),
      rgb_scratch,
      one_plane_start,
      one_plane_end,
      w,
      h,
    )?;
    xyz12_to_rgb_row::<BE>(xyz_in, rgb_row, w, target_gamut, use_simd);

    if let Some(luma) = luma.as_deref_mut() {
      xyz12_rgb_to_luma_row(
        rgb_row,
        &mut luma[one_plane_start..one_plane_end],
        w,
        row.luma_q15(),
        use_simd,
      );
    }

    if let Some(luma_buf) = luma_u16.as_deref_mut() {
      xyz12_rgb_to_luma_u16_row(
        rgb_row,
        &mut luma_buf[one_plane_start..one_plane_end],
        w,
        row.luma_q15(),
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

    // 9. u8 RGBA combined-with-RGB path — direct from XYZ source so we
    // do not pay for an `expand_rgb_to_rgba_row` pass over the staged
    // u8 RGB row.
    if let Some(buf) = rgba.as_deref_mut() {
      let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      xyz12_to_rgba_row::<BE>(xyz_in, rgba_row, w, target_gamut, use_simd);
    }

    Ok(())
  }
}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, Xyz12<BE>, R> {
  type Input<'r> = Xyz12Row<'r, BE>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    if let Some(stream) = self.xyz_stream_f32.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.xyz_filter_stream_f32.as_mut() {
      stream.reset();
    }
    self.resample_outputs = None;
    Ok(())
  }

  fn process(&mut self, row: Xyz12Row<'_, BE>) -> Result<(), Self::Error> {
    let w = self.width;
    let idx = row.row();
    let use_simd = self.simd;
    let target_gamut = row.target_gamut();
    let luma_q15 = row.luma_q15();

    if row.xyz().len() != w * 3 {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Xyz12Packed,
        idx,
        w * 3,
        row.xyz().len(),
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
      rgb_f32,
      rgb_f16,
      rgba,
      rgba_u16,
      rgba_f16,
      luma,
      luma_u16,
      hsv,
      xyz_f32,
      rgb_scratch,
      xyz_scratch_f32,
      xyz_stream_f32,
      xyz_filter_stream_f32,
      resample_outputs,
      plan,
      ..
    } = self;

    // Non-identity plan: convert the wire row to source-width LINEAR XYZ
    // f32 (inverse-OETF only, pre-matrix), bin it in float (the area
    // mean is thus taken in linear light), then derive every attached
    // output from each finalized binned linear-XYZ row. `xyz_f32` copies
    // the binned XYZ; every other output applies the direct DCP path's
    // matrix (+ gamma/clamp/narrow) to it — the matrix commutes with the
    // bin because both are linear, so `M . mean(xyz) == mean(M . xyz)`.
    if let Some(plan) = plan.as_ref() {
      // The span kind picks the engine — linear-light float area or
      // signed-coefficient filter. Both bin the SAME staged linear-XYZ
      // f32 row (inverse-OETF only, pre-matrix) and feed the SAME emit;
      // the matrix + gamma + narrow run per finalized output row. The
      // `rgb_f32` / `xyz_f32` outputs stay full-range (HDR > 1 and
      // out-of-gamut negatives preserved) and the integer / f16 outputs
      // clamp `[0, 1]` in their own narrows, so a filter overshoot needs
      // no native-depth clamp.
      let ow = plan.out_w();
      let need_any = rgb.is_some()
        || rgba.is_some()
        || luma.is_some()
        || luma_u16.is_some()
        || rgb_u16.is_some()
        || rgba_u16.is_some()
        || rgb_f32.is_some()
        || xyz_f32.is_some()
        || rgb_f16.is_some()
        || rgba_f16.is_some()
        || hsv.is_some();
      // Sequence-counter row for the shared L1 preflight (the kind-appropriate
      // stream is the sole per-row stream), or `None` when no output is attached
      // so the preflight short-circuits to a no-op.
      let expected = if need_any {
        Some(match plan.kind() {
          crate::resample::SpanKind::Area => xyz_stream_f32.as_ref().map_or(0, |s| s.next_y()),
          crate::resample::SpanKind::Filter => {
            xyz_filter_stream_f32.as_ref().map_or(0, |s| s.next_y())
          }
        })
      } else {
        None
      };
      // Compare-only preflight (NO commit): the output-set freeze commits only
      // AFTER the fallible stream alloc + scratch growth below succeed (RFC #238
      // tail-collapse), so a first-row allocation failure leaves `resample_outputs`
      // uncommitted — retryable with a changed output attachment.
      if let core::ops::ControlFlow::Break(()) = resample_preflight_check_only(
        resample_outputs,
        luma,
        luma_u16,
        rgb,
        rgba,
        rgb_u16,
        rgba_u16,
        rgb_f32,
        &None,
        xyz_f32,
        rgb_f16,
        rgba_f16,
        hsv,
        &None,
        expected,
        idx,
      )? {
        return Ok(());
      }
      // The u8 RGB / luma / luma_u16 / hsv outputs stage through the emit's own
      // out-width narrow scratch; pre-grow it (and the source-width linear-XYZ
      // staging) BEFORE the freeze so every fallible pre-feed alloc precedes the
      // commit and NOTHING fallible runs post-freeze. The predicate is the one
      // `xyz12_resample_emit` uses, so the pre-grow and its re-grow cannot drift.
      let need_narrow = rgb.is_some() || luma.is_some() || luma_u16.is_some() || hsv.is_some();
      // Build the missing kind-appropriate stream into a LOCAL and grow every
      // scratch with NO field mutation, so the stream field insert and the freeze
      // commit TOGETHER only after they all succeed (a scratch OOM leaves the
      // stream field `None` AND `resample_outputs` uncommitted, never a partial
      // commit).
      return match plan.kind() {
        crate::resample::SpanKind::Area => {
          let new_stream = if xyz_stream_f32.is_none() {
            let stream =
              crate::resample::AreaStream::new(plan.h(), plan.v(), plan.src_w(), plan.src_h(), 3)?;
            Some(crate::resample::try_box(stream).map_err(|_| {
              MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
                crate::resample::PlanGeometry::new(
                  plan.src_w(),
                  plan.src_h(),
                  plan.out_w(),
                  plan.out_h(),
                ),
              ))
            })?)
          } else {
            None
          };
          source_xyz_f32_scratch(xyz_scratch_f32, w, plan)?;
          if need_narrow {
            source_rgb_scratch(rgb_scratch, ow, plan)?;
          }
          if let Some(s) = new_stream {
            *xyz_stream_f32 = Some(s);
          }
          frozen_outputs_check(
            resample_outputs,
            luma,
            luma_u16,
            rgb,
            rgba,
            rgb_u16,
            rgba_u16,
            rgb_f32,
            &None,
            xyz_f32,
            rgb_f16,
            rgba_f16,
            hsv,
            &None,
            idx,
          )?;
          let stream = xyz_stream_f32.as_mut().expect("created above");
          let src_xyz = source_xyz_f32_scratch(xyz_scratch_f32, w, plan)?;
          xyz12_to_xyz_f32_row::<BE>(row.xyz(), src_xyz, w, use_simd);
          xyz12_resample_emit(
            stream,
            plan,
            rgb,
            rgba,
            luma,
            luma_u16,
            rgb_u16,
            rgba_u16,
            rgb_f32,
            xyz_f32,
            rgb_f16,
            rgba_f16,
            hsv,
            src_xyz,
            rgb_scratch,
            target_gamut,
            luma_q15,
            idx,
            use_simd,
          )
        }
        crate::resample::SpanKind::Filter => {
          // Single-kernel filter: reject a BICUBLIN plan before any allocation
          // (its chroma windows are read only by the `Yuv420p` per-plane route).
          plan.ensure_single_kernel_filter()?;
          let new_stream = if xyz_filter_stream_f32.is_none() {
            let (fh, fv) = (
              plan
                .filter_h()
                .expect("filter plan carries horizontal windows"),
              plan
                .filter_v()
                .expect("filter plan carries vertical windows"),
            );
            let stream = crate::resample::FilterStream::new(fh, fv, plan.src_w(), plan.src_h(), 3)?;
            Some(crate::resample::try_box(stream).map_err(|_| {
              MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
                crate::resample::PlanGeometry::new(
                  plan.src_w(),
                  plan.src_h(),
                  plan.out_w(),
                  plan.out_h(),
                ),
              ))
            })?)
          } else {
            None
          };
          source_xyz_f32_scratch(xyz_scratch_f32, w, plan)?;
          if need_narrow {
            source_rgb_scratch(rgb_scratch, ow, plan)?;
          }
          if let Some(s) = new_stream {
            *xyz_filter_stream_f32 = Some(s);
          }
          frozen_outputs_check(
            resample_outputs,
            luma,
            luma_u16,
            rgb,
            rgba,
            rgb_u16,
            rgba_u16,
            rgb_f32,
            &None,
            xyz_f32,
            rgb_f16,
            rgba_f16,
            hsv,
            &None,
            idx,
          )?;
          let stream = xyz_filter_stream_f32.as_mut().expect("created above");
          let src_xyz = source_xyz_f32_scratch(xyz_scratch_f32, w, plan)?;
          xyz12_to_xyz_f32_row::<BE>(row.xyz(), src_xyz, w, use_simd);
          xyz12_resample_emit(
            stream,
            plan,
            rgb,
            rgba,
            luma,
            luma_u16,
            rgb_u16,
            rgba_u16,
            rgb_f32,
            xyz_f32,
            rgb_f16,
            rgba_f16,
            hsv,
            src_xyz,
            rgb_scratch,
            target_gamut,
            luma_q15,
            idx,
            use_simd,
          )
        }
      };
    }

    self.xyz12_process_direct(row, use_simd)
  }
}
