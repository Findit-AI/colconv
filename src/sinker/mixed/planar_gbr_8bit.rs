//! Sinker impls for the planar **GBR** source family (Tier 10).
//!
//! Two formats covered:
//! - [`Gbrp`] (`AV_PIX_FMT_GBRP`) — three planes (G, B, R), no alpha.
//! - [`Gbrap`] (`AV_PIX_FMT_GBRAP`) — four planes (G, B, R, A), source α.
//!
//! Output paths:
//! - `with_rgb` — interleave G/B/R → packed `R, G, B` via the
//!   dedicated `gbr_to_rgb_row` SIMD/scalar kernel (no chroma matrix).
//! - `with_rgba` — for `Gbrp`: standalone path uses
//!   `gbr_to_rgba_opaque_row` (α = `0xFF`); combo path with `with_rgb`
//!   uses Strategy A (expand RGB → RGBA after the RGB kernel).
//!   For `Gbrap`: standalone uses `gbra_to_rgba_row` (real α from the
//!   source A plane); combo path with `with_rgb` uses Strategy A+
//!   (expand RGB → RGBA + α-overwrite from the source plane).
//! - `with_luma` / `with_luma_u16` — derived from RGB via the existing
//!   `rgb_to_luma_row` after a staged-RGB pass into `rgb_scratch`
//!   (matches the pattern used by [`super::Bgr24`] / [`super::Bgra`] etc.).
//! - `with_hsv` — derived from staged RGB via `rgb_to_hsv_row`
//!   (existing kernel — no new HSV variant).
//!
//! **Fused area-resample** (`with_resampler`): both `Gbrp` and `Gbrap`
//! are generic over `R: Resampler`. On a non-identity plan `Gbrp`
//! scatters its G/B/R planes into the source-width packed-RGB scratch and
//! feeds the shared transactional 3-channel packed-RGB resample helper
//! ([`packed_rgb_resample`](super::packed_rgb_resample) /
//! [`packed_rgb_resample_filter`](super::packed_rgb_resample_filter)) — the same
//! path the `Bgr24` / padding-byte sources take, so every output
//! (rgb, rgba, luma, luma_u16, hsv)
//! derives from the binned RGB and matches a direct conversion of the
//! pre-binned frame. `Gbrap` is **alpha-aware**: it de-interleaves its
//! G/B/R/A planes into the canonical source-width `R, G, B, A` row
//! (`gbra_to_rgba_row`) and feeds the 4-channel packed RGBA tail
//! ([`packed_rgba_resample`](super::packed_rgba_resample)) the
//! `Rgba` / `Bgra` / `Argb` / `Abgr` sources take, so resampled alpha is
//! a real area mean (not forced opaque) and — under
//! [`AlphaMode::Premultiplied`](super::AlphaMode::Premultiplied) — the
//! color is binned premultiplied. The per-format default is straight
//! alpha; a straight rgb-only sink (alpha dropped) keeps the 3-channel
//! path with no regression.
//!
//! 8-bit planar GBR has no `u16` output flavour — `with_rgb_u16` /
//! `with_rgba_u16` are not declared on these source impls (they'd be
//! identity passes from u8 source which doesn't justify the extra
//! API surface; high-bit-depth Gbrp9/10/12/14/16 get `with_rgb_u16`
//! when they land in Tier 10b).

use super::{
  InsufficientBuffer, MixedSinker, MixedSinkerError, RowIndexOutOfRange, RowShapeMismatch,
  RowSlice, check_dimensions_match, check_frozen_alpha_mode, packed_rgb_resample,
  packed_rgb_resample_filter, packed_rgba_filter_resample, packed_rgba_resample,
  rgb_row_buf_or_scratch, rgba_plane_row_slice,
};
use crate::{
  PixelSink,
  row::{
    expand_rgb_to_rgba_row, gbr_to_rgb_row, gbr_to_rgba_opaque_row, gbra_to_rgba_row,
    rgb_to_hsv_row, rgb_to_luma_row, rgb_to_luma_u16_row,
  },
  source::{Gbrap, GbrapRow, GbrapSink, Gbrp, GbrpRow, GbrpSink},
};

// ---- Gbrp impl ----------------------------------------------------------

impl<'a, R> MixedSinker<'a, Gbrp, R> {
  /// Attaches a packed **8-bit** RGBA output buffer. The 8-bit GBR
  /// source has no alpha channel, so every alpha byte is filled with
  /// constant `0xFF` (opaque).
  ///
  /// Returns `Err(InsufficientRgbaBuffer)` if `buf.len() < width x height
  /// x 4`, or `Err(GeometryOverflow)` on 32-bit overflow.
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

  /// Attaches a `u16` luma output buffer. Luma is derived from G/B/R
  /// via the standard `rgb_to_luma_row` kernel and zero-extended into
  /// `u16` (output range `[0, 255]` in u16 elements).
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

impl<R> GbrpSink for MixedSinker<'_, Gbrp, R> {}

impl<R> PixelSink for MixedSinker<'_, Gbrp, R> {
  type Input<'r> = GbrpRow<'r>;
  type Error = MixedSinkerError;

  #[cfg_attr(not(tarpaulin), inline(always))]
  fn kernel_matrix(&self) -> crate::KernelMatrix {
    self.kernel_matrix
  }
  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    if let Some(stream) = self.rgb_stream.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.rgb_filter_stream.as_mut() {
      stream.reset();
    }
    self.resample_outputs = None;
    Ok(())
  }

  fn process(&mut self, row: GbrpRow<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    // Defense-in-depth row-shape checks; the walker normally pre-
    // validates these via `begin_frame` + the per-row slice math.
    if row.g().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::GPlane,
        idx,
        w,
        row.g().len(),
      )));
    }
    if row.b().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::BPlane,
        idx,
        w,
        row.b().len(),
      )));
    }
    if row.r().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::RPlane,
        idx,
        w,
        row.r().len(),
      )));
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange(
        RowIndexOutOfRange::new(idx, self.height),
      ));
    }

    let Self {
      rgb,
      rgba,
      luma,
      luma_u16,
      hsv,
      rgb_scratch,
      plan,
      rgb_stream,
      rgb_filter_stream,
      resample_outputs,
      ..
    } = self;
    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;
    let g_in = row.g();
    let b_in = row.b();
    let r_in = row.r();

    // Non-identity plan: freeze the output set, then check stream
    // sequencing — both before touching the scratch — so a no-output
    // sink stays a no-op and an out-of-sequence row is rejected without
    // the source-width allocation/interleave. Only then scatter the
    // G/B/R planes into the source-width packed-RGB scratch and feed the
    // shared packed-RGB resample tail. luma_u16 derives through the same
    // tail, mirroring the direct path below for parity (RGBA alpha is
    // forced to 0xFF — the 3-plane source has no alpha). The plan's span
    // kind picks the engine — the integer area stream or the signed-
    // coefficient filter stream — both feed the one shared emit.
    if let Some(plan) = plan.as_ref() {
      // Scatter the G/B/R planes into the source-width packed-RGB scratch and
      // feed the shared transactional resample helper (RFC #238 tail-collapse);
      // the plan's span kind picks the engine (integer area / signed-coefficient
      // filter). RGBA alpha is forced to 0xFF (the 3-plane source has no alpha).
      return match plan.kind() {
        crate::resample::SpanKind::Area => packed_rgb_resample(
          rgb_stream,
          resample_outputs,
          rgb,
          rgba,
          luma,
          luma_u16,
          hsv,
          rgb_scratch,
          None,
          w,
          plan,
          idx,
          use_simd,
          row.matrix(),
          row.full_range(),
          |scratch| gbr_to_rgb_row(g_in, b_in, r_in, scratch, w, use_simd),
        ),
        crate::resample::SpanKind::Filter => packed_rgb_resample_filter(
          rgb_filter_stream,
          resample_outputs,
          rgb,
          rgba,
          luma,
          luma_u16,
          hsv,
          rgb_scratch,
          None,
          w,
          plan,
          idx,
          use_simd,
          row.matrix(),
          row.full_range(),
          |scratch| gbr_to_rgb_row(g_in, b_in, r_in, scratch, w, use_simd),
        ),
      };
    }

    // ---- Output mode resolution (Strategy A) -------------------------
    //
    // - RGBA-only (no RGB / luma / luma_u16 / HSV): use the dedicated
    //   `gbr_to_rgba_opaque_row` to write the 4-byte output without
    //   staging RGB first.
    // - Otherwise: stage the RGB row into the user's RGB buffer (or
    //   `rgb_scratch` when only luma/HSV/RGBA are requested), then
    //   derive luma / HSV / RGBA from the staged RGB.
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_luma = luma.is_some();
    let want_luma_u16 = luma_u16.is_some();
    let want_hsv = hsv.is_some();
    let need_rgb_buffer = want_rgb || want_rgba || want_luma || want_luma_u16 || want_hsv;

    if want_rgba && !need_rgb_buffer_other(want_rgb, want_luma, want_luma_u16, want_hsv) {
      // RGBA-only — direct write, skip the staged RGB scratch.
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      gbr_to_rgba_opaque_row(g_in, b_in, r_in, rgba_row, w, use_simd);
      return Ok(());
    }

    if !need_rgb_buffer {
      return Ok(());
    }

    // Stage RGB once into the user's RGB buffer (if attached) or
    // `rgb_scratch`; reused for luma / HSV / RGBA fan-out below.
    let rgb_row = rgb_row_buf_or_scratch(
      rgb.as_deref_mut(),
      rgb_scratch,
      one_plane_start,
      one_plane_end,
      w,
      h,
    )?;
    gbr_to_rgb_row(g_in, b_in, r_in, rgb_row, w, use_simd);

    if let Some(luma) = luma.as_deref_mut() {
      rgb_to_luma_row(
        rgb_row,
        &mut luma[one_plane_start..one_plane_end],
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
    }

    if let Some(luma_u16) = luma_u16.as_deref_mut() {
      // Direct u8-RGB → u16 luma — `rgb_to_luma_u16_row` produces the
      // same byte values as the staged u8-luma + zero-extend path but
      // without any per-row scratch (no stack array, no heap alloc).
      rgb_to_luma_u16_row(
        rgb_row,
        &mut luma_u16[one_plane_start..one_plane_end],
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

    if let Some(buf) = rgba.as_deref_mut() {
      // Strategy A: expand the already-computed rgb_row → rgba_row
      // (constant α = `0xFF`). Avoids running a second per-pixel
      // interleave kernel.
      let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
    }

    Ok(())
  }
}

// ---- Gbrap impl ---------------------------------------------------------

impl<'a, R> MixedSinker<'a, Gbrap, R> {
  /// Attaches a packed **8-bit** RGBA output buffer. Alpha is sourced
  /// from the source's A plane (real per-pixel α, not constant
  /// `0xFF`).
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

  /// Attaches a `u16` luma output buffer. Same derivation as the
  /// `Gbrp` sibling — luma is computed from G/B/R via
  /// `rgb_to_luma_row`, then zero-extended into `u16`.
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

impl<R> GbrapSink for MixedSinker<'_, Gbrap, R> {}

impl<R> PixelSink for MixedSinker<'_, Gbrap, R> {
  type Input<'r> = GbrapRow<'r>;
  type Error = MixedSinkerError;

  #[cfg_attr(not(tarpaulin), inline(always))]
  fn kernel_matrix(&self) -> crate::KernelMatrix {
    self.kernel_matrix
  }
  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    if let Some(stream) = self.rgb_stream.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.rgba_stream.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.rgb_filter_stream.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.rgba_filter_stream.as_mut() {
      stream.reset();
    }
    self.resample_outputs = None;
    self.frozen_alpha_mode = Some(self.alpha_mode);
    Ok(())
  }

  fn process(&mut self, row: GbrapRow<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if row.g().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::GPlane,
        idx,
        w,
        row.g().len(),
      )));
    }
    if row.b().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::BPlane,
        idx,
        w,
        row.b().len(),
      )));
    }
    if row.r().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::RPlane,
        idx,
        w,
        row.r().len(),
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

    let Self {
      rgb,
      rgba,
      luma,
      luma_u16,
      hsv,
      rgb_scratch,
      rgba_scratch,
      plan,
      rgb_stream,
      rgba_stream,
      rgb_filter_stream,
      rgba_filter_stream,
      resample_outputs,
      alpha_mode,
      frozen_alpha_mode,
      ..
    } = self;
    let alpha_mode = *alpha_mode;
    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;
    let g_in = row.g();
    let b_in = row.b();
    let r_in = row.r();
    let a_in = row.a();

    // Non-identity plan. Route the alpha-aware 4-channel tail when the
    // resampled alpha would otherwise be dropped (rgba attached) or the
    // color must be alpha-weighted (premultiplied mode); otherwise the
    // rgb-only straight outputs keep the 3-channel RGB path. `Gbrap`
    // de-interleaves its G/B/R/A planes into the same canonical
    // source-width RGBA row the packed sources stage (`gbra_to_rgba_row`),
    // then feeds the shared tail — so resampled alpha is a real area mean
    // and luma derives from the binned RGB (GBR->luma == RGB->luma). The
    // span kind picks the engine: the integer area stream or the
    // signed-coefficient filter stream (PIL parity, straight alpha only).
    if let Some(plan) = plan.as_ref() {
      // The alpha mode is snapshotted at begin_frame; reject a mid-frame
      // change here, before route selection (it picks the 4-channel vs
      // 3-channel route), so a flip can neither reroute nor mix modes.
      check_frozen_alpha_mode(*frozen_alpha_mode, alpha_mode, idx)?;
      match plan.kind() {
        crate::resample::SpanKind::Area => {
          if rgba.is_some() || alpha_mode.is_premultiplied() {
            return packed_rgba_resample::<false>(
              rgba_stream,
              // No native-Y luma stream: `Gbrap8` luma is color-derived
              // (`NATIVE_Y_LUMA = false`), so the Y stream / scratch /
              // de-interleave are inert placeholders.
              &mut None,
              resample_outputs,
              rgb,
              rgba,
              // Gbrap has no u16 RGB outputs (8-bit planar source).
              &mut None,
              &mut None,
              luma,
              luma_u16,
              hsv,
              rgba_scratch,
              rgb_scratch,
              &mut std::vec::Vec::new(),
              w,
              plan,
              idx,
              use_simd,
              alpha_mode,
              row.matrix(),
              row.full_range(),
              |dst| gbra_to_rgba_row(g_in, b_in, r_in, a_in, dst, w, use_simd),
              |_| {},
            );
          }
          // Straight rgb-only (alpha dropped): stage drop-alpha RGB via the
          // 3-plane `gbr_to_rgb_row` and feed the 3-channel tail (luma_u16
          // included, for parity with the direct path).
          return packed_rgb_resample(
            rgb_stream,
            resample_outputs,
            rgb,
            rgba,
            luma,
            luma_u16,
            hsv,
            rgb_scratch,
            None,
            w,
            plan,
            idx,
            use_simd,
            row.matrix(),
            row.full_range(),
            |scratch| gbr_to_rgb_row(g_in, b_in, r_in, scratch, w, use_simd),
          );
        }
        crate::resample::SpanKind::Filter => {
          // Premultiplied alpha has no filter analogue (the filter engine
          // cannot un-premultiply); surface the typed `UnsupportedFilter`
          // via the area tail's reject rather than emit straight-filtered
          // premultiplied color. Straight alpha: filter all four channels
          // independently (PIL RGBA semantics) when alpha survives, else
          // the 3-channel filter for rgb-only outputs (alpha dropped).
          if alpha_mode.is_premultiplied() {
            return packed_rgba_resample::<false>(
              rgba_stream,
              &mut None,
              resample_outputs,
              rgb,
              rgba,
              &mut None,
              &mut None,
              luma,
              luma_u16,
              hsv,
              rgba_scratch,
              rgb_scratch,
              &mut std::vec::Vec::new(),
              w,
              plan,
              idx,
              use_simd,
              alpha_mode,
              row.matrix(),
              row.full_range(),
              |dst| gbra_to_rgba_row(g_in, b_in, r_in, a_in, dst, w, use_simd),
              |_| {},
            );
          }
          if rgba.is_some() {
            return packed_rgba_filter_resample(
              rgba_filter_stream,
              resample_outputs,
              rgb,
              rgba,
              luma,
              luma_u16,
              hsv,
              rgba_scratch,
              rgb_scratch,
              w,
              plan,
              idx,
              use_simd,
              row.matrix(),
              row.full_range(),
              |dst| gbra_to_rgba_row(g_in, b_in, r_in, a_in, dst, w, use_simd),
            );
          }
          // Straight rgb-only (alpha dropped): stage drop-alpha RGB via the
          // 3-plane `gbr_to_rgb_row` and feed the 3-channel filter tail.
          return packed_rgb_resample_filter(
            rgb_filter_stream,
            resample_outputs,
            rgb,
            rgba,
            luma,
            luma_u16,
            hsv,
            rgb_scratch,
            None,
            w,
            plan,
            idx,
            use_simd,
            row.matrix(),
            row.full_range(),
            |scratch| gbr_to_rgb_row(g_in, b_in, r_in, scratch, w, use_simd),
          );
        }
      }
    }

    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_luma = luma.is_some();
    let want_luma_u16 = luma_u16.is_some();
    let want_hsv = hsv.is_some();
    let need_rgb_buffer = want_rgb || want_luma || want_luma_u16 || want_hsv;

    // RGBA-only (no RGB / luma / HSV): direct planar→packed RGBA with
    // source α (no staged RGB scratch).
    if want_rgba && !need_rgb_buffer {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      gbra_to_rgba_row(g_in, b_in, r_in, a_in, rgba_row, w, use_simd);
      return Ok(());
    }

    if !need_rgb_buffer && !want_rgba {
      return Ok(());
    }

    // Stage RGB once. Reused for luma / HSV / RGBA fan-out.
    let rgb_row = rgb_row_buf_or_scratch(
      rgb.as_deref_mut(),
      rgb_scratch,
      one_plane_start,
      one_plane_end,
      w,
      h,
    )?;
    gbr_to_rgb_row(g_in, b_in, r_in, rgb_row, w, use_simd);

    if let Some(luma) = luma.as_deref_mut() {
      rgb_to_luma_row(
        rgb_row,
        &mut luma[one_plane_start..one_plane_end],
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
    }

    if let Some(luma_u16) = luma_u16.as_deref_mut() {
      // Direct u8-RGB → u16 luma — see Gbrp `with_luma_u16` above for
      // the rationale (no per-row scratch, byte-identical to the
      // staged u8-luma + zero-extend equivalent).
      rgb_to_luma_u16_row(
        rgb_row,
        &mut luma_u16[one_plane_start..one_plane_end],
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

    if let Some(buf) = rgba.as_deref_mut() {
      // Strategy A+: expand RGB row → RGBA (constant α = 0xFF), then
      // overwrite the α byte from the source A plane. Saves the
      // second per-pixel interleave kernel call.
      let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
      crate::row::alpha_extract::copy_alpha_plane_u8(a_in, rgba_row, w, use_simd);
    }

    Ok(())
  }
}

// ---- helpers ------------------------------------------------------------

/// Returns `true` iff *any* output other than `with_rgba` is requested.
/// Used by `Gbrp::process` to decide between the standalone-RGBA fast
/// path and the staged-RGB combo path.
#[cfg_attr(not(tarpaulin), inline(always))]
const fn need_rgb_buffer_other(
  want_rgb: bool,
  want_luma: bool,
  want_luma_u16: bool,
  want_hsv: bool,
) -> bool {
  want_rgb || want_luma || want_luma_u16 || want_hsv
}
