//! Sinker impls for packed-RGB **source** formats (Tier 6) — 8-bit
//! per channel.
//!
//! Source family covered here:
//! - [`Rgb24`] — packed `R, G, B` bytes.
//! - [`Bgr24`] — packed `B, G, R` bytes (channel order swapped).
//!
//! Unlike every other source family in this crate, the input is
//! already RGB — there's no chroma matrix work. Outputs map to the
//! sink's standard channels:
//! - `with_rgb` — identity copy for `Rgb24`; `bgr_to_rgb_row` swap
//!   for `Bgr24`.
//! - `with_rgba` — `expand_rgb_to_rgba_row` (constant `0xFF` alpha)
//!   on top of the RGB row above.
//! - `with_luma` — `rgb_to_luma_row` (BT.* coefficient set per the
//!   row's `matrix` + `full_range`); for `Bgr24` the row is swapped
//!   into the existing `rgb_scratch` buffer first.
//! - `with_hsv` — `rgb_to_hsv_row` (existing kernel); same scratch
//!   pattern for `Bgr24`.
//!
//! The 4-byte packed sources ([`Rgba`], [`Bgra`]) added in Ship 9b
//! follow the same rgb_scratch-staging pattern but use the alpha-
//! aware row primitives `rgba_to_rgb_row`, `bgra_to_rgba_row`, and
//! `bgra_to_rgb_row`. RGBA output is alpha pass-through (not forced
//! to `0xFF`). All three new kernels dispatch to NEON / SSE4.1 /
//! AVX2 / AVX-512 / wasm-simd128 alongside the existing 3-byte
//! `bgr_to_rgb_row`.
//!
//! Ship 9c adds the **leading-alpha** family ([`Argb`], [`Abgr`])
//! using the row primitives `argb_to_rgb_row`, `abgr_to_rgb_row`,
//! `argb_to_rgba_row`, `abgr_to_rgba_row` — same scratch-staging
//! pattern, alpha is rotated from the leading byte to the trailing
//! byte for `with_rgba` output. All four kernels also dispatch to
//! the full SIMD backend matrix.
//!
//! Ship 9d closes Tier 6 with the **padding-byte family** ([`Xrgb`],
//! [`Rgbx`], [`Xbgr`], [`Bgrx`] — FFmpeg `0rgb` / `rgb0` / `0bgr` /
//! `bgr0`). The 4th byte position is *ignored padding* (not real
//! alpha), so `with_rgba` output forces alpha to `0xFF`. The
//! `with_rgb` paths reuse the Ship 9b/9c kernels (`argb_to_rgb_row`
//! for Xrgb, `rgba_to_rgb_row` for Rgbx, `abgr_to_rgb_row` for Xbgr,
//! `bgra_to_rgb_row` for Bgrx) — at the byte level, "drop alpha" and
//! "drop padding" are identical operations because both ignore the
//! same byte position. The `with_rgba` paths use four new SIMD
//! kernels (`xrgb_to_rgba_row` etc.) that fold the byte rearrangement
//! and the constant-`0xFF` alpha into a single pass.
//!
//! 8-bit packed RGB has no `u16` output flavor — `with_rgb_u16` /
//! `with_rgba_u16` are not declared on these source impls.

use super::{
  InsufficientBuffer, MixedSinker, MixedSinkerError, RowIndexOutOfRange, RowShapeMismatch,
  RowSlice, check_dimensions_match, check_frozen_alpha_mode, packed_rgb_resample,
  packed_rgb_resample_filter, packed_rgba_filter_resample, packed_rgba_resample,
  rgb_row_buf_or_scratch, rgba_plane_row_slice,
};
use crate::{
  PixelSink,
  row::{
    abgr_to_rgb_row, abgr_to_rgba_row, argb_to_rgb_row, argb_to_rgba_row, bgr_to_rgb_row,
    bgra_to_rgb_row, bgra_to_rgba_row, bgrx_to_rgba_row, expand_rgb_to_rgba_row, rgb_to_hsv_row,
    rgb_to_luma_row, rgba_to_rgb_row, rgbx_to_rgba_row, xbgr_to_rgba_row, xrgb_to_rgba_row,
  },
  source::{
    Abgr, AbgrRow, AbgrSink, Argb, ArgbRow, ArgbSink, Bgr24, Bgr24Row, Bgr24Sink, Bgra, BgraRow,
    BgraSink, Bgrx, BgrxRow, BgrxSink, Rgb24, Rgb24Row, Rgb24Sink, Rgba, RgbaRow, RgbaSink, Rgbx,
    RgbxRow, RgbxSink, Xbgr, XbgrRow, XbgrSink, Xrgb, XrgbRow, XrgbSink,
  },
};

// ---- Rgb24 impl --------------------------------------------------------

impl<'a, R> MixedSinker<'a, Rgb24, R> {
  /// Attaches a packed **8-bit** RGBA output buffer. Alpha is filled
  /// with constant `0xFF` (the source has no alpha channel).
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
}

impl<R> Rgb24Sink for MixedSinker<'_, Rgb24, R> {}

impl<R> PixelSink for MixedSinker<'_, Rgb24, R> {
  type Input<'r> = Rgb24Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    if let Some(stream) = self.rgb_stream.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.rgb_filter_stream.as_mut() {
      stream.reset();
    }
    self.resample_outputs = None;
    self.frozen_alpha_mode = Some(self.alpha_mode);
    Ok(())
  }

  fn process(&mut self, row: Rgb24Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if row.rgb().len() != w * 3 {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::RgbPacked,
        idx,
        w * 3,
        row.rgb().len(),
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
      hsv,
      rgb_scratch,
      plan,
      rgb_stream,
      rgb_filter_stream,
      resample_outputs,
      ..
    } = self;

    // Non-identity plan: the source row IS interleaved RGB, so the
    // fused path feeds it to the resample stream directly (zero-copy —
    // `direct`), making the native and row-stage tiers one and the same
    // for this family. Luma / HSV / RGBA derive from each finalized
    // output row. The plan's span kind picks the engine — the integer
    // area stream or the signed-coefficient filter stream — behind the
    // shared transactional resample helper (RFC #238 tail-collapse).
    if let Some(plan) = plan.as_ref() {
      return match plan.kind() {
        crate::resample::SpanKind::Area => packed_rgb_resample(
          rgb_stream,
          resample_outputs,
          rgb,
          rgba,
          luma,
          &mut None,
          hsv,
          rgb_scratch,
          Some(row.rgb()),
          w,
          plan,
          idx,
          use_simd,
          row.matrix(),
          row.full_range(),
          |_| {},
        ),
        crate::resample::SpanKind::Filter => packed_rgb_resample_filter(
          rgb_filter_stream,
          resample_outputs,
          rgb,
          rgba,
          luma,
          &mut None,
          hsv,
          rgb_scratch,
          Some(row.rgb()),
          w,
          plan,
          idx,
          use_simd,
          row.matrix(),
          row.full_range(),
          |_| {},
        ),
      };
    }
    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // The source row IS RGB — sinks read directly from `row.rgb()`
    // for HSV/luma without copying through rgb_scratch first.
    let rgb_in = row.rgb();

    // Luma — derive Y' from RGB.
    if let Some(luma) = luma.as_deref_mut() {
      rgb_to_luma_row(
        rgb_in,
        &mut luma[one_plane_start..one_plane_end],
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
    }

    // HSV — direct from source RGB.
    if let Some(hsv) = hsv.as_mut() {
      let (h, s, v) = hsv.hsv();
      rgb_to_hsv_row(
        rgb_in,
        &mut h[one_plane_start..one_plane_end],
        &mut s[one_plane_start..one_plane_end],
        &mut v[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    // RGB output — identity copy.
    if let Some(rgb_buf) = rgb.as_deref_mut() {
      let rgb_start = one_plane_start * 3;
      let rgb_end = one_plane_end * 3;
      rgb_buf[rgb_start..rgb_end].copy_from_slice(rgb_in);
    }

    // RGBA output — append 0xFF alpha.
    if let Some(buf) = rgba.as_deref_mut() {
      let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      expand_rgb_to_rgba_row(rgb_in, rgba_row, w);
    }

    Ok(())
  }
}

// ---- Bgr24 impl --------------------------------------------------------

impl<'a, R> MixedSinker<'a, Bgr24, R> {
  /// Attaches a packed **8-bit** RGBA output buffer. Channel order
  /// is swapped on output (input is `B, G, R`; output is `R, G, B,
  /// 0xFF`). Alpha is filled with constant `0xFF` (the source has
  /// no alpha channel).
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
}

impl<R> Bgr24Sink for MixedSinker<'_, Bgr24, R> {}

impl<R> PixelSink for MixedSinker<'_, Bgr24, R> {
  type Input<'r> = Bgr24Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    if let Some(stream) = self.rgb_stream.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.rgb_filter_stream.as_mut() {
      stream.reset();
    }
    self.resample_outputs = None;
    self.frozen_alpha_mode = Some(self.alpha_mode);
    Ok(())
  }

  fn process(&mut self, row: Bgr24Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if row.bgr().len() != w * 3 {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::BgrPacked,
        idx,
        w * 3,
        row.bgr().len(),
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
      hsv,
      rgb_scratch,
      plan,
      rgb_stream,
      resample_outputs,
      ..
    } = self;

    // Non-identity plan: stage the BGR->RGB row into the source-width
    // scratch and feed the shared transactional resample helper (RFC #238
    // tail-collapse). Area-only — a filter plan is rejected inside the
    // helper (`Bgr24` is not routed to the filter engine).
    if let Some(plan) = plan.as_ref() {
      return packed_rgb_resample(
        rgb_stream,
        resample_outputs,
        rgb,
        rgba,
        luma,
        &mut None,
        hsv,
        rgb_scratch,
        None,
        w,
        plan,
        idx,
        use_simd,
        row.matrix(),
        row.full_range(),
        |scratch| bgr_to_rgb_row(row.bgr(), scratch, w, use_simd),
      );
    }

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_luma = luma.is_some();
    let want_hsv = hsv.is_some();

    // For Bgr24, every RGB-input kernel needs the row already swapped
    // to RGB order. Stage it once into `rgb_scratch` (or directly
    // into the user's RGB output buffer if attached) and reuse for
    // luma / HSV / RGBA.
    let need_rgb_buffer = want_rgb || want_rgba || want_luma || want_hsv;
    if !need_rgb_buffer {
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
    bgr_to_rgb_row(row.bgr(), rgb_row, w, use_simd);

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

// ---- Rgba impl (Ship 9b) -----------------------------------------------

impl<'a, R> MixedSinker<'a, Rgba, R> {
  /// Attaches a packed **8-bit** RGBA output buffer. The source row
  /// is already RGBA — the per-pixel write is a memcpy of the
  /// source bytes (alpha is **passed through**, not forced to
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
}

impl<R> RgbaSink for MixedSinker<'_, Rgba, R> {}

impl<R> PixelSink for MixedSinker<'_, Rgba, R> {
  type Input<'r> = RgbaRow<'r>;
  type Error = MixedSinkerError;

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

  fn process(&mut self, row: RgbaRow<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if row.rgba().len() != w * 4 {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::RgbaPacked,
        idx,
        w * 4,
        row.rgba().len(),
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

    // Non-identity plan. Route the alpha-aware 4-channel tail when the
    // resampled alpha would otherwise be dropped (rgba attached) or the
    // color must be alpha-weighted (premultiplied mode); otherwise the
    // rgb-only straight outputs keep the 3-channel RGB path unchanged. The
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
              // No native-Y luma stream: these packed-RGBA sources are
              // chromatic, so luma is color-derived (`NATIVE_Y_LUMA = false`)
              // and the Y stream / scratch / de-interleave are inert.
              &mut None,
              resample_outputs,
              rgb,
              rgba,
              // 8-bit packed RGBA sources expose no u16 RGB outputs.
              &mut None,
              &mut None,
              luma,
              &mut None,
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
              |dst| dst.copy_from_slice(row.rgba()),
              |_| {},
            );
          }
          return packed_rgb_resample(
            rgb_stream,
            resample_outputs,
            rgb,
            rgba,
            luma,
            &mut None,
            hsv,
            rgb_scratch,
            None,
            w,
            plan,
            idx,
            use_simd,
            row.matrix(),
            row.full_range(),
            |scratch| rgba_to_rgb_row(row.rgba(), scratch, w, use_simd),
          );
        }
        crate::resample::SpanKind::Filter => {
          // Premultiplied alpha has no filter analogue (the filter engine
          // cannot un-premultiply); surface the typed `UnsupportedFilter`
          // via the area tail's reject rather than emit straight-filtered
          // premultiplied color. Straight alpha: filter all four channels
          // independently (PIL RGBA semantics) when alpha survives, else
          // the 3-channel filter for rgb-only outputs.
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
              &mut None,
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
              |dst| dst.copy_from_slice(row.rgba()),
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
              // 8-bit packed RGBA sources expose no u16 luma output.
              &mut None,
              hsv,
              rgba_scratch,
              rgb_scratch,
              w,
              plan,
              idx,
              use_simd,
              row.matrix(),
              row.full_range(),
              |dst| dst.copy_from_slice(row.rgba()),
            );
          }
          return packed_rgb_resample_filter(
            rgb_filter_stream,
            resample_outputs,
            rgb,
            rgba,
            luma,
            &mut None,
            hsv,
            rgb_scratch,
            None,
            w,
            plan,
            idx,
            use_simd,
            row.matrix(),
            row.full_range(),
            |scratch| rgba_to_rgb_row(row.rgba(), scratch, w, use_simd),
          );
        }
      }
    }

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;
    let rgba_in = row.rgba();

    let want_rgb = rgb.is_some();
    let want_luma = luma.is_some();
    let want_hsv = hsv.is_some();
    let need_rgb_buffer = want_rgb || want_luma || want_hsv;

    // Stage drop-alpha RGB once into the user's RGB buffer (if
    // attached) or `rgb_scratch`. Reused for luma + HSV.
    if need_rgb_buffer {
      let rgb_row = rgb_row_buf_or_scratch(
        rgb.as_deref_mut(),
        rgb_scratch,
        one_plane_start,
        one_plane_end,
        w,
        h,
      )?;
      rgba_to_rgb_row(rgba_in, rgb_row, w, use_simd);

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
    }

    // RGBA output — identity copy (source layout matches output).
    if let Some(buf) = rgba.as_deref_mut() {
      let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      rgba_row.copy_from_slice(rgba_in);
    }

    Ok(())
  }
}

// ---- Bgra impl (Ship 9b) -----------------------------------------------

impl<'a, R> MixedSinker<'a, Bgra, R> {
  /// Attaches a packed **8-bit** RGBA output buffer. Channel order
  /// is swapped on output (input is `B, G, R, A`; output is
  /// `R, G, B, A`). Alpha is **passed through** from the source,
  /// not forced to `0xFF`.
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
}

impl<R> BgraSink for MixedSinker<'_, Bgra, R> {}

impl<R> PixelSink for MixedSinker<'_, Bgra, R> {
  type Input<'r> = BgraRow<'r>;
  type Error = MixedSinkerError;

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

  fn process(&mut self, row: BgraRow<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if row.bgra().len() != w * 4 {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::BgraPacked,
        idx,
        w * 4,
        row.bgra().len(),
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

    // Non-identity plan — see the `Rgba` impl for the 4-channel-vs-3
    // routing rationale and the area-vs-filter engine split. Bgra stages
    // canonical RGBA via `bgra_to_rgba_row` (swap R↔B, α pass-through) and
    // drop-alpha RGB via `bgra_to_rgb_row`.
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
              // No native-Y luma stream: these packed-RGBA sources are
              // chromatic, so luma is color-derived (`NATIVE_Y_LUMA = false`)
              // and the Y stream / scratch / de-interleave are inert.
              &mut None,
              resample_outputs,
              rgb,
              rgba,
              // 8-bit packed RGBA sources expose no u16 RGB outputs.
              &mut None,
              &mut None,
              luma,
              &mut None,
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
              |dst| bgra_to_rgba_row(row.bgra(), dst, w, use_simd),
              |_| {},
            );
          }
          return packed_rgb_resample(
            rgb_stream,
            resample_outputs,
            rgb,
            rgba,
            luma,
            &mut None,
            hsv,
            rgb_scratch,
            None,
            w,
            plan,
            idx,
            use_simd,
            row.matrix(),
            row.full_range(),
            |scratch| bgra_to_rgb_row(row.bgra(), scratch, w, use_simd),
          );
        }
        crate::resample::SpanKind::Filter => {
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
              &mut None,
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
              |dst| bgra_to_rgba_row(row.bgra(), dst, w, use_simd),
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
              // 8-bit packed RGBA sources expose no u16 luma output.
              &mut None,
              hsv,
              rgba_scratch,
              rgb_scratch,
              w,
              plan,
              idx,
              use_simd,
              row.matrix(),
              row.full_range(),
              |dst| bgra_to_rgba_row(row.bgra(), dst, w, use_simd),
            );
          }
          return packed_rgb_resample_filter(
            rgb_filter_stream,
            resample_outputs,
            rgb,
            rgba,
            luma,
            &mut None,
            hsv,
            rgb_scratch,
            None,
            w,
            plan,
            idx,
            use_simd,
            row.matrix(),
            row.full_range(),
            |scratch| bgra_to_rgb_row(row.bgra(), scratch, w, use_simd),
          );
        }
      }
    }

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;
    let bgra_in = row.bgra();

    let want_rgb = rgb.is_some();
    let want_luma = luma.is_some();
    let want_hsv = hsv.is_some();
    let need_rgb_buffer = want_rgb || want_luma || want_hsv;

    // Stage swap+drop into `rgb_scratch` (or the user's RGB buffer
    // if attached) once — reused for luma / HSV / RGB output.
    if need_rgb_buffer {
      let rgb_row = rgb_row_buf_or_scratch(
        rgb.as_deref_mut(),
        rgb_scratch,
        one_plane_start,
        one_plane_end,
        w,
        h,
      )?;
      bgra_to_rgb_row(bgra_in, rgb_row, w, use_simd);

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
    }

    // RGBA output — swap R↔B, alpha pass-through.
    if let Some(buf) = rgba.as_deref_mut() {
      let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      bgra_to_rgba_row(bgra_in, rgba_row, w, use_simd);
    }

    Ok(())
  }
}

// ---- Argb impl (Ship 9c) -----------------------------------------------

impl<'a, R> MixedSinker<'a, Argb, R> {
  /// Attaches a packed **8-bit** RGBA output buffer. Channel layout
  /// is rotated on output (input is `A, R, G, B`; output is
  /// `R, G, B, A`). Alpha is **passed through** from the source,
  /// not forced to `0xFF`.
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
}

impl<R> ArgbSink for MixedSinker<'_, Argb, R> {}

impl<R> PixelSink for MixedSinker<'_, Argb, R> {
  type Input<'r> = ArgbRow<'r>;
  type Error = MixedSinkerError;

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

  fn process(&mut self, row: ArgbRow<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if row.argb().len() != w * 4 {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::ArgbPacked,
        idx,
        w * 4,
        row.argb().len(),
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

    // Non-identity plan — see the `Rgba` impl. Argb stages canonical
    // RGBA via `argb_to_rgba_row` (rotate leading α to slot 3) and
    // drop-alpha RGB via `argb_to_rgb_row`.
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
              // No native-Y luma stream: these packed-RGBA sources are
              // chromatic, so luma is color-derived (`NATIVE_Y_LUMA = false`)
              // and the Y stream / scratch / de-interleave are inert.
              &mut None,
              resample_outputs,
              rgb,
              rgba,
              // 8-bit packed RGBA sources expose no u16 RGB outputs.
              &mut None,
              &mut None,
              luma,
              &mut None,
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
              |dst| argb_to_rgba_row(row.argb(), dst, w, use_simd),
              |_| {},
            );
          }
          return packed_rgb_resample(
            rgb_stream,
            resample_outputs,
            rgb,
            rgba,
            luma,
            &mut None,
            hsv,
            rgb_scratch,
            None,
            w,
            plan,
            idx,
            use_simd,
            row.matrix(),
            row.full_range(),
            |scratch| argb_to_rgb_row(row.argb(), scratch, w, use_simd),
          );
        }
        crate::resample::SpanKind::Filter => {
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
              &mut None,
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
              |dst| argb_to_rgba_row(row.argb(), dst, w, use_simd),
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
              // 8-bit packed RGBA sources expose no u16 luma output.
              &mut None,
              hsv,
              rgba_scratch,
              rgb_scratch,
              w,
              plan,
              idx,
              use_simd,
              row.matrix(),
              row.full_range(),
              |dst| argb_to_rgba_row(row.argb(), dst, w, use_simd),
            );
          }
          return packed_rgb_resample_filter(
            rgb_filter_stream,
            resample_outputs,
            rgb,
            rgba,
            luma,
            &mut None,
            hsv,
            rgb_scratch,
            None,
            w,
            plan,
            idx,
            use_simd,
            row.matrix(),
            row.full_range(),
            |scratch| argb_to_rgb_row(row.argb(), scratch, w, use_simd),
          );
        }
      }
    }

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;
    let argb_in = row.argb();

    let want_rgb = rgb.is_some();
    let want_luma = luma.is_some();
    let want_hsv = hsv.is_some();
    let need_rgb_buffer = want_rgb || want_luma || want_hsv;

    // Stage drop-leading-alpha RGB once into the user's RGB buffer
    // (if attached) or `rgb_scratch`. Reused for luma + HSV.
    if need_rgb_buffer {
      let rgb_row = rgb_row_buf_or_scratch(
        rgb.as_deref_mut(),
        rgb_scratch,
        one_plane_start,
        one_plane_end,
        w,
        h,
      )?;
      argb_to_rgb_row(argb_in, rgb_row, w, use_simd);

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
    }

    // RGBA output — rotate alpha to trailing position.
    if let Some(buf) = rgba.as_deref_mut() {
      let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      argb_to_rgba_row(argb_in, rgba_row, w, use_simd);
    }

    Ok(())
  }
}

// ---- Abgr impl (Ship 9c) -----------------------------------------------

impl<'a, R> MixedSinker<'a, Abgr, R> {
  /// Attaches a packed **8-bit** RGBA output buffer. Channel layout
  /// is fully reversed on output (input is `A, B, G, R`; output is
  /// `R, G, B, A`). Alpha is **passed through** from the source,
  /// not forced to `0xFF`.
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
}

impl<R> AbgrSink for MixedSinker<'_, Abgr, R> {}

impl<R> PixelSink for MixedSinker<'_, Abgr, R> {
  type Input<'r> = AbgrRow<'r>;
  type Error = MixedSinkerError;

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

  fn process(&mut self, row: AbgrRow<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if row.abgr().len() != w * 4 {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::AbgrPacked,
        idx,
        w * 4,
        row.abgr().len(),
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

    // Non-identity plan — see the `Rgba` impl. Abgr stages canonical
    // RGBA via `abgr_to_rgba_row` (full byte reverse) and drop-alpha RGB
    // via `abgr_to_rgb_row`.
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
              // No native-Y luma stream: these packed-RGBA sources are
              // chromatic, so luma is color-derived (`NATIVE_Y_LUMA = false`)
              // and the Y stream / scratch / de-interleave are inert.
              &mut None,
              resample_outputs,
              rgb,
              rgba,
              // 8-bit packed RGBA sources expose no u16 RGB outputs.
              &mut None,
              &mut None,
              luma,
              &mut None,
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
              |dst| abgr_to_rgba_row(row.abgr(), dst, w, use_simd),
              |_| {},
            );
          }
          return packed_rgb_resample(
            rgb_stream,
            resample_outputs,
            rgb,
            rgba,
            luma,
            &mut None,
            hsv,
            rgb_scratch,
            None,
            w,
            plan,
            idx,
            use_simd,
            row.matrix(),
            row.full_range(),
            |scratch| abgr_to_rgb_row(row.abgr(), scratch, w, use_simd),
          );
        }
        crate::resample::SpanKind::Filter => {
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
              &mut None,
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
              |dst| abgr_to_rgba_row(row.abgr(), dst, w, use_simd),
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
              // 8-bit packed RGBA sources expose no u16 luma output.
              &mut None,
              hsv,
              rgba_scratch,
              rgb_scratch,
              w,
              plan,
              idx,
              use_simd,
              row.matrix(),
              row.full_range(),
              |dst| abgr_to_rgba_row(row.abgr(), dst, w, use_simd),
            );
          }
          return packed_rgb_resample_filter(
            rgb_filter_stream,
            resample_outputs,
            rgb,
            rgba,
            luma,
            &mut None,
            hsv,
            rgb_scratch,
            None,
            w,
            plan,
            idx,
            use_simd,
            row.matrix(),
            row.full_range(),
            |scratch| abgr_to_rgb_row(row.abgr(), scratch, w, use_simd),
          );
        }
      }
    }

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;
    let abgr_in = row.abgr();

    let want_rgb = rgb.is_some();
    let want_luma = luma.is_some();
    let want_hsv = hsv.is_some();
    let need_rgb_buffer = want_rgb || want_luma || want_hsv;

    // Stage swap+drop into `rgb_scratch` (or the user's RGB buffer
    // if attached) once — reused for luma / HSV / RGB output.
    if need_rgb_buffer {
      let rgb_row = rgb_row_buf_or_scratch(
        rgb.as_deref_mut(),
        rgb_scratch,
        one_plane_start,
        one_plane_end,
        w,
        h,
      )?;
      abgr_to_rgb_row(abgr_in, rgb_row, w, use_simd);

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
    }

    // RGBA output — full byte reverse.
    if let Some(buf) = rgba.as_deref_mut() {
      let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      abgr_to_rgba_row(abgr_in, rgba_row, w, use_simd);
    }

    Ok(())
  }
}

// ---- Xrgb impl (Ship 9d) -----------------------------------------------

impl<'a, R> MixedSinker<'a, Xrgb, R> {
  /// Attaches a packed **8-bit** RGBA output buffer. The leading
  /// padding byte from the source is dropped and alpha is forced to
  /// `0xFF` (the source has no real alpha — the X byte's value is
  /// undefined).
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
}

impl<R> XrgbSink for MixedSinker<'_, Xrgb, R> {}

impl<R> PixelSink for MixedSinker<'_, Xrgb, R> {
  type Input<'r> = XrgbRow<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    if let Some(stream) = self.rgb_stream.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.rgb_filter_stream.as_mut() {
      stream.reset();
    }
    self.resample_outputs = None;
    self.frozen_alpha_mode = Some(self.alpha_mode);
    Ok(())
  }

  fn process(&mut self, row: XrgbRow<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if row.xrgb().len() != w * 4 {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::XrgbPacked,
        idx,
        w * 4,
        row.xrgb().len(),
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
      hsv,
      rgb_scratch,
      plan,
      rgb_stream,
      rgb_filter_stream,
      resample_outputs,
      ..
    } = self;

    // Non-identity plan: drop the padding byte to canonical RGB at
    // source width in the shared scratch, then feed the shared
    // packed-RGB resample tail (RGBA output gets 0xFF alpha, the
    // padding-byte contract). The X byte is padding, never a filtered
    // channel — the 3 real RGB channels resample (area or filter per the
    // plan kind) and X is reset to 0xFF on output, the same as the area
    // path treats it.
    if let Some(plan) = plan.as_ref() {
      // The X byte is padding, never a filtered channel — the 3 real RGB
      // channels resample (area or filter per the plan kind) through the shared
      // transactional helper (RFC #238 tail-collapse) and X is reset to 0xFF on
      // output, the same as the area path treats it.
      return match plan.kind() {
        crate::resample::SpanKind::Area => packed_rgb_resample(
          rgb_stream,
          resample_outputs,
          rgb,
          rgba,
          luma,
          &mut None,
          hsv,
          rgb_scratch,
          None,
          w,
          plan,
          idx,
          use_simd,
          row.matrix(),
          row.full_range(),
          |scratch| argb_to_rgb_row(row.xrgb(), scratch, w, use_simd),
        ),
        crate::resample::SpanKind::Filter => packed_rgb_resample_filter(
          rgb_filter_stream,
          resample_outputs,
          rgb,
          rgba,
          luma,
          &mut None,
          hsv,
          rgb_scratch,
          None,
          w,
          plan,
          idx,
          use_simd,
          row.matrix(),
          row.full_range(),
          |scratch| argb_to_rgb_row(row.xrgb(), scratch, w, use_simd),
        ),
      };
    }

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;
    let xrgb_in = row.xrgb();

    let want_rgb = rgb.is_some();
    let want_luma = luma.is_some();
    let want_hsv = hsv.is_some();
    let need_rgb_buffer = want_rgb || want_luma || want_hsv;

    // Drop leading padding byte using the existing `argb_to_rgb_row`
    // kernel — at the byte level, "drop alpha" and "drop padding"
    // are identical operations because both target byte 0.
    if need_rgb_buffer {
      let rgb_row = rgb_row_buf_or_scratch(
        rgb.as_deref_mut(),
        rgb_scratch,
        one_plane_start,
        one_plane_end,
        w,
        h,
      )?;
      argb_to_rgb_row(xrgb_in, rgb_row, w, use_simd);

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
    }

    // RGBA output — drop leading padding + force alpha to 0xFF.
    if let Some(buf) = rgba.as_deref_mut() {
      let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      xrgb_to_rgba_row(xrgb_in, rgba_row, w, use_simd);
    }

    Ok(())
  }
}

// ---- Rgbx impl (Ship 9d) -----------------------------------------------

impl<'a, R> MixedSinker<'a, Rgbx, R> {
  /// Attaches a packed **8-bit** RGBA output buffer. The trailing
  /// padding byte from the source is replaced with `A = 0xFF`.
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
}

impl<R> RgbxSink for MixedSinker<'_, Rgbx, R> {}

impl<R> PixelSink for MixedSinker<'_, Rgbx, R> {
  type Input<'r> = RgbxRow<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    if let Some(stream) = self.rgb_stream.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.rgb_filter_stream.as_mut() {
      stream.reset();
    }
    self.resample_outputs = None;
    self.frozen_alpha_mode = Some(self.alpha_mode);
    Ok(())
  }

  fn process(&mut self, row: RgbxRow<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if row.rgbx().len() != w * 4 {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::RgbxPacked,
        idx,
        w * 4,
        row.rgbx().len(),
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
      hsv,
      rgb_scratch,
      plan,
      rgb_stream,
      rgb_filter_stream,
      resample_outputs,
      ..
    } = self;

    // Non-identity plan: drop the padding byte to canonical RGB at
    // source width in the shared scratch, then feed the shared
    // packed-RGB resample tail (RGBA output gets 0xFF alpha, the
    // padding-byte contract). The X byte is padding, never a filtered
    // channel — the 3 real RGB channels resample (area or filter per the
    // plan kind) and X is reset to 0xFF on output.
    if let Some(plan) = plan.as_ref() {
      // The X byte is padding, never a filtered channel — the 3 real RGB
      // channels resample (area or filter per the plan kind) through the shared
      // transactional helper (RFC #238 tail-collapse) and X is reset to 0xFF on
      // output, the same as the area path treats it.
      return match plan.kind() {
        crate::resample::SpanKind::Area => packed_rgb_resample(
          rgb_stream,
          resample_outputs,
          rgb,
          rgba,
          luma,
          &mut None,
          hsv,
          rgb_scratch,
          None,
          w,
          plan,
          idx,
          use_simd,
          row.matrix(),
          row.full_range(),
          |scratch| rgba_to_rgb_row(row.rgbx(), scratch, w, use_simd),
        ),
        crate::resample::SpanKind::Filter => packed_rgb_resample_filter(
          rgb_filter_stream,
          resample_outputs,
          rgb,
          rgba,
          luma,
          &mut None,
          hsv,
          rgb_scratch,
          None,
          w,
          plan,
          idx,
          use_simd,
          row.matrix(),
          row.full_range(),
          |scratch| rgba_to_rgb_row(row.rgbx(), scratch, w, use_simd),
        ),
      };
    }

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;
    let rgbx_in = row.rgbx();

    let want_rgb = rgb.is_some();
    let want_luma = luma.is_some();
    let want_hsv = hsv.is_some();
    let need_rgb_buffer = want_rgb || want_luma || want_hsv;

    if need_rgb_buffer {
      let rgb_row = rgb_row_buf_or_scratch(
        rgb.as_deref_mut(),
        rgb_scratch,
        one_plane_start,
        one_plane_end,
        w,
        h,
      )?;
      // "Drop alpha" and "drop padding" target the same trailing byte.
      rgba_to_rgb_row(rgbx_in, rgb_row, w, use_simd);

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
    }

    if let Some(buf) = rgba.as_deref_mut() {
      let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      rgbx_to_rgba_row(rgbx_in, rgba_row, w, use_simd);
    }

    Ok(())
  }
}

// ---- Xbgr impl (Ship 9d) -----------------------------------------------

impl<'a, R> MixedSinker<'a, Xbgr, R> {
  /// Attaches a packed **8-bit** RGBA output buffer. Channel order
  /// is reversed and the leading padding byte is replaced with
  /// `A = 0xFF` (input is `X, B, G, R`; output is `R, G, B, 0xFF`).
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
}

impl<R> XbgrSink for MixedSinker<'_, Xbgr, R> {}

impl<R> PixelSink for MixedSinker<'_, Xbgr, R> {
  type Input<'r> = XbgrRow<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    if let Some(stream) = self.rgb_stream.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.rgb_filter_stream.as_mut() {
      stream.reset();
    }
    self.resample_outputs = None;
    self.frozen_alpha_mode = Some(self.alpha_mode);
    Ok(())
  }

  fn process(&mut self, row: XbgrRow<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if row.xbgr().len() != w * 4 {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::XbgrPacked,
        idx,
        w * 4,
        row.xbgr().len(),
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
      hsv,
      rgb_scratch,
      plan,
      rgb_stream,
      rgb_filter_stream,
      resample_outputs,
      ..
    } = self;

    // Non-identity plan: drop the padding byte to canonical RGB at
    // source width in the shared scratch, then feed the shared
    // packed-RGB resample tail (RGBA output gets 0xFF alpha, the
    // padding-byte contract). The X byte is padding, never a filtered
    // channel — the 3 real RGB channels resample (area or filter per the
    // plan kind) and X is reset to 0xFF on output.
    if let Some(plan) = plan.as_ref() {
      // The X byte is padding, never a filtered channel — the 3 real RGB
      // channels resample (area or filter per the plan kind) through the shared
      // transactional helper (RFC #238 tail-collapse) and X is reset to 0xFF on
      // output, the same as the area path treats it.
      return match plan.kind() {
        crate::resample::SpanKind::Area => packed_rgb_resample(
          rgb_stream,
          resample_outputs,
          rgb,
          rgba,
          luma,
          &mut None,
          hsv,
          rgb_scratch,
          None,
          w,
          plan,
          idx,
          use_simd,
          row.matrix(),
          row.full_range(),
          |scratch| abgr_to_rgb_row(row.xbgr(), scratch, w, use_simd),
        ),
        crate::resample::SpanKind::Filter => packed_rgb_resample_filter(
          rgb_filter_stream,
          resample_outputs,
          rgb,
          rgba,
          luma,
          &mut None,
          hsv,
          rgb_scratch,
          None,
          w,
          plan,
          idx,
          use_simd,
          row.matrix(),
          row.full_range(),
          |scratch| abgr_to_rgb_row(row.xbgr(), scratch, w, use_simd),
        ),
      };
    }

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;
    let xbgr_in = row.xbgr();

    let want_rgb = rgb.is_some();
    let want_luma = luma.is_some();
    let want_hsv = hsv.is_some();
    let need_rgb_buffer = want_rgb || want_luma || want_hsv;

    if need_rgb_buffer {
      let rgb_row = rgb_row_buf_or_scratch(
        rgb.as_deref_mut(),
        rgb_scratch,
        one_plane_start,
        one_plane_end,
        w,
        h,
      )?;
      // Reuse `abgr_to_rgb_row`: drops byte 0 and reverses the inner
      // three bytes — identical operation for Abgr and Xbgr inputs.
      abgr_to_rgb_row(xbgr_in, rgb_row, w, use_simd);

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
    }

    if let Some(buf) = rgba.as_deref_mut() {
      let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      xbgr_to_rgba_row(xbgr_in, rgba_row, w, use_simd);
    }

    Ok(())
  }
}

// ---- Bgrx impl (Ship 9d) -----------------------------------------------

impl<'a, R> MixedSinker<'a, Bgrx, R> {
  /// Attaches a packed **8-bit** RGBA output buffer. Channel order
  /// is reversed and the trailing padding byte is replaced with
  /// `A = 0xFF` (input is `B, G, R, X`; output is `R, G, B, 0xFF`).
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
}

impl<R> BgrxSink for MixedSinker<'_, Bgrx, R> {}

impl<R> PixelSink for MixedSinker<'_, Bgrx, R> {
  type Input<'r> = BgrxRow<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    if let Some(stream) = self.rgb_stream.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.rgb_filter_stream.as_mut() {
      stream.reset();
    }
    self.resample_outputs = None;
    self.frozen_alpha_mode = Some(self.alpha_mode);
    Ok(())
  }

  fn process(&mut self, row: BgrxRow<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if row.bgrx().len() != w * 4 {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::BgrxPacked,
        idx,
        w * 4,
        row.bgrx().len(),
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
      hsv,
      rgb_scratch,
      plan,
      rgb_stream,
      rgb_filter_stream,
      resample_outputs,
      ..
    } = self;

    // Non-identity plan: drop the padding byte to canonical RGB at
    // source width in the shared scratch, then feed the shared
    // packed-RGB resample tail (RGBA output gets 0xFF alpha, the
    // padding-byte contract). The X byte is padding, never a filtered
    // channel — the 3 real RGB channels resample (area or filter per the
    // plan kind) and X is reset to 0xFF on output.
    if let Some(plan) = plan.as_ref() {
      // The X byte is padding, never a filtered channel — the 3 real RGB
      // channels resample (area or filter per the plan kind) through the shared
      // transactional helper (RFC #238 tail-collapse) and X is reset to 0xFF on
      // output, the same as the area path treats it.
      return match plan.kind() {
        crate::resample::SpanKind::Area => packed_rgb_resample(
          rgb_stream,
          resample_outputs,
          rgb,
          rgba,
          luma,
          &mut None,
          hsv,
          rgb_scratch,
          None,
          w,
          plan,
          idx,
          use_simd,
          row.matrix(),
          row.full_range(),
          |scratch| bgra_to_rgb_row(row.bgrx(), scratch, w, use_simd),
        ),
        crate::resample::SpanKind::Filter => packed_rgb_resample_filter(
          rgb_filter_stream,
          resample_outputs,
          rgb,
          rgba,
          luma,
          &mut None,
          hsv,
          rgb_scratch,
          None,
          w,
          plan,
          idx,
          use_simd,
          row.matrix(),
          row.full_range(),
          |scratch| bgra_to_rgb_row(row.bgrx(), scratch, w, use_simd),
        ),
      };
    }

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;
    let bgrx_in = row.bgrx();

    let want_rgb = rgb.is_some();
    let want_luma = luma.is_some();
    let want_hsv = hsv.is_some();
    let need_rgb_buffer = want_rgb || want_luma || want_hsv;

    if need_rgb_buffer {
      let rgb_row = rgb_row_buf_or_scratch(
        rgb.as_deref_mut(),
        rgb_scratch,
        one_plane_start,
        one_plane_end,
        w,
        h,
      )?;
      // Reuse `bgra_to_rgb_row`: drops byte 3 and reverses inner
      // three — identical for Bgra and Bgrx.
      bgra_to_rgb_row(bgrx_in, rgb_row, w, use_simd);

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
    }

    if let Some(buf) = rgba.as_deref_mut() {
      let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      bgrx_to_rgba_row(bgrx_in, rgba_row, w, use_simd);
    }

    Ok(())
  }
}
