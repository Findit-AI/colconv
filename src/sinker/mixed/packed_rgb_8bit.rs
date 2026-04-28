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
  MixedSinker, MixedSinkerError, RowSlice, check_dimensions_match, rgb_row_buf_or_scratch,
  rgba_plane_row_slice,
};
use crate::{
  PixelSink,
  row::{
    abgr_to_rgb_row, abgr_to_rgba_row, argb_to_rgb_row, argb_to_rgba_row, bgr_to_rgb_row,
    bgra_to_rgb_row, bgra_to_rgba_row, bgrx_to_rgba_row, expand_rgb_to_rgba_row, rgb_to_hsv_row,
    rgb_to_luma_row, rgba_to_rgb_row, rgbx_to_rgba_row, xbgr_to_rgba_row, xrgb_to_rgba_row,
  },
  yuv::{
    Abgr, AbgrRow, AbgrSink, Argb, ArgbRow, ArgbSink, Bgr24, Bgr24Row, Bgr24Sink, Bgra, BgraRow,
    BgraSink, Bgrx, BgrxRow, BgrxSink, Rgb24, Rgb24Row, Rgb24Sink, Rgba, RgbaRow, RgbaSink, Rgbx,
    RgbxRow, RgbxSink, Xbgr, XbgrRow, XbgrSink, Xrgb, XrgbRow, XrgbSink,
  },
};

// ---- Rgb24 impl --------------------------------------------------------

impl<'a> MixedSinker<'a, Rgb24> {
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
    let expected = self.frame_bytes(4)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::RgbaBufferTooShort {
        expected,
        actual: buf.len(),
      });
    }
    self.rgba = Some(buf);
    Ok(self)
  }
}

impl Rgb24Sink for MixedSinker<'_, Rgb24> {}

impl PixelSink for MixedSinker<'_, Rgb24> {
  type Input<'r> = Rgb24Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: Rgb24Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if row.rgb().len() != w * 3 {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::RgbPacked,
        row: idx,
        expected: w * 3,
        actual: row.rgb().len(),
      });
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange {
        row: idx,
        configured_height: self.height,
      });
    }

    let Self {
      rgb,
      rgba,
      luma,
      hsv,
      rgb_scratch: _,
      ..
    } = self;
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
      rgb_to_hsv_row(
        rgb_in,
        &mut hsv.h[one_plane_start..one_plane_end],
        &mut hsv.s[one_plane_start..one_plane_end],
        &mut hsv.v[one_plane_start..one_plane_end],
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

impl<'a> MixedSinker<'a, Bgr24> {
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
    let expected = self.frame_bytes(4)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::RgbaBufferTooShort {
        expected,
        actual: buf.len(),
      });
    }
    self.rgba = Some(buf);
    Ok(self)
  }
}

impl Bgr24Sink for MixedSinker<'_, Bgr24> {}

impl PixelSink for MixedSinker<'_, Bgr24> {
  type Input<'r> = Bgr24Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: Bgr24Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if row.bgr().len() != w * 3 {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::BgrPacked,
        row: idx,
        expected: w * 3,
        actual: row.bgr().len(),
      });
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange {
        row: idx,
        configured_height: self.height,
      });
    }

    let Self {
      rgb,
      rgba,
      luma,
      hsv,
      rgb_scratch,
      ..
    } = self;
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
      rgb_to_hsv_row(
        rgb_row,
        &mut hsv.h[one_plane_start..one_plane_end],
        &mut hsv.s[one_plane_start..one_plane_end],
        &mut hsv.v[one_plane_start..one_plane_end],
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

impl<'a> MixedSinker<'a, Rgba> {
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
    let expected = self.frame_bytes(4)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::RgbaBufferTooShort {
        expected,
        actual: buf.len(),
      });
    }
    self.rgba = Some(buf);
    Ok(self)
  }
}

impl RgbaSink for MixedSinker<'_, Rgba> {}

impl PixelSink for MixedSinker<'_, Rgba> {
  type Input<'r> = RgbaRow<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: RgbaRow<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if row.rgba().len() != w * 4 {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::RgbaPacked,
        row: idx,
        expected: w * 4,
        actual: row.rgba().len(),
      });
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange {
        row: idx,
        configured_height: self.height,
      });
    }

    let Self {
      rgb,
      rgba,
      luma,
      hsv,
      rgb_scratch,
      ..
    } = self;
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
        rgb_to_hsv_row(
          rgb_row,
          &mut hsv.h[one_plane_start..one_plane_end],
          &mut hsv.s[one_plane_start..one_plane_end],
          &mut hsv.v[one_plane_start..one_plane_end],
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

impl<'a> MixedSinker<'a, Bgra> {
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
    let expected = self.frame_bytes(4)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::RgbaBufferTooShort {
        expected,
        actual: buf.len(),
      });
    }
    self.rgba = Some(buf);
    Ok(self)
  }
}

impl BgraSink for MixedSinker<'_, Bgra> {}

impl PixelSink for MixedSinker<'_, Bgra> {
  type Input<'r> = BgraRow<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: BgraRow<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if row.bgra().len() != w * 4 {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::BgraPacked,
        row: idx,
        expected: w * 4,
        actual: row.bgra().len(),
      });
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange {
        row: idx,
        configured_height: self.height,
      });
    }

    let Self {
      rgb,
      rgba,
      luma,
      hsv,
      rgb_scratch,
      ..
    } = self;
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
        rgb_to_hsv_row(
          rgb_row,
          &mut hsv.h[one_plane_start..one_plane_end],
          &mut hsv.s[one_plane_start..one_plane_end],
          &mut hsv.v[one_plane_start..one_plane_end],
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

impl<'a> MixedSinker<'a, Argb> {
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
    let expected = self.frame_bytes(4)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::RgbaBufferTooShort {
        expected,
        actual: buf.len(),
      });
    }
    self.rgba = Some(buf);
    Ok(self)
  }
}

impl ArgbSink for MixedSinker<'_, Argb> {}

impl PixelSink for MixedSinker<'_, Argb> {
  type Input<'r> = ArgbRow<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: ArgbRow<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if row.argb().len() != w * 4 {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::ArgbPacked,
        row: idx,
        expected: w * 4,
        actual: row.argb().len(),
      });
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange {
        row: idx,
        configured_height: self.height,
      });
    }

    let Self {
      rgb,
      rgba,
      luma,
      hsv,
      rgb_scratch,
      ..
    } = self;
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
        rgb_to_hsv_row(
          rgb_row,
          &mut hsv.h[one_plane_start..one_plane_end],
          &mut hsv.s[one_plane_start..one_plane_end],
          &mut hsv.v[one_plane_start..one_plane_end],
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

impl<'a> MixedSinker<'a, Abgr> {
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
    let expected = self.frame_bytes(4)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::RgbaBufferTooShort {
        expected,
        actual: buf.len(),
      });
    }
    self.rgba = Some(buf);
    Ok(self)
  }
}

impl AbgrSink for MixedSinker<'_, Abgr> {}

impl PixelSink for MixedSinker<'_, Abgr> {
  type Input<'r> = AbgrRow<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: AbgrRow<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if row.abgr().len() != w * 4 {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::AbgrPacked,
        row: idx,
        expected: w * 4,
        actual: row.abgr().len(),
      });
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange {
        row: idx,
        configured_height: self.height,
      });
    }

    let Self {
      rgb,
      rgba,
      luma,
      hsv,
      rgb_scratch,
      ..
    } = self;
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
        rgb_to_hsv_row(
          rgb_row,
          &mut hsv.h[one_plane_start..one_plane_end],
          &mut hsv.s[one_plane_start..one_plane_end],
          &mut hsv.v[one_plane_start..one_plane_end],
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

impl<'a> MixedSinker<'a, Xrgb> {
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
    let expected = self.frame_bytes(4)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::RgbaBufferTooShort {
        expected,
        actual: buf.len(),
      });
    }
    self.rgba = Some(buf);
    Ok(self)
  }
}

impl XrgbSink for MixedSinker<'_, Xrgb> {}

impl PixelSink for MixedSinker<'_, Xrgb> {
  type Input<'r> = XrgbRow<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: XrgbRow<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if row.xrgb().len() != w * 4 {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::XrgbPacked,
        row: idx,
        expected: w * 4,
        actual: row.xrgb().len(),
      });
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange {
        row: idx,
        configured_height: self.height,
      });
    }

    let Self {
      rgb,
      rgba,
      luma,
      hsv,
      rgb_scratch,
      ..
    } = self;
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
        rgb_to_hsv_row(
          rgb_row,
          &mut hsv.h[one_plane_start..one_plane_end],
          &mut hsv.s[one_plane_start..one_plane_end],
          &mut hsv.v[one_plane_start..one_plane_end],
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

impl<'a> MixedSinker<'a, Rgbx> {
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
    let expected = self.frame_bytes(4)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::RgbaBufferTooShort {
        expected,
        actual: buf.len(),
      });
    }
    self.rgba = Some(buf);
    Ok(self)
  }
}

impl RgbxSink for MixedSinker<'_, Rgbx> {}

impl PixelSink for MixedSinker<'_, Rgbx> {
  type Input<'r> = RgbxRow<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: RgbxRow<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if row.rgbx().len() != w * 4 {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::RgbxPacked,
        row: idx,
        expected: w * 4,
        actual: row.rgbx().len(),
      });
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange {
        row: idx,
        configured_height: self.height,
      });
    }

    let Self {
      rgb,
      rgba,
      luma,
      hsv,
      rgb_scratch,
      ..
    } = self;
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
        rgb_to_hsv_row(
          rgb_row,
          &mut hsv.h[one_plane_start..one_plane_end],
          &mut hsv.s[one_plane_start..one_plane_end],
          &mut hsv.v[one_plane_start..one_plane_end],
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

impl<'a> MixedSinker<'a, Xbgr> {
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
    let expected = self.frame_bytes(4)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::RgbaBufferTooShort {
        expected,
        actual: buf.len(),
      });
    }
    self.rgba = Some(buf);
    Ok(self)
  }
}

impl XbgrSink for MixedSinker<'_, Xbgr> {}

impl PixelSink for MixedSinker<'_, Xbgr> {
  type Input<'r> = XbgrRow<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: XbgrRow<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if row.xbgr().len() != w * 4 {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::XbgrPacked,
        row: idx,
        expected: w * 4,
        actual: row.xbgr().len(),
      });
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange {
        row: idx,
        configured_height: self.height,
      });
    }

    let Self {
      rgb,
      rgba,
      luma,
      hsv,
      rgb_scratch,
      ..
    } = self;
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
        rgb_to_hsv_row(
          rgb_row,
          &mut hsv.h[one_plane_start..one_plane_end],
          &mut hsv.s[one_plane_start..one_plane_end],
          &mut hsv.v[one_plane_start..one_plane_end],
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

impl<'a> MixedSinker<'a, Bgrx> {
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
    let expected = self.frame_bytes(4)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::RgbaBufferTooShort {
        expected,
        actual: buf.len(),
      });
    }
    self.rgba = Some(buf);
    Ok(self)
  }
}

impl BgrxSink for MixedSinker<'_, Bgrx> {}

impl PixelSink for MixedSinker<'_, Bgrx> {
  type Input<'r> = BgrxRow<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: BgrxRow<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if row.bgrx().len() != w * 4 {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::BgrxPacked,
        row: idx,
        expected: w * 4,
        actual: row.bgrx().len(),
      });
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange {
        row: idx,
        configured_height: self.height,
      });
    }

    let Self {
      rgb,
      rgba,
      luma,
      hsv,
      rgb_scratch,
      ..
    } = self;
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
        rgb_to_hsv_row(
          rgb_row,
          &mut hsv.h[one_plane_start..one_plane_end],
          &mut hsv.s[one_plane_start..one_plane_end],
          &mut hsv.v[one_plane_start..one_plane_end],
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
