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
//! 8-bit packed RGB has no `u16` output flavor — `with_rgb_u16` /
//! `with_rgba_u16` are not declared on these source impls.

use super::{
  MixedSinker, MixedSinkerError, RowSlice, check_dimensions_match, rgb_row_buf_or_scratch,
  rgba_plane_row_slice,
};
use crate::{
  PixelSink,
  row::{
    bgr_to_rgb_row, bgra_to_rgb_row, bgra_to_rgba_row, expand_rgb_to_rgba_row, rgb_to_hsv_row,
    rgb_to_luma_row, rgba_to_rgb_row,
  },
  yuv::{
    Bgr24, Bgr24Row, Bgr24Sink, Bgra, BgraRow, BgraSink, Rgb24, Rgb24Row, Rgb24Sink, Rgba, RgbaRow,
    RgbaSink,
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
