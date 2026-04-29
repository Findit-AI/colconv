//! Sinker impls for packed YUV 4:2:2 (8-bit) source formats — Tier 3,
//! Ship 10.
//!
//! Source family covered here:
//! - [`Yuyv422`] — `Y0, U0, Y1, V0, …` (FFmpeg `yuyv422` / YUY2).
//! - [`Uyvy422`] — `U0, Y0, V0, Y1, …` (FFmpeg `uyvy422` / UYVY).
//! - [`Yvyu422`] — `Y0, V0, Y1, U0, …` (FFmpeg `yvyu422` / YVYU).
//!
//! All three formats carry one packed plane of `2 * width` bytes per
//! row. The differences are pure byte permutation within each
//! 4-byte / 2-pixel block; the three dispatchers
//! ([`yuyv422_to_rgb_row`], [`uyvy422_to_rgb_row`],
//! [`yvyu422_to_rgb_row`] and the matching `_to_rgba_row` /
//! `_to_luma_row` siblings) hide that permutation behind a single
//! const-generic kernel template.
//!
//! Outputs map to the sink's standard channels:
//! - `with_rgb` / `with_rgba` — packed YUV → RGB Q15 pipeline (full
//!   `ColorMatrix` + range support inherited from the row); RGBA
//!   alpha is forced to `0xFF` (the source has no alpha channel).
//! - `with_luma` — extracts the Y bytes from the packed plane via
//!   the dedicated luma kernel (much cheaper than a full YUV→RGB
//!   pass).
//! - `with_hsv` — stages an internal RGB scratch (or the user's RGB
//!   buffer if attached) and runs the existing `rgb_to_hsv_row`
//!   kernel.
//!
//! When both RGB and RGBA outputs are requested, the RGBA plane is
//! derived from the just-computed RGB row via
//! [`expand_rgb_to_rgba_row`] (Strategy A — memory-bound copy + 0xFF
//! alpha pad) instead of running a second YUV→RGB kernel. When only
//! RGBA is wanted, the dedicated `_to_rgba_row` kernel writes the
//! RGBA buffer directly without staging RGB.

use super::{
  MixedSinker, MixedSinkerError, RowSlice, check_dimensions_match, rgb_row_buf_or_scratch,
  rgba_plane_row_slice,
};
use crate::{
  PixelSink,
  row::{
    expand_rgb_to_rgba_row, rgb_to_hsv_row, uyvy422_to_luma_row, uyvy422_to_rgb_row,
    uyvy422_to_rgba_row, yuyv422_to_luma_row, yuyv422_to_rgb_row, yuyv422_to_rgba_row,
    yvyu422_to_luma_row, yvyu422_to_rgb_row, yvyu422_to_rgba_row,
  },
  yuv::{
    Uyvy422, Uyvy422Row, Uyvy422Sink, Yuyv422, Yuyv422Row, Yuyv422Sink, Yvyu422, Yvyu422Row,
    Yvyu422Sink,
  },
};

// ---- Yuyv422 impl ------------------------------------------------------

impl<'a> MixedSinker<'a, Yuyv422> {
  /// Attaches a packed **8-bit** RGBA output buffer. Alpha is filled
  /// with constant `0xFF` (the source has no alpha channel).
  ///
  /// Returns `Err(RgbaBufferTooShort)` if
  /// `buf.len() < width × height × 4`, or `Err(GeometryOverflow)` on
  /// 32‑bit targets when the product overflows.
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

impl Yuyv422Sink for MixedSinker<'_, Yuyv422> {}

impl PixelSink for MixedSinker<'_, Yuyv422> {
  type Input<'r> = Yuyv422Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::OddWidth { width: self.width });
    }
    Ok(())
  }

  fn process(&mut self, row: Yuyv422Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if w & 1 != 0 {
      return Err(MixedSinkerError::OddWidth { width: w });
    }

    let packed_expected = w.checked_mul(2).ok_or(MixedSinkerError::GeometryOverflow {
      width: w,
      height: h,
      channels: 2,
    })?;
    if row.yuyv().len() != packed_expected {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::Yuyv422Packed,
        row: idx,
        expected: packed_expected,
        actual: row.yuyv().len(),
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
    let packed = row.yuyv();

    // Luma — extract Y bytes from packed plane via dedicated kernel.
    if let Some(luma) = luma.as_deref_mut() {
      yuyv422_to_luma_row(
        packed,
        &mut luma[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();
    let need_rgb_kernel = want_rgb || want_hsv;

    // Standalone RGBA fast path — no RGB / HSV requested. Run the
    // dedicated RGBA kernel directly into the output buffer; avoids
    // both the scratch allocation and the RGB→RGBA expand pass.
    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      yuyv422_to_rgba_row(
        packed,
        rgba_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
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
    yuyv422_to_rgb_row(packed, rgb_row, w, row.matrix(), row.full_range(), use_simd);

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

    // Strategy A: when both RGB and RGBA are requested, derive RGBA
    // from the just-computed RGB row instead of running a second
    // YUV→RGB kernel.
    if let Some(buf) = rgba.as_deref_mut() {
      let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
    }

    Ok(())
  }
}

// ---- Uyvy422 impl ------------------------------------------------------

impl<'a> MixedSinker<'a, Uyvy422> {
  /// Attaches a packed **8-bit** RGBA output buffer. Alpha is filled
  /// with constant `0xFF` (the source has no alpha channel).
  ///
  /// See [`MixedSinker::<Yuyv422>::with_rgba`] for the same rationale
  /// and constraints; UYVY differs only in byte position (Y in odd
  /// vs even slots).
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

impl Uyvy422Sink for MixedSinker<'_, Uyvy422> {}

impl PixelSink for MixedSinker<'_, Uyvy422> {
  type Input<'r> = Uyvy422Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::OddWidth { width: self.width });
    }
    Ok(())
  }

  fn process(&mut self, row: Uyvy422Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if w & 1 != 0 {
      return Err(MixedSinkerError::OddWidth { width: w });
    }

    let packed_expected = w.checked_mul(2).ok_or(MixedSinkerError::GeometryOverflow {
      width: w,
      height: h,
      channels: 2,
    })?;
    if row.uyvy().len() != packed_expected {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::Uyvy422Packed,
        row: idx,
        expected: packed_expected,
        actual: row.uyvy().len(),
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
    let packed = row.uyvy();

    if let Some(luma) = luma.as_deref_mut() {
      uyvy422_to_luma_row(
        packed,
        &mut luma[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();
    let need_rgb_kernel = want_rgb || want_hsv;

    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      uyvy422_to_rgba_row(
        packed,
        rgba_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
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
    uyvy422_to_rgb_row(packed, rgb_row, w, row.matrix(), row.full_range(), use_simd);

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

// ---- Yvyu422 impl ------------------------------------------------------

impl<'a> MixedSinker<'a, Yvyu422> {
  /// Attaches a packed **8-bit** RGBA output buffer. Alpha is filled
  /// with constant `0xFF` (the source has no alpha channel).
  ///
  /// See [`MixedSinker::<Yuyv422>::with_rgba`] for the same rationale
  /// and constraints; YVYU differs only in chroma byte order (V
  /// before U).
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

impl Yvyu422Sink for MixedSinker<'_, Yvyu422> {}

impl PixelSink for MixedSinker<'_, Yvyu422> {
  type Input<'r> = Yvyu422Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::OddWidth { width: self.width });
    }
    Ok(())
  }

  fn process(&mut self, row: Yvyu422Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if w & 1 != 0 {
      return Err(MixedSinkerError::OddWidth { width: w });
    }

    let packed_expected = w.checked_mul(2).ok_or(MixedSinkerError::GeometryOverflow {
      width: w,
      height: h,
      channels: 2,
    })?;
    if row.yvyu().len() != packed_expected {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::Yvyu422Packed,
        row: idx,
        expected: packed_expected,
        actual: row.yvyu().len(),
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
    let packed = row.yvyu();

    if let Some(luma) = luma.as_deref_mut() {
      yvyu422_to_luma_row(
        packed,
        &mut luma[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();
    let need_rgb_kernel = want_rgb || want_hsv;

    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      yvyu422_to_rgba_row(
        packed,
        rgba_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
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
    yvyu422_to_rgb_row(packed, rgb_row, w, row.matrix(), row.full_range(), use_simd);

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
