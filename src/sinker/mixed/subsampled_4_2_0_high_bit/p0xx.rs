use super::super::{
  MixedSinker, MixedSinkerError, RowSlice, check_dimensions_match, rgb_row_buf_or_scratch,
  rgba_plane_row_slice, rgba_u16_plane_row_slice,
};
use crate::{PixelSink, row::*, yuv::*};

// ---- P010 impl ---------------------------------------------------------

impl<'a> MixedSinker<'a, P010> {
  /// Attaches a packed **`u16`** RGB output buffer. Mirrors
  /// [`MixedSinker<Yuv420p10>::with_rgb_u16`] — compile‑time gated to
  /// sinkers whose source format populates native‑depth RGB.
  ///
  /// Length is measured in `u16` **elements** (not bytes): minimum
  /// `width × height × 3`. Output is **low‑bit‑packed** (10‑bit
  /// values in the low 10 of each `u16`, upper 6 zero) — matches
  /// FFmpeg `yuv420p10le` convention. This is **not** P010 packing
  /// (which puts the 10 bits in the high 10); callers feeding a P010
  /// consumer must shift the output left by 6.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgb_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_rgb_u16(buf)?;
    Ok(self)
  }

  /// In-place variant of [`with_rgb_u16`](Self::with_rgb_u16). The
  /// required length is measured in `u16` **elements**, not bytes.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgb_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
    let expected_elements = self.frame_bytes(3)?;
    if buf.len() < expected_elements {
      return Err(MixedSinkerError::RgbU16BufferTooShort {
        expected: expected_elements,
        actual: buf.len(),
      });
    }
    self.rgb_u16 = Some(buf);
    Ok(self)
  }

  /// Attaches a packed **8‑bit** RGBA output buffer. The 10‑bit P010
  /// source (semi‑planar, high‑bit‑packed) is converted to 8‑bit RGBA
  /// via the `BITS = 10` Q15 kernel family; alpha = `0xFF` (P010 has
  /// no alpha plane).
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

  /// Attaches a packed **`u16`** RGBA output buffer. Output is
  /// **low‑bit‑packed** 10‑bit values (`yuv420p10le` convention) — not
  /// P010 high‑bit packing. Length is measured in `u16` **elements**
  /// (`width × height × 4`). Alpha element is `(1 << 10) - 1`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgba_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_rgba_u16(buf)?;
    Ok(self)
  }
  /// In-place variant of [`with_rgba_u16`](Self::with_rgba_u16).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgba_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_bytes(4)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::RgbaU16BufferTooShort {
        expected,
        actual: buf.len(),
      });
    }
    self.rgba_u16 = Some(buf);
    Ok(self)
  }
}

impl P010Sink for MixedSinker<'_, P010> {}

impl PixelSink for MixedSinker<'_, P010> {
  type Input<'r> = P010Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::OddWidth { width: self.width });
    }
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: P010Row<'_>) -> Result<(), Self::Error> {
    // P010 stores 10‑bit samples high‑bit‑packed; bit depth is fixed
    // by the format. Used for the u16 RGBA expand path's alpha pad.
    const BITS: u32 = 10;

    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if w & 1 != 0 {
      return Err(MixedSinkerError::OddWidth { width: w });
    }
    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::Y10,
        row: idx,
        expected: w,
        actual: row.y().len(),
      });
    }
    // Semi-planar UV: `width` u16 elements total (`width / 2` pairs).
    if row.uv_half().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::UvHalf10,
        row: idx,
        expected: w,
        actual: row.uv_half().len(),
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
      rgb_u16,
      rgba,
      rgba_u16,
      luma,
      hsv,
      rgb_scratch,
      ..
    } = self;

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // Luma: P010 samples are high-bit-packed (`value << 6`). Taking
    // the high byte via `>> 8` gives the top 8 bits of the 10-bit
    // value — functionally equivalent to
    // `(value >> 2)` for the yuv420p10 path.
    if let Some(luma) = luma.as_deref_mut() {
      let dst = &mut luma[one_plane_start..one_plane_end];
      for (d, &s) in dst.iter_mut().zip(row.y().iter()) {
        *d = (s >> 8) as u8;
      }
    }

    // ===== u16 RGB / RGBA path (Strategy A) — see Yuv420p10 for rationale.
    // u16 outputs are low-bit-packed (yuv420p10le convention), not
    // P010's high-bit packing.
    let want_rgb_u16 = rgb_u16.is_some();
    let want_rgba_u16 = rgba_u16.is_some();

    if want_rgba_u16 && !want_rgb_u16 {
      let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row =
        rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
      p010_to_rgba_u16_row(
        row.y(),
        row.uv_half(),
        rgba_u16_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
    } else if want_rgb_u16 {
      let rgb_u16_buf = rgb_u16.as_deref_mut().unwrap();
      let rgb_plane_end =
        one_plane_end
          .checked_mul(3)
          .ok_or(MixedSinkerError::GeometryOverflow {
            width: w,
            height: h,
            channels: 3,
          })?;
      let rgb_plane_start = one_plane_start * 3;
      let rgb_u16_row = &mut rgb_u16_buf[rgb_plane_start..rgb_plane_end];
      p010_to_rgb_u16_row(
        row.y(),
        row.uv_half(),
        rgb_u16_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
      if want_rgba_u16 {
        let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
        let rgba_u16_row =
          rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
        expand_rgb_u16_to_rgba_u16_row::<BITS>(rgb_u16_row, rgba_u16_row, w);
      }
    }

    // ===== u8 RGB / RGBA / HSV path (Strategy A) =====
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();
    let need_rgb_kernel = want_rgb || want_hsv;

    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      p010_to_rgba_row(
        row.y(),
        row.uv_half(),
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

    p010_to_rgb_row(
      row.y(),
      row.uv_half(),
      rgb_row,
      w,
      row.matrix(),
      row.full_range(),
      use_simd,
    );

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

// ---- P012 impl ---------------------------------------------------------

impl<'a> MixedSinker<'a, P012> {
  /// Attaches a packed **`u16`** RGB output buffer. Produces 12‑bit
  /// output in **low‑bit‑packed** `yuv420p12le` convention (values in
  /// `[0, 4095]` in the low 12 of each `u16`, upper 4 zero) —
  /// **not** P012's high‑bit packing. Callers feeding a P012 consumer
  /// must shift the output left by 4.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgb_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_rgb_u16(buf)?;
    Ok(self)
  }

  /// In-place variant of [`with_rgb_u16`](Self::with_rgb_u16).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgb_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
    let expected_elements = self.frame_bytes(3)?;
    if buf.len() < expected_elements {
      return Err(MixedSinkerError::RgbU16BufferTooShort {
        expected: expected_elements,
        actual: buf.len(),
      });
    }
    self.rgb_u16 = Some(buf);
    Ok(self)
  }

  /// Attaches a packed **8‑bit** RGBA output buffer. The 12‑bit P012
  /// source (semi‑planar, high‑bit‑packed) is converted to 8‑bit RGBA
  /// via the `BITS = 12` Q15 kernel family; alpha = `0xFF`.
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

  /// Attaches a packed **`u16`** RGBA output buffer. Output is
  /// **low‑bit‑packed** 12‑bit values (`yuv420p12le` convention) —
  /// not P012 high‑bit packing. Length is measured in `u16`
  /// **elements** (`width × height × 4`). Alpha element is
  /// `(1 << 12) - 1`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgba_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_rgba_u16(buf)?;
    Ok(self)
  }
  /// In-place variant of [`with_rgba_u16`](Self::with_rgba_u16).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgba_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_bytes(4)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::RgbaU16BufferTooShort {
        expected,
        actual: buf.len(),
      });
    }
    self.rgba_u16 = Some(buf);
    Ok(self)
  }
}

impl P012Sink for MixedSinker<'_, P012> {}

impl PixelSink for MixedSinker<'_, P012> {
  type Input<'r> = P012Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::OddWidth { width: self.width });
    }
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: P012Row<'_>) -> Result<(), Self::Error> {
    // P012 stores 12‑bit samples high‑bit‑packed; bit depth is fixed
    // by the format. Used for the u16 RGBA expand path's alpha pad.
    const BITS: u32 = 12;

    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if w & 1 != 0 {
      return Err(MixedSinkerError::OddWidth { width: w });
    }
    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::Y12,
        row: idx,
        expected: w,
        actual: row.y().len(),
      });
    }
    if row.uv_half().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::UvHalf12,
        row: idx,
        expected: w,
        actual: row.uv_half().len(),
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
      rgb_u16,
      rgba,
      rgba_u16,
      luma,
      hsv,
      rgb_scratch,
      ..
    } = self;

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // Luma: P012 samples are high‑bit‑packed (`value << 4`). Taking
    // the high byte via `>> 8` gives the top 8 bits of the 12‑bit
    // value — identical accessor to P010 (both put active bits in the
    // high `BITS` positions of the `u16`).
    if let Some(luma) = luma.as_deref_mut() {
      let dst = &mut luma[one_plane_start..one_plane_end];
      for (d, &s) in dst.iter_mut().zip(row.y().iter()) {
        *d = (s >> 8) as u8;
      }
    }

    // ===== u16 RGB / RGBA path (Strategy A) — see Yuv420p10 for rationale.
    // u16 outputs are low-bit-packed (yuv420p12le convention), not
    // P012's high-bit packing.
    let want_rgb_u16 = rgb_u16.is_some();
    let want_rgba_u16 = rgba_u16.is_some();

    if want_rgba_u16 && !want_rgb_u16 {
      let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row =
        rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
      p012_to_rgba_u16_row(
        row.y(),
        row.uv_half(),
        rgba_u16_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
    } else if want_rgb_u16 {
      let rgb_u16_buf = rgb_u16.as_deref_mut().unwrap();
      let rgb_plane_end =
        one_plane_end
          .checked_mul(3)
          .ok_or(MixedSinkerError::GeometryOverflow {
            width: w,
            height: h,
            channels: 3,
          })?;
      let rgb_plane_start = one_plane_start * 3;
      let rgb_u16_row = &mut rgb_u16_buf[rgb_plane_start..rgb_plane_end];
      p012_to_rgb_u16_row(
        row.y(),
        row.uv_half(),
        rgb_u16_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
      if want_rgba_u16 {
        let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
        let rgba_u16_row =
          rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
        expand_rgb_u16_to_rgba_u16_row::<BITS>(rgb_u16_row, rgba_u16_row, w);
      }
    }

    // ===== u8 RGB / RGBA / HSV path (Strategy A) =====
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();
    let need_rgb_kernel = want_rgb || want_hsv;

    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      p012_to_rgba_row(
        row.y(),
        row.uv_half(),
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

    p012_to_rgb_row(
      row.y(),
      row.uv_half(),
      rgb_row,
      w,
      row.matrix(),
      row.full_range(),
      use_simd,
    );

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

// ---- P016 impl ---------------------------------------------------------

impl<'a> MixedSinker<'a, P016> {
  /// Attaches a packed **`u16`** RGB output buffer. Produces 16‑bit
  /// output in `[0, 65535]` — at 16 bits there is no high‑ vs
  /// low‑packing distinction, so the output matches
  /// [`MixedSinker<Yuv420p16>::with_rgb_u16`] numerically.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgb_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_rgb_u16(buf)?;
    Ok(self)
  }

  /// In-place variant of [`with_rgb_u16`](Self::with_rgb_u16).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgb_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
    let expected_elements = self.frame_bytes(3)?;
    if buf.len() < expected_elements {
      return Err(MixedSinkerError::RgbU16BufferTooShort {
        expected: expected_elements,
        actual: buf.len(),
      });
    }
    self.rgb_u16 = Some(buf);
    Ok(self)
  }

  /// Attaches a packed **8‑bit** RGBA output buffer. The 16‑bit P016
  /// source (semi‑planar) is converted to 8‑bit RGBA via the dedicated
  /// `BITS = 16` kernel family; alpha = `0xFF`.
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

  /// Attaches a packed **`u16`** RGBA output buffer. 16‑bit output
  /// (full `u16` range). Length is measured in `u16` **elements**
  /// (`width × height × 4`). Alpha element is `u16::MAX`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgba_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_rgba_u16(buf)?;
    Ok(self)
  }
  /// In-place variant of [`with_rgba_u16`](Self::with_rgba_u16).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgba_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_bytes(4)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::RgbaU16BufferTooShort {
        expected,
        actual: buf.len(),
      });
    }
    self.rgba_u16 = Some(buf);
    Ok(self)
  }
}

impl P016Sink for MixedSinker<'_, P016> {}

impl PixelSink for MixedSinker<'_, P016> {
  type Input<'r> = P016Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::OddWidth { width: self.width });
    }
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: P016Row<'_>) -> Result<(), Self::Error> {
    // Bit depth is fixed by the format (16). Used for the u16 RGBA
    // expand path's alpha pad (`alpha = u16::MAX` at this depth).
    const BITS: u32 = 16;

    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if w & 1 != 0 {
      return Err(MixedSinkerError::OddWidth { width: w });
    }
    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::Y16,
        row: idx,
        expected: w,
        actual: row.y().len(),
      });
    }
    if row.uv_half().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::UvHalf16,
        row: idx,
        expected: w,
        actual: row.uv_half().len(),
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
      rgb_u16,
      rgba,
      rgba_u16,
      luma,
      hsv,
      rgb_scratch,
      ..
    } = self;

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // Luma: 16‑bit Y value >> 8 is the top byte.
    if let Some(luma) = luma.as_deref_mut() {
      let dst = &mut luma[one_plane_start..one_plane_end];
      for (d, &s) in dst.iter_mut().zip(row.y().iter()) {
        *d = (s >> 8) as u8;
      }
    }

    // ===== u16 RGB / RGBA path (Strategy A) — see Yuv420p10 for rationale.
    let want_rgb_u16 = rgb_u16.is_some();
    let want_rgba_u16 = rgba_u16.is_some();

    if want_rgba_u16 && !want_rgb_u16 {
      let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row =
        rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
      p016_to_rgba_u16_row(
        row.y(),
        row.uv_half(),
        rgba_u16_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
    } else if want_rgb_u16 {
      let rgb_u16_buf = rgb_u16.as_deref_mut().unwrap();
      let rgb_plane_end =
        one_plane_end
          .checked_mul(3)
          .ok_or(MixedSinkerError::GeometryOverflow {
            width: w,
            height: h,
            channels: 3,
          })?;
      let rgb_plane_start = one_plane_start * 3;
      let rgb_u16_row = &mut rgb_u16_buf[rgb_plane_start..rgb_plane_end];
      p016_to_rgb_u16_row(
        row.y(),
        row.uv_half(),
        rgb_u16_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
      if want_rgba_u16 {
        let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
        let rgba_u16_row =
          rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
        expand_rgb_u16_to_rgba_u16_row::<BITS>(rgb_u16_row, rgba_u16_row, w);
      }
    }

    // ===== u8 RGB / RGBA / HSV path (Strategy A) =====
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();
    let need_rgb_kernel = want_rgb || want_hsv;

    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      p016_to_rgba_row(
        row.y(),
        row.uv_half(),
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

    p016_to_rgb_row(
      row.y(),
      row.uv_half(),
      rgb_row,
      w,
      row.matrix(),
      row.full_range(),
      use_simd,
    );

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
