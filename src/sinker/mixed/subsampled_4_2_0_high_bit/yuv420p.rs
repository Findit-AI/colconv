use super::super::{
  MixedSinker, MixedSinkerError, RowSlice, check_dimensions_match, rgb_row_buf_or_scratch,
  rgba_plane_row_slice, rgba_u16_plane_row_slice,
};
use crate::{PixelSink, row::*, yuv::*};

// ---- Yuv420p9 impl -----------------------------------------------------
//
// 9-bit 4:2:0 planar. AV_PIX_FMT_YUV420P9LE — niche AVC High 9 only.
// Reuses the Q15 i32 kernel family at `BITS = 9` via the
// `yuv420p9_to_rgb_*` row primitives (which dispatch to
// `yuv_420p_n_to_rgb_*<9>` internally).

impl<'a> MixedSinker<'a, Yuv420p9> {
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
    let expected = self.frame_bytes(3)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::RgbU16BufferTooShort {
        expected,
        actual: buf.len(),
      });
    }
    self.rgb_u16 = Some(buf);
    Ok(self)
  }

  /// Attaches a packed **8‑bit** RGBA output buffer. The 9‑bit YUV
  /// source is converted to 8‑bit RGBA via the same `BITS = 9` Q15
  /// kernel family used by [`Self::with_rgb`]; the fourth byte per
  /// pixel is alpha = `0xFF` (Yuv420p9 has no alpha plane).
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

  /// Attaches a packed **`u16`** RGBA output buffer. 9‑bit low‑packed
  /// (`(1 << 9) - 1 = 511` max). Length is measured in `u16`
  /// **elements** (`width × height × 4`). Alpha element is
  /// `(1 << 9) - 1`.
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

impl Yuv420p9Sink for MixedSinker<'_, Yuv420p9> {}

impl PixelSink for MixedSinker<'_, Yuv420p9> {
  type Input<'r> = Yuv420p9Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::OddWidth { width: self.width });
    }
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: Yuv420p9Row<'_>) -> Result<(), Self::Error> {
    const BITS: u32 = 9;
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if w & 1 != 0 {
      return Err(MixedSinkerError::OddWidth { width: w });
    }
    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::Y9,
        row: idx,
        expected: w,
        actual: row.y().len(),
      });
    }
    if row.u_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::UHalf9,
        row: idx,
        expected: w / 2,
        actual: row.u_half().len(),
      });
    }
    if row.v_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::VHalf9,
        row: idx,
        expected: w / 2,
        actual: row.v_half().len(),
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

    if let Some(luma) = luma.as_deref_mut() {
      let dst = &mut luma[one_plane_start..one_plane_end];
      for (d, &s) in dst.iter_mut().zip(row.y().iter()) {
        *d = (s >> (BITS - 8)) as u8;
      }
    }

    // ===== u16 RGB / RGBA path (Strategy A) =====
    // Compute u16 RGB once (to caller's buffer when attached) and fan
    // out to u16 RGBA via the cheap per-pixel pad. RGBA-only avoids the
    // RGB kernel entirely and writes RGBA directly.
    let want_rgb_u16 = rgb_u16.is_some();
    let want_rgba_u16 = rgba_u16.is_some();

    if want_rgba_u16 && !want_rgb_u16 {
      let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row =
        rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
      yuv420p9_to_rgba_u16_row(
        row.y(),
        row.u_half(),
        row.v_half(),
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
      yuv420p9_to_rgb_u16_row(
        row.y(),
        row.u_half(),
        row.v_half(),
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
      yuv420p9_to_rgba_row(
        row.y(),
        row.u_half(),
        row.v_half(),
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

    yuv420p9_to_rgb_row(
      row.y(),
      row.u_half(),
      row.v_half(),
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

// ---- Yuv420p10 impl -----------------------------------------------------

impl<'a> MixedSinker<'a, Yuv420p10> {
  /// Attaches a packed **`u16`** RGB output buffer. Only available on
  /// sinkers whose source format populates native‑depth `u16` RGB —
  /// calling `with_rgb_u16` on an 8‑bit source sinker (e.g.
  /// [`MixedSinker<Yuv420p>`]) is a compile error rather than a
  /// silent no‑op that would leave the caller's buffer stale.
  ///
  /// Length is measured in `u16` **elements** (not bytes): minimum
  /// `width × height × 3`. Each element carries a 10‑bit value in
  /// the **low** 10 bits (upper 6 bits zero), matching FFmpeg's
  /// `yuv420p10le` convention. This is **not** the `p010` layout
  /// (which stores samples in the high 10 bits); callers feeding a
  /// p010 consumer must shift the output left by 6.
  ///
  /// Returns `Err(RgbU16BufferTooShort)` if
  /// `buf.len() < width × height × 3`, or `Err(GeometryOverflow)`
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
    // Packed RGB requires `width × height × 3` channel values —
    // that's the same count whether the element type is `u8` or
    // `u16`, so the [`Self::frame_bytes`] helper (named for the u8
    // RGB path's byte count) gives the element count here too. No
    // size conversion needed.
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

  /// Attaches a packed **`u16`** RGBA output buffer. 10‑bit
  /// low‑packed (`(1 << 10) - 1 = 1023` max). Length is measured in
  /// `u16` **elements** (`width × height × 4`). Alpha element is
  /// `(1 << 10) - 1`.
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

impl Yuv420p10Sink for MixedSinker<'_, Yuv420p10> {}

impl PixelSink for MixedSinker<'_, Yuv420p10> {
  type Input<'r> = Yuv420p10Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::OddWidth { width: self.width });
    }
    check_dimensions_match(self.width, self.height, width, height)
  }

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
    if row.u_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::UHalf10,
        row: idx,
        expected: w / 2,
        actual: row.u_half().len(),
      });
    }
    if row.v_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::VHalf10,
        row: idx,
        expected: w / 2,
        actual: row.v_half().len(),
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

    // Luma: downshift 10‑bit Y to 8‑bit for the existing u8 luma
    // buffer contract. Bit‑extension by `(BITS - 8)` preserves the
    // most significant bits — functionally equivalent to FFmpeg's
    // `>> (BITS - 8)` conversion used by many downstream analyses.
    if let Some(luma) = luma.as_deref_mut() {
      let dst = &mut luma[one_plane_start..one_plane_end];
      for (d, &s) in dst.iter_mut().zip(row.y().iter()) {
        *d = (s >> (BITS - 8)) as u8;
      }
    }

    // ===== u16 RGB / RGBA path (Strategy A) =====
    // u16 outputs are written via the native-depth row primitive, kept
    // independent of the u8 path: the two have different scale params
    // inside `range_params_n` and can't share an intermediate without
    // losing precision. Within the u16 family, however, the RGB row
    // and RGBA row are bit-identical for R/G/B, so we run the RGB
    // kernel once and fan out to RGBA via the cheap pad.
    let want_rgb_u16 = rgb_u16.is_some();
    let want_rgba_u16 = rgba_u16.is_some();

    if want_rgba_u16 && !want_rgb_u16 {
      let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row =
        rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
      yuv420p10_to_rgba_u16_row(
        row.y(),
        row.u_half(),
        row.v_half(),
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
      yuv420p10_to_rgb_u16_row(
        row.y(),
        row.u_half(),
        row.v_half(),
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
      yuv420p10_to_rgba_row(
        row.y(),
        row.u_half(),
        row.v_half(),
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

    yuv420p10_to_rgb_row(
      row.y(),
      row.u_half(),
      row.v_half(),
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

// ---- Yuv420p12 impl ----------------------------------------------------

impl<'a> MixedSinker<'a, Yuv420p12> {
  /// Attaches a packed **`u16`** RGB output buffer. Mirrors
  /// [`MixedSinker<Yuv420p10>::with_rgb_u16`] but produces 12‑bit
  /// output (values in `[0, 4095]` in the low 12 of each `u16`, upper
  /// 4 zero). Length is measured in `u16` **elements** (`width ×
  /// height × 3`).
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

  /// Attaches a packed **`u16`** RGBA output buffer. 12‑bit
  /// low‑packed (`(1 << 12) - 1 = 4095` max). Length is measured in
  /// `u16` **elements** (`width × height × 4`). Alpha element is
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

impl Yuv420p12Sink for MixedSinker<'_, Yuv420p12> {}

impl PixelSink for MixedSinker<'_, Yuv420p12> {
  type Input<'r> = Yuv420p12Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::OddWidth { width: self.width });
    }
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: Yuv420p12Row<'_>) -> Result<(), Self::Error> {
    // Bit depth is fixed by the format (12) — declared as a const so
    // the downshift for u8 luma stays obvious at the call site.
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
    if row.u_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::UHalf12,
        row: idx,
        expected: w / 2,
        actual: row.u_half().len(),
      });
    }
    if row.v_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::VHalf12,
        row: idx,
        expected: w / 2,
        actual: row.v_half().len(),
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

    if let Some(luma) = luma.as_deref_mut() {
      let dst = &mut luma[one_plane_start..one_plane_end];
      for (d, &s) in dst.iter_mut().zip(row.y().iter()) {
        *d = (s >> (BITS - 8)) as u8;
      }
    }

    // ===== u16 RGB / RGBA path (Strategy A) — see Yuv420p10 for rationale.
    let want_rgb_u16 = rgb_u16.is_some();
    let want_rgba_u16 = rgba_u16.is_some();

    if want_rgba_u16 && !want_rgb_u16 {
      let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row =
        rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
      yuv420p12_to_rgba_u16_row(
        row.y(),
        row.u_half(),
        row.v_half(),
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
      yuv420p12_to_rgb_u16_row(
        row.y(),
        row.u_half(),
        row.v_half(),
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
      yuv420p12_to_rgba_row(
        row.y(),
        row.u_half(),
        row.v_half(),
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

    yuv420p12_to_rgb_row(
      row.y(),
      row.u_half(),
      row.v_half(),
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

// ---- Yuv420p14 impl ----------------------------------------------------

impl<'a> MixedSinker<'a, Yuv420p14> {
  /// Attaches a packed **`u16`** RGB output buffer. Produces 14‑bit
  /// output (values in `[0, 16383]` in the low 14 of each `u16`, upper
  /// 2 zero). Length is measured in `u16` **elements** (`width ×
  /// height × 3`).
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

  /// Attaches a packed **`u16`** RGBA output buffer. 14‑bit
  /// low‑packed (`(1 << 14) - 1 = 16383` max). Length is measured in
  /// `u16` **elements** (`width × height × 4`). Alpha element is
  /// `(1 << 14) - 1`.
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

impl Yuv420p14Sink for MixedSinker<'_, Yuv420p14> {}

impl PixelSink for MixedSinker<'_, Yuv420p14> {
  type Input<'r> = Yuv420p14Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::OddWidth { width: self.width });
    }
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: Yuv420p14Row<'_>) -> Result<(), Self::Error> {
    const BITS: u32 = 14;

    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if w & 1 != 0 {
      return Err(MixedSinkerError::OddWidth { width: w });
    }
    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::Y14,
        row: idx,
        expected: w,
        actual: row.y().len(),
      });
    }
    if row.u_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::UHalf14,
        row: idx,
        expected: w / 2,
        actual: row.u_half().len(),
      });
    }
    if row.v_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::VHalf14,
        row: idx,
        expected: w / 2,
        actual: row.v_half().len(),
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

    if let Some(luma) = luma.as_deref_mut() {
      let dst = &mut luma[one_plane_start..one_plane_end];
      for (d, &s) in dst.iter_mut().zip(row.y().iter()) {
        *d = (s >> (BITS - 8)) as u8;
      }
    }

    // ===== u16 RGB / RGBA path (Strategy A) — see Yuv420p10 for rationale.
    let want_rgb_u16 = rgb_u16.is_some();
    let want_rgba_u16 = rgba_u16.is_some();

    if want_rgba_u16 && !want_rgb_u16 {
      let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row =
        rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
      yuv420p14_to_rgba_u16_row(
        row.y(),
        row.u_half(),
        row.v_half(),
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
      yuv420p14_to_rgb_u16_row(
        row.y(),
        row.u_half(),
        row.v_half(),
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
      yuv420p14_to_rgba_row(
        row.y(),
        row.u_half(),
        row.v_half(),
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

    yuv420p14_to_rgb_row(
      row.y(),
      row.u_half(),
      row.v_half(),
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

// ---- Yuv420p16 impl ----------------------------------------------------

impl<'a> MixedSinker<'a, Yuv420p16> {
  /// Attaches a packed **`u16`** RGB output buffer. Produces 16‑bit
  /// output (values in `[0, 65535]` — full `u16` range). Length is
  /// measured in `u16` **elements** (`width × height × 3`).
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

impl Yuv420p16Sink for MixedSinker<'_, Yuv420p16> {}

impl PixelSink for MixedSinker<'_, Yuv420p16> {
  type Input<'r> = Yuv420p16Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::OddWidth { width: self.width });
    }
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: Yuv420p16Row<'_>) -> Result<(), Self::Error> {
    // Luma downshift is `>> 8` — top 8 bits of the 16-bit Y value.
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
    if row.u_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::UHalf16,
        row: idx,
        expected: w / 2,
        actual: row.u_half().len(),
      });
    }
    if row.v_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::VHalf16,
        row: idx,
        expected: w / 2,
        actual: row.v_half().len(),
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

    if let Some(luma) = luma.as_deref_mut() {
      let dst = &mut luma[one_plane_start..one_plane_end];
      for (d, &s) in dst.iter_mut().zip(row.y().iter()) {
        *d = (s >> (BITS - 8)) as u8;
      }
    }

    // ===== u16 RGB / RGBA path (Strategy A) — see Yuv420p10 for rationale.
    let want_rgb_u16 = rgb_u16.is_some();
    let want_rgba_u16 = rgba_u16.is_some();

    if want_rgba_u16 && !want_rgb_u16 {
      let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row =
        rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
      yuv420p16_to_rgba_u16_row(
        row.y(),
        row.u_half(),
        row.v_half(),
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
      yuv420p16_to_rgb_u16_row(
        row.y(),
        row.u_half(),
        row.v_half(),
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
      yuv420p16_to_rgba_row(
        row.y(),
        row.u_half(),
        row.v_half(),
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

    yuv420p16_to_rgb_row(
      row.y(),
      row.u_half(),
      row.v_half(),
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
