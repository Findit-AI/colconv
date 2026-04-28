//! Sinker impls for source-side YUVA 4:2:2 formats — Yuva422p (8-bit),
//! Yuva422p9, Yuva422p10, Yuva422p16. Wiring-only: per-row chroma
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

use super::{
  MixedSinker, MixedSinkerError, RowSlice, check_dimensions_match, rgb_row_buf_or_scratch,
  rgba_plane_row_slice, rgba_u16_plane_row_slice,
};
use crate::{PixelSink, row::*, yuv::*};

// ---- Yuva422p impl (8-bit) ---------------------------------------------

impl<'a> MixedSinker<'a, Yuva422p> {
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

impl Yuva422pSink for MixedSinker<'_, Yuva422p> {}

impl PixelSink for MixedSinker<'_, Yuva422p> {
  type Input<'r> = Yuva422pRow<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::OddWidth { width: self.width });
    }
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: Yuva422pRow<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if w & 1 != 0 {
      return Err(MixedSinkerError::OddWidth { width: w });
    }
    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::Y,
        row: idx,
        expected: w,
        actual: row.y().len(),
      });
    }
    if row.u_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::UHalf,
        row: idx,
        expected: w / 2,
        actual: row.u_half().len(),
      });
    }
    if row.v_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::VHalf,
        row: idx,
        expected: w / 2,
        actual: row.v_half().len(),
      });
    }
    if row.a().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::AFull,
        row: idx,
        expected: w,
        actual: row.a().len(),
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

    // Luma — copy the Y plane verbatim (8-bit YUVA's Y is already u8).
    if let Some(luma) = luma.as_deref_mut() {
      luma[one_plane_start..one_plane_end].copy_from_slice(&row.y()[..w]);
    }

    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();
    let need_rgb_kernel = want_rgb || want_hsv;

    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
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
      return Ok(());
    }

    if !need_rgb_kernel {
      return Ok(());
    }

    // RGB kernel — alpha-drop reuses the 4:2:0 row dispatcher
    // (yuv_420_to_rgb_row); per-row chroma layout is identical to
    // 4:2:2.
    let rgb_row = rgb_row_buf_or_scratch(
      rgb.as_deref_mut(),
      rgb_scratch,
      one_plane_start,
      one_plane_end,
      w,
      h,
    )?;
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

    if want_rgba {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
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

    Ok(())
  }
}

// ---- Yuva422p9 impl ---------------------------------------------------

impl<'a> MixedSinker<'a, Yuva422p9> {
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
}

impl Yuva422p9Sink for MixedSinker<'_, Yuva422p9> {}

impl PixelSink for MixedSinker<'_, Yuva422p9> {
  type Input<'r> = Yuva422p9Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::OddWidth { width: self.width });
    }
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: Yuva422p9Row<'_>) -> Result<(), Self::Error> {
    yuva422p_high_bit_process::<9, _, _, _, _>(
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
      yuv420p9_to_rgb_row,
      yuv420p9_to_rgb_u16_row,
      yuva420p9_to_rgba_row,
      yuva420p9_to_rgba_u16_row,
    )
  }
}

// ---- Yuva422p10 impl --------------------------------------------------

impl<'a> MixedSinker<'a, Yuva422p10> {
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

  /// Attaches a packed **`u16`** RGB output buffer (alpha-drop).
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
}

impl Yuva422p10Sink for MixedSinker<'_, Yuva422p10> {}

impl PixelSink for MixedSinker<'_, Yuva422p10> {
  type Input<'r> = Yuva422p10Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::OddWidth { width: self.width });
    }
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: Yuva422p10Row<'_>) -> Result<(), Self::Error> {
    yuva422p_high_bit_process::<10, _, _, _, _>(
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
      yuv420p10_to_rgb_row,
      yuv420p10_to_rgb_u16_row,
      yuva420p10_to_rgba_row,
      yuva420p10_to_rgba_u16_row,
    )
  }
}

// ---- Yuva422p12 impl --------------------------------------------------

impl<'a> MixedSinker<'a, Yuva422p12> {
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

  /// Attaches a packed **`u16`** RGB output buffer (alpha-drop).
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
}

impl Yuva422p12Sink for MixedSinker<'_, Yuva422p12> {}

impl PixelSink for MixedSinker<'_, Yuva422p12> {
  type Input<'r> = Yuva422p12Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::OddWidth { width: self.width });
    }
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: Yuva422p12Row<'_>) -> Result<(), Self::Error> {
    yuva422p_high_bit_process::<12, _, _, _, _>(
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
      yuv420p12_to_rgb_row,
      yuv420p12_to_rgb_u16_row,
      yuva420p12_to_rgba_row,
      yuva420p12_to_rgba_u16_row,
    )
  }
}

// ---- Yuva422p16 impl --------------------------------------------------

impl<'a> MixedSinker<'a, Yuva422p16> {
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

  /// Attaches a packed **`u16`** RGB output buffer (alpha-drop).
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
}

impl Yuva422p16Sink for MixedSinker<'_, Yuva422p16> {}

impl PixelSink for MixedSinker<'_, Yuva422p16> {
  type Input<'r> = Yuva422p16Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::OddWidth { width: self.width });
    }
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: Yuva422p16Row<'_>) -> Result<(), Self::Error> {
    yuva422p_high_bit_process::<16, _, _, _, _>(
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
      yuv420p16_to_rgb_row,
      yuv420p16_to_rgb_u16_row,
      yuva420p16_to_rgba_row,
      yuva420p16_to_rgba_u16_row,
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
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
#[cfg_attr(not(tarpaulin), inline(always))]
fn yuva422p_high_bit_process<
  const BITS: u32,
  F: crate::SourceFormat,
  RgbRowFn: Fn(&[u16], &[u16], &[u16], &mut [u8], usize, crate::ColorMatrix, bool, bool),
  RgbU16RowFn: Fn(&[u16], &[u16], &[u16], &mut [u16], usize, crate::ColorMatrix, bool, bool),
  RgbaRowFn: Fn(&[u16], &[u16], &[u16], &[u16], &mut [u8], usize, crate::ColorMatrix, bool, bool),
>(
  sinker: &mut MixedSinker<'_, F>,
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
  rgb_dispatch: RgbRowFn,
  rgb_u16_dispatch: RgbU16RowFn,
  rgba_dispatch: RgbaRowFn,
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
  ),
) -> Result<(), MixedSinkerError> {
  let w = sinker.width;
  let h = sinker.height;
  let use_simd = sinker.simd;

  if w & 1 != 0 {
    return Err(MixedSinkerError::OddWidth { width: w });
  }
  if y_row.len() != w {
    return Err(MixedSinkerError::RowShapeMismatch {
      which: y_slice,
      row: idx,
      expected: w,
      actual: y_row.len(),
    });
  }
  if u_half_row.len() != w / 2 {
    return Err(MixedSinkerError::RowShapeMismatch {
      which: u_slice,
      row: idx,
      expected: w / 2,
      actual: u_half_row.len(),
    });
  }
  if v_half_row.len() != w / 2 {
    return Err(MixedSinkerError::RowShapeMismatch {
      which: v_slice,
      row: idx,
      expected: w / 2,
      actual: v_half_row.len(),
    });
  }
  if a_row.len() != w {
    return Err(MixedSinkerError::RowShapeMismatch {
      which: a_slice,
      row: idx,
      expected: w,
      actual: a_row.len(),
    });
  }
  if idx >= sinker.height {
    return Err(MixedSinkerError::RowIndexOutOfRange {
      row: idx,
      configured_height: sinker.height,
    });
  }

  let MixedSinker {
    rgb,
    rgb_u16,
    rgba,
    rgba_u16,
    luma,
    hsv,
    rgb_scratch,
    ..
  } = sinker;
  let one_plane_start = idx * w;
  let one_plane_end = one_plane_start + w;

  // ---- luma (alpha-drop; downshift by BITS - 8 to fit u8) -------
  if let Some(luma) = luma.as_deref_mut() {
    let dst = &mut luma[one_plane_start..one_plane_end];
    for (d, &s) in dst.iter_mut().zip(y_row.iter()) {
      *d = (s >> (BITS - 8)) as u8;
    }
  }

  // ---- u16 RGB / RGBA path --------------------------------------
  if let Some(buf) = rgb_u16.as_deref_mut() {
    let rgb_plane_end = one_plane_end
      .checked_mul(3)
      .ok_or(MixedSinkerError::GeometryOverflow {
        width: w,
        height: h,
        channels: 3,
      })?;
    let rgb_plane_start = one_plane_start * 3;
    let rgb_u16_row = &mut buf[rgb_plane_start..rgb_plane_end];
    rgb_u16_dispatch(
      y_row,
      u_half_row,
      v_half_row,
      rgb_u16_row,
      w,
      matrix,
      full_range,
      use_simd,
    );
  }
  if let Some(buf) = rgba_u16.as_deref_mut() {
    let rgba_u16_row = rgba_u16_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
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
    );
  }

  // ---- u8 RGB / RGBA / HSV path ----------------------------------
  let want_rgb = rgb.is_some();
  let want_rgba = rgba.is_some();
  let want_hsv = hsv.is_some();
  let need_rgb_kernel = want_rgb || want_hsv;

  if want_rgba && !need_rgb_kernel {
    let rgba_buf = rgba.as_deref_mut().unwrap();
    let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
    rgba_dispatch(
      y_row, u_half_row, v_half_row, a_row, rgba_row, w, matrix, full_range, use_simd,
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
  rgb_dispatch(
    y_row, u_half_row, v_half_row, rgb_row, w, matrix, full_range, use_simd,
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

  if want_rgba {
    let rgba_buf = rgba.as_deref_mut().unwrap();
    let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
    rgba_dispatch(
      y_row, u_half_row, v_half_row, a_row, rgba_row, w, matrix, full_range, use_simd,
    );
  }

  Ok(())
}
