//! Sinker impls for source-side YUVA 4:4:4 formats —
//! [`Yuva444p9`](crate::yuv::Yuva444p9),
//! [`Yuva444p10`](crate::yuv::Yuva444p10),
//! [`Yuva444p12`](crate::yuv::Yuva444p12), and
//! [`Yuva444p14`](crate::yuv::Yuva444p14). The 16-bit sibling
//! (`Yuva444p16`) needs its own SIMD work and lands in Ship 8b‑5.
//!
//! For each format:
//! - **RGBA output paths (u8 + native-depth u16)** — alpha is sourced
//!   from the source alpha plane via the
//!   `yuv_444p_n_to_rgba*_with_alpha_src_row::<BITS>` SIMD/scalar
//!   kernels (BITS-generic across {9, 10, 12, 14}).
//! - **RGB / RGB-u16 / luma / HSV alpha-drop paths** — these reuse the
//!   existing non-alpha 4:4:4 row dispatchers verbatim. The alpha
//!   plane is simply ignored; output bytes / elements match what the
//!   corresponding `Yuv444p<BITS>` source would produce given the
//!   same Y/U/V data. Without these paths
//!   `MixedSinker::with_rgb` / `with_luma` / `with_hsv` (declared on
//!   the generic impl) would silently accept a buffer and never write
//!   it.
//!
//! The 9 / 10 / 12 / 14-bit `process` bodies are structurally
//! identical — only the depth-named row primitives, the `RowSlice`
//! variants used in error reports, and the depth-conversion shift
//! (`BITS - 8`) for luma differ. They share the
//! [`yuva444p_high_bit_process`] helper to avoid 4× of the same ~70
//! lines.

use super::{
  MixedSinker, MixedSinkerError, RowSlice, check_dimensions_match, rgb_row_buf_or_scratch,
  rgba_plane_row_slice, rgba_u16_plane_row_slice,
};
use crate::{PixelSink, row::*, yuv::*};

// ---- Yuva444p9 impl ---------------------------------------------------

impl<'a> MixedSinker<'a, Yuva444p9> {
  /// Attaches a packed **8-bit** RGBA output buffer. The 9-bit YUVA
  /// source is converted to 8-bit RGBA via the same `BITS = 9` Q15
  /// kernel family used by [`MixedSinker<Yuv444p9>::with_rgba`]; the
  /// per-pixel alpha byte is **sourced from the alpha plane**
  /// (depth-converted via `a >> 1` to fit `u8`) — not constant `0xFF`.
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

  /// Attaches a packed **`u16`** RGBA output buffer. 9-bit
  /// low-packed (`[0, 511]`); the per-pixel alpha element is
  /// **sourced from the alpha plane** at native depth.
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
  /// Output is identical to [`MixedSinker<Yuv444p9>::with_rgb_u16`] —
  /// the alpha plane is ignored.
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

impl Yuva444p9Sink for MixedSinker<'_, Yuva444p9> {}

impl PixelSink for MixedSinker<'_, Yuva444p9> {
  type Input<'r> = Yuva444p9Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: Yuva444p9Row<'_>) -> Result<(), Self::Error> {
    yuva444p_high_bit_process::<9, _, _, _, _>(
      self,
      row.row(),
      row.y(),
      row.u(),
      row.v(),
      row.a(),
      row.matrix(),
      row.full_range(),
      RowSlice::Y9,
      RowSlice::UFull9,
      RowSlice::VFull9,
      RowSlice::AFull9,
      yuv444p9_to_rgb_row,
      yuv444p9_to_rgb_u16_row,
      yuva444p9_to_rgba_row,
      yuva444p9_to_rgba_u16_row,
    )
  }
}

// ---- Yuva444p10 impl --------------------------------------------------

impl<'a> MixedSinker<'a, Yuva444p10> {
  /// Attaches a packed **8-bit** RGBA output buffer. The 10-bit YUVA
  /// source is converted to 8-bit RGBA via the same `BITS = 10` Q15
  /// kernel family used by [`MixedSinker<Yuv444p10>::with_rgba`]; the
  /// per-pixel alpha byte is **sourced from the alpha plane**
  /// (depth-converted via `a >> 2` to fit `u8`) — not constant `0xFF`.
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

  /// Attaches a packed **`u16`** RGBA output buffer. 10-bit
  /// low-packed (`[0, 1023]`); the per-pixel alpha element is
  /// **sourced from the alpha plane** at native depth.
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
  /// Output is identical to [`MixedSinker<Yuv444p10>::with_rgb_u16`] —
  /// the alpha plane is ignored.
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

impl Yuva444p10Sink for MixedSinker<'_, Yuva444p10> {}

impl PixelSink for MixedSinker<'_, Yuva444p10> {
  type Input<'r> = Yuva444p10Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: Yuva444p10Row<'_>) -> Result<(), Self::Error> {
    yuva444p_high_bit_process::<10, _, _, _, _>(
      self,
      row.row(),
      row.y(),
      row.u(),
      row.v(),
      row.a(),
      row.matrix(),
      row.full_range(),
      RowSlice::Y10,
      RowSlice::UFull10,
      RowSlice::VFull10,
      RowSlice::AFull10,
      yuv444p10_to_rgb_row,
      yuv444p10_to_rgb_u16_row,
      yuva444p10_to_rgba_row,
      yuva444p10_to_rgba_u16_row,
    )
  }
}

// ---- Yuva444p12 impl --------------------------------------------------

impl<'a> MixedSinker<'a, Yuva444p12> {
  /// Attaches a packed **8-bit** RGBA output buffer. The 12-bit YUVA
  /// source is converted to 8-bit RGBA via the same `BITS = 12` Q15
  /// kernel family used by [`MixedSinker<Yuv444p12>::with_rgba`]; the
  /// per-pixel alpha byte is **sourced from the alpha plane**
  /// (depth-converted via `a >> 4` to fit `u8`) — not constant `0xFF`.
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

  /// Attaches a packed **`u16`** RGBA output buffer. 12-bit
  /// low-packed (`[0, 4095]`); the per-pixel alpha element is
  /// **sourced from the alpha plane** at native depth.
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
  /// Output is identical to [`MixedSinker<Yuv444p12>::with_rgb_u16`] —
  /// the alpha plane is ignored.
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

impl Yuva444p12Sink for MixedSinker<'_, Yuva444p12> {}

impl PixelSink for MixedSinker<'_, Yuva444p12> {
  type Input<'r> = Yuva444p12Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: Yuva444p12Row<'_>) -> Result<(), Self::Error> {
    yuva444p_high_bit_process::<12, _, _, _, _>(
      self,
      row.row(),
      row.y(),
      row.u(),
      row.v(),
      row.a(),
      row.matrix(),
      row.full_range(),
      RowSlice::Y12,
      RowSlice::UFull12,
      RowSlice::VFull12,
      RowSlice::AFull12,
      yuv444p12_to_rgb_row,
      yuv444p12_to_rgb_u16_row,
      yuva444p12_to_rgba_row,
      yuva444p12_to_rgba_u16_row,
    )
  }
}

// ---- Yuva444p14 impl --------------------------------------------------

impl<'a> MixedSinker<'a, Yuva444p14> {
  /// Attaches a packed **8-bit** RGBA output buffer. The 14-bit YUVA
  /// source is converted to 8-bit RGBA via the same `BITS = 14` Q15
  /// kernel family used by [`MixedSinker<Yuv444p14>::with_rgba`]; the
  /// per-pixel alpha byte is **sourced from the alpha plane**
  /// (depth-converted via `a >> 6` to fit `u8`) — not constant `0xFF`.
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

  /// Attaches a packed **`u16`** RGBA output buffer. 14-bit
  /// low-packed (`[0, 16383]`); the per-pixel alpha element is
  /// **sourced from the alpha plane** at native depth.
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
  /// Output is identical to [`MixedSinker<Yuv444p14>::with_rgb_u16`] —
  /// the alpha plane is ignored.
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

impl Yuva444p14Sink for MixedSinker<'_, Yuva444p14> {}

impl PixelSink for MixedSinker<'_, Yuva444p14> {
  type Input<'r> = Yuva444p14Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: Yuva444p14Row<'_>) -> Result<(), Self::Error> {
    yuva444p_high_bit_process::<14, _, _, _, _>(
      self,
      row.row(),
      row.y(),
      row.u(),
      row.v(),
      row.a(),
      row.matrix(),
      row.full_range(),
      RowSlice::Y14,
      RowSlice::UFull14,
      RowSlice::VFull14,
      RowSlice::AFull14,
      yuv444p14_to_rgb_row,
      yuv444p14_to_rgb_u16_row,
      yuva444p14_to_rgba_row,
      yuva444p14_to_rgba_u16_row,
    )
  }
}

// ---- Yuva444p16 impl --------------------------------------------------

impl<'a> MixedSinker<'a, Yuva444p16> {
  /// Attaches a packed **8-bit** RGBA output buffer. The 16-bit YUVA
  /// source is converted to 8-bit RGBA via the same `BITS = 16` Q15
  /// kernel family used by [`MixedSinker<Yuv444p16>::with_rgba`]; the
  /// per-pixel alpha byte is **sourced from the alpha plane**
  /// (depth-converted via `a >> 8` to fit `u8`) — not constant `0xFF`.
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

  /// Attaches a packed **`u16`** RGBA output buffer. 16-bit
  /// low-packed (`[0, 65535]`); the per-pixel alpha element is
  /// **sourced from the alpha plane** at native depth.
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
  /// Output is identical to [`MixedSinker<Yuv444p16>::with_rgb_u16`] —
  /// the alpha plane is ignored.
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

impl Yuva444p16Sink for MixedSinker<'_, Yuva444p16> {}

impl PixelSink for MixedSinker<'_, Yuva444p16> {
  type Input<'r> = Yuva444p16Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: Yuva444p16Row<'_>) -> Result<(), Self::Error> {
    yuva444p_high_bit_process::<16, _, _, _, _>(
      self,
      row.row(),
      row.y(),
      row.u(),
      row.v(),
      row.a(),
      row.matrix(),
      row.full_range(),
      RowSlice::Y16,
      RowSlice::UFull16,
      RowSlice::VFull16,
      RowSlice::AFull16,
      yuv444p16_to_rgb_row,
      yuv444p16_to_rgb_u16_row,
      yuva444p16_to_rgba_row,
      yuva444p16_to_rgba_u16_row,
    )
  }
}

// ---- Shared high-bit YUVA 4:4:4 process body --------------------------
//
// The 9 / 10-bit YUVA 4:4:4 sinker `process` bodies are structurally
// identical — only the depth-named row primitives, the `RowSlice`
// variants used in error reports, and the depth-conversion shift
// (`BITS - 8`) for luma differ. Factor into a generic helper to avoid
// 2× of the same ~70 lines.
//
// 4:4:4 chroma is full-width (one U / V sample per Y pixel — no
// chroma duplication step), unlike the 4:2:0 sibling helper which
// takes half-width chroma rows.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
#[cfg_attr(not(tarpaulin), inline(always))]
fn yuva444p_high_bit_process<
  const BITS: u32,
  F: crate::SourceFormat,
  RgbRowFn: Fn(&[u16], &[u16], &[u16], &mut [u8], usize, crate::ColorMatrix, bool, bool),
  RgbU16RowFn: Fn(&[u16], &[u16], &[u16], &mut [u16], usize, crate::ColorMatrix, bool, bool),
  RgbaRowFn: Fn(&[u16], &[u16], &[u16], &[u16], &mut [u8], usize, crate::ColorMatrix, bool, bool),
>(
  sinker: &mut MixedSinker<'_, F>,
  idx: usize,
  y_row: &[u16],
  u_row: &[u16],
  v_row: &[u16],
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

  if y_row.len() != w {
    return Err(MixedSinkerError::RowShapeMismatch {
      which: y_slice,
      row: idx,
      expected: w,
      actual: y_row.len(),
    });
  }
  if u_row.len() != w {
    return Err(MixedSinkerError::RowShapeMismatch {
      which: u_slice,
      row: idx,
      expected: w,
      actual: u_row.len(),
    });
  }
  if v_row.len() != w {
    return Err(MixedSinkerError::RowShapeMismatch {
      which: v_slice,
      row: idx,
      expected: w,
      actual: v_row.len(),
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
  // rgb_u16 (alpha-drop) reuses the non-alpha dispatcher; rgba_u16
  // routes through the alpha-source-aware dispatcher. Both attached
  // → run separately (Strategy B fork — alpha must come from source
  // plane, so cheap pad won't work).
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
      u_row,
      v_row,
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
      u_row,
      v_row,
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
      y_row, u_row, v_row, a_row, rgba_row, w, matrix, full_range, use_simd,
    );
    return Ok(());
  }

  if !need_rgb_kernel {
    return Ok(());
  }

  // RGB kernel (alpha-drop reuses the non-alpha dispatcher verbatim).
  let rgb_row = rgb_row_buf_or_scratch(
    rgb.as_deref_mut(),
    rgb_scratch,
    one_plane_start,
    one_plane_end,
    w,
    h,
  )?;
  rgb_dispatch(
    y_row, u_row, v_row, rgb_row, w, matrix, full_range, use_simd,
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

  // Both rgb and rgba attached: run the RGBA kernel for the
  // alpha-aware buffer (cannot reuse rgb_row's RGB output because
  // alpha must come from the source plane; expand-to-rgba would
  // splat 0xFF). Strategy B forks per buffer when alpha is present.
  if want_rgba {
    let rgba_buf = rgba.as_deref_mut().unwrap();
    let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
    rgba_dispatch(
      y_row, u_row, v_row, a_row, rgba_row, w, matrix, full_range, use_simd,
    );
  }

  Ok(())
}
