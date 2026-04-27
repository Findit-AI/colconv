//! Sinker impls for source-side YUVA 4:4:4 formats
//! ([`Yuva444p10`](crate::yuv::Yuva444p10) in this PR; other Yuva444p
//! variants land in later Ship 8b tranches).
//!
//! Tranche 8b‑1a wires only the RGBA output paths (u8 + native-depth
//! u16) — the source alpha plane flows through to the
//! `yuva444p10_to_rgba_*` row dispatchers, which currently route to the
//! scalar `yuv_444p_n_to_rgba*_with_alpha_src_row::<10>` path. SIMD
//! lands in 8b‑1b (u8) / 8b‑1c (u16). RGB / luma / HSV builders are
//! intentionally not added in this tranche.

use super::{
  MixedSinker, MixedSinkerError, RowSlice, check_dimensions_match, rgba_plane_row_slice,
  rgba_u16_plane_row_slice,
};
use crate::{PixelSink, row::*, yuv::*};

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
}

impl Yuva444p10Sink for MixedSinker<'_, Yuva444p10> {}

impl PixelSink for MixedSinker<'_, Yuva444p10> {
  type Input<'r> = Yuva444p10Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: Yuva444p10Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::Y10,
        row: idx,
        expected: w,
        actual: row.y().len(),
      });
    }
    if row.u().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::UFull10,
        row: idx,
        expected: w,
        actual: row.u().len(),
      });
    }
    if row.v().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::VFull10,
        row: idx,
        expected: w,
        actual: row.v().len(),
      });
    }
    if row.a().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::AFull10,
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

    let Self { rgba, rgba_u16, .. } = self;
    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // Direct RGBA dispatch — no Strategy A combine path in tranche 8b‑1a
    // (Yuva444p10 has no `with_rgb` builder yet, so there's no shared
    // RGB kernel result to expand). Each attached output buffer gets
    // its own dedicated kernel call.
    if let Some(buf) = rgba.as_deref_mut() {
      let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      yuva444p10_to_rgba_row(
        row.y(),
        row.u(),
        row.v(),
        row.a(),
        rgba_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
    }
    if let Some(buf) = rgba_u16.as_deref_mut() {
      let rgba_u16_row = rgba_u16_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      yuva444p10_to_rgba_u16_row(
        row.y(),
        row.u(),
        row.v(),
        row.a(),
        rgba_u16_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
    }

    Ok(())
  }
}
