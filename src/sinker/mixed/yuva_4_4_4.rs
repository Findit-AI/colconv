//! Sinker impls for source-side YUVA 4:4:4 formats
//! ([`Yuva444p10`](crate::yuv::Yuva444p10) in this PR; other Yuva444p
//! variants land in later Ship 8b tranches).
//!
//! Tranche 8b‑1a wires:
//! - **RGBA output paths (u8 + native-depth u16)** — alpha is sourced
//!   from the source alpha plane via the new
//!   `yuv_444p_n_to_rgba*_with_alpha_src_row::<10>` scalar kernels.
//!   SIMD lands in 8b‑1b (u8) / 8b‑1c (u16).
//! - **RGB / RGB-u16 / luma / HSV alpha-drop paths** — these reuse the
//!   existing `Yuv444p10` row dispatchers verbatim. The alpha plane is
//!   simply ignored; output bytes/elements match what `Yuv444p10` would
//!   produce given the same Y/U/V data. Without these paths
//!   `MixedSinker::with_rgb` / `with_luma` / `with_hsv` (declared on
//!   the generic impl) would silently accept a buffer and never write
//!   it — the Strategy A combine routes through the same patterns the
//!   non-alpha 4:4:4 sinker uses.

use super::{
  MixedSinker, MixedSinkerError, RowSlice, check_dimensions_match, rgb_row_buf_or_scratch,
  rgba_plane_row_slice, rgba_u16_plane_row_slice,
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
    const BITS: u32 = 10;
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

    // ---- luma (alpha-drop; same as Yuv444p10) ----------------------
    if let Some(luma) = luma.as_deref_mut() {
      let dst = &mut luma[one_plane_start..one_plane_end];
      for (d, &s) in dst.iter_mut().zip(row.y().iter()) {
        *d = (s >> (BITS - 8)) as u8;
      }
    }

    // ---- u16 RGB / RGBA path ---------------------------------------
    // RGBA-only routes through the alpha-source-aware kernel; rgb_u16
    // (alpha-drop) reuses the Yuv444p10 dispatcher verbatim. Both
    // attached → run RGB kernel into the caller buffer + RGBA kernel
    // separately (no Strategy A expand for the YUVA path because the
    // RGBA buffer has source-derived alpha that the expand helper
    // doesn't know how to produce).
    if let Some(buf) = rgb_u16.as_deref_mut() {
      let rgb_plane_end =
        one_plane_end
          .checked_mul(3)
          .ok_or(MixedSinkerError::GeometryOverflow {
            width: w,
            height: h,
            channels: 3,
          })?;
      let rgb_plane_start = one_plane_start * 3;
      let rgb_u16_row = &mut buf[rgb_plane_start..rgb_plane_end];
      yuv444p10_to_rgb_u16_row(
        row.y(),
        row.u(),
        row.v(),
        rgb_u16_row,
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

    // ---- u8 RGB / RGBA / HSV path ----------------------------------
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();
    let need_rgb_kernel = want_rgb || want_hsv;

    // Direct-RGBA-only fast path mirrors Yuv444p10 but routes through
    // the alpha-source-aware dispatcher.
    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
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
      return Ok(());
    }

    if !need_rgb_kernel {
      return Ok(());
    }

    // RGB kernel (caller buffer or scratch for HSV-only). Alpha-drop:
    // reuses the Yuv444p10 dispatcher verbatim — Y/U/V are the same
    // shape, alpha is ignored.
    let rgb_row = rgb_row_buf_or_scratch(
      rgb.as_deref_mut(),
      rgb_scratch,
      one_plane_start,
      one_plane_end,
      w,
      h,
    )?;
    yuv444p10_to_rgb_row(
      row.y(),
      row.u(),
      row.v(),
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

    // Both rgb and rgba attached: run the RGBA kernel for the
    // alpha-aware buffer (cannot reuse rgb_row's RGB output because
    // alpha must come from the source plane; expand-to-rgba would
    // splat 0xFF). Cheaper than a third combine kernel; keeps the YUV
    // math exactly twice in the both-buffers case (acceptable for the
    // alpha-source path — Strategy B forks per buffer when alpha is
    // present, by design).
    if want_rgba {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
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

    Ok(())
  }
}
