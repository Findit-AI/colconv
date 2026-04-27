//! Sinker impls for source-side YUVA 4:2:0 formats — Yuva420p (8-bit),
//! Yuva420p9, Yuva420p10, Yuva420p16.
//!
//! Tranche 8b‑2a wires:
//! - **RGBA output paths (u8 + native-depth u16)** — alpha is sourced
//!   from the source alpha plane via the new
//!   `yuv_420*_to_rgba*_with_alpha_src_row` scalar kernels. SIMD lands
//!   in 8b‑2b (u8) / 8b‑2c (u16). The 8-bit `Yuva420p` has no native
//!   `u16` RGBA output.
//! - **RGB / RGB-u16 / luma / HSV alpha-drop paths** — these reuse the
//!   existing non-alpha 4:2:0 row dispatchers verbatim. The alpha plane
//!   is simply ignored; output bytes/elements match what the
//!   corresponding `Yuv420p*` source would produce given the same
//!   Y/U/V data. Without these paths `MixedSinker::with_rgb` /
//!   `with_luma` / `with_hsv` (declared on the generic impl) would
//!   silently accept a buffer and never write it (Codex PR #32 review
//!   fix #1 — applied upfront here).

use super::{
  MixedSinker, MixedSinkerError, RowSlice, check_dimensions_match, rgb_row_buf_or_scratch,
  rgba_plane_row_slice, rgba_u16_plane_row_slice,
};
use crate::{PixelSink, row::*, yuv::*};

// ---- Yuva420p impl (8-bit) ---------------------------------------------

impl<'a> MixedSinker<'a, Yuva420p> {
  /// Attaches a packed **8‑bit** RGBA output buffer. The 8‑bit YUVA
  /// source is converted to 8‑bit RGBA via the same Q15 i32 8‑bit
  /// kernel that backs [`MixedSinker<Yuv420p>::with_rgba`]; the
  /// per-pixel alpha byte is **sourced from the alpha plane** — not
  /// constant `0xFF`.
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

impl Yuva420pSink for MixedSinker<'_, Yuva420p> {}

impl PixelSink for MixedSinker<'_, Yuva420p> {
  type Input<'r> = Yuva420pRow<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::OddWidth { width: self.width });
    }
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: Yuva420pRow<'_>) -> Result<(), Self::Error> {
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

    // ---- u8 RGB / RGBA / HSV path ----------------------------------
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();
    let need_rgb_kernel = want_rgb || want_hsv;

    // Direct-RGBA-only fast path routes through the alpha-source-aware
    // dispatcher.
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

    // RGB kernel — alpha-drop reuses the Yuv420p dispatcher verbatim.
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

    // Both rgb and rgba attached: run the RGBA kernel for the
    // alpha-aware buffer (cannot reuse rgb_row's RGB output because
    // alpha must come from the source plane; expand-to-rgba would
    // splat 0xFF). Strategy B forks per buffer when alpha is present.
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

// ---- Yuva420p9 impl ---------------------------------------------------

impl<'a> MixedSinker<'a, Yuva420p9> {
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
  /// Output is identical to [`MixedSinker<Yuv420p9>::with_rgb_u16`].
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

impl Yuva420p9Sink for MixedSinker<'_, Yuva420p9> {}

impl PixelSink for MixedSinker<'_, Yuva420p9> {
  type Input<'r> = Yuva420p9Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::OddWidth { width: self.width });
    }
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: Yuva420p9Row<'_>) -> Result<(), Self::Error> {
    yuva420p_high_bit_process::<9, _, _, _>(
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

// ---- Yuva420p10 impl --------------------------------------------------

impl<'a> MixedSinker<'a, Yuva420p10> {
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

impl Yuva420p10Sink for MixedSinker<'_, Yuva420p10> {}

impl PixelSink for MixedSinker<'_, Yuva420p10> {
  type Input<'r> = Yuva420p10Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::OddWidth { width: self.width });
    }
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: Yuva420p10Row<'_>) -> Result<(), Self::Error> {
    yuva420p_high_bit_process::<10, _, _, _>(
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

// ---- Yuva420p16 impl --------------------------------------------------

impl<'a> MixedSinker<'a, Yuva420p16> {
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

impl Yuva420p16Sink for MixedSinker<'_, Yuva420p16> {}

impl PixelSink for MixedSinker<'_, Yuva420p16> {
  type Input<'r> = Yuva420p16Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::OddWidth { width: self.width });
    }
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: Yuva420p16Row<'_>) -> Result<(), Self::Error> {
    yuva420p_high_bit_process::<16, _, _, _>(
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

// ---- Shared high-bit YUVA 4:2:0 process body --------------------------
//
// The 9 / 10 / 16-bit YUVA 4:2:0 sinker `process` bodies are
// structurally identical — only the depth-named row primitives, the
// `RowSlice` variants used in error reports, and the depth-conversion
// shift (`BITS - 8`) for luma differ. Factor the shared body into a
// generic helper to avoid 3× of the same ~70 lines.
//
// Strategy A combine for the alpha-drop paths (rgb_u16 alpha-drop +
// rgba_u16 source alpha): runs the rgb_u16 kernel into the caller
// buffer + the rgba_u16 kernel separately. Cannot reuse rgb_u16's
// output because alpha comes from the source plane; the expand
// helper would splat opaque alpha.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
#[cfg_attr(not(tarpaulin), inline(always))]
fn yuva420p_high_bit_process<
  const BITS: u32,
  RgbRowFn: Fn(&[u16], &[u16], &[u16], &mut [u8], usize, crate::ColorMatrix, bool, bool),
  RgbU16RowFn: Fn(&[u16], &[u16], &[u16], &mut [u16], usize, crate::ColorMatrix, bool, bool),
  RgbaRowFn: Fn(&[u16], &[u16], &[u16], &[u16], &mut [u8], usize, crate::ColorMatrix, bool, bool),
>(
  sinker: &mut MixedSinker<'_, impl crate::SourceFormat>,
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
  // rgb_u16 (alpha-drop) reuses the non-alpha dispatcher. rgba_u16
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

  // Both rgb and rgba attached: run the RGBA kernel for the
  // alpha-aware buffer.
  if want_rgba {
    let rgba_buf = rgba.as_deref_mut().unwrap();
    let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
    rgba_dispatch(
      y_row, u_half_row, v_half_row, a_row, rgba_row, w, matrix, full_range, use_simd,
    );
  }

  Ok(())
}
