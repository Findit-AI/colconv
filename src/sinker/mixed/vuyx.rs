//! Sinker impl for the packed VUYX source format — Ship 12c (Tier 5
//! 8-bit packed YUV 4:4:4 with padding α byte).
//!
//! VUYX (FFmpeg `AV_PIX_FMT_VUYX`) packs **four u8 bytes per pixel**
//! (`[V, U, Y, X]`). The X byte is **padding** — not real source alpha.
//! RGBA outputs always force α to `0xFF`; the padding byte is ignored.
//! The packed slice type is `&[u8]`, with `4 × width` byte elements per
//! row. There is no chroma subsampling — every pixel carries its own
//! independent V / U / Y triplet (4:4:4).
//!
//! Outputs map to the sink's standard channels:
//! - `with_rgb` — packed YUV → RGB 8-bit pipeline; padding discarded.
//! - `with_rgba` — packed YUV → RGBA 8-bit pipeline; **α is forced to
//!   `0xFF`** (the X byte is padding, never real alpha).
//! - `with_luma` — extracts the Y byte at offset 2 of each pixel
//!   directly (no YUV→RGB pipeline).
//! - `with_hsv` — stages u8 RGB into the user's RGB buffer (if
//!   attached) or a scratch buffer, then runs `rgb_to_hsv_row`.
//!
//! VUYX is an 8-bit source. There are no u16 output variants.
//!
//! ## Alpha semantics (`§ 8.3` / `§ 8.4` rules — Strategy A)
//!
//! Because VUYX's α is always `0xFF` in every code path (padding byte
//! is never real alpha), the RGB + RGBA combo can use **Strategy A**
//! (spec § 8.4): derive RGBA from the just-computed RGB row via
//! `expand_rgb_to_rgba_row` instead of running a second YUV→RGB kernel.
//! This produces bit-identical output to calling `vuyx_to_rgba_row`
//! directly — both paths always produce α=`0xFF`.
//!
//! - **Standalone RGBA** (`with_rgba` attached, no `with_rgb`, no
//!   `with_hsv`): `vuyx_to_rgba_row` runs directly — α forced to
//!   `0xFF` via the kernel.
//! - **RGB + RGBA** (both attached, with or without HSV): `with_rgb`
//!   calls `vuyx_to_rgb_row`; `with_rgba` is derived via Strategy A
//!   `expand_rgb_to_rgba_row` (α=`0xFF`). No second kernel call.

use super::{
  MixedSinker, MixedSinkerError, RowSlice, check_dimensions_match, rgb_row_buf_or_scratch,
  rgba_plane_row_slice,
};
use crate::{
  PixelSink,
  row::{
    expand_rgb_to_rgba_row, rgb_to_hsv_row, vuyx_to_luma_row, vuyx_to_rgb_row, vuyx_to_rgba_row,
  },
  yuv::{Vuyx, VuyxRow, VuyxSink},
};

impl<'a> MixedSinker<'a, Vuyx> {
  /// Attaches a packed **8-bit** RGBA output buffer. When VUYX is the
  /// source, the per-pixel alpha byte is always forced to `0xFF` —
  /// the X (padding) byte in the source is never read as alpha.
  ///
  /// Returns `Err(RgbaBufferTooShort)` if
  /// `buf.len() < width × height × 4`, or `Err(GeometryOverflow)` on
  /// 32‑bit targets when the product overflows.
  ///
  /// ## Strategy note
  ///
  /// α=`0xFF` is guaranteed in **all** paths (standalone or combined
  /// with `with_rgb` / `with_hsv`). When combined with `with_rgb`,
  /// RGBA is derived via Strategy A fan-out (`expand_rgb_to_rgba_row`)
  /// instead of a second YUV→RGB kernel call — both produce α=`0xFF`,
  /// so the outputs are semantically identical (spec § 8.4).
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

impl VuyxSink for MixedSinker<'_, Vuyx> {}

impl PixelSink for MixedSinker<'_, Vuyx> {
  type Input<'r> = VuyxRow<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    Ok(())
  }

  fn process(&mut self, row: VuyxRow<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    // VUYX row = `width × 4` bytes (one quadruple per pixel).
    let packed_expected = w.checked_mul(4).ok_or(MixedSinkerError::GeometryOverflow {
      width: w,
      height: h,
      channels: 4,
    })?;
    if row.packed().len() != packed_expected {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::VuyxPacked,
        row: idx,
        expected: packed_expected,
        actual: row.packed().len(),
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
    let packed = row.packed();

    // Luma — extract Y byte (offset 2 in each VUYX quadruple) directly.
    // `vuyx_to_luma_row` is a re-export of `vuya_to_luma_row` — the
    // byte stream is identical (Y at offset 2 regardless of α semantics).
    if let Some(buf) = luma.as_deref_mut() {
      vuyx_to_luma_row(
        packed,
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    // ===== u8 RGB / RGBA / HSV path (Strategy A for VUYX) =====
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();
    let need_rgb_kernel = want_rgb || want_hsv;

    // Standalone RGBA fast path — no RGB / HSV requested. Run the
    // dedicated RGBA kernel directly into the output buffer; avoids
    // both the scratch allocation and the RGB→RGBA expand pass.
    // α is forced to `0xFF` by `vuyx_to_rgba_row` (ALPHA_SRC = false).
    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      vuyx_to_rgba_row(
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

    // RGB kernel — write into the user's RGB buffer (if attached) or the
    // internal scratch buffer. Required when with_rgb or with_hsv is set.
    let rgb_row = rgb_row_buf_or_scratch(
      rgb.as_deref_mut(),
      rgb_scratch,
      one_plane_start,
      one_plane_end,
      w,
      h,
    )?;
    // `vuyx_to_rgb_row` is a re-export of `vuya_to_rgb_row` — the padding
    // byte is irrelevant when there is no α channel in the output.
    vuyx_to_rgb_row(packed, rgb_row, w, row.matrix(), row.full_range(), use_simd);

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

    // Strategy A u8 fan-out — derive RGBA from the just-computed RGB
    // row instead of running a second YUV→RGB kernel. For VUYX,
    // α=`0xFF` is semantically correct in both paths (padding byte
    // is never real alpha), so Strategy A applies (spec § 8.4).
    if let Some(buf) = rgba.as_deref_mut() {
      let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
    }

    Ok(())
  }
}
