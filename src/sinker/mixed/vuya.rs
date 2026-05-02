//! Sinker impl for the packed VUYA source format — Ship 12c (Tier 5
//! 8-bit packed YUV 4:4:4 with real source alpha).
//!
//! VUYA (FFmpeg `AV_PIX_FMT_VUYA`) packs **four u8 bytes per pixel**
//! (`[V, U, Y, A]`). The A byte is **real source alpha** — not padding.
//! The packed slice type is `&[u8]`, with `4 × width` byte elements per
//! row. There is no chroma subsampling — every pixel carries its own
//! independent V / U / Y triplet (4:4:4).
//!
//! Outputs map to the sink's standard channels:
//! - `with_rgb` — packed YUV → RGB 8-bit pipeline; alpha discarded.
//! - `with_rgba` — packed YUV → RGBA 8-bit pipeline; **source α byte
//!   is passed through** verbatim from byte 3 of each pixel (not
//!   substituted with `0xFF`).
//! - `with_luma` — extracts the Y byte at offset 2 of each pixel
//!   directly (no YUV→RGB pipeline).
//! - `with_hsv` — stages u8 RGB into the user's RGB buffer (if
//!   attached) or a scratch buffer, then runs `rgb_to_hsv_row`.
//!
//! VUYA is an 8-bit source. There are no u16 output variants.
//!
//! ## Alpha semantics (`§ 7.2` / `§ 7.3` rules)
//!
//! - **Standalone RGBA** (`with_rgba` attached, no `with_rgb`, no
//!   `with_hsv`): `vuya_to_rgba_row` runs directly — source α passes
//!   through via the kernel.
//! - **RGB + RGBA** (both attached, with or without HSV): each output
//!   runs its own independent kernel call reading from the same packed
//!   input. `with_rgb` calls `vuya_to_rgb_row` (α discarded);
//!   `with_rgba` calls `vuya_to_rgba_row` directly (source α preserved,
//!   per spec § 7.2). Strategy A fan-out (`expand_rgb_to_rgba_row`) is
//!   **never** used for VUYA — that path is reserved for VUYX where
//!   α = `0xFF` is the correct semantic.

use super::{
  MixedSinker, MixedSinkerError, RowSlice, check_dimensions_match, rgb_row_buf_or_scratch,
  rgba_plane_row_slice,
};
use crate::{
  PixelSink,
  row::{rgb_to_hsv_row, vuya_to_luma_row, vuya_to_rgb_row, vuya_to_rgba_row},
  yuv::{Vuya, VuyaRow, VuyaSink},
};

impl<'a> MixedSinker<'a, Vuya> {
  /// Attaches a packed **8-bit** RGBA output buffer. When VUYA is the
  /// source, the per-pixel alpha byte is **sourced from the A byte of
  /// each pixel quadruple** — not forced to `0xFF`.
  ///
  /// Returns `Err(RgbaBufferTooShort)` if
  /// `buf.len() < width × height × 4`, or `Err(GeometryOverflow)` on
  /// 32‑bit targets when the product overflows.
  ///
  /// ## Strategy note
  ///
  /// Source-α pass-through is guaranteed in **all** paths (standalone or
  /// combined with `with_rgb` / `with_hsv`). When combined, `with_rgba`
  /// runs its own `vuya_to_rgba_row` kernel call directly from the packed
  /// source — it is never derived from the RGB output (spec § 7.2).
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

impl VuyaSink for MixedSinker<'_, Vuya> {}

impl PixelSink for MixedSinker<'_, Vuya> {
  type Input<'r> = VuyaRow<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    Ok(())
  }

  fn process(&mut self, row: VuyaRow<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    // VUYA row = `width × 4` bytes (one quadruple per pixel).
    let packed_expected = w.checked_mul(4).ok_or(MixedSinkerError::GeometryOverflow {
      width: w,
      height: h,
      channels: 4,
    })?;
    if row.packed().len() != packed_expected {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::VuyaPacked,
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

    // Luma — extract Y byte (offset 2 in each VUYA quadruple) directly.
    if let Some(buf) = luma.as_deref_mut() {
      vuya_to_luma_row(
        packed,
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    // ===== u8 RGB / RGBA / HSV path =====
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();
    let need_rgb_kernel = want_rgb || want_hsv;

    // RGB kernel — write into the user's RGB buffer (if attached) or the
    // internal scratch buffer. Required when with_rgb or with_hsv is set.
    if need_rgb_kernel {
      let rgb_row = rgb_row_buf_or_scratch(
        rgb.as_deref_mut(),
        rgb_scratch,
        one_plane_start,
        one_plane_end,
        w,
        h,
      )?;
      vuya_to_rgb_row(packed, rgb_row, w, row.matrix(), row.full_range(), use_simd);

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

    // RGBA direct path — spec § 7.2: always run vuya_to_rgba_row directly
    // from the packed source, preserving source α verbatim. This applies
    // whether or not with_rgb / with_hsv are also attached. Strategy A
    // fan-out (expand_rgb_to_rgba_row, α=0xFF) is NEVER used for VUYA.
    if want_rgba {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      vuya_to_rgba_row(
        packed,
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
