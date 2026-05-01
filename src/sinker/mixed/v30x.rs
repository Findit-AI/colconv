//! Sinker impl for the packed V30X source format — Ship 12a (Tier 5
//! 10-bit packed YUV 4:4:4). Full output coverage: u8 + native-depth
//! u16 RGB / RGBA + u8 luma + u8 HSV.
//!
//! V30X is a **sibling of V410** with the opposite padding position.
//! V410 packs `(V << 20) | (Y << 10) | U` (2-bit pad at the top),
//! while V30X packs `(V << 22) | (Y << 12) | (U << 2)` (2-bit pad at
//! the bottom). Both use one `u32` word per pixel with 10-bit channels
//! and no chroma subsampling (4:4:4). The packed slice type is `&[u32]`
//! in both formats.
//!
//! The **numerical contract is identical to V410**: `BITS = 10`, the
//! u16 alpha constant is `0x3FF` (10-bit max), and `expand_rgb_u16_to_rgba_u16_row::<10>`
//! is used for the Strategy A u16 fan-out. Dispatcher functions are the
//! `v30x_to_*` family exported from `crate::row::dispatch::v30x`.
//!
//! Outputs map to the sink's standard channels:
//! - `with_rgb` / `with_rgba` — packed YUV → RGB Q15 pipeline at
//!   `BITS = 10`, downshifted to u8; RGBA alpha is forced to `0xFF`
//!   (V30X has no alpha channel).
//! - `with_rgb_u16` / `with_rgba_u16` — same pipeline at native
//!   10-bit depth, low-bit-packed in `u16` (`[0, 1023]`); RGBA alpha
//!   is `0x3FF` (10-bit max).
//! - `with_luma` — extracts the 10-bit Y values from each V30X word
//!   and downshifts `>> 2` to u8.
//! - `with_hsv` — stages an internal RGB scratch (or the user's RGB
//!   buffer if attached) and runs the existing `rgb_to_hsv_row`
//!   kernel on the staged u8 RGB.
//!
//! **`with_luma_u16` is intentionally NOT supported** for V30X —
//! deferred until a real consumer surfaces (Spec § 11). Use
//! `with_luma` (u8 downshifted `>> 2`) if native-depth luma is
//! needed in the interim.
//!
//! When both u8 RGB and u8 RGBA outputs are requested, the RGBA plane
//! is derived from the just-computed u8 RGB row via
//! [`expand_rgb_to_rgba_row`] (Strategy A) instead of running a
//! second YUV→RGB kernel. The same Strategy A applies on the u16
//! path via [`expand_rgb_u16_to_rgba_u16_row::<10>`]. When only the
//! RGBA variant is wanted, the dedicated `_to_rgba_row` /
//! `_to_rgba_u16_row` kernel writes the output buffer directly
//! without staging RGB.

use super::{
  MixedSinker, MixedSinkerError, RowSlice, check_dimensions_match, rgb_row_buf_or_scratch,
  rgba_plane_row_slice, rgba_u16_plane_row_slice,
};
use crate::{
  PixelSink,
  row::{
    expand_rgb_to_rgba_row, expand_rgb_u16_to_rgba_u16_row, rgb_to_hsv_row, v30x_to_luma_row,
    v30x_to_rgb_row, v30x_to_rgb_u16_row, v30x_to_rgba_row, v30x_to_rgba_u16_row,
  },
  yuv::{V30X, V30XRow, V30XSink},
};

impl<'a> MixedSinker<'a, V30X> {
  /// Attaches a packed **8-bit** RGBA output buffer. Alpha is filled
  /// with constant `0xFF` (V30X has no alpha channel).
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

  /// Attaches a packed **`u16`** RGB output buffer. 10-bit
  /// low-bit-packed (`[0, 1023]`); length is measured in `u16`
  /// **elements** (`width × height × 3`).
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

  /// Attaches a packed **`u16`** RGBA output buffer. 10-bit
  /// low-bit-packed (`[0, 1023]`); alpha element is `0x3FF` (10-bit
  /// max). Length is measured in `u16` **elements**
  /// (`width × height × 4`).
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

impl V30XSink for MixedSinker<'_, V30X> {}

impl PixelSink for MixedSinker<'_, V30X> {
  type Input<'r> = V30XRow<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    Ok(())
  }

  fn process(&mut self, row: V30XRow<'_>) -> Result<(), Self::Error> {
    const BITS: u32 = 10;
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    // V30X row = `width` u32 elements (one pixel per word).
    let packed_expected = w;
    if row.packed().len() != packed_expected {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::V30XPacked,
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
    let packed = row.packed();

    // Luma u8 — extract 8-bit Y bytes from the V30X plane via the
    // dedicated kernel (downshifts 10-bit Y >> 2 to u8).
    if let Some(buf) = luma.as_deref_mut() {
      v30x_to_luma_row(
        packed,
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    // ===== u16 RGB / RGBA path (Strategy A) =====
    let want_rgb_u16 = rgb_u16.is_some();
    let want_rgba_u16 = rgba_u16.is_some();

    if want_rgba_u16 && !want_rgb_u16 {
      // Standalone u16 RGBA fast path — write directly into the
      // caller's buffer; no staging.
      let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row =
        rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
      v30x_to_rgba_u16_row(
        packed,
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
      v30x_to_rgb_u16_row(
        packed,
        rgb_u16_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
      if want_rgba_u16 {
        // Strategy A u16 fan-out — derive RGBA from the just-computed
        // RGB row instead of running a second YUV→RGB kernel.
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
    let need_u8_rgb_kernel = want_rgb || want_hsv;

    // Standalone u8 RGBA fast path — no RGB / HSV requested. Run the
    // dedicated RGBA kernel directly into the output buffer; avoids
    // both the scratch allocation and the RGB→RGBA expand pass.
    if want_rgba && !need_u8_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      v30x_to_rgba_row(
        packed,
        rgba_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
      return Ok(());
    }

    if !need_u8_rgb_kernel {
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
    v30x_to_rgb_row(packed, rgb_row, w, row.matrix(), row.full_range(), use_simd);

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
    // row instead of running a second YUV→RGB kernel.
    if let Some(buf) = rgba.as_deref_mut() {
      let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
    }

    Ok(())
  }
}
