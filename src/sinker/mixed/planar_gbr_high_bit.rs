//! Sinker impls for high-bit-depth planar GBR source formats (Tier 10b).
//!
//! Covers all nine high-bit formats:
//! - `Gbrp9` / `Gbrp10` / `Gbrp12` / `Gbrp14` / `Gbrp16` — three planes
//!   (G, B, R), no alpha. `AV_PIX_FMT_GBRP{9,10,12,14,16}LE`.
//! - `Gbrap10` / `Gbrap12` / `Gbrap14` / `Gbrap16` — four planes
//!   (G, B, R, A), real per-pixel α. `AV_PIX_FMT_GBRAP{10,12,14,16}LE`.
//!   (FFmpeg has no 9-bit Gbrap variant.)
//!
//! # Output paths
//!
//! - `with_rgb` — interleave G/B/R → packed `R, G, B` bytes (downshift by
//!   `BITS - 8`).
//! - `with_rgb_u16` — interleave G/B/R → packed `R, G, B` u16 elements at
//!   native depth (no shift; values in `[0, (1 << BITS) - 1]`).
//! - `with_rgba` — for `GbrpN`: standalone `gbr_to_rgba_opaque_high_bit_row`
//!   (α = `0xFF`); combo with `with_rgb` uses Strategy A (expand). For
//!   `GbrapN`: standalone `gbra_to_rgba_high_bit_row` (real α downshifted by
//!   `BITS - 8`); combo with `with_rgb` uses Strategy A+ (expand + α-overwrite
//!   from source plane).
//! - `with_rgba_u16` — same as above but u16 output; opaque α =
//!   `(1 << BITS) - 1` for `GbrpN`; real α at native depth for `GbrapN`.
//! - `with_luma` — derived from staged RGB (u8) via `rgb_to_luma_row`.
//! - `with_luma_u16` — derived directly from native-precision G/B/R planes via
//!   `gbr_to_luma_u16_high_bit_row`. Uses Q15 coefficients and i64 intermediates
//!   (required for BITS=16). Produces full native-precision output — no 256-level
//!   banding from the old u8-intermediate approach.
//! - `with_hsv` — derived from staged RGB via `rgb_to_hsv_row`.
//!
//! # Strategy A+ (Gbrap combo path)
//!
//! When both `with_rgb` and `with_rgba` are attached to a `GbrapN` sinker:
//! 1. Stage G/B/R → RGB row.
//! 2. Expand RGB → RGBA (α = `0xFF` stub).
//! 3. Overwrite α bytes from the source A plane via
//!    `alpha_extract::copy_alpha_plane_u16_to_u8::<BITS>`.
//!
//! This avoids two calls to the full 4-channel kernel and matches the shape
//! of the 8-bit `Gbrap` sinker.
//!
//! # Fused area-resample (`with_resampler`)
//!
//! The `GbrpN` sinkers are generic over `R: Resampler`. On a non-identity
//! plan each scatters its native-depth G/B/R planes into a source-width
//! packed `u16` RGB row (`gbr_to_rgb_u16_high_bit_row`) and feeds the shared
//! high-bit packed-RGB resample tail
//! ([`packed_rgb_u16_resample_emit`](super::packed_rgb_u16_resample_emit) and
//! siblings) — the `u16` analog of the 8-bit `Gbrp` routing and the same tail
//! the `Rgb48` / `Bgr48` sources take, parameterized by the source depth
//! `BITS` so the u8 narrowing (`>> (BITS - 8)`) and the opaque `rgba_u16`
//! alpha (`(1 << BITS) - 1`) track the native depth rather than a hard-coded
//! 16. Binning runs at native depth: `rgb_u16` / `rgba_u16` copy the binned
//! row, while `rgb` / `rgba` / `luma` / `hsv` derive from a single narrowing
//! of it.
//!
//! `luma_u16` is computed at full native precision from the binned native
//! RGB — via `rgb_to_luma_u16_native_row`, the packed / runtime-`bits` twin
//! of the direct path's `gbr_to_luma_u16_high_bit_row` — so a resampled
//! `GbrpN` `luma_u16` is byte-identical to the direct path: full parity on
//! every output. (`Rgb48` / `Bgr48` keep the tail's narrowed `luma_u16`,
//! which matches their own narrowed direct path.)
//!
//! The `GbrapN` sinkers are likewise generic over `R: Resampler`. On a
//! non-identity plan each de-interleaves its native-depth G/B/R/A planes
//! into a canonical host-native `R, G, B, A` u16 row
//! (`gbra_to_rgba_u16_high_bit_row`) and feeds the **alpha-aware**
//! 4-channel high-bit packed RGBA tail
//! ([`packed_rgba_u16_resample`](super::packed_rgba_u16_resample)) at
//! `BITS` — the `u16` analog of the 8-bit `Gbrap` routing and the same
//! tail `Rgba64` / `Bgra64` take. Resampled alpha is a real native area
//! mean (not expand-stubbed), and under
//! [`AlphaMode::Premultiplied`](super::AlphaMode::Premultiplied) the color
//! is binned premultiplied and un-premultiplied per output row. The
//! per-format default is straight alpha. A straight rgb-only sink (alpha
//! dropped) keeps the 3-channel u16 RGB path with no regression.

use super::{
  GeometryOverflow, InsufficientBuffer, MixedSinker, MixedSinkerError, RowIndexOutOfRange,
  RowShapeMismatch, RowSlice, check_dimensions_match, check_frozen_alpha_mode,
  packed_rgb_u16_resample, packed_rgb_u16_resample_filter, packed_rgba_u16_filter_resample,
  packed_rgba_u16_resample, rgb_row_buf_or_scratch, rgba_plane_row_slice, rgba_u16_plane_row_slice,
};
use crate::{
  PixelSink,
  row::{
    alpha_extract, expand_rgb_to_rgba_row, expand_rgb_u16_to_rgba_u16_row,
    gbr_to_luma_u16_high_bit_row, gbr_to_rgb_high_bit_row, gbr_to_rgb_u16_high_bit_row,
    gbr_to_rgba_opaque_high_bit_row, gbr_to_rgba_opaque_u16_high_bit_row,
    gbra_to_rgba_high_bit_row, gbra_to_rgba_u16_high_bit_row, rgb_to_hsv_row, rgb_to_luma_row,
  },
};

// ---- Gbrp accessors (via format-specific impl blocks) -------------------
//
// Each format gets its own impl block. The pattern is identical for all 5
// depths — only the const BITS value and the concrete Row/Sink types differ.
// We use a local macro to avoid 400 lines of repetition.
//
// The macro generates:
//   impl<'a> MixedSinker<'a, $marker> { with_rgb_u16 / set_rgb_u16 /
//                                        with_rgba / set_rgba /
//                                        with_rgba_u16 / set_rgba_u16 /
//                                        with_luma_u16 / set_luma_u16 }
//   impl $sink for MixedSinker<'_, $marker> {}
//   impl PixelSink for MixedSinker<'_, $marker> { ... }

macro_rules! impl_gbrp_high_bit {
  ($marker:ident, $sink:ident, $row:ident, $bits:literal) => {
    impl<'a, R, const BE: bool> MixedSinker<'a, crate::source::$marker<BE>, R> {
      /// Attaches a packed **`u16`** RGB output buffer. Samples are in
      /// `[0, (1 << BITS) - 1]` (native depth, no depth conversion).
      /// Length is measured in `u16` **elements** (`width x height x 3`).
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub fn with_rgb_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
        self.set_rgb_u16(buf)?;
        Ok(self)
      }
      /// In-place variant of [`with_rgb_u16`](Self::with_rgb_u16).
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub fn set_rgb_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
        let expected = self.frame_elems(3)?;
        if buf.len() < expected {
          return Err(MixedSinkerError::InsufficientRgbU16Buffer(
            InsufficientBuffer::new(expected, buf.len()),
          ));
        }
        self.rgb_u16 = Some(buf);
        Ok(self)
      }

      /// Attaches a packed **8-bit** RGBA output buffer. Alpha is opaque
      /// (`0xFF`) — the GBR format has no alpha plane. Length in bytes
      /// (`width x height x 4`).
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub fn with_rgba(mut self, buf: &'a mut [u8]) -> Result<Self, MixedSinkerError> {
        self.set_rgba(buf)?;
        Ok(self)
      }
      /// In-place variant of [`with_rgba`](Self::with_rgba).
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub fn set_rgba(&mut self, buf: &'a mut [u8]) -> Result<&mut Self, MixedSinkerError> {
        let expected = self.frame_elems(4)?;
        if buf.len() < expected {
          return Err(MixedSinkerError::InsufficientRgbaBuffer(
            InsufficientBuffer::new(expected, buf.len()),
          ));
        }
        self.rgba = Some(buf);
        Ok(self)
      }

      /// Attaches a packed **`u16`** RGBA output buffer. Alpha is opaque
      /// (`(1 << BITS) - 1`). Length in `u16` elements (`width x height x 4`).
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub fn with_rgba_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
        self.set_rgba_u16(buf)?;
        Ok(self)
      }
      /// In-place variant of [`with_rgba_u16`](Self::with_rgba_u16).
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub fn set_rgba_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
        let expected = self.frame_elems(4)?;
        if buf.len() < expected {
          return Err(MixedSinkerError::InsufficientRgbaU16Buffer(
            InsufficientBuffer::new(expected, buf.len()),
          ));
        }
        self.rgba_u16 = Some(buf);
        Ok(self)
      }

      /// Attaches a `u16` luma output buffer. Luma is computed directly from
      /// the native-precision G/B/R planes via Q15 coefficients, avoiding the
      /// 256-level banding that the old u8-intermediate path produced. Values
      /// are in `[0, (1 << BITS) - 1]` (full-range) or
      /// `[16 << (BITS - 8), 235 << (BITS - 8)]` (limited-range). Length in
      /// `u16` elements (`width x height`).
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub fn with_luma_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
        self.set_luma_u16(buf)?;
        Ok(self)
      }
      /// In-place variant of [`with_luma_u16`](Self::with_luma_u16).
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub fn set_luma_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
        let expected = self.frame_pixels()?;
        if buf.len() < expected {
          return Err(MixedSinkerError::InsufficientLumaU16Buffer(
            InsufficientBuffer::new(expected, buf.len()),
          ));
        }
        self.luma_u16 = Some(buf);
        Ok(self)
      }
    }

    impl<R, const BE: bool> crate::source::$sink<BE>
      for MixedSinker<'_, crate::source::$marker<BE>, R>
    {
    }

    impl<R, const BE: bool> PixelSink for MixedSinker<'_, crate::source::$marker<BE>, R> {
      type Input<'r> = crate::source::$row<'r>;
      type Error = MixedSinkerError;

      #[cfg_attr(not(tarpaulin), inline(always))]
      fn kernel_matrix(&self) -> crate::KernelMatrix {
        self.kernel_matrix
      }
      fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
        check_dimensions_match(self.width, self.height, width, height)?;
        if let Some(stream) = self.rgb_stream_u16.as_mut() {
          stream.reset();
        }
        if let Some(stream) = self.rgb_filter_stream_u16.as_mut() {
          stream.reset();
        }
        self.resample_outputs = None;
        Ok(())
      }

      fn process(&mut self, row: crate::source::$row<'_>) -> Result<(), Self::Error> {
        const BITS: u32 = $bits;
        let w = self.width;
        let h = self.height;
        let idx = row.row();
        let use_simd = self.simd;

        if row.g().len() != w {
          return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
            RowSlice::GPlane,
            idx,
            w,
            row.g().len(),
          )));
        }
        if row.b().len() != w {
          return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
            RowSlice::BPlane,
            idx,
            w,
            row.b().len(),
          )));
        }
        if row.r().len() != w {
          return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
            RowSlice::RPlane,
            idx,
            w,
            row.r().len(),
          )));
        }
        if idx >= h {
          return Err(MixedSinkerError::RowIndexOutOfRange(
            RowIndexOutOfRange::new(idx, h),
          ));
        }

        // Non-identity plan: scatter the native-depth G/B/R planes into a
        // source-width packed u16 RGB row and feed the shared high-bit
        // packed-RGB resample tail (parameterized by BITS) — the u16 analog
        // of the `Rgb48` routing. Binning runs at native depth; rgb_u16
        // / rgba_u16 copy the binned row, u8 / hsv derive from its
        // `>> (BITS - 8)` narrowing, and luma_u16 is computed at native
        // precision from the binned RGB (the packed twin of the direct
        // path's `gbr_to_luma_u16_high_bit_row`). The plan's span kind picks
        // the engine — the integer area stream or the signed-coefficient
        // filter stream — both binning the same staged native-u16 row.
        // Freeze the output set and sequence-check before staging so a
        // no-output sink stays a no-op and an out-of-sequence row is rejected
        // without the allocation.
        if let Some(plan) = self.plan.as_ref() {
          let Self {
            rgb,
            rgb_u16,
            rgba,
            rgba_u16,
            luma,
            luma_u16,
            hsv,
            rgb_scratch,
            rgb_scratch_u16,
            rgb_stream_u16,
            rgb_filter_stream_u16,
            resample_outputs,
            ..
          } = self;
          // Transactional resample (RFC #238 tail-collapse): scatter the native-depth
          // G/B/R planes into the source-width packed-u16 RGB scratch and feed the
          // shared 3-channel u16 helper (`SRC_BITS = BITS`, `NATIVE_LUMA16 = true`
          // for native-precision `luma_u16` parity); the span kind picks the engine.
          return match plan.kind() {
            crate::resample::SpanKind::Area => packed_rgb_u16_resample::<BITS, true>(
              rgb_stream_u16,
              resample_outputs,
              rgb,
              rgba,
              luma,
              rgb_u16,
              rgba_u16,
              luma_u16,
              hsv,
              rgb_scratch_u16,
              rgb_scratch,
              w,
              plan,
              idx,
              use_simd,
              row.matrix(),
              row.full_range(),
              |src_u16| {
                gbr_to_rgb_u16_high_bit_row::<BITS, BE>(
                  row.g(),
                  row.b(),
                  row.r(),
                  src_u16,
                  w,
                  use_simd,
                )
              },
            ),
            crate::resample::SpanKind::Filter => packed_rgb_u16_resample_filter::<BITS, true>(
              rgb_filter_stream_u16,
              resample_outputs,
              rgb,
              rgba,
              luma,
              rgb_u16,
              rgba_u16,
              luma_u16,
              hsv,
              rgb_scratch_u16,
              rgb_scratch,
              w,
              plan,
              idx,
              use_simd,
              row.matrix(),
              row.full_range(),
              |src_u16| {
                gbr_to_rgb_u16_high_bit_row::<BITS, BE>(
                  row.g(),
                  row.b(),
                  row.r(),
                  src_u16,
                  w,
                  use_simd,
                )
              },
            ),
          };
        }

        let Self {
          rgb,
          rgb_u16,
          rgba,
          rgba_u16,
          luma,
          luma_u16,
          hsv,
          rgb_scratch,
          ..
        } = self;
        let one_plane_start = idx * w;
        let one_plane_end = one_plane_start + w;
        let g_in = row.g();
        let b_in = row.b();
        let r_in = row.r();

        // ---- u16 RGB / RGBA output (Strategy A) -------------------------
        //
        // Native-depth path: no u8 staging needed. When both rgb_u16 and
        // rgba_u16 are requested, stage into rgb_u16 then expand; when only
        // rgba_u16 is requested, use the opaque direct kernel.
        let want_rgb_u16 = rgb_u16.is_some();
        let want_rgba_u16 = rgba_u16.is_some();

        if want_rgba_u16 && !want_rgb_u16 {
          let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
          let rgba_u16_row =
            rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
          gbr_to_rgba_opaque_u16_high_bit_row::<BITS, BE>(
            g_in,
            b_in,
            r_in,
            rgba_u16_row,
            w,
            use_simd,
          );
        } else if want_rgb_u16 {
          let rgb_u16_buf = rgb_u16.as_deref_mut().unwrap();
          let rgb_plane_end =
            one_plane_end
              .checked_mul(3)
              .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
                w, h, 3,
              )))?;
          let rgb_plane_start = one_plane_start * 3;
          let rgb_u16_row = &mut rgb_u16_buf[rgb_plane_start..rgb_plane_end];
          gbr_to_rgb_u16_high_bit_row::<BITS, BE>(g_in, b_in, r_in, rgb_u16_row, w, use_simd);
          if want_rgba_u16 {
            let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
            let rgba_u16_row =
              rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
            expand_rgb_u16_to_rgba_u16_row::<BITS>(rgb_u16_row, rgba_u16_row, w);
          }
        }

        // ---- native-depth luma output (no RGB staging needed) -----------
        // Compute luma_u16 first — it reads G/B/R planes directly without
        // going through the u8 staging path, so it is independent of whether
        // RGB staging happens below.
        if let Some(luma_u16_buf) = luma_u16.as_deref_mut() {
          gbr_to_luma_u16_high_bit_row::<BITS, BE>(
            g_in,
            b_in,
            r_in,
            &mut luma_u16_buf[one_plane_start..one_plane_end],
            w,
            row.matrix(),
            row.full_range(),
            use_simd,
          );
        }

        // ---- u8 RGB / RGBA / luma / HSV output (Strategy A) -----------
        let want_rgb = rgb.is_some();
        let want_rgba = rgba.is_some();
        let want_luma = luma.is_some();
        let want_hsv = hsv.is_some();
        let need_rgb_staging = want_rgb || want_luma || want_hsv;

        // RGBA-only fast path: use the 4-channel opaque kernel directly.
        if want_rgba && !need_rgb_staging {
          let rgba_buf = rgba.as_deref_mut().unwrap();
          let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
          gbr_to_rgba_opaque_high_bit_row::<BITS, BE>(g_in, b_in, r_in, rgba_row, w, use_simd);
          return Ok(());
        }

        if !need_rgb_staging && !want_rgba {
          return Ok(());
        }

        // Stage RGB once (into user buffer or scratch).
        let rgb_row = rgb_row_buf_or_scratch(
          rgb.as_deref_mut(),
          rgb_scratch,
          one_plane_start,
          one_plane_end,
          w,
          h,
        )?;
        gbr_to_rgb_high_bit_row::<BITS, BE>(g_in, b_in, r_in, rgb_row, w, use_simd);

        if let Some(luma) = luma.as_deref_mut() {
          rgb_to_luma_row(
            rgb_row,
            &mut luma[one_plane_start..one_plane_end],
            w,
            row.matrix(),
            row.full_range(),
            use_simd,
          );
        }

        if let Some(hsv) = hsv.as_mut() {
          let (h, s, v) = hsv.hsv();
          rgb_to_hsv_row(
            rgb_row,
            &mut h[one_plane_start..one_plane_end],
            &mut s[one_plane_start..one_plane_end],
            &mut v[one_plane_start..one_plane_end],
            w,
            use_simd,
          );
        }

        if let Some(buf) = rgba.as_deref_mut() {
          // Strategy A: expand already-computed rgb_row → rgba (opaque).
          let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
          expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
        }

        Ok(())
      }
    }
  };
}

macro_rules! impl_gbrap_high_bit {
  ($marker:ident, $sink:ident, $row:ident, $bits:literal) => {
    impl<'a, R, const BE: bool> MixedSinker<'a, crate::source::$marker<BE>, R> {
      /// Attaches a packed **`u16`** RGB output buffer. Samples are in
      /// `[0, (1 << BITS) - 1]` (native depth, no depth conversion).
      /// Length is measured in `u16` **elements** (`width x height x 3`).
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub fn with_rgb_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
        self.set_rgb_u16(buf)?;
        Ok(self)
      }
      /// In-place variant of [`with_rgb_u16`](Self::with_rgb_u16).
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub fn set_rgb_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
        let expected = self.frame_elems(3)?;
        if buf.len() < expected {
          return Err(MixedSinkerError::InsufficientRgbU16Buffer(
            InsufficientBuffer::new(expected, buf.len()),
          ));
        }
        self.rgb_u16 = Some(buf);
        Ok(self)
      }

      /// Attaches a packed **8-bit** RGBA output buffer. Alpha is sourced
      /// from the source A plane, downshifted by `BITS - 8`.
      /// Length in bytes (`width x height x 4`).
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub fn with_rgba(mut self, buf: &'a mut [u8]) -> Result<Self, MixedSinkerError> {
        self.set_rgba(buf)?;
        Ok(self)
      }
      /// In-place variant of [`with_rgba`](Self::with_rgba).
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub fn set_rgba(&mut self, buf: &'a mut [u8]) -> Result<&mut Self, MixedSinkerError> {
        let expected = self.frame_elems(4)?;
        if buf.len() < expected {
          return Err(MixedSinkerError::InsufficientRgbaBuffer(
            InsufficientBuffer::new(expected, buf.len()),
          ));
        }
        self.rgba = Some(buf);
        Ok(self)
      }

      /// Attaches a packed **`u16`** RGBA output buffer. Alpha is sourced
      /// from the source A plane at native depth (no depth conversion).
      /// Length in `u16` elements (`width x height x 4`).
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub fn with_rgba_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
        self.set_rgba_u16(buf)?;
        Ok(self)
      }
      /// In-place variant of [`with_rgba_u16`](Self::with_rgba_u16).
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub fn set_rgba_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
        let expected = self.frame_elems(4)?;
        if buf.len() < expected {
          return Err(MixedSinkerError::InsufficientRgbaU16Buffer(
            InsufficientBuffer::new(expected, buf.len()),
          ));
        }
        self.rgba_u16 = Some(buf);
        Ok(self)
      }

      /// Attaches a `u16` luma output buffer. Same derivation as `GbrpN` —
      /// computed directly from native-precision G/B/R planes via Q15
      /// coefficients (native-depth, no banding). Length in `u16` elements
      /// (`width x height`).
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub fn with_luma_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
        self.set_luma_u16(buf)?;
        Ok(self)
      }
      /// In-place variant of [`with_luma_u16`](Self::with_luma_u16).
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub fn set_luma_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
        let expected = self.frame_pixels()?;
        if buf.len() < expected {
          return Err(MixedSinkerError::InsufficientLumaU16Buffer(
            InsufficientBuffer::new(expected, buf.len()),
          ));
        }
        self.luma_u16 = Some(buf);
        Ok(self)
      }
    }

    impl<R, const BE: bool> crate::source::$sink<BE>
      for MixedSinker<'_, crate::source::$marker<BE>, R>
    {
    }

    impl<R, const BE: bool> PixelSink for MixedSinker<'_, crate::source::$marker<BE>, R> {
      type Input<'r> = crate::source::$row<'r>;
      type Error = MixedSinkerError;

      #[cfg_attr(not(tarpaulin), inline(always))]
      fn kernel_matrix(&self) -> crate::KernelMatrix {
        self.kernel_matrix
      }
      fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
        check_dimensions_match(self.width, self.height, width, height)?;
        if let Some(stream) = self.rgb_stream_u16.as_mut() {
          stream.reset();
        }
        if let Some(stream) = self.rgba_stream_u16.as_mut() {
          stream.reset();
        }
        if let Some(stream) = self.rgb_filter_stream_u16.as_mut() {
          stream.reset();
        }
        if let Some(stream) = self.rgba_filter_stream_u16.as_mut() {
          stream.reset();
        }
        self.resample_outputs = None;
        self.frozen_alpha_mode = Some(self.alpha_mode);
        Ok(())
      }

      fn process(&mut self, row: crate::source::$row<'_>) -> Result<(), Self::Error> {
        const BITS: u32 = $bits;
        let w = self.width;
        let h = self.height;
        let idx = row.row();
        let use_simd = self.simd;

        if row.g().len() != w {
          return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
            RowSlice::GPlane,
            idx,
            w,
            row.g().len(),
          )));
        }
        if row.b().len() != w {
          return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
            RowSlice::BPlane,
            idx,
            w,
            row.b().len(),
          )));
        }
        if row.r().len() != w {
          return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
            RowSlice::RPlane,
            idx,
            w,
            row.r().len(),
          )));
        }
        if row.a().len() != w {
          return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
            RowSlice::AFull,
            idx,
            w,
            row.a().len(),
          )));
        }
        if idx >= h {
          return Err(MixedSinkerError::RowIndexOutOfRange(
            RowIndexOutOfRange::new(idx, h),
          ));
        }

        // Non-identity plan. Route the alpha-aware 4-channel u16 tail when
        // resampled alpha would be dropped (rgba / rgba_u16 attached) or the
        // color must be alpha-weighted (premultiplied); otherwise the
        // rgb-only straight outputs keep the 3-channel u16 RGB path.
        // `GbrapN` de-interleaves its native-depth G/B/R/A planes into the
        // canonical host-native RGBA row (`gbra_to_rgba_u16_high_bit_row`)
        // the high-bit packed RGBA tail bins at `BITS`, so resampled alpha
        // is a real native area mean and luma derives from the binned RGB.
        // The span kind picks the engine: the integer area stream or the
        // signed-coefficient filter stream (PIL parity, straight alpha only).
        if self.plan.is_some() {
          let alpha_mode = self.alpha_mode;
          let matrix = row.matrix();
          let full_range = row.full_range();
          let g_in = row.g();
          let b_in = row.b();
          let r_in = row.r();
          let a_in = row.a();
          let Self {
            rgb,
            rgb_u16,
            rgba,
            rgba_u16,
            luma,
            luma_u16,
            hsv,
            rgb_scratch,
            rgb_scratch_u16,
            rgba_scratch_u16,
            rgba_color_scratch_u16,
            rgb_stream_u16,
            rgba_stream_u16,
            rgb_filter_stream_u16,
            rgba_filter_stream_u16,
            resample_outputs,
            frozen_alpha_mode,
            plan,
            ..
          } = self;
          let plan = plan.as_ref().expect("plan.is_some() checked above");
          // The alpha mode is snapshotted at begin_frame; reject a mid-frame
          // change here, before route selection (it picks the 4-channel vs
          // 3-channel route), so a flip can neither reroute nor mix modes.
          check_frozen_alpha_mode(*frozen_alpha_mode, alpha_mode, idx)?;
          match plan.kind() {
            crate::resample::SpanKind::Area => {
              if rgba.is_some() || rgba_u16.is_some() || alpha_mode.is_premultiplied() {
                return packed_rgba_u16_resample::<BITS, true, false>(
                  rgba_stream_u16,
                  // No native-Y luma stream: `GbrapN` luma_u16 is native-precision
                  // color-derived (`NATIVE_LUMA16 = true`, `NATIVE_Y_LUMA = false`),
                  // so the Y stream / scratch / de-interleave are inert.
                  &mut None,
                  resample_outputs,
                  rgb,
                  rgba,
                  rgb_u16,
                  rgba_u16,
                  luma,
                  luma_u16,
                  hsv,
                  rgba_scratch_u16,
                  rgba_color_scratch_u16,
                  rgb_scratch,
                  rgb_scratch_u16,
                  &mut std::vec::Vec::new(),
                  w,
                  plan,
                  idx,
                  use_simd,
                  alpha_mode,
                  matrix,
                  full_range,
                  |dst| {
                    gbra_to_rgba_u16_high_bit_row::<BITS, BE>(
                      g_in, b_in, r_in, a_in, dst, w, use_simd,
                    )
                  },
                  |_| {},
                );
              }
              // Straight rgb-only (alpha dropped): scatter the native-depth
              // G/B/R planes into the source-width packed u16 RGB row and feed
              // the 3-channel high-bit tail (luma_u16 at native precision —
              // `NATIVE_LUMA16 = true` — for parity with the direct path).
              // Transactional 3-channel u16 area resample (RFC #238 tail-collapse).
              return packed_rgb_u16_resample::<BITS, true>(
                rgb_stream_u16,
                resample_outputs,
                rgb,
                rgba,
                luma,
                rgb_u16,
                rgba_u16,
                luma_u16,
                hsv,
                rgb_scratch_u16,
                rgb_scratch,
                w,
                plan,
                idx,
                use_simd,
                matrix,
                full_range,
                |src_u16| {
                  gbr_to_rgb_u16_high_bit_row::<BITS, BE>(g_in, b_in, r_in, src_u16, w, use_simd)
                },
              );
            }
            crate::resample::SpanKind::Filter => {
              // Premultiplied alpha has no filter analogue (the filter engine
              // cannot un-premultiply); surface the typed `UnsupportedFilter`
              // via the area tail's reject rather than emit straight-filtered
              // premultiplied color. Straight alpha: filter all four native u16
              // channels independently (PIL RGBA semantics) when alpha survives,
              // else the 3-channel u16 filter for rgb-only outputs.
              if alpha_mode.is_premultiplied() {
                return packed_rgba_u16_resample::<BITS, true, false>(
                  rgba_stream_u16,
                  &mut None,
                  resample_outputs,
                  rgb,
                  rgba,
                  rgb_u16,
                  rgba_u16,
                  luma,
                  luma_u16,
                  hsv,
                  rgba_scratch_u16,
                  rgba_color_scratch_u16,
                  rgb_scratch,
                  rgb_scratch_u16,
                  &mut std::vec::Vec::new(),
                  w,
                  plan,
                  idx,
                  use_simd,
                  alpha_mode,
                  matrix,
                  full_range,
                  |dst| {
                    gbra_to_rgba_u16_high_bit_row::<BITS, BE>(
                      g_in, b_in, r_in, a_in, dst, w, use_simd,
                    )
                  },
                  |_| {},
                );
              }
              if rgba.is_some() || rgba_u16.is_some() {
                return packed_rgba_u16_filter_resample::<BITS, true>(
                  rgba_filter_stream_u16,
                  resample_outputs,
                  rgb,
                  rgba,
                  rgb_u16,
                  rgba_u16,
                  luma,
                  luma_u16,
                  hsv,
                  rgba_scratch_u16,
                  rgba_color_scratch_u16,
                  rgb_scratch,
                  w,
                  plan,
                  idx,
                  use_simd,
                  matrix,
                  full_range,
                  |dst| {
                    gbra_to_rgba_u16_high_bit_row::<BITS, BE>(
                      g_in, b_in, r_in, a_in, dst, w, use_simd,
                    )
                  },
                );
              }
              // Straight rgb-only (alpha dropped): scatter the native-depth
              // G/B/R planes into the source-width packed u16 RGB row and feed
              // the 3-channel high-bit filter tail (luma_u16 at native
              // precision — `NATIVE_LUMA16 = true` — for parity).
              // Transactional 3-channel u16 filter resample (RFC #238 tail-collapse).
              return packed_rgb_u16_resample_filter::<BITS, true>(
                rgb_filter_stream_u16,
                resample_outputs,
                rgb,
                rgba,
                luma,
                rgb_u16,
                rgba_u16,
                luma_u16,
                hsv,
                rgb_scratch_u16,
                rgb_scratch,
                w,
                plan,
                idx,
                use_simd,
                matrix,
                full_range,
                |src_u16| {
                  gbr_to_rgb_u16_high_bit_row::<BITS, BE>(g_in, b_in, r_in, src_u16, w, use_simd)
                },
              );
            }
          }
        }

        let Self {
          rgb,
          rgb_u16,
          rgba,
          rgba_u16,
          luma,
          luma_u16,
          hsv,
          rgb_scratch,
          ..
        } = self;
        let one_plane_start = idx * w;
        let one_plane_end = one_plane_start + w;
        let g_in = row.g();
        let b_in = row.b();
        let r_in = row.r();
        let a_in = row.a();

        // ---- u16 RGB / RGBA output (Strategy A+) -------------------------
        //
        // For GbrapN: rgba_u16 = stage rgb_u16 → expand → α-overwrite from
        // source plane (no depth conv needed — both at native BITS depth).
        let want_rgb_u16 = rgb_u16.is_some();
        let want_rgba_u16 = rgba_u16.is_some();

        if want_rgba_u16 && !want_rgb_u16 {
          // Standalone u16 RGBA — direct 4-channel kernel with real α.
          let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
          let rgba_u16_row =
            rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
          gbra_to_rgba_u16_high_bit_row::<BITS, BE>(
            g_in,
            b_in,
            r_in,
            a_in,
            rgba_u16_row,
            w,
            use_simd,
          );
        } else if want_rgb_u16 {
          let rgb_u16_buf = rgb_u16.as_deref_mut().unwrap();
          let rgb_plane_end =
            one_plane_end
              .checked_mul(3)
              .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
                w, h, 3,
              )))?;
          let rgb_plane_start = one_plane_start * 3;
          let rgb_u16_row = &mut rgb_u16_buf[rgb_plane_start..rgb_plane_end];
          gbr_to_rgb_u16_high_bit_row::<BITS, BE>(g_in, b_in, r_in, rgb_u16_row, w, use_simd);
          if want_rgba_u16 {
            // Strategy A+: expand RGB → RGBA, then overwrite α from source plane.
            let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
            let rgba_u16_row =
              rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
            expand_rgb_u16_to_rgba_u16_row::<BITS>(rgb_u16_row, rgba_u16_row, w);
            // Overwrite α slot from source plane (native depth, no shift).
            // BE propagated from the parent `GbrapHighBitFrame<'_, BITS, BE>`
            // via the sinker's `MixedSinker<GbrapN<BE>>` monomorphization.
            alpha_extract::copy_alpha_plane_u16::<BITS, BE>(a_in, rgba_u16_row, w, use_simd);
          }
        }

        // ---- native-depth luma output (no RGB staging needed) -----------
        // Compute luma_u16 first — it reads G/B/R planes directly without
        // going through the u8 staging path, so it is independent of whether
        // RGB staging happens below.
        if let Some(luma_u16_buf) = luma_u16.as_deref_mut() {
          gbr_to_luma_u16_high_bit_row::<BITS, BE>(
            g_in,
            b_in,
            r_in,
            &mut luma_u16_buf[one_plane_start..one_plane_end],
            w,
            row.matrix(),
            row.full_range(),
            use_simd,
          );
        }

        // ---- u8 RGB / RGBA / luma / HSV output --------------------------
        let want_rgb = rgb.is_some();
        let want_rgba = rgba.is_some();
        let want_luma = luma.is_some();
        let want_hsv = hsv.is_some();
        let need_rgb_staging = want_rgb || want_luma || want_hsv;

        // RGBA-only fast path — direct 4-channel kernel with real α.
        if want_rgba && !need_rgb_staging {
          let rgba_buf = rgba.as_deref_mut().unwrap();
          let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
          gbra_to_rgba_high_bit_row::<BITS, BE>(g_in, b_in, r_in, a_in, rgba_row, w, use_simd);
          return Ok(());
        }

        if !need_rgb_staging && !want_rgba {
          return Ok(());
        }

        // Stage RGB once.
        let rgb_row = rgb_row_buf_or_scratch(
          rgb.as_deref_mut(),
          rgb_scratch,
          one_plane_start,
          one_plane_end,
          w,
          h,
        )?;
        gbr_to_rgb_high_bit_row::<BITS, BE>(g_in, b_in, r_in, rgb_row, w, use_simd);

        if let Some(luma) = luma.as_deref_mut() {
          rgb_to_luma_row(
            rgb_row,
            &mut luma[one_plane_start..one_plane_end],
            w,
            row.matrix(),
            row.full_range(),
            use_simd,
          );
        }

        if let Some(hsv) = hsv.as_mut() {
          let (h, s, v) = hsv.hsv();
          rgb_to_hsv_row(
            rgb_row,
            &mut h[one_plane_start..one_plane_end],
            &mut s[one_plane_start..one_plane_end],
            &mut v[one_plane_start..one_plane_end],
            w,
            use_simd,
          );
        }

        if let Some(buf) = rgba.as_deref_mut() {
          // Strategy A+: expand rgb_row → RGBA (opaque stub), then
          // overwrite α bytes from the source A plane.
          let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
          expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
          // BE propagated from the parent frame.
          alpha_extract::copy_alpha_plane_u16_to_u8::<BITS, BE>(a_in, rgba_row, w, use_simd);
        }

        Ok(())
      }
    }
  };
}

// ---- Gbrp formats (no alpha) -------------------------------------------

impl_gbrp_high_bit!(Gbrp9, Gbrp9Sink, Gbrp9Row, 9);
impl_gbrp_high_bit!(Gbrp10, Gbrp10Sink, Gbrp10Row, 10);
impl_gbrp_high_bit!(Gbrp12, Gbrp12Sink, Gbrp12Row, 12);
impl_gbrp_high_bit!(Gbrp14, Gbrp14Sink, Gbrp14Row, 14);
impl_gbrp_high_bit!(Gbrp16, Gbrp16Sink, Gbrp16Row, 16);

// ---- Gbrap formats (with real α plane) ---------------------------------

impl_gbrap_high_bit!(Gbrap10, Gbrap10Sink, Gbrap10Row, 10);
impl_gbrap_high_bit!(Gbrap12, Gbrap12Sink, Gbrap12Row, 12);
impl_gbrap_high_bit!(Gbrap14, Gbrap14Sink, Gbrap14Row, 14);
impl_gbrap_high_bit!(Gbrap16, Gbrap16Sink, Gbrap16Row, 16);
