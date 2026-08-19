//! Sinker impls for MSB-aligned high-bit planar GBR source formats
//! (`AV_PIX_FMT_GBRP10MSB{LE,BE}` / `AV_PIX_FMT_GBRP12MSB{LE,BE}`).
//!
//! The MSB-aligned twins of the low-bit-packed `GbrpN` sinkers in
//! [`planar_gbr_high_bit`](super::planar_gbr_high_bit). The sample is in the
//! high `BITS` bits of each `u16` plane; the `gbr_*_msb_row` kernels recover it
//! (`>> (16 - BITS)`) into the same `[0, (1 << BITS) - 1]` native range as the
//! low-bit family, so every output path — including the fused area / filter
//! resample tails, which bin a staged native-depth `u16` RGB row — is the same
//! and reuses the same `BITS`-parameterized shared machinery. These formats
//! carry no alpha plane (three planes — G, B, R); `with_rgba` / `with_rgba_u16`
//! emit a constant opaque alpha.
//!
//! See [`planar_gbr_high_bit`](super::planar_gbr_high_bit) for the full output
//! path / Strategy-A documentation — this module is a kernel-swap mirror.

use super::{
  GeometryOverflow, InsufficientBuffer, MixedSinker, MixedSinkerError, RowIndexOutOfRange,
  RowShapeMismatch, RowSlice, check_dimensions_match, packed_rgb_u16_resample,
  packed_rgb_u16_resample_filter, rgb_row_buf_or_scratch, rgba_plane_row_slice,
  rgba_u16_plane_row_slice,
};
use crate::{
  PixelSink,
  row::{
    expand_rgb_to_rgba_row, expand_rgb_u16_to_rgba_u16_row, gbr_to_luma_u16_msb_row,
    gbr_to_rgb_msb_row, gbr_to_rgb_u16_msb_row, gbr_to_rgba_opaque_msb_row,
    gbr_to_rgba_opaque_u16_msb_row, rgb_to_hsv_row, rgb_to_luma_row,
  },
};

// The pattern is identical for both depths — only the const BITS value and the
// concrete Row/Sink types differ. Mirrors `impl_gbrp_high_bit!` with the
// staging kernels swapped to the MSB-recovering variants.
macro_rules! impl_gbrp_msb {
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
      /// the native-precision (MSB-recovered) G/B/R planes via Q15
      /// coefficients. Values are in `[0, (1 << BITS) - 1]` (full-range) or
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

        // Non-identity plan: scatter the native-depth (MSB-recovered) G/B/R
        // planes into a source-width packed u16 RGB row and feed the shared
        // high-bit packed-RGB resample tail (parameterized by BITS) — the same
        // tail the low-bit `GbrpN` family takes, since the staged native-u16
        // row is identical. The plan's span kind picks the engine.
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
          // Transactional resample (RFC #238 tail-collapse): scatter the MSB-aligned
          // native-depth G/B/R planes into the source-width packed-u16 RGB scratch
          // (the staged native-u16 row is identical to the low-bit `GbrpN` family's,
          // so the same shared 3-channel u16 helper serves both — `SRC_BITS = BITS`,
          // `NATIVE_LUMA16 = true`); the span kind picks the engine.
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
                gbr_to_rgb_u16_msb_row::<BITS, BE>(row.g(), row.b(), row.r(), src_u16, w, use_simd)
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
                gbr_to_rgb_u16_msb_row::<BITS, BE>(row.g(), row.b(), row.r(), src_u16, w, use_simd)
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
        let want_rgb_u16 = rgb_u16.is_some();
        let want_rgba_u16 = rgba_u16.is_some();

        if want_rgba_u16 && !want_rgb_u16 {
          let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
          let rgba_u16_row =
            rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
          gbr_to_rgba_opaque_u16_msb_row::<BITS, BE>(g_in, b_in, r_in, rgba_u16_row, w, use_simd);
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
          gbr_to_rgb_u16_msb_row::<BITS, BE>(g_in, b_in, r_in, rgb_u16_row, w, use_simd);
          if want_rgba_u16 {
            let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
            let rgba_u16_row =
              rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
            expand_rgb_u16_to_rgba_u16_row::<BITS>(rgb_u16_row, rgba_u16_row, w);
          }
        }

        // ---- native-depth luma output (no RGB staging needed) -----------
        if let Some(luma_u16_buf) = luma_u16.as_deref_mut() {
          gbr_to_luma_u16_msb_row::<BITS, BE>(
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

        if want_rgba && !need_rgb_staging {
          let rgba_buf = rgba.as_deref_mut().unwrap();
          let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
          gbr_to_rgba_opaque_msb_row::<BITS, BE>(g_in, b_in, r_in, rgba_row, w, use_simd);
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
        gbr_to_rgb_msb_row::<BITS, BE>(g_in, b_in, r_in, rgb_row, w, use_simd);

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

impl_gbrp_msb!(Gbrp10Msb, Gbrp10MsbSink, Gbrp10MsbRow, 10);
impl_gbrp_msb!(Gbrp12Msb, Gbrp12MsbSink, Gbrp12MsbRow, 12);
