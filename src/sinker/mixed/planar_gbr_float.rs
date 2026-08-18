//! Sinker impls for the float planar GBR source family (Tier 10 float).
//!
//! Two formats covered in this file:
//! - [`Gbrpf32`] (`AV_PIX_FMT_GBRPF32LE`) — three planes (G, B, R), f32,
//!   no alpha.
//! - [`Gbrapf32`] (`AV_PIX_FMT_GBRAPF32LE`) — four planes (G, B, R, A), f32,
//!   real per-pixel α sourced from the A plane.
//!
//! # Output paths (Gbrpf32)
//!
//! - `with_rgb` — clamp+scale f32 → packed `R, G, B` u8.
//! - `with_rgba` — standalone: `gbrpf32_to_rgba_row`; combo with `with_rgb`:
//!   Strategy A (expand RGB → RGBA, constant α = `0xFF`).
//! - `with_rgb_u16` — clamp+scale f32 → packed `R, G, B` u16 (x65535).
//! - `with_rgba_u16` — direct `gbrpf32_to_rgba_u16_row` (constant α = 0xFFFF).
//! - `with_rgb_f32` — lossless scatter: HDR > 1.0 and negatives preserved.
//! - `with_rgba_f32` — `gbrpf32_to_rgba_f32_row` (α = 1.0f32).
//! - `with_rgb_f16` — `gbrpf32_to_rgb_f16_row` (narrowing; IEEE-754 RNE).
//! - `with_rgba_f16` — `gbrpf32_to_rgba_f16_row` (α = f16(1.0)).
//! - `with_luma` — `gbrpf32_to_luma_row`.
//! - `with_luma_u16` — `gbrpf32_to_luma_u16_row`.
//! - `with_hsv` — `gbrpf32_to_hsv_row`.
//!
//! # Output paths (Gbrapf32)
//!
//! Same integer/luma/HSV paths as Gbrpf32 (source α is discarded for those).
//! RGBA outputs use real source α:
//! - `with_rgba` — standalone: `gbrapf32_to_rgba_row`; combo with `with_rgb`:
//!   **Strategy A+** (expand RGB → RGBA, then `copy_alpha_plane_f32_to_u8`
//!   overwrites slot 3 from the source α plane).
//! - `with_rgba_u16` — `gbrapf32_to_rgba_u16_row` (α clamped x65535).
//! - `with_rgba_f32` — `gbrapf32_to_rgba_f32_row` (lossless α pass-through).
//! - `with_rgba_f16` — `gbrapf32_to_rgba_f16_row` (α narrowed f32 → f16 RNE).

use super::{
  GeometryOverflow, InsufficientBuffer, MixedSinker, MixedSinkerError, RowIndexOutOfRange,
  RowShapeMismatch, RowSlice, check_dimensions_match, check_frozen_alpha_mode,
  packed_rgb_f32_resample, packed_rgb_f32_resample_filter, packed_rgba_f32_filter_resample,
  packed_rgba_f32_resample, planar_gbr_f32_resample_emit, rgb_row_buf_or_scratch,
  rgba_plane_row_slice, rgba_u16_plane_row_slice,
};
use crate::{
  KernelMatrix, PixelSink,
  row::{
    expand_rgb_to_rgba_row, gbrapf32_to_rgba_f16_row, gbrapf32_to_rgba_f32_row,
    gbrapf32_to_rgba_row, gbrapf32_to_rgba_u16_row, gbrpf32_to_hsv_row, gbrpf32_to_luma_row,
    gbrpf32_to_luma_u16_row, gbrpf32_to_rgb_f16_row, gbrpf32_to_rgb_f32_row, gbrpf32_to_rgb_row,
    gbrpf32_to_rgb_u16_row, gbrpf32_to_rgba_f16_row, gbrpf32_to_rgba_f32_row, gbrpf32_to_rgba_row,
    gbrpf32_to_rgba_u16_row, scalar::alpha_extract::copy_alpha_plane_f32_to_u8,
  },
  source::{Gbrapf32, Gbrapf32Row, Gbrapf32Sink, Gbrpf32, Gbrpf32Row, Gbrpf32Sink},
};

// Float-planar GBR sources are already component RGB (no chroma matrix).
// For luma derivation, BT.709 full-range weights are the conventional default.
const GBR_FLOAT_LUMA_MATRIX: KernelMatrix = KernelMatrix::Bt709;
const GBR_FLOAT_FULL_RANGE: bool = true;

// ---- Gbrpf32 accessor impl block ----------------------------------------

impl<'a, R, const BE: bool> MixedSinker<'a, Gbrpf32<BE>, R> {
  /// Attaches a packed **8-bit** RGBA output buffer. α is forced to `0xFF`
  /// (Gbrpf32 has no alpha channel). Length in bytes (`width x height x 4`).
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

  /// Attaches a packed **`u16`** RGB output buffer. Each f32 channel is
  /// clamped to `[0, 1]` and scaled to the full u16 range (x 65535).
  /// Length in `u16` elements (`width x height x 3`).
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

  /// Attaches a packed **`u16`** RGBA output buffer. Full-range scaling
  /// (x 65535); α is constant `0xFFFF`. Length in `u16` elements
  /// (`width x height x 4`).
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

  /// Attaches a packed **`f32`** RGB output buffer. Lossless planar →
  /// packed scatter — HDR values > 1.0, negatives, NaN, and Inf are
  /// preserved bit-exact. Length in `f32` elements (`width x height x 3`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgb_f32(mut self, buf: &'a mut [f32]) -> Result<Self, MixedSinkerError> {
    self.set_rgb_f32(buf)?;
    Ok(self)
  }
  /// In-place variant of [`with_rgb_f32`](Self::with_rgb_f32).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgb_f32(&mut self, buf: &'a mut [f32]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_elems(3)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::InsufficientRgbF32Buffer(
        InsufficientBuffer::new(expected, buf.len()),
      ));
    }
    self.rgb_f32 = Some(buf);
    Ok(self)
  }

  /// Attaches a packed **`f32`** RGBA output buffer. Lossless scatter with
  /// constant α = `1.0f32`. Length in `f32` elements (`width x height x 4`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgba_f32(mut self, buf: &'a mut [f32]) -> Result<Self, MixedSinkerError> {
    self.set_rgba_f32(buf)?;
    Ok(self)
  }
  /// In-place variant of [`with_rgba_f32`](Self::with_rgba_f32).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgba_f32(&mut self, buf: &'a mut [f32]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_elems(4)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::InsufficientRgbaF32Buffer(
        InsufficientBuffer::new(expected, buf.len()),
      ));
    }
    self.rgba_f32 = Some(buf);
    Ok(self)
  }

  /// Attaches a packed **`half::f16`** RGB output buffer. f32 channels are
  /// narrowed to f16 (IEEE-754 round-to-nearest-even; HDR > 65504 saturates
  /// to `f16::INFINITY`). Length in `half::f16` elements
  /// (`width x height x 3`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgb_f16(mut self, buf: &'a mut [half::f16]) -> Result<Self, MixedSinkerError> {
    self.set_rgb_f16(buf)?;
    Ok(self)
  }
  /// In-place variant of [`with_rgb_f16`](Self::with_rgb_f16).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgb_f16(&mut self, buf: &'a mut [half::f16]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_elems(3)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::InsufficientRgbF16Buffer(
        InsufficientBuffer::new(expected, buf.len()),
      ));
    }
    self.rgb_f16 = Some(buf);
    Ok(self)
  }

  /// Attaches a packed **`half::f16`** RGBA output buffer. f32 channels are
  /// narrowed to f16 (IEEE-754 RNE); α is constant `half::f16::from_f32(1.0)`.
  /// Length in `half::f16` elements (`width x height x 4`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgba_f16(mut self, buf: &'a mut [half::f16]) -> Result<Self, MixedSinkerError> {
    self.set_rgba_f16(buf)?;
    Ok(self)
  }
  /// In-place variant of [`with_rgba_f16`](Self::with_rgba_f16).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgba_f16(&mut self, buf: &'a mut [half::f16]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_elems(4)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::InsufficientRgbaF16Buffer(
        InsufficientBuffer::new(expected, buf.len()),
      ));
    }
    self.rgba_f16 = Some(buf);
    Ok(self)
  }

  /// Attaches a `u16` luma output buffer. f32 G/B/R channels are converted
  /// to u8 luma (clamp + round-half-up) then zero-extended into u16.
  /// Length in `u16` elements (`width x height`).
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

// The float planar-GBR sink is generic over the resampler `R`. The
// `gbr` feature pulls the area-resample engine in (the #146 cascade), so
// `Gbrpf32` needs no engine fence — its non-identity plan scatters the
// G/B/R planes into a source-width packed `R, G, B` f32 row, bins in
// float on the shared `AreaStream<f32>`, then de-interleaves each binned
// row back into G/B/R planes and runs the exact direct `gbrpf32_*`
// kernels (the `rgb-float` tail's packed `rgbf32_*` kernels are not
// compiled under `gbr`). Every output, `luma_u16` included, is therefore
// byte-identical to the direct path. `Gbrapf32` (the 4-plane alpha
// variant) stays pinned to `NoopResampler`: alpha-aware area resampling
// is a later batch.

impl<R, const BE: bool> Gbrpf32Sink<BE> for MixedSinker<'_, Gbrpf32<BE>, R> {}

impl<R, const BE: bool> MixedSinker<'_, Gbrpf32<BE>, R> {
  /// Runs the identity (no-resample) output derivation for one source
  /// row over every attached output buffer — the parity oracle the
  /// area-resample tail must reproduce for an identity plan. Assumes the
  /// row shape and index are already validated by the caller.
  fn gbrpf32_process_direct(
    &mut self,
    row: Gbrpf32Row<'_>,
    use_simd: bool,
  ) -> Result<(), MixedSinkerError> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();

    let g_in = row.g();
    let b_in = row.b();
    let r_in = row.r();
    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // ---- Lossless f32 pass-through (independent of all other paths) ------

    if let Some(buf) = self.rgb_f32.as_deref_mut() {
      let start = one_plane_start * 3;
      let end = one_plane_end * 3;
      gbrpf32_to_rgb_f32_row::<BE>(g_in, b_in, r_in, &mut buf[start..end], w, use_simd);
    }

    if let Some(buf) = self.rgba_f32.as_deref_mut() {
      let start = one_plane_start * 4;
      let end = one_plane_end
        .checked_mul(4)
        .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
          w, h, 4,
        )))?;
      gbrpf32_to_rgba_f32_row::<BE>(g_in, b_in, r_in, &mut buf[start..end], w, use_simd);
    }

    // ---- f16 narrowing (independent of integer paths) --------------------

    if let Some(buf) = self.rgb_f16.as_deref_mut() {
      let start = one_plane_start * 3;
      let end = one_plane_end * 3;
      gbrpf32_to_rgb_f16_row::<BE>(g_in, b_in, r_in, &mut buf[start..end], w, use_simd);
    }

    if let Some(buf) = self.rgba_f16.as_deref_mut() {
      let start = one_plane_start * 4;
      let end = one_plane_end
        .checked_mul(4)
        .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
          w, h, 4,
        )))?;
      gbrpf32_to_rgba_f16_row::<BE>(g_in, b_in, r_in, &mut buf[start..end], w, use_simd);
    }

    // ---- u16 RGB / RGBA path (direct float → u16, no staging) -----------

    if let Some(buf) = self.rgb_u16.as_deref_mut() {
      let start = one_plane_start * 3;
      let end = one_plane_end * 3;
      gbrpf32_to_rgb_u16_row::<BE>(g_in, b_in, r_in, &mut buf[start..end], w, use_simd);
    }

    if let Some(buf) = self.rgba_u16.as_deref_mut() {
      let rgba_row = rgba_u16_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      gbrpf32_to_rgba_u16_row::<BE>(g_in, b_in, r_in, rgba_row, w, use_simd);
    }

    // ---- u8 RGBA standalone fast path (no RGB / luma / HSV needed) -------

    let want_rgba = self.rgba.is_some();
    let want_rgb = self.rgb.is_some();
    let want_luma = self.luma.is_some();
    let want_luma_u16 = self.luma_u16.is_some();
    let want_hsv = self.hsv.is_some();
    let need_u8_rgb = want_rgb || want_luma || want_luma_u16 || want_hsv;

    if want_rgba && !need_u8_rgb {
      let rgba_buf = self.rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      gbrpf32_to_rgba_row::<BE>(g_in, b_in, r_in, rgba_row, w, use_simd);
      return Ok(());
    }

    if !need_u8_rgb && !want_rgba {
      return Ok(());
    }

    // ---- Stage u8 RGB once for luma / HSV / RGBA fan-out -----------------

    let Self {
      rgb,
      rgba,
      luma,
      luma_u16,
      hsv,
      rgb_scratch,
      ..
    } = self;
    let rgb_row = rgb_row_buf_or_scratch(
      rgb.as_deref_mut(),
      rgb_scratch,
      one_plane_start,
      one_plane_end,
      w,
      h,
    )?;
    gbrpf32_to_rgb_row::<BE>(g_in, b_in, r_in, rgb_row, w, use_simd);

    if let Some(luma) = luma.as_deref_mut() {
      gbrpf32_to_luma_row::<BE>(
        g_in,
        b_in,
        r_in,
        &mut luma[one_plane_start..one_plane_end],
        w,
        GBR_FLOAT_LUMA_MATRIX,
        GBR_FLOAT_FULL_RANGE,
        use_simd,
      );
    }

    if let Some(luma_u16) = luma_u16.as_deref_mut() {
      gbrpf32_to_luma_u16_row::<BE>(
        g_in,
        b_in,
        r_in,
        &mut luma_u16[one_plane_start..one_plane_end],
        w,
        GBR_FLOAT_LUMA_MATRIX,
        GBR_FLOAT_FULL_RANGE,
        use_simd,
      );
    }

    if let Some(hsv) = hsv.as_mut() {
      let (h, s, v) = hsv.hsv();
      gbrpf32_to_hsv_row::<BE>(
        g_in,
        b_in,
        r_in,
        &mut h[one_plane_start..one_plane_end],
        &mut s[one_plane_start..one_plane_end],
        &mut v[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    // Strategy A: expand RGB → RGBA (constant α = 0xFF).
    if let Some(buf) = rgba.as_deref_mut() {
      let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
    }

    Ok(())
  }
}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, Gbrpf32<BE>, R> {
  type Input<'r> = Gbrpf32Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    if let Some(stream) = self.rgb_stream_f32.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.rgb_filter_stream_f32.as_mut() {
      stream.reset();
    }
    self.resample_outputs = None;
    Ok(())
  }

  fn process(&mut self, row: Gbrpf32Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    // Defense-in-depth row-shape checks before any unsafe kernel.
    if row.g().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::GbrF32Plane,
        idx,
        w,
        row.g().len(),
      )));
    }
    if row.b().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::GbrF32Plane,
        idx,
        w,
        row.b().len(),
      )));
    }
    if row.r().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::GbrF32Plane,
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

    // Non-identity plan: scatter the G/B/R planes into a source-width
    // packed `R, G, B` f32 row and bin in float. `rgb_f32` copies the
    // binned row losslessly; every other output is derived by
    // de-interleaving the binned row back into G/B/R planes and running
    // the exact direct `gbrpf32_*` kernels, so `luma_u16` (and all
    // outputs) stay byte-identical to the direct path. The `rgb-float`
    // tail's packed `rgbf32_*` kernels are absent in a `gbr` build, hence
    // the dedicated planar emit. The span kind picks the engine — float
    // area or signed-coefficient filter (both bin the same staged
    // host-native f32 row and feed the same emit). f32 is full-range
    // float, so there is no native-depth clamp on the filter output.
    if let Some(plan) = self.plan.as_ref() {
      let Self {
        rgb,
        rgb_u16,
        rgb_f32,
        rgba_f32,
        rgb_f16,
        rgba_f16,
        rgba,
        rgba_u16,
        luma,
        luma_u16,
        hsv,
        rgb_scratch_f32,
        rgb_plane_scratch_f32,
        rgb_stream_f32,
        rgb_filter_stream_f32,
        resample_outputs,
        ..
      } = self;

      // Transactional resample (RFC #238 tail-collapse): the span kind picks the
      // shared 3-channel float driver (float area / signed-coefficient filter, both
      // binning the staged host-native f32 RGB row scattered from the G/B/R planes);
      // each derives every output via the dedicated planar `emit` that de-interleaves
      // the binned row and re-runs the exact direct `gbrpf32_*` kernels (full parity;
      // the binned row is host-native, so they run `::<HOST_NATIVE_BE>` inside the
      // emit).
      return match plan.kind() {
        crate::resample::SpanKind::Area => packed_rgb_f32_resample(
          rgb_stream_f32,
          resample_outputs,
          rgb,
          rgba,
          luma,
          rgb_u16,
          rgba_u16,
          luma_u16,
          rgb_f32,
          rgba_f32,
          rgb_f16,
          rgba_f16,
          hsv,
          rgb_scratch_f32,
          w,
          plan,
          idx,
          |src_f32| gbrpf32_to_rgb_f32_row::<BE>(row.g(), row.b(), row.r(), src_f32, w, use_simd),
          |stream,
           src,
           rgb,
           rgba,
           luma,
           rgb_u16,
           rgba_u16,
           luma_u16,
           rgb_f32,
           rgba_f32,
           rgb_f16,
           rgba_f16,
           hsv| {
            planar_gbr_f32_resample_emit(
              stream,
              plan,
              rgb,
              rgba,
              luma,
              rgb_u16,
              rgba_u16,
              luma_u16,
              rgb_f32,
              rgba_f32,
              rgb_f16,
              rgba_f16,
              hsv,
              src,
              rgb_plane_scratch_f32,
              GBR_FLOAT_LUMA_MATRIX,
              GBR_FLOAT_FULL_RANGE,
              idx,
              use_simd,
            )
          },
        ),
        crate::resample::SpanKind::Filter => packed_rgb_f32_resample_filter(
          rgb_filter_stream_f32,
          resample_outputs,
          rgb,
          rgba,
          luma,
          rgb_u16,
          rgba_u16,
          luma_u16,
          rgb_f32,
          rgba_f32,
          rgb_f16,
          rgba_f16,
          hsv,
          rgb_scratch_f32,
          w,
          plan,
          idx,
          |src_f32| gbrpf32_to_rgb_f32_row::<BE>(row.g(), row.b(), row.r(), src_f32, w, use_simd),
          |stream,
           src,
           rgb,
           rgba,
           luma,
           rgb_u16,
           rgba_u16,
           luma_u16,
           rgb_f32,
           rgba_f32,
           rgb_f16,
           rgba_f16,
           hsv| {
            planar_gbr_f32_resample_emit(
              stream,
              plan,
              rgb,
              rgba,
              luma,
              rgb_u16,
              rgba_u16,
              luma_u16,
              rgb_f32,
              rgba_f32,
              rgb_f16,
              rgba_f16,
              hsv,
              src,
              rgb_plane_scratch_f32,
              GBR_FLOAT_LUMA_MATRIX,
              GBR_FLOAT_FULL_RANGE,
              idx,
              use_simd,
            )
          },
        ),
      };
    }

    self.gbrpf32_process_direct(row, use_simd)
  }
}

// ---- Gbrapf32 accessor impl block ----------------------------------------

impl<'a, R, const BE: bool> MixedSinker<'a, Gbrapf32<BE>, R> {
  /// Attaches a packed **8-bit** RGBA output buffer. α is sourced from the
  /// A plane (real per-pixel α, clamped to `[0, 1]` and scaled x 255).
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

  /// Attaches a packed **`u16`** RGB output buffer. Clamped to `[0, 1]`
  /// and scaled x 65535. Length in `u16` elements (`width x height x 3`).
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

  /// Attaches a packed **`u16`** RGBA output buffer. Source α clamped to
  /// `[0, 1]` and scaled x 65535. Length in `u16` elements
  /// (`width x height x 4`).
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

  /// Attaches a packed **`f32`** RGB output buffer. Lossless planar →
  /// packed scatter. Length in `f32` elements (`width x height x 3`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgb_f32(mut self, buf: &'a mut [f32]) -> Result<Self, MixedSinkerError> {
    self.set_rgb_f32(buf)?;
    Ok(self)
  }
  /// In-place variant of [`with_rgb_f32`](Self::with_rgb_f32).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgb_f32(&mut self, buf: &'a mut [f32]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_elems(3)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::InsufficientRgbF32Buffer(
        InsufficientBuffer::new(expected, buf.len()),
      ));
    }
    self.rgb_f32 = Some(buf);
    Ok(self)
  }

  /// Attaches a packed **`f32`** RGBA output buffer. Source α is passed
  /// through losslessly (HDR, NaN, Inf preserved bit-exact). Length in
  /// `f32` elements (`width x height x 4`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgba_f32(mut self, buf: &'a mut [f32]) -> Result<Self, MixedSinkerError> {
    self.set_rgba_f32(buf)?;
    Ok(self)
  }
  /// In-place variant of [`with_rgba_f32`](Self::with_rgba_f32).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgba_f32(&mut self, buf: &'a mut [f32]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_elems(4)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::InsufficientRgbaF32Buffer(
        InsufficientBuffer::new(expected, buf.len()),
      ));
    }
    self.rgba_f32 = Some(buf);
    Ok(self)
  }

  /// Attaches a packed **`half::f16`** RGB output buffer. f32 → f16 narrowing
  /// (IEEE-754 RNE). Length in `half::f16` elements (`width x height x 3`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgb_f16(mut self, buf: &'a mut [half::f16]) -> Result<Self, MixedSinkerError> {
    self.set_rgb_f16(buf)?;
    Ok(self)
  }
  /// In-place variant of [`with_rgb_f16`](Self::with_rgb_f16).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgb_f16(&mut self, buf: &'a mut [half::f16]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_elems(3)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::InsufficientRgbF16Buffer(
        InsufficientBuffer::new(expected, buf.len()),
      ));
    }
    self.rgb_f16 = Some(buf);
    Ok(self)
  }

  /// Attaches a packed **`half::f16`** RGBA output buffer. Source α narrowed
  /// f32 → f16 (IEEE-754 RNE; values > 65504 saturate to `f16::INFINITY`).
  /// Length in `half::f16` elements (`width x height x 4`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgba_f16(mut self, buf: &'a mut [half::f16]) -> Result<Self, MixedSinkerError> {
    self.set_rgba_f16(buf)?;
    Ok(self)
  }
  /// In-place variant of [`with_rgba_f16`](Self::with_rgba_f16).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgba_f16(&mut self, buf: &'a mut [half::f16]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_elems(4)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::InsufficientRgbaF16Buffer(
        InsufficientBuffer::new(expected, buf.len()),
      ));
    }
    self.rgba_f16 = Some(buf);
    Ok(self)
  }

  /// Attaches a `u16` luma output buffer. Luma derived from G/B/R via
  /// clamp + round-half-up + zero-extend to u16 (α plane ignored).
  /// Length in `u16` elements (`width x height`).
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

// The float planar-GBR+alpha sink is generic over the resampler `R`. The
// `gbr` feature pulls the area-resample engine in (the #146 cascade), so
// `Gbrapf32` needs no engine fence — its non-identity plan scatters the
// G/B/R/A planes into a source-width packed `R, G, B, A` f32 row and bins
// all four channels in float on a dedicated 4-channel `AreaStream<f32>`,
// so resampled alpha is a real area mean (not forced opaque) and — under
// `Premultiplied` — color is binned premultiplied. Per finalized output
// row the resolved straight color de-interleaves into G/B/R/A planes and
// the exact direct `gbrapf32_*` / `gbrpf32_*` kernels run, so every output
// is byte-identical to a direct `Gbrapf32` conversion of the binned frame.
impl<R, const BE: bool> Gbrapf32Sink<BE> for MixedSinker<'_, Gbrapf32<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, Gbrapf32<BE>, R> {
  type Input<'r> = Gbrapf32Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    if let Some(stream) = self.rgb_stream_f32.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.rgba_stream_f32.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.rgba_filter_stream_f32.as_mut() {
      stream.reset();
    }
    self.resample_outputs = None;
    self.frozen_alpha_mode = Some(self.alpha_mode);
    Ok(())
  }

  fn process(&mut self, row: Gbrapf32Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    // Defense-in-depth row-shape checks before any unsafe kernel.
    if row.g().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::GbrF32Plane,
        idx,
        w,
        row.g().len(),
      )));
    }
    if row.b().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::GbrF32Plane,
        idx,
        w,
        row.b().len(),
      )));
    }
    if row.r().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::GbrF32Plane,
        idx,
        w,
        row.r().len(),
      )));
    }
    if row.a().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::GbrF32Plane,
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

    // Non-identity plan: scatter the G/B/R/A planes into a source-width
    // packed `R, G, B, A` f32 row and bin all four channels in float. The
    // dedicated alpha-aware tail resolves the straight color per output row
    // and runs the exact direct `gbrapf32_*` / `gbrpf32_*` kernels, so every
    // output stays byte-identical to a direct `Gbrapf32` conversion of the
    // binned frame. `gbrapf32_to_rgba_f32_row::<BE>` decodes the source wire
    // bytes to host-native f32, so the binned row is host-native and the
    // emit kernels run `::<HOST_NATIVE_BE>`. The span kind picks the engine:
    // the float area stream or the signed-coefficient filter stream (PIL
    // parity, straight alpha only — premultiplied has no filter analogue, so
    // it stays on the area tail, which un-premultiplies). f32 is full-range,
    // so there is no native-depth clamp on the filter output.
    if let Some(plan) = self.plan.as_ref() {
      let alpha_mode = self.alpha_mode;
      let g_in = row.g();
      let b_in = row.b();
      let r_in = row.r();
      let a_in = row.a();
      let Self {
        rgb,
        rgb_u16,
        rgb_f32,
        rgba_f32,
        rgb_f16,
        rgba_f16,
        rgba,
        rgba_u16,
        luma,
        luma_u16,
        hsv,
        rgba_scratch_f32,
        rgba_color_scratch_f32,
        rgba_plane_scratch_f32,
        rgba_stream_f32,
        rgba_filter_stream_f32,
        resample_outputs,
        frozen_alpha_mode,
        ..
      } = self;
      // The alpha mode is snapshotted at begin_frame; reject a mid-frame
      // change before route selection (it picks the filter vs area tail) so a
      // flip can neither reroute nor mix modes (mirrors the integer alpha
      // tails).
      check_frozen_alpha_mode(*frozen_alpha_mode, alpha_mode, idx)?;
      match plan.kind() {
        crate::resample::SpanKind::Area => {
          return packed_rgba_f32_resample(
            rgba_stream_f32,
            resample_outputs,
            rgb,
            rgba,
            luma,
            rgb_u16,
            rgba_u16,
            luma_u16,
            rgb_f32,
            rgba_f32,
            rgb_f16,
            rgba_f16,
            hsv,
            rgba_scratch_f32,
            rgba_color_scratch_f32,
            rgba_plane_scratch_f32,
            w,
            plan,
            idx,
            use_simd,
            alpha_mode,
            GBR_FLOAT_LUMA_MATRIX,
            GBR_FLOAT_FULL_RANGE,
            |dst| gbrapf32_to_rgba_f32_row::<BE>(g_in, b_in, r_in, a_in, dst, w, use_simd),
          );
        }
        crate::resample::SpanKind::Filter => {
          // Premultiplied alpha has no filter analogue (the filter engine
          // cannot un-premultiply); route to the area tail, which rejects the
          // filter plan and surfaces the typed `UnsupportedFilter` rather than
          // emit straight-filtered premultiplied color. Straight alpha filters
          // all four channels independently (PIL RGBA semantics).
          if alpha_mode.is_premultiplied() {
            return packed_rgba_f32_resample(
              rgba_stream_f32,
              resample_outputs,
              rgb,
              rgba,
              luma,
              rgb_u16,
              rgba_u16,
              luma_u16,
              rgb_f32,
              rgba_f32,
              rgb_f16,
              rgba_f16,
              hsv,
              rgba_scratch_f32,
              rgba_color_scratch_f32,
              rgba_plane_scratch_f32,
              w,
              plan,
              idx,
              use_simd,
              alpha_mode,
              GBR_FLOAT_LUMA_MATRIX,
              GBR_FLOAT_FULL_RANGE,
              |dst| gbrapf32_to_rgba_f32_row::<BE>(g_in, b_in, r_in, a_in, dst, w, use_simd),
            );
          }
          return packed_rgba_f32_filter_resample(
            rgba_filter_stream_f32,
            resample_outputs,
            rgb,
            rgba,
            luma,
            rgb_u16,
            rgba_u16,
            luma_u16,
            rgb_f32,
            rgba_f32,
            rgb_f16,
            rgba_f16,
            hsv,
            rgba_scratch_f32,
            rgba_color_scratch_f32,
            rgba_plane_scratch_f32,
            w,
            plan,
            idx,
            use_simd,
            GBR_FLOAT_LUMA_MATRIX,
            GBR_FLOAT_FULL_RANGE,
            |dst| gbrapf32_to_rgba_f32_row::<BE>(g_in, b_in, r_in, a_in, dst, w, use_simd),
          );
        }
      }
    }

    let g_in = row.g();
    let b_in = row.b();
    let r_in = row.r();
    let a_in = row.a();
    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // ---- Lossless f32 pass-through (independent of integer paths) --------
    //
    // rgb_f32 and rgba_f32 are independent — rgb_f32 scatters G/B/R only
    // (no α), rgba_f32 includes lossless source α. Run both unconditionally.

    if let Some(buf) = self.rgb_f32.as_deref_mut() {
      let start = one_plane_start * 3;
      let end = one_plane_end * 3;
      gbrpf32_to_rgb_f32_row::<BE>(g_in, b_in, r_in, &mut buf[start..end], w, use_simd);
    }

    if let Some(buf) = self.rgba_f32.as_deref_mut() {
      let start = one_plane_start * 4;
      let end = one_plane_end
        .checked_mul(4)
        .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
          w, h, 4,
        )))?;
      gbrapf32_to_rgba_f32_row::<BE>(g_in, b_in, r_in, a_in, &mut buf[start..end], w, use_simd);
    }

    // ---- f16 narrowing (independent of integer paths) --------------------

    if let Some(buf) = self.rgb_f16.as_deref_mut() {
      let start = one_plane_start * 3;
      let end = one_plane_end * 3;
      gbrpf32_to_rgb_f16_row::<BE>(g_in, b_in, r_in, &mut buf[start..end], w, use_simd);
    }

    if let Some(buf) = self.rgba_f16.as_deref_mut() {
      let start = one_plane_start * 4;
      let end = one_plane_end
        .checked_mul(4)
        .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
          w, h, 4,
        )))?;
      gbrapf32_to_rgba_f16_row::<BE>(g_in, b_in, r_in, a_in, &mut buf[start..end], w, use_simd);
    }

    // ---- u16 RGB path (direct, no staging) ------------------------------

    if let Some(buf) = self.rgb_u16.as_deref_mut() {
      let start = one_plane_start * 3;
      let end = one_plane_end * 3;
      gbrpf32_to_rgb_u16_row::<BE>(g_in, b_in, r_in, &mut buf[start..end], w, use_simd);
    }

    // ---- u16 RGBA path (direct — source α clamped + scaled) -------------

    if let Some(buf) = self.rgba_u16.as_deref_mut() {
      let rgba_row = rgba_u16_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      gbrapf32_to_rgba_u16_row::<BE>(g_in, b_in, r_in, a_in, rgba_row, w, use_simd);
    }

    // ---- u8 RGBA standalone fast path ------------------------------------

    let want_rgba = self.rgba.is_some();
    let want_rgb = self.rgb.is_some();
    let want_luma = self.luma.is_some();
    let want_luma_u16 = self.luma_u16.is_some();
    let want_hsv = self.hsv.is_some();
    let need_u8_rgb = want_rgb || want_luma || want_luma_u16 || want_hsv;

    if want_rgba && !need_u8_rgb {
      let rgba_buf = self.rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      gbrapf32_to_rgba_row::<BE>(g_in, b_in, r_in, a_in, rgba_row, w, use_simd);
      return Ok(());
    }

    if !need_u8_rgb && !want_rgba {
      return Ok(());
    }

    // ---- Stage u8 RGB once for luma / HSV / RGBA fan-out -----------------

    let Self {
      rgb,
      rgba,
      luma,
      luma_u16,
      hsv,
      rgb_scratch,
      ..
    } = self;
    let rgb_row = rgb_row_buf_or_scratch(
      rgb.as_deref_mut(),
      rgb_scratch,
      one_plane_start,
      one_plane_end,
      w,
      h,
    )?;
    gbrpf32_to_rgb_row::<BE>(g_in, b_in, r_in, rgb_row, w, use_simd);

    if let Some(luma) = luma.as_deref_mut() {
      gbrpf32_to_luma_row::<BE>(
        g_in,
        b_in,
        r_in,
        &mut luma[one_plane_start..one_plane_end],
        w,
        GBR_FLOAT_LUMA_MATRIX,
        GBR_FLOAT_FULL_RANGE,
        use_simd,
      );
    }

    if let Some(luma_u16) = luma_u16.as_deref_mut() {
      gbrpf32_to_luma_u16_row::<BE>(
        g_in,
        b_in,
        r_in,
        &mut luma_u16[one_plane_start..one_plane_end],
        w,
        GBR_FLOAT_LUMA_MATRIX,
        GBR_FLOAT_FULL_RANGE,
        use_simd,
      );
    }

    if let Some(hsv) = hsv.as_mut() {
      let (h, s, v) = hsv.hsv();
      gbrpf32_to_hsv_row::<BE>(
        g_in,
        b_in,
        r_in,
        &mut h[one_plane_start..one_plane_end],
        &mut s[one_plane_start..one_plane_end],
        &mut v[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    // Strategy A+: expand RGB → RGBA (0xFF stub), then overwrite α from
    // the source f32 α plane (clamped x 255 → u8).
    //
    // `BE = false`: `a_in` is the **direct** Gbrapf32Frame α plane, which
    // is LE-encoded f32 per the Phase-1 unified Frame contract. The helper
    // bit-normalises each f32 to host-native order before clamp/scale, so
    // the conversion compiles to a no-op on LE hosts and a `swap_bytes` on
    // BE hosts (e.g., s390x). Without this BE hosts would clamp byte-
    // swapped garbage and emit α = 0 / 255 regardless of intent. Distinct
    // from the **post-widen** routing in `planar_gbr_f16.rs`
    // (`widen_and_scatter_f16_alpha_to_u8`), which feeds host-native f32
    // scratch into the same helper with `BE = HOST_NATIVE_BE`.
    if let Some(buf) = rgba.as_deref_mut() {
      let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
      copy_alpha_plane_f32_to_u8::<BE>(a_in, rgba_row, w);
    }

    Ok(())
  }
}
