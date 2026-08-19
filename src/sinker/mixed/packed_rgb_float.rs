//! Sinker impl for the Tier 9 packed-float-RGB **source** format
//! ([`Rgbf32`] — FFmpeg `AV_PIX_FMT_RGBF32`).
//!
//! Each pixel is `3 x f32` (linear `R, G, B`). Output paths:
//! - `with_rgb` — clamp `[0, 1]` x 255 → packed `R, G, B` u8
//!   (`rgbf32_to_rgb_row`).
//! - `with_rgba` — same conversion + constant `0xFF` alpha.
//! - `with_rgb_u16` — clamp `[0, 1]` x 65535 → packed `R, G, B` u16.
//! - `with_rgba_u16` — same + constant `0xFFFF` alpha.
//! - `with_rgb_f32` — **lossless** float pass-through (HDR values >
//!   1.0 and negatives are preserved).
//! - `with_luma` / `with_luma_u16` — staged through a u8 RGB scratch
//!   row (or the user's `with_rgb` buffer if attached) and the
//!   existing `rgb_to_luma_row` / `rgb_to_luma_u16_row` kernels —
//!   matches the design used by every other RGB-input sinker.
//! - `with_hsv` — same staging, then `rgb_to_hsv_row`.
//!
//! HDR values > 1.0 saturate to the integer output range; the float
//! output preserves them losslessly.

use super::{
  InsufficientBuffer, MixedSinker, MixedSinkerError, RowIndexOutOfRange, RowShapeMismatch,
  RowSlice, check_dimensions_match, rgb_row_buf_or_scratch, rgba_plane_row_slice,
  rgba_u16_plane_row_slice,
};
#[cfg(any(feature = "yuv-planar", feature = "rgb"))]
use super::{
  packed_rgb_f32_resample, packed_rgb_f32_resample_emit, packed_rgb_f32_resample_filter,
};
use crate::{
  PixelSink,
  row::{
    rgb_to_hsv_row, rgb_to_luma_row, rgb_to_luma_u16_row, rgbf32_to_rgb_f32_row, rgbf32_to_rgb_row,
    rgbf32_to_rgb_u16_row, rgbf32_to_rgba_row, rgbf32_to_rgba_u16_row,
  },
  source::{Rgbf32, Rgbf32Row, Rgbf32Sink},
};

// ---- Rgbf32 impl -------------------------------------------------------

impl<'a, R, const BE: bool> MixedSinker<'a, Rgbf32<BE>, R> {
  /// Attaches a packed **8-bit** RGBA output buffer. Source values are
  /// clamped to `[0, 1]` and scaled by 255; alpha is forced to `0xFF`
  /// (the float source has no alpha channel).
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

  /// Attaches a `u16` RGB output buffer (`width x height x 3`
  /// elements). Each `f32` channel is clamped to `[0, 1]` and **scaled
  /// to the full u16 range** (x65535).
  ///
  /// # Naming consistency note
  ///
  /// Other source families' `with_rgb_u16` accessor preserves the
  /// source's *native integer precision* in a u16 carrier (e.g.
  /// 10-bit YUV stays in `[0, 1023]`). The `Rgbf32` variant has no
  /// native integer range to preserve, so it instead applies full-
  /// range scaling — a deliberate divergence to give callers a useful
  /// u16 output rather than refusing the operation.
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

  /// Attaches a `u16` RGBA output buffer. Same `[0, 1]` x 65535
  /// **full-range scaling** as
  /// [`with_rgb_u16`](Self::with_rgb_u16); alpha is forced to `0xFFFF`
  /// (the float source has no alpha channel). See
  /// [`with_rgb_u16`](Self::with_rgb_u16) for the divergence note vs
  /// integer-source families.
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

  /// Attaches a **`u16`** luma output buffer. Y' is computed at u8
  /// precision (matching `with_luma`'s output) and zero-extended to
  /// `u16` — same convention as the packed-YUV `with_luma_u16` family.
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

  /// Attaches a packed f32 RGB output buffer. Lossless pass-through —
  /// HDR values > 1.0 and negative values are preserved bit-exact.
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

  /// Runs the identity (no-resample) output derivation for one source
  /// row over every attached output buffer.
  ///
  /// This is the parity oracle: the byte-exact output every other path
  /// (including the area-resample tail) must reproduce for an identity
  /// plan. It recomputes `w`/`h`/`idx` from `self` + `row` and assumes
  /// the row shape and index are already validated by the caller.
  fn rgbf32_process_direct(
    &mut self,
    row: Rgbf32Row<'_>,
    use_simd: bool,
  ) -> Result<(), MixedSinkerError> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();

    let Self {
      rgb,
      rgb_u16,
      rgb_f32,
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
    let rgb_in = row.rgb();

    // Lossless f32 pass-through is independent of all other paths —
    // emit it first so the SIMD memcpy doesn't share scratch usage
    // with downstream conversions.
    if let Some(buf) = rgb_f32.as_deref_mut() {
      let f32_start = one_plane_start * 3;
      let f32_end = one_plane_end * 3;
      rgbf32_to_rgb_f32_row::<BE>(rgb_in, &mut buf[f32_start..f32_end], w, use_simd);
    }

    // u16 RGB output — direct float→u16 conversion (no staging).
    if let Some(buf) = rgb_u16.as_deref_mut() {
      let u16_start = one_plane_start * 3;
      let u16_end = one_plane_end * 3;
      rgbf32_to_rgb_u16_row::<BE>(rgb_in, &mut buf[u16_start..u16_end], w, use_simd);
    }

    // u16 RGBA output — direct float→u16 conversion (no staging).
    if let Some(buf) = rgba_u16.as_deref_mut() {
      let rgba_row = rgba_u16_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      rgbf32_to_rgba_u16_row::<BE>(rgb_in, rgba_row, w, use_simd);
    }

    // u8 RGBA standalone fast path — direct float→u8 conversion when
    // no RGB / luma / HSV consumer needs the staged u8 RGB row.
    let want_rgba_u8 = rgba.is_some();
    let want_rgb_u8 = rgb.is_some();
    let want_luma_u8 = luma.is_some();
    let want_luma_u16 = luma_u16.is_some();
    let want_hsv = hsv.is_some();
    let need_u8_rgb = want_rgb_u8 || want_luma_u8 || want_luma_u16 || want_hsv;

    if want_rgba_u8 && !need_u8_rgb {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      rgbf32_to_rgba_row::<BE>(rgb_in, rgba_row, w, use_simd);
      return Ok(());
    }

    if !need_u8_rgb && !want_rgba_u8 {
      return Ok(());
    }

    // Stage the u8 RGB scratch row once. This is the same
    // rgb_scratch-sharing pattern the Bgr24 / Rgba / etc. sinkers use:
    // when the user requested an RGB output buffer it doubles as the
    // shared u8 RGB row; otherwise we use the lazily-grown scratch.
    let rgb_row = rgb_row_buf_or_scratch(
      rgb.as_deref_mut(),
      rgb_scratch,
      one_plane_start,
      one_plane_end,
      w,
      h,
    )?;
    rgbf32_to_rgb_row::<BE>(rgb_in, rgb_row, w, use_simd);

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

    if let Some(luma_buf) = luma_u16.as_deref_mut() {
      rgb_to_luma_u16_row(
        rgb_row,
        &mut luma_buf[one_plane_start..one_plane_end],
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

    // u8 RGBA output (combined with RGB/luma/HSV path) — direct from
    // float source to keep alpha-fill cheap; the alternative would be
    // expanding from `rgb_row` via `expand_rgb_to_rgba_row`, which is
    // the same cost minus a pass over the float input. Direct is one
    // less memory pass for combined `with_rgb + with_rgba` callers.
    if let Some(buf) = rgba.as_deref_mut() {
      let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      rgbf32_to_rgba_row::<BE>(rgb_in, rgba_row, w, use_simd);
    }

    Ok(())
  }
}

// The float area stream + its staging fields exist only when the engine
// is compiled in (`rgb-float` alone does not pull in the `AreaStream`
// machinery). When the engine is present the sink accepts any resampler
// `R` and routes a non-identity plan through the float area tail. When
// it is absent the sink is pinned to the identity-only `NoopResampler`
// (the default `R`): there is no `PixelSink` for any other `R`, so a
// `MixedSinker<Rgbf32, AreaResampler>` cannot be fed rows at all — the
// type-level fence that keeps output-sized buffers from being indexed
// with source offsets on a downscale.

#[cfg(any(feature = "yuv-planar", feature = "rgb"))]
impl<R, const BE: bool> Rgbf32Sink<BE> for MixedSinker<'_, Rgbf32<BE>, R> {}

#[cfg(any(feature = "yuv-planar", feature = "rgb"))]
impl<R, const BE: bool> PixelSink for MixedSinker<'_, Rgbf32<BE>, R> {
  type Input<'r> = Rgbf32Row<'r>;
  type Error = MixedSinkerError;

  #[cfg_attr(not(tarpaulin), inline(always))]
  fn kernel_matrix(&self) -> crate::KernelMatrix {
    self.kernel_matrix
  }
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

  fn process(&mut self, row: Rgbf32Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let idx = row.row();
    let use_simd = self.simd;

    if row.rgb().len() != w * 3 {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::RgbF32Packed,
        idx,
        w * 3,
        row.rgb().len(),
      )));
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange(
        RowIndexOutOfRange::new(idx, self.height),
      ));
    }

    let Self {
      rgb,
      rgb_u16,
      rgb_f32,
      rgba,
      rgba_u16,
      luma,
      luma_u16,
      hsv,
      rgb_scratch,
      rgb_scratch_f32,
      rgb_stream_f32,
      rgb_filter_stream_f32,
      resample_outputs,
      plan,
      ..
    } = self;

    // Non-identity plan: convert the wire row to source-width host f32
    // RGB (lossless), bin it in float, then derive every attached
    // output from each finalized output row — `rgb_f32` copies the
    // binned row, every integer output mirrors the direct path's
    // clamp+scale kernels run over it. The span kind picks the engine —
    // float area or signed-coefficient filter (both bin the same staged
    // host-native f32 row and feed the same emit). f32 is full-range
    // float, so there is no native-depth clamp on the filter output.
    if let Some(plan) = plan.as_ref() {
      // Transactional resample (RFC #238 tail-collapse): the span kind picks the
      // shared 3-channel float driver (float area / signed-coefficient filter, both
      // binning the staged host-native f32 RGB row); each stages the wire row via
      // `convert` and derives every output via the `Rgbf32` `emit` (narrowed
      // outputs only — no `rgba_f32` / `rgb_f16` / `rgba_f16`).
      let matrix = row.matrix();
      let full_range = row.full_range();
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
          &mut None,
          &mut None,
          &mut None,
          hsv,
          rgb_scratch_f32,
          w,
          plan,
          idx,
          |src_f32| crate::row::rgbf32_to_rgb_f32_row::<BE>(row.rgb(), src_f32, w, use_simd),
          |stream, src, rgb, rgba, luma, rgb_u16, rgba_u16, luma_u16, rgb_f32, _, _, _, hsv| {
            packed_rgb_f32_resample_emit(
              stream,
              plan,
              rgb,
              rgba,
              luma,
              rgb_u16,
              rgba_u16,
              luma_u16,
              rgb_f32,
              hsv,
              src,
              rgb_scratch,
              matrix,
              full_range,
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
          &mut None,
          &mut None,
          &mut None,
          hsv,
          rgb_scratch_f32,
          w,
          plan,
          idx,
          |src_f32| crate::row::rgbf32_to_rgb_f32_row::<BE>(row.rgb(), src_f32, w, use_simd),
          |stream, src, rgb, rgba, luma, rgb_u16, rgba_u16, luma_u16, rgb_f32, _, _, _, hsv| {
            packed_rgb_f32_resample_emit(
              stream,
              plan,
              rgb,
              rgba,
              luma,
              rgb_u16,
              rgba_u16,
              luma_u16,
              rgb_f32,
              hsv,
              src,
              rgb_scratch,
              matrix,
              full_range,
              idx,
              use_simd,
            )
          },
        ),
      };
    }

    self.rgbf32_process_direct(row, use_simd)
  }
}

/// Identity-only fence for engine-absent builds.
///
/// Without the resample engine (`yuv-planar`/`rgb`) the float area
/// machinery is not compiled, so this sink is pinned to the default
/// [`NoopResampler`](crate::resample::NoopResampler): only the
/// identity output derivation exists, and no `PixelSink` is provided
/// for any non-identity resampler. That keeps a downscaling
/// `MixedSinker<Rgbf32, _>` from existing at the type level.
#[cfg(not(any(feature = "yuv-planar", feature = "rgb")))]
impl<const BE: bool> Rgbf32Sink<BE> for MixedSinker<'_, Rgbf32<BE>> {}

/// See the [`Rgbf32Sink`] fence note above: identity-only sink for
/// engine-absent builds, pinned to `NoopResampler`.
#[cfg(not(any(feature = "yuv-planar", feature = "rgb")))]
impl<const BE: bool> PixelSink for MixedSinker<'_, Rgbf32<BE>> {
  type Input<'r> = Rgbf32Row<'r>;
  type Error = MixedSinkerError;

  #[cfg_attr(not(tarpaulin), inline(always))]
  fn kernel_matrix(&self) -> crate::KernelMatrix {
    self.kernel_matrix
  }
  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: Rgbf32Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let idx = row.row();
    let use_simd = self.simd;

    if row.rgb().len() != w * 3 {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::RgbF32Packed,
        idx,
        w * 3,
        row.rgb().len(),
      )));
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange(
        RowIndexOutOfRange::new(idx, self.height),
      ));
    }

    self.rgbf32_process_direct(row, use_simd)
  }
}

// ---- Rgbaf32 impl (packed float RGBA — real source alpha) --------------
//
// The alpha-bearing twin of the [`Rgbf32`] sinker. Each pixel is
// `4 x f32` (linear `R, G, B, A`). Output paths mirror `Rgbf32` exactly,
// extended with the real source alpha:
// - `with_rgb` — clamp `[0, 1]` x 255 → packed `R, G, B` u8 (alpha dropped).
// - `with_rgba` — same RGB conversion + the source alpha (clamp x 255).
// - `with_rgb_u16` / `with_rgba_u16` — clamp x 65535 (full-range scaling;
//   alpha real for the RGBA form).
// - `with_rgb_f32` — lossless RGB pass-through (alpha dropped).
// - `with_rgba_f32` — lossless 4-channel pass-through (alpha preserved).
// - `with_luma` / `with_luma_u16` / `with_hsv` — staged through a u8 RGB
//   scratch row (alpha ignored), matching `Rgbf32`.

#[cfg(any(feature = "yuv-planar", feature = "rgb"))]
use super::{
  packed_rgba_f32_packed_filter, packed_rgba_f32_packed_resample, packed_rgba_f32_resample_emit,
};
use crate::{
  row::{
    rgbaf32_to_rgb_f32_row, rgbaf32_to_rgb_row, rgbaf32_to_rgb_u16_row, rgbaf32_to_rgba_f32_row,
    rgbaf32_to_rgba_row, rgbaf32_to_rgba_u16_row,
  },
  source::{Rgbaf32, Rgbaf32Row, Rgbaf32Sink},
};

impl<'a, R, const BE: bool> MixedSinker<'a, Rgbaf32<BE>, R> {
  /// Attaches a packed **8-bit** RGBA output buffer. Colour channels are
  /// clamped to `[0, 1]` and scaled by 255; alpha is the **source** alpha,
  /// also clamped to `[0, 1]` and scaled by 255 (real per-pixel alpha).
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

  /// Attaches a `u16` RGB output buffer (`width x height x 3` elements).
  /// Each colour `f32` is clamped to `[0, 1]` and **scaled to the full u16
  /// range** (x65535); alpha dropped. See the [`Rgbf32`] `with_rgb_u16`
  /// divergence note vs integer-source families.
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

  /// Attaches a `u16` RGBA output buffer. Same `[0, 1]` x 65535 full-range
  /// scaling as [`with_rgb_u16`](Self::with_rgb_u16); alpha is the real
  /// source alpha (clamp x 65535).
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

  /// Attaches a **`u16`** luma output buffer. Y' is computed at u8
  /// precision and zero-extended to `u16` (alpha ignored).
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

  /// Attaches a packed f32 RGB output buffer. Lossless pass-through of the
  /// `R, G, B` channels (alpha dropped) — HDR values > 1.0 and negatives
  /// preserved bit-exact.
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

  /// Attaches a packed f32 RGBA output buffer. **Lossless** 4-channel
  /// pass-through including the source alpha — HDR values > 1.0, negatives,
  /// NaN, and Inf are preserved bit-exact.
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

  /// Runs the identity (no-resample) output derivation for one source row
  /// over every attached output buffer — the parity oracle the
  /// area/filter-resample tail must reproduce for an identity plan.
  fn rgbaf32_process_direct(
    &mut self,
    row: Rgbaf32Row<'_>,
    use_simd: bool,
  ) -> Result<(), MixedSinkerError> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();

    let Self {
      rgb,
      rgb_u16,
      rgb_f32,
      rgba,
      rgba_u16,
      rgba_f32,
      luma,
      luma_u16,
      hsv,
      rgb_scratch,
      ..
    } = self;
    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;
    let rgba_in = row.rgba();

    // Lossless 4-channel f32 pass-through — emit first (independent).
    if let Some(buf) = rgba_f32.as_deref_mut() {
      let start = one_plane_start * 4;
      let end = one_plane_end * 4;
      rgbaf32_to_rgba_f32_row::<BE>(rgba_in, &mut buf[start..end], w, use_simd);
    }

    // Lossless RGB f32 pass-through (alpha dropped).
    if let Some(buf) = rgb_f32.as_deref_mut() {
      let start = one_plane_start * 3;
      let end = one_plane_end * 3;
      rgbaf32_to_rgb_f32_row::<BE>(rgba_in, &mut buf[start..end], w, use_simd);
    }

    // u16 RGB output (drop alpha) — direct float→u16 conversion.
    if let Some(buf) = rgb_u16.as_deref_mut() {
      let start = one_plane_start * 3;
      let end = one_plane_end * 3;
      rgbaf32_to_rgb_u16_row::<BE>(rgba_in, &mut buf[start..end], w, use_simd);
    }

    // u16 RGBA output (real alpha) — direct float→u16 conversion.
    if let Some(buf) = rgba_u16.as_deref_mut() {
      let rgba_row = rgba_u16_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      rgbaf32_to_rgba_u16_row::<BE>(rgba_in, rgba_row, w, use_simd);
    }

    // u8 RGBA standalone fast path — direct float→u8 when no RGB / luma /
    // HSV consumer needs the staged u8 RGB row.
    let want_rgba_u8 = rgba.is_some();
    let want_rgb_u8 = rgb.is_some();
    let want_luma_u8 = luma.is_some();
    let want_luma_u16 = luma_u16.is_some();
    let want_hsv = hsv.is_some();
    let need_u8_rgb = want_rgb_u8 || want_luma_u8 || want_luma_u16 || want_hsv;

    if want_rgba_u8 && !need_u8_rgb {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      rgbaf32_to_rgba_row::<BE>(rgba_in, rgba_row, w, use_simd);
      return Ok(());
    }

    if !need_u8_rgb && !want_rgba_u8 {
      return Ok(());
    }

    // Stage the u8 RGB scratch row once (alpha dropped). When the user
    // requested an RGB output buffer it doubles as the shared u8 RGB row.
    let rgb_row = rgb_row_buf_or_scratch(
      rgb.as_deref_mut(),
      rgb_scratch,
      one_plane_start,
      one_plane_end,
      w,
      h,
    )?;
    rgbaf32_to_rgb_row::<BE>(rgba_in, rgb_row, w, use_simd);

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

    if let Some(luma_buf) = luma_u16.as_deref_mut() {
      rgb_to_luma_u16_row(
        rgb_row,
        &mut luma_buf[one_plane_start..one_plane_end],
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

    // u8 RGBA output (combined with RGB/luma/HSV path) — direct from the
    // float source so the real source alpha lands in slot 3.
    if let Some(buf) = rgba.as_deref_mut() {
      let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      rgbaf32_to_rgba_row::<BE>(rgba_in, rgba_row, w, use_simd);
    }

    Ok(())
  }
}

#[cfg(any(feature = "yuv-planar", feature = "rgb"))]
impl<R, const BE: bool> Rgbaf32Sink<BE> for MixedSinker<'_, Rgbaf32<BE>, R> {}

#[cfg(any(feature = "yuv-planar", feature = "rgb"))]
impl<R, const BE: bool> PixelSink for MixedSinker<'_, Rgbaf32<BE>, R> {
  type Input<'r> = Rgbaf32Row<'r>;
  type Error = MixedSinkerError;

  #[cfg_attr(not(tarpaulin), inline(always))]
  fn kernel_matrix(&self) -> crate::KernelMatrix {
    self.kernel_matrix
  }
  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    if let Some(stream) = self.rgba_stream_f32.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.rgba_filter_stream_f32.as_mut() {
      stream.reset();
    }
    self.resample_outputs = None;
    Ok(())
  }

  fn process(&mut self, row: Rgbaf32Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let idx = row.row();
    let use_simd = self.simd;

    if row.rgba().len() != w * 4 {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::RgbaF32Packed,
        idx,
        w * 4,
        row.rgba().len(),
      )));
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange(
        RowIndexOutOfRange::new(idx, self.height),
      ));
    }

    let Self {
      rgb,
      rgb_u16,
      rgb_f32,
      rgba,
      rgba_u16,
      rgba_f32,
      luma,
      luma_u16,
      hsv,
      rgb_scratch,
      rgba_scratch_f32,
      rgba_stream_f32,
      rgba_filter_stream_f32,
      resample_outputs,
      plan,
      ..
    } = self;

    // Non-identity plan: convert the wire row to source-width host-native
    // packed RGBA f32 (lossless), bin all four channels (straight alpha),
    // then derive every attached output from each finalized output row via
    // the exact direct `rgbaf32_*` kernels. The span kind picks the engine
    // — float area or signed-coefficient filter. f32 is full-range, so
    // there is no native-depth clamp on the filter output.
    if let Some(plan) = plan.as_ref() {
      // Transactional resample (RFC #238 tail-collapse): the span kind picks the
      // shared 4-channel packed-float driver (float area / signed-coefficient
      // filter, both binning the staged host-native f32 RGBA row, straight alpha);
      // each stages the wire row via `convert` and derives every output via the
      // `Rgbaf32` `emit` (`rgb_f32` / `rgba_f32` lossless, no `rgb_f16` /
      // `rgba_f16`).
      let matrix = row.matrix();
      let full_range = row.full_range();
      return match plan.kind() {
        crate::resample::SpanKind::Area => packed_rgba_f32_packed_resample(
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
          &mut None,
          &mut None,
          hsv,
          rgba_scratch_f32,
          w,
          plan,
          idx,
          |src_rgba| crate::row::rgbaf32_to_rgba_f32_row::<BE>(row.rgba(), src_rgba, w, use_simd),
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
           _,
           _,
           hsv| {
            packed_rgba_f32_resample_emit(
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
              hsv,
              src,
              rgb_scratch,
              matrix,
              full_range,
              idx,
              use_simd,
            )
          },
        ),
        crate::resample::SpanKind::Filter => packed_rgba_f32_packed_filter(
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
          &mut None,
          &mut None,
          hsv,
          rgba_scratch_f32,
          w,
          plan,
          idx,
          |src_rgba| crate::row::rgbaf32_to_rgba_f32_row::<BE>(row.rgba(), src_rgba, w, use_simd),
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
           _,
           _,
           hsv| {
            packed_rgba_f32_resample_emit(
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
              hsv,
              src,
              rgb_scratch,
              matrix,
              full_range,
              idx,
              use_simd,
            )
          },
        ),
      };
    }

    self.rgbaf32_process_direct(row, use_simd)
  }
}

/// Identity-only fence for engine-absent builds (see the [`Rgbf32`] note).
#[cfg(not(any(feature = "yuv-planar", feature = "rgb")))]
impl<const BE: bool> Rgbaf32Sink<BE> for MixedSinker<'_, Rgbaf32<BE>> {}

/// See the [`Rgbaf32Sink`] fence note above: identity-only sink for
/// engine-absent builds, pinned to `NoopResampler`.
#[cfg(not(any(feature = "yuv-planar", feature = "rgb")))]
impl<const BE: bool> PixelSink for MixedSinker<'_, Rgbaf32<BE>> {
  type Input<'r> = Rgbaf32Row<'r>;
  type Error = MixedSinkerError;

  #[cfg_attr(not(tarpaulin), inline(always))]
  fn kernel_matrix(&self) -> crate::KernelMatrix {
    self.kernel_matrix
  }
  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: Rgbaf32Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let idx = row.row();
    let use_simd = self.simd;

    if row.rgba().len() != w * 4 {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::RgbaF32Packed,
        idx,
        w * 4,
        row.rgba().len(),
      )));
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange(
        RowIndexOutOfRange::new(idx, self.height),
      ));
    }

    self.rgbaf32_process_direct(row, use_simd)
  }
}
