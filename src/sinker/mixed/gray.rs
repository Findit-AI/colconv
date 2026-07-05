//! `MixedSinker` impls for gray source formats: `Gray8`, `GrayN<BITS>`, `Gray16`.
//!
//! Gray sources are achromatic — every pixel has luma only, no chroma.
//! All gray→RGB conversions broadcast Y to R=G=B. All gray→HSV outputs
//! have H=0 and S=0 (achromatic convention, matching OpenCV).
//!
//! Gray8 (u8 plane):
//! - `with_rgb`  → broadcast Y to [Y, Y, Y] u8.
//! - `with_rgba` → broadcast Y to [Y, Y, Y, 0xFF] u8.
//! - `with_luma` → copy Y plane (memcpy); no dedicated kernel needed.
//! - `with_luma_u16` → zero-extend Y bytes to u16.
//! - `with_hsv`  → H=0, S=0, V=Y.
//!
//! GrayN (u16 low-bit-packed, BITS ∈ {9,10,12,14}):
//! - `with_rgb`       → mask + shift (BITS→8) → broadcast to u8 RGB.
//! - `with_rgba`      → same + alpha=0xFF.
//! - `with_rgb_u16`   → mask → broadcast to u16 RGB.
//! - `with_rgba_u16`  → mask → broadcast + alpha = bits_mask<BITS>().
//! - `with_luma`      → mask + shift → u8.
//! - `with_luma_u16`  → mask → u16.
//! - `with_hsv`       → H=0, S=0, V = mask+shift→u8.
//!
//! Gray16 (u16 native):
//! - `with_rgb`       → `>> 8` → broadcast to u8 RGB.
//! - `with_rgba`      → `>> 8` → broadcast + alpha=0xFF.
//! - `with_rgb_u16`   → identity → broadcast to u16 RGB.
//! - `with_rgba_u16`  → identity → broadcast + alpha=0xFFFF.
//! - `with_luma`      → `>> 8` → u8.
//! - `with_luma_u16`  → copy (memcpy).
//! - `with_hsv`       → H=0, S=0, V = `>> 8`.
//!
//! Strategy A: when both u8 RGB and u8 RGBA are requested, compute RGB once
//! then fan out to RGBA via `expand_rgb_to_rgba_row`. Same on the u16 path.

use super::{
  GeometryOverflow, InsufficientBuffer, MixedSinker, MixedSinkerError, RowIndexOutOfRange,
  RowShapeMismatch, RowSlice, check_dimensions_match, check_frozen_alpha_mode,
  frozen_outputs_check, packed_rgba_resample, packed_rgba_u16_resample,
  packed_yuva444_filter_resample, rgb_row_buf_or_scratch, rgba_plane_row_slice,
  rgba_u16_plane_row_slice, source_luma_u16_scratch, source_luma_u32_scratch,
};
use crate::{
  PixelSink,
  resample::{AreaStream, ResampleError, ResamplePlan},
  row::{
    expand_rgb_to_rgba_row, expand_rgb_u16_to_rgba_u16_row, gray_n_to_hsv_row, gray_n_to_luma_row,
    gray_n_to_luma_u16_row, gray_n_to_rgb_row, gray_n_to_rgb_u16_row, gray_n_to_rgba_row,
    gray_n_to_rgba_u16_row, gray8_to_hsv_row, gray8_to_rgb_row, gray8_to_rgba_row,
    gray16_to_hsv_row, gray16_to_luma_row, gray16_to_luma_u16_row, gray16_to_rgb_row,
    gray16_to_rgb_u16_row, gray16_to_rgba_row, gray16_to_rgba_u16_row, gray32_to_hsv_row,
    gray32_to_luma_row, gray32_to_luma_u16_row, gray32_to_rgb_row, gray32_to_rgb_u16_row,
    gray32_to_rgba_row, gray32_to_rgba_u16_row, grayf16_to_hsv_row, grayf16_to_luma_f32_row,
    grayf16_to_luma_row, grayf16_to_luma_u16_row, grayf16_to_rgb_f32_row, grayf16_to_rgb_row,
    grayf16_to_rgb_u16_row, grayf16_to_rgba_row, grayf16_to_rgba_u16_row, grayf32_to_hsv_row,
    grayf32_to_luma_f32_row, grayf32_to_luma_row, grayf32_to_luma_u16_row, grayf32_to_rgb_f32_row,
    grayf32_to_rgb_row, grayf32_to_rgb_u16_row, grayf32_to_rgba_row, grayf32_to_rgba_u16_row,
    rgb_to_hsv_row,
    scalar::alpha_extract::{copy_alpha_ya_u8, copy_alpha_ya_u16, copy_alpha_ya_u16_to_u8},
    y_plane_to_luma_u16_row, ya8_to_hsv_row, ya8_to_luma_row, ya8_to_luma_u16_row, ya8_to_rgb_row,
    ya8_to_rgb_u16_row, ya8_to_rgba_row, ya8_to_rgba_u16_row, ya16_to_hsv_row, ya16_to_luma_row,
    ya16_to_luma_u16_row, ya16_to_rgb_row, ya16_to_rgb_u16_row, ya16_to_rgba_row,
    ya16_to_rgba_u16_row, yaf16_to_hsv_row, yaf16_to_luma_f32_row, yaf16_to_luma_row,
    yaf16_to_luma_u16_row, yaf16_to_rgb_f32_row, yaf16_to_rgb_row, yaf16_to_rgb_u16_row,
    yaf16_to_rgba_row, yaf16_to_rgba_u16_row, yaf32_to_hsv_row, yaf32_to_luma_f32_row,
    yaf32_to_luma_row, yaf32_to_luma_u16_row, yaf32_to_rgb_f32_row, yaf32_to_rgb_row,
    yaf32_to_rgb_u16_row, yaf32_to_rgba_row, yaf32_to_rgba_u16_row,
  },
  source::{
    Gray8, Gray8Row, Gray8Sink, Gray16, Gray16Row, Gray16Sink, Gray32, Gray32Row, Gray32Sink,
    Grayf16, Grayf16Row, Grayf16Sink, Grayf32, Grayf32Row, Grayf32Sink, Ya8, Ya8Row, Ya8Sink, Ya16,
    Ya16Row, Ya16Sink, Yaf16, Yaf16Row, Yaf16Sink, Yaf32, Yaf32Row, Yaf32Sink,
  },
};

// ---- Gray8 impl -------------------------------------------------------------

impl<'a, R> MixedSinker<'a, Gray8, R> {
  /// Attaches an 8-bit RGBA output buffer. Alpha is forced to `0xFF`
  /// (Gray8 has no alpha channel).
  ///
  /// Returns `Err(InsufficientRgbaBuffer)` if `buf.len() < width x height x 4`,
  /// or `Err(GeometryOverflow)` on 32-bit overflow.
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

  /// Attaches a u16 luma output buffer. Gray8 Y bytes are zero-extended
  /// to u16 (each output element equals `y_byte as u16`). Length measured
  /// in `u16` elements (`width x height`).
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

impl<R> Gray8Sink for MixedSinker<'_, Gray8, R> {}

impl<R> PixelSink for MixedSinker<'_, Gray8, R> {
  type Input<'r> = Gray8Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    // New frame: restart the luma stream(s) (lazily created in `process`,
    // so a direct-`process` caller that skips a fresh stream still gets
    // a correctly initialized first frame) and clear the frozen output
    // snapshot.
    if let Some(stream) = self.luma_stream.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.luma_filter_stream.as_mut() {
      stream.reset();
    }
    self.resample_outputs = None;
    Ok(())
  }

  fn process(&mut self, row: Gray8Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    // Row shape check — defense-in-depth before any unsafe kernel.
    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Y,
        idx,
        w,
        row.y().len(),
      )));
    }
    if idx >= h {
      return Err(MixedSinkerError::RowIndexOutOfRange(
        RowIndexOutOfRange::new(idx, h),
      ));
    }

    let Self {
      rgb,
      rgba,
      luma,
      luma_u16,
      hsv,
      rgb_scratch,
      plan,
      luma_stream,
      luma_filter_stream,
      resample_outputs,
      ..
    } = self;

    // Non-identity plan: Gray *is* a luma plane, so a single 1-channel
    // stream resamples the source Y row, then every attached output
    // derives from each finalized Y row exactly as the direct path does
    // below — luma copy, luma_u16 zero-extend, RGB broadcast, RGBA
    // broadcast + 0xFF, HSV (H=0/S=0/V=Y). The span kind picks the engine
    // (area bin or signed-coefficient filter). Row-stage only.
    if let Some(plan) = plan.as_ref() {
      let full_range = row.full_range();
      return gray8_process_resampled(
        luma_stream,
        luma_filter_stream,
        resample_outputs,
        rgb,
        rgba,
        luma,
        luma_u16,
        hsv,
        rgb_scratch,
        row.y(),
        plan,
        idx,
        use_simd,
        full_range,
      );
    }

    let y_plane = row.y();
    let full_range = row.full_range();
    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // Output-mode flags — resolved BEFORE any output row is written so the
    // atomicity preflight below runs ahead of the luma copy.
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();

    // Atomicity preflight (#308, cf. the crate's #180 resample fix and the
    // planar / semi-planar / packed-YUV siblings): the only fallible row
    // scratch this path can grow is the RGB row buffer, and its allocating
    // (`None`) arm is reached for exactly one identity shape —
    // `want_hsv && want_rgba && !want_rgb` (every other `!want_rgb` case is
    // served by a standalone RGBA / HSV fast path below that never stages RGB).
    // Because the luma plane (Gray8: Y IS luma, copied verbatim) and luma_u16
    // are written before that scratch, reserve it up front so an allocator
    // refusal returns a typed `AllocationFailed` leaving the output frame
    // untouched rather than partially mutated. The later staging call reuses
    // the already-sized buffer (default path byte-identical).
    if want_hsv && want_rgba && !want_rgb {
      rgb_row_buf_or_scratch(
        rgb.as_deref_mut(),
        rgb_scratch,
        one_plane_start,
        one_plane_end,
        w,
        h,
      )?;
    }

    // Luma u8 — Gray8: Y IS luma; copy directly (no kernel overhead).
    // Luma outputs always pass raw Y through — no full_range rescaling.
    if let Some(buf) = luma.as_deref_mut() {
      buf[one_plane_start..one_plane_end].copy_from_slice(y_plane);
    }

    // Luma u16 — zero-extend u8 Y to u16.
    if let Some(buf) = luma_u16.as_deref_mut() {
      y_plane_to_luma_u16_row(
        y_plane,
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    // u8 RGB / RGBA / HSV path (flags hoisted above for the atomicity preflight).

    // Standalone RGBA fast path — no RGB or HSV requested.
    if want_rgba && !want_rgb && !want_hsv {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      gray8_to_rgba_row(y_plane, rgba_row, w, use_simd, full_range);
      return Ok(());
    }

    // Standalone HSV fast path — for gray sources, H=0/S=0/V=Y (rescaled if
    // limited-range) without any RGB computation. Use the dedicated kernel
    // when neither RGB nor RGBA is also requested.
    if want_hsv && !want_rgb && !want_rgba {
      let hsv = hsv.as_mut().unwrap();
      let (h, s, v) = hsv.hsv();
      gray8_to_hsv_row(
        y_plane,
        &mut h[one_plane_start..one_plane_end],
        &mut s[one_plane_start..one_plane_end],
        &mut v[one_plane_start..one_plane_end],
        w,
        use_simd,
        full_range,
      );
      return Ok(());
    }

    if !want_rgb && !want_rgba && !want_hsv {
      return Ok(());
    }

    // At least RGB or RGBA (or HSV+RGB/RGBA) requested — run the RGB kernel.
    let rgb_row = rgb_row_buf_or_scratch(
      rgb.as_deref_mut(),
      rgb_scratch,
      one_plane_start,
      one_plane_end,
      w,
      h,
    )?;
    gray8_to_rgb_row(y_plane, rgb_row, w, use_simd, full_range);

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

    // Strategy A fan-out — derive RGBA from the just-computed RGB row.
    if let Some(buf) = rgba.as_deref_mut() {
      let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
    }

    Ok(())
  }
}

/// Row-stage fused resize for [`Gray8`]: a single 1-channel `u8` stream
/// resamples the source Y plane (Gray *is* a luma plane — luma is not
/// re-derived from RGB), then every attached output derives from each
/// finalized Y row using the very kernels the direct path uses, so a
/// resampled output equals the direct Gray8 path run over a frame that
/// already holds the resampled Y. The span kind picks the engine: `Area`
/// bins through [`AreaStream<u8>`], `Filter` runs the signed-coefficient
/// single-channel [`FilterStream<u8>`] (the filter twin of the bin) —
/// full-range u8, so no native-depth clamp on either. Transactional
/// preflight (RFC #238 tail-collapse): the shared
/// [`gray_luma_resample`](super::gray_luma_resample) driver runs the compare-only
/// preflight, builds the missing luma stream into a local, pregrows the emit-owned
/// RGB scratch, then commits the stream insert + the output-set freeze together —
/// so a first-row allocation failure leaves the stream field `None` and
/// `resample_outputs` uncommitted (retryable).
#[allow(clippy::too_many_arguments)]
fn gray8_process_resampled(
  luma_stream: &mut Option<std::boxed::Box<AreaStream<u8>>>,
  luma_filter_stream: &mut Option<std::boxed::Box<crate::resample::FilterStream<u8>>>,
  resample_outputs: &mut Option<super::FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  hsv: &mut Option<mediaframe::source::HsvFrameMut<'_>>,
  rgb_scratch: &mut std::vec::Vec<u8>,
  y_row: &[u8],
  plan: &ResamplePlan,
  idx: usize,
  use_simd: bool,
  full_range: bool,
) -> Result<(), MixedSinkerError> {
  let ow = plan.out_w();
  // Gray8 feeds the source Y row directly (Gray *is* a full-range u8 luma plane), so
  // the emit needs no source-luma staging scratch — its only fallible scratch is the
  // emit-owned out-width RGB row (the `need_rgb_kernel && !want_rgb` shape), pregrown
  // before the freeze via the `None`-stream pass.
  super::gray_luma_resample::<u8>(
    luma_stream,
    luma_filter_stream,
    resample_outputs,
    rgb,
    rgba,
    &mut None,
    &mut None,
    luma,
    luma_u16,
    &mut None,
    &mut None,
    hsv,
    plan,
    idx,
    |stream, rgb, rgba, _rgb_u16, _rgba_u16, luma, luma_u16, _rgb_f32, _luma_f32, hsv| {
      let want_rgb = rgb.is_some();
      let want_rgba = rgba.is_some();
      let want_hsv = hsv.is_some();
      // The RGB kernel runs when RGB output is requested, or HSV is wanted
      // alongside RGBA (HSV-only and RGBA-only take dedicated fast paths).
      let need_rgb_kernel = want_rgb || (want_hsv && want_rgba);
      // Pregrow the emit-owned out-width RGB scratch BEFORE the freeze — only the
      // `need_rgb_kernel && !want_rgb` shape stages RGB into it (every other
      // `!want_rgb` case is a standalone RGBA / HSV fast path that never touches it).
      // Routing the grow through `rgb_row_buf_or_scratch` sizes it AND consults the
      // shared RGB-scratch failpoint, keeping the feed closure infallible.
      if need_rgb_kernel && !want_rgb {
        rgb_row_buf_or_scratch(None, rgb_scratch, 0, ow, ow, plan.out_h())?;
      }
      let Some(stream) = stream else {
        return Ok(());
      };
      let mut emit = |oy: usize, binned_y: &[u8]| {
        // Luma u8 — Gray8: Y IS luma; copy the resampled row directly.
        if let Some(buf) = luma.as_deref_mut() {
          buf[oy * ow..(oy + 1) * ow].copy_from_slice(binned_y);
        }
        // Luma u16 — zero-extend the resampled Y bytes to u16.
        if let Some(buf) = luma_u16.as_deref_mut() {
          y_plane_to_luma_u16_row(binned_y, &mut buf[oy * ow..(oy + 1) * ow], ow, use_simd);
        }

        // Standalone RGBA fast path — no RGB or HSV requested.
        if want_rgba && !want_rgb && !want_hsv {
          let buf = rgba.as_deref_mut().unwrap();
          gray8_to_rgba_row(
            binned_y,
            &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow],
            ow,
            use_simd,
            full_range,
          );
          return;
        }

        // Standalone HSV fast path — H=0/S=0/V=Y with no RGB computation.
        if want_hsv && !want_rgb && !want_rgba {
          let hsv = hsv.as_mut().unwrap();
          let (hp, sp, vp) = hsv.hsv();
          gray8_to_hsv_row(
            binned_y,
            &mut hp[oy * ow..(oy + 1) * ow],
            &mut sp[oy * ow..(oy + 1) * ow],
            &mut vp[oy * ow..(oy + 1) * ow],
            ow,
            use_simd,
            full_range,
          );
          return;
        }

        if !need_rgb_kernel {
          return;
        }

        // RGB kernel once — into the user buffer if attached, else scratch.
        if let Some(buf) = rgb.as_deref_mut() {
          let rgb_row = &mut buf[oy * 3 * ow..(oy + 1) * 3 * ow];
          gray8_to_rgb_row(binned_y, rgb_row, ow, use_simd, full_range);
          if let Some(hsv) = hsv.as_mut() {
            let (hp, sp, vp) = hsv.hsv();
            rgb_to_hsv_row(
              rgb_row,
              &mut hp[oy * ow..(oy + 1) * ow],
              &mut sp[oy * ow..(oy + 1) * ow],
              &mut vp[oy * ow..(oy + 1) * ow],
              ow,
              use_simd,
            );
          }
          if let Some(buf) = rgba.as_deref_mut() {
            expand_rgb_to_rgba_row(rgb_row, &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow], ow);
          }
        } else {
          let rgb_row = &mut rgb_scratch[..ow * 3];
          gray8_to_rgb_row(binned_y, rgb_row, ow, use_simd, full_range);
          if let Some(hsv) = hsv.as_mut() {
            let (hp, sp, vp) = hsv.hsv();
            rgb_to_hsv_row(
              rgb_row,
              &mut hp[oy * ow..(oy + 1) * ow],
              &mut sp[oy * ow..(oy + 1) * ow],
              &mut vp[oy * ow..(oy + 1) * ow],
              ow,
              use_simd,
            );
          }
          if let Some(buf) = rgba.as_deref_mut() {
            expand_rgb_to_rgba_row(rgb_row, &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow], ow);
          }
        }
      };
      // The per-output fan-out is identical for both engines, so feed it to
      // whichever stream the span kind selected (`&mut F` is itself `FnMut`);
      // only one engine runs per frame.
      match stream {
        super::GrayLumaStreamMut::Area(s) => s.feed_row(idx, y_row, use_simd, &mut emit),
        super::GrayLumaStreamMut::Filter(s) => s.feed_row(idx, y_row, use_simd, &mut emit),
      }?;
      Ok(())
    },
  )
}

// ---- GrayN impl (const BITS) ------------------------------------------------
//
// We ship one const-generic helper that serves all 4 bit depths (9/10/12/14).
// Each alias (Gray9/10/12/14) gets its own builder impl, all forwarding to
// the same MixedSinker fields and the same const-generic kernels.

/// Internal process implementation for GrayN formats. Called by all four
/// `PixelSink::process` impls via their per-format `const BITS: u32`.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn process_gray_n<'a, const BITS: u32, const BE: bool>(
  w: usize,
  h: usize,
  idx: usize,
  use_simd: bool,
  full_range: bool,
  y_plane: &[u16],
  rgb: &mut Option<&'a mut [u8]>,
  rgb_u16: &mut Option<&'a mut [u16]>,
  rgba: &mut Option<&'a mut [u8]>,
  rgba_u16: &mut Option<&'a mut [u16]>,
  luma: &mut Option<&'a mut [u8]>,
  luma_u16: &mut Option<&'a mut [u16]>,
  hsv: &mut Option<mediaframe::source::HsvFrameMut<'a>>,
  rgb_scratch: &mut std::vec::Vec<u8>,
) -> Result<(), MixedSinkerError> {
  let one_plane_start = idx * w;
  let one_plane_end = one_plane_start + w;

  // Output-mode flags — resolved BEFORE any output row is written so the
  // atomicity preflight below runs ahead of the luma writes.
  let want_rgb = rgb.is_some();
  let want_rgba = rgba.is_some();
  let want_hsv = hsv.is_some();

  // Atomicity preflight (#308, cf. the crate's #180 resample fix and the
  // planar / semi-planar / packed-YUV siblings): the only fallible row scratch
  // this path can grow is the RGB row buffer, and its allocating (`None`) arm
  // is reached for exactly one identity shape — `want_hsv && want_rgba &&
  // !want_rgb` (every other `!want_rgb` case is served by a standalone RGBA /
  // HSV fast path below that never stages RGB). Because raw-Y luma, luma_u16
  // and the u16 RGB/RGBA rows are written before that scratch, reserve it up
  // front so an allocator refusal returns a typed `AllocationFailed` leaving
  // the output frame untouched rather than partially mutated. The later staging
  // call reuses the already-sized buffer (default path byte-identical).
  if want_hsv && want_rgba && !want_rgb {
    rgb_row_buf_or_scratch(
      rgb.as_deref_mut(),
      rgb_scratch,
      one_plane_start,
      one_plane_end,
      w,
      h,
    )?;
  }

  // Luma u8 — always passes raw Y through, no full_range rescaling.
  if let Some(buf) = luma.as_deref_mut() {
    gray_n_to_luma_row::<BITS, BE>(
      y_plane,
      &mut buf[one_plane_start..one_plane_end],
      w,
      use_simd,
    );
  }

  // Luma u16 — always passes raw Y through, no full_range rescaling.
  if let Some(buf) = luma_u16.as_deref_mut() {
    gray_n_to_luma_u16_row::<BITS, BE>(
      y_plane,
      &mut buf[one_plane_start..one_plane_end],
      w,
      use_simd,
    );
  }

  // u16 RGB / RGBA path (Strategy A).
  let want_rgb_u16 = rgb_u16.is_some();
  let want_rgba_u16 = rgba_u16.is_some();

  if want_rgba_u16 && !want_rgb_u16 {
    let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
    let rgba_u16_row =
      rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
    gray_n_to_rgba_u16_row::<BITS, BE>(y_plane, rgba_u16_row, w, use_simd, full_range);
  } else if want_rgb_u16 {
    let rgb_u16_buf = rgb_u16.as_deref_mut().unwrap();
    let rgb_plane_start = one_plane_start * 3;
    let rgb_plane_end = one_plane_end
      .checked_mul(3)
      .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
        w, h, 3,
      )))?;
    let rgb_u16_row = &mut rgb_u16_buf[rgb_plane_start..rgb_plane_end];
    gray_n_to_rgb_u16_row::<BITS, BE>(y_plane, rgb_u16_row, w, use_simd, full_range);
    if want_rgba_u16 {
      let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row =
        rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
      expand_rgb_u16_to_rgba_u16_row::<BITS>(rgb_u16_row, rgba_u16_row, w);
    }
  }

  // u8 RGB / RGBA / HSV path (flags hoisted above for the atomicity preflight).

  // Standalone RGBA fast path.
  if want_rgba && !want_rgb && !want_hsv {
    let rgba_buf = rgba.as_deref_mut().unwrap();
    let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
    gray_n_to_rgba_row::<BITS, BE>(y_plane, rgba_row, w, use_simd, full_range);
    return Ok(());
  }

  // Standalone HSV fast path — gray sources always have H=0, S=0, V=Y8
  // (rescaled if limited-range).
  if want_hsv && !want_rgb && !want_rgba {
    let hsv = hsv.as_mut().unwrap();
    let (h, s, v) = hsv.hsv();
    gray_n_to_hsv_row::<BITS, BE>(
      y_plane,
      &mut h[one_plane_start..one_plane_end],
      &mut s[one_plane_start..one_plane_end],
      &mut v[one_plane_start..one_plane_end],
      w,
      use_simd,
      full_range,
    );
    return Ok(());
  }

  if !want_rgb && !want_rgba && !want_hsv {
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
  gray_n_to_rgb_row::<BITS, BE>(y_plane, rgb_row, w, use_simd, full_range);

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
    let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
    expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
  }

  Ok(())
}

/// Common row-shape validator for GrayN sinkers.
#[inline(always)]
fn check_gray_n_row_shape(
  y_len: usize,
  w: usize,
  idx: usize,
  h: usize,
) -> Result<(), MixedSinkerError> {
  if y_len != w {
    return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
      RowSlice::Y,
      idx,
      w,
      y_len,
    )));
  }
  if idx >= h {
    return Err(MixedSinkerError::RowIndexOutOfRange(
      RowIndexOutOfRange::new(idx, h),
    ));
  }
  Ok(())
}

// ---- Per-bit-depth builder impls for GrayN ----------------------------------

macro_rules! impl_gray_n_sinker {
  ($marker:ident, $row:ident, $sink:ident, $bits:expr) => {
    impl<'a, R, const BE: bool> MixedSinker<'a, $marker<BE>, R> {
      /// Attaches an 8-bit RGBA output buffer. Alpha is forced to `0xFF`.
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

      /// Attaches a u16 RGB output buffer. Samples are masked to the low
      /// `BITS` bits; length is in `u16` elements (`width x height x 3`).
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

      /// Attaches a u16 RGBA output buffer. Samples masked to low `BITS` bits;
      /// alpha = `(1 << BITS) - 1` (full-range opaque). Length in `u16` elements
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

      /// Attaches a u16 luma output buffer. Samples masked to low `BITS`
      /// bits; length in `u16` elements (`width x height`).
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

    impl<R, const BE: bool> $sink<BE> for MixedSinker<'_, $marker<BE>, R> {}

    impl<R, const BE: bool> PixelSink for MixedSinker<'_, $marker<BE>, R> {
      type Input<'r> = $row<'r>;
      type Error = MixedSinkerError;

      fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
        check_dimensions_match(self.width, self.height, width, height)?;
        // New frame: restart the u16 luma stream(s) (lazily created in
        // `process`) and clear the frozen output snapshot, mirroring the
        // Gray16 path so a reused resampling sink re-sequences from row 0.
        if let Some(stream) = self.luma_stream_u16.as_mut() {
          stream.reset();
        }
        if let Some(stream) = self.luma_filter_stream_u16.as_mut() {
          stream.reset();
        }
        self.resample_outputs = None;
        Ok(())
      }

      fn process(&mut self, row: $row<'_>) -> Result<(), Self::Error> {
        let w = self.width;
        let h = self.height;
        let use_simd = self.simd;
        let idx = row.row();
        let full_range = row.full_range();
        check_gray_n_row_shape(row.y().len(), w, idx, h)?;
        let Self {
          rgb,
          rgb_u16,
          rgba,
          rgba_u16,
          luma,
          luma_u16,
          hsv,
          rgb_scratch,
          plan,
          luma_stream_u16,
          luma_filter_stream_u16,
          luma_scratch_u16,
          resample_outputs,
          ..
        } = self;

        // Non-identity plan: GrayN *is* a (low-bit-packed) u16 luma plane,
        // so the wire row converts to a source-width host-native u16 luma
        // plane (the same kernel the direct `luma_u16` path uses), a single
        // 1-channel stream resamples it at u16 precision, then every
        // attached output derives from each finalized u16 luma row exactly
        // as the direct path does. The span kind picks the engine (area bin
        // or signed-coefficient filter); the binned luma is clamped to the
        // native max `(1 << BITS) - 1` before any derive so a signed-kernel
        // overshoot can't wrap the sub-16-bit mask. Row-stage only.
        if let Some(plan) = plan.as_ref() {
          return gray_n_process_resampled::<$bits, BE>(
            luma_stream_u16,
            luma_filter_stream_u16,
            luma_scratch_u16,
            resample_outputs,
            rgb,
            rgb_u16,
            rgba,
            rgba_u16,
            luma,
            luma_u16,
            hsv,
            row.y(),
            plan,
            w,
            idx,
            use_simd,
            full_range,
          );
        }

        process_gray_n::<$bits, BE>(
          w,
          h,
          idx,
          use_simd,
          full_range,
          row.y(),
          rgb,
          rgb_u16,
          rgba,
          rgba_u16,
          luma,
          luma_u16,
          hsv,
          rgb_scratch,
        )
      }
    }
  };
}

// Import the gray walker types for the macro instantiation.
use crate::source::{
  Gray9, Gray9Row, Gray9Sink, Gray10, Gray10Row, Gray10Sink, Gray12, Gray12Row, Gray12Sink, Gray14,
  Gray14Row, Gray14Sink,
};

impl_gray_n_sinker!(Gray9, Gray9Row, Gray9Sink, 9);
impl_gray_n_sinker!(Gray10, Gray10Row, Gray10Sink, 10);
impl_gray_n_sinker!(Gray12, Gray12Row, Gray12Sink, 12);
impl_gray_n_sinker!(Gray14, Gray14Row, Gray14Sink, 14);

/// Row-stage fused resize for [`GrayN`](crate::source::GrayNFrame) (`BITS`
/// ∈ {9, 10, 12, 14}): the wire row converts to a source-width
/// **host-native** `u16` luma plane via the kernel the direct `luma_u16`
/// path uses (`gray_n_to_luma_u16_row::<BITS, BE>` with the source wire
/// `BE`, masking to the low `BITS` bits), a single 1-channel u16 stream
/// resamples it (`Area` bins, `Filter` runs the signed-coefficient
/// `FilterStream<u16>`), then every attached output derives from each
/// finalized resampled u16 luma row using the direct kernels. Because the
/// resampled row is already host-native, those derive kernels run with
/// `HOST_NATIVE_BE = cfg!(target_endian = "big")` — the identity recovery
/// for an already-host-native sample, so on a BE host the source→luma swap
/// and the luma→output no-op are not double-swapped.
///
/// Native-depth clamp: `GrayN` is sub-16-bit, so a signed filter kernel
/// (`CatmullRom` / `Lanczos3`) can overshoot a legal edge above the native
/// max `(1 << BITS) - 1` even though the `FilterStream` clamps to the full
/// `u16` range. The `gray_n_to_*` derive kernels finish with `raw & mask`,
/// which **wraps** an over-range sample (e.g. `4100 & 0xFFF == 4` for a
/// 12-bit source) instead of clipping it. So every resampled sample is
/// clamped to the native max before any derive — the mask then a value
/// no-op. The area path never overshoots, so the clamp is a value no-op
/// there. Atomic preflight: freeze, sequence check, stream creation, and
/// source + clamp staging all precede the first feed, so a failure mutates
/// no caller output.
#[allow(clippy::too_many_arguments)]
fn gray_n_process_resampled<'a, const BITS: u32, const BE: bool>(
  luma_stream_u16: &mut Option<std::boxed::Box<AreaStream<u16>>>,
  luma_filter_stream_u16: &mut Option<std::boxed::Box<crate::resample::FilterStream<u16>>>,
  luma_scratch_u16: &mut std::vec::Vec<u16>,
  resample_outputs: &mut Option<super::FrozenOutputs>,
  rgb: &mut Option<&'a mut [u8]>,
  rgb_u16: &mut Option<&'a mut [u16]>,
  rgba: &mut Option<&'a mut [u8]>,
  rgba_u16: &mut Option<&'a mut [u16]>,
  luma: &mut Option<&'a mut [u8]>,
  luma_u16: &mut Option<&'a mut [u16]>,
  hsv: &mut Option<mediaframe::source::HsvFrameMut<'a>>,
  y_row: &[u16],
  plan: &ResamplePlan,
  w: usize,
  idx: usize,
  use_simd: bool,
  full_range: bool,
) -> Result<(), MixedSinkerError> {
  const { assert!(BITS < 16, "GrayN carries fewer than 16 active bits") };
  // The resampled u16 luma row is host-native; the direct kernels recover
  // an already-host-native sample with `::<HOST_NATIVE_BE>` (a no-op swap),
  // matching the direct path's `::<BE>` applied to a wire sample.
  const HOST_NATIVE_BE: bool = cfg!(target_endian = "big");
  // The native max `(1 << BITS) - 1`: a signed filter overshoot is clamped
  // here so the derive kernels' `& mask` doesn't wrap an over-range sample.
  let native_max: u16 = ((1u32 << BITS) - 1) as u16;
  let ow = plan.out_w();
  // Transactional 1-channel gray luma resample (RFC #238 tail-collapse): GrayN *is*
  // a low-bit-packed u16 luma plane — the wire row converts to a source-width
  // host-native u16 luma plane (staged in the emit's `None` pregrow), a single u16
  // stream resamples it, and every output derives from each finalized
  // native-max-clamped u16 luma row via the direct kernels.
  super::gray_luma_resample::<u16>(
    luma_stream_u16,
    luma_filter_stream_u16,
    resample_outputs,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    luma,
    luma_u16,
    &mut None,
    &mut None,
    hsv,
    plan,
    idx,
    |stream, rgb, rgba, rgb_u16, rgba_u16, luma, luma_u16, _rgb_f32, _luma_f32, hsv| {
      let want_rgb = rgb.is_some();
      let want_rgb_u16 = rgb_u16.is_some();
      let want_rgba = rgba.is_some();
      let want_rgba_u16 = rgba_u16.is_some();
      let want_hsv = hsv.is_some();
      // The u8 RGB kernel runs only when RGB output is requested; HSV (with or
      // without RGBA) and standalone RGBA derive directly from luma, so the
      // resample path needs no RGB scratch.
      let need_rgb_kernel = want_rgb;
      // Pregrow the source-width host-native u16 luma staging plus the out-width
      // native-max clamp staging BEFORE the freeze (the clamp holds the clamped
      // resampled row the derive kernels read, so their `& mask` is a value no-op).
      let src_len = w;
      let clamp_len = ow;
      if luma_scratch_u16.len() < src_len + clamp_len {
        let extra = src_len + clamp_len - luma_scratch_u16.len();
        luma_scratch_u16.try_reserve_exact(extra).map_err(|_| {
          MixedSinkerError::Resample(ResampleError::AllocationFailed(
            crate::resample::PlanGeometry::new(
              plan.src_w(),
              plan.src_h(),
              plan.out_w(),
              plan.out_h(),
            ),
          ))
        })?;
        luma_scratch_u16.resize(src_len + clamp_len, 0);
      }
      let Some(stream) = stream else {
        return Ok(());
      };
      let (src_luma, clamp_scratch) = luma_scratch_u16.split_at_mut(src_len);
      let src_luma = &mut src_luma[..src_len];
      let clamp_scratch = &mut clamp_scratch[..clamp_len];
      // Convert the wire GrayN row to host-native u16 luma — the source wire
      // `::<BE>`, the same kernel the direct `luma_u16` path uses (masks to
      // the low `BITS` bits).
      gray_n_to_luma_u16_row::<BITS, BE>(y_row, src_luma, w, use_simd);
      let src_luma: &[u16] = src_luma;
      let mut emit = |oy: usize, resampled_y: &[u16]| {
        // Clamp the resampled row to the native max so the derive kernels'
        // `& mask` doesn't wrap a signed-filter overshoot (value no-op for
        // the area path).
        let binned_y = &mut clamp_scratch[..ow];
        for (d, &s) in binned_y.iter_mut().zip(resampled_y.iter()) {
          *d = s.min(native_max);
        }
        let binned_y: &[u16] = binned_y;

        // Luma u16 — host-native pass-through of the clamped u16 luma (the
        // kernel's mask is a value no-op after the clamp).
        if let Some(buf) = luma_u16.as_deref_mut() {
          gray_n_to_luma_u16_row::<BITS, HOST_NATIVE_BE>(
            binned_y,
            &mut buf[oy * ow..(oy + 1) * ow],
            ow,
            use_simd,
          );
        }
        // Luma u8 — `>> (BITS - 8)` narrowing of the clamped u16 luma.
        if let Some(buf) = luma.as_deref_mut() {
          gray_n_to_luma_row::<BITS, HOST_NATIVE_BE>(
            binned_y,
            &mut buf[oy * ow..(oy + 1) * ow],
            ow,
            use_simd,
          );
        }

        // u16 RGB / RGBA (Strategy A) — native broadcast at the source depth.
        if want_rgba_u16 && !want_rgb_u16 {
          let buf = rgba_u16.as_deref_mut().unwrap();
          gray_n_to_rgba_u16_row::<BITS, HOST_NATIVE_BE>(
            binned_y,
            &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow],
            ow,
            use_simd,
            full_range,
          );
        } else if want_rgb_u16 {
          let buf = rgb_u16.as_deref_mut().unwrap();
          let rgb_u16_row = &mut buf[oy * 3 * ow..(oy + 1) * 3 * ow];
          gray_n_to_rgb_u16_row::<BITS, HOST_NATIVE_BE>(
            binned_y,
            rgb_u16_row,
            ow,
            use_simd,
            full_range,
          );
          if let Some(buf) = rgba_u16.as_deref_mut() {
            expand_rgb_u16_to_rgba_u16_row::<BITS>(
              rgb_u16_row,
              &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow],
              ow,
            );
          }
        }

        // Standalone u8 RGBA fast path — no RGB or HSV requested.
        if want_rgba && !need_rgb_kernel && !want_hsv {
          let buf = rgba.as_deref_mut().unwrap();
          gray_n_to_rgba_row::<BITS, HOST_NATIVE_BE>(
            binned_y,
            &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow],
            ow,
            use_simd,
            full_range,
          );
          return;
        }

        // Standalone HSV fast path — H=0 / S=0 / V=Y8 with no RGB computation
        // (plus an optional standalone RGBA, matching the direct path).
        if want_hsv && !want_rgb {
          let hsv = hsv.as_mut().unwrap();
          let (hp, sp, vp) = hsv.hsv();
          gray_n_to_hsv_row::<BITS, HOST_NATIVE_BE>(
            binned_y,
            &mut hp[oy * ow..(oy + 1) * ow],
            &mut sp[oy * ow..(oy + 1) * ow],
            &mut vp[oy * ow..(oy + 1) * ow],
            ow,
            use_simd,
            full_range,
          );
          if let Some(buf) = rgba.as_deref_mut() {
            gray_n_to_rgba_row::<BITS, HOST_NATIVE_BE>(
              binned_y,
              &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow],
              ow,
              use_simd,
              full_range,
            );
          }
          return;
        }

        if !need_rgb_kernel {
          return;
        }

        // Reached only when RGB is attached (need_rgb_kernel == want_rgb), so
        // the kernel writes the user buffer; HSV-from-RGB and the RGBA fan-out
        // follow, exactly as the direct path's RGB-kernel branch does.
        let buf = rgb
          .as_deref_mut()
          .expect("need_rgb_kernel implies RGB is attached");
        let rgb_row = &mut buf[oy * 3 * ow..(oy + 1) * 3 * ow];
        gray_n_to_rgb_row::<BITS, HOST_NATIVE_BE>(binned_y, rgb_row, ow, use_simd, full_range);
        if let Some(hsv) = hsv.as_mut() {
          let (hp, sp, vp) = hsv.hsv();
          rgb_to_hsv_row(
            rgb_row,
            &mut hp[oy * ow..(oy + 1) * ow],
            &mut sp[oy * ow..(oy + 1) * ow],
            &mut vp[oy * ow..(oy + 1) * ow],
            ow,
            use_simd,
          );
        }
        if let Some(buf) = rgba.as_deref_mut() {
          expand_rgb_to_rgba_row(rgb_row, &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow], ow);
        }
      };
      // The per-output fan-out is identical for both engines, so feed it to
      // whichever stream the span kind selected; only one runs per frame.
      match stream {
        super::GrayLumaStreamMut::Area(s) => s.feed_row(idx, src_luma, use_simd, &mut emit),
        super::GrayLumaStreamMut::Filter(s) => s.feed_row(idx, src_luma, use_simd, &mut emit),
      }?;
      Ok(())
    },
  )
}

// ---- Gray16 impl ------------------------------------------------------------

impl<'a, R, const BE: bool> MixedSinker<'a, Gray16<BE>, R> {
  /// Attaches an 8-bit RGBA output buffer. Alpha is forced to `0xFF`.
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

  /// Attaches a u16 RGB output buffer (`>> 8` is NOT applied — native
  /// 16-bit broadcast). Length in `u16` elements (`width x height x 3`).
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

  /// Attaches a u16 RGBA output buffer (native 16-bit broadcast; alpha
  /// = `0xFFFF`). Length in `u16` elements (`width x height x 4`).
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

  /// Attaches a u16 luma output buffer (identity copy of the Gray16 Y
  /// plane). Length in `u16` elements (`width x height`).
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

impl<R, const BE: bool> Gray16Sink<BE> for MixedSinker<'_, Gray16<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, Gray16<BE>, R> {
  type Input<'r> = Gray16Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    // New frame: restart the u16 luma stream(s) (lazily created in
    // `process`) and clear the frozen output snapshot, mirroring the
    // Gray8 path so a reused resampling sink re-sequences from row 0.
    if let Some(stream) = self.luma_stream_u16.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.luma_filter_stream_u16.as_mut() {
      stream.reset();
    }
    self.resample_outputs = None;
    Ok(())
  }

  fn process(&mut self, row: Gray16Row<'_>) -> Result<(), Self::Error> {
    const BITS: u32 = 16;
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;
    let full_range = row.full_range();

    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Y,
        idx,
        w,
        row.y().len(),
      )));
    }
    if idx >= h {
      return Err(MixedSinkerError::RowIndexOutOfRange(
        RowIndexOutOfRange::new(idx, h),
      ));
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
      plan,
      luma_stream_u16,
      luma_filter_stream_u16,
      luma_scratch_u16,
      resample_outputs,
      ..
    } = self;

    // Non-identity plan: Gray16 *is* a u16 luma plane, so the wire row
    // converts to a source-width host-native u16 luma plane (the same
    // kernel the direct `luma_u16` path uses), a single 1-channel stream
    // resamples it at u16 precision, then every attached output derives
    // from each finalized u16 luma row exactly as the direct path does
    // below. The span kind picks the engine (area bin or signed-coefficient
    // filter). Row-stage only.
    if let Some(plan) = plan.as_ref() {
      return gray16_process_resampled::<BE>(
        luma_stream_u16,
        luma_filter_stream_u16,
        luma_scratch_u16,
        resample_outputs,
        rgb,
        rgb_u16,
        rgba,
        rgba_u16,
        luma,
        luma_u16,
        hsv,
        row.y(),
        plan,
        w,
        idx,
        use_simd,
        full_range,
      );
    }

    let y_plane = row.y();
    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // Luma u8 — shift >> 8.
    if let Some(buf) = luma.as_deref_mut() {
      gray16_to_luma_row::<BE>(
        y_plane,
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    // Luma u16 — identity copy.
    if let Some(buf) = luma_u16.as_deref_mut() {
      gray16_to_luma_u16_row::<BE>(
        y_plane,
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    // u16 RGB / RGBA path (Strategy A).
    let want_rgb_u16 = rgb_u16.is_some();
    let want_rgba_u16 = rgba_u16.is_some();

    if want_rgba_u16 && !want_rgb_u16 {
      let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row =
        rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
      gray16_to_rgba_u16_row::<BE>(y_plane, rgba_u16_row, w, use_simd, full_range);
    } else if want_rgb_u16 {
      let rgb_u16_buf = rgb_u16.as_deref_mut().unwrap();
      let rgb_plane_start = one_plane_start * 3;
      let rgb_plane_end =
        one_plane_end
          .checked_mul(3)
          .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
            w, h, 3,
          )))?;
      let rgb_u16_row = &mut rgb_u16_buf[rgb_plane_start..rgb_plane_end];
      gray16_to_rgb_u16_row::<BE>(y_plane, rgb_u16_row, w, use_simd, full_range);
      if want_rgba_u16 {
        let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
        let rgba_u16_row =
          rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
        expand_rgb_u16_to_rgba_u16_row::<BITS>(rgb_u16_row, rgba_u16_row, w);
      }
    }

    // u8 RGB / RGBA / HSV (Strategy A).
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();
    // Only need the RGB kernel when an RGB output is requested, or when both
    // HSV and at least one u8 RGB/RGBA output are requested simultaneously.
    let need_rgb_kernel = want_rgb || (want_hsv && want_rgba);

    // Standalone RGBA fast path (no RGB or HSV output needed).
    if want_rgba && !need_rgb_kernel && !want_hsv {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      gray16_to_rgba_row::<BE>(y_plane, rgba_row, w, use_simd, full_range);
      return Ok(());
    }

    // Standalone HSV fast path — gray sources always have H=0, S=0, V=Y>>8.
    // Skip RGB scratch entirely when only HSV (and optionally RGBA) is needed.
    if want_hsv && !want_rgb {
      let hsv = hsv.as_mut().unwrap();
      let (hp, sp, vp) = hsv.hsv();
      gray16_to_hsv_row::<BE>(
        y_plane,
        &mut hp[one_plane_start..one_plane_end],
        &mut sp[one_plane_start..one_plane_end],
        &mut vp[one_plane_start..one_plane_end],
        w,
        use_simd,
        full_range,
      );
      if let Some(buf) = rgba.as_deref_mut() {
        let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
        gray16_to_rgba_row::<BE>(y_plane, rgba_row, w, use_simd, full_range);
      }
      return Ok(());
    }

    if !need_rgb_kernel {
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
    gray16_to_rgb_row::<BE>(y_plane, rgb_row, w, use_simd, full_range);

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
      let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
    }

    Ok(())
  }
}

/// Row-stage fused downscale for [`Gray16`]: the wire row converts to a
/// source-width **host-native** `u16` luma plane via the very kernel the
/// direct `luma_u16` path uses (`gray16_to_luma_u16_row::<BE>` with the
/// source wire `BE`), a single 1-channel `AreaStream<u16>` bins it at
/// u16 precision, then every attached output derives from each finalized
/// binned u16 luma row using the direct kernels. Because the binned row
/// is already host-native, those derive kernels run with
/// `HOST_NATIVE_BE = cfg!(target_endian = "big")` — `::<HOST_NATIVE_BE>`
/// is the identity recovery for an already-host-native sample, so on a BE
/// host the source→luma swap and the luma→output no-op are not double-
/// swapped. The result equals the direct Gray16 path run over a frame
/// that already holds the binned u16 luma. Atomic preflight: freeze,
/// sequence check, stream creation, and (for the colour group) source +
/// scratch growth all precede the first feed, so a failure mutates no
/// caller output.
#[allow(clippy::too_many_arguments)]
fn gray16_process_resampled<const BE: bool>(
  luma_stream_u16: &mut Option<std::boxed::Box<AreaStream<u16>>>,
  luma_filter_stream_u16: &mut Option<std::boxed::Box<crate::resample::FilterStream<u16>>>,
  luma_scratch_u16: &mut std::vec::Vec<u16>,
  resample_outputs: &mut Option<super::FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgb_u16: &mut Option<&mut [u16]>,
  rgba: &mut Option<&mut [u8]>,
  rgba_u16: &mut Option<&mut [u16]>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  hsv: &mut Option<mediaframe::source::HsvFrameMut<'_>>,
  y_row: &[u16],
  plan: &ResamplePlan,
  w: usize,
  idx: usize,
  use_simd: bool,
  full_range: bool,
) -> Result<(), MixedSinkerError> {
  // The binned u16 luma row is host-native; the direct kernels recover
  // an already-host-native sample with `::<HOST_NATIVE_BE>` (a no-op
  // swap), matching the direct path's `::<BE>` applied to a wire sample.
  // Gray16 is full 16-bit (native max == u16 max), so a signed filter
  // kernel's overshoot is already clipped by the `FilterStream`'s
  // `0..=65535` clamp — no extra native-depth clamp is needed.
  const HOST_NATIVE_BE: bool = cfg!(target_endian = "big");
  let ow = plan.out_w();
  // Transactional 1-channel gray luma resample (RFC #238 tail-collapse): Gray16 *is*
  // a u16 luma plane — the wire row converts to a source-width host-native u16 luma
  // plane (staged in the emit's `None` pregrow), a single u16 stream resamples it,
  // and every output derives from each finalized u16 luma row via the direct kernels.
  super::gray_luma_resample::<u16>(
    luma_stream_u16,
    luma_filter_stream_u16,
    resample_outputs,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    luma,
    luma_u16,
    &mut None,
    &mut None,
    hsv,
    plan,
    idx,
    |stream, rgb, rgba, rgb_u16, rgba_u16, luma, luma_u16, _rgb_f32, _luma_f32, hsv| {
      let want_rgb = rgb.is_some();
      let want_rgb_u16 = rgb_u16.is_some();
      let want_rgba = rgba.is_some();
      let want_rgba_u16 = rgba_u16.is_some();
      let want_hsv = hsv.is_some();
      // The u8 RGB kernel runs only when RGB output is requested; HSV (with
      // or without RGBA) and standalone RGBA derive directly from luma, so
      // the resample path needs no RGB scratch.
      let need_rgb_kernel = want_rgb;
      // Pregrow the source-width host-native u16 luma staging BEFORE the freeze.
      let src_luma = source_luma_u16_scratch(luma_scratch_u16, w, plan)?;
      let Some(stream) = stream else {
        return Ok(());
      };
      // Convert the wire Gray16 row to host-native u16 luma — the source
      // wire `::<BE>`, the same kernel the direct `luma_u16` path uses.
      gray16_to_luma_u16_row::<BE>(y_row, src_luma, w, use_simd);
      let src_luma: &[u16] = src_luma;
      let mut emit = |oy: usize, binned_y: &[u16]| {
        // Luma u16 — host-native pass-through of the binned u16 luma.
        if let Some(buf) = luma_u16.as_deref_mut() {
          gray16_to_luma_u16_row::<HOST_NATIVE_BE>(
            binned_y,
            &mut buf[oy * ow..(oy + 1) * ow],
            ow,
            use_simd,
          );
        }
        // Luma u8 — `>> 8` narrowing of the binned u16 luma.
        if let Some(buf) = luma.as_deref_mut() {
          gray16_to_luma_row::<HOST_NATIVE_BE>(
            binned_y,
            &mut buf[oy * ow..(oy + 1) * ow],
            ow,
            use_simd,
          );
        }

        // u16 RGB / RGBA (Strategy A) — native 16-bit broadcast.
        if want_rgba_u16 && !want_rgb_u16 {
          let buf = rgba_u16.as_deref_mut().unwrap();
          gray16_to_rgba_u16_row::<HOST_NATIVE_BE>(
            binned_y,
            &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow],
            ow,
            use_simd,
            full_range,
          );
        } else if want_rgb_u16 {
          let buf = rgb_u16.as_deref_mut().unwrap();
          let rgb_u16_row = &mut buf[oy * 3 * ow..(oy + 1) * 3 * ow];
          gray16_to_rgb_u16_row::<HOST_NATIVE_BE>(binned_y, rgb_u16_row, ow, use_simd, full_range);
          if let Some(buf) = rgba_u16.as_deref_mut() {
            expand_rgb_u16_to_rgba_u16_row::<16>(
              rgb_u16_row,
              &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow],
              ow,
            );
          }
        }

        // Standalone u8 RGBA fast path — no RGB or HSV requested.
        if want_rgba && !need_rgb_kernel && !want_hsv {
          let buf = rgba.as_deref_mut().unwrap();
          gray16_to_rgba_row::<HOST_NATIVE_BE>(
            binned_y,
            &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow],
            ow,
            use_simd,
            full_range,
          );
          return;
        }

        // Standalone HSV fast path — H=0/S=0/V=Y>>8 with no RGB computation
        // (plus an optional standalone RGBA, matching the direct path).
        if want_hsv && !want_rgb {
          let hsv = hsv.as_mut().unwrap();
          let (hp, sp, vp) = hsv.hsv();
          gray16_to_hsv_row::<HOST_NATIVE_BE>(
            binned_y,
            &mut hp[oy * ow..(oy + 1) * ow],
            &mut sp[oy * ow..(oy + 1) * ow],
            &mut vp[oy * ow..(oy + 1) * ow],
            ow,
            use_simd,
            full_range,
          );
          if let Some(buf) = rgba.as_deref_mut() {
            gray16_to_rgba_row::<HOST_NATIVE_BE>(
              binned_y,
              &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow],
              ow,
              use_simd,
              full_range,
            );
          }
          return;
        }

        if !need_rgb_kernel {
          return;
        }

        // Reached only when RGB is attached (need_rgb_kernel == want_rgb), so
        // the kernel writes the user buffer; HSV-from-RGB and the RGBA
        // fan-out follow, exactly as the direct path's RGB-kernel branch does.
        let buf = rgb
          .as_deref_mut()
          .expect("need_rgb_kernel implies RGB is attached");
        let rgb_row = &mut buf[oy * 3 * ow..(oy + 1) * 3 * ow];
        gray16_to_rgb_row::<HOST_NATIVE_BE>(binned_y, rgb_row, ow, use_simd, full_range);
        if let Some(hsv) = hsv.as_mut() {
          let (hp, sp, vp) = hsv.hsv();
          rgb_to_hsv_row(
            rgb_row,
            &mut hp[oy * ow..(oy + 1) * ow],
            &mut sp[oy * ow..(oy + 1) * ow],
            &mut vp[oy * ow..(oy + 1) * ow],
            ow,
            use_simd,
          );
        }
        if let Some(buf) = rgba.as_deref_mut() {
          expand_rgb_to_rgba_row(rgb_row, &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow], ow);
        }
      };
      // The per-output fan-out is identical for both engines, so feed it to
      // whichever stream the span kind selected; only one runs per frame.
      match stream {
        super::GrayLumaStreamMut::Area(s) => s.feed_row(idx, src_luma, use_simd, &mut emit),
        super::GrayLumaStreamMut::Filter(s) => s.feed_row(idx, src_luma, use_simd, &mut emit),
      }?;
      Ok(())
    },
  )
}

// ---- Gray32 impl ------------------------------------------------------------

impl<'a, R, const BE: bool> MixedSinker<'a, Gray32<BE>, R> {
  /// Attaches an 8-bit RGBA output buffer. Alpha is forced to `0xFF`.
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

  /// Attaches a u16 RGB output buffer (native `>> 16` broadcast — the widest
  /// depth colconv emits). Length in `u16` elements (`width x height x 3`).
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

  /// Attaches a u16 RGBA output buffer (native `>> 16` broadcast; alpha
  /// = `0xFFFF`). Length in `u16` elements (`width x height x 4`).
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

  /// Attaches a u16 luma output buffer (native `>> 16` narrow of the Gray32 Y
  /// plane). Length in `u16` elements (`width x height`).
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

impl<R, const BE: bool> Gray32Sink<BE> for MixedSinker<'_, Gray32<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, Gray32<BE>, R> {
  type Input<'r> = Gray32Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    // New frame: restart the u32 luma stream(s) (lazily created in
    // `process`) and clear the frozen output snapshot so a reused resampling
    // sink re-sequences from row 0.
    if let Some(stream) = self.luma_stream_u32.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.luma_filter_stream_u32.as_mut() {
      stream.reset();
    }
    self.resample_outputs = None;
    Ok(())
  }

  fn process(&mut self, row: Gray32Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;
    let full_range = row.full_range();

    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Y,
        idx,
        w,
        row.y().len(),
      )));
    }
    if idx >= h {
      return Err(MixedSinkerError::RowIndexOutOfRange(
        RowIndexOutOfRange::new(idx, h),
      ));
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
      plan,
      luma_stream_u32,
      luma_filter_stream_u32,
      luma_scratch_u32,
      resample_outputs,
      ..
    } = self;

    // Non-identity plan: Gray32 *is* a u32 luma plane, so the wire row
    // converts to a source-width host-native u32 luma plane (the wire `BE`
    // swap only — no depth narrow), a single 1-channel stream resamples it at
    // native u32 precision, then every attached output derives from each
    // finalized u32 luma row (narrowing only afterwards) exactly as the direct
    // Gray32 path does. Binning at u32 and narrowing after is 0-ULP for both
    // ranges (closes issue #289). The span kind picks the engine (area bin or
    // signed-coefficient filter). Row-stage only.
    if let Some(plan) = plan.as_ref() {
      return gray32_process_resampled::<BE>(
        luma_stream_u32,
        luma_filter_stream_u32,
        luma_scratch_u32,
        resample_outputs,
        rgb,
        rgb_u16,
        rgba,
        rgba_u16,
        luma,
        luma_u16,
        hsv,
        row.y(),
        plan,
        w,
        idx,
        use_simd,
        full_range,
      );
    }

    let y_plane = row.y();
    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // Luma u8 — shift >> 24.
    if let Some(buf) = luma.as_deref_mut() {
      gray32_to_luma_row::<BE>(
        y_plane,
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    // Luma u16 — shift >> 16 (native depth).
    if let Some(buf) = luma_u16.as_deref_mut() {
      gray32_to_luma_u16_row::<BE>(
        y_plane,
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    // u16 RGB / RGBA path (Strategy A).
    let want_rgb_u16 = rgb_u16.is_some();
    let want_rgba_u16 = rgba_u16.is_some();

    if want_rgba_u16 && !want_rgb_u16 {
      let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row =
        rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
      gray32_to_rgba_u16_row::<BE>(y_plane, rgba_u16_row, w, use_simd, full_range);
    } else if want_rgb_u16 {
      let rgb_u16_buf = rgb_u16.as_deref_mut().unwrap();
      let rgb_plane_start = one_plane_start * 3;
      let rgb_plane_end =
        one_plane_end
          .checked_mul(3)
          .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
            w, h, 3,
          )))?;
      let rgb_u16_row = &mut rgb_u16_buf[rgb_plane_start..rgb_plane_end];
      gray32_to_rgb_u16_row::<BE>(y_plane, rgb_u16_row, w, use_simd, full_range);
      if want_rgba_u16 {
        let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
        let rgba_u16_row =
          rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
        expand_rgb_u16_to_rgba_u16_row::<16>(rgb_u16_row, rgba_u16_row, w);
      }
    }

    // u8 RGB / RGBA / HSV (Strategy A).
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();
    // Only need the RGB kernel when an RGB output is requested, or when both
    // HSV and at least one u8 RGB/RGBA output are requested simultaneously.
    let need_rgb_kernel = want_rgb || (want_hsv && want_rgba);

    // Standalone RGBA fast path (no RGB or HSV output needed).
    if want_rgba && !need_rgb_kernel && !want_hsv {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      gray32_to_rgba_row::<BE>(y_plane, rgba_row, w, use_simd, full_range);
      return Ok(());
    }

    // Standalone HSV fast path — gray sources always have H=0, S=0, V=Y>>24.
    // Skip RGB scratch entirely when only HSV (and optionally RGBA) is needed.
    if want_hsv && !want_rgb {
      let hsv = hsv.as_mut().unwrap();
      let (hp, sp, vp) = hsv.hsv();
      gray32_to_hsv_row::<BE>(
        y_plane,
        &mut hp[one_plane_start..one_plane_end],
        &mut sp[one_plane_start..one_plane_end],
        &mut vp[one_plane_start..one_plane_end],
        w,
        use_simd,
        full_range,
      );
      if let Some(buf) = rgba.as_deref_mut() {
        let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
        gray32_to_rgba_row::<BE>(y_plane, rgba_row, w, use_simd, full_range);
      }
      return Ok(());
    }

    if !need_rgb_kernel {
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
    gray32_to_rgb_row::<BE>(y_plane, rgb_row, w, use_simd, full_range);

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
      let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
    }

    Ok(())
  }
}

/// Row-stage fused downscale for [`Gray32`]: the wire row converts to a
/// source-width **host-native** `u32` luma plane via
/// `gray32_to_native_u32_row::<BE>` (the wire `BE` swap only, **no depth
/// narrow**), a single 1-channel `AreaStream<u32>` / `FilterStream<u32>`
/// resamples it at **native `u32` precision**, then every attached output
/// derives from each finalized `u32` luma row using the very `gray32_to_*`
/// kernels the direct path uses (`luma_u16 = >> 16`, `luma = >> 24`, native
/// broadcast / colorimetry on the full `u32`). Binning the full `u32` and
/// narrowing only **after** is exact — the resample is byte-identical to the
/// direct Gray32 sink run over a frame already holding the `u32`-binned luma,
/// for **both** `full_range` true and false (closes issue #289; the prior
/// `>> 16`-narrow-first staging was ≤1 LSB off the exact `u32`-domain mean at
/// limited range). Because the resampled row is already host-native, those
/// derive kernels run with `HOST_NATIVE_BE = cfg!(target_endian = "big")` —
/// the identity recovery for an already-host-native sample, so on a BE host
/// the source→native conversion and the derive are not double-swapped. The
/// `FilterStream<u32>`'s `0..=u32::MAX` clamp is the full native range, so a
/// signed filter overshoot is already clipped — no extra clamp is needed.
/// Atomic preflight: freeze, sequence check, stream creation, and source
/// staging all precede the first feed, so a failure mutates no caller output.
#[allow(clippy::too_many_arguments)]
fn gray32_process_resampled<const BE: bool>(
  luma_stream_u32: &mut Option<std::boxed::Box<AreaStream<u32>>>,
  luma_filter_stream_u32: &mut Option<std::boxed::Box<crate::resample::FilterStream<u32>>>,
  luma_scratch_u32: &mut std::vec::Vec<u32>,
  resample_outputs: &mut Option<super::FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgb_u16: &mut Option<&mut [u16]>,
  rgba: &mut Option<&mut [u8]>,
  rgba_u16: &mut Option<&mut [u16]>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  hsv: &mut Option<mediaframe::source::HsvFrameMut<'_>>,
  y_row: &[u32],
  plan: &ResamplePlan,
  w: usize,
  idx: usize,
  use_simd: bool,
  full_range: bool,
) -> Result<(), MixedSinkerError> {
  // The resampled luma row is host-native `u32`; the `gray32_to_*` derive
  // kernels recover an already-host-native sample with `::<HOST_NATIVE_BE>`
  // (a no-op swap), matching the direct path's `::<BE>` applied to a wire
  // sample. Binning at native `u32` and narrowing only in the derive is
  // 0-ULP versus the direct sink over the binned luma; a signed filter
  // overshoot is already clipped by the `FilterStream<u32>`'s `0..=u32::MAX`
  // clamp (the full native range), so no extra clamp is needed.
  const HOST_NATIVE_BE: bool = cfg!(target_endian = "big");
  let ow = plan.out_w();
  // Transactional 1-channel gray luma resample (RFC #238 tail-collapse): Gray32 *is*
  // a u32 luma plane — the wire row converts to a source-width host-native u32 luma
  // plane (staged in the emit's `None` pregrow), a single u32 stream resamples it at
  // full precision, and every output narrows only in the direct-path derive kernels.
  super::gray_luma_resample::<u32>(
    luma_stream_u32,
    luma_filter_stream_u32,
    resample_outputs,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    luma,
    luma_u16,
    &mut None,
    &mut None,
    hsv,
    plan,
    idx,
    |stream, rgb, rgba, rgb_u16, rgba_u16, luma, luma_u16, _rgb_f32, _luma_f32, hsv| {
      let want_rgb = rgb.is_some();
      let want_rgb_u16 = rgb_u16.is_some();
      let want_rgba = rgba.is_some();
      let want_rgba_u16 = rgba_u16.is_some();
      let want_hsv = hsv.is_some();
      // The u8 RGB kernel runs only when RGB output is requested; HSV (with
      // or without RGBA) and standalone RGBA derive directly from luma, so
      // the resample path needs no RGB scratch.
      let need_rgb_kernel = want_rgb;
      // Pregrow the source-width host-native u32 luma staging BEFORE the freeze.
      let src_luma = source_luma_u32_scratch(luma_scratch_u32, w, plan)?;
      let Some(stream) = stream else {
        return Ok(());
      };
      // Convert the wire Gray32 row to the host-native u32 value (the wire `BE`
      // swap only — NO depth narrow), so binning runs at full u32 precision and
      // each output narrows only afterwards. Binning the full u32 then narrowing
      // is byte-identical to the direct Gray32 sink over the u32-binned luma, for
      // both full_range true and false (closes issue #289 — the prior
      // `>> 16`-narrow-first staging dropped the low 16 bits ahead of the affine,
      // ≤1 LSB off the exact u32-domain mean).
      crate::row::scalar::gray::gray32_to_native_u32_row::<BE>(y_row, src_luma, w);
      let src_luma: &[u32] = src_luma;
      let mut emit = |oy: usize, binned_y: &[u32]| {
        // Luma u16 — `>> 16` native narrow of the binned u32 luma.
        if let Some(buf) = luma_u16.as_deref_mut() {
          gray32_to_luma_u16_row::<HOST_NATIVE_BE>(
            binned_y,
            &mut buf[oy * ow..(oy + 1) * ow],
            ow,
            use_simd,
          );
        }
        // Luma u8 — `>> 24` narrow of the binned u32 luma.
        if let Some(buf) = luma.as_deref_mut() {
          gray32_to_luma_row::<HOST_NATIVE_BE>(
            binned_y,
            &mut buf[oy * ow..(oy + 1) * ow],
            ow,
            use_simd,
          );
        }

        // u16 RGB / RGBA (Strategy A) — native `>> 16` broadcast of the binned u32.
        if want_rgba_u16 && !want_rgb_u16 {
          let buf = rgba_u16.as_deref_mut().unwrap();
          gray32_to_rgba_u16_row::<HOST_NATIVE_BE>(
            binned_y,
            &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow],
            ow,
            use_simd,
            full_range,
          );
        } else if want_rgb_u16 {
          let buf = rgb_u16.as_deref_mut().unwrap();
          let rgb_u16_row = &mut buf[oy * 3 * ow..(oy + 1) * 3 * ow];
          gray32_to_rgb_u16_row::<HOST_NATIVE_BE>(binned_y, rgb_u16_row, ow, use_simd, full_range);
          if let Some(buf) = rgba_u16.as_deref_mut() {
            expand_rgb_u16_to_rgba_u16_row::<16>(
              rgb_u16_row,
              &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow],
              ow,
            );
          }
        }

        // Standalone u8 RGBA fast path — no RGB or HSV requested.
        if want_rgba && !need_rgb_kernel && !want_hsv {
          let buf = rgba.as_deref_mut().unwrap();
          gray32_to_rgba_row::<HOST_NATIVE_BE>(
            binned_y,
            &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow],
            ow,
            use_simd,
            full_range,
          );
          return;
        }

        // Standalone HSV fast path — H=0/S=0/V=Y>>24 with no RGB computation
        // (plus an optional standalone RGBA, matching the direct path).
        if want_hsv && !want_rgb {
          let hsv = hsv.as_mut().unwrap();
          let (hp, sp, vp) = hsv.hsv();
          gray32_to_hsv_row::<HOST_NATIVE_BE>(
            binned_y,
            &mut hp[oy * ow..(oy + 1) * ow],
            &mut sp[oy * ow..(oy + 1) * ow],
            &mut vp[oy * ow..(oy + 1) * ow],
            ow,
            use_simd,
            full_range,
          );
          if let Some(buf) = rgba.as_deref_mut() {
            gray32_to_rgba_row::<HOST_NATIVE_BE>(
              binned_y,
              &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow],
              ow,
              use_simd,
              full_range,
            );
          }
          return;
        }

        if !need_rgb_kernel {
          return;
        }

        // Reached only when RGB is attached (need_rgb_kernel == want_rgb), so
        // the kernel writes the user buffer; HSV-from-RGB and the RGBA
        // fan-out follow, exactly as the direct path's RGB-kernel branch does.
        let buf = rgb
          .as_deref_mut()
          .expect("need_rgb_kernel implies RGB is attached");
        let rgb_row = &mut buf[oy * 3 * ow..(oy + 1) * 3 * ow];
        gray32_to_rgb_row::<HOST_NATIVE_BE>(binned_y, rgb_row, ow, use_simd, full_range);
        if let Some(hsv) = hsv.as_mut() {
          let (hp, sp, vp) = hsv.hsv();
          rgb_to_hsv_row(
            rgb_row,
            &mut hp[oy * ow..(oy + 1) * ow],
            &mut sp[oy * ow..(oy + 1) * ow],
            &mut vp[oy * ow..(oy + 1) * ow],
            ow,
            use_simd,
          );
        }
        if let Some(buf) = rgba.as_deref_mut() {
          expand_rgb_to_rgba_row(rgb_row, &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow], ow);
        }
      };
      // The per-output fan-out is identical for both engines, so feed it to
      // whichever stream the span kind selected; only one runs per frame.
      match stream {
        super::GrayLumaStreamMut::Area(s) => s.feed_row(idx, src_luma, use_simd, &mut emit),
        super::GrayLumaStreamMut::Filter(s) => s.feed_row(idx, src_luma, use_simd, &mut emit),
      }?;
      Ok(())
    },
  )
}

// ---- Grayf32 impl -----------------------------------------------------------

impl<'a, R, const BE: bool> MixedSinker<'a, Grayf32<BE>, R> {
  /// Attaches an 8-bit RGBA output buffer. α is forced to `0xFF`
  /// (Grayf32 has no alpha channel).
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

  /// Attaches a u16 RGB output buffer.
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

  /// Attaches a u16 RGBA output buffer. α = `0xFFFF`.
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

  /// Attaches a u16 luma output buffer (`clamp(Y,0,1) x 65535`).
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

  /// Attaches a packed f32 RGB output buffer. Lossless replicate of Y → R=G=B.
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

  /// Attaches an f32 luma output buffer. Lossless pass-through of Y.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_luma_f32(mut self, buf: &'a mut [f32]) -> Result<Self, MixedSinkerError> {
    self.set_luma_f32(buf)?;
    Ok(self)
  }
  /// In-place variant of [`with_luma_f32`](Self::with_luma_f32).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_luma_f32(&mut self, buf: &'a mut [f32]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_elems(1)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::InsufficientLumaF32Buffer(
        InsufficientBuffer::new(expected, buf.len()),
      ));
    }
    self.luma_f32 = Some(buf);
    Ok(self)
  }
}

impl<R, const BE: bool> Grayf32Sink<BE> for MixedSinker<'_, Grayf32<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, Grayf32<BE>, R> {
  type Input<'r> = Grayf32Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    // New frame: restart the f32 luma stream(s) (lazily created in
    // `process`) and clear the frozen output snapshot, mirroring the
    // Gray8 / Gray16 paths so a reused resampling sink re-sequences from
    // row 0.
    if let Some(stream) = self.luma_stream_f32.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.luma_filter_stream_f32.as_mut() {
      stream.reset();
    }
    self.resample_outputs = None;
    Ok(())
  }

  fn process(&mut self, row: Grayf32Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Y,
        idx,
        w,
        row.y().len(),
      )));
    }
    if idx >= h {
      return Err(MixedSinkerError::RowIndexOutOfRange(
        RowIndexOutOfRange::new(idx, h),
      ));
    }

    let Self {
      rgb,
      rgb_u16,
      rgba,
      rgba_u16,
      luma,
      luma_u16,
      rgb_f32,
      luma_f32,
      hsv,
      rgb_scratch,
      plan,
      luma_stream_f32,
      luma_filter_stream_f32,
      luma_scratch_f32,
      resample_outputs,
      ..
    } = self;

    // Non-identity plan: Grayf32 *is* an f32 luma plane, so the wire row
    // converts to a source-width host-native f32 luma plane (the same
    // kernel the direct `luma_f32` path uses), a single 1-channel f32
    // stream bins it at f32 precision, then every attached output derives
    // from each finalized binned f32 luma row exactly as the direct path
    // does below. The span kind picks the engine (area or filter).
    if let Some(plan) = plan.as_ref() {
      return grayf32_process_resampled::<BE>(
        luma_stream_f32,
        luma_filter_stream_f32,
        luma_scratch_f32,
        resample_outputs,
        rgb,
        rgb_u16,
        rgba,
        rgba_u16,
        luma,
        luma_u16,
        rgb_f32,
        luma_f32,
        hsv,
        row.y(),
        plan,
        w,
        idx,
        use_simd,
      );
    }

    let y_plane = row.y();
    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // luma f32 pass-through — highest priority (no clamp, no round).
    if let Some(buf) = luma_f32.as_deref_mut() {
      grayf32_to_luma_f32_row::<BE>(
        y_plane,
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    // rgb_f32 — lossless replicate Y → R=G=B.
    if let Some(buf) = rgb_f32.as_deref_mut() {
      let rgb_f32_start = one_plane_start * 3;
      let rgb_f32_end = one_plane_end
        .checked_mul(3)
        .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
          w, h, 3,
        )))?;
      grayf32_to_rgb_f32_row::<BE>(y_plane, &mut buf[rgb_f32_start..rgb_f32_end], w, use_simd);
    }

    // luma u8.
    if let Some(buf) = luma.as_deref_mut() {
      grayf32_to_luma_row::<BE>(
        y_plane,
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    // luma u16.
    if let Some(buf) = luma_u16.as_deref_mut() {
      grayf32_to_luma_u16_row::<BE>(
        y_plane,
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    // u16 RGB / RGBA path (Strategy A).
    let want_rgb_u16 = rgb_u16.is_some();
    let want_rgba_u16 = rgba_u16.is_some();

    if want_rgba_u16 && !want_rgb_u16 {
      let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row =
        rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
      grayf32_to_rgba_u16_row::<BE>(y_plane, rgba_u16_row, w, use_simd);
    } else if want_rgb_u16 {
      let rgb_u16_buf = rgb_u16.as_deref_mut().unwrap();
      let rgb_plane_start = one_plane_start * 3;
      let rgb_plane_end =
        one_plane_end
          .checked_mul(3)
          .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
            w, h, 3,
          )))?;
      let rgb_u16_row = &mut rgb_u16_buf[rgb_plane_start..rgb_plane_end];
      grayf32_to_rgb_u16_row::<BE>(y_plane, rgb_u16_row, w, use_simd);
      if want_rgba_u16 {
        let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
        let rgba_u16_row =
          rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
        expand_rgb_u16_to_rgba_u16_row::<16>(rgb_u16_row, rgba_u16_row, w);
      }
    }

    // u8 RGB / RGBA / HSV path.
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();

    // Standalone RGBA fast path.
    if want_rgba && !want_rgb && !want_hsv {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      grayf32_to_rgba_row::<BE>(y_plane, rgba_row, w, use_simd);
      return Ok(());
    }

    // Standalone HSV fast path — Grayf32 always has H=0, S=0, V=clamp(Y)x255.
    if want_hsv && !want_rgb {
      let hsv = hsv.as_mut().unwrap();
      let (hp, sp, vp) = hsv.hsv();
      grayf32_to_hsv_row::<BE>(
        y_plane,
        &mut hp[one_plane_start..one_plane_end],
        &mut sp[one_plane_start..one_plane_end],
        &mut vp[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
      if let Some(buf) = rgba.as_deref_mut() {
        let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
        grayf32_to_rgba_row::<BE>(y_plane, rgba_row, w, use_simd);
      }
      return Ok(());
    }

    if !want_rgb && !want_rgba && !want_hsv {
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
    grayf32_to_rgb_row::<BE>(y_plane, rgb_row, w, use_simd);

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
      let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
    }

    Ok(())
  }
}

/// Row-stage fused downscale for [`Grayf32`]: the wire `Grayf32` row
/// converts to a source-width host-native `f32` luma plane (the same
/// kernel the direct `luma_f32` path uses), a single 1-channel
/// `AreaStream<f32>` bins it at f32 precision (Grayf32 *is* an f32 luma
/// plane — luma is not re-derived from RGB), then every attached output
/// derives from each finalized binned f32 luma row using the very
/// kernels the direct path uses, so a resampled output equals the direct
/// Grayf32 path run over a frame that already holds the binned f32 luma.
/// Atomic preflight: freeze, sequence check, source-luma staging, and
/// stream creation all precede the first feed, so a failure mutates no
/// caller output.
#[allow(clippy::too_many_arguments)]
fn grayf32_process_resampled<const BE: bool>(
  luma_stream_f32: &mut Option<std::boxed::Box<AreaStream<f32>>>,
  luma_filter_stream_f32: &mut Option<std::boxed::Box<crate::resample::FilterStream<f32>>>,
  luma_scratch_f32: &mut std::vec::Vec<f32>,
  resample_outputs: &mut Option<super::FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgb_u16: &mut Option<&mut [u16]>,
  rgba: &mut Option<&mut [u8]>,
  rgba_u16: &mut Option<&mut [u16]>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  rgb_f32: &mut Option<&mut [f32]>,
  luma_f32: &mut Option<&mut [f32]>,
  hsv: &mut Option<mediaframe::source::HsvFrameMut<'_>>,
  y_row: &[f32],
  plan: &ResamplePlan,
  w: usize,
  idx: usize,
  use_simd: bool,
) -> Result<(), MixedSinkerError> {
  // The binned f32 luma row is host-native; the direct kernels recover
  // an already-host-native sample with `::<HOST_NATIVE_BE>` (a no-op
  // swap for the lossless `luma_f32` / `rgb_f32` paths, and the correct
  // load for the clamp/scale integer paths), matching the direct path's
  // `::<BE>` applied to a wire sample. Passing `::<false>` here would
  // byte-swap every binned sample on a big-endian host.
  const HOST_NATIVE_BE: bool = cfg!(target_endian = "big");
  let ow = plan.out_w();
  // Transactional 1-channel gray luma resample (RFC #238 tail-collapse): Grayf32 *is*
  // an f32 luma plane — the wire row converts to a source-width host-native f32 luma
  // plane (staged in the emit's `None` pregrow), a single f32 stream bins it at f32
  // precision, and every output derives from each finalized f32 luma row.
  super::gray_luma_resample::<f32>(
    luma_stream_f32,
    luma_filter_stream_f32,
    resample_outputs,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    luma,
    luma_u16,
    rgb_f32,
    luma_f32,
    hsv,
    plan,
    idx,
    |stream, rgb, rgba, rgb_u16, rgba_u16, luma, luma_u16, rgb_f32, luma_f32, hsv| {
      let want_rgb = rgb.is_some();
      let want_rgb_u16 = rgb_u16.is_some();
      let want_rgba = rgba.is_some();
      let want_rgba_u16 = rgba_u16.is_some();
      let want_hsv = hsv.is_some();
      // The u8 RGB kernel runs only when RGB output is requested; HSV (with
      // or without RGBA) and standalone RGBA derive directly from luma, so
      // the resample path needs no RGB scratch.
      let need_rgb_kernel = want_rgb;
      // Pregrow the source-width host-native f32 luma staging BEFORE the freeze.
      if luma_scratch_f32.len() < w {
        luma_scratch_f32
          .try_reserve_exact(w - luma_scratch_f32.len())
          .map_err(|_| {
            MixedSinkerError::Resample(ResampleError::AllocationFailed(
              crate::resample::PlanGeometry::new(
                plan.src_w(),
                plan.src_h(),
                plan.out_w(),
                plan.out_h(),
              ),
            ))
          })?;
        luma_scratch_f32.resize(w, 0.0);
      }
      let Some(stream) = stream else {
        return Ok(());
      };
      // Convert the wire Grayf32 row to host-native f32 luma — the source
      // wire `::<BE>`, the same kernel the direct `luma_f32` path uses.
      let src_luma = &mut luma_scratch_f32[..w];
      grayf32_to_luma_f32_row::<BE>(y_row, src_luma, w, use_simd);
      let src_luma: &[f32] = src_luma;
      let mut emit = |oy: usize, binned_y: &[f32]| {
        // luma f32 — host-native pass-through of the binned f32 luma.
        if let Some(buf) = luma_f32.as_deref_mut() {
          grayf32_to_luma_f32_row::<HOST_NATIVE_BE>(
            binned_y,
            &mut buf[oy * ow..(oy + 1) * ow],
            ow,
            use_simd,
          );
        }
        // rgb_f32 — lossless replicate of the binned f32 luma Y → R=G=B.
        if let Some(buf) = rgb_f32.as_deref_mut() {
          grayf32_to_rgb_f32_row::<HOST_NATIVE_BE>(
            binned_y,
            &mut buf[oy * 3 * ow..(oy + 1) * 3 * ow],
            ow,
            use_simd,
          );
        }
        // luma u8 — clamp [0,1] x 255 of the binned f32 luma.
        if let Some(buf) = luma.as_deref_mut() {
          grayf32_to_luma_row::<HOST_NATIVE_BE>(
            binned_y,
            &mut buf[oy * ow..(oy + 1) * ow],
            ow,
            use_simd,
          );
        }
        // luma u16 — clamp [0,1] x 65535 of the binned f32 luma.
        if let Some(buf) = luma_u16.as_deref_mut() {
          grayf32_to_luma_u16_row::<HOST_NATIVE_BE>(
            binned_y,
            &mut buf[oy * ow..(oy + 1) * ow],
            ow,
            use_simd,
          );
        }

        // u16 RGB / RGBA (Strategy A) — clamp [0,1] x 65535 broadcast.
        if want_rgba_u16 && !want_rgb_u16 {
          let buf = rgba_u16.as_deref_mut().unwrap();
          grayf32_to_rgba_u16_row::<HOST_NATIVE_BE>(
            binned_y,
            &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow],
            ow,
            use_simd,
          );
        } else if want_rgb_u16 {
          let buf = rgb_u16.as_deref_mut().unwrap();
          let rgb_u16_row = &mut buf[oy * 3 * ow..(oy + 1) * 3 * ow];
          grayf32_to_rgb_u16_row::<HOST_NATIVE_BE>(binned_y, rgb_u16_row, ow, use_simd);
          if let Some(buf) = rgba_u16.as_deref_mut() {
            expand_rgb_u16_to_rgba_u16_row::<16>(
              rgb_u16_row,
              &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow],
              ow,
            );
          }
        }

        // Standalone u8 RGBA fast path — no RGB or HSV requested.
        if want_rgba && !need_rgb_kernel && !want_hsv {
          let buf = rgba.as_deref_mut().unwrap();
          grayf32_to_rgba_row::<HOST_NATIVE_BE>(
            binned_y,
            &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow],
            ow,
            use_simd,
          );
          return;
        }

        // Standalone HSV fast path — H=0/S=0/V=clamp(Y)x255 with no RGB
        // computation (plus an optional standalone RGBA, matching the direct
        // path).
        if want_hsv && !want_rgb {
          let hsv = hsv.as_mut().unwrap();
          let (hp, sp, vp) = hsv.hsv();
          grayf32_to_hsv_row::<HOST_NATIVE_BE>(
            binned_y,
            &mut hp[oy * ow..(oy + 1) * ow],
            &mut sp[oy * ow..(oy + 1) * ow],
            &mut vp[oy * ow..(oy + 1) * ow],
            ow,
            use_simd,
          );
          if let Some(buf) = rgba.as_deref_mut() {
            grayf32_to_rgba_row::<HOST_NATIVE_BE>(
              binned_y,
              &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow],
              ow,
              use_simd,
            );
          }
          return;
        }

        if !need_rgb_kernel {
          return;
        }

        // Reached only when RGB is attached (need_rgb_kernel == want_rgb), so
        // the kernel writes the user buffer; HSV-from-RGB and the RGBA
        // fan-out follow, exactly as the direct path's RGB-kernel branch does.
        let buf = rgb
          .as_deref_mut()
          .expect("need_rgb_kernel implies RGB is attached");
        let rgb_row = &mut buf[oy * 3 * ow..(oy + 1) * 3 * ow];
        grayf32_to_rgb_row::<HOST_NATIVE_BE>(binned_y, rgb_row, ow, use_simd);
        if let Some(hsv) = hsv.as_mut() {
          let (hp, sp, vp) = hsv.hsv();
          rgb_to_hsv_row(
            rgb_row,
            &mut hp[oy * ow..(oy + 1) * ow],
            &mut sp[oy * ow..(oy + 1) * ow],
            &mut vp[oy * ow..(oy + 1) * ow],
            ow,
            use_simd,
          );
        }
        if let Some(buf) = rgba.as_deref_mut() {
          expand_rgb_to_rgba_row(rgb_row, &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow], ow);
        }
      };
      // The per-output fan-out is identical for both engines, so feed it to
      // whichever stream the span kind selected; only one runs per frame.
      match stream {
        super::GrayLumaStreamMut::Area(s) => s.feed_row(idx, src_luma, use_simd, &mut emit),
        super::GrayLumaStreamMut::Filter(s) => s.feed_row(idx, src_luma, use_simd, &mut emit),
      }?;
      Ok(())
    },
  )
}

// ---- Grayf16 impl -----------------------------------------------------------

impl<'a, R, const BE: bool> MixedSinker<'a, Grayf16<BE>, R> {
  /// Attaches an 8-bit RGBA output buffer. α is forced to `0xFF`
  /// (Grayf16 has no alpha channel).
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

  /// Attaches a u16 RGB output buffer.
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

  /// Attaches a u16 RGBA output buffer. α = `0xFFFF`.
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

  /// Attaches a u16 luma output buffer (`clamp(Y,0,1) x 65535`).
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

  /// Attaches a packed f32 RGB output buffer. Lossless widen f16 → f32 then
  /// replicate Y → R=G=B.
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

  /// Attaches an f32 luma output buffer. Lossless widen f16 → f32 of Y.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_luma_f32(mut self, buf: &'a mut [f32]) -> Result<Self, MixedSinkerError> {
    self.set_luma_f32(buf)?;
    Ok(self)
  }
  /// In-place variant of [`with_luma_f32`](Self::with_luma_f32).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_luma_f32(&mut self, buf: &'a mut [f32]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_elems(1)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::InsufficientLumaF32Buffer(
        InsufficientBuffer::new(expected, buf.len()),
      ));
    }
    self.luma_f32 = Some(buf);
    Ok(self)
  }
}

impl<R, const BE: bool> Grayf16Sink<BE> for MixedSinker<'_, Grayf16<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, Grayf16<BE>, R> {
  type Input<'r> = Grayf16Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    // New frame: restart the f32 luma stream(s) (lazily created in `process`)
    // and clear the frozen output snapshot, mirroring the Grayf32 path so a
    // reused resampling sink re-sequences from row 0.
    if let Some(stream) = self.luma_stream_f32.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.luma_filter_stream_f32.as_mut() {
      stream.reset();
    }
    self.resample_outputs = None;
    Ok(())
  }

  fn process(&mut self, row: Grayf16Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Y,
        idx,
        w,
        row.y().len(),
      )));
    }
    if idx >= h {
      return Err(MixedSinkerError::RowIndexOutOfRange(
        RowIndexOutOfRange::new(idx, h),
      ));
    }

    let Self {
      rgb,
      rgb_u16,
      rgba,
      rgba_u16,
      luma,
      luma_u16,
      rgb_f32,
      luma_f32,
      hsv,
      rgb_scratch,
      plan,
      luma_stream_f32,
      luma_filter_stream_f32,
      luma_scratch_f32,
      resample_outputs,
      ..
    } = self;

    // Non-identity plan: Grayf16 *is* an f16 luma plane, so the wire row widens
    // to a source-width host-native f32 luma plane (the same kernel the direct
    // `luma_f32` path uses), a single 1-channel f32 stream bins it at f32
    // precision, then every attached output derives from each finalized binned
    // f32 luma row exactly as the direct path does below. The span kind picks
    // the engine (area or filter).
    if let Some(plan) = plan.as_ref() {
      return grayf16_process_resampled::<BE>(
        luma_stream_f32,
        luma_filter_stream_f32,
        luma_scratch_f32,
        resample_outputs,
        rgb,
        rgb_u16,
        rgba,
        rgba_u16,
        luma,
        luma_u16,
        rgb_f32,
        luma_f32,
        hsv,
        row.y(),
        plan,
        w,
        idx,
        use_simd,
      );
    }

    let y_plane = row.y();
    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // luma f32 — lossless widen f16 → f32 (no clamp, no round).
    if let Some(buf) = luma_f32.as_deref_mut() {
      grayf16_to_luma_f32_row::<BE>(
        y_plane,
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    // rgb_f32 — lossless widen then replicate Y → R=G=B.
    if let Some(buf) = rgb_f32.as_deref_mut() {
      let rgb_f32_start = one_plane_start * 3;
      let rgb_f32_end = one_plane_end
        .checked_mul(3)
        .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
          w, h, 3,
        )))?;
      grayf16_to_rgb_f32_row::<BE>(y_plane, &mut buf[rgb_f32_start..rgb_f32_end], w, use_simd);
    }

    // luma u8.
    if let Some(buf) = luma.as_deref_mut() {
      grayf16_to_luma_row::<BE>(
        y_plane,
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    // luma u16.
    if let Some(buf) = luma_u16.as_deref_mut() {
      grayf16_to_luma_u16_row::<BE>(
        y_plane,
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    // u16 RGB / RGBA path (Strategy A).
    let want_rgb_u16 = rgb_u16.is_some();
    let want_rgba_u16 = rgba_u16.is_some();

    if want_rgba_u16 && !want_rgb_u16 {
      let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row =
        rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
      grayf16_to_rgba_u16_row::<BE>(y_plane, rgba_u16_row, w, use_simd);
    } else if want_rgb_u16 {
      let rgb_u16_buf = rgb_u16.as_deref_mut().unwrap();
      let rgb_plane_start = one_plane_start * 3;
      let rgb_plane_end =
        one_plane_end
          .checked_mul(3)
          .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
            w, h, 3,
          )))?;
      let rgb_u16_row = &mut rgb_u16_buf[rgb_plane_start..rgb_plane_end];
      grayf16_to_rgb_u16_row::<BE>(y_plane, rgb_u16_row, w, use_simd);
      if want_rgba_u16 {
        let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
        let rgba_u16_row =
          rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
        expand_rgb_u16_to_rgba_u16_row::<16>(rgb_u16_row, rgba_u16_row, w);
      }
    }

    // u8 RGB / RGBA / HSV path.
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();

    // Standalone RGBA fast path.
    if want_rgba && !want_rgb && !want_hsv {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      grayf16_to_rgba_row::<BE>(y_plane, rgba_row, w, use_simd);
      return Ok(());
    }

    // Standalone HSV fast path — Grayf16 always has H=0, S=0, V=clamp(Y)x255.
    if want_hsv && !want_rgb {
      let hsv = hsv.as_mut().unwrap();
      let (hp, sp, vp) = hsv.hsv();
      grayf16_to_hsv_row::<BE>(
        y_plane,
        &mut hp[one_plane_start..one_plane_end],
        &mut sp[one_plane_start..one_plane_end],
        &mut vp[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
      if let Some(buf) = rgba.as_deref_mut() {
        let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
        grayf16_to_rgba_row::<BE>(y_plane, rgba_row, w, use_simd);
      }
      return Ok(());
    }

    if !want_rgb && !want_rgba && !want_hsv {
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
    grayf16_to_rgb_row::<BE>(y_plane, rgb_row, w, use_simd);

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
      let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
    }

    Ok(())
  }
}

/// Row-stage fused downscale for [`Grayf16`]: the wire `Grayf16` row widens to a
/// source-width host-native `f32` luma plane (the same kernel the direct
/// `luma_f32` path uses), a single 1-channel `AreaStream<f32>` bins it at f32
/// precision (Grayf16 *is* an f16 luma plane — luma is not re-derived from
/// RGB), then every attached output derives from each finalized binned f32 luma
/// row using the very `grayf32` kernels the direct path's binned domain uses, so
/// a resampled output equals the direct Grayf16 path run over a frame that
/// already holds the binned f32 luma. Atomic preflight: freeze, sequence check,
/// source-luma staging, and stream creation all precede the first feed, so a
/// failure mutates no caller output.
#[allow(clippy::too_many_arguments)]
fn grayf16_process_resampled<const BE: bool>(
  luma_stream_f32: &mut Option<std::boxed::Box<AreaStream<f32>>>,
  luma_filter_stream_f32: &mut Option<std::boxed::Box<crate::resample::FilterStream<f32>>>,
  luma_scratch_f32: &mut std::vec::Vec<f32>,
  resample_outputs: &mut Option<super::FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgb_u16: &mut Option<&mut [u16]>,
  rgba: &mut Option<&mut [u8]>,
  rgba_u16: &mut Option<&mut [u16]>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  rgb_f32: &mut Option<&mut [f32]>,
  luma_f32: &mut Option<&mut [f32]>,
  hsv: &mut Option<mediaframe::source::HsvFrameMut<'_>>,
  y_row: &[half::f16],
  plan: &ResamplePlan,
  w: usize,
  idx: usize,
  use_simd: bool,
) -> Result<(), MixedSinkerError> {
  // The binned f32 luma row is host-native; the `grayf32` kernels recover an
  // already-host-native sample with `::<HOST_NATIVE_BE>` (a no-op swap for the
  // lossless `luma_f32` / `rgb_f32` paths, and the correct load for the
  // clamp/scale integer paths). Passing `::<false>` here would byte-swap every
  // binned sample on a big-endian host.
  const HOST_NATIVE_BE: bool = cfg!(target_endian = "big");
  let ow = plan.out_w();
  // Transactional 1-channel gray luma resample (RFC #238 tail-collapse): Grayf16 *is*
  // an f16 luma plane — the wire row widens to a source-width host-native f32 luma
  // plane (staged in the emit's `None` pregrow), a single f32 stream bins it at f32
  // precision, and every output derives from each finalized f32 luma row.
  super::gray_luma_resample::<f32>(
    luma_stream_f32,
    luma_filter_stream_f32,
    resample_outputs,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    luma,
    luma_u16,
    rgb_f32,
    luma_f32,
    hsv,
    plan,
    idx,
    |stream, rgb, rgba, rgb_u16, rgba_u16, luma, luma_u16, rgb_f32, luma_f32, hsv| {
      let want_rgb = rgb.is_some();
      let want_rgb_u16 = rgb_u16.is_some();
      let want_rgba = rgba.is_some();
      let want_rgba_u16 = rgba_u16.is_some();
      let want_hsv = hsv.is_some();
      let need_rgb_kernel = want_rgb;
      // Pregrow the source-width host-native f32 luma staging BEFORE the freeze.
      if luma_scratch_f32.len() < w {
        luma_scratch_f32
          .try_reserve_exact(w - luma_scratch_f32.len())
          .map_err(|_| {
            MixedSinkerError::Resample(ResampleError::AllocationFailed(
              crate::resample::PlanGeometry::new(
                plan.src_w(),
                plan.src_h(),
                plan.out_w(),
                plan.out_h(),
              ),
            ))
          })?;
        luma_scratch_f32.resize(w, 0.0);
      }
      let Some(stream) = stream else {
        return Ok(());
      };
      // Widen the wire Grayf16 row to host-native f32 luma — the source wire
      // `::<BE>`, the same kernel the direct `luma_f32` path uses.
      let src_luma = &mut luma_scratch_f32[..w];
      grayf16_to_luma_f32_row::<BE>(y_row, src_luma, w, use_simd);
      let src_luma: &[f32] = src_luma;
      let mut emit = |oy: usize, binned_y: &[f32]| {
        // luma f32 — host-native pass-through of the binned f32 luma.
        if let Some(buf) = luma_f32.as_deref_mut() {
          grayf32_to_luma_f32_row::<HOST_NATIVE_BE>(
            binned_y,
            &mut buf[oy * ow..(oy + 1) * ow],
            ow,
            use_simd,
          );
        }
        // rgb_f32 — lossless replicate of the binned f32 luma Y → R=G=B.
        if let Some(buf) = rgb_f32.as_deref_mut() {
          grayf32_to_rgb_f32_row::<HOST_NATIVE_BE>(
            binned_y,
            &mut buf[oy * 3 * ow..(oy + 1) * 3 * ow],
            ow,
            use_simd,
          );
        }
        // luma u8 — clamp [0,1] x 255 of the binned f32 luma.
        if let Some(buf) = luma.as_deref_mut() {
          grayf32_to_luma_row::<HOST_NATIVE_BE>(
            binned_y,
            &mut buf[oy * ow..(oy + 1) * ow],
            ow,
            use_simd,
          );
        }
        // luma u16 — clamp [0,1] x 65535 of the binned f32 luma.
        if let Some(buf) = luma_u16.as_deref_mut() {
          grayf32_to_luma_u16_row::<HOST_NATIVE_BE>(
            binned_y,
            &mut buf[oy * ow..(oy + 1) * ow],
            ow,
            use_simd,
          );
        }

        // u16 RGB / RGBA (Strategy A) — clamp [0,1] x 65535 broadcast.
        if want_rgba_u16 && !want_rgb_u16 {
          let buf = rgba_u16.as_deref_mut().unwrap();
          grayf32_to_rgba_u16_row::<HOST_NATIVE_BE>(
            binned_y,
            &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow],
            ow,
            use_simd,
          );
        } else if want_rgb_u16 {
          let buf = rgb_u16.as_deref_mut().unwrap();
          let rgb_u16_row = &mut buf[oy * 3 * ow..(oy + 1) * 3 * ow];
          grayf32_to_rgb_u16_row::<HOST_NATIVE_BE>(binned_y, rgb_u16_row, ow, use_simd);
          if let Some(buf) = rgba_u16.as_deref_mut() {
            expand_rgb_u16_to_rgba_u16_row::<16>(
              rgb_u16_row,
              &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow],
              ow,
            );
          }
        }

        // Standalone u8 RGBA fast path — no RGB or HSV requested.
        if want_rgba && !need_rgb_kernel && !want_hsv {
          let buf = rgba.as_deref_mut().unwrap();
          grayf32_to_rgba_row::<HOST_NATIVE_BE>(
            binned_y,
            &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow],
            ow,
            use_simd,
          );
          return;
        }

        // Standalone HSV fast path — H=0/S=0/V=clamp(Y)x255 with no RGB computation.
        if want_hsv && !want_rgb {
          let hsv = hsv.as_mut().unwrap();
          let (hp, sp, vp) = hsv.hsv();
          grayf32_to_hsv_row::<HOST_NATIVE_BE>(
            binned_y,
            &mut hp[oy * ow..(oy + 1) * ow],
            &mut sp[oy * ow..(oy + 1) * ow],
            &mut vp[oy * ow..(oy + 1) * ow],
            ow,
            use_simd,
          );
          if let Some(buf) = rgba.as_deref_mut() {
            grayf32_to_rgba_row::<HOST_NATIVE_BE>(
              binned_y,
              &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow],
              ow,
              use_simd,
            );
          }
          return;
        }

        if !need_rgb_kernel {
          return;
        }

        let buf = rgb
          .as_deref_mut()
          .expect("need_rgb_kernel implies RGB is attached");
        let rgb_row = &mut buf[oy * 3 * ow..(oy + 1) * 3 * ow];
        grayf32_to_rgb_row::<HOST_NATIVE_BE>(binned_y, rgb_row, ow, use_simd);
        if let Some(hsv) = hsv.as_mut() {
          let (hp, sp, vp) = hsv.hsv();
          rgb_to_hsv_row(
            rgb_row,
            &mut hp[oy * ow..(oy + 1) * ow],
            &mut sp[oy * ow..(oy + 1) * ow],
            &mut vp[oy * ow..(oy + 1) * ow],
            ow,
            use_simd,
          );
        }
        if let Some(buf) = rgba.as_deref_mut() {
          expand_rgb_to_rgba_row(rgb_row, &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow], ow);
        }
      };
      // The per-output fan-out is identical for both engines, so feed it to
      // whichever stream the span kind selected; only one runs per frame.
      match stream {
        super::GrayLumaStreamMut::Area(s) => s.feed_row(idx, src_luma, use_simd, &mut emit),
        super::GrayLumaStreamMut::Filter(s) => s.feed_row(idx, src_luma, use_simd, &mut emit),
      }?;
      Ok(())
    },
  )
}

// ---- Ya8 impl ---------------------------------------------------------------

impl<'a, R> MixedSinker<'a, Ya8, R> {
  /// Attaches an 8-bit RGBA output buffer. α is passed from the source.
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

  /// Attaches a u16 RGB output buffer.
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

  /// Attaches a u16 RGBA output buffer. α zero-extended from source.
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

  /// Attaches a u16 luma output buffer (zero-extend Y → u16).
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

impl<R> Ya8Sink for MixedSinker<'_, Ya8, R> {}

impl<R> PixelSink for MixedSinker<'_, Ya8, R> {
  type Input<'r> = Ya8Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    // New frame: restart the 4-channel RGBA colour stream and the independent
    // native-Y luma stream — area or filter kind (all lazily created in
    // `process`) — and re-arm the alpha-mode snapshot, mirroring the
    // alpha-aware packed-RGBA / Gbrap / `Vuya` sinks. The filter path also
    // uses the u16 colour stream (rgb_u16 / rgba_u16 filtered at native depth).
    if let Some(stream) = self.rgba_stream.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.luma_stream.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.rgba_filter_stream.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.rgba_filter_stream_u16.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.luma_filter_stream.as_mut() {
      stream.reset();
    }
    self.resample_outputs = None;
    self.frozen_alpha_mode = Some(self.alpha_mode);
    Ok(())
  }

  fn process(&mut self, row: Ya8Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;
    let packed = row.packed(); // &[u8], length = width * 2

    if packed.len() != w * 2 {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Y,
        idx,
        w * 2,
        packed.len(),
      )));
    }
    if idx >= h {
      return Err(MixedSinkerError::RowIndexOutOfRange(
        RowIndexOutOfRange::new(idx, h),
      ));
    }

    // Non-identity plan: `Ya8` is gray+alpha — structurally a degenerate
    // YUVA (`R = G = B = Y`, neutral chroma) plus an independent native-Y
    // luma. Decode each packed `[Y, A]` row into the canonical source-width
    // `R, G, B, A` row (`ya8_to_rgba_row`) and resample the four channels,
    // so binning yields `(binY, binY, binY, binA)` and a real area-mean /
    // filtered alpha; `luma` / `luma_u16` are an INDEPENDENT native-Y stream
    // over the de-interleaved Y (`ya8_to_luma_row` — the exact Y bytes the
    // direct path emits), NOT colour-derived (a limited-range `rgb_to_luma*`
    // would mis-map a valid `full_range = false` gray, and under
    // premultiplied the colour collapses to `mean(Y*A)/mean(A)` while native
    // Y stays `mean(Y)`).
    //
    // The span kind picks the engine. `Area` bins through the alpha-aware
    // packed-RGBA tail (`packed_rgba_resample::<true>`, `NATIVE_Y_LUMA`):
    // colour is binned premultiplied under `AlphaMode::Premultiplied`, luma
    // is a native-Y `AreaStream<u8>`, and rgb_u16 / rgba_u16 zero-extend the
    // binned u8 colour. `Filter` runs the signed-coefficient 4:4:4
    // packed-YUVA filter tail at `SRC_BITS = 8` with `NATIVE_LUMA_U8`: the
    // four u8 channels filter independently (PIL straight-alpha RGBA), the
    // native-depth `[Y, Y, Y, A]` filters through the shared u16 colour
    // stream for rgb_u16 / rgba_u16 (the degenerate-YUVA twin of `Ayuv64`'s
    // native u16 colour binning), and native-Y luma rides a `FilterStream<u8>`
    // (the 8bpc-grid stream `Gray8` uses — so `Ya8` luma is byte-identical to
    // `Gray8`'s for the same Y).
    // Premultiplied alpha has no filter analogue (the engine cannot
    // un-premultiply), so a premultiplied `Filter` plan routes to the area
    // tail with the filter plan, which surfaces the typed `UnsupportedFilter`.
    if self.plan.is_some() {
      let alpha_mode = self.alpha_mode;
      let matrix = row.matrix();
      let full_range = row.full_range();
      let Self {
        rgb,
        rgba,
        rgb_u16,
        rgba_u16,
        luma,
        luma_u16,
        hsv,
        rgba_scratch,
        rgb_scratch,
        luma_scratch,
        rgba_scratch_u16,
        rgba_color_scratch_u16,
        luma_scratch_u16,
        plan,
        rgba_stream,
        luma_stream,
        rgba_filter_stream,
        rgba_filter_stream_u16,
        luma_filter_stream,
        luma_filter_stream_u16,
        resample_outputs,
        frozen_alpha_mode,
        ..
      } = self;
      let plan = plan.as_ref().expect("plan.is_some() checked above");
      // Snapshotted at begin_frame; reject a mid-frame change before the
      // single binning / filtering route runs.
      check_frozen_alpha_mode(*frozen_alpha_mode, alpha_mode, idx)?;
      return match plan.kind() {
        crate::resample::SpanKind::Area => packed_rgba_resample::<true>(
          rgba_stream,
          luma_stream,
          resample_outputs,
          rgb,
          rgba,
          rgb_u16,
          rgba_u16,
          luma,
          luma_u16,
          hsv,
          rgba_scratch,
          rgb_scratch,
          luma_scratch,
          w,
          plan,
          idx,
          use_simd,
          alpha_mode,
          matrix,
          full_range,
          |dst| ya8_to_rgba_row(packed, dst, w, use_simd),
          |dst| ya8_to_luma_row(packed, dst, w, use_simd),
        ),
        crate::resample::SpanKind::Filter if alpha_mode.is_premultiplied() => {
          // Premultiplied + filter has no analogue: route to the area tail
          // with the filter plan so it returns the typed `UnsupportedFilter`.
          packed_rgba_resample::<true>(
            rgba_stream,
            luma_stream,
            resample_outputs,
            rgb,
            rgba,
            rgb_u16,
            rgba_u16,
            luma,
            luma_u16,
            hsv,
            rgba_scratch,
            rgb_scratch,
            luma_scratch,
            w,
            plan,
            idx,
            use_simd,
            alpha_mode,
            matrix,
            full_range,
            |dst| ya8_to_rgba_row(packed, dst, w, use_simd),
            |dst| ya8_to_luma_row(packed, dst, w, use_simd),
          )
        }
        // `ZEXT_U16_COLOR = true`: `Ya8` is an 8-bit source, so its
        // native-depth colour is the binned u8 colour zero-extended
        // (`rgba_u16 == rgba as u16`, `rgb_u16 == rgb as u16`) — byte-for-byte
        // the area path's contract ([`packed_rgba_resample`], which likewise
        // zero-extends, never an independent native-u16 bin). The u16 colour
        // outputs therefore ride the u8 colour stream and the independent
        // native-u16 stream is never created or fed (the
        // `convert_rgba_u16` closure below is dead).
        crate::resample::SpanKind::Filter => packed_yuva444_filter_resample::<8, true, true>(
          rgba_filter_stream,
          rgba_filter_stream_u16,
          luma_filter_stream,
          luma_filter_stream_u16,
          resample_outputs,
          rgb,
          rgba,
          rgb_u16,
          rgba_u16,
          luma,
          luma_u16,
          hsv,
          rgba_scratch,
          rgb_scratch,
          rgba_scratch_u16,
          rgba_color_scratch_u16,
          luma_scratch_u16,
          w,
          plan,
          idx,
          use_simd,
          // Packed `[Y, A]`: no contiguous Y plane, so the u8-luma path
          // de-interleaves native Y into `luma_scratch` (via the u8
          // closure below) rather than feeding a direct plane.
          &[],
          Some(luma_scratch),
          |dst| ya8_to_rgba_row(packed, dst, w, use_simd),
          // Dead under `ZEXT_U16_COLOR`: the u16 colour is the zero-extended
          // u8 colour (above), so no independent native-u16 colour stream is
          // built and this conversion is never invoked.
          |_dst: &mut [u16]| {},
          // u8-luma path: the u16 luma stream is detached, so this is never
          // called.
          |_dst: &mut [u16]| {},
          // Native-Y u8 de-interleave (the exact Y bytes the direct
          // `ya8_to_luma_row` emits) — parity with `Gray8`'s native-Y filter.
          |dst| ya8_to_luma_row(packed, dst, w, use_simd),
        ),
      };
    }

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // Output-mode flags — resolved BEFORE any output row is written so the
    // atomicity preflight below runs ahead of the luma writes.
    let want_rgb = self.rgb.is_some();
    let want_rgba = self.rgba.is_some();
    let want_hsv = self.hsv.is_some();

    // Atomicity preflight (#308, cf. the crate's #180 resample fix and the
    // planar / semi-planar / packed-YUV siblings): the only fallible row
    // scratch this path can grow is the RGB row buffer, and its allocating
    // (`None`) arm is reached for exactly one identity shape —
    // `want_hsv && want_rgba && !want_rgb` (every other `!want_rgb` case is
    // served by a standalone RGBA / HSV fast path below that never stages RGB).
    // Because native-Y luma, luma_u16 and the u16 RGB/RGBA rows are written
    // before that scratch, reserve it up front so an allocator refusal returns
    // a typed `AllocationFailed` leaving the output frame untouched rather than
    // partially mutated. The later staging call reuses the already-sized buffer
    // (default path byte-identical).
    if want_hsv && want_rgba && !want_rgb {
      rgb_row_buf_or_scratch(
        self.rgb.as_deref_mut(),
        &mut self.rgb_scratch,
        one_plane_start,
        one_plane_end,
        w,
        h,
      )?;
    }

    // luma u8.
    if let Some(buf) = self.luma.as_deref_mut() {
      ya8_to_luma_row(
        packed,
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    // luma u16.
    if let Some(buf) = self.luma_u16.as_deref_mut() {
      ya8_to_luma_u16_row(
        packed,
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    // u16 RGB / RGBA path. Each path is independent (α is embedded in ya8_to_rgba_u16_row).
    if let Some(buf) = self.rgb_u16.as_deref_mut() {
      let rgb_plane_start = one_plane_start * 3;
      let rgb_plane_end =
        one_plane_end
          .checked_mul(3)
          .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
            w, h, 3,
          )))?;
      ya8_to_rgb_u16_row(
        packed,
        &mut buf[rgb_plane_start..rgb_plane_end],
        w,
        use_simd,
      );
    }
    if let Some(buf) = self.rgba_u16.as_deref_mut() {
      let rgba_u16_row = rgba_u16_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      ya8_to_rgba_u16_row(packed, rgba_u16_row, w, use_simd);
    }

    // u8 RGB / RGBA / HSV path (flags hoisted above for the atomicity
    // preflight). Strategy A+: rgb first, then copy α into rgba.

    // Standalone RGBA fast path (no RGB or HSV).
    if want_rgba && !want_rgb && !want_hsv {
      let rgba_buf = self.rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      ya8_to_rgba_row(packed, rgba_row, w, use_simd);
      return Ok(());
    }

    // Standalone HSV fast path.
    if want_hsv && !want_rgb && !want_rgba {
      let hsv = self.hsv.as_mut().unwrap();
      let (h, s, v) = hsv.hsv();
      ya8_to_hsv_row(
        packed,
        &mut h[one_plane_start..one_plane_end],
        &mut s[one_plane_start..one_plane_end],
        &mut v[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
      return Ok(());
    }

    if !want_rgb && !want_rgba && !want_hsv {
      return Ok(());
    }

    // RGB kernel (used for HSV + Strategy A+ fan-out).
    let rgb_row = rgb_row_buf_or_scratch(
      self.rgb.as_deref_mut(),
      &mut self.rgb_scratch,
      one_plane_start,
      one_plane_end,
      w,
      h,
    )?;
    ya8_to_rgb_row(packed, rgb_row, w, use_simd);

    if let Some(hsv) = self.hsv.as_mut() {
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

    // Strategy A+: expand RGB→RGBA then patch α from source.
    if let Some(buf) = self.rgba.as_deref_mut() {
      let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
      // Overwrite the α channel with real source α.
      copy_alpha_ya_u8(packed, rgba_row, w);
    }

    Ok(())
  }
}

// ---- Ya16 impl --------------------------------------------------------------

impl<'a, R, const BE: bool> MixedSinker<'a, Ya16<BE>, R> {
  /// Attaches an 8-bit RGBA output buffer. α is `source_A >> 8`.
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

  /// Attaches a u16 RGB output buffer.
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

  /// Attaches a u16 RGBA output buffer. α from source (native u16).
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

  /// Attaches a u16 luma output buffer (native pass-through).
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

impl<R, const BE: bool> Ya16Sink<BE> for MixedSinker<'_, Ya16<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, Ya16<BE>, R> {
  type Input<'r> = Ya16Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    // New frame: restart the 4-channel u16 RGBA colour stream and the
    // independent native-Y u16 luma stream — area or filter kind (all lazily
    // created in `process`) — and re-arm the alpha-mode snapshot, mirroring
    // the high-bit packed-RGBA / Gbrap16 / `Vuya` sinks. The filter path also
    // uses the u8 colour stream (rgb / rgba from the `>> 8` narrowed RGBA).
    if let Some(stream) = self.rgba_stream_u16.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.luma_stream_u16.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.rgba_filter_stream.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.rgba_filter_stream_u16.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.luma_filter_stream_u16.as_mut() {
      stream.reset();
    }
    self.resample_outputs = None;
    self.frozen_alpha_mode = Some(self.alpha_mode);
    Ok(())
  }

  fn process(&mut self, row: Ya16Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;
    let packed = row.packed(); // &[u16], length = width * 2

    if packed.len() != w * 2 {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Y,
        idx,
        w * 2,
        packed.len(),
      )));
    }
    if idx >= h {
      return Err(MixedSinkerError::RowIndexOutOfRange(
        RowIndexOutOfRange::new(idx, h),
      ));
    }

    // Non-identity plan: `Ya16` is gray+alpha — the high-bit analogue of
    // `Ya8`, structurally a degenerate full-16-bit YUVA (`R = G = B = Y`,
    // neutral chroma, real straight alpha) plus an independent native-Y luma.
    // The direct `Ya16 -> rgba` conversion is `R = G = B = Y`, `A = α`; decode
    // each packed `[Y, A]` u16 row into the canonical host-native source-width
    // `R, G, B, A` u16 row (`ya16_to_rgba_u16_row::<BE>`) and resample the four
    // channels at native depth so resampled alpha is a real native mean.
    // luma / luma_u16 are an INDEPENDENT native-Y bin / filter over the
    // de-interleaved host-native Y (`ya16_to_luma_u16_row::<BE>` — the exact Y
    // the direct path emits), NOT colour-derived (byte-exact for every matrix,
    // unlike `rgb_to_luma_u16_native_row` which drifts for matrices whose Q15
    // weights do not sum to exactly 32768, e.g. SMPTE-240M; every range; AND
    // every alpha mode — under premultiplied the colour collapses to
    // `mean(Y*A)/mean(A)`, but native Y stays `mean(Y)`).
    //
    // The span kind picks the engine. `Area` bins through the high-bit
    // packed-RGBA tail (`packed_rgba_u16_resample::<16, false, true>`,
    // `NATIVE_Y_LUMA`): native u16 colour binned, the u8 colour narrowed from
    // it, luma a native-Y `AreaStream<u16>`. `Filter` runs the
    // signed-coefficient 4:4:4 packed-YUVA filter tail at `SRC_BITS = 16` with
    // `NATIVE_LUMA_U8 = false`: native u16 colour filters through the u16
    // colour stream (full 16-bit, so the `FilterStream`'s `0..=65535` clamp is
    // the native clamp), the u8 colour filters the `>> 8` narrowed RGBA, and
    // native-Y luma rides a `FilterStream<u16>` (the stream `Gray16` uses — so
    // `Ya16` luma_u16 is byte-identical to `Gray16`'s for the same Y).
    // Premultiplied alpha has no filter analogue (the engine cannot
    // un-premultiply), so a premultiplied `Filter` plan routes to the area
    // tail with the filter plan, which surfaces the typed `UnsupportedFilter`.
    if self.plan.is_some() {
      let alpha_mode = self.alpha_mode;
      let matrix = row.matrix();
      let full_range = row.full_range();
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
        rgba_scratch,
        rgba_scratch_u16,
        rgba_color_scratch_u16,
        luma_scratch_u16,
        rgba_stream_u16,
        luma_stream_u16,
        rgba_filter_stream,
        rgba_filter_stream_u16,
        luma_filter_stream,
        luma_filter_stream_u16,
        resample_outputs,
        frozen_alpha_mode,
        plan,
        ..
      } = self;
      let plan = plan.as_ref().expect("plan.is_some() checked above");
      check_frozen_alpha_mode(*frozen_alpha_mode, alpha_mode, idx)?;
      return match plan.kind() {
        crate::resample::SpanKind::Area => packed_rgba_u16_resample::<16, false, true>(
          rgba_stream_u16,
          luma_stream_u16,
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
          luma_scratch_u16,
          w,
          plan,
          idx,
          use_simd,
          alpha_mode,
          matrix,
          full_range,
          |dst| ya16_to_rgba_u16_row::<BE>(packed, dst, w, use_simd),
          |dst| ya16_to_luma_u16_row::<BE>(packed, dst, w, use_simd),
        ),
        crate::resample::SpanKind::Filter if alpha_mode.is_premultiplied() => {
          // Premultiplied + filter has no analogue: route to the area tail
          // with the filter plan so it returns the typed `UnsupportedFilter`.
          packed_rgba_u16_resample::<16, false, true>(
            rgba_stream_u16,
            luma_stream_u16,
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
            luma_scratch_u16,
            w,
            plan,
            idx,
            use_simd,
            alpha_mode,
            matrix,
            full_range,
            |dst| ya16_to_rgba_u16_row::<BE>(packed, dst, w, use_simd),
            |dst| ya16_to_luma_u16_row::<BE>(packed, dst, w, use_simd),
          )
        }
        crate::resample::SpanKind::Filter => packed_yuva444_filter_resample::<16, false, false>(
          rgba_filter_stream,
          rgba_filter_stream_u16,
          luma_filter_stream,
          luma_filter_stream_u16,
          resample_outputs,
          rgb,
          rgba,
          rgb_u16,
          rgba_u16,
          luma,
          luma_u16,
          hsv,
          rgba_scratch,
          rgb_scratch,
          rgba_scratch_u16,
          rgba_color_scratch_u16,
          luma_scratch_u16,
          w,
          plan,
          idx,
          use_simd,
          // `<16, false>` rides the u16 luma stream (parity with `Gray16`), so
          // there is no contiguous u8 Y plane and no u8 de-interleave scratch.
          &[],
          None,
          |dst| ya16_to_rgba_row::<BE>(packed, dst, w, use_simd),
          |dst| ya16_to_rgba_u16_row::<BE>(packed, dst, w, use_simd),
          // Native-Y u16 de-interleave (the exact host-native Y the direct
          // `ya16_to_luma_u16_row` emits) — parity with `Gray16`'s native-Y
          // filter.
          |dst| ya16_to_luma_u16_row::<BE>(packed, dst, w, use_simd),
          // u16-luma path, so this u8 de-interleave is never called.
          |_dst: &mut [u8]| {},
        ),
      };
    }

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // Output-mode flags — resolved BEFORE any output row is written so the
    // atomicity preflight below runs ahead of the luma writes.
    let want_rgb = self.rgb.is_some();
    let want_rgba = self.rgba.is_some();
    let want_hsv = self.hsv.is_some();

    // Atomicity preflight (#308, cf. the crate's #180 resample fix and the
    // planar / semi-planar / packed-YUV siblings): the only fallible row
    // scratch this path can grow is the RGB row buffer, and its allocating
    // (`None`) arm is reached for exactly one identity shape —
    // `want_hsv && want_rgba && !want_rgb` (every other `!want_rgb` case is
    // served by a standalone RGBA / HSV fast path below that never stages RGB).
    // Because `Y >> 8` luma, luma_u16 and the u16 RGB/RGBA rows are written
    // before that scratch, reserve it up front so an allocator refusal returns
    // a typed `AllocationFailed` leaving the output frame untouched rather than
    // partially mutated. The later staging call reuses the already-sized buffer
    // (default path byte-identical).
    if want_hsv && want_rgba && !want_rgb {
      rgb_row_buf_or_scratch(
        self.rgb.as_deref_mut(),
        &mut self.rgb_scratch,
        one_plane_start,
        one_plane_end,
        w,
        h,
      )?;
    }

    // luma u8 — `Y >> 8`.
    if let Some(buf) = self.luma.as_deref_mut() {
      ya16_to_luma_row::<BE>(
        packed,
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    // luma u16 — native pass-through.
    if let Some(buf) = self.luma_u16.as_deref_mut() {
      ya16_to_luma_u16_row::<BE>(
        packed,
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    // u16 RGB / RGBA path. Strategy A + α-patch for RGBA.
    let want_rgb_u16 = self.rgb_u16.is_some();
    let want_rgba_u16 = self.rgba_u16.is_some();

    if want_rgba_u16 && !want_rgb_u16 {
      let rgba_u16_buf = self.rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row =
        rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
      ya16_to_rgba_u16_row::<BE>(packed, rgba_u16_row, w, use_simd);
    } else if want_rgb_u16 {
      let rgb_u16_buf = self.rgb_u16.as_deref_mut().unwrap();
      let rgb_plane_start = one_plane_start * 3;
      let rgb_plane_end =
        one_plane_end
          .checked_mul(3)
          .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
            w, h, 3,
          )))?;
      let rgb_u16_row = &mut rgb_u16_buf[rgb_plane_start..rgb_plane_end];
      ya16_to_rgb_u16_row::<BE>(packed, rgb_u16_row, w, use_simd);
      if want_rgba_u16 {
        let rgba_u16_buf = self.rgba_u16.as_deref_mut().unwrap();
        let rgba_u16_row =
          rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
        expand_rgb_u16_to_rgba_u16_row::<16>(rgb_u16_row, rgba_u16_row, w);
        // Patch α from source (native u16 depth). `BE` is propagated from
        // the parent `Ya16Frame<'_, BE>` so the loader byte-swaps correctly
        // for both LE and BE inputs.
        copy_alpha_ya_u16::<BE>(packed, rgba_u16_row, w);
      }
    }

    // u8 RGB / RGBA / HSV path (flags hoisted above for the atomicity
    // preflight). Strategy A+: rgb first, then copy α into rgba.

    // Standalone RGBA fast path.
    if want_rgba && !want_rgb && !want_hsv {
      let rgba_buf = self.rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      ya16_to_rgba_row::<BE>(packed, rgba_row, w, use_simd);
      return Ok(());
    }

    // Standalone HSV fast path.
    if want_hsv && !want_rgb && !want_rgba {
      let hsv = self.hsv.as_mut().unwrap();
      let (h, s, v) = hsv.hsv();
      ya16_to_hsv_row::<BE>(
        packed,
        &mut h[one_plane_start..one_plane_end],
        &mut s[one_plane_start..one_plane_end],
        &mut v[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
      return Ok(());
    }

    if !want_rgb && !want_rgba && !want_hsv {
      return Ok(());
    }

    // RGB kernel.
    let rgb_row = rgb_row_buf_or_scratch(
      self.rgb.as_deref_mut(),
      &mut self.rgb_scratch,
      one_plane_start,
      one_plane_end,
      w,
      h,
    )?;
    ya16_to_rgb_row::<BE>(packed, rgb_row, w, use_simd);

    if let Some(hsv) = self.hsv.as_mut() {
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

    // Strategy A+: expand RGB→RGBA then patch α from source.
    if let Some(buf) = self.rgba.as_deref_mut() {
      let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
      // Overwrite the α channel with real source α (>> 8 for u8 output).
      // `BE` is propagated from the parent `Ya16Frame<'_, BE>`.
      copy_alpha_ya_u16_to_u8::<BE>(packed, rgba_row, w);
    }

    Ok(())
  }
}

// ---- Yaf32 impl -------------------------------------------------------------

impl<'a, R, const BE: bool> MixedSinker<'a, Yaf32<BE>, R> {
  /// Attaches an 8-bit RGBA output buffer. α is `clamp(source_A, 0, 1) x 255`
  /// (real source alpha).
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

  /// Attaches a u16 RGB output buffer.
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

  /// Attaches a u16 RGBA output buffer. α is `clamp(source_A, 0, 1) x 65535`.
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

  /// Attaches a u16 luma output buffer (`clamp(Y,0,1) x 65535`).
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

  /// Attaches a packed f32 RGB output buffer. Lossless replicate of Y → R=G=B
  /// (α dropped).
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

  /// Attaches an f32 luma output buffer. Lossless pass-through of Y.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_luma_f32(mut self, buf: &'a mut [f32]) -> Result<Self, MixedSinkerError> {
    self.set_luma_f32(buf)?;
    Ok(self)
  }
  /// In-place variant of [`with_luma_f32`](Self::with_luma_f32).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_luma_f32(&mut self, buf: &'a mut [f32]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_elems(1)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::InsufficientLumaF32Buffer(
        InsufficientBuffer::new(expected, buf.len()),
      ));
    }
    self.luma_f32 = Some(buf);
    Ok(self)
  }
}

impl<R, const BE: bool> Yaf32Sink<BE> for MixedSinker<'_, Yaf32<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, Yaf32<BE>, R> {
  type Input<'r> = Yaf32Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    // New frame: restart the 3-channel `[Yc, A, Y]` colour+alpha+native-Y
    // stream (area or filter kind, lazily created in `process`) and re-arm the
    // alpha-mode snapshot, mirroring the alpha-aware `Ya8` / `Ya16` sinks.
    if let Some(stream) = self.yaf_stream_f32.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.yaf_filter_stream_f32.as_mut() {
      stream.reset();
    }
    self.resample_outputs = None;
    self.frozen_alpha_mode = Some(self.alpha_mode);
    Ok(())
  }

  fn process(&mut self, row: Yaf32Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;
    let packed = row.packed(); // &[f32], length = width * 2

    if packed.len() != w * 2 {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Y,
        idx,
        w * 2,
        packed.len(),
      )));
    }
    if idx >= h {
      return Err(MixedSinkerError::RowIndexOutOfRange(
        RowIndexOutOfRange::new(idx, h),
      ));
    }

    // Non-identity plan: `Yaf32` is f32 gray+alpha — the float twin of `Ya16`.
    // The wire `[Y, A]` row stages as a host-native 3-channel `[Yc, A, Y]` row
    // (`Yc` = colour luma, premultiplied when straight-alpha is off; the third
    // channel is the independent native luma), all binned in f32, then every
    // output derives from each finalized binned row via the exact direct
    // `grayf32_*` kernels (colour Y broadcast for RGB/HSV, native Y for luma,
    // source A patched into the α channel). The span kind picks the engine;
    // premultiplied + filter has no analogue, so it routes to the area tail
    // (which surfaces the typed `UnsupportedFilter`).
    if self.plan.is_some() {
      let alpha_mode = self.alpha_mode;
      let premult = alpha_mode.is_premultiplied();
      let Self {
        rgb,
        rgb_u16,
        rgba,
        rgba_u16,
        luma,
        luma_u16,
        rgb_f32,
        luma_f32,
        hsv,
        yaf_stream_f32,
        yaf_filter_stream_f32,
        yaf_stage_scratch_f32,
        yaf_plane_scratch_f32,
        resample_outputs,
        frozen_alpha_mode,
        plan,
        ..
      } = self;
      let plan = plan.as_ref().expect("plan.is_some() checked above");
      check_frozen_alpha_mode(*frozen_alpha_mode, alpha_mode, idx)?;
      return yaf_process_resampled(
        yaf_stream_f32,
        yaf_filter_stream_f32,
        yaf_stage_scratch_f32,
        yaf_plane_scratch_f32,
        resample_outputs,
        rgb,
        rgb_u16,
        rgba,
        rgba_u16,
        luma,
        luma_u16,
        rgb_f32,
        luma_f32,
        hsv,
        plan,
        w,
        idx,
        use_simd,
        premult,
        |stage| {
          // Decode wire `[Y, A]` f32 (source `BE`) into host-native `[Yc=Y, A, Y]`.
          for x in 0..w {
            let y = if BE {
              f32::from_bits(u32::from_be(packed[x * 2].to_bits()))
            } else {
              f32::from_bits(u32::from_le(packed[x * 2].to_bits()))
            };
            let a = if BE {
              f32::from_bits(u32::from_be(packed[x * 2 + 1].to_bits()))
            } else {
              f32::from_bits(u32::from_le(packed[x * 2 + 1].to_bits()))
            };
            stage[x * 3] = y;
            stage[x * 3 + 1] = a;
            stage[x * 3 + 2] = y;
          }
        },
      );
    }

    yaf_direct_emit(
      packed,
      w,
      h,
      idx,
      use_simd,
      &mut self.luma_f32,
      &mut self.rgb_f32,
      &mut self.luma,
      &mut self.luma_u16,
      &mut self.rgb_u16,
      &mut self.rgba_u16,
      &mut self.rgb,
      &mut self.rgba,
      &mut self.hsv,
      &mut self.rgb_scratch,
      yaf32_to_luma_f32_row::<BE>,
      yaf32_to_rgb_f32_row::<BE>,
      yaf32_to_luma_row::<BE>,
      yaf32_to_luma_u16_row::<BE>,
      yaf32_to_rgb_u16_row::<BE>,
      yaf32_to_rgba_u16_row::<BE>,
      yaf32_to_rgb_row::<BE>,
      yaf32_to_rgba_row::<BE>,
      yaf32_to_hsv_row::<BE>,
    )
  }
}

// ---- Yaf16 impl -------------------------------------------------------------

impl<'a, R, const BE: bool> MixedSinker<'a, Yaf16<BE>, R> {
  /// Attaches an 8-bit RGBA output buffer. α is `clamp(source_A, 0, 1) x 255`
  /// (real source alpha, widened from f16).
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

  /// Attaches a u16 RGB output buffer.
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

  /// Attaches a u16 RGBA output buffer. α is `clamp(source_A, 0, 1) x 65535`.
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

  /// Attaches a u16 luma output buffer (`clamp(Y,0,1) x 65535`).
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

  /// Attaches a packed f32 RGB output buffer. Lossless widen Y f16 → f32 then
  /// replicate Y → R=G=B (α dropped).
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

  /// Attaches an f32 luma output buffer. Lossless widen Y f16 → f32.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_luma_f32(mut self, buf: &'a mut [f32]) -> Result<Self, MixedSinkerError> {
    self.set_luma_f32(buf)?;
    Ok(self)
  }
  /// In-place variant of [`with_luma_f32`](Self::with_luma_f32).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_luma_f32(&mut self, buf: &'a mut [f32]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_elems(1)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::InsufficientLumaF32Buffer(
        InsufficientBuffer::new(expected, buf.len()),
      ));
    }
    self.luma_f32 = Some(buf);
    Ok(self)
  }
}

impl<R, const BE: bool> Yaf16Sink<BE> for MixedSinker<'_, Yaf16<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, Yaf16<BE>, R> {
  type Input<'r> = Yaf16Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    if let Some(stream) = self.yaf_stream_f32.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.yaf_filter_stream_f32.as_mut() {
      stream.reset();
    }
    self.resample_outputs = None;
    self.frozen_alpha_mode = Some(self.alpha_mode);
    Ok(())
  }

  fn process(&mut self, row: Yaf16Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;
    let packed = row.packed(); // &[half::f16], length = width * 2

    if packed.len() != w * 2 {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Y,
        idx,
        w * 2,
        packed.len(),
      )));
    }
    if idx >= h {
      return Err(MixedSinkerError::RowIndexOutOfRange(
        RowIndexOutOfRange::new(idx, h),
      ));
    }

    // The half-float twin of the `Yaf32` resample: the staging widens each
    // wire `[Y, A]` f16 to host-native f32 (lossless), so once staged the
    // 3-channel `[Yc, A, Y]` bin + emit is byte-identical to `Yaf32`.
    if self.plan.is_some() {
      let alpha_mode = self.alpha_mode;
      let premult = alpha_mode.is_premultiplied();
      let Self {
        rgb,
        rgb_u16,
        rgba,
        rgba_u16,
        luma,
        luma_u16,
        rgb_f32,
        luma_f32,
        hsv,
        yaf_stream_f32,
        yaf_filter_stream_f32,
        yaf_stage_scratch_f32,
        yaf_plane_scratch_f32,
        resample_outputs,
        frozen_alpha_mode,
        plan,
        ..
      } = self;
      let plan = plan.as_ref().expect("plan.is_some() checked above");
      check_frozen_alpha_mode(*frozen_alpha_mode, alpha_mode, idx)?;
      return yaf_process_resampled(
        yaf_stream_f32,
        yaf_filter_stream_f32,
        yaf_stage_scratch_f32,
        yaf_plane_scratch_f32,
        resample_outputs,
        rgb,
        rgb_u16,
        rgba,
        rgba_u16,
        luma,
        luma_u16,
        rgb_f32,
        luma_f32,
        hsv,
        plan,
        w,
        idx,
        use_simd,
        premult,
        |stage| {
          // Widen wire `[Y, A]` f16 (source `BE`) into host-native `[Yc=Y, A, Y]`.
          for x in 0..w {
            let y = half::f16::from_bits(if BE {
              u16::from_be(packed[x * 2].to_bits())
            } else {
              u16::from_le(packed[x * 2].to_bits())
            })
            .to_f32();
            let a = half::f16::from_bits(if BE {
              u16::from_be(packed[x * 2 + 1].to_bits())
            } else {
              u16::from_le(packed[x * 2 + 1].to_bits())
            })
            .to_f32();
            stage[x * 3] = y;
            stage[x * 3 + 1] = a;
            stage[x * 3 + 2] = y;
          }
        },
      );
    }

    yaf_direct_emit(
      packed,
      w,
      h,
      idx,
      use_simd,
      &mut self.luma_f32,
      &mut self.rgb_f32,
      &mut self.luma,
      &mut self.luma_u16,
      &mut self.rgb_u16,
      &mut self.rgba_u16,
      &mut self.rgb,
      &mut self.rgba,
      &mut self.hsv,
      &mut self.rgb_scratch,
      yaf16_to_luma_f32_row::<BE>,
      yaf16_to_rgb_f32_row::<BE>,
      yaf16_to_luma_row::<BE>,
      yaf16_to_luma_u16_row::<BE>,
      yaf16_to_rgb_u16_row::<BE>,
      yaf16_to_rgba_u16_row::<BE>,
      yaf16_to_rgb_row::<BE>,
      yaf16_to_rgba_row::<BE>,
      yaf16_to_hsv_row::<BE>,
    )
  }
}

/// Direct (non-resampling) per-row emit shared by the [`Yaf32`] and [`Yaf16`]
/// sinks. The source element type `T` (`f32` / `half::f16`) and the per-output
/// conversion kernels are passed in (each with its source `BE` already baked
/// in), so the branch structure — luma f32/u8/u16, rgb f32, the u16 RGB/RGBA
/// Strategy A, and the u8 RGB/RGBA/HSV fan-out — is written once. RGBA outputs
/// carry real source α via the dedicated `*_to_rgba*` kernel (not a `0xFF`
/// fill), so the rgb+rgba combo derives each independently rather than
/// expand-then-patch.
#[allow(clippy::too_many_arguments)]
fn yaf_direct_emit<T, FLf32, FRf32, FL, FLu16, FRu16, FRAu16, FRgb, FRgba, FHsv>(
  packed: &[T],
  w: usize,
  h: usize,
  idx: usize,
  use_simd: bool,
  luma_f32: &mut Option<&mut [f32]>,
  rgb_f32: &mut Option<&mut [f32]>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  rgb_u16: &mut Option<&mut [u16]>,
  rgba_u16: &mut Option<&mut [u16]>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  hsv: &mut Option<mediaframe::source::HsvFrameMut<'_>>,
  rgb_scratch: &mut std::vec::Vec<u8>,
  to_luma_f32: FLf32,
  to_rgb_f32: FRf32,
  to_luma: FL,
  to_luma_u16: FLu16,
  to_rgb_u16: FRu16,
  to_rgba_u16: FRAu16,
  to_rgb: FRgb,
  to_rgba: FRgba,
  to_hsv: FHsv,
) -> Result<(), MixedSinkerError>
where
  FLf32: Fn(&[T], &mut [f32], usize, bool),
  FRf32: Fn(&[T], &mut [f32], usize, bool),
  FL: Fn(&[T], &mut [u8], usize, bool),
  FLu16: Fn(&[T], &mut [u16], usize, bool),
  FRu16: Fn(&[T], &mut [u16], usize, bool),
  FRAu16: Fn(&[T], &mut [u16], usize, bool),
  FRgb: Fn(&[T], &mut [u8], usize, bool),
  FRgba: Fn(&[T], &mut [u8], usize, bool),
  FHsv: Fn(&[T], &mut [u8], &mut [u8], &mut [u8], usize, bool),
{
  let one_plane_start = idx * w;
  let one_plane_end = one_plane_start + w;

  // luma f32 pass-through — highest priority (no clamp, no round).
  if let Some(buf) = luma_f32.as_deref_mut() {
    to_luma_f32(
      packed,
      &mut buf[one_plane_start..one_plane_end],
      w,
      use_simd,
    );
  }
  // rgb_f32 — lossless replicate Y → R=G=B.
  if let Some(buf) = rgb_f32.as_deref_mut() {
    let rgb_f32_start = one_plane_start * 3;
    let rgb_f32_end = one_plane_end
      .checked_mul(3)
      .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
        w, h, 3,
      )))?;
    to_rgb_f32(packed, &mut buf[rgb_f32_start..rgb_f32_end], w, use_simd);
  }
  // luma u8.
  if let Some(buf) = luma.as_deref_mut() {
    to_luma(
      packed,
      &mut buf[one_plane_start..one_plane_end],
      w,
      use_simd,
    );
  }
  // luma u16.
  if let Some(buf) = luma_u16.as_deref_mut() {
    to_luma_u16(
      packed,
      &mut buf[one_plane_start..one_plane_end],
      w,
      use_simd,
    );
  }

  // u16 RGB / RGBA path. RGBA carries real source α (its own kernel).
  let want_rgb_u16 = rgb_u16.is_some();
  let want_rgba_u16 = rgba_u16.is_some();
  if want_rgba_u16 && !want_rgb_u16 {
    let buf = rgba_u16.as_deref_mut().unwrap();
    let row = rgba_u16_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
    to_rgba_u16(packed, row, w, use_simd);
  } else if want_rgb_u16 {
    let buf = rgb_u16.as_deref_mut().unwrap();
    let rgb_plane_start = one_plane_start * 3;
    let rgb_plane_end = one_plane_end
      .checked_mul(3)
      .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
        w, h, 3,
      )))?;
    to_rgb_u16(
      packed,
      &mut buf[rgb_plane_start..rgb_plane_end],
      w,
      use_simd,
    );
    if want_rgba_u16 {
      let buf = rgba_u16.as_deref_mut().unwrap();
      let row = rgba_u16_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      to_rgba_u16(packed, row, w, use_simd);
    }
  }

  // u8 RGB / RGBA / HSV path.
  let want_rgb = rgb.is_some();
  let want_rgba = rgba.is_some();
  let want_hsv = hsv.is_some();

  // Standalone RGBA fast path.
  if want_rgba && !want_rgb && !want_hsv {
    let buf = rgba.as_deref_mut().unwrap();
    let row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
    to_rgba(packed, row, w, use_simd);
    return Ok(());
  }

  // Standalone HSV fast path — gray achromatic (H=0, S=0, V=clamp(Y)x255).
  if want_hsv && !want_rgb {
    let hsv = hsv.as_mut().unwrap();
    let (hp, sp, vp) = hsv.hsv();
    to_hsv(
      packed,
      &mut hp[one_plane_start..one_plane_end],
      &mut sp[one_plane_start..one_plane_end],
      &mut vp[one_plane_start..one_plane_end],
      w,
      use_simd,
    );
    if let Some(buf) = rgba.as_deref_mut() {
      let row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      to_rgba(packed, row, w, use_simd);
    }
    return Ok(());
  }

  if !want_rgb && !want_rgba && !want_hsv {
    return Ok(());
  }

  // RGB kernel (`want_rgb` is implied here — the standalone fast paths above
  // caught every `!want_rgb` case).
  let rgb_row = rgb_row_buf_or_scratch(
    rgb.as_deref_mut(),
    rgb_scratch,
    one_plane_start,
    one_plane_end,
    w,
    h,
  )?;
  to_rgb(packed, rgb_row, w, use_simd);

  if let Some(hsv) = hsv.as_mut() {
    let (hp, sp, vp) = hsv.hsv();
    rgb_to_hsv_row(
      rgb_row,
      &mut hp[one_plane_start..one_plane_end],
      &mut sp[one_plane_start..one_plane_end],
      &mut vp[one_plane_start..one_plane_end],
      w,
      use_simd,
    );
  }

  if let Some(buf) = rgba.as_deref_mut() {
    let row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
    to_rgba(packed, row, w, use_simd);
  }

  Ok(())
}

/// Patches the α channel of an 8-bit packed RGBA row with `clamp(A,0,1) x 255`
/// (round-to-nearest), the float-alpha analogue of the integer `copy_alpha_ya*`
/// helpers used by the resampled Yaf emit after a `grayf32_to_rgba_row`
/// (which leaves α = `0xFF`).
#[cfg(feature = "gray")]
#[cfg_attr(not(tarpaulin), inline(always))]
fn yaf_patch_alpha_u8(alpha: &[f32], rgba: &mut [u8], width: usize) {
  for (i, &a) in alpha[..width].iter().enumerate() {
    rgba[i * 4 + 3] = (a.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
  }
}

/// Patches the α channel of a u16 packed RGBA row with `clamp(A,0,1) x 65535`
/// (round-to-nearest). The u16 twin of [`yaf_patch_alpha_u8`].
#[cfg(feature = "gray")]
#[cfg_attr(not(tarpaulin), inline(always))]
fn yaf_patch_alpha_u16(alpha: &[f32], rgba: &mut [u16], width: usize) {
  for (i, &a) in alpha[..width].iter().enumerate() {
    rgba[i * 4 + 3] = (a.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
  }
}

/// Row-stage fused downscale shared by [`Yaf32`] and [`Yaf16`]: `stage_yay`
/// decodes the wire `[Y, A]` row into a source-width host-native 3-channel
/// `[Yc, A, Y]` staging row (Yaf32 byte-swaps f32, Yaf16 widens f16); this tail
/// premultiplies the colour channel when `premult`, bins all three channels in
/// f32, and per finalized output row de-interleaves the binned `[cy, ca, ny]`
/// into contiguous `colorY | alpha | nativeY` planes — resolving the straight
/// colour Y (`cy / ca`, `ca == 0 -> 0`) — then runs the exact direct
/// `grayf32_*` kernels (colour Y for RGB/HSV, native Y for luma, source α
/// patched into the RGBA α channel). Every output is byte-identical to a direct
/// Yaf conversion of the binned frame.
///
/// Atomic preflight mirrors [`grayf32_process_resampled`]: a no-output call
/// returns before any freeze; an out-of-sequence row is rejected before the
/// freeze; staging and stream creation precede the single feed, so a failure
/// mutates no caller output. A premultiplied filter plan has no analogue (the
/// engine cannot un-premultiply), so the caller leaves it on this area tail,
/// which surfaces the typed `UnsupportedFilter`.
#[allow(clippy::too_many_arguments)]
fn yaf_process_resampled(
  yaf_stream_f32: &mut Option<std::boxed::Box<AreaStream<f32>>>,
  yaf_filter_stream_f32: &mut Option<std::boxed::Box<crate::resample::FilterStream<f32>>>,
  stage_scratch: &mut std::vec::Vec<f32>,
  plane_scratch: &mut std::vec::Vec<f32>,
  resample_outputs: &mut Option<super::FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgb_u16: &mut Option<&mut [u16]>,
  rgba: &mut Option<&mut [u8]>,
  rgba_u16: &mut Option<&mut [u16]>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  rgb_f32: &mut Option<&mut [f32]>,
  luma_f32: &mut Option<&mut [f32]>,
  hsv: &mut Option<mediaframe::source::HsvFrameMut<'_>>,
  plan: &ResamplePlan,
  w: usize,
  idx: usize,
  use_simd: bool,
  premult: bool,
  stage_yay: impl FnOnce(&mut [f32]),
) -> Result<(), MixedSinkerError> {
  // The binned planes are host-native f32, so the emit kernels recover an
  // already-host-native sample with `::<HOST_NATIVE_BE>` (the source `BE` was
  // consumed by `stage_yay`).
  const HOST_NATIVE_BE: bool = cfg!(target_endian = "big");
  // Single-kernel filter tail — reject a BICUBLIN plan before any state change.
  plan.ensure_single_kernel_filter()?;
  // Premultiplied alpha has no filter analogue (the engine cannot
  // un-premultiply), so a premultiplied filter plan surfaces the typed
  // `UnsupportedFilter` before any state change — matching the integer `Ya8` /
  // `Ya16` premultiplied-filter routing and the float `packed_rgba` area tail.
  if premult && plan.kind().is_filter() {
    return Err(plan.unsupported_filter().into());
  }
  let ow = plan.out_w();
  let want_rgb = rgb.is_some();
  let want_rgb_u16 = rgb_u16.is_some();
  let want_rgba = rgba.is_some();
  let want_rgba_u16 = rgba_u16.is_some();
  let want_hsv = hsv.is_some();
  // The u8 RGB kernel runs only when RGB output is requested; HSV / standalone
  // RGBA derive directly from the colour-Y plane.
  let need_rgb_kernel = want_rgb;

  let need_any = luma.is_some()
    || luma_u16.is_some()
    || luma_f32.is_some()
    || rgb_f32.is_some()
    || want_rgb
    || want_rgb_u16
    || want_rgba
    || want_rgba_u16
    || want_hsv;
  let is_filter = plan.kind().is_filter();
  // Sequence-counter row for the shared L1 preflight (the fed stream is the sole
  // per-row stream), or `None` when no output is attached so the preflight
  // short-circuits to a no-op. The span kind selects which engine's stream advances.
  let expected = if need_any {
    Some(if is_filter {
      yaf_filter_stream_f32.as_ref().map_or(0, |s| s.next_y())
    } else {
      yaf_stream_f32.as_ref().map_or(0, |s| s.next_y())
    })
  } else {
    None
  };
  // Compare-only preflight (NO commit): the output-set freeze commits only AFTER the
  // fallible stream alloc + scratch grows below succeed, so a first-row allocation
  // failure leaves `resample_outputs` uncommitted (retryable).
  if let core::ops::ControlFlow::Break(()) = super::resample_preflight_check_only(
    resample_outputs,
    luma,
    luma_u16,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    rgb_f32,
    &None,
    &None,
    &None,
    &None,
    hsv,
    luma_f32,
    expected,
    idx,
  )? {
    return Ok(());
  }

  // Recoverable source-width 3-channel `[Yc, A, Y]` staging, grown before the freeze.
  let stage_len =
    w.checked_mul(3)
      .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
        w,
        plan.src_h(),
        3,
      )))?;
  if stage_scratch.len() < stage_len {
    stage_scratch
      .try_reserve_exact(stage_len - stage_scratch.len())
      .map_err(|_| {
        MixedSinkerError::Resample(ResampleError::AllocationFailed(
          crate::resample::PlanGeometry::new(
            plan.src_w(),
            plan.src_h(),
            plan.out_w(),
            plan.out_h(),
          ),
        ))
      })?;
    stage_scratch.resize(stage_len, 0.0);
  }
  let stage = &mut stage_scratch[..stage_len];
  stage_yay(stage);
  if premult {
    // Premultiply the colour-luma channel only (`Yc *= A`); native Y (channel
    // 2) stays `mean(Y)`, alpha (channel 1) stays unchanged.
    for x in 0..w {
      let a = stage[x * 3 + 1];
      stage[x * 3] *= a;
    }
  }

  // Out-width 3-channel de-interleave staging, sized before the feed.
  let plane_len =
    ow.checked_mul(3)
      .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
        ow,
        plan.out_h(),
        3,
      )))?;
  if plane_scratch.len() < plane_len {
    plane_scratch
      .try_reserve_exact(plane_len - plane_scratch.len())
      .map_err(|_| {
        MixedSinkerError::Resample(ResampleError::AllocationFailed(
          crate::resample::PlanGeometry::new(
            plan.src_w(),
            plan.src_h(),
            plan.out_w(),
            plan.out_h(),
          ),
        ))
      })?;
    plane_scratch.resize(plane_len, 0.0);
  }
  // Build the missing 3-channel stream into a LOCAL, then insert the stream field and
  // freeze the output set TOGETHER — so a scratch OOM (above) leaves the field `None`
  // AND `resample_outputs` uncommitted (the #180 commit-together discipline; a pure
  // reordering of the pre-collapse split path).
  let new_area = if !is_filter && yaf_stream_f32.is_none() {
    let stream = AreaStream::new(plan.h(), plan.v(), plan.src_w(), plan.src_h(), 3)?;
    Some(crate::resample::try_box(stream).map_err(|_| {
      MixedSinkerError::Resample(ResampleError::AllocationFailed(
        crate::resample::PlanGeometry::new(plan.src_w(), plan.src_h(), plan.out_w(), plan.out_h()),
      ))
    })?)
  } else {
    None
  };
  let new_filter = if is_filter && yaf_filter_stream_f32.is_none() {
    let (fh, fv) = (
      plan
        .filter_h()
        .expect("filter plan carries horizontal windows"),
      plan
        .filter_v()
        .expect("filter plan carries vertical windows"),
    );
    let stream = crate::resample::FilterStream::new(fh, fv, plan.src_w(), plan.src_h(), 3)?;
    Some(crate::resample::try_box(stream).map_err(|_| {
      MixedSinkerError::Resample(ResampleError::AllocationFailed(
        crate::resample::PlanGeometry::new(plan.src_w(), plan.src_h(), plan.out_w(), plan.out_h()),
      ))
    })?)
  } else {
    None
  };
  if let Some(s) = new_area {
    *yaf_stream_f32 = Some(s);
  }
  if let Some(s) = new_filter {
    *yaf_filter_stream_f32 = Some(s);
  }
  frozen_outputs_check(
    resample_outputs,
    luma,
    luma_u16,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    rgb_f32,
    &None,
    &None,
    &None,
    &None,
    hsv,
    luma_f32,
    idx,
  )?;

  let mut emit = |oy: usize, binned: &[f32]| {
    let planes = &mut plane_scratch[..plane_len];
    let (color_y, rest) = planes.split_at_mut(ow);
    let (alpha, native_y) = rest.split_at_mut(ow);
    for x in 0..ow {
      color_y[x] = binned[x * 3];
      alpha[x] = binned[x * 3 + 1];
      native_y[x] = binned[x * 3 + 2];
    }
    if premult {
      for x in 0..ow {
        let a = alpha[x];
        color_y[x] = if a == 0.0 { 0.0 } else { color_y[x] / a };
      }
    }
    let color_y: &[f32] = color_y;
    let alpha: &[f32] = alpha;
    let native_y: &[f32] = native_y;

    // luma outputs — independent native Y (mode-independent `mean(Y)`).
    if let Some(buf) = luma_f32.as_deref_mut() {
      grayf32_to_luma_f32_row::<HOST_NATIVE_BE>(
        native_y,
        &mut buf[oy * ow..(oy + 1) * ow],
        ow,
        use_simd,
      );
    }
    if let Some(buf) = luma.as_deref_mut() {
      grayf32_to_luma_row::<HOST_NATIVE_BE>(
        native_y,
        &mut buf[oy * ow..(oy + 1) * ow],
        ow,
        use_simd,
      );
    }
    if let Some(buf) = luma_u16.as_deref_mut() {
      grayf32_to_luma_u16_row::<HOST_NATIVE_BE>(
        native_y,
        &mut buf[oy * ow..(oy + 1) * ow],
        ow,
        use_simd,
      );
    }
    // rgb_f32 — lossless replicate of the resolved colour Y.
    if let Some(buf) = rgb_f32.as_deref_mut() {
      grayf32_to_rgb_f32_row::<HOST_NATIVE_BE>(
        color_y,
        &mut buf[oy * 3 * ow..(oy + 1) * 3 * ow],
        ow,
        use_simd,
      );
    }

    // u16 RGB / RGBA.
    if want_rgba_u16 && !want_rgb_u16 {
      let buf = rgba_u16.as_deref_mut().unwrap();
      let row = &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow];
      grayf32_to_rgba_u16_row::<HOST_NATIVE_BE>(color_y, row, ow, use_simd);
      yaf_patch_alpha_u16(alpha, row, ow);
    } else if want_rgb_u16 {
      let buf = rgb_u16.as_deref_mut().unwrap();
      grayf32_to_rgb_u16_row::<HOST_NATIVE_BE>(
        color_y,
        &mut buf[oy * 3 * ow..(oy + 1) * 3 * ow],
        ow,
        use_simd,
      );
      if let Some(buf) = rgba_u16.as_deref_mut() {
        let row = &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow];
        grayf32_to_rgba_u16_row::<HOST_NATIVE_BE>(color_y, row, ow, use_simd);
        yaf_patch_alpha_u16(alpha, row, ow);
      }
    }

    // Standalone u8 RGBA — no RGB or HSV requested.
    if want_rgba && !need_rgb_kernel && !want_hsv {
      let buf = rgba.as_deref_mut().unwrap();
      let row = &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow];
      grayf32_to_rgba_row::<HOST_NATIVE_BE>(color_y, row, ow, use_simd);
      yaf_patch_alpha_u8(alpha, row, ow);
      return;
    }

    // Standalone u8 HSV (plus optional RGBA).
    if want_hsv && !want_rgb {
      let hsv = hsv.as_mut().unwrap();
      let (hp, sp, vp) = hsv.hsv();
      grayf32_to_hsv_row::<HOST_NATIVE_BE>(
        color_y,
        &mut hp[oy * ow..(oy + 1) * ow],
        &mut sp[oy * ow..(oy + 1) * ow],
        &mut vp[oy * ow..(oy + 1) * ow],
        ow,
        use_simd,
      );
      if let Some(buf) = rgba.as_deref_mut() {
        let row = &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow];
        grayf32_to_rgba_row::<HOST_NATIVE_BE>(color_y, row, ow, use_simd);
        yaf_patch_alpha_u8(alpha, row, ow);
      }
      return;
    }

    if !need_rgb_kernel {
      return;
    }

    // RGB attached: write the user buffer, then HSV-from-RGB and the RGBA
    // fan-out, mirroring the direct path's RGB-kernel branch.
    let buf = rgb
      .as_deref_mut()
      .expect("need_rgb_kernel implies RGB is attached");
    let rgb_row = &mut buf[oy * 3 * ow..(oy + 1) * 3 * ow];
    grayf32_to_rgb_row::<HOST_NATIVE_BE>(color_y, rgb_row, ow, use_simd);
    if let Some(hsv) = hsv.as_mut() {
      let (hp, sp, vp) = hsv.hsv();
      rgb_to_hsv_row(
        rgb_row,
        &mut hp[oy * ow..(oy + 1) * ow],
        &mut sp[oy * ow..(oy + 1) * ow],
        &mut vp[oy * ow..(oy + 1) * ow],
        ow,
        use_simd,
      );
    }
    if let Some(buf) = rgba.as_deref_mut() {
      let row = &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow];
      grayf32_to_rgba_row::<HOST_NATIVE_BE>(color_y, row, ow, use_simd);
      yaf_patch_alpha_u8(alpha, row, ow);
    }
  };

  // Feed the committed kind-appropriate 3-channel f32 stream; only one runs per frame.
  match plan.kind() {
    crate::resample::SpanKind::Area => {
      let stream = yaf_stream_f32.as_mut().expect("created above");
      stream.feed_row(idx, stage, use_simd, &mut emit)?;
    }
    crate::resample::SpanKind::Filter => {
      let stream = yaf_filter_stream_f32.as_mut().expect("created above");
      stream.feed_row(idx, stage, use_simd, &mut emit)?;
    }
  }

  Ok(())
}
