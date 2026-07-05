//! [`PixelSink`] implementations for Monoblack and Monowhite sources.
//!
//! Both formats are 1-bit-per-pixel achromatic bilevel: each source bit
//! expands to a u8 luma of `0` or `255` (Monoblack: bit=1 → 255;
//! Monowhite: bit=0 → 255). Every output is a broadcast of that luma —
//! `with_rgb` / `with_rgba` / `with_rgb_u16` / `with_rgba_u16` replicate
//! Y to R=G=B (alpha forced opaque), `with_luma` copies it,
//! `with_luma_u16` zero-extends it, and `with_hsv` emits H=0, S=0, V=Y
//! (achromatic convention). There is no chroma matrix and no palette, so
//! a resampled path expands the bits to a source-width 0/255 luma plane
//! and resamples that single plane, deriving every attached output from
//! each finalized resampled luma row — identical to the direct path with
//! the resampled value standing in for the per-pixel expanded value.
//!
//! An **area** plan bins the expanded luma through a single-channel
//! [`AreaStream<u8>`](crate::resample::AreaStream); a **filter** plan
//! resamples it through the signed-coefficient single-channel
//! [`FilterStream<u8>`](crate::resample::FilterStream) (the filter twin of
//! the bin). Both feed the **same** emit, so the filter output matches the
//! area output up to the kernel weights. Filtering a bilevel image to
//! continuous grayscale *is* antialiasing it: a hard 0/255 edge resampled
//! with a Triangle/Catmull-Rom/Lanczos window becomes a smooth gray ramp.
//! The expanded luma spans the full `u8` range (0..=255), so no
//! native-depth clamp applies — the stream's `0..=255` clamp *is* the
//! native range.

use super::{
  InsufficientBuffer, MixedSinker, MixedSinkerError, RowIndexOutOfRange, RowShapeMismatch,
  RowSlice, check_dimensions_match, frozen_outputs_check, source_luma_scratch,
};
use crate::{
  PixelSink,
  resample::{AreaStream, FilterStream, ResamplePlan, RowResampler},
  row,
  source::{Monoblack, MonoblackRow, MonoblackSink, Monowhite, MonowhiteRow, MonowhiteSink},
};
use mediaframe::source::HsvFrameMut;

/// Fused resample tail shared by Monoblack and Monowhite, for both an
/// **area** and a **filter** plan. The closure `expand_luma` fills a
/// source-width `u8` scratch with the expanded 0/255 luma of the current
/// 1-bit source row (the same `mono*_to_luma_row` kernel the direct path
/// uses); that single luma plane is resampled — binned through the
/// single-channel [`AreaStream<u8>`] on an area plan, or filtered through
/// the single-channel [`FilterStream<u8>`] (the filter twin of the bin) on
/// a filter plan — and every attached output is derived from each
/// finalized resampled luma row: a copy for luma, a zero-extend for
/// luma_u16, a broadcast to R=G=B for the RGB outputs (alpha forced
/// opaque), and H=0/S=0/V=Y for HSV — mirroring the direct mono kernels
/// exactly. The expanded luma is full-range `u8` (0..=255), so no
/// native-depth clamp applies on either path.
///
/// The two plans differ only in the resampler kind built here; the
/// expand-then-resample staging and per-output emit are identical
/// (factored into [`mono_luma_feed_emit`] behind
/// [`RowResampler`](crate::resample::RowResampler), which both stream kinds
/// implement), so the filter output matches the area output up to the
/// kernel weights and the area path stays bit-identical.
///
/// Atomic preflight: the output set is frozen and stream sequencing
/// checked **before** any scratch allocation or stream feed, so a
/// failed call mutates no caller output and a no-output sink stays a
/// legal no-op.
#[allow(clippy::too_many_arguments)]
fn mono_luma_resample(
  luma_stream: &mut Option<std::boxed::Box<AreaStream<u8>>>,
  luma_filter_stream: &mut Option<std::boxed::Box<FilterStream<u8>>>,
  resample_outputs: &mut Option<super::FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  rgb_u16: &mut Option<&mut [u16]>,
  rgba_u16: &mut Option<&mut [u16]>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  scratch: &mut std::vec::Vec<u8>,
  w: usize,
  plan: &ResamplePlan,
  idx: usize,
  use_simd: bool,
  expand_luma: impl FnOnce(&mut [u8]),
) -> Result<(), MixedSinkerError> {
  // Single-kernel filter tail — reject a BICUBLIN plan (its chroma windows are
  // read only by the `Yuv420p` per-plane route) before any state change. The
  // mono luma stream is single-kernel, so a bicublin plan would mis-filter.
  plan.ensure_single_kernel_filter()?;
  let is_filter = plan.kind().is_filter();
  let need_any = luma.is_some()
    || luma_u16.is_some()
    || rgb.is_some()
    || rgba.is_some()
    || rgb_u16.is_some()
    || rgba_u16.is_some()
    || hsv.is_some();
  // Sequence-counter row for the shared L1 preflight (the fed stream is the sole
  // per-row stream), or `None` when no output is attached so the preflight
  // short-circuits to a no-op. The plan kind is fixed per sink, so only one of the
  // two streams is ever fed.
  let expected = if need_any {
    Some(if is_filter {
      luma_filter_stream
        .as_ref()
        .map_or(0, |stream| stream.next_y())
    } else {
      luma_stream.as_ref().map_or(0, |stream| stream.next_y())
    })
  } else {
    None
  };
  // Compare-only preflight (NO commit): the output-set freeze commits only AFTER the
  // fallible stream alloc + scratch pregrow below succeed, so a first-row allocation
  // failure leaves `resample_outputs` uncommitted (retryable with a changed output
  // attachment). A no-output sink short-circuits to a no-op.
  if let core::ops::ControlFlow::Break(()) = super::resample_preflight_check_only(
    resample_outputs,
    luma,
    luma_u16,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    &None,
    &None,
    &None,
    &None,
    &None,
    hsv,
    &None,
    expected,
    idx,
  )? {
    return Ok(());
  }

  // Build the missing per-kind stream into a LOCAL and pregrow the source-luma
  // scratch, then insert the stream field and freeze the output set TOGETHER — so a
  // scratch OOM after a successful stream build leaves the field `None` AND
  // `resample_outputs` uncommitted (the #180 commit-together discipline). The
  // expanded luma spans the full `u8` range, so neither stream needs a native-depth
  // clamp. Stage + feed + emit is shared via [`mono_luma_feed_emit`], which re-runs
  // the (now no-op) source-luma grow before the feed.
  if is_filter {
    let (fh, fv) = (
      plan
        .filter_h()
        .expect("filter plan carries horizontal windows"),
      plan
        .filter_v()
        .expect("filter plan carries vertical windows"),
    );
    let new_stream = if luma_filter_stream.is_none() {
      let stream = FilterStream::new(fh, fv, plan.src_w(), plan.src_h(), 1)?;
      Some(crate::resample::try_box(stream).map_err(|_| {
        MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
          crate::resample::PlanGeometry::new(
            plan.src_w(),
            plan.src_h(),
            plan.out_w(),
            plan.out_h(),
          ),
        ))
      })?)
    } else {
      None
    };
    source_luma_scratch(scratch, w, plan)?;
    if let Some(s) = new_stream {
      *luma_filter_stream = Some(s);
    }
    frozen_outputs_check(
      resample_outputs,
      luma,
      luma_u16,
      rgb,
      rgba,
      rgb_u16,
      rgba_u16,
      &None,
      &None,
      &None,
      &None,
      &None,
      hsv,
      &None,
      idx,
    )?;
    let stream = luma_filter_stream.as_mut().expect("created above");
    mono_luma_feed_emit(
      stream,
      rgb,
      rgba,
      rgb_u16,
      rgba_u16,
      luma,
      luma_u16,
      hsv,
      scratch,
      w,
      plan,
      idx,
      use_simd,
      expand_luma,
    )
  } else {
    let new_stream = if luma_stream.is_none() {
      let stream = AreaStream::new(plan.h(), plan.v(), plan.src_w(), plan.src_h(), 1)?;
      Some(crate::resample::try_box(stream).map_err(|_| {
        MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
          crate::resample::PlanGeometry::new(
            plan.src_w(),
            plan.src_h(),
            plan.out_w(),
            plan.out_h(),
          ),
        ))
      })?)
    } else {
      None
    };
    source_luma_scratch(scratch, w, plan)?;
    if let Some(s) = new_stream {
      *luma_stream = Some(s);
    }
    frozen_outputs_check(
      resample_outputs,
      luma,
      luma_u16,
      rgb,
      rgba,
      rgb_u16,
      rgba_u16,
      &None,
      &None,
      &None,
      &None,
      &None,
      hsv,
      &None,
      idx,
    )?;
    let stream = luma_stream.as_mut().expect("created above");
    mono_luma_feed_emit(
      stream,
      rgb,
      rgba,
      rgb_u16,
      rgba_u16,
      luma,
      luma_u16,
      hsv,
      scratch,
      w,
      plan,
      idx,
      use_simd,
      expand_luma,
    )
  }
}

/// Shared stage-then-feed tail for the mono resample, used by both the area
/// and filter arms of [`mono_luma_resample`]. The two paths differ only in
/// the resampler kind (`stream`), so the expand-then-resample staging and
/// the per-output emit are identical and live here behind the
/// [`RowResampler`](crate::resample::RowResampler) trait — keeping the emit
/// byte-identical between the arms is what makes the filter output match the
/// area output up to the kernel weights, and the area path bit-identical to
/// before this routing.
///
/// Expands the 1-bit source row to source-width 0/255 luma in the shared
/// scratch (fallible growth runs before the feed, keeping the call atomic),
/// then resamples it through the single-channel `stream`, deriving every
/// attached output from each finalized resampled luma row.
#[allow(clippy::too_many_arguments)]
fn mono_luma_feed_emit<S: RowResampler<u8>>(
  stream: &mut S,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  rgb_u16: &mut Option<&mut [u16]>,
  rgba_u16: &mut Option<&mut [u16]>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  scratch: &mut std::vec::Vec<u8>,
  w: usize,
  plan: &ResamplePlan,
  idx: usize,
  use_simd: bool,
  expand_luma: impl FnOnce(&mut [u8]),
) -> Result<(), MixedSinkerError> {
  let ow = plan.out_w();
  let luma_row = source_luma_scratch(scratch, w, plan)?;
  expand_luma(luma_row);

  stream.feed_row(idx, luma_row, use_simd, |oy, out_row| {
    if let Some(buf) = luma.as_deref_mut() {
      buf[oy * ow..(oy + 1) * ow].copy_from_slice(out_row);
    }
    if let Some(buf) = luma_u16.as_deref_mut() {
      for (dst, &y) in buf[oy * ow..(oy + 1) * ow].iter_mut().zip(out_row) {
        *dst = y as u16;
      }
    }
    if let Some(buf) = rgb.as_deref_mut() {
      for (px, &y) in buf[oy * 3 * ow..(oy + 1) * 3 * ow]
        .chunks_exact_mut(3)
        .zip(out_row)
      {
        px[0] = y;
        px[1] = y;
        px[2] = y;
      }
    }
    if let Some(buf) = rgba.as_deref_mut() {
      for (px, &y) in buf[oy * 4 * ow..(oy + 1) * 4 * ow]
        .chunks_exact_mut(4)
        .zip(out_row)
      {
        px[0] = y;
        px[1] = y;
        px[2] = y;
        px[3] = 0xFF;
      }
    }
    if let Some(buf) = rgb_u16.as_deref_mut() {
      for (px, &y) in buf[oy * 3 * ow..(oy + 1) * 3 * ow]
        .chunks_exact_mut(3)
        .zip(out_row)
      {
        let y16 = y as u16;
        px[0] = y16;
        px[1] = y16;
        px[2] = y16;
      }
    }
    if let Some(buf) = rgba_u16.as_deref_mut() {
      for (px, &y) in buf[oy * 4 * ow..(oy + 1) * 4 * ow]
        .chunks_exact_mut(4)
        .zip(out_row)
      {
        let y16 = y as u16;
        px[0] = y16;
        px[1] = y16;
        px[2] = y16;
        px[3] = 0x00FF;
      }
    }
    if let Some(hsv) = hsv.as_mut() {
      let (h, s, v) = hsv.hsv();
      let span = oy * ow..(oy + 1) * ow;
      v[span.clone()].copy_from_slice(out_row);
      for px in &mut h[span.clone()] {
        *px = 0;
      }
      for px in &mut s[span] {
        *px = 0;
      }
    }
  })?;

  Ok(())
}

// ---- Monoblack impl ---------------------------------------------------------

impl<'a, R> MixedSinker<'a, Monoblack, R> {
  /// Attaches a packed **`u8`** RGBA output buffer.
  ///
  /// Length is measured in `u8` **bytes**: minimum `width * height * 4`.
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

  /// Attaches a packed **`u16`** RGB output buffer.
  ///
  /// Length is measured in `u16` **elements** (not bytes): minimum
  /// `width * height * 3`.
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

  /// Attaches a packed **`u16`** RGBA output buffer.
  ///
  /// Length is measured in `u16` **elements** (not bytes): minimum
  /// `width * height * 4`.
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

  /// Attaches a planar **`u16`** luma output buffer.
  ///
  /// Luma is derived from RGB via BT.709 weights (by default).
  /// Length: minimum `width * height`.
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

impl<R> MonoblackSink for MixedSinker<'_, Monoblack, R> {}

impl<R> PixelSink for MixedSinker<'_, Monoblack, R> {
  type Input<'i> = MonoblackRow<'i>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    if let Some(stream) = self.luma_stream.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.luma_filter_stream.as_mut() {
      stream.reset();
    }
    self.resample_outputs = None;
    Ok(())
  }

  fn process(&mut self, row: Self::Input<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    let min_bytes = w.div_ceil(8);
    if row.data().len() < min_bytes {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Y,
        idx,
        min_bytes,
        row.data().len(),
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
      rgb_u16,
      rgba_u16,
      luma,
      luma_u16,
      hsv,
      plan,
      luma_stream,
      luma_filter_stream,
      resample_outputs,
      rgb_scratch,
      ..
    } = self;

    // Non-identity plan: expand the 1-bit source row to source-width
    // 0/255 luma (the same Monoblack→luma kernel the direct path uses),
    // resample that luma plane (area-bin or filter, per the plan kind), and
    // derive every output from each finalized resampled luma row. The
    // resample is of the EXPANDED luma — expand then resample, the resample
    // oracle for an achromatic source; a filter window over the 0/255 edge
    // is the desired antialiasing (a smooth gray ramp).
    if let Some(plan) = plan.as_ref() {
      let data = row.data();
      return mono_luma_resample(
        luma_stream,
        luma_filter_stream,
        resample_outputs,
        rgb,
        rgba,
        rgb_u16,
        rgba_u16,
        luma,
        luma_u16,
        hsv,
        rgb_scratch,
        w,
        plan,
        idx,
        use_simd,
        |scratch| row::monoblack_to_luma_row(data, scratch, w, use_simd),
      );
    }

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    if let Some(buf) = rgb.as_deref_mut() {
      row::monoblack_to_rgb_or_rgba_row::<false>(
        row.data(),
        &mut buf[one_plane_start * 3..one_plane_end * 3],
        w,
        use_simd,
      );
    }

    if let Some(buf) = rgba.as_deref_mut() {
      row::monoblack_to_rgb_or_rgba_row::<true>(
        row.data(),
        &mut buf[one_plane_start * 4..one_plane_end * 4],
        w,
        use_simd,
      );
    }

    if let Some(buf) = rgb_u16.as_deref_mut() {
      row::monoblack_to_rgb_u16_or_rgba_u16_row::<false>(
        row.data(),
        &mut buf[one_plane_start * 3..one_plane_end * 3],
        w,
        use_simd,
      );
    }

    if let Some(buf) = rgba_u16.as_deref_mut() {
      row::monoblack_to_rgb_u16_or_rgba_u16_row::<true>(
        row.data(),
        &mut buf[one_plane_start * 4..one_plane_end * 4],
        w,
        use_simd,
      );
    }

    if let Some(buf) = luma.as_deref_mut() {
      row::monoblack_to_luma_row(
        row.data(),
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    if let Some(buf) = luma_u16.as_deref_mut() {
      row::monoblack_to_luma_u16_row(
        row.data(),
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    if let Some(hsv) = hsv.as_mut() {
      let (h, s, v) = hsv.hsv();
      row::monoblack_to_hsv_row(
        row.data(),
        &mut h[one_plane_start..one_plane_end],
        &mut s[one_plane_start..one_plane_end],
        &mut v[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    Ok(())
  }
}

// ---- Monowhite impl ---------------------------------------------------------

impl<'a, R> MixedSinker<'a, Monowhite, R> {
  /// Attaches a packed **`u8`** RGBA output buffer.
  ///
  /// Length is measured in `u8` **bytes**: minimum `width * height * 4`.
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

  /// Attaches a packed **`u16`** RGB output buffer.
  ///
  /// Length is measured in `u16` **elements** (not bytes): minimum
  /// `width * height * 3`.
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

  /// Attaches a packed **`u16`** RGBA output buffer.
  ///
  /// Length is measured in `u16` **elements** (not bytes): minimum
  /// `width * height * 4`.
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

  /// Attaches a planar **`u16`** luma output buffer.
  ///
  /// Luma is derived from RGB via BT.709 weights (by default).
  /// Length: minimum `width * height`.
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

impl<R> MonowhiteSink for MixedSinker<'_, Monowhite, R> {}

impl<R> PixelSink for MixedSinker<'_, Monowhite, R> {
  type Input<'i> = MonowhiteRow<'i>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    if let Some(stream) = self.luma_stream.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.luma_filter_stream.as_mut() {
      stream.reset();
    }
    self.resample_outputs = None;
    Ok(())
  }

  fn process(&mut self, row: Self::Input<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    let min_bytes = w.div_ceil(8);
    if row.data().len() < min_bytes {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Y,
        idx,
        min_bytes,
        row.data().len(),
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
      rgb_u16,
      rgba_u16,
      luma,
      luma_u16,
      hsv,
      plan,
      luma_stream,
      luma_filter_stream,
      resample_outputs,
      rgb_scratch,
      ..
    } = self;

    // Non-identity plan: same expand-then-resample oracle as Monoblack,
    // with Monowhite's inverted polarity baked into the luma kernel.
    if let Some(plan) = plan.as_ref() {
      let data = row.data();
      return mono_luma_resample(
        luma_stream,
        luma_filter_stream,
        resample_outputs,
        rgb,
        rgba,
        rgb_u16,
        rgba_u16,
        luma,
        luma_u16,
        hsv,
        rgb_scratch,
        w,
        plan,
        idx,
        use_simd,
        |scratch| row::monowhite_to_luma_row(data, scratch, w, use_simd),
      );
    }

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    if let Some(buf) = rgb.as_deref_mut() {
      row::monowhite_to_rgb_or_rgba_row::<false>(
        row.data(),
        &mut buf[one_plane_start * 3..one_plane_end * 3],
        w,
        use_simd,
      );
    }

    if let Some(buf) = rgba.as_deref_mut() {
      row::monowhite_to_rgb_or_rgba_row::<true>(
        row.data(),
        &mut buf[one_plane_start * 4..one_plane_end * 4],
        w,
        use_simd,
      );
    }

    if let Some(buf) = rgb_u16.as_deref_mut() {
      row::monowhite_to_rgb_u16_or_rgba_u16_row::<false>(
        row.data(),
        &mut buf[one_plane_start * 3..one_plane_end * 3],
        w,
        use_simd,
      );
    }

    if let Some(buf) = rgba_u16.as_deref_mut() {
      row::monowhite_to_rgb_u16_or_rgba_u16_row::<true>(
        row.data(),
        &mut buf[one_plane_start * 4..one_plane_end * 4],
        w,
        use_simd,
      );
    }

    if let Some(buf) = luma.as_deref_mut() {
      row::monowhite_to_luma_row(
        row.data(),
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    if let Some(buf) = luma_u16.as_deref_mut() {
      row::monowhite_to_luma_u16_row(
        row.data(),
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    if let Some(hsv) = hsv.as_mut() {
      let (h, s, v) = hsv.hsv();
      row::monowhite_to_hsv_row(
        row.data(),
        &mut h[one_plane_start..one_plane_end],
        &mut s[one_plane_start..one_plane_end],
        &mut v[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    Ok(())
  }
}
