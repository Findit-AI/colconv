//! Sinker impl for the packed VUYX source format — Ship 12c (Tier 5
//! 8-bit packed YUV 4:4:4 with padding α byte).
//!
//! VUYX (FFmpeg `AV_PIX_FMT_VUYX`) packs **four u8 bytes per pixel**
//! (`[V, U, Y, X]`). The X byte is **padding** — not real source alpha.
//! RGBA outputs always force α to `0xFF`; the padding byte is ignored.
//! The packed slice type is `&[u8]`, with `4 x width` byte elements per
//! row. There is no chroma subsampling — every pixel carries its own
//! independent V / U / Y triplet (4:4:4).
//!
//! Outputs map to the sink's standard channels:
//! - `with_rgb` — packed YUV → RGB 8-bit pipeline; padding discarded.
//! - `with_rgba` — packed YUV → RGBA 8-bit pipeline; **α is forced to
//!   `0xFF`** (the X byte is padding, never real alpha).
//! - `with_luma` — extracts the Y byte at offset 2 of each pixel
//!   directly (no YUV→RGB pipeline).
//! - `with_luma_u16` — zero-extends the Y byte to u16
//!   (`out[x] = Y_byte as u16`).
//! - `with_hsv` — stages u8 RGB into the user's RGB buffer (if
//!   attached) or a scratch buffer, then runs `rgb_to_hsv_row`.

use super::{
  GeometryOverflow, InsufficientBuffer, MixedSinker, MixedSinkerError, RowIndexOutOfRange,
  RowShapeMismatch, RowSlice, check_dimensions_match, packed_yuv444_triple_resample,
  packed_yuva444_filter_resample, rgb_row_buf_or_scratch, rgba_plane_row_slice,
};
// `NativeRouteChanged` is raised only by the native fast tier's route-flip
// guard, and `packed_vuyx_process_native` exists only when the reused 8-bit
// planar join is compiled in. Gated to the native tier's feature intersection.
#[cfg(all(feature = "yuv-444-packed", feature = "yuv-planar"))]
use super::{NativeRouteChanged, packed_vuyx_process_native};
// The RFC #238 insertion-point selector decides the native-vs-row-stage
// splice; consulted only inside the native tier's `cfg`, so its import
// shares that intersection.
#[cfg(all(feature = "yuv-444-packed", feature = "yuv-planar"))]
use crate::resample::{AveragingDomain, InsertionContext, InsertionPoint, select_insertion_point};
use crate::{
  PixelSink,
  row::{
    expand_rgb_to_rgba_row, rgb_to_hsv_row, vuyx_to_hsv_row, vuyx_to_luma_row,
    vuyx_to_luma_u16_row, vuyx_to_rgb_row, vuyx_to_rgba_row,
  },
  source::{Vuyx, VuyxRow, VuyxSink},
};

impl<'a, R> MixedSinker<'a, Vuyx, R> {
  /// Attaches a **`u16`** luma output buffer. Y bytes from the packed VUYX
  /// `[V, U, Y, X]` layout are zero-extended to u16
  /// (`out[x] = Y_byte as u16`). Length in u16 **elements**
  /// (`width x height`).
  ///
  /// Returns `Err(InsufficientLumaU16Buffer)` if `buf.len() < width x height`,
  /// or `Err(GeometryOverflow)` on 32-bit targets.
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

  /// Attaches a packed **8-bit** RGBA output buffer. When VUYX is the
  /// source, the per-pixel alpha byte is always forced to `0xFF` —
  /// the X (padding) byte in the source is never read as alpha.
  ///
  /// Returns `Err(InsufficientRgbaBuffer)` if
  /// `buf.len() < width x height x 4`, or `Err(GeometryOverflow)` on
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
    let expected = self.frame_elems(4)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::InsufficientRgbaBuffer(
        InsufficientBuffer::new(expected, buf.len()),
      ));
    }
    self.rgba = Some(buf);
    Ok(self)
  }
}

impl<R> VuyxSink for MixedSinker<'_, Vuyx, R> {}

impl<R> PixelSink for MixedSinker<'_, Vuyx, R> {
  type Input<'r> = VuyxRow<'r>;
  type Error = MixedSinkerError;

  #[cfg_attr(not(tarpaulin), inline(always))]
  fn kernel_matrix(&self) -> crate::KernelMatrix {
    self.kernel_matrix
  }
  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    // New frame: restart the row-stage streams (lazily created in
    // `process`, area or filter kind) and drop the frozen output set.
    // `Vuyx` exposes no u16 colour outputs, so its u16 colour streams are
    // never created; resetting them unconditionally is a harmless no-op. The
    // area arm uses the 3-channel colour streams; the filter arm uses the
    // 4-channel `rgba_filter_stream` (the padding byte filters as a constant
    // opaque α) and the native-Y `luma_filter_stream_u16`.
    if let Some(stream) = self.rgb_stream.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.rgb_stream_u16.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.luma_stream_u16.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.rgba_filter_stream.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.luma_filter_stream_u16.as_mut() {
      stream.reset();
    }
    // New frame: restart the native join and clear the per-frame frozen
    // native/row-stage route so the next frame may pick either tier; a
    // mid-frame flip stays rejected. Gated to the native tier's feature
    // intersection (the planar join the native tier reuses is compiled only
    // under `yuv-planar`).
    #[cfg(all(feature = "yuv-444-packed", feature = "yuv-planar"))]
    if let Some(native) = self.native_planar.as_mut() {
      native.reset();
    }
    #[cfg(all(feature = "yuv-444-packed", feature = "yuv-planar"))]
    {
      self.frozen_native_route = None;
    }
    self.resample_outputs = None;
    Ok(())
  }

  fn process(&mut self, row: VuyxRow<'_>) -> Result<(), Self::Error> {
    const BITS: u32 = 8;
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    // VUYX row = `width x 4` bytes (one quadruple per pixel).
    let packed_expected =
      w.checked_mul(4)
        .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
          w, h, 4,
        )))?;
    if row.packed().len() != packed_expected {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::VuyxPacked,
        idx,
        packed_expected,
        row.packed().len(),
      )));
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange(
        RowIndexOutOfRange::new(idx, self.height),
      ));
    }

    // Non-identity plan: `Vuyx` has NO alpha (the X byte is padding,
    // forced opaque). The span kind picks the engine:
    //
    // - `Area` bins through the no-alpha three-stream tail exactly like the
    //   no-alpha packed 4:4:4 YUV siblings (`V30X` / `V410` / `Xv36`) — u8
    //   colour bins the converted u8 RGB row, native Y bins the
    //   de-interleaved Y plane (the colour outputs force α opaque, the
    //   padding byte is never read).
    // - `Filter` runs the signed-coefficient filter on the converted RGBA
    //   via the shared packed-YUVA filter tail. `vuyx_to_rgba_row` writes a
    //   constant `0xFF` α plane, and a constant channel filters to itself
    //   (partition of unity), so the 4-channel filter reproduces the
    //   no-alpha 3-channel result with α pinned opaque — no separate
    //   padding-byte filter path is needed. Straight alpha (`Vuyx` has no
    //   alpha mode).
    //
    // `Vuyx` exposes no u16 colour outputs, so the u16 colour resampling is
    // never active (`rgb_u16` / `rgba_u16` stay `None`) and the
    // `convert_*_u16` closure is therefore never invoked on either arm.
    if let Some(plan) = self.plan.as_ref() {
      let matrix = row.matrix();
      let full_range = row.full_range();
      let packed = row.packed();
      let Self {
        rgb,
        rgba,
        rgb_u16,
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
        rgb_stream,
        rgb_stream_u16,
        luma_stream_u16,
        rgba_filter_stream,
        rgba_filter_stream_u16,
        luma_filter_stream_u16,
        resample_outputs,
        #[cfg(all(feature = "yuv-444-packed", feature = "yuv-planar"))]
        native,
        #[cfg(all(feature = "yuv-444-packed", feature = "yuv-planar"))]
        native_planar,
        #[cfg(all(feature = "yuv-444-packed", feature = "yuv-planar"))]
        packed_444_y_full,
        #[cfg(all(feature = "yuv-444-packed", feature = "yuv-planar"))]
        packed_444_u_full,
        #[cfg(all(feature = "yuv-444-packed", feature = "yuv-planar"))]
        packed_444_v_full,
        #[cfg(all(feature = "yuv-444-packed", feature = "yuv-planar"))]
        frozen_native_route,
        ..
      } = self;
      // A `Filter` plan routes straight to the signed-coefficient filter tail
      // (the native fast tier is area-only); branched FIRST, before the
      // area native-route machinery below.
      if let crate::resample::SpanKind::Filter = plan.kind() {
        return packed_yuva444_filter_resample::<BITS, false, false>(
          rgba_filter_stream,
          rgba_filter_stream_u16,
          // Packed `Vuyx` never uses the u8 native-Y luma stream
          // (`NATIVE_LUMA_U8 = false`); pass an inert slot.
          &mut None,
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
          // Packed `Vuyx` routes luma through `deinterleave_y` + the u16
          // stream (no contiguous native-Y plane), so the u8-luma input and
          // its de-interleave scratch are unused.
          &[],
          None,
          // α forced opaque (`0xFF`) by `vuyx_to_rgba_row` — the padding
          // byte is never read as alpha.
          |dst| vuyx_to_rgba_row(packed, dst, w, matrix, full_range, use_simd),
          // `Vuyx` has no u16 colour outputs, so this closure is never called.
          |_dst: &mut [u16]| {},
          |dst| vuyx_to_luma_u16_row(packed, dst, w, use_simd),
          // u16-luma path, so this u8 de-interleave is never called.
          |_dst: &mut [u8]| {},
        );
      }
      // Area plan. When the native tier is enabled (and the planar join it
      // reuses is compiled in), de-pack the interleaved `V U Y X` row into
      // Y / U / V scratch (4:4:4, full-width chroma) and bin those planes at
      // output resolution, converting once per output row (Vuyx: V at 0 / U at
      // 1 / Y at 2 / X padding at 3). Otherwise (or under `with_native(false)`)
      // take the row-stage tier. The output set is frozen on the first
      // resampled row, so the native/row-stage route stays stable across a
      // frame and the mid-frame flip guard catches a `set_native` toggle.
      #[cfg(all(feature = "yuv-444-packed", feature = "yuv-planar"))]
      {
        let need_output =
          luma.is_some() || luma_u16.is_some() || rgb.is_some() || rgba.is_some() || hsv.is_some();
        // The RFC #238 splice stage. A filter plan already returned above, so
        // `area_plan` is true and the selector reproduces the former `*native`
        // boolean bit-for-bit (`cfg!` is true wherever this block compiles).
        let take_native = matches!(
          select_insertion_point(
            AveragingDomain::Encoded,
            InsertionContext {
              native_eligible: cfg!(all(feature = "yuv-444-packed", feature = "yuv-planar")),
              with_native: *native,
              area_plan: true,
            },
          ),
          InsertionPoint::NativeCodes
        );
        // Reject a mid-frame native/row-stage route flip BEFORE either tier's
        // dispatch (the #186 CHECK-before / SET-after template).
        if need_output
          && let Some(frozen) = *frozen_native_route
          && frozen != take_native
        {
          return Err(MixedSinkerError::NativeRouteChanged(
            NativeRouteChanged::new(idx),
          ));
        }
        if take_native {
          // Dispatch first; freeze the route to native ONLY after the call
          // returns Ok on an output-bearing row.
          packed_vuyx_process_native(
            plan,
            native_planar,
            packed_444_y_full,
            packed_444_u_full,
            packed_444_v_full,
            resample_outputs,
            rgb,
            rgba,
            luma,
            luma_u16,
            hsv,
            rgb_scratch,
            packed,
            matrix,
            full_range,
            idx,
            w,
            h,
            use_simd,
          )?;
          if frozen_native_route.is_none() && need_output {
            *frozen_native_route = Some(true);
          }
          return Ok(());
        }
      }
      // Row-stage area tail (the only route under a `yuv-444-packed`-solo build
      // where the planar join is absent). Same CHECK-before / SET-after split.
      packed_yuv444_triple_resample::<BITS>(
        rgb_stream,
        rgb_stream_u16,
        luma_stream_u16,
        resample_outputs,
        rgb,
        rgba,
        rgb_u16,
        rgba_u16,
        luma,
        luma_u16,
        hsv,
        rgb_scratch,
        rgb_scratch_u16,
        luma_scratch_u16,
        w,
        plan,
        idx,
        use_simd,
        matrix,
        full_range,
        |scratch| vuyx_to_rgb_row(packed, scratch, w, matrix, full_range, use_simd),
        // `Vuyx` has no u16 colour outputs, so `need_u16_color` is always
        // false in the tail and this closure is never called.
        |_scratch: &mut [u16]| {},
        |scratch| vuyx_to_luma_u16_row(packed, scratch, w, use_simd),
      )?;
      #[cfg(all(feature = "yuv-444-packed", feature = "yuv-planar"))]
      {
        let need_output =
          luma.is_some() || luma_u16.is_some() || rgb.is_some() || rgba.is_some() || hsv.is_some();
        if frozen_native_route.is_none() && need_output {
          *frozen_native_route = Some(false);
        }
      }
      return Ok(());
    }

    let Self {
      rgb,
      rgba,
      luma,
      luma_u16,
      hsv,
      rgb_scratch,
      ..
    } = self;
    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;
    let packed = row.packed();

    // Output mode resolution — hoisted ABOVE the luma writes so the atomicity
    // preflight runs ahead of any output mutation.
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();

    // Atomicity preflight (#308, cf. the crate's #180 resample fix and the
    // merged packed 4:2:2 / planar / semi-planar siblings): reserve the only
    // fallible row scratch this identity row can grow BEFORE any output row
    // (luma / luma_u16 included) is written, so an allocator refusal returns a
    // typed `AllocationFailed` leaving the output frame untouched rather than
    // partially mutated. The sole growable scratch is the RGB row buffer, taken
    // exactly when the colour decode needs an RGB row but no caller RGB buffer
    // is borrowable — `want_hsv && want_rgba && !want_rgb` (the
    // `rgb_row_buf_or_scratch` scratch arm; an attached RGB buffer is borrowed
    // and never allocates, and standalone `with_rgba` uses the dedicated
    // `vuyx_to_rgba_row` kernel — α forced opaque). The later `need_rgb_kernel`
    // decode reuses the already-sized buffer.
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

    // Luma u8 — extract Y byte (offset 2 in each VUYX quadruple) directly.
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

    // Luma u16 — extract Y bytes and zero-extend to u16.
    if let Some(buf) = luma_u16.as_deref_mut() {
      vuyx_to_luma_u16_row(
        packed,
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    // ===== u8 RGB / RGBA / HSV path (Strategy A for VUYX) =====
    // HSV-without-RGB-or-RGBA goes through the direct `vuyx_to_hsv_row`
    // kernel — no source-width RGB scratch (the SIMD path stages a fixed
    // 64-pixel 8-bit RGB chunk internally). When RGB or RGBA is also
    // attached the RGB kernel runs anyway, so HSV derives off that buffer
    // for free and `need_rgb_kernel` keeps it alive. Resample row-stage
    // HSV-only is a #263 follow-up — HSV stays correct via the
    // convert-once path.
    let want_hsv_direct = want_hsv && !want_rgb && !want_rgba;
    let need_rgb_kernel = want_rgb || (want_hsv && want_rgba);

    if want_hsv_direct {
      let hsv = hsv.as_mut().expect("want_hsv_direct implies hsv attached");
      let (h, s, v) = hsv.hsv();
      vuyx_to_hsv_row(
        packed,
        &mut h[one_plane_start..one_plane_end],
        &mut s[one_plane_start..one_plane_end],
        &mut v[one_plane_start..one_plane_end],
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
      return Ok(());
    }

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
