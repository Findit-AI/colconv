//! Sinker impl for the packed V410 source format — Ship 12a (Tier 5
//! 10-bit packed YUV 4:4:4). Full output coverage: u8 + native-depth
//! u16 RGB / RGBA + u8 / u16 luma + u8 HSV.
//!
//! V410 packs **one pixel per 32-bit word** (`(V << 20) | (Y << 10) | U`)
//! with 10-bit channels and 2 bits of padding (unlike the MSB-aligned
//! u16 quadruple layout used by Y2xx formats). The packed slice type
//! is `&[u32]`, not `&[u16]`. There is no chroma subsampling — every
//! pixel carries its own independent U / Y / V triplet (4:4:4).
//!
//! Outputs map to the sink's standard channels:
//! - `with_rgb` / `with_rgba` — packed YUV → RGB Q15 pipeline at
//!   `BITS = 10`, downshifted to u8; RGBA alpha is forced to `0xFF`
//!   (V410 has no alpha channel).
//! - `with_rgb_u16` / `with_rgba_u16` — same pipeline at native
//!   10-bit depth, low-bit-packed in `u16` (`[0, 1023]`); RGBA alpha
//!   is `0x3FF` (10-bit max).
//! - `with_luma` — extracts the 10-bit Y values from each V410 word
//!   and downshifts `>> 2` to u8.
//! - `with_luma_u16` — extracts the 10-bit Y values at native depth,
//!   low-bit-packed in `u16` (`[0, 1023]`). Each 10-bit Y is read
//!   directly from bits `[19:10]` of the V410 word (no shift needed
//!   beyond the bit-field extraction), yielding values in `[0, 0x3FF]`.
//! - `with_hsv` — when HSV is the only u8 colour output (no `with_rgb`
//!   / `with_rgba`), the direct `v410_to_hsv_row` kernel computes HSV
//!   straight from the packed YUV with no source-width RGB scratch
//!   (#263). When RGB or RGBA is also attached, HSV derives from the
//!   already-staged u8 RGB row via `rgb_to_hsv_row` (the cheap path).
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
  GeometryOverflow, InsufficientBuffer, MixedSinker, MixedSinkerError, RowIndexOutOfRange,
  RowShapeMismatch, RowSlice, check_dimensions_match, packed_yuv444_triple_filter_resample,
  packed_yuv444_triple_resample, rgb_row_buf_or_scratch, rgba_plane_row_slice,
  rgba_u16_plane_row_slice,
};
// `NativeRouteChanged` is raised only by the native fast tier's route-flip
// guard, and `packed_yuv444_hb_process_native` exists only when the reused
// high-bit planar join is compiled in. Gated to the native tier's feature
// intersection.
#[cfg(all(feature = "yuv-444-packed", feature = "yuv-planar"))]
use super::{NativeRouteChanged, packed_yuv444_hb_process_native};
// The RFC #238 insertion-point selector decides the native-vs-row-stage
// splice; consulted only inside the native tier's `cfg`, so its import
// shares that intersection.
#[cfg(all(feature = "yuv-444-packed", feature = "yuv-planar"))]
use crate::resample::{AveragingDomain, InsertionContext, InsertionPoint, select_insertion_point};
use crate::{
  PixelSink,
  row::{
    expand_rgb_to_rgba_row, expand_rgb_u16_to_rgba_u16_row, rgb_to_hsv_row, v410_to_hsv_row,
    v410_to_luma_row, v410_to_luma_u16_row, v410_to_rgb_row, v410_to_rgb_u16_row, v410_to_rgba_row,
    v410_to_rgba_u16_row,
  },
  source::{V410, V410Row, V410Sink},
};

impl<'a, R, const BE: bool> MixedSinker<'a, V410<BE>, R> {
  /// Attaches a packed **8-bit** RGBA output buffer. Alpha is filled
  /// with constant `0xFF` (V410 has no alpha channel).
  ///
  /// Returns `Err(InsufficientRgbaBuffer)` if
  /// `buf.len() < width x height x 4`, or `Err(GeometryOverflow)` on
  /// 32‑bit targets when the product overflows.
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

  /// Attaches a packed **`u16`** RGB output buffer. 10-bit
  /// low-bit-packed (`[0, 1023]`); length is measured in `u16`
  /// **elements** (`width x height x 3`).
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

  /// Attaches a packed **`u16`** RGBA output buffer. 10-bit
  /// low-bit-packed (`[0, 1023]`); alpha element is `0x3FF` (10-bit
  /// max). Length is measured in `u16` **elements**
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

  /// Attaches a native-depth **`u16`** luma output buffer. The 10-bit
  /// Y samples are extracted from each V410 word at native depth,
  /// low-bit-packed in `u16` (`[0, 1023]`). Length is measured in
  /// `u16` **elements** (`width x height`).
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

impl<const BE: bool, R> V410Sink<BE> for MixedSinker<'_, V410<BE>, R> {}

impl<const BE: bool, R> PixelSink for MixedSinker<'_, V410<BE>, R> {
  type Input<'r> = V410Row<'r>;
  type Error = MixedSinkerError;

  #[cfg_attr(not(tarpaulin), inline(always))]
  fn kernel_matrix(&self) -> crate::KernelMatrix {
    self.kernel_matrix
  }
  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    // New frame: restart the row-stage streams (lazily created in
    // `process`) and drop the frozen output set.
    if let Some(stream) = self.rgb_stream.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.rgb_stream_u16.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.luma_stream_u16.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.rgb_filter_stream.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.rgb_filter_stream_u16.as_mut() {
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
    if let Some(native) = self.native_planar_u16.as_mut() {
      native.reset();
    }
    #[cfg(all(feature = "yuv-444-packed", feature = "yuv-planar"))]
    {
      self.frozen_native_route = None;
    }
    self.resample_outputs = None;
    Ok(())
  }

  fn process(&mut self, row: V410Row<'_>) -> Result<(), Self::Error> {
    const BITS: u32 = 10;
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    // V410 row = `width` u32 elements (one pixel per word).
    let packed_expected = w;
    if row.packed().len() != packed_expected {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::V410Packed,
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
      luma_scratch_u16,
      plan,
      rgb_stream,
      rgb_stream_u16,
      luma_stream_u16,
      rgb_filter_stream,
      rgb_filter_stream_u16,
      luma_filter_stream_u16,
      resample_outputs,
      #[cfg(all(feature = "yuv-444-packed", feature = "yuv-planar"))]
      native,
      #[cfg(all(feature = "yuv-444-packed", feature = "yuv-planar"))]
      native_planar_u16,
      #[cfg(all(feature = "yuv-444-packed", feature = "yuv-planar"))]
      packed_444_y_full_u16,
      #[cfg(all(feature = "yuv-444-packed", feature = "yuv-planar"))]
      packed_444_u_full_u16,
      #[cfg(all(feature = "yuv-444-packed", feature = "yuv-planar"))]
      packed_444_v_full_u16,
      #[cfg(all(feature = "yuv-444-packed", feature = "yuv-planar"))]
      frozen_native_route,
      ..
    } = self;

    // Non-identity plan: feed the shared three-stream tail. `BE` from the
    // marker selects the source decode wire for every conversion. The span
    // kind picks the engine — area binning or signed-coefficient filter
    // (both convert the YUV to RGB with the same closures and resample in
    // RGB space, so filter colour equals the RGB filter of the converted
    // pixels and matches area up to the kernel). Freeze + sequence-check
    // before staging, so a no-output sink stays a no-op and an
    // out-of-sequence row is rejected without allocating.
    if let Some(plan) = plan.as_ref() {
      let matrix = row.matrix();
      let full_range = row.full_range();
      let packed = row.packed();
      // A `Filter` plan routes to the filter resampler (the native fast tier is
      // area-only); branched FIRST, before the area native-route machinery.
      if let crate::resample::SpanKind::Filter = plan.kind() {
        return packed_yuv444_triple_filter_resample::<BITS>(
          rgb_filter_stream,
          rgb_filter_stream_u16,
          luma_filter_stream_u16,
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
          |scratch| v410_to_rgb_row(packed, scratch, w, matrix, full_range, use_simd, BE),
          |scratch| v410_to_rgb_u16_row(packed, scratch, w, matrix, full_range, use_simd, BE),
          |scratch| v410_to_luma_u16_row(packed, scratch, w, use_simd, BE),
        );
      }
      // Area plan. When the native tier is enabled (and the planar join it
      // reuses is compiled in), bit-extract each V410 word into separate
      // host-native LOGICAL u16 Y / U / V planes (4:4:4, full-width chroma) and
      // bin those at output resolution, converting once per output row.
      // Otherwise (or under `with_native(false)`) feed the row-stage tail. The
      // reused planar join emits the native-depth `luma_u16` (the clamped binned
      // Y), so attaching `luma_u16` no longer forces row-stage; the route
      // depends only on `with_native`. The output set is frozen on the first
      // resampled row, so the route stays stable across a frame and the mid-frame
      // flip guard catches a `set_native` toggle.
      #[cfg(all(feature = "yuv-444-packed", feature = "yuv-planar"))]
      {
        let need_output = luma.is_some()
          || luma_u16.is_some()
          || rgb.is_some()
          || rgba.is_some()
          || rgb_u16.is_some()
          || rgba_u16.is_some()
          || hsv.is_some();
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
          // returns Ok on an output-bearing row. V410 word layout: bits[9:0] U,
          // bits[19:10] Y, bits[29:20] V, bits[31:30] padding — `BE` selects the
          // u32 wire decode before the bit-field extract.
          packed_yuv444_hb_process_native::<BITS>(
            plan,
            native_planar_u16,
            packed_444_y_full_u16,
            packed_444_u_full_u16,
            packed_444_v_full_u16,
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
            |y_dst| {
              for (dst, &raw) in y_dst.iter_mut().zip(packed) {
                let word = if BE {
                  u32::from_be(raw)
                } else {
                  u32::from_le(raw)
                };
                *dst = ((word >> 10) & 0x3FF) as u16;
              }
            },
            |u_dst, v_dst| {
              for ((u, v), &raw) in u_dst.iter_mut().zip(v_dst.iter_mut()).zip(packed) {
                let word = if BE {
                  u32::from_be(raw)
                } else {
                  u32::from_le(raw)
                };
                *u = (word & 0x3FF) as u16;
                *v = ((word >> 20) & 0x3FF) as u16;
              }
            },
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
        |scratch| v410_to_rgb_row(packed, scratch, w, matrix, full_range, use_simd, BE),
        |scratch| v410_to_rgb_u16_row(packed, scratch, w, matrix, full_range, use_simd, BE),
        |scratch| v410_to_luma_u16_row(packed, scratch, w, use_simd, BE),
      )?;
      #[cfg(all(feature = "yuv-444-packed", feature = "yuv-planar"))]
      {
        let need_output = luma.is_some()
          || luma_u16.is_some()
          || rgb.is_some()
          || rgba.is_some()
          || rgb_u16.is_some()
          || rgba_u16.is_some()
          || hsv.is_some();
        if frozen_native_route.is_none() && need_output {
          *frozen_native_route = Some(false);
        }
      }
      return Ok(());
    }

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;
    let packed = row.packed();

    // Resolve the output set up front so the atomicity preflight below runs
    // before any output row is written.
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();

    // Atomicity preflight (#308, cf. the crate's #180 resample fix and the
    // high-bit semi-planar sibling): reserve the only growable row scratch this
    // identity row can touch — the u8 RGB row buffer — BEFORE any output row is
    // written (the luma / luma_u16 planes below, then the u16 RGB / RGBA
    // fan-out), so an allocator refusal returns a typed `AllocationFailed`
    // leaving the output frame untouched rather than partially mutated. The
    // luma / luma_u16 and u16 RGB / RGBA outputs write straight into their
    // caller buffers and never grow a scratch. `rgb_row_buf_or_scratch`'s
    // allocating (rgb = None) arm is reached exactly when a colour decode needs
    // an RGB row but no caller RGB buffer is borrowable — for this
    // convert-once-then-derive path that is `want_hsv && want_rgba && !want_rgb`
    // (HSV-only routes through the direct `v410_to_hsv_row` kernel, which needs
    // no RGB scratch). The later decode reuses the already-sized buffer, so the
    // default path is byte-identical; only the failure-path ordering changes.
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

    // Luma u8 — extract 8-bit Y bytes from the V410 plane via the
    // dedicated kernel (downshifts 10-bit Y >> 2 to u8).
    if let Some(buf) = luma.as_deref_mut() {
      v410_to_luma_row(
        packed,
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
        BE,
      );
    }
    // Luma u16 — extract 10-bit Y values at native depth (low-bit-packed
    // in u16, range [0, 0x3FF]).
    if let Some(buf) = luma_u16.as_deref_mut() {
      v410_to_luma_u16_row(
        packed,
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
        BE,
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
      v410_to_rgba_u16_row(
        packed,
        rgba_u16_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
        BE,
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
      v410_to_rgb_u16_row(
        packed,
        rgb_u16_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
        BE,
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
    // HSV-without-RGB-or-RGBA goes through the direct `v410_to_hsv_row`
    // kernel (no source-width RGB scratch). When RGB or RGBA is *also*
    // attached the RGB kernel runs anyway, so HSV derives off that buffer
    // for free (the cheap path) and `need_u8_rgb_kernel` keeps it alive.
    // (Resample row-stage HSV stays correct via the convert-once path in
    // the plan branch above.)
    // `want_rgb` / `want_rgba` / `want_hsv` were resolved up front for the
    // atomicity preflight (#308).
    let want_hsv_direct = want_hsv && !want_rgb && !want_rgba;
    let need_u8_rgb_kernel = want_rgb || (want_hsv && want_rgba);

    if want_hsv_direct {
      let hsv = hsv.as_mut().expect("want_hsv_direct implies hsv attached");
      let (h, s, v) = hsv.hsv();
      v410_to_hsv_row(
        packed,
        &mut h[one_plane_start..one_plane_end],
        &mut s[one_plane_start..one_plane_end],
        &mut v[one_plane_start..one_plane_end],
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
        BE,
      );
      return Ok(());
    }

    // Standalone u8 RGBA fast path — no RGB / HSV requested. Run the
    // dedicated RGBA kernel directly into the output buffer; avoids
    // both the scratch allocation and the RGB→RGBA expand pass.
    if want_rgba && !need_u8_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      v410_to_rgba_row(
        packed,
        rgba_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
        BE,
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
    v410_to_rgb_row(
      packed,
      rgb_row,
      w,
      row.matrix(),
      row.full_range(),
      use_simd,
      BE,
    );

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
    // row instead of running a second YUV→RGB kernel.
    if let Some(buf) = rgba.as_deref_mut() {
      let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
    }

    Ok(())
  }
}
