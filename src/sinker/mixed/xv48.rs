//! Sinker impl for the packed XV48 source format — Tier 5 16-bit packed
//! YUV 4:4:4. Full output coverage: u8 + native-depth u16 RGB / RGBA +
//! u8 / u16 luma + u8 HSV.
//!
//! XV48 (FFmpeg `AV_PIX_FMT_XV48LE`) packs **four u16 slots per pixel**
//! (`[U, Y, V, X]`) with every channel using the full 16 bits (no MSB
//! shift — the full-depth sibling of XV36, which is 12-bit MSB-aligned).
//! The `X` prefix means the X slot is padding — it is read but always
//! discarded; RGBA outputs force α to the 16-bit opaque maximum. The
//! packed slice type is `&[u16]`, with `4 x width` u16 elements per row.
//! There is no chroma subsampling — every pixel carries its own
//! independent U / Y / V triplet (4:4:4).
//!
//! Outputs map to the sink's standard channels:
//! - `with_rgb` / `with_rgba` — packed YUV → RGB Q15 pipeline at
//!   `BITS = 16`, downshifted to u8; RGBA alpha is forced to `0xFF`
//!   (XV48 X slot is padding, not a real alpha channel).
//! - `with_rgb_u16` / `with_rgba_u16` — same pipeline at native
//!   16-bit depth (`[0, 65535]`); RGBA alpha is `0xFFFF` (16-bit max).
//!   The u16 path uses **i64 chroma** (Q15 sums overflow i32 at
//!   BITS=16/16), exactly like the AYUV64 16-bit sibling.
//! - `with_luma` — extracts the 16-bit Y values from each XV48
//!   quadruple and downshifts `>> 8` to u8.
//! - `with_luma_u16` — passes the 16-bit Y values straight through
//!   into u16 (full-depth, no shift).
//! - `with_hsv` — stages an internal RGB scratch (or the user's RGB
//!   buffer if attached) and runs the existing `rgb_to_hsv_row`
//!   kernel on the staged u8 RGB.
//!
//! When both u8 RGB and u8 RGBA outputs are requested, the RGBA plane
//! is derived from the just-computed u8 RGB row via
//! [`expand_rgb_to_rgba_row`] (Strategy A) instead of running a
//! second YUV→RGB kernel. The same Strategy A applies on the u16
//! path via [`expand_rgb_u16_to_rgba_u16_row::<16>`]. When only the
//! RGBA variant is wanted, the dedicated `_to_rgba_row` /
//! `_to_rgba_u16_row` kernel writes the output buffer directly
//! without staging RGB.

use super::{
  GeometryOverflow, InsufficientBuffer, MixedSinker, MixedSinkerError, RowIndexOutOfRange,
  RowShapeMismatch, RowSlice, check_dimensions_match, packed_yuv444_triple_resample,
  rgb_row_buf_or_scratch, rgba_plane_row_slice, rgba_u16_plane_row_slice,
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
    expand_rgb_to_rgba_row, expand_rgb_u16_to_rgba_u16_row, rgb_to_hsv_row, xv48_to_hsv_row,
    xv48_to_luma_row, xv48_to_luma_u16_row, xv48_to_rgb_row, xv48_to_rgb_u16_row, xv48_to_rgba_row,
    xv48_to_rgba_u16_row,
  },
  source::{Xv48, Xv48Row, Xv48Sink},
};

impl<'a, R, const BE: bool> MixedSinker<'a, Xv48<BE>, R> {
  /// Attaches a packed **8-bit** RGBA output buffer. Alpha is filled
  /// with constant `0xFF` (XV48 X slot is padding — not a real alpha
  /// channel).
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

  /// Attaches a packed **`u16`** RGB output buffer. Native 16-bit depth
  /// (`[0, 65535]`); length is measured in `u16` **elements**
  /// (`width x height x 3`).
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

  /// Attaches a packed **`u16`** RGBA output buffer. Native 16-bit depth
  /// (`[0, 65535]`); alpha element is `0xFFFF` (16-bit max). Length is
  /// measured in `u16` **elements** (`width x height x 4`).
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

  /// Attaches a native-depth **`u16`** luma output buffer. The 16-bit Y
  /// samples are extracted from each XV48 quadruple direct (no shift —
  /// 16-bit native), yielding values in `[0, 65535]`. Length is measured
  /// in `u16` **elements** (`width x height`).
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

impl<const BE: bool, R> Xv48Sink<BE> for MixedSinker<'_, Xv48<BE>, R> {}

impl<const BE: bool, R> PixelSink for MixedSinker<'_, Xv48<BE>, R> {
  type Input<'r> = Xv48Row<'r>;
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

  fn process(&mut self, row: Xv48Row<'_>) -> Result<(), Self::Error> {
    const BITS: u32 = 16;
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    // XV48 row = `width x 4` u16 elements (one quadruple per pixel).
    let packed_expected =
      w.checked_mul(4)
        .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
          w, h, 4,
        )))?;
    if row.packed().len() != packed_expected {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Xv48Packed,
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
    // marker selects the source decode wire for every conversion. Freeze +
    // sequence-check before staging, so a no-output sink stays a no-op and
    // an out-of-sequence row is rejected without allocating.
    if let Some(plan) = plan.as_ref() {
      let matrix = row.matrix();
      let full_range = row.full_range();
      let packed = row.packed();
      // Area plan with the native tier enabled (and the planar join it reuses
      // compiled in): de-pack each XV48 quad into separate host-native LOGICAL
      // u16 Y / U / V planes (4:4:4, full-width chroma — full 16-bit, no shift)
      // and bin those at output resolution, converting once per output row.
      // Xv48 has no filter-resample path (`packed_yuv444_triple_resample`
      // rejects a filter plan below), and the native tier is area-only, so the
      // native branch is gated off a filter plan. Otherwise (filter plan, or
      // `with_native(false)`) fall to the row-stage tail. The reused planar join
      // emits the native-depth `luma_u16` (the clamped binned Y), so attaching
      // `luma_u16` keeps the native route. The output set is frozen on the first
      // resampled row, so the route stays stable across a frame and the
      // mid-frame flip guard catches a `set_native` toggle.
      #[cfg(all(feature = "yuv-444-packed", feature = "yuv-planar"))]
      if !plan.kind().is_filter() {
        let need_output = luma.is_some()
          || luma_u16.is_some()
          || rgb.is_some()
          || rgba.is_some()
          || rgb_u16.is_some()
          || rgba_u16.is_some()
          || hsv.is_some();
        // The RFC #238 splice stage. This arm is the area branch (the
        // enclosing guard already excluded a filter plan), so `area_plan` is
        // true and the selector reproduces the former `*native` boolean
        // bit-for-bit (`cfg!` is true wherever this block compiles).
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
          // returns Ok on an output-bearing row. XV48 quad `[U, Y, V, X]`:
          // U at slot 0, Y at slot 1, V at slot 2, X (padding) at slot 3 —
          // `BE` selects the per-u16 wire decode; no shift (full 16-bit).
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
              for (i, dst) in y_dst.iter_mut().enumerate() {
                let raw = packed[i * 4 + 1];
                *dst = if BE {
                  u16::from_be(raw)
                } else {
                  u16::from_le(raw)
                };
              }
            },
            |u_dst, v_dst| {
              for (i, (u, v)) in u_dst.iter_mut().zip(v_dst.iter_mut()).enumerate() {
                let u_raw = packed[i * 4];
                let v_raw = packed[i * 4 + 2];
                *u = if BE {
                  u16::from_be(u_raw)
                } else {
                  u16::from_le(u_raw)
                };
                *v = if BE {
                  u16::from_be(v_raw)
                } else {
                  u16::from_le(v_raw)
                };
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
      // Row-stage tail. `packed_yuv444_triple_resample` rejects a filter plan
      // (Xv48 exposes no filter resampler), and otherwise runs the area
      // convert-then-bin path. Under the native tier, freeze the route to
      // row-stage on an accepted output-bearing row.
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
        |scratch| xv48_to_rgb_row(packed, scratch, w, matrix, full_range, use_simd, BE),
        |scratch| xv48_to_rgb_u16_row(packed, scratch, w, matrix, full_range, use_simd, BE),
        |scratch| xv48_to_luma_u16_row(packed, scratch, w, use_simd, BE),
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
        // Only freeze on an ACCEPTED output-bearing row. A filter plan was
        // rejected above (no route consumed), so guard the freeze off it too.
        if !plan.kind().is_filter() && frozen_native_route.is_none() && need_output {
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
    // (HSV-only routes through the direct `xv48_to_hsv_row` kernel, which needs
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

    // Luma u8 — extract 8-bit Y bytes from the XV48 plane via the
    // dedicated kernel (downshifts 16-bit Y >> 8 to u8).
    if let Some(buf) = luma.as_deref_mut() {
      xv48_to_luma_row(
        packed,
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
        BE,
      );
    }
    // Luma u16 — extract 16-bit Y values at native depth (no shift —
    // 16-bit native, written direct).
    if let Some(buf) = luma_u16.as_deref_mut() {
      xv48_to_luma_u16_row(
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
      xv48_to_rgba_u16_row(
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
      xv48_to_rgb_u16_row(
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
    // HSV-without-RGB-or-RGBA goes through the direct `xv48_to_hsv_row`
    // kernel — no source-width RGB scratch (the SIMD path stages a fixed
    // 64-pixel 8-bit RGB chunk internally). When RGB or RGBA is also
    // attached the RGB kernel runs anyway, so HSV derives off that buffer
    // for free and `need_u8_rgb_kernel` keeps it alive. The u16 RGB/RGBA
    // paths above already ran, so this HSV-direct early return is safe.
    // Resample row-stage HSV-only is a #263 follow-up — HSV stays correct
    // via the convert-once path.
    // `want_rgb` / `want_rgba` / `want_hsv` were resolved up front for the
    // atomicity preflight (#308).
    let want_hsv_direct = want_hsv && !want_rgb && !want_rgba;
    let need_u8_rgb_kernel = want_rgb || (want_hsv && want_rgba);

    if want_hsv_direct {
      let hsv = hsv.as_mut().expect("want_hsv_direct implies hsv attached");
      let (h, s, v) = hsv.hsv();
      xv48_to_hsv_row(
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
      xv48_to_rgba_row(
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
    xv48_to_rgb_row(
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
