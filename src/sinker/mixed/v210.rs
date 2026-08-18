//! Sinker impl for the packed v210 source format — Ship 11a (Tier 4
//! 10-bit pro-broadcast SDI). Full output coverage: u8 + native-depth
//! u16 RGB / RGBA / luma + u8 HSV.
//!
//! v210 packs 12 x 10-bit samples per 16-byte word = 6 pixels (4:2:2
//! with 6 Y + 3 Cb + 3 Cr per word). The sinker's configured width
//! must be **even** (4:2:2 chroma pair) — partial last words (widths
//! not divisible by 6, e.g. 720p = 1280) are supported. Odd widths
//! surface as [`MixedSinkerError::WidthAlignment`] (with
//! [`WidthAlignmentRequirement::Even`]) before any kernel runs,
//! preserving the no-panic contract.
//!
//! [`WidthAlignmentRequirement::Even`]: super::WidthAlignmentRequirement::Even
//!
//! Outputs map to the sink's standard channels:
//! - `with_rgb` / `with_rgba` — packed YUV → RGB Q15 pipeline at
//!   `BITS = 10`, downshifted to u8; RGBA alpha is forced to `0xFF`
//!   (v210 has no alpha channel).
//! - `with_rgb_u16` / `with_rgba_u16` — same pipeline at native
//!   10-bit depth, low-bit-packed in `u16`; RGBA alpha is `1023`.
//! - `with_luma` — extracts the 6 Y values per word and downshifts
//!   `>> 2` to u8.
//! - `with_luma_u16` — extracts the 10-bit Y values into u16
//!   (low-bit-packed). Tier 4 is the first consumer of this API.
//! - `with_hsv` — when HSV is the only u8 colour output (no `with_rgb`
//!   / `with_rgba`), the direct `v210_to_hsv_row_endian` kernel computes
//!   HSV straight from the packed YUV with no source-width RGB scratch
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
  RowShapeMismatch, RowSlice, WidthAlignment, check_dimensions_match,
  packed_yuv422_triple_resample, rgb_row_buf_or_scratch, rgba_plane_row_slice,
  rgba_u16_plane_row_slice,
};
// `NativeRouteChanged` is raised only by the native fast tier's route-flip
// guard, and `v210_process_native` exists only when the reused high-bit planar
// join is compiled in. The horizontal chroma-siting machinery
// (`chroma_422_center_sited_h` + the centered full-width reconstruct through
// `v210_center_reconstruct` + the 4:4:4 decode) likewise needs the planar
// reconstruction stage. Gated to the native tier's feature intersection.
#[cfg(all(feature = "v210", feature = "yuv-planar"))]
use super::{
  ChromaSitingChanged, NativeRouteChanged, chroma_422_center_sited_h,
  planar_8bit::YUV422P_CENTERED_H_PHASE, resample_preflight_check_only, v210_center_reconstruct,
  v210_process_native,
};
// The insertion-point selector decides the native-vs-row-stage splice; it is only
// consulted inside the native tier's `cfg`, so its import shares that intersection.
#[cfg(all(feature = "v210", feature = "yuv-planar"))]
use crate::resample::{AveragingDomain, InsertionContext, InsertionPoint, select_insertion_point};
// The centered horizontal siting reconstructs full-width chroma then decodes via
// the planar 4:4:4 kernels; co-sited keeps the fused packed decode below.
#[cfg(all(feature = "v210", feature = "yuv-planar"))]
use crate::row::{yuv444p10_to_rgb_row_endian, yuv444p10_to_rgb_u16_row_endian};
use crate::{
  PixelSink,
  row::{
    expand_rgb_to_rgba_row, expand_rgb_u16_to_rgba_u16_row, rgb_to_hsv_row, v210_to_hsv_row_endian,
    v210_to_luma_row_endian, v210_to_luma_u16_row_endian, v210_to_rgb_row_endian,
    v210_to_rgb_u16_row_endian, v210_to_rgba_row_endian, v210_to_rgba_u16_row_endian,
  },
  source::{V210, V210Row, V210Sink},
};

impl<'a, R, const BE: bool> MixedSinker<'a, V210<BE>, R> {
  /// Attaches a packed **8-bit** RGBA output buffer. Alpha is filled
  /// with constant `0xFF` (v210 has no alpha channel).
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
  /// low-bit-packed (`[0, 1023]`); alpha element is `1023`. Length
  /// is measured in `u16` **elements** (`width x height x 4`).
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

  /// Attaches a native-depth **`u16`** luma output buffer. v210 is
  /// the first consumer of this Tier 4 API: the 10-bit Y samples
  /// are extracted directly out of the v210 word packing into the
  /// caller's `u16` buffer (low-bit-packed, `[0, 1023]`). Length
  /// is measured in `u16` **elements** (`width x height`).
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

impl<R, const BE: bool> V210Sink<BE> for MixedSinker<'_, V210<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, V210<BE>, R> {
  type Input<'r> = V210Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    if !self.width.is_multiple_of(2) {
      return Err(MixedSinkerError::WidthAlignment(WidthAlignment::odd(
        self.width,
      )));
    }
    // New frame: restart the three row-stage streams (lazily created in
    // `process`, so a direct-`process` caller that skips `begin_frame`
    // still gets a correctly initialized first frame) and drop the frozen
    // output set.
    if let Some(stream) = self.luma_stream_u16.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.rgb_stream.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.rgb_stream_u16.as_mut() {
      stream.reset();
    }
    // New frame: restart the native join and clear the per-frame frozen
    // native/row-stage route so the next frame may pick either tier; a
    // mid-frame flip stays rejected. Gated to the native tier's feature
    // intersection (the planar join the native tier reuses is compiled only
    // under `yuv-planar`).
    #[cfg(all(feature = "v210", feature = "yuv-planar"))]
    if let Some(native) = self.native_planar_u16.as_mut() {
      native.reset();
    }
    #[cfg(all(feature = "v210", feature = "yuv-planar"))]
    {
      self.frozen_native_route = None;
      // Clear the per-frame frozen 4:2:2 chroma siting so the next frame may pick
      // either phase; a mid-frame flip stays rejected.
      self.frozen_chroma_centered = None;
    }
    self.resample_outputs = None;
    Ok(())
  }

  fn process(&mut self, row: V210Row<'_>) -> Result<(), Self::Error> {
    const BITS: u32 = 10;
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if !w.is_multiple_of(2) {
      return Err(MixedSinkerError::WidthAlignment(WidthAlignment::odd(w)));
    }

    let packed_expected =
      w.div_ceil(6)
        .checked_mul(16)
        .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
          w, h, 16,
        )))?;
    if row.v210().len() != packed_expected {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::V210Packed,
        idx,
        packed_expected,
        row.v210().len(),
      )));
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange(
        RowIndexOutOfRange::new(idx, self.height),
      ));
    }

    // Chroma siting drives the horizontal chroma phase; `Copy`, so read it out
    // before the field split-borrow below.
    #[cfg(all(feature = "v210", feature = "yuv-planar"))]
    let chroma_location = self.chroma_location.clone();

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
      rgb_stream,
      rgb_stream_u16,
      luma_stream_u16,
      resample_outputs,
      plan,
      #[cfg(all(feature = "v210", feature = "yuv-planar"))]
      native,
      #[cfg(all(feature = "v210", feature = "yuv-planar"))]
      native_planar_u16,
      #[cfg(all(feature = "v210", feature = "yuv-planar"))]
      v210_y_full,
      #[cfg(all(feature = "v210", feature = "yuv-planar"))]
      v210_u_half,
      #[cfg(all(feature = "v210", feature = "yuv-planar"))]
      v210_v_half,
      #[cfg(all(feature = "v210", feature = "yuv-planar"))]
      frozen_native_route,
      #[cfg(all(feature = "v210", feature = "yuv-planar"))]
      frozen_chroma_centered,
      #[cfg(all(feature = "v210", feature = "yuv-planar"))]
      chroma_full_u16,
      ..
    } = self;
    let packed = row.v210();

    // Non-identity plan. V210 is area-only at the row stage (no filter route),
    // so a `Filter` plan falls straight through to `packed_yuv422_triple_resample`,
    // which rejects it as `unsupported_filter` before any work. For an `Area`
    // plan: when the native tier is enabled (and the planar join it reuses is
    // compiled in), bin the native Y / U / V planes at output resolution and
    // convert once per output row, de-packing the V210 word packing into
    // wrapper-owned logical scratch first; otherwise (under `with_native(false)`,
    // or a `v210`-solo build where the planar join is absent) feed the shared
    // area triple-resample tail, which bins the three native-precision
    // conversions (u8 colour, native u16 colour, native Y) directly. The reused
    // planar join now emits the native-depth `luma_u16` (the clamped binned Y),
    // so attaching `luma_u16` no longer forces row-stage — the route depends only
    // on `with_native`. The output set is frozen on the first resampled row, so
    // the native/row-stage route stays stable across a frame and the mid-frame
    // flip guard catches a `set_native` toggle.
    if let Some(plan) = plan.as_ref() {
      let matrix = row.matrix();
      let full_range = row.full_range();
      // Horizontal chroma siting for the packed 10-bit V210, mirroring the packed
      // high-bit Y2xx twin on the word de-pack. The centered group (`Center` /
      // `Top` / `Bottom`, `chroma_422_center_sited_h`) samples chroma at `+0.25`
      // chroma-sample; the co-sited / unspecified group is phase 0 (byte-identical
      // to the pre-siting resample). The native fast tier folds the phase into the
      // `area_chroma_422` chroma weights; the row-stage colour tier reconstructs
      // full-width chroma (de-pack + phase-0.5 upsample) and decodes 4:4:4 via the
      // `yuv444p10` full-chroma kernels — the co-sited arm keeps the fused
      // `v210_*` half-chroma decode. V210 is area-only, so a filter plan never
      // reconstructs; it falls to the co-sited tail's `unsupported_filter` reject.
      #[cfg(all(feature = "v210", feature = "yuv-planar"))]
      let center_sited = chroma_422_center_sited_h(&chroma_location);
      #[cfg(all(feature = "v210", feature = "yuv-planar"))]
      let chroma_h_phase = if center_sited {
        YUV422P_CENTERED_H_PHASE
      } else {
        0.0
      };
      // Only the colour tier reconstructs full-width chroma for the centered
      // decode; a luma-only centered row bins native Y unchanged (siting is a
      // chroma-only property).
      #[cfg(all(feature = "v210", feature = "yuv-planar"))]
      let want_color =
        rgb.is_some() || rgba.is_some() || hsv.is_some() || rgb_u16.is_some() || rgba_u16.is_some();
      // Whether this call carries any output a tier ACCEPTS — the EXACT set the
      // tier preflight tests. The route (and the siting phase) freezes only on an
      // output-bearing row a tier accepts; a no-output call consumes no stream
      // state, so it must not freeze. (Only consulted when the native tier is
      // compiled in; a filter plan never reaches here — it falls through to the
      // row-stage reject, so it never freezes the siting phase either.)
      #[cfg(all(feature = "v210", feature = "yuv-planar"))]
      let need_output = !plan.kind().is_filter()
        && (luma.is_some()
          || luma_u16.is_some()
          || rgb.is_some()
          || rgba.is_some()
          || rgb_u16.is_some()
          || rgba_u16.is_some()
          || hsv.is_some());
      // Freeze the effective 4:2:2 chroma siting on the first output-bearing row.
      // A later row observing a different phase — in sequence or not — would bin a
      // mixture of co-sited and centered chroma, so it is rejected HERE before any
      // reconstruction or dispatch; the matching SET rides each tier's accept path
      // below (never on a reject, so a corrected retry is not falsely rejected).
      #[cfg(all(feature = "v210", feature = "yuv-planar"))]
      if need_output
        && let Some(frozen) = *frozen_chroma_centered
        && frozen != center_sited
      {
        return Err(MixedSinkerError::ChromaSitingChanged(
          ChromaSitingChanged::new(idx),
        ));
      }
      // The reused high-bit non-4:2:0 planar join now emits BOTH the u8 `luma`
      // and the native-depth `luma_u16` (the clamped binned Y), so the native
      // tier serves every output set V210 exposes; route to native purely on
      // `with_native`, for an `Area` plan only (the native tier is area-only).
      // The output-set freeze keeps this invariant across a frame. The
      // splice stage is the RFC #238 selector's; folding the area-only guard
      // into `area_plan` reproduces the former `*native && !is_filter`
      // boolean bit-for-bit (`cfg!` is true wherever this block compiles).
      #[cfg(all(feature = "v210", feature = "yuv-planar"))]
      let take_native = matches!(
        select_insertion_point(
          AveragingDomain::Encoded,
          InsertionContext {
            native_eligible: cfg!(all(feature = "v210", feature = "yuv-planar")),
            with_native: *native,
            area_plan: !plan.kind().is_filter(),
          },
        ),
        InsertionPoint::NativeCodes
      );
      // Reject a mid-frame native/row-stage route flip BEFORE either tier's
      // dispatch (the two tiers carry independent, in-order, once-only stream
      // state). CHECKED here and frozen below ONLY on an output-bearing row a
      // tier ACCEPTS — both gate on `need_output`. (Mirrors the high-bit packed
      // 4:2:2 `y2xx`.)
      #[cfg(all(feature = "v210", feature = "yuv-planar"))]
      if need_output
        && let Some(frozen) = *frozen_native_route
        && frozen != take_native
      {
        return Err(MixedSinkerError::NativeRouteChanged(
          NativeRouteChanged::new(idx),
        ));
      }
      #[cfg(all(feature = "v210", feature = "yuv-planar"))]
      if take_native {
        // A reused sink's cached join is only `reset` between frames, so a frame
        // whose `chroma_location` moved to a different phase must REBUILD it. Drop
        // the stale-phase join ONLY on the in-sequence first row of a fresh frame
        // (`idx == 0`, `next_y() == 0`) so a mid-frame / out-of-sequence row
        // rejects against the INTACT join and a corrected retry rebuilds cleanly;
        // a luma-only join carries no chroma phase and is never dropped. Move it
        // OUT (the delegate builds the replacement into the field, untouched until
        // every pre-feed allocation succeeds) and restore the intact prior-phase
        // join on a rejected rebuild so the row mutates no join state.
        let stale_native = idx == 0
          && native_planar_u16.as_ref().is_some_and(|join| {
            join.chroma_phase_centered() == Some(!center_sited) && join.next_y() == 0
          });
        let prev_native = if stale_native {
          native_planar_u16.take()
        } else {
          None
        };
        // Dispatch first; freeze the route + siting ONLY after the call returns Ok
        // on an output-bearing row.
        let native_result = v210_process_native::<BE>(
          plan,
          native_planar_u16,
          v210_y_full,
          v210_u_half,
          v210_v_half,
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
          packed,
          chroma_h_phase,
          matrix,
          full_range,
          idx,
          w,
          h,
          use_simd,
        );
        // Restore the taken stale-phase join if the delegate's rebuild was
        // rejected at any pre-feed step: it leaves the field `None` on such a
        // failure, so restoring the intact prior-phase join leaves the rejected
        // row mutating no join state. A non-stale row took nothing.
        if stale_native && native_result.is_err() {
          *native_planar_u16 = prev_native;
        }
        native_result?;
        if frozen_native_route.is_none() && need_output {
          *frozen_native_route = Some(true);
        }
        if frozen_chroma_centered.is_none() && need_output {
          *frozen_chroma_centered = Some(center_sited);
        }
        return Ok(());
      }
      // Centered row-stage colour reconstructs full-width chroma (de-pack +
      // phase-0.5 upsample) and decodes 4:4:4 — but ONLY after the resample
      // preflight (frozen-output + sequence), so an out-of-sequence / rejected row
      // allocates no chroma scratch. A luma-only centered row stays on the
      // co-sited arm (which only bins luma), and a filter plan is excluded here so
      // it still falls to the co-sited tail's `unsupported_filter` reject with no
      // scratch grown.
      #[cfg(all(feature = "v210", feature = "yuv-planar"))]
      if center_sited && want_color && !plan.kind().is_filter() {
        let expected = if luma.is_some() || luma_u16.is_some() {
          luma_stream_u16.as_ref().map_or(0, |s| s.next_y())
        } else if rgb.is_some() || rgba.is_some() || hsv.is_some() {
          rgb_stream.as_ref().map_or(0, |s| s.next_y())
        } else {
          rgb_stream_u16.as_ref().map_or(0, |s| s.next_y())
        };
        if let core::ops::ControlFlow::Break(()) = resample_preflight_check_only(
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
          Some(expected),
          idx,
        )? {
          return Ok(());
        }
        let (u_full, v_full) = v210_center_reconstruct(
          packed,
          v210_y_full,
          v210_u_half,
          v210_v_half,
          chroma_full_u16,
          w,
          h,
          BE,
          use_simd,
        )?;
        let y_full = &v210_y_full[..w];
        packed_yuv422_triple_resample::<BITS>(
          luma_stream_u16,
          rgb_stream,
          rgb_stream_u16,
          resample_outputs,
          rgb,
          rgba,
          rgb_u16,
          rgba_u16,
          luma,
          luma_u16,
          hsv,
          luma_scratch_u16,
          rgb_scratch,
          rgb_scratch_u16,
          w,
          plan,
          idx,
          use_simd,
          matrix,
          full_range,
          |scratch| v210_to_luma_u16_row_endian(packed, scratch, w, use_simd, BE),
          |scratch| {
            yuv444p10_to_rgb_row_endian(
              y_full, u_full, v_full, scratch, w, matrix, full_range, use_simd, BE,
            )
          },
          |scratch| {
            yuv444p10_to_rgb_u16_row_endian(
              y_full, u_full, v_full, scratch, w, matrix, full_range, use_simd, BE,
            )
          },
        )?;
        if frozen_native_route.is_none() && need_output {
          *frozen_native_route = Some(false);
        }
        if frozen_chroma_centered.is_none() && need_output {
          *frozen_chroma_centered = Some(center_sited);
        }
        return Ok(());
      }
      // Row-stage area tail (the route under `with_native(false)`, the only
      // route under a `v210`-solo build where the planar join is absent, and the
      // path a `Filter` plan takes to be rejected as `unsupported_filter`). Same
      // CHECK-before / SET-after split. Co-sited keeps the fused `v210_*` decode;
      // V210 reuses the y2xx route at `BITS = 10` with its own word-packing decode
      // closures.
      packed_yuv422_triple_resample::<BITS>(
        luma_stream_u16,
        rgb_stream,
        rgb_stream_u16,
        resample_outputs,
        rgb,
        rgba,
        rgb_u16,
        rgba_u16,
        luma,
        luma_u16,
        hsv,
        luma_scratch_u16,
        rgb_scratch,
        rgb_scratch_u16,
        w,
        plan,
        idx,
        use_simd,
        matrix,
        full_range,
        |scratch| v210_to_luma_u16_row_endian(packed, scratch, w, use_simd, BE),
        |scratch| v210_to_rgb_row_endian(packed, scratch, w, matrix, full_range, use_simd, BE),
        |scratch| v210_to_rgb_u16_row_endian(packed, scratch, w, matrix, full_range, use_simd, BE),
      )?;
      #[cfg(all(feature = "v210", feature = "yuv-planar"))]
      {
        if frozen_native_route.is_none() && need_output {
          *frozen_native_route = Some(false);
        }
        if frozen_chroma_centered.is_none() && need_output {
          *frozen_chroma_centered = Some(center_sited);
        }
      }
      return Ok(());
    }

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

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
    // (HSV-only routes through the direct `v210_to_hsv_row_endian` kernel, which
    // needs no RGB scratch). The later decode reuses the already-sized buffer,
    // so the default path is byte-identical; only the failure-path ordering
    // changes.
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

    // Luma u8 — extract 8-bit Y bytes from the v210 plane via the
    // dedicated kernel (downshifts 10→8 inline).
    if let Some(buf) = luma.as_deref_mut() {
      v210_to_luma_row_endian(
        packed,
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
        BE,
      );
    }
    // Luma u16 — extract 10-bit Y values at native depth.
    if let Some(buf) = luma_u16.as_deref_mut() {
      v210_to_luma_u16_row_endian(
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
      v210_to_rgba_u16_row_endian(
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
      v210_to_rgb_u16_row_endian(
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
    // HSV-without-RGB-or-RGBA goes through the direct
    // `v210_to_hsv_row_endian` kernel (no source-width RGB scratch). When
    // RGB or RGBA is *also* attached the RGB kernel runs anyway, so HSV
    // derives off that buffer for free (the cheap path) and
    // `need_u8_rgb_kernel` keeps it alive. (Resample row-stage HSV stays
    // correct via the convert-once path in the plan branch above.)
    // `want_rgb` / `want_rgba` / `want_hsv` were resolved up front for the
    // atomicity preflight (#308).
    let want_hsv_direct = want_hsv && !want_rgb && !want_rgba;
    let need_u8_rgb_kernel = want_rgb || (want_hsv && want_rgba);

    if want_hsv_direct {
      let hsv = hsv.as_mut().expect("want_hsv_direct implies hsv attached");
      let (h, s, v) = hsv.hsv();
      v210_to_hsv_row_endian(
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
      v210_to_rgba_row_endian(
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
    v210_to_rgb_row_endian(
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
