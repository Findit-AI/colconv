//! Sinker impl for the Tier 5.25 packed YUV 4:1:1 source format —
//! UYYVYY411 (`AV_PIX_FMT_UYYVYY411`, DV legacy).
//!
//! Single packed plane carrying `width * 3 / 2` bytes per row (12 bpp)
//! with byte order `U, Y, Y, V, Y, Y` per 6-byte / 4-pixel block —
//! one (U, V) chroma pair shared by four luma samples. Width must be
//! a multiple of 4.
//!
//! Output channels mirror the Tier 3 packed YUV 4:2:2 sinker
//! ([`super::packed_yuv_8bit`]):
//!
//! - `with_rgb` / `with_rgba` — packed YUV → RGB Q15 pipeline (full
//!   `KernelMatrix` + range support inherited from the row); RGBA
//!   alpha is forced to `0xFF` (the source has no alpha channel).
//! - `with_luma` — extracts the Y bytes from the packed plane via
//!   the dedicated luma kernel.
//! - `with_luma_u16` — zero-extends Y bytes to u16.
//! - `with_hsv` — stages an internal RGB scratch (or the user's RGB
//!   buffer if attached) and runs the existing `rgb_to_hsv_row`
//!   kernel.
//!
//! When both RGB and RGBA outputs are requested, the RGBA plane is
//! derived from the just-computed RGB row via
//! [`expand_rgb_to_rgba_row`] (Strategy A — memory-bound copy + 0xFF
//! alpha pad) instead of running a second YUV→RGB kernel.

use super::{
  GeometryOverflow, InsufficientBuffer, MixedSinker, MixedSinkerError, RowIndexOutOfRange,
  RowShapeMismatch, RowSlice, WidthAlignment, check_dimensions_match,
  packed_yuv_8bit::packed_yuv422_dual_filter_resample, planar_resample::packed_yuv_dual_resample,
  rgb_row_buf_or_scratch, rgba_plane_row_slice,
};
use crate::{
  PixelSink,
  resample::SpanKind,
  row::{
    expand_rgb_to_rgba_row, rgb_to_hsv_row, uyyvyy411_to_hsv_row, uyyvyy411_to_luma_row,
    uyyvyy411_to_luma_u16_row, uyyvyy411_to_rgb_row, uyyvyy411_to_rgba_row,
  },
  source::{Uyyvyy411, Uyyvyy411Row, Uyyvyy411Sink},
};

#[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
use super::{
  Chroma411U8, ChromaSitingChanged, FrozenOutputs, HsvFrameMut, NativeRouteChanged,
  chroma_411_center_sited_h,
  planar_8bit::{
    NativePlanarYuv, YUV41X_CENTERED_H_PHASE, native_planar_preflight_check_only,
    reserve_420_chroma_full, yuv_planar_process_native,
  },
  reconstruct_chroma,
};
#[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
use crate::{
  KernelMatrix,
  resample::{
    AveragingDomain, InsertionContext, InsertionPoint, PlanGeometry, ResampleError, ResamplePlan,
    select_insertion_point,
  },
  row::{yuv_444_to_hsv_row, yuv_444_to_rgb_row, yuv_444_to_rgba_row},
};

/// Native fast-tier decimator for the 8-bit PACKED 4:1:1 YUV format
/// ([`Uyyvyy411`](crate::source::Uyyvyy411) — `AV_PIX_FMT_UYYVYY411`, DV
/// legacy; `U Y Y V Y Y` per 6-byte / 4-pixel block, one `(U, V)` chroma pair
/// shared by FOUR luma samples). The 4:1:1 analog of the PACKED 4:2:2 native
/// wrapper ([`super::packed_yuv_8bit`]'s `packed_yuv422_process_native`): the
/// chroma is subsampled 4:1 HORIZONTALLY (and 1:1 vertically), so it differs
/// from 4:2:2 only in the de-pack stride and the chroma width (`w / 4` vs
/// `w / 2`).
///
/// De-PACKS the fully-interleaved source row into the sink's separate
/// Y (`w`) / U (`w / 4`) / V (`w / 4`) scratch planes at the fixed byte offsets
/// (`U0 @ 0, Y0 @ 1, Y1 @ 2, V0 @ 3, Y2 @ 4, Y3 @ 5` per 6-byte group — the
/// same layout the row-stage de-pack reads), then reuses the planar twin's
/// non-4:2:0 join verbatim ([`yuv_planar_process_native`]) at
/// [`Yuv411p`](crate::source::Yuv411p) geometry (chroma `w / 4 x h`,
/// `chroma_vsub = 1`, chroma plan a plain [`ResamplePlan::area`] over the
/// `w / 4`-wide source chroma — exactly as the 4:2:2 wrapper uses `area` over
/// `w / 2`). So every output is byte-identical to a
/// [`Yuv411p`](crate::source::Yuv411p) native conversion of those de-packed
/// planes (there is no Yuv411p native tier — Yuv411p is row-stage only — so the
/// guard is an independent bin-then-convert oracle + native-within-tolerance of
/// the packed row-stage tier; the conversion-order rounding caveat the planar
/// tiers already carry). Luma is bit-identical (both bin the same native Y).
///
/// Like the 4:2:2 wrapper the chroma cadence is one row per Y row
/// (`chroma_vsub = 1`), so the U / V de-pack runs on EVERY colour row; on
/// luma-only / no-colour rows only Y is de-packed and the join gets empty
/// U / V slices, so a luma-only sink never plans or allocates chroma state (the
/// lazy-chroma contract [`NativePlanarYuv::new`] upholds under `need_color`).
///
/// 8-bit source, so no native-depth clamp is needed (the source's native range
/// is the full `u8` range and the join's averaging keeps every sample in
/// range).
///
/// Atomicity mirrors the 4:2:2 wrapper: the join's COMPLETE pre-feed rejection
/// preflight runs FIRST (via [`native_planar_preflight_check_only`]), before the
/// fallible Y / U / V scratch grow, so a rejected row returns its deterministic
/// typed error (`OutOfSequenceRow` / `ResampleOutputsChanged`), never
/// `AllocationFailed`, and grows no sink state. It is compare-only (no output-set
/// freeze), so the scratch grow stays a pre-feed step ahead of the delegate's
/// commit; the de-pack writes only the private scratch, so no caller output is
/// touched until the delegate clears.
#[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
#[allow(clippy::too_many_arguments)]
fn packed_yuv411_process_native(
  plan: &ResamplePlan,
  native_planar: &mut Option<std::boxed::Box<NativePlanarYuv>>,
  y_scratch: &mut std::vec::Vec<u8>,
  u_scratch: &mut std::vec::Vec<u8>,
  v_scratch: &mut std::vec::Vec<u8>,
  resample_outputs: &mut Option<FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  rgb_scratch: &mut std::vec::Vec<u8>,
  packed: &[u8],
  chroma_h_phase: f64,
  matrix: KernelMatrix,
  full_range: bool,
  idx: usize,
  w: usize,
  h: usize,
  use_simd: bool,
) -> Result<(), MixedSinkerError> {
  let need_luma = luma.is_some() || luma_u16.is_some();
  let need_color = rgb.is_some() || rgba.is_some() || hsv.is_some();
  let cw = w / 4;

  // Run the join's COMPLETE pre-feed rejection preflight FIRST — before the
  // fallible Y / U / V de-pack scratch grow — so EVERY rejection case
  // (out-of-sequence first row OR mid-frame output change) returns its
  // deterministic typed error, never AllocationFailed under allocation
  // pressure, and leaves the scratch untouched (the crate's preflight-atomicity
  // contract). Compare-only (no output-set freeze), so the scratch grow below
  // stays a pre-feed step ahead of the delegate's commit. `Ok(false)` is the
  // no-output no-op: return without reserving. `yuv_planar_process_native`
  // re-runs this identical compare harmlessly and owns the commit, keeping a
  // single source of truth.
  if !native_planar_preflight_check_only(
    native_planar,
    resample_outputs,
    rgb,
    rgba,
    luma,
    luma_u16,
    hsv,
    idx,
    need_luma,
    need_color,
  )? {
    return Ok(());
  }

  // De-pack the interleaved row into the private Y / U / V scratch. Y is always
  // de-packed (the join bins Y for both luma and colour); U / V are de-packed
  // only on a colour row (chroma_vsub == 1: a chroma row per Y row). The
  // de-pack writes only this private scratch, so no caller output is touched
  // until the join's own preflight (re-run inside the delegate below) clears.
  // On luma-only / no-colour rows the join never reads chroma, so the scratch
  // is left as-is and the join gets empty U / V slices — keeping a luma-only
  // sink from planning or allocating chroma state. Each 6-byte group
  // (`U0 Y0 Y1 V0 Y2 Y3`) carries four luma (bytes 1, 2, 4, 5) and one chroma
  // pair (U @ 0, V @ 3).
  let grow = |scratch: &mut std::vec::Vec<u8>, len: usize| -> Result<(), MixedSinkerError> {
    if scratch.len() < len {
      scratch
        .try_reserve_exact(len - scratch.len())
        .map_err(|_| {
          MixedSinkerError::Resample(ResampleError::AllocationFailed(PlanGeometry::new(
            w,
            h,
            plan.out_w(),
            plan.out_h(),
          )))
        })?;
      scratch.resize(len, 0);
    }
    Ok(())
  };
  grow(y_scratch, w)?;
  for (i, group) in packed.chunks_exact(6).enumerate() {
    y_scratch[i * 4] = group[1];
    y_scratch[i * 4 + 1] = group[2];
    y_scratch[i * 4 + 2] = group[4];
    y_scratch[i * 4 + 3] = group[5];
  }
  if need_color {
    grow(u_scratch, cw)?;
    grow(v_scratch, cw)?;
    for (i, group) in packed.chunks_exact(6).enumerate() {
      u_scratch[i] = group[0];
      v_scratch[i] = group[3];
    }
  }

  let (u_plane, v_plane): (&[u8], &[u8]) = if need_color {
    (&u_scratch[..cw], &v_scratch[..cw])
  } else {
    (&[], &[])
  };
  yuv_planar_process_native(
    plan,
    native_planar,
    resample_outputs,
    rgb,
    rgba,
    luma,
    luma_u16,
    hsv,
    rgb_scratch,
    &y_scratch[..w],
    u_plane,
    v_plane,
    matrix,
    full_range,
    idx,
    w,
    h,
    1,
    || ResamplePlan::area_chroma_411(w, h, plan.out_w(), plan.out_h(), chroma_h_phase, 0.0),
    use_simd,
  )
}

/// **Fallible preflight** for the packed 4:1:1 centered-siting de-pack scratch
/// (#302): grows the full-width Y buffer (`width`) and the quarter-width U / V
/// buffers (`width / 4` each) so the later infallible de-pack + `1→4`
/// reconstruction + 4:4:4 decode reuse already-sized buffers. The packed 4:1:1
/// sibling of the 4:2:2 `reserve_packed_center_chroma`: split from the de-pack so
/// it runs **before any output row is written** (the crate's preflight-ordering
/// atomicity contract, cf. the #180 resample fix). Reuses the native tier's
/// de-pack scratch (`width` Y + `width / 4` U / V), so a centered colour row and
/// an area native row never both run in one `process` call. A grow refusal is the
/// typed, recoverable [`ResampleError::AllocationFailed`]; `height` feeds the
/// payload.
#[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
fn reserve_packed_411_center_chroma(
  y_full: &mut std::vec::Vec<u8>,
  u_quarter: &mut std::vec::Vec<u8>,
  v_quarter: &mut std::vec::Vec<u8>,
  width: usize,
  height: usize,
) -> Result<(), MixedSinkerError> {
  let grow = |scratch: &mut std::vec::Vec<u8>, len: usize| -> Result<(), MixedSinkerError> {
    if scratch.len() < len {
      scratch
        .try_reserve_exact(len - scratch.len())
        .map_err(|_| {
          MixedSinkerError::Resample(ResampleError::AllocationFailed(PlanGeometry::new(
            width, height, width, height,
          )))
        })?;
      scratch.resize(len, 0);
    }
    Ok(())
  };
  grow(y_full, width)?;
  let cw = width / 4;
  grow(u_quarter, cw)?;
  grow(v_quarter, cw)?;
  Ok(())
}

/// De-packs a packed 4:1:1 (`Uyyvyy411`) row's chroma into the already-reserved
/// quarter-width U / V scratch, then `1→4` phase-reconstructs each plane to full
/// width in `chroma_full`, returning the full-width `(u_full, v_full)` the 4:4:4
/// decode kernels consume (#302). Each 6-byte / 4-pixel group carries `U @ 0`,
/// `V @ 3` — the SAME offsets the native tier de-packs, so a U/V swap here would
/// diverge from the planar twin. The row's Y is de-packed separately by
/// [`uyyvyy411_to_luma_row`] (the exact Y→luma derivation the default path uses).
///
/// **Infallible**: the caller must have run [`reserve_420_chroma_full`] and
/// [`reserve_packed_411_center_chroma`] up front (every centered output path
/// does, before any output write), so all three buffers are guaranteed long
/// enough.
#[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
fn packed_411_center_upsample_chroma<'s>(
  chroma_full: &'s mut [u8],
  u_quarter: &mut [u8],
  v_quarter: &mut [u8],
  packed: &[u8],
  width: usize,
  use_simd: bool,
) -> (&'s [u8], &'s [u8]) {
  let cw = width / 4;
  debug_assert!(
    u_quarter.len() >= cw && v_quarter.len() >= cw,
    "quarter-width chroma scratch must be reserved via reserve_packed_411_center_chroma first"
  );
  for (i, group) in packed.chunks_exact(6).take(cw).enumerate() {
    u_quarter[i] = group[0];
    v_quarter[i] = group[3];
  }
  reconstruct_chroma(
    Chroma411U8,
    chroma_full,
    &u_quarter[..cw],
    &v_quarter[..cw],
    width,
    use_simd,
  )
}

impl<'a, R> MixedSinker<'a, Uyyvyy411, R> {
  /// Attaches a packed **8-bit** RGBA output buffer. Alpha is filled
  /// with constant `0xFF` (the source has no alpha channel).
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

  /// Attaches a **`u16`** luma output buffer. Y bytes are zero-extended
  /// to u16 (`out[x] = Y_byte as u16`). Length in u16 **elements**
  /// (`width x height`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_luma_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_luma_u16(buf)?;
    Ok(self)
  }
  /// In-place variant of [`with_luma_u16`](Self::with_luma_u16).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_luma_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
    let expected_elems = self.frame_pixels()?;
    if buf.len() < expected_elems {
      return Err(MixedSinkerError::InsufficientLumaU16Buffer(
        InsufficientBuffer::new(expected_elems, buf.len()),
      ));
    }
    self.luma_u16 = Some(buf);
    Ok(self)
  }
}

impl<R> Uyyvyy411Sink for MixedSinker<'_, Uyyvyy411, R> {}

impl<R> PixelSink for MixedSinker<'_, Uyyvyy411, R> {
  type Input<'r> = Uyyvyy411Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    if self.width & 3 != 0 {
      return Err(MixedSinkerError::WidthAlignment(
        WidthAlignment::multiple_of_four(self.width),
      ));
    }
    // New frame: restart the row-stage resample streams and re-freeze
    // the output set so a reused sink starts each frame clean.
    if let Some(stream) = self.rgb_stream.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.luma_stream.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.rgb_filter_stream.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.luma_filter_stream.as_mut() {
      stream.reset();
    }
    // New frame: restart the native join and clear the per-frame frozen
    // native/row-stage route so the next frame may pick either tier; a
    // mid-frame flip stays rejected. Gated to the native tier's feature
    // intersection (the planar join the native tier reuses is compiled only
    // under `yuv-planar`).
    #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
    if let Some(native) = self.native_planar.as_mut() {
      native.reset();
    }
    #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
    {
      self.frozen_native_route = None;
      // RFC #238 #302: clear the per-frame frozen chroma siting so the next frame
      // may pick either phase; a mid-frame flip stays rejected within a frame.
      self.frozen_chroma_centered = None;
    }
    self.resample_outputs = None;
    Ok(())
  }

  fn process(&mut self, row: Uyyvyy411Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if w & 3 != 0 {
      return Err(MixedSinkerError::WidthAlignment(
        WidthAlignment::multiple_of_four(w),
      ));
    }

    // Row length: `width * 3 / 2` (12 bpp). `w` is a multiple of 4 by
    // the gate above, so `w * 3` is also a multiple of 4 and the
    // `/ 2` is exact. Check the `* 3` for 32-bit overflow.
    let packed_expected =
      w.checked_mul(3)
        .map(|n| n / 2)
        .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
          w, h, 3,
        )))?;
    if row.uyyvyy().len() != packed_expected {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Uyyvyy411Packed,
        idx,
        packed_expected,
        row.uyyvyy().len(),
      )));
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange(
        RowIndexOutOfRange::new(idx, self.height),
      ));
    }

    // Chroma siting (#302): drives the horizontal chroma phase. `Copy`, so read
    // it before the field split-borrow below. Gated like its only consumer
    // (`chroma_411_center_sited_h` + the 4:4:4 kernels need `yuv-planar`); a
    // `yuv-packed`-only build keeps the default nearest decode.
    #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
    let chroma_location = self.chroma_location.clone();

    let Self {
      rgb,
      rgba,
      luma,
      luma_u16,
      hsv,
      rgb_scratch,
      luma_scratch,
      plan,
      rgb_stream,
      luma_stream,
      rgb_filter_stream,
      luma_filter_stream,
      resample_outputs,
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      native,
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      native_planar,
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      packed_yuv_y_full,
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      packed_yuv_u_half,
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      packed_yuv_v_half,
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      frozen_native_route,
      // RFC #238 #302: the 4:1:1 chroma phase frozen on the first output row.
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      frozen_chroma_centered,
      // Centered chroma-siting (#302) stages full-width U + V here.
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      chroma_full,
      ..
    } = self;

    // Non-identity plan: row-stage fused resample. De-interleave the Y
    // bytes out of the packed plane for luma (the YUV luma contract —
    // luma resamples Y, never RGB-derived luma); for colour, convert the
    // packed row to a source-width RGB row with the same fused
    // `uyyvyy411_to_rgb_row` kernel the identity path uses (chroma
    // de-interleave + 4:1:1 horizontal upsample in registers), then resample
    // it. The span kind picks the engine — area binning (RGB equals an
    // `Rgb24` area-resample of the identity-converted frame) or
    // signed-coefficient filter (RGB equals the `Rgb24` filter of those
    // converted pixels; luma stays native Y). The filter arm shares the 4:2:2
    // tail: 4:1:1 differs only in the convert closures, which have the same
    // shape.
    if let Some(plan) = plan.as_ref() {
      let matrix = row.matrix();
      let full_range = row.full_range();
      let packed = row.uyyvyy();
      // RFC #238 #302 — 4:1:1 horizontal chroma siting, mirroring the planar
      // Yuv411p twin. The centered group (`Center` / `Top` / `Bottom`,
      // [`chroma_411_center_sited_h`]) samples chroma at the CENTER of its four
      // luma columns; the co-sited / unspecified group is phase 0 (byte-identical
      // to the pre-siting resample). The native fast tier folds the phase into the
      // chroma area weights (`area_chroma_411`); the filter and row-stage tiers
      // reconstruct full-width chroma (de-pack + `1→4` upsample) and decode 4:4:4.
      // The co-sited path keeps the fused `uyyvyy411_to_rgb_row` decode.
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      let center_sited = chroma_411_center_sited_h(&chroma_location);
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      let chroma_h_phase = if center_sited {
        YUV41X_CENTERED_H_PHASE
      } else {
        0.0
      };
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      let want_color = rgb.is_some() || rgba.is_some() || hsv.is_some();
      // Whether this call carries any output — the EXACT set both tiers' preflight
      // tests. The route (and the siting phase) freezes only on an output-bearing
      // row a tier ACCEPTS; a no-output call must not freeze.
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      let need_output =
        luma.is_some() || luma_u16.is_some() || rgb.is_some() || rgba.is_some() || hsv.is_some();
      // RFC #238 #302: freeze the effective 4:1:1 chroma siting on the first
      // output-bearing row. A later row observing a different phase — in sequence
      // or not — would bin a mixture of co-sited and centered chroma, so it is
      // rejected HERE before any reconstruction; the matching SET rides each
      // tier's accept-time freeze below.
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      if need_output
        && let Some(frozen) = *frozen_chroma_centered
        && frozen != center_sited
      {
        return Err(MixedSinkerError::ChromaSitingChanged(
          ChromaSitingChanged::new(idx),
        ));
      }
      // A `Filter` plan routes to the filter resampler (the native fast tier is
      // area-only and never sees a filter plan; the per-sink plan kind is fixed
      // at construction, so a filter sink bypasses the native/row-stage route
      // machinery entirely). Branched FIRST, before the native-route guard
      // below.
      if let SpanKind::Filter = plan.kind() {
        // Centered filter reconstructs full-width chroma (de-pack + `1→4`
        // upsample) and decodes 4:4:4 on the de-packed Y, but ONLY after the
        // resample preflight (frozen-output + sequence), so an out-of-sequence /
        // rejected row is caught before the chroma reservation (#180).
        // `packed_yuv422_dual_filter_resample` re-runs the idempotent preflight. A
        // luma-only centered row never calls the RGB converter, so it stays on the
        // co-sited arm (which only bins luma). Co-sited keeps the fused
        // `uyyvyy411_to_rgb_row` decode.
        #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
        {
          // Reject a multi-kernel (BICUBLIN) filter plan BEFORE the centered
          // reserve below, mirroring the delegate's own first act (idempotent).
          plan.ensure_single_kernel_filter()?;
          if center_sited && want_color {
            let need_luma = luma.is_some() || luma_u16.is_some();
            let expected = if need_luma {
              luma_filter_stream.as_ref().map_or(0, |s| s.next_y())
            } else {
              rgb_filter_stream.as_ref().map_or(0, |s| s.next_y())
            };
            if let core::ops::ControlFlow::Break(()) = super::resample_preflight_check_only(
              resample_outputs,
              luma,
              luma_u16,
              rgb,
              rgba,
              &None,
              &None,
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
            reserve_420_chroma_full(chroma_full, w, h)?;
            reserve_packed_411_center_chroma(
              packed_yuv_y_full,
              packed_yuv_u_half,
              packed_yuv_v_half,
              w,
              h,
            )?;
            uyyvyy411_to_luma_row(packed, &mut packed_yuv_y_full[..w], w, use_simd);
            let (u_full, v_full) = packed_411_center_upsample_chroma(
              chroma_full,
              packed_yuv_u_half,
              packed_yuv_v_half,
              packed,
              w,
              use_simd,
            );
            let r = packed_yuv422_dual_filter_resample(
              luma_filter_stream,
              rgb_filter_stream,
              resample_outputs,
              rgb,
              rgba,
              luma,
              luma_u16,
              hsv,
              luma_scratch,
              rgb_scratch,
              w,
              plan,
              idx,
              use_simd,
              |scratch| uyyvyy411_to_luma_row(packed, scratch, w, use_simd),
              |scratch| {
                yuv_444_to_rgb_row(
                  &packed_yuv_y_full[..w],
                  u_full,
                  v_full,
                  scratch,
                  w,
                  matrix,
                  full_range,
                  use_simd,
                );
              },
            );
            if r.is_ok() && need_output && frozen_chroma_centered.is_none() {
              *frozen_chroma_centered = Some(center_sited);
            }
            return r;
          }
        }
        packed_yuv422_dual_filter_resample(
          luma_filter_stream,
          rgb_filter_stream,
          resample_outputs,
          rgb,
          rgba,
          luma,
          luma_u16,
          hsv,
          luma_scratch,
          rgb_scratch,
          w,
          plan,
          idx,
          use_simd,
          |scratch| {
            uyyvyy411_to_luma_row(packed, scratch, w, use_simd);
          },
          |scratch| {
            uyyvyy411_to_rgb_row(packed, scratch, w, matrix, full_range, use_simd);
          },
        )?;
        #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
        if need_output && frozen_chroma_centered.is_none() {
          *frozen_chroma_centered = Some(center_sited);
        }
        return Ok(());
      }
      // Area plan. When the native tier is enabled (and the planar join it
      // reuses is compiled in), de-pack the interleaved row into Y / U / V
      // scratch and bin those planes at output resolution, converting once per
      // output row at output width (4:1:1: Y at bytes 1,2,4,5 / U at 0 / V at 3
      // per 6-byte group, chroma `w / 4` wide). Otherwise (or under
      // `with_native(false)`) take the row-stage tier: bin the de-interleaved Y
      // for luma, convert the packed row to a source-width RGB row and bin that.
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      {
        // The RFC #238 splice stage. A filter plan already returned above, so
        // `area_plan` is true and the selector reproduces the former `*native`
        // boolean bit-for-bit (`cfg!` is true wherever this block compiles).
        let take_native = matches!(
          select_insertion_point(
            AveragingDomain::Encoded,
            InsertionContext {
              native_eligible: cfg!(all(feature = "yuv-packed", feature = "yuv-planar")),
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
          // RFC #238 #302 point-of-use siting invalidation, mirroring the packed
          // 4:2:2 native arm: `chroma_location` can change at ANY point before
          // this row (including AFTER `begin_frame`, before row 0), so re-check the
          // cached join HERE and drop it when its folded chroma plan was built for
          // a different phase; `packed_yuv411_process_native` (via
          // `yuv_planar_process_native`) then rebuilds it with the current siting.
          // Retry-atomic: drop ONLY on the IN-SEQUENCE fresh-frame first row
          // (`idx == 0`, `next_y() == 0`); an out-of-sequence first row is left for
          // the delegate to reject against the INTACT join. Transactional: move the
          // stale join OUT, let the delegate build the replacement, and restore the
          // intact prior-phase join on a rejected rebuild so the REJECTED row
          // mutates nothing.
          let stale_native = idx == 0
            && native_planar.as_ref().is_some_and(|join| {
              join.has_chroma() && join.chroma_centered() != center_sited && join.next_y() == 0
            });
          let prev_native = if stale_native {
            native_planar.take()
          } else {
            None
          };
          // Dispatch first; freeze the route + siting ONLY after the call returns
          // Ok on an output-bearing row.
          let native_result = packed_yuv411_process_native(
            plan,
            native_planar,
            packed_yuv_y_full,
            packed_yuv_u_half,
            packed_yuv_v_half,
            resample_outputs,
            rgb,
            rgba,
            luma,
            luma_u16,
            hsv,
            rgb_scratch,
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
          // rejected at any pre-feed step: it leaves the field `None` and
          // `resample_outputs` uncommitted on such a failure, so restoring the
          // intact prior-phase join leaves the rejected row mutating nothing.
          if stale_native && native_result.is_err() {
            *native_planar = prev_native;
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
      }
      // Row-stage tail (the only route under a `yuv-packed`-solo build).
      // Centered colour reconstructs full-width chroma (de-pack + `1→4` upsample)
      // and decodes 4:4:4 — but ONLY after the resample preflight (frozen-output +
      // sequence), so an out-of-sequence / rejected row is caught before the
      // chroma reservation (#180). A luma-only centered row never calls the RGB
      // converter, so it stays on the co-sited arm (which only bins luma).
      // `packed_yuv_dual_resample` re-runs the idempotent preflight. Dispatch, then
      // under the native tier freeze the route + siting only when the call accepts
      // an output-bearing row.
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      if center_sited && want_color {
        let need_luma = luma.is_some() || luma_u16.is_some();
        let expected = if need_luma {
          luma_stream.as_ref().map_or(0, |s| s.next_y())
        } else {
          rgb_stream.as_ref().map_or(0, |s| s.next_y())
        };
        if let core::ops::ControlFlow::Break(()) = super::resample_preflight_check_only(
          resample_outputs,
          luma,
          luma_u16,
          rgb,
          rgba,
          &None,
          &None,
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
        reserve_420_chroma_full(chroma_full, w, h)?;
        reserve_packed_411_center_chroma(
          packed_yuv_y_full,
          packed_yuv_u_half,
          packed_yuv_v_half,
          w,
          h,
        )?;
        uyyvyy411_to_luma_row(packed, &mut packed_yuv_y_full[..w], w, use_simd);
        let (u_full, v_full) = packed_411_center_upsample_chroma(
          chroma_full,
          packed_yuv_u_half,
          packed_yuv_v_half,
          packed,
          w,
          use_simd,
        );
        packed_yuv_dual_resample(
          luma_stream,
          rgb_stream,
          resample_outputs,
          rgb,
          rgba,
          luma,
          luma_u16,
          hsv,
          luma_scratch,
          rgb_scratch,
          w,
          plan,
          idx,
          use_simd,
          |scratch| uyyvyy411_to_luma_row(packed, scratch, w, use_simd),
          |scratch| {
            yuv_444_to_rgb_row(
              &packed_yuv_y_full[..w],
              u_full,
              v_full,
              scratch,
              w,
              matrix,
              full_range,
              use_simd,
            );
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
      packed_yuv_dual_resample(
        luma_stream,
        rgb_stream,
        resample_outputs,
        rgb,
        rgba,
        luma,
        luma_u16,
        hsv,
        luma_scratch,
        rgb_scratch,
        w,
        plan,
        idx,
        use_simd,
        |scratch| {
          uyyvyy411_to_luma_row(packed, scratch, w, use_simd);
        },
        |scratch| {
          uyyvyy411_to_rgb_row(packed, scratch, w, matrix, full_range, use_simd);
        },
      )?;
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
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
    let packed = row.uyyvyy();

    // Strategy A output mode resolution — resolved BEFORE any output write so
    // the atomicity preflight below runs ahead of the luma writes. The centered
    // horizontal sitings (#302) reconstruct chroma at the `1→4` phase-0.375
    // position then reuse the 4:4:4 dispatchers on the de-packed Y.
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();

    // Chroma siting (#302): the centered horizontal sitings de-pack Y + `1→4`
    // reconstruct chroma to full width, then decode 4:4:4; the default / co-sited
    // path keeps the byte-identical nearest decode. Gated like its consumers
    // (`yuv-planar`); a `yuv-packed`-only build keeps co-sited.
    #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
    let center_sited = chroma_411_center_sited_h(&chroma_location);

    // RFC #238 #302: mirror the resample branch's per-frame siting freeze on the
    // identity path. `chroma_location` is public and `Copy`, so a caller can flip
    // the phase mid-frame; without this a single frame could mix centered (`1→4`)
    // and co-sited nearest reconstruction. Freeze the effective phase on the first
    // output-bearing row and reject a later differing phase BEFORE any centered
    // reservation or output write, so a rejected row mutates nothing (#180). The
    // matching SET rides the accept-time freeze after the reservations below.
    #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
    let need_output = luma.is_some() || luma_u16.is_some() || want_rgb || want_rgba || want_hsv;
    #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
    if need_output
      && let Some(frozen) = *frozen_chroma_centered
      && frozen != center_sited
    {
      return Err(MixedSinkerError::ChromaSitingChanged(
        ChromaSitingChanged::new(idx),
      ));
    }

    // Atomicity preflight (#302 / #308, cf. the crate's #180 resample fix and the
    // planar / semi-planar siblings): reserve EVERY fallible row scratch this row
    // needs BEFORE any output row (luma / luma_u16 included) is written, so an
    // allocator refusal returns a typed `AllocationFailed` leaving the output
    // frame untouched rather than partially mutated. Two groups can grow:
    //  1. the centered-siting full-width chroma (`chroma_full`) plus the packed
    //     de-pack scratch (full-width Y + quarter-width U / V); and
    //  2. the RGB row buffer (`want_hsv && want_rgba && !want_rgb`).
    // The later de-pack / decode calls reuse the already-sized buffers, so the
    // default path is byte-identical; only the failure-path ordering changes.
    #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
    if center_sited && (want_rgb || want_rgba || want_hsv) {
      reserve_420_chroma_full(chroma_full, w, h)?;
      reserve_packed_411_center_chroma(
        packed_yuv_y_full,
        packed_yuv_u_half,
        packed_yuv_v_half,
        w,
        h,
      )?;
    }
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

    // RFC #238 #302: every fallible per-row reservation above has succeeded, so
    // freeze the accepted phase for the frame (mirrors the resample branch's
    // accept-time SET). Never before the reservations, so a rejected or failed row
    // leaves the freeze clear for a corrected retry.
    #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
    if need_output && frozen_chroma_centered.is_none() {
      *frozen_chroma_centered = Some(center_sited);
    }

    // Luma u8 — extract Y bytes from packed plane via dedicated kernel.
    if let Some(luma) = luma.as_deref_mut() {
      uyyvyy411_to_luma_row(
        packed,
        &mut luma[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }
    // Luma u16 — zero-extend Y bytes to u16.
    if let Some(buf) = luma_u16.as_deref_mut() {
      uyyvyy411_to_luma_u16_row(
        packed,
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    // HSV-without-RGB-or-RGBA goes through the direct `uyyvyy411_to_hsv_row`
    // kernel (no source-width RGB scratch). When RGB or RGBA is *also*
    // attached the RGB kernel runs anyway, so HSV derives off that buffer
    // for free (the cheap path) and `need_rgb_kernel` keeps it alive.
    let want_hsv_direct = want_hsv && !want_rgb && !want_rgba;
    let need_rgb_kernel = want_rgb || (want_hsv && want_rgba);

    if want_hsv_direct {
      let hsv = hsv.as_mut().expect("want_hsv_direct implies hsv attached");
      let (h_out, s_out, v_out) = hsv.hsv();
      // Centered siting (#302): de-pack Y + `1→4` upsample chroma to full width,
      // then run the 4:4:4 HSV kernel (scratch reserved above).
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      if center_sited {
        uyyvyy411_to_luma_row(packed, &mut packed_yuv_y_full[..w], w, use_simd);
        let (u_full, v_full) = packed_411_center_upsample_chroma(
          chroma_full,
          packed_yuv_u_half,
          packed_yuv_v_half,
          packed,
          w,
          use_simd,
        );
        yuv_444_to_hsv_row(
          &packed_yuv_y_full[..w],
          u_full,
          v_full,
          &mut h_out[one_plane_start..one_plane_end],
          &mut s_out[one_plane_start..one_plane_end],
          &mut v_out[one_plane_start..one_plane_end],
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
        );
        return Ok(());
      }
      uyyvyy411_to_hsv_row(
        packed,
        &mut h_out[one_plane_start..one_plane_end],
        &mut s_out[one_plane_start..one_plane_end],
        &mut v_out[one_plane_start..one_plane_end],
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
    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      // Centered siting (#302): full-width `1→4` chroma + the 4:4:4 RGBA kernel on
      // the de-packed Y; default keeps `uyyvyy411_to_rgba_row`.
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      if center_sited {
        uyyvyy411_to_luma_row(packed, &mut packed_yuv_y_full[..w], w, use_simd);
        let (u_full, v_full) = packed_411_center_upsample_chroma(
          chroma_full,
          packed_yuv_u_half,
          packed_yuv_v_half,
          packed,
          w,
          use_simd,
        );
        yuv_444_to_rgba_row(
          &packed_yuv_y_full[..w],
          u_full,
          v_full,
          rgba_row,
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
        );
        return Ok(());
      }
      uyyvyy411_to_rgba_row(
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

    let rgb_row = rgb_row_buf_or_scratch(
      rgb.as_deref_mut(),
      rgb_scratch,
      one_plane_start,
      one_plane_end,
      w,
      h,
    )?;
    // Centered siting (#302) decodes via the 4:4:4 RGB kernel on the de-packed Y +
    // `1→4` upsampled chroma; the HSV / RGBA follow-ups below derive off the
    // produced RGB row either way. Default keeps the fused `uyyvyy411_to_rgb_row`.
    #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
    let centered = if center_sited {
      uyyvyy411_to_luma_row(packed, &mut packed_yuv_y_full[..w], w, use_simd);
      let (u_full, v_full) = packed_411_center_upsample_chroma(
        chroma_full,
        packed_yuv_u_half,
        packed_yuv_v_half,
        packed,
        w,
        use_simd,
      );
      yuv_444_to_rgb_row(
        &packed_yuv_y_full[..w],
        u_full,
        v_full,
        rgb_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
      true
    } else {
      false
    };
    #[cfg(not(all(feature = "yuv-packed", feature = "yuv-planar")))]
    let centered = false;
    if !centered {
      uyyvyy411_to_rgb_row(packed, rgb_row, w, row.matrix(), row.full_range(), use_simd);
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

    // Strategy A: when both RGB and RGBA are requested, derive RGBA
    // from the just-computed RGB row instead of running a second
    // YUV→RGB kernel.
    if let Some(buf) = rgba.as_deref_mut() {
      let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
    }

    Ok(())
  }
}
