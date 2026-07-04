//! Sinker impls for the half-float planar GBR source family (Tier 10 f16).
//!
//! Two formats covered in this file:
//! - [`Gbrpf16`] (`AV_PIX_FMT_GBRPF16LE`) — three planes (G, B, R), `half::f16`,
//!   no alpha.
//! - [`Gbrapf16`] (`AV_PIX_FMT_GBRAPF16LE`) — four planes (G, B, R, A),
//!   `half::f16`, real per-pixel α.
//!
//! # Output paths
//!
//! - `with_rgb` / `with_rgba` — delegate to `gbrpf16_to_rgb_row` /
//!   `gbrpf16_to_rgba_row` (dispatcher handles f16 → f32 widening internally
//!   where no fp16/F16C SIMD is available).
//! - `with_rgb_f16` / `with_rgba_f16` — lossless f16 interleave via
//!   `gbrpf16_to_rgb_f16_row` / `gbrpf16_to_rgba_f16_row`; no conversion.
//! - `with_rgb_f32` / `with_rgba_f32` — widen f16 → f32 per-row (using a
//!   `Vec`-backed scratch buffer grown lazily), then call `gbrpf32_to_rgb_f32_row`
//!   / `gbrpf32_to_rgba_f32_row`.
//! - `with_rgb_u16` / `with_rgba_u16` — same widen + `gbrpf32_to_rgb_u16_row`
//!   / `gbrpf32_to_rgba_u16_row`.
//! - `with_luma` / `with_luma_u16` — same widen + `gbrpf32_to_luma_row` /
//!   `gbrpf32_to_luma_u16_row`.
//! - `with_hsv` — same widen + `gbrpf32_to_hsv_row`.
//!
//! For `Gbrapf16`, RGBA outputs use real source α from the A plane:
//! - `with_rgba` — `gbrapf16` → u8 RGBA: combo with `with_rgb` uses Strategy
//!   A+ (expand RGB u8 → RGBA u8, then `copy_alpha_plane_f16_to_u8` overwrites
//!   slot 3 from the source α plane narrowed/scaled).
//! - `with_rgba_f16` — `gbrapf16_to_rgba_f16_row` (lossless source α).
//! - `with_rgba_f32` / `with_rgba_u16` — widen + gbrapf32 kernels with real α.
//!
//! # F16→F32 widening scratch
//!
//! The `MixedSinker` for f16 formats uses the existing `rgb_scratch` heap-
//! allocated buffer re-purposed as a `u8` byte region **plus** three inline
//! f32 scratch slices obtained via `Vec<f32>` grown on demand (analogous to
//! `rgb_scratch` for u8). However, to keep the struct generic and avoid new
//! fields, the sinker widens f16 → f32 into a fresh per-row stack chunk
//! (`const CHUNK: usize = 64`) and calls gbrpf32 dispatchers in strided
//! chunks — identical to the widening pattern used by the dispatch layer
//! for backends without native fp16/F16C support.

use super::{
  GeometryOverflow, InsufficientBuffer, MixedSinker, MixedSinkerError, RowIndexOutOfRange,
  RowShapeMismatch, RowSlice, check_dimensions_match, check_frozen_alpha_mode,
  packed_rgb_f32_resample, packed_rgb_f32_resample_filter, packed_rgba_f16_filter_resample,
  packed_rgba_f16_resample, planar_gbr_f16_resample_emit, rgb_row_buf_or_scratch,
  rgba_plane_row_slice,
};
use crate::{
  ColorMatrix, PixelSink,
  row::{
    expand_rgb_to_rgba_row, gbrapf16_to_rgba_f16_row, gbrapf32_to_rgba_f32_row,
    gbrapf32_to_rgba_u16_row, gbrpf16_to_rgb_f16_row, gbrpf16_to_rgb_row, gbrpf16_to_rgba_f16_row,
    gbrpf16_to_rgba_row, gbrpf32_to_hsv_row, gbrpf32_to_luma_row, gbrpf32_to_luma_u16_row,
    gbrpf32_to_rgb_f32_row, gbrpf32_to_rgb_u16_row, gbrpf32_to_rgba_f32_row,
    gbrpf32_to_rgba_u16_row,
    scalar::{alpha_extract::copy_alpha_plane_f32_to_u8, planar_gbr_f16::widen_f16_be_to_host_f32},
  },
  source::{Gbrapf16, Gbrapf16Row, Gbrapf16Sink, Gbrpf16, Gbrpf16Row, Gbrpf16Sink},
};

// Float-planar GBR sources are already component RGB (no chroma matrix).
// BT.709 full-range is the conventional default for luma derivation.
const GBR_F16_LUMA_MATRIX: ColorMatrix = ColorMatrix::Bt709;
const GBR_F16_FULL_RANGE: bool = true;

// Chunk size for the inline f16→f32 widening scratch arrays (stack-allocated).
const WIDEN_CHUNK: usize = 64;

// Endianness routing for **post-widen** `gbrpf32_to_*` calls.
//
// `widen_f16_be_to_host_f32::<BE>` produces **host-native f32 scratch** from
// LE-encoded f16 plane bits (it normalises bits before widening), so the
// downstream `gbrpf32_to_*::<HOST_NATIVE_BE>` kernel sees input that already
// matches the host's byte order. The kernel's `from_le` / `from_be` then
// becomes a no-op on every host — correct.
//
// Distinct from the **direct** Frame-to-row-kernel pattern elsewhere in this
// file (the `gbrpf16_to_*::<false>` u8/f16 calls): those receive raw LE-encoded
// `&[half::f16]` plane bytes per the unified Frame contract, so they pass
// `BE = false` to tell the kernel to apply `from_le`. Post-widen scratch is
// already host-native, so it must use `BE = HOST_NATIVE_BE` to keep the kernel
// byte-swap a no-op on every host.
const HOST_NATIVE_BE: bool = cfg!(target_endian = "big");

// ---- Gbrpf16 accessor impl block ----------------------------------------

impl<'a, R, const BE: bool> MixedSinker<'a, Gbrpf16<BE>, R> {
  /// Attaches a packed **8-bit** RGBA output buffer. α is forced to `0xFF`
  /// (Gbrpf16 has no alpha channel). Length in bytes (`width x height x 4`).
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

  /// Attaches a packed **`u16`** RGB output buffer. Each f16 channel is
  /// widened to f32, clamped to `[0, 1]`, and scaled x 65535.
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

  /// Attaches a packed **`u16`** RGBA output buffer. Same full-range scaling
  /// (x 65535) as `with_rgb_u16`; α is constant `0xFFFF`.
  /// Length in `u16` elements (`width x height x 4`).
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

  /// Attaches a packed **`f32`** RGB output buffer. f16 channels are widened
  /// to f32 — lossless (f16 ⊂ f32). HDR and special values preserved.
  /// Length in `f32` elements (`width x height x 3`).
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

  /// Attaches a packed **`f32`** RGBA output buffer. f16 widened to f32;
  /// α is constant `1.0f32`. Length in `f32` elements (`width x height x 4`).
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

  /// Attaches a packed **`half::f16`** RGB output buffer. Lossless planar →
  /// packed interleave — no conversion. HDR values, NaN, and Inf preserved
  /// bit-exact. Length in `half::f16` elements (`width x height x 3`).
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

  /// Attaches a packed **`half::f16`** RGBA output buffer. Lossless planar →
  /// packed interleave with constant α = `half::f16::from_f32(1.0)`.
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

  /// Attaches a `u16` luma output buffer. f16 channels are widened to f32,
  /// then luma is derived (clamp + round-half-up) and zero-extended to u16.
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

// The half-float planar-GBR sink is generic over the resampler `R`. The
// `gbr` feature pulls the area-resample engine in (the #146 cascade), so
// `Gbrpf16` needs no engine fence. There is no `AreaStream<f16>`, so its
// non-identity plan widens the G/B/R f16 planes to host-native f32,
// scatters them into a source-width packed `R, G, B` f32 row, and bins in
// float on the shared `AreaStream<f32>` (the f32 tail's stream) for
// precision. Per finalized output row the dedicated `gbr` emit
// de-interleaves the binned packed row directly into G/B/R `half::f16`
// planes (**rounding** each element with `half::f16::from_f32`) and runs
// the exact direct `gbrpf16_*` kernels — so every output is
// byte-identical to a direct `Gbrpf16` conversion of the f32 block-mean
// rounded to f16. `Gbrapf16` (the 4-plane alpha variant) stays pinned to
// `NoopResampler`: alpha-aware area resampling is a later batch.

impl<R, const BE: bool> Gbrpf16Sink<BE> for MixedSinker<'_, Gbrpf16<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, Gbrpf16<BE>, R> {
  type Input<'r> = Gbrpf16Row<'r>;
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

  fn process(&mut self, row: Gbrpf16Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    // Defense-in-depth row-shape checks before any unsafe kernel.
    if row.g().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::GbrF16Plane,
        idx,
        w,
        row.g().len(),
      )));
    }
    if row.b().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::GbrF16Plane,
        idx,
        w,
        row.b().len(),
      )));
    }
    if row.r().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::GbrF16Plane,
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

    // Non-identity plan: widen the f16 G/B/R planes to host-native f32,
    // scatter into a source-width packed `R, G, B` f32 row, and bin in
    // float. The dedicated emit de-interleaves each binned row directly
    // into G/B/R `half::f16` planes (rounding via `from_f32`) and runs the
    // exact direct `gbrpf16_*` kernels, so every output stays
    // byte-identical to a direct `Gbrpf16` conversion of the f32
    // block-mean rounded to f16. The span kind picks the engine — float
    // area or signed-coefficient filter (both bin the same widened
    // host-native f32 row and feed the same emit). There is no
    // `FilterStream<f16>`, so the filter path also bins in f32; f16 is
    // full-range, so there is no native-depth clamp on the filter output.
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
        rgb_plane_scratch_f16,
        rgb_stream_f32,
        rgb_filter_stream_f32,
        resample_outputs,
        ..
      } = self;

      // Transactional resample (RFC #238 tail-collapse): the span kind picks the
      // shared 3-channel float driver (float area / signed-coefficient filter, both
      // binning the staged host-native f32 RGB row widened + scattered from the
      // f16 G/B/R planes); each derives every output via the dedicated planar
      // f16-round `emit` (the widen produces host-native f32 from the wire-encoded
      // f16 bits, so the interleave / de-interleave kernels run `::<HOST_NATIVE_BE>`
      // inside the emit).
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
          |src_f32| {
            scatter_gbrpf16_to_packed_f32::<BE>(row.g(), row.b(), row.r(), src_f32, w, use_simd)
          },
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
            planar_gbr_f16_resample_emit(
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
              rgb_plane_scratch_f16,
              GBR_F16_LUMA_MATRIX,
              GBR_F16_FULL_RANGE,
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
          |src_f32| {
            scatter_gbrpf16_to_packed_f32::<BE>(row.g(), row.b(), row.r(), src_f32, w, use_simd)
          },
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
            planar_gbr_f16_resample_emit(
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
              rgb_plane_scratch_f16,
              GBR_F16_LUMA_MATRIX,
              GBR_F16_FULL_RANGE,
              idx,
              use_simd,
            )
          },
        ),
      };
    }

    let g_in = row.g();
    let b_in = row.b();
    let r_in = row.r();
    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // ---- Lossless f16 native pass-through (no conversion) ----------------

    if let Some(buf) = self.rgb_f16.as_deref_mut() {
      let start = one_plane_start * 3;
      let end = one_plane_end * 3;
      gbrpf16_to_rgb_f16_row::<BE>(g_in, b_in, r_in, &mut buf[start..end], w, use_simd);
    }

    if let Some(buf) = self.rgba_f16.as_deref_mut() {
      let start = one_plane_start * 4;
      let end = one_plane_end
        .checked_mul(4)
        .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
          w, h, 4,
        )))?;
      gbrpf16_to_rgba_f16_row::<BE>(g_in, b_in, r_in, &mut buf[start..end], w, use_simd);
    }

    // ---- Paths that require widening f16 → f32 ---------------------------
    //
    // Use a chunk-based inline scratch to avoid heap allocation per row.
    // The chunk size of 64 matches the dispatch layer's widening pattern.
    // When no f32/u16/luma/HSV outputs are attached this block is a no-op.

    let need_wide = self.rgb_f32.is_some()
      || self.rgba_f32.is_some()
      || self.rgb_u16.is_some()
      || self.rgba_u16.is_some()
      || self.luma.is_some()
      || self.luma_u16.is_some()
      || self.hsv.is_some();

    if need_wide {
      let mut gf_chunk = [0.0f32; WIDEN_CHUNK];
      let mut bf_chunk = [0.0f32; WIDEN_CHUNK];
      let mut rf_chunk = [0.0f32; WIDEN_CHUNK];
      let mut offset = 0;
      while offset < w {
        let n = (w - offset).min(WIDEN_CHUNK);
        // Bit-normalise LE-encoded f16 plane bits → host-native f32 so the
        // downstream `gbrpf32_to_*` kernel (invoked with `BE = HOST_NATIVE_BE`
        // — see module-scope constant) sees host-native f32 on every host.
        // The post-widen scratch is host-native, distinct from the direct-
        // Frame paths which use `<false>` per the LE-encoded byte contract.
        widen_f16_be_to_host_f32::<BE>(g_in, offset, &mut gf_chunk, n);
        widen_f16_be_to_host_f32::<BE>(b_in, offset, &mut bf_chunk, n);
        widen_f16_be_to_host_f32::<BE>(r_in, offset, &mut rf_chunk, n);
        let gf = &gf_chunk[..n];
        let bf = &bf_chunk[..n];
        let rf = &rf_chunk[..n];

        let chunk_plane_start = one_plane_start + offset;
        let chunk_plane_end = chunk_plane_start + n;

        if let Some(buf) = self.rgb_f32.as_deref_mut() {
          let start = chunk_plane_start * 3;
          let end = chunk_plane_end * 3;
          gbrpf32_to_rgb_f32_row::<HOST_NATIVE_BE>(gf, bf, rf, &mut buf[start..end], n, use_simd);
        }

        if let Some(buf) = self.rgba_f32.as_deref_mut() {
          let start = chunk_plane_start * 4;
          let end = chunk_plane_end * 4;
          gbrpf32_to_rgba_f32_row::<HOST_NATIVE_BE>(gf, bf, rf, &mut buf[start..end], n, use_simd);
        }

        if let Some(buf) = self.rgb_u16.as_deref_mut() {
          let start = chunk_plane_start * 3;
          let end = chunk_plane_end * 3;
          gbrpf32_to_rgb_u16_row::<HOST_NATIVE_BE>(gf, bf, rf, &mut buf[start..end], n, use_simd);
        }

        if let Some(buf) = self.rgba_u16.as_deref_mut() {
          let start = chunk_plane_start * 4;
          let end = chunk_plane_end * 4;
          gbrpf32_to_rgba_u16_row::<HOST_NATIVE_BE>(gf, bf, rf, &mut buf[start..end], n, use_simd);
        }

        if let Some(buf) = self.luma.as_deref_mut() {
          gbrpf32_to_luma_row::<HOST_NATIVE_BE>(
            gf,
            bf,
            rf,
            &mut buf[chunk_plane_start..chunk_plane_end],
            n,
            GBR_F16_LUMA_MATRIX,
            GBR_F16_FULL_RANGE,
            use_simd,
          );
        }

        if let Some(buf) = self.luma_u16.as_deref_mut() {
          gbrpf32_to_luma_u16_row::<HOST_NATIVE_BE>(
            gf,
            bf,
            rf,
            &mut buf[chunk_plane_start..chunk_plane_end],
            n,
            GBR_F16_LUMA_MATRIX,
            GBR_F16_FULL_RANGE,
            use_simd,
          );
        }

        if let Some(hsv) = self.hsv.as_mut() {
          let (h, s, v) = hsv.hsv();
          gbrpf32_to_hsv_row::<HOST_NATIVE_BE>(
            gf,
            bf,
            rf,
            &mut h[chunk_plane_start..chunk_plane_end],
            &mut s[chunk_plane_start..chunk_plane_end],
            &mut v[chunk_plane_start..chunk_plane_end],
            n,
            use_simd,
          );
        }

        offset += n;
      }
    }

    // ---- u8 RGBA standalone fast path ------------------------------------

    let want_rgba = self.rgba.is_some();
    let want_rgb = self.rgb.is_some();
    let need_u8_rgb = want_rgb;

    if want_rgba && !need_u8_rgb {
      let rgba_buf = self.rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      gbrpf16_to_rgba_row::<BE>(g_in, b_in, r_in, rgba_row, w, use_simd);
      return Ok(());
    }

    if !need_u8_rgb && !want_rgba {
      return Ok(());
    }

    // ---- Stage u8 RGB once for RGBA fan-out ------------------------------

    let Self {
      rgb,
      rgba,
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
    gbrpf16_to_rgb_row::<BE>(g_in, b_in, r_in, rgb_row, w, use_simd);

    // Strategy A: expand RGB → RGBA (constant α = 0xFF).
    if let Some(buf) = rgba.as_deref_mut() {
      let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
    }

    Ok(())
  }
}

/// Widen the `Gbrpf16` G/B/R `half::f16` planes to host-native f32 and
/// scatter them into a source-width packed `R, G, B` f32 row — the input
/// the shared `AreaStream<f32>` bins. There is no `AreaStream<f16>`, so
/// the resample tail bins in f32 for precision; this prepares that input.
///
/// Endian routing: `widen_f16_be_to_host_f32::<BE>` bit-normalises the
/// source-encoded f16 plane bits (LE if `BE = false`, BE if `BE = true`)
/// to **host-native** f32, then the direct `gbrpf32_to_rgb_f32_row`
/// interleave runs with `HOST_NATIVE_BE` so its load is a no-op on every
/// host (the post-widen routing, distinct from the direct Frame-to-kernel
/// `<BE>` paths). Widening is chunked into stack scratch — no allocation.
#[cfg_attr(not(tarpaulin), inline(always))]
fn scatter_gbrpf16_to_packed_f32<const BE: bool>(
  g: &[half::f16],
  b: &[half::f16],
  r: &[half::f16],
  rgb_out: &mut [f32],
  width: usize,
  use_simd: bool,
) {
  let mut gf = [0.0f32; WIDEN_CHUNK];
  let mut bf = [0.0f32; WIDEN_CHUNK];
  let mut rf = [0.0f32; WIDEN_CHUNK];
  let mut offset = 0;
  while offset < width {
    let n = (width - offset).min(WIDEN_CHUNK);
    widen_f16_be_to_host_f32::<BE>(g, offset, &mut gf, n);
    widen_f16_be_to_host_f32::<BE>(b, offset, &mut bf, n);
    widen_f16_be_to_host_f32::<BE>(r, offset, &mut rf, n);
    gbrpf32_to_rgb_f32_row::<HOST_NATIVE_BE>(
      &gf[..n],
      &bf[..n],
      &rf[..n],
      &mut rgb_out[offset * 3..],
      n,
      use_simd,
    );
    offset += n;
  }
}

/// Widen the `Gbrapf16` G/B/R/A `half::f16` planes to host-native f32 and
/// scatter them into a source-width packed `R, G, B, A` f32 row — the input
/// the 4-channel `AreaStream<f32>` bins. The 4-channel twin of
/// [`scatter_gbrpf16_to_packed_f32`]: the α plane is widened and interleaved
/// into slot 3 too (lossless, so resampled alpha is a real area mean).
///
/// Endian routing matches the 3-channel scatter: `widen_f16_be_to_host_f32::<BE>`
/// bit-normalises the source-encoded f16 plane bits to **host-native** f32,
/// then `gbrapf32_to_rgba_f32_row` interleaves with `HOST_NATIVE_BE` so its
/// load is a no-op on every host. Widening is chunked into stack scratch.
#[cfg_attr(not(tarpaulin), inline(always))]
fn scatter_gbrapf16_to_packed_rgba_f32<const BE: bool>(
  g: &[half::f16],
  b: &[half::f16],
  r: &[half::f16],
  a: &[half::f16],
  rgba_out: &mut [f32],
  width: usize,
  use_simd: bool,
) {
  let mut gf = [0.0f32; WIDEN_CHUNK];
  let mut bf = [0.0f32; WIDEN_CHUNK];
  let mut rf = [0.0f32; WIDEN_CHUNK];
  let mut af = [0.0f32; WIDEN_CHUNK];
  let mut offset = 0;
  while offset < width {
    let n = (width - offset).min(WIDEN_CHUNK);
    widen_f16_be_to_host_f32::<BE>(g, offset, &mut gf, n);
    widen_f16_be_to_host_f32::<BE>(b, offset, &mut bf, n);
    widen_f16_be_to_host_f32::<BE>(r, offset, &mut rf, n);
    widen_f16_be_to_host_f32::<BE>(a, offset, &mut af, n);
    gbrapf32_to_rgba_f32_row::<HOST_NATIVE_BE>(
      &gf[..n],
      &bf[..n],
      &rf[..n],
      &af[..n],
      &mut rgba_out[offset * 4..],
      n,
      use_simd,
    );
    offset += n;
  }
}

// ---- Gbrapf16 accessor impl block ----------------------------------------

impl<'a, R, const BE: bool> MixedSinker<'a, Gbrapf16<BE>, R> {
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

  /// Attaches a packed **`u16`** RGB output buffer. Widened f16 → f32,
  /// clamped, scaled x 65535. Length in `u16` elements (`width x height x 3`).
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

  /// Attaches a packed **`u16`** RGBA output buffer. Source α widened f16 → f32,
  /// clamped x 65535. Length in `u16` elements (`width x height x 4`).
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

  /// Attaches a packed **`f32`** RGB output buffer. f16 widened to f32
  /// (lossless). Length in `f32` elements (`width x height x 3`).
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

  /// Attaches a packed **`f32`** RGBA output buffer. Source α widened f16 → f32
  /// (lossless; HDR, NaN, Inf preserved). Length in `f32` elements
  /// (`width x height x 4`).
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

  /// Attaches a packed **`half::f16`** RGB output buffer. Lossless f16
  /// interleave. Length in `half::f16` elements (`width x height x 3`).
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

  /// Attaches a packed **`half::f16`** RGBA output buffer. Source α is passed
  /// through losslessly (HDR, NaN, Inf preserved bit-exact).
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

  /// Attaches a `u16` luma output buffer. f16 channels widened to f32, then
  /// luma derived (clamp + round-half-up) and zero-extended to u16.
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

// The half-float planar-GBR+alpha sink is generic over the resampler `R`.
// The `gbr` feature pulls the area-resample engine in (the #146 cascade),
// so `Gbrapf16` needs no engine fence. There is no `AreaStream<f16>`, so
// its non-identity plan widens the G/B/R/A f16 planes to host-native f32,
// scatters them into a source-width packed `R, G, B, A` f32 row, and bins
// all four channels in float on the shared 4-channel `AreaStream<f32>`. Per
// finalized output row the dedicated alpha-aware emit resolves the straight
// color, de-interleaves it into G/B/R/A `half::f16` planes (**rounding**
// each element), and runs the exact direct `gbrapf16_*` / `gbrpf16_*`
// kernels — so every output is byte-identical to a direct `Gbrapf16`
// conversion of the f32 block-mean rounded to f16.
impl<R, const BE: bool> Gbrapf16Sink<BE> for MixedSinker<'_, Gbrapf16<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, Gbrapf16<BE>, R> {
  type Input<'r> = Gbrapf16Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
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

  fn process(&mut self, row: Gbrapf16Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    // Defense-in-depth row-shape checks before any unsafe kernel.
    if row.g().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::GbrF16Plane,
        idx,
        w,
        row.g().len(),
      )));
    }
    if row.b().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::GbrF16Plane,
        idx,
        w,
        row.b().len(),
      )));
    }
    if row.r().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::GbrF16Plane,
        idx,
        w,
        row.r().len(),
      )));
    }
    if row.a().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::GbrF16Plane,
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

    // Non-identity plan: widen the f16 G/B/R/A planes to host-native f32,
    // scatter into a source-width packed `R, G, B, A` f32 row, and bin all
    // four channels in float. The dedicated alpha-aware emit resolves the
    // straight color, de-interleaves into G/B/R/A f16 planes (rounding via
    // `from_f32`), and runs the exact direct `gbrapf16_*` / `gbrpf16_*`
    // kernels, so every output stays byte-identical to a direct `Gbrapf16`
    // conversion of the f32 block-mean rounded to f16. The span kind picks
    // the engine: the float area stream or the signed-coefficient filter
    // stream (PIL parity, straight alpha only — premultiplied has no filter
    // analogue, so it stays on the area tail, which un-premultiplies). There
    // is no `FilterStream<f16>`, so the filter path also bins in f32; f16 is
    // full-range, so there is no native-depth clamp on the filter output.
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
        rgba_plane_scratch_f16,
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
          return packed_rgba_f16_resample(
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
            rgba_plane_scratch_f16,
            w,
            plan,
            idx,
            use_simd,
            alpha_mode,
            GBR_F16_LUMA_MATRIX,
            GBR_F16_FULL_RANGE,
            |dst| {
              scatter_gbrapf16_to_packed_rgba_f32::<BE>(g_in, b_in, r_in, a_in, dst, w, use_simd)
            },
          );
        }
        crate::resample::SpanKind::Filter => {
          // Premultiplied alpha has no filter analogue (the filter engine
          // cannot un-premultiply); route to the area tail, which rejects the
          // filter plan and surfaces the typed `UnsupportedFilter` rather than
          // emit straight-filtered premultiplied color. Straight alpha filters
          // all four channels independently (PIL RGBA semantics).
          if alpha_mode.is_premultiplied() {
            return packed_rgba_f16_resample(
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
              rgba_plane_scratch_f16,
              w,
              plan,
              idx,
              use_simd,
              alpha_mode,
              GBR_F16_LUMA_MATRIX,
              GBR_F16_FULL_RANGE,
              |dst| {
                scatter_gbrapf16_to_packed_rgba_f32::<BE>(g_in, b_in, r_in, a_in, dst, w, use_simd)
              },
            );
          }
          return packed_rgba_f16_filter_resample(
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
            rgba_plane_scratch_f16,
            w,
            plan,
            idx,
            use_simd,
            GBR_F16_LUMA_MATRIX,
            GBR_F16_FULL_RANGE,
            |dst| {
              scatter_gbrapf16_to_packed_rgba_f32::<BE>(g_in, b_in, r_in, a_in, dst, w, use_simd)
            },
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

    // ---- Lossless f16 native paths (no conversion) -----------------------

    if let Some(buf) = self.rgb_f16.as_deref_mut() {
      // rgb_f16: no source α — use the no-α kernel (lossless scatter).
      let start = one_plane_start * 3;
      let end = one_plane_end * 3;
      gbrpf16_to_rgb_f16_row::<BE>(g_in, b_in, r_in, &mut buf[start..end], w, use_simd);
    }

    if let Some(buf) = self.rgba_f16.as_deref_mut() {
      // rgba_f16: source α included losslessly via gbrapf16_to_rgba_f16_row.
      let start = one_plane_start * 4;
      let end = one_plane_end
        .checked_mul(4)
        .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
          w, h, 4,
        )))?;
      gbrapf16_to_rgba_f16_row::<BE>(g_in, b_in, r_in, a_in, &mut buf[start..end], w, use_simd);
    }

    // ---- Paths that require widening f16 → f32 ---------------------------
    //
    // For Gbrapf16, RGBA outputs include the widened source α. RGB-only
    // outputs (rgb_f32, rgb_u16) discard α.

    let need_wide = self.rgb_f32.is_some()
      || self.rgba_f32.is_some()
      || self.rgb_u16.is_some()
      || self.rgba_u16.is_some()
      || self.luma.is_some()
      || self.luma_u16.is_some()
      || self.hsv.is_some();

    if need_wide {
      let mut gf_chunk = [0.0f32; WIDEN_CHUNK];
      let mut bf_chunk = [0.0f32; WIDEN_CHUNK];
      let mut rf_chunk = [0.0f32; WIDEN_CHUNK];
      let mut af_chunk = [0.0f32; WIDEN_CHUNK];
      let mut offset = 0;
      while offset < w {
        let n = (w - offset).min(WIDEN_CHUNK);
        // Bit-normalise LE-encoded f16 plane bits → host-native f32 (see the
        // canonical helper's docs); downstream kernel uses `BE = false`.
        widen_f16_be_to_host_f32::<BE>(g_in, offset, &mut gf_chunk, n);
        widen_f16_be_to_host_f32::<BE>(b_in, offset, &mut bf_chunk, n);
        widen_f16_be_to_host_f32::<BE>(r_in, offset, &mut rf_chunk, n);
        widen_f16_be_to_host_f32::<BE>(a_in, offset, &mut af_chunk, n);
        let gf = &gf_chunk[..n];
        let bf = &bf_chunk[..n];
        let rf = &rf_chunk[..n];
        let af = &af_chunk[..n];

        let chunk_plane_start = one_plane_start + offset;
        let chunk_plane_end = chunk_plane_start + n;

        if let Some(buf) = self.rgb_f32.as_deref_mut() {
          let start = chunk_plane_start * 3;
          let end = chunk_plane_end * 3;
          gbrpf32_to_rgb_f32_row::<HOST_NATIVE_BE>(gf, bf, rf, &mut buf[start..end], n, use_simd);
        }

        if let Some(buf) = self.rgba_f32.as_deref_mut() {
          // gbrapf32_to_rgba_f32_row with widened source α (lossless).
          let start = chunk_plane_start * 4;
          let end = chunk_plane_end * 4;
          gbrapf32_to_rgba_f32_row::<HOST_NATIVE_BE>(
            gf,
            bf,
            rf,
            af,
            &mut buf[start..end],
            n,
            use_simd,
          );
        }

        if let Some(buf) = self.rgb_u16.as_deref_mut() {
          let start = chunk_plane_start * 3;
          let end = chunk_plane_end * 3;
          gbrpf32_to_rgb_u16_row::<HOST_NATIVE_BE>(gf, bf, rf, &mut buf[start..end], n, use_simd);
        }

        if let Some(buf) = self.rgba_u16.as_deref_mut() {
          // gbrapf32_to_rgba_u16_row with widened source α.
          let start = chunk_plane_start * 4;
          let end = chunk_plane_end * 4;
          gbrapf32_to_rgba_u16_row::<HOST_NATIVE_BE>(
            gf,
            bf,
            rf,
            af,
            &mut buf[start..end],
            n,
            use_simd,
          );
        }

        if let Some(buf) = self.luma.as_deref_mut() {
          gbrpf32_to_luma_row::<HOST_NATIVE_BE>(
            gf,
            bf,
            rf,
            &mut buf[chunk_plane_start..chunk_plane_end],
            n,
            GBR_F16_LUMA_MATRIX,
            GBR_F16_FULL_RANGE,
            use_simd,
          );
        }

        if let Some(buf) = self.luma_u16.as_deref_mut() {
          gbrpf32_to_luma_u16_row::<HOST_NATIVE_BE>(
            gf,
            bf,
            rf,
            &mut buf[chunk_plane_start..chunk_plane_end],
            n,
            GBR_F16_LUMA_MATRIX,
            GBR_F16_FULL_RANGE,
            use_simd,
          );
        }

        if let Some(hsv) = self.hsv.as_mut() {
          let (h, s, v) = hsv.hsv();
          gbrpf32_to_hsv_row::<HOST_NATIVE_BE>(
            gf,
            bf,
            rf,
            &mut h[chunk_plane_start..chunk_plane_end],
            &mut s[chunk_plane_start..chunk_plane_end],
            &mut v[chunk_plane_start..chunk_plane_end],
            n,
            use_simd,
          );
        }

        offset += n;
      }
    }

    // ---- u8 RGBA standalone fast path (source α from f16 A plane) --------
    //
    // For standalone RGBA without other u8-dependent outputs, widen the α
    // plane to f32 per-row and use the existing copy_alpha_plane_f32_to_u8.
    // The chroma planes are converted via gbrpf16_to_rgba_row first (which
    // writes opaque α = 0xFF), then α is overwritten from the source.

    let want_rgba = self.rgba.is_some();
    let want_rgb = self.rgb.is_some();
    let need_u8_rgb = want_rgb;

    if want_rgba && !need_u8_rgb {
      let rgba_buf = self.rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      // Write opaque RGB → RGBA (α = 0xFF), then overwrite α from source.
      gbrpf16_to_rgba_row::<BE>(g_in, b_in, r_in, rgba_row, w, use_simd);
      // Scatter f16 α → u8 slot 3: widen + clamp + scale.
      widen_and_scatter_f16_alpha_to_u8::<BE>(a_in, rgba_row, w);
      return Ok(());
    }

    if !need_u8_rgb && !want_rgba {
      return Ok(());
    }

    // ---- Stage u8 RGB once for RGBA fan-out ------------------------------

    let Self {
      rgb,
      rgba,
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
    gbrpf16_to_rgb_row::<BE>(g_in, b_in, r_in, rgb_row, w, use_simd);

    // Strategy A+: expand RGB → RGBA (0xFF stub), then overwrite α from source.
    if let Some(buf) = rgba.as_deref_mut() {
      let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
      widen_and_scatter_f16_alpha_to_u8::<BE>(a_in, rgba_row, w);
    }

    Ok(())
  }
}

/// Widen a `half::f16` α plane to f32, clamp to `[0, 1]`, scale x 255,
/// and scatter into the RGBA slot 3 of a u8 RGBA buffer.
///
/// Used by `Gbrapf16` Strategy A+ and standalone-RGBA paths to overwrite
/// the per-pixel alpha byte from the f16 source α plane.
///
/// Endian routing: `widen_f16_be_to_host_f32::<BE>` converts the
/// `Gbrapf16Frame` α plane bits (LE if `BE = false`, BE if `BE = true`)
/// into **host-native f32** scratch. The downstream
/// `copy_alpha_plane_f32_to_u8` therefore receives host-native f32 input,
/// regardless of source byte order, and must be invoked with
/// `BE = HOST_NATIVE_BE` so the kernel's `from_le` / `from_be` is a no-op
/// on every host (no second byte-swap). This is the **post-widen** routing
/// pattern; contrast with `planar_gbr_float.rs` which calls the same
/// helper with `BE = <source BE>` because it consumes the **direct**
/// `Gbrapf32Frame` α plane in source byte order.
#[cfg_attr(not(tarpaulin), inline(always))]
fn widen_and_scatter_f16_alpha_to_u8<const BE: bool>(
  alpha_f16: &[half::f16],
  rgba_out: &mut [u8],
  width: usize,
) {
  let mut af_chunk = [0.0f32; WIDEN_CHUNK];
  let mut offset = 0;
  while offset < width {
    let n = (width - offset).min(WIDEN_CHUNK);
    // Bit-normalise source-encoded f16 α bits → host-native f32 before
    // clamping and scaling to u8 — correct on both LE and BE hosts.
    widen_f16_be_to_host_f32::<BE>(alpha_f16, offset, &mut af_chunk, n);
    copy_alpha_plane_f32_to_u8::<HOST_NATIVE_BE>(&af_chunk[..n], &mut rgba_out[offset * 4..], n);
    offset += n;
  }
}
