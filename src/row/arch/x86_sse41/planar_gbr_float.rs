//! SSE4.1 SIMD kernels for planar GBR float sources (Tier 10).
//!
//! f32 path: 4 pixels / iteration via `__m128` (one G, B, R [+ A] register
//! each). f16 narrowing / widening gated on the x86 `f16c` feature (detected
//! at runtime via `is_x86_feature_detected!("f16c")`); scalar fallback used
//! when F16C is absent.
//!
//! # Rounding (f32 → u8 / u16)
//!
//! `_mm_add_ps(scaled, _mm_set1_ps(0.5))` then `_mm_cvttps_epi32` (truncate
//! toward zero). This is the round-half-up contract shared with the scalar
//! kernels — MXCSR-independent and consistent with the Grayf32 path.
//!
//! **Do NOT use `_MM_FROUND_TO_NEAREST_INT` for integer narrowing** —
//! that gives banker's rounding, not round-half-up.
//!
//! # Rounding (f32 → f16)
//!
//! F16C `_mm_cvtps_ph::<{ _MM_FROUND_TO_NEAREST_INT }>`:
//! IEEE-754 round-to-nearest-even, matching `half::f16::from_f32`. This IS
//! correct for f16 narrowing (different semantic from integer narrowing).
//!
//! The intel-spec recommendation `_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC`
//! cannot be expressed because the Rust stdarch
//! `static_assert_uimm_bits!(IMM_ROUNDING, 3)` rejects the OR result (= 8). For
//! normalized RGB inputs in `[0, 1]` the FP-exception suppression
//! (`_MM_FROUND_NO_EXC`) is a no-op anyway, so the bare
//! `_MM_FROUND_TO_NEAREST_INT` is functionally equivalent.
//!
//! # f16 lossless interleave
//!
//! Treat f16 lanes as opaque `u16` — no arithmetic, no F16C gate needed.

#[cfg_attr(miri, allow(unused_imports))]
use core::arch::x86_64::*;

use crate::{
  KernelMatrix,
  row::{
    arch::x86_sse41::endian,
    scalar::{planar_gbr_f16 as scalar_f16, planar_gbr_float as scalar},
  },
};

/// `BE` value that makes the downstream `scalar::gbrpf32_to_*` kernels treat
/// their `f32` scratch input as **host-native** (no `from_be` / `from_le`
/// byte-swap). After we widen f16 → f32 via
/// [`scalar_f16::widen_f16_be_to_host_f32`] (which normalizes the source
/// f16 bits per the source `BE` and produces host-native f32), the resulting
/// scratch must be routed via `HOST_NATIVE_BE` so the downstream kernel's
/// `from_le` / `from_be` loaders no-op the swap. Without this routing the
/// SIMD scalar tail double-byte-swaps on `BE`-source-on-LE-host (and
/// symmetrically `LE`-source-on-BE-host).
const HOST_NATIVE_BE: bool = cfg!(target_endian = "big");

// ---- shared helpers ----------------------------------------------------------

/// Clamp a `__m128` (f32x4) to `[0.0, 1.0]`.
#[inline(always)]
unsafe fn clamp01(v: __m128, zero: __m128, one: __m128) -> __m128 {
  unsafe { _mm_min_ps(_mm_max_ps(v, zero), one) }
}

/// Scale, add 0.5, truncate → `__m128i` (i32x4). Round-half-up.
#[inline(always)]
unsafe fn scale_round_i32(v: __m128, scale: __m128) -> __m128i {
  unsafe { _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(v, scale), _mm_set1_ps(0.5))) }
}

/// Extract 4 bytes from the low lane of an i32x4 result (after saturating
/// narrow i32→i16→u8). Returns `[v0, v1, v2, v3]`.
#[inline(always)]
unsafe fn i32x4_to_u8x4(i32v: __m128i) -> [u8; 4] {
  unsafe {
    let pack16 = _mm_packs_epi32(i32v, i32v);
    let pack8 = _mm_packus_epi16(pack16, pack16);
    [
      _mm_extract_epi8::<0>(pack8) as u8,
      _mm_extract_epi8::<1>(pack8) as u8,
      _mm_extract_epi8::<2>(pack8) as u8,
      _mm_extract_epi8::<3>(pack8) as u8,
    ]
  }
}

/// Extract 4 u16 values from an i32x4 result (for u16 output).
#[inline(always)]
unsafe fn i32x4_to_u16x4(i32v: __m128i) -> [u16; 4] {
  unsafe {
    [
      _mm_extract_epi32::<0>(i32v) as u16,
      _mm_extract_epi32::<1>(i32v) as u16,
      _mm_extract_epi32::<2>(i32v) as u16,
      _mm_extract_epi32::<3>(i32v) as u16,
    ]
  }
}

// ---- Gbrpf32 → u8 RGB -------------------------------------------------------

/// SSE4.1: planar Gbrpf32 → packed `R, G, B` bytes. 4 px / iter.
///
/// Round-half-up: `+ 0.5` then `_mm_cvttps_epi32`.
///
/// # Safety
///
/// 1. SSE4.1 must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `3 * width`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn gbrpf32_to_rgb_row<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  out: &mut [u8],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(out.len() >= width * 3, "out row too short");

  unsafe {
    let zero = _mm_setzero_ps();
    let one = _mm_set1_ps(1.0);
    let scale = _mm_set1_ps(255.0);

    let mut x = 0usize;
    while x + 4 <= width {
      let gv = clamp01(
        _mm_castsi128_ps(endian::load_endian_u32x4::<BE>(
          g.as_ptr().add(x).cast::<u8>(),
        )),
        zero,
        one,
      );
      let bv = clamp01(
        _mm_castsi128_ps(endian::load_endian_u32x4::<BE>(
          b.as_ptr().add(x).cast::<u8>(),
        )),
        zero,
        one,
      );
      let rv = clamp01(
        _mm_castsi128_ps(endian::load_endian_u32x4::<BE>(
          r.as_ptr().add(x).cast::<u8>(),
        )),
        zero,
        one,
      );
      let gi = i32x4_to_u8x4(scale_round_i32(gv, scale));
      let bi = i32x4_to_u8x4(scale_round_i32(bv, scale));
      let ri = i32x4_to_u8x4(scale_round_i32(rv, scale));
      let base = x * 3;
      for p in 0..4 {
        out[base + p * 3] = ri[p];
        out[base + p * 3 + 1] = gi[p];
        out[base + p * 3 + 2] = bi[p];
      }
      x += 4;
    }
    if x < width {
      scalar::gbrpf32_to_rgb_row::<BE>(&g[x..], &b[x..], &r[x..], &mut out[x * 3..], width - x);
    }
  }
}

// ---- Gbrpf32 → u8 RGBA (opaque α) ------------------------------------------

/// SSE4.1: planar Gbrpf32 → packed `R, G, B, A` bytes (α = 0xFF). 4 px / iter.
///
/// # Safety
///
/// 1. SSE4.1 must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn gbrpf32_to_rgba_row<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  out: &mut [u8],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(out.len() >= width * 4, "out row too short");

  unsafe {
    let zero = _mm_setzero_ps();
    let one = _mm_set1_ps(1.0);
    let scale = _mm_set1_ps(255.0);

    let mut x = 0usize;
    while x + 4 <= width {
      let gv = clamp01(
        _mm_castsi128_ps(endian::load_endian_u32x4::<BE>(
          g.as_ptr().add(x).cast::<u8>(),
        )),
        zero,
        one,
      );
      let bv = clamp01(
        _mm_castsi128_ps(endian::load_endian_u32x4::<BE>(
          b.as_ptr().add(x).cast::<u8>(),
        )),
        zero,
        one,
      );
      let rv = clamp01(
        _mm_castsi128_ps(endian::load_endian_u32x4::<BE>(
          r.as_ptr().add(x).cast::<u8>(),
        )),
        zero,
        one,
      );
      let gi = i32x4_to_u8x4(scale_round_i32(gv, scale));
      let bi = i32x4_to_u8x4(scale_round_i32(bv, scale));
      let ri = i32x4_to_u8x4(scale_round_i32(rv, scale));
      let base = x * 4;
      for p in 0..4 {
        out[base + p * 4] = ri[p];
        out[base + p * 4 + 1] = gi[p];
        out[base + p * 4 + 2] = bi[p];
        out[base + p * 4 + 3] = 0xFF;
      }
      x += 4;
    }
    if x < width {
      scalar::gbrpf32_to_rgba_row::<BE>(&g[x..], &b[x..], &r[x..], &mut out[x * 4..], width - x);
    }
  }
}

// ---- Gbrpf32 → u16 RGB ------------------------------------------------------

/// SSE4.1: planar Gbrpf32 → packed `R, G, B` u16. 4 px / iter.
///
/// # Safety
///
/// 1. SSE4.1 must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `3 * width`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn gbrpf32_to_rgb_u16_row<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  out: &mut [u16],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(out.len() >= width * 3, "out row too short");

  unsafe {
    let zero = _mm_setzero_ps();
    let one = _mm_set1_ps(1.0);
    let scale = _mm_set1_ps(65535.0);

    let mut x = 0usize;
    while x + 4 <= width {
      let gv = clamp01(
        _mm_castsi128_ps(endian::load_endian_u32x4::<BE>(
          g.as_ptr().add(x).cast::<u8>(),
        )),
        zero,
        one,
      );
      let bv = clamp01(
        _mm_castsi128_ps(endian::load_endian_u32x4::<BE>(
          b.as_ptr().add(x).cast::<u8>(),
        )),
        zero,
        one,
      );
      let rv = clamp01(
        _mm_castsi128_ps(endian::load_endian_u32x4::<BE>(
          r.as_ptr().add(x).cast::<u8>(),
        )),
        zero,
        one,
      );
      let gu = i32x4_to_u16x4(scale_round_i32(gv, scale));
      let bu = i32x4_to_u16x4(scale_round_i32(bv, scale));
      let ru = i32x4_to_u16x4(scale_round_i32(rv, scale));
      let base = x * 3;
      for p in 0..4 {
        out[base + p * 3] = ru[p];
        out[base + p * 3 + 1] = gu[p];
        out[base + p * 3 + 2] = bu[p];
      }
      x += 4;
    }
    if x < width {
      scalar::gbrpf32_to_rgb_u16_row::<BE>(&g[x..], &b[x..], &r[x..], &mut out[x * 3..], width - x);
    }
  }
}

// ---- Gbrpf32 → u16 RGBA (opaque α) -----------------------------------------

/// SSE4.1: planar Gbrpf32 → packed `R, G, B, A` u16 (α = 0xFFFF). 4 px / iter.
///
/// # Safety
///
/// 1. SSE4.1 must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn gbrpf32_to_rgba_u16_row<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  out: &mut [u16],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(out.len() >= width * 4, "out row too short");

  unsafe {
    let zero = _mm_setzero_ps();
    let one = _mm_set1_ps(1.0);
    let scale = _mm_set1_ps(65535.0);

    let mut x = 0usize;
    while x + 4 <= width {
      let gv = clamp01(
        _mm_castsi128_ps(endian::load_endian_u32x4::<BE>(
          g.as_ptr().add(x).cast::<u8>(),
        )),
        zero,
        one,
      );
      let bv = clamp01(
        _mm_castsi128_ps(endian::load_endian_u32x4::<BE>(
          b.as_ptr().add(x).cast::<u8>(),
        )),
        zero,
        one,
      );
      let rv = clamp01(
        _mm_castsi128_ps(endian::load_endian_u32x4::<BE>(
          r.as_ptr().add(x).cast::<u8>(),
        )),
        zero,
        one,
      );
      let gu = i32x4_to_u16x4(scale_round_i32(gv, scale));
      let bu = i32x4_to_u16x4(scale_round_i32(bv, scale));
      let ru = i32x4_to_u16x4(scale_round_i32(rv, scale));
      let base = x * 4;
      for p in 0..4 {
        out[base + p * 4] = ru[p];
        out[base + p * 4 + 1] = gu[p];
        out[base + p * 4 + 2] = bu[p];
        out[base + p * 4 + 3] = 0xFFFF;
      }
      x += 4;
    }
    if x < width {
      scalar::gbrpf32_to_rgba_u16_row::<BE>(
        &g[x..],
        &b[x..],
        &r[x..],
        &mut out[x * 4..],
        width - x,
      );
    }
  }
}

// ---- Gbrpf32 → f32 RGB (lossless) ------------------------------------------

/// SSE4.1: planar Gbrpf32 → packed `R, G, B` f32 (lossless interleave). 4 px / iter.
///
/// # Safety
///
/// 1. SSE4.1 must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `3 * width`.
#[inline]
#[target_feature(enable = "sse4.1")]
#[allow(dead_code)] // dispatcher delegates to scalar for lossless f32 interleave
pub(crate) unsafe fn gbrpf32_to_rgb_f32_row<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  out: &mut [f32],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(out.len() >= width * 3, "out row too short");

  // SSE4.1 has no vst3-style intrinsic; use scalar (well-vectorised by compiler).
  scalar::gbrpf32_to_rgb_f32_row::<BE>(g, b, r, out, width);
}

// ---- Gbrpf32 → f32 RGBA (lossless, α = 1.0) ---------------------------------

/// SSE4.1: planar Gbrpf32 → packed `R, G, B, A` f32 (lossless, α = 1.0).
///
/// # Safety
///
/// 1. SSE4.1 must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "sse4.1")]
#[allow(dead_code)] // dispatcher delegates to scalar for lossless f32 interleave
pub(crate) unsafe fn gbrpf32_to_rgba_f32_row<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  out: &mut [f32],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(out.len() >= width * 4, "out row too short");

  // SSE4.1 has no vst4-style intrinsic; use scalar.
  scalar::gbrpf32_to_rgba_f32_row::<BE>(g, b, r, out, width);
}

// ---- Gbrpf32 → f16 RGB (F16C narrow) ----------------------------------------

/// SSE4.1 + F16C: planar Gbrpf32 → packed `R, G, B` f16 (fused narrow). 4 px / iter.
///
/// Uses `_mm_cvtps_ph` with `_MM_FROUND_TO_NEAREST_INT`
/// (IEEE-754 round-to-nearest-even).
///
/// # Safety
///
/// 1. SSE4.1 and F16C must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `3 * width`.
#[inline]
#[target_feature(enable = "sse4.1,f16c")]
pub(crate) unsafe fn gbrpf32_to_rgb_f16_row_f16c<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  out: &mut [half::f16],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(out.len() >= width * 3, "out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 4 <= width {
      let gv = _mm_castsi128_ps(endian::load_endian_u32x4::<BE>(
        g.as_ptr().add(x).cast::<u8>(),
      ));
      let bv = _mm_castsi128_ps(endian::load_endian_u32x4::<BE>(
        b.as_ptr().add(x).cast::<u8>(),
      ));
      let rv = _mm_castsi128_ps(endian::load_endian_u32x4::<BE>(
        r.as_ptr().add(x).cast::<u8>(),
      ));
      // F16C narrow: IEEE-754 round-to-nearest-even (NOT round-half-up).
      let gh = _mm_cvtps_ph::<{ _MM_FROUND_TO_NEAREST_INT }>(gv);
      let bh = _mm_cvtps_ph::<{ _MM_FROUND_TO_NEAREST_INT }>(bv);
      let rh = _mm_cvtps_ph::<{ _MM_FROUND_TO_NEAREST_INT }>(rv);
      // Extract 4 u16 lanes from each __m128i (low 64 bits = 4 f16).
      let mut rh_buf = [0u16; 4];
      let mut gh_buf = [0u16; 4];
      let mut bh_buf = [0u16; 4];
      _mm_storel_epi64(rh_buf.as_mut_ptr().cast(), rh);
      _mm_storel_epi64(gh_buf.as_mut_ptr().cast(), gh);
      _mm_storel_epi64(bh_buf.as_mut_ptr().cast(), bh);
      // Scatter 4 RGB triples.
      let base = x * 3;
      for p in 0..4 {
        let dst = out.as_mut_ptr().add(base + p * 3);
        *dst.cast::<u16>() = rh_buf[p];
        *dst.add(1).cast::<u16>() = gh_buf[p];
        *dst.add(2).cast::<u16>() = bh_buf[p];
      }
      x += 4;
    }
    if x < width {
      scalar::gbrpf32_to_rgb_f16_row::<BE>(&g[x..], &b[x..], &r[x..], &mut out[x * 3..], width - x);
    }
  }
}

// ---- Gbrpf32 → f16 RGBA (F16C narrow, α = f16(1.0)) -------------------------

/// SSE4.1 + F16C: planar Gbrpf32 → packed `R, G, B, A` f16 (fused narrow,
/// α = f16(1.0) = 0x3C00). 4 px / iter.
///
/// # Safety
///
/// 1. SSE4.1 and F16C must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "sse4.1,f16c")]
pub(crate) unsafe fn gbrpf32_to_rgba_f16_row_f16c<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  out: &mut [half::f16],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(out.len() >= width * 4, "out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 4 <= width {
      let gv = _mm_castsi128_ps(endian::load_endian_u32x4::<BE>(
        g.as_ptr().add(x).cast::<u8>(),
      ));
      let bv = _mm_castsi128_ps(endian::load_endian_u32x4::<BE>(
        b.as_ptr().add(x).cast::<u8>(),
      ));
      let rv = _mm_castsi128_ps(endian::load_endian_u32x4::<BE>(
        r.as_ptr().add(x).cast::<u8>(),
      ));
      let gh = _mm_cvtps_ph::<{ _MM_FROUND_TO_NEAREST_INT }>(gv);
      let bh = _mm_cvtps_ph::<{ _MM_FROUND_TO_NEAREST_INT }>(bv);
      let rh = _mm_cvtps_ph::<{ _MM_FROUND_TO_NEAREST_INT }>(rv);
      let mut rh_buf = [0u16; 4];
      let mut gh_buf = [0u16; 4];
      let mut bh_buf = [0u16; 4];
      _mm_storel_epi64(rh_buf.as_mut_ptr().cast(), rh);
      _mm_storel_epi64(gh_buf.as_mut_ptr().cast(), gh);
      _mm_storel_epi64(bh_buf.as_mut_ptr().cast(), bh);
      let base = x * 4;
      for p in 0..4 {
        let dst = out.as_mut_ptr().add(base + p * 4);
        *dst.cast::<u16>() = rh_buf[p];
        *dst.add(1).cast::<u16>() = gh_buf[p];
        *dst.add(2).cast::<u16>() = bh_buf[p];
        *dst.add(3).cast::<u16>() = 0x3C00u16; // f16(1.0)
      }
      x += 4;
    }
    if x < width {
      scalar::gbrpf32_to_rgba_f16_row::<BE>(
        &g[x..],
        &b[x..],
        &r[x..],
        &mut out[x * 4..],
        width - x,
      );
    }
  }
}

// ---- Gbrpf32 → u8 luma (staged via RGB scratch) ----------------------------

/// SSE4.1: planar Gbrpf32 → u8 luma (staged via SSE4.1 RGB kernel).
///
/// # Safety
///
/// 1. SSE4.1 must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `width`.
#[inline]
#[target_feature(enable = "sse4.1")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn gbrpf32_to_luma_row<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(out.len() >= width, "out row too short");

  const CHUNK: usize = 64;
  let mut scratch = [0u8; CHUNK * 3];
  let mut offset = 0;
  while offset < width {
    let n = (width - offset).min(CHUNK);
    unsafe {
      gbrpf32_to_rgb_row::<BE>(
        &g[offset..],
        &b[offset..],
        &r[offset..],
        &mut scratch[..n * 3],
        n,
      );
    }
    crate::row::scalar::rgb_to_luma_row(
      &scratch[..n * 3],
      &mut out[offset..offset + n],
      n,
      matrix,
      full_range,
    );
    offset += n;
  }
}

// ---- Gbrpf32 → u16 luma (staged via RGB scratch) ---------------------------

/// SSE4.1: planar Gbrpf32 → u16 luma (staged via SSE4.1 RGB kernel).
///
/// # Safety
///
/// 1. SSE4.1 must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `width`.
#[inline]
#[target_feature(enable = "sse4.1")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn gbrpf32_to_luma_u16_row<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(out.len() >= width, "out row too short");

  const CHUNK: usize = 64;
  let mut scratch = [0u8; CHUNK * 3];
  let mut offset = 0;
  while offset < width {
    let n = (width - offset).min(CHUNK);
    unsafe {
      gbrpf32_to_rgb_row::<BE>(
        &g[offset..],
        &b[offset..],
        &r[offset..],
        &mut scratch[..n * 3],
        n,
      );
    }
    crate::row::scalar::rgb_to_luma_u16_row(
      &scratch[..n * 3],
      &mut out[offset..offset + n],
      n,
      matrix,
      full_range,
    );
    offset += n;
  }
}

// ---- Gbrpf32 → HSV (staged via RGB scratch) ---------------------------------

/// SSE4.1: planar Gbrpf32 → planar HSV bytes (staged via SSE4.1 RGB kernel).
///
/// # Safety
///
/// 1. SSE4.1 must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `h_out.len()`, `s_out.len()`, `v_out.len()` ≥ `width`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn gbrpf32_to_hsv_row<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(h_out.len() >= width, "h_out row too short");
  debug_assert!(s_out.len() >= width, "s_out row too short");
  debug_assert!(v_out.len() >= width, "v_out row too short");

  const CHUNK: usize = 64;
  let mut scratch = [0u8; CHUNK * 3];
  let mut offset = 0;
  while offset < width {
    let n = (width - offset).min(CHUNK);
    unsafe {
      gbrpf32_to_rgb_row::<BE>(
        &g[offset..],
        &b[offset..],
        &r[offset..],
        &mut scratch[..n * 3],
        n,
      );
    }
    crate::row::scalar::rgb_to_hsv_row(
      &scratch[..n * 3],
      &mut h_out[offset..offset + n],
      &mut s_out[offset..offset + n],
      &mut v_out[offset..offset + n],
      n,
    );
    offset += n;
  }
}

// ---- Gbrapf32 → u8 RGBA (source α) -----------------------------------------

/// SSE4.1: planar Gbrapf32 → packed `R, G, B, A` bytes (source α). 4 px / iter.
///
/// # Safety
///
/// 1. SSE4.1 must be available.
/// 2. `g.len()`, `b.len()`, `r.len()`, `a.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn gbrapf32_to_rgba_row<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  a: &[f32],
  out: &mut [u8],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(a.len() >= width, "a row too short");
  debug_assert!(out.len() >= width * 4, "out row too short");

  unsafe {
    let zero = _mm_setzero_ps();
    let one = _mm_set1_ps(1.0);
    let scale = _mm_set1_ps(255.0);

    let mut x = 0usize;
    while x + 4 <= width {
      let gv = clamp01(
        _mm_castsi128_ps(endian::load_endian_u32x4::<BE>(
          g.as_ptr().add(x).cast::<u8>(),
        )),
        zero,
        one,
      );
      let bv = clamp01(
        _mm_castsi128_ps(endian::load_endian_u32x4::<BE>(
          b.as_ptr().add(x).cast::<u8>(),
        )),
        zero,
        one,
      );
      let rv = clamp01(
        _mm_castsi128_ps(endian::load_endian_u32x4::<BE>(
          r.as_ptr().add(x).cast::<u8>(),
        )),
        zero,
        one,
      );
      let av = clamp01(
        _mm_castsi128_ps(endian::load_endian_u32x4::<BE>(
          a.as_ptr().add(x).cast::<u8>(),
        )),
        zero,
        one,
      );
      let gi = i32x4_to_u8x4(scale_round_i32(gv, scale));
      let bi = i32x4_to_u8x4(scale_round_i32(bv, scale));
      let ri = i32x4_to_u8x4(scale_round_i32(rv, scale));
      let ai = i32x4_to_u8x4(scale_round_i32(av, scale));
      let base = x * 4;
      for p in 0..4 {
        out[base + p * 4] = ri[p];
        out[base + p * 4 + 1] = gi[p];
        out[base + p * 4 + 2] = bi[p];
        out[base + p * 4 + 3] = ai[p];
      }
      x += 4;
    }
    if x < width {
      scalar::gbrapf32_to_rgba_row::<BE>(
        &g[x..],
        &b[x..],
        &r[x..],
        &a[x..],
        &mut out[x * 4..],
        width - x,
      );
    }
  }
}

// ---- Gbrapf32 → u16 RGBA (source α) ----------------------------------------

/// SSE4.1: planar Gbrapf32 → packed `R, G, B, A` u16 (source α). 4 px / iter.
///
/// # Safety
///
/// 1. SSE4.1 must be available.
/// 2. `g.len()`, `b.len()`, `r.len()`, `a.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn gbrapf32_to_rgba_u16_row<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  a: &[f32],
  out: &mut [u16],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(a.len() >= width, "a row too short");
  debug_assert!(out.len() >= width * 4, "out row too short");

  unsafe {
    let zero = _mm_setzero_ps();
    let one = _mm_set1_ps(1.0);
    let scale = _mm_set1_ps(65535.0);

    let mut x = 0usize;
    while x + 4 <= width {
      let gv = clamp01(
        _mm_castsi128_ps(endian::load_endian_u32x4::<BE>(
          g.as_ptr().add(x).cast::<u8>(),
        )),
        zero,
        one,
      );
      let bv = clamp01(
        _mm_castsi128_ps(endian::load_endian_u32x4::<BE>(
          b.as_ptr().add(x).cast::<u8>(),
        )),
        zero,
        one,
      );
      let rv = clamp01(
        _mm_castsi128_ps(endian::load_endian_u32x4::<BE>(
          r.as_ptr().add(x).cast::<u8>(),
        )),
        zero,
        one,
      );
      let av = clamp01(
        _mm_castsi128_ps(endian::load_endian_u32x4::<BE>(
          a.as_ptr().add(x).cast::<u8>(),
        )),
        zero,
        one,
      );
      let gu = i32x4_to_u16x4(scale_round_i32(gv, scale));
      let bu = i32x4_to_u16x4(scale_round_i32(bv, scale));
      let ru = i32x4_to_u16x4(scale_round_i32(rv, scale));
      let au = i32x4_to_u16x4(scale_round_i32(av, scale));
      let base = x * 4;
      for p in 0..4 {
        out[base + p * 4] = ru[p];
        out[base + p * 4 + 1] = gu[p];
        out[base + p * 4 + 2] = bu[p];
        out[base + p * 4 + 3] = au[p];
      }
      x += 4;
    }
    if x < width {
      scalar::gbrapf32_to_rgba_u16_row::<BE>(
        &g[x..],
        &b[x..],
        &r[x..],
        &a[x..],
        &mut out[x * 4..],
        width - x,
      );
    }
  }
}

// ---- Gbrapf32 → f32 RGBA (lossless source α) --------------------------------

/// SSE4.1: planar Gbrapf32 → packed `R, G, B, A` f32 (lossless, source α).
///
/// # Safety
///
/// 1. SSE4.1 must be available.
/// 2. `g.len()`, `b.len()`, `r.len()`, `a.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "sse4.1")]
#[allow(dead_code)] // dispatcher delegates to scalar for lossless f32 interleave
pub(crate) unsafe fn gbrapf32_to_rgba_f32_row<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  a: &[f32],
  out: &mut [f32],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(a.len() >= width, "a row too short");
  debug_assert!(out.len() >= width * 4, "out row too short");

  // SSE4.1 has no 4-channel interleave store; use scalar.
  scalar::gbrapf32_to_rgba_f32_row::<BE>(g, b, r, a, out, width);
}

// ---- Gbrapf32 → f16 RGBA (F16C narrow, source α) ----------------------------

/// SSE4.1 + F16C: planar Gbrapf32 → packed `R, G, B, A` f16 (fused narrow,
/// source α). 4 px / iter.
///
/// # Safety
///
/// 1. SSE4.1 and F16C must be available.
/// 2. `g.len()`, `b.len()`, `r.len()`, `a.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "sse4.1,f16c")]
pub(crate) unsafe fn gbrapf32_to_rgba_f16_row_f16c<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  a: &[f32],
  out: &mut [half::f16],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(a.len() >= width, "a row too short");
  debug_assert!(out.len() >= width * 4, "out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 4 <= width {
      let gv = _mm_castsi128_ps(endian::load_endian_u32x4::<BE>(
        g.as_ptr().add(x).cast::<u8>(),
      ));
      let bv = _mm_castsi128_ps(endian::load_endian_u32x4::<BE>(
        b.as_ptr().add(x).cast::<u8>(),
      ));
      let rv = _mm_castsi128_ps(endian::load_endian_u32x4::<BE>(
        r.as_ptr().add(x).cast::<u8>(),
      ));
      let av = _mm_castsi128_ps(endian::load_endian_u32x4::<BE>(
        a.as_ptr().add(x).cast::<u8>(),
      ));
      let gh = _mm_cvtps_ph::<{ _MM_FROUND_TO_NEAREST_INT }>(gv);
      let bh = _mm_cvtps_ph::<{ _MM_FROUND_TO_NEAREST_INT }>(bv);
      let rh = _mm_cvtps_ph::<{ _MM_FROUND_TO_NEAREST_INT }>(rv);
      let ah = _mm_cvtps_ph::<{ _MM_FROUND_TO_NEAREST_INT }>(av);
      let mut rh_buf = [0u16; 4];
      let mut gh_buf = [0u16; 4];
      let mut bh_buf = [0u16; 4];
      let mut ah_buf = [0u16; 4];
      _mm_storel_epi64(rh_buf.as_mut_ptr().cast(), rh);
      _mm_storel_epi64(gh_buf.as_mut_ptr().cast(), gh);
      _mm_storel_epi64(bh_buf.as_mut_ptr().cast(), bh);
      _mm_storel_epi64(ah_buf.as_mut_ptr().cast(), ah);
      let base = x * 4;
      for p in 0..4 {
        let dst = out.as_mut_ptr().add(base + p * 4);
        *dst.cast::<u16>() = rh_buf[p];
        *dst.add(1).cast::<u16>() = gh_buf[p];
        *dst.add(2).cast::<u16>() = bh_buf[p];
        *dst.add(3).cast::<u16>() = ah_buf[p];
      }
      x += 4;
    }
    if x < width {
      scalar::gbrapf32_to_rgba_f16_row::<BE>(
        &g[x..],
        &b[x..],
        &r[x..],
        &a[x..],
        &mut out[x * 4..],
        width - x,
      );
    }
  }
}

// ---- Gbrpf16 → u8 RGB (F16C widen) -----------------------------------------

/// SSE4.1 + F16C: planar Gbrpf16 → packed `R, G, B` bytes (widen f16→f32,
/// then convert). 4 px / iter.
///
/// # Safety
///
/// 1. SSE4.1 and F16C must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `3 * width`.
#[inline]
#[target_feature(enable = "sse4.1,f16c")]
pub(crate) unsafe fn gbrpf16_to_rgb_row_f16c<const BE: bool>(
  g: &[half::f16],
  b: &[half::f16],
  r: &[half::f16],
  out: &mut [u8],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(out.len() >= width * 3, "out row too short");

  unsafe {
    let zero = _mm_setzero_ps();
    let one = _mm_set1_ps(1.0);
    let scale = _mm_set1_ps(255.0);

    let mut x = 0usize;
    while x + 4 <= width {
      // _mm_loadl_epi64: 64-bit load into the low half of __m128i (4 × u16 = 4 × f16).
      let gv = _mm_cvtph_ps(endian::load_endian_u16x4::<BE>(
        g.as_ptr().add(x).cast::<u8>(),
      ));
      let bv = _mm_cvtph_ps(endian::load_endian_u16x4::<BE>(
        b.as_ptr().add(x).cast::<u8>(),
      ));
      let rv = _mm_cvtph_ps(endian::load_endian_u16x4::<BE>(
        r.as_ptr().add(x).cast::<u8>(),
      ));
      let gc = clamp01(gv, zero, one);
      let bc = clamp01(bv, zero, one);
      let rc = clamp01(rv, zero, one);
      let gi = i32x4_to_u8x4(scale_round_i32(gc, scale));
      let bi = i32x4_to_u8x4(scale_round_i32(bc, scale));
      let ri = i32x4_to_u8x4(scale_round_i32(rc, scale));
      let base = x * 3;
      for p in 0..4 {
        out[base + p * 3] = ri[p];
        out[base + p * 3 + 1] = gi[p];
        out[base + p * 3 + 2] = bi[p];
      }
      x += 4;
    }
    if x < width {
      // Scalar tail: bit-normalize f16 → host-native f32 (via
      // `scalar_f16::widen_f16_be_to_host_f32::<BE>` which `from_be` /
      // `from_le`-loads the source bits BEFORE the f16 → f32 conversion),
      // then route the scalar kernel via `HOST_NATIVE_BE` to avoid double
      // byte-swap.
      let tail = width - x;
      let mut gf = [0.0f32; 4];
      let mut bf = [0.0f32; 4];
      let mut rf = [0.0f32; 4];
      scalar_f16::widen_f16_be_to_host_f32::<BE>(g, x, &mut gf, tail);
      scalar_f16::widen_f16_be_to_host_f32::<BE>(b, x, &mut bf, tail);
      scalar_f16::widen_f16_be_to_host_f32::<BE>(r, x, &mut rf, tail);
      scalar::gbrpf32_to_rgb_row::<HOST_NATIVE_BE>(
        &gf[..tail],
        &bf[..tail],
        &rf[..tail],
        &mut out[x * 3..],
        tail,
      );
    }
  }
}

// ---- Gbrpf16 → u8 RGBA (F16C widen) ----------------------------------------

/// SSE4.1 + F16C: planar Gbrpf16 → packed `R, G, B, A` bytes (widen f16→f32,
/// α = 0xFF). 4 px / iter.
///
/// # Safety
///
/// 1. SSE4.1 and F16C must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "sse4.1,f16c")]
pub(crate) unsafe fn gbrpf16_to_rgba_row_f16c<const BE: bool>(
  g: &[half::f16],
  b: &[half::f16],
  r: &[half::f16],
  out: &mut [u8],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(out.len() >= width * 4, "out row too short");

  unsafe {
    let zero = _mm_setzero_ps();
    let one = _mm_set1_ps(1.0);
    let scale = _mm_set1_ps(255.0);

    let mut x = 0usize;
    while x + 4 <= width {
      let gv = _mm_cvtph_ps(endian::load_endian_u16x4::<BE>(
        g.as_ptr().add(x).cast::<u8>(),
      ));
      let bv = _mm_cvtph_ps(endian::load_endian_u16x4::<BE>(
        b.as_ptr().add(x).cast::<u8>(),
      ));
      let rv = _mm_cvtph_ps(endian::load_endian_u16x4::<BE>(
        r.as_ptr().add(x).cast::<u8>(),
      ));
      let gc = clamp01(gv, zero, one);
      let bc = clamp01(bv, zero, one);
      let rc = clamp01(rv, zero, one);
      let gi = i32x4_to_u8x4(scale_round_i32(gc, scale));
      let bi = i32x4_to_u8x4(scale_round_i32(bc, scale));
      let ri = i32x4_to_u8x4(scale_round_i32(rc, scale));
      let base = x * 4;
      for p in 0..4 {
        out[base + p * 4] = ri[p];
        out[base + p * 4 + 1] = gi[p];
        out[base + p * 4 + 2] = bi[p];
        out[base + p * 4 + 3] = 0xFF;
      }
      x += 4;
    }
    if x < width {
      // Scalar tail: bit-normalize f16 → host-native f32, then route the
      // scalar kernel via `HOST_NATIVE_BE` to avoid double-byte-swap.
      let tail = width - x;
      let mut gf = [0.0f32; 4];
      let mut bf = [0.0f32; 4];
      let mut rf = [0.0f32; 4];
      scalar_f16::widen_f16_be_to_host_f32::<BE>(g, x, &mut gf, tail);
      scalar_f16::widen_f16_be_to_host_f32::<BE>(b, x, &mut bf, tail);
      scalar_f16::widen_f16_be_to_host_f32::<BE>(r, x, &mut rf, tail);
      scalar::gbrpf32_to_rgba_row::<HOST_NATIVE_BE>(
        &gf[..tail],
        &bf[..tail],
        &rf[..tail],
        &mut out[x * 4..],
        tail,
      );
    }
  }
}

// ---- Gbrpf16 → u16 RGB (F16C widen) ----------------------------------------

/// SSE4.1 + F16C: planar Gbrpf16 → packed `R, G, B` u16 (widen f16→f32). 4 px / iter.
///
/// # Safety
///
/// 1. SSE4.1 and F16C must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `3 * width`.
#[inline]
#[target_feature(enable = "sse4.1,f16c")]
#[allow(dead_code)] // dispatch wired in Task 8 (MixedSinker)
pub(crate) unsafe fn gbrpf16_to_rgb_u16_row_f16c<const BE: bool>(
  g: &[half::f16],
  b: &[half::f16],
  r: &[half::f16],
  out: &mut [u16],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(out.len() >= width * 3, "out row too short");

  unsafe {
    let zero = _mm_setzero_ps();
    let one = _mm_set1_ps(1.0);
    let scale = _mm_set1_ps(65535.0);

    let mut x = 0usize;
    while x + 4 <= width {
      let gv = _mm_cvtph_ps(endian::load_endian_u16x4::<BE>(
        g.as_ptr().add(x).cast::<u8>(),
      ));
      let bv = _mm_cvtph_ps(endian::load_endian_u16x4::<BE>(
        b.as_ptr().add(x).cast::<u8>(),
      ));
      let rv = _mm_cvtph_ps(endian::load_endian_u16x4::<BE>(
        r.as_ptr().add(x).cast::<u8>(),
      ));
      let gc = clamp01(gv, zero, one);
      let bc = clamp01(bv, zero, one);
      let rc = clamp01(rv, zero, one);
      let gu = i32x4_to_u16x4(scale_round_i32(gc, scale));
      let bu = i32x4_to_u16x4(scale_round_i32(bc, scale));
      let ru = i32x4_to_u16x4(scale_round_i32(rc, scale));
      let base = x * 3;
      for p in 0..4 {
        out[base + p * 3] = ru[p];
        out[base + p * 3 + 1] = gu[p];
        out[base + p * 3 + 2] = bu[p];
      }
      x += 4;
    }
    if x < width {
      // Scalar tail: bit-normalize f16 → host-native f32, then route the
      // scalar kernel via `HOST_NATIVE_BE` to avoid double-byte-swap.
      let tail = width - x;
      let mut gf = [0.0f32; 4];
      let mut bf = [0.0f32; 4];
      let mut rf = [0.0f32; 4];
      scalar_f16::widen_f16_be_to_host_f32::<BE>(g, x, &mut gf, tail);
      scalar_f16::widen_f16_be_to_host_f32::<BE>(b, x, &mut bf, tail);
      scalar_f16::widen_f16_be_to_host_f32::<BE>(r, x, &mut rf, tail);
      scalar::gbrpf32_to_rgb_u16_row::<HOST_NATIVE_BE>(
        &gf[..tail],
        &bf[..tail],
        &rf[..tail],
        &mut out[x * 3..],
        tail,
      );
    }
  }
}

// ---- Gbrpf16 → u16 RGBA (F16C widen) ----------------------------------------

/// SSE4.1 + F16C: planar Gbrpf16 → packed `R, G, B, A` u16 (widen, α = 0xFFFF).
/// 4 px / iter.
///
/// # Safety
///
/// 1. SSE4.1 and F16C must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "sse4.1,f16c")]
#[allow(dead_code)] // dispatch wired in Task 8 (MixedSinker)
pub(crate) unsafe fn gbrpf16_to_rgba_u16_row_f16c<const BE: bool>(
  g: &[half::f16],
  b: &[half::f16],
  r: &[half::f16],
  out: &mut [u16],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(out.len() >= width * 4, "out row too short");

  unsafe {
    let zero = _mm_setzero_ps();
    let one = _mm_set1_ps(1.0);
    let scale = _mm_set1_ps(65535.0);

    let mut x = 0usize;
    while x + 4 <= width {
      let gv = _mm_cvtph_ps(endian::load_endian_u16x4::<BE>(
        g.as_ptr().add(x).cast::<u8>(),
      ));
      let bv = _mm_cvtph_ps(endian::load_endian_u16x4::<BE>(
        b.as_ptr().add(x).cast::<u8>(),
      ));
      let rv = _mm_cvtph_ps(endian::load_endian_u16x4::<BE>(
        r.as_ptr().add(x).cast::<u8>(),
      ));
      let gc = clamp01(gv, zero, one);
      let bc = clamp01(bv, zero, one);
      let rc = clamp01(rv, zero, one);
      let gu = i32x4_to_u16x4(scale_round_i32(gc, scale));
      let bu = i32x4_to_u16x4(scale_round_i32(bc, scale));
      let ru = i32x4_to_u16x4(scale_round_i32(rc, scale));
      let base = x * 4;
      for p in 0..4 {
        out[base + p * 4] = ru[p];
        out[base + p * 4 + 1] = gu[p];
        out[base + p * 4 + 2] = bu[p];
        out[base + p * 4 + 3] = 0xFFFF;
      }
      x += 4;
    }
    if x < width {
      // Scalar tail: bit-normalize f16 → host-native f32, then route the
      // scalar kernel via `HOST_NATIVE_BE` to avoid double-byte-swap.
      let tail = width - x;
      let mut gf = [0.0f32; 4];
      let mut bf = [0.0f32; 4];
      let mut rf = [0.0f32; 4];
      scalar_f16::widen_f16_be_to_host_f32::<BE>(g, x, &mut gf, tail);
      scalar_f16::widen_f16_be_to_host_f32::<BE>(b, x, &mut bf, tail);
      scalar_f16::widen_f16_be_to_host_f32::<BE>(r, x, &mut rf, tail);
      scalar::gbrpf32_to_rgba_u16_row::<HOST_NATIVE_BE>(
        &gf[..tail],
        &bf[..tail],
        &rf[..tail],
        &mut out[x * 4..],
        tail,
      );
    }
  }
}

// ---- Gbrpf16 → f32 RGB (F16C widen, lossless) -------------------------------

/// SSE4.1 + F16C: planar Gbrpf16 → packed `R, G, B` f32 (lossless widen).
/// 4 px / iter.
///
/// # Safety
///
/// 1. SSE4.1 and F16C must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `3 * width`.
#[inline]
#[target_feature(enable = "sse4.1,f16c")]
#[allow(dead_code)] // dispatch wired in Task 8 (MixedSinker)
pub(crate) unsafe fn gbrpf16_to_rgb_f32_row_f16c<const BE: bool>(
  g: &[half::f16],
  b: &[half::f16],
  r: &[half::f16],
  out: &mut [f32],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(out.len() >= width * 3, "out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 4 <= width {
      let gv = _mm_cvtph_ps(endian::load_endian_u16x4::<BE>(
        g.as_ptr().add(x).cast::<u8>(),
      ));
      let bv = _mm_cvtph_ps(endian::load_endian_u16x4::<BE>(
        b.as_ptr().add(x).cast::<u8>(),
      ));
      let rv = _mm_cvtph_ps(endian::load_endian_u16x4::<BE>(
        r.as_ptr().add(x).cast::<u8>(),
      ));
      // No interleave intrinsic in SSE4.1 — scatter via scalar loop.
      let mut gf = [0.0f32; 4];
      let mut bf = [0.0f32; 4];
      let mut rf = [0.0f32; 4];
      _mm_storeu_ps(gf.as_mut_ptr(), gv);
      _mm_storeu_ps(bf.as_mut_ptr(), bv);
      _mm_storeu_ps(rf.as_mut_ptr(), rv);
      let base = x * 3;
      for p in 0..4 {
        out[base + p * 3] = rf[p];
        out[base + p * 3 + 1] = gf[p];
        out[base + p * 3 + 2] = bf[p];
      }
      x += 4;
    }
    if x < width {
      // Scalar tail: widen f16 → host-native f32 (normalize source bits via
      // `from_be` / `from_le` BEFORE the f16 → f32 conversion), then route the
      // scalar kernel via `HOST_NATIVE_BE` to avoid double-byte-swap.
      let tail = width - x;
      let mut gf = [0.0f32; 4];
      let mut bf = [0.0f32; 4];
      let mut rf = [0.0f32; 4];
      scalar_f16::widen_f16_be_to_host_f32::<BE>(g, x, &mut gf, tail);
      scalar_f16::widen_f16_be_to_host_f32::<BE>(b, x, &mut bf, tail);
      scalar_f16::widen_f16_be_to_host_f32::<BE>(r, x, &mut rf, tail);
      scalar::gbrpf32_to_rgb_f32_row::<HOST_NATIVE_BE>(
        &gf[..tail],
        &bf[..tail],
        &rf[..tail],
        &mut out[x * 3..],
        tail,
      );
    }
  }
}

// ---- Gbrpf16 → f32 RGBA (F16C widen, lossless, α = 1.0) --------------------

/// SSE4.1 + F16C: planar Gbrpf16 → packed `R, G, B, A` f32 (lossless, α = 1.0).
/// 4 px / iter.
///
/// # Safety
///
/// 1. SSE4.1 and F16C must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "sse4.1,f16c")]
#[allow(dead_code)] // dispatch wired in Task 8 (MixedSinker)
pub(crate) unsafe fn gbrpf16_to_rgba_f32_row_f16c<const BE: bool>(
  g: &[half::f16],
  b: &[half::f16],
  r: &[half::f16],
  out: &mut [f32],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(out.len() >= width * 4, "out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 4 <= width {
      let gv = _mm_cvtph_ps(endian::load_endian_u16x4::<BE>(
        g.as_ptr().add(x).cast::<u8>(),
      ));
      let bv = _mm_cvtph_ps(endian::load_endian_u16x4::<BE>(
        b.as_ptr().add(x).cast::<u8>(),
      ));
      let rv = _mm_cvtph_ps(endian::load_endian_u16x4::<BE>(
        r.as_ptr().add(x).cast::<u8>(),
      ));
      let mut gf = [0.0f32; 4];
      let mut bf = [0.0f32; 4];
      let mut rf = [0.0f32; 4];
      _mm_storeu_ps(gf.as_mut_ptr(), gv);
      _mm_storeu_ps(bf.as_mut_ptr(), bv);
      _mm_storeu_ps(rf.as_mut_ptr(), rv);
      let base = x * 4;
      for p in 0..4 {
        out[base + p * 4] = rf[p];
        out[base + p * 4 + 1] = gf[p];
        out[base + p * 4 + 2] = bf[p];
        out[base + p * 4 + 3] = 1.0;
      }
      x += 4;
    }
    if x < width {
      // Scalar tail: widen f16 → host-native f32 (normalize source bits via
      // `from_be` / `from_le` BEFORE the f16 → f32 conversion), then route the
      // scalar kernel via `HOST_NATIVE_BE` to avoid double-byte-swap.
      let tail = width - x;
      let mut gf = [0.0f32; 4];
      let mut bf = [0.0f32; 4];
      let mut rf = [0.0f32; 4];
      scalar_f16::widen_f16_be_to_host_f32::<BE>(g, x, &mut gf, tail);
      scalar_f16::widen_f16_be_to_host_f32::<BE>(b, x, &mut bf, tail);
      scalar_f16::widen_f16_be_to_host_f32::<BE>(r, x, &mut rf, tail);
      scalar::gbrpf32_to_rgba_f32_row::<HOST_NATIVE_BE>(
        &gf[..tail],
        &bf[..tail],
        &rf[..tail],
        &mut out[x * 4..],
        tail,
      );
    }
  }
}

// ---- Gbrpf16 → f16 RGB (lossless, opaque u16 interleave) -------------------

/// SSE4.1: planar Gbrpf16 → packed `R, G, B` f16 (lossless — f16 treated as u16).
///
/// No F16C gate needed: f16 planes are bit-copied as opaque u16 lanes.
/// 4 px / iter via SSE4.1 extract + manual scatter (no vst3 equivalent in SSE).
///
/// # Safety
///
/// 1. SSE4.1 must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `3 * width`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn gbrpf16_to_rgb_f16_row<const BE: bool>(
  g: &[half::f16],
  b: &[half::f16],
  r: &[half::f16],
  out: &mut [half::f16],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(out.len() >= width * 3, "out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 4 <= width {
      // Load 4 × u16 from each plane into the low 64 bits of __m128i.
      let gu = endian::load_endian_u16x4::<BE>(g.as_ptr().add(x).cast::<u8>());
      let bu = endian::load_endian_u16x4::<BE>(b.as_ptr().add(x).cast::<u8>());
      let ru = endian::load_endian_u16x4::<BE>(r.as_ptr().add(x).cast::<u8>());
      let base = x * 3;
      for p in 0..4usize {
        let dst = out.as_mut_ptr().add(base + p * 3);
        let g_word = match p {
          0 => _mm_extract_epi16::<0>(gu) as u16,
          1 => _mm_extract_epi16::<1>(gu) as u16,
          2 => _mm_extract_epi16::<2>(gu) as u16,
          _ => _mm_extract_epi16::<3>(gu) as u16,
        };
        let b_word = match p {
          0 => _mm_extract_epi16::<0>(bu) as u16,
          1 => _mm_extract_epi16::<1>(bu) as u16,
          2 => _mm_extract_epi16::<2>(bu) as u16,
          _ => _mm_extract_epi16::<3>(bu) as u16,
        };
        let r_word = match p {
          0 => _mm_extract_epi16::<0>(ru) as u16,
          1 => _mm_extract_epi16::<1>(ru) as u16,
          2 => _mm_extract_epi16::<2>(ru) as u16,
          _ => _mm_extract_epi16::<3>(ru) as u16,
        };
        *dst.cast::<u16>() = r_word;
        *dst.add(1).cast::<u16>() = g_word;
        *dst.add(2).cast::<u16>() = b_word;
      }
      x += 4;
    }
    if x < width {
      scalar_f16::gbrpf16_to_rgb_f16_row::<BE>(
        &g[x..],
        &b[x..],
        &r[x..],
        &mut out[x * 3..],
        width - x,
      );
    }
  }
}

// ---- Gbrpf16 → f16 RGBA (lossless, opaque u16, α = f16(1.0)) ---------------

/// SSE4.1: planar Gbrpf16 → packed `R, G, B, A` f16 (lossless, α = f16(1.0) = 0x3C00).
///
/// No F16C gate needed.
///
/// # Safety
///
/// 1. SSE4.1 must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn gbrpf16_to_rgba_f16_row<const BE: bool>(
  g: &[half::f16],
  b: &[half::f16],
  r: &[half::f16],
  out: &mut [half::f16],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(out.len() >= width * 4, "out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 4 <= width {
      let gu = endian::load_endian_u16x4::<BE>(g.as_ptr().add(x).cast::<u8>());
      let bu = endian::load_endian_u16x4::<BE>(b.as_ptr().add(x).cast::<u8>());
      let ru = endian::load_endian_u16x4::<BE>(r.as_ptr().add(x).cast::<u8>());
      let base = x * 4;
      for p in 0..4usize {
        let g_word = match p {
          0 => _mm_extract_epi16::<0>(gu) as u16,
          1 => _mm_extract_epi16::<1>(gu) as u16,
          2 => _mm_extract_epi16::<2>(gu) as u16,
          _ => _mm_extract_epi16::<3>(gu) as u16,
        };
        let b_word = match p {
          0 => _mm_extract_epi16::<0>(bu) as u16,
          1 => _mm_extract_epi16::<1>(bu) as u16,
          2 => _mm_extract_epi16::<2>(bu) as u16,
          _ => _mm_extract_epi16::<3>(bu) as u16,
        };
        let r_word = match p {
          0 => _mm_extract_epi16::<0>(ru) as u16,
          1 => _mm_extract_epi16::<1>(ru) as u16,
          2 => _mm_extract_epi16::<2>(ru) as u16,
          _ => _mm_extract_epi16::<3>(ru) as u16,
        };
        let dst = out.as_mut_ptr().add(base + p * 4);
        *dst.cast::<u16>() = r_word;
        *dst.add(1).cast::<u16>() = g_word;
        *dst.add(2).cast::<u16>() = b_word;
        *dst.add(3).cast::<u16>() = 0x3C00u16; // f16(1.0)
      }
      x += 4;
    }
    if x < width {
      scalar_f16::gbrpf16_to_rgba_f16_row::<BE>(
        &g[x..],
        &b[x..],
        &r[x..],
        &mut out[x * 4..],
        width - x,
      );
    }
  }
}

// ---- Gbrpf16 → u8 luma (F16C widen, staged via RGB scratch) ----------------

/// SSE4.1 + F16C: planar Gbrpf16 → u8 luma (widen + staged via RGB scratch).
///
/// # Safety
///
/// 1. SSE4.1 and F16C must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `width`.
#[inline]
#[target_feature(enable = "sse4.1,f16c")]
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)] // dispatch wired in Task 8 (MixedSinker)
pub(crate) unsafe fn gbrpf16_to_luma_row_f16c<const BE: bool>(
  g: &[half::f16],
  b: &[half::f16],
  r: &[half::f16],
  out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(out.len() >= width, "out row too short");

  const CHUNK: usize = 64;
  let mut scratch = [0u8; CHUNK * 3];
  let mut offset = 0;
  while offset < width {
    let n = (width - offset).min(CHUNK);
    unsafe {
      gbrpf16_to_rgb_row_f16c::<BE>(
        &g[offset..],
        &b[offset..],
        &r[offset..],
        &mut scratch[..n * 3],
        n,
      );
    }
    crate::row::scalar::rgb_to_luma_row(
      &scratch[..n * 3],
      &mut out[offset..offset + n],
      n,
      matrix,
      full_range,
    );
    offset += n;
  }
}

// ---- Gbrpf16 → u16 luma (F16C widen, staged via RGB scratch) ----------------

/// SSE4.1 + F16C: planar Gbrpf16 → u16 luma (widen + staged via RGB scratch).
///
/// # Safety
///
/// 1. SSE4.1 and F16C must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `width`.
#[inline]
#[target_feature(enable = "sse4.1,f16c")]
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)] // dispatch wired in Task 8 (MixedSinker)
pub(crate) unsafe fn gbrpf16_to_luma_u16_row_f16c<const BE: bool>(
  g: &[half::f16],
  b: &[half::f16],
  r: &[half::f16],
  out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(out.len() >= width, "out row too short");

  const CHUNK: usize = 64;
  let mut scratch = [0u8; CHUNK * 3];
  let mut offset = 0;
  while offset < width {
    let n = (width - offset).min(CHUNK);
    unsafe {
      gbrpf16_to_rgb_row_f16c::<BE>(
        &g[offset..],
        &b[offset..],
        &r[offset..],
        &mut scratch[..n * 3],
        n,
      );
    }
    crate::row::scalar::rgb_to_luma_u16_row(
      &scratch[..n * 3],
      &mut out[offset..offset + n],
      n,
      matrix,
      full_range,
    );
    offset += n;
  }
}

// ---- Gbrpf16 → HSV (F16C widen, staged via RGB scratch) ---------------------

/// SSE4.1 + F16C: planar Gbrpf16 → planar HSV bytes (widen + staged via RGB).
///
/// # Safety
///
/// 1. SSE4.1 and F16C must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `h_out.len()`, `s_out.len()`, `v_out.len()` ≥ `width`.
#[inline]
#[target_feature(enable = "sse4.1,f16c")]
#[allow(dead_code)] // dispatch wired in Task 8 (MixedSinker)
pub(crate) unsafe fn gbrpf16_to_hsv_row_f16c<const BE: bool>(
  g: &[half::f16],
  b: &[half::f16],
  r: &[half::f16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(h_out.len() >= width, "h_out row too short");
  debug_assert!(s_out.len() >= width, "s_out row too short");
  debug_assert!(v_out.len() >= width, "v_out row too short");

  const CHUNK: usize = 64;
  let mut scratch = [0u8; CHUNK * 3];
  let mut offset = 0;
  while offset < width {
    let n = (width - offset).min(CHUNK);
    unsafe {
      gbrpf16_to_rgb_row_f16c::<BE>(
        &g[offset..],
        &b[offset..],
        &r[offset..],
        &mut scratch[..n * 3],
        n,
      );
    }
    crate::row::scalar::rgb_to_hsv_row(
      &scratch[..n * 3],
      &mut h_out[offset..offset + n],
      &mut s_out[offset..offset + n],
      &mut v_out[offset..offset + n],
      n,
    );
    offset += n;
  }
}

// ---- Gbrapf16 → u8 RGBA (F16C widen, source α) ------------------------------

/// SSE4.1 + F16C: planar Gbrapf16 → packed `R, G, B, A` bytes (widen, source α).
/// 4 px / iter.
///
/// # Safety
///
/// 1. SSE4.1 and F16C must be available.
/// 2. `g.len()`, `b.len()`, `r.len()`, `a.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "sse4.1,f16c")]
#[allow(dead_code)] // dispatch wired in Task 8 (MixedSinker)
pub(crate) unsafe fn gbrapf16_to_rgba_row_f16c<const BE: bool>(
  g: &[half::f16],
  b: &[half::f16],
  r: &[half::f16],
  a: &[half::f16],
  out: &mut [u8],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(a.len() >= width, "a row too short");
  debug_assert!(out.len() >= width * 4, "out row too short");

  unsafe {
    let zero = _mm_setzero_ps();
    let one = _mm_set1_ps(1.0);
    let scale = _mm_set1_ps(255.0);

    let mut x = 0usize;
    while x + 4 <= width {
      let gv = _mm_cvtph_ps(endian::load_endian_u16x4::<BE>(
        g.as_ptr().add(x).cast::<u8>(),
      ));
      let bv = _mm_cvtph_ps(endian::load_endian_u16x4::<BE>(
        b.as_ptr().add(x).cast::<u8>(),
      ));
      let rv = _mm_cvtph_ps(endian::load_endian_u16x4::<BE>(
        r.as_ptr().add(x).cast::<u8>(),
      ));
      let av = _mm_cvtph_ps(endian::load_endian_u16x4::<BE>(
        a.as_ptr().add(x).cast::<u8>(),
      ));
      let gc = clamp01(gv, zero, one);
      let bc = clamp01(bv, zero, one);
      let rc = clamp01(rv, zero, one);
      let ac = clamp01(av, zero, one);
      let gi = i32x4_to_u8x4(scale_round_i32(gc, scale));
      let bi = i32x4_to_u8x4(scale_round_i32(bc, scale));
      let ri = i32x4_to_u8x4(scale_round_i32(rc, scale));
      let ai = i32x4_to_u8x4(scale_round_i32(ac, scale));
      let base = x * 4;
      for p in 0..4 {
        out[base + p * 4] = ri[p];
        out[base + p * 4 + 1] = gi[p];
        out[base + p * 4 + 2] = bi[p];
        out[base + p * 4 + 3] = ai[p];
      }
      x += 4;
    }
    if x < width {
      // Scalar tail: bit-normalize f16 → host-native f32, then route the
      // scalar kernel via `HOST_NATIVE_BE` to avoid double-byte-swap.
      let tail = width - x;
      let mut gf = [0.0f32; 4];
      let mut bf = [0.0f32; 4];
      let mut rf = [0.0f32; 4];
      let mut af = [0.0f32; 4];
      scalar_f16::widen_f16_be_to_host_f32::<BE>(g, x, &mut gf, tail);
      scalar_f16::widen_f16_be_to_host_f32::<BE>(b, x, &mut bf, tail);
      scalar_f16::widen_f16_be_to_host_f32::<BE>(r, x, &mut rf, tail);
      scalar_f16::widen_f16_be_to_host_f32::<BE>(a, x, &mut af, tail);
      scalar::gbrapf32_to_rgba_row::<HOST_NATIVE_BE>(
        &gf[..tail],
        &bf[..tail],
        &rf[..tail],
        &af[..tail],
        &mut out[x * 4..],
        tail,
      );
    }
  }
}

// ---- Gbrapf16 → u16 RGBA (F16C widen, source α) -----------------------------

/// SSE4.1 + F16C: planar Gbrapf16 → packed `R, G, B, A` u16 (widen, source α).
/// 4 px / iter.
///
/// # Safety
///
/// 1. SSE4.1 and F16C must be available.
/// 2. `g.len()`, `b.len()`, `r.len()`, `a.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "sse4.1,f16c")]
#[allow(dead_code)] // dispatch wired in Task 8 (MixedSinker)
pub(crate) unsafe fn gbrapf16_to_rgba_u16_row_f16c<const BE: bool>(
  g: &[half::f16],
  b: &[half::f16],
  r: &[half::f16],
  a: &[half::f16],
  out: &mut [u16],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(a.len() >= width, "a row too short");
  debug_assert!(out.len() >= width * 4, "out row too short");

  unsafe {
    let zero = _mm_setzero_ps();
    let one = _mm_set1_ps(1.0);
    let scale = _mm_set1_ps(65535.0);

    let mut x = 0usize;
    while x + 4 <= width {
      let gv = _mm_cvtph_ps(endian::load_endian_u16x4::<BE>(
        g.as_ptr().add(x).cast::<u8>(),
      ));
      let bv = _mm_cvtph_ps(endian::load_endian_u16x4::<BE>(
        b.as_ptr().add(x).cast::<u8>(),
      ));
      let rv = _mm_cvtph_ps(endian::load_endian_u16x4::<BE>(
        r.as_ptr().add(x).cast::<u8>(),
      ));
      let av = _mm_cvtph_ps(endian::load_endian_u16x4::<BE>(
        a.as_ptr().add(x).cast::<u8>(),
      ));
      let gc = clamp01(gv, zero, one);
      let bc = clamp01(bv, zero, one);
      let rc = clamp01(rv, zero, one);
      let ac = clamp01(av, zero, one);
      let gu = i32x4_to_u16x4(scale_round_i32(gc, scale));
      let bu = i32x4_to_u16x4(scale_round_i32(bc, scale));
      let ru = i32x4_to_u16x4(scale_round_i32(rc, scale));
      let au = i32x4_to_u16x4(scale_round_i32(ac, scale));
      let base = x * 4;
      for p in 0..4 {
        out[base + p * 4] = ru[p];
        out[base + p * 4 + 1] = gu[p];
        out[base + p * 4 + 2] = bu[p];
        out[base + p * 4 + 3] = au[p];
      }
      x += 4;
    }
    if x < width {
      // Scalar tail: bit-normalize f16 → host-native f32, then route the
      // scalar kernel via `HOST_NATIVE_BE` to avoid double-byte-swap.
      let tail = width - x;
      let mut gf = [0.0f32; 4];
      let mut bf = [0.0f32; 4];
      let mut rf = [0.0f32; 4];
      let mut af = [0.0f32; 4];
      scalar_f16::widen_f16_be_to_host_f32::<BE>(g, x, &mut gf, tail);
      scalar_f16::widen_f16_be_to_host_f32::<BE>(b, x, &mut bf, tail);
      scalar_f16::widen_f16_be_to_host_f32::<BE>(r, x, &mut rf, tail);
      scalar_f16::widen_f16_be_to_host_f32::<BE>(a, x, &mut af, tail);
      scalar::gbrapf32_to_rgba_u16_row::<HOST_NATIVE_BE>(
        &gf[..tail],
        &bf[..tail],
        &rf[..tail],
        &af[..tail],
        &mut out[x * 4..],
        tail,
      );
    }
  }
}

// ---- Gbrapf16 → f32 RGBA (F16C widen, lossless, source α) ------------------

/// SSE4.1 + F16C: planar Gbrapf16 → packed `R, G, B, A` f32 (lossless, source α).
/// 4 px / iter.
///
/// # Safety
///
/// 1. SSE4.1 and F16C must be available.
/// 2. `g.len()`, `b.len()`, `r.len()`, `a.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "sse4.1,f16c")]
#[allow(dead_code)] // dispatch wired in Task 8 (MixedSinker)
pub(crate) unsafe fn gbrapf16_to_rgba_f32_row_f16c<const BE: bool>(
  g: &[half::f16],
  b: &[half::f16],
  r: &[half::f16],
  a: &[half::f16],
  out: &mut [f32],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(a.len() >= width, "a row too short");
  debug_assert!(out.len() >= width * 4, "out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 4 <= width {
      let gv = _mm_cvtph_ps(endian::load_endian_u16x4::<BE>(
        g.as_ptr().add(x).cast::<u8>(),
      ));
      let bv = _mm_cvtph_ps(endian::load_endian_u16x4::<BE>(
        b.as_ptr().add(x).cast::<u8>(),
      ));
      let rv = _mm_cvtph_ps(endian::load_endian_u16x4::<BE>(
        r.as_ptr().add(x).cast::<u8>(),
      ));
      let av = _mm_cvtph_ps(endian::load_endian_u16x4::<BE>(
        a.as_ptr().add(x).cast::<u8>(),
      ));
      let mut gf = [0.0f32; 4];
      let mut bf = [0.0f32; 4];
      let mut rf = [0.0f32; 4];
      let mut af = [0.0f32; 4];
      _mm_storeu_ps(gf.as_mut_ptr(), gv);
      _mm_storeu_ps(bf.as_mut_ptr(), bv);
      _mm_storeu_ps(rf.as_mut_ptr(), rv);
      _mm_storeu_ps(af.as_mut_ptr(), av);
      let base = x * 4;
      for p in 0..4 {
        out[base + p * 4] = rf[p];
        out[base + p * 4 + 1] = gf[p];
        out[base + p * 4 + 2] = bf[p];
        out[base + p * 4 + 3] = af[p];
      }
      x += 4;
    }
    if x < width {
      // Scalar tail: widen f16 → host-native f32 (normalize source bits via
      // `from_be` / `from_le` BEFORE the f16 → f32 conversion), then route the
      // scalar kernel via `HOST_NATIVE_BE` to avoid double-byte-swap.
      let tail = width - x;
      let mut gf = [0.0f32; 4];
      let mut bf = [0.0f32; 4];
      let mut rf = [0.0f32; 4];
      let mut af = [0.0f32; 4];
      scalar_f16::widen_f16_be_to_host_f32::<BE>(g, x, &mut gf, tail);
      scalar_f16::widen_f16_be_to_host_f32::<BE>(b, x, &mut bf, tail);
      scalar_f16::widen_f16_be_to_host_f32::<BE>(r, x, &mut rf, tail);
      scalar_f16::widen_f16_be_to_host_f32::<BE>(a, x, &mut af, tail);
      scalar::gbrapf32_to_rgba_f32_row::<HOST_NATIVE_BE>(
        &gf[..tail],
        &bf[..tail],
        &rf[..tail],
        &af[..tail],
        &mut out[x * 4..],
        tail,
      );
    }
  }
}

// ---- Gbrapf16 → f16 RGBA (lossless, opaque u16, source α) ------------------

/// SSE4.1: planar Gbrapf16 → packed `R, G, B, A` f16 (lossless, source α).
///
/// No F16C gate needed: f16 planes are bit-copied as opaque u16.
///
/// # Safety
///
/// 1. SSE4.1 must be available.
/// 2. `g.len()`, `b.len()`, `r.len()`, `a.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn gbrapf16_to_rgba_f16_row<const BE: bool>(
  g: &[half::f16],
  b: &[half::f16],
  r: &[half::f16],
  a: &[half::f16],
  out: &mut [half::f16],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(a.len() >= width, "a row too short");
  debug_assert!(out.len() >= width * 4, "out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 4 <= width {
      let gu = endian::load_endian_u16x4::<BE>(g.as_ptr().add(x).cast::<u8>());
      let bu = endian::load_endian_u16x4::<BE>(b.as_ptr().add(x).cast::<u8>());
      let ru = endian::load_endian_u16x4::<BE>(r.as_ptr().add(x).cast::<u8>());
      let au = endian::load_endian_u16x4::<BE>(a.as_ptr().add(x).cast::<u8>());
      let base = x * 4;
      for p in 0..4usize {
        let g_word = match p {
          0 => _mm_extract_epi16::<0>(gu) as u16,
          1 => _mm_extract_epi16::<1>(gu) as u16,
          2 => _mm_extract_epi16::<2>(gu) as u16,
          _ => _mm_extract_epi16::<3>(gu) as u16,
        };
        let b_word = match p {
          0 => _mm_extract_epi16::<0>(bu) as u16,
          1 => _mm_extract_epi16::<1>(bu) as u16,
          2 => _mm_extract_epi16::<2>(bu) as u16,
          _ => _mm_extract_epi16::<3>(bu) as u16,
        };
        let r_word = match p {
          0 => _mm_extract_epi16::<0>(ru) as u16,
          1 => _mm_extract_epi16::<1>(ru) as u16,
          2 => _mm_extract_epi16::<2>(ru) as u16,
          _ => _mm_extract_epi16::<3>(ru) as u16,
        };
        let a_word = match p {
          0 => _mm_extract_epi16::<0>(au) as u16,
          1 => _mm_extract_epi16::<1>(au) as u16,
          2 => _mm_extract_epi16::<2>(au) as u16,
          _ => _mm_extract_epi16::<3>(au) as u16,
        };
        let dst = out.as_mut_ptr().add(base + p * 4);
        *dst.cast::<u16>() = r_word;
        *dst.add(1).cast::<u16>() = g_word;
        *dst.add(2).cast::<u16>() = b_word;
        *dst.add(3).cast::<u16>() = a_word;
      }
      x += 4;
    }
    if x < width {
      scalar_f16::gbrapf16_to_rgba_f16_row::<BE>(
        &g[x..],
        &b[x..],
        &r[x..],
        &a[x..],
        &mut out[x * 4..],
        width - x,
      );
    }
  }
}
