//! AVX-512 (F + BW) SIMD kernels for planar GBR float sources (Tier 10).
//!
//! 16 pixels / iteration via `__m512` (one G, B, R [+ A] register each).
//! f16 narrowing / widening via `_mm512_cvtph_ps` / `_mm512_cvtps_ph` —
//! gated at runtime through `is_x86_feature_detected!("f16c")` because
//! the underlying VCVTPH2PS/VCVTPS2PH encoding bit is reported via the
//! F16C CPU-feature bit even on AVX-512 cores.
//!
//! # Rounding (f32 → u8 / u16)
//!
//! `_mm512_add_ps(scaled, _mm512_set1_ps(0.5))` then `_mm512_cvttps_epi32`
//! (truncate toward zero). This is the round-half-up contract shared with
//! the scalar / NEON / SSE4.1 / AVX2 kernels — MXCSR-independent and
//! consistent with the Grayf32 path.
//!
//! **Do NOT use `_mm512_cvt_roundps_epi32` with `_MM_FROUND_TO_NEAREST_INT`
//! for integer narrowing** — that gives banker's rounding, not
//! round-half-up.
//!
//! # Rounding (f32 → f16)
//!
//! `_mm512_cvtps_ph::<{ _MM_FROUND_TO_NEAREST_INT }>`: IEEE-754
//! round-to-nearest-even, matching `half::f16::from_f32` and the AVX2 /
//! SSE4.1 backends. `_mm512_cvtps_ph` accepts the extended-rounding
//! constant set (the AVX-512 `static_assert_extended_rounding!` macro
//! permits 0..=4, 8..=12) but we use bare `_MM_FROUND_TO_NEAREST_INT`
//! (= 0) for consistency with the Task 5 fix and the AVX2 backend; the
//! `_MM_FROUND_NO_EXC` bit is a no-op for normalized RGB inputs in
//! `[0, 1]` anyway.
//!
//! # Lane discipline
//!
//! AVX-512F native saturating narrows `_mm512_cvtusepi32_epi8` (i32x16
//! → u8x16, returned as `__m128i`) and `_mm512_cvtusepi32_epi16` (i32x16
//! → u16x16, returned as `__m256i`) preserve natural element order — no
//! pack-fixup permute needed (unlike `_mm512_packus_epi16` which is
//! per-128-bit-lane). This matches the AVX-512 Grayf32 / Rgbf32 pattern.
//!
//! `_mm512_cvtph_ps` widens 16 × f16 (`__m256i`) to 16 × f32 (`__m512`)
//! in natural order; `_mm512_cvtps_ph` narrows in natural order. No
//! lane-cross fixup needed for either f16 path.
//!
//! # f16 lossless interleave
//!
//! Treat f16 lanes as opaque `u16` — no arithmetic, no F16C gate needed.
//! The 16-pixel loop loads 32 bytes (16 × u16) per plane via
//! `_mm256_loadu_si256` and scatters as triples / quads in the destination.

#[cfg_attr(miri, allow(unused_imports))]
use core::arch::x86_64::*;

use crate::{
  KernelMatrix,
  row::{
    arch::x86_avx512::endian,
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

/// Clamp a `__m512` (f32x16) to `[0.0, 1.0]`.
#[inline(always)]
unsafe fn clamp01(v: __m512, zero: __m512, one: __m512) -> __m512 {
  unsafe { _mm512_min_ps(_mm512_max_ps(v, zero), one) }
}

/// Scale, add 0.5, truncate → `__m512i` (i32x16). Round-half-up.
#[inline(always)]
unsafe fn scale_round_i32(v: __m512, scale: __m512) -> __m512i {
  unsafe { _mm512_cvttps_epi32(_mm512_add_ps(_mm512_mul_ps(v, scale), _mm512_set1_ps(0.5))) }
}

// ---- Gbrpf32 → u8 RGB -------------------------------------------------------

/// AVX-512: planar Gbrpf32 → packed `R, G, B` bytes. 16 px / iter.
///
/// Round-half-up: `+ 0.5` then `_mm512_cvttps_epi32`.
///
/// # Safety
///
/// 1. AVX-512F + AVX-512BW must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `3 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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
    let zero = _mm512_setzero_ps();
    let one = _mm512_set1_ps(1.0);
    let scale = _mm512_set1_ps(255.0);

    let mut x = 0usize;
    while x + 16 <= width {
      let gv = clamp01(
        _mm512_castsi512_ps(endian::load_endian_u32x16::<BE>(
          g.as_ptr().add(x).cast::<u8>(),
        )),
        zero,
        one,
      );
      let bv = clamp01(
        _mm512_castsi512_ps(endian::load_endian_u32x16::<BE>(
          b.as_ptr().add(x).cast::<u8>(),
        )),
        zero,
        one,
      );
      let rv = clamp01(
        _mm512_castsi512_ps(endian::load_endian_u32x16::<BE>(
          r.as_ptr().add(x).cast::<u8>(),
        )),
        zero,
        one,
      );
      let g8 = _mm512_cvtusepi32_epi8(scale_round_i32(gv, scale));
      let b8 = _mm512_cvtusepi32_epi8(scale_round_i32(bv, scale));
      let r8 = _mm512_cvtusepi32_epi8(scale_round_i32(rv, scale));
      let mut g_buf = [0u8; 16];
      let mut b_buf = [0u8; 16];
      let mut r_buf = [0u8; 16];
      _mm_storeu_si128(g_buf.as_mut_ptr().cast(), g8);
      _mm_storeu_si128(b_buf.as_mut_ptr().cast(), b8);
      _mm_storeu_si128(r_buf.as_mut_ptr().cast(), r8);
      let base = x * 3;
      for p in 0..16 {
        out[base + p * 3] = r_buf[p];
        out[base + p * 3 + 1] = g_buf[p];
        out[base + p * 3 + 2] = b_buf[p];
      }
      x += 16;
    }
    if x < width {
      scalar::gbrpf32_to_rgb_row::<BE>(&g[x..], &b[x..], &r[x..], &mut out[x * 3..], width - x);
    }
  }
}

// ---- Gbrpf32 → u8 RGBA (opaque α) ------------------------------------------

/// AVX-512: planar Gbrpf32 → packed `R, G, B, A` bytes (α = 0xFF). 16 px / iter.
///
/// # Safety
///
/// 1. AVX-512F + AVX-512BW must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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
    let zero = _mm512_setzero_ps();
    let one = _mm512_set1_ps(1.0);
    let scale = _mm512_set1_ps(255.0);

    let mut x = 0usize;
    while x + 16 <= width {
      let gv = clamp01(
        _mm512_castsi512_ps(endian::load_endian_u32x16::<BE>(
          g.as_ptr().add(x).cast::<u8>(),
        )),
        zero,
        one,
      );
      let bv = clamp01(
        _mm512_castsi512_ps(endian::load_endian_u32x16::<BE>(
          b.as_ptr().add(x).cast::<u8>(),
        )),
        zero,
        one,
      );
      let rv = clamp01(
        _mm512_castsi512_ps(endian::load_endian_u32x16::<BE>(
          r.as_ptr().add(x).cast::<u8>(),
        )),
        zero,
        one,
      );
      let g8 = _mm512_cvtusepi32_epi8(scale_round_i32(gv, scale));
      let b8 = _mm512_cvtusepi32_epi8(scale_round_i32(bv, scale));
      let r8 = _mm512_cvtusepi32_epi8(scale_round_i32(rv, scale));
      let mut g_buf = [0u8; 16];
      let mut b_buf = [0u8; 16];
      let mut r_buf = [0u8; 16];
      _mm_storeu_si128(g_buf.as_mut_ptr().cast(), g8);
      _mm_storeu_si128(b_buf.as_mut_ptr().cast(), b8);
      _mm_storeu_si128(r_buf.as_mut_ptr().cast(), r8);
      let base = x * 4;
      for p in 0..16 {
        out[base + p * 4] = r_buf[p];
        out[base + p * 4 + 1] = g_buf[p];
        out[base + p * 4 + 2] = b_buf[p];
        out[base + p * 4 + 3] = 0xFF;
      }
      x += 16;
    }
    if x < width {
      scalar::gbrpf32_to_rgba_row::<BE>(&g[x..], &b[x..], &r[x..], &mut out[x * 4..], width - x);
    }
  }
}

// ---- Gbrpf32 → u16 RGB ------------------------------------------------------

/// AVX-512: planar Gbrpf32 → packed `R, G, B` u16. 16 px / iter.
///
/// # Safety
///
/// 1. AVX-512F + AVX-512BW must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `3 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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
    let zero = _mm512_setzero_ps();
    let one = _mm512_set1_ps(1.0);
    let scale = _mm512_set1_ps(65535.0);

    let mut x = 0usize;
    while x + 16 <= width {
      let gv = clamp01(
        _mm512_castsi512_ps(endian::load_endian_u32x16::<BE>(
          g.as_ptr().add(x).cast::<u8>(),
        )),
        zero,
        one,
      );
      let bv = clamp01(
        _mm512_castsi512_ps(endian::load_endian_u32x16::<BE>(
          b.as_ptr().add(x).cast::<u8>(),
        )),
        zero,
        one,
      );
      let rv = clamp01(
        _mm512_castsi512_ps(endian::load_endian_u32x16::<BE>(
          r.as_ptr().add(x).cast::<u8>(),
        )),
        zero,
        one,
      );
      let gw = _mm512_cvtusepi32_epi16(scale_round_i32(gv, scale));
      let bw = _mm512_cvtusepi32_epi16(scale_round_i32(bv, scale));
      let rw = _mm512_cvtusepi32_epi16(scale_round_i32(rv, scale));
      let mut g_buf = [0u16; 16];
      let mut b_buf = [0u16; 16];
      let mut r_buf = [0u16; 16];
      _mm256_storeu_si256(g_buf.as_mut_ptr().cast(), gw);
      _mm256_storeu_si256(b_buf.as_mut_ptr().cast(), bw);
      _mm256_storeu_si256(r_buf.as_mut_ptr().cast(), rw);
      let base = x * 3;
      for p in 0..16 {
        out[base + p * 3] = r_buf[p];
        out[base + p * 3 + 1] = g_buf[p];
        out[base + p * 3 + 2] = b_buf[p];
      }
      x += 16;
    }
    if x < width {
      scalar::gbrpf32_to_rgb_u16_row::<BE>(&g[x..], &b[x..], &r[x..], &mut out[x * 3..], width - x);
    }
  }
}

// ---- Gbrpf32 → u16 RGBA (opaque α) -----------------------------------------

/// AVX-512: planar Gbrpf32 → packed `R, G, B, A` u16 (α = 0xFFFF). 16 px / iter.
///
/// # Safety
///
/// 1. AVX-512F + AVX-512BW must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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
    let zero = _mm512_setzero_ps();
    let one = _mm512_set1_ps(1.0);
    let scale = _mm512_set1_ps(65535.0);

    let mut x = 0usize;
    while x + 16 <= width {
      let gv = clamp01(
        _mm512_castsi512_ps(endian::load_endian_u32x16::<BE>(
          g.as_ptr().add(x).cast::<u8>(),
        )),
        zero,
        one,
      );
      let bv = clamp01(
        _mm512_castsi512_ps(endian::load_endian_u32x16::<BE>(
          b.as_ptr().add(x).cast::<u8>(),
        )),
        zero,
        one,
      );
      let rv = clamp01(
        _mm512_castsi512_ps(endian::load_endian_u32x16::<BE>(
          r.as_ptr().add(x).cast::<u8>(),
        )),
        zero,
        one,
      );
      let gw = _mm512_cvtusepi32_epi16(scale_round_i32(gv, scale));
      let bw = _mm512_cvtusepi32_epi16(scale_round_i32(bv, scale));
      let rw = _mm512_cvtusepi32_epi16(scale_round_i32(rv, scale));
      let mut g_buf = [0u16; 16];
      let mut b_buf = [0u16; 16];
      let mut r_buf = [0u16; 16];
      _mm256_storeu_si256(g_buf.as_mut_ptr().cast(), gw);
      _mm256_storeu_si256(b_buf.as_mut_ptr().cast(), bw);
      _mm256_storeu_si256(r_buf.as_mut_ptr().cast(), rw);
      let base = x * 4;
      for p in 0..16 {
        out[base + p * 4] = r_buf[p];
        out[base + p * 4 + 1] = g_buf[p];
        out[base + p * 4 + 2] = b_buf[p];
        out[base + p * 4 + 3] = 0xFFFF;
      }
      x += 16;
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

/// AVX-512: planar Gbrpf32 → packed `R, G, B` f32 (lossless interleave).
///
/// AVX-512 has no 3-channel interleave store; use scalar (compiler vectorises
/// the simple per-element copy well).
///
/// # Safety
///
/// 1. AVX-512F + AVX-512BW must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `3 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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

  scalar::gbrpf32_to_rgb_f32_row::<BE>(g, b, r, out, width);
}

// ---- Gbrpf32 → f32 RGBA (lossless, α = 1.0) ---------------------------------

/// AVX-512: planar Gbrpf32 → packed `R, G, B, A` f32 (lossless, α = 1.0).
///
/// AVX-512 has no 4-channel interleave store; use scalar.
///
/// # Safety
///
/// 1. AVX-512F + AVX-512BW must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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

  scalar::gbrpf32_to_rgba_f32_row::<BE>(g, b, r, out, width);
}

// ---- Gbrpf32 → f16 RGB (F16C narrow) ----------------------------------------

/// AVX-512 + F16C: planar Gbrpf32 → packed `R, G, B` f16 (fused narrow).
/// 16 px / iter.
///
/// Uses `_mm512_cvtps_ph` with `_MM_FROUND_TO_NEAREST_INT`
/// (IEEE-754 round-to-nearest-even).
///
/// # Safety
///
/// 1. AVX-512F + AVX-512BW + F16C must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `3 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,f16c")]
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
    while x + 16 <= width {
      let gv = _mm512_castsi512_ps(endian::load_endian_u32x16::<BE>(
        g.as_ptr().add(x).cast::<u8>(),
      ));
      let bv = _mm512_castsi512_ps(endian::load_endian_u32x16::<BE>(
        b.as_ptr().add(x).cast::<u8>(),
      ));
      let rv = _mm512_castsi512_ps(endian::load_endian_u32x16::<BE>(
        r.as_ptr().add(x).cast::<u8>(),
      ));
      // F16C narrow: IEEE-754 round-to-nearest-even (NOT round-half-up).
      let gh = _mm512_cvtps_ph::<{ _MM_FROUND_TO_NEAREST_INT }>(gv);
      let bh = _mm512_cvtps_ph::<{ _MM_FROUND_TO_NEAREST_INT }>(bv);
      let rh = _mm512_cvtps_ph::<{ _MM_FROUND_TO_NEAREST_INT }>(rv);
      // Each `__m256i` holds 16 f16 lanes in natural order.
      let mut g_buf = [0u16; 16];
      let mut b_buf = [0u16; 16];
      let mut r_buf = [0u16; 16];
      _mm256_storeu_si256(g_buf.as_mut_ptr().cast(), gh);
      _mm256_storeu_si256(b_buf.as_mut_ptr().cast(), bh);
      _mm256_storeu_si256(r_buf.as_mut_ptr().cast(), rh);
      let base = x * 3;
      for p in 0..16 {
        let dst = out.as_mut_ptr().add(base + p * 3);
        *dst.cast::<u16>() = r_buf[p];
        *dst.add(1).cast::<u16>() = g_buf[p];
        *dst.add(2).cast::<u16>() = b_buf[p];
      }
      x += 16;
    }
    if x < width {
      scalar::gbrpf32_to_rgb_f16_row::<BE>(&g[x..], &b[x..], &r[x..], &mut out[x * 3..], width - x);
    }
  }
}

// ---- Gbrpf32 → f16 RGBA (F16C narrow, α = f16(1.0)) -------------------------

/// AVX-512 + F16C: planar Gbrpf32 → packed `R, G, B, A` f16 (fused narrow,
/// α = f16(1.0) = 0x3C00). 16 px / iter.
///
/// # Safety
///
/// 1. AVX-512F + AVX-512BW + F16C must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,f16c")]
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
    while x + 16 <= width {
      let gv = _mm512_castsi512_ps(endian::load_endian_u32x16::<BE>(
        g.as_ptr().add(x).cast::<u8>(),
      ));
      let bv = _mm512_castsi512_ps(endian::load_endian_u32x16::<BE>(
        b.as_ptr().add(x).cast::<u8>(),
      ));
      let rv = _mm512_castsi512_ps(endian::load_endian_u32x16::<BE>(
        r.as_ptr().add(x).cast::<u8>(),
      ));
      let gh = _mm512_cvtps_ph::<{ _MM_FROUND_TO_NEAREST_INT }>(gv);
      let bh = _mm512_cvtps_ph::<{ _MM_FROUND_TO_NEAREST_INT }>(bv);
      let rh = _mm512_cvtps_ph::<{ _MM_FROUND_TO_NEAREST_INT }>(rv);
      let mut g_buf = [0u16; 16];
      let mut b_buf = [0u16; 16];
      let mut r_buf = [0u16; 16];
      _mm256_storeu_si256(g_buf.as_mut_ptr().cast(), gh);
      _mm256_storeu_si256(b_buf.as_mut_ptr().cast(), bh);
      _mm256_storeu_si256(r_buf.as_mut_ptr().cast(), rh);
      let base = x * 4;
      for p in 0..16 {
        let dst = out.as_mut_ptr().add(base + p * 4);
        *dst.cast::<u16>() = r_buf[p];
        *dst.add(1).cast::<u16>() = g_buf[p];
        *dst.add(2).cast::<u16>() = b_buf[p];
        *dst.add(3).cast::<u16>() = 0x3C00u16; // f16(1.0)
      }
      x += 16;
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

/// AVX-512: planar Gbrpf32 → u8 luma (staged via AVX-512 RGB kernel).
///
/// # Safety
///
/// 1. AVX-512F + AVX-512BW must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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

/// AVX-512: planar Gbrpf32 → u16 luma (staged via AVX-512 RGB kernel).
///
/// # Safety
///
/// 1. AVX-512F + AVX-512BW must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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

/// AVX-512: planar Gbrpf32 → planar HSV bytes (staged via AVX-512 RGB kernel).
///
/// # Safety
///
/// 1. AVX-512F + AVX-512BW must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `h_out.len()`, `s_out.len()`, `v_out.len()` ≥ `width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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

/// AVX-512: planar Gbrapf32 → packed `R, G, B, A` bytes (source α).
/// 16 px / iter.
///
/// # Safety
///
/// 1. AVX-512F + AVX-512BW must be available.
/// 2. `g.len()`, `b.len()`, `r.len()`, `a.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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
    let zero = _mm512_setzero_ps();
    let one = _mm512_set1_ps(1.0);
    let scale = _mm512_set1_ps(255.0);

    let mut x = 0usize;
    while x + 16 <= width {
      let gv = clamp01(
        _mm512_castsi512_ps(endian::load_endian_u32x16::<BE>(
          g.as_ptr().add(x).cast::<u8>(),
        )),
        zero,
        one,
      );
      let bv = clamp01(
        _mm512_castsi512_ps(endian::load_endian_u32x16::<BE>(
          b.as_ptr().add(x).cast::<u8>(),
        )),
        zero,
        one,
      );
      let rv = clamp01(
        _mm512_castsi512_ps(endian::load_endian_u32x16::<BE>(
          r.as_ptr().add(x).cast::<u8>(),
        )),
        zero,
        one,
      );
      let av = clamp01(
        _mm512_castsi512_ps(endian::load_endian_u32x16::<BE>(
          a.as_ptr().add(x).cast::<u8>(),
        )),
        zero,
        one,
      );
      let g8 = _mm512_cvtusepi32_epi8(scale_round_i32(gv, scale));
      let b8 = _mm512_cvtusepi32_epi8(scale_round_i32(bv, scale));
      let r8 = _mm512_cvtusepi32_epi8(scale_round_i32(rv, scale));
      let a8 = _mm512_cvtusepi32_epi8(scale_round_i32(av, scale));
      let mut g_buf = [0u8; 16];
      let mut b_buf = [0u8; 16];
      let mut r_buf = [0u8; 16];
      let mut a_buf = [0u8; 16];
      _mm_storeu_si128(g_buf.as_mut_ptr().cast(), g8);
      _mm_storeu_si128(b_buf.as_mut_ptr().cast(), b8);
      _mm_storeu_si128(r_buf.as_mut_ptr().cast(), r8);
      _mm_storeu_si128(a_buf.as_mut_ptr().cast(), a8);
      let base = x * 4;
      for p in 0..16 {
        out[base + p * 4] = r_buf[p];
        out[base + p * 4 + 1] = g_buf[p];
        out[base + p * 4 + 2] = b_buf[p];
        out[base + p * 4 + 3] = a_buf[p];
      }
      x += 16;
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

/// AVX-512: planar Gbrapf32 → packed `R, G, B, A` u16 (source α).
/// 16 px / iter.
///
/// # Safety
///
/// 1. AVX-512F + AVX-512BW must be available.
/// 2. `g.len()`, `b.len()`, `r.len()`, `a.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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
    let zero = _mm512_setzero_ps();
    let one = _mm512_set1_ps(1.0);
    let scale = _mm512_set1_ps(65535.0);

    let mut x = 0usize;
    while x + 16 <= width {
      let gv = clamp01(
        _mm512_castsi512_ps(endian::load_endian_u32x16::<BE>(
          g.as_ptr().add(x).cast::<u8>(),
        )),
        zero,
        one,
      );
      let bv = clamp01(
        _mm512_castsi512_ps(endian::load_endian_u32x16::<BE>(
          b.as_ptr().add(x).cast::<u8>(),
        )),
        zero,
        one,
      );
      let rv = clamp01(
        _mm512_castsi512_ps(endian::load_endian_u32x16::<BE>(
          r.as_ptr().add(x).cast::<u8>(),
        )),
        zero,
        one,
      );
      let av = clamp01(
        _mm512_castsi512_ps(endian::load_endian_u32x16::<BE>(
          a.as_ptr().add(x).cast::<u8>(),
        )),
        zero,
        one,
      );
      let gw = _mm512_cvtusepi32_epi16(scale_round_i32(gv, scale));
      let bw = _mm512_cvtusepi32_epi16(scale_round_i32(bv, scale));
      let rw = _mm512_cvtusepi32_epi16(scale_round_i32(rv, scale));
      let aw = _mm512_cvtusepi32_epi16(scale_round_i32(av, scale));
      let mut g_buf = [0u16; 16];
      let mut b_buf = [0u16; 16];
      let mut r_buf = [0u16; 16];
      let mut a_buf = [0u16; 16];
      _mm256_storeu_si256(g_buf.as_mut_ptr().cast(), gw);
      _mm256_storeu_si256(b_buf.as_mut_ptr().cast(), bw);
      _mm256_storeu_si256(r_buf.as_mut_ptr().cast(), rw);
      _mm256_storeu_si256(a_buf.as_mut_ptr().cast(), aw);
      let base = x * 4;
      for p in 0..16 {
        out[base + p * 4] = r_buf[p];
        out[base + p * 4 + 1] = g_buf[p];
        out[base + p * 4 + 2] = b_buf[p];
        out[base + p * 4 + 3] = a_buf[p];
      }
      x += 16;
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

/// AVX-512: planar Gbrapf32 → packed `R, G, B, A` f32 (lossless, source α).
///
/// AVX-512 has no 4-channel interleave store; use scalar.
///
/// # Safety
///
/// 1. AVX-512F + AVX-512BW must be available.
/// 2. `g.len()`, `b.len()`, `r.len()`, `a.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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

  scalar::gbrapf32_to_rgba_f32_row::<BE>(g, b, r, a, out, width);
}

// ---- Gbrapf32 → f16 RGBA (F16C narrow, source α) ----------------------------

/// AVX-512 + F16C: planar Gbrapf32 → packed `R, G, B, A` f16 (fused narrow,
/// source α). 16 px / iter.
///
/// # Safety
///
/// 1. AVX-512F + AVX-512BW + F16C must be available.
/// 2. `g.len()`, `b.len()`, `r.len()`, `a.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,f16c")]
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
    while x + 16 <= width {
      let gv = _mm512_castsi512_ps(endian::load_endian_u32x16::<BE>(
        g.as_ptr().add(x).cast::<u8>(),
      ));
      let bv = _mm512_castsi512_ps(endian::load_endian_u32x16::<BE>(
        b.as_ptr().add(x).cast::<u8>(),
      ));
      let rv = _mm512_castsi512_ps(endian::load_endian_u32x16::<BE>(
        r.as_ptr().add(x).cast::<u8>(),
      ));
      let av = _mm512_castsi512_ps(endian::load_endian_u32x16::<BE>(
        a.as_ptr().add(x).cast::<u8>(),
      ));
      let gh = _mm512_cvtps_ph::<{ _MM_FROUND_TO_NEAREST_INT }>(gv);
      let bh = _mm512_cvtps_ph::<{ _MM_FROUND_TO_NEAREST_INT }>(bv);
      let rh = _mm512_cvtps_ph::<{ _MM_FROUND_TO_NEAREST_INT }>(rv);
      let ah = _mm512_cvtps_ph::<{ _MM_FROUND_TO_NEAREST_INT }>(av);
      let mut g_buf = [0u16; 16];
      let mut b_buf = [0u16; 16];
      let mut r_buf = [0u16; 16];
      let mut a_buf = [0u16; 16];
      _mm256_storeu_si256(g_buf.as_mut_ptr().cast(), gh);
      _mm256_storeu_si256(b_buf.as_mut_ptr().cast(), bh);
      _mm256_storeu_si256(r_buf.as_mut_ptr().cast(), rh);
      _mm256_storeu_si256(a_buf.as_mut_ptr().cast(), ah);
      let base = x * 4;
      for p in 0..16 {
        let dst = out.as_mut_ptr().add(base + p * 4);
        *dst.cast::<u16>() = r_buf[p];
        *dst.add(1).cast::<u16>() = g_buf[p];
        *dst.add(2).cast::<u16>() = b_buf[p];
        *dst.add(3).cast::<u16>() = a_buf[p];
      }
      x += 16;
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

/// AVX-512 + F16C: planar Gbrpf16 → packed `R, G, B` bytes (widen f16→f32,
/// then convert). 16 px / iter.
///
/// # Safety
///
/// 1. AVX-512F + AVX-512BW + F16C must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `3 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,f16c")]
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
    let zero = _mm512_setzero_ps();
    let one = _mm512_set1_ps(1.0);
    let scale = _mm512_set1_ps(255.0);

    let mut x = 0usize;
    while x + 16 <= width {
      // Load 16 f16 lanes (32 bytes) per plane and widen to f32x16.
      let gv = _mm512_cvtph_ps(endian::load_endian_u16x16::<BE>(
        g.as_ptr().add(x).cast::<u8>(),
      ));
      let bv = _mm512_cvtph_ps(endian::load_endian_u16x16::<BE>(
        b.as_ptr().add(x).cast::<u8>(),
      ));
      let rv = _mm512_cvtph_ps(endian::load_endian_u16x16::<BE>(
        r.as_ptr().add(x).cast::<u8>(),
      ));
      let gc = clamp01(gv, zero, one);
      let bc = clamp01(bv, zero, one);
      let rc = clamp01(rv, zero, one);
      let g8 = _mm512_cvtusepi32_epi8(scale_round_i32(gc, scale));
      let b8 = _mm512_cvtusepi32_epi8(scale_round_i32(bc, scale));
      let r8 = _mm512_cvtusepi32_epi8(scale_round_i32(rc, scale));
      let mut g_buf = [0u8; 16];
      let mut b_buf = [0u8; 16];
      let mut r_buf = [0u8; 16];
      _mm_storeu_si128(g_buf.as_mut_ptr().cast(), g8);
      _mm_storeu_si128(b_buf.as_mut_ptr().cast(), b8);
      _mm_storeu_si128(r_buf.as_mut_ptr().cast(), r8);
      let base = x * 3;
      for p in 0..16 {
        out[base + p * 3] = r_buf[p];
        out[base + p * 3 + 1] = g_buf[p];
        out[base + p * 3 + 2] = b_buf[p];
      }
      x += 16;
    }
    if x < width {
      // Scalar tail: bit-normalize f16 → host-native f32 (via
      // `scalar_f16::widen_f16_be_to_host_f32::<BE>` which `from_be` /
      // `from_le`-loads the source bits BEFORE the f16 → f32 conversion),
      // then route the scalar kernel via `HOST_NATIVE_BE` to avoid double
      // byte-swap.
      let tail = width - x;
      let mut gf = [0.0f32; 16];
      let mut bf = [0.0f32; 16];
      let mut rf = [0.0f32; 16];
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

/// AVX-512 + F16C: planar Gbrpf16 → packed `R, G, B, A` bytes (widen f16→f32,
/// α = 0xFF). 16 px / iter.
///
/// # Safety
///
/// 1. AVX-512F + AVX-512BW + F16C must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,f16c")]
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
    let zero = _mm512_setzero_ps();
    let one = _mm512_set1_ps(1.0);
    let scale = _mm512_set1_ps(255.0);

    let mut x = 0usize;
    while x + 16 <= width {
      let gv = _mm512_cvtph_ps(endian::load_endian_u16x16::<BE>(
        g.as_ptr().add(x).cast::<u8>(),
      ));
      let bv = _mm512_cvtph_ps(endian::load_endian_u16x16::<BE>(
        b.as_ptr().add(x).cast::<u8>(),
      ));
      let rv = _mm512_cvtph_ps(endian::load_endian_u16x16::<BE>(
        r.as_ptr().add(x).cast::<u8>(),
      ));
      let gc = clamp01(gv, zero, one);
      let bc = clamp01(bv, zero, one);
      let rc = clamp01(rv, zero, one);
      let g8 = _mm512_cvtusepi32_epi8(scale_round_i32(gc, scale));
      let b8 = _mm512_cvtusepi32_epi8(scale_round_i32(bc, scale));
      let r8 = _mm512_cvtusepi32_epi8(scale_round_i32(rc, scale));
      let mut g_buf = [0u8; 16];
      let mut b_buf = [0u8; 16];
      let mut r_buf = [0u8; 16];
      _mm_storeu_si128(g_buf.as_mut_ptr().cast(), g8);
      _mm_storeu_si128(b_buf.as_mut_ptr().cast(), b8);
      _mm_storeu_si128(r_buf.as_mut_ptr().cast(), r8);
      let base = x * 4;
      for p in 0..16 {
        out[base + p * 4] = r_buf[p];
        out[base + p * 4 + 1] = g_buf[p];
        out[base + p * 4 + 2] = b_buf[p];
        out[base + p * 4 + 3] = 0xFF;
      }
      x += 16;
    }
    if x < width {
      // Scalar tail: bit-normalize f16 → host-native f32, then route the
      // scalar kernel via `HOST_NATIVE_BE` to avoid double-byte-swap.
      let tail = width - x;
      let mut gf = [0.0f32; 16];
      let mut bf = [0.0f32; 16];
      let mut rf = [0.0f32; 16];
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

/// AVX-512 + F16C: planar Gbrpf16 → packed `R, G, B` u16 (widen f16→f32).
/// 16 px / iter.
///
/// # Safety
///
/// 1. AVX-512F + AVX-512BW + F16C must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `3 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,f16c")]
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
    let zero = _mm512_setzero_ps();
    let one = _mm512_set1_ps(1.0);
    let scale = _mm512_set1_ps(65535.0);

    let mut x = 0usize;
    while x + 16 <= width {
      let gv = _mm512_cvtph_ps(endian::load_endian_u16x16::<BE>(
        g.as_ptr().add(x).cast::<u8>(),
      ));
      let bv = _mm512_cvtph_ps(endian::load_endian_u16x16::<BE>(
        b.as_ptr().add(x).cast::<u8>(),
      ));
      let rv = _mm512_cvtph_ps(endian::load_endian_u16x16::<BE>(
        r.as_ptr().add(x).cast::<u8>(),
      ));
      let gc = clamp01(gv, zero, one);
      let bc = clamp01(bv, zero, one);
      let rc = clamp01(rv, zero, one);
      let gw = _mm512_cvtusepi32_epi16(scale_round_i32(gc, scale));
      let bw = _mm512_cvtusepi32_epi16(scale_round_i32(bc, scale));
      let rw = _mm512_cvtusepi32_epi16(scale_round_i32(rc, scale));
      let mut g_buf = [0u16; 16];
      let mut b_buf = [0u16; 16];
      let mut r_buf = [0u16; 16];
      _mm256_storeu_si256(g_buf.as_mut_ptr().cast(), gw);
      _mm256_storeu_si256(b_buf.as_mut_ptr().cast(), bw);
      _mm256_storeu_si256(r_buf.as_mut_ptr().cast(), rw);
      let base = x * 3;
      for p in 0..16 {
        out[base + p * 3] = r_buf[p];
        out[base + p * 3 + 1] = g_buf[p];
        out[base + p * 3 + 2] = b_buf[p];
      }
      x += 16;
    }
    if x < width {
      // Scalar tail: bit-normalize f16 → host-native f32, then route the
      // scalar kernel via `HOST_NATIVE_BE` to avoid double-byte-swap.
      let tail = width - x;
      let mut gf = [0.0f32; 16];
      let mut bf = [0.0f32; 16];
      let mut rf = [0.0f32; 16];
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

/// AVX-512 + F16C: planar Gbrpf16 → packed `R, G, B, A` u16 (widen,
/// α = 0xFFFF). 16 px / iter.
///
/// # Safety
///
/// 1. AVX-512F + AVX-512BW + F16C must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,f16c")]
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
    let zero = _mm512_setzero_ps();
    let one = _mm512_set1_ps(1.0);
    let scale = _mm512_set1_ps(65535.0);

    let mut x = 0usize;
    while x + 16 <= width {
      let gv = _mm512_cvtph_ps(endian::load_endian_u16x16::<BE>(
        g.as_ptr().add(x).cast::<u8>(),
      ));
      let bv = _mm512_cvtph_ps(endian::load_endian_u16x16::<BE>(
        b.as_ptr().add(x).cast::<u8>(),
      ));
      let rv = _mm512_cvtph_ps(endian::load_endian_u16x16::<BE>(
        r.as_ptr().add(x).cast::<u8>(),
      ));
      let gc = clamp01(gv, zero, one);
      let bc = clamp01(bv, zero, one);
      let rc = clamp01(rv, zero, one);
      let gw = _mm512_cvtusepi32_epi16(scale_round_i32(gc, scale));
      let bw = _mm512_cvtusepi32_epi16(scale_round_i32(bc, scale));
      let rw = _mm512_cvtusepi32_epi16(scale_round_i32(rc, scale));
      let mut g_buf = [0u16; 16];
      let mut b_buf = [0u16; 16];
      let mut r_buf = [0u16; 16];
      _mm256_storeu_si256(g_buf.as_mut_ptr().cast(), gw);
      _mm256_storeu_si256(b_buf.as_mut_ptr().cast(), bw);
      _mm256_storeu_si256(r_buf.as_mut_ptr().cast(), rw);
      let base = x * 4;
      for p in 0..16 {
        out[base + p * 4] = r_buf[p];
        out[base + p * 4 + 1] = g_buf[p];
        out[base + p * 4 + 2] = b_buf[p];
        out[base + p * 4 + 3] = 0xFFFF;
      }
      x += 16;
    }
    if x < width {
      // Scalar tail: bit-normalize f16 → host-native f32, then route the
      // scalar kernel via `HOST_NATIVE_BE` to avoid double-byte-swap.
      let tail = width - x;
      let mut gf = [0.0f32; 16];
      let mut bf = [0.0f32; 16];
      let mut rf = [0.0f32; 16];
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

/// AVX-512 + F16C: planar Gbrpf16 → packed `R, G, B` f32 (lossless widen).
/// 16 px / iter.
///
/// # Safety
///
/// 1. AVX-512F + AVX-512BW + F16C must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `3 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,f16c")]
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
    while x + 16 <= width {
      let gv = _mm512_cvtph_ps(endian::load_endian_u16x16::<BE>(
        g.as_ptr().add(x).cast::<u8>(),
      ));
      let bv = _mm512_cvtph_ps(endian::load_endian_u16x16::<BE>(
        b.as_ptr().add(x).cast::<u8>(),
      ));
      let rv = _mm512_cvtph_ps(endian::load_endian_u16x16::<BE>(
        r.as_ptr().add(x).cast::<u8>(),
      ));
      // No 3-channel interleave intrinsic in AVX-512 — scatter via scalar loop.
      let mut gf = [0.0f32; 16];
      let mut bf = [0.0f32; 16];
      let mut rf = [0.0f32; 16];
      _mm512_storeu_ps(gf.as_mut_ptr(), gv);
      _mm512_storeu_ps(bf.as_mut_ptr(), bv);
      _mm512_storeu_ps(rf.as_mut_ptr(), rv);
      let base = x * 3;
      for p in 0..16 {
        out[base + p * 3] = rf[p];
        out[base + p * 3 + 1] = gf[p];
        out[base + p * 3 + 2] = bf[p];
      }
      x += 16;
    }
    if x < width {
      // Scalar tail: widen f16 → host-native f32 (normalize source bits via
      // `from_be` / `from_le` BEFORE the f16 → f32 conversion), then route the
      // scalar kernel via `HOST_NATIVE_BE` to avoid double-byte-swap.
      let tail = width - x;
      let mut gf = [0.0f32; 16];
      let mut bf = [0.0f32; 16];
      let mut rf = [0.0f32; 16];
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

/// AVX-512 + F16C: planar Gbrpf16 → packed `R, G, B, A` f32 (lossless,
/// α = 1.0). 16 px / iter.
///
/// # Safety
///
/// 1. AVX-512F + AVX-512BW + F16C must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,f16c")]
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
    while x + 16 <= width {
      let gv = _mm512_cvtph_ps(endian::load_endian_u16x16::<BE>(
        g.as_ptr().add(x).cast::<u8>(),
      ));
      let bv = _mm512_cvtph_ps(endian::load_endian_u16x16::<BE>(
        b.as_ptr().add(x).cast::<u8>(),
      ));
      let rv = _mm512_cvtph_ps(endian::load_endian_u16x16::<BE>(
        r.as_ptr().add(x).cast::<u8>(),
      ));
      let mut gf = [0.0f32; 16];
      let mut bf = [0.0f32; 16];
      let mut rf = [0.0f32; 16];
      _mm512_storeu_ps(gf.as_mut_ptr(), gv);
      _mm512_storeu_ps(bf.as_mut_ptr(), bv);
      _mm512_storeu_ps(rf.as_mut_ptr(), rv);
      let base = x * 4;
      for p in 0..16 {
        out[base + p * 4] = rf[p];
        out[base + p * 4 + 1] = gf[p];
        out[base + p * 4 + 2] = bf[p];
        out[base + p * 4 + 3] = 1.0;
      }
      x += 16;
    }
    if x < width {
      // Scalar tail: widen f16 → host-native f32 (normalize source bits via
      // `from_be` / `from_le` BEFORE the f16 → f32 conversion), then route the
      // scalar kernel via `HOST_NATIVE_BE` to avoid double-byte-swap.
      let tail = width - x;
      let mut gf = [0.0f32; 16];
      let mut bf = [0.0f32; 16];
      let mut rf = [0.0f32; 16];
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

/// AVX-512: planar Gbrpf16 → packed `R, G, B` f16 (lossless — f16 treated
/// as u16).
///
/// No F16C gate needed: f16 planes are bit-copied as opaque u16 lanes.
/// 16 px / iter via 32-byte unaligned loads + manual scatter (no vst3
/// equivalent in AVX-512).
///
/// # Safety
///
/// 1. AVX-512F + AVX-512BW must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `3 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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
    while x + 16 <= width {
      // Load 16 × u16 (32 bytes) per plane.
      let gu = endian::load_endian_u16x16::<BE>(g.as_ptr().add(x).cast::<u8>());
      let bu = endian::load_endian_u16x16::<BE>(b.as_ptr().add(x).cast::<u8>());
      let ru = endian::load_endian_u16x16::<BE>(r.as_ptr().add(x).cast::<u8>());
      let mut g_buf = [0u16; 16];
      let mut b_buf = [0u16; 16];
      let mut r_buf = [0u16; 16];
      _mm256_storeu_si256(g_buf.as_mut_ptr().cast(), gu);
      _mm256_storeu_si256(b_buf.as_mut_ptr().cast(), bu);
      _mm256_storeu_si256(r_buf.as_mut_ptr().cast(), ru);
      let base = x * 3;
      for p in 0..16 {
        let dst = out.as_mut_ptr().add(base + p * 3);
        *dst.cast::<u16>() = r_buf[p];
        *dst.add(1).cast::<u16>() = g_buf[p];
        *dst.add(2).cast::<u16>() = b_buf[p];
      }
      x += 16;
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

/// AVX-512: planar Gbrpf16 → packed `R, G, B, A` f16 (lossless,
/// α = f16(1.0) = 0x3C00).
///
/// No F16C gate needed.
///
/// # Safety
///
/// 1. AVX-512F + AVX-512BW must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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
    while x + 16 <= width {
      let gu = endian::load_endian_u16x16::<BE>(g.as_ptr().add(x).cast::<u8>());
      let bu = endian::load_endian_u16x16::<BE>(b.as_ptr().add(x).cast::<u8>());
      let ru = endian::load_endian_u16x16::<BE>(r.as_ptr().add(x).cast::<u8>());
      let mut g_buf = [0u16; 16];
      let mut b_buf = [0u16; 16];
      let mut r_buf = [0u16; 16];
      _mm256_storeu_si256(g_buf.as_mut_ptr().cast(), gu);
      _mm256_storeu_si256(b_buf.as_mut_ptr().cast(), bu);
      _mm256_storeu_si256(r_buf.as_mut_ptr().cast(), ru);
      let base = x * 4;
      for p in 0..16 {
        let dst = out.as_mut_ptr().add(base + p * 4);
        *dst.cast::<u16>() = r_buf[p];
        *dst.add(1).cast::<u16>() = g_buf[p];
        *dst.add(2).cast::<u16>() = b_buf[p];
        *dst.add(3).cast::<u16>() = 0x3C00u16; // f16(1.0)
      }
      x += 16;
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

/// AVX-512 + F16C: planar Gbrpf16 → u8 luma (widen + staged via RGB scratch).
///
/// # Safety
///
/// 1. AVX-512F + AVX-512BW + F16C must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,f16c")]
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

/// AVX-512 + F16C: planar Gbrpf16 → u16 luma (widen + staged via RGB scratch).
///
/// # Safety
///
/// 1. AVX-512F + AVX-512BW + F16C must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,f16c")]
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

/// AVX-512 + F16C: planar Gbrpf16 → planar HSV bytes (widen + staged via RGB).
///
/// # Safety
///
/// 1. AVX-512F + AVX-512BW + F16C must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `h_out.len()`, `s_out.len()`, `v_out.len()` ≥ `width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,f16c")]
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

/// AVX-512 + F16C: planar Gbrapf16 → packed `R, G, B, A` bytes (widen,
/// source α). 16 px / iter.
///
/// # Safety
///
/// 1. AVX-512F + AVX-512BW + F16C must be available.
/// 2. `g.len()`, `b.len()`, `r.len()`, `a.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,f16c")]
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
    let zero = _mm512_setzero_ps();
    let one = _mm512_set1_ps(1.0);
    let scale = _mm512_set1_ps(255.0);

    let mut x = 0usize;
    while x + 16 <= width {
      let gv = _mm512_cvtph_ps(endian::load_endian_u16x16::<BE>(
        g.as_ptr().add(x).cast::<u8>(),
      ));
      let bv = _mm512_cvtph_ps(endian::load_endian_u16x16::<BE>(
        b.as_ptr().add(x).cast::<u8>(),
      ));
      let rv = _mm512_cvtph_ps(endian::load_endian_u16x16::<BE>(
        r.as_ptr().add(x).cast::<u8>(),
      ));
      let av = _mm512_cvtph_ps(endian::load_endian_u16x16::<BE>(
        a.as_ptr().add(x).cast::<u8>(),
      ));
      let gc = clamp01(gv, zero, one);
      let bc = clamp01(bv, zero, one);
      let rc = clamp01(rv, zero, one);
      let ac = clamp01(av, zero, one);
      let g8 = _mm512_cvtusepi32_epi8(scale_round_i32(gc, scale));
      let b8 = _mm512_cvtusepi32_epi8(scale_round_i32(bc, scale));
      let r8 = _mm512_cvtusepi32_epi8(scale_round_i32(rc, scale));
      let a8 = _mm512_cvtusepi32_epi8(scale_round_i32(ac, scale));
      let mut g_buf = [0u8; 16];
      let mut b_buf = [0u8; 16];
      let mut r_buf = [0u8; 16];
      let mut a_buf = [0u8; 16];
      _mm_storeu_si128(g_buf.as_mut_ptr().cast(), g8);
      _mm_storeu_si128(b_buf.as_mut_ptr().cast(), b8);
      _mm_storeu_si128(r_buf.as_mut_ptr().cast(), r8);
      _mm_storeu_si128(a_buf.as_mut_ptr().cast(), a8);
      let base = x * 4;
      for p in 0..16 {
        out[base + p * 4] = r_buf[p];
        out[base + p * 4 + 1] = g_buf[p];
        out[base + p * 4 + 2] = b_buf[p];
        out[base + p * 4 + 3] = a_buf[p];
      }
      x += 16;
    }
    if x < width {
      // Scalar tail: bit-normalize f16 → host-native f32, then route the
      // scalar kernel via `HOST_NATIVE_BE` to avoid double-byte-swap.
      let tail = width - x;
      let mut gf = [0.0f32; 16];
      let mut bf = [0.0f32; 16];
      let mut rf = [0.0f32; 16];
      let mut af = [0.0f32; 16];
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

/// AVX-512 + F16C: planar Gbrapf16 → packed `R, G, B, A` u16 (widen,
/// source α). 16 px / iter.
///
/// # Safety
///
/// 1. AVX-512F + AVX-512BW + F16C must be available.
/// 2. `g.len()`, `b.len()`, `r.len()`, `a.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,f16c")]
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
    let zero = _mm512_setzero_ps();
    let one = _mm512_set1_ps(1.0);
    let scale = _mm512_set1_ps(65535.0);

    let mut x = 0usize;
    while x + 16 <= width {
      let gv = _mm512_cvtph_ps(endian::load_endian_u16x16::<BE>(
        g.as_ptr().add(x).cast::<u8>(),
      ));
      let bv = _mm512_cvtph_ps(endian::load_endian_u16x16::<BE>(
        b.as_ptr().add(x).cast::<u8>(),
      ));
      let rv = _mm512_cvtph_ps(endian::load_endian_u16x16::<BE>(
        r.as_ptr().add(x).cast::<u8>(),
      ));
      let av = _mm512_cvtph_ps(endian::load_endian_u16x16::<BE>(
        a.as_ptr().add(x).cast::<u8>(),
      ));
      let gc = clamp01(gv, zero, one);
      let bc = clamp01(bv, zero, one);
      let rc = clamp01(rv, zero, one);
      let ac = clamp01(av, zero, one);
      let gw = _mm512_cvtusepi32_epi16(scale_round_i32(gc, scale));
      let bw = _mm512_cvtusepi32_epi16(scale_round_i32(bc, scale));
      let rw = _mm512_cvtusepi32_epi16(scale_round_i32(rc, scale));
      let aw = _mm512_cvtusepi32_epi16(scale_round_i32(ac, scale));
      let mut g_buf = [0u16; 16];
      let mut b_buf = [0u16; 16];
      let mut r_buf = [0u16; 16];
      let mut a_buf = [0u16; 16];
      _mm256_storeu_si256(g_buf.as_mut_ptr().cast(), gw);
      _mm256_storeu_si256(b_buf.as_mut_ptr().cast(), bw);
      _mm256_storeu_si256(r_buf.as_mut_ptr().cast(), rw);
      _mm256_storeu_si256(a_buf.as_mut_ptr().cast(), aw);
      let base = x * 4;
      for p in 0..16 {
        out[base + p * 4] = r_buf[p];
        out[base + p * 4 + 1] = g_buf[p];
        out[base + p * 4 + 2] = b_buf[p];
        out[base + p * 4 + 3] = a_buf[p];
      }
      x += 16;
    }
    if x < width {
      // Scalar tail: bit-normalize f16 → host-native f32, then route the
      // scalar kernel via `HOST_NATIVE_BE` to avoid double-byte-swap.
      let tail = width - x;
      let mut gf = [0.0f32; 16];
      let mut bf = [0.0f32; 16];
      let mut rf = [0.0f32; 16];
      let mut af = [0.0f32; 16];
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

/// AVX-512 + F16C: planar Gbrapf16 → packed `R, G, B, A` f32 (lossless,
/// source α). 16 px / iter.
///
/// # Safety
///
/// 1. AVX-512F + AVX-512BW + F16C must be available.
/// 2. `g.len()`, `b.len()`, `r.len()`, `a.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,f16c")]
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
    while x + 16 <= width {
      let gv = _mm512_cvtph_ps(endian::load_endian_u16x16::<BE>(
        g.as_ptr().add(x).cast::<u8>(),
      ));
      let bv = _mm512_cvtph_ps(endian::load_endian_u16x16::<BE>(
        b.as_ptr().add(x).cast::<u8>(),
      ));
      let rv = _mm512_cvtph_ps(endian::load_endian_u16x16::<BE>(
        r.as_ptr().add(x).cast::<u8>(),
      ));
      let av = _mm512_cvtph_ps(endian::load_endian_u16x16::<BE>(
        a.as_ptr().add(x).cast::<u8>(),
      ));
      let mut gf = [0.0f32; 16];
      let mut bf = [0.0f32; 16];
      let mut rf = [0.0f32; 16];
      let mut af = [0.0f32; 16];
      _mm512_storeu_ps(gf.as_mut_ptr(), gv);
      _mm512_storeu_ps(bf.as_mut_ptr(), bv);
      _mm512_storeu_ps(rf.as_mut_ptr(), rv);
      _mm512_storeu_ps(af.as_mut_ptr(), av);
      let base = x * 4;
      for p in 0..16 {
        out[base + p * 4] = rf[p];
        out[base + p * 4 + 1] = gf[p];
        out[base + p * 4 + 2] = bf[p];
        out[base + p * 4 + 3] = af[p];
      }
      x += 16;
    }
    if x < width {
      // Scalar tail: widen f16 → host-native f32 (normalize source bits via
      // `from_be` / `from_le` BEFORE the f16 → f32 conversion), then route the
      // scalar kernel via `HOST_NATIVE_BE` to avoid double-byte-swap.
      let tail = width - x;
      let mut gf = [0.0f32; 16];
      let mut bf = [0.0f32; 16];
      let mut rf = [0.0f32; 16];
      let mut af = [0.0f32; 16];
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

/// AVX-512: planar Gbrapf16 → packed `R, G, B, A` f16 (lossless, source α).
///
/// No F16C gate needed: f16 planes are bit-copied as opaque u16.
///
/// # Safety
///
/// 1. AVX-512F + AVX-512BW must be available.
/// 2. `g.len()`, `b.len()`, `r.len()`, `a.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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
    while x + 16 <= width {
      let gu = endian::load_endian_u16x16::<BE>(g.as_ptr().add(x).cast::<u8>());
      let bu = endian::load_endian_u16x16::<BE>(b.as_ptr().add(x).cast::<u8>());
      let ru = endian::load_endian_u16x16::<BE>(r.as_ptr().add(x).cast::<u8>());
      let au = endian::load_endian_u16x16::<BE>(a.as_ptr().add(x).cast::<u8>());
      let mut g_buf = [0u16; 16];
      let mut b_buf = [0u16; 16];
      let mut r_buf = [0u16; 16];
      let mut a_buf = [0u16; 16];
      _mm256_storeu_si256(g_buf.as_mut_ptr().cast(), gu);
      _mm256_storeu_si256(b_buf.as_mut_ptr().cast(), bu);
      _mm256_storeu_si256(r_buf.as_mut_ptr().cast(), ru);
      _mm256_storeu_si256(a_buf.as_mut_ptr().cast(), au);
      let base = x * 4;
      for p in 0..16 {
        let dst = out.as_mut_ptr().add(base + p * 4);
        *dst.cast::<u16>() = r_buf[p];
        *dst.add(1).cast::<u16>() = g_buf[p];
        *dst.add(2).cast::<u16>() = b_buf[p];
        *dst.add(3).cast::<u16>() = a_buf[p];
      }
      x += 16;
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
