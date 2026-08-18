//! AVX2 SIMD kernels for planar GBR float sources (Tier 10).
//!
//! f32 path: 8 pixels / iteration via `__m256` (one G, B, R [+ A] register
//! each). f16 narrowing / widening gated on the x86 `f16c` feature
//! (detected at runtime via `is_x86_feature_detected!("f16c")`); scalar
//! fallback used when F16C is absent.
//!
//! # Rounding (f32 → u8 / u16)
//!
//! `_mm256_add_ps(scaled, _mm256_set1_ps(0.5))` then `_mm256_cvttps_epi32`
//! (truncate toward zero). This is the round-half-up contract shared with
//! the scalar kernels — MXCSR-independent and consistent with the Grayf32
//! path.
//!
//! **Do NOT use `_MM_FROUND_TO_NEAREST_INT` for integer narrowing** —
//! that gives banker's rounding, not round-half-up.
//!
//! # Rounding (f32 → f16)
//!
//! F16C `_mm256_cvtps_ph::<{ _MM_FROUND_TO_NEAREST_INT }>`:
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
//! # Lane-cross discipline
//!
//! AVX2 has the 128-bit lane boundary. We narrow `__m256i` (8xi32) to u8/u16
//! by splitting into two `__m128i` halves and using SSE-style packs
//! (`_mm_packs_epi32`, `_mm_packus_epi32`, `_mm_packus_epi16`); this avoids
//! the lane-split that 256-bit packs leave behind and matches the Grayf32
//! AVX2 pattern.
//!
//! For f16 narrow, `_mm256_cvtps_ph` returns a `__m128i` with 8 in-order
//! f16 lanes — no lane fixup needed. For f16 widen, `_mm256_cvtph_ps`
//! takes a `__m128i` and returns 8 in-order f32 lanes — also no fixup.
//!
//! # f16 lossless interleave
//!
//! Treat f16 lanes as opaque `u16` — no arithmetic, no F16C gate needed.

#[cfg_attr(miri, allow(unused_imports))]
use core::arch::x86_64::*;

use crate::{
  KernelMatrix,
  row::{
    arch::x86_avx2::endian,
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

/// Clamp a `__m256` (f32x8) to `[0.0, 1.0]`.
#[inline(always)]
unsafe fn clamp01(v: __m256, zero: __m256, one: __m256) -> __m256 {
  unsafe { _mm256_min_ps(_mm256_max_ps(v, zero), one) }
}

/// Scale, add 0.5, truncate → `__m256i` (i32x8). Round-half-up.
#[inline(always)]
unsafe fn scale_round_i32(v: __m256, scale: __m256) -> __m256i {
  unsafe { _mm256_cvttps_epi32(_mm256_add_ps(_mm256_mul_ps(v, scale), _mm256_set1_ps(0.5))) }
}

/// Narrow an `__m256i` (i32x8) to 8 u8 bytes (returned as the low 64 bits
/// of an `__m128i`). Saturates via `_mm_packs_epi32` then `_mm_packus_epi16`.
/// Lane-safe: extracts the two 128-bit halves first, then uses SSE packs.
#[inline(always)]
unsafe fn narrow_i32x8_to_u8x8(v: __m256i) -> __m128i {
  unsafe {
    let lo = _mm256_castsi256_si128(v);
    let hi = _mm256_extracti128_si256::<1>(v);
    let pack16 = _mm_packs_epi32(lo, hi); // 8xi16
    _mm_packus_epi16(pack16, pack16) // 8xu8 in low 8 bytes
  }
}

/// Narrow an `__m256i` (i32x8) to 8 u16 lanes (returned as a full `__m128i`).
/// Uses `_mm_packus_epi32` (SSE4.1) on the two extracted 128-bit halves.
#[inline(always)]
unsafe fn narrow_i32x8_to_u16x8(v: __m256i) -> __m128i {
  unsafe {
    let lo = _mm256_castsi256_si128(v);
    let hi = _mm256_extracti128_si256::<1>(v);
    _mm_packus_epi32(lo, hi)
  }
}

// ---- Gbrpf32 → u8 RGB -------------------------------------------------------

/// AVX2: planar Gbrpf32 → packed `R, G, B` bytes. 8 px / iter.
///
/// Round-half-up: `+ 0.5` then `_mm256_cvttps_epi32`.
///
/// # Safety
///
/// 1. AVX2 must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `3 * width`.
#[inline]
#[target_feature(enable = "avx2")]
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
    let zero = _mm256_setzero_ps();
    let one = _mm256_set1_ps(1.0);
    let scale = _mm256_set1_ps(255.0);

    let mut x = 0usize;
    while x + 8 <= width {
      let gv = clamp01(
        _mm256_castsi256_ps(endian::load_endian_u32x8::<BE>(
          g.as_ptr().add(x).cast::<u8>(),
        )),
        zero,
        one,
      );
      let bv = clamp01(
        _mm256_castsi256_ps(endian::load_endian_u32x8::<BE>(
          b.as_ptr().add(x).cast::<u8>(),
        )),
        zero,
        one,
      );
      let rv = clamp01(
        _mm256_castsi256_ps(endian::load_endian_u32x8::<BE>(
          r.as_ptr().add(x).cast::<u8>(),
        )),
        zero,
        one,
      );
      let g8 = narrow_i32x8_to_u8x8(scale_round_i32(gv, scale));
      let b8 = narrow_i32x8_to_u8x8(scale_round_i32(bv, scale));
      let r8 = narrow_i32x8_to_u8x8(scale_round_i32(rv, scale));
      let mut g_buf = [0u8; 16];
      let mut b_buf = [0u8; 16];
      let mut r_buf = [0u8; 16];
      _mm_storeu_si128(g_buf.as_mut_ptr().cast(), g8);
      _mm_storeu_si128(b_buf.as_mut_ptr().cast(), b8);
      _mm_storeu_si128(r_buf.as_mut_ptr().cast(), r8);
      let base = x * 3;
      for p in 0..8 {
        out[base + p * 3] = r_buf[p];
        out[base + p * 3 + 1] = g_buf[p];
        out[base + p * 3 + 2] = b_buf[p];
      }
      x += 8;
    }
    if x < width {
      scalar::gbrpf32_to_rgb_row::<BE>(&g[x..], &b[x..], &r[x..], &mut out[x * 3..], width - x);
    }
  }
}

// ---- Gbrpf32 → u8 RGBA (opaque α) ------------------------------------------

/// AVX2: planar Gbrpf32 → packed `R, G, B, A` bytes (α = 0xFF). 8 px / iter.
///
/// # Safety
///
/// 1. AVX2 must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "avx2")]
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
    let zero = _mm256_setzero_ps();
    let one = _mm256_set1_ps(1.0);
    let scale = _mm256_set1_ps(255.0);

    let mut x = 0usize;
    while x + 8 <= width {
      let gv = clamp01(
        _mm256_castsi256_ps(endian::load_endian_u32x8::<BE>(
          g.as_ptr().add(x).cast::<u8>(),
        )),
        zero,
        one,
      );
      let bv = clamp01(
        _mm256_castsi256_ps(endian::load_endian_u32x8::<BE>(
          b.as_ptr().add(x).cast::<u8>(),
        )),
        zero,
        one,
      );
      let rv = clamp01(
        _mm256_castsi256_ps(endian::load_endian_u32x8::<BE>(
          r.as_ptr().add(x).cast::<u8>(),
        )),
        zero,
        one,
      );
      let g8 = narrow_i32x8_to_u8x8(scale_round_i32(gv, scale));
      let b8 = narrow_i32x8_to_u8x8(scale_round_i32(bv, scale));
      let r8 = narrow_i32x8_to_u8x8(scale_round_i32(rv, scale));
      let mut g_buf = [0u8; 16];
      let mut b_buf = [0u8; 16];
      let mut r_buf = [0u8; 16];
      _mm_storeu_si128(g_buf.as_mut_ptr().cast(), g8);
      _mm_storeu_si128(b_buf.as_mut_ptr().cast(), b8);
      _mm_storeu_si128(r_buf.as_mut_ptr().cast(), r8);
      let base = x * 4;
      for p in 0..8 {
        out[base + p * 4] = r_buf[p];
        out[base + p * 4 + 1] = g_buf[p];
        out[base + p * 4 + 2] = b_buf[p];
        out[base + p * 4 + 3] = 0xFF;
      }
      x += 8;
    }
    if x < width {
      scalar::gbrpf32_to_rgba_row::<BE>(&g[x..], &b[x..], &r[x..], &mut out[x * 4..], width - x);
    }
  }
}

// ---- Gbrpf32 → u16 RGB ------------------------------------------------------

/// AVX2: planar Gbrpf32 → packed `R, G, B` u16. 8 px / iter.
///
/// # Safety
///
/// 1. AVX2 must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `3 * width`.
#[inline]
#[target_feature(enable = "avx2")]
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
    let zero = _mm256_setzero_ps();
    let one = _mm256_set1_ps(1.0);
    let scale = _mm256_set1_ps(65535.0);

    let mut x = 0usize;
    while x + 8 <= width {
      let gv = clamp01(
        _mm256_castsi256_ps(endian::load_endian_u32x8::<BE>(
          g.as_ptr().add(x).cast::<u8>(),
        )),
        zero,
        one,
      );
      let bv = clamp01(
        _mm256_castsi256_ps(endian::load_endian_u32x8::<BE>(
          b.as_ptr().add(x).cast::<u8>(),
        )),
        zero,
        one,
      );
      let rv = clamp01(
        _mm256_castsi256_ps(endian::load_endian_u32x8::<BE>(
          r.as_ptr().add(x).cast::<u8>(),
        )),
        zero,
        one,
      );
      let gw = narrow_i32x8_to_u16x8(scale_round_i32(gv, scale));
      let bw = narrow_i32x8_to_u16x8(scale_round_i32(bv, scale));
      let rw = narrow_i32x8_to_u16x8(scale_round_i32(rv, scale));
      let mut g_buf = [0u16; 8];
      let mut b_buf = [0u16; 8];
      let mut r_buf = [0u16; 8];
      _mm_storeu_si128(g_buf.as_mut_ptr().cast(), gw);
      _mm_storeu_si128(b_buf.as_mut_ptr().cast(), bw);
      _mm_storeu_si128(r_buf.as_mut_ptr().cast(), rw);
      let base = x * 3;
      for p in 0..8 {
        out[base + p * 3] = r_buf[p];
        out[base + p * 3 + 1] = g_buf[p];
        out[base + p * 3 + 2] = b_buf[p];
      }
      x += 8;
    }
    if x < width {
      scalar::gbrpf32_to_rgb_u16_row::<BE>(&g[x..], &b[x..], &r[x..], &mut out[x * 3..], width - x);
    }
  }
}

// ---- Gbrpf32 → u16 RGBA (opaque α) -----------------------------------------

/// AVX2: planar Gbrpf32 → packed `R, G, B, A` u16 (α = 0xFFFF). 8 px / iter.
///
/// # Safety
///
/// 1. AVX2 must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "avx2")]
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
    let zero = _mm256_setzero_ps();
    let one = _mm256_set1_ps(1.0);
    let scale = _mm256_set1_ps(65535.0);

    let mut x = 0usize;
    while x + 8 <= width {
      let gv = clamp01(
        _mm256_castsi256_ps(endian::load_endian_u32x8::<BE>(
          g.as_ptr().add(x).cast::<u8>(),
        )),
        zero,
        one,
      );
      let bv = clamp01(
        _mm256_castsi256_ps(endian::load_endian_u32x8::<BE>(
          b.as_ptr().add(x).cast::<u8>(),
        )),
        zero,
        one,
      );
      let rv = clamp01(
        _mm256_castsi256_ps(endian::load_endian_u32x8::<BE>(
          r.as_ptr().add(x).cast::<u8>(),
        )),
        zero,
        one,
      );
      let gw = narrow_i32x8_to_u16x8(scale_round_i32(gv, scale));
      let bw = narrow_i32x8_to_u16x8(scale_round_i32(bv, scale));
      let rw = narrow_i32x8_to_u16x8(scale_round_i32(rv, scale));
      let mut g_buf = [0u16; 8];
      let mut b_buf = [0u16; 8];
      let mut r_buf = [0u16; 8];
      _mm_storeu_si128(g_buf.as_mut_ptr().cast(), gw);
      _mm_storeu_si128(b_buf.as_mut_ptr().cast(), bw);
      _mm_storeu_si128(r_buf.as_mut_ptr().cast(), rw);
      let base = x * 4;
      for p in 0..8 {
        out[base + p * 4] = r_buf[p];
        out[base + p * 4 + 1] = g_buf[p];
        out[base + p * 4 + 2] = b_buf[p];
        out[base + p * 4 + 3] = 0xFFFF;
      }
      x += 8;
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

/// AVX2: planar Gbrpf32 → packed `R, G, B` f32 (lossless interleave).
///
/// AVX2 has no 3-channel interleave store; use scalar (compiler vectorises
/// the simple per-element copy well).
///
/// # Safety
///
/// 1. AVX2 must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `3 * width`.
#[inline]
#[target_feature(enable = "avx2")]
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

/// AVX2: planar Gbrpf32 → packed `R, G, B, A` f32 (lossless, α = 1.0).
///
/// AVX2 has no 4-channel interleave store; use scalar.
///
/// # Safety
///
/// 1. AVX2 must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "avx2")]
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

/// AVX2 + F16C: planar Gbrpf32 → packed `R, G, B` f16 (fused narrow).
/// 8 px / iter.
///
/// Uses `_mm256_cvtps_ph` with `_MM_FROUND_TO_NEAREST_INT`
/// (IEEE-754 round-to-nearest-even).
///
/// # Safety
///
/// 1. AVX2 and F16C must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `3 * width`.
#[inline]
#[target_feature(enable = "avx2,f16c")]
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
    while x + 8 <= width {
      let gv = _mm256_castsi256_ps(endian::load_endian_u32x8::<BE>(
        g.as_ptr().add(x).cast::<u8>(),
      ));
      let bv = _mm256_castsi256_ps(endian::load_endian_u32x8::<BE>(
        b.as_ptr().add(x).cast::<u8>(),
      ));
      let rv = _mm256_castsi256_ps(endian::load_endian_u32x8::<BE>(
        r.as_ptr().add(x).cast::<u8>(),
      ));
      // F16C narrow: IEEE-754 round-to-nearest-even (NOT round-half-up).
      let gh = _mm256_cvtps_ph::<{ _MM_FROUND_TO_NEAREST_INT }>(gv);
      let bh = _mm256_cvtps_ph::<{ _MM_FROUND_TO_NEAREST_INT }>(bv);
      let rh = _mm256_cvtps_ph::<{ _MM_FROUND_TO_NEAREST_INT }>(rv);
      // Each `__m128i` holds 8 f16 lanes in natural order.
      let mut g_buf = [0u16; 8];
      let mut b_buf = [0u16; 8];
      let mut r_buf = [0u16; 8];
      _mm_storeu_si128(g_buf.as_mut_ptr().cast(), gh);
      _mm_storeu_si128(b_buf.as_mut_ptr().cast(), bh);
      _mm_storeu_si128(r_buf.as_mut_ptr().cast(), rh);
      let base = x * 3;
      for p in 0..8 {
        let dst = out.as_mut_ptr().add(base + p * 3);
        *dst.cast::<u16>() = r_buf[p];
        *dst.add(1).cast::<u16>() = g_buf[p];
        *dst.add(2).cast::<u16>() = b_buf[p];
      }
      x += 8;
    }
    if x < width {
      scalar::gbrpf32_to_rgb_f16_row::<BE>(&g[x..], &b[x..], &r[x..], &mut out[x * 3..], width - x);
    }
  }
}

// ---- Gbrpf32 → f16 RGBA (F16C narrow, α = f16(1.0)) -------------------------

/// AVX2 + F16C: planar Gbrpf32 → packed `R, G, B, A` f16 (fused narrow,
/// α = f16(1.0) = 0x3C00). 8 px / iter.
///
/// # Safety
///
/// 1. AVX2 and F16C must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "avx2,f16c")]
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
    while x + 8 <= width {
      let gv = _mm256_castsi256_ps(endian::load_endian_u32x8::<BE>(
        g.as_ptr().add(x).cast::<u8>(),
      ));
      let bv = _mm256_castsi256_ps(endian::load_endian_u32x8::<BE>(
        b.as_ptr().add(x).cast::<u8>(),
      ));
      let rv = _mm256_castsi256_ps(endian::load_endian_u32x8::<BE>(
        r.as_ptr().add(x).cast::<u8>(),
      ));
      let gh = _mm256_cvtps_ph::<{ _MM_FROUND_TO_NEAREST_INT }>(gv);
      let bh = _mm256_cvtps_ph::<{ _MM_FROUND_TO_NEAREST_INT }>(bv);
      let rh = _mm256_cvtps_ph::<{ _MM_FROUND_TO_NEAREST_INT }>(rv);
      let mut g_buf = [0u16; 8];
      let mut b_buf = [0u16; 8];
      let mut r_buf = [0u16; 8];
      _mm_storeu_si128(g_buf.as_mut_ptr().cast(), gh);
      _mm_storeu_si128(b_buf.as_mut_ptr().cast(), bh);
      _mm_storeu_si128(r_buf.as_mut_ptr().cast(), rh);
      let base = x * 4;
      for p in 0..8 {
        let dst = out.as_mut_ptr().add(base + p * 4);
        *dst.cast::<u16>() = r_buf[p];
        *dst.add(1).cast::<u16>() = g_buf[p];
        *dst.add(2).cast::<u16>() = b_buf[p];
        *dst.add(3).cast::<u16>() = 0x3C00u16; // f16(1.0)
      }
      x += 8;
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

/// AVX2: planar Gbrpf32 → u8 luma (staged via AVX2 RGB kernel).
///
/// # Safety
///
/// 1. AVX2 must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `width`.
#[inline]
#[target_feature(enable = "avx2")]
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

/// AVX2: planar Gbrpf32 → u16 luma (staged via AVX2 RGB kernel).
///
/// # Safety
///
/// 1. AVX2 must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `width`.
#[inline]
#[target_feature(enable = "avx2")]
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

/// AVX2: planar Gbrpf32 → planar HSV bytes (staged via AVX2 RGB kernel).
///
/// # Safety
///
/// 1. AVX2 must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `h_out.len()`, `s_out.len()`, `v_out.len()` ≥ `width`.
#[inline]
#[target_feature(enable = "avx2")]
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

/// AVX2: planar Gbrapf32 → packed `R, G, B, A` bytes (source α). 8 px / iter.
///
/// # Safety
///
/// 1. AVX2 must be available.
/// 2. `g.len()`, `b.len()`, `r.len()`, `a.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "avx2")]
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
    let zero = _mm256_setzero_ps();
    let one = _mm256_set1_ps(1.0);
    let scale = _mm256_set1_ps(255.0);

    let mut x = 0usize;
    while x + 8 <= width {
      let gv = clamp01(
        _mm256_castsi256_ps(endian::load_endian_u32x8::<BE>(
          g.as_ptr().add(x).cast::<u8>(),
        )),
        zero,
        one,
      );
      let bv = clamp01(
        _mm256_castsi256_ps(endian::load_endian_u32x8::<BE>(
          b.as_ptr().add(x).cast::<u8>(),
        )),
        zero,
        one,
      );
      let rv = clamp01(
        _mm256_castsi256_ps(endian::load_endian_u32x8::<BE>(
          r.as_ptr().add(x).cast::<u8>(),
        )),
        zero,
        one,
      );
      let av = clamp01(
        _mm256_castsi256_ps(endian::load_endian_u32x8::<BE>(
          a.as_ptr().add(x).cast::<u8>(),
        )),
        zero,
        one,
      );
      let g8 = narrow_i32x8_to_u8x8(scale_round_i32(gv, scale));
      let b8 = narrow_i32x8_to_u8x8(scale_round_i32(bv, scale));
      let r8 = narrow_i32x8_to_u8x8(scale_round_i32(rv, scale));
      let a8 = narrow_i32x8_to_u8x8(scale_round_i32(av, scale));
      let mut g_buf = [0u8; 16];
      let mut b_buf = [0u8; 16];
      let mut r_buf = [0u8; 16];
      let mut a_buf = [0u8; 16];
      _mm_storeu_si128(g_buf.as_mut_ptr().cast(), g8);
      _mm_storeu_si128(b_buf.as_mut_ptr().cast(), b8);
      _mm_storeu_si128(r_buf.as_mut_ptr().cast(), r8);
      _mm_storeu_si128(a_buf.as_mut_ptr().cast(), a8);
      let base = x * 4;
      for p in 0..8 {
        out[base + p * 4] = r_buf[p];
        out[base + p * 4 + 1] = g_buf[p];
        out[base + p * 4 + 2] = b_buf[p];
        out[base + p * 4 + 3] = a_buf[p];
      }
      x += 8;
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

/// AVX2: planar Gbrapf32 → packed `R, G, B, A` u16 (source α). 8 px / iter.
///
/// # Safety
///
/// 1. AVX2 must be available.
/// 2. `g.len()`, `b.len()`, `r.len()`, `a.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "avx2")]
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
    let zero = _mm256_setzero_ps();
    let one = _mm256_set1_ps(1.0);
    let scale = _mm256_set1_ps(65535.0);

    let mut x = 0usize;
    while x + 8 <= width {
      let gv = clamp01(
        _mm256_castsi256_ps(endian::load_endian_u32x8::<BE>(
          g.as_ptr().add(x).cast::<u8>(),
        )),
        zero,
        one,
      );
      let bv = clamp01(
        _mm256_castsi256_ps(endian::load_endian_u32x8::<BE>(
          b.as_ptr().add(x).cast::<u8>(),
        )),
        zero,
        one,
      );
      let rv = clamp01(
        _mm256_castsi256_ps(endian::load_endian_u32x8::<BE>(
          r.as_ptr().add(x).cast::<u8>(),
        )),
        zero,
        one,
      );
      let av = clamp01(
        _mm256_castsi256_ps(endian::load_endian_u32x8::<BE>(
          a.as_ptr().add(x).cast::<u8>(),
        )),
        zero,
        one,
      );
      let gw = narrow_i32x8_to_u16x8(scale_round_i32(gv, scale));
      let bw = narrow_i32x8_to_u16x8(scale_round_i32(bv, scale));
      let rw = narrow_i32x8_to_u16x8(scale_round_i32(rv, scale));
      let aw = narrow_i32x8_to_u16x8(scale_round_i32(av, scale));
      let mut g_buf = [0u16; 8];
      let mut b_buf = [0u16; 8];
      let mut r_buf = [0u16; 8];
      let mut a_buf = [0u16; 8];
      _mm_storeu_si128(g_buf.as_mut_ptr().cast(), gw);
      _mm_storeu_si128(b_buf.as_mut_ptr().cast(), bw);
      _mm_storeu_si128(r_buf.as_mut_ptr().cast(), rw);
      _mm_storeu_si128(a_buf.as_mut_ptr().cast(), aw);
      let base = x * 4;
      for p in 0..8 {
        out[base + p * 4] = r_buf[p];
        out[base + p * 4 + 1] = g_buf[p];
        out[base + p * 4 + 2] = b_buf[p];
        out[base + p * 4 + 3] = a_buf[p];
      }
      x += 8;
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

/// AVX2: planar Gbrapf32 → packed `R, G, B, A` f32 (lossless, source α).
///
/// AVX2 has no 4-channel interleave store; use scalar.
///
/// # Safety
///
/// 1. AVX2 must be available.
/// 2. `g.len()`, `b.len()`, `r.len()`, `a.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "avx2")]
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

/// AVX2 + F16C: planar Gbrapf32 → packed `R, G, B, A` f16 (fused narrow,
/// source α). 8 px / iter.
///
/// # Safety
///
/// 1. AVX2 and F16C must be available.
/// 2. `g.len()`, `b.len()`, `r.len()`, `a.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "avx2,f16c")]
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
    while x + 8 <= width {
      let gv = _mm256_castsi256_ps(endian::load_endian_u32x8::<BE>(
        g.as_ptr().add(x).cast::<u8>(),
      ));
      let bv = _mm256_castsi256_ps(endian::load_endian_u32x8::<BE>(
        b.as_ptr().add(x).cast::<u8>(),
      ));
      let rv = _mm256_castsi256_ps(endian::load_endian_u32x8::<BE>(
        r.as_ptr().add(x).cast::<u8>(),
      ));
      let av = _mm256_castsi256_ps(endian::load_endian_u32x8::<BE>(
        a.as_ptr().add(x).cast::<u8>(),
      ));
      let gh = _mm256_cvtps_ph::<{ _MM_FROUND_TO_NEAREST_INT }>(gv);
      let bh = _mm256_cvtps_ph::<{ _MM_FROUND_TO_NEAREST_INT }>(bv);
      let rh = _mm256_cvtps_ph::<{ _MM_FROUND_TO_NEAREST_INT }>(rv);
      let ah = _mm256_cvtps_ph::<{ _MM_FROUND_TO_NEAREST_INT }>(av);
      let mut g_buf = [0u16; 8];
      let mut b_buf = [0u16; 8];
      let mut r_buf = [0u16; 8];
      let mut a_buf = [0u16; 8];
      _mm_storeu_si128(g_buf.as_mut_ptr().cast(), gh);
      _mm_storeu_si128(b_buf.as_mut_ptr().cast(), bh);
      _mm_storeu_si128(r_buf.as_mut_ptr().cast(), rh);
      _mm_storeu_si128(a_buf.as_mut_ptr().cast(), ah);
      let base = x * 4;
      for p in 0..8 {
        let dst = out.as_mut_ptr().add(base + p * 4);
        *dst.cast::<u16>() = r_buf[p];
        *dst.add(1).cast::<u16>() = g_buf[p];
        *dst.add(2).cast::<u16>() = b_buf[p];
        *dst.add(3).cast::<u16>() = a_buf[p];
      }
      x += 8;
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

/// AVX2 + F16C: planar Gbrpf16 → packed `R, G, B` bytes (widen f16→f32,
/// then convert). 8 px / iter.
///
/// # Safety
///
/// 1. AVX2 and F16C must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `3 * width`.
#[inline]
#[target_feature(enable = "avx2,f16c")]
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
    let zero = _mm256_setzero_ps();
    let one = _mm256_set1_ps(1.0);
    let scale = _mm256_set1_ps(255.0);

    let mut x = 0usize;
    while x + 8 <= width {
      // Load 8 f16 lanes (16 bytes) per plane and widen to f32x8.
      let gv = _mm256_cvtph_ps(endian::load_endian_u16x8::<BE>(
        g.as_ptr().add(x).cast::<u8>(),
      ));
      let bv = _mm256_cvtph_ps(endian::load_endian_u16x8::<BE>(
        b.as_ptr().add(x).cast::<u8>(),
      ));
      let rv = _mm256_cvtph_ps(endian::load_endian_u16x8::<BE>(
        r.as_ptr().add(x).cast::<u8>(),
      ));
      let gc = clamp01(gv, zero, one);
      let bc = clamp01(bv, zero, one);
      let rc = clamp01(rv, zero, one);
      let g8 = narrow_i32x8_to_u8x8(scale_round_i32(gc, scale));
      let b8 = narrow_i32x8_to_u8x8(scale_round_i32(bc, scale));
      let r8 = narrow_i32x8_to_u8x8(scale_round_i32(rc, scale));
      let mut g_buf = [0u8; 16];
      let mut b_buf = [0u8; 16];
      let mut r_buf = [0u8; 16];
      _mm_storeu_si128(g_buf.as_mut_ptr().cast(), g8);
      _mm_storeu_si128(b_buf.as_mut_ptr().cast(), b8);
      _mm_storeu_si128(r_buf.as_mut_ptr().cast(), r8);
      let base = x * 3;
      for p in 0..8 {
        out[base + p * 3] = r_buf[p];
        out[base + p * 3 + 1] = g_buf[p];
        out[base + p * 3 + 2] = b_buf[p];
      }
      x += 8;
    }
    if x < width {
      // Scalar tail: bit-normalize f16 → host-native f32 (via
      // `scalar_f16::widen_f16_be_to_host_f32::<BE>` which `from_be` /
      // `from_le`-loads the source bits BEFORE the f16 → f32 conversion),
      // then route the scalar kernel via `HOST_NATIVE_BE` to avoid double
      // byte-swap.
      let tail = width - x;
      let mut gf = [0.0f32; 8];
      let mut bf = [0.0f32; 8];
      let mut rf = [0.0f32; 8];
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

/// AVX2 + F16C: planar Gbrpf16 → packed `R, G, B, A` bytes (widen f16→f32,
/// α = 0xFF). 8 px / iter.
///
/// # Safety
///
/// 1. AVX2 and F16C must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "avx2,f16c")]
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
    let zero = _mm256_setzero_ps();
    let one = _mm256_set1_ps(1.0);
    let scale = _mm256_set1_ps(255.0);

    let mut x = 0usize;
    while x + 8 <= width {
      let gv = _mm256_cvtph_ps(endian::load_endian_u16x8::<BE>(
        g.as_ptr().add(x).cast::<u8>(),
      ));
      let bv = _mm256_cvtph_ps(endian::load_endian_u16x8::<BE>(
        b.as_ptr().add(x).cast::<u8>(),
      ));
      let rv = _mm256_cvtph_ps(endian::load_endian_u16x8::<BE>(
        r.as_ptr().add(x).cast::<u8>(),
      ));
      let gc = clamp01(gv, zero, one);
      let bc = clamp01(bv, zero, one);
      let rc = clamp01(rv, zero, one);
      let g8 = narrow_i32x8_to_u8x8(scale_round_i32(gc, scale));
      let b8 = narrow_i32x8_to_u8x8(scale_round_i32(bc, scale));
      let r8 = narrow_i32x8_to_u8x8(scale_round_i32(rc, scale));
      let mut g_buf = [0u8; 16];
      let mut b_buf = [0u8; 16];
      let mut r_buf = [0u8; 16];
      _mm_storeu_si128(g_buf.as_mut_ptr().cast(), g8);
      _mm_storeu_si128(b_buf.as_mut_ptr().cast(), b8);
      _mm_storeu_si128(r_buf.as_mut_ptr().cast(), r8);
      let base = x * 4;
      for p in 0..8 {
        out[base + p * 4] = r_buf[p];
        out[base + p * 4 + 1] = g_buf[p];
        out[base + p * 4 + 2] = b_buf[p];
        out[base + p * 4 + 3] = 0xFF;
      }
      x += 8;
    }
    if x < width {
      // Scalar tail: bit-normalize f16 → host-native f32, then route the
      // scalar kernel via `HOST_NATIVE_BE` to avoid double-byte-swap.
      let tail = width - x;
      let mut gf = [0.0f32; 8];
      let mut bf = [0.0f32; 8];
      let mut rf = [0.0f32; 8];
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

/// AVX2 + F16C: planar Gbrpf16 → packed `R, G, B` u16 (widen f16→f32).
/// 8 px / iter.
///
/// # Safety
///
/// 1. AVX2 and F16C must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `3 * width`.
#[inline]
#[target_feature(enable = "avx2,f16c")]
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
    let zero = _mm256_setzero_ps();
    let one = _mm256_set1_ps(1.0);
    let scale = _mm256_set1_ps(65535.0);

    let mut x = 0usize;
    while x + 8 <= width {
      let gv = _mm256_cvtph_ps(endian::load_endian_u16x8::<BE>(
        g.as_ptr().add(x).cast::<u8>(),
      ));
      let bv = _mm256_cvtph_ps(endian::load_endian_u16x8::<BE>(
        b.as_ptr().add(x).cast::<u8>(),
      ));
      let rv = _mm256_cvtph_ps(endian::load_endian_u16x8::<BE>(
        r.as_ptr().add(x).cast::<u8>(),
      ));
      let gc = clamp01(gv, zero, one);
      let bc = clamp01(bv, zero, one);
      let rc = clamp01(rv, zero, one);
      let gw = narrow_i32x8_to_u16x8(scale_round_i32(gc, scale));
      let bw = narrow_i32x8_to_u16x8(scale_round_i32(bc, scale));
      let rw = narrow_i32x8_to_u16x8(scale_round_i32(rc, scale));
      let mut g_buf = [0u16; 8];
      let mut b_buf = [0u16; 8];
      let mut r_buf = [0u16; 8];
      _mm_storeu_si128(g_buf.as_mut_ptr().cast(), gw);
      _mm_storeu_si128(b_buf.as_mut_ptr().cast(), bw);
      _mm_storeu_si128(r_buf.as_mut_ptr().cast(), rw);
      let base = x * 3;
      for p in 0..8 {
        out[base + p * 3] = r_buf[p];
        out[base + p * 3 + 1] = g_buf[p];
        out[base + p * 3 + 2] = b_buf[p];
      }
      x += 8;
    }
    if x < width {
      // Scalar tail: bit-normalize f16 → host-native f32, then route the
      // scalar kernel via `HOST_NATIVE_BE` to avoid double-byte-swap.
      let tail = width - x;
      let mut gf = [0.0f32; 8];
      let mut bf = [0.0f32; 8];
      let mut rf = [0.0f32; 8];
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

/// AVX2 + F16C: planar Gbrpf16 → packed `R, G, B, A` u16 (widen, α = 0xFFFF).
/// 8 px / iter.
///
/// # Safety
///
/// 1. AVX2 and F16C must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "avx2,f16c")]
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
    let zero = _mm256_setzero_ps();
    let one = _mm256_set1_ps(1.0);
    let scale = _mm256_set1_ps(65535.0);

    let mut x = 0usize;
    while x + 8 <= width {
      let gv = _mm256_cvtph_ps(endian::load_endian_u16x8::<BE>(
        g.as_ptr().add(x).cast::<u8>(),
      ));
      let bv = _mm256_cvtph_ps(endian::load_endian_u16x8::<BE>(
        b.as_ptr().add(x).cast::<u8>(),
      ));
      let rv = _mm256_cvtph_ps(endian::load_endian_u16x8::<BE>(
        r.as_ptr().add(x).cast::<u8>(),
      ));
      let gc = clamp01(gv, zero, one);
      let bc = clamp01(bv, zero, one);
      let rc = clamp01(rv, zero, one);
      let gw = narrow_i32x8_to_u16x8(scale_round_i32(gc, scale));
      let bw = narrow_i32x8_to_u16x8(scale_round_i32(bc, scale));
      let rw = narrow_i32x8_to_u16x8(scale_round_i32(rc, scale));
      let mut g_buf = [0u16; 8];
      let mut b_buf = [0u16; 8];
      let mut r_buf = [0u16; 8];
      _mm_storeu_si128(g_buf.as_mut_ptr().cast(), gw);
      _mm_storeu_si128(b_buf.as_mut_ptr().cast(), bw);
      _mm_storeu_si128(r_buf.as_mut_ptr().cast(), rw);
      let base = x * 4;
      for p in 0..8 {
        out[base + p * 4] = r_buf[p];
        out[base + p * 4 + 1] = g_buf[p];
        out[base + p * 4 + 2] = b_buf[p];
        out[base + p * 4 + 3] = 0xFFFF;
      }
      x += 8;
    }
    if x < width {
      // Scalar tail: bit-normalize f16 → host-native f32, then route the
      // scalar kernel via `HOST_NATIVE_BE` to avoid double-byte-swap.
      let tail = width - x;
      let mut gf = [0.0f32; 8];
      let mut bf = [0.0f32; 8];
      let mut rf = [0.0f32; 8];
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

/// AVX2 + F16C: planar Gbrpf16 → packed `R, G, B` f32 (lossless widen).
/// 8 px / iter.
///
/// # Safety
///
/// 1. AVX2 and F16C must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `3 * width`.
#[inline]
#[target_feature(enable = "avx2,f16c")]
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
    while x + 8 <= width {
      let gv = _mm256_cvtph_ps(endian::load_endian_u16x8::<BE>(
        g.as_ptr().add(x).cast::<u8>(),
      ));
      let bv = _mm256_cvtph_ps(endian::load_endian_u16x8::<BE>(
        b.as_ptr().add(x).cast::<u8>(),
      ));
      let rv = _mm256_cvtph_ps(endian::load_endian_u16x8::<BE>(
        r.as_ptr().add(x).cast::<u8>(),
      ));
      // No 3-channel interleave intrinsic in AVX2 — scatter via scalar loop.
      let mut gf = [0.0f32; 8];
      let mut bf = [0.0f32; 8];
      let mut rf = [0.0f32; 8];
      _mm256_storeu_ps(gf.as_mut_ptr(), gv);
      _mm256_storeu_ps(bf.as_mut_ptr(), bv);
      _mm256_storeu_ps(rf.as_mut_ptr(), rv);
      let base = x * 3;
      for p in 0..8 {
        out[base + p * 3] = rf[p];
        out[base + p * 3 + 1] = gf[p];
        out[base + p * 3 + 2] = bf[p];
      }
      x += 8;
    }
    if x < width {
      // Scalar tail: widen f16 → host-native f32 (normalize source bits via
      // `from_be` / `from_le` BEFORE the f16 → f32 conversion), then route the
      // scalar kernel via `HOST_NATIVE_BE` to avoid double-byte-swap.
      let tail = width - x;
      let mut gf = [0.0f32; 8];
      let mut bf = [0.0f32; 8];
      let mut rf = [0.0f32; 8];
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

/// AVX2 + F16C: planar Gbrpf16 → packed `R, G, B, A` f32 (lossless, α = 1.0).
/// 8 px / iter.
///
/// # Safety
///
/// 1. AVX2 and F16C must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "avx2,f16c")]
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
    while x + 8 <= width {
      let gv = _mm256_cvtph_ps(endian::load_endian_u16x8::<BE>(
        g.as_ptr().add(x).cast::<u8>(),
      ));
      let bv = _mm256_cvtph_ps(endian::load_endian_u16x8::<BE>(
        b.as_ptr().add(x).cast::<u8>(),
      ));
      let rv = _mm256_cvtph_ps(endian::load_endian_u16x8::<BE>(
        r.as_ptr().add(x).cast::<u8>(),
      ));
      let mut gf = [0.0f32; 8];
      let mut bf = [0.0f32; 8];
      let mut rf = [0.0f32; 8];
      _mm256_storeu_ps(gf.as_mut_ptr(), gv);
      _mm256_storeu_ps(bf.as_mut_ptr(), bv);
      _mm256_storeu_ps(rf.as_mut_ptr(), rv);
      let base = x * 4;
      for p in 0..8 {
        out[base + p * 4] = rf[p];
        out[base + p * 4 + 1] = gf[p];
        out[base + p * 4 + 2] = bf[p];
        out[base + p * 4 + 3] = 1.0;
      }
      x += 8;
    }
    if x < width {
      // Scalar tail: widen f16 → host-native f32 (normalize source bits via
      // `from_be` / `from_le` BEFORE the f16 → f32 conversion), then route the
      // scalar kernel via `HOST_NATIVE_BE` to avoid double-byte-swap.
      let tail = width - x;
      let mut gf = [0.0f32; 8];
      let mut bf = [0.0f32; 8];
      let mut rf = [0.0f32; 8];
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

/// AVX2: planar Gbrpf16 → packed `R, G, B` f16 (lossless — f16 treated as u16).
///
/// No F16C gate needed: f16 planes are bit-copied as opaque u16 lanes.
/// 8 px / iter via 16-byte unaligned loads + manual scatter (no vst3
/// equivalent in AVX2).
///
/// # Safety
///
/// 1. AVX2 must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `3 * width`.
#[inline]
#[target_feature(enable = "avx2")]
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
    while x + 8 <= width {
      // Load 8 x u16 (16 bytes) per plane.
      let gu = endian::load_endian_u16x8::<BE>(g.as_ptr().add(x).cast::<u8>());
      let bu = endian::load_endian_u16x8::<BE>(b.as_ptr().add(x).cast::<u8>());
      let ru = endian::load_endian_u16x8::<BE>(r.as_ptr().add(x).cast::<u8>());
      let mut g_buf = [0u16; 8];
      let mut b_buf = [0u16; 8];
      let mut r_buf = [0u16; 8];
      _mm_storeu_si128(g_buf.as_mut_ptr().cast(), gu);
      _mm_storeu_si128(b_buf.as_mut_ptr().cast(), bu);
      _mm_storeu_si128(r_buf.as_mut_ptr().cast(), ru);
      let base = x * 3;
      for p in 0..8 {
        let dst = out.as_mut_ptr().add(base + p * 3);
        *dst.cast::<u16>() = r_buf[p];
        *dst.add(1).cast::<u16>() = g_buf[p];
        *dst.add(2).cast::<u16>() = b_buf[p];
      }
      x += 8;
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

/// AVX2: planar Gbrpf16 → packed `R, G, B, A` f16 (lossless,
/// α = f16(1.0) = 0x3C00).
///
/// No F16C gate needed.
///
/// # Safety
///
/// 1. AVX2 must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "avx2")]
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
    while x + 8 <= width {
      let gu = endian::load_endian_u16x8::<BE>(g.as_ptr().add(x).cast::<u8>());
      let bu = endian::load_endian_u16x8::<BE>(b.as_ptr().add(x).cast::<u8>());
      let ru = endian::load_endian_u16x8::<BE>(r.as_ptr().add(x).cast::<u8>());
      let mut g_buf = [0u16; 8];
      let mut b_buf = [0u16; 8];
      let mut r_buf = [0u16; 8];
      _mm_storeu_si128(g_buf.as_mut_ptr().cast(), gu);
      _mm_storeu_si128(b_buf.as_mut_ptr().cast(), bu);
      _mm_storeu_si128(r_buf.as_mut_ptr().cast(), ru);
      let base = x * 4;
      for p in 0..8 {
        let dst = out.as_mut_ptr().add(base + p * 4);
        *dst.cast::<u16>() = r_buf[p];
        *dst.add(1).cast::<u16>() = g_buf[p];
        *dst.add(2).cast::<u16>() = b_buf[p];
        *dst.add(3).cast::<u16>() = 0x3C00u16; // f16(1.0)
      }
      x += 8;
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

/// AVX2 + F16C: planar Gbrpf16 → u8 luma (widen + staged via RGB scratch).
///
/// # Safety
///
/// 1. AVX2 and F16C must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `width`.
#[inline]
#[target_feature(enable = "avx2,f16c")]
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

/// AVX2 + F16C: planar Gbrpf16 → u16 luma (widen + staged via RGB scratch).
///
/// # Safety
///
/// 1. AVX2 and F16C must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `width`.
#[inline]
#[target_feature(enable = "avx2,f16c")]
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

/// AVX2 + F16C: planar Gbrpf16 → planar HSV bytes (widen + staged via RGB).
///
/// # Safety
///
/// 1. AVX2 and F16C must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `h_out.len()`, `s_out.len()`, `v_out.len()` ≥ `width`.
#[inline]
#[target_feature(enable = "avx2,f16c")]
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

/// AVX2 + F16C: planar Gbrapf16 → packed `R, G, B, A` bytes (widen, source α).
/// 8 px / iter.
///
/// # Safety
///
/// 1. AVX2 and F16C must be available.
/// 2. `g.len()`, `b.len()`, `r.len()`, `a.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "avx2,f16c")]
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
    let zero = _mm256_setzero_ps();
    let one = _mm256_set1_ps(1.0);
    let scale = _mm256_set1_ps(255.0);

    let mut x = 0usize;
    while x + 8 <= width {
      let gv = _mm256_cvtph_ps(endian::load_endian_u16x8::<BE>(
        g.as_ptr().add(x).cast::<u8>(),
      ));
      let bv = _mm256_cvtph_ps(endian::load_endian_u16x8::<BE>(
        b.as_ptr().add(x).cast::<u8>(),
      ));
      let rv = _mm256_cvtph_ps(endian::load_endian_u16x8::<BE>(
        r.as_ptr().add(x).cast::<u8>(),
      ));
      let av = _mm256_cvtph_ps(endian::load_endian_u16x8::<BE>(
        a.as_ptr().add(x).cast::<u8>(),
      ));
      let gc = clamp01(gv, zero, one);
      let bc = clamp01(bv, zero, one);
      let rc = clamp01(rv, zero, one);
      let ac = clamp01(av, zero, one);
      let g8 = narrow_i32x8_to_u8x8(scale_round_i32(gc, scale));
      let b8 = narrow_i32x8_to_u8x8(scale_round_i32(bc, scale));
      let r8 = narrow_i32x8_to_u8x8(scale_round_i32(rc, scale));
      let a8 = narrow_i32x8_to_u8x8(scale_round_i32(ac, scale));
      let mut g_buf = [0u8; 16];
      let mut b_buf = [0u8; 16];
      let mut r_buf = [0u8; 16];
      let mut a_buf = [0u8; 16];
      _mm_storeu_si128(g_buf.as_mut_ptr().cast(), g8);
      _mm_storeu_si128(b_buf.as_mut_ptr().cast(), b8);
      _mm_storeu_si128(r_buf.as_mut_ptr().cast(), r8);
      _mm_storeu_si128(a_buf.as_mut_ptr().cast(), a8);
      let base = x * 4;
      for p in 0..8 {
        out[base + p * 4] = r_buf[p];
        out[base + p * 4 + 1] = g_buf[p];
        out[base + p * 4 + 2] = b_buf[p];
        out[base + p * 4 + 3] = a_buf[p];
      }
      x += 8;
    }
    if x < width {
      // Scalar tail: bit-normalize f16 → host-native f32, then route the
      // scalar kernel via `HOST_NATIVE_BE` to avoid double-byte-swap.
      let tail = width - x;
      let mut gf = [0.0f32; 8];
      let mut bf = [0.0f32; 8];
      let mut rf = [0.0f32; 8];
      let mut af = [0.0f32; 8];
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

/// AVX2 + F16C: planar Gbrapf16 → packed `R, G, B, A` u16 (widen, source α).
/// 8 px / iter.
///
/// # Safety
///
/// 1. AVX2 and F16C must be available.
/// 2. `g.len()`, `b.len()`, `r.len()`, `a.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "avx2,f16c")]
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
    let zero = _mm256_setzero_ps();
    let one = _mm256_set1_ps(1.0);
    let scale = _mm256_set1_ps(65535.0);

    let mut x = 0usize;
    while x + 8 <= width {
      let gv = _mm256_cvtph_ps(endian::load_endian_u16x8::<BE>(
        g.as_ptr().add(x).cast::<u8>(),
      ));
      let bv = _mm256_cvtph_ps(endian::load_endian_u16x8::<BE>(
        b.as_ptr().add(x).cast::<u8>(),
      ));
      let rv = _mm256_cvtph_ps(endian::load_endian_u16x8::<BE>(
        r.as_ptr().add(x).cast::<u8>(),
      ));
      let av = _mm256_cvtph_ps(endian::load_endian_u16x8::<BE>(
        a.as_ptr().add(x).cast::<u8>(),
      ));
      let gc = clamp01(gv, zero, one);
      let bc = clamp01(bv, zero, one);
      let rc = clamp01(rv, zero, one);
      let ac = clamp01(av, zero, one);
      let gw = narrow_i32x8_to_u16x8(scale_round_i32(gc, scale));
      let bw = narrow_i32x8_to_u16x8(scale_round_i32(bc, scale));
      let rw = narrow_i32x8_to_u16x8(scale_round_i32(rc, scale));
      let aw = narrow_i32x8_to_u16x8(scale_round_i32(ac, scale));
      let mut g_buf = [0u16; 8];
      let mut b_buf = [0u16; 8];
      let mut r_buf = [0u16; 8];
      let mut a_buf = [0u16; 8];
      _mm_storeu_si128(g_buf.as_mut_ptr().cast(), gw);
      _mm_storeu_si128(b_buf.as_mut_ptr().cast(), bw);
      _mm_storeu_si128(r_buf.as_mut_ptr().cast(), rw);
      _mm_storeu_si128(a_buf.as_mut_ptr().cast(), aw);
      let base = x * 4;
      for p in 0..8 {
        out[base + p * 4] = r_buf[p];
        out[base + p * 4 + 1] = g_buf[p];
        out[base + p * 4 + 2] = b_buf[p];
        out[base + p * 4 + 3] = a_buf[p];
      }
      x += 8;
    }
    if x < width {
      // Scalar tail: bit-normalize f16 → host-native f32, then route the
      // scalar kernel via `HOST_NATIVE_BE` to avoid double-byte-swap.
      let tail = width - x;
      let mut gf = [0.0f32; 8];
      let mut bf = [0.0f32; 8];
      let mut rf = [0.0f32; 8];
      let mut af = [0.0f32; 8];
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

/// AVX2 + F16C: planar Gbrapf16 → packed `R, G, B, A` f32 (lossless,
/// source α). 8 px / iter.
///
/// # Safety
///
/// 1. AVX2 and F16C must be available.
/// 2. `g.len()`, `b.len()`, `r.len()`, `a.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "avx2,f16c")]
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
    while x + 8 <= width {
      let gv = _mm256_cvtph_ps(endian::load_endian_u16x8::<BE>(
        g.as_ptr().add(x).cast::<u8>(),
      ));
      let bv = _mm256_cvtph_ps(endian::load_endian_u16x8::<BE>(
        b.as_ptr().add(x).cast::<u8>(),
      ));
      let rv = _mm256_cvtph_ps(endian::load_endian_u16x8::<BE>(
        r.as_ptr().add(x).cast::<u8>(),
      ));
      let av = _mm256_cvtph_ps(endian::load_endian_u16x8::<BE>(
        a.as_ptr().add(x).cast::<u8>(),
      ));
      let mut gf = [0.0f32; 8];
      let mut bf = [0.0f32; 8];
      let mut rf = [0.0f32; 8];
      let mut af = [0.0f32; 8];
      _mm256_storeu_ps(gf.as_mut_ptr(), gv);
      _mm256_storeu_ps(bf.as_mut_ptr(), bv);
      _mm256_storeu_ps(rf.as_mut_ptr(), rv);
      _mm256_storeu_ps(af.as_mut_ptr(), av);
      let base = x * 4;
      for p in 0..8 {
        out[base + p * 4] = rf[p];
        out[base + p * 4 + 1] = gf[p];
        out[base + p * 4 + 2] = bf[p];
        out[base + p * 4 + 3] = af[p];
      }
      x += 8;
    }
    if x < width {
      // Scalar tail: widen f16 → host-native f32 (normalize source bits via
      // `from_be` / `from_le` BEFORE the f16 → f32 conversion), then route the
      // scalar kernel via `HOST_NATIVE_BE` to avoid double-byte-swap.
      let tail = width - x;
      let mut gf = [0.0f32; 8];
      let mut bf = [0.0f32; 8];
      let mut rf = [0.0f32; 8];
      let mut af = [0.0f32; 8];
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

/// AVX2: planar Gbrapf16 → packed `R, G, B, A` f16 (lossless, source α).
///
/// No F16C gate needed: f16 planes are bit-copied as opaque u16.
///
/// # Safety
///
/// 1. AVX2 must be available.
/// 2. `g.len()`, `b.len()`, `r.len()`, `a.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "avx2")]
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
    while x + 8 <= width {
      let gu = endian::load_endian_u16x8::<BE>(g.as_ptr().add(x).cast::<u8>());
      let bu = endian::load_endian_u16x8::<BE>(b.as_ptr().add(x).cast::<u8>());
      let ru = endian::load_endian_u16x8::<BE>(r.as_ptr().add(x).cast::<u8>());
      let au = endian::load_endian_u16x8::<BE>(a.as_ptr().add(x).cast::<u8>());
      let mut g_buf = [0u16; 8];
      let mut b_buf = [0u16; 8];
      let mut r_buf = [0u16; 8];
      let mut a_buf = [0u16; 8];
      _mm_storeu_si128(g_buf.as_mut_ptr().cast(), gu);
      _mm_storeu_si128(b_buf.as_mut_ptr().cast(), bu);
      _mm_storeu_si128(r_buf.as_mut_ptr().cast(), ru);
      _mm_storeu_si128(a_buf.as_mut_ptr().cast(), au);
      let base = x * 4;
      for p in 0..8 {
        let dst = out.as_mut_ptr().add(base + p * 4);
        *dst.cast::<u16>() = r_buf[p];
        *dst.add(1).cast::<u16>() = g_buf[p];
        *dst.add(2).cast::<u16>() = b_buf[p];
        *dst.add(3).cast::<u16>() = a_buf[p];
      }
      x += 8;
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
