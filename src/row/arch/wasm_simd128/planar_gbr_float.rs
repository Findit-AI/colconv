//! wasm-simd128 SIMD kernels for planar GBR float sources (Tier 10).
//!
//! 4 pixels / iteration via `f32x4` (one G, B, R [+ A] register each).
//!
//! # Rounding (f32 → u8 / u16)
//!
//! `f32x4_add(scaled, half)` then `i32x4_trunc_sat_f32x4` (truncate toward
//! zero). This is the round-half-up contract shared with the scalar / NEON /
//! SSE4.1 / AVX2 / AVX-512 kernels — consistent with the Grayf32 path.
//!
//! **Do NOT use `f32x4_nearest` for integer narrowing** — that gives
//! banker's rounding, not round-half-up.
//!
//! # Rounding (f32 → f16)
//!
//! IEEE-754 round-to-nearest-even via scalar `half::f16::from_f32` per
//! element (wasm-simd128 has no native f16 widening / narrowing intrinsic).
//!
//! # f16 lossless interleave
//!
//! Treat f16 lanes as opaque `u16` — no arithmetic. Scalar loop writes each
//! lane individually via the `half::f16` API; no unsafe pointer casts.

#![cfg_attr(not(feature = "std"), allow(dead_code))]

use core::arch::wasm32::*;

use crate::{
  KernelMatrix,
  row::{
    arch::wasm_simd128::endian,
    scalar::{planar_gbr_f16 as scalar_f16, planar_gbr_float as scalar},
  },
};

/// `BE` value that makes the downstream `gbrpf32_to_*` kernels (both the
/// SIMD-aligned head and the scalar tail in this file's f16 row kernels)
/// treat their `f32` scratch input as **host-native** (no `from_be` /
/// `from_le` byte-swap). After we widen f16 → f32 via
/// [`scalar_f16::widen_f16_be_to_host_f32`] (which normalizes the source
/// f16 bits per the source `BE` and produces host-native f32), the resulting
/// scratch must be routed via `HOST_NATIVE_BE` so the downstream kernel's
/// `from_le` / `from_be` loaders no-op the swap.
///
/// **Note:** wasm-simd128 has no native f16 widening intrinsic, so the
/// "SIMD-aligned" body of the f16 row kernels in this file uses the same
/// scalar widening pattern as the tail. Both code paths therefore need the
/// same `HOST_NATIVE_BE` routing.
const HOST_NATIVE_BE: bool = cfg!(target_endian = "big");

// ---- shared helpers ----------------------------------------------------------

/// Clamp a `f32x4` to `[0.0, 1.0]`.
#[inline(always)]
fn clamp01(v: v128, zero: v128, one: v128) -> v128 {
  f32x4_min(f32x4_max(v, zero), one)
}

/// Scale, add 0.5, truncate-toward-zero → `i32x4`. Round-half-up.
#[inline(always)]
fn scale_round_i32(v: v128, scale: v128, half: v128) -> v128 {
  i32x4_trunc_sat_f32x4(f32x4_add(f32x4_mul(v, scale), half))
}

// ---- Gbrpf32 → u8 RGB -------------------------------------------------------

/// wasm-simd128: planar Gbrpf32 → packed `R, G, B` bytes. 4 px / iter.
///
/// Round-half-up: `+ 0.5` then `i32x4_trunc_sat_f32x4`.
///
/// # Safety
///
/// 1. simd128 must be available (compile-time `target_feature`).
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `3 * width`.
#[inline]
#[target_feature(enable = "simd128")]
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

  let zero = f32x4_splat(0.0);
  let one = f32x4_splat(1.0);
  let scale = f32x4_splat(255.0);
  let half = f32x4_splat(0.5);

  let mut x = 0usize;
  while x + 4 <= width {
    unsafe {
      let gv = clamp01(
        endian::load_endian_u32x4::<BE>(g.as_ptr().add(x).cast::<u8>()),
        zero,
        one,
      );
      let bv = clamp01(
        endian::load_endian_u32x4::<BE>(b.as_ptr().add(x).cast::<u8>()),
        zero,
        one,
      );
      let rv = clamp01(
        endian::load_endian_u32x4::<BE>(r.as_ptr().add(x).cast::<u8>()),
        zero,
        one,
      );
      let gi = scale_round_i32(gv, scale, half);
      let bi = scale_round_i32(bv, scale, half);
      let ri = scale_round_i32(rv, scale, half);
      // Narrow i32x4 → i16x8 (low 4 lanes valid), then → u8x16 (low 4 valid).
      let g16 = i16x8_narrow_i32x4(gi, gi);
      let b16 = i16x8_narrow_i32x4(bi, bi);
      let r16 = i16x8_narrow_i32x4(ri, ri);
      let g8 = u8x16_narrow_i16x8(g16, g16);
      let b8 = u8x16_narrow_i16x8(b16, b16);
      let r8 = u8x16_narrow_i16x8(r16, r16);
      let mut g_buf = [0u8; 16];
      let mut b_buf = [0u8; 16];
      let mut r_buf = [0u8; 16];
      v128_store(g_buf.as_mut_ptr().cast(), g8);
      v128_store(b_buf.as_mut_ptr().cast(), b8);
      v128_store(r_buf.as_mut_ptr().cast(), r8);
      let base = x * 3;
      for p in 0..4 {
        out[base + p * 3] = r_buf[p];
        out[base + p * 3 + 1] = g_buf[p];
        out[base + p * 3 + 2] = b_buf[p];
      }
    }
    x += 4;
  }
  if x < width {
    scalar::gbrpf32_to_rgb_row::<BE>(&g[x..], &b[x..], &r[x..], &mut out[x * 3..], width - x);
  }
}

// ---- Gbrpf32 → u8 RGBA (opaque α) ------------------------------------------

/// wasm-simd128: planar Gbrpf32 → packed `R, G, B, A` bytes (α = 0xFF).
/// 4 px / iter.
///
/// # Safety
///
/// 1. simd128 must be available (compile-time `target_feature`).
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "simd128")]
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

  let zero = f32x4_splat(0.0);
  let one = f32x4_splat(1.0);
  let scale = f32x4_splat(255.0);
  let half = f32x4_splat(0.5);

  let mut x = 0usize;
  while x + 4 <= width {
    unsafe {
      let gv = clamp01(
        endian::load_endian_u32x4::<BE>(g.as_ptr().add(x).cast::<u8>()),
        zero,
        one,
      );
      let bv = clamp01(
        endian::load_endian_u32x4::<BE>(b.as_ptr().add(x).cast::<u8>()),
        zero,
        one,
      );
      let rv = clamp01(
        endian::load_endian_u32x4::<BE>(r.as_ptr().add(x).cast::<u8>()),
        zero,
        one,
      );
      let gi = scale_round_i32(gv, scale, half);
      let bi = scale_round_i32(bv, scale, half);
      let ri = scale_round_i32(rv, scale, half);
      let g16 = i16x8_narrow_i32x4(gi, gi);
      let b16 = i16x8_narrow_i32x4(bi, bi);
      let r16 = i16x8_narrow_i32x4(ri, ri);
      let g8 = u8x16_narrow_i16x8(g16, g16);
      let b8 = u8x16_narrow_i16x8(b16, b16);
      let r8 = u8x16_narrow_i16x8(r16, r16);
      let mut g_buf = [0u8; 16];
      let mut b_buf = [0u8; 16];
      let mut r_buf = [0u8; 16];
      v128_store(g_buf.as_mut_ptr().cast(), g8);
      v128_store(b_buf.as_mut_ptr().cast(), b8);
      v128_store(r_buf.as_mut_ptr().cast(), r8);
      let base = x * 4;
      for p in 0..4 {
        out[base + p * 4] = r_buf[p];
        out[base + p * 4 + 1] = g_buf[p];
        out[base + p * 4 + 2] = b_buf[p];
        out[base + p * 4 + 3] = 0xFF;
      }
    }
    x += 4;
  }
  if x < width {
    scalar::gbrpf32_to_rgba_row::<BE>(&g[x..], &b[x..], &r[x..], &mut out[x * 4..], width - x);
  }
}

// ---- Gbrpf32 → u16 RGB ------------------------------------------------------

/// wasm-simd128: planar Gbrpf32 → packed `R, G, B` u16. 4 px / iter.
///
/// # Safety
///
/// 1. simd128 must be available (compile-time `target_feature`).
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `3 * width`.
#[inline]
#[target_feature(enable = "simd128")]
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

  let zero = f32x4_splat(0.0);
  let one = f32x4_splat(1.0);
  let scale = f32x4_splat(65535.0);
  let half = f32x4_splat(0.5);

  let mut x = 0usize;
  while x + 4 <= width {
    unsafe {
      let gv = clamp01(
        endian::load_endian_u32x4::<BE>(g.as_ptr().add(x).cast::<u8>()),
        zero,
        one,
      );
      let bv = clamp01(
        endian::load_endian_u32x4::<BE>(b.as_ptr().add(x).cast::<u8>()),
        zero,
        one,
      );
      let rv = clamp01(
        endian::load_endian_u32x4::<BE>(r.as_ptr().add(x).cast::<u8>()),
        zero,
        one,
      );
      let gi = scale_round_i32(gv, scale, half);
      let bi = scale_round_i32(bv, scale, half);
      let ri = scale_round_i32(rv, scale, half);
      // i32x4 → u16x8 saturating narrow (4 valid lanes each).
      let gw = u16x8_narrow_i32x4(gi, gi);
      let bw = u16x8_narrow_i32x4(bi, bi);
      let rw = u16x8_narrow_i32x4(ri, ri);
      let mut g_buf = [0u16; 8];
      let mut b_buf = [0u16; 8];
      let mut r_buf = [0u16; 8];
      v128_store(g_buf.as_mut_ptr().cast(), gw);
      v128_store(b_buf.as_mut_ptr().cast(), bw);
      v128_store(r_buf.as_mut_ptr().cast(), rw);
      let base = x * 3;
      for p in 0..4 {
        out[base + p * 3] = r_buf[p];
        out[base + p * 3 + 1] = g_buf[p];
        out[base + p * 3 + 2] = b_buf[p];
      }
    }
    x += 4;
  }
  if x < width {
    scalar::gbrpf32_to_rgb_u16_row::<BE>(&g[x..], &b[x..], &r[x..], &mut out[x * 3..], width - x);
  }
}

// ---- Gbrpf32 → u16 RGBA (opaque α) -----------------------------------------

/// wasm-simd128: planar Gbrpf32 → packed `R, G, B, A` u16 (α = 0xFFFF).
/// 4 px / iter.
///
/// # Safety
///
/// 1. simd128 must be available (compile-time `target_feature`).
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "simd128")]
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

  let zero = f32x4_splat(0.0);
  let one = f32x4_splat(1.0);
  let scale = f32x4_splat(65535.0);
  let half = f32x4_splat(0.5);

  let mut x = 0usize;
  while x + 4 <= width {
    unsafe {
      let gv = clamp01(
        endian::load_endian_u32x4::<BE>(g.as_ptr().add(x).cast::<u8>()),
        zero,
        one,
      );
      let bv = clamp01(
        endian::load_endian_u32x4::<BE>(b.as_ptr().add(x).cast::<u8>()),
        zero,
        one,
      );
      let rv = clamp01(
        endian::load_endian_u32x4::<BE>(r.as_ptr().add(x).cast::<u8>()),
        zero,
        one,
      );
      let gi = scale_round_i32(gv, scale, half);
      let bi = scale_round_i32(bv, scale, half);
      let ri = scale_round_i32(rv, scale, half);
      let gw = u16x8_narrow_i32x4(gi, gi);
      let bw = u16x8_narrow_i32x4(bi, bi);
      let rw = u16x8_narrow_i32x4(ri, ri);
      let mut g_buf = [0u16; 8];
      let mut b_buf = [0u16; 8];
      let mut r_buf = [0u16; 8];
      v128_store(g_buf.as_mut_ptr().cast(), gw);
      v128_store(b_buf.as_mut_ptr().cast(), bw);
      v128_store(r_buf.as_mut_ptr().cast(), rw);
      let base = x * 4;
      for p in 0..4 {
        out[base + p * 4] = r_buf[p];
        out[base + p * 4 + 1] = g_buf[p];
        out[base + p * 4 + 2] = b_buf[p];
        out[base + p * 4 + 3] = 0xFFFF;
      }
    }
    x += 4;
  }
  if x < width {
    scalar::gbrpf32_to_rgba_u16_row::<BE>(&g[x..], &b[x..], &r[x..], &mut out[x * 4..], width - x);
  }
}

// ---- Gbrpf32 → f32 RGB (lossless) ------------------------------------------

/// wasm-simd128: planar Gbrpf32 → packed `R, G, B` f32 (lossless interleave).
///
/// wasm-simd128 has no 3-channel interleave store; delegates to scalar
/// (compiler auto-vectorises the per-element copy).
///
/// # Safety
///
/// 1. simd128 must be available (compile-time `target_feature`).
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `3 * width`.
#[inline]
#[target_feature(enable = "simd128")]
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

// ---- Gbrpf32 → f32 RGBA (lossless, α = 1.0) --------------------------------

/// wasm-simd128: planar Gbrpf32 → packed `R, G, B, A` f32 (lossless, α = 1.0).
///
/// Delegates to scalar (no 4-channel interleave intrinsic).
///
/// # Safety
///
/// 1. simd128 must be available (compile-time `target_feature`).
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "simd128")]
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

// ---- Gbrpf32 → f16 RGB (scalar narrow) ---------------------------------------

/// wasm-simd128: planar Gbrpf32 → packed `R, G, B` f16.
///
/// wasm-simd128 has no native f16 narrowing intrinsic. Widens f32 planes to
/// 4-element scratch, then calls the SIMD u8-output kernel for the integer
/// path. For f16 output, uses scalar `half::f16::from_f32` per element.
///
/// # Safety
///
/// 1. simd128 must be available (compile-time `target_feature`).
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `3 * width`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn gbrpf32_to_rgb_f16_row<const BE: bool>(
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

  // Scalar narrow: IEEE-754 round-to-nearest-even via half::f16::from_f32.
  scalar::gbrpf32_to_rgb_f16_row::<BE>(g, b, r, out, width);
}

// ---- Gbrpf32 → f16 RGBA (scalar narrow, α = f16(1.0)) ----------------------

/// wasm-simd128: planar Gbrpf32 → packed `R, G, B, A` f16 (α = f16(1.0)).
///
/// # Safety
///
/// 1. simd128 must be available (compile-time `target_feature`).
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn gbrpf32_to_rgba_f16_row<const BE: bool>(
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

  scalar::gbrpf32_to_rgba_f16_row::<BE>(g, b, r, out, width);
}

// ---- Gbrpf32 → u8 luma (staged via RGB scratch) -----------------------------

/// wasm-simd128: planar Gbrpf32 → u8 luma (staged via SIMD RGB kernel).
///
/// # Safety
///
/// 1. simd128 must be available (compile-time `target_feature`).
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `width`.
#[inline]
#[target_feature(enable = "simd128")]
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

  const CHUNK: usize = 16;
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

// ---- Gbrpf32 → u16 luma (staged via RGB scratch) ----------------------------

/// wasm-simd128: planar Gbrpf32 → u16 luma (staged via SIMD RGB kernel).
///
/// # Safety
///
/// 1. simd128 must be available (compile-time `target_feature`).
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `width`.
#[inline]
#[target_feature(enable = "simd128")]
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

  const CHUNK: usize = 16;
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

// ---- Gbrpf32 → HSV (staged via RGB scratch) ----------------------------------

/// wasm-simd128: planar Gbrpf32 → planar HSV bytes (staged via SIMD RGB kernel).
///
/// # Safety
///
/// 1. simd128 must be available (compile-time `target_feature`).
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `h_out.len()`, `s_out.len()`, `v_out.len()` ≥ `width`.
#[inline]
#[target_feature(enable = "simd128")]
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

  const CHUNK: usize = 16;
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

/// wasm-simd128: planar Gbrapf32 → packed `R, G, B, A` bytes (source α).
/// 4 px / iter.
///
/// # Safety
///
/// 1. simd128 must be available (compile-time `target_feature`).
/// 2. `g.len()`, `b.len()`, `r.len()`, `a.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "simd128")]
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

  let zero = f32x4_splat(0.0);
  let one = f32x4_splat(1.0);
  let scale = f32x4_splat(255.0);
  let half = f32x4_splat(0.5);

  let mut x = 0usize;
  while x + 4 <= width {
    unsafe {
      let gv = clamp01(
        endian::load_endian_u32x4::<BE>(g.as_ptr().add(x).cast::<u8>()),
        zero,
        one,
      );
      let bv = clamp01(
        endian::load_endian_u32x4::<BE>(b.as_ptr().add(x).cast::<u8>()),
        zero,
        one,
      );
      let rv = clamp01(
        endian::load_endian_u32x4::<BE>(r.as_ptr().add(x).cast::<u8>()),
        zero,
        one,
      );
      let av = clamp01(
        endian::load_endian_u32x4::<BE>(a.as_ptr().add(x).cast::<u8>()),
        zero,
        one,
      );
      let gi = scale_round_i32(gv, scale, half);
      let bi = scale_round_i32(bv, scale, half);
      let ri = scale_round_i32(rv, scale, half);
      let ai = scale_round_i32(av, scale, half);
      let g16 = i16x8_narrow_i32x4(gi, gi);
      let b16 = i16x8_narrow_i32x4(bi, bi);
      let r16 = i16x8_narrow_i32x4(ri, ri);
      let a16 = i16x8_narrow_i32x4(ai, ai);
      let g8 = u8x16_narrow_i16x8(g16, g16);
      let b8 = u8x16_narrow_i16x8(b16, b16);
      let r8 = u8x16_narrow_i16x8(r16, r16);
      let a8 = u8x16_narrow_i16x8(a16, a16);
      let mut g_buf = [0u8; 16];
      let mut b_buf = [0u8; 16];
      let mut r_buf = [0u8; 16];
      let mut a_buf = [0u8; 16];
      v128_store(g_buf.as_mut_ptr().cast(), g8);
      v128_store(b_buf.as_mut_ptr().cast(), b8);
      v128_store(r_buf.as_mut_ptr().cast(), r8);
      v128_store(a_buf.as_mut_ptr().cast(), a8);
      let base = x * 4;
      for p in 0..4 {
        out[base + p * 4] = r_buf[p];
        out[base + p * 4 + 1] = g_buf[p];
        out[base + p * 4 + 2] = b_buf[p];
        out[base + p * 4 + 3] = a_buf[p];
      }
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

// ---- Gbrapf32 → u16 RGBA (source α) ----------------------------------------

/// wasm-simd128: planar Gbrapf32 → packed `R, G, B, A` u16 (source α).
/// 4 px / iter.
///
/// # Safety
///
/// 1. simd128 must be available (compile-time `target_feature`).
/// 2. `g.len()`, `b.len()`, `r.len()`, `a.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "simd128")]
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

  let zero = f32x4_splat(0.0);
  let one = f32x4_splat(1.0);
  let scale = f32x4_splat(65535.0);
  let half = f32x4_splat(0.5);

  let mut x = 0usize;
  while x + 4 <= width {
    unsafe {
      let gv = clamp01(
        endian::load_endian_u32x4::<BE>(g.as_ptr().add(x).cast::<u8>()),
        zero,
        one,
      );
      let bv = clamp01(
        endian::load_endian_u32x4::<BE>(b.as_ptr().add(x).cast::<u8>()),
        zero,
        one,
      );
      let rv = clamp01(
        endian::load_endian_u32x4::<BE>(r.as_ptr().add(x).cast::<u8>()),
        zero,
        one,
      );
      let av = clamp01(
        endian::load_endian_u32x4::<BE>(a.as_ptr().add(x).cast::<u8>()),
        zero,
        one,
      );
      let gi = scale_round_i32(gv, scale, half);
      let bi = scale_round_i32(bv, scale, half);
      let ri = scale_round_i32(rv, scale, half);
      let ai = scale_round_i32(av, scale, half);
      let gw = u16x8_narrow_i32x4(gi, gi);
      let bw = u16x8_narrow_i32x4(bi, bi);
      let rw = u16x8_narrow_i32x4(ri, ri);
      let aw = u16x8_narrow_i32x4(ai, ai);
      let mut g_buf = [0u16; 8];
      let mut b_buf = [0u16; 8];
      let mut r_buf = [0u16; 8];
      let mut a_buf = [0u16; 8];
      v128_store(g_buf.as_mut_ptr().cast(), gw);
      v128_store(b_buf.as_mut_ptr().cast(), bw);
      v128_store(r_buf.as_mut_ptr().cast(), rw);
      v128_store(a_buf.as_mut_ptr().cast(), aw);
      let base = x * 4;
      for p in 0..4 {
        out[base + p * 4] = r_buf[p];
        out[base + p * 4 + 1] = g_buf[p];
        out[base + p * 4 + 2] = b_buf[p];
        out[base + p * 4 + 3] = a_buf[p];
      }
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

// ---- Gbrapf32 → f32 RGBA (lossless, source α) --------------------------------

/// wasm-simd128: planar Gbrapf32 → packed `R, G, B, A` f32 (lossless).
///
/// Delegates to scalar (no 4-channel interleave intrinsic).
///
/// # Safety
///
/// 1. simd128 must be available (compile-time `target_feature`).
/// 2. `g.len()`, `b.len()`, `r.len()`, `a.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "simd128")]
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

// ---- Gbrapf32 → f16 RGBA (scalar narrow, source α) --------------------------

/// wasm-simd128: planar Gbrapf32 → packed `R, G, B, A` f16 (source α).
///
/// wasm-simd128 has no native f16 narrowing. Uses scalar `half::f16::from_f32`.
///
/// # Safety
///
/// 1. simd128 must be available (compile-time `target_feature`).
/// 2. `g.len()`, `b.len()`, `r.len()`, `a.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn gbrapf32_to_rgba_f16_row<const BE: bool>(
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

  scalar::gbrapf32_to_rgba_f16_row::<BE>(g, b, r, a, out, width);
}

// ---- Gbrpf16 → f16 RGB (lossless, f16-native) --------------------------------

/// wasm-simd128: planar Gbrpf16 → packed `R, G, B` f16 (lossless interleave).
///
/// No arithmetic — treats f16 lanes as opaque `u16`. Delegates to the scalar
/// f16-native kernel (which copies via the `half::f16` API per element).
///
/// # Safety
///
/// 1. simd128 must be available (compile-time `target_feature`).
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `3 * width`.
#[inline]
#[target_feature(enable = "simd128")]
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

  scalar_f16::gbrpf16_to_rgb_f16_row::<BE>(g, b, r, out, width);
}

// ---- Gbrpf16 → f16 RGBA (lossless, opaque α = f16(1.0)) ---------------------

/// wasm-simd128: planar Gbrpf16 → packed `R, G, B, A` f16
/// (lossless + α = f16(1.0) = 0x3C00).
///
/// # Safety
///
/// 1. simd128 must be available (compile-time `target_feature`).
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "simd128")]
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

  scalar_f16::gbrpf16_to_rgba_f16_row::<BE>(g, b, r, out, width);
}

// ---- Gbrapf16 → f16 RGBA (lossless, source α) --------------------------------

/// wasm-simd128: planar Gbrapf16 → packed `R, G, B, A` f16
/// (lossless + source α plane).
///
/// # Safety
///
/// 1. simd128 must be available (compile-time `target_feature`).
/// 2. `g.len()`, `b.len()`, `r.len()`, `a.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "simd128")]
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

  scalar_f16::gbrapf16_to_rgba_f16_row::<BE>(g, b, r, a, out, width);
}

// ---- Gbrpf16 → u8 RGB (widen f16→f32 scalar, then SIMD f32→u8) --------------

/// wasm-simd128: planar Gbrpf16 → packed `R, G, B` bytes.
///
/// wasm-simd128 has no native f16 widening. Strategy: widen each plane
/// to f32 in 4-element stack scratch via
/// [`scalar_f16::widen_f16_be_to_host_f32::<BE>`] (which normalizes the
/// source f16 bits per `BE` BEFORE the f16 → f32 conversion), then call the
/// SIMD `gbrpf32_to_rgb_row` kernel routed via `HOST_NATIVE_BE` (since the
/// f32 scratch is now host-native). Both the SIMD-aligned body chunks and
/// the scalar tail share this contract — see file-level `HOST_NATIVE_BE`
/// doc.
///
/// # Safety
///
/// 1. simd128 must be available (compile-time `target_feature`).
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `3 * width`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn gbrpf16_to_rgb_row<const BE: bool>(
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

  const CHUNK: usize = 4;
  let mut gf = [0.0f32; CHUNK];
  let mut bf = [0.0f32; CHUNK];
  let mut rf = [0.0f32; CHUNK];
  let mut x = 0usize;
  while x + CHUNK <= width {
    scalar_f16::widen_f16_be_to_host_f32::<BE>(g, x, &mut gf, CHUNK);
    scalar_f16::widen_f16_be_to_host_f32::<BE>(b, x, &mut bf, CHUNK);
    scalar_f16::widen_f16_be_to_host_f32::<BE>(r, x, &mut rf, CHUNK);
    unsafe {
      gbrpf32_to_rgb_row::<HOST_NATIVE_BE>(&gf, &bf, &rf, &mut out[x * 3..(x + CHUNK) * 3], CHUNK);
    }
    x += CHUNK;
  }
  if x < width {
    let n = width - x;
    scalar_f16::widen_f16_be_to_host_f32::<BE>(g, x, &mut gf, n);
    scalar_f16::widen_f16_be_to_host_f32::<BE>(b, x, &mut bf, n);
    scalar_f16::widen_f16_be_to_host_f32::<BE>(r, x, &mut rf, n);
    scalar::gbrpf32_to_rgb_row::<HOST_NATIVE_BE>(
      &gf[..n],
      &bf[..n],
      &rf[..n],
      &mut out[x * 3..width * 3],
      n,
    );
  }
}

// ---- Gbrpf16 → u8 RGBA (widen f16→f32 scalar, then SIMD f32→u8) -------------

/// wasm-simd128: planar Gbrpf16 → packed `R, G, B, A` bytes (α = 0xFF).
///
/// # Safety
///
/// 1. simd128 must be available (compile-time `target_feature`).
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn gbrpf16_to_rgba_row<const BE: bool>(
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

  const CHUNK: usize = 4;
  let mut gf = [0.0f32; CHUNK];
  let mut bf = [0.0f32; CHUNK];
  let mut rf = [0.0f32; CHUNK];
  let mut x = 0usize;
  while x + CHUNK <= width {
    scalar_f16::widen_f16_be_to_host_f32::<BE>(g, x, &mut gf, CHUNK);
    scalar_f16::widen_f16_be_to_host_f32::<BE>(b, x, &mut bf, CHUNK);
    scalar_f16::widen_f16_be_to_host_f32::<BE>(r, x, &mut rf, CHUNK);
    unsafe {
      gbrpf32_to_rgba_row::<HOST_NATIVE_BE>(&gf, &bf, &rf, &mut out[x * 4..(x + CHUNK) * 4], CHUNK);
    }
    x += CHUNK;
  }
  if x < width {
    let n = width - x;
    scalar_f16::widen_f16_be_to_host_f32::<BE>(g, x, &mut gf, n);
    scalar_f16::widen_f16_be_to_host_f32::<BE>(b, x, &mut bf, n);
    scalar_f16::widen_f16_be_to_host_f32::<BE>(r, x, &mut rf, n);
    scalar::gbrpf32_to_rgba_row::<HOST_NATIVE_BE>(
      &gf[..n],
      &bf[..n],
      &rf[..n],
      &mut out[x * 4..width * 4],
      n,
    );
  }
}

// ---- Gbrpf16 → u16 RGB (widen f16→f32 scalar, then SIMD f32→u16) ------------

/// wasm-simd128: planar Gbrpf16 → packed `R, G, B` u16.
///
/// # Safety
///
/// 1. simd128 must be available (compile-time `target_feature`).
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `3 * width`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn gbrpf16_to_rgb_u16_row<const BE: bool>(
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

  const CHUNK: usize = 4;
  let mut gf = [0.0f32; CHUNK];
  let mut bf = [0.0f32; CHUNK];
  let mut rf = [0.0f32; CHUNK];
  let mut x = 0usize;
  while x + CHUNK <= width {
    scalar_f16::widen_f16_be_to_host_f32::<BE>(g, x, &mut gf, CHUNK);
    scalar_f16::widen_f16_be_to_host_f32::<BE>(b, x, &mut bf, CHUNK);
    scalar_f16::widen_f16_be_to_host_f32::<BE>(r, x, &mut rf, CHUNK);
    unsafe {
      gbrpf32_to_rgb_u16_row::<HOST_NATIVE_BE>(
        &gf,
        &bf,
        &rf,
        &mut out[x * 3..(x + CHUNK) * 3],
        CHUNK,
      );
    }
    x += CHUNK;
  }
  if x < width {
    let n = width - x;
    scalar_f16::widen_f16_be_to_host_f32::<BE>(g, x, &mut gf, n);
    scalar_f16::widen_f16_be_to_host_f32::<BE>(b, x, &mut bf, n);
    scalar_f16::widen_f16_be_to_host_f32::<BE>(r, x, &mut rf, n);
    scalar::gbrpf32_to_rgb_u16_row::<HOST_NATIVE_BE>(
      &gf[..n],
      &bf[..n],
      &rf[..n],
      &mut out[x * 3..width * 3],
      n,
    );
  }
}

// ---- Gbrpf16 → u16 RGBA (widen f16→f32 scalar, then SIMD f32→u16) -----------

/// wasm-simd128: planar Gbrpf16 → packed `R, G, B, A` u16 (α = 0xFFFF).
///
/// # Safety
///
/// 1. simd128 must be available (compile-time `target_feature`).
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn gbrpf16_to_rgba_u16_row<const BE: bool>(
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

  const CHUNK: usize = 4;
  let mut gf = [0.0f32; CHUNK];
  let mut bf = [0.0f32; CHUNK];
  let mut rf = [0.0f32; CHUNK];
  let mut x = 0usize;
  while x + CHUNK <= width {
    scalar_f16::widen_f16_be_to_host_f32::<BE>(g, x, &mut gf, CHUNK);
    scalar_f16::widen_f16_be_to_host_f32::<BE>(b, x, &mut bf, CHUNK);
    scalar_f16::widen_f16_be_to_host_f32::<BE>(r, x, &mut rf, CHUNK);
    unsafe {
      gbrpf32_to_rgba_u16_row::<HOST_NATIVE_BE>(
        &gf,
        &bf,
        &rf,
        &mut out[x * 4..(x + CHUNK) * 4],
        CHUNK,
      );
    }
    x += CHUNK;
  }
  if x < width {
    let n = width - x;
    scalar_f16::widen_f16_be_to_host_f32::<BE>(g, x, &mut gf, n);
    scalar_f16::widen_f16_be_to_host_f32::<BE>(b, x, &mut bf, n);
    scalar_f16::widen_f16_be_to_host_f32::<BE>(r, x, &mut rf, n);
    scalar::gbrpf32_to_rgba_u16_row::<HOST_NATIVE_BE>(
      &gf[..n],
      &bf[..n],
      &rf[..n],
      &mut out[x * 4..width * 4],
      n,
    );
  }
}
