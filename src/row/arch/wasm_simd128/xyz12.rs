//! WASM simd128 kernels for the Tier 12 (DCP / `Xyz12`) source.
//!
//! Each kernel processes **8 pixels per SIMD iteration**. simd128 is a
//! 128-bit ISA so each register holds 4 `f32` lanes — same shape as
//! the SSE4.1 / NEON backends. The matmul, integer narrow, and
//! interleaved store are vectorized; the SMPTE 428-1 inverse OETF and
//! sRGB-shape forward OETF run **scalar per lane** via the same scalar
//! functions the reference kernel uses, preserving the 0-ULP scalar↔SIMD
//! parity contract by construction.
//!
//! Pipeline:
//! 1. Three `load_endian_u16x8::<BE>` loads (3 × 16 B = 48 B = 8 px ×
//!    XYZ in 24 u16) feed a 3-source byte-swizzle deinterleave (three
//!    `u8x16_swizzle` + `v128_or` per channel — same OR-mask pattern
//!    as the SSE4.1 `_mm_shuffle_epi8` cascade).
//! 2. Each `(X8, Y8, Z8)` u16x8 vector is right-shifted by 4
//!    (`u16x8_shr(.., 4)`) to extract the active 12-bit code from the
//!    high-bit-packed `u16` (FFmpeg `AV_PIX_FMT_XYZ12LE/BE`: code in
//!    `[15:4]`, low 4 bits zero), defensively masked with `v128_and`,
//!    then split into low/high i32x4 halves (`i32x4_extend_low_i16x8`
//!    / `i32x4_extend_high_i16x8`) and cast to f32x4 via
//!    `f32x4_convert_i32x4`. The 12-bit mask keeps values in the
//!    positive i32 range, so signed widening matches the scalar `as
//!    f32` of a u16.
//! 3. Per-lane scalar `smpte428_inverse_oetf` produces linear XYZ
//!    f32x4 halves (stored to a stack array, scalar-called four times,
//!    reloaded).
//! 4. Vectorized 3×3 matmul uses plain `f32x4_mul + f32x4_add` — no
//!    fused multiply-add primitive on simd128, so the rounding profile
//!    is identical to the scalar reference.
//! 5. Per-lane scalar `oetf_srgb` (only for u8 / u16 / f16 outputs).
//! 6. Clamp `[0, 1]` × `scale` + `+ 0.5` then `i32x4_trunc_sat_f32x4`
//!    (truncate toward zero matches scalar's `(c * scale + 0.5) as int`
//!    for non-negative pre-clamped values). Saturating `u16x8_narrow_i32x4`
//!    packs to u16x8; `u8x16_narrow_i16x8` packs further to u8x16.
//!
//! Width remainder (`width % 8`) is handled by the scalar reference.
//!
//! # Numerical contract
//!
//! Bit-identical scalar↔SIMD output across all integer / f16 / f32
//! paths (verified by `tests::xyz12`).

use core::arch::wasm32::*;

use super::endian::load_endian_u16x8;
use crate::{
  KernelGamut,
  row::scalar::{
    self,
    xyz12::{oetf_srgb, smpte428_inverse_oetf},
    xyz12_constants::xyz_to_rgb_matrix,
  },
};

const PIXELS_PER_ITER: usize = 8;
const SAMPLE_MASK_U16: u16 = 0x0FFF;

// ---- Internal helpers --------------------------------------------------

/// Deinterleaves 3 `v128` registers (24 u16 = 8 packed XYZ pixels) into
/// `(X8, Y8, Z8)` u16x8 channel vectors. Mirrors the SSE4.1
/// `_mm_shuffle_epi8` byte pattern exactly: `u8x16_swizzle` matches
/// `_mm_shuffle_epi8` semantics (indices ≥ 16 zero the lane, identical
/// to setting the high bit on x86), and `v128_or` is bit-for-bit the
/// same as `_mm_or_si128`.
#[inline(always)]
unsafe fn deinterleave_xyz12_8px(v0: v128, v1: v128, v2: v128) -> (v128, v128, v128) {
  // ---- ch0 (X) -------------------------------------------------
  // From v0: u16 positions 0, 3, 6 → output positions 0, 1, 2.
  // -1 = lane zeroed; matches the `_mm_setr_epi8(..., -1, ...)` pattern.
  let x_v0 = i8x16(0, 1, 6, 7, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
  let x_v1 = i8x16(-1, -1, -1, -1, -1, -1, 2, 3, 8, 9, 14, 15, -1, -1, -1, -1);
  let x_v2 = i8x16(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 5, 10, 11);
  let x = v128_or(
    v128_or(u8x16_swizzle(v0, x_v0), u8x16_swizzle(v1, x_v1)),
    u8x16_swizzle(v2, x_v2),
  );

  // ---- ch1 (Y) -------------------------------------------------
  let y_v0 = i8x16(2, 3, 8, 9, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
  let y_v1 = i8x16(-1, -1, -1, -1, -1, -1, 4, 5, 10, 11, -1, -1, -1, -1, -1, -1);
  let y_v2 = i8x16(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 6, 7, 12, 13);
  let y = v128_or(
    v128_or(u8x16_swizzle(v0, y_v0), u8x16_swizzle(v1, y_v1)),
    u8x16_swizzle(v2, y_v2),
  );

  // ---- ch2 (Z) -------------------------------------------------
  let z_v0 = i8x16(4, 5, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
  let z_v1 = i8x16(-1, -1, -1, -1, 0, 1, 6, 7, 12, 13, -1, -1, -1, -1, -1, -1);
  let z_v2 = i8x16(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 3, 8, 9, 14, 15);
  let z = v128_or(
    v128_or(u8x16_swizzle(v0, z_v0), u8x16_swizzle(v1, z_v1)),
    u8x16_swizzle(v2, z_v2),
  );

  (x, y, z)
}

/// Splits a u16x8 channel vector (already 12-bit-masked) into low/high
/// f32x4 halves via signed widening (`i32x4_extend_*_i16x8`) + signed
/// i32 → f32 cvt (`f32x4_convert_i32x4`). The 12-bit mask keeps values
/// `< 4096`, well below `i32::MAX`, so signed/unsigned widening agree.
#[inline(always)]
unsafe fn u16x8_to_f32x4_pair(v: v128) -> (v128, v128) {
  let lo_i32 = i32x4_extend_low_i16x8(v);
  let hi_i32 = i32x4_extend_high_i16x8(v);
  (f32x4_convert_i32x4(lo_i32), f32x4_convert_i32x4(hi_i32))
}

/// Per-lane scalar SMPTE 428-1 inverse OETF on a `v128` (4 f32 lanes).
/// Stores to a stack array, calls the scalar function 4 times, reloads.
#[inline(always)]
unsafe fn smpte428_inv_oetf_scalar4(v: v128) -> v128 {
  unsafe {
    let mut buf = [0.0_f32; 4];
    v128_store(buf.as_mut_ptr() as *mut v128, v);
    for slot in &mut buf {
      *slot = smpte428_inverse_oetf(*slot as u16);
    }
    v128_load(buf.as_ptr() as *const v128)
  }
}

/// Per-lane scalar sRGB OETF on a `v128` (4 f32 lanes).
#[inline(always)]
unsafe fn oetf_srgb_scalar4(v: v128) -> v128 {
  unsafe {
    let mut buf = [0.0_f32; 4];
    v128_store(buf.as_mut_ptr() as *mut v128, v);
    for slot in &mut buf {
      *slot = oetf_srgb(*slot);
    }
    v128_load(buf.as_ptr() as *const v128)
  }
}

/// Vectorized 3×3 matmul on a 4-lane f32 vector: `[R G B]^T = M ·
/// [X Y Z]^T`. Plain mul + add (NOT FMA) — single-rounding FMA breaks
/// the 0-ULP parity contract on integer-narrow output paths, same as
/// the NEON / SSE4.1 backends. simd128 has no fused multiply-add
/// primitive anyway.
#[inline(always)]
unsafe fn matmul_xyz_to_rgb_4lane(
  m: &[[f32; 3]; 3],
  x: v128,
  y: v128,
  z: v128,
) -> (v128, v128, v128) {
  let m00 = f32x4_splat(m[0][0]);
  let m01 = f32x4_splat(m[0][1]);
  let m02 = f32x4_splat(m[0][2]);
  let m10 = f32x4_splat(m[1][0]);
  let m11 = f32x4_splat(m[1][1]);
  let m12 = f32x4_splat(m[1][2]);
  let m20 = f32x4_splat(m[2][0]);
  let m21 = f32x4_splat(m[2][1]);
  let m22 = f32x4_splat(m[2][2]);
  let r = f32x4_add(
    f32x4_add(f32x4_mul(m00, x), f32x4_mul(m01, y)),
    f32x4_mul(m02, z),
  );
  let g = f32x4_add(
    f32x4_add(f32x4_mul(m10, x), f32x4_mul(m11, y)),
    f32x4_mul(m12, z),
  );
  let b = f32x4_add(
    f32x4_add(f32x4_mul(m20, x), f32x4_mul(m21, y)),
    f32x4_mul(m22, z),
  );
  (r, g, b)
}

/// Loads 8 XYZ12 pixels and produces the linear RGB f32 lanes after
/// the inverse-OETF + 3×3 matmul. Returns six `v128` vectors:
/// low/high halves of R, G, B (in that order).
#[inline(always)]
unsafe fn load_and_matmul_8px<const BE: bool>(
  p: *const u8,
  m: &[[f32; 3]; 3],
) -> ((v128, v128), (v128, v128), (v128, v128)) {
  unsafe {
    let v0 = load_endian_u16x8::<BE>(p);
    let v1 = load_endian_u16x8::<BE>(p.add(16));
    let v2 = load_endian_u16x8::<BE>(p.add(32));
    let (x_u, y_u, z_u) = deinterleave_xyz12_8px(v0, v1, v2);
    // Shift right 4 to extract active 12-bit code (high-bit-packed
    // FFmpeg `AV_PIX_FMT_XYZ12LE/BE`); mask defensively.
    let mask = u16x8_splat(SAMPLE_MASK_U16);
    let x_shr = v128_and(u16x8_shr(x_u, 4), mask);
    let y_shr = v128_and(u16x8_shr(y_u, 4), mask);
    let z_shr = v128_and(u16x8_shr(z_u, 4), mask);
    let (x_lo, x_hi) = u16x8_to_f32x4_pair(x_shr);
    let (y_lo, y_hi) = u16x8_to_f32x4_pair(y_shr);
    let (z_lo, z_hi) = u16x8_to_f32x4_pair(z_shr);
    let x_lo = smpte428_inv_oetf_scalar4(x_lo);
    let x_hi = smpte428_inv_oetf_scalar4(x_hi);
    let y_lo = smpte428_inv_oetf_scalar4(y_lo);
    let y_hi = smpte428_inv_oetf_scalar4(y_hi);
    let z_lo = smpte428_inv_oetf_scalar4(z_lo);
    let z_hi = smpte428_inv_oetf_scalar4(z_hi);
    let (r_lo, g_lo, b_lo) = matmul_xyz_to_rgb_4lane(m, x_lo, y_lo, z_lo);
    let (r_hi, g_hi, b_hi) = matmul_xyz_to_rgb_4lane(m, x_hi, y_hi, z_hi);
    ((r_lo, r_hi), (g_lo, g_hi), (b_lo, b_hi))
  }
}

/// Loads 8 XYZ12 pixels and produces 6 `v128` linear-XYZ f32 vectors
/// (low/high halves of X, Y, Z) — step 1 only, no matmul.
#[inline(always)]
unsafe fn load_xyz_linear_8px<const BE: bool>(
  p: *const u8,
) -> ((v128, v128), (v128, v128), (v128, v128)) {
  unsafe {
    let v0 = load_endian_u16x8::<BE>(p);
    let v1 = load_endian_u16x8::<BE>(p.add(16));
    let v2 = load_endian_u16x8::<BE>(p.add(32));
    let (x_u, y_u, z_u) = deinterleave_xyz12_8px(v0, v1, v2);
    let mask = u16x8_splat(SAMPLE_MASK_U16);
    let x_shr = v128_and(u16x8_shr(x_u, 4), mask);
    let y_shr = v128_and(u16x8_shr(y_u, 4), mask);
    let z_shr = v128_and(u16x8_shr(z_u, 4), mask);
    let (x_lo, x_hi) = u16x8_to_f32x4_pair(x_shr);
    let (y_lo, y_hi) = u16x8_to_f32x4_pair(y_shr);
    let (z_lo, z_hi) = u16x8_to_f32x4_pair(z_shr);
    (
      (
        smpte428_inv_oetf_scalar4(x_lo),
        smpte428_inv_oetf_scalar4(x_hi),
      ),
      (
        smpte428_inv_oetf_scalar4(y_lo),
        smpte428_inv_oetf_scalar4(y_hi),
      ),
      (
        smpte428_inv_oetf_scalar4(z_lo),
        smpte428_inv_oetf_scalar4(z_hi),
      ),
    )
  }
}

/// Vectorized clamp `[0, 1]` × scale + `+0.5` truncate-toward-zero,
/// returning an i32x4 ready to feed the saturating `u16x8_narrow_i32x4`.
///
/// Matches the scalar `(c.clamp(0, 1) * scale + 0.5) as int` exactly:
/// clamp via `f32x4_min/max`, scale, add 0.5, truncate via
/// `i32x4_trunc_sat_f32x4`. Pre-clamping keeps the f32 well below
/// `i32::MAX` so saturation is a no-op on valid input.
#[inline(always)]
unsafe fn clamp_scale_to_i32x4(v: v128, zero: v128, one: v128, scale: v128) -> v128 {
  let half = f32x4_splat(0.5);
  let clamped = f32x4_min(f32x4_max(v, zero), one);
  let scaled = f32x4_add(f32x4_mul(clamped, scale), half);
  i32x4_trunc_sat_f32x4(scaled)
}

// ---- Per-output kernels ------------------------------------------------

/// XYZ12 → packed u8 RGB. 8 pixels per SIMD iteration.
///
/// # Safety
///
/// 1. simd128 must be available (compile-time `target_feature`).
/// 2. `xyz.len() >= width * 3`; `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn xyz12_to_rgb_row<const BE: bool>(
  xyz: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  target_gamut: KernelGamut,
) {
  debug_assert!(xyz.len() >= width * 3, "xyz row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");
  let m = xyz_to_rgb_matrix(target_gamut);

  unsafe {
    let zero_ps = f32x4_splat(0.0);
    let one_ps = f32x4_splat(1.0);
    let scale = f32x4_splat(255.0);
    let zero_si = i32x4_splat(0);

    let mut x = 0usize;
    while x + PIXELS_PER_ITER <= width {
      let p = xyz.as_ptr().add(x * 3) as *const u8;
      let ((r_lo, r_hi), (g_lo, g_hi), (b_lo, b_hi)) = load_and_matmul_8px::<BE>(p, &m);
      let r_lo = oetf_srgb_scalar4(r_lo);
      let r_hi = oetf_srgb_scalar4(r_hi);
      let g_lo = oetf_srgb_scalar4(g_lo);
      let g_hi = oetf_srgb_scalar4(g_hi);
      let b_lo = oetf_srgb_scalar4(b_lo);
      let b_hi = oetf_srgb_scalar4(b_hi);
      let r_lo_i = clamp_scale_to_i32x4(r_lo, zero_ps, one_ps, scale);
      let r_hi_i = clamp_scale_to_i32x4(r_hi, zero_ps, one_ps, scale);
      let g_lo_i = clamp_scale_to_i32x4(g_lo, zero_ps, one_ps, scale);
      let g_hi_i = clamp_scale_to_i32x4(g_hi, zero_ps, one_ps, scale);
      let b_lo_i = clamp_scale_to_i32x4(b_lo, zero_ps, one_ps, scale);
      let b_hi_i = clamp_scale_to_i32x4(b_hi, zero_ps, one_ps, scale);
      let r_u16 = u16x8_narrow_i32x4(r_lo_i, r_hi_i);
      let g_u16 = u16x8_narrow_i32x4(g_lo_i, g_hi_i);
      let b_u16 = u16x8_narrow_i32x4(b_lo_i, b_hi_i);
      // Narrow each u16x8 to u8x8 (low 8 bytes) by saturating-narrow with zero.
      let r_u8 = u8x16_narrow_i16x8(r_u16, zero_si);
      let g_u8 = u8x16_narrow_i16x8(g_u16, zero_si);
      let b_u8 = u8x16_narrow_i16x8(b_u16, zero_si);
      let mut tmp_r = [0u8; 16];
      let mut tmp_g = [0u8; 16];
      let mut tmp_b = [0u8; 16];
      v128_store(tmp_r.as_mut_ptr() as *mut v128, r_u8);
      v128_store(tmp_g.as_mut_ptr() as *mut v128, g_u8);
      v128_store(tmp_b.as_mut_ptr() as *mut v128, b_u8);
      let dst = rgb_out.as_mut_ptr().add(x * 3);
      for i in 0..PIXELS_PER_ITER {
        *dst.add(i * 3) = tmp_r[i];
        *dst.add(i * 3 + 1) = tmp_g[i];
        *dst.add(i * 3 + 2) = tmp_b[i];
      }
      x += PIXELS_PER_ITER;
    }
    if x < width {
      scalar::xyz12::xyz12_to_rgb_row::<BE>(
        &xyz[x * 3..width * 3],
        &mut rgb_out[x * 3..width * 3],
        width - x,
        target_gamut,
      );
    }
  }
}

/// XYZ12 → packed u8 RGBA (alpha = `0xFF`).
///
/// # Safety
///
/// 1. simd128 must be available.
/// 2. `xyz.len() >= width * 3`; `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn xyz12_to_rgba_row<const BE: bool>(
  xyz: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  target_gamut: KernelGamut,
) {
  debug_assert!(xyz.len() >= width * 3, "xyz row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  let m = xyz_to_rgb_matrix(target_gamut);

  unsafe {
    let zero_ps = f32x4_splat(0.0);
    let one_ps = f32x4_splat(1.0);
    let scale = f32x4_splat(255.0);
    let zero_si = i32x4_splat(0);

    let mut x = 0usize;
    while x + PIXELS_PER_ITER <= width {
      let p = xyz.as_ptr().add(x * 3) as *const u8;
      let ((r_lo, r_hi), (g_lo, g_hi), (b_lo, b_hi)) = load_and_matmul_8px::<BE>(p, &m);
      let r_lo = oetf_srgb_scalar4(r_lo);
      let r_hi = oetf_srgb_scalar4(r_hi);
      let g_lo = oetf_srgb_scalar4(g_lo);
      let g_hi = oetf_srgb_scalar4(g_hi);
      let b_lo = oetf_srgb_scalar4(b_lo);
      let b_hi = oetf_srgb_scalar4(b_hi);
      let r_lo_i = clamp_scale_to_i32x4(r_lo, zero_ps, one_ps, scale);
      let r_hi_i = clamp_scale_to_i32x4(r_hi, zero_ps, one_ps, scale);
      let g_lo_i = clamp_scale_to_i32x4(g_lo, zero_ps, one_ps, scale);
      let g_hi_i = clamp_scale_to_i32x4(g_hi, zero_ps, one_ps, scale);
      let b_lo_i = clamp_scale_to_i32x4(b_lo, zero_ps, one_ps, scale);
      let b_hi_i = clamp_scale_to_i32x4(b_hi, zero_ps, one_ps, scale);
      let r_u16 = u16x8_narrow_i32x4(r_lo_i, r_hi_i);
      let g_u16 = u16x8_narrow_i32x4(g_lo_i, g_hi_i);
      let b_u16 = u16x8_narrow_i32x4(b_lo_i, b_hi_i);
      let r_u8 = u8x16_narrow_i16x8(r_u16, zero_si);
      let g_u8 = u8x16_narrow_i16x8(g_u16, zero_si);
      let b_u8 = u8x16_narrow_i16x8(b_u16, zero_si);
      let mut tmp_r = [0u8; 16];
      let mut tmp_g = [0u8; 16];
      let mut tmp_b = [0u8; 16];
      v128_store(tmp_r.as_mut_ptr() as *mut v128, r_u8);
      v128_store(tmp_g.as_mut_ptr() as *mut v128, g_u8);
      v128_store(tmp_b.as_mut_ptr() as *mut v128, b_u8);
      let dst = rgba_out.as_mut_ptr().add(x * 4);
      for i in 0..PIXELS_PER_ITER {
        *dst.add(i * 4) = tmp_r[i];
        *dst.add(i * 4 + 1) = tmp_g[i];
        *dst.add(i * 4 + 2) = tmp_b[i];
        *dst.add(i * 4 + 3) = 0xFF;
      }
      x += PIXELS_PER_ITER;
    }
    if x < width {
      scalar::xyz12::xyz12_to_rgba_row::<BE>(
        &xyz[x * 3..width * 3],
        &mut rgba_out[x * 4..width * 4],
        width - x,
        target_gamut,
      );
    }
  }
}

/// XYZ12 → packed u16 RGB (full-range scaling, ×65535).
///
/// # Safety
///
/// 1. simd128 must be available.
/// 2. `xyz.len() >= width * 3`; `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn xyz12_to_rgb_u16_row<const BE: bool>(
  xyz: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  target_gamut: KernelGamut,
) {
  debug_assert!(xyz.len() >= width * 3, "xyz row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");
  let m = xyz_to_rgb_matrix(target_gamut);

  unsafe {
    let zero_ps = f32x4_splat(0.0);
    let one_ps = f32x4_splat(1.0);
    let scale = f32x4_splat(65535.0);

    let mut x = 0usize;
    while x + PIXELS_PER_ITER <= width {
      let p = xyz.as_ptr().add(x * 3) as *const u8;
      let ((r_lo, r_hi), (g_lo, g_hi), (b_lo, b_hi)) = load_and_matmul_8px::<BE>(p, &m);
      let r_lo = oetf_srgb_scalar4(r_lo);
      let r_hi = oetf_srgb_scalar4(r_hi);
      let g_lo = oetf_srgb_scalar4(g_lo);
      let g_hi = oetf_srgb_scalar4(g_hi);
      let b_lo = oetf_srgb_scalar4(b_lo);
      let b_hi = oetf_srgb_scalar4(b_hi);
      let r_lo_i = clamp_scale_to_i32x4(r_lo, zero_ps, one_ps, scale);
      let r_hi_i = clamp_scale_to_i32x4(r_hi, zero_ps, one_ps, scale);
      let g_lo_i = clamp_scale_to_i32x4(g_lo, zero_ps, one_ps, scale);
      let g_hi_i = clamp_scale_to_i32x4(g_hi, zero_ps, one_ps, scale);
      let b_lo_i = clamp_scale_to_i32x4(b_lo, zero_ps, one_ps, scale);
      let b_hi_i = clamp_scale_to_i32x4(b_hi, zero_ps, one_ps, scale);
      let r_u16 = u16x8_narrow_i32x4(r_lo_i, r_hi_i);
      let g_u16 = u16x8_narrow_i32x4(g_lo_i, g_hi_i);
      let b_u16 = u16x8_narrow_i32x4(b_lo_i, b_hi_i);
      let mut tmp_r = [0u16; 8];
      let mut tmp_g = [0u16; 8];
      let mut tmp_b = [0u16; 8];
      v128_store(tmp_r.as_mut_ptr() as *mut v128, r_u16);
      v128_store(tmp_g.as_mut_ptr() as *mut v128, g_u16);
      v128_store(tmp_b.as_mut_ptr() as *mut v128, b_u16);
      let dst = rgb_out.as_mut_ptr().add(x * 3);
      for i in 0..PIXELS_PER_ITER {
        *dst.add(i * 3) = tmp_r[i];
        *dst.add(i * 3 + 1) = tmp_g[i];
        *dst.add(i * 3 + 2) = tmp_b[i];
      }
      x += PIXELS_PER_ITER;
    }
    if x < width {
      scalar::xyz12::xyz12_to_rgb_u16_row::<BE>(
        &xyz[x * 3..width * 3],
        &mut rgb_out[x * 3..width * 3],
        width - x,
        target_gamut,
      );
    }
  }
}

/// XYZ12 → packed u16 RGBA (alpha = `0xFFFF`).
///
/// # Safety
///
/// 1. simd128 must be available.
/// 2. `xyz.len() >= width * 3`; `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn xyz12_to_rgba_u16_row<const BE: bool>(
  xyz: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  target_gamut: KernelGamut,
) {
  debug_assert!(xyz.len() >= width * 3, "xyz row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  let m = xyz_to_rgb_matrix(target_gamut);

  unsafe {
    let zero_ps = f32x4_splat(0.0);
    let one_ps = f32x4_splat(1.0);
    let scale = f32x4_splat(65535.0);

    let mut x = 0usize;
    while x + PIXELS_PER_ITER <= width {
      let p = xyz.as_ptr().add(x * 3) as *const u8;
      let ((r_lo, r_hi), (g_lo, g_hi), (b_lo, b_hi)) = load_and_matmul_8px::<BE>(p, &m);
      let r_lo = oetf_srgb_scalar4(r_lo);
      let r_hi = oetf_srgb_scalar4(r_hi);
      let g_lo = oetf_srgb_scalar4(g_lo);
      let g_hi = oetf_srgb_scalar4(g_hi);
      let b_lo = oetf_srgb_scalar4(b_lo);
      let b_hi = oetf_srgb_scalar4(b_hi);
      let r_lo_i = clamp_scale_to_i32x4(r_lo, zero_ps, one_ps, scale);
      let r_hi_i = clamp_scale_to_i32x4(r_hi, zero_ps, one_ps, scale);
      let g_lo_i = clamp_scale_to_i32x4(g_lo, zero_ps, one_ps, scale);
      let g_hi_i = clamp_scale_to_i32x4(g_hi, zero_ps, one_ps, scale);
      let b_lo_i = clamp_scale_to_i32x4(b_lo, zero_ps, one_ps, scale);
      let b_hi_i = clamp_scale_to_i32x4(b_hi, zero_ps, one_ps, scale);
      let r_u16 = u16x8_narrow_i32x4(r_lo_i, r_hi_i);
      let g_u16 = u16x8_narrow_i32x4(g_lo_i, g_hi_i);
      let b_u16 = u16x8_narrow_i32x4(b_lo_i, b_hi_i);
      let mut tmp_r = [0u16; 8];
      let mut tmp_g = [0u16; 8];
      let mut tmp_b = [0u16; 8];
      v128_store(tmp_r.as_mut_ptr() as *mut v128, r_u16);
      v128_store(tmp_g.as_mut_ptr() as *mut v128, g_u16);
      v128_store(tmp_b.as_mut_ptr() as *mut v128, b_u16);
      let dst = rgba_out.as_mut_ptr().add(x * 4);
      for i in 0..PIXELS_PER_ITER {
        *dst.add(i * 4) = tmp_r[i];
        *dst.add(i * 4 + 1) = tmp_g[i];
        *dst.add(i * 4 + 2) = tmp_b[i];
        *dst.add(i * 4 + 3) = 0xFFFF;
      }
      x += PIXELS_PER_ITER;
    }
    if x < width {
      scalar::xyz12::xyz12_to_rgba_u16_row::<BE>(
        &xyz[x * 3..width * 3],
        &mut rgba_out[x * 4..width * 4],
        width - x,
        target_gamut,
      );
    }
  }
}

/// XYZ12 → packed linear RGB f32 (lossless: matrix only, no OETF, no
/// clamp).
///
/// # Safety
///
/// 1. simd128 must be available.
/// 2. `xyz.len() >= width * 3`; `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn xyz12_to_rgb_f32_row<const BE: bool>(
  xyz: &[u16],
  rgb_out: &mut [f32],
  width: usize,
  target_gamut: KernelGamut,
) {
  debug_assert!(xyz.len() >= width * 3, "xyz row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");
  let m = xyz_to_rgb_matrix(target_gamut);

  unsafe {
    let mut x = 0usize;
    while x + PIXELS_PER_ITER <= width {
      let p = xyz.as_ptr().add(x * 3) as *const u8;
      let ((r_lo, r_hi), (g_lo, g_hi), (b_lo, b_hi)) = load_and_matmul_8px::<BE>(p, &m);
      let mut rb = [0.0_f32; 8];
      let mut gb = [0.0_f32; 8];
      let mut bb = [0.0_f32; 8];
      v128_store(rb.as_mut_ptr() as *mut v128, r_lo);
      v128_store(rb.as_mut_ptr().add(4) as *mut v128, r_hi);
      v128_store(gb.as_mut_ptr() as *mut v128, g_lo);
      v128_store(gb.as_mut_ptr().add(4) as *mut v128, g_hi);
      v128_store(bb.as_mut_ptr() as *mut v128, b_lo);
      v128_store(bb.as_mut_ptr().add(4) as *mut v128, b_hi);
      let dst = rgb_out.as_mut_ptr().add(x * 3);
      for i in 0..PIXELS_PER_ITER {
        *dst.add(i * 3) = rb[i];
        *dst.add(i * 3 + 1) = gb[i];
        *dst.add(i * 3 + 2) = bb[i];
      }
      x += PIXELS_PER_ITER;
    }
    if x < width {
      scalar::xyz12::xyz12_to_rgb_f32_row::<BE>(
        &xyz[x * 3..width * 3],
        &mut rgb_out[x * 3..width * 3],
        width - x,
        target_gamut,
      );
    }
  }
}

/// XYZ12 → packed linear XYZ f32 (step 1 only).
///
/// # Safety
///
/// 1. simd128 must be available.
/// 2. `xyz.len() >= width * 3`; `xyz_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn xyz12_to_xyz_f32_row<const BE: bool>(
  xyz: &[u16],
  xyz_out: &mut [f32],
  width: usize,
) {
  debug_assert!(xyz.len() >= width * 3, "xyz row too short");
  debug_assert!(xyz_out.len() >= width * 3, "xyz_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + PIXELS_PER_ITER <= width {
      let p = xyz.as_ptr().add(x * 3) as *const u8;
      let ((x_lo, x_hi), (y_lo, y_hi), (z_lo, z_hi)) = load_xyz_linear_8px::<BE>(p);
      let mut xb = [0.0_f32; 8];
      let mut yb = [0.0_f32; 8];
      let mut zb = [0.0_f32; 8];
      v128_store(xb.as_mut_ptr() as *mut v128, x_lo);
      v128_store(xb.as_mut_ptr().add(4) as *mut v128, x_hi);
      v128_store(yb.as_mut_ptr() as *mut v128, y_lo);
      v128_store(yb.as_mut_ptr().add(4) as *mut v128, y_hi);
      v128_store(zb.as_mut_ptr() as *mut v128, z_lo);
      v128_store(zb.as_mut_ptr().add(4) as *mut v128, z_hi);
      let dst = xyz_out.as_mut_ptr().add(x * 3);
      for i in 0..PIXELS_PER_ITER {
        *dst.add(i * 3) = xb[i];
        *dst.add(i * 3 + 1) = yb[i];
        *dst.add(i * 3 + 2) = zb[i];
      }
      x += PIXELS_PER_ITER;
    }
    if x < width {
      scalar::xyz12::xyz12_to_xyz_f32_row::<BE>(
        &xyz[x * 3..width * 3],
        &mut xyz_out[x * 3..width * 3],
        width - x,
      );
    }
  }
}

/// XYZ12 → packed f16 RGB (gamma-encoded, clamped to `[0, 1]`).
///
/// f16 narrow uses scalar `half::f16::from_f32` (no native wasm f16
/// primitive) — same pattern as the other `Rgbf16` wasm kernels.
///
/// # Safety
///
/// 1. simd128 must be available.
/// 2. `xyz.len() >= width * 3`; `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn xyz12_to_rgb_f16_row<const BE: bool>(
  xyz: &[u16],
  rgb_out: &mut [half::f16],
  width: usize,
  target_gamut: KernelGamut,
) {
  debug_assert!(xyz.len() >= width * 3, "xyz row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");
  let m = xyz_to_rgb_matrix(target_gamut);

  unsafe {
    let zero_ps = f32x4_splat(0.0);
    let one_ps = f32x4_splat(1.0);
    let mut x = 0usize;
    while x + PIXELS_PER_ITER <= width {
      let p = xyz.as_ptr().add(x * 3) as *const u8;
      let ((r_lo, r_hi), (g_lo, g_hi), (b_lo, b_hi)) = load_and_matmul_8px::<BE>(p, &m);
      let r_lo = oetf_srgb_scalar4(r_lo);
      let r_hi = oetf_srgb_scalar4(r_hi);
      let g_lo = oetf_srgb_scalar4(g_lo);
      let g_hi = oetf_srgb_scalar4(g_hi);
      let b_lo = oetf_srgb_scalar4(b_lo);
      let b_hi = oetf_srgb_scalar4(b_hi);
      let r_lo = f32x4_min(f32x4_max(r_lo, zero_ps), one_ps);
      let r_hi = f32x4_min(f32x4_max(r_hi, zero_ps), one_ps);
      let g_lo = f32x4_min(f32x4_max(g_lo, zero_ps), one_ps);
      let g_hi = f32x4_min(f32x4_max(g_hi, zero_ps), one_ps);
      let b_lo = f32x4_min(f32x4_max(b_lo, zero_ps), one_ps);
      let b_hi = f32x4_min(f32x4_max(b_hi, zero_ps), one_ps);
      let mut rb = [0.0_f32; 8];
      let mut gb = [0.0_f32; 8];
      let mut bb = [0.0_f32; 8];
      v128_store(rb.as_mut_ptr() as *mut v128, r_lo);
      v128_store(rb.as_mut_ptr().add(4) as *mut v128, r_hi);
      v128_store(gb.as_mut_ptr() as *mut v128, g_lo);
      v128_store(gb.as_mut_ptr().add(4) as *mut v128, g_hi);
      v128_store(bb.as_mut_ptr() as *mut v128, b_lo);
      v128_store(bb.as_mut_ptr().add(4) as *mut v128, b_hi);
      for i in 0..PIXELS_PER_ITER {
        let oi = (x + i) * 3;
        rgb_out[oi] = half::f16::from_f32(rb[i]);
        rgb_out[oi + 1] = half::f16::from_f32(gb[i]);
        rgb_out[oi + 2] = half::f16::from_f32(bb[i]);
      }
      x += PIXELS_PER_ITER;
    }
    if x < width {
      scalar::xyz12::xyz12_to_rgb_f16_row::<BE>(
        &xyz[x * 3..width * 3],
        &mut rgb_out[x * 3..width * 3],
        width - x,
        target_gamut,
      );
    }
  }
}

/// XYZ12 → packed f16 RGBA (alpha = `1.0`).
///
/// # Safety
///
/// 1. simd128 must be available.
/// 2. `xyz.len() >= width * 3`; `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn xyz12_to_rgba_f16_row<const BE: bool>(
  xyz: &[u16],
  rgba_out: &mut [half::f16],
  width: usize,
  target_gamut: KernelGamut,
) {
  debug_assert!(xyz.len() >= width * 3, "xyz row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  let m = xyz_to_rgb_matrix(target_gamut);
  let one_f16 = half::f16::from_f32(1.0);

  unsafe {
    let zero_ps = f32x4_splat(0.0);
    let one_ps = f32x4_splat(1.0);
    let mut x = 0usize;
    while x + PIXELS_PER_ITER <= width {
      let p = xyz.as_ptr().add(x * 3) as *const u8;
      let ((r_lo, r_hi), (g_lo, g_hi), (b_lo, b_hi)) = load_and_matmul_8px::<BE>(p, &m);
      let r_lo = oetf_srgb_scalar4(r_lo);
      let r_hi = oetf_srgb_scalar4(r_hi);
      let g_lo = oetf_srgb_scalar4(g_lo);
      let g_hi = oetf_srgb_scalar4(g_hi);
      let b_lo = oetf_srgb_scalar4(b_lo);
      let b_hi = oetf_srgb_scalar4(b_hi);
      let r_lo = f32x4_min(f32x4_max(r_lo, zero_ps), one_ps);
      let r_hi = f32x4_min(f32x4_max(r_hi, zero_ps), one_ps);
      let g_lo = f32x4_min(f32x4_max(g_lo, zero_ps), one_ps);
      let g_hi = f32x4_min(f32x4_max(g_hi, zero_ps), one_ps);
      let b_lo = f32x4_min(f32x4_max(b_lo, zero_ps), one_ps);
      let b_hi = f32x4_min(f32x4_max(b_hi, zero_ps), one_ps);
      let mut rb = [0.0_f32; 8];
      let mut gb = [0.0_f32; 8];
      let mut bb = [0.0_f32; 8];
      v128_store(rb.as_mut_ptr() as *mut v128, r_lo);
      v128_store(rb.as_mut_ptr().add(4) as *mut v128, r_hi);
      v128_store(gb.as_mut_ptr() as *mut v128, g_lo);
      v128_store(gb.as_mut_ptr().add(4) as *mut v128, g_hi);
      v128_store(bb.as_mut_ptr() as *mut v128, b_lo);
      v128_store(bb.as_mut_ptr().add(4) as *mut v128, b_hi);
      for i in 0..PIXELS_PER_ITER {
        let oi = (x + i) * 4;
        rgba_out[oi] = half::f16::from_f32(rb[i]);
        rgba_out[oi + 1] = half::f16::from_f32(gb[i]);
        rgba_out[oi + 2] = half::f16::from_f32(bb[i]);
        rgba_out[oi + 3] = one_f16;
      }
      x += PIXELS_PER_ITER;
    }
    if x < width {
      scalar::xyz12::xyz12_to_rgba_f16_row::<BE>(
        &xyz[x * 3..width * 3],
        &mut rgba_out[x * 4..width * 4],
        width - x,
        target_gamut,
      );
    }
  }
}
