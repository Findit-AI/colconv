//! wasm-simd128 V410 (packed YUV 4:4:4, 10-bit) kernels.
//!
//! ## Layout
//!
//! One `u32` per pixel: `bits[9:0]` = U, `bits[19:10]` = Y,
//! `bits[29:20]` = V (2 bits padding at top). No chroma subsampling
//! (4:4:4) — each word yields a complete `(U, Y, V)` triple.
//!
//! ## Per-iter pipeline (4 px / 4 u32 / 16 bytes)
//!
//! Load one `v128` = 4 × u32 lanes. Three `AND + shift` ops extract U /
//! Y / V fields. `i16x8_narrow_i32x4(field, i32x4_splat(0))` narrows each
//! 4-lane i32 to a v128 with 4 valid i16 lanes (lo) + 4 zero lanes (hi).
//!
//! The narrow result feeds the same `chroma_i16x8` / `scale_y` /
//! `q15_shift` helpers used by `v210.rs` and `yuv_planar_high_bit.rs`.
//! Only the low 4 lanes carry valid data; the high 4 are don't-care.
//!
//! ## Tail
//!
//! `width % 4` remaining pixels fall through to `scalar::v410_*`.

use core::arch::wasm32::*;

use super::*;
use crate::{ColorMatrix, row::scalar};

// ---- u8 RGB / RGBA output -----------------------------------------------

/// wasm-simd128 V410 → packed u8 RGB or RGBA.
///
/// Byte-identical to `scalar::v410_to_rgb_or_rgba_row::<ALPHA>`.
///
/// # Safety
///
/// 1. **`simd128` must be enabled at compile time.**
/// 2. `packed.len() >= width`.
/// 3. `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn v410_to_rgb_or_rgba_row<const ALPHA: bool>(
  packed: &[u32],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(packed.len() >= width, "packed row too short");
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp, "out row too short");

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<10, 8>(full_range);
  let bias = scalar::chroma_bias::<10>();
  const RND: i32 = 1 << 14;

  unsafe {
    let rnd_v = i32x4_splat(RND);
    let y_off_v = i16x8_splat(y_off as i16);
    let y_scale_v = i32x4_splat(y_scale);
    let c_scale_v = i32x4_splat(c_scale);
    let bias_v = i16x8_splat(bias as i16);
    let zero4 = i32x4_splat(0);
    let cru = i32x4_splat(coeffs.r_u());
    let crv = i32x4_splat(coeffs.r_v());
    let cgu = i32x4_splat(coeffs.g_u());
    let cgv = i32x4_splat(coeffs.g_v());
    let cbu = i32x4_splat(coeffs.b_u());
    let cbv = i32x4_splat(coeffs.b_v());
    let mask = u32x4_splat(0x3FF);

    let mut x = 0usize;
    while x + 4 <= width {
      // Load 4 V410 words.
      let words = v128_load(packed.as_ptr().add(x).cast());

      // Extract U (bits 9:0), Y (bits 19:10), V (bits 29:20).
      let u_i32 = v128_and(words, mask);
      let y_i32 = v128_and(u32x4_shr(words, 10), mask);
      let v_i32 = v128_and(u32x4_shr(words, 20), mask);

      // Narrow i32x4 → i16x8: 4 valid lanes lo + 4 zero lanes hi.
      // Values ≤ 1023 fit in i16 with no saturation.
      let u_i16x8 = i16x8_narrow_i32x4(u_i32, zero4);
      let y_i16x8 = i16x8_narrow_i32x4(y_i32, zero4);
      let v_i16x8 = i16x8_narrow_i32x4(v_i32, zero4);

      // Chroma bias subtract.
      let u_sub = i16x8_sub(u_i16x8, bias_v);
      let v_sub = i16x8_sub(v_i16x8, bias_v);

      // Widen i16x8 → two i32x4 halves for Q15 multiply.
      let u_lo_i32 = i32x4_extend_low_i16x8(u_sub);
      let u_hi_i32 = i32x4_extend_high_i16x8(u_sub);
      let v_lo_i32 = i32x4_extend_low_i16x8(v_sub);
      let v_hi_i32 = i32x4_extend_high_i16x8(v_sub);

      let u_d_lo = q15_shift(i32x4_add(i32x4_mul(u_lo_i32, c_scale_v), rnd_v));
      let u_d_hi = q15_shift(i32x4_add(i32x4_mul(u_hi_i32, c_scale_v), rnd_v));
      let v_d_lo = q15_shift(i32x4_add(i32x4_mul(v_lo_i32, c_scale_v), rnd_v));
      let v_d_hi = q15_shift(i32x4_add(i32x4_mul(v_hi_i32, c_scale_v), rnd_v));

      // 8-lane i16 chroma vectors (lanes 0..3 valid; lanes 4..7 don't-care).
      let r_chroma = chroma_i16x8(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let g_chroma = chroma_i16x8(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let b_chroma = chroma_i16x8(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);

      // V410 is 4:4:4 — no chroma duplication needed; each pixel has its
      // own unique chroma triple. chroma lanes 0..3 align with Y 0..3.

      // Scale Y: `(Y - y_off) * y_scale + RND >> 15`. Values ≤ 1023 → safe i16.
      let y_scaled = scale_y(y_i16x8, y_off_v, y_scale_v, rnd_v);

      // Saturate-add and narrow to u8. Only low 4 lanes are valid.
      let r_sum = i16x8_add_sat(y_scaled, r_chroma);
      let g_sum = i16x8_add_sat(y_scaled, g_chroma);
      let b_sum = i16x8_add_sat(y_scaled, b_chroma);
      let r_u8 = u8x16_narrow_i16x8(r_sum, r_sum);
      let g_u8 = u8x16_narrow_i16x8(g_sum, g_sum);
      let b_u8 = u8x16_narrow_i16x8(b_sum, b_sum);

      // 4-pixel partial store via stack buffer.
      let mut r_tmp = [0u8; 16];
      let mut g_tmp = [0u8; 16];
      let mut b_tmp = [0u8; 16];
      v128_store(r_tmp.as_mut_ptr().cast(), r_u8);
      v128_store(g_tmp.as_mut_ptr().cast(), g_u8);
      v128_store(b_tmp.as_mut_ptr().cast(), b_u8);

      if ALPHA {
        let dst = &mut out[x * 4..x * 4 + 4 * 4];
        for i in 0..4 {
          dst[i * 4] = r_tmp[i];
          dst[i * 4 + 1] = g_tmp[i];
          dst[i * 4 + 2] = b_tmp[i];
          dst[i * 4 + 3] = 0xFF;
        }
      } else {
        let dst = &mut out[x * 3..x * 3 + 4 * 3];
        for i in 0..4 {
          dst[i * 3] = r_tmp[i];
          dst[i * 3 + 1] = g_tmp[i];
          dst[i * 3 + 2] = b_tmp[i];
        }
      }

      x += 4;
    }

    // Scalar tail — remaining < 4 pixels.
    if x < width {
      let tail_packed = &packed[x..width];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      scalar::v410_to_rgb_or_rgba_row::<ALPHA>(tail_packed, tail_out, tail_w, matrix, full_range);
    }
  }
}

// ---- u16 RGB / RGBA native-depth output ---------------------------------

/// wasm-simd128 V410 → packed native-depth u16 RGB or RGBA (low-bit-packed
/// at 10-bit).
///
/// Byte-identical to `scalar::v410_to_rgb_u16_or_rgba_u16_row::<ALPHA>`.
///
/// # Safety
///
/// 1. **`simd128` must be enabled at compile time.**
/// 2. `packed.len() >= width`.
/// 3. `out.len() >= width * (if ALPHA { 4 } else { 3 })` (u16 elements).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn v410_to_rgb_u16_or_rgba_u16_row<const ALPHA: bool>(
  packed: &[u32],
  out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(packed.len() >= width, "packed row too short");
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp, "out row too short");

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<10, 10>(full_range);
  let bias = scalar::chroma_bias::<10>();
  const RND: i32 = 1 << 14;
  let out_max: i16 = ((1i32 << 10) - 1) as i16;

  unsafe {
    let rnd_v = i32x4_splat(RND);
    let y_off_v = i16x8_splat(y_off as i16);
    let y_scale_v = i32x4_splat(y_scale);
    let c_scale_v = i32x4_splat(c_scale);
    let bias_v = i16x8_splat(bias as i16);
    let max_v = i16x8_splat(out_max);
    let zero_v = i16x8_splat(0);
    let zero4 = i32x4_splat(0);
    let cru = i32x4_splat(coeffs.r_u());
    let crv = i32x4_splat(coeffs.r_v());
    let cgu = i32x4_splat(coeffs.g_u());
    let cgv = i32x4_splat(coeffs.g_v());
    let cbu = i32x4_splat(coeffs.b_u());
    let cbv = i32x4_splat(coeffs.b_v());
    let mask = u32x4_splat(0x3FF);
    let alpha_u16 = out_max as u16;

    let mut x = 0usize;
    while x + 4 <= width {
      let words = v128_load(packed.as_ptr().add(x).cast());

      let u_i32 = v128_and(words, mask);
      let y_i32 = v128_and(u32x4_shr(words, 10), mask);
      let v_i32 = v128_and(u32x4_shr(words, 20), mask);

      // Narrow i32x4 → i16x8: 4 valid lo lanes + 4 zero hi lanes.
      let u_i16x8 = i16x8_narrow_i32x4(u_i32, zero4);
      let y_i16x8 = i16x8_narrow_i32x4(y_i32, zero4);
      let v_i16x8 = i16x8_narrow_i32x4(v_i32, zero4);

      let u_sub = i16x8_sub(u_i16x8, bias_v);
      let v_sub = i16x8_sub(v_i16x8, bias_v);

      let u_lo_i32 = i32x4_extend_low_i16x8(u_sub);
      let u_hi_i32 = i32x4_extend_high_i16x8(u_sub);
      let v_lo_i32 = i32x4_extend_low_i16x8(v_sub);
      let v_hi_i32 = i32x4_extend_high_i16x8(v_sub);

      let u_d_lo = q15_shift(i32x4_add(i32x4_mul(u_lo_i32, c_scale_v), rnd_v));
      let u_d_hi = q15_shift(i32x4_add(i32x4_mul(u_hi_i32, c_scale_v), rnd_v));
      let v_d_lo = q15_shift(i32x4_add(i32x4_mul(v_lo_i32, c_scale_v), rnd_v));
      let v_d_hi = q15_shift(i32x4_add(i32x4_mul(v_hi_i32, c_scale_v), rnd_v));

      let r_chroma = chroma_i16x8(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let g_chroma = chroma_i16x8(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let b_chroma = chroma_i16x8(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);

      let y_scaled = scale_y(y_i16x8, y_off_v, y_scale_v, rnd_v);

      // Clamp to [0, 0x3FF] (native 10-bit range).
      let r = clamp_u16_max_wasm(i16x8_add_sat(y_scaled, r_chroma), zero_v, max_v);
      let g = clamp_u16_max_wasm(i16x8_add_sat(y_scaled, g_chroma), zero_v, max_v);
      let b = clamp_u16_max_wasm(i16x8_add_sat(y_scaled, b_chroma), zero_v, max_v);

      // 4-pixel partial u16 store via stack buffer.
      let mut r_tmp = [0u16; 8];
      let mut g_tmp = [0u16; 8];
      let mut b_tmp = [0u16; 8];
      v128_store(r_tmp.as_mut_ptr().cast(), r);
      v128_store(g_tmp.as_mut_ptr().cast(), g);
      v128_store(b_tmp.as_mut_ptr().cast(), b);

      if ALPHA {
        let dst = &mut out[x * 4..x * 4 + 4 * 4];
        for i in 0..4 {
          dst[i * 4] = r_tmp[i];
          dst[i * 4 + 1] = g_tmp[i];
          dst[i * 4 + 2] = b_tmp[i];
          dst[i * 4 + 3] = alpha_u16;
        }
      } else {
        let dst = &mut out[x * 3..x * 3 + 4 * 3];
        for i in 0..4 {
          dst[i * 3] = r_tmp[i];
          dst[i * 3 + 1] = g_tmp[i];
          dst[i * 3 + 2] = b_tmp[i];
        }
      }

      x += 4;
    }

    // Scalar tail.
    if x < width {
      let tail_packed = &packed[x..width];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      scalar::v410_to_rgb_u16_or_rgba_u16_row::<ALPHA>(
        tail_packed,
        tail_out,
        tail_w,
        matrix,
        full_range,
      );
    }
  }
}

// ---- Luma u8 (4 px/iter) -----------------------------------------------

/// wasm-simd128 V410 → u8 luma. Y is `(word >> 10) & 0x3FF`, then `>> 2`.
///
/// Byte-identical to `scalar::v410_to_luma_row`.
///
/// # Safety
///
/// 1. **`simd128` must be enabled at compile time.**
/// 2. `packed.len() >= width`.
/// 3. `out.len() >= width`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn v410_to_luma_row(packed: &[u32], out: &mut [u8], width: usize) {
  debug_assert!(packed.len() >= width);
  debug_assert!(out.len() >= width);

  unsafe {
    let mask = u32x4_splat(0x3FF);
    let zero4 = i32x4_splat(0);

    let mut x = 0usize;
    while x + 4 <= width {
      let words = v128_load(packed.as_ptr().add(x).cast());
      // Y field: bits 19:10 → shift right 10, mask to 10-bit.
      let y_i32 = v128_and(u32x4_shr(words, 10), mask);
      // Narrow i32x4 → i16x8 (4 valid lo lanes + 4 zero hi lanes).
      let y_i16 = i16x8_narrow_i32x4(y_i32, zero4);
      // >> 2 → narrow to u8 via saturating narrow.
      let y_shr = u16x8_shr(y_i16, 2);
      let y_u8 = u8x16_narrow_i16x8(y_shr, y_shr);
      // Store 4 of the 16 lanes.
      let mut tmp = [0u8; 16];
      v128_store(tmp.as_mut_ptr().cast(), y_u8);
      out[x..x + 4].copy_from_slice(&tmp[..4]);
      x += 4;
    }

    // Scalar tail — remaining < 4 pixels.
    if x < width {
      scalar::v410_to_luma_row(&packed[x..width], &mut out[x..width], width - x);
    }
  }
}

// ---- Luma u16 (4 px/iter) -----------------------------------------------

/// wasm-simd128 V410 → u16 luma (low-bit-packed at 10-bit).
///
/// Byte-identical to `scalar::v410_to_luma_u16_row`.
///
/// # Safety
///
/// 1. **`simd128` must be enabled at compile time.**
/// 2. `packed.len() >= width`.
/// 3. `out.len() >= width`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn v410_to_luma_u16_row(packed: &[u32], out: &mut [u16], width: usize) {
  debug_assert!(packed.len() >= width);
  debug_assert!(out.len() >= width);

  unsafe {
    let mask = u32x4_splat(0x3FF);
    let zero4 = i32x4_splat(0);

    let mut x = 0usize;
    while x + 4 <= width {
      let words = v128_load(packed.as_ptr().add(x).cast());
      let y_i32 = v128_and(u32x4_shr(words, 10), mask);
      // Narrow i32x4 → i16x8: 4 valid lo lanes (values ≤ 1023, no saturation).
      let y_i16 = i16x8_narrow_i32x4(y_i32, zero4);
      // Store 4 u16 lanes via stack buffer.
      let mut tmp = [0u16; 8];
      v128_store(tmp.as_mut_ptr().cast(), y_i16);
      out[x..x + 4].copy_from_slice(&tmp[..4]);
      x += 4;
    }

    // Scalar tail.
    if x < width {
      scalar::v410_to_luma_u16_row(&packed[x..width], &mut out[x..width], width - x);
    }
  }
}
