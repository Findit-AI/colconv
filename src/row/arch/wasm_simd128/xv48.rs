//! wasm-simd128 kernels for XV48 packed YUV 4:4:4 16-bit
//! (FFmpeg `AV_PIX_FMT_XV48LE`).
//!
//! ## Layout
//!
//! Four `u16` elements per pixel: `[U(16), Y(16), V(16), X(16)]`, each
//! holding a full 16-bit sample (no padding bits, no right-shift on
//! load — the full-depth sibling of XV36). The `X` slot is **padding** —
//! loaded but discarded. RGBA outputs force α = max.
//!
//! In memory (little-endian, 8 bytes per pixel):
//! ```text
//!   byte 0,1 = U   byte 2,3 = Y   byte 4,5 = V   byte 6,7 = X
//! ```
//!
//! ## u8 pipeline (16 px / iter)
//!
//! Two 8-pixel half-iterations per SIMD block. Chroma bias via wrapping
//! 0x8000 i16 trick; Q15 pipeline. Y scaled via `scale_y_u16_wasm`
//! (unsigned widening — Y > 32767 must not be sign-extended).
//!
//! ## u16 pipeline (8 px / iter)
//!
//! `chroma_i64x2_wasm` (i64 chroma) + `scale_y_i32x4_i64_wasm` to avoid
//! overflow at BITS=16/16.
//!
//! ## Tail
//!
//! `width % block_size` remaining pixels fall through to `scalar::xv48_*`.

use core::arch::wasm32::*;

use super::{endian, *};
use crate::{KernelMatrix, row::scalar};

// ---- Deinterleave helper ------------------------------------------------

/// Deinterleaves 8 XV48 quadruples (32 u16 = 64 bytes) from `ptr` into
/// `(u_vec, y_vec, v_vec)` — three `v128` vectors each holding 8 `u16`
/// samples (full 16-bit native — no shift). The X channel (slot 3) is
/// not extracted.
///
/// Channel slot order in memory: U=bytes 0,1; Y=2,3; V=4,5; X=6,7.
///
/// # Safety
///
/// `ptr` must point to at least 64 readable bytes (32 `u16` elements).
/// `simd128` must be enabled at compile time.
#[inline]
#[target_feature(enable = "simd128")]
unsafe fn deinterleave_xv48_8px<const BE: bool>(ptr: *const u16) -> (v128, v128, v128) {
  unsafe {
    // Load 4 × v128, each covering 2 pixels (8 × u16 = 16 bytes).
    let raw0 = endian::load_endian_u16x8::<BE>(ptr as *const u8); // [U0,Y0,V0,X0, U1,Y1,V1,X1]
    let raw1 = endian::load_endian_u16x8::<BE>(ptr.add(8) as *const u8); // [U2..X2, U3..X3]
    let raw2 = endian::load_endian_u16x8::<BE>(ptr.add(16) as *const u8); // [U4..X4, U5..X5]
    let raw3 = endian::load_endian_u16x8::<BE>(ptr.add(24) as *const u8); // [U6..X6, U7..X7]

    // Per-channel byte positions within a 2-pixel v128 (16 bytes):
    //   U → bytes  0,1  (pixel n) and  8,9  (pixel n+1)
    //   Y → bytes  2,3            and 10,11
    //   V → bytes  4,5            and 12,13
    //   X → bytes  6,7            and 14,15 (discarded)
    let u_idx = i8x16(0, 1, 8, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let y_idx = i8x16(2, 3, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let v_idx = i8x16(4, 5, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);

    let u0 = u8x16_swizzle(raw0, u_idx);
    let u1 = u8x16_swizzle(raw1, u_idx);
    let u2 = u8x16_swizzle(raw2, u_idx);
    let u3 = u8x16_swizzle(raw3, u_idx);

    let y0 = u8x16_swizzle(raw0, y_idx);
    let y1 = u8x16_swizzle(raw1, y_idx);
    let y2 = u8x16_swizzle(raw2, y_idx);
    let y3 = u8x16_swizzle(raw3, y_idx);

    let v0 = u8x16_swizzle(raw0, v_idx);
    let v1 = u8x16_swizzle(raw1, v_idx);
    let v2 = u8x16_swizzle(raw2, v_idx);
    let v3 = u8x16_swizzle(raw3, v_idx);

    // Level 1: concatenate pairs of 4-byte fragments into 8-byte fragments.
    let u01 = i8x16_shuffle::<0, 1, 2, 3, 16, 17, 18, 19, 0, 0, 0, 0, 0, 0, 0, 0>(u0, u1);
    let u23 = i8x16_shuffle::<0, 1, 2, 3, 16, 17, 18, 19, 0, 0, 0, 0, 0, 0, 0, 0>(u2, u3);
    let y01 = i8x16_shuffle::<0, 1, 2, 3, 16, 17, 18, 19, 0, 0, 0, 0, 0, 0, 0, 0>(y0, y1);
    let y23 = i8x16_shuffle::<0, 1, 2, 3, 16, 17, 18, 19, 0, 0, 0, 0, 0, 0, 0, 0>(y2, y3);
    let v01 = i8x16_shuffle::<0, 1, 2, 3, 16, 17, 18, 19, 0, 0, 0, 0, 0, 0, 0, 0>(v0, v1);
    let v23 = i8x16_shuffle::<0, 1, 2, 3, 16, 17, 18, 19, 0, 0, 0, 0, 0, 0, 0, 0>(v2, v3);

    // Level 2: combine two 8-byte fragments into full 8-lane u16x8 vectors.
    let u_vec = i8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23>(u01, u23);
    let y_vec = i8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23>(y01, y23);
    let v_vec = i8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23>(v01, v23);

    (u_vec, y_vec, v_vec)
  }
}

// ---- u8 RGB / RGBA output (16 px/iter) ----------------------------------

/// wasm-simd128 XV48 → packed u8 RGB or RGBA. 16 pixels per iteration.
///
/// Byte-identical to `scalar::xv48_to_rgb_or_rgba_row::<ALPHA, BE>`.
///
/// # Safety
///
/// 1. **`simd128` must be enabled at compile time.**
/// 2. `packed.len() >= width * 4`.
/// 3. `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn xv48_to_rgb_or_rgba_row<const ALPHA: bool, const BE: bool>(
  packed: &[u16],
  out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  debug_assert!(packed.len() >= width * 4, "packed row too short");
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp, "out row too short");

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<16, 8>(full_range);
  const RND: i32 = 1 << 14;

  unsafe {
    let rnd_v = i32x4_splat(RND);
    let y_off32_v = i32x4_splat(y_off);
    let y_scale_v = i32x4_splat(y_scale);
    let c_scale_v = i32x4_splat(c_scale);
    let bias16_v = i16x8_splat(-32768i16);
    let cru = i32x4_splat(coeffs.r_u());
    let crv = i32x4_splat(coeffs.r_v());
    let cgu = i32x4_splat(coeffs.g_u());
    let cgv = i32x4_splat(coeffs.g_v());
    let cbu = i32x4_splat(coeffs.b_u());
    let cbv = i32x4_splat(coeffs.b_v());

    let mut x = 0usize;
    while x + 16 <= width {
      // --- lo half: pixels x..x+7 -------------------------------------------
      let (u_lo_u16, y_lo_u16, v_lo_u16) = deinterleave_xv48_8px::<BE>(packed.as_ptr().add(x * 4));

      let u_lo_i16 = i16x8_sub(u_lo_u16, bias16_v);
      let v_lo_i16 = i16x8_sub(v_lo_u16, bias16_v);

      let u_lo_a = i32x4_extend_low_i16x8(u_lo_i16);
      let u_lo_b = i32x4_extend_high_i16x8(u_lo_i16);
      let v_lo_a = i32x4_extend_low_i16x8(v_lo_i16);
      let v_lo_b = i32x4_extend_high_i16x8(v_lo_i16);

      let u_d_lo_a = q15_shift(i32x4_add(i32x4_mul(u_lo_a, c_scale_v), rnd_v));
      let u_d_lo_b = q15_shift(i32x4_add(i32x4_mul(u_lo_b, c_scale_v), rnd_v));
      let v_d_lo_a = q15_shift(i32x4_add(i32x4_mul(v_lo_a, c_scale_v), rnd_v));
      let v_d_lo_b = q15_shift(i32x4_add(i32x4_mul(v_lo_b, c_scale_v), rnd_v));

      let r_chroma_lo = chroma_i16x8(cru, crv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let g_chroma_lo = chroma_i16x8(cgu, cgv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let b_chroma_lo = chroma_i16x8(cbu, cbv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);

      // Y: unsigned-widen u16 → i32, scale. Returns i16x8.
      let y_lo_scaled = scale_y_u16_wasm(y_lo_u16, y_off32_v, y_scale_v, rnd_v);

      let r_lo_u8 = u8x16_narrow_i16x8(i16x8_add_sat(y_lo_scaled, r_chroma_lo), i16x8_splat(0));
      let g_lo_u8 = u8x16_narrow_i16x8(i16x8_add_sat(y_lo_scaled, g_chroma_lo), i16x8_splat(0));
      let b_lo_u8 = u8x16_narrow_i16x8(i16x8_add_sat(y_lo_scaled, b_chroma_lo), i16x8_splat(0));

      // --- hi half: pixels x+8..x+15 ----------------------------------------
      let (u_hi_u16, y_hi_u16, v_hi_u16) =
        deinterleave_xv48_8px::<BE>(packed.as_ptr().add(x * 4 + 32));

      let u_hi_i16 = i16x8_sub(u_hi_u16, bias16_v);
      let v_hi_i16 = i16x8_sub(v_hi_u16, bias16_v);

      let u_hi_a = i32x4_extend_low_i16x8(u_hi_i16);
      let u_hi_b = i32x4_extend_high_i16x8(u_hi_i16);
      let v_hi_a = i32x4_extend_low_i16x8(v_hi_i16);
      let v_hi_b = i32x4_extend_high_i16x8(v_hi_i16);

      let u_d_hi_a = q15_shift(i32x4_add(i32x4_mul(u_hi_a, c_scale_v), rnd_v));
      let u_d_hi_b = q15_shift(i32x4_add(i32x4_mul(u_hi_b, c_scale_v), rnd_v));
      let v_d_hi_a = q15_shift(i32x4_add(i32x4_mul(v_hi_a, c_scale_v), rnd_v));
      let v_d_hi_b = q15_shift(i32x4_add(i32x4_mul(v_hi_b, c_scale_v), rnd_v));

      let r_chroma_hi = chroma_i16x8(cru, crv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let g_chroma_hi = chroma_i16x8(cgu, cgv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let b_chroma_hi = chroma_i16x8(cbu, cbv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);

      let y_hi_scaled = scale_y_u16_wasm(y_hi_u16, y_off32_v, y_scale_v, rnd_v);

      let r_hi_u8 = u8x16_narrow_i16x8(i16x8_add_sat(y_hi_scaled, r_chroma_hi), i16x8_splat(0));
      let g_hi_u8 = u8x16_narrow_i16x8(i16x8_add_sat(y_hi_scaled, g_chroma_hi), i16x8_splat(0));
      let b_hi_u8 = u8x16_narrow_i16x8(i16x8_add_sat(y_hi_scaled, b_chroma_hi), i16x8_splat(0));

      // Combine lo+hi 8-byte halves into full 16-byte channel vectors.
      let r_u8 =
        i8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23>(r_lo_u8, r_hi_u8);
      let g_u8 =
        i8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23>(g_lo_u8, g_hi_u8);
      let b_u8 =
        i8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23>(b_lo_u8, b_hi_u8);

      let out_ptr = out.as_mut_ptr().add(x * bpp);
      if ALPHA {
        // X slot is padding — RGBA forces α = 0xFF.
        let a_vec: v128 = u8x16_splat(0xFF);
        write_rgba_16(r_u8, g_u8, b_u8, a_vec, out_ptr);
      } else {
        write_rgb_16(r_u8, g_u8, b_u8, out_ptr);
      }

      x += 16;
    }

    // Scalar tail — remaining < 16 pixels.
    if x < width {
      let tail_packed = &packed[x * 4..width * 4];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      scalar::xv48_to_rgb_or_rgba_row::<ALPHA, BE>(
        tail_packed,
        tail_out,
        tail_w,
        matrix,
        full_range,
      );
    }
  }
}

// ---- thin wrappers (u8 output) -------------------------------------------

/// wasm-simd128 XV48 → packed **RGB** (3 bpp). X slot dropped.
///
/// # Safety
///
/// 1. **`simd128` must be enabled at compile time.**
/// 2. `packed.len() >= width * 4`.
/// 3. `out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn xv48_to_rgb_row<const BE: bool>(
  packed: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  unsafe {
    xv48_to_rgb_or_rgba_row::<false, BE>(packed, rgb_out, width, matrix, full_range);
  }
}

// ---- u16 RGB / RGBA native-depth output (8 px/iter) ----------------------

/// wasm-simd128 XV48 → packed native-depth u16 RGB or RGBA.
///
/// Uses i64 chroma (`chroma_i64x2_wasm`) to avoid overflow at BITS=16/16.
/// Byte-identical to `scalar::xv48_to_rgb_u16_or_rgba_u16_row::<ALPHA, BE>`.
///
/// # Safety
///
/// 1. **`simd128` must be enabled at compile time.**
/// 2. `packed.len() >= width * 4`.
/// 3. `out.len() >= width * (if ALPHA { 4 } else { 3 })` (u16 elements).
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn xv48_to_rgb_u16_or_rgba_u16_row<const ALPHA: bool, const BE: bool>(
  packed: &[u16],
  out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  debug_assert!(packed.len() >= width * 4, "packed row too short");
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp, "out row too short");

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<16, 16>(full_range);
  const RND_I64: i64 = 1 << 14;
  const RND_I32: i32 = 1 << 14;

  unsafe {
    let alpha_u16 = u16x8_splat(0xFFFF);
    let rnd_i64 = i64x2_splat(RND_I64);
    let rnd_i32 = i32x4_splat(RND_I32);
    let y_off32 = i32x4_splat(y_off);
    let y_scale_i64 = i64x2_splat(y_scale as i64);
    let c_scale_i32 = i32x4_splat(c_scale);
    let bias16 = i16x8_splat(-32768i16);
    let cru = i64x2_extend_low_i32x4(i32x4_splat(coeffs.r_u()));
    let crv = i64x2_extend_low_i32x4(i32x4_splat(coeffs.r_v()));
    let cgu = i64x2_extend_low_i32x4(i32x4_splat(coeffs.g_u()));
    let cgv = i64x2_extend_low_i32x4(i32x4_splat(coeffs.g_v()));
    let cbu = i64x2_extend_low_i32x4(i32x4_splat(coeffs.b_u()));
    let cbv = i64x2_extend_low_i32x4(i32x4_splat(coeffs.b_v()));

    let mut x = 0usize;
    while x + 8 <= width {
      // Deinterleave 8 XV48 quadruples → U, Y, V as u16x8.
      let (u_u16, y_vec, v_u16) = deinterleave_xv48_8px::<BE>(packed.as_ptr().add(x * 4));

      let u_i16 = i16x8_sub(u_u16, bias16);
      let v_i16 = i16x8_sub(v_u16, bias16);

      let u_lo_i32 = i32x4_extend_low_i16x8(u_i16);
      let v_lo_i32 = i32x4_extend_low_i16x8(v_i16);
      let u_d_lo = i32x4_shr(i32x4_add(i32x4_mul(u_lo_i32, c_scale_i32), rnd_i32), 15);
      let v_d_lo = i32x4_shr(i32x4_add(i32x4_mul(v_lo_i32, c_scale_i32), rnd_i32), 15);

      let u_hi_i32 = i32x4_extend_high_i16x8(u_i16);
      let v_hi_i32 = i32x4_extend_high_i16x8(v_i16);
      let u_d_hi = i32x4_shr(i32x4_add(i32x4_mul(u_hi_i32, c_scale_i32), rnd_i32), 15);
      let v_d_hi = i32x4_shr(i32x4_add(i32x4_mul(v_hi_i32, c_scale_i32), rnd_i32), 15);

      let u_d_lo_lo = i64x2_extend_low_i32x4(u_d_lo);
      let u_d_lo_hi = i64x2_extend_high_i32x4(u_d_lo);
      let v_d_lo_lo = i64x2_extend_low_i32x4(v_d_lo);
      let v_d_lo_hi = i64x2_extend_high_i32x4(v_d_lo);
      let u_d_hi_lo = i64x2_extend_low_i32x4(u_d_hi);
      let u_d_hi_hi = i64x2_extend_high_i32x4(u_d_hi);
      let v_d_hi_lo = i64x2_extend_low_i32x4(v_d_hi);
      let v_d_hi_hi = i64x2_extend_high_i32x4(v_d_hi);

      let r_ch_lo_lo = chroma_i64x2_wasm(cru, crv, u_d_lo_lo, v_d_lo_lo, rnd_i64);
      let r_ch_lo_hi = chroma_i64x2_wasm(cru, crv, u_d_lo_hi, v_d_lo_hi, rnd_i64);
      let g_ch_lo_lo = chroma_i64x2_wasm(cgu, cgv, u_d_lo_lo, v_d_lo_lo, rnd_i64);
      let g_ch_lo_hi = chroma_i64x2_wasm(cgu, cgv, u_d_lo_hi, v_d_lo_hi, rnd_i64);
      let b_ch_lo_lo = chroma_i64x2_wasm(cbu, cbv, u_d_lo_lo, v_d_lo_lo, rnd_i64);
      let b_ch_lo_hi = chroma_i64x2_wasm(cbu, cbv, u_d_lo_hi, v_d_lo_hi, rnd_i64);

      let r_ch_hi_lo = chroma_i64x2_wasm(cru, crv, u_d_hi_lo, v_d_hi_lo, rnd_i64);
      let r_ch_hi_hi = chroma_i64x2_wasm(cru, crv, u_d_hi_hi, v_d_hi_hi, rnd_i64);
      let g_ch_hi_lo = chroma_i64x2_wasm(cgu, cgv, u_d_hi_lo, v_d_hi_lo, rnd_i64);
      let g_ch_hi_hi = chroma_i64x2_wasm(cgu, cgv, u_d_hi_hi, v_d_hi_hi, rnd_i64);
      let b_ch_hi_lo = chroma_i64x2_wasm(cbu, cbv, u_d_hi_lo, v_d_hi_lo, rnd_i64);
      let b_ch_hi_hi = chroma_i64x2_wasm(cbu, cbv, u_d_hi_hi, v_d_hi_hi, rnd_i64);

      // Combine each i64x2 pair → i32x4 low-32-bits.
      let r_ch_lo_i32 = combine_i64x2_pair_to_i32x4(r_ch_lo_lo, r_ch_lo_hi);
      let g_ch_lo_i32 = combine_i64x2_pair_to_i32x4(g_ch_lo_lo, g_ch_lo_hi);
      let b_ch_lo_i32 = combine_i64x2_pair_to_i32x4(b_ch_lo_lo, b_ch_lo_hi);
      let r_ch_hi_i32 = combine_i64x2_pair_to_i32x4(r_ch_hi_lo, r_ch_hi_hi);
      let g_ch_hi_i32 = combine_i64x2_pair_to_i32x4(g_ch_hi_lo, g_ch_hi_hi);
      let b_ch_hi_i32 = combine_i64x2_pair_to_i32x4(b_ch_hi_lo, b_ch_hi_hi);

      // Y: unsigned-widen u16 → i32, subtract y_off, scale via i64.
      let y_lo_u32 = u32x4_extend_low_u16x8(y_vec);
      let y_hi_u32 = u32x4_extend_high_u16x8(y_vec);
      let y_lo_i32 = i32x4_sub(y_lo_u32, y_off32);
      let y_hi_i32 = i32x4_sub(y_hi_u32, y_off32);

      let y_lo_scaled = scale_y_i32x4_i64_wasm(y_lo_i32, y_scale_i64, rnd_i64);
      let y_hi_scaled = scale_y_i32x4_i64_wasm(y_hi_i32, y_scale_i64, rnd_i64);

      // Add Y + chroma, saturate i32 → u16 via u16x8_narrow_i32x4.
      let r_u16 = u16x8_narrow_i32x4(
        i32x4_add(y_lo_scaled, r_ch_lo_i32),
        i32x4_add(y_hi_scaled, r_ch_hi_i32),
      );
      let g_u16 = u16x8_narrow_i32x4(
        i32x4_add(y_lo_scaled, g_ch_lo_i32),
        i32x4_add(y_hi_scaled, g_ch_hi_i32),
      );
      let b_u16 = u16x8_narrow_i32x4(
        i32x4_add(y_lo_scaled, b_ch_lo_i32),
        i32x4_add(y_hi_scaled, b_ch_hi_i32),
      );

      if ALPHA {
        // X slot is padding — RGBA forces α = 0xFFFF.
        write_rgba_u16_8(r_u16, g_u16, b_u16, alpha_u16, out.as_mut_ptr().add(x * 4));
      } else {
        write_rgb_u16_8(r_u16, g_u16, b_u16, out.as_mut_ptr().add(x * 3));
      }

      x += 8;
    }

    // Scalar tail — remaining < 8 pixels.
    if x < width {
      let tail_packed = &packed[x * 4..width * 4];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      scalar::xv48_to_rgb_u16_or_rgba_u16_row::<ALPHA, BE>(
        tail_packed,
        tail_out,
        tail_w,
        matrix,
        full_range,
      );
    }
  }
}

// ---- Luma u8 (16 px/iter) -----------------------------------------------

/// wasm-simd128 XV48 → u8 luma. Y is slot 1 (bytes 2,3); `>> 8` extracts
/// the high byte. Uses two deinterleave calls per 16-pixel block.
///
/// Byte-identical to `scalar::xv48_to_luma_row`.
///
/// # Safety
///
/// 1. **`simd128` must be enabled at compile time.**
/// 2. `packed.len() >= width * 4`.
/// 3. `luma_out.len() >= width`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn xv48_to_luma_row<const BE: bool>(
  packed: &[u16],
  luma_out: &mut [u8],
  width: usize,
) {
  debug_assert!(packed.len() >= width * 4, "packed row too short");
  debug_assert!(luma_out.len() >= width, "luma row too short");

  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      let (_u_lo, y_lo, _v_lo) = deinterleave_xv48_8px::<BE>(packed.as_ptr().add(x * 4));
      let (_u_hi, y_hi, _v_hi) = deinterleave_xv48_8px::<BE>(packed.as_ptr().add(x * 4 + 32));

      // >> 8 to get u8 luma. Logical shift (u16x8_shr) — arithmetic shift
      // would sign-extend Y ≥ 0x8000 to a negative i16, which
      // u8x16_narrow_i16x8 would then saturate to 0, corrupting half the
      // luma range.
      let y_lo_shr = u16x8_shr(y_lo, 8);
      let y_hi_shr = u16x8_shr(y_hi, 8);
      let y_u8 = u8x16_narrow_i16x8(y_lo_shr, y_hi_shr);
      v128_store(luma_out.as_mut_ptr().add(x).cast(), y_u8);

      x += 16;
    }

    // Scalar tail.
    if x < width {
      scalar::xv48_to_luma_row::<BE>(
        &packed[x * 4..width * 4],
        &mut luma_out[x..width],
        width - x,
      );
    }
  }
}

// ---- Luma u16 (16 px/iter) -----------------------------------------------

/// wasm-simd128 XV48 → u16 luma. Direct copy of Y samples (slot 1, no
/// shift — 16-bit native). Uses two deinterleave calls per 16-pixel block.
///
/// Byte-identical to `scalar::xv48_to_luma_u16_row`.
///
/// # Safety
///
/// 1. **`simd128` must be enabled at compile time.**
/// 2. `packed.len() >= width * 4`.
/// 3. `luma_out.len() >= width`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn xv48_to_luma_u16_row<const BE: bool>(
  packed: &[u16],
  luma_out: &mut [u16],
  width: usize,
) {
  debug_assert!(packed.len() >= width * 4, "packed row too short");
  debug_assert!(luma_out.len() >= width, "luma row too short");

  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      let (_u_lo, y_lo, _v_lo) = deinterleave_xv48_8px::<BE>(packed.as_ptr().add(x * 4));
      let (_u_hi, y_hi, _v_hi) = deinterleave_xv48_8px::<BE>(packed.as_ptr().add(x * 4 + 32));

      // Direct copy — Y samples are 16-bit native (no shift needed).
      v128_store(luma_out.as_mut_ptr().add(x).cast(), y_lo);
      v128_store(luma_out.as_mut_ptr().add(x + 8).cast(), y_hi);

      x += 16;
    }

    // Scalar tail.
    if x < width {
      scalar::xv48_to_luma_u16_row::<BE>(
        &packed[x * 4..width * 4],
        &mut luma_out[x..width],
        width - x,
      );
    }
  }
}

// ---- XV48 → HSV (staged via a reused 8-bit RGB chunk) ----------------
//
// The SIMD twin of the scalar `xv48_to_hsv_row` kernel — fills a small
// reused 8-bit RGB scratch via the production `xv48_to_rgb_row` kernel,
// then runs `rgb_to_hsv_row`. The X slot is dropped (HSV is colour-only).

/// One reused 8-bit RGB chunk's worth of pixels staged before the HSV
/// pass.
const HSV_CHUNK: usize = 64;

/// Shared SIMD driver: walks `width` in `HSV_CHUNK`-pixel chunks, fills a
/// small reused stack RGB scratch via `fill_rgb`, then runs the SIMD
/// [`rgb_to_hsv_row`] on that chunk into the H/S/V planes.
///
/// `fill_rgb` receives `(offset, n, &mut rgb_chunk)` and must write
/// `n * 3` packed RGB bytes for the `n` pixels at `offset`.
///
/// # Safety
///
/// The SIMD feature must be available, and `fill_rgb` must uphold the
/// underlying RGB kernel's safety contract for each chunk. Each of
/// `h_out` / `s_out` / `v_out` must be `>= width`.
#[inline]
unsafe fn xv48_hsv_via_rgb_chunks(
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  mut fill_rgb: impl FnMut(usize, usize, &mut [u8]),
) {
  let mut scratch = [0u8; HSV_CHUNK * 3];
  let mut offset = 0;
  while offset < width {
    let n = (width - offset).min(HSV_CHUNK);
    fill_rgb(offset, n, &mut scratch[..n * 3]);
    // SAFETY: the SIMD feature is verified by the wrapper's
    // `#[target_feature]`; the chunk and the output sub-slices are all
    // length `n`.
    unsafe {
      rgb_to_hsv_row(
        &scratch[..n * 3],
        &mut h_out[offset..offset + n],
        &mut s_out[offset..offset + n],
        &mut v_out[offset..offset + n],
        n,
      );
    }
    offset += n;
  }
}

/// SIMD: XV48 (packed 4:4:4, 16-bit) → planar HSV bytes (OpenCV
/// encoding), staged via the reused-8-bit-RGB-chunk pattern over the
/// SIMD [`xv48_to_rgb_row`] + [`rgb_to_hsv_row`]. Byte-identical to
/// `rgb_to_hsv_row(xv48_to_rgb_or_rgba_row::<false, BE>(...))` within
/// this SIMD tier. The padding X slot is dropped (HSV is colour-only).
///
/// # Safety
///
/// 1. The SIMD feature must be available.
/// 2. `packed.len() >= width * 4`.
/// 3. `h_out.len()`, `s_out.len()`, `v_out.len()` `>= width`.
#[inline]
#[target_feature(enable = "simd128")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn xv48_to_hsv_row<const BE: bool>(
  packed: &[u16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  debug_assert!(packed.len() >= width * 4, "packed row too short");
  debug_assert!(h_out.len() >= width, "h_out row too short");
  debug_assert!(s_out.len() >= width, "s_out row too short");
  debug_assert!(v_out.len() >= width, "v_out row too short");

  // SAFETY: the feature is the caller's obligation; the chunk filler
  // forwards the per-chunk sub-slices to the SIMD XV48 RGB kernel under
  // the same contract (its own scalar tail covers small n).
  unsafe {
    xv48_hsv_via_rgb_chunks(h_out, s_out, v_out, width, |offset, n, rgb| {
      xv48_to_rgb_row::<BE>(&packed[offset * 4..], rgb, n, matrix, full_range);
    });
  }
}
