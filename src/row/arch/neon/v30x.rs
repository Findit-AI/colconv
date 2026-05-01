//! NEON V30X (packed YUV 4:4:4, 10-bit, LSB-padded) kernels.
//!
//! ## Layout
//!
//! One `u32` per pixel: `bits[11:2]` = U, `bits[21:12]` = Y,
//! `bits[31:22]` = V (2 bits LSB-padding at bottom of each field).
//! No chroma subsampling (4:4:4) — each word yields a complete `(U, Y, V)`
//! triple. V30X is the LSB-padding sibling of V410 — the only difference
//! is that all three shift offsets are 2 greater (2/12/22 vs 0/10/20).
//!
//! ## Per-iter pipeline (4 px / 4 u32 / 16 bytes)
//!
//! Load one `uint32x4_t` → three `shift+AND` ops extract U / Y / V
//! fields. Narrow each to `uint16x4_t` via `vmovn_u32`, reinterpret
//! as `int16x4_t`.
//!
//! Combine with a zero-filled high-half to build `int16x8_t` operands
//! for `chroma_i16x8` / `scale_y`. Only the low 4 lanes carry valid
//! data; the high 4 are don't-care.
//!
//! ## Tail
//!
//! `width % 4` remaining pixels fall through to `scalar::v30x_*`.

use core::arch::aarch64::*;

use super::*;
use crate::{ColorMatrix, row::scalar};

// ---- u8 RGB / RGBA output -----------------------------------------------

/// NEON V30X → packed u8 RGB or RGBA.
///
/// Byte-identical to `scalar::v30x_to_rgb_or_rgba_row::<ALPHA>`.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `packed.len() >= width`.
/// 3. `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn v30x_to_rgb_or_rgba_row<const ALPHA: bool>(
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
    let rnd_v = vdupq_n_s32(RND);
    let y_off_v = vdupq_n_s16(y_off as i16);
    let y_scale_v = vdupq_n_s32(y_scale);
    let c_scale_v = vdupq_n_s32(c_scale);
    let bias_v = vdupq_n_s16(bias as i16);
    let cru = vdupq_n_s32(coeffs.r_u());
    let crv = vdupq_n_s32(coeffs.r_v());
    let cgu = vdupq_n_s32(coeffs.g_u());
    let cgv = vdupq_n_s32(coeffs.g_v());
    let cbu = vdupq_n_s32(coeffs.b_u());
    let cbv = vdupq_n_s32(coeffs.b_v());
    let mask = vdupq_n_u32(0x3FF);
    let zero4 = vdup_n_s16(0);

    let mut x = 0usize;
    while x + 4 <= width {
      // Load 4 V30X words.
      let words = vld1q_u32(packed.as_ptr().add(x));

      // Extract U (bits 11:2), Y (bits 21:12), V (bits 31:22).
      let u_u32 = vandq_u32(vshrq_n_u32::<2>(words), mask);
      let y_u32 = vandq_u32(vshrq_n_u32::<12>(words), mask);
      let v_u32 = vandq_u32(vshrq_n_u32::<22>(words), mask);

      // Narrow u32→u16, reinterpret as i16 (values ≤ 1023, safe).
      let u_i16x4 = vreinterpret_s16_u16(vmovn_u32(u_u32));
      let y_i16x4 = vreinterpret_s16_u16(vmovn_u32(y_u32));
      let v_i16x4 = vreinterpret_s16_u16(vmovn_u32(v_u32));

      // Combine into 8-lane vectors: low 4 valid, high 4 = zero don't-care.
      let u_i16x8 = vcombine_s16(u_i16x4, zero4);
      let y_i16x8 = vcombine_s16(y_i16x4, zero4);
      let v_i16x8 = vcombine_s16(v_i16x4, zero4);

      // Chroma bias subtract, then Q15 chroma scale.
      let u_sub = vsubq_s16(u_i16x8, bias_v);
      let v_sub = vsubq_s16(v_i16x8, bias_v);

      // Widen to i32x4 lo/hi for Q15 multiply.
      let u_lo_i32 = vmovl_s16(vget_low_s16(u_sub));
      let u_hi_i32 = vmovl_s16(vget_high_s16(u_sub));
      let v_lo_i32 = vmovl_s16(vget_low_s16(v_sub));
      let v_hi_i32 = vmovl_s16(vget_high_s16(v_sub));

      let u_d_lo = q15_shift(vaddq_s32(vmulq_s32(u_lo_i32, c_scale_v), rnd_v));
      let u_d_hi = q15_shift(vaddq_s32(vmulq_s32(u_hi_i32, c_scale_v), rnd_v));
      let v_d_lo = q15_shift(vaddq_s32(vmulq_s32(v_lo_i32, c_scale_v), rnd_v));
      let v_d_hi = q15_shift(vaddq_s32(vmulq_s32(v_hi_i32, c_scale_v), rnd_v));

      // Build 8-lane chroma contributions (lanes 4..7 are don't-care).
      let r_chroma = chroma_i16x8(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let g_chroma = chroma_i16x8(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let b_chroma = chroma_i16x8(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);

      // V30X is 4:4:4 — no chroma duplication needed; each pixel has its own
      // unique chroma triple. `r_chroma` lanes 0..3 already align with Y 0..3.

      // Scale Y: `(Y - y_off) * y_scale + RND >> 15`. Values ≤ 1023 → safe i16.
      let y_scaled = scale_y(y_i16x8, y_off_v, y_scale_v, rnd_v);

      // Saturate-add and narrow to u8. Only low 4 lanes are valid.
      let r_u8 = vqmovun_s16(vqaddq_s16(y_scaled, r_chroma));
      let g_u8 = vqmovun_s16(vqaddq_s16(y_scaled, g_chroma));
      let b_u8 = vqmovun_s16(vqaddq_s16(y_scaled, b_chroma));

      // 4-pixel partial store via stack buffer.
      if ALPHA {
        let alpha = vdup_n_u8(0xFF);
        let mut tmp = [0u8; 8 * 4];
        vst4_u8(tmp.as_mut_ptr(), uint8x8x4_t(r_u8, g_u8, b_u8, alpha));
        out[x * 4..x * 4 + 4 * 4].copy_from_slice(&tmp[..4 * 4]);
      } else {
        let mut tmp = [0u8; 8 * 3];
        vst3_u8(tmp.as_mut_ptr(), uint8x8x3_t(r_u8, g_u8, b_u8));
        out[x * 3..x * 3 + 4 * 3].copy_from_slice(&tmp[..4 * 3]);
      }

      x += 4;
    }

    // Scalar tail — remaining < 4 pixels.
    if x < width {
      let tail_packed = &packed[x..width];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      scalar::v30x_to_rgb_or_rgba_row::<ALPHA>(tail_packed, tail_out, tail_w, matrix, full_range);
    }
  }
}

// ---- u16 RGB / RGBA native-depth output ---------------------------------

/// NEON V30X → packed native-depth u16 RGB or RGBA (low-bit-packed at
/// 10-bit).
///
/// Byte-identical to `scalar::v30x_to_rgb_u16_or_rgba_u16_row::<ALPHA>`.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `packed.len() >= width`.
/// 3. `out.len() >= width * (if ALPHA { 4 } else { 3 })` (u16 elements).
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn v30x_to_rgb_u16_or_rgba_u16_row<const ALPHA: bool>(
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
    let rnd_v = vdupq_n_s32(RND);
    let y_off_v = vdupq_n_s16(y_off as i16);
    let y_scale_v = vdupq_n_s32(y_scale);
    let c_scale_v = vdupq_n_s32(c_scale);
    let bias_v = vdupq_n_s16(bias as i16);
    let max_v = vdupq_n_s16(out_max);
    let zero_v = vdupq_n_s16(0);
    let cru = vdupq_n_s32(coeffs.r_u());
    let crv = vdupq_n_s32(coeffs.r_v());
    let cgu = vdupq_n_s32(coeffs.g_u());
    let cgv = vdupq_n_s32(coeffs.g_v());
    let cbu = vdupq_n_s32(coeffs.b_u());
    let cbv = vdupq_n_s32(coeffs.b_v());
    let mask = vdupq_n_u32(0x3FF);
    let alpha_u16 = vdupq_n_u16(out_max as u16);
    let zero4 = vdup_n_s16(0);

    let mut x = 0usize;
    while x + 4 <= width {
      let words = vld1q_u32(packed.as_ptr().add(x));

      let u_u32 = vandq_u32(vshrq_n_u32::<2>(words), mask);
      let y_u32 = vandq_u32(vshrq_n_u32::<12>(words), mask);
      let v_u32 = vandq_u32(vshrq_n_u32::<22>(words), mask);

      let u_i16x4 = vreinterpret_s16_u16(vmovn_u32(u_u32));
      let y_i16x4 = vreinterpret_s16_u16(vmovn_u32(y_u32));
      let v_i16x4 = vreinterpret_s16_u16(vmovn_u32(v_u32));

      let u_i16x8 = vcombine_s16(u_i16x4, zero4);
      let y_i16x8 = vcombine_s16(y_i16x4, zero4);
      let v_i16x8 = vcombine_s16(v_i16x4, zero4);

      let u_sub = vsubq_s16(u_i16x8, bias_v);
      let v_sub = vsubq_s16(v_i16x8, bias_v);

      let u_lo_i32 = vmovl_s16(vget_low_s16(u_sub));
      let u_hi_i32 = vmovl_s16(vget_high_s16(u_sub));
      let v_lo_i32 = vmovl_s16(vget_low_s16(v_sub));
      let v_hi_i32 = vmovl_s16(vget_high_s16(v_sub));

      let u_d_lo = q15_shift(vaddq_s32(vmulq_s32(u_lo_i32, c_scale_v), rnd_v));
      let u_d_hi = q15_shift(vaddq_s32(vmulq_s32(u_hi_i32, c_scale_v), rnd_v));
      let v_d_lo = q15_shift(vaddq_s32(vmulq_s32(v_lo_i32, c_scale_v), rnd_v));
      let v_d_hi = q15_shift(vaddq_s32(vmulq_s32(v_hi_i32, c_scale_v), rnd_v));

      let r_chroma = chroma_i16x8(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let g_chroma = chroma_i16x8(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let b_chroma = chroma_i16x8(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);

      let y_scaled = scale_y(y_i16x8, y_off_v, y_scale_v, rnd_v);

      // Clamp to [0, 0x3FF] (native 10-bit range).
      let r = clamp_u16_max(vqaddq_s16(y_scaled, r_chroma), zero_v, max_v);
      let g = clamp_u16_max(vqaddq_s16(y_scaled, g_chroma), zero_v, max_v);
      let b = clamp_u16_max(vqaddq_s16(y_scaled, b_chroma), zero_v, max_v);

      // 4-pixel partial u16 store via stack buffer.
      if ALPHA {
        let mut tmp = [0u16; 8 * 4];
        vst4q_u16(tmp.as_mut_ptr(), uint16x8x4_t(r, g, b, alpha_u16));
        out[x * 4..x * 4 + 4 * 4].copy_from_slice(&tmp[..4 * 4]);
      } else {
        let mut tmp = [0u16; 8 * 3];
        vst3q_u16(tmp.as_mut_ptr(), uint16x8x3_t(r, g, b));
        out[x * 3..x * 3 + 4 * 3].copy_from_slice(&tmp[..4 * 3]);
      }

      x += 4;
    }

    // Scalar tail.
    if x < width {
      let tail_packed = &packed[x..width];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      scalar::v30x_to_rgb_u16_or_rgba_u16_row::<ALPHA>(
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

/// NEON V30X → u8 luma. Y is `(word >> 12) & 0x3FF`, then `>> 2`.
///
/// Byte-identical to `scalar::v30x_to_luma_row`.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `packed.len() >= width`.
/// 3. `out.len() >= width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn v30x_to_luma_row(packed: &[u32], out: &mut [u8], width: usize) {
  debug_assert!(packed.len() >= width);
  debug_assert!(out.len() >= width);

  unsafe {
    let mask = vdupq_n_u32(0x3FF);
    let mut x = 0usize;
    while x + 4 <= width {
      let words = vld1q_u32(packed.as_ptr().add(x));
      // Y field: bits 21:12 → shift right 12, mask to 10-bit.
      let y_u32 = vandq_u32(vshrq_n_u32::<12>(words), mask);
      // Narrow u32→u16, then >> 2, then narrow u16→u8.
      let y_u16 = vmovn_u32(y_u32);
      // y_u16 values are ≤ 1023; combine into dummy 8-lane for vshrn.
      let y_u16x8 = vcombine_u16(y_u16, vdup_n_u16(0));
      // vshrn_n_u16::<2> narrows (u16 >> 2) → u8x8; low 4 lanes valid.
      let y_u8 = vshrn_n_u16::<2>(y_u16x8);
      // Store 4 of the 8 lanes.
      let mut tmp = [0u8; 8];
      vst1_u8(tmp.as_mut_ptr(), y_u8);
      out[x..x + 4].copy_from_slice(&tmp[..4]);
      x += 4;
    }
    if x < width {
      scalar::v30x_to_luma_row(&packed[x..width], &mut out[x..width], width - x);
    }
  }
}

// ---- Luma u16 (4 px/iter) -----------------------------------------------

/// NEON V30X → u16 luma (low-bit-packed at 10-bit).
///
/// Byte-identical to `scalar::v30x_to_luma_u16_row`.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `packed.len() >= width`.
/// 3. `out.len() >= width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn v30x_to_luma_u16_row(packed: &[u32], out: &mut [u16], width: usize) {
  debug_assert!(packed.len() >= width);
  debug_assert!(out.len() >= width);

  unsafe {
    let mask = vdupq_n_u32(0x3FF);
    let mut x = 0usize;
    while x + 4 <= width {
      let words = vld1q_u32(packed.as_ptr().add(x));
      let y_u32 = vandq_u32(vshrq_n_u32::<12>(words), mask);
      // Narrow u32→u16 (values ≤ 1023, no saturation needed).
      let y_u16 = vmovn_u32(y_u32);
      // Store 4 lanes.
      let mut tmp = [0u16; 4];
      vst1_u16(tmp.as_mut_ptr(), y_u16);
      out[x..x + 4].copy_from_slice(&tmp[..4]);
      x += 4;
    }
    if x < width {
      scalar::v30x_to_luma_u16_row(&packed[x..width], &mut out[x..width], width - x);
    }
  }
}
