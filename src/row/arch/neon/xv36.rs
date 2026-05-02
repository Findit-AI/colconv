//! NEON kernels for XV36 packed YUV 4:4:4 12-bit family.
//!
//! ## Layout
//!
//! Four `u16` elements per pixel: `[U(16), Y(16), V(16), A(16)]`
//! little-endian, each holding a 12-bit sample MSB-aligned in the
//! high 12 bits (low 4 bits zero). The `X` prefix means the A slot
//! is **padding** — read by `vld4q_u16` but discarded. RGBA outputs
//! force α = max (`0xFF` u8 / `0x0FFF` u16).
//!
//! ## Per-iter pipeline (8 px / iter)
//!
//! `vld4q_u16` loads 8 quadruples in one call, returning a
//! `uint16x8x4_t` where `.0 = U`, `.1 = Y`, `.2 = V`, `.3 = A`
//! (padding). Each channel is right-shifted by 4 (`vshrq_n_u16::<4>`)
//! to bring the 12-bit value into `[0, 4095]`. No chroma duplication
//! needed — 4:4:4 means each pixel has its own U/V. Y values ≤ 4095
//! fit in i16, so `scale_y` is used (not `scale_y_u16_to_i16`).
//! The Q15 pipeline uses i32 chroma (`chroma_i16x8`) at BITS=12.
//!
//! ## Tail
//!
//! `width % 8` remaining pixels fall through to `scalar::xv36_*`.

use core::arch::aarch64::*;

use super::*;
use crate::{ColorMatrix, row::scalar};

// ---- u8 RGB / RGBA output -----------------------------------------------

/// NEON XV36 → packed u8 RGB or RGBA.
///
/// Byte-identical to `scalar::xv36_to_rgb_or_rgba_row::<ALPHA>`.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `packed.len() >= width * 4`.
/// 3. `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn xv36_to_rgb_or_rgba_row<const ALPHA: bool>(
  packed: &[u16],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(packed.len() >= width * 4, "packed row too short");
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp, "out row too short");

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<12, 8>(full_range);
  let bias = scalar::chroma_bias::<12>();
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

    let mut x = 0usize;
    while x + 8 <= width {
      // Load 8 XV36 quadruples (8 × 4 × u16 = 64 bytes).
      // vld4q_u16 deinterleaves: .0=U8, .1=Y8, .2=V8, .3=A8 (padding).
      let q = vld4q_u16(packed.as_ptr().add(x * 4));
      // Right-shift by 4 to drop the 4 padding LSBs → 12-bit range [0, 4095].
      let u_u16 = vshrq_n_u16::<4>(q.0); // 8 lanes of U
      let y_u16 = vshrq_n_u16::<4>(q.1); // 8 lanes of Y
      let v_u16 = vshrq_n_u16::<4>(q.2); // 8 lanes of V
      // q.3 (A) is padding — discarded.

      // Reinterpret as signed i16 (values ≤ 4095 < 32767, safe).
      let u_i16 = vreinterpretq_s16_u16(u_u16);
      let y_i16 = vreinterpretq_s16_u16(y_u16);
      let v_i16 = vreinterpretq_s16_u16(v_u16);

      // Subtract chroma bias (2048 for 12-bit).
      let u_sub = vsubq_s16(u_i16, bias_v);
      let v_sub = vsubq_s16(v_i16, bias_v);

      // Widen to i32x4 lo/hi for Q15 chroma-scale multiply.
      let u_lo_i32 = vmovl_s16(vget_low_s16(u_sub));
      let u_hi_i32 = vmovl_s16(vget_high_s16(u_sub));
      let v_lo_i32 = vmovl_s16(vget_low_s16(v_sub));
      let v_hi_i32 = vmovl_s16(vget_high_s16(v_sub));

      let u_d_lo = q15_shift(vaddq_s32(vmulq_s32(u_lo_i32, c_scale_v), rnd_v));
      let u_d_hi = q15_shift(vaddq_s32(vmulq_s32(u_hi_i32, c_scale_v), rnd_v));
      let v_d_lo = q15_shift(vaddq_s32(vmulq_s32(v_lo_i32, c_scale_v), rnd_v));
      let v_d_hi = q15_shift(vaddq_s32(vmulq_s32(v_hi_i32, c_scale_v), rnd_v));

      // 4:4:4 — no chroma duplication; all 8 lanes carry unique U/V per pixel.
      let r_chroma = chroma_i16x8(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let g_chroma = chroma_i16x8(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let b_chroma = chroma_i16x8(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);

      // Y values ≤ 4095 fit in i16; use scale_y (NOT scale_y_u16_to_i16).
      let y_scaled = scale_y(y_i16, y_off_v, y_scale_v, rnd_v);

      // Saturate-add Y + chroma, narrow to u8 with saturation.
      let r_u8 = vqmovun_s16(vqaddq_s16(y_scaled, r_chroma));
      let g_u8 = vqmovun_s16(vqaddq_s16(y_scaled, g_chroma));
      let b_u8 = vqmovun_s16(vqaddq_s16(y_scaled, b_chroma));

      // Store 8 pixels.
      let off = x * bpp;
      if ALPHA {
        let alpha = vdup_n_u8(0xFF);
        vst4_u8(out.as_mut_ptr().add(off), uint8x8x4_t(r_u8, g_u8, b_u8, alpha));
      } else {
        vst3_u8(out.as_mut_ptr().add(off), uint8x8x3_t(r_u8, g_u8, b_u8));
      }

      x += 8;
    }

    // Scalar tail — remaining < 8 pixels.
    if x < width {
      let tail_packed = &packed[x * 4..width * 4];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      scalar::xv36_to_rgb_or_rgba_row::<ALPHA>(tail_packed, tail_out, tail_w, matrix, full_range);
    }
  }
}

// ---- u16 RGB / RGBA native-depth output ---------------------------------

/// NEON XV36 → packed native-depth u16 RGB or RGBA (low-bit-packed at
/// 12-bit).
///
/// Byte-identical to `scalar::xv36_to_rgb_u16_or_rgba_u16_row::<ALPHA>`.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `packed.len() >= width * 4`.
/// 3. `out.len() >= width * (if ALPHA { 4 } else { 3 })` (u16 elements).
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn xv36_to_rgb_u16_or_rgba_u16_row<const ALPHA: bool>(
  packed: &[u16],
  out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(packed.len() >= width * 4, "packed row too short");
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp, "out row too short");

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<12, 12>(full_range);
  let bias = scalar::chroma_bias::<12>();
  const RND: i32 = 1 << 14;
  // 12-bit output max (low-bit-packed): [0, 0x0FFF].
  let out_max: i16 = 0x0FFF;
  let alpha_u16: u16 = 0x0FFF;

  unsafe {
    let rnd_v = vdupq_n_s32(RND);
    let y_off_v = vdupq_n_s16(y_off as i16);
    let y_scale_v = vdupq_n_s32(y_scale);
    let c_scale_v = vdupq_n_s32(c_scale);
    let bias_v = vdupq_n_s16(bias as i16);
    let max_v = vdupq_n_s16(out_max);
    let zero_v = vdupq_n_s16(0);
    let alpha_v = vdupq_n_u16(alpha_u16);
    let cru = vdupq_n_s32(coeffs.r_u());
    let crv = vdupq_n_s32(coeffs.r_v());
    let cgu = vdupq_n_s32(coeffs.g_u());
    let cgv = vdupq_n_s32(coeffs.g_v());
    let cbu = vdupq_n_s32(coeffs.b_u());
    let cbv = vdupq_n_s32(coeffs.b_v());

    let mut x = 0usize;
    while x + 8 <= width {
      let q = vld4q_u16(packed.as_ptr().add(x * 4));
      let u_u16 = vshrq_n_u16::<4>(q.0);
      let y_u16 = vshrq_n_u16::<4>(q.1);
      let v_u16 = vshrq_n_u16::<4>(q.2);
      // q.3 (A) is padding — discarded.

      let u_i16 = vreinterpretq_s16_u16(u_u16);
      let y_i16 = vreinterpretq_s16_u16(y_u16);
      let v_i16 = vreinterpretq_s16_u16(v_u16);

      let u_sub = vsubq_s16(u_i16, bias_v);
      let v_sub = vsubq_s16(v_i16, bias_v);

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

      let y_scaled = scale_y(y_i16, y_off_v, y_scale_v, rnd_v);

      // Clamp to [0, 0x0FFF] (12-bit low-bit-packed output range).
      let r = clamp_u16_max(vqaddq_s16(y_scaled, r_chroma), zero_v, max_v);
      let g = clamp_u16_max(vqaddq_s16(y_scaled, g_chroma), zero_v, max_v);
      let b = clamp_u16_max(vqaddq_s16(y_scaled, b_chroma), zero_v, max_v);

      // Store 8 pixels.
      let off = x * bpp;
      if ALPHA {
        vst4q_u16(out.as_mut_ptr().add(off), uint16x8x4_t(r, g, b, alpha_v));
      } else {
        vst3q_u16(out.as_mut_ptr().add(off), uint16x8x3_t(r, g, b));
      }

      x += 8;
    }

    // Scalar tail.
    if x < width {
      let tail_packed = &packed[x * 4..width * 4];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      scalar::xv36_to_rgb_u16_or_rgba_u16_row::<ALPHA>(
        tail_packed,
        tail_out,
        tail_w,
        matrix,
        full_range,
      );
    }
  }
}

// ---- Luma u8 (8 px/iter) -----------------------------------------------

/// NEON XV36 → u8 luma. Y is quadruple element 1; `>> 8` brings the
/// 12-bit MSB-aligned sample to 8-bit (drops 4 padding LSBs + 4 more).
///
/// Byte-identical to `scalar::xv36_to_luma_row`.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `packed.len() >= width * 4`.
/// 3. `out.len() >= width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn xv36_to_luma_row(packed: &[u16], out: &mut [u8], width: usize) {
  debug_assert!(packed.len() >= width * 4);
  debug_assert!(out.len() >= width);

  unsafe {
    let mut x = 0usize;
    while x + 8 <= width {
      let q = vld4q_u16(packed.as_ptr().add(x * 4));
      // Y is q.1. Scalar does `packed[x*4+1] >> 8`; apply the same shift.
      // vshrn_n_u16::<8> narrows (u16 >> 8) → u8x8, handling 8 lanes.
      let y_u8 = vshrn_n_u16::<8>(q.1);
      vst1_u8(out.as_mut_ptr().add(x), y_u8);
      x += 8;
    }
    // Scalar tail.
    if x < width {
      scalar::xv36_to_luma_row(&packed[x * 4..width * 4], &mut out[x..width], width - x);
    }
  }
}

// ---- Luma u16 (8 px/iter) -----------------------------------------------

/// NEON XV36 → u16 luma (low-bit-packed at 12-bit). Y is quadruple
/// element 1; `>> 4` drops the 4 padding LSBs to give a 12-bit value
/// in `[0, 4095]`.
///
/// Byte-identical to `scalar::xv36_to_luma_u16_row`.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `packed.len() >= width * 4`.
/// 3. `out.len() >= width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn xv36_to_luma_u16_row(packed: &[u16], out: &mut [u16], width: usize) {
  debug_assert!(packed.len() >= width * 4);
  debug_assert!(out.len() >= width);

  unsafe {
    let mut x = 0usize;
    while x + 8 <= width {
      let q = vld4q_u16(packed.as_ptr().add(x * 4));
      // Y is q.1. Scalar does `packed[x*4+1] >> 4`.
      let y_u16 = vshrq_n_u16::<4>(q.1);
      vst1q_u16(out.as_mut_ptr().add(x), y_u16);
      x += 8;
    }
    // Scalar tail.
    if x < width {
      scalar::xv36_to_luma_u16_row(&packed[x * 4..width * 4], &mut out[x..width], width - x);
    }
  }
}
