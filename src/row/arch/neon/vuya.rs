#![allow(dead_code)] // dispatcher (Tasks 10-13) not yet wired
//! NEON kernels for VUYA / VUYX packed YUV 4:4:4 8-bit family.
//!
//! ## Layout
//!
//! Four `u8` elements per pixel: `V(8) ‖ U(8) ‖ Y(8) ‖ A(8)`.
//! VUYA carries a real alpha channel in byte 3. VUYX treats byte 3 as
//! padding and forces output α to `0xFF`.
//!
//! ## Per-iter pipeline (16 px / iter)
//!
//! `vld4q_u8` loads 16 quadruples (64 bytes) in one call, returning a
//! `uint8x16x4_t` where `.0 = V`, `.1 = U`, `.2 = Y`, `.3 = A`.
//! No shift is needed — samples are natively 8-bit.
//!
//! Each channel is split into low (lanes 0-7) and high (lanes 8-15)
//! halves, zero-extended to `int16x8_t`, and run through the shared
//! Q15 chroma + Y pipeline. The two halves are then narrowed to `u8`
//! and combined into `uint8x16_t` for interleaved store via `vst3q_u8`
//! (RGB) or `vst4q_u8` (RGBA).
//!
//! ## Tail
//!
//! `width % 16` remaining pixels fall through to `scalar::vuya_to_rgb_or_rgba_row`.

use core::arch::aarch64::*;

use super::*;
use crate::{ColorMatrix, row::scalar};

// ---- shared kernel template ---------------------------------------------

/// NEON VUYA/VUYX → packed u8 RGB or RGBA.
///
/// Byte-identical to `scalar::vuya_to_rgb_or_rgba_row::<ALPHA, ALPHA_SRC>`.
///
/// The three valid monomorphizations are:
/// - `<false, false>` — RGB (drops α)
/// - `<true, true>`  — RGBA, source α pass-through (VUYA)
/// - `<true, false>` — RGBA, force α = `0xFF` (VUYX)
///
/// `<false, true>` is rejected at monomorphization via `const { assert! }`.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `packed.len() >= width * 4`.
/// 3. `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn vuya_to_rgb_or_rgba_row<const ALPHA: bool, const ALPHA_SRC: bool>(
  packed: &[u8],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // Source alpha requires RGBA output.
  const { assert!(!ALPHA_SRC || ALPHA) };
  debug_assert!(packed.len() >= width * 4, "packed row too short");
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp, "out row too short");

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<8, 8>(full_range);
  let bias = scalar::chroma_bias::<8>();
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
    while x + 16 <= width {
      // Load 16 VUYA quadruples (16 × 4 × u8 = 64 bytes).
      // vld4q_u8 deinterleaves: .0=V16, .1=U16, .2=Y16, .3=A16.
      let q = vld4q_u8(packed.as_ptr().add(x * 4));
      let v_raw = q.0; // uint8x16_t — 16 V bytes
      let u_raw = q.1; // uint8x16_t — 16 U bytes
      let y_raw = q.2; // uint8x16_t — 16 Y bytes
      let a_raw = q.3; // uint8x16_t — 16 A bytes (may be padding for VUYX)

      // Zero-extend V/U/Y halves to i16x8 (8 lanes each).
      let v_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v_raw)));
      let v_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(v_raw)));
      let u_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(u_raw)));
      let u_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(u_raw)));
      let y_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(y_raw)));
      let y_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(y_raw)));

      // Subtract chroma bias (128 for 8-bit).
      let u_sub_lo = vsubq_s16(u_lo, bias_v);
      let u_sub_hi = vsubq_s16(u_hi, bias_v);
      let v_sub_lo = vsubq_s16(v_lo, bias_v);
      let v_sub_hi = vsubq_s16(v_hi, bias_v);

      // Widen to i32x4 lo/hi for Q15 chroma-scale multiply (low half).
      let u_lo_lo_i32 = vmovl_s16(vget_low_s16(u_sub_lo));
      let u_lo_hi_i32 = vmovl_s16(vget_high_s16(u_sub_lo));
      let v_lo_lo_i32 = vmovl_s16(vget_low_s16(v_sub_lo));
      let v_lo_hi_i32 = vmovl_s16(vget_high_s16(v_sub_lo));

      let u_d_lo_lo = q15_shift(vaddq_s32(vmulq_s32(u_lo_lo_i32, c_scale_v), rnd_v));
      let u_d_lo_hi = q15_shift(vaddq_s32(vmulq_s32(u_lo_hi_i32, c_scale_v), rnd_v));
      let v_d_lo_lo = q15_shift(vaddq_s32(vmulq_s32(v_lo_lo_i32, c_scale_v), rnd_v));
      let v_d_lo_hi = q15_shift(vaddq_s32(vmulq_s32(v_lo_hi_i32, c_scale_v), rnd_v));

      // Chroma for low 8 lanes.
      let r_chroma_lo = chroma_i16x8(cru, crv, u_d_lo_lo, v_d_lo_lo, u_d_lo_hi, v_d_lo_hi, rnd_v);
      let g_chroma_lo = chroma_i16x8(cgu, cgv, u_d_lo_lo, v_d_lo_lo, u_d_lo_hi, v_d_lo_hi, rnd_v);
      let b_chroma_lo = chroma_i16x8(cbu, cbv, u_d_lo_lo, v_d_lo_lo, u_d_lo_hi, v_d_lo_hi, rnd_v);

      // Widen to i32x4 lo/hi for Q15 chroma-scale multiply (high half).
      let u_hi_lo_i32 = vmovl_s16(vget_low_s16(u_sub_hi));
      let u_hi_hi_i32 = vmovl_s16(vget_high_s16(u_sub_hi));
      let v_hi_lo_i32 = vmovl_s16(vget_low_s16(v_sub_hi));
      let v_hi_hi_i32 = vmovl_s16(vget_high_s16(v_sub_hi));

      let u_d_hi_lo = q15_shift(vaddq_s32(vmulq_s32(u_hi_lo_i32, c_scale_v), rnd_v));
      let u_d_hi_hi = q15_shift(vaddq_s32(vmulq_s32(u_hi_hi_i32, c_scale_v), rnd_v));
      let v_d_hi_lo = q15_shift(vaddq_s32(vmulq_s32(v_hi_lo_i32, c_scale_v), rnd_v));
      let v_d_hi_hi = q15_shift(vaddq_s32(vmulq_s32(v_hi_hi_i32, c_scale_v), rnd_v));

      // Chroma for high 8 lanes.
      let r_chroma_hi = chroma_i16x8(cru, crv, u_d_hi_lo, v_d_hi_lo, u_d_hi_hi, v_d_hi_hi, rnd_v);
      let g_chroma_hi = chroma_i16x8(cgu, cgv, u_d_hi_lo, v_d_hi_lo, u_d_hi_hi, v_d_hi_hi, rnd_v);
      let b_chroma_hi = chroma_i16x8(cbu, cbv, u_d_hi_lo, v_d_hi_lo, u_d_hi_hi, v_d_hi_hi, rnd_v);

      // Y: scale both halves.
      let y_scaled_lo = scale_y(y_lo, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_hi, y_off_v, y_scale_v, rnd_v);

      // Saturate-add Y + chroma per channel, narrow both halves to u8,
      // then combine into a uint8x16_t.
      let r_u8 = vcombine_u8(
        vqmovun_s16(vqaddq_s16(y_scaled_lo, r_chroma_lo)),
        vqmovun_s16(vqaddq_s16(y_scaled_hi, r_chroma_hi)),
      );
      let g_u8 = vcombine_u8(
        vqmovun_s16(vqaddq_s16(y_scaled_lo, g_chroma_lo)),
        vqmovun_s16(vqaddq_s16(y_scaled_hi, g_chroma_hi)),
      );
      let b_u8 = vcombine_u8(
        vqmovun_s16(vqaddq_s16(y_scaled_lo, b_chroma_lo)),
        vqmovun_s16(vqaddq_s16(y_scaled_hi, b_chroma_hi)),
      );

      // Store 16 pixels.
      let off = x * bpp;
      if ALPHA {
        let a_vec: uint8x16_t = if ALPHA_SRC {
          a_raw // source alpha pass-through (VUYA)
        } else {
          vdupq_n_u8(0xFFu8) // opaque (VUYX)
        };
        vst4q_u8(out.as_mut_ptr().add(off), uint8x16x4_t(r_u8, g_u8, b_u8, a_vec));
      } else {
        vst3q_u8(out.as_mut_ptr().add(off), uint8x16x3_t(r_u8, g_u8, b_u8));
      }

      x += 16;
    }

    // Scalar tail — remaining < 16 pixels.
    let processed = x;
    if processed < width {
      let tail_packed = &packed[processed * 4..];
      let tail_out = &mut out[processed * bpp..];
      scalar::vuya_to_rgb_or_rgba_row::<ALPHA, ALPHA_SRC>(
        tail_packed,
        tail_out,
        width - processed,
        matrix,
        full_range,
      );
    }
  }
}

// ---- thin wrappers -------------------------------------------------------

/// NEON VUYA / VUYX → packed **RGB** (3 bpp). Alpha byte in source is
/// discarded — RGB output has no alpha channel.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn vuya_to_rgb_row(
  packed: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  unsafe { vuya_to_rgb_or_rgba_row::<false, false>(packed, rgb_out, width, matrix, full_range); }
}

/// NEON VUYA → packed **RGBA** (4 bpp). Source A byte is passed through
/// verbatim.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn vuya_to_rgba_row(
  packed: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  unsafe { vuya_to_rgb_or_rgba_row::<true, true>(packed, rgba_out, width, matrix, full_range); }
}

/// NEON VUYX → packed **RGBA** (4 bpp). Source A byte is padding;
/// output α is forced to `0xFF` (opaque).
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn vuyx_to_rgba_row(
  packed: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  unsafe { vuya_to_rgb_or_rgba_row::<true, false>(packed, rgba_out, width, matrix, full_range); }
}

// ---- luma extraction ----------------------------------------------------

/// NEON VUYA / VUYX → u8 luma. Y is the third byte (offset 2) of each
/// pixel quadruple; `vld4q_u8`'s channel 2 delivers it directly.
///
/// Byte-identical to `scalar::vuya_to_luma_row`.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `packed.len() >= width * 4`.
/// 3. `luma_out.len() >= width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn vuya_to_luma_row(packed: &[u8], luma_out: &mut [u8], width: usize) {
  debug_assert!(packed.len() >= width * 4, "packed row too short");
  debug_assert!(luma_out.len() >= width, "luma row too short");

  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      // vld4q_u8 deinterleaves; channel 2 (.2) = Y for all 16 pixels.
      let q = vld4q_u8(packed.as_ptr().add(x * 4));
      vst1q_u8(luma_out.as_mut_ptr().add(x), q.2);
      x += 16;
    }
    // Scalar tail.
    if x < width {
      scalar::vuya_to_luma_row(&packed[x * 4..], &mut luma_out[x..], width - x);
    }
  }
}
