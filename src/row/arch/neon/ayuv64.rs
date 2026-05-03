//! NEON kernels for AYUV64 packed YUV 4:4:4 16-bit family
//! (FFmpeg `AV_PIX_FMT_AYUV64LE`).
//!
//! ## Layout
//!
//! Four `u16` elements per pixel: `A(16) ‖ Y(16) ‖ U(16) ‖ V(16)`.
//! All channels are 16-bit native — no padding bits, no right-shift on
//! load. `vld4q_u16` deinterleaves: `.0 = A`, `.1 = Y`, `.2 = U`,
//! `.3 = V`.
//!
//! ## Per-iter pipeline (16 px / iter)
//!
//! Two `vld4q_u16` calls load 8 pixels each (32 u16 per channel total),
//! producing `uint16x8_t` halves for each of the four channels:
//! `a_lo/a_hi`, `y_lo/y_hi`, `u_lo/u_hi`, `v_lo/v_hi`.
//!
//! - u8 output: Y values are full 16-bit (0..65535), so
//!   `scale_y_u16_to_i16` is used (not `scale_y`, which would corrupt
//!   values > 32767). i32 chroma via `chroma_i16x8`.
//!   Source α (A channel): `vshrn_n_u16::<8>` narrows u16 → u8 (high byte).
//!
//! - u16 output: i64 chroma via `chroma_i64x4_neon` to avoid i32
//!   overflow at BITS=16/16 (peak ~3.7×10⁹ for BT.2020 limited). Y
//!   scaled via `scale_y_u16_i64`. Source α written direct as u16.
//!
//! ## Tail
//!
//! `width % 16` remaining pixels fall through to the scalar
//! `ayuv64_to_rgb_or_rgba_row::<ALPHA, ALPHA_SRC>` (or u16 version).

use core::arch::aarch64::*;

use super::*;
use crate::{ColorMatrix, row::scalar};

// ---- u8 RGB / RGBA output -----------------------------------------------

/// NEON AYUV64 → packed u8 RGB or RGBA.
///
/// Byte-identical to `scalar::ayuv64_to_rgb_or_rgba_row::<ALPHA, ALPHA_SRC>`.
///
/// Valid monomorphizations:
/// - `<false, false>` — RGB (α dropped)
/// - `<true, true>`  — RGBA, source α depth-converted u16 → u8 (`>> 8`)
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
pub(crate) unsafe fn ayuv64_to_rgb_or_rgba_row<const ALPHA: bool, const ALPHA_SRC: bool>(
  packed: &[u16],
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
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<16, 8>(full_range);
  let bias = scalar::chroma_bias::<16>();
  const RND: i32 = 1 << 14;

  unsafe {
    let rnd_v = vdupq_n_s32(RND);
    // Y values are full u16 (0..65535); use i32 y_off for scale_y_u16_to_i16.
    let y_off_v = vdupq_n_s32(y_off);
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
      // Two vld4q_u16 loads: each deinterleaves 8 pixels (8 × 4 × u16 = 64 bytes).
      // Channel order: .0=A, .1=Y, .2=U, .3=V (AYUV64 native layout).
      let q_lo = vld4q_u16(packed.as_ptr().add(x * 4));
      let q_hi = vld4q_u16(packed.as_ptr().add(x * 4 + 32));

      // Extract channels (no shift needed — 16-bit native samples).
      let a_lo_u16 = q_lo.0; // uint16x8_t — A for pixels 0..7
      let y_lo_u16 = q_lo.1; // uint16x8_t — Y for pixels 0..7
      let u_lo_u16 = q_lo.2; // uint16x8_t — U for pixels 0..7
      let v_lo_u16 = q_lo.3; // uint16x8_t — V for pixels 0..7

      let a_hi_u16 = q_hi.0; // uint16x8_t — A for pixels 8..15
      let y_hi_u16 = q_hi.1; // uint16x8_t — Y for pixels 8..15
      let u_hi_u16 = q_hi.2; // uint16x8_t — U for pixels 8..15
      let v_hi_u16 = q_hi.3; // uint16x8_t — V for pixels 8..15

      // Reinterpret chroma as signed i16 (bias subtraction fits i16:
      // chroma ∈ [0,65535], bias=32768, so (chroma-bias) ∈ [-32768,32767]).
      let u_lo_i16 = vreinterpretq_s16_u16(u_lo_u16);
      let v_lo_i16 = vreinterpretq_s16_u16(v_lo_u16);
      let u_hi_i16 = vreinterpretq_s16_u16(u_hi_u16);
      let v_hi_i16 = vreinterpretq_s16_u16(v_hi_u16);

      // Subtract chroma bias (32768 for 16-bit).
      let u_sub_lo = vsubq_s16(u_lo_i16, bias_v);
      let v_sub_lo = vsubq_s16(v_lo_i16, bias_v);
      let u_sub_hi = vsubq_s16(u_hi_i16, bias_v);
      let v_sub_hi = vsubq_s16(v_hi_i16, bias_v);

      // Widen to i32x4 lo/hi for Q15 chroma-scale multiply — low half.
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

      // Widen to i32x4 lo/hi for Q15 chroma-scale multiply — high half.
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

      // Y: full u16 values → use u16-aware scale helper (not scale_y).
      let y_scaled_lo = scale_y_u16_to_i16(y_lo_u16, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y_u16_to_i16(y_hi_u16, y_off_v, y_scale_v, rnd_v);

      // Saturate-add Y + chroma per channel; narrow both halves to u8;
      // combine halves into uint8x16_t.
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
        // Source α: depth-convert u16 → u8 via vshrn_n_u16::<8> (high byte).
        // vshrn_n_u16::<8> narrows 8 u16 lanes to 8 u8 lanes (drops low byte).
        let a_lo_u8 = vshrn_n_u16::<8>(a_lo_u16);
        let a_hi_u8 = vshrn_n_u16::<8>(a_hi_u16);
        let a_vec: uint8x16_t = if ALPHA_SRC {
          vcombine_u8(a_lo_u8, a_hi_u8) // source alpha pass-through (depth-converted)
        } else {
          vdupq_n_u8(0xFF) // opaque (unused — no AYUV64x sibling, but allowed)
        };
        vst4q_u8(
          out.as_mut_ptr().add(off),
          uint8x16x4_t(r_u8, g_u8, b_u8, a_vec),
        );
      } else {
        vst3q_u8(out.as_mut_ptr().add(off), uint8x16x3_t(r_u8, g_u8, b_u8));
      }

      x += 16;
    }

    // Scalar tail — remaining < 16 pixels.
    if x < width {
      let tail_packed = &packed[x * 4..width * 4];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      scalar::ayuv64_to_rgb_or_rgba_row::<ALPHA, ALPHA_SRC>(
        tail_packed,
        tail_out,
        tail_w,
        matrix,
        full_range,
      );
    }
  }
}

// ---- u16 RGB / RGBA native-depth output ---------------------------------

/// NEON AYUV64 → packed native-depth u16 RGB or RGBA.
///
/// Uses i64 chroma (`chroma_i64x4`) to avoid overflow at BITS=16/16.
/// Byte-identical to `scalar::ayuv64_to_rgb_u16_or_rgba_u16_row::<ALPHA, ALPHA_SRC>`.
///
/// Valid monomorphizations:
/// - `<false, false>` — RGB u16 (α dropped)
/// - `<true, true>`  — RGBA u16, source α written direct (no conversion)
///
/// `<false, true>` is rejected at monomorphization via `const { assert! }`.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `packed.len() >= width * 4`.
/// 3. `out.len() >= width * (if ALPHA { 4 } else { 3 })` (u16 elements).
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn ayuv64_to_rgb_u16_or_rgba_u16_row<const ALPHA: bool, const ALPHA_SRC: bool>(
  packed: &[u16],
  out: &mut [u16],
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
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<16, 16>(full_range);
  let bias = scalar::chroma_bias::<16>();
  const RND: i32 = 1 << 14;

  unsafe {
    let alpha_u16 = vdupq_n_u16(0xFFFF);
    let rnd_v = vdupq_n_s32(RND);
    let rnd64 = vdupq_n_s64(RND as i64);
    let y_off_v = vdupq_n_s32(y_off);
    let y_scale_d = vdup_n_s32(y_scale); // int32x2_t for vmull_s32
    let c_scale_v = vdupq_n_s32(c_scale);
    let bias_v = vdupq_n_s32(bias);
    let cru = vdupq_n_s32(coeffs.r_u());
    let crv = vdupq_n_s32(coeffs.r_v());
    let cgu = vdupq_n_s32(coeffs.g_u());
    let cgv = vdupq_n_s32(coeffs.g_v());
    let cbu = vdupq_n_s32(coeffs.b_u());
    let cbv = vdupq_n_s32(coeffs.b_v());

    let mut x = 0usize;
    while x + 16 <= width {
      // Two vld4q_u16 loads: each handles 8 pixels.
      // Channel order: .0=A, .1=Y, .2=U, .3=V.
      let q_lo = vld4q_u16(packed.as_ptr().add(x * 4));
      let q_hi = vld4q_u16(packed.as_ptr().add(x * 4 + 32));

      let a_lo_u16 = q_lo.0;
      let y_lo_u16 = q_lo.1;
      let u_lo_u16 = q_lo.2;
      let v_lo_u16 = q_lo.3;

      let a_hi_u16 = q_hi.0;
      let y_hi_u16 = q_hi.1;
      let u_hi_u16 = q_hi.2;
      let v_hi_u16 = q_hi.3;

      // Chroma: widen u16 → i32, subtract bias, apply c_scale (Q15).
      // 4:4:4 — 8 per-pixel chroma values per half, split into 2 × i32x4.
      // lo half: pixels 0..7, further split into lo0 (0..3) and lo1 (4..7).
      let u_lo0_i32 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(u_lo_u16)));
      let u_lo1_i32 = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(u_lo_u16)));
      let v_lo0_i32 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v_lo_u16)));
      let v_lo1_i32 = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(v_lo_u16)));

      let u_d_lo0 = q15_shift(vaddq_s32(
        vmulq_s32(vsubq_s32(u_lo0_i32, bias_v), c_scale_v),
        rnd_v,
      ));
      let u_d_lo1 = q15_shift(vaddq_s32(
        vmulq_s32(vsubq_s32(u_lo1_i32, bias_v), c_scale_v),
        rnd_v,
      ));
      let v_d_lo0 = q15_shift(vaddq_s32(
        vmulq_s32(vsubq_s32(v_lo0_i32, bias_v), c_scale_v),
        rnd_v,
      ));
      let v_d_lo1 = q15_shift(vaddq_s32(
        vmulq_s32(vsubq_s32(v_lo1_i32, bias_v), c_scale_v),
        rnd_v,
      ));

      // hi half: pixels 8..15.
      let u_hi0_i32 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(u_hi_u16)));
      let u_hi1_i32 = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(u_hi_u16)));
      let v_hi0_i32 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v_hi_u16)));
      let v_hi1_i32 = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(v_hi_u16)));

      let u_d_hi0 = q15_shift(vaddq_s32(
        vmulq_s32(vsubq_s32(u_hi0_i32, bias_v), c_scale_v),
        rnd_v,
      ));
      let u_d_hi1 = q15_shift(vaddq_s32(
        vmulq_s32(vsubq_s32(u_hi1_i32, bias_v), c_scale_v),
        rnd_v,
      ));
      let v_d_hi0 = q15_shift(vaddq_s32(
        vmulq_s32(vsubq_s32(v_hi0_i32, bias_v), c_scale_v),
        rnd_v,
      ));
      let v_d_hi1 = q15_shift(vaddq_s32(
        vmulq_s32(vsubq_s32(v_hi1_i32, bias_v), c_scale_v),
        rnd_v,
      ));

      // i64 chroma: 4 values each → i32x4 (via vmull_s32 widening).
      // 4:4:4 — no duplication needed; each pixel has unique U/V.
      let r_ch_lo0 = chroma_i64x4(cru, crv, u_d_lo0, v_d_lo0, rnd64);
      let r_ch_lo1 = chroma_i64x4(cru, crv, u_d_lo1, v_d_lo1, rnd64);
      let g_ch_lo0 = chroma_i64x4(cgu, cgv, u_d_lo0, v_d_lo0, rnd64);
      let g_ch_lo1 = chroma_i64x4(cgu, cgv, u_d_lo1, v_d_lo1, rnd64);
      let b_ch_lo0 = chroma_i64x4(cbu, cbv, u_d_lo0, v_d_lo0, rnd64);
      let b_ch_lo1 = chroma_i64x4(cbu, cbv, u_d_lo1, v_d_lo1, rnd64);

      let r_ch_hi0 = chroma_i64x4(cru, crv, u_d_hi0, v_d_hi0, rnd64);
      let r_ch_hi1 = chroma_i64x4(cru, crv, u_d_hi1, v_d_hi1, rnd64);
      let g_ch_hi0 = chroma_i64x4(cgu, cgv, u_d_hi0, v_d_hi0, rnd64);
      let g_ch_hi1 = chroma_i64x4(cgu, cgv, u_d_hi1, v_d_hi1, rnd64);
      let b_ch_hi0 = chroma_i64x4(cbu, cbv, u_d_hi0, v_d_hi0, rnd64);
      let b_ch_hi1 = chroma_i64x4(cbu, cbv, u_d_hi1, v_d_hi1, rnd64);

      // i64 Y scale: split each 8-lane Y into two i32x4 halves.
      // y_lo_0 = Y0..Y3, y_lo_1 = Y4..Y7; y_hi_0 = Y8..Y11, y_hi_1 = Y12..Y15.
      let y_lo_0 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(y_lo_u16)));
      let y_lo_1 = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(y_lo_u16)));
      let y_hi_0 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(y_hi_u16)));
      let y_hi_1 = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(y_hi_u16)));

      let ys_lo_0 = scale_y_u16_i64(y_lo_0, y_off_v, y_scale_d, rnd64);
      let ys_lo_1 = scale_y_u16_i64(y_lo_1, y_off_v, y_scale_d, rnd64);
      let ys_hi_0 = scale_y_u16_i64(y_hi_0, y_off_v, y_scale_d, rnd64);
      let ys_hi_1 = scale_y_u16_i64(y_hi_1, y_off_v, y_scale_d, rnd64);

      // Y + chroma; vqmovun_s32 saturates i32 → u16 (clamps [0, 65535]).
      // vcombine_u16(A, B) packs two u16x4 into one u16x8.
      let r_lo_u16 = vcombine_u16(
        vqmovun_s32(vaddq_s32(ys_lo_0, r_ch_lo0)),
        vqmovun_s32(vaddq_s32(ys_lo_1, r_ch_lo1)),
      );
      let g_lo_u16 = vcombine_u16(
        vqmovun_s32(vaddq_s32(ys_lo_0, g_ch_lo0)),
        vqmovun_s32(vaddq_s32(ys_lo_1, g_ch_lo1)),
      );
      let b_lo_u16 = vcombine_u16(
        vqmovun_s32(vaddq_s32(ys_lo_0, b_ch_lo0)),
        vqmovun_s32(vaddq_s32(ys_lo_1, b_ch_lo1)),
      );
      let r_hi_u16 = vcombine_u16(
        vqmovun_s32(vaddq_s32(ys_hi_0, r_ch_hi0)),
        vqmovun_s32(vaddq_s32(ys_hi_1, r_ch_hi1)),
      );
      let g_hi_u16 = vcombine_u16(
        vqmovun_s32(vaddq_s32(ys_hi_0, g_ch_hi0)),
        vqmovun_s32(vaddq_s32(ys_hi_1, g_ch_hi1)),
      );
      let b_hi_u16 = vcombine_u16(
        vqmovun_s32(vaddq_s32(ys_hi_0, b_ch_hi0)),
        vqmovun_s32(vaddq_s32(ys_hi_1, b_ch_hi1)),
      );

      // Store 16 pixels. Each vst4q_u16 / vst3q_u16 writes 8 pixels.
      // For RGBA: offset lo = x*4 u16; hi = x*4+32 u16.
      // For RGB:  offset lo = x*3 u16; hi = x*3+24 u16.
      if ALPHA {
        let a_lo_vec: uint16x8_t = if ALPHA_SRC { a_lo_u16 } else { alpha_u16 };
        let a_hi_vec: uint16x8_t = if ALPHA_SRC { a_hi_u16 } else { alpha_u16 };
        vst4q_u16(
          out.as_mut_ptr().add(x * 4),
          uint16x8x4_t(r_lo_u16, g_lo_u16, b_lo_u16, a_lo_vec),
        );
        vst4q_u16(
          out.as_mut_ptr().add(x * 4 + 32),
          uint16x8x4_t(r_hi_u16, g_hi_u16, b_hi_u16, a_hi_vec),
        );
      } else {
        vst3q_u16(
          out.as_mut_ptr().add(x * 3),
          uint16x8x3_t(r_lo_u16, g_lo_u16, b_lo_u16),
        );
        vst3q_u16(
          out.as_mut_ptr().add(x * 3 + 24),
          uint16x8x3_t(r_hi_u16, g_hi_u16, b_hi_u16),
        );
      }

      x += 16;
    }

    // Scalar tail — remaining < 16 pixels.
    if x < width {
      let tail_packed = &packed[x * 4..width * 4];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      scalar::ayuv64_to_rgb_u16_or_rgba_u16_row::<ALPHA, ALPHA_SRC>(
        tail_packed,
        tail_out,
        tail_w,
        matrix,
        full_range,
      );
    }
  }
}

// ---- thin wrappers -------------------------------------------------------

/// NEON AYUV64 → packed **RGB** (3 bpp). Source α is discarded.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn ayuv64_to_rgb_row(
  packed: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  unsafe {
    ayuv64_to_rgb_or_rgba_row::<false, false>(packed, rgb_out, width, matrix, full_range);
  }
}

/// NEON AYUV64 → packed **RGBA** (4 bpp). Source A u16 is depth-converted
/// to u8 via `>> 8`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn ayuv64_to_rgba_row(
  packed: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  unsafe {
    ayuv64_to_rgb_or_rgba_row::<true, true>(packed, rgba_out, width, matrix, full_range);
  }
}

/// NEON AYUV64 → packed **RGB u16** (3 × u16 per pixel). Source α discarded.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn ayuv64_to_rgb_u16_row(
  packed: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  unsafe {
    ayuv64_to_rgb_u16_or_rgba_u16_row::<false, false>(packed, rgb_out, width, matrix, full_range);
  }
}

/// NEON AYUV64 → packed **RGBA u16** (4 × u16 per pixel). Source A u16
/// is written direct (no conversion).
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn ayuv64_to_rgba_u16_row(
  packed: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  unsafe {
    ayuv64_to_rgb_u16_or_rgba_u16_row::<true, true>(packed, rgba_out, width, matrix, full_range);
  }
}

// ---- Luma u8 (16 px/iter) -----------------------------------------------

/// NEON AYUV64 → u8 luma. Y is the second u16 (slot 1) of each pixel
/// quadruple; `vshrn_n_u16::<8>` narrows u16 → u8 (high byte = `>> 8`).
///
/// Byte-identical to `scalar::ayuv64_to_luma_row`.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `packed.len() >= width * 4`.
/// 3. `luma_out.len() >= width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn ayuv64_to_luma_row(packed: &[u16], luma_out: &mut [u8], width: usize) {
  debug_assert!(packed.len() >= width * 4, "packed row too short");
  debug_assert!(luma_out.len() >= width, "luma row too short");

  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      // Two vld4q_u16 loads: channel 1 (.1) = Y for each group of 8 pixels.
      let q_lo = vld4q_u16(packed.as_ptr().add(x * 4));
      let q_hi = vld4q_u16(packed.as_ptr().add(x * 4 + 32));
      // vshrn_n_u16::<8>: narrows 8 u16 → 8 u8 by taking high byte (>> 8).
      let y_lo_u8 = vshrn_n_u16::<8>(q_lo.1);
      let y_hi_u8 = vshrn_n_u16::<8>(q_hi.1);
      vst1_u8(luma_out.as_mut_ptr().add(x), y_lo_u8);
      vst1_u8(luma_out.as_mut_ptr().add(x + 8), y_hi_u8);
      x += 16;
    }
    // Scalar tail.
    if x < width {
      scalar::ayuv64_to_luma_row(
        &packed[x * 4..width * 4],
        &mut luma_out[x..width],
        width - x,
      );
    }
  }
}

// ---- Luma u16 (16 px/iter) ----------------------------------------------

/// NEON AYUV64 → u16 luma. Direct copy of Y samples (slot 1, no shift —
/// 16-bit native).
///
/// Byte-identical to `scalar::ayuv64_to_luma_u16_row`.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `packed.len() >= width * 4`.
/// 3. `luma_out.len() >= width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn ayuv64_to_luma_u16_row(packed: &[u16], luma_out: &mut [u16], width: usize) {
  debug_assert!(packed.len() >= width * 4, "packed row too short");
  debug_assert!(luma_out.len() >= width, "luma row too short");

  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      // Two vld4q_u16 loads: channel 1 (.1) = Y.
      let q_lo = vld4q_u16(packed.as_ptr().add(x * 4));
      let q_hi = vld4q_u16(packed.as_ptr().add(x * 4 + 32));
      // Direct copy — Y samples are 16-bit native (no shift needed).
      vst1q_u16(luma_out.as_mut_ptr().add(x), q_lo.1);
      vst1q_u16(luma_out.as_mut_ptr().add(x + 8), q_hi.1);
      x += 16;
    }
    // Scalar tail.
    if x < width {
      scalar::ayuv64_to_luma_u16_row(
        &packed[x * 4..width * 4],
        &mut luma_out[x..width],
        width - x,
      );
    }
  }
}
