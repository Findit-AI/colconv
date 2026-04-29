use core::arch::aarch64::*;

use crate::{ColorMatrix, row::scalar};

use super::*;

// ===== Pn 4:4:4 (semi-planar high-bit-packed) → RGB =======================
//
// NEON kernels for `p_n_444_to_rgb_*<BITS>` (BITS ∈ {10, 12}, Q15 i32
// pipeline) and `p_n_444_16_to_rgb_*` (BITS = 16, parallel i64-chroma
// for u16 output). The inner math mirrors `yuv_444p_n_to_rgb_row`
// (chroma is 1:1 with Y — no horizontal duplication) but the chroma
// load uses `vld2q_u16` to deinterleave the full-width UV plane in
// register, like `nv24_to_rgb_row`. Each iteration consumes 16 Y
// pixels and 32 UV `u16` elements (= 16 interleaved U/V pairs).

/// NEON Pn 4:4:4 high-bit-packed (BITS ∈ {10, 12}) → packed **u8** RGB.
///
/// Thin wrapper over [`p_n_444_to_rgb_or_rgba_row`] with `ALPHA = false`.
///
/// # Safety
///
/// 1. NEON must be available on the current CPU.
/// 2. `y.len() >= width`, `uv_full.len() >= 2 * width`,
///    `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn p_n_444_to_rgb_row<const BITS: u32>(
  y: &[u16],
  uv_full: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    p_n_444_to_rgb_or_rgba_row::<BITS, false>(y, uv_full, rgb_out, width, matrix, full_range);
  }
}

/// NEON Pn 4:4:4 high-bit-packed (BITS ∈ {10, 12}) → packed **8-bit
/// RGBA** (`R, G, B, 0xFF`).
///
/// Thin wrapper over [`p_n_444_to_rgb_or_rgba_row`] with `ALPHA = true`.
///
/// # Safety
///
/// Same as [`p_n_444_to_rgb_row`] but `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn p_n_444_to_rgba_row<const BITS: u32>(
  y: &[u16],
  uv_full: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    p_n_444_to_rgb_or_rgba_row::<BITS, true>(y, uv_full, rgba_out, width, matrix, full_range);
  }
}

/// Shared NEON Pn 4:4:4 high-bit-packed kernel for
/// [`p_n_444_to_rgb_row`] (`ALPHA = false`, `vst3q_u8`) and
/// [`p_n_444_to_rgba_row`] (`ALPHA = true`, `vst4q_u8` with constant
/// `0xFF` alpha).
///
/// # Safety
///
/// 1. NEON must be available on the current CPU.
/// 2. `y.len() >= width`, `uv_full.len() >= 2 * width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
/// 3. `BITS` must be one of `{10, 12}`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn p_n_444_to_rgb_or_rgba_row<const BITS: u32, const ALPHA: bool>(
  y: &[u16],
  uv_full: &[u16],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  const { assert!(BITS == 10 || BITS == 12) };
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(y.len() >= width);
  debug_assert!(uv_full.len() >= 2 * width);
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<BITS, 8>(full_range);
  let bias = scalar::chroma_bias::<BITS>();
  const RND: i32 = 1 << 14;

  unsafe {
    let rnd_v = vdupq_n_s32(RND);
    let y_off_v = vdupq_n_s16(y_off as i16);
    let y_scale_v = vdupq_n_s32(y_scale);
    let c_scale_v = vdupq_n_s32(c_scale);
    let bias_v = vdupq_n_s16(bias as i16);
    // `vshlq_u16(_, vdupq_n_s16(-(16 - BITS)))` is a logical right shift
    // by `(16 - BITS)` (NEON variable shift treats negative count as right
    // shift). Same pattern as `p_n_to_rgb_row<BITS>` for the high-bit
    // packing extraction.
    let shr_count = vdupq_n_s16(-((16 - BITS) as i16));
    let cru = vdupq_n_s32(coeffs.r_u());
    let crv = vdupq_n_s32(coeffs.r_v());
    let cgu = vdupq_n_s32(coeffs.g_u());
    let cgv = vdupq_n_s32(coeffs.g_v());
    let cbu = vdupq_n_s32(coeffs.b_u());
    let cbv = vdupq_n_s32(coeffs.b_v());
    let alpha_u8 = vdupq_n_u8(0xFF);

    let mut x = 0usize;
    while x + 16 <= width {
      // 16 Y pixels in two u16x8 loads; high-bit-extracted via the
      // logical right shift.
      let y_vec_lo = vshlq_u16(vld1q_u16(y.as_ptr().add(x)), shr_count);
      let y_vec_hi = vshlq_u16(vld1q_u16(y.as_ptr().add(x + 8)), shr_count);

      // 32 UV elements = 16 interleaved (U, V) pairs. Two `vld2q_u16`
      // calls deinterleave them into two pairs of (U, V) u16x8 vectors.
      let uv_pair_lo = vld2q_u16(uv_full.as_ptr().add(x * 2));
      let uv_pair_hi = vld2q_u16(uv_full.as_ptr().add(x * 2 + 16));
      let u_lo_u16 = vshlq_u16(uv_pair_lo.0, shr_count);
      let v_lo_u16 = vshlq_u16(uv_pair_lo.1, shr_count);
      let u_hi_u16 = vshlq_u16(uv_pair_hi.0, shr_count);
      let v_hi_u16 = vshlq_u16(uv_pair_hi.1, shr_count);

      let y_lo = vreinterpretq_s16_u16(y_vec_lo);
      let y_hi = vreinterpretq_s16_u16(y_vec_hi);

      let u_lo_i16 = vsubq_s16(vreinterpretq_s16_u16(u_lo_u16), bias_v);
      let u_hi_i16 = vsubq_s16(vreinterpretq_s16_u16(u_hi_u16), bias_v);
      let v_lo_i16 = vsubq_s16(vreinterpretq_s16_u16(v_lo_u16), bias_v);
      let v_hi_i16 = vsubq_s16(vreinterpretq_s16_u16(v_hi_u16), bias_v);

      // Widen each i16x8 → two i32x4 halves. 1:1 chroma per Y, no
      // duplication.
      let u_lo_a = vmovl_s16(vget_low_s16(u_lo_i16));
      let u_lo_b = vmovl_s16(vget_high_s16(u_lo_i16));
      let u_hi_a = vmovl_s16(vget_low_s16(u_hi_i16));
      let u_hi_b = vmovl_s16(vget_high_s16(u_hi_i16));
      let v_lo_a = vmovl_s16(vget_low_s16(v_lo_i16));
      let v_lo_b = vmovl_s16(vget_high_s16(v_lo_i16));
      let v_hi_a = vmovl_s16(vget_low_s16(v_hi_i16));
      let v_hi_b = vmovl_s16(vget_high_s16(v_hi_i16));

      let u_d_lo_a = q15_shift(vaddq_s32(vmulq_s32(u_lo_a, c_scale_v), rnd_v));
      let u_d_lo_b = q15_shift(vaddq_s32(vmulq_s32(u_lo_b, c_scale_v), rnd_v));
      let u_d_hi_a = q15_shift(vaddq_s32(vmulq_s32(u_hi_a, c_scale_v), rnd_v));
      let u_d_hi_b = q15_shift(vaddq_s32(vmulq_s32(u_hi_b, c_scale_v), rnd_v));
      let v_d_lo_a = q15_shift(vaddq_s32(vmulq_s32(v_lo_a, c_scale_v), rnd_v));
      let v_d_lo_b = q15_shift(vaddq_s32(vmulq_s32(v_lo_b, c_scale_v), rnd_v));
      let v_d_hi_a = q15_shift(vaddq_s32(vmulq_s32(v_hi_a, c_scale_v), rnd_v));
      let v_d_hi_b = q15_shift(vaddq_s32(vmulq_s32(v_hi_b, c_scale_v), rnd_v));

      let r_chroma_lo = chroma_i16x8(cru, crv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let r_chroma_hi = chroma_i16x8(cru, crv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let g_chroma_lo = chroma_i16x8(cgu, cgv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let g_chroma_hi = chroma_i16x8(cgu, cgv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let b_chroma_lo = chroma_i16x8(cbu, cbv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let b_chroma_hi = chroma_i16x8(cbu, cbv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);

      let y_scaled_lo = scale_y(y_lo, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_hi, y_off_v, y_scale_v, rnd_v);

      let b_u8 = vcombine_u8(
        vqmovun_s16(vqaddq_s16(y_scaled_lo, b_chroma_lo)),
        vqmovun_s16(vqaddq_s16(y_scaled_hi, b_chroma_hi)),
      );
      let g_u8 = vcombine_u8(
        vqmovun_s16(vqaddq_s16(y_scaled_lo, g_chroma_lo)),
        vqmovun_s16(vqaddq_s16(y_scaled_hi, g_chroma_hi)),
      );
      let r_u8 = vcombine_u8(
        vqmovun_s16(vqaddq_s16(y_scaled_lo, r_chroma_lo)),
        vqmovun_s16(vqaddq_s16(y_scaled_hi, r_chroma_hi)),
      );

      if ALPHA {
        vst4q_u8(
          out.as_mut_ptr().add(x * 4),
          uint8x16x4_t(r_u8, g_u8, b_u8, alpha_u8),
        );
      } else {
        vst3q_u8(out.as_mut_ptr().add(x * 3), uint8x16x3_t(r_u8, g_u8, b_u8));
      }

      x += 16;
    }

    if x < width {
      let tail_y = &y[x..width];
      let tail_uv = &uv_full[x * 2..width * 2];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      if ALPHA {
        scalar::p_n_444_to_rgba_row::<BITS>(tail_y, tail_uv, tail_out, tail_w, matrix, full_range);
      } else {
        scalar::p_n_444_to_rgb_row::<BITS>(tail_y, tail_uv, tail_out, tail_w, matrix, full_range);
      }
    }
  }
}

/// NEON Pn 4:4:4 high-bit-packed (BITS ∈ {10, 12}) → packed
/// **native-depth `u16`** RGB. Output is low-bit-packed.
///
/// Thin wrapper over [`p_n_444_to_rgb_or_rgba_u16_row`] with `ALPHA = false`.
///
/// # Safety
///
/// 1. NEON must be available on the current CPU.
/// 2. `y.len() >= width`, `uv_full.len() >= 2 * width`,
///    `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn p_n_444_to_rgb_u16_row<const BITS: u32>(
  y: &[u16],
  uv_full: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    p_n_444_to_rgb_or_rgba_u16_row::<BITS, false>(y, uv_full, rgb_out, width, matrix, full_range);
  }
}

/// NEON sibling of [`p_n_444_to_rgba_row`] for native-depth `u16`
/// output. Alpha samples are `(1 << BITS) - 1` (opaque maximum at the
/// input bit depth) — matches `scalar::p_n_444_to_rgba_u16_row`.
///
/// Thin wrapper over [`p_n_444_to_rgb_or_rgba_u16_row`] with `ALPHA = true`.
///
/// # Safety
///
/// Same as [`p_n_444_to_rgb_u16_row`] plus `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn p_n_444_to_rgba_u16_row<const BITS: u32>(
  y: &[u16],
  uv_full: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    p_n_444_to_rgb_or_rgba_u16_row::<BITS, true>(y, uv_full, rgba_out, width, matrix, full_range);
  }
}

/// Shared NEON Pn 4:4:4 high-bit-packed → native-depth `u16` kernel.
/// `ALPHA = false` writes RGB triples via `vst3q_u16`; `ALPHA = true`
/// writes RGBA quads via `vst4q_u16` with constant alpha
/// `(1 << BITS) - 1`.
///
/// # Safety
///
/// 1. NEON must be available on the current CPU.
/// 2. `y.len() >= width`, `uv_full.len() >= 2 * width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
/// 3. `BITS` ∈ `{10, 12}`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn p_n_444_to_rgb_or_rgba_u16_row<const BITS: u32, const ALPHA: bool>(
  y: &[u16],
  uv_full: &[u16],
  out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  const { assert!(BITS == 10 || BITS == 12) };
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(y.len() >= width);
  debug_assert!(uv_full.len() >= 2 * width);
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<BITS, BITS>(full_range);
  let bias = scalar::chroma_bias::<BITS>();
  const RND: i32 = 1 << 14;
  let out_max: i16 = ((1i32 << BITS) - 1) as i16;

  unsafe {
    let rnd_v = vdupq_n_s32(RND);
    let y_off_v = vdupq_n_s16(y_off as i16);
    let y_scale_v = vdupq_n_s32(y_scale);
    let c_scale_v = vdupq_n_s32(c_scale);
    let bias_v = vdupq_n_s16(bias as i16);
    let shr_count = vdupq_n_s16(-((16 - BITS) as i16));
    let zero_v = vdupq_n_s16(0);
    let max_v = vdupq_n_s16(out_max);
    let cru = vdupq_n_s32(coeffs.r_u());
    let crv = vdupq_n_s32(coeffs.r_v());
    let cgu = vdupq_n_s32(coeffs.g_u());
    let cgv = vdupq_n_s32(coeffs.g_v());
    let cbu = vdupq_n_s32(coeffs.b_u());
    let cbv = vdupq_n_s32(coeffs.b_v());
    let alpha_u16 = vdupq_n_u16(out_max as u16);

    let mut x = 0usize;
    while x + 16 <= width {
      let y_vec_lo = vshlq_u16(vld1q_u16(y.as_ptr().add(x)), shr_count);
      let y_vec_hi = vshlq_u16(vld1q_u16(y.as_ptr().add(x + 8)), shr_count);

      let uv_pair_lo = vld2q_u16(uv_full.as_ptr().add(x * 2));
      let uv_pair_hi = vld2q_u16(uv_full.as_ptr().add(x * 2 + 16));
      let u_lo_u16 = vshlq_u16(uv_pair_lo.0, shr_count);
      let v_lo_u16 = vshlq_u16(uv_pair_lo.1, shr_count);
      let u_hi_u16 = vshlq_u16(uv_pair_hi.0, shr_count);
      let v_hi_u16 = vshlq_u16(uv_pair_hi.1, shr_count);

      let y_lo = vreinterpretq_s16_u16(y_vec_lo);
      let y_hi = vreinterpretq_s16_u16(y_vec_hi);

      let u_lo_i16 = vsubq_s16(vreinterpretq_s16_u16(u_lo_u16), bias_v);
      let u_hi_i16 = vsubq_s16(vreinterpretq_s16_u16(u_hi_u16), bias_v);
      let v_lo_i16 = vsubq_s16(vreinterpretq_s16_u16(v_lo_u16), bias_v);
      let v_hi_i16 = vsubq_s16(vreinterpretq_s16_u16(v_hi_u16), bias_v);

      let u_lo_a = vmovl_s16(vget_low_s16(u_lo_i16));
      let u_lo_b = vmovl_s16(vget_high_s16(u_lo_i16));
      let u_hi_a = vmovl_s16(vget_low_s16(u_hi_i16));
      let u_hi_b = vmovl_s16(vget_high_s16(u_hi_i16));
      let v_lo_a = vmovl_s16(vget_low_s16(v_lo_i16));
      let v_lo_b = vmovl_s16(vget_high_s16(v_lo_i16));
      let v_hi_a = vmovl_s16(vget_low_s16(v_hi_i16));
      let v_hi_b = vmovl_s16(vget_high_s16(v_hi_i16));

      let u_d_lo_a = q15_shift(vaddq_s32(vmulq_s32(u_lo_a, c_scale_v), rnd_v));
      let u_d_lo_b = q15_shift(vaddq_s32(vmulq_s32(u_lo_b, c_scale_v), rnd_v));
      let u_d_hi_a = q15_shift(vaddq_s32(vmulq_s32(u_hi_a, c_scale_v), rnd_v));
      let u_d_hi_b = q15_shift(vaddq_s32(vmulq_s32(u_hi_b, c_scale_v), rnd_v));
      let v_d_lo_a = q15_shift(vaddq_s32(vmulq_s32(v_lo_a, c_scale_v), rnd_v));
      let v_d_lo_b = q15_shift(vaddq_s32(vmulq_s32(v_lo_b, c_scale_v), rnd_v));
      let v_d_hi_a = q15_shift(vaddq_s32(vmulq_s32(v_hi_a, c_scale_v), rnd_v));
      let v_d_hi_b = q15_shift(vaddq_s32(vmulq_s32(v_hi_b, c_scale_v), rnd_v));

      let r_chroma_lo = chroma_i16x8(cru, crv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let r_chroma_hi = chroma_i16x8(cru, crv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let g_chroma_lo = chroma_i16x8(cgu, cgv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let g_chroma_hi = chroma_i16x8(cgu, cgv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let b_chroma_lo = chroma_i16x8(cbu, cbv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let b_chroma_hi = chroma_i16x8(cbu, cbv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);

      let y_scaled_lo = scale_y(y_lo, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_hi, y_off_v, y_scale_v, rnd_v);

      // Clamp [0, out_max] in i16, then reinterpret as u16 (all values
      // are non-negative after clamp).
      let r_lo = clamp_u16_max(vaddq_s16(y_scaled_lo, r_chroma_lo), zero_v, max_v);
      let r_hi = clamp_u16_max(vaddq_s16(y_scaled_hi, r_chroma_hi), zero_v, max_v);
      let g_lo = clamp_u16_max(vaddq_s16(y_scaled_lo, g_chroma_lo), zero_v, max_v);
      let g_hi = clamp_u16_max(vaddq_s16(y_scaled_hi, g_chroma_hi), zero_v, max_v);
      let b_lo = clamp_u16_max(vaddq_s16(y_scaled_lo, b_chroma_lo), zero_v, max_v);
      let b_hi = clamp_u16_max(vaddq_s16(y_scaled_hi, b_chroma_hi), zero_v, max_v);

      if ALPHA {
        let rgba_lo = uint16x8x4_t(r_lo, g_lo, b_lo, alpha_u16);
        let rgba_hi = uint16x8x4_t(r_hi, g_hi, b_hi, alpha_u16);
        vst4q_u16(out.as_mut_ptr().add(x * 4), rgba_lo);
        vst4q_u16(out.as_mut_ptr().add(x * 4 + 32), rgba_hi);
      } else {
        let rgb_lo = uint16x8x3_t(r_lo, g_lo, b_lo);
        let rgb_hi = uint16x8x3_t(r_hi, g_hi, b_hi);
        vst3q_u16(out.as_mut_ptr().add(x * 3), rgb_lo);
        vst3q_u16(out.as_mut_ptr().add(x * 3 + 24), rgb_hi);
      }

      x += 16;
    }

    if x < width {
      let tail_y = &y[x..width];
      let tail_uv = &uv_full[x * 2..width * 2];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      if ALPHA {
        scalar::p_n_444_to_rgba_u16_row::<BITS>(
          tail_y, tail_uv, tail_out, tail_w, matrix, full_range,
        );
      } else {
        scalar::p_n_444_to_rgb_u16_row::<BITS>(
          tail_y, tail_uv, tail_out, tail_w, matrix, full_range,
        );
      }
    }
  }
}

/// NEON P416 (semi-planar 4:4:4, 16-bit) → packed **u8** RGB.
/// Y and chroma both stay on i32 (output-range scaling keeps `coeff
/// × u_d` within i32 for u8 output). Mirror `yuv_444p16_to_rgb_row`
/// with full-width interleaved UV via `vld2q_u16`.
///
/// Thin wrapper over [`p_n_444_16_to_rgb_or_rgba_row`] with `ALPHA = false`.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `y.len() >= width`, `uv_full.len() >= 2 * width`,
///    `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn p_n_444_16_to_rgb_row(
  y: &[u16],
  uv_full: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    p_n_444_16_to_rgb_or_rgba_row::<false>(y, uv_full, rgb_out, width, matrix, full_range);
  }
}

/// NEON P416 (semi-planar 4:4:4, 16-bit) → packed **8-bit RGBA**
/// (`R, G, B, 0xFF`). Same numerical contract as
/// [`p_n_444_16_to_rgb_row`].
///
/// Thin wrapper over [`p_n_444_16_to_rgb_or_rgba_row`] with `ALPHA = true`.
///
/// # Safety
///
/// Same as [`p_n_444_16_to_rgb_row`] but `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn p_n_444_16_to_rgba_row(
  y: &[u16],
  uv_full: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    p_n_444_16_to_rgb_or_rgba_row::<true>(y, uv_full, rgba_out, width, matrix, full_range);
  }
}

/// Shared NEON P416 (semi-planar 4:4:4, 16-bit) kernel for
/// [`p_n_444_16_to_rgb_row`] (`ALPHA = false`, `vst3q_u8`) and
/// [`p_n_444_16_to_rgba_row`] (`ALPHA = true`, `vst4q_u8` with constant
/// `0xFF` alpha).
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `y.len() >= width`, `uv_full.len() >= 2 * width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn p_n_444_16_to_rgb_or_rgba_row<const ALPHA: bool>(
  y: &[u16],
  uv_full: &[u16],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(y.len() >= width);
  debug_assert!(uv_full.len() >= 2 * width);
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<16, 8>(full_range);
  let bias = scalar::chroma_bias::<16>();
  const RND: i32 = 1 << 14;

  unsafe {
    let rnd_v = vdupq_n_s32(RND);
    let y_off_v = vdupq_n_s32(y_off);
    let y_scale_v = vdupq_n_s32(y_scale);
    let c_scale_v = vdupq_n_s32(c_scale);
    let bias_v = vdupq_n_s32(bias);
    let cru = vdupq_n_s32(coeffs.r_u());
    let crv = vdupq_n_s32(coeffs.r_v());
    let cgu = vdupq_n_s32(coeffs.g_u());
    let cgv = vdupq_n_s32(coeffs.g_v());
    let cbu = vdupq_n_s32(coeffs.b_u());
    let cbv = vdupq_n_s32(coeffs.b_v());
    let alpha_u8 = vdupq_n_u8(0xFF);

    let mut x = 0usize;
    while x + 16 <= width {
      let y_vec_lo = vld1q_u16(y.as_ptr().add(x));
      let y_vec_hi = vld1q_u16(y.as_ptr().add(x + 8));

      // 16 chroma pairs per iter — two `vld2q_u16` calls deinterleave
      // 32 UV `u16` elements into two pairs of (U, V) u16x8 vectors.
      let uv_pair_lo = vld2q_u16(uv_full.as_ptr().add(x * 2));
      let uv_pair_hi = vld2q_u16(uv_full.as_ptr().add(x * 2 + 16));
      let u_vec_lo = uv_pair_lo.0;
      let v_vec_lo = uv_pair_lo.1;
      let u_vec_hi = uv_pair_hi.0;
      let v_vec_hi = uv_pair_hi.1;

      // Unsigned-widen + bias subtract in i32 (16-bit chroma can't fit
      // i16 after subtracting 32768).
      let u_lo_a = vsubq_s32(
        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(u_vec_lo))),
        bias_v,
      );
      let u_lo_b = vsubq_s32(
        vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(u_vec_lo))),
        bias_v,
      );
      let u_hi_a = vsubq_s32(
        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(u_vec_hi))),
        bias_v,
      );
      let u_hi_b = vsubq_s32(
        vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(u_vec_hi))),
        bias_v,
      );
      let v_lo_a = vsubq_s32(
        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v_vec_lo))),
        bias_v,
      );
      let v_lo_b = vsubq_s32(
        vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(v_vec_lo))),
        bias_v,
      );
      let v_hi_a = vsubq_s32(
        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v_vec_hi))),
        bias_v,
      );
      let v_hi_b = vsubq_s32(
        vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(v_vec_hi))),
        bias_v,
      );

      let u_d_lo_a = q15_shift(vaddq_s32(vmulq_s32(u_lo_a, c_scale_v), rnd_v));
      let u_d_lo_b = q15_shift(vaddq_s32(vmulq_s32(u_lo_b, c_scale_v), rnd_v));
      let u_d_hi_a = q15_shift(vaddq_s32(vmulq_s32(u_hi_a, c_scale_v), rnd_v));
      let u_d_hi_b = q15_shift(vaddq_s32(vmulq_s32(u_hi_b, c_scale_v), rnd_v));
      let v_d_lo_a = q15_shift(vaddq_s32(vmulq_s32(v_lo_a, c_scale_v), rnd_v));
      let v_d_lo_b = q15_shift(vaddq_s32(vmulq_s32(v_lo_b, c_scale_v), rnd_v));
      let v_d_hi_a = q15_shift(vaddq_s32(vmulq_s32(v_hi_a, c_scale_v), rnd_v));
      let v_d_hi_b = q15_shift(vaddq_s32(vmulq_s32(v_hi_b, c_scale_v), rnd_v));

      let r_chroma_lo = chroma_i16x8(cru, crv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let r_chroma_hi = chroma_i16x8(cru, crv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let g_chroma_lo = chroma_i16x8(cgu, cgv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let g_chroma_hi = chroma_i16x8(cgu, cgv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let b_chroma_lo = chroma_i16x8(cbu, cbv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let b_chroma_hi = chroma_i16x8(cbu, cbv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);

      let y_scaled_lo = scale_y_u16_to_i16(y_vec_lo, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y_u16_to_i16(y_vec_hi, y_off_v, y_scale_v, rnd_v);

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

      if ALPHA {
        vst4q_u8(
          out.as_mut_ptr().add(x * 4),
          uint8x16x4_t(r_u8, g_u8, b_u8, alpha_u8),
        );
      } else {
        vst3q_u8(out.as_mut_ptr().add(x * 3), uint8x16x3_t(r_u8, g_u8, b_u8));
      }
      x += 16;
    }

    if x < width {
      let tail_y = &y[x..width];
      let tail_uv = &uv_full[x * 2..width * 2];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      if ALPHA {
        scalar::p_n_444_16_to_rgba_row(tail_y, tail_uv, tail_out, tail_w, matrix, full_range);
      } else {
        scalar::p_n_444_16_to_rgb_row(tail_y, tail_uv, tail_out, tail_w, matrix, full_range);
      }
    }
  }
}

/// NEON P416 (semi-planar 4:4:4, 16-bit) → packed **native-depth u16**
/// RGB. i64 chroma + i64 Y (chroma matrix multiply-add overflows i32
/// at u16 output for the BT.2020 blue coefficient). 8 pixels per iter.
///
/// Thin wrapper over [`p_n_444_16_to_rgb_or_rgba_u16_row`] with `ALPHA = false`.
///
/// # Safety
///
/// Same as [`p_n_444_16_to_rgb_row`] but `rgb_out: &mut [u16]`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn p_n_444_16_to_rgb_u16_row(
  y: &[u16],
  uv_full: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    p_n_444_16_to_rgb_or_rgba_u16_row::<false>(y, uv_full, rgb_out, width, matrix, full_range);
  }
}

/// NEON sibling of [`p_n_444_16_to_rgba_row`] for native-depth `u16`
/// output. Alpha samples are `0xFFFF` (opaque maximum at u16 range) —
/// matches `scalar::p_n_444_16_to_rgba_u16_row`.
///
/// Thin wrapper over [`p_n_444_16_to_rgb_or_rgba_u16_row`] with `ALPHA = true`.
///
/// # Safety
///
/// Same as [`p_n_444_16_to_rgb_u16_row`] plus `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn p_n_444_16_to_rgba_u16_row(
  y: &[u16],
  uv_full: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    p_n_444_16_to_rgb_or_rgba_u16_row::<true>(y, uv_full, rgba_out, width, matrix, full_range);
  }
}

/// Shared NEON P416 (semi-planar 4:4:4, 16-bit) → native-depth `u16`
/// kernel. `ALPHA = false` writes RGB triples via `vst3q_u16`;
/// `ALPHA = true` writes RGBA quads via `vst4q_u16` with constant alpha
/// `0xFFFF`.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `y.len() >= width`, `uv_full.len() >= 2 * width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn p_n_444_16_to_rgb_or_rgba_u16_row<const ALPHA: bool>(
  y: &[u16],
  uv_full: &[u16],
  out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(y.len() >= width);
  debug_assert!(uv_full.len() >= 2 * width);
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<16, 16>(full_range);
  let bias = scalar::chroma_bias::<16>();
  const RND: i32 = 1 << 14;

  unsafe {
    let rnd_v = vdupq_n_s32(RND);
    let rnd64 = vdupq_n_s64(RND as i64);
    let y_off_v = vdupq_n_s32(y_off);
    let y_scale_d = vdup_n_s32(y_scale);
    let c_scale_v = vdupq_n_s32(c_scale);
    let bias_v = vdupq_n_s32(bias);
    let cru = vdupq_n_s32(coeffs.r_u());
    let crv = vdupq_n_s32(coeffs.r_v());
    let cgu = vdupq_n_s32(coeffs.g_u());
    let cgv = vdupq_n_s32(coeffs.g_v());
    let cbu = vdupq_n_s32(coeffs.b_u());
    let cbv = vdupq_n_s32(coeffs.b_v());
    let alpha_u16 = vdupq_n_u16(0xFFFF);

    let mut x = 0usize;
    while x + 8 <= width {
      // 8 Y + 8 chroma pairs per iter — tighter block because i64
      // chroma narrows throughput; matches `yuv_444p16_to_rgb_u16_row`.
      let y_vec = vld1q_u16(y.as_ptr().add(x));
      let uv_pair = vld2q_u16(uv_full.as_ptr().add(x * 2));
      let u_vec = uv_pair.0;
      let v_vec = uv_pair.1;

      let u_lo_i32 = vsubq_s32(
        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(u_vec))),
        bias_v,
      );
      let u_hi_i32 = vsubq_s32(
        vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(u_vec))),
        bias_v,
      );
      let v_lo_i32 = vsubq_s32(
        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v_vec))),
        bias_v,
      );
      let v_hi_i32 = vsubq_s32(
        vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(v_vec))),
        bias_v,
      );

      let u_d_lo = q15_shift(vaddq_s32(vmulq_s32(u_lo_i32, c_scale_v), rnd_v));
      let u_d_hi = q15_shift(vaddq_s32(vmulq_s32(u_hi_i32, c_scale_v), rnd_v));
      let v_d_lo = q15_shift(vaddq_s32(vmulq_s32(v_lo_i32, c_scale_v), rnd_v));
      let v_d_hi = q15_shift(vaddq_s32(vmulq_s32(v_hi_i32, c_scale_v), rnd_v));

      // i64 chroma — 8 chroma values via two `chroma_i64x4` calls.
      let r_ch_lo = chroma_i64x4(cru, crv, u_d_lo, v_d_lo, rnd64);
      let r_ch_hi = chroma_i64x4(cru, crv, u_d_hi, v_d_hi, rnd64);
      let g_ch_lo = chroma_i64x4(cgu, cgv, u_d_lo, v_d_lo, rnd64);
      let g_ch_hi = chroma_i64x4(cgu, cgv, u_d_hi, v_d_hi, rnd64);
      let b_ch_lo = chroma_i64x4(cbu, cbv, u_d_lo, v_d_lo, rnd64);
      let b_ch_hi = chroma_i64x4(cbu, cbv, u_d_hi, v_d_hi, rnd64);

      let y_lo_i32 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(y_vec)));
      let y_hi_i32 = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(y_vec)));
      let ys_lo = scale_y_u16_i64(y_lo_i32, y_off_v, y_scale_d, rnd64);
      let ys_hi = scale_y_u16_i64(y_hi_i32, y_off_v, y_scale_d, rnd64);

      let r_u16 = vcombine_u16(
        vqmovun_s32(vaddq_s32(ys_lo, r_ch_lo)),
        vqmovun_s32(vaddq_s32(ys_hi, r_ch_hi)),
      );
      let g_u16 = vcombine_u16(
        vqmovun_s32(vaddq_s32(ys_lo, g_ch_lo)),
        vqmovun_s32(vaddq_s32(ys_hi, g_ch_hi)),
      );
      let b_u16 = vcombine_u16(
        vqmovun_s32(vaddq_s32(ys_lo, b_ch_lo)),
        vqmovun_s32(vaddq_s32(ys_hi, b_ch_hi)),
      );

      if ALPHA {
        vst4q_u16(
          out.as_mut_ptr().add(x * 4),
          uint16x8x4_t(r_u16, g_u16, b_u16, alpha_u16),
        );
      } else {
        vst3q_u16(
          out.as_mut_ptr().add(x * 3),
          uint16x8x3_t(r_u16, g_u16, b_u16),
        );
      }
      x += 8;
    }

    if x < width {
      let tail_y = &y[x..width];
      let tail_uv = &uv_full[x * 2..width * 2];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      if ALPHA {
        scalar::p_n_444_16_to_rgba_u16_row(tail_y, tail_uv, tail_out, tail_w, matrix, full_range);
      } else {
        scalar::p_n_444_16_to_rgb_u16_row(tail_y, tail_uv, tail_out, tail_w, matrix, full_range);
      }
    }
  }
}
