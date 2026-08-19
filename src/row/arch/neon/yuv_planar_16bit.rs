use crate::{KernelMatrix, row::scalar};

use super::*;

// ===== 16-bit YUV → RGB ==================================================
//
// At 16-bit, two precision issues arise compared to the 9/10/12/14-bit generic:
//
// 1. The chroma bias (32768) and full-range u16 values (0..65535) do not fit
//    in i16, so all bias-subtractions happen in i32 after unsigned widening
//    (`vmovl_u16` → `vreinterpretq_s32_u32`).
//
// 2. For u16 output: `c_scale ≈ 37445` (limited range), so `coeff * u_d`
//    reaches ~2.17x10⁹ > i32 max; `y_scale ≈ 38304`, so `(y−y_off)*y_scale`
//    reaches ~2.35x10⁹ > i32 max. Both Y and chroma are widened to i64 via
//    `vmull_s32` and shifted back with `vshrq_n_s64::<15>`.
//
// For u8 output: `c_scale ≈ 127`, so i32 is sufficient throughout.

/// NEON 16-bit planar YUV 4:2:0 → packed 8-bit RGB.
///
/// Byte-identical to [`scalar::yuv_420p16_to_rgb_row`].
///
/// # Safety
///
/// 1. NEON must be available on the current CPU.
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `u_half.len() >= width / 2`,
///    `v_half.len() >= width / 2`, `rgb_out.len() >= 3 * width`.
///
/// Thin wrapper over [`yuv_420p16_to_rgb_or_rgba_row`] with `ALPHA = false`.
#[cfg(feature = "yuv-planar")]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn yuv_420p16_to_rgb_row<const BE: bool>(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    yuv_420p16_to_rgb_or_rgba_row::<false, false, BE>(
      y, u_half, v_half, None, rgb_out, width, matrix, full_range,
    );
  }
}

/// NEON 16-bit YUV 4:2:0 → packed **8-bit RGBA** (`R, G, B, 0xFF`).
///
/// Thin wrapper over [`yuv_420p16_to_rgb_or_rgba_row`] with `ALPHA = true`.
///
/// # Safety
///
/// Same as [`yuv_420p16_to_rgb_row`] but `rgba_out.len() >= 4 * width`.
#[cfg(feature = "yuv-planar")]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn yuv_420p16_to_rgba_row<const BE: bool>(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    yuv_420p16_to_rgb_or_rgba_row::<true, false, BE>(
      y, u_half, v_half, None, rgba_out, width, matrix, full_range,
    );
  }
}

/// NEON 16-bit YUVA 4:2:0 → packed **8-bit RGBA** with the per-pixel
/// alpha byte **sourced from `a_src`** (depth-converted via `>> 8` to
/// fit `u8`). 16-bit alpha is full-range u16 — no AND-mask step.
/// Same numerical contract as [`yuv_420p16_to_rgba_row`] for R/G/B.
///
/// Thin wrapper over [`yuv_420p16_to_rgb_or_rgba_row`] with
/// `ALPHA = true, ALPHA_SRC = true`.
///
/// # Safety
///
/// Same as [`yuv_420p16_to_rgba_row`] plus `a_src.len() >= width`.
#[cfg(feature = "yuva")]
#[inline]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuv_420p16_to_rgba_with_alpha_src_row<const BE: bool>(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  a_src: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    yuv_420p16_to_rgb_or_rgba_row::<true, true, BE>(
      y,
      u_half,
      v_half,
      Some(a_src),
      rgba_out,
      width,
      matrix,
      full_range,
    );
  }
}

/// Shared NEON 16-bit YUV 4:2:0 kernel for [`yuv_420p16_to_rgb_row`]
/// (`ALPHA = false, ALPHA_SRC = false`, `vst3q_u8`),
/// [`yuv_420p16_to_rgba_row`] (`ALPHA = true, ALPHA_SRC = false`,
/// `vst4q_u8` with constant `0xFF` alpha) and
/// [`yuv_420p16_to_rgba_with_alpha_src_row`] (`ALPHA = true,
/// ALPHA_SRC = true`, `vst4q_u8` with the alpha lane loaded from
/// `a_src` and depth-converted via `vshrq_n_u16::<8>`).
///
/// # Safety
///
/// 1. NEON must be available on the current CPU.
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `u_half.len() >= width / 2`,
///    `v_half.len() >= width / 2`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
/// 4. When `ALPHA_SRC = true`: `a_src` must be `Some(_)` and
///    `a_src.unwrap().len() >= width`.
#[cfg(feature = "yuv-planar")]
#[inline]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuv_420p16_to_rgb_or_rgba_row<
  const ALPHA: bool,
  const ALPHA_SRC: bool,
  const BE: bool,
>(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  a_src: Option<&[u16]>,
  out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  // Source alpha requires RGBA output.
  const { assert!(!ALPHA_SRC || ALPHA) };
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert_eq!(width & 1, 0);
  debug_assert!(y.len() >= width);
  debug_assert!(u_half.len() >= width / 2);
  debug_assert!(v_half.len() >= width / 2);
  debug_assert!(out.len() >= width * bpp);
  if ALPHA_SRC {
    debug_assert!(a_src.as_ref().is_some_and(|s| s.len() >= width));
  }

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<16, 8>(full_range);
  let bias = scalar::chroma_bias::<16>(); // = 32768
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
      let y_vec_lo = endian::load_endian_u16x8::<BE>(y.as_ptr().add(x) as *const u8);
      let y_vec_hi = endian::load_endian_u16x8::<BE>(y.as_ptr().add(x + 8) as *const u8);
      let u_vec = endian::load_endian_u16x8::<BE>(u_half.as_ptr().add(x / 2) as *const u8);
      let v_vec = endian::load_endian_u16x8::<BE>(v_half.as_ptr().add(x / 2) as *const u8);

      // Unsigned-widen U/V to i32, subtract bias (32768 — does not fit i16).
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

      // i32 chroma is enough for u8 output (c_scale ≈ 127 keeps u_d small).
      let r_chroma = chroma_i16x8(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let g_chroma = chroma_i16x8(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let b_chroma = chroma_i16x8(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);

      let r_dup_lo = vzip1q_s16(r_chroma, r_chroma);
      let r_dup_hi = vzip2q_s16(r_chroma, r_chroma);
      let g_dup_lo = vzip1q_s16(g_chroma, g_chroma);
      let g_dup_hi = vzip2q_s16(g_chroma, g_chroma);
      let b_dup_lo = vzip1q_s16(b_chroma, b_chroma);
      let b_dup_hi = vzip2q_s16(b_chroma, b_chroma);

      let y_scaled_lo = scale_y_u16_to_i16(y_vec_lo, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y_u16_to_i16(y_vec_hi, y_off_v, y_scale_v, rnd_v);

      let r_u8 = vcombine_u8(
        vqmovun_s16_compat(vqaddq_s16(y_scaled_lo, r_dup_lo)),
        vqmovun_s16_compat(vqaddq_s16(y_scaled_hi, r_dup_hi)),
      );
      let g_u8 = vcombine_u8(
        vqmovun_s16_compat(vqaddq_s16(y_scaled_lo, g_dup_lo)),
        vqmovun_s16_compat(vqaddq_s16(y_scaled_hi, g_dup_hi)),
      );
      let b_u8 = vcombine_u8(
        vqmovun_s16_compat(vqaddq_s16(y_scaled_lo, b_dup_lo)),
        vqmovun_s16_compat(vqaddq_s16(y_scaled_hi, b_dup_hi)),
      );

      if ALPHA {
        let a_u8 = if ALPHA_SRC {
          // SAFETY (const-checked): ALPHA_SRC = true implies the
          // wrapper passed Some(_), validated by debug_assert above.
          // 16-bit alpha is full-range u16 — byte-swap if BE, then
          // `>> 8` to fit u8. `vshrq_n_u16` takes a const literal
          // shift; 8 is a literal so the intrinsic is well-formed.
          let a_ptr = a_src.as_ref().unwrap_unchecked().as_ptr();
          let a_lo_raw = endian::load_endian_u16x8::<BE>(a_ptr.add(x) as *const u8);
          let a_hi_raw = endian::load_endian_u16x8::<BE>(a_ptr.add(x + 8) as *const u8);
          let a_lo_u16 = vshrq_n_u16::<8>(a_lo_raw);
          let a_hi_u16 = vshrq_n_u16::<8>(a_hi_raw);
          vcombine_u8(vqmovn_u16_compat(a_lo_u16), vqmovn_u16_compat(a_hi_u16))
        } else {
          alpha_u8
        };
        vst4q_u8(
          out.as_mut_ptr().add(x * 4),
          uint8x16x4_t(r_u8, g_u8, b_u8, a_u8),
        );
      } else {
        vst3q_u8(out.as_mut_ptr().add(x * 3), uint8x16x3_t(r_u8, g_u8, b_u8));
      }
      x += 16;
    }

    if x < width {
      let tail_y = &y[x..width];
      let tail_u = &u_half[x / 2..width / 2];
      let tail_v = &v_half[x / 2..width / 2];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      if ALPHA_SRC {
        // SAFETY (const-checked): ALPHA_SRC = true implies Some(_).
        let tail_a = &a_src.as_ref().unwrap_unchecked()[x..width];
        scalar::yuv_420p16_to_rgba_with_alpha_src_row::<BE>(
          tail_y, tail_u, tail_v, tail_a, tail_out, tail_w, matrix, full_range,
        );
      } else if ALPHA {
        scalar::yuv_420p16_to_rgba_row::<BE>(
          tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range,
        );
      } else {
        scalar::yuv_420p16_to_rgb_row::<BE>(
          tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range,
        );
      }
    }
  }
}

/// NEON 16-bit planar YUV 4:2:0 → packed native-depth u16 RGB.
///
/// Both Y scaling and chroma multiply run in i64 (via `vmull_s32` +
/// `vshrq_n_s64::<15>`) to avoid i32 overflow at 16-bit limited-range scales.
/// Byte-identical to [`scalar::yuv_420p16_to_rgb_u16_row`].
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `u_half.len() >= width / 2`,
///    `v_half.len() >= width / 2`, `rgb_out.len() >= 3 * width`.
#[cfg(feature = "yuv-planar")]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn yuv_420p16_to_rgb_u16_row<const BE: bool>(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  unsafe {
    yuv_420p16_to_rgb_or_rgba_u16_row::<false, false, BE>(
      y, u_half, v_half, None, rgb_out, width, matrix, full_range,
    );
  }
}

/// NEON sibling of [`yuv_420p16_to_rgba_row`] for native-depth `u16`
/// output. Alpha is `0xFFFF` — matches `scalar::yuv_420p16_to_rgba_u16_row`.
///
/// # Safety
///
/// Same as [`yuv_420p16_to_rgb_u16_row`] plus `rgba_out.len() >= 4 * width`.
#[cfg(feature = "yuv-planar")]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn yuv_420p16_to_rgba_u16_row<const BE: bool>(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  unsafe {
    yuv_420p16_to_rgb_or_rgba_u16_row::<true, false, BE>(
      y, u_half, v_half, None, rgba_out, width, matrix, full_range,
    );
  }
}

/// NEON 16-bit YUVA 4:2:0 → **native-depth `u16`** packed RGBA with
/// the per-pixel alpha element **sourced from `a_src`** (full-range
/// u16, no mask, no shift) instead of being constant `0xFFFF`. Same
/// numerical contract as [`yuv_420p16_to_rgba_u16_row`] for R/G/B.
///
/// Thin wrapper over [`yuv_420p16_to_rgb_or_rgba_u16_row`] with
/// `ALPHA = true, ALPHA_SRC = true`.
///
/// # Safety
///
/// Same as [`yuv_420p16_to_rgba_u16_row`] plus `a_src.len() >= width`.
#[cfg(feature = "yuva")]
#[inline]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuv_420p16_to_rgba_u16_with_alpha_src_row<const BE: bool>(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  a_src: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    yuv_420p16_to_rgb_or_rgba_u16_row::<true, true, BE>(
      y,
      u_half,
      v_half,
      Some(a_src),
      rgba_out,
      width,
      matrix,
      full_range,
    );
  }
}

/// Shared NEON 16-bit YUV 4:2:0 → native-depth `u16` kernel.
/// - `ALPHA = false, ALPHA_SRC = false`: `vst3q_u16`.
/// - `ALPHA = true, ALPHA_SRC = false`: `vst4q_u16` with constant
///   alpha `0xFFFF`.
/// - `ALPHA = true, ALPHA_SRC = true`: `vst4q_u16` with the alpha
///   lane loaded directly from `a_src` (full-range u16, no mask).
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `u_half.len() >= width / 2`,
///    `v_half.len() >= width / 2`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
/// 4. When `ALPHA_SRC = true`: `a_src` must be `Some(_)` and
///    `a_src.unwrap().len() >= width`.
#[cfg(feature = "yuv-planar")]
#[inline]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuv_420p16_to_rgb_or_rgba_u16_row<
  const ALPHA: bool,
  const ALPHA_SRC: bool,
  const BE: bool,
>(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  a_src: Option<&[u16]>,
  out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  // Source alpha requires RGBA output.
  const { assert!(!ALPHA_SRC || ALPHA) };
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert_eq!(width & 1, 0);
  debug_assert!(y.len() >= width);
  debug_assert!(u_half.len() >= width / 2);
  debug_assert!(v_half.len() >= width / 2);
  debug_assert!(out.len() >= width * bpp);
  if ALPHA_SRC {
    debug_assert!(a_src.as_ref().is_some_and(|s| s.len() >= width));
  }

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
      let y_vec_lo = endian::load_endian_u16x8::<BE>(y.as_ptr().add(x) as *const u8);
      let y_vec_hi = endian::load_endian_u16x8::<BE>(y.as_ptr().add(x + 8) as *const u8);
      let u_vec = endian::load_endian_u16x8::<BE>(u_half.as_ptr().add(x / 2) as *const u8);
      let v_vec = endian::load_endian_u16x8::<BE>(v_half.as_ptr().add(x / 2) as *const u8);

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

      // i64 chroma: coeff * u_d can reach ~2.17x10⁹ at 16-bit scales.
      let r_ch_lo = chroma_i64x4(cru, crv, u_d_lo, v_d_lo, rnd64);
      let r_ch_hi = chroma_i64x4(cru, crv, u_d_hi, v_d_hi, rnd64);
      let g_ch_lo = chroma_i64x4(cgu, cgv, u_d_lo, v_d_lo, rnd64);
      let g_ch_hi = chroma_i64x4(cgu, cgv, u_d_hi, v_d_hi, rnd64);
      let b_ch_lo = chroma_i64x4(cbu, cbv, u_d_lo, v_d_lo, rnd64);
      let b_ch_hi = chroma_i64x4(cbu, cbv, u_d_hi, v_d_hi, rnd64);

      // Duplicate each chroma value into the slot for its 2 Y pixels.
      let r_cd_lo0 = vzip1q_s32(r_ch_lo, r_ch_lo);
      let r_cd_lo1 = vzip2q_s32(r_ch_lo, r_ch_lo);
      let r_cd_hi0 = vzip1q_s32(r_ch_hi, r_ch_hi);
      let r_cd_hi1 = vzip2q_s32(r_ch_hi, r_ch_hi);
      let g_cd_lo0 = vzip1q_s32(g_ch_lo, g_ch_lo);
      let g_cd_lo1 = vzip2q_s32(g_ch_lo, g_ch_lo);
      let g_cd_hi0 = vzip1q_s32(g_ch_hi, g_ch_hi);
      let g_cd_hi1 = vzip2q_s32(g_ch_hi, g_ch_hi);
      let b_cd_lo0 = vzip1q_s32(b_ch_lo, b_ch_lo);
      let b_cd_lo1 = vzip2q_s32(b_ch_lo, b_ch_lo);
      let b_cd_hi0 = vzip1q_s32(b_ch_hi, b_ch_hi);
      let b_cd_hi1 = vzip2q_s32(b_ch_hi, b_ch_hi);

      // i64 Y: (y - y_off) * y_scale can reach ~2.35x10⁹ at limited range.
      let y_lo_0 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(y_vec_lo)));
      let y_lo_1 = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(y_vec_lo)));
      let y_hi_0 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(y_vec_hi)));
      let y_hi_1 = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(y_vec_hi)));
      let ys_lo_0 = scale_y_u16_i64(y_lo_0, y_off_v, y_scale_d, rnd64);
      let ys_lo_1 = scale_y_u16_i64(y_lo_1, y_off_v, y_scale_d, rnd64);
      let ys_hi_0 = scale_y_u16_i64(y_hi_0, y_off_v, y_scale_d, rnd64);
      let ys_hi_1 = scale_y_u16_i64(y_hi_1, y_off_v, y_scale_d, rnd64);

      // Add Y + chroma; vqmovun_s32 saturates i32→u16 (clamps to [0, 65535]).
      let r_lo_u16 = vcombine_u16(
        vqmovun_s32_compat(vaddq_s32(ys_lo_0, r_cd_lo0)),
        vqmovun_s32_compat(vaddq_s32(ys_lo_1, r_cd_lo1)),
      );
      let r_hi_u16 = vcombine_u16(
        vqmovun_s32_compat(vaddq_s32(ys_hi_0, r_cd_hi0)),
        vqmovun_s32_compat(vaddq_s32(ys_hi_1, r_cd_hi1)),
      );
      let g_lo_u16 = vcombine_u16(
        vqmovun_s32_compat(vaddq_s32(ys_lo_0, g_cd_lo0)),
        vqmovun_s32_compat(vaddq_s32(ys_lo_1, g_cd_lo1)),
      );
      let g_hi_u16 = vcombine_u16(
        vqmovun_s32_compat(vaddq_s32(ys_hi_0, g_cd_hi0)),
        vqmovun_s32_compat(vaddq_s32(ys_hi_1, g_cd_hi1)),
      );
      let b_lo_u16 = vcombine_u16(
        vqmovun_s32_compat(vaddq_s32(ys_lo_0, b_cd_lo0)),
        vqmovun_s32_compat(vaddq_s32(ys_lo_1, b_cd_lo1)),
      );
      let b_hi_u16 = vcombine_u16(
        vqmovun_s32_compat(vaddq_s32(ys_hi_0, b_cd_hi0)),
        vqmovun_s32_compat(vaddq_s32(ys_hi_1, b_cd_hi1)),
      );

      if ALPHA {
        let (a_lo_v, a_hi_v) = if ALPHA_SRC {
          // SAFETY (const-checked): ALPHA_SRC = true implies the
          // wrapper passed Some(_), validated by debug_assert above.
          // 16-bit alpha is full-range u16 — byte-swap if BE, then
          // load 16 lanes directly, no further shift needed.
          let a_ptr = a_src.as_ref().unwrap_unchecked().as_ptr();
          (
            endian::load_endian_u16x8::<BE>(a_ptr.add(x) as *const u8),
            endian::load_endian_u16x8::<BE>(a_ptr.add(x + 8) as *const u8),
          )
        } else {
          (alpha_u16, alpha_u16)
        };
        vst4q_u16(
          out.as_mut_ptr().add(x * 4),
          uint16x8x4_t(r_lo_u16, g_lo_u16, b_lo_u16, a_lo_v),
        );
        vst4q_u16(
          out.as_mut_ptr().add(x * 4 + 32),
          uint16x8x4_t(r_hi_u16, g_hi_u16, b_hi_u16, a_hi_v),
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

    if x < width {
      let tail_y = &y[x..width];
      let tail_u = &u_half[x / 2..width / 2];
      let tail_v = &v_half[x / 2..width / 2];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      if ALPHA_SRC {
        // SAFETY (const-checked): ALPHA_SRC = true implies Some(_).
        let tail_a = &a_src.as_ref().unwrap_unchecked()[x..width];
        scalar::yuv_420p16_to_rgba_u16_with_alpha_src_row::<BE>(
          tail_y, tail_u, tail_v, tail_a, tail_out, tail_w, matrix, full_range,
        );
      } else if ALPHA {
        scalar::yuv_420p16_to_rgba_u16_row::<BE>(
          tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range,
        );
      } else {
        scalar::yuv_420p16_to_rgb_u16_row::<BE>(
          tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range,
        );
      }
    }
  }
}
/// NEON YUV 4:4:4 planar **16-bit** → packed **8-bit** RGB. Same i32
/// chroma pipeline as 10/12/14 (u8 output clamps `c_scale` down);
/// 1:1 chroma per Y pixel, no width parity.
///
/// Thin wrapper over [`yuv_444p16_to_rgb_or_rgba_row`] with `ALPHA = false`.
///
/// # Safety
///
/// Same as [`yuv_444p_n_to_rgb_row`] but with full `u16` samples.
#[cfg(feature = "yuv-planar")]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn yuv_444p16_to_rgb_row<const BE: bool>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    yuv_444p16_to_rgb_or_rgba_row::<false, false, BE>(
      y, u, v, None, rgb_out, width, matrix, full_range,
    );
  }
}

/// NEON YUV 4:4:4 planar **16-bit** → packed **8-bit RGBA**
/// (`R, G, B, 0xFF`). Same numerical contract as
/// [`yuv_444p16_to_rgb_row`].
///
/// Thin wrapper over [`yuv_444p16_to_rgb_or_rgba_row`] with `ALPHA = true`.
///
/// # Safety
///
/// Same as [`yuv_444p16_to_rgb_row`] but `rgba_out.len() >= 4 * width`.
#[cfg(feature = "yuv-planar")]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn yuv_444p16_to_rgba_row<const BE: bool>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    yuv_444p16_to_rgb_or_rgba_row::<true, false, BE>(
      y, u, v, None, rgba_out, width, matrix, full_range,
    );
  }
}

/// NEON YUVA 4:4:4 16-bit → packed **8-bit RGBA** with source alpha.
/// Same R/G/B numerical contract as [`yuv_444p16_to_rgba_row`]; the
/// per-pixel alpha byte is **sourced from `a_src`** (depth-converted
/// via `vshrq_n_u16::<8>` to fit `u8`).
///
/// Thin wrapper over [`yuv_444p16_to_rgb_or_rgba_row`] with
/// `ALPHA = true, ALPHA_SRC = true`.
///
/// # Safety
///
/// Same as [`yuv_444p16_to_rgba_row`] plus `a_src.len() >= width`.
#[cfg(feature = "yuva")]
#[inline]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuv_444p16_to_rgba_with_alpha_src_row<const BE: bool>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  a_src: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    yuv_444p16_to_rgb_or_rgba_row::<true, true, BE>(
      y,
      u,
      v,
      Some(a_src),
      rgba_out,
      width,
      matrix,
      full_range,
    );
  }
}

/// Shared NEON 16-bit YUV 4:4:4 kernel.
/// - `ALPHA = false, ALPHA_SRC = false`: `vst3q_u8`.
/// - `ALPHA = true, ALPHA_SRC = false`: `vst4q_u8` with constant
///   `0xFF` alpha.
/// - `ALPHA = true, ALPHA_SRC = true`: `vst4q_u8` with the alpha
///   lane loaded from `a_src` and depth-converted via
///   `vshrq_n_u16::<8>`.
///
/// # Safety
///
/// 1. NEON must be available on the current CPU.
/// 2. `y.len() >= width`, `u.len() >= width`, `v.len() >= width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
/// 3. If `ALPHA_SRC = true`, `a_src` is `Some(_)` with
///    `a_src.len() >= width`.
#[cfg(feature = "yuv-planar")]
#[inline]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuv_444p16_to_rgb_or_rgba_row<
  const ALPHA: bool,
  const ALPHA_SRC: bool,
  const BE: bool,
>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  a_src: Option<&[u16]>,
  out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  // Source alpha requires RGBA output.
  const { assert!(!ALPHA_SRC || ALPHA) };
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(y.len() >= width);
  debug_assert!(u.len() >= width);
  debug_assert!(v.len() >= width);
  debug_assert!(out.len() >= width * bpp);
  if ALPHA_SRC {
    debug_assert!(a_src.as_ref().is_some_and(|s| s.len() >= width));
  }

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
      let y_vec_lo = endian::load_endian_u16x8::<BE>(y.as_ptr().add(x) as *const u8);
      let y_vec_hi = endian::load_endian_u16x8::<BE>(y.as_ptr().add(x + 8) as *const u8);
      let u_vec_lo = endian::load_endian_u16x8::<BE>(u.as_ptr().add(x) as *const u8);
      let u_vec_hi = endian::load_endian_u16x8::<BE>(u.as_ptr().add(x + 8) as *const u8);
      let v_vec_lo = endian::load_endian_u16x8::<BE>(v.as_ptr().add(x) as *const u8);
      let v_vec_hi = endian::load_endian_u16x8::<BE>(v.as_ptr().add(x + 8) as *const u8);

      // Unsigned-widen + subtract 32768 in i32 (doesn't fit i16).
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
        vqmovun_s16_compat(vqaddq_s16(y_scaled_lo, r_chroma_lo)),
        vqmovun_s16_compat(vqaddq_s16(y_scaled_hi, r_chroma_hi)),
      );
      let g_u8 = vcombine_u8(
        vqmovun_s16_compat(vqaddq_s16(y_scaled_lo, g_chroma_lo)),
        vqmovun_s16_compat(vqaddq_s16(y_scaled_hi, g_chroma_hi)),
      );
      let b_u8 = vcombine_u8(
        vqmovun_s16_compat(vqaddq_s16(y_scaled_lo, b_chroma_lo)),
        vqmovun_s16_compat(vqaddq_s16(y_scaled_hi, b_chroma_hi)),
      );

      if ALPHA {
        let a_u8 = if ALPHA_SRC {
          // SAFETY (const-checked): ALPHA_SRC = true implies the
          // wrapper passed Some(_), validated by debug_assert above.
          // 16-bit alpha is full-range u16 — byte-swap if BE, then
          // `>> 8` to fit u8.
          let a_ptr = a_src.as_ref().unwrap_unchecked().as_ptr();
          let a_lo_raw = endian::load_endian_u16x8::<BE>(a_ptr.add(x) as *const u8);
          let a_hi_raw = endian::load_endian_u16x8::<BE>(a_ptr.add(x + 8) as *const u8);
          let a_lo_u16 = vshrq_n_u16::<8>(a_lo_raw);
          let a_hi_u16 = vshrq_n_u16::<8>(a_hi_raw);
          vcombine_u8(vqmovn_u16_compat(a_lo_u16), vqmovn_u16_compat(a_hi_u16))
        } else {
          alpha_u8
        };
        vst4q_u8(
          out.as_mut_ptr().add(x * 4),
          uint8x16x4_t(r_u8, g_u8, b_u8, a_u8),
        );
      } else {
        vst3q_u8(out.as_mut_ptr().add(x * 3), uint8x16x3_t(r_u8, g_u8, b_u8));
      }
      x += 16;
    }

    if x < width {
      let tail_y = &y[x..width];
      let tail_u = &u[x..width];
      let tail_v = &v[x..width];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      if ALPHA_SRC {
        // SAFETY (const-checked): ALPHA_SRC = true implies Some(_).
        let tail_a = &a_src.as_ref().unwrap_unchecked()[x..width];
        scalar::yuv_444p16_to_rgba_with_alpha_src_row::<BE>(
          tail_y, tail_u, tail_v, tail_a, tail_out, tail_w, matrix, full_range,
        );
      } else if ALPHA {
        scalar::yuv_444p16_to_rgba_row::<BE>(
          tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range,
        );
      } else {
        scalar::yuv_444p16_to_rgb_row::<BE>(
          tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range,
        );
      }
    }
  }
}

/// NEON YUV 4:4:4 planar **16-bit** → packed **native-depth u16** RGB.
/// i64 chroma + i64 Y (same widening as `yuv_420p16_to_rgb_u16_row`);
/// full-width U/V (no chroma duplication step).
///
/// Thin wrapper over [`yuv_444p16_to_rgb_or_rgba_u16_row`] with `ALPHA = false`.
///
/// # Safety
///
/// Same as [`yuv_444p16_to_rgb_row`] but `rgb_out: &mut [u16]`.
#[cfg(feature = "yuv-planar")]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn yuv_444p16_to_rgb_u16_row<const BE: bool>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    yuv_444p16_to_rgb_or_rgba_u16_row::<false, false, BE>(
      y, u, v, None, rgb_out, width, matrix, full_range,
    );
  }
}

/// NEON sibling of [`yuv_444p16_to_rgba_row`] for native-depth `u16`
/// output. Alpha samples are `0xFFFF` (opaque maximum at u16 range) —
/// matches `scalar::yuv_444p16_to_rgba_u16_row`.
///
/// Thin wrapper over [`yuv_444p16_to_rgb_or_rgba_u16_row`] with `ALPHA = true`.
///
/// # Safety
///
/// Same as [`yuv_444p16_to_rgb_u16_row`] plus `rgba_out.len() >= 4 * width`.
#[cfg(feature = "yuv-planar")]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn yuv_444p16_to_rgba_u16_row<const BE: bool>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    yuv_444p16_to_rgb_or_rgba_u16_row::<true, false, BE>(
      y, u, v, None, rgba_out, width, matrix, full_range,
    );
  }
}

/// NEON YUVA 4:4:4 16-bit → packed **native-depth `u16`** RGBA with
/// source alpha. Same R/G/B numerical contract as
/// [`yuv_444p16_to_rgba_u16_row`]; the per-pixel alpha element is
/// **sourced from `a_src`** at native depth (no shift needed).
///
/// Thin wrapper over [`yuv_444p16_to_rgb_or_rgba_u16_row`] with
/// `ALPHA = true, ALPHA_SRC = true`.
///
/// # Safety
///
/// Same as [`yuv_444p16_to_rgba_u16_row`] plus `a_src.len() >= width`.
#[cfg(feature = "yuva")]
#[inline]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuv_444p16_to_rgba_u16_with_alpha_src_row<const BE: bool>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  a_src: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    yuv_444p16_to_rgb_or_rgba_u16_row::<true, true, BE>(
      y,
      u,
      v,
      Some(a_src),
      rgba_out,
      width,
      matrix,
      full_range,
    );
  }
}

/// Shared NEON 16-bit YUV 4:4:4 → native-depth `u16` kernel.
/// - `ALPHA = false, ALPHA_SRC = false`: writes RGB triples via
///   `vst3q_u16`.
/// - `ALPHA = true, ALPHA_SRC = false`: writes RGBA quads via
///   `vst4q_u16` with constant alpha `0xFFFF`.
/// - `ALPHA = true, ALPHA_SRC = true`: writes RGBA quads with the
///   alpha lane loaded from `a_src` (16-bit input is full-range —
///   no shift needed).
///
/// # Safety
///
/// 1. **NEON must be available.**
/// 2. `y.len() >= width`, `u.len() >= width`, `v.len() >= width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
/// 3. If `ALPHA_SRC = true`, `a_src` is `Some(_)` with
///    `a_src.len() >= width`.
#[cfg(feature = "yuv-planar")]
#[inline]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuv_444p16_to_rgb_or_rgba_u16_row<
  const ALPHA: bool,
  const ALPHA_SRC: bool,
  const BE: bool,
>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  a_src: Option<&[u16]>,
  out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  // Source alpha requires RGBA output.
  const { assert!(!ALPHA_SRC || ALPHA) };
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(y.len() >= width);
  debug_assert!(u.len() >= width);
  debug_assert!(v.len() >= width);
  debug_assert!(out.len() >= width * bpp);
  if ALPHA_SRC {
    debug_assert!(a_src.as_ref().is_some_and(|s| s.len() >= width));
  }

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
      // 8 Y + 8 U + 8 V per iter — tighter block than 16 Y because
      // i64 chroma narrows throughput; matches the yuv_420p16 u16
      // kernel's cadence.
      let y_vec = endian::load_endian_u16x8::<BE>(y.as_ptr().add(x) as *const u8);
      let u_vec = endian::load_endian_u16x8::<BE>(u.as_ptr().add(x) as *const u8);
      let v_vec = endian::load_endian_u16x8::<BE>(v.as_ptr().add(x) as *const u8);

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

      // i64 chroma matches `yuv_420p16_to_rgb_u16_row`. 8 chroma
      // values computed as two `chroma_i64x4` calls.
      let r_ch_lo = chroma_i64x4(cru, crv, u_d_lo, v_d_lo, rnd64);
      let r_ch_hi = chroma_i64x4(cru, crv, u_d_hi, v_d_hi, rnd64);
      let g_ch_lo = chroma_i64x4(cgu, cgv, u_d_lo, v_d_lo, rnd64);
      let g_ch_hi = chroma_i64x4(cgu, cgv, u_d_hi, v_d_hi, rnd64);
      let b_ch_lo = chroma_i64x4(cbu, cbv, u_d_lo, v_d_lo, rnd64);
      let b_ch_hi = chroma_i64x4(cbu, cbv, u_d_hi, v_d_hi, rnd64);

      // i64 Y: 8 values as two i32x4 halves, scaled via i64 helper.
      let y_lo_i32 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(y_vec)));
      let y_hi_i32 = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(y_vec)));
      let ys_lo = scale_y_u16_i64(y_lo_i32, y_off_v, y_scale_d, rnd64);
      let ys_hi = scale_y_u16_i64(y_hi_i32, y_off_v, y_scale_d, rnd64);

      let r_u16 = vcombine_u16(
        vqmovun_s32_compat(vaddq_s32(ys_lo, r_ch_lo)),
        vqmovun_s32_compat(vaddq_s32(ys_hi, r_ch_hi)),
      );
      let g_u16 = vcombine_u16(
        vqmovun_s32_compat(vaddq_s32(ys_lo, g_ch_lo)),
        vqmovun_s32_compat(vaddq_s32(ys_hi, g_ch_hi)),
      );
      let b_u16 = vcombine_u16(
        vqmovun_s32_compat(vaddq_s32(ys_lo, b_ch_lo)),
        vqmovun_s32_compat(vaddq_s32(ys_hi, b_ch_hi)),
      );

      if ALPHA {
        let a_v = if ALPHA_SRC {
          // SAFETY (const-checked): ALPHA_SRC = true implies the
          // wrapper passed Some(_), validated by debug_assert above.
          // 16-bit alpha is full-range u16 — byte-swap if BE, then
          // load 8 lanes verbatim, no shift needed.
          endian::load_endian_u16x8::<BE>(
            a_src.as_ref().unwrap_unchecked().as_ptr().add(x) as *const u8
          )
        } else {
          alpha_u16
        };
        vst4q_u16(
          out.as_mut_ptr().add(x * 4),
          uint16x8x4_t(r_u16, g_u16, b_u16, a_v),
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
      let tail_u = &u[x..width];
      let tail_v = &v[x..width];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      if ALPHA_SRC {
        // SAFETY (const-checked): ALPHA_SRC = true implies Some(_).
        let tail_a = &a_src.as_ref().unwrap_unchecked()[x..width];
        scalar::yuv_444p16_to_rgba_u16_with_alpha_src_row::<BE>(
          tail_y, tail_u, tail_v, tail_a, tail_out, tail_w, matrix, full_range,
        );
      } else if ALPHA {
        scalar::yuv_444p16_to_rgba_u16_row::<BE>(
          tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range,
        );
      } else {
        scalar::yuv_444p16_to_rgb_u16_row::<BE>(
          tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range,
        );
      }
    }
  }
}

// ---- 16-bit planar YUV → HSV (staged via a reused 8-bit RGB chunk) ----
//
// The NEON twins of the scalar `yuv_*16_to_hsv_row` kernels. Same
// reused-8-bit-RGB-chunk staging as the high-bit (9/10/12/14) HSV
// kernels, over the NEON `yuv_*16_to_rgb_row::<BE>` 8-bit kernel +
// `rgb_to_hsv_row` — byte-identical to
// `rgb_to_hsv_row(yuv_*16_to_rgb_row::<BE>(...))` within the NEON
// tier, the same intermediate the existing 16-bit HSV path uses.

/// NEON: 16-bit YUV 4:2:0 planar → planar HSV bytes (OpenCV
/// encoding), staged via the NEON [`yuv_420p16_to_rgb_row`] +
/// [`rgb_to_hsv_row`]. `BE` selects the source byte order. Also serves
/// 4:2:2.
///
/// # Safety
///
/// 1. The NEON feature must be available.
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `u_half.len() >= width / 2`,
///    `v_half.len() >= width / 2`.
/// 4. `h_out.len()`, `s_out.len()`, `v_out.len()` `>= width`.
#[cfg(feature = "yuv-planar")]
#[inline]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuv_420p16_to_hsv_row<const BE: bool>(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0, "YUV 4:2:0 requires even width");
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(u_half.len() >= width / 2, "u_half row too short");
  debug_assert!(v_half.len() >= width / 2, "v_half row too short");
  debug_assert!(h_out.len() >= width, "h_out row too short");
  debug_assert!(s_out.len() >= width, "s_out row too short");
  debug_assert!(v_out.len() >= width, "v_out row too short");

  // SAFETY: the feature is the caller's obligation; the chunk filler
  // forwards to the NEON 16-bit 4:2:0 RGB kernel under the same
  // contract (its own scalar tail covers small n).
  unsafe {
    yuv_to_hsv_via_rgb_chunks(h_out, s_out, v_out, width, |offset, n, rgb| {
      yuv_420p16_to_rgb_row::<BE>(
        &y[offset..],
        &u_half[offset / 2..],
        &v_half[offset / 2..],
        rgb,
        n,
        matrix,
        full_range,
      );
    });
  }
}

/// NEON: 16-bit YUV 4:4:4 planar → planar HSV bytes, staged via the
/// NEON [`yuv_444p16_to_rgb_row`] + [`rgb_to_hsv_row`]. `BE` selects
/// the source byte order. Also serves 4:4:0.
///
/// # Safety
///
/// 1. The NEON feature must be available.
/// 2. `y.len() >= width`, `u.len() >= width`, `v.len() >= width`.
/// 3. `h_out.len()`, `s_out.len()`, `v_out.len()` `>= width`.
#[cfg(feature = "yuv-planar")]
#[inline]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuv_444p16_to_hsv_row<const BE: bool>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(u.len() >= width, "u row too short");
  debug_assert!(v.len() >= width, "v row too short");
  debug_assert!(h_out.len() >= width, "h_out row too short");
  debug_assert!(s_out.len() >= width, "s_out row too short");
  debug_assert!(v_out.len() >= width, "v_out row too short");

  // SAFETY: the feature is the caller's obligation; the chunk filler
  // forwards to the NEON 16-bit 4:4:4 RGB kernel under the same
  // contract (its own scalar tail covers small n).
  unsafe {
    yuv_to_hsv_via_rgb_chunks(h_out, s_out, v_out, width, |offset, n, rgb| {
      yuv_444p16_to_rgb_row::<BE>(
        &y[offset..],
        &u[offset..],
        &v[offset..],
        rgb,
        n,
        matrix,
        full_range,
      );
    });
  }
}
