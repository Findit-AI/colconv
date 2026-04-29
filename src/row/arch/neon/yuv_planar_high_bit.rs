use core::arch::aarch64::*;

use crate::{ColorMatrix, row::scalar};

use super::*;

/// NEON high‑bit‑depth YUV 4:2:0 (`BITS` ∈ {10, 12, 14}) → packed
/// **8‑bit** RGB.
///
/// Block size is 16 Y pixels / 8 chroma pairs per iteration. The
/// pipeline mirrors [`yuv_420_to_rgb_row`] byte‑for‑byte; the only
/// structural differences are:
/// - Loads are `vld1q_u16` (8 lanes of `u16`) instead of `vld1q_u8`
///   (16 lanes of `u8`), so each Y iteration needs two Y loads to
///   cover 16 pixels — there's no widening step because the samples
///   already live in 16‑bit lanes.
/// - Chroma bias is `128 << (BITS - 8)` (512 for 10‑bit, 2048 for
///   12‑bit, 8192 for 14‑bit) rather than 128.
/// - Range‑scaling params come from [`scalar::range_params_n`] with
///   the matching `BITS` const, so `y_scale` / `c_scale` map the
///   source depth to 8‑bit output in a single Q15 shift.
/// - Each load is AND‑masked to the low `BITS` bits so out‑of‑range
///   samples (e.g. high‑bit‑packed data mistakenly handed to the
///   low‑packed kernel) produce deterministic, backend‑consistent
///   output.
///
/// # Numerical contract
///
/// Byte‑identical to [`scalar::yuv_420p_n_to_rgb_row::<BITS>`] across
/// all supported bit depths.
///
/// # Safety
///
/// 1. **NEON must be available on the current CPU.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `u_half.len() >= width / 2`,
///    `v_half.len() >= width / 2`, `rgb_out.len() >= 3 * width`.
/// 4. `BITS` must be one of `{9, 10, 12, 14}` — the Q15 pipeline
///    overflows i32 at 16 bits; see [`scalar::range_params_n`].
///
/// Thin wrapper over [`yuv_420p_n_to_rgb_or_rgba_row`] with `ALPHA = false`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn yuv_420p_n_to_rgb_row<const BITS: u32>(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    yuv_420p_n_to_rgb_or_rgba_row::<BITS, false, false>(
      y, u_half, v_half, None, rgb_out, width, matrix, full_range,
    );
  }
}

/// NEON high-bit-depth YUV 4:2:0 → packed **8-bit RGBA** (`R, G, B, 0xFF`).
///
/// Same numerical contract as [`yuv_420p_n_to_rgb_row`]; the only
/// differences are the per-pixel stride (4 vs 3) and the constant
/// alpha byte (`0xFF`, opaque).
///
/// Thin wrapper over [`yuv_420p_n_to_rgb_or_rgba_row`] with `ALPHA = true`.
///
/// # Safety
///
/// Same as [`yuv_420p_n_to_rgb_row`] but `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn yuv_420p_n_to_rgba_row<const BITS: u32>(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    yuv_420p_n_to_rgb_or_rgba_row::<BITS, true, false>(
      y, u_half, v_half, None, rgba_out, width, matrix, full_range,
    );
  }
}

/// NEON YUVA 4:2:0 high-bit-depth → packed **8-bit RGBA** with the
/// per-pixel alpha byte **sourced from `a_src`** (depth-converted via
/// `>> (BITS - 8)` to fit `u8`) instead of being constant `0xFF`.
/// Same numerical contract as [`yuv_420p_n_to_rgba_row`] for R/G/B.
///
/// Thin wrapper over [`yuv_420p_n_to_rgb_or_rgba_row`] with
/// `ALPHA = true, ALPHA_SRC = true`.
///
/// # Safety
///
/// Same as [`yuv_420p_n_to_rgba_row`] plus `a_src.len() >= width`.
#[inline]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuv_420p_n_to_rgba_with_alpha_src_row<const BITS: u32>(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  a_src: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    yuv_420p_n_to_rgb_or_rgba_row::<BITS, true, true>(
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

/// Shared NEON high-bit-depth YUV 4:2:0 kernel for
/// [`yuv_420p_n_to_rgb_row`] (`ALPHA = false, ALPHA_SRC = false`,
/// `vst3q_u8`), [`yuv_420p_n_to_rgba_row`] (`ALPHA = true,
/// ALPHA_SRC = false`, `vst4q_u8` with constant `0xFF` alpha) and
/// [`yuv_420p_n_to_rgba_with_alpha_src_row`] (`ALPHA = true,
/// ALPHA_SRC = true`, `vst4q_u8` with the alpha lane loaded from
/// `a_src`, masked to BITS, and depth-converted via the variable
/// shift `>> (BITS - 8)`).
///
/// # Safety
///
/// 1. **NEON must be available on the current CPU.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `u_half.len() >= width / 2`,
///    `v_half.len() >= width / 2`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
/// 4. When `ALPHA_SRC = true`: `a_src` must be `Some(_)` and
///    `a_src.unwrap().len() >= width`.
/// 5. `BITS` must be one of `{9, 10, 12, 14}`.
#[inline]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuv_420p_n_to_rgb_or_rgba_row<
  const BITS: u32,
  const ALPHA: bool,
  const ALPHA_SRC: bool,
>(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  a_src: Option<&[u16]>,
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  const { assert!(BITS == 9 || BITS == 10 || BITS == 12 || BITS == 14) };
  // Source alpha requires RGBA output — there is no 3 bpp store with
  // alpha to put it in.
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
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<BITS, 8>(full_range);
  let bias = scalar::chroma_bias::<BITS>();
  const RND: i32 = 1 << 14;

  // SAFETY: NEON availability is the caller's obligation; the
  // dispatcher in `crate::row` verifies it. Pointer adds are bounded
  // by the `while x + 16 <= width` loop condition and the caller‑
  // promised slice lengths checked above.
  unsafe {
    let rnd_v = vdupq_n_s32(RND);
    let y_off_v = vdupq_n_s16(y_off as i16);
    let y_scale_v = vdupq_n_s32(y_scale);
    let c_scale_v = vdupq_n_s32(c_scale);
    let bias_v = vdupq_n_s16(bias as i16);
    let mask_v = vdupq_n_u16(scalar::bits_mask::<BITS>());
    let cru = vdupq_n_s32(coeffs.r_u());
    let crv = vdupq_n_s32(coeffs.r_v());
    let cgu = vdupq_n_s32(coeffs.g_u());
    let cgv = vdupq_n_s32(coeffs.g_v());
    let cbu = vdupq_n_s32(coeffs.b_u());
    let cbv = vdupq_n_s32(coeffs.b_v());
    let alpha_u8 = vdupq_n_u8(0xFF);

    let mut x = 0usize;
    while x + 16 <= width {
      // Two Y loads cover 16 lanes; one U load + one V load cover 8
      // chroma each. Each load is AND‑masked to the low BITS bits so
      // out‑of‑range samples (e.g. high‑bit‑packed data handed to
      // the low‑packed kernel) can never push an intermediate past
      // i16 range. For valid input the AND is a no‑op.
      let y_vec_lo = vandq_u16(vld1q_u16(y.as_ptr().add(x)), mask_v);
      let y_vec_hi = vandq_u16(vld1q_u16(y.as_ptr().add(x + 8)), mask_v);
      let u_vec = vandq_u16(vld1q_u16(u_half.as_ptr().add(x / 2)), mask_v);
      let v_vec = vandq_u16(vld1q_u16(v_half.as_ptr().add(x / 2)), mask_v);

      let y_lo = vreinterpretq_s16_u16(y_vec_lo);
      let y_hi = vreinterpretq_s16_u16(y_vec_hi);

      // c - 512 for 10‑bit chroma, fits i16 since c ≤ 1023.
      let u_i16 = vsubq_s16(vreinterpretq_s16_u16(u_vec), bias_v);
      let v_i16 = vsubq_s16(vreinterpretq_s16_u16(v_vec), bias_v);

      // Widen to i32x4 halves so the Q15 multiplies don't overflow.
      let u_lo_i32 = vmovl_s16(vget_low_s16(u_i16));
      let u_hi_i32 = vmovl_s16(vget_high_s16(u_i16));
      let v_lo_i32 = vmovl_s16(vget_low_s16(v_i16));
      let v_hi_i32 = vmovl_s16(vget_high_s16(v_i16));

      let u_d_lo = q15_shift(vaddq_s32(vmulq_s32(u_lo_i32, c_scale_v), rnd_v));
      let u_d_hi = q15_shift(vaddq_s32(vmulq_s32(u_hi_i32, c_scale_v), rnd_v));
      let v_d_lo = q15_shift(vaddq_s32(vmulq_s32(v_lo_i32, c_scale_v), rnd_v));
      let v_d_hi = q15_shift(vaddq_s32(vmulq_s32(v_hi_i32, c_scale_v), rnd_v));

      let r_chroma = chroma_i16x8(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let g_chroma = chroma_i16x8(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let b_chroma = chroma_i16x8(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);

      // Duplicate the 8 chroma lanes into 16‑lane pairs — identical
      // nearest‑neighbor upsample strategy as the 8‑bit kernel.
      let r_dup_lo = vzip1q_s16(r_chroma, r_chroma);
      let r_dup_hi = vzip2q_s16(r_chroma, r_chroma);
      let g_dup_lo = vzip1q_s16(g_chroma, g_chroma);
      let g_dup_hi = vzip2q_s16(g_chroma, g_chroma);
      let b_dup_lo = vzip1q_s16(b_chroma, b_chroma);
      let b_dup_hi = vzip2q_s16(b_chroma, b_chroma);

      let y_scaled_lo = scale_y(y_lo, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_hi, y_off_v, y_scale_v, rnd_v);

      // u8 output: saturate‑narrow i16 → u8 clamps to [0, 255].
      let b_u8 = vcombine_u8(
        vqmovun_s16(vqaddq_s16(y_scaled_lo, b_dup_lo)),
        vqmovun_s16(vqaddq_s16(y_scaled_hi, b_dup_hi)),
      );
      let g_u8 = vcombine_u8(
        vqmovun_s16(vqaddq_s16(y_scaled_lo, g_dup_lo)),
        vqmovun_s16(vqaddq_s16(y_scaled_hi, g_dup_hi)),
      );
      let r_u8 = vcombine_u8(
        vqmovun_s16(vqaddq_s16(y_scaled_lo, r_dup_lo)),
        vqmovun_s16(vqaddq_s16(y_scaled_hi, r_dup_hi)),
      );

      if ALPHA {
        let a_u8 = if ALPHA_SRC {
          // SAFETY (const-checked): ALPHA_SRC = true implies the
          // wrapper passed Some(_), validated by debug_assert above.
          let a_ptr = a_src.as_ref().unwrap_unchecked().as_ptr();
          let a_lo_u16 = vandq_u16(vld1q_u16(a_ptr.add(x)), mask_v);
          let a_hi_u16 = vandq_u16(vld1q_u16(a_ptr.add(x + 8)), mask_v);
          // Mask before shifting to harden against over-range source
          // alpha (e.g. 1024 at BITS=10), matching scalar. NEON's
          // `vshrq_n_u16` requires a literal const generic shift, but
          // `BITS - 8` is not a stable const expression on a const
          // generic — `vshlq_u16` with a negative count vector
          // performs the same logical right shift dynamically.
          let a_shr = vdupq_n_s16(-((BITS - 8) as i16));
          let a_lo_shifted = vshlq_u16(a_lo_u16, a_shr);
          let a_hi_shifted = vshlq_u16(a_hi_u16, a_shr);
          vcombine_u8(vqmovn_u16(a_lo_shifted), vqmovn_u16(a_hi_shifted))
        } else {
          alpha_u8
        };
        let rgba = uint8x16x4_t(r_u8, g_u8, b_u8, a_u8);
        vst4q_u8(out.as_mut_ptr().add(x * 4), rgba);
      } else {
        let rgb = uint8x16x3_t(r_u8, g_u8, b_u8);
        vst3q_u8(out.as_mut_ptr().add(x * 3), rgb);
      }

      x += 16;
    }

    // Scalar tail — remaining < 16 pixels (always even per 4:2:0).
    if x < width {
      let tail_y = &y[x..width];
      let tail_u = &u_half[x / 2..width / 2];
      let tail_v = &v_half[x / 2..width / 2];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      if ALPHA_SRC {
        // SAFETY (const-checked): ALPHA_SRC = true implies Some(_).
        let tail_a = &a_src.as_ref().unwrap_unchecked()[x..width];
        scalar::yuv_420p_n_to_rgba_with_alpha_src_row::<BITS>(
          tail_y, tail_u, tail_v, tail_a, tail_out, tail_w, matrix, full_range,
        );
      } else if ALPHA {
        scalar::yuv_420p_n_to_rgba_row::<BITS>(
          tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range,
        );
      } else {
        scalar::yuv_420p_n_to_rgb_row::<BITS>(
          tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range,
        );
      }
    }
  }
}

/// NEON high‑bit‑depth YUV 4:2:0 (`BITS` ∈ {10, 12, 14}) → packed
/// **native‑depth `u16`** RGB.
///
/// Block size is 16 Y pixels / 8 chroma pairs per iteration. Shares
/// all pre‑write math with [`yuv_420p_n_to_rgb_row`]; the only
/// difference is the final clamp + write:
/// - Y‑path scale is calibrated for `OUT_BITS = BITS` rather than 8,
///   so `y_scaled` lives in `[0, (1 << BITS) - 1]`.
/// - The `y_scaled + chroma` sum is clamped to `[0, (1 << BITS) - 1]`
///   with `vmaxq_s16(vminq_s16(_, max), 0)` — a simple saturate‑
///   narrow doesn't suffice because the sum can overshoot the
///   `BITS`-bit max without saturating at i16 bounds.
/// - Writes use two `vst3q_u16` calls per iteration — each handles 8
///   pixels × 3 channels = 24 `u16` elements, so two cover 16 pixels.
///
/// # Numerical contract
///
/// Identical to [`scalar::yuv_420p_n_to_rgb_u16_row::<BITS>`] across
/// supported `BITS` values.
///
/// # Safety
///
/// 1. **NEON must be available on the current CPU.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `u_half.len() >= width / 2`,
///    `v_half.len() >= width / 2`, `rgb_out.len() >= 3 * width`.
/// 4. `BITS` must be one of `{10, 12, 14}`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn yuv_420p_n_to_rgb_u16_row<const BITS: u32>(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  unsafe {
    yuv_420p_n_to_rgb_or_rgba_u16_row::<BITS, false, false>(
      y, u_half, v_half, None, rgb_out, width, matrix, full_range,
    );
  }
}

/// NEON sibling of [`yuv_420p_n_to_rgba_row`] for native-depth `u16`
/// output. Alpha samples are `(1 << BITS) - 1` (opaque maximum at the
/// input bit depth) — matches `scalar::yuv_420p_n_to_rgba_u16_row`.
///
/// # Safety
///
/// Same as [`yuv_420p_n_to_rgb_u16_row`], plus
/// `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn yuv_420p_n_to_rgba_u16_row<const BITS: u32>(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  unsafe {
    yuv_420p_n_to_rgb_or_rgba_u16_row::<BITS, true, false>(
      y, u_half, v_half, None, rgba_out, width, matrix, full_range,
    );
  }
}

/// NEON YUVA 4:2:0 high-bit-depth → **native-depth `u16`** packed
/// RGBA with the per-pixel alpha element **sourced from `a_src`**
/// (already at the source's native bit depth — no depth conversion)
/// instead of being the opaque maximum `(1 << BITS) - 1`. Same
/// numerical contract as [`yuv_420p_n_to_rgba_u16_row`] for R/G/B.
///
/// Thin wrapper over [`yuv_420p_n_to_rgb_or_rgba_u16_row`] with
/// `ALPHA = true, ALPHA_SRC = true`.
///
/// # Safety
///
/// Same as [`yuv_420p_n_to_rgba_u16_row`] plus `a_src.len() >= width`.
#[inline]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuv_420p_n_to_rgba_u16_with_alpha_src_row<const BITS: u32>(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  a_src: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    yuv_420p_n_to_rgb_or_rgba_u16_row::<BITS, true, true>(
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

/// Shared NEON high-bit YUV 4:2:0 → native-depth `u16` kernel for
/// [`yuv_420p_n_to_rgb_u16_row`] (`ALPHA = false, ALPHA_SRC = false`,
/// `vst3q_u16`), [`yuv_420p_n_to_rgba_u16_row`] (`ALPHA = true,
/// ALPHA_SRC = false`, `vst4q_u16` with constant alpha
/// `(1 << BITS) - 1`) and [`yuv_420p_n_to_rgba_u16_with_alpha_src_row`]
/// (`ALPHA = true, ALPHA_SRC = true`, `vst4q_u16` with the alpha lane
/// loaded from `a_src` and masked to native bit depth — no shift since
/// both the source alpha and the u16 output element are at the same
/// native bit depth).
///
/// # Safety
///
/// 1. **NEON must be available.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `u_half.len() >= width / 2`,
///    `v_half.len() >= width / 2`, `out.len() >= width * if ALPHA { 4 } else { 3 }`.
/// 4. When `ALPHA_SRC = true`: `a_src` must be `Some(_)` and
///    `a_src.unwrap().len() >= width`.
/// 5. `BITS` ∈ `{9, 10, 12, 14}`.
#[inline]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuv_420p_n_to_rgb_or_rgba_u16_row<
  const BITS: u32,
  const ALPHA: bool,
  const ALPHA_SRC: bool,
>(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  a_src: Option<&[u16]>,
  out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  const { assert!(BITS == 9 || BITS == 10 || BITS == 12 || BITS == 14) };
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
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<BITS, BITS>(full_range);
  let bias = scalar::chroma_bias::<BITS>();
  const RND: i32 = 1 << 14;
  let out_max: i16 = ((1i32 << BITS) - 1) as i16;

  // SAFETY: NEON availability is the caller's obligation; the
  // dispatcher in `crate::row` verifies it. Pointer adds are bounded
  // by the `while x + 16 <= width` loop condition and the caller‑
  // promised slice lengths.
  unsafe {
    let rnd_v = vdupq_n_s32(RND);
    let y_off_v = vdupq_n_s16(y_off as i16);
    let y_scale_v = vdupq_n_s32(y_scale);
    let c_scale_v = vdupq_n_s32(c_scale);
    let bias_v = vdupq_n_s16(bias as i16);
    let mask_v = vdupq_n_u16(scalar::bits_mask::<BITS>());
    let max_v = vdupq_n_s16(out_max);
    let zero_v = vdupq_n_s16(0);
    let cru = vdupq_n_s32(coeffs.r_u());
    let crv = vdupq_n_s32(coeffs.r_v());
    let cgu = vdupq_n_s32(coeffs.g_u());
    let cgv = vdupq_n_s32(coeffs.g_v());
    let cbu = vdupq_n_s32(coeffs.b_u());
    let cbv = vdupq_n_s32(coeffs.b_v());
    let alpha_u16 = vdupq_n_u16(out_max as u16);

    let mut x = 0usize;
    while x + 16 <= width {
      // AND‑mask each load to the low BITS bits so intermediates
      // stay within the i16 range the Q15 narrow steps expect — see
      // matching comment in [`yuv_420p_n_to_rgb_row`].
      let y_vec_lo = vandq_u16(vld1q_u16(y.as_ptr().add(x)), mask_v);
      let y_vec_hi = vandq_u16(vld1q_u16(y.as_ptr().add(x + 8)), mask_v);
      let u_vec = vandq_u16(vld1q_u16(u_half.as_ptr().add(x / 2)), mask_v);
      let v_vec = vandq_u16(vld1q_u16(v_half.as_ptr().add(x / 2)), mask_v);

      let y_lo = vreinterpretq_s16_u16(y_vec_lo);
      let y_hi = vreinterpretq_s16_u16(y_vec_hi);

      let u_i16 = vsubq_s16(vreinterpretq_s16_u16(u_vec), bias_v);
      let v_i16 = vsubq_s16(vreinterpretq_s16_u16(v_vec), bias_v);

      let u_lo_i32 = vmovl_s16(vget_low_s16(u_i16));
      let u_hi_i32 = vmovl_s16(vget_high_s16(u_i16));
      let v_lo_i32 = vmovl_s16(vget_low_s16(v_i16));
      let v_hi_i32 = vmovl_s16(vget_high_s16(v_i16));

      let u_d_lo = q15_shift(vaddq_s32(vmulq_s32(u_lo_i32, c_scale_v), rnd_v));
      let u_d_hi = q15_shift(vaddq_s32(vmulq_s32(u_hi_i32, c_scale_v), rnd_v));
      let v_d_lo = q15_shift(vaddq_s32(vmulq_s32(v_lo_i32, c_scale_v), rnd_v));
      let v_d_hi = q15_shift(vaddq_s32(vmulq_s32(v_hi_i32, c_scale_v), rnd_v));

      let r_chroma = chroma_i16x8(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let g_chroma = chroma_i16x8(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let b_chroma = chroma_i16x8(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);

      let r_dup_lo = vzip1q_s16(r_chroma, r_chroma);
      let r_dup_hi = vzip2q_s16(r_chroma, r_chroma);
      let g_dup_lo = vzip1q_s16(g_chroma, g_chroma);
      let g_dup_hi = vzip2q_s16(g_chroma, g_chroma);
      let b_dup_lo = vzip1q_s16(b_chroma, b_chroma);
      let b_dup_hi = vzip2q_s16(b_chroma, b_chroma);

      let y_scaled_lo = scale_y(y_lo, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_hi, y_off_v, y_scale_v, rnd_v);

      // Native‑depth output: add Y + chroma in i16, then clamp to
      // [0, (1 << BITS) - 1] explicitly. `vqaddq_s16` saturates at
      // i16 bounds (irrelevant here: |sum| stays well inside i16
      // for BITS ≤ 14), so the subsequent max/min clamps to the
      // native bit depth.
      let r_lo = clamp_u16_max(vqaddq_s16(y_scaled_lo, r_dup_lo), zero_v, max_v);
      let r_hi = clamp_u16_max(vqaddq_s16(y_scaled_hi, r_dup_hi), zero_v, max_v);
      let g_lo = clamp_u16_max(vqaddq_s16(y_scaled_lo, g_dup_lo), zero_v, max_v);
      let g_hi = clamp_u16_max(vqaddq_s16(y_scaled_hi, g_dup_hi), zero_v, max_v);
      let b_lo = clamp_u16_max(vqaddq_s16(y_scaled_lo, b_dup_lo), zero_v, max_v);
      let b_hi = clamp_u16_max(vqaddq_s16(y_scaled_hi, b_dup_hi), zero_v, max_v);

      if ALPHA {
        let (a_lo_v, a_hi_v) = if ALPHA_SRC {
          // SAFETY (const-checked): ALPHA_SRC = true implies the
          // wrapper passed Some(_), validated by debug_assert above.
          // No depth conversion — both source alpha and u16 output are
          // at the same native bit depth (BITS), so just mask off any
          // over-range bits to match the scalar reference.
          let a_ptr = a_src.as_ref().unwrap_unchecked().as_ptr();
          let lo = vandq_u16(vld1q_u16(a_ptr.add(x)), mask_v);
          let hi = vandq_u16(vld1q_u16(a_ptr.add(x + 8)), mask_v);
          (lo, hi)
        } else {
          (alpha_u16, alpha_u16)
        };
        let rgba_lo = uint16x8x4_t(r_lo, g_lo, b_lo, a_lo_v);
        let rgba_hi = uint16x8x4_t(r_hi, g_hi, b_hi, a_hi_v);
        vst4q_u16(out.as_mut_ptr().add(x * 4), rgba_lo);
        vst4q_u16(out.as_mut_ptr().add(x * 4 + 32), rgba_hi);
      } else {
        // Two interleaved u16 writes — each `vst3q_u16` covers 8 pixels.
        let rgb_lo = uint16x8x3_t(r_lo, g_lo, b_lo);
        let rgb_hi = uint16x8x3_t(r_hi, g_hi, b_hi);
        vst3q_u16(out.as_mut_ptr().add(x * 3), rgb_lo);
        vst3q_u16(out.as_mut_ptr().add(x * 3 + 24), rgb_hi);
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
        scalar::yuv_420p_n_to_rgba_u16_with_alpha_src_row::<BITS>(
          tail_y, tail_u, tail_v, tail_a, tail_out, tail_w, matrix, full_range,
        );
      } else if ALPHA {
        scalar::yuv_420p_n_to_rgba_u16_row::<BITS>(
          tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range,
        );
      } else {
        scalar::yuv_420p_n_to_rgb_u16_row::<BITS>(
          tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range,
        );
      }
    }
  }
}

/// Const-generic over `BITS ∈ {9, 10, 12, 14}`. Same structure as
/// [`yuv_420p_n_to_rgb_row`] but with full-width U/V (no chroma
/// duplication) and no width parity constraint.
///
/// Thin wrapper over [`yuv_444p_n_to_rgb_or_rgba_row`] with
/// `ALPHA = false, ALPHA_SRC = false`.
///
/// # Safety
///
/// 1. **NEON must be available.** 2. `y.len() >= width`, `u.len() >= width`, `v.len() >= width`, `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn yuv_444p_n_to_rgb_row<const BITS: u32>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    yuv_444p_n_to_rgb_or_rgba_row::<BITS, false, false>(
      y, u, v, rgb_out, width, matrix, full_range, None,
    );
  }
}

/// NEON YUV 4:4:4 planar high-bit-depth → packed **8-bit RGBA**
/// (`R, G, B, 0xFF`). Same numerical contract as
/// [`yuv_444p_n_to_rgb_row`]; the only differences are the per-pixel
/// stride (4 vs 3) and the constant alpha byte (`0xFF`, opaque).
///
/// Thin wrapper over [`yuv_444p_n_to_rgb_or_rgba_row`] with
/// `ALPHA = true, ALPHA_SRC = false`.
///
/// # Safety
///
/// Same as [`yuv_444p_n_to_rgb_row`] but `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn yuv_444p_n_to_rgba_row<const BITS: u32>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    yuv_444p_n_to_rgb_or_rgba_row::<BITS, true, false>(
      y, u, v, rgba_out, width, matrix, full_range, None,
    );
  }
}

/// NEON YUVA 4:4:4 planar high-bit-depth → packed **8-bit RGBA** with
/// the per-pixel alpha byte **sourced from `a_src`** (depth-converted
/// via `>> (BITS - 8)` to fit `u8`) instead of being constant `0xFF`.
/// Same numerical contract as [`yuv_444p_n_to_rgba_row`] for R/G/B.
///
/// Thin wrapper over [`yuv_444p_n_to_rgb_or_rgba_row`] with
/// `ALPHA = true, ALPHA_SRC = true`.
///
/// # Safety
///
/// Same as [`yuv_444p_n_to_rgba_row`] plus `a_src.len() >= width`.
#[inline]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuv_444p_n_to_rgba_with_alpha_src_row<const BITS: u32>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  a_src: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    yuv_444p_n_to_rgb_or_rgba_row::<BITS, true, true>(
      y,
      u,
      v,
      rgba_out,
      width,
      matrix,
      full_range,
      Some(a_src),
    );
  }
}

/// Shared NEON high-bit-depth YUV 4:4:4 kernel for
/// [`yuv_444p_n_to_rgb_row`] (`ALPHA = false, ALPHA_SRC = false`,
/// `vst3q_u8`), [`yuv_444p_n_to_rgba_row`] (`ALPHA = true,
/// ALPHA_SRC = false`, `vst4q_u8` with constant `0xFF` alpha vector)
/// and [`yuv_444p_n_to_rgba_with_alpha_src_row`] (`ALPHA = true,
/// ALPHA_SRC = true`, `vst4q_u8` with the alpha lane loaded and
/// depth-converted from `a_src`).
///
/// # Safety
///
/// 1. **NEON must be available.**
/// 2. `y.len() >= width`, `u.len() >= width`, `v.len() >= width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
/// 3. When `ALPHA_SRC = true`: `a_src` must be `Some(_)` and
///    `a_src.unwrap().len() >= width`.
/// 4. `BITS` must be one of `{9, 10, 12, 14}`.
#[inline]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuv_444p_n_to_rgb_or_rgba_row<
  const BITS: u32,
  const ALPHA: bool,
  const ALPHA_SRC: bool,
>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  a_src: Option<&[u16]>,
) {
  const { assert!(BITS == 9 || BITS == 10 || BITS == 12 || BITS == 14) };
  // Source alpha requires RGBA output — there is no 3 bpp store with
  // alpha to put it in.
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
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<BITS, 8>(full_range);
  let bias = scalar::chroma_bias::<BITS>();
  const RND: i32 = 1 << 14;

  unsafe {
    let rnd_v = vdupq_n_s32(RND);
    let y_off_v = vdupq_n_s16(y_off as i16);
    let y_scale_v = vdupq_n_s32(y_scale);
    let c_scale_v = vdupq_n_s32(c_scale);
    let bias_v = vdupq_n_s16(bias as i16);
    let mask_v = vdupq_n_u16(scalar::bits_mask::<BITS>());
    let cru = vdupq_n_s32(coeffs.r_u());
    let crv = vdupq_n_s32(coeffs.r_v());
    let cgu = vdupq_n_s32(coeffs.g_u());
    let cgv = vdupq_n_s32(coeffs.g_v());
    let cbu = vdupq_n_s32(coeffs.b_u());
    let cbv = vdupq_n_s32(coeffs.b_v());
    let alpha_u8 = vdupq_n_u8(0xFF);

    let mut x = 0usize;
    while x + 16 <= width {
      // 16 Y + 16 U + 16 V per iter, loaded as two u16x8 halves each.
      let y_vec_lo = vandq_u16(vld1q_u16(y.as_ptr().add(x)), mask_v);
      let y_vec_hi = vandq_u16(vld1q_u16(y.as_ptr().add(x + 8)), mask_v);
      let u_lo_u16 = vandq_u16(vld1q_u16(u.as_ptr().add(x)), mask_v);
      let u_hi_u16 = vandq_u16(vld1q_u16(u.as_ptr().add(x + 8)), mask_v);
      let v_lo_u16 = vandq_u16(vld1q_u16(v.as_ptr().add(x)), mask_v);
      let v_hi_u16 = vandq_u16(vld1q_u16(v.as_ptr().add(x + 8)), mask_v);

      let y_lo = vreinterpretq_s16_u16(y_vec_lo);
      let y_hi = vreinterpretq_s16_u16(y_vec_hi);

      let u_lo_i16 = vsubq_s16(vreinterpretq_s16_u16(u_lo_u16), bias_v);
      let u_hi_i16 = vsubq_s16(vreinterpretq_s16_u16(u_hi_u16), bias_v);
      let v_lo_i16 = vsubq_s16(vreinterpretq_s16_u16(v_lo_u16), bias_v);
      let v_hi_i16 = vsubq_s16(vreinterpretq_s16_u16(v_hi_u16), bias_v);

      // Widen each i16x8 → two i32x4 halves. Chroma is 1:1 with Y,
      // so we compute 8 chroma per Y-half directly.
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
        let a_u8 = if ALPHA_SRC {
          // SAFETY (const-checked): ALPHA_SRC = true implies the
          // wrapper passed Some(_), validated by debug_assert above.
          let a_ptr = a_src.as_ref().unwrap_unchecked().as_ptr();
          let a_lo_u16 = vandq_u16(vld1q_u16(a_ptr.add(x)), mask_v);
          let a_hi_u16 = vandq_u16(vld1q_u16(a_ptr.add(x + 8)), mask_v);
          // Mask before shifting to harden against over-range source
          // alpha (e.g. 1024 at BITS=10), matching scalar. NEON's
          // `vshrq_n_u16` requires a literal const generic shift, but
          // `BITS - 8` is not a stable const expression on a const
          // generic — `vshlq_u16` with a negative count vector
          // performs the same logical right shift dynamically.
          let a_shr = vdupq_n_s16(-((BITS - 8) as i16));
          let a_lo_shifted = vshlq_u16(a_lo_u16, a_shr);
          let a_hi_shifted = vshlq_u16(a_hi_u16, a_shr);
          vcombine_u8(vqmovn_u16(a_lo_shifted), vqmovn_u16(a_hi_shifted))
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
        scalar::yuv_444p_n_to_rgba_with_alpha_src_row::<BITS>(
          tail_y, tail_u, tail_v, tail_a, tail_out, tail_w, matrix, full_range,
        );
      } else if ALPHA {
        scalar::yuv_444p_n_to_rgba_row::<BITS>(
          tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range,
        );
      } else {
        scalar::yuv_444p_n_to_rgb_row::<BITS>(
          tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range,
        );
      }
    }
  }
}

/// NEON YUV 4:4:4 planar high-bit-depth → **native-depth u16** RGB.
/// Const-generic over `BITS ∈ {9, 10, 12, 14}`.
///
/// Thin wrapper over [`yuv_444p_n_to_rgb_or_rgba_u16_row`] with
/// `ALPHA = false, ALPHA_SRC = false`.
///
/// # Safety
///
/// Same as [`yuv_444p_n_to_rgb_row`] but `rgb_out: &mut [u16]`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn yuv_444p_n_to_rgb_u16_row<const BITS: u32>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    yuv_444p_n_to_rgb_or_rgba_u16_row::<BITS, false, false>(
      y, u, v, rgb_out, width, matrix, full_range, None,
    );
  }
}

/// NEON sibling of [`yuv_444p_n_to_rgba_row`] for native-depth `u16`
/// output. Alpha samples are `(1 << BITS) - 1` (opaque maximum at the
/// input bit depth) — matches `scalar::yuv_444p_n_to_rgba_u16_row`.
///
/// Thin wrapper over [`yuv_444p_n_to_rgb_or_rgba_u16_row`] with
/// `ALPHA = true, ALPHA_SRC = false`.
///
/// # Safety
///
/// Same as [`yuv_444p_n_to_rgb_u16_row`], plus
/// `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn yuv_444p_n_to_rgba_u16_row<const BITS: u32>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    yuv_444p_n_to_rgb_or_rgba_u16_row::<BITS, true, false>(
      y, u, v, rgba_out, width, matrix, full_range, None,
    );
  }
}

/// NEON YUVA 4:4:4 planar high-bit-depth → **native-depth `u16`**
/// packed RGBA with the per-pixel alpha element **sourced from
/// `a_src`** (already at the source's native bit depth — no depth
/// conversion) instead of being the opaque maximum `(1 << BITS) - 1`.
/// Same numerical contract as [`yuv_444p_n_to_rgba_u16_row`] for R/G/B.
///
/// Thin wrapper over [`yuv_444p_n_to_rgb_or_rgba_u16_row`] with
/// `ALPHA = true, ALPHA_SRC = true`.
///
/// # Safety
///
/// Same as [`yuv_444p_n_to_rgba_u16_row`] plus `a_src.len() >= width`.
#[inline]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuv_444p_n_to_rgba_u16_with_alpha_src_row<const BITS: u32>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  a_src: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    yuv_444p_n_to_rgb_or_rgba_u16_row::<BITS, true, true>(
      y,
      u,
      v,
      rgba_out,
      width,
      matrix,
      full_range,
      Some(a_src),
    );
  }
}

/// Shared NEON high-bit YUV 4:4:4 → native-depth `u16` kernel for
/// [`yuv_444p_n_to_rgb_u16_row`] (`ALPHA = false, ALPHA_SRC = false`,
/// `vst3q_u16`), [`yuv_444p_n_to_rgba_u16_row`] (`ALPHA = true,
/// ALPHA_SRC = false`, `vst4q_u16` with constant alpha
/// `(1 << BITS) - 1`) and [`yuv_444p_n_to_rgba_u16_with_alpha_src_row`]
/// (`ALPHA = true, ALPHA_SRC = true`, `vst4q_u16` with the alpha lane
/// loaded from `a_src` and masked to native bit depth — no shift since
/// both the source alpha and the u16 output element are at the same
/// native bit depth).
///
/// # Safety
///
/// 1. **NEON must be available.**
/// 2. `y.len() >= width`, `u.len() >= width`, `v.len() >= width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
/// 3. When `ALPHA_SRC = true`: `a_src` must be `Some(_)` and
///    `a_src.unwrap().len() >= width`.
/// 4. `BITS` ∈ `{9, 10, 12, 14}`.
#[inline]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuv_444p_n_to_rgb_or_rgba_u16_row<
  const BITS: u32,
  const ALPHA: bool,
  const ALPHA_SRC: bool,
>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  a_src: Option<&[u16]>,
) {
  // Compile-time guard — `out_max = ((1 << BITS) - 1) as i16` below
  // silently wraps to -1 at BITS=16, corrupting the u16 clamp. The
  // dedicated 16-bit u16-output path is `yuv_444p16_to_rgb_u16_row`.
  const { assert!(BITS == 9 || BITS == 10 || BITS == 12 || BITS == 14) };
  // Source alpha requires RGBA output — there is no 3 bpp store with
  // alpha to put it in.
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
    let mask_v = vdupq_n_u16(scalar::bits_mask::<BITS>());
    let max_v = vdupq_n_s16(out_max);
    let zero_v = vdupq_n_s16(0);
    let cru = vdupq_n_s32(coeffs.r_u());
    let crv = vdupq_n_s32(coeffs.r_v());
    let cgu = vdupq_n_s32(coeffs.g_u());
    let cgv = vdupq_n_s32(coeffs.g_v());
    let cbu = vdupq_n_s32(coeffs.b_u());
    let cbv = vdupq_n_s32(coeffs.b_v());
    let alpha_u16 = vdupq_n_u16(out_max as u16);

    let mut x = 0usize;
    while x + 16 <= width {
      let y_vec_lo = vandq_u16(vld1q_u16(y.as_ptr().add(x)), mask_v);
      let y_vec_hi = vandq_u16(vld1q_u16(y.as_ptr().add(x + 8)), mask_v);
      let u_lo_u16 = vandq_u16(vld1q_u16(u.as_ptr().add(x)), mask_v);
      let u_hi_u16 = vandq_u16(vld1q_u16(u.as_ptr().add(x + 8)), mask_v);
      let v_lo_u16 = vandq_u16(vld1q_u16(v.as_ptr().add(x)), mask_v);
      let v_hi_u16 = vandq_u16(vld1q_u16(v.as_ptr().add(x + 8)), mask_v);

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

      let r_lo = clamp_u16_max(vqaddq_s16(y_scaled_lo, r_chroma_lo), zero_v, max_v);
      let r_hi = clamp_u16_max(vqaddq_s16(y_scaled_hi, r_chroma_hi), zero_v, max_v);
      let g_lo = clamp_u16_max(vqaddq_s16(y_scaled_lo, g_chroma_lo), zero_v, max_v);
      let g_hi = clamp_u16_max(vqaddq_s16(y_scaled_hi, g_chroma_hi), zero_v, max_v);
      let b_lo = clamp_u16_max(vqaddq_s16(y_scaled_lo, b_chroma_lo), zero_v, max_v);
      let b_hi = clamp_u16_max(vqaddq_s16(y_scaled_hi, b_chroma_hi), zero_v, max_v);

      if ALPHA {
        let (a_lo_v, a_hi_v) = if ALPHA_SRC {
          // SAFETY (const-checked): ALPHA_SRC = true implies the
          // wrapper passed Some(_), validated by debug_assert above.
          // No depth conversion — both source alpha and u16 output are
          // at the same native bit depth (BITS), so just mask off any
          // over-range bits to match the scalar reference.
          let a_ptr = a_src.as_ref().unwrap_unchecked().as_ptr();
          let lo = vandq_u16(vld1q_u16(a_ptr.add(x)), mask_v);
          let hi = vandq_u16(vld1q_u16(a_ptr.add(x + 8)), mask_v);
          (lo, hi)
        } else {
          (alpha_u16, alpha_u16)
        };
        let rgba_lo = uint16x8x4_t(r_lo, g_lo, b_lo, a_lo_v);
        let rgba_hi = uint16x8x4_t(r_hi, g_hi, b_hi, a_hi_v);
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
      let tail_u = &u[x..width];
      let tail_v = &v[x..width];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      if ALPHA_SRC {
        // SAFETY (const-checked): ALPHA_SRC = true implies Some(_).
        let tail_a = &a_src.as_ref().unwrap_unchecked()[x..width];
        scalar::yuv_444p_n_to_rgba_u16_with_alpha_src_row::<BITS>(
          tail_y, tail_u, tail_v, tail_a, tail_out, tail_w, matrix, full_range,
        );
      } else if ALPHA {
        scalar::yuv_444p_n_to_rgba_u16_row::<BITS>(
          tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range,
        );
      } else {
        scalar::yuv_444p_n_to_rgb_u16_row::<BITS>(
          tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range,
        );
      }
    }
  }
}
