//! aarch64 NEON backend for the row primitives.
//!
//! Selected by [`crate::row`]'s dispatcher after
//! `is_aarch64_feature_detected!("neon")` returns true (runtime,
//! std‑gated) or `cfg!(target_feature = "neon")` evaluates true
//! (compile‑time, no‑std). The kernel itself carries
//! `#[target_feature(enable = "neon")]` so its intrinsics execute in
//! an explicitly NEON‑enabled context rather than one merely inherited
//! from the aarch64 target's default feature set.
//!
//! # Numerical contract
//!
//! The kernel uses i32 widening multiplies and the same
//! `(prod + (1 << 14)) >> 15` Q15 rounding as
//! [`crate::row::scalar::yuv_420_to_rgb_row`], so output is
//! **byte‑identical** to the scalar reference for every input. This is
//! asserted by the equivalence tests below.
//!
//! # Pipeline (per 16 Y pixels / 8 chroma samples)
//!
//! 1. Load 16 Y (`vld1q_u8`) + 8 U (`vld1_u8`) + 8 V (`vld1_u8`).
//! 2. Widen U/V to i16, subtract 128 → `u_i16`, `v_i16`.
//! 3. Widen to i32 and apply `c_scale` (Q15) → `u_d`, `v_d` (i32x4 × 2).
//! 4. Per channel C ∈ {R, G, B}:
//!    `C_chroma = (C_u * u_d + C_v * v_d + RND) >> 15` in i32,
//!    narrow‑saturate to i16x8 (8 lanes = 8 chroma pairs).
//! 5. Duplicate each chroma lane into its Y‑pair slot with
//!    `vzip1q_s16` / `vzip2q_s16` → 16 i16 chroma lanes matching the
//!    16 Y lanes (nearest‑neighbor upsample in registers, no memory
//!    traffic).
//! 6. Y path: `(Y - y_off) * y_scale + RND >> 15` in i32, narrow to i16.
//! 7. Saturating add Y + chroma per channel → i16x16.
//! 8. Saturate‑narrow to u8x16 and interleave with `vst3q_u8`.

use core::arch::aarch64::{
  float32x4_t, int16x8_t, int32x2_t, int32x4_t, int64x2_t, uint8x16_t, uint8x16x3_t, uint16x8_t,
  uint16x8x3_t, vaddq_f32, vaddq_s32, vaddq_s64, vandq_u16, vbslq_f32, vceqq_f32, vcltq_f32,
  vcombine_s16, vcombine_s32, vcombine_u8, vcombine_u16, vcvtq_f32_u32, vcvtq_u32_f32, vdivq_f32,
  vdup_n_s32, vdupq_n_f32, vdupq_n_s16, vdupq_n_s32, vdupq_n_s64, vdupq_n_u16, vget_high_s16,
  vget_high_s32, vget_high_u8, vget_high_u16, vget_low_s16, vget_low_s32, vget_low_u8,
  vget_low_u16, vld1_u8, vld1q_u8, vld1q_u16, vld2_u8, vld2q_u16, vld3q_u8, vmaxq_f32, vmaxq_s16,
  vminq_f32, vminq_s16, vmovl_s16, vmovl_u8, vmovl_u16, vmovn_s64, vmovn_u16, vmovn_u32, vmull_s32,
  vmulq_f32, vmulq_s32, vmvnq_u32, vqaddq_s16, vqmovn_s32, vqmovun_s16, vqmovun_s32,
  vreinterpretq_s16_u16, vreinterpretq_s32_u32, vreinterpretq_u16_s16, vshlq_u16, vshrq_n_s32,
  vshrq_n_s64, vst1q_u8, vst3q_u8, vst3q_u16, vsubq_f32, vsubq_s16, vsubq_s32, vzip1q_s16,
  vzip1q_s32, vzip2q_s16, vzip2q_s32,
};

use crate::{ColorMatrix, row::scalar};

/// NEON YUV 4:2:0 → packed RGB. Semantics match
/// [`scalar::yuv_420_to_rgb_row`] byte‑identically.
///
/// # Safety
///
/// The caller must uphold **all** of the following. Violating any
/// causes undefined behavior:
///
/// 1. **NEON must be available on the current CPU.** The dispatcher
///    in [`crate::row`] verifies this with
///    `is_aarch64_feature_detected!("neon")` (runtime) or
///    `cfg!(target_feature = "neon")` (compile‑time, no‑std). If you
///    call this kernel directly, you are responsible for the check —
///    executing NEON instructions on a CPU without NEON traps.
/// 2. `width & 1 == 0` (4:2:0 requires even width).
/// 3. `y.len() >= width`.
/// 4. `u_half.len() >= width / 2`.
/// 5. `v_half.len() >= width / 2`.
/// 6. `rgb_out.len() >= 3 * width`.
///
/// Bounds are verified by `debug_assert` in debug builds; release
/// builds trust the caller because the kernel relies on unchecked
/// pointer arithmetic (`vld1q_u8`, `vld1_u8`, `vst3q_u8`).
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn yuv_420_to_rgb_row(
  y: &[u8],
  u_half: &[u8],
  v_half: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0, "YUV 4:2:0 requires even width");
  debug_assert!(y.len() >= width);
  debug_assert!(u_half.len() >= width / 2);
  debug_assert!(v_half.len() >= width / 2);
  debug_assert!(rgb_out.len() >= width * 3);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params(full_range);
  const RND: i32 = 1 << 14;

  // SAFETY: NEON availability is the caller's obligation per the
  // `# Safety` section above; the dispatcher in `crate::row` checks
  // it. All pointer adds below are bounded by the
  // `while x + 16 <= width` loop condition and the caller‑promised
  // slice lengths checked above.
  unsafe {
    let rnd_v = vdupq_n_s32(RND);
    let y_off_v = vdupq_n_s16(y_off as i16);
    let y_scale_v = vdupq_n_s32(y_scale);
    let c_scale_v = vdupq_n_s32(c_scale);
    let mid128 = vdupq_n_s16(128);
    let cru = vdupq_n_s32(coeffs.r_u());
    let crv = vdupq_n_s32(coeffs.r_v());
    let cgu = vdupq_n_s32(coeffs.g_u());
    let cgv = vdupq_n_s32(coeffs.g_v());
    let cbu = vdupq_n_s32(coeffs.b_u());
    let cbv = vdupq_n_s32(coeffs.b_v());

    let mut x = 0usize;
    while x + 16 <= width {
      let y_vec = vld1q_u8(y.as_ptr().add(x));
      let u_vec = vld1_u8(u_half.as_ptr().add(x / 2));
      let v_vec = vld1_u8(v_half.as_ptr().add(x / 2));

      // Widen Y halves to i16x8 (unsigned → signed, Y ≤ 255 fits).
      let y_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(y_vec)));
      let y_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(y_vec)));

      // Widen U, V to i16x8 and subtract 128.
      let u_i16 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(u_vec)), mid128);
      let v_i16 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(v_vec)), mid128);

      // Split to i32x4 halves so the Q15 multiplies don't overflow.
      let u_lo_i32 = vmovl_s16(vget_low_s16(u_i16));
      let u_hi_i32 = vmovl_s16(vget_high_s16(u_i16));
      let v_lo_i32 = vmovl_s16(vget_low_s16(v_i16));
      let v_hi_i32 = vmovl_s16(vget_high_s16(v_i16));

      // u_d = (u * c_scale + RND) >> 15, bit‑exact to scalar.
      let u_d_lo = q15_shift(vaddq_s32(vmulq_s32(u_lo_i32, c_scale_v), rnd_v));
      let u_d_hi = q15_shift(vaddq_s32(vmulq_s32(u_hi_i32, c_scale_v), rnd_v));
      let v_d_lo = q15_shift(vaddq_s32(vmulq_s32(v_lo_i32, c_scale_v), rnd_v));
      let v_d_hi = q15_shift(vaddq_s32(vmulq_s32(v_hi_i32, c_scale_v), rnd_v));

      // Per‑channel chroma contribution, narrow to i16 for later adds.
      let r_chroma = chroma_i16x8(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let g_chroma = chroma_i16x8(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let b_chroma = chroma_i16x8(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);

      // Nearest‑neighbor upsample: duplicate each of the 8 chroma
      // lanes into an adjacent pair to cover 16 Y lanes. vzip1q takes
      // lanes 0..3 from both operands interleaved → [c0,c0,c1,c1,...];
      // vzip2q does the same for lanes 4..7.
      let r_dup_lo = vzip1q_s16(r_chroma, r_chroma);
      let r_dup_hi = vzip2q_s16(r_chroma, r_chroma);
      let g_dup_lo = vzip1q_s16(g_chroma, g_chroma);
      let g_dup_hi = vzip2q_s16(g_chroma, g_chroma);
      let b_dup_lo = vzip1q_s16(b_chroma, b_chroma);
      let b_dup_hi = vzip2q_s16(b_chroma, b_chroma);

      // Y path → i16x8 (two vectors covering 16 pixels).
      let y_scaled_lo = scale_y(y_lo, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_hi, y_off_v, y_scale_v, rnd_v);

      // B, G, R = saturating_add(Y, chroma); saturate‑narrow to u8.
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

      // vst3q_u8 writes 48 bytes as interleaved R, G, B triples.
      let rgb = uint8x16x3_t(r_u8, g_u8, b_u8);
      vst3q_u8(rgb_out.as_mut_ptr().add(x * 3), rgb);

      x += 16;
    }

    // Scalar tail for the 0..14 leftover pixels (always even, 4:2:0
    // requires even width so x/2 and width/2 are well‑defined).
    if x < width {
      scalar::yuv_420_to_rgb_row(
        &y[x..width],
        &u_half[x / 2..width / 2],
        &v_half[x / 2..width / 2],
        &mut rgb_out[x * 3..width * 3],
        width - x,
        matrix,
        full_range,
      );
    }
  }
}

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
/// 4. `BITS` must be one of `{10, 12, 14}` — the Q15 pipeline
///    overflows i32 at 16 bits; see [`scalar::range_params_n`].
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
  debug_assert_eq!(width & 1, 0);
  debug_assert!(y.len() >= width);
  debug_assert!(u_half.len() >= width / 2);
  debug_assert!(v_half.len() >= width / 2);
  debug_assert!(rgb_out.len() >= width * 3);

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

      let rgb = uint8x16x3_t(r_u8, g_u8, b_u8);
      vst3q_u8(rgb_out.as_mut_ptr().add(x * 3), rgb);

      x += 16;
    }

    // Scalar tail — remaining < 16 pixels (always even per 4:2:0).
    if x < width {
      scalar::yuv_420p_n_to_rgb_row::<BITS>(
        &y[x..width],
        &u_half[x / 2..width / 2],
        &v_half[x / 2..width / 2],
        &mut rgb_out[x * 3..width * 3],
        width - x,
        matrix,
        full_range,
      );
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
  debug_assert_eq!(width & 1, 0);
  debug_assert!(y.len() >= width);
  debug_assert!(u_half.len() >= width / 2);
  debug_assert!(v_half.len() >= width / 2);
  debug_assert!(rgb_out.len() >= width * 3);

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
      let r_lo = clamp_u10(vqaddq_s16(y_scaled_lo, r_dup_lo), zero_v, max_v);
      let r_hi = clamp_u10(vqaddq_s16(y_scaled_hi, r_dup_hi), zero_v, max_v);
      let g_lo = clamp_u10(vqaddq_s16(y_scaled_lo, g_dup_lo), zero_v, max_v);
      let g_hi = clamp_u10(vqaddq_s16(y_scaled_hi, g_dup_hi), zero_v, max_v);
      let b_lo = clamp_u10(vqaddq_s16(y_scaled_lo, b_dup_lo), zero_v, max_v);
      let b_hi = clamp_u10(vqaddq_s16(y_scaled_hi, b_dup_hi), zero_v, max_v);

      // Two interleaved u16 writes — each `vst3q_u16` covers 8 pixels.
      let rgb_lo = uint16x8x3_t(r_lo, g_lo, b_lo);
      let rgb_hi = uint16x8x3_t(r_hi, g_hi, b_hi);
      vst3q_u16(rgb_out.as_mut_ptr().add(x * 3), rgb_lo);
      vst3q_u16(rgb_out.as_mut_ptr().add(x * 3 + 24), rgb_hi);

      x += 16;
    }

    if x < width {
      scalar::yuv_420p_n_to_rgb_u16_row::<BITS>(
        &y[x..width],
        &u_half[x / 2..width / 2],
        &v_half[x / 2..width / 2],
        &mut rgb_out[x * 3..width * 3],
        width - x,
        matrix,
        full_range,
      );
    }
  }
}

/// Clamps an i16x8 vector to `[0, max]` and reinterprets to u16x8.
/// Used by the native‑depth u16 output paths to avoid `vqmovun_s16`'s
/// u8 saturation.
#[inline(always)]
fn clamp_u10(v: int16x8_t, zero_v: int16x8_t, max_v: int16x8_t) -> uint16x8_t {
  unsafe { vreinterpretq_u16_s16(vminq_s16(vmaxq_s16(v, zero_v), max_v)) }
}

/// NEON high‑bit‑packed semi‑planar (`BITS` ∈ {10, 12}: P010, P012)
/// → packed **8‑bit** RGB.
///
/// Block size 16 Y pixels / 8 chroma pairs per iteration. Differences
/// from [`yuv_420p_n_to_rgb_row`]:
/// - UV is semi‑planar interleaved (`U0, V0, U1, V1, …`), split in
///   one shot via `vld2q_u16` (returns separate U and V vectors).
/// - Each `u16` load is **right‑shifted by `16 - BITS`** — 6 for
///   P010, 4 for P012 — extracting the `BITS` active bits from the
///   high bits of each `u16` and clearing the low bits. The shift
///   runs via `vshlq_u16` with a negative loop‑invariant count so a
///   single kernel serves all supported bit depths.
///
/// After the shift, the rest of the pipeline is identical to the
/// low‑bit‑packed planar path — same `chroma_i16x8` / `scale_y` /
/// `chroma_dup` / `vst3q_u8` write, with `range_params_n::<BITS, 8>`
/// scaling.
///
/// # Numerical contract
///
/// Byte‑identical to [`scalar::p_n_to_rgb_row::<BITS>`] across all
/// supported `BITS` values.
///
/// # Safety
///
/// 1. **NEON must be available on the current CPU.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`,
///    `rgb_out.len() >= 3 * width`.
/// 4. `BITS` must be one of `{10, 12}`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn p_n_to_rgb_row<const BITS: u32>(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0);
  debug_assert!(y.len() >= width);
  debug_assert!(uv_half.len() >= width);
  debug_assert!(rgb_out.len() >= width * 3);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<BITS, 8>(full_range);
  let bias = scalar::chroma_bias::<BITS>();
  const RND: i32 = 1 << 14;

  // SAFETY: NEON availability is the caller's obligation.
  unsafe {
    let rnd_v = vdupq_n_s32(RND);
    let y_off_v = vdupq_n_s16(y_off as i16);
    let y_scale_v = vdupq_n_s32(y_scale);
    let c_scale_v = vdupq_n_s32(c_scale);
    let bias_v = vdupq_n_s16(bias as i16);
    // `vshlq_u16` performs right shift when the count is negative.
    // Count = -(16 - BITS) extracts the `BITS` active high bits.
    let shr_count = vdupq_n_s16(-((16 - BITS) as i16));
    let cru = vdupq_n_s32(coeffs.r_u());
    let crv = vdupq_n_s32(coeffs.r_v());
    let cgu = vdupq_n_s32(coeffs.g_u());
    let cgv = vdupq_n_s32(coeffs.g_v());
    let cbu = vdupq_n_s32(coeffs.b_u());
    let cbv = vdupq_n_s32(coeffs.b_v());

    let mut x = 0usize;
    while x + 16 <= width {
      // 16 Y pixels in two u16x8 loads, right-shifted by 16-BITS to
      // extract the active bits from the high-bit packing.
      let y_vec_lo = vshlq_u16(vld1q_u16(y.as_ptr().add(x)), shr_count);
      let y_vec_hi = vshlq_u16(vld1q_u16(y.as_ptr().add(x + 8)), shr_count);

      // Semi‑planar UV: `vld2q_u16` loads 16 interleaved `u16` elements
      // and returns (evens, odds) = (U, V) in one shot.
      let uv_pair = vld2q_u16(uv_half.as_ptr().add(x));
      let u_vec = vshlq_u16(uv_pair.0, shr_count);
      let v_vec = vshlq_u16(uv_pair.1, shr_count);

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

      let rgb = uint8x16x3_t(r_u8, g_u8, b_u8);
      vst3q_u8(rgb_out.as_mut_ptr().add(x * 3), rgb);

      x += 16;
    }

    if x < width {
      scalar::p_n_to_rgb_row::<BITS>(
        &y[x..width],
        &uv_half[x..width],
        &mut rgb_out[x * 3..width * 3],
        width - x,
        matrix,
        full_range,
      );
    }
  }
}

/// NEON high‑bit‑packed semi‑planar (`BITS` ∈ {10, 12}) → packed
/// **native‑depth `u16`** RGB (low‑bit‑packed output,
/// `yuv420p10le` / `yuv420p12le` convention — not P010/P012).
///
/// Same structure as [`super::neon::p_n_to_rgb_row`] up to the
/// chroma compute; the only differences are:
/// - `range_params_n::<BITS, BITS>` → larger scales targeting the
///   native‑depth output range.
/// - Clamp is explicit min/max to `[0, (1 << BITS) - 1]` via
///   [`clamp_u10`](crate::row::arch::neon::clamp_u10) — the helper
///   name is historical; the actual max is derived from `BITS` at
///   the call site (1023 for P010, 4095 for P012).
/// - Writes use two `vst3q_u16` calls per 16‑pixel block.
///
/// # Numerical contract
///
/// Byte‑identical to [`scalar::p_n_to_rgb_u16_row::<BITS>`] for the
/// monomorphized `BITS`.
///
/// # Safety
///
/// 1. **NEON must be available on the current CPU.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`,
///    `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn p_n_to_rgb_u16_row<const BITS: u32>(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0);
  debug_assert!(y.len() >= width);
  debug_assert!(uv_half.len() >= width);
  debug_assert!(rgb_out.len() >= width * 3);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<BITS, BITS>(full_range);
  let bias = scalar::chroma_bias::<BITS>();
  const RND: i32 = 1 << 14;
  let out_max: i16 = ((1i32 << BITS) - 1) as i16;

  // SAFETY: NEON availability is the caller's obligation.
  unsafe {
    let rnd_v = vdupq_n_s32(RND);
    let y_off_v = vdupq_n_s16(y_off as i16);
    let y_scale_v = vdupq_n_s32(y_scale);
    let c_scale_v = vdupq_n_s32(c_scale);
    let bias_v = vdupq_n_s16(bias as i16);
    let shr_count = vdupq_n_s16(-((16 - BITS) as i16));
    let max_v = vdupq_n_s16(out_max);
    let zero_v = vdupq_n_s16(0);
    let cru = vdupq_n_s32(coeffs.r_u());
    let crv = vdupq_n_s32(coeffs.r_v());
    let cgu = vdupq_n_s32(coeffs.g_u());
    let cgv = vdupq_n_s32(coeffs.g_v());
    let cbu = vdupq_n_s32(coeffs.b_u());
    let cbv = vdupq_n_s32(coeffs.b_v());

    let mut x = 0usize;
    while x + 16 <= width {
      let y_vec_lo = vshlq_u16(vld1q_u16(y.as_ptr().add(x)), shr_count);
      let y_vec_hi = vshlq_u16(vld1q_u16(y.as_ptr().add(x + 8)), shr_count);
      let uv_pair = vld2q_u16(uv_half.as_ptr().add(x));
      let u_vec = vshlq_u16(uv_pair.0, shr_count);
      let v_vec = vshlq_u16(uv_pair.1, shr_count);

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

      let r_lo = clamp_u10(vqaddq_s16(y_scaled_lo, r_dup_lo), zero_v, max_v);
      let r_hi = clamp_u10(vqaddq_s16(y_scaled_hi, r_dup_hi), zero_v, max_v);
      let g_lo = clamp_u10(vqaddq_s16(y_scaled_lo, g_dup_lo), zero_v, max_v);
      let g_hi = clamp_u10(vqaddq_s16(y_scaled_hi, g_dup_hi), zero_v, max_v);
      let b_lo = clamp_u10(vqaddq_s16(y_scaled_lo, b_dup_lo), zero_v, max_v);
      let b_hi = clamp_u10(vqaddq_s16(y_scaled_hi, b_dup_hi), zero_v, max_v);

      let rgb_lo = uint16x8x3_t(r_lo, g_lo, b_lo);
      let rgb_hi = uint16x8x3_t(r_hi, g_hi, b_hi);
      vst3q_u16(rgb_out.as_mut_ptr().add(x * 3), rgb_lo);
      vst3q_u16(rgb_out.as_mut_ptr().add(x * 3 + 24), rgb_hi);

      x += 16;
    }

    if x < width {
      scalar::p_n_to_rgb_u16_row::<BITS>(
        &y[x..width],
        &uv_half[x..width],
        &mut rgb_out[x * 3..width * 3],
        width - x,
        matrix,
        full_range,
      );
    }
  }
}

/// NEON NV12 → packed RGB (UV-ordered chroma). Thin wrapper over the
/// shared [`nv12_or_nv21_to_rgb_row_impl`] with `SWAP_UV = false`.
///
/// # Safety
///
/// Same as [`nv12_or_nv21_to_rgb_row_impl`].
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn nv12_to_rgb_row(
  y: &[u8],
  uv_half: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    nv12_or_nv21_to_rgb_row_impl::<false>(y, uv_half, rgb_out, width, matrix, full_range);
  }
}

/// NEON NV21 → packed RGB (VU-ordered chroma). Thin wrapper over
/// [`nv12_or_nv21_to_rgb_row_impl`] with `SWAP_UV = true`.
///
/// # Safety
///
/// Same as [`nv12_or_nv21_to_rgb_row_impl`].
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn nv21_to_rgb_row(
  y: &[u8],
  vu_half: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    nv12_or_nv21_to_rgb_row_impl::<true>(y, vu_half, rgb_out, width, matrix, full_range);
  }
}

/// Shared NEON NV12/NV21 kernel. `SWAP_UV = false` selects NV12
/// (even byte = U, odd = V); `SWAP_UV = true` selects NV21 (even =
/// V, odd = U). The const generic drives monomorphization — the
/// branch is eliminated in each instantiation and both wrappers
/// produce byte‑identical output to the scalar reference.
///
/// # Safety
///
/// 1. **NEON must be available on the current CPU.** The dispatcher
///    verifies this; direct callers are responsible.
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`.
/// 4. `uv_or_vu_half.len() >= width` (2 × (width / 2) interleaved bytes).
/// 5. `rgb_out.len() >= 3 * width`.
///
/// Bounds are `debug_assert`-checked; release builds trust the caller
/// because the kernel uses unchecked pointer arithmetic (`vld1q_u8`,
/// `vld2_u8`, `vst3q_u8`).
#[inline]
#[target_feature(enable = "neon")]
unsafe fn nv12_or_nv21_to_rgb_row_impl<const SWAP_UV: bool>(
  y: &[u8],
  uv_or_vu_half: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0, "NV12/NV21 require even width");
  debug_assert!(y.len() >= width);
  debug_assert!(uv_or_vu_half.len() >= width);
  debug_assert!(rgb_out.len() >= width * 3);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params(full_range);
  const RND: i32 = 1 << 14;

  // SAFETY: NEON availability is the caller's obligation; all pointer
  // adds below are bounded by the `while x + 16 <= width` loop
  // condition and the caller‑promised slice lengths above.
  unsafe {
    let rnd_v = vdupq_n_s32(RND);
    let y_off_v = vdupq_n_s16(y_off as i16);
    let y_scale_v = vdupq_n_s32(y_scale);
    let c_scale_v = vdupq_n_s32(c_scale);
    let mid128 = vdupq_n_s16(128);
    let cru = vdupq_n_s32(coeffs.r_u());
    let crv = vdupq_n_s32(coeffs.r_v());
    let cgu = vdupq_n_s32(coeffs.g_u());
    let cgv = vdupq_n_s32(coeffs.g_v());
    let cbu = vdupq_n_s32(coeffs.b_u());
    let cbv = vdupq_n_s32(coeffs.b_v());

    let mut x = 0usize;
    while x + 16 <= width {
      let y_vec = vld1q_u8(y.as_ptr().add(x));
      // 16 Y pixels → 8 chroma pairs. `vld2_u8` loads 16 interleaved
      // bytes and splits into (even-offset bytes, odd-offset bytes).
      // For NV12: even=U, odd=V. For NV21: even=V, odd=U, so we
      // swap which lane becomes `u_vec`. The `const SWAP_UV` makes
      // this a compile-time choice — no runtime branch.
      let uv_pair = vld2_u8(uv_or_vu_half.as_ptr().add(x));
      let (u_vec, v_vec) = if SWAP_UV {
        (uv_pair.1, uv_pair.0)
      } else {
        (uv_pair.0, uv_pair.1)
      };

      let y_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(y_vec)));
      let y_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(y_vec)));

      let u_i16 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(u_vec)), mid128);
      let v_i16 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(v_vec)), mid128);

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

      let rgb = uint8x16x3_t(r_u8, g_u8, b_u8);
      vst3q_u8(rgb_out.as_mut_ptr().add(x * 3), rgb);

      x += 16;
    }

    // Scalar tail for the 0..14 leftover pixels. Dispatch to the
    // matching scalar kernel based on SWAP_UV.
    if x < width {
      if SWAP_UV {
        scalar::nv21_to_rgb_row(
          &y[x..width],
          &uv_or_vu_half[x..width],
          &mut rgb_out[x * 3..width * 3],
          width - x,
          matrix,
          full_range,
        );
      } else {
        scalar::nv12_to_rgb_row(
          &y[x..width],
          &uv_or_vu_half[x..width],
          &mut rgb_out[x * 3..width * 3],
          width - x,
          matrix,
          full_range,
        );
      }
    }
  }
}

// The helpers below wrap NEON register‑only intrinsics (shifts, adds,
// multiplies, narrowing conversions, lane movers). None of them touch
// memory or take pointers, so there is no safety invariant to hoist to
// the caller — the functions themselves are safe. The `unsafe { ... }`
// blocks inside are only required because `core::arch::aarch64`
// intrinsics are marked `unsafe fn` in the standard library.
//
// `#[inline(always)]` guarantees these are inlined into the NEON‑
// enabled caller (`yuv_420_to_rgb_row` has
// `#[target_feature(enable = "neon")]`), so the intrinsics execute in
// a context where NEON is explicitly enabled — not just implicitly
// via the aarch64 target's default feature set.

/// `>>_a 15` shift (arithmetic, sign‑extending).
#[inline(always)]
fn q15_shift(v: int32x4_t) -> int32x4_t {
  unsafe { vshrq_n_s32::<15>(v) }
}

/// Build an i16x8 channel chroma vector from the 8 paired i32 chroma
/// samples. Mirrors the scalar
/// `(coeff_u * u_d + coeff_v * v_d + RND) >> 15`.
#[inline(always)]
fn chroma_i16x8(
  cu: int32x4_t,
  cv: int32x4_t,
  u_d_lo: int32x4_t,
  v_d_lo: int32x4_t,
  u_d_hi: int32x4_t,
  v_d_hi: int32x4_t,
  rnd: int32x4_t,
) -> int16x8_t {
  unsafe {
    let lo = vshrq_n_s32::<15>(vaddq_s32(
      vaddq_s32(vmulq_s32(cu, u_d_lo), vmulq_s32(cv, v_d_lo)),
      rnd,
    ));
    let hi = vshrq_n_s32::<15>(vaddq_s32(
      vaddq_s32(vmulq_s32(cu, u_d_hi), vmulq_s32(cv, v_d_hi)),
      rnd,
    ));
    vcombine_s16(vqmovn_s32(lo), vqmovn_s32(hi))
  }
}

/// `(Y - y_off) * y_scale + RND >> 15` returned as i16x8 (8 Y pixels).
#[inline(always)]
fn scale_y(
  y_i16: int16x8_t,
  y_off_v: int16x8_t,
  y_scale_v: int32x4_t,
  rnd: int32x4_t,
) -> int16x8_t {
  unsafe {
    let shifted = vsubq_s16(y_i16, y_off_v);
    let lo = vshrq_n_s32::<15>(vaddq_s32(
      vmulq_s32(vmovl_s16(vget_low_s16(shifted)), y_scale_v),
      rnd,
    ));
    let hi = vshrq_n_s32::<15>(vaddq_s32(
      vmulq_s32(vmovl_s16(vget_high_s16(shifted)), y_scale_v),
      rnd,
    ));
    vcombine_s16(vqmovn_s32(lo), vqmovn_s32(hi))
  }
}

// ===== 16-bit YUV → RGB ==================================================
//
// At 16-bit, two precision issues arise compared to the 10/12/14-bit generic:
//
// 1. The chroma bias (32768) and full-range u16 values (0..65535) do not fit
//    in i16, so all bias-subtractions happen in i32 after unsigned widening
//    (`vmovl_u16` → `vreinterpretq_s32_u32`).
//
// 2. For u16 output: `c_scale ≈ 37445` (limited range), so `coeff * u_d`
//    reaches ~2.17×10⁹ > i32 max; `y_scale ≈ 38304`, so `(y−y_off)*y_scale`
//    reaches ~2.35×10⁹ > i32 max. Both Y and chroma are widened to i64 via
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
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn yuv_420p16_to_rgb_row(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0);
  debug_assert!(y.len() >= width);
  debug_assert!(u_half.len() >= width / 2);
  debug_assert!(v_half.len() >= width / 2);
  debug_assert!(rgb_out.len() >= width * 3);

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

    let mut x = 0usize;
    while x + 16 <= width {
      let y_vec_lo = vld1q_u16(y.as_ptr().add(x));
      let y_vec_hi = vld1q_u16(y.as_ptr().add(x + 8));
      let u_vec = vld1q_u16(u_half.as_ptr().add(x / 2));
      let v_vec = vld1q_u16(v_half.as_ptr().add(x / 2));

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
        vqmovun_s16(vqaddq_s16(y_scaled_lo, r_dup_lo)),
        vqmovun_s16(vqaddq_s16(y_scaled_hi, r_dup_hi)),
      );
      let g_u8 = vcombine_u8(
        vqmovun_s16(vqaddq_s16(y_scaled_lo, g_dup_lo)),
        vqmovun_s16(vqaddq_s16(y_scaled_hi, g_dup_hi)),
      );
      let b_u8 = vcombine_u8(
        vqmovun_s16(vqaddq_s16(y_scaled_lo, b_dup_lo)),
        vqmovun_s16(vqaddq_s16(y_scaled_hi, b_dup_hi)),
      );

      vst3q_u8(
        rgb_out.as_mut_ptr().add(x * 3),
        uint8x16x3_t(r_u8, g_u8, b_u8),
      );
      x += 16;
    }

    if x < width {
      scalar::yuv_420p16_to_rgb_row(
        &y[x..width],
        &u_half[x / 2..width / 2],
        &v_half[x / 2..width / 2],
        &mut rgb_out[x * 3..width * 3],
        width - x,
        matrix,
        full_range,
      );
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
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn yuv_420p16_to_rgb_u16_row(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0);
  debug_assert!(y.len() >= width);
  debug_assert!(u_half.len() >= width / 2);
  debug_assert!(v_half.len() >= width / 2);
  debug_assert!(rgb_out.len() >= width * 3);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<16, 16>(full_range);
  let bias = scalar::chroma_bias::<16>();
  const RND: i32 = 1 << 14;

  unsafe {
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
      let y_vec_lo = vld1q_u16(y.as_ptr().add(x));
      let y_vec_hi = vld1q_u16(y.as_ptr().add(x + 8));
      let u_vec = vld1q_u16(u_half.as_ptr().add(x / 2));
      let v_vec = vld1q_u16(v_half.as_ptr().add(x / 2));

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

      // i64 chroma: coeff * u_d can reach ~2.17×10⁹ at 16-bit scales.
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

      // i64 Y: (y - y_off) * y_scale can reach ~2.35×10⁹ at limited range.
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
        vqmovun_s32(vaddq_s32(ys_lo_0, r_cd_lo0)),
        vqmovun_s32(vaddq_s32(ys_lo_1, r_cd_lo1)),
      );
      let r_hi_u16 = vcombine_u16(
        vqmovun_s32(vaddq_s32(ys_hi_0, r_cd_hi0)),
        vqmovun_s32(vaddq_s32(ys_hi_1, r_cd_hi1)),
      );
      let g_lo_u16 = vcombine_u16(
        vqmovun_s32(vaddq_s32(ys_lo_0, g_cd_lo0)),
        vqmovun_s32(vaddq_s32(ys_lo_1, g_cd_lo1)),
      );
      let g_hi_u16 = vcombine_u16(
        vqmovun_s32(vaddq_s32(ys_hi_0, g_cd_hi0)),
        vqmovun_s32(vaddq_s32(ys_hi_1, g_cd_hi1)),
      );
      let b_lo_u16 = vcombine_u16(
        vqmovun_s32(vaddq_s32(ys_lo_0, b_cd_lo0)),
        vqmovun_s32(vaddq_s32(ys_lo_1, b_cd_lo1)),
      );
      let b_hi_u16 = vcombine_u16(
        vqmovun_s32(vaddq_s32(ys_hi_0, b_cd_hi0)),
        vqmovun_s32(vaddq_s32(ys_hi_1, b_cd_hi1)),
      );

      vst3q_u16(
        rgb_out.as_mut_ptr().add(x * 3),
        uint16x8x3_t(r_lo_u16, g_lo_u16, b_lo_u16),
      );
      vst3q_u16(
        rgb_out.as_mut_ptr().add(x * 3 + 24),
        uint16x8x3_t(r_hi_u16, g_hi_u16, b_hi_u16),
      );
      x += 16;
    }

    if x < width {
      scalar::yuv_420p16_to_rgb_u16_row(
        &y[x..width],
        &u_half[x / 2..width / 2],
        &v_half[x / 2..width / 2],
        &mut rgb_out[x * 3..width * 3],
        width - x,
        matrix,
        full_range,
      );
    }
  }
}

/// NEON P016 (semi-planar 16-bit) → packed 8-bit RGB.
///
/// UV is interleaved (`U0, V0, U1, V1, …`), split via `vld2q_u16`.
/// Byte-identical to [`scalar::p16_to_rgb_row`].
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`, `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn p16_to_rgb_row(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0);
  debug_assert!(y.len() >= width);
  debug_assert!(uv_half.len() >= width);
  debug_assert!(rgb_out.len() >= width * 3);

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

    let mut x = 0usize;
    while x + 16 <= width {
      let y_vec_lo = vld1q_u16(y.as_ptr().add(x));
      let y_vec_hi = vld1q_u16(y.as_ptr().add(x + 8));
      let uv_pair = vld2q_u16(uv_half.as_ptr().add(x));
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
        vqmovun_s16(vqaddq_s16(y_scaled_lo, r_dup_lo)),
        vqmovun_s16(vqaddq_s16(y_scaled_hi, r_dup_hi)),
      );
      let g_u8 = vcombine_u8(
        vqmovun_s16(vqaddq_s16(y_scaled_lo, g_dup_lo)),
        vqmovun_s16(vqaddq_s16(y_scaled_hi, g_dup_hi)),
      );
      let b_u8 = vcombine_u8(
        vqmovun_s16(vqaddq_s16(y_scaled_lo, b_dup_lo)),
        vqmovun_s16(vqaddq_s16(y_scaled_hi, b_dup_hi)),
      );

      vst3q_u8(
        rgb_out.as_mut_ptr().add(x * 3),
        uint8x16x3_t(r_u8, g_u8, b_u8),
      );
      x += 16;
    }

    if x < width {
      scalar::p16_to_rgb_row(
        &y[x..width],
        &uv_half[x..width],
        &mut rgb_out[x * 3..width * 3],
        width - x,
        matrix,
        full_range,
      );
    }
  }
}

/// NEON P016 (semi-planar 16-bit) → packed native-depth u16 RGB.
///
/// Byte-identical to [`scalar::p16_to_rgb_u16_row`].
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`, `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn p16_to_rgb_u16_row(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0);
  debug_assert!(y.len() >= width);
  debug_assert!(uv_half.len() >= width);
  debug_assert!(rgb_out.len() >= width * 3);

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

    let mut x = 0usize;
    while x + 16 <= width {
      let y_vec_lo = vld1q_u16(y.as_ptr().add(x));
      let y_vec_hi = vld1q_u16(y.as_ptr().add(x + 8));
      let uv_pair = vld2q_u16(uv_half.as_ptr().add(x));
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

      let r_ch_lo = chroma_i64x4(cru, crv, u_d_lo, v_d_lo, rnd64);
      let r_ch_hi = chroma_i64x4(cru, crv, u_d_hi, v_d_hi, rnd64);
      let g_ch_lo = chroma_i64x4(cgu, cgv, u_d_lo, v_d_lo, rnd64);
      let g_ch_hi = chroma_i64x4(cgu, cgv, u_d_hi, v_d_hi, rnd64);
      let b_ch_lo = chroma_i64x4(cbu, cbv, u_d_lo, v_d_lo, rnd64);
      let b_ch_hi = chroma_i64x4(cbu, cbv, u_d_hi, v_d_hi, rnd64);

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

      let y_lo_0 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(y_vec_lo)));
      let y_lo_1 = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(y_vec_lo)));
      let y_hi_0 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(y_vec_hi)));
      let y_hi_1 = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(y_vec_hi)));
      let ys_lo_0 = scale_y_u16_i64(y_lo_0, y_off_v, y_scale_d, rnd64);
      let ys_lo_1 = scale_y_u16_i64(y_lo_1, y_off_v, y_scale_d, rnd64);
      let ys_hi_0 = scale_y_u16_i64(y_hi_0, y_off_v, y_scale_d, rnd64);
      let ys_hi_1 = scale_y_u16_i64(y_hi_1, y_off_v, y_scale_d, rnd64);

      let r_lo_u16 = vcombine_u16(
        vqmovun_s32(vaddq_s32(ys_lo_0, r_cd_lo0)),
        vqmovun_s32(vaddq_s32(ys_lo_1, r_cd_lo1)),
      );
      let r_hi_u16 = vcombine_u16(
        vqmovun_s32(vaddq_s32(ys_hi_0, r_cd_hi0)),
        vqmovun_s32(vaddq_s32(ys_hi_1, r_cd_hi1)),
      );
      let g_lo_u16 = vcombine_u16(
        vqmovun_s32(vaddq_s32(ys_lo_0, g_cd_lo0)),
        vqmovun_s32(vaddq_s32(ys_lo_1, g_cd_lo1)),
      );
      let g_hi_u16 = vcombine_u16(
        vqmovun_s32(vaddq_s32(ys_hi_0, g_cd_hi0)),
        vqmovun_s32(vaddq_s32(ys_hi_1, g_cd_hi1)),
      );
      let b_lo_u16 = vcombine_u16(
        vqmovun_s32(vaddq_s32(ys_lo_0, b_cd_lo0)),
        vqmovun_s32(vaddq_s32(ys_lo_1, b_cd_lo1)),
      );
      let b_hi_u16 = vcombine_u16(
        vqmovun_s32(vaddq_s32(ys_hi_0, b_cd_hi0)),
        vqmovun_s32(vaddq_s32(ys_hi_1, b_cd_hi1)),
      );

      vst3q_u16(
        rgb_out.as_mut_ptr().add(x * 3),
        uint16x8x3_t(r_lo_u16, g_lo_u16, b_lo_u16),
      );
      vst3q_u16(
        rgb_out.as_mut_ptr().add(x * 3 + 24),
        uint16x8x3_t(r_hi_u16, g_hi_u16, b_hi_u16),
      );
      x += 16;
    }

    if x < width {
      scalar::p16_to_rgb_u16_row(
        &y[x..width],
        &uv_half[x..width],
        &mut rgb_out[x * 3..width * 3],
        width - x,
        matrix,
        full_range,
      );
    }
  }
}

// ===== 16-bit helpers =====================================================

/// Scale 8 u16 Y pixels to i16x8 for the 16-bit u8-output path.
///
/// Unsigned-widens via `vmovl_u16`, subtracts `y_off` in i32, multiplies
/// by `y_scale` (small for u8 output — no i32 overflow), Q15-shifts, and
/// narrows to i16x8 with `vqmovn_s32`.
#[inline(always)]
fn scale_y_u16_to_i16(
  y_vec: uint16x8_t,
  y_off_v: int32x4_t,
  y_scale_v: int32x4_t,
  rnd_v: int32x4_t,
) -> int16x8_t {
  unsafe {
    let lo = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(y_vec)));
    let hi = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(y_vec)));
    let lo_s = vshrq_n_s32::<15>(vaddq_s32(
      vmulq_s32(vsubq_s32(lo, y_off_v), y_scale_v),
      rnd_v,
    ));
    let hi_s = vshrq_n_s32::<15>(vaddq_s32(
      vmulq_s32(vsubq_s32(hi, y_off_v), y_scale_v),
      rnd_v,
    ));
    vcombine_s16(vqmovn_s32(lo_s), vqmovn_s32(hi_s))
  }
}

/// `(cu*u_d + cv*v_d + RND) >> 15` in i64 for 4 chroma values → i32x4.
///
/// Used by the 16-bit u16-output path where `coeff * u_d` exceeds i32.
/// `vmull_s32` widens each 32×32 product to 64 bits, avoiding overflow.
#[inline(always)]
fn chroma_i64x4(
  cu: int32x4_t,
  cv: int32x4_t,
  u_d: int32x4_t,
  v_d: int32x4_t,
  rnd64: int64x2_t,
) -> int32x4_t {
  unsafe {
    let sum_lo = vshrq_n_s64::<15>(vaddq_s64(
      vaddq_s64(
        vmull_s32(vget_low_s32(cu), vget_low_s32(u_d)),
        vmull_s32(vget_low_s32(cv), vget_low_s32(v_d)),
      ),
      rnd64,
    ));
    let sum_hi = vshrq_n_s64::<15>(vaddq_s64(
      vaddq_s64(
        vmull_s32(vget_high_s32(cu), vget_high_s32(u_d)),
        vmull_s32(vget_high_s32(cv), vget_high_s32(v_d)),
      ),
      rnd64,
    ));
    vcombine_s32(vmovn_s64(sum_lo), vmovn_s64(sum_hi))
  }
}

/// Scale 4 u16 Y pixels via i64 widening for the 16-bit u16-output path.
///
/// `(y - y_off) * y_scale` can reach ~2.35×10⁹ at 16-bit limited range,
/// overflowing i32. `vmull_s32` widens to i64 before the Q15 shift.
/// Input `y_u32` is already unsigned-widened and reinterpreted as i32.
#[inline(always)]
fn scale_y_u16_i64(
  y_i32: int32x4_t,
  y_off_v: int32x4_t,
  y_scale_d: int32x2_t,
  rnd64: int64x2_t,
) -> int32x4_t {
  unsafe {
    let sub = vsubq_s32(y_i32, y_off_v);
    let lo = vshrq_n_s64::<15>(vaddq_s64(vmull_s32(vget_low_s32(sub), y_scale_d), rnd64));
    let hi = vshrq_n_s64::<15>(vaddq_s64(vmull_s32(vget_high_s32(sub), y_scale_d), rnd64));
    vcombine_s32(vmovn_s64(lo), vmovn_s64(hi))
  }
}

// ===== RGB → HSV =========================================================

/// NEON RGB → planar HSV. Semantics match
/// [`scalar::rgb_to_hsv_row`] byte‑identically.
///
/// # Safety
///
/// The caller must uphold **all** of the following. Violating any
/// causes undefined behavior:
///
/// 1. **NEON must be available on the current CPU** (same obligation
///    as `yuv_420_to_rgb_row`; the dispatcher checks this via
///    `is_aarch64_feature_detected!("neon")`).
/// 2. `rgb.len() >= 3 * width`.
/// 3. `h_out.len() >= width`.
/// 4. `s_out.len() >= width`.
/// 5. `v_out.len() >= width`.
///
/// Bounds are verified by `debug_assert` in debug builds. The kernel
/// relies on unchecked pointer arithmetic (`vld3q_u8`, `vst1q_u8`).
///
/// # Numerical contract
///
/// Bit‑identical to the scalar reference. Every scalar op has the
/// same SIMD counterpart in the same order: `vmaxq_f32` / `vminq_f32`
/// mirror `f32::max` / `f32::min`; `vdivq_f32` is true f32 division
/// (not reciprocal estimate); branch cascade uses `vbslq_f32` in the
/// same `delta == 0 → v == r → v == g → v == b` priority.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn rgb_to_hsv_row(
  rgb: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
) {
  debug_assert!(rgb.len() >= width * 3, "rgb row too short");
  debug_assert!(h_out.len() >= width, "H row too short");
  debug_assert!(s_out.len() >= width, "S row too short");
  debug_assert!(v_out.len() >= width, "V row too short");

  // SAFETY: NEON availability is the caller's obligation per the
  // `# Safety` section. All pointer adds below are bounded by the
  // `while x + 16 <= width` loop condition and the caller‑promised
  // slice lengths.
  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      // Deinterleave 16 RGB pixels → three u8x16 channel vectors.
      let rgb_vec = vld3q_u8(rgb.as_ptr().add(x * 3));
      let r_u8 = rgb_vec.0;
      let g_u8 = rgb_vec.1;
      let b_u8 = rgb_vec.2;

      // Widen each u8x16 to four f32x4 (16 values split into four
      // 4‑pixel groups) for the f32 HSV math.
      let (b0, b1, b2, b3) = u8x16_to_f32x4_quad(b_u8);
      let (g0, g1, g2, g3) = u8x16_to_f32x4_quad(g_u8);
      let (r0, r1, r2, r3) = u8x16_to_f32x4_quad(r_u8);

      // HSV per 4‑pixel group. Each returns (h_quant, s_quant, v_quant)
      // as f32x4 values already in [0, 179] / [0, 255] / [0, 255].
      let (h0, s0, v0) = hsv_group(b0, g0, r0);
      let (h1, s1, v1) = hsv_group(b1, g1, r1);
      let (h2, s2, v2) = hsv_group(b2, g2, r2);
      let (h3, s3, v3) = hsv_group(b3, g3, r3);

      // Truncate f32 → u8 via u32 intermediate, matching scalar `as u8`
      // (which saturates then truncates; values are pre‑clamped so the
      // narrow is safe).
      let h_u8 = f32x4_quad_to_u8x16(h0, h1, h2, h3);
      let s_u8 = f32x4_quad_to_u8x16(s0, s1, s2, s3);
      let v_u8 = f32x4_quad_to_u8x16(v0, v1, v2, v3);

      vst1q_u8(h_out.as_mut_ptr().add(x), h_u8);
      vst1q_u8(s_out.as_mut_ptr().add(x), s_u8);
      vst1q_u8(v_out.as_mut_ptr().add(x), v_u8);

      x += 16;
    }

    // Scalar tail for the 0..15 leftover pixels.
    if x < width {
      scalar::rgb_to_hsv_row(
        &rgb[x * 3..width * 3],
        &mut h_out[x..width],
        &mut s_out[x..width],
        &mut v_out[x..width],
        width - x,
      );
    }
  }
}

/// Widens a u8x16 to four f32x4 groups (covering lanes 0..3, 4..7,
/// 8..11, 12..15 respectively). Lanes are zero‑extended at each
/// widening step, so f32 values land exactly in `[0.0, 255.0]`.
#[inline(always)]
fn u8x16_to_f32x4_quad(v: uint8x16_t) -> (float32x4_t, float32x4_t, float32x4_t, float32x4_t) {
  unsafe {
    let u16_lo = vmovl_u8(vget_low_u8(v)); // u16x8 = lanes 0..7
    let u16_hi = vmovl_u8(vget_high_u8(v)); // u16x8 = lanes 8..15
    let u32_0 = vmovl_u16(vget_low_u16(u16_lo)); // lanes 0..3
    let u32_1 = vmovl_u16(vget_high_u16(u16_lo)); // lanes 4..7
    let u32_2 = vmovl_u16(vget_low_u16(u16_hi)); // lanes 8..11
    let u32_3 = vmovl_u16(vget_high_u16(u16_hi)); // lanes 12..15
    (
      vcvtq_f32_u32(u32_0),
      vcvtq_f32_u32(u32_1),
      vcvtq_f32_u32(u32_2),
      vcvtq_f32_u32(u32_3),
    )
  }
}

/// Computes HSV for 4 pixels. Mirrors the scalar `rgb_to_hsv_pixel`
/// op‑for‑op. Returns `(h_quant, s_quant, v_quant)` — each already
/// clamped to the scalar's output range (`h ≤ 179`, `s ≤ 255`,
/// `v ≤ 255`), still as f32 awaiting u8 conversion in the caller.
#[inline(always)]
fn hsv_group(
  b: float32x4_t,
  g: float32x4_t,
  r: float32x4_t,
) -> (float32x4_t, float32x4_t, float32x4_t) {
  unsafe {
    let zero = vdupq_n_f32(0.0);
    let half = vdupq_n_f32(0.5);
    let sixty = vdupq_n_f32(60.0);
    let one_twenty = vdupq_n_f32(120.0);
    let two_forty = vdupq_n_f32(240.0);
    let three_sixty = vdupq_n_f32(360.0);
    let one_seventy_nine = vdupq_n_f32(179.0);
    let two_fifty_five = vdupq_n_f32(255.0);

    // V = max(b, g, r); min = min(b, g, r); delta = V - min.
    // vmaxq_f32 / vminq_f32 are NaN‑tolerant, matching f32::max / f32::min.
    let v = vmaxq_f32(vmaxq_f32(b, g), r);
    let min_bgr = vminq_f32(vminq_f32(b, g), r);
    let delta = vsubq_f32(v, min_bgr);

    // S = if v == 0 { 0 } else { 255 * delta / v }.
    let mask_v_nonzero = vmvnq_u32(vceqq_f32(v, zero));
    let s_nonzero = vdivq_f32(vmulq_f32(two_fifty_five, delta), v);
    let s = vbslq_f32(mask_v_nonzero, s_nonzero, zero);

    // Hue — compute all three candidate formulas then select.
    let mask_delta_zero = vceqq_f32(delta, zero);
    let mask_v_is_r = vceqq_f32(v, r);
    let mask_v_is_g = vceqq_f32(v, g);

    // Branch 1 (v == r): 60 * (g - b) / delta, wrap negatives by +360.
    let h_r = {
      let raw = vdivq_f32(vmulq_f32(sixty, vsubq_f32(g, b)), delta);
      let mask_neg = vcltq_f32(raw, zero);
      vbslq_f32(mask_neg, vaddq_f32(raw, three_sixty), raw)
    };
    // Branch 2 (v == g): 60 * (b - r) / delta + 120.
    let h_g = vaddq_f32(
      vdivq_f32(vmulq_f32(sixty, vsubq_f32(b, r)), delta),
      one_twenty,
    );
    // Branch 3 (v == b, implicit): 60 * (r - g) / delta + 240.
    let h_b = vaddq_f32(
      vdivq_f32(vmulq_f32(sixty, vsubq_f32(r, g)), delta),
      two_forty,
    );

    // Cascade: if delta == 0 → 0; else if v == r → h_r; else if v == g
    // → h_g; else → h_b. Same priority order as the scalar.
    let hue_g_or_b = vbslq_f32(mask_v_is_g, h_g, h_b);
    let hue_nonzero_delta = vbslq_f32(mask_v_is_r, h_r, hue_g_or_b);
    let hue = vbslq_f32(mask_delta_zero, zero, hue_nonzero_delta);

    // Quantize to the scalar's output ranges. Scalar:
    //   h_quant = (hue * 0.5 + 0.5).clamp(0, 179)
    //   s_quant = (s + 0.5).clamp(0, 255)
    //   v_quant = (v + 0.5).clamp(0, 255)
    // clamp → vminq(vmaxq(v, lo), hi). Inputs are all finite so NaN
    // handling is irrelevant here.
    let h_quant = vminq_f32(
      vmaxq_f32(vaddq_f32(vmulq_f32(hue, half), half), zero),
      one_seventy_nine,
    );
    let s_quant = vminq_f32(vmaxq_f32(vaddq_f32(s, half), zero), two_fifty_five);
    let v_quant = vminq_f32(vmaxq_f32(vaddq_f32(v, half), zero), two_fifty_five);

    (h_quant, s_quant, v_quant)
  }
}

/// Converts four f32x4 vectors (16 values in [0, 255]) to one u8x16.
/// Truncates f32 → u32 via `vcvtq_u32_f32` (matches scalar `as u8`
/// which saturates‑then‑truncates; values are pre‑clamped so the
/// narrowing steps below are exact).
#[inline(always)]
fn f32x4_quad_to_u8x16(
  a: float32x4_t,
  b: float32x4_t,
  c: float32x4_t,
  d: float32x4_t,
) -> uint8x16_t {
  unsafe {
    let a_u32 = vcvtq_u32_f32(a);
    let b_u32 = vcvtq_u32_f32(b);
    let c_u32 = vcvtq_u32_f32(c);
    let d_u32 = vcvtq_u32_f32(d);
    let ab_u16 = vcombine_u16(vmovn_u32(a_u32), vmovn_u32(b_u32));
    let cd_u16 = vcombine_u16(vmovn_u32(c_u32), vmovn_u32(d_u32));
    vcombine_u8(vmovn_u16(ab_u16), vmovn_u16(cd_u16))
  }
}

// ===== BGR ↔ RGB byte swap ==============================================

/// Swaps the outer two channels of each packed 3‑byte triple. Drives
/// both `bgr_to_rgb_row` and `rgb_to_bgr_row` since the transformation
/// is self‑inverse.
///
/// NEON makes this almost free: `vld3q_u8` deinterleaves 16 pixels into
/// three channel vectors `(ch0, ch1, ch2)`, and `vst3q_u8` re‑interleaves
/// them — passing the deinterleaved vectors back in reversed order
/// `(ch2, ch1, ch0)` swaps the outer channels in a single store.
///
/// # Safety
///
/// 1. NEON must be available (same obligation as the other NEON kernels).
/// 2. `input.len() >= 3 * width`.
/// 3. `output.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn bgr_rgb_swap_row(input: &[u8], output: &mut [u8], width: usize) {
  debug_assert!(input.len() >= width * 3, "input row too short");
  debug_assert!(output.len() >= width * 3, "output row too short");

  // SAFETY: NEON availability is the caller's obligation per the
  // `# Safety` section. All pointer adds are bounded by the
  // `while x + 16 <= width` condition and the caller‑promised
  // slice lengths.
  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      let triple = vld3q_u8(input.as_ptr().add(x * 3));
      let swapped = uint8x16x3_t(triple.2, triple.1, triple.0);
      vst3q_u8(output.as_mut_ptr().add(x * 3), swapped);
      x += 16;
    }
    if x < width {
      scalar::bgr_rgb_swap_row(
        &input[x * 3..width * 3],
        &mut output[x * 3..width * 3],
        width - x,
      );
    }
  }
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;

  /// Deterministic scalar‑equivalence fixture. Fills Y/U/V with a
  /// hash‑like sequence so every byte varies, then compares byte‑exact.
  fn check_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
    let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
    let u: std::vec::Vec<u8> = (0..width / 2)
      .map(|i| ((i * 53 + 23) & 0xFF) as u8)
      .collect();
    let v: std::vec::Vec<u8> = (0..width / 2)
      .map(|i| ((i * 71 + 91) & 0xFF) as u8)
      .collect();
    let mut rgb_scalar = std::vec![0u8; width * 3];
    let mut rgb_neon = std::vec![0u8; width * 3];

    scalar::yuv_420_to_rgb_row(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
    unsafe {
      yuv_420_to_rgb_row(&y, &u, &v, &mut rgb_neon, width, matrix, full_range);
    }

    if rgb_scalar != rgb_neon {
      let first_diff = rgb_scalar
        .iter()
        .zip(rgb_neon.iter())
        .position(|(a, b)| a != b)
        .unwrap();
      panic!(
        "NEON diverges from scalar at byte {first_diff} (width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} neon={}",
        rgb_scalar[first_diff], rgb_neon[first_diff]
      );
    }
  }

  #[test]
  #[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
  fn neon_matches_scalar_all_matrices_16() {
    for m in [
      ColorMatrix::Bt601,
      ColorMatrix::Bt709,
      ColorMatrix::Bt2020Ncl,
      ColorMatrix::Smpte240m,
      ColorMatrix::Fcc,
      ColorMatrix::YCgCo,
    ] {
      for full in [true, false] {
        check_equivalence(16, m, full);
      }
    }
  }

  #[test]
  #[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
  fn neon_matches_scalar_width_32() {
    check_equivalence(32, ColorMatrix::Bt601, true);
    check_equivalence(32, ColorMatrix::Bt709, false);
    check_equivalence(32, ColorMatrix::YCgCo, true);
  }

  #[test]
  #[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
  fn neon_matches_scalar_width_1920() {
    check_equivalence(1920, ColorMatrix::Bt709, false);
  }

  #[test]
  #[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
  fn neon_matches_scalar_odd_tail_widths() {
    // Widths that leave a non‑trivial scalar tail (non‑multiple of 16).
    for w in [18usize, 30, 34, 1922] {
      check_equivalence(w, ColorMatrix::Bt601, false);
    }
  }

  // ---- nv12_to_rgb_row equivalence ------------------------------------

  /// Scalar‑equivalence fixture for NV12. Builds an interleaved UV row
  /// from the same U/V byte sequences used by the yuv420p fixture so a
  /// single NV12 call should produce byte‑identical output to the
  /// scalar NV12 reference.
  fn check_nv12_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
    let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
    let uv: std::vec::Vec<u8> = (0..width / 2)
      .flat_map(|i| {
        [
          ((i * 53 + 23) & 0xFF) as u8, // U_i
          ((i * 71 + 91) & 0xFF) as u8, // V_i
        ]
      })
      .collect();
    let mut rgb_scalar = std::vec![0u8; width * 3];
    let mut rgb_neon = std::vec![0u8; width * 3];

    scalar::nv12_to_rgb_row(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
    unsafe {
      nv12_to_rgb_row(&y, &uv, &mut rgb_neon, width, matrix, full_range);
    }

    if rgb_scalar != rgb_neon {
      let first_diff = rgb_scalar
        .iter()
        .zip(rgb_neon.iter())
        .position(|(a, b)| a != b)
        .unwrap();
      panic!(
        "NEON NV12 diverges from scalar at byte {first_diff} (width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} neon={}",
        rgb_scalar[first_diff], rgb_neon[first_diff]
      );
    }
  }

  /// Cross-format equivalence: the NV12 output must match the YUV420P
  /// output when fed the same U / V bytes interleaved. Guards against
  /// any stray deinterleave bug.
  fn check_nv12_matches_yuv420p(width: usize, matrix: ColorMatrix, full_range: bool) {
    let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
    let u: std::vec::Vec<u8> = (0..width / 2)
      .map(|i| ((i * 53 + 23) & 0xFF) as u8)
      .collect();
    let v: std::vec::Vec<u8> = (0..width / 2)
      .map(|i| ((i * 71 + 91) & 0xFF) as u8)
      .collect();
    let uv: std::vec::Vec<u8> = u.iter().zip(v.iter()).flat_map(|(a, b)| [*a, *b]).collect();

    let mut rgb_yuv420p = std::vec![0u8; width * 3];
    let mut rgb_nv12 = std::vec![0u8; width * 3];
    unsafe {
      yuv_420_to_rgb_row(&y, &u, &v, &mut rgb_yuv420p, width, matrix, full_range);
      nv12_to_rgb_row(&y, &uv, &mut rgb_nv12, width, matrix, full_range);
    }
    assert_eq!(
      rgb_yuv420p, rgb_nv12,
      "NV12 and YUV420P must produce byte-identical output for equivalent UV (width={width}, matrix={matrix:?}, full_range={full_range})"
    );
  }

  #[test]
  #[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
  fn nv12_neon_matches_scalar_all_matrices_16() {
    for m in [
      ColorMatrix::Bt601,
      ColorMatrix::Bt709,
      ColorMatrix::Bt2020Ncl,
      ColorMatrix::Smpte240m,
      ColorMatrix::Fcc,
      ColorMatrix::YCgCo,
    ] {
      for full in [true, false] {
        check_nv12_equivalence(16, m, full);
      }
    }
  }

  #[test]
  #[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
  fn nv12_neon_matches_scalar_width_1920() {
    check_nv12_equivalence(1920, ColorMatrix::Bt709, false);
  }

  #[test]
  #[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
  fn nv12_neon_matches_scalar_odd_tail_widths() {
    for w in [18usize, 30, 34, 1922] {
      check_nv12_equivalence(w, ColorMatrix::Bt601, false);
    }
  }

  #[test]
  #[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
  fn nv12_neon_matches_yuv420p_neon() {
    for w in [16usize, 30, 64, 1920] {
      check_nv12_matches_yuv420p(w, ColorMatrix::Bt709, false);
      check_nv12_matches_yuv420p(w, ColorMatrix::YCgCo, true);
    }
  }

  // ---- nv21_to_rgb_row equivalence ------------------------------------

  /// Scalar-equivalence for NV21. Same pseudo-random byte stream as
  /// the NV12 fixture, just handed to the VU-ordered kernel.
  fn check_nv21_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
    let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
    let vu: std::vec::Vec<u8> = (0..width / 2)
      .flat_map(|i| {
        [
          ((i * 53 + 23) & 0xFF) as u8, // V_i
          ((i * 71 + 91) & 0xFF) as u8, // U_i
        ]
      })
      .collect();
    let mut rgb_scalar = std::vec![0u8; width * 3];
    let mut rgb_neon = std::vec![0u8; width * 3];

    scalar::nv21_to_rgb_row(&y, &vu, &mut rgb_scalar, width, matrix, full_range);
    unsafe {
      nv21_to_rgb_row(&y, &vu, &mut rgb_neon, width, matrix, full_range);
    }

    if rgb_scalar != rgb_neon {
      let first_diff = rgb_scalar
        .iter()
        .zip(rgb_neon.iter())
        .position(|(a, b)| a != b)
        .unwrap();
      panic!(
        "NEON NV21 diverges from scalar at byte {first_diff} (width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} neon={}",
        rgb_scalar[first_diff], rgb_neon[first_diff]
      );
    }
  }

  /// Cross-format invariant: NV21 kernel on a VU-swapped byte stream
  /// must produce byte-identical output to the NV12 kernel on the
  /// UV-ordered original — proves the const-generic `SWAP_UV` path
  /// actually inverts the byte order.
  fn check_nv21_matches_nv12_with_swapped_uv(width: usize, matrix: ColorMatrix, full_range: bool) {
    let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
    // Build the UV stream (NV12 order), then the VU stream as the
    // same pairs byte-swapped.
    let uv: std::vec::Vec<u8> = (0..width / 2)
      .flat_map(|i| {
        [
          ((i * 53 + 23) & 0xFF) as u8, // U_i
          ((i * 71 + 91) & 0xFF) as u8, // V_i
        ]
      })
      .collect();
    let mut vu = std::vec![0u8; width];
    for i in 0..width / 2 {
      vu[2 * i] = uv[2 * i + 1]; // V_i
      vu[2 * i + 1] = uv[2 * i]; // U_i
    }

    let mut rgb_nv12 = std::vec![0u8; width * 3];
    let mut rgb_nv21 = std::vec![0u8; width * 3];
    unsafe {
      nv12_to_rgb_row(&y, &uv, &mut rgb_nv12, width, matrix, full_range);
      nv21_to_rgb_row(&y, &vu, &mut rgb_nv21, width, matrix, full_range);
    }
    assert_eq!(
      rgb_nv12, rgb_nv21,
      "NV21 should produce identical output to NV12 with byte-swapped chroma (width={width}, matrix={matrix:?})"
    );
  }

  #[test]
  #[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
  fn nv21_neon_matches_scalar_all_matrices_16() {
    for m in [
      ColorMatrix::Bt601,
      ColorMatrix::Bt709,
      ColorMatrix::Bt2020Ncl,
      ColorMatrix::Smpte240m,
      ColorMatrix::Fcc,
      ColorMatrix::YCgCo,
    ] {
      for full in [true, false] {
        check_nv21_equivalence(16, m, full);
      }
    }
  }

  #[test]
  #[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
  fn nv21_neon_matches_scalar_widths() {
    for w in [32usize, 1920, 18, 30, 34, 1922] {
      check_nv21_equivalence(w, ColorMatrix::Bt709, false);
    }
  }

  #[test]
  #[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
  fn nv21_neon_matches_nv12_swapped() {
    for w in [16usize, 30, 64, 1920] {
      check_nv21_matches_nv12_with_swapped_uv(w, ColorMatrix::Bt709, false);
      check_nv21_matches_nv12_with_swapped_uv(w, ColorMatrix::YCgCo, true);
    }
  }

  // ---- rgb_to_hsv_row equivalence ------------------------------------

  fn check_hsv_equivalence(rgb: &[u8], width: usize) {
    let mut h_scalar = std::vec![0u8; width];
    let mut s_scalar = std::vec![0u8; width];
    let mut v_scalar = std::vec![0u8; width];
    let mut h_neon = std::vec![0u8; width];
    let mut s_neon = std::vec![0u8; width];
    let mut v_neon = std::vec![0u8; width];

    scalar::rgb_to_hsv_row(rgb, &mut h_scalar, &mut s_scalar, &mut v_scalar, width);
    unsafe {
      rgb_to_hsv_row(rgb, &mut h_neon, &mut s_neon, &mut v_neon, width);
    }

    // Scalar uses integer LUT (matches OpenCV byte-exact), NEON uses
    // true f32 division. They can disagree by ±1 LSB at boundary
    // pixels — identical tolerance to what OpenCV reports between
    // their own scalar and SIMD HSV paths. Hue uses *circular*
    // distance since 0 and 179 are neighbors on the hue wheel: a pixel
    // at 360°≈0 in one path can land at 358°≈179 in the other due to
    // sign flips in delta with tiny f32 rounding.
    for (i, (a, b)) in h_scalar.iter().zip(h_neon.iter()).enumerate() {
      let d = a.abs_diff(*b);
      let circ = d.min(180 - d);
      assert!(circ <= 1, "H divergence at pixel {i}: scalar={a} neon={b}");
    }
    for (i, (a, b)) in s_scalar.iter().zip(s_neon.iter()).enumerate() {
      assert!(
        a.abs_diff(*b) <= 1,
        "S divergence at pixel {i}: scalar={a} neon={b}"
      );
    }
    for (i, (a, b)) in v_scalar.iter().zip(v_neon.iter()).enumerate() {
      assert!(
        a.abs_diff(*b) <= 1,
        "V divergence at pixel {i}: scalar={a} neon={b}"
      );
    }
  }

  fn pseudo_random_bgr(width: usize) -> std::vec::Vec<u8> {
    let n = width * 3;
    let mut out = std::vec::Vec::with_capacity(n);
    let mut state: u32 = 0x9E37_79B9;
    for _ in 0..n {
      state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
      out.push((state >> 8) as u8);
    }
    out
  }

  #[test]
  #[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
  fn hsv_neon_matches_scalar_pseudo_random_16() {
    let rgb = pseudo_random_bgr(16);
    check_hsv_equivalence(&rgb, 16);
  }

  #[test]
  #[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
  fn hsv_neon_matches_scalar_pseudo_random_1920() {
    let rgb = pseudo_random_bgr(1920);
    check_hsv_equivalence(&rgb, 1920);
  }

  #[test]
  #[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
  fn hsv_neon_matches_scalar_tail_widths() {
    // Widths that force a non‑trivial scalar tail (non‑multiple of 16).
    for w in [1usize, 7, 15, 17, 31, 1921] {
      let rgb = pseudo_random_bgr(w);
      check_hsv_equivalence(&rgb, w);
    }
  }

  #[test]
  #[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
  fn hsv_neon_matches_scalar_primaries_and_edges() {
    // Primary colors, grays, near‑saturation — exercise each hue branch
    // and the v==0, delta==0, h<0 wrap paths.
    let rgb: std::vec::Vec<u8> = [
      (0, 0, 0),       // black: v = 0 → s = 0, h = 0
      (255, 255, 255), // white: delta = 0 → s = 0, h = 0
      (128, 128, 128), // gray: delta = 0
      (255, 0, 0),     // pure red: v == r path
      (0, 255, 0),     // pure green: v == g path
      (0, 0, 255),     // pure blue: v == b path
      (255, 127, 0),   // red→yellow transition
      (0, 127, 255),   // blue→cyan
      (255, 0, 127),   // red→magenta
      (1, 2, 3),       // near black: small delta
      (254, 253, 252), // near white
      (150, 200, 10),  // arbitrary: v == g path, h > 0
      (150, 10, 200),  // arbitrary: v == b path
      (10, 200, 150),  // arbitrary: v == g
      (200, 100, 50),  // arbitrary: v == r
      (0, 64, 128),    // arbitrary: v == b
    ]
    .iter()
    .flat_map(|&(r, g, b)| [r, g, b])
    .collect();
    check_hsv_equivalence(&rgb, 16);
  }

  // ---- bgr_rgb_swap_row equivalence -----------------------------------

  fn check_swap_equivalence(width: usize) {
    let input = pseudo_random_bgr(width);
    let mut out_scalar = std::vec![0u8; width * 3];
    let mut out_neon = std::vec![0u8; width * 3];

    scalar::bgr_rgb_swap_row(&input, &mut out_scalar, width);
    unsafe {
      bgr_rgb_swap_row(&input, &mut out_neon, width);
    }

    assert_eq!(out_scalar, out_neon, "NEON swap diverges from scalar");

    // Byte 0 ↔ byte 2 should be swapped, byte 1 unchanged. Verify
    // the semantic directly.
    for x in 0..width {
      assert_eq!(
        out_scalar[x * 3],
        input[x * 3 + 2],
        "byte 0 != input byte 2"
      );
      assert_eq!(
        out_scalar[x * 3 + 1],
        input[x * 3 + 1],
        "middle byte changed"
      );
      assert_eq!(
        out_scalar[x * 3 + 2],
        input[x * 3],
        "byte 2 != input byte 0"
      );
    }
  }

  #[test]
  #[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
  fn swap_neon_matches_scalar_widths() {
    for w in [1usize, 15, 16, 17, 31, 32, 1920, 1921] {
      check_swap_equivalence(w);
    }
  }

  #[test]
  #[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
  fn swap_is_self_inverse() {
    let input = pseudo_random_bgr(64);
    let mut round_trip = std::vec![0u8; 64 * 3];
    let mut back = std::vec![0u8; 64 * 3];

    scalar::bgr_rgb_swap_row(&input, &mut round_trip, 64);
    scalar::bgr_rgb_swap_row(&round_trip, &mut back, 64);

    assert_eq!(input, back, "swap is not self-inverse");
  }

  // ---- yuv420p10 scalar-equivalence -----------------------------------

  /// Deterministic pseudo‑random `u16` samples in `[0, 1023]` — the
  /// 10‑bit range. Upper 6 bits always zero, so the generator matches
  /// real `yuv420p10le` bit patterns.
  fn p10_plane(n: usize, seed: usize) -> std::vec::Vec<u16> {
    (0..n)
      .map(|i| ((i * seed + seed * 3) & 0x3FF) as u16)
      .collect()
  }

  fn check_p10_u8_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
    let y = p10_plane(width, 37);
    let u = p10_plane(width / 2, 53);
    let v = p10_plane(width / 2, 71);
    let mut rgb_scalar = std::vec![0u8; width * 3];
    let mut rgb_neon = std::vec![0u8; width * 3];

    scalar::yuv_420p_n_to_rgb_row::<10>(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
    unsafe {
      yuv_420p_n_to_rgb_row::<10>(&y, &u, &v, &mut rgb_neon, width, matrix, full_range);
    }

    if rgb_scalar != rgb_neon {
      let first_diff = rgb_scalar
        .iter()
        .zip(rgb_neon.iter())
        .position(|(a, b)| a != b)
        .unwrap();
      panic!(
        "NEON 10→u8 diverges from scalar at byte {first_diff} (width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} neon={}",
        rgb_scalar[first_diff], rgb_neon[first_diff]
      );
    }
  }

  fn check_p10_u16_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
    let y = p10_plane(width, 37);
    let u = p10_plane(width / 2, 53);
    let v = p10_plane(width / 2, 71);
    let mut rgb_scalar = std::vec![0u16; width * 3];
    let mut rgb_neon = std::vec![0u16; width * 3];

    scalar::yuv_420p_n_to_rgb_u16_row::<10>(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
    unsafe {
      yuv_420p_n_to_rgb_u16_row::<10>(&y, &u, &v, &mut rgb_neon, width, matrix, full_range);
    }

    if rgb_scalar != rgb_neon {
      let first_diff = rgb_scalar
        .iter()
        .zip(rgb_neon.iter())
        .position(|(a, b)| a != b)
        .unwrap();
      panic!(
        "NEON 10→u16 diverges from scalar at elem {first_diff} (width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} neon={}",
        rgb_scalar[first_diff], rgb_neon[first_diff]
      );
    }
  }

  #[test]
  #[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
  fn neon_p10_u8_matches_scalar_all_matrices_16() {
    for m in [
      ColorMatrix::Bt601,
      ColorMatrix::Bt709,
      ColorMatrix::Bt2020Ncl,
      ColorMatrix::Smpte240m,
      ColorMatrix::Fcc,
      ColorMatrix::YCgCo,
    ] {
      for full in [true, false] {
        check_p10_u8_equivalence(16, m, full);
      }
    }
  }

  #[test]
  #[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
  fn neon_p10_u16_matches_scalar_all_matrices_16() {
    for m in [
      ColorMatrix::Bt601,
      ColorMatrix::Bt709,
      ColorMatrix::Bt2020Ncl,
      ColorMatrix::Smpte240m,
      ColorMatrix::Fcc,
      ColorMatrix::YCgCo,
    ] {
      for full in [true, false] {
        check_p10_u16_equivalence(16, m, full);
      }
    }
  }

  #[test]
  #[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
  fn neon_p10_matches_scalar_odd_tail_widths() {
    for w in [18usize, 30, 34, 1922] {
      check_p10_u8_equivalence(w, ColorMatrix::Bt601, false);
      check_p10_u16_equivalence(w, ColorMatrix::Bt709, true);
    }
  }

  #[test]
  #[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
  fn neon_p10_matches_scalar_1920() {
    check_p10_u8_equivalence(1920, ColorMatrix::Bt709, false);
    check_p10_u16_equivalence(1920, ColorMatrix::Bt2020Ncl, false);
  }

  /// Out‑of‑range regression: every kernel AND‑masks each `u16` load
  /// to the low `BITS` bits, so **arbitrary** upper‑bit corruption
  /// (not just p010 packing) produces scalar/NEON bit‑identical
  /// output. This test sweeps three adversarial input shapes:
  ///
  /// - `p010`: 10 active bits in the high 10 of each `u16`
  ///   (`sample << 6`) — the canonical mispacking mistake.
  /// - `ycgco_worst`: `Y=[0x8000; W]`, `U=[0; W/2]`, `V=[0x8000; W/2]`
  ///   — the specific Codex‑identified case that used to produce
  ///   `(1023, 0, 0)` on scalar vs `(0, 0, 0)` on NEON before the
  ///   load‑time mask was added.
  /// - `random`: arbitrary upper‑bit flips with no particular pattern.
  ///
  /// Each variant runs through every color matrix × range × both
  /// output paths (u8 + native‑depth u16) and asserts byte equality.
  #[test]
  #[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
  fn neon_p10_matches_scalar_on_out_of_range_samples() {
    let width = 32;

    let p010_variant =
      |i: usize, seed: u16| 0xFC00u16.wrapping_add(((i as u16).wrapping_mul(seed)) << 6);
    let random_variant = |i: usize, seed: u16| {
      let x = (i as u32)
        .wrapping_mul(seed as u32)
        .wrapping_add(0xDEAD_BEEF) as u16;
      x ^ 0xA5A5
    };

    for variant_name in ["p010", "ycgco_worst", "random"] {
      let y: std::vec::Vec<u16> = match variant_name {
        "ycgco_worst" => std::vec![0x8000u16; width],
        "p010" => (0..width).map(|i| p010_variant(i, 37)).collect(),
        _ => (0..width).map(|i| random_variant(i, 37)).collect(),
      };
      let u: std::vec::Vec<u16> = match variant_name {
        "ycgco_worst" => std::vec![0x0u16; width / 2],
        "p010" => (0..width / 2).map(|i| p010_variant(i, 53)).collect(),
        _ => (0..width / 2).map(|i| random_variant(i, 53)).collect(),
      };
      let v: std::vec::Vec<u16> = match variant_name {
        "ycgco_worst" => std::vec![0x8000u16; width / 2],
        "p010" => (0..width / 2).map(|i| p010_variant(i, 71)).collect(),
        _ => (0..width / 2).map(|i| random_variant(i, 71)).collect(),
      };

      for matrix in [ColorMatrix::Bt601, ColorMatrix::Bt709, ColorMatrix::YCgCo] {
        for full_range in [true, false] {
          let mut rgb_scalar = std::vec![0u8; width * 3];
          let mut rgb_neon = std::vec![0u8; width * 3];
          scalar::yuv_420p_n_to_rgb_row::<10>(
            &y,
            &u,
            &v,
            &mut rgb_scalar,
            width,
            matrix,
            full_range,
          );
          unsafe {
            yuv_420p_n_to_rgb_row::<10>(&y, &u, &v, &mut rgb_neon, width, matrix, full_range);
          }
          assert_eq!(
            rgb_scalar, rgb_neon,
            "scalar and NEON diverge on {variant_name} input (matrix={matrix:?}, full_range={full_range})"
          );

          let mut rgb16_scalar = std::vec![0u16; width * 3];
          let mut rgb16_neon = std::vec![0u16; width * 3];
          scalar::yuv_420p_n_to_rgb_u16_row::<10>(
            &y,
            &u,
            &v,
            &mut rgb16_scalar,
            width,
            matrix,
            full_range,
          );
          unsafe {
            yuv_420p_n_to_rgb_u16_row::<10>(&y, &u, &v, &mut rgb16_neon, width, matrix, full_range);
          }
          assert_eq!(
            rgb16_scalar, rgb16_neon,
            "scalar and NEON diverge on {variant_name} u16 output (matrix={matrix:?}, full_range={full_range})"
          );
        }
      }
    }
  }

  // ---- P010 NEON scalar-equivalence --------------------------------------

  /// P010 test samples: 10‑bit values shifted into the high 10 bits
  /// (`value << 6`). Deterministic pseudo‑random generator keyed by
  /// index × seed so U, V, Y vectors are mutually distinct.
  fn p010_plane(n: usize, seed: usize) -> std::vec::Vec<u16> {
    (0..n)
      .map(|i| (((i * seed + seed * 3) & 0x3FF) as u16) << 6)
      .collect()
  }

  /// Interleaves per‑pair U, V samples into P010's semi‑planar UV
  /// layout: `[U0, V0, U1, V1, …]`.
  fn p010_uv_interleave(u: &[u16], v: &[u16]) -> std::vec::Vec<u16> {
    let pairs = u.len();
    debug_assert_eq!(u.len(), v.len());
    let mut out = std::vec::Vec::with_capacity(pairs * 2);
    for i in 0..pairs {
      out.push(u[i]);
      out.push(v[i]);
    }
    out
  }

  fn check_p010_u8_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
    let y = p010_plane(width, 37);
    let u_plane = p010_plane(width / 2, 53);
    let v_plane = p010_plane(width / 2, 71);
    let uv = p010_uv_interleave(&u_plane, &v_plane);
    let mut rgb_scalar = std::vec![0u8; width * 3];
    let mut rgb_neon = std::vec![0u8; width * 3];

    scalar::p_n_to_rgb_row::<10>(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
    unsafe {
      p_n_to_rgb_row::<10>(&y, &uv, &mut rgb_neon, width, matrix, full_range);
    }
    if rgb_scalar != rgb_neon {
      let diff = rgb_scalar
        .iter()
        .zip(rgb_neon.iter())
        .position(|(a, b)| a != b)
        .unwrap();
      panic!(
        "NEON P010→u8 diverges at byte {diff} (width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} neon={}",
        rgb_scalar[diff], rgb_neon[diff]
      );
    }
  }

  fn check_p010_u16_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
    let y = p010_plane(width, 37);
    let u_plane = p010_plane(width / 2, 53);
    let v_plane = p010_plane(width / 2, 71);
    let uv = p010_uv_interleave(&u_plane, &v_plane);
    let mut rgb_scalar = std::vec![0u16; width * 3];
    let mut rgb_neon = std::vec![0u16; width * 3];

    scalar::p_n_to_rgb_u16_row::<10>(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
    unsafe {
      p_n_to_rgb_u16_row::<10>(&y, &uv, &mut rgb_neon, width, matrix, full_range);
    }
    if rgb_scalar != rgb_neon {
      let diff = rgb_scalar
        .iter()
        .zip(rgb_neon.iter())
        .position(|(a, b)| a != b)
        .unwrap();
      panic!(
        "NEON P010→u16 diverges at elem {diff} (width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} neon={}",
        rgb_scalar[diff], rgb_neon[diff]
      );
    }
  }

  #[test]
  #[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
  fn neon_p010_u8_matches_scalar_all_matrices_16() {
    for m in [
      ColorMatrix::Bt601,
      ColorMatrix::Bt709,
      ColorMatrix::Bt2020Ncl,
      ColorMatrix::Smpte240m,
      ColorMatrix::Fcc,
      ColorMatrix::YCgCo,
    ] {
      for full in [true, false] {
        check_p010_u8_equivalence(16, m, full);
      }
    }
  }

  #[test]
  #[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
  fn neon_p010_u16_matches_scalar_all_matrices_16() {
    for m in [
      ColorMatrix::Bt601,
      ColorMatrix::Bt709,
      ColorMatrix::Bt2020Ncl,
      ColorMatrix::Smpte240m,
      ColorMatrix::Fcc,
      ColorMatrix::YCgCo,
    ] {
      for full in [true, false] {
        check_p010_u16_equivalence(16, m, full);
      }
    }
  }

  #[test]
  #[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
  fn neon_p010_matches_scalar_odd_tail_widths() {
    for w in [18usize, 30, 34, 1922] {
      check_p010_u8_equivalence(w, ColorMatrix::Bt601, false);
      check_p010_u16_equivalence(w, ColorMatrix::Bt709, true);
    }
  }

  #[test]
  #[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
  fn neon_p010_matches_scalar_1920() {
    check_p010_u8_equivalence(1920, ColorMatrix::Bt709, false);
    check_p010_u16_equivalence(1920, ColorMatrix::Bt2020Ncl, false);
  }

  /// Adversarial regression: mispacked input — `yuv420p10le` values
  /// (10 bits in low 10) accidentally handed to the P010 kernel, or
  /// arbitrary bit corruption — must still produce bit‑identical
  /// output on scalar and NEON. The kernel's `>> 6` load extracts
  /// only the high 10 bits, so any low‑6‑bits data gets deterministically
  /// discarded in both paths.
  #[test]
  #[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
  fn neon_p010_matches_scalar_on_mispacked_input() {
    let width = 32;

    // Three input variants:
    //   - `yuv420p10le_style`: values in low 10 bits (wrong packing
    //     for P010 — `>> 6` drops the actual data, producing near‑black).
    //   - `noise`: arbitrary 16‑bit noise, no particular pattern.
    //   - `every_bit`: each sample has every bit set (0xFFFF).
    for variant in ["yuv420p10le_style", "noise", "every_bit"] {
      let y: std::vec::Vec<u16> = match variant {
        "every_bit" => std::vec![0xFFFFu16; width],
        "yuv420p10le_style" => (0..width).map(|i| ((i * 37 + 11) & 0x3FF) as u16).collect(),
        _ => (0..width)
          .map(|i| ((i as u32 * 53 + 0xDEAD) as u16) ^ 0xA5A5)
          .collect(),
      };
      let uv: std::vec::Vec<u16> = match variant {
        "every_bit" => std::vec![0xFFFFu16; width],
        "yuv420p10le_style" => (0..width).map(|i| ((i * 71 + 23) & 0x3FF) as u16).collect(),
        _ => (0..width)
          .map(|i| ((i as u32 * 91 + 0xBEEF) as u16) ^ 0x5A5A)
          .collect(),
      };

      for matrix in [ColorMatrix::Bt601, ColorMatrix::Bt709, ColorMatrix::YCgCo] {
        for full_range in [true, false] {
          let mut rgb_scalar = std::vec![0u8; width * 3];
          let mut rgb_neon = std::vec![0u8; width * 3];
          scalar::p_n_to_rgb_row::<10>(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
          unsafe {
            p_n_to_rgb_row::<10>(&y, &uv, &mut rgb_neon, width, matrix, full_range);
          }
          assert_eq!(
            rgb_scalar, rgb_neon,
            "scalar and NEON diverge on {variant} P010 input (matrix={matrix:?}, full_range={full_range})"
          );

          let mut rgb16_scalar = std::vec![0u16; width * 3];
          let mut rgb16_neon = std::vec![0u16; width * 3];
          scalar::p_n_to_rgb_u16_row::<10>(&y, &uv, &mut rgb16_scalar, width, matrix, full_range);
          unsafe {
            p_n_to_rgb_u16_row::<10>(&y, &uv, &mut rgb16_neon, width, matrix, full_range);
          }
          assert_eq!(
            rgb16_scalar, rgb16_neon,
            "scalar and NEON diverge on {variant} P010 u16 output (matrix={matrix:?}, full_range={full_range})"
          );
        }
      }
    }
  }

  // ---- Generic BITS equivalence (12/14-bit coverage) ------------------

  fn planar_n_plane<const BITS: u32>(n: usize, seed: usize) -> std::vec::Vec<u16> {
    let mask = (1u32 << BITS) - 1;
    (0..n)
      .map(|i| ((i * seed + seed * 3) as u32 & mask) as u16)
      .collect()
  }

  fn p_n_packed_plane<const BITS: u32>(n: usize, seed: usize) -> std::vec::Vec<u16> {
    let mask = (1u32 << BITS) - 1;
    let shift = 16 - BITS;
    (0..n)
      .map(|i| (((i * seed + seed * 3) as u32 & mask) as u16) << shift)
      .collect()
  }

  fn check_planar_u8_neon_equivalence_n<const BITS: u32>(
    width: usize,
    matrix: ColorMatrix,
    full_range: bool,
  ) {
    let y = planar_n_plane::<BITS>(width, 37);
    let u = planar_n_plane::<BITS>(width / 2, 53);
    let v = planar_n_plane::<BITS>(width / 2, 71);
    let mut rgb_scalar = std::vec![0u8; width * 3];
    let mut rgb_neon = std::vec![0u8; width * 3];
    scalar::yuv_420p_n_to_rgb_row::<BITS>(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
    unsafe {
      yuv_420p_n_to_rgb_row::<BITS>(&y, &u, &v, &mut rgb_neon, width, matrix, full_range);
    }
    assert_eq!(rgb_scalar, rgb_neon, "NEON planar {BITS}-bit → u8 diverges");
  }

  fn check_planar_u16_neon_equivalence_n<const BITS: u32>(
    width: usize,
    matrix: ColorMatrix,
    full_range: bool,
  ) {
    let y = planar_n_plane::<BITS>(width, 37);
    let u = planar_n_plane::<BITS>(width / 2, 53);
    let v = planar_n_plane::<BITS>(width / 2, 71);
    let mut rgb_scalar = std::vec![0u16; width * 3];
    let mut rgb_neon = std::vec![0u16; width * 3];
    scalar::yuv_420p_n_to_rgb_u16_row::<BITS>(
      &y,
      &u,
      &v,
      &mut rgb_scalar,
      width,
      matrix,
      full_range,
    );
    unsafe {
      yuv_420p_n_to_rgb_u16_row::<BITS>(&y, &u, &v, &mut rgb_neon, width, matrix, full_range);
    }
    assert_eq!(
      rgb_scalar, rgb_neon,
      "NEON planar {BITS}-bit → u16 diverges"
    );
  }

  fn check_pn_u8_neon_equivalence_n<const BITS: u32>(
    width: usize,
    matrix: ColorMatrix,
    full_range: bool,
  ) {
    let y = p_n_packed_plane::<BITS>(width, 37);
    let u = p_n_packed_plane::<BITS>(width / 2, 53);
    let v = p_n_packed_plane::<BITS>(width / 2, 71);
    let uv = p010_uv_interleave(&u, &v);
    let mut rgb_scalar = std::vec![0u8; width * 3];
    let mut rgb_neon = std::vec![0u8; width * 3];
    scalar::p_n_to_rgb_row::<BITS>(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
    unsafe {
      p_n_to_rgb_row::<BITS>(&y, &uv, &mut rgb_neon, width, matrix, full_range);
    }
    assert_eq!(rgb_scalar, rgb_neon, "NEON Pn {BITS}-bit → u8 diverges");
  }

  fn check_pn_u16_neon_equivalence_n<const BITS: u32>(
    width: usize,
    matrix: ColorMatrix,
    full_range: bool,
  ) {
    let y = p_n_packed_plane::<BITS>(width, 37);
    let u = p_n_packed_plane::<BITS>(width / 2, 53);
    let v = p_n_packed_plane::<BITS>(width / 2, 71);
    let uv = p010_uv_interleave(&u, &v);
    let mut rgb_scalar = std::vec![0u16; width * 3];
    let mut rgb_neon = std::vec![0u16; width * 3];
    scalar::p_n_to_rgb_u16_row::<BITS>(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
    unsafe {
      p_n_to_rgb_u16_row::<BITS>(&y, &uv, &mut rgb_neon, width, matrix, full_range);
    }
    assert_eq!(rgb_scalar, rgb_neon, "NEON Pn {BITS}-bit → u16 diverges");
  }

  #[test]
  #[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
  fn neon_p12_matches_scalar_all_matrices() {
    for m in [
      ColorMatrix::Bt601,
      ColorMatrix::Bt709,
      ColorMatrix::Bt2020Ncl,
      ColorMatrix::Smpte240m,
      ColorMatrix::Fcc,
      ColorMatrix::YCgCo,
    ] {
      for full in [true, false] {
        check_planar_u8_neon_equivalence_n::<12>(16, m, full);
        check_planar_u16_neon_equivalence_n::<12>(16, m, full);
        check_pn_u8_neon_equivalence_n::<12>(16, m, full);
        check_pn_u16_neon_equivalence_n::<12>(16, m, full);
      }
    }
  }

  #[test]
  #[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
  fn neon_p14_matches_scalar_all_matrices() {
    for m in [
      ColorMatrix::Bt601,
      ColorMatrix::Bt709,
      ColorMatrix::Bt2020Ncl,
      ColorMatrix::Smpte240m,
      ColorMatrix::Fcc,
      ColorMatrix::YCgCo,
    ] {
      for full in [true, false] {
        check_planar_u8_neon_equivalence_n::<14>(16, m, full);
        check_planar_u16_neon_equivalence_n::<14>(16, m, full);
      }
    }
  }

  #[test]
  #[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
  fn neon_p12_matches_scalar_tail_widths() {
    for w in [18usize, 30, 34, 1922] {
      check_planar_u8_neon_equivalence_n::<12>(w, ColorMatrix::Bt601, false);
      check_planar_u16_neon_equivalence_n::<12>(w, ColorMatrix::Bt709, true);
      check_pn_u8_neon_equivalence_n::<12>(w, ColorMatrix::Bt601, false);
      check_pn_u16_neon_equivalence_n::<12>(w, ColorMatrix::Bt2020Ncl, false);
    }
  }

  #[test]
  #[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
  fn neon_p14_matches_scalar_tail_widths() {
    for w in [18usize, 30, 34, 1922] {
      check_planar_u8_neon_equivalence_n::<14>(w, ColorMatrix::Bt601, false);
      check_planar_u16_neon_equivalence_n::<14>(w, ColorMatrix::Bt709, true);
    }
  }
}
