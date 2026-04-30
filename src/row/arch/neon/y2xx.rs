//! NEON Y2xx (Tier 4 packed YUV 4:2:2 high‑bit‑depth) kernels for
//! `BITS ∈ {10, 12}`. One iteration processes **8 pixels** = 16 u16
//! samples = 32 bytes via `vld2q_u16` deinterleave.
//!
//! Layout per row: u16 quadruples `(Y₀, U, Y₁, V)` with the active
//! `BITS` bits sitting in the **high** bits of each `u16` (low
//! `(16 - BITS)` bits are zero, MSB‑aligned). Right‑shifting by
//! `(16 - BITS)` brings the active samples into `[0, 2^BITS - 1]`.
//!
//! ## Per‑iter pipeline (8 px / 16 u16 / 32 bytes)
//!
//! `vld2q_u16` reads 16 interleaved u16s and returns:
//!   - `pair.0` = even u16 lanes = `[Y0, Y1, Y2, Y3, Y4, Y5, Y6, Y7]`
//!     (every quadruple's Y samples sit at quadruple-positions 0 and 2,
//!     i.e. even u16 indices in the row).
//!   - `pair.1` = odd u16 lanes = `[U0, V0, U1, V1, U2, V2, U3, V3]`
//!     (chroma samples at quadruple-positions 1 and 3 = odd u16
//!     indices in the row).
//!
//! `vuzp1q_u16(chroma, chroma)` then puts U0..U3 in the low 4 lanes
//! and `vuzp2q_u16(chroma, chroma)` puts V0..V3 in the low 4 lanes —
//! the high 4 lanes are duplicates we discard via the chroma‑duplicate
//! step that follows the Q15 chroma compute.
//!
//! From there the kernel mirrors `yuv_planar_high_bit.rs::
//! yuv_420p_n_to_rgb_or_rgba_row<BITS, _>` byte‑for‑byte: subtract
//! chroma bias, Q15‑scale chroma to `u_d`/`v_d`, compute
//! `chroma_i16x8` for r/g/b, scale Y, sum + saturate / clamp, write.
//!
//! ## Tail
//!
//! Pixels less than the next 8‑px multiple fall through to scalar.

use core::arch::aarch64::*;

use super::*;
use crate::{ColorMatrix, row::scalar};

/// Loads 8 Y2xx pixels (16 u16 samples = 32 bytes) and unpacks them
/// into three 8‑lane vectors:
/// - `y_vec`: lanes 0..8 = Y0..Y7 in `[0, 2^BITS - 1]`.
/// - `u_vec`: lanes 0..4 = U0..U3 in `[0, 2^BITS - 1]` (lanes 4..7
///   are duplicates of lanes 0..3, treated as don't-care).
/// - `v_vec`: lanes 0..4 = V0..V3 in `[0, 2^BITS - 1]` (lanes 4..7
///   are duplicates of lanes 0..3, treated as don't-care).
///
/// Strategy: `vld2q_u16` deinterleaves even / odd u16 lanes; the
/// even‑lane half is Y, the odd‑lane half is chroma in `[U, V]`
/// pairs. A pair of `vuzp1q` / `vuzp2q` then separates U from V.
/// Each result is right‑shifted dynamically by `-(16 - BITS)` via
/// `vshlq_u16` with a negative count (matching the existing
/// `subsampled_high_bit_pn_4_2_0.rs` and `yuv_planar_high_bit.rs`
/// alpha‑shift pattern — `vshrq_n_u16` requires a literal const
/// shift, but `16 - BITS` is not a stable const generic expression
/// on stable Rust).
///
/// # Safety
///
/// Caller must ensure `ptr` has at least 32 bytes (16 u16) readable.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn unpack_y2xx_8px_neon(
  ptr: *const u16,
  shr_count: int16x8_t,
) -> (uint16x8_t, uint16x8_t, uint16x8_t) {
  // SAFETY: caller obligation — `ptr` has 16 u16 readable.
  unsafe {
    let pair = vld2q_u16(ptr);
    // `vshlq_u16` performs a logical right shift when the count is
    // negative; `shr_count = -(16 - BITS)`. For BITS=10 → shift by 6.
    let y_vec = vshlq_u16(pair.0, shr_count);
    let chroma = vshlq_u16(pair.1, shr_count);
    // `chroma` lanes are `[U0, V0, U1, V1, U2, V2, U3, V3]`.
    //   vuzp1q_u16(c, c) = even lanes of c, then even lanes of c
    //                    = [U0, U1, U2, U3, U0, U1, U2, U3]
    //   vuzp2q_u16(c, c) = odd  lanes of c, then odd  lanes of c
    //                    = [V0, V1, V2, V3, V0, V1, V2, V3]
    // Only lanes 0..4 of u_vec / v_vec carry valid data.
    let u_vec = vuzp1q_u16(chroma, chroma);
    let v_vec = vuzp2q_u16(chroma, chroma);
    (y_vec, u_vec, v_vec)
  }
}

/// NEON Y2xx → packed RGB / RGBA u8. Const‑generic over
/// `BITS ∈ {10, 12}` and `ALPHA ∈ {false, true}`. Output bit depth is
/// u8 (downshifted from the native BITS Q15 pipeline via
/// `range_params_n::<BITS, 8>`).
///
/// Byte‑identical to `scalar::y2xx_n_to_rgb_or_rgba_row::<BITS, ALPHA>`
/// for every input.
///
/// # Safety
///
/// 1. **NEON must be available on the current CPU.**
/// 2. `width % 2 == 0` (4:2:2 chroma pair).
/// 3. `packed.len() >= width * 2`.
/// 4. `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn y2xx_n_to_rgb_or_rgba_row<const BITS: u32, const ALPHA: bool>(
  packed: &[u16],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  const {
    assert!(
      BITS == 10 || BITS == 12,
      "y2xx_n_to_rgb_or_rgba_row requires BITS in {{10, 12}}"
    );
  }
  debug_assert!(width.is_multiple_of(2), "Y2xx requires even width");
  debug_assert!(packed.len() >= width * 2);
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<BITS, 8>(full_range);
  let bias = scalar::chroma_bias::<BITS>();
  const RND: i32 = 1 << 14;

  // SAFETY: NEON availability is the caller's obligation; the
  // dispatcher in `crate::row` verifies it. Pointer adds are bounded
  // by the `while x + 8 <= width` loop and the caller-promised slice
  // lengths checked above.
  unsafe {
    let rnd_v = vdupq_n_s32(RND);
    let y_off_v = vdupq_n_s16(y_off as i16);
    let y_scale_v = vdupq_n_s32(y_scale);
    let c_scale_v = vdupq_n_s32(c_scale);
    let bias_v = vdupq_n_s16(bias as i16);
    let shr_count = vdupq_n_s16(-((16 - BITS) as i16));
    let cru = vdupq_n_s32(coeffs.r_u());
    let crv = vdupq_n_s32(coeffs.r_v());
    let cgu = vdupq_n_s32(coeffs.g_u());
    let cgv = vdupq_n_s32(coeffs.g_v());
    let cbu = vdupq_n_s32(coeffs.b_u());
    let cbv = vdupq_n_s32(coeffs.b_v());

    let mut x = 0usize;
    while x + 8 <= width {
      let (y_vec, u_vec, v_vec) = unpack_y2xx_8px_neon(packed.as_ptr().add(x * 2), shr_count);

      let y_i16 = vreinterpretq_s16_u16(y_vec);

      // Subtract chroma bias (e.g. 512 for 10‑bit) — fits i16 since
      // each chroma sample is ≤ 2^BITS - 1 ≤ 4095.
      let u_i16 = vsubq_s16(vreinterpretq_s16_u16(u_vec), bias_v);
      let v_i16 = vsubq_s16(vreinterpretq_s16_u16(v_vec), bias_v);

      // Widen 8‑lane i16 chroma to two i32x4 halves for the Q15
      // multiplies. Only lanes 0..3 of `_lo` are valid; `_hi` is
      // entirely don't-care (duplicate of `_lo`). We feed both
      // halves through `chroma_i16x8` to recycle the helper exactly;
      // the don't-care output lanes are discarded by `vzip1q_s16`
      // below (which only consumes lanes 0..3).
      let u_lo_i32 = vmovl_s16(vget_low_s16(u_i16));
      let u_hi_i32 = vmovl_s16(vget_high_s16(u_i16));
      let v_lo_i32 = vmovl_s16(vget_low_s16(v_i16));
      let v_hi_i32 = vmovl_s16(vget_high_s16(v_i16));

      let u_d_lo = q15_shift(vaddq_s32(vmulq_s32(u_lo_i32, c_scale_v), rnd_v));
      let u_d_hi = q15_shift(vaddq_s32(vmulq_s32(u_hi_i32, c_scale_v), rnd_v));
      let v_d_lo = q15_shift(vaddq_s32(vmulq_s32(v_lo_i32, c_scale_v), rnd_v));
      let v_d_hi = q15_shift(vaddq_s32(vmulq_s32(v_hi_i32, c_scale_v), rnd_v));

      // 8‑lane chroma vectors with valid data in lanes 0..3.
      let r_chroma = chroma_i16x8(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let g_chroma = chroma_i16x8(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let b_chroma = chroma_i16x8(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);

      // Each chroma sample covers 2 Y lanes (4:2:2): duplicate via
      // `vzip1q_s16` so lanes 0..7 of `r_dup` align with Y0..Y7.
      // `vzip1q_s16` interleaves the low 4 lanes of each operand:
      //   [c0, c0, c1, c1, c2, c2, c3, c3]
      let r_dup = vzip1q_s16(r_chroma, r_chroma);
      let g_dup = vzip1q_s16(g_chroma, g_chroma);
      let b_dup = vzip1q_s16(b_chroma, b_chroma);

      // Y scale: `(Y - y_off) * y_scale + RND >> 15` → i16x8.
      let y_scaled = scale_y(y_i16, y_off_v, y_scale_v, rnd_v);

      // u8 narrow with saturation. 8 valid lanes per channel.
      let r_u8 = vqmovun_s16(vqaddq_s16(y_scaled, r_dup));
      let g_u8 = vqmovun_s16(vqaddq_s16(y_scaled, g_dup));
      let b_u8 = vqmovun_s16(vqaddq_s16(y_scaled, b_dup));

      if ALPHA {
        let alpha = vdup_n_u8(0xFF);
        vst4_u8(
          out.as_mut_ptr().add(x * 4),
          uint8x8x4_t(r_u8, g_u8, b_u8, alpha),
        );
      } else {
        vst3_u8(out.as_mut_ptr().add(x * 3), uint8x8x3_t(r_u8, g_u8, b_u8));
      }

      x += 8;
    }

    // Scalar tail — remaining < 8 pixels (always even per 4:2:2).
    if x < width {
      let tail_packed = &packed[x * 2..width * 2];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      scalar::y2xx_n_to_rgb_or_rgba_row::<BITS, ALPHA>(
        tail_packed,
        tail_out,
        tail_w,
        matrix,
        full_range,
      );
    }
  }
}

/// NEON Y2xx → packed `u16` RGB / RGBA at native BITS depth
/// (low‑bit‑packed: BITS active bits in the low N of each `u16`).
/// Const‑generic over `BITS ∈ {10, 12}`.
///
/// Byte‑identical to
/// `scalar::y2xx_n_to_rgb_u16_or_rgba_u16_row::<BITS, ALPHA>`.
///
/// # Safety
///
/// 1. **NEON must be available.**
/// 2. `width % 2 == 0` (4:2:2 chroma pair).
/// 3. `packed.len() >= width * 2`.
/// 4. `out.len() >= width * (if ALPHA { 4 } else { 3 })` (`u16` elements).
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn y2xx_n_to_rgb_u16_or_rgba_u16_row<const BITS: u32, const ALPHA: bool>(
  packed: &[u16],
  out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  const {
    assert!(
      BITS == 10 || BITS == 12,
      "y2xx_n_to_rgb_u16_or_rgba_u16_row requires BITS in {{10, 12}}"
    );
  }
  debug_assert!(width.is_multiple_of(2), "Y2xx requires even width");
  debug_assert!(packed.len() >= width * 2);
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<BITS, BITS>(full_range);
  let bias = scalar::chroma_bias::<BITS>();
  const RND: i32 = 1 << 14;
  let out_max: i16 = ((1i32 << BITS) - 1) as i16;

  // SAFETY: caller's obligation per the safety contract above.
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
    while x + 8 <= width {
      let (y_vec, u_vec, v_vec) = unpack_y2xx_8px_neon(packed.as_ptr().add(x * 2), shr_count);

      let y_i16 = vreinterpretq_s16_u16(y_vec);
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

      let r_dup = vzip1q_s16(r_chroma, r_chroma);
      let g_dup = vzip1q_s16(g_chroma, g_chroma);
      let b_dup = vzip1q_s16(b_chroma, b_chroma);

      let y_scaled = scale_y(y_i16, y_off_v, y_scale_v, rnd_v);

      // Native‑depth output: clamp to [0, (1 << BITS) - 1]. `vqaddq_s16`
      // saturates at i16 bounds (no‑op here since |sum| stays well
      // inside i16 for BITS ≤ 12), then max/min clamps to the BITS range.
      let r = clamp_u16_max(vqaddq_s16(y_scaled, r_dup), zero_v, max_v);
      let g = clamp_u16_max(vqaddq_s16(y_scaled, g_dup), zero_v, max_v);
      let b = clamp_u16_max(vqaddq_s16(y_scaled, b_dup), zero_v, max_v);

      if ALPHA {
        let alpha = vdupq_n_u16(out_max as u16);
        vst4q_u16(out.as_mut_ptr().add(x * 4), uint16x8x4_t(r, g, b, alpha));
      } else {
        vst3q_u16(out.as_mut_ptr().add(x * 3), uint16x8x3_t(r, g, b));
      }

      x += 8;
    }

    if x < width {
      let tail_packed = &packed[x * 2..width * 2];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      scalar::y2xx_n_to_rgb_u16_or_rgba_u16_row::<BITS, ALPHA>(
        tail_packed,
        tail_out,
        tail_w,
        matrix,
        full_range,
      );
    }
  }
}

/// NEON Y2xx → 8‑bit luma. Y values are downshifted from BITS to 8
/// via `>> (BITS - 8)` after the `>> (16 - BITS)` MSB‑alignment, i.e.
/// a single `>> 8` from the raw u16 sample. Bypasses the YUV → RGB
/// pipeline entirely.
///
/// Byte‑identical to `scalar::y2xx_n_to_luma_row::<BITS>`.
///
/// # Safety
///
/// 1. **NEON must be available.**
/// 2. `width % 2 == 0` (4:2:2 chroma pair).
/// 3. `packed.len() >= width * 2`.
/// 4. `luma_out.len() >= width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn y2xx_n_to_luma_row<const BITS: u32>(
  packed: &[u16],
  luma_out: &mut [u8],
  width: usize,
) {
  const {
    assert!(
      BITS == 10 || BITS == 12,
      "y2xx_n_to_luma_row requires BITS in {{10, 12}}"
    );
  }
  debug_assert!(width.is_multiple_of(2), "Y2xx requires even width");
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(luma_out.len() >= width);

  // SAFETY: caller's obligation per the safety contract above.
  unsafe {
    let mut x = 0usize;
    while x + 8 <= width {
      // `vld2q_u16` deinterleaves; `pair.0` is 8 raw Y u16 samples
      // (still MSB‑aligned at BITS ≤ 12, low bits zero).
      let pair = vld2q_u16(packed.as_ptr().add(x * 2));
      // `>> (16 - BITS)` then `>> (BITS - 8)` collapses to `>> 8`
      // for any BITS ∈ {10, 12} — the constant fold gives the same
      // result whether we shift in two stages or one.
      let y_u8 = vshrn_n_u16::<8>(pair.0);
      vst1_u8(luma_out.as_mut_ptr().add(x), y_u8);
      x += 8;
    }
    if x < width {
      let tail_packed = &packed[x * 2..width * 2];
      let tail_out = &mut luma_out[x..width];
      let tail_w = width - x;
      scalar::y2xx_n_to_luma_row::<BITS>(tail_packed, tail_out, tail_w);
    }
  }
}

/// NEON Y2xx → native‑depth `u16` luma (low‑bit‑packed). Each output
/// `u16` carries the source's BITS-bit Y value in its low BITS bits.
/// Byte‑identical to `scalar::y2xx_n_to_luma_u16_row::<BITS>`.
///
/// # Safety
///
/// 1. **NEON must be available.**
/// 2. `width % 2 == 0` (4:2:2 chroma pair).
/// 3. `packed.len() >= width * 2`.
/// 4. `luma_out.len() >= width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn y2xx_n_to_luma_u16_row<const BITS: u32>(
  packed: &[u16],
  luma_out: &mut [u16],
  width: usize,
) {
  const {
    assert!(
      BITS == 10 || BITS == 12,
      "y2xx_n_to_luma_u16_row requires BITS in {{10, 12}}"
    );
  }
  debug_assert!(width.is_multiple_of(2), "Y2xx requires even width");
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(luma_out.len() >= width);

  // SAFETY: caller's obligation per the safety contract above.
  unsafe {
    let shr_count = vdupq_n_s16(-((16 - BITS) as i16));
    let mut x = 0usize;
    while x + 8 <= width {
      let pair = vld2q_u16(packed.as_ptr().add(x * 2));
      // Right‑shift by `(16 - BITS)` to bring MSB‑aligned samples
      // into low‑bit‑packed form for the native‑depth u16 output.
      let y_low = vshlq_u16(pair.0, shr_count);
      vst1q_u16(luma_out.as_mut_ptr().add(x), y_low);
      x += 8;
    }
    if x < width {
      let tail_packed = &packed[x * 2..width * 2];
      let tail_out = &mut luma_out[x..width];
      let tail_w = width - x;
      scalar::y2xx_n_to_luma_u16_row::<BITS>(tail_packed, tail_out, tail_w);
    }
  }
}
