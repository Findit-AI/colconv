//! NEON v210 (Tier 4 packed YUV 4:2:2 10-bit) kernels. One v210
//! word = 16 bytes = 6 pixels processed per iteration.
//!
//! Bit extraction uses three shifted-AND ops to pull the three
//! 10-bit fields from each 32-bit lane, then `vqtbl3q_u8` permutes
//! the resulting u16 lanes into Y[6], U[3], V[3] vectors. The Q15
//! pipeline that follows mirrors `yuv_planar_high_bit.rs`'s
//! `yuv_420p_n_to_rgb_or_rgba_row<10, _>` byte-for-byte — same
//! `chroma_i16x8` / `scale_y` / `q15_shift` / `clamp_u16_max` calls.

use core::arch::aarch64::*;

use super::*;
use crate::{ColorMatrix, row::scalar};

/// Unpacks one 16-byte v210 word into three u16x8 vectors holding
/// 10-bit samples in their low bits:
/// - `y_vec`: lanes 0..6 = Y0..Y5 (lanes 6, 7 are don't-care).
/// - `u_vec`: lanes 0..3 = Cb0..Cb2 (lanes 3..7 are don't-care).
/// - `v_vec`: lanes 0..3 = Cr0..Cr2 (lanes 3..7 are don't-care).
///
/// Strategy: load 4 × u32, then three shifted-AND ops yield arrays
/// `low10`, `mid10`, `high10` (one 10-bit field per 32-bit lane).
/// Because each 10-bit value sits in the low 16 bits of its 32-bit
/// lane, reinterpreting these as `uint8x16_t` places valid bytes at
/// `(lane * 4, lane * 4 + 1)`. A `vqtbl3q_u8` over the 3-vector
/// composite `{low, mid, high}` then gathers Y/U/V in one permute.
///
/// # Safety
///
/// Caller must ensure `ptr` has at least 16 bytes readable.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn unpack_v210_word_neon(ptr: *const u8) -> (uint16x8_t, uint16x8_t, uint16x8_t) {
  // SAFETY: caller obligation — `ptr` has 16 bytes readable.
  unsafe {
    let words = vld1q_u32(ptr.cast());
    let mask10 = vdupq_n_u32(0x3FF);
    let low10 = vandq_u32(words, mask10);
    let mid10 = vandq_u32(vshrq_n_u32::<10>(words), mask10);
    let high10 = vandq_u32(vshrq_n_u32::<20>(words), mask10);

    // The three 10-bit fields per 32-bit word, in order:
    //   word 0: low=Cb0, mid=Y0, high=Cr0
    //   word 1: low=Y1,  mid=Cb1, high=Y2
    //   word 2: low=Cr1, mid=Y3, high=Cb2
    //   word 3: low=Y4,  mid=Cr2, high=Y5
    //
    // After the AND-mask, each 10-bit sample is in the low 16 bits
    // of its 32-bit lane. Reinterpreted as bytes, the i-th sample's
    // low byte is at byte index `i * 4` and high byte at `i * 4 + 1`;
    // bytes `i * 4 + 2` and `i * 4 + 3` are zero.
    //
    // The 3-vector composite `{low, mid, high}` lays out 48 bytes:
    //   low  bytes 0..15   → indices 0..15
    //   mid  bytes 16..31  → indices 16..31
    //   high bytes 32..47  → indices 32..47
    //
    // Y vector [Y0, Y1, Y2, Y3, Y4, Y5]:
    //   Y0 = mid[w0]  → bytes [16, 17]
    //   Y1 = low[w1]  → bytes [4, 5]
    //   Y2 = high[w1] → bytes [36, 37]
    //   Y3 = mid[w2]  → bytes [24, 25]
    //   Y4 = low[w3]  → bytes [12, 13]
    //   Y5 = high[w3] → bytes [44, 45]
    //
    // U vector [Cb0, Cb1, Cb2]:
    //   Cb0 = low[w0]  → bytes [0, 1]
    //   Cb1 = mid[w1]  → bytes [20, 21]
    //   Cb2 = high[w2] → bytes [40, 41]
    //
    // V vector [Cr0, Cr1, Cr2]:
    //   Cr0 = high[w0] → bytes [32, 33]
    //   Cr1 = low[w2]  → bytes [8, 9]
    //   Cr2 = mid[w3]  → bytes [28, 29]
    //
    // Indices that exceed the 48-byte source range produce zero
    // (per `vqtbl3q_u8`'s spec), so the trailing 0xFF lanes leave
    // the Y/U/V destination's high lanes as don't-care zeros.
    let three: uint8x16x3_t = uint8x16x3_t(
      vreinterpretq_u8_u32(low10),
      vreinterpretq_u8_u32(mid10),
      vreinterpretq_u8_u32(high10),
    );
    let y_idx: uint8x16_t = core::mem::transmute([
      16u8, 17, 4, 5, 36, 37, 24, 25, 12, 13, 44, 45, 0xFF, 0xFF, 0xFF, 0xFF,
    ]);
    let u_idx: uint8x16_t = core::mem::transmute([
      0u8, 1, 20, 21, 40, 41, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    ]);
    let v_idx: uint8x16_t = core::mem::transmute([
      32u8, 33, 8, 9, 28, 29, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    ]);
    let y_vec = vreinterpretq_u16_u8(vqtbl3q_u8(three, y_idx));
    let u_vec = vreinterpretq_u16_u8(vqtbl3q_u8(three, u_idx));
    let v_vec = vreinterpretq_u16_u8(vqtbl3q_u8(three, v_idx));
    (y_vec, u_vec, v_vec)
  }
}

/// NEON v210 → packed RGB / RGBA (u8). Const-generic on `ALPHA`:
/// `false` writes 3 bytes per pixel, `true` writes 4 bytes per
/// pixel with `α = 0xFF`. Output bit depth is u8 (downshifted from
/// the native 10-bit Q15 pipeline via `range_params_n::<10, 8>`).
///
/// Byte-identical to `scalar::v210_to_rgb_or_rgba_row::<ALPHA>` for
/// every input.
///
/// # Safety
///
/// 1. **NEON must be available on the current CPU.**
/// 2. `width % 6 == 0`.
/// 3. `packed.len() >= (width / 6) * 16`.
/// 4. `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn v210_to_rgb_or_rgba_row<const ALPHA: bool>(
  packed: &[u8],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(
    width.is_multiple_of(6),
    "v210 requires width divisible by 6"
  );
  let words = width / 6;
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(packed.len() >= words * 16);
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<10, 8>(full_range);
  let bias = scalar::chroma_bias::<10>();
  const RND: i32 = 1 << 14;

  // SAFETY: NEON availability is the caller's obligation; the
  // dispatcher in `crate::row` verifies it. Pointer adds are bounded
  // by the `for w in 0..words` loop and the caller-promised slice
  // lengths checked above.
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

    for w in 0..words {
      let (y_vec, u_vec, v_vec) = unpack_v210_word_neon(packed.as_ptr().add(w * 16));

      let y_i16 = vreinterpretq_s16_u16(y_vec);

      // Subtract chroma bias (512 for 10-bit) — fits i16 since each
      // chroma sample is ≤ 1023.
      let u_i16 = vsubq_s16(vreinterpretq_s16_u16(u_vec), bias_v);
      let v_i16 = vsubq_s16(vreinterpretq_s16_u16(v_vec), bias_v);

      // Widen 8-lane i16 chroma to two i32x4 halves so the Q15
      // multiplies don't overflow. Only lanes 0..2 of `_lo` are
      // valid; `_hi` is entirely don't-care. We feed both halves
      // through `chroma_i16x8` to recycle the helper's exact code
      // path; the don't-care output lanes are discarded by the
      // 6-pixel partial store.
      let u_lo_i32 = vmovl_s16(vget_low_s16(u_i16));
      let u_hi_i32 = vmovl_s16(vget_high_s16(u_i16));
      let v_lo_i32 = vmovl_s16(vget_low_s16(v_i16));
      let v_hi_i32 = vmovl_s16(vget_high_s16(v_i16));

      let u_d_lo = q15_shift(vaddq_s32(vmulq_s32(u_lo_i32, c_scale_v), rnd_v));
      let u_d_hi = q15_shift(vaddq_s32(vmulq_s32(u_hi_i32, c_scale_v), rnd_v));
      let v_d_lo = q15_shift(vaddq_s32(vmulq_s32(v_lo_i32, c_scale_v), rnd_v));
      let v_d_hi = q15_shift(vaddq_s32(vmulq_s32(v_hi_i32, c_scale_v), rnd_v));

      // 8-lane chroma vectors with valid data in lanes 0..2.
      let r_chroma = chroma_i16x8(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let g_chroma = chroma_i16x8(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let b_chroma = chroma_i16x8(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);

      // Each chroma sample covers 2 Y lanes (4:2:2): duplicate via
      // `vzip1q_s16` so lanes 0..6 of `r_dup` align with Y0..Y5.
      // Lane order after vzip1: [c0, c0, c1, c1, c2, c2, c3, c3].
      let r_dup = vzip1q_s16(r_chroma, r_chroma);
      let g_dup = vzip1q_s16(g_chroma, g_chroma);
      let b_dup = vzip1q_s16(b_chroma, b_chroma);

      // Y scale: `(Y - y_off) * y_scale + RND >> 15` → i16x8.
      let y_scaled = scale_y(y_i16, y_off_v, y_scale_v, rnd_v);

      // u8 narrow with saturation. Only the low 8 lanes matter; we
      // store 6 of them per channel.
      let r_u8 = vqmovun_s16(vqaddq_s16(y_scaled, r_dup));
      let g_u8 = vqmovun_s16(vqaddq_s16(y_scaled, g_dup));
      let b_u8 = vqmovun_s16(vqaddq_s16(y_scaled, b_dup));

      // 6-pixel partial store: write into a stack buffer via
      // `vst3_u8` / `vst4_u8` (8-pixel stores) then memcpy the
      // valid prefix. NEON has no 6-lane interleaved store.
      if ALPHA {
        let alpha = vdup_n_u8(0xFF);
        let mut tmp = [0u8; 8 * 4];
        vst4_u8(tmp.as_mut_ptr(), uint8x8x4_t(r_u8, g_u8, b_u8, alpha));
        out[w * 6 * 4..w * 6 * 4 + 6 * 4].copy_from_slice(&tmp[..6 * 4]);
      } else {
        let mut tmp = [0u8; 8 * 3];
        vst3_u8(tmp.as_mut_ptr(), uint8x8x3_t(r_u8, g_u8, b_u8));
        out[w * 6 * 3..w * 6 * 3 + 6 * 3].copy_from_slice(&tmp[..6 * 3]);
      }
    }
  }
}

/// NEON v210 → packed `u16` RGB / RGBA at native 10-bit depth
/// (low-bit-packed). Byte-identical to
/// `scalar::v210_to_rgb_u16_or_rgba_u16_row::<ALPHA>`.
///
/// # Safety
///
/// 1. **NEON must be available.**
/// 2. `width % 6 == 0`.
/// 3. `packed.len() >= (width / 6) * 16`.
/// 4. `out.len() >= width * (if ALPHA { 4 } else { 3 })` (`u16` elements).
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn v210_to_rgb_u16_or_rgba_u16_row<const ALPHA: bool>(
  packed: &[u8],
  out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(
    width.is_multiple_of(6),
    "v210 requires width divisible by 6"
  );
  let words = width / 6;
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(packed.len() >= words * 16);
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<10, 10>(full_range);
  let bias = scalar::chroma_bias::<10>();
  const RND: i32 = 1 << 14;
  let out_max: i16 = ((1i32 << 10) - 1) as i16;

  // SAFETY: caller's obligation per the safety contract above.
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

    for w in 0..words {
      let (y_vec, u_vec, v_vec) = unpack_v210_word_neon(packed.as_ptr().add(w * 16));

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

      // Native-depth output: clamp to [0, (1 << 10) - 1]. `vqaddq_s16`
      // saturates at i16 bounds (no-op here since |sum| stays well
      // inside i16 for 10-bit), then max/min clamps to 10-bit range.
      let r = clamp_u16_max(vqaddq_s16(y_scaled, r_dup), zero_v, max_v);
      let g = clamp_u16_max(vqaddq_s16(y_scaled, g_dup), zero_v, max_v);
      let b = clamp_u16_max(vqaddq_s16(y_scaled, b_dup), zero_v, max_v);

      // 6-pixel partial u16 store via stack buffer + copy_from_slice.
      // No NEON 6-lane interleaved u16 store exists.
      if ALPHA {
        let alpha = vdupq_n_u16(out_max as u16);
        let mut tmp = [0u16; 8 * 4];
        vst4q_u16(tmp.as_mut_ptr(), uint16x8x4_t(r, g, b, alpha));
        out[w * 6 * 4..w * 6 * 4 + 6 * 4].copy_from_slice(&tmp[..6 * 4]);
      } else {
        let mut tmp = [0u16; 8 * 3];
        vst3q_u16(tmp.as_mut_ptr(), uint16x8x3_t(r, g, b));
        out[w * 6 * 3..w * 6 * 3 + 6 * 3].copy_from_slice(&tmp[..6 * 3]);
      }
    }
  }
}

/// NEON v210 → 8-bit luma. Y values are downshifted from 10-bit to
/// 8-bit via `>> 2`. Bypasses the YUV → RGB pipeline entirely.
/// Byte-identical to `scalar::v210_to_luma_row`.
///
/// # Safety
///
/// 1. **NEON must be available.**
/// 2. `width % 6 == 0`.
/// 3. `packed.len() >= (width / 6) * 16`.
/// 4. `luma_out.len() >= width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn v210_to_luma_row(packed: &[u8], luma_out: &mut [u8], width: usize) {
  debug_assert!(
    width.is_multiple_of(6),
    "v210 requires width divisible by 6"
  );
  let words = width / 6;
  debug_assert!(packed.len() >= words * 16);
  debug_assert!(luma_out.len() >= width);

  // SAFETY: caller's obligation per the safety contract above.
  unsafe {
    for w in 0..words {
      let (y_vec, _, _) = unpack_v210_word_neon(packed.as_ptr().add(w * 16));
      // Downshift 10-bit Y by 2 → 8-bit, narrow to u8x8.
      let y_u8 = vqmovn_u16(vshrq_n_u16::<2>(y_vec));
      // Store 6 of the 8 lanes: stack buffer + copy_from_slice.
      let mut tmp = [0u8; 8];
      vst1_u8(tmp.as_mut_ptr(), y_u8);
      luma_out[w * 6..w * 6 + 6].copy_from_slice(&tmp[..6]);
    }
  }
}

/// NEON v210 → native-depth `u16` luma (low-bit-packed). Each output
/// `u16` carries the source's 10-bit Y value in its low 10 bits.
/// Byte-identical to `scalar::v210_to_luma_u16_row`.
///
/// # Safety
///
/// 1. **NEON must be available.**
/// 2. `width % 6 == 0`.
/// 3. `packed.len() >= (width / 6) * 16`.
/// 4. `luma_out.len() >= width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn v210_to_luma_u16_row(packed: &[u8], luma_out: &mut [u16], width: usize) {
  debug_assert!(
    width.is_multiple_of(6),
    "v210 requires width divisible by 6"
  );
  let words = width / 6;
  debug_assert!(packed.len() >= words * 16);
  debug_assert!(luma_out.len() >= width);

  // SAFETY: caller's obligation per the safety contract above.
  unsafe {
    for w in 0..words {
      let (y_vec, _, _) = unpack_v210_word_neon(packed.as_ptr().add(w * 16));
      // Store 6 of the 8 u16 lanes via stack buffer + copy_from_slice.
      let mut tmp = [0u16; 8];
      vst1q_u16(tmp.as_mut_ptr(), y_vec);
      luma_out[w * 6..w * 6 + 6].copy_from_slice(&tmp[..6]);
    }
  }
}
