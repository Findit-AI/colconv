//! wasm-simd128 v210 (Tier 4 packed YUV 4:2:2 10-bit) kernels. One v210
//! word = 16 bytes = 6 pixels processed per iteration — same block size
//! as NEON / SSE4.1.
//!
//! Bit extraction uses three shifted-AND ops to pull the three 10-bit
//! fields from each 32-bit lane, then `u8x16_swizzle` permutes the
//! resulting u16 lanes into Y[6], U[3], V[3] vectors. wasm-simd128's
//! `u8x16_swizzle` is the single-source variant matching SSSE3
//! `_mm_shuffle_epi8` semantics — out-of-range index bytes (high bit
//! set, i.e. `-1` = `0xFF`) produce zero, so the same triple-permute-
//! and-OR strategy as SSE4.1 applies. The Q15 pipeline that follows
//! mirrors `yuv_planar_high_bit.rs`'s
//! `yuv_420p_n_to_rgb_or_rgba_row<10, _, _>` byte-for-byte — same
//! `chroma_i16x8` / `scale_y` / `q15_shift` / `clamp_u16_max_wasm`
//! calls.

use core::arch::wasm32::*;

use super::*;
use crate::{ColorMatrix, row::scalar};

/// Unpacks one 16-byte v210 word into three `v128` vectors holding
/// 10-bit samples in their low bits (each lane an i16):
/// - `y_vec`: lanes 0..6 = Y0..Y5 (lanes 6, 7 are don't-care).
/// - `u_vec`: lanes 0..3 = Cb0..Cb2 (lanes 3..7 are don't-care).
/// - `v_vec`: lanes 0..3 = Cr0..Cr2 (lanes 3..7 are don't-care).
///
/// Strategy: load 4 × u32, then three shifted-AND ops yield vectors
/// `low10`, `mid10`, `high10` (one 10-bit field per 32-bit lane).
/// Because each 10-bit value sits in the low 16 bits of its 32-bit
/// lane, reinterpreting the 128-bit register as 16 bytes places valid
/// bytes at `(lane * 4, lane * 4 + 1)`. Three `u8x16_swizzle`
/// permutes (one per source vector) plus two `v128_or` ops then
/// gather Y/U/V from the three sources.
///
/// `u8x16_swizzle` writes zero whenever the index byte's high bit is
/// set (here we use `-1` = `0xFF`), so each shuffled vector
/// contributes only at its assigned lanes; the OR merges them — same
/// semantic as SSSE3 `_mm_shuffle_epi8`.
///
/// # Safety
///
/// Caller must ensure `ptr` has at least 16 bytes readable, and
/// `target_feature` includes `simd128` (verified at compile time on
/// wasm).
#[inline]
#[target_feature(enable = "simd128")]
unsafe fn unpack_v210_word_wasm(ptr: *const u8) -> (v128, v128, v128) {
  // SAFETY: caller obligation — `ptr` has 16 bytes readable; simd128
  // is enabled at compile time.
  unsafe {
    let words = v128_load(ptr.cast());
    let mask10 = i32x4_splat(0x3FF);
    let low10 = v128_and(words, mask10);
    let mid10 = v128_and(u32x4_shr(words, 10), mask10);
    let high10 = v128_and(u32x4_shr(words, 20), mask10);

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
    // Y vector [Y0, Y1, Y2, Y3, Y4, Y5]:
    //   Y0 = mid[w0]  → bytes 0,1 of mid10  → result lane 0 (bytes 0,1)
    //   Y1 = low[w1]  → bytes 4,5 of low10  → result lane 1 (bytes 2,3)
    //   Y2 = high[w1] → bytes 4,5 of high10 → result lane 2 (bytes 4,5)
    //   Y3 = mid[w2]  → bytes 8,9 of mid10  → result lane 3 (bytes 6,7)
    //   Y4 = low[w3]  → bytes 12,13 of low10 → result lane 4 (bytes 8,9)
    //   Y5 = high[w3] → bytes 12,13 of high10 → result lane 5 (bytes 10,11)
    //
    // U vector [Cb0, Cb1, Cb2]:
    //   Cb0 = low[w0]  → bytes 0,1 of low10  → result lane 0 (bytes 0,1)
    //   Cb1 = mid[w1]  → bytes 4,5 of mid10  → result lane 1 (bytes 2,3)
    //   Cb2 = high[w2] → bytes 8,9 of high10 → result lane 2 (bytes 4,5)
    //
    // V vector [Cr0, Cr1, Cr2]:
    //   Cr0 = high[w0] → bytes 0,1 of high10 → result lane 0 (bytes 0,1)
    //   Cr1 = low[w2]  → bytes 8,9 of low10  → result lane 1 (bytes 2,3)
    //   Cr2 = mid[w3]  → bytes 12,13 of mid10 → result lane 2 (bytes 4,5)
    //
    // `u8x16_swizzle` writes 0 wherever the index byte has its high
    // bit set (we use `-1` = `0xFF`), so each per-source shuffle below
    // contributes only at its assigned lanes; the OR merges them.

    // ---- Y assembly ------------------------------------------------------
    // From mid10: lanes 0 (Y0) and 3 (Y3).
    let y_idx_mid = i8x16(0, 1, -1, -1, -1, -1, 8, 9, -1, -1, -1, -1, -1, -1, -1, -1);
    // From low10: lanes 1 (Y1) and 4 (Y4).
    let y_idx_low = i8x16(-1, -1, 4, 5, -1, -1, -1, -1, 12, 13, -1, -1, -1, -1, -1, -1);
    // From high10: lanes 2 (Y2) and 5 (Y5).
    let y_idx_high = i8x16(-1, -1, -1, -1, 4, 5, -1, -1, -1, -1, 12, 13, -1, -1, -1, -1);
    let y_from_mid = u8x16_swizzle(mid10, y_idx_mid);
    let y_from_low = u8x16_swizzle(low10, y_idx_low);
    let y_from_high = u8x16_swizzle(high10, y_idx_high);
    let y_vec = v128_or(v128_or(y_from_mid, y_from_low), y_from_high);

    // ---- U assembly ------------------------------------------------------
    // From low10: lane 0 (Cb0).
    let u_idx_low = i8x16(0, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    // From mid10: lane 1 (Cb1).
    let u_idx_mid = i8x16(-1, -1, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    // From high10: lane 2 (Cb2).
    let u_idx_high = i8x16(-1, -1, -1, -1, 8, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let u_from_low = u8x16_swizzle(low10, u_idx_low);
    let u_from_mid = u8x16_swizzle(mid10, u_idx_mid);
    let u_from_high = u8x16_swizzle(high10, u_idx_high);
    let u_vec = v128_or(v128_or(u_from_low, u_from_mid), u_from_high);

    // ---- V assembly ------------------------------------------------------
    // From high10: lane 0 (Cr0).
    let v_idx_high = i8x16(0, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    // From low10: lane 1 (Cr1).
    let v_idx_low = i8x16(-1, -1, 8, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    // From mid10: lane 2 (Cr2).
    let v_idx_mid = i8x16(
      -1, -1, -1, -1, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    );
    let v_from_high = u8x16_swizzle(high10, v_idx_high);
    let v_from_low = u8x16_swizzle(low10, v_idx_low);
    let v_from_mid = u8x16_swizzle(mid10, v_idx_mid);
    let v_vec = v128_or(v128_or(v_from_high, v_from_low), v_from_mid);

    (y_vec, u_vec, v_vec)
  }
}

/// wasm-simd128 v210 → packed RGB / RGBA (u8). Const-generic on `ALPHA`:
/// `false` writes 3 bytes per pixel, `true` writes 4 bytes per pixel
/// with `α = 0xFF`. Output bit depth is u8 (downshifted from the
/// native 10-bit Q15 pipeline via `range_params_n::<10, 8>`).
///
/// Byte-identical to `scalar::v210_to_rgb_or_rgba_row::<ALPHA>` for
/// every input.
///
/// # Safety
///
/// 1. **`simd128` must be enabled at compile time.**
/// 2. `width % 2 == 0` (4:2:2 chroma pair).
/// 3. `packed.len() >= ceil(width / 6) * 16`.
/// 4. `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn v210_to_rgb_or_rgba_row<const ALPHA: bool>(
  packed: &[u8],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(width.is_multiple_of(2), "v210 requires even width");
  let total_words = width.div_ceil(6);
  let words = width / 6;
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(packed.len() >= total_words * 16);
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<10, 8>(full_range);
  let bias = scalar::chroma_bias::<10>();
  const RND: i32 = 1 << 14;

  // SAFETY: simd128 compile-time availability is the caller's
  // obligation; the dispatcher in `crate::row` verifies it. Pointer
  // adds are bounded by the `for w in 0..words` loop and the
  // caller-promised slice lengths checked above.
  unsafe {
    let rnd_v = i32x4_splat(RND);
    let y_off_v = i16x8_splat(y_off as i16);
    let y_scale_v = i32x4_splat(y_scale);
    let c_scale_v = i32x4_splat(c_scale);
    let bias_v = i16x8_splat(bias as i16);
    let cru = i32x4_splat(coeffs.r_u());
    let crv = i32x4_splat(coeffs.r_v());
    let cgu = i32x4_splat(coeffs.g_u());
    let cgv = i32x4_splat(coeffs.g_v());
    let cbu = i32x4_splat(coeffs.b_u());
    let cbv = i32x4_splat(coeffs.b_v());

    for w in 0..words {
      let (y_vec, u_vec, v_vec) = unpack_v210_word_wasm(packed.as_ptr().add(w * 16));

      let y_i16 = y_vec;

      // Subtract chroma bias (512 for 10-bit) — fits i16 since each
      // chroma sample is ≤ 1023.
      let u_i16 = i16x8_sub(u_vec, bias_v);
      let v_i16 = i16x8_sub(v_vec, bias_v);

      // Widen 8-lane i16 chroma to two i32x4 halves so the Q15
      // multiplies don't overflow. Only lanes 0..2 of `_lo` are
      // valid; `_hi` is entirely don't-care. We feed both halves
      // through `chroma_i16x8` to recycle the helper's exact code
      // path; the don't-care output lanes are discarded by the
      // 6-pixel partial store.
      let u_lo_i32 = i32x4_extend_low_i16x8(u_i16);
      let u_hi_i32 = i32x4_extend_high_i16x8(u_i16);
      let v_lo_i32 = i32x4_extend_low_i16x8(v_i16);
      let v_hi_i32 = i32x4_extend_high_i16x8(v_i16);

      let u_d_lo = q15_shift(i32x4_add(i32x4_mul(u_lo_i32, c_scale_v), rnd_v));
      let u_d_hi = q15_shift(i32x4_add(i32x4_mul(u_hi_i32, c_scale_v), rnd_v));
      let v_d_lo = q15_shift(i32x4_add(i32x4_mul(v_lo_i32, c_scale_v), rnd_v));
      let v_d_hi = q15_shift(i32x4_add(i32x4_mul(v_hi_i32, c_scale_v), rnd_v));

      // 8-lane chroma vectors with valid data in lanes 0..2.
      let r_chroma = chroma_i16x8(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let g_chroma = chroma_i16x8(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let b_chroma = chroma_i16x8(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);

      // Each chroma sample covers 2 Y lanes (4:2:2): duplicate the
      // low 4 lanes via [`dup_lo`] so lanes 0..6 of `r_dup` align with
      // Y0..Y5. Lane order: [c0, c0, c1, c1, c2, c2, c3, c3].
      let r_dup = dup_lo(r_chroma);
      let g_dup = dup_lo(g_chroma);
      let b_dup = dup_lo(b_chroma);

      // Y scale: `(Y - y_off) * y_scale + RND >> 15` → i16x8.
      let y_scaled = scale_y(y_i16, y_off_v, y_scale_v, rnd_v);

      // u8 narrow with saturation. `u8x16_narrow_i16x8(lo, hi)` emits
      // 16 u8 lanes from 16 i16 lanes; we feed `lo == hi` so the low
      // 8 bytes of the result hold the saturated u8 of the input
      // i16x8. Only the first 6 bytes per channel matter.
      let r_sum = i16x8_add_sat(y_scaled, r_dup);
      let g_sum = i16x8_add_sat(y_scaled, g_dup);
      let b_sum = i16x8_add_sat(y_scaled, b_dup);
      let r_u8 = u8x16_narrow_i16x8(r_sum, r_sum);
      let g_u8 = u8x16_narrow_i16x8(g_sum, g_sum);
      let b_u8 = u8x16_narrow_i16x8(b_sum, b_sum);

      // 6-pixel partial store: wasm-simd128 has no 6-lane interleaved
      // store, so write the per-channel 16 u8 lanes into stack
      // scratch then build the interleaved output via scalar copies
      // for the valid 6-pixel prefix. (Mirrors NEON Task 4 / SSE4.1
      // Task 5 stack-buffer pattern.)
      let mut r_tmp = [0u8; 16];
      let mut g_tmp = [0u8; 16];
      let mut b_tmp = [0u8; 16];
      v128_store(r_tmp.as_mut_ptr().cast(), r_u8);
      v128_store(g_tmp.as_mut_ptr().cast(), g_u8);
      v128_store(b_tmp.as_mut_ptr().cast(), b_u8);

      if ALPHA {
        let dst = &mut out[w * 6 * 4..w * 6 * 4 + 6 * 4];
        for i in 0..6 {
          dst[i * 4] = r_tmp[i];
          dst[i * 4 + 1] = g_tmp[i];
          dst[i * 4 + 2] = b_tmp[i];
          dst[i * 4 + 3] = 0xFF;
        }
      } else {
        let dst = &mut out[w * 6 * 3..w * 6 * 3 + 6 * 3];
        for i in 0..6 {
          dst[i * 3] = r_tmp[i];
          dst[i * 3 + 1] = g_tmp[i];
          dst[i * 3 + 2] = b_tmp[i];
        }
      }
    }

    // Partial-word tail (2 or 4 px) goes through scalar.
    if words * 6 < width {
      let tail_start_px = words * 6;
      let tail_packed = &packed[words * 16..total_words * 16];
      let tail_out = &mut out[tail_start_px * bpp..width * bpp];
      let tail_w = width - tail_start_px;
      scalar::v210_to_rgb_or_rgba_row::<ALPHA>(tail_packed, tail_out, tail_w, matrix, full_range);
    }
  }
}

/// wasm-simd128 v210 → packed `u16` RGB / RGBA at native 10-bit depth
/// (low-bit-packed). Byte-identical to
/// `scalar::v210_to_rgb_u16_or_rgba_u16_row::<ALPHA>`.
///
/// # Safety
///
/// 1. **`simd128` must be enabled at compile time.**
/// 2. `width % 2 == 0` (4:2:2 chroma pair).
/// 3. `packed.len() >= ceil(width / 6) * 16`.
/// 4. `out.len() >= width * (if ALPHA { 4 } else { 3 })` (`u16` elements).
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn v210_to_rgb_u16_or_rgba_u16_row<const ALPHA: bool>(
  packed: &[u8],
  out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(width.is_multiple_of(2), "v210 requires even width");
  let total_words = width.div_ceil(6);
  let words = width / 6;
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(packed.len() >= total_words * 16);
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<10, 10>(full_range);
  let bias = scalar::chroma_bias::<10>();
  const RND: i32 = 1 << 14;
  let out_max: i16 = ((1i32 << 10) - 1) as i16;

  // SAFETY: caller's obligation per the safety contract above.
  unsafe {
    let rnd_v = i32x4_splat(RND);
    let y_off_v = i16x8_splat(y_off as i16);
    let y_scale_v = i32x4_splat(y_scale);
    let c_scale_v = i32x4_splat(c_scale);
    let bias_v = i16x8_splat(bias as i16);
    let max_v = i16x8_splat(out_max);
    let zero_v = i16x8_splat(0);
    let cru = i32x4_splat(coeffs.r_u());
    let crv = i32x4_splat(coeffs.r_v());
    let cgu = i32x4_splat(coeffs.g_u());
    let cgv = i32x4_splat(coeffs.g_v());
    let cbu = i32x4_splat(coeffs.b_u());
    let cbv = i32x4_splat(coeffs.b_v());

    for w in 0..words {
      let (y_vec, u_vec, v_vec) = unpack_v210_word_wasm(packed.as_ptr().add(w * 16));

      let y_i16 = y_vec;
      let u_i16 = i16x8_sub(u_vec, bias_v);
      let v_i16 = i16x8_sub(v_vec, bias_v);

      let u_lo_i32 = i32x4_extend_low_i16x8(u_i16);
      let u_hi_i32 = i32x4_extend_high_i16x8(u_i16);
      let v_lo_i32 = i32x4_extend_low_i16x8(v_i16);
      let v_hi_i32 = i32x4_extend_high_i16x8(v_i16);

      let u_d_lo = q15_shift(i32x4_add(i32x4_mul(u_lo_i32, c_scale_v), rnd_v));
      let u_d_hi = q15_shift(i32x4_add(i32x4_mul(u_hi_i32, c_scale_v), rnd_v));
      let v_d_lo = q15_shift(i32x4_add(i32x4_mul(v_lo_i32, c_scale_v), rnd_v));
      let v_d_hi = q15_shift(i32x4_add(i32x4_mul(v_hi_i32, c_scale_v), rnd_v));

      let r_chroma = chroma_i16x8(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let g_chroma = chroma_i16x8(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let b_chroma = chroma_i16x8(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);

      let r_dup = dup_lo(r_chroma);
      let g_dup = dup_lo(g_chroma);
      let b_dup = dup_lo(b_chroma);

      let y_scaled = scale_y(y_i16, y_off_v, y_scale_v, rnd_v);

      // Native-depth output: clamp to `[0, (1 << 10) - 1]`.
      // `i16x8_add_sat` saturates at i16 bounds (no-op here since
      // |sum| stays well inside i16 for 10-bit), then min/max clamps
      // to 10-bit range.
      let r = clamp_u16_max_wasm(i16x8_add_sat(y_scaled, r_dup), zero_v, max_v);
      let g = clamp_u16_max_wasm(i16x8_add_sat(y_scaled, g_dup), zero_v, max_v);
      let b = clamp_u16_max_wasm(i16x8_add_sat(y_scaled, b_dup), zero_v, max_v);

      // 6-pixel partial u16 store via stack buffer + scalar interleave.
      let mut r_tmp = [0u16; 8];
      let mut g_tmp = [0u16; 8];
      let mut b_tmp = [0u16; 8];
      v128_store(r_tmp.as_mut_ptr().cast(), r);
      v128_store(g_tmp.as_mut_ptr().cast(), g);
      v128_store(b_tmp.as_mut_ptr().cast(), b);

      if ALPHA {
        let dst = &mut out[w * 6 * 4..w * 6 * 4 + 6 * 4];
        let alpha = out_max as u16;
        for i in 0..6 {
          dst[i * 4] = r_tmp[i];
          dst[i * 4 + 1] = g_tmp[i];
          dst[i * 4 + 2] = b_tmp[i];
          dst[i * 4 + 3] = alpha;
        }
      } else {
        let dst = &mut out[w * 6 * 3..w * 6 * 3 + 6 * 3];
        for i in 0..6 {
          dst[i * 3] = r_tmp[i];
          dst[i * 3 + 1] = g_tmp[i];
          dst[i * 3 + 2] = b_tmp[i];
        }
      }
    }

    // Partial-word tail (2 or 4 px) goes through scalar.
    if words * 6 < width {
      let tail_start_px = words * 6;
      let tail_packed = &packed[words * 16..total_words * 16];
      let tail_out = &mut out[tail_start_px * bpp..width * bpp];
      let tail_w = width - tail_start_px;
      scalar::v210_to_rgb_u16_or_rgba_u16_row::<ALPHA>(
        tail_packed,
        tail_out,
        tail_w,
        matrix,
        full_range,
      );
    }
  }
}

/// wasm-simd128 v210 → 8-bit luma. Y values are downshifted from
/// 10-bit to 8-bit via `>> 2`. Bypasses the YUV → RGB pipeline
/// entirely. Byte-identical to `scalar::v210_to_luma_row`.
///
/// # Safety
///
/// 1. **`simd128` must be enabled at compile time.**
/// 2. `width % 2 == 0` (4:2:2 chroma pair).
/// 3. `packed.len() >= ceil(width / 6) * 16`.
/// 4. `luma_out.len() >= width`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn v210_to_luma_row(packed: &[u8], luma_out: &mut [u8], width: usize) {
  debug_assert!(width.is_multiple_of(2), "v210 requires even width");
  let total_words = width.div_ceil(6);
  let words = width / 6;
  debug_assert!(packed.len() >= total_words * 16);
  debug_assert!(luma_out.len() >= width);

  // SAFETY: caller's obligation per the safety contract above.
  unsafe {
    for w in 0..words {
      let (y_vec, _, _) = unpack_v210_word_wasm(packed.as_ptr().add(w * 16));
      // Downshift 10-bit Y by 2 → 8-bit, narrow to u8x16 via
      // saturating narrow (Y ≤ 1023 stays well inside [0, 255] post-shift).
      let y_shr = u16x8_shr(y_vec, 2);
      let y_u8 = u8x16_narrow_i16x8(y_shr, y_shr);
      // Store 6 of the 16 u8 lanes: stack buffer + copy_from_slice.
      let mut tmp = [0u8; 16];
      v128_store(tmp.as_mut_ptr().cast(), y_u8);
      luma_out[w * 6..w * 6 + 6].copy_from_slice(&tmp[..6]);
    }
    if words * 6 < width {
      let tail_start_px = words * 6;
      let tail_packed = &packed[words * 16..total_words * 16];
      let tail_out = &mut luma_out[tail_start_px..width];
      let tail_w = width - tail_start_px;
      scalar::v210_to_luma_row(tail_packed, tail_out, tail_w);
    }
  }
}

/// wasm-simd128 v210 → native-depth `u16` luma (low-bit-packed). Each
/// output `u16` carries the source's 10-bit Y value in its low 10
/// bits. Byte-identical to `scalar::v210_to_luma_u16_row`.
///
/// # Safety
///
/// 1. **`simd128` must be enabled at compile time.**
/// 2. `width % 2 == 0` (4:2:2 chroma pair).
/// 3. `packed.len() >= ceil(width / 6) * 16`.
/// 4. `luma_out.len() >= width`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn v210_to_luma_u16_row(packed: &[u8], luma_out: &mut [u16], width: usize) {
  debug_assert!(width.is_multiple_of(2), "v210 requires even width");
  let total_words = width.div_ceil(6);
  let words = width / 6;
  debug_assert!(packed.len() >= total_words * 16);
  debug_assert!(luma_out.len() >= width);

  // SAFETY: caller's obligation per the safety contract above.
  unsafe {
    for w in 0..words {
      let (y_vec, _, _) = unpack_v210_word_wasm(packed.as_ptr().add(w * 16));
      // Store 6 of the 8 u16 lanes via stack buffer + copy_from_slice.
      let mut tmp = [0u16; 8];
      v128_store(tmp.as_mut_ptr().cast(), y_vec);
      luma_out[w * 6..w * 6 + 6].copy_from_slice(&tmp[..6]);
    }
    if words * 6 < width {
      let tail_start_px = words * 6;
      let tail_packed = &packed[words * 16..total_words * 16];
      let tail_out = &mut luma_out[tail_start_px..width];
      let tail_w = width - tail_start_px;
      scalar::v210_to_luma_u16_row(tail_packed, tail_out, tail_w);
    }
  }
}
