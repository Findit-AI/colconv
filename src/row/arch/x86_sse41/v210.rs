//! SSE4.1 v210 (Tier 4 packed YUV 4:2:2 10-bit) kernels. One v210
//! word = 16 bytes = 6 pixels processed per iteration.
//!
//! Bit extraction uses three shifted-AND ops to pull the three
//! 10-bit fields from each 32-bit lane, then `_mm_shuffle_epi8`
//! permutes the resulting u16 lanes into Y[6], U[3], V[3] vectors.
//! Unlike NEON's `vqtbl3q_u8`, SSE4.1 has no 3-vector table lookup,
//! so each output gathers via three single-source `_mm_shuffle_epi8`
//! calls (one per source `low10`/`mid10`/`high10` vector) plus two
//! `_mm_or_si128` ops to consolidate. The Q15 pipeline that follows
//! mirrors `yuv_planar_high_bit.rs`'s
//! `yuv_420p_n_to_rgb_or_rgba_row<10, _, _>` byte-for-byte — same
//! `chroma_i16x8` / `scale_y` / `q15_shift` / `clamp_u16_max` calls.

use core::arch::x86_64::*;

use super::*;
use crate::{ColorMatrix, row::scalar};

/// Unpacks one 16-byte v210 word into three `__m128i` vectors holding
/// 10-bit samples in their low bits (each lane an i16):
/// - `y_vec`: lanes 0..6 = Y0..Y5 (lanes 6, 7 are don't-care).
/// - `u_vec`: lanes 0..3 = Cb0..Cb2 (lanes 3..7 are don't-care).
/// - `v_vec`: lanes 0..3 = Cr0..Cr2 (lanes 3..7 are don't-care).
///
/// Strategy: load 4 × u32, then three shifted-AND ops yield vectors
/// `low10`, `mid10`, `high10` (one 10-bit field per 32-bit lane).
/// Because each 10-bit value sits in the low 16 bits of its 32-bit
/// lane, reinterpreting the 128-bit register as 16 bytes places valid
/// bytes at `(lane * 4, lane * 4 + 1)`. Three `_mm_shuffle_epi8`
/// permutes (one per source vector) plus two `_mm_or_si128` then
/// gather Y/U/V from the three sources.
///
/// `_mm_shuffle_epi8` writes zero whenever the index byte's high bit
/// is set (here we use `-1` = `0xFF`), so each shuffled vector
/// contributes only at its assigned lanes; the OR merges them.
///
/// # Safety
///
/// Caller must ensure `ptr` has at least 16 bytes readable, and
/// `target_feature` includes SSE4.1 (which implies SSSE3 for
/// `_mm_shuffle_epi8`).
//
// Ship 11a prep: this kernel and its public siblings below are the
// SSE4.1 fallbacks for the v210 dispatcher landing in Task 9. Until
// then they have no callers outside the per-arch tests, so the
// unused-helper warnings would fail CI's `-D warnings`. The
// `dead_code` allow matches the scalar reference at
// `src/row/scalar/v210.rs` and the NEON sibling at
// `src/row/arch/neon/v210.rs`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn unpack_v210_word_sse41(ptr: *const u8) -> (__m128i, __m128i, __m128i) {
  // SAFETY: caller obligation — `ptr` has 16 bytes readable; SSE4.1
  // (and thus SSSE3) is available.
  unsafe {
    let words = _mm_loadu_si128(ptr.cast());
    let mask10 = _mm_set1_epi32(0x3FF);
    let low10 = _mm_and_si128(words, mask10);
    let mid10 = _mm_and_si128(_mm_srli_epi32::<10>(words), mask10);
    let high10 = _mm_and_si128(_mm_srli_epi32::<20>(words), mask10);

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
    // `_mm_shuffle_epi8` writes 0 wherever the index byte has its high
    // bit set (we use `-1` = `0xFF`), so each per-source shuffle below
    // contributes only at its assigned lanes; the OR merges them.

    // ---- Y assembly ------------------------------------------------------
    // From mid10: lanes 0 (Y0) and 3 (Y3).
    let y_idx_mid = _mm_setr_epi8(0, 1, -1, -1, -1, -1, 8, 9, -1, -1, -1, -1, -1, -1, -1, -1);
    // From low10: lanes 1 (Y1) and 4 (Y4).
    let y_idx_low = _mm_setr_epi8(-1, -1, 4, 5, -1, -1, -1, -1, 12, 13, -1, -1, -1, -1, -1, -1);
    // From high10: lanes 2 (Y2) and 5 (Y5).
    let y_idx_high = _mm_setr_epi8(-1, -1, -1, -1, 4, 5, -1, -1, -1, -1, 12, 13, -1, -1, -1, -1);
    let y_from_mid = _mm_shuffle_epi8(mid10, y_idx_mid);
    let y_from_low = _mm_shuffle_epi8(low10, y_idx_low);
    let y_from_high = _mm_shuffle_epi8(high10, y_idx_high);
    let y_vec = _mm_or_si128(_mm_or_si128(y_from_mid, y_from_low), y_from_high);

    // ---- U assembly ------------------------------------------------------
    // From low10: lane 0 (Cb0).
    let u_idx_low = _mm_setr_epi8(0, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    // From mid10: lane 1 (Cb1).
    let u_idx_mid = _mm_setr_epi8(-1, -1, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    // From high10: lane 2 (Cb2).
    let u_idx_high = _mm_setr_epi8(-1, -1, -1, -1, 8, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let u_from_low = _mm_shuffle_epi8(low10, u_idx_low);
    let u_from_mid = _mm_shuffle_epi8(mid10, u_idx_mid);
    let u_from_high = _mm_shuffle_epi8(high10, u_idx_high);
    let u_vec = _mm_or_si128(_mm_or_si128(u_from_low, u_from_mid), u_from_high);

    // ---- V assembly ------------------------------------------------------
    // From high10: lane 0 (Cr0).
    let v_idx_high = _mm_setr_epi8(0, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    // From low10: lane 1 (Cr1).
    let v_idx_low = _mm_setr_epi8(-1, -1, 8, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    // From mid10: lane 2 (Cr2).
    let v_idx_mid = _mm_setr_epi8(
      -1, -1, -1, -1, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    );
    let v_from_high = _mm_shuffle_epi8(high10, v_idx_high);
    let v_from_low = _mm_shuffle_epi8(low10, v_idx_low);
    let v_from_mid = _mm_shuffle_epi8(mid10, v_idx_mid);
    let v_vec = _mm_or_si128(_mm_or_si128(v_from_high, v_from_low), v_from_mid);

    (y_vec, u_vec, v_vec)
  }
}

/// SSE4.1 v210 → packed RGB / RGBA (u8). Const-generic on `ALPHA`:
/// `false` writes 3 bytes per pixel, `true` writes 4 bytes per
/// pixel with `α = 0xFF`. Output bit depth is u8 (downshifted from
/// the native 10-bit Q15 pipeline via `range_params_n::<10, 8>`).
///
/// Byte-identical to `scalar::v210_to_rgb_or_rgba_row::<ALPHA>` for
/// every input.
///
/// # Safety
///
/// 1. **SSE4.1 must be available on the current CPU.**
/// 2. `width % 6 == 0`.
/// 3. `packed.len() >= (width / 6) * 16`.
/// 4. `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
#[allow(dead_code)] // Ship 11a prep — see note on `unpack_v210_word_sse41`.
#[inline]
#[target_feature(enable = "sse4.1")]
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

  // SAFETY: SSE4.1 availability is the caller's obligation; the
  // dispatcher in `crate::row` verifies it. Pointer adds are bounded
  // by the `for w in 0..words` loop and the caller-promised slice
  // lengths checked above.
  unsafe {
    let rnd_v = _mm_set1_epi32(RND);
    let y_off_v = _mm_set1_epi16(y_off as i16);
    let y_scale_v = _mm_set1_epi32(y_scale);
    let c_scale_v = _mm_set1_epi32(c_scale);
    let bias_v = _mm_set1_epi16(bias as i16);
    let cru = _mm_set1_epi32(coeffs.r_u());
    let crv = _mm_set1_epi32(coeffs.r_v());
    let cgu = _mm_set1_epi32(coeffs.g_u());
    let cgv = _mm_set1_epi32(coeffs.g_v());
    let cbu = _mm_set1_epi32(coeffs.b_u());
    let cbv = _mm_set1_epi32(coeffs.b_v());

    for w in 0..words {
      let (y_vec, u_vec, v_vec) = unpack_v210_word_sse41(packed.as_ptr().add(w * 16));

      let y_i16 = y_vec;

      // Subtract chroma bias (512 for 10-bit) — fits i16 since each
      // chroma sample is ≤ 1023.
      let u_i16 = _mm_sub_epi16(u_vec, bias_v);
      let v_i16 = _mm_sub_epi16(v_vec, bias_v);

      // Widen 8-lane i16 chroma to two i32x4 halves so the Q15
      // multiplies don't overflow. Only lanes 0..2 of `_lo` are
      // valid; `_hi` is entirely don't-care. We feed both halves
      // through `chroma_i16x8` to recycle the helper's exact code
      // path; the don't-care output lanes are discarded by the
      // 6-pixel partial store.
      let u_lo_i32 = _mm_cvtepi16_epi32(u_i16);
      let u_hi_i32 = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(u_i16));
      let v_lo_i32 = _mm_cvtepi16_epi32(v_i16);
      let v_hi_i32 = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(v_i16));

      let u_d_lo = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_lo_i32, c_scale_v), rnd_v));
      let u_d_hi = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_hi_i32, c_scale_v), rnd_v));
      let v_d_lo = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_lo_i32, c_scale_v), rnd_v));
      let v_d_hi = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_hi_i32, c_scale_v), rnd_v));

      // 8-lane chroma vectors with valid data in lanes 0..2.
      let r_chroma = chroma_i16x8(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let g_chroma = chroma_i16x8(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let b_chroma = chroma_i16x8(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);

      // Each chroma sample covers 2 Y lanes (4:2:2): duplicate via
      // `_mm_unpacklo_epi16` so lanes 0..6 of `r_dup` align with
      // Y0..Y5. Lane order: [c0, c0, c1, c1, c2, c2, c3, c3].
      let r_dup = _mm_unpacklo_epi16(r_chroma, r_chroma);
      let g_dup = _mm_unpacklo_epi16(g_chroma, g_chroma);
      let b_dup = _mm_unpacklo_epi16(b_chroma, b_chroma);

      // Y scale: `(Y - y_off) * y_scale + RND >> 15` → i16x8.
      let y_scaled = scale_y(y_i16, y_off_v, y_scale_v, rnd_v);

      // u8 narrow with saturation. `_mm_packus_epi16(lo, hi)` emits
      // 16 u8 lanes from 16 i16 lanes; we feed `lo == hi` so the low
      // 8 bytes of the result hold the saturated u8 of the input
      // i16x8. Only the first 6 bytes per channel matter.
      let r_u8 = _mm_packus_epi16(_mm_adds_epi16(y_scaled, r_dup), _mm_setzero_si128());
      let g_u8 = _mm_packus_epi16(_mm_adds_epi16(y_scaled, g_dup), _mm_setzero_si128());
      let b_u8 = _mm_packus_epi16(_mm_adds_epi16(y_scaled, b_dup), _mm_setzero_si128());

      // 6-pixel partial store: SSE4.1 has no 6-lane interleaved store,
      // so write the per-channel 8 u8 lanes into a stack scratch then
      // build the interleaved output via scalar copies for the valid
      // 6-pixel prefix. (Mirrors NEON Task 4's stack-buffer pattern.)
      let mut r_tmp = [0u8; 16];
      let mut g_tmp = [0u8; 16];
      let mut b_tmp = [0u8; 16];
      _mm_storeu_si128(r_tmp.as_mut_ptr().cast(), r_u8);
      _mm_storeu_si128(g_tmp.as_mut_ptr().cast(), g_u8);
      _mm_storeu_si128(b_tmp.as_mut_ptr().cast(), b_u8);

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
  }
}

/// SSE4.1 v210 → packed `u16` RGB / RGBA at native 10-bit depth
/// (low-bit-packed). Byte-identical to
/// `scalar::v210_to_rgb_u16_or_rgba_u16_row::<ALPHA>`.
///
/// # Safety
///
/// 1. **SSE4.1 must be available.**
/// 2. `width % 6 == 0`.
/// 3. `packed.len() >= (width / 6) * 16`.
/// 4. `out.len() >= width * (if ALPHA { 4 } else { 3 })` (`u16` elements).
#[allow(dead_code)] // Ship 11a prep — see note on `unpack_v210_word_sse41`.
#[inline]
#[target_feature(enable = "sse4.1")]
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
    let rnd_v = _mm_set1_epi32(RND);
    let y_off_v = _mm_set1_epi16(y_off as i16);
    let y_scale_v = _mm_set1_epi32(y_scale);
    let c_scale_v = _mm_set1_epi32(c_scale);
    let bias_v = _mm_set1_epi16(bias as i16);
    let max_v = _mm_set1_epi16(out_max);
    let zero_v = _mm_set1_epi16(0);
    let cru = _mm_set1_epi32(coeffs.r_u());
    let crv = _mm_set1_epi32(coeffs.r_v());
    let cgu = _mm_set1_epi32(coeffs.g_u());
    let cgv = _mm_set1_epi32(coeffs.g_v());
    let cbu = _mm_set1_epi32(coeffs.b_u());
    let cbv = _mm_set1_epi32(coeffs.b_v());

    for w in 0..words {
      let (y_vec, u_vec, v_vec) = unpack_v210_word_sse41(packed.as_ptr().add(w * 16));

      let y_i16 = y_vec;
      let u_i16 = _mm_sub_epi16(u_vec, bias_v);
      let v_i16 = _mm_sub_epi16(v_vec, bias_v);

      let u_lo_i32 = _mm_cvtepi16_epi32(u_i16);
      let u_hi_i32 = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(u_i16));
      let v_lo_i32 = _mm_cvtepi16_epi32(v_i16);
      let v_hi_i32 = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(v_i16));

      let u_d_lo = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_lo_i32, c_scale_v), rnd_v));
      let u_d_hi = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_hi_i32, c_scale_v), rnd_v));
      let v_d_lo = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_lo_i32, c_scale_v), rnd_v));
      let v_d_hi = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_hi_i32, c_scale_v), rnd_v));

      let r_chroma = chroma_i16x8(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let g_chroma = chroma_i16x8(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let b_chroma = chroma_i16x8(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);

      let r_dup = _mm_unpacklo_epi16(r_chroma, r_chroma);
      let g_dup = _mm_unpacklo_epi16(g_chroma, g_chroma);
      let b_dup = _mm_unpacklo_epi16(b_chroma, b_chroma);

      let y_scaled = scale_y(y_i16, y_off_v, y_scale_v, rnd_v);

      // Native-depth output: clamp to [0, (1 << 10) - 1]. `_mm_adds_epi16`
      // saturates at i16 bounds (no-op here since |sum| stays well
      // inside i16 for 10-bit), then min/max clamps to 10-bit range.
      let r = clamp_u16_max(_mm_adds_epi16(y_scaled, r_dup), zero_v, max_v);
      let g = clamp_u16_max(_mm_adds_epi16(y_scaled, g_dup), zero_v, max_v);
      let b = clamp_u16_max(_mm_adds_epi16(y_scaled, b_dup), zero_v, max_v);

      // 6-pixel partial u16 store via stack buffer + scalar interleave.
      let mut r_tmp = [0u16; 8];
      let mut g_tmp = [0u16; 8];
      let mut b_tmp = [0u16; 8];
      _mm_storeu_si128(r_tmp.as_mut_ptr().cast(), r);
      _mm_storeu_si128(g_tmp.as_mut_ptr().cast(), g);
      _mm_storeu_si128(b_tmp.as_mut_ptr().cast(), b);

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
  }
}

/// SSE4.1 v210 → 8-bit luma. Y values are downshifted from 10-bit to
/// 8-bit via `>> 2`. Bypasses the YUV → RGB pipeline entirely.
/// Byte-identical to `scalar::v210_to_luma_row`.
///
/// # Safety
///
/// 1. **SSE4.1 must be available.**
/// 2. `width % 6 == 0`.
/// 3. `packed.len() >= (width / 6) * 16`.
/// 4. `luma_out.len() >= width`.
#[allow(dead_code)] // Ship 11a prep — see note on `unpack_v210_word_sse41`.
#[inline]
#[target_feature(enable = "sse4.1")]
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
      let (y_vec, _, _) = unpack_v210_word_sse41(packed.as_ptr().add(w * 16));
      // Downshift 10-bit Y by 2 → 8-bit, narrow to u8x8 via packus.
      let y_shr = _mm_srli_epi16::<2>(y_vec);
      let y_u8 = _mm_packus_epi16(y_shr, _mm_setzero_si128());
      // Store 6 of the 8 lanes: stack buffer + copy_from_slice.
      let mut tmp = [0u8; 16];
      _mm_storeu_si128(tmp.as_mut_ptr().cast(), y_u8);
      luma_out[w * 6..w * 6 + 6].copy_from_slice(&tmp[..6]);
    }
  }
}

/// SSE4.1 v210 → native-depth `u16` luma (low-bit-packed). Each output
/// `u16` carries the source's 10-bit Y value in its low 10 bits.
/// Byte-identical to `scalar::v210_to_luma_u16_row`.
///
/// # Safety
///
/// 1. **SSE4.1 must be available.**
/// 2. `width % 6 == 0`.
/// 3. `packed.len() >= (width / 6) * 16`.
/// 4. `luma_out.len() >= width`.
#[allow(dead_code)] // Ship 11a prep — see note on `unpack_v210_word_sse41`.
#[inline]
#[target_feature(enable = "sse4.1")]
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
      let (y_vec, _, _) = unpack_v210_word_sse41(packed.as_ptr().add(w * 16));
      // Store 6 of the 8 u16 lanes via stack buffer + copy_from_slice.
      let mut tmp = [0u16; 8];
      _mm_storeu_si128(tmp.as_mut_ptr().cast(), y_vec);
      luma_out[w * 6..w * 6 + 6].copy_from_slice(&tmp[..6]);
    }
  }
}
