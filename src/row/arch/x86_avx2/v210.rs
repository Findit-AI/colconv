//! AVX2 v210 (Tier 4 packed YUV 4:2:2 10-bit) kernels. Two v210
//! words = 32 bytes = 12 pixels processed per iteration.
//!
//! Bit extraction uses three shifted-AND ops at 256-bit width to
//! pull the three 10-bit fields from each 32-bit lane, then
//! `_mm256_shuffle_epi8` permutes the resulting u16 lanes into
//! per-128-bit-lane Y[6] / U[3] / V[3] slabs (each 128-bit lane
//! independently extracts its own word's samples). A cross-lane
//! consolidate then packs the two slabs into a single 12-Y / 6-U /
//! 6-V vector usable by the AVX2 Q15 helpers (`chroma_i16x16`,
//! `chroma_dup`, `scale_y`, `narrow_u8x32`, `clamp_u16_max_x16`).
//!
//! - For Y: a single `_mm256_permutevar8x32_epi32` gathers Y[0..12]
//!   into 32-bit lanes 0..6 of an `__m256i`. AVX2 has no
//!   single-instruction u16 cross-lane permute, but Y[i] sits in the
//!   low 16 bits of a 32-bit lane after the per-lane shuffle, so the
//!   32-bit gather works cleanly.
//! - For U / V: 3 valid u16 per lane = 6 bytes does not fit a clean
//!   32-bit-lane gather (the third u16 spans i32-lane boundaries).
//!   We instead arrange the per-lane shuffle so lane 0 holds U[0..3]
//!   in bytes 0-5 and lane 1 holds U[3..6] in bytes 6-11 (zero
//!   elsewhere), then OR the two 128-bit lanes together to land
//!   U[0..6] contiguously in the low 12 bytes of an `__m128i`. That
//!   `__m128i` is then lifted to an `__m256i` (lower half = data,
//!   upper half = don't-care) so the AVX2 chroma helpers can consume
//!   it. Only 6 of the 16 chroma lanes carry valid data; the
//!   remaining lanes are don't-care because the 12-pixel partial
//!   store discards them.
//!
//! The Q15 pipeline that follows mirrors `yuv_planar_high_bit.rs`'s
//! `yuv_420p_n_to_rgb_or_rgba_row<10, _, _>` byte-for-byte — same
//! `chroma_i16x16` / `chroma_dup` / `scale_y` / `q15_shift` /
//! `clamp_u16_max_x16` calls.

use core::arch::x86_64::*;

use super::*;
use crate::{ColorMatrix, row::scalar};

/// Unpacks two consecutive 16-byte v210 words (= 12 pixels) into
/// three vectors holding 10-bit samples in their low bits:
/// - `y_vec`: i16x16 with lanes 0..12 = Y0..Y11 (lanes 12..16 are
///   don't-care).
/// - `u_vec`: i16x16 with lanes 0..6 = Cb0..Cb5 (rest don't-care).
/// - `v_vec`: i16x16 with lanes 0..6 = Cr0..Cr5 (rest don't-care).
///
/// Strategy: load 32 bytes (= 8 × u32 = 2 v210 words), then three
/// shifted-AND ops yield vectors `low10`, `mid10`, `high10` (one
/// 10-bit field per 32-bit lane). `_mm256_shuffle_epi8` operates
/// per 128-bit lane, so each lane independently extracts its own 6
/// Y / 3 U / 3 V values into stable byte positions; a single
/// cross-lane permute (Y) or extract-and-OR (U / V) then
/// consolidates the two 128-bit lanes' results.
///
/// `_mm256_shuffle_epi8` writes zero whenever the index byte's high
/// bit is set (here we use `-1` = `0xFF`), so each shuffled vector
/// contributes only at its assigned lanes; the OR merges them per
/// 128-bit lane (same as SSE4.1).
///
/// # Safety
///
/// Caller must ensure `ptr` has at least 32 bytes readable, and
/// `target_feature` includes AVX2 (which implies AVX, SSSE3, etc.).
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn unpack_v210_2words_avx2(ptr: *const u8) -> (__m256i, __m256i, __m256i) {
  // SAFETY: caller obligation — `ptr` has 32 bytes readable; AVX2
  // (and thus SSSE3) is available.
  unsafe {
    let words = _mm256_loadu_si256(ptr.cast());
    let mask10 = _mm256_set1_epi32(0x3FF);
    let low10 = _mm256_and_si256(words, mask10);
    let mid10 = _mm256_and_si256(_mm256_srli_epi32::<10>(words), mask10);
    let high10 = _mm256_and_si256(_mm256_srli_epi32::<20>(words), mask10);

    // Per 128-bit lane, the three 10-bit fields per 32-bit word, in
    // order (each lane = one v210 word):
    //   word 0: low=Cb0, mid=Y0, high=Cr0
    //   word 1: low=Y1,  mid=Cb1, high=Y2
    //   word 2: low=Cr1, mid=Y3, high=Cb2
    //   word 3: low=Y4,  mid=Cr2, high=Y5
    //
    // After the AND-mask, each 10-bit sample is in the low 16 bits
    // of its 32-bit lane. Reinterpreted as bytes (within a 128-bit
    // lane), the i-th sample's low byte is at byte index `i * 4`
    // and high byte at `i * 4 + 1`; bytes `i * 4 + 2` and `i * 4 + 3`
    // are zero.
    //
    // ---- Y per-lane shuffle (lanes 0..6 of i16) -----------------------
    // Same pattern as SSE4.1, replicated to both 128-bit lanes:
    //   lane 0 (Y from word 0..3) → lane outputs bytes 0..11 = Y0..Y5
    //   lane 1 (Y from word 4..7) → lane outputs bytes 0..11 = Y6..Y11
    let y_idx_mid = _mm256_setr_epi8(
      0, 1, -1, -1, -1, -1, 8, 9, -1, -1, -1, -1, -1, -1, -1, -1, // low lane
      0, 1, -1, -1, -1, -1, 8, 9, -1, -1, -1, -1, -1, -1, -1, -1, // high lane
    );
    let y_idx_low = _mm256_setr_epi8(
      -1, -1, 4, 5, -1, -1, -1, -1, 12, 13, -1, -1, -1, -1, -1, -1, // low lane
      -1, -1, 4, 5, -1, -1, -1, -1, 12, 13, -1, -1, -1, -1, -1, -1, // high lane
    );
    let y_idx_high = _mm256_setr_epi8(
      -1, -1, -1, -1, 4, 5, -1, -1, -1, -1, 12, 13, -1, -1, -1, -1, // low lane
      -1, -1, -1, -1, 4, 5, -1, -1, -1, -1, 12, 13, -1, -1, -1, -1, // high lane
    );
    let y_from_mid = _mm256_shuffle_epi8(mid10, y_idx_mid);
    let y_from_low = _mm256_shuffle_epi8(low10, y_idx_low);
    let y_from_high = _mm256_shuffle_epi8(high10, y_idx_high);
    // Per-lane Y vector: lane 0 holds Y0..Y5 in bytes 0-11; lane 1
    // holds Y6..Y11 in bytes 0-11. Bytes 12-15 of each lane are zero.
    let y_per_lane = _mm256_or_si256(_mm256_or_si256(y_from_mid, y_from_low), y_from_high);

    // Cross-lane gather: pack i32 lanes [0, 1, 2] (= Y0..Y5) of low
    // 128 with i32 lanes [4, 5, 6] (= Y6..Y11) of high 128 into i32
    // lanes 0..6 of the result. Lanes 6, 7 are filled with index 7
    // (any zero-or-don't-care lane) and consumed as garbage by the
    // 12-pixel partial store.
    let y_gather_idx = _mm256_setr_epi32(0, 1, 2, 4, 5, 6, 7, 7);
    let y_vec = _mm256_permutevar8x32_epi32(y_per_lane, y_gather_idx);

    // ---- U per-lane shuffle ------------------------------------------
    // Within each 128-bit lane, place the three valid u16 samples in
    // a *different* byte range per lane so that ORing the two halves
    // yields contiguous U[0..6]:
    //   lane 0: U0,U1,U2 at bytes 0-5 (rest zero)
    //   lane 1: U3,U4,U5 at bytes 6-11 (rest zero)
    //
    // The SSE4.1 mapping for U was:
    //   U0 = low[w0]  → bytes 0,1 of low10  → lane output 0,1
    //   U1 = mid[w1]  → bytes 4,5 of mid10  → lane output 2,3
    //   U2 = high[w2] → bytes 8,9 of high10 → lane output 4,5
    // For lane 0 we keep that mapping. For lane 1 we shift the same
    // outputs by 6 bytes (so they land at bytes 6,7 / 8,9 / 10,11
    // within lane 1):
    //   U3 = low[w0_in_lane1]  → bytes 0,1 of low10[lane1]   → lane output 6,7
    //   U4 = mid[w1_in_lane1]  → bytes 4,5 of mid10[lane1]   → lane output 8,9
    //   U5 = high[w2_in_lane1] → bytes 8,9 of high10[lane1]  → lane output 10,11
    let u_idx_low = _mm256_setr_epi8(
      // lane 0: U0 from low10[w0] → output bytes 0,1
      0, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
      // lane 1: U3 from low10[w0_lane1] → output bytes 6,7
      -1, -1, -1, -1, -1, -1, 0, 1, -1, -1, -1, -1, -1, -1, -1, -1,
    );
    let u_idx_mid = _mm256_setr_epi8(
      // lane 0: U1 from mid10[w1] → output bytes 2,3
      -1, -1, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
      // lane 1: U4 from mid10[w1_lane1] → output bytes 8,9
      -1, -1, -1, -1, -1, -1, -1, -1, 4, 5, -1, -1, -1, -1, -1, -1,
    );
    let u_idx_high = _mm256_setr_epi8(
      // lane 0: U2 from high10[w2] → output bytes 4,5
      -1, -1, -1, -1, 8, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
      // lane 1: U5 from high10[w2_lane1] → output bytes 10,11
      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8, 9, -1, -1, -1, -1,
    );
    let u_from_low = _mm256_shuffle_epi8(low10, u_idx_low);
    let u_from_mid = _mm256_shuffle_epi8(mid10, u_idx_mid);
    let u_from_high = _mm256_shuffle_epi8(high10, u_idx_high);
    // Per-lane U vector: lane 0 has U0..U2 in bytes 0-5; lane 1 has
    // U3..U5 in bytes 6-11. Byte ranges do not overlap between
    // lanes, so ORing the two 128-bit halves (next step) gives
    // contiguous U[0..6] at bytes 0-11.
    let u_per_lane = _mm256_or_si256(_mm256_or_si256(u_from_low, u_from_mid), u_from_high);

    // Cross-lane consolidate: extract each 128-bit half and OR. The
    // result lives in the low 128 bits; the high 128 bits are filled
    // with don't-care (the AVX2 chroma helpers consume only the low
    // 6 lanes for our 12-pixel block).
    let u_lo128 = _mm256_castsi256_si128(u_per_lane);
    let u_hi128 = _mm256_extracti128_si256::<1>(u_per_lane);
    let u_low_combined = _mm_or_si128(u_lo128, u_hi128);
    let u_vec = _mm256_castsi128_si256(u_low_combined);

    // ---- V per-lane shuffle ------------------------------------------
    // V mapping per SSE4.1:
    //   V0 = high[w0] → bytes 0,1 of high10  → lane output 0,1
    //   V1 = low[w2]  → bytes 8,9 of low10   → lane output 2,3
    //   V2 = mid[w3]  → bytes 12,13 of mid10 → lane output 4,5
    // Lane 1 shifts each by 6 bytes (output bytes 6,7 / 8,9 / 10,11).
    let v_idx_high = _mm256_setr_epi8(
      // lane 0: V0 from high10[w0] → output bytes 0,1
      0, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
      // lane 1: V3 from high10[w0_lane1] → output bytes 6,7
      -1, -1, -1, -1, -1, -1, 0, 1, -1, -1, -1, -1, -1, -1, -1, -1,
    );
    let v_idx_low = _mm256_setr_epi8(
      // lane 0: V1 from low10[w2] → output bytes 2,3
      -1, -1, 8, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
      // lane 1: V4 from low10[w2_lane1] → output bytes 8,9
      -1, -1, -1, -1, -1, -1, -1, -1, 8, 9, -1, -1, -1, -1, -1, -1,
    );
    let v_idx_mid = _mm256_setr_epi8(
      // lane 0: V2 from mid10[w3] → output bytes 4,5
      -1, -1, -1, -1, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
      // lane 1: V5 from mid10[w3_lane1] → output bytes 10,11
      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, 13, -1, -1, -1, -1,
    );
    let v_from_high = _mm256_shuffle_epi8(high10, v_idx_high);
    let v_from_low = _mm256_shuffle_epi8(low10, v_idx_low);
    let v_from_mid = _mm256_shuffle_epi8(mid10, v_idx_mid);
    let v_per_lane = _mm256_or_si256(_mm256_or_si256(v_from_high, v_from_low), v_from_mid);

    let v_lo128 = _mm256_castsi256_si128(v_per_lane);
    let v_hi128 = _mm256_extracti128_si256::<1>(v_per_lane);
    let v_low_combined = _mm_or_si128(v_lo128, v_hi128);
    let v_vec = _mm256_castsi128_si256(v_low_combined);

    (y_vec, u_vec, v_vec)
  }
}

/// AVX2 v210 → packed RGB / RGBA (u8). Const-generic on `ALPHA`:
/// `false` writes 3 bytes per pixel, `true` writes 4 bytes per
/// pixel with `α = 0xFF`. Output bit depth is u8 (downshifted from
/// the native 10-bit Q15 pipeline via `range_params_n::<10, 8>`).
///
/// Byte-identical to `scalar::v210_to_rgb_or_rgba_row::<ALPHA>` for
/// every input.
///
/// # Safety
///
/// 1. **AVX2 must be available on the current CPU.**
/// 2. `width % 6 == 0`.
/// 3. `packed.len() >= (width / 6) * 16`.
/// 4. `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
#[inline]
#[target_feature(enable = "avx2")]
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

  // SAFETY: AVX2 availability is the caller's obligation; the
  // dispatcher in `crate::row` verifies it. Pointer adds are bounded
  // by the `for w in 0..pairs` / tail loop and the caller-promised
  // slice lengths checked above.
  unsafe {
    let rnd_v = _mm256_set1_epi32(RND);
    let y_off_v = _mm256_set1_epi16(y_off as i16);
    let y_scale_v = _mm256_set1_epi32(y_scale);
    let c_scale_v = _mm256_set1_epi32(c_scale);
    let bias_v = _mm256_set1_epi16(bias as i16);
    let cru = _mm256_set1_epi32(coeffs.r_u());
    let crv = _mm256_set1_epi32(coeffs.r_v());
    let cgu = _mm256_set1_epi32(coeffs.g_u());
    let cgv = _mm256_set1_epi32(coeffs.g_v());
    let cbu = _mm256_set1_epi32(coeffs.b_u());
    let cbv = _mm256_set1_epi32(coeffs.b_v());

    // Main loop: 12 pixels (2 v210 words = 32 bytes) per iteration.
    let pairs = words / 2;
    for p in 0..pairs {
      let (y_vec, u_vec, v_vec) = unpack_v210_2words_avx2(packed.as_ptr().add(p * 32));

      let y_i16 = y_vec;

      // Subtract chroma bias (512 for 10-bit) — fits i16 since each
      // chroma sample is ≤ 1023.
      let u_i16 = _mm256_sub_epi16(u_vec, bias_v);
      let v_i16 = _mm256_sub_epi16(v_vec, bias_v);

      // Widen 16-lane i16 chroma to two i32x8 halves so the Q15
      // multiplies don't overflow. Only lanes 0..6 of `_lo` are
      // valid; `_hi` is entirely don't-care. We feed both halves
      // through `chroma_i16x16` to recycle the helper's exact code
      // path; the don't-care output lanes are discarded by the
      // 12-pixel partial store.
      let u_lo_i32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(u_i16));
      let u_hi_i32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(u_i16));
      let v_lo_i32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v_i16));
      let v_hi_i32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(v_i16));

      let u_d_lo = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_lo_i32, c_scale_v),
        rnd_v,
      ));
      let u_d_hi = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_hi_i32, c_scale_v),
        rnd_v,
      ));
      let v_d_lo = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(v_lo_i32, c_scale_v),
        rnd_v,
      ));
      let v_d_hi = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(v_hi_i32, c_scale_v),
        rnd_v,
      ));

      // 16-lane chroma vectors with valid data in lanes 0..6.
      let r_chroma = chroma_i16x16(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let g_chroma = chroma_i16x16(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let b_chroma = chroma_i16x16(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);

      // Each chroma sample covers 2 Y lanes (4:2:2): `chroma_dup`
      // duplicates each of 16 chroma lanes into its pair slot,
      // splitting across two i16x16 vectors. With 6 valid chroma in
      // lanes 0..6, `lo16` lanes 0..12 are valid (= [c0,c0, c1,c1,
      // ..., c5,c5]); the rest is don't-care. `hi16` is entirely
      // don't-care.
      let (r_dup_lo, _r_dup_hi) = chroma_dup(r_chroma);
      let (g_dup_lo, _g_dup_hi) = chroma_dup(g_chroma);
      let (b_dup_lo, _b_dup_hi) = chroma_dup(b_chroma);

      // Y scale: `(Y - y_off) * y_scale + RND >> 15` → i16x16.
      let y_scaled = scale_y(y_i16, y_off_v, y_scale_v, rnd_v);

      // Per-channel saturating add. Result first 12 lanes valid u8.
      let r_sum = _mm256_adds_epi16(y_scaled, r_dup_lo);
      let g_sum = _mm256_adds_epi16(y_scaled, g_dup_lo);
      let b_sum = _mm256_adds_epi16(y_scaled, b_dup_lo);

      // u8 narrow with saturation. `narrow_u8x32(lo, _mm256_setzero_si256())`
      // packs 16 lanes of `lo` to u8 in the result's first 16 bytes
      // (next 16 zero, after the lane-fixup permute). Only the first
      // 12 bytes per channel matter.
      let zero = _mm256_setzero_si256();
      let r_u8 = narrow_u8x32(r_sum, zero);
      let g_u8 = narrow_u8x32(g_sum, zero);
      let b_u8 = narrow_u8x32(b_sum, zero);

      // 12-pixel partial store: dump per-channel u8 into stack
      // scratch then build the interleaved RGB / RGBA output via
      // scalar copies (mirrors SSE4.1 / NEON pattern; AVX2 has no
      // 12-lane interleaved store).
      let mut r_tmp = [0u8; 32];
      let mut g_tmp = [0u8; 32];
      let mut b_tmp = [0u8; 32];
      _mm256_storeu_si256(r_tmp.as_mut_ptr().cast(), r_u8);
      _mm256_storeu_si256(g_tmp.as_mut_ptr().cast(), g_u8);
      _mm256_storeu_si256(b_tmp.as_mut_ptr().cast(), b_u8);

      let base_px = p * 12;
      if ALPHA {
        let dst = &mut out[base_px * 4..base_px * 4 + 12 * 4];
        for i in 0..12 {
          dst[i * 4] = r_tmp[i];
          dst[i * 4 + 1] = g_tmp[i];
          dst[i * 4 + 2] = b_tmp[i];
          dst[i * 4 + 3] = 0xFF;
        }
      } else {
        let dst = &mut out[base_px * 3..base_px * 3 + 12 * 3];
        for i in 0..12 {
          dst[i * 3] = r_tmp[i];
          dst[i * 3 + 1] = g_tmp[i];
          dst[i * 3 + 2] = b_tmp[i];
        }
      }
    }

    // Tail: any remaining single word (6 px) goes through scalar.
    if pairs * 2 < words {
      let tail_start_px = pairs * 12;
      let tail_packed = &packed[pairs * 32..words * 16];
      let tail_out = &mut out[tail_start_px * bpp..width * bpp];
      let tail_w = width - tail_start_px;
      scalar::v210_to_rgb_or_rgba_row::<ALPHA>(tail_packed, tail_out, tail_w, matrix, full_range);
    }
  }
}

/// AVX2 v210 → packed `u16` RGB / RGBA at native 10-bit depth
/// (low-bit-packed). Byte-identical to
/// `scalar::v210_to_rgb_u16_or_rgba_u16_row::<ALPHA>`.
///
/// # Safety
///
/// 1. **AVX2 must be available.**
/// 2. `width % 6 == 0`.
/// 3. `packed.len() >= (width / 6) * 16`.
/// 4. `out.len() >= width * (if ALPHA { 4 } else { 3 })` (`u16` elements).
#[inline]
#[target_feature(enable = "avx2")]
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
    let rnd_v = _mm256_set1_epi32(RND);
    let y_off_v = _mm256_set1_epi16(y_off as i16);
    let y_scale_v = _mm256_set1_epi32(y_scale);
    let c_scale_v = _mm256_set1_epi32(c_scale);
    let bias_v = _mm256_set1_epi16(bias as i16);
    let max_v = _mm256_set1_epi16(out_max);
    let zero_v = _mm256_set1_epi16(0);
    let cru = _mm256_set1_epi32(coeffs.r_u());
    let crv = _mm256_set1_epi32(coeffs.r_v());
    let cgu = _mm256_set1_epi32(coeffs.g_u());
    let cgv = _mm256_set1_epi32(coeffs.g_v());
    let cbu = _mm256_set1_epi32(coeffs.b_u());
    let cbv = _mm256_set1_epi32(coeffs.b_v());

    let pairs = words / 2;
    for p in 0..pairs {
      let (y_vec, u_vec, v_vec) = unpack_v210_2words_avx2(packed.as_ptr().add(p * 32));

      let y_i16 = y_vec;
      let u_i16 = _mm256_sub_epi16(u_vec, bias_v);
      let v_i16 = _mm256_sub_epi16(v_vec, bias_v);

      let u_lo_i32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(u_i16));
      let u_hi_i32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(u_i16));
      let v_lo_i32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v_i16));
      let v_hi_i32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(v_i16));

      let u_d_lo = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_lo_i32, c_scale_v),
        rnd_v,
      ));
      let u_d_hi = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_hi_i32, c_scale_v),
        rnd_v,
      ));
      let v_d_lo = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(v_lo_i32, c_scale_v),
        rnd_v,
      ));
      let v_d_hi = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(v_hi_i32, c_scale_v),
        rnd_v,
      ));

      let r_chroma = chroma_i16x16(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let g_chroma = chroma_i16x16(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let b_chroma = chroma_i16x16(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);

      let (r_dup_lo, _r_dup_hi) = chroma_dup(r_chroma);
      let (g_dup_lo, _g_dup_hi) = chroma_dup(g_chroma);
      let (b_dup_lo, _b_dup_hi) = chroma_dup(b_chroma);

      let y_scaled = scale_y(y_i16, y_off_v, y_scale_v, rnd_v);

      // Native-depth output: clamp to [0, (1 << 10) - 1]. The AVX2
      // `clamp_u16_max_x16` mirrors SSE4.1's `clamp_u16_max`.
      let r = clamp_u16_max_x16(_mm256_adds_epi16(y_scaled, r_dup_lo), zero_v, max_v);
      let g = clamp_u16_max_x16(_mm256_adds_epi16(y_scaled, g_dup_lo), zero_v, max_v);
      let b = clamp_u16_max_x16(_mm256_adds_epi16(y_scaled, b_dup_lo), zero_v, max_v);

      // 12-pixel partial u16 store via stack buffer + scalar interleave.
      let mut r_tmp = [0u16; 16];
      let mut g_tmp = [0u16; 16];
      let mut b_tmp = [0u16; 16];
      _mm256_storeu_si256(r_tmp.as_mut_ptr().cast(), r);
      _mm256_storeu_si256(g_tmp.as_mut_ptr().cast(), g);
      _mm256_storeu_si256(b_tmp.as_mut_ptr().cast(), b);

      let base_px = p * 12;
      if ALPHA {
        let dst = &mut out[base_px * 4..base_px * 4 + 12 * 4];
        let alpha = out_max as u16;
        for i in 0..12 {
          dst[i * 4] = r_tmp[i];
          dst[i * 4 + 1] = g_tmp[i];
          dst[i * 4 + 2] = b_tmp[i];
          dst[i * 4 + 3] = alpha;
        }
      } else {
        let dst = &mut out[base_px * 3..base_px * 3 + 12 * 3];
        for i in 0..12 {
          dst[i * 3] = r_tmp[i];
          dst[i * 3 + 1] = g_tmp[i];
          dst[i * 3 + 2] = b_tmp[i];
        }
      }
    }

    // Tail: any remaining single word (6 px) goes through scalar.
    if pairs * 2 < words {
      let tail_start_px = pairs * 12;
      let tail_packed = &packed[pairs * 32..words * 16];
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

/// AVX2 v210 → 8-bit luma. Y values are downshifted from 10-bit to
/// 8-bit via `>> 2`. Bypasses the YUV → RGB pipeline entirely.
/// Byte-identical to `scalar::v210_to_luma_row`.
///
/// # Safety
///
/// 1. **AVX2 must be available.**
/// 2. `width % 6 == 0`.
/// 3. `packed.len() >= (width / 6) * 16`.
/// 4. `luma_out.len() >= width`.
#[inline]
#[target_feature(enable = "avx2")]
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
    let pairs = words / 2;
    for p in 0..pairs {
      let (y_vec, _, _) = unpack_v210_2words_avx2(packed.as_ptr().add(p * 32));
      // Downshift 10-bit Y by 2 → 8-bit, narrow to u8x32 via packus.
      let y_shr = _mm256_srli_epi16::<2>(y_vec);
      let y_u8 = narrow_u8x32(y_shr, _mm256_setzero_si256());
      // Store first 12 of the u8x32 lanes via stack buffer + copy_from_slice.
      let mut tmp = [0u8; 32];
      _mm256_storeu_si256(tmp.as_mut_ptr().cast(), y_u8);
      luma_out[p * 12..p * 12 + 12].copy_from_slice(&tmp[..12]);
    }

    // Tail: any remaining single word (6 px) goes through scalar.
    if pairs * 2 < words {
      let tail_start_px = pairs * 12;
      let tail_packed = &packed[pairs * 32..words * 16];
      let tail_out = &mut luma_out[tail_start_px..width];
      let tail_w = width - tail_start_px;
      scalar::v210_to_luma_row(tail_packed, tail_out, tail_w);
    }
  }
}

/// AVX2 v210 → native-depth `u16` luma (low-bit-packed). Each output
/// `u16` carries the source's 10-bit Y value in its low 10 bits.
/// Byte-identical to `scalar::v210_to_luma_u16_row`.
///
/// # Safety
///
/// 1. **AVX2 must be available.**
/// 2. `width % 6 == 0`.
/// 3. `packed.len() >= (width / 6) * 16`.
/// 4. `luma_out.len() >= width`.
#[inline]
#[target_feature(enable = "avx2")]
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
    let pairs = words / 2;
    for p in 0..pairs {
      let (y_vec, _, _) = unpack_v210_2words_avx2(packed.as_ptr().add(p * 32));
      // Store first 12 of the 16 u16 lanes via stack buffer + copy_from_slice.
      let mut tmp = [0u16; 16];
      _mm256_storeu_si256(tmp.as_mut_ptr().cast(), y_vec);
      luma_out[p * 12..p * 12 + 12].copy_from_slice(&tmp[..12]);
    }

    // Tail: any remaining single word (6 px) goes through scalar.
    if pairs * 2 < words {
      let tail_start_px = pairs * 12;
      let tail_packed = &packed[pairs * 32..words * 16];
      let tail_out = &mut luma_out[tail_start_px..width];
      let tail_w = width - tail_start_px;
      scalar::v210_to_luma_u16_row(tail_packed, tail_out, tail_w);
    }
  }
}
