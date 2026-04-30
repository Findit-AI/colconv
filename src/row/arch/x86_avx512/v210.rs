//! AVX-512 v210 (Tier 4 packed YUV 4:2:2 10-bit) kernels. Four v210
//! words = 64 bytes = 24 pixels processed per iteration.
//!
//! Bit extraction uses three shifted-AND ops at 512-bit width to pull
//! the three 10-bit fields from each 32-bit lane (16 lanes per
//! `__m512i`), then **three `_mm512_permutexvar_epi16` calls plus two
//! `_mm512_or_si512` ops** consolidate the per-source contributions
//! into a single 32-lane u16 vector for each of Y, U, V. Unlike AVX2
//! (which needs per-128-bit-lane shuffle + cross-lane permute), AVX-512's
//! cross-lane u16 permute makes this one-shot — the permute index for
//! each (output, source) pair is a static `[i16; 32]` table.
//!
//! Source-vector layout: after the AND-mask, each 10-bit sample sits in
//! the **low 16 bits** of its 32-bit lane (the upper 16 bits are zero).
//! Reinterpreted as 32 u16 lanes, u32 lane `n` corresponds to u16 lane
//! `2n` (data) and `2n+1` (zero). The permute indices below are built
//! against this u16 view; "don't-care" output positions point to one
//! of the always-zero odd lanes (we use index `1`).
//!
//! Per the v210 spec, for word `w` ∈ `0..4` inside the 64-byte block,
//! the per-word u32 layout is:
//!   word 0: low=Cb0, mid=Y0,  high=Cr0
//!   word 1: low=Y1,  mid=Cb1, high=Y2
//!   word 2: low=Cr1, mid=Y3,  high=Cb2
//!   word 3: low=Y4,  mid=Cr2, high=Y5
//! and that pattern repeats for words 4..7 (next 24 px), but each
//! 64-byte block is one iteration so we map the same shape onto u32
//! lanes `4w + 0..3` for `w ∈ 0..4`.
//!
//! After unpack we feed the resulting 24-Y-lane / 12-chroma-lane
//! vectors into the existing AVX-512 BITS=10 helpers ([`chroma_i16x32`],
//! [`chroma_dup`], [`scale_y`], [`narrow_u8x64`],
//! [`clamp_u16_max_x32`]). The full helpers process 64 Y / 32 chroma
//! per call; we fold our partial vectors (24 / 12) through the same
//! code path — `chroma_dup`'s `lo32` output captures all 24 valid
//! duplicated chroma in lanes 0..24, and the unused upper output lanes
//! are discarded by the 24-pixel partial store. Output stores are
//! built via stack scratch + scalar interleave (same pattern as the
//! AVX2 sibling).

use core::arch::x86_64::*;

use super::*;
use crate::{ColorMatrix, row::scalar};

// ---- Static permute index tables --------------------------------------
//
// Each table is a 32-lane u16 permute index for `_mm512_permutexvar_epi16`.
// Output lane `i` picks from source u16 lane `tbl[i]`. Don't-care output
// positions use index `1` — an always-zero odd u16 lane in `low10`,
// `mid10`, `high10` (the upper 16 bits of u32 lane 0, zero after the
// 0x3FF AND-mask). ORing zeros from the two non-contributing sources
// into each valid output lane preserves the third source's value.
//
// Per word `w ∈ 0..4`:
//   Y[6w + 0] = mid [u32 lane 4w + 0] → u16 lane 8w + 0
//   Y[6w + 1] = low [u32 lane 4w + 1] → u16 lane 8w + 2
//   Y[6w + 2] = high[u32 lane 4w + 1] → u16 lane 8w + 2
//   Y[6w + 3] = mid [u32 lane 4w + 2] → u16 lane 8w + 4
//   Y[6w + 4] = low [u32 lane 4w + 3] → u16 lane 8w + 6
//   Y[6w + 5] = high[u32 lane 4w + 3] → u16 lane 8w + 6
//   U[3w + 0] = low [u32 lane 4w + 0] → u16 lane 8w + 0
//   U[3w + 1] = mid [u32 lane 4w + 1] → u16 lane 8w + 2
//   U[3w + 2] = high[u32 lane 4w + 2] → u16 lane 8w + 4
//   V[3w + 0] = high[u32 lane 4w + 0] → u16 lane 8w + 0
//   V[3w + 1] = low [u32 lane 4w + 2] → u16 lane 8w + 4
//   V[3w + 2] = mid [u32 lane 4w + 3] → u16 lane 8w + 6
//
// `_mm512_setr_epi16` is **not** available in stable stdarch, so we use
// `static [i16; 32]` arrays loaded via `_mm512_loadu_si512(ptr.cast())`
// (matches Ship 10's similar workaround).

#[rustfmt::skip]
static Y_FROM_MID: [i16; 32] = [
  // Y output lane → mid source u16 lane (Y[6w + 0] at lane 6w, Y[6w + 3]
  // at lane 6w + 3). Other output lanes (1, 2, 4, 5, 7, 8, 10, 11, 13,
  // 14, 16, 17, 19, 20, 22, 23, and the don't-care 24..32) → 1.
  0,  1, 1, 4,  1, 1,  // w=0: lane 0=mid[0],  lane 3=mid[4]
  8,  1, 1, 12, 1, 1,  // w=1: lane 6=mid[8],  lane 9=mid[12]
  16, 1, 1, 20, 1, 1,  // w=2: lane 12=mid[16], lane 15=mid[20]
  24, 1, 1, 28, 1, 1,  // w=3: lane 18=mid[24], lane 21=mid[28]
  1, 1, 1, 1, 1, 1, 1, 1, // lanes 24..32: don't-care
];

#[rustfmt::skip]
static Y_FROM_LOW: [i16; 32] = [
  // Y[6w + 1] at output lane 6w + 1 = low[u32 lane 4w + 1] = u16 lane 8w + 2
  // Y[6w + 4] at output lane 6w + 4 = low[u32 lane 4w + 3] = u16 lane 8w + 6
  1, 2, 1, 1, 6, 1,    // w=0: lane 1=low[2], lane 4=low[6]
  1, 10, 1, 1, 14, 1,  // w=1: lane 7=low[10], lane 10=low[14]
  1, 18, 1, 1, 22, 1,  // w=2: lane 13=low[18], lane 16=low[22]
  1, 26, 1, 1, 30, 1,  // w=3: lane 19=low[26], lane 22=low[30]
  1, 1, 1, 1, 1, 1, 1, 1, // lanes 24..32: don't-care
];

#[rustfmt::skip]
static Y_FROM_HIGH: [i16; 32] = [
  // Y[6w + 2] at output lane 6w + 2 = high[u32 lane 4w + 1] = u16 lane 8w + 2
  // Y[6w + 5] at output lane 6w + 5 = high[u32 lane 4w + 3] = u16 lane 8w + 6
  1, 1, 2, 1, 1, 6,    // w=0: lane 2=high[2], lane 5=high[6]
  1, 1, 10, 1, 1, 14,  // w=1: lane 8=high[10], lane 11=high[14]
  1, 1, 18, 1, 1, 22,  // w=2: lane 14=high[18], lane 17=high[22]
  1, 1, 26, 1, 1, 30,  // w=3: lane 20=high[26], lane 23=high[30]
  1, 1, 1, 1, 1, 1, 1, 1, // lanes 24..32: don't-care
];

#[rustfmt::skip]
static U_FROM_LOW: [i16; 32] = [
  // U[3w + 0] at output lane 3w = low[u32 lane 4w + 0] = u16 lane 8w
  0, 1, 1,             // w=0: lane 0=low[0]
  8, 1, 1,             // w=1: lane 3=low[8]
  16, 1, 1,            // w=2: lane 6=low[16]
  24, 1, 1,            // w=3: lane 9=low[24]
  // lanes 12..32: don't-care
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
];

#[rustfmt::skip]
static U_FROM_MID: [i16; 32] = [
  // U[3w + 1] at output lane 3w + 1 = mid[u32 lane 4w + 1] = u16 lane 8w + 2
  1, 2, 1,             // w=0: lane 1=mid[2]
  1, 10, 1,            // w=1: lane 4=mid[10]
  1, 18, 1,            // w=2: lane 7=mid[18]
  1, 26, 1,            // w=3: lane 10=mid[26]
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
];

#[rustfmt::skip]
static U_FROM_HIGH: [i16; 32] = [
  // U[3w + 2] at output lane 3w + 2 = high[u32 lane 4w + 2] = u16 lane 8w + 4
  1, 1, 4,             // w=0: lane 2=high[4]
  1, 1, 12,            // w=1: lane 5=high[12]
  1, 1, 20,            // w=2: lane 8=high[20]
  1, 1, 28,            // w=3: lane 11=high[28]
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
];

#[rustfmt::skip]
static V_FROM_HIGH: [i16; 32] = [
  // V[3w + 0] at output lane 3w = high[u32 lane 4w + 0] = u16 lane 8w
  0, 1, 1,             // w=0: lane 0=high[0]
  8, 1, 1,             // w=1: lane 3=high[8]
  16, 1, 1,            // w=2: lane 6=high[16]
  24, 1, 1,            // w=3: lane 9=high[24]
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
];

#[rustfmt::skip]
static V_FROM_LOW: [i16; 32] = [
  // V[3w + 1] at output lane 3w + 1 = low[u32 lane 4w + 2] = u16 lane 8w + 4
  1, 4, 1,             // w=0: lane 1=low[4]
  1, 12, 1,            // w=1: lane 4=low[12]
  1, 20, 1,            // w=2: lane 7=low[20]
  1, 28, 1,            // w=3: lane 10=low[28]
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
];

#[rustfmt::skip]
static V_FROM_MID: [i16; 32] = [
  // V[3w + 2] at output lane 3w + 2 = mid[u32 lane 4w + 3] = u16 lane 8w + 6
  1, 1, 6,             // w=0: lane 2=mid[6]
  1, 1, 14,            // w=1: lane 5=mid[14]
  1, 1, 22,            // w=2: lane 8=mid[22]
  1, 1, 30,            // w=3: lane 11=mid[30]
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
];

/// Unpacks four consecutive 16-byte v210 words (= 24 pixels) into
/// three vectors holding 10-bit samples in their low bits:
/// - `y_vec`: i16x32 with lanes 0..24 = Y0..Y23 (lanes 24..32 are
///   don't-care).
/// - `u_vec`: i16x32 with lanes 0..12 = Cb0..Cb11 (rest don't-care).
/// - `v_vec`: i16x32 with lanes 0..12 = Cr0..Cr11 (rest don't-care).
///
/// Strategy: load 64 bytes (= 16 × u32 = 4 v210 words) via
/// `_mm512_loadu_si512`, apply three shifted-AND ops to materialize
/// `low10` / `mid10` / `high10` (each 10-bit field per 32-bit lane).
/// Then for each of Y/U/V issue three `_mm512_permutexvar_epi16` calls
/// (one per source) and OR the results. Don't-care output lanes pick
/// an always-zero odd u16 lane (index 1) so the OR consolidation is
/// well-defined.
///
/// # Safety
///
/// Caller must ensure `ptr` has at least 64 bytes readable, and
/// `target_feature` includes AVX-512F + AVX-512BW (BW provides the u16
/// `permutexvar` op `vpermw`).
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn unpack_v210_4words_avx512(ptr: *const u8) -> (__m512i, __m512i, __m512i) {
  // SAFETY: caller obligation — `ptr` has 64 bytes readable; AVX-512F
  // + AVX-512BW are available.
  unsafe {
    let words = _mm512_loadu_si512(ptr.cast());
    let mask10 = _mm512_set1_epi32(0x3FF);
    let low10 = _mm512_and_si512(words, mask10);
    let mid10 = _mm512_and_si512(_mm512_srli_epi32::<10>(words), mask10);
    let high10 = _mm512_and_si512(_mm512_srli_epi32::<20>(words), mask10);

    // ---- Y assembly -----------------------------------------------------
    // 24 valid output lanes = Y0..Y23 (lanes 24..32 don't-care).
    let y_idx_mid = _mm512_loadu_si512(Y_FROM_MID.as_ptr().cast());
    let y_idx_low = _mm512_loadu_si512(Y_FROM_LOW.as_ptr().cast());
    let y_idx_high = _mm512_loadu_si512(Y_FROM_HIGH.as_ptr().cast());
    let y_from_mid = _mm512_permutexvar_epi16(y_idx_mid, mid10);
    let y_from_low = _mm512_permutexvar_epi16(y_idx_low, low10);
    let y_from_high = _mm512_permutexvar_epi16(y_idx_high, high10);
    let y_vec = _mm512_or_si512(_mm512_or_si512(y_from_mid, y_from_low), y_from_high);

    // ---- U assembly -----------------------------------------------------
    // 12 valid output lanes = Cb0..Cb11 (lanes 12..32 don't-care).
    let u_idx_low = _mm512_loadu_si512(U_FROM_LOW.as_ptr().cast());
    let u_idx_mid = _mm512_loadu_si512(U_FROM_MID.as_ptr().cast());
    let u_idx_high = _mm512_loadu_si512(U_FROM_HIGH.as_ptr().cast());
    let u_from_low = _mm512_permutexvar_epi16(u_idx_low, low10);
    let u_from_mid = _mm512_permutexvar_epi16(u_idx_mid, mid10);
    let u_from_high = _mm512_permutexvar_epi16(u_idx_high, high10);
    let u_vec = _mm512_or_si512(_mm512_or_si512(u_from_low, u_from_mid), u_from_high);

    // ---- V assembly -----------------------------------------------------
    // 12 valid output lanes = Cr0..Cr11 (lanes 12..32 don't-care).
    let v_idx_high = _mm512_loadu_si512(V_FROM_HIGH.as_ptr().cast());
    let v_idx_low = _mm512_loadu_si512(V_FROM_LOW.as_ptr().cast());
    let v_idx_mid = _mm512_loadu_si512(V_FROM_MID.as_ptr().cast());
    let v_from_high = _mm512_permutexvar_epi16(v_idx_high, high10);
    let v_from_low = _mm512_permutexvar_epi16(v_idx_low, low10);
    let v_from_mid = _mm512_permutexvar_epi16(v_idx_mid, mid10);
    let v_vec = _mm512_or_si512(_mm512_or_si512(v_from_high, v_from_low), v_from_mid);

    (y_vec, u_vec, v_vec)
  }
}

/// AVX-512 v210 → packed RGB / RGBA (u8). Const-generic on `ALPHA`:
/// `false` writes 3 bytes per pixel, `true` writes 4 bytes per pixel
/// with `α = 0xFF`. Output bit depth is u8 (downshifted from the
/// native 10-bit Q15 pipeline via `range_params_n::<10, 8>`).
///
/// Byte-identical to `scalar::v210_to_rgb_or_rgba_row::<ALPHA>` for
/// every input.
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available on the current CPU.**
/// 2. `width % 6 == 0`.
/// 3. `packed.len() >= (width / 6) * 16`.
/// 4. `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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

  // SAFETY: AVX-512BW availability is the caller's obligation; the
  // dispatcher in `crate::row` verifies it. Pointer adds are bounded
  // by the `for q in 0..quads` / tail loop and the caller-promised
  // slice lengths checked above.
  unsafe {
    let rnd_v = _mm512_set1_epi32(RND);
    let y_off_v = _mm512_set1_epi16(y_off as i16);
    let y_scale_v = _mm512_set1_epi32(y_scale);
    let c_scale_v = _mm512_set1_epi32(c_scale);
    let bias_v = _mm512_set1_epi16(bias as i16);
    let cru = _mm512_set1_epi32(coeffs.r_u());
    let crv = _mm512_set1_epi32(coeffs.r_v());
    let cgu = _mm512_set1_epi32(coeffs.g_u());
    let cgv = _mm512_set1_epi32(coeffs.g_v());
    let cbu = _mm512_set1_epi32(coeffs.b_u());
    let cbv = _mm512_set1_epi32(coeffs.b_v());

    let pack_fixup = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
    let dup_lo_idx = _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11);
    let dup_hi_idx = _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15);

    // Main loop: 24 pixels (4 v210 words = 64 bytes) per iteration.
    let quads = words / 4;
    for q in 0..quads {
      let (y_vec, u_vec, v_vec) = unpack_v210_4words_avx512(packed.as_ptr().add(q * 64));

      let y_i16 = y_vec;

      // Subtract chroma bias (512 for 10-bit). Only lanes 0..12 carry
      // valid samples, but applying the bias to the don't-care lanes
      // is harmless since they're discarded by the 24-pixel partial
      // store.
      let u_i16 = _mm512_sub_epi16(u_vec, bias_v);
      let v_i16 = _mm512_sub_epi16(v_vec, bias_v);

      let u_lo_i32 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_i16));
      let u_hi_i32 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(u_i16));
      let v_lo_i32 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_i16));
      let v_hi_i32 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(v_i16));

      let u_d_lo = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_lo_i32, c_scale_v),
        rnd_v,
      ));
      let u_d_hi = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_hi_i32, c_scale_v),
        rnd_v,
      ));
      let v_d_lo = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_lo_i32, c_scale_v),
        rnd_v,
      ));
      let v_d_hi = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_hi_i32, c_scale_v),
        rnd_v,
      ));

      // i16x32 chroma vectors. Lanes 0..12 valid; 12..32 don't-care.
      let r_chroma = chroma_i16x32(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v, pack_fixup);
      let g_chroma = chroma_i16x32(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v, pack_fixup);
      let b_chroma = chroma_i16x32(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v, pack_fixup);

      // Each chroma sample covers 2 Y lanes (4:2:2). `chroma_dup`
      // duplicates each of 32 chroma lanes into its pair slot,
      // splitting across two i16x32 vectors. With 12 valid chroma in
      // lanes 0..12, `lo32` lanes 0..24 are valid (= [c0,c0, c1,c1,
      // ..., c11,c11]); lanes 24..32 of `lo32` and all of `hi32` are
      // don't-care.
      let (r_dup_lo, _r_dup_hi) = chroma_dup(r_chroma, dup_lo_idx, dup_hi_idx);
      let (g_dup_lo, _g_dup_hi) = chroma_dup(g_chroma, dup_lo_idx, dup_hi_idx);
      let (b_dup_lo, _b_dup_hi) = chroma_dup(b_chroma, dup_lo_idx, dup_hi_idx);

      // Y scale: `(Y - y_off) * y_scale + RND >> 15` → i16x32.
      let y_scaled = scale_y(y_i16, y_off_v, y_scale_v, rnd_v, pack_fixup);

      // Per-channel saturating add. Result first 24 lanes valid u8.
      let r_sum = _mm512_adds_epi16(y_scaled, r_dup_lo);
      let g_sum = _mm512_adds_epi16(y_scaled, g_dup_lo);
      let b_sum = _mm512_adds_epi16(y_scaled, b_dup_lo);

      // u8 narrow with saturation. `narrow_u8x64(lo, _mm512_setzero_si512(), pack_fixup)`
      // packs 32 lanes of `lo` to u8 in the result's first 32 bytes
      // (next 32 zero, after the lane-fixup permute). Only the first
      // 24 bytes per channel matter.
      let zero = _mm512_setzero_si512();
      let r_u8 = narrow_u8x64(r_sum, zero, pack_fixup);
      let g_u8 = narrow_u8x64(g_sum, zero, pack_fixup);
      let b_u8 = narrow_u8x64(b_sum, zero, pack_fixup);

      // 24-pixel partial store: dump per-channel u8 into stack scratch
      // then build the interleaved RGB / RGBA output via scalar copies
      // (mirrors AVX2 / SSE4.1 / NEON pattern; AVX-512 has no
      // 24-lane interleaved store).
      let mut r_tmp = [0u8; 64];
      let mut g_tmp = [0u8; 64];
      let mut b_tmp = [0u8; 64];
      _mm512_storeu_si512(r_tmp.as_mut_ptr().cast(), r_u8);
      _mm512_storeu_si512(g_tmp.as_mut_ptr().cast(), g_u8);
      _mm512_storeu_si512(b_tmp.as_mut_ptr().cast(), b_u8);

      let base_px = q * 24;
      if ALPHA {
        let dst = &mut out[base_px * 4..base_px * 4 + 24 * 4];
        for i in 0..24 {
          dst[i * 4] = r_tmp[i];
          dst[i * 4 + 1] = g_tmp[i];
          dst[i * 4 + 2] = b_tmp[i];
          dst[i * 4 + 3] = 0xFF;
        }
      } else {
        let dst = &mut out[base_px * 3..base_px * 3 + 24 * 3];
        for i in 0..24 {
          dst[i * 3] = r_tmp[i];
          dst[i * 3 + 1] = g_tmp[i];
          dst[i * 3 + 2] = b_tmp[i];
        }
      }
    }

    // Tail: any remaining 1, 2, or 3 words (6, 12, or 18 px) goes
    // through scalar.
    if quads * 4 < words {
      let tail_start_px = quads * 24;
      let tail_packed = &packed[quads * 64..words * 16];
      let tail_out = &mut out[tail_start_px * bpp..width * bpp];
      let tail_w = width - tail_start_px;
      scalar::v210_to_rgb_or_rgba_row::<ALPHA>(tail_packed, tail_out, tail_w, matrix, full_range);
    }
  }
}

/// AVX-512 v210 → packed `u16` RGB / RGBA at native 10-bit depth
/// (low-bit-packed). Byte-identical to
/// `scalar::v210_to_rgb_u16_or_rgba_u16_row::<ALPHA>`.
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available.**
/// 2. `width % 6 == 0`.
/// 3. `packed.len() >= (width / 6) * 16`.
/// 4. `out.len() >= width * (if ALPHA { 4 } else { 3 })` (`u16` elements).
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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
    let rnd_v = _mm512_set1_epi32(RND);
    let y_off_v = _mm512_set1_epi16(y_off as i16);
    let y_scale_v = _mm512_set1_epi32(y_scale);
    let c_scale_v = _mm512_set1_epi32(c_scale);
    let bias_v = _mm512_set1_epi16(bias as i16);
    let max_v = _mm512_set1_epi16(out_max);
    let zero_v = _mm512_set1_epi16(0);
    let cru = _mm512_set1_epi32(coeffs.r_u());
    let crv = _mm512_set1_epi32(coeffs.r_v());
    let cgu = _mm512_set1_epi32(coeffs.g_u());
    let cgv = _mm512_set1_epi32(coeffs.g_v());
    let cbu = _mm512_set1_epi32(coeffs.b_u());
    let cbv = _mm512_set1_epi32(coeffs.b_v());

    let pack_fixup = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
    let dup_lo_idx = _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11);
    let dup_hi_idx = _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15);

    let quads = words / 4;
    for q in 0..quads {
      let (y_vec, u_vec, v_vec) = unpack_v210_4words_avx512(packed.as_ptr().add(q * 64));

      let y_i16 = y_vec;
      let u_i16 = _mm512_sub_epi16(u_vec, bias_v);
      let v_i16 = _mm512_sub_epi16(v_vec, bias_v);

      let u_lo_i32 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_i16));
      let u_hi_i32 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(u_i16));
      let v_lo_i32 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_i16));
      let v_hi_i32 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(v_i16));

      let u_d_lo = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_lo_i32, c_scale_v),
        rnd_v,
      ));
      let u_d_hi = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_hi_i32, c_scale_v),
        rnd_v,
      ));
      let v_d_lo = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_lo_i32, c_scale_v),
        rnd_v,
      ));
      let v_d_hi = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_hi_i32, c_scale_v),
        rnd_v,
      ));

      let r_chroma = chroma_i16x32(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v, pack_fixup);
      let g_chroma = chroma_i16x32(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v, pack_fixup);
      let b_chroma = chroma_i16x32(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v, pack_fixup);

      let (r_dup_lo, _r_dup_hi) = chroma_dup(r_chroma, dup_lo_idx, dup_hi_idx);
      let (g_dup_lo, _g_dup_hi) = chroma_dup(g_chroma, dup_lo_idx, dup_hi_idx);
      let (b_dup_lo, _b_dup_hi) = chroma_dup(b_chroma, dup_lo_idx, dup_hi_idx);

      let y_scaled = scale_y(y_i16, y_off_v, y_scale_v, rnd_v, pack_fixup);

      // Native-depth output: clamp to [0, (1 << 10) - 1].
      let r = clamp_u16_max_x32(_mm512_adds_epi16(y_scaled, r_dup_lo), zero_v, max_v);
      let g = clamp_u16_max_x32(_mm512_adds_epi16(y_scaled, g_dup_lo), zero_v, max_v);
      let b = clamp_u16_max_x32(_mm512_adds_epi16(y_scaled, b_dup_lo), zero_v, max_v);

      // 24-pixel partial u16 store via stack buffer + scalar interleave.
      let mut r_tmp = [0u16; 32];
      let mut g_tmp = [0u16; 32];
      let mut b_tmp = [0u16; 32];
      _mm512_storeu_si512(r_tmp.as_mut_ptr().cast(), r);
      _mm512_storeu_si512(g_tmp.as_mut_ptr().cast(), g);
      _mm512_storeu_si512(b_tmp.as_mut_ptr().cast(), b);

      let base_px = q * 24;
      if ALPHA {
        let dst = &mut out[base_px * 4..base_px * 4 + 24 * 4];
        let alpha = out_max as u16;
        for i in 0..24 {
          dst[i * 4] = r_tmp[i];
          dst[i * 4 + 1] = g_tmp[i];
          dst[i * 4 + 2] = b_tmp[i];
          dst[i * 4 + 3] = alpha;
        }
      } else {
        let dst = &mut out[base_px * 3..base_px * 3 + 24 * 3];
        for i in 0..24 {
          dst[i * 3] = r_tmp[i];
          dst[i * 3 + 1] = g_tmp[i];
          dst[i * 3 + 2] = b_tmp[i];
        }
      }
    }

    // Tail: any remaining 1, 2, or 3 words (6, 12, or 18 px) goes
    // through scalar.
    if quads * 4 < words {
      let tail_start_px = quads * 24;
      let tail_packed = &packed[quads * 64..words * 16];
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

/// AVX-512 v210 → 8-bit luma. Y values are downshifted from 10-bit to
/// 8-bit via `>> 2`. Bypasses the YUV → RGB pipeline entirely.
/// Byte-identical to `scalar::v210_to_luma_row`.
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available.**
/// 2. `width % 6 == 0`.
/// 3. `packed.len() >= (width / 6) * 16`.
/// 4. `luma_out.len() >= width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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
    let pack_fixup = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
    let zero = _mm512_setzero_si512();

    let quads = words / 4;
    for q in 0..quads {
      let (y_vec, _, _) = unpack_v210_4words_avx512(packed.as_ptr().add(q * 64));
      // Downshift 10-bit Y by 2 → 8-bit, narrow to u8x64 via packus
      // (only first 32 lanes carry data, paired with a zero hi half;
      // first 24 bytes of the result are valid Y0..Y23).
      let y_shr = _mm512_srli_epi16::<2>(y_vec);
      let y_u8 = narrow_u8x64(y_shr, zero, pack_fixup);
      // Store first 24 of the u8x64 lanes via stack buffer + copy_from_slice.
      let mut tmp = [0u8; 64];
      _mm512_storeu_si512(tmp.as_mut_ptr().cast(), y_u8);
      luma_out[q * 24..q * 24 + 24].copy_from_slice(&tmp[..24]);
    }

    // Tail: any remaining 1, 2, or 3 words goes through scalar.
    if quads * 4 < words {
      let tail_start_px = quads * 24;
      let tail_packed = &packed[quads * 64..words * 16];
      let tail_out = &mut luma_out[tail_start_px..width];
      let tail_w = width - tail_start_px;
      scalar::v210_to_luma_row(tail_packed, tail_out, tail_w);
    }
  }
}

/// AVX-512 v210 → native-depth `u16` luma (low-bit-packed). Each output
/// `u16` carries the source's 10-bit Y value in its low 10 bits.
/// Byte-identical to `scalar::v210_to_luma_u16_row`.
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available.**
/// 2. `width % 6 == 0`.
/// 3. `packed.len() >= (width / 6) * 16`.
/// 4. `luma_out.len() >= width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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
    let quads = words / 4;
    for q in 0..quads {
      let (y_vec, _, _) = unpack_v210_4words_avx512(packed.as_ptr().add(q * 64));
      // Store first 24 of the 32 u16 lanes via stack buffer + copy_from_slice.
      let mut tmp = [0u16; 32];
      _mm512_storeu_si512(tmp.as_mut_ptr().cast(), y_vec);
      luma_out[q * 24..q * 24 + 24].copy_from_slice(&tmp[..24]);
    }

    // Tail: any remaining 1, 2, or 3 words goes through scalar.
    if quads * 4 < words {
      let tail_start_px = quads * 24;
      let tail_packed = &packed[quads * 64..words * 16];
      let tail_out = &mut luma_out[tail_start_px..width];
      let tail_w = width - tail_start_px;
      scalar::v210_to_luma_u16_row(tail_packed, tail_out, tail_w);
    }
  }
}
