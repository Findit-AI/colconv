//! AVX-512 XV36 (packed YUV 4:4:4, 12-bit) kernels.
//!
//! ## Layout
//!
//! Four `u16` elements per pixel: `[U(16), Y(16), V(16), A(16)]`
//! little-endian, each holding a 12-bit sample MSB-aligned in the
//! high 12 bits (low 4 bits zero). The `X` prefix means the A slot
//! is **padding** — loaded but discarded. RGBA outputs force α = max
//! (`0xFF` u8 / `0x0FFF` u16).
//!
//! ## Per-iter pipeline (32 px / iter)
//!
//! Four `_mm512_loadu_si512` loads fetch 128 u16 lanes (32 pixels ×
//! 4 channels). Two rounds of `_mm512_permutex2var_epi16` (cross-vector
//! u16 gather) separate the four channels into 32-lane vectors:
//!
//! - Round 1: gather U+Y from each pair (v0,v1) and (v2,v3) using indices
//!   that pick even-slot lanes (0,4,8,...) and (1,5,9,...) respectively.
//! - Round 2: combine the two half-results into full 32-lane U, Y, V, A
//!   vectors via a second `_mm512_permutex2var_epi16`.
//!
//! After deinterleave, `_mm512_srli_epi16::<4>` drops the 4 MSB-alignment
//! padding LSBs from each channel, bringing samples into `[0, 4095]`.
//!
//! ## Q15 pipeline at BITS=12
//!
//! From there the pipeline mirrors the AVX-512 v410.rs / y2xx.rs path:
//! subtract chroma bias, Q15-scale chroma to `u_d` / `v_d` via
//! `chroma_i16x32` (i32 chroma — NOT i64), scale Y via `scale_y`
//! (Y ≤ 4095 fits in i16 — NOT `scale_y_u16_avx512`), sum + saturate.
//!
//! ## 4:4:4 vs. 4:2:2
//!
//! XV36 is 4:4:4 — no chroma duplication (`chroma_dup`) is needed.
//! All 32 lanes carry unique `(U, Y, V)` triples.
//!
//! ## Tail
//!
//! `width % 32` remaining pixels fall through to `scalar::xv36_*`.

use core::arch::x86_64::*;

use super::*;
use crate::{ColorMatrix, row::scalar};

// ---- Static permute index tables -----------------------------------------
//
// XV36 layout per pixel: [U, Y, V, A] (4 u16 per pixel).
// 32 pixels occupy 128 u16 elements across 4 × __m512i loads:
//   v0: u16 lanes [0..32)  → pixels  0.. 7 (each pixel = 4 u16 lanes)
//   v1: u16 lanes [0..32)  → pixels  8..15
//   v2: u16 lanes [0..32)  → pixels 16..23
//   v3: u16 lanes [0..32)  → pixels 24..31
//
// Within one __m512i, pixel p (0-based within the vector) occupies
// lanes [4p, 4p+1, 4p+2, 4p+3] = [U, Y, V, A].
//
// Strategy: two rounds of _mm512_permutex2var_epi16.
// For `_mm512_permutex2var_epi16(a, idx, b)`:
//   idx[i] < 32  → output lane i = a[idx[i]]
//   idx[i] >= 32 → output lane i = b[idx[i] - 32]
//
// Round 1 (pairs):
//   from (v0, v1): U for pixels 0..15 (u_01), Y for pixels 0..15 (y_01)
//   from (v2, v3): U for pixels 16..31 (u_23), Y for pixels 16..31 (y_23)
//   from (v0, v1): V for pixels 0..15 (v_01), A for pixels 0..15 (a_01) [unused]
//   from (v2, v3): V for pixels 16..31 (v_23), A for pixels 16..31 (a_23) [unused]
//
// Round 2 (combine):
//   U = permutex2(u_01, idx_combine_lo, u_23)  → lanes [U0..U31]
//   Y = permutex2(y_01, idx_combine_lo, y_23)  → lanes [Y0..Y31]
//   V = permutex2(v_01, idx_combine_lo, v_23)  → lanes [V0..V31]
//
// For Round 1, "U from (v0,v1)" picks lanes 0,4,8,12,16,20,24,28 from v0
// (pixels 0-7 of v0) and lanes 0,4,8,12,16,20,24,28 from v1 shifted by 32
// (pixels 0-7 of v1 = pixels 8-15 of the full sequence).
// That gives 16 U values in the low 16 lanes; lanes 16..32 are don't-care.

// Round-1 index: pick U (or V) channel from two consecutive __m512i vectors.
// Lane i picks: v0[4i] for i in 0..8 (pixels 0-7 from v0),
//               v1[4(i-8)] = v1[4i-32] via idx[i] >= 32 for i in 8..16,
//               don't-care for i in 16..32 (use lane 0 as safe index).
#[rustfmt::skip]
static UV_FROM_PAIR_IDX: [i16; 32] = [
  // U (or V, offset +2) from v0 (lanes 0..8 of output).
   0,  4,  8, 12, 16, 20, 24, 28,
  // U (or V, offset +2) from v1 (lanes 8..16 of output), idx >= 32.
  32, 36, 40, 44, 48, 52, 56, 60,
  // Don't-care lanes 16..32: safe index 0.
   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
];

// Round-1 index: pick Y channel (offset +1 within each pixel group of 4).
#[rustfmt::skip]
static Y_FROM_PAIR_IDX: [i16; 32] = [
  // Y from v0 (lanes 0..8 of output).
   1,  5,  9, 13, 17, 21, 25, 29,
  // Y from v1 (lanes 8..16 of output), idx >= 32.
  33, 37, 41, 45, 49, 53, 57, 61,
  // Don't-care lanes 16..32: safe index 1.
   1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
];

// Round-1 index: pick V channel (offset +2 within each pixel group of 4).
#[rustfmt::skip]
static V_FROM_PAIR_IDX: [i16; 32] = [
  // V from v0 (lanes 0..8 of output).
   2,  6, 10, 14, 18, 22, 26, 30,
  // V from v1 (lanes 8..16 of output), idx >= 32.
  34, 38, 42, 46, 50, 54, 58, 62,
  // Don't-care lanes 16..32: safe index 2.
   2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
];

// Round-2 index: combine two 16-value half-vectors into a full 32-lane vector.
// Low 16 lanes come from the first source (half-vector for pixels 0-15),
// high 16 lanes come from the second source (half-vector for pixels 16-31).
#[rustfmt::skip]
static COMBINE_IDX: [i16; 32] = [
   0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
  32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
];

// ---- Deinterleave helper -------------------------------------------------

/// Loads 32 XV36 quadruples (128 u16 = 256 bytes) from `ptr` and unpacks
/// them into three `__m512i` vectors holding 12-bit samples in their low
/// bits (each of the 32 lanes is an i16 in `[0, 4095]`):
/// - `u_vec`: lanes 0..32 = U0..U31.
/// - `y_vec`: lanes 0..32 = Y0..Y31.
/// - `v_vec`: lanes 0..32 = V0..V31.
///
/// Strategy:
/// - 4 × `_mm512_loadu_si512` (one per 8-pixel group).
/// - Round 1: two `_mm512_permutex2var_epi16` per channel to gather 16
///   values from each consecutive pair (v0,v1) and (v2,v3).
/// - Round 2: one `_mm512_permutex2var_epi16` per channel to combine the
///   two 16-value half-vectors into a full 32-lane channel vector.
/// - `_mm512_srli_epi16::<4>` per channel to drop MSB-alignment padding.
///
/// # Safety
///
/// `ptr` must point to at least 256 readable bytes (128 `u16` elements).
/// Caller's `target_feature` must include AVX-512F + AVX-512BW (BW provides
/// `vpermt2w` — the u16 cross-vector permute).
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn unpack_xv36_32px_avx512(ptr: *const u16) -> (__m512i, __m512i, __m512i) {
  // SAFETY: caller obligation — `ptr` has 256 bytes readable; AVX-512F +
  // AVX-512BW are available.
  unsafe {
    // Load 4 × __m512i (32 pixels × 4 u16 channels = 128 u16 = 256 bytes).
    let v0 = _mm512_loadu_si512(ptr.cast()); // pixels  0.. 7
    let v1 = _mm512_loadu_si512(ptr.add(32).cast()); // pixels  8..15
    let v2 = _mm512_loadu_si512(ptr.add(64).cast()); // pixels 16..23
    let v3 = _mm512_loadu_si512(ptr.add(96).cast()); // pixels 24..31

    // Load permute index tables.
    let uv_idx = _mm512_loadu_si512(UV_FROM_PAIR_IDX.as_ptr().cast());
    let y_idx = _mm512_loadu_si512(Y_FROM_PAIR_IDX.as_ptr().cast());
    let v_idx_tbl = _mm512_loadu_si512(V_FROM_PAIR_IDX.as_ptr().cast());
    let comb_idx = _mm512_loadu_si512(COMBINE_IDX.as_ptr().cast());

    // Round 1: gather U / Y / V from each pair of __m512i vectors.
    // Result has 16 valid lanes (0..16) and 16 don't-care lanes (16..32).
    let u_01 = _mm512_permutex2var_epi16(v0, uv_idx, v1); // U for pixels  0..15
    let u_23 = _mm512_permutex2var_epi16(v2, uv_idx, v3); // U for pixels 16..31
    let y_01 = _mm512_permutex2var_epi16(v0, y_idx, v1); // Y for pixels  0..15
    let y_23 = _mm512_permutex2var_epi16(v2, y_idx, v3); // Y for pixels 16..31
    let v_01 = _mm512_permutex2var_epi16(v0, v_idx_tbl, v1); // V for pixels  0..15
    let v_23 = _mm512_permutex2var_epi16(v2, v_idx_tbl, v3); // V for pixels 16..31

    // Round 2: combine the two 16-value half-vectors into 32-lane channel
    // vectors. `COMBINE_IDX` picks lanes 0..16 from the first source and
    // lanes 32..48 (= second source lanes 0..16) for the high half.
    let u_raw = _mm512_permutex2var_epi16(u_01, comb_idx, u_23);
    let y_raw = _mm512_permutex2var_epi16(y_01, comb_idx, y_23);
    let v_raw = _mm512_permutex2var_epi16(v_01, comb_idx, v_23);

    // Drop 4 MSB-alignment padding LSBs → 12-bit values in [0, 4095].
    let u_vec = _mm512_srli_epi16::<4>(u_raw);
    let y_vec = _mm512_srli_epi16::<4>(y_raw);
    let v_vec = _mm512_srli_epi16::<4>(v_raw);

    (u_vec, y_vec, v_vec)
  }
}

// ---- u8 RGB / RGBA output (32 px/iter) ----------------------------------

/// AVX-512 XV36 → packed u8 RGB or RGBA.
///
/// Byte-identical to `scalar::xv36_to_rgb_or_rgba_row::<ALPHA>`.
///
/// Block size: 32 pixels per SIMD iteration (four `_mm512_loadu_si512`).
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available.**
/// 2. `packed.len() >= width * 4`.
/// 3. `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn xv36_to_rgb_or_rgba_row<const ALPHA: bool>(
  packed: &[u16],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(packed.len() >= width * 4, "packed row too short");
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp, "out row too short");

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<12, 8>(full_range);
  let bias = scalar::chroma_bias::<12>();
  const RND: i32 = 1 << 14;

  // SAFETY: AVX-512F + AVX-512BW availability is the caller's obligation.
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

    let mut x = 0usize;
    while x + 32 <= width {
      // Deinterleave 32 XV36 quadruples → U, Y, V as i16x32 in [0, 4095].
      let (u_u16, y_u16, v_u16) = unpack_xv36_32px_avx512(packed.as_ptr().add(x * 4));

      // Values ≤ 4095 < 32767 — safe to treat as signed i16.
      let u_i16 = u_u16;
      let y_i16 = y_u16;
      let v_i16 = v_u16;

      // Subtract chroma bias (2048 for 12-bit).
      let u_sub = _mm512_sub_epi16(u_i16, bias_v);
      let v_sub = _mm512_sub_epi16(v_i16, bias_v);

      // Widen to i32x16 lo/hi halves for Q15 chroma-scale multiply.
      let u_lo_i32 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_sub));
      let u_hi_i32 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(u_sub));
      let v_lo_i32 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_sub));
      let v_hi_i32 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(v_sub));

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

      // 4:4:4 — no chroma duplication; all 32 lanes carry unique U/V.
      // chroma_i16x32 uses i32 arithmetic (sufficient for 12-bit samples).
      let r_chroma = chroma_i16x32(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v, pack_fixup);
      let g_chroma = chroma_i16x32(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v, pack_fixup);
      let b_chroma = chroma_i16x32(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v, pack_fixup);

      // XV36 Y ≤ 4095 fits i16 — use scale_y (NOT scale_y_u16_avx512).
      let y_scaled = scale_y(y_i16, y_off_v, y_scale_v, rnd_v, pack_fixup);

      // u8 narrow with saturation. All 32 lanes carry valid results.
      let zero = _mm512_setzero_si512();
      let r_u8 = narrow_u8x64(_mm512_adds_epi16(y_scaled, r_chroma), zero, pack_fixup);
      let g_u8 = narrow_u8x64(_mm512_adds_epi16(y_scaled, g_chroma), zero, pack_fixup);
      let b_u8 = narrow_u8x64(_mm512_adds_epi16(y_scaled, b_chroma), zero, pack_fixup);

      // 32-pixel store via two write_rgb_16 / write_rgba_16 calls
      // (each writes 16 px = 48 / 64 bytes via 128-bit quarter extract).
      if ALPHA {
        let alpha = _mm_set1_epi8(-1i8);
        let r0 = _mm512_castsi512_si128(r_u8);
        let r1 = _mm512_extracti32x4_epi32::<1>(r_u8);
        let g0 = _mm512_castsi512_si128(g_u8);
        let g1 = _mm512_extracti32x4_epi32::<1>(g_u8);
        let b0 = _mm512_castsi512_si128(b_u8);
        let b1 = _mm512_extracti32x4_epi32::<1>(b_u8);
        let dst = out.as_mut_ptr().add(x * 4);
        write_rgba_16(r0, g0, b0, alpha, dst);
        write_rgba_16(r1, g1, b1, alpha, dst.add(64));
      } else {
        let r0 = _mm512_castsi512_si128(r_u8);
        let r1 = _mm512_extracti32x4_epi32::<1>(r_u8);
        let g0 = _mm512_castsi512_si128(g_u8);
        let g1 = _mm512_extracti32x4_epi32::<1>(g_u8);
        let b0 = _mm512_castsi512_si128(b_u8);
        let b1 = _mm512_extracti32x4_epi32::<1>(b_u8);
        let dst = out.as_mut_ptr().add(x * 3);
        write_rgb_16(r0, g0, b0, dst);
        write_rgb_16(r1, g1, b1, dst.add(48));
      }

      x += 32;
    }

    // Scalar tail — remaining < 32 pixels.
    if x < width {
      let tail_packed = &packed[x * 4..width * 4];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      scalar::xv36_to_rgb_or_rgba_row::<ALPHA>(tail_packed, tail_out, tail_w, matrix, full_range);
    }
  }
}

// ---- u16 RGB / RGBA native-depth output (32 px/iter) --------------------

/// AVX-512 XV36 → packed native-depth u16 RGB or RGBA (low-bit-packed at
/// 12-bit).
///
/// Byte-identical to `scalar::xv36_to_rgb_u16_or_rgba_u16_row::<ALPHA>`.
///
/// Block size: 32 pixels per SIMD iteration.
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available.**
/// 2. `packed.len() >= width * 4`.
/// 3. `out.len() >= width * (if ALPHA { 4 } else { 3 })` (u16 elements).
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn xv36_to_rgb_u16_or_rgba_u16_row<const ALPHA: bool>(
  packed: &[u16],
  out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(packed.len() >= width * 4, "packed row too short");
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp, "out row too short");

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<12, 12>(full_range);
  let bias = scalar::chroma_bias::<12>();
  const RND: i32 = 1 << 14;
  // 12-bit output max (low-bit-packed): [0, 0x0FFF].
  let out_max: i16 = 0x0FFF;
  let alpha_u16: u16 = 0x0FFF;

  // SAFETY: AVX-512F + AVX-512BW availability is the caller's obligation.
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

    let mut x = 0usize;
    while x + 32 <= width {
      let (u_u16, y_u16, v_u16) = unpack_xv36_32px_avx512(packed.as_ptr().add(x * 4));

      let u_i16 = u_u16;
      let y_i16 = y_u16;
      let v_i16 = v_u16;

      let u_sub = _mm512_sub_epi16(u_i16, bias_v);
      let v_sub = _mm512_sub_epi16(v_i16, bias_v);

      let u_lo_i32 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_sub));
      let u_hi_i32 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(u_sub));
      let v_lo_i32 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_sub));
      let v_hi_i32 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(v_sub));

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

      // 12-bit chroma: i32 arithmetic is sufficient (no overflow at 12-bit).
      let r_chroma = chroma_i16x32(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v, pack_fixup);
      let g_chroma = chroma_i16x32(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v, pack_fixup);
      let b_chroma = chroma_i16x32(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v, pack_fixup);

      // XV36 Y ≤ 4095 fits i16 — use scale_y (NOT scale_y_u16_avx512).
      let y_scaled = scale_y(y_i16, y_off_v, y_scale_v, rnd_v, pack_fixup);

      // Clamp to [0, 0x0FFF] (12-bit low-bit-packed output range).
      let r = clamp_u16_max_x32(_mm512_adds_epi16(y_scaled, r_chroma), zero_v, max_v);
      let g = clamp_u16_max_x32(_mm512_adds_epi16(y_scaled, g_chroma), zero_v, max_v);
      let b = clamp_u16_max_x32(_mm512_adds_epi16(y_scaled, b_chroma), zero_v, max_v);

      // 32-pixel u16 store via write_rgb_u16_32 / write_rgba_u16_32.
      if ALPHA {
        let alpha_v = _mm_set1_epi16(out_max);
        write_rgba_u16_32(r, g, b, alpha_v, out.as_mut_ptr().add(x * 4));
        let _ = alpha_u16; // suppress unused warning
      } else {
        write_rgb_u16_32(r, g, b, out.as_mut_ptr().add(x * 3));
      }

      x += 32;
    }

    // Scalar tail — remaining < 32 pixels.
    if x < width {
      let tail_packed = &packed[x * 4..width * 4];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      scalar::xv36_to_rgb_u16_or_rgba_u16_row::<ALPHA>(
        tail_packed,
        tail_out,
        tail_w,
        matrix,
        full_range,
      );
    }
  }
}

// ---- Luma u8 (32 px/iter) -----------------------------------------------

/// AVX-512 XV36 → u8 luma. Y is quadruple element 1 (offset 1 in each
/// group of 4 u16). The deinterleave yields Y in [0, 4095] (>> 4 already
/// applied); one more `>> 4` gives 8-bit (same as scalar
/// `packed[x*4+1] >> 8`).
///
/// Byte-identical to `scalar::xv36_to_luma_row`.
///
/// Block size: 32 pixels per SIMD iteration.
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available.**
/// 2. `packed.len() >= width * 4`.
/// 3. `out.len() >= width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn xv36_to_luma_row(packed: &[u16], out: &mut [u8], width: usize) {
  debug_assert!(packed.len() >= width * 4);
  debug_assert!(out.len() >= width);

  // SAFETY: AVX-512F + AVX-512BW availability is the caller's obligation.
  unsafe {
    let pack_fixup = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
    let zero = _mm512_setzero_si512();

    let mut x = 0usize;
    while x + 32 <= width {
      let (_u_vec, y_vec, _v_vec) = unpack_xv36_32px_avx512(packed.as_ptr().add(x * 4));

      // y_vec is already >> 4 (values in [0, 4095]).
      // Scalar does `packed[x*4+1] >> 8` — that is MSB-aligned >> 4 to get
      // 12-bit, then >> 4 more to get 8-bit. Apply one more >> 4.
      let y_shr = _mm512_srli_epi16::<4>(y_vec);

      // Narrow to u8 (values ≤ 255). All 32 lanes valid.
      let y_u8 = narrow_u8x64(y_shr, zero, pack_fixup);

      // Store 32 valid bytes via the low 256-bit half.
      _mm256_storeu_si256(out.as_mut_ptr().add(x).cast(), _mm512_castsi512_si256(y_u8));

      x += 32;
    }

    // Scalar tail — remaining < 32 pixels.
    if x < width {
      scalar::xv36_to_luma_row(&packed[x * 4..width * 4], &mut out[x..width], width - x);
    }
  }
}

// ---- Luma u16 (32 px/iter) -----------------------------------------------

/// AVX-512 XV36 → u16 luma (low-bit-packed at 12-bit). Y is quadruple
/// element 1; `>> 4` (already applied by `unpack_xv36_32px_avx512`) drops
/// the 4 padding LSBs to give a 12-bit value in `[0, 4095]`.
///
/// Byte-identical to `scalar::xv36_to_luma_u16_row`.
///
/// Block size: 32 pixels per SIMD iteration.
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available.**
/// 2. `packed.len() >= width * 4`.
/// 3. `out.len() >= width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn xv36_to_luma_u16_row(packed: &[u16], out: &mut [u16], width: usize) {
  debug_assert!(packed.len() >= width * 4);
  debug_assert!(out.len() >= width);

  // SAFETY: AVX-512F + AVX-512BW availability is the caller's obligation.
  unsafe {
    let mut x = 0usize;
    while x + 32 <= width {
      let (_u_vec, y_vec, _v_vec) = unpack_xv36_32px_avx512(packed.as_ptr().add(x * 4));

      // y_vec already has >> 4 applied (= 12-bit value in [0, 4095]).
      // Direct store of 32 × u16.
      _mm512_storeu_si512(out.as_mut_ptr().add(x).cast(), y_vec);

      x += 32;
    }

    // Scalar tail — remaining < 32 pixels.
    if x < width {
      scalar::xv36_to_luma_u16_row(&packed[x * 4..width * 4], &mut out[x..width], width - x);
    }
  }
}
