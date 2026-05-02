//! AVX-512 VUYA / VUYX (packed YUV 4:4:4, 8-bit) kernels.
//!
//! ## Layout
//!
//! Four `u8` elements per pixel: `V(8) ‖ U(8) ‖ Y(8) ‖ A(8)`.
//! VUYA carries a real alpha channel in byte 3. VUYX treats byte 3 as
//! padding and forces output α to `0xFF`.
//!
//! ## Per-iter pipeline (64 px / iter)
//!
//! Four contiguous `_mm512_loadu_si512` loads fetch 256 bytes = 64
//! pixels of `V U Y A`. Each 512-bit register holds 16 pixels (4 in
//! each of its four 128-bit lanes). Per-128-bit-lane `_mm512_shuffle_epi8`
//! gathers each channel's 4 bytes from each lane into the lane's low
//! 32 bits (one `i32` lane); the upper 12 bytes of each lane are zeroed.
//!
//! After per-lane shuffle (e.g. for V on `raw_c0` covering pixels 0..15):
//! ```text
//!   v_c0 i32 lanes:
//!     [V0..V3, _, _, _, V4..V7, _, _, _, V8..V11, _, _, _, V12..V15, _, _, _]
//! ```
//! (lane 0 holds the 4-byte i32 packing `V0|V1|V2|V3`; lanes 1, 2, 3
//! are zero; lane 4 holds `V4|V5|V6|V7`; etc.) Two rounds of
//! `_mm512_permutex2var_epi32` (cross-vector i32 gather) then
//! consolidate the 16 valid i32 lanes scattered across the four
//! per-load partials into a single naturally-ordered 64-byte channel
//! vector:
//!
//! - Round 1: from `(v_c0, v_c1)` pick valid i32 lanes (0, 4, 8, 12) of
//!   each, producing 8 valid lanes covering V0..V31 in i32-lanes 0..7
//!   of the result. Same for `(v_c2, v_c3)` covering V32..V63.
//! - Round 2: combine the two half-results into a full 16-lane vector
//!   via `_mm512_permutex2var_epi32` with index `[0..7, 16..23]`.
//!
//! Net deinterleave: 4 loads + 16 shuffles + 12 permutes for all four
//! channels (each per-load shuffle is per-channel; channels are
//! independent and parallelizable). The cross-lane primitive used is
//! `vpermt2d` (i32) from AVX-512F — no AVX-512VBMI required (which
//! would be needed for `_mm512_permutex2var_epi8`).
//!
//! After deinterleave: zero-extend each 64-byte channel to two
//! `i16x32` halves via `_mm512_cvtepu8_epi16` on the low / high 256-bit
//! halves. The Q15 chroma + Y pipeline at BITS=8 is byte-identical to
//! the AVX-512 NV24 / packed YUV422 kernels — `chroma_i16x32`,
//! `scale_y`, `narrow_u8x64` from `mod.rs`.
//!
//! ## α handling
//!
//! When `ALPHA && ALPHA_SRC`, the A channel from the deinterleave is
//! passed straight through. When `ALPHA && !ALPHA_SRC`,
//! `_mm512_set1_epi8(-1)` (= 0xFF) is used. The `<false, false>`
//! monomorphization (RGB) drops the A channel entirely. The
//! `<false, true>` combination is rejected at monomorphization via
//! `const { assert! }`.
//!
//! ## Tail
//!
//! `width % 64` remaining pixels fall through to
//! `scalar::vuya_to_rgb_or_rgba_row`.
use core::arch::x86_64::*;

use super::*;
use crate::{ColorMatrix, row::scalar};

// ---- Static permute index tables ----------------------------------------
//
// VUYA layout per pixel: [V, U, Y, A] (4 u8 per pixel).
// 64 pixels occupy 256 bytes across 4 × __m512i loads:
//   raw_c0: bytes  0..63  → pixels  0..15
//   raw_c1: bytes 64..127 → pixels 16..31
//   raw_c2: bytes 128..191 → pixels 32..47
//   raw_c3: bytes 192..255 → pixels 48..63
//
// Within one __m512i, each of its 4 × 128-bit lanes holds 4 pixels' worth
// of VUYA (16 bytes). Per-128-bit-lane shuffle masks gather each channel's
// 4 bytes into the low 4 bytes of each lane (one `i32` lane in the low
// 32 bits of the lane); the upper 12 bytes of each lane are zeroed.
//
// After per-lane shuffle, the result holds the channel's bytes at i32
// lanes 0, 4, 8, 12 (one valid i32 per 128-bit lane). All other i32
// lanes are zero.
//
// Two rounds of `_mm512_permutex2var_epi32` then consolidate four
// such per-load partials into a single 16-lane (i32) channel vector
// holding 64 bytes in natural pixel order.

/// Round-1 index for `_mm512_permutex2var_epi32(a, idx, b)`: gather the
/// valid i32 lanes (0, 4, 8, 12) of `a` (covering 16 pixels of one
/// channel) and the valid i32 lanes (0, 4, 8, 12) of `b` (covering the
/// next 16 pixels) into the low 8 lanes of the result. Lanes 8..16 are
/// don't-care (use lane 0 as a safe index).
///
/// For `_mm512_permutex2var_epi32`:
///   `idx[i] < 16`  → output lane i = a[idx[i]]
///   `idx[i] >= 16` → output lane i = b[idx[i] - 16]
#[rustfmt::skip]
static GATHER_PAIR_IDX: [i32; 16] = [
  // From a (lanes 0..4 of output): 16 pixels of channel from one __m512i.
  0, 4, 8, 12,
  // From b (lanes 4..8 of output): 16 pixels from the next __m512i.
  16, 20, 24, 28,
  // Don't-care lanes 8..16: safe index 0.
  0, 0, 0, 0, 0, 0, 0, 0,
];

/// Round-2 index: combine two 8-i32-lane half-vectors into a full
/// 16-lane vector. Low 8 lanes from `a` (pixels 0..31), high 8 lanes
/// from `b` (pixels 32..63).
#[rustfmt::skip]
static COMBINE_IDX: [i32; 16] = [
  0, 1, 2, 3, 4, 5, 6, 7,
  16, 17, 18, 19, 20, 21, 22, 23,
];

// ---- Deinterleave helper ------------------------------------------------

/// Loads 64 VUYA quadruples (256 bytes = 64 pixels) from `ptr` and
/// unpacks them into four `__m512i` channel vectors holding 64 bytes
/// in natural pixel order (lane n = byte from pixel n):
/// - `v_vec`, `u_vec`, `y_vec`, `a_vec` — each 64 bytes, lane n = V/U/Y/A
///   from pixel n.
///
/// Strategy:
/// 1. 4 × `_mm512_loadu_si512` (one per 16-pixel group).
/// 2. Per-128-bit-lane `_mm512_shuffle_epi8` per channel, gathering each
///    channel's 4 bytes from each of the 4 × 128-bit lanes into the
///    low 4 bytes of each lane (one `i32` lane). Upper bytes zeroed.
/// 3. Two rounds of `_mm512_permutex2var_epi32` per channel: round 1
///    gathers the valid i32 lanes (0, 4, 8, 12) from each of two
///    consecutive `__m512i` partials into the low 8 lanes of an
///    intermediate; round 2 combines the two 8-lane halves into the
///    full 16-lane channel vector.
///
/// # Safety
///
/// `ptr` must point to at least 256 readable bytes (64 VUYA quadruples).
/// Caller's `target_feature` must include AVX-512F + AVX-512BW (BW
/// provides `_mm512_shuffle_epi8`; F provides `_mm512_permutex2var_epi32`).
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn deinterleave_vuya_avx512(ptr: *const u8) -> (__m512i, __m512i, __m512i, __m512i) {
  // SAFETY: caller obligation — `ptr` has 256 bytes readable; AVX-512F
  // + AVX-512BW are available.
  unsafe {
    // Load 4 × __m512i contiguously (64 pixels × 4 channels × u8 = 256 bytes).
    //
    // Each load covers 16 contiguous pixels (4 pixels per 128-bit lane):
    //   raw_c0 lanes: P0..3, P4..7, P8..11, P12..15
    //   raw_c1 lanes: P16..19, P20..23, P24..27, P28..31
    //   raw_c2 lanes: P32..35, P36..39, P40..43, P44..47
    //   raw_c3 lanes: P48..51, P52..55, P56..59, P60..63
    let raw_c0 = _mm512_loadu_si512(ptr.cast());
    let raw_c1 = _mm512_loadu_si512(ptr.add(64).cast());
    let raw_c2 = _mm512_loadu_si512(ptr.add(128).cast());
    let raw_c3 = _mm512_loadu_si512(ptr.add(192).cast());

    // Per-128-bit-lane shuffle masks: gather each channel's 4 bytes from
    // a 128-bit lane (4 pixels of VUYA = 16 bytes) into the low 4 bytes
    // of the lane. -1 zeroes the upper 12 bytes of each lane.
    //
    // `_mm512_broadcast_i32x4` replicates a 16-byte mask across all four
    // 128-bit lanes of the __m512i.
    let v_lane_mask = _mm_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let u_lane_mask = _mm_setr_epi8(1, 5, 9, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let y_lane_mask = _mm_setr_epi8(2, 6, 10, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let a_lane_mask = _mm_setr_epi8(3, 7, 11, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let v_mask = _mm512_broadcast_i32x4(v_lane_mask);
    let u_mask = _mm512_broadcast_i32x4(u_lane_mask);
    let y_mask = _mm512_broadcast_i32x4(y_lane_mask);
    let a_mask = _mm512_broadcast_i32x4(a_lane_mask);

    // Apply the per-channel masks to each load. Each result holds
    // the channel's 16 valid bytes spread across i32 lanes 0, 4, 8, 12.
    //
    // For V (analogous for U/Y/A), v_c0 (covering pixels 0..15) holds:
    //   i32-lane 0:  V0V1V2V3   (4 bytes, low 32 of lane 0)
    //   i32-lane 1:  zero
    //   i32-lane 2:  zero
    //   i32-lane 3:  zero
    //   i32-lane 4:  V4V5V6V7
    //   i32-lane 5:  zero
    //   i32-lane 6:  zero
    //   i32-lane 7:  zero
    //   i32-lane 8:  V8V9V10V11
    //   ...
    //   i32-lane 12: V12V13V14V15
    //   i32-lanes 13, 14, 15: zero
    let v_c0 = _mm512_shuffle_epi8(raw_c0, v_mask);
    let v_c1 = _mm512_shuffle_epi8(raw_c1, v_mask);
    let v_c2 = _mm512_shuffle_epi8(raw_c2, v_mask);
    let v_c3 = _mm512_shuffle_epi8(raw_c3, v_mask);
    let u_c0 = _mm512_shuffle_epi8(raw_c0, u_mask);
    let u_c1 = _mm512_shuffle_epi8(raw_c1, u_mask);
    let u_c2 = _mm512_shuffle_epi8(raw_c2, u_mask);
    let u_c3 = _mm512_shuffle_epi8(raw_c3, u_mask);
    let y_c0 = _mm512_shuffle_epi8(raw_c0, y_mask);
    let y_c1 = _mm512_shuffle_epi8(raw_c1, y_mask);
    let y_c2 = _mm512_shuffle_epi8(raw_c2, y_mask);
    let y_c3 = _mm512_shuffle_epi8(raw_c3, y_mask);
    let a_c0 = _mm512_shuffle_epi8(raw_c0, a_mask);
    let a_c1 = _mm512_shuffle_epi8(raw_c1, a_mask);
    let a_c2 = _mm512_shuffle_epi8(raw_c2, a_mask);
    let a_c3 = _mm512_shuffle_epi8(raw_c3, a_mask);

    // Permute index tables.
    let pair_idx = _mm512_loadu_si512(GATHER_PAIR_IDX.as_ptr().cast());
    let comb_idx = _mm512_loadu_si512(COMBINE_IDX.as_ptr().cast());

    // Round 1: gather valid i32 lanes (0, 4, 8, 12) from each pair of
    // partials. Result has 8 valid lanes (0..8) covering 32 pixels'
    // worth of channel bytes; lanes 8..16 are don't-care.
    //
    // For V: v_01 covers pixels 0..31 (lanes 0..8 = V0..V31 packed
    // 4 bytes per i32-lane); v_23 covers pixels 32..63.
    let v_01 = _mm512_permutex2var_epi32(v_c0, pair_idx, v_c1);
    let v_23 = _mm512_permutex2var_epi32(v_c2, pair_idx, v_c3);
    let u_01 = _mm512_permutex2var_epi32(u_c0, pair_idx, u_c1);
    let u_23 = _mm512_permutex2var_epi32(u_c2, pair_idx, u_c3);
    let y_01 = _mm512_permutex2var_epi32(y_c0, pair_idx, y_c1);
    let y_23 = _mm512_permutex2var_epi32(y_c2, pair_idx, y_c3);
    let a_01 = _mm512_permutex2var_epi32(a_c0, pair_idx, a_c1);
    let a_23 = _mm512_permutex2var_epi32(a_c2, pair_idx, a_c3);

    // Round 2: combine two 8-lane half-vectors into a full 16-lane
    // (i32) channel vector. `COMBINE_IDX` picks lanes 0..8 from `a` and
    // lanes 16..24 (= `b` lanes 0..8) for the high half.
    //
    // For V: v_vec is a __m512i with 64 bytes in natural pixel order:
    // byte n = V from pixel n.
    let v_vec = _mm512_permutex2var_epi32(v_01, comb_idx, v_23);
    let u_vec = _mm512_permutex2var_epi32(u_01, comb_idx, u_23);
    let y_vec = _mm512_permutex2var_epi32(y_01, comb_idx, y_23);
    let a_vec = _mm512_permutex2var_epi32(a_01, comb_idx, a_23);

    (v_vec, u_vec, y_vec, a_vec)
  }
}

// ---- Shared RGB / RGBA kernel (64 px/iter) ------------------------------

/// AVX-512 VUYA / VUYX → packed u8 RGB or RGBA.
///
/// Byte-identical to `scalar::vuya_to_rgb_or_rgba_row::<ALPHA, ALPHA_SRC>`.
///
/// Block size: 64 pixels per SIMD iteration (four `_mm512_loadu_si512`
/// loads, 256 bytes total).
///
/// The three valid monomorphizations are:
/// - `<false, false>` — RGB (drops α)
/// - `<true, true>`  — RGBA, source α pass-through (VUYA)
/// - `<true, false>` — RGBA, force α = `0xFF` (VUYX)
///
/// `<false, true>` is rejected at monomorphization via `const { assert! }`.
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available.**
/// 2. `packed.len() >= width * 4`.
/// 3. `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn vuya_to_rgb_or_rgba_row<const ALPHA: bool, const ALPHA_SRC: bool>(
  packed: &[u8],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // Source alpha requires RGBA output.
  const { assert!(!ALPHA_SRC || ALPHA) };
  debug_assert!(packed.len() >= width * 4, "packed row too short");
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp, "out row too short");

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<8, 8>(full_range);
  let bias = scalar::chroma_bias::<8>();
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
    // 0xFF for VUYX forced-opaque path.
    let alpha_u8 = _mm512_set1_epi8(-1i8);

    let mut x = 0usize;
    while x + 64 <= width {
      // Deinterleave 64 VUYA quadruples → V, U, Y, A as u8x64 in
      // natural pixel order.
      let (v_u8, u_u8, y_u8, a_u8) = deinterleave_vuya_avx512(packed.as_ptr().add(x * 4));

      // Zero-extend each channel to two i16x32 halves (low 32 bytes →
      // pixels 0..31, high 32 bytes → pixels 32..63).
      let v_lo_i16 = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(v_u8));
      let v_hi_i16 = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64::<1>(v_u8));
      let u_lo_i16 = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(u_u8));
      let u_hi_i16 = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64::<1>(u_u8));
      let y_lo_i16 = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(y_u8));
      let y_hi_i16 = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64::<1>(y_u8));

      // Subtract chroma bias (128 for 8-bit).
      let u_lo_sub = _mm512_sub_epi16(u_lo_i16, bias_v);
      let u_hi_sub = _mm512_sub_epi16(u_hi_i16, bias_v);
      let v_lo_sub = _mm512_sub_epi16(v_lo_i16, bias_v);
      let v_hi_sub = _mm512_sub_epi16(v_hi_i16, bias_v);

      // Widen each i16x32 chroma half into two i32x16 halves for Q15
      // multiply.
      let u_lo_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_lo_sub));
      let u_lo_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(u_lo_sub));
      let u_hi_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_hi_sub));
      let u_hi_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(u_hi_sub));
      let v_lo_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_lo_sub));
      let v_lo_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(v_lo_sub));
      let v_hi_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_hi_sub));
      let v_hi_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(v_hi_sub));

      // u_d / v_d = (u * c_scale + RND) >> 15.
      let u_d_lo_a = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_lo_a, c_scale_v),
        rnd_v,
      ));
      let u_d_lo_b = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_lo_b, c_scale_v),
        rnd_v,
      ));
      let u_d_hi_a = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_hi_a, c_scale_v),
        rnd_v,
      ));
      let u_d_hi_b = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_hi_b, c_scale_v),
        rnd_v,
      ));
      let v_d_lo_a = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_lo_a, c_scale_v),
        rnd_v,
      ));
      let v_d_lo_b = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_lo_b, c_scale_v),
        rnd_v,
      ));
      let v_d_hi_a = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_hi_a, c_scale_v),
        rnd_v,
      ));
      let v_d_hi_b = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_hi_b, c_scale_v),
        rnd_v,
      ));

      // 64 chroma per channel: two `chroma_i16x32` calls per channel
      // (no chroma duplication at 4:4:4 — one chroma sample per Y pixel).
      let r_chroma_lo = chroma_i16x32(
        cru, crv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v, pack_fixup,
      );
      let r_chroma_hi = chroma_i16x32(
        cru, crv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v, pack_fixup,
      );
      let g_chroma_lo = chroma_i16x32(
        cgu, cgv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v, pack_fixup,
      );
      let g_chroma_hi = chroma_i16x32(
        cgu, cgv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v, pack_fixup,
      );
      let b_chroma_lo = chroma_i16x32(
        cbu, cbv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v, pack_fixup,
      );
      let b_chroma_hi = chroma_i16x32(
        cbu, cbv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v, pack_fixup,
      );

      // Y path: scale each i16x32 half independently.
      let y_scaled_lo = scale_y(y_lo_i16, y_off_v, y_scale_v, rnd_v, pack_fixup);
      let y_scaled_hi = scale_y(y_hi_i16, y_off_v, y_scale_v, rnd_v, pack_fixup);

      // Saturating i16 add Y + chroma per channel, then narrow to u8x64
      // with natural lane order via `narrow_u8x64`.
      let r_lo = _mm512_adds_epi16(y_scaled_lo, r_chroma_lo);
      let r_hi = _mm512_adds_epi16(y_scaled_hi, r_chroma_hi);
      let g_lo = _mm512_adds_epi16(y_scaled_lo, g_chroma_lo);
      let g_hi = _mm512_adds_epi16(y_scaled_hi, g_chroma_hi);
      let b_lo = _mm512_adds_epi16(y_scaled_lo, b_chroma_lo);
      let b_hi = _mm512_adds_epi16(y_scaled_hi, b_chroma_hi);

      let r_u8 = narrow_u8x64(r_lo, r_hi, pack_fixup);
      let g_u8 = narrow_u8x64(g_lo, g_hi, pack_fixup);
      let b_u8 = narrow_u8x64(b_lo, b_hi, pack_fixup);

      let out_ptr = out.as_mut_ptr().add(x * bpp);
      if ALPHA {
        let a_vec = if ALPHA_SRC { a_u8 } else { alpha_u8 };
        write_rgba_64(r_u8, g_u8, b_u8, a_vec, out_ptr);
      } else {
        write_rgb_64(r_u8, g_u8, b_u8, out_ptr);
      }

      x += 64;
    }

    // Scalar tail — remaining < 64 pixels.
    if x < width {
      scalar::vuya_to_rgb_or_rgba_row::<ALPHA, ALPHA_SRC>(
        &packed[x * 4..],
        &mut out[x * bpp..],
        width - x,
        matrix,
        full_range,
      );
    }
  }
}

// ---- Thin wrappers ------------------------------------------------------

/// AVX-512 VUYA / VUYX → packed **RGB** (3 bpp). Alpha byte in source is
/// discarded — RGB output has no alpha channel.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn vuya_to_rgb_row(
  packed: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: AVX-512F + AVX-512BW availability is the caller's obligation.
  unsafe {
    vuya_to_rgb_or_rgba_row::<false, false>(packed, rgb_out, width, matrix, full_range);
  }
}

/// AVX-512 VUYA → packed **RGBA** (4 bpp). Source A byte is passed
/// through verbatim.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn vuya_to_rgba_row(
  packed: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: AVX-512F + AVX-512BW availability is the caller's obligation.
  unsafe {
    vuya_to_rgb_or_rgba_row::<true, true>(packed, rgba_out, width, matrix, full_range);
  }
}

/// AVX-512 VUYX → packed **RGBA** (4 bpp). Source A byte is padding;
/// output α is forced to `0xFF` (opaque).
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn vuyx_to_rgba_row(
  packed: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: AVX-512F + AVX-512BW availability is the caller's obligation.
  unsafe {
    vuya_to_rgb_or_rgba_row::<true, false>(packed, rgba_out, width, matrix, full_range);
  }
}

// ---- Luma extraction (64 px/iter) ---------------------------------------

/// AVX-512 VUYA / VUYX → u8 luma. Y is the third byte (offset 2) of each
/// pixel quadruple.
///
/// Byte-identical to `scalar::vuya_to_luma_row`.
///
/// Block size: 64 pixels per SIMD iteration. Reuses the full 4-channel
/// deinterleave and discards V/U/A: keeping the same code path gives the
/// lane-order regression test the strongest possible coverage — any
/// deinterleave bug in the V/U/Y path manifests identically here.
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available.**
/// 2. `packed.len() >= width * 4`.
/// 3. `luma_out.len() >= width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn vuya_to_luma_row(packed: &[u8], luma_out: &mut [u8], width: usize) {
  debug_assert!(packed.len() >= width * 4, "packed row too short");
  debug_assert!(luma_out.len() >= width, "luma row too short");

  // SAFETY: AVX-512F + AVX-512BW availability is the caller's obligation.
  unsafe {
    let mut x = 0usize;
    while x + 64 <= width {
      let (_v, _u, y_vec, _a) = deinterleave_vuya_avx512(packed.as_ptr().add(x * 4));
      _mm512_storeu_si512(luma_out.as_mut_ptr().add(x).cast(), y_vec);
      x += 64;
    }

    // Scalar tail — remaining < 64 pixels.
    if x < width {
      scalar::vuya_to_luma_row(&packed[x * 4..], &mut luma_out[x..], width - x);
    }
  }
}
