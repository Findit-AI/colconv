//! AVX-512 kernels for AYUV64 packed YUV 4:4:4 16-bit family
//! (FFmpeg `AV_PIX_FMT_AYUV64LE`).
//!
//! ## Layout
//!
//! Four `u16` elements per pixel: `A(16) ‖ Y(16) ‖ U(16) ‖ V(16)`.
//! All channels are 16-bit native — no padding bits, no right-shift on
//! load. Channel slot order at deinterleave output: **A=0, Y=1, U=2,
//! V=3** (differs from XV36's U/Y/V/A).
//!
//! ## Per-iter pipeline (64 px / iter for u8, 32 px / iter for u16)
//!
//! Both paths use the same 32-pixel deinterleave helper. The u8 path
//! runs the helper twice per main-loop iteration (lo half = pixels
//! 0..31, hi half = pixels 32..63) and then narrows to u8x64. The u16
//! path runs the helper once and emits 32 pixels of u16 RGBA/RGB.
//!
//! Per 32-pixel deinterleave, four `_mm512_loadu_si512` loads fetch
//! 128 u16 = 256 bytes (32 pixels of 4-channel u16). Two rounds of
//! `_mm512_permutex2var_epi16` (cross-vector u16 gather, AVX-512BW
//! `vpermt2w`) separate the four channels into 32-lane vectors:
//!
//! - Round 1: gather A/Y/U/V from each pair (v0,v1) and (v2,v3) using
//!   indices that pick lanes (0,4,8,...) for A, (1,5,9,...) for Y, etc.
//!   Result has 16 valid lanes (lanes 0..16) and 16 don't-care lanes.
//! - Round 2: combine the two 16-lane half-vectors into a full 32-lane
//!   channel vector via a second `_mm512_permutex2var_epi16`.
//!
//! Cross-lane primitive is `vpermt2w` (u16) from AVX-512BW — no
//! AVX-512VBMI required. The xv36 backend uses the same pattern
//! (Ship 12b canonical template).
//!
//! ```text
//! After 4 contiguous loads (one per 8-pixel group):
//!   v0 lanes [0..32) = pixel slots: P0..P7   each [A, Y, U, V]
//!   v1 lanes [0..32) = pixel slots: P8..P15
//!   v2 lanes [0..32) = pixel slots: P16..P23
//!   v3 lanes [0..32) = pixel slots: P24..P31
//!
//! Round 1 (per channel, gather lanes [0,4,8,12,...] from pair):
//!   a_01 lanes [0..16)  = A0..A15  (lanes 16..32 don't-care)
//!   a_23 lanes [0..16)  = A16..A31 (lanes 16..32 don't-care)
//!
//! Round 2 (combine half-vectors):
//!   a_vec lanes [0..32) = A0..A31  (natural pixel order)
//!   y_vec lanes [0..32) = Y0..Y31
//!   u_vec lanes [0..32) = U0..U31
//!   v_vec lanes [0..32) = V0..V31
//! ```
//!
//! ## u8 pipeline (64 px / iter)
//!
//! Two halves × 32-pixel deinterleaves. Per half: chroma centered
//! (subtract 32768 via wrapping `-32768i16` trick), Q15 chroma scale via
//! `chroma_i16x32` (i32 widening — no overflow at BITS=16/8). Y scaled
//! via `scale_y_u16_avx512` (unsigned-widened to avoid sign-bit
//! corruption for Y > 32767). Saturating add Y + chroma → narrow to
//! u8x64 via `narrow_u8x64`. Source α: `_mm512_srli_epi16::<8>` (high
//! byte) + `_mm512_packus_epi16` to depth-convert u16 → u8.
//!
//! ## u16 pipeline (32 px / iter)
//!
//! i64 chroma via `chroma_i64x8_avx512` to avoid i32 overflow at
//! BITS=16/16. Y scaled via `scale_y_i32x16_i64`. Per pixel: even/odd
//! i64x8 halves → reassembled to i32x16 → saturating-narrow to u16 via
//! `_mm512_packus_epi32` + `pack_fixup`. Source α: deinterleaved A
//! vector written direct (no conversion).
//!
//! ## Tail
//!
//! `width % block_size` remaining pixels fall through to
//! `scalar::ayuv64_to_rgb_or_rgba_row::<ALPHA, ALPHA_SRC>` (or u16
//! variant).

use core::arch::x86_64::*;

use super::*;
use crate::{ColorMatrix, row::scalar};

// ---- Static permute index tables ----------------------------------------
//
// AYUV64 layout per pixel: [A, Y, U, V] (4 u16 per pixel).
// 32 pixels occupy 128 u16 elements across 4 × __m512i loads:
//   v0: u16 lanes [0..32)  → pixels  0.. 7 (each pixel = 4 u16 lanes)
//   v1: u16 lanes [0..32)  → pixels  8..15
//   v2: u16 lanes [0..32)  → pixels 16..23
//   v3: u16 lanes [0..32)  → pixels 24..31
//
// Within one __m512i, pixel p (0-based within the vector) occupies
// lanes [4p, 4p+1, 4p+2, 4p+3] = [A, Y, U, V].
//
// Strategy: two rounds of _mm512_permutex2var_epi16.
// For `_mm512_permutex2var_epi16(a, idx, b)`:
//   idx[i] < 32  → output lane i = a[idx[i]]
//   idx[i] >= 32 → output lane i = b[idx[i] - 32]
//
// Round 1 (pairs):
//   from (v0, v1): A for pixels 0..15 (a_01), Y for pixels 0..15 (y_01),
//                  U for pixels 0..15 (u_01), V for pixels 0..15 (v_01)
//   from (v2, v3): A for pixels 16..31 (a_23), Y for pixels 16..31 (y_23),
//                  U for pixels 16..31 (u_23), V for pixels 16..31 (v_23)
//
// Round 2 (combine):
//   A = permutex2(a_01, idx_combine, a_23)  → lanes [A0..A31]
//   Y = permutex2(y_01, idx_combine, y_23)  → lanes [Y0..Y31]
//   U = permutex2(u_01, idx_combine, u_23)  → lanes [U0..U31]
//   V = permutex2(v_01, idx_combine, v_23)  → lanes [V0..V31]

// Round-1 index: pick A channel (offset 0 within each pixel group of 4).
// Lane i picks: v0[4i] for i in 0..8 (pixels 0-7 from v0),
//               v1[4(i-8)] = v1[4i-32] via idx[i] >= 32 for i in 8..16,
//               don't-care for i in 16..32 (use lane 0 as safe index).
#[rustfmt::skip]
static A_FROM_PAIR_IDX: [i16; 32] = [
  // A from v0 (lanes 0..8 of output).
   0,  4,  8, 12, 16, 20, 24, 28,
  // A from v1 (lanes 8..16 of output), idx >= 32.
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

// Round-1 index: pick U channel (offset +2 within each pixel group of 4).
#[rustfmt::skip]
static U_FROM_PAIR_IDX: [i16; 32] = [
  // U from v0 (lanes 0..8 of output).
   2,  6, 10, 14, 18, 22, 26, 30,
  // U from v1 (lanes 8..16 of output), idx >= 32.
  34, 38, 42, 46, 50, 54, 58, 62,
  // Don't-care lanes 16..32: safe index 2.
   2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
];

// Round-1 index: pick V channel (offset +3 within each pixel group of 4).
#[rustfmt::skip]
static V_FROM_PAIR_IDX: [i16; 32] = [
  // V from v0 (lanes 0..8 of output).
   3,  7, 11, 15, 19, 23, 27, 31,
  // V from v1 (lanes 8..16 of output), idx >= 32.
  35, 39, 43, 47, 51, 55, 59, 63,
  // Don't-care lanes 16..32: safe index 3.
   3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
];

// Round-2 index: combine two 16-value half-vectors into a full 32-lane vector.
// Low 16 lanes come from the first source (half-vector for pixels 0-15),
// high 16 lanes come from the second source (half-vector for pixels 16-31).
#[rustfmt::skip]
static COMBINE_IDX: [i16; 32] = [
   0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
  32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
];

// ---- Deinterleave helper (32 pixels / 128 u16 / 256 bytes) --------------

/// Deinterleaves 32 AYUV64 quadruples (128 u16 = 256 bytes) from `ptr`
/// into `(a_vec, y_vec, u_vec, v_vec)` — four `__m512i` vectors each
/// holding 32 `u16` samples in **natural pixel order** (lane n = u16
/// from pixel n).
///
/// Channel slot order in source: A=0, Y=1, U=2, V=3 (AYUV64 native).
/// No shift is applied (16-bit native samples).
///
/// ## Strategy (xv36 pattern)
///
/// 1. Four contiguous `_mm512_loadu_si512` loads fetch 32 pixels'
///    worth of A/Y/U/V (256 bytes).
/// 2. Per channel: two `_mm512_permutex2var_epi16` (`vpermt2w`) gather
///    16 valid lanes (each) from each consecutive pair (v0,v1) and
///    (v2,v3) into the low 16 lanes of an intermediate vector.
/// 3. Per channel: one `_mm512_permutex2var_epi16` combines the two
///    16-value half-vectors into the full 32-lane channel vector.
///
/// Cross-lane primitive `vpermt2w` is part of AVX-512BW — no
/// AVX-512VBMI required.
///
/// # Safety
///
/// `ptr` must point to at least 256 readable bytes (128 `u16`
/// elements). Caller's `target_feature` must include AVX-512F +
/// AVX-512BW (BW provides `vpermt2w`).
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn deinterleave_ayuv64_32px_avx512(ptr: *const u16) -> (__m512i, __m512i, __m512i, __m512i) {
  // SAFETY: caller obligation — `ptr` has 256 bytes readable; AVX-512F +
  // AVX-512BW are available.
  unsafe {
    // Load 4 × __m512i (32 pixels × 4 u16 channels = 128 u16 = 256 bytes).
    //
    // Each load covers 8 contiguous pixels (4 u16 channels × 8 = 32 u16 = 64 bytes):
    //   v0 lanes: A0,Y0,U0,V0,...,A7,Y7,U7,V7    (pixels  0.. 7)
    //   v1 lanes: A8..V8,...,A15..V15            (pixels  8..15)
    //   v2 lanes: A16..V16,...,A23..V23          (pixels 16..23)
    //   v3 lanes: A24..V24,...,A31..V31          (pixels 24..31)
    let v0 = _mm512_loadu_si512(ptr.cast());
    let v1 = _mm512_loadu_si512(ptr.add(32).cast());
    let v2 = _mm512_loadu_si512(ptr.add(64).cast());
    let v3 = _mm512_loadu_si512(ptr.add(96).cast());

    // Load permute index tables.
    let a_idx = _mm512_loadu_si512(A_FROM_PAIR_IDX.as_ptr().cast());
    let y_idx = _mm512_loadu_si512(Y_FROM_PAIR_IDX.as_ptr().cast());
    let u_idx = _mm512_loadu_si512(U_FROM_PAIR_IDX.as_ptr().cast());
    let v_idx_tbl = _mm512_loadu_si512(V_FROM_PAIR_IDX.as_ptr().cast());
    let comb_idx = _mm512_loadu_si512(COMBINE_IDX.as_ptr().cast());

    // Round 1: gather A / Y / U / V from each pair of __m512i vectors.
    // Result has 16 valid lanes (0..16) and 16 don't-care lanes (16..32).
    let a_01 = _mm512_permutex2var_epi16(v0, a_idx, v1); // A for pixels  0..15
    let a_23 = _mm512_permutex2var_epi16(v2, a_idx, v3); // A for pixels 16..31
    let y_01 = _mm512_permutex2var_epi16(v0, y_idx, v1); // Y for pixels  0..15
    let y_23 = _mm512_permutex2var_epi16(v2, y_idx, v3); // Y for pixels 16..31
    let u_01 = _mm512_permutex2var_epi16(v0, u_idx, v1); // U for pixels  0..15
    let u_23 = _mm512_permutex2var_epi16(v2, u_idx, v3); // U for pixels 16..31
    let v_01 = _mm512_permutex2var_epi16(v0, v_idx_tbl, v1); // V for pixels  0..15
    let v_23 = _mm512_permutex2var_epi16(v2, v_idx_tbl, v3); // V for pixels 16..31

    // Round 2: combine the two 16-value half-vectors into 32-lane channel
    // vectors. `COMBINE_IDX` picks lanes 0..16 from the first source and
    // lanes 32..48 (= second source lanes 0..16) for the high half.
    let a_vec = _mm512_permutex2var_epi16(a_01, comb_idx, a_23);
    let y_vec = _mm512_permutex2var_epi16(y_01, comb_idx, y_23);
    let u_vec = _mm512_permutex2var_epi16(u_01, comb_idx, u_23);
    let v_vec = _mm512_permutex2var_epi16(v_01, comb_idx, v_23);

    (a_vec, y_vec, u_vec, v_vec)
  }
}

// ---- u8 RGB / RGBA output (64 px/iter) ----------------------------------

/// AVX-512 AYUV64 → packed u8 RGB or RGBA.
///
/// Byte-identical to `scalar::ayuv64_to_rgb_or_rgba_row::<ALPHA, ALPHA_SRC>`.
///
/// Block size: 64 pixels per SIMD iteration (two 32-pixel deinterleaves).
///
/// Valid monomorphizations:
/// - `<false, false>` — RGB (α dropped)
/// - `<true, true>`  — RGBA, source α depth-converted u16 → u8 (`>> 8`)
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
pub(crate) unsafe fn ayuv64_to_rgb_or_rgba_row<const ALPHA: bool, const ALPHA_SRC: bool>(
  packed: &[u16],
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
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<16, 8>(full_range);
  const RND: i32 = 1 << 14;

  // SAFETY: AVX-512F + AVX-512BW availability is the caller's obligation.
  unsafe {
    let rnd_v = _mm512_set1_epi32(RND);
    // Y values are full u16 (0..65535); use i32 y_off for scale_y_u16_avx512.
    let y_off_v = _mm512_set1_epi32(y_off);
    let y_scale_v = _mm512_set1_epi32(y_scale);
    let c_scale_v = _mm512_set1_epi32(c_scale);
    // Subtract chroma bias (32768) via wrapping i16 trick: -32768i16 == 0x8000.
    let bias16_v = _mm512_set1_epi16(-32768i16);
    let cru = _mm512_set1_epi32(coeffs.r_u());
    let crv = _mm512_set1_epi32(coeffs.r_v());
    let cgu = _mm512_set1_epi32(coeffs.g_u());
    let cgv = _mm512_set1_epi32(coeffs.g_v());
    let cbu = _mm512_set1_epi32(coeffs.b_u());
    let cbv = _mm512_set1_epi32(coeffs.b_v());
    let pack_fixup = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
    // 0xFF for the (theoretical) opaque path — not emitted by valid
    // monomorphizations, but kept for symmetry with the AVX2 sibling.
    let alpha_u8 = _mm512_set1_epi8(-1i8);

    let mut x = 0usize;
    while x + 64 <= width {
      // --- lo half: pixels x..x+31 (one 32-pixel deinterleave) ----------
      let (a_lo_u16, y_lo_u16, u_lo_u16, v_lo_u16) =
        deinterleave_ayuv64_32px_avx512(packed.as_ptr().add(x * 4));

      // Center chroma: subtract 32768 via wrapping i16 (-32768i16 == 0x8000).
      let u_lo_i16 = _mm512_sub_epi16(u_lo_u16, bias16_v);
      let v_lo_i16 = _mm512_sub_epi16(v_lo_u16, bias16_v);

      // Widen each i16x32 chroma into two i32x16 halves for Q15 multiply.
      let u_lo_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_lo_i16));
      let u_lo_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(u_lo_i16));
      let v_lo_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_lo_i16));
      let v_lo_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(v_lo_i16));

      let u_d_lo_a = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_lo_a, c_scale_v),
        rnd_v,
      ));
      let u_d_lo_b = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_lo_b, c_scale_v),
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

      // 4:4:4 — no chroma duplication; one chroma sample per Y pixel.
      let r_chroma_lo = chroma_i16x32(
        cru, crv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v, pack_fixup,
      );
      let g_chroma_lo = chroma_i16x32(
        cgu, cgv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v, pack_fixup,
      );
      let b_chroma_lo = chroma_i16x32(
        cbu, cbv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v, pack_fixup,
      );

      // Y: full u16 values → use scale_y_u16_avx512 (NOT scale_y, which
      // would corrupt Y > 32767 by treating as signed).
      let y_lo_scaled = scale_y_u16_avx512(y_lo_u16, y_off_v, y_scale_v, rnd_v, pack_fixup);

      // --- hi half: pixels x+32..x+63 (one more 32-pixel deinterleave) --
      let (a_hi_u16, y_hi_u16, u_hi_u16, v_hi_u16) =
        deinterleave_ayuv64_32px_avx512(packed.as_ptr().add(x * 4 + 128));

      let u_hi_i16 = _mm512_sub_epi16(u_hi_u16, bias16_v);
      let v_hi_i16 = _mm512_sub_epi16(v_hi_u16, bias16_v);

      let u_hi_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_hi_i16));
      let u_hi_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(u_hi_i16));
      let v_hi_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_hi_i16));
      let v_hi_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(v_hi_i16));

      let u_d_hi_a = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_hi_a, c_scale_v),
        rnd_v,
      ));
      let u_d_hi_b = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_hi_b, c_scale_v),
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

      let r_chroma_hi = chroma_i16x32(
        cru, crv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v, pack_fixup,
      );
      let g_chroma_hi = chroma_i16x32(
        cgu, cgv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v, pack_fixup,
      );
      let b_chroma_hi = chroma_i16x32(
        cbu, cbv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v, pack_fixup,
      );

      let y_hi_scaled = scale_y_u16_avx512(y_hi_u16, y_off_v, y_scale_v, rnd_v, pack_fixup);

      // Saturating add Y + chroma per channel; narrow both halves into
      // u8x64 with natural lane order via `narrow_u8x64`.
      let r_u8 = narrow_u8x64(
        _mm512_adds_epi16(y_lo_scaled, r_chroma_lo),
        _mm512_adds_epi16(y_hi_scaled, r_chroma_hi),
        pack_fixup,
      );
      let g_u8 = narrow_u8x64(
        _mm512_adds_epi16(y_lo_scaled, g_chroma_lo),
        _mm512_adds_epi16(y_hi_scaled, g_chroma_hi),
        pack_fixup,
      );
      let b_u8 = narrow_u8x64(
        _mm512_adds_epi16(y_lo_scaled, b_chroma_lo),
        _mm512_adds_epi16(y_hi_scaled, b_chroma_hi),
        pack_fixup,
      );

      let out_ptr = out.as_mut_ptr().add(x * bpp);
      if ALPHA {
        // Source α: depth-convert u16 → u8 via >> 8 (high byte).
        // _mm512_srli_epi16::<8> shifts each u16 right by 8, putting the
        // high byte into the low 8 bits of each 16-bit lane. The lo/hi
        // halves are then narrowed via narrow_u8x64 (which already
        // applies the pack_fixup permute).
        let a_vec: __m512i = if ALPHA_SRC {
          let a_lo_shr = _mm512_srli_epi16::<8>(a_lo_u16);
          let a_hi_shr = _mm512_srli_epi16::<8>(a_hi_u16);
          narrow_u8x64(a_lo_shr, a_hi_shr, pack_fixup)
        } else {
          alpha_u8 // 0xFF — opaque (unused, but allowed)
        };
        write_rgba_64(r_u8, g_u8, b_u8, a_vec, out_ptr);
      } else {
        write_rgb_64(r_u8, g_u8, b_u8, out_ptr);
      }

      x += 64;
    }

    // Scalar tail — remaining < 64 pixels.
    if x < width {
      let tail_packed = &packed[x * 4..width * 4];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      scalar::ayuv64_to_rgb_or_rgba_row::<ALPHA, ALPHA_SRC>(
        tail_packed,
        tail_out,
        tail_w,
        matrix,
        full_range,
      );
    }
  }
}

// ---- u16 RGB / RGBA native-depth output (32 px/iter) --------------------

/// AVX-512 AYUV64 → packed native-depth u16 RGB or RGBA.
///
/// Uses i64 chroma (`chroma_i64x8_avx512`) to avoid overflow at BITS=16/16.
/// Byte-identical to `scalar::ayuv64_to_rgb_u16_or_rgba_u16_row::<ALPHA, ALPHA_SRC>`.
///
/// Block size: 32 pixels per SIMD iteration (one 32-pixel deinterleave).
///
/// Valid monomorphizations:
/// - `<false, false>` — RGB u16 (α dropped)
/// - `<true, true>`  — RGBA u16, source α written direct (no conversion)
///
/// `<false, true>` is rejected at monomorphization via `const { assert! }`.
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available.**
/// 2. `packed.len() >= width * 4`.
/// 3. `out.len() >= width * (if ALPHA { 4 } else { 3 })` (u16 elements).
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn ayuv64_to_rgb_u16_or_rgba_u16_row<const ALPHA: bool, const ALPHA_SRC: bool>(
  packed: &[u16],
  out: &mut [u16],
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
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<16, 16>(full_range);
  const RND_I64: i64 = 1 << 14;
  const RND_I32: i32 = 1 << 14;

  // SAFETY: AVX-512F + AVX-512BW availability is the caller's obligation.
  unsafe {
    let alpha_u16_v = _mm_set1_epi16(-1i16); // 0xFFFF for forced-opaque path.
    let rnd_i64_v = _mm512_set1_epi64(RND_I64);
    let rnd_i32_v = _mm512_set1_epi32(RND_I32);
    let y_off_v = _mm512_set1_epi32(y_off);
    let y_scale_v = _mm512_set1_epi32(y_scale);
    let c_scale_v = _mm512_set1_epi32(c_scale);
    // Subtract chroma bias (32768) via wrapping i16 trick.
    let bias16_v = _mm512_set1_epi16(-32768i16);
    let cru = _mm512_set1_epi32(coeffs.r_u());
    let crv = _mm512_set1_epi32(coeffs.r_v());
    let cgu = _mm512_set1_epi32(coeffs.g_u());
    let cgv = _mm512_set1_epi32(coeffs.g_v());
    let cbu = _mm512_set1_epi32(coeffs.b_u());
    let cbv = _mm512_set1_epi32(coeffs.b_v());

    // Permute indices.
    // interleave_idx: even i32x8 + odd i32x8 → i32x16 [e0,o0,e1,o1,...].
    let interleave_idx = _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
    let pack_fixup = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);

    let mut x = 0usize;
    while x + 32 <= width {
      // Deinterleave 32 AYUV64 quadruples → A, Y, U, V as u16x32 in
      // natural pixel order.
      let (a_u16, y_vec, u_u16, v_u16) =
        deinterleave_ayuv64_32px_avx512(packed.as_ptr().add(x * 4));

      // Center chroma via wrapping i16 subtraction.
      let u_i16 = _mm512_sub_epi16(u_u16, bias16_v);
      let v_i16 = _mm512_sub_epi16(v_u16, bias16_v);

      // Widen each i16x32 chroma into two i32x16 halves for Q15 multiply.
      let u_lo_i32 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_i16));
      let u_hi_i32 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(u_i16));
      let v_lo_i32 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_i16));
      let v_hi_i32 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(v_i16));

      // Scale UV in i32: |u_centered| ≤ 32768, |c_scale| ≤ ~38300 →
      // product ≤ ~1.26·10⁹ — fits i32.
      let u_d_lo = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_lo_i32, c_scale_v),
        rnd_i32_v,
      ));
      let u_d_hi = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(u_hi_i32, c_scale_v),
        rnd_i32_v,
      ));
      let v_d_lo = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_lo_i32, c_scale_v),
        rnd_i32_v,
      ));
      let v_d_hi = q15_shift(_mm512_add_epi32(
        _mm512_mullo_epi32(v_hi_i32, c_scale_v),
        rnd_i32_v,
      ));

      // i64 chroma: even/odd i32 lanes via 0xF5 shuffle.
      // Each `chroma_i64x8_avx512` call processes 8 i64 values from the
      // even-indexed i32 lanes of u_d/v_d. We need 16 per half → two calls
      // per half (even + odd), then reassemble.
      let u_d_lo_odd = _mm512_shuffle_epi32::<0xF5>(u_d_lo);
      let v_d_lo_odd = _mm512_shuffle_epi32::<0xF5>(v_d_lo);
      let u_d_hi_odd = _mm512_shuffle_epi32::<0xF5>(u_d_hi);
      let v_d_hi_odd = _mm512_shuffle_epi32::<0xF5>(v_d_hi);

      let r_ch_lo_even = chroma_i64x8_avx512(cru, crv, u_d_lo, v_d_lo, rnd_i64_v);
      let r_ch_lo_odd = chroma_i64x8_avx512(cru, crv, u_d_lo_odd, v_d_lo_odd, rnd_i64_v);
      let g_ch_lo_even = chroma_i64x8_avx512(cgu, cgv, u_d_lo, v_d_lo, rnd_i64_v);
      let g_ch_lo_odd = chroma_i64x8_avx512(cgu, cgv, u_d_lo_odd, v_d_lo_odd, rnd_i64_v);
      let b_ch_lo_even = chroma_i64x8_avx512(cbu, cbv, u_d_lo, v_d_lo, rnd_i64_v);
      let b_ch_lo_odd = chroma_i64x8_avx512(cbu, cbv, u_d_lo_odd, v_d_lo_odd, rnd_i64_v);

      let r_ch_hi_even = chroma_i64x8_avx512(cru, crv, u_d_hi, v_d_hi, rnd_i64_v);
      let r_ch_hi_odd = chroma_i64x8_avx512(cru, crv, u_d_hi_odd, v_d_hi_odd, rnd_i64_v);
      let g_ch_hi_even = chroma_i64x8_avx512(cgu, cgv, u_d_hi, v_d_hi, rnd_i64_v);
      let g_ch_hi_odd = chroma_i64x8_avx512(cgu, cgv, u_d_hi_odd, v_d_hi_odd, rnd_i64_v);
      let b_ch_hi_even = chroma_i64x8_avx512(cbu, cbv, u_d_hi, v_d_hi, rnd_i64_v);
      let b_ch_hi_odd = chroma_i64x8_avx512(cbu, cbv, u_d_hi_odd, v_d_hi_odd, rnd_i64_v);

      // Reassemble each pair of i64x8 → i32x16.
      let r_ch_lo_i32 = reassemble_i32x16(r_ch_lo_even, r_ch_lo_odd, interleave_idx);
      let g_ch_lo_i32 = reassemble_i32x16(g_ch_lo_even, g_ch_lo_odd, interleave_idx);
      let b_ch_lo_i32 = reassemble_i32x16(b_ch_lo_even, b_ch_lo_odd, interleave_idx);
      let r_ch_hi_i32 = reassemble_i32x16(r_ch_hi_even, r_ch_hi_odd, interleave_idx);
      let g_ch_hi_i32 = reassemble_i32x16(g_ch_hi_even, g_ch_hi_odd, interleave_idx);
      let b_ch_hi_i32 = reassemble_i32x16(b_ch_hi_even, b_ch_hi_odd, interleave_idx);

      // Y: unsigned-widen u16 → i32, subtract y_off, scale via i64.
      // y_vec is __m512i with 32 u16 lanes (Y0..Y31).
      let y_lo_u16 = _mm512_castsi512_si256(y_vec);
      let y_hi_u16 = _mm512_extracti64x4_epi64::<1>(y_vec);
      let y_lo_i32 = _mm512_sub_epi32(_mm512_cvtepu16_epi32(y_lo_u16), y_off_v);
      let y_hi_i32 = _mm512_sub_epi32(_mm512_cvtepu16_epi32(y_hi_u16), y_off_v);

      let y_lo_scaled = scale_y_i32x16_i64(y_lo_i32, y_scale_v, rnd_i64_v, interleave_idx);
      let y_hi_scaled = scale_y_i32x16_i64(y_hi_i32, y_scale_v, rnd_i64_v, interleave_idx);

      // Add Y + chroma in i32; saturate-narrow to u16 via _mm512_packus_epi32
      // + pack_fixup (packus is per-128-bit-lane; produces lane-split result).
      let r_u16 = _mm512_permutexvar_epi64(
        pack_fixup,
        _mm512_packus_epi32(
          _mm512_add_epi32(y_lo_scaled, r_ch_lo_i32),
          _mm512_add_epi32(y_hi_scaled, r_ch_hi_i32),
        ),
      );
      let g_u16 = _mm512_permutexvar_epi64(
        pack_fixup,
        _mm512_packus_epi32(
          _mm512_add_epi32(y_lo_scaled, g_ch_lo_i32),
          _mm512_add_epi32(y_hi_scaled, g_ch_hi_i32),
        ),
      );
      let b_u16 = _mm512_permutexvar_epi64(
        pack_fixup,
        _mm512_packus_epi32(
          _mm512_add_epi32(y_lo_scaled, b_ch_lo_i32),
          _mm512_add_epi32(y_hi_scaled, b_ch_hi_i32),
        ),
      );

      // Write 32 pixels via write_rgb_u16_32 / write_rgba_u16_32 helpers.
      if ALPHA {
        // Source α: direct write (no conversion needed for u16 output).
        // The shared write_rgba_u16_32 helper splatters one i16x8 alpha
        // to all 32 lanes; we need per-pixel α from the deinterleaved
        // a_u16 vector, so write each 8-pixel quarter manually with the
        // matching α quarter.
        let dst = out.as_mut_ptr().add(x * 4);
        let r0: __m128i = _mm512_castsi512_si128(r_u16);
        let r1: __m128i = _mm512_extracti32x4_epi32::<1>(r_u16);
        let r2: __m128i = _mm512_extracti32x4_epi32::<2>(r_u16);
        let r3: __m128i = _mm512_extracti32x4_epi32::<3>(r_u16);
        let g0: __m128i = _mm512_castsi512_si128(g_u16);
        let g1: __m128i = _mm512_extracti32x4_epi32::<1>(g_u16);
        let g2: __m128i = _mm512_extracti32x4_epi32::<2>(g_u16);
        let g3: __m128i = _mm512_extracti32x4_epi32::<3>(g_u16);
        let b0: __m128i = _mm512_castsi512_si128(b_u16);
        let b1: __m128i = _mm512_extracti32x4_epi32::<1>(b_u16);
        let b2: __m128i = _mm512_extracti32x4_epi32::<2>(b_u16);
        let b3: __m128i = _mm512_extracti32x4_epi32::<3>(b_u16);
        let (a0, a1, a2, a3) = if ALPHA_SRC {
          (
            _mm512_castsi512_si128(a_u16),
            _mm512_extracti32x4_epi32::<1>(a_u16),
            _mm512_extracti32x4_epi32::<2>(a_u16),
            _mm512_extracti32x4_epi32::<3>(a_u16),
          )
        } else {
          (alpha_u16_v, alpha_u16_v, alpha_u16_v, alpha_u16_v)
        };
        // Each `write_rgba_u16_8` writes 8 pixels × 4 × u16 = 32 u16 elements.
        write_rgba_u16_8(r0, g0, b0, a0, dst);
        write_rgba_u16_8(r1, g1, b1, a1, dst.add(32));
        write_rgba_u16_8(r2, g2, b2, a2, dst.add(64));
        write_rgba_u16_8(r3, g3, b3, a3, dst.add(96));
      } else {
        write_rgb_u16_32(r_u16, g_u16, b_u16, out.as_mut_ptr().add(x * 3));
      }

      x += 32;
    }

    // Scalar tail — remaining < 32 pixels.
    if x < width {
      let tail_packed = &packed[x * 4..width * 4];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      scalar::ayuv64_to_rgb_u16_or_rgba_u16_row::<ALPHA, ALPHA_SRC>(
        tail_packed,
        tail_out,
        tail_w,
        matrix,
        full_range,
      );
    }
  }
}

// ---- thin wrappers -------------------------------------------------------

/// AVX-512 AYUV64 → packed **RGB** (3 bpp). Source α is discarded.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn ayuv64_to_rgb_row(
  packed: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  unsafe {
    ayuv64_to_rgb_or_rgba_row::<false, false>(packed, rgb_out, width, matrix, full_range);
  }
}

/// AVX-512 AYUV64 → packed **RGBA** (4 bpp). Source A u16 is depth-converted
/// to u8 via `>> 8`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn ayuv64_to_rgba_row(
  packed: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  unsafe {
    ayuv64_to_rgb_or_rgba_row::<true, true>(packed, rgba_out, width, matrix, full_range);
  }
}

/// AVX-512 AYUV64 → packed **RGB u16** (3 × u16 per pixel). Source α discarded.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn ayuv64_to_rgb_u16_row(
  packed: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  unsafe {
    ayuv64_to_rgb_u16_or_rgba_u16_row::<false, false>(packed, rgb_out, width, matrix, full_range);
  }
}

/// AVX-512 AYUV64 → packed **RGBA u16** (4 × u16 per pixel). Source A u16
/// is written direct (no conversion).
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn ayuv64_to_rgba_u16_row(
  packed: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  unsafe {
    ayuv64_to_rgb_u16_or_rgba_u16_row::<true, true>(packed, rgba_out, width, matrix, full_range);
  }
}

// ---- Luma u8 (32 px/iter) -----------------------------------------------

/// AVX-512 AYUV64 → u8 luma. Y is the second u16 (slot 1) of each pixel
/// quadruple; `>> 8` extracts the high byte.
///
/// Block size: 32 pixels per SIMD iteration (one 32-pixel deinterleave).
/// Reuses the full deinterleave helper and discards A/U/V — the
/// compiler lifts the dead per-channel ops, and keeping the same code
/// path gives the lane-order regression test the strongest possible
/// coverage.
///
/// Byte-identical to `scalar::ayuv64_to_luma_row`.
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available.**
/// 2. `packed.len() >= width * 4`.
/// 3. `luma_out.len() >= width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn ayuv64_to_luma_row(packed: &[u16], luma_out: &mut [u8], width: usize) {
  debug_assert!(packed.len() >= width * 4, "packed row too short");
  debug_assert!(luma_out.len() >= width, "luma row too short");

  // SAFETY: AVX-512F + AVX-512BW availability is the caller's obligation.
  unsafe {
    let pack_fixup = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
    let zero = _mm512_setzero_si512();

    let mut x = 0usize;
    while x + 32 <= width {
      // Deinterleave 32 pixels and discard A/U/V.
      let (_a, y_vec, _u, _v) = deinterleave_ayuv64_32px_avx512(packed.as_ptr().add(x * 4));

      // y_vec is i16x32 with Y0..Y31 (16-bit native).
      // `>> 8` → high byte of each Y u16. Then narrow to u8.
      let y_shr = _mm512_srli_epi16::<8>(y_vec);

      // Narrow to u8x64: only low 32 bytes carry valid data; high 32 from zero.
      let y_u8 = narrow_u8x64(y_shr, zero, pack_fixup);

      // Store low 32 bytes (the valid Y values) via the low 256-bit half.
      _mm256_storeu_si256(
        luma_out.as_mut_ptr().add(x).cast(),
        _mm512_castsi512_si256(y_u8),
      );

      x += 32;
    }

    // Scalar tail.
    if x < width {
      scalar::ayuv64_to_luma_row(
        &packed[x * 4..width * 4],
        &mut luma_out[x..width],
        width - x,
      );
    }
  }
}

// ---- Luma u16 (32 px/iter) ----------------------------------------------

/// AVX-512 AYUV64 → u16 luma. Direct copy of Y samples (slot 1, no shift —
/// 16-bit native).
///
/// Block size: 32 pixels per SIMD iteration. Reuses the full
/// deinterleave helper and discards A/U/V — compiler lifts dead ops.
///
/// Byte-identical to `scalar::ayuv64_to_luma_u16_row`.
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available.**
/// 2. `packed.len() >= width * 4`.
/// 3. `luma_out.len() >= width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn ayuv64_to_luma_u16_row(packed: &[u16], luma_out: &mut [u16], width: usize) {
  debug_assert!(packed.len() >= width * 4, "packed row too short");
  debug_assert!(luma_out.len() >= width, "luma row too short");

  // SAFETY: AVX-512F + AVX-512BW availability is the caller's obligation.
  unsafe {
    let mut x = 0usize;
    while x + 32 <= width {
      let (_a, y_vec, _u, _v) = deinterleave_ayuv64_32px_avx512(packed.as_ptr().add(x * 4));
      // Direct store — Y samples are 16-bit native, in natural pixel order.
      _mm512_storeu_si512(luma_out.as_mut_ptr().add(x).cast(), y_vec);
      x += 32;
    }

    // Scalar tail.
    if x < width {
      scalar::ayuv64_to_luma_u16_row(
        &packed[x * 4..width * 4],
        &mut luma_out[x..width],
        width - x,
      );
    }
  }
}
