//! AVX2 XV36 (packed YUV 4:4:4, 12-bit) kernels.
//!
//! ## Layout
//!
//! Four `u16` elements per pixel: `[U(16), Y(16), V(16), A(16)]`
//! little-endian, each holding a 12-bit sample MSB-aligned in the
//! high 12 bits (low 4 bits zero). The `X` prefix means the A slot
//! is **padding** — loaded but discarded. RGBA outputs force α = max
//! (`0xFF` u8 / `0x0FFF` u16).
//!
//! ## Per-iter pipeline (16 px / iter)
//!
//! Four `_mm256_loadu_si256` loads fetch 64 u16 lanes (16 pixels × 4
//! channels). A cascade of `_mm256_unpacklo_epi16` / `_mm256_unpackhi_epi16`
//! deinterleaves them into four channel vectors (U, Y, V, A) of 16 u16
//! lanes each, followed by `_mm256_permute4x64_epi64` lane-fixup to
//! restore natural order across the two 128-bit AVX2 lanes:
//!
//! ```text
//! raw0 = [U0..U3,Y0..Y3,V0..V3,A0..A3, U8..U11,Y8..Y11,V8..V11,A8..A11]  (pixels 0-3, 8-11)
//! raw1 = [U4..U7,Y4..Y7,V4..V7,A4..A7, U12..U15,…]                       (pixels 4-7, 12-15)
//! raw2 = similar for pixels 0-3, 8-11 (second half of 4-u16 groups)
//! raw3 = similar for pixels 4-7, 12-15
//!
//! (3-level unpack cascade per SSE4.1 shape, lifted to 256-bit)
//!
//! After cascade, each channel vector is lane-split:
//! u_raw = [U0..U7, U8..U15] (but in AVX2 lane-split form needing fixup)
//!
//! _mm256_permute4x64_epi64::<0xD8> fixes [0,2,1,3] → natural order.
//! ```
//!
//! Each channel is then right-shifted by 4 (`_mm256_srli_epi16::<4>`) to
//! drop the 4 padding LSBs, bringing the 12-bit MSB-aligned sample to
//! `[0, 4095]`. From there the Q15 pipeline at BITS=12 is identical to
//! the SSE4.1 sibling: `chroma_i16x16` (i32 chroma) + `scale_y`.
//!
//! ## 4:4:4 vs. 4:2:2
//!
//! XV36 is 4:4:4 — no chroma duplication (`chroma_dup`) is needed.
//! All 16 lanes carry unique `(U, Y, V)` triples.
//!
//! ## Tail
//!
//! `width % 16` remaining pixels fall through to `scalar::xv36_*`.

use core::arch::x86_64::*;

use super::*;
use crate::{ColorMatrix, row::scalar};

// ---- Deinterleave helper ------------------------------------------------

/// Deinterleaves 16 XV36 quadruples (64 u16 = 128 bytes) from `ptr` into
/// `(u_vec, y_vec, v_vec)` — three `__m256i` vectors each holding 16
/// `u16` samples **after** the 4-bit right-shift to drop padding LSBs.
/// The A channel is computed but discarded by the caller.
///
/// Strategy: 4 × `_mm256_loadu_si256` + 3-level `_mm256_unpacklo/hi_epi16`
/// cascade + `_mm256_permute4x64_epi64::<0xD8>` lane-fixup on each result.
///
/// Each 256-bit register holds data from two 128-bit halves, each half
/// being a natural continuation of the SSE4.1 deinterleave for 4 pixels.
/// The AVX2 unpack ops are per-128-bit-lane, producing lane-split results
/// that need the 0xD8 permute to restore natural [0..16) order.
///
/// # Safety
///
/// `ptr` must point to at least 128 readable bytes (64 `u16` elements).
/// Caller's `target_feature` must include AVX2.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn unpack_xv36_16px_avx2(ptr: *const u16) -> (__m256i, __m256i, __m256i) {
  // SAFETY: caller obligation — `ptr` has 128 bytes readable; AVX2 is
  // available.
  unsafe {
    // Load 4 × __m256i (16 pixels × 4 channels × u16 = 128 bytes).
    // Each 256-bit register holds 16 u16 lanes split across two 128-bit
    // halves. Within each half, the layout is identical to one SSE4.1
    // load: [U_n, Y_n, V_n, A_n, U_{n+1}, Y_{n+1}, V_{n+1}, A_{n+1}].
    let raw0 = _mm256_loadu_si256(ptr.cast());           // pixels 0-1 (lo), 8-9  (hi)
    let raw1 = _mm256_loadu_si256(ptr.add(16).cast());   // pixels 2-3 (lo), 10-11 (hi)
    let raw2 = _mm256_loadu_si256(ptr.add(32).cast());   // pixels 4-5 (lo), 12-13 (hi)
    let raw3 = _mm256_loadu_si256(ptr.add(48).cast());   // pixels 6-7 (lo), 14-15 (hi)

    // Level 1: unpack pairs (0-1, 2-3) and (4-5, 6-7) within each lane.
    // Per-lane result per 128 bits mirrors SSE4.1 step 1:
    //   s1_lo per lane: [U0,U2,Y0,Y2,V0,V2,A0,A2] (lo: px0/2; hi: px8/10)
    //   s1_hi per lane: [U1,U3,Y1,Y3,V1,V3,A1,A3] (lo: px1/3; hi: px9/11)
    //   s2_lo per lane: [U4,U6,Y4,Y6,V4,V6,A4,A6] (lo: px4/6; hi: px12/14)
    //   s2_hi per lane: [U5,U7,Y5,Y7,V5,V7,A5,A7] (lo: px5/7; hi: px13/15)
    let s1_lo = _mm256_unpacklo_epi16(raw0, raw1);
    let s1_hi = _mm256_unpackhi_epi16(raw0, raw1);
    let s2_lo = _mm256_unpacklo_epi16(raw2, raw3);
    let s2_hi = _mm256_unpackhi_epi16(raw2, raw3);

    // Level 2: merge lo/hi within each group.
    // Per-lane:
    //   s3_lo: [U0,U1,U2,U3,Y0,Y1,Y2,Y3] (lo: px0-3; hi: px8-11)
    //   s3_hi: [V0,V1,V2,V3,A0,A1,A2,A3] (lo: px0-3; hi: px8-11)
    //   s4_lo: [U4,U5,U6,U7,Y4,Y5,Y6,Y7] (lo: px4-7; hi: px12-15)
    //   s4_hi: [V4,V5,V6,V7,A4,A5,A6,A7] (lo: px4-7; hi: px12-15)
    let s3_lo = _mm256_unpacklo_epi16(s1_lo, s1_hi);
    let s3_hi = _mm256_unpackhi_epi16(s1_lo, s1_hi);
    let s4_lo = _mm256_unpacklo_epi16(s2_lo, s2_hi);
    let s4_hi = _mm256_unpackhi_epi16(s2_lo, s2_hi);

    // Level 3: combine two groups to get full 16-lane channel vectors
    // (still lane-split — each 256-bit result has the first group's data
    // in the low 64 bits of each 128-bit lane and the second group's data
    // in the high 64 bits of each 128-bit lane).
    // _mm256_unpacklo_epi64 per-lane → [U0..U3, U4..U7] (lo lane: px0-7)
    //                                   [U8..U11, U12..U15] (hi lane: px8-15)
    // _mm256_permute4x64_epi64::<0xD8> reorders 64-bit chunks [0,2,1,3]
    // → brings the two "px0..7" 64-bit blocks together in the low 128,
    // and the "px8..15" 64-bit blocks together in the high 128, giving
    // natural order [U0..U7, U8..U15].
    let u_raw = _mm256_permute4x64_epi64::<0xD8>(_mm256_unpacklo_epi64(s3_lo, s4_lo));
    let y_raw = _mm256_permute4x64_epi64::<0xD8>(_mm256_unpackhi_epi64(s3_lo, s4_lo));
    let v_raw = _mm256_permute4x64_epi64::<0xD8>(_mm256_unpacklo_epi64(s3_hi, s4_hi));
    // a_raw would be: _mm256_permute4x64_epi64::<0xD8>(_mm256_unpackhi_epi64(s3_hi, s4_hi))
    // Discarded.

    // Right-shift by 4 to drop MSB-alignment padding → 12-bit range [0, 4095].
    let u_vec = _mm256_srli_epi16::<4>(u_raw);
    let y_vec = _mm256_srli_epi16::<4>(y_raw);
    let v_vec = _mm256_srli_epi16::<4>(v_raw);

    (u_vec, y_vec, v_vec)
  }
}

// ---- u8 RGB / RGBA output (16 px/iter) ----------------------------------

/// AVX2 XV36 → packed u8 RGB or RGBA.
///
/// Byte-identical to `scalar::xv36_to_rgb_or_rgba_row::<ALPHA>`.
///
/// Block size: 16 pixels per SIMD iteration (four `_mm256_loadu_si256`).
///
/// # Safety
///
/// 1. **AVX2 must be available.**
/// 2. `packed.len() >= width * 4`.
/// 3. `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx2")]
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

  // SAFETY: AVX2 availability is the caller's obligation.
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

    let mut x = 0usize;
    while x + 16 <= width {
      // Deinterleave 16 XV36 quadruples → U, Y, V as i16x16 in [0, 4095].
      let (u_u16, y_u16, v_u16) = unpack_xv36_16px_avx2(packed.as_ptr().add(x * 4));

      // Values ≤ 4095 < 32767 — safe to treat as signed i16.
      let u_i16 = u_u16;
      let y_i16 = y_u16;
      let v_i16 = v_u16;

      // Subtract chroma bias (2048 for 12-bit).
      let u_sub = _mm256_sub_epi16(u_i16, bias_v);
      let v_sub = _mm256_sub_epi16(v_i16, bias_v);

      // Widen to i32x8 lo/hi halves for Q15 chroma-scale multiply.
      let u_lo_i32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(u_sub));
      let u_hi_i32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(u_sub));
      let v_lo_i32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v_sub));
      let v_hi_i32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(v_sub));

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

      // 4:4:4 — no chroma duplication; all 16 lanes carry unique U/V per pixel.
      // chroma_i16x16 uses i32 arithmetic (sufficient for 12-bit samples).
      let r_chroma = chroma_i16x16(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let g_chroma = chroma_i16x16(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let b_chroma = chroma_i16x16(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);

      // XV36 Y ≤ 4095 fits in i16 — use scale_y (NOT scale_y_u16_avx2).
      let y_scaled = scale_y(y_i16, y_off_v, y_scale_v, rnd_v);

      // u8 narrow with saturation. All 16 lanes carry valid results.
      let zero = _mm256_setzero_si256();
      let r_u8 = narrow_u8x32(_mm256_adds_epi16(y_scaled, r_chroma), zero);
      let g_u8 = narrow_u8x32(_mm256_adds_epi16(y_scaled, g_chroma), zero);
      let b_u8 = narrow_u8x32(_mm256_adds_epi16(y_scaled, b_chroma), zero);

      // 16-pixel partial store via stack buffer + scalar interleave.
      let mut r_tmp = [0u8; 32];
      let mut g_tmp = [0u8; 32];
      let mut b_tmp = [0u8; 32];
      _mm256_storeu_si256(r_tmp.as_mut_ptr().cast(), r_u8);
      _mm256_storeu_si256(g_tmp.as_mut_ptr().cast(), g_u8);
      _mm256_storeu_si256(b_tmp.as_mut_ptr().cast(), b_u8);

      if ALPHA {
        let dst = &mut out[x * 4..x * 4 + 16 * 4];
        for i in 0..16 {
          dst[i * 4] = r_tmp[i];
          dst[i * 4 + 1] = g_tmp[i];
          dst[i * 4 + 2] = b_tmp[i];
          dst[i * 4 + 3] = 0xFF;
        }
      } else {
        let dst = &mut out[x * 3..x * 3 + 16 * 3];
        for i in 0..16 {
          dst[i * 3] = r_tmp[i];
          dst[i * 3 + 1] = g_tmp[i];
          dst[i * 3 + 2] = b_tmp[i];
        }
      }

      x += 16;
    }

    // Scalar tail — remaining < 16 pixels.
    if x < width {
      let tail_packed = &packed[x * 4..width * 4];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      scalar::xv36_to_rgb_or_rgba_row::<ALPHA>(tail_packed, tail_out, tail_w, matrix, full_range);
    }
  }
}

// ---- u16 RGB / RGBA native-depth output (16 px/iter) --------------------

/// AVX2 XV36 → packed native-depth u16 RGB or RGBA (low-bit-packed at
/// 12-bit).
///
/// Byte-identical to `scalar::xv36_to_rgb_u16_or_rgba_u16_row::<ALPHA>`.
///
/// Block size: 16 pixels per SIMD iteration.
///
/// # Safety
///
/// 1. **AVX2 must be available.**
/// 2. `packed.len() >= width * 4`.
/// 3. `out.len() >= width * (if ALPHA { 4 } else { 3 })` (u16 elements).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx2")]
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

  // SAFETY: AVX2 availability is the caller's obligation.
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

    let mut x = 0usize;
    while x + 16 <= width {
      let (u_u16, y_u16, v_u16) = unpack_xv36_16px_avx2(packed.as_ptr().add(x * 4));

      let u_i16 = u_u16;
      let y_i16 = y_u16;
      let v_i16 = v_u16;

      let u_sub = _mm256_sub_epi16(u_i16, bias_v);
      let v_sub = _mm256_sub_epi16(v_i16, bias_v);

      let u_lo_i32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(u_sub));
      let u_hi_i32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(u_sub));
      let v_lo_i32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v_sub));
      let v_hi_i32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(v_sub));

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

      // 12-bit chroma: i32 arithmetic is sufficient (no overflow at 12-bit).
      let r_chroma = chroma_i16x16(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let g_chroma = chroma_i16x16(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let b_chroma = chroma_i16x16(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);

      // XV36 Y ≤ 4095 fits in i16 — use scale_y (NOT scale_y_u16_avx2).
      let y_scaled = scale_y(y_i16, y_off_v, y_scale_v, rnd_v);

      // Clamp to [0, 0x0FFF] (12-bit low-bit-packed output range).
      let r = clamp_u16_max_x16(_mm256_adds_epi16(y_scaled, r_chroma), zero_v, max_v);
      let g = clamp_u16_max_x16(_mm256_adds_epi16(y_scaled, g_chroma), zero_v, max_v);
      let b = clamp_u16_max_x16(_mm256_adds_epi16(y_scaled, b_chroma), zero_v, max_v);

      // 16-pixel u16 store via stack buffer + scalar interleave.
      let mut r_tmp = [0u16; 16];
      let mut g_tmp = [0u16; 16];
      let mut b_tmp = [0u16; 16];
      _mm256_storeu_si256(r_tmp.as_mut_ptr().cast(), r);
      _mm256_storeu_si256(g_tmp.as_mut_ptr().cast(), g);
      _mm256_storeu_si256(b_tmp.as_mut_ptr().cast(), b);

      if ALPHA {
        let dst = &mut out[x * 4..x * 4 + 16 * 4];
        for i in 0..16 {
          dst[i * 4] = r_tmp[i];
          dst[i * 4 + 1] = g_tmp[i];
          dst[i * 4 + 2] = b_tmp[i];
          dst[i * 4 + 3] = alpha_u16;
        }
      } else {
        let dst = &mut out[x * 3..x * 3 + 16 * 3];
        for i in 0..16 {
          dst[i * 3] = r_tmp[i];
          dst[i * 3 + 1] = g_tmp[i];
          dst[i * 3 + 2] = b_tmp[i];
        }
      }

      x += 16;
    }

    // Scalar tail — remaining < 16 pixels.
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

// ---- Luma u8 (16 px/iter) -----------------------------------------------

/// AVX2 XV36 → u8 luma. Y is quadruple element 1 (offset 1 in each
/// group of 4 u16). The deinterleave yields Y in [0, 4095] (>> 4
/// already applied); one more `>> 4` gives 8-bit (same as scalar
/// `packed[x*4+1] >> 8`).
///
/// Byte-identical to `scalar::xv36_to_luma_row`.
///
/// Block size: 16 pixels per SIMD iteration.
///
/// # Safety
///
/// 1. **AVX2 must be available.**
/// 2. `packed.len() >= width * 4`.
/// 3. `out.len() >= width`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn xv36_to_luma_row(packed: &[u16], out: &mut [u8], width: usize) {
  debug_assert!(packed.len() >= width * 4);
  debug_assert!(out.len() >= width);

  // SAFETY: AVX2 availability is the caller's obligation.
  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      let (_u_vec, y_vec, _v_vec) = unpack_xv36_16px_avx2(packed.as_ptr().add(x * 4));

      // y_vec is already >> 4 (values in [0, 4095]).
      // Scalar does `packed[x*4+1] >> 8` — that is the MSB-aligned value >> 4
      // to get 12-bit, then >> 4 more to get 8-bit. Apply one more >> 4.
      let y_shr = _mm256_srli_epi16::<4>(y_vec);

      // Narrow to u8 (values ≤ 255, no saturation). Low 16 bytes hold valid
      // results; high 16 bytes (from zero) are zero.
      let zero = _mm256_setzero_si256();
      let y_u8 = narrow_u8x32(y_shr, zero);

      // Store 16 valid bytes via stack buffer + copy_from_slice.
      let mut tmp = [0u8; 32];
      _mm256_storeu_si256(tmp.as_mut_ptr().cast(), y_u8);
      out[x..x + 16].copy_from_slice(&tmp[..16]);

      x += 16;
    }

    // Scalar tail — remaining < 16 pixels.
    if x < width {
      scalar::xv36_to_luma_row(&packed[x * 4..width * 4], &mut out[x..width], width - x);
    }
  }
}

// ---- Luma u16 (16 px/iter) -----------------------------------------------

/// AVX2 XV36 → u16 luma (low-bit-packed at 12-bit). Y is quadruple
/// element 1; `>> 4` (already applied by `unpack_xv36_16px_avx2`) drops
/// the 4 padding LSBs to give a 12-bit value in `[0, 4095]`.
///
/// Byte-identical to `scalar::xv36_to_luma_u16_row`.
///
/// Block size: 16 pixels per SIMD iteration.
///
/// # Safety
///
/// 1. **AVX2 must be available.**
/// 2. `packed.len() >= width * 4`.
/// 3. `out.len() >= width`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn xv36_to_luma_u16_row(packed: &[u16], out: &mut [u16], width: usize) {
  debug_assert!(packed.len() >= width * 4);
  debug_assert!(out.len() >= width);

  // SAFETY: AVX2 availability is the caller's obligation.
  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      let (_u_vec, y_vec, _v_vec) = unpack_xv36_16px_avx2(packed.as_ptr().add(x * 4));

      // y_vec already has >> 4 applied (= 12-bit value in [0, 4095]).
      // Direct store of 16 × u16.
      _mm256_storeu_si256(out.as_mut_ptr().add(x).cast(), y_vec);

      x += 16;
    }

    // Scalar tail — remaining < 16 pixels.
    if x < width {
      scalar::xv36_to_luma_u16_row(&packed[x * 4..width * 4], &mut out[x..width], width - x);
    }
  }
}
