//! SSE4.1 kernels for VUYA / VUYX packed YUV 4:4:4 8-bit family.
//!
//! ## Layout
//!
//! Four `u8` elements per pixel: `V(8) ‖ U(8) ‖ Y(8) ‖ A(8)`.
//! VUYA carries a real alpha channel in byte 3. VUYX treats byte 3 as
//! padding and forces output α to `0xFF`.
//!
//! ## Per-iter pipeline (16 px / iter)
//!
//! Four `_mm_loadu_si128` loads fetch 64 bytes = 16 pixels of
//! `V U Y A V U Y A V U Y A V U Y A`. Each 16-byte register holds 4
//! pixels. Four `_mm_shuffle_epi8` masks extract bytes at positions
//! 0/4/8/12 (V), 1/5/9/13 (U), 2/6/10/14 (Y), 3/7/11/15 (A) —
//! placing each channel's 4 bytes in the low lanes with zeros elsewhere.
//! A `_mm_unpacklo_epi32` / `_mm_unpackhi_epi32` cascade merges the
//! 4 × 4-byte chunks into a full 16-byte channel vector.
//!
//! For each combined channel vector (V/U/Y), zero-extend low/high halves
//! to i16x8 via `_mm_cvtepu8_epi16`, subtract chroma bias (128), widen
//! to i32x4, and run the Q15 pipeline identical to the NV24 and XView36
//! SSE4.1 siblings. Pack RGB output via `_mm_packus_epi16` cascades.
//!
//! α handling: when `ALPHA && ALPHA_SRC`, use the A vector from the
//! deinterleave. When `ALPHA && !ALPHA_SRC`, use `_mm_set1_epi8(-1)`
//! (= 0xFF). RGB interleave via `write_rgb_16`; RGBA via `write_rgba_16`.
//!
//! ## Tail
//!
//! `width % 16` remaining pixels fall through to
//! `scalar::vuya_to_rgb_or_rgba_row`.

use core::arch::x86_64::*;

use super::*;
use crate::{ColorMatrix, row::scalar};

// ---- Deinterleave helper ------------------------------------------------

/// Deinterleaves 16 VUYA quadruples (64 bytes = 16 pixels) from `ptr`
/// into `(v_vec, u_vec, y_vec, a_vec)` — four `__m128i` vectors each
/// holding 16 `u8` samples.
///
/// ## Strategy
///
/// Load 4 × 16 bytes (4 pixels each). Each load contains bytes in the
/// order `V U Y A V U Y A V U Y A V U Y A`. Four shuffle masks extract:
/// - V: bytes at offsets 0, 4, 8, 12 → first 4 bytes, rest zero
/// - U: bytes at offsets 1, 5, 9, 13
/// - Y: bytes at offsets 2, 6, 10, 14
/// - A: bytes at offsets 3, 7, 11, 15
///
/// Combining with `_mm_unpacklo_epi32` / `_mm_unpackhi_epi32` assembles
/// the 4 × 4-byte chunks from all 4 loads into a single 16-byte vector
/// per channel.
///
/// # Safety
///
/// `ptr` must point to at least 64 readable bytes (16 VUYA quadruples).
/// Caller's `target_feature` must include SSE4.1 (implies SSSE3 for
/// `_mm_shuffle_epi8`).
#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn deinterleave_vuya(ptr: *const u8) -> (__m128i, __m128i, __m128i, __m128i) {
  unsafe {
    // Load 4 × 16 bytes (4 pixels each).
    let raw0 = _mm_loadu_si128(ptr.cast()); // pixels 0-3
    let raw1 = _mm_loadu_si128(ptr.add(16).cast()); // pixels 4-7
    let raw2 = _mm_loadu_si128(ptr.add(32).cast()); // pixels 8-11
    let raw3 = _mm_loadu_si128(ptr.add(48).cast()); // pixels 12-15

    // Shuffle masks: gather the relevant byte from each pixel quadruple
    // into the low 4 bytes; upper 12 bytes are zeroed (0x80 source index).
    // VUYA layout in each 16-byte register: V0 U0 Y0 A0 V1 U1 Y1 A1 V2 U2 Y2 A2 V3 U3 Y3 A3
    //   V at positions 0,  4,  8, 12
    //   U at positions 1,  5,  9, 13
    //   Y at positions 2,  6, 10, 14
    //   A at positions 3,  7, 11, 15
    let v_mask = _mm_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let u_mask = _mm_setr_epi8(1, 5, 9, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let y_mask = _mm_setr_epi8(2, 6, 10, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let a_mask = _mm_setr_epi8(3, 7, 11, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);

    // Apply masks: each result has 4 valid bytes in the low 4 lanes.
    let v0 = _mm_shuffle_epi8(raw0, v_mask); // [V0..V3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    let v1 = _mm_shuffle_epi8(raw1, v_mask); // [V4..V7, ...]
    let v2 = _mm_shuffle_epi8(raw2, v_mask); // [V8..V11, ...]
    let v3 = _mm_shuffle_epi8(raw3, v_mask); // [V12..V15, ...]

    let u0 = _mm_shuffle_epi8(raw0, u_mask);
    let u1 = _mm_shuffle_epi8(raw1, u_mask);
    let u2 = _mm_shuffle_epi8(raw2, u_mask);
    let u3 = _mm_shuffle_epi8(raw3, u_mask);

    let y0 = _mm_shuffle_epi8(raw0, y_mask);
    let y1 = _mm_shuffle_epi8(raw1, y_mask);
    let y2 = _mm_shuffle_epi8(raw2, y_mask);
    let y3 = _mm_shuffle_epi8(raw3, y_mask);

    let a0 = _mm_shuffle_epi8(raw0, a_mask);
    let a1 = _mm_shuffle_epi8(raw1, a_mask);
    let a2 = _mm_shuffle_epi8(raw2, a_mask);
    let a3 = _mm_shuffle_epi8(raw3, a_mask);

    // Combine 4 × 4-byte chunks into a single 16-byte vector per channel.
    // unpacklo_epi32([A,B,C,D,...],[E,F,G,H,...]) = [A,B,E,F, C,D,G,H]
    // (where A,B,C,D etc. are 32-bit words)
    // Since only the first 32-bit word of each chunk is non-zero, this
    // effectively merges two 4-byte groups into the low 8 bytes.
    //
    // Step 1: interleave pairs (0+1, 2+3) → 8 valid bytes each.
    let v_01 = _mm_unpacklo_epi32(v0, v1); // [V0,V1,V2,V3, V4,V5,V6,V7, 0...]
    let v_23 = _mm_unpacklo_epi32(v2, v3); // [V8..V11, V12..V15, 0...]
    let u_01 = _mm_unpacklo_epi32(u0, u1);
    let u_23 = _mm_unpacklo_epi32(u2, u3);
    let y_01 = _mm_unpacklo_epi32(y0, y1);
    let y_23 = _mm_unpacklo_epi32(y2, y3);
    let a_01 = _mm_unpacklo_epi32(a0, a1);
    let a_23 = _mm_unpacklo_epi32(a2, a3);

    // Step 2: combine the two 8-byte halves into a full 16-byte vector.
    let v_vec = _mm_unpacklo_epi64(v_01, v_23); // [V0..V15]
    let u_vec = _mm_unpacklo_epi64(u_01, u_23); // [U0..U15]
    let y_vec = _mm_unpacklo_epi64(y_01, y_23); // [Y0..Y15]
    let a_vec = _mm_unpacklo_epi64(a_01, a_23); // [A0..A15]

    (v_vec, u_vec, y_vec, a_vec)
  }
}

// ---- shared RGB/RGBA kernel (16 px/iter) --------------------------------

/// SSE4.1 VUYA/VUYX → packed u8 RGB or RGBA.
///
/// Byte-identical to `scalar::vuya_to_rgb_or_rgba_row::<ALPHA, ALPHA_SRC>`.
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
/// 1. **SSE4.1 must be available.**
/// 2. `packed.len() >= width * 4`.
/// 3. `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
#[inline]
#[target_feature(enable = "sse4.1")]
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
    let alpha_u8 = _mm_set1_epi8(-1i8); // 0xFF for VUYX forced-opaque path

    let mut x = 0usize;
    while x + 16 <= width {
      // Deinterleave 16 VUYA quadruples → V, U, Y, A as u8x16.
      let (v_u8, u_u8, y_u8, a_u8) = deinterleave_vuya(packed.as_ptr().add(x * 4));

      // Zero-extend low/high halves to i16x8.
      let v_lo_i16 = _mm_cvtepu8_epi16(v_u8);
      let v_hi_i16 = _mm_cvtepu8_epi16(_mm_srli_si128::<8>(v_u8));
      let u_lo_i16 = _mm_cvtepu8_epi16(u_u8);
      let u_hi_i16 = _mm_cvtepu8_epi16(_mm_srli_si128::<8>(u_u8));
      let y_lo_i16 = _mm_cvtepu8_epi16(y_u8);
      let y_hi_i16 = _mm_cvtepu8_epi16(_mm_srli_si128::<8>(y_u8));

      // Subtract chroma bias (128 for 8-bit).
      let u_sub_lo = _mm_sub_epi16(u_lo_i16, bias_v);
      let u_sub_hi = _mm_sub_epi16(u_hi_i16, bias_v);
      let v_sub_lo = _mm_sub_epi16(v_lo_i16, bias_v);
      let v_sub_hi = _mm_sub_epi16(v_hi_i16, bias_v);

      // Widen to i32x4 for Q15 chroma-scale multiply — low half.
      let u_lo_a = _mm_cvtepi16_epi32(u_sub_lo);
      let u_lo_b = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(u_sub_lo));
      let v_lo_a = _mm_cvtepi16_epi32(v_sub_lo);
      let v_lo_b = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(v_sub_lo));

      let u_d_lo_a = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_lo_a, c_scale_v), rnd_v));
      let u_d_lo_b = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_lo_b, c_scale_v), rnd_v));
      let v_d_lo_a = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_lo_a, c_scale_v), rnd_v));
      let v_d_lo_b = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_lo_b, c_scale_v), rnd_v));

      // Chroma for low 8 lanes.
      let r_chroma_lo = chroma_i16x8(cru, crv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let g_chroma_lo = chroma_i16x8(cgu, cgv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let b_chroma_lo = chroma_i16x8(cbu, cbv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);

      // Widen to i32x4 for Q15 chroma-scale multiply — high half.
      let u_hi_a = _mm_cvtepi16_epi32(u_sub_hi);
      let u_hi_b = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(u_sub_hi));
      let v_hi_a = _mm_cvtepi16_epi32(v_sub_hi);
      let v_hi_b = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(v_sub_hi));

      let u_d_hi_a = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_hi_a, c_scale_v), rnd_v));
      let u_d_hi_b = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_hi_b, c_scale_v), rnd_v));
      let v_d_hi_a = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_hi_a, c_scale_v), rnd_v));
      let v_d_hi_b = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_hi_b, c_scale_v), rnd_v));

      // Chroma for high 8 lanes.
      let r_chroma_hi = chroma_i16x8(cru, crv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let g_chroma_hi = chroma_i16x8(cgu, cgv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let b_chroma_hi = chroma_i16x8(cbu, cbv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);

      // Y: scale both halves.
      let y_scaled_lo = scale_y(y_lo_i16, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_hi_i16, y_off_v, y_scale_v, rnd_v);

      // Saturate-add Y + chroma, then saturate-narrow to u8x16 per channel.
      let r_lo = _mm_adds_epi16(y_scaled_lo, r_chroma_lo);
      let r_hi = _mm_adds_epi16(y_scaled_hi, r_chroma_hi);
      let g_lo = _mm_adds_epi16(y_scaled_lo, g_chroma_lo);
      let g_hi = _mm_adds_epi16(y_scaled_hi, g_chroma_hi);
      let b_lo = _mm_adds_epi16(y_scaled_lo, b_chroma_lo);
      let b_hi = _mm_adds_epi16(y_scaled_hi, b_chroma_hi);

      let r_u8 = _mm_packus_epi16(r_lo, r_hi);
      let g_u8 = _mm_packus_epi16(g_lo, g_hi);
      let b_u8 = _mm_packus_epi16(b_lo, b_hi);

      let out_ptr = out.as_mut_ptr().add(x * bpp);
      if ALPHA {
        let a_vec = if ALPHA_SRC { a_u8 } else { alpha_u8 };
        write_rgba_16(r_u8, g_u8, b_u8, a_vec, out_ptr);
      } else {
        write_rgb_16(r_u8, g_u8, b_u8, out_ptr);
      }

      x += 16;
    }

    // Scalar tail — remaining < 16 pixels.
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

// ---- thin wrappers -------------------------------------------------------

/// SSE4.1 VUYA / VUYX → packed **RGB** (3 bpp). Alpha byte in source is
/// discarded — RGB output has no alpha channel.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn vuya_to_rgb_row(
  packed: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: SSE4.1 availability is the caller's obligation.
  unsafe {
    vuya_to_rgb_or_rgba_row::<false, false>(packed, rgb_out, width, matrix, full_range);
  }
}

/// SSE4.1 VUYA → packed **RGBA** (4 bpp). Source A byte is passed through
/// verbatim.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn vuya_to_rgba_row(
  packed: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: SSE4.1 availability is the caller's obligation.
  unsafe {
    vuya_to_rgb_or_rgba_row::<true, true>(packed, rgba_out, width, matrix, full_range);
  }
}

/// SSE4.1 VUYX → packed **RGBA** (4 bpp). Source A byte is padding;
/// output α is forced to `0xFF` (opaque).
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn vuyx_to_rgba_row(
  packed: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: SSE4.1 availability is the caller's obligation.
  unsafe {
    vuya_to_rgb_or_rgba_row::<true, false>(packed, rgba_out, width, matrix, full_range);
  }
}

// ---- luma extraction (16 px/iter) ---------------------------------------

/// SSE4.1 VUYA / VUYX → u8 luma. Y is the third byte (offset 2) of each
/// pixel quadruple.
///
/// Byte-identical to `scalar::vuya_to_luma_row`.
///
/// # Safety
///
/// 1. **SSE4.1 must be available.**
/// 2. `packed.len() >= width * 4`.
/// 3. `luma_out.len() >= width`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn vuya_to_luma_row(packed: &[u8], luma_out: &mut [u8], width: usize) {
  debug_assert!(packed.len() >= width * 4, "packed row too short");
  debug_assert!(luma_out.len() >= width, "luma row too short");

  unsafe {
    // Y bytes are at positions 2, 6, 10, 14 within each 16-byte chunk.
    let y_mask = _mm_setr_epi8(2, 6, 10, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);

    let mut x = 0usize;
    while x + 16 <= width {
      let raw0 = _mm_loadu_si128(packed.as_ptr().add(x * 4).cast());
      let raw1 = _mm_loadu_si128(packed.as_ptr().add(x * 4 + 16).cast());
      let raw2 = _mm_loadu_si128(packed.as_ptr().add(x * 4 + 32).cast());
      let raw3 = _mm_loadu_si128(packed.as_ptr().add(x * 4 + 48).cast());

      let y0 = _mm_shuffle_epi8(raw0, y_mask);
      let y1 = _mm_shuffle_epi8(raw1, y_mask);
      let y2 = _mm_shuffle_epi8(raw2, y_mask);
      let y3 = _mm_shuffle_epi8(raw3, y_mask);

      let y_01 = _mm_unpacklo_epi32(y0, y1);
      let y_23 = _mm_unpacklo_epi32(y2, y3);
      let y_vec = _mm_unpacklo_epi64(y_01, y_23);

      _mm_storeu_si128(luma_out.as_mut_ptr().add(x).cast(), y_vec);
      x += 16;
    }

    // Scalar tail.
    if x < width {
      scalar::vuya_to_luma_row(&packed[x * 4..], &mut luma_out[x..], width - x);
    }
  }
}
