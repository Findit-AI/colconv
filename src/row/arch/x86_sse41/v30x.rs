//! SSE4.1 V30X (packed YUV 4:4:4, 10-bit) kernels.
//!
//! ## Layout
//!
//! One `u32` per pixel: `bits[11:2]` = U, `bits[21:12]` = Y,
//! `bits[31:22]` = V (2 bits padding at bottom, 2 bits padding at top).
//! No chroma subsampling (4:4:4) — each word yields a complete `(U, Y, V)`
//! triple.
//!
//! ## Per-iter pipeline (8 px / 8 u32 / 32 bytes)
//!
//! Two `_mm_loadu_si128` loads fetch 8 u32 words (4 pixels each).
//! For each 4-pixel batch, three `shift+AND` ops extract U / Y / V
//! fields as i32x4. The two i32x4 halves are packed via
//! `_mm_packs_epi32` into i16x8 for the 8-lane Q15 pipeline.
//!
//! ## 4:4:4 vs. 4:2:2
//!
//! V30X is 4:4:4 — no chroma duplication (`_mm_unpacklo_epi16`) is
//! needed. Each pixel has its own unique `(U, Y, V)` triple.
//!
//! ## Tail
//!
//! `width % 8` remaining pixels fall through to `scalar::v30x_*`.

#[allow(unused_imports)]
use core::arch::x86_64::*;

use super::*;
use crate::{ColorMatrix, row::scalar};

// ---- u8 RGB / RGBA output (8 px/iter) -----------------------------------

/// SSE4.1 V30X → packed u8 RGB or RGBA.
///
/// Byte-identical to `scalar::v30x_to_rgb_or_rgba_row::<ALPHA>`.
///
/// # Safety
///
/// 1. **SSE4.1 must be available.**
/// 2. `packed.len() >= width`.
/// 3. `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn v30x_to_rgb_or_rgba_row<const ALPHA: bool>(
  packed: &[u32],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(packed.len() >= width, "packed row too short");
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp, "out row too short");

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<10, 8>(full_range);
  let bias = scalar::chroma_bias::<10>();
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
    let mask = _mm_set1_epi32(0x3FF);

    let mut x = 0usize;
    while x + 8 <= width {
      // Load 8 V30X words = 8 pixels (32 bytes = 2 × __m128i).
      let words_lo = _mm_loadu_si128(packed.as_ptr().add(x).cast());
      let words_hi = _mm_loadu_si128(packed.as_ptr().add(x + 4).cast());

      // Extract U (bits 11:2), Y (bits 21:12), V (bits 31:22) for each
      // 4-pixel batch as i32x4. Values ≤ 1023 — safe for i16.
      let u_lo_i32 = _mm_and_si128(_mm_srli_epi32::<2>(words_lo), mask);
      let y_lo_i32 = _mm_and_si128(_mm_srli_epi32::<12>(words_lo), mask);
      let v_lo_i32 = _mm_and_si128(_mm_srli_epi32::<22>(words_lo), mask);

      let u_hi_i32 = _mm_and_si128(_mm_srli_epi32::<2>(words_hi), mask);
      let y_hi_i32 = _mm_and_si128(_mm_srli_epi32::<12>(words_hi), mask);
      let v_hi_i32 = _mm_and_si128(_mm_srli_epi32::<22>(words_hi), mask);

      // Pack two i32x4 halves into i16x8 (values ≤ 1023, no saturation).
      let u_i16 = _mm_packs_epi32(u_lo_i32, u_hi_i32);
      let y_i16 = _mm_packs_epi32(y_lo_i32, y_hi_i32);
      let v_i16 = _mm_packs_epi32(v_lo_i32, v_hi_i32);

      // Subtract chroma bias (512 for 10-bit).
      let u_sub = _mm_sub_epi16(u_i16, bias_v);
      let v_sub = _mm_sub_epi16(v_i16, bias_v);

      // Widen to i32x4 lo/hi for Q15 scale.
      let u_d_lo_i32 = _mm_cvtepi16_epi32(u_sub);
      let u_d_hi_i32 = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(u_sub));
      let v_d_lo_i32 = _mm_cvtepi16_epi32(v_sub);
      let v_d_hi_i32 = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(v_sub));

      let u_d_lo = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_d_lo_i32, c_scale_v), rnd_v));
      let u_d_hi = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_d_hi_i32, c_scale_v), rnd_v));
      let v_d_lo = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_d_lo_i32, c_scale_v), rnd_v));
      let v_d_hi = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_d_hi_i32, c_scale_v), rnd_v));

      // 8-lane chroma vectors (all 8 lanes valid — 4:4:4, no duplication).
      let r_chroma = chroma_i16x8(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let g_chroma = chroma_i16x8(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let b_chroma = chroma_i16x8(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);

      // Y scale: V30X Y ≤ 1023 fits in i16 — use scale_y (not scale_y_u16).
      let y_scaled = scale_y(y_i16, y_off_v, y_scale_v, rnd_v);

      // u8 narrow with saturation. Low 8 bytes per channel hold valid
      // results; high 8 bytes (from _mm_setzero_si128 hi arg) are zero.
      let zero = _mm_setzero_si128();
      let r_u8 = _mm_packus_epi16(_mm_adds_epi16(y_scaled, r_chroma), zero);
      let g_u8 = _mm_packus_epi16(_mm_adds_epi16(y_scaled, g_chroma), zero);
      let b_u8 = _mm_packus_epi16(_mm_adds_epi16(y_scaled, b_chroma), zero);

      // 8-pixel partial store via stack buffer + scalar interleave.
      let mut r_tmp = [0u8; 16];
      let mut g_tmp = [0u8; 16];
      let mut b_tmp = [0u8; 16];
      _mm_storeu_si128(r_tmp.as_mut_ptr().cast(), r_u8);
      _mm_storeu_si128(g_tmp.as_mut_ptr().cast(), g_u8);
      _mm_storeu_si128(b_tmp.as_mut_ptr().cast(), b_u8);

      if ALPHA {
        let dst = &mut out[x * 4..x * 4 + 8 * 4];
        for i in 0..8 {
          dst[i * 4] = r_tmp[i];
          dst[i * 4 + 1] = g_tmp[i];
          dst[i * 4 + 2] = b_tmp[i];
          dst[i * 4 + 3] = 0xFF;
        }
      } else {
        let dst = &mut out[x * 3..x * 3 + 8 * 3];
        for i in 0..8 {
          dst[i * 3] = r_tmp[i];
          dst[i * 3 + 1] = g_tmp[i];
          dst[i * 3 + 2] = b_tmp[i];
        }
      }

      x += 8;
    }

    // Scalar tail — remaining < 8 pixels.
    if x < width {
      let tail_packed = &packed[x..width];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      scalar::v30x_to_rgb_or_rgba_row::<ALPHA>(tail_packed, tail_out, tail_w, matrix, full_range);
    }
  }
}

// ---- u16 RGB / RGBA native-depth output (8 px/iter) ---------------------

/// SSE4.1 V30X → packed native-depth u16 RGB or RGBA (low-bit-packed at
/// 10-bit).
///
/// Byte-identical to `scalar::v30x_to_rgb_u16_or_rgba_u16_row::<ALPHA>`.
///
/// # Safety
///
/// 1. **SSE4.1 must be available.**
/// 2. `packed.len() >= width`.
/// 3. `out.len() >= width * (if ALPHA { 4 } else { 3 })` (u16 elements).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn v30x_to_rgb_u16_or_rgba_u16_row<const ALPHA: bool>(
  packed: &[u32],
  out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(packed.len() >= width, "packed row too short");
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp, "out row too short");

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<10, 10>(full_range);
  let bias = scalar::chroma_bias::<10>();
  const RND: i32 = 1 << 14;
  let out_max: i16 = ((1i32 << 10) - 1) as i16; // 0x3FF

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
    let mask = _mm_set1_epi32(0x3FF);

    let mut x = 0usize;
    while x + 8 <= width {
      let words_lo = _mm_loadu_si128(packed.as_ptr().add(x).cast());
      let words_hi = _mm_loadu_si128(packed.as_ptr().add(x + 4).cast());

      let u_lo_i32 = _mm_and_si128(_mm_srli_epi32::<2>(words_lo), mask);
      let y_lo_i32 = _mm_and_si128(_mm_srli_epi32::<12>(words_lo), mask);
      let v_lo_i32 = _mm_and_si128(_mm_srli_epi32::<22>(words_lo), mask);

      let u_hi_i32 = _mm_and_si128(_mm_srli_epi32::<2>(words_hi), mask);
      let y_hi_i32 = _mm_and_si128(_mm_srli_epi32::<12>(words_hi), mask);
      let v_hi_i32 = _mm_and_si128(_mm_srli_epi32::<22>(words_hi), mask);

      let u_i16 = _mm_packs_epi32(u_lo_i32, u_hi_i32);
      let y_i16 = _mm_packs_epi32(y_lo_i32, y_hi_i32);
      let v_i16 = _mm_packs_epi32(v_lo_i32, v_hi_i32);

      let u_sub = _mm_sub_epi16(u_i16, bias_v);
      let v_sub = _mm_sub_epi16(v_i16, bias_v);

      let u_d_lo_i32 = _mm_cvtepi16_epi32(u_sub);
      let u_d_hi_i32 = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(u_sub));
      let v_d_lo_i32 = _mm_cvtepi16_epi32(v_sub);
      let v_d_hi_i32 = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(v_sub));

      let u_d_lo = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_d_lo_i32, c_scale_v), rnd_v));
      let u_d_hi = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_d_hi_i32, c_scale_v), rnd_v));
      let v_d_lo = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_d_lo_i32, c_scale_v), rnd_v));
      let v_d_hi = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_d_hi_i32, c_scale_v), rnd_v));

      // 10-bit chroma: i32 arithmetic is sufficient (no overflow at 10-bit).
      let r_chroma = chroma_i16x8(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let g_chroma = chroma_i16x8(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let b_chroma = chroma_i16x8(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);

      // V30X Y ≤ 1023 fits in i16 — use scale_y (not scale_y_u16).
      let y_scaled = scale_y(y_i16, y_off_v, y_scale_v, rnd_v);

      // Clamp to [0, 0x3FF] (native 10-bit range).
      let r = clamp_u16_max(_mm_adds_epi16(y_scaled, r_chroma), zero_v, max_v);
      let g = clamp_u16_max(_mm_adds_epi16(y_scaled, g_chroma), zero_v, max_v);
      let b = clamp_u16_max(_mm_adds_epi16(y_scaled, b_chroma), zero_v, max_v);

      // 8-pixel u16 store via stack buffer + scalar interleave.
      let mut r_tmp = [0u16; 8];
      let mut g_tmp = [0u16; 8];
      let mut b_tmp = [0u16; 8];
      _mm_storeu_si128(r_tmp.as_mut_ptr().cast(), r);
      _mm_storeu_si128(g_tmp.as_mut_ptr().cast(), g);
      _mm_storeu_si128(b_tmp.as_mut_ptr().cast(), b);

      if ALPHA {
        let dst = &mut out[x * 4..x * 4 + 8 * 4];
        let alpha = out_max as u16; // 0x3FF
        for i in 0..8 {
          dst[i * 4] = r_tmp[i];
          dst[i * 4 + 1] = g_tmp[i];
          dst[i * 4 + 2] = b_tmp[i];
          dst[i * 4 + 3] = alpha;
        }
      } else {
        let dst = &mut out[x * 3..x * 3 + 8 * 3];
        for i in 0..8 {
          dst[i * 3] = r_tmp[i];
          dst[i * 3 + 1] = g_tmp[i];
          dst[i * 3 + 2] = b_tmp[i];
        }
      }

      x += 8;
    }

    // Scalar tail — remaining < 8 pixels.
    if x < width {
      let tail_packed = &packed[x..width];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      scalar::v30x_to_rgb_u16_or_rgba_u16_row::<ALPHA>(
        tail_packed,
        tail_out,
        tail_w,
        matrix,
        full_range,
      );
    }
  }
}

// ---- Luma u8 (8 px/iter) ------------------------------------------------

/// SSE4.1 V30X → u8 luma. Y is `(word >> 12) & 0x3FF`, then `>> 2`.
///
/// Byte-identical to `scalar::v30x_to_luma_row`.
///
/// # Safety
///
/// 1. **SSE4.1 must be available.**
/// 2. `packed.len() >= width`.
/// 3. `out.len() >= width`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn v30x_to_luma_row(packed: &[u32], out: &mut [u8], width: usize) {
  debug_assert!(packed.len() >= width);
  debug_assert!(out.len() >= width);

  unsafe {
    let mask = _mm_set1_epi32(0x3FF);

    let mut x = 0usize;
    while x + 8 <= width {
      let words_lo = _mm_loadu_si128(packed.as_ptr().add(x).cast());
      let words_hi = _mm_loadu_si128(packed.as_ptr().add(x + 4).cast());

      // Y = (word >> 12) & 0x3FF for each lane.
      let y_lo_i32 = _mm_and_si128(_mm_srli_epi32::<12>(words_lo), mask);
      let y_hi_i32 = _mm_and_si128(_mm_srli_epi32::<12>(words_hi), mask);

      // Pack two i32x4 into i16x8 (values ≤ 1023, no saturation).
      let y_i16 = _mm_packs_epi32(y_lo_i32, y_hi_i32);

      // Downshift 10-bit Y by 2 → 8-bit, narrow to u8x8 via packus.
      let y_shr = _mm_srli_epi16::<2>(y_i16);
      let y_u8 = _mm_packus_epi16(y_shr, _mm_setzero_si128());

      // Store 8 of the 16 lanes via stack buffer + copy_from_slice.
      let mut tmp = [0u8; 16];
      _mm_storeu_si128(tmp.as_mut_ptr().cast(), y_u8);
      out[x..x + 8].copy_from_slice(&tmp[..8]);

      x += 8;
    }

    // Scalar tail.
    if x < width {
      scalar::v30x_to_luma_row(&packed[x..width], &mut out[x..width], width - x);
    }
  }
}

// ---- Luma u16 (8 px/iter) -----------------------------------------------

/// SSE4.1 V30X → u16 luma (low-bit-packed at 10-bit). Each output `u16`
/// carries the source's 10-bit Y value in its low 10 bits.
///
/// Byte-identical to `scalar::v30x_to_luma_u16_row`.
///
/// # Safety
///
/// 1. **SSE4.1 must be available.**
/// 2. `packed.len() >= width`.
/// 3. `out.len() >= width`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn v30x_to_luma_u16_row(packed: &[u32], out: &mut [u16], width: usize) {
  debug_assert!(packed.len() >= width);
  debug_assert!(out.len() >= width);

  unsafe {
    let mask = _mm_set1_epi32(0x3FF);

    let mut x = 0usize;
    while x + 8 <= width {
      let words_lo = _mm_loadu_si128(packed.as_ptr().add(x).cast());
      let words_hi = _mm_loadu_si128(packed.as_ptr().add(x + 4).cast());

      // Y = (word >> 12) & 0x3FF for each lane.
      let y_lo_i32 = _mm_and_si128(_mm_srli_epi32::<12>(words_lo), mask);
      let y_hi_i32 = _mm_and_si128(_mm_srli_epi32::<12>(words_hi), mask);

      // Pack to i16x8 (values ≤ 1023, safe).
      let y_i16 = _mm_packs_epi32(y_lo_i32, y_hi_i32);

      // Direct store of 8 × u16 (10-bit values already in low bits).
      _mm_storeu_si128(out.as_mut_ptr().add(x).cast(), y_i16);

      x += 8;
    }

    // Scalar tail.
    if x < width {
      scalar::v30x_to_luma_u16_row(&packed[x..width], &mut out[x..width], width - x);
    }
  }
}
