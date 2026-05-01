//! AVX-512 V410 (packed YUV 4:4:4, 10-bit) kernels.
//!
//! ## Layout
//!
//! One `u32` per pixel: `bits[9:0]` = U, `bits[19:10]` = Y,
//! `bits[29:20]` = V (2 bits padding at top). No chroma subsampling
//! (4:4:4) — each word yields a complete `(U, Y, V)` triple.
//!
//! ## Per-iter pipeline (16 px / 16 u32 / 64 bytes)
//!
//! One `_mm512_loadu_si512` load fetches 16 u32 words (= one `__m512i`).
//! Three `AND + srli_epi32` ops extract U / Y / V as i32x16. The 16-lane
//! i32 vectors are narrowed directly via `_mm512_cvtepi32_epi16` (no
//! permute needed — AVX-512 intrinsic truncates lane-by-lane), producing
//! 16-lane i16 in the **low** 16 lanes of the i16x32 register (high 16
//! lanes are zero / don't-care). These feed the standard AVX-512 Q15
//! pipeline helpers ([`chroma_i16x32`], [`scale_y`]).
//!
//! ## 4:4:4 vs. 4:2:2
//!
//! V410 is 4:4:4 — no chroma duplication (`chroma_dup`) is needed.
//! Each pixel has its own unique `(U, Y, V)` triple. We use
//! `chroma_i16x32` with all 16 valid lanes in the low half and 16
//! don't-care lanes in the high half, then consume only the low 16
//! lanes of the result via stack buffer + scalar interleave.
//!
//! ## Tail
//!
//! `width % 16` remaining pixels fall through to `scalar::v410_*`.

use core::arch::x86_64::*;

use super::*;
use crate::{ColorMatrix, row::scalar};

// ---- Bit-extraction helper -----------------------------------------------

/// Unpacks 16 V410 u32 words (one `__m512i`) into three i16x32 vectors
/// holding the 10-bit U / Y / V fields in their low 16 lanes (lanes 16..32
/// are zero / don't-care).
///
/// Strategy: one 512-bit load → three `AND + srli_epi32` ops in i32x16 →
/// `_mm512_cvtepi32_epi16` direct narrow (no saturation — values ≤ 1023)
/// to produce natural-order i16 in the low 16 lanes.
///
/// # Safety
///
/// Caller must ensure `ptr` has at least 64 bytes (16 u32) readable, and
/// that `target_feature` includes AVX-512F + AVX-512BW.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn unpack_v410_16px_avx512(ptr: *const u32) -> (__m512i, __m512i, __m512i) {
  // SAFETY: caller obligation — `ptr` has 64 bytes readable; AVX-512F
  // + AVX-512BW are available.
  unsafe {
    let words = _mm512_loadu_si512(ptr.cast());
    let mask = _mm512_set1_epi32(0x3FF);

    // Extract 10-bit fields in i32x16 (values ≤ 1023 — no overflow risk).
    let u_i32 = _mm512_and_si512(words, mask);                          // bits [9:0]
    let y_i32 = _mm512_and_si512(_mm512_srli_epi32::<10>(words), mask); // bits [19:10]
    let v_i32 = _mm512_and_si512(_mm512_srli_epi32::<20>(words), mask); // bits [29:20]

    // Narrow i32x16 → i16x16 (low 16 lanes) via direct truncation.
    // Values ≤ 1023 fit in i16 — `_mm512_cvtepi32_epi16` returns
    // a __m256i but we widen it to __m512i via zero-extension.
    let u_i16 = _mm512_castsi256_si512(_mm512_cvtepi32_epi16(u_i32));
    let y_i16 = _mm512_castsi256_si512(_mm512_cvtepi32_epi16(y_i32));
    let v_i16 = _mm512_castsi256_si512(_mm512_cvtepi32_epi16(v_i32));

    (u_i16, y_i16, v_i16)
  }
}

// ---- u8 RGB / RGBA output (16 px/iter) -----------------------------------

/// AVX-512 V410 → packed u8 RGB or RGBA.
///
/// Byte-identical to `scalar::v410_to_rgb_or_rgba_row::<ALPHA>`.
///
/// Block size: 16 pixels per SIMD iteration (one `_mm512_loadu_si512`).
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available.**
/// 2. `packed.len() >= width`.
/// 3. `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn v410_to_rgb_or_rgba_row<const ALPHA: bool>(
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
    while x + 16 <= width {
      // Unpack 16 V410 words → three i16x32 with valid data in lanes 0..16.
      let (u_i16, y_i16, v_i16) = unpack_v410_16px_avx512(packed.as_ptr().add(x));

      // Subtract chroma bias (512 for 10-bit).
      let u_sub = _mm512_sub_epi16(u_i16, bias_v);
      let v_sub = _mm512_sub_epi16(v_i16, bias_v);

      // Widen low 16 lanes (valid) and high 16 lanes (don't-care) to
      // i32x16 halves for Q15 scale.
      let u_lo_i32 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_sub));
      let u_hi_i32 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(u_sub));
      let v_lo_i32 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_sub));
      let v_hi_i32 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(v_sub));

      let u_d_lo = q15_shift(_mm512_add_epi32(_mm512_mullo_epi32(u_lo_i32, c_scale_v), rnd_v));
      let u_d_hi = q15_shift(_mm512_add_epi32(_mm512_mullo_epi32(u_hi_i32, c_scale_v), rnd_v));
      let v_d_lo = q15_shift(_mm512_add_epi32(_mm512_mullo_epi32(v_lo_i32, c_scale_v), rnd_v));
      let v_d_hi = q15_shift(_mm512_add_epi32(_mm512_mullo_epi32(v_hi_i32, c_scale_v), rnd_v));

      // chroma_i16x32: 32-lane vector; lanes 0..16 carry valid data
      // (V410 is 4:4:4 — no duplication needed). Lanes 16..32 are
      // don't-care and discarded by the 16-pixel partial store below.
      let r_chroma = chroma_i16x32(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v, pack_fixup);
      let g_chroma = chroma_i16x32(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v, pack_fixup);
      let b_chroma = chroma_i16x32(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v, pack_fixup);

      // Y scale: V410 Y ≤ 1023 fits i16 — use scale_y (not scale_y_u16_avx512).
      let y_scaled = scale_y(y_i16, y_off_v, y_scale_v, rnd_v, pack_fixup);

      // u8 narrow with saturation. Low 16 bytes per channel hold valid
      // results; high 16 bytes (from the zero-extended upper half) are zero.
      let zero = _mm512_setzero_si512();
      let r_u8 = narrow_u8x64(_mm512_adds_epi16(y_scaled, r_chroma), zero, pack_fixup);
      let g_u8 = narrow_u8x64(_mm512_adds_epi16(y_scaled, g_chroma), zero, pack_fixup);
      let b_u8 = narrow_u8x64(_mm512_adds_epi16(y_scaled, b_chroma), zero, pack_fixup);

      // 16-pixel partial store via stack buffer + scalar interleave.
      let mut r_tmp = [0u8; 64];
      let mut g_tmp = [0u8; 64];
      let mut b_tmp = [0u8; 64];
      _mm512_storeu_si512(r_tmp.as_mut_ptr().cast(), r_u8);
      _mm512_storeu_si512(g_tmp.as_mut_ptr().cast(), g_u8);
      _mm512_storeu_si512(b_tmp.as_mut_ptr().cast(), b_u8);

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
      let tail_packed = &packed[x..width];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      scalar::v410_to_rgb_or_rgba_row::<ALPHA>(tail_packed, tail_out, tail_w, matrix, full_range);
    }
  }
}

// ---- u16 RGB / RGBA native-depth output (16 px/iter) --------------------

/// AVX-512 V410 → packed native-depth u16 RGB or RGBA (low-bit-packed at
/// 10-bit).
///
/// Byte-identical to `scalar::v410_to_rgb_u16_or_rgba_u16_row::<ALPHA>`.
///
/// Block size: 16 pixels per SIMD iteration.
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available.**
/// 2. `packed.len() >= width`.
/// 3. `out.len() >= width * (if ALPHA { 4 } else { 3 })` (u16 elements).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn v410_to_rgb_u16_or_rgba_u16_row<const ALPHA: bool>(
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
    while x + 16 <= width {
      let (u_i16, y_i16, v_i16) = unpack_v410_16px_avx512(packed.as_ptr().add(x));

      let u_sub = _mm512_sub_epi16(u_i16, bias_v);
      let v_sub = _mm512_sub_epi16(v_i16, bias_v);

      let u_lo_i32 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_sub));
      let u_hi_i32 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(u_sub));
      let v_lo_i32 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_sub));
      let v_hi_i32 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(v_sub));

      let u_d_lo = q15_shift(_mm512_add_epi32(_mm512_mullo_epi32(u_lo_i32, c_scale_v), rnd_v));
      let u_d_hi = q15_shift(_mm512_add_epi32(_mm512_mullo_epi32(u_hi_i32, c_scale_v), rnd_v));
      let v_d_lo = q15_shift(_mm512_add_epi32(_mm512_mullo_epi32(v_lo_i32, c_scale_v), rnd_v));
      let v_d_hi = q15_shift(_mm512_add_epi32(_mm512_mullo_epi32(v_hi_i32, c_scale_v), rnd_v));

      // 10-bit chroma: i32 arithmetic is sufficient (no overflow at 10-bit).
      let r_chroma = chroma_i16x32(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v, pack_fixup);
      let g_chroma = chroma_i16x32(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v, pack_fixup);
      let b_chroma = chroma_i16x32(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v, pack_fixup);

      // V410 Y ≤ 1023 fits i16 — use scale_y (not scale_y_u16_avx512).
      let y_scaled = scale_y(y_i16, y_off_v, y_scale_v, rnd_v, pack_fixup);

      // Clamp to [0, 0x3FF] (native 10-bit range).
      let r = clamp_u16_max_x32(_mm512_adds_epi16(y_scaled, r_chroma), zero_v, max_v);
      let g = clamp_u16_max_x32(_mm512_adds_epi16(y_scaled, g_chroma), zero_v, max_v);
      let b = clamp_u16_max_x32(_mm512_adds_epi16(y_scaled, b_chroma), zero_v, max_v);

      // 16-pixel u16 store via stack buffer + scalar interleave.
      let mut r_tmp = [0u16; 32];
      let mut g_tmp = [0u16; 32];
      let mut b_tmp = [0u16; 32];
      _mm512_storeu_si512(r_tmp.as_mut_ptr().cast(), r);
      _mm512_storeu_si512(g_tmp.as_mut_ptr().cast(), g);
      _mm512_storeu_si512(b_tmp.as_mut_ptr().cast(), b);

      if ALPHA {
        let dst = &mut out[x * 4..x * 4 + 16 * 4];
        let alpha = out_max as u16; // 0x3FF
        for i in 0..16 {
          dst[i * 4] = r_tmp[i];
          dst[i * 4 + 1] = g_tmp[i];
          dst[i * 4 + 2] = b_tmp[i];
          dst[i * 4 + 3] = alpha;
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
      let tail_packed = &packed[x..width];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      scalar::v410_to_rgb_u16_or_rgba_u16_row::<ALPHA>(
        tail_packed,
        tail_out,
        tail_w,
        matrix,
        full_range,
      );
    }
  }
}

// ---- Luma u8 (16 px/iter) ------------------------------------------------

/// AVX-512 V410 → u8 luma. Y is `(word >> 10) & 0x3FF`, then `>> 2`.
///
/// Byte-identical to `scalar::v410_to_luma_row`.
///
/// Block size: 16 pixels per SIMD iteration.
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available.**
/// 2. `packed.len() >= width`.
/// 3. `out.len() >= width`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn v410_to_luma_row(packed: &[u32], out: &mut [u8], width: usize) {
  debug_assert!(packed.len() >= width);
  debug_assert!(out.len() >= width);

  // SAFETY: AVX-512F + AVX-512BW availability is the caller's obligation.
  unsafe {
    let mask = _mm512_set1_epi32(0x3FF);
    let pack_fixup = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);

    let mut x = 0usize;
    while x + 16 <= width {
      let words = _mm512_loadu_si512(packed.as_ptr().add(x).cast());

      // Y = (word >> 10) & 0x3FF for each i32 lane.
      let y_i32 = _mm512_and_si512(_mm512_srli_epi32::<10>(words), mask);

      // Narrow i32x16 → i16 in low 16 lanes of a __m512i.
      let y_i16 = _mm512_castsi256_si512(_mm512_cvtepi32_epi16(y_i32));

      // Downshift 10-bit Y by 2 → 8-bit. Then narrow with zero hi half
      // to u8x64 (only first 16 bytes valid).
      let y_shr = _mm512_srli_epi16::<2>(y_i16);
      let zero = _mm512_setzero_si512();
      let y_u8 = narrow_u8x64(y_shr, zero, pack_fixup);

      // Store first 16 of the u8x64 lanes via stack buffer + copy_from_slice.
      let mut tmp = [0u8; 64];
      _mm512_storeu_si512(tmp.as_mut_ptr().cast(), y_u8);
      out[x..x + 16].copy_from_slice(&tmp[..16]);

      x += 16;
    }

    // Scalar tail — remaining < 16 pixels.
    if x < width {
      scalar::v410_to_luma_row(&packed[x..width], &mut out[x..width], width - x);
    }
  }
}

// ---- Luma u16 (16 px/iter) -----------------------------------------------

/// AVX-512 V410 → u16 luma (low-bit-packed at 10-bit). Each output `u16`
/// carries the source's 10-bit Y value in its low 10 bits.
///
/// Byte-identical to `scalar::v410_to_luma_u16_row`.
///
/// Block size: 16 pixels per SIMD iteration.
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available.**
/// 2. `packed.len() >= width`.
/// 3. `out.len() >= width`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn v410_to_luma_u16_row(packed: &[u32], out: &mut [u16], width: usize) {
  debug_assert!(packed.len() >= width);
  debug_assert!(out.len() >= width);

  // SAFETY: AVX-512F + AVX-512BW availability is the caller's obligation.
  unsafe {
    let mask = _mm512_set1_epi32(0x3FF);

    let mut x = 0usize;
    while x + 16 <= width {
      let words = _mm512_loadu_si512(packed.as_ptr().add(x).cast());

      // Y = (word >> 10) & 0x3FF for each i32 lane.
      let y_i32 = _mm512_and_si512(_mm512_srli_epi32::<10>(words), mask);

      // Narrow i32x16 → i16x16 in low 16 lanes of __m512i.
      let y_i16 = _mm512_castsi256_si512(_mm512_cvtepi32_epi16(y_i32));

      // Store first 16 of the 32 u16 lanes via stack buffer + copy_from_slice.
      let mut tmp = [0u16; 32];
      _mm512_storeu_si512(tmp.as_mut_ptr().cast(), y_i16);
      out[x..x + 16].copy_from_slice(&tmp[..16]);

      x += 16;
    }

    // Scalar tail — remaining < 16 pixels.
    if x < width {
      scalar::v410_to_luma_u16_row(&packed[x..width], &mut out[x..width], width - x);
    }
  }
}
