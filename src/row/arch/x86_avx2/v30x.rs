//! AVX2 V30X (packed YUV 4:4:4, 10-bit) kernels.
//!
//! ## Layout
//!
//! One `u32` per pixel: `bits[11:2]` = U, `bits[21:12]` = Y,
//! `bits[31:22]` = V (2 bits padding at bottom). No chroma subsampling
//! (4:4:4) — each word yields a complete `(U, Y, V)` triple.
//!
//! ## Per-iter pipeline (8 px / 8 u32 / 32 bytes)
//!
//! One `_mm256_loadu_si256` load fetches 8 u32 words. Three
//! `AND + shift` ops extract U / Y / V fields as i32x8. The i32x8
//! halves are packed via `_mm256_packs_epi32` + `_mm256_permute4x64_epi64::<0xD8>`
//! lane fixup to produce natural-order i16x16 vectors, which feed the
//! standard Q15 pipeline.
//!
//! ## 4:4:4 vs. 4:2:2
//!
//! V30X is 4:4:4 — no chroma duplication (`chroma_dup`) is needed.
//! Each pixel has its own unique `(U, Y, V)` triple. We use
//! `chroma_i16x16` with all 8 valid lanes in the low half and 8
//! don't-care lanes in the high half, then consume only the low 8
//! lanes of the result.
//!
//! ## Tail
//!
//! `width % 8` remaining pixels fall through to `scalar::v30x_*`.

use core::arch::x86_64::*;

use super::*;
use crate::{ColorMatrix, row::scalar};

// ---- Bit-extraction helper -----------------------------------------------

/// Unpacks 8 V30X u32 words (one `__m256i`) into three i16x16 vectors
/// holding the 10-bit U / Y / V fields in their low 8 lanes (lanes 8..15
/// are zero / don't-care).
///
/// Strategy: one 256-bit load → three `shift + AND` ops (all in
/// i32x8) → `_mm256_packs_epi32` + `_mm256_permute4x64_epi64::<0xD8>`
/// lane fixup to obtain natural-order i16x16.
///
/// # Safety
///
/// Caller must ensure `ptr` has at least 32 bytes (8 u32) readable, and
/// that `target_feature` includes AVX2.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn unpack_v30x_8px_avx2(ptr: *const u32) -> (__m256i, __m256i, __m256i) {
  // SAFETY: caller obligation — `ptr` has 32 bytes readable; AVX2 is
  // available.
  unsafe {
    let words = _mm256_loadu_si256(ptr.cast());
    let mask = _mm256_set1_epi32(0x3FF);

    // Extract 10-bit fields in i32x8 (values ≤ 1023 — no overflow risk).
    let u_i32 = _mm256_and_si256(_mm256_srli_epi32::<2>(words), mask); // bits [11:2]
    let y_i32 = _mm256_and_si256(_mm256_srli_epi32::<12>(words), mask); // bits [21:12]
    let v_i32 = _mm256_and_si256(_mm256_srli_epi32::<22>(words), mask); // bits [31:22]

    // Pack two i32x8 halves → i16x16. Values ≤ 1023 so no saturation
    // occurs. `_mm256_packs_epi32` interleaves across 128-bit lane
    // boundaries: [lo0..3, hi0..3, lo4..7, hi4..7]. The following
    // `permute4x64_epi64::<0xD8>` reorders 64-bit chunks [0, 2, 1, 3]
    // to give natural [lo0..7, hi0..7] order.
    let zero = _mm256_setzero_si256();
    let u_i16 = _mm256_permute4x64_epi64::<0xD8>(_mm256_packs_epi32(u_i32, zero));
    let y_i16 = _mm256_permute4x64_epi64::<0xD8>(_mm256_packs_epi32(y_i32, zero));
    let v_i16 = _mm256_permute4x64_epi64::<0xD8>(_mm256_packs_epi32(v_i32, zero));

    (u_i16, y_i16, v_i16)
  }
}

// ---- u8 RGB / RGBA output (8 px/iter) ------------------------------------

/// AVX2 V30X → packed u8 RGB or RGBA.
///
/// Byte-identical to `scalar::v30x_to_rgb_or_rgba_row::<ALPHA>`.
///
/// Block size: 8 pixels per SIMD iteration (one `_mm256_loadu_si256`).
///
/// # Safety
///
/// 1. **AVX2 must be available.**
/// 2. `packed.len() >= width`.
/// 3. `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
#[inline]
#[target_feature(enable = "avx2")]
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
    while x + 8 <= width {
      // Unpack 8 V30X words → three i16x16 with valid data in lanes 0..7.
      let (u_i16, y_i16, v_i16) = unpack_v30x_8px_avx2(packed.as_ptr().add(x));

      // Subtract chroma bias (512 for 10-bit).
      let u_sub = _mm256_sub_epi16(u_i16, bias_v);
      let v_sub = _mm256_sub_epi16(v_i16, bias_v);

      // Widen to i32x8 halves for Q15 scale (only low half has valid data;
      // high half is zero / don't-care from the zero-packed packs above).
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

      // chroma_i16x16: 16-lane vector; only lanes 0..7 carry valid data
      // (V30X is 4:4:4 — no duplication needed). Lanes 8..15 are
      // don't-care and discarded by the 8-pixel partial store below.
      let r_chroma = chroma_i16x16(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let g_chroma = chroma_i16x16(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let b_chroma = chroma_i16x16(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);

      // Y scale: V30X Y ≤ 1023 fits i16 — use scale_y (not scale_y_u16_avx2).
      let y_scaled = scale_y(y_i16, y_off_v, y_scale_v, rnd_v);

      // u8 narrow with saturation. Low 16 bytes per channel hold valid
      // results; high 16 bytes (from zero hi arg) are zero.
      let zero = _mm256_setzero_si256();
      let r_u8 = narrow_u8x32(_mm256_adds_epi16(y_scaled, r_chroma), zero);
      let g_u8 = narrow_u8x32(_mm256_adds_epi16(y_scaled, g_chroma), zero);
      let b_u8 = narrow_u8x32(_mm256_adds_epi16(y_scaled, b_chroma), zero);

      // 8-pixel partial store via stack buffer + scalar interleave.
      let mut r_tmp = [0u8; 32];
      let mut g_tmp = [0u8; 32];
      let mut b_tmp = [0u8; 32];
      _mm256_storeu_si256(r_tmp.as_mut_ptr().cast(), r_u8);
      _mm256_storeu_si256(g_tmp.as_mut_ptr().cast(), g_u8);
      _mm256_storeu_si256(b_tmp.as_mut_ptr().cast(), b_u8);

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

/// AVX2 V30X → packed native-depth u16 RGB or RGBA (low-bit-packed at
/// 10-bit).
///
/// Byte-identical to `scalar::v30x_to_rgb_u16_or_rgba_u16_row::<ALPHA>`.
///
/// Block size: 8 pixels per SIMD iteration.
///
/// # Safety
///
/// 1. **AVX2 must be available.**
/// 2. `packed.len() >= width`.
/// 3. `out.len() >= width * (if ALPHA { 4 } else { 3 })` (u16 elements).
#[inline]
#[target_feature(enable = "avx2")]
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
    while x + 8 <= width {
      let (u_i16, y_i16, v_i16) = unpack_v30x_8px_avx2(packed.as_ptr().add(x));

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

      // 10-bit chroma: i32 arithmetic is sufficient (no overflow at 10-bit).
      let r_chroma = chroma_i16x16(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let g_chroma = chroma_i16x16(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let b_chroma = chroma_i16x16(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);

      // V30X Y ≤ 1023 fits i16 — use scale_y (not scale_y_u16_avx2).
      let y_scaled = scale_y(y_i16, y_off_v, y_scale_v, rnd_v);

      // Clamp to [0, 0x3FF] (native 10-bit range).
      let r = clamp_u16_max_x16(_mm256_adds_epi16(y_scaled, r_chroma), zero_v, max_v);
      let g = clamp_u16_max_x16(_mm256_adds_epi16(y_scaled, g_chroma), zero_v, max_v);
      let b = clamp_u16_max_x16(_mm256_adds_epi16(y_scaled, b_chroma), zero_v, max_v);

      // 8-pixel u16 store via stack buffer + scalar interleave.
      let mut r_tmp = [0u16; 16];
      let mut g_tmp = [0u16; 16];
      let mut b_tmp = [0u16; 16];
      _mm256_storeu_si256(r_tmp.as_mut_ptr().cast(), r);
      _mm256_storeu_si256(g_tmp.as_mut_ptr().cast(), g);
      _mm256_storeu_si256(b_tmp.as_mut_ptr().cast(), b);

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

// ---- Luma u8 (8 px/iter) -------------------------------------------------

/// AVX2 V30X → u8 luma. Y is `(word >> 12) & 0x3FF`, then `>> 2`.
///
/// Byte-identical to `scalar::v30x_to_luma_row`.
///
/// Block size: 8 pixels per SIMD iteration.
///
/// # Safety
///
/// 1. **AVX2 must be available.**
/// 2. `packed.len() >= width`.
/// 3. `out.len() >= width`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn v30x_to_luma_row(packed: &[u32], out: &mut [u8], width: usize) {
  debug_assert!(packed.len() >= width);
  debug_assert!(out.len() >= width);

  // SAFETY: AVX2 availability is the caller's obligation.
  unsafe {
    let mask = _mm256_set1_epi32(0x3FF);

    let mut x = 0usize;
    while x + 8 <= width {
      let words = _mm256_loadu_si256(packed.as_ptr().add(x).cast());

      // Y = (word >> 12) & 0x3FF for each i32 lane.
      let y_i32 = _mm256_and_si256(_mm256_srli_epi32::<12>(words), mask);

      // Pack i32x8 to i16x16 (values ≤ 1023, no saturation); fix lane order.
      let zero = _mm256_setzero_si256();
      let y_i16 = _mm256_permute4x64_epi64::<0xD8>(_mm256_packs_epi32(y_i32, zero));

      // Downshift 10-bit Y by 2 → 8-bit, narrow to u8 via packus.
      let y_shr = _mm256_srli_epi16::<2>(y_i16);
      let y_u8 = narrow_u8x32(y_shr, zero);

      // Store first 8 of the 32 u8 lanes via stack buffer + copy_from_slice.
      let mut tmp = [0u8; 32];
      _mm256_storeu_si256(tmp.as_mut_ptr().cast(), y_u8);
      out[x..x + 8].copy_from_slice(&tmp[..8]);

      x += 8;
    }

    // Scalar tail — remaining < 8 pixels.
    if x < width {
      scalar::v30x_to_luma_row(&packed[x..width], &mut out[x..width], width - x);
    }
  }
}

// ---- Luma u16 (8 px/iter) ------------------------------------------------

/// AVX2 V30X → u16 luma (low-bit-packed at 10-bit). Each output `u16`
/// carries the source's 10-bit Y value in its low 10 bits.
///
/// Byte-identical to `scalar::v30x_to_luma_u16_row`.
///
/// Block size: 8 pixels per SIMD iteration.
///
/// # Safety
///
/// 1. **AVX2 must be available.**
/// 2. `packed.len() >= width`.
/// 3. `out.len() >= width`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn v30x_to_luma_u16_row(packed: &[u32], out: &mut [u16], width: usize) {
  debug_assert!(packed.len() >= width);
  debug_assert!(out.len() >= width);

  // SAFETY: AVX2 availability is the caller's obligation.
  unsafe {
    let mask = _mm256_set1_epi32(0x3FF);

    let mut x = 0usize;
    while x + 8 <= width {
      let words = _mm256_loadu_si256(packed.as_ptr().add(x).cast());

      // Y = (word >> 12) & 0x3FF for each i32 lane.
      let y_i32 = _mm256_and_si256(_mm256_srli_epi32::<12>(words), mask);

      // Pack i32x8 to i16x16 (values ≤ 1023); fix lane order.
      let zero = _mm256_setzero_si256();
      let y_i16 = _mm256_permute4x64_epi64::<0xD8>(_mm256_packs_epi32(y_i32, zero));

      // Store first 8 of the 16 u16 lanes via stack buffer + copy_from_slice.
      let mut tmp = [0u16; 16];
      _mm256_storeu_si256(tmp.as_mut_ptr().cast(), y_i16);
      out[x..x + 8].copy_from_slice(&tmp[..8]);

      x += 8;
    }

    // Scalar tail — remaining < 8 pixels.
    if x < width {
      scalar::v30x_to_luma_u16_row(&packed[x..width], &mut out[x..width], width - x);
    }
  }
}
