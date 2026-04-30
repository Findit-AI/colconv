//! AVX2 Y216 (packed YUV 4:2:2, BITS=16) kernels.
//!
//! Layout per row: u16 quadruples `(Y₀, U, Y₁, V)` where each
//! sample occupies the full 16-bit word (no MSB/LSB alignment shift
//! needed — unlike Y210/Y212 which require `>> (16 - BITS)`).
//!
//! ## u8 pipeline (32 px / iter)
//!
//! Four `_mm256_loadu_si256` loads fetch 64 u16 = 128 bytes = 32 pixels.
//! Two loads cover pixels x..x+15 (lo group) and two cover x+16..x+31
//! (hi group). Within each group the same deinterleave as y2xx.rs but
//! without the right-shift: `_mm256_shuffle_epi8` + `_mm256_permute4x64_epi64`
//! + `_mm256_permute2x128_si256` separate Y (16 u16) and chroma (8 UV pairs).
//!
//! Chroma arithmetic via `chroma_i16x16` (i32 widening, Q15); Y via
//! `scale_y_u16_avx2` (unsigned-widened to avoid i16 overflow for Y > 32767).
//! Output via `write_rgb_32` / `write_rgba_32`.
//!
//! ## u16 pipeline (16 px / iter)
//!
//! i64 chroma arithmetic via `chroma_i64x4_avx2` + `scale_y_i32x8_i64`,
//! mirroring `yuv_420p16_to_rgb_or_rgba_u16_row`. Two `_mm256_loadu_si256`
//! loads cover 16 pixels; the deinterleave gives 16 Y and 8 UV pairs.
//!
//! ## Tail
//!
//! `width % 32` (u8) or `width % 16` (u16) → `scalar::y216_*` fallback.

use core::arch::x86_64::*;

use super::*;
use crate::{ColorMatrix, row::scalar};

// ---- Deinterleave helper (shared by u8 and u16 paths) -------------------

/// Deinterleaves 16 YUYV u16 quadruples (= 32 u16 = 64 bytes) loaded
/// from two `_mm256_loadu_si256` into:
/// - `y_vec`:      16 u16 Y samples  (`[Y0..Y15]`).
/// - `u_vec_8`:    8  u16 U samples  (lanes 0..7 valid, high 8 garbage).
/// - `v_vec_8`:    8  u16 V samples  (lanes 0..7 valid, high 8 garbage).
///
/// No right-shift is applied — Y216 samples are full-range 16-bit.
///
/// # Safety
///
/// `ptr` must point to at least 64 readable bytes (32 u16). Caller's
/// `target_feature` must include AVX2.
#[inline(always)]
unsafe fn unpack_y216_16px_avx2(ptr: *const u16) -> (__m256i, __m256i, __m256i) {
  unsafe {
    // Load 32 u16 = 64 bytes = 16 pixels (2 × __m256i).
    // v0 = [Y0,U0,Y1,V0, Y2,U1,Y3,V1,  Y4,U2,Y5,V2, Y6,U3,Y7,V3]
    // v1 = [Y8,U4,Y9,V4, Y10,U5,Y11,V5, Y12,U6,Y13,V6, Y14,U7,Y15,V7]
    let v0 = _mm256_loadu_si256(ptr.cast());
    let v1 = _mm256_loadu_si256(ptr.add(16).cast());

    // Per-128-bit-lane shuffle: even u16 (Y) → low 8 bytes, odd u16 (chroma)
    // → high 8 bytes. Same as y2xx's `split_idx`.
    let split_idx = _mm256_setr_epi8(
      0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15, // low lane
      0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15, // high lane
    );
    let v0s = _mm256_shuffle_epi8(v0, split_idx);
    let v1s = _mm256_shuffle_epi8(v1, split_idx);

    // 0xD8 permute: [A, B, C, D] → [A, C, B, D] — all Y in low 128, chroma in high 128.
    let v0p = _mm256_permute4x64_epi64::<0xD8>(v0s);
    let v1p = _mm256_permute4x64_epi64::<0xD8>(v1s);

    // Cross-vector merge.
    let y_vec = _mm256_permute2x128_si256::<0x20>(v0p, v1p); // [Y0..Y15]
    let chroma_raw = _mm256_permute2x128_si256::<0x31>(v0p, v1p); // [U0,V0,U1,V1, ..., U7,V7]

    // Split U / V from interleaved chroma using per-lane shuffle.
    // chroma per 128-bit lane = 8 u16: [U,V,U,V, U,V,U,V].
    // 0x88 = [0, 2, 0, 2] → low 128 = [U0..U3, U4..U7].
    let u_idx = _mm256_setr_epi8(
      0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1,
      0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1,
    );
    let v_idx = _mm256_setr_epi8(
      2, 3, 6, 7, 10, 11, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1,
      2, 3, 6, 7, 10, 11, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1,
    );
    let u_per_lane = _mm256_shuffle_epi8(chroma_raw, u_idx);
    let v_per_lane = _mm256_shuffle_epi8(chroma_raw, v_idx);
    // 0x88 = [0, 2, 0, 2]: pack lane0_low + lane1_low → low 128 (U0..U7 / V0..V7).
    let u_vec = _mm256_permute4x64_epi64::<0x88>(u_per_lane);
    let v_vec = _mm256_permute4x64_epi64::<0x88>(v_per_lane);

    (y_vec, u_vec, v_vec)
  }
}

// ---- u8 output (i32 chroma, 32 px/iter) ---------------------------------

/// AVX2 Y216 → packed u8 RGB or RGBA.
///
/// Byte-identical to `scalar::y216_to_rgb_or_rgba_row::<ALPHA>`.
///
/// Block size: 32 pixels per SIMD iteration (four `_mm256_loadu_si256`).
///
/// # Safety
///
/// 1. **AVX2 must be available.**
/// 2. `width % 2 == 0`.
/// 3. `packed.len() >= width * 2`.
/// 4. `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
#[allow(dead_code)] // dispatcher lands in follow-up PR (Task 9)
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn y216_to_rgb_or_rgba_row<const ALPHA: bool>(
  packed: &[u16],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(width.is_multiple_of(2), "Y216 requires even width");
  debug_assert!(packed.len() >= width * 2);
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  // range_params_n::<16, 8> → y_off is i32 (full range or limited).
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<16, 8>(full_range);
  const RND: i32 = 1 << 14;

  // SAFETY: AVX2 availability is the caller's obligation.
  unsafe {
    let rnd_v = _mm256_set1_epi32(RND);
    // y_off as i32 — scale_y_u16_avx2 takes i32x8 y_off.
    let y_off_v = _mm256_set1_epi32(y_off);
    let y_scale_v = _mm256_set1_epi32(y_scale);
    let c_scale_v = _mm256_set1_epi32(c_scale);
    // Chroma bias: 32768 via wrapping 0x8000 = -32768i16.
    let bias16_v = _mm256_set1_epi16(-32768i16);
    let cru = _mm256_set1_epi32(coeffs.r_u());
    let crv = _mm256_set1_epi32(coeffs.r_v());
    let cgu = _mm256_set1_epi32(coeffs.g_u());
    let cgv = _mm256_set1_epi32(coeffs.g_v());
    let cbu = _mm256_set1_epi32(coeffs.b_u());
    let cbv = _mm256_set1_epi32(coeffs.b_v());
    let alpha_u8 = _mm256_set1_epi8(-1i8);

    let mut x = 0usize;
    while x + 32 <= width {
      // --- lo group: pixels x..x+15 (two 256-bit loads, 16 pixels) ------
      let (y_lo_vec, u_lo_vec, v_lo_vec) =
        unpack_y216_16px_avx2(packed.as_ptr().add(x * 2));

      // Chroma bias subtraction (wrapping).
      let u_lo_i16 = _mm256_sub_epi16(u_lo_vec, bias16_v);
      let v_lo_i16 = _mm256_sub_epi16(v_lo_vec, bias16_v);

      // Widen 8 valid chroma i16 lanes to two i32x8 halves.
      // Only the low 128 bits of u_lo_vec carry valid U0..U7;
      // the high 128 bits are zeroed by the 0x88 permute (don't-care).
      let u_lo_a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(u_lo_i16));
      let u_lo_b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(u_lo_i16));
      let v_lo_a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v_lo_i16));
      let v_lo_b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(v_lo_i16));

      let u_d_lo_a = q15_shift(_mm256_add_epi32(_mm256_mullo_epi32(u_lo_a, c_scale_v), rnd_v));
      let u_d_lo_b = q15_shift(_mm256_add_epi32(_mm256_mullo_epi32(u_lo_b, c_scale_v), rnd_v));
      let v_d_lo_a = q15_shift(_mm256_add_epi32(_mm256_mullo_epi32(v_lo_a, c_scale_v), rnd_v));
      let v_d_lo_b = q15_shift(_mm256_add_epi32(_mm256_mullo_epi32(v_lo_b, c_scale_v), rnd_v));

      // chroma_i16x16: 16-lane vector with valid data in lanes 0..7 (lo).
      let r_chroma_lo = chroma_i16x16(cru, crv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let g_chroma_lo = chroma_i16x16(cgu, cgv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let b_chroma_lo = chroma_i16x16(cbu, cbv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);

      // Duplicate each chroma into its 4:2:2 Y-pair slot.
      // chroma_dup returns (lo16, hi16); only lo16 (lanes 0..15) is used
      // here since we have only 8 chroma samples per 16-px half.
      let (r_dup_lo, _) = chroma_dup(r_chroma_lo);
      let (g_dup_lo, _) = chroma_dup(g_chroma_lo);
      let (b_dup_lo, _) = chroma_dup(b_chroma_lo);

      // Y scale: unsigned-widened to avoid i16 overflow for Y > 32767.
      let y_lo_scaled = scale_y_u16_avx2(y_lo_vec, y_off_v, y_scale_v, rnd_v);

      // --- hi group: pixels x+16..x+31 -----------------------------------
      let (y_hi_vec, u_hi_vec, v_hi_vec) =
        unpack_y216_16px_avx2(packed.as_ptr().add(x * 2 + 32));

      let u_hi_i16 = _mm256_sub_epi16(u_hi_vec, bias16_v);
      let v_hi_i16 = _mm256_sub_epi16(v_hi_vec, bias16_v);

      let u_hi_a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(u_hi_i16));
      let u_hi_b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(u_hi_i16));
      let v_hi_a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v_hi_i16));
      let v_hi_b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(v_hi_i16));

      let u_d_hi_a = q15_shift(_mm256_add_epi32(_mm256_mullo_epi32(u_hi_a, c_scale_v), rnd_v));
      let u_d_hi_b = q15_shift(_mm256_add_epi32(_mm256_mullo_epi32(u_hi_b, c_scale_v), rnd_v));
      let v_d_hi_a = q15_shift(_mm256_add_epi32(_mm256_mullo_epi32(v_hi_a, c_scale_v), rnd_v));
      let v_d_hi_b = q15_shift(_mm256_add_epi32(_mm256_mullo_epi32(v_hi_b, c_scale_v), rnd_v));

      let r_chroma_hi = chroma_i16x16(cru, crv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let g_chroma_hi = chroma_i16x16(cgu, cgv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let b_chroma_hi = chroma_i16x16(cbu, cbv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);

      let (r_dup_hi, _) = chroma_dup(r_chroma_hi);
      let (g_dup_hi, _) = chroma_dup(g_chroma_hi);
      let (b_dup_hi, _) = chroma_dup(b_chroma_hi);

      let y_hi_scaled = scale_y_u16_avx2(y_hi_vec, y_off_v, y_scale_v, rnd_v);

      // Saturating add + narrow to u8x32 (32 pixels per channel).
      let r_u8 = narrow_u8x32(
        _mm256_adds_epi16(y_lo_scaled, r_dup_lo),
        _mm256_adds_epi16(y_hi_scaled, r_dup_hi),
      );
      let g_u8 = narrow_u8x32(
        _mm256_adds_epi16(y_lo_scaled, g_dup_lo),
        _mm256_adds_epi16(y_hi_scaled, g_dup_hi),
      );
      let b_u8 = narrow_u8x32(
        _mm256_adds_epi16(y_lo_scaled, b_dup_lo),
        _mm256_adds_epi16(y_hi_scaled, b_dup_hi),
      );

      if ALPHA {
        write_rgba_32(r_u8, g_u8, b_u8, alpha_u8, out.as_mut_ptr().add(x * 4));
      } else {
        write_rgb_32(r_u8, g_u8, b_u8, out.as_mut_ptr().add(x * 3));
      }

      x += 32;
    }

    // Scalar tail — remaining < 32 pixels.
    if x < width {
      let tail_packed = &packed[x * 2..width * 2];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      scalar::y216_to_rgb_or_rgba_row::<ALPHA>(tail_packed, tail_out, tail_w, matrix, full_range);
    }
  }
}

// ---- u16 output (i64 chroma, 16 px/iter) --------------------------------

/// AVX2 Y216 → packed native-depth u16 RGB or RGBA.
///
/// Uses i64 chroma (`chroma_i64x4_avx2`) to avoid overflow at 16-bit scales.
/// Byte-identical to `scalar::y216_to_rgb_u16_or_rgba_u16_row::<ALPHA>`.
///
/// Block size: 16 pixels per SIMD iteration.
///
/// # Safety
///
/// 1. **AVX2 must be available.**
/// 2. `width % 2 == 0`.
/// 3. `packed.len() >= width * 2`.
/// 4. `out.len() >= width * (if ALPHA { 4 } else { 3 })` (u16 elements).
#[allow(dead_code)] // dispatcher lands in follow-up PR (Task 9)
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn y216_to_rgb_u16_or_rgba_u16_row<const ALPHA: bool>(
  packed: &[u16],
  out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(width.is_multiple_of(2), "Y216 requires even width");
  debug_assert!(packed.len() >= width * 2);
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<16, 16>(full_range);
  const RND: i64 = 1 << 14;

  // SAFETY: AVX2 availability is the caller's obligation.
  unsafe {
    let alpha_u16 = _mm_set1_epi16(-1i16);
    let rnd_v = _mm256_set1_epi64x(RND);
    let rnd32_v = _mm256_set1_epi32(1 << 14);
    let y_off_v = _mm256_set1_epi32(y_off);
    let y_scale_v = _mm256_set1_epi32(y_scale);
    let c_scale_v = _mm256_set1_epi32(c_scale);
    // Chroma bias via wrapping 0x8000 trick.
    let bias16_v = _mm256_set1_epi16(-32768i16);
    let cru = _mm256_set1_epi32(coeffs.r_u());
    let crv = _mm256_set1_epi32(coeffs.r_v());
    let cgu = _mm256_set1_epi32(coeffs.g_u());
    let cgv = _mm256_set1_epi32(coeffs.g_v());
    let cbu = _mm256_set1_epi32(coeffs.b_u());
    let cbv = _mm256_set1_epi32(coeffs.b_v());

    let mut x = 0usize;
    while x + 16 <= width {
      // Two 256-bit loads → 16 pixels, 8 UV pairs.
      let (y_vec, u_vec, v_vec) = unpack_y216_16px_avx2(packed.as_ptr().add(x * 2));

      // Subtract chroma bias.
      let u_i16 = _mm256_sub_epi16(u_vec, bias16_v);
      let v_i16 = _mm256_sub_epi16(v_vec, bias16_v);

      // Widen 8 valid chroma i16 lanes to i32x8.
      // Low 128 of u_vec / v_vec hold U0..U7 / V0..V7 after 0x88 permute.
      let u_i32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(u_i16));
      let v_i32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v_i16));

      // Scale UV in i32 (8 lanes; |chroma_centered × c_scale| fits i32).
      let u_d = q15_shift(_mm256_add_epi32(_mm256_mullo_epi32(u_i32, c_scale_v), rnd32_v));
      let v_d = q15_shift(_mm256_add_epi32(_mm256_mullo_epi32(v_i32, c_scale_v), rnd32_v));

      // i64 chroma: even/odd i32 lanes via 0xF5 shuffle.
      let u_d_odd = _mm256_shuffle_epi32::<0xF5>(u_d);
      let v_d_odd = _mm256_shuffle_epi32::<0xF5>(v_d);

      let r_ch_even = chroma_i64x4_avx2(cru, crv, u_d, v_d, rnd_v);
      let r_ch_odd = chroma_i64x4_avx2(cru, crv, u_d_odd, v_d_odd, rnd_v);
      let g_ch_even = chroma_i64x4_avx2(cgu, cgv, u_d, v_d, rnd_v);
      let g_ch_odd = chroma_i64x4_avx2(cgu, cgv, u_d_odd, v_d_odd, rnd_v);
      let b_ch_even = chroma_i64x4_avx2(cbu, cbv, u_d, v_d, rnd_v);
      let b_ch_odd = chroma_i64x4_avx2(cbu, cbv, u_d_odd, v_d_odd, rnd_v);

      // Reassemble i64x4 pairs → i32x8 [c0..c7].
      let r_ch_i32 = reassemble_i64x4_to_i32x8(r_ch_even, r_ch_odd);
      let g_ch_i32 = reassemble_i64x4_to_i32x8(g_ch_even, g_ch_odd);
      let b_ch_i32 = reassemble_i64x4_to_i32x8(b_ch_even, b_ch_odd);

      // Duplicate each of 8 chroma values into 2 per-pixel slots (4:2:2).
      let (r_dup_lo, r_dup_hi) = chroma_dup_i32(r_ch_i32);
      let (g_dup_lo, g_dup_hi) = chroma_dup_i32(g_ch_i32);
      let (b_dup_lo, b_dup_hi) = chroma_dup_i32(b_ch_i32);

      // Y: unsigned-widen u16 → i32, subtract y_off, scale via i64.
      // y_vec from unpack_y216_16px_avx2 is __m256i with 16 u16 lanes.
      let y_lo_u16 = _mm256_castsi256_si128(y_vec);
      let y_hi_u16 = _mm256_extracti128_si256::<1>(y_vec);
      let y_lo_i32 = _mm256_sub_epi32(_mm256_cvtepu16_epi32(y_lo_u16), y_off_v);
      let y_hi_i32 = _mm256_sub_epi32(_mm256_cvtepu16_epi32(y_hi_u16), y_off_v);

      let y_lo_scaled = scale_y_i32x8_i64(y_lo_i32, y_scale_v, rnd_v);
      let y_hi_scaled = scale_y_i32x8_i64(y_hi_i32, y_scale_v, rnd_v);

      // Add Y + chroma, saturate to u16 via _mm256_packus_epi32 + 0xD8 fixup.
      let r_u16 = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi32(
        _mm256_add_epi32(y_lo_scaled, r_dup_lo),
        _mm256_add_epi32(y_hi_scaled, r_dup_hi),
      ));
      let g_u16 = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi32(
        _mm256_add_epi32(y_lo_scaled, g_dup_lo),
        _mm256_add_epi32(y_hi_scaled, g_dup_hi),
      ));
      let b_u16 = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi32(
        _mm256_add_epi32(y_lo_scaled, b_dup_lo),
        _mm256_add_epi32(y_hi_scaled, b_dup_hi),
      ));

      // Write 16 pixels via two 8-pixel helpers.
      if ALPHA {
        let dst = out.as_mut_ptr().add(x * 4);
        write_rgba_u16_8(
          _mm256_castsi256_si128(r_u16),
          _mm256_castsi256_si128(g_u16),
          _mm256_castsi256_si128(b_u16),
          alpha_u16,
          dst,
        );
        write_rgba_u16_8(
          _mm256_extracti128_si256::<1>(r_u16),
          _mm256_extracti128_si256::<1>(g_u16),
          _mm256_extracti128_si256::<1>(b_u16),
          alpha_u16,
          dst.add(32),
        );
      } else {
        let dst = out.as_mut_ptr().add(x * 3);
        write_rgb_u16_8(
          _mm256_castsi256_si128(r_u16),
          _mm256_castsi256_si128(g_u16),
          _mm256_castsi256_si128(b_u16),
          dst,
        );
        write_rgb_u16_8(
          _mm256_extracti128_si256::<1>(r_u16),
          _mm256_extracti128_si256::<1>(g_u16),
          _mm256_extracti128_si256::<1>(b_u16),
          dst.add(24),
        );
      }

      x += 16;
    }

    // Scalar tail — remaining < 16 pixels.
    if x < width {
      let tail_packed = &packed[x * 2..width * 2];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      scalar::y216_to_rgb_u16_or_rgba_u16_row::<ALPHA>(
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

/// AVX2 Y216 → u8 luma. Extracts Y via `>> 8`.
///
/// Byte-identical to `scalar::y216_to_luma_row`.
///
/// Block size: 32 pixels per SIMD iteration.
///
/// # Safety
///
/// 1. **AVX2 must be available.**
/// 2. `width % 2 == 0`.
/// 3. `packed.len() >= width * 2`.
/// 4. `out.len() >= width`.
#[allow(dead_code)] // dispatcher lands in follow-up PR (Task 9)
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn y216_to_luma_row(packed: &[u16], out: &mut [u8], width: usize) {
  debug_assert!(width.is_multiple_of(2));
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width);

  // SAFETY: AVX2 availability is the caller's obligation.
  unsafe {
    // Per-lane Y permute mask: pick even u16 lanes (low byte first) into
    // the low 8 bytes of each 128-bit lane; high 8 bytes zeroed.
    let split_idx = _mm256_setr_epi8(
      0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1, // low lane
      0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1, // high lane
    );

    let mut x = 0usize;
    while x + 32 <= width {
      // Four 256-bit loads: v0/v1 for pixels x..x+15, v2/v3 for x+16..x+31.
      let v0 = _mm256_loadu_si256(packed.as_ptr().add(x * 2).cast());
      let v1 = _mm256_loadu_si256(packed.as_ptr().add(x * 2 + 16).cast());
      let v2 = _mm256_loadu_si256(packed.as_ptr().add(x * 2 + 32).cast());
      let v3 = _mm256_loadu_si256(packed.as_ptr().add(x * 2 + 48).cast());

      // Per-lane shuffle → Y into low 64-bit chunk of each 128-bit lane.
      let v0s = _mm256_shuffle_epi8(v0, split_idx);
      let v1s = _mm256_shuffle_epi8(v1, split_idx);
      let v2s = _mm256_shuffle_epi8(v2, split_idx);
      let v3s = _mm256_shuffle_epi8(v3, split_idx);

      // 0x88 = [0, 2, 0, 2]: pack low 64-bit chunks (lane0 + lane1) into low 128 bits.
      let v0p = _mm256_permute4x64_epi64::<0x88>(v0s);
      let v1p = _mm256_permute4x64_epi64::<0x88>(v1s);
      let v2p = _mm256_permute4x64_epi64::<0x88>(v2s);
      let v3p = _mm256_permute4x64_epi64::<0x88>(v3s);

      // Cross-vector merge: lo 128 of v0p + lo 128 of v1p → Y0..Y15 (16 u16).
      let y_lo = _mm256_permute2x128_si256::<0x20>(v0p, v1p); // [Y0..Y15]
      let y_hi = _mm256_permute2x128_si256::<0x20>(v2p, v3p); // [Y16..Y31]

      // `>> 8` to obtain u8 luma (high byte of each Y u16 sample).
      // `_mm256_srli_epi16::<8>` has a literal const count.
      let y_lo_shr = _mm256_srli_epi16::<8>(y_lo);
      let y_hi_shr = _mm256_srli_epi16::<8>(y_hi);

      // Narrow 32 × i16 → 32 × u8. narrow_u8x32 already applies 0xD8 lane fixup.
      let y_u8 = narrow_u8x32(y_lo_shr, y_hi_shr);
      _mm256_storeu_si256(out.as_mut_ptr().add(x).cast(), y_u8);

      x += 32;
    }

    // Scalar tail — remaining < 32 pixels.
    if x < width {
      let tail_packed = &packed[x * 2..width * 2];
      let tail_out = &mut out[x..width];
      let tail_w = width - x;
      scalar::y216_to_luma_row(tail_packed, tail_out, tail_w);
    }
  }
}

// ---- Luma u16 (32 px/iter) ----------------------------------------------

/// AVX2 Y216 → u16 luma. Direct copy of Y samples (no shift).
///
/// Byte-identical to `scalar::y216_to_luma_u16_row`.
///
/// Block size: 32 pixels per SIMD iteration.
///
/// # Safety
///
/// 1. **AVX2 must be available.**
/// 2. `width % 2 == 0`.
/// 3. `packed.len() >= width * 2`.
/// 4. `out.len() >= width`.
#[allow(dead_code)] // dispatcher lands in follow-up PR (Task 9)
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn y216_to_luma_u16_row(packed: &[u16], out: &mut [u16], width: usize) {
  debug_assert!(width.is_multiple_of(2));
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width);

  // SAFETY: AVX2 availability is the caller's obligation.
  unsafe {
    // Per-lane Y permute mask (same as luma_row above).
    let split_idx = _mm256_setr_epi8(
      0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1,
      0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1,
    );

    let mut x = 0usize;
    while x + 32 <= width {
      let v0 = _mm256_loadu_si256(packed.as_ptr().add(x * 2).cast());
      let v1 = _mm256_loadu_si256(packed.as_ptr().add(x * 2 + 16).cast());
      let v2 = _mm256_loadu_si256(packed.as_ptr().add(x * 2 + 32).cast());
      let v3 = _mm256_loadu_si256(packed.as_ptr().add(x * 2 + 48).cast());

      let v0s = _mm256_shuffle_epi8(v0, split_idx);
      let v1s = _mm256_shuffle_epi8(v1, split_idx);
      let v2s = _mm256_shuffle_epi8(v2, split_idx);
      let v3s = _mm256_shuffle_epi8(v3, split_idx);

      let v0p = _mm256_permute4x64_epi64::<0x88>(v0s);
      let v1p = _mm256_permute4x64_epi64::<0x88>(v1s);
      let v2p = _mm256_permute4x64_epi64::<0x88>(v2s);
      let v3p = _mm256_permute4x64_epi64::<0x88>(v3s);

      let y_lo = _mm256_permute2x128_si256::<0x20>(v0p, v1p); // [Y0..Y15]
      let y_hi = _mm256_permute2x128_si256::<0x20>(v2p, v3p); // [Y16..Y31]

      // Direct store — full 16-bit Y values, no shift.
      _mm256_storeu_si256(out.as_mut_ptr().add(x).cast(), y_lo);
      _mm256_storeu_si256(out.as_mut_ptr().add(x + 16).cast(), y_hi);

      x += 32;
    }

    // Scalar tail — remaining < 32 pixels.
    if x < width {
      let tail_packed = &packed[x * 2..width * 2];
      let tail_out = &mut out[x..width];
      let tail_w = width - x;
      scalar::y216_to_luma_u16_row(tail_packed, tail_out, tail_w);
    }
  }
}
