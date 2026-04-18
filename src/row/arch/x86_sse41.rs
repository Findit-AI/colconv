//! x86_64 SSE4.1 backend for the row primitives.
//!
//! Selected by [`crate::row`]'s dispatcher as a fallback when AVX2 is
//! not available. SSE4.1 is a wide baseline on x86 (Penryn and newer,
//! ~2008), so this covers essentially all x86 hardware still in
//! production use that lacks AVX2.
//!
//! The kernel carries `#[target_feature(enable = "sse4.1")]` so its
//! intrinsics execute in an explicitly feature‑enabled context. The
//! shared [`super::x86_common::write_rgb_16`] helper uses SSSE3
//! (`_mm_shuffle_epi8`), which is a subset of SSE4.1 and thus
//! available here.
//!
//! # Numerical contract
//!
//! Bit‑identical to
//! [`crate::row::scalar::yuv_420_to_rgb_row`]. All Q15 multiplies
//! are i32‑widened with `(prod + (1 << 14)) >> 15` rounding — same
//! structure as the NEON and AVX2 backends.
//!
//! # Pipeline (per 16 Y pixels / 8 chroma samples)
//!
//! 1. Load 16 Y (`_mm_loadu_si128`) + 8 U + 8 V (low 8 bytes of each
//!    via `_mm_loadl_epi64`).
//! 2. Widen U, V to i16x8 (`_mm_cvtepu8_epi16`), subtract 128.
//! 3. Split each i16x8 into two i32x4 halves and apply `c_scale`.
//! 4. Per channel C ∈ {R, G, B}: `(C_u*u_d + C_v*v_d + RND) >> 15` in
//!    i32, narrow‑saturate to i16x8.
//! 5. Nearest‑neighbor chroma upsample: `_mm_unpacklo_epi16` /
//!    `_mm_unpackhi_epi16` duplicate each of 8 chroma lanes into its
//!    pair slot → two i16x8 vectors covering 16 Y lanes. No lane‑
//!    crossing fixups are needed at 128 bits.
//! 6. Y path: widen low/high 8 Y to i16x8, apply `y_off` / `y_scale`.
//! 7. Saturating i16 add Y + chroma per channel.
//! 8. Saturate‑narrow to u8x16 per channel, then interleave via
//!    `super::x86_common::write_rgb_16`.

use core::arch::x86_64::{
  __m128i, _mm_add_epi32, _mm_adds_epi16, _mm_cvtepi16_epi32, _mm_cvtepu8_epi16, _mm_loadl_epi64,
  _mm_loadu_si128, _mm_mullo_epi32, _mm_packs_epi32, _mm_packus_epi16, _mm_set1_epi16,
  _mm_set1_epi32, _mm_srai_epi32, _mm_srli_si128, _mm_sub_epi16, _mm_unpackhi_epi16,
  _mm_unpacklo_epi16,
};

use crate::{
  ColorMatrix,
  row::{
    arch::x86_common::{swap_rb_16_pixels, write_rgb_16},
    scalar,
  },
};

/// SSE4.1 YUV 4:2:0 → packed RGB. Semantics match
/// [`scalar::yuv_420_to_rgb_row`] byte‑identically.
///
/// # Safety
///
/// The caller must uphold **all** of the following. Violating any
/// causes undefined behavior:
///
/// 1. **SSE4.1 must be available on the current CPU.** The dispatcher
///    in [`crate::row`] verifies this with
///    `is_x86_feature_detected!("sse4.1")` (runtime, std) or
///    `cfg!(target_feature = "sse4.1")` (compile‑time, no‑std).
///    Calling this kernel on a CPU without SSE4.1 triggers an
///    illegal‑instruction trap.
/// 2. `width & 1 == 0` (4:2:0 requires even width).
/// 3. `y.len() >= width`.
/// 4. `u_half.len() >= width / 2`.
/// 5. `v_half.len() >= width / 2`.
/// 6. `rgb_out.len() >= 3 * width`.
///
/// Bounds are verified by `debug_assert` in debug builds; release
/// builds trust the caller because the kernel relies on unchecked
/// pointer arithmetic (`_mm_loadu_si128`, `_mm_loadl_epi64`,
/// `_mm_storeu_si128` inside `write_rgb_16`).
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn yuv_420_to_rgb_row(
  y: &[u8],
  u_half: &[u8],
  v_half: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0, "YUV 4:2:0 requires even width");
  debug_assert!(y.len() >= width);
  debug_assert!(u_half.len() >= width / 2);
  debug_assert!(v_half.len() >= width / 2);
  debug_assert!(rgb_out.len() >= width * 3);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params(full_range);
  const RND: i32 = 1 << 14;

  // SAFETY: SSE4.1 availability is the caller's obligation per the
  // `# Safety` section; the dispatcher in `crate::row` checks it.
  // All pointer adds below are bounded by the `while x + 16 <= width`
  // loop condition and the caller‑promised slice lengths.
  unsafe {
    let rnd_v = _mm_set1_epi32(RND);
    let y_off_v = _mm_set1_epi16(y_off as i16);
    let y_scale_v = _mm_set1_epi32(y_scale);
    let c_scale_v = _mm_set1_epi32(c_scale);
    let mid128 = _mm_set1_epi16(128);
    let cru = _mm_set1_epi32(coeffs.r_u());
    let crv = _mm_set1_epi32(coeffs.r_v());
    let cgu = _mm_set1_epi32(coeffs.g_u());
    let cgv = _mm_set1_epi32(coeffs.g_v());
    let cbu = _mm_set1_epi32(coeffs.b_u());
    let cbv = _mm_set1_epi32(coeffs.b_v());

    let mut x = 0usize;
    while x + 16 <= width {
      // Load 16 Y, 8 U, 8 V.
      let y_vec = _mm_loadu_si128(y.as_ptr().add(x).cast());
      let u_vec = _mm_loadl_epi64(u_half.as_ptr().add(x / 2).cast());
      let v_vec = _mm_loadl_epi64(v_half.as_ptr().add(x / 2).cast());

      // Widen U/V to i16x8 and subtract 128.
      let u_i16 = _mm_sub_epi16(_mm_cvtepu8_epi16(u_vec), mid128);
      let v_i16 = _mm_sub_epi16(_mm_cvtepu8_epi16(v_vec), mid128);

      // Split each i16x8 into two i32x4 halves.
      let u_lo_i32 = _mm_cvtepi16_epi32(u_i16);
      let u_hi_i32 = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(u_i16));
      let v_lo_i32 = _mm_cvtepi16_epi32(v_i16);
      let v_hi_i32 = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(v_i16));

      // u_d, v_d = (u * c_scale + RND) >> 15 — bit‑exact to scalar.
      let u_d_lo = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_lo_i32, c_scale_v), rnd_v));
      let u_d_hi = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_hi_i32, c_scale_v), rnd_v));
      let v_d_lo = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_lo_i32, c_scale_v), rnd_v));
      let v_d_hi = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_hi_i32, c_scale_v), rnd_v));

      // Per‑channel chroma → i16x8 (8 chroma values per channel).
      let r_chroma = chroma_i16x8(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let g_chroma = chroma_i16x8(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let b_chroma = chroma_i16x8(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);

      // Nearest‑neighbor upsample: duplicate each of 8 chroma lanes
      // into its pair slot → two i16x8 vectors covering 16 Y lanes.
      // At 128 bits there's no lane‑crossing issue, so a plain unpack
      // is correct.
      let r_dup_lo = _mm_unpacklo_epi16(r_chroma, r_chroma);
      let r_dup_hi = _mm_unpackhi_epi16(r_chroma, r_chroma);
      let g_dup_lo = _mm_unpacklo_epi16(g_chroma, g_chroma);
      let g_dup_hi = _mm_unpackhi_epi16(g_chroma, g_chroma);
      let b_dup_lo = _mm_unpacklo_epi16(b_chroma, b_chroma);
      let b_dup_hi = _mm_unpackhi_epi16(b_chroma, b_chroma);

      // Y path: widen low/high 8 Y to i16x8, scale.
      let y_low_i16 = _mm_cvtepu8_epi16(y_vec);
      let y_high_i16 = _mm_cvtepu8_epi16(_mm_srli_si128::<8>(y_vec));
      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v);

      // Saturating i16 add Y + chroma per channel.
      let b_lo = _mm_adds_epi16(y_scaled_lo, b_dup_lo);
      let b_hi = _mm_adds_epi16(y_scaled_hi, b_dup_hi);
      let g_lo = _mm_adds_epi16(y_scaled_lo, g_dup_lo);
      let g_hi = _mm_adds_epi16(y_scaled_hi, g_dup_hi);
      let r_lo = _mm_adds_epi16(y_scaled_lo, r_dup_lo);
      let r_hi = _mm_adds_epi16(y_scaled_hi, r_dup_hi);

      // Saturate‑narrow to u8x16 per channel (no lane fixup needed at
      // 128 bits).
      let b_u8 = _mm_packus_epi16(b_lo, b_hi);
      let g_u8 = _mm_packus_epi16(g_lo, g_hi);
      let r_u8 = _mm_packus_epi16(r_lo, r_hi);

      // 3‑way interleave → packed RGB (48 bytes).
      write_rgb_16(r_u8, g_u8, b_u8, rgb_out.as_mut_ptr().add(x * 3));

      x += 16;
    }

    // Scalar tail for the 0..14 leftover pixels.
    if x < width {
      scalar::yuv_420_to_rgb_row(
        &y[x..width],
        &u_half[x / 2..width / 2],
        &v_half[x / 2..width / 2],
        &mut rgb_out[x * 3..width * 3],
        width - x,
        matrix,
        full_range,
      );
    }
  }
}

// ---- helpers (inlined into the target_feature‑enabled caller) ----------

/// `>>_a 15` shift (arithmetic, sign‑extending).
#[inline(always)]
fn q15_shift(v: __m128i) -> __m128i {
  unsafe { _mm_srai_epi32::<15>(v) }
}

/// Computes one i16x8 chroma channel vector from the 4 × i32x4 chroma
/// inputs. Mirrors the scalar
/// `(coeff_u * u_d + coeff_v * v_d + RND) >> 15`, then saturating‑packs
/// to i16x8. No lane fixup needed at 128 bits.
#[inline(always)]
fn chroma_i16x8(
  cu: __m128i,
  cv: __m128i,
  u_d_lo: __m128i,
  v_d_lo: __m128i,
  u_d_hi: __m128i,
  v_d_hi: __m128i,
  rnd: __m128i,
) -> __m128i {
  unsafe {
    let lo = _mm_srai_epi32::<15>(_mm_add_epi32(
      _mm_add_epi32(_mm_mullo_epi32(cu, u_d_lo), _mm_mullo_epi32(cv, v_d_lo)),
      rnd,
    ));
    let hi = _mm_srai_epi32::<15>(_mm_add_epi32(
      _mm_add_epi32(_mm_mullo_epi32(cu, u_d_hi), _mm_mullo_epi32(cv, v_d_hi)),
      rnd,
    ));
    _mm_packs_epi32(lo, hi)
  }
}

/// `(Y - y_off) * y_scale + RND >> 15` applied to an i16x8 vector,
/// returned as i16x8.
#[inline(always)]
fn scale_y(y_i16: __m128i, y_off_v: __m128i, y_scale_v: __m128i, rnd: __m128i) -> __m128i {
  unsafe {
    let shifted = _mm_sub_epi16(y_i16, y_off_v);
    let lo_i32 = _mm_cvtepi16_epi32(shifted);
    let hi_i32 = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(shifted));
    let lo_scaled = _mm_srai_epi32::<15>(_mm_add_epi32(_mm_mullo_epi32(lo_i32, y_scale_v), rnd));
    let hi_scaled = _mm_srai_epi32::<15>(_mm_add_epi32(_mm_mullo_epi32(hi_i32, y_scale_v), rnd));
    _mm_packs_epi32(lo_scaled, hi_scaled)
  }
}

// ===== BGR ↔ RGB byte swap ==============================================

/// SSE4.1 BGR ↔ RGB byte swap. 16 pixels per iteration via the shared
/// [`super::x86_common::swap_rb_16_pixels`] helper (SSSE3 `_mm_shuffle_epi8`
/// underneath). Drives both conversion directions since the swap is
/// self‑inverse.
///
/// # Safety
///
/// 1. SSE4.1 must be available (dispatcher obligation).
/// 2. `input.len() >= 3 * width`.
/// 3. `output.len() >= 3 * width`.
/// 4. `input` / `output` must not alias.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn bgr_rgb_swap_row(input: &[u8], output: &mut [u8], width: usize) {
  debug_assert!(input.len() >= width * 3, "input row too short");
  debug_assert!(output.len() >= width * 3, "output row too short");

  // SAFETY: SSE4.1 is available per caller obligation; SSSE3 (required
  // by `swap_rb_16_pixels`) is a subset. All pointer adds are bounded
  // by the `while x + 16 <= width` condition.
  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      swap_rb_16_pixels(input.as_ptr().add(x * 3), output.as_mut_ptr().add(x * 3));
      x += 16;
    }
    if x < width {
      scalar::bgr_rgb_swap_row(
        &input[x * 3..width * 3],
        &mut output[x * 3..width * 3],
        width - x,
      );
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn check_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
    let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
    let u: std::vec::Vec<u8> = (0..width / 2)
      .map(|i| ((i * 53 + 23) & 0xFF) as u8)
      .collect();
    let v: std::vec::Vec<u8> = (0..width / 2)
      .map(|i| ((i * 71 + 91) & 0xFF) as u8)
      .collect();
    let mut bgr_scalar = std::vec![0u8; width * 3];
    let mut bgr_sse41 = std::vec![0u8; width * 3];

    scalar::yuv_420_to_rgb_row(&y, &u, &v, &mut bgr_scalar, width, matrix, full_range);
    unsafe {
      yuv_420_to_rgb_row(&y, &u, &v, &mut bgr_sse41, width, matrix, full_range);
    }

    if bgr_scalar != bgr_sse41 {
      let first_diff = bgr_scalar
        .iter()
        .zip(bgr_sse41.iter())
        .position(|(a, b)| a != b)
        .unwrap();
      panic!(
        "SSE4.1 diverges from scalar at byte {first_diff} (width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} sse41={}",
        bgr_scalar[first_diff], bgr_sse41[first_diff]
      );
    }
  }

  #[test]
  fn sse41_matches_scalar_all_matrices_16() {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
      return;
    }
    for m in [
      ColorMatrix::Bt601,
      ColorMatrix::Bt709,
      ColorMatrix::Bt2020Ncl,
      ColorMatrix::Smpte240m,
      ColorMatrix::Fcc,
      ColorMatrix::YCgCo,
    ] {
      for full in [true, false] {
        check_equivalence(16, m, full);
      }
    }
  }

  #[test]
  fn sse41_matches_scalar_width_32() {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
      return;
    }
    check_equivalence(32, ColorMatrix::Bt601, true);
    check_equivalence(32, ColorMatrix::Bt709, false);
    check_equivalence(32, ColorMatrix::YCgCo, true);
  }

  #[test]
  fn sse41_matches_scalar_width_1920() {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
      return;
    }
    check_equivalence(1920, ColorMatrix::Bt709, false);
  }

  #[test]
  fn sse41_matches_scalar_odd_tail_widths() {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
      return;
    }
    // Widths that leave a non‑trivial scalar tail (non‑multiple of 16).
    for w in [18usize, 30, 34, 1922] {
      check_equivalence(w, ColorMatrix::Bt601, false);
    }
  }

  // ---- bgr_rgb_swap_row equivalence -----------------------------------

  fn check_swap_equivalence(width: usize) {
    let input: std::vec::Vec<u8> = (0..width * 3)
      .map(|i| ((i * 17 + 41) & 0xFF) as u8)
      .collect();
    let mut out_scalar = std::vec![0u8; width * 3];
    let mut out_sse41 = std::vec![0u8; width * 3];

    scalar::bgr_rgb_swap_row(&input, &mut out_scalar, width);
    unsafe {
      bgr_rgb_swap_row(&input, &mut out_sse41, width);
    }
    assert_eq!(out_scalar, out_sse41, "SSE4.1 swap diverges from scalar");
  }

  #[test]
  fn sse41_swap_matches_scalar() {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
      return;
    }
    for w in [1usize, 15, 16, 17, 31, 32, 33, 1920, 1921] {
      check_swap_equivalence(w);
    }
  }
}
