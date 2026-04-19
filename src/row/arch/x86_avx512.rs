//! x86_64 AVX‑512 backend (F + BW) for the row primitives.
//!
//! Selected by [`crate::row`]'s dispatcher after
//! `is_x86_feature_detected!("avx512bw")` returns true (runtime,
//! std‑gated) or `cfg!(target_feature = "avx512bw")` evaluates true
//! (compile‑time, no‑std). The kernel carries
//! `#[target_feature(enable = "avx512f,avx512bw")]` so its intrinsics
//! execute in an explicitly feature‑enabled context.
//!
//! Requires AVX‑512F (foundation) and AVX‑512BW (byte/word integer
//! ops). All real AVX‑512 CPUs have both — Intel Skylake‑X / Cascade
//! Lake / Ice Lake / Sapphire Rapids Xeons, AMD Zen 4+ (Genoa,
//! Ryzen 7000+).
//!
//! # Numerical contract
//!
//! Bit‑identical to
//! [`crate::row::scalar::yuv_420_to_rgb_row`]. All Q15 multiplies
//! are i32‑widened with `(prod + (1 << 14)) >> 15` rounding — same
//! structure as the NEON / SSE4.1 / AVX2 backends.
//!
//! # Pipeline (per 64 Y pixels / 32 chroma samples)
//!
//! 1. Load 64 Y (`_mm512_loadu_si512`) + 32 U + 32 V (`_mm256_loadu_si256`).
//! 2. Widen U, V to i16x32 (`_mm512_cvtepu8_epi16`), subtract 128.
//! 3. Split each i16x32 into two i32x16 halves and apply `c_scale`.
//! 4. Per channel C ∈ {R, G, B}: `(C_u*u_d + C_v*v_d + RND) >> 15` in
//!    i32, narrow‑saturate to i16x32.
//! 5. Nearest‑neighbor chroma upsample: duplicate each of the 32 chroma
//!    lanes into its pair slot → two i16x32 vectors covering 64 Y lanes.
//! 6. Y path: widen 64 Y to two i16x32 vectors, apply `y_off` / `y_scale`.
//! 7. Saturating i16 add Y + chroma per channel.
//! 8. Saturate‑narrow to u8x64 per channel, then interleave as packed
//!    RGB via four calls to the shared [`super::x86_common::write_rgb_16`]
//!    (192 output bytes = 4 × 48).
//!
//! # AVX‑512 lane‑crossing fixups
//!
//! AVX‑512 registers act as four 128‑bit lanes for most of the ops we
//! use. `_mm512_packs_epi32`, `_mm512_packus_epi16`, and
//! `_mm512_unpack{lo,hi}_epi16` all operate per 128‑bit lane,
//! producing lane‑split results.
//!
//! - **Pack fixup** (shared by `packs_epi32` → i16x32 and
//!   `packus_epi16` → u8x64): after either pack, 64‑bit lane order is
//!   `[lo0, hi0, lo1, hi1, lo2, hi2, lo3, hi3]`. Permute via
//!   `_mm512_permutexvar_epi64` with index `[0, 2, 4, 6, 1, 3, 5, 7]`
//!   restores natural `[lo0..3 contiguous, hi0..3 contiguous]`.
//! - **Chroma‑dup fixup**: `unpacklo`/`unpackhi` each produce per‑lane
//!   duplicated pairs but the halves for a given Y block are split
//!   across lanes. `_mm512_permutex2var_epi64` with indices
//!   `[0,1,8,9,2,3,10,11]` and `[4,5,12,13,6,7,14,15]` rebuilds the
//!   two 32‑Y‑block‑aligned vectors from unpacklo + unpackhi.

use core::arch::x86_64::{
  __m128i, __m512i, _mm_setr_epi8, _mm256_loadu_si256, _mm512_add_epi32, _mm512_adds_epi16,
  _mm512_broadcast_i32x4, _mm512_castsi512_si128, _mm512_castsi512_si256, _mm512_cvtepi16_epi32,
  _mm512_cvtepu8_epi16, _mm512_extracti32x4_epi32, _mm512_extracti64x4_epi64, _mm512_loadu_si512,
  _mm512_max_epi16, _mm512_min_epi16, _mm512_mullo_epi32, _mm512_packs_epi32, _mm512_packus_epi16,
  _mm512_permutex2var_epi64, _mm512_permutexvar_epi64, _mm512_set1_epi16, _mm512_set1_epi32,
  _mm512_setr_epi64, _mm512_shuffle_epi8, _mm512_srai_epi32, _mm512_sub_epi16,
  _mm512_unpackhi_epi16, _mm512_unpacklo_epi16,
};

use crate::{
  ColorMatrix,
  row::{
    arch::x86_common::{rgb_to_hsv_16_pixels, swap_rb_16_pixels, write_rgb_16, write_rgb_u16_8},
    scalar,
  },
};

/// AVX‑512 YUV 4:2:0 → packed RGB. Semantics match
/// [`scalar::yuv_420_to_rgb_row`] byte‑identically.
///
/// # Safety
///
/// The caller must uphold **all** of the following. Violating any
/// causes undefined behavior:
///
/// 1. **AVX‑512F + AVX‑512BW must be available on the current CPU.**
///    The dispatcher in [`crate::row`] verifies this with
///    `is_x86_feature_detected!("avx512bw")` (runtime, std) or
///    `cfg!(target_feature = "avx512bw")` (compile‑time, no‑std).
///    AVX‑512BW implies AVX‑512F on all real CPUs. Calling this kernel
///    on a CPU without AVX‑512BW triggers an illegal‑instruction trap.
/// 2. `width & 1 == 0` (4:2:0 requires even width).
/// 3. `y.len() >= width`.
/// 4. `u_half.len() >= width / 2`.
/// 5. `v_half.len() >= width / 2`.
/// 6. `rgb_out.len() >= 3 * width`.
///
/// Bounds are verified by `debug_assert` in debug builds; release
/// builds trust the caller because the kernel relies on unchecked
/// pointer arithmetic (`_mm512_loadu_si512`, `_mm256_loadu_si256`,
/// `_mm_storeu_si128` inside `write_rgb_16`).
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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

  // SAFETY: AVX‑512BW availability is the caller's obligation per the
  // `# Safety` section; the dispatcher in `crate::row` checks it.
  // All pointer adds below are bounded by the `while x + 64 <= width`
  // loop condition and the caller‑promised slice lengths.
  unsafe {
    let rnd_v = _mm512_set1_epi32(RND);
    let y_off_v = _mm512_set1_epi16(y_off as i16);
    let y_scale_v = _mm512_set1_epi32(y_scale);
    let c_scale_v = _mm512_set1_epi32(c_scale);
    let mid128 = _mm512_set1_epi16(128);
    let cru = _mm512_set1_epi32(coeffs.r_u());
    let crv = _mm512_set1_epi32(coeffs.r_v());
    let cgu = _mm512_set1_epi32(coeffs.g_u());
    let cgv = _mm512_set1_epi32(coeffs.g_v());
    let cbu = _mm512_set1_epi32(coeffs.b_u());
    let cbv = _mm512_set1_epi32(coeffs.b_v());

    // Lane‑fixup permute indices, computed once per call.
    let pack_fixup = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
    let dup_lo_idx = _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11);
    let dup_hi_idx = _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15);

    let mut x = 0usize;
    while x + 64 <= width {
      let y_vec = _mm512_loadu_si512(y.as_ptr().add(x).cast());
      let u_vec_256 = _mm256_loadu_si256(u_half.as_ptr().add(x / 2).cast());
      let v_vec_256 = _mm256_loadu_si256(v_half.as_ptr().add(x / 2).cast());

      // Widen U/V to i16x32 and subtract 128.
      let u_i16 = _mm512_sub_epi16(_mm512_cvtepu8_epi16(u_vec_256), mid128);
      let v_i16 = _mm512_sub_epi16(_mm512_cvtepu8_epi16(v_vec_256), mid128);

      // Split each i16x32 into two i32x16 halves for the Q15 multiplies.
      let u_lo_i32 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_i16));
      let u_hi_i32 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(u_i16));
      let v_lo_i32 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_i16));
      let v_hi_i32 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(v_i16));

      // u_d, v_d = (u * c_scale + RND) >> 15 — bit‑exact to scalar.
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

      // Per‑channel chroma → i16x32 (natural order after pack fixup).
      let r_chroma = chroma_i16x32(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v, pack_fixup);
      let g_chroma = chroma_i16x32(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v, pack_fixup);
      let b_chroma = chroma_i16x32(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v, pack_fixup);

      // Nearest‑neighbor upsample: pair‑duplicate each chroma lane into
      // two i16x32 vectors covering 64 Y lanes.
      let (r_dup_lo, r_dup_hi) = chroma_dup(r_chroma, dup_lo_idx, dup_hi_idx);
      let (g_dup_lo, g_dup_hi) = chroma_dup(g_chroma, dup_lo_idx, dup_hi_idx);
      let (b_dup_lo, b_dup_hi) = chroma_dup(b_chroma, dup_lo_idx, dup_hi_idx);

      // Y path: widen 64 Y to two i16x32, scale.
      let y_low_i16 = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(y_vec));
      let y_high_i16 = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64::<1>(y_vec));
      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v, pack_fixup);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v, pack_fixup);

      // Saturating i16 add Y + chroma per channel.
      let b_lo = _mm512_adds_epi16(y_scaled_lo, b_dup_lo);
      let b_hi = _mm512_adds_epi16(y_scaled_hi, b_dup_hi);
      let g_lo = _mm512_adds_epi16(y_scaled_lo, g_dup_lo);
      let g_hi = _mm512_adds_epi16(y_scaled_hi, g_dup_hi);
      let r_lo = _mm512_adds_epi16(y_scaled_lo, r_dup_lo);
      let r_hi = _mm512_adds_epi16(y_scaled_hi, r_dup_hi);

      // Saturate‑narrow to u8x64 per channel with the same pack fixup.
      let b_u8 = narrow_u8x64(b_lo, b_hi, pack_fixup);
      let g_u8 = narrow_u8x64(g_lo, g_hi, pack_fixup);
      let r_u8 = narrow_u8x64(r_lo, r_hi, pack_fixup);

      // 3‑way interleave → packed RGB (192 bytes = 4 × 48).
      write_rgb_64(r_u8, g_u8, b_u8, rgb_out.as_mut_ptr().add(x * 3));

      x += 64;
    }

    // Scalar tail for the 0..62 leftover pixels (always even; 4:2:0
    // requires even width so x/2 and width/2 are well‑defined).
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

/// AVX‑512 YUV 4:2:0 10‑bit → packed **8‑bit** RGB.
///
/// Block size 64 Y pixels / 32 chroma pairs per iteration (matching
/// the 8‑bit AVX‑512 kernel). Structural differences:
/// - Two `_mm512_loadu_si512` loads for Y (each 32 `u16` = 64 bytes);
///   one `_mm512_loadu_si512` each for U / V (32 `u16`).
/// - No u8→i16 widening — 10‑bit samples already occupy 16‑bit lanes.
/// - Chroma bias is 512 (10‑bit center).
/// - `range_params_n::<10, 8>` calibrates scales for 10→8 in one shift.
///
/// Reuses [`chroma_i16x32`], [`chroma_dup`], [`scale_y`],
/// [`narrow_u8x64`], and [`write_rgb_64`] along with the pack / dup
/// lane‑fixup indices from the 8‑bit path — post‑chroma math is
/// identical across bit depths.
///
/// # Numerical contract
///
/// Byte‑identical to [`scalar::yuv_420p_n_to_rgb_row::<10>`].
///
/// # Safety
///
/// 1. **AVX‑512F + AVX‑512BW must be available on the current CPU.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `u_half.len() >= width / 2`,
///    `v_half.len() >= width / 2`, `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn yuv420p10_to_rgb_row(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0);
  debug_assert!(y.len() >= width);
  debug_assert!(u_half.len() >= width / 2);
  debug_assert!(v_half.len() >= width / 2);
  debug_assert!(rgb_out.len() >= width * 3);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<10, 8>(full_range);
  let bias = scalar::chroma_bias::<10>();
  const RND: i32 = 1 << 14;

  // SAFETY: AVX‑512BW availability is the caller's obligation.
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
    let dup_lo_idx = _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11);
    let dup_hi_idx = _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15);

    let mut x = 0usize;
    while x + 64 <= width {
      let y_low_i16 = _mm512_loadu_si512(y.as_ptr().add(x).cast());
      let y_high_i16 = _mm512_loadu_si512(y.as_ptr().add(x + 32).cast());
      let u_vec = _mm512_loadu_si512(u_half.as_ptr().add(x / 2).cast());
      let v_vec = _mm512_loadu_si512(v_half.as_ptr().add(x / 2).cast());

      let u_i16 = _mm512_sub_epi16(u_vec, bias_v);
      let v_i16 = _mm512_sub_epi16(v_vec, bias_v);

      let u_lo_i32 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_i16));
      let u_hi_i32 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(u_i16));
      let v_lo_i32 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_i16));
      let v_hi_i32 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(v_i16));

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

      let r_chroma = chroma_i16x32(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v, pack_fixup);
      let g_chroma = chroma_i16x32(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v, pack_fixup);
      let b_chroma = chroma_i16x32(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v, pack_fixup);

      let (r_dup_lo, r_dup_hi) = chroma_dup(r_chroma, dup_lo_idx, dup_hi_idx);
      let (g_dup_lo, g_dup_hi) = chroma_dup(g_chroma, dup_lo_idx, dup_hi_idx);
      let (b_dup_lo, b_dup_hi) = chroma_dup(b_chroma, dup_lo_idx, dup_hi_idx);

      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v, pack_fixup);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v, pack_fixup);

      let b_lo = _mm512_adds_epi16(y_scaled_lo, b_dup_lo);
      let b_hi = _mm512_adds_epi16(y_scaled_hi, b_dup_hi);
      let g_lo = _mm512_adds_epi16(y_scaled_lo, g_dup_lo);
      let g_hi = _mm512_adds_epi16(y_scaled_hi, g_dup_hi);
      let r_lo = _mm512_adds_epi16(y_scaled_lo, r_dup_lo);
      let r_hi = _mm512_adds_epi16(y_scaled_hi, r_dup_hi);

      let b_u8 = narrow_u8x64(b_lo, b_hi, pack_fixup);
      let g_u8 = narrow_u8x64(g_lo, g_hi, pack_fixup);
      let r_u8 = narrow_u8x64(r_lo, r_hi, pack_fixup);

      write_rgb_64(r_u8, g_u8, b_u8, rgb_out.as_mut_ptr().add(x * 3));

      x += 64;
    }

    if x < width {
      scalar::yuv_420p_n_to_rgb_row::<10>(
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

/// AVX‑512 YUV 4:2:0 10‑bit → packed **10‑bit `u16`** RGB.
///
/// Block size 64 Y pixels per iteration. Mirrors
/// [`yuv420p10_to_rgb_row`]'s pre‑write math; output uses explicit
/// min/max clamp to `[0, 1023]` and 8 calls to [`write_rgb_u16_8`]
/// per block (each handles 8 pixels). A true AVX‑512 u16 interleave
/// would cut store count ~8×; left as a follow‑up optimization.
///
/// # Numerical contract
///
/// Identical to [`scalar::yuv_420p_n_to_rgb_u16_row::<10>`].
///
/// # Safety
///
/// 1. **AVX‑512F + AVX‑512BW must be available on the current CPU.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `u_half.len() >= width / 2`,
///    `v_half.len() >= width / 2`, `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn yuv420p10_to_rgb_u16_row(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0);
  debug_assert!(y.len() >= width);
  debug_assert!(u_half.len() >= width / 2);
  debug_assert!(v_half.len() >= width / 2);
  debug_assert!(rgb_out.len() >= width * 3);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<10, 10>(full_range);
  let bias = scalar::chroma_bias::<10>();
  const RND: i32 = 1 << 14;
  const OUT_MAX_10: i16 = 1023;

  // SAFETY: AVX‑512BW availability is the caller's obligation.
  unsafe {
    let rnd_v = _mm512_set1_epi32(RND);
    let y_off_v = _mm512_set1_epi16(y_off as i16);
    let y_scale_v = _mm512_set1_epi32(y_scale);
    let c_scale_v = _mm512_set1_epi32(c_scale);
    let bias_v = _mm512_set1_epi16(bias as i16);
    let max_v = _mm512_set1_epi16(OUT_MAX_10);
    let zero_v = _mm512_set1_epi16(0);
    let cru = _mm512_set1_epi32(coeffs.r_u());
    let crv = _mm512_set1_epi32(coeffs.r_v());
    let cgu = _mm512_set1_epi32(coeffs.g_u());
    let cgv = _mm512_set1_epi32(coeffs.g_v());
    let cbu = _mm512_set1_epi32(coeffs.b_u());
    let cbv = _mm512_set1_epi32(coeffs.b_v());

    let pack_fixup = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
    let dup_lo_idx = _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11);
    let dup_hi_idx = _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15);

    let mut x = 0usize;
    while x + 64 <= width {
      let y_low_i16 = _mm512_loadu_si512(y.as_ptr().add(x).cast());
      let y_high_i16 = _mm512_loadu_si512(y.as_ptr().add(x + 32).cast());
      let u_vec = _mm512_loadu_si512(u_half.as_ptr().add(x / 2).cast());
      let v_vec = _mm512_loadu_si512(v_half.as_ptr().add(x / 2).cast());

      let u_i16 = _mm512_sub_epi16(u_vec, bias_v);
      let v_i16 = _mm512_sub_epi16(v_vec, bias_v);

      let u_lo_i32 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_i16));
      let u_hi_i32 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(u_i16));
      let v_lo_i32 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_i16));
      let v_hi_i32 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(v_i16));

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

      let r_chroma = chroma_i16x32(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v, pack_fixup);
      let g_chroma = chroma_i16x32(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v, pack_fixup);
      let b_chroma = chroma_i16x32(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v, pack_fixup);

      let (r_dup_lo, r_dup_hi) = chroma_dup(r_chroma, dup_lo_idx, dup_hi_idx);
      let (g_dup_lo, g_dup_hi) = chroma_dup(g_chroma, dup_lo_idx, dup_hi_idx);
      let (b_dup_lo, b_dup_hi) = chroma_dup(b_chroma, dup_lo_idx, dup_hi_idx);

      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v, pack_fixup);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v, pack_fixup);

      let r_lo = clamp_u10_x32(_mm512_adds_epi16(y_scaled_lo, r_dup_lo), zero_v, max_v);
      let r_hi = clamp_u10_x32(_mm512_adds_epi16(y_scaled_hi, r_dup_hi), zero_v, max_v);
      let g_lo = clamp_u10_x32(_mm512_adds_epi16(y_scaled_lo, g_dup_lo), zero_v, max_v);
      let g_hi = clamp_u10_x32(_mm512_adds_epi16(y_scaled_hi, g_dup_hi), zero_v, max_v);
      let b_lo = clamp_u10_x32(_mm512_adds_epi16(y_scaled_lo, b_dup_lo), zero_v, max_v);
      let b_hi = clamp_u10_x32(_mm512_adds_epi16(y_scaled_hi, b_dup_hi), zero_v, max_v);

      // Eight 8‑pixel u16 writes per 64‑pixel block. For each i16x32
      // channel vector we extract four 128‑bit quarters and hand each
      // to the shared SSE4.1 u16 interleave helper.
      let dst = rgb_out.as_mut_ptr().add(x * 3);
      write_quarter(r_lo, g_lo, b_lo, 0, dst);
      write_quarter(r_lo, g_lo, b_lo, 1, dst.add(24));
      write_quarter(r_lo, g_lo, b_lo, 2, dst.add(48));
      write_quarter(r_lo, g_lo, b_lo, 3, dst.add(72));
      write_quarter(r_hi, g_hi, b_hi, 0, dst.add(96));
      write_quarter(r_hi, g_hi, b_hi, 1, dst.add(120));
      write_quarter(r_hi, g_hi, b_hi, 2, dst.add(144));
      write_quarter(r_hi, g_hi, b_hi, 3, dst.add(168));

      x += 64;
    }

    if x < width {
      scalar::yuv_420p_n_to_rgb_u16_row::<10>(
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

/// Clamps an `i16x32` vector to `[0, max]` via AVX‑512
/// `_mm512_min_epi16` / `_mm512_max_epi16`. Used by the 10‑bit u16
/// output path.
#[inline(always)]
fn clamp_u10_x32(v: __m512i, zero_v: __m512i, max_v: __m512i) -> __m512i {
  unsafe { _mm512_min_epi16(_mm512_max_epi16(v, zero_v), max_v) }
}

/// Writes one 8‑pixel u16 RGB chunk using a 128‑bit quarter of each
/// `i16x32` channel vector. `idx` ∈ `{0,1,2,3}` selects which of the
/// four 128‑bit lanes to extract via `_mm512_extracti32x4_epi32`.
///
/// # Safety
///
/// Same as [`write_rgb_u16_8`] — `ptr` must point to at least 48
/// writable bytes (24 `u16`). Caller's `target_feature` must include
/// AVX‑512F + AVX‑512BW (so `_mm512_extracti32x4_epi32` is available)
/// and SSSE3 (for the underlying `_mm_shuffle_epi8` inside
/// `write_rgb_u16_8`).
#[inline(always)]
unsafe fn write_quarter(r: __m512i, g: __m512i, b: __m512i, idx: u8, ptr: *mut u16) {
  // SAFETY: caller holds the AVX‑512F + SSSE3 target‑feature context.
  // Constant generic arg `IDX` picks one of four 128‑bit lanes; `idx`
  // is bounded to 0..=3 by call sites.
  unsafe {
    let (rq, gq, bq) = match idx {
      0 => (
        _mm512_extracti32x4_epi32::<0>(r),
        _mm512_extracti32x4_epi32::<0>(g),
        _mm512_extracti32x4_epi32::<0>(b),
      ),
      1 => (
        _mm512_extracti32x4_epi32::<1>(r),
        _mm512_extracti32x4_epi32::<1>(g),
        _mm512_extracti32x4_epi32::<1>(b),
      ),
      2 => (
        _mm512_extracti32x4_epi32::<2>(r),
        _mm512_extracti32x4_epi32::<2>(g),
        _mm512_extracti32x4_epi32::<2>(b),
      ),
      _ => (
        _mm512_extracti32x4_epi32::<3>(r),
        _mm512_extracti32x4_epi32::<3>(g),
        _mm512_extracti32x4_epi32::<3>(b),
      ),
    };
    write_rgb_u16_8(rq, gq, bq, ptr);
  }
}

/// AVX‑512 NV12 → packed RGB (UV-ordered chroma). Thin wrapper over
/// [`nv12_or_nv21_to_rgb_row_impl`] with `SWAP_UV = false`.
///
/// # Safety
///
/// Same as [`nv12_or_nv21_to_rgb_row_impl`].
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn nv12_to_rgb_row(
  y: &[u8],
  uv_half: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    nv12_or_nv21_to_rgb_row_impl::<false>(y, uv_half, rgb_out, width, matrix, full_range);
  }
}

/// AVX‑512 NV21 → packed RGB (VU-ordered chroma). Thin wrapper over
/// [`nv12_or_nv21_to_rgb_row_impl`] with `SWAP_UV = true`.
///
/// # Safety
///
/// Same as [`nv12_or_nv21_to_rgb_row_impl`].
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn nv21_to_rgb_row(
  y: &[u8],
  vu_half: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    nv12_or_nv21_to_rgb_row_impl::<true>(y, vu_half, rgb_out, width, matrix, full_range);
  }
}

/// Shared AVX‑512 NV12/NV21 kernel. `SWAP_UV` selects chroma byte
/// order at compile time.
///
/// # Safety
///
/// 1. **AVX‑512F + AVX‑512BW must be available on the current CPU**
///    (same obligation as [`yuv_420_to_rgb_row`]).
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`.
/// 4. `uv_or_vu_half.len() >= width` (64 interleaved bytes per 64 Y pixels).
/// 5. `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn nv12_or_nv21_to_rgb_row_impl<const SWAP_UV: bool>(
  y: &[u8],
  uv_or_vu_half: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0, "NV12/NV21 require even width");
  debug_assert!(y.len() >= width);
  debug_assert!(uv_or_vu_half.len() >= width);
  debug_assert!(rgb_out.len() >= width * 3);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params(full_range);
  const RND: i32 = 1 << 14;

  // SAFETY: AVX‑512BW availability is the caller's obligation; all
  // pointer adds below are bounded by the `while x + 64 <= width`
  // condition and the caller‑promised slice lengths.
  unsafe {
    let rnd_v = _mm512_set1_epi32(RND);
    let y_off_v = _mm512_set1_epi16(y_off as i16);
    let y_scale_v = _mm512_set1_epi32(y_scale);
    let c_scale_v = _mm512_set1_epi32(c_scale);
    let mid128 = _mm512_set1_epi16(128);
    let cru = _mm512_set1_epi32(coeffs.r_u());
    let crv = _mm512_set1_epi32(coeffs.r_v());
    let cgu = _mm512_set1_epi32(coeffs.g_u());
    let cgv = _mm512_set1_epi32(coeffs.g_v());
    let cbu = _mm512_set1_epi32(coeffs.b_u());
    let cbv = _mm512_set1_epi32(coeffs.b_v());

    let pack_fixup = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
    let dup_lo_idx = _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11);
    let dup_hi_idx = _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15);

    // Per‑128‑bit‑lane UV deinterleave mask. Broadcast to all 4 lanes.
    // Within each 16‑byte chunk, pack even‑offset (U) bytes into the
    // low 8 lanes and odd‑offset (V) bytes into the high 8 lanes.
    let uv_lane_mask = _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);
    let uv_deint_mask = _mm512_broadcast_i32x4(uv_lane_mask);
    // After per‑lane shuffle the 64‑bit lane layout is
    // `[U0, V0, U1, V1, U2, V2, U3, V3]`; permuting with
    // `[0, 2, 4, 6, 1, 3, 5, 7]` compacts to
    // `[U0, U1, U2, U3 | V0, V1, V2, V3]` — low 256 = U, high 256 = V.
    let uv_collect = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);

    let mut x = 0usize;
    while x + 64 <= width {
      let y_vec = _mm512_loadu_si512(y.as_ptr().add(x).cast());
      // 64 Y pixels → 32 chroma pairs = 64 interleaved bytes at
      // offset `x` in the chroma row.
      let uv_vec = _mm512_loadu_si512(uv_or_vu_half.as_ptr().add(x).cast());

      // Per-lane shuffle + permute packs even-offset bytes into low
      // 256 and odd-offset bytes into high 256. For NV12 that's
      // (U, V); for NV21 the roles swap.
      let deint = _mm512_shuffle_epi8(uv_vec, uv_deint_mask);
      let uv_compact = _mm512_permutexvar_epi64(uv_collect, deint);
      let (u_vec_256, v_vec_256) = if SWAP_UV {
        (
          _mm512_extracti64x4_epi64::<1>(uv_compact),
          _mm512_castsi512_si256(uv_compact),
        )
      } else {
        (
          _mm512_castsi512_si256(uv_compact),
          _mm512_extracti64x4_epi64::<1>(uv_compact),
        )
      };

      let u_i16 = _mm512_sub_epi16(_mm512_cvtepu8_epi16(u_vec_256), mid128);
      let v_i16 = _mm512_sub_epi16(_mm512_cvtepu8_epi16(v_vec_256), mid128);

      let u_lo_i32 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_i16));
      let u_hi_i32 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(u_i16));
      let v_lo_i32 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_i16));
      let v_hi_i32 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(v_i16));

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

      let r_chroma = chroma_i16x32(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v, pack_fixup);
      let g_chroma = chroma_i16x32(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v, pack_fixup);
      let b_chroma = chroma_i16x32(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v, pack_fixup);

      let (r_dup_lo, r_dup_hi) = chroma_dup(r_chroma, dup_lo_idx, dup_hi_idx);
      let (g_dup_lo, g_dup_hi) = chroma_dup(g_chroma, dup_lo_idx, dup_hi_idx);
      let (b_dup_lo, b_dup_hi) = chroma_dup(b_chroma, dup_lo_idx, dup_hi_idx);

      let y_low_i16 = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(y_vec));
      let y_high_i16 = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64::<1>(y_vec));
      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v, pack_fixup);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v, pack_fixup);

      let b_lo = _mm512_adds_epi16(y_scaled_lo, b_dup_lo);
      let b_hi = _mm512_adds_epi16(y_scaled_hi, b_dup_hi);
      let g_lo = _mm512_adds_epi16(y_scaled_lo, g_dup_lo);
      let g_hi = _mm512_adds_epi16(y_scaled_hi, g_dup_hi);
      let r_lo = _mm512_adds_epi16(y_scaled_lo, r_dup_lo);
      let r_hi = _mm512_adds_epi16(y_scaled_hi, r_dup_hi);

      let b_u8 = narrow_u8x64(b_lo, b_hi, pack_fixup);
      let g_u8 = narrow_u8x64(g_lo, g_hi, pack_fixup);
      let r_u8 = narrow_u8x64(r_lo, r_hi, pack_fixup);

      write_rgb_64(r_u8, g_u8, b_u8, rgb_out.as_mut_ptr().add(x * 3));

      x += 64;
    }

    if x < width {
      if SWAP_UV {
        scalar::nv21_to_rgb_row(
          &y[x..width],
          &uv_or_vu_half[x..width],
          &mut rgb_out[x * 3..width * 3],
          width - x,
          matrix,
          full_range,
        );
      } else {
        scalar::nv12_to_rgb_row(
          &y[x..width],
          &uv_or_vu_half[x..width],
          &mut rgb_out[x * 3..width * 3],
          width - x,
          matrix,
          full_range,
        );
      }
    }
  }
}

// ---- helpers (inlined into the target_feature‑enabled caller) ----------

/// `>>_a 15` shift (arithmetic, sign‑extending).
#[inline(always)]
fn q15_shift(v: __m512i) -> __m512i {
  unsafe { _mm512_srai_epi32::<15>(v) }
}

/// Computes one i16x32 chroma channel vector from the four i32x16
/// chroma inputs (lo/hi halves of `u_d` and `v_d`). Mirrors the scalar
/// `(coeff_u * u_d + coeff_v * v_d + RND) >> 15`, saturating‑packs to
/// i16x32, then applies `pack_fixup` to restore natural element order.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn chroma_i16x32(
  cu: __m512i,
  cv: __m512i,
  u_d_lo: __m512i,
  v_d_lo: __m512i,
  u_d_hi: __m512i,
  v_d_hi: __m512i,
  rnd: __m512i,
  pack_fixup: __m512i,
) -> __m512i {
  unsafe {
    let lo = _mm512_srai_epi32::<15>(_mm512_add_epi32(
      _mm512_add_epi32(
        _mm512_mullo_epi32(cu, u_d_lo),
        _mm512_mullo_epi32(cv, v_d_lo),
      ),
      rnd,
    ));
    let hi = _mm512_srai_epi32::<15>(_mm512_add_epi32(
      _mm512_add_epi32(
        _mm512_mullo_epi32(cu, u_d_hi),
        _mm512_mullo_epi32(cv, v_d_hi),
      ),
      rnd,
    ));
    _mm512_permutexvar_epi64(pack_fixup, _mm512_packs_epi32(lo, hi))
  }
}

/// `(Y - y_off) * y_scale + RND >> 15` applied to an i16x32 vector,
/// returned as i16x32 (with pack fixup applied).
#[inline(always)]
fn scale_y(
  y_i16: __m512i,
  y_off_v: __m512i,
  y_scale_v: __m512i,
  rnd: __m512i,
  pack_fixup: __m512i,
) -> __m512i {
  unsafe {
    let shifted = _mm512_sub_epi16(y_i16, y_off_v);
    let lo_i32 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(shifted));
    let hi_i32 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(shifted));
    let lo_scaled =
      _mm512_srai_epi32::<15>(_mm512_add_epi32(_mm512_mullo_epi32(lo_i32, y_scale_v), rnd));
    let hi_scaled =
      _mm512_srai_epi32::<15>(_mm512_add_epi32(_mm512_mullo_epi32(hi_i32, y_scale_v), rnd));
    _mm512_permutexvar_epi64(pack_fixup, _mm512_packs_epi32(lo_scaled, hi_scaled))
  }
}

/// Duplicates each of 32 chroma lanes into its adjacent pair slot,
/// splitting across two i16x32 vectors covering 64 Y lanes.
#[inline(always)]
fn chroma_dup(chroma: __m512i, dup_lo_idx: __m512i, dup_hi_idx: __m512i) -> (__m512i, __m512i) {
  unsafe {
    let a = _mm512_unpacklo_epi16(chroma, chroma);
    let b = _mm512_unpackhi_epi16(chroma, chroma);
    let lo32 = _mm512_permutex2var_epi64(a, dup_lo_idx, b);
    let hi32 = _mm512_permutex2var_epi64(a, dup_hi_idx, b);
    (lo32, hi32)
  }
}

/// Saturating‑narrows two i16x32 vectors into one u8x64 with natural
/// element order.
#[inline(always)]
fn narrow_u8x64(lo: __m512i, hi: __m512i, pack_fixup: __m512i) -> __m512i {
  unsafe { _mm512_permutexvar_epi64(pack_fixup, _mm512_packus_epi16(lo, hi)) }
}

/// Writes 64 pixels of packed RGB (192 bytes) by splitting the u8x64
/// channel vectors into four 128‑bit halves and calling the shared
/// [`write_rgb_16`] helper four times.
///
/// # Safety
///
/// `ptr` must point to at least 192 writable bytes.
#[inline(always)]
unsafe fn write_rgb_64(r: __m512i, g: __m512i, b: __m512i, ptr: *mut u8) {
  unsafe {
    let r0: __m128i = _mm512_castsi512_si128(r);
    let r1: __m128i = _mm512_extracti32x4_epi32::<1>(r);
    let r2: __m128i = _mm512_extracti32x4_epi32::<2>(r);
    let r3: __m128i = _mm512_extracti32x4_epi32::<3>(r);
    let g0: __m128i = _mm512_castsi512_si128(g);
    let g1: __m128i = _mm512_extracti32x4_epi32::<1>(g);
    let g2: __m128i = _mm512_extracti32x4_epi32::<2>(g);
    let g3: __m128i = _mm512_extracti32x4_epi32::<3>(g);
    let b0: __m128i = _mm512_castsi512_si128(b);
    let b1: __m128i = _mm512_extracti32x4_epi32::<1>(b);
    let b2: __m128i = _mm512_extracti32x4_epi32::<2>(b);
    let b3: __m128i = _mm512_extracti32x4_epi32::<3>(b);

    write_rgb_16(r0, g0, b0, ptr);
    write_rgb_16(r1, g1, b1, ptr.add(48));
    write_rgb_16(r2, g2, b2, ptr.add(96));
    write_rgb_16(r3, g3, b3, ptr.add(144));
  }
}

// ===== BGR ↔ RGB byte swap ==============================================

/// AVX‑512 BGR ↔ RGB byte swap. 64 pixels per iteration via four calls
/// to [`super::x86_common::swap_rb_16_pixels`]. The helper uses SSSE3
/// `_mm_shuffle_epi8`, which AVX‑512BW (a superset) allows.
///
/// # Safety
///
/// 1. AVX‑512BW must be available (dispatcher obligation).
/// 2. `input.len() >= 3 * width`.
/// 3. `output.len() >= 3 * width`.
/// 4. `input` / `output` must not alias.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn bgr_rgb_swap_row(input: &[u8], output: &mut [u8], width: usize) {
  debug_assert!(input.len() >= width * 3, "input row too short");
  debug_assert!(output.len() >= width * 3, "output row too short");

  unsafe {
    let mut x = 0usize;
    while x + 64 <= width {
      let base_in = input.as_ptr().add(x * 3);
      let base_out = output.as_mut_ptr().add(x * 3);
      swap_rb_16_pixels(base_in, base_out);
      swap_rb_16_pixels(base_in.add(48), base_out.add(48));
      swap_rb_16_pixels(base_in.add(96), base_out.add(96));
      swap_rb_16_pixels(base_in.add(144), base_out.add(144));
      x += 64;
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

// ===== RGB → HSV =========================================================

/// AVX‑512 RGB → planar HSV. 64 pixels per iteration via four calls to
/// the shared [`super::x86_common::rgb_to_hsv_16_pixels`] helper
/// (SSE4.1‑level compute under AVX‑512 target_feature). Matches the
/// scalar reference within ±1 LSB — the shared helper uses `_mm_rcp_ps`
/// + one Newton‑Raphson step instead of true division (see `x86_common.rs`).
///
/// # Safety
///
/// 1. AVX‑512BW must be available (dispatcher obligation).
/// 2. `rgb.len() >= 3 * width`; each output plane `>= width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn rgb_to_hsv_row(
  rgb: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
) {
  debug_assert!(rgb.len() >= width * 3);
  debug_assert!(h_out.len() >= width);
  debug_assert!(s_out.len() >= width);
  debug_assert!(v_out.len() >= width);

  unsafe {
    let mut x = 0usize;
    while x + 64 <= width {
      let base_in = rgb.as_ptr().add(x * 3);
      let base_h = h_out.as_mut_ptr().add(x);
      let base_s = s_out.as_mut_ptr().add(x);
      let base_v = v_out.as_mut_ptr().add(x);
      rgb_to_hsv_16_pixels(base_in, base_h, base_s, base_v);
      rgb_to_hsv_16_pixels(
        base_in.add(48),
        base_h.add(16),
        base_s.add(16),
        base_v.add(16),
      );
      rgb_to_hsv_16_pixels(
        base_in.add(96),
        base_h.add(32),
        base_s.add(32),
        base_v.add(32),
      );
      rgb_to_hsv_16_pixels(
        base_in.add(144),
        base_h.add(48),
        base_s.add(48),
        base_v.add(48),
      );
      x += 64;
    }
    if x < width {
      scalar::rgb_to_hsv_row(
        &rgb[x * 3..width * 3],
        &mut h_out[x..width],
        &mut s_out[x..width],
        &mut v_out[x..width],
        width - x,
      );
    }
  }
}

#[cfg(all(test, feature = "std"))]
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
    let mut rgb_scalar = std::vec![0u8; width * 3];
    let mut rgb_avx512 = std::vec![0u8; width * 3];

    scalar::yuv_420_to_rgb_row(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
    unsafe {
      yuv_420_to_rgb_row(&y, &u, &v, &mut rgb_avx512, width, matrix, full_range);
    }

    if rgb_scalar != rgb_avx512 {
      let first_diff = rgb_scalar
        .iter()
        .zip(rgb_avx512.iter())
        .position(|(a, b)| a != b)
        .unwrap();
      panic!(
        "AVX‑512 diverges from scalar at byte {first_diff} (width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} avx512={}",
        rgb_scalar[first_diff], rgb_avx512[first_diff]
      );
    }
  }

  #[test]
  fn avx512_matches_scalar_all_matrices_64() {
    if !std::arch::is_x86_feature_detected!("avx512bw") {
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
        check_equivalence(64, m, full);
      }
    }
  }

  #[test]
  fn avx512_matches_scalar_width_128() {
    if !std::arch::is_x86_feature_detected!("avx512bw") {
      return;
    }
    check_equivalence(128, ColorMatrix::Bt601, true);
    check_equivalence(128, ColorMatrix::Bt709, false);
    check_equivalence(128, ColorMatrix::YCgCo, true);
  }

  #[test]
  fn avx512_matches_scalar_width_1920() {
    if !std::arch::is_x86_feature_detected!("avx512bw") {
      return;
    }
    check_equivalence(1920, ColorMatrix::Bt709, false);
  }

  #[test]
  fn avx512_matches_scalar_odd_tail_widths() {
    if !std::arch::is_x86_feature_detected!("avx512bw") {
      return;
    }
    // Widths that leave a non‑trivial scalar tail (non‑multiple of 64).
    for w in [66usize, 94, 126, 1922] {
      check_equivalence(w, ColorMatrix::Bt601, false);
    }
  }

  // ---- nv12_to_rgb_row equivalence ------------------------------------

  fn check_nv12_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
    let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
    let uv: std::vec::Vec<u8> = (0..width / 2)
      .flat_map(|i| [((i * 53 + 23) & 0xFF) as u8, ((i * 71 + 91) & 0xFF) as u8])
      .collect();
    let mut rgb_scalar = std::vec![0u8; width * 3];
    let mut rgb_avx512 = std::vec![0u8; width * 3];

    scalar::nv12_to_rgb_row(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
    unsafe {
      nv12_to_rgb_row(&y, &uv, &mut rgb_avx512, width, matrix, full_range);
    }
    assert_eq!(
      rgb_scalar, rgb_avx512,
      "AVX‑512 NV12 ≠ scalar (width={width}, matrix={matrix:?})"
    );
  }

  fn check_nv12_matches_yuv420p(width: usize, matrix: ColorMatrix, full_range: bool) {
    let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
    let u: std::vec::Vec<u8> = (0..width / 2)
      .map(|i| ((i * 53 + 23) & 0xFF) as u8)
      .collect();
    let v: std::vec::Vec<u8> = (0..width / 2)
      .map(|i| ((i * 71 + 91) & 0xFF) as u8)
      .collect();
    let uv: std::vec::Vec<u8> = u.iter().zip(v.iter()).flat_map(|(a, b)| [*a, *b]).collect();

    let mut rgb_yuv420p = std::vec![0u8; width * 3];
    let mut rgb_nv12 = std::vec![0u8; width * 3];
    unsafe {
      yuv_420_to_rgb_row(&y, &u, &v, &mut rgb_yuv420p, width, matrix, full_range);
      nv12_to_rgb_row(&y, &uv, &mut rgb_nv12, width, matrix, full_range);
    }
    assert_eq!(
      rgb_yuv420p, rgb_nv12,
      "AVX‑512 NV12 ≠ YUV420P for equivalent UV"
    );
  }

  #[test]
  fn avx512_nv12_matches_scalar_all_matrices_64() {
    if !std::arch::is_x86_feature_detected!("avx512bw") {
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
        check_nv12_equivalence(64, m, full);
      }
    }
  }

  #[test]
  fn avx512_nv12_matches_scalar_widths() {
    if !std::arch::is_x86_feature_detected!("avx512bw") {
      return;
    }
    for w in [128usize, 1920, 66, 94, 126, 1922] {
      check_nv12_equivalence(w, ColorMatrix::Bt709, false);
    }
  }

  #[test]
  fn avx512_nv12_matches_yuv420p() {
    if !std::arch::is_x86_feature_detected!("avx512bw") {
      return;
    }
    for w in [64usize, 126, 256, 1920] {
      check_nv12_matches_yuv420p(w, ColorMatrix::Bt709, false);
      check_nv12_matches_yuv420p(w, ColorMatrix::YCgCo, true);
    }
  }

  // ---- bgr_rgb_swap_row equivalence -----------------------------------

  fn check_swap_equivalence(width: usize) {
    let input: std::vec::Vec<u8> = (0..width * 3)
      .map(|i| ((i * 17 + 41) & 0xFF) as u8)
      .collect();
    let mut out_scalar = std::vec![0u8; width * 3];
    let mut out_avx512 = std::vec![0u8; width * 3];

    scalar::bgr_rgb_swap_row(&input, &mut out_scalar, width);
    unsafe {
      bgr_rgb_swap_row(&input, &mut out_avx512, width);
    }
    assert_eq!(out_scalar, out_avx512, "AVX‑512 swap diverges from scalar");
  }

  #[test]
  fn avx512_swap_matches_scalar() {
    if !std::arch::is_x86_feature_detected!("avx512bw") {
      return;
    }
    for w in [1usize, 31, 63, 64, 65, 95, 127, 128, 1920, 1921] {
      check_swap_equivalence(w);
    }
  }

  // ---- nv21_to_rgb_row equivalence ------------------------------------

  fn check_nv21_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
    let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
    let vu: std::vec::Vec<u8> = (0..width / 2)
      .flat_map(|i| [((i * 53 + 23) & 0xFF) as u8, ((i * 71 + 91) & 0xFF) as u8])
      .collect();
    let mut rgb_scalar = std::vec![0u8; width * 3];
    let mut rgb_avx512 = std::vec![0u8; width * 3];

    scalar::nv21_to_rgb_row(&y, &vu, &mut rgb_scalar, width, matrix, full_range);
    unsafe {
      nv21_to_rgb_row(&y, &vu, &mut rgb_avx512, width, matrix, full_range);
    }
    assert_eq!(
      rgb_scalar, rgb_avx512,
      "AVX-512 NV21 ≠ scalar (width={width}, matrix={matrix:?})"
    );
  }

  fn check_nv21_matches_nv12_swapped(width: usize, matrix: ColorMatrix, full_range: bool) {
    let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
    let uv: std::vec::Vec<u8> = (0..width / 2)
      .flat_map(|i| [((i * 53 + 23) & 0xFF) as u8, ((i * 71 + 91) & 0xFF) as u8])
      .collect();
    let mut vu = std::vec![0u8; width];
    for i in 0..width / 2 {
      vu[2 * i] = uv[2 * i + 1];
      vu[2 * i + 1] = uv[2 * i];
    }

    let mut rgb_nv12 = std::vec![0u8; width * 3];
    let mut rgb_nv21 = std::vec![0u8; width * 3];
    unsafe {
      nv12_to_rgb_row(&y, &uv, &mut rgb_nv12, width, matrix, full_range);
      nv21_to_rgb_row(&y, &vu, &mut rgb_nv21, width, matrix, full_range);
    }
    assert_eq!(
      rgb_nv12, rgb_nv21,
      "AVX-512 NV21 ≠ NV12 with byte-swapped chroma"
    );
  }

  #[test]
  fn nv21_avx512_matches_scalar_all_matrices_16() {
    if !std::arch::is_x86_feature_detected!("avx512bw") {
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
        check_nv21_equivalence(16, m, full);
      }
    }
  }

  #[test]
  fn nv21_avx512_matches_scalar_widths() {
    if !std::arch::is_x86_feature_detected!("avx512bw") {
      return;
    }
    for w in [32usize, 1920, 18, 30, 34, 1922] {
      check_nv21_equivalence(w, ColorMatrix::Bt709, false);
    }
  }

  #[test]
  fn nv21_avx512_matches_nv12_swapped() {
    if !std::arch::is_x86_feature_detected!("avx512bw") {
      return;
    }
    for w in [16usize, 30, 64, 1920] {
      check_nv21_matches_nv12_swapped(w, ColorMatrix::Bt709, false);
      check_nv21_matches_nv12_swapped(w, ColorMatrix::YCgCo, true);
    }
  }
  // ---- rgb_to_hsv_row equivalence --------------------------------------

  fn check_hsv_equivalence(rgb: &[u8], width: usize) {
    let mut h_s = std::vec![0u8; width];
    let mut s_s = std::vec![0u8; width];
    let mut v_s = std::vec![0u8; width];
    let mut h_k = std::vec![0u8; width];
    let mut s_k = std::vec![0u8; width];
    let mut v_k = std::vec![0u8; width];
    scalar::rgb_to_hsv_row(rgb, &mut h_s, &mut s_s, &mut v_s, width);
    unsafe {
      rgb_to_hsv_row(rgb, &mut h_k, &mut s_k, &mut v_k, width);
    }
    for (i, (a, b)) in h_s.iter().zip(h_k.iter()).enumerate() {
      assert!(
        a.abs_diff(*b) <= 1,
        "H divergence at pixel {i}: scalar={a} simd={b}"
      );
    }
    for (i, (a, b)) in s_s.iter().zip(s_k.iter()).enumerate() {
      assert!(
        a.abs_diff(*b) <= 1,
        "S divergence at pixel {i}: scalar={a} simd={b}"
      );
    }
    for (i, (a, b)) in v_s.iter().zip(v_k.iter()).enumerate() {
      assert!(
        a.abs_diff(*b) <= 1,
        "V divergence at pixel {i}: scalar={a} simd={b}"
      );
    }
  }

  #[test]
  fn avx512_hsv_matches_scalar() {
    if !std::arch::is_x86_feature_detected!("avx512bw") {
      return;
    }
    let rgb: std::vec::Vec<u8> = (0..1921 * 3)
      .map(|i| ((i * 37 + 11) & 0xFF) as u8)
      .collect();
    for w in [1usize, 63, 64, 65, 127, 128, 1920, 1921] {
      check_hsv_equivalence(&rgb[..w * 3], w);
    }
  }

  // ---- yuv420p10 AVX-512 scalar-equivalence ---------------------------

  fn p10_plane(n: usize, seed: usize) -> std::vec::Vec<u16> {
    (0..n)
      .map(|i| ((i * seed + seed * 3) & 0x3FF) as u16)
      .collect()
  }

  fn check_p10_u8_avx512_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
    if !std::arch::is_x86_feature_detected!("avx512bw") {
      return;
    }
    let y = p10_plane(width, 37);
    let u = p10_plane(width / 2, 53);
    let v = p10_plane(width / 2, 71);
    let mut rgb_scalar = std::vec![0u8; width * 3];
    let mut rgb_simd = std::vec![0u8; width * 3];

    scalar::yuv_420p_n_to_rgb_row::<10>(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
    unsafe {
      yuv420p10_to_rgb_row(&y, &u, &v, &mut rgb_simd, width, matrix, full_range);
    }

    if rgb_scalar != rgb_simd {
      let first_diff = rgb_scalar
        .iter()
        .zip(rgb_simd.iter())
        .position(|(a, b)| a != b)
        .unwrap();
      panic!(
        "AVX-512 10→u8 diverges at byte {first_diff} (width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} simd={}",
        rgb_scalar[first_diff], rgb_simd[first_diff]
      );
    }
  }

  fn check_p10_u16_avx512_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
    if !std::arch::is_x86_feature_detected!("avx512bw") {
      return;
    }
    let y = p10_plane(width, 37);
    let u = p10_plane(width / 2, 53);
    let v = p10_plane(width / 2, 71);
    let mut rgb_scalar = std::vec![0u16; width * 3];
    let mut rgb_simd = std::vec![0u16; width * 3];

    scalar::yuv_420p_n_to_rgb_u16_row::<10>(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
    unsafe {
      yuv420p10_to_rgb_u16_row(&y, &u, &v, &mut rgb_simd, width, matrix, full_range);
    }

    if rgb_scalar != rgb_simd {
      let first_diff = rgb_scalar
        .iter()
        .zip(rgb_simd.iter())
        .position(|(a, b)| a != b)
        .unwrap();
      panic!(
        "AVX-512 10→u16 diverges at elem {first_diff} (width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} simd={}",
        rgb_scalar[first_diff], rgb_simd[first_diff]
      );
    }
  }

  #[test]
  fn avx512_p10_u8_matches_scalar_all_matrices() {
    for m in [
      ColorMatrix::Bt601,
      ColorMatrix::Bt709,
      ColorMatrix::Bt2020Ncl,
      ColorMatrix::Smpte240m,
      ColorMatrix::Fcc,
      ColorMatrix::YCgCo,
    ] {
      for full in [true, false] {
        check_p10_u8_avx512_equivalence(64, m, full);
      }
    }
  }

  #[test]
  fn avx512_p10_u16_matches_scalar_all_matrices() {
    for m in [
      ColorMatrix::Bt601,
      ColorMatrix::Bt709,
      ColorMatrix::Bt2020Ncl,
      ColorMatrix::Smpte240m,
      ColorMatrix::Fcc,
      ColorMatrix::YCgCo,
    ] {
      for full in [true, false] {
        check_p10_u16_avx512_equivalence(64, m, full);
      }
    }
  }

  #[test]
  fn avx512_p10_matches_scalar_odd_tail_widths() {
    for w in [66usize, 126, 130, 1922] {
      check_p10_u8_avx512_equivalence(w, ColorMatrix::Bt601, false);
      check_p10_u16_avx512_equivalence(w, ColorMatrix::Bt709, true);
    }
  }

  #[test]
  fn avx512_p10_matches_scalar_1920() {
    check_p10_u8_avx512_equivalence(1920, ColorMatrix::Bt709, false);
    check_p10_u16_avx512_equivalence(1920, ColorMatrix::Bt2020Ncl, false);
  }
}
