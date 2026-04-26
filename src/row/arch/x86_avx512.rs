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

use core::arch::x86_64::*;

use crate::{
  ColorMatrix,
  row::{
    arch::x86_common::{
      rgb_to_hsv_16_pixels, swap_rb_16_pixels, write_rgb_16, write_rgb_u16_8, write_rgba_16,
    },
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
  // SAFETY: caller-checked AVX-512BW availability + slice bounds —
  // see [`yuv_420_to_rgb_or_rgba_row`] safety contract.
  unsafe {
    yuv_420_to_rgb_or_rgba_row::<false>(y, u_half, v_half, rgb_out, width, matrix, full_range);
  }
}

/// AVX‑512 YUV 4:2:0 → packed **RGBA** (8-bit). Same contract as
/// [`yuv_420_to_rgb_row`] but writes 4 bytes per pixel (R, G, B,
/// `0xFF`).
///
/// # Safety
///
/// 1. AVX‑512F + AVX‑512BW must be available on the current CPU.
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `u_half.len() >= width / 2`,
///    `v_half.len() >= width / 2`, `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn yuv_420_to_rgba_row(
  y: &[u8],
  u_half: &[u8],
  v_half: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller-checked AVX-512BW availability + slice bounds —
  // see [`yuv_420_to_rgb_or_rgba_row`] safety contract.
  unsafe {
    yuv_420_to_rgb_or_rgba_row::<true>(y, u_half, v_half, rgba_out, width, matrix, full_range);
  }
}

/// Shared AVX‑512 kernel for [`yuv_420_to_rgb_row`] (`ALPHA = false`,
/// [`write_rgb_64`]) and [`yuv_420_to_rgba_row`] (`ALPHA = true`,
/// [`write_rgba_64`] with constant `0xFF` alpha). Math is
/// byte-identical to `scalar::yuv_420_to_rgb_or_rgba_row::<ALPHA>`.
///
/// # Safety
///
/// Same as [`yuv_420_to_rgb_row`] / [`yuv_420_to_rgba_row`]; the
/// `out` slice must be `>= width * (if ALPHA { 4 } else { 3 })`
/// bytes long.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn yuv_420_to_rgb_or_rgba_row<const ALPHA: bool>(
  y: &[u8],
  u_half: &[u8],
  v_half: &[u8],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0, "YUV 4:2:0 requires even width");
  debug_assert!(y.len() >= width);
  debug_assert!(u_half.len() >= width / 2);
  debug_assert!(v_half.len() >= width / 2);
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp);

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
    // Constant opaque-alpha vector for the RGBA path; DCE'd when
    // ALPHA = false.
    let alpha_u8 = _mm512_set1_epi8(-1); // 0xFF as i8

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

      if ALPHA {
        // 4‑way interleave → packed RGBA (256 bytes = 4 × 64).
        write_rgba_64(r_u8, g_u8, b_u8, alpha_u8, out.as_mut_ptr().add(x * 4));
      } else {
        // 3‑way interleave → packed RGB (192 bytes = 4 × 48).
        write_rgb_64(r_u8, g_u8, b_u8, out.as_mut_ptr().add(x * 3));
      }

      x += 64;
    }

    // Scalar tail for the 0..62 leftover pixels (always even; 4:2:0
    // requires even width so x/2 and width/2 are well‑defined).
    if x < width {
      scalar::yuv_420_to_rgb_or_rgba_row::<ALPHA>(
        &y[x..width],
        &u_half[x / 2..width / 2],
        &v_half[x / 2..width / 2],
        &mut out[x * bpp..width * bpp],
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
pub(crate) unsafe fn yuv_420p_n_to_rgb_row<const BITS: u32>(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  const { assert!(BITS == 9 || BITS == 10 || BITS == 12 || BITS == 14) };
  debug_assert_eq!(width & 1, 0);
  debug_assert!(y.len() >= width);
  debug_assert!(u_half.len() >= width / 2);
  debug_assert!(v_half.len() >= width / 2);
  debug_assert!(rgb_out.len() >= width * 3);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<BITS, 8>(full_range);
  let bias = scalar::chroma_bias::<BITS>();
  const RND: i32 = 1 << 14;

  // SAFETY: AVX‑512BW availability is the caller's obligation.
  unsafe {
    let rnd_v = _mm512_set1_epi32(RND);
    let y_off_v = _mm512_set1_epi16(y_off as i16);
    let y_scale_v = _mm512_set1_epi32(y_scale);
    let c_scale_v = _mm512_set1_epi32(c_scale);
    let bias_v = _mm512_set1_epi16(bias as i16);
    let mask_v = _mm512_set1_epi16(scalar::bits_mask::<BITS>() as i16);
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
      // AND‑mask every load to the low 10 bits — see matching
      // comment in [`crate::row::scalar::yuv_420p_n_to_rgb_row`].
      let y_low_i16 = _mm512_and_si512(_mm512_loadu_si512(y.as_ptr().add(x).cast()), mask_v);
      let y_high_i16 = _mm512_and_si512(_mm512_loadu_si512(y.as_ptr().add(x + 32).cast()), mask_v);
      let u_vec = _mm512_and_si512(
        _mm512_loadu_si512(u_half.as_ptr().add(x / 2).cast()),
        mask_v,
      );
      let v_vec = _mm512_and_si512(
        _mm512_loadu_si512(v_half.as_ptr().add(x / 2).cast()),
        mask_v,
      );

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
      scalar::yuv_420p_n_to_rgb_row::<BITS>(
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
pub(crate) unsafe fn yuv_420p_n_to_rgb_u16_row<const BITS: u32>(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  const { assert!(BITS == 9 || BITS == 10 || BITS == 12 || BITS == 14) };
  debug_assert_eq!(width & 1, 0);
  debug_assert!(y.len() >= width);
  debug_assert!(u_half.len() >= width / 2);
  debug_assert!(v_half.len() >= width / 2);
  debug_assert!(rgb_out.len() >= width * 3);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<BITS, BITS>(full_range);
  let bias = scalar::chroma_bias::<BITS>();
  const RND: i32 = 1 << 14;
  let out_max: i16 = ((1i32 << BITS) - 1) as i16;

  // SAFETY: AVX‑512BW availability is the caller's obligation.
  unsafe {
    let rnd_v = _mm512_set1_epi32(RND);
    let y_off_v = _mm512_set1_epi16(y_off as i16);
    let y_scale_v = _mm512_set1_epi32(y_scale);
    let c_scale_v = _mm512_set1_epi32(c_scale);
    let bias_v = _mm512_set1_epi16(bias as i16);
    let mask_v = _mm512_set1_epi16(scalar::bits_mask::<BITS>() as i16);
    let max_v = _mm512_set1_epi16(out_max);
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
      // AND‑mask loads to the low 10 bits so `chroma_i16x32`'s
      // `_mm512_packs_epi32` narrow stays lossless.
      let y_low_i16 = _mm512_and_si512(_mm512_loadu_si512(y.as_ptr().add(x).cast()), mask_v);
      let y_high_i16 = _mm512_and_si512(_mm512_loadu_si512(y.as_ptr().add(x + 32).cast()), mask_v);
      let u_vec = _mm512_and_si512(
        _mm512_loadu_si512(u_half.as_ptr().add(x / 2).cast()),
        mask_v,
      );
      let v_vec = _mm512_and_si512(
        _mm512_loadu_si512(v_half.as_ptr().add(x / 2).cast()),
        mask_v,
      );

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

      let r_lo = clamp_u16_max_x32(_mm512_adds_epi16(y_scaled_lo, r_dup_lo), zero_v, max_v);
      let r_hi = clamp_u16_max_x32(_mm512_adds_epi16(y_scaled_hi, r_dup_hi), zero_v, max_v);
      let g_lo = clamp_u16_max_x32(_mm512_adds_epi16(y_scaled_lo, g_dup_lo), zero_v, max_v);
      let g_hi = clamp_u16_max_x32(_mm512_adds_epi16(y_scaled_hi, g_dup_hi), zero_v, max_v);
      let b_lo = clamp_u16_max_x32(_mm512_adds_epi16(y_scaled_lo, b_dup_lo), zero_v, max_v);
      let b_hi = clamp_u16_max_x32(_mm512_adds_epi16(y_scaled_hi, b_dup_hi), zero_v, max_v);

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
      scalar::yuv_420p_n_to_rgb_u16_row::<BITS>(
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
/// `_mm512_min_epi16` / `_mm512_max_epi16`. Used by native-depth
/// u16 output paths (10/12/14 bit).
#[inline(always)]
fn clamp_u16_max_x32(v: __m512i, zero_v: __m512i, max_v: __m512i) -> __m512i {
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

/// AVX-512 YUV 4:4:4 planar 10/12/14-bit → packed **u8** RGB.
/// Const-generic over `BITS ∈ {10, 12, 14}`. Block size 64 pixels.
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available.**
/// 2. `y.len() >= width`, `u.len() >= width`, `v.len() >= width`,
///    `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn yuv_444p_n_to_rgb_row<const BITS: u32>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  const { assert!(BITS == 9 || BITS == 10 || BITS == 12 || BITS == 14) };
  debug_assert!(y.len() >= width);
  debug_assert!(u.len() >= width);
  debug_assert!(v.len() >= width);
  debug_assert!(rgb_out.len() >= width * 3);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<BITS, 8>(full_range);
  let bias = scalar::chroma_bias::<BITS>();
  const RND: i32 = 1 << 14;

  unsafe {
    let rnd_v = _mm512_set1_epi32(RND);
    let y_off_v = _mm512_set1_epi16(y_off as i16);
    let y_scale_v = _mm512_set1_epi32(y_scale);
    let c_scale_v = _mm512_set1_epi32(c_scale);
    let bias_v = _mm512_set1_epi16(bias as i16);
    let mask_v = _mm512_set1_epi16(scalar::bits_mask::<BITS>() as i16);
    let cru = _mm512_set1_epi32(coeffs.r_u());
    let crv = _mm512_set1_epi32(coeffs.r_v());
    let cgu = _mm512_set1_epi32(coeffs.g_u());
    let cgv = _mm512_set1_epi32(coeffs.g_v());
    let cbu = _mm512_set1_epi32(coeffs.b_u());
    let cbv = _mm512_set1_epi32(coeffs.b_v());

    let pack_fixup = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);

    let mut x = 0usize;
    while x + 64 <= width {
      // 64 Y + 64 U + 64 V per iter. Full-width chroma (two 512-bit
      // loads each) — no horizontal duplication, 4:4:4 is 1:1.
      let y_low_i16 = _mm512_and_si512(_mm512_loadu_si512(y.as_ptr().add(x).cast()), mask_v);
      let y_high_i16 = _mm512_and_si512(_mm512_loadu_si512(y.as_ptr().add(x + 32).cast()), mask_v);
      let u_lo_vec = _mm512_and_si512(_mm512_loadu_si512(u.as_ptr().add(x).cast()), mask_v);
      let u_hi_vec = _mm512_and_si512(_mm512_loadu_si512(u.as_ptr().add(x + 32).cast()), mask_v);
      let v_lo_vec = _mm512_and_si512(_mm512_loadu_si512(v.as_ptr().add(x).cast()), mask_v);
      let v_hi_vec = _mm512_and_si512(_mm512_loadu_si512(v.as_ptr().add(x + 32).cast()), mask_v);

      let u_lo_i16 = _mm512_sub_epi16(u_lo_vec, bias_v);
      let u_hi_i16 = _mm512_sub_epi16(u_hi_vec, bias_v);
      let v_lo_i16 = _mm512_sub_epi16(v_lo_vec, bias_v);
      let v_hi_i16 = _mm512_sub_epi16(v_hi_vec, bias_v);

      let u_lo_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_lo_i16));
      let u_lo_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(u_lo_i16));
      let u_hi_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_hi_i16));
      let u_hi_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(u_hi_i16));
      let v_lo_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_lo_i16));
      let v_lo_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(v_lo_i16));
      let v_hi_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_hi_i16));
      let v_hi_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(v_hi_i16));

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

      // Two chroma_i16x32 per channel → 64 chroma values. No dup.
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

      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v, pack_fixup);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v, pack_fixup);

      let r_lo = _mm512_adds_epi16(y_scaled_lo, r_chroma_lo);
      let r_hi = _mm512_adds_epi16(y_scaled_hi, r_chroma_hi);
      let g_lo = _mm512_adds_epi16(y_scaled_lo, g_chroma_lo);
      let g_hi = _mm512_adds_epi16(y_scaled_hi, g_chroma_hi);
      let b_lo = _mm512_adds_epi16(y_scaled_lo, b_chroma_lo);
      let b_hi = _mm512_adds_epi16(y_scaled_hi, b_chroma_hi);

      let b_u8 = narrow_u8x64(b_lo, b_hi, pack_fixup);
      let g_u8 = narrow_u8x64(g_lo, g_hi, pack_fixup);
      let r_u8 = narrow_u8x64(r_lo, r_hi, pack_fixup);

      write_rgb_64(r_u8, g_u8, b_u8, rgb_out.as_mut_ptr().add(x * 3));

      x += 64;
    }

    if x < width {
      scalar::yuv_444p_n_to_rgb_row::<BITS>(
        &y[x..width],
        &u[x..width],
        &v[x..width],
        &mut rgb_out[x * 3..width * 3],
        width - x,
        matrix,
        full_range,
      );
    }
  }
}

/// AVX-512 YUV 4:4:4 planar 10/12/14-bit → **native-depth u16** RGB.
/// Const-generic over `BITS ∈ {10, 12, 14}`. 64 pixels per iter.
///
/// # Safety
///
/// Same as [`yuv_444p_n_to_rgb_row`] but `rgb_out: &mut [u16]`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn yuv_444p_n_to_rgb_u16_row<const BITS: u32>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  const { assert!(BITS == 9 || BITS == 10 || BITS == 12 || BITS == 14) };
  debug_assert!(y.len() >= width);
  debug_assert!(u.len() >= width);
  debug_assert!(v.len() >= width);
  debug_assert!(rgb_out.len() >= width * 3);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<BITS, BITS>(full_range);
  let bias = scalar::chroma_bias::<BITS>();
  const RND: i32 = 1 << 14;
  let out_max: i16 = ((1i32 << BITS) - 1) as i16;

  unsafe {
    let rnd_v = _mm512_set1_epi32(RND);
    let y_off_v = _mm512_set1_epi16(y_off as i16);
    let y_scale_v = _mm512_set1_epi32(y_scale);
    let c_scale_v = _mm512_set1_epi32(c_scale);
    let bias_v = _mm512_set1_epi16(bias as i16);
    let mask_v = _mm512_set1_epi16(scalar::bits_mask::<BITS>() as i16);
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
    while x + 64 <= width {
      let y_low_i16 = _mm512_and_si512(_mm512_loadu_si512(y.as_ptr().add(x).cast()), mask_v);
      let y_high_i16 = _mm512_and_si512(_mm512_loadu_si512(y.as_ptr().add(x + 32).cast()), mask_v);
      let u_lo_vec = _mm512_and_si512(_mm512_loadu_si512(u.as_ptr().add(x).cast()), mask_v);
      let u_hi_vec = _mm512_and_si512(_mm512_loadu_si512(u.as_ptr().add(x + 32).cast()), mask_v);
      let v_lo_vec = _mm512_and_si512(_mm512_loadu_si512(v.as_ptr().add(x).cast()), mask_v);
      let v_hi_vec = _mm512_and_si512(_mm512_loadu_si512(v.as_ptr().add(x + 32).cast()), mask_v);

      let u_lo_i16 = _mm512_sub_epi16(u_lo_vec, bias_v);
      let u_hi_i16 = _mm512_sub_epi16(u_hi_vec, bias_v);
      let v_lo_i16 = _mm512_sub_epi16(v_lo_vec, bias_v);
      let v_hi_i16 = _mm512_sub_epi16(v_hi_vec, bias_v);

      let u_lo_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_lo_i16));
      let u_lo_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(u_lo_i16));
      let u_hi_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_hi_i16));
      let u_hi_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(u_hi_i16));
      let v_lo_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_lo_i16));
      let v_lo_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(v_lo_i16));
      let v_hi_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_hi_i16));
      let v_hi_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(v_hi_i16));

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

      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v, pack_fixup);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v, pack_fixup);

      let r_lo = clamp_u16_max_x32(_mm512_adds_epi16(y_scaled_lo, r_chroma_lo), zero_v, max_v);
      let r_hi = clamp_u16_max_x32(_mm512_adds_epi16(y_scaled_hi, r_chroma_hi), zero_v, max_v);
      let g_lo = clamp_u16_max_x32(_mm512_adds_epi16(y_scaled_lo, g_chroma_lo), zero_v, max_v);
      let g_hi = clamp_u16_max_x32(_mm512_adds_epi16(y_scaled_hi, g_chroma_hi), zero_v, max_v);
      let b_lo = clamp_u16_max_x32(_mm512_adds_epi16(y_scaled_lo, b_chroma_lo), zero_v, max_v);
      let b_hi = clamp_u16_max_x32(_mm512_adds_epi16(y_scaled_hi, b_chroma_hi), zero_v, max_v);

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
      scalar::yuv_444p_n_to_rgb_u16_row::<BITS>(
        &y[x..width],
        &u[x..width],
        &v[x..width],
        &mut rgb_out[x * 3..width * 3],
        width - x,
        matrix,
        full_range,
      );
    }
  }
}

/// AVX-512 YUV 4:4:4 planar **16-bit** → packed **u8** RGB. Stays on
/// the i32 Q15 pipeline — output-range scaling keeps `coeff × u_d`
/// within i32 for u8 output. 64 pixels per iter.
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available.**
/// 2. `y.len() >= width`, `u.len() >= width`, `v.len() >= width`,
///    `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn yuv_444p16_to_rgb_row(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(y.len() >= width);
  debug_assert!(u.len() >= width);
  debug_assert!(v.len() >= width);
  debug_assert!(rgb_out.len() >= width * 3);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<16, 8>(full_range);
  const RND: i32 = 1 << 14;

  unsafe {
    let rnd_v = _mm512_set1_epi32(RND);
    let y_off_v = _mm512_set1_epi32(y_off);
    let y_scale_v = _mm512_set1_epi32(y_scale);
    let c_scale_v = _mm512_set1_epi32(c_scale);
    let bias16_v = _mm512_set1_epi16(-32768i16);
    let cru = _mm512_set1_epi32(coeffs.r_u());
    let crv = _mm512_set1_epi32(coeffs.r_v());
    let cgu = _mm512_set1_epi32(coeffs.g_u());
    let cgv = _mm512_set1_epi32(coeffs.g_v());
    let cbu = _mm512_set1_epi32(coeffs.b_u());
    let cbv = _mm512_set1_epi32(coeffs.b_v());
    let pack_fixup = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);

    let mut x = 0usize;
    while x + 64 <= width {
      let y_low = _mm512_loadu_si512(y.as_ptr().add(x).cast());
      let y_high = _mm512_loadu_si512(y.as_ptr().add(x + 32).cast());
      let u_lo_vec = _mm512_loadu_si512(u.as_ptr().add(x).cast());
      let u_hi_vec = _mm512_loadu_si512(u.as_ptr().add(x + 32).cast());
      let v_lo_vec = _mm512_loadu_si512(v.as_ptr().add(x).cast());
      let v_hi_vec = _mm512_loadu_si512(v.as_ptr().add(x + 32).cast());

      let u_lo_i16 = _mm512_sub_epi16(u_lo_vec, bias16_v);
      let u_hi_i16 = _mm512_sub_epi16(u_hi_vec, bias16_v);
      let v_lo_i16 = _mm512_sub_epi16(v_lo_vec, bias16_v);
      let v_hi_i16 = _mm512_sub_epi16(v_hi_vec, bias16_v);

      let u_lo_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_lo_i16));
      let u_lo_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(u_lo_i16));
      let u_hi_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_hi_i16));
      let u_hi_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(u_hi_i16));
      let v_lo_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_lo_i16));
      let v_lo_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(v_lo_i16));
      let v_hi_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_hi_i16));
      let v_hi_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(v_hi_i16));

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

      let y_scaled_lo = scale_y_u16_avx512(y_low, y_off_v, y_scale_v, rnd_v, pack_fixup);
      let y_scaled_hi = scale_y_u16_avx512(y_high, y_off_v, y_scale_v, rnd_v, pack_fixup);

      let r_lo = _mm512_adds_epi16(y_scaled_lo, r_chroma_lo);
      let r_hi = _mm512_adds_epi16(y_scaled_hi, r_chroma_hi);
      let g_lo = _mm512_adds_epi16(y_scaled_lo, g_chroma_lo);
      let g_hi = _mm512_adds_epi16(y_scaled_hi, g_chroma_hi);
      let b_lo = _mm512_adds_epi16(y_scaled_lo, b_chroma_lo);
      let b_hi = _mm512_adds_epi16(y_scaled_hi, b_chroma_hi);

      let r_u8 = narrow_u8x64(r_lo, r_hi, pack_fixup);
      let g_u8 = narrow_u8x64(g_lo, g_hi, pack_fixup);
      let b_u8 = narrow_u8x64(b_lo, b_hi, pack_fixup);

      write_rgb_64(r_u8, g_u8, b_u8, rgb_out.as_mut_ptr().add(x * 3));
      x += 64;
    }

    if x < width {
      scalar::yuv_444p16_to_rgb_row(
        &y[x..width],
        &u[x..width],
        &v[x..width],
        &mut rgb_out[x * 3..width * 3],
        width - x,
        matrix,
        full_range,
      );
    }
  }
}

/// AVX-512 YUV 4:4:4 planar **16-bit** → packed **u16** RGB. Native
/// 512-bit i64-chroma kernel using `_mm512_srai_epi64`.
///
/// Block size 32 pixels per iter. Mirrors
/// [`yuv_420p16_to_rgb_u16_row`] but with full-width chroma loads
/// and no duplication step (4:4:4 is 1:1 with Y).
///
/// # Safety
///
/// Same as [`yuv_444p16_to_rgb_row`] but `rgb_out: &mut [u16]`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn yuv_444p16_to_rgb_u16_row(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(y.len() >= width);
  debug_assert!(u.len() >= width);
  debug_assert!(v.len() >= width);
  debug_assert!(rgb_out.len() >= width * 3);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<16, 16>(full_range);
  const RND_I64: i64 = 1 << 14;
  const RND_I32: i32 = 1 << 14;

  unsafe {
    let rnd_i64_v = _mm512_set1_epi64(RND_I64);
    let rnd_i32_v = _mm512_set1_epi32(RND_I32);
    let y_off_v = _mm512_set1_epi32(y_off);
    let y_scale_v = _mm512_set1_epi32(y_scale);
    let c_scale_v = _mm512_set1_epi32(c_scale);
    let bias16_v = _mm512_set1_epi16(-32768i16);
    let cru = _mm512_set1_epi32(coeffs.r_u());
    let crv = _mm512_set1_epi32(coeffs.r_v());
    let cgu = _mm512_set1_epi32(coeffs.g_u());
    let cgv = _mm512_set1_epi32(coeffs.g_v());
    let cbu = _mm512_set1_epi32(coeffs.b_u());
    let cbv = _mm512_set1_epi32(coeffs.b_v());

    let interleave_idx = _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
    let pack_fixup = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);

    let mut x = 0usize;
    while x + 32 <= width {
      // 32 pixels/iter. 4:4:4: full-width chroma (32 samples each).
      let y_vec = _mm512_loadu_si512(y.as_ptr().add(x).cast());
      let u_vec = _mm512_loadu_si512(u.as_ptr().add(x).cast());
      let v_vec = _mm512_loadu_si512(v.as_ptr().add(x).cast());

      let u_i16 = _mm512_sub_epi16(u_vec, bias16_v);
      let v_i16 = _mm512_sub_epi16(v_vec, bias16_v);

      // Widen each i16x32 → two i32x16 halves.
      let u_lo_i32 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_i16));
      let u_hi_i32 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(u_i16));
      let v_lo_i32 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_i16));
      let v_hi_i32 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(v_i16));

      // Scale in i32 — `u_centered * c_scale` fits i32 after >> 15.
      let u_d_lo = _mm512_srai_epi32::<15>(_mm512_add_epi32(
        _mm512_mullo_epi32(u_lo_i32, c_scale_v),
        rnd_i32_v,
      ));
      let u_d_hi = _mm512_srai_epi32::<15>(_mm512_add_epi32(
        _mm512_mullo_epi32(u_hi_i32, c_scale_v),
        rnd_i32_v,
      ));
      let v_d_lo = _mm512_srai_epi32::<15>(_mm512_add_epi32(
        _mm512_mullo_epi32(v_lo_i32, c_scale_v),
        rnd_i32_v,
      ));
      let v_d_hi = _mm512_srai_epi32::<15>(_mm512_add_epi32(
        _mm512_mullo_epi32(v_hi_i32, c_scale_v),
        rnd_i32_v,
      ));

      // i64 chroma: 4 calls per channel (even/odd of u_d_lo and u_d_hi).
      let u_d_lo_odd = _mm512_shuffle_epi32::<0xF5>(u_d_lo);
      let u_d_hi_odd = _mm512_shuffle_epi32::<0xF5>(u_d_hi);
      let v_d_lo_odd = _mm512_shuffle_epi32::<0xF5>(v_d_lo);
      let v_d_hi_odd = _mm512_shuffle_epi32::<0xF5>(v_d_hi);

      let r_ch_lo_e = chroma_i64x8_avx512(cru, crv, u_d_lo, v_d_lo, rnd_i64_v);
      let r_ch_lo_o = chroma_i64x8_avx512(cru, crv, u_d_lo_odd, v_d_lo_odd, rnd_i64_v);
      let r_ch_hi_e = chroma_i64x8_avx512(cru, crv, u_d_hi, v_d_hi, rnd_i64_v);
      let r_ch_hi_o = chroma_i64x8_avx512(cru, crv, u_d_hi_odd, v_d_hi_odd, rnd_i64_v);
      let g_ch_lo_e = chroma_i64x8_avx512(cgu, cgv, u_d_lo, v_d_lo, rnd_i64_v);
      let g_ch_lo_o = chroma_i64x8_avx512(cgu, cgv, u_d_lo_odd, v_d_lo_odd, rnd_i64_v);
      let g_ch_hi_e = chroma_i64x8_avx512(cgu, cgv, u_d_hi, v_d_hi, rnd_i64_v);
      let g_ch_hi_o = chroma_i64x8_avx512(cgu, cgv, u_d_hi_odd, v_d_hi_odd, rnd_i64_v);
      let b_ch_lo_e = chroma_i64x8_avx512(cbu, cbv, u_d_lo, v_d_lo, rnd_i64_v);
      let b_ch_lo_o = chroma_i64x8_avx512(cbu, cbv, u_d_lo_odd, v_d_lo_odd, rnd_i64_v);
      let b_ch_hi_e = chroma_i64x8_avx512(cbu, cbv, u_d_hi, v_d_hi, rnd_i64_v);
      let b_ch_hi_o = chroma_i64x8_avx512(cbu, cbv, u_d_hi_odd, v_d_hi_odd, rnd_i64_v);

      // Reassemble to i32x16 per half.
      let r_ch_lo = reassemble_i32x16(r_ch_lo_e, r_ch_lo_o, interleave_idx);
      let r_ch_hi = reassemble_i32x16(r_ch_hi_e, r_ch_hi_o, interleave_idx);
      let g_ch_lo = reassemble_i32x16(g_ch_lo_e, g_ch_lo_o, interleave_idx);
      let g_ch_hi = reassemble_i32x16(g_ch_hi_e, g_ch_hi_o, interleave_idx);
      let b_ch_lo = reassemble_i32x16(b_ch_lo_e, b_ch_lo_o, interleave_idx);
      let b_ch_hi = reassemble_i32x16(b_ch_hi_e, b_ch_hi_o, interleave_idx);

      // Y path: widen 32 u16 → two i32x16, subtract y_off, scale in i64.
      let y_lo_u16 = _mm512_castsi512_si256(y_vec);
      let y_hi_u16 = _mm512_extracti64x4_epi64::<1>(y_vec);
      let y_lo_i32 = _mm512_sub_epi32(_mm512_cvtepu16_epi32(y_lo_u16), y_off_v);
      let y_hi_i32 = _mm512_sub_epi32(_mm512_cvtepu16_epi32(y_hi_u16), y_off_v);

      let y_lo_scaled = scale_y_i32x16_i64(y_lo_i32, y_scale_v, rnd_i64_v, interleave_idx);
      let y_hi_scaled = scale_y_i32x16_i64(y_hi_i32, y_scale_v, rnd_i64_v, interleave_idx);

      // Add Y + chroma (no dup — 4:4:4 is 1:1), saturate to u16.
      let r_lo_i32 = _mm512_add_epi32(y_lo_scaled, r_ch_lo);
      let r_hi_i32 = _mm512_add_epi32(y_hi_scaled, r_ch_hi);
      let g_lo_i32 = _mm512_add_epi32(y_lo_scaled, g_ch_lo);
      let g_hi_i32 = _mm512_add_epi32(y_hi_scaled, g_ch_hi);
      let b_lo_i32 = _mm512_add_epi32(y_lo_scaled, b_ch_lo);
      let b_hi_i32 = _mm512_add_epi32(y_hi_scaled, b_ch_hi);

      let r_u16 = _mm512_permutexvar_epi64(pack_fixup, _mm512_packus_epi32(r_lo_i32, r_hi_i32));
      let g_u16 = _mm512_permutexvar_epi64(pack_fixup, _mm512_packus_epi32(g_lo_i32, g_hi_i32));
      let b_u16 = _mm512_permutexvar_epi64(pack_fixup, _mm512_packus_epi32(b_lo_i32, b_hi_i32));

      write_rgb_u16_32(r_u16, g_u16, b_u16, rgb_out.as_mut_ptr().add(x * 3));

      x += 32;
    }

    if x < width {
      scalar::yuv_444p16_to_rgb_u16_row(
        &y[x..width],
        &u[x..width],
        &v[x..width],
        &mut rgb_out[x * 3..width * 3],
        width - x,
        matrix,
        full_range,
      );
    }
  }
}

/// AVX‑512 high‑bit‑packed semi‑planar (`BITS` ∈ {10, 12}) → packed
/// **8‑bit** RGB.
///
/// Block size 64 Y pixels / 32 chroma pairs per iteration. Mirrors
/// [`super::x86_avx512::yuv_420p_n_to_rgb_row`] with two structural
/// differences:
/// - Samples are shifted right by `16 - BITS` (`_mm512_srl_epi16`,
///   with a shift count computed from `BITS` once per call) instead
///   of AND‑masked.
/// - Semi‑planar UV is deinterleaved via [`deinterleave_uv_u16_avx512`]
///   — per‑128‑lane shuffle + 64‑bit permute + cross‑vector
///   `_mm512_permutex2var_epi64` to produce 32‑sample U and V
///   vectors.
///
/// # Numerical contract
///
/// Byte‑identical to [`scalar::p_n_to_rgb_row::<BITS>`] for the
/// monomorphized `BITS`.
///
/// # Safety
///
/// 1. **AVX‑512F + AVX‑512BW must be available on the current CPU.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`,
///    `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn p_n_to_rgb_row<const BITS: u32>(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0);
  debug_assert!(y.len() >= width);
  debug_assert!(uv_half.len() >= width);
  debug_assert!(rgb_out.len() >= width * 3);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<BITS, 8>(full_range);
  let bias = scalar::chroma_bias::<BITS>();
  const RND: i32 = 1 << 14;

  // SAFETY: AVX‑512BW availability is the caller's obligation.
  unsafe {
    let rnd_v = _mm512_set1_epi32(RND);
    let y_off_v = _mm512_set1_epi16(y_off as i16);
    let y_scale_v = _mm512_set1_epi32(y_scale);
    let c_scale_v = _mm512_set1_epi32(c_scale);
    let bias_v = _mm512_set1_epi16(bias as i16);
    // High-bit-packed samples: shift right by `16 - BITS`.
    let shr_count = _mm_cvtsi32_si128((16 - BITS) as i32);
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
      let y_low_i16 = _mm512_srl_epi16(_mm512_loadu_si512(y.as_ptr().add(x).cast()), shr_count);
      let y_high_i16 =
        _mm512_srl_epi16(_mm512_loadu_si512(y.as_ptr().add(x + 32).cast()), shr_count);
      let (u_vec, v_vec) = deinterleave_uv_u16_avx512(uv_half.as_ptr().add(x));
      let u_vec = _mm512_srl_epi16(u_vec, shr_count);
      let v_vec = _mm512_srl_epi16(v_vec, shr_count);

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
      scalar::p_n_to_rgb_row::<BITS>(
        &y[x..width],
        &uv_half[x..width],
        &mut rgb_out[x * 3..width * 3],
        width - x,
        matrix,
        full_range,
      );
    }
  }
}

/// AVX‑512 high‑bit‑packed semi‑planar (`BITS` ∈ {10, 12}) → packed
/// **native‑depth `u16`** RGB (low‑bit‑packed output, `yuv420pNle`
/// convention).
///
/// # Numerical contract
///
/// Byte‑identical to [`scalar::p_n_to_rgb_u16_row::<BITS>`] for the
/// monomorphized `BITS`.
///
/// # Safety
///
/// 1. **AVX‑512F + AVX‑512BW must be available on the current CPU.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`,
///    `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn p_n_to_rgb_u16_row<const BITS: u32>(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0);
  debug_assert!(y.len() >= width);
  debug_assert!(uv_half.len() >= width);
  debug_assert!(rgb_out.len() >= width * 3);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<BITS, BITS>(full_range);
  let bias = scalar::chroma_bias::<BITS>();
  const RND: i32 = 1 << 14;
  let out_max: i16 = ((1i32 << BITS) - 1) as i16;

  // SAFETY: AVX‑512BW availability is the caller's obligation.
  unsafe {
    let rnd_v = _mm512_set1_epi32(RND);
    let y_off_v = _mm512_set1_epi16(y_off as i16);
    let y_scale_v = _mm512_set1_epi32(y_scale);
    let c_scale_v = _mm512_set1_epi32(c_scale);
    let bias_v = _mm512_set1_epi16(bias as i16);
    let max_v = _mm512_set1_epi16(out_max);
    let zero_v = _mm512_set1_epi16(0);
    // High-bit-packed samples: shift right by `16 - BITS`.
    let shr_count = _mm_cvtsi32_si128((16 - BITS) as i32);
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
      let y_low_i16 = _mm512_srl_epi16(_mm512_loadu_si512(y.as_ptr().add(x).cast()), shr_count);
      let y_high_i16 =
        _mm512_srl_epi16(_mm512_loadu_si512(y.as_ptr().add(x + 32).cast()), shr_count);
      let (u_vec, v_vec) = deinterleave_uv_u16_avx512(uv_half.as_ptr().add(x));
      let u_vec = _mm512_srl_epi16(u_vec, shr_count);
      let v_vec = _mm512_srl_epi16(v_vec, shr_count);

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

      let r_lo = clamp_u16_max_x32(_mm512_adds_epi16(y_scaled_lo, r_dup_lo), zero_v, max_v);
      let r_hi = clamp_u16_max_x32(_mm512_adds_epi16(y_scaled_hi, r_dup_hi), zero_v, max_v);
      let g_lo = clamp_u16_max_x32(_mm512_adds_epi16(y_scaled_lo, g_dup_lo), zero_v, max_v);
      let g_hi = clamp_u16_max_x32(_mm512_adds_epi16(y_scaled_hi, g_dup_hi), zero_v, max_v);
      let b_lo = clamp_u16_max_x32(_mm512_adds_epi16(y_scaled_lo, b_dup_lo), zero_v, max_v);
      let b_hi = clamp_u16_max_x32(_mm512_adds_epi16(y_scaled_hi, b_dup_hi), zero_v, max_v);

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
      scalar::p_n_to_rgb_u16_row::<BITS>(
        &y[x..width],
        &uv_half[x..width],
        &mut rgb_out[x * 3..width * 3],
        width - x,
        matrix,
        full_range,
      );
    }
  }
}

/// Deinterleaves 64 `u16` elements at `ptr` into `(u_vec, v_vec)` —
/// two AVX‑512 vectors each holding 32 packed `u16` samples.
///
/// Per‑128‑bit‑lane `_mm512_shuffle_epi8` packs even u16s (U's) into
/// each lane's low 64 bits, odd u16s (V's) into the high 64. Then
/// `_mm512_permutexvar_epi64` with the existing `pack_fixup` index
/// `[0, 2, 4, 6, 1, 3, 5, 7]` rearranges the 64‑bit chunks so each
/// vector becomes `[U0..U15 | V0..V15]`. Finally
/// `_mm512_permutex2var_epi64` combines the two vectors into the
/// full 32‑sample U and V vectors.
///
/// # Safety
///
/// `ptr` must point to at least 128 readable bytes (64 `u16`
/// elements). Caller's `target_feature` must include AVX‑512F +
/// AVX‑512BW.
#[inline(always)]
unsafe fn deinterleave_uv_u16_avx512(ptr: *const u16) -> (__m512i, __m512i) {
  unsafe {
    // Per‑128‑lane mask (same byte pattern replicated across the 4
    // lanes of a `__m512i`).
    let split_mask = _mm512_broadcast_i32x4(_mm_setr_epi8(
      0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15,
    ));
    let pack_fixup = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
    // Cross-vector 2x8 permute indices:
    //   u_vec = low 256 of each vec → chunks [0..3 of a, 0..3 of b]
    //   v_vec = high 256 of each vec → chunks [4..7 of a, 4..7 of b]
    let u_perm = _mm512_setr_epi64(0, 1, 2, 3, 8, 9, 10, 11);
    let v_perm = _mm512_setr_epi64(4, 5, 6, 7, 12, 13, 14, 15);

    let uv0 = _mm512_loadu_si512(ptr.cast());
    let uv1 = _mm512_loadu_si512(ptr.add(32).cast());

    let s0 = _mm512_shuffle_epi8(uv0, split_mask);
    let s1 = _mm512_shuffle_epi8(uv1, split_mask);

    // After per-lane shuffle + per-vector 64-bit permute, each vector
    // is `[U0..U15 | V0..V15]` (low 256 = U's, high 256 = V's).
    let s0_p = _mm512_permutexvar_epi64(pack_fixup, s0);
    let s1_p = _mm512_permutexvar_epi64(pack_fixup, s1);

    let u_vec = _mm512_permutex2var_epi64(s0_p, u_perm, s1_p);
    let v_vec = _mm512_permutex2var_epi64(s0_p, v_perm, s1_p);
    (u_vec, v_vec)
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

/// AVX-512 NV24 → packed RGB (UV-ordered, 4:4:4). Thin wrapper over
/// [`nv24_or_nv42_to_rgb_row_impl`] with `SWAP_UV = false`.
///
/// # Safety
///
/// Same as [`nv24_or_nv42_to_rgb_row_impl`].
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn nv24_to_rgb_row(
  y: &[u8],
  uv: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    nv24_or_nv42_to_rgb_row_impl::<false>(y, uv, rgb_out, width, matrix, full_range);
  }
}

/// AVX-512 NV42 → packed RGB (VU-ordered, 4:4:4).
///
/// # Safety
///
/// Same as [`nv24_or_nv42_to_rgb_row_impl`].
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn nv42_to_rgb_row(
  y: &[u8],
  vu: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    nv24_or_nv42_to_rgb_row_impl::<true>(y, vu, rgb_out, width, matrix, full_range);
  }
}

/// Shared AVX-512 NV24/NV42 kernel (4:4:4 semi-planar). 64 Y pixels /
/// 64 chroma pairs / 128 UV bytes per iteration. Unlike
/// [`nv12_or_nv21_to_rgb_row_impl`], chroma is not subsampled — one
/// UV pair per Y pixel — so the `chroma_dup` step disappears; two
/// `chroma_i16x32` calls per channel produce 64 chroma values
/// directly.
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available.**
/// 2. `y.len() >= width`.
/// 3. `uv_or_vu.len() >= 2 * width`.
/// 4. `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn nv24_or_nv42_to_rgb_row_impl<const SWAP_UV: bool>(
  y: &[u8],
  uv_or_vu: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(y.len() >= width);
  debug_assert!(uv_or_vu.len() >= 2 * width);
  debug_assert!(rgb_out.len() >= width * 3);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params(full_range);
  const RND: i32 = 1 << 14;

  // SAFETY: AVX-512BW availability is the caller's obligation; all
  // pointer adds below are bounded by the `while x + 64 <= width`
  // loop and the caller-promised slice lengths.
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

    // Same lane fixups as NV12 kernel — inherited verbatim.
    let pack_fixup = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);

    // Per-128-bit-lane UV deinterleave mask, broadcast to all 4 lanes.
    let uv_lane_mask = _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);
    let uv_deint_mask = _mm512_broadcast_i32x4(uv_lane_mask);
    // After per-lane shuffle the 64-bit lane layout is
    // `[U0, V0, U1, V1, U2, V2, U3, V3]`; this permute collects to
    // `[U0, U1, U2, U3 | V0, V1, V2, V3]` — low 256 = U, high 256 = V.
    let uv_collect = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);

    let mut x = 0usize;
    while x + 64 <= width {
      let y_vec = _mm512_loadu_si512(y.as_ptr().add(x).cast());
      // 64 Y pixels → 128 UV bytes (two 512-bit loads).
      let uv_vec_lo = _mm512_loadu_si512(uv_or_vu.as_ptr().add(x * 2).cast());
      let uv_vec_hi = _mm512_loadu_si512(uv_or_vu.as_ptr().add(x * 2 + 64).cast());

      // Per 512-bit vec: deinterleave → low 256 = U (32 bytes),
      // high 256 = V (32 bytes). Roles swap for NV42.
      let d_lo =
        _mm512_permutexvar_epi64(uv_collect, _mm512_shuffle_epi8(uv_vec_lo, uv_deint_mask));
      let d_hi =
        _mm512_permutexvar_epi64(uv_collect, _mm512_shuffle_epi8(uv_vec_hi, uv_deint_mask));
      let (u_bytes_lo_256, v_bytes_lo_256, u_bytes_hi_256, v_bytes_hi_256) = if SWAP_UV {
        (
          _mm512_extracti64x4_epi64::<1>(d_lo),
          _mm512_castsi512_si256(d_lo),
          _mm512_extracti64x4_epi64::<1>(d_hi),
          _mm512_castsi512_si256(d_hi),
        )
      } else {
        (
          _mm512_castsi512_si256(d_lo),
          _mm512_extracti64x4_epi64::<1>(d_lo),
          _mm512_castsi512_si256(d_hi),
          _mm512_extracti64x4_epi64::<1>(d_hi),
        )
      };

      // Widen each 32-byte U/V chunk to i16x32 and subtract 128.
      let u_lo_i16 = _mm512_sub_epi16(_mm512_cvtepu8_epi16(u_bytes_lo_256), mid128);
      let u_hi_i16 = _mm512_sub_epi16(_mm512_cvtepu8_epi16(u_bytes_hi_256), mid128);
      let v_lo_i16 = _mm512_sub_epi16(_mm512_cvtepu8_epi16(v_bytes_lo_256), mid128);
      let v_hi_i16 = _mm512_sub_epi16(_mm512_cvtepu8_epi16(v_bytes_hi_256), mid128);

      // Split each i16x32 into two i32x16 halves.
      let u_lo_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_lo_i16));
      let u_lo_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(u_lo_i16));
      let u_hi_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_hi_i16));
      let u_hi_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(u_hi_i16));
      let v_lo_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_lo_i16));
      let v_lo_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(v_lo_i16));
      let v_hi_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_hi_i16));
      let v_hi_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(v_hi_i16));

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

      // 64 chroma per channel (two chroma_i16x32 per channel, no
      // duplication).
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

      let y_low_i16 = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(y_vec));
      let y_high_i16 = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64::<1>(y_vec));
      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v, pack_fixup);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v, pack_fixup);

      let b_lo = _mm512_adds_epi16(y_scaled_lo, b_chroma_lo);
      let b_hi = _mm512_adds_epi16(y_scaled_hi, b_chroma_hi);
      let g_lo = _mm512_adds_epi16(y_scaled_lo, g_chroma_lo);
      let g_hi = _mm512_adds_epi16(y_scaled_hi, g_chroma_hi);
      let r_lo = _mm512_adds_epi16(y_scaled_lo, r_chroma_lo);
      let r_hi = _mm512_adds_epi16(y_scaled_hi, r_chroma_hi);

      let b_u8 = narrow_u8x64(b_lo, b_hi, pack_fixup);
      let g_u8 = narrow_u8x64(g_lo, g_hi, pack_fixup);
      let r_u8 = narrow_u8x64(r_lo, r_hi, pack_fixup);

      write_rgb_64(r_u8, g_u8, b_u8, rgb_out.as_mut_ptr().add(x * 3));

      x += 64;
    }

    if x < width {
      if SWAP_UV {
        scalar::nv42_to_rgb_row(
          &y[x..width],
          &uv_or_vu[x * 2..width * 2],
          &mut rgb_out[x * 3..width * 3],
          width - x,
          matrix,
          full_range,
        );
      } else {
        scalar::nv24_to_rgb_row(
          &y[x..width],
          &uv_or_vu[x * 2..width * 2],
          &mut rgb_out[x * 3..width * 3],
          width - x,
          matrix,
          full_range,
        );
      }
    }
  }
}

/// AVX-512 YUV 4:4:4 planar → packed RGB. 64 Y pixels + 64 U + 64 V
/// per iteration. Same arithmetic as [`nv24_to_rgb_row`] with U / V
/// loaded directly from separate planes (no deinterleave step).
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available.**
/// 2. `y.len() >= width`, `u.len() >= width`, `v.len() >= width`,
///    `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn yuv_444_to_rgb_row(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(y.len() >= width);
  debug_assert!(u.len() >= width);
  debug_assert!(v.len() >= width);
  debug_assert!(rgb_out.len() >= width * 3);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params(full_range);
  const RND: i32 = 1 << 14;

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

    let mut x = 0usize;
    while x + 64 <= width {
      let y_vec = _mm512_loadu_si512(y.as_ptr().add(x).cast());
      // 4:4:4: 64 U + 64 V directly, no deinterleave.
      let u_vec = _mm512_loadu_si512(u.as_ptr().add(x).cast());
      let v_vec = _mm512_loadu_si512(v.as_ptr().add(x).cast());

      // Widen low / high halves of U / V (32 bytes each) to i16x32.
      let u_lo_i16 = _mm512_sub_epi16(_mm512_cvtepu8_epi16(_mm512_castsi512_si256(u_vec)), mid128);
      let u_hi_i16 = _mm512_sub_epi16(
        _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64::<1>(u_vec)),
        mid128,
      );
      let v_lo_i16 = _mm512_sub_epi16(_mm512_cvtepu8_epi16(_mm512_castsi512_si256(v_vec)), mid128);
      let v_hi_i16 = _mm512_sub_epi16(
        _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64::<1>(v_vec)),
        mid128,
      );

      let u_lo_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_lo_i16));
      let u_lo_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(u_lo_i16));
      let u_hi_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_hi_i16));
      let u_hi_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(u_hi_i16));
      let v_lo_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_lo_i16));
      let v_lo_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(v_lo_i16));
      let v_hi_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_hi_i16));
      let v_hi_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(v_hi_i16));

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

      let y_low_i16 = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(y_vec));
      let y_high_i16 = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64::<1>(y_vec));
      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v, pack_fixup);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v, pack_fixup);

      let b_lo = _mm512_adds_epi16(y_scaled_lo, b_chroma_lo);
      let b_hi = _mm512_adds_epi16(y_scaled_hi, b_chroma_hi);
      let g_lo = _mm512_adds_epi16(y_scaled_lo, g_chroma_lo);
      let g_hi = _mm512_adds_epi16(y_scaled_hi, g_chroma_hi);
      let r_lo = _mm512_adds_epi16(y_scaled_lo, r_chroma_lo);
      let r_hi = _mm512_adds_epi16(y_scaled_hi, r_chroma_hi);

      let b_u8 = narrow_u8x64(b_lo, b_hi, pack_fixup);
      let g_u8 = narrow_u8x64(g_lo, g_hi, pack_fixup);
      let r_u8 = narrow_u8x64(r_lo, r_hi, pack_fixup);

      write_rgb_64(r_u8, g_u8, b_u8, rgb_out.as_mut_ptr().add(x * 3));

      x += 64;
    }

    if x < width {
      scalar::yuv_444_to_rgb_row(
        &y[x..width],
        &u[x..width],
        &v[x..width],
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

/// Writes 64 pixels of packed RGBA (256 bytes) by splitting the
/// u8x64 channel vectors into four 128‑bit quarters and calling the
/// shared [`write_rgba_16`] helper four times.
///
/// # Safety
///
/// `ptr` must point to at least 256 writable bytes.
#[inline(always)]
unsafe fn write_rgba_64(r: __m512i, g: __m512i, b: __m512i, a: __m512i, ptr: *mut u8) {
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
    let a0: __m128i = _mm512_castsi512_si128(a);
    let a1: __m128i = _mm512_extracti32x4_epi32::<1>(a);
    let a2: __m128i = _mm512_extracti32x4_epi32::<2>(a);
    let a3: __m128i = _mm512_extracti32x4_epi32::<3>(a);

    write_rgba_16(r0, g0, b0, a0, ptr);
    write_rgba_16(r1, g1, b1, a1, ptr.add(64));
    write_rgba_16(r2, g2, b2, a2, ptr.add(128));
    write_rgba_16(r3, g3, b3, a3, ptr.add(192));
  }
}

// ===== 16-bit u16-output helpers ========================================

/// `(c_u * u_d + c_v * v_d + RND) >> 15` in i64 via two
/// `_mm512_mul_epi32` (even i32 lanes → i64x8 products) plus native
/// `_mm512_srai_epi64`. Result is i64x8 with each lane's low 32 bits
/// holding the i32-range output.
#[inline(always)]
fn chroma_i64x8_avx512(
  cu: __m512i,
  cv: __m512i,
  u_d_even: __m512i,
  v_d_even: __m512i,
  rnd_i64: __m512i,
) -> __m512i {
  unsafe {
    _mm512_srai_epi64::<15>(_mm512_add_epi64(
      _mm512_add_epi64(
        _mm512_mul_epi32(cu, u_d_even),
        _mm512_mul_epi32(cv, v_d_even),
      ),
      rnd_i64,
    ))
  }
}

/// Combines two i64x8 results (one from even-indexed, one from
/// odd-indexed i32 lanes of the pre-multiply source) into an
/// interleaved i32x16 `[even_0, odd_0, even_1, odd_1, ..., even_7,
/// odd_7]` — i.e. the natural sequential order of 16 scalar results.
///
/// Each i64 lane's low 32 bits contain the result (high 32 are sign);
/// `_mm512_cvtepi64_epi32` truncates to i32x8 per vector, then
/// `_mm512_permutex2var_epi32` interleaves them.
#[inline(always)]
fn reassemble_i32x16(even_i64: __m512i, odd_i64: __m512i, interleave_idx: __m512i) -> __m512i {
  unsafe {
    let even_i32 = _mm512_cvtepi64_epi32(even_i64); // __m256i i32x8
    let odd_i32 = _mm512_cvtepi64_epi32(odd_i64);
    _mm512_permutex2var_epi32(
      _mm512_castsi256_si512(even_i32),
      interleave_idx,
      _mm512_castsi256_si512(odd_i32),
    )
  }
}

/// `(y_minus_off * y_scale + RND) >> 15` computed in i64 for all 16
/// lanes of an i32x16 Y stream, returning an i32x16 result. Needs
/// i64 because limited-range 16→u16 `(Y - y_off) * y_scale` can
/// reach ~2.35·10⁹ (> i32::MAX). Splits the input into even and
/// odd-indexed i32 lanes, multiplies each set via
/// `_mm512_mul_epi32`, shifts in i64, and reassembles to i32x16.
#[inline(always)]
fn scale_y_i32x16_i64(
  y_minus_off: __m512i,
  y_scale_v: __m512i,
  rnd_i64: __m512i,
  interleave_idx: __m512i,
) -> __m512i {
  unsafe {
    let even = _mm512_srai_epi64::<15>(_mm512_add_epi64(
      _mm512_mul_epi32(y_scale_v, y_minus_off),
      rnd_i64,
    ));
    let odd = _mm512_srai_epi64::<15>(_mm512_add_epi64(
      _mm512_mul_epi32(y_scale_v, _mm512_shuffle_epi32::<0xF5>(y_minus_off)),
      rnd_i64,
    ));
    reassemble_i32x16(even, odd, interleave_idx)
  }
}

/// Writes 32 pixels of packed RGB-u16 (192 u16 = 384 bytes) by
/// splitting each u16x32 channel vector into four 128-bit halves and
/// calling the shared [`write_rgb_u16_8`] helper four times.
///
/// # Safety
///
/// `ptr` must point to at least 384 writable bytes.
#[inline(always)]
unsafe fn write_rgb_u16_32(r: __m512i, g: __m512i, b: __m512i, ptr: *mut u16) {
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

    // Each `write_rgb_u16_8` writes 8 pixels × 3 × u16 = 48 bytes =
    // 24 u16 elements. Four calls → 96 u16 = 32 pixels.
    write_rgb_u16_8(r0, g0, b0, ptr);
    write_rgb_u16_8(r1, g1, b1, ptr.add(24));
    write_rgb_u16_8(r2, g2, b2, ptr.add(48));
    write_rgb_u16_8(r3, g3, b3, ptr.add(72));
  }
}

// ===== 16-bit YUV → RGB ==================================================

/// `(Y_u16x32 - y_off) * y_scale + RND >> 15` for full u16 Y samples.
/// Unsigned widening via `_mm512_cvtepu16_epi32`. Returns i16x32.
#[inline(always)]
fn scale_y_u16_avx512(
  y_u16x32: __m512i,
  y_off_v: __m512i,
  y_scale_v: __m512i,
  rnd: __m512i,
  pack_fixup: __m512i,
) -> __m512i {
  unsafe {
    let y_lo_i32 = _mm512_sub_epi32(
      _mm512_cvtepu16_epi32(_mm512_castsi512_si256(y_u16x32)),
      y_off_v,
    );
    let y_hi_i32 = _mm512_sub_epi32(
      _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64::<1>(y_u16x32)),
      y_off_v,
    );
    let lo = _mm512_srai_epi32::<15>(_mm512_add_epi32(
      _mm512_mullo_epi32(y_lo_i32, y_scale_v),
      rnd,
    ));
    let hi = _mm512_srai_epi32::<15>(_mm512_add_epi32(
      _mm512_mullo_epi32(y_hi_i32, y_scale_v),
      rnd,
    ));
    _mm512_permutexvar_epi64(pack_fixup, _mm512_packs_epi32(lo, hi))
  }
}

/// AVX-512 YUV 4:2:0 16-bit → packed **8-bit** RGB. 64 pixels per iteration.
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `u_half.len() >= width / 2`,
///    `v_half.len() >= width / 2`, `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn yuv_420p16_to_rgb_row(
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
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<16, 8>(full_range);
  const RND: i32 = 1 << 14;

  unsafe {
    let rnd_v = _mm512_set1_epi32(RND);
    let y_off_v = _mm512_set1_epi32(y_off);
    let y_scale_v = _mm512_set1_epi32(y_scale);
    let c_scale_v = _mm512_set1_epi32(c_scale);
    let bias16_v = _mm512_set1_epi16(-32768i16);
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
      let y_low = _mm512_loadu_si512(y.as_ptr().add(x).cast());
      let y_high = _mm512_loadu_si512(y.as_ptr().add(x + 32).cast());
      let u_vec = _mm512_loadu_si512(u_half.as_ptr().add(x / 2).cast());
      let v_vec = _mm512_loadu_si512(v_half.as_ptr().add(x / 2).cast());

      let u_i16 = _mm512_sub_epi16(u_vec, bias16_v);
      let v_i16 = _mm512_sub_epi16(v_vec, bias16_v);

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

      let y_scaled_lo = scale_y_u16_avx512(y_low, y_off_v, y_scale_v, rnd_v, pack_fixup);
      let y_scaled_hi = scale_y_u16_avx512(y_high, y_off_v, y_scale_v, rnd_v, pack_fixup);

      let r_lo = _mm512_adds_epi16(y_scaled_lo, r_dup_lo);
      let r_hi = _mm512_adds_epi16(y_scaled_hi, r_dup_hi);
      let g_lo = _mm512_adds_epi16(y_scaled_lo, g_dup_lo);
      let g_hi = _mm512_adds_epi16(y_scaled_hi, g_dup_hi);
      let b_lo = _mm512_adds_epi16(y_scaled_lo, b_dup_lo);
      let b_hi = _mm512_adds_epi16(y_scaled_hi, b_dup_hi);

      let r_u8 = narrow_u8x64(r_lo, r_hi, pack_fixup);
      let g_u8 = narrow_u8x64(g_lo, g_hi, pack_fixup);
      let b_u8 = narrow_u8x64(b_lo, b_hi, pack_fixup);

      write_rgb_64(r_u8, g_u8, b_u8, rgb_out.as_mut_ptr().add(x * 3));
      x += 64;
    }

    if x < width {
      scalar::yuv_420p16_to_rgb_row(
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

/// AVX-512 YUV 4:2:0 16-bit → packed **16-bit** RGB. Native 512-bit
/// implementation, 32 pixels per iteration.
///
/// Uses AVX-512's native `_mm512_srai_epi64` — unlike the SSE4.1 /
/// AVX2 u16 paths which need the `srai64_15` bias trick. Processes
/// chroma and Y in i64 lanes via `_mm512_mul_epi32` (even i32 lanes
/// → i64x8 products), handling even- and odd-indexed lanes
/// separately and reassembling via `_mm512_permutex2var_epi32`.
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `u_half.len() >= width / 2`,
///    `v_half.len() >= width / 2`, `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn yuv_420p16_to_rgb_u16_row(
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
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<16, 16>(full_range);
  const RND_I64: i64 = 1 << 14;
  const RND_I32: i32 = 1 << 14;

  // SAFETY: AVX-512BW availability is the caller's obligation; pointer
  // adds below are bounded by `while x + 32 <= width` and the caller-
  // promised slice lengths.
  unsafe {
    let rnd_i64_v = _mm512_set1_epi64(RND_I64);
    let rnd_i32_v = _mm512_set1_epi32(RND_I32);
    let y_off_v = _mm512_set1_epi32(y_off);
    let y_scale_v = _mm512_set1_epi32(y_scale);
    let c_scale_v = _mm512_set1_epi32(c_scale);
    // UV centering: subtract 32768 via wrapping i16 add of -32768.
    let bias16_v = _mm512_set1_epi16(-32768i16);
    let cru = _mm512_set1_epi32(coeffs.r_u());
    let crv = _mm512_set1_epi32(coeffs.r_v());
    let cgu = _mm512_set1_epi32(coeffs.g_u());
    let cgv = _mm512_set1_epi32(coeffs.g_v());
    let cbu = _mm512_set1_epi32(coeffs.b_u());
    let cbv = _mm512_set1_epi32(coeffs.b_v());

    // Permute indices, built once per call.
    let dup_lo_idx = _mm512_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7);
    let dup_hi_idx = _mm512_setr_epi32(8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15);
    // Interleave two i32x8 (each in the low 256 bits of a __m512i) into
    // an i32x16: [e0, o0, e1, o1, ..., e7, o7].
    let interleave_idx = _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
    // `_mm512_packus_epi32` per-128-bit-lane fixup (same as the 8-bit
    // AVX-512 kernels).
    let pack_fixup = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);

    let mut x = 0usize;
    while x + 32 <= width {
      // 32 Y pixels / 16 chroma pairs per iter. The Y load is a
      // full 512-bit read (32 × u16); UV only needs 16 × u16 per
      // plane, so a 256-bit load is sufficient — a 512-bit load
      // would read past the end of `u_half` / `v_half` on the last
      // iteration where `x / 2 + 16 == u_half.len()`.
      let y_vec = _mm512_loadu_si512(y.as_ptr().add(x).cast());
      let u_vec = _mm256_loadu_si256(u_half.as_ptr().add(x / 2).cast());
      let v_vec = _mm256_loadu_si256(v_half.as_ptr().add(x / 2).cast());

      // Center UV by subtracting 32768 (wrapping i16 sub). Using
      // `_mm256_sub_epi16` with bias16 (which carries -32768 as i16):
      // `sample - (-32768) == sample + 32768` mod 2^16, giving the
      // centered signed value.
      let u_i16 = _mm256_sub_epi16(u_vec, _mm512_castsi512_si256(bias16_v));
      let v_i16 = _mm256_sub_epi16(v_vec, _mm512_castsi512_si256(bias16_v));

      // Widen i16x16 → i32x16 (each value sign-extended).
      let u_i32 = _mm512_cvtepi16_epi32(u_i16);
      let v_i32 = _mm512_cvtepi16_epi32(v_i16);

      // Scale UV in i32: |u_centered * c_scale| peaks at
      // 32768 * ~38300 ≈ 1.26·10⁹ — fits i32. Result `u_d` range is
      // ~[-37450, 37449].
      let u_d = _mm512_srai_epi32::<15>(_mm512_add_epi32(
        _mm512_mullo_epi32(u_i32, c_scale_v),
        rnd_i32_v,
      ));
      let v_d = _mm512_srai_epi32::<15>(_mm512_add_epi32(
        _mm512_mullo_epi32(v_i32, c_scale_v),
        rnd_i32_v,
      ));

      // Chroma in i64 via `_mm512_mul_epi32` on even lanes, then
      // shuffle odd lanes to even and repeat. Each call produces 8
      // i64 products from the 8 even-indexed i32 lanes.
      let u_d_odd = _mm512_shuffle_epi32::<0xF5>(u_d); // lanes [1,1,3,3,5,5,7,7] per 128-bit
      let v_d_odd = _mm512_shuffle_epi32::<0xF5>(v_d);

      let r_ch_even = chroma_i64x8_avx512(cru, crv, u_d, v_d, rnd_i64_v);
      let r_ch_odd = chroma_i64x8_avx512(cru, crv, u_d_odd, v_d_odd, rnd_i64_v);
      let g_ch_even = chroma_i64x8_avx512(cgu, cgv, u_d, v_d, rnd_i64_v);
      let g_ch_odd = chroma_i64x8_avx512(cgu, cgv, u_d_odd, v_d_odd, rnd_i64_v);
      let b_ch_even = chroma_i64x8_avx512(cbu, cbv, u_d, v_d, rnd_i64_v);
      let b_ch_odd = chroma_i64x8_avx512(cbu, cbv, u_d_odd, v_d_odd, rnd_i64_v);

      // Reassemble i64x8 pairs to i32x16.
      let r_ch_i32 = reassemble_i32x16(r_ch_even, r_ch_odd, interleave_idx);
      let g_ch_i32 = reassemble_i32x16(g_ch_even, g_ch_odd, interleave_idx);
      let b_ch_i32 = reassemble_i32x16(b_ch_even, b_ch_odd, interleave_idx);

      // Duplicate chroma for 2 Y pixels each: 16 chroma → 32 values.
      let r_dup_lo = _mm512_permutexvar_epi32(dup_lo_idx, r_ch_i32);
      let r_dup_hi = _mm512_permutexvar_epi32(dup_hi_idx, r_ch_i32);
      let g_dup_lo = _mm512_permutexvar_epi32(dup_lo_idx, g_ch_i32);
      let g_dup_hi = _mm512_permutexvar_epi32(dup_hi_idx, g_ch_i32);
      let b_dup_lo = _mm512_permutexvar_epi32(dup_lo_idx, b_ch_i32);
      let b_dup_hi = _mm512_permutexvar_epi32(dup_hi_idx, b_ch_i32);

      // Y path: split 32 u16 into two halves, widen to i32x16 each,
      // subtract y_off, scale in i64.
      let y_lo_u16 = _mm512_castsi512_si256(y_vec);
      let y_hi_u16 = _mm512_extracti64x4_epi64::<1>(y_vec);
      let y_lo_i32 = _mm512_sub_epi32(_mm512_cvtepu16_epi32(y_lo_u16), y_off_v);
      let y_hi_i32 = _mm512_sub_epi32(_mm512_cvtepu16_epi32(y_hi_u16), y_off_v);

      let y_lo_scaled = scale_y_i32x16_i64(y_lo_i32, y_scale_v, rnd_i64_v, interleave_idx);
      let y_hi_scaled = scale_y_i32x16_i64(y_hi_i32, y_scale_v, rnd_i64_v, interleave_idx);

      // Add Y + chroma, pack with unsigned saturation to u16.
      let r_lo_i32 = _mm512_add_epi32(y_lo_scaled, r_dup_lo);
      let r_hi_i32 = _mm512_add_epi32(y_hi_scaled, r_dup_hi);
      let g_lo_i32 = _mm512_add_epi32(y_lo_scaled, g_dup_lo);
      let g_hi_i32 = _mm512_add_epi32(y_hi_scaled, g_dup_hi);
      let b_lo_i32 = _mm512_add_epi32(y_lo_scaled, b_dup_lo);
      let b_hi_i32 = _mm512_add_epi32(y_hi_scaled, b_dup_hi);

      // `_mm512_packus_epi32` signed i32 → unsigned u16 with
      // saturation. Produces u16x32 with per-128-bit-lane split order;
      // fix via `pack_fixup` permute (same as NV12 AVX-512 u8 kernel).
      let r_u16 = _mm512_permutexvar_epi64(pack_fixup, _mm512_packus_epi32(r_lo_i32, r_hi_i32));
      let g_u16 = _mm512_permutexvar_epi64(pack_fixup, _mm512_packus_epi32(g_lo_i32, g_hi_i32));
      let b_u16 = _mm512_permutexvar_epi64(pack_fixup, _mm512_packus_epi32(b_lo_i32, b_hi_i32));

      // Write 32 pixels (96 u16) via 4× 8-pixel helper.
      write_rgb_u16_32(r_u16, g_u16, b_u16, rgb_out.as_mut_ptr().add(x * 3));

      x += 32;
    }

    if x < width {
      scalar::yuv_420p16_to_rgb_u16_row(
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

/// AVX-512 P016 → packed **8-bit** RGB. 64 pixels per iteration.
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`, `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn p16_to_rgb_row(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0);
  debug_assert!(y.len() >= width);
  debug_assert!(uv_half.len() >= width);
  debug_assert!(rgb_out.len() >= width * 3);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<16, 8>(full_range);
  const RND: i32 = 1 << 14;

  unsafe {
    let rnd_v = _mm512_set1_epi32(RND);
    let y_off_v = _mm512_set1_epi32(y_off);
    let y_scale_v = _mm512_set1_epi32(y_scale);
    let c_scale_v = _mm512_set1_epi32(c_scale);
    let bias16_v = _mm512_set1_epi16(-32768i16);
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
      let y_low = _mm512_loadu_si512(y.as_ptr().add(x).cast());
      let y_high = _mm512_loadu_si512(y.as_ptr().add(x + 32).cast());
      let (u_vec, v_vec) = deinterleave_uv_u16_avx512(uv_half.as_ptr().add(x));

      let u_i16 = _mm512_sub_epi16(u_vec, bias16_v);
      let v_i16 = _mm512_sub_epi16(v_vec, bias16_v);

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

      let y_scaled_lo = scale_y_u16_avx512(y_low, y_off_v, y_scale_v, rnd_v, pack_fixup);
      let y_scaled_hi = scale_y_u16_avx512(y_high, y_off_v, y_scale_v, rnd_v, pack_fixup);

      let r_lo = _mm512_adds_epi16(y_scaled_lo, r_dup_lo);
      let r_hi = _mm512_adds_epi16(y_scaled_hi, r_dup_hi);
      let g_lo = _mm512_adds_epi16(y_scaled_lo, g_dup_lo);
      let g_hi = _mm512_adds_epi16(y_scaled_hi, g_dup_hi);
      let b_lo = _mm512_adds_epi16(y_scaled_lo, b_dup_lo);
      let b_hi = _mm512_adds_epi16(y_scaled_hi, b_dup_hi);

      let r_u8 = narrow_u8x64(r_lo, r_hi, pack_fixup);
      let g_u8 = narrow_u8x64(g_lo, g_hi, pack_fixup);
      let b_u8 = narrow_u8x64(b_lo, b_hi, pack_fixup);

      write_rgb_64(r_u8, g_u8, b_u8, rgb_out.as_mut_ptr().add(x * 3));
      x += 64;
    }

    if x < width {
      scalar::p16_to_rgb_row(
        &y[x..width],
        &uv_half[x..width],
        &mut rgb_out[x * 3..width * 3],
        width - x,
        matrix,
        full_range,
      );
    }
  }
}

/// AVX-512 P016 → packed **16-bit** RGB.
/// Delegates to SSE4.1 (i64 arithmetic).
///
/// # Safety
///
/// Same as [`p16_to_rgb_row`] but `rgb_out` is `&mut [u16]`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn p16_to_rgb_u16_row(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  unsafe {
    p16_to_rgb_u16_row_impl(y, uv_half, rgb_out, width, matrix, full_range);
  }
}

/// Native AVX-512 P016 → u16 kernel, 32 pixels per iter. Shares the
/// 16-bit u16 arithmetic structure with
/// [`yuv_420p16_to_rgb_u16_row`]; the only difference is an inline
/// U/V deinterleave at the UV load.
///
/// # Safety
///
/// Must be called from an AVX-512BW-enabled context. Caller upholds
/// `width & 1 == 0`, `y.len() >= width`, `uv_half.len() >= width`,
/// `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn p16_to_rgb_u16_row_impl(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0);
  debug_assert!(y.len() >= width);
  debug_assert!(uv_half.len() >= width);
  debug_assert!(rgb_out.len() >= width * 3);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<16, 16>(full_range);
  const RND_I64: i64 = 1 << 14;
  const RND_I32: i32 = 1 << 14;

  // SAFETY: AVX-512BW availability is the caller's obligation; pointer
  // adds are bounded by `while x + 32 <= width` and caller-promised
  // slice lengths.
  unsafe {
    let rnd_i64_v = _mm512_set1_epi64(RND_I64);
    let rnd_i32_v = _mm512_set1_epi32(RND_I32);
    let y_off_v = _mm512_set1_epi32(y_off);
    let y_scale_v = _mm512_set1_epi32(y_scale);
    let c_scale_v = _mm512_set1_epi32(c_scale);
    let bias16_v = _mm512_set1_epi16(-32768i16);
    let cru = _mm512_set1_epi32(coeffs.r_u());
    let crv = _mm512_set1_epi32(coeffs.r_v());
    let cgu = _mm512_set1_epi32(coeffs.g_u());
    let cgv = _mm512_set1_epi32(coeffs.g_v());
    let cbu = _mm512_set1_epi32(coeffs.b_u());
    let cbv = _mm512_set1_epi32(coeffs.b_v());

    let dup_lo_idx = _mm512_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7);
    let dup_hi_idx = _mm512_setr_epi32(8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15);
    let interleave_idx = _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
    let pack_fixup = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);

    // Per-128-bit-lane shuffle to deinterleave u16 UV pairs within
    // each lane: `[u0,v0,u1,v1,u2,v2,u3,v3] → [u0,u1,u2,u3,v0,v1,v2,v3]`
    // as u16 = byte indices `[0,1, 4,5, 8,9, 12,13 | 2,3, 6,7, 10,11, 14,15]`.
    let uv_lane_mask = _mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15);
    let uv_deint_mask = _mm512_broadcast_i32x4(uv_lane_mask);
    // After the per-lane shuffle the 64-bit lane layout is
    // `[U0_3, V0_3, U4_7, V4_7, U8_11, V8_11, U12_15, V12_15]`; permute
    // to `[U0_3, U4_7, U8_11, U12_15 | V0_3, V4_7, V8_11, V12_15]` so
    // low 256 = all U, high 256 = all V.
    let uv_collect = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);

    let mut x = 0usize;
    while x + 32 <= width {
      let y_vec = _mm512_loadu_si512(y.as_ptr().add(x).cast());
      // 16 UV pairs = 32 u16 = one 512-bit load.
      let uv_raw = _mm512_loadu_si512(uv_half.as_ptr().add(x).cast());
      let uv_deint = _mm512_shuffle_epi8(uv_raw, uv_deint_mask);
      let uv_split = _mm512_permutexvar_epi64(uv_collect, uv_deint);
      let u_vec = _mm512_castsi512_si256(uv_split);
      let v_vec = _mm512_extracti64x4_epi64::<1>(uv_split);

      // Center UV (same wrapping trick as planar kernel).
      let u_i16 = _mm256_sub_epi16(u_vec, _mm512_castsi512_si256(bias16_v));
      let v_i16 = _mm256_sub_epi16(v_vec, _mm512_castsi512_si256(bias16_v));

      let u_i32 = _mm512_cvtepi16_epi32(u_i16);
      let v_i32 = _mm512_cvtepi16_epi32(v_i16);

      let u_d = _mm512_srai_epi32::<15>(_mm512_add_epi32(
        _mm512_mullo_epi32(u_i32, c_scale_v),
        rnd_i32_v,
      ));
      let v_d = _mm512_srai_epi32::<15>(_mm512_add_epi32(
        _mm512_mullo_epi32(v_i32, c_scale_v),
        rnd_i32_v,
      ));

      let u_d_odd = _mm512_shuffle_epi32::<0xF5>(u_d);
      let v_d_odd = _mm512_shuffle_epi32::<0xF5>(v_d);

      let r_ch_even = chroma_i64x8_avx512(cru, crv, u_d, v_d, rnd_i64_v);
      let r_ch_odd = chroma_i64x8_avx512(cru, crv, u_d_odd, v_d_odd, rnd_i64_v);
      let g_ch_even = chroma_i64x8_avx512(cgu, cgv, u_d, v_d, rnd_i64_v);
      let g_ch_odd = chroma_i64x8_avx512(cgu, cgv, u_d_odd, v_d_odd, rnd_i64_v);
      let b_ch_even = chroma_i64x8_avx512(cbu, cbv, u_d, v_d, rnd_i64_v);
      let b_ch_odd = chroma_i64x8_avx512(cbu, cbv, u_d_odd, v_d_odd, rnd_i64_v);

      let r_ch_i32 = reassemble_i32x16(r_ch_even, r_ch_odd, interleave_idx);
      let g_ch_i32 = reassemble_i32x16(g_ch_even, g_ch_odd, interleave_idx);
      let b_ch_i32 = reassemble_i32x16(b_ch_even, b_ch_odd, interleave_idx);

      let r_dup_lo = _mm512_permutexvar_epi32(dup_lo_idx, r_ch_i32);
      let r_dup_hi = _mm512_permutexvar_epi32(dup_hi_idx, r_ch_i32);
      let g_dup_lo = _mm512_permutexvar_epi32(dup_lo_idx, g_ch_i32);
      let g_dup_hi = _mm512_permutexvar_epi32(dup_hi_idx, g_ch_i32);
      let b_dup_lo = _mm512_permutexvar_epi32(dup_lo_idx, b_ch_i32);
      let b_dup_hi = _mm512_permutexvar_epi32(dup_hi_idx, b_ch_i32);

      let y_lo_u16 = _mm512_castsi512_si256(y_vec);
      let y_hi_u16 = _mm512_extracti64x4_epi64::<1>(y_vec);
      let y_lo_i32 = _mm512_sub_epi32(_mm512_cvtepu16_epi32(y_lo_u16), y_off_v);
      let y_hi_i32 = _mm512_sub_epi32(_mm512_cvtepu16_epi32(y_hi_u16), y_off_v);

      let y_lo_scaled = scale_y_i32x16_i64(y_lo_i32, y_scale_v, rnd_i64_v, interleave_idx);
      let y_hi_scaled = scale_y_i32x16_i64(y_hi_i32, y_scale_v, rnd_i64_v, interleave_idx);

      let r_lo_i32 = _mm512_add_epi32(y_lo_scaled, r_dup_lo);
      let r_hi_i32 = _mm512_add_epi32(y_hi_scaled, r_dup_hi);
      let g_lo_i32 = _mm512_add_epi32(y_lo_scaled, g_dup_lo);
      let g_hi_i32 = _mm512_add_epi32(y_hi_scaled, g_dup_hi);
      let b_lo_i32 = _mm512_add_epi32(y_lo_scaled, b_dup_lo);
      let b_hi_i32 = _mm512_add_epi32(y_hi_scaled, b_dup_hi);

      let r_u16 = _mm512_permutexvar_epi64(pack_fixup, _mm512_packus_epi32(r_lo_i32, r_hi_i32));
      let g_u16 = _mm512_permutexvar_epi64(pack_fixup, _mm512_packus_epi32(g_lo_i32, g_hi_i32));
      let b_u16 = _mm512_permutexvar_epi64(pack_fixup, _mm512_packus_epi32(b_lo_i32, b_hi_i32));

      write_rgb_u16_32(r_u16, g_u16, b_u16, rgb_out.as_mut_ptr().add(x * 3));

      x += 32;
    }

    if x < width {
      scalar::p16_to_rgb_u16_row(
        &y[x..width],
        &uv_half[x..width],
        &mut rgb_out[x * 3..width * 3],
        width - x,
        matrix,
        full_range,
      );
    }
  }
}

// ===== Pn 4:4:4 (semi-planar high-bit-packed) → RGB =======================
//
// Native AVX-512 4:4:4 Pn kernels — combine `yuv_444p_n_to_rgb_row`'s
// 1:1 chroma compute (no duplication, two `chroma_i16x32` per channel)
// with `p_n_to_rgb_row`'s `deinterleave_uv_u16_avx512` pattern. 64 Y
// pixels per iter for the i32 paths; 32 for the i64 u16-output path
// (matches `yuv_444p16_to_rgb_u16_row` cadence). The 16-bit u16 output
// uses native `_mm512_srai_epi64` via `chroma_i64x8_avx512` —
// no bias trick.

/// AVX-512 Pn 4:4:4 high-bit-packed (BITS ∈ {10, 12}) → packed **u8** RGB.
/// 64 pixels per iter via 512-bit vectors; 128 UV elements (= 64 pairs)
/// deinterleaved per iter via two `deinterleave_uv_u16_avx512` calls.
///
/// # Safety
///
/// 1. AVX-512F + AVX-512BW must be available on the current CPU.
/// 2. `y.len() >= width`, `uv_full.len() >= 2 * width`,
///    `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "avx512bw,avx512f")]
pub(crate) unsafe fn p_n_444_to_rgb_row<const BITS: u32>(
  y: &[u16],
  uv_full: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  const { assert!(BITS == 10 || BITS == 12) };
  debug_assert!(y.len() >= width);
  debug_assert!(uv_full.len() >= 2 * width);
  debug_assert!(rgb_out.len() >= width * 3);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<BITS, 8>(full_range);
  let bias = scalar::chroma_bias::<BITS>();
  const RND: i32 = 1 << 14;

  unsafe {
    let rnd_v = _mm512_set1_epi32(RND);
    let y_off_v = _mm512_set1_epi16(y_off as i16);
    let y_scale_v = _mm512_set1_epi32(y_scale);
    let c_scale_v = _mm512_set1_epi32(c_scale);
    let bias_v = _mm512_set1_epi16(bias as i16);
    let shr_count = _mm_cvtsi32_si128((16 - BITS) as i32);
    let cru = _mm512_set1_epi32(coeffs.r_u());
    let crv = _mm512_set1_epi32(coeffs.r_v());
    let cgu = _mm512_set1_epi32(coeffs.g_u());
    let cgv = _mm512_set1_epi32(coeffs.g_v());
    let cbu = _mm512_set1_epi32(coeffs.b_u());
    let cbv = _mm512_set1_epi32(coeffs.b_v());

    let pack_fixup = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);

    let mut x = 0usize;
    while x + 64 <= width {
      let y_low_i16 = _mm512_srl_epi16(_mm512_loadu_si512(y.as_ptr().add(x).cast()), shr_count);
      let y_high_i16 =
        _mm512_srl_epi16(_mm512_loadu_si512(y.as_ptr().add(x + 32).cast()), shr_count);

      // 128 UV elements (= 64 pairs) per iter — two deinterleave calls.
      let (u_lo_vec, v_lo_vec) = deinterleave_uv_u16_avx512(uv_full.as_ptr().add(x * 2));
      let (u_hi_vec, v_hi_vec) = deinterleave_uv_u16_avx512(uv_full.as_ptr().add(x * 2 + 64));
      let u_lo_vec = _mm512_srl_epi16(u_lo_vec, shr_count);
      let v_lo_vec = _mm512_srl_epi16(v_lo_vec, shr_count);
      let u_hi_vec = _mm512_srl_epi16(u_hi_vec, shr_count);
      let v_hi_vec = _mm512_srl_epi16(v_hi_vec, shr_count);

      let u_lo_i16 = _mm512_sub_epi16(u_lo_vec, bias_v);
      let u_hi_i16 = _mm512_sub_epi16(u_hi_vec, bias_v);
      let v_lo_i16 = _mm512_sub_epi16(v_lo_vec, bias_v);
      let v_hi_i16 = _mm512_sub_epi16(v_hi_vec, bias_v);

      let u_lo_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_lo_i16));
      let u_lo_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(u_lo_i16));
      let u_hi_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_hi_i16));
      let u_hi_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(u_hi_i16));
      let v_lo_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_lo_i16));
      let v_lo_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(v_lo_i16));
      let v_hi_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_hi_i16));
      let v_hi_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(v_hi_i16));

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

      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v, pack_fixup);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v, pack_fixup);

      let r_lo = _mm512_adds_epi16(y_scaled_lo, r_chroma_lo);
      let r_hi = _mm512_adds_epi16(y_scaled_hi, r_chroma_hi);
      let g_lo = _mm512_adds_epi16(y_scaled_lo, g_chroma_lo);
      let g_hi = _mm512_adds_epi16(y_scaled_hi, g_chroma_hi);
      let b_lo = _mm512_adds_epi16(y_scaled_lo, b_chroma_lo);
      let b_hi = _mm512_adds_epi16(y_scaled_hi, b_chroma_hi);

      let b_u8 = narrow_u8x64(b_lo, b_hi, pack_fixup);
      let g_u8 = narrow_u8x64(g_lo, g_hi, pack_fixup);
      let r_u8 = narrow_u8x64(r_lo, r_hi, pack_fixup);

      write_rgb_64(r_u8, g_u8, b_u8, rgb_out.as_mut_ptr().add(x * 3));

      x += 64;
    }

    if x < width {
      scalar::p_n_444_to_rgb_row::<BITS>(
        &y[x..width],
        &uv_full[x * 2..width * 2],
        &mut rgb_out[x * 3..width * 3],
        width - x,
        matrix,
        full_range,
      );
    }
  }
}

/// AVX-512 Pn 4:4:4 high-bit-packed (BITS ∈ {10, 12}) → packed
/// **native-depth `u16`** RGB. 64 pixels per iter.
///
/// # Safety
///
/// Same as [`p_n_444_to_rgb_row`] but `rgb_out: &mut [u16]`.
#[inline]
#[target_feature(enable = "avx512bw,avx512f")]
pub(crate) unsafe fn p_n_444_to_rgb_u16_row<const BITS: u32>(
  y: &[u16],
  uv_full: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  const { assert!(BITS == 10 || BITS == 12) };
  debug_assert!(y.len() >= width);
  debug_assert!(uv_full.len() >= 2 * width);
  debug_assert!(rgb_out.len() >= width * 3);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<BITS, BITS>(full_range);
  let bias = scalar::chroma_bias::<BITS>();
  const RND: i32 = 1 << 14;
  let out_max: i16 = ((1i32 << BITS) - 1) as i16;

  unsafe {
    let rnd_v = _mm512_set1_epi32(RND);
    let y_off_v = _mm512_set1_epi16(y_off as i16);
    let y_scale_v = _mm512_set1_epi32(y_scale);
    let c_scale_v = _mm512_set1_epi32(c_scale);
    let bias_v = _mm512_set1_epi16(bias as i16);
    let max_v = _mm512_set1_epi16(out_max);
    let zero_v = _mm512_set1_epi16(0);
    let shr_count = _mm_cvtsi32_si128((16 - BITS) as i32);
    let cru = _mm512_set1_epi32(coeffs.r_u());
    let crv = _mm512_set1_epi32(coeffs.r_v());
    let cgu = _mm512_set1_epi32(coeffs.g_u());
    let cgv = _mm512_set1_epi32(coeffs.g_v());
    let cbu = _mm512_set1_epi32(coeffs.b_u());
    let cbv = _mm512_set1_epi32(coeffs.b_v());

    let pack_fixup = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);

    let mut x = 0usize;
    while x + 64 <= width {
      let y_low_i16 = _mm512_srl_epi16(_mm512_loadu_si512(y.as_ptr().add(x).cast()), shr_count);
      let y_high_i16 =
        _mm512_srl_epi16(_mm512_loadu_si512(y.as_ptr().add(x + 32).cast()), shr_count);

      let (u_lo_vec, v_lo_vec) = deinterleave_uv_u16_avx512(uv_full.as_ptr().add(x * 2));
      let (u_hi_vec, v_hi_vec) = deinterleave_uv_u16_avx512(uv_full.as_ptr().add(x * 2 + 64));
      let u_lo_vec = _mm512_srl_epi16(u_lo_vec, shr_count);
      let v_lo_vec = _mm512_srl_epi16(v_lo_vec, shr_count);
      let u_hi_vec = _mm512_srl_epi16(u_hi_vec, shr_count);
      let v_hi_vec = _mm512_srl_epi16(v_hi_vec, shr_count);

      let u_lo_i16 = _mm512_sub_epi16(u_lo_vec, bias_v);
      let u_hi_i16 = _mm512_sub_epi16(u_hi_vec, bias_v);
      let v_lo_i16 = _mm512_sub_epi16(v_lo_vec, bias_v);
      let v_hi_i16 = _mm512_sub_epi16(v_hi_vec, bias_v);

      let u_lo_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_lo_i16));
      let u_lo_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(u_lo_i16));
      let u_hi_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_hi_i16));
      let u_hi_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(u_hi_i16));
      let v_lo_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_lo_i16));
      let v_lo_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(v_lo_i16));
      let v_hi_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_hi_i16));
      let v_hi_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(v_hi_i16));

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

      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v, pack_fixup);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v, pack_fixup);

      let r_lo = clamp_u16_max_x32(_mm512_adds_epi16(y_scaled_lo, r_chroma_lo), zero_v, max_v);
      let r_hi = clamp_u16_max_x32(_mm512_adds_epi16(y_scaled_hi, r_chroma_hi), zero_v, max_v);
      let g_lo = clamp_u16_max_x32(_mm512_adds_epi16(y_scaled_lo, g_chroma_lo), zero_v, max_v);
      let g_hi = clamp_u16_max_x32(_mm512_adds_epi16(y_scaled_hi, g_chroma_hi), zero_v, max_v);
      let b_lo = clamp_u16_max_x32(_mm512_adds_epi16(y_scaled_lo, b_chroma_lo), zero_v, max_v);
      let b_hi = clamp_u16_max_x32(_mm512_adds_epi16(y_scaled_hi, b_chroma_hi), zero_v, max_v);

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
      scalar::p_n_444_to_rgb_u16_row::<BITS>(
        &y[x..width],
        &uv_full[x * 2..width * 2],
        &mut rgb_out[x * 3..width * 3],
        width - x,
        matrix,
        full_range,
      );
    }
  }
}

/// AVX-512 P416 (semi-planar 4:4:4, 16-bit) → packed **u8** RGB.
/// 64 pixels per iter; Y stays on i32.
///
/// # Safety
///
/// 1. AVX-512F + AVX-512BW must be available.
/// 2. `y.len() >= width`, `uv_full.len() >= 2 * width`,
///    `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "avx512bw,avx512f")]
pub(crate) unsafe fn p_n_444_16_to_rgb_row(
  y: &[u16],
  uv_full: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(y.len() >= width);
  debug_assert!(uv_full.len() >= 2 * width);
  debug_assert!(rgb_out.len() >= width * 3);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<16, 8>(full_range);
  const RND: i32 = 1 << 14;

  unsafe {
    let rnd_v = _mm512_set1_epi32(RND);
    let y_off_v = _mm512_set1_epi32(y_off);
    let y_scale_v = _mm512_set1_epi32(y_scale);
    let c_scale_v = _mm512_set1_epi32(c_scale);
    let bias16_v = _mm512_set1_epi16(-32768i16);
    let cru = _mm512_set1_epi32(coeffs.r_u());
    let crv = _mm512_set1_epi32(coeffs.r_v());
    let cgu = _mm512_set1_epi32(coeffs.g_u());
    let cgv = _mm512_set1_epi32(coeffs.g_v());
    let cbu = _mm512_set1_epi32(coeffs.b_u());
    let cbv = _mm512_set1_epi32(coeffs.b_v());

    let pack_fixup = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);

    let mut x = 0usize;
    while x + 64 <= width {
      let y_low = _mm512_loadu_si512(y.as_ptr().add(x).cast());
      let y_high = _mm512_loadu_si512(y.as_ptr().add(x + 32).cast());

      let (u_lo_vec, v_lo_vec) = deinterleave_uv_u16_avx512(uv_full.as_ptr().add(x * 2));
      let (u_hi_vec, v_hi_vec) = deinterleave_uv_u16_avx512(uv_full.as_ptr().add(x * 2 + 64));

      let u_lo_i16 = _mm512_sub_epi16(u_lo_vec, bias16_v);
      let u_hi_i16 = _mm512_sub_epi16(u_hi_vec, bias16_v);
      let v_lo_i16 = _mm512_sub_epi16(v_lo_vec, bias16_v);
      let v_hi_i16 = _mm512_sub_epi16(v_hi_vec, bias16_v);

      let u_lo_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_lo_i16));
      let u_lo_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(u_lo_i16));
      let u_hi_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_hi_i16));
      let u_hi_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(u_hi_i16));
      let v_lo_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_lo_i16));
      let v_lo_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(v_lo_i16));
      let v_hi_a = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_hi_i16));
      let v_hi_b = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(v_hi_i16));

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

      let y_scaled_lo = scale_y_u16_avx512(y_low, y_off_v, y_scale_v, rnd_v, pack_fixup);
      let y_scaled_hi = scale_y_u16_avx512(y_high, y_off_v, y_scale_v, rnd_v, pack_fixup);

      let r_lo = _mm512_adds_epi16(y_scaled_lo, r_chroma_lo);
      let r_hi = _mm512_adds_epi16(y_scaled_hi, r_chroma_hi);
      let g_lo = _mm512_adds_epi16(y_scaled_lo, g_chroma_lo);
      let g_hi = _mm512_adds_epi16(y_scaled_hi, g_chroma_hi);
      let b_lo = _mm512_adds_epi16(y_scaled_lo, b_chroma_lo);
      let b_hi = _mm512_adds_epi16(y_scaled_hi, b_chroma_hi);

      let r_u8 = narrow_u8x64(r_lo, r_hi, pack_fixup);
      let g_u8 = narrow_u8x64(g_lo, g_hi, pack_fixup);
      let b_u8 = narrow_u8x64(b_lo, b_hi, pack_fixup);

      write_rgb_64(r_u8, g_u8, b_u8, rgb_out.as_mut_ptr().add(x * 3));
      x += 64;
    }

    if x < width {
      scalar::p_n_444_16_to_rgb_row(
        &y[x..width],
        &uv_full[x * 2..width * 2],
        &mut rgb_out[x * 3..width * 3],
        width - x,
        matrix,
        full_range,
      );
    }
  }
}

/// AVX-512 P416 → packed **native-depth `u16`** RGB. 32 pixels per
/// iter (i64 narrows). Native `_mm512_srai_epi64` via
/// `chroma_i64x8_avx512` + `scale_y_i32x16_i64` — no bias trick.
///
/// # Safety
///
/// Same as [`p_n_444_16_to_rgb_row`] but `rgb_out: &mut [u16]`.
#[inline]
#[target_feature(enable = "avx512bw,avx512f")]
pub(crate) unsafe fn p_n_444_16_to_rgb_u16_row(
  y: &[u16],
  uv_full: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(y.len() >= width);
  debug_assert!(uv_full.len() >= 2 * width);
  debug_assert!(rgb_out.len() >= width * 3);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<16, 16>(full_range);
  const RND_I64: i64 = 1 << 14;
  const RND_I32: i32 = 1 << 14;

  unsafe {
    let rnd_i64_v = _mm512_set1_epi64(RND_I64);
    let rnd_i32_v = _mm512_set1_epi32(RND_I32);
    let y_off_v = _mm512_set1_epi32(y_off);
    let y_scale_v = _mm512_set1_epi32(y_scale);
    let c_scale_v = _mm512_set1_epi32(c_scale);
    let bias16_v = _mm512_set1_epi16(-32768i16);
    let cru = _mm512_set1_epi32(coeffs.r_u());
    let crv = _mm512_set1_epi32(coeffs.r_v());
    let cgu = _mm512_set1_epi32(coeffs.g_u());
    let cgv = _mm512_set1_epi32(coeffs.g_v());
    let cbu = _mm512_set1_epi32(coeffs.b_u());
    let cbv = _mm512_set1_epi32(coeffs.b_v());

    let interleave_idx = _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
    let pack_fixup = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);

    let mut x = 0usize;
    while x + 32 <= width {
      // 32 pixels per iter — one deinterleave (64 UV elements = 32 pairs).
      let y_vec = _mm512_loadu_si512(y.as_ptr().add(x).cast());
      let (u_vec, v_vec) = deinterleave_uv_u16_avx512(uv_full.as_ptr().add(x * 2));

      let u_i16 = _mm512_sub_epi16(u_vec, bias16_v);
      let v_i16 = _mm512_sub_epi16(v_vec, bias16_v);

      let u_lo_i32 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u_i16));
      let u_hi_i32 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(u_i16));
      let v_lo_i32 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v_i16));
      let v_hi_i32 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(v_i16));

      let u_d_lo = _mm512_srai_epi32::<15>(_mm512_add_epi32(
        _mm512_mullo_epi32(u_lo_i32, c_scale_v),
        rnd_i32_v,
      ));
      let u_d_hi = _mm512_srai_epi32::<15>(_mm512_add_epi32(
        _mm512_mullo_epi32(u_hi_i32, c_scale_v),
        rnd_i32_v,
      ));
      let v_d_lo = _mm512_srai_epi32::<15>(_mm512_add_epi32(
        _mm512_mullo_epi32(v_lo_i32, c_scale_v),
        rnd_i32_v,
      ));
      let v_d_hi = _mm512_srai_epi32::<15>(_mm512_add_epi32(
        _mm512_mullo_epi32(v_hi_i32, c_scale_v),
        rnd_i32_v,
      ));

      // i64 chroma via even/odd splits — uses native
      // `_mm512_srai_epi64` inside `chroma_i64x8_avx512`. No bias trick.
      let u_d_lo_odd = _mm512_shuffle_epi32::<0xF5>(u_d_lo);
      let u_d_hi_odd = _mm512_shuffle_epi32::<0xF5>(u_d_hi);
      let v_d_lo_odd = _mm512_shuffle_epi32::<0xF5>(v_d_lo);
      let v_d_hi_odd = _mm512_shuffle_epi32::<0xF5>(v_d_hi);

      let r_ch_lo_e = chroma_i64x8_avx512(cru, crv, u_d_lo, v_d_lo, rnd_i64_v);
      let r_ch_lo_o = chroma_i64x8_avx512(cru, crv, u_d_lo_odd, v_d_lo_odd, rnd_i64_v);
      let r_ch_hi_e = chroma_i64x8_avx512(cru, crv, u_d_hi, v_d_hi, rnd_i64_v);
      let r_ch_hi_o = chroma_i64x8_avx512(cru, crv, u_d_hi_odd, v_d_hi_odd, rnd_i64_v);
      let g_ch_lo_e = chroma_i64x8_avx512(cgu, cgv, u_d_lo, v_d_lo, rnd_i64_v);
      let g_ch_lo_o = chroma_i64x8_avx512(cgu, cgv, u_d_lo_odd, v_d_lo_odd, rnd_i64_v);
      let g_ch_hi_e = chroma_i64x8_avx512(cgu, cgv, u_d_hi, v_d_hi, rnd_i64_v);
      let g_ch_hi_o = chroma_i64x8_avx512(cgu, cgv, u_d_hi_odd, v_d_hi_odd, rnd_i64_v);
      let b_ch_lo_e = chroma_i64x8_avx512(cbu, cbv, u_d_lo, v_d_lo, rnd_i64_v);
      let b_ch_lo_o = chroma_i64x8_avx512(cbu, cbv, u_d_lo_odd, v_d_lo_odd, rnd_i64_v);
      let b_ch_hi_e = chroma_i64x8_avx512(cbu, cbv, u_d_hi, v_d_hi, rnd_i64_v);
      let b_ch_hi_o = chroma_i64x8_avx512(cbu, cbv, u_d_hi_odd, v_d_hi_odd, rnd_i64_v);

      let r_ch_lo = reassemble_i32x16(r_ch_lo_e, r_ch_lo_o, interleave_idx);
      let r_ch_hi = reassemble_i32x16(r_ch_hi_e, r_ch_hi_o, interleave_idx);
      let g_ch_lo = reassemble_i32x16(g_ch_lo_e, g_ch_lo_o, interleave_idx);
      let g_ch_hi = reassemble_i32x16(g_ch_hi_e, g_ch_hi_o, interleave_idx);
      let b_ch_lo = reassemble_i32x16(b_ch_lo_e, b_ch_lo_o, interleave_idx);
      let b_ch_hi = reassemble_i32x16(b_ch_hi_e, b_ch_hi_o, interleave_idx);

      // Y: widen 32 u16 → two i32x16 halves, subtract y_off, scale i64.
      let y_lo_u16 = _mm512_castsi512_si256(y_vec);
      let y_hi_u16 = _mm512_extracti64x4_epi64::<1>(y_vec);
      let y_lo_i32 = _mm512_sub_epi32(_mm512_cvtepu16_epi32(y_lo_u16), y_off_v);
      let y_hi_i32 = _mm512_sub_epi32(_mm512_cvtepu16_epi32(y_hi_u16), y_off_v);

      let y_lo_scaled = scale_y_i32x16_i64(y_lo_i32, y_scale_v, rnd_i64_v, interleave_idx);
      let y_hi_scaled = scale_y_i32x16_i64(y_hi_i32, y_scale_v, rnd_i64_v, interleave_idx);

      let r_lo_i32 = _mm512_add_epi32(y_lo_scaled, r_ch_lo);
      let r_hi_i32 = _mm512_add_epi32(y_hi_scaled, r_ch_hi);
      let g_lo_i32 = _mm512_add_epi32(y_lo_scaled, g_ch_lo);
      let g_hi_i32 = _mm512_add_epi32(y_hi_scaled, g_ch_hi);
      let b_lo_i32 = _mm512_add_epi32(y_lo_scaled, b_ch_lo);
      let b_hi_i32 = _mm512_add_epi32(y_hi_scaled, b_ch_hi);

      let r_u16 = _mm512_permutexvar_epi64(pack_fixup, _mm512_packus_epi32(r_lo_i32, r_hi_i32));
      let g_u16 = _mm512_permutexvar_epi64(pack_fixup, _mm512_packus_epi32(g_lo_i32, g_hi_i32));
      let b_u16 = _mm512_permutexvar_epi64(pack_fixup, _mm512_packus_epi32(b_lo_i32, b_hi_i32));

      write_rgb_u16_32(r_u16, g_u16, b_u16, rgb_out.as_mut_ptr().add(x * 3));
      x += 32;
    }

    if x < width {
      scalar::p_n_444_16_to_rgb_u16_row(
        &y[x..width],
        &uv_full[x * 2..width * 2],
        &mut rgb_out[x * 3..width * 3],
        width - x,
        matrix,
        full_range,
      );
    }
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

  // ---- nv24_to_rgb_row / nv42_to_rgb_row equivalence ------------------

  fn check_nv24_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
    let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
    let uv: std::vec::Vec<u8> = (0..width)
      .flat_map(|i| [((i * 53 + 23) & 0xFF) as u8, ((i * 71 + 91) & 0xFF) as u8])
      .collect();
    let mut rgb_scalar = std::vec![0u8; width * 3];
    let mut rgb_avx512 = std::vec![0u8; width * 3];

    scalar::nv24_to_rgb_row(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
    unsafe {
      nv24_to_rgb_row(&y, &uv, &mut rgb_avx512, width, matrix, full_range);
    }
    assert_eq!(
      rgb_scalar, rgb_avx512,
      "AVX-512 NV24 ≠ scalar (width={width}, matrix={matrix:?}, full_range={full_range})"
    );
  }

  fn check_nv42_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
    let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
    let vu: std::vec::Vec<u8> = (0..width)
      .flat_map(|i| [((i * 53 + 23) & 0xFF) as u8, ((i * 71 + 91) & 0xFF) as u8])
      .collect();
    let mut rgb_scalar = std::vec![0u8; width * 3];
    let mut rgb_avx512 = std::vec![0u8; width * 3];

    scalar::nv42_to_rgb_row(&y, &vu, &mut rgb_scalar, width, matrix, full_range);
    unsafe {
      nv42_to_rgb_row(&y, &vu, &mut rgb_avx512, width, matrix, full_range);
    }
    assert_eq!(
      rgb_scalar, rgb_avx512,
      "AVX-512 NV42 ≠ scalar (width={width}, matrix={matrix:?}, full_range={full_range})"
    );
  }

  #[test]
  fn avx512_nv24_matches_scalar_all_matrices_64() {
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
        check_nv24_equivalence(64, m, full);
      }
    }
  }

  #[test]
  fn avx512_nv24_matches_scalar_widths() {
    if !std::arch::is_x86_feature_detected!("avx512bw") {
      return;
    }
    // 64 / 128 → main loop; 65 / 129 → main + 1-px tail; 63 → pure
    // scalar tail (< block size); 127 → main + 63-px tail; 1920 →
    // wide real-world baseline.
    for w in [63usize, 64, 65, 127, 128, 129, 1920, 1921] {
      check_nv24_equivalence(w, ColorMatrix::Bt709, false);
    }
  }

  #[test]
  fn avx512_nv42_matches_scalar_all_matrices_64() {
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
        check_nv42_equivalence(64, m, full);
      }
    }
  }

  #[test]
  fn avx512_nv42_matches_scalar_widths() {
    if !std::arch::is_x86_feature_detected!("avx512bw") {
      return;
    }
    for w in [63usize, 64, 65, 127, 128, 129, 1920, 1921] {
      check_nv42_equivalence(w, ColorMatrix::Bt709, false);
    }
  }

  // ---- yuv_444_to_rgb_row equivalence ---------------------------------

  fn check_yuv_444_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
    let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
    let u: std::vec::Vec<u8> = (0..width).map(|i| ((i * 53 + 23) & 0xFF) as u8).collect();
    let v: std::vec::Vec<u8> = (0..width).map(|i| ((i * 71 + 91) & 0xFF) as u8).collect();
    let mut rgb_scalar = std::vec![0u8; width * 3];
    let mut rgb_avx512 = std::vec![0u8; width * 3];

    scalar::yuv_444_to_rgb_row(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
    unsafe {
      yuv_444_to_rgb_row(&y, &u, &v, &mut rgb_avx512, width, matrix, full_range);
    }
    assert_eq!(
      rgb_scalar, rgb_avx512,
      "AVX-512 yuv_444 ≠ scalar (width={width}, matrix={matrix:?}, full_range={full_range})"
    );
  }

  #[test]
  fn avx512_yuv_444_matches_scalar_all_matrices_64() {
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
        check_yuv_444_equivalence(64, m, full);
      }
    }
  }

  #[test]
  fn avx512_yuv_444_matches_scalar_widths() {
    if !std::arch::is_x86_feature_detected!("avx512bw") {
      return;
    }
    // Widths straddling the 64-pixel AVX-512 block, AVX2's 32-pixel
    // block, and SSE4.1's 16-pixel tail fallback. Odd widths validate
    // the 4:4:4 no-parity contract.
    for w in [63usize, 64, 65, 127, 128, 129, 1920, 1921] {
      check_yuv_444_equivalence(w, ColorMatrix::Bt709, false);
    }
  }

  // ---- yuv_444p_n<BITS> + yuv_444p16 equivalence ----------------------

  fn check_yuv_444p_n_equivalence<const BITS: u32>(
    width: usize,
    matrix: ColorMatrix,
    full_range: bool,
  ) {
    let max_val = (1u16 << BITS) - 1;
    let y: std::vec::Vec<u16> = (0..width)
      .map(|i| ((i * 37 + 11) as u16) & max_val)
      .collect();
    let u: std::vec::Vec<u16> = (0..width)
      .map(|i| ((i * 53 + 23) as u16) & max_val)
      .collect();
    let v: std::vec::Vec<u16> = (0..width)
      .map(|i| ((i * 71 + 91) as u16) & max_val)
      .collect();
    let mut rgb_scalar = std::vec![0u8; width * 3];
    let mut rgb_avx512 = std::vec![0u8; width * 3];
    let mut u16_scalar = std::vec![0u16; width * 3];
    let mut u16_avx512 = std::vec![0u16; width * 3];

    scalar::yuv_444p_n_to_rgb_row::<BITS>(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
    scalar::yuv_444p_n_to_rgb_u16_row::<BITS>(
      &y,
      &u,
      &v,
      &mut u16_scalar,
      width,
      matrix,
      full_range,
    );
    unsafe {
      yuv_444p_n_to_rgb_row::<BITS>(&y, &u, &v, &mut rgb_avx512, width, matrix, full_range);
      yuv_444p_n_to_rgb_u16_row::<BITS>(&y, &u, &v, &mut u16_avx512, width, matrix, full_range);
    }
    assert_eq!(
      rgb_scalar, rgb_avx512,
      "AVX-512 yuv_444p_n<{BITS}> u8 ≠ scalar"
    );
    assert_eq!(
      u16_scalar, u16_avx512,
      "AVX-512 yuv_444p_n<{BITS}> u16 ≠ scalar"
    );
  }

  #[test]
  fn avx512_yuv_444p9_matches_scalar_all_matrices() {
    if !std::arch::is_x86_feature_detected!("avx512bw") {
      return;
    }
    for m in [ColorMatrix::Bt709, ColorMatrix::Bt2020Ncl] {
      for full in [true, false] {
        check_yuv_444p_n_equivalence::<9>(64, m, full);
      }
    }
  }

  #[test]
  fn avx512_yuv_444p10_matches_scalar_all_matrices() {
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
        check_yuv_444p_n_equivalence::<10>(64, m, full);
      }
    }
  }

  #[test]
  fn avx512_yuv_444p12_matches_scalar_all_matrices() {
    if !std::arch::is_x86_feature_detected!("avx512bw") {
      return;
    }
    for m in [ColorMatrix::Bt709, ColorMatrix::Bt2020Ncl] {
      for full in [true, false] {
        check_yuv_444p_n_equivalence::<12>(64, m, full);
      }
    }
  }

  #[test]
  fn avx512_yuv_444p14_matches_scalar_all_matrices() {
    if !std::arch::is_x86_feature_detected!("avx512bw") {
      return;
    }
    for m in [ColorMatrix::Bt709, ColorMatrix::Bt2020Ncl] {
      for full in [true, false] {
        check_yuv_444p_n_equivalence::<14>(64, m, full);
      }
    }
  }

  #[test]
  fn avx512_yuv_444p_n_matches_scalar_widths() {
    if !std::arch::is_x86_feature_detected!("avx512bw") {
      return;
    }
    for w in [1usize, 3, 31, 32, 33, 63, 64, 65, 127, 128, 129, 1920, 1921] {
      check_yuv_444p_n_equivalence::<10>(w, ColorMatrix::Bt709, false);
    }
  }

  fn check_yuv_444p16_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
    let y: std::vec::Vec<u16> = (0..width).map(|i| (i * 2027 + 11) as u16).collect();
    let u: std::vec::Vec<u16> = (0..width).map(|i| (i * 2671 + 23) as u16).collect();
    let v: std::vec::Vec<u16> = (0..width).map(|i| (i * 3329 + 91) as u16).collect();
    let mut rgb_scalar = std::vec![0u8; width * 3];
    let mut rgb_avx512 = std::vec![0u8; width * 3];
    let mut u16_scalar = std::vec![0u16; width * 3];
    let mut u16_avx512 = std::vec![0u16; width * 3];

    scalar::yuv_444p16_to_rgb_row(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
    scalar::yuv_444p16_to_rgb_u16_row(&y, &u, &v, &mut u16_scalar, width, matrix, full_range);
    unsafe {
      yuv_444p16_to_rgb_row(&y, &u, &v, &mut rgb_avx512, width, matrix, full_range);
      yuv_444p16_to_rgb_u16_row(&y, &u, &v, &mut u16_avx512, width, matrix, full_range);
    }
    assert_eq!(rgb_scalar, rgb_avx512, "AVX-512 yuv_444p16 u8 ≠ scalar");
    assert_eq!(u16_scalar, u16_avx512, "AVX-512 yuv_444p16 u16 ≠ scalar");
  }

  #[test]
  fn avx512_yuv_444p16_matches_scalar_all_matrices() {
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
        check_yuv_444p16_equivalence(64, m, full);
      }
    }
  }

  #[test]
  fn avx512_yuv_444p16_matches_scalar_widths() {
    if !std::arch::is_x86_feature_detected!("avx512bw") {
      return;
    }
    // The u16 kernel is 32-pixel per iter; the u8 kernel is 64.
    for w in [
      1usize, 15, 31, 32, 33, 63, 64, 65, 127, 128, 129, 1920, 1921,
    ] {
      check_yuv_444p16_equivalence(w, ColorMatrix::Bt709, false);
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
      yuv_420p_n_to_rgb_row::<10>(&y, &u, &v, &mut rgb_simd, width, matrix, full_range);
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
      yuv_420p_n_to_rgb_u16_row::<10>(&y, &u, &v, &mut rgb_simd, width, matrix, full_range);
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

  // ---- yuv420p_n<BITS> AVX-512 scalar-equivalence (BITS=9 coverage) ---

  fn p_n_plane_avx512<const BITS: u32>(n: usize, seed: usize) -> std::vec::Vec<u16> {
    let mask = ((1u32 << BITS) - 1) as u16;
    (0..n)
      .map(|i| ((i.wrapping_mul(seed).wrapping_add(seed * 3)) as u16) & mask)
      .collect()
  }

  fn check_p_n_u8_avx512_equivalence<const BITS: u32>(
    width: usize,
    matrix: ColorMatrix,
    full_range: bool,
  ) {
    if !std::arch::is_x86_feature_detected!("avx512bw") {
      return;
    }
    let y = p_n_plane_avx512::<BITS>(width, 37);
    let u = p_n_plane_avx512::<BITS>(width / 2, 53);
    let v = p_n_plane_avx512::<BITS>(width / 2, 71);
    let mut rgb_scalar = std::vec![0u8; width * 3];
    let mut rgb_simd = std::vec![0u8; width * 3];
    scalar::yuv_420p_n_to_rgb_row::<BITS>(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
    unsafe {
      yuv_420p_n_to_rgb_row::<BITS>(&y, &u, &v, &mut rgb_simd, width, matrix, full_range);
    }
    assert_eq!(
      rgb_scalar, rgb_simd,
      "AVX-512 yuv_420p_n<{BITS}>→u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
    );
  }

  fn check_p_n_u16_avx512_equivalence<const BITS: u32>(
    width: usize,
    matrix: ColorMatrix,
    full_range: bool,
  ) {
    if !std::arch::is_x86_feature_detected!("avx512bw") {
      return;
    }
    let y = p_n_plane_avx512::<BITS>(width, 37);
    let u = p_n_plane_avx512::<BITS>(width / 2, 53);
    let v = p_n_plane_avx512::<BITS>(width / 2, 71);
    let mut rgb_scalar = std::vec![0u16; width * 3];
    let mut rgb_simd = std::vec![0u16; width * 3];
    scalar::yuv_420p_n_to_rgb_u16_row::<BITS>(
      &y,
      &u,
      &v,
      &mut rgb_scalar,
      width,
      matrix,
      full_range,
    );
    unsafe {
      yuv_420p_n_to_rgb_u16_row::<BITS>(&y, &u, &v, &mut rgb_simd, width, matrix, full_range);
    }
    assert_eq!(
      rgb_scalar, rgb_simd,
      "AVX-512 yuv_420p_n<{BITS}>→u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
    );
  }

  #[test]
  fn avx512_yuv420p9_matches_scalar_all_matrices_and_ranges() {
    for m in [
      ColorMatrix::Bt601,
      ColorMatrix::Bt709,
      ColorMatrix::Bt2020Ncl,
      ColorMatrix::Smpte240m,
      ColorMatrix::Fcc,
      ColorMatrix::YCgCo,
    ] {
      for full in [true, false] {
        check_p_n_u8_avx512_equivalence::<9>(64, m, full);
        check_p_n_u16_avx512_equivalence::<9>(64, m, full);
      }
    }
  }

  #[test]
  fn avx512_yuv420p9_matches_scalar_tail_and_large_widths() {
    // AVX-512 main loop is 64 px; widths chosen to stress tail handling
    // both below and above the SIMD lane width.
    for w in [18usize, 30, 34, 62, 66, 126, 130, 1922] {
      check_p_n_u8_avx512_equivalence::<9>(w, ColorMatrix::Bt601, false);
      check_p_n_u16_avx512_equivalence::<9>(w, ColorMatrix::Bt709, true);
    }
    check_p_n_u8_avx512_equivalence::<9>(1920, ColorMatrix::Bt709, false);
    check_p_n_u16_avx512_equivalence::<9>(1920, ColorMatrix::Bt2020Ncl, false);
  }

  // ---- P010 AVX-512 scalar-equivalence --------------------------------

  fn p010_plane(n: usize, seed: usize) -> std::vec::Vec<u16> {
    (0..n)
      .map(|i| (((i * seed + seed * 3) & 0x3FF) as u16) << 6)
      .collect()
  }

  fn p010_uv_interleave(u: &[u16], v: &[u16]) -> std::vec::Vec<u16> {
    let pairs = u.len();
    debug_assert_eq!(u.len(), v.len());
    let mut out = std::vec::Vec::with_capacity(pairs * 2);
    for i in 0..pairs {
      out.push(u[i]);
      out.push(v[i]);
    }
    out
  }

  fn check_p010_u8_avx512_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
    if !std::arch::is_x86_feature_detected!("avx512bw") {
      return;
    }
    let y = p010_plane(width, 37);
    let u = p010_plane(width / 2, 53);
    let v = p010_plane(width / 2, 71);
    let uv = p010_uv_interleave(&u, &v);
    let mut rgb_scalar = std::vec![0u8; width * 3];
    let mut rgb_simd = std::vec![0u8; width * 3];
    scalar::p_n_to_rgb_row::<10>(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
    unsafe {
      p_n_to_rgb_row::<10>(&y, &uv, &mut rgb_simd, width, matrix, full_range);
    }
    assert_eq!(rgb_scalar, rgb_simd, "AVX-512 P010→u8 diverges");
  }

  fn check_p010_u16_avx512_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
    if !std::arch::is_x86_feature_detected!("avx512bw") {
      return;
    }
    let y = p010_plane(width, 37);
    let u = p010_plane(width / 2, 53);
    let v = p010_plane(width / 2, 71);
    let uv = p010_uv_interleave(&u, &v);
    let mut rgb_scalar = std::vec![0u16; width * 3];
    let mut rgb_simd = std::vec![0u16; width * 3];
    scalar::p_n_to_rgb_u16_row::<10>(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
    unsafe {
      p_n_to_rgb_u16_row::<10>(&y, &uv, &mut rgb_simd, width, matrix, full_range);
    }
    assert_eq!(rgb_scalar, rgb_simd, "AVX-512 P010→u16 diverges");
  }

  #[test]
  fn avx512_p010_u8_matches_scalar_all_matrices() {
    for m in [
      ColorMatrix::Bt601,
      ColorMatrix::Bt709,
      ColorMatrix::Bt2020Ncl,
      ColorMatrix::Smpte240m,
      ColorMatrix::Fcc,
      ColorMatrix::YCgCo,
    ] {
      for full in [true, false] {
        check_p010_u8_avx512_equivalence(64, m, full);
      }
    }
  }

  #[test]
  fn avx512_p010_u16_matches_scalar_all_matrices() {
    for m in [
      ColorMatrix::Bt601,
      ColorMatrix::Bt709,
      ColorMatrix::Bt2020Ncl,
      ColorMatrix::Smpte240m,
      ColorMatrix::Fcc,
      ColorMatrix::YCgCo,
    ] {
      for full in [true, false] {
        check_p010_u16_avx512_equivalence(64, m, full);
      }
    }
  }

  #[test]
  fn avx512_p010_matches_scalar_odd_tail_widths() {
    for w in [66usize, 126, 130, 1922] {
      check_p010_u8_avx512_equivalence(w, ColorMatrix::Bt601, false);
      check_p010_u16_avx512_equivalence(w, ColorMatrix::Bt709, true);
    }
  }

  #[test]
  fn avx512_p010_matches_scalar_1920() {
    check_p010_u8_avx512_equivalence(1920, ColorMatrix::Bt709, false);
    check_p010_u16_avx512_equivalence(1920, ColorMatrix::Bt2020Ncl, false);
  }

  // ---- Generic BITS equivalence (12/14-bit coverage) ------------------

  fn planar_n_plane<const BITS: u32>(n: usize, seed: usize) -> std::vec::Vec<u16> {
    let mask = (1u32 << BITS) - 1;
    (0..n)
      .map(|i| ((i * seed + seed * 3) as u32 & mask) as u16)
      .collect()
  }

  fn p_n_packed_plane<const BITS: u32>(n: usize, seed: usize) -> std::vec::Vec<u16> {
    let mask = (1u32 << BITS) - 1;
    let shift = 16 - BITS;
    (0..n)
      .map(|i| (((i * seed + seed * 3) as u32 & mask) as u16) << shift)
      .collect()
  }

  fn check_planar_u8_avx512_equivalence_n<const BITS: u32>(
    width: usize,
    matrix: ColorMatrix,
    full_range: bool,
  ) {
    if !std::arch::is_x86_feature_detected!("avx512bw") {
      return;
    }
    let y = planar_n_plane::<BITS>(width, 37);
    let u = planar_n_plane::<BITS>(width / 2, 53);
    let v = planar_n_plane::<BITS>(width / 2, 71);
    let mut rgb_scalar = std::vec![0u8; width * 3];
    let mut rgb_simd = std::vec![0u8; width * 3];
    scalar::yuv_420p_n_to_rgb_row::<BITS>(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
    unsafe {
      yuv_420p_n_to_rgb_row::<BITS>(&y, &u, &v, &mut rgb_simd, width, matrix, full_range);
    }
    assert_eq!(
      rgb_scalar, rgb_simd,
      "AVX-512 planar {BITS}-bit → u8 diverges"
    );
  }

  fn check_planar_u16_avx512_equivalence_n<const BITS: u32>(
    width: usize,
    matrix: ColorMatrix,
    full_range: bool,
  ) {
    if !std::arch::is_x86_feature_detected!("avx512bw") {
      return;
    }
    let y = planar_n_plane::<BITS>(width, 37);
    let u = planar_n_plane::<BITS>(width / 2, 53);
    let v = planar_n_plane::<BITS>(width / 2, 71);
    let mut rgb_scalar = std::vec![0u16; width * 3];
    let mut rgb_simd = std::vec![0u16; width * 3];
    scalar::yuv_420p_n_to_rgb_u16_row::<BITS>(
      &y,
      &u,
      &v,
      &mut rgb_scalar,
      width,
      matrix,
      full_range,
    );
    unsafe {
      yuv_420p_n_to_rgb_u16_row::<BITS>(&y, &u, &v, &mut rgb_simd, width, matrix, full_range);
    }
    assert_eq!(
      rgb_scalar, rgb_simd,
      "AVX-512 planar {BITS}-bit → u16 diverges"
    );
  }

  fn check_pn_u8_avx512_equivalence_n<const BITS: u32>(
    width: usize,
    matrix: ColorMatrix,
    full_range: bool,
  ) {
    if !std::arch::is_x86_feature_detected!("avx512bw") {
      return;
    }
    let y = p_n_packed_plane::<BITS>(width, 37);
    let u = p_n_packed_plane::<BITS>(width / 2, 53);
    let v = p_n_packed_plane::<BITS>(width / 2, 71);
    let uv = p010_uv_interleave(&u, &v);
    let mut rgb_scalar = std::vec![0u8; width * 3];
    let mut rgb_simd = std::vec![0u8; width * 3];
    scalar::p_n_to_rgb_row::<BITS>(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
    unsafe {
      p_n_to_rgb_row::<BITS>(&y, &uv, &mut rgb_simd, width, matrix, full_range);
    }
    assert_eq!(rgb_scalar, rgb_simd, "AVX-512 Pn {BITS}-bit → u8 diverges");
  }

  fn check_pn_u16_avx512_equivalence_n<const BITS: u32>(
    width: usize,
    matrix: ColorMatrix,
    full_range: bool,
  ) {
    if !std::arch::is_x86_feature_detected!("avx512bw") {
      return;
    }
    let y = p_n_packed_plane::<BITS>(width, 37);
    let u = p_n_packed_plane::<BITS>(width / 2, 53);
    let v = p_n_packed_plane::<BITS>(width / 2, 71);
    let uv = p010_uv_interleave(&u, &v);
    let mut rgb_scalar = std::vec![0u16; width * 3];
    let mut rgb_simd = std::vec![0u16; width * 3];
    scalar::p_n_to_rgb_u16_row::<BITS>(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
    unsafe {
      p_n_to_rgb_u16_row::<BITS>(&y, &uv, &mut rgb_simd, width, matrix, full_range);
    }
    assert_eq!(rgb_scalar, rgb_simd, "AVX-512 Pn {BITS}-bit → u16 diverges");
  }

  #[test]
  fn avx512_p12_matches_scalar_all_matrices() {
    for m in [
      ColorMatrix::Bt601,
      ColorMatrix::Bt709,
      ColorMatrix::Bt2020Ncl,
      ColorMatrix::Smpte240m,
      ColorMatrix::Fcc,
      ColorMatrix::YCgCo,
    ] {
      for full in [true, false] {
        check_planar_u8_avx512_equivalence_n::<12>(64, m, full);
        check_planar_u16_avx512_equivalence_n::<12>(64, m, full);
        check_pn_u8_avx512_equivalence_n::<12>(64, m, full);
        check_pn_u16_avx512_equivalence_n::<12>(64, m, full);
      }
    }
  }

  #[test]
  fn avx512_p14_matches_scalar_all_matrices() {
    for m in [
      ColorMatrix::Bt601,
      ColorMatrix::Bt709,
      ColorMatrix::Bt2020Ncl,
      ColorMatrix::Smpte240m,
      ColorMatrix::Fcc,
      ColorMatrix::YCgCo,
    ] {
      for full in [true, false] {
        check_planar_u8_avx512_equivalence_n::<14>(64, m, full);
        check_planar_u16_avx512_equivalence_n::<14>(64, m, full);
      }
    }
  }

  #[test]
  fn avx512_p12_matches_scalar_tail_widths() {
    for w in [66usize, 126, 130, 1922] {
      check_planar_u8_avx512_equivalence_n::<12>(w, ColorMatrix::Bt601, false);
      check_planar_u16_avx512_equivalence_n::<12>(w, ColorMatrix::Bt709, true);
      check_pn_u8_avx512_equivalence_n::<12>(w, ColorMatrix::Bt601, false);
      check_pn_u16_avx512_equivalence_n::<12>(w, ColorMatrix::Bt2020Ncl, false);
    }
  }

  #[test]
  fn avx512_p14_matches_scalar_tail_widths() {
    for w in [66usize, 126, 130, 1922] {
      check_planar_u8_avx512_equivalence_n::<14>(w, ColorMatrix::Bt601, false);
      check_planar_u16_avx512_equivalence_n::<14>(w, ColorMatrix::Bt709, true);
    }
  }

  // ---- 16-bit (full-range u16 samples) AVX-512 equivalence ------------

  fn p16_plane_avx512(n: usize, seed: usize) -> std::vec::Vec<u16> {
    (0..n)
      .map(|i| ((i.wrapping_mul(seed).wrapping_add(seed * 3)) & 0xFFFF) as u16)
      .collect()
  }

  fn check_yuv420p16_u8_avx512_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
    if !std::arch::is_x86_feature_detected!("avx512bw") {
      return;
    }
    let y = p16_plane_avx512(width, 37);
    let u = p16_plane_avx512(width / 2, 53);
    let v = p16_plane_avx512(width / 2, 71);
    let mut rgb_scalar = std::vec![0u8; width * 3];
    let mut rgb_simd = std::vec![0u8; width * 3];
    scalar::yuv_420p16_to_rgb_row(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
    unsafe {
      yuv_420p16_to_rgb_row(&y, &u, &v, &mut rgb_simd, width, matrix, full_range);
    }
    assert_eq!(
      rgb_scalar, rgb_simd,
      "AVX-512 yuv420p16→u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
    );
  }

  fn check_yuv420p16_u16_avx512_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
    if !std::arch::is_x86_feature_detected!("avx512bw") {
      return;
    }
    let y = p16_plane_avx512(width, 37);
    let u = p16_plane_avx512(width / 2, 53);
    let v = p16_plane_avx512(width / 2, 71);
    let mut rgb_scalar = std::vec![0u16; width * 3];
    let mut rgb_simd = std::vec![0u16; width * 3];
    scalar::yuv_420p16_to_rgb_u16_row(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
    unsafe {
      yuv_420p16_to_rgb_u16_row(&y, &u, &v, &mut rgb_simd, width, matrix, full_range);
    }
    assert_eq!(
      rgb_scalar, rgb_simd,
      "AVX-512 yuv420p16→u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
    );
  }

  fn check_p16_u8_avx512_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
    if !std::arch::is_x86_feature_detected!("avx512bw") {
      return;
    }
    let y = p16_plane_avx512(width, 37);
    let u = p16_plane_avx512(width / 2, 53);
    let v = p16_plane_avx512(width / 2, 71);
    let uv = p010_uv_interleave(&u, &v);
    let mut rgb_scalar = std::vec![0u8; width * 3];
    let mut rgb_simd = std::vec![0u8; width * 3];
    scalar::p16_to_rgb_row(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
    unsafe {
      p16_to_rgb_row(&y, &uv, &mut rgb_simd, width, matrix, full_range);
    }
    assert_eq!(
      rgb_scalar, rgb_simd,
      "AVX-512 p016→u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
    );
  }

  fn check_p16_u16_avx512_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
    if !std::arch::is_x86_feature_detected!("avx512bw") {
      return;
    }
    let y = p16_plane_avx512(width, 37);
    let u = p16_plane_avx512(width / 2, 53);
    let v = p16_plane_avx512(width / 2, 71);
    let uv = p010_uv_interleave(&u, &v);
    let mut rgb_scalar = std::vec![0u16; width * 3];
    let mut rgb_simd = std::vec![0u16; width * 3];
    scalar::p16_to_rgb_u16_row(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
    unsafe {
      p16_to_rgb_u16_row(&y, &uv, &mut rgb_simd, width, matrix, full_range);
    }
    assert_eq!(
      rgb_scalar, rgb_simd,
      "AVX-512 p016→u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
    );
  }

  #[test]
  fn avx512_p16_matches_scalar_all_matrices() {
    for m in [
      ColorMatrix::Bt601,
      ColorMatrix::Bt709,
      ColorMatrix::Bt2020Ncl,
      ColorMatrix::Smpte240m,
      ColorMatrix::Fcc,
      ColorMatrix::YCgCo,
    ] {
      for full in [true, false] {
        check_yuv420p16_u8_avx512_equivalence(64, m, full);
        check_yuv420p16_u16_avx512_equivalence(64, m, full);
        check_p16_u8_avx512_equivalence(64, m, full);
        check_p16_u16_avx512_equivalence(64, m, full);
      }
    }
  }

  #[test]
  fn avx512_p16_matches_scalar_tail_widths() {
    for w in [66usize, 126, 130, 1922] {
      check_yuv420p16_u8_avx512_equivalence(w, ColorMatrix::Bt601, false);
      check_yuv420p16_u16_avx512_equivalence(w, ColorMatrix::Bt709, true);
      check_p16_u8_avx512_equivalence(w, ColorMatrix::Bt601, false);
      check_p16_u16_avx512_equivalence(w, ColorMatrix::Bt2020Ncl, false);
    }
  }

  #[test]
  fn avx512_p16_matches_scalar_1920() {
    check_yuv420p16_u8_avx512_equivalence(1920, ColorMatrix::Bt709, false);
    check_yuv420p16_u16_avx512_equivalence(1920, ColorMatrix::Bt2020Ncl, false);
    check_p16_u8_avx512_equivalence(1920, ColorMatrix::Bt709, false);
    check_p16_u16_avx512_equivalence(1920, ColorMatrix::Bt2020Ncl, false);
  }

  // ---- Pn 4:4:4 (P410 / P412 / P416) AVX-512 equivalence -------------

  fn high_bit_plane_avx512<const BITS: u32>(n: usize, seed: usize) -> std::vec::Vec<u16> {
    let mask = ((1u32 << BITS) - 1) as u16;
    let shift = 16 - BITS;
    (0..n)
      .map(|i| (((i.wrapping_mul(seed).wrapping_add(seed * 3)) as u16) & mask) << shift)
      .collect()
  }

  fn interleave_uv_avx512(u_full: &[u16], v_full: &[u16]) -> std::vec::Vec<u16> {
    debug_assert_eq!(u_full.len(), v_full.len());
    let mut out = std::vec::Vec::with_capacity(u_full.len() * 2);
    for i in 0..u_full.len() {
      out.push(u_full[i]);
      out.push(v_full[i]);
    }
    out
  }

  fn check_p_n_444_u8_avx512_equivalence<const BITS: u32>(
    width: usize,
    matrix: ColorMatrix,
    full_range: bool,
  ) {
    if !std::arch::is_x86_feature_detected!("avx512bw") {
      return;
    }
    let y = high_bit_plane_avx512::<BITS>(width, 37);
    let u = high_bit_plane_avx512::<BITS>(width, 53);
    let v = high_bit_plane_avx512::<BITS>(width, 71);
    let uv = interleave_uv_avx512(&u, &v);
    let mut rgb_scalar = std::vec![0u8; width * 3];
    let mut rgb_simd = std::vec![0u8; width * 3];
    scalar::p_n_444_to_rgb_row::<BITS>(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
    unsafe {
      p_n_444_to_rgb_row::<BITS>(&y, &uv, &mut rgb_simd, width, matrix, full_range);
    }
    assert_eq!(
      rgb_scalar, rgb_simd,
      "AVX-512 Pn4:4:4 {BITS}-bit → u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
    );
  }

  fn check_p_n_444_u16_avx512_equivalence<const BITS: u32>(
    width: usize,
    matrix: ColorMatrix,
    full_range: bool,
  ) {
    if !std::arch::is_x86_feature_detected!("avx512bw") {
      return;
    }
    let y = high_bit_plane_avx512::<BITS>(width, 37);
    let u = high_bit_plane_avx512::<BITS>(width, 53);
    let v = high_bit_plane_avx512::<BITS>(width, 71);
    let uv = interleave_uv_avx512(&u, &v);
    let mut rgb_scalar = std::vec![0u16; width * 3];
    let mut rgb_simd = std::vec![0u16; width * 3];
    scalar::p_n_444_to_rgb_u16_row::<BITS>(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
    unsafe {
      p_n_444_to_rgb_u16_row::<BITS>(&y, &uv, &mut rgb_simd, width, matrix, full_range);
    }
    assert_eq!(
      rgb_scalar, rgb_simd,
      "AVX-512 Pn4:4:4 {BITS}-bit → u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
    );
  }

  fn check_p_n_444_16_u8_avx512_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
    if !std::arch::is_x86_feature_detected!("avx512bw") {
      return;
    }
    let y = p16_plane_avx512(width, 37);
    let u = p16_plane_avx512(width, 53);
    let v = p16_plane_avx512(width, 71);
    let uv = interleave_uv_avx512(&u, &v);
    let mut rgb_scalar = std::vec![0u8; width * 3];
    let mut rgb_simd = std::vec![0u8; width * 3];
    scalar::p_n_444_16_to_rgb_row(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
    unsafe {
      p_n_444_16_to_rgb_row(&y, &uv, &mut rgb_simd, width, matrix, full_range);
    }
    assert_eq!(
      rgb_scalar, rgb_simd,
      "AVX-512 P416 → u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
    );
  }

  fn check_p_n_444_16_u16_avx512_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
    if !std::arch::is_x86_feature_detected!("avx512bw") {
      return;
    }
    let y = p16_plane_avx512(width, 37);
    let u = p16_plane_avx512(width, 53);
    let v = p16_plane_avx512(width, 71);
    let uv = interleave_uv_avx512(&u, &v);
    let mut rgb_scalar = std::vec![0u16; width * 3];
    let mut rgb_simd = std::vec![0u16; width * 3];
    scalar::p_n_444_16_to_rgb_u16_row(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
    unsafe {
      p_n_444_16_to_rgb_u16_row(&y, &uv, &mut rgb_simd, width, matrix, full_range);
    }
    assert_eq!(
      rgb_scalar, rgb_simd,
      "AVX-512 P416 → u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
    );
  }

  #[test]
  fn avx512_p410_matches_scalar_all_matrices() {
    for m in [
      ColorMatrix::Bt601,
      ColorMatrix::Bt709,
      ColorMatrix::Bt2020Ncl,
      ColorMatrix::Smpte240m,
      ColorMatrix::Fcc,
      ColorMatrix::YCgCo,
    ] {
      for full in [true, false] {
        check_p_n_444_u8_avx512_equivalence::<10>(64, m, full);
        check_p_n_444_u16_avx512_equivalence::<10>(64, m, full);
      }
    }
  }

  #[test]
  fn avx512_p412_matches_scalar_all_matrices() {
    for m in [ColorMatrix::Bt709, ColorMatrix::Bt2020Ncl] {
      for full in [true, false] {
        check_p_n_444_u8_avx512_equivalence::<12>(64, m, full);
        check_p_n_444_u16_avx512_equivalence::<12>(64, m, full);
      }
    }
  }

  #[test]
  fn avx512_p410_p412_matches_scalar_tail_widths() {
    // AVX-512 main loop is 64 px (or 32 px for the i64 u16 path);
    // tail widths force scalar fallback.
    for w in [1usize, 3, 33, 63, 65, 95, 127, 129, 1920, 1921] {
      check_p_n_444_u8_avx512_equivalence::<10>(w, ColorMatrix::Bt601, false);
      check_p_n_444_u16_avx512_equivalence::<10>(w, ColorMatrix::Bt709, true);
      check_p_n_444_u8_avx512_equivalence::<12>(w, ColorMatrix::Bt2020Ncl, false);
    }
  }

  #[test]
  fn avx512_p416_matches_scalar_all_matrices() {
    for m in [
      ColorMatrix::Bt601,
      ColorMatrix::Bt709,
      ColorMatrix::Bt2020Ncl,
      ColorMatrix::Smpte240m,
      ColorMatrix::Fcc,
      ColorMatrix::YCgCo,
    ] {
      for full in [true, false] {
        check_p_n_444_16_u8_avx512_equivalence(64, m, full);
        check_p_n_444_16_u16_avx512_equivalence(64, m, full);
      }
    }
  }

  #[test]
  fn avx512_p416_matches_scalar_tail_widths() {
    for w in [1usize, 3, 31, 33, 63, 65, 95, 127, 129, 1920, 1921] {
      check_p_n_444_16_u8_avx512_equivalence(w, ColorMatrix::Bt709, false);
      check_p_n_444_16_u16_avx512_equivalence(w, ColorMatrix::Bt2020Ncl, true);
    }
  }
}
