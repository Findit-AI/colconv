//! x86_64 AVX2 backend for the row primitives.
//!
//! Selected by [`crate::row`]'s dispatcher after
//! `is_x86_feature_detected!("avx2")` returns true (runtime, std‑gated)
//! or `cfg!(target_feature = "avx2")` evaluates true (compile‑time,
//! no‑std). The kernel itself carries `#[target_feature(enable = "avx2")]`
//! so its intrinsics execute in an explicitly AVX2‑enabled context.
//!
//! # Numerical contract
//!
//! Bit‑identical to
//! [`crate::row::scalar::yuv_420_to_rgb_row`]. All Q15 multiplies
//! are i32‑widened with `(prod + (1 << 14)) >> 15` rounding — same
//! structure as the NEON backend.
//!
//! # Pipeline (per 32 Y pixels / 16 chroma samples)
//!
//! 1. Load 32 Y (`_mm256_loadu_si256`) + 16 U (`_mm_loadu_si128`) +
//!    16 V (`_mm_loadu_si128`).
//! 2. Widen U, V to i16x16, subtract 128.
//! 3. Split each i16x16 into two i32x8 halves and apply `c_scale`.
//! 4. Per channel C ∈ {R, G, B}: compute `(C_u*u_d + C_v*v_d + RND) >> 15`
//!    in i32, narrow‑saturate to i16x16.
//! 5. Nearest‑neighbor chroma upsample: duplicate each of the 16 chroma
//!    lanes into its pair slot → two i16x16 vectors covering 32 Y
//!    lanes.
//! 6. Y path: widen 32 Y to two i16x16 vectors, apply `y_off` / `y_scale`.
//! 7. Saturating i16 add Y + chroma per channel.
//! 8. Saturate‑narrow to u8x32 per channel, then interleave as packed
//!    RGB via two halves of `_mm_shuffle_epi8` 3‑way interleave.
//!
//! # AVX2 lane‑crossing fixups
//!
//! Several AVX2 ops (`packs_epi32`, `packus_epi16`, `unpack*_epi16`,
//! `permute2x128_si256`) operate per 128‑bit lane, producing
//! lane‑split results. Each such op is immediately followed by the
//! correct permute (`permute4x64_epi64::<0xD8>` for pack results,
//! `permute2x128_si256` for unpack‑and‑split) to restore natural
//! element order. Every fixup is called out inline.

use core::arch::x86_64::{
  __m256i, _mm_loadu_si128, _mm256_add_epi32, _mm256_adds_epi16, _mm256_castsi256_si128,
  _mm256_cvtepi16_epi32, _mm256_cvtepu8_epi16, _mm256_extracti128_si256, _mm256_loadu_si256,
  _mm256_mullo_epi32, _mm256_packs_epi32, _mm256_packus_epi16, _mm256_permute2x128_si256,
  _mm256_permute4x64_epi64, _mm256_set1_epi16, _mm256_set1_epi32, _mm256_setr_epi8,
  _mm256_shuffle_epi8, _mm256_srai_epi32, _mm256_sub_epi16, _mm256_unpackhi_epi16,
  _mm256_unpacklo_epi16,
};

use crate::{
  ColorMatrix,
  row::{
    arch::x86_common::{rgb_to_hsv_16_pixels, swap_rb_16_pixels, write_rgb_16},
    scalar,
  },
};

/// AVX2 YUV 4:2:0 → packed RGB. Semantics match
/// [`scalar::yuv_420_to_rgb_row`] byte‑identically.
///
/// # Safety
///
/// The caller must uphold **all** of the following. Violating any
/// causes undefined behavior:
///
/// 1. **AVX2 must be available on the current CPU.** The dispatcher
///    in [`crate::row`] verifies this with
///    `is_x86_feature_detected!("avx2")` (runtime, std) or
///    `cfg!(target_feature = "avx2")` (compile‑time, no‑std). Calling
///    this kernel on a CPU without AVX2 triggers an illegal‑instruction
///    trap.
/// 2. `width & 1 == 0` (4:2:0 requires even width).
/// 3. `y.len() >= width`.
/// 4. `u_half.len() >= width / 2`.
/// 5. `v_half.len() >= width / 2`.
/// 6. `rgb_out.len() >= 3 * width`.
///
/// Bounds are verified by `debug_assert` in debug builds; release
/// builds trust the caller because the kernel relies on unchecked
/// pointer arithmetic (`_mm256_loadu_si256`, `_mm_loadu_si128`,
/// `_mm_storeu_si128`).
#[inline]
#[target_feature(enable = "avx2")]
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

  // SAFETY: AVX2 availability is the caller's obligation per the
  // `# Safety` section; the dispatcher in `crate::row` checks it.
  // All pointer adds below are bounded by the `while x + 32 <= width`
  // loop condition and the caller‑promised slice lengths.
  unsafe {
    let rnd_v = _mm256_set1_epi32(RND);
    let y_off_v = _mm256_set1_epi16(y_off as i16);
    let y_scale_v = _mm256_set1_epi32(y_scale);
    let c_scale_v = _mm256_set1_epi32(c_scale);
    let mid128 = _mm256_set1_epi16(128);
    let cru = _mm256_set1_epi32(coeffs.r_u());
    let crv = _mm256_set1_epi32(coeffs.r_v());
    let cgu = _mm256_set1_epi32(coeffs.g_u());
    let cgv = _mm256_set1_epi32(coeffs.g_v());
    let cbu = _mm256_set1_epi32(coeffs.b_u());
    let cbv = _mm256_set1_epi32(coeffs.b_v());

    let mut x = 0usize;
    while x + 32 <= width {
      // Load 32 Y, 16 U, 16 V.
      let y_vec = _mm256_loadu_si256(y.as_ptr().add(x).cast());
      let u_vec_128 = _mm_loadu_si128(u_half.as_ptr().add(x / 2).cast());
      let v_vec_128 = _mm_loadu_si128(v_half.as_ptr().add(x / 2).cast());

      // Widen U/V to i16x16 and subtract 128.
      let u_i16 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(u_vec_128), mid128);
      let v_i16 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(v_vec_128), mid128);

      // Split each i16x16 into two i32x8 halves for the Q15 multiplies
      // (coefficients exceed i16, so i32 precision is required).
      let u_lo_i32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(u_i16));
      let u_hi_i32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(u_i16));
      let v_lo_i32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v_i16));
      let v_hi_i32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(v_i16));

      // u_d, v_d = (u * c_scale + RND) >> 15 — bit‑exact to scalar.
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

      // Per‑channel chroma → i16x16 (natural order, fixup included).
      let r_chroma = chroma_i16x16(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let g_chroma = chroma_i16x16(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let b_chroma = chroma_i16x16(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);

      // Nearest‑neighbor upsample: each of the 16 chroma lanes →
      // an adjacent pair, covering 32 Y lanes (split into low‑16 and
      // high‑16 i16x16 vectors).
      let (r_dup_lo, r_dup_hi) = chroma_dup(r_chroma);
      let (g_dup_lo, g_dup_hi) = chroma_dup(g_chroma);
      let (b_dup_lo, b_dup_hi) = chroma_dup(b_chroma);

      // Y path: widen 32 Y to two i16x16 vectors, subtract y_off,
      // apply y_scale in Q15, narrow back to i16.
      let y_low_i16 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(y_vec));
      let y_high_i16 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(y_vec));
      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v);

      // Saturating i16 add Y + chroma per channel.
      let b_lo = _mm256_adds_epi16(y_scaled_lo, b_dup_lo);
      let b_hi = _mm256_adds_epi16(y_scaled_hi, b_dup_hi);
      let g_lo = _mm256_adds_epi16(y_scaled_lo, g_dup_lo);
      let g_hi = _mm256_adds_epi16(y_scaled_hi, g_dup_hi);
      let r_lo = _mm256_adds_epi16(y_scaled_lo, r_dup_lo);
      let r_hi = _mm256_adds_epi16(y_scaled_hi, r_dup_hi);

      // Saturate‑narrow to u8x32 per channel (lane‑fixup included).
      let b_u8 = narrow_u8x32(b_lo, b_hi);
      let g_u8 = narrow_u8x32(g_lo, g_hi);
      let r_u8 = narrow_u8x32(r_lo, r_hi);

      // 3‑way interleave → packed RGB (96 bytes = 3 × 32).
      write_rgb_32(r_u8, g_u8, b_u8, rgb_out.as_mut_ptr().add(x * 3));

      x += 32;
    }

    // Scalar tail for the 0..30 leftover pixels (always even; 4:2:0
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

/// AVX2 YUV 4:2:0 10‑bit → packed **8‑bit** RGB. **Stub**.
///
/// # Safety
///
/// 1. **AVX2 must be available on the current CPU.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `u_half.len() >= width / 2`,
///    `v_half.len() >= width / 2`, `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn yuv420p10_to_rgb_row(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  scalar::yuv_420p_n_to_rgb_row::<10>(y, u_half, v_half, rgb_out, width, matrix, full_range);
}

/// AVX2 YUV 4:2:0 10‑bit → packed **10‑bit `u16`** RGB. **Stub**.
///
/// # Safety
///
/// 1. **AVX2 must be available on the current CPU.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `u_half.len() >= width / 2`,
///    `v_half.len() >= width / 2`, `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn yuv420p10_to_rgb_u16_row(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  scalar::yuv_420p_n_to_rgb_u16_row::<10>(y, u_half, v_half, rgb_out, width, matrix, full_range);
}

/// AVX2 NV12 → packed RGB (UV-ordered chroma). Thin wrapper over
/// [`nv12_or_nv21_to_rgb_row_impl`] with `SWAP_UV = false`.
///
/// # Safety
///
/// Same as [`nv12_or_nv21_to_rgb_row_impl`].
#[inline]
#[target_feature(enable = "avx2")]
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

/// AVX2 NV21 → packed RGB (VU-ordered chroma). Thin wrapper over
/// [`nv12_or_nv21_to_rgb_row_impl`] with `SWAP_UV = true`.
///
/// # Safety
///
/// Same as [`nv12_or_nv21_to_rgb_row_impl`].
#[inline]
#[target_feature(enable = "avx2")]
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

/// Shared AVX2 NV12/NV21 kernel. `SWAP_UV` selects chroma byte order
/// at compile time.
///
/// # Safety
///
/// 1. **AVX2 must be available on the current CPU** (same obligation
///    as [`yuv_420_to_rgb_row`]).
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`.
/// 4. `uv_or_vu_half.len() >= width` (32 interleaved bytes per 32 Y pixels).
/// 5. `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "avx2")]
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

  // SAFETY: AVX2 availability is the caller's obligation; all pointer
  // adds below are bounded by the `while x + 32 <= width` condition and
  // the caller‑promised slice lengths.
  unsafe {
    let rnd_v = _mm256_set1_epi32(RND);
    let y_off_v = _mm256_set1_epi16(y_off as i16);
    let y_scale_v = _mm256_set1_epi32(y_scale);
    let c_scale_v = _mm256_set1_epi32(c_scale);
    let mid128 = _mm256_set1_epi16(128);
    let cru = _mm256_set1_epi32(coeffs.r_u());
    let crv = _mm256_set1_epi32(coeffs.r_v());
    let cgu = _mm256_set1_epi32(coeffs.g_u());
    let cgv = _mm256_set1_epi32(coeffs.g_v());
    let cbu = _mm256_set1_epi32(coeffs.b_u());
    let cbv = _mm256_set1_epi32(coeffs.b_v());

    // Per‑lane shuffle: pack U bytes (even offsets) into low 8 of each
    // 128‑bit lane, V bytes (odd offsets) into the high 8. Applied to a
    // `[u0v0..u7v7 | u8v8..u15v15]` load, the result is
    // `[u0..u7, v0..v7 | u8..u15, v8..v15]`.
    let deint_mask = _mm256_setr_epi8(
      0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15, // low 128: per-lane dedup
      0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15, // high 128: same
    );

    let mut x = 0usize;
    while x + 32 <= width {
      let y_vec = _mm256_loadu_si256(y.as_ptr().add(x).cast());
      // 32 Y pixels → 16 chroma pairs = 32 interleaved bytes at
      // offset `x` in the chroma row.
      let uv_vec = _mm256_loadu_si256(uv_or_vu_half.as_ptr().add(x).cast());

      // Per‑lane deinterleave: even-offset bytes → low 8, odd-offset
      // bytes → high 8 (per 128-bit lane). After the 64-bit permute,
      // low 128 = even bytes, high 128 = odd bytes. For NV12 that
      // means low=U, high=V; for NV21 the roles swap.
      let deint = _mm256_shuffle_epi8(uv_vec, deint_mask);
      let uv_fixed = _mm256_permute4x64_epi64::<0xD8>(deint);
      let (u_vec_128, v_vec_128) = if SWAP_UV {
        (
          _mm256_extracti128_si256::<1>(uv_fixed),
          _mm256_castsi256_si128(uv_fixed),
        )
      } else {
        (
          _mm256_castsi256_si128(uv_fixed),
          _mm256_extracti128_si256::<1>(uv_fixed),
        )
      };

      let u_i16 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(u_vec_128), mid128);
      let v_i16 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(v_vec_128), mid128);

      let u_lo_i32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(u_i16));
      let u_hi_i32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(u_i16));
      let v_lo_i32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v_i16));
      let v_hi_i32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(v_i16));

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

      let r_chroma = chroma_i16x16(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let g_chroma = chroma_i16x16(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let b_chroma = chroma_i16x16(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);

      let (r_dup_lo, r_dup_hi) = chroma_dup(r_chroma);
      let (g_dup_lo, g_dup_hi) = chroma_dup(g_chroma);
      let (b_dup_lo, b_dup_hi) = chroma_dup(b_chroma);

      let y_low_i16 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(y_vec));
      let y_high_i16 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(y_vec));
      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v);

      let b_lo = _mm256_adds_epi16(y_scaled_lo, b_dup_lo);
      let b_hi = _mm256_adds_epi16(y_scaled_hi, b_dup_hi);
      let g_lo = _mm256_adds_epi16(y_scaled_lo, g_dup_lo);
      let g_hi = _mm256_adds_epi16(y_scaled_hi, g_dup_hi);
      let r_lo = _mm256_adds_epi16(y_scaled_lo, r_dup_lo);
      let r_hi = _mm256_adds_epi16(y_scaled_hi, r_dup_hi);

      let b_u8 = narrow_u8x32(b_lo, b_hi);
      let g_u8 = narrow_u8x32(g_lo, g_hi);
      let r_u8 = narrow_u8x32(r_lo, r_hi);

      write_rgb_32(r_u8, g_u8, b_u8, rgb_out.as_mut_ptr().add(x * 3));

      x += 32;
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

// ---- helpers (all `#[inline(always)]` so the `#[target_feature]`
// context from the caller flows through) --------------------------------

/// `>>_a 15` shift (arithmetic, sign‑extending).
#[inline(always)]
fn q15_shift(v: __m256i) -> __m256i {
  unsafe { _mm256_srai_epi32::<15>(v) }
}

/// Computes one i16x16 chroma channel vector from the 4 × i32x8 chroma
/// inputs (lo/hi splits of u_d and v_d). Mirrors the scalar
/// `(coeff_u * u_d + coeff_v * v_d + RND) >> 15`, then saturating‑packs
/// to i16x16 and **fixes the lane order** with
/// `permute4x64_epi64::<0xD8>` so the result is in natural
/// `[0..16)` element order rather than the per‑lane‑split form
/// `_mm256_packs_epi32` produces.
#[inline(always)]
fn chroma_i16x16(
  cu: __m256i,
  cv: __m256i,
  u_d_lo: __m256i,
  v_d_lo: __m256i,
  u_d_hi: __m256i,
  v_d_hi: __m256i,
  rnd: __m256i,
) -> __m256i {
  unsafe {
    let lo = _mm256_srai_epi32::<15>(_mm256_add_epi32(
      _mm256_add_epi32(
        _mm256_mullo_epi32(cu, u_d_lo),
        _mm256_mullo_epi32(cv, v_d_lo),
      ),
      rnd,
    ));
    let hi = _mm256_srai_epi32::<15>(_mm256_add_epi32(
      _mm256_add_epi32(
        _mm256_mullo_epi32(cu, u_d_hi),
        _mm256_mullo_epi32(cv, v_d_hi),
      ),
      rnd,
    ));
    // `packs_epi32` produces lane‑split [lo0..3, hi0..3, lo4..7, hi4..7];
    // 0xD8 = 0b11_01_10_00 reorders 64‑bit lanes to [0, 2, 1, 3] giving
    // natural [lo0..7, hi0..7].
    _mm256_permute4x64_epi64::<0xD8>(_mm256_packs_epi32(lo, hi))
  }
}

/// `(Y - y_off) * y_scale + RND >> 15` applied to an i16x16 vector,
/// returned as i16x16. The Q15 multiply uses i32 widening identical to
/// scalar, then the result is saturating‑packed back to i16 (result is
/// in [0, 255] range so no saturation occurs in practice).
#[inline(always)]
fn scale_y(y_i16: __m256i, y_off_v: __m256i, y_scale_v: __m256i, rnd: __m256i) -> __m256i {
  unsafe {
    let shifted = _mm256_sub_epi16(y_i16, y_off_v);
    // Widen to two i32x8 halves.
    let lo_i32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(shifted));
    let hi_i32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(shifted));
    let lo_scaled =
      _mm256_srai_epi32::<15>(_mm256_add_epi32(_mm256_mullo_epi32(lo_i32, y_scale_v), rnd));
    let hi_scaled =
      _mm256_srai_epi32::<15>(_mm256_add_epi32(_mm256_mullo_epi32(hi_i32, y_scale_v), rnd));
    // Narrow + lane fixup (same pattern as `chroma_i16x16`).
    _mm256_permute4x64_epi64::<0xD8>(_mm256_packs_epi32(lo_scaled, hi_scaled))
  }
}

/// Duplicates each of the 16 chroma lanes in `chroma` into its adjacent
/// pair slot, splitting the result across two i16x16 vectors that
/// cover 32 Y lanes:
///
/// - Return.0 (for Y[0..16]): `[c0,c0, c1,c1, ..., c7,c7]`.
/// - Return.1 (for Y[16..32]): `[c8,c8, c9,c9, ..., c15,c15]`.
///
/// `_mm256_unpack*_epi16` are per‑128‑bit‑lane, so they produce
/// interleaved‑but‑lane‑split outputs; `_mm256_permute2x128_si256`
/// with selectors 0x20 / 0x31 selects the matching halves from each
/// unpack to restore the per‑Y‑block order above.
#[inline(always)]
fn chroma_dup(chroma: __m256i) -> (__m256i, __m256i) {
  unsafe {
    // unpacklo per‑lane: [c0,c0,c1,c1,c2,c2,c3,c3, c8,c8,c9,c9,c10,c10,c11,c11]
    // unpackhi per‑lane: [c4,c4,c5,c5,c6,c6,c7,c7, c12,c12,c13,c13,c14,c14,c15,c15]
    let a = _mm256_unpacklo_epi16(chroma, chroma);
    let b = _mm256_unpackhi_epi16(chroma, chroma);
    // 0x20 = take 128‑bit lane 0 from a, lane 0 from b
    //      → [c0..3 dup, c4..7 dup] = pair‑expanded c0..c7.
    // 0x31 = take lane 1 from a, lane 1 from b
    //      → [c8..11 dup, c12..15 dup] = pair‑expanded c8..c15.
    let lo16 = _mm256_permute2x128_si256::<0x20>(a, b);
    let hi16 = _mm256_permute2x128_si256::<0x31>(a, b);
    (lo16, hi16)
  }
}

/// Saturating‑narrows two i16x16 vectors into one u8x32 with natural
/// element order. `_mm256_packus_epi16` is per‑lane and produces
/// lane‑split u8x32; `permute4x64_epi64::<0xD8>` fixes it.
#[inline(always)]
fn narrow_u8x32(lo: __m256i, hi: __m256i) -> __m256i {
  unsafe { _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(lo, hi)) }
}

/// Writes 32 pixels of packed RGB (96 bytes) by interleaving three
/// u8x32 B/G/R channel vectors. Processed as two 16‑pixel halves via
/// the shared [`write_rgb_16`](super::x86_common::write_rgb_16) helper.
///
/// # Safety
///
/// `ptr` must point to at least 96 writable bytes.
#[inline(always)]
unsafe fn write_rgb_32(r: __m256i, g: __m256i, b: __m256i, ptr: *mut u8) {
  unsafe {
    let r_lo = _mm256_castsi256_si128(r);
    let r_hi = _mm256_extracti128_si256::<1>(r);
    let g_lo = _mm256_castsi256_si128(g);
    let g_hi = _mm256_extracti128_si256::<1>(g);
    let b_lo = _mm256_castsi256_si128(b);
    let b_hi = _mm256_extracti128_si256::<1>(b);

    write_rgb_16(r_lo, g_lo, b_lo, ptr);
    write_rgb_16(r_hi, g_hi, b_hi, ptr.add(48));
  }
}

// ===== BGR ↔ RGB byte swap ==============================================

/// AVX2 BGR ↔ RGB byte swap. 32 pixels per iteration by invoking the
/// shared [`super::x86_common::swap_rb_16_pixels`] helper twice — the op
/// is memory‑bandwidth‑bound, so wider registers wouldn't change the
/// practical throughput.
///
/// # Safety
///
/// 1. AVX2 must be available (dispatcher obligation) — AVX2 is a
///    superset of SSSE3, which the shared helper requires.
/// 2. `input.len() >= 3 * width`.
/// 3. `output.len() >= 3 * width`.
/// 4. `input` / `output` must not alias.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn bgr_rgb_swap_row(input: &[u8], output: &mut [u8], width: usize) {
  debug_assert!(input.len() >= width * 3, "input row too short");
  debug_assert!(output.len() >= width * 3, "output row too short");

  unsafe {
    let mut x = 0usize;
    while x + 32 <= width {
      swap_rb_16_pixels(input.as_ptr().add(x * 3), output.as_mut_ptr().add(x * 3));
      swap_rb_16_pixels(
        input.as_ptr().add(x * 3 + 48),
        output.as_mut_ptr().add(x * 3 + 48),
      );
      x += 32;
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

/// AVX2 RGB → planar HSV. 32 pixels per iteration via two calls to the
/// shared [`super::x86_common::rgb_to_hsv_16_pixels`] helper (SSE4.1
/// level compute, memory‑bandwidth‑bound — wider f32 registers would
/// help if we restructured, but the current structure already wins
/// versus scalar).
///
/// # Safety
///
/// 1. AVX2 must be available (dispatcher obligation).
/// 2. `rgb.len() >= 3 * width`; each output plane `>= width`.
#[inline]
#[target_feature(enable = "avx2")]
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
    while x + 32 <= width {
      rgb_to_hsv_16_pixels(
        rgb.as_ptr().add(x * 3),
        h_out.as_mut_ptr().add(x),
        s_out.as_mut_ptr().add(x),
        v_out.as_mut_ptr().add(x),
      );
      rgb_to_hsv_16_pixels(
        rgb.as_ptr().add(x * 3 + 48),
        h_out.as_mut_ptr().add(x + 16),
        s_out.as_mut_ptr().add(x + 16),
        v_out.as_mut_ptr().add(x + 16),
      );
      x += 32;
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
    let mut rgb_avx2 = std::vec![0u8; width * 3];

    scalar::yuv_420_to_rgb_row(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
    unsafe {
      yuv_420_to_rgb_row(&y, &u, &v, &mut rgb_avx2, width, matrix, full_range);
    }

    if rgb_scalar != rgb_avx2 {
      let first_diff = rgb_scalar
        .iter()
        .zip(rgb_avx2.iter())
        .position(|(a, b)| a != b)
        .unwrap();
      panic!(
        "AVX2 diverges from scalar at byte {first_diff} (width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} avx2={}",
        rgb_scalar[first_diff], rgb_avx2[first_diff]
      );
    }
  }

  #[test]
  fn avx2_matches_scalar_all_matrices_32() {
    if !std::arch::is_x86_feature_detected!("avx2") {
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
        check_equivalence(32, m, full);
      }
    }
  }

  #[test]
  fn avx2_matches_scalar_width_64() {
    if !std::arch::is_x86_feature_detected!("avx2") {
      return;
    }
    check_equivalence(64, ColorMatrix::Bt601, true);
    check_equivalence(64, ColorMatrix::Bt709, false);
    check_equivalence(64, ColorMatrix::YCgCo, true);
  }

  #[test]
  fn avx2_matches_scalar_width_1920() {
    if !std::arch::is_x86_feature_detected!("avx2") {
      return;
    }
    check_equivalence(1920, ColorMatrix::Bt709, false);
  }

  #[test]
  fn avx2_matches_scalar_odd_tail_widths() {
    if !std::arch::is_x86_feature_detected!("avx2") {
      return;
    }
    // Widths that leave a non‑trivial scalar tail (non‑multiple of 32).
    for w in [34usize, 46, 62, 1922] {
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
    let mut rgb_avx2 = std::vec![0u8; width * 3];

    scalar::nv12_to_rgb_row(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
    unsafe {
      nv12_to_rgb_row(&y, &uv, &mut rgb_avx2, width, matrix, full_range);
    }
    assert_eq!(
      rgb_scalar, rgb_avx2,
      "AVX2 NV12 ≠ scalar (width={width}, matrix={matrix:?})"
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
      "AVX2 NV12 ≠ YUV420P for equivalent UV"
    );
  }

  #[test]
  fn avx2_nv12_matches_scalar_all_matrices_32() {
    if !std::arch::is_x86_feature_detected!("avx2") {
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
        check_nv12_equivalence(32, m, full);
      }
    }
  }

  #[test]
  fn avx2_nv12_matches_scalar_widths() {
    if !std::arch::is_x86_feature_detected!("avx2") {
      return;
    }
    for w in [64usize, 1920, 34, 46, 62, 1922] {
      check_nv12_equivalence(w, ColorMatrix::Bt709, false);
    }
  }

  #[test]
  fn avx2_nv12_matches_yuv420p() {
    if !std::arch::is_x86_feature_detected!("avx2") {
      return;
    }
    for w in [32usize, 62, 128, 1920] {
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
    let mut out_avx2 = std::vec![0u8; width * 3];

    scalar::bgr_rgb_swap_row(&input, &mut out_scalar, width);
    unsafe {
      bgr_rgb_swap_row(&input, &mut out_avx2, width);
    }
    assert_eq!(out_scalar, out_avx2, "AVX2 swap diverges from scalar");
  }

  #[test]
  fn avx2_swap_matches_scalar() {
    if !std::arch::is_x86_feature_detected!("avx2") {
      return;
    }
    for w in [1usize, 15, 31, 32, 33, 47, 48, 63, 64, 1920, 1921] {
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
    let mut rgb_avx2 = std::vec![0u8; width * 3];

    scalar::nv21_to_rgb_row(&y, &vu, &mut rgb_scalar, width, matrix, full_range);
    unsafe {
      nv21_to_rgb_row(&y, &vu, &mut rgb_avx2, width, matrix, full_range);
    }
    assert_eq!(
      rgb_scalar, rgb_avx2,
      "AVX2 NV21 ≠ scalar (width={width}, matrix={matrix:?})"
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
      "AVX2 NV21 ≠ NV12 with byte-swapped chroma"
    );
  }

  #[test]
  fn nv21_avx2_matches_scalar_all_matrices_16() {
    if !std::arch::is_x86_feature_detected!("avx2") {
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
  fn nv21_avx2_matches_scalar_widths() {
    if !std::arch::is_x86_feature_detected!("avx2") {
      return;
    }
    for w in [32usize, 1920, 18, 30, 34, 1922] {
      check_nv21_equivalence(w, ColorMatrix::Bt709, false);
    }
  }

  #[test]
  fn nv21_avx2_matches_nv12_swapped() {
    if !std::arch::is_x86_feature_detected!("avx2") {
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
  fn avx2_hsv_matches_scalar() {
    if !std::arch::is_x86_feature_detected!("avx2") {
      return;
    }
    let rgb: std::vec::Vec<u8> = (0..1921 * 3)
      .map(|i| ((i * 37 + 11) & 0xFF) as u8)
      .collect();
    for w in [1usize, 31, 32, 33, 63, 64, 1920, 1921] {
      check_hsv_equivalence(&rgb[..w * 3], w);
    }
  }
}
