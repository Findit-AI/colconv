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

use core::arch::x86_64::*;

use crate::{
  ColorMatrix,
  row::{
    arch::x86_common::{
      rgb_to_hsv_16_pixels, swap_rb_16_pixels, write_rgb_16, write_rgb_u16_8, write_rgba_16,
      write_rgba_u16_8,
    },
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
  // SAFETY: caller-checked AVX2 availability + slice bounds — see
  // [`yuv_420_to_rgb_or_rgba_row`] safety contract.
  unsafe {
    yuv_420_to_rgb_or_rgba_row::<false>(y, u_half, v_half, rgb_out, width, matrix, full_range);
  }
}

/// AVX2 YUV 4:2:0 → packed **RGBA** (8-bit). Same contract as
/// [`yuv_420_to_rgb_row`] but writes 4 bytes per pixel (R, G, B,
/// `0xFF`).
///
/// # Safety
///
/// 1. AVX2 must be available on the current CPU.
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `u_half.len() >= width / 2`,
///    `v_half.len() >= width / 2`, `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn yuv_420_to_rgba_row(
  y: &[u8],
  u_half: &[u8],
  v_half: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller-checked AVX2 availability + slice bounds — see
  // [`yuv_420_to_rgb_or_rgba_row`] safety contract.
  unsafe {
    yuv_420_to_rgb_or_rgba_row::<true>(y, u_half, v_half, rgba_out, width, matrix, full_range);
  }
}

/// Shared AVX2 kernel for [`yuv_420_to_rgb_row`] (`ALPHA = false`,
/// [`write_rgb_32`]) and [`yuv_420_to_rgba_row`] (`ALPHA = true`,
/// [`write_rgba_32`] with constant `0xFF` alpha). Math is
/// byte-identical to `scalar::yuv_420_to_rgb_or_rgba_row::<ALPHA>`.
///
/// # Safety
///
/// Same as [`yuv_420_to_rgb_row`] / [`yuv_420_to_rgba_row`]; the
/// `out` slice must be `>= width * (if ALPHA { 4 } else { 3 })`
/// bytes long.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn yuv_420_to_rgb_or_rgba_row<const ALPHA: bool>(
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
    // Constant opaque-alpha vector for the RGBA path; DCE'd when
    // ALPHA = false.
    let alpha_u8 = _mm256_set1_epi8(-1); // 0xFF as i8

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

      if ALPHA {
        // 4‑way interleave → packed RGBA (128 bytes = 4 × 32).
        write_rgba_32(r_u8, g_u8, b_u8, alpha_u8, out.as_mut_ptr().add(x * 4));
      } else {
        // 3‑way interleave → packed RGB (96 bytes = 3 × 32).
        write_rgb_32(r_u8, g_u8, b_u8, out.as_mut_ptr().add(x * 3));
      }

      x += 32;
    }

    // Scalar tail for the 0..30 leftover pixels (always even; 4:2:0
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

/// AVX2 YUV 4:2:0 10‑bit → packed **8‑bit** RGB.
///
/// Block size 32 Y pixels per iteration (matching the 8‑bit AVX2
/// kernel). Key differences:
/// - Two `_mm256_loadu_si256` loads for Y (each 16 `u16` = 32 bytes);
///   one load each for U / V (16 `u16` = 32 bytes).
/// - No u8→i16 widening — 10‑bit samples already occupy 16‑bit lanes
///   and fit i16 without overflow.
/// - Chroma bias is 512 (10‑bit center).
/// - `range_params_n::<10, 8>` calibrates scales for 10→8 in one shift.
///
/// Reuses [`chroma_i16x16`], [`chroma_dup`], [`scale_y`],
/// [`narrow_u8x32`], and [`write_rgb_32`] from the 8‑bit path — the
/// post‑chroma math is identical across bit depths.
///
/// # Numerical contract
///
/// Byte‑identical to [`scalar::yuv_420p_n_to_rgb_row::<10>`].
///
/// # Safety
///
/// 1. **AVX2 must be available on the current CPU.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `u_half.len() >= width / 2`,
///    `v_half.len() >= width / 2`, `rgb_out.len() >= 3 * width`.
///
/// Thin wrapper over [`yuv_420p_n_to_rgb_or_rgba_row`] with `ALPHA = false`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn yuv_420p_n_to_rgb_row<const BITS: u32>(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    yuv_420p_n_to_rgb_or_rgba_row::<BITS, false>(
      y, u_half, v_half, rgb_out, width, matrix, full_range,
    );
  }
}

/// AVX2 high-bit-depth YUV 4:2:0 → packed **8-bit RGBA** (`R, G, B, 0xFF`).
///
/// Thin wrapper over [`yuv_420p_n_to_rgb_or_rgba_row`] with `ALPHA = true`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn yuv_420p_n_to_rgba_row<const BITS: u32>(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    yuv_420p_n_to_rgb_or_rgba_row::<BITS, true>(
      y, u_half, v_half, rgba_out, width, matrix, full_range,
    );
  }
}

/// Shared AVX2 high-bit YUV 4:2:0 kernel. `ALPHA = false` uses
/// `write_rgb_32`; `ALPHA = true` uses `write_rgba_32` with constant
/// `0xFF` alpha.
///
/// # Safety
///
/// 1. **AVX2 must be available.**
/// 2. `width & 1 == 0`. 3. slices long enough for `BITS` semantics +
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`. 4. `BITS` ∈
///    `{9, 10, 12, 14}`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn yuv_420p_n_to_rgb_or_rgba_row<const BITS: u32, const ALPHA: bool>(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  const { assert!(BITS == 9 || BITS == 10 || BITS == 12 || BITS == 14) };
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert_eq!(width & 1, 0);
  debug_assert!(y.len() >= width);
  debug_assert!(u_half.len() >= width / 2);
  debug_assert!(v_half.len() >= width / 2);
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<BITS, 8>(full_range);
  let bias = scalar::chroma_bias::<BITS>();
  const RND: i32 = 1 << 14;

  // SAFETY: AVX2 availability is the caller's obligation.
  unsafe {
    let rnd_v = _mm256_set1_epi32(RND);
    let y_off_v = _mm256_set1_epi16(y_off as i16);
    let y_scale_v = _mm256_set1_epi32(y_scale);
    let c_scale_v = _mm256_set1_epi32(c_scale);
    let bias_v = _mm256_set1_epi16(bias as i16);
    let mask_v = _mm256_set1_epi16(scalar::bits_mask::<BITS>() as i16);
    let cru = _mm256_set1_epi32(coeffs.r_u());
    let crv = _mm256_set1_epi32(coeffs.r_v());
    let cgu = _mm256_set1_epi32(coeffs.g_u());
    let cgv = _mm256_set1_epi32(coeffs.g_v());
    let cbu = _mm256_set1_epi32(coeffs.b_u());
    let cbv = _mm256_set1_epi32(coeffs.b_v());
    let alpha_u8 = _mm256_set1_epi8(-1);

    let mut x = 0usize;
    while x + 32 <= width {
      // 32 Y = two `_mm256_loadu_si256` (16 u16 each). U/V each = one
      // load of 16 u16. AND‑mask each load to the low 10 bits — see
      // matching comment in [`crate::row::scalar::yuv_420p_n_to_rgb_row`].
      let y_low_i16 = _mm256_and_si256(_mm256_loadu_si256(y.as_ptr().add(x).cast()), mask_v);
      let y_high_i16 = _mm256_and_si256(_mm256_loadu_si256(y.as_ptr().add(x + 16).cast()), mask_v);
      let u_vec = _mm256_and_si256(
        _mm256_loadu_si256(u_half.as_ptr().add(x / 2).cast()),
        mask_v,
      );
      let v_vec = _mm256_and_si256(
        _mm256_loadu_si256(v_half.as_ptr().add(x / 2).cast()),
        mask_v,
      );

      let u_i16 = _mm256_sub_epi16(u_vec, bias_v);
      let v_i16 = _mm256_sub_epi16(v_vec, bias_v);

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

      if ALPHA {
        write_rgba_32(r_u8, g_u8, b_u8, alpha_u8, out.as_mut_ptr().add(x * 4));
      } else {
        write_rgb_32(r_u8, g_u8, b_u8, out.as_mut_ptr().add(x * 3));
      }

      x += 32;
    }

    if x < width {
      let tail_y = &y[x..width];
      let tail_u = &u_half[x / 2..width / 2];
      let tail_v = &v_half[x / 2..width / 2];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      if ALPHA {
        scalar::yuv_420p_n_to_rgba_row::<BITS>(
          tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range,
        );
      } else {
        scalar::yuv_420p_n_to_rgb_row::<BITS>(
          tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range,
        );
      }
    }
  }
}

/// AVX2 YUV 4:2:0 10‑bit → packed **10‑bit `u16`** RGB.
///
/// Block size 32 Y pixels. Mirrors [`yuv420p10_to_rgb_row`]'s
/// pre‑write math; output uses explicit min/max clamp to `[0, 1023]`
/// (`_mm256_packus_epi16` would clip to u8). Writes are issued via
/// four `write_rgb_u16_8` calls per 32‑pixel block — each extracts a
/// 128‑bit half of the AVX2 `i16x16` channel vectors and hands them
/// to the shared SSE4.1 u16 interleave helper. A 256‑bit AVX2 u16
/// interleave would cut store count in half; left as a follow‑up
/// optimization, since the u16 path is fidelity‑driven rather than
/// throughput‑critical.
///
/// # Numerical contract
///
/// Identical to [`scalar::yuv_420p_n_to_rgb_u16_row::<10>`].
///
/// # Safety
///
/// 1. **AVX2 must be available on the current CPU.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `u_half.len() >= width / 2`,
///    `v_half.len() >= width / 2`, `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn yuv_420p_n_to_rgb_u16_row<const BITS: u32>(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  unsafe {
    yuv_420p_n_to_rgb_or_rgba_u16_row::<BITS, false>(
      y, u_half, v_half, rgb_out, width, matrix, full_range,
    );
  }
}

/// AVX2 sibling of [`yuv_420p_n_to_rgba_row`] for native-depth `u16`
/// output. Alpha samples are `(1 << BITS) - 1` (opaque maximum at the
/// input bit depth).
///
/// # Safety
///
/// Same as [`yuv_420p_n_to_rgb_u16_row`] plus `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn yuv_420p_n_to_rgba_u16_row<const BITS: u32>(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  unsafe {
    yuv_420p_n_to_rgb_or_rgba_u16_row::<BITS, true>(
      y, u_half, v_half, rgba_out, width, matrix, full_range,
    );
  }
}

/// Shared AVX2 high-bit YUV 4:2:0 → native-depth `u16` kernel.
/// `ALPHA = false` writes RGB triples via 4× `write_rgb_u16_8` per
/// 32-pixel block; `ALPHA = true` writes RGBA quads via 4×
/// `write_rgba_u16_8` with constant alpha `(1 << BITS) - 1`.
///
/// # Safety
///
/// 1. **AVX2 must be available on the current CPU.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `u_half.len() >= width / 2`,
///    `v_half.len() >= width / 2`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
/// 4. `BITS` ∈ `{9, 10, 12, 14}`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn yuv_420p_n_to_rgb_or_rgba_u16_row<const BITS: u32, const ALPHA: bool>(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  const { assert!(BITS == 9 || BITS == 10 || BITS == 12 || BITS == 14) };
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert_eq!(width & 1, 0);
  debug_assert!(y.len() >= width);
  debug_assert!(u_half.len() >= width / 2);
  debug_assert!(v_half.len() >= width / 2);
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<BITS, BITS>(full_range);
  let bias = scalar::chroma_bias::<BITS>();
  const RND: i32 = 1 << 14;
  let out_max: i16 = ((1i32 << BITS) - 1) as i16;

  // SAFETY: AVX2 availability is the caller's obligation.
  unsafe {
    let rnd_v = _mm256_set1_epi32(RND);
    let y_off_v = _mm256_set1_epi16(y_off as i16);
    let y_scale_v = _mm256_set1_epi32(y_scale);
    let c_scale_v = _mm256_set1_epi32(c_scale);
    let bias_v = _mm256_set1_epi16(bias as i16);
    let mask_v = _mm256_set1_epi16(scalar::bits_mask::<BITS>() as i16);
    let max_v = _mm256_set1_epi16(out_max);
    let zero_v = _mm256_set1_epi16(0);
    let cru = _mm256_set1_epi32(coeffs.r_u());
    let crv = _mm256_set1_epi32(coeffs.r_v());
    let cgu = _mm256_set1_epi32(coeffs.g_u());
    let cgv = _mm256_set1_epi32(coeffs.g_v());
    let cbu = _mm256_set1_epi32(coeffs.b_u());
    let cbv = _mm256_set1_epi32(coeffs.b_v());
    let alpha_u16 = _mm_set1_epi16(out_max);

    let mut x = 0usize;
    while x + 32 <= width {
      // AND‑mask loads to the low 10 bits so `chroma_i16x16`'s
      // `_mm256_packs_epi32` narrow stays lossless.
      let y_low_i16 = _mm256_and_si256(_mm256_loadu_si256(y.as_ptr().add(x).cast()), mask_v);
      let y_high_i16 = _mm256_and_si256(_mm256_loadu_si256(y.as_ptr().add(x + 16).cast()), mask_v);
      let u_vec = _mm256_and_si256(
        _mm256_loadu_si256(u_half.as_ptr().add(x / 2).cast()),
        mask_v,
      );
      let v_vec = _mm256_and_si256(
        _mm256_loadu_si256(v_half.as_ptr().add(x / 2).cast()),
        mask_v,
      );

      let u_i16 = _mm256_sub_epi16(u_vec, bias_v);
      let v_i16 = _mm256_sub_epi16(v_vec, bias_v);

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

      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v);

      // Per‑channel saturating add + explicit clamp to [0, 1023].
      let r_lo = clamp_u16_max_x16(_mm256_adds_epi16(y_scaled_lo, r_dup_lo), zero_v, max_v);
      let r_hi = clamp_u16_max_x16(_mm256_adds_epi16(y_scaled_hi, r_dup_hi), zero_v, max_v);
      let g_lo = clamp_u16_max_x16(_mm256_adds_epi16(y_scaled_lo, g_dup_lo), zero_v, max_v);
      let g_hi = clamp_u16_max_x16(_mm256_adds_epi16(y_scaled_hi, g_dup_hi), zero_v, max_v);
      let b_lo = clamp_u16_max_x16(_mm256_adds_epi16(y_scaled_lo, b_dup_lo), zero_v, max_v);
      let b_hi = clamp_u16_max_x16(_mm256_adds_epi16(y_scaled_hi, b_dup_hi), zero_v, max_v);

      // Four 8‑pixel u16 writes per 32‑pixel block. Each extracts a
      // 128‑bit half of an i16x16 channel and hands it to the shared
      // SSE4.1 u16 interleave helper.
      if ALPHA {
        let dst = out.as_mut_ptr().add(x * 4);
        write_rgba_u16_8(
          _mm256_castsi256_si128(r_lo),
          _mm256_castsi256_si128(g_lo),
          _mm256_castsi256_si128(b_lo),
          alpha_u16,
          dst,
        );
        write_rgba_u16_8(
          _mm256_extracti128_si256::<1>(r_lo),
          _mm256_extracti128_si256::<1>(g_lo),
          _mm256_extracti128_si256::<1>(b_lo),
          alpha_u16,
          dst.add(32),
        );
        write_rgba_u16_8(
          _mm256_castsi256_si128(r_hi),
          _mm256_castsi256_si128(g_hi),
          _mm256_castsi256_si128(b_hi),
          alpha_u16,
          dst.add(64),
        );
        write_rgba_u16_8(
          _mm256_extracti128_si256::<1>(r_hi),
          _mm256_extracti128_si256::<1>(g_hi),
          _mm256_extracti128_si256::<1>(b_hi),
          alpha_u16,
          dst.add(96),
        );
      } else {
        let dst = out.as_mut_ptr().add(x * 3);
        write_rgb_u16_8(
          _mm256_castsi256_si128(r_lo),
          _mm256_castsi256_si128(g_lo),
          _mm256_castsi256_si128(b_lo),
          dst,
        );
        write_rgb_u16_8(
          _mm256_extracti128_si256::<1>(r_lo),
          _mm256_extracti128_si256::<1>(g_lo),
          _mm256_extracti128_si256::<1>(b_lo),
          dst.add(24),
        );
        write_rgb_u16_8(
          _mm256_castsi256_si128(r_hi),
          _mm256_castsi256_si128(g_hi),
          _mm256_castsi256_si128(b_hi),
          dst.add(48),
        );
        write_rgb_u16_8(
          _mm256_extracti128_si256::<1>(r_hi),
          _mm256_extracti128_si256::<1>(g_hi),
          _mm256_extracti128_si256::<1>(b_hi),
          dst.add(72),
        );
      }

      x += 32;
    }

    if x < width {
      let tail_y = &y[x..width];
      let tail_u = &u_half[x / 2..width / 2];
      let tail_v = &v_half[x / 2..width / 2];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      if ALPHA {
        scalar::yuv_420p_n_to_rgba_u16_row::<BITS>(
          tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range,
        );
      } else {
        scalar::yuv_420p_n_to_rgb_u16_row::<BITS>(
          tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range,
        );
      }
    }
  }
}

/// Clamps an `i16x16` vector to `[0, max]` via AVX2 `_mm256_min_epi16`
/// / `_mm256_max_epi16`. Used by native-depth u16 output paths
/// (10/12/14 bit) where `_mm256_packus_epi16` would incorrectly
/// clip to u8.
#[inline(always)]
fn clamp_u16_max_x16(v: __m256i, zero_v: __m256i, max_v: __m256i) -> __m256i {
  unsafe { _mm256_min_epi16(_mm256_max_epi16(v, zero_v), max_v) }
}

/// AVX2 YUV 4:4:4 planar 9/10/12/14-bit → packed **u8** RGB.
/// Const-generic over `BITS ∈ {9, 10, 12, 14}`. Block size 32 pixels.
///
/// Thin wrapper over [`yuv_444p_n_to_rgb_or_rgba_row`] with `ALPHA = false`.
///
/// # Safety
///
/// 1. **AVX2 must be available on the current CPU.**
/// 2. `y.len() >= width`, `u.len() >= width`, `v.len() >= width`,
///    `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn yuv_444p_n_to_rgb_row<const BITS: u32>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    yuv_444p_n_to_rgb_or_rgba_row::<BITS, false>(y, u, v, rgb_out, width, matrix, full_range);
  }
}

/// AVX2 YUV 4:4:4 planar 9/10/12/14-bit → packed **8-bit RGBA**
/// (`R, G, B, 0xFF`). Same numerical contract as
/// [`yuv_444p_n_to_rgb_row`].
///
/// Thin wrapper over [`yuv_444p_n_to_rgb_or_rgba_row`] with `ALPHA = true`.
///
/// # Safety
///
/// Same as [`yuv_444p_n_to_rgb_row`] but `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn yuv_444p_n_to_rgba_row<const BITS: u32>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    yuv_444p_n_to_rgb_or_rgba_row::<BITS, true>(y, u, v, rgba_out, width, matrix, full_range);
  }
}

/// Shared AVX2 high-bit-depth YUV 4:4:4 kernel for
/// [`yuv_444p_n_to_rgb_row`] (`ALPHA = false`, `write_rgb_32`) and
/// [`yuv_444p_n_to_rgba_row`] (`ALPHA = true`, `write_rgba_32` with
/// constant `0xFF` alpha vector).
///
/// # Safety
///
/// 1. **AVX2 must be available on the current CPU.**
/// 2. `y.len() >= width`, `u.len() >= width`, `v.len() >= width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
/// 3. `BITS` must be one of `{9, 10, 12, 14}`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn yuv_444p_n_to_rgb_or_rgba_row<const BITS: u32, const ALPHA: bool>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  const { assert!(BITS == 9 || BITS == 10 || BITS == 12 || BITS == 14) };
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(y.len() >= width);
  debug_assert!(u.len() >= width);
  debug_assert!(v.len() >= width);
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<BITS, 8>(full_range);
  let bias = scalar::chroma_bias::<BITS>();
  const RND: i32 = 1 << 14;

  unsafe {
    let rnd_v = _mm256_set1_epi32(RND);
    let y_off_v = _mm256_set1_epi16(y_off as i16);
    let y_scale_v = _mm256_set1_epi32(y_scale);
    let c_scale_v = _mm256_set1_epi32(c_scale);
    let bias_v = _mm256_set1_epi16(bias as i16);
    let mask_v = _mm256_set1_epi16(scalar::bits_mask::<BITS>() as i16);
    let cru = _mm256_set1_epi32(coeffs.r_u());
    let crv = _mm256_set1_epi32(coeffs.r_v());
    let cgu = _mm256_set1_epi32(coeffs.g_u());
    let cgv = _mm256_set1_epi32(coeffs.g_v());
    let cbu = _mm256_set1_epi32(coeffs.b_u());
    let cbv = _mm256_set1_epi32(coeffs.b_v());
    let alpha_u8 = _mm256_set1_epi8(-1);

    let mut x = 0usize;
    while x + 32 <= width {
      // 32 Y + 32 U + 32 V per iter. Full-width chroma (two 16-u16
      // loads each) — no horizontal duplication, 4:4:4 is 1:1.
      let y_low_i16 = _mm256_and_si256(_mm256_loadu_si256(y.as_ptr().add(x).cast()), mask_v);
      let y_high_i16 = _mm256_and_si256(_mm256_loadu_si256(y.as_ptr().add(x + 16).cast()), mask_v);
      let u_lo_vec = _mm256_and_si256(_mm256_loadu_si256(u.as_ptr().add(x).cast()), mask_v);
      let u_hi_vec = _mm256_and_si256(_mm256_loadu_si256(u.as_ptr().add(x + 16).cast()), mask_v);
      let v_lo_vec = _mm256_and_si256(_mm256_loadu_si256(v.as_ptr().add(x).cast()), mask_v);
      let v_hi_vec = _mm256_and_si256(_mm256_loadu_si256(v.as_ptr().add(x + 16).cast()), mask_v);

      let u_lo_i16 = _mm256_sub_epi16(u_lo_vec, bias_v);
      let u_hi_i16 = _mm256_sub_epi16(u_hi_vec, bias_v);
      let v_lo_i16 = _mm256_sub_epi16(v_lo_vec, bias_v);
      let v_hi_i16 = _mm256_sub_epi16(v_hi_vec, bias_v);

      let u_lo_a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(u_lo_i16));
      let u_lo_b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(u_lo_i16));
      let u_hi_a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(u_hi_i16));
      let u_hi_b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(u_hi_i16));
      let v_lo_a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v_lo_i16));
      let v_lo_b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(v_lo_i16));
      let v_hi_a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v_hi_i16));
      let v_hi_b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(v_hi_i16));

      let u_d_lo_a = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_lo_a, c_scale_v),
        rnd_v,
      ));
      let u_d_lo_b = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_lo_b, c_scale_v),
        rnd_v,
      ));
      let u_d_hi_a = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_hi_a, c_scale_v),
        rnd_v,
      ));
      let u_d_hi_b = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_hi_b, c_scale_v),
        rnd_v,
      ));
      let v_d_lo_a = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(v_lo_a, c_scale_v),
        rnd_v,
      ));
      let v_d_lo_b = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(v_lo_b, c_scale_v),
        rnd_v,
      ));
      let v_d_hi_a = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(v_hi_a, c_scale_v),
        rnd_v,
      ));
      let v_d_hi_b = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(v_hi_b, c_scale_v),
        rnd_v,
      ));

      // Two chroma_i16x16 per channel produce 32 chroma values.
      let r_chroma_lo = chroma_i16x16(cru, crv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let r_chroma_hi = chroma_i16x16(cru, crv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let g_chroma_lo = chroma_i16x16(cgu, cgv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let g_chroma_hi = chroma_i16x16(cgu, cgv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let b_chroma_lo = chroma_i16x16(cbu, cbv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let b_chroma_hi = chroma_i16x16(cbu, cbv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);

      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v);

      let r_lo = _mm256_adds_epi16(y_scaled_lo, r_chroma_lo);
      let r_hi = _mm256_adds_epi16(y_scaled_hi, r_chroma_hi);
      let g_lo = _mm256_adds_epi16(y_scaled_lo, g_chroma_lo);
      let g_hi = _mm256_adds_epi16(y_scaled_hi, g_chroma_hi);
      let b_lo = _mm256_adds_epi16(y_scaled_lo, b_chroma_lo);
      let b_hi = _mm256_adds_epi16(y_scaled_hi, b_chroma_hi);

      let b_u8 = narrow_u8x32(b_lo, b_hi);
      let g_u8 = narrow_u8x32(g_lo, g_hi);
      let r_u8 = narrow_u8x32(r_lo, r_hi);

      if ALPHA {
        write_rgba_32(r_u8, g_u8, b_u8, alpha_u8, out.as_mut_ptr().add(x * 4));
      } else {
        write_rgb_32(r_u8, g_u8, b_u8, out.as_mut_ptr().add(x * 3));
      }

      x += 32;
    }

    if x < width {
      let tail_y = &y[x..width];
      let tail_u = &u[x..width];
      let tail_v = &v[x..width];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      if ALPHA {
        scalar::yuv_444p_n_to_rgba_row::<BITS>(
          tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range,
        );
      } else {
        scalar::yuv_444p_n_to_rgb_row::<BITS>(
          tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range,
        );
      }
    }
  }
}

/// AVX2 YUV 4:4:4 planar 9/10/12/14-bit → **native-depth u16** RGB.
/// Const-generic over `BITS ∈ {9, 10, 12, 14}`. 32 pixels per iter.
///
/// Thin wrapper over [`yuv_444p_n_to_rgb_or_rgba_u16_row`] with `ALPHA = false`.
///
/// # Safety
///
/// Same as [`yuv_444p_n_to_rgb_row`] but `rgb_out: &mut [u16]`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn yuv_444p_n_to_rgb_u16_row<const BITS: u32>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    yuv_444p_n_to_rgb_or_rgba_u16_row::<BITS, false>(y, u, v, rgb_out, width, matrix, full_range);
  }
}

/// AVX2 sibling of [`yuv_444p_n_to_rgba_row`] for native-depth `u16`
/// output. Alpha samples are `(1 << BITS) - 1` (opaque maximum at the
/// input bit depth).
///
/// Thin wrapper over [`yuv_444p_n_to_rgb_or_rgba_u16_row`] with `ALPHA = true`.
///
/// # Safety
///
/// Same as [`yuv_444p_n_to_rgb_u16_row`] plus `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn yuv_444p_n_to_rgba_u16_row<const BITS: u32>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    yuv_444p_n_to_rgb_or_rgba_u16_row::<BITS, true>(y, u, v, rgba_out, width, matrix, full_range);
  }
}

/// Shared AVX2 high-bit YUV 4:4:4 → native-depth `u16` kernel.
/// `ALPHA = false` writes RGB triples via 4× `write_rgb_u16_8`;
/// `ALPHA = true` writes RGBA quads via 4× `write_rgba_u16_8` with
/// constant alpha `(1 << BITS) - 1`.
///
/// # Safety
///
/// 1. **AVX2 must be available on the current CPU.**
/// 2. `y.len() >= width`, `u.len() >= width`, `v.len() >= width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
/// 3. `BITS` ∈ `{9, 10, 12, 14}`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn yuv_444p_n_to_rgb_or_rgba_u16_row<const BITS: u32, const ALPHA: bool>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  const { assert!(BITS == 9 || BITS == 10 || BITS == 12 || BITS == 14) };
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(y.len() >= width);
  debug_assert!(u.len() >= width);
  debug_assert!(v.len() >= width);
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<BITS, BITS>(full_range);
  let bias = scalar::chroma_bias::<BITS>();
  const RND: i32 = 1 << 14;
  let out_max: i16 = ((1i32 << BITS) - 1) as i16;

  unsafe {
    let rnd_v = _mm256_set1_epi32(RND);
    let y_off_v = _mm256_set1_epi16(y_off as i16);
    let y_scale_v = _mm256_set1_epi32(y_scale);
    let c_scale_v = _mm256_set1_epi32(c_scale);
    let bias_v = _mm256_set1_epi16(bias as i16);
    let mask_v = _mm256_set1_epi16(scalar::bits_mask::<BITS>() as i16);
    let max_v = _mm256_set1_epi16(out_max);
    let zero_v = _mm256_set1_epi16(0);
    let cru = _mm256_set1_epi32(coeffs.r_u());
    let crv = _mm256_set1_epi32(coeffs.r_v());
    let cgu = _mm256_set1_epi32(coeffs.g_u());
    let cgv = _mm256_set1_epi32(coeffs.g_v());
    let cbu = _mm256_set1_epi32(coeffs.b_u());
    let cbv = _mm256_set1_epi32(coeffs.b_v());
    let alpha_u16 = _mm_set1_epi16(out_max);

    let mut x = 0usize;
    while x + 32 <= width {
      let y_low_i16 = _mm256_and_si256(_mm256_loadu_si256(y.as_ptr().add(x).cast()), mask_v);
      let y_high_i16 = _mm256_and_si256(_mm256_loadu_si256(y.as_ptr().add(x + 16).cast()), mask_v);
      let u_lo_vec = _mm256_and_si256(_mm256_loadu_si256(u.as_ptr().add(x).cast()), mask_v);
      let u_hi_vec = _mm256_and_si256(_mm256_loadu_si256(u.as_ptr().add(x + 16).cast()), mask_v);
      let v_lo_vec = _mm256_and_si256(_mm256_loadu_si256(v.as_ptr().add(x).cast()), mask_v);
      let v_hi_vec = _mm256_and_si256(_mm256_loadu_si256(v.as_ptr().add(x + 16).cast()), mask_v);

      let u_lo_i16 = _mm256_sub_epi16(u_lo_vec, bias_v);
      let u_hi_i16 = _mm256_sub_epi16(u_hi_vec, bias_v);
      let v_lo_i16 = _mm256_sub_epi16(v_lo_vec, bias_v);
      let v_hi_i16 = _mm256_sub_epi16(v_hi_vec, bias_v);

      let u_lo_a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(u_lo_i16));
      let u_lo_b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(u_lo_i16));
      let u_hi_a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(u_hi_i16));
      let u_hi_b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(u_hi_i16));
      let v_lo_a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v_lo_i16));
      let v_lo_b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(v_lo_i16));
      let v_hi_a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v_hi_i16));
      let v_hi_b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(v_hi_i16));

      let u_d_lo_a = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_lo_a, c_scale_v),
        rnd_v,
      ));
      let u_d_lo_b = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_lo_b, c_scale_v),
        rnd_v,
      ));
      let u_d_hi_a = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_hi_a, c_scale_v),
        rnd_v,
      ));
      let u_d_hi_b = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_hi_b, c_scale_v),
        rnd_v,
      ));
      let v_d_lo_a = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(v_lo_a, c_scale_v),
        rnd_v,
      ));
      let v_d_lo_b = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(v_lo_b, c_scale_v),
        rnd_v,
      ));
      let v_d_hi_a = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(v_hi_a, c_scale_v),
        rnd_v,
      ));
      let v_d_hi_b = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(v_hi_b, c_scale_v),
        rnd_v,
      ));

      let r_chroma_lo = chroma_i16x16(cru, crv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let r_chroma_hi = chroma_i16x16(cru, crv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let g_chroma_lo = chroma_i16x16(cgu, cgv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let g_chroma_hi = chroma_i16x16(cgu, cgv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let b_chroma_lo = chroma_i16x16(cbu, cbv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let b_chroma_hi = chroma_i16x16(cbu, cbv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);

      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v);

      let r_lo = clamp_u16_max_x16(_mm256_adds_epi16(y_scaled_lo, r_chroma_lo), zero_v, max_v);
      let r_hi = clamp_u16_max_x16(_mm256_adds_epi16(y_scaled_hi, r_chroma_hi), zero_v, max_v);
      let g_lo = clamp_u16_max_x16(_mm256_adds_epi16(y_scaled_lo, g_chroma_lo), zero_v, max_v);
      let g_hi = clamp_u16_max_x16(_mm256_adds_epi16(y_scaled_hi, g_chroma_hi), zero_v, max_v);
      let b_lo = clamp_u16_max_x16(_mm256_adds_epi16(y_scaled_lo, b_chroma_lo), zero_v, max_v);
      let b_hi = clamp_u16_max_x16(_mm256_adds_epi16(y_scaled_hi, b_chroma_hi), zero_v, max_v);

      if ALPHA {
        let dst = out.as_mut_ptr().add(x * 4);
        write_rgba_u16_8(
          _mm256_castsi256_si128(r_lo),
          _mm256_castsi256_si128(g_lo),
          _mm256_castsi256_si128(b_lo),
          alpha_u16,
          dst,
        );
        write_rgba_u16_8(
          _mm256_extracti128_si256::<1>(r_lo),
          _mm256_extracti128_si256::<1>(g_lo),
          _mm256_extracti128_si256::<1>(b_lo),
          alpha_u16,
          dst.add(32),
        );
        write_rgba_u16_8(
          _mm256_castsi256_si128(r_hi),
          _mm256_castsi256_si128(g_hi),
          _mm256_castsi256_si128(b_hi),
          alpha_u16,
          dst.add(64),
        );
        write_rgba_u16_8(
          _mm256_extracti128_si256::<1>(r_hi),
          _mm256_extracti128_si256::<1>(g_hi),
          _mm256_extracti128_si256::<1>(b_hi),
          alpha_u16,
          dst.add(96),
        );
      } else {
        let dst = out.as_mut_ptr().add(x * 3);
        write_rgb_u16_8(
          _mm256_castsi256_si128(r_lo),
          _mm256_castsi256_si128(g_lo),
          _mm256_castsi256_si128(b_lo),
          dst,
        );
        write_rgb_u16_8(
          _mm256_extracti128_si256::<1>(r_lo),
          _mm256_extracti128_si256::<1>(g_lo),
          _mm256_extracti128_si256::<1>(b_lo),
          dst.add(24),
        );
        write_rgb_u16_8(
          _mm256_castsi256_si128(r_hi),
          _mm256_castsi256_si128(g_hi),
          _mm256_castsi256_si128(b_hi),
          dst.add(48),
        );
        write_rgb_u16_8(
          _mm256_extracti128_si256::<1>(r_hi),
          _mm256_extracti128_si256::<1>(g_hi),
          _mm256_extracti128_si256::<1>(b_hi),
          dst.add(72),
        );
      }

      x += 32;
    }

    if x < width {
      let tail_y = &y[x..width];
      let tail_u = &u[x..width];
      let tail_v = &v[x..width];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      if ALPHA {
        scalar::yuv_444p_n_to_rgba_u16_row::<BITS>(
          tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range,
        );
      } else {
        scalar::yuv_444p_n_to_rgb_u16_row::<BITS>(
          tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range,
        );
      }
    }
  }
}

/// AVX2 YUV 4:4:4 planar **16-bit** → packed **u8** RGB. Stays on
/// the i32 Q15 pipeline — output-range scaling keeps `coeff × u_d`
/// within i32 for u8 output. 32 pixels per iter.
///
/// Thin wrapper over [`yuv_444p16_to_rgb_or_rgba_row`] with `ALPHA = false`.
///
/// # Safety
///
/// 1. **AVX2 must be available.**
/// 2. `y.len() >= width`, `u.len() >= width`, `v.len() >= width`,
///    `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn yuv_444p16_to_rgb_row(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    yuv_444p16_to_rgb_or_rgba_row::<false>(y, u, v, rgb_out, width, matrix, full_range);
  }
}

/// AVX2 YUV 4:4:4 planar **16-bit** → packed **8-bit RGBA**
/// (`R, G, B, 0xFF`). Same numerical contract as
/// [`yuv_444p16_to_rgb_row`].
///
/// Thin wrapper over [`yuv_444p16_to_rgb_or_rgba_row`] with `ALPHA = true`.
///
/// # Safety
///
/// Same as [`yuv_444p16_to_rgb_row`] but `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn yuv_444p16_to_rgba_row(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    yuv_444p16_to_rgb_or_rgba_row::<true>(y, u, v, rgba_out, width, matrix, full_range);
  }
}

/// Shared AVX2 16-bit YUV 4:4:4 kernel for [`yuv_444p16_to_rgb_row`]
/// (`ALPHA = false`, `write_rgb_32`) and [`yuv_444p16_to_rgba_row`]
/// (`ALPHA = true`, `write_rgba_32` with constant `0xFF` alpha).
///
/// # Safety
///
/// 1. **AVX2 must be available.**
/// 2. `y.len() >= width`, `u.len() >= width`, `v.len() >= width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn yuv_444p16_to_rgb_or_rgba_row<const ALPHA: bool>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(y.len() >= width);
  debug_assert!(u.len() >= width);
  debug_assert!(v.len() >= width);
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<16, 8>(full_range);
  const RND: i32 = 1 << 14;

  unsafe {
    let rnd_v = _mm256_set1_epi32(RND);
    let y_off_v = _mm256_set1_epi32(y_off);
    let y_scale_v = _mm256_set1_epi32(y_scale);
    let c_scale_v = _mm256_set1_epi32(c_scale);
    let bias16_v = _mm256_set1_epi16(-32768i16);
    let cru = _mm256_set1_epi32(coeffs.r_u());
    let crv = _mm256_set1_epi32(coeffs.r_v());
    let cgu = _mm256_set1_epi32(coeffs.g_u());
    let cgv = _mm256_set1_epi32(coeffs.g_v());
    let cbu = _mm256_set1_epi32(coeffs.b_u());
    let cbv = _mm256_set1_epi32(coeffs.b_v());
    let alpha_u8 = _mm256_set1_epi8(-1);

    let mut x = 0usize;
    while x + 32 <= width {
      let y_low = _mm256_loadu_si256(y.as_ptr().add(x).cast());
      let y_high = _mm256_loadu_si256(y.as_ptr().add(x + 16).cast());
      let u_lo_vec = _mm256_loadu_si256(u.as_ptr().add(x).cast());
      let u_hi_vec = _mm256_loadu_si256(u.as_ptr().add(x + 16).cast());
      let v_lo_vec = _mm256_loadu_si256(v.as_ptr().add(x).cast());
      let v_hi_vec = _mm256_loadu_si256(v.as_ptr().add(x + 16).cast());

      let u_lo_i16 = _mm256_sub_epi16(u_lo_vec, bias16_v);
      let u_hi_i16 = _mm256_sub_epi16(u_hi_vec, bias16_v);
      let v_lo_i16 = _mm256_sub_epi16(v_lo_vec, bias16_v);
      let v_hi_i16 = _mm256_sub_epi16(v_hi_vec, bias16_v);

      let u_lo_a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(u_lo_i16));
      let u_lo_b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(u_lo_i16));
      let u_hi_a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(u_hi_i16));
      let u_hi_b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(u_hi_i16));
      let v_lo_a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v_lo_i16));
      let v_lo_b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(v_lo_i16));
      let v_hi_a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v_hi_i16));
      let v_hi_b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(v_hi_i16));

      let u_d_lo_a = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_lo_a, c_scale_v),
        rnd_v,
      ));
      let u_d_lo_b = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_lo_b, c_scale_v),
        rnd_v,
      ));
      let u_d_hi_a = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_hi_a, c_scale_v),
        rnd_v,
      ));
      let u_d_hi_b = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_hi_b, c_scale_v),
        rnd_v,
      ));
      let v_d_lo_a = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(v_lo_a, c_scale_v),
        rnd_v,
      ));
      let v_d_lo_b = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(v_lo_b, c_scale_v),
        rnd_v,
      ));
      let v_d_hi_a = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(v_hi_a, c_scale_v),
        rnd_v,
      ));
      let v_d_hi_b = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(v_hi_b, c_scale_v),
        rnd_v,
      ));

      let r_chroma_lo = chroma_i16x16(cru, crv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let r_chroma_hi = chroma_i16x16(cru, crv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let g_chroma_lo = chroma_i16x16(cgu, cgv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let g_chroma_hi = chroma_i16x16(cgu, cgv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let b_chroma_lo = chroma_i16x16(cbu, cbv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let b_chroma_hi = chroma_i16x16(cbu, cbv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);

      let y_scaled_lo = scale_y_u16_avx2(y_low, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y_u16_avx2(y_high, y_off_v, y_scale_v, rnd_v);

      let r_lo = _mm256_adds_epi16(y_scaled_lo, r_chroma_lo);
      let r_hi = _mm256_adds_epi16(y_scaled_hi, r_chroma_hi);
      let g_lo = _mm256_adds_epi16(y_scaled_lo, g_chroma_lo);
      let g_hi = _mm256_adds_epi16(y_scaled_hi, g_chroma_hi);
      let b_lo = _mm256_adds_epi16(y_scaled_lo, b_chroma_lo);
      let b_hi = _mm256_adds_epi16(y_scaled_hi, b_chroma_hi);

      let r_u8 = narrow_u8x32(r_lo, r_hi);
      let g_u8 = narrow_u8x32(g_lo, g_hi);
      let b_u8 = narrow_u8x32(b_lo, b_hi);

      if ALPHA {
        write_rgba_32(r_u8, g_u8, b_u8, alpha_u8, out.as_mut_ptr().add(x * 4));
      } else {
        write_rgb_32(r_u8, g_u8, b_u8, out.as_mut_ptr().add(x * 3));
      }
      x += 32;
    }

    if x < width {
      let tail_y = &y[x..width];
      let tail_u = &u[x..width];
      let tail_v = &v[x..width];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      if ALPHA {
        scalar::yuv_444p16_to_rgba_row(
          tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range,
        );
      } else {
        scalar::yuv_444p16_to_rgb_row(tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range);
      }
    }
  }
}

/// AVX2 YUV 4:4:4 planar **16-bit** → packed **u16** RGB. Native
/// 256-bit kernel using the [`srai64_15_x4`] bias trick (AVX2 lacks
/// `_mm256_srai_epi64`). Processes 16 pixels per iteration — 2× the
/// SSE4.1 rate, and no chroma-duplication step since 4:4:4 chroma
/// is 1:1 with Y.
///
/// Thin wrapper over [`yuv_444p16_to_rgb_or_rgba_u16_row`] with `ALPHA = false`.
///
/// # Safety
///
/// Same as [`yuv_444p16_to_rgb_row`] but `rgb_out: &mut [u16]`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn yuv_444p16_to_rgb_u16_row(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    yuv_444p16_to_rgb_or_rgba_u16_row::<false>(y, u, v, rgb_out, width, matrix, full_range);
  }
}

/// AVX2 sibling of [`yuv_444p16_to_rgba_row`] for native-depth `u16`
/// output. Alpha samples are `0xFFFF`.
///
/// Thin wrapper over [`yuv_444p16_to_rgb_or_rgba_u16_row`] with `ALPHA = true`.
///
/// # Safety
///
/// Same as [`yuv_444p16_to_rgb_u16_row`] plus `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn yuv_444p16_to_rgba_u16_row(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    yuv_444p16_to_rgb_or_rgba_u16_row::<true>(y, u, v, rgba_out, width, matrix, full_range);
  }
}

/// Shared AVX2 16-bit YUV 4:4:4 → native-depth `u16` kernel.
/// `ALPHA = false` writes RGB triples; `ALPHA = true` writes RGBA
/// quads with constant alpha `0xFFFF`.
///
/// # Safety
///
/// 1. **AVX2 must be available on the current CPU.**
/// 2. `y.len() >= width`, `u.len() >= width`, `v.len() >= width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn yuv_444p16_to_rgb_or_rgba_u16_row<const ALPHA: bool>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(y.len() >= width);
  debug_assert!(u.len() >= width);
  debug_assert!(v.len() >= width);
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<16, 16>(full_range);
  const RND: i64 = 1 << 14;

  unsafe {
    let alpha_u16 = _mm_set1_epi16(-1i16);
    let rnd_v = _mm256_set1_epi64x(RND);
    let y_off_v = _mm256_set1_epi32(y_off);
    let y_scale_v = _mm256_set1_epi32(y_scale);
    let c_scale_v = _mm256_set1_epi32(c_scale);
    let bias16_v = _mm256_set1_epi16(-32768i16);
    let cru = _mm256_set1_epi32(coeffs.r_u());
    let crv = _mm256_set1_epi32(coeffs.r_v());
    let cgu = _mm256_set1_epi32(coeffs.g_u());
    let cgv = _mm256_set1_epi32(coeffs.g_v());
    let cbu = _mm256_set1_epi32(coeffs.b_u());
    let cbv = _mm256_set1_epi32(coeffs.b_v());

    let mut x = 0usize;
    while x + 16 <= width {
      // 16 Y + 16 U + 16 V per iter — full-width chroma, no dup.
      let y_vec = _mm256_loadu_si256(y.as_ptr().add(x).cast());
      let u_vec = _mm256_loadu_si256(u.as_ptr().add(x).cast());
      let v_vec = _mm256_loadu_si256(v.as_ptr().add(x).cast());

      let u_i16 = _mm256_sub_epi16(u_vec, bias16_v);
      let v_i16 = _mm256_sub_epi16(v_vec, bias16_v);

      // Widen each i16x16 → two i32x8 halves.
      let u_lo_i32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(u_i16));
      let u_hi_i32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(u_i16));
      let v_lo_i32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v_i16));
      let v_hi_i32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(v_i16));

      let rnd32_v = _mm256_set1_epi32(1 << 14);
      let u_d_lo = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_lo_i32, c_scale_v),
        rnd32_v,
      ));
      let u_d_hi = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_hi_i32, c_scale_v),
        rnd32_v,
      ));
      let v_d_lo = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(v_lo_i32, c_scale_v),
        rnd32_v,
      ));
      let v_d_hi = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(v_hi_i32, c_scale_v),
        rnd32_v,
      ));

      // i64 chroma: 4 calls per channel (lo even/odd + hi even/odd).
      let u_d_lo_odd = _mm256_shuffle_epi32::<0xF5>(u_d_lo);
      let u_d_hi_odd = _mm256_shuffle_epi32::<0xF5>(u_d_hi);
      let v_d_lo_odd = _mm256_shuffle_epi32::<0xF5>(v_d_lo);
      let v_d_hi_odd = _mm256_shuffle_epi32::<0xF5>(v_d_hi);

      let r_ch_lo_e = chroma_i64x4_avx2(cru, crv, u_d_lo, v_d_lo, rnd_v);
      let r_ch_lo_o = chroma_i64x4_avx2(cru, crv, u_d_lo_odd, v_d_lo_odd, rnd_v);
      let r_ch_hi_e = chroma_i64x4_avx2(cru, crv, u_d_hi, v_d_hi, rnd_v);
      let r_ch_hi_o = chroma_i64x4_avx2(cru, crv, u_d_hi_odd, v_d_hi_odd, rnd_v);
      let g_ch_lo_e = chroma_i64x4_avx2(cgu, cgv, u_d_lo, v_d_lo, rnd_v);
      let g_ch_lo_o = chroma_i64x4_avx2(cgu, cgv, u_d_lo_odd, v_d_lo_odd, rnd_v);
      let g_ch_hi_e = chroma_i64x4_avx2(cgu, cgv, u_d_hi, v_d_hi, rnd_v);
      let g_ch_hi_o = chroma_i64x4_avx2(cgu, cgv, u_d_hi_odd, v_d_hi_odd, rnd_v);
      let b_ch_lo_e = chroma_i64x4_avx2(cbu, cbv, u_d_lo, v_d_lo, rnd_v);
      let b_ch_lo_o = chroma_i64x4_avx2(cbu, cbv, u_d_lo_odd, v_d_lo_odd, rnd_v);
      let b_ch_hi_e = chroma_i64x4_avx2(cbu, cbv, u_d_hi, v_d_hi, rnd_v);
      let b_ch_hi_o = chroma_i64x4_avx2(cbu, cbv, u_d_hi_odd, v_d_hi_odd, rnd_v);

      // Reassemble → i32x8 per half.
      let r_ch_lo = reassemble_i64x4_to_i32x8(r_ch_lo_e, r_ch_lo_o);
      let r_ch_hi = reassemble_i64x4_to_i32x8(r_ch_hi_e, r_ch_hi_o);
      let g_ch_lo = reassemble_i64x4_to_i32x8(g_ch_lo_e, g_ch_lo_o);
      let g_ch_hi = reassemble_i64x4_to_i32x8(g_ch_hi_e, g_ch_hi_o);
      let b_ch_lo = reassemble_i64x4_to_i32x8(b_ch_lo_e, b_ch_lo_o);
      let b_ch_hi = reassemble_i64x4_to_i32x8(b_ch_hi_e, b_ch_hi_o);

      // Y scaled in i64, two i32x8 halves.
      let y_lo_u16 = _mm256_castsi256_si128(y_vec);
      let y_hi_u16 = _mm256_extracti128_si256::<1>(y_vec);
      let y_lo_i32 = _mm256_sub_epi32(_mm256_cvtepu16_epi32(y_lo_u16), y_off_v);
      let y_hi_i32 = _mm256_sub_epi32(_mm256_cvtepu16_epi32(y_hi_u16), y_off_v);

      let y_lo_scaled = scale_y_i32x8_i64(y_lo_i32, y_scale_v, rnd_v);
      let y_hi_scaled = scale_y_i32x8_i64(y_hi_i32, y_scale_v, rnd_v);

      // Add Y + chroma (no dup — 4:4:4 is 1:1), saturate to u16.
      let r_u16 = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi32(
        _mm256_add_epi32(y_lo_scaled, r_ch_lo),
        _mm256_add_epi32(y_hi_scaled, r_ch_hi),
      ));
      let g_u16 = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi32(
        _mm256_add_epi32(y_lo_scaled, g_ch_lo),
        _mm256_add_epi32(y_hi_scaled, g_ch_hi),
      ));
      let b_u16 = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi32(
        _mm256_add_epi32(y_lo_scaled, b_ch_lo),
        _mm256_add_epi32(y_hi_scaled, b_ch_hi),
      ));

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

    if x < width {
      let tail_y = &y[x..width];
      let tail_u = &u[x..width];
      let tail_v = &v[x..width];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      if ALPHA {
        scalar::yuv_444p16_to_rgba_u16_row(
          tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range,
        );
      } else {
        scalar::yuv_444p16_to_rgb_u16_row(
          tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range,
        );
      }
    }
  }
}

/// AVX2 high‑bit‑packed semi‑planar (`BITS` ∈ {10, 12}) → packed
/// **8‑bit** RGB.
///
/// Block size 32 Y pixels / 16 chroma pairs per iteration. Mirrors
/// [`super::x86_avx2::yuv_420p_n_to_rgb_row`] with two structural
/// differences:
/// - Samples are shifted right by `16 - BITS` (`_mm256_srl_epi16`,
///   with a shift count computed from `BITS` once per call) instead
///   of AND‑masked.
/// - Semi‑planar UV is deinterleaved via [`deinterleave_uv_u16_avx2`]
///   (two `_mm256_shuffle_epi8` + two `_mm256_permute4x64_epi64` +
///   two `_mm256_permute2x128_si256` per 32 chroma elements).
///
/// # Numerical contract
///
/// Byte‑identical to [`scalar::p_n_to_rgb_row::<BITS>`] for the
/// monomorphized `BITS`.
///
/// # Safety
///
/// 1. **AVX2 must be available on the current CPU.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`,
///    `rgb_out.len() >= 3 * width`.
///
/// Thin wrapper over [`p_n_to_rgb_or_rgba_row`] with `ALPHA = false`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn p_n_to_rgb_row<const BITS: u32>(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    p_n_to_rgb_or_rgba_row::<BITS, false>(y, uv_half, rgb_out, width, matrix, full_range);
  }
}

/// AVX2 high-bit-packed semi-planar 4:2:0 → packed **8-bit RGBA**
/// (`R, G, B, 0xFF`).
///
/// Thin wrapper over [`p_n_to_rgb_or_rgba_row`] with `ALPHA = true`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn p_n_to_rgba_row<const BITS: u32>(
  y: &[u16],
  uv_half: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    p_n_to_rgb_or_rgba_row::<BITS, true>(y, uv_half, rgba_out, width, matrix, full_range);
  }
}

/// Shared AVX2 P010/P012 kernel. `ALPHA = false` uses `write_rgb_32`;
/// `ALPHA = true` uses `write_rgba_32` with constant `0xFF` alpha.
///
/// # Safety
///
/// 1. **AVX2 must be available.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
/// 4. `BITS` ∈ `{10, 12}`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn p_n_to_rgb_or_rgba_row<const BITS: u32, const ALPHA: bool>(
  y: &[u16],
  uv_half: &[u16],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  const { assert!(BITS == 10 || BITS == 12) };
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert_eq!(width & 1, 0);
  debug_assert!(y.len() >= width);
  debug_assert!(uv_half.len() >= width);
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<BITS, 8>(full_range);
  let bias = scalar::chroma_bias::<BITS>();
  const RND: i32 = 1 << 14;

  // SAFETY: AVX2 availability is the caller's obligation.
  unsafe {
    let rnd_v = _mm256_set1_epi32(RND);
    let y_off_v = _mm256_set1_epi16(y_off as i16);
    let y_scale_v = _mm256_set1_epi32(y_scale);
    let c_scale_v = _mm256_set1_epi32(c_scale);
    let bias_v = _mm256_set1_epi16(bias as i16);
    // High-bit-packed samples: shift right by `16 - BITS`.
    let shr_count = _mm_cvtsi32_si128((16 - BITS) as i32);
    let cru = _mm256_set1_epi32(coeffs.r_u());
    let crv = _mm256_set1_epi32(coeffs.r_v());
    let cgu = _mm256_set1_epi32(coeffs.g_u());
    let cgv = _mm256_set1_epi32(coeffs.g_v());
    let cbu = _mm256_set1_epi32(coeffs.b_u());
    let cbv = _mm256_set1_epi32(coeffs.b_v());
    let alpha_u8 = _mm256_set1_epi8(-1);

    let mut x = 0usize;
    while x + 32 <= width {
      // 32 Y = two u16×16 loads, shifted right by `16 - BITS`.
      let y_low_i16 = _mm256_srl_epi16(_mm256_loadu_si256(y.as_ptr().add(x).cast()), shr_count);
      let y_high_i16 =
        _mm256_srl_epi16(_mm256_loadu_si256(y.as_ptr().add(x + 16).cast()), shr_count);

      // 32 UV (16 pairs) — deinterleave + shift.
      let (u_vec, v_vec) = deinterleave_uv_u16_avx2(uv_half.as_ptr().add(x));
      let u_vec = _mm256_srl_epi16(u_vec, shr_count);
      let v_vec = _mm256_srl_epi16(v_vec, shr_count);

      let u_i16 = _mm256_sub_epi16(u_vec, bias_v);
      let v_i16 = _mm256_sub_epi16(v_vec, bias_v);

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

      if ALPHA {
        write_rgba_32(r_u8, g_u8, b_u8, alpha_u8, out.as_mut_ptr().add(x * 4));
      } else {
        write_rgb_32(r_u8, g_u8, b_u8, out.as_mut_ptr().add(x * 3));
      }

      x += 32;
    }

    if x < width {
      let tail_y = &y[x..width];
      let tail_uv = &uv_half[x..width];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      if ALPHA {
        scalar::p_n_to_rgba_row::<BITS>(tail_y, tail_uv, tail_out, tail_w, matrix, full_range);
      } else {
        scalar::p_n_to_rgb_row::<BITS>(tail_y, tail_uv, tail_out, tail_w, matrix, full_range);
      }
    }
  }
}

/// AVX2 high‑bit‑packed semi‑planar (`BITS` ∈ {10, 12}) → packed
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
/// 1. **AVX2 must be available on the current CPU.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`,
///    `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn p_n_to_rgb_u16_row<const BITS: u32>(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  unsafe {
    p_n_to_rgb_or_rgba_u16_row::<BITS, false>(y, uv_half, rgb_out, width, matrix, full_range);
  }
}

/// AVX2 sibling of [`p_n_to_rgba_row`] for native-depth `u16` output.
/// Alpha samples are `(1 << BITS) - 1` (opaque maximum at the input
/// bit depth). P016 has its own kernel family — never routed here.
///
/// # Safety
///
/// Same as [`p_n_to_rgb_u16_row`] plus `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn p_n_to_rgba_u16_row<const BITS: u32>(
  y: &[u16],
  uv_half: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  unsafe {
    p_n_to_rgb_or_rgba_u16_row::<BITS, true>(y, uv_half, rgba_out, width, matrix, full_range);
  }
}

/// Shared AVX2 Pn → native-depth `u16` kernel. `ALPHA = false` writes
/// RGB triples via 4× `write_rgb_u16_8` per 32-pixel block;
/// `ALPHA = true` writes RGBA quads via 4× `write_rgba_u16_8` with
/// constant alpha `(1 << BITS) - 1`. P016 has its own kernel family —
/// never routed here.
///
/// # Safety
///
/// 1. **AVX2 must be available on the current CPU.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
/// 4. `BITS` ∈ `{10, 12}`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn p_n_to_rgb_or_rgba_u16_row<const BITS: u32, const ALPHA: bool>(
  y: &[u16],
  uv_half: &[u16],
  out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  const { assert!(BITS == 10 || BITS == 12) };
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert_eq!(width & 1, 0);
  debug_assert!(y.len() >= width);
  debug_assert!(uv_half.len() >= width);
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<BITS, BITS>(full_range);
  let bias = scalar::chroma_bias::<BITS>();
  const RND: i32 = 1 << 14;
  let out_max: i16 = ((1i32 << BITS) - 1) as i16;

  // SAFETY: AVX2 availability is the caller's obligation.
  unsafe {
    let rnd_v = _mm256_set1_epi32(RND);
    let y_off_v = _mm256_set1_epi16(y_off as i16);
    let y_scale_v = _mm256_set1_epi32(y_scale);
    let c_scale_v = _mm256_set1_epi32(c_scale);
    let bias_v = _mm256_set1_epi16(bias as i16);
    let max_v = _mm256_set1_epi16(out_max);
    let zero_v = _mm256_set1_epi16(0);
    // High-bit-packed samples: shift right by `16 - BITS`.
    let shr_count = _mm_cvtsi32_si128((16 - BITS) as i32);
    let cru = _mm256_set1_epi32(coeffs.r_u());
    let crv = _mm256_set1_epi32(coeffs.r_v());
    let cgu = _mm256_set1_epi32(coeffs.g_u());
    let cgv = _mm256_set1_epi32(coeffs.g_v());
    let cbu = _mm256_set1_epi32(coeffs.b_u());
    let cbv = _mm256_set1_epi32(coeffs.b_v());
    let alpha_u16 = _mm_set1_epi16(out_max);

    let mut x = 0usize;
    while x + 32 <= width {
      let y_low_i16 = _mm256_srl_epi16(_mm256_loadu_si256(y.as_ptr().add(x).cast()), shr_count);
      let y_high_i16 =
        _mm256_srl_epi16(_mm256_loadu_si256(y.as_ptr().add(x + 16).cast()), shr_count);
      let (u_vec, v_vec) = deinterleave_uv_u16_avx2(uv_half.as_ptr().add(x));
      let u_vec = _mm256_srl_epi16(u_vec, shr_count);
      let v_vec = _mm256_srl_epi16(v_vec, shr_count);

      let u_i16 = _mm256_sub_epi16(u_vec, bias_v);
      let v_i16 = _mm256_sub_epi16(v_vec, bias_v);

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

      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v);

      let r_lo = clamp_u16_max_x16(_mm256_adds_epi16(y_scaled_lo, r_dup_lo), zero_v, max_v);
      let r_hi = clamp_u16_max_x16(_mm256_adds_epi16(y_scaled_hi, r_dup_hi), zero_v, max_v);
      let g_lo = clamp_u16_max_x16(_mm256_adds_epi16(y_scaled_lo, g_dup_lo), zero_v, max_v);
      let g_hi = clamp_u16_max_x16(_mm256_adds_epi16(y_scaled_hi, g_dup_hi), zero_v, max_v);
      let b_lo = clamp_u16_max_x16(_mm256_adds_epi16(y_scaled_lo, b_dup_lo), zero_v, max_v);
      let b_hi = clamp_u16_max_x16(_mm256_adds_epi16(y_scaled_hi, b_dup_hi), zero_v, max_v);

      if ALPHA {
        let dst = out.as_mut_ptr().add(x * 4);
        write_rgba_u16_8(
          _mm256_castsi256_si128(r_lo),
          _mm256_castsi256_si128(g_lo),
          _mm256_castsi256_si128(b_lo),
          alpha_u16,
          dst,
        );
        write_rgba_u16_8(
          _mm256_extracti128_si256::<1>(r_lo),
          _mm256_extracti128_si256::<1>(g_lo),
          _mm256_extracti128_si256::<1>(b_lo),
          alpha_u16,
          dst.add(32),
        );
        write_rgba_u16_8(
          _mm256_castsi256_si128(r_hi),
          _mm256_castsi256_si128(g_hi),
          _mm256_castsi256_si128(b_hi),
          alpha_u16,
          dst.add(64),
        );
        write_rgba_u16_8(
          _mm256_extracti128_si256::<1>(r_hi),
          _mm256_extracti128_si256::<1>(g_hi),
          _mm256_extracti128_si256::<1>(b_hi),
          alpha_u16,
          dst.add(96),
        );
      } else {
        let dst = out.as_mut_ptr().add(x * 3);
        write_rgb_u16_8(
          _mm256_castsi256_si128(r_lo),
          _mm256_castsi256_si128(g_lo),
          _mm256_castsi256_si128(b_lo),
          dst,
        );
        write_rgb_u16_8(
          _mm256_extracti128_si256::<1>(r_lo),
          _mm256_extracti128_si256::<1>(g_lo),
          _mm256_extracti128_si256::<1>(b_lo),
          dst.add(24),
        );
        write_rgb_u16_8(
          _mm256_castsi256_si128(r_hi),
          _mm256_castsi256_si128(g_hi),
          _mm256_castsi256_si128(b_hi),
          dst.add(48),
        );
        write_rgb_u16_8(
          _mm256_extracti128_si256::<1>(r_hi),
          _mm256_extracti128_si256::<1>(g_hi),
          _mm256_extracti128_si256::<1>(b_hi),
          dst.add(72),
        );
      }

      x += 32;
    }

    if x < width {
      let tail_y = &y[x..width];
      let tail_uv = &uv_half[x..width];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      if ALPHA {
        scalar::p_n_to_rgba_u16_row::<BITS>(tail_y, tail_uv, tail_out, tail_w, matrix, full_range);
      } else {
        scalar::p_n_to_rgb_u16_row::<BITS>(tail_y, tail_uv, tail_out, tail_w, matrix, full_range);
      }
    }
  }
}

/// Deinterleaves 32 `u16` elements at `ptr` (`[U0, V0, U1, V1, …,
/// U15, V15]`) into `(u_vec, v_vec)` — two AVX2 vectors each holding
/// 16 packed `u16` samples.
///
/// Uses per‑lane `_mm256_shuffle_epi8` to pack each 128‑bit lane's
/// U/V samples into the low/high 64 bits, then
/// `_mm256_permute4x64_epi64::<0xD8>` to move the two U halves
/// together (low 128) and the two V halves together (high 128) within
/// each source vector, and finally `_mm256_permute2x128_si256` to
/// combine the four U halves and the four V halves across the two
/// vectors. 2 loads + 2 shuffles + 2 per-vector permutes + 2 cross-
/// vector permutes = 8 ops.
///
/// # Safety
///
/// `ptr` must point to at least 64 readable bytes (32 `u16`
/// elements). Caller's `target_feature` must include AVX2.
#[inline(always)]
unsafe fn deinterleave_uv_u16_avx2(ptr: *const u16) -> (__m256i, __m256i) {
  unsafe {
    // Per‑lane byte mask: within each 128‑bit lane, pack even u16s
    // (U's) into low 8 bytes, odd u16s (V's) into high 8 bytes.
    let split_mask = _mm256_setr_epi8(
      0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15, // low lane
      0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15, // high lane
    );

    let uv0 = _mm256_loadu_si256(ptr.cast());
    let uv1 = _mm256_loadu_si256(ptr.add(16).cast());

    // After per‑lane shuffle: each vector is
    // `[U_lane0_lo, V_lane0_lo, U_lane1_lo, V_lane1_lo]` in 64‑bit
    // chunks.
    let s0 = _mm256_shuffle_epi8(uv0, split_mask);
    let s1 = _mm256_shuffle_epi8(uv1, split_mask);

    // Permute 4×64 within each vector to get [U0..U7, V0..V7] and
    // [U8..U15, V8..V15]. Mask 0xD8 = (3,1,2,0) → picks 64-bit
    // chunks 0, 2, 1, 3 from the source, rearranging
    // [A, B, C, D] → [A, C, B, D].
    let s0_p = _mm256_permute4x64_epi64::<0xD8>(s0);
    let s1_p = _mm256_permute4x64_epi64::<0xD8>(s1);

    // Cross-vector permute: low 128 of s0_p + low 128 of s1_p → U's;
    // high 128 of s0_p + high 128 of s1_p → V's.
    let u_vec = _mm256_permute2x128_si256::<0x20>(s0_p, s1_p);
    let v_vec = _mm256_permute2x128_si256::<0x31>(s0_p, s1_p);
    (u_vec, v_vec)
  }
}

/// AVX2 NV12 → packed RGB. Thin wrapper over
/// [`nv12_or_nv21_to_rgb_or_rgba_row_impl`] with
/// `SWAP_UV = false, ALPHA = false`.
///
/// # Safety
///
/// Same contract as [`nv12_or_nv21_to_rgb_or_rgba_row_impl`]:
///
/// 1. **AVX2 must be available on the current CPU.** Direct callers
///    are responsible for verifying this; the dispatcher in
///    [`crate::row::nv12_to_rgb_row`] checks it.
/// 2. `width & 1 == 0` (4:2:0 requires even width).
/// 3. `y.len() >= width`.
/// 4. `uv_half.len() >= width` (interleaved UV bytes, 2 per chroma pair).
/// 5. `rgb_out.len() >= 3 * width`.
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
  unsafe {
    nv12_or_nv21_to_rgb_or_rgba_row_impl::<false, false>(
      y, uv_half, rgb_out, width, matrix, full_range,
    );
  }
}

/// AVX2 NV21 → packed RGB. Thin wrapper over
/// [`nv12_or_nv21_to_rgb_or_rgba_row_impl`] with
/// `SWAP_UV = true, ALPHA = false`.
///
/// # Safety
///
/// Same contract as [`nv12_to_rgb_row`]; `vu_half` carries the same
/// number of bytes (`>= width`) but in V-then-U order per chroma
/// pair.
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
  unsafe {
    nv12_or_nv21_to_rgb_or_rgba_row_impl::<true, false>(
      y, vu_half, rgb_out, width, matrix, full_range,
    );
  }
}

/// AVX2 NV12 → packed RGBA. Same contract as [`nv12_to_rgb_row`]
/// but writes 4 bytes per pixel via [`write_rgba_32`].
/// `rgba_out.len() >= 4 * width`.
///
/// # Safety
///
/// Same as [`nv12_to_rgb_row`] except the output slice must be
/// `>= 4 * width` bytes (one extra byte per pixel for the opaque
/// alpha).
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn nv12_to_rgba_row(
  y: &[u8],
  uv_half: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  unsafe {
    nv12_or_nv21_to_rgb_or_rgba_row_impl::<false, true>(
      y, uv_half, rgba_out, width, matrix, full_range,
    );
  }
}

/// AVX2 NV21 → packed RGBA. Same contract as [`nv21_to_rgb_row`]
/// but writes 4 bytes per pixel via [`write_rgba_32`].
/// `rgba_out.len() >= 4 * width`.
///
/// # Safety
///
/// Same as [`nv21_to_rgb_row`] except the output slice must be
/// `>= 4 * width` bytes.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn nv21_to_rgba_row(
  y: &[u8],
  vu_half: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  unsafe {
    nv12_or_nv21_to_rgb_or_rgba_row_impl::<true, true>(
      y, vu_half, rgba_out, width, matrix, full_range,
    );
  }
}

/// Shared AVX2 NV12/NV21 kernel at 3 bpp (RGB) or 4 bpp + opaque
/// alpha (RGBA). `SWAP_UV` selects chroma byte order; `ALPHA = true`
/// writes via [`write_rgba_32`], `ALPHA = false` via [`write_rgb_32`].
/// Both const generics drive compile-time monomorphization.
///
/// # Safety
///
/// 1. **AVX2 must be available on the current CPU.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`.
/// 4. `uv_or_vu_half.len() >= width` (32 interleaved bytes per 32 Y pixels).
/// 5. `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn nv12_or_nv21_to_rgb_or_rgba_row_impl<const SWAP_UV: bool, const ALPHA: bool>(
  y: &[u8],
  uv_or_vu_half: &[u8],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0, "NV12/NV21 require even width");
  debug_assert!(y.len() >= width);
  debug_assert!(uv_or_vu_half.len() >= width);
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp);

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
    let alpha_u8 = _mm256_set1_epi8(-1); // 0xFF as i8

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

      if ALPHA {
        write_rgba_32(r_u8, g_u8, b_u8, alpha_u8, out.as_mut_ptr().add(x * 4));
      } else {
        write_rgb_32(r_u8, g_u8, b_u8, out.as_mut_ptr().add(x * 3));
      }

      x += 32;
    }

    if x < width {
      let tail_y = &y[x..width];
      let tail_uv = &uv_or_vu_half[x..width];
      let tail_w = width - x;
      let tail_out = &mut out[x * bpp..width * bpp];
      match (SWAP_UV, ALPHA) {
        (false, false) => {
          scalar::nv12_to_rgb_row(tail_y, tail_uv, tail_out, tail_w, matrix, full_range)
        }
        (true, false) => {
          scalar::nv21_to_rgb_row(tail_y, tail_uv, tail_out, tail_w, matrix, full_range)
        }
        (false, true) => {
          scalar::nv12_to_rgba_row(tail_y, tail_uv, tail_out, tail_w, matrix, full_range)
        }
        (true, true) => {
          scalar::nv21_to_rgba_row(tail_y, tail_uv, tail_out, tail_w, matrix, full_range)
        }
      }
    }
  }
}

/// AVX2 NV24 → packed RGB (UV-ordered, 4:4:4).
///
/// # Safety
///
/// Same as [`nv24_or_nv42_to_rgb_or_rgba_row_impl`].
#[inline]
#[target_feature(enable = "avx2")]
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
    nv24_or_nv42_to_rgb_or_rgba_row_impl::<false, false>(y, uv, rgb_out, width, matrix, full_range);
  }
}

/// AVX2 NV42 → packed RGB (VU-ordered, 4:4:4).
///
/// # Safety
///
/// Same as [`nv24_or_nv42_to_rgb_or_rgba_row_impl`].
#[inline]
#[target_feature(enable = "avx2")]
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
    nv24_or_nv42_to_rgb_or_rgba_row_impl::<true, false>(y, vu, rgb_out, width, matrix, full_range);
  }
}

/// AVX2 NV24 → packed RGBA (UV-ordered, 4:4:4, opaque alpha).
///
/// # Safety
///
/// Same as [`nv24_or_nv42_to_rgb_or_rgba_row_impl`] but `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn nv24_to_rgba_row(
  y: &[u8],
  uv: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    nv24_or_nv42_to_rgb_or_rgba_row_impl::<false, true>(y, uv, rgba_out, width, matrix, full_range);
  }
}

/// AVX2 NV42 → packed RGBA (VU-ordered, 4:4:4, opaque alpha).
///
/// # Safety
///
/// Same as [`nv24_or_nv42_to_rgb_or_rgba_row_impl`] but `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn nv42_to_rgba_row(
  y: &[u8],
  vu: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    nv24_or_nv42_to_rgb_or_rgba_row_impl::<true, true>(y, vu, rgba_out, width, matrix, full_range);
  }
}

/// Shared AVX2 NV24/NV42 kernel (4:4:4 semi-planar). 32 Y pixels / 32
/// chroma pairs / 64 UV bytes per iteration. Unlike
/// [`nv12_or_nv21_to_rgb_or_rgba_row_impl`], chroma is not subsampled — one
/// UV pair per Y pixel — so the `chroma_dup` step disappears; two
/// `chroma_i16x16` calls per channel produce 32 chroma values
/// directly.
///
/// # Safety
///
/// 1. **AVX2 must be available on the current CPU.**
/// 2. `y.len() >= width`.
/// 3. `uv_or_vu.len() >= 2 * width`.
/// 4. `out.len() >= width * if ALPHA { 4 } else { 3 }`.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn nv24_or_nv42_to_rgb_or_rgba_row_impl<const SWAP_UV: bool, const ALPHA: bool>(
  y: &[u8],
  uv_or_vu: &[u8],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(y.len() >= width);
  debug_assert!(uv_or_vu.len() >= 2 * width);
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params(full_range);
  const RND: i32 = 1 << 14;

  // SAFETY: AVX2 availability is the caller's obligation; all pointer
  // adds below are bounded by the `while x + 32 <= width` loop and
  // the caller-promised slice lengths.
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
    let alpha_u8 = _mm256_set1_epi8(-1);

    // Same per-lane deinterleave mask as the NV12 kernel: within each
    // 128-bit lane, pack even bytes into low 8, odd bytes into high 8.
    // The permute4x64_0xD8 fixup then compacts [even | odd] across the
    // full 256 bits → low 128 = even bytes, high 128 = odd bytes.
    let deint_mask = _mm256_setr_epi8(
      0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15, // low 128: per-lane
      0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15, // high 128: same
    );

    let mut x = 0usize;
    while x + 32 <= width {
      let y_vec = _mm256_loadu_si256(y.as_ptr().add(x).cast());
      // 32 Y pixels → 64 UV bytes (two 256-bit loads).
      let uv_vec_lo = _mm256_loadu_si256(uv_or_vu.as_ptr().add(x * 2).cast());
      let uv_vec_hi = _mm256_loadu_si256(uv_or_vu.as_ptr().add(x * 2 + 32).cast());

      // Per 256-bit vec: deinterleave → low 128 = U, high 128 = V
      // (roles swap for NV42).
      let d_lo = _mm256_permute4x64_epi64::<0xD8>(_mm256_shuffle_epi8(uv_vec_lo, deint_mask));
      let d_hi = _mm256_permute4x64_epi64::<0xD8>(_mm256_shuffle_epi8(uv_vec_hi, deint_mask));
      let (u_bytes_lo, v_bytes_lo, u_bytes_hi, v_bytes_hi) = if SWAP_UV {
        (
          _mm256_extracti128_si256::<1>(d_lo),
          _mm256_castsi256_si128(d_lo),
          _mm256_extracti128_si256::<1>(d_hi),
          _mm256_castsi256_si128(d_hi),
        )
      } else {
        (
          _mm256_castsi256_si128(d_lo),
          _mm256_extracti128_si256::<1>(d_lo),
          _mm256_castsi256_si128(d_hi),
          _mm256_extracti128_si256::<1>(d_hi),
        )
      };

      // Widen each 16-byte U/V chunk to i16x16 and subtract 128.
      let u_lo_i16 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(u_bytes_lo), mid128);
      let u_hi_i16 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(u_bytes_hi), mid128);
      let v_lo_i16 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(v_bytes_lo), mid128);
      let v_hi_i16 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(v_bytes_hi), mid128);

      // Split each i16x16 into two i32x8 halves for the Q15 multiply.
      let u_lo_a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(u_lo_i16));
      let u_lo_b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(u_lo_i16));
      let u_hi_a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(u_hi_i16));
      let u_hi_b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(u_hi_i16));
      let v_lo_a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v_lo_i16));
      let v_lo_b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(v_lo_i16));
      let v_hi_a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v_hi_i16));
      let v_hi_b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(v_hi_i16));

      // u_d / v_d = (u * c_scale + RND) >> 15.
      let u_d_lo_a = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_lo_a, c_scale_v),
        rnd_v,
      ));
      let u_d_lo_b = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_lo_b, c_scale_v),
        rnd_v,
      ));
      let u_d_hi_a = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_hi_a, c_scale_v),
        rnd_v,
      ));
      let u_d_hi_b = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_hi_b, c_scale_v),
        rnd_v,
      ));
      let v_d_lo_a = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(v_lo_a, c_scale_v),
        rnd_v,
      ));
      let v_d_lo_b = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(v_lo_b, c_scale_v),
        rnd_v,
      ));
      let v_d_hi_a = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(v_hi_a, c_scale_v),
        rnd_v,
      ));
      let v_d_hi_b = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(v_hi_b, c_scale_v),
        rnd_v,
      ));

      // 32 chroma per channel (two chroma_i16x16 per channel, no
      // duplication since UV is 1:1 with Y).
      let r_chroma_lo = chroma_i16x16(cru, crv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let r_chroma_hi = chroma_i16x16(cru, crv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let g_chroma_lo = chroma_i16x16(cgu, cgv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let g_chroma_hi = chroma_i16x16(cgu, cgv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let b_chroma_lo = chroma_i16x16(cbu, cbv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let b_chroma_hi = chroma_i16x16(cbu, cbv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);

      // Y path: widen 32 Y bytes to two i16x16, subtract y_off, apply
      // y_scale in Q15, narrow back to i16.
      let y_low_i16 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(y_vec));
      let y_high_i16 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(y_vec));
      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v);

      // Saturating i16 add Y + chroma, then saturating-narrow to u8x32.
      let b_lo = _mm256_adds_epi16(y_scaled_lo, b_chroma_lo);
      let b_hi = _mm256_adds_epi16(y_scaled_hi, b_chroma_hi);
      let g_lo = _mm256_adds_epi16(y_scaled_lo, g_chroma_lo);
      let g_hi = _mm256_adds_epi16(y_scaled_hi, g_chroma_hi);
      let r_lo = _mm256_adds_epi16(y_scaled_lo, r_chroma_lo);
      let r_hi = _mm256_adds_epi16(y_scaled_hi, r_chroma_hi);

      let b_u8 = narrow_u8x32(b_lo, b_hi);
      let g_u8 = narrow_u8x32(g_lo, g_hi);
      let r_u8 = narrow_u8x32(r_lo, r_hi);

      if ALPHA {
        write_rgba_32(r_u8, g_u8, b_u8, alpha_u8, out.as_mut_ptr().add(x * 4));
      } else {
        write_rgb_32(r_u8, g_u8, b_u8, out.as_mut_ptr().add(x * 3));
      }

      x += 32;
    }

    if x < width {
      let tail_y = &y[x..width];
      let tail_uv = &uv_or_vu[x * 2..width * 2];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      match (SWAP_UV, ALPHA) {
        (false, false) => {
          scalar::nv24_to_rgb_row(tail_y, tail_uv, tail_out, tail_w, matrix, full_range)
        }
        (true, false) => {
          scalar::nv42_to_rgb_row(tail_y, tail_uv, tail_out, tail_w, matrix, full_range)
        }
        (false, true) => {
          scalar::nv24_to_rgba_row(tail_y, tail_uv, tail_out, tail_w, matrix, full_range)
        }
        (true, true) => {
          scalar::nv42_to_rgba_row(tail_y, tail_uv, tail_out, tail_w, matrix, full_range)
        }
      }
    }
  }
}

/// AVX2 YUV 4:4:4 planar → packed RGB. Thin wrapper over
/// [`yuv_444_to_rgb_or_rgba_row`] with `ALPHA = false`.
///
/// # Safety
///
/// 1. **AVX2 must be available on the current CPU.**
/// 2. `y.len() >= width`, `u.len() >= width`, `v.len() >= width`,
///    `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn yuv_444_to_rgb_row(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller-checked AVX2 availability + slice bounds — see
  // [`yuv_444_to_rgb_or_rgba_row`] safety contract.
  unsafe {
    yuv_444_to_rgb_or_rgba_row::<false>(y, u, v, rgb_out, width, matrix, full_range);
  }
}

/// AVX2 YUV 4:4:4 planar → packed **RGBA** (8-bit). Same contract
/// as [`yuv_444_to_rgb_row`] but writes 4 bytes per pixel via
/// [`write_rgba_32`] (R, G, B, `0xFF`).
///
/// # Safety
///
/// Same as [`yuv_444_to_rgb_row`] except the output slice must be
/// `>= 4 * width` bytes.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn yuv_444_to_rgba_row(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller-checked AVX2 availability + slice bounds — see
  // [`yuv_444_to_rgb_or_rgba_row`] safety contract.
  unsafe {
    yuv_444_to_rgb_or_rgba_row::<true>(y, u, v, rgba_out, width, matrix, full_range);
  }
}

/// Shared AVX2 YUV 4:4:4 kernel for [`yuv_444_to_rgb_row`]
/// (`ALPHA = false`, [`write_rgb_32`]) and [`yuv_444_to_rgba_row`]
/// (`ALPHA = true`, [`write_rgba_32`] with constant `0xFF` alpha).
///
/// # Safety
///
/// 1. **AVX2 must be available on the current CPU.**
/// 2. `y.len() >= width`, `u.len() >= width`, `v.len() >= width`.
/// 3. `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn yuv_444_to_rgb_or_rgba_row<const ALPHA: bool>(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(y.len() >= width);
  debug_assert!(u.len() >= width);
  debug_assert!(v.len() >= width);
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params(full_range);
  const RND: i32 = 1 << 14;

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
    let alpha_u8 = _mm256_set1_epi8(-1); // 0xFF as i8

    let mut x = 0usize;
    while x + 32 <= width {
      let y_vec = _mm256_loadu_si256(y.as_ptr().add(x).cast());
      // 4:4:4: 32 U + 32 V directly, no deinterleave.
      let u_vec = _mm256_loadu_si256(u.as_ptr().add(x).cast());
      let v_vec = _mm256_loadu_si256(v.as_ptr().add(x).cast());

      // Widen low / high halves of U / V (16 bytes each) to i16x16.
      let u_lo_i16 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(_mm256_castsi256_si128(u_vec)), mid128);
      let u_hi_i16 = _mm256_sub_epi16(
        _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(u_vec)),
        mid128,
      );
      let v_lo_i16 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(_mm256_castsi256_si128(v_vec)), mid128);
      let v_hi_i16 = _mm256_sub_epi16(
        _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(v_vec)),
        mid128,
      );

      let u_lo_a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(u_lo_i16));
      let u_lo_b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(u_lo_i16));
      let u_hi_a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(u_hi_i16));
      let u_hi_b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(u_hi_i16));
      let v_lo_a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v_lo_i16));
      let v_lo_b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(v_lo_i16));
      let v_hi_a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v_hi_i16));
      let v_hi_b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(v_hi_i16));

      let u_d_lo_a = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_lo_a, c_scale_v),
        rnd_v,
      ));
      let u_d_lo_b = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_lo_b, c_scale_v),
        rnd_v,
      ));
      let u_d_hi_a = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_hi_a, c_scale_v),
        rnd_v,
      ));
      let u_d_hi_b = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_hi_b, c_scale_v),
        rnd_v,
      ));
      let v_d_lo_a = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(v_lo_a, c_scale_v),
        rnd_v,
      ));
      let v_d_lo_b = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(v_lo_b, c_scale_v),
        rnd_v,
      ));
      let v_d_hi_a = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(v_hi_a, c_scale_v),
        rnd_v,
      ));
      let v_d_hi_b = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(v_hi_b, c_scale_v),
        rnd_v,
      ));

      let r_chroma_lo = chroma_i16x16(cru, crv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let r_chroma_hi = chroma_i16x16(cru, crv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let g_chroma_lo = chroma_i16x16(cgu, cgv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let g_chroma_hi = chroma_i16x16(cgu, cgv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let b_chroma_lo = chroma_i16x16(cbu, cbv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let b_chroma_hi = chroma_i16x16(cbu, cbv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);

      let y_low_i16 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(y_vec));
      let y_high_i16 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(y_vec));
      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v);

      let b_lo = _mm256_adds_epi16(y_scaled_lo, b_chroma_lo);
      let b_hi = _mm256_adds_epi16(y_scaled_hi, b_chroma_hi);
      let g_lo = _mm256_adds_epi16(y_scaled_lo, g_chroma_lo);
      let g_hi = _mm256_adds_epi16(y_scaled_hi, g_chroma_hi);
      let r_lo = _mm256_adds_epi16(y_scaled_lo, r_chroma_lo);
      let r_hi = _mm256_adds_epi16(y_scaled_hi, r_chroma_hi);

      let b_u8 = narrow_u8x32(b_lo, b_hi);
      let g_u8 = narrow_u8x32(g_lo, g_hi);
      let r_u8 = narrow_u8x32(r_lo, r_hi);

      if ALPHA {
        write_rgba_32(r_u8, g_u8, b_u8, alpha_u8, out.as_mut_ptr().add(x * 4));
      } else {
        write_rgb_32(r_u8, g_u8, b_u8, out.as_mut_ptr().add(x * 3));
      }

      x += 32;
    }

    if x < width {
      let tail_y = &y[x..width];
      let tail_u = &u[x..width];
      let tail_v = &v[x..width];
      let tail_w = width - x;
      let tail_out = &mut out[x * bpp..width * bpp];
      if ALPHA {
        scalar::yuv_444_to_rgba_row(tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range);
      } else {
        scalar::yuv_444_to_rgb_row(tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range);
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

/// Writes 32 pixels of packed RGBA (128 bytes) by interleaving four
/// u8x32 R/G/B/A channel vectors. Processed as two 16‑pixel halves
/// via the shared
/// [`write_rgba_16`](super::x86_common::write_rgba_16) helper.
///
/// # Safety
///
/// `ptr` must point to at least 128 writable bytes.
#[inline(always)]
unsafe fn write_rgba_32(r: __m256i, g: __m256i, b: __m256i, a: __m256i, ptr: *mut u8) {
  unsafe {
    let r_lo = _mm256_castsi256_si128(r);
    let r_hi = _mm256_extracti128_si256::<1>(r);
    let g_lo = _mm256_castsi256_si128(g);
    let g_hi = _mm256_extracti128_si256::<1>(g);
    let b_lo = _mm256_castsi256_si128(b);
    let b_hi = _mm256_extracti128_si256::<1>(b);
    let a_lo = _mm256_castsi256_si128(a);
    let a_hi = _mm256_extracti128_si256::<1>(a);

    write_rgba_16(r_lo, g_lo, b_lo, a_lo, ptr);
    write_rgba_16(r_hi, g_hi, b_hi, a_hi, ptr.add(64));
  }
}

// ===== 16-bit YUV → RGB ==================================================

/// `(Y_u16x16 - y_off) * y_scale + RND >> 15` for full u16 Y samples.
/// Unsigned widening via `_mm256_cvtepu16_epi32`. Returns i16x16.
#[inline(always)]
fn scale_y_u16_avx2(
  y_u16x16: __m256i,
  y_off_v: __m256i,
  y_scale_v: __m256i,
  rnd_v: __m256i,
) -> __m256i {
  unsafe {
    let y_lo_i32 = _mm256_sub_epi32(
      _mm256_cvtepu16_epi32(_mm256_castsi256_si128(y_u16x16)),
      y_off_v,
    );
    let y_hi_i32 = _mm256_sub_epi32(
      _mm256_cvtepu16_epi32(_mm256_extracti128_si256::<1>(y_u16x16)),
      y_off_v,
    );
    let lo = _mm256_srai_epi32::<15>(_mm256_add_epi32(
      _mm256_mullo_epi32(y_lo_i32, y_scale_v),
      rnd_v,
    ));
    let hi = _mm256_srai_epi32::<15>(_mm256_add_epi32(
      _mm256_mullo_epi32(y_hi_i32, y_scale_v),
      rnd_v,
    ));
    _mm256_permute4x64_epi64::<0xD8>(_mm256_packs_epi32(lo, hi))
  }
}

/// Arithmetic right shift by 15 on an `i64x4` vector via the
/// "bias trick" — AVX2 lacks `_mm256_srai_epi64` until AVX-512VL, so
/// we add `2^32` before an unsigned shift and subtract the resulting
/// `2^17` offset. Valid for `|x| < 2^32`, which easily holds for our
/// chroma and Y products at 16-bit limited range.
#[inline(always)]
fn srai64_15_x4(x: __m256i) -> __m256i {
  unsafe {
    let biased = _mm256_add_epi64(x, _mm256_set1_epi64x(1i64 << 32));
    let shifted = _mm256_srli_epi64::<15>(biased);
    _mm256_sub_epi64(shifted, _mm256_set1_epi64x(1i64 << 17))
  }
}

/// Computes one i64x4 chroma channel from 4 × i64 (u_d, v_d) inputs
/// using `_mm256_mul_epi32` (even-indexed i32 lanes → 4 i64 products
/// per call). Returns i64x4 with [`srai64_15_x4`]-shifted results in
/// the low 32 bits of each i64 lane.
#[inline(always)]
fn chroma_i64x4_avx2(
  cu: __m256i,
  cv: __m256i,
  u_d: __m256i,
  v_d: __m256i,
  rnd_v: __m256i,
) -> __m256i {
  unsafe {
    srai64_15_x4(_mm256_add_epi64(
      _mm256_add_epi64(_mm256_mul_epi32(cu, u_d), _mm256_mul_epi32(cv, v_d)),
      rnd_v,
    ))
  }
}

/// Combines two i64x4 results (from even + odd i32 lanes of the
/// pre-multiply source) into an interleaved i32x8 = `[even.low32[0],
/// odd.low32[0], even.low32[1], odd.low32[1], ..., even.low32[3],
/// odd.low32[3]]`. Same shape as the SSE4.1
/// `_mm_unpacklo_epi64(_mm_unpacklo_epi32(even, odd),
/// _mm_unpackhi_epi32(even, odd))` pattern, lifted to 256 bits.
#[inline(always)]
fn reassemble_i64x4_to_i32x8(even: __m256i, odd: __m256i) -> __m256i {
  unsafe {
    _mm256_unpacklo_epi64(
      _mm256_unpacklo_epi32(even, odd),
      _mm256_unpackhi_epi32(even, odd),
    )
  }
}

/// `(y_minus_off * y_scale + RND) >> 15` computed in i64 for all 8
/// lanes of an i32x8 Y stream, returning an i32x8 result. Needs i64
/// because limited-range 16→u16 `(Y - y_off) * y_scale` can reach
/// ~2.35·10⁹ (> i32::MAX).
#[inline(always)]
fn scale_y_i32x8_i64(y_minus_off: __m256i, y_scale_v: __m256i, rnd_v: __m256i) -> __m256i {
  unsafe {
    let even = srai64_15_x4(_mm256_add_epi64(
      _mm256_mul_epi32(y_minus_off, y_scale_v),
      rnd_v,
    ));
    let odd = srai64_15_x4(_mm256_add_epi64(
      _mm256_mul_epi32(_mm256_shuffle_epi32::<0xF5>(y_minus_off), y_scale_v),
      rnd_v,
    ));
    reassemble_i64x4_to_i32x8(even, odd)
  }
}

/// Duplicates an i32x8 chroma vector into two i32x8 `"2-Y-per-chroma"`
/// vectors covering 16 pixels:
/// - Return.0 (for Y[0..8]): `[c0,c0, c1,c1, c2,c2, c3,c3]`
/// - Return.1 (for Y[8..16]): `[c4,c4, c5,c5, c6,c6, c7,c7]`
///
/// Mirrors the i16 `chroma_dup` helper's lane-cross restoration
/// pattern (`_mm256_permute2x128_si256::<0x20>` / `<0x31>`).
#[inline(always)]
fn chroma_dup_i32(chroma: __m256i) -> (__m256i, __m256i) {
  unsafe {
    let a = _mm256_unpacklo_epi32(chroma, chroma);
    let b = _mm256_unpackhi_epi32(chroma, chroma);
    let lo = _mm256_permute2x128_si256::<0x20>(a, b);
    let hi = _mm256_permute2x128_si256::<0x31>(a, b);
    (lo, hi)
  }
}

/// AVX2 YUV 4:2:0 16-bit → packed **8-bit** RGB. 32 pixels per iteration.
/// UV centering via wrapping 0x8000 trick; unsigned Y widening.
///
/// # Safety
///
/// 1. **AVX2 must be available.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `u_half.len() >= width / 2`,
///    `v_half.len() >= width / 2`, `rgb_out.len() >= 3 * width`.
///
/// Thin wrapper over [`yuv_420p16_to_rgb_or_rgba_row`] with `ALPHA = false`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn yuv_420p16_to_rgb_row(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    yuv_420p16_to_rgb_or_rgba_row::<false>(y, u_half, v_half, rgb_out, width, matrix, full_range);
  }
}

/// AVX2 16-bit YUV 4:2:0 → packed **8-bit RGBA** (`R, G, B, 0xFF`).
///
/// Thin wrapper over [`yuv_420p16_to_rgb_or_rgba_row`] with `ALPHA = true`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn yuv_420p16_to_rgba_row(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    yuv_420p16_to_rgb_or_rgba_row::<true>(y, u_half, v_half, rgba_out, width, matrix, full_range);
  }
}

/// Shared AVX2 16-bit YUV 4:2:0 kernel. `ALPHA = false` uses
/// `write_rgb_32`; `ALPHA = true` uses `write_rgba_32` with constant
/// `0xFF` alpha.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn yuv_420p16_to_rgb_or_rgba_row<const ALPHA: bool>(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert_eq!(width & 1, 0);
  debug_assert!(y.len() >= width);
  debug_assert!(u_half.len() >= width / 2);
  debug_assert!(v_half.len() >= width / 2);
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<16, 8>(full_range);
  const RND: i32 = 1 << 14;

  unsafe {
    let rnd_v = _mm256_set1_epi32(RND);
    let y_off_v = _mm256_set1_epi32(y_off);
    let y_scale_v = _mm256_set1_epi32(y_scale);
    let c_scale_v = _mm256_set1_epi32(c_scale);
    let bias16_v = _mm256_set1_epi16(-32768i16);
    let cru = _mm256_set1_epi32(coeffs.r_u());
    let crv = _mm256_set1_epi32(coeffs.r_v());
    let cgu = _mm256_set1_epi32(coeffs.g_u());
    let cgv = _mm256_set1_epi32(coeffs.g_v());
    let cbu = _mm256_set1_epi32(coeffs.b_u());
    let cbv = _mm256_set1_epi32(coeffs.b_v());
    let alpha_u8 = _mm256_set1_epi8(-1);

    let mut x = 0usize;
    while x + 32 <= width {
      let y_low = _mm256_loadu_si256(y.as_ptr().add(x).cast());
      let y_high = _mm256_loadu_si256(y.as_ptr().add(x + 16).cast());
      let u_vec = _mm256_loadu_si256(u_half.as_ptr().add(x / 2).cast());
      let v_vec = _mm256_loadu_si256(v_half.as_ptr().add(x / 2).cast());

      let u_i16 = _mm256_sub_epi16(u_vec, bias16_v);
      let v_i16 = _mm256_sub_epi16(v_vec, bias16_v);

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

      let y_scaled_lo = scale_y_u16_avx2(y_low, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y_u16_avx2(y_high, y_off_v, y_scale_v, rnd_v);

      let r_lo = _mm256_adds_epi16(y_scaled_lo, r_dup_lo);
      let r_hi = _mm256_adds_epi16(y_scaled_hi, r_dup_hi);
      let g_lo = _mm256_adds_epi16(y_scaled_lo, g_dup_lo);
      let g_hi = _mm256_adds_epi16(y_scaled_hi, g_dup_hi);
      let b_lo = _mm256_adds_epi16(y_scaled_lo, b_dup_lo);
      let b_hi = _mm256_adds_epi16(y_scaled_hi, b_dup_hi);

      let r_u8 = narrow_u8x32(r_lo, r_hi);
      let g_u8 = narrow_u8x32(g_lo, g_hi);
      let b_u8 = narrow_u8x32(b_lo, b_hi);

      if ALPHA {
        write_rgba_32(r_u8, g_u8, b_u8, alpha_u8, out.as_mut_ptr().add(x * 4));
      } else {
        write_rgb_32(r_u8, g_u8, b_u8, out.as_mut_ptr().add(x * 3));
      }
      x += 32;
    }

    if x < width {
      let tail_y = &y[x..width];
      let tail_u = &u_half[x / 2..width / 2];
      let tail_v = &v_half[x / 2..width / 2];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      if ALPHA {
        scalar::yuv_420p16_to_rgba_row(
          tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range,
        );
      } else {
        scalar::yuv_420p16_to_rgb_row(tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range);
      }
    }
  }
}

/// AVX2 YUV 4:2:0 16-bit → packed **16-bit** RGB. Native 256-bit
/// kernel using the [`srai64_15_x4`] bias trick (AVX2 lacks
/// `_mm256_srai_epi64`; `_mm256_srli_epi64` + offset gives the same
/// result for `|x| < 2^32`). 16 pixels per iteration — 2× SSE4.1's
/// 8-pixel rate.
///
/// # Safety
///
/// Same as [`yuv_420p16_to_rgb_row`] but `rgb_out: &mut [u16]`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn yuv_420p16_to_rgb_u16_row(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  unsafe {
    yuv_420p16_to_rgb_or_rgba_u16_row::<false>(
      y, u_half, v_half, rgb_out, width, matrix, full_range,
    );
  }
}

/// AVX2 sibling of [`yuv_420p16_to_rgba_row`] for native-depth `u16`
/// output. Alpha is `0xFFFF`.
///
/// # Safety
///
/// Same as [`yuv_420p16_to_rgb_u16_row`] plus `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn yuv_420p16_to_rgba_u16_row(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  unsafe {
    yuv_420p16_to_rgb_or_rgba_u16_row::<true>(
      y, u_half, v_half, rgba_out, width, matrix, full_range,
    );
  }
}

/// Shared AVX2 16-bit YUV 4:2:0 → native-depth `u16` kernel.
/// `ALPHA = false` writes RGB triples; `ALPHA = true` writes RGBA
/// quads with constant alpha `0xFFFF`.
///
/// # Safety
///
/// 1. **AVX2 must be available.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `u_half.len() >= width / 2`,
///    `v_half.len() >= width / 2`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn yuv_420p16_to_rgb_or_rgba_u16_row<const ALPHA: bool>(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert_eq!(width & 1, 0);
  debug_assert!(y.len() >= width);
  debug_assert!(u_half.len() >= width / 2);
  debug_assert!(v_half.len() >= width / 2);
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<16, 16>(full_range);
  const RND: i64 = 1 << 14;

  unsafe {
    let alpha_u16 = _mm_set1_epi16(-1i16);
    let rnd_v = _mm256_set1_epi64x(RND);
    let y_off_v = _mm256_set1_epi32(y_off);
    let y_scale_v = _mm256_set1_epi32(y_scale);
    let c_scale_v = _mm256_set1_epi32(c_scale);
    let bias16_v = _mm256_set1_epi16(-32768i16);
    let cru = _mm256_set1_epi32(coeffs.r_u());
    let crv = _mm256_set1_epi32(coeffs.r_v());
    let cgu = _mm256_set1_epi32(coeffs.g_u());
    let cgv = _mm256_set1_epi32(coeffs.g_v());
    let cbu = _mm256_set1_epi32(coeffs.b_u());
    let cbv = _mm256_set1_epi32(coeffs.b_v());

    let mut x = 0usize;
    while x + 16 <= width {
      // 16 Y (one __m256i) + 8 U + 8 V (one __m128i each).
      let y_vec = _mm256_loadu_si256(y.as_ptr().add(x).cast());
      let u_vec_128 = _mm_loadu_si128(u_half.as_ptr().add(x / 2).cast());
      let v_vec_128 = _mm_loadu_si128(v_half.as_ptr().add(x / 2).cast());

      // Center UV via wrapping `-(-32768)` trick.
      let bias16_128 = _mm256_castsi256_si128(bias16_v);
      let u_i16 = _mm_sub_epi16(u_vec_128, bias16_128);
      let v_i16 = _mm_sub_epi16(v_vec_128, bias16_128);

      // Widen i16x8 → i32x8.
      let u_i32 = _mm256_cvtepi16_epi32(u_i16);
      let v_i32 = _mm256_cvtepi16_epi32(v_i16);

      // Scale UV in i32 (|u_centered × c_scale| ≤ 32768 × ~38300
      // ≈ 1.26·10⁹ — fits i32).
      let rnd32_v = _mm256_set1_epi32(1 << 14);
      let u_d = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_i32, c_scale_v),
        rnd32_v,
      ));
      let v_d = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(v_i32, c_scale_v),
        rnd32_v,
      ));

      // i64 chroma: even/odd i32 lanes via shuffle.
      let u_d_odd = _mm256_shuffle_epi32::<0xF5>(u_d);
      let v_d_odd = _mm256_shuffle_epi32::<0xF5>(v_d);

      let r_ch_even = chroma_i64x4_avx2(cru, crv, u_d, v_d, rnd_v);
      let r_ch_odd = chroma_i64x4_avx2(cru, crv, u_d_odd, v_d_odd, rnd_v);
      let g_ch_even = chroma_i64x4_avx2(cgu, cgv, u_d, v_d, rnd_v);
      let g_ch_odd = chroma_i64x4_avx2(cgu, cgv, u_d_odd, v_d_odd, rnd_v);
      let b_ch_even = chroma_i64x4_avx2(cbu, cbv, u_d, v_d, rnd_v);
      let b_ch_odd = chroma_i64x4_avx2(cbu, cbv, u_d_odd, v_d_odd, rnd_v);

      // Reassemble i64x4 pairs → i32x8 [r0..r7].
      let r_ch_i32 = reassemble_i64x4_to_i32x8(r_ch_even, r_ch_odd);
      let g_ch_i32 = reassemble_i64x4_to_i32x8(g_ch_even, g_ch_odd);
      let b_ch_i32 = reassemble_i64x4_to_i32x8(b_ch_even, b_ch_odd);

      // Duplicate chroma for 2 Y per pair (16 chroma-dup → 2 × 8 lanes).
      let (r_dup_lo, r_dup_hi) = chroma_dup_i32(r_ch_i32);
      let (g_dup_lo, g_dup_hi) = chroma_dup_i32(g_ch_i32);
      let (b_dup_lo, b_dup_hi) = chroma_dup_i32(b_ch_i32);

      // Y scale in i64: split into two i32x8 halves, subtract y_off,
      // scale via even/odd mul_epi32.
      let y_lo_u16 = _mm256_castsi256_si128(y_vec);
      let y_hi_u16 = _mm256_extracti128_si256::<1>(y_vec);
      let y_lo_i32 = _mm256_sub_epi32(_mm256_cvtepu16_epi32(y_lo_u16), y_off_v);
      let y_hi_i32 = _mm256_sub_epi32(_mm256_cvtepu16_epi32(y_hi_u16), y_off_v);

      let y_lo_scaled = scale_y_i32x8_i64(y_lo_i32, y_scale_v, rnd_v);
      let y_hi_scaled = scale_y_i32x8_i64(y_hi_i32, y_scale_v, rnd_v);

      // Add Y + chroma, saturate to u16 via `_mm256_packus_epi32`,
      // fix up lanes via 0xD8 permute.
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

      // Write 16 pixels via two 8-pixel helper calls.
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

    if x < width {
      let tail_y = &y[x..width];
      let tail_u = &u_half[x / 2..width / 2];
      let tail_v = &v_half[x / 2..width / 2];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      if ALPHA {
        scalar::yuv_420p16_to_rgba_u16_row(
          tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range,
        );
      } else {
        scalar::yuv_420p16_to_rgb_u16_row(
          tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range,
        );
      }
    }
  }
}

/// AVX2 P016 → packed **8-bit** RGB. 32 pixels per iteration.
///
/// # Safety
///
/// 1. **AVX2 must be available.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`, `rgb_out.len() >= 3 * width`.
///
/// Thin wrapper over [`p16_to_rgb_or_rgba_row`] with `ALPHA = false`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn p16_to_rgb_row(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    p16_to_rgb_or_rgba_row::<false>(y, uv_half, rgb_out, width, matrix, full_range);
  }
}

/// AVX2 P016 → packed **8-bit RGBA** (`R, G, B, 0xFF`).
///
/// Thin wrapper over [`p16_to_rgb_or_rgba_row`] with `ALPHA = true`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn p16_to_rgba_row(
  y: &[u16],
  uv_half: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    p16_to_rgb_or_rgba_row::<true>(y, uv_half, rgba_out, width, matrix, full_range);
  }
}

/// Shared AVX2 P016 kernel. `ALPHA = false` uses `write_rgb_32`;
/// `ALPHA = true` uses `write_rgba_32` with constant `0xFF` alpha.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn p16_to_rgb_or_rgba_row<const ALPHA: bool>(
  y: &[u16],
  uv_half: &[u16],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert_eq!(width & 1, 0);
  debug_assert!(y.len() >= width);
  debug_assert!(uv_half.len() >= width);
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<16, 8>(full_range);
  const RND: i32 = 1 << 14;

  unsafe {
    let rnd_v = _mm256_set1_epi32(RND);
    let y_off_v = _mm256_set1_epi32(y_off);
    let y_scale_v = _mm256_set1_epi32(y_scale);
    let c_scale_v = _mm256_set1_epi32(c_scale);
    let bias16_v = _mm256_set1_epi16(-32768i16);
    let cru = _mm256_set1_epi32(coeffs.r_u());
    let crv = _mm256_set1_epi32(coeffs.r_v());
    let cgu = _mm256_set1_epi32(coeffs.g_u());
    let cgv = _mm256_set1_epi32(coeffs.g_v());
    let cbu = _mm256_set1_epi32(coeffs.b_u());
    let cbv = _mm256_set1_epi32(coeffs.b_v());
    let alpha_u8 = _mm256_set1_epi8(-1);

    let mut x = 0usize;
    while x + 32 <= width {
      let y_low = _mm256_loadu_si256(y.as_ptr().add(x).cast());
      let y_high = _mm256_loadu_si256(y.as_ptr().add(x + 16).cast());
      // Deinterleave 32 UV pairs (64 u16) from uv_half[x..x+32].
      // Uses the shared AVX2 deinterleave helper for Pn formats.
      let (u_vec, v_vec) = deinterleave_uv_u16_avx2(uv_half.as_ptr().add(x));

      let u_i16 = _mm256_sub_epi16(u_vec, bias16_v);
      let v_i16 = _mm256_sub_epi16(v_vec, bias16_v);

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

      let y_scaled_lo = scale_y_u16_avx2(y_low, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y_u16_avx2(y_high, y_off_v, y_scale_v, rnd_v);

      let r_lo = _mm256_adds_epi16(y_scaled_lo, r_dup_lo);
      let r_hi = _mm256_adds_epi16(y_scaled_hi, r_dup_hi);
      let g_lo = _mm256_adds_epi16(y_scaled_lo, g_dup_lo);
      let g_hi = _mm256_adds_epi16(y_scaled_hi, g_dup_hi);
      let b_lo = _mm256_adds_epi16(y_scaled_lo, b_dup_lo);
      let b_hi = _mm256_adds_epi16(y_scaled_hi, b_dup_hi);

      let r_u8 = narrow_u8x32(r_lo, r_hi);
      let g_u8 = narrow_u8x32(g_lo, g_hi);
      let b_u8 = narrow_u8x32(b_lo, b_hi);

      if ALPHA {
        write_rgba_32(r_u8, g_u8, b_u8, alpha_u8, out.as_mut_ptr().add(x * 4));
      } else {
        write_rgb_32(r_u8, g_u8, b_u8, out.as_mut_ptr().add(x * 3));
      }
      x += 32;
    }

    if x < width {
      let tail_y = &y[x..width];
      let tail_uv = &uv_half[x..width];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      if ALPHA {
        scalar::p16_to_rgba_row(tail_y, tail_uv, tail_out, tail_w, matrix, full_range);
      } else {
        scalar::p16_to_rgb_row(tail_y, tail_uv, tail_out, tail_w, matrix, full_range);
      }
    }
  }
}

/// AVX2 P016 → packed **16-bit** RGB.
/// Delegates to SSE4.1 (i64 arithmetic; no AVX2 srai_epi64).
///
/// # Safety
///
/// Same as [`p16_to_rgb_row`] but `rgb_out` is `&mut [u16]`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn p16_to_rgb_u16_row(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  unsafe {
    p16_to_rgb_or_rgba_u16_row::<false>(y, uv_half, rgb_out, width, matrix, full_range);
  }
}

/// AVX2 sibling of [`p16_to_rgba_row`] for native-depth `u16` output.
/// Alpha is `0xFFFF`.
///
/// # Safety
///
/// Same as [`p16_to_rgb_u16_row`] plus `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn p16_to_rgba_u16_row(
  y: &[u16],
  uv_half: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  unsafe {
    p16_to_rgb_or_rgba_u16_row::<true>(y, uv_half, rgba_out, width, matrix, full_range);
  }
}

/// Shared AVX2 16-bit P016 → native-depth `u16` kernel.
/// `ALPHA = false` writes RGB triples; `ALPHA = true` writes RGBA
/// quads with constant alpha `0xFFFF`.
///
/// # Safety
///
/// 1. **AVX2 must be available.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn p16_to_rgb_or_rgba_u16_row<const ALPHA: bool>(
  y: &[u16],
  uv_half: &[u16],
  out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert_eq!(width & 1, 0);
  debug_assert!(y.len() >= width);
  debug_assert!(uv_half.len() >= width);
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<16, 16>(full_range);
  const RND: i64 = 1 << 14;

  unsafe {
    let alpha_u16 = _mm_set1_epi16(-1i16);
    let rnd_v = _mm256_set1_epi64x(RND);
    let y_off_v = _mm256_set1_epi32(y_off);
    let y_scale_v = _mm256_set1_epi32(y_scale);
    let c_scale_v = _mm256_set1_epi32(c_scale);
    let bias16_v = _mm256_set1_epi16(-32768i16);
    let cru = _mm256_set1_epi32(coeffs.r_u());
    let crv = _mm256_set1_epi32(coeffs.r_v());
    let cgu = _mm256_set1_epi32(coeffs.g_u());
    let cgv = _mm256_set1_epi32(coeffs.g_v());
    let cbu = _mm256_set1_epi32(coeffs.b_u());
    let cbv = _mm256_set1_epi32(coeffs.b_v());

    // 16 pixels/iter needs 8 UV pairs = 16 u16 = 32 bytes of UV data.
    // Load as two __m128i halves so we can reuse the SSE4.1 128-bit
    // byte-shuffle mask. Each half carries 4 UV pairs; we deinterleave
    // each to [U's | V's] and then join the two U halves / two V halves.
    let split_mask_128 = _mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15);

    let mut x = 0usize;
    while x + 16 <= width {
      let y_vec = _mm256_loadu_si256(y.as_ptr().add(x).cast());
      // Two 128-bit UV loads: bytes [0..16) and [16..32). `x + 8` is
      // in u16 units (8 u16 = 16 bytes) — the second load starts at
      // byte offset 16, which is UV pair index 4.
      let uv_lo_raw = _mm_loadu_si128(uv_half.as_ptr().add(x).cast());
      let uv_hi_raw = _mm_loadu_si128(uv_half.as_ptr().add(x + 8).cast());
      // Deinterleave each half: [U0,V0,U1,V1,U2,V2,U3,V3] →
      // [U0,U1,U2,U3, V0,V1,V2,V3] (low 64b = U's, high 64b = V's).
      let uv_lo_split = _mm_shuffle_epi8(uv_lo_raw, split_mask_128);
      let uv_hi_split = _mm_shuffle_epi8(uv_hi_raw, split_mask_128);
      // Combine: low 64 of each → 8 U samples; high 64 of each → 8 V.
      let u_vec_128 = _mm_unpacklo_epi64(uv_lo_split, uv_hi_split);
      let v_vec_128 = _mm_unpackhi_epi64(uv_lo_split, uv_hi_split);

      let bias16_128 = _mm256_castsi256_si128(bias16_v);
      let u_i16 = _mm_sub_epi16(u_vec_128, bias16_128);
      let v_i16 = _mm_sub_epi16(v_vec_128, bias16_128);

      let u_i32 = _mm256_cvtepi16_epi32(u_i16);
      let v_i32 = _mm256_cvtepi16_epi32(v_i16);

      let rnd32_v = _mm256_set1_epi32(1 << 14);
      let u_d = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_i32, c_scale_v),
        rnd32_v,
      ));
      let v_d = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(v_i32, c_scale_v),
        rnd32_v,
      ));

      let u_d_odd = _mm256_shuffle_epi32::<0xF5>(u_d);
      let v_d_odd = _mm256_shuffle_epi32::<0xF5>(v_d);

      let r_ch_even = chroma_i64x4_avx2(cru, crv, u_d, v_d, rnd_v);
      let r_ch_odd = chroma_i64x4_avx2(cru, crv, u_d_odd, v_d_odd, rnd_v);
      let g_ch_even = chroma_i64x4_avx2(cgu, cgv, u_d, v_d, rnd_v);
      let g_ch_odd = chroma_i64x4_avx2(cgu, cgv, u_d_odd, v_d_odd, rnd_v);
      let b_ch_even = chroma_i64x4_avx2(cbu, cbv, u_d, v_d, rnd_v);
      let b_ch_odd = chroma_i64x4_avx2(cbu, cbv, u_d_odd, v_d_odd, rnd_v);

      let r_ch_i32 = reassemble_i64x4_to_i32x8(r_ch_even, r_ch_odd);
      let g_ch_i32 = reassemble_i64x4_to_i32x8(g_ch_even, g_ch_odd);
      let b_ch_i32 = reassemble_i64x4_to_i32x8(b_ch_even, b_ch_odd);

      let (r_dup_lo, r_dup_hi) = chroma_dup_i32(r_ch_i32);
      let (g_dup_lo, g_dup_hi) = chroma_dup_i32(g_ch_i32);
      let (b_dup_lo, b_dup_hi) = chroma_dup_i32(b_ch_i32);

      let y_lo_u16 = _mm256_castsi256_si128(y_vec);
      let y_hi_u16 = _mm256_extracti128_si256::<1>(y_vec);
      let y_lo_i32 = _mm256_sub_epi32(_mm256_cvtepu16_epi32(y_lo_u16), y_off_v);
      let y_hi_i32 = _mm256_sub_epi32(_mm256_cvtepu16_epi32(y_hi_u16), y_off_v);

      let y_lo_scaled = scale_y_i32x8_i64(y_lo_i32, y_scale_v, rnd_v);
      let y_hi_scaled = scale_y_i32x8_i64(y_hi_i32, y_scale_v, rnd_v);

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

    if x < width {
      let tail_y = &y[x..width];
      let tail_uv = &uv_half[x..width];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      if ALPHA {
        scalar::p16_to_rgba_u16_row(tail_y, tail_uv, tail_out, tail_w, matrix, full_range);
      } else {
        scalar::p16_to_rgb_u16_row(tail_y, tail_uv, tail_out, tail_w, matrix, full_range);
      }
    }
  }
}

// ===== Pn 4:4:4 (semi-planar high-bit-packed) → RGB =======================
//
// Native AVX2 4:4:4 Pn kernels — combine `yuv_444p_n_to_rgb_row`'s
// 1:1 chroma compute (no horizontal duplication) with `p_n_to_rgb_row`'s
// `deinterleave_uv_u16_avx2` pattern for the interleaved UV layout.
// 32 Y pixels per iter for the i32 paths, 16 for the i64 u16-output
// path (matches `yuv_444p16_to_rgb_u16_row`).

/// AVX2 Pn 4:4:4 high-bit-packed (BITS ∈ {10, 12}) → packed **u8** RGB.
///
/// Thin wrapper over [`p_n_444_to_rgb_or_rgba_row`] with `ALPHA = false`.
///
/// # Safety
///
/// 1. AVX2 must be available on the current CPU.
/// 2. `y.len() >= width`, `uv_full.len() >= 2 * width`,
///    `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn p_n_444_to_rgb_row<const BITS: u32>(
  y: &[u16],
  uv_full: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    p_n_444_to_rgb_or_rgba_row::<BITS, false>(y, uv_full, rgb_out, width, matrix, full_range);
  }
}

/// AVX2 Pn 4:4:4 high-bit-packed (BITS ∈ {10, 12}) → packed **8-bit
/// RGBA** (`R, G, B, 0xFF`).
///
/// Thin wrapper over [`p_n_444_to_rgb_or_rgba_row`] with `ALPHA = true`.
///
/// # Safety
///
/// Same as [`p_n_444_to_rgb_row`] but `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn p_n_444_to_rgba_row<const BITS: u32>(
  y: &[u16],
  uv_full: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    p_n_444_to_rgb_or_rgba_row::<BITS, true>(y, uv_full, rgba_out, width, matrix, full_range);
  }
}

/// Shared AVX2 Pn 4:4:4 high-bit-packed kernel for
/// [`p_n_444_to_rgb_row`] (`ALPHA = false`, `write_rgb_32`) and
/// [`p_n_444_to_rgba_row`] (`ALPHA = true`, `write_rgba_32` with
/// constant `0xFF` alpha).
///
/// # Safety
///
/// 1. AVX2 must be available on the current CPU.
/// 2. `y.len() >= width`, `uv_full.len() >= 2 * width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
/// 3. `BITS` must be one of `{10, 12}`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn p_n_444_to_rgb_or_rgba_row<const BITS: u32, const ALPHA: bool>(
  y: &[u16],
  uv_full: &[u16],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  const { assert!(BITS == 10 || BITS == 12) };
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(y.len() >= width);
  debug_assert!(uv_full.len() >= 2 * width);
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<BITS, 8>(full_range);
  let bias = scalar::chroma_bias::<BITS>();
  const RND: i32 = 1 << 14;

  unsafe {
    let rnd_v = _mm256_set1_epi32(RND);
    let y_off_v = _mm256_set1_epi16(y_off as i16);
    let y_scale_v = _mm256_set1_epi32(y_scale);
    let c_scale_v = _mm256_set1_epi32(c_scale);
    let bias_v = _mm256_set1_epi16(bias as i16);
    let shr_count = _mm_cvtsi32_si128((16 - BITS) as i32);
    let cru = _mm256_set1_epi32(coeffs.r_u());
    let crv = _mm256_set1_epi32(coeffs.r_v());
    let cgu = _mm256_set1_epi32(coeffs.g_u());
    let cgv = _mm256_set1_epi32(coeffs.g_v());
    let cbu = _mm256_set1_epi32(coeffs.b_u());
    let cbv = _mm256_set1_epi32(coeffs.b_v());
    let alpha_u8 = _mm256_set1_epi8(-1);

    let mut x = 0usize;
    while x + 32 <= width {
      let y_low_i16 = _mm256_srl_epi16(_mm256_loadu_si256(y.as_ptr().add(x).cast()), shr_count);
      let y_high_i16 =
        _mm256_srl_epi16(_mm256_loadu_si256(y.as_ptr().add(x + 16).cast()), shr_count);

      // 64 UV elements (= 32 pairs) — two deinterleave calls.
      let (u_lo_vec, v_lo_vec) = deinterleave_uv_u16_avx2(uv_full.as_ptr().add(x * 2));
      let (u_hi_vec, v_hi_vec) = deinterleave_uv_u16_avx2(uv_full.as_ptr().add(x * 2 + 32));
      let u_lo_vec = _mm256_srl_epi16(u_lo_vec, shr_count);
      let v_lo_vec = _mm256_srl_epi16(v_lo_vec, shr_count);
      let u_hi_vec = _mm256_srl_epi16(u_hi_vec, shr_count);
      let v_hi_vec = _mm256_srl_epi16(v_hi_vec, shr_count);

      let u_lo_i16 = _mm256_sub_epi16(u_lo_vec, bias_v);
      let u_hi_i16 = _mm256_sub_epi16(u_hi_vec, bias_v);
      let v_lo_i16 = _mm256_sub_epi16(v_lo_vec, bias_v);
      let v_hi_i16 = _mm256_sub_epi16(v_hi_vec, bias_v);

      let u_lo_a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(u_lo_i16));
      let u_lo_b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(u_lo_i16));
      let u_hi_a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(u_hi_i16));
      let u_hi_b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(u_hi_i16));
      let v_lo_a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v_lo_i16));
      let v_lo_b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(v_lo_i16));
      let v_hi_a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v_hi_i16));
      let v_hi_b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(v_hi_i16));

      let u_d_lo_a = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_lo_a, c_scale_v),
        rnd_v,
      ));
      let u_d_lo_b = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_lo_b, c_scale_v),
        rnd_v,
      ));
      let u_d_hi_a = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_hi_a, c_scale_v),
        rnd_v,
      ));
      let u_d_hi_b = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_hi_b, c_scale_v),
        rnd_v,
      ));
      let v_d_lo_a = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(v_lo_a, c_scale_v),
        rnd_v,
      ));
      let v_d_lo_b = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(v_lo_b, c_scale_v),
        rnd_v,
      ));
      let v_d_hi_a = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(v_hi_a, c_scale_v),
        rnd_v,
      ));
      let v_d_hi_b = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(v_hi_b, c_scale_v),
        rnd_v,
      ));

      let r_chroma_lo = chroma_i16x16(cru, crv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let r_chroma_hi = chroma_i16x16(cru, crv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let g_chroma_lo = chroma_i16x16(cgu, cgv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let g_chroma_hi = chroma_i16x16(cgu, cgv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let b_chroma_lo = chroma_i16x16(cbu, cbv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let b_chroma_hi = chroma_i16x16(cbu, cbv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);

      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v);

      let r_lo = _mm256_adds_epi16(y_scaled_lo, r_chroma_lo);
      let r_hi = _mm256_adds_epi16(y_scaled_hi, r_chroma_hi);
      let g_lo = _mm256_adds_epi16(y_scaled_lo, g_chroma_lo);
      let g_hi = _mm256_adds_epi16(y_scaled_hi, g_chroma_hi);
      let b_lo = _mm256_adds_epi16(y_scaled_lo, b_chroma_lo);
      let b_hi = _mm256_adds_epi16(y_scaled_hi, b_chroma_hi);

      let b_u8 = narrow_u8x32(b_lo, b_hi);
      let g_u8 = narrow_u8x32(g_lo, g_hi);
      let r_u8 = narrow_u8x32(r_lo, r_hi);

      if ALPHA {
        write_rgba_32(r_u8, g_u8, b_u8, alpha_u8, out.as_mut_ptr().add(x * 4));
      } else {
        write_rgb_32(r_u8, g_u8, b_u8, out.as_mut_ptr().add(x * 3));
      }

      x += 32;
    }

    if x < width {
      let tail_y = &y[x..width];
      let tail_uv = &uv_full[x * 2..width * 2];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      if ALPHA {
        scalar::p_n_444_to_rgba_row::<BITS>(tail_y, tail_uv, tail_out, tail_w, matrix, full_range);
      } else {
        scalar::p_n_444_to_rgb_row::<BITS>(tail_y, tail_uv, tail_out, tail_w, matrix, full_range);
      }
    }
  }
}

/// AVX2 Pn 4:4:4 high-bit-packed (BITS ∈ {10, 12}) → packed
/// **native-depth `u16`** RGB.
///
/// Thin wrapper over [`p_n_444_to_rgb_or_rgba_u16_row`] with `ALPHA = false`.
///
/// # Safety
///
/// 1. AVX2 must be available on the current CPU.
/// 2. `y.len() >= width`, `uv_full.len() >= 2 * width`,
///    `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn p_n_444_to_rgb_u16_row<const BITS: u32>(
  y: &[u16],
  uv_full: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    p_n_444_to_rgb_or_rgba_u16_row::<BITS, false>(y, uv_full, rgb_out, width, matrix, full_range);
  }
}

/// AVX2 sibling of [`p_n_444_to_rgba_row`] for native-depth `u16`
/// output. Alpha samples are `(1 << BITS) - 1`.
///
/// Thin wrapper over [`p_n_444_to_rgb_or_rgba_u16_row`] with `ALPHA = true`.
///
/// # Safety
///
/// Same as [`p_n_444_to_rgb_u16_row`] plus `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn p_n_444_to_rgba_u16_row<const BITS: u32>(
  y: &[u16],
  uv_full: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    p_n_444_to_rgb_or_rgba_u16_row::<BITS, true>(y, uv_full, rgba_out, width, matrix, full_range);
  }
}

/// Shared AVX2 Pn 4:4:4 high-bit-packed → native-depth `u16` kernel.
/// `ALPHA = false` writes RGB triples via 4× `write_rgb_u16_8`;
/// `ALPHA = true` writes RGBA quads via 4× `write_rgba_u16_8` with
/// constant alpha `(1 << BITS) - 1`.
///
/// # Safety
///
/// 1. AVX2 must be available on the current CPU.
/// 2. `y.len() >= width`, `uv_full.len() >= 2 * width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
/// 3. `BITS` ∈ `{10, 12}`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn p_n_444_to_rgb_or_rgba_u16_row<const BITS: u32, const ALPHA: bool>(
  y: &[u16],
  uv_full: &[u16],
  out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  const { assert!(BITS == 10 || BITS == 12) };
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(y.len() >= width);
  debug_assert!(uv_full.len() >= 2 * width);
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<BITS, BITS>(full_range);
  let bias = scalar::chroma_bias::<BITS>();
  const RND: i32 = 1 << 14;
  let out_max: i16 = ((1i32 << BITS) - 1) as i16;

  unsafe {
    let rnd_v = _mm256_set1_epi32(RND);
    let y_off_v = _mm256_set1_epi16(y_off as i16);
    let y_scale_v = _mm256_set1_epi32(y_scale);
    let c_scale_v = _mm256_set1_epi32(c_scale);
    let bias_v = _mm256_set1_epi16(bias as i16);
    let max_v = _mm256_set1_epi16(out_max);
    let zero_v = _mm256_set1_epi16(0);
    let shr_count = _mm_cvtsi32_si128((16 - BITS) as i32);
    let cru = _mm256_set1_epi32(coeffs.r_u());
    let crv = _mm256_set1_epi32(coeffs.r_v());
    let cgu = _mm256_set1_epi32(coeffs.g_u());
    let cgv = _mm256_set1_epi32(coeffs.g_v());
    let cbu = _mm256_set1_epi32(coeffs.b_u());
    let cbv = _mm256_set1_epi32(coeffs.b_v());
    let alpha_u16 = _mm_set1_epi16(out_max);

    let mut x = 0usize;
    while x + 32 <= width {
      let y_low_i16 = _mm256_srl_epi16(_mm256_loadu_si256(y.as_ptr().add(x).cast()), shr_count);
      let y_high_i16 =
        _mm256_srl_epi16(_mm256_loadu_si256(y.as_ptr().add(x + 16).cast()), shr_count);

      let (u_lo_vec, v_lo_vec) = deinterleave_uv_u16_avx2(uv_full.as_ptr().add(x * 2));
      let (u_hi_vec, v_hi_vec) = deinterleave_uv_u16_avx2(uv_full.as_ptr().add(x * 2 + 32));
      let u_lo_vec = _mm256_srl_epi16(u_lo_vec, shr_count);
      let v_lo_vec = _mm256_srl_epi16(v_lo_vec, shr_count);
      let u_hi_vec = _mm256_srl_epi16(u_hi_vec, shr_count);
      let v_hi_vec = _mm256_srl_epi16(v_hi_vec, shr_count);

      let u_lo_i16 = _mm256_sub_epi16(u_lo_vec, bias_v);
      let u_hi_i16 = _mm256_sub_epi16(u_hi_vec, bias_v);
      let v_lo_i16 = _mm256_sub_epi16(v_lo_vec, bias_v);
      let v_hi_i16 = _mm256_sub_epi16(v_hi_vec, bias_v);

      let u_lo_a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(u_lo_i16));
      let u_lo_b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(u_lo_i16));
      let u_hi_a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(u_hi_i16));
      let u_hi_b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(u_hi_i16));
      let v_lo_a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v_lo_i16));
      let v_lo_b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(v_lo_i16));
      let v_hi_a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v_hi_i16));
      let v_hi_b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(v_hi_i16));

      let u_d_lo_a = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_lo_a, c_scale_v),
        rnd_v,
      ));
      let u_d_lo_b = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_lo_b, c_scale_v),
        rnd_v,
      ));
      let u_d_hi_a = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_hi_a, c_scale_v),
        rnd_v,
      ));
      let u_d_hi_b = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_hi_b, c_scale_v),
        rnd_v,
      ));
      let v_d_lo_a = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(v_lo_a, c_scale_v),
        rnd_v,
      ));
      let v_d_lo_b = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(v_lo_b, c_scale_v),
        rnd_v,
      ));
      let v_d_hi_a = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(v_hi_a, c_scale_v),
        rnd_v,
      ));
      let v_d_hi_b = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(v_hi_b, c_scale_v),
        rnd_v,
      ));

      let r_chroma_lo = chroma_i16x16(cru, crv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let r_chroma_hi = chroma_i16x16(cru, crv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let g_chroma_lo = chroma_i16x16(cgu, cgv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let g_chroma_hi = chroma_i16x16(cgu, cgv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let b_chroma_lo = chroma_i16x16(cbu, cbv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let b_chroma_hi = chroma_i16x16(cbu, cbv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);

      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v);

      let r_lo = clamp_u16_max_x16(_mm256_adds_epi16(y_scaled_lo, r_chroma_lo), zero_v, max_v);
      let r_hi = clamp_u16_max_x16(_mm256_adds_epi16(y_scaled_hi, r_chroma_hi), zero_v, max_v);
      let g_lo = clamp_u16_max_x16(_mm256_adds_epi16(y_scaled_lo, g_chroma_lo), zero_v, max_v);
      let g_hi = clamp_u16_max_x16(_mm256_adds_epi16(y_scaled_hi, g_chroma_hi), zero_v, max_v);
      let b_lo = clamp_u16_max_x16(_mm256_adds_epi16(y_scaled_lo, b_chroma_lo), zero_v, max_v);
      let b_hi = clamp_u16_max_x16(_mm256_adds_epi16(y_scaled_hi, b_chroma_hi), zero_v, max_v);

      if ALPHA {
        let dst = out.as_mut_ptr().add(x * 4);
        write_rgba_u16_8(
          _mm256_castsi256_si128(r_lo),
          _mm256_castsi256_si128(g_lo),
          _mm256_castsi256_si128(b_lo),
          alpha_u16,
          dst,
        );
        write_rgba_u16_8(
          _mm256_extracti128_si256::<1>(r_lo),
          _mm256_extracti128_si256::<1>(g_lo),
          _mm256_extracti128_si256::<1>(b_lo),
          alpha_u16,
          dst.add(32),
        );
        write_rgba_u16_8(
          _mm256_castsi256_si128(r_hi),
          _mm256_castsi256_si128(g_hi),
          _mm256_castsi256_si128(b_hi),
          alpha_u16,
          dst.add(64),
        );
        write_rgba_u16_8(
          _mm256_extracti128_si256::<1>(r_hi),
          _mm256_extracti128_si256::<1>(g_hi),
          _mm256_extracti128_si256::<1>(b_hi),
          alpha_u16,
          dst.add(96),
        );
      } else {
        let dst = out.as_mut_ptr().add(x * 3);
        write_rgb_u16_8(
          _mm256_castsi256_si128(r_lo),
          _mm256_castsi256_si128(g_lo),
          _mm256_castsi256_si128(b_lo),
          dst,
        );
        write_rgb_u16_8(
          _mm256_extracti128_si256::<1>(r_lo),
          _mm256_extracti128_si256::<1>(g_lo),
          _mm256_extracti128_si256::<1>(b_lo),
          dst.add(24),
        );
        write_rgb_u16_8(
          _mm256_castsi256_si128(r_hi),
          _mm256_castsi256_si128(g_hi),
          _mm256_castsi256_si128(b_hi),
          dst.add(48),
        );
        write_rgb_u16_8(
          _mm256_extracti128_si256::<1>(r_hi),
          _mm256_extracti128_si256::<1>(g_hi),
          _mm256_extracti128_si256::<1>(b_hi),
          dst.add(72),
        );
      }

      x += 32;
    }

    if x < width {
      let tail_y = &y[x..width];
      let tail_uv = &uv_full[x * 2..width * 2];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      if ALPHA {
        scalar::p_n_444_to_rgba_u16_row::<BITS>(
          tail_y, tail_uv, tail_out, tail_w, matrix, full_range,
        );
      } else {
        scalar::p_n_444_to_rgb_u16_row::<BITS>(
          tail_y, tail_uv, tail_out, tail_w, matrix, full_range,
        );
      }
    }
  }
}

/// AVX2 P416 (semi-planar 4:4:4, 16-bit) → packed **u8** RGB.
/// 32 pixels per iter.
///
/// Thin wrapper over [`p_n_444_16_to_rgb_or_rgba_row`] with `ALPHA = false`.
///
/// # Safety
///
/// 1. AVX2 must be available.
/// 2. `y.len() >= width`, `uv_full.len() >= 2 * width`,
///    `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn p_n_444_16_to_rgb_row(
  y: &[u16],
  uv_full: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    p_n_444_16_to_rgb_or_rgba_row::<false>(y, uv_full, rgb_out, width, matrix, full_range);
  }
}

/// AVX2 P416 (semi-planar 4:4:4, 16-bit) → packed **8-bit RGBA**
/// (`R, G, B, 0xFF`).
///
/// Thin wrapper over [`p_n_444_16_to_rgb_or_rgba_row`] with `ALPHA = true`.
///
/// # Safety
///
/// Same as [`p_n_444_16_to_rgb_row`] but `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn p_n_444_16_to_rgba_row(
  y: &[u16],
  uv_full: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    p_n_444_16_to_rgb_or_rgba_row::<true>(y, uv_full, rgba_out, width, matrix, full_range);
  }
}

/// Shared AVX2 P416 kernel for [`p_n_444_16_to_rgb_row`]
/// (`ALPHA = false`, `write_rgb_32`) and [`p_n_444_16_to_rgba_row`]
/// (`ALPHA = true`, `write_rgba_32` with constant `0xFF` alpha).
///
/// # Safety
///
/// 1. AVX2 must be available.
/// 2. `y.len() >= width`, `uv_full.len() >= 2 * width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn p_n_444_16_to_rgb_or_rgba_row<const ALPHA: bool>(
  y: &[u16],
  uv_full: &[u16],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(y.len() >= width);
  debug_assert!(uv_full.len() >= 2 * width);
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<16, 8>(full_range);
  const RND: i32 = 1 << 14;

  unsafe {
    let rnd_v = _mm256_set1_epi32(RND);
    let y_off_v = _mm256_set1_epi32(y_off);
    let y_scale_v = _mm256_set1_epi32(y_scale);
    let c_scale_v = _mm256_set1_epi32(c_scale);
    let bias16_v = _mm256_set1_epi16(-32768i16);
    let cru = _mm256_set1_epi32(coeffs.r_u());
    let crv = _mm256_set1_epi32(coeffs.r_v());
    let cgu = _mm256_set1_epi32(coeffs.g_u());
    let cgv = _mm256_set1_epi32(coeffs.g_v());
    let cbu = _mm256_set1_epi32(coeffs.b_u());
    let cbv = _mm256_set1_epi32(coeffs.b_v());
    let alpha_u8 = _mm256_set1_epi8(-1);

    let mut x = 0usize;
    while x + 32 <= width {
      let y_low = _mm256_loadu_si256(y.as_ptr().add(x).cast());
      let y_high = _mm256_loadu_si256(y.as_ptr().add(x + 16).cast());

      let (u_lo_vec, v_lo_vec) = deinterleave_uv_u16_avx2(uv_full.as_ptr().add(x * 2));
      let (u_hi_vec, v_hi_vec) = deinterleave_uv_u16_avx2(uv_full.as_ptr().add(x * 2 + 32));

      let u_lo_i16 = _mm256_sub_epi16(u_lo_vec, bias16_v);
      let u_hi_i16 = _mm256_sub_epi16(u_hi_vec, bias16_v);
      let v_lo_i16 = _mm256_sub_epi16(v_lo_vec, bias16_v);
      let v_hi_i16 = _mm256_sub_epi16(v_hi_vec, bias16_v);

      let u_lo_a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(u_lo_i16));
      let u_lo_b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(u_lo_i16));
      let u_hi_a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(u_hi_i16));
      let u_hi_b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(u_hi_i16));
      let v_lo_a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v_lo_i16));
      let v_lo_b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(v_lo_i16));
      let v_hi_a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v_hi_i16));
      let v_hi_b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(v_hi_i16));

      let u_d_lo_a = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_lo_a, c_scale_v),
        rnd_v,
      ));
      let u_d_lo_b = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_lo_b, c_scale_v),
        rnd_v,
      ));
      let u_d_hi_a = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_hi_a, c_scale_v),
        rnd_v,
      ));
      let u_d_hi_b = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_hi_b, c_scale_v),
        rnd_v,
      ));
      let v_d_lo_a = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(v_lo_a, c_scale_v),
        rnd_v,
      ));
      let v_d_lo_b = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(v_lo_b, c_scale_v),
        rnd_v,
      ));
      let v_d_hi_a = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(v_hi_a, c_scale_v),
        rnd_v,
      ));
      let v_d_hi_b = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(v_hi_b, c_scale_v),
        rnd_v,
      ));

      let r_chroma_lo = chroma_i16x16(cru, crv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let r_chroma_hi = chroma_i16x16(cru, crv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let g_chroma_lo = chroma_i16x16(cgu, cgv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let g_chroma_hi = chroma_i16x16(cgu, cgv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let b_chroma_lo = chroma_i16x16(cbu, cbv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let b_chroma_hi = chroma_i16x16(cbu, cbv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);

      let y_scaled_lo = scale_y_u16_avx2(y_low, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y_u16_avx2(y_high, y_off_v, y_scale_v, rnd_v);

      let r_lo = _mm256_adds_epi16(y_scaled_lo, r_chroma_lo);
      let r_hi = _mm256_adds_epi16(y_scaled_hi, r_chroma_hi);
      let g_lo = _mm256_adds_epi16(y_scaled_lo, g_chroma_lo);
      let g_hi = _mm256_adds_epi16(y_scaled_hi, g_chroma_hi);
      let b_lo = _mm256_adds_epi16(y_scaled_lo, b_chroma_lo);
      let b_hi = _mm256_adds_epi16(y_scaled_hi, b_chroma_hi);

      let r_u8 = narrow_u8x32(r_lo, r_hi);
      let g_u8 = narrow_u8x32(g_lo, g_hi);
      let b_u8 = narrow_u8x32(b_lo, b_hi);

      if ALPHA {
        write_rgba_32(r_u8, g_u8, b_u8, alpha_u8, out.as_mut_ptr().add(x * 4));
      } else {
        write_rgb_32(r_u8, g_u8, b_u8, out.as_mut_ptr().add(x * 3));
      }
      x += 32;
    }

    if x < width {
      let tail_y = &y[x..width];
      let tail_uv = &uv_full[x * 2..width * 2];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      if ALPHA {
        scalar::p_n_444_16_to_rgba_row(tail_y, tail_uv, tail_out, tail_w, matrix, full_range);
      } else {
        scalar::p_n_444_16_to_rgb_row(tail_y, tail_uv, tail_out, tail_w, matrix, full_range);
      }
    }
  }
}

/// AVX2 P416 → packed **native-depth `u16`** RGB. i64 chroma via the
/// `srai64_15_x4` bias trick (AVX2 lacks `_mm256_srai_epi64`).
/// 16 pixels per iter (i64 narrows throughput).
///
/// Thin wrapper over [`p_n_444_16_to_rgb_or_rgba_u16_row`] with `ALPHA = false`.
///
/// # Safety
///
/// Same as [`p_n_444_16_to_rgb_row`] but `rgb_out: &mut [u16]`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn p_n_444_16_to_rgb_u16_row(
  y: &[u16],
  uv_full: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    p_n_444_16_to_rgb_or_rgba_u16_row::<false>(y, uv_full, rgb_out, width, matrix, full_range);
  }
}

/// AVX2 sibling of [`p_n_444_16_to_rgba_row`] for native-depth `u16`
/// output. Alpha samples are `0xFFFF`.
///
/// Thin wrapper over [`p_n_444_16_to_rgb_or_rgba_u16_row`] with `ALPHA = true`.
///
/// # Safety
///
/// Same as [`p_n_444_16_to_rgb_u16_row`] plus `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn p_n_444_16_to_rgba_u16_row(
  y: &[u16],
  uv_full: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    p_n_444_16_to_rgb_or_rgba_u16_row::<true>(y, uv_full, rgba_out, width, matrix, full_range);
  }
}

/// Shared AVX2 P416 (semi-planar 4:4:4, 16-bit) → native-depth `u16`
/// kernel. `ALPHA = false` writes RGB triples; `ALPHA = true` writes
/// RGBA quads with constant alpha `0xFFFF`.
///
/// # Safety
///
/// 1. AVX2 must be available.
/// 2. `y.len() >= width`, `uv_full.len() >= 2 * width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn p_n_444_16_to_rgb_or_rgba_u16_row<const ALPHA: bool>(
  y: &[u16],
  uv_full: &[u16],
  out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(y.len() >= width);
  debug_assert!(uv_full.len() >= 2 * width);
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<16, 16>(full_range);
  const RND: i64 = 1 << 14;

  unsafe {
    let alpha_u16 = _mm_set1_epi16(-1i16);
    let rnd_v = _mm256_set1_epi64x(RND);
    let y_off_v = _mm256_set1_epi32(y_off);
    let y_scale_v = _mm256_set1_epi32(y_scale);
    let c_scale_v = _mm256_set1_epi32(c_scale);
    let bias16_v = _mm256_set1_epi16(-32768i16);
    let cru = _mm256_set1_epi32(coeffs.r_u());
    let crv = _mm256_set1_epi32(coeffs.r_v());
    let cgu = _mm256_set1_epi32(coeffs.g_u());
    let cgv = _mm256_set1_epi32(coeffs.g_v());
    let cbu = _mm256_set1_epi32(coeffs.b_u());
    let cbv = _mm256_set1_epi32(coeffs.b_v());
    let rnd32_v = _mm256_set1_epi32(1 << 14);

    let mut x = 0usize;
    while x + 16 <= width {
      let y_vec = _mm256_loadu_si256(y.as_ptr().add(x).cast());
      // 32 UV elements (= 16 pairs) — one deinterleave call.
      let (u_vec, v_vec) = deinterleave_uv_u16_avx2(uv_full.as_ptr().add(x * 2));

      let u_i16 = _mm256_sub_epi16(u_vec, bias16_v);
      let v_i16 = _mm256_sub_epi16(v_vec, bias16_v);

      let u_lo_i32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(u_i16));
      let u_hi_i32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(u_i16));
      let v_lo_i32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v_i16));
      let v_hi_i32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(v_i16));

      let u_d_lo = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_lo_i32, c_scale_v),
        rnd32_v,
      ));
      let u_d_hi = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_hi_i32, c_scale_v),
        rnd32_v,
      ));
      let v_d_lo = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(v_lo_i32, c_scale_v),
        rnd32_v,
      ));
      let v_d_hi = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(v_hi_i32, c_scale_v),
        rnd32_v,
      ));

      // i64 chroma: even/odd lane splits (same pattern as
      // yuv_444p16_to_rgb_u16_row).
      let u_d_lo_odd = _mm256_shuffle_epi32::<0xF5>(u_d_lo);
      let u_d_hi_odd = _mm256_shuffle_epi32::<0xF5>(u_d_hi);
      let v_d_lo_odd = _mm256_shuffle_epi32::<0xF5>(v_d_lo);
      let v_d_hi_odd = _mm256_shuffle_epi32::<0xF5>(v_d_hi);

      let r_ch_lo_e = chroma_i64x4_avx2(cru, crv, u_d_lo, v_d_lo, rnd_v);
      let r_ch_lo_o = chroma_i64x4_avx2(cru, crv, u_d_lo_odd, v_d_lo_odd, rnd_v);
      let r_ch_hi_e = chroma_i64x4_avx2(cru, crv, u_d_hi, v_d_hi, rnd_v);
      let r_ch_hi_o = chroma_i64x4_avx2(cru, crv, u_d_hi_odd, v_d_hi_odd, rnd_v);
      let g_ch_lo_e = chroma_i64x4_avx2(cgu, cgv, u_d_lo, v_d_lo, rnd_v);
      let g_ch_lo_o = chroma_i64x4_avx2(cgu, cgv, u_d_lo_odd, v_d_lo_odd, rnd_v);
      let g_ch_hi_e = chroma_i64x4_avx2(cgu, cgv, u_d_hi, v_d_hi, rnd_v);
      let g_ch_hi_o = chroma_i64x4_avx2(cgu, cgv, u_d_hi_odd, v_d_hi_odd, rnd_v);
      let b_ch_lo_e = chroma_i64x4_avx2(cbu, cbv, u_d_lo, v_d_lo, rnd_v);
      let b_ch_lo_o = chroma_i64x4_avx2(cbu, cbv, u_d_lo_odd, v_d_lo_odd, rnd_v);
      let b_ch_hi_e = chroma_i64x4_avx2(cbu, cbv, u_d_hi, v_d_hi, rnd_v);
      let b_ch_hi_o = chroma_i64x4_avx2(cbu, cbv, u_d_hi_odd, v_d_hi_odd, rnd_v);

      let r_ch_lo = reassemble_i64x4_to_i32x8(r_ch_lo_e, r_ch_lo_o);
      let r_ch_hi = reassemble_i64x4_to_i32x8(r_ch_hi_e, r_ch_hi_o);
      let g_ch_lo = reassemble_i64x4_to_i32x8(g_ch_lo_e, g_ch_lo_o);
      let g_ch_hi = reassemble_i64x4_to_i32x8(g_ch_hi_e, g_ch_hi_o);
      let b_ch_lo = reassemble_i64x4_to_i32x8(b_ch_lo_e, b_ch_lo_o);
      let b_ch_hi = reassemble_i64x4_to_i32x8(b_ch_hi_e, b_ch_hi_o);

      let y_lo_u16 = _mm256_castsi256_si128(y_vec);
      let y_hi_u16 = _mm256_extracti128_si256::<1>(y_vec);
      let y_lo_i32 = _mm256_sub_epi32(_mm256_cvtepu16_epi32(y_lo_u16), y_off_v);
      let y_hi_i32 = _mm256_sub_epi32(_mm256_cvtepu16_epi32(y_hi_u16), y_off_v);

      let y_lo_scaled = scale_y_i32x8_i64(y_lo_i32, y_scale_v, rnd_v);
      let y_hi_scaled = scale_y_i32x8_i64(y_hi_i32, y_scale_v, rnd_v);

      let r_u16 = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi32(
        _mm256_add_epi32(y_lo_scaled, r_ch_lo),
        _mm256_add_epi32(y_hi_scaled, r_ch_hi),
      ));
      let g_u16 = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi32(
        _mm256_add_epi32(y_lo_scaled, g_ch_lo),
        _mm256_add_epi32(y_hi_scaled, g_ch_hi),
      ));
      let b_u16 = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi32(
        _mm256_add_epi32(y_lo_scaled, b_ch_lo),
        _mm256_add_epi32(y_hi_scaled, b_ch_hi),
      ));

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

    if x < width {
      let tail_y = &y[x..width];
      let tail_uv = &uv_full[x * 2..width * 2];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      if ALPHA {
        scalar::p_n_444_16_to_rgba_u16_row(tail_y, tail_uv, tail_out, tail_w, matrix, full_range);
      } else {
        scalar::p_n_444_16_to_rgb_u16_row(tail_y, tail_uv, tail_out, tail_w, matrix, full_range);
      }
    }
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
mod tests;
