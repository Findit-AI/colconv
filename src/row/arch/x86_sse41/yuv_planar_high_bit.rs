use core::arch::x86_64::*;

use super::*;

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
    yuv_420p_n_to_rgb_or_rgba_row::<BITS, false, false>(
      y, u_half, v_half, None, rgb_out, width, matrix, full_range,
    );
  }
}

/// SSE4.1 high-bit-depth YUV 4:2:0 → packed **8-bit RGBA** (`R, G, B, 0xFF`).
///
/// Thin wrapper over [`yuv_420p_n_to_rgb_or_rgba_row`] with `ALPHA = true`.
///
/// # Safety
///
/// Same as [`yuv_420p_n_to_rgb_row`] but `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "sse4.1")]
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
    yuv_420p_n_to_rgb_or_rgba_row::<BITS, true, false>(
      y, u_half, v_half, None, rgba_out, width, matrix, full_range,
    );
  }
}

/// SSE4.1 YUVA 4:2:0 high-bit-depth → packed **8-bit RGBA** with the
/// per-pixel alpha byte **sourced from `a_src`** (depth-converted via
/// `>> (BITS - 8)` to fit `u8`). Same numerical contract as
/// [`yuv_420p_n_to_rgba_row`] for R/G/B.
///
/// Thin wrapper over [`yuv_420p_n_to_rgb_or_rgba_row`] with
/// `ALPHA = true, ALPHA_SRC = true`.
///
/// # Safety
///
/// Same as [`yuv_420p_n_to_rgba_row`] plus `a_src.len() >= width`.
#[inline]
#[target_feature(enable = "sse4.1")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuv_420p_n_to_rgba_with_alpha_src_row<const BITS: u32>(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  a_src: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    yuv_420p_n_to_rgb_or_rgba_row::<BITS, true, true>(
      y,
      u_half,
      v_half,
      Some(a_src),
      rgba_out,
      width,
      matrix,
      full_range,
    );
  }
}

/// Shared SSE4.1 high-bit YUV 4:2:0 kernel for `yuv_420p_n_to_rgb_row`
/// (`ALPHA = false, ALPHA_SRC = false`, `write_rgb_16`),
/// `yuv_420p_n_to_rgba_row` (`ALPHA = true, ALPHA_SRC = false`,
/// `write_rgba_16` with constant `0xFF` alpha) and
/// `yuv_420p_n_to_rgba_with_alpha_src_row` (`ALPHA = true,
/// ALPHA_SRC = true`, `write_rgba_16` with the alpha lane loaded
/// from `a_src`, masked to BITS, and depth-converted via
/// `_mm_srl_epi16` with a count of `BITS - 8`).
///
/// # Safety
///
/// 1. **SSE4.1 must be available on the current CPU.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `u_half.len() >= width / 2`,
///    `v_half.len() >= width / 2`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
/// 4. When `ALPHA_SRC = true`: `a_src` must be `Some(_)` and
///    `a_src.unwrap().len() >= width`.
/// 5. `BITS` ∈ `{9, 10, 12, 14}`.
#[inline]
#[target_feature(enable = "sse4.1")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuv_420p_n_to_rgb_or_rgba_row<
  const BITS: u32,
  const ALPHA: bool,
  const ALPHA_SRC: bool,
>(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  a_src: Option<&[u16]>,
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  const { assert!(BITS == 9 || BITS == 10 || BITS == 12 || BITS == 14) };
  // Source alpha requires RGBA output.
  const { assert!(!ALPHA_SRC || ALPHA) };
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert_eq!(width & 1, 0);
  debug_assert!(y.len() >= width);
  debug_assert!(u_half.len() >= width / 2);
  debug_assert!(v_half.len() >= width / 2);
  debug_assert!(out.len() >= width * bpp);
  if ALPHA_SRC {
    debug_assert!(a_src.as_ref().is_some_and(|s| s.len() >= width));
  }

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<BITS, 8>(full_range);
  let bias = scalar::chroma_bias::<BITS>();
  const RND: i32 = 1 << 14;

  // SAFETY: SSE4.1 availability is the caller's obligation; the
  // dispatcher in `crate::row` verifies it. Pointer adds are bounded
  // by the `while x + 16 <= width` loop condition and the caller‑
  // promised slice lengths.
  unsafe {
    let rnd_v = _mm_set1_epi32(RND);
    let y_off_v = _mm_set1_epi16(y_off as i16);
    let y_scale_v = _mm_set1_epi32(y_scale);
    let c_scale_v = _mm_set1_epi32(c_scale);
    let bias_v = _mm_set1_epi16(bias as i16);
    let mask_v = _mm_set1_epi16(scalar::bits_mask::<BITS>() as i16);
    let cru = _mm_set1_epi32(coeffs.r_u());
    let crv = _mm_set1_epi32(coeffs.r_v());
    let cgu = _mm_set1_epi32(coeffs.g_u());
    let cgv = _mm_set1_epi32(coeffs.g_v());
    let cbu = _mm_set1_epi32(coeffs.b_u());
    let cbv = _mm_set1_epi32(coeffs.b_v());
    let alpha_u8 = _mm_set1_epi8(-1);

    let mut x = 0usize;
    while x + 16 <= width {
      // 16 Y = two `u16x8` loads; 8 U + 8 V = one load each. Each
      // load is AND‑masked to the low 10 bits (see matching comment
      // in [`crate::row::scalar::yuv_420p_n_to_rgb_row`]). Valid
      // 10‑bit samples ≤ 1023 pass through unchanged.
      let y_low_i16 = _mm_and_si128(_mm_loadu_si128(y.as_ptr().add(x).cast()), mask_v);
      let y_high_i16 = _mm_and_si128(_mm_loadu_si128(y.as_ptr().add(x + 8).cast()), mask_v);
      let u_vec = _mm_and_si128(_mm_loadu_si128(u_half.as_ptr().add(x / 2).cast()), mask_v);
      let v_vec = _mm_and_si128(_mm_loadu_si128(v_half.as_ptr().add(x / 2).cast()), mask_v);

      let u_i16 = _mm_sub_epi16(u_vec, bias_v);
      let v_i16 = _mm_sub_epi16(v_vec, bias_v);

      let u_lo_i32 = _mm_cvtepi16_epi32(u_i16);
      let u_hi_i32 = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(u_i16));
      let v_lo_i32 = _mm_cvtepi16_epi32(v_i16);
      let v_hi_i32 = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(v_i16));

      let u_d_lo = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_lo_i32, c_scale_v), rnd_v));
      let u_d_hi = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_hi_i32, c_scale_v), rnd_v));
      let v_d_lo = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_lo_i32, c_scale_v), rnd_v));
      let v_d_hi = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_hi_i32, c_scale_v), rnd_v));

      let r_chroma = chroma_i16x8(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let g_chroma = chroma_i16x8(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let b_chroma = chroma_i16x8(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);

      let r_dup_lo = _mm_unpacklo_epi16(r_chroma, r_chroma);
      let r_dup_hi = _mm_unpackhi_epi16(r_chroma, r_chroma);
      let g_dup_lo = _mm_unpacklo_epi16(g_chroma, g_chroma);
      let g_dup_hi = _mm_unpackhi_epi16(g_chroma, g_chroma);
      let b_dup_lo = _mm_unpacklo_epi16(b_chroma, b_chroma);
      let b_dup_hi = _mm_unpackhi_epi16(b_chroma, b_chroma);

      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v);

      let b_lo = _mm_adds_epi16(y_scaled_lo, b_dup_lo);
      let b_hi = _mm_adds_epi16(y_scaled_hi, b_dup_hi);
      let g_lo = _mm_adds_epi16(y_scaled_lo, g_dup_lo);
      let g_hi = _mm_adds_epi16(y_scaled_hi, g_dup_hi);
      let r_lo = _mm_adds_epi16(y_scaled_lo, r_dup_lo);
      let r_hi = _mm_adds_epi16(y_scaled_hi, r_dup_hi);

      let b_u8 = _mm_packus_epi16(b_lo, b_hi);
      let g_u8 = _mm_packus_epi16(g_lo, g_hi);
      let r_u8 = _mm_packus_epi16(r_lo, r_hi);

      if ALPHA {
        let a_u8 = if ALPHA_SRC {
          // SAFETY (const-checked): ALPHA_SRC = true implies the
          // wrapper passed Some(_), validated by debug_assert above.
          // Mask before shifting to harden against over-range source
          // alpha (e.g. 1024 at BITS=10), matching scalar.
          // `_mm_srli_epi16::<IMM8>` requires a literal const generic
          // shift (not stable for `BITS - 8`); use `_mm_srl_epi16`
          // with a count vector built from `BITS - 8` instead.
          let a_ptr = a_src.as_ref().unwrap_unchecked().as_ptr();
          let a_lo = _mm_and_si128(_mm_loadu_si128(a_ptr.add(x).cast()), mask_v);
          let a_hi = _mm_and_si128(_mm_loadu_si128(a_ptr.add(x + 8).cast()), mask_v);
          let a_shr = _mm_cvtsi32_si128((BITS - 8) as i32);
          let a_lo_shifted = _mm_srl_epi16(a_lo, a_shr);
          let a_hi_shifted = _mm_srl_epi16(a_hi, a_shr);
          _mm_packus_epi16(a_lo_shifted, a_hi_shifted)
        } else {
          alpha_u8
        };
        write_rgba_16(r_u8, g_u8, b_u8, a_u8, out.as_mut_ptr().add(x * 4));
      } else {
        write_rgb_16(r_u8, g_u8, b_u8, out.as_mut_ptr().add(x * 3));
      }

      x += 16;
    }

    if x < width {
      let tail_y = &y[x..width];
      let tail_u = &u_half[x / 2..width / 2];
      let tail_v = &v_half[x / 2..width / 2];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      if ALPHA_SRC {
        // SAFETY (const-checked): ALPHA_SRC = true implies Some(_).
        let tail_a = &a_src.as_ref().unwrap_unchecked()[x..width];
        scalar::yuv_420p_n_to_rgba_with_alpha_src_row::<BITS>(
          tail_y, tail_u, tail_v, tail_a, tail_out, tail_w, matrix, full_range,
        );
      } else if ALPHA {
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

/// SSE4.1 YUV 4:2:0 10‑bit → packed **10‑bit `u16`** RGB.
///
/// Block size 16 Y pixels per iteration; writes two 8‑pixel u16 RGB
/// chunks via [`write_rgb_u16_8`]. Shares all pre‑write math with the
/// u8 output path; the key differences:
/// - `range_params_n::<10, 10>` → `y_scale` / `c_scale` target the
///   10‑bit output range (values in `[0, 1023]` at Q15 exit).
/// - Clamp is explicit min/max to `[0, 1023]` — `_mm_packus_epi16`
///   would clip to u8, so we can't reuse it here.
///
/// # Numerical contract
///
/// Identical to [`scalar::yuv_420p_n_to_rgb_u16_row::<10>`].
///
/// # Safety
///
/// 1. **SSE4.1 must be available on the current CPU.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `u_half.len() >= width / 2`,
///    `v_half.len() >= width / 2`, `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "sse4.1")]
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
    yuv_420p_n_to_rgb_or_rgba_u16_row::<BITS, false, false>(
      y, u_half, v_half, None, rgb_out, width, matrix, full_range,
    );
  }
}

/// SSE4.1 sibling of [`yuv_420p_n_to_rgba_row`] for native-depth `u16`
/// output. Alpha samples are `(1 << BITS) - 1` (opaque maximum at the
/// input bit depth).
///
/// # Safety
///
/// Same as [`yuv_420p_n_to_rgb_u16_row`] plus `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "sse4.1")]
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
    yuv_420p_n_to_rgb_or_rgba_u16_row::<BITS, true, false>(
      y, u_half, v_half, None, rgba_out, width, matrix, full_range,
    );
  }
}

/// SSE4.1 YUVA 4:2:0 high-bit-depth → **native-depth `u16`** packed
/// RGBA with the per-pixel alpha element **sourced from `a_src`**
/// (already at the source's native bit depth — masked to BITS, no
/// shift) instead of being the opaque maximum `(1 << BITS) - 1`.
///
/// Thin wrapper over [`yuv_420p_n_to_rgb_or_rgba_u16_row`] with
/// `ALPHA = true, ALPHA_SRC = true`.
///
/// # Safety
///
/// Same as [`yuv_420p_n_to_rgba_u16_row`] plus `a_src.len() >= width`.
#[inline]
#[target_feature(enable = "sse4.1")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuv_420p_n_to_rgba_u16_with_alpha_src_row<const BITS: u32>(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  a_src: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    yuv_420p_n_to_rgb_or_rgba_u16_row::<BITS, true, true>(
      y,
      u_half,
      v_half,
      Some(a_src),
      rgba_out,
      width,
      matrix,
      full_range,
    );
  }
}

/// Shared SSE4.1 high-bit YUV 4:2:0 → native-depth `u16` kernel.
/// - `ALPHA = false, ALPHA_SRC = false`: `write_rgb_u16_8`.
/// - `ALPHA = true, ALPHA_SRC = false`: `write_rgba_u16_8` with
///   constant alpha `(1 << BITS) - 1`.
/// - `ALPHA = true, ALPHA_SRC = true`: `write_rgba_u16_8` with the
///   alpha lane loaded from `a_src` and masked to BITS.
///
/// # Safety
///
/// 1. **SSE4.1 must be available on the current CPU.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `u_half.len() >= width / 2`,
///    `v_half.len() >= width / 2`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
/// 4. When `ALPHA_SRC = true`: `a_src` must be `Some(_)` and
///    `a_src.unwrap().len() >= width`.
/// 5. `BITS` ∈ `{9, 10, 12, 14}`.
#[inline]
#[target_feature(enable = "sse4.1")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuv_420p_n_to_rgb_or_rgba_u16_row<
  const BITS: u32,
  const ALPHA: bool,
  const ALPHA_SRC: bool,
>(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  a_src: Option<&[u16]>,
  out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  const { assert!(BITS == 9 || BITS == 10 || BITS == 12 || BITS == 14) };
  // Source alpha requires RGBA output.
  const { assert!(!ALPHA_SRC || ALPHA) };
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert_eq!(width & 1, 0);
  debug_assert!(y.len() >= width);
  debug_assert!(u_half.len() >= width / 2);
  debug_assert!(v_half.len() >= width / 2);
  debug_assert!(out.len() >= width * bpp);
  if ALPHA_SRC {
    debug_assert!(a_src.as_ref().is_some_and(|s| s.len() >= width));
  }

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<BITS, BITS>(full_range);
  let bias = scalar::chroma_bias::<BITS>();
  const RND: i32 = 1 << 14;
  let out_max: i16 = ((1i32 << BITS) - 1) as i16;

  // SAFETY: SSE4.1 availability is the caller's obligation.
  unsafe {
    let rnd_v = _mm_set1_epi32(RND);
    let y_off_v = _mm_set1_epi16(y_off as i16);
    let y_scale_v = _mm_set1_epi32(y_scale);
    let c_scale_v = _mm_set1_epi32(c_scale);
    let bias_v = _mm_set1_epi16(bias as i16);
    let mask_v = _mm_set1_epi16(scalar::bits_mask::<BITS>() as i16);
    let max_v = _mm_set1_epi16(out_max);
    let zero_v = _mm_set1_epi16(0);
    let cru = _mm_set1_epi32(coeffs.r_u());
    let crv = _mm_set1_epi32(coeffs.r_v());
    let cgu = _mm_set1_epi32(coeffs.g_u());
    let cgv = _mm_set1_epi32(coeffs.g_v());
    let cbu = _mm_set1_epi32(coeffs.b_u());
    let cbv = _mm_set1_epi32(coeffs.b_v());
    let alpha_u16 = _mm_set1_epi16(out_max);

    let mut x = 0usize;
    while x + 16 <= width {
      // AND‑mask each load to the low 10 bits — critical for the
      // u16 output path since its larger `y_scale` / `c_scale`
      // (32768 for 10→10 full range) would let an out‑of‑range
      // sample push a `coeff * v_d` product past i16 range,
      // triggering information loss in the subsequent
      // `_mm_packs_epi32` narrow step inside `chroma_i16x8`.
      let y_low_i16 = _mm_and_si128(_mm_loadu_si128(y.as_ptr().add(x).cast()), mask_v);
      let y_high_i16 = _mm_and_si128(_mm_loadu_si128(y.as_ptr().add(x + 8).cast()), mask_v);
      let u_vec = _mm_and_si128(_mm_loadu_si128(u_half.as_ptr().add(x / 2).cast()), mask_v);
      let v_vec = _mm_and_si128(_mm_loadu_si128(v_half.as_ptr().add(x / 2).cast()), mask_v);

      let u_i16 = _mm_sub_epi16(u_vec, bias_v);
      let v_i16 = _mm_sub_epi16(v_vec, bias_v);

      let u_lo_i32 = _mm_cvtepi16_epi32(u_i16);
      let u_hi_i32 = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(u_i16));
      let v_lo_i32 = _mm_cvtepi16_epi32(v_i16);
      let v_hi_i32 = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(v_i16));

      let u_d_lo = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_lo_i32, c_scale_v), rnd_v));
      let u_d_hi = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_hi_i32, c_scale_v), rnd_v));
      let v_d_lo = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_lo_i32, c_scale_v), rnd_v));
      let v_d_hi = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_hi_i32, c_scale_v), rnd_v));

      let r_chroma = chroma_i16x8(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let g_chroma = chroma_i16x8(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let b_chroma = chroma_i16x8(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);

      let r_dup_lo = _mm_unpacklo_epi16(r_chroma, r_chroma);
      let r_dup_hi = _mm_unpackhi_epi16(r_chroma, r_chroma);
      let g_dup_lo = _mm_unpacklo_epi16(g_chroma, g_chroma);
      let g_dup_hi = _mm_unpackhi_epi16(g_chroma, g_chroma);
      let b_dup_lo = _mm_unpacklo_epi16(b_chroma, b_chroma);
      let b_dup_hi = _mm_unpackhi_epi16(b_chroma, b_chroma);

      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v);

      // Per‑channel sum + clamp to [0, 1023].
      let r_lo = clamp_u16_max(_mm_adds_epi16(y_scaled_lo, r_dup_lo), zero_v, max_v);
      let r_hi = clamp_u16_max(_mm_adds_epi16(y_scaled_hi, r_dup_hi), zero_v, max_v);
      let g_lo = clamp_u16_max(_mm_adds_epi16(y_scaled_lo, g_dup_lo), zero_v, max_v);
      let g_hi = clamp_u16_max(_mm_adds_epi16(y_scaled_hi, g_dup_hi), zero_v, max_v);
      let b_lo = clamp_u16_max(_mm_adds_epi16(y_scaled_lo, b_dup_lo), zero_v, max_v);
      let b_hi = clamp_u16_max(_mm_adds_epi16(y_scaled_hi, b_dup_hi), zero_v, max_v);

      // Two 8‑pixel u16 writes cover the 16‑pixel block.
      if ALPHA {
        let (a_lo_v, a_hi_v) = if ALPHA_SRC {
          // SAFETY (const-checked): ALPHA_SRC = true implies the
          // wrapper passed Some(_), validated by debug_assert above.
          // No depth conversion — both source alpha and u16 output are
          // at the same native bit depth (BITS), so just mask.
          let a_ptr = a_src.as_ref().unwrap_unchecked().as_ptr();
          let lo = _mm_and_si128(_mm_loadu_si128(a_ptr.add(x).cast()), mask_v);
          let hi = _mm_and_si128(_mm_loadu_si128(a_ptr.add(x + 8).cast()), mask_v);
          (lo, hi)
        } else {
          (alpha_u16, alpha_u16)
        };
        write_rgba_u16_8(r_lo, g_lo, b_lo, a_lo_v, out.as_mut_ptr().add(x * 4));
        write_rgba_u16_8(r_hi, g_hi, b_hi, a_hi_v, out.as_mut_ptr().add(x * 4 + 32));
      } else {
        write_rgb_u16_8(r_lo, g_lo, b_lo, out.as_mut_ptr().add(x * 3));
        write_rgb_u16_8(r_hi, g_hi, b_hi, out.as_mut_ptr().add(x * 3 + 24));
      }

      x += 16;
    }

    if x < width {
      let tail_y = &y[x..width];
      let tail_u = &u_half[x / 2..width / 2];
      let tail_v = &v_half[x / 2..width / 2];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      if ALPHA_SRC {
        // SAFETY (const-checked): ALPHA_SRC = true implies Some(_).
        let tail_a = &a_src.as_ref().unwrap_unchecked()[x..width];
        scalar::yuv_420p_n_to_rgba_u16_with_alpha_src_row::<BITS>(
          tail_y, tail_u, tail_v, tail_a, tail_out, tail_w, matrix, full_range,
        );
      } else if ALPHA {
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

/// SSE4.1 YUV 4:4:4 planar 9/10/12/14-bit → packed **u8** RGB.
/// Const-generic over `BITS ∈ {9, 10, 12, 14}`.
///
/// Block size: 16 pixels per iteration (same as the 4:2:0 sibling).
/// Differs from [`yuv_420p_n_to_rgb_row`] by loading full-width U/V
/// (16 samples each) and computing two chroma-per-Y-half vectors,
/// skipping the horizontal chroma-duplication step (4:4:4 chroma is
/// 1:1 with Y, not paired).
///
/// Thin wrapper over [`yuv_444p_n_to_rgb_or_rgba_row`] with
/// `ALPHA = false, ALPHA_SRC = false`.
///
/// # Numerical contract
///
/// Byte-identical to [`scalar::yuv_444p_n_to_rgb_row::<BITS>`].
///
/// # Safety
///
/// 1. **SSE4.1 must be available on the current CPU.**
/// 2. `y.len() >= width`, `u.len() >= width`, `v.len() >= width`,
///    `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "sse4.1")]
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
    yuv_444p_n_to_rgb_or_rgba_row::<BITS, false, false>(
      y, u, v, rgb_out, width, matrix, full_range, None,
    );
  }
}

/// SSE4.1 YUV 4:4:4 planar 9/10/12/14-bit → packed **8-bit RGBA**
/// (`R, G, B, 0xFF`). Same numerical contract as
/// [`yuv_444p_n_to_rgb_row`].
///
/// Thin wrapper over [`yuv_444p_n_to_rgb_or_rgba_row`] with
/// `ALPHA = true, ALPHA_SRC = false`.
///
/// # Safety
///
/// Same as [`yuv_444p_n_to_rgb_row`] but `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "sse4.1")]
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
    yuv_444p_n_to_rgb_or_rgba_row::<BITS, true, false>(
      y, u, v, rgba_out, width, matrix, full_range, None,
    );
  }
}

/// SSE4.1 YUVA 4:4:4 planar 9/10/12/14-bit → packed **8-bit RGBA** with
/// the per-pixel alpha byte **sourced from `a_src`** (depth-converted
/// via `>> (BITS - 8)`) instead of being constant `0xFF`. Same
/// numerical contract as [`yuv_444p_n_to_rgba_row`] for R/G/B.
///
/// Thin wrapper over [`yuv_444p_n_to_rgb_or_rgba_row`] with
/// `ALPHA = true, ALPHA_SRC = true`.
///
/// # Safety
///
/// Same as [`yuv_444p_n_to_rgba_row`] plus `a_src.len() >= width`.
#[inline]
#[target_feature(enable = "sse4.1")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuv_444p_n_to_rgba_with_alpha_src_row<const BITS: u32>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  a_src: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    yuv_444p_n_to_rgb_or_rgba_row::<BITS, true, true>(
      y,
      u,
      v,
      rgba_out,
      width,
      matrix,
      full_range,
      Some(a_src),
    );
  }
}

/// Shared SSE4.1 high-bit-depth YUV 4:4:4 kernel for
/// [`yuv_444p_n_to_rgb_row`] (`ALPHA = false, ALPHA_SRC = false`,
/// `write_rgb_16`), [`yuv_444p_n_to_rgba_row`] (`ALPHA = true,
/// ALPHA_SRC = false`, `write_rgba_16` with constant `0xFF` alpha) and
/// [`yuv_444p_n_to_rgba_with_alpha_src_row`] (`ALPHA = true,
/// ALPHA_SRC = true`, `write_rgba_16` with the alpha lane loaded and
/// depth-converted from `a_src`).
///
/// # Safety
///
/// 1. **SSE4.1 must be available on the current CPU.**
/// 2. `y.len() >= width`, `u.len() >= width`, `v.len() >= width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
/// 3. When `ALPHA_SRC = true`: `a_src` must be `Some(_)` and
///    `a_src.unwrap().len() >= width`.
/// 4. `BITS` must be one of `{9, 10, 12, 14}`.
#[inline]
#[target_feature(enable = "sse4.1")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuv_444p_n_to_rgb_or_rgba_row<
  const BITS: u32,
  const ALPHA: bool,
  const ALPHA_SRC: bool,
>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  a_src: Option<&[u16]>,
) {
  const { assert!(BITS == 9 || BITS == 10 || BITS == 12 || BITS == 14) };
  // Source alpha requires RGBA output — there is no 3 bpp store with
  // alpha to put it in.
  const { assert!(!ALPHA_SRC || ALPHA) };
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(y.len() >= width);
  debug_assert!(u.len() >= width);
  debug_assert!(v.len() >= width);
  debug_assert!(out.len() >= width * bpp);
  if ALPHA_SRC {
    debug_assert!(a_src.as_ref().is_some_and(|s| s.len() >= width));
  }

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<BITS, 8>(full_range);
  let bias = scalar::chroma_bias::<BITS>();
  const RND: i32 = 1 << 14;

  unsafe {
    let rnd_v = _mm_set1_epi32(RND);
    let y_off_v = _mm_set1_epi16(y_off as i16);
    let y_scale_v = _mm_set1_epi32(y_scale);
    let c_scale_v = _mm_set1_epi32(c_scale);
    let bias_v = _mm_set1_epi16(bias as i16);
    let mask_v = _mm_set1_epi16(scalar::bits_mask::<BITS>() as i16);
    let cru = _mm_set1_epi32(coeffs.r_u());
    let crv = _mm_set1_epi32(coeffs.r_v());
    let cgu = _mm_set1_epi32(coeffs.g_u());
    let cgv = _mm_set1_epi32(coeffs.g_v());
    let cbu = _mm_set1_epi32(coeffs.b_u());
    let cbv = _mm_set1_epi32(coeffs.b_v());
    let alpha_u8 = _mm_set1_epi8(-1);

    let mut x = 0usize;
    while x + 16 <= width {
      // 16 Y + 16 U + 16 V per iter. Full-width chroma load (two
      // u16x8 each) — no horizontal duplication needed.
      let y_low_i16 = _mm_and_si128(_mm_loadu_si128(y.as_ptr().add(x).cast()), mask_v);
      let y_high_i16 = _mm_and_si128(_mm_loadu_si128(y.as_ptr().add(x + 8).cast()), mask_v);
      let u_lo_vec = _mm_and_si128(_mm_loadu_si128(u.as_ptr().add(x).cast()), mask_v);
      let u_hi_vec = _mm_and_si128(_mm_loadu_si128(u.as_ptr().add(x + 8).cast()), mask_v);
      let v_lo_vec = _mm_and_si128(_mm_loadu_si128(v.as_ptr().add(x).cast()), mask_v);
      let v_hi_vec = _mm_and_si128(_mm_loadu_si128(v.as_ptr().add(x + 8).cast()), mask_v);

      let u_lo_i16 = _mm_sub_epi16(u_lo_vec, bias_v);
      let u_hi_i16 = _mm_sub_epi16(u_hi_vec, bias_v);
      let v_lo_i16 = _mm_sub_epi16(v_lo_vec, bias_v);
      let v_hi_i16 = _mm_sub_epi16(v_hi_vec, bias_v);

      // Widen each i16x8 → two i32x4 (4+4 per half).
      let u_lo_a = _mm_cvtepi16_epi32(u_lo_i16);
      let u_lo_b = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(u_lo_i16));
      let u_hi_a = _mm_cvtepi16_epi32(u_hi_i16);
      let u_hi_b = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(u_hi_i16));
      let v_lo_a = _mm_cvtepi16_epi32(v_lo_i16);
      let v_lo_b = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(v_lo_i16));
      let v_hi_a = _mm_cvtepi16_epi32(v_hi_i16);
      let v_hi_b = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(v_hi_i16));

      let u_d_lo_a = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_lo_a, c_scale_v), rnd_v));
      let u_d_lo_b = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_lo_b, c_scale_v), rnd_v));
      let u_d_hi_a = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_hi_a, c_scale_v), rnd_v));
      let u_d_hi_b = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_hi_b, c_scale_v), rnd_v));
      let v_d_lo_a = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_lo_a, c_scale_v), rnd_v));
      let v_d_lo_b = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_lo_b, c_scale_v), rnd_v));
      let v_d_hi_a = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_hi_a, c_scale_v), rnd_v));
      let v_d_hi_b = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_hi_b, c_scale_v), rnd_v));

      // Two chroma_i16x8 calls per channel produce 16 chroma values.
      let r_chroma_lo = chroma_i16x8(cru, crv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let r_chroma_hi = chroma_i16x8(cru, crv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let g_chroma_lo = chroma_i16x8(cgu, cgv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let g_chroma_hi = chroma_i16x8(cgu, cgv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let b_chroma_lo = chroma_i16x8(cbu, cbv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let b_chroma_hi = chroma_i16x8(cbu, cbv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);

      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v);

      let r_lo = _mm_adds_epi16(y_scaled_lo, r_chroma_lo);
      let r_hi = _mm_adds_epi16(y_scaled_hi, r_chroma_hi);
      let g_lo = _mm_adds_epi16(y_scaled_lo, g_chroma_lo);
      let g_hi = _mm_adds_epi16(y_scaled_hi, g_chroma_hi);
      let b_lo = _mm_adds_epi16(y_scaled_lo, b_chroma_lo);
      let b_hi = _mm_adds_epi16(y_scaled_hi, b_chroma_hi);

      let b_u8 = _mm_packus_epi16(b_lo, b_hi);
      let g_u8 = _mm_packus_epi16(g_lo, g_hi);
      let r_u8 = _mm_packus_epi16(r_lo, r_hi);

      if ALPHA {
        let a_u8 = if ALPHA_SRC {
          // SAFETY (const-checked): ALPHA_SRC = true implies the
          // wrapper passed Some(_), validated by debug_assert.
          let a_ptr = a_src.as_ref().unwrap_unchecked().as_ptr();
          let a_lo = _mm_and_si128(_mm_loadu_si128(a_ptr.add(x).cast()), mask_v);
          let a_hi = _mm_and_si128(_mm_loadu_si128(a_ptr.add(x + 8).cast()), mask_v);
          // Mask before shifting to harden against over-range source
          // alpha (e.g. 1024 at BITS=10), matching scalar. SSE4.1
          // `_mm_srli_epi16::<IMM8>` requires a literal const generic
          // shift, so use `_mm_srl_epi16` with a count vector built
          // from `BITS - 8`.
          let a_shr = _mm_cvtsi32_si128((BITS - 8) as i32);
          let a_lo_shifted = _mm_srl_epi16(a_lo, a_shr);
          let a_hi_shifted = _mm_srl_epi16(a_hi, a_shr);
          _mm_packus_epi16(a_lo_shifted, a_hi_shifted)
        } else {
          alpha_u8
        };
        write_rgba_16(r_u8, g_u8, b_u8, a_u8, out.as_mut_ptr().add(x * 4));
      } else {
        write_rgb_16(r_u8, g_u8, b_u8, out.as_mut_ptr().add(x * 3));
      }

      x += 16;
    }

    if x < width {
      let tail_y = &y[x..width];
      let tail_u = &u[x..width];
      let tail_v = &v[x..width];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      if ALPHA_SRC {
        // SAFETY (const-checked): ALPHA_SRC = true implies Some(_).
        let tail_a = &a_src.as_ref().unwrap_unchecked()[x..width];
        scalar::yuv_444p_n_to_rgba_with_alpha_src_row::<BITS>(
          tail_y, tail_u, tail_v, tail_a, tail_out, tail_w, matrix, full_range,
        );
      } else if ALPHA {
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

/// SSE4.1 YUV 4:4:4 planar 9/10/12/14-bit → **native-depth u16** RGB.
/// Const-generic over `BITS ∈ {9, 10, 12, 14}`.
///
/// Thin wrapper over [`yuv_444p_n_to_rgb_or_rgba_u16_row`] with
/// `ALPHA = false, ALPHA_SRC = false`.
///
/// # Safety
///
/// Same as [`yuv_444p_n_to_rgb_row`] but `rgb_out: &mut [u16]`.
#[inline]
#[target_feature(enable = "sse4.1")]
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
    yuv_444p_n_to_rgb_or_rgba_u16_row::<BITS, false, false>(
      y, u, v, rgb_out, width, matrix, full_range, None,
    );
  }
}

/// SSE4.1 sibling of [`yuv_444p_n_to_rgba_row`] for native-depth `u16`
/// output. Alpha samples are `(1 << BITS) - 1` (opaque maximum at the
/// input bit depth).
///
/// Thin wrapper over [`yuv_444p_n_to_rgb_or_rgba_u16_row`] with
/// `ALPHA = true, ALPHA_SRC = false`.
///
/// # Safety
///
/// Same as [`yuv_444p_n_to_rgb_u16_row`] plus `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "sse4.1")]
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
    yuv_444p_n_to_rgb_or_rgba_u16_row::<BITS, true, false>(
      y, u, v, rgba_out, width, matrix, full_range, None,
    );
  }
}

/// SSE4.1 YUVA 4:4:4 planar 9/10/12/14-bit → **native-depth `u16`**
/// packed RGBA with the per-pixel alpha element **sourced from
/// `a_src`** (already at the source's native bit depth — no depth
/// conversion) instead of being the opaque maximum `(1 << BITS) - 1`.
/// Same numerical contract as [`yuv_444p_n_to_rgba_u16_row`] for R/G/B.
///
/// Thin wrapper over [`yuv_444p_n_to_rgb_or_rgba_u16_row`] with
/// `ALPHA = true, ALPHA_SRC = true`.
///
/// # Safety
///
/// Same as [`yuv_444p_n_to_rgba_u16_row`] plus `a_src.len() >= width`.
#[inline]
#[target_feature(enable = "sse4.1")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuv_444p_n_to_rgba_u16_with_alpha_src_row<const BITS: u32>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  a_src: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    yuv_444p_n_to_rgb_or_rgba_u16_row::<BITS, true, true>(
      y,
      u,
      v,
      rgba_out,
      width,
      matrix,
      full_range,
      Some(a_src),
    );
  }
}

/// Shared SSE4.1 high-bit YUV 4:4:4 → native-depth `u16` kernel for
/// [`yuv_444p_n_to_rgb_u16_row`] (`ALPHA = false, ALPHA_SRC = false`,
/// `write_rgb_u16_8`), [`yuv_444p_n_to_rgba_u16_row`] (`ALPHA = true,
/// ALPHA_SRC = false`, `write_rgba_u16_8` with constant alpha
/// `(1 << BITS) - 1`) and
/// [`yuv_444p_n_to_rgba_u16_with_alpha_src_row`] (`ALPHA = true,
/// ALPHA_SRC = true`, `write_rgba_u16_8` with the alpha lane loaded
/// from `a_src` and masked to native bit depth — no shift since both
/// the source alpha and the u16 output element are at the same native
/// bit depth).
///
/// # Safety
///
/// 1. **SSE4.1 must be available on the current CPU.**
/// 2. `y.len() >= width`, `u.len() >= width`, `v.len() >= width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
/// 3. When `ALPHA_SRC = true`: `a_src` must be `Some(_)` and
///    `a_src.unwrap().len() >= width`.
/// 4. `BITS` ∈ `{9, 10, 12, 14}`.
#[inline]
#[target_feature(enable = "sse4.1")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuv_444p_n_to_rgb_or_rgba_u16_row<
  const BITS: u32,
  const ALPHA: bool,
  const ALPHA_SRC: bool,
>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  a_src: Option<&[u16]>,
) {
  // Compile-time guard — `out_max = ((1 << BITS) - 1) as i16` below
  // silently wraps to -1 at BITS=16, corrupting the u16 clamp. The
  // dedicated 16-bit u16-output path is `yuv_444p16_to_rgb_u16_row`.
  const { assert!(BITS == 9 || BITS == 10 || BITS == 12 || BITS == 14) };
  // Source alpha requires RGBA output — there is no 3 bpp store with
  // alpha to put it in.
  const { assert!(!ALPHA_SRC || ALPHA) };
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(y.len() >= width);
  debug_assert!(u.len() >= width);
  debug_assert!(v.len() >= width);
  debug_assert!(out.len() >= width * bpp);
  if ALPHA_SRC {
    debug_assert!(a_src.as_ref().is_some_and(|s| s.len() >= width));
  }

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<BITS, BITS>(full_range);
  let bias = scalar::chroma_bias::<BITS>();
  const RND: i32 = 1 << 14;
  let out_max: i16 = ((1i32 << BITS) - 1) as i16;

  unsafe {
    let rnd_v = _mm_set1_epi32(RND);
    let y_off_v = _mm_set1_epi16(y_off as i16);
    let y_scale_v = _mm_set1_epi32(y_scale);
    let c_scale_v = _mm_set1_epi32(c_scale);
    let bias_v = _mm_set1_epi16(bias as i16);
    let mask_v = _mm_set1_epi16(scalar::bits_mask::<BITS>() as i16);
    let max_v = _mm_set1_epi16(out_max);
    let zero_v = _mm_set1_epi16(0);
    let cru = _mm_set1_epi32(coeffs.r_u());
    let crv = _mm_set1_epi32(coeffs.r_v());
    let cgu = _mm_set1_epi32(coeffs.g_u());
    let cgv = _mm_set1_epi32(coeffs.g_v());
    let cbu = _mm_set1_epi32(coeffs.b_u());
    let cbv = _mm_set1_epi32(coeffs.b_v());
    let alpha_u16 = _mm_set1_epi16(out_max);

    let mut x = 0usize;
    while x + 16 <= width {
      let y_low_i16 = _mm_and_si128(_mm_loadu_si128(y.as_ptr().add(x).cast()), mask_v);
      let y_high_i16 = _mm_and_si128(_mm_loadu_si128(y.as_ptr().add(x + 8).cast()), mask_v);
      let u_lo_vec = _mm_and_si128(_mm_loadu_si128(u.as_ptr().add(x).cast()), mask_v);
      let u_hi_vec = _mm_and_si128(_mm_loadu_si128(u.as_ptr().add(x + 8).cast()), mask_v);
      let v_lo_vec = _mm_and_si128(_mm_loadu_si128(v.as_ptr().add(x).cast()), mask_v);
      let v_hi_vec = _mm_and_si128(_mm_loadu_si128(v.as_ptr().add(x + 8).cast()), mask_v);

      let u_lo_i16 = _mm_sub_epi16(u_lo_vec, bias_v);
      let u_hi_i16 = _mm_sub_epi16(u_hi_vec, bias_v);
      let v_lo_i16 = _mm_sub_epi16(v_lo_vec, bias_v);
      let v_hi_i16 = _mm_sub_epi16(v_hi_vec, bias_v);

      let u_lo_a = _mm_cvtepi16_epi32(u_lo_i16);
      let u_lo_b = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(u_lo_i16));
      let u_hi_a = _mm_cvtepi16_epi32(u_hi_i16);
      let u_hi_b = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(u_hi_i16));
      let v_lo_a = _mm_cvtepi16_epi32(v_lo_i16);
      let v_lo_b = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(v_lo_i16));
      let v_hi_a = _mm_cvtepi16_epi32(v_hi_i16);
      let v_hi_b = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(v_hi_i16));

      let u_d_lo_a = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_lo_a, c_scale_v), rnd_v));
      let u_d_lo_b = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_lo_b, c_scale_v), rnd_v));
      let u_d_hi_a = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_hi_a, c_scale_v), rnd_v));
      let u_d_hi_b = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_hi_b, c_scale_v), rnd_v));
      let v_d_lo_a = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_lo_a, c_scale_v), rnd_v));
      let v_d_lo_b = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_lo_b, c_scale_v), rnd_v));
      let v_d_hi_a = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_hi_a, c_scale_v), rnd_v));
      let v_d_hi_b = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_hi_b, c_scale_v), rnd_v));

      let r_chroma_lo = chroma_i16x8(cru, crv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let r_chroma_hi = chroma_i16x8(cru, crv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let g_chroma_lo = chroma_i16x8(cgu, cgv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let g_chroma_hi = chroma_i16x8(cgu, cgv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let b_chroma_lo = chroma_i16x8(cbu, cbv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let b_chroma_hi = chroma_i16x8(cbu, cbv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);

      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v);

      let r_lo = clamp_u16_max(_mm_adds_epi16(y_scaled_lo, r_chroma_lo), zero_v, max_v);
      let r_hi = clamp_u16_max(_mm_adds_epi16(y_scaled_hi, r_chroma_hi), zero_v, max_v);
      let g_lo = clamp_u16_max(_mm_adds_epi16(y_scaled_lo, g_chroma_lo), zero_v, max_v);
      let g_hi = clamp_u16_max(_mm_adds_epi16(y_scaled_hi, g_chroma_hi), zero_v, max_v);
      let b_lo = clamp_u16_max(_mm_adds_epi16(y_scaled_lo, b_chroma_lo), zero_v, max_v);
      let b_hi = clamp_u16_max(_mm_adds_epi16(y_scaled_hi, b_chroma_hi), zero_v, max_v);

      if ALPHA {
        let (a_lo_v, a_hi_v) = if ALPHA_SRC {
          // SAFETY (const-checked): ALPHA_SRC = true implies the
          // wrapper passed Some(_), validated by debug_assert above.
          // No depth conversion — both source alpha and u16 output are
          // at the same native bit depth (BITS), so just AND-mask any
          // over-range bits to match the scalar reference.
          let a_ptr = a_src.as_ref().unwrap_unchecked().as_ptr();
          let lo = _mm_and_si128(_mm_loadu_si128(a_ptr.add(x).cast()), mask_v);
          let hi = _mm_and_si128(_mm_loadu_si128(a_ptr.add(x + 8).cast()), mask_v);
          (lo, hi)
        } else {
          (alpha_u16, alpha_u16)
        };
        write_rgba_u16_8(r_lo, g_lo, b_lo, a_lo_v, out.as_mut_ptr().add(x * 4));
        write_rgba_u16_8(r_hi, g_hi, b_hi, a_hi_v, out.as_mut_ptr().add(x * 4 + 32));
      } else {
        write_rgb_u16_8(r_lo, g_lo, b_lo, out.as_mut_ptr().add(x * 3));
        write_rgb_u16_8(r_hi, g_hi, b_hi, out.as_mut_ptr().add(x * 3 + 24));
      }

      x += 16;
    }

    if x < width {
      let tail_y = &y[x..width];
      let tail_u = &u[x..width];
      let tail_v = &v[x..width];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      if ALPHA_SRC {
        // SAFETY (const-checked): ALPHA_SRC = true implies Some(_).
        let tail_a = &a_src.as_ref().unwrap_unchecked()[x..width];
        scalar::yuv_444p_n_to_rgba_u16_with_alpha_src_row::<BITS>(
          tail_y, tail_u, tail_v, tail_a, tail_out, tail_w, matrix, full_range,
        );
      } else if ALPHA {
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
