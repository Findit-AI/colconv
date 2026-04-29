use core::arch::x86_64::*;

use super::*;

// ===== Pn 4:4:4 (semi-planar high-bit-packed) → RGB =======================
//
// SSE4.1 kernels for `p_n_444_to_rgb_*<BITS>` (BITS ∈ {10, 12}) and
// `p_n_444_16_to_rgb_*` (BITS = 16). Combine the deinterleave of
// `p_n_to_rgb_row` (UV via `deinterleave_uv_u16`) with the 1:1 chroma
// compute of `yuv_444p_n_to_rgb_row` (no duplication step). Block
// size: 16 Y pixels + 32 UV `u16` elements per iter (two
// `deinterleave_uv_u16` calls).

/// SSE4.1 Pn 4:4:4 high-bit-packed (BITS ∈ {10, 12}) → packed **u8** RGB.
///
/// Thin wrapper over [`p_n_444_to_rgb_or_rgba_row`] with `ALPHA = false`.
///
/// # Safety
///
/// 1. SSE4.1 must be available on the current CPU.
/// 2. `y.len() >= width`, `uv_full.len() >= 2 * width`,
///    `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "sse4.1")]
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

/// SSE4.1 Pn 4:4:4 high-bit-packed (BITS ∈ {10, 12}) → packed **8-bit
/// RGBA** (`R, G, B, 0xFF`).
///
/// Thin wrapper over [`p_n_444_to_rgb_or_rgba_row`] with `ALPHA = true`.
///
/// # Safety
///
/// Same as [`p_n_444_to_rgb_row`] but `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "sse4.1")]
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

/// Shared SSE4.1 Pn 4:4:4 high-bit-packed kernel for
/// [`p_n_444_to_rgb_row`] (`ALPHA = false`, `write_rgb_16`) and
/// [`p_n_444_to_rgba_row`] (`ALPHA = true`, `write_rgba_16` with
/// constant `0xFF` alpha).
///
/// # Safety
///
/// 1. SSE4.1 must be available on the current CPU.
/// 2. `y.len() >= width`, `uv_full.len() >= 2 * width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
/// 3. `BITS` must be one of `{10, 12}`.
#[inline]
#[target_feature(enable = "sse4.1")]
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
    let rnd_v = _mm_set1_epi32(RND);
    let y_off_v = _mm_set1_epi16(y_off as i16);
    let y_scale_v = _mm_set1_epi32(y_scale);
    let c_scale_v = _mm_set1_epi32(c_scale);
    let bias_v = _mm_set1_epi16(bias as i16);
    let shr_count = _mm_cvtsi32_si128((16 - BITS) as i32);
    let cru = _mm_set1_epi32(coeffs.r_u());
    let crv = _mm_set1_epi32(coeffs.r_v());
    let cgu = _mm_set1_epi32(coeffs.g_u());
    let cgv = _mm_set1_epi32(coeffs.g_v());
    let cbu = _mm_set1_epi32(coeffs.b_u());
    let cbv = _mm_set1_epi32(coeffs.b_v());
    let alpha_u8 = _mm_set1_epi8(-1);

    let mut x = 0usize;
    while x + 16 <= width {
      let y_low_i16 = _mm_srl_epi16(_mm_loadu_si128(y.as_ptr().add(x).cast()), shr_count);
      let y_high_i16 = _mm_srl_epi16(_mm_loadu_si128(y.as_ptr().add(x + 8).cast()), shr_count);

      // Two deinterleave calls — 32 UV u16 elements (= 16 pairs).
      let (u_lo_vec, v_lo_vec) = deinterleave_uv_u16(uv_full.as_ptr().add(x * 2));
      let (u_hi_vec, v_hi_vec) = deinterleave_uv_u16(uv_full.as_ptr().add(x * 2 + 16));
      let u_lo_vec = _mm_srl_epi16(u_lo_vec, shr_count);
      let v_lo_vec = _mm_srl_epi16(v_lo_vec, shr_count);
      let u_hi_vec = _mm_srl_epi16(u_hi_vec, shr_count);
      let v_hi_vec = _mm_srl_epi16(v_hi_vec, shr_count);

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
        write_rgba_16(r_u8, g_u8, b_u8, alpha_u8, out.as_mut_ptr().add(x * 4));
      } else {
        write_rgb_16(r_u8, g_u8, b_u8, out.as_mut_ptr().add(x * 3));
      }

      x += 16;
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

/// SSE4.1 Pn 4:4:4 high-bit-packed (BITS ∈ {10, 12}) → packed
/// **native-depth `u16`** RGB. Output is low-bit-packed.
///
/// Thin wrapper over [`p_n_444_to_rgb_or_rgba_u16_row`] with `ALPHA = false`.
///
/// # Safety
///
/// 1. SSE4.1 must be available on the current CPU.
/// 2. `y.len() >= width`, `uv_full.len() >= 2 * width`,
///    `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "sse4.1")]
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

/// SSE4.1 sibling of [`p_n_444_to_rgba_row`] for native-depth `u16`
/// output. Alpha samples are `(1 << BITS) - 1` (opaque maximum at the
/// input bit depth).
///
/// Thin wrapper over [`p_n_444_to_rgb_or_rgba_u16_row`] with `ALPHA = true`.
///
/// # Safety
///
/// Same as [`p_n_444_to_rgb_u16_row`] plus `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "sse4.1")]
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

/// Shared SSE4.1 Pn 4:4:4 high-bit-packed → native-depth `u16` kernel.
/// `ALPHA = false` writes RGB triples via `write_rgb_u16_8`;
/// `ALPHA = true` writes RGBA quads via `write_rgba_u16_8` with
/// constant alpha `(1 << BITS) - 1`.
///
/// # Safety
///
/// 1. SSE4.1 must be available on the current CPU.
/// 2. `y.len() >= width`, `uv_full.len() >= 2 * width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
/// 3. `BITS` ∈ `{10, 12}`.
#[inline]
#[target_feature(enable = "sse4.1")]
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
    let rnd_v = _mm_set1_epi32(RND);
    let y_off_v = _mm_set1_epi16(y_off as i16);
    let y_scale_v = _mm_set1_epi32(y_scale);
    let c_scale_v = _mm_set1_epi32(c_scale);
    let bias_v = _mm_set1_epi16(bias as i16);
    let max_v = _mm_set1_epi16(out_max);
    let zero_v = _mm_set1_epi16(0);
    let shr_count = _mm_cvtsi32_si128((16 - BITS) as i32);
    let cru = _mm_set1_epi32(coeffs.r_u());
    let crv = _mm_set1_epi32(coeffs.r_v());
    let cgu = _mm_set1_epi32(coeffs.g_u());
    let cgv = _mm_set1_epi32(coeffs.g_v());
    let cbu = _mm_set1_epi32(coeffs.b_u());
    let cbv = _mm_set1_epi32(coeffs.b_v());
    let alpha_u16 = _mm_set1_epi16(out_max);

    let mut x = 0usize;
    while x + 16 <= width {
      let y_low_i16 = _mm_srl_epi16(_mm_loadu_si128(y.as_ptr().add(x).cast()), shr_count);
      let y_high_i16 = _mm_srl_epi16(_mm_loadu_si128(y.as_ptr().add(x + 8).cast()), shr_count);

      let (u_lo_vec, v_lo_vec) = deinterleave_uv_u16(uv_full.as_ptr().add(x * 2));
      let (u_hi_vec, v_hi_vec) = deinterleave_uv_u16(uv_full.as_ptr().add(x * 2 + 16));
      let u_lo_vec = _mm_srl_epi16(u_lo_vec, shr_count);
      let v_lo_vec = _mm_srl_epi16(v_lo_vec, shr_count);
      let u_hi_vec = _mm_srl_epi16(u_hi_vec, shr_count);
      let v_hi_vec = _mm_srl_epi16(v_hi_vec, shr_count);

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
        write_rgba_u16_8(r_lo, g_lo, b_lo, alpha_u16, out.as_mut_ptr().add(x * 4));
        write_rgba_u16_8(
          r_hi,
          g_hi,
          b_hi,
          alpha_u16,
          out.as_mut_ptr().add(x * 4 + 32),
        );
      } else {
        write_rgb_u16_8(r_lo, g_lo, b_lo, out.as_mut_ptr().add(x * 3));
        write_rgb_u16_8(r_hi, g_hi, b_hi, out.as_mut_ptr().add(x * 3 + 24));
      }

      x += 16;
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

/// SSE4.1 P416 (semi-planar 4:4:4, 16-bit) → packed **u8** RGB. Y +
/// chroma both stay on i32 (output-range scaling keeps `coeff × u_d`
/// within i32 for u8 output). Mirrors `yuv_444p16_to_rgb_row` with
/// full-width interleaved UV via `deinterleave_uv_u16`.
///
/// Thin wrapper over [`p_n_444_16_to_rgb_or_rgba_row`] with `ALPHA = false`.
///
/// # Safety
///
/// 1. SSE4.1 must be available.
/// 2. `y.len() >= width`, `uv_full.len() >= 2 * width`,
///    `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "sse4.1")]
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

/// SSE4.1 P416 (semi-planar 4:4:4, 16-bit) → packed **8-bit RGBA**
/// (`R, G, B, 0xFF`). Same numerical contract as
/// [`p_n_444_16_to_rgb_row`].
///
/// Thin wrapper over [`p_n_444_16_to_rgb_or_rgba_row`] with `ALPHA = true`.
///
/// # Safety
///
/// Same as [`p_n_444_16_to_rgb_row`] but `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "sse4.1")]
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

/// Shared SSE4.1 P416 kernel for [`p_n_444_16_to_rgb_row`]
/// (`ALPHA = false`, `write_rgb_16`) and [`p_n_444_16_to_rgba_row`]
/// (`ALPHA = true`, `write_rgba_16` with constant `0xFF` alpha).
///
/// # Safety
///
/// 1. SSE4.1 must be available.
/// 2. `y.len() >= width`, `uv_full.len() >= 2 * width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
#[inline]
#[target_feature(enable = "sse4.1")]
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
    let rnd_v = _mm_set1_epi32(RND);
    let y_off_v = _mm_set1_epi32(y_off);
    let y_scale_v = _mm_set1_epi32(y_scale);
    let c_scale_v = _mm_set1_epi32(c_scale);
    let bias16_v = _mm_set1_epi16(-32768i16);
    let cru = _mm_set1_epi32(coeffs.r_u());
    let crv = _mm_set1_epi32(coeffs.r_v());
    let cgu = _mm_set1_epi32(coeffs.g_u());
    let cgv = _mm_set1_epi32(coeffs.g_v());
    let cbu = _mm_set1_epi32(coeffs.b_u());
    let cbv = _mm_set1_epi32(coeffs.b_v());
    let alpha_u8 = _mm_set1_epi8(-1);

    let mut x = 0usize;
    while x + 16 <= width {
      let y_low = _mm_loadu_si128(y.as_ptr().add(x).cast());
      let y_high = _mm_loadu_si128(y.as_ptr().add(x + 8).cast());

      // 32 UV elements per iter — two deinterleave calls.
      let (u_lo_vec, v_lo_vec) = deinterleave_uv_u16(uv_full.as_ptr().add(x * 2));
      let (u_hi_vec, v_hi_vec) = deinterleave_uv_u16(uv_full.as_ptr().add(x * 2 + 16));

      let u_lo_i16 = _mm_sub_epi16(u_lo_vec, bias16_v);
      let u_hi_i16 = _mm_sub_epi16(u_hi_vec, bias16_v);
      let v_lo_i16 = _mm_sub_epi16(v_lo_vec, bias16_v);
      let v_hi_i16 = _mm_sub_epi16(v_hi_vec, bias16_v);

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

      let y_scaled_lo = scale_y_u16(y_low, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y_u16(y_high, y_off_v, y_scale_v, rnd_v);

      let r_lo = _mm_adds_epi16(y_scaled_lo, r_chroma_lo);
      let r_hi = _mm_adds_epi16(y_scaled_hi, r_chroma_hi);
      let g_lo = _mm_adds_epi16(y_scaled_lo, g_chroma_lo);
      let g_hi = _mm_adds_epi16(y_scaled_hi, g_chroma_hi);
      let b_lo = _mm_adds_epi16(y_scaled_lo, b_chroma_lo);
      let b_hi = _mm_adds_epi16(y_scaled_hi, b_chroma_hi);

      let r_u8 = _mm_packus_epi16(r_lo, r_hi);
      let g_u8 = _mm_packus_epi16(g_lo, g_hi);
      let b_u8 = _mm_packus_epi16(b_lo, b_hi);

      if ALPHA {
        write_rgba_16(r_u8, g_u8, b_u8, alpha_u8, out.as_mut_ptr().add(x * 4));
      } else {
        write_rgb_16(r_u8, g_u8, b_u8, out.as_mut_ptr().add(x * 3));
      }
      x += 16;
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

/// SSE4.1 P416 → packed **native-depth u16** RGB. i64 chroma via
/// `_mm_mul_epi32` + `srai64_15` bias trick (mirroring
/// `yuv_444p16_to_rgb_u16_row`). 8 pixels per iter.
///
/// Thin wrapper over [`p_n_444_16_to_rgb_or_rgba_u16_row`] with `ALPHA = false`.
///
/// # Safety
///
/// Same as [`p_n_444_16_to_rgb_row`] but `rgb_out: &mut [u16]`.
#[inline]
#[target_feature(enable = "sse4.1")]
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

/// SSE4.1 sibling of [`p_n_444_16_to_rgba_row`] for native-depth `u16`
/// output. Alpha samples are `0xFFFF`.
///
/// Thin wrapper over [`p_n_444_16_to_rgb_or_rgba_u16_row`] with `ALPHA = true`.
///
/// # Safety
///
/// Same as [`p_n_444_16_to_rgb_u16_row`] plus `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "sse4.1")]
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

/// Shared SSE4.1 P416 (semi-planar 4:4:4, 16-bit) → native-depth `u16`
/// kernel. `ALPHA = false` writes RGB triples; `ALPHA = true` writes
/// RGBA quads with constant alpha `0xFFFF`.
///
/// # Safety
///
/// 1. SSE4.1 must be available.
/// 2. `y.len() >= width`, `uv_full.len() >= 2 * width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
#[inline]
#[target_feature(enable = "sse4.1")]
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
    let rnd_v = _mm_set1_epi64x(RND);
    let y_off_v = _mm_set1_epi32(y_off);
    let y_scale_v = _mm_set1_epi32(y_scale);
    let c_scale_v = _mm_set1_epi32(c_scale);
    let bias16_v = _mm_set1_epi16(-32768i16);
    let cru = _mm_set1_epi32(coeffs.r_u());
    let crv = _mm_set1_epi32(coeffs.r_v());
    let cgu = _mm_set1_epi32(coeffs.g_u());
    let cgv = _mm_set1_epi32(coeffs.g_v());
    let cbu = _mm_set1_epi32(coeffs.b_u());
    let cbv = _mm_set1_epi32(coeffs.b_v());

    let rnd32_v = _mm_set1_epi32(1 << 14);

    let mut x = 0usize;
    while x + 8 <= width {
      // 8 pixels per iter (i64 narrows). 16 UV u16 elements (= 8 pairs).
      let y_vec = _mm_loadu_si128(y.as_ptr().add(x).cast());
      let (u_vec, v_vec) = deinterleave_uv_u16(uv_full.as_ptr().add(x * 2));

      let u_i16 = _mm_sub_epi16(u_vec, bias16_v);
      let v_i16 = _mm_sub_epi16(v_vec, bias16_v);

      let u_lo_i32 = _mm_cvtepi16_epi32(u_i16);
      let u_hi_i32 = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(u_i16));
      let v_lo_i32 = _mm_cvtepi16_epi32(v_i16);
      let v_hi_i32 = _mm_cvtepi16_epi32(_mm_srli_si128::<8>(v_i16));

      let u_d_lo = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_lo_i32, c_scale_v), rnd32_v));
      let u_d_hi = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_hi_i32, c_scale_v), rnd32_v));
      let v_d_lo = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_lo_i32, c_scale_v), rnd32_v));
      let v_d_hi = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_hi_i32, c_scale_v), rnd32_v));

      // i64 chroma via even/odd splits — same pattern as
      // yuv_444p16_to_rgb_u16_row.
      let u_d_lo_even = u_d_lo;
      let u_d_lo_odd = _mm_shuffle_epi32::<0xF5>(u_d_lo);
      let v_d_lo_even = v_d_lo;
      let v_d_lo_odd = _mm_shuffle_epi32::<0xF5>(v_d_lo);
      let u_d_hi_even = u_d_hi;
      let u_d_hi_odd = _mm_shuffle_epi32::<0xF5>(u_d_hi);
      let v_d_hi_even = v_d_hi;
      let v_d_hi_odd = _mm_shuffle_epi32::<0xF5>(v_d_hi);

      let r_ch_lo_even = chroma_i64x2(cru, crv, u_d_lo_even, v_d_lo_even, rnd_v);
      let r_ch_lo_odd = chroma_i64x2(cru, crv, u_d_lo_odd, v_d_lo_odd, rnd_v);
      let r_ch_hi_even = chroma_i64x2(cru, crv, u_d_hi_even, v_d_hi_even, rnd_v);
      let r_ch_hi_odd = chroma_i64x2(cru, crv, u_d_hi_odd, v_d_hi_odd, rnd_v);
      let g_ch_lo_even = chroma_i64x2(cgu, cgv, u_d_lo_even, v_d_lo_even, rnd_v);
      let g_ch_lo_odd = chroma_i64x2(cgu, cgv, u_d_lo_odd, v_d_lo_odd, rnd_v);
      let g_ch_hi_even = chroma_i64x2(cgu, cgv, u_d_hi_even, v_d_hi_even, rnd_v);
      let g_ch_hi_odd = chroma_i64x2(cgu, cgv, u_d_hi_odd, v_d_hi_odd, rnd_v);
      let b_ch_lo_even = chroma_i64x2(cbu, cbv, u_d_lo_even, v_d_lo_even, rnd_v);
      let b_ch_lo_odd = chroma_i64x2(cbu, cbv, u_d_lo_odd, v_d_lo_odd, rnd_v);
      let b_ch_hi_even = chroma_i64x2(cbu, cbv, u_d_hi_even, v_d_hi_even, rnd_v);
      let b_ch_hi_odd = chroma_i64x2(cbu, cbv, u_d_hi_odd, v_d_hi_odd, rnd_v);

      let r_ch_lo_i32 = _mm_unpacklo_epi64(
        _mm_unpacklo_epi32(r_ch_lo_even, r_ch_lo_odd),
        _mm_unpackhi_epi32(r_ch_lo_even, r_ch_lo_odd),
      );
      let r_ch_hi_i32 = _mm_unpacklo_epi64(
        _mm_unpacklo_epi32(r_ch_hi_even, r_ch_hi_odd),
        _mm_unpackhi_epi32(r_ch_hi_even, r_ch_hi_odd),
      );
      let g_ch_lo_i32 = _mm_unpacklo_epi64(
        _mm_unpacklo_epi32(g_ch_lo_even, g_ch_lo_odd),
        _mm_unpackhi_epi32(g_ch_lo_even, g_ch_lo_odd),
      );
      let g_ch_hi_i32 = _mm_unpacklo_epi64(
        _mm_unpacklo_epi32(g_ch_hi_even, g_ch_hi_odd),
        _mm_unpackhi_epi32(g_ch_hi_even, g_ch_hi_odd),
      );
      let b_ch_lo_i32 = _mm_unpacklo_epi64(
        _mm_unpacklo_epi32(b_ch_lo_even, b_ch_lo_odd),
        _mm_unpackhi_epi32(b_ch_lo_even, b_ch_lo_odd),
      );
      let b_ch_hi_i32 = _mm_unpacklo_epi64(
        _mm_unpacklo_epi32(b_ch_hi_even, b_ch_hi_odd),
        _mm_unpackhi_epi32(b_ch_hi_even, b_ch_hi_odd),
      );

      let y_lo_pair = _mm_cvtepu16_epi32(y_vec);
      let y_hi_pair = _mm_cvtepu16_epi32(_mm_srli_si128::<8>(y_vec));
      let y_lo_sub = _mm_sub_epi32(y_lo_pair, y_off_v);
      let y_hi_sub = _mm_sub_epi32(y_hi_pair, y_off_v);
      let y_lo_even = scale_y16_i64(y_lo_sub, y_scale_v, rnd_v);
      let y_lo_odd = scale_y16_i64(_mm_shuffle_epi32::<0xF5>(y_lo_sub), y_scale_v, rnd_v);
      let y_hi_even = scale_y16_i64(y_hi_sub, y_scale_v, rnd_v);
      let y_hi_odd = scale_y16_i64(_mm_shuffle_epi32::<0xF5>(y_hi_sub), y_scale_v, rnd_v);
      let y_lo_i32 = _mm_unpacklo_epi64(
        _mm_unpacklo_epi32(y_lo_even, y_lo_odd),
        _mm_unpackhi_epi32(y_lo_even, y_lo_odd),
      );
      let y_hi_i32 = _mm_unpacklo_epi64(
        _mm_unpacklo_epi32(y_hi_even, y_hi_odd),
        _mm_unpackhi_epi32(y_hi_even, y_hi_odd),
      );

      let r_u16 = _mm_packus_epi32(
        _mm_add_epi32(y_lo_i32, r_ch_lo_i32),
        _mm_add_epi32(y_hi_i32, r_ch_hi_i32),
      );
      let g_u16 = _mm_packus_epi32(
        _mm_add_epi32(y_lo_i32, g_ch_lo_i32),
        _mm_add_epi32(y_hi_i32, g_ch_hi_i32),
      );
      let b_u16 = _mm_packus_epi32(
        _mm_add_epi32(y_lo_i32, b_ch_lo_i32),
        _mm_add_epi32(y_hi_i32, b_ch_hi_i32),
      );

      if ALPHA {
        write_rgba_u16_8(r_u16, g_u16, b_u16, alpha_u16, out.as_mut_ptr().add(x * 4));
      } else {
        write_rgb_u16_8(r_u16, g_u16, b_u16, out.as_mut_ptr().add(x * 3));
      }
      x += 8;
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
