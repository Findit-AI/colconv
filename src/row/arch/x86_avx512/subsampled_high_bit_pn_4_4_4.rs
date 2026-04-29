use core::arch::x86_64::*;

use super::*;

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
/// Thin wrapper over [`p_n_444_to_rgb_or_rgba_row`] with `ALPHA = false`.
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
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    p_n_444_to_rgb_or_rgba_row::<BITS, false>(y, uv_full, rgb_out, width, matrix, full_range);
  }
}

/// AVX-512 Pn 4:4:4 high-bit-packed (BITS ∈ {10, 12}) → packed **8-bit
/// RGBA** (`R, G, B, 0xFF`).
///
/// Thin wrapper over [`p_n_444_to_rgb_or_rgba_row`] with `ALPHA = true`.
///
/// # Safety
///
/// Same as [`p_n_444_to_rgb_row`] but `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "avx512bw,avx512f")]
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

/// Shared AVX-512 Pn 4:4:4 high-bit-packed kernel for
/// [`p_n_444_to_rgb_row`] (`ALPHA = false`, `write_rgb_64`) and
/// [`p_n_444_to_rgba_row`] (`ALPHA = true`, `write_rgba_64` with
/// constant `0xFF` alpha).
///
/// # Safety
///
/// 1. AVX-512F + AVX-512BW must be available on the current CPU.
/// 2. `y.len() >= width`, `uv_full.len() >= 2 * width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
/// 3. `BITS` must be one of `{10, 12}`.
#[inline]
#[target_feature(enable = "avx512bw,avx512f")]
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
    let alpha_u8 = _mm512_set1_epi8(-1);

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

      if ALPHA {
        write_rgba_64(r_u8, g_u8, b_u8, alpha_u8, out.as_mut_ptr().add(x * 4));
      } else {
        write_rgb_64(r_u8, g_u8, b_u8, out.as_mut_ptr().add(x * 3));
      }

      x += 64;
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

/// AVX-512 Pn 4:4:4 high-bit-packed (BITS ∈ {10, 12}) → packed
/// **native-depth `u16`** RGB. 64 pixels per iter.
///
/// Thin wrapper over [`p_n_444_to_rgb_or_rgba_u16_row`] with `ALPHA = false`.
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
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    p_n_444_to_rgb_or_rgba_u16_row::<BITS, false>(y, uv_full, rgb_out, width, matrix, full_range);
  }
}

/// AVX-512 sibling of [`p_n_444_to_rgba_row`] for native-depth `u16`
/// output. Alpha samples are `(1 << BITS) - 1`.
///
/// Thin wrapper over [`p_n_444_to_rgb_or_rgba_u16_row`] with `ALPHA = true`.
///
/// # Safety
///
/// Same as [`p_n_444_to_rgb_u16_row`] plus `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "avx512bw,avx512f")]
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

/// Shared AVX-512 Pn 4:4:4 high-bit-packed → native-depth `u16` kernel.
/// `ALPHA = false` writes RGB triples via 8× `write_quarter`;
/// `ALPHA = true` writes RGBA quads via 8× `write_quarter_rgba` with
/// constant alpha `(1 << BITS) - 1`.
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available.**
/// 2. `y.len() >= width`, `uv_full.len() >= 2 * width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
/// 3. `BITS` ∈ `{10, 12}`.
#[inline]
#[target_feature(enable = "avx512bw,avx512f")]
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
    let alpha_u16 = _mm_set1_epi16(out_max);

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

      if ALPHA {
        let dst = out.as_mut_ptr().add(x * 4);
        write_quarter_rgba(r_lo, g_lo, b_lo, alpha_u16, 0, dst);
        write_quarter_rgba(r_lo, g_lo, b_lo, alpha_u16, 1, dst.add(32));
        write_quarter_rgba(r_lo, g_lo, b_lo, alpha_u16, 2, dst.add(64));
        write_quarter_rgba(r_lo, g_lo, b_lo, alpha_u16, 3, dst.add(96));
        write_quarter_rgba(r_hi, g_hi, b_hi, alpha_u16, 0, dst.add(128));
        write_quarter_rgba(r_hi, g_hi, b_hi, alpha_u16, 1, dst.add(160));
        write_quarter_rgba(r_hi, g_hi, b_hi, alpha_u16, 2, dst.add(192));
        write_quarter_rgba(r_hi, g_hi, b_hi, alpha_u16, 3, dst.add(224));
      } else {
        let dst = out.as_mut_ptr().add(x * 3);
        write_quarter(r_lo, g_lo, b_lo, 0, dst);
        write_quarter(r_lo, g_lo, b_lo, 1, dst.add(24));
        write_quarter(r_lo, g_lo, b_lo, 2, dst.add(48));
        write_quarter(r_lo, g_lo, b_lo, 3, dst.add(72));
        write_quarter(r_hi, g_hi, b_hi, 0, dst.add(96));
        write_quarter(r_hi, g_hi, b_hi, 1, dst.add(120));
        write_quarter(r_hi, g_hi, b_hi, 2, dst.add(144));
        write_quarter(r_hi, g_hi, b_hi, 3, dst.add(168));
      }

      x += 64;
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

/// AVX-512 P416 (semi-planar 4:4:4, 16-bit) → packed **u8** RGB.
/// 64 pixels per iter; Y stays on i32.
///
/// Thin wrapper over [`p_n_444_16_to_rgb_or_rgba_row`] with `ALPHA = false`.
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
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    p_n_444_16_to_rgb_or_rgba_row::<false>(y, uv_full, rgb_out, width, matrix, full_range);
  }
}

/// AVX-512 P416 (semi-planar 4:4:4, 16-bit) → packed **8-bit RGBA**
/// (`R, G, B, 0xFF`).
///
/// Thin wrapper over [`p_n_444_16_to_rgb_or_rgba_row`] with `ALPHA = true`.
///
/// # Safety
///
/// Same as [`p_n_444_16_to_rgb_row`] but `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "avx512bw,avx512f")]
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

/// Shared AVX-512 P416 kernel for [`p_n_444_16_to_rgb_row`]
/// (`ALPHA = false`, `write_rgb_64`) and [`p_n_444_16_to_rgba_row`]
/// (`ALPHA = true`, `write_rgba_64` with constant `0xFF` alpha).
///
/// # Safety
///
/// 1. AVX-512F + AVX-512BW must be available.
/// 2. `y.len() >= width`, `uv_full.len() >= 2 * width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
#[inline]
#[target_feature(enable = "avx512bw,avx512f")]
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
    let alpha_u8 = _mm512_set1_epi8(-1);

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

      if ALPHA {
        write_rgba_64(r_u8, g_u8, b_u8, alpha_u8, out.as_mut_ptr().add(x * 4));
      } else {
        write_rgb_64(r_u8, g_u8, b_u8, out.as_mut_ptr().add(x * 3));
      }
      x += 64;
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

/// AVX-512 P416 → packed **native-depth `u16`** RGB. 32 pixels per
/// iter (i64 narrows). Native `_mm512_srai_epi64` via
/// `chroma_i64x8_avx512` + `scale_y_i32x16_i64` — no bias trick.
///
/// Thin wrapper over [`p_n_444_16_to_rgb_or_rgba_u16_row`] with `ALPHA = false`.
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
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    p_n_444_16_to_rgb_or_rgba_u16_row::<false>(y, uv_full, rgb_out, width, matrix, full_range);
  }
}

/// AVX-512 sibling of [`p_n_444_16_to_rgba_row`] for native-depth `u16`
/// output. Alpha samples are `0xFFFF`.
///
/// Thin wrapper over [`p_n_444_16_to_rgb_or_rgba_u16_row`] with `ALPHA = true`.
///
/// # Safety
///
/// Same as [`p_n_444_16_to_rgb_u16_row`] plus `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "avx512bw,avx512f")]
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

/// Shared AVX-512 P416 (semi-planar 4:4:4, 16-bit) → native-depth `u16`
/// kernel. `ALPHA = false` writes RGB triples; `ALPHA = true` writes
/// RGBA quads with constant alpha `0xFFFF`.
///
/// # Safety
///
/// 1. **AVX-512F + AVX-512BW must be available.**
/// 2. `y.len() >= width`, `uv_full.len() >= 2 * width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
#[inline]
#[target_feature(enable = "avx512bw,avx512f")]
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
  const RND_I64: i64 = 1 << 14;
  const RND_I32: i32 = 1 << 14;

  unsafe {
    let alpha_u16 = _mm_set1_epi16(-1i16);
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

      if ALPHA {
        write_rgba_u16_32(r_u16, g_u16, b_u16, alpha_u16, out.as_mut_ptr().add(x * 4));
      } else {
        write_rgb_u16_32(r_u16, g_u16, b_u16, out.as_mut_ptr().add(x * 3));
      }
      x += 32;
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
