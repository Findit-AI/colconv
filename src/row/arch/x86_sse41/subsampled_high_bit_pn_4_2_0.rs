use core::arch::x86_64::*;

use super::*;

/// SSE4.1 high‑bit‑packed semi‑planar (`BITS` ∈ {10, 12}) → packed
/// **8‑bit** RGB.
///
/// Block size 16 Y pixels / 8 chroma pairs per iteration. Differences
/// from [`super::x86_sse41::yuv_420p_n_to_rgb_row`]:
/// - Samples are shifted right by `16 - BITS` (`_mm_srl_epi16`, with
///   a shift count computed from `BITS` once per call) instead of
///   AND‑masked — Pn's `BITS` active bits live in the HIGH `BITS` of
///   each `u16`.
/// - Semi‑planar UV is deinterleaved via [`deinterleave_uv_u16`]
///   below (one `_mm_shuffle_epi8` + two 64‑bit unpacks per 16
///   chroma elements).
///
/// # Numerical contract
///
/// Byte‑identical to [`scalar::p_n_to_rgb_row::<BITS>`] for the
/// monomorphized `BITS`.
///
/// # Safety
///
/// 1. **SSE4.1 must be available on the current CPU.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`,
///    `rgb_out.len() >= 3 * width`.
///
/// Thin wrapper over [`p_n_to_rgb_or_rgba_row`] with `ALPHA = false`.
#[inline]
#[target_feature(enable = "sse4.1")]
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

/// SSE4.1 high-bit-packed semi-planar 4:2:0 → packed **8-bit RGBA**
/// (`R, G, B, 0xFF`).
///
/// Thin wrapper over [`p_n_to_rgb_or_rgba_row`] with `ALPHA = true`.
///
/// # Safety
///
/// Same as [`p_n_to_rgb_row`] but `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "sse4.1")]
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

/// Shared SSE4.1 P010/P012 kernel for [`p_n_to_rgb_row`] (`ALPHA = false`,
/// `write_rgb_16`) and [`p_n_to_rgba_row`] (`ALPHA = true`, `write_rgba_16`
/// with constant `0xFF` alpha).
///
/// # Safety
///
/// 1. **SSE4.1 must be available on the current CPU.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
/// 4. `BITS` ∈ `{10, 12}` — P016 has its own kernel family.
#[inline]
#[target_feature(enable = "sse4.1")]
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

  // SAFETY: SSE4.1 availability is the caller's obligation.
  unsafe {
    let rnd_v = _mm_set1_epi32(RND);
    let y_off_v = _mm_set1_epi16(y_off as i16);
    let y_scale_v = _mm_set1_epi32(y_scale);
    let c_scale_v = _mm_set1_epi32(c_scale);
    let bias_v = _mm_set1_epi16(bias as i16);
    // High-bit-packed samples: shift right by `16 - BITS` to extract
    // the BITS-bit value. Loop-invariant, loaded once into the low 64b
    // of `shr_count` for `_mm_srl_epi16`.
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
      // Y: two u16×8 loads, each shifted right by `16 - BITS`.
      let y_low_i16 = _mm_srl_epi16(_mm_loadu_si128(y.as_ptr().add(x).cast()), shr_count);
      let y_high_i16 = _mm_srl_epi16(_mm_loadu_si128(y.as_ptr().add(x + 8).cast()), shr_count);

      // UV: two u16×8 loads of interleaved [U0,V0,U1,V1,...], then
      // deinterleave into separate u_vec + v_vec.
      let (u_vec, v_vec) = deinterleave_uv_u16(uv_half.as_ptr().add(x));
      let u_vec = _mm_srl_epi16(u_vec, shr_count);
      let v_vec = _mm_srl_epi16(v_vec, shr_count);

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
        write_rgba_16(r_u8, g_u8, b_u8, alpha_u8, out.as_mut_ptr().add(x * 4));
      } else {
        write_rgb_16(r_u8, g_u8, b_u8, out.as_mut_ptr().add(x * 3));
      }

      x += 16;
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

/// SSE4.1 high‑bit‑packed semi‑planar (`BITS` ∈ {10, 12}) → packed
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
/// 1. **SSE4.1 must be available on the current CPU.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`,
///    `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "sse4.1")]
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

/// SSE4.1 sibling of [`p_n_to_rgba_row`] for native-depth `u16` output.
/// Alpha samples are `(1 << BITS) - 1` (opaque maximum at the input
/// bit depth). P016 has its own kernel family — never routed here.
///
/// # Safety
///
/// Same as [`p_n_to_rgb_u16_row`] plus `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "sse4.1")]
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

/// Shared SSE4.1 Pn → native-depth `u16` kernel. `ALPHA = false`
/// writes RGB triples via `write_rgb_u16_8`; `ALPHA = true` writes
/// RGBA quads via `write_rgba_u16_8` with constant alpha
/// `(1 << BITS) - 1`. P016 has its own kernel family — never routed
/// here.
///
/// # Safety
///
/// 1. **SSE4.1 must be available on the current CPU.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
/// 4. `BITS` ∈ `{10, 12}`.
#[inline]
#[target_feature(enable = "sse4.1")]
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

  // SAFETY: SSE4.1 availability is the caller's obligation.
  unsafe {
    let rnd_v = _mm_set1_epi32(RND);
    let y_off_v = _mm_set1_epi16(y_off as i16);
    let y_scale_v = _mm_set1_epi32(y_scale);
    let c_scale_v = _mm_set1_epi32(c_scale);
    let bias_v = _mm_set1_epi16(bias as i16);
    let max_v = _mm_set1_epi16(out_max);
    let zero_v = _mm_set1_epi16(0);
    // High-bit-packed samples: shift right by `16 - BITS` to extract
    // the BITS-bit value.
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
      let (u_vec, v_vec) = deinterleave_uv_u16(uv_half.as_ptr().add(x));
      let u_vec = _mm_srl_epi16(u_vec, shr_count);
      let v_vec = _mm_srl_epi16(v_vec, shr_count);

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

      let r_lo = clamp_u16_max(_mm_adds_epi16(y_scaled_lo, r_dup_lo), zero_v, max_v);
      let r_hi = clamp_u16_max(_mm_adds_epi16(y_scaled_hi, r_dup_hi), zero_v, max_v);
      let g_lo = clamp_u16_max(_mm_adds_epi16(y_scaled_lo, g_dup_lo), zero_v, max_v);
      let g_hi = clamp_u16_max(_mm_adds_epi16(y_scaled_hi, g_dup_hi), zero_v, max_v);
      let b_lo = clamp_u16_max(_mm_adds_epi16(y_scaled_lo, b_dup_lo), zero_v, max_v);
      let b_hi = clamp_u16_max(_mm_adds_epi16(y_scaled_hi, b_dup_hi), zero_v, max_v);

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
// ===== 16-bit semi-planar (P016) → RGB ===================================

/// SSE4.1 P016 → packed **8-bit** RGB. Thin wrapper: deinterleaves UV
/// via [`deinterleave_uv_u16`] then delegates to the shared planar kernel.
///
/// # Safety
///
/// 1. **SSE4.1 must be available on the current CPU.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`, `rgb_out.len() >= 3 * width`.
///
/// Thin wrapper over [`p16_to_rgb_or_rgba_row`] with `ALPHA = false`.
#[inline]
#[target_feature(enable = "sse4.1")]
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

/// SSE4.1 P016 → packed **8-bit RGBA** (`R, G, B, 0xFF`).
///
/// Thin wrapper over [`p16_to_rgb_or_rgba_row`] with `ALPHA = true`.
///
/// # Safety
///
/// Same as [`p16_to_rgb_row`] but `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "sse4.1")]
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

/// Shared SSE4.1 P016 kernel for [`p16_to_rgb_row`] (`ALPHA = false`,
/// `write_rgb_16`) and [`p16_to_rgba_row`] (`ALPHA = true`,
/// `write_rgba_16` with constant `0xFF` alpha).
///
/// # Safety
///
/// 1. **SSE4.1 must be available on the current CPU.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
#[inline]
#[target_feature(enable = "sse4.1")]
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
      let (u_vec, v_vec) = deinterleave_uv_u16(uv_half.as_ptr().add(x));

      let u_i16 = _mm_sub_epi16(u_vec, bias16_v);
      let v_i16 = _mm_sub_epi16(v_vec, bias16_v);

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

      let y_scaled_lo = scale_y_u16(y_low, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y_u16(y_high, y_off_v, y_scale_v, rnd_v);

      let r_lo = _mm_adds_epi16(y_scaled_lo, r_dup_lo);
      let r_hi = _mm_adds_epi16(y_scaled_hi, r_dup_hi);
      let g_lo = _mm_adds_epi16(y_scaled_lo, g_dup_lo);
      let g_hi = _mm_adds_epi16(y_scaled_hi, g_dup_hi);
      let b_lo = _mm_adds_epi16(y_scaled_lo, b_dup_lo);
      let b_hi = _mm_adds_epi16(y_scaled_hi, b_dup_hi);

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

/// SSE4.1 P016 → packed **16-bit** RGB. i64 chroma arithmetic, 8 pixels per iteration.
///
/// # Safety
///
/// Same as [`p16_to_rgb_row`] but `rgb_out` is `&mut [u16]`.
#[inline]
#[target_feature(enable = "sse4.1")]
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

/// SSE4.1 sibling of [`p16_to_rgba_row`] for native-depth `u16` output.
/// Alpha is `0xFFFF`.
///
/// # Safety
///
/// Same as [`p16_to_rgb_u16_row`] plus `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "sse4.1")]
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

/// Shared SSE4.1 16-bit P016 → native-depth `u16` kernel.
/// `ALPHA = false` writes RGB triples; `ALPHA = true` writes RGBA
/// quads with constant alpha `0xFFFF`.
///
/// # Safety
///
/// 1. **SSE4.1 must be available on the current CPU.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
#[inline]
#[target_feature(enable = "sse4.1")]
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

    let mut x = 0usize;
    while x + 8 <= width {
      let y_vec = _mm_loadu_si128(y.as_ptr().add(x).cast());
      // Load 4 UV pairs = 8 u16 = 16 bytes; deinterleave inline.
      // uv_half.len() >= width >= x + 8 guarantees 8 u16 readable.
      let uv_raw = _mm_loadu_si128(uv_half.as_ptr().add(x).cast());
      // [U0,V0,U1,V1,U2,V2,U3,V3] → [U0,U1,U2,U3, V0,V1,V2,V3]
      let split_mask = _mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15);
      let uv_split = _mm_shuffle_epi8(uv_raw, split_mask);
      let u_vec4 = uv_split;
      let v_vec4 = _mm_srli_si128::<8>(uv_split);

      let u_i16 = _mm_sub_epi16(u_vec4, bias16_v);
      let v_i16 = _mm_sub_epi16(v_vec4, bias16_v);

      let rnd32_v = _mm_set1_epi32(1 << 14);
      let u_i32 = _mm_cvtepi16_epi32(u_i16);
      let v_i32 = _mm_cvtepi16_epi32(v_i16);
      let u_d = q15_shift(_mm_add_epi32(_mm_mullo_epi32(u_i32, c_scale_v), rnd32_v));
      let v_d = q15_shift(_mm_add_epi32(_mm_mullo_epi32(v_i32, c_scale_v), rnd32_v));

      let u_d_even = u_d;
      let v_d_even = v_d;
      let u_d_odd = _mm_shuffle_epi32::<0xF5>(u_d);
      let v_d_odd = _mm_shuffle_epi32::<0xF5>(v_d);

      let r_ch_even = chroma_i64x2(cru, crv, u_d_even, v_d_even, rnd_v);
      let r_ch_odd = chroma_i64x2(cru, crv, u_d_odd, v_d_odd, rnd_v);
      let g_ch_even = chroma_i64x2(cgu, cgv, u_d_even, v_d_even, rnd_v);
      let g_ch_odd = chroma_i64x2(cgu, cgv, u_d_odd, v_d_odd, rnd_v);
      let b_ch_even = chroma_i64x2(cbu, cbv, u_d_even, v_d_even, rnd_v);
      let b_ch_odd = chroma_i64x2(cbu, cbv, u_d_odd, v_d_odd, rnd_v);

      let r_ch_i32 = _mm_unpacklo_epi64(
        _mm_unpacklo_epi32(r_ch_even, r_ch_odd),
        _mm_unpackhi_epi32(r_ch_even, r_ch_odd),
      );
      let g_ch_i32 = _mm_unpacklo_epi64(
        _mm_unpacklo_epi32(g_ch_even, g_ch_odd),
        _mm_unpackhi_epi32(g_ch_even, g_ch_odd),
      );
      let b_ch_i32 = _mm_unpacklo_epi64(
        _mm_unpacklo_epi32(b_ch_even, b_ch_odd),
        _mm_unpackhi_epi32(b_ch_even, b_ch_odd),
      );

      let r_dup_lo = _mm_unpacklo_epi32(r_ch_i32, r_ch_i32);
      let r_dup_hi = _mm_unpackhi_epi32(r_ch_i32, r_ch_i32);
      let g_dup_lo = _mm_unpacklo_epi32(g_ch_i32, g_ch_i32);
      let g_dup_hi = _mm_unpackhi_epi32(g_ch_i32, g_ch_i32);
      let b_dup_lo = _mm_unpacklo_epi32(b_ch_i32, b_ch_i32);
      let b_dup_hi = _mm_unpackhi_epi32(b_ch_i32, b_ch_i32);

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
        _mm_add_epi32(y_lo_i32, r_dup_lo),
        _mm_add_epi32(y_hi_i32, r_dup_hi),
      );
      let g_u16 = _mm_packus_epi32(
        _mm_add_epi32(y_lo_i32, g_dup_lo),
        _mm_add_epi32(y_hi_i32, g_dup_hi),
      );
      let b_u16 = _mm_packus_epi32(
        _mm_add_epi32(y_lo_i32, b_dup_lo),
        _mm_add_epi32(y_hi_i32, b_dup_hi),
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
