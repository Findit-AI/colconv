use core::arch::wasm32::*;

use super::*;

// ===== Pn 4:4:4 (semi-planar high-bit-packed) → RGB =======================
//
// Native wasm simd128 4:4:4 Pn kernels — combine `yuv_444p_n_to_rgb_row`'s
// 1:1 chroma compute with `p_n_to_rgb_row`'s `deinterleave_uv_u16_wasm`
// pattern. 16 Y pixels per iter for the i32 Q15 paths; 8 for the
// i64 chroma u16-output path (matches `yuv_444p16_to_rgb_u16_row`).

/// wasm simd128 Pn 4:4:4 high-bit-packed (BITS ∈ {10, 12}) → packed
/// **u8** RGB.
///
/// Thin wrapper over [`p_n_444_to_rgb_or_rgba_row`] with `ALPHA = false`.
///
/// # Safety
///
/// 1. simd128 must be enabled at compile time.
/// 2. `y.len() >= width`, `uv_full.len() >= 2 * width`,
///    `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "simd128")]
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

/// wasm simd128 Pn 4:4:4 high-bit-packed (BITS ∈ {10, 12}) → packed
/// **8-bit RGBA** (`R, G, B, 0xFF`).
///
/// Thin wrapper over [`p_n_444_to_rgb_or_rgba_row`] with `ALPHA = true`.
///
/// # Safety
///
/// Same as [`p_n_444_to_rgb_row`] but `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "simd128")]
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

/// Shared wasm simd128 Pn 4:4:4 high-bit-packed kernel for
/// [`p_n_444_to_rgb_row`] (`ALPHA = false`, `write_rgb_16`) and
/// [`p_n_444_to_rgba_row`] (`ALPHA = true`, `write_rgba_16` with
/// constant `0xFF` alpha).
///
/// # Safety
///
/// 1. simd128 must be enabled at compile time.
/// 2. `y.len() >= width`, `uv_full.len() >= 2 * width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
/// 3. `BITS` must be one of `{10, 12}`.
#[inline]
#[target_feature(enable = "simd128")]
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
    let rnd_v = i32x4_splat(RND);
    let y_off_v = i16x8_splat(y_off as i16);
    let y_scale_v = i32x4_splat(y_scale);
    let c_scale_v = i32x4_splat(c_scale);
    let bias_v = i16x8_splat(bias as i16);
    let cru = i32x4_splat(coeffs.r_u());
    let crv = i32x4_splat(coeffs.r_v());
    let cgu = i32x4_splat(coeffs.g_u());
    let cgv = i32x4_splat(coeffs.g_v());
    let cbu = i32x4_splat(coeffs.b_u());
    let cbv = i32x4_splat(coeffs.b_v());
    let alpha_u8 = u8x16_splat(0xFF);

    let shr = 16 - BITS;

    let mut x = 0usize;
    while x + 16 <= width {
      let y_low_i16 = u16x8_shr(v128_load(y.as_ptr().add(x).cast()), shr);
      let y_high_i16 = u16x8_shr(v128_load(y.as_ptr().add(x + 8).cast()), shr);

      // 32 UV elements (= 16 pairs) — two deinterleave calls.
      let (u_lo_vec, v_lo_vec) = deinterleave_uv_u16_wasm(uv_full.as_ptr().add(x * 2));
      let (u_hi_vec, v_hi_vec) = deinterleave_uv_u16_wasm(uv_full.as_ptr().add(x * 2 + 16));
      let u_lo_vec = u16x8_shr(u_lo_vec, shr);
      let v_lo_vec = u16x8_shr(v_lo_vec, shr);
      let u_hi_vec = u16x8_shr(u_hi_vec, shr);
      let v_hi_vec = u16x8_shr(v_hi_vec, shr);

      let u_lo_i16 = i16x8_sub(u_lo_vec, bias_v);
      let u_hi_i16 = i16x8_sub(u_hi_vec, bias_v);
      let v_lo_i16 = i16x8_sub(v_lo_vec, bias_v);
      let v_hi_i16 = i16x8_sub(v_hi_vec, bias_v);

      let u_lo_a = i32x4_extend_low_i16x8(u_lo_i16);
      let u_lo_b = i32x4_extend_high_i16x8(u_lo_i16);
      let u_hi_a = i32x4_extend_low_i16x8(u_hi_i16);
      let u_hi_b = i32x4_extend_high_i16x8(u_hi_i16);
      let v_lo_a = i32x4_extend_low_i16x8(v_lo_i16);
      let v_lo_b = i32x4_extend_high_i16x8(v_lo_i16);
      let v_hi_a = i32x4_extend_low_i16x8(v_hi_i16);
      let v_hi_b = i32x4_extend_high_i16x8(v_hi_i16);

      let u_d_lo_a = q15_shift(i32x4_add(i32x4_mul(u_lo_a, c_scale_v), rnd_v));
      let u_d_lo_b = q15_shift(i32x4_add(i32x4_mul(u_lo_b, c_scale_v), rnd_v));
      let u_d_hi_a = q15_shift(i32x4_add(i32x4_mul(u_hi_a, c_scale_v), rnd_v));
      let u_d_hi_b = q15_shift(i32x4_add(i32x4_mul(u_hi_b, c_scale_v), rnd_v));
      let v_d_lo_a = q15_shift(i32x4_add(i32x4_mul(v_lo_a, c_scale_v), rnd_v));
      let v_d_lo_b = q15_shift(i32x4_add(i32x4_mul(v_lo_b, c_scale_v), rnd_v));
      let v_d_hi_a = q15_shift(i32x4_add(i32x4_mul(v_hi_a, c_scale_v), rnd_v));
      let v_d_hi_b = q15_shift(i32x4_add(i32x4_mul(v_hi_b, c_scale_v), rnd_v));

      let r_chroma_lo = chroma_i16x8(cru, crv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let r_chroma_hi = chroma_i16x8(cru, crv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let g_chroma_lo = chroma_i16x8(cgu, cgv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let g_chroma_hi = chroma_i16x8(cgu, cgv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let b_chroma_lo = chroma_i16x8(cbu, cbv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let b_chroma_hi = chroma_i16x8(cbu, cbv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);

      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v);

      let r_lo = i16x8_add_sat(y_scaled_lo, r_chroma_lo);
      let r_hi = i16x8_add_sat(y_scaled_hi, r_chroma_hi);
      let g_lo = i16x8_add_sat(y_scaled_lo, g_chroma_lo);
      let g_hi = i16x8_add_sat(y_scaled_hi, g_chroma_hi);
      let b_lo = i16x8_add_sat(y_scaled_lo, b_chroma_lo);
      let b_hi = i16x8_add_sat(y_scaled_hi, b_chroma_hi);

      let b_u8 = u8x16_narrow_i16x8(b_lo, b_hi);
      let g_u8 = u8x16_narrow_i16x8(g_lo, g_hi);
      let r_u8 = u8x16_narrow_i16x8(r_lo, r_hi);

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

/// wasm simd128 Pn 4:4:4 high-bit-packed (BITS ∈ {10, 12}) → packed
/// **native-depth `u16`** RGB.
///
/// Thin wrapper over [`p_n_444_to_rgb_or_rgba_u16_row`] with `ALPHA = false`.
///
/// # Safety
///
/// Same as [`p_n_444_to_rgb_row`] but `rgb_out: &mut [u16]`.
#[inline]
#[target_feature(enable = "simd128")]
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

/// wasm simd128 sibling of [`p_n_444_to_rgba_row`] for native-depth
/// `u16` output. Alpha samples are `(1 << BITS) - 1` (opaque maximum
/// at the input bit depth).
///
/// # Safety
///
/// Same as [`p_n_444_to_rgb_u16_row`] but `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "simd128")]
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

/// Shared wasm simd128 Pn 4:4:4 high-bit-packed → native-depth `u16`
/// kernel. `ALPHA = false` writes RGB triples via `write_rgb_u16_8`;
/// `ALPHA = true` writes RGBA quads via `write_rgba_u16_8` with
/// constant alpha `(1 << BITS) - 1`.
///
/// # Safety
///
/// 1. simd128 must be enabled at compile time.
/// 2. `y.len() >= width`, `uv_full.len() >= 2 * width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
/// 3. `BITS` must be one of `{10, 12}`.
#[inline]
#[target_feature(enable = "simd128")]
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
    let rnd_v = i32x4_splat(RND);
    let y_off_v = i16x8_splat(y_off as i16);
    let y_scale_v = i32x4_splat(y_scale);
    let c_scale_v = i32x4_splat(c_scale);
    let bias_v = i16x8_splat(bias as i16);
    let max_v = i16x8_splat(out_max);
    let zero_v = i16x8_splat(0);
    let cru = i32x4_splat(coeffs.r_u());
    let crv = i32x4_splat(coeffs.r_v());
    let cgu = i32x4_splat(coeffs.g_u());
    let cgv = i32x4_splat(coeffs.g_v());
    let cbu = i32x4_splat(coeffs.b_u());
    let cbv = i32x4_splat(coeffs.b_v());
    let alpha_u16 = u16x8_splat(out_max as u16);

    let shr = 16 - BITS;

    let mut x = 0usize;
    while x + 16 <= width {
      let y_low_i16 = u16x8_shr(v128_load(y.as_ptr().add(x).cast()), shr);
      let y_high_i16 = u16x8_shr(v128_load(y.as_ptr().add(x + 8).cast()), shr);

      let (u_lo_vec, v_lo_vec) = deinterleave_uv_u16_wasm(uv_full.as_ptr().add(x * 2));
      let (u_hi_vec, v_hi_vec) = deinterleave_uv_u16_wasm(uv_full.as_ptr().add(x * 2 + 16));
      let u_lo_vec = u16x8_shr(u_lo_vec, shr);
      let v_lo_vec = u16x8_shr(v_lo_vec, shr);
      let u_hi_vec = u16x8_shr(u_hi_vec, shr);
      let v_hi_vec = u16x8_shr(v_hi_vec, shr);

      let u_lo_i16 = i16x8_sub(u_lo_vec, bias_v);
      let u_hi_i16 = i16x8_sub(u_hi_vec, bias_v);
      let v_lo_i16 = i16x8_sub(v_lo_vec, bias_v);
      let v_hi_i16 = i16x8_sub(v_hi_vec, bias_v);

      let u_lo_a = i32x4_extend_low_i16x8(u_lo_i16);
      let u_lo_b = i32x4_extend_high_i16x8(u_lo_i16);
      let u_hi_a = i32x4_extend_low_i16x8(u_hi_i16);
      let u_hi_b = i32x4_extend_high_i16x8(u_hi_i16);
      let v_lo_a = i32x4_extend_low_i16x8(v_lo_i16);
      let v_lo_b = i32x4_extend_high_i16x8(v_lo_i16);
      let v_hi_a = i32x4_extend_low_i16x8(v_hi_i16);
      let v_hi_b = i32x4_extend_high_i16x8(v_hi_i16);

      let u_d_lo_a = q15_shift(i32x4_add(i32x4_mul(u_lo_a, c_scale_v), rnd_v));
      let u_d_lo_b = q15_shift(i32x4_add(i32x4_mul(u_lo_b, c_scale_v), rnd_v));
      let u_d_hi_a = q15_shift(i32x4_add(i32x4_mul(u_hi_a, c_scale_v), rnd_v));
      let u_d_hi_b = q15_shift(i32x4_add(i32x4_mul(u_hi_b, c_scale_v), rnd_v));
      let v_d_lo_a = q15_shift(i32x4_add(i32x4_mul(v_lo_a, c_scale_v), rnd_v));
      let v_d_lo_b = q15_shift(i32x4_add(i32x4_mul(v_lo_b, c_scale_v), rnd_v));
      let v_d_hi_a = q15_shift(i32x4_add(i32x4_mul(v_hi_a, c_scale_v), rnd_v));
      let v_d_hi_b = q15_shift(i32x4_add(i32x4_mul(v_hi_b, c_scale_v), rnd_v));

      let r_chroma_lo = chroma_i16x8(cru, crv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let r_chroma_hi = chroma_i16x8(cru, crv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let g_chroma_lo = chroma_i16x8(cgu, cgv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let g_chroma_hi = chroma_i16x8(cgu, cgv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let b_chroma_lo = chroma_i16x8(cbu, cbv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let b_chroma_hi = chroma_i16x8(cbu, cbv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);

      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v);

      // Saturating i16 add: y_scaled + chroma can exceed i16 range
      // for near-max samples; wrapping `i16x8_add` would silently flip
      // sign and clamp to 0. `i16x8_add_sat` saturates to i16::MAX,
      // then `clamp_u16_max_wasm` produces the correct out_max.
      // Matches the existing wasm u8 / u16 kernels' convention.
      let r_lo = clamp_u16_max_wasm(i16x8_add_sat(y_scaled_lo, r_chroma_lo), zero_v, max_v);
      let r_hi = clamp_u16_max_wasm(i16x8_add_sat(y_scaled_hi, r_chroma_hi), zero_v, max_v);
      let g_lo = clamp_u16_max_wasm(i16x8_add_sat(y_scaled_lo, g_chroma_lo), zero_v, max_v);
      let g_hi = clamp_u16_max_wasm(i16x8_add_sat(y_scaled_hi, g_chroma_hi), zero_v, max_v);
      let b_lo = clamp_u16_max_wasm(i16x8_add_sat(y_scaled_lo, b_chroma_lo), zero_v, max_v);
      let b_hi = clamp_u16_max_wasm(i16x8_add_sat(y_scaled_hi, b_chroma_hi), zero_v, max_v);

      if ALPHA {
        let dst = out.as_mut_ptr().add(x * 4);
        write_rgba_u16_8(r_lo, g_lo, b_lo, alpha_u16, dst);
        write_rgba_u16_8(r_hi, g_hi, b_hi, alpha_u16, dst.add(32));
      } else {
        let dst = out.as_mut_ptr().add(x * 3);
        write_rgb_u16_8(r_lo, g_lo, b_lo, dst);
        write_rgb_u16_8(r_hi, g_hi, b_hi, dst.add(24));
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

/// wasm simd128 P416 (semi-planar 4:4:4, 16-bit) → packed **u8** RGB.
/// 16 pixels per iter.
///
/// Thin wrapper over [`p_n_444_16_to_rgb_or_rgba_row`] with `ALPHA = false`.
///
/// # Safety
///
/// 1. simd128 must be enabled at compile time.
/// 2. `y.len() >= width`, `uv_full.len() >= 2 * width`,
///    `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "simd128")]
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

/// wasm simd128 P416 (semi-planar 4:4:4, 16-bit) → packed **8-bit
/// RGBA** (`R, G, B, 0xFF`).
///
/// Thin wrapper over [`p_n_444_16_to_rgb_or_rgba_row`] with `ALPHA = true`.
///
/// # Safety
///
/// Same as [`p_n_444_16_to_rgb_row`] but `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "simd128")]
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

/// Shared wasm simd128 P416 kernel for [`p_n_444_16_to_rgb_row`]
/// (`ALPHA = false`, `write_rgb_16`) and [`p_n_444_16_to_rgba_row`]
/// (`ALPHA = true`, `write_rgba_16` with constant `0xFF` alpha).
///
/// # Safety
///
/// 1. simd128 must be enabled at compile time.
/// 2. `y.len() >= width`, `uv_full.len() >= 2 * width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
#[inline]
#[target_feature(enable = "simd128")]
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
    let rnd_v = i32x4_splat(RND);
    let y_off32_v = i32x4_splat(y_off);
    let y_scale_v = i32x4_splat(y_scale);
    let c_scale_v = i32x4_splat(c_scale);
    let bias16_v = i16x8_splat(-32768i16);
    let cru = i32x4_splat(coeffs.r_u());
    let crv = i32x4_splat(coeffs.r_v());
    let cgu = i32x4_splat(coeffs.g_u());
    let cgv = i32x4_splat(coeffs.g_v());
    let cbu = i32x4_splat(coeffs.b_u());
    let cbv = i32x4_splat(coeffs.b_v());
    let alpha_u8 = u8x16_splat(0xFF);

    let mut x = 0usize;
    while x + 16 <= width {
      let y_low = v128_load(y.as_ptr().add(x).cast());
      let y_high = v128_load(y.as_ptr().add(x + 8).cast());

      let (u_lo_vec, v_lo_vec) = deinterleave_uv_u16_wasm(uv_full.as_ptr().add(x * 2));
      let (u_hi_vec, v_hi_vec) = deinterleave_uv_u16_wasm(uv_full.as_ptr().add(x * 2 + 16));

      let u_lo_i16 = i16x8_sub(u_lo_vec, bias16_v);
      let u_hi_i16 = i16x8_sub(u_hi_vec, bias16_v);
      let v_lo_i16 = i16x8_sub(v_lo_vec, bias16_v);
      let v_hi_i16 = i16x8_sub(v_hi_vec, bias16_v);

      let u_lo_a = i32x4_extend_low_i16x8(u_lo_i16);
      let u_lo_b = i32x4_extend_high_i16x8(u_lo_i16);
      let u_hi_a = i32x4_extend_low_i16x8(u_hi_i16);
      let u_hi_b = i32x4_extend_high_i16x8(u_hi_i16);
      let v_lo_a = i32x4_extend_low_i16x8(v_lo_i16);
      let v_lo_b = i32x4_extend_high_i16x8(v_lo_i16);
      let v_hi_a = i32x4_extend_low_i16x8(v_hi_i16);
      let v_hi_b = i32x4_extend_high_i16x8(v_hi_i16);

      let u_d_lo_a = q15_shift(i32x4_add(i32x4_mul(u_lo_a, c_scale_v), rnd_v));
      let u_d_lo_b = q15_shift(i32x4_add(i32x4_mul(u_lo_b, c_scale_v), rnd_v));
      let u_d_hi_a = q15_shift(i32x4_add(i32x4_mul(u_hi_a, c_scale_v), rnd_v));
      let u_d_hi_b = q15_shift(i32x4_add(i32x4_mul(u_hi_b, c_scale_v), rnd_v));
      let v_d_lo_a = q15_shift(i32x4_add(i32x4_mul(v_lo_a, c_scale_v), rnd_v));
      let v_d_lo_b = q15_shift(i32x4_add(i32x4_mul(v_lo_b, c_scale_v), rnd_v));
      let v_d_hi_a = q15_shift(i32x4_add(i32x4_mul(v_hi_a, c_scale_v), rnd_v));
      let v_d_hi_b = q15_shift(i32x4_add(i32x4_mul(v_hi_b, c_scale_v), rnd_v));

      let r_chroma_lo = chroma_i16x8(cru, crv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let r_chroma_hi = chroma_i16x8(cru, crv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let g_chroma_lo = chroma_i16x8(cgu, cgv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let g_chroma_hi = chroma_i16x8(cgu, cgv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let b_chroma_lo = chroma_i16x8(cbu, cbv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let b_chroma_hi = chroma_i16x8(cbu, cbv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);

      let y_scaled_lo = scale_y_u16_wasm(y_low, y_off32_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y_u16_wasm(y_high, y_off32_v, y_scale_v, rnd_v);

      let r_lo = i16x8_add_sat(y_scaled_lo, r_chroma_lo);
      let r_hi = i16x8_add_sat(y_scaled_hi, r_chroma_hi);
      let g_lo = i16x8_add_sat(y_scaled_lo, g_chroma_lo);
      let g_hi = i16x8_add_sat(y_scaled_hi, g_chroma_hi);
      let b_lo = i16x8_add_sat(y_scaled_lo, b_chroma_lo);
      let b_hi = i16x8_add_sat(y_scaled_hi, b_chroma_hi);

      let r_u8 = u8x16_narrow_i16x8(r_lo, r_hi);
      let g_u8 = u8x16_narrow_i16x8(g_lo, g_hi);
      let b_u8 = u8x16_narrow_i16x8(b_lo, b_hi);

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

/// wasm simd128 P416 → packed **native-depth `u16`** RGB. i64 chroma
/// via native `i64x2_shr` (no bias trick needed). 8 pixels per iter.
///
/// Thin wrapper over [`p_n_444_16_to_rgb_or_rgba_u16_row`] with `ALPHA = false`.
///
/// # Safety
///
/// Same as [`p_n_444_16_to_rgb_row`] but `rgb_out: &mut [u16]`.
#[inline]
#[target_feature(enable = "simd128")]
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

/// wasm simd128 sibling of [`p_n_444_16_to_rgba_row`] for native-depth
/// `u16` output. Alpha is `0xFFFF`.
///
/// # Safety
///
/// Same as [`p_n_444_16_to_rgb_u16_row`] but `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "simd128")]
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

/// Shared wasm simd128 P416 → native-depth `u16` kernel.
/// `ALPHA = false` writes RGB triples via `write_rgb_u16_8`;
/// `ALPHA = true` writes RGBA quads via `write_rgba_u16_8` with
/// constant alpha `0xFFFF`.
///
/// # Safety
///
/// 1. simd128 must be enabled at compile time.
/// 2. `y.len() >= width`, `uv_full.len() >= 2 * width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
#[inline]
#[target_feature(enable = "simd128")]
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
    let alpha_u16 = u16x8_splat(0xFFFF);
    let rnd_i64 = i64x2_splat(RND_I64);
    let rnd_i32 = i32x4_splat(RND_I32);
    let y_off32 = i32x4_splat(y_off);
    let y_scale_i64 = i64x2_splat(y_scale as i64);
    let c_scale_i32 = i32x4_splat(c_scale);
    let bias16 = i16x8_splat(-32768i16);
    let cru = i64x2_extend_low_i32x4(i32x4_splat(coeffs.r_u()));
    let crv = i64x2_extend_low_i32x4(i32x4_splat(coeffs.r_v()));
    let cgu = i64x2_extend_low_i32x4(i32x4_splat(coeffs.g_u()));
    let cgv = i64x2_extend_low_i32x4(i32x4_splat(coeffs.g_v()));
    let cbu = i64x2_extend_low_i32x4(i32x4_splat(coeffs.b_u()));
    let cbv = i64x2_extend_low_i32x4(i32x4_splat(coeffs.b_v()));

    let mut x = 0usize;
    while x + 8 <= width {
      // 8 Y + 8 chroma pairs (= 16 UV elements) — one deinterleave call.
      let y_vec = v128_load(y.as_ptr().add(x).cast());
      let (u_vec, v_vec) = deinterleave_uv_u16_wasm(uv_full.as_ptr().add(x * 2));

      let u_i16 = i16x8_sub(u_vec, bias16);
      let v_i16 = i16x8_sub(v_vec, bias16);

      let u_lo_i32 = i32x4_extend_low_i16x8(u_i16);
      let u_hi_i32 = i32x4_extend_high_i16x8(u_i16);
      let v_lo_i32 = i32x4_extend_low_i16x8(v_i16);
      let v_hi_i32 = i32x4_extend_high_i16x8(v_i16);

      let u_d_lo = i32x4_shr(i32x4_add(i32x4_mul(u_lo_i32, c_scale_i32), rnd_i32), 15);
      let u_d_hi = i32x4_shr(i32x4_add(i32x4_mul(u_hi_i32, c_scale_i32), rnd_i32), 15);
      let v_d_lo = i32x4_shr(i32x4_add(i32x4_mul(v_lo_i32, c_scale_i32), rnd_i32), 15);
      let v_d_hi = i32x4_shr(i32x4_add(i32x4_mul(v_hi_i32, c_scale_i32), rnd_i32), 15);

      // 4 chroma_i64x2 calls per channel (2 halves × 2 sub-halves).
      let u_d_lo_lo = i64x2_extend_low_i32x4(u_d_lo);
      let u_d_lo_hi = i64x2_extend_high_i32x4(u_d_lo);
      let u_d_hi_lo = i64x2_extend_low_i32x4(u_d_hi);
      let u_d_hi_hi = i64x2_extend_high_i32x4(u_d_hi);
      let v_d_lo_lo = i64x2_extend_low_i32x4(v_d_lo);
      let v_d_lo_hi = i64x2_extend_high_i32x4(v_d_lo);
      let v_d_hi_lo = i64x2_extend_low_i32x4(v_d_hi);
      let v_d_hi_hi = i64x2_extend_high_i32x4(v_d_hi);

      let r_ch_lo_lo = chroma_i64x2_wasm(cru, crv, u_d_lo_lo, v_d_lo_lo, rnd_i64);
      let r_ch_lo_hi = chroma_i64x2_wasm(cru, crv, u_d_lo_hi, v_d_lo_hi, rnd_i64);
      let r_ch_hi_lo = chroma_i64x2_wasm(cru, crv, u_d_hi_lo, v_d_hi_lo, rnd_i64);
      let r_ch_hi_hi = chroma_i64x2_wasm(cru, crv, u_d_hi_hi, v_d_hi_hi, rnd_i64);
      let g_ch_lo_lo = chroma_i64x2_wasm(cgu, cgv, u_d_lo_lo, v_d_lo_lo, rnd_i64);
      let g_ch_lo_hi = chroma_i64x2_wasm(cgu, cgv, u_d_lo_hi, v_d_lo_hi, rnd_i64);
      let g_ch_hi_lo = chroma_i64x2_wasm(cgu, cgv, u_d_hi_lo, v_d_hi_lo, rnd_i64);
      let g_ch_hi_hi = chroma_i64x2_wasm(cgu, cgv, u_d_hi_hi, v_d_hi_hi, rnd_i64);
      let b_ch_lo_lo = chroma_i64x2_wasm(cbu, cbv, u_d_lo_lo, v_d_lo_lo, rnd_i64);
      let b_ch_lo_hi = chroma_i64x2_wasm(cbu, cbv, u_d_lo_hi, v_d_lo_hi, rnd_i64);
      let b_ch_hi_lo = chroma_i64x2_wasm(cbu, cbv, u_d_hi_lo, v_d_hi_lo, rnd_i64);
      let b_ch_hi_hi = chroma_i64x2_wasm(cbu, cbv, u_d_hi_hi, v_d_hi_hi, rnd_i64);

      let r_ch_lo = combine_i64x2_pair_to_i32x4(r_ch_lo_lo, r_ch_lo_hi);
      let r_ch_hi = combine_i64x2_pair_to_i32x4(r_ch_hi_lo, r_ch_hi_hi);
      let g_ch_lo = combine_i64x2_pair_to_i32x4(g_ch_lo_lo, g_ch_lo_hi);
      let g_ch_hi = combine_i64x2_pair_to_i32x4(g_ch_hi_lo, g_ch_hi_hi);
      let b_ch_lo = combine_i64x2_pair_to_i32x4(b_ch_lo_lo, b_ch_lo_hi);
      let b_ch_hi = combine_i64x2_pair_to_i32x4(b_ch_hi_lo, b_ch_hi_hi);

      let y_lo_u32 = u32x4_extend_low_u16x8(y_vec);
      let y_hi_u32 = u32x4_extend_high_u16x8(y_vec);
      let y_lo_i32 = i32x4_sub(y_lo_u32, y_off32);
      let y_hi_i32 = i32x4_sub(y_hi_u32, y_off32);

      let y_lo_scaled = scale_y_i32x4_i64_wasm(y_lo_i32, y_scale_i64, rnd_i64);
      let y_hi_scaled = scale_y_i32x4_i64_wasm(y_hi_i32, y_scale_i64, rnd_i64);

      let r_u16 = u16x8_narrow_i32x4(
        i32x4_add(y_lo_scaled, r_ch_lo),
        i32x4_add(y_hi_scaled, r_ch_hi),
      );
      let g_u16 = u16x8_narrow_i32x4(
        i32x4_add(y_lo_scaled, g_ch_lo),
        i32x4_add(y_hi_scaled, g_ch_hi),
      );
      let b_u16 = u16x8_narrow_i32x4(
        i32x4_add(y_lo_scaled, b_ch_lo),
        i32x4_add(y_hi_scaled, b_ch_hi),
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
