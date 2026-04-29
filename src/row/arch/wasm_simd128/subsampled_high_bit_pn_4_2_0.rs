use core::arch::wasm32::*;

use super::*;

/// WASM simd128 high‑bit‑packed semi‑planar (`BITS` ∈ {10, 12}) →
/// packed **8‑bit** RGB.
///
/// Block size 16 Y pixels / 8 chroma pairs per iteration. Mirrors
/// [`super::wasm_simd128::yuv_420p_n_to_rgb_row`] with two structural
/// differences:
/// - Samples are shifted right by `16 - BITS` (`u16x8_shr`, with
///   the shift amount computed from `BITS` once per call) instead
///   of AND‑masked.
/// - Semi‑planar UV is deinterleaved via [`deinterleave_uv_u16_wasm`]
///   (two `u8x16_swizzle` + two `i8x16_shuffle` combines).
///
/// # Numerical contract
///
/// Byte‑identical to [`scalar::p_n_to_rgb_row::<BITS>`] for the
/// monomorphized `BITS`.
///
/// # Safety
///
/// 1. **simd128 must be enabled at compile time.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`,
///    `rgb_out.len() >= 3 * width`.
///
/// Thin wrapper over [`p_n_to_rgb_or_rgba_row`] with `ALPHA = false`.
#[inline]
#[target_feature(enable = "simd128")]
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

/// wasm simd128 high-bit-packed semi-planar 4:2:0 → packed **8-bit
/// RGBA** (`R, G, B, 0xFF`).
///
/// Thin wrapper over [`p_n_to_rgb_or_rgba_row`] with `ALPHA = true`.
#[inline]
#[target_feature(enable = "simd128")]
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

/// Shared wasm simd128 P010/P012 kernel. `ALPHA = false` uses
/// `write_rgb_16`; `ALPHA = true` uses `write_rgba_16` with constant
/// `0xFF` alpha.
///
/// # Safety
///
/// 1. **simd128 enabled at compile time.**
/// 2. `width & 1 == 0`. 3. slices long enough +
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`. 4. `BITS` ∈ `{10, 12}`.
#[inline]
#[target_feature(enable = "simd128")]
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

  // SAFETY: simd128 compile‑time availability is the caller's
  // obligation.
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

    // High-bit-packed samples: shift right by `16 - BITS`.
    let shr = 16 - BITS;

    let mut x = 0usize;
    while x + 16 <= width {
      let y_low_i16 = u16x8_shr(v128_load(y.as_ptr().add(x).cast()), shr);
      let y_high_i16 = u16x8_shr(v128_load(y.as_ptr().add(x + 8).cast()), shr);
      let (u_vec, v_vec) = deinterleave_uv_u16_wasm(uv_half.as_ptr().add(x));
      let u_vec = u16x8_shr(u_vec, shr);
      let v_vec = u16x8_shr(v_vec, shr);

      let u_i16 = i16x8_sub(u_vec, bias_v);
      let v_i16 = i16x8_sub(v_vec, bias_v);

      let u_lo_i32 = i32x4_extend_low_i16x8(u_i16);
      let u_hi_i32 = i32x4_extend_high_i16x8(u_i16);
      let v_lo_i32 = i32x4_extend_low_i16x8(v_i16);
      let v_hi_i32 = i32x4_extend_high_i16x8(v_i16);

      let u_d_lo = q15_shift(i32x4_add(i32x4_mul(u_lo_i32, c_scale_v), rnd_v));
      let u_d_hi = q15_shift(i32x4_add(i32x4_mul(u_hi_i32, c_scale_v), rnd_v));
      let v_d_lo = q15_shift(i32x4_add(i32x4_mul(v_lo_i32, c_scale_v), rnd_v));
      let v_d_hi = q15_shift(i32x4_add(i32x4_mul(v_hi_i32, c_scale_v), rnd_v));

      let r_chroma = chroma_i16x8(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let g_chroma = chroma_i16x8(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let b_chroma = chroma_i16x8(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);

      let r_dup_lo = dup_lo(r_chroma);
      let r_dup_hi = dup_hi(r_chroma);
      let g_dup_lo = dup_lo(g_chroma);
      let g_dup_hi = dup_hi(g_chroma);
      let b_dup_lo = dup_lo(b_chroma);
      let b_dup_hi = dup_hi(b_chroma);

      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v);

      let b_lo = i16x8_add_sat(y_scaled_lo, b_dup_lo);
      let b_hi = i16x8_add_sat(y_scaled_hi, b_dup_hi);
      let g_lo = i16x8_add_sat(y_scaled_lo, g_dup_lo);
      let g_hi = i16x8_add_sat(y_scaled_hi, g_dup_hi);
      let r_lo = i16x8_add_sat(y_scaled_lo, r_dup_lo);
      let r_hi = i16x8_add_sat(y_scaled_hi, r_dup_hi);

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

/// WASM simd128 high‑bit‑packed semi‑planar (`BITS` ∈ {10, 12}) →
/// packed **native‑depth `u16`** RGB (low‑bit‑packed output,
/// `yuv420pNle` convention).
///
/// # Numerical contract
///
/// Byte‑identical to [`scalar::p_n_to_rgb_u16_row::<BITS>`] for the
/// monomorphized `BITS`.
///
/// # Safety
///
/// 1. **simd128 must be enabled at compile time.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`,
///    `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "simd128")]
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

/// wasm simd128 sibling of [`p_n_to_rgba_row`] for native-depth `u16`
/// output. Alpha samples are `(1 << BITS) - 1` (opaque maximum at the
/// input bit depth). P016 has its own kernel family — never routed here.
///
/// # Safety
///
/// Same as [`p_n_to_rgb_u16_row`] plus `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "simd128")]
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

/// Shared wasm simd128 Pn → native-depth `u16` kernel. `ALPHA = false`
/// writes RGB triples via `write_rgb_u16_8`; `ALPHA = true` writes
/// RGBA quads via `write_rgba_u16_8` with constant alpha
/// `(1 << BITS) - 1`. P016 has its own kernel family — never routed
/// here.
///
/// # Safety
///
/// 1. **simd128 must be enabled at compile time.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
/// 4. `BITS` ∈ `{10, 12}`.
#[inline]
#[target_feature(enable = "simd128")]
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

  // SAFETY: simd128 compile‑time availability is the caller's
  // obligation.
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

    // High-bit-packed samples: shift right by `16 - BITS`.
    let shr = 16 - BITS;

    let mut x = 0usize;
    while x + 16 <= width {
      let y_low_i16 = u16x8_shr(v128_load(y.as_ptr().add(x).cast()), shr);
      let y_high_i16 = u16x8_shr(v128_load(y.as_ptr().add(x + 8).cast()), shr);
      let (u_vec, v_vec) = deinterleave_uv_u16_wasm(uv_half.as_ptr().add(x));
      let u_vec = u16x8_shr(u_vec, shr);
      let v_vec = u16x8_shr(v_vec, shr);

      let u_i16 = i16x8_sub(u_vec, bias_v);
      let v_i16 = i16x8_sub(v_vec, bias_v);

      let u_lo_i32 = i32x4_extend_low_i16x8(u_i16);
      let u_hi_i32 = i32x4_extend_high_i16x8(u_i16);
      let v_lo_i32 = i32x4_extend_low_i16x8(v_i16);
      let v_hi_i32 = i32x4_extend_high_i16x8(v_i16);

      let u_d_lo = q15_shift(i32x4_add(i32x4_mul(u_lo_i32, c_scale_v), rnd_v));
      let u_d_hi = q15_shift(i32x4_add(i32x4_mul(u_hi_i32, c_scale_v), rnd_v));
      let v_d_lo = q15_shift(i32x4_add(i32x4_mul(v_lo_i32, c_scale_v), rnd_v));
      let v_d_hi = q15_shift(i32x4_add(i32x4_mul(v_hi_i32, c_scale_v), rnd_v));

      let r_chroma = chroma_i16x8(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let g_chroma = chroma_i16x8(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let b_chroma = chroma_i16x8(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);

      let r_dup_lo = dup_lo(r_chroma);
      let r_dup_hi = dup_hi(r_chroma);
      let g_dup_lo = dup_lo(g_chroma);
      let g_dup_hi = dup_hi(g_chroma);
      let b_dup_lo = dup_lo(b_chroma);
      let b_dup_hi = dup_hi(b_chroma);

      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v);

      let r_lo = clamp_u16_max_wasm(i16x8_add_sat(y_scaled_lo, r_dup_lo), zero_v, max_v);
      let r_hi = clamp_u16_max_wasm(i16x8_add_sat(y_scaled_hi, r_dup_hi), zero_v, max_v);
      let g_lo = clamp_u16_max_wasm(i16x8_add_sat(y_scaled_lo, g_dup_lo), zero_v, max_v);
      let g_hi = clamp_u16_max_wasm(i16x8_add_sat(y_scaled_hi, g_dup_hi), zero_v, max_v);
      let b_lo = clamp_u16_max_wasm(i16x8_add_sat(y_scaled_lo, b_dup_lo), zero_v, max_v);
      let b_hi = clamp_u16_max_wasm(i16x8_add_sat(y_scaled_hi, b_dup_hi), zero_v, max_v);

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
/// WASM simd128 P016 → packed **8-bit** RGB. 16 pixels per iteration.
///
/// # Safety
///
/// 1. **simd128 must be enabled at compile time.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`, `rgb_out.len() >= 3 * width`.
///
/// Thin wrapper over [`p16_to_rgb_or_rgba_row`] with `ALPHA = false`.
#[inline]
#[target_feature(enable = "simd128")]
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

/// wasm simd128 P016 → packed **8-bit RGBA** (`R, G, B, 0xFF`).
///
/// Thin wrapper over [`p16_to_rgb_or_rgba_row`] with `ALPHA = true`.
#[inline]
#[target_feature(enable = "simd128")]
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

/// Shared wasm simd128 P016 kernel. `ALPHA = false` uses
/// `write_rgb_16`; `ALPHA = true` uses `write_rgba_16` with constant
/// `0xFF` alpha.
#[inline]
#[target_feature(enable = "simd128")]
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
      let (u_vec, v_vec) = deinterleave_uv_u16_wasm(uv_half.as_ptr().add(x));

      let u_i16 = i16x8_sub(u_vec, bias16_v);
      let v_i16 = i16x8_sub(v_vec, bias16_v);

      let u_lo_i32 = i32x4_extend_low_i16x8(u_i16);
      let u_hi_i32 = i32x4_extend_high_i16x8(u_i16);
      let v_lo_i32 = i32x4_extend_low_i16x8(v_i16);
      let v_hi_i32 = i32x4_extend_high_i16x8(v_i16);

      let u_d_lo = q15_shift(i32x4_add(i32x4_mul(u_lo_i32, c_scale_v), rnd_v));
      let u_d_hi = q15_shift(i32x4_add(i32x4_mul(u_hi_i32, c_scale_v), rnd_v));
      let v_d_lo = q15_shift(i32x4_add(i32x4_mul(v_lo_i32, c_scale_v), rnd_v));
      let v_d_hi = q15_shift(i32x4_add(i32x4_mul(v_hi_i32, c_scale_v), rnd_v));

      let r_chroma = chroma_i16x8(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let g_chroma = chroma_i16x8(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let b_chroma = chroma_i16x8(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);

      let r_dup_lo = dup_lo(r_chroma);
      let r_dup_hi = dup_hi(r_chroma);
      let g_dup_lo = dup_lo(g_chroma);
      let g_dup_hi = dup_hi(g_chroma);
      let b_dup_lo = dup_lo(b_chroma);
      let b_dup_hi = dup_hi(b_chroma);

      let y_scaled_lo = scale_y_u16_wasm(y_low, y_off32_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y_u16_wasm(y_high, y_off32_v, y_scale_v, rnd_v);

      let r_lo = i16x8_add_sat(y_scaled_lo, r_dup_lo);
      let r_hi = i16x8_add_sat(y_scaled_hi, r_dup_hi);
      let g_lo = i16x8_add_sat(y_scaled_lo, g_dup_lo);
      let g_hi = i16x8_add_sat(y_scaled_hi, g_dup_hi);
      let b_lo = i16x8_add_sat(y_scaled_lo, b_dup_lo);
      let b_hi = i16x8_add_sat(y_scaled_hi, b_dup_hi);

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

/// WASM simd128 P016 → packed **16-bit** RGB. Delegates to scalar.
///
/// # Safety
///
/// 1. **simd128 must be enabled at compile time.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`, `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "simd128")]
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

/// wasm simd128 sibling of [`p16_to_rgba_row`] for native-depth `u16`
/// output. Alpha is `0xFFFF`.
///
/// # Safety
///
/// Same as [`p16_to_rgb_u16_row`] plus `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "simd128")]
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

/// Shared wasm simd128 16-bit P016 → native-depth `u16` kernel.
/// `ALPHA = false` writes RGB triples via `write_rgb_u16_8`;
/// `ALPHA = true` writes RGBA quads via `write_rgba_u16_8` with
/// constant alpha `0xFFFF`.
///
/// # Safety
///
/// 1. **simd128 must be enabled at compile time.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
#[inline]
#[target_feature(enable = "simd128")]
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
      // 8 Y + 4 UV pairs (= 8 u16 = 16 bytes). Deinterleave via
      // `i8x16_shuffle`: [U0,V0,U1,V1,U2,V2,U3,V3] →
      // [U0,U1,U2,U3, V0,V1,V2,V3].
      let y_vec = v128_load(y.as_ptr().add(x).cast());
      let uv_raw = v128_load(uv_half.as_ptr().add(x).cast());
      let uv_split =
        i8x16_shuffle::<0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15>(uv_raw, uv_raw);
      // u occupies the low 8 bytes, v the high 8.
      let u_vec = i8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23>(
        uv_split,
        i16x8_splat(0),
      );
      let v_vec = i8x16_shuffle::<8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23>(
        uv_split,
        i16x8_splat(0),
      );

      let u_i16 = i16x8_sub(u_vec, bias16);
      let v_i16 = i16x8_sub(v_vec, bias16);

      let u_i32 = i32x4_extend_low_i16x8(u_i16);
      let v_i32 = i32x4_extend_low_i16x8(v_i16);

      let u_d = i32x4_shr(i32x4_add(i32x4_mul(u_i32, c_scale_i32), rnd_i32), 15);
      let v_d = i32x4_shr(i32x4_add(i32x4_mul(v_i32, c_scale_i32), rnd_i32), 15);

      let u_d_lo = i64x2_extend_low_i32x4(u_d);
      let u_d_hi = i64x2_extend_high_i32x4(u_d);
      let v_d_lo = i64x2_extend_low_i32x4(v_d);
      let v_d_hi = i64x2_extend_high_i32x4(v_d);

      let r_ch_lo = chroma_i64x2_wasm(cru, crv, u_d_lo, v_d_lo, rnd_i64);
      let r_ch_hi = chroma_i64x2_wasm(cru, crv, u_d_hi, v_d_hi, rnd_i64);
      let g_ch_lo = chroma_i64x2_wasm(cgu, cgv, u_d_lo, v_d_lo, rnd_i64);
      let g_ch_hi = chroma_i64x2_wasm(cgu, cgv, u_d_hi, v_d_hi, rnd_i64);
      let b_ch_lo = chroma_i64x2_wasm(cbu, cbv, u_d_lo, v_d_lo, rnd_i64);
      let b_ch_hi = chroma_i64x2_wasm(cbu, cbv, u_d_hi, v_d_hi, rnd_i64);

      let r_ch_i32 = combine_i64x2_pair_to_i32x4(r_ch_lo, r_ch_hi);
      let g_ch_i32 = combine_i64x2_pair_to_i32x4(g_ch_lo, g_ch_hi);
      let b_ch_i32 = combine_i64x2_pair_to_i32x4(b_ch_lo, b_ch_hi);

      let (r_dup_lo, r_dup_hi) = chroma_dup_i32x4_u16(r_ch_i32);
      let (g_dup_lo, g_dup_hi) = chroma_dup_i32x4_u16(g_ch_i32);
      let (b_dup_lo, b_dup_hi) = chroma_dup_i32x4_u16(b_ch_i32);

      let y_lo_u32 = u32x4_extend_low_u16x8(y_vec);
      let y_hi_u32 = u32x4_extend_high_u16x8(y_vec);
      let y_lo_i32 = i32x4_sub(y_lo_u32, y_off32);
      let y_hi_i32 = i32x4_sub(y_hi_u32, y_off32);

      let y_lo_scaled = scale_y_i32x4_i64_wasm(y_lo_i32, y_scale_i64, rnd_i64);
      let y_hi_scaled = scale_y_i32x4_i64_wasm(y_hi_i32, y_scale_i64, rnd_i64);

      let r_u16 = u16x8_narrow_i32x4(
        i32x4_add(y_lo_scaled, r_dup_lo),
        i32x4_add(y_hi_scaled, r_dup_hi),
      );
      let g_u16 = u16x8_narrow_i32x4(
        i32x4_add(y_lo_scaled, g_dup_lo),
        i32x4_add(y_hi_scaled, g_dup_hi),
      );
      let b_u16 = u16x8_narrow_i32x4(
        i32x4_add(y_lo_scaled, b_dup_lo),
        i32x4_add(y_hi_scaled, b_dup_hi),
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
