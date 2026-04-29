use core::arch::wasm32::*;

use super::*;

/// WASM simd128 YUV 4:4:4 planar **16-bit** → packed **u8** RGB.
/// Stays on the i32 Q15 pipeline. 16 pixels per iter.
///
/// Thin wrapper over [`yuv_444p16_to_rgb_or_rgba_row`] with `ALPHA = false`.
///
/// # Safety
///
/// 1. **simd128 must be enabled at compile time.**
/// 2. `y.len() >= width`, `u.len() >= width`, `v.len() >= width`,
///    `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "simd128")]
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
    yuv_444p16_to_rgb_or_rgba_row::<false, false>(
      y, u, v, None, rgb_out, width, matrix, full_range,
    );
  }
}

/// WASM simd128 YUV 4:4:4 planar **16-bit** → packed **8-bit RGBA**
/// (`R, G, B, 0xFF`).
///
/// Thin wrapper over [`yuv_444p16_to_rgb_or_rgba_row`] with `ALPHA = true`.
///
/// # Safety
///
/// Same as [`yuv_444p16_to_rgb_row`] but `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "simd128")]
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
    yuv_444p16_to_rgb_or_rgba_row::<true, false>(
      y, u, v, None, rgba_out, width, matrix, full_range,
    );
  }
}

/// WASM simd128 YUVA 4:4:4 16-bit → packed **8-bit RGBA** with source
/// alpha. Same R/G/B numerical contract as [`yuv_444p16_to_rgba_row`];
/// the per-pixel alpha byte is **sourced from `a_src`** (depth-converted
/// via `u16x8_shr(_, 8)` to fit `u8`).
///
/// Thin wrapper over [`yuv_444p16_to_rgb_or_rgba_row`] with
/// `ALPHA = true, ALPHA_SRC = true`.
///
/// # Safety
///
/// Same as [`yuv_444p16_to_rgba_row`] plus `a_src.len() >= width`.
#[inline]
#[target_feature(enable = "simd128")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuv_444p16_to_rgba_with_alpha_src_row(
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
    yuv_444p16_to_rgb_or_rgba_row::<true, true>(
      y,
      u,
      v,
      Some(a_src),
      rgba_out,
      width,
      matrix,
      full_range,
    );
  }
}

/// Shared WASM simd128 16-bit YUV 4:4:4 kernel.
/// - `ALPHA = false, ALPHA_SRC = false`: `write_rgb_16`.
/// - `ALPHA = true, ALPHA_SRC = false`: `write_rgba_16` with constant
///   `0xFF` alpha.
/// - `ALPHA = true, ALPHA_SRC = true`: `write_rgba_16` with the alpha
///   lane loaded from `a_src` and depth-converted via `u16x8_shr(_, 8)`.
///
/// # Safety
///
/// 1. **simd128 must be enabled at compile time.**
/// 2. `y.len() >= width`, `u.len() >= width`, `v.len() >= width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
/// 3. If `ALPHA_SRC = true`, `a_src` is `Some(_)` with
///    `a_src.len() >= width`.
#[inline]
#[target_feature(enable = "simd128")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuv_444p16_to_rgb_or_rgba_row<const ALPHA: bool, const ALPHA_SRC: bool>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  a_src: Option<&[u16]>,
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // Source alpha requires RGBA output.
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
      let u_lo_vec = v128_load(u.as_ptr().add(x).cast());
      let u_hi_vec = v128_load(u.as_ptr().add(x + 8).cast());
      let v_lo_vec = v128_load(v.as_ptr().add(x).cast());
      let v_hi_vec = v128_load(v.as_ptr().add(x + 8).cast());

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
        let a_u8 = if ALPHA_SRC {
          // SAFETY (const-checked): ALPHA_SRC = true implies the
          // wrapper passed Some(_), validated by debug_assert above.
          // 16-bit alpha is full-range u16 — `>> 8` to fit u8.
          let a_ptr = a_src.as_ref().unwrap_unchecked().as_ptr();
          let a_lo = u16x8_shr(v128_load(a_ptr.add(x).cast()), 8);
          let a_hi = u16x8_shr(v128_load(a_ptr.add(x + 8).cast()), 8);
          u8x16_narrow_i16x8(a_lo, a_hi)
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
        scalar::yuv_444p16_to_rgba_with_alpha_src_row(
          tail_y, tail_u, tail_v, tail_a, tail_out, tail_w, matrix, full_range,
        );
      } else if ALPHA {
        scalar::yuv_444p16_to_rgba_row(
          tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range,
        );
      } else {
        scalar::yuv_444p16_to_rgb_row(tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range);
      }
    }
  }
}

/// WASM simd128 YUV 4:4:4 planar **16-bit** → packed **u16** RGB.
/// 8 pixels per iter on the i64 chroma pipeline.
///
/// Thin wrapper over [`yuv_444p16_to_rgb_or_rgba_u16_row`] with `ALPHA = false`.
///
/// # Safety
///
/// Same as [`yuv_444p16_to_rgb_row`] but `rgb_out: &mut [u16]`.
#[inline]
#[target_feature(enable = "simd128")]
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
    yuv_444p16_to_rgb_or_rgba_u16_row::<false, false>(
      y, u, v, None, rgb_out, width, matrix, full_range,
    );
  }
}

/// wasm simd128 sibling of [`yuv_444p16_to_rgba_row`] for native-depth
/// `u16` output. Alpha is `0xFFFF`.
///
/// # Safety
///
/// Same as [`yuv_444p16_to_rgb_u16_row`] but `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "simd128")]
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
    yuv_444p16_to_rgb_or_rgba_u16_row::<true, false>(
      y, u, v, None, rgba_out, width, matrix, full_range,
    );
  }
}

/// wasm simd128 YUVA 4:4:4 16-bit → packed **native-depth `u16`** RGBA
/// with source alpha. Same R/G/B numerical contract as
/// [`yuv_444p16_to_rgba_u16_row`]; the per-pixel alpha element is
/// **sourced from `a_src`** at native depth (no shift needed).
///
/// Thin wrapper over [`yuv_444p16_to_rgb_or_rgba_u16_row`] with
/// `ALPHA = true, ALPHA_SRC = true`.
///
/// # Safety
///
/// Same as [`yuv_444p16_to_rgba_u16_row`] plus `a_src.len() >= width`.
#[inline]
#[target_feature(enable = "simd128")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuv_444p16_to_rgba_u16_with_alpha_src_row(
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
    yuv_444p16_to_rgb_or_rgba_u16_row::<true, true>(
      y,
      u,
      v,
      Some(a_src),
      rgba_out,
      width,
      matrix,
      full_range,
    );
  }
}

/// Shared wasm simd128 16-bit YUV 4:4:4 → native-depth `u16` kernel.
/// - `ALPHA = false, ALPHA_SRC = false`: writes RGB triples via
///   `write_rgb_u16_8`.
/// - `ALPHA = true, ALPHA_SRC = false`: writes RGBA quads via
///   `write_rgba_u16_8` with constant alpha `0xFFFF`.
/// - `ALPHA = true, ALPHA_SRC = true`: writes RGBA quads with the
///   alpha element loaded from `a_src` (16-bit input is full-range —
///   no shift needed).
///
/// # Safety
///
/// 1. **simd128 must be enabled at compile time.**
/// 2. `y.len() >= width`, `u.len() >= width`, `v.len() >= width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
/// 3. If `ALPHA_SRC = true`, `a_src` is `Some(_)` with
///    `a_src.len() >= width`.
#[inline]
#[target_feature(enable = "simd128")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuv_444p16_to_rgb_or_rgba_u16_row<const ALPHA: bool, const ALPHA_SRC: bool>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  a_src: Option<&[u16]>,
  out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // Source alpha requires RGBA output.
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
      // 8 Y + 8 U + 8 V per iter. 4:4:4 is 1:1 — no chroma dup.
      let y_vec = v128_load(y.as_ptr().add(x).cast());
      let u_vec = v128_load(u.as_ptr().add(x).cast());
      let v_vec = v128_load(v.as_ptr().add(x).cast());

      let u_i16 = i16x8_sub(u_vec, bias16);
      let v_i16 = i16x8_sub(v_vec, bias16);

      // Widen each i16x8 → two i32x4 halves.
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

      // Combine each pair into i32x4 → 8 chroma values as (lo, hi).
      let r_ch_lo = combine_i64x2_pair_to_i32x4(r_ch_lo_lo, r_ch_lo_hi);
      let r_ch_hi = combine_i64x2_pair_to_i32x4(r_ch_hi_lo, r_ch_hi_hi);
      let g_ch_lo = combine_i64x2_pair_to_i32x4(g_ch_lo_lo, g_ch_lo_hi);
      let g_ch_hi = combine_i64x2_pair_to_i32x4(g_ch_hi_lo, g_ch_hi_hi);
      let b_ch_lo = combine_i64x2_pair_to_i32x4(b_ch_lo_lo, b_ch_lo_hi);
      let b_ch_hi = combine_i64x2_pair_to_i32x4(b_ch_hi_lo, b_ch_hi_hi);

      // Y: widen 8 u16 → 2 × i32x4, subtract y_off, scale in i64.
      let y_lo_u32 = u32x4_extend_low_u16x8(y_vec);
      let y_hi_u32 = u32x4_extend_high_u16x8(y_vec);
      let y_lo_i32 = i32x4_sub(y_lo_u32, y_off32);
      let y_hi_i32 = i32x4_sub(y_hi_u32, y_off32);

      let y_lo_scaled = scale_y_i32x4_i64_wasm(y_lo_i32, y_scale_i64, rnd_i64);
      let y_hi_scaled = scale_y_i32x4_i64_wasm(y_hi_i32, y_scale_i64, rnd_i64);

      // Add Y + chroma (no dup — 4:4:4 is 1:1). Saturating narrow to u16.
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
        let a_v = if ALPHA_SRC {
          // SAFETY (const-checked): ALPHA_SRC = true implies the
          // wrapper passed Some(_), validated by debug_assert above.
          // 16-bit alpha is full-range u16 — load 8 lanes verbatim,
          // no shift needed.
          v128_load(a_src.as_ref().unwrap_unchecked().as_ptr().add(x).cast())
        } else {
          alpha_u16
        };
        write_rgba_u16_8(r_u16, g_u16, b_u16, a_v, out.as_mut_ptr().add(x * 4));
      } else {
        write_rgb_u16_8(r_u16, g_u16, b_u16, out.as_mut_ptr().add(x * 3));
      }
      x += 8;
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
        scalar::yuv_444p16_to_rgba_u16_with_alpha_src_row(
          tail_y, tail_u, tail_v, tail_a, tail_out, tail_w, matrix, full_range,
        );
      } else if ALPHA {
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
/// Thin wrapper over [`yuv_420p16_to_rgb_or_rgba_row`] with `ALPHA = false`.
#[inline]
#[target_feature(enable = "simd128")]
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
    yuv_420p16_to_rgb_or_rgba_row::<false, false>(
      y, u_half, v_half, None, rgb_out, width, matrix, full_range,
    );
  }
}

/// wasm simd128 16-bit YUV 4:2:0 → packed **8-bit RGBA** (`R, G, B, 0xFF`).
///
/// Thin wrapper over [`yuv_420p16_to_rgb_or_rgba_row`] with `ALPHA = true`.
#[inline]
#[target_feature(enable = "simd128")]
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
    yuv_420p16_to_rgb_or_rgba_row::<true, false>(
      y, u_half, v_half, None, rgba_out, width, matrix, full_range,
    );
  }
}

/// wasm simd128 16-bit YUVA 4:2:0 → packed **8-bit RGBA** with the
/// per-pixel alpha byte **sourced from `a_src`** (depth-converted via
/// `>> 8` to fit `u8`). 16-bit alpha is full-range u16 — no AND-mask
/// step. Same numerical contract as [`yuv_420p16_to_rgba_row`] for
/// R/G/B.
///
/// Thin wrapper over [`yuv_420p16_to_rgb_or_rgba_row`] with
/// `ALPHA = true, ALPHA_SRC = true`.
///
/// # Safety
///
/// Same as [`yuv_420p16_to_rgba_row`] plus `a_src.len() >= width`.
#[inline]
#[target_feature(enable = "simd128")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuv_420p16_to_rgba_with_alpha_src_row(
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
    yuv_420p16_to_rgb_or_rgba_row::<true, true>(
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

/// Shared wasm simd128 16-bit YUV 4:2:0 kernel.
/// - `ALPHA = false, ALPHA_SRC = false`: `write_rgb_16`.
/// - `ALPHA = true, ALPHA_SRC = false`: `write_rgba_16` with constant
///   `0xFF` alpha.
/// - `ALPHA = true, ALPHA_SRC = true`: `write_rgba_16` with the alpha
///   lane loaded from `a_src` and depth-converted via `u16x8_shr`
///   (count = 8).
#[inline]
#[target_feature(enable = "simd128")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuv_420p16_to_rgb_or_rgba_row<const ALPHA: bool, const ALPHA_SRC: bool>(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  a_src: Option<&[u16]>,
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
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
      let u_vec = v128_load(u_half.as_ptr().add(x / 2).cast());
      let v_vec = v128_load(v_half.as_ptr().add(x / 2).cast());

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
        let a_u8 = if ALPHA_SRC {
          // SAFETY (const-checked): ALPHA_SRC = true implies the
          // wrapper passed Some(_), validated by debug_assert above.
          // 16-bit alpha is full-range u16 — `>> 8` to fit u8 directly.
          let a_ptr = a_src.as_ref().unwrap_unchecked().as_ptr();
          let a_lo = u16x8_shr(v128_load(a_ptr.add(x).cast()), 8);
          let a_hi = u16x8_shr(v128_load(a_ptr.add(x + 8).cast()), 8);
          u8x16_narrow_i16x8(a_lo, a_hi)
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
        scalar::yuv_420p16_to_rgba_with_alpha_src_row(
          tail_y, tail_u, tail_v, tail_a, tail_out, tail_w, matrix, full_range,
        );
      } else if ALPHA {
        scalar::yuv_420p16_to_rgba_row(
          tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range,
        );
      } else {
        scalar::yuv_420p16_to_rgb_row(tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range);
      }
    }
  }
}

/// WASM simd128 YUV 4:2:0 16-bit → packed **16-bit** RGB.
/// Delegates to scalar (no native i64 arithmetic shift in simd128 at this time).
///
/// # Safety
///
/// 1. **simd128 must be enabled at compile time.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `u_half.len() >= width / 2`,
///    `v_half.len() >= width / 2`, `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "simd128")]
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
    yuv_420p16_to_rgb_or_rgba_u16_row::<false, false>(
      y, u_half, v_half, None, rgb_out, width, matrix, full_range,
    );
  }
}

/// wasm simd128 sibling of [`yuv_420p16_to_rgba_row`] for native-depth
/// `u16` output. Alpha is `0xFFFF`.
///
/// # Safety
///
/// Same as [`yuv_420p16_to_rgb_u16_row`] plus `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "simd128")]
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
    yuv_420p16_to_rgb_or_rgba_u16_row::<true, false>(
      y, u_half, v_half, None, rgba_out, width, matrix, full_range,
    );
  }
}

/// wasm simd128 16-bit YUVA 4:2:0 → **native-depth `u16`** packed
/// RGBA with the per-pixel alpha element **sourced from `a_src`**
/// (full-range u16, no mask, no shift) instead of being constant
/// `0xFFFF`.
///
/// Thin wrapper over [`yuv_420p16_to_rgb_or_rgba_u16_row`] with
/// `ALPHA = true, ALPHA_SRC = true`.
///
/// # Safety
///
/// Same as [`yuv_420p16_to_rgba_u16_row`] plus `a_src.len() >= width`.
#[inline]
#[target_feature(enable = "simd128")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuv_420p16_to_rgba_u16_with_alpha_src_row(
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
    yuv_420p16_to_rgb_or_rgba_u16_row::<true, true>(
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

/// Shared wasm simd128 16-bit YUV 4:2:0 → native-depth `u16` kernel.
/// - `ALPHA = false, ALPHA_SRC = false`: `write_rgb_u16_8`.
/// - `ALPHA = true, ALPHA_SRC = false`: `write_rgba_u16_8` with
///   constant alpha `0xFFFF`.
/// - `ALPHA = true, ALPHA_SRC = true`: `write_rgba_u16_8` with the
///   alpha lane loaded from `a_src` (full-range u16).
///
/// # Safety
///
/// 1. **simd128 must be enabled at compile time.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `u_half.len() >= width / 2`,
///    `v_half.len() >= width / 2`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
/// 4. When `ALPHA_SRC = true`: `a_src` must be `Some(_)` and
///    `a_src.unwrap().len() >= width`.
#[inline]
#[target_feature(enable = "simd128")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuv_420p16_to_rgb_or_rgba_u16_row<const ALPHA: bool, const ALPHA_SRC: bool>(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  a_src: Option<&[u16]>,
  out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
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
    // Coefficients widened once to i64x2 (value replicated, so extend_low
    // suffices — both i64 lanes receive the same coeff).
    let cru = i64x2_extend_low_i32x4(i32x4_splat(coeffs.r_u()));
    let crv = i64x2_extend_low_i32x4(i32x4_splat(coeffs.r_v()));
    let cgu = i64x2_extend_low_i32x4(i32x4_splat(coeffs.g_u()));
    let cgv = i64x2_extend_low_i32x4(i32x4_splat(coeffs.g_v()));
    let cbu = i64x2_extend_low_i32x4(i32x4_splat(coeffs.b_u()));
    let cbv = i64x2_extend_low_i32x4(i32x4_splat(coeffs.b_v()));

    let mut x = 0usize;
    while x + 8 <= width {
      // 8 Y pixels / 4 chroma pairs per iter (i64x2 constraint).
      let y_vec = v128_load(y.as_ptr().add(x).cast());
      // 4 U + 4 V samples = 8 bytes each. Use `v128_load64_zero` so we
      // don't over-read 8 bytes past the chroma plane — the public
      // contract only promises `u_half.len() >= width / 2`, and at
      // tight width=16 the second iteration's `v128_load` at
      // `u_half[4..]` would read 8 bytes past the end.
      let u_vec = v128_load64_zero(u_half.as_ptr().add(x / 2).cast());
      let v_vec = v128_load64_zero(v_half.as_ptr().add(x / 2).cast());

      let u_i16 = i16x8_sub(u_vec, bias16);
      let v_i16 = i16x8_sub(v_vec, bias16);

      let u_i32 = i32x4_extend_low_i16x8(u_i16);
      let v_i32 = i32x4_extend_low_i16x8(v_i16);

      let u_d = i32x4_shr(i32x4_add(i32x4_mul(u_i32, c_scale_i32), rnd_i32), 15);
      let v_d = i32x4_shr(i32x4_add(i32x4_mul(v_i32, c_scale_i32), rnd_i32), 15);

      // Widen to 2 × i64x2 for the chroma i64 pipeline.
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

      // Combine i64x2 pairs → i32x4 [r0, r1, r2, r3].
      let r_ch_i32 = combine_i64x2_pair_to_i32x4(r_ch_lo, r_ch_hi);
      let g_ch_i32 = combine_i64x2_pair_to_i32x4(g_ch_lo, g_ch_hi);
      let b_ch_i32 = combine_i64x2_pair_to_i32x4(b_ch_lo, b_ch_hi);

      // Dup for 2 Y per chroma pair.
      let (r_dup_lo, r_dup_hi) = chroma_dup_i32x4_u16(r_ch_i32);
      let (g_dup_lo, g_dup_hi) = chroma_dup_i32x4_u16(g_ch_i32);
      let (b_dup_lo, b_dup_hi) = chroma_dup_i32x4_u16(b_ch_i32);

      // Y: widen 8 u16 → 2 × i32x4, subtract y_off, scale in i64.
      let y_lo_u32 = u32x4_extend_low_u16x8(y_vec);
      let y_hi_u32 = u32x4_extend_high_u16x8(y_vec);
      let y_lo_i32 = i32x4_sub(y_lo_u32, y_off32);
      let y_hi_i32 = i32x4_sub(y_hi_u32, y_off32);

      let y_lo_scaled = scale_y_i32x4_i64_wasm(y_lo_i32, y_scale_i64, rnd_i64);
      let y_hi_scaled = scale_y_i32x4_i64_wasm(y_hi_i32, y_scale_i64, rnd_i64);

      // Add Y + chroma, saturating narrow i32 → u16.
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
        let a_v = if ALPHA_SRC {
          // SAFETY (const-checked): ALPHA_SRC = true implies the
          // wrapper passed Some(_), validated by debug_assert above.
          // 16-bit alpha is full-range u16 — load 8 lanes directly.
          v128_load(a_src.as_ref().unwrap_unchecked().as_ptr().add(x).cast())
        } else {
          alpha_u16
        };
        write_rgba_u16_8(r_u16, g_u16, b_u16, a_v, out.as_mut_ptr().add(x * 4));
      } else {
        write_rgb_u16_8(r_u16, g_u16, b_u16, out.as_mut_ptr().add(x * 3));
      }
      x += 8;
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
        scalar::yuv_420p16_to_rgba_u16_with_alpha_src_row(
          tail_y, tail_u, tail_v, tail_a, tail_out, tail_w, matrix, full_range,
        );
      } else if ALPHA {
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
