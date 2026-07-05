use core::arch::wasm32::*;

use super::*;

/// Compile-time host endianness. `true` on BE targets, `false` on LE
/// targets (always `false` on `wasm32` in practice).
const HOST_NATIVE_BE: bool = cfg!(target_endian = "big");

/// Byte-swap every u16 lane of `v` in-register when the source (wire)
/// endian differs from the host's native u16 byte order.
///
/// Used after `deinterleave_uv_u16_wasm` to apply per-lane byte-swapping.
/// Gated on `BE != HOST_NATIVE_BE` so a hypothetical BE-wasm host would
/// not double-swap. When the gate folds to `false` at compile time, the
/// call compiles away entirely.
#[inline(always)]
unsafe fn byteswap_u16x8<const BE: bool>(v: v128) -> v128 {
  if BE != HOST_NATIVE_BE {
    let mask = i8x16(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14);
    u8x16_swizzle(v, mask)
  } else {
    v
  }
}

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
pub(crate) unsafe fn p_n_444_to_rgb_row<const BITS: u32, const BE: bool>(
  y: &[u16],
  uv_full: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    p_n_444_to_rgb_or_rgba_row::<BITS, false, BE, false>(
      y, uv_full, rgb_out, width, matrix, full_range,
    );
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
pub(crate) unsafe fn p_n_444_to_rgba_row<const BITS: u32, const BE: bool>(
  y: &[u16],
  uv_full: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    p_n_444_to_rgb_or_rgba_row::<BITS, true, BE, false>(
      y, uv_full, rgba_out, width, matrix, full_range,
    );
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
pub(crate) unsafe fn p_n_444_to_rgb_or_rgba_row<
  const BITS: u32,
  const ALPHA: bool,
  const BE: bool,
  const LOW_PACKED: bool,
>(
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

    // `depack_u16x8::<LOW_PACKED>`: `>> (16 - BITS)` (high-packed) or
    // `& ((1 << BITS) - 1)` (low-packed NV20), byte-identical to scalar.
    let shr = 16 - BITS;
    let low_mask = u16x8_splat(((1u32 << BITS) - 1) as u16);

    let mut x = 0usize;
    while x + 16 <= width {
      // BE input is byte-swapped via `load_endian_u16x8::<BE>` for Y,
      // and via `byteswap_u16x8::<BE>` after deinterleave for UV.
      let y_low_i16 = depack_u16x8::<LOW_PACKED>(
        endian::load_endian_u16x8::<BE>(y.as_ptr().add(x) as *const u8),
        shr,
        low_mask,
      );
      let y_high_i16 = depack_u16x8::<LOW_PACKED>(
        endian::load_endian_u16x8::<BE>(y.as_ptr().add(x + 8) as *const u8),
        shr,
        low_mask,
      );

      // 32 UV elements (= 16 pairs) — two deinterleave calls.
      let (u_lo_vec, v_lo_vec) = deinterleave_uv_u16_wasm(uv_full.as_ptr().add(x * 2));
      let (u_hi_vec, v_hi_vec) = deinterleave_uv_u16_wasm(uv_full.as_ptr().add(x * 2 + 16));
      let u_lo_vec = byteswap_u16x8::<BE>(u_lo_vec);
      let v_lo_vec = byteswap_u16x8::<BE>(v_lo_vec);
      let u_hi_vec = byteswap_u16x8::<BE>(u_hi_vec);
      let v_hi_vec = byteswap_u16x8::<BE>(v_hi_vec);
      let u_lo_vec = depack_u16x8::<LOW_PACKED>(u_lo_vec, shr, low_mask);
      let v_lo_vec = depack_u16x8::<LOW_PACKED>(v_lo_vec, shr, low_mask);
      let u_hi_vec = depack_u16x8::<LOW_PACKED>(u_hi_vec, shr, low_mask);
      let v_hi_vec = depack_u16x8::<LOW_PACKED>(v_hi_vec, shr, low_mask);

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
      // Route the scalar tail through the same `LOW_PACKED` de-pack.
      scalar::p_n_444_to_rgb_or_rgba_row::<BITS, ALPHA, BE, LOW_PACKED>(
        tail_y, tail_uv, tail_out, tail_w, matrix, full_range,
      );
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
pub(crate) unsafe fn p_n_444_to_rgb_u16_row<const BITS: u32, const BE: bool>(
  y: &[u16],
  uv_full: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    p_n_444_to_rgb_or_rgba_u16_row::<BITS, false, BE, false>(
      y, uv_full, rgb_out, width, matrix, full_range,
    );
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
pub(crate) unsafe fn p_n_444_to_rgba_u16_row<const BITS: u32, const BE: bool>(
  y: &[u16],
  uv_full: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    p_n_444_to_rgb_or_rgba_u16_row::<BITS, true, BE, false>(
      y, uv_full, rgba_out, width, matrix, full_range,
    );
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
pub(crate) unsafe fn p_n_444_to_rgb_or_rgba_u16_row<
  const BITS: u32,
  const ALPHA: bool,
  const BE: bool,
  const LOW_PACKED: bool,
>(
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

    // `depack_u16x8::<LOW_PACKED>`: `>> (16 - BITS)` (high-packed) or
    // `& ((1 << BITS) - 1)` (low-packed NV20), byte-identical to scalar.
    let shr = 16 - BITS;
    let low_mask = u16x8_splat(((1u32 << BITS) - 1) as u16);

    let mut x = 0usize;
    while x + 16 <= width {
      // BE input is byte-swapped via `load_endian_u16x8::<BE>` for Y,
      // and via `byteswap_u16x8::<BE>` after deinterleave for UV.
      let y_low_i16 = depack_u16x8::<LOW_PACKED>(
        endian::load_endian_u16x8::<BE>(y.as_ptr().add(x) as *const u8),
        shr,
        low_mask,
      );
      let y_high_i16 = depack_u16x8::<LOW_PACKED>(
        endian::load_endian_u16x8::<BE>(y.as_ptr().add(x + 8) as *const u8),
        shr,
        low_mask,
      );

      let (u_lo_vec, v_lo_vec) = deinterleave_uv_u16_wasm(uv_full.as_ptr().add(x * 2));
      let (u_hi_vec, v_hi_vec) = deinterleave_uv_u16_wasm(uv_full.as_ptr().add(x * 2 + 16));
      let u_lo_vec = byteswap_u16x8::<BE>(u_lo_vec);
      let v_lo_vec = byteswap_u16x8::<BE>(v_lo_vec);
      let u_hi_vec = byteswap_u16x8::<BE>(u_hi_vec);
      let v_hi_vec = byteswap_u16x8::<BE>(v_hi_vec);
      let u_lo_vec = depack_u16x8::<LOW_PACKED>(u_lo_vec, shr, low_mask);
      let v_lo_vec = depack_u16x8::<LOW_PACKED>(v_lo_vec, shr, low_mask);
      let u_hi_vec = depack_u16x8::<LOW_PACKED>(u_hi_vec, shr, low_mask);
      let v_hi_vec = depack_u16x8::<LOW_PACKED>(v_hi_vec, shr, low_mask);

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
      // Route the scalar tail through the same `LOW_PACKED` de-pack.
      scalar::p_n_444_to_rgb_or_rgba_u16_row::<BITS, ALPHA, BE, LOW_PACKED>(
        tail_y, tail_uv, tail_out, tail_w, matrix, full_range,
      );
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
pub(crate) unsafe fn p_n_444_16_to_rgb_row<const BE: bool>(
  y: &[u16],
  uv_full: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    p_n_444_16_to_rgb_or_rgba_row::<false, BE>(y, uv_full, rgb_out, width, matrix, full_range);
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
pub(crate) unsafe fn p_n_444_16_to_rgba_row<const BE: bool>(
  y: &[u16],
  uv_full: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    p_n_444_16_to_rgb_or_rgba_row::<true, BE>(y, uv_full, rgba_out, width, matrix, full_range);
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
pub(crate) unsafe fn p_n_444_16_to_rgb_or_rgba_row<const ALPHA: bool, const BE: bool>(
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
      // BE input is byte-swapped via `load_endian_u16x8::<BE>` for Y,
      // and via `byteswap_u16x8::<BE>` after deinterleave for UV.
      let y_low = endian::load_endian_u16x8::<BE>(y.as_ptr().add(x) as *const u8);
      let y_high = endian::load_endian_u16x8::<BE>(y.as_ptr().add(x + 8) as *const u8);

      let (u_lo_vec, v_lo_vec) = deinterleave_uv_u16_wasm(uv_full.as_ptr().add(x * 2));
      let (u_hi_vec, v_hi_vec) = deinterleave_uv_u16_wasm(uv_full.as_ptr().add(x * 2 + 16));
      let u_lo_vec = byteswap_u16x8::<BE>(u_lo_vec);
      let v_lo_vec = byteswap_u16x8::<BE>(v_lo_vec);
      let u_hi_vec = byteswap_u16x8::<BE>(u_hi_vec);
      let v_hi_vec = byteswap_u16x8::<BE>(v_hi_vec);

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
        scalar::p_n_444_16_to_rgba_row::<BE>(tail_y, tail_uv, tail_out, tail_w, matrix, full_range);
      } else {
        scalar::p_n_444_16_to_rgb_row::<BE>(tail_y, tail_uv, tail_out, tail_w, matrix, full_range);
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
pub(crate) unsafe fn p_n_444_16_to_rgb_u16_row<const BE: bool>(
  y: &[u16],
  uv_full: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    p_n_444_16_to_rgb_or_rgba_u16_row::<false, BE>(y, uv_full, rgb_out, width, matrix, full_range);
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
pub(crate) unsafe fn p_n_444_16_to_rgba_u16_row<const BE: bool>(
  y: &[u16],
  uv_full: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    p_n_444_16_to_rgb_or_rgba_u16_row::<true, BE>(y, uv_full, rgba_out, width, matrix, full_range);
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
pub(crate) unsafe fn p_n_444_16_to_rgb_or_rgba_u16_row<const ALPHA: bool, const BE: bool>(
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
      // BE input is byte-swapped via `load_endian_u16x8::<BE>` for Y,
      // and via `byteswap_u16x8::<BE>` after deinterleave for UV.
      let y_vec = endian::load_endian_u16x8::<BE>(y.as_ptr().add(x) as *const u8);
      let (u_vec, v_vec) = deinterleave_uv_u16_wasm(uv_full.as_ptr().add(x * 2));
      let u_vec = byteswap_u16x8::<BE>(u_vec);
      let v_vec = byteswap_u16x8::<BE>(v_vec);

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
        scalar::p_n_444_16_to_rgba_u16_row::<BE>(
          tail_y, tail_uv, tail_out, tail_w, matrix, full_range,
        );
      } else {
        scalar::p_n_444_16_to_rgb_u16_row::<BE>(
          tail_y, tail_uv, tail_out, tail_w, matrix, full_range,
        );
      }
    }
  }
}

// ---- Pn 4:4:4 → HSV (staged via a reused 8-bit RGB chunk) -------------
//
// The wasm simd128 twins of the scalar `p_n_444_to_hsv_row` /
// `p_n_444_16_to_hsv_row`. Rather than re-derive an HSV-specific register
// pipeline, each fills a small fixed reused **8-bit** RGB scratch (one
// `HSV_CHUNK`-pixel chunk at a time) with the EXISTING wasm
// `p_n_444_to_rgb_row::<BITS, BE>` / `p_n_444_16_to_rgb_row::<BE>` kernel
// of this file — so the chunk filler IS the production 8-bit RGB kernel —
// then runs the wasm `rgb_to_hsv_row` on the chunk. The result is
// byte-identical to `rgb_to_hsv_row(p_n_444*_to_rgb_row(...))` within the
// wasm tier, with no source-width RGB allocation. The scalar tail of each
// underlying RGB kernel handles widths below the SIMD block, so no
// separate tail is needed here. The driver is defined locally (mirroring
// the 4:2:0 sibling); both compile under the same `yuv-semi-planar` gate.

/// One reused 8-bit RGB chunk's worth of pixels staged before the HSV
/// pass.
const HSV_CHUNK: usize = 64;

/// Shared wasm driver: walks `width` in `HSV_CHUNK`-pixel chunks, fills a
/// small reused stack RGB scratch via `fill_rgb` (the existing wasm 4:4:4
/// RGB kernel for the format, passed the chunk `offset` and length `n`),
/// then runs the wasm [`rgb_to_hsv_row`] on that chunk into the H/S/V
/// planes. Byte-identical to `rgb_to_hsv_row(p_n_444*_to_rgb_row(...))`
/// within the wasm tier, with no source-width RGB allocation.
///
/// `fill_rgb` receives `(offset, n, &mut rgb_chunk)` and must write
/// `n * 3` packed RGB bytes for the `n` pixels at `offset`.
///
/// # Safety
///
/// simd128 must be available, and `fill_rgb` must uphold the underlying
/// RGB kernel's safety contract for each chunk. Each of `h_out` /
/// `s_out` / `v_out` must be `>= width`.
#[inline]
unsafe fn pn_hsv_via_rgb_chunks(
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  mut fill_rgb: impl FnMut(usize, usize, &mut [u8]),
) {
  let mut scratch = [0u8; HSV_CHUNK * 3];
  let mut offset = 0;
  while offset < width {
    let n = (width - offset).min(HSV_CHUNK);
    fill_rgb(offset, n, &mut scratch[..n * 3]);
    // SAFETY: simd128 verified by the wrapper's `#[target_feature]`; the
    // chunk and the output sub-slices are all length `n`.
    unsafe {
      rgb_to_hsv_row(
        &scratch[..n * 3],
        &mut h_out[offset..offset + n],
        &mut s_out[offset..offset + n],
        &mut v_out[offset..offset + n],
        n,
      );
    }
    offset += n;
  }
}

/// wasm: high-bit-packed semi-planar 4:4:4 (P410/P412) → planar HSV bytes
/// (OpenCV encoding), staged via the reused-8-bit-RGB-chunk pattern over
/// the wasm [`p_n_444_to_rgb_row`] + [`rgb_to_hsv_row`]. Const-generic
/// over `BITS ∈ {10, 12}` and `BE`. Byte-identical to
/// `rgb_to_hsv_row(p_n_444_to_rgb_row::<BITS, BE>(...))` within the wasm
/// tier.
///
/// # Safety
///
/// 1. The simd128 feature must be available.
/// 2. `y.len() >= width`, `uv_full.len() >= 2 * width`.
/// 3. `h_out.len()`, `s_out.len()`, `v_out.len()` `>= width`.
#[inline]
#[target_feature(enable = "simd128")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn p_n_444_to_hsv_row<const BITS: u32, const BE: bool, const LOW_PACKED: bool>(
  y: &[u16],
  uv_full: &[u16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(uv_full.len() >= 2 * width, "uv_full row too short");
  debug_assert!(h_out.len() >= width, "h_out row too short");
  debug_assert!(s_out.len() >= width, "s_out row too short");
  debug_assert!(v_out.len() >= width, "v_out row too short");

  // SAFETY: the feature is the caller's obligation; the chunk filler
  // forwards the per-chunk sub-slices to the wasm 4:4:4 RGB kernel under
  // the same contract (its own scalar tail covers small n), threading the
  // same `LOW_PACKED` de-pack. The UV sub-slice is offset by `offset * 2`
  // because 4:4:4 carries one interleaved U/V pair per pixel.
  unsafe {
    pn_hsv_via_rgb_chunks(h_out, s_out, v_out, width, |offset, n, rgb| {
      p_n_444_to_rgb_or_rgba_row::<BITS, false, BE, LOW_PACKED>(
        &y[offset..],
        &uv_full[offset * 2..],
        rgb,
        n,
        matrix,
        full_range,
      );
    });
  }
}

/// wasm: P416 (semi-planar 4:4:4, 16-bit) → planar HSV bytes (OpenCV
/// encoding), staged via the wasm [`p_n_444_16_to_rgb_row`] +
/// [`rgb_to_hsv_row`]. `BE` selects the source byte order. Byte-identical
/// to `rgb_to_hsv_row(p_n_444_16_to_rgb_row::<BE>(...))` within the wasm
/// tier.
///
/// # Safety
///
/// Same contract as [`p_n_444_to_hsv_row`].
#[inline]
#[target_feature(enable = "simd128")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn p_n_444_16_to_hsv_row<const BE: bool>(
  y: &[u16],
  uv_full: &[u16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(uv_full.len() >= 2 * width, "uv_full row too short");
  debug_assert!(h_out.len() >= width, "h_out row too short");
  debug_assert!(s_out.len() >= width, "s_out row too short");
  debug_assert!(v_out.len() >= width, "v_out row too short");

  // SAFETY: the feature is the caller's obligation; the chunk filler
  // forwards to the wasm P416 RGB kernel under the same contract (its own
  // scalar tail covers small n).
  unsafe {
    pn_hsv_via_rgb_chunks(h_out, s_out, v_out, width, |offset, n, rgb| {
      p_n_444_16_to_rgb_or_rgba_row::<false, BE>(
        &y[offset..],
        &uv_full[offset * 2..],
        rgb,
        n,
        matrix,
        full_range,
      );
    });
  }
}
