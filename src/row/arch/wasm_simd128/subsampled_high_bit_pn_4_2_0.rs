use core::arch::wasm32::*;

use super::*;

/// Compile-time host endianness. `true` on BE targets, `false` on LE
/// targets (always `false` on `wasm32` in practice).
const HOST_NATIVE_BE: bool = cfg!(target_endian = "big");

/// De-packs a host-native `v128` of wire u16 samples to the `BITS`-bit
/// active value: a `u16x8_shr` right shift by `16 - BITS` for the
/// high-bit-packed P-formats (`LOW_PACKED = false`), or a low-mask
/// `v128_and(v, low_mask)` for low-bit-packed NV20 (`LOW_PACKED = true`).
/// The wasm-simd128 twin of the scalar `depack_pn`. `LOW_PACKED` is
/// const, so the unused branch is eliminated; byte-identical to the
/// scalar de-pack per lane.
#[inline(always)]
pub(crate) fn depack_u16x8<const LOW_PACKED: bool>(v: v128, shr: u32, low_mask: v128) -> v128 {
  if LOW_PACKED {
    v128_and(v, low_mask)
  } else {
    u16x8_shr(v, shr)
  }
}

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
pub(crate) unsafe fn p_n_to_rgb_row<const BITS: u32, const BE: bool>(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    p_n_to_rgb_or_rgba_row::<BITS, false, BE, false>(
      y, uv_half, rgb_out, width, matrix, full_range,
    );
  }
}

/// wasm simd128 high-bit-packed semi-planar 4:2:0 → packed **8-bit
/// RGBA** (`R, G, B, 0xFF`).
///
/// Thin wrapper over [`p_n_to_rgb_or_rgba_row`] with `ALPHA = true`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn p_n_to_rgba_row<const BITS: u32, const BE: bool>(
  y: &[u16],
  uv_half: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    p_n_to_rgb_or_rgba_row::<BITS, true, BE, false>(
      y, uv_half, rgba_out, width, matrix, full_range,
    );
  }
}

/// wasm simd128 NV20 (low-bit-packed 4:2:2, 10-bit) → packed **8-bit**
/// RGB. Low-bit twin of [`p_n_to_rgb_row`] (`LOW_PACKED = true`,
/// `BITS = 10`). Byte-identical to [`scalar::nv20_to_rgb_row::<BE>`].
///
/// # Safety
///
/// Same as [`p_n_to_rgb_row`].
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn nv20_to_rgb_row<const BE: bool>(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    p_n_to_rgb_or_rgba_row::<10, false, BE, true>(y, uv_half, rgb_out, width, matrix, full_range);
  }
}

/// wasm simd128 NV20 → packed **8-bit RGBA** (`R, G, B, 0xFF`). Low-bit
/// twin of [`p_n_to_rgba_row`]. Byte-identical to
/// [`scalar::nv20_to_rgba_row::<BE>`].
///
/// # Safety
///
/// Same as [`p_n_to_rgba_row`].
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn nv20_to_rgba_row<const BE: bool>(
  y: &[u16],
  uv_half: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    p_n_to_rgb_or_rgba_row::<10, true, BE, true>(y, uv_half, rgba_out, width, matrix, full_range);
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
pub(crate) unsafe fn p_n_to_rgb_or_rgba_row<
  const BITS: u32,
  const ALPHA: bool,
  const BE: bool,
  const LOW_PACKED: bool,
>(
  y: &[u16],
  uv_half: &[u16],
  out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
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

    // High-bit-packed: `u16x8_shr` by `16 - BITS`; low-bit-packed NV20:
    // the `low_mask`. One path is dead per monomorphization.
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
      let (u_vec, v_vec) = deinterleave_uv_u16_wasm(uv_half.as_ptr().add(x));
      let u_vec = byteswap_u16x8::<BE>(u_vec);
      let v_vec = byteswap_u16x8::<BE>(v_vec);
      let u_vec = depack_u16x8::<LOW_PACKED>(u_vec, shr, low_mask);
      let v_vec = depack_u16x8::<LOW_PACKED>(v_vec, shr, low_mask);

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
      // Route the scalar tail through the same `LOW_PACKED` de-pack.
      scalar::p_n_to_rgb_or_rgba_row::<BITS, ALPHA, BE, LOW_PACKED>(
        tail_y, tail_uv, tail_out, tail_w, matrix, full_range,
      );
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
pub(crate) unsafe fn p_n_to_rgb_u16_row<const BITS: u32, const BE: bool>(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  unsafe {
    p_n_to_rgb_or_rgba_u16_row::<BITS, false, BE, false>(
      y, uv_half, rgb_out, width, matrix, full_range,
    );
  }
}

/// wasm simd128 NV20 → packed **native-depth `u16`** RGB (low-bit-packed,
/// `[0, 1023]`). Low-bit twin of [`p_n_to_rgb_u16_row`]. Byte-identical
/// to [`scalar::nv20_to_rgb_u16_row::<BE>`].
///
/// # Safety
///
/// Same as [`p_n_to_rgb_u16_row`].
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn nv20_to_rgb_u16_row<const BE: bool>(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  unsafe {
    p_n_to_rgb_or_rgba_u16_row::<10, false, BE, true>(
      y, uv_half, rgb_out, width, matrix, full_range,
    );
  }
}

/// wasm simd128 NV20 → packed **native-depth `u16` RGBA**
/// (low-bit-packed; alpha `1023`). Low-bit twin of
/// [`p_n_to_rgba_u16_row`]. Byte-identical to
/// [`scalar::nv20_to_rgba_u16_row::<BE>`].
///
/// # Safety
///
/// Same as [`p_n_to_rgba_u16_row`].
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn nv20_to_rgba_u16_row<const BE: bool>(
  y: &[u16],
  uv_half: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  unsafe {
    p_n_to_rgb_or_rgba_u16_row::<10, true, BE, true>(
      y, uv_half, rgba_out, width, matrix, full_range,
    );
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
pub(crate) unsafe fn p_n_to_rgba_u16_row<const BITS: u32, const BE: bool>(
  y: &[u16],
  uv_half: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  unsafe {
    p_n_to_rgb_or_rgba_u16_row::<BITS, true, BE, false>(
      y, uv_half, rgba_out, width, matrix, full_range,
    );
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
pub(crate) unsafe fn p_n_to_rgb_or_rgba_u16_row<
  const BITS: u32,
  const ALPHA: bool,
  const BE: bool,
  const LOW_PACKED: bool,
>(
  y: &[u16],
  uv_half: &[u16],
  out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
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

    // High-bit-packed: `u16x8_shr` by `16 - BITS`; low-bit-packed NV20:
    // the `low_mask`. One path is dead per monomorphization.
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
      let (u_vec, v_vec) = deinterleave_uv_u16_wasm(uv_half.as_ptr().add(x));
      let u_vec = byteswap_u16x8::<BE>(u_vec);
      let v_vec = byteswap_u16x8::<BE>(v_vec);
      let u_vec = depack_u16x8::<LOW_PACKED>(u_vec, shr, low_mask);
      let v_vec = depack_u16x8::<LOW_PACKED>(v_vec, shr, low_mask);

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
      // Route the scalar tail through the same `LOW_PACKED` de-pack.
      scalar::p_n_to_rgb_or_rgba_u16_row::<BITS, ALPHA, BE, LOW_PACKED>(
        tail_y, tail_uv, tail_out, tail_w, matrix, full_range,
      );
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
pub(crate) unsafe fn p16_to_rgb_row<const BE: bool>(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    p16_to_rgb_or_rgba_row::<false, BE>(y, uv_half, rgb_out, width, matrix, full_range);
  }
}

/// wasm simd128 P016 → packed **8-bit RGBA** (`R, G, B, 0xFF`).
///
/// Thin wrapper over [`p16_to_rgb_or_rgba_row`] with `ALPHA = true`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn p16_to_rgba_row<const BE: bool>(
  y: &[u16],
  uv_half: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    p16_to_rgb_or_rgba_row::<true, BE>(y, uv_half, rgba_out, width, matrix, full_range);
  }
}

/// Shared wasm simd128 P016 kernel. `ALPHA = false` uses
/// `write_rgb_16`; `ALPHA = true` uses `write_rgba_16` with constant
/// `0xFF` alpha.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn p16_to_rgb_or_rgba_row<const ALPHA: bool, const BE: bool>(
  y: &[u16],
  uv_half: &[u16],
  out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
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
      // BE input is byte-swapped via `load_endian_u16x8::<BE>` for Y,
      // and via `byteswap_u16x8::<BE>` after deinterleave for UV.
      let y_low = endian::load_endian_u16x8::<BE>(y.as_ptr().add(x) as *const u8);
      let y_high = endian::load_endian_u16x8::<BE>(y.as_ptr().add(x + 8) as *const u8);
      let (u_vec, v_vec) = deinterleave_uv_u16_wasm(uv_half.as_ptr().add(x));
      let u_vec = byteswap_u16x8::<BE>(u_vec);
      let v_vec = byteswap_u16x8::<BE>(v_vec);

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
        scalar::p16_to_rgba_row::<BE>(tail_y, tail_uv, tail_out, tail_w, matrix, full_range);
      } else {
        scalar::p16_to_rgb_row::<BE>(tail_y, tail_uv, tail_out, tail_w, matrix, full_range);
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
pub(crate) unsafe fn p16_to_rgb_u16_row<const BE: bool>(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  unsafe {
    p16_to_rgb_or_rgba_u16_row::<false, BE>(y, uv_half, rgb_out, width, matrix, full_range);
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
pub(crate) unsafe fn p16_to_rgba_u16_row<const BE: bool>(
  y: &[u16],
  uv_half: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  unsafe {
    p16_to_rgb_or_rgba_u16_row::<true, BE>(y, uv_half, rgba_out, width, matrix, full_range);
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
pub(crate) unsafe fn p16_to_rgb_or_rgba_u16_row<const ALPHA: bool, const BE: bool>(
  y: &[u16],
  uv_half: &[u16],
  out: &mut [u16],
  width: usize,
  matrix: KernelMatrix,
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
      // [U0,U1,U2,U3, V0,V1,V2,V3]. BE input is byte-swapped via
      // `load_endian_u16x8::<BE>` for Y and via `byteswap_u16x8::<BE>`
      // after the deinterleave for UV.
      let y_vec = endian::load_endian_u16x8::<BE>(y.as_ptr().add(x) as *const u8);
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
      let u_vec = byteswap_u16x8::<BE>(u_vec);
      let v_vec = byteswap_u16x8::<BE>(v_vec);

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
        scalar::p16_to_rgba_u16_row::<BE>(tail_y, tail_uv, tail_out, tail_w, matrix, full_range);
      } else {
        scalar::p16_to_rgb_u16_row::<BE>(tail_y, tail_uv, tail_out, tail_w, matrix, full_range);
      }
    }
  }
}

// ---- High-bit semi-planar P-format → HSV (staged via a reused 8-bit RGB chunk) ----
//
// The wasm simd128 twins of the scalar `p_n_to_hsv_row` /
// `p16_to_hsv_row` kernels. Rather than re-derive an HSV-specific
// register pipeline, each fills a small fixed reused **8-bit** RGB
// scratch (one `HSV_CHUNK`-pixel chunk at a time) using the EXISTING
// wasm `p_n_to_rgb_row::<BITS, BE>` / `p16_to_rgb_row::<BE>` kernel of
// this file — so the chunk filler IS the production 8-bit RGB kernel —
// then runs the wasm `rgb_to_hsv_row` on the chunk. This makes the
// result byte-identical to `rgb_to_hsv_row(p_n_to_rgb_row::<BITS,
// BE>(...))` (and the P016 analogue) within the wasm tier — the same
// 8-bit RGB intermediate the existing high-bit HSV path uses — with no
// source-width RGB allocation. The scalar tail of each underlying RGB
// kernel handles widths below the SIMD block, so no separate tail is
// needed here. `HSV_CHUNK` (64) is a multiple of 2, so every chunk
// offset lands on a chroma-sample boundary for the 1→2 (4:2:0)
// interleaved-UV shape (`uv` byte offset = pixel offset).
//
// The chunked driver is defined locally (mirroring the semi-planar 8-bit
// `nv_to_hsv_via_rgb_chunks`) rather than reusing the planar
// `yuv_to_hsv_via_rgb_chunks`, because this file compiles under
// `yuv-semi-planar` alone while the planar driver lives behind the
// `yuv-planar` gate. Only `rgb_to_hsv_row` (ungated) is shared.

/// One reused 8-bit RGB chunk's worth of pixels staged before the HSV
/// pass.
const HSV_CHUNK: usize = 64;

/// Shared wasm driver: walks `width` in `HSV_CHUNK`-pixel chunks, fills a
/// small reused stack RGB scratch via `fill_rgb` (the existing wasm RGB
/// kernel for the format, passed the chunk `offset` and length `n`),
/// then runs the wasm [`rgb_to_hsv_row`] on that chunk into the H/S/V
/// planes. Byte-identical to `rgb_to_hsv_row(p*_to_rgb_row(...))` within
/// the wasm tier, with no source-width RGB allocation.
///
/// `fill_rgb` receives `(offset, n, &mut rgb_chunk)` and must write
/// `n * 3` packed RGB bytes for the `n` pixels at `offset`.
///
/// # Safety
///
/// simd128 must be available (compile-time on wasm32), and `fill_rgb`
/// must uphold the underlying RGB kernel's safety contract for each
/// chunk. Each of `h_out` / `s_out` / `v_out` must be `>= width`.
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
    // SAFETY: simd128 verified at compile time; the chunk and the output
    // sub-slices are all length `n`.
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

/// wasm simd128: high-bit-packed semi-planar 4:2:0 (P010/P012) → planar
/// HSV bytes (OpenCV encoding), staged via the reused-8-bit-RGB-chunk
/// pattern over the wasm [`p_n_to_rgb_row`] + [`rgb_to_hsv_row`].
/// Const-generic over `BITS ∈ {10, 12}` and `BE`. Byte-identical to
/// `rgb_to_hsv_row(p_n_to_rgb_row::<BITS, BE>(...))` within the wasm
/// tier.
///
/// # Safety
///
/// 1. simd128 must be available (compile-time on wasm32).
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`.
/// 4. `h_out.len()`, `s_out.len()`, `v_out.len()` `>= width`.
#[inline]
#[target_feature(enable = "simd128")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn p_n_to_hsv_row<const BITS: u32, const BE: bool, const LOW_PACKED: bool>(
  y: &[u16],
  uv_half: &[u16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0, "semi-planar high-bit requires even width");
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(uv_half.len() >= width, "uv row too short");
  debug_assert!(h_out.len() >= width, "h_out row too short");
  debug_assert!(s_out.len() >= width, "s_out row too short");
  debug_assert!(v_out.len() >= width, "v_out row too short");

  // SAFETY: simd128 verified at compile time; the chunk filler forwards
  // the per-chunk sub-slices to the wasm high-bit 4:2:0 RGB kernel under
  // the same contract (its own scalar tail covers small n). The same
  // `LOW_PACKED` de-pack threads through the shared RGB kernel.
  unsafe {
    pn_hsv_via_rgb_chunks(h_out, s_out, v_out, width, |offset, n, rgb| {
      p_n_to_rgb_or_rgba_row::<BITS, false, BE, LOW_PACKED>(
        &y[offset..],
        &uv_half[offset..],
        rgb,
        n,
        matrix,
        full_range,
      );
    });
  }
}

/// wasm simd128 NV20 → planar HSV bytes (OpenCV encoding), the low-bit
/// twin of [`p_n_to_hsv_row`] (`LOW_PACKED = true`, `BITS = 10`).
/// Byte-identical to `rgb_to_hsv_row(nv20_to_rgb_row::<BE>(...))` within
/// the wasm tier.
///
/// # Safety
///
/// Same contract as [`p_n_to_hsv_row`].
#[inline]
#[target_feature(enable = "simd128")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn nv20_to_hsv_row<const BE: bool>(
  y: &[u16],
  uv_half: &[u16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  // SAFETY: forwarded to the shared wasm HSV kernel with `LOW_PACKED = true`.
  unsafe {
    p_n_to_hsv_row::<10, BE, true>(y, uv_half, h_out, s_out, v_out, width, matrix, full_range);
  }
}

/// wasm simd128: P016 (semi-planar 4:2:0, 16-bit) → planar HSV bytes
/// (OpenCV encoding), staged via the wasm [`p16_to_rgb_row`] +
/// [`rgb_to_hsv_row`]. `BE` selects the source byte order. Byte-
/// identical to `rgb_to_hsv_row(p16_to_rgb_row::<BE>(...))` within the
/// wasm tier.
///
/// # Safety
///
/// Same contract as [`p_n_to_hsv_row`].
#[inline]
#[target_feature(enable = "simd128")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn p16_to_hsv_row<const BE: bool>(
  y: &[u16],
  uv_half: &[u16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0, "semi-planar 4:2:0 requires even width");
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(uv_half.len() >= width, "uv row too short");
  debug_assert!(h_out.len() >= width, "h_out row too short");
  debug_assert!(s_out.len() >= width, "s_out row too short");
  debug_assert!(v_out.len() >= width, "v_out row too short");

  // SAFETY: simd128 verified at compile time; the chunk filler forwards
  // to the wasm P016 RGB kernel under the same contract (its own scalar
  // tail covers small n).
  unsafe {
    pn_hsv_via_rgb_chunks(h_out, s_out, v_out, width, |offset, n, rgb| {
      p16_to_rgb_row::<BE>(&y[offset..], &uv_half[offset..], rgb, n, matrix, full_range);
    });
  }
}
