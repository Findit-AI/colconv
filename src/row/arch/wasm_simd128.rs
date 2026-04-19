//! WebAssembly simd128 backend for the row primitives.
//!
//! Selected by [`crate::row`]'s dispatcher when
//! `cfg!(target_feature = "simd128")` evaluates true at compile time.
//! WASM does **not** support runtime CPU feature detection — a WASM
//! module either contains SIMD opcodes (which require runtime support
//! at instantiation) or it doesn't. So the gate is always
//! compile‑time, regardless of `feature = "std"`.
//!
//! The kernel carries `#[target_feature(enable = "simd128")]` so its
//! intrinsics are accessible to the function body even when simd128 is
//! not enabled for the whole crate.
//!
//! # Numerical contract
//!
//! Bit‑identical to
//! [`crate::row::scalar::yuv_420_to_rgb_row`]. All Q15 multiplies
//! are i32‑widened with `(prod + (1 << 14)) >> 15` rounding — same
//! structure as the NEON / SSE4.1 / AVX2 / AVX‑512 backends.
//!
//! # Pipeline (per 16 Y pixels / 8 chroma samples)
//!
//! 1. Load 16 Y (`v128_load`) + 8 U + 8 V (`u16x8_load_extend_u8x8`,
//!    which loads 8 u8 and zero‑extends to 8 u16 in one op).
//! 2. Subtract 128 from U, V (as i16x8) to get `u_i16`, `v_i16`.
//! 3. Split each i16x8 into two i32x4 halves via
//!    `i32x4_extend_{low,high}_i16x8` and apply `c_scale`.
//! 4. Per channel: `(C_u*u_d + C_v*v_d + RND) >> 15` in i32,
//!    saturating‑narrow to i16x8 via `i16x8_narrow_i32x4`.
//! 5. Nearest‑neighbor chroma upsample with two `i8x16_shuffle`
//!    invocations (compile‑time byte indices duplicate each 16‑bit
//!    chroma lane into its pair slot).
//! 6. Y path: widen low / high 8 Y to i16x8, apply `y_off` / `y_scale`.
//! 7. Saturating i16 add Y + chroma per channel (`i16x8_add_sat`).
//! 8. Saturate‑narrow to u8x16 per channel (`u8x16_narrow_i16x8`),
//!    interleave as packed RGB via three `u8x16_swizzle` calls.

use core::arch::wasm32::{
  f32x4_add, f32x4_convert_i32x4, f32x4_div, f32x4_eq, f32x4_lt, f32x4_max, f32x4_min, f32x4_mul,
  f32x4_splat, f32x4_sub, i8x16, i8x16_shuffle, i16x8_add_sat, i16x8_max, i16x8_min,
  i16x8_narrow_i32x4, i16x8_splat, i16x8_sub, i32x4_add, i32x4_extend_high_i16x8,
  i32x4_extend_low_i16x8, i32x4_mul, i32x4_shr, i32x4_splat, i32x4_trunc_sat_f32x4,
  u8x16_narrow_i16x8, u8x16_swizzle, u16x8_extend_high_u8x16, u16x8_extend_low_u8x16,
  u16x8_load_extend_u8x8, u16x8_shr, u16x8_splat, u32x4_extend_high_u16x8, u32x4_extend_low_u16x8,
  v128, v128_and, v128_bitselect, v128_load, v128_or, v128_store,
};

use crate::{ColorMatrix, row::scalar};

/// WASM simd128 YUV 4:2:0 → packed RGB. Semantics match
/// [`scalar::yuv_420_to_rgb_row`] byte‑identically.
///
/// # Safety
///
/// The caller must uphold **all** of the following. Violating any
/// causes undefined behavior:
///
/// 1. **simd128 must be enabled at compile time.** Verified by the
///    dispatcher via `cfg!(target_feature = "simd128")`. WASM has no
///    runtime CPU detection, so the obligation is purely compile‑time:
///    the WASM module was produced with `-C target-feature=+simd128`
///    (or equivalent), and it is being executed in a WASM runtime that
///    supports the SIMD proposal.
/// 2. `width & 1 == 0` (4:2:0 requires even width).
/// 3. `y.len() >= width`.
/// 4. `u_half.len() >= width / 2`.
/// 5. `v_half.len() >= width / 2`.
/// 6. `rgb_out.len() >= 3 * width`.
///
/// Bounds are verified by `debug_assert` in debug builds; release
/// builds trust the caller because the kernel relies on unchecked
/// pointer arithmetic (`v128_load`, `u16x8_load_extend_u8x8`,
/// `v128_store`).
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn yuv_420_to_rgb_row(
  y: &[u8],
  u_half: &[u8],
  v_half: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0, "YUV 4:2:0 requires even width");
  debug_assert!(y.len() >= width);
  debug_assert!(u_half.len() >= width / 2);
  debug_assert!(v_half.len() >= width / 2);
  debug_assert!(rgb_out.len() >= width * 3);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params(full_range);
  const RND: i32 = 1 << 14;

  // SAFETY: simd128 availability is the caller's compile‑time
  // obligation per the `# Safety` section. All pointer adds below are
  // bounded by the `while x + 16 <= width` loop condition and the
  // caller‑promised slice lengths.
  unsafe {
    let rnd_v = i32x4_splat(RND);
    let y_off_v = i16x8_splat(y_off as i16);
    let y_scale_v = i32x4_splat(y_scale);
    let c_scale_v = i32x4_splat(c_scale);
    let mid128 = i16x8_splat(128);
    let cru = i32x4_splat(coeffs.r_u());
    let crv = i32x4_splat(coeffs.r_v());
    let cgu = i32x4_splat(coeffs.g_u());
    let cgv = i32x4_splat(coeffs.g_v());
    let cbu = i32x4_splat(coeffs.b_u());
    let cbv = i32x4_splat(coeffs.b_v());

    let mut x = 0usize;
    while x + 16 <= width {
      // Load 16 Y (16 bytes) and 8 U / 8 V (extending each to i16x8).
      let y_vec = v128_load(y.as_ptr().add(x).cast());
      let u_i16_zero = u16x8_load_extend_u8x8(u_half.as_ptr().add(x / 2));
      let v_i16_zero = u16x8_load_extend_u8x8(v_half.as_ptr().add(x / 2));

      // Subtract 128 from chroma (u16 treated as i16).
      let u_i16 = i16x8_sub(u_i16_zero, mid128);
      let v_i16 = i16x8_sub(v_i16_zero, mid128);

      // Split each i16x8 into two i32x4 halves (sign‑extending).
      let u_lo_i32 = i32x4_extend_low_i16x8(u_i16);
      let u_hi_i32 = i32x4_extend_high_i16x8(u_i16);
      let v_lo_i32 = i32x4_extend_low_i16x8(v_i16);
      let v_hi_i32 = i32x4_extend_high_i16x8(v_i16);

      // u_d, v_d = (u * c_scale + RND) >> 15 — bit‑exact to scalar.
      let u_d_lo = q15_shift(i32x4_add(i32x4_mul(u_lo_i32, c_scale_v), rnd_v));
      let u_d_hi = q15_shift(i32x4_add(i32x4_mul(u_hi_i32, c_scale_v), rnd_v));
      let v_d_lo = q15_shift(i32x4_add(i32x4_mul(v_lo_i32, c_scale_v), rnd_v));
      let v_d_hi = q15_shift(i32x4_add(i32x4_mul(v_hi_i32, c_scale_v), rnd_v));

      // Per‑channel chroma → i16x8 (8 chroma values per channel).
      let r_chroma = chroma_i16x8(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let g_chroma = chroma_i16x8(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let b_chroma = chroma_i16x8(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);

      // Nearest‑neighbor upsample: duplicate each of 8 chroma lanes
      // into its pair slot → two i16x8 vectors covering 16 Y lanes.
      // Each i16 value is 2 bytes, so byte‑level shuffle indices
      // `[0,1,0,1, 2,3,2,3, 4,5,4,5, 6,7,6,7]` duplicate the low
      // 4 × i16 lanes; `[8..15 paired]` duplicates the high 4.
      let r_dup_lo = dup_lo(r_chroma);
      let r_dup_hi = dup_hi(r_chroma);
      let g_dup_lo = dup_lo(g_chroma);
      let g_dup_hi = dup_hi(g_chroma);
      let b_dup_lo = dup_lo(b_chroma);
      let b_dup_hi = dup_hi(b_chroma);

      // Y path: widen low / high 8 Y to i16x8, scale.
      let y_low_i16 = u8_low_to_i16x8(y_vec);
      let y_high_i16 = u8_high_to_i16x8(y_vec);
      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v);

      // Saturating i16 add Y + chroma per channel.
      let b_lo = i16x8_add_sat(y_scaled_lo, b_dup_lo);
      let b_hi = i16x8_add_sat(y_scaled_hi, b_dup_hi);
      let g_lo = i16x8_add_sat(y_scaled_lo, g_dup_lo);
      let g_hi = i16x8_add_sat(y_scaled_hi, g_dup_hi);
      let r_lo = i16x8_add_sat(y_scaled_lo, r_dup_lo);
      let r_hi = i16x8_add_sat(y_scaled_hi, r_dup_hi);

      // Saturate‑narrow to u8x16 per channel.
      let b_u8 = u8x16_narrow_i16x8(b_lo, b_hi);
      let g_u8 = u8x16_narrow_i16x8(g_lo, g_hi);
      let r_u8 = u8x16_narrow_i16x8(r_lo, r_hi);

      // 3‑way interleave → packed RGB (48 bytes).
      write_rgb_16(r_u8, g_u8, b_u8, rgb_out.as_mut_ptr().add(x * 3));

      x += 16;
    }

    // Scalar tail for the 0..14 leftover pixels.
    if x < width {
      scalar::yuv_420_to_rgb_row(
        &y[x..width],
        &u_half[x / 2..width / 2],
        &v_half[x / 2..width / 2],
        &mut rgb_out[x * 3..width * 3],
        width - x,
        matrix,
        full_range,
      );
    }
  }
}

/// WASM simd128 YUV 4:2:0 10‑bit → packed **8‑bit** RGB.
///
/// Block size 16 Y pixels / 8 chroma pairs per iteration. Differences
/// from [`yuv_420_to_rgb_row`]:
/// - Y loads are two `v128_load` (each holds 8 `u16` = 16 bytes); U / V
///   each one `v128_load` (8 `u16`).
/// - No u8→u16 widening — samples already in 16‑bit lanes.
/// - Chroma bias 512 (10‑bit center).
/// - `range_params_n::<10, 8>` calibrates scales for 10→8 in one shift.
///
/// Reuses [`chroma_i16x8`], [`dup_lo`], [`dup_hi`], [`scale_y`], and
/// [`write_rgb_16`].
///
/// # Numerical contract
///
/// Byte‑identical to [`scalar::yuv_420p_n_to_rgb_row::<10>`].
///
/// # Safety
///
/// 1. **simd128 must be enabled at compile time.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `u_half.len() >= width / 2`,
///    `v_half.len() >= width / 2`, `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn yuv_420p_n_to_rgb_row<const BITS: u32>(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0);
  debug_assert!(y.len() >= width);
  debug_assert!(u_half.len() >= width / 2);
  debug_assert!(v_half.len() >= width / 2);
  debug_assert!(rgb_out.len() >= width * 3);

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
    let mask_v = u16x8_splat(scalar::bits_mask::<BITS>());
    let cru = i32x4_splat(coeffs.r_u());
    let crv = i32x4_splat(coeffs.r_v());
    let cgu = i32x4_splat(coeffs.g_u());
    let cgv = i32x4_splat(coeffs.g_v());
    let cbu = i32x4_splat(coeffs.b_u());
    let cbv = i32x4_splat(coeffs.b_v());

    let mut x = 0usize;
    while x + 16 <= width {
      // AND‑mask each load to the low 10 bits — see matching comment
      // in [`crate::row::scalar::yuv_420p_n_to_rgb_row`].
      let y_low_i16 = v128_and(v128_load(y.as_ptr().add(x).cast()), mask_v);
      let y_high_i16 = v128_and(v128_load(y.as_ptr().add(x + 8).cast()), mask_v);
      let u_vec = v128_and(v128_load(u_half.as_ptr().add(x / 2).cast()), mask_v);
      let v_vec = v128_and(v128_load(v_half.as_ptr().add(x / 2).cast()), mask_v);

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

      write_rgb_16(r_u8, g_u8, b_u8, rgb_out.as_mut_ptr().add(x * 3));

      x += 16;
    }

    if x < width {
      scalar::yuv_420p_n_to_rgb_row::<BITS>(
        &y[x..width],
        &u_half[x / 2..width / 2],
        &v_half[x / 2..width / 2],
        &mut rgb_out[x * 3..width * 3],
        width - x,
        matrix,
        full_range,
      );
    }
  }
}

/// WASM simd128 YUV 4:2:0 10‑bit → packed **10‑bit `u16`** RGB.
///
/// Block 16 Y pixels. Mirrors [`yuv420p10_to_rgb_row`]'s pre‑write
/// math; output uses explicit `i16x8_min` / `i16x8_max` clamp to
/// `[0, 1023]` and two calls to [`write_rgb_u16_8`] per block.
///
/// # Numerical contract
///
/// Identical to [`scalar::yuv_420p_n_to_rgb_u16_row::<10>`].
///
/// # Safety
///
/// 1. **simd128 must be enabled at compile time.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `u_half.len() >= width / 2`,
///    `v_half.len() >= width / 2`, `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn yuv_420p_n_to_rgb_u16_row<const BITS: u32>(
  y: &[u16],
  u_half: &[u16],
  v_half: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0);
  debug_assert!(y.len() >= width);
  debug_assert!(u_half.len() >= width / 2);
  debug_assert!(v_half.len() >= width / 2);
  debug_assert!(rgb_out.len() >= width * 3);

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
    let mask_v = u16x8_splat(scalar::bits_mask::<BITS>());
    let max_v = i16x8_splat(out_max);
    let zero_v = i16x8_splat(0);
    let cru = i32x4_splat(coeffs.r_u());
    let crv = i32x4_splat(coeffs.r_v());
    let cgu = i32x4_splat(coeffs.g_u());
    let cgv = i32x4_splat(coeffs.g_v());
    let cbu = i32x4_splat(coeffs.b_u());
    let cbv = i32x4_splat(coeffs.b_v());

    let mut x = 0usize;
    while x + 16 <= width {
      // AND‑mask loads to the low 10 bits so `chroma_i16x8`'s
      // `i16x8_narrow_i32x4` stays lossless.
      let y_low_i16 = v128_and(v128_load(y.as_ptr().add(x).cast()), mask_v);
      let y_high_i16 = v128_and(v128_load(y.as_ptr().add(x + 8).cast()), mask_v);
      let u_vec = v128_and(v128_load(u_half.as_ptr().add(x / 2).cast()), mask_v);
      let v_vec = v128_and(v128_load(v_half.as_ptr().add(x / 2).cast()), mask_v);

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

      let r_lo = clamp_u10_wasm(i16x8_add_sat(y_scaled_lo, r_dup_lo), zero_v, max_v);
      let r_hi = clamp_u10_wasm(i16x8_add_sat(y_scaled_hi, r_dup_hi), zero_v, max_v);
      let g_lo = clamp_u10_wasm(i16x8_add_sat(y_scaled_lo, g_dup_lo), zero_v, max_v);
      let g_hi = clamp_u10_wasm(i16x8_add_sat(y_scaled_hi, g_dup_hi), zero_v, max_v);
      let b_lo = clamp_u10_wasm(i16x8_add_sat(y_scaled_lo, b_dup_lo), zero_v, max_v);
      let b_hi = clamp_u10_wasm(i16x8_add_sat(y_scaled_hi, b_dup_hi), zero_v, max_v);

      let dst = rgb_out.as_mut_ptr().add(x * 3);
      write_rgb_u16_8(r_lo, g_lo, b_lo, dst);
      write_rgb_u16_8(r_hi, g_hi, b_hi, dst.add(24));

      x += 16;
    }

    if x < width {
      scalar::yuv_420p_n_to_rgb_u16_row::<BITS>(
        &y[x..width],
        &u_half[x / 2..width / 2],
        &v_half[x / 2..width / 2],
        &mut rgb_out[x * 3..width * 3],
        width - x,
        matrix,
        full_range,
      );
    }
  }
}

/// Clamps an i16x8 vector to `[0, max]`. Used by the 10‑bit u16
/// output path.
#[inline(always)]
fn clamp_u10_wasm(v: v128, zero_v: v128, max_v: v128) -> v128 {
  i16x8_min(i16x8_max(v, zero_v), max_v)
}

/// Writes 8 pixels of packed `u16` RGB (24 `u16` = 48 bytes) using
/// the SSSE3‑style 3‑way interleave pattern adapted to 16‑bit lanes.
/// Mirrors [`crate::row::arch::x86_common::write_rgb_u16_8`] — each
/// output u16 is two adjacent bytes sourced from one of the three
/// channel vectors via `u8x16_swizzle` with a compile‑time byte
/// mask (0xFF / negative zeros the lane, matching `_mm_shuffle_epi8`
/// semantics).
///
/// # Safety
///
/// `ptr` must point to at least 48 writable bytes (24 `u16`). Caller
/// must have simd128 enabled at compile time.
#[inline(always)]
unsafe fn write_rgb_u16_8(r: v128, g: v128, b: v128, ptr: *mut u16) {
  unsafe {
    // Block 0 = [R0 G0 B0 R1 G1 B1 R2 G2]. Masks identical in shape
    // to x86_common::write_rgb_u16_8 — each output u16 pulls two
    // adjacent bytes from one channel.
    let r0 = i8x16(0, 1, -1, -1, -1, -1, 2, 3, -1, -1, -1, -1, 4, 5, -1, -1);
    let g0 = i8x16(-1, -1, 0, 1, -1, -1, -1, -1, 2, 3, -1, -1, -1, -1, 4, 5);
    let b0 = i8x16(-1, -1, -1, -1, 0, 1, -1, -1, -1, -1, 2, 3, -1, -1, -1, -1);
    let out0 = v128_or(
      v128_or(u8x16_swizzle(r, r0), u8x16_swizzle(g, g0)),
      u8x16_swizzle(b, b0),
    );

    // Block 1 = [B2 R3 G3 B3 R4 G4 B4 R5].
    let r1 = i8x16(-1, -1, 6, 7, -1, -1, -1, -1, 8, 9, -1, -1, -1, -1, 10, 11);
    let g1 = i8x16(-1, -1, -1, -1, 6, 7, -1, -1, -1, -1, 8, 9, -1, -1, -1, -1);
    let b1 = i8x16(4, 5, -1, -1, -1, -1, 6, 7, -1, -1, -1, -1, 8, 9, -1, -1);
    let out1 = v128_or(
      v128_or(u8x16_swizzle(r, r1), u8x16_swizzle(g, g1)),
      u8x16_swizzle(b, b1),
    );

    // Block 2 = [G5 B5 R6 G6 B6 R7 G7 B7].
    let r2 = i8x16(
      -1, -1, -1, -1, 12, 13, -1, -1, -1, -1, 14, 15, -1, -1, -1, -1,
    );
    let g2 = i8x16(
      10, 11, -1, -1, -1, -1, 12, 13, -1, -1, -1, -1, 14, 15, -1, -1,
    );
    let b2 = i8x16(
      -1, -1, 10, 11, -1, -1, -1, -1, 12, 13, -1, -1, -1, -1, 14, 15,
    );
    let out2 = v128_or(
      v128_or(u8x16_swizzle(r, r2), u8x16_swizzle(g, g2)),
      u8x16_swizzle(b, b2),
    );

    v128_store(ptr.cast(), out0);
    v128_store(ptr.add(8).cast(), out1);
    v128_store(ptr.add(16).cast(), out2);
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
  debug_assert_eq!(width & 1, 0);
  debug_assert!(y.len() >= width);
  debug_assert!(uv_half.len() >= width);
  debug_assert!(rgb_out.len() >= width * 3);

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

    // High-bit-packed samples: shift right by `16 - BITS`.
    let shr = (16 - BITS) as u32;

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

      write_rgb_16(r_u8, g_u8, b_u8, rgb_out.as_mut_ptr().add(x * 3));

      x += 16;
    }

    if x < width {
      scalar::p_n_to_rgb_row::<BITS>(
        &y[x..width],
        &uv_half[x..width],
        &mut rgb_out[x * 3..width * 3],
        width - x,
        matrix,
        full_range,
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
pub(crate) unsafe fn p_n_to_rgb_u16_row<const BITS: u32>(
  y: &[u16],
  uv_half: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0);
  debug_assert!(y.len() >= width);
  debug_assert!(uv_half.len() >= width);
  debug_assert!(rgb_out.len() >= width * 3);

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

    // High-bit-packed samples: shift right by `16 - BITS`.
    let shr = (16 - BITS) as u32;

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

      let r_lo = clamp_u10_wasm(i16x8_add_sat(y_scaled_lo, r_dup_lo), zero_v, max_v);
      let r_hi = clamp_u10_wasm(i16x8_add_sat(y_scaled_hi, r_dup_hi), zero_v, max_v);
      let g_lo = clamp_u10_wasm(i16x8_add_sat(y_scaled_lo, g_dup_lo), zero_v, max_v);
      let g_hi = clamp_u10_wasm(i16x8_add_sat(y_scaled_hi, g_dup_hi), zero_v, max_v);
      let b_lo = clamp_u10_wasm(i16x8_add_sat(y_scaled_lo, b_dup_lo), zero_v, max_v);
      let b_hi = clamp_u10_wasm(i16x8_add_sat(y_scaled_hi, b_dup_hi), zero_v, max_v);

      let dst = rgb_out.as_mut_ptr().add(x * 3);
      write_rgb_u16_8(r_lo, g_lo, b_lo, dst);
      write_rgb_u16_8(r_hi, g_hi, b_hi, dst.add(24));

      x += 16;
    }

    if x < width {
      scalar::p_n_to_rgb_u16_row::<BITS>(
        &y[x..width],
        &uv_half[x..width],
        &mut rgb_out[x * 3..width * 3],
        width - x,
        matrix,
        full_range,
      );
    }
  }
}

/// Deinterleaves 16 `u16` elements at `ptr` into `(u_vec, v_vec)` —
/// two 128‑bit vectors each holding 8 `u16` samples. Wasm's
/// `u8x16_swizzle` is semantically equivalent to SSSE3
/// `_mm_shuffle_epi8` (indices ≥ 16 zero the lane), so the same
/// split‑mask pattern applies. `i8x16_shuffle` is used for the
/// cross‑vector 64‑bit recombine.
///
/// # Safety
///
/// `ptr` must point to at least 32 readable bytes (16 `u16`
/// elements). Caller must have simd128 enabled at compile time.
#[inline(always)]
unsafe fn deinterleave_uv_u16_wasm(ptr: *const u16) -> (v128, v128) {
  unsafe {
    // Pack evens (U's) into low 8 bytes, odds (V's) into high 8 bytes.
    let split_mask = i8x16(0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15);

    let chunk0 = v128_load(ptr.cast());
    let chunk1 = v128_load(ptr.add(8).cast());

    let s0 = u8x16_swizzle(chunk0, split_mask);
    let s1 = u8x16_swizzle(chunk1, split_mask);

    // u_vec = low 8 bytes of s0 + low 8 bytes of s1.
    // v_vec = high 8 bytes of s0 + high 8 bytes of s1.
    let u_vec = i8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23>(s0, s1);
    let v_vec =
      i8x16_shuffle::<8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31>(s0, s1);
    (u_vec, v_vec)
  }
}

/// WASM simd128 NV12 → packed RGB (UV-ordered chroma). Thin wrapper
/// over [`nv12_or_nv21_to_rgb_row_impl`] with `SWAP_UV = false`.
///
/// # Safety
///
/// Same as [`nv12_or_nv21_to_rgb_row_impl`].
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn nv12_to_rgb_row(
  y: &[u8],
  uv_half: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    nv12_or_nv21_to_rgb_row_impl::<false>(y, uv_half, rgb_out, width, matrix, full_range);
  }
}

/// WASM simd128 NV21 → packed RGB (VU-ordered chroma). Thin wrapper
/// over [`nv12_or_nv21_to_rgb_row_impl`] with `SWAP_UV = true`.
///
/// # Safety
///
/// Same as [`nv12_or_nv21_to_rgb_row_impl`].
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn nv21_to_rgb_row(
  y: &[u8],
  vu_half: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    nv12_or_nv21_to_rgb_row_impl::<true>(y, vu_half, rgb_out, width, matrix, full_range);
  }
}

/// Shared wasm simd128 NV12/NV21 kernel. `SWAP_UV` selects chroma
/// byte order at compile time.
///
/// # Safety
///
/// 1. **simd128 must be enabled at compile time** (same obligation as
///    [`yuv_420_to_rgb_row`]).
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`.
/// 4. `uv_or_vu_half.len() >= width` (16 interleaved bytes per 16 Y pixels).
/// 5. `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "simd128")]
unsafe fn nv12_or_nv21_to_rgb_row_impl<const SWAP_UV: bool>(
  y: &[u8],
  uv_or_vu_half: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0, "NV12/NV21 require even width");
  debug_assert!(y.len() >= width);
  debug_assert!(uv_or_vu_half.len() >= width);
  debug_assert!(rgb_out.len() >= width * 3);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params(full_range);
  const RND: i32 = 1 << 14;

  // SAFETY: simd128 availability is the caller's compile‑time
  // obligation; all pointer adds below are bounded by the
  // `while x + 16 <= width` condition and the caller‑promised slice
  // lengths.
  unsafe {
    let rnd_v = i32x4_splat(RND);
    let y_off_v = i16x8_splat(y_off as i16);
    let y_scale_v = i32x4_splat(y_scale);
    let c_scale_v = i32x4_splat(c_scale);
    let mid128 = i16x8_splat(128);
    let cru = i32x4_splat(coeffs.r_u());
    let crv = i32x4_splat(coeffs.r_v());
    let cgu = i32x4_splat(coeffs.g_u());
    let cgv = i32x4_splat(coeffs.g_v());
    let cbu = i32x4_splat(coeffs.b_u());
    let cbv = i32x4_splat(coeffs.b_v());

    let mut x = 0usize;
    while x + 16 <= width {
      let y_vec = v128_load(y.as_ptr().add(x).cast());
      // 16 Y pixels → 8 chroma pairs = 16 interleaved bytes at
      // offset `x` in the chroma row.
      let uv_vec = v128_load(uv_or_vu_half.as_ptr().add(x).cast());

      // Deinterleave: `even_bytes` pulls even-offset bytes into low
      // 8, `odd_bytes` pulls odd-offset bytes. For NV12 that's
      // (U, V); for NV21 the roles swap.
      let even_bytes = i8x16_shuffle::<
        0,
        2,
        4,
        6,
        8,
        10,
        12,
        14, //
        0,
        2,
        4,
        6,
        8,
        10,
        12,
        14, //
      >(uv_vec, uv_vec);
      let odd_bytes = i8x16_shuffle::<
        1,
        3,
        5,
        7,
        9,
        11,
        13,
        15, //
        1,
        3,
        5,
        7,
        9,
        11,
        13,
        15, //
      >(uv_vec, uv_vec);
      let (u_bytes, v_bytes) = if SWAP_UV {
        (odd_bytes, even_bytes)
      } else {
        (even_bytes, odd_bytes)
      };
      let u_i16_zero = u16x8_extend_low_u8x16(u_bytes);
      let v_i16_zero = u16x8_extend_low_u8x16(v_bytes);

      let u_i16 = i16x8_sub(u_i16_zero, mid128);
      let v_i16 = i16x8_sub(v_i16_zero, mid128);

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

      let y_low_i16 = u8_low_to_i16x8(y_vec);
      let y_high_i16 = u8_high_to_i16x8(y_vec);
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

      write_rgb_16(r_u8, g_u8, b_u8, rgb_out.as_mut_ptr().add(x * 3));

      x += 16;
    }

    if x < width {
      if SWAP_UV {
        scalar::nv21_to_rgb_row(
          &y[x..width],
          &uv_or_vu_half[x..width],
          &mut rgb_out[x * 3..width * 3],
          width - x,
          matrix,
          full_range,
        );
      } else {
        scalar::nv12_to_rgb_row(
          &y[x..width],
          &uv_or_vu_half[x..width],
          &mut rgb_out[x * 3..width * 3],
          width - x,
          matrix,
          full_range,
        );
      }
    }
  }
}

// ---- helpers -----------------------------------------------------------

/// `>>_a 15` shift (arithmetic, sign‑extending).
#[inline(always)]
fn q15_shift(v: v128) -> v128 {
  i32x4_shr(v, 15)
}

/// Computes one i16x8 chroma channel vector from the 4 × i32x4 chroma
/// inputs. Mirrors the scalar
/// `(coeff_u * u_d + coeff_v * v_d + RND) >> 15`, then
/// saturating‑packs to i16x8. No lane fixup needed at 128 bits.
#[inline(always)]
fn chroma_i16x8(
  cu: v128,
  cv: v128,
  u_d_lo: v128,
  v_d_lo: v128,
  u_d_hi: v128,
  v_d_hi: v128,
  rnd: v128,
) -> v128 {
  let lo = i32x4_shr(
    i32x4_add(i32x4_add(i32x4_mul(cu, u_d_lo), i32x4_mul(cv, v_d_lo)), rnd),
    15,
  );
  let hi = i32x4_shr(
    i32x4_add(i32x4_add(i32x4_mul(cu, u_d_hi), i32x4_mul(cv, v_d_hi)), rnd),
    15,
  );
  i16x8_narrow_i32x4(lo, hi)
}

/// `(Y - y_off) * y_scale + RND >> 15` applied to an i16x8 vector,
/// returned as i16x8.
#[inline(always)]
fn scale_y(y_i16: v128, y_off_v: v128, y_scale_v: v128, rnd: v128) -> v128 {
  let shifted = i16x8_sub(y_i16, y_off_v);
  let lo_i32 = i32x4_extend_low_i16x8(shifted);
  let hi_i32 = i32x4_extend_high_i16x8(shifted);
  let lo_scaled = i32x4_shr(i32x4_add(i32x4_mul(lo_i32, y_scale_v), rnd), 15);
  let hi_scaled = i32x4_shr(i32x4_add(i32x4_mul(hi_i32, y_scale_v), rnd), 15);
  i16x8_narrow_i32x4(lo_scaled, hi_scaled)
}

/// Widens the low 8 bytes of a u8x16 to i16x8 (zero‑extended since
/// Y ∈ [0, 255] fits in non‑negative i16).
#[inline(always)]
fn u8_low_to_i16x8(v: v128) -> v128 {
  // i8x16_shuffle picks bytes pairwise: for each output i16 lane i,
  // take byte i of the source as the low byte and pad with a zero
  // byte from the all‑zero operand.
  i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(v, i16x8_splat(0))
}

/// Widens the high 8 bytes of a u8x16 to i16x8 (zero‑extended).
#[inline(always)]
fn u8_high_to_i16x8(v: v128) -> v128 {
  i8x16_shuffle::<8, 16, 9, 17, 10, 18, 11, 19, 12, 20, 13, 21, 14, 22, 15, 23>(v, i16x8_splat(0))
}

/// Duplicates the low 4 × i16 lanes of `chroma` into 8 lanes
/// `[c0,c0, c1,c1, c2,c2, c3,c3]` — nearest‑neighbor upsample for the
/// low 8 Y lanes of a 16‑pixel block.
#[inline(always)]
fn dup_lo(chroma: v128) -> v128 {
  i8x16_shuffle::<0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7, 6, 7>(chroma, chroma)
}

/// Duplicates the high 4 × i16 lanes of `chroma` into 8 lanes
/// `[c4,c4, c5,c5, c6,c6, c7,c7]` — upsample for the high 8 Y lanes.
#[inline(always)]
fn dup_hi(chroma: v128) -> v128 {
  i8x16_shuffle::<8, 9, 8, 9, 10, 11, 10, 11, 12, 13, 12, 13, 14, 15, 14, 15>(chroma, chroma)
}

/// Writes 16 pixels of packed RGB (48 bytes) from three u8x16 channel
/// vectors, using the SSSE3‑style 3‑way interleave pattern. `u8x16_swizzle`
/// treats indices ≥ 16 as "zero the lane" — same semantics as
/// `_mm_shuffle_epi8`, so the same shuffle masks apply.
///
/// # Safety
///
/// `ptr` must point to at least 48 writable bytes.
#[inline(always)]
unsafe fn write_rgb_16(r: v128, g: v128, b: v128, ptr: *mut u8) {
  unsafe {
    // Block 0 (bytes 0..16): [R0,G0,B0, R1,G1,B1, ..., R5].
    // `-1` as i8 is 0xFF ≥ 16 → zeroes that output lane.
    let r0 = i8x16(0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1, -1, 5);
    let g0 = i8x16(-1, 0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1, -1);
    let b0 = i8x16(-1, -1, 0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1);
    let out0 = v128_or(
      v128_or(u8x16_swizzle(r, r0), u8x16_swizzle(g, g0)),
      u8x16_swizzle(b, b0),
    );

    // Block 1 (bytes 16..32): [G5,B5, R6,G6,B6, ..., G10].
    let r1 = i8x16(-1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1, 10, -1);
    let g1 = i8x16(5, -1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1, 10);
    let b1 = i8x16(-1, 5, -1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1);
    let out1 = v128_or(
      v128_or(u8x16_swizzle(r, r1), u8x16_swizzle(g, g1)),
      u8x16_swizzle(b, b1),
    );

    // Block 2 (bytes 32..48): [B10, R11,G11,B11, ..., B15].
    let r2 = i8x16(
      -1, 11, -1, -1, 12, -1, -1, 13, -1, -1, 14, -1, -1, 15, -1, -1,
    );
    let g2 = i8x16(
      -1, -1, 11, -1, -1, 12, -1, -1, 13, -1, -1, 14, -1, -1, 15, -1,
    );
    let b2 = i8x16(
      10, -1, -1, 11, -1, -1, 12, -1, -1, 13, -1, -1, 14, -1, -1, 15,
    );
    let out2 = v128_or(
      v128_or(u8x16_swizzle(r, r2), u8x16_swizzle(g, g2)),
      u8x16_swizzle(b, b2),
    );

    v128_store(ptr.cast(), out0);
    v128_store(ptr.add(16).cast(), out1);
    v128_store(ptr.add(32).cast(), out2);
  }
}

// ===== BGR ↔ RGB byte swap ==============================================

/// WASM simd128 BGR ↔ RGB byte swap. 16 pixels per iteration via the
/// same 7‑shuffle + 4‑OR pattern as the x86 / NEON backends.
/// `u8x16_swizzle` matches `_mm_shuffle_epi8` semantics (indices ≥ 16
/// zero the output lane), so the mask values translate directly.
///
/// # Safety
///
/// 1. simd128 must be enabled at compile time.
/// 2. `input.len() >= 3 * width`.
/// 3. `output.len() >= 3 * width`.
/// 4. `input` / `output` must not alias.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn bgr_rgb_swap_row(input: &[u8], output: &mut [u8], width: usize) {
  debug_assert!(input.len() >= width * 3, "input row too short");
  debug_assert!(output.len() >= width * 3, "output row too short");

  unsafe {
    // Precomputed byte‑shuffle masks. See the x86_common::swap_rb_16_pixels
    // comments for the derivation — identical pattern at 128‑bit width.
    let m00 = i8x16(2, 1, 0, 5, 4, 3, 8, 7, 6, 11, 10, 9, 14, 13, 12, -1);
    let m01 = i8x16(
      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
    );
    let m10 = i8x16(
      -1, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    );
    let m11 = i8x16(0, -1, 4, 3, 2, 7, 6, 5, 10, 9, 8, 13, 12, 11, -1, 15);
    let m12 = i8x16(
      -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1,
    );
    let m20 = i8x16(
      14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    );
    let m21 = i8x16(-1, 3, 2, 1, 6, 5, 4, 9, 8, 7, 12, 11, 10, 15, 14, 13);

    let mut x = 0usize;
    while x + 16 <= width {
      let in0 = v128_load(input.as_ptr().add(x * 3).cast());
      let in1 = v128_load(input.as_ptr().add(x * 3 + 16).cast());
      let in2 = v128_load(input.as_ptr().add(x * 3 + 32).cast());

      let out0 = v128_or(u8x16_swizzle(in0, m00), u8x16_swizzle(in1, m01));
      let out1 = v128_or(
        v128_or(u8x16_swizzle(in0, m10), u8x16_swizzle(in1, m11)),
        u8x16_swizzle(in2, m12),
      );
      let out2 = v128_or(u8x16_swizzle(in1, m20), u8x16_swizzle(in2, m21));

      v128_store(output.as_mut_ptr().add(x * 3).cast(), out0);
      v128_store(output.as_mut_ptr().add(x * 3 + 16).cast(), out1);
      v128_store(output.as_mut_ptr().add(x * 3 + 32).cast(), out2);

      x += 16;
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

/// WASM simd128 RGB → planar HSV. 16 pixels per iteration using
/// byte‑shuffle deinterleave + four f32x4 HSV groups. Mirrors the NEON
/// and x86 kernels op‑for‑op (true `f32x4_div` for the two divisions,
/// `v128_bitselect` for the branch cascade). Bit‑identical to
/// [`scalar::rgb_to_hsv_row`].
///
/// # Safety
///
/// 1. simd128 must be enabled at compile time.
/// 2. `rgb.len() >= 3 * width`; each output plane `>= width`.
#[inline]
#[target_feature(enable = "simd128")]
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
    while x + 16 <= width {
      let in0 = v128_load(rgb.as_ptr().add(x * 3).cast());
      let in1 = v128_load(rgb.as_ptr().add(x * 3 + 16).cast());
      let in2 = v128_load(rgb.as_ptr().add(x * 3 + 32).cast());

      // 3‑channel deinterleave — mirror of the x86 mask pattern.
      let mr0 = i8x16(0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
      let mr1 = i8x16(-1, -1, -1, -1, -1, -1, 2, 5, 8, 11, 14, -1, -1, -1, -1, -1);
      let mr2 = i8x16(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 4, 7, 10, 13);
      let r_u8 = v128_or(
        v128_or(u8x16_swizzle(in0, mr0), u8x16_swizzle(in1, mr1)),
        u8x16_swizzle(in2, mr2),
      );

      let mg0 = i8x16(1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
      let mg1 = i8x16(-1, -1, -1, -1, -1, 0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1);
      let mg2 = i8x16(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 5, 8, 11, 14);
      let g_u8 = v128_or(
        v128_or(u8x16_swizzle(in0, mg0), u8x16_swizzle(in1, mg1)),
        u8x16_swizzle(in2, mg2),
      );

      let mb0 = i8x16(2, 5, 8, 11, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
      let mb1 = i8x16(-1, -1, -1, -1, -1, 1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1);
      let mb2 = i8x16(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 3, 6, 9, 12, 15);
      let b_u8 = v128_or(
        v128_or(u8x16_swizzle(in0, mb0), u8x16_swizzle(in1, mb1)),
        u8x16_swizzle(in2, mb2),
      );

      // Widen each u8x16 to 4 f32x4 groups.
      let (r0, r1, r2, r3) = u8x16_to_f32x4_quad(r_u8);
      let (g0, g1, g2, g3) = u8x16_to_f32x4_quad(g_u8);
      let (b0, b1, b2, b3) = u8x16_to_f32x4_quad(b_u8);

      let (h0, s0, v0) = hsv_group(r0, g0, b0);
      let (h1, s1, v1) = hsv_group(r1, g1, b1);
      let (h2, s2, v2) = hsv_group(r2, g2, b2);
      let (h3, s3, v3) = hsv_group(r3, g3, b3);

      v128_store(
        h_out.as_mut_ptr().add(x).cast(),
        f32x4_quad_to_u8x16(h0, h1, h2, h3),
      );
      v128_store(
        s_out.as_mut_ptr().add(x).cast(),
        f32x4_quad_to_u8x16(s0, s1, s2, s3),
      );
      v128_store(
        v_out.as_mut_ptr().add(x).cast(),
        f32x4_quad_to_u8x16(v0, v1, v2, v3),
      );

      x += 16;
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

// ---- RGB→HSV helpers (wasm simd128) ----------------------------------

/// Widens a u8x16 to four f32x4 groups.
#[inline(always)]
fn u8x16_to_f32x4_quad(v: v128) -> (v128, v128, v128, v128) {
  // u8x16 → u16x8 × 2 → u32x4 × 4 → f32x4 × 4.
  let u16_lo = u16x8_extend_low_u8x16(v);
  let u16_hi = u16x8_extend_high_u8x16(v);
  let u32_0 = u32x4_extend_low_u16x8(u16_lo);
  let u32_1 = u32x4_extend_high_u16x8(u16_lo);
  let u32_2 = u32x4_extend_low_u16x8(u16_hi);
  let u32_3 = u32x4_extend_high_u16x8(u16_hi);
  (
    f32x4_convert_i32x4(u32_0),
    f32x4_convert_i32x4(u32_1),
    f32x4_convert_i32x4(u32_2),
    f32x4_convert_i32x4(u32_3),
  )
}

/// Packs four f32x4 vectors to one u8x16. Values are pre‑clamped to
/// [0, 255] so the two narrowing steps don't clip.
#[inline(always)]
fn f32x4_quad_to_u8x16(a: v128, b: v128, c: v128, d: v128) -> v128 {
  let ai = i32x4_trunc_sat_f32x4(a);
  let bi = i32x4_trunc_sat_f32x4(b);
  let ci = i32x4_trunc_sat_f32x4(c);
  let di = i32x4_trunc_sat_f32x4(d);
  // i32x4 × 2 → i16x8 (signed saturating — fits since values in [0, 255]).
  let ab = i16x8_narrow_i32x4(ai, bi);
  let cd = i16x8_narrow_i32x4(ci, di);
  // i16x8 × 2 → u8x16 (unsigned saturating).
  u8x16_narrow_i16x8(ab, cd)
}

/// HSV compute for 4 pixels in f32x4 lanes. Mirrors the scalar
/// `rgb_to_hsv_pixel` op‑for‑op; returns already‑clamped H/S/V values
/// as f32x4 awaiting the truncating cast in the caller.
#[inline(always)]
fn hsv_group(r: v128, g: v128, b: v128) -> (v128, v128, v128) {
  let zero = f32x4_splat(0.0);
  let half = f32x4_splat(0.5);
  let sixty = f32x4_splat(60.0);
  let one_twenty = f32x4_splat(120.0);
  let two_forty = f32x4_splat(240.0);
  let three_sixty = f32x4_splat(360.0);
  let one_seventy_nine = f32x4_splat(179.0);
  let two_fifty_five = f32x4_splat(255.0);

  let v = f32x4_max(f32x4_max(r, g), b);
  let min_rgb = f32x4_min(f32x4_min(r, g), b);
  let delta = f32x4_sub(v, min_rgb);

  // S = if v == 0 { 0 } else { 255 * delta / v }.
  let mask_v_zero = f32x4_eq(v, zero);
  let s_nonzero = f32x4_div(f32x4_mul(two_fifty_five, delta), v);
  // `v128_bitselect(a, b, mask)`: per‑bit, pick a where mask bit = 1,
  // else b. Mask from f32 compare is all‑ones in "true" lanes.
  let s = v128_bitselect(zero, s_nonzero, mask_v_zero);

  let mask_delta_zero = f32x4_eq(delta, zero);
  let mask_v_is_r = f32x4_eq(v, r);
  let mask_v_is_g = f32x4_eq(v, g);

  let h_r_raw = f32x4_div(f32x4_mul(sixty, f32x4_sub(g, b)), delta);
  let mask_neg = f32x4_lt(h_r_raw, zero);
  let h_r = v128_bitselect(f32x4_add(h_r_raw, three_sixty), h_r_raw, mask_neg);

  let h_g = f32x4_add(
    f32x4_div(f32x4_mul(sixty, f32x4_sub(b, r)), delta),
    one_twenty,
  );
  let h_b = f32x4_add(
    f32x4_div(f32x4_mul(sixty, f32x4_sub(r, g)), delta),
    two_forty,
  );

  // Cascade: delta == 0 → 0; v == r → h_r; v == g → h_g; else → h_b.
  let h_g_or_b = v128_bitselect(h_g, h_b, mask_v_is_g);
  let h_nonzero = v128_bitselect(h_r, h_g_or_b, mask_v_is_r);
  let hue = v128_bitselect(zero, h_nonzero, mask_delta_zero);

  // Quantize to scalar output ranges.
  let h_quant = f32x4_min(
    f32x4_max(f32x4_add(f32x4_mul(hue, half), half), zero),
    one_seventy_nine,
  );
  let s_quant = f32x4_min(f32x4_max(f32x4_add(s, half), zero), two_fifty_five);
  let v_quant = f32x4_min(f32x4_max(f32x4_add(v, half), zero), two_fifty_five);

  (h_quant, s_quant, v_quant)
}

#[cfg(all(test, feature = "std", target_feature = "simd128"))]
mod tests {
  use super::*;

  fn check_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
    let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
    let u: std::vec::Vec<u8> = (0..width / 2)
      .map(|i| ((i * 53 + 23) & 0xFF) as u8)
      .collect();
    let v: std::vec::Vec<u8> = (0..width / 2)
      .map(|i| ((i * 71 + 91) & 0xFF) as u8)
      .collect();
    let mut rgb_scalar = std::vec![0u8; width * 3];
    let mut rgb_wasm = std::vec![0u8; width * 3];

    scalar::yuv_420_to_rgb_row(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
    unsafe {
      yuv_420_to_rgb_row(&y, &u, &v, &mut rgb_wasm, width, matrix, full_range);
    }

    assert_eq!(rgb_scalar, rgb_wasm, "simd128 diverges from scalar");
  }

  #[test]
  fn simd128_matches_scalar_all_matrices_16() {
    for m in [
      ColorMatrix::Bt601,
      ColorMatrix::Bt709,
      ColorMatrix::Bt2020Ncl,
      ColorMatrix::Smpte240m,
      ColorMatrix::Fcc,
      ColorMatrix::YCgCo,
    ] {
      for full in [true, false] {
        check_equivalence(16, m, full);
      }
    }
  }

  #[test]
  fn simd128_matches_scalar_tail_widths() {
    for w in [18usize, 30, 34, 1922] {
      check_equivalence(w, ColorMatrix::Bt601, false);
    }
  }

  // ---- nv12_to_rgb_row equivalence ------------------------------------

  fn check_nv12_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
    let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
    let uv: std::vec::Vec<u8> = (0..width / 2)
      .flat_map(|i| [((i * 53 + 23) & 0xFF) as u8, ((i * 71 + 91) & 0xFF) as u8])
      .collect();
    let mut rgb_scalar = std::vec![0u8; width * 3];
    let mut rgb_wasm = std::vec![0u8; width * 3];

    scalar::nv12_to_rgb_row(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
    unsafe {
      nv12_to_rgb_row(&y, &uv, &mut rgb_wasm, width, matrix, full_range);
    }
    assert_eq!(rgb_scalar, rgb_wasm, "simd128 NV12 ≠ scalar");
  }

  #[test]
  fn simd128_nv12_matches_scalar_all_matrices_16() {
    for m in [
      ColorMatrix::Bt601,
      ColorMatrix::Bt709,
      ColorMatrix::Bt2020Ncl,
      ColorMatrix::Smpte240m,
      ColorMatrix::Fcc,
      ColorMatrix::YCgCo,
    ] {
      for full in [true, false] {
        check_nv12_equivalence(16, m, full);
      }
    }
  }

  #[test]
  fn simd128_nv12_matches_scalar_widths() {
    for w in [32usize, 1920, 18, 30, 34, 1922] {
      check_nv12_equivalence(w, ColorMatrix::Bt709, false);
    }
  }

  // ---- bgr_rgb_swap_row equivalence -----------------------------------

  fn check_swap_equivalence(width: usize) {
    let input: std::vec::Vec<u8> = (0..width * 3)
      .map(|i| ((i * 17 + 41) & 0xFF) as u8)
      .collect();
    let mut out_scalar = std::vec![0u8; width * 3];
    let mut out_wasm = std::vec![0u8; width * 3];

    scalar::bgr_rgb_swap_row(&input, &mut out_scalar, width);
    unsafe {
      bgr_rgb_swap_row(&input, &mut out_wasm, width);
    }
    assert_eq!(out_scalar, out_wasm, "simd128 swap diverges from scalar");
  }

  #[test]
  fn simd128_swap_matches_scalar() {
    for w in [1usize, 15, 16, 17, 31, 32, 1920, 1921] {
      check_swap_equivalence(w);
    }
  }

  // ---- nv21_to_rgb_row equivalence ------------------------------------

  fn check_nv21_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
    let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
    let vu: std::vec::Vec<u8> = (0..width / 2)
      .flat_map(|i| [((i * 53 + 23) & 0xFF) as u8, ((i * 71 + 91) & 0xFF) as u8])
      .collect();
    let mut rgb_scalar = std::vec![0u8; width * 3];
    let mut rgb_wasm = std::vec![0u8; width * 3];

    scalar::nv21_to_rgb_row(&y, &vu, &mut rgb_scalar, width, matrix, full_range);
    unsafe {
      nv21_to_rgb_row(&y, &vu, &mut rgb_wasm, width, matrix, full_range);
    }
    assert_eq!(
      rgb_scalar, rgb_wasm,
      "simd128 NV21 ≠ scalar (width={width}, matrix={matrix:?})"
    );
  }

  fn check_nv21_matches_nv12_swapped(width: usize, matrix: ColorMatrix, full_range: bool) {
    let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
    let uv: std::vec::Vec<u8> = (0..width / 2)
      .flat_map(|i| [((i * 53 + 23) & 0xFF) as u8, ((i * 71 + 91) & 0xFF) as u8])
      .collect();
    let mut vu = std::vec![0u8; width];
    for i in 0..width / 2 {
      vu[2 * i] = uv[2 * i + 1];
      vu[2 * i + 1] = uv[2 * i];
    }

    let mut rgb_nv12 = std::vec![0u8; width * 3];
    let mut rgb_nv21 = std::vec![0u8; width * 3];
    unsafe {
      nv12_to_rgb_row(&y, &uv, &mut rgb_nv12, width, matrix, full_range);
      nv21_to_rgb_row(&y, &vu, &mut rgb_nv21, width, matrix, full_range);
    }
    assert_eq!(
      rgb_nv12, rgb_nv21,
      "simd128 NV21 ≠ NV12 with byte-swapped chroma"
    );
  }

  #[test]
  fn nv21_wasm_matches_scalar_all_matrices_16() {
    for m in [
      ColorMatrix::Bt601,
      ColorMatrix::Bt709,
      ColorMatrix::Bt2020Ncl,
      ColorMatrix::Smpte240m,
      ColorMatrix::Fcc,
      ColorMatrix::YCgCo,
    ] {
      for full in [true, false] {
        check_nv21_equivalence(16, m, full);
      }
    }
  }

  #[test]
  fn nv21_wasm_matches_scalar_widths() {
    for w in [32usize, 1920, 18, 30, 34, 1922] {
      check_nv21_equivalence(w, ColorMatrix::Bt709, false);
    }
  }

  #[test]
  fn nv21_wasm_matches_nv12_swapped() {
    for w in [16usize, 30, 64, 1920] {
      check_nv21_matches_nv12_swapped(w, ColorMatrix::Bt709, false);
      check_nv21_matches_nv12_swapped(w, ColorMatrix::YCgCo, true);
    }
  }
  // ---- rgb_to_hsv_row equivalence --------------------------------------

  fn check_hsv_equivalence(rgb: &[u8], width: usize) {
    let mut h_s = std::vec![0u8; width];
    let mut s_s = std::vec![0u8; width];
    let mut v_s = std::vec![0u8; width];
    let mut h_k = std::vec![0u8; width];
    let mut s_k = std::vec![0u8; width];
    let mut v_k = std::vec![0u8; width];
    scalar::rgb_to_hsv_row(rgb, &mut h_s, &mut s_s, &mut v_s, width);
    unsafe {
      rgb_to_hsv_row(rgb, &mut h_k, &mut s_k, &mut v_k, width);
    }
    for (i, (a, b)) in h_s.iter().zip(h_k.iter()).enumerate() {
      assert!(
        a.abs_diff(*b) <= 1,
        "H divergence at pixel {i}: scalar={a} simd={b}"
      );
    }
    for (i, (a, b)) in s_s.iter().zip(s_k.iter()).enumerate() {
      assert!(
        a.abs_diff(*b) <= 1,
        "S divergence at pixel {i}: scalar={a} simd={b}"
      );
    }
    for (i, (a, b)) in v_s.iter().zip(v_k.iter()).enumerate() {
      assert!(
        a.abs_diff(*b) <= 1,
        "V divergence at pixel {i}: scalar={a} simd={b}"
      );
    }
  }

  #[test]
  fn simd128_hsv_matches_scalar() {
    let rgb: std::vec::Vec<u8> = (0..1921 * 3)
      .map(|i| ((i * 37 + 11) & 0xFF) as u8)
      .collect();
    for w in [1usize, 15, 16, 17, 31, 1920, 1921] {
      check_hsv_equivalence(&rgb[..w * 3], w);
    }
  }

  // ---- yuv420p10 scalar-equivalence -----------------------------------
  //
  // These tests compile only for `target_arch = "wasm32"` (via the
  // outer `target_feature = "simd128"` gate on the module). CI
  // executes them under wasmtime in the `test-wasm-simd128` job
  // (see `.github/workflows/ci.yml`): the lib is compiled for
  // `wasm32-wasip1` with `-C target-feature=+simd128` and
  // `CARGO_TARGET_WASM32_WASIP1_RUNNER=wasmtime run --` passes each
  // compiled `.wasm` test binary to wasmtime. Every scalar‑
  // equivalence check below runs on real SIMD instructions, not
  // just a compile check.

  fn p10_plane(n: usize, seed: usize) -> std::vec::Vec<u16> {
    (0..n)
      .map(|i| ((i * seed + seed * 3) & 0x3FF) as u16)
      .collect()
  }

  fn check_p10_u8_simd128_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
    let y = p10_plane(width, 37);
    let u = p10_plane(width / 2, 53);
    let v = p10_plane(width / 2, 71);
    let mut rgb_scalar = std::vec![0u8; width * 3];
    let mut rgb_simd = std::vec![0u8; width * 3];

    scalar::yuv_420p_n_to_rgb_row::<10>(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
    unsafe {
      yuv_420p_n_to_rgb_row::<10>(&y, &u, &v, &mut rgb_simd, width, matrix, full_range);
    }

    if rgb_scalar != rgb_simd {
      let first_diff = rgb_scalar
        .iter()
        .zip(rgb_simd.iter())
        .position(|(a, b)| a != b)
        .unwrap();
      panic!(
        "simd128 10→u8 diverges at byte {first_diff} (width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} simd={}",
        rgb_scalar[first_diff], rgb_simd[first_diff]
      );
    }
  }

  fn check_p10_u16_simd128_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
    let y = p10_plane(width, 37);
    let u = p10_plane(width / 2, 53);
    let v = p10_plane(width / 2, 71);
    let mut rgb_scalar = std::vec![0u16; width * 3];
    let mut rgb_simd = std::vec![0u16; width * 3];

    scalar::yuv_420p_n_to_rgb_u16_row::<10>(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
    unsafe {
      yuv_420p_n_to_rgb_u16_row::<10>(&y, &u, &v, &mut rgb_simd, width, matrix, full_range);
    }

    if rgb_scalar != rgb_simd {
      let first_diff = rgb_scalar
        .iter()
        .zip(rgb_simd.iter())
        .position(|(a, b)| a != b)
        .unwrap();
      panic!(
        "simd128 10→u16 diverges at elem {first_diff} (width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} simd={}",
        rgb_scalar[first_diff], rgb_simd[first_diff]
      );
    }
  }

  #[test]
  fn simd128_p10_u8_matches_scalar_all_matrices() {
    for m in [
      ColorMatrix::Bt601,
      ColorMatrix::Bt709,
      ColorMatrix::Bt2020Ncl,
      ColorMatrix::Smpte240m,
      ColorMatrix::Fcc,
      ColorMatrix::YCgCo,
    ] {
      for full in [true, false] {
        check_p10_u8_simd128_equivalence(16, m, full);
      }
    }
  }

  #[test]
  fn simd128_p10_u16_matches_scalar_all_matrices() {
    for m in [
      ColorMatrix::Bt601,
      ColorMatrix::Bt709,
      ColorMatrix::Bt2020Ncl,
      ColorMatrix::Smpte240m,
      ColorMatrix::Fcc,
      ColorMatrix::YCgCo,
    ] {
      for full in [true, false] {
        check_p10_u16_simd128_equivalence(16, m, full);
      }
    }
  }

  #[test]
  fn simd128_p10_matches_scalar_tail_widths() {
    for w in [18usize, 30, 34, 1922] {
      check_p10_u8_simd128_equivalence(w, ColorMatrix::Bt601, false);
      check_p10_u16_simd128_equivalence(w, ColorMatrix::Bt709, true);
    }
  }

  #[test]
  fn simd128_p10_matches_scalar_1920() {
    check_p10_u8_simd128_equivalence(1920, ColorMatrix::Bt709, false);
    check_p10_u16_simd128_equivalence(1920, ColorMatrix::Bt2020Ncl, false);
  }

  // ---- P010 simd128 scalar-equivalence --------------------------------

  fn p010_plane(n: usize, seed: usize) -> std::vec::Vec<u16> {
    (0..n)
      .map(|i| (((i * seed + seed * 3) & 0x3FF) as u16) << 6)
      .collect()
  }

  fn p010_uv_interleave(u: &[u16], v: &[u16]) -> std::vec::Vec<u16> {
    let pairs = u.len();
    debug_assert_eq!(u.len(), v.len());
    let mut out = std::vec::Vec::with_capacity(pairs * 2);
    for i in 0..pairs {
      out.push(u[i]);
      out.push(v[i]);
    }
    out
  }

  fn check_p010_u8_simd128_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
    let y = p010_plane(width, 37);
    let u = p010_plane(width / 2, 53);
    let v = p010_plane(width / 2, 71);
    let uv = p010_uv_interleave(&u, &v);
    let mut rgb_scalar = std::vec![0u8; width * 3];
    let mut rgb_simd = std::vec![0u8; width * 3];
    scalar::p_n_to_rgb_row::<10>(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
    unsafe {
      p_n_to_rgb_row::<10>(&y, &uv, &mut rgb_simd, width, matrix, full_range);
    }
    assert_eq!(rgb_scalar, rgb_simd, "simd128 P010→u8 diverges");
  }

  fn check_p010_u16_simd128_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
    let y = p010_plane(width, 37);
    let u = p010_plane(width / 2, 53);
    let v = p010_plane(width / 2, 71);
    let uv = p010_uv_interleave(&u, &v);
    let mut rgb_scalar = std::vec![0u16; width * 3];
    let mut rgb_simd = std::vec![0u16; width * 3];
    scalar::p_n_to_rgb_u16_row::<10>(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
    unsafe {
      p_n_to_rgb_u16_row::<10>(&y, &uv, &mut rgb_simd, width, matrix, full_range);
    }
    assert_eq!(rgb_scalar, rgb_simd, "simd128 P010→u16 diverges");
  }

  #[test]
  fn simd128_p010_u8_matches_scalar_all_matrices() {
    for m in [
      ColorMatrix::Bt601,
      ColorMatrix::Bt709,
      ColorMatrix::Bt2020Ncl,
      ColorMatrix::Smpte240m,
      ColorMatrix::Fcc,
      ColorMatrix::YCgCo,
    ] {
      for full in [true, false] {
        check_p010_u8_simd128_equivalence(16, m, full);
      }
    }
  }

  #[test]
  fn simd128_p010_u16_matches_scalar_all_matrices() {
    for m in [
      ColorMatrix::Bt601,
      ColorMatrix::Bt709,
      ColorMatrix::Bt2020Ncl,
      ColorMatrix::Smpte240m,
      ColorMatrix::Fcc,
      ColorMatrix::YCgCo,
    ] {
      for full in [true, false] {
        check_p010_u16_simd128_equivalence(16, m, full);
      }
    }
  }

  #[test]
  fn simd128_p010_matches_scalar_tail_widths() {
    for w in [18usize, 30, 34, 1922] {
      check_p010_u8_simd128_equivalence(w, ColorMatrix::Bt601, false);
      check_p010_u16_simd128_equivalence(w, ColorMatrix::Bt709, true);
    }
  }

  #[test]
  fn simd128_p010_matches_scalar_1920() {
    check_p010_u8_simd128_equivalence(1920, ColorMatrix::Bt709, false);
    check_p010_u16_simd128_equivalence(1920, ColorMatrix::Bt2020Ncl, false);
  }

  // ---- Generic BITS equivalence (12/14-bit coverage) ------------------

  fn planar_n_plane<const BITS: u32>(n: usize, seed: usize) -> std::vec::Vec<u16> {
    let mask = (1u32 << BITS) - 1;
    (0..n)
      .map(|i| ((i * seed + seed * 3) as u32 & mask) as u16)
      .collect()
  }

  fn p_n_packed_plane<const BITS: u32>(n: usize, seed: usize) -> std::vec::Vec<u16> {
    let mask = (1u32 << BITS) - 1;
    let shift = 16 - BITS;
    (0..n)
      .map(|i| (((i * seed + seed * 3) as u32 & mask) as u16) << shift)
      .collect()
  }

  fn check_planar_u8_simd128_equivalence_n<const BITS: u32>(
    width: usize,
    matrix: ColorMatrix,
    full_range: bool,
  ) {
    let y = planar_n_plane::<BITS>(width, 37);
    let u = planar_n_plane::<BITS>(width / 2, 53);
    let v = planar_n_plane::<BITS>(width / 2, 71);
    let mut rgb_scalar = std::vec![0u8; width * 3];
    let mut rgb_simd = std::vec![0u8; width * 3];
    scalar::yuv_420p_n_to_rgb_row::<BITS>(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
    unsafe {
      yuv_420p_n_to_rgb_row::<BITS>(&y, &u, &v, &mut rgb_simd, width, matrix, full_range);
    }
    assert_eq!(
      rgb_scalar, rgb_simd,
      "simd128 planar {BITS}-bit → u8 diverges"
    );
  }

  fn check_planar_u16_simd128_equivalence_n<const BITS: u32>(
    width: usize,
    matrix: ColorMatrix,
    full_range: bool,
  ) {
    let y = planar_n_plane::<BITS>(width, 37);
    let u = planar_n_plane::<BITS>(width / 2, 53);
    let v = planar_n_plane::<BITS>(width / 2, 71);
    let mut rgb_scalar = std::vec![0u16; width * 3];
    let mut rgb_simd = std::vec![0u16; width * 3];
    scalar::yuv_420p_n_to_rgb_u16_row::<BITS>(
      &y,
      &u,
      &v,
      &mut rgb_scalar,
      width,
      matrix,
      full_range,
    );
    unsafe {
      yuv_420p_n_to_rgb_u16_row::<BITS>(&y, &u, &v, &mut rgb_simd, width, matrix, full_range);
    }
    assert_eq!(
      rgb_scalar, rgb_simd,
      "simd128 planar {BITS}-bit → u16 diverges"
    );
  }

  fn check_pn_u8_simd128_equivalence_n<const BITS: u32>(
    width: usize,
    matrix: ColorMatrix,
    full_range: bool,
  ) {
    let y = p_n_packed_plane::<BITS>(width, 37);
    let u = p_n_packed_plane::<BITS>(width / 2, 53);
    let v = p_n_packed_plane::<BITS>(width / 2, 71);
    let uv = p010_uv_interleave(&u, &v);
    let mut rgb_scalar = std::vec![0u8; width * 3];
    let mut rgb_simd = std::vec![0u8; width * 3];
    scalar::p_n_to_rgb_row::<BITS>(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
    unsafe {
      p_n_to_rgb_row::<BITS>(&y, &uv, &mut rgb_simd, width, matrix, full_range);
    }
    assert_eq!(rgb_scalar, rgb_simd, "simd128 Pn {BITS}-bit → u8 diverges");
  }

  fn check_pn_u16_simd128_equivalence_n<const BITS: u32>(
    width: usize,
    matrix: ColorMatrix,
    full_range: bool,
  ) {
    let y = p_n_packed_plane::<BITS>(width, 37);
    let u = p_n_packed_plane::<BITS>(width / 2, 53);
    let v = p_n_packed_plane::<BITS>(width / 2, 71);
    let uv = p010_uv_interleave(&u, &v);
    let mut rgb_scalar = std::vec![0u16; width * 3];
    let mut rgb_simd = std::vec![0u16; width * 3];
    scalar::p_n_to_rgb_u16_row::<BITS>(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
    unsafe {
      p_n_to_rgb_u16_row::<BITS>(&y, &uv, &mut rgb_simd, width, matrix, full_range);
    }
    assert_eq!(rgb_scalar, rgb_simd, "simd128 Pn {BITS}-bit → u16 diverges");
  }

  #[test]
  fn simd128_p12_matches_scalar_all_matrices() {
    for m in [
      ColorMatrix::Bt601,
      ColorMatrix::Bt709,
      ColorMatrix::Bt2020Ncl,
      ColorMatrix::Smpte240m,
      ColorMatrix::Fcc,
      ColorMatrix::YCgCo,
    ] {
      for full in [true, false] {
        check_planar_u8_simd128_equivalence_n::<12>(16, m, full);
        check_planar_u16_simd128_equivalence_n::<12>(16, m, full);
        check_pn_u8_simd128_equivalence_n::<12>(16, m, full);
        check_pn_u16_simd128_equivalence_n::<12>(16, m, full);
      }
    }
  }

  #[test]
  fn simd128_p14_matches_scalar_all_matrices() {
    for m in [
      ColorMatrix::Bt601,
      ColorMatrix::Bt709,
      ColorMatrix::Bt2020Ncl,
      ColorMatrix::Smpte240m,
      ColorMatrix::Fcc,
      ColorMatrix::YCgCo,
    ] {
      for full in [true, false] {
        check_planar_u8_simd128_equivalence_n::<14>(16, m, full);
        check_planar_u16_simd128_equivalence_n::<14>(16, m, full);
      }
    }
  }

  #[test]
  fn simd128_p12_matches_scalar_tail_widths() {
    for w in [18usize, 30, 34, 1922] {
      check_planar_u8_simd128_equivalence_n::<12>(w, ColorMatrix::Bt601, false);
      check_planar_u16_simd128_equivalence_n::<12>(w, ColorMatrix::Bt709, true);
      check_pn_u8_simd128_equivalence_n::<12>(w, ColorMatrix::Bt601, false);
      check_pn_u16_simd128_equivalence_n::<12>(w, ColorMatrix::Bt2020Ncl, false);
    }
  }

  #[test]
  fn simd128_p14_matches_scalar_tail_widths() {
    for w in [18usize, 30, 34, 1922] {
      check_planar_u8_simd128_equivalence_n::<14>(w, ColorMatrix::Bt601, false);
      check_planar_u16_simd128_equivalence_n::<14>(w, ColorMatrix::Bt709, true);
    }
  }
}
