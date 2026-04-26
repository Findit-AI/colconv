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

use core::arch::wasm32::*;

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
  // SAFETY: caller-checked simd128 availability + slice bounds —
  // see [`yuv_420_to_rgb_or_rgba_row`] safety contract.
  unsafe {
    yuv_420_to_rgb_or_rgba_row::<false>(y, u_half, v_half, rgb_out, width, matrix, full_range);
  }
}

/// WASM simd128 YUV 4:2:0 → packed **RGBA** (8-bit). Same contract
/// as [`yuv_420_to_rgb_row`] but writes 4 bytes per pixel (R, G, B,
/// `0xFF`).
///
/// # Safety
///
/// 1. simd128 must be enabled at compile time.
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `u_half.len() >= width / 2`,
///    `v_half.len() >= width / 2`, `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn yuv_420_to_rgba_row(
  y: &[u8],
  u_half: &[u8],
  v_half: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller-checked simd128 availability + slice bounds —
  // see [`yuv_420_to_rgb_or_rgba_row`] safety contract.
  unsafe {
    yuv_420_to_rgb_or_rgba_row::<true>(y, u_half, v_half, rgba_out, width, matrix, full_range);
  }
}

/// Shared WASM simd128 kernel for [`yuv_420_to_rgb_row`]
/// (`ALPHA = false`, [`write_rgb_16`]) and [`yuv_420_to_rgba_row`]
/// (`ALPHA = true`, [`write_rgba_16`] with constant `0xFF` alpha).
/// Math is byte-identical to
/// `scalar::yuv_420_to_rgb_or_rgba_row::<ALPHA>`.
///
/// # Safety
///
/// Same as [`yuv_420_to_rgb_row`] / [`yuv_420_to_rgba_row`]; the
/// `out` slice must be `>= width * (if ALPHA { 4 } else { 3 })`
/// bytes long.
#[inline]
#[target_feature(enable = "simd128")]
unsafe fn yuv_420_to_rgb_or_rgba_row<const ALPHA: bool>(
  y: &[u8],
  u_half: &[u8],
  v_half: &[u8],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0, "YUV 4:2:0 requires even width");
  debug_assert!(y.len() >= width);
  debug_assert!(u_half.len() >= width / 2);
  debug_assert!(v_half.len() >= width / 2);
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp);

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
    // Constant opaque-alpha vector for the RGBA path; DCE'd when
    // ALPHA = false.
    let alpha_u8 = u8x16_splat(0xFF);

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

      if ALPHA {
        // 4‑way interleave → packed RGBA (64 bytes).
        write_rgba_16(r_u8, g_u8, b_u8, alpha_u8, out.as_mut_ptr().add(x * 4));
      } else {
        // 3‑way interleave → packed RGB (48 bytes).
        write_rgb_16(r_u8, g_u8, b_u8, out.as_mut_ptr().add(x * 3));
      }

      x += 16;
    }

    // Scalar tail for the 0..14 leftover pixels.
    if x < width {
      scalar::yuv_420_to_rgb_or_rgba_row::<ALPHA>(
        &y[x..width],
        &u_half[x / 2..width / 2],
        &v_half[x / 2..width / 2],
        &mut out[x * bpp..width * bpp],
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
  const { assert!(BITS == 9 || BITS == 10 || BITS == 12 || BITS == 14) };
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
  const { assert!(BITS == 9 || BITS == 10 || BITS == 12 || BITS == 14) };
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

      let r_lo = clamp_u16_max_wasm(i16x8_add_sat(y_scaled_lo, r_dup_lo), zero_v, max_v);
      let r_hi = clamp_u16_max_wasm(i16x8_add_sat(y_scaled_hi, r_dup_hi), zero_v, max_v);
      let g_lo = clamp_u16_max_wasm(i16x8_add_sat(y_scaled_lo, g_dup_lo), zero_v, max_v);
      let g_hi = clamp_u16_max_wasm(i16x8_add_sat(y_scaled_hi, g_dup_hi), zero_v, max_v);
      let b_lo = clamp_u16_max_wasm(i16x8_add_sat(y_scaled_lo, b_dup_lo), zero_v, max_v);
      let b_hi = clamp_u16_max_wasm(i16x8_add_sat(y_scaled_hi, b_dup_hi), zero_v, max_v);

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

/// Clamps an i16x8 vector to `[0, max]`. Used by native-depth u16
/// output paths (10/12/14 bit).
#[inline(always)]
fn clamp_u16_max_wasm(v: v128, zero_v: v128, max_v: v128) -> v128 {
  i16x8_min(i16x8_max(v, zero_v), max_v)
}

/// WASM simd128 YUV 4:4:4 planar 10/12/14-bit → packed **u8** RGB.
/// Const-generic over `BITS ∈ {10, 12, 14}`. Block size 16 pixels.
///
/// # Safety
///
/// 1. **simd128 must be enabled at compile time.**
/// 2. `y.len() >= width`, `u.len() >= width`, `v.len() >= width`,
///    `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn yuv_444p_n_to_rgb_row<const BITS: u32>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  const { assert!(BITS == 9 || BITS == 10 || BITS == 12 || BITS == 14) };
  debug_assert!(y.len() >= width);
  debug_assert!(u.len() >= width);
  debug_assert!(v.len() >= width);
  debug_assert!(rgb_out.len() >= width * 3);

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
    let mask_v = u16x8_splat(scalar::bits_mask::<BITS>());
    let cru = i32x4_splat(coeffs.r_u());
    let crv = i32x4_splat(coeffs.r_v());
    let cgu = i32x4_splat(coeffs.g_u());
    let cgv = i32x4_splat(coeffs.g_v());
    let cbu = i32x4_splat(coeffs.b_u());
    let cbv = i32x4_splat(coeffs.b_v());

    let mut x = 0usize;
    while x + 16 <= width {
      // 16 Y + 16 U + 16 V per iter. Full-width chroma (two u16x8
      // loads each) — no horizontal duplication, 4:4:4 is 1:1.
      let y_low_i16 = v128_and(v128_load(y.as_ptr().add(x).cast()), mask_v);
      let y_high_i16 = v128_and(v128_load(y.as_ptr().add(x + 8).cast()), mask_v);
      let u_lo_vec = v128_and(v128_load(u.as_ptr().add(x).cast()), mask_v);
      let u_hi_vec = v128_and(v128_load(u.as_ptr().add(x + 8).cast()), mask_v);
      let v_lo_vec = v128_and(v128_load(v.as_ptr().add(x).cast()), mask_v);
      let v_hi_vec = v128_and(v128_load(v.as_ptr().add(x + 8).cast()), mask_v);

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

      write_rgb_16(r_u8, g_u8, b_u8, rgb_out.as_mut_ptr().add(x * 3));

      x += 16;
    }

    if x < width {
      scalar::yuv_444p_n_to_rgb_row::<BITS>(
        &y[x..width],
        &u[x..width],
        &v[x..width],
        &mut rgb_out[x * 3..width * 3],
        width - x,
        matrix,
        full_range,
      );
    }
  }
}

/// WASM simd128 YUV 4:4:4 planar 10/12/14-bit → **native-depth u16** RGB.
/// Const-generic over `BITS ∈ {10, 12, 14}`. 16 pixels per iter.
///
/// # Safety
///
/// Same as [`yuv_444p_n_to_rgb_row`] but `rgb_out: &mut [u16]`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn yuv_444p_n_to_rgb_u16_row<const BITS: u32>(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  const { assert!(BITS == 9 || BITS == 10 || BITS == 12 || BITS == 14) };
  debug_assert!(y.len() >= width);
  debug_assert!(u.len() >= width);
  debug_assert!(v.len() >= width);
  debug_assert!(rgb_out.len() >= width * 3);

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
      let y_low_i16 = v128_and(v128_load(y.as_ptr().add(x).cast()), mask_v);
      let y_high_i16 = v128_and(v128_load(y.as_ptr().add(x + 8).cast()), mask_v);
      let u_lo_vec = v128_and(v128_load(u.as_ptr().add(x).cast()), mask_v);
      let u_hi_vec = v128_and(v128_load(u.as_ptr().add(x + 8).cast()), mask_v);
      let v_lo_vec = v128_and(v128_load(v.as_ptr().add(x).cast()), mask_v);
      let v_hi_vec = v128_and(v128_load(v.as_ptr().add(x + 8).cast()), mask_v);

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

      let r_lo = clamp_u16_max_wasm(i16x8_add_sat(y_scaled_lo, r_chroma_lo), zero_v, max_v);
      let r_hi = clamp_u16_max_wasm(i16x8_add_sat(y_scaled_hi, r_chroma_hi), zero_v, max_v);
      let g_lo = clamp_u16_max_wasm(i16x8_add_sat(y_scaled_lo, g_chroma_lo), zero_v, max_v);
      let g_hi = clamp_u16_max_wasm(i16x8_add_sat(y_scaled_hi, g_chroma_hi), zero_v, max_v);
      let b_lo = clamp_u16_max_wasm(i16x8_add_sat(y_scaled_lo, b_chroma_lo), zero_v, max_v);
      let b_hi = clamp_u16_max_wasm(i16x8_add_sat(y_scaled_hi, b_chroma_hi), zero_v, max_v);

      let dst = rgb_out.as_mut_ptr().add(x * 3);
      write_rgb_u16_8(r_lo, g_lo, b_lo, dst);
      write_rgb_u16_8(r_hi, g_hi, b_hi, dst.add(24));

      x += 16;
    }

    if x < width {
      scalar::yuv_444p_n_to_rgb_u16_row::<BITS>(
        &y[x..width],
        &u[x..width],
        &v[x..width],
        &mut rgb_out[x * 3..width * 3],
        width - x,
        matrix,
        full_range,
      );
    }
  }
}

/// WASM simd128 YUV 4:4:4 planar **16-bit** → packed **u8** RGB.
/// Stays on the i32 Q15 pipeline. 16 pixels per iter.
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
  debug_assert!(y.len() >= width);
  debug_assert!(u.len() >= width);
  debug_assert!(v.len() >= width);
  debug_assert!(rgb_out.len() >= width * 3);

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

      write_rgb_16(r_u8, g_u8, b_u8, rgb_out.as_mut_ptr().add(x * 3));
      x += 16;
    }

    if x < width {
      scalar::yuv_444p16_to_rgb_row(
        &y[x..width],
        &u[x..width],
        &v[x..width],
        &mut rgb_out[x * 3..width * 3],
        width - x,
        matrix,
        full_range,
      );
    }
  }
}

/// WASM simd128 YUV 4:4:4 planar **16-bit** → packed **u16** RGB.
/// Falls through to scalar — simd128 has no native `i64x2` arithmetic
/// shift, and the `srai64_15` bias trick at 128 bits (single i64 pair
/// per lane) is not cheaper than scalar. Same rationale as
/// [`yuv_420p16_to_rgb_u16_row`].
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
  debug_assert!(y.len() >= width);
  debug_assert!(u.len() >= width);
  debug_assert!(v.len() >= width);
  debug_assert!(rgb_out.len() >= width * 3);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<16, 16>(full_range);
  const RND_I64: i64 = 1 << 14;
  const RND_I32: i32 = 1 << 14;

  unsafe {
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

      write_rgb_u16_8(r_u16, g_u16, b_u16, rgb_out.as_mut_ptr().add(x * 3));
      x += 8;
    }

    if x < width {
      scalar::yuv_444p16_to_rgb_u16_row(
        &y[x..width],
        &u[x..width],
        &v[x..width],
        &mut rgb_out[x * 3..width * 3],
        width - x,
        matrix,
        full_range,
      );
    }
  }
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

      let r_lo = clamp_u16_max_wasm(i16x8_add_sat(y_scaled_lo, r_dup_lo), zero_v, max_v);
      let r_hi = clamp_u16_max_wasm(i16x8_add_sat(y_scaled_hi, r_dup_hi), zero_v, max_v);
      let g_lo = clamp_u16_max_wasm(i16x8_add_sat(y_scaled_lo, g_dup_lo), zero_v, max_v);
      let g_hi = clamp_u16_max_wasm(i16x8_add_sat(y_scaled_hi, g_dup_hi), zero_v, max_v);
      let b_lo = clamp_u16_max_wasm(i16x8_add_sat(y_scaled_lo, b_dup_lo), zero_v, max_v);
      let b_hi = clamp_u16_max_wasm(i16x8_add_sat(y_scaled_hi, b_dup_hi), zero_v, max_v);

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

/// WASM simd128 NV12 → packed RGB. Thin wrapper over
/// [`nv12_or_nv21_to_rgb_or_rgba_row_impl`] with
/// `SWAP_UV = false, ALPHA = false`.
///
/// # Safety
///
/// Same contract as [`nv12_or_nv21_to_rgb_or_rgba_row_impl`]:
///
/// 1. **simd128 must be enabled at compile time.** WASM has no
///    runtime CPU detection — the module's SIMD support is fixed at
///    produce time.
/// 2. `width & 1 == 0` (4:2:0 requires even width).
/// 3. `y.len() >= width`.
/// 4. `uv_half.len() >= width` (interleaved UV bytes, 2 per chroma pair).
/// 5. `rgb_out.len() >= 3 * width`.
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
  unsafe {
    nv12_or_nv21_to_rgb_or_rgba_row_impl::<false, false>(
      y, uv_half, rgb_out, width, matrix, full_range,
    );
  }
}

/// WASM simd128 NV21 → packed RGB. Thin wrapper over
/// [`nv12_or_nv21_to_rgb_or_rgba_row_impl`] with
/// `SWAP_UV = true, ALPHA = false`.
///
/// # Safety
///
/// Same contract as [`nv12_to_rgb_row`]; `vu_half` carries the same
/// number of bytes (`>= width`) but in V-then-U order per chroma
/// pair.
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
  unsafe {
    nv12_or_nv21_to_rgb_or_rgba_row_impl::<true, false>(
      y, vu_half, rgb_out, width, matrix, full_range,
    );
  }
}

/// WASM simd128 NV12 → packed RGBA. Same contract as
/// [`nv12_to_rgb_row`] but writes 4 bytes per pixel via
/// [`write_rgba_16`]. `rgba_out.len() >= 4 * width`.
///
/// # Safety
///
/// Same as [`nv12_to_rgb_row`] except the output slice must be
/// `>= 4 * width` bytes (one extra byte per pixel for the opaque
/// alpha).
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn nv12_to_rgba_row(
  y: &[u8],
  uv_half: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  unsafe {
    nv12_or_nv21_to_rgb_or_rgba_row_impl::<false, true>(
      y, uv_half, rgba_out, width, matrix, full_range,
    );
  }
}

/// WASM simd128 NV21 → packed RGBA. Same contract as
/// [`nv21_to_rgb_row`] but writes 4 bytes per pixel via
/// [`write_rgba_16`]. `rgba_out.len() >= 4 * width`.
///
/// # Safety
///
/// Same as [`nv21_to_rgb_row`] except the output slice must be
/// `>= 4 * width` bytes.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn nv21_to_rgba_row(
  y: &[u8],
  vu_half: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  unsafe {
    nv12_or_nv21_to_rgb_or_rgba_row_impl::<true, true>(
      y, vu_half, rgba_out, width, matrix, full_range,
    );
  }
}

/// Shared wasm simd128 NV12/NV21 kernel at 3 bpp (RGB) or 4 bpp +
/// opaque alpha (RGBA). `SWAP_UV` selects chroma byte order;
/// `ALPHA = true` writes via [`write_rgba_16`], `ALPHA = false` via
/// [`write_rgb_16`]. Both const generics drive compile-time
/// monomorphization.
///
/// # Safety
///
/// 1. **simd128 must be enabled at compile time.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`.
/// 4. `uv_or_vu_half.len() >= width` (16 interleaved bytes per 16 Y pixels).
/// 5. `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
#[inline]
#[target_feature(enable = "simd128")]
unsafe fn nv12_or_nv21_to_rgb_or_rgba_row_impl<
  const SWAP_UV: bool,
  const ALPHA: bool,
>(
  y: &[u8],
  uv_or_vu_half: &[u8],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0, "NV12/NV21 require even width");
  debug_assert!(y.len() >= width);
  debug_assert!(uv_or_vu_half.len() >= width);
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp);

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
    let alpha_u8 = u8x16_splat(0xFF);

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

      if ALPHA {
        write_rgba_16(r_u8, g_u8, b_u8, alpha_u8, out.as_mut_ptr().add(x * 4));
      } else {
        write_rgb_16(r_u8, g_u8, b_u8, out.as_mut_ptr().add(x * 3));
      }

      x += 16;
    }

    if x < width {
      let tail_y = &y[x..width];
      let tail_uv = &uv_or_vu_half[x..width];
      let tail_w = width - x;
      let tail_out = &mut out[x * bpp..width * bpp];
      match (SWAP_UV, ALPHA) {
        (false, false) => {
          scalar::nv12_to_rgb_row(tail_y, tail_uv, tail_out, tail_w, matrix, full_range)
        }
        (true, false) => {
          scalar::nv21_to_rgb_row(tail_y, tail_uv, tail_out, tail_w, matrix, full_range)
        }
        (false, true) => {
          scalar::nv12_to_rgba_row(tail_y, tail_uv, tail_out, tail_w, matrix, full_range)
        }
        (true, true) => {
          scalar::nv21_to_rgba_row(tail_y, tail_uv, tail_out, tail_w, matrix, full_range)
        }
      }
    }
  }
}

/// wasm simd128 NV24 → packed RGB (UV-ordered, 4:4:4).
///
/// # Safety
///
/// Same as [`nv24_or_nv42_to_rgb_or_rgba_row_impl`].
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn nv24_to_rgb_row(
  y: &[u8],
  uv: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    nv24_or_nv42_to_rgb_or_rgba_row_impl::<false, false>(y, uv, rgb_out, width, matrix, full_range);
  }
}

/// wasm simd128 NV42 → packed RGB (VU-ordered, 4:4:4).
///
/// # Safety
///
/// Same as [`nv24_or_nv42_to_rgb_or_rgba_row_impl`].
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn nv42_to_rgb_row(
  y: &[u8],
  vu: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    nv24_or_nv42_to_rgb_or_rgba_row_impl::<true, false>(y, vu, rgb_out, width, matrix, full_range);
  }
}

/// wasm simd128 NV24 → packed RGBA (UV-ordered, 4:4:4, opaque alpha).
///
/// # Safety
///
/// Same as [`nv24_or_nv42_to_rgb_or_rgba_row_impl`] but `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn nv24_to_rgba_row(
  y: &[u8],
  uv: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    nv24_or_nv42_to_rgb_or_rgba_row_impl::<false, true>(y, uv, rgba_out, width, matrix, full_range);
  }
}

/// wasm simd128 NV42 → packed RGBA (VU-ordered, 4:4:4, opaque alpha).
///
/// # Safety
///
/// Same as [`nv24_or_nv42_to_rgb_or_rgba_row_impl`] but `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn nv42_to_rgba_row(
  y: &[u8],
  vu: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
    nv24_or_nv42_to_rgb_or_rgba_row_impl::<true, true>(y, vu, rgba_out, width, matrix, full_range);
  }
}

/// Shared wasm simd128 NV24/NV42 kernel (4:4:4 semi-planar). Unlike
/// the 4:2:0 variant, chroma is 1:1 with Y — load 32 UV bytes per 16
/// Y pixels, compute 16 chroma values per channel directly, skip the
/// `dup_lo/hi` fan-out.
///
/// # Safety
///
/// 1. **simd128 must be available** (compile-time `target_feature`).
/// 2. `y.len() >= width`, `uv_or_vu.len() >= 2 * width`,
///    `out.len() >= width * if ALPHA { 4 } else { 3 }`.
#[inline]
#[target_feature(enable = "simd128")]
unsafe fn nv24_or_nv42_to_rgb_or_rgba_row_impl<
  const SWAP_UV: bool,
  const ALPHA: bool,
>(
  y: &[u8],
  uv_or_vu: &[u8],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(y.len() >= width);
  debug_assert!(uv_or_vu.len() >= 2 * width);
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params(full_range);
  const RND: i32 = 1 << 14;

  // SAFETY: simd128 availability is the caller's compile-time
  // obligation; pointer adds are bounded by the loop condition.
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
    let alpha_u8 = u8x16_splat(0xFF);

    let mut x = 0usize;
    while x + 16 <= width {
      let y_vec = v128_load(y.as_ptr().add(x).cast());
      // 16 Y pixels → 32 UV bytes (two loads).
      let uv_lo_vec = v128_load(uv_or_vu.as_ptr().add(x * 2).cast());
      let uv_hi_vec = v128_load(uv_or_vu.as_ptr().add(x * 2 + 16).cast());

      // Deinterleave each 16-byte vec into 8 even + 8 odd bytes.
      let even_lo = i8x16_shuffle::<
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
        14,
      >(uv_lo_vec, uv_lo_vec);
      let odd_lo = i8x16_shuffle::<
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
        15,
      >(uv_lo_vec, uv_lo_vec);
      let even_hi = i8x16_shuffle::<
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
        14,
      >(uv_hi_vec, uv_hi_vec);
      let odd_hi = i8x16_shuffle::<
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
        15,
      >(uv_hi_vec, uv_hi_vec);
      let (u_lo_bytes, v_lo_bytes, u_hi_bytes, v_hi_bytes) = if SWAP_UV {
        (odd_lo, even_lo, odd_hi, even_hi)
      } else {
        (even_lo, odd_lo, even_hi, odd_hi)
      };

      // Widen U/V halves to i16x8.
      let u_lo_i16 = i16x8_sub(u16x8_extend_low_u8x16(u_lo_bytes), mid128);
      let u_hi_i16 = i16x8_sub(u16x8_extend_low_u8x16(u_hi_bytes), mid128);
      let v_lo_i16 = i16x8_sub(u16x8_extend_low_u8x16(v_lo_bytes), mid128);
      let v_hi_i16 = i16x8_sub(u16x8_extend_low_u8x16(v_hi_bytes), mid128);

      // Split each i16x8 into two i32x4 halves.
      let u_lo_a = i32x4_extend_low_i16x8(u_lo_i16);
      let u_lo_b = i32x4_extend_high_i16x8(u_lo_i16);
      let u_hi_a = i32x4_extend_low_i16x8(u_hi_i16);
      let u_hi_b = i32x4_extend_high_i16x8(u_hi_i16);
      let v_lo_a = i32x4_extend_low_i16x8(v_lo_i16);
      let v_lo_b = i32x4_extend_high_i16x8(v_lo_i16);
      let v_hi_a = i32x4_extend_low_i16x8(v_hi_i16);
      let v_hi_b = i32x4_extend_high_i16x8(v_hi_i16);

      // u_d / v_d = (u * c_scale + RND) >> 15.
      let u_d_lo_a = q15_shift(i32x4_add(i32x4_mul(u_lo_a, c_scale_v), rnd_v));
      let u_d_lo_b = q15_shift(i32x4_add(i32x4_mul(u_lo_b, c_scale_v), rnd_v));
      let u_d_hi_a = q15_shift(i32x4_add(i32x4_mul(u_hi_a, c_scale_v), rnd_v));
      let u_d_hi_b = q15_shift(i32x4_add(i32x4_mul(u_hi_b, c_scale_v), rnd_v));
      let v_d_lo_a = q15_shift(i32x4_add(i32x4_mul(v_lo_a, c_scale_v), rnd_v));
      let v_d_lo_b = q15_shift(i32x4_add(i32x4_mul(v_lo_b, c_scale_v), rnd_v));
      let v_d_hi_a = q15_shift(i32x4_add(i32x4_mul(v_hi_a, c_scale_v), rnd_v));
      let v_d_hi_b = q15_shift(i32x4_add(i32x4_mul(v_hi_b, c_scale_v), rnd_v));

      // 16 chroma per channel (no duplication).
      let r_chroma_lo = chroma_i16x8(cru, crv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let r_chroma_hi = chroma_i16x8(cru, crv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let g_chroma_lo = chroma_i16x8(cgu, cgv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let g_chroma_hi = chroma_i16x8(cgu, cgv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let b_chroma_lo = chroma_i16x8(cbu, cbv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let b_chroma_hi = chroma_i16x8(cbu, cbv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);

      let y_low_i16 = u8_low_to_i16x8(y_vec);
      let y_high_i16 = u8_high_to_i16x8(y_vec);
      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v);

      let b_lo = i16x8_add_sat(y_scaled_lo, b_chroma_lo);
      let b_hi = i16x8_add_sat(y_scaled_hi, b_chroma_hi);
      let g_lo = i16x8_add_sat(y_scaled_lo, g_chroma_lo);
      let g_hi = i16x8_add_sat(y_scaled_hi, g_chroma_hi);
      let r_lo = i16x8_add_sat(y_scaled_lo, r_chroma_lo);
      let r_hi = i16x8_add_sat(y_scaled_hi, r_chroma_hi);

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
      let tail_uv = &uv_or_vu[x * 2..width * 2];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      match (SWAP_UV, ALPHA) {
        (false, false) => {
          scalar::nv24_to_rgb_row(tail_y, tail_uv, tail_out, tail_w, matrix, full_range)
        }
        (true, false) => {
          scalar::nv42_to_rgb_row(tail_y, tail_uv, tail_out, tail_w, matrix, full_range)
        }
        (false, true) => {
          scalar::nv24_to_rgba_row(tail_y, tail_uv, tail_out, tail_w, matrix, full_range)
        }
        (true, true) => {
          scalar::nv42_to_rgba_row(tail_y, tail_uv, tail_out, tail_w, matrix, full_range)
        }
      }
    }
  }
}

/// wasm simd128 YUV 4:4:4 planar → packed RGB. Thin wrapper over
/// [`yuv_444_to_rgb_or_rgba_row`] with `ALPHA = false`.
///
/// # Safety
///
/// 1. **simd128 must be available** (compile-time `target_feature`).
/// 2. `y.len() >= width`, `u.len() >= width`, `v.len() >= width`,
///    `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn yuv_444_to_rgb_row(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller-checked simd128 availability + slice bounds — see
  // [`yuv_444_to_rgb_or_rgba_row`] safety contract.
  unsafe {
    yuv_444_to_rgb_or_rgba_row::<false>(y, u, v, rgb_out, width, matrix, full_range);
  }
}

/// wasm simd128 YUV 4:4:4 planar → packed **RGBA** (8-bit). Same
/// contract as [`yuv_444_to_rgb_row`] but writes 4 bytes per pixel
/// via [`write_rgba_16`] (R, G, B, `0xFF`).
///
/// # Safety
///
/// Same as [`yuv_444_to_rgb_row`] except the output slice must be
/// `>= 4 * width` bytes.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn yuv_444_to_rgba_row(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller-checked simd128 availability + slice bounds — see
  // [`yuv_444_to_rgb_or_rgba_row`] safety contract.
  unsafe {
    yuv_444_to_rgb_or_rgba_row::<true>(y, u, v, rgba_out, width, matrix, full_range);
  }
}

/// Shared wasm simd128 YUV 4:4:4 kernel for [`yuv_444_to_rgb_row`]
/// (`ALPHA = false`, [`write_rgb_16`]) and [`yuv_444_to_rgba_row`]
/// (`ALPHA = true`, [`write_rgba_16`] with constant `0xFF` alpha).
///
/// # Safety
///
/// 1. **simd128 must be enabled at compile time.**
/// 2. `y.len() >= width`, `u.len() >= width`, `v.len() >= width`.
/// 3. `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
#[inline]
#[target_feature(enable = "simd128")]
unsafe fn yuv_444_to_rgb_or_rgba_row<const ALPHA: bool>(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(y.len() >= width);
  debug_assert!(u.len() >= width);
  debug_assert!(v.len() >= width);
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params(full_range);
  const RND: i32 = 1 << 14;

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
    let alpha_u8 = u8x16_splat(0xFF);

    let mut x = 0usize;
    while x + 16 <= width {
      let y_vec = v128_load(y.as_ptr().add(x).cast());
      // 4:4:4: 16 U + 16 V directly (no deinterleave).
      let u_vec = v128_load(u.as_ptr().add(x).cast());
      let v_vec = v128_load(v.as_ptr().add(x).cast());

      // Widen low / high halves of U / V to i16x8 and subtract 128.
      let u_lo_i16 = i16x8_sub(u16x8_extend_low_u8x16(u_vec), mid128);
      let u_hi_i16 = i16x8_sub(u16x8_extend_high_u8x16(u_vec), mid128);
      let v_lo_i16 = i16x8_sub(u16x8_extend_low_u8x16(v_vec), mid128);
      let v_hi_i16 = i16x8_sub(u16x8_extend_high_u8x16(v_vec), mid128);

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

      let y_low_i16 = u8_low_to_i16x8(y_vec);
      let y_high_i16 = u8_high_to_i16x8(y_vec);
      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v);

      let b_lo = i16x8_add_sat(y_scaled_lo, b_chroma_lo);
      let b_hi = i16x8_add_sat(y_scaled_hi, b_chroma_hi);
      let g_lo = i16x8_add_sat(y_scaled_lo, g_chroma_lo);
      let g_hi = i16x8_add_sat(y_scaled_hi, g_chroma_hi);
      let r_lo = i16x8_add_sat(y_scaled_lo, r_chroma_lo);
      let r_hi = i16x8_add_sat(y_scaled_hi, r_chroma_hi);

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
      let tail_u = &u[x..width];
      let tail_v = &v[x..width];
      let tail_w = width - x;
      let tail_out = &mut out[x * bpp..width * bpp];
      if ALPHA {
        scalar::yuv_444_to_rgba_row(tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range);
      } else {
        scalar::yuv_444_to_rgb_row(tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range);
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

/// Writes 16 pixels of packed RGBA (64 bytes) from four u8x16 channel
/// vectors. Mirror of [`write_rgb_16`] for the 4-channel output path.
///
/// The 4-byte stride aligns cleanly with the 16-byte register width:
/// each output block holds exactly 4 RGBA quads (16 bytes), with R,
/// G, B, A interleaved at positions `(0, 1, 2, 3)`, `(4, 5, 6, 7)`,
/// etc. `u8x16_swizzle` indices ≥ 16 zero the lane.
///
/// # Safety
///
/// `ptr` must point to at least 64 writable bytes.
#[inline(always)]
unsafe fn write_rgba_16(r: v128, g: v128, b: v128, a: v128, ptr: *mut u8) {
  unsafe {
    // Block 0 (bytes 0..16): pixels 0..3, source bytes 0..3.
    let r0 = i8x16(0, -1, -1, -1, 1, -1, -1, -1, 2, -1, -1, -1, 3, -1, -1, -1);
    let g0 = i8x16(-1, 0, -1, -1, -1, 1, -1, -1, -1, 2, -1, -1, -1, 3, -1, -1);
    let b0 = i8x16(-1, -1, 0, -1, -1, -1, 1, -1, -1, -1, 2, -1, -1, -1, 3, -1);
    let a0 = i8x16(-1, -1, -1, 0, -1, -1, -1, 1, -1, -1, -1, 2, -1, -1, -1, 3);
    let out0 = v128_or(
      v128_or(u8x16_swizzle(r, r0), u8x16_swizzle(g, g0)),
      v128_or(u8x16_swizzle(b, b0), u8x16_swizzle(a, a0)),
    );

    // Block 1 (bytes 16..32): pixels 4..7, source bytes 4..7.
    let r1 = i8x16(4, -1, -1, -1, 5, -1, -1, -1, 6, -1, -1, -1, 7, -1, -1, -1);
    let g1 = i8x16(-1, 4, -1, -1, -1, 5, -1, -1, -1, 6, -1, -1, -1, 7, -1, -1);
    let b1 = i8x16(-1, -1, 4, -1, -1, -1, 5, -1, -1, -1, 6, -1, -1, -1, 7, -1);
    let a1 = i8x16(-1, -1, -1, 4, -1, -1, -1, 5, -1, -1, -1, 6, -1, -1, -1, 7);
    let out1 = v128_or(
      v128_or(u8x16_swizzle(r, r1), u8x16_swizzle(g, g1)),
      v128_or(u8x16_swizzle(b, b1), u8x16_swizzle(a, a1)),
    );

    // Block 2 (bytes 32..48): pixels 8..11, source bytes 8..11.
    let r2 = i8x16(8, -1, -1, -1, 9, -1, -1, -1, 10, -1, -1, -1, 11, -1, -1, -1);
    let g2 = i8x16(-1, 8, -1, -1, -1, 9, -1, -1, -1, 10, -1, -1, -1, 11, -1, -1);
    let b2 = i8x16(-1, -1, 8, -1, -1, -1, 9, -1, -1, -1, 10, -1, -1, -1, 11, -1);
    let a2 = i8x16(-1, -1, -1, 8, -1, -1, -1, 9, -1, -1, -1, 10, -1, -1, -1, 11);
    let out2 = v128_or(
      v128_or(u8x16_swizzle(r, r2), u8x16_swizzle(g, g2)),
      v128_or(u8x16_swizzle(b, b2), u8x16_swizzle(a, a2)),
    );

    // Block 3 (bytes 48..64): pixels 12..15, source bytes 12..15.
    let r3 = i8x16(
      12, -1, -1, -1, 13, -1, -1, -1, 14, -1, -1, -1, 15, -1, -1, -1,
    );
    let g3 = i8x16(
      -1, 12, -1, -1, -1, 13, -1, -1, -1, 14, -1, -1, -1, 15, -1, -1,
    );
    let b3 = i8x16(
      -1, -1, 12, -1, -1, -1, 13, -1, -1, -1, 14, -1, -1, -1, 15, -1,
    );
    let a3 = i8x16(
      -1, -1, -1, 12, -1, -1, -1, 13, -1, -1, -1, 14, -1, -1, -1, 15,
    );
    let out3 = v128_or(
      v128_or(u8x16_swizzle(r, r3), u8x16_swizzle(g, g3)),
      v128_or(u8x16_swizzle(b, b3), u8x16_swizzle(a, a3)),
    );

    v128_store(ptr.cast(), out0);
    v128_store(ptr.add(16).cast(), out1);
    v128_store(ptr.add(32).cast(), out2);
    v128_store(ptr.add(48).cast(), out3);
  }
}

// ===== 16-bit YUV → RGB ==================================================

/// `(Y_u16x8 - y_off) * y_scale + RND >> 15` for full u16 Y samples.
/// Unsigned widening via `u32x4_extend_{low,high}_u16x8`. Returns i16x8.
#[inline(always)]
fn scale_y_u16_wasm(y_u16: v128, y_off32_v: v128, y_scale_v: v128, rnd_v: v128) -> v128 {
  // y_off32_v = i32x4_splat(y_off)
  let lo_u32 = u32x4_extend_low_u16x8(y_u16);
  let hi_u32 = u32x4_extend_high_u16x8(y_u16);
  let lo_i32 = i32x4_sub(lo_u32, y_off32_v);
  let hi_i32 = i32x4_sub(hi_u32, y_off32_v);
  let lo = q15_shift(i32x4_add(i32x4_mul(lo_i32, y_scale_v), rnd_v));
  let hi = q15_shift(i32x4_add(i32x4_mul(hi_i32, y_scale_v), rnd_v));
  i16x8_narrow_i32x4(lo, hi)
}

/// Computes 2 × i64 chroma products with the Q15 shift using native
/// wasm simd128 `i64x2_mul` + `i64x2_shr` (signed arithmetic right
/// shift). All inputs are i64x2; `cu_i64` / `cv_i64` are the
/// coefficient broadcast widened once per row via
/// `i64x2_extend_low_i32x4`.
#[inline(always)]
fn chroma_i64x2_wasm(
  cu_i64: v128,
  cv_i64: v128,
  u_d_i64: v128,
  v_d_i64: v128,
  rnd_i64: v128,
) -> v128 {
  let sum = i64x2_add(
    i64x2_add(i64x2_mul(cu_i64, u_d_i64), i64x2_mul(cv_i64, v_d_i64)),
    rnd_i64,
  );
  i64x2_shr(sum, 15)
}

/// Combines two i64x2 vectors into an i32x4 of their low 32 bits.
/// Valid when each i64 fits in i32 (true for our Q15-shifted chroma
/// and Y-scale results).
///
/// Uses `i8x16_shuffle` since wasm simd128 does not provide a direct
/// i64x2 → i32x2 narrow primitive.
#[inline(always)]
fn combine_i64x2_pair_to_i32x4(lo: v128, hi: v128) -> v128 {
  // Byte indices: low 4 bytes of each i64 lane.
  i8x16_shuffle::<0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27>(lo, hi)
}

/// Duplicates each i32 lane of `chroma` into a pair for the 4:2:0 u16
/// output pipeline: `[c0, c1, c2, c3]` →
/// Return.0 = `[c0, c0, c1, c1]`, Return.1 = `[c2, c2, c3, c3]`.
#[inline(always)]
fn chroma_dup_i32x4_u16(chroma: v128) -> (v128, v128) {
  let lo = i8x16_shuffle::<0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 7>(chroma, chroma);
  let hi =
    i8x16_shuffle::<8, 9, 10, 11, 8, 9, 10, 11, 12, 13, 14, 15, 12, 13, 14, 15>(chroma, chroma);
  (lo, hi)
}

/// `(y_minus_off * y_scale + RND) >> 15` computed in i64 for all 4
/// lanes of an i32x4 Y stream, returning i32x4.
#[inline(always)]
fn scale_y_i32x4_i64_wasm(y_minus_off: v128, y_scale_i64: v128, rnd_i64: v128) -> v128 {
  let lo = i64x2_shr(
    i64x2_add(
      i64x2_mul(y_scale_i64, i64x2_extend_low_i32x4(y_minus_off)),
      rnd_i64,
    ),
    15,
  );
  let hi = i64x2_shr(
    i64x2_add(
      i64x2_mul(y_scale_i64, i64x2_extend_high_i32x4(y_minus_off)),
      rnd_i64,
    ),
    15,
  );
  combine_i64x2_pair_to_i32x4(lo, hi)
}

/// WASM simd128 YUV 4:2:0 16-bit → packed **8-bit** RGB. 16 pixels per iteration.
/// UV centering via wrapping 0x8000 trick; unsigned Y widening.
///
/// # Safety
///
/// 1. **simd128 must be enabled at compile time.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `u_half.len() >= width / 2`,
///    `v_half.len() >= width / 2`, `rgb_out.len() >= 3 * width`.
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
  debug_assert_eq!(width & 1, 0);
  debug_assert!(y.len() >= width);
  debug_assert!(u_half.len() >= width / 2);
  debug_assert!(v_half.len() >= width / 2);
  debug_assert!(rgb_out.len() >= width * 3);

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

      write_rgb_16(r_u8, g_u8, b_u8, rgb_out.as_mut_ptr().add(x * 3));
      x += 16;
    }

    if x < width {
      scalar::yuv_420p16_to_rgb_row(
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
  debug_assert_eq!(width & 1, 0);
  debug_assert!(y.len() >= width);
  debug_assert!(u_half.len() >= width / 2);
  debug_assert!(v_half.len() >= width / 2);
  debug_assert!(rgb_out.len() >= width * 3);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<16, 16>(full_range);
  const RND_I64: i64 = 1 << 14;
  const RND_I32: i32 = 1 << 14;

  unsafe {
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

      write_rgb_u16_8(r_u16, g_u16, b_u16, rgb_out.as_mut_ptr().add(x * 3));
      x += 8;
    }

    if x < width {
      scalar::yuv_420p16_to_rgb_u16_row(
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

/// WASM simd128 P016 → packed **8-bit** RGB. 16 pixels per iteration.
///
/// # Safety
///
/// 1. **simd128 must be enabled at compile time.**
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `uv_half.len() >= width`, `rgb_out.len() >= 3 * width`.
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
  debug_assert_eq!(width & 1, 0);
  debug_assert!(y.len() >= width);
  debug_assert!(uv_half.len() >= width);
  debug_assert!(rgb_out.len() >= width * 3);

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

      write_rgb_16(r_u8, g_u8, b_u8, rgb_out.as_mut_ptr().add(x * 3));
      x += 16;
    }

    if x < width {
      scalar::p16_to_rgb_row(
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
  debug_assert_eq!(width & 1, 0);
  debug_assert!(y.len() >= width);
  debug_assert!(uv_half.len() >= width);
  debug_assert!(rgb_out.len() >= width * 3);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<16, 16>(full_range);
  const RND_I64: i64 = 1 << 14;
  const RND_I32: i32 = 1 << 14;

  unsafe {
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

      write_rgb_u16_8(r_u16, g_u16, b_u16, rgb_out.as_mut_ptr().add(x * 3));
      x += 8;
    }

    if x < width {
      scalar::p16_to_rgb_u16_row(
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

// ===== Pn 4:4:4 (semi-planar high-bit-packed) → RGB =======================
//
// Native wasm simd128 4:4:4 Pn kernels — combine `yuv_444p_n_to_rgb_row`'s
// 1:1 chroma compute with `p_n_to_rgb_row`'s `deinterleave_uv_u16_wasm`
// pattern. 16 Y pixels per iter for the i32 Q15 paths; 8 for the
// i64 chroma u16-output path (matches `yuv_444p16_to_rgb_u16_row`).

/// wasm simd128 Pn 4:4:4 high-bit-packed (BITS ∈ {10, 12}) → packed
/// **u8** RGB.
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
  const { assert!(BITS == 10 || BITS == 12) };
  debug_assert!(y.len() >= width);
  debug_assert!(uv_full.len() >= 2 * width);
  debug_assert!(rgb_out.len() >= width * 3);

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

    let shr = (16 - BITS) as u32;

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

      write_rgb_16(r_u8, g_u8, b_u8, rgb_out.as_mut_ptr().add(x * 3));

      x += 16;
    }

    if x < width {
      scalar::p_n_444_to_rgb_row::<BITS>(
        &y[x..width],
        &uv_full[x * 2..width * 2],
        &mut rgb_out[x * 3..width * 3],
        width - x,
        matrix,
        full_range,
      );
    }
  }
}

/// wasm simd128 Pn 4:4:4 high-bit-packed (BITS ∈ {10, 12}) → packed
/// **native-depth `u16`** RGB.
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
  const { assert!(BITS == 10 || BITS == 12) };
  debug_assert!(y.len() >= width);
  debug_assert!(uv_full.len() >= 2 * width);
  debug_assert!(rgb_out.len() >= width * 3);

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

    let shr = (16 - BITS) as u32;

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

      let dst = rgb_out.as_mut_ptr().add(x * 3);
      write_rgb_u16_8(r_lo, g_lo, b_lo, dst);
      write_rgb_u16_8(r_hi, g_hi, b_hi, dst.add(24));

      x += 16;
    }

    if x < width {
      scalar::p_n_444_to_rgb_u16_row::<BITS>(
        &y[x..width],
        &uv_full[x * 2..width * 2],
        &mut rgb_out[x * 3..width * 3],
        width - x,
        matrix,
        full_range,
      );
    }
  }
}

/// wasm simd128 P416 (semi-planar 4:4:4, 16-bit) → packed **u8** RGB.
/// 16 pixels per iter.
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
  debug_assert!(y.len() >= width);
  debug_assert!(uv_full.len() >= 2 * width);
  debug_assert!(rgb_out.len() >= width * 3);

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

      write_rgb_16(r_u8, g_u8, b_u8, rgb_out.as_mut_ptr().add(x * 3));
      x += 16;
    }

    if x < width {
      scalar::p_n_444_16_to_rgb_row(
        &y[x..width],
        &uv_full[x * 2..width * 2],
        &mut rgb_out[x * 3..width * 3],
        width - x,
        matrix,
        full_range,
      );
    }
  }
}

/// wasm simd128 P416 → packed **native-depth `u16`** RGB. i64 chroma
/// via native `i64x2_shr` (no bias trick needed). 8 pixels per iter.
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
  debug_assert!(y.len() >= width);
  debug_assert!(uv_full.len() >= 2 * width);
  debug_assert!(rgb_out.len() >= width * 3);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<16, 16>(full_range);
  const RND_I64: i64 = 1 << 14;
  const RND_I32: i32 = 1 << 14;

  unsafe {
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

      write_rgb_u16_8(r_u16, g_u16, b_u16, rgb_out.as_mut_ptr().add(x * 3));
      x += 8;
    }

    if x < width {
      scalar::p_n_444_16_to_rgb_u16_row(
        &y[x..width],
        &uv_full[x * 2..width * 2],
        &mut rgb_out[x * 3..width * 3],
        width - x,
        matrix,
        full_range,
      );
    }
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
mod tests;
