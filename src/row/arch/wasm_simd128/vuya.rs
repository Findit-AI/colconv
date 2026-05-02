#![allow(dead_code)] // dispatcher (Tasks 10-13) not yet wired
//! wasm-simd128 kernels for VUYA / VUYX packed YUV 4:4:4 8-bit family.
//!
//! ## Layout
//!
//! Four `u8` elements per pixel: `V(8) ‖ U(8) ‖ Y(8) ‖ A(8)`.
//! VUYA carries a real alpha channel in byte 3. VUYX treats byte 3 as
//! padding and forces output α to `0xFF`.
//!
//! ## Per-iter pipeline (16 px / iter)
//!
//! One iteration processes 16 pixels = 64 bytes = 4 × `v128_load`.
//! Each `v128` holds 4 pixels (16 bytes). `u8x16_swizzle` isolates one
//! channel's 4 bytes (from positions 0/4/8/12, 1/5/9/13, 2/6/10/14, or
//! 3/7/11/15) into the low 4 bytes per register. Four such results are
//! concatenated by `i8x16_shuffle` cascades (valid indices 0–31) to
//! form full 16-lane channel vectors.
//!
//! No shift is needed — samples are natively 8-bit.
//!
//! The Q15 chroma + Y pipeline mirrors other 8-bit wasm-simd128
//! backends (`packed_yuv_8bit.rs`, `semi_planar_8bit.rs`):
//! `chroma_i16x8`, `scale_y`, `u8x16_narrow_i16x8`.
//!
//! ## Tail
//!
//! `width % 16` remaining pixels fall through to
//! `scalar::vuya_to_rgb_or_rgba_row::<ALPHA, ALPHA_SRC>`.

use core::arch::wasm32::*;

use super::*;
use crate::{ColorMatrix, row::scalar};

// ---- deinterleave helper -----------------------------------------------

/// Deinterleave 16 VUYA pixels (4 × v128 = 64 bytes) into separate u8x16
/// channel vectors (v_out, u_out, y_out, a_out).
///
/// Layout in memory (4 bytes per pixel):
/// ```text
///   byte 0 = V,  byte 1 = U,  byte 2 = Y,  byte 3 = A
/// ```
///
/// Each `v128` load covers 4 pixels. `u8x16_swizzle` with a 4-byte
/// extract mask (indices ≥ 16 zeroed) pulls the 4 bytes for one channel
/// from each register into the low 4 bytes. Three-level `i8x16_shuffle`
/// cascades assemble 4 × 4-byte fragments into a single 16-byte
/// channel vector.
///
/// # Safety
///
/// `ptr` must point to at least 64 readable bytes (16 × 4 = 64).
/// `simd128` must be enabled at compile time.
#[inline]
#[target_feature(enable = "simd128")]
unsafe fn deinterleave_vuya_16px(ptr: *const u8) -> (v128, v128, v128, v128) {
  unsafe {
    // Load 4 × v128, each covering 4 pixels (16 bytes).
    let raw0 = v128_load(ptr.cast()); // [V0,U0,Y0,A0, V1,U1,Y1,A1, V2,U2,Y2,A2, V3,U3,Y3,A3]
    let raw1 = v128_load(ptr.add(16).cast()); // pixels 4-7
    let raw2 = v128_load(ptr.add(32).cast()); // pixels 8-11
    let raw3 = v128_load(ptr.add(48).cast()); // pixels 12-15

    // Channel byte positions within a 4-pixel (16-byte) v128:
    //   V → bytes 0, 4, 8,  12  (stride 4, offset 0)
    //   U → bytes 1, 5, 9,  13  (stride 4, offset 1)
    //   Y → bytes 2, 6, 10, 14  (stride 4, offset 2)
    //   A → bytes 3, 7, 11, 15  (stride 4, offset 3)
    //
    // `u8x16_swizzle` with index ≥ 16 zeroes the output byte.
    // We pack each channel's 4 bytes into the LOW 4 bytes of the result
    // so downstream `i8x16_shuffle` can pick bytes 0..3 from each
    // register (indices 0-3 and 16-19 for the two operands).

    // V: bytes [0, 4, 8, 12] → low 4 bytes; high 12 bytes zeroed.
    let v_idx = i8x16(0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    // U: bytes [1, 5, 9, 13] → low 4 bytes.
    let u_idx = i8x16(1, 5, 9, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    // Y: bytes [2, 6, 10, 14] → low 4 bytes.
    let y_idx = i8x16(2, 6, 10, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    // A: bytes [3, 7, 11, 15] → low 4 bytes.
    let a_idx = i8x16(3, 7, 11, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);

    // Extract per-channel 4-byte fragments from each 4-pixel register.
    // Result: low 4 bytes = channel samples for pixels {4n, 4n+1, 4n+2, 4n+3}.
    let v0 = u8x16_swizzle(raw0, v_idx); // [V0, V1, V2, V3,  0, 0, 0, 0, ...]
    let v1 = u8x16_swizzle(raw1, v_idx); // [V4, V5, V6, V7,  ...]
    let v2 = u8x16_swizzle(raw2, v_idx); // [V8..V11, ...]
    let v3 = u8x16_swizzle(raw3, v_idx); // [V12..V15, ...]

    let u0 = u8x16_swizzle(raw0, u_idx);
    let u1 = u8x16_swizzle(raw1, u_idx);
    let u2 = u8x16_swizzle(raw2, u_idx);
    let u3 = u8x16_swizzle(raw3, u_idx);

    let y0 = u8x16_swizzle(raw0, y_idx);
    let y1 = u8x16_swizzle(raw1, y_idx);
    let y2 = u8x16_swizzle(raw2, y_idx);
    let y3 = u8x16_swizzle(raw3, y_idx);

    let a0 = u8x16_swizzle(raw0, a_idx);
    let a1 = u8x16_swizzle(raw1, a_idx);
    let a2 = u8x16_swizzle(raw2, a_idx);
    let a3 = u8x16_swizzle(raw3, a_idx);

    // Concatenate pairs of 4-byte fragments into 8-byte fragments.
    // `i8x16_shuffle::<0,1,2,3, 16,17,18,19, ...>` picks bytes 0-3 from
    // operand 0 (low 4 bytes of v0) and bytes 0-3 from operand 1
    // (indices 16-19 = bytes 0-3 of v1). High 8 bytes zero-filled.

    let v01 = i8x16_shuffle::<0, 1, 2, 3, 16, 17, 18, 19, 0, 0, 0, 0, 0, 0, 0, 0>(v0, v1);
    let v23 = i8x16_shuffle::<0, 1, 2, 3, 16, 17, 18, 19, 0, 0, 0, 0, 0, 0, 0, 0>(v2, v3);
    // Full V vector: u8x16 [V0..V15].
    // Low 8 bytes from v01, high 8 bytes from v23.
    let v_out = i8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23>(v01, v23);

    let u01 = i8x16_shuffle::<0, 1, 2, 3, 16, 17, 18, 19, 0, 0, 0, 0, 0, 0, 0, 0>(u0, u1);
    let u23 = i8x16_shuffle::<0, 1, 2, 3, 16, 17, 18, 19, 0, 0, 0, 0, 0, 0, 0, 0>(u2, u3);
    let u_out = i8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23>(u01, u23);

    let y01 = i8x16_shuffle::<0, 1, 2, 3, 16, 17, 18, 19, 0, 0, 0, 0, 0, 0, 0, 0>(y0, y1);
    let y23 = i8x16_shuffle::<0, 1, 2, 3, 16, 17, 18, 19, 0, 0, 0, 0, 0, 0, 0, 0>(y2, y3);
    let y_out = i8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23>(y01, y23);

    let a01 = i8x16_shuffle::<0, 1, 2, 3, 16, 17, 18, 19, 0, 0, 0, 0, 0, 0, 0, 0>(a0, a1);
    let a23 = i8x16_shuffle::<0, 1, 2, 3, 16, 17, 18, 19, 0, 0, 0, 0, 0, 0, 0, 0>(a2, a3);
    let a_out = i8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23>(a01, a23);

    (v_out, u_out, y_out, a_out)
  }
}

// ---- shared kernel template -----------------------------------------------

/// wasm-simd128 VUYA/VUYX → packed u8 RGB or RGBA. 16 pixels per iteration.
///
/// Byte-identical to `scalar::vuya_to_rgb_or_rgba_row::<ALPHA, ALPHA_SRC>`.
///
/// The three valid monomorphizations are:
/// - `<false, false>` — RGB (drops α)
/// - `<true, true>`  — RGBA, source α pass-through (VUYA)
/// - `<true, false>` — RGBA, force α = `0xFF` (VUYX)
///
/// `<false, true>` is rejected at monomorphization via `const { assert! }`.
///
/// # Safety
///
/// 1. **`simd128` must be enabled at compile time.**
/// 2. `packed.len() >= width * 4`.
/// 3. `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn vuya_to_rgb_or_rgba_row<const ALPHA: bool, const ALPHA_SRC: bool>(
  packed: &[u8],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // Source alpha requires RGBA output.
  const { assert!(!ALPHA_SRC || ALPHA) };
  debug_assert!(packed.len() >= width * 4, "packed row too short");
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp, "out row too short");

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<8, 8>(full_range);
  let bias = scalar::chroma_bias::<8>();
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

    let mut x = 0usize;
    while x + 16 <= width {
      // Deinterleave 16 VUYA pixels (64 bytes) into V/U/Y/A channel vectors.
      let (v_raw, u_raw, y_raw, a_raw) =
        deinterleave_vuya_16px(packed.as_ptr().add(x * 4));

      // Zero-extend U/V/Y bytes to i16x8 (low half and high half).
      // u8_low_to_i16x8 / u8_high_to_i16x8 are defined in mod.rs.
      let u_lo = u8_low_to_i16x8(u_raw);  // lanes 0-7 of U, as i16
      let u_hi = u8_high_to_i16x8(u_raw); // lanes 8-15 of U, as i16
      let v_lo = u8_low_to_i16x8(v_raw);
      let v_hi = u8_high_to_i16x8(v_raw);
      let y_lo = u8_low_to_i16x8(y_raw);
      let y_hi = u8_high_to_i16x8(y_raw);

      // Subtract chroma bias (128 for 8-bit).
      let u_sub_lo = i16x8_sub(u_lo, bias_v);
      let u_sub_hi = i16x8_sub(u_hi, bias_v);
      let v_sub_lo = i16x8_sub(v_lo, bias_v);
      let v_sub_hi = i16x8_sub(v_hi, bias_v);

      // Widen to i32x4 lo/hi for Q15 chroma-scale multiply.
      let u_lo_lo = i32x4_extend_low_i16x8(u_sub_lo);
      let u_lo_hi = i32x4_extend_high_i16x8(u_sub_lo);
      let u_hi_lo = i32x4_extend_low_i16x8(u_sub_hi);
      let u_hi_hi = i32x4_extend_high_i16x8(u_sub_hi);
      let v_lo_lo = i32x4_extend_low_i16x8(v_sub_lo);
      let v_lo_hi = i32x4_extend_high_i16x8(v_sub_lo);
      let v_hi_lo = i32x4_extend_low_i16x8(v_sub_hi);
      let v_hi_hi = i32x4_extend_high_i16x8(v_sub_hi);

      // Q15 chroma scale.
      let u_d_lo_lo = q15_shift(i32x4_add(i32x4_mul(u_lo_lo, c_scale_v), rnd_v));
      let u_d_lo_hi = q15_shift(i32x4_add(i32x4_mul(u_lo_hi, c_scale_v), rnd_v));
      let u_d_hi_lo = q15_shift(i32x4_add(i32x4_mul(u_hi_lo, c_scale_v), rnd_v));
      let u_d_hi_hi = q15_shift(i32x4_add(i32x4_mul(u_hi_hi, c_scale_v), rnd_v));
      let v_d_lo_lo = q15_shift(i32x4_add(i32x4_mul(v_lo_lo, c_scale_v), rnd_v));
      let v_d_lo_hi = q15_shift(i32x4_add(i32x4_mul(v_lo_hi, c_scale_v), rnd_v));
      let v_d_hi_lo = q15_shift(i32x4_add(i32x4_mul(v_hi_lo, c_scale_v), rnd_v));
      let v_d_hi_hi = q15_shift(i32x4_add(i32x4_mul(v_hi_hi, c_scale_v), rnd_v));

      // 4:4:4 — no chroma duplication. Each of 16 pixels has unique U/V.
      // Compute chroma contribution for low 8 lanes (pixels 0-7).
      let r_chroma_lo = chroma_i16x8(cru, crv, u_d_lo_lo, v_d_lo_lo, u_d_lo_hi, v_d_lo_hi, rnd_v);
      let g_chroma_lo = chroma_i16x8(cgu, cgv, u_d_lo_lo, v_d_lo_lo, u_d_lo_hi, v_d_lo_hi, rnd_v);
      let b_chroma_lo = chroma_i16x8(cbu, cbv, u_d_lo_lo, v_d_lo_lo, u_d_lo_hi, v_d_lo_hi, rnd_v);

      // Chroma for high 8 lanes (pixels 8-15).
      let r_chroma_hi = chroma_i16x8(cru, crv, u_d_hi_lo, v_d_hi_lo, u_d_hi_hi, v_d_hi_hi, rnd_v);
      let g_chroma_hi = chroma_i16x8(cgu, cgv, u_d_hi_lo, v_d_hi_lo, u_d_hi_hi, v_d_hi_hi, rnd_v);
      let b_chroma_hi = chroma_i16x8(cbu, cbv, u_d_hi_lo, v_d_hi_lo, u_d_hi_hi, v_d_hi_hi, rnd_v);

      // Y: scale both halves. Y ∈ [0, 255] fits in non-negative i16.
      let y_scaled_lo = scale_y(y_lo, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_hi, y_off_v, y_scale_v, rnd_v);

      // Saturate-add Y + chroma per channel, narrow both halves to u8.
      let r_lo = i16x8_add_sat(y_scaled_lo, r_chroma_lo);
      let r_hi = i16x8_add_sat(y_scaled_hi, r_chroma_hi);
      let g_lo = i16x8_add_sat(y_scaled_lo, g_chroma_lo);
      let g_hi = i16x8_add_sat(y_scaled_hi, g_chroma_hi);
      let b_lo = i16x8_add_sat(y_scaled_lo, b_chroma_lo);
      let b_hi = i16x8_add_sat(y_scaled_hi, b_chroma_hi);

      let r_u8 = u8x16_narrow_i16x8(r_lo, r_hi);
      let g_u8 = u8x16_narrow_i16x8(g_lo, g_hi);
      let b_u8 = u8x16_narrow_i16x8(b_lo, b_hi);

      // Alpha: ALPHA_SRC=true uses the A vector from deinterleave;
      //        ALPHA_SRC=false uses 0xFF (opaque).
      let a_u8: v128 = if ALPHA_SRC { a_raw } else { alpha_u8 };

      // Write 16 pixels.
      if ALPHA {
        write_rgba_16(r_u8, g_u8, b_u8, a_u8, out.as_mut_ptr().add(x * 4));
      } else {
        write_rgb_16(r_u8, g_u8, b_u8, out.as_mut_ptr().add(x * 3));
      }

      x += 16;
    }

    // Scalar tail — remaining < 16 pixels.
    if x < width {
      let tail_packed = &packed[x * 4..];
      let tail_out = &mut out[x * bpp..];
      scalar::vuya_to_rgb_or_rgba_row::<ALPHA, ALPHA_SRC>(
        tail_packed,
        tail_out,
        width - x,
        matrix,
        full_range,
      );
    }
  }
}

// ---- thin wrappers ---------------------------------------------------------

/// wasm-simd128 VUYA / VUYX → packed **RGB** (3 bpp).
/// Alpha byte in source is discarded — RGB output has no alpha channel.
///
/// # Safety
///
/// 1. **`simd128` must be enabled at compile time.**
/// 2. `packed.len() >= width * 4`.
/// 3. `out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn vuya_to_rgb_row(
  packed: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  unsafe { vuya_to_rgb_or_rgba_row::<false, false>(packed, rgb_out, width, matrix, full_range); }
}

/// wasm-simd128 VUYA → packed **RGBA** (4 bpp).
/// Source A byte at offset 3 of each pixel quadruple is passed through verbatim.
///
/// # Safety
///
/// 1. **`simd128` must be enabled at compile time.**
/// 2. `packed.len() >= width * 4`.
/// 3. `out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn vuya_to_rgba_row(
  packed: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  unsafe { vuya_to_rgb_or_rgba_row::<true, true>(packed, rgba_out, width, matrix, full_range); }
}

/// wasm-simd128 VUYX → packed **RGBA** (4 bpp).
/// Source A byte is padding and is ignored; output α is forced to `0xFF` (opaque).
///
/// # Safety
///
/// 1. **`simd128` must be enabled at compile time.**
/// 2. `packed.len() >= width * 4`.
/// 3. `out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn vuyx_to_rgba_row(
  packed: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  unsafe { vuya_to_rgb_or_rgba_row::<true, false>(packed, rgba_out, width, matrix, full_range); }
}

// ---- luma extraction -------------------------------------------------------

/// wasm-simd128 VUYA / VUYX → u8 luma. Y is the third byte (offset 2) of
/// each pixel quadruple. 16 pixels per iteration.
///
/// Byte-identical to `scalar::vuya_to_luma_row`.
///
/// # Safety
///
/// 1. **`simd128` must be enabled at compile time.**
/// 2. `packed.len() >= width * 4`.
/// 3. `luma_out.len() >= width`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn vuya_to_luma_row(packed: &[u8], luma_out: &mut [u8], width: usize) {
  debug_assert!(packed.len() >= width * 4, "packed row too short");
  debug_assert!(luma_out.len() >= width, "luma row too short");

  unsafe {
    // Y is at byte offset 2 within each 4-byte pixel quadruple.
    // Within a 16-byte v128 (4 pixels): positions 2, 6, 10, 14.
    let y_idx = i8x16(2, 6, 10, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);

    let mut x = 0usize;
    while x + 16 <= width {
      let ptr = packed.as_ptr().add(x * 4);
      let raw0 = v128_load(ptr.cast()); // pixels 0-3
      let raw1 = v128_load(ptr.add(16).cast()); // pixels 4-7
      let raw2 = v128_load(ptr.add(32).cast()); // pixels 8-11
      let raw3 = v128_load(ptr.add(48).cast()); // pixels 12-15

      // Extract Y from each 4-pixel register → 4 bytes in low 4 bytes.
      let y0 = u8x16_swizzle(raw0, y_idx); // [Y0, Y1, Y2, Y3, 0, ...]
      let y1 = u8x16_swizzle(raw1, y_idx); // [Y4, Y5, Y6, Y7, 0, ...]
      let y2 = u8x16_swizzle(raw2, y_idx); // [Y8..Y11, 0, ...]
      let y3 = u8x16_swizzle(raw3, y_idx); // [Y12..Y15, 0, ...]

      // Concatenate pairs into 8-byte fragments.
      let y01 = i8x16_shuffle::<0, 1, 2, 3, 16, 17, 18, 19, 0, 0, 0, 0, 0, 0, 0, 0>(y0, y1);
      let y23 = i8x16_shuffle::<0, 1, 2, 3, 16, 17, 18, 19, 0, 0, 0, 0, 0, 0, 0, 0>(y2, y3);
      // Full 16-lane Y vector.
      let y_vec = i8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23>(y01, y23);

      v128_store(luma_out.as_mut_ptr().add(x).cast(), y_vec);
      x += 16;
    }

    // Scalar tail.
    if x < width {
      scalar::vuya_to_luma_row(&packed[x * 4..], &mut luma_out[x..], width - x);
    }
  }
}
