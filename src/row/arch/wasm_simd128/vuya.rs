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
use crate::{KernelMatrix, row::scalar};

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
/// Build a 4-byte gather index vector for one channel at byte offset
/// `OFF` within each 4-byte pixel (`OFF, OFF+4, OFF+8, OFF+12` → low 4
/// bytes; the high 12 zeroed via index `-1`). The const offset keeps it a
/// compile-time choice.
#[inline]
#[target_feature(enable = "simd128")]
fn chan_idx_wasm<const OFF: usize>() -> v128 {
  let o = OFF as i8;
  i8x16(
    o,
    o + 4,
    o + 8,
    o + 12,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
  )
}

/// Offset-parameterized wasm-simd128 deinterleave of 16 four-byte pixels
/// into `(v_out, u_out, y_out, a_out)` (each `v128` of 16 natural-order
/// channel bytes). The four byte offsets select which source byte feeds
/// each channel, serving every channel re-ordering of the 4-byte family.
///
/// # Safety
///
/// `ptr` must point to at least 64 readable bytes (16 pixels). `simd128`
/// must be enabled at compile time.
#[inline]
#[target_feature(enable = "simd128")]
unsafe fn deinterleave_packed444_16px<
  const V_OFF: usize,
  const U_OFF: usize,
  const Y_OFF: usize,
  const A_OFF: usize,
>(
  ptr: *const u8,
) -> (v128, v128, v128, v128) {
  unsafe {
    // Load 4 × v128, each covering 4 pixels (16 bytes).
    let raw0 = v128_load(ptr.cast());
    let raw1 = v128_load(ptr.add(16).cast()); // pixels 4-7
    let raw2 = v128_load(ptr.add(32).cast()); // pixels 8-11
    let raw3 = v128_load(ptr.add(48).cast()); // pixels 12-15

    let v_idx = chan_idx_wasm::<V_OFF>();
    let u_idx = chan_idx_wasm::<U_OFF>();
    let y_idx = chan_idx_wasm::<Y_OFF>();
    let a_idx = chan_idx_wasm::<A_OFF>();

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

/// VUYA / VUYX byte order (`V=0, U=1, Y=2, A=3`) over the offset-generic
/// wasm deinterleave.
///
/// # Safety
///
/// Same contract as [`deinterleave_packed444_16px`].
#[inline]
#[target_feature(enable = "simd128")]
unsafe fn deinterleave_vuya_16px(ptr: *const u8) -> (v128, v128, v128, v128) {
  // SAFETY: caller obligation — `ptr` has 64 bytes readable; simd128 enabled.
  unsafe { deinterleave_packed444_16px::<0, 1, 2, 3>(ptr) }
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
pub(crate) unsafe fn packed444_to_rgb_or_rgba_row<
  const ALPHA: bool,
  const ALPHA_SRC: bool,
  const V_OFF: usize,
  const U_OFF: usize,
  const Y_OFF: usize,
  const A_OFF: usize,
>(
  packed: &[u8],
  out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
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
      // Deinterleave 16 pixels (64 bytes) into V/U/Y/A vectors per offsets.
      let (v_raw, u_raw, y_raw, a_raw) =
        deinterleave_packed444_16px::<V_OFF, U_OFF, Y_OFF, A_OFF>(packed.as_ptr().add(x * 4));

      // Zero-extend U/V/Y bytes to i16x8 (low half and high half).
      // u8_low_to_i16x8 / u8_high_to_i16x8 are defined in mod.rs.
      let u_lo = u8_low_to_i16x8(u_raw); // lanes 0-7 of U, as i16
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
      scalar::packed444_to_rgb_or_rgba_row::<ALPHA, ALPHA_SRC, V_OFF, U_OFF, Y_OFF, A_OFF>(
        tail_packed,
        tail_out,
        width - x,
        matrix,
        full_range,
      );
    }
  }
}

/// VUYA / VUYX byte order (`V=0,U=1,Y=2,A=3`) over the offset-generic wasm
/// kernel.
///
/// # Safety
///
/// Same contract as [`packed444_to_rgb_or_rgba_row`].
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn vuya_to_rgb_or_rgba_row<const ALPHA: bool, const ALPHA_SRC: bool>(
  packed: &[u8],
  out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  unsafe {
    packed444_to_rgb_or_rgba_row::<ALPHA, ALPHA_SRC, 0, 1, 2, 3>(
      packed, out, width, matrix, full_range,
    );
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
  matrix: KernelMatrix,
  full_range: bool,
) {
  unsafe {
    vuya_to_rgb_or_rgba_row::<false, false>(packed, rgb_out, width, matrix, full_range);
  }
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
  matrix: KernelMatrix,
  full_range: bool,
) {
  unsafe {
    vuya_to_rgb_or_rgba_row::<true, true>(packed, rgba_out, width, matrix, full_range);
  }
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
  matrix: KernelMatrix,
  full_range: bool,
) {
  unsafe {
    vuya_to_rgb_or_rgba_row::<true, false>(packed, rgba_out, width, matrix, full_range);
  }
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
pub(crate) unsafe fn packed444_to_luma_row<const Y_OFF: usize>(
  packed: &[u8],
  luma_out: &mut [u8],
  width: usize,
) {
  debug_assert!(packed.len() >= width * 4, "packed row too short");
  debug_assert!(luma_out.len() >= width, "luma row too short");

  unsafe {
    // Y at `Y_OFF` within each 4-byte pixel → positions `Y_OFF, +4, +8, +12`.
    let y_idx = chan_idx_wasm::<Y_OFF>();

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
      scalar::packed444_to_luma_row::<Y_OFF>(&packed[x * 4..], &mut luma_out[x..], width - x);
    }
  }
}

/// wasm-simd128 VUYA / VUYX u8 luma (Y at offset 2) over
/// [`packed444_to_luma_row`].
///
/// # Safety
///
/// Same contract as [`packed444_to_luma_row`].
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn vuya_to_luma_row(packed: &[u8], luma_out: &mut [u8], width: usize) {
  unsafe {
    packed444_to_luma_row::<2>(packed, luma_out, width);
  }
}

/// wasm-simd128 VUYA → u16 luma (zero-extended Y bytes). Y is the third
/// byte (offset 2) of each pixel quadruple. 16 pixels per SIMD iteration.
///
/// Strategy: same 4-load + swizzle cascade as the u8 path to collect 16 Y
/// bytes into a `v128`, then `u16x8_extend_low_u8x16` and
/// `u16x8_extend_high_u8x16` widen the two halves to u16x8 each.
///
/// Byte-identical to `scalar::vuya_to_luma_u16_row`.
///
/// # Safety
///
/// 1. **`simd128` must be enabled at compile time.**
/// 2. `packed.len() >= width * 4`.
/// 3. `out.len() >= width`.
#[cfg_attr(not(any(feature = "std", feature = "alloc")), allow(dead_code))]
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn packed444_to_luma_u16_row<const Y_OFF: usize>(
  packed: &[u8],
  out: &mut [u16],
  width: usize,
) {
  debug_assert!(packed.len() >= width * 4, "packed row too short");
  debug_assert!(out.len() >= width, "out too short");

  unsafe {
    // Y at `Y_OFF` within each 4-byte pixel → positions `Y_OFF, +4, +8, +12`.
    let y_idx = chan_idx_wasm::<Y_OFF>();

    let mut x = 0usize;
    while x + 16 <= width {
      let ptr = packed.as_ptr().add(x * 4);
      let raw0 = v128_load(ptr.cast()); // pixels 0-3
      let raw1 = v128_load(ptr.add(16).cast()); // pixels 4-7
      let raw2 = v128_load(ptr.add(32).cast()); // pixels 8-11
      let raw3 = v128_load(ptr.add(48).cast()); // pixels 12-15

      // Extract Y from each 4-pixel register → 4 bytes in low 4 bytes.
      let y0 = u8x16_swizzle(raw0, y_idx);
      let y1 = u8x16_swizzle(raw1, y_idx);
      let y2 = u8x16_swizzle(raw2, y_idx);
      let y3 = u8x16_swizzle(raw3, y_idx);

      // Concatenate pairs into 8-byte fragments.
      let y01 = i8x16_shuffle::<0, 1, 2, 3, 16, 17, 18, 19, 0, 0, 0, 0, 0, 0, 0, 0>(y0, y1);
      let y23 = i8x16_shuffle::<0, 1, 2, 3, 16, 17, 18, 19, 0, 0, 0, 0, 0, 0, 0, 0>(y2, y3);
      // Full 16-lane Y vector.
      let y_vec = i8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23>(y01, y23);

      // Zero-extend low 8 bytes → u16x8, high 8 bytes → u16x8.
      let low = u16x8_extend_low_u8x16(y_vec);
      let high = u16x8_extend_high_u8x16(y_vec);
      v128_store(out.as_mut_ptr().add(x).cast(), low);
      v128_store(out.as_mut_ptr().add(x + 8).cast(), high);
      x += 16;
    }

    // Scalar tail.
    if x < width {
      scalar::packed444_to_luma_u16_row::<Y_OFF>(&packed[x * 4..], &mut out[x..], width - x);
    }
  }
}

/// wasm-simd128 VUYA u16 luma (Y at offset 2) over
/// [`packed444_to_luma_u16_row`].
///
/// # Safety
///
/// Same contract as [`packed444_to_luma_u16_row`].
#[cfg_attr(not(any(feature = "std", feature = "alloc")), allow(dead_code))]
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn vuya_to_luma_u16_row(packed: &[u8], out: &mut [u16], width: usize) {
  unsafe {
    packed444_to_luma_u16_row::<2>(packed, out, width);
  }
}

/// wasm-simd128 VUYX → u16 luma (zero-extended Y bytes). Byte-identical
/// to [`vuya_to_luma_u16_row`] — Y is at byte offset 2 of each quadruple
/// regardless of α semantics; the X byte is discarded.
///
/// # Safety
///
/// 1. **`simd128` must be enabled at compile time.**
/// 2. `packed.len() >= width * 4`.
/// 3. `out.len() >= width`.
//
// Always dead-code: the dispatcher (`row::dispatch::vuyx`) re-uses the
// shared VUYA luma_u16 kernel directly via `vuya_to_luma_u16_row`,
// rather than calling this per-arch shim. Keep it for symmetry with
// the other backends and to document the equivalence. The
// `not(std/alloc)` gate previously here did not cover wasm32-emscripten
// builds with std enabled.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn vuyx_to_luma_u16_row(packed: &[u8], out: &mut [u16], width: usize) {
  unsafe {
    vuya_to_luma_u16_row(packed, out, width);
  }
}

// ---- VUYA → HSV (staged via a reused 8-bit RGB chunk) ----------------
//
// The SIMD twin of the scalar `vuya_to_hsv_row` kernel. Rather than
// re-derive an HSV-specific register pipeline, it fills a small fixed
// reused 8-bit RGB scratch (one `HSV_CHUNK`-pixel chunk at a time)
// using the EXISTING vuya_to_rgb_row kernel of this file — so the chunk
// filler IS the production 8-bit RGB kernel — then runs the SIMD
// `rgb_to_hsv_row` on the chunk. This makes the result byte-identical to
// `rgb_to_hsv_row(vuya_to_rgb_row(...))` within this SIMD tier with no source-width RGB allocation. The
// scalar tail of the underlying RGB kernel handles widths below the SIMD
// block, so no separate tail is needed here. The α byte (slot 3) is
// dropped by the RGB kernel — HSV is colour-only — so a single kernel
// serves both VUYA (real α) and VUYX (padding); `vuyx_to_hsv_row` is a
// thin re-export.
//
// The chunked driver is defined locally (mirroring the semi-planar
// high-bit `pn_hsv_via_rgb_chunks`) and gated `yuv-444-packed` with the
// rest of this file. Only `rgb_to_hsv_row` (ungated) is shared.

/// One reused 8-bit RGB chunk's worth of pixels staged before the HSV
/// pass.
const HSV_CHUNK: usize = 64;

/// Shared SIMD driver: walks `width` in `HSV_CHUNK`-pixel chunks, fills a
/// small reused stack RGB scratch via `fill_rgb` (the existing SIMD RGB
/// kernel for the format, passed the chunk `offset` and length `n`),
/// then runs the SIMD [`rgb_to_hsv_row`] on that chunk into the H/S/V
/// planes. Byte-identical to `rgb_to_hsv_row(vuya_to_rgb_row(...))` within this SIMD tier, with no
/// source-width RGB allocation.
///
/// `fill_rgb` receives `(offset, n, &mut rgb_chunk)` and must write
/// `n * 3` packed RGB bytes for the `n` pixels at `offset`.
///
/// # Safety
///
/// The SIMD feature must be available, and `fill_rgb` must uphold the
/// underlying RGB kernel's safety contract for each chunk. Each of
/// `h_out` / `s_out` / `v_out` must be `>= width`.
#[inline]
unsafe fn vuya_hsv_via_rgb_chunks(
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
    // SAFETY: the SIMD feature is verified by the wrapper's
    // `#[target_feature]`; the chunk and the output sub-slices are all
    // length `n`.
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

/// SIMD: VUYA (packed 4:4:4, 8-bit) → planar HSV bytes (OpenCV
/// encoding), staged via the reused-8-bit-RGB-chunk pattern over the
/// SIMD [`vuya_to_rgb_row`] + [`rgb_to_hsv_row`]. Byte-identical to
/// `rgb_to_hsv_row(vuya_to_rgb_row(...))` within this SIMD tier. The α
/// byte is dropped (HSV is colour-only).
///
/// # Safety
///
/// 1. The SIMD feature must be available.
/// 2. `packed.len() >= width * 4`.
/// 3. `h_out.len()`, `s_out.len()`, `v_out.len()` `>= width`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn packed444_to_hsv_row<
  const V_OFF: usize,
  const U_OFF: usize,
  const Y_OFF: usize,
>(
  packed: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  debug_assert!(packed.len() >= width * 4, "packed row too short");
  debug_assert!(h_out.len() >= width, "h_out row too short");
  debug_assert!(s_out.len() >= width, "s_out row too short");
  debug_assert!(v_out.len() >= width, "v_out row too short");

  // SAFETY: the feature is the caller's obligation; the chunk filler
  // forwards the per-chunk sub-slices to the offset-generic wasm RGB kernel
  // (no alpha). `A_OFF` unused in the RGB path; reuse `Y_OFF`.
  unsafe {
    vuya_hsv_via_rgb_chunks(h_out, s_out, v_out, width, |offset, n, rgb| {
      packed444_to_rgb_or_rgba_row::<false, false, V_OFF, U_OFF, Y_OFF, Y_OFF>(
        &packed[offset * 4..],
        rgb,
        n,
        matrix,
        full_range,
      );
    });
  }
}

/// VUYA / VUYX HSV (`V=0,U=1,Y=2`) over [`packed444_to_hsv_row`].
///
/// # Safety
///
/// Same contract as [`packed444_to_hsv_row`].
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn vuya_to_hsv_row(
  packed: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  unsafe {
    packed444_to_hsv_row::<0, 1, 2>(packed, h_out, s_out, v_out, width, matrix, full_range);
  }
}

// ---- AYUV / UYVA wasm-simd128 wrappers --------------------------------

/// wasm-simd128 AYUV (`A=0,Y=1,U=2,V=3`) → packed RGB.
///
/// # Safety
///
/// Same contract as [`packed444_to_rgb_or_rgba_row`].
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn ayuv_to_rgb_row(
  packed: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  unsafe {
    packed444_to_rgb_or_rgba_row::<false, false, 3, 2, 1, 0>(
      packed, rgb_out, width, matrix, full_range,
    );
  }
}

/// wasm-simd128 AYUV → packed RGBA (source α at offset 0).
///
/// # Safety
///
/// Same contract as [`packed444_to_rgb_or_rgba_row`].
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn ayuv_to_rgba_row(
  packed: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  unsafe {
    packed444_to_rgb_or_rgba_row::<true, true, 3, 2, 1, 0>(
      packed, rgba_out, width, matrix, full_range,
    );
  }
}

/// wasm-simd128 AYUV → planar HSV bytes.
///
/// # Safety
///
/// Same contract as [`packed444_to_hsv_row`].
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn ayuv_to_hsv_row(
  packed: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  unsafe {
    packed444_to_hsv_row::<3, 2, 1>(packed, h_out, s_out, v_out, width, matrix, full_range);
  }
}

/// wasm-simd128 AYUV → u8 luma (Y at offset 1).
///
/// # Safety
///
/// Same contract as [`packed444_to_luma_row`].
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn ayuv_to_luma_row(packed: &[u8], luma_out: &mut [u8], width: usize) {
  unsafe {
    packed444_to_luma_row::<1>(packed, luma_out, width);
  }
}

/// wasm-simd128 AYUV → u16 luma (Y at offset 1).
///
/// # Safety
///
/// Same contract as [`packed444_to_luma_u16_row`].
#[cfg_attr(not(any(feature = "std", feature = "alloc")), allow(dead_code))]
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn ayuv_to_luma_u16_row(packed: &[u8], out: &mut [u16], width: usize) {
  unsafe {
    packed444_to_luma_u16_row::<1>(packed, out, width);
  }
}

/// wasm-simd128 UYVA (`U=0,Y=1,V=2,A=3`) → packed RGB.
///
/// # Safety
///
/// Same contract as [`packed444_to_rgb_or_rgba_row`].
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn uyva_to_rgb_row(
  packed: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  unsafe {
    packed444_to_rgb_or_rgba_row::<false, false, 2, 0, 1, 3>(
      packed, rgb_out, width, matrix, full_range,
    );
  }
}

/// wasm-simd128 UYVA → packed RGBA (source α at offset 3).
///
/// # Safety
///
/// Same contract as [`packed444_to_rgb_or_rgba_row`].
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn uyva_to_rgba_row(
  packed: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  unsafe {
    packed444_to_rgb_or_rgba_row::<true, true, 2, 0, 1, 3>(
      packed, rgba_out, width, matrix, full_range,
    );
  }
}

/// wasm-simd128 UYVA → planar HSV bytes.
///
/// # Safety
///
/// Same contract as [`packed444_to_hsv_row`].
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn uyva_to_hsv_row(
  packed: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  unsafe {
    packed444_to_hsv_row::<2, 0, 1>(packed, h_out, s_out, v_out, width, matrix, full_range);
  }
}

/// wasm-simd128 UYVA → u8 luma (Y at offset 1).
///
/// # Safety
///
/// Same contract as [`packed444_to_luma_row`].
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn uyva_to_luma_row(packed: &[u8], luma_out: &mut [u8], width: usize) {
  unsafe {
    packed444_to_luma_row::<1>(packed, luma_out, width);
  }
}

/// wasm-simd128 UYVA → u16 luma (Y at offset 1).
///
/// # Safety
///
/// Same contract as [`packed444_to_luma_u16_row`].
#[cfg_attr(not(any(feature = "std", feature = "alloc")), allow(dead_code))]
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn uyva_to_luma_u16_row(packed: &[u8], out: &mut [u16], width: usize) {
  unsafe {
    packed444_to_luma_u16_row::<1>(packed, out, width);
  }
}

// ---- VYU444 (V=0, Y=1, U=2; 3 bytes per pixel, no alpha) wasm-simd128 --
//
// VYU444 packs three bytes per pixel (`V ‖ Y ‖ U`, 24bpp). The 3-byte
// de-interleave below mirrors the x86 SSE4.1 `deinterleave_rgb_16` /
// wasm `rgb_to_hsv` 9-mask `u8x16_swizzle` pattern bgr24/rgb24 use: three
// `v128_load`s + nine swizzles + `v128_or` split 16 triples into three
// channel vectors at byte offsets 0/1/2 = `(V, Y, U)`. Those feed the same
// Q15 chroma + Y pipeline as the 4-byte VUYA kernel above. RGBA output
// forces α = `0xFF` (no source alpha). The scalar tail handles `width % 16`.

/// Deinterleaves 16 packed VYU444 pixels (48 bytes at `ptr`) into three
/// 16-lane `v128`s holding contiguous V / Y / U byte planes (offsets
/// 0/1/2). Same 9-mask byte-shuffle pattern as the SSE4.1
/// `deinterleave_rgb_16` and the wasm RGB-input kernels.
///
/// # Safety
///
/// `ptr` must point to at least 48 readable bytes (16 px × 3 ch). simd128
/// must be enabled.
#[inline]
#[target_feature(enable = "simd128")]
unsafe fn deinterleave_vyu444_16px(ptr: *const u8) -> (v128, v128, v128) {
  unsafe {
    let in0 = v128_load(ptr.cast());
    let in1 = v128_load(ptr.add(16).cast());
    let in2 = v128_load(ptr.add(32).cast());

    // V (channel 0) bytes at absolute positions 3k: chunk0 [0,3,6,9,12,15],
    // chunk1 [2,5,8,11,14], chunk2 [1,4,7,10,13].
    let mv0 = i8x16(0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let mv1 = i8x16(-1, -1, -1, -1, -1, -1, 2, 5, 8, 11, 14, -1, -1, -1, -1, -1);
    let mv2 = i8x16(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 4, 7, 10, 13);
    let v_u8 = v128_or(
      v128_or(u8x16_swizzle(in0, mv0), u8x16_swizzle(in1, mv1)),
      u8x16_swizzle(in2, mv2),
    );

    // Y (channel 1) bytes at positions 3k+1: chunk0 [1,4,7,10,13], chunk1
    // [0,3,6,9,12,15], chunk2 [2,5,8,11,14].
    let my0 = i8x16(1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let my1 = i8x16(-1, -1, -1, -1, -1, 0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1);
    let my2 = i8x16(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 5, 8, 11, 14);
    let y_u8 = v128_or(
      v128_or(u8x16_swizzle(in0, my0), u8x16_swizzle(in1, my1)),
      u8x16_swizzle(in2, my2),
    );

    // U (channel 2) bytes at positions 3k+2: chunk0 [2,5,8,11,14], chunk1
    // [1,4,7,10,13], chunk2 [0,3,6,9,12,15].
    let mu0 = i8x16(2, 5, 8, 11, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let mu1 = i8x16(-1, -1, -1, -1, -1, 1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1);
    let mu2 = i8x16(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 3, 6, 9, 12, 15);
    let u_u8 = v128_or(
      v128_or(u8x16_swizzle(in0, mu0), u8x16_swizzle(in1, mu1)),
      u8x16_swizzle(in2, mu2),
    );

    (v_u8, y_u8, u_u8)
  }
}

/// wasm-simd128 VYU444 → packed u8 RGB (`ALPHA = false`) or RGBA
/// (`ALPHA = true`, α forced `0xFF`). 16 pixels per iteration.
///
/// Byte-identical to `scalar::vyu444_to_rgb_or_rgba_row::<ALPHA>`.
///
/// # Safety
///
/// 1. **`simd128` must be enabled at compile time.**
/// 2. `packed.len() >= width * 3`.
/// 3. `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn vyu444_to_rgb_or_rgba_row<const ALPHA: bool>(
  packed: &[u8],
  out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  debug_assert!(packed.len() >= width * 3, "packed row too short");
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
      // 3-byte de-interleave: channels at offsets 0/1/2 → (V, Y, U).
      let (v_raw, y_raw, u_raw) = deinterleave_vyu444_16px(packed.as_ptr().add(x * 3));

      // Zero-extend U/V/Y bytes to i16x8 (low half and high half).
      let u_lo = u8_low_to_i16x8(u_raw);
      let u_hi = u8_high_to_i16x8(u_raw);
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

      // 4:4:4 — each of 16 pixels has unique U/V. Chroma for low 8 lanes.
      let r_chroma_lo = chroma_i16x8(cru, crv, u_d_lo_lo, v_d_lo_lo, u_d_lo_hi, v_d_lo_hi, rnd_v);
      let g_chroma_lo = chroma_i16x8(cgu, cgv, u_d_lo_lo, v_d_lo_lo, u_d_lo_hi, v_d_lo_hi, rnd_v);
      let b_chroma_lo = chroma_i16x8(cbu, cbv, u_d_lo_lo, v_d_lo_lo, u_d_lo_hi, v_d_lo_hi, rnd_v);

      // Chroma for high 8 lanes.
      let r_chroma_hi = chroma_i16x8(cru, crv, u_d_hi_lo, v_d_hi_lo, u_d_hi_hi, v_d_hi_hi, rnd_v);
      let g_chroma_hi = chroma_i16x8(cgu, cgv, u_d_hi_lo, v_d_hi_lo, u_d_hi_hi, v_d_hi_hi, rnd_v);
      let b_chroma_hi = chroma_i16x8(cbu, cbv, u_d_hi_lo, v_d_hi_lo, u_d_hi_hi, v_d_hi_hi, rnd_v);

      // Y: scale both halves.
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

      // Write 16 pixels. RGBA α is forced opaque (no source alpha).
      if ALPHA {
        write_rgba_16(r_u8, g_u8, b_u8, alpha_u8, out.as_mut_ptr().add(x * 4));
      } else {
        write_rgb_16(r_u8, g_u8, b_u8, out.as_mut_ptr().add(x * 3));
      }

      x += 16;
    }

    // Scalar tail — remaining < 16 pixels.
    if x < width {
      scalar::vyu444_to_rgb_or_rgba_row::<ALPHA>(
        &packed[x * 3..],
        &mut out[x * bpp..],
        width - x,
        matrix,
        full_range,
      );
    }
  }
}

/// wasm-simd128 VYU444 → packed RGB (3 bpp).
///
/// # Safety
///
/// `packed.len() >= width * 3`; `rgb_out.len() >= width * 3`. simd128 must
/// be enabled.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn vyu444_to_rgb_row(
  packed: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  unsafe {
    vyu444_to_rgb_or_rgba_row::<false>(packed, rgb_out, width, matrix, full_range);
  }
}

/// wasm-simd128 VYU444 → packed RGBA (4 bpp, α forced `0xFF`).
///
/// # Safety
///
/// `packed.len() >= width * 3`; `rgba_out.len() >= width * 4`. simd128 must
/// be enabled.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn vyu444_to_rgba_row(
  packed: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  unsafe {
    vyu444_to_rgb_or_rgba_row::<true>(packed, rgba_out, width, matrix, full_range);
  }
}

/// wasm-simd128 VYU444 → planar HSV bytes, staged via the
/// reused-8-bit-RGB-chunk pattern over the wasm [`vyu444_to_rgb_row`] +
/// [`rgb_to_hsv_row`]. Byte-identical to
/// `rgb_to_hsv_row(vyu444_to_rgb_row(...))` within this tier.
///
/// # Safety
///
/// `packed.len() >= width * 3`; each H/S/V plane `>= width`. simd128 must
/// be enabled.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn vyu444_to_hsv_row(
  packed: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  debug_assert!(packed.len() >= width * 3, "packed row too short");
  debug_assert!(h_out.len() >= width, "h_out row too short");
  debug_assert!(s_out.len() >= width, "s_out row too short");
  debug_assert!(v_out.len() >= width, "v_out row too short");
  unsafe {
    vuya_hsv_via_rgb_chunks(h_out, s_out, v_out, width, |offset, n, rgb| {
      vyu444_to_rgb_row(&packed[offset * 3..], rgb, n, matrix, full_range);
    });
  }
}

/// wasm-simd128 VYU444 → u8 luma (Y at offset 1, 3-byte stride). The
/// de-interleave's channel 1 delivers Y for all 16 pixels.
///
/// # Safety
///
/// `packed.len() >= width * 3`; `luma_out.len() >= width`. simd128 must be
/// enabled.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn vyu444_to_luma_row(packed: &[u8], luma_out: &mut [u8], width: usize) {
  debug_assert!(packed.len() >= width * 3, "packed row too short");
  debug_assert!(luma_out.len() >= width, "luma row too short");
  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      let (_v, y_u8, _u) = deinterleave_vyu444_16px(packed.as_ptr().add(x * 3));
      v128_store(luma_out.as_mut_ptr().add(x).cast(), y_u8);
      x += 16;
    }
    if x < width {
      scalar::vyu444_to_luma_row(&packed[x * 3..], &mut luma_out[x..], width - x);
    }
  }
}

/// wasm-simd128 VYU444 → u16 luma (zero-extended Y at offset 1, 3-byte
/// stride).
///
/// # Safety
///
/// `packed.len() >= width * 3`; `out.len() >= width`. simd128 must be
/// enabled.
#[cfg_attr(not(any(feature = "std", feature = "alloc")), allow(dead_code))]
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn vyu444_to_luma_u16_row(packed: &[u8], out: &mut [u16], width: usize) {
  debug_assert!(packed.len() >= width * 3, "packed row too short");
  debug_assert!(out.len() >= width, "out too short");
  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      let (_v, y_u8, _u) = deinterleave_vyu444_16px(packed.as_ptr().add(x * 3));
      let lo_u16 = u16x8_extend_low_u8x16(y_u8);
      let hi_u16 = u16x8_extend_high_u8x16(y_u8);
      v128_store(out.as_mut_ptr().add(x).cast(), lo_u16);
      v128_store(out.as_mut_ptr().add(x + 8).cast(), hi_u16);
      x += 16;
    }
    if x < width {
      scalar::vyu444_to_luma_u16_row(&packed[x * 3..], &mut out[x..], width - x);
    }
  }
}
