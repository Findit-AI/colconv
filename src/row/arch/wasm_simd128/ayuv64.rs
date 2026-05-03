//! wasm-simd128 kernels for AYUV64 packed YUV 4:4:4 16-bit
//! (FFmpeg `AV_PIX_FMT_AYUV64LE`).
//!
//! ## Layout
//!
//! Four `u16` elements per pixel: `A(16) ‖ Y(16) ‖ U(16) ‖ V(16)`.
//! All channels are 16-bit native — no padding bits, no right-shift on load.
//! Channel slot order at deinterleave output: **A=0, Y=1, U=2, V=3**.
//!
//! In memory (little-endian, 8 bytes per pixel):
//! ```text
//!   byte 0,1 = A   byte 2,3 = Y   byte 4,5 = U   byte 6,7 = V
//! ```
//!
//! ## u8 pipeline (16 px / iter)
//!
//! Two 8-pixel half-iterations per SIMD block. Each half:
//!
//! 1. Load 4 × `v128` (8 pixels × 4 u16 = 64 bytes).
//! 2. `u8x16_swizzle` isolates each channel's 2 u16 samples from each
//!    2-pixel register into the low 4 bytes (high 12 bytes zeroed via 0xFF).
//! 3. `i8x16_shuffle` cascade assembles 4 × 4-byte fragments → full
//!    8-lane u16x8 channel vector (same 3-level pattern as XV36).
//! 4. Chroma bias via wrapping 0x8000 i16 trick; Q15 pipeline.
//! 5. Y scaled via `scale_y_u16_wasm` (unsigned widening).
//! 6. Source α (A channel): `u16x8_shr(a_u16, 8)` → high byte in low byte
//!    slot, then `u8x16_narrow_i16x8` to narrow to u8.
//!    MUST be logical (u16x8_shr), not arithmetic (i16x8_shr): for A ≥ 0x8000
//!    the arithmetic shift sign-extends to a negative i16, and
//!    `u8x16_narrow_i16x8` then saturates to 0 — corrupting ~50 % of values.
//! 7. `write_rgba_16` / `write_rgb_16`.
//!
//! ## u16 pipeline (8 px / iter)
//!
//! Single 8-pixel block per iteration via the same 4-load + swizzle +
//! shuffle deinterleave. Uses `chroma_i64x2_wasm` (i64 chroma) +
//! `scale_y_i32x4_i64_wasm` to avoid overflow at BITS=16/16.
//! Source α: direct write (no conversion for u16 output).
//!
//! ## Tail
//!
//! `width % block_size` remaining pixels fall through to
//! `scalar::ayuv64_to_rgb_or_rgba_row::<ALPHA, ALPHA_SRC>` (or u16 variant).

use core::arch::wasm32::*;

use super::*;
use crate::{ColorMatrix, row::scalar};

// ---- Deinterleave helper ------------------------------------------------

/// Deinterleaves 8 AYUV64 quadruples (32 u16 = 64 bytes) from `ptr` into
/// `(a_vec, y_vec, u_vec, v_vec)` — four `v128` vectors each holding 8
/// `u16` samples.  No right-shift (16-bit native samples).
///
/// Channel slot order in memory: A=bytes 0,1; Y=bytes 2,3; U=bytes 4,5; V=bytes 6,7.
///
/// Strategy: same 3-level `u8x16_swizzle` + `i8x16_shuffle` cascade as XV36.
/// Each of the 4 raw registers holds 2 pixels (16 bytes).  `u8x16_swizzle`
/// packs each channel's 2 u16 samples into the low 4 bytes (high 12 zeroed).
/// Two levels of `i8x16_shuffle` concatenate pairs → 8-byte fragments →
/// final 8-lane channel vectors.
///
/// # Safety
///
/// `ptr` must point to at least 64 readable bytes (32 `u16` elements).
/// `simd128` must be enabled at compile time.
#[inline]
#[target_feature(enable = "simd128")]
unsafe fn deinterleave_ayuv64_8px(ptr: *const u16) -> (v128, v128, v128, v128) {
  unsafe {
    // Load 4 × v128, each covering 2 pixels (8 × u16 = 16 bytes).
    let raw0 = v128_load(ptr.cast()); // [A0,Y0,U0,V0, A1,Y1,U1,V1]
    let raw1 = v128_load(ptr.add(8).cast()); // [A2,Y2,U2,V2, A3,Y3,U3,V3]
    let raw2 = v128_load(ptr.add(16).cast()); // [A4,Y4,U4,V4, A5,Y5,U5,V5]
    let raw3 = v128_load(ptr.add(24).cast()); // [A6,Y6,U6,V6, A7,Y7,U7,V7]

    // Per-channel byte positions within a 2-pixel v128 (16 bytes):
    //   A → bytes  0,1  (pixel n) and  8,9  (pixel n+1)
    //   Y → bytes  2,3            and 10,11
    //   U → bytes  4,5            and 12,13
    //   V → bytes  6,7            and 14,15
    //
    // `u8x16_swizzle` with index ≥ 16 zeroes the output byte.
    // Pack each channel's 2 u16 samples into the LOW 4 bytes.

    let a_idx = i8x16(0, 1, 8, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let y_idx = i8x16(2, 3, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let u_idx = i8x16(4, 5, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let v_idx = i8x16(6, 7, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);

    let a0 = u8x16_swizzle(raw0, a_idx); // [A0 lo,hi, A1 lo,hi, 0..12]
    let a1 = u8x16_swizzle(raw1, a_idx);
    let a2 = u8x16_swizzle(raw2, a_idx);
    let a3 = u8x16_swizzle(raw3, a_idx);

    let y0 = u8x16_swizzle(raw0, y_idx);
    let y1 = u8x16_swizzle(raw1, y_idx);
    let y2 = u8x16_swizzle(raw2, y_idx);
    let y3 = u8x16_swizzle(raw3, y_idx);

    let u0 = u8x16_swizzle(raw0, u_idx);
    let u1 = u8x16_swizzle(raw1, u_idx);
    let u2 = u8x16_swizzle(raw2, u_idx);
    let u3 = u8x16_swizzle(raw3, u_idx);

    let v0 = u8x16_swizzle(raw0, v_idx);
    let v1 = u8x16_swizzle(raw1, v_idx);
    let v2 = u8x16_swizzle(raw2, v_idx);
    let v3 = u8x16_swizzle(raw3, v_idx);

    // Level 1: concatenate pairs of 4-byte fragments into 8-byte fragments.
    let a01 = i8x16_shuffle::<0, 1, 2, 3, 16, 17, 18, 19, 0, 0, 0, 0, 0, 0, 0, 0>(a0, a1);
    let a23 = i8x16_shuffle::<0, 1, 2, 3, 16, 17, 18, 19, 0, 0, 0, 0, 0, 0, 0, 0>(a2, a3);
    let y01 = i8x16_shuffle::<0, 1, 2, 3, 16, 17, 18, 19, 0, 0, 0, 0, 0, 0, 0, 0>(y0, y1);
    let y23 = i8x16_shuffle::<0, 1, 2, 3, 16, 17, 18, 19, 0, 0, 0, 0, 0, 0, 0, 0>(y2, y3);
    let u01 = i8x16_shuffle::<0, 1, 2, 3, 16, 17, 18, 19, 0, 0, 0, 0, 0, 0, 0, 0>(u0, u1);
    let u23 = i8x16_shuffle::<0, 1, 2, 3, 16, 17, 18, 19, 0, 0, 0, 0, 0, 0, 0, 0>(u2, u3);
    let v01 = i8x16_shuffle::<0, 1, 2, 3, 16, 17, 18, 19, 0, 0, 0, 0, 0, 0, 0, 0>(v0, v1);
    let v23 = i8x16_shuffle::<0, 1, 2, 3, 16, 17, 18, 19, 0, 0, 0, 0, 0, 0, 0, 0>(v2, v3);

    // Level 2: combine two 8-byte fragments into full 8-lane u16x8 vectors.
    let a_vec = i8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23>(a01, a23);
    let y_vec = i8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23>(y01, y23);
    let u_vec = i8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23>(u01, u23);
    let v_vec = i8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23>(v01, v23);

    (a_vec, y_vec, u_vec, v_vec)
  }
}

// ---- u8 RGB / RGBA output (16 px/iter) ----------------------------------

/// wasm-simd128 AYUV64 → packed u8 RGB or RGBA. 16 pixels per iteration.
///
/// Byte-identical to `scalar::ayuv64_to_rgb_or_rgba_row::<ALPHA, ALPHA_SRC>`.
///
/// Valid monomorphizations:
/// - `<false, false>` — RGB (α dropped)
/// - `<true, true>`  — RGBA, source α depth-converted u16 → u8 via `>> 8`
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
pub(crate) unsafe fn ayuv64_to_rgb_or_rgba_row<const ALPHA: bool, const ALPHA_SRC: bool>(
  packed: &[u16],
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
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<16, 8>(full_range);
  const RND: i32 = 1 << 14;

  unsafe {
    let rnd_v = i32x4_splat(RND);
    // Y values are full u16 (0..65535); use i32 y_off for scale_y_u16_wasm.
    let y_off32_v = i32x4_splat(y_off);
    let y_scale_v = i32x4_splat(y_scale);
    let c_scale_v = i32x4_splat(c_scale);
    // Subtract chroma bias (32768) via wrapping i16 trick: -32768i16 == 0x8000.
    let bias16_v = i16x8_splat(-32768i16);
    let cru = i32x4_splat(coeffs.r_u());
    let crv = i32x4_splat(coeffs.r_v());
    let cgu = i32x4_splat(coeffs.g_u());
    let cgv = i32x4_splat(coeffs.g_v());
    let cbu = i32x4_splat(coeffs.b_u());
    let cbv = i32x4_splat(coeffs.b_v());

    let mut x = 0usize;
    while x + 16 <= width {
      // --- lo half: pixels x..x+7 -------------------------------------------
      let (a_lo_u16, y_lo_u16, u_lo_u16, v_lo_u16) =
        deinterleave_ayuv64_8px(packed.as_ptr().add(x * 4));

      // Center chroma via wrapping i16 subtraction.
      let u_lo_i16 = i16x8_sub(u_lo_u16, bias16_v);
      let v_lo_i16 = i16x8_sub(v_lo_u16, bias16_v);

      // Widen chroma to i32x4 lo/hi for Q15 scale multiply.
      let u_lo_a = i32x4_extend_low_i16x8(u_lo_i16);
      let u_lo_b = i32x4_extend_high_i16x8(u_lo_i16);
      let v_lo_a = i32x4_extend_low_i16x8(v_lo_i16);
      let v_lo_b = i32x4_extend_high_i16x8(v_lo_i16);

      let u_d_lo_a = q15_shift(i32x4_add(i32x4_mul(u_lo_a, c_scale_v), rnd_v));
      let u_d_lo_b = q15_shift(i32x4_add(i32x4_mul(u_lo_b, c_scale_v), rnd_v));
      let v_d_lo_a = q15_shift(i32x4_add(i32x4_mul(v_lo_a, c_scale_v), rnd_v));
      let v_d_lo_b = q15_shift(i32x4_add(i32x4_mul(v_lo_b, c_scale_v), rnd_v));

      // 4:4:4 chroma for lo 8 lanes.
      let r_chroma_lo = chroma_i16x8(cru, crv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let g_chroma_lo = chroma_i16x8(cgu, cgv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let b_chroma_lo = chroma_i16x8(cbu, cbv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);

      // Y: unsigned-widen u16 → i32, scale. Returns i16x8.
      let y_lo_scaled = scale_y_u16_wasm(y_lo_u16, y_off32_v, y_scale_v, rnd_v);

      // Saturate-narrow to u8x8 (pack into low 8 bytes).
      let r_lo_u8 = u8x16_narrow_i16x8(i16x8_add_sat(y_lo_scaled, r_chroma_lo), i16x8_splat(0));
      let g_lo_u8 = u8x16_narrow_i16x8(i16x8_add_sat(y_lo_scaled, g_chroma_lo), i16x8_splat(0));
      let b_lo_u8 = u8x16_narrow_i16x8(i16x8_add_sat(y_lo_scaled, b_chroma_lo), i16x8_splat(0));

      // --- hi half: pixels x+8..x+15 ----------------------------------------
      let (a_hi_u16, y_hi_u16, u_hi_u16, v_hi_u16) =
        deinterleave_ayuv64_8px(packed.as_ptr().add(x * 4 + 32));

      let u_hi_i16 = i16x8_sub(u_hi_u16, bias16_v);
      let v_hi_i16 = i16x8_sub(v_hi_u16, bias16_v);

      let u_hi_a = i32x4_extend_low_i16x8(u_hi_i16);
      let u_hi_b = i32x4_extend_high_i16x8(u_hi_i16);
      let v_hi_a = i32x4_extend_low_i16x8(v_hi_i16);
      let v_hi_b = i32x4_extend_high_i16x8(v_hi_i16);

      let u_d_hi_a = q15_shift(i32x4_add(i32x4_mul(u_hi_a, c_scale_v), rnd_v));
      let u_d_hi_b = q15_shift(i32x4_add(i32x4_mul(u_hi_b, c_scale_v), rnd_v));
      let v_d_hi_a = q15_shift(i32x4_add(i32x4_mul(v_hi_a, c_scale_v), rnd_v));
      let v_d_hi_b = q15_shift(i32x4_add(i32x4_mul(v_hi_b, c_scale_v), rnd_v));

      let r_chroma_hi = chroma_i16x8(cru, crv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let g_chroma_hi = chroma_i16x8(cgu, cgv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let b_chroma_hi = chroma_i16x8(cbu, cbv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);

      let y_hi_scaled = scale_y_u16_wasm(y_hi_u16, y_off32_v, y_scale_v, rnd_v);

      let r_hi_u8 = u8x16_narrow_i16x8(i16x8_add_sat(y_hi_scaled, r_chroma_hi), i16x8_splat(0));
      let g_hi_u8 = u8x16_narrow_i16x8(i16x8_add_sat(y_hi_scaled, g_chroma_hi), i16x8_splat(0));
      let b_hi_u8 = u8x16_narrow_i16x8(i16x8_add_sat(y_hi_scaled, b_chroma_hi), i16x8_splat(0));

      // Combine lo+hi 8-byte halves into full 16-byte channel vectors.
      // Low 8 bytes of each narrow result hold pixels 0-7; combine via shuffle.
      let r_u8 =
        i8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23>(r_lo_u8, r_hi_u8);
      let g_u8 =
        i8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23>(g_lo_u8, g_hi_u8);
      let b_u8 =
        i8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23>(b_lo_u8, b_hi_u8);

      let out_ptr = out.as_mut_ptr().add(x * bpp);
      if ALPHA {
        // Depth-convert u16 → u8 via >> 8 (take high byte).
        // u16x8_shr is a LOGICAL shift — high byte lands in [0, 255].
        // Must NOT use i16x8_shr (arithmetic): for A ≥ 0x8000 the sign bit
        // propagates, making the result a negative i16 that u8x16_narrow_i16x8
        // saturates to 0 — corrupting all alpha values in the upper half of
        // the range. See xv36.rs:417 for the same correct pattern.
        let a_vec: v128 = if ALPHA_SRC {
          let a_lo_shr = u16x8_shr(a_lo_u16, 8);
          let a_hi_shr = u16x8_shr(a_hi_u16, 8);
          u8x16_narrow_i16x8(a_lo_shr, a_hi_shr)
        } else {
          u8x16_splat(0xFF)
        };
        write_rgba_16(r_u8, g_u8, b_u8, a_vec, out_ptr);
      } else {
        write_rgb_16(r_u8, g_u8, b_u8, out_ptr);
      }

      x += 16;
    }

    // Scalar tail — remaining < 16 pixels.
    if x < width {
      let tail_packed = &packed[x * 4..width * 4];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      scalar::ayuv64_to_rgb_or_rgba_row::<ALPHA, ALPHA_SRC>(
        tail_packed,
        tail_out,
        tail_w,
        matrix,
        full_range,
      );
    }
  }
}

// ---- thin wrappers (u8 output) -------------------------------------------

/// wasm-simd128 AYUV64 → packed **RGB** (3 bpp). Source α is discarded.
///
/// # Safety
///
/// 1. **`simd128` must be enabled at compile time.**
/// 2. `packed.len() >= width * 4`.
/// 3. `out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn ayuv64_to_rgb_row(
  packed: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  unsafe {
    ayuv64_to_rgb_or_rgba_row::<false, false>(packed, rgb_out, width, matrix, full_range);
  }
}

/// wasm-simd128 AYUV64 → packed **RGBA** (4 bpp). Source A u16 is
/// depth-converted to u8 via `>> 8`.
///
/// # Safety
///
/// 1. **`simd128` must be enabled at compile time.**
/// 2. `packed.len() >= width * 4`.
/// 3. `out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn ayuv64_to_rgba_row(
  packed: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  unsafe {
    ayuv64_to_rgb_or_rgba_row::<true, true>(packed, rgba_out, width, matrix, full_range);
  }
}

// ---- u16 RGB / RGBA native-depth output (8 px/iter) ----------------------

/// wasm-simd128 AYUV64 → packed native-depth u16 RGB or RGBA.
///
/// Uses i64 chroma (`chroma_i64x2_wasm`) to avoid overflow at BITS=16/16.
/// Byte-identical to
/// `scalar::ayuv64_to_rgb_u16_or_rgba_u16_row::<ALPHA, ALPHA_SRC>`.
///
/// Valid monomorphizations:
/// - `<false, false>` — RGB u16 (α dropped)
/// - `<true, true>`  — RGBA u16, source α written direct (no conversion)
///
/// `<false, true>` is rejected at monomorphization via `const { assert! }`.
///
/// # Safety
///
/// 1. **`simd128` must be enabled at compile time.**
/// 2. `packed.len() >= width * 4`.
/// 3. `out.len() >= width * (if ALPHA { 4 } else { 3 })` (u16 elements).
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn ayuv64_to_rgb_u16_or_rgba_u16_row<const ALPHA: bool, const ALPHA_SRC: bool>(
  packed: &[u16],
  out: &mut [u16],
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
    // Wrapping 0x8000 bias trick for full-u16 chroma.
    let bias16 = i16x8_splat(-32768i16);
    // Coefficients widened to i64x2 for chroma_i64x2_wasm.
    let cru = i64x2_extend_low_i32x4(i32x4_splat(coeffs.r_u()));
    let crv = i64x2_extend_low_i32x4(i32x4_splat(coeffs.r_v()));
    let cgu = i64x2_extend_low_i32x4(i32x4_splat(coeffs.g_u()));
    let cgv = i64x2_extend_low_i32x4(i32x4_splat(coeffs.g_v()));
    let cbu = i64x2_extend_low_i32x4(i32x4_splat(coeffs.b_u()));
    let cbv = i64x2_extend_low_i32x4(i32x4_splat(coeffs.b_v()));

    let mut x = 0usize;
    while x + 8 <= width {
      // Deinterleave 8 AYUV64 quadruples → A, Y, U, V as u16x8.
      let (a_u16, y_vec, u_u16, v_u16) = deinterleave_ayuv64_8px(packed.as_ptr().add(x * 4));

      // Center chroma via wrapping i16 subtraction.
      let u_i16 = i16x8_sub(u_u16, bias16);
      let v_i16 = i16x8_sub(v_u16, bias16);

      // Widen low 4 chroma lanes to i32x4.
      let u_lo_i32 = i32x4_extend_low_i16x8(u_i16);
      let v_lo_i32 = i32x4_extend_low_i16x8(v_i16);
      let u_d_lo = i32x4_shr(i32x4_add(i32x4_mul(u_lo_i32, c_scale_i32), rnd_i32), 15);
      let v_d_lo = i32x4_shr(i32x4_add(i32x4_mul(v_lo_i32, c_scale_i32), rnd_i32), 15);

      // Widen high 4 chroma lanes to i32x4.
      let u_hi_i32 = i32x4_extend_high_i16x8(u_i16);
      let v_hi_i32 = i32x4_extend_high_i16x8(v_i16);
      let u_d_hi = i32x4_shr(i32x4_add(i32x4_mul(u_hi_i32, c_scale_i32), rnd_i32), 15);
      let v_d_hi = i32x4_shr(i32x4_add(i32x4_mul(v_hi_i32, c_scale_i32), rnd_i32), 15);

      // Widen i32x4 chroma deltas to 2 × i64x2 for i64 chroma pipeline.
      let u_d_lo_lo = i64x2_extend_low_i32x4(u_d_lo);
      let u_d_lo_hi = i64x2_extend_high_i32x4(u_d_lo);
      let v_d_lo_lo = i64x2_extend_low_i32x4(v_d_lo);
      let v_d_lo_hi = i64x2_extend_high_i32x4(v_d_lo);
      let u_d_hi_lo = i64x2_extend_low_i32x4(u_d_hi);
      let u_d_hi_hi = i64x2_extend_high_i32x4(u_d_hi);
      let v_d_hi_lo = i64x2_extend_low_i32x4(v_d_hi);
      let v_d_hi_hi = i64x2_extend_high_i32x4(v_d_hi);

      // i64 chroma pipeline — 4:4:4, all 8 lanes unique.
      let r_ch_lo_lo = chroma_i64x2_wasm(cru, crv, u_d_lo_lo, v_d_lo_lo, rnd_i64);
      let r_ch_lo_hi = chroma_i64x2_wasm(cru, crv, u_d_lo_hi, v_d_lo_hi, rnd_i64);
      let g_ch_lo_lo = chroma_i64x2_wasm(cgu, cgv, u_d_lo_lo, v_d_lo_lo, rnd_i64);
      let g_ch_lo_hi = chroma_i64x2_wasm(cgu, cgv, u_d_lo_hi, v_d_lo_hi, rnd_i64);
      let b_ch_lo_lo = chroma_i64x2_wasm(cbu, cbv, u_d_lo_lo, v_d_lo_lo, rnd_i64);
      let b_ch_lo_hi = chroma_i64x2_wasm(cbu, cbv, u_d_lo_hi, v_d_lo_hi, rnd_i64);

      let r_ch_hi_lo = chroma_i64x2_wasm(cru, crv, u_d_hi_lo, v_d_hi_lo, rnd_i64);
      let r_ch_hi_hi = chroma_i64x2_wasm(cru, crv, u_d_hi_hi, v_d_hi_hi, rnd_i64);
      let g_ch_hi_lo = chroma_i64x2_wasm(cgu, cgv, u_d_hi_lo, v_d_hi_lo, rnd_i64);
      let g_ch_hi_hi = chroma_i64x2_wasm(cgu, cgv, u_d_hi_hi, v_d_hi_hi, rnd_i64);
      let b_ch_hi_lo = chroma_i64x2_wasm(cbu, cbv, u_d_hi_lo, v_d_hi_lo, rnd_i64);
      let b_ch_hi_hi = chroma_i64x2_wasm(cbu, cbv, u_d_hi_hi, v_d_hi_hi, rnd_i64);

      // Combine each i64x2 pair → i32x4 low-32-bits.
      let r_ch_lo_i32 = combine_i64x2_pair_to_i32x4(r_ch_lo_lo, r_ch_lo_hi);
      let g_ch_lo_i32 = combine_i64x2_pair_to_i32x4(g_ch_lo_lo, g_ch_lo_hi);
      let b_ch_lo_i32 = combine_i64x2_pair_to_i32x4(b_ch_lo_lo, b_ch_lo_hi);
      let r_ch_hi_i32 = combine_i64x2_pair_to_i32x4(r_ch_hi_lo, r_ch_hi_hi);
      let g_ch_hi_i32 = combine_i64x2_pair_to_i32x4(g_ch_hi_lo, g_ch_hi_hi);
      let b_ch_hi_i32 = combine_i64x2_pair_to_i32x4(b_ch_hi_lo, b_ch_hi_hi);

      // Y: unsigned-widen u16 → i32, subtract y_off, scale via i64.
      let y_lo_u32 = u32x4_extend_low_u16x8(y_vec);
      let y_hi_u32 = u32x4_extend_high_u16x8(y_vec);
      let y_lo_i32 = i32x4_sub(y_lo_u32, y_off32);
      let y_hi_i32 = i32x4_sub(y_hi_u32, y_off32);

      let y_lo_scaled = scale_y_i32x4_i64_wasm(y_lo_i32, y_scale_i64, rnd_i64);
      let y_hi_scaled = scale_y_i32x4_i64_wasm(y_hi_i32, y_scale_i64, rnd_i64);

      // Add Y + chroma, saturate i32 → u16 via u16x8_narrow_i32x4.
      let r_u16 = u16x8_narrow_i32x4(
        i32x4_add(y_lo_scaled, r_ch_lo_i32),
        i32x4_add(y_hi_scaled, r_ch_hi_i32),
      );
      let g_u16 = u16x8_narrow_i32x4(
        i32x4_add(y_lo_scaled, g_ch_lo_i32),
        i32x4_add(y_hi_scaled, g_ch_hi_i32),
      );
      let b_u16 = u16x8_narrow_i32x4(
        i32x4_add(y_lo_scaled, b_ch_lo_i32),
        i32x4_add(y_hi_scaled, b_ch_hi_i32),
      );

      if ALPHA {
        // Source α: direct write (no conversion needed for u16 output).
        let a_vec: v128 = if ALPHA_SRC { a_u16 } else { alpha_u16 };
        write_rgba_u16_8(r_u16, g_u16, b_u16, a_vec, out.as_mut_ptr().add(x * 4));
      } else {
        write_rgb_u16_8(r_u16, g_u16, b_u16, out.as_mut_ptr().add(x * 3));
      }

      x += 8;
    }

    // Scalar tail — remaining < 8 pixels.
    if x < width {
      let tail_packed = &packed[x * 4..width * 4];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      scalar::ayuv64_to_rgb_u16_or_rgba_u16_row::<ALPHA, ALPHA_SRC>(
        tail_packed,
        tail_out,
        tail_w,
        matrix,
        full_range,
      );
    }
  }
}

// ---- thin wrappers (u16 output) ------------------------------------------

/// wasm-simd128 AYUV64 → packed **RGB u16** (3 × u16 per pixel). Source α discarded.
///
/// # Safety
///
/// 1. **`simd128` must be enabled at compile time.**
/// 2. `packed.len() >= width * 4`.
/// 3. `out.len() >= width * 3` (u16 elements).
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn ayuv64_to_rgb_u16_row(
  packed: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  unsafe {
    ayuv64_to_rgb_u16_or_rgba_u16_row::<false, false>(packed, rgb_out, width, matrix, full_range);
  }
}

/// wasm-simd128 AYUV64 → packed **RGBA u16** (4 × u16 per pixel). Source A u16
/// is written direct (no conversion).
///
/// # Safety
///
/// 1. **`simd128` must be enabled at compile time.**
/// 2. `packed.len() >= width * 4`.
/// 3. `out.len() >= width * 4` (u16 elements).
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn ayuv64_to_rgba_u16_row(
  packed: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  unsafe {
    ayuv64_to_rgb_u16_or_rgba_u16_row::<true, true>(packed, rgba_out, width, matrix, full_range);
  }
}

// ---- Luma u8 (16 px/iter) -----------------------------------------------

/// wasm-simd128 AYUV64 → u8 luma. Y is slot 1 (bytes 2,3) of each pixel
/// quadruple; `>> 8` extracts the high byte giving an 8-bit value.
///
/// Uses two deinterleave calls (8 pixels each) per 16-pixel SIMD block.
///
/// Byte-identical to `scalar::ayuv64_to_luma_row`.
///
/// # Safety
///
/// 1. **`simd128` must be enabled at compile time.**
/// 2. `packed.len() >= width * 4`.
/// 3. `luma_out.len() >= width`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn ayuv64_to_luma_row(packed: &[u16], luma_out: &mut [u8], width: usize) {
  debug_assert!(packed.len() >= width * 4, "packed row too short");
  debug_assert!(luma_out.len() >= width, "luma row too short");

  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      // Two deinterleaves for 8 pixels each.
      let (_a_lo, y_lo, _u_lo, _v_lo) = deinterleave_ayuv64_8px(packed.as_ptr().add(x * 4));
      let (_a_hi, y_hi, _u_hi, _v_hi) = deinterleave_ayuv64_8px(packed.as_ptr().add(x * 4 + 32));

      // >> 8 to get u8 luma (high byte of each Y u16 sample).
      // Logical shift (u16x8_shr) — arithmetic shift (i16x8_shr) would
      // sign-extend Y ≥ 0x8000 to a negative i16, which u8x16_narrow_i16x8
      // would then saturate to 0, corrupting half the luma range.
      let y_lo_shr = u16x8_shr(y_lo, 8);
      let y_hi_shr = u16x8_shr(y_hi, 8);
      // Pack 16 × i16 → 16 × u8.
      let y_u8 = u8x16_narrow_i16x8(y_lo_shr, y_hi_shr);
      v128_store(luma_out.as_mut_ptr().add(x).cast(), y_u8);

      x += 16;
    }

    // Scalar tail.
    if x < width {
      scalar::ayuv64_to_luma_row(
        &packed[x * 4..width * 4],
        &mut luma_out[x..width],
        width - x,
      );
    }
  }
}

// ---- Luma u16 (16 px/iter) -----------------------------------------------

/// wasm-simd128 AYUV64 → u16 luma. Direct copy of Y samples (slot 1, no
/// shift — 16-bit native). Uses two deinterleave calls per 16-pixel block.
///
/// Byte-identical to `scalar::ayuv64_to_luma_u16_row`.
///
/// # Safety
///
/// 1. **`simd128` must be enabled at compile time.**
/// 2. `packed.len() >= width * 4`.
/// 3. `luma_out.len() >= width`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn ayuv64_to_luma_u16_row(packed: &[u16], luma_out: &mut [u16], width: usize) {
  debug_assert!(packed.len() >= width * 4, "packed row too short");
  debug_assert!(luma_out.len() >= width, "luma row too short");

  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      // Two deinterleaves for 8 pixels each.
      let (_a_lo, y_lo, _u_lo, _v_lo) = deinterleave_ayuv64_8px(packed.as_ptr().add(x * 4));
      let (_a_hi, y_hi, _u_hi, _v_hi) = deinterleave_ayuv64_8px(packed.as_ptr().add(x * 4 + 32));

      // Direct copy — Y samples are 16-bit native (no shift needed).
      v128_store(luma_out.as_mut_ptr().add(x).cast(), y_lo);
      v128_store(luma_out.as_mut_ptr().add(x + 8).cast(), y_hi);

      x += 16;
    }

    // Scalar tail.
    if x < width {
      scalar::ayuv64_to_luma_u16_row(
        &packed[x * 4..width * 4],
        &mut luma_out[x..width],
        width - x,
      );
    }
  }
}
