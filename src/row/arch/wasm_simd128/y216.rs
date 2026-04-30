//! wasm-simd128 Y216 (packed YUV 4:2:2, BITS=16) kernels.
//!
//! Layout per row: u16 quadruples `(Y₀, U, Y₁, V)` where each sample
//! occupies the full 16-bit word. **No right-shift on load** — unlike
//! Y210/Y212, BITS=16 means the samples are already full-range u16.
//!
//! ## Per-iter pipelines
//!
//! ### u8 RGB/RGBA output — 16 px / iter
//!
//! Four `v128_load` calls (two lo-group + two hi-group) feed the same
//! `u8x16_swizzle` + `i8x16_shuffle` deinterleave pattern as `y2xx.rs`,
//! but with shr_count = 0 (no MSB alignment shift). Chroma bias uses the
//! wrapping `0x8000` trick (`i16x8_splat(-32768i16)`), because after
//! zero-shift the full u16 chroma range causes the plain `i16x8_sub`
//! overflow avoidance that y2xx uses (bias fits in u16 = 32768 ≤ 32767
//! is FALSE for Y216 — bias IS 32768 = 0x8000 which as i16 is -32768,
//! so wrapping subtraction gives the correct signed value). Mirrors the
//! `yuv_420p16_to_rgb_or_rgba_row` pattern from `yuv_planar_16bit.rs`.
//!
//! The Y scale uses `scale_y_u16_wasm` (unsigned widening via
//! `u32x4_extend_{low,high}_u16x8`) to avoid i16 overflow for Y > 32767.
//!
//! ### u16 RGB/RGBA output — 8 px / iter
//!
//! Uses the i64 chroma pipeline (`chroma_i64x2_wasm`,
//! `scale_y_i32x4_i64_wasm`) — identical in structure to
//! `yuv_420p16_to_rgb_or_rgba_u16_row` but sourced from the YUYV
//! quadruple layout rather than separate planes.
//!
//! ### Luma u8 — 16 px / iter
//!
//! `>> 8` (high byte of Y) via two load + swizzle + shuffle, same as
//! `y2xx_n_to_luma_row` with a fixed count of 8.
//!
//! ### Luma u16 — 16 px / iter
//!
//! Direct extraction of Y lanes (no shift); stored via `v128_store`.
//!
//! ## Tail
//!
//! Pixels less than the next block multiple fall through to `scalar::y216_*`.

use core::arch::wasm32::*;

use super::*;
use crate::{ColorMatrix, row::scalar};

/// Loads 8 Y216 pixels (16 u16 samples = 32 bytes) and extracts
/// the Y, U, V vectors **without any right-shift** (BITS=16, already
/// full-range). Returns `(y_vec, u_vec, v_vec)` where:
/// - `y_vec`: 8 × u16 lanes holding Y0..Y7.
/// - `u_vec`: low 4 × u16 lanes U0..U3 (lanes 4..7 zeroed; don't-care).
/// - `v_vec`: low 4 × u16 lanes V0..V3 (lanes 4..7 zeroed; don't-care).
///
/// Mirrors `unpack_y2xx_8px_wasm` from `y2xx.rs` but omits the
/// `u16x8_shr` calls (shr_count = 0 would be a no-op).
///
/// # Safety
///
/// `ptr` must have at least 32 bytes (16 u16) readable; `simd128` must
/// be enabled at compile time.
#[inline]
#[target_feature(enable = "simd128")]
unsafe fn unpack_y216_8px_wasm(ptr: *const u16) -> (v128, v128, v128) {
  unsafe {
    // [Y0, U0, Y1, V0, Y2, U1, Y3, V1]
    let lo = v128_load(ptr.cast());
    // [Y4, U2, Y5, V2, Y6, U3, Y7, V3]
    let hi = v128_load(ptr.add(8).cast());

    // Y: even u16 lanes → low 8 bytes; high 8 zeroed by 0xFF mask.
    let y_idx = i8x16(0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1);
    let y_lo = u8x16_swizzle(lo, y_idx); // [Y0, Y1, Y2, Y3, _, _, _, _]
    let y_hi = u8x16_swizzle(hi, y_idx); // [Y4, Y5, Y6, Y7, _, _, _, _]
    // Merge low 8 bytes of each → [Y0..Y7]
    let y_vec =
      i8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23>(y_lo, y_hi);

    // Chroma: odd u16 lanes → [U0,V0,U1,V1, _, _, _, _] / [U2,V2,U3,V3, _, _, _, _]
    let c_idx = i8x16(2, 3, 6, 7, 10, 11, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1);
    let c_lo = u8x16_swizzle(lo, c_idx);
    let c_hi = u8x16_swizzle(hi, c_idx);
    // [U0,V0,U1,V1,U2,V2,U3,V3]
    let chroma =
      i8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23>(c_lo, c_hi);

    // No right-shift — BITS=16, samples already full-range.

    // Split U and V from interleaved chroma.
    let u_idx = i8x16(0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1);
    let v_idx = i8x16(2, 3, 6, 7, 10, 11, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1);
    let u_vec = u8x16_swizzle(chroma, u_idx);
    let v_vec = u8x16_swizzle(chroma, v_idx);
    (y_vec, u_vec, v_vec)
  }
}

/// wasm-simd128 Y216 → packed u8 RGB or RGBA. 16 pixels per iteration.
///
/// Byte-identical to `scalar::y216_to_rgb_or_rgba_row::<ALPHA>`.
///
/// # Safety
///
/// 1. **`simd128` must be enabled at compile time.**
/// 2. `width % 2 == 0`.
/// 3. `packed.len() >= width * 2`.
/// 4. `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn y216_to_rgb_or_rgba_row<const ALPHA: bool>(
  packed: &[u16],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(width.is_multiple_of(2), "Y216 requires even width");
  debug_assert!(packed.len() >= width * 2);
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<16, 8>(full_range);
  const RND: i32 = 1 << 14;

  unsafe {
    let rnd_v = i32x4_splat(RND);
    let y_off32_v = i32x4_splat(y_off);
    let y_scale_v = i32x4_splat(y_scale);
    let c_scale_v = i32x4_splat(c_scale);
    // Bias = 32768 = 0x8000; as i16 this wraps to -32768.
    // Using the wrapping trick (i16x8_sub with bias16 = -32768) correctly
    // maps full-u16 chroma [0, 65535] to [-32768, 32767].
    let bias16_v = i16x8_splat(-32768i16);
    let alpha_u8 = u8x16_splat(0xFF);
    let cru = i32x4_splat(coeffs.r_u());
    let crv = i32x4_splat(coeffs.r_v());
    let cgu = i32x4_splat(coeffs.g_u());
    let cgv = i32x4_splat(coeffs.g_v());
    let cbu = i32x4_splat(coeffs.b_u());
    let cbv = i32x4_splat(coeffs.b_v());

    let mut x = 0usize;
    // 16 px/iter: two groups of 8 (lo = Y0..Y7, hi = Y8..Y15).
    while x + 16 <= width {
      let (y_lo_vec, u_lo_vec, v_lo_vec) = unpack_y216_8px_wasm(packed.as_ptr().add(x * 2));
      let (y_hi_vec, u_hi_vec, v_hi_vec) = unpack_y216_8px_wasm(packed.as_ptr().add(x * 2 + 16));

      // Chroma bias subtraction (wrapping trick for full-u16 range).
      let u_lo_i16 = i16x8_sub(u_lo_vec, bias16_v);
      let v_lo_i16 = i16x8_sub(v_lo_vec, bias16_v);
      let u_hi_i16 = i16x8_sub(u_hi_vec, bias16_v);
      let v_hi_i16 = i16x8_sub(v_hi_vec, bias16_v);

      // Widen to i32x4 halves; only lo halves (lanes 0..3) are valid.
      // Hi halves hold zeros (from the swizzle mask) — don't-care since
      // `chroma_i16x8` discards lanes 4..7 after `dup_lo`.
      let u_lo_lo = i32x4_extend_low_i16x8(u_lo_i16);
      let u_lo_hi = i32x4_extend_high_i16x8(u_lo_i16);
      let v_lo_lo = i32x4_extend_low_i16x8(v_lo_i16);
      let v_lo_hi = i32x4_extend_high_i16x8(v_lo_i16);
      let u_hi_lo = i32x4_extend_low_i16x8(u_hi_i16);
      let u_hi_hi = i32x4_extend_high_i16x8(u_hi_i16);
      let v_hi_lo = i32x4_extend_low_i16x8(v_hi_i16);
      let v_hi_hi = i32x4_extend_high_i16x8(v_hi_i16);

      // Q15 chroma scale → i32x4 (scaled chroma deltas).
      let u_d_lo_lo = q15_shift(i32x4_add(i32x4_mul(u_lo_lo, c_scale_v), rnd_v));
      let u_d_lo_hi = q15_shift(i32x4_add(i32x4_mul(u_lo_hi, c_scale_v), rnd_v));
      let v_d_lo_lo = q15_shift(i32x4_add(i32x4_mul(v_lo_lo, c_scale_v), rnd_v));
      let v_d_lo_hi = q15_shift(i32x4_add(i32x4_mul(v_lo_hi, c_scale_v), rnd_v));
      let u_d_hi_lo = q15_shift(i32x4_add(i32x4_mul(u_hi_lo, c_scale_v), rnd_v));
      let u_d_hi_hi = q15_shift(i32x4_add(i32x4_mul(u_hi_hi, c_scale_v), rnd_v));
      let v_d_hi_lo = q15_shift(i32x4_add(i32x4_mul(v_hi_lo, c_scale_v), rnd_v));
      let v_d_hi_hi = q15_shift(i32x4_add(i32x4_mul(v_hi_hi, c_scale_v), rnd_v));

      // 8-lane i16 chroma vectors (valid in lanes 0..3; lanes 4..7 don't-care).
      let r_chroma_lo = chroma_i16x8(cru, crv, u_d_lo_lo, v_d_lo_lo, u_d_lo_hi, v_d_lo_hi, rnd_v);
      let g_chroma_lo = chroma_i16x8(cgu, cgv, u_d_lo_lo, v_d_lo_lo, u_d_lo_hi, v_d_lo_hi, rnd_v);
      let b_chroma_lo = chroma_i16x8(cbu, cbv, u_d_lo_lo, v_d_lo_lo, u_d_lo_hi, v_d_lo_hi, rnd_v);
      let r_chroma_hi = chroma_i16x8(cru, crv, u_d_hi_lo, v_d_hi_lo, u_d_hi_hi, v_d_hi_hi, rnd_v);
      let g_chroma_hi = chroma_i16x8(cgu, cgv, u_d_hi_lo, v_d_hi_lo, u_d_hi_hi, v_d_hi_hi, rnd_v);
      let b_chroma_hi = chroma_i16x8(cbu, cbv, u_d_hi_lo, v_d_hi_lo, u_d_hi_hi, v_d_hi_hi, rnd_v);

      // Duplicate chroma into Y-pair slots (4:2:2 nearest-neighbor upsample).
      let r_dup_lo = dup_lo(r_chroma_lo);
      let g_dup_lo = dup_lo(g_chroma_lo);
      let b_dup_lo = dup_lo(b_chroma_lo);
      let r_dup_hi = dup_lo(r_chroma_hi);
      let g_dup_hi = dup_lo(g_chroma_hi);
      let b_dup_hi = dup_lo(b_chroma_hi);

      // Y scale via unsigned widening (Y216 has full u16 range; i16 would
      // overflow for Y > 32767).
      let y_lo_scaled = scale_y_u16_wasm(y_lo_vec, y_off32_v, y_scale_v, rnd_v);
      let y_hi_scaled = scale_y_u16_wasm(y_hi_vec, y_off32_v, y_scale_v, rnd_v);

      // Saturating add → saturating narrow to u8x16.
      let r_lo = i16x8_add_sat(y_lo_scaled, r_dup_lo);
      let r_hi = i16x8_add_sat(y_hi_scaled, r_dup_hi);
      let g_lo = i16x8_add_sat(y_lo_scaled, g_dup_lo);
      let g_hi = i16x8_add_sat(y_hi_scaled, g_dup_hi);
      let b_lo = i16x8_add_sat(y_lo_scaled, b_dup_lo);
      let b_hi = i16x8_add_sat(y_hi_scaled, b_dup_hi);
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

    // Scalar tail — remaining < 16 pixels.
    if x < width {
      let tail_packed = &packed[x * 2..width * 2];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      scalar::y216_to_rgb_or_rgba_row::<ALPHA>(tail_packed, tail_out, tail_w, matrix, full_range);
    }
  }
}

/// wasm-simd128 Y216 → packed native-depth u16 RGB or RGBA.
/// 8 pixels per iteration using the i64 chroma pipeline.
///
/// Byte-identical to `scalar::y216_to_rgb_u16_or_rgba_u16_row::<ALPHA>`.
///
/// # Safety
///
/// 1. **`simd128` must be enabled at compile time.**
/// 2. `width % 2 == 0`.
/// 3. `packed.len() >= width * 2`.
/// 4. `out.len() >= width * (if ALPHA { 4 } else { 3 })` (u16 elements).
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn y216_to_rgb_u16_or_rgba_u16_row<const ALPHA: bool>(
  packed: &[u16],
  out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(width.is_multiple_of(2), "Y216 requires even width");
  debug_assert!(packed.len() >= width * 2);
  let bpp: usize = if ALPHA { 4 } else { 3 };
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
    // Wrapping 0x8000 bias trick for full-u16 chroma.
    let bias16 = i16x8_splat(-32768i16);
    // Coefficients widened once to i64x2.
    let cru = i64x2_extend_low_i32x4(i32x4_splat(coeffs.r_u()));
    let crv = i64x2_extend_low_i32x4(i32x4_splat(coeffs.r_v()));
    let cgu = i64x2_extend_low_i32x4(i32x4_splat(coeffs.g_u()));
    let cgv = i64x2_extend_low_i32x4(i32x4_splat(coeffs.g_v()));
    let cbu = i64x2_extend_low_i32x4(i32x4_splat(coeffs.b_u()));
    let cbv = i64x2_extend_low_i32x4(i32x4_splat(coeffs.b_v()));

    let mut x = 0usize;
    // 8 px/iter: one call to unpack_y216_8px_wasm gives Y0..Y7 and 4 UV pairs.
    while x + 8 <= width {
      let (y_vec, u_vec, v_vec) = unpack_y216_8px_wasm(packed.as_ptr().add(x * 2));

      // Chroma bias (wrapping trick).
      let u_i16 = i16x8_sub(u_vec, bias16);
      let v_i16 = i16x8_sub(v_vec, bias16);

      // Widen low 4 lanes to i32x4 (high 4 are zeroed don't-cares).
      let u_i32 = i32x4_extend_low_i16x8(u_i16);
      let v_i32 = i32x4_extend_low_i16x8(v_i16);

      // Q15 scale → 4 × i32 chroma deltas.
      let u_d = i32x4_shr(i32x4_add(i32x4_mul(u_i32, c_scale_i32), rnd_i32), 15);
      let v_d = i32x4_shr(i32x4_add(i32x4_mul(v_i32, c_scale_i32), rnd_i32), 15);

      // Widen to 2 × i64x2 for i64 chroma pipeline.
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

      // Combine each i64x2 pair → i32x4 [c0, c1, c2, c3].
      let r_ch_i32 = combine_i64x2_pair_to_i32x4(r_ch_lo, r_ch_hi);
      let g_ch_i32 = combine_i64x2_pair_to_i32x4(g_ch_lo, g_ch_hi);
      let b_ch_i32 = combine_i64x2_pair_to_i32x4(b_ch_lo, b_ch_hi);

      // Duplicate 4 chroma values into 8 per-pixel slots (4:2:2).
      // chroma_dup_i32x4_u16([c0,c1,c2,c3]) →
      //   lo = [c0,c0,c1,c1], hi = [c2,c2,c3,c3]
      let (r_dup_lo, r_dup_hi) = chroma_dup_i32x4_u16(r_ch_i32);
      let (g_dup_lo, g_dup_hi) = chroma_dup_i32x4_u16(g_ch_i32);
      let (b_dup_lo, b_dup_hi) = chroma_dup_i32x4_u16(b_ch_i32);

      // Y: unsigned widen 8 u16 → 2 × i32x4, subtract y_off, scale in i64.
      let y_lo_u32 = u32x4_extend_low_u16x8(y_vec);
      let y_hi_u32 = u32x4_extend_high_u16x8(y_vec);
      let y_lo_i32 = i32x4_sub(y_lo_u32, y_off32);
      let y_hi_i32 = i32x4_sub(y_hi_u32, y_off32);

      let y_lo_scaled = scale_y_i32x4_i64_wasm(y_lo_i32, y_scale_i64, rnd_i64);
      let y_hi_scaled = scale_y_i32x4_i64_wasm(y_hi_i32, y_scale_i64, rnd_i64);

      // Add Y + chroma, saturating narrow i32 → u16 (clamps [0, 65535]).
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

    // Scalar tail.
    if x < width {
      let tail_packed = &packed[x * 2..width * 2];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      scalar::y216_to_rgb_u16_or_rgba_u16_row::<ALPHA>(
        tail_packed,
        tail_out,
        tail_w,
        matrix,
        full_range,
      );
    }
  }
}

/// wasm-simd128 Y216 → u8 luma. Extracts Y via `>> 8`.
/// 16 pixels per iteration.
///
/// Byte-identical to `scalar::y216_to_luma_row`.
///
/// # Safety
///
/// 1. **`simd128` must be enabled at compile time.**
/// 2. `width % 2 == 0`.
/// 3. `packed.len() >= width * 2`.
/// 4. `luma_out.len() >= width`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn y216_to_luma_row(packed: &[u16], luma_out: &mut [u8], width: usize) {
  debug_assert!(width.is_multiple_of(2));
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(luma_out.len() >= width);

  unsafe {
    // Y permute: even u16 lanes → low 8 bytes; zeroed high.
    let y_idx = i8x16(0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1);

    let mut x = 0usize;
    // 16 px/iter: two groups of 8 Y samples.
    while x + 16 <= width {
      // lo group: Y0..Y7 from bytes x*2 .. x*2+32.
      let lo0 = v128_load(packed.as_ptr().add(x * 2).cast());
      let lo1 = v128_load(packed.as_ptr().add(x * 2 + 8).cast());
      let y_lo0 = u8x16_swizzle(lo0, y_idx);
      let y_lo1 = u8x16_swizzle(lo1, y_idx);
      let y_lo =
        i8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23>(y_lo0, y_lo1);

      // hi group: Y8..Y15 from bytes x*2+32 .. x*2+64.
      let hi0 = v128_load(packed.as_ptr().add(x * 2 + 16).cast());
      let hi1 = v128_load(packed.as_ptr().add(x * 2 + 24).cast());
      let y_hi0 = u8x16_swizzle(hi0, y_idx);
      let y_hi1 = u8x16_swizzle(hi1, y_idx);
      let y_hi =
        i8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23>(y_hi0, y_hi1);

      // >> 8: extract high byte of each u16 Y sample.
      let y_shr_lo = u16x8_shr(y_lo, 8);
      let y_shr_hi = u16x8_shr(y_hi, 8);
      // Narrow 16 i16 → 16 u8 (no saturation needed; values ≤ 255).
      let y_u8 = u8x16_narrow_i16x8(y_shr_lo, y_shr_hi);
      v128_store(luma_out.as_mut_ptr().add(x).cast(), y_u8);
      x += 16;
    }

    if x < width {
      let tail_packed = &packed[x * 2..width * 2];
      let tail_out = &mut luma_out[x..width];
      let tail_w = width - x;
      scalar::y216_to_luma_row(tail_packed, tail_out, tail_w);
    }
  }
}

/// wasm-simd128 Y216 → u16 luma. Direct extraction of Y samples
/// (no shift). 16 pixels per iteration.
///
/// Byte-identical to `scalar::y216_to_luma_u16_row`.
///
/// # Safety
///
/// 1. **`simd128` must be enabled at compile time.**
/// 2. `width % 2 == 0`.
/// 3. `packed.len() >= width * 2`.
/// 4. `luma_out.len() >= width`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn y216_to_luma_u16_row(packed: &[u16], luma_out: &mut [u16], width: usize) {
  debug_assert!(width.is_multiple_of(2));
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(luma_out.len() >= width);

  unsafe {
    let y_idx = i8x16(0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1);

    let mut x = 0usize;
    // 16 px/iter: two groups of 8 Y samples (u16 direct copy, no shift).
    while x + 16 <= width {
      // lo group: Y0..Y7
      let lo0 = v128_load(packed.as_ptr().add(x * 2).cast());
      let lo1 = v128_load(packed.as_ptr().add(x * 2 + 8).cast());
      let y_lo0 = u8x16_swizzle(lo0, y_idx);
      let y_lo1 = u8x16_swizzle(lo1, y_idx);
      let y_lo =
        i8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23>(y_lo0, y_lo1);

      // hi group: Y8..Y15
      let hi0 = v128_load(packed.as_ptr().add(x * 2 + 16).cast());
      let hi1 = v128_load(packed.as_ptr().add(x * 2 + 24).cast());
      let y_hi0 = u8x16_swizzle(hi0, y_idx);
      let y_hi1 = u8x16_swizzle(hi1, y_idx);
      let y_hi =
        i8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23>(y_hi0, y_hi1);

      // Direct store — full 16-bit Y, no shift.
      v128_store(luma_out.as_mut_ptr().add(x).cast(), y_lo);
      v128_store(luma_out.as_mut_ptr().add(x + 8).cast(), y_hi);
      x += 16;
    }

    if x < width {
      let tail_packed = &packed[x * 2..width * 2];
      let tail_out = &mut luma_out[x..width];
      let tail_w = width - x;
      scalar::y216_to_luma_u16_row(tail_packed, tail_out, tail_w);
    }
  }
}
