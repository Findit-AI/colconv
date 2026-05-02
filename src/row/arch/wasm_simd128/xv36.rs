//! wasm-simd128 XV36 (packed YUV 4:4:4, 12-bit) kernels.
//!
//! ## Layout
//!
//! Four `u16` per pixel: `[U(16), Y(16), V(16), A(16)]`, each holding
//! a 12-bit sample MSB-aligned in the high 12 bits (low 4 bits zero).
//! The `X` prefix denotes the A slot as **padding** — it is loaded but
//! discarded. RGBA outputs force α = max (`0xFF` u8 / `0x0FFF` u16).
//!
//! ## Per-iter pipeline (8 px / iter)
//!
//! One iteration processes 8 pixels = 32 u16 = 64 bytes = 4 × `v128_load`.
//! `u8x16_swizzle` permutes within each register to isolate channels;
//! `i8x16_shuffle` (valid indices 0–31 only) concatenates halves across
//! registers into full 8-lane channel vectors.
//!
//! Layout in memory (little-endian, 8 bytes per pixel):
//! ```text
//!   byte 0,1 = U low,high   byte 2,3 = Y   byte 4,5 = V   byte 6,7 = A
//! ```
//!
//! Each `v128` holds 2 pixels (8 × u16).  After `u8x16_swizzle`:
//!
//! - U from pixels {n, n+1}: bytes [0,1,8,9]     → low 4 bytes of swizzled reg.
//! - Y from pixels {n, n+1}: bytes [2,3,10,11]   → low 4 bytes.
//! - V from pixels {n, n+1}: bytes [4,5,12,13]   → low 4 bytes.
//!
//! Two consecutive swizzled pairs (covering pixels 0–3 and 4–7) are then
//! concatenated by `i8x16_shuffle::<0..7, 16..23>` to give each 8-lane
//! channel vector.  `u16x8_shr(v, 4)` drops the 4 padding LSBs.
//!
//! The Q15 pipeline mirrors `v410.rs`: `chroma_i16x8` (i32 chroma),
//! `scale_y` (Y ≤ 4095 fits i16), `clamp_u16_max_wasm` for u16 output.
//! 4:4:4 — no chroma duplication.
//!
//! ## Tail
//!
//! `width % 8` remaining pixels fall through to `scalar::xv36_*`.

use core::arch::wasm32::*;

use super::*;
use crate::{ColorMatrix, row::scalar};

/// Deinterleave 8 XV36 pixels (4 × v128 = 64 bytes) into separate
/// i16x8 channel vectors (U, Y, V).
///
/// Strategy: `u8x16_swizzle` isolates two samples of each channel from
/// each v128 into the low 4 bytes (high 12 bytes zeroed via 0xFF index).
/// Pairs of swizzled results (pixels 0–3 and 4–7) are then concatenated
/// by `i8x16_shuffle` (only valid indices 0–31) to form full 8-lane
/// vectors.
///
/// Returns `(u_raw, y_raw, v_raw)` — u16x8 with MSB-aligned values;
/// caller must `u16x8_shr(v, 4)` to drop the 4 padding LSBs.
#[inline]
#[target_feature(enable = "simd128")]
unsafe fn deinterleave_xv36_8px(ptr: *const u16) -> (v128, v128, v128) {
  unsafe {
    // Load 4 × v128, each covering 2 pixels.
    let raw0 = v128_load(ptr.cast()); // [U0,Y0,V0,A0,  U1,Y1,V1,A1]
    let raw1 = v128_load(ptr.add(8).cast()); // [U2,Y2,V2,A2,  U3,Y3,V3,A3]
    let raw2 = v128_load(ptr.add(16).cast()); // [U4,Y4,V4,A4,  U5,Y5,V5,A5]
    let raw3 = v128_load(ptr.add(24).cast()); // [U6,Y6,V6,A6,  U7,Y7,V7,A7]

    // Per-channel byte positions inside a 2-pixel v128:
    //   U → bytes 0,1 (pixel n) and 8,9 (pixel n+1)
    //   Y → bytes 2,3 and 10,11
    //   V → bytes 4,5 and 12,13
    //
    // `u8x16_swizzle` with index ≥ 16 zeroes the output byte (SSSE3
    // _mm_shuffle_epi8 semantics).  We pack the 2 u16 samples from each
    // channel into the *low* 4 bytes so downstream `i8x16_shuffle` can
    // concatenate two such results with indices 0..7 + 16..23.

    // U: bytes [0,1, 8,9] → low 4 bytes; high 12 bytes = 0.
    let u_idx = i8x16(0, 1, 8, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    // Y: bytes [2,3, 10,11] → low 4 bytes.
    let y_idx = i8x16(2, 3, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    // V: bytes [4,5, 12,13] → low 4 bytes.
    let v_idx = i8x16(4, 5, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);

    // Apply swizzle to each of the 4 raw registers.
    // u8x16_swizzle interprets each index byte as u8; -1_i8 = 0xFF ≥ 16 → zeroes lane.
    let u0 = u8x16_swizzle(raw0, u_idx); // [U0 lo,hi, U1 lo,hi, 0..12]
    let u1 = u8x16_swizzle(raw1, u_idx); // [U2, U3, 0..12]
    let u2 = u8x16_swizzle(raw2, u_idx); // [U4, U5, 0..12]
    let u3 = u8x16_swizzle(raw3, u_idx); // [U6, U7, 0..12]

    let y0 = u8x16_swizzle(raw0, y_idx);
    let y1 = u8x16_swizzle(raw1, y_idx);
    let y2 = u8x16_swizzle(raw2, y_idx);
    let y3 = u8x16_swizzle(raw3, y_idx);

    let v0 = u8x16_swizzle(raw0, v_idx);
    let v1 = u8x16_swizzle(raw1, v_idx);
    let v2 = u8x16_swizzle(raw2, v_idx);
    let v3 = u8x16_swizzle(raw3, v_idx);

    // Concatenate pairs: low 4 bytes of a + low 4 bytes of b → 8 bytes.
    // `i8x16_shuffle::<0,1,2,3, 16,17,18,19, ...>` picks bytes 0-3 from the
    // first operand and bytes 0-3 from the second operand (indices 16-19).
    // We need 8 bytes total, so we pack them into the low 8 bytes and zero
    // the high 8 using the all-zero identity trick: combine two such 4-byte
    // fragments from a pair into one 8-byte fragment, then combine two
    // fragments into the full 8-lane vector.

    // Combine pixels {0,1} and {2,3} for U → bytes 0..7 hold U0,U1,U2,U3.
    let u01 =
      i8x16_shuffle::<0, 1, 2, 3, 16, 17, 18, 19, 0, 0, 0, 0, 0, 0, 0, 0>(u0, u1);
    // Combine pixels {4,5} and {6,7} for U → bytes 0..7 hold U4,U5,U6,U7.
    let u23 =
      i8x16_shuffle::<0, 1, 2, 3, 16, 17, 18, 19, 0, 0, 0, 0, 0, 0, 0, 0>(u2, u3);
    // Full U vector: u16x8 [U0..U7] — low 8 bytes from u01, high 8 from u23.
    let u_raw =
      i8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23>(u01, u23);

    let y01 =
      i8x16_shuffle::<0, 1, 2, 3, 16, 17, 18, 19, 0, 0, 0, 0, 0, 0, 0, 0>(y0, y1);
    let y23 =
      i8x16_shuffle::<0, 1, 2, 3, 16, 17, 18, 19, 0, 0, 0, 0, 0, 0, 0, 0>(y2, y3);
    let y_raw =
      i8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23>(y01, y23);

    let v01 =
      i8x16_shuffle::<0, 1, 2, 3, 16, 17, 18, 19, 0, 0, 0, 0, 0, 0, 0, 0>(v0, v1);
    let v23 =
      i8x16_shuffle::<0, 1, 2, 3, 16, 17, 18, 19, 0, 0, 0, 0, 0, 0, 0, 0>(v2, v3);
    let v_raw =
      i8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23>(v01, v23);

    (u_raw, y_raw, v_raw)
  }
}

// ---- u8 RGB / RGBA output -----------------------------------------------

/// wasm-simd128 XV36 → packed u8 RGB or RGBA. 8 pixels per iteration.
///
/// Byte-identical to `scalar::xv36_to_rgb_or_rgba_row::<ALPHA>`.
///
/// # Safety
///
/// 1. **`simd128` must be enabled at compile time.**
/// 2. `packed.len() >= width * 4`.
/// 3. `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn xv36_to_rgb_or_rgba_row<const ALPHA: bool>(
  packed: &[u16],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(packed.len() >= width * 4, "packed row too short");
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp, "out row too short");

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<12, 8>(full_range);
  let bias = scalar::chroma_bias::<12>();
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

    let mut x = 0usize;
    while x + 8 <= width {
      // Deinterleave 8 XV36 pixels (64 bytes) into U/Y/V channels.
      let (u_raw, y_raw, v_raw) = deinterleave_xv36_8px(packed.as_ptr().add(x * 4));

      // Right-shift by 4 to drop the 4 padding LSBs → 12-bit [0, 4095].
      // Values ≤ 4095 fit safely in i16.
      let u_i16 = u16x8_shr(u_raw, 4);
      let y_i16 = u16x8_shr(y_raw, 4);
      let v_i16 = u16x8_shr(v_raw, 4);

      // Subtract chroma bias (2048 for 12-bit).
      let u_sub = i16x8_sub(u_i16, bias_v);
      let v_sub = i16x8_sub(v_i16, bias_v);

      // Widen to i32x4 lo/hi for Q15 multiply.
      let u_lo_i32 = i32x4_extend_low_i16x8(u_sub);
      let u_hi_i32 = i32x4_extend_high_i16x8(u_sub);
      let v_lo_i32 = i32x4_extend_low_i16x8(v_sub);
      let v_hi_i32 = i32x4_extend_high_i16x8(v_sub);

      let u_d_lo = q15_shift(i32x4_add(i32x4_mul(u_lo_i32, c_scale_v), rnd_v));
      let u_d_hi = q15_shift(i32x4_add(i32x4_mul(u_hi_i32, c_scale_v), rnd_v));
      let v_d_lo = q15_shift(i32x4_add(i32x4_mul(v_lo_i32, c_scale_v), rnd_v));
      let v_d_hi = q15_shift(i32x4_add(i32x4_mul(v_hi_i32, c_scale_v), rnd_v));

      // 4:4:4 — no chroma duplication; all 8 lanes carry unique U/V.
      let r_chroma = chroma_i16x8(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let g_chroma = chroma_i16x8(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let b_chroma = chroma_i16x8(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);

      // Y values ≤ 4095 fit in i16; use scale_y (NOT scale_y_u16_wasm).
      let y_scaled = scale_y(y_i16, y_off_v, y_scale_v, rnd_v);

      // Saturate-add and narrow to u8.
      let r_sum = i16x8_add_sat(y_scaled, r_chroma);
      let g_sum = i16x8_add_sat(y_scaled, g_chroma);
      let b_sum = i16x8_add_sat(y_scaled, b_chroma);
      let r_u8 = u8x16_narrow_i16x8(r_sum, r_sum);
      let g_u8 = u8x16_narrow_i16x8(g_sum, g_sum);
      let b_u8 = u8x16_narrow_i16x8(b_sum, b_sum);

      // 8-pixel store via stack buffer (low 8 bytes of each narrow result).
      let mut r_tmp = [0u8; 16];
      let mut g_tmp = [0u8; 16];
      let mut b_tmp = [0u8; 16];
      v128_store(r_tmp.as_mut_ptr().cast(), r_u8);
      v128_store(g_tmp.as_mut_ptr().cast(), g_u8);
      v128_store(b_tmp.as_mut_ptr().cast(), b_u8);

      if ALPHA {
        let dst = &mut out[x * 4..x * 4 + 8 * 4];
        for i in 0..8 {
          dst[i * 4] = r_tmp[i];
          dst[i * 4 + 1] = g_tmp[i];
          dst[i * 4 + 2] = b_tmp[i];
          dst[i * 4 + 3] = 0xFF;
        }
      } else {
        let dst = &mut out[x * 3..x * 3 + 8 * 3];
        for i in 0..8 {
          dst[i * 3] = r_tmp[i];
          dst[i * 3 + 1] = g_tmp[i];
          dst[i * 3 + 2] = b_tmp[i];
        }
      }

      x += 8;
    }

    // Scalar tail — remaining < 8 pixels.
    if x < width {
      let tail_packed = &packed[x * 4..width * 4];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      scalar::xv36_to_rgb_or_rgba_row::<ALPHA>(tail_packed, tail_out, tail_w, matrix, full_range);
    }
  }
}

// ---- u16 RGB / RGBA native-depth output ---------------------------------

/// wasm-simd128 XV36 → packed native-depth u16 RGB or RGBA
/// (low-bit-packed at 12-bit). 8 pixels per iteration.
///
/// Byte-identical to
/// `scalar::xv36_to_rgb_u16_or_rgba_u16_row::<ALPHA>`.
///
/// # Safety
///
/// 1. **`simd128` must be enabled at compile time.**
/// 2. `packed.len() >= width * 4`.
/// 3. `out.len() >= width * (if ALPHA { 4 } else { 3 })` (u16 elements).
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn xv36_to_rgb_u16_or_rgba_u16_row<const ALPHA: bool>(
  packed: &[u16],
  out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(packed.len() >= width * 4, "packed row too short");
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp, "out row too short");

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<12, 12>(full_range);
  let bias = scalar::chroma_bias::<12>();
  const RND: i32 = 1 << 14;
  let out_max: i16 = 0x0FFF;
  let alpha_u16: u16 = 0x0FFF;

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

    let mut x = 0usize;
    while x + 8 <= width {
      let (u_raw, y_raw, v_raw) = deinterleave_xv36_8px(packed.as_ptr().add(x * 4));

      let u_i16 = u16x8_shr(u_raw, 4);
      let y_i16 = u16x8_shr(y_raw, 4);
      let v_i16 = u16x8_shr(v_raw, 4);

      let u_sub = i16x8_sub(u_i16, bias_v);
      let v_sub = i16x8_sub(v_i16, bias_v);

      let u_lo_i32 = i32x4_extend_low_i16x8(u_sub);
      let u_hi_i32 = i32x4_extend_high_i16x8(u_sub);
      let v_lo_i32 = i32x4_extend_low_i16x8(v_sub);
      let v_hi_i32 = i32x4_extend_high_i16x8(v_sub);

      let u_d_lo = q15_shift(i32x4_add(i32x4_mul(u_lo_i32, c_scale_v), rnd_v));
      let u_d_hi = q15_shift(i32x4_add(i32x4_mul(u_hi_i32, c_scale_v), rnd_v));
      let v_d_lo = q15_shift(i32x4_add(i32x4_mul(v_lo_i32, c_scale_v), rnd_v));
      let v_d_hi = q15_shift(i32x4_add(i32x4_mul(v_hi_i32, c_scale_v), rnd_v));

      let r_chroma = chroma_i16x8(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let g_chroma = chroma_i16x8(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let b_chroma = chroma_i16x8(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);

      let y_scaled = scale_y(y_i16, y_off_v, y_scale_v, rnd_v);

      // Clamp to [0, 0x0FFF] (12-bit low-bit-packed output range).
      let r = clamp_u16_max_wasm(i16x8_add_sat(y_scaled, r_chroma), zero_v, max_v);
      let g = clamp_u16_max_wasm(i16x8_add_sat(y_scaled, g_chroma), zero_v, max_v);
      let b = clamp_u16_max_wasm(i16x8_add_sat(y_scaled, b_chroma), zero_v, max_v);

      // 8-pixel u16 store via stack buffer.
      let mut r_tmp = [0u16; 8];
      let mut g_tmp = [0u16; 8];
      let mut b_tmp = [0u16; 8];
      v128_store(r_tmp.as_mut_ptr().cast(), r);
      v128_store(g_tmp.as_mut_ptr().cast(), g);
      v128_store(b_tmp.as_mut_ptr().cast(), b);

      if ALPHA {
        let dst = &mut out[x * 4..x * 4 + 8 * 4];
        for i in 0..8 {
          dst[i * 4] = r_tmp[i];
          dst[i * 4 + 1] = g_tmp[i];
          dst[i * 4 + 2] = b_tmp[i];
          dst[i * 4 + 3] = alpha_u16;
        }
      } else {
        let dst = &mut out[x * 3..x * 3 + 8 * 3];
        for i in 0..8 {
          dst[i * 3] = r_tmp[i];
          dst[i * 3 + 1] = g_tmp[i];
          dst[i * 3 + 2] = b_tmp[i];
        }
      }

      x += 8;
    }

    // Scalar tail.
    if x < width {
      let tail_packed = &packed[x * 4..width * 4];
      let tail_out = &mut out[x * bpp..width * bpp];
      let tail_w = width - x;
      scalar::xv36_to_rgb_u16_or_rgba_u16_row::<ALPHA>(
        tail_packed,
        tail_out,
        tail_w,
        matrix,
        full_range,
      );
    }
  }
}

// ---- Luma u8 (8 px/iter) -----------------------------------------------

/// wasm-simd128 XV36 → u8 luma. Y is quadruple element 1; `>> 8` drops
/// 4 padding LSBs + 4 more to give an 8-bit value.
///
/// Byte-identical to `scalar::xv36_to_luma_row`.
///
/// # Safety
///
/// 1. **`simd128` must be enabled at compile time.**
/// 2. `packed.len() >= width * 4`.
/// 3. `out.len() >= width`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn xv36_to_luma_row(packed: &[u16], out: &mut [u8], width: usize) {
  debug_assert!(packed.len() >= width * 4);
  debug_assert!(out.len() >= width);

  unsafe {
    // Y byte positions within a 2-pixel v128: bytes 2,3 (Y0) and 10,11 (Y1).
    // Pack both into the low 4 bytes; high 12 bytes zeroed via 0xFF index.
    let y_idx = i8x16(2, 3, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);

    let mut x = 0usize;
    while x + 8 <= width {
      let ptr = packed.as_ptr().add(x * 4);
      let raw0 = v128_load(ptr.cast()); // pixels 0,1
      let raw1 = v128_load(ptr.add(8).cast()); // pixels 2,3
      let raw2 = v128_load(ptr.add(16).cast()); // pixels 4,5
      let raw3 = v128_load(ptr.add(24).cast()); // pixels 6,7

      // Extract Y from each pair → 2 u16 in low 4 bytes.
      let y0 = u8x16_swizzle(raw0, y_idx); // [Y0,Y1, 0..12]
      let y1 = u8x16_swizzle(raw1, y_idx); // [Y2,Y3, 0..12]
      let y2 = u8x16_swizzle(raw2, y_idx); // [Y4,Y5, 0..12]
      let y3 = u8x16_swizzle(raw3, y_idx); // [Y6,Y7, 0..12]

      // Concatenate pairs: [bytes 0..3 of y0] + [bytes 0..3 of y1] → 8 bytes.
      let y01 =
        i8x16_shuffle::<0, 1, 2, 3, 16, 17, 18, 19, 0, 0, 0, 0, 0, 0, 0, 0>(y0, y1);
      let y23 =
        i8x16_shuffle::<0, 1, 2, 3, 16, 17, 18, 19, 0, 0, 0, 0, 0, 0, 0, 0>(y2, y3);
      // Full 8-lane Y vector.
      let y_vec =
        i8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23>(y01, y23);

      // >> 8: drop 4 padding LSBs + 4 more → 8-bit value.
      let y_shr = u16x8_shr(y_vec, 8);
      let y_u8 = u8x16_narrow_i16x8(y_shr, y_shr);
      let mut tmp = [0u8; 16];
      v128_store(tmp.as_mut_ptr().cast(), y_u8);
      out[x..x + 8].copy_from_slice(&tmp[..8]);

      x += 8;
    }

    // Scalar tail.
    if x < width {
      scalar::xv36_to_luma_row(&packed[x * 4..width * 4], &mut out[x..width], width - x);
    }
  }
}

// ---- Luma u16 (8 px/iter) -----------------------------------------------

/// wasm-simd128 XV36 → u16 luma (low-bit-packed at 12-bit). Y is
/// quadruple element 1; `>> 4` drops the 4 padding LSBs giving a
/// 12-bit value in `[0, 4095]`.
///
/// Byte-identical to `scalar::xv36_to_luma_u16_row`.
///
/// # Safety
///
/// 1. **`simd128` must be enabled at compile time.**
/// 2. `packed.len() >= width * 4`.
/// 3. `out.len() >= width`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn xv36_to_luma_u16_row(packed: &[u16], out: &mut [u16], width: usize) {
  debug_assert!(packed.len() >= width * 4);
  debug_assert!(out.len() >= width);

  unsafe {
    let y_idx = i8x16(2, 3, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);

    let mut x = 0usize;
    while x + 8 <= width {
      let ptr = packed.as_ptr().add(x * 4);
      let raw0 = v128_load(ptr.cast());
      let raw1 = v128_load(ptr.add(8).cast());
      let raw2 = v128_load(ptr.add(16).cast());
      let raw3 = v128_load(ptr.add(24).cast());

      let y0 = u8x16_swizzle(raw0, y_idx);
      let y1 = u8x16_swizzle(raw1, y_idx);
      let y2 = u8x16_swizzle(raw2, y_idx);
      let y3 = u8x16_swizzle(raw3, y_idx);

      let y01 =
        i8x16_shuffle::<0, 1, 2, 3, 16, 17, 18, 19, 0, 0, 0, 0, 0, 0, 0, 0>(y0, y1);
      let y23 =
        i8x16_shuffle::<0, 1, 2, 3, 16, 17, 18, 19, 0, 0, 0, 0, 0, 0, 0, 0>(y2, y3);
      let y_vec =
        i8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23>(y01, y23);

      // >> 4 to drop 4 padding LSBs → 12-bit value in low 12 bits.
      let y_low = u16x8_shr(y_vec, 4);
      v128_store(out.as_mut_ptr().add(x).cast(), y_low);
      x += 8;
    }

    // Scalar tail.
    if x < width {
      scalar::xv36_to_luma_u16_row(&packed[x * 4..width * 4], &mut out[x..width], width - x);
    }
  }
}
