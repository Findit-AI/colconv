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
//! [`crate::row::scalar::yuv_420_to_bgr_row_scalar`]. All Q15 multiplies
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
//!    interleave as packed BGR via three `u8x16_swizzle` calls.

use core::arch::wasm32::{
  i8x16, i8x16_shuffle, i16x8_add_sat, i16x8_narrow_i32x4, i16x8_splat, i16x8_sub, i32x4_add,
  i32x4_extend_high_i16x8, i32x4_extend_low_i16x8, i32x4_mul, i32x4_shr, i32x4_splat,
  u8x16_narrow_i16x8, u8x16_swizzle, u16x8_load_extend_u8x8, v128, v128_load, v128_or, v128_store,
};

use crate::{ColorMatrix, row::scalar};

/// WASM simd128 YUV 4:2:0 → packed BGR. Semantics match
/// [`scalar::yuv_420_to_bgr_row_scalar`] byte‑identically.
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
/// 6. `bgr_out.len() >= 3 * width`.
///
/// Bounds are verified by `debug_assert` in debug builds; release
/// builds trust the caller because the kernel relies on unchecked
/// pointer arithmetic (`v128_load`, `u16x8_load_extend_u8x8`,
/// `v128_store`).
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn yuv_420_to_bgr_row_wasm_simd128(
  y: &[u8],
  u_half: &[u8],
  v_half: &[u8],
  bgr_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0, "YUV 4:2:0 requires even width");
  debug_assert!(y.len() >= width);
  debug_assert!(u_half.len() >= width / 2);
  debug_assert!(v_half.len() >= width / 2);
  debug_assert!(bgr_out.len() >= width * 3);

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

      // 3‑way interleave → packed BGR (48 bytes).
      write_bgr_16(b_u8, g_u8, r_u8, bgr_out.as_mut_ptr().add(x * 3));

      x += 16;
    }

    // Scalar tail for the 0..14 leftover pixels.
    if x < width {
      scalar::yuv_420_to_bgr_row_scalar(
        &y[x..width],
        &u_half[x / 2..width / 2],
        &v_half[x / 2..width / 2],
        &mut bgr_out[x * 3..width * 3],
        width - x,
        matrix,
        full_range,
      );
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

/// Writes 16 pixels of packed BGR (48 bytes) from three u8x16 channel
/// vectors, using the SSSE3‑style 3‑way interleave pattern. `u8x16_swizzle`
/// treats indices ≥ 16 as "zero the lane" — same semantics as
/// `_mm_shuffle_epi8`, so the same shuffle masks apply.
///
/// # Safety
///
/// `ptr` must point to at least 48 writable bytes.
#[inline(always)]
unsafe fn write_bgr_16(b: v128, g: v128, r: v128, ptr: *mut u8) {
  unsafe {
    // Block 0 (bytes 0..16): [B0,G0,R0, B1,G1,R1, ..., B5].
    // `-1` as i8 is 0xFF ≥ 16 → zeroes that output lane.
    let b0 = i8x16(0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1, -1, 5);
    let g0 = i8x16(-1, 0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1, -1);
    let r0 = i8x16(-1, -1, 0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1);
    let out0 = v128_or(
      v128_or(u8x16_swizzle(b, b0), u8x16_swizzle(g, g0)),
      u8x16_swizzle(r, r0),
    );

    // Block 1 (bytes 16..32): [G5,R5, B6,G6,R6, ..., G10].
    let b1 = i8x16(-1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1, 10, -1);
    let g1 = i8x16(5, -1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1, 10);
    let r1 = i8x16(-1, 5, -1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1);
    let out1 = v128_or(
      v128_or(u8x16_swizzle(b, b1), u8x16_swizzle(g, g1)),
      u8x16_swizzle(r, r1),
    );

    // Block 2 (bytes 32..48): [R10, B11,G11,R11, ..., R15].
    let b2 = i8x16(
      -1, 11, -1, -1, 12, -1, -1, 13, -1, -1, 14, -1, -1, 15, -1, -1,
    );
    let g2 = i8x16(
      -1, -1, 11, -1, -1, 12, -1, -1, 13, -1, -1, 14, -1, -1, 15, -1,
    );
    let r2 = i8x16(
      10, -1, -1, 11, -1, -1, 12, -1, -1, 13, -1, -1, 14, -1, -1, 15,
    );
    let out2 = v128_or(
      v128_or(u8x16_swizzle(b, b2), u8x16_swizzle(g, g2)),
      u8x16_swizzle(r, r2),
    );

    v128_store(ptr.cast(), out0);
    v128_store(ptr.add(16).cast(), out1);
    v128_store(ptr.add(32).cast(), out2);
  }
}

#[cfg(all(test, target_feature = "simd128"))]
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
    let mut bgr_scalar = std::vec![0u8; width * 3];
    let mut bgr_wasm = std::vec![0u8; width * 3];

    scalar::yuv_420_to_bgr_row_scalar(&y, &u, &v, &mut bgr_scalar, width, matrix, full_range);
    unsafe {
      yuv_420_to_bgr_row_wasm_simd128(&y, &u, &v, &mut bgr_wasm, width, matrix, full_range);
    }

    assert_eq!(bgr_scalar, bgr_wasm, "simd128 diverges from scalar");
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
}
