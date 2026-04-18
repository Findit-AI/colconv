//! aarch64 NEON backend for the row primitives.
//!
//! NEON is mandatory baseline on aarch64 in Rust, so no runtime
//! feature detection is needed — the dispatcher in [`crate::row`]
//! selects this backend unconditionally when `target_arch = "aarch64"`.
//!
//! # Numerical contract
//!
//! The kernel uses i32 widening multiplies and the same
//! `(prod + (1 << 14)) >> 15` Q15 rounding as
//! [`crate::row::scalar::yuv_420_to_bgr_row_scalar`], so output is
//! **byte‑identical** to the scalar reference for every input. This is
//! asserted by the equivalence tests below.
//!
//! # Pipeline (per 16 Y pixels / 8 chroma samples)
//!
//! 1. Load 16 Y (`vld1q_u8`) + 8 U (`vld1_u8`) + 8 V (`vld1_u8`).
//! 2. Widen U/V to i16, subtract 128 → `u_i16`, `v_i16`.
//! 3. Widen to i32 and apply `c_scale` (Q15) → `u_d`, `v_d` (i32x4 × 2).
//! 4. Per channel C ∈ {R, G, B}:
//!    `C_chroma = (C_u * u_d + C_v * v_d + RND) >> 15` in i32,
//!    narrow‑saturate to i16x8 (8 lanes = 8 chroma pairs).
//! 5. Duplicate each chroma lane into its Y‑pair slot with
//!    `vzip1q_s16` / `vzip2q_s16` → 16 i16 chroma lanes matching the
//!    16 Y lanes (nearest‑neighbor upsample in registers, no memory
//!    traffic).
//! 6. Y path: `(Y - y_off) * y_scale + RND >> 15` in i32, narrow to i16.
//! 7. Saturating add Y + chroma per channel → i16x16.
//! 8. Saturate‑narrow to u8x16 and interleave with `vst3q_u8`.

use core::arch::aarch64::{
  int16x8_t, int32x4_t, uint8x16x3_t, vaddq_s32, vcombine_s16, vcombine_u8, vdupq_n_s16,
  vdupq_n_s32, vget_high_s16, vget_high_u8, vget_low_s16, vget_low_u8, vld1_u8, vld1q_u8,
  vmovl_s16, vmovl_u8, vmulq_s32, vqaddq_s16, vqmovn_s32, vqmovun_s16, vreinterpretq_s16_u16,
  vshrq_n_s32, vst3q_u8, vsubq_s16, vzip1q_s16, vzip2q_s16,
};

use crate::{ColorMatrix, row::scalar};

/// NEON YUV 4:2:0 → packed BGR. Semantics match
/// [`scalar::yuv_420_to_bgr_row_scalar`] byte‑identically.
///
/// # Safety
///
/// The caller must uphold **all** of the following. Violating any
/// causes undefined behavior:
///
/// 1. **NEON must be available on the current CPU.** The dispatcher
///    in [`crate::row`] verifies this with
///    `is_aarch64_feature_detected!("neon")` (runtime) or
///    `cfg!(target_feature = "neon")` (compile‑time, no‑std). If you
///    call this kernel directly, you are responsible for the check —
///    executing NEON instructions on a CPU without NEON traps.
/// 2. `width & 1 == 0` (4:2:0 requires even width).
/// 3. `y.len() >= width`.
/// 4. `u_half.len() >= width / 2`.
/// 5. `v_half.len() >= width / 2`.
/// 6. `bgr_out.len() >= 3 * width`.
///
/// Bounds are verified by `debug_assert` in debug builds; release
/// builds trust the caller because the kernel relies on unchecked
/// pointer arithmetic (`vld1q_u8`, `vld1_u8`, `vst3q_u8`).
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn yuv_420_to_bgr_row_neon(
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

  // SAFETY: NEON is mandatory baseline on aarch64 (no feature
  // detection needed). All pointer adds below are bounded by the
  // `while x + 16 <= width` loop condition and the caller‑promised
  // slice lengths checked above.
  unsafe {
    let rnd_v = vdupq_n_s32(RND);
    let y_off_v = vdupq_n_s16(y_off as i16);
    let y_scale_v = vdupq_n_s32(y_scale);
    let c_scale_v = vdupq_n_s32(c_scale);
    let mid128 = vdupq_n_s16(128);
    let cru = vdupq_n_s32(coeffs.r_u());
    let crv = vdupq_n_s32(coeffs.r_v());
    let cgu = vdupq_n_s32(coeffs.g_u());
    let cgv = vdupq_n_s32(coeffs.g_v());
    let cbu = vdupq_n_s32(coeffs.b_u());
    let cbv = vdupq_n_s32(coeffs.b_v());

    let mut x = 0usize;
    while x + 16 <= width {
      let y_vec = vld1q_u8(y.as_ptr().add(x));
      let u_vec = vld1_u8(u_half.as_ptr().add(x / 2));
      let v_vec = vld1_u8(v_half.as_ptr().add(x / 2));

      // Widen Y halves to i16x8 (unsigned → signed, Y ≤ 255 fits).
      let y_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(y_vec)));
      let y_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(y_vec)));

      // Widen U, V to i16x8 and subtract 128.
      let u_i16 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(u_vec)), mid128);
      let v_i16 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(v_vec)), mid128);

      // Split to i32x4 halves so the Q15 multiplies don't overflow.
      let u_lo_i32 = vmovl_s16(vget_low_s16(u_i16));
      let u_hi_i32 = vmovl_s16(vget_high_s16(u_i16));
      let v_lo_i32 = vmovl_s16(vget_low_s16(v_i16));
      let v_hi_i32 = vmovl_s16(vget_high_s16(v_i16));

      // u_d = (u * c_scale + RND) >> 15, bit‑exact to scalar.
      let u_d_lo = q15_shift(vaddq_s32(vmulq_s32(u_lo_i32, c_scale_v), rnd_v));
      let u_d_hi = q15_shift(vaddq_s32(vmulq_s32(u_hi_i32, c_scale_v), rnd_v));
      let v_d_lo = q15_shift(vaddq_s32(vmulq_s32(v_lo_i32, c_scale_v), rnd_v));
      let v_d_hi = q15_shift(vaddq_s32(vmulq_s32(v_hi_i32, c_scale_v), rnd_v));

      // Per‑channel chroma contribution, narrow to i16 for later adds.
      let r_chroma = chroma_i16x8(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let g_chroma = chroma_i16x8(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let b_chroma = chroma_i16x8(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);

      // Nearest‑neighbor upsample: duplicate each of the 8 chroma
      // lanes into an adjacent pair to cover 16 Y lanes. vzip1q takes
      // lanes 0..3 from both operands interleaved → [c0,c0,c1,c1,...];
      // vzip2q does the same for lanes 4..7.
      let r_dup_lo = vzip1q_s16(r_chroma, r_chroma);
      let r_dup_hi = vzip2q_s16(r_chroma, r_chroma);
      let g_dup_lo = vzip1q_s16(g_chroma, g_chroma);
      let g_dup_hi = vzip2q_s16(g_chroma, g_chroma);
      let b_dup_lo = vzip1q_s16(b_chroma, b_chroma);
      let b_dup_hi = vzip2q_s16(b_chroma, b_chroma);

      // Y path → i16x8 (two vectors covering 16 pixels).
      let y_scaled_lo = scale_y(y_lo, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_hi, y_off_v, y_scale_v, rnd_v);

      // B, G, R = saturating_add(Y, chroma); saturate‑narrow to u8.
      let b_u8 = vcombine_u8(
        vqmovun_s16(vqaddq_s16(y_scaled_lo, b_dup_lo)),
        vqmovun_s16(vqaddq_s16(y_scaled_hi, b_dup_hi)),
      );
      let g_u8 = vcombine_u8(
        vqmovun_s16(vqaddq_s16(y_scaled_lo, g_dup_lo)),
        vqmovun_s16(vqaddq_s16(y_scaled_hi, g_dup_hi)),
      );
      let r_u8 = vcombine_u8(
        vqmovun_s16(vqaddq_s16(y_scaled_lo, r_dup_lo)),
        vqmovun_s16(vqaddq_s16(y_scaled_hi, r_dup_hi)),
      );

      // vst3q_u8 writes 48 bytes as interleaved B, G, R triples.
      let bgr = uint8x16x3_t(b_u8, g_u8, r_u8);
      vst3q_u8(bgr_out.as_mut_ptr().add(x * 3), bgr);

      x += 16;
    }

    // Scalar tail for the 0..14 leftover pixels (always even, 4:2:0
    // requires even width so x/2 and width/2 are well‑defined).
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

// The helpers below wrap NEON register‑only intrinsics (shifts, adds,
// multiplies, narrowing conversions, lane movers). None of them touch
// memory or take pointers, so there is no safety invariant to hoist to
// the caller — the functions themselves are safe. The `unsafe { ... }`
// blocks inside are only required because `core::arch::aarch64`
// intrinsics are marked `unsafe fn` in the standard library.
//
// `#[inline(always)]` guarantees these are inlined into the NEON‑
// enabled caller (`yuv_420_to_bgr_row_neon` has
// `#[target_feature(enable = "neon")]`), so the intrinsics execute in
// a context where NEON is explicitly enabled — not just implicitly
// via the aarch64 target's default feature set.

/// `>>_a 15` shift (arithmetic, sign‑extending).
#[inline(always)]
fn q15_shift(v: int32x4_t) -> int32x4_t {
  unsafe { vshrq_n_s32::<15>(v) }
}

/// Build an i16x8 channel chroma vector from the 8 paired i32 chroma
/// samples. Mirrors the scalar
/// `(coeff_u * u_d + coeff_v * v_d + RND) >> 15`.
#[inline(always)]
fn chroma_i16x8(
  cu: int32x4_t,
  cv: int32x4_t,
  u_d_lo: int32x4_t,
  v_d_lo: int32x4_t,
  u_d_hi: int32x4_t,
  v_d_hi: int32x4_t,
  rnd: int32x4_t,
) -> int16x8_t {
  unsafe {
    let lo = vshrq_n_s32::<15>(vaddq_s32(
      vaddq_s32(vmulq_s32(cu, u_d_lo), vmulq_s32(cv, v_d_lo)),
      rnd,
    ));
    let hi = vshrq_n_s32::<15>(vaddq_s32(
      vaddq_s32(vmulq_s32(cu, u_d_hi), vmulq_s32(cv, v_d_hi)),
      rnd,
    ));
    vcombine_s16(vqmovn_s32(lo), vqmovn_s32(hi))
  }
}

/// `(Y - y_off) * y_scale + RND >> 15` returned as i16x8 (8 Y pixels).
#[inline(always)]
fn scale_y(
  y_i16: int16x8_t,
  y_off_v: int16x8_t,
  y_scale_v: int32x4_t,
  rnd: int32x4_t,
) -> int16x8_t {
  unsafe {
    let shifted = vsubq_s16(y_i16, y_off_v);
    let lo = vshrq_n_s32::<15>(vaddq_s32(
      vmulq_s32(vmovl_s16(vget_low_s16(shifted)), y_scale_v),
      rnd,
    ));
    let hi = vshrq_n_s32::<15>(vaddq_s32(
      vmulq_s32(vmovl_s16(vget_high_s16(shifted)), y_scale_v),
      rnd,
    ));
    vcombine_s16(vqmovn_s32(lo), vqmovn_s32(hi))
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  /// Deterministic scalar‑equivalence fixture. Fills Y/U/V with a
  /// hash‑like sequence so every byte varies, then compares byte‑exact.
  fn check_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
    let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
    let u: std::vec::Vec<u8> = (0..width / 2)
      .map(|i| ((i * 53 + 23) & 0xFF) as u8)
      .collect();
    let v: std::vec::Vec<u8> = (0..width / 2)
      .map(|i| ((i * 71 + 91) & 0xFF) as u8)
      .collect();
    let mut bgr_scalar = std::vec![0u8; width * 3];
    let mut bgr_neon = std::vec![0u8; width * 3];

    scalar::yuv_420_to_bgr_row_scalar(&y, &u, &v, &mut bgr_scalar, width, matrix, full_range);
    unsafe {
      yuv_420_to_bgr_row_neon(&y, &u, &v, &mut bgr_neon, width, matrix, full_range);
    }

    if bgr_scalar != bgr_neon {
      let first_diff = bgr_scalar
        .iter()
        .zip(bgr_neon.iter())
        .position(|(a, b)| a != b)
        .unwrap();
      panic!(
        "NEON diverges from scalar at byte {first_diff} (width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} neon={}",
        bgr_scalar[first_diff], bgr_neon[first_diff]
      );
    }
  }

  #[test]
  fn neon_matches_scalar_all_matrices_16() {
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
  fn neon_matches_scalar_width_32() {
    check_equivalence(32, ColorMatrix::Bt601, true);
    check_equivalence(32, ColorMatrix::Bt709, false);
    check_equivalence(32, ColorMatrix::YCgCo, true);
  }

  #[test]
  fn neon_matches_scalar_width_1920() {
    check_equivalence(1920, ColorMatrix::Bt709, false);
  }

  #[test]
  fn neon_matches_scalar_odd_tail_widths() {
    // Widths that leave a non‑trivial scalar tail (non‑multiple of 16).
    for w in [18usize, 30, 34, 1922] {
      check_equivalence(w, ColorMatrix::Bt601, false);
    }
  }
}
