//! aarch64 NEON backend for the row primitives.
//!
//! Selected by [`crate::row`]'s dispatcher after
//! `is_aarch64_feature_detected!("neon")` returns true (runtime,
//! std‑gated) or `cfg!(target_feature = "neon")` evaluates true
//! (compile‑time, no‑std). The kernel itself carries
//! `#[target_feature(enable = "neon")]` so its intrinsics execute in
//! an explicitly NEON‑enabled context rather than one merely inherited
//! from the aarch64 target's default feature set.
//!
//! # Numerical contract
//!
//! The kernel uses i32 widening multiplies and the same
//! `(prod + (1 << 14)) >> 15` Q15 rounding as
//! [`crate::row::scalar::yuv_420_to_rgb_row`], so output is
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
  float32x4_t, int16x8_t, int32x4_t, uint8x16_t, uint8x16x3_t, vaddq_f32, vaddq_s32, vbslq_f32,
  vceqq_f32, vcltq_f32, vcombine_s16, vcombine_u8, vcombine_u16, vcvtq_f32_u32, vcvtq_u32_f32,
  vdivq_f32, vdupq_n_f32, vdupq_n_s16, vdupq_n_s32, vget_high_s16, vget_high_u8, vget_high_u16,
  vget_low_s16, vget_low_u8, vget_low_u16, vld1_u8, vld1q_u8, vld2_u8, vld3q_u8, vmaxq_f32,
  vminq_f32, vmovl_s16, vmovl_u8, vmovl_u16, vmovn_u16, vmovn_u32, vmulq_f32, vmulq_s32, vmvnq_u32,
  vqaddq_s16, vqmovn_s32, vqmovun_s16, vreinterpretq_s16_u16, vshrq_n_s32, vst1q_u8, vst3q_u8,
  vsubq_f32, vsubq_s16, vzip1q_s16, vzip2q_s16,
};

use crate::{ColorMatrix, row::scalar};

/// NEON YUV 4:2:0 → packed RGB. Semantics match
/// [`scalar::yuv_420_to_rgb_row`] byte‑identically.
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
/// 6. `rgb_out.len() >= 3 * width`.
///
/// Bounds are verified by `debug_assert` in debug builds; release
/// builds trust the caller because the kernel relies on unchecked
/// pointer arithmetic (`vld1q_u8`, `vld1_u8`, `vst3q_u8`).
#[inline]
#[target_feature(enable = "neon")]
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

  // SAFETY: NEON availability is the caller's obligation per the
  // `# Safety` section above; the dispatcher in `crate::row` checks
  // it. All pointer adds below are bounded by the
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

      // vst3q_u8 writes 48 bytes as interleaved R, G, B triples.
      let rgb = uint8x16x3_t(r_u8, g_u8, b_u8);
      vst3q_u8(rgb_out.as_mut_ptr().add(x * 3), rgb);

      x += 16;
    }

    // Scalar tail for the 0..14 leftover pixels (always even, 4:2:0
    // requires even width so x/2 and width/2 are well‑defined).
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

/// NEON NV12 → packed RGB. Identical math to [`yuv_420_to_rgb_row`];
/// the only difference is UV ingestion — `vld2_u8` deinterleaves 16
/// interleaved UV bytes into u8x8 U and u8x8 V vectors in one
/// instruction, matching the shape expected by the rest of the
/// chroma→RGB pipeline.
///
/// # Safety
///
/// 1. **NEON must be available on the current CPU.** The dispatcher
///    verifies this; direct callers are responsible.
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`.
/// 4. `uv_half.len() >= width` (`2 * (width / 2)` interleaved bytes).
/// 5. `rgb_out.len() >= 3 * width`.
///
/// Bounds are `debug_assert`‑checked; release builds trust the caller
/// because the kernel uses unchecked pointer arithmetic (`vld1q_u8`,
/// `vld2_u8`, `vst3q_u8`).
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn nv12_to_rgb_row(
  y: &[u8],
  uv_half: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0, "NV12 requires even width");
  debug_assert!(y.len() >= width);
  debug_assert!(uv_half.len() >= width);
  debug_assert!(rgb_out.len() >= width * 3);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params(full_range);
  const RND: i32 = 1 << 14;

  // SAFETY: NEON availability is the caller's obligation; all pointer
  // adds below are bounded by the `while x + 16 <= width` loop
  // condition and the caller‑promised slice lengths above.
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
      // 16 Y pixels → 8 chroma pairs. `vld2_u8` loads 16 UV bytes
      // starting at offset `x` (= `x / 2 * 2`) and deinterleaves them
      // into (u8x8 U, u8x8 V) — the shape the rest of the pipeline
      // expects.
      let uv_pair = vld2_u8(uv_half.as_ptr().add(x));
      let u_vec = uv_pair.0;
      let v_vec = uv_pair.1;

      let y_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(y_vec)));
      let y_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(y_vec)));

      let u_i16 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(u_vec)), mid128);
      let v_i16 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(v_vec)), mid128);

      let u_lo_i32 = vmovl_s16(vget_low_s16(u_i16));
      let u_hi_i32 = vmovl_s16(vget_high_s16(u_i16));
      let v_lo_i32 = vmovl_s16(vget_low_s16(v_i16));
      let v_hi_i32 = vmovl_s16(vget_high_s16(v_i16));

      let u_d_lo = q15_shift(vaddq_s32(vmulq_s32(u_lo_i32, c_scale_v), rnd_v));
      let u_d_hi = q15_shift(vaddq_s32(vmulq_s32(u_hi_i32, c_scale_v), rnd_v));
      let v_d_lo = q15_shift(vaddq_s32(vmulq_s32(v_lo_i32, c_scale_v), rnd_v));
      let v_d_hi = q15_shift(vaddq_s32(vmulq_s32(v_hi_i32, c_scale_v), rnd_v));

      let r_chroma = chroma_i16x8(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let g_chroma = chroma_i16x8(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let b_chroma = chroma_i16x8(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);

      let r_dup_lo = vzip1q_s16(r_chroma, r_chroma);
      let r_dup_hi = vzip2q_s16(r_chroma, r_chroma);
      let g_dup_lo = vzip1q_s16(g_chroma, g_chroma);
      let g_dup_hi = vzip2q_s16(g_chroma, g_chroma);
      let b_dup_lo = vzip1q_s16(b_chroma, b_chroma);
      let b_dup_hi = vzip2q_s16(b_chroma, b_chroma);

      let y_scaled_lo = scale_y(y_lo, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_hi, y_off_v, y_scale_v, rnd_v);

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

      let rgb = uint8x16x3_t(r_u8, g_u8, b_u8);
      vst3q_u8(rgb_out.as_mut_ptr().add(x * 3), rgb);

      x += 16;
    }

    // Scalar NV12 tail. UV slice stride matches Y stride (`width` each,
    // with `x` already consumed from both).
    if x < width {
      scalar::nv12_to_rgb_row(
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

// The helpers below wrap NEON register‑only intrinsics (shifts, adds,
// multiplies, narrowing conversions, lane movers). None of them touch
// memory or take pointers, so there is no safety invariant to hoist to
// the caller — the functions themselves are safe. The `unsafe { ... }`
// blocks inside are only required because `core::arch::aarch64`
// intrinsics are marked `unsafe fn` in the standard library.
//
// `#[inline(always)]` guarantees these are inlined into the NEON‑
// enabled caller (`yuv_420_to_rgb_row` has
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

// ===== RGB → HSV =========================================================

/// NEON RGB → planar HSV. Semantics match
/// [`scalar::rgb_to_hsv_row`] byte‑identically.
///
/// # Safety
///
/// The caller must uphold **all** of the following. Violating any
/// causes undefined behavior:
///
/// 1. **NEON must be available on the current CPU** (same obligation
///    as `yuv_420_to_rgb_row`; the dispatcher checks this via
///    `is_aarch64_feature_detected!("neon")`).
/// 2. `rgb.len() >= 3 * width`.
/// 3. `h_out.len() >= width`.
/// 4. `s_out.len() >= width`.
/// 5. `v_out.len() >= width`.
///
/// Bounds are verified by `debug_assert` in debug builds. The kernel
/// relies on unchecked pointer arithmetic (`vld3q_u8`, `vst1q_u8`).
///
/// # Numerical contract
///
/// Bit‑identical to the scalar reference. Every scalar op has the
/// same SIMD counterpart in the same order: `vmaxq_f32` / `vminq_f32`
/// mirror `f32::max` / `f32::min`; `vdivq_f32` is true f32 division
/// (not reciprocal estimate); branch cascade uses `vbslq_f32` in the
/// same `delta == 0 → v == r → v == g → v == b` priority.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn rgb_to_hsv_row(
  rgb: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
) {
  debug_assert!(rgb.len() >= width * 3, "rgb row too short");
  debug_assert!(h_out.len() >= width, "H row too short");
  debug_assert!(s_out.len() >= width, "S row too short");
  debug_assert!(v_out.len() >= width, "V row too short");

  // SAFETY: NEON availability is the caller's obligation per the
  // `# Safety` section. All pointer adds below are bounded by the
  // `while x + 16 <= width` loop condition and the caller‑promised
  // slice lengths.
  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      // Deinterleave 16 RGB pixels → three u8x16 channel vectors.
      let rgb_vec = vld3q_u8(rgb.as_ptr().add(x * 3));
      let r_u8 = rgb_vec.0;
      let g_u8 = rgb_vec.1;
      let b_u8 = rgb_vec.2;

      // Widen each u8x16 to four f32x4 (16 values split into four
      // 4‑pixel groups) for the f32 HSV math.
      let (b0, b1, b2, b3) = u8x16_to_f32x4_quad(b_u8);
      let (g0, g1, g2, g3) = u8x16_to_f32x4_quad(g_u8);
      let (r0, r1, r2, r3) = u8x16_to_f32x4_quad(r_u8);

      // HSV per 4‑pixel group. Each returns (h_quant, s_quant, v_quant)
      // as f32x4 values already in [0, 179] / [0, 255] / [0, 255].
      let (h0, s0, v0) = hsv_group(b0, g0, r0);
      let (h1, s1, v1) = hsv_group(b1, g1, r1);
      let (h2, s2, v2) = hsv_group(b2, g2, r2);
      let (h3, s3, v3) = hsv_group(b3, g3, r3);

      // Truncate f32 → u8 via u32 intermediate, matching scalar `as u8`
      // (which saturates then truncates; values are pre‑clamped so the
      // narrow is safe).
      let h_u8 = f32x4_quad_to_u8x16(h0, h1, h2, h3);
      let s_u8 = f32x4_quad_to_u8x16(s0, s1, s2, s3);
      let v_u8 = f32x4_quad_to_u8x16(v0, v1, v2, v3);

      vst1q_u8(h_out.as_mut_ptr().add(x), h_u8);
      vst1q_u8(s_out.as_mut_ptr().add(x), s_u8);
      vst1q_u8(v_out.as_mut_ptr().add(x), v_u8);

      x += 16;
    }

    // Scalar tail for the 0..15 leftover pixels.
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

/// Widens a u8x16 to four f32x4 groups (covering lanes 0..3, 4..7,
/// 8..11, 12..15 respectively). Lanes are zero‑extended at each
/// widening step, so f32 values land exactly in `[0.0, 255.0]`.
#[inline(always)]
fn u8x16_to_f32x4_quad(v: uint8x16_t) -> (float32x4_t, float32x4_t, float32x4_t, float32x4_t) {
  unsafe {
    let u16_lo = vmovl_u8(vget_low_u8(v)); // u16x8 = lanes 0..7
    let u16_hi = vmovl_u8(vget_high_u8(v)); // u16x8 = lanes 8..15
    let u32_0 = vmovl_u16(vget_low_u16(u16_lo)); // lanes 0..3
    let u32_1 = vmovl_u16(vget_high_u16(u16_lo)); // lanes 4..7
    let u32_2 = vmovl_u16(vget_low_u16(u16_hi)); // lanes 8..11
    let u32_3 = vmovl_u16(vget_high_u16(u16_hi)); // lanes 12..15
    (
      vcvtq_f32_u32(u32_0),
      vcvtq_f32_u32(u32_1),
      vcvtq_f32_u32(u32_2),
      vcvtq_f32_u32(u32_3),
    )
  }
}

/// Computes HSV for 4 pixels. Mirrors the scalar `rgb_to_hsv_pixel`
/// op‑for‑op. Returns `(h_quant, s_quant, v_quant)` — each already
/// clamped to the scalar's output range (`h ≤ 179`, `s ≤ 255`,
/// `v ≤ 255`), still as f32 awaiting u8 conversion in the caller.
#[inline(always)]
fn hsv_group(
  b: float32x4_t,
  g: float32x4_t,
  r: float32x4_t,
) -> (float32x4_t, float32x4_t, float32x4_t) {
  unsafe {
    let zero = vdupq_n_f32(0.0);
    let half = vdupq_n_f32(0.5);
    let sixty = vdupq_n_f32(60.0);
    let one_twenty = vdupq_n_f32(120.0);
    let two_forty = vdupq_n_f32(240.0);
    let three_sixty = vdupq_n_f32(360.0);
    let one_seventy_nine = vdupq_n_f32(179.0);
    let two_fifty_five = vdupq_n_f32(255.0);

    // V = max(b, g, r); min = min(b, g, r); delta = V - min.
    // vmaxq_f32 / vminq_f32 are NaN‑tolerant, matching f32::max / f32::min.
    let v = vmaxq_f32(vmaxq_f32(b, g), r);
    let min_bgr = vminq_f32(vminq_f32(b, g), r);
    let delta = vsubq_f32(v, min_bgr);

    // S = if v == 0 { 0 } else { 255 * delta / v }.
    let mask_v_nonzero = vmvnq_u32(vceqq_f32(v, zero));
    let s_nonzero = vdivq_f32(vmulq_f32(two_fifty_five, delta), v);
    let s = vbslq_f32(mask_v_nonzero, s_nonzero, zero);

    // Hue — compute all three candidate formulas then select.
    let mask_delta_zero = vceqq_f32(delta, zero);
    let mask_v_is_r = vceqq_f32(v, r);
    let mask_v_is_g = vceqq_f32(v, g);

    // Branch 1 (v == r): 60 * (g - b) / delta, wrap negatives by +360.
    let h_r = {
      let raw = vdivq_f32(vmulq_f32(sixty, vsubq_f32(g, b)), delta);
      let mask_neg = vcltq_f32(raw, zero);
      vbslq_f32(mask_neg, vaddq_f32(raw, three_sixty), raw)
    };
    // Branch 2 (v == g): 60 * (b - r) / delta + 120.
    let h_g = vaddq_f32(
      vdivq_f32(vmulq_f32(sixty, vsubq_f32(b, r)), delta),
      one_twenty,
    );
    // Branch 3 (v == b, implicit): 60 * (r - g) / delta + 240.
    let h_b = vaddq_f32(
      vdivq_f32(vmulq_f32(sixty, vsubq_f32(r, g)), delta),
      two_forty,
    );

    // Cascade: if delta == 0 → 0; else if v == r → h_r; else if v == g
    // → h_g; else → h_b. Same priority order as the scalar.
    let hue_g_or_b = vbslq_f32(mask_v_is_g, h_g, h_b);
    let hue_nonzero_delta = vbslq_f32(mask_v_is_r, h_r, hue_g_or_b);
    let hue = vbslq_f32(mask_delta_zero, zero, hue_nonzero_delta);

    // Quantize to the scalar's output ranges. Scalar:
    //   h_quant = (hue * 0.5 + 0.5).clamp(0, 179)
    //   s_quant = (s + 0.5).clamp(0, 255)
    //   v_quant = (v + 0.5).clamp(0, 255)
    // clamp → vminq(vmaxq(v, lo), hi). Inputs are all finite so NaN
    // handling is irrelevant here.
    let h_quant = vminq_f32(
      vmaxq_f32(vaddq_f32(vmulq_f32(hue, half), half), zero),
      one_seventy_nine,
    );
    let s_quant = vminq_f32(vmaxq_f32(vaddq_f32(s, half), zero), two_fifty_five);
    let v_quant = vminq_f32(vmaxq_f32(vaddq_f32(v, half), zero), two_fifty_five);

    (h_quant, s_quant, v_quant)
  }
}

/// Converts four f32x4 vectors (16 values in [0, 255]) to one u8x16.
/// Truncates f32 → u32 via `vcvtq_u32_f32` (matches scalar `as u8`
/// which saturates‑then‑truncates; values are pre‑clamped so the
/// narrowing steps below are exact).
#[inline(always)]
fn f32x4_quad_to_u8x16(
  a: float32x4_t,
  b: float32x4_t,
  c: float32x4_t,
  d: float32x4_t,
) -> uint8x16_t {
  unsafe {
    let a_u32 = vcvtq_u32_f32(a);
    let b_u32 = vcvtq_u32_f32(b);
    let c_u32 = vcvtq_u32_f32(c);
    let d_u32 = vcvtq_u32_f32(d);
    let ab_u16 = vcombine_u16(vmovn_u32(a_u32), vmovn_u32(b_u32));
    let cd_u16 = vcombine_u16(vmovn_u32(c_u32), vmovn_u32(d_u32));
    vcombine_u8(vmovn_u16(ab_u16), vmovn_u16(cd_u16))
  }
}

// ===== BGR ↔ RGB byte swap ==============================================

/// Swaps the outer two channels of each packed 3‑byte triple. Drives
/// both `bgr_to_rgb_row` and `rgb_to_bgr_row` since the transformation
/// is self‑inverse.
///
/// NEON makes this almost free: `vld3q_u8` deinterleaves 16 pixels into
/// three channel vectors `(ch0, ch1, ch2)`, and `vst3q_u8` re‑interleaves
/// them — passing the deinterleaved vectors back in reversed order
/// `(ch2, ch1, ch0)` swaps the outer channels in a single store.
///
/// # Safety
///
/// 1. NEON must be available (same obligation as the other NEON kernels).
/// 2. `input.len() >= 3 * width`.
/// 3. `output.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn bgr_rgb_swap_row(input: &[u8], output: &mut [u8], width: usize) {
  debug_assert!(input.len() >= width * 3, "input row too short");
  debug_assert!(output.len() >= width * 3, "output row too short");

  // SAFETY: NEON availability is the caller's obligation per the
  // `# Safety` section. All pointer adds are bounded by the
  // `while x + 16 <= width` condition and the caller‑promised
  // slice lengths.
  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      let triple = vld3q_u8(input.as_ptr().add(x * 3));
      let swapped = uint8x16x3_t(triple.2, triple.1, triple.0);
      vst3q_u8(output.as_mut_ptr().add(x * 3), swapped);
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

#[cfg(all(test, feature = "std"))]
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
    let mut rgb_scalar = std::vec![0u8; width * 3];
    let mut rgb_neon = std::vec![0u8; width * 3];

    scalar::yuv_420_to_rgb_row(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
    unsafe {
      yuv_420_to_rgb_row(&y, &u, &v, &mut rgb_neon, width, matrix, full_range);
    }

    if rgb_scalar != rgb_neon {
      let first_diff = rgb_scalar
        .iter()
        .zip(rgb_neon.iter())
        .position(|(a, b)| a != b)
        .unwrap();
      panic!(
        "NEON diverges from scalar at byte {first_diff} (width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} neon={}",
        rgb_scalar[first_diff], rgb_neon[first_diff]
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

  // ---- nv12_to_rgb_row equivalence ------------------------------------

  /// Scalar‑equivalence fixture for NV12. Builds an interleaved UV row
  /// from the same U/V byte sequences used by the yuv420p fixture so a
  /// single NV12 call should produce byte‑identical output to the
  /// scalar NV12 reference.
  fn check_nv12_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
    let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
    let uv: std::vec::Vec<u8> = (0..width / 2)
      .flat_map(|i| {
        [
          ((i * 53 + 23) & 0xFF) as u8, // U_i
          ((i * 71 + 91) & 0xFF) as u8, // V_i
        ]
      })
      .collect();
    let mut rgb_scalar = std::vec![0u8; width * 3];
    let mut rgb_neon = std::vec![0u8; width * 3];

    scalar::nv12_to_rgb_row(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
    unsafe {
      nv12_to_rgb_row(&y, &uv, &mut rgb_neon, width, matrix, full_range);
    }

    if rgb_scalar != rgb_neon {
      let first_diff = rgb_scalar
        .iter()
        .zip(rgb_neon.iter())
        .position(|(a, b)| a != b)
        .unwrap();
      panic!(
        "NEON NV12 diverges from scalar at byte {first_diff} (width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} neon={}",
        rgb_scalar[first_diff], rgb_neon[first_diff]
      );
    }
  }

  /// Cross-format equivalence: the NV12 output must match the YUV420P
  /// output when fed the same U / V bytes interleaved. Guards against
  /// any stray deinterleave bug.
  fn check_nv12_matches_yuv420p(width: usize, matrix: ColorMatrix, full_range: bool) {
    let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
    let u: std::vec::Vec<u8> = (0..width / 2)
      .map(|i| ((i * 53 + 23) & 0xFF) as u8)
      .collect();
    let v: std::vec::Vec<u8> = (0..width / 2)
      .map(|i| ((i * 71 + 91) & 0xFF) as u8)
      .collect();
    let uv: std::vec::Vec<u8> = u.iter().zip(v.iter()).flat_map(|(a, b)| [*a, *b]).collect();

    let mut rgb_yuv420p = std::vec![0u8; width * 3];
    let mut rgb_nv12 = std::vec![0u8; width * 3];
    unsafe {
      yuv_420_to_rgb_row(&y, &u, &v, &mut rgb_yuv420p, width, matrix, full_range);
      nv12_to_rgb_row(&y, &uv, &mut rgb_nv12, width, matrix, full_range);
    }
    assert_eq!(
      rgb_yuv420p, rgb_nv12,
      "NV12 and YUV420P must produce byte-identical output for equivalent UV (width={width}, matrix={matrix:?}, full_range={full_range})"
    );
  }

  #[test]
  fn nv12_neon_matches_scalar_all_matrices_16() {
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
  fn nv12_neon_matches_scalar_width_1920() {
    check_nv12_equivalence(1920, ColorMatrix::Bt709, false);
  }

  #[test]
  fn nv12_neon_matches_scalar_odd_tail_widths() {
    for w in [18usize, 30, 34, 1922] {
      check_nv12_equivalence(w, ColorMatrix::Bt601, false);
    }
  }

  #[test]
  fn nv12_neon_matches_yuv420p_neon() {
    for w in [16usize, 30, 64, 1920] {
      check_nv12_matches_yuv420p(w, ColorMatrix::Bt709, false);
      check_nv12_matches_yuv420p(w, ColorMatrix::YCgCo, true);
    }
  }

  // ---- rgb_to_hsv_row equivalence ------------------------------------

  fn check_hsv_equivalence(rgb: &[u8], width: usize) {
    let mut h_scalar = std::vec![0u8; width];
    let mut s_scalar = std::vec![0u8; width];
    let mut v_scalar = std::vec![0u8; width];
    let mut h_neon = std::vec![0u8; width];
    let mut s_neon = std::vec![0u8; width];
    let mut v_neon = std::vec![0u8; width];

    scalar::rgb_to_hsv_row(rgb, &mut h_scalar, &mut s_scalar, &mut v_scalar, width);
    unsafe {
      rgb_to_hsv_row(rgb, &mut h_neon, &mut s_neon, &mut v_neon, width);
    }

    // Scalar uses integer LUT (matches OpenCV byte-exact), NEON uses
    // true f32 division. They can disagree by ±1 LSB at boundary
    // pixels — identical tolerance to what OpenCV reports between
    // their own scalar and SIMD HSV paths. Hue uses *circular*
    // distance since 0 and 179 are neighbors on the hue wheel: a pixel
    // at 360°≈0 in one path can land at 358°≈179 in the other due to
    // sign flips in delta with tiny f32 rounding.
    for (i, (a, b)) in h_scalar.iter().zip(h_neon.iter()).enumerate() {
      let d = a.abs_diff(*b);
      let circ = d.min(180 - d);
      assert!(circ <= 1, "H divergence at pixel {i}: scalar={a} neon={b}");
    }
    for (i, (a, b)) in s_scalar.iter().zip(s_neon.iter()).enumerate() {
      assert!(
        a.abs_diff(*b) <= 1,
        "S divergence at pixel {i}: scalar={a} neon={b}"
      );
    }
    for (i, (a, b)) in v_scalar.iter().zip(v_neon.iter()).enumerate() {
      assert!(
        a.abs_diff(*b) <= 1,
        "V divergence at pixel {i}: scalar={a} neon={b}"
      );
    }
  }

  fn pseudo_random_bgr(width: usize) -> std::vec::Vec<u8> {
    let n = width * 3;
    let mut out = std::vec::Vec::with_capacity(n);
    let mut state: u32 = 0x9E37_79B9;
    for _ in 0..n {
      state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
      out.push((state >> 8) as u8);
    }
    out
  }

  #[test]
  fn hsv_neon_matches_scalar_pseudo_random_16() {
    let rgb = pseudo_random_bgr(16);
    check_hsv_equivalence(&rgb, 16);
  }

  #[test]
  fn hsv_neon_matches_scalar_pseudo_random_1920() {
    let rgb = pseudo_random_bgr(1920);
    check_hsv_equivalence(&rgb, 1920);
  }

  #[test]
  fn hsv_neon_matches_scalar_tail_widths() {
    // Widths that force a non‑trivial scalar tail (non‑multiple of 16).
    for w in [1usize, 7, 15, 17, 31, 1921] {
      let rgb = pseudo_random_bgr(w);
      check_hsv_equivalence(&rgb, w);
    }
  }

  #[test]
  fn hsv_neon_matches_scalar_primaries_and_edges() {
    // Primary colors, grays, near‑saturation — exercise each hue branch
    // and the v==0, delta==0, h<0 wrap paths.
    let rgb: std::vec::Vec<u8> = [
      (0, 0, 0),       // black: v = 0 → s = 0, h = 0
      (255, 255, 255), // white: delta = 0 → s = 0, h = 0
      (128, 128, 128), // gray: delta = 0
      (255, 0, 0),     // pure red: v == r path
      (0, 255, 0),     // pure green: v == g path
      (0, 0, 255),     // pure blue: v == b path
      (255, 127, 0),   // red→yellow transition
      (0, 127, 255),   // blue→cyan
      (255, 0, 127),   // red→magenta
      (1, 2, 3),       // near black: small delta
      (254, 253, 252), // near white
      (150, 200, 10),  // arbitrary: v == g path, h > 0
      (150, 10, 200),  // arbitrary: v == b path
      (10, 200, 150),  // arbitrary: v == g
      (200, 100, 50),  // arbitrary: v == r
      (0, 64, 128),    // arbitrary: v == b
    ]
    .iter()
    .flat_map(|&(r, g, b)| [r, g, b])
    .collect();
    check_hsv_equivalence(&rgb, 16);
  }

  // ---- bgr_rgb_swap_row equivalence -----------------------------------

  fn check_swap_equivalence(width: usize) {
    let input = pseudo_random_bgr(width);
    let mut out_scalar = std::vec![0u8; width * 3];
    let mut out_neon = std::vec![0u8; width * 3];

    scalar::bgr_rgb_swap_row(&input, &mut out_scalar, width);
    unsafe {
      bgr_rgb_swap_row(&input, &mut out_neon, width);
    }

    assert_eq!(out_scalar, out_neon, "NEON swap diverges from scalar");

    // Byte 0 ↔ byte 2 should be swapped, byte 1 unchanged. Verify
    // the semantic directly.
    for x in 0..width {
      assert_eq!(
        out_scalar[x * 3],
        input[x * 3 + 2],
        "byte 0 != input byte 2"
      );
      assert_eq!(
        out_scalar[x * 3 + 1],
        input[x * 3 + 1],
        "middle byte changed"
      );
      assert_eq!(
        out_scalar[x * 3 + 2],
        input[x * 3],
        "byte 2 != input byte 0"
      );
    }
  }

  #[test]
  fn swap_neon_matches_scalar_widths() {
    for w in [1usize, 15, 16, 17, 31, 32, 1920, 1921] {
      check_swap_equivalence(w);
    }
  }

  #[test]
  fn swap_is_self_inverse() {
    let input = pseudo_random_bgr(64);
    let mut round_trip = std::vec![0u8; 64 * 3];
    let mut back = std::vec![0u8; 64 * 3];

    scalar::bgr_rgb_swap_row(&input, &mut round_trip, 64);
    scalar::bgr_rgb_swap_row(&round_trip, &mut back, 64);

    assert_eq!(input, back, "swap is not self-inverse");
  }
}
