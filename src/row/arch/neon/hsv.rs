#[cfg_attr(miri, allow(unused_imports))]
use core::arch::aarch64::*;

use super::miri_compat::*;
use crate::{KernelMatrix, row::scalar};

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
    let v = vmaxq_f32_compat(vmaxq_f32_compat(b, g), r);
    let min_bgr = vminq_f32_compat(vminq_f32_compat(b, g), r);
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
    let h_quant = vminq_f32_compat(
      vmaxq_f32_compat(vaddq_f32(vmulq_f32(hue, half), half), zero),
      one_seventy_nine,
    );
    let s_quant = vminq_f32_compat(vmaxq_f32_compat(vaddq_f32(s, half), zero), two_fifty_five);
    let v_quant = vminq_f32_compat(vmaxq_f32_compat(vaddq_f32(v, half), zero), two_fifty_five);

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

// ===== RGB → luma (Y') ===================================================

/// NEON RGB → planar luma (Y'). Byte‑identical to
/// [`scalar::rgb_to_luma_row`] (Q15 weighted sum, optional limited‑range
/// post‑scale to `[16, 235]`).
///
/// Block size: 16 px / iter (one `vld3q_u8` deinterleave = 48 input
/// bytes, one `vst1q_u8` luma store = 16 output bytes). Coefficients
/// are hoisted outside the loop — `matrix` is selected once via
/// `luma_coefficients_q15`, then splatted to i16x4 vectors so the
/// in‑loop multiplies use `vmull_s16` (i16 x i16 → i32 widening).
///
/// # Safety
///
/// 1. **NEON must be available on the current CPU** (caller obligation).
/// 2. `rgb.len() >= 3 * width`.
/// 3. `luma_out.len() >= width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn rgb_to_luma_row(
  rgb: &[u8],
  luma_out: &mut [u8],
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  debug_assert!(rgb.len() >= width * 3, "rgb row too short");
  debug_assert!(luma_out.len() >= width, "luma row too short");

  let (k_r, k_g, k_b) = scalar::luma_coefficients_q15(matrix);
  // SAFETY block carries unchecked NEON intrinsics; the splats below
  // are safe (only register operations) but live inside the same
  // unsafe context as the load/multiply chain.
  // All matrix coefficients fit in i16 (max ≈ 23436 ≪ 32767), so we
  // can use the cheap i16xi16 → i32 widening multiply (`vmull_s16`).
  let kr_v = vdup_n_s16(k_r as i16);
  let kg_v = vdup_n_s16(k_g as i16);
  let kb_v = vdup_n_s16(k_b as i16);
  let rnd_v = vdupq_n_s32(1 << 14);
  // Limited‑range post‑scale constants. Hoisted once even when unused;
  // unused branches inline as dead and the splats don't cost anything.
  let lim_scale_v = vdup_n_s16(28142);
  let lim_off_v = vdupq_n_s16(16);

  // SAFETY: NEON availability is the caller's obligation; loop guard
  // `x + 16 <= width` keeps the 48‑byte read and 16‑byte write inside
  // the caller‑promised slice lengths.
  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      // Deinterleave 16 RGB pixels (48 bytes) → 3 x u8x16 channels.
      let rgb_vec = vld3q_u8(rgb.as_ptr().add(x * 3));
      let r_u8 = rgb_vec.0;
      let g_u8 = rgb_vec.1;
      let b_u8 = rgb_vec.2;

      // Widen each u8x16 to two i16x8 halves (zero‑extend), since
      // every sample is in [0, 255] and the multiply target is i16 x i16.
      let r_lo_u16 = vmovl_u8(vget_low_u8(r_u8));
      let r_hi_u16 = vmovl_u8(vget_high_u8(r_u8));
      let g_lo_u16 = vmovl_u8(vget_low_u8(g_u8));
      let g_hi_u16 = vmovl_u8(vget_high_u8(g_u8));
      let b_lo_u16 = vmovl_u8(vget_low_u8(b_u8));
      let b_hi_u16 = vmovl_u8(vget_high_u8(b_u8));
      let r_lo = vreinterpretq_s16_u16(r_lo_u16);
      let r_hi = vreinterpretq_s16_u16(r_hi_u16);
      let g_lo = vreinterpretq_s16_u16(g_lo_u16);
      let g_hi = vreinterpretq_s16_u16(g_hi_u16);
      let b_lo = vreinterpretq_s16_u16(b_lo_u16);
      let b_hi = vreinterpretq_s16_u16(b_hi_u16);

      // Y_full per i32x4 quarter: (k_r·R + k_g·G + k_b·B + RND) >> 15.
      let y0 = q15_luma(
        vget_low_s16(r_lo),
        vget_low_s16(g_lo),
        vget_low_s16(b_lo),
        kr_v,
        kg_v,
        kb_v,
        rnd_v,
      );
      let y1 = q15_luma(
        vget_high_s16(r_lo),
        vget_high_s16(g_lo),
        vget_high_s16(b_lo),
        kr_v,
        kg_v,
        kb_v,
        rnd_v,
      );
      let y2 = q15_luma(
        vget_low_s16(r_hi),
        vget_low_s16(g_hi),
        vget_low_s16(b_hi),
        kr_v,
        kg_v,
        kb_v,
        rnd_v,
      );
      let y3 = q15_luma(
        vget_high_s16(r_hi),
        vget_high_s16(g_hi),
        vget_high_s16(b_hi),
        kr_v,
        kg_v,
        kb_v,
        rnd_v,
      );

      // Saturate‑narrow to i16x8 (clamps negatives → 0 and >32767 → 32767;
      // both extremes are well outside our [0,255] expected range and
      // can only occur when coefficient sums round slightly above 1.0).
      let y_lo_i16 = vcombine_s16(vqmovn_s32_compat(y0), vqmovn_s32_compat(y1));
      let y_hi_i16 = vcombine_s16(vqmovn_s32_compat(y2), vqmovn_s32_compat(y3));

      let y_u8 = if full_range {
        // Saturate‑narrow i16x8x2 → u8x16 ([0,255] clamp).
        vcombine_u8(vqmovun_s16_compat(y_lo_i16), vqmovun_s16_compat(y_hi_i16))
      } else {
        // Limited‑range post‑scale: clamp Y_full to [0,255] first
        // (matches the scalar's `y_full_clamped` step), then apply the
        // 28142/32768 Q15 multiply and add 16. Re‑use Q15 widening
        // multiply via vmull_s16.
        let y_clamp_u8_lo = vqmovun_s16_compat(y_lo_i16);
        let y_clamp_u8_hi = vqmovun_s16_compat(y_hi_i16);
        // Re‑widen u8 → i16 (always non‑negative, so signed and
        // unsigned widen produce the same bit pattern).
        let yc_lo_i16 = vreinterpretq_s16_u16(vmovl_u8(y_clamp_u8_lo));
        let yc_hi_i16 = vreinterpretq_s16_u16(vmovl_u8(y_clamp_u8_hi));
        let y_lim_lo = limited_range_scale(yc_lo_i16, lim_scale_v, lim_off_v, rnd_v);
        let y_lim_hi = limited_range_scale(yc_hi_i16, lim_scale_v, lim_off_v, rnd_v);
        vcombine_u8(vqmovun_s16_compat(y_lim_lo), vqmovun_s16_compat(y_lim_hi))
      };

      vst1q_u8(luma_out.as_mut_ptr().add(x), y_u8);
      x += 16;
    }

    if x < width {
      scalar::rgb_to_luma_row(
        &rgb[x * 3..width * 3],
        &mut luma_out[x..width],
        width - x,
        matrix,
        full_range,
      );
    }
  }
}

/// Q15 weighted sum for one 4‑pixel group.
/// Returns `(k_r·R + k_g·G + k_b·B + RND) >> 15` as i32x4.
#[inline(always)]
fn q15_luma(
  r: int16x4_t,
  g: int16x4_t,
  b: int16x4_t,
  kr: int16x4_t,
  kg: int16x4_t,
  kb: int16x4_t,
  rnd: int32x4_t,
) -> int32x4_t {
  unsafe {
    let acc = vmull_s16(r, kr);
    let acc = vmlal_s16(acc, g, kg);
    let acc = vmlal_s16(acc, b, kb);
    let acc = vaddq_s32(acc, rnd);
    vshrq_n_s32::<15>(acc)
  }
}

/// Limited‑range post‑scale: `16 + ((y_clamped * 28142 + RND) >> 15)`,
/// applied in i16 arithmetic. `y_clamped` is in `[0, 255]` (already
/// clamped by the caller), so `y_clamped * 28142 ≤ 7.18M` fits in i32.
#[inline(always)]
fn limited_range_scale(
  yc: int16x8_t,
  scale: int16x4_t,
  off: int16x8_t,
  rnd: int32x4_t,
) -> int16x8_t {
  unsafe {
    let lo = vmull_s16(vget_low_s16(yc), scale);
    let hi = vmull_s16(vget_high_s16(yc), scale);
    let lo = vshrq_n_s32::<15>(vaddq_s32(lo, rnd));
    let hi = vshrq_n_s32::<15>(vaddq_s32(hi, rnd));
    let scaled_i16 = vcombine_s16(vqmovn_s32_compat(lo), vqmovn_s32_compat(hi));
    vaddq_s16(scaled_i16, off)
  }
}
