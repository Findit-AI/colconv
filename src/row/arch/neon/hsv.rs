use core::arch::aarch64::*;

use crate::{ColorMatrix, row::scalar};

use super::*;

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
