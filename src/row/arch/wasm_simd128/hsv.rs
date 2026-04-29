use core::arch::wasm32::*;

use super::*;

// ===== RGB → HSV =========================================================

/// WASM simd128 RGB → planar HSV. 16 pixels per iteration using
/// byte‑shuffle deinterleave + four f32x4 HSV groups. Mirrors the NEON
/// and x86 kernels op‑for‑op (true `f32x4_div` for the two divisions,
/// `v128_bitselect` for the branch cascade). Bit‑identical to
/// [`scalar::rgb_to_hsv_row`].
///
/// # Safety
///
/// 1. simd128 must be enabled at compile time.
/// 2. `rgb.len() >= 3 * width`; each output plane `>= width`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn rgb_to_hsv_row(
  rgb: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
) {
  debug_assert!(rgb.len() >= width * 3);
  debug_assert!(h_out.len() >= width);
  debug_assert!(s_out.len() >= width);
  debug_assert!(v_out.len() >= width);

  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      let in0 = v128_load(rgb.as_ptr().add(x * 3).cast());
      let in1 = v128_load(rgb.as_ptr().add(x * 3 + 16).cast());
      let in2 = v128_load(rgb.as_ptr().add(x * 3 + 32).cast());

      // 3‑channel deinterleave — mirror of the x86 mask pattern.
      let mr0 = i8x16(0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
      let mr1 = i8x16(-1, -1, -1, -1, -1, -1, 2, 5, 8, 11, 14, -1, -1, -1, -1, -1);
      let mr2 = i8x16(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 4, 7, 10, 13);
      let r_u8 = v128_or(
        v128_or(u8x16_swizzle(in0, mr0), u8x16_swizzle(in1, mr1)),
        u8x16_swizzle(in2, mr2),
      );

      let mg0 = i8x16(1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
      let mg1 = i8x16(-1, -1, -1, -1, -1, 0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1);
      let mg2 = i8x16(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 5, 8, 11, 14);
      let g_u8 = v128_or(
        v128_or(u8x16_swizzle(in0, mg0), u8x16_swizzle(in1, mg1)),
        u8x16_swizzle(in2, mg2),
      );

      let mb0 = i8x16(2, 5, 8, 11, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
      let mb1 = i8x16(-1, -1, -1, -1, -1, 1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1);
      let mb2 = i8x16(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 3, 6, 9, 12, 15);
      let b_u8 = v128_or(
        v128_or(u8x16_swizzle(in0, mb0), u8x16_swizzle(in1, mb1)),
        u8x16_swizzle(in2, mb2),
      );

      // Widen each u8x16 to 4 f32x4 groups.
      let (r0, r1, r2, r3) = u8x16_to_f32x4_quad(r_u8);
      let (g0, g1, g2, g3) = u8x16_to_f32x4_quad(g_u8);
      let (b0, b1, b2, b3) = u8x16_to_f32x4_quad(b_u8);

      let (h0, s0, v0) = hsv_group(r0, g0, b0);
      let (h1, s1, v1) = hsv_group(r1, g1, b1);
      let (h2, s2, v2) = hsv_group(r2, g2, b2);
      let (h3, s3, v3) = hsv_group(r3, g3, b3);

      v128_store(
        h_out.as_mut_ptr().add(x).cast(),
        f32x4_quad_to_u8x16(h0, h1, h2, h3),
      );
      v128_store(
        s_out.as_mut_ptr().add(x).cast(),
        f32x4_quad_to_u8x16(s0, s1, s2, s3),
      );
      v128_store(
        v_out.as_mut_ptr().add(x).cast(),
        f32x4_quad_to_u8x16(v0, v1, v2, v3),
      );

      x += 16;
    }
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

// ---- RGB→HSV helpers (wasm simd128) ----------------------------------

/// Widens a u8x16 to four f32x4 groups.
#[inline(always)]
fn u8x16_to_f32x4_quad(v: v128) -> (v128, v128, v128, v128) {
  // u8x16 → u16x8 × 2 → u32x4 × 4 → f32x4 × 4.
  let u16_lo = u16x8_extend_low_u8x16(v);
  let u16_hi = u16x8_extend_high_u8x16(v);
  let u32_0 = u32x4_extend_low_u16x8(u16_lo);
  let u32_1 = u32x4_extend_high_u16x8(u16_lo);
  let u32_2 = u32x4_extend_low_u16x8(u16_hi);
  let u32_3 = u32x4_extend_high_u16x8(u16_hi);
  (
    f32x4_convert_i32x4(u32_0),
    f32x4_convert_i32x4(u32_1),
    f32x4_convert_i32x4(u32_2),
    f32x4_convert_i32x4(u32_3),
  )
}

/// Packs four f32x4 vectors to one u8x16. Values are pre‑clamped to
/// [0, 255] so the two narrowing steps don't clip.
#[inline(always)]
fn f32x4_quad_to_u8x16(a: v128, b: v128, c: v128, d: v128) -> v128 {
  let ai = i32x4_trunc_sat_f32x4(a);
  let bi = i32x4_trunc_sat_f32x4(b);
  let ci = i32x4_trunc_sat_f32x4(c);
  let di = i32x4_trunc_sat_f32x4(d);
  // i32x4 × 2 → i16x8 (signed saturating — fits since values in [0, 255]).
  let ab = i16x8_narrow_i32x4(ai, bi);
  let cd = i16x8_narrow_i32x4(ci, di);
  // i16x8 × 2 → u8x16 (unsigned saturating).
  u8x16_narrow_i16x8(ab, cd)
}

/// HSV compute for 4 pixels in f32x4 lanes. Mirrors the scalar
/// `rgb_to_hsv_pixel` op‑for‑op; returns already‑clamped H/S/V values
/// as f32x4 awaiting the truncating cast in the caller.
#[inline(always)]
fn hsv_group(r: v128, g: v128, b: v128) -> (v128, v128, v128) {
  let zero = f32x4_splat(0.0);
  let half = f32x4_splat(0.5);
  let sixty = f32x4_splat(60.0);
  let one_twenty = f32x4_splat(120.0);
  let two_forty = f32x4_splat(240.0);
  let three_sixty = f32x4_splat(360.0);
  let one_seventy_nine = f32x4_splat(179.0);
  let two_fifty_five = f32x4_splat(255.0);

  let v = f32x4_max(f32x4_max(r, g), b);
  let min_rgb = f32x4_min(f32x4_min(r, g), b);
  let delta = f32x4_sub(v, min_rgb);

  // S = if v == 0 { 0 } else { 255 * delta / v }.
  let mask_v_zero = f32x4_eq(v, zero);
  let s_nonzero = f32x4_div(f32x4_mul(two_fifty_five, delta), v);
  // `v128_bitselect(a, b, mask)`: per‑bit, pick a where mask bit = 1,
  // else b. Mask from f32 compare is all‑ones in "true" lanes.
  let s = v128_bitselect(zero, s_nonzero, mask_v_zero);

  let mask_delta_zero = f32x4_eq(delta, zero);
  let mask_v_is_r = f32x4_eq(v, r);
  let mask_v_is_g = f32x4_eq(v, g);

  let h_r_raw = f32x4_div(f32x4_mul(sixty, f32x4_sub(g, b)), delta);
  let mask_neg = f32x4_lt(h_r_raw, zero);
  let h_r = v128_bitselect(f32x4_add(h_r_raw, three_sixty), h_r_raw, mask_neg);

  let h_g = f32x4_add(
    f32x4_div(f32x4_mul(sixty, f32x4_sub(b, r)), delta),
    one_twenty,
  );
  let h_b = f32x4_add(
    f32x4_div(f32x4_mul(sixty, f32x4_sub(r, g)), delta),
    two_forty,
  );

  // Cascade: delta == 0 → 0; v == r → h_r; v == g → h_g; else → h_b.
  let h_g_or_b = v128_bitselect(h_g, h_b, mask_v_is_g);
  let h_nonzero = v128_bitselect(h_r, h_g_or_b, mask_v_is_r);
  let hue = v128_bitselect(zero, h_nonzero, mask_delta_zero);

  // Quantize to scalar output ranges.
  let h_quant = f32x4_min(
    f32x4_max(f32x4_add(f32x4_mul(hue, half), half), zero),
    one_seventy_nine,
  );
  let s_quant = f32x4_min(f32x4_max(f32x4_add(s, half), zero), two_fifty_five);
  let v_quant = f32x4_min(f32x4_max(f32x4_add(v, half), zero), two_fifty_five);

  (h_quant, s_quant, v_quant)
}
