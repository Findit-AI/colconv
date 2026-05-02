//! Scalar reference kernels for the XV36 packed YUV 4:4:4 12-bit
//! family (FFmpeg `AV_PIX_FMT_XV36LE`). Each pixel is a u16
//! quadruple `[U, Y, V, A]` MSB-aligned at 12-bit (low 4 bits zero
//! per sample). The A slot is padding — read but discarded; RGBA
//! outputs force α = max.
//!
//! 4:4:4 means no chroma deinterleave step — each pixel's U / Y / V
//! are independent. Bit extraction is `>> 4` to drop the 4 padding
//! LSBs, then the standard Q15 chroma + Y pipeline at BITS=12 (i32
//! chroma — same depth as Y2xx at BITS=12).

use super::*;

/// Extract `(u, y, v)` from one XV36 pixel. The `a` slot at index 3
/// is padding and is not returned. Each channel is `>> 4` to drop
/// the 4 padding LSBs, bringing the 12-bit MSB-aligned sample to
/// the BITS=12 range `[0, 4095]`.
#[cfg_attr(not(tarpaulin), inline(always))]
const fn extract_xv36(quad: &[u16]) -> (i32, i32, i32) {
  let u = (quad[0] >> 4) as i32;
  let y = (quad[1] >> 4) as i32;
  let v = (quad[2] >> 4) as i32;
  // quad[3] is padding A — ignored
  (u, y, v)
}

// ---- u8 RGB / RGBA output ----------------------------------------------

#[allow(dead_code)]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn xv36_to_rgb_or_rgba_row<const ALPHA: bool>(
  packed: &[u16],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(packed.len() >= width * 4, "packed row too short");
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp, "out row too short");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<12, 8>(full_range);
  let bias = chroma_bias::<12>();

  for x in 0..width {
    let quad = &packed[x * 4..x * 4 + 4];
    let (u, y, v) = extract_xv36(quad);
    let u_d = q15_scale(u - bias, c_scale);
    let v_d = q15_scale(v - bias, c_scale);
    let r_chroma = q15_chroma(coeffs.r_u(), u_d, coeffs.r_v(), v_d);
    let g_chroma = q15_chroma(coeffs.g_u(), u_d, coeffs.g_v(), v_d);
    let b_chroma = q15_chroma(coeffs.b_u(), u_d, coeffs.b_v(), v_d);

    let y_s = q15_scale(y - y_off, y_scale);
    let off = x * bpp;
    out[off] = clamp_u8(y_s + r_chroma);
    out[off + 1] = clamp_u8(y_s + g_chroma);
    out[off + 2] = clamp_u8(y_s + b_chroma);
    if ALPHA {
      out[off + 3] = 0xFF;
    }
  }
}

// ---- u16 RGB / RGBA native-depth output --------------------------------

#[allow(dead_code)]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn xv36_to_rgb_u16_or_rgba_u16_row<const ALPHA: bool>(
  packed: &[u16],
  out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(packed.len() >= width * 4, "packed row too short");
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp, "out row too short");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<12, 12>(full_range);
  let bias = chroma_bias::<12>();
  let alpha_max: u16 = 0x0FFF;
  let out_max: i32 = 0x0FFF;

  for x in 0..width {
    let quad = &packed[x * 4..x * 4 + 4];
    let (u, y, v) = extract_xv36(quad);
    let u_d = q15_scale(u - bias, c_scale);
    let v_d = q15_scale(v - bias, c_scale);
    let r_chroma = q15_chroma(coeffs.r_u(), u_d, coeffs.r_v(), v_d);
    let g_chroma = q15_chroma(coeffs.g_u(), u_d, coeffs.g_v(), v_d);
    let b_chroma = q15_chroma(coeffs.b_u(), u_d, coeffs.b_v(), v_d);

    let y_s = q15_scale(y - y_off, y_scale);
    let off = x * bpp;
    out[off] = (y_s + r_chroma).clamp(0, out_max) as u16;
    out[off + 1] = (y_s + g_chroma).clamp(0, out_max) as u16;
    out[off + 2] = (y_s + b_chroma).clamp(0, out_max) as u16;
    if ALPHA {
      out[off + 3] = alpha_max;
    }
  }
}

// ---- Luma (u8) — `>> 8` (drops 4 padding bits + 4 LSBs) ----------------

#[allow(dead_code)]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn xv36_to_luma_row(packed: &[u16], out: &mut [u8], width: usize) {
  debug_assert!(packed.len() >= width * 4);
  debug_assert!(out.len() >= width);
  for x in 0..width {
    let y = packed[x * 4 + 1] >> 8;
    out[x] = y as u8;
  }
}

// ---- Luma (u16, low-bit-packed at 12-bit) ------------------------------

#[allow(dead_code)]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn xv36_to_luma_u16_row(packed: &[u16], out: &mut [u16], width: usize) {
  debug_assert!(packed.len() >= width * 4);
  debug_assert!(out.len() >= width);
  for x in 0..width {
    let y = packed[x * 4 + 1] >> 4;
    out[x] = y;
  }
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;
  use crate::ColorMatrix;

  /// Pack one XV36 pixel from explicit U / Y / V / A samples.
  /// Each channel value must be in `[0, 0xFFF]`; the helper shifts
  /// each by 4 to MSB-align into the high 12 bits.
  fn pack_xv36(u: u16, y: u16, v: u16, a: u16) -> [u16; 4] {
    debug_assert!(u <= 0xFFF && y <= 0xFFF && v <= 0xFFF && a <= 0xFFF);
    [u << 4, y << 4, v << 4, a << 4]
  }

  #[test]
  fn xv36_known_pattern_rgb() {
    // Limited-range BT.709, gray Y=256 (≈ 0 in u8) with neutral
    // chroma U=V=2048. Then white Y=3760 (≈ 255).
    let p0 = pack_xv36(2048, 256, 2048, 0);
    let p1 = pack_xv36(2048, 256, 2048, 0);
    let p2 = pack_xv36(2048, 3760, 2048, 0);
    let p3 = pack_xv36(2048, 3760, 2048, 0);
    let packed: Vec<u16> = [p0, p1, p2, p3].iter().flatten().copied().collect();
    let mut out = vec![0u8; 4 * 3];
    xv36_to_rgb_or_rgba_row::<false>(&packed, &mut out, 4, ColorMatrix::Bt709, false);
    assert_eq!(&out[0..3], &[0u8, 0, 0]);
    assert_eq!(&out[3..6], &[0u8, 0, 0]);
    assert_eq!(&out[6..9], &[255u8, 255, 255]);
    assert_eq!(&out[9..12], &[255u8, 255, 255]);
  }

  #[test]
  fn xv36_known_pattern_rgba_alpha_max() {
    let p = pack_xv36(2048, 3760, 2048, 0);
    let packed: Vec<u16> = p.into_iter().collect();
    let mut out = vec![0u8; 4];
    xv36_to_rgb_or_rgba_row::<true>(&packed, &mut out, 1, ColorMatrix::Bt709, false);
    // X = padding; RGBA forces α=0xFF regardless of source A byte.
    assert_eq!(out[3], 0xFF);
  }

  #[test]
  fn xv36_known_pattern_rgba_ignores_source_alpha_bits() {
    // Source A=0x123 (low 12 bits set) — should not leak into RGB or affect α.
    let p = pack_xv36(2048, 3760, 2048, 0xFFF);
    let packed: Vec<u16> = p.into_iter().collect();
    let mut out = vec![0u8; 4];
    xv36_to_rgb_or_rgba_row::<true>(&packed, &mut out, 1, ColorMatrix::Bt709, false);
    assert_eq!(out[3], 0xFF);
  }

  #[test]
  fn xv36_luma_extract_u8() {
    // Y = 0xFFF → 0xFFF >> 4 = 0xFF (12-bit max); Y = 0x100 → 0x10
    let packed: Vec<u16> = [pack_xv36(0, 0xFFF, 0, 0), pack_xv36(0, 0x100, 0, 0)]
      .iter()
      .flatten()
      .copied()
      .collect();
    let mut out = vec![0u8; 2];
    xv36_to_luma_row(&packed, &mut out, 2);
    assert_eq!(&out[..], &[0xFFu8, 0x10]);
  }

  #[test]
  fn xv36_luma_extract_u16_low_bit_packed() {
    let packed: Vec<u16> = [pack_xv36(0, 0xFFF, 0, 0), pack_xv36(0, 0x123, 0, 0)]
      .iter()
      .flatten()
      .copied()
      .collect();
    let mut out = vec![0u16; 2];
    xv36_to_luma_u16_row(&packed, &mut out, 2);
    assert_eq!(&out[..], &[0xFFFu16, 0x123]);
  }

  #[test]
  fn xv36_known_pattern_rgba_u16_alpha_max() {
    let p = pack_xv36(2048, 3760, 2048, 0xFFF);
    let packed: Vec<u16> = p.into_iter().collect();
    let mut out = vec![0u16; 4];
    xv36_to_rgb_u16_or_rgba_u16_row::<true>(&packed, &mut out, 1, ColorMatrix::Bt709, false);
    // 12-bit alpha max = 0x0FFF; X = padding so source A byte is ignored.
    assert_eq!(out[3], 0x0FFF);
  }
}
