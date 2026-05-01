//! Scalar reference kernels for the V410 packed YUV 4:4:4 10-bit
//! family (FFmpeg `AV_PIX_FMT_V410`). One pixel per 32-bit word;
//! 10-bit U / Y / V channels with 2-bit padding (see
//! [`crate::frame::V410Frame`]). 4:4:4 means no chroma deinterleave
//! step — each word yields a complete `(Y, U, V)` triple.

use super::*;

/// Extract `(u, y, v)` from one V410 word.
#[allow(dead_code)]
#[cfg_attr(not(tarpaulin), inline(always))]
const fn extract_v410(word: u32) -> (i32, i32, i32) {
  let u = (word & 0x3FF) as i32;
  let y = ((word >> 10) & 0x3FF) as i32;
  let v = ((word >> 20) & 0x3FF) as i32;
  (u, y, v)
}

// ---- u8 RGB / RGBA output ----------------------------------------------

#[allow(dead_code)]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn v410_to_rgb_or_rgba_row<const ALPHA: bool>(
  packed: &[u32],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(packed.len() >= width, "packed row too short");
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp, "out row too short");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<10, 8>(full_range);
  let bias = chroma_bias::<10>();

  for (x, &word) in packed[..width].iter().enumerate() {
    let (u, y, v) = extract_v410(word);
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
pub(crate) fn v410_to_rgb_u16_or_rgba_u16_row<const ALPHA: bool>(
  packed: &[u32],
  out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(packed.len() >= width, "packed row too short");
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp, "out row too short");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<10, 10>(full_range);
  let bias = chroma_bias::<10>();
  let alpha_max: u16 = 0x3FF;
  let out_max: i32 = 0x3FF;

  for (x, &word) in packed[..width].iter().enumerate() {
    let (u, y, v) = extract_v410(word);
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

// ---- Luma (u8) — `>> 2` ------------------------------------------------

#[allow(dead_code)]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn v410_to_luma_row(packed: &[u32], out: &mut [u8], width: usize) {
  debug_assert!(packed.len() >= width);
  debug_assert!(out.len() >= width);
  for x in 0..width {
    let y = (packed[x] >> 10) & 0x3FF;
    out[x] = (y >> 2) as u8;
  }
}

// ---- Luma (u16, low-bit-packed at 10-bit) ------------------------------

#[allow(dead_code)]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn v410_to_luma_u16_row(packed: &[u32], out: &mut [u16], width: usize) {
  debug_assert!(packed.len() >= width);
  debug_assert!(out.len() >= width);
  for x in 0..width {
    let y = (packed[x] >> 10) & 0x3FF;
    out[x] = y as u16;
  }
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;
  use crate::ColorMatrix;

  /// Pack one V410 word from explicit U / Y / V samples.
  fn pack_v410(u: u32, y: u32, v: u32) -> u32 {
    debug_assert!(u < 1024 && y < 1024 && v < 1024);
    (v << 20) | (y << 10) | u
  }

  #[test]
  fn v410_known_pattern_rgb() {
    // Limited-range BT.709, gray Y=64 (≈ 0 in [0, 255]) with neutral
    // chroma U=V=512. Both pixels should produce ~black [0, 0, 0]
    // before saturation.
    let p = vec![
      pack_v410(512, 64, 512),
      pack_v410(512, 64, 512),
      pack_v410(512, 940, 512), // Y=940 ≈ 255 (limited-range white)
      pack_v410(512, 940, 512),
    ];
    let mut out = vec![0u8; 4 * 3];
    v410_to_rgb_or_rgba_row::<false>(&p, &mut out, 4, ColorMatrix::Bt709, false);
    // Two black pixels followed by two white pixels.
    assert_eq!(&out[0..3], &[0u8, 0, 0]);
    assert_eq!(&out[3..6], &[0u8, 0, 0]);
    assert_eq!(&out[6..9], &[255u8, 255, 255]);
    assert_eq!(&out[9..12], &[255u8, 255, 255]);
  }

  #[test]
  fn v410_known_pattern_rgba_alpha_max() {
    let p = vec![pack_v410(512, 940, 512)];
    let mut out = vec![0u8; 4];
    v410_to_rgb_or_rgba_row::<true>(&p, &mut out, 1, ColorMatrix::Bt709, false);
    assert_eq!(out[3], 0xFF);
  }

  #[test]
  fn v410_luma_extract_u8() {
    let p = vec![
      pack_v410(0, 0x3FF, 0), // Y = 0x3FF (10-bit max)
      pack_v410(0, 0x100, 0), // Y = 0x100
    ];
    let mut out = vec![0u8; 2];
    v410_to_luma_row(&p, &mut out, 2);
    // 0x3FF >> 2 = 0xFF; 0x100 >> 2 = 0x40.
    assert_eq!(&out[..], &[0xFFu8, 0x40]);
  }

  #[test]
  fn v410_luma_extract_u16_low_bit_packed() {
    let p = vec![pack_v410(0, 0x3FF, 0), pack_v410(0, 0x123, 0)];
    let mut out = vec![0u16; 2];
    v410_to_luma_u16_row(&p, &mut out, 2);
    assert_eq!(&out[..], &[0x3FFu16, 0x123]);
  }

  #[test]
  fn v410_known_pattern_rgba_u16_alpha_max() {
    let p = vec![pack_v410(512, 940, 512)];
    let mut out = vec![0u16; 4];
    v410_to_rgb_u16_or_rgba_u16_row::<true>(&p, &mut out, 1, ColorMatrix::Bt709, false);
    // 10-bit alpha max is 0x3FF (low-bit-packed).
    assert_eq!(out[3], 0x3FF);
  }
}
