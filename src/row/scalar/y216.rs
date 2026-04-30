//! Scalar reference kernels for the Y216 packed YUV 4:2:2 family
//! (BITS=16, full-range u16 samples). Separated from `y2xx.rs`
//! because the u16 native-depth output path uses i64 chroma to
//! avoid overflow — at BITS=16, `q15_coeff × chroma + q15_coeff
//! × chroma` exceeds i32 range. Mirrors
//! `src/row/scalar/yuv_planar_16bit.rs`'s i64 chroma scalar
//! pattern but sourced from YUYV-shaped u16 quadruples rather
//! than separate Y/U/V planes.

use super::*;

// ---- u8 RGB / RGBA output (i32 chroma — same as Y210/Y212) -------

// Dispatcher (`src/row/dispatch/y216.rs`) lands in a follow-up PR.
// `dead_code` allow lets this scalar prep ship without the eventual
// call site, matching the `rgb_expand.rs` precedent.
#[allow(dead_code)]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn y216_to_rgb_or_rgba_row<const ALPHA: bool>(
  packed: &[u16],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(width.is_multiple_of(2), "Y216 requires even width");
  debug_assert!(packed.len() >= width * 2, "packed row too short");
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp, "out row too short");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<16, 8>(full_range);
  let bias = chroma_bias::<16>();

  let pairs = width / 2;
  for p in 0..pairs {
    let q = &packed[p * 4..p * 4 + 4];
    // No right-shift: BITS=16 means samples are already full-width.
    let y0 = q[0] as i32;
    let u = q[1] as i32;
    let y1 = q[2] as i32;
    let v = q[3] as i32;

    let u_d = q15_scale(u - bias, c_scale);
    let v_d = q15_scale(v - bias, c_scale);

    let r_chroma = q15_chroma(coeffs.r_u(), u_d, coeffs.r_v(), v_d);
    let g_chroma = q15_chroma(coeffs.g_u(), u_d, coeffs.g_v(), v_d);
    let b_chroma = q15_chroma(coeffs.b_u(), u_d, coeffs.b_v(), v_d);

    for (k, &y) in [y0, y1].iter().enumerate() {
      let y_s = q15_scale(y - y_off, y_scale);
      let off = (p * 2 + k) * bpp;
      out[off] = clamp_u8(y_s + r_chroma);
      out[off + 1] = clamp_u8(y_s + g_chroma);
      out[off + 2] = clamp_u8(y_s + b_chroma);
      if ALPHA {
        out[off + 3] = 0xFF;
      }
    }
  }
}

// ---- u16 RGB / RGBA native-depth output (i64 chroma) ----------------

#[allow(dead_code)]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn y216_to_rgb_u16_or_rgba_u16_row<const ALPHA: bool>(
  packed: &[u16],
  out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(width.is_multiple_of(2), "Y216 requires even width");
  debug_assert!(packed.len() >= width * 2, "packed row too short");
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp, "out row too short");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<16, 16>(full_range);
  let bias = chroma_bias::<16>();
  let out_max: i32 = 0xFFFF;

  let pairs = width / 2;
  for p in 0..pairs {
    let q = &packed[p * 4..p * 4 + 4];
    let y0 = q[0] as i32;
    let u = q[1] as i32;
    let y1 = q[2] as i32;
    let v = q[3] as i32;

    let u_d = q15_scale(u - bias, c_scale);
    let v_d = q15_scale(v - bias, c_scale);

    let r_chroma = q15_chroma64(coeffs.r_u(), u_d, coeffs.r_v(), v_d);
    let g_chroma = q15_chroma64(coeffs.g_u(), u_d, coeffs.g_v(), v_d);
    let b_chroma = q15_chroma64(coeffs.b_u(), u_d, coeffs.b_v(), v_d);

    for (k, &y) in [y0, y1].iter().enumerate() {
      let y_s = q15_scale64(y - y_off, y_scale);
      let off = (p * 2 + k) * bpp;
      out[off] = (y_s + r_chroma).clamp(0, out_max) as u16;
      out[off + 1] = (y_s + g_chroma).clamp(0, out_max) as u16;
      out[off + 2] = (y_s + b_chroma).clamp(0, out_max) as u16;
      if ALPHA {
        out[off + 3] = 0xFFFF;
      }
    }
  }
}

// ---- Luma (u8) — `>> 8` ----------------------------------------------

#[allow(dead_code)]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn y216_to_luma_row(packed: &[u16], out: &mut [u8], width: usize) {
  debug_assert!(width.is_multiple_of(2));
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width);
  let pairs = width / 2;
  for p in 0..pairs {
    let q = &packed[p * 4..p * 4 + 4];
    out[p * 2] = (q[0] >> 8) as u8;
    out[p * 2 + 1] = (q[2] >> 8) as u8;
  }
}

// ---- Luma (u16, direct extract) ---------------------------------------

#[allow(dead_code)]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn y216_to_luma_u16_row(packed: &[u16], out: &mut [u16], width: usize) {
  debug_assert!(width.is_multiple_of(2));
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width);
  let pairs = width / 2;
  for p in 0..pairs {
    let q = &packed[p * 4..p * 4 + 4];
    out[p * 2] = q[0]; // direct extract — full 16 bits, no shift
    out[p * 2 + 1] = q[2];
  }
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;
  use crate::ColorMatrix;

  // Test input layout: two YUYV quadruples (= 4 pixels, width=4).
  // Pair 0: Y0=4096, U=32768, Y1=32000, V=32768  (neutral chroma, Bt709 limited floor / mid-gray)
  // Pair 1: Y2=0,    U=16384, Y3=65535, V=49152  (non-neutral chroma, below-floor / above-ceil Y)
  fn test_input() -> [u16; 8] {
    [4096, 32768, 32000, 32768, 0, 16384, 65535, 49152]
  }

  /// u8 RGB output — hand-derived expected values for Bt709 limited range.
  ///
  /// Pair 0 (neutral chroma, U=V=32768=bias → u_d=v_d=0 → chroma=0):
  ///   Y0=4096 = y_off → y_s=0 → clamp → R=G=B=0
  ///   Y1=32000 → y_s=127 → R=G=B=127
  /// Pair 1 (U=16384, V=49152 → u_d=-73, v_d=73 → r_c=115, g_c=-21, b_c=-135):
  ///   Y2=0 (below floor, y_s=-19) → R=clamp(-19+115,0,255)=96, G=0, B=0
  ///   Y3=65535 (above ceil, y_s=279) → R=255, G=255, B=clamp(279-135,0,255)=144
  #[test]
  fn y216_known_pattern_rgb() {
    let packed = test_input();
    let mut out = [0u8; 4 * 3];
    y216_to_rgb_or_rgba_row::<false>(&packed, &mut out, 4, ColorMatrix::Bt709, false);

    // Pixel 0: Y=4096 (limited-range black), neutral chroma → (0, 0, 0)
    assert_eq!(&out[0..3], &[0, 0, 0], "pixel 0 (Y=4096, neutral chroma)");
    // Pixel 1: Y=32000, neutral chroma → (127, 127, 127)
    assert_eq!(&out[3..6], &[127, 127, 127], "pixel 1 (Y=32000, neutral chroma)");
    // Pixel 2: Y=0 (below floor), U=16384, V=49152 → (96, 0, 0)
    assert_eq!(&out[6..9], &[96, 0, 0], "pixel 2 (Y=0, non-neutral chroma)");
    // Pixel 3: Y=65535 (above ceil), U=16384, V=49152 → (255, 255, 144)
    assert_eq!(&out[9..12], &[255, 255, 144], "pixel 3 (Y=65535, non-neutral chroma)");
  }

  /// RGBA output — same values as RGB plus alpha=0xFF at byte [3].
  #[test]
  fn y216_known_pattern_rgba() {
    let packed = test_input();
    let mut out = [0u8; 4 * 4];
    y216_to_rgb_or_rgba_row::<true>(&packed, &mut out, 4, ColorMatrix::Bt709, false);

    assert_eq!(&out[0..4], &[0, 0, 0, 0xFF]);
    assert_eq!(&out[4..8], &[127, 127, 127, 0xFF]);
    assert_eq!(&out[8..12], &[96, 0, 0, 0xFF]);
    assert_eq!(&out[12..16], &[255, 255, 144, 0xFF]);
  }

  /// u16 RGB output (i64 chroma path) — Bt709 limited range.
  ///
  /// Pair 0 neutral chroma:
  ///   Y=4096 (floor) → y_s=0 → R=G=B=0
  ///   Y=32000 → y_s=32618 → R=G=B=32618
  #[test]
  fn y216_known_pattern_rgb_u16() {
    let packed = test_input();
    let mut out = [0u16; 4 * 3];
    y216_to_rgb_u16_or_rgba_u16_row::<false>(&packed, &mut out, 4, ColorMatrix::Bt709, false);

    // Pixel 0: Y=4096 = limited-range floor → all channels 0
    assert_eq!(&out[0..3], &[0, 0, 0], "pixel 0 (Y=4096, neutral chroma, u16)");
    // Pixel 1: Y=32000 neutral chroma → 32618 on all channels
    assert_eq!(&out[3..6], &[32618, 32618, 32618], "pixel 1 (Y=32000, neutral chroma, u16)");
    // Pixel 2: non-neutral chroma — pixel-exact i64-path values
    assert_eq!(&out[6..9], &[24702_u16, 0, 0]);
    // Pixel 3
    assert_eq!(&out[9..12], &[65535_u16, 65535, 37073]);
  }

  /// u16 RGBA output — same color values, alpha=0xFFFF.
  #[test]
  fn y216_known_pattern_rgba_u16() {
    let packed = test_input();
    let mut out = [0u16; 4 * 4];
    y216_to_rgb_u16_or_rgba_u16_row::<true>(&packed, &mut out, 4, ColorMatrix::Bt709, false);

    assert_eq!(&out[0..4], &[0, 0, 0, 0xFFFF]);
    assert_eq!(&out[4..8], &[32618, 32618, 32618, 0xFFFF]);
    // Pixel 2: non-neutral chroma — pixel-exact i64-path values
    assert_eq!(&out[8..12], &[24702_u16, 0, 0, 0xFFFF]);
    // Pixel 3
    assert_eq!(&out[12..16], &[65535_u16, 65535, 37073, 0xFFFF]);
  }

  /// Luma u8: each Y extracted via `>> 8`. Input quadruple
  /// [0xAB12, 0x4444, 0xCD34, 0x5555] → luma [0xAB, 0xCD].
  #[test]
  fn y216_luma_extract() {
    let packed = [0xAB12u16, 0x4444, 0xCD34, 0x5555];
    let mut out = [0u8; 2];
    y216_to_luma_row(&packed, &mut out, 2);
    assert_eq!(out[0], 0xAB, "Y0 luma u8");
    assert_eq!(out[1], 0xCD, "Y1 luma u8");
  }

  /// Luma u16: each Y extracted directly (no shift). Input quadruple
  /// [0xAB12, 0x4444, 0xCD34, 0x5555] → luma [0xAB12, 0xCD34].
  #[test]
  fn y216_luma_u16_extract() {
    let packed = [0xAB12u16, 0x4444, 0xCD34, 0x5555];
    let mut out = [0u16; 2];
    y216_to_luma_u16_row(&packed, &mut out, 2);
    assert_eq!(out[0], 0xAB12, "Y0 luma u16");
    assert_eq!(out[1], 0xCD34, "Y1 luma u16");
  }
}
