use super::super::*;
use crate::{ColorMatrix, row::scalar};

/// Build a deterministic pseudo-random VUYA packed stream.
/// Returns `width * 4` bytes with channels varying across all 8-bit values.
fn pseudo_random_vuya(width: usize, seed: usize) -> std::vec::Vec<u8> {
  (0..width * 4)
    .map(|i| {
      let s = i.wrapping_mul(seed).wrapping_add(seed.wrapping_mul(3));
      (s & 0xFF) as u8
    })
    .collect()
}

fn check_rgb<const ALPHA: bool, const ALPHA_SRC: bool>(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let p = pseudo_random_vuya(width, 0xAA55);
  let bpp = if ALPHA { 4 } else { 3 };
  let mut s = std::vec![0u8; width * bpp];
  let mut k = std::vec![0u8; width * bpp];
  scalar::vuya_to_rgb_or_rgba_row::<ALPHA, ALPHA_SRC>(&p, &mut s, width, matrix, full_range);
  unsafe {
    vuya_to_rgb_or_rgba_row::<ALPHA, ALPHA_SRC>(&p, &mut k, width, matrix, full_range);
  }
  assert_eq!(
    s,
    k,
    "wasm vuya<ALPHA={ALPHA}, ALPHA_SRC={ALPHA_SRC}>→{} diverges (width={width}, matrix={matrix:?}, full_range={full_range})",
    if ALPHA { "RGBA" } else { "RGB" }
  );
}

fn check_luma(width: usize) {
  let p = pseudo_random_vuya(width, 0xC001);
  let mut s = std::vec![0u8; width];
  let mut k = std::vec![0u8; width];
  scalar::vuya_to_luma_row(&p, &mut s, width);
  unsafe {
    vuya_to_luma_row(&p, &mut k, width);
  }
  assert_eq!(s, k, "wasm vuya→luma diverges (width={width})");
}

// wasm has no runtime CPU detection — `simd128` is a compile-time
// feature, so no `is_*_feature_detected!` early-return guard. The
// `#[cfg_attr(miri, ignore)]` attribute is included for parity with
// other backends; miri does not currently target wasm32-wasip1.

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn wasm_vuya_matches_scalar_widths() {
  for w in [
    1usize, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 1920, 1921, 1923,
  ] {
    // All 3 valid (ALPHA, ALPHA_SRC) combinations.
    check_rgb::<false, false>(w, ColorMatrix::Bt709, false);
    check_rgb::<true, true>(w, ColorMatrix::Bt709, true);
    check_rgb::<true, false>(w, ColorMatrix::Bt2020Ncl, true);
    check_luma(w);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn wasm_vuya_rgb_matches_scalar_all_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      // All 3 valid (ALPHA, ALPHA_SRC) combinations.
      check_rgb::<false, false>(16, m, full);
      check_rgb::<true, true>(16, m, full);
      check_rgb::<true, false>(16, m, full);
    }
  }
}

/// Lane-order regression: encode Y[n] = n+1 for n in 0..16, assert
/// luma output matches natural order. Catches deinterleave permutation
/// bugs that solid-value tests would miss.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn wasm_vuya_luma_lane_order_per_pixel() {
  // Build 16 pixels with Y[n] = n+1, V=U=A=128.
  let mut packed = std::vec![0u8; 16 * 4];
  for n in 0..16usize {
    packed[n * 4] = 128; // V
    packed[n * 4 + 1] = 128; // U
    packed[n * 4 + 2] = (n + 1) as u8; // Y = n+1 (1..=16)
    packed[n * 4 + 3] = 128; // A
  }
  let mut luma = std::vec![0u8; 16];
  unsafe {
    vuya_to_luma_row(&packed, &mut luma, 16);
  }
  for n in 0..16usize {
    assert_eq!(
      luma[n],
      (n + 1) as u8,
      "wasm vuya luma lane order wrong at pixel {n}: expected {}, got {}",
      n + 1,
      luma[n]
    );
  }
}
