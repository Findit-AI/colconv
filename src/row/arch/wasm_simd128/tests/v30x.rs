use super::super::*;
use crate::{ColorMatrix, row::scalar};

/// Pack one V30X word from explicit U / Y / V samples.
fn pack_v30x(u: u32, y: u32, v: u32) -> u32 {
  debug_assert!(u < 1024 && y < 1024 && v < 1024);
  (v << 22) | (y << 12) | (u << 2)
}

/// Build a deterministic pseudo-random V30X packed row of `n` pixels.
/// Each word's U / Y / V fields are in `[0, 1023]`.
fn pseudo_random_v30x_words(n: usize, seed: usize) -> std::vec::Vec<u32> {
  (0..n)
    .map(|i| {
      let u = ((i * 37 + seed) & 0x3FF) as u32;
      let y = ((i * 53 + seed * 3) & 0x3FF) as u32;
      let v = ((i * 71 + seed * 7) & 0x3FF) as u32;
      pack_v30x(u, y, v)
    })
    .collect()
}

fn check_rgb(width: usize, matrix: ColorMatrix, full_range: bool) {
  let p = pseudo_random_v30x_words(width, 0xAA55);
  let mut s = std::vec![0u8; width * 3];
  let mut k = std::vec![0u8; width * 3];
  scalar::v30x_to_rgb_or_rgba_row::<false>(&p, &mut s, width, matrix, full_range);
  unsafe {
    v30x_to_rgb_or_rgba_row::<false>(&p, &mut k, width, matrix, full_range);
  }
  assert_eq!(
    s, k,
    "simd128 v30x→RGB diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_rgba(width: usize, matrix: ColorMatrix, full_range: bool) {
  let p = pseudo_random_v30x_words(width, 0xAA55);
  let mut s = std::vec![0u8; width * 4];
  let mut k = std::vec![0u8; width * 4];
  scalar::v30x_to_rgb_or_rgba_row::<true>(&p, &mut s, width, matrix, full_range);
  unsafe {
    v30x_to_rgb_or_rgba_row::<true>(&p, &mut k, width, matrix, full_range);
  }
  assert_eq!(
    s, k,
    "simd128 v30x→RGBA diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_rgb_u16(width: usize, matrix: ColorMatrix, full_range: bool) {
  let p = pseudo_random_v30x_words(width, 0xAA55);
  let mut s = std::vec![0u16; width * 3];
  let mut k = std::vec![0u16; width * 3];
  scalar::v30x_to_rgb_u16_or_rgba_u16_row::<false>(&p, &mut s, width, matrix, full_range);
  unsafe {
    v30x_to_rgb_u16_or_rgba_u16_row::<false>(&p, &mut k, width, matrix, full_range);
  }
  assert_eq!(
    s, k,
    "simd128 v30x→RGB u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_rgba_u16(width: usize, matrix: ColorMatrix, full_range: bool) {
  let p = pseudo_random_v30x_words(width, 0xAA55);
  let mut s = std::vec![0u16; width * 4];
  let mut k = std::vec![0u16; width * 4];
  scalar::v30x_to_rgb_u16_or_rgba_u16_row::<true>(&p, &mut s, width, matrix, full_range);
  unsafe {
    v30x_to_rgb_u16_or_rgba_u16_row::<true>(&p, &mut k, width, matrix, full_range);
  }
  assert_eq!(
    s, k,
    "simd128 v30x→RGBA u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_luma(width: usize) {
  let p = pseudo_random_v30x_words(width, 0xC001);
  let mut s = std::vec![0u8; width];
  let mut k = std::vec![0u8; width];
  scalar::v30x_to_luma_row(&p, &mut s, width);
  unsafe {
    v30x_to_luma_row(&p, &mut k, width);
  }
  assert_eq!(s, k, "simd128 v30x→luma diverges (width={width})");
}

fn check_luma_u16(width: usize) {
  let p = pseudo_random_v30x_words(width, 0xC001);
  let mut s = std::vec![0u16; width];
  let mut k = std::vec![0u16; width];
  scalar::v30x_to_luma_u16_row(&p, &mut s, width);
  unsafe {
    v30x_to_luma_u16_row(&p, &mut k, width);
  }
  assert_eq!(s, k, "simd128 v30x→luma u16 diverges (width={width})");
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
fn wasm_simd128_v30x_rgb_matches_scalar_all_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_rgb(4, m, full);
      check_rgba(4, m, full);
      check_rgb_u16(4, m, full);
      check_rgba_u16(4, m, full);
    }
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn wasm_simd128_v30x_matches_scalar_widths() {
  // Widths include: exact multiples of 4, multiples + tails of 1/2/3.
  for w in [
    1usize, 2, 3, 4, 5, 6, 7, 8, 9, 11, 16, 17, 31, 32, 33, 1280, 1920, 1921, 1922, 1923,
  ] {
    check_rgb(w, ColorMatrix::Bt709, false);
    check_rgba(w, ColorMatrix::Bt709, true);
    check_rgb_u16(w, ColorMatrix::Bt2020Ncl, true);
    check_rgba_u16(w, ColorMatrix::Bt601, false);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn wasm_simd128_v30x_luma_matches_scalar_widths() {
  for w in [
    1usize, 2, 3, 4, 5, 6, 7, 8, 9, 11, 16, 17, 31, 32, 33, 1280, 1920, 1921, 1922, 1923,
  ] {
    check_luma(w);
    check_luma_u16(w);
  }
}
