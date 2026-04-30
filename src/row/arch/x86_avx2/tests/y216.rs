use super::super::*;
use crate::{ColorMatrix, row::scalar};

/// Builds a deterministic pseudo-random Y216-shaped u16 buffer with
/// `width * 2` u16 samples (one quadruple = 4 u16 = 2 pixels). Each
/// u16 sample spans the full 16-bit range (no MSB alignment — Y216
/// uses full-range u16 unlike Y210/Y212).
fn pseudo_random_y216(width: usize, seed: usize) -> std::vec::Vec<u16> {
  (0..width * 2)
    .map(|i| ((i.wrapping_mul(seed).wrapping_add(seed * 3)) & 0xFFFF) as u16)
    .collect()
}

fn check_rgb(width: usize, matrix: ColorMatrix, full_range: bool) {
  let p = pseudo_random_y216(width, 0xAA55);
  let mut s = std::vec![0u8; width * 3];
  let mut k = std::vec![0u8; width * 3];
  scalar::y216_to_rgb_or_rgba_row::<false>(&p, &mut s, width, matrix, full_range);
  unsafe {
    y216_to_rgb_or_rgba_row::<false>(&p, &mut k, width, matrix, full_range);
  }
  assert_eq!(
    s, k,
    "AVX2 y216→RGB diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_rgb_u16(width: usize, matrix: ColorMatrix, full_range: bool) {
  let p = pseudo_random_y216(width, 0xAA55);
  let mut s = std::vec![0u16; width * 3];
  let mut k = std::vec![0u16; width * 3];
  scalar::y216_to_rgb_u16_or_rgba_u16_row::<false>(&p, &mut s, width, matrix, full_range);
  unsafe {
    y216_to_rgb_u16_or_rgba_u16_row::<false>(&p, &mut k, width, matrix, full_range);
  }
  assert_eq!(
    s, k,
    "AVX2 y216→RGB u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_luma(width: usize) {
  let p = pseudo_random_y216(width, 0xC001);
  let mut s = std::vec![0u8; width];
  let mut k = std::vec![0u8; width];
  scalar::y216_to_luma_row(&p, &mut s, width);
  unsafe {
    y216_to_luma_row(&p, &mut k, width);
  }
  assert_eq!(s, k, "AVX2 y216→luma u8 diverges (width={width})");
}

fn check_luma_u16(width: usize) {
  let p = pseudo_random_y216(width, 0xC001);
  let mut s = std::vec![0u16; width];
  let mut k = std::vec![0u16; width];
  scalar::y216_to_luma_u16_row(&p, &mut s, width);
  unsafe {
    y216_to_luma_u16_row(&p, &mut k, width);
  }
  assert_eq!(s, k, "AVX2 y216→luma u16 diverges (width={width})");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn avx2_y216_rgb_matches_scalar_all_matrices() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_rgb(32, m, full);
      check_rgb_u16(32, m, full);
    }
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn avx2_y216_matches_scalar_widths() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  // 16 = u16 natural block (falls straight through to scalar for u8 path).
  // 32 = u8 natural block. Widths < 16 fall to scalar for both paths.
  for w in [16usize, 18, 32, 34, 62, 64, 66, 128, 1920, 1922] {
    check_rgb(w, ColorMatrix::Bt709, false);
    check_rgb_u16(w, ColorMatrix::Bt2020Ncl, true);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn avx2_y216_luma_matches_scalar_widths() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [16usize, 18, 32, 34, 62, 64, 66, 128, 1920, 1922] {
    check_luma(w);
    check_luma_u16(w);
  }
}
