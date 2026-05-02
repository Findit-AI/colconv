use super::super::*;
use crate::{ColorMatrix, row::scalar};

/// Build a deterministic pseudo-random XV36 quadruple stream.
/// Each channel is a 12-bit value MSB-aligned in u16 (low 4 bits zero).
fn pseudo_random_xv36(width: usize, seed: usize) -> std::vec::Vec<u16> {
  (0..width * 4)
    .map(|i| {
      let s = i.wrapping_mul(seed).wrapping_add(seed * 3);
      ((s & 0xFFF) << 4) as u16 // 12-bit value, MSB-aligned
    })
    .collect()
}

fn check_rgb<const ALPHA: bool>(width: usize, matrix: ColorMatrix, full_range: bool) {
  let p = pseudo_random_xv36(width, 0xAA55);
  let bpp = if ALPHA { 4 } else { 3 };
  let mut s = std::vec![0u8; width * bpp];
  let mut k = std::vec![0u8; width * bpp];
  scalar::xv36_to_rgb_or_rgba_row::<ALPHA>(&p, &mut s, width, matrix, full_range);
  unsafe {
    xv36_to_rgb_or_rgba_row::<ALPHA>(&p, &mut k, width, matrix, full_range);
  }
  assert_eq!(
    s,
    k,
    "NEON xv36<ALPHA={ALPHA}>→{} diverges (width={width}, matrix={matrix:?}, full_range={full_range})",
    if ALPHA { "RGBA" } else { "RGB" }
  );
}

fn check_rgb_u16<const ALPHA: bool>(width: usize, matrix: ColorMatrix, full_range: bool) {
  let p = pseudo_random_xv36(width, 0xAA55);
  let bpp = if ALPHA { 4 } else { 3 };
  let mut s = std::vec![0u16; width * bpp];
  let mut k = std::vec![0u16; width * bpp];
  scalar::xv36_to_rgb_u16_or_rgba_u16_row::<ALPHA>(&p, &mut s, width, matrix, full_range);
  unsafe {
    xv36_to_rgb_u16_or_rgba_u16_row::<ALPHA>(&p, &mut k, width, matrix, full_range);
  }
  assert_eq!(
    s,
    k,
    "NEON xv36<ALPHA={ALPHA}>→{} u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})",
    if ALPHA { "RGBA" } else { "RGB" }
  );
}

fn check_luma(width: usize) {
  let p = pseudo_random_xv36(width, 0xC001);
  let mut s = std::vec![0u8; width];
  let mut k = std::vec![0u8; width];
  scalar::xv36_to_luma_row(&p, &mut s, width);
  unsafe {
    xv36_to_luma_row(&p, &mut k, width);
  }
  assert_eq!(s, k, "NEON xv36→luma diverges (width={width})");
}

fn check_luma_u16(width: usize) {
  let p = pseudo_random_xv36(width, 0xC001);
  let mut s = std::vec![0u16; width];
  let mut k = std::vec![0u16; width];
  scalar::xv36_to_luma_u16_row(&p, &mut s, width);
  unsafe {
    xv36_to_luma_u16_row(&p, &mut k, width);
  }
  assert_eq!(s, k, "NEON xv36→luma u16 diverges (width={width})");
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_xv36_rgb_matches_scalar_all_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_rgb::<false>(8, m, full);
      check_rgb::<true>(8, m, full);
      check_rgb_u16::<false>(8, m, full);
      check_rgb_u16::<true>(8, m, full);
    }
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_xv36_matches_scalar_widths() {
  for w in [1usize, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 1920, 1921, 1923] {
    check_rgb::<false>(w, ColorMatrix::Bt709, false);
    check_rgb::<true>(w, ColorMatrix::Bt709, true);
    check_rgb_u16::<false>(w, ColorMatrix::Bt2020Ncl, true);
    check_rgb_u16::<true>(w, ColorMatrix::Bt601, false);
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_xv36_luma_matches_scalar_widths() {
  for w in [1usize, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 1920, 1921, 1923] {
    check_luma(w);
    check_luma_u16(w);
  }
}
