use super::super::*;
use crate::{ColorMatrix, row::scalar};

/// Pack one V410 word: `(v << 20) | (y << 10) | u` with each channel
/// masked to 10-bit range.
fn pack_v410(u: u32, y: u32, v: u32) -> u32 {
  let u = u & 0x3FF;
  let y = y & 0x3FF;
  let v = v & 0x3FF;
  (v << 20) | (y << 10) | u
}

/// Builds a deterministic pseudo-random V410 packed buffer of `width` words.
fn pseudo_random_v410(width: usize, seed: usize) -> std::vec::Vec<u32> {
  (0..width)
    .map(|i| {
      let u = (i.wrapping_mul(seed).wrapping_add(seed * 3)) & 0x3FF;
      let y = (i.wrapping_mul(seed * 5).wrapping_add(seed * 7)) & 0x3FF;
      let v = (i.wrapping_mul(seed * 11).wrapping_add(seed * 13)) & 0x3FF;
      pack_v410(u as u32, y as u32, v as u32)
    })
    .collect()
}

fn check_rgb<const ALPHA: bool>(width: usize, matrix: ColorMatrix, full_range: bool) {
  let p = pseudo_random_v410(width, 0xAA55);
  let bpp = if ALPHA { 4 } else { 3 };
  let mut s = std::vec![0u8; width * bpp];
  let mut k = std::vec![0u8; width * bpp];
  scalar::v410_to_rgb_or_rgba_row::<ALPHA>(&p, &mut s, width, matrix, full_range);
  unsafe {
    v410_to_rgb_or_rgba_row::<ALPHA>(&p, &mut k, width, matrix, full_range);
  }
  assert_eq!(
    s,
    k,
    "AVX2 v410→{} diverges (width={width}, matrix={matrix:?}, full_range={full_range})",
    if ALPHA { "RGBA" } else { "RGB" }
  );
}

fn check_rgb_u16<const ALPHA: bool>(width: usize, matrix: ColorMatrix, full_range: bool) {
  let p = pseudo_random_v410(width, 0xAA55);
  let bpp = if ALPHA { 4 } else { 3 };
  let mut s = std::vec![0u16; width * bpp];
  let mut k = std::vec![0u16; width * bpp];
  scalar::v410_to_rgb_u16_or_rgba_u16_row::<ALPHA>(&p, &mut s, width, matrix, full_range);
  unsafe {
    v410_to_rgb_u16_or_rgba_u16_row::<ALPHA>(&p, &mut k, width, matrix, full_range);
  }
  assert_eq!(
    s,
    k,
    "AVX2 v410→{} u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})",
    if ALPHA { "RGBA u16" } else { "RGB u16" }
  );
}

fn check_luma(width: usize) {
  let p = pseudo_random_v410(width, 0xC001);
  let mut s = std::vec![0u8; width];
  let mut k = std::vec![0u8; width];
  scalar::v410_to_luma_row(&p, &mut s, width);
  unsafe {
    v410_to_luma_row(&p, &mut k, width);
  }
  assert_eq!(s, k, "AVX2 v410→luma diverges (width={width})");
}

fn check_luma_u16(width: usize) {
  let p = pseudo_random_v410(width, 0xC001);
  let mut s = std::vec![0u16; width];
  let mut k = std::vec![0u16; width];
  scalar::v410_to_luma_u16_row(&p, &mut s, width);
  unsafe {
    v410_to_luma_u16_row(&p, &mut k, width);
  }
  assert_eq!(s, k, "AVX2 v410→luma u16 diverges (width={width})");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn avx2_v410_rgb_matches_scalar_all_matrices() {
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
      // Width 16 = two main-loop iterations (no tail).
      check_rgb::<false>(16, m, full);
      check_rgb::<true>(16, m, full);
      check_rgb_u16::<false>(16, m, full);
      check_rgb_u16::<true>(16, m, full);
    }
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn avx2_v410_matches_scalar_widths() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  // Includes widths with SIMD main loop (multiples of 8), scalar tails
  // (1..7 — <8 pixels, no main loop), and large production widths
  // (1920p, 1921 = 1920+1 tail, 1923 = 1920+3 tail).
  for w in [
    1usize, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 1920, 1921, 1923,
  ] {
    check_rgb::<false>(w, ColorMatrix::Bt709, false);
    check_rgb::<true>(w, ColorMatrix::Bt709, true);
    check_rgb_u16::<false>(w, ColorMatrix::Bt2020Ncl, true);
    check_rgb_u16::<true>(w, ColorMatrix::Bt601, false);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn avx2_v410_luma_matches_scalar_widths() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [
    1usize, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 1920, 1921, 1923,
  ] {
    check_luma(w);
    check_luma_u16(w);
  }
}
