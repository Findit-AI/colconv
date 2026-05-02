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
    "AVX-512 xv36<ALPHA={ALPHA}>→{} diverges (width={width}, matrix={matrix:?}, full_range={full_range})",
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
    "AVX-512 xv36<ALPHA={ALPHA}>→{} u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})",
    if ALPHA { "RGBA u16" } else { "RGB u16" }
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
  assert_eq!(s, k, "AVX-512 xv36→luma diverges (width={width})");
}

fn check_luma_u16(width: usize) {
  let p = pseudo_random_xv36(width, 0xC001);
  let mut s = std::vec![0u16; width];
  let mut k = std::vec![0u16; width];
  scalar::xv36_to_luma_u16_row(&p, &mut s, width);
  unsafe {
    xv36_to_luma_u16_row(&p, &mut k, width);
  }
  assert_eq!(s, k, "AVX-512 xv36→luma u16 diverges (width={width})");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn avx512_xv36_rgb_matches_scalar_all_matrices() {
  if !std::arch::is_x86_feature_detected!("avx512f")
    || !std::arch::is_x86_feature_detected!("avx512bw")
  {
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
      // Width 32 = exactly one main-loop iteration (no tail).
      check_rgb::<false>(32, m, full);
      check_rgb::<true>(32, m, full);
      check_rgb_u16::<false>(32, m, full);
      check_rgb_u16::<true>(32, m, full);
    }
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn avx512_xv36_matches_scalar_widths() {
  if !std::arch::is_x86_feature_detected!("avx512f")
    || !std::arch::is_x86_feature_detected!("avx512bw")
  {
    return;
  }
  // Widths cover: pure scalar tail (< 32), exactly one block (32),
  // main-loop + tail (e.g. 33, 63), multiple blocks (64, 96),
  // and production sizes (1280, 1920, 1921, 1937).
  for w in [
    1usize, 4, 8, 15, 16, 31, 32, 33, 63, 64, 96, 1280, 1920, 1921, 1937,
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
fn avx512_xv36_luma_matches_scalar_widths() {
  if !std::arch::is_x86_feature_detected!("avx512f")
    || !std::arch::is_x86_feature_detected!("avx512bw")
  {
    return;
  }
  for w in [
    1usize, 4, 8, 15, 16, 31, 32, 33, 63, 64, 96, 1280, 1920, 1921, 1937,
  ] {
    check_luma(w);
    check_luma_u16(w);
  }
}
