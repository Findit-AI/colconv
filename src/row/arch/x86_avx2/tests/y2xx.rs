use super::super::*;
use crate::{ColorMatrix, row::scalar};

/// Builds a deterministic pseudo-random Y210-shaped u16 buffer with
/// `width * 2` u16 samples (one quadruple = 4 u16 = 2 pixels). Each
/// u16 sample has 10 active bits sitting in the high bits, low 6
/// bits zero (matches Y210's MSB-aligned encoding).
fn pseudo_random_y210(width: usize, seed: usize) -> std::vec::Vec<u16> {
  (0..width * 2)
    .map(|i| {
      let s = ((i.wrapping_mul(seed).wrapping_add(seed * 3)) & 0x3FF) as u16;
      s << 6
    })
    .collect()
}

fn check_rgb<const BITS: u32>(width: usize, matrix: ColorMatrix, full_range: bool) {
  let p = pseudo_random_y210(width, 0xAA55);
  let mut s = std::vec![0u8; width * 3];
  let mut k = std::vec![0u8; width * 3];
  scalar::y2xx_n_to_rgb_or_rgba_row::<BITS, false>(&p, &mut s, width, matrix, full_range);
  unsafe {
    y2xx_n_to_rgb_or_rgba_row::<BITS, false>(&p, &mut k, width, matrix, full_range);
  }
  assert_eq!(
    s, k,
    "AVX2 y2xx<{BITS}>→RGB diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_rgba<const BITS: u32>(width: usize, matrix: ColorMatrix, full_range: bool) {
  let p = pseudo_random_y210(width, 0xAA55);
  let mut s = std::vec![0u8; width * 4];
  let mut k = std::vec![0u8; width * 4];
  scalar::y2xx_n_to_rgb_or_rgba_row::<BITS, true>(&p, &mut s, width, matrix, full_range);
  unsafe {
    y2xx_n_to_rgb_or_rgba_row::<BITS, true>(&p, &mut k, width, matrix, full_range);
  }
  assert_eq!(
    s, k,
    "AVX2 y2xx<{BITS}>→RGBA diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_rgb_u16<const BITS: u32>(width: usize, matrix: ColorMatrix, full_range: bool) {
  let p = pseudo_random_y210(width, 0xAA55);
  let mut s = std::vec![0u16; width * 3];
  let mut k = std::vec![0u16; width * 3];
  scalar::y2xx_n_to_rgb_u16_or_rgba_u16_row::<BITS, false>(&p, &mut s, width, matrix, full_range);
  unsafe {
    y2xx_n_to_rgb_u16_or_rgba_u16_row::<BITS, false>(&p, &mut k, width, matrix, full_range);
  }
  assert_eq!(
    s, k,
    "AVX2 y2xx<{BITS}>→RGB u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_rgba_u16<const BITS: u32>(width: usize, matrix: ColorMatrix, full_range: bool) {
  let p = pseudo_random_y210(width, 0xAA55);
  let mut s = std::vec![0u16; width * 4];
  let mut k = std::vec![0u16; width * 4];
  scalar::y2xx_n_to_rgb_u16_or_rgba_u16_row::<BITS, true>(&p, &mut s, width, matrix, full_range);
  unsafe {
    y2xx_n_to_rgb_u16_or_rgba_u16_row::<BITS, true>(&p, &mut k, width, matrix, full_range);
  }
  assert_eq!(
    s, k,
    "AVX2 y2xx<{BITS}>→RGBA u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_luma<const BITS: u32>(width: usize) {
  let p = pseudo_random_y210(width, 0xC001);
  let mut s = std::vec![0u8; width];
  let mut k = std::vec![0u8; width];
  scalar::y2xx_n_to_luma_row::<BITS>(&p, &mut s, width);
  unsafe {
    y2xx_n_to_luma_row::<BITS>(&p, &mut k, width);
  }
  assert_eq!(s, k, "AVX2 y2xx<{BITS}>→luma diverges (width={width})");
}

fn check_luma_u16<const BITS: u32>(width: usize) {
  let p = pseudo_random_y210(width, 0xC001);
  let mut s = std::vec![0u16; width];
  let mut k = std::vec![0u16; width];
  scalar::y2xx_n_to_luma_u16_row::<BITS>(&p, &mut s, width);
  unsafe {
    y2xx_n_to_luma_u16_row::<BITS>(&p, &mut k, width);
  }
  assert_eq!(s, k, "AVX2 y2xx<{BITS}>→luma u16 diverges (width={width})");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn avx2_y210_rgb_matches_scalar_all_matrices() {
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
      check_rgb::<10>(32, m, full);
      check_rgba::<10>(32, m, full);
      check_rgb_u16::<10>(32, m, full);
      check_rgba_u16::<10>(32, m, full);
    }
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn avx2_y210_matches_scalar_widths() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  // 16 = AVX2 natural block; 1922 forces tail.
  for w in [2usize, 4, 14, 16, 18, 32, 34, 62, 64, 66, 128, 1920, 1922] {
    check_rgb::<10>(w, ColorMatrix::Bt709, false);
    check_rgba::<10>(w, ColorMatrix::Bt709, true);
    check_rgb_u16::<10>(w, ColorMatrix::Bt2020Ncl, true);
    check_rgba_u16::<10>(w, ColorMatrix::Bt601, false);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn avx2_y210_luma_matches_scalar_widths() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [2usize, 4, 14, 16, 18, 32, 34, 62, 64, 66, 128, 1920, 1922] {
    check_luma::<10>(w);
    check_luma_u16::<10>(w);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn avx2_y212_matches_scalar_widths() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  // Use width-12 BITS pseudo-random generator: shift by 4 instead of 6.
  fn pseudo_random_y212(width: usize, seed: usize) -> std::vec::Vec<u16> {
    (0..width * 2)
      .map(|i| {
        let s = ((i.wrapping_mul(seed).wrapping_add(seed * 3)) & 0xFFF) as u16;
        s << 4
      })
      .collect()
  }
  for w in [16usize, 18, 32, 34, 62, 64, 66, 128, 1920, 1922] {
    let p = pseudo_random_y212(w, 0xAA55);
    let mut s = std::vec![0u8; w * 3];
    let mut k = std::vec![0u8; w * 3];
    scalar::y2xx_n_to_rgb_or_rgba_row::<12, false>(&p, &mut s, w, ColorMatrix::Bt709, false);
    unsafe {
      y2xx_n_to_rgb_or_rgba_row::<12, false>(&p, &mut k, w, ColorMatrix::Bt709, false);
    }
    assert_eq!(s, k, "AVX2 y2xx<12>→RGB diverges (width={w})");

    let mut s_u16 = std::vec![0u16; w * 4];
    let mut k_u16 = std::vec![0u16; w * 4];
    scalar::y2xx_n_to_rgb_u16_or_rgba_u16_row::<12, true>(
      &p,
      &mut s_u16,
      w,
      ColorMatrix::Bt2020Ncl,
      true,
    );
    unsafe {
      y2xx_n_to_rgb_u16_or_rgba_u16_row::<12, true>(
        &p,
        &mut k_u16,
        w,
        ColorMatrix::Bt2020Ncl,
        true,
      );
    }
    assert_eq!(s_u16, k_u16, "AVX2 y2xx<12>→RGBA u16 diverges (width={w})");

    let mut sl = std::vec![0u8; w];
    let mut kl = std::vec![0u8; w];
    scalar::y2xx_n_to_luma_row::<12>(&p, &mut sl, w);
    unsafe {
      y2xx_n_to_luma_row::<12>(&p, &mut kl, w);
    }
    assert_eq!(sl, kl, "AVX2 y2xx<12>→luma diverges (width={w})");

    let mut slu = std::vec![0u16; w];
    let mut klu = std::vec![0u16; w];
    scalar::y2xx_n_to_luma_u16_row::<12>(&p, &mut slu, w);
    unsafe {
      y2xx_n_to_luma_u16_row::<12>(&p, &mut klu, w);
    }
    assert_eq!(slu, klu, "AVX2 y2xx<12>→luma u16 diverges (width={w})");
  }
}
