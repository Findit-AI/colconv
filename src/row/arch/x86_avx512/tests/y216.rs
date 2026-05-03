use super::super::*;
use crate::{ColorMatrix, row::scalar};

/// Builds a deterministic pseudo-random Y216-shaped u16 buffer with
/// `width * 2` u16 samples (one YUYV quadruple = 4 u16 = 2 pixels).
/// Each u16 spans the full 16-bit range — Y216 uses full-range u16
/// unlike Y210/Y212.
fn pseudo_random_y216(width: usize, seed: usize) -> std::vec::Vec<u16> {
  (0..width * 2)
    .map(|i| ((i.wrapping_mul(seed).wrapping_add(seed * 3)) & 0xFFFF) as u16)
    .collect()
}

fn check_rgb<const ALPHA: bool>(width: usize, matrix: ColorMatrix, full_range: bool) {
  let p = pseudo_random_y216(width, 0xAA55);
  let bpp = if ALPHA { 4 } else { 3 };
  let mut s = std::vec![0u8; width * bpp];
  let mut k = std::vec![0u8; width * bpp];
  scalar::y216_to_rgb_or_rgba_row::<ALPHA>(&p, &mut s, width, matrix, full_range);
  unsafe {
    y216_to_rgb_or_rgba_row::<ALPHA>(&p, &mut k, width, matrix, full_range);
  }
  assert_eq!(
    s,
    k,
    "AVX-512 y216<ALPHA={ALPHA}>→{} diverges (width={width}, matrix={matrix:?}, full_range={full_range})",
    if ALPHA { "RGBA" } else { "RGB" }
  );
}

fn check_rgb_u16<const ALPHA: bool>(width: usize, matrix: ColorMatrix, full_range: bool) {
  let p = pseudo_random_y216(width, 0xAA55);
  let bpp = if ALPHA { 4 } else { 3 };
  let mut s = std::vec![0u16; width * bpp];
  let mut k = std::vec![0u16; width * bpp];
  scalar::y216_to_rgb_u16_or_rgba_u16_row::<ALPHA>(&p, &mut s, width, matrix, full_range);
  unsafe {
    y216_to_rgb_u16_or_rgba_u16_row::<ALPHA>(&p, &mut k, width, matrix, full_range);
  }
  assert_eq!(
    s,
    k,
    "AVX-512 y216<ALPHA={ALPHA}>→{} u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})",
    if ALPHA { "RGBA" } else { "RGB" }
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
  assert_eq!(s, k, "AVX-512 y216→luma u8 diverges (width={width})");
}

fn check_luma_u16(width: usize) {
  let p = pseudo_random_y216(width, 0xC001);
  let mut s = std::vec![0u16; width];
  let mut k = std::vec![0u16; width];
  scalar::y216_to_luma_u16_row(&p, &mut s, width);
  unsafe {
    y216_to_luma_u16_row(&p, &mut k, width);
  }
  assert_eq!(s, k, "AVX-512 y216→luma u16 diverges (width={width})");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn avx512_y216_rgb_matches_scalar_all_matrices() {
  if !std::arch::is_x86_feature_detected!("avx512f")
    || !std::arch::is_x86_feature_detected!("avx512bw")
  {
    return;
  }
  // Width 64 = one full u8-path iteration; 32 = one full u16-path iter.
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_rgb::<false>(64, m, full);
      check_rgb::<true>(64, m, full);
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
fn avx512_y216_matches_scalar_widths() {
  if !std::arch::is_x86_feature_detected!("avx512f")
    || !std::arch::is_x86_feature_detected!("avx512bw")
  {
    return;
  }
  // u8 path: 64 = one block; u16 path: 32 = one block. Widths exercise
  // main loop and scalar tail.
  for w in [32usize, 34, 64, 66, 128, 130, 1920, 1952] {
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
fn avx512_y216_luma_matches_scalar_widths() {
  if !std::arch::is_x86_feature_detected!("avx512f")
    || !std::arch::is_x86_feature_detected!("avx512bw")
  {
    return;
  }
  for w in [32usize, 34, 64, 66, 128, 130, 1920, 1952] {
    check_luma(w);
    check_luma_u16(w);
  }
}

/// Multi-channel lane-order regression — encodes pixel index in
/// BOTH Y AND U so we catch per-channel asymmetric mask bugs that
/// a Y-only test would miss. Pattern from Ship 12d AYUV64 backport.
///
/// Y216 is 16-bit native (no MSB shift) — exercises the i64 chroma
/// path at u16 RGB output.
///
/// Y216 layout: YUYV-shape u16x2 `[Y0, U, Y1, V]` per 2 pixels.
/// - `Y[n] = (n + 1) as u16` (16-bit native, no shift)
/// - `U[k] = (2k + 1) as u16` (one U per pair)
/// - `V = 0x8000` (neutral u16 midpoint)
///
/// AVX-512 u16-RGB path threshold: 32 px/iter. W=64 covers 2 full iterations.
fn build_y216_packed_y_n_plus_1_u_2k_plus_1_v_neutral(width: usize) -> std::vec::Vec<u16> {
  let mut packed = std::vec![0u16; width * 2];
  for k in 0..(width / 2) {
    let y0 = (2 * k as u16) + 1;
    let y1 = (2 * k as u16) + 2;
    let u = (2 * k as u16) + 1;
    packed[k * 4] = y0;
    packed[k * 4 + 1] = u;
    packed[k * 4 + 2] = y1;
    packed[k * 4 + 3] = 0x8000; // V neutral
  }
  packed
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn avx512_y216_lane_order_per_pixel_y_and_u() {
  if !std::arch::is_x86_feature_detected!("avx512f")
    || !std::arch::is_x86_feature_detected!("avx512bw")
  {
    return;
  }
  const W: usize = 64;
  let packed = build_y216_packed_y_n_plus_1_u_2k_plus_1_v_neutral(W);

  // Part 1: Luma natural-order at u16
  let mut luma_u16 = std::vec![0u16; W];
  unsafe {
    y216_to_luma_u16_row(&packed, &mut luma_u16, W);
  }
  let expected_luma: std::vec::Vec<u16> = (1..=W as u16).collect();
  assert_eq!(luma_u16, expected_luma, "AVX-512 y216 luma_u16 reorder bug");

  // Part 2: SIMD vs scalar parity at u16 RGB (i64 chroma path)
  let mut simd_rgb = std::vec![0u16; W * 3];
  let mut scalar_rgb = std::vec![0u16; W * 3];
  unsafe {
    y216_to_rgb_u16_or_rgba_u16_row::<false>(&packed, &mut simd_rgb, W, ColorMatrix::Bt709, false);
  }
  scalar::y216_to_rgb_u16_or_rgba_u16_row::<false>(
    &packed,
    &mut scalar_rgb,
    W,
    ColorMatrix::Bt709,
    false,
  );
  assert_eq!(
    simd_rgb, scalar_rgb,
    "AVX-512 y216 SIMD vs scalar diverges (u16 RGB, i64 chroma)"
  );
}
