use super::super::*;
use crate::{ColorMatrix, row::scalar};

/// Pack one V410 word from explicit U / Y / V samples.
fn pack_v410(u: u32, y: u32, v: u32) -> u32 {
  debug_assert!(u < 1024 && y < 1024 && v < 1024);
  (v << 20) | (y << 10) | u
}

/// Builds a deterministic pseudo-random V410 buffer with `n` pixels.
/// Each u32 word packs three 10-bit fields in [0, 1023].
fn pseudo_random_v410_words(n: usize, seed: usize) -> std::vec::Vec<u32> {
  (0..n)
    .map(|i| {
      let u = ((i * 37 + seed) & 0x3FF) as u32;
      let y = ((i * 53 + seed * 3) & 0x3FF) as u32;
      let v = ((i * 71 + seed * 7) & 0x3FF) as u32;
      pack_v410(u, y, v)
    })
    .collect()
}

fn check_rgb(width: usize, matrix: ColorMatrix, full_range: bool) {
  let p = pseudo_random_v410_words(width, 0xAA55);
  let mut s = std::vec![0u8; width * 3];
  let mut k = std::vec![0u8; width * 3];
  scalar::v410_to_rgb_or_rgba_row::<false>(&p, &mut s, width, matrix, full_range);
  unsafe {
    v410_to_rgb_or_rgba_row::<false>(&p, &mut k, width, matrix, full_range);
  }
  assert_eq!(
    s, k,
    "AVX-512 v410→RGB diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_rgba(width: usize, matrix: ColorMatrix, full_range: bool) {
  let p = pseudo_random_v410_words(width, 0xAA55);
  let mut s = std::vec![0u8; width * 4];
  let mut k = std::vec![0u8; width * 4];
  scalar::v410_to_rgb_or_rgba_row::<true>(&p, &mut s, width, matrix, full_range);
  unsafe {
    v410_to_rgb_or_rgba_row::<true>(&p, &mut k, width, matrix, full_range);
  }
  assert_eq!(
    s, k,
    "AVX-512 v410→RGBA diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_rgb_u16(width: usize, matrix: ColorMatrix, full_range: bool) {
  let p = pseudo_random_v410_words(width, 0xAA55);
  let mut s = std::vec![0u16; width * 3];
  let mut k = std::vec![0u16; width * 3];
  scalar::v410_to_rgb_u16_or_rgba_u16_row::<false>(&p, &mut s, width, matrix, full_range);
  unsafe {
    v410_to_rgb_u16_or_rgba_u16_row::<false>(&p, &mut k, width, matrix, full_range);
  }
  assert_eq!(
    s, k,
    "AVX-512 v410→RGB u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_rgba_u16(width: usize, matrix: ColorMatrix, full_range: bool) {
  let p = pseudo_random_v410_words(width, 0xAA55);
  let mut s = std::vec![0u16; width * 4];
  let mut k = std::vec![0u16; width * 4];
  scalar::v410_to_rgb_u16_or_rgba_u16_row::<true>(&p, &mut s, width, matrix, full_range);
  unsafe {
    v410_to_rgb_u16_or_rgba_u16_row::<true>(&p, &mut k, width, matrix, full_range);
  }
  assert_eq!(
    s, k,
    "AVX-512 v410→RGBA u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_luma(width: usize) {
  let p = pseudo_random_v410_words(width, 0xC001);
  let mut s = std::vec![0u8; width];
  let mut k = std::vec![0u8; width];
  scalar::v410_to_luma_row(&p, &mut s, width);
  unsafe {
    v410_to_luma_row(&p, &mut k, width);
  }
  assert_eq!(s, k, "AVX-512 v410→luma diverges (width={width})");
}

fn check_luma_u16(width: usize) {
  let p = pseudo_random_v410_words(width, 0xC001);
  let mut s = std::vec![0u16; width];
  let mut k = std::vec![0u16; width];
  scalar::v410_to_luma_u16_row(&p, &mut s, width);
  unsafe {
    v410_to_luma_u16_row(&p, &mut k, width);
  }
  assert_eq!(s, k, "AVX-512 v410→luma u16 diverges (width={width})");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn avx512_v410_rgb_matches_scalar_all_matrices() {
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
      // Width 16 = exactly one main-loop iteration (no tail).
      check_rgb(16, m, full);
      check_rgba(16, m, full);
      check_rgb_u16(16, m, full);
      check_rgba_u16(16, m, full);
    }
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn avx512_v410_matches_scalar_widths() {
  if !std::arch::is_x86_feature_detected!("avx512f")
    || !std::arch::is_x86_feature_detected!("avx512bw")
  {
    return;
  }
  // Widths cover: pure scalar tail (< 16), exactly one block (16),
  // main-loop + tail (e.g. 17, 31), multiple blocks (32, 48),
  // and production sizes (1280, 1920, 1921, 1937).
  for w in [
    1usize, 4, 8, 15, 16, 17, 31, 32, 48, 64, 1280, 1920, 1921, 1937,
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
fn avx512_v410_luma_matches_scalar_widths() {
  if !std::arch::is_x86_feature_detected!("avx512f")
    || !std::arch::is_x86_feature_detected!("avx512bw")
  {
    return;
  }
  for w in [
    1usize, 4, 8, 15, 16, 17, 31, 32, 48, 64, 1280, 1920, 1921, 1937,
  ] {
    check_luma(w);
    check_luma_u16(w);
  }
}

/// Build a V410 packed buffer for `width` pixels where:
/// - Y[n] = n + 1 (10-bit value; luma_u16 output = n+1 directly)
/// - U[n] = 2n + 1 (10-bit value; one U per pixel — V410 is 4:4:4)
/// - V[n] = 512 (neutral 10-bit midpoint, bias-subtracted = 0)
///
/// V410 layout (per u32 word):
///   bits[29:20] = V (10-bit)
///   bits[19:10] = Y (10-bit)
///   bits[9:0]   = U (10-bit)
///   bits[31:30] = padding (zero)
fn build_v410_packed_y_n_plus_1_u_2n_plus_1_v_neutral(width: usize) -> std::vec::Vec<u32> {
  (0..width)
    .map(|n| {
      let y = (n as u32) + 1;
      let u = 2 * (n as u32) + 1;
      let v = 512u32;
      (v << 20) | (y << 10) | u
    })
    .collect()
}

/// Multi-channel lane-order regression — encodes pixel index in BOTH Y AND U
/// so we catch per-channel asymmetric mask bugs that a Y-only test would miss.
/// Pattern from Ship 12d AYUV64 backport.
///
/// Asserts:
/// - `luma_u16_row` output = [1..=W] (Y values direct, no shift)
/// - SIMD RGB output == scalar RGB output (any lane-order bug diverges)
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn avx512_v410_lane_order_per_pixel_y_and_u() {
  if !std::arch::is_x86_feature_detected!("avx512f")
    || !std::arch::is_x86_feature_detected!("avx512bw")
  {
    return;
  }
  const W: usize = 32;
  let packed = build_v410_packed_y_n_plus_1_u_2n_plus_1_v_neutral(W);

  // Part 1: Luma natural-order (u16, no shift loss)
  let mut luma = std::vec![0u16; W];
  unsafe {
    v410_to_luma_u16_row(&packed, &mut luma, W);
  }
  let expected_luma: std::vec::Vec<u16> = (1..=W as u16).collect();
  assert_eq!(luma, expected_luma, "avx512 v410 luma reorder bug");

  // Part 2: SIMD vs scalar parity (catches U/Y channel swap bugs)
  let mut simd_rgb = std::vec![0u8; W * 3];
  let mut scalar_rgb = std::vec![0u8; W * 3];
  unsafe {
    v410_to_rgb_or_rgba_row::<false>(&packed, &mut simd_rgb, W, crate::ColorMatrix::Bt709, false);
  }
  scalar::v410_to_rgb_or_rgba_row::<false>(
    &packed,
    &mut scalar_rgb,
    W,
    crate::ColorMatrix::Bt709,
    false,
  );
  assert_eq!(
    simd_rgb, scalar_rgb,
    "avx512 v410 SIMD vs scalar diverges — lane-order bug"
  );
}
