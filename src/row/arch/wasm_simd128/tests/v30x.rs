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

/// Build a V30X packed buffer for `width` pixels where:
/// - Y[n] = n + 1 (10-bit value; luma_u16 output = n+1 directly)
/// - U[n] = 2n + 1 (10-bit value; one U per pixel — V30X is 4:4:4)
/// - V[n] = 512 (neutral 10-bit midpoint, bias-subtracted = 0)
///
/// V30X layout (per u32 word):
///   bits[31:22] = V (10-bit)
///   bits[21:12] = Y (10-bit)
///   bits[11:2]  = U (10-bit)
///   bits[1:0]   = padding (zero)
fn build_v30x_packed_y_n_plus_1_u_2n_plus_1_v_neutral(width: usize) -> std::vec::Vec<u32> {
  (0..width)
    .map(|n| {
      let y = (n as u32) + 1;
      let u = 2 * (n as u32) + 1;
      let v = 512u32;
      (v << 22) | (y << 12) | (u << 2)
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
fn wasm_simd128_v30x_lane_order_per_pixel_y_and_u() {
  const W: usize = 4;
  let packed = build_v30x_packed_y_n_plus_1_u_2n_plus_1_v_neutral(W);

  // Part 1: Luma natural-order (u16, no shift loss)
  let mut luma_out = std::vec![0u16; W];
  unsafe {
    v30x_to_luma_u16_row(&packed, &mut luma_out, W);
  }
  let expected_luma: std::vec::Vec<u16> = (1..=W as u16).collect();
  assert_eq!(
    luma_out, expected_luma,
    "wasm_simd128 v30x luma reorder bug"
  );

  // Part 2: SIMD vs scalar parity (catches U/Y channel swap bugs)
  let mut simd_rgb = std::vec![0u8; W * 3];
  let mut scalar_rgb = std::vec![0u8; W * 3];
  unsafe {
    v30x_to_rgb_or_rgba_row::<false>(&packed, &mut simd_rgb, W, crate::ColorMatrix::Bt709, false);
  }
  scalar::v30x_to_rgb_or_rgba_row::<false>(
    &packed,
    &mut scalar_rgb,
    W,
    crate::ColorMatrix::Bt709,
    false,
  );
  assert_eq!(
    simd_rgb, scalar_rgb,
    "wasm_simd128 v30x SIMD vs scalar diverges — lane-order bug"
  );
}
