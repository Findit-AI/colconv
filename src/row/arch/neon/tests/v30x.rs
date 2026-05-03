use super::super::*;
use crate::{ColorMatrix, row::scalar};

/// Pack one V30X word: `(v << 22) | (y << 12) | (u << 2)` with each channel
/// masked to 10-bit range.
fn pack_v30x(u: u32, y: u32, v: u32) -> u32 {
  let u = u & 0x3FF;
  let y = y & 0x3FF;
  let v = v & 0x3FF;
  (v << 22) | (y << 12) | (u << 2)
}

/// Builds a deterministic pseudo-random V30X packed buffer of `width` words.
fn pseudo_random_v30x(width: usize, seed: usize) -> std::vec::Vec<u32> {
  (0..width)
    .map(|i| {
      let u = (i.wrapping_mul(seed).wrapping_add(seed * 3)) & 0x3FF;
      let y = (i.wrapping_mul(seed * 5).wrapping_add(seed * 7)) & 0x3FF;
      let v = (i.wrapping_mul(seed * 11).wrapping_add(seed * 13)) & 0x3FF;
      pack_v30x(u as u32, y as u32, v as u32)
    })
    .collect()
}

fn check_rgb<const ALPHA: bool>(width: usize, matrix: ColorMatrix, full_range: bool) {
  let p = pseudo_random_v30x(width, 0xAA55);
  let bpp = if ALPHA { 4 } else { 3 };
  let mut s = std::vec![0u8; width * bpp];
  let mut k = std::vec![0u8; width * bpp];
  scalar::v30x_to_rgb_or_rgba_row::<ALPHA>(&p, &mut s, width, matrix, full_range);
  unsafe {
    v30x_to_rgb_or_rgba_row::<ALPHA>(&p, &mut k, width, matrix, full_range);
  }
  assert_eq!(
    s,
    k,
    "NEON v30x→{} diverges (width={width}, matrix={matrix:?}, full_range={full_range})",
    if ALPHA { "RGBA" } else { "RGB" }
  );
}

fn check_rgb_u16<const ALPHA: bool>(width: usize, matrix: ColorMatrix, full_range: bool) {
  let p = pseudo_random_v30x(width, 0xAA55);
  let bpp = if ALPHA { 4 } else { 3 };
  let mut s = std::vec![0u16; width * bpp];
  let mut k = std::vec![0u16; width * bpp];
  scalar::v30x_to_rgb_u16_or_rgba_u16_row::<ALPHA>(&p, &mut s, width, matrix, full_range);
  unsafe {
    v30x_to_rgb_u16_or_rgba_u16_row::<ALPHA>(&p, &mut k, width, matrix, full_range);
  }
  assert_eq!(
    s,
    k,
    "NEON v30x→{} u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})",
    if ALPHA { "RGBA" } else { "RGB" }
  );
}

fn check_luma(width: usize) {
  let p = pseudo_random_v30x(width, 0xC001);
  let mut s = std::vec![0u8; width];
  let mut k = std::vec![0u8; width];
  scalar::v30x_to_luma_row(&p, &mut s, width);
  unsafe {
    v30x_to_luma_row(&p, &mut k, width);
  }
  assert_eq!(s, k, "NEON v30x→luma diverges (width={width})");
}

fn check_luma_u16(width: usize) {
  let p = pseudo_random_v30x(width, 0xC001);
  let mut s = std::vec![0u16; width];
  let mut k = std::vec![0u16; width];
  scalar::v30x_to_luma_u16_row(&p, &mut s, width);
  unsafe {
    v30x_to_luma_u16_row(&p, &mut k, width);
  }
  assert_eq!(s, k, "NEON v30x→luma u16 diverges (width={width})");
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_v30x_rgb_matches_scalar_all_matrices() {
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
fn neon_v30x_matches_scalar_widths() {
  for w in [1usize, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 1920, 1921, 1923] {
    check_rgb::<false>(w, ColorMatrix::Bt709, false);
    check_rgb::<true>(w, ColorMatrix::Bt709, true);
    check_rgb_u16::<false>(w, ColorMatrix::Bt2020Ncl, true);
    check_rgb_u16::<true>(w, ColorMatrix::Bt601, false);
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_v30x_luma_matches_scalar_widths() {
  for w in [1usize, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 1920, 1921, 1923] {
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
fn neon_v30x_lane_order_per_pixel_y_and_u() {
  const W: usize = 4;
  let packed = build_v30x_packed_y_n_plus_1_u_2n_plus_1_v_neutral(W);

  // Part 1: Luma natural-order (u16, no shift loss)
  let mut luma = std::vec![0u16; W];
  unsafe {
    v30x_to_luma_u16_row(&packed, &mut luma, W);
  }
  let expected_luma: std::vec::Vec<u16> = (1..=W as u16).collect();
  assert_eq!(luma, expected_luma, "neon v30x luma reorder bug");

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
    "neon v30x SIMD vs scalar diverges — lane-order bug"
  );
}
