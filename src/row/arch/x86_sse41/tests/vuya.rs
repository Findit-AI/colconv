use super::super::*;
use crate::{ColorMatrix, row::scalar};

/// Build a VUYA packed stream with Y[n] = n+1, A[n] = 2n+1, V=U=128.
///
/// VUYA layout per pixel: `[V(8), U(8), Y(8), A(8)]`. Source α is real
/// (not padding). Encoding:
/// - V = 128 (neutral 8-bit midpoint)
/// - U = 128 (neutral)
/// - Y[n] = n + 1
/// - A[n] = 2n + 1  (source α — distinct values per pixel)
fn build_vuya_packed_y_n_plus_1_a_2n_plus_1_u_v_neutral(width: usize) -> std::vec::Vec<u8> {
  let mut packed = std::vec![0u8; width * 4];
  for n in 0..width {
    packed[n * 4] = 128; // V
    packed[n * 4 + 1] = 128; // U
    packed[n * 4 + 2] = (n as u8) + 1; // Y = n+1
    packed[n * 4 + 3] = (n as u8) * 2 + 1; // A = 2n+1
  }
  packed
}

/// Build a deterministic pseudo-random VUYA packed stream.
/// Returns `width * 4` bytes with channels varying across all 8-bit values.
fn pseudo_random_vuya(width: usize, seed: usize) -> std::vec::Vec<u8> {
  (0..width * 4)
    .map(|i| {
      let s = i.wrapping_mul(seed).wrapping_add(seed.wrapping_mul(3));
      (s & 0xFF) as u8
    })
    .collect()
}

fn check_rgb<const ALPHA: bool, const ALPHA_SRC: bool>(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let p = pseudo_random_vuya(width, 0xAA55);
  let bpp = if ALPHA { 4 } else { 3 };
  let mut s = std::vec![0u8; width * bpp];
  let mut k = std::vec![0u8; width * bpp];
  scalar::vuya_to_rgb_or_rgba_row::<ALPHA, ALPHA_SRC>(&p, &mut s, width, matrix, full_range);
  unsafe {
    vuya_to_rgb_or_rgba_row::<ALPHA, ALPHA_SRC>(&p, &mut k, width, matrix, full_range);
  }
  assert_eq!(
    s,
    k,
    "SSE4.1 vuya<ALPHA={ALPHA}, ALPHA_SRC={ALPHA_SRC}>→{} diverges (width={width}, matrix={matrix:?}, full_range={full_range})",
    if ALPHA { "RGBA" } else { "RGB" }
  );
}

fn check_luma(width: usize) {
  let p = pseudo_random_vuya(width, 0xC001);
  let mut s = std::vec![0u8; width];
  let mut k = std::vec![0u8; width];
  scalar::vuya_to_luma_row(&p, &mut s, width);
  unsafe {
    vuya_to_luma_row(&p, &mut k, width);
  }
  assert_eq!(s, k, "SSE4.1 vuya→luma diverges (width={width})");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn sse41_vuya_matches_scalar_widths() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  for w in [
    1usize, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 1920, 1921, 1923,
  ] {
    check_rgb::<false, false>(w, ColorMatrix::Bt709, false);
    check_rgb::<true, true>(w, ColorMatrix::Bt709, true);
    check_rgb::<true, false>(w, ColorMatrix::Bt2020Ncl, true);
    check_luma(w);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn sse41_vuya_rgb_matches_scalar_all_matrices() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
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
      check_rgb::<false, false>(16, m, full);
      check_rgb::<true, true>(16, m, full);
      check_rgb::<true, false>(16, m, full);
    }
  }
}

/// Multi-channel lane-order regression — encodes pixel index in
/// BOTH Y AND A so we catch per-channel asymmetric mask bugs that
/// a Y-only test would miss. Pattern from Ship 12d AYUV64 backport.
/// VUYA has source α — assert the α slot directly.
///
/// SSE4.1 SIMD threshold: 16 px/iter. W=32 covers exactly 2 full
/// SIMD iterations.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn sse41_vuya_lane_order_per_pixel_y_and_a() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  const W: usize = 32;
  let packed = build_vuya_packed_y_n_plus_1_a_2n_plus_1_u_v_neutral(W);

  // Part 1: Luma natural-order (u8 path, Y is direct).
  let mut luma = std::vec![0u8; W];
  unsafe {
    vuya_to_luma_row(&packed, &mut luma, W);
  }
  let expected_luma: std::vec::Vec<u8> = (1..=W as u8).collect();
  assert_eq!(luma, expected_luma, "sse41 vuya luma reorder bug");

  // Part 2: u8 RGBA — α slot (every 4th byte) directly verifies
  // A-channel deinterleave. neutral U/V → chroma contribution is zero.
  let mut rgba = std::vec![0u8; W * 4];
  unsafe {
    vuya_to_rgb_or_rgba_row::<true, true>(&packed, &mut rgba, W, ColorMatrix::Bt709, false);
  }
  let alpha_out: std::vec::Vec<u8> = (0..W).map(|n| rgba[n * 4 + 3]).collect();
  let expected_alpha: std::vec::Vec<u8> = (0..W).map(|n| (n as u8) * 2 + 1).collect();
  assert_eq!(alpha_out, expected_alpha, "sse41 vuya rgba α reorder bug");
}
