use super::super::*;
use crate::{ColorMatrix, row::scalar};

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
    "AVX2 vuya<ALPHA={ALPHA}, ALPHA_SRC={ALPHA_SRC}>→{} diverges (width={width}, matrix={matrix:?}, full_range={full_range})",
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
  assert_eq!(s, k, "AVX2 vuya→luma diverges (width={width})");
}

/// Encodes pixel index `n` (0..32) into Y so we can detect lane
/// reordering bugs in the AVX2 deinterleave. The luma kernel just
/// copies Y bytes through, so output[n] must equal `n + 1` for
/// naturally-ordered AVX2 output. This is the test that surfaced
/// Ship 12b XV36 AVX2 bugs retroactively — write it from day 1.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn avx2_vuya_luma_lane_order_per_pixel() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  // 32 pixels: encode pixel index n into Y byte (n + 1 to avoid 0).
  // Y is at byte offset 2 of each pixel quadruple.
  let mut packed = std::vec![0u8; 32 * 4];
  for n in 0..32 {
    packed[n * 4 + 2] = (n as u8) + 1;
  }
  let mut out = std::vec![0u8; 32];
  unsafe {
    vuya_to_luma_row(&packed, &mut out, 32);
  }
  let expected: std::vec::Vec<u8> = (1..=32u8).collect();
  assert_eq!(
    out, expected,
    "AVX2 vuya→luma pixel reorder bug: {:?} != {:?}",
    out, expected
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn avx2_vuya_matches_scalar_widths() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  // Width sweep covers:
  //   - tail-only widths < 32 (no SIMD main loop)
  //   - the SIMD block-boundary 32 (one main-loop iteration, no tail)
  //   - partial-block-plus-tail 47/48/49 (one main-loop + 15/16/17-px tail)
  //   - production 1920p widths and odd tails (1921, 1923).
  for w in [
    1usize, 2, 3, 15, 16, 17, 31, 32, 33, 47, 48, 49, 1920, 1921, 1923,
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
fn avx2_vuya_rgb_matches_scalar_all_matrices() {
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
      // Width 32 = one main-loop iteration (no tail).
      check_rgb::<false, false>(32, m, full);
      check_rgb::<true, true>(32, m, full);
      check_rgb::<true, false>(32, m, full);
    }
  }
}
