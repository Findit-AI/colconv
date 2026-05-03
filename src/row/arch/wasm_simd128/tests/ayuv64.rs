use super::super::*;
use crate::{ColorMatrix, row::scalar};

/// Build a deterministic pseudo-random AYUV64 packed stream.
/// Returns `width * 4` u16 elements. Channels vary across the full u16 range.
fn pseudo_random_ayuv64(width: usize, seed: usize) -> std::vec::Vec<u16> {
  (0..width * 4)
    .map(|i| {
      let s = i.wrapping_mul(seed).wrapping_add(seed.wrapping_mul(3));
      (s & 0xFFFF) as u16
    })
    .collect()
}

fn check_rgb<const ALPHA: bool, const ALPHA_SRC: bool>(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let p = pseudo_random_ayuv64(width, 0xAA55);
  let bpp = if ALPHA { 4 } else { 3 };
  let mut s = std::vec![0u8; width * bpp];
  let mut k = std::vec![0u8; width * bpp];
  scalar::ayuv64_to_rgb_or_rgba_row::<ALPHA, ALPHA_SRC>(&p, &mut s, width, matrix, full_range);
  unsafe {
    ayuv64_to_rgb_or_rgba_row::<ALPHA, ALPHA_SRC>(&p, &mut k, width, matrix, full_range);
  }
  assert_eq!(
    s,
    k,
    "wasm ayuv64<ALPHA={ALPHA}, ALPHA_SRC={ALPHA_SRC}>→{} diverges (width={width}, matrix={matrix:?}, full_range={full_range})",
    if ALPHA { "RGBA" } else { "RGB" }
  );
}

fn check_rgb_u16<const ALPHA: bool, const ALPHA_SRC: bool>(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let p = pseudo_random_ayuv64(width, 0xAA55);
  let bpp = if ALPHA { 4 } else { 3 };
  let mut s = std::vec![0u16; width * bpp];
  let mut k = std::vec![0u16; width * bpp];
  scalar::ayuv64_to_rgb_u16_or_rgba_u16_row::<ALPHA, ALPHA_SRC>(
    &p, &mut s, width, matrix, full_range,
  );
  unsafe {
    ayuv64_to_rgb_u16_or_rgba_u16_row::<ALPHA, ALPHA_SRC>(&p, &mut k, width, matrix, full_range);
  }
  assert_eq!(
    s,
    k,
    "wasm ayuv64<ALPHA={ALPHA}, ALPHA_SRC={ALPHA_SRC}>→{} u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})",
    if ALPHA { "RGBA" } else { "RGB" }
  );
}

fn check_luma(width: usize) {
  let p = pseudo_random_ayuv64(width, 0xC001);
  let mut s = std::vec![0u8; width];
  let mut k = std::vec![0u8; width];
  scalar::ayuv64_to_luma_row(&p, &mut s, width);
  unsafe {
    ayuv64_to_luma_row(&p, &mut k, width);
  }
  assert_eq!(s, k, "wasm ayuv64→luma diverges (width={width})");
}

fn check_luma_u16(width: usize) {
  let p = pseudo_random_ayuv64(width, 0xC001);
  let mut s = std::vec![0u16; width];
  let mut k = std::vec![0u16; width];
  scalar::ayuv64_to_luma_u16_row(&p, &mut s, width);
  unsafe {
    ayuv64_to_luma_u16_row(&p, &mut k, width);
  }
  assert_eq!(s, k, "wasm ayuv64→luma u16 diverges (width={width})");
}

// wasm has no runtime CPU detection — `simd128` is a compile-time feature,
// so no `is_*_feature_detected!` early-return guard. The
// `#[cfg_attr(miri, ignore)]` attribute is included for parity with other
// backends; miri does not currently target wasm32-wasip1.

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn wasm_ayuv64_rgb_matches_scalar_all_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      // Width 16 = one full u8-path iteration AND two u16-path iterations.
      check_rgb::<false, false>(16, m, full);
      check_rgb::<true, true>(16, m, full);
      check_rgb_u16::<false, false>(16, m, full);
      check_rgb_u16::<true, true>(16, m, full);
    }
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn wasm_ayuv64_matches_scalar_widths() {
  for w in [
    1usize, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 1920, 1921, 1923,
  ] {
    check_rgb::<false, false>(w, ColorMatrix::Bt709, false);
    check_rgb::<true, true>(w, ColorMatrix::Bt709, true);
    check_rgb_u16::<false, false>(w, ColorMatrix::Bt2020Ncl, true);
    check_rgb_u16::<true, true>(w, ColorMatrix::Bt601, false);
    check_luma(w);
    check_luma_u16(w);
  }
}

/// Multi-channel Y+A lane-order regression test.
///
/// Encodes Y[n] = n + 1 (range 1..=16) AND A[n] = 2n + 1 (range 1..=31)
/// for n in 0..16. Uses 16 pixels to cover:
/// - the u8 SIMD block (16 px/iter, one full block)
/// - the u16 SIMD block (8 px/iter, two full blocks since 16 = 2×8)
///
/// Uses neutral chroma (U = V = 32768) so bias-subtracted chroma = 0,
/// making the conversion purely Y-dependent for R/G/B channels.
///
/// Asserts:
/// - `luma_u16_row` output = `[1, 2, …, 16]` (Y values direct)
/// - α u16 channel (slot 3 of each RGBA u16 quadruple) = `[1, 3, 5, …, 31]`
///
/// Catches both deinterleave permutation bugs (A and Y lanes crossed) and
/// asymmetric-mask bugs (bytes gathered from the wrong channel slot).
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn wasm_ayuv64_lane_order_per_pixel_y_and_a() {
  const W: usize = 16;
  // Build packed AYUV64: A[n]=2n+1, Y[n]=n+1, U=32768, V=32768.
  let mut packed = std::vec::Vec::with_capacity(W * 4);
  for n in 0..W {
    packed.push((2 * n + 1) as u16); // slot 0 = A
    packed.push((n + 1) as u16); // slot 1 = Y
    packed.push(32768u16); // slot 2 = U (neutral — bias-subtracted = 0)
    packed.push(32768u16); // slot 3 = V (neutral)
  }

  // --- luma_u16 path: Y values should be direct (no shift, no conversion). ---
  let mut luma_out = std::vec![0u16; W];
  unsafe {
    ayuv64_to_luma_u16_row(&packed, &mut luma_out, W);
  }
  let expected_luma: std::vec::Vec<u16> = (1..=16u16).collect();
  assert_eq!(luma_out, expected_luma, "wasm ayuv64→luma_u16 reorder bug");

  // --- RGBA u16 path: α channel (slot 3 of each output quadruple) = A direct. ---
  // Use full_range=true so neutral chroma gives a well-defined Y output.
  let mut rgba_out = std::vec![0u16; W * 4];
  unsafe {
    ayuv64_to_rgb_u16_or_rgba_u16_row::<true, true>(
      &packed,
      &mut rgba_out,
      W,
      ColorMatrix::Bt709,
      true, // full_range
    );
  }
  // α is at slot 3 (index 3) of each RGBA quadruple in the output.
  let alpha_out: std::vec::Vec<u16> = (0..W).map(|n| rgba_out[n * 4 + 3]).collect();
  let expected_alpha: std::vec::Vec<u16> = (0..W as u16).map(|n| 2 * n + 1).collect();
  assert_eq!(alpha_out, expected_alpha, "wasm ayuv64→rgba_u16 A lane reorder bug");
}
