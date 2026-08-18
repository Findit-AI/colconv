use super::super::*;
use crate::{KernelMatrix, row::scalar};

/// Build a deterministic pseudo-random XV48 quadruple stream.
/// Each channel is a full 16-bit value (no shift — unlike XV36).
fn pseudo_random_xv48(width: usize, seed: usize) -> std::vec::Vec<u16> {
  (0..width * 4)
    .map(|i| {
      let s = i.wrapping_mul(seed).wrapping_add(seed * 3);
      (s & 0xFFFF) as u16 // full 16-bit value
    })
    .collect()
}

fn check_rgb<const ALPHA: bool>(width: usize, matrix: KernelMatrix, full_range: bool) {
  let p = pseudo_random_xv48(width, 0xAA55);
  let bpp = if ALPHA { 4 } else { 3 };
  let mut s = std::vec![0u8; width * bpp];
  let mut k = std::vec![0u8; width * bpp];
  scalar::xv48_to_rgb_or_rgba_row::<ALPHA, false>(&p, &mut s, width, matrix, full_range);
  unsafe {
    xv48_to_rgb_or_rgba_row::<ALPHA, false>(&p, &mut k, width, matrix, full_range);
  }
  assert_eq!(
    s,
    k,
    "wasm xv48<ALPHA={ALPHA}>→{} diverges (width={width}, matrix={matrix:?}, full_range={full_range})",
    if ALPHA { "RGBA" } else { "RGB" }
  );
}

fn check_rgb_u16<const ALPHA: bool>(width: usize, matrix: KernelMatrix, full_range: bool) {
  let p = pseudo_random_xv48(width, 0xAA55);
  let bpp = if ALPHA { 4 } else { 3 };
  let mut s = std::vec![0u16; width * bpp];
  let mut k = std::vec![0u16; width * bpp];
  scalar::xv48_to_rgb_u16_or_rgba_u16_row::<ALPHA, false>(&p, &mut s, width, matrix, full_range);
  unsafe {
    xv48_to_rgb_u16_or_rgba_u16_row::<ALPHA, false>(&p, &mut k, width, matrix, full_range);
  }
  assert_eq!(
    s,
    k,
    "wasm xv48<ALPHA={ALPHA}>→{} u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})",
    if ALPHA { "RGBA" } else { "RGB" }
  );
}

fn check_luma(width: usize) {
  let p = pseudo_random_xv48(width, 0xC001);
  let mut s = std::vec![0u8; width];
  let mut k = std::vec![0u8; width];
  scalar::xv48_to_luma_row::<false>(&p, &mut s, width);
  unsafe {
    xv48_to_luma_row::<false>(&p, &mut k, width);
  }
  assert_eq!(s, k, "wasm xv48→luma diverges (width={width})");
}

fn check_luma_u16(width: usize) {
  let p = pseudo_random_xv48(width, 0xC001);
  let mut s = std::vec![0u16; width];
  let mut k = std::vec![0u16; width];
  scalar::xv48_to_luma_u16_row::<false>(&p, &mut s, width);
  unsafe {
    xv48_to_luma_u16_row::<false>(&p, &mut k, width);
  }
  assert_eq!(s, k, "wasm xv48→luma u16 diverges (width={width})");
}

// wasm has no runtime CPU detection — `simd128` is a compile-time
// feature, so no `is_*_feature_detected!` early-return guard.

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn wasm_simd128_xv48_rgb_matches_scalar_all_matrices() {
  for m in [
    KernelMatrix::Bt601,
    KernelMatrix::Bt709,
    KernelMatrix::Bt2020Ncl,
    KernelMatrix::Smpte240m,
    KernelMatrix::Fcc,
    KernelMatrix::YCgCo,
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
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn wasm_simd128_xv48_matches_scalar_widths() {
  for w in [
    1usize, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 1920, 1921, 1923,
  ] {
    check_rgb::<false>(w, KernelMatrix::Bt709, false);
    check_rgb::<true>(w, KernelMatrix::Bt709, true);
    check_rgb_u16::<false>(w, KernelMatrix::Bt2020Ncl, true);
    check_rgb_u16::<true>(w, KernelMatrix::Bt601, false);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn wasm_simd128_xv48_luma_matches_scalar_widths() {
  for w in [
    1usize, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 1920, 1921, 1923,
  ] {
    check_luma(w);
    check_luma_u16(w);
  }
}

/// Build an XV48 packed buffer for `width` pixels where:
/// - Y[n] = (n + 1) << 8 (full 16-bit, encodes pixel index in high byte)
/// - U[n] = (2n + 1) << 8 (full 16-bit; XV48 is 4:4:4)
/// - V    = 0x8000 (neutral 16-bit midpoint)
/// - X    = 0 (slot 3 is padding for the X-prefix variant)
fn build_xv48_packed_y_n_plus_1_u_2n_plus_1_v_neutral_x_zero(width: usize) -> std::vec::Vec<u16> {
  let mut packed = std::vec::Vec::with_capacity(width * 4);
  for n in 0..width {
    let y = ((n as u16) + 1) << 8;
    let u = ((2 * (n as u16)) + 1) << 8;
    packed.push(u);
    packed.push(y);
    packed.push(0x8000u16);
    packed.push(0u16);
  }
  packed
}

/// Multi-channel lane-order regression — encodes pixel index in BOTH Y AND U.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn wasm_simd128_xv48_lane_order_per_pixel_y_and_u() {
  const W: usize = 16;
  let packed = build_xv48_packed_y_n_plus_1_u_2n_plus_1_v_neutral_x_zero(W);

  let mut luma_u16 = std::vec![0u16; W];
  unsafe {
    xv48_to_luma_u16_row::<false>(&packed, &mut luma_u16, W);
  }
  let expected_luma: std::vec::Vec<u16> = (1..=W as u16).map(|n| n << 8).collect();
  assert_eq!(
    luma_u16, expected_luma,
    "wasm_simd128 xv48 luma_u16 reorder bug"
  );

  let mut simd_rgb = std::vec![0u16; W * 3];
  let mut scalar_rgb = std::vec![0u16; W * 3];
  unsafe {
    xv48_to_rgb_u16_or_rgba_u16_row::<false, false>(
      &packed,
      &mut simd_rgb,
      W,
      KernelMatrix::Bt709,
      false,
    );
  }
  scalar::xv48_to_rgb_u16_or_rgba_u16_row::<false, false>(
    &packed,
    &mut scalar_rgb,
    W,
    KernelMatrix::Bt709,
    false,
  );
  assert_eq!(
    simd_rgb, scalar_rgb,
    "wasm_simd128 xv48 SIMD vs scalar diverges (u16 RGB) — lane-order bug"
  );
}
