use super::super::*;
use crate::{KernelMatrix, row::scalar};

/// Builds a deterministic pseudo-random Y216-shaped u16 buffer with
/// `width * 2` u16 samples. Each sample is a full 16-bit value
/// (no MSB shift — Y216 samples occupy the entire u16 word).
fn pseudo_random_y216(width: usize, seed: usize) -> std::vec::Vec<u16> {
  (0..width * 2)
    .map(|i| ((i.wrapping_mul(seed).wrapping_add(seed * 3)) & 0xFFFF) as u16)
    .collect()
}

fn check_rgb<const ALPHA: bool>(width: usize, matrix: KernelMatrix, full_range: bool) {
  let p = pseudo_random_y216(width, 0xAA55);
  let bpp = if ALPHA { 4 } else { 3 };
  let mut s = std::vec![0u8; width * bpp];
  let mut k = std::vec![0u8; width * bpp];
  scalar::y216_to_rgb_or_rgba_row::<ALPHA, false>(&p, &mut s, width, matrix, full_range);
  unsafe {
    y216_to_rgb_or_rgba_row::<ALPHA, false>(&p, &mut k, width, matrix, full_range);
  }
  assert_eq!(
    s,
    k,
    "NEON y216→{} diverges (width={width}, matrix={matrix:?}, full_range={full_range})",
    if ALPHA { "RGBA" } else { "RGB" }
  );
}

fn check_rgb_u16<const ALPHA: bool>(width: usize, matrix: KernelMatrix, full_range: bool) {
  let p = pseudo_random_y216(width, 0xAA55);
  let bpp = if ALPHA { 4 } else { 3 };
  let mut s = std::vec![0u16; width * bpp];
  let mut k = std::vec![0u16; width * bpp];
  scalar::y216_to_rgb_u16_or_rgba_u16_row::<ALPHA, false>(&p, &mut s, width, matrix, full_range);
  unsafe {
    y216_to_rgb_u16_or_rgba_u16_row::<ALPHA, false>(&p, &mut k, width, matrix, full_range);
  }
  assert_eq!(
    s,
    k,
    "NEON y216→{} u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})",
    if ALPHA { "RGBA" } else { "RGB" }
  );
}

fn check_luma(width: usize) {
  let p = pseudo_random_y216(width, 0xC001);
  let mut s = std::vec![0u8; width];
  let mut k = std::vec![0u8; width];
  scalar::y216_to_luma_row::<false>(&p, &mut s, width);
  unsafe {
    y216_to_luma_row::<false>(&p, &mut k, width);
  }
  assert_eq!(s, k, "NEON y216→luma diverges (width={width})");
}

fn check_luma_u16(width: usize) {
  let p = pseudo_random_y216(width, 0xC001);
  let mut s = std::vec![0u16; width];
  let mut k = std::vec![0u16; width];
  scalar::y216_to_luma_u16_row::<false>(&p, &mut s, width);
  unsafe {
    y216_to_luma_u16_row::<false>(&p, &mut k, width);
  }
  assert_eq!(s, k, "NEON y216→luma u16 diverges (width={width})");
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_y216_rgb_matches_scalar_all_matrices() {
  for m in [
    KernelMatrix::Bt601,
    KernelMatrix::Bt709,
    KernelMatrix::Bt2020Ncl,
    KernelMatrix::Smpte240m,
    KernelMatrix::Fcc,
    KernelMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_rgb::<false>(16, m, full);
      check_rgb::<true>(16, m, full);
      check_rgb_u16::<false>(16, m, full);
      check_rgb_u16::<true>(16, m, full);
    }
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_y216_matches_scalar_widths() {
  for w in [2usize, 4, 14, 16, 18, 30, 32, 34, 62, 64, 66, 1920, 1922] {
    check_rgb::<false>(w, KernelMatrix::Bt709, false);
    check_rgb::<true>(w, KernelMatrix::Bt709, true);
    check_rgb_u16::<false>(w, KernelMatrix::Bt2020Ncl, true);
    check_rgb_u16::<true>(w, KernelMatrix::Bt601, false);
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_y216_luma_matches_scalar_widths() {
  for w in [2usize, 4, 14, 16, 18, 30, 32, 34, 62, 64, 66, 1920, 1922] {
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
/// NEON u16-RGB path threshold: 16 px/iter. W=32 covers 2 full iterations.
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
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_y216_lane_order_per_pixel_y_and_u() {
  const W: usize = 32;
  let packed = build_y216_packed_y_n_plus_1_u_2k_plus_1_v_neutral(W);

  // Part 1: Luma natural-order at u16
  let mut luma_u16 = std::vec![0u16; W];
  unsafe {
    y216_to_luma_u16_row::<false>(&packed, &mut luma_u16, W);
  }
  let expected_luma: std::vec::Vec<u16> = (1..=W as u16).collect();
  assert_eq!(luma_u16, expected_luma, "NEON y216 luma_u16 reorder bug");

  // Part 2: SIMD vs scalar parity at u16 RGB (i64 chroma path)
  let mut simd_rgb = std::vec![0u16; W * 3];
  let mut scalar_rgb = std::vec![0u16; W * 3];
  unsafe {
    y216_to_rgb_u16_or_rgba_u16_row::<false, false>(
      &packed,
      &mut simd_rgb,
      W,
      KernelMatrix::Bt709,
      false,
    );
  }
  scalar::y216_to_rgb_u16_or_rgba_u16_row::<false, false>(
    &packed,
    &mut scalar_rgb,
    W,
    KernelMatrix::Bt709,
    false,
  );
  assert_eq!(
    simd_rgb, scalar_rgb,
    "NEON y216 SIMD vs scalar diverges (u16 RGB, i64 chroma)"
  );
}

// Host-independent BE/LE SIMD parity tests.
//
// Constructs LE/BE buffers from raw bytes via `to_le_bytes` /
// `to_be_bytes` and reinterprets as host-native `u16` via `from_ne_bytes`.
// The byte-level encoding is host-independent — on every host the LE
// buffer carries the intended values as LE-encoded bytes and the BE
// buffer carries the same values as BE-encoded bytes — so both kernel
// monomorphizations decode to the same logical values and produce
// byte-identical output on both LE and BE hosts. Locks down the
// `BE == HOST_NATIVE_BE` host-endian gate on the NEON Y216 SIMD bodies.

fn build_le_be_y216(width: usize, seed: usize) -> (std::vec::Vec<u16>, std::vec::Vec<u16>) {
  let intended = pseudo_random_y216(width, seed);
  let le_bytes: std::vec::Vec<u8> = intended.iter().flat_map(|v| v.to_le_bytes()).collect();
  let be_bytes: std::vec::Vec<u8> = intended.iter().flat_map(|v| v.to_be_bytes()).collect();
  let le: std::vec::Vec<u16> = le_bytes
    .chunks_exact(2)
    .map(|b| u16::from_ne_bytes([b[0], b[1]]))
    .collect();
  let be: std::vec::Vec<u16> = be_bytes
    .chunks_exact(2)
    .map(|b| u16::from_ne_bytes([b[0], b[1]]))
    .collect();
  (le, be)
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_y216_be_le_simd_parity() {
  // Widths covering tail-only (< 16), full SIMD body (16 px), and
  // body+tail to exercise both code paths on every host.
  for w in [8usize, 14, 16, 22, 32, 1920] {
    let (le, be) = build_le_be_y216(w, 0xBEEF);

    // u8 RGB
    let mut le_rgb = std::vec![0u8; w * 3];
    let mut be_rgb = std::vec![0u8; w * 3];
    unsafe {
      y216_to_rgb_or_rgba_row::<false, false>(&le, &mut le_rgb, w, KernelMatrix::Bt709, false);
      y216_to_rgb_or_rgba_row::<false, true>(&be, &mut be_rgb, w, KernelMatrix::Bt709, false);
    }
    assert_eq!(le_rgb, be_rgb, "y216 NEON LE vs BE RGB parity (w={w})");

    // u8 RGBA
    let mut le_rgba = std::vec![0u8; w * 4];
    let mut be_rgba = std::vec![0u8; w * 4];
    unsafe {
      y216_to_rgb_or_rgba_row::<true, false>(&le, &mut le_rgba, w, KernelMatrix::Bt709, false);
      y216_to_rgb_or_rgba_row::<true, true>(&be, &mut be_rgba, w, KernelMatrix::Bt709, false);
    }
    assert_eq!(le_rgba, be_rgba, "y216 NEON LE vs BE RGBA parity (w={w})");

    // u16 RGB (i64-chroma path)
    let mut le_u16 = std::vec![0u16; w * 3];
    let mut be_u16 = std::vec![0u16; w * 3];
    unsafe {
      y216_to_rgb_u16_or_rgba_u16_row::<false, false>(
        &le,
        &mut le_u16,
        w,
        KernelMatrix::Bt2020Ncl,
        true,
      );
      y216_to_rgb_u16_or_rgba_u16_row::<false, true>(
        &be,
        &mut be_u16,
        w,
        KernelMatrix::Bt2020Ncl,
        true,
      );
    }
    assert_eq!(le_u16, be_u16, "y216 NEON LE vs BE RGB u16 parity (w={w})");

    // luma u8
    let mut le_l = std::vec![0u8; w];
    let mut be_l = std::vec![0u8; w];
    unsafe {
      y216_to_luma_row::<false>(&le, &mut le_l, w);
      y216_to_luma_row::<true>(&be, &mut be_l, w);
    }
    assert_eq!(le_l, be_l, "y216 NEON LE vs BE luma u8 parity (w={w})");

    // luma u16
    let mut le_lu = std::vec![0u16; w];
    let mut be_lu = std::vec![0u16; w];
    unsafe {
      y216_to_luma_u16_row::<false>(&le, &mut le_lu, w);
      y216_to_luma_u16_row::<true>(&be, &mut be_lu, w);
    }
    assert_eq!(le_lu, be_lu, "y216 NEON LE vs BE luma u16 parity (w={w})");
  }
}
