use super::super::*;
use crate::{ColorMatrix, row::scalar};

/// Builds a deterministic pseudo-random v210 buffer with `words` words
/// (each word = 6 pixels = 16 bytes). Each 32-bit word holds three
/// 10-bit samples in `[0, 1023]` so the resulting buffer is a valid
/// v210 row.
fn pseudo_random_v210_words(words: usize, seed: usize) -> std::vec::Vec<u8> {
  let mut out = std::vec::Vec::with_capacity(words * 16);
  for w in 0..words {
    let mut buf = [0u8; 16];
    for k in 0..4 {
      // Build a u32 with three 10-bit fields; clamp each field to [0, 1023].
      let s0 = ((w * 4 + k) * 37 + seed) & 0x3FF;
      let s1 = ((w * 4 + k) * 53 + seed * 3) & 0x3FF;
      let s2 = ((w * 4 + k) * 71 + seed * 7) & 0x3FF;
      let word = (s0 as u32) | ((s1 as u32) << 10) | ((s2 as u32) << 20);
      buf[k * 4..k * 4 + 4].copy_from_slice(&word.to_le_bytes());
    }
    out.extend_from_slice(&buf);
  }
  out
}

fn check_rgb(width: usize, matrix: ColorMatrix, full_range: bool) {
  let p = pseudo_random_v210_words(width.div_ceil(6), 0xAA55);
  let mut s = std::vec![0u8; width * 3];
  let mut k = std::vec![0u8; width * 3];
  scalar::v210_to_rgb_or_rgba_row::<false>(&p, &mut s, width, matrix, full_range);
  unsafe {
    v210_to_rgb_or_rgba_row::<false>(&p, &mut k, width, matrix, full_range);
  }
  assert_eq!(
    s, k,
    "AVX2 v210→RGB diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_rgba(width: usize, matrix: ColorMatrix, full_range: bool) {
  let p = pseudo_random_v210_words(width.div_ceil(6), 0xAA55);
  let mut s = std::vec![0u8; width * 4];
  let mut k = std::vec![0u8; width * 4];
  scalar::v210_to_rgb_or_rgba_row::<true>(&p, &mut s, width, matrix, full_range);
  unsafe {
    v210_to_rgb_or_rgba_row::<true>(&p, &mut k, width, matrix, full_range);
  }
  assert_eq!(
    s, k,
    "AVX2 v210→RGBA diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_rgb_u16(width: usize, matrix: ColorMatrix, full_range: bool) {
  let p = pseudo_random_v210_words(width.div_ceil(6), 0xAA55);
  let mut s = std::vec![0u16; width * 3];
  let mut k = std::vec![0u16; width * 3];
  scalar::v210_to_rgb_u16_or_rgba_u16_row::<false>(&p, &mut s, width, matrix, full_range);
  unsafe {
    v210_to_rgb_u16_or_rgba_u16_row::<false>(&p, &mut k, width, matrix, full_range);
  }
  assert_eq!(
    s, k,
    "AVX2 v210→RGB u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_rgba_u16(width: usize, matrix: ColorMatrix, full_range: bool) {
  let p = pseudo_random_v210_words(width.div_ceil(6), 0xAA55);
  let mut s = std::vec![0u16; width * 4];
  let mut k = std::vec![0u16; width * 4];
  scalar::v210_to_rgb_u16_or_rgba_u16_row::<true>(&p, &mut s, width, matrix, full_range);
  unsafe {
    v210_to_rgb_u16_or_rgba_u16_row::<true>(&p, &mut k, width, matrix, full_range);
  }
  assert_eq!(
    s, k,
    "AVX2 v210→RGBA u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_luma(width: usize) {
  let p = pseudo_random_v210_words(width.div_ceil(6), 0xC001);
  let mut s = std::vec![0u8; width];
  let mut k = std::vec![0u8; width];
  scalar::v210_to_luma_row(&p, &mut s, width);
  unsafe {
    v210_to_luma_row(&p, &mut k, width);
  }
  assert_eq!(s, k, "AVX2 v210→luma diverges (width={width})");
}

fn check_luma_u16(width: usize) {
  let p = pseudo_random_v210_words(width.div_ceil(6), 0xC001);
  let mut s = std::vec![0u16; width];
  let mut k = std::vec![0u16; width];
  scalar::v210_to_luma_u16_row(&p, &mut s, width);
  unsafe {
    v210_to_luma_u16_row(&p, &mut k, width);
  }
  assert_eq!(s, k, "AVX2 v210→luma u16 diverges (width={width})");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn avx2_v210_rgb_matches_scalar_all_matrices() {
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
      // Width 12 = one main-loop pair (no tail).
      check_rgb(12, m, full);
      check_rgba(12, m, full);
      check_rgb_u16(12, m, full);
      check_rgba_u16(12, m, full);
    }
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn avx2_v210_matches_scalar_widths() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  // Widths cover the existing AVX2 cells (main-loop pair = 12 px) plus
  // partial-word cases (2, 4, 8, 10, 14, 1280, 1922) that route through
  // the scalar tail. Specifically:
  //   2, 4 = pure scalar tail (no main loop iterations)
  //   8 = 0 pairs + 6 px full word + 2 px partial (scalar tail)
  //   10 = 0 pairs + 6 px full word + 4 px partial
  //   14 = 1 pair + 2 px partial
  //   1280 = 106 pairs + 8 px scalar tail (1 full + partial 2)
  //   1922 = 160 pairs + 2 px partial-only tail
  for w in [2usize, 4, 8, 10, 12, 14, 18, 24, 30, 1280, 1920, 1922, 1932] {
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
fn avx2_v210_luma_matches_scalar_widths() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [2usize, 4, 8, 10, 12, 14, 18, 24, 30, 1280, 1920, 1922, 1932] {
    check_luma(w);
    check_luma_u16(w);
  }
}

/// Build a v210 packed buffer for `W` pixels where:
/// - Y[n] = n + 1 (at 10-bit position, so luma u16 output = n+1 directly)
/// - U[k] = 2k + 1 (one U per 2 pixels; k is chroma-pair index across all groups)
/// - V = 512 (neutral 10-bit midpoint, bias-subtracted = 0)
///
/// `W` must be a multiple of 6.
fn build_v210_packed_y_n_plus_1_u_2k_plus_1_v_neutral(w: usize) -> std::vec::Vec<u8> {
  assert!(w.is_multiple_of(6), "W must be a multiple of 6");
  let groups = w / 6;
  let mut out = std::vec::Vec::with_capacity(groups * 16);
  for g in 0..groups {
    // Within each group of 6 pixels:
    //   Y: y0=6g+1, y1=6g+2, y2=6g+3, y3=6g+4, y4=6g+5, y5=6g+6
    //   U (Cb): u0=2*(3g)+1, u1=2*(3g+1)+1, u2=2*(3g+2)+1
    //   V (Cr): 512 (neutral)
    let y0 = (g * 6 + 1) as u32;
    let y1 = (g * 6 + 2) as u32;
    let y2 = (g * 6 + 3) as u32;
    let y3 = (g * 6 + 4) as u32;
    let y4 = (g * 6 + 5) as u32;
    let y5 = (g * 6 + 6) as u32;
    let u0 = (2 * g * 3 + 1) as u32;
    let u1 = (2 * (g * 3 + 1) + 1) as u32;
    let u2 = (2 * (g * 3 + 2) + 1) as u32;
    let v: u32 = 512;
    // Word 0: [Cb0, Y0, Cr0] = [u0, y0, v]
    let w0 = (u0 & 0x3FF) | ((y0 & 0x3FF) << 10) | ((v & 0x3FF) << 20);
    // Word 1: [Y1, Cb1, Y2] = [y1, u1, y2]
    let w1 = (y1 & 0x3FF) | ((u1 & 0x3FF) << 10) | ((y2 & 0x3FF) << 20);
    // Word 2: [Cr1, Y3, Cb2] = [v, y3, u2]
    let w2 = (v & 0x3FF) | ((y3 & 0x3FF) << 10) | ((u2 & 0x3FF) << 20);
    // Word 3: [Y4, Cr2, Y5] = [y4, v, y5]
    let w3 = (y4 & 0x3FF) | ((v & 0x3FF) << 10) | ((y5 & 0x3FF) << 20);
    let mut buf = [0u8; 16];
    buf[0..4].copy_from_slice(&w0.to_le_bytes());
    buf[4..8].copy_from_slice(&w1.to_le_bytes());
    buf[8..12].copy_from_slice(&w2.to_le_bytes());
    buf[12..16].copy_from_slice(&w3.to_le_bytes());
    out.extend_from_slice(&buf);
  }
  out
}

/// Multi-channel Y+U lane-order regression test.
///
/// Encodes Y[n] = n + 1 (10-bit value, luma u16 output direct) AND
/// U[k] = 2k + 1 (one chroma sample per 2 pixels at 4:2:2) for four
/// V210 6-pixel groups (24 px = two AVX2 main-loop pairs). V = 512
/// (neutral midpoint, bias-subtracted = 0).
///
/// Asserts:
/// - `luma_u16_row` output = [1..=24] (Y values direct, no shift)
/// - SIMD RGB output == scalar RGB output (any chroma deinterleave bug diverges)
///
/// This catches asymmetric per-channel mask bugs that a Y-only test would miss.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn avx2_v210_lane_order_per_pixel_y_and_u() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  const W: usize = 24;
  let packed = build_v210_packed_y_n_plus_1_u_2k_plus_1_v_neutral(W);

  // Part 1: Luma natural-order (u16, no shift loss)
  let mut luma = std::vec![0u16; W];
  unsafe {
    v210_to_luma_u16_row(&packed, &mut luma, W);
  }
  let expected_luma: std::vec::Vec<u16> = (1..=W as u16).collect();
  assert_eq!(luma, expected_luma, "avx2 v210 luma reorder bug");

  // Part 2: SIMD vs scalar parity (catches chroma deinterleave bugs)
  let mut simd_rgb = std::vec![0u8; W * 3];
  let mut scalar_rgb = std::vec![0u8; W * 3];
  unsafe {
    v210_to_rgb_or_rgba_row::<false>(&packed, &mut simd_rgb, W, crate::ColorMatrix::Bt709, false);
  }
  scalar::v210_to_rgb_or_rgba_row::<false>(
    &packed,
    &mut scalar_rgb,
    W,
    crate::ColorMatrix::Bt709,
    false,
  );
  assert_eq!(
    simd_rgb, scalar_rgb,
    "avx2 v210 SIMD vs scalar diverges — chroma deinterleave bug"
  );
}
