use super::super::*;
use crate::{ColorMatrix, row::scalar};

/// Builds a deterministic pseudo‑random v210 buffer with `words` words
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
  let p = pseudo_random_v210_words(width / 6, 0xAA55);
  let mut s = std::vec![0u8; width * 3];
  let mut k = std::vec![0u8; width * 3];
  scalar::v210_to_rgb_or_rgba_row::<false>(&p, &mut s, width, matrix, full_range);
  unsafe {
    v210_to_rgb_or_rgba_row::<false>(&p, &mut k, width, matrix, full_range);
  }
  assert_eq!(
    s, k,
    "NEON v210→RGB diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_rgba(width: usize, matrix: ColorMatrix, full_range: bool) {
  let p = pseudo_random_v210_words(width / 6, 0xAA55);
  let mut s = std::vec![0u8; width * 4];
  let mut k = std::vec![0u8; width * 4];
  scalar::v210_to_rgb_or_rgba_row::<true>(&p, &mut s, width, matrix, full_range);
  unsafe {
    v210_to_rgb_or_rgba_row::<true>(&p, &mut k, width, matrix, full_range);
  }
  assert_eq!(
    s, k,
    "NEON v210→RGBA diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_rgb_u16(width: usize, matrix: ColorMatrix, full_range: bool) {
  let p = pseudo_random_v210_words(width / 6, 0xAA55);
  let mut s = std::vec![0u16; width * 3];
  let mut k = std::vec![0u16; width * 3];
  scalar::v210_to_rgb_u16_or_rgba_u16_row::<false>(&p, &mut s, width, matrix, full_range);
  unsafe {
    v210_to_rgb_u16_or_rgba_u16_row::<false>(&p, &mut k, width, matrix, full_range);
  }
  assert_eq!(
    s, k,
    "NEON v210→RGB u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_rgba_u16(width: usize, matrix: ColorMatrix, full_range: bool) {
  let p = pseudo_random_v210_words(width / 6, 0xAA55);
  let mut s = std::vec![0u16; width * 4];
  let mut k = std::vec![0u16; width * 4];
  scalar::v210_to_rgb_u16_or_rgba_u16_row::<true>(&p, &mut s, width, matrix, full_range);
  unsafe {
    v210_to_rgb_u16_or_rgba_u16_row::<true>(&p, &mut k, width, matrix, full_range);
  }
  assert_eq!(
    s, k,
    "NEON v210→RGBA u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_luma(width: usize) {
  let p = pseudo_random_v210_words(width / 6, 0xC001);
  let mut s = std::vec![0u8; width];
  let mut k = std::vec![0u8; width];
  scalar::v210_to_luma_row(&p, &mut s, width);
  unsafe {
    v210_to_luma_row(&p, &mut k, width);
  }
  assert_eq!(s, k, "NEON v210→luma diverges (width={width})");
}

fn check_luma_u16(width: usize) {
  let p = pseudo_random_v210_words(width / 6, 0xC001);
  let mut s = std::vec![0u16; width];
  let mut k = std::vec![0u16; width];
  scalar::v210_to_luma_u16_row(&p, &mut s, width);
  unsafe {
    v210_to_luma_u16_row(&p, &mut k, width);
  }
  assert_eq!(s, k, "NEON v210→luma u16 diverges (width={width})");
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_v210_rgb_matches_scalar_all_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_rgb(6, m, full);
      check_rgba(6, m, full);
      check_rgb_u16(6, m, full);
      check_rgba_u16(6, m, full);
    }
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_v210_matches_scalar_widths() {
  for w in [6usize, 12, 18, 24, 30, 1920, 1926] {
    check_rgb(w, ColorMatrix::Bt709, false);
    check_rgba(w, ColorMatrix::Bt709, true);
    check_rgb_u16(w, ColorMatrix::Bt2020Ncl, true);
    check_rgba_u16(w, ColorMatrix::Bt601, false);
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_v210_luma_matches_scalar_widths() {
  for w in [6usize, 12, 18, 24, 30, 1920, 1926] {
    check_luma(w);
    check_luma_u16(w);
  }
}
