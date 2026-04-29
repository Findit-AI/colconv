use super::{
  super::*, high_bit_plane_sse41, interleave_uv_sse41, p_n_packed_plane, p010_uv_interleave,
  p16_plane, planar_n_plane,
};

// ---- YUVA 4:4:4 u8 RGBA equivalence (Ship 8b‑1b) --------------------
//
// Mirrors the no-alpha 4:4:4 RGBA pattern above for the alpha-source
// path: per-pixel alpha byte is loaded from the source plane, masked
// with `bits_mask::<10>()`, and depth-converted via `>> 2`. Pseudo-
// random alpha is used to flush out lane-order corruption that a
// solid-alpha buffer would mask.

fn check_yuv444p_n_u8_sse41_rgba_with_alpha_src_equivalence<const BITS: u32>(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  alpha_seed: usize,
) {
  let y = planar_n_plane::<BITS>(width, 37);
  let u = planar_n_plane::<BITS>(width, 53);
  let v = planar_n_plane::<BITS>(width, 71);
  let a_src = planar_n_plane::<BITS>(width, alpha_seed);
  let mut rgba_scalar = std::vec![0u8; width * 4];
  let mut rgba_simd = std::vec![0u8; width * 4];
  scalar::yuv_444p_n_to_rgba_with_alpha_src_row::<BITS>(
    &y,
    &u,
    &v,
    &a_src,
    &mut rgba_scalar,
    width,
    matrix,
    full_range,
  );
  unsafe {
    yuv_444p_n_to_rgba_with_alpha_src_row::<BITS>(
      &y,
      &u,
      &v,
      &a_src,
      &mut rgba_simd,
      width,
      matrix,
      full_range,
    );
  }
  assert_eq!(
    rgba_scalar, rgba_simd,
    "SSE4.1 Yuva444p<{BITS}> → RGBA u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range}, alpha_seed={alpha_seed})"
  );
}

#[test]
fn sse41_yuva444p10_rgba_matches_scalar_all_matrices_16() {
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
      check_yuv444p_n_u8_sse41_rgba_with_alpha_src_equivalence::<10>(16, m, full, 89);
    }
  }
}

#[test]
fn sse41_yuva444p10_rgba_matches_scalar_widths() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  // Natural width + tail widths forcing scalar-tail dispatch.
  for w in [16usize, 17, 31, 47, 63, 1920, 1922] {
    check_yuv444p_n_u8_sse41_rgba_with_alpha_src_equivalence::<10>(w, ColorMatrix::Bt709, true, 89);
  }
}

#[test]
fn sse41_yuva444p10_rgba_matches_scalar_random_alpha() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  // Different alpha seeds — `_mm_packus_epi16` lane order through
  // `write_rgba_16` must put alpha in the 4th channel, not collide
  // with R/G/B.
  for seed in [13usize, 41, 89, 127, 211] {
    check_yuv444p_n_u8_sse41_rgba_with_alpha_src_equivalence::<10>(
      16,
      ColorMatrix::Bt601,
      false,
      seed,
    );
    check_yuv444p_n_u8_sse41_rgba_with_alpha_src_equivalence::<10>(
      31,
      ColorMatrix::Bt2020Ncl,
      true,
      seed,
    );
  }
}

#[test]
fn sse41_yuva444p_n_rgba_matches_scalar_all_bits() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  // BITS = 9, 12, 14 (BITS = 10 covered above with full matrix sweep).
  // Confirms `_mm_srl_epi16` with count `(BITS - 8)` resolves
  // correctly across the supported bit depths.
  for full in [true, false] {
    check_yuv444p_n_u8_sse41_rgba_with_alpha_src_equivalence::<9>(16, ColorMatrix::Bt601, full, 53);
    check_yuv444p_n_u8_sse41_rgba_with_alpha_src_equivalence::<12>(
      16,
      ColorMatrix::Bt709,
      full,
      53,
    );
    check_yuv444p_n_u8_sse41_rgba_with_alpha_src_equivalence::<14>(
      16,
      ColorMatrix::Bt2020Ncl,
      full,
      53,
    );
  }
}

#[test]
fn sse41_yuva444p_n_rgba_matches_scalar_all_bits_widths() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  for w in [17usize, 47, 1922] {
    check_yuv444p_n_u8_sse41_rgba_with_alpha_src_equivalence::<9>(
      w,
      ColorMatrix::Smpte240m,
      false,
      89,
    );
    check_yuv444p_n_u8_sse41_rgba_with_alpha_src_equivalence::<12>(w, ColorMatrix::Fcc, true, 89);
    check_yuv444p_n_u8_sse41_rgba_with_alpha_src_equivalence::<14>(
      w,
      ColorMatrix::YCgCo,
      false,
      89,
    );
  }
}

// ---- YUVA 4:2:0 u8 RGBA equivalence (Ship 8b‑2b) --------------------
//
// Mirrors the 4:4:4 alpha-source pattern for the 4:2:0 family —
// 8-bit (Yuva420p), high-bit BITS-generic (Yuva420p9 / Yuva420p10),
// and 16-bit (Yuva420p16). Direct backend call so `write_rgba_16` is
// exercised even when the dispatcher would pick a higher tier.

fn check_yuv_420_u8_sse41_rgba_with_alpha_src_equivalence(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  alpha_seed: usize,
) {
  let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
  let u: std::vec::Vec<u8> = (0..width / 2)
    .map(|i| ((i * 53 + 23) & 0xFF) as u8)
    .collect();
  let v: std::vec::Vec<u8> = (0..width / 2)
    .map(|i| ((i * 71 + 91) & 0xFF) as u8)
    .collect();
  let a_src: std::vec::Vec<u8> = (0..width)
    .map(|i| ((i * alpha_seed + 17) & 0xFF) as u8)
    .collect();
  let mut rgba_scalar = std::vec![0u8; width * 4];
  let mut rgba_simd = std::vec![0u8; width * 4];
  scalar::yuv_420_to_rgba_with_alpha_src_row(
    &y,
    &u,
    &v,
    &a_src,
    &mut rgba_scalar,
    width,
    matrix,
    full_range,
  );
  unsafe {
    yuv_420_to_rgba_with_alpha_src_row(
      &y,
      &u,
      &v,
      &a_src,
      &mut rgba_simd,
      width,
      matrix,
      full_range,
    );
  }
  assert_eq!(
    rgba_scalar, rgba_simd,
    "SSE4.1 Yuva420p (8-bit) → RGBA u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range}, alpha_seed={alpha_seed})"
  );
}

fn check_yuv420p_n_u8_sse41_rgba_with_alpha_src_equivalence<const BITS: u32>(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  alpha_seed: usize,
) {
  let y = planar_n_plane::<BITS>(width, 37);
  let u = planar_n_plane::<BITS>(width / 2, 53);
  let v = planar_n_plane::<BITS>(width / 2, 71);
  let a_src = planar_n_plane::<BITS>(width, alpha_seed);
  let mut rgba_scalar = std::vec![0u8; width * 4];
  let mut rgba_simd = std::vec![0u8; width * 4];
  scalar::yuv_420p_n_to_rgba_with_alpha_src_row::<BITS>(
    &y,
    &u,
    &v,
    &a_src,
    &mut rgba_scalar,
    width,
    matrix,
    full_range,
  );
  unsafe {
    yuv_420p_n_to_rgba_with_alpha_src_row::<BITS>(
      &y,
      &u,
      &v,
      &a_src,
      &mut rgba_simd,
      width,
      matrix,
      full_range,
    );
  }
  assert_eq!(
    rgba_scalar, rgba_simd,
    "SSE4.1 Yuva420p<{BITS}> → RGBA u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range}, alpha_seed={alpha_seed})"
  );
}

fn check_yuv420p16_u8_sse41_rgba_with_alpha_src_equivalence(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  alpha_seed: usize,
) {
  let y = p16_plane(width, 37);
  let u = p16_plane(width / 2, 53);
  let v = p16_plane(width / 2, 71);
  let a_src = p16_plane(width, alpha_seed);
  let mut rgba_scalar = std::vec![0u8; width * 4];
  let mut rgba_simd = std::vec![0u8; width * 4];
  scalar::yuv_420p16_to_rgba_with_alpha_src_row(
    &y,
    &u,
    &v,
    &a_src,
    &mut rgba_scalar,
    width,
    matrix,
    full_range,
  );
  unsafe {
    yuv_420p16_to_rgba_with_alpha_src_row(
      &y,
      &u,
      &v,
      &a_src,
      &mut rgba_simd,
      width,
      matrix,
      full_range,
    );
  }
  assert_eq!(
    rgba_scalar, rgba_simd,
    "SSE4.1 Yuva420p16 → RGBA u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range}, alpha_seed={alpha_seed})"
  );
}

#[test]
fn sse41_yuva420p_rgba_matches_scalar_all_matrices() {
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
      check_yuv_420_u8_sse41_rgba_with_alpha_src_equivalence(16, m, full, 89);
    }
  }
}

#[test]
fn sse41_yuva420p_rgba_matches_scalar_widths_and_alpha() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  for w in [16usize, 18, 30, 34, 1920, 1922] {
    check_yuv_420_u8_sse41_rgba_with_alpha_src_equivalence(w, ColorMatrix::Bt709, true, 89);
  }
  for seed in [13usize, 41, 127, 211] {
    check_yuv_420_u8_sse41_rgba_with_alpha_src_equivalence(16, ColorMatrix::Bt601, false, seed);
  }
}

#[test]
fn sse41_yuva420p_n_rgba_matches_scalar_all_bits() {
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
      check_yuv420p_n_u8_sse41_rgba_with_alpha_src_equivalence::<9>(16, m, full, 89);
      check_yuv420p_n_u8_sse41_rgba_with_alpha_src_equivalence::<10>(16, m, full, 89);
      check_yuv420p_n_u8_sse41_rgba_with_alpha_src_equivalence::<12>(16, m, full, 89);
    }
  }
}

#[test]
fn sse41_yuva420p_n_rgba_matches_scalar_widths() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  for w in [16usize, 18, 30, 34, 1920, 1922] {
    check_yuv420p_n_u8_sse41_rgba_with_alpha_src_equivalence::<9>(w, ColorMatrix::Bt601, false, 89);
    check_yuv420p_n_u8_sse41_rgba_with_alpha_src_equivalence::<10>(w, ColorMatrix::Bt709, true, 89);
    check_yuv420p_n_u8_sse41_rgba_with_alpha_src_equivalence::<12>(w, ColorMatrix::Bt709, true, 89);
  }
}

#[test]
fn sse41_yuva420p16_rgba_matches_scalar_all_matrices() {
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
      check_yuv420p16_u8_sse41_rgba_with_alpha_src_equivalence(16, m, full, 89);
    }
  }
}

#[test]
fn sse41_yuva420p16_rgba_matches_scalar_widths_and_alpha() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  for w in [16usize, 18, 30, 34, 1920, 1922] {
    check_yuv420p16_u8_sse41_rgba_with_alpha_src_equivalence(w, ColorMatrix::Bt709, false, 89);
  }
  for seed in [13usize, 41, 127, 211] {
    check_yuv420p16_u8_sse41_rgba_with_alpha_src_equivalence(16, ColorMatrix::Bt601, true, seed);
  }
}

// ---- High-bit 4:4:4 native-depth `u16` RGBA equivalence (Ship 8 Tranche 7c) ----

fn check_yuv444p_n_u16_sse41_rgba_equivalence<const BITS: u32>(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let y = planar_n_plane::<BITS>(width, 37);
  let u = planar_n_plane::<BITS>(width, 53);
  let v = planar_n_plane::<BITS>(width, 71);
  let mut rgba_scalar = std::vec![0u16; width * 4];
  let mut rgba_simd = std::vec![0u16; width * 4];
  scalar::yuv_444p_n_to_rgba_u16_row::<BITS>(
    &y,
    &u,
    &v,
    &mut rgba_scalar,
    width,
    matrix,
    full_range,
  );
  unsafe {
    yuv_444p_n_to_rgba_u16_row::<BITS>(&y, &u, &v, &mut rgba_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgba_scalar, rgba_simd,
    "SSE4.1 Yuv444p<{BITS}> → RGBA u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_pn_444_u16_sse41_rgba_equivalence<const BITS: u32>(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let y = high_bit_plane_sse41::<BITS>(width, 37);
  let u = high_bit_plane_sse41::<BITS>(width, 53);
  let v = high_bit_plane_sse41::<BITS>(width, 71);
  let uv = interleave_uv_sse41(&u, &v);
  let mut rgba_scalar = std::vec![0u16; width * 4];
  let mut rgba_simd = std::vec![0u16; width * 4];
  scalar::p_n_444_to_rgba_u16_row::<BITS>(&y, &uv, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    p_n_444_to_rgba_u16_row::<BITS>(&y, &uv, &mut rgba_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgba_scalar, rgba_simd,
    "SSE4.1 Pn4:4:4<{BITS}> → RGBA u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_yuv444p16_u16_sse41_rgba_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y = p16_plane(width, 37);
  let u = p16_plane(width, 53);
  let v = p16_plane(width, 71);
  let mut rgba_scalar = std::vec![0u16; width * 4];
  let mut rgba_simd = std::vec![0u16; width * 4];
  scalar::yuv_444p16_to_rgba_u16_row(&y, &u, &v, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    yuv_444p16_to_rgba_u16_row(&y, &u, &v, &mut rgba_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgba_scalar, rgba_simd,
    "SSE4.1 Yuv444p16 → RGBA u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_p_n_444_16_u16_sse41_rgba_equivalence(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let y = p16_plane(width, 37);
  let u = p16_plane(width, 53);
  let v = p16_plane(width, 71);
  let uv = interleave_uv_sse41(&u, &v);
  let mut rgba_scalar = std::vec![0u16; width * 4];
  let mut rgba_simd = std::vec![0u16; width * 4];
  scalar::p_n_444_16_to_rgba_u16_row(&y, &uv, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    p_n_444_16_to_rgba_u16_row(&y, &uv, &mut rgba_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgba_scalar, rgba_simd,
    "SSE4.1 P416 → RGBA u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[test]
fn sse41_yuv444p_n_rgba_u16_matches_scalar_all_bits() {
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
      check_yuv444p_n_u16_sse41_rgba_equivalence::<9>(16, m, full);
      check_yuv444p_n_u16_sse41_rgba_equivalence::<10>(16, m, full);
      check_yuv444p_n_u16_sse41_rgba_equivalence::<12>(16, m, full);
      check_yuv444p_n_u16_sse41_rgba_equivalence::<14>(16, m, full);
    }
  }
}

#[test]
fn sse41_yuv444p_n_rgba_u16_matches_scalar_tail_and_widths() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  for w in [17usize, 31, 47, 63, 1920, 1922] {
    check_yuv444p_n_u16_sse41_rgba_equivalence::<9>(w, ColorMatrix::Bt601, false);
    check_yuv444p_n_u16_sse41_rgba_equivalence::<10>(w, ColorMatrix::Bt709, true);
    check_yuv444p_n_u16_sse41_rgba_equivalence::<12>(w, ColorMatrix::Bt2020Ncl, false);
    check_yuv444p_n_u16_sse41_rgba_equivalence::<14>(w, ColorMatrix::YCgCo, true);
  }
}

#[test]
fn sse41_pn_444_rgba_u16_matches_scalar_all_bits() {
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
      check_pn_444_u16_sse41_rgba_equivalence::<10>(16, m, full);
      check_pn_444_u16_sse41_rgba_equivalence::<12>(16, m, full);
    }
  }
}

#[test]
fn sse41_pn_444_rgba_u16_matches_scalar_tail_and_widths() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  for w in [17usize, 31, 47, 63, 1920, 1922] {
    check_pn_444_u16_sse41_rgba_equivalence::<10>(w, ColorMatrix::Bt601, false);
    check_pn_444_u16_sse41_rgba_equivalence::<12>(w, ColorMatrix::Bt709, true);
  }
}

#[test]
fn sse41_yuv444p16_rgba_u16_matches_scalar_all_matrices() {
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
      check_yuv444p16_u16_sse41_rgba_equivalence(16, m, full);
    }
  }
  for w in [17usize, 31, 47, 63, 1920, 1922] {
    check_yuv444p16_u16_sse41_rgba_equivalence(w, ColorMatrix::Bt709, false);
  }
}

fn check_yuv444p16_u16_sse41_rgba_with_alpha_src_equivalence(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  alpha_seed: usize,
) {
  let y = p16_plane(width, 37);
  let u = p16_plane(width, 53);
  let v = p16_plane(width, 71);
  let a_src = p16_plane(width, alpha_seed);
  let mut rgba_scalar = std::vec![0u16; width * 4];
  let mut rgba_simd = std::vec![0u16; width * 4];
  scalar::yuv_444p16_to_rgba_u16_with_alpha_src_row(
    &y,
    &u,
    &v,
    &a_src,
    &mut rgba_scalar,
    width,
    matrix,
    full_range,
  );
  unsafe {
    yuv_444p16_to_rgba_u16_with_alpha_src_row(
      &y,
      &u,
      &v,
      &a_src,
      &mut rgba_simd,
      width,
      matrix,
      full_range,
    );
  }
  assert_eq!(
    rgba_scalar, rgba_simd,
    "SSE4.1 Yuva444p16 → RGBA u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range}, alpha_seed={alpha_seed})"
  );
}

#[test]
fn sse41_yuva444p16_rgba_u16_matches_scalar_all_matrices() {
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
      check_yuv444p16_u16_sse41_rgba_with_alpha_src_equivalence(8, m, full, 89);
    }
  }
}

#[test]
fn sse41_yuva444p16_rgba_u16_matches_scalar_widths_and_alpha() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  for w in [8usize, 9, 15, 31, 47, 63, 1920, 1922] {
    check_yuv444p16_u16_sse41_rgba_with_alpha_src_equivalence(w, ColorMatrix::Bt709, true, 89);
  }
  for seed in [13usize, 41, 127, 211] {
    check_yuv444p16_u16_sse41_rgba_with_alpha_src_equivalence(8, ColorMatrix::Bt601, false, seed);
  }
}

#[test]
fn sse41_p416_rgba_u16_matches_scalar_all_matrices() {
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
      check_p_n_444_16_u16_sse41_rgba_equivalence(16, m, full);
    }
  }
  for w in [17usize, 31, 47, 63, 1920, 1922] {
    check_p_n_444_16_u16_sse41_rgba_equivalence(w, ColorMatrix::Bt709, false);
  }
}

// ---- YUVA 4:4:4 native-depth `u16` RGBA equivalence (Ship 8b‑1c) ----
//
// Mirrors the u8 RGBA alpha-source tests above for the u16 output
// path: per-pixel alpha element is loaded from the source plane,
// AND-masked with `bits_mask::<10>()`, and stored at native depth (no
// `>> (BITS - 8)` since both source alpha and output element are at
// the same bit depth). Pseudo-random alpha flushes lane-order
// corruption that a solid-alpha buffer would mask.

fn check_yuv444p_n_u16_sse41_rgba_with_alpha_src_equivalence<const BITS: u32>(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  alpha_seed: usize,
) {
  let y = planar_n_plane::<BITS>(width, 37);
  let u = planar_n_plane::<BITS>(width, 53);
  let v = planar_n_plane::<BITS>(width, 71);
  let a_src = planar_n_plane::<BITS>(width, alpha_seed);
  let mut rgba_scalar = std::vec![0u16; width * 4];
  let mut rgba_simd = std::vec![0u16; width * 4];
  scalar::yuv_444p_n_to_rgba_u16_with_alpha_src_row::<BITS>(
    &y,
    &u,
    &v,
    &a_src,
    &mut rgba_scalar,
    width,
    matrix,
    full_range,
  );
  unsafe {
    yuv_444p_n_to_rgba_u16_with_alpha_src_row::<BITS>(
      &y,
      &u,
      &v,
      &a_src,
      &mut rgba_simd,
      width,
      matrix,
      full_range,
    );
  }
  assert_eq!(
    rgba_scalar, rgba_simd,
    "SSE4.1 Yuva444p<{BITS}> → RGBA u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range}, alpha_seed={alpha_seed})"
  );
}

#[test]
fn sse41_yuva444p10_rgba_u16_matches_scalar_all_matrices_16() {
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
      check_yuv444p_n_u16_sse41_rgba_with_alpha_src_equivalence::<10>(16, m, full, 89);
    }
  }
}

#[test]
fn sse41_yuva444p10_rgba_u16_matches_scalar_widths() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  // Natural width + tail widths forcing scalar-tail dispatch.
  for w in [16usize, 17, 31, 47, 63, 1920, 1922] {
    check_yuv444p_n_u16_sse41_rgba_with_alpha_src_equivalence::<10>(
      w,
      ColorMatrix::Bt709,
      true,
      89,
    );
  }
}

#[test]
fn sse41_yuva444p10_rgba_u16_matches_scalar_random_alpha() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  // Different alpha seeds — `write_rgba_u16_8` lane order must put
  // alpha in the 4th channel, not collide with R/G/B.
  for seed in [13usize, 41, 89, 127, 211] {
    check_yuv444p_n_u16_sse41_rgba_with_alpha_src_equivalence::<10>(
      16,
      ColorMatrix::Bt601,
      false,
      seed,
    );
    check_yuv444p_n_u16_sse41_rgba_with_alpha_src_equivalence::<10>(
      31,
      ColorMatrix::Bt2020Ncl,
      true,
      seed,
    );
  }
}

#[test]
fn sse41_yuva444p_n_rgba_u16_matches_scalar_all_bits() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  // BITS = 9, 12, 14 (BITS = 10 covered above). Confirms the
  // AND-mask `mask_v` resolves correctly across the supported bit
  // depths.
  for full in [true, false] {
    check_yuv444p_n_u16_sse41_rgba_with_alpha_src_equivalence::<9>(
      16,
      ColorMatrix::Bt601,
      full,
      53,
    );
    check_yuv444p_n_u16_sse41_rgba_with_alpha_src_equivalence::<12>(
      16,
      ColorMatrix::Bt709,
      full,
      53,
    );
    check_yuv444p_n_u16_sse41_rgba_with_alpha_src_equivalence::<14>(
      16,
      ColorMatrix::Bt2020Ncl,
      full,
      53,
    );
  }
}

#[test]
fn sse41_yuva444p_n_rgba_u16_matches_scalar_all_bits_widths() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  for w in [17usize, 47, 1922] {
    check_yuv444p_n_u16_sse41_rgba_with_alpha_src_equivalence::<9>(
      w,
      ColorMatrix::Smpte240m,
      false,
      89,
    );
    check_yuv444p_n_u16_sse41_rgba_with_alpha_src_equivalence::<12>(w, ColorMatrix::Fcc, true, 89);
    check_yuv444p_n_u16_sse41_rgba_with_alpha_src_equivalence::<14>(
      w,
      ColorMatrix::YCgCo,
      false,
      89,
    );
  }
}

// ---- YUVA 4:2:0 native-depth `u16` RGBA equivalence (Ship 8b‑2c) ----

fn check_yuv420p_n_u16_sse41_rgba_with_alpha_src_equivalence<const BITS: u32>(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  alpha_seed: usize,
) {
  let y = planar_n_plane::<BITS>(width, 37);
  let u = planar_n_plane::<BITS>(width / 2, 53);
  let v = planar_n_plane::<BITS>(width / 2, 71);
  let a_src = planar_n_plane::<BITS>(width, alpha_seed);
  let mut rgba_scalar = std::vec![0u16; width * 4];
  let mut rgba_simd = std::vec![0u16; width * 4];
  scalar::yuv_420p_n_to_rgba_u16_with_alpha_src_row::<BITS>(
    &y,
    &u,
    &v,
    &a_src,
    &mut rgba_scalar,
    width,
    matrix,
    full_range,
  );
  unsafe {
    yuv_420p_n_to_rgba_u16_with_alpha_src_row::<BITS>(
      &y,
      &u,
      &v,
      &a_src,
      &mut rgba_simd,
      width,
      matrix,
      full_range,
    );
  }
  assert_eq!(
    rgba_scalar, rgba_simd,
    "SSE4.1 Yuva420p<{BITS}> → RGBA u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range}, alpha_seed={alpha_seed})"
  );
}

fn check_yuv420p16_u16_sse41_rgba_with_alpha_src_equivalence(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  alpha_seed: usize,
) {
  let y = p16_plane(width, 37);
  let u = p16_plane(width / 2, 53);
  let v = p16_plane(width / 2, 71);
  let a_src = p16_plane(width, alpha_seed);
  let mut rgba_scalar = std::vec![0u16; width * 4];
  let mut rgba_simd = std::vec![0u16; width * 4];
  scalar::yuv_420p16_to_rgba_u16_with_alpha_src_row(
    &y,
    &u,
    &v,
    &a_src,
    &mut rgba_scalar,
    width,
    matrix,
    full_range,
  );
  unsafe {
    yuv_420p16_to_rgba_u16_with_alpha_src_row(
      &y,
      &u,
      &v,
      &a_src,
      &mut rgba_simd,
      width,
      matrix,
      full_range,
    );
  }
  assert_eq!(
    rgba_scalar, rgba_simd,
    "SSE4.1 Yuva420p16 → RGBA u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range}, alpha_seed={alpha_seed})"
  );
}

#[test]
fn sse41_yuva420p_n_rgba_u16_matches_scalar_all_bits() {
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
      check_yuv420p_n_u16_sse41_rgba_with_alpha_src_equivalence::<9>(16, m, full, 89);
      check_yuv420p_n_u16_sse41_rgba_with_alpha_src_equivalence::<10>(16, m, full, 89);
      check_yuv420p_n_u16_sse41_rgba_with_alpha_src_equivalence::<12>(16, m, full, 89);
    }
  }
}

#[test]
fn sse41_yuva420p_n_rgba_u16_matches_scalar_widths() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  for w in [16usize, 18, 30, 34, 1920, 1922] {
    check_yuv420p_n_u16_sse41_rgba_with_alpha_src_equivalence::<9>(
      w,
      ColorMatrix::Bt601,
      false,
      89,
    );
    check_yuv420p_n_u16_sse41_rgba_with_alpha_src_equivalence::<10>(
      w,
      ColorMatrix::Bt709,
      true,
      89,
    );
    check_yuv420p_n_u16_sse41_rgba_with_alpha_src_equivalence::<12>(
      w,
      ColorMatrix::Smpte240m,
      true,
      89,
    );
  }
}

#[test]
fn sse41_yuva420p16_rgba_u16_matches_scalar_all_matrices() {
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
      check_yuv420p16_u16_sse41_rgba_with_alpha_src_equivalence(16, m, full, 89);
    }
  }
}

#[test]
fn sse41_yuva420p16_rgba_u16_matches_scalar_widths_and_alpha() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  for w in [16usize, 18, 30, 34, 1920, 1922] {
    check_yuv420p16_u16_sse41_rgba_with_alpha_src_equivalence(w, ColorMatrix::Bt709, false, 89);
  }
  for seed in [13usize, 41, 127, 211] {
    check_yuv420p16_u16_sse41_rgba_with_alpha_src_equivalence(16, ColorMatrix::Bt601, true, seed);
  }
}
