use super::*;

fn check_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
  let u: std::vec::Vec<u8> = (0..width / 2)
    .map(|i| ((i * 53 + 23) & 0xFF) as u8)
    .collect();
  let v: std::vec::Vec<u8> = (0..width / 2)
    .map(|i| ((i * 71 + 91) & 0xFF) as u8)
    .collect();
  let mut rgb_scalar = std::vec![0u8; width * 3];
  let mut rgb_avx2 = std::vec![0u8; width * 3];

  scalar::yuv_420_to_rgb_row(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    yuv_420_to_rgb_row(&y, &u, &v, &mut rgb_avx2, width, matrix, full_range);
  }

  if rgb_scalar != rgb_avx2 {
    let first_diff = rgb_scalar
      .iter()
      .zip(rgb_avx2.iter())
      .position(|(a, b)| a != b)
      .unwrap();
    panic!(
      "AVX2 diverges from scalar at byte {first_diff} (width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} avx2={}",
      rgb_scalar[first_diff], rgb_avx2[first_diff]
    );
  }
}

#[test]
fn avx2_matches_scalar_all_matrices_32() {
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
      check_equivalence(32, m, full);
    }
  }
}

#[test]
fn avx2_matches_scalar_width_64() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  check_equivalence(64, ColorMatrix::Bt601, true);
  check_equivalence(64, ColorMatrix::Bt709, false);
  check_equivalence(64, ColorMatrix::YCgCo, true);
}

#[test]
fn avx2_matches_scalar_width_1920() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  check_equivalence(1920, ColorMatrix::Bt709, false);
}

#[test]
fn avx2_matches_scalar_odd_tail_widths() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  // Widths that leave a non‑trivial scalar tail (non‑multiple of 32).
  for w in [34usize, 46, 62, 1922] {
    check_equivalence(w, ColorMatrix::Bt601, false);
  }
}

// ---- yuv_420_to_rgba_row equivalence --------------------------------
//
// Direct backend test for the new RGBA path: bypasses the public
// dispatcher so the AVX2 `write_rgba_32` path (two halves through
// `write_rgba_16`) is exercised regardless of what tier the
// dispatcher would pick on the current runner.

fn check_rgba_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
  let u: std::vec::Vec<u8> = (0..width / 2)
    .map(|i| ((i * 53 + 23) & 0xFF) as u8)
    .collect();
  let v: std::vec::Vec<u8> = (0..width / 2)
    .map(|i| ((i * 71 + 91) & 0xFF) as u8)
    .collect();
  let mut rgba_scalar = std::vec![0u8; width * 4];
  let mut rgba_avx2 = std::vec![0u8; width * 4];

  scalar::yuv_420_to_rgba_row(&y, &u, &v, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    yuv_420_to_rgba_row(&y, &u, &v, &mut rgba_avx2, width, matrix, full_range);
  }

  if rgba_scalar != rgba_avx2 {
    let first_diff = rgba_scalar
      .iter()
      .zip(rgba_avx2.iter())
      .position(|(a, b)| a != b)
      .unwrap();
    let pixel = first_diff / 4;
    let channel = ["R", "G", "B", "A"][first_diff % 4];
    panic!(
      "AVX2 RGBA diverges from scalar at byte {first_diff} (px {pixel} {channel}, width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} avx2={}",
      rgba_scalar[first_diff], rgba_avx2[first_diff]
    );
  }
}

#[test]
fn avx2_rgba_matches_scalar_all_matrices_32() {
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
      check_rgba_equivalence(32, m, full);
    }
  }
}

#[test]
fn avx2_rgba_matches_scalar_width_64() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  check_rgba_equivalence(64, ColorMatrix::Bt601, true);
  check_rgba_equivalence(64, ColorMatrix::Bt709, false);
  check_rgba_equivalence(64, ColorMatrix::YCgCo, true);
}

#[test]
fn avx2_rgba_matches_scalar_width_1920() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  check_rgba_equivalence(1920, ColorMatrix::Bt709, false);
}

#[test]
fn avx2_rgba_matches_scalar_odd_tail_widths() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  // Widths that leave a non‑trivial scalar tail (non‑multiple of 32).
  for w in [34usize, 46, 62, 1922] {
    check_rgba_equivalence(w, ColorMatrix::Bt601, false);
  }
}

// ---- nv12_to_rgb_row equivalence ------------------------------------

fn check_nv12_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
  let uv: std::vec::Vec<u8> = (0..width / 2)
    .flat_map(|i| [((i * 53 + 23) & 0xFF) as u8, ((i * 71 + 91) & 0xFF) as u8])
    .collect();
  let mut rgb_scalar = std::vec![0u8; width * 3];
  let mut rgb_avx2 = std::vec![0u8; width * 3];

  scalar::nv12_to_rgb_row(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    nv12_to_rgb_row(&y, &uv, &mut rgb_avx2, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_avx2,
    "AVX2 NV12 ≠ scalar (width={width}, matrix={matrix:?})"
  );
}

fn check_nv12_matches_yuv420p(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
  let u: std::vec::Vec<u8> = (0..width / 2)
    .map(|i| ((i * 53 + 23) & 0xFF) as u8)
    .collect();
  let v: std::vec::Vec<u8> = (0..width / 2)
    .map(|i| ((i * 71 + 91) & 0xFF) as u8)
    .collect();
  let uv: std::vec::Vec<u8> = u.iter().zip(v.iter()).flat_map(|(a, b)| [*a, *b]).collect();

  let mut rgb_yuv420p = std::vec![0u8; width * 3];
  let mut rgb_nv12 = std::vec![0u8; width * 3];
  unsafe {
    yuv_420_to_rgb_row(&y, &u, &v, &mut rgb_yuv420p, width, matrix, full_range);
    nv12_to_rgb_row(&y, &uv, &mut rgb_nv12, width, matrix, full_range);
  }
  assert_eq!(
    rgb_yuv420p, rgb_nv12,
    "AVX2 NV12 ≠ YUV420P for equivalent UV"
  );
}

#[test]
fn avx2_nv12_matches_scalar_all_matrices_32() {
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
      check_nv12_equivalence(32, m, full);
    }
  }
}

#[test]
fn avx2_nv12_matches_scalar_widths() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [64usize, 1920, 34, 46, 62, 1922] {
    check_nv12_equivalence(w, ColorMatrix::Bt709, false);
  }
}

#[test]
fn avx2_nv12_matches_yuv420p() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [32usize, 62, 128, 1920] {
    check_nv12_matches_yuv420p(w, ColorMatrix::Bt709, false);
    check_nv12_matches_yuv420p(w, ColorMatrix::YCgCo, true);
  }
}

// ---- nv24_to_rgb_row / nv42_to_rgb_row equivalence ------------------

fn check_nv24_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
  let uv: std::vec::Vec<u8> = (0..width)
    .flat_map(|i| [((i * 53 + 23) & 0xFF) as u8, ((i * 71 + 91) & 0xFF) as u8])
    .collect();
  let mut rgb_scalar = std::vec![0u8; width * 3];
  let mut rgb_avx2 = std::vec![0u8; width * 3];

  scalar::nv24_to_rgb_row(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    nv24_to_rgb_row(&y, &uv, &mut rgb_avx2, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_avx2,
    "AVX2 NV24 ≠ scalar (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_nv42_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
  let vu: std::vec::Vec<u8> = (0..width)
    .flat_map(|i| [((i * 53 + 23) & 0xFF) as u8, ((i * 71 + 91) & 0xFF) as u8])
    .collect();
  let mut rgb_scalar = std::vec![0u8; width * 3];
  let mut rgb_avx2 = std::vec![0u8; width * 3];

  scalar::nv42_to_rgb_row(&y, &vu, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    nv42_to_rgb_row(&y, &vu, &mut rgb_avx2, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_avx2,
    "AVX2 NV42 ≠ scalar (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[test]
fn avx2_nv24_matches_scalar_all_matrices_32() {
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
      check_nv24_equivalence(32, m, full);
    }
  }
}

#[test]
fn avx2_nv24_matches_scalar_widths() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  // 32 / 64 → main loop; 33 / 65 → main + 1-px tail; 31 → pure
  // scalar tail (< block size); 1920 → wide.
  for w in [31usize, 32, 33, 63, 64, 65, 1920, 1921] {
    check_nv24_equivalence(w, ColorMatrix::Bt709, false);
  }
}

#[test]
fn avx2_nv42_matches_scalar_all_matrices_32() {
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
      check_nv42_equivalence(32, m, full);
    }
  }
}

#[test]
fn avx2_nv42_matches_scalar_widths() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [31usize, 32, 33, 63, 64, 65, 1920, 1921] {
    check_nv42_equivalence(w, ColorMatrix::Bt709, false);
  }
}

// ---- nv24_to_rgba_row / nv42_to_rgba_row equivalence ----------------

fn check_nv24_rgba_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
  let uv: std::vec::Vec<u8> = (0..width)
    .flat_map(|i| [((i * 53 + 23) & 0xFF) as u8, ((i * 71 + 91) & 0xFF) as u8])
    .collect();
  let mut rgba_scalar = std::vec![0u8; width * 4];
  let mut rgba_avx2 = std::vec![0u8; width * 4];

  scalar::nv24_to_rgba_row(&y, &uv, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    nv24_to_rgba_row(&y, &uv, &mut rgba_avx2, width, matrix, full_range);
  }

  if rgba_scalar != rgba_avx2 {
    let first_diff = rgba_scalar
      .iter()
      .zip(rgba_avx2.iter())
      .position(|(a, b)| a != b)
      .unwrap();
    let pixel = first_diff / 4;
    let channel = ["R", "G", "B", "A"][first_diff % 4];
    panic!(
      "AVX2 NV24 RGBA diverges from scalar at byte {first_diff} (px {pixel} {channel}, width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} avx2={}",
      rgba_scalar[first_diff], rgba_avx2[first_diff]
    );
  }
}

fn check_nv42_rgba_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
  let vu: std::vec::Vec<u8> = (0..width)
    .flat_map(|i| [((i * 53 + 23) & 0xFF) as u8, ((i * 71 + 91) & 0xFF) as u8])
    .collect();
  let mut rgba_scalar = std::vec![0u8; width * 4];
  let mut rgba_avx2 = std::vec![0u8; width * 4];

  scalar::nv42_to_rgba_row(&y, &vu, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    nv42_to_rgba_row(&y, &vu, &mut rgba_avx2, width, matrix, full_range);
  }

  if rgba_scalar != rgba_avx2 {
    let first_diff = rgba_scalar
      .iter()
      .zip(rgba_avx2.iter())
      .position(|(a, b)| a != b)
      .unwrap();
    let pixel = first_diff / 4;
    let channel = ["R", "G", "B", "A"][first_diff % 4];
    panic!(
      "AVX2 NV42 RGBA diverges from scalar at byte {first_diff} (px {pixel} {channel}, width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} avx2={}",
      rgba_scalar[first_diff], rgba_avx2[first_diff]
    );
  }
}

#[test]
fn avx2_nv24_rgba_matches_scalar_all_matrices_32() {
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
      check_nv24_rgba_equivalence(32, m, full);
    }
  }
}

#[test]
fn avx2_nv24_rgba_matches_scalar_widths() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [31usize, 32, 33, 63, 64, 65, 1920, 1921] {
    check_nv24_rgba_equivalence(w, ColorMatrix::Bt709, false);
  }
}

#[test]
fn avx2_nv42_rgba_matches_scalar_all_matrices_32() {
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
      check_nv42_rgba_equivalence(32, m, full);
    }
  }
}

#[test]
fn avx2_nv42_rgba_matches_scalar_widths() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [31usize, 32, 33, 63, 64, 65, 1920, 1921] {
    check_nv42_rgba_equivalence(w, ColorMatrix::Bt709, false);
  }
}

// ---- yuv_444_to_rgb_row equivalence ---------------------------------

fn check_yuv_444_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
  let u: std::vec::Vec<u8> = (0..width).map(|i| ((i * 53 + 23) & 0xFF) as u8).collect();
  let v: std::vec::Vec<u8> = (0..width).map(|i| ((i * 71 + 91) & 0xFF) as u8).collect();
  let mut rgb_scalar = std::vec![0u8; width * 3];
  let mut rgb_avx2 = std::vec![0u8; width * 3];

  scalar::yuv_444_to_rgb_row(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    yuv_444_to_rgb_row(&y, &u, &v, &mut rgb_avx2, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_avx2,
    "AVX2 yuv_444 ≠ scalar (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[test]
fn avx2_yuv_444_matches_scalar_all_matrices_32() {
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
      check_yuv_444_equivalence(32, m, full);
    }
  }
}

#[test]
fn avx2_yuv_444_matches_scalar_widths() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  // Widths straddling the 32-pixel AVX2 block and the 16-pixel
  // SSE4.1 tail; odd widths validate the 4:4:4 no-parity contract.
  for w in [31usize, 32, 33, 63, 64, 65, 1920, 1921] {
    check_yuv_444_equivalence(w, ColorMatrix::Bt709, false);
  }
}

// ---- yuv_444_to_rgba_row equivalence --------------------------------

fn check_yuv_444_rgba_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
  let u: std::vec::Vec<u8> = (0..width).map(|i| ((i * 53 + 23) & 0xFF) as u8).collect();
  let v: std::vec::Vec<u8> = (0..width).map(|i| ((i * 71 + 91) & 0xFF) as u8).collect();
  let mut rgba_scalar = std::vec![0u8; width * 4];
  let mut rgba_avx2 = std::vec![0u8; width * 4];

  scalar::yuv_444_to_rgba_row(&y, &u, &v, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    yuv_444_to_rgba_row(&y, &u, &v, &mut rgba_avx2, width, matrix, full_range);
  }

  if rgba_scalar != rgba_avx2 {
    let first_diff = rgba_scalar
      .iter()
      .zip(rgba_avx2.iter())
      .position(|(a, b)| a != b)
      .unwrap();
    let pixel = first_diff / 4;
    let channel = ["R", "G", "B", "A"][first_diff % 4];
    panic!(
      "AVX2 yuv_444 RGBA diverges from scalar at byte {first_diff} (px {pixel} {channel}, width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} avx2={}",
      rgba_scalar[first_diff], rgba_avx2[first_diff]
    );
  }
}

#[test]
fn avx2_yuv_444_rgba_matches_scalar_all_matrices_32() {
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
      check_yuv_444_rgba_equivalence(32, m, full);
    }
  }
}

#[test]
fn avx2_yuv_444_rgba_matches_scalar_widths() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [31usize, 32, 33, 63, 64, 65, 1920, 1921] {
    check_yuv_444_rgba_equivalence(w, ColorMatrix::Bt709, false);
  }
}

fn check_yuv_444_rgba_with_alpha_src_equivalence(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  alpha_seed: usize,
) {
  let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
  let u: std::vec::Vec<u8> = (0..width).map(|i| ((i * 53 + 23) & 0xFF) as u8).collect();
  let v: std::vec::Vec<u8> = (0..width).map(|i| ((i * 71 + 91) & 0xFF) as u8).collect();
  let a_src: std::vec::Vec<u8> = (0..width)
    .map(|i| ((i * alpha_seed + 17) & 0xFF) as u8)
    .collect();
  let mut rgba_scalar = std::vec![0u8; width * 4];
  let mut rgba_avx2 = std::vec![0u8; width * 4];

  scalar::yuv_444_to_rgba_with_alpha_src_row(
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
    yuv_444_to_rgba_with_alpha_src_row(
      &y,
      &u,
      &v,
      &a_src,
      &mut rgba_avx2,
      width,
      matrix,
      full_range,
    );
  }

  assert_eq!(
    rgba_scalar, rgba_avx2,
    "AVX2 Yuva444p → RGBA u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range}, alpha_seed={alpha_seed})"
  );
}

#[test]
fn avx2_yuva444p_rgba_matches_scalar_all_matrices() {
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
      check_yuv_444_rgba_with_alpha_src_equivalence(32, m, full, 89);
    }
  }
}

#[test]
fn avx2_yuva444p_rgba_matches_scalar_widths_and_alpha() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [32usize, 33, 47, 63, 1920, 1922] {
    check_yuv_444_rgba_with_alpha_src_equivalence(w, ColorMatrix::Bt709, true, 89);
  }
  for seed in [13usize, 41, 127, 211] {
    check_yuv_444_rgba_with_alpha_src_equivalence(32, ColorMatrix::Bt601, false, seed);
  }
}

// ---- yuv_444p_n<BITS> + yuv_444p16 equivalence ----------------------

fn check_yuv_444p_n_equivalence<const BITS: u32>(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let max_val = (1u16 << BITS) - 1;
  let y: std::vec::Vec<u16> = (0..width)
    .map(|i| ((i * 37 + 11) as u16) & max_val)
    .collect();
  let u: std::vec::Vec<u16> = (0..width)
    .map(|i| ((i * 53 + 23) as u16) & max_val)
    .collect();
  let v: std::vec::Vec<u16> = (0..width)
    .map(|i| ((i * 71 + 91) as u16) & max_val)
    .collect();
  let mut rgb_scalar = std::vec![0u8; width * 3];
  let mut rgb_avx2 = std::vec![0u8; width * 3];
  let mut u16_scalar = std::vec![0u16; width * 3];
  let mut u16_avx2 = std::vec![0u16; width * 3];

  scalar::yuv_444p_n_to_rgb_row::<BITS>(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
  scalar::yuv_444p_n_to_rgb_u16_row::<BITS>(&y, &u, &v, &mut u16_scalar, width, matrix, full_range);
  unsafe {
    yuv_444p_n_to_rgb_row::<BITS>(&y, &u, &v, &mut rgb_avx2, width, matrix, full_range);
    yuv_444p_n_to_rgb_u16_row::<BITS>(&y, &u, &v, &mut u16_avx2, width, matrix, full_range);
  }
  assert_eq!(rgb_scalar, rgb_avx2, "AVX2 yuv_444p_n<{BITS}> u8 ≠ scalar");
  assert_eq!(u16_scalar, u16_avx2, "AVX2 yuv_444p_n<{BITS}> u16 ≠ scalar");
}

#[test]
fn avx2_yuv_444p9_matches_scalar_all_matrices() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for m in [ColorMatrix::Bt709, ColorMatrix::Bt2020Ncl] {
    for full in [true, false] {
      check_yuv_444p_n_equivalence::<9>(32, m, full);
    }
  }
}

#[test]
fn avx2_yuv_444p10_matches_scalar_all_matrices() {
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
      check_yuv_444p_n_equivalence::<10>(32, m, full);
    }
  }
}

#[test]
fn avx2_yuv_444p12_matches_scalar_all_matrices() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for m in [ColorMatrix::Bt709, ColorMatrix::Bt2020Ncl] {
    for full in [true, false] {
      check_yuv_444p_n_equivalence::<12>(32, m, full);
    }
  }
}

#[test]
fn avx2_yuv_444p14_matches_scalar_all_matrices() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for m in [ColorMatrix::Bt709, ColorMatrix::Bt2020Ncl] {
    for full in [true, false] {
      check_yuv_444p_n_equivalence::<14>(32, m, full);
    }
  }
}

#[test]
fn avx2_yuv_444p_n_matches_scalar_widths() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [1usize, 3, 15, 17, 31, 32, 33, 63, 64, 65, 1920, 1921] {
    check_yuv_444p_n_equivalence::<10>(w, ColorMatrix::Bt709, false);
  }
}

fn check_yuv_444p16_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y: std::vec::Vec<u16> = (0..width).map(|i| (i * 2027 + 11) as u16).collect();
  let u: std::vec::Vec<u16> = (0..width).map(|i| (i * 2671 + 23) as u16).collect();
  let v: std::vec::Vec<u16> = (0..width).map(|i| (i * 3329 + 91) as u16).collect();
  let mut rgb_scalar = std::vec![0u8; width * 3];
  let mut rgb_avx2 = std::vec![0u8; width * 3];
  let mut u16_scalar = std::vec![0u16; width * 3];
  let mut u16_avx2 = std::vec![0u16; width * 3];

  scalar::yuv_444p16_to_rgb_row(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
  scalar::yuv_444p16_to_rgb_u16_row(&y, &u, &v, &mut u16_scalar, width, matrix, full_range);
  unsafe {
    yuv_444p16_to_rgb_row(&y, &u, &v, &mut rgb_avx2, width, matrix, full_range);
    yuv_444p16_to_rgb_u16_row(&y, &u, &v, &mut u16_avx2, width, matrix, full_range);
  }
  assert_eq!(rgb_scalar, rgb_avx2, "AVX2 yuv_444p16 u8 ≠ scalar");
  assert_eq!(u16_scalar, u16_avx2, "AVX2 yuv_444p16 u16 ≠ scalar");
}

#[test]
fn avx2_yuv_444p16_matches_scalar_all_matrices() {
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
      check_yuv_444p16_equivalence(32, m, full);
    }
  }
}

#[test]
fn avx2_yuv_444p16_matches_scalar_widths() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [1usize, 7, 8, 15, 31, 32, 33, 63, 64, 65, 1920, 1921] {
    check_yuv_444p16_equivalence(w, ColorMatrix::Bt709, false);
  }
}

// ---- bgr_rgb_swap_row equivalence -----------------------------------

fn check_swap_equivalence(width: usize) {
  let input: std::vec::Vec<u8> = (0..width * 3)
    .map(|i| ((i * 17 + 41) & 0xFF) as u8)
    .collect();
  let mut out_scalar = std::vec![0u8; width * 3];
  let mut out_avx2 = std::vec![0u8; width * 3];

  scalar::bgr_rgb_swap_row(&input, &mut out_scalar, width);
  unsafe {
    bgr_rgb_swap_row(&input, &mut out_avx2, width);
  }
  assert_eq!(out_scalar, out_avx2, "AVX2 swap diverges from scalar");
}

#[test]
fn avx2_swap_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [1usize, 15, 31, 32, 33, 47, 48, 63, 64, 1920, 1921] {
    check_swap_equivalence(w);
  }
}

// ---- nv21_to_rgb_row equivalence ------------------------------------

fn check_nv21_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
  let vu: std::vec::Vec<u8> = (0..width / 2)
    .flat_map(|i| [((i * 53 + 23) & 0xFF) as u8, ((i * 71 + 91) & 0xFF) as u8])
    .collect();
  let mut rgb_scalar = std::vec![0u8; width * 3];
  let mut rgb_avx2 = std::vec![0u8; width * 3];

  scalar::nv21_to_rgb_row(&y, &vu, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    nv21_to_rgb_row(&y, &vu, &mut rgb_avx2, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_avx2,
    "AVX2 NV21 ≠ scalar (width={width}, matrix={matrix:?})"
  );
}

fn check_nv21_matches_nv12_swapped(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
  let uv: std::vec::Vec<u8> = (0..width / 2)
    .flat_map(|i| [((i * 53 + 23) & 0xFF) as u8, ((i * 71 + 91) & 0xFF) as u8])
    .collect();
  let mut vu = std::vec![0u8; width];
  for i in 0..width / 2 {
    vu[2 * i] = uv[2 * i + 1];
    vu[2 * i + 1] = uv[2 * i];
  }

  let mut rgb_nv12 = std::vec![0u8; width * 3];
  let mut rgb_nv21 = std::vec![0u8; width * 3];
  unsafe {
    nv12_to_rgb_row(&y, &uv, &mut rgb_nv12, width, matrix, full_range);
    nv21_to_rgb_row(&y, &vu, &mut rgb_nv21, width, matrix, full_range);
  }
  assert_eq!(
    rgb_nv12, rgb_nv21,
    "AVX2 NV21 ≠ NV12 with byte-swapped chroma"
  );
}

#[test]
fn nv21_avx2_matches_scalar_all_matrices_16() {
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
      check_nv21_equivalence(16, m, full);
    }
  }
}

#[test]
fn nv21_avx2_matches_scalar_widths() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [32usize, 1920, 18, 30, 34, 1922] {
    check_nv21_equivalence(w, ColorMatrix::Bt709, false);
  }
}

#[test]
fn nv21_avx2_matches_nv12_swapped() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [16usize, 30, 64, 1920] {
    check_nv21_matches_nv12_swapped(w, ColorMatrix::Bt709, false);
    check_nv21_matches_nv12_swapped(w, ColorMatrix::YCgCo, true);
  }
}

// ---- nv12_to_rgba_row / nv21_to_rgba_row equivalence ----------------

fn check_nv12_rgba_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
  let uv: std::vec::Vec<u8> = (0..width / 2)
    .flat_map(|i| [((i * 53 + 23) & 0xFF) as u8, ((i * 71 + 91) & 0xFF) as u8])
    .collect();
  let mut rgba_scalar = std::vec![0u8; width * 4];
  let mut rgba_avx2 = std::vec![0u8; width * 4];

  scalar::nv12_to_rgba_row(&y, &uv, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    nv12_to_rgba_row(&y, &uv, &mut rgba_avx2, width, matrix, full_range);
  }

  if rgba_scalar != rgba_avx2 {
    let first_diff = rgba_scalar
      .iter()
      .zip(rgba_avx2.iter())
      .position(|(a, b)| a != b)
      .unwrap();
    let pixel = first_diff / 4;
    let channel = ["R", "G", "B", "A"][first_diff % 4];
    panic!(
      "AVX2 NV12 RGBA diverges from scalar at byte {first_diff} (px {pixel} {channel}, width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} avx2={}",
      rgba_scalar[first_diff], rgba_avx2[first_diff]
    );
  }
}

fn check_nv21_rgba_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
  let vu: std::vec::Vec<u8> = (0..width / 2)
    .flat_map(|i| [((i * 53 + 23) & 0xFF) as u8, ((i * 71 + 91) & 0xFF) as u8])
    .collect();
  let mut rgba_scalar = std::vec![0u8; width * 4];
  let mut rgba_avx2 = std::vec![0u8; width * 4];

  scalar::nv21_to_rgba_row(&y, &vu, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    nv21_to_rgba_row(&y, &vu, &mut rgba_avx2, width, matrix, full_range);
  }

  if rgba_scalar != rgba_avx2 {
    let first_diff = rgba_scalar
      .iter()
      .zip(rgba_avx2.iter())
      .position(|(a, b)| a != b)
      .unwrap();
    let pixel = first_diff / 4;
    let channel = ["R", "G", "B", "A"][first_diff % 4];
    panic!(
      "AVX2 NV21 RGBA diverges from scalar at byte {first_diff} (px {pixel} {channel}, width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} avx2={}",
      rgba_scalar[first_diff], rgba_avx2[first_diff]
    );
  }
}

fn check_nv12_rgba_matches_yuv420p_rgba(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
  let u: std::vec::Vec<u8> = (0..width / 2)
    .map(|i| ((i * 53 + 23) & 0xFF) as u8)
    .collect();
  let v: std::vec::Vec<u8> = (0..width / 2)
    .map(|i| ((i * 71 + 91) & 0xFF) as u8)
    .collect();
  let uv: std::vec::Vec<u8> = u.iter().zip(v.iter()).flat_map(|(a, b)| [*a, *b]).collect();

  let mut rgba_yuv420p = std::vec![0u8; width * 4];
  let mut rgba_nv12 = std::vec![0u8; width * 4];
  unsafe {
    yuv_420_to_rgba_row(&y, &u, &v, &mut rgba_yuv420p, width, matrix, full_range);
    nv12_to_rgba_row(&y, &uv, &mut rgba_nv12, width, matrix, full_range);
  }
  assert_eq!(
    rgba_yuv420p, rgba_nv12,
    "AVX2 NV12 RGBA must match Yuv420p RGBA for equivalent UV (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[test]
fn nv12_avx2_rgba_matches_scalar_all_matrices_32() {
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
      check_nv12_rgba_equivalence(32, m, full);
    }
  }
}

#[test]
fn nv12_avx2_rgba_matches_scalar_widths() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [34usize, 46, 62, 1920, 1922] {
    check_nv12_rgba_equivalence(w, ColorMatrix::Bt601, false);
  }
}

#[test]
fn nv12_avx2_rgba_matches_yuv420p_rgba_avx2() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [32usize, 64, 1920] {
    check_nv12_rgba_matches_yuv420p_rgba(w, ColorMatrix::Bt709, false);
    check_nv12_rgba_matches_yuv420p_rgba(w, ColorMatrix::YCgCo, true);
  }
}

#[test]
fn nv21_avx2_rgba_matches_scalar_all_matrices_32() {
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
      check_nv21_rgba_equivalence(32, m, full);
    }
  }
}

#[test]
fn nv21_avx2_rgba_matches_scalar_widths() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [34usize, 46, 62, 1920, 1922] {
    check_nv21_rgba_equivalence(w, ColorMatrix::Bt601, false);
  }
}

// ---- rgb_to_hsv_row equivalence --------------------------------------

fn check_hsv_equivalence(rgb: &[u8], width: usize) {
  let mut h_s = std::vec![0u8; width];
  let mut s_s = std::vec![0u8; width];
  let mut v_s = std::vec![0u8; width];
  let mut h_k = std::vec![0u8; width];
  let mut s_k = std::vec![0u8; width];
  let mut v_k = std::vec![0u8; width];
  scalar::rgb_to_hsv_row(rgb, &mut h_s, &mut s_s, &mut v_s, width);
  unsafe {
    rgb_to_hsv_row(rgb, &mut h_k, &mut s_k, &mut v_k, width);
  }
  for (i, (a, b)) in h_s.iter().zip(h_k.iter()).enumerate() {
    assert!(
      a.abs_diff(*b) <= 1,
      "H divergence at pixel {i}: scalar={a} simd={b}"
    );
  }
  for (i, (a, b)) in s_s.iter().zip(s_k.iter()).enumerate() {
    assert!(
      a.abs_diff(*b) <= 1,
      "S divergence at pixel {i}: scalar={a} simd={b}"
    );
  }
  for (i, (a, b)) in v_s.iter().zip(v_k.iter()).enumerate() {
    assert!(
      a.abs_diff(*b) <= 1,
      "V divergence at pixel {i}: scalar={a} simd={b}"
    );
  }
}

#[test]
fn avx2_hsv_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  let rgb: std::vec::Vec<u8> = (0..1921 * 3)
    .map(|i| ((i * 37 + 11) & 0xFF) as u8)
    .collect();
  for w in [1usize, 31, 32, 33, 63, 64, 1920, 1921] {
    check_hsv_equivalence(&rgb[..w * 3], w);
  }
}

// ---- yuv420p10 AVX2 scalar-equivalence ------------------------------

fn p10_plane(n: usize, seed: usize) -> std::vec::Vec<u16> {
  (0..n)
    .map(|i| ((i * seed + seed * 3) & 0x3FF) as u16)
    .collect()
}

fn check_p10_u8_avx2_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  let y = p10_plane(width, 37);
  let u = p10_plane(width / 2, 53);
  let v = p10_plane(width / 2, 71);
  let mut rgb_scalar = std::vec![0u8; width * 3];
  let mut rgb_simd = std::vec![0u8; width * 3];

  scalar::yuv_420p_n_to_rgb_row::<10>(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    yuv_420p_n_to_rgb_row::<10>(&y, &u, &v, &mut rgb_simd, width, matrix, full_range);
  }

  if rgb_scalar != rgb_simd {
    let first_diff = rgb_scalar
      .iter()
      .zip(rgb_simd.iter())
      .position(|(a, b)| a != b)
      .unwrap();
    panic!(
      "AVX2 10→u8 diverges at byte {first_diff} (width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} simd={}",
      rgb_scalar[first_diff], rgb_simd[first_diff]
    );
  }
}

fn check_p10_u16_avx2_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  let y = p10_plane(width, 37);
  let u = p10_plane(width / 2, 53);
  let v = p10_plane(width / 2, 71);
  let mut rgb_scalar = std::vec![0u16; width * 3];
  let mut rgb_simd = std::vec![0u16; width * 3];

  scalar::yuv_420p_n_to_rgb_u16_row::<10>(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    yuv_420p_n_to_rgb_u16_row::<10>(&y, &u, &v, &mut rgb_simd, width, matrix, full_range);
  }

  if rgb_scalar != rgb_simd {
    let first_diff = rgb_scalar
      .iter()
      .zip(rgb_simd.iter())
      .position(|(a, b)| a != b)
      .unwrap();
    panic!(
      "AVX2 10→u16 diverges at elem {first_diff} (width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} simd={}",
      rgb_scalar[first_diff], rgb_simd[first_diff]
    );
  }
}

#[test]
fn avx2_p10_u8_matches_scalar_all_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_p10_u8_avx2_equivalence(32, m, full);
    }
  }
}

#[test]
fn avx2_p10_u16_matches_scalar_all_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_p10_u16_avx2_equivalence(32, m, full);
    }
  }
}

#[test]
fn avx2_p10_matches_scalar_odd_tail_widths() {
  for w in [34usize, 62, 66, 1922] {
    check_p10_u8_avx2_equivalence(w, ColorMatrix::Bt601, false);
    check_p10_u16_avx2_equivalence(w, ColorMatrix::Bt709, true);
  }
}

#[test]
fn avx2_p10_matches_scalar_1920() {
  check_p10_u8_avx2_equivalence(1920, ColorMatrix::Bt709, false);
  check_p10_u16_avx2_equivalence(1920, ColorMatrix::Bt2020Ncl, false);
}

// ---- yuv420p_n<BITS> AVX2 scalar-equivalence (BITS=9 coverage) ------

fn p_n_plane_avx2<const BITS: u32>(n: usize, seed: usize) -> std::vec::Vec<u16> {
  let mask = ((1u32 << BITS) - 1) as u16;
  (0..n)
    .map(|i| ((i.wrapping_mul(seed).wrapping_add(seed * 3)) as u16) & mask)
    .collect()
}

fn check_p_n_u8_avx2_equivalence<const BITS: u32>(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  let y = p_n_plane_avx2::<BITS>(width, 37);
  let u = p_n_plane_avx2::<BITS>(width / 2, 53);
  let v = p_n_plane_avx2::<BITS>(width / 2, 71);
  let mut rgb_scalar = std::vec![0u8; width * 3];
  let mut rgb_simd = std::vec![0u8; width * 3];
  scalar::yuv_420p_n_to_rgb_row::<BITS>(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    yuv_420p_n_to_rgb_row::<BITS>(&y, &u, &v, &mut rgb_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_simd,
    "AVX2 yuv_420p_n<{BITS}>→u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_p_n_u16_avx2_equivalence<const BITS: u32>(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  let y = p_n_plane_avx2::<BITS>(width, 37);
  let u = p_n_plane_avx2::<BITS>(width / 2, 53);
  let v = p_n_plane_avx2::<BITS>(width / 2, 71);
  let mut rgb_scalar = std::vec![0u16; width * 3];
  let mut rgb_simd = std::vec![0u16; width * 3];
  scalar::yuv_420p_n_to_rgb_u16_row::<BITS>(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    yuv_420p_n_to_rgb_u16_row::<BITS>(&y, &u, &v, &mut rgb_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_simd,
    "AVX2 yuv_420p_n<{BITS}>→u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[test]
fn avx2_yuv420p9_matches_scalar_all_matrices_and_ranges() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_p_n_u8_avx2_equivalence::<9>(32, m, full);
      check_p_n_u16_avx2_equivalence::<9>(32, m, full);
    }
  }
}

#[test]
fn avx2_yuv420p9_matches_scalar_tail_and_large_widths() {
  // AVX2 main loop is 32 px; widths chosen to stress tail handling
  // both below and above the SIMD lane width.
  for w in [18usize, 30, 34, 62, 66, 1922] {
    check_p_n_u8_avx2_equivalence::<9>(w, ColorMatrix::Bt601, false);
    check_p_n_u16_avx2_equivalence::<9>(w, ColorMatrix::Bt709, true);
  }
  check_p_n_u8_avx2_equivalence::<9>(1920, ColorMatrix::Bt709, false);
  check_p_n_u16_avx2_equivalence::<9>(1920, ColorMatrix::Bt2020Ncl, false);
}

// ---- P010 AVX2 scalar-equivalence -----------------------------------

fn p010_plane(n: usize, seed: usize) -> std::vec::Vec<u16> {
  (0..n)
    .map(|i| (((i * seed + seed * 3) & 0x3FF) as u16) << 6)
    .collect()
}

fn p010_uv_interleave(u: &[u16], v: &[u16]) -> std::vec::Vec<u16> {
  let pairs = u.len();
  debug_assert_eq!(u.len(), v.len());
  let mut out = std::vec::Vec::with_capacity(pairs * 2);
  for i in 0..pairs {
    out.push(u[i]);
    out.push(v[i]);
  }
  out
}

fn check_p010_u8_avx2_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  let y = p010_plane(width, 37);
  let u = p010_plane(width / 2, 53);
  let v = p010_plane(width / 2, 71);
  let uv = p010_uv_interleave(&u, &v);
  let mut rgb_scalar = std::vec![0u8; width * 3];
  let mut rgb_simd = std::vec![0u8; width * 3];
  scalar::p_n_to_rgb_row::<10>(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    p_n_to_rgb_row::<10>(&y, &uv, &mut rgb_simd, width, matrix, full_range);
  }
  assert_eq!(rgb_scalar, rgb_simd, "AVX2 P010→u8 diverges");
}

fn check_p010_u16_avx2_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  let y = p010_plane(width, 37);
  let u = p010_plane(width / 2, 53);
  let v = p010_plane(width / 2, 71);
  let uv = p010_uv_interleave(&u, &v);
  let mut rgb_scalar = std::vec![0u16; width * 3];
  let mut rgb_simd = std::vec![0u16; width * 3];
  scalar::p_n_to_rgb_u16_row::<10>(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    p_n_to_rgb_u16_row::<10>(&y, &uv, &mut rgb_simd, width, matrix, full_range);
  }
  assert_eq!(rgb_scalar, rgb_simd, "AVX2 P010→u16 diverges");
}

#[test]
fn avx2_p010_u8_matches_scalar_all_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_p010_u8_avx2_equivalence(32, m, full);
    }
  }
}

#[test]
fn avx2_p010_u16_matches_scalar_all_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_p010_u16_avx2_equivalence(32, m, full);
    }
  }
}

#[test]
fn avx2_p010_matches_scalar_odd_tail_widths() {
  for w in [34usize, 62, 66, 1922] {
    check_p010_u8_avx2_equivalence(w, ColorMatrix::Bt601, false);
    check_p010_u16_avx2_equivalence(w, ColorMatrix::Bt709, true);
  }
}

#[test]
fn avx2_p010_matches_scalar_1920() {
  check_p010_u8_avx2_equivalence(1920, ColorMatrix::Bt709, false);
  check_p010_u16_avx2_equivalence(1920, ColorMatrix::Bt2020Ncl, false);
}

// ---- Generic BITS equivalence (12/14-bit coverage) ------------------

fn planar_n_plane<const BITS: u32>(n: usize, seed: usize) -> std::vec::Vec<u16> {
  let mask = (1u32 << BITS) - 1;
  (0..n)
    .map(|i| ((i * seed + seed * 3) as u32 & mask) as u16)
    .collect()
}

fn p_n_packed_plane<const BITS: u32>(n: usize, seed: usize) -> std::vec::Vec<u16> {
  let mask = (1u32 << BITS) - 1;
  let shift = 16 - BITS;
  (0..n)
    .map(|i| (((i * seed + seed * 3) as u32 & mask) as u16) << shift)
    .collect()
}

fn check_planar_u8_avx2_equivalence_n<const BITS: u32>(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  let y = planar_n_plane::<BITS>(width, 37);
  let u = planar_n_plane::<BITS>(width / 2, 53);
  let v = planar_n_plane::<BITS>(width / 2, 71);
  let mut rgb_scalar = std::vec![0u8; width * 3];
  let mut rgb_simd = std::vec![0u8; width * 3];
  scalar::yuv_420p_n_to_rgb_row::<BITS>(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    yuv_420p_n_to_rgb_row::<BITS>(&y, &u, &v, &mut rgb_simd, width, matrix, full_range);
  }
  assert_eq!(rgb_scalar, rgb_simd, "AVX2 planar {BITS}-bit → u8 diverges");
}

fn check_planar_u16_avx2_equivalence_n<const BITS: u32>(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  let y = planar_n_plane::<BITS>(width, 37);
  let u = planar_n_plane::<BITS>(width / 2, 53);
  let v = planar_n_plane::<BITS>(width / 2, 71);
  let mut rgb_scalar = std::vec![0u16; width * 3];
  let mut rgb_simd = std::vec![0u16; width * 3];
  scalar::yuv_420p_n_to_rgb_u16_row::<BITS>(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    yuv_420p_n_to_rgb_u16_row::<BITS>(&y, &u, &v, &mut rgb_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_simd,
    "AVX2 planar {BITS}-bit → u16 diverges"
  );
}

fn check_pn_u8_avx2_equivalence_n<const BITS: u32>(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  let y = p_n_packed_plane::<BITS>(width, 37);
  let u = p_n_packed_plane::<BITS>(width / 2, 53);
  let v = p_n_packed_plane::<BITS>(width / 2, 71);
  let uv = p010_uv_interleave(&u, &v);
  let mut rgb_scalar = std::vec![0u8; width * 3];
  let mut rgb_simd = std::vec![0u8; width * 3];
  scalar::p_n_to_rgb_row::<BITS>(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    p_n_to_rgb_row::<BITS>(&y, &uv, &mut rgb_simd, width, matrix, full_range);
  }
  assert_eq!(rgb_scalar, rgb_simd, "AVX2 Pn {BITS}-bit → u8 diverges");
}

fn check_pn_u16_avx2_equivalence_n<const BITS: u32>(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  let y = p_n_packed_plane::<BITS>(width, 37);
  let u = p_n_packed_plane::<BITS>(width / 2, 53);
  let v = p_n_packed_plane::<BITS>(width / 2, 71);
  let uv = p010_uv_interleave(&u, &v);
  let mut rgb_scalar = std::vec![0u16; width * 3];
  let mut rgb_simd = std::vec![0u16; width * 3];
  scalar::p_n_to_rgb_u16_row::<BITS>(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    p_n_to_rgb_u16_row::<BITS>(&y, &uv, &mut rgb_simd, width, matrix, full_range);
  }
  assert_eq!(rgb_scalar, rgb_simd, "AVX2 Pn {BITS}-bit → u16 diverges");
}

#[test]
fn avx2_p12_matches_scalar_all_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_planar_u8_avx2_equivalence_n::<12>(32, m, full);
      check_planar_u16_avx2_equivalence_n::<12>(32, m, full);
      check_pn_u8_avx2_equivalence_n::<12>(32, m, full);
      check_pn_u16_avx2_equivalence_n::<12>(32, m, full);
    }
  }
}

#[test]
fn avx2_p14_matches_scalar_all_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_planar_u8_avx2_equivalence_n::<14>(32, m, full);
      check_planar_u16_avx2_equivalence_n::<14>(32, m, full);
    }
  }
}

#[test]
fn avx2_p12_matches_scalar_tail_widths() {
  for w in [34usize, 62, 66, 1922] {
    check_planar_u8_avx2_equivalence_n::<12>(w, ColorMatrix::Bt601, false);
    check_planar_u16_avx2_equivalence_n::<12>(w, ColorMatrix::Bt709, true);
    check_pn_u8_avx2_equivalence_n::<12>(w, ColorMatrix::Bt601, false);
    check_pn_u16_avx2_equivalence_n::<12>(w, ColorMatrix::Bt2020Ncl, false);
  }
}

#[test]
fn avx2_p14_matches_scalar_tail_widths() {
  for w in [34usize, 62, 66, 1922] {
    check_planar_u8_avx2_equivalence_n::<14>(w, ColorMatrix::Bt601, false);
    check_planar_u16_avx2_equivalence_n::<14>(w, ColorMatrix::Bt709, true);
  }
}

// ---- 16-bit (full-range u16 samples) AVX2 equivalence ---------------

fn p16_plane_avx2(n: usize, seed: usize) -> std::vec::Vec<u16> {
  (0..n)
    .map(|i| ((i.wrapping_mul(seed).wrapping_add(seed * 3)) & 0xFFFF) as u16)
    .collect()
}

fn check_yuv420p16_u8_avx2_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  let y = p16_plane_avx2(width, 37);
  let u = p16_plane_avx2(width / 2, 53);
  let v = p16_plane_avx2(width / 2, 71);
  let mut rgb_scalar = std::vec![0u8; width * 3];
  let mut rgb_simd = std::vec![0u8; width * 3];
  scalar::yuv_420p16_to_rgb_row(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    yuv_420p16_to_rgb_row(&y, &u, &v, &mut rgb_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_simd,
    "AVX2 yuv420p16→u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_yuv420p16_u16_avx2_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  let y = p16_plane_avx2(width, 37);
  let u = p16_plane_avx2(width / 2, 53);
  let v = p16_plane_avx2(width / 2, 71);
  let mut rgb_scalar = std::vec![0u16; width * 3];
  let mut rgb_simd = std::vec![0u16; width * 3];
  scalar::yuv_420p16_to_rgb_u16_row(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    yuv_420p16_to_rgb_u16_row(&y, &u, &v, &mut rgb_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_simd,
    "AVX2 yuv420p16→u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_p16_u8_avx2_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  let y = p16_plane_avx2(width, 37);
  let u = p16_plane_avx2(width / 2, 53);
  let v = p16_plane_avx2(width / 2, 71);
  let uv = p010_uv_interleave(&u, &v);
  let mut rgb_scalar = std::vec![0u8; width * 3];
  let mut rgb_simd = std::vec![0u8; width * 3];
  scalar::p16_to_rgb_row(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    p16_to_rgb_row(&y, &uv, &mut rgb_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_simd,
    "AVX2 p016→u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_p16_u16_avx2_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  let y = p16_plane_avx2(width, 37);
  let u = p16_plane_avx2(width / 2, 53);
  let v = p16_plane_avx2(width / 2, 71);
  let uv = p010_uv_interleave(&u, &v);
  let mut rgb_scalar = std::vec![0u16; width * 3];
  let mut rgb_simd = std::vec![0u16; width * 3];
  scalar::p16_to_rgb_u16_row(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    p16_to_rgb_u16_row(&y, &uv, &mut rgb_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_simd,
    "AVX2 p016→u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[test]
fn avx2_p16_matches_scalar_all_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_yuv420p16_u8_avx2_equivalence(32, m, full);
      check_yuv420p16_u16_avx2_equivalence(32, m, full);
      check_p16_u8_avx2_equivalence(32, m, full);
      check_p16_u16_avx2_equivalence(32, m, full);
    }
  }
}

#[test]
fn avx2_p16_matches_scalar_tail_widths() {
  for w in [34usize, 62, 66, 1922] {
    check_yuv420p16_u8_avx2_equivalence(w, ColorMatrix::Bt601, false);
    check_yuv420p16_u16_avx2_equivalence(w, ColorMatrix::Bt709, true);
    check_p16_u8_avx2_equivalence(w, ColorMatrix::Bt601, false);
    check_p16_u16_avx2_equivalence(w, ColorMatrix::Bt2020Ncl, false);
  }
}

#[test]
fn avx2_p16_matches_scalar_1920() {
  check_yuv420p16_u8_avx2_equivalence(1920, ColorMatrix::Bt709, false);
  check_yuv420p16_u16_avx2_equivalence(1920, ColorMatrix::Bt2020Ncl, false);
  check_p16_u8_avx2_equivalence(1920, ColorMatrix::Bt709, false);
  check_p16_u16_avx2_equivalence(1920, ColorMatrix::Bt2020Ncl, false);
}

// ---- High-bit 4:2:0 RGBA equivalence (Ship 8 Tranche 5a) ----------
//
// RGBA wrappers share the math of their RGB siblings — only the store
// (and tail dispatch) branches on `ALPHA`. These tests pin that the
// SIMD RGBA path produces byte-identical output to the scalar RGBA
// reference.

fn check_planar_u8_avx2_rgba_equivalence_n<const BITS: u32>(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let y = planar_n_plane::<BITS>(width, 37);
  let u = planar_n_plane::<BITS>(width / 2, 53);
  let v = planar_n_plane::<BITS>(width / 2, 71);
  let mut rgba_scalar = std::vec![0u8; width * 4];
  let mut rgba_simd = std::vec![0u8; width * 4];
  scalar::yuv_420p_n_to_rgba_row::<BITS>(&y, &u, &v, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    yuv_420p_n_to_rgba_row::<BITS>(&y, &u, &v, &mut rgba_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgba_scalar, rgba_simd,
    "AVX2 yuv_420p_n<{BITS}>→RGBA u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_pn_u8_avx2_rgba_equivalence_n<const BITS: u32>(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let y = p_n_packed_plane::<BITS>(width, 37);
  let u = p_n_packed_plane::<BITS>(width / 2, 53);
  let v = p_n_packed_plane::<BITS>(width / 2, 71);
  let uv = p010_uv_interleave(&u, &v);
  let mut rgba_scalar = std::vec![0u8; width * 4];
  let mut rgba_simd = std::vec![0u8; width * 4];
  scalar::p_n_to_rgba_row::<BITS>(&y, &uv, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    p_n_to_rgba_row::<BITS>(&y, &uv, &mut rgba_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgba_scalar, rgba_simd,
    "AVX2 Pn<{BITS}>→RGBA u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_yuv420p16_u8_avx2_rgba_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y = p16_plane_avx2(width, 37);
  let u = p16_plane_avx2(width / 2, 53);
  let v = p16_plane_avx2(width / 2, 71);
  let mut rgba_scalar = std::vec![0u8; width * 4];
  let mut rgba_simd = std::vec![0u8; width * 4];
  scalar::yuv_420p16_to_rgba_row(&y, &u, &v, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    yuv_420p16_to_rgba_row(&y, &u, &v, &mut rgba_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgba_scalar, rgba_simd,
    "AVX2 yuv_420p16→RGBA u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_p16_u8_avx2_rgba_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y = p16_plane_avx2(width, 37);
  let u = p16_plane_avx2(width / 2, 53);
  let v = p16_plane_avx2(width / 2, 71);
  let uv = p010_uv_interleave(&u, &v);
  let mut rgba_scalar = std::vec![0u8; width * 4];
  let mut rgba_simd = std::vec![0u8; width * 4];
  scalar::p16_to_rgba_row(&y, &uv, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    p16_to_rgba_row(&y, &uv, &mut rgba_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgba_scalar, rgba_simd,
    "AVX2 P016→RGBA u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[test]
fn avx2_yuv420p_n_rgba_matches_scalar_all_bits() {
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
      check_planar_u8_avx2_rgba_equivalence_n::<9>(32, m, full);
      check_planar_u8_avx2_rgba_equivalence_n::<10>(32, m, full);
      check_planar_u8_avx2_rgba_equivalence_n::<12>(32, m, full);
      check_planar_u8_avx2_rgba_equivalence_n::<14>(32, m, full);
    }
  }
}

#[test]
fn avx2_yuv420p_n_rgba_matches_scalar_tail_and_1920() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [34usize, 48, 62, 1920, 1922] {
    check_planar_u8_avx2_rgba_equivalence_n::<9>(w, ColorMatrix::Bt601, false);
    check_planar_u8_avx2_rgba_equivalence_n::<10>(w, ColorMatrix::Bt709, true);
    check_planar_u8_avx2_rgba_equivalence_n::<12>(w, ColorMatrix::Bt2020Ncl, false);
    check_planar_u8_avx2_rgba_equivalence_n::<14>(w, ColorMatrix::YCgCo, true);
  }
}

#[test]
fn avx2_pn_rgba_matches_scalar_all_bits() {
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
      check_pn_u8_avx2_rgba_equivalence_n::<10>(32, m, full);
      check_pn_u8_avx2_rgba_equivalence_n::<12>(32, m, full);
    }
  }
}

#[test]
fn avx2_pn_rgba_matches_scalar_tail_and_1920() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [34usize, 48, 62, 1920, 1922] {
    check_pn_u8_avx2_rgba_equivalence_n::<10>(w, ColorMatrix::Bt601, false);
    check_pn_u8_avx2_rgba_equivalence_n::<12>(w, ColorMatrix::Bt709, true);
  }
}

#[test]
fn avx2_yuv420p16_rgba_matches_scalar_all_matrices() {
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
      check_yuv420p16_u8_avx2_rgba_equivalence(32, m, full);
    }
  }
  for w in [34usize, 48, 62, 1920, 1922] {
    check_yuv420p16_u8_avx2_rgba_equivalence(w, ColorMatrix::Bt709, false);
  }
}

#[test]
fn avx2_p016_rgba_matches_scalar_all_matrices() {
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
      check_p16_u8_avx2_rgba_equivalence(32, m, full);
    }
  }
  for w in [34usize, 48, 62, 1920, 1922] {
    check_p16_u8_avx2_rgba_equivalence(w, ColorMatrix::Bt709, false);
  }
}

// ---- High-bit 4:2:0 native-depth `u16` RGBA equivalence (Ship 8 Tranche 5b) ----

fn check_planar_u16_avx2_rgba_equivalence_n<const BITS: u32>(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let y = planar_n_plane::<BITS>(width, 37);
  let u = planar_n_plane::<BITS>(width / 2, 53);
  let v = planar_n_plane::<BITS>(width / 2, 71);
  let mut rgba_scalar = std::vec![0u16; width * 4];
  let mut rgba_simd = std::vec![0u16; width * 4];
  scalar::yuv_420p_n_to_rgba_u16_row::<BITS>(
    &y,
    &u,
    &v,
    &mut rgba_scalar,
    width,
    matrix,
    full_range,
  );
  unsafe {
    yuv_420p_n_to_rgba_u16_row::<BITS>(&y, &u, &v, &mut rgba_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgba_scalar, rgba_simd,
    "AVX2 yuv_420p_n<{BITS}>→RGBA u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_pn_u16_avx2_rgba_equivalence_n<const BITS: u32>(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let y = p_n_packed_plane::<BITS>(width, 37);
  let u = p_n_packed_plane::<BITS>(width / 2, 53);
  let v = p_n_packed_plane::<BITS>(width / 2, 71);
  let uv = p010_uv_interleave(&u, &v);
  let mut rgba_scalar = std::vec![0u16; width * 4];
  let mut rgba_simd = std::vec![0u16; width * 4];
  scalar::p_n_to_rgba_u16_row::<BITS>(&y, &uv, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    p_n_to_rgba_u16_row::<BITS>(&y, &uv, &mut rgba_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgba_scalar, rgba_simd,
    "AVX2 Pn<{BITS}>→RGBA u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_yuv420p16_u16_avx2_rgba_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y = p16_plane_avx2(width, 37);
  let u = p16_plane_avx2(width / 2, 53);
  let v = p16_plane_avx2(width / 2, 71);
  let mut rgba_scalar = std::vec![0u16; width * 4];
  let mut rgba_simd = std::vec![0u16; width * 4];
  scalar::yuv_420p16_to_rgba_u16_row(&y, &u, &v, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    yuv_420p16_to_rgba_u16_row(&y, &u, &v, &mut rgba_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgba_scalar, rgba_simd,
    "AVX2 yuv_420p16→RGBA u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_p16_u16_avx2_rgba_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y = p16_plane_avx2(width, 37);
  let u = p16_plane_avx2(width / 2, 53);
  let v = p16_plane_avx2(width / 2, 71);
  let uv = p010_uv_interleave(&u, &v);
  let mut rgba_scalar = std::vec![0u16; width * 4];
  let mut rgba_simd = std::vec![0u16; width * 4];
  scalar::p16_to_rgba_u16_row(&y, &uv, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    p16_to_rgba_u16_row(&y, &uv, &mut rgba_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgba_scalar, rgba_simd,
    "AVX2 P016→RGBA u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[test]
fn avx2_yuv420p_n_rgba_u16_matches_scalar_all_bits() {
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
      check_planar_u16_avx2_rgba_equivalence_n::<9>(32, m, full);
      check_planar_u16_avx2_rgba_equivalence_n::<10>(32, m, full);
      check_planar_u16_avx2_rgba_equivalence_n::<12>(32, m, full);
      check_planar_u16_avx2_rgba_equivalence_n::<14>(32, m, full);
    }
  }
}

#[test]
fn avx2_yuv420p_n_rgba_u16_matches_scalar_tail_and_1920() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [34usize, 48, 62, 1920, 1922] {
    check_planar_u16_avx2_rgba_equivalence_n::<9>(w, ColorMatrix::Bt601, false);
    check_planar_u16_avx2_rgba_equivalence_n::<10>(w, ColorMatrix::Bt709, true);
    check_planar_u16_avx2_rgba_equivalence_n::<12>(w, ColorMatrix::Bt2020Ncl, false);
    check_planar_u16_avx2_rgba_equivalence_n::<14>(w, ColorMatrix::YCgCo, true);
  }
}

#[test]
fn avx2_pn_rgba_u16_matches_scalar_all_bits() {
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
      check_pn_u16_avx2_rgba_equivalence_n::<10>(32, m, full);
      check_pn_u16_avx2_rgba_equivalence_n::<12>(32, m, full);
    }
  }
}

#[test]
fn avx2_pn_rgba_u16_matches_scalar_tail_and_1920() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [34usize, 48, 62, 1920, 1922] {
    check_pn_u16_avx2_rgba_equivalence_n::<10>(w, ColorMatrix::Bt601, false);
    check_pn_u16_avx2_rgba_equivalence_n::<12>(w, ColorMatrix::Bt709, true);
  }
}

#[test]
fn avx2_yuv420p16_rgba_u16_matches_scalar_all_matrices() {
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
      check_yuv420p16_u16_avx2_rgba_equivalence(32, m, full);
    }
  }
  for w in [34usize, 48, 62, 1920, 1922] {
    check_yuv420p16_u16_avx2_rgba_equivalence(w, ColorMatrix::Bt709, false);
  }
}

#[test]
fn avx2_p016_rgba_u16_matches_scalar_all_matrices() {
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
      check_p16_u16_avx2_rgba_equivalence(32, m, full);
    }
  }
  for w in [34usize, 48, 62, 1920, 1922] {
    check_p16_u16_avx2_rgba_equivalence(w, ColorMatrix::Bt709, false);
  }
}

// ---- Pn 4:4:4 (P410 / P412 / P416) AVX2 equivalence -----------------

fn high_bit_plane_avx2<const BITS: u32>(n: usize, seed: usize) -> std::vec::Vec<u16> {
  let mask = ((1u32 << BITS) - 1) as u16;
  let shift = 16 - BITS;
  (0..n)
    .map(|i| (((i.wrapping_mul(seed).wrapping_add(seed * 3)) as u16) & mask) << shift)
    .collect()
}

fn interleave_uv_avx2(u_full: &[u16], v_full: &[u16]) -> std::vec::Vec<u16> {
  debug_assert_eq!(u_full.len(), v_full.len());
  let mut out = std::vec::Vec::with_capacity(u_full.len() * 2);
  for i in 0..u_full.len() {
    out.push(u_full[i]);
    out.push(v_full[i]);
  }
  out
}

fn check_p_n_444_u8_avx2_equivalence<const BITS: u32>(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  let y = high_bit_plane_avx2::<BITS>(width, 37);
  let u = high_bit_plane_avx2::<BITS>(width, 53);
  let v = high_bit_plane_avx2::<BITS>(width, 71);
  let uv = interleave_uv_avx2(&u, &v);
  let mut rgb_scalar = std::vec![0u8; width * 3];
  let mut rgb_simd = std::vec![0u8; width * 3];
  scalar::p_n_444_to_rgb_row::<BITS>(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    p_n_444_to_rgb_row::<BITS>(&y, &uv, &mut rgb_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_simd,
    "AVX2 Pn4:4:4 {BITS}-bit → u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_p_n_444_u16_avx2_equivalence<const BITS: u32>(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  let y = high_bit_plane_avx2::<BITS>(width, 37);
  let u = high_bit_plane_avx2::<BITS>(width, 53);
  let v = high_bit_plane_avx2::<BITS>(width, 71);
  let uv = interleave_uv_avx2(&u, &v);
  let mut rgb_scalar = std::vec![0u16; width * 3];
  let mut rgb_simd = std::vec![0u16; width * 3];
  scalar::p_n_444_to_rgb_u16_row::<BITS>(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    p_n_444_to_rgb_u16_row::<BITS>(&y, &uv, &mut rgb_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_simd,
    "AVX2 Pn4:4:4 {BITS}-bit → u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_p_n_444_16_u8_avx2_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  let y = p16_plane_avx2(width, 37);
  let u = p16_plane_avx2(width, 53);
  let v = p16_plane_avx2(width, 71);
  let uv = interleave_uv_avx2(&u, &v);
  let mut rgb_scalar = std::vec![0u8; width * 3];
  let mut rgb_simd = std::vec![0u8; width * 3];
  scalar::p_n_444_16_to_rgb_row(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    p_n_444_16_to_rgb_row(&y, &uv, &mut rgb_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_simd,
    "AVX2 P416 → u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_p_n_444_16_u16_avx2_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  let y = p16_plane_avx2(width, 37);
  let u = p16_plane_avx2(width, 53);
  let v = p16_plane_avx2(width, 71);
  let uv = interleave_uv_avx2(&u, &v);
  let mut rgb_scalar = std::vec![0u16; width * 3];
  let mut rgb_simd = std::vec![0u16; width * 3];
  scalar::p_n_444_16_to_rgb_u16_row(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    p_n_444_16_to_rgb_u16_row(&y, &uv, &mut rgb_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_simd,
    "AVX2 P416 → u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[test]
fn avx2_p410_matches_scalar_all_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_p_n_444_u8_avx2_equivalence::<10>(32, m, full);
      check_p_n_444_u16_avx2_equivalence::<10>(32, m, full);
    }
  }
}

#[test]
fn avx2_p412_matches_scalar_all_matrices() {
  for m in [ColorMatrix::Bt709, ColorMatrix::Bt2020Ncl] {
    for full in [true, false] {
      check_p_n_444_u8_avx2_equivalence::<12>(32, m, full);
      check_p_n_444_u16_avx2_equivalence::<12>(32, m, full);
    }
  }
}

#[test]
fn avx2_p410_p412_matches_scalar_tail_widths() {
  for w in [1usize, 3, 17, 31, 33, 47, 63, 65, 1920, 1921] {
    check_p_n_444_u8_avx2_equivalence::<10>(w, ColorMatrix::Bt601, false);
    check_p_n_444_u16_avx2_equivalence::<10>(w, ColorMatrix::Bt709, true);
    check_p_n_444_u8_avx2_equivalence::<12>(w, ColorMatrix::Bt2020Ncl, false);
  }
}

#[test]
fn avx2_p416_matches_scalar_all_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_p_n_444_16_u8_avx2_equivalence(32, m, full);
      check_p_n_444_16_u16_avx2_equivalence(32, m, full);
    }
  }
}

#[test]
fn avx2_p416_matches_scalar_tail_widths() {
  for w in [1usize, 3, 15, 17, 31, 33, 47, 63, 65, 1920, 1921] {
    check_p_n_444_16_u8_avx2_equivalence(w, ColorMatrix::Bt709, false);
    check_p_n_444_16_u16_avx2_equivalence(w, ColorMatrix::Bt2020Ncl, true);
  }
}

// ---- High-bit 4:4:4 u8 RGBA equivalence (Ship 8 Tranche 7b) ---------
//
// Mirrors the 4:2:0 RGBA pattern in PR #25 (Tranche 5a). Each kernel
// family — Yuv444p_n (BITS-generic), Yuv444p16, Pn_444 (BITS-generic),
// Pn_444_16 — has its AVX2 RGBA kernel byte-pinned against the scalar
// reference at the natural width and a sweep of tail widths. Each test
// gates on `is_x86_feature_detected!("avx2")` to stay clean under
// sanitizer/Miri/non-feature-flagged CI runners.

fn check_yuv444p_n_u8_avx2_rgba_equivalence<const BITS: u32>(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let y = planar_n_plane::<BITS>(width, 37);
  let u = planar_n_plane::<BITS>(width, 53);
  let v = planar_n_plane::<BITS>(width, 71);
  let mut rgba_scalar = std::vec![0u8; width * 4];
  let mut rgba_simd = std::vec![0u8; width * 4];
  scalar::yuv_444p_n_to_rgba_row::<BITS>(&y, &u, &v, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    yuv_444p_n_to_rgba_row::<BITS>(&y, &u, &v, &mut rgba_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgba_scalar, rgba_simd,
    "AVX2 Yuv444p<{BITS}> → RGBA u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_pn_444_u8_avx2_rgba_equivalence<const BITS: u32>(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let y = high_bit_plane_avx2::<BITS>(width, 37);
  let u = high_bit_plane_avx2::<BITS>(width, 53);
  let v = high_bit_plane_avx2::<BITS>(width, 71);
  let uv = interleave_uv_avx2(&u, &v);
  let mut rgba_scalar = std::vec![0u8; width * 4];
  let mut rgba_simd = std::vec![0u8; width * 4];
  scalar::p_n_444_to_rgba_row::<BITS>(&y, &uv, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    p_n_444_to_rgba_row::<BITS>(&y, &uv, &mut rgba_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgba_scalar, rgba_simd,
    "AVX2 Pn4:4:4<{BITS}> → RGBA u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_yuv444p16_u8_avx2_rgba_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y = p16_plane_avx2(width, 37);
  let u = p16_plane_avx2(width, 53);
  let v = p16_plane_avx2(width, 71);
  let mut rgba_scalar = std::vec![0u8; width * 4];
  let mut rgba_simd = std::vec![0u8; width * 4];
  scalar::yuv_444p16_to_rgba_row(&y, &u, &v, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    yuv_444p16_to_rgba_row(&y, &u, &v, &mut rgba_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgba_scalar, rgba_simd,
    "AVX2 Yuv444p16 → RGBA u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_p_n_444_16_u8_avx2_rgba_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y = p16_plane_avx2(width, 37);
  let u = p16_plane_avx2(width, 53);
  let v = p16_plane_avx2(width, 71);
  let uv = interleave_uv_avx2(&u, &v);
  let mut rgba_scalar = std::vec![0u8; width * 4];
  let mut rgba_simd = std::vec![0u8; width * 4];
  scalar::p_n_444_16_to_rgba_row(&y, &uv, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    p_n_444_16_to_rgba_row(&y, &uv, &mut rgba_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgba_scalar, rgba_simd,
    "AVX2 P416 → RGBA u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[test]
fn avx2_yuv444p_n_rgba_matches_scalar_all_bits() {
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
      check_yuv444p_n_u8_avx2_rgba_equivalence::<9>(32, m, full);
      check_yuv444p_n_u8_avx2_rgba_equivalence::<10>(32, m, full);
      check_yuv444p_n_u8_avx2_rgba_equivalence::<12>(32, m, full);
      check_yuv444p_n_u8_avx2_rgba_equivalence::<14>(32, m, full);
    }
  }
}

#[test]
fn avx2_yuv444p_n_rgba_matches_scalar_tail_and_widths() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [17usize, 31, 47, 63, 1920, 1922] {
    check_yuv444p_n_u8_avx2_rgba_equivalence::<9>(w, ColorMatrix::Bt601, false);
    check_yuv444p_n_u8_avx2_rgba_equivalence::<10>(w, ColorMatrix::Bt709, true);
    check_yuv444p_n_u8_avx2_rgba_equivalence::<12>(w, ColorMatrix::Bt2020Ncl, false);
    check_yuv444p_n_u8_avx2_rgba_equivalence::<14>(w, ColorMatrix::YCgCo, true);
  }
}

#[test]
fn avx2_pn_444_rgba_matches_scalar_all_bits() {
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
      check_pn_444_u8_avx2_rgba_equivalence::<10>(32, m, full);
      check_pn_444_u8_avx2_rgba_equivalence::<12>(32, m, full);
    }
  }
}

#[test]
fn avx2_pn_444_rgba_matches_scalar_tail_and_widths() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [17usize, 31, 47, 63, 1920, 1922] {
    check_pn_444_u8_avx2_rgba_equivalence::<10>(w, ColorMatrix::Bt601, false);
    check_pn_444_u8_avx2_rgba_equivalence::<12>(w, ColorMatrix::Bt709, true);
  }
}

#[test]
fn avx2_yuv444p16_rgba_matches_scalar_all_matrices() {
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
      check_yuv444p16_u8_avx2_rgba_equivalence(32, m, full);
    }
  }
  for w in [17usize, 31, 47, 63, 1920, 1922] {
    check_yuv444p16_u8_avx2_rgba_equivalence(w, ColorMatrix::Bt709, false);
  }
}

fn check_yuv444p16_u8_avx2_rgba_with_alpha_src_equivalence(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  alpha_seed: usize,
) {
  let y = p16_plane_avx2(width, 37);
  let u = p16_plane_avx2(width, 53);
  let v = p16_plane_avx2(width, 71);
  let a_src = p16_plane_avx2(width, alpha_seed);
  let mut rgba_scalar = std::vec![0u8; width * 4];
  let mut rgba_simd = std::vec![0u8; width * 4];
  scalar::yuv_444p16_to_rgba_with_alpha_src_row(
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
    yuv_444p16_to_rgba_with_alpha_src_row(
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
    "AVX2 Yuva444p16 → RGBA u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range}, alpha_seed={alpha_seed})"
  );
}

#[test]
fn avx2_yuva444p16_rgba_matches_scalar_all_matrices() {
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
      check_yuv444p16_u8_avx2_rgba_with_alpha_src_equivalence(32, m, full, 89);
    }
  }
}

#[test]
fn avx2_yuva444p16_rgba_matches_scalar_widths_and_alpha() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [32usize, 33, 47, 63, 1920, 1922] {
    check_yuv444p16_u8_avx2_rgba_with_alpha_src_equivalence(w, ColorMatrix::Bt709, true, 89);
  }
  for seed in [13usize, 41, 127, 211] {
    check_yuv444p16_u8_avx2_rgba_with_alpha_src_equivalence(32, ColorMatrix::Bt601, false, seed);
  }
}

#[test]
fn avx2_p416_rgba_matches_scalar_all_matrices() {
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
      check_p_n_444_16_u8_avx2_rgba_equivalence(32, m, full);
    }
  }
  for w in [17usize, 31, 47, 63, 1920, 1922] {
    check_p_n_444_16_u8_avx2_rgba_equivalence(w, ColorMatrix::Bt709, false);
  }
}

// ---- YUVA 4:4:4 u8 RGBA equivalence (Ship 8b‑1b) --------------------
//
// Mirrors the no-alpha 4:4:4 RGBA pattern above for the alpha-source
// path: per-pixel alpha byte is loaded from the source plane, masked
// with `bits_mask::<10>()`, and depth-converted via `>> 2`. Pseudo-
// random alpha is used to flush out lane-order corruption that a
// solid-alpha buffer would mask. AVX2's `narrow_u8x32` per-lane
// permute fixup is exercised on the alpha lane just like R/G/B.

fn check_yuv444p_n_u8_avx2_rgba_with_alpha_src_equivalence<const BITS: u32>(
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
    "AVX2 Yuva444p<{BITS}> → RGBA u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range}, alpha_seed={alpha_seed})"
  );
}

#[test]
fn avx2_yuva444p10_rgba_matches_scalar_all_matrices_32() {
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
      check_yuv444p_n_u8_avx2_rgba_with_alpha_src_equivalence::<10>(32, m, full, 89);
    }
  }
}

#[test]
fn avx2_yuva444p10_rgba_matches_scalar_widths() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  // Natural width + tail widths forcing scalar-tail dispatch.
  for w in [32usize, 17, 31, 47, 63, 1920, 1922] {
    check_yuv444p_n_u8_avx2_rgba_with_alpha_src_equivalence::<10>(w, ColorMatrix::Bt709, true, 89);
  }
}

#[test]
fn avx2_yuva444p10_rgba_matches_scalar_random_alpha() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  // Different alpha seeds — `_mm256_packus_epi16` followed by the
  // `narrow_u8x32` per-lane permute fixup must place alpha in the 4th
  // channel without lane-order corruption.
  for seed in [13usize, 41, 89, 127, 211] {
    check_yuv444p_n_u8_avx2_rgba_with_alpha_src_equivalence::<10>(
      32,
      ColorMatrix::Bt601,
      false,
      seed,
    );
    check_yuv444p_n_u8_avx2_rgba_with_alpha_src_equivalence::<10>(
      47,
      ColorMatrix::Bt2020Ncl,
      true,
      seed,
    );
  }
}

#[test]
fn avx2_yuva444p_n_rgba_matches_scalar_all_bits() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  // BITS = 9, 12, 14 (BITS = 10 covered above). Confirms
  // `_mm256_srl_epi16` with count `(BITS - 8)` resolves correctly
  // across the supported bit depths.
  for full in [true, false] {
    check_yuv444p_n_u8_avx2_rgba_with_alpha_src_equivalence::<9>(32, ColorMatrix::Bt601, full, 53);
    check_yuv444p_n_u8_avx2_rgba_with_alpha_src_equivalence::<12>(32, ColorMatrix::Bt709, full, 53);
    check_yuv444p_n_u8_avx2_rgba_with_alpha_src_equivalence::<14>(
      32,
      ColorMatrix::Bt2020Ncl,
      full,
      53,
    );
  }
}

#[test]
fn avx2_yuva444p_n_rgba_matches_scalar_all_bits_widths() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [17usize, 47, 1922] {
    check_yuv444p_n_u8_avx2_rgba_with_alpha_src_equivalence::<9>(
      w,
      ColorMatrix::Smpte240m,
      false,
      89,
    );
    check_yuv444p_n_u8_avx2_rgba_with_alpha_src_equivalence::<12>(w, ColorMatrix::Fcc, true, 89);
    check_yuv444p_n_u8_avx2_rgba_with_alpha_src_equivalence::<14>(w, ColorMatrix::YCgCo, false, 89);
  }
}

// ---- YUVA 4:2:0 u8 RGBA equivalence (Ship 8b‑2b) --------------------

fn check_yuv_420_u8_avx2_rgba_with_alpha_src_equivalence(
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
    "AVX2 Yuva420p (8-bit) → RGBA u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range}, alpha_seed={alpha_seed})"
  );
}

fn check_yuv420p_n_u8_avx2_rgba_with_alpha_src_equivalence<const BITS: u32>(
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
    "AVX2 Yuva420p<{BITS}> → RGBA u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range}, alpha_seed={alpha_seed})"
  );
}

fn check_yuv420p16_u8_avx2_rgba_with_alpha_src_equivalence(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  alpha_seed: usize,
) {
  let y = p16_plane_avx2(width, 37);
  let u = p16_plane_avx2(width / 2, 53);
  let v = p16_plane_avx2(width / 2, 71);
  let a_src = p16_plane_avx2(width, alpha_seed);
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
    "AVX2 Yuva420p16 → RGBA u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range}, alpha_seed={alpha_seed})"
  );
}

#[test]
fn avx2_yuva420p_rgba_matches_scalar_all_matrices() {
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
      check_yuv_420_u8_avx2_rgba_with_alpha_src_equivalence(32, m, full, 89);
    }
  }
}

#[test]
fn avx2_yuva420p_rgba_matches_scalar_widths_and_alpha() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [32usize, 34, 46, 62, 1920, 1922] {
    check_yuv_420_u8_avx2_rgba_with_alpha_src_equivalence(w, ColorMatrix::Bt709, true, 89);
  }
  for seed in [13usize, 41, 127, 211] {
    check_yuv_420_u8_avx2_rgba_with_alpha_src_equivalence(32, ColorMatrix::Bt601, false, seed);
  }
}

#[test]
fn avx2_yuva420p_n_rgba_matches_scalar_all_bits() {
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
      check_yuv420p_n_u8_avx2_rgba_with_alpha_src_equivalence::<9>(32, m, full, 89);
      check_yuv420p_n_u8_avx2_rgba_with_alpha_src_equivalence::<10>(32, m, full, 89);
      check_yuv420p_n_u8_avx2_rgba_with_alpha_src_equivalence::<12>(32, m, full, 89);
    }
  }
}

#[test]
fn avx2_yuva420p_n_rgba_matches_scalar_widths() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [32usize, 34, 46, 62, 1920, 1922] {
    check_yuv420p_n_u8_avx2_rgba_with_alpha_src_equivalence::<9>(w, ColorMatrix::Bt601, false, 89);
    check_yuv420p_n_u8_avx2_rgba_with_alpha_src_equivalence::<10>(w, ColorMatrix::Bt709, true, 89);
    check_yuv420p_n_u8_avx2_rgba_with_alpha_src_equivalence::<12>(w, ColorMatrix::Bt709, true, 89);
  }
}

#[test]
fn avx2_yuva420p16_rgba_matches_scalar_all_matrices() {
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
      check_yuv420p16_u8_avx2_rgba_with_alpha_src_equivalence(32, m, full, 89);
    }
  }
}

#[test]
fn avx2_yuva420p16_rgba_matches_scalar_widths_and_alpha() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [32usize, 34, 46, 62, 1920, 1922] {
    check_yuv420p16_u8_avx2_rgba_with_alpha_src_equivalence(w, ColorMatrix::Bt709, false, 89);
  }
  for seed in [13usize, 41, 127, 211] {
    check_yuv420p16_u8_avx2_rgba_with_alpha_src_equivalence(32, ColorMatrix::Bt601, true, seed);
  }
}

// ---- High-bit 4:4:4 native-depth `u16` RGBA equivalence (Ship 8 Tranche 7c) ----

fn check_yuv444p_n_u16_avx2_rgba_equivalence<const BITS: u32>(
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
    "AVX2 Yuv444p<{BITS}> → RGBA u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_pn_444_u16_avx2_rgba_equivalence<const BITS: u32>(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let y = high_bit_plane_avx2::<BITS>(width, 37);
  let u = high_bit_plane_avx2::<BITS>(width, 53);
  let v = high_bit_plane_avx2::<BITS>(width, 71);
  let uv = interleave_uv_avx2(&u, &v);
  let mut rgba_scalar = std::vec![0u16; width * 4];
  let mut rgba_simd = std::vec![0u16; width * 4];
  scalar::p_n_444_to_rgba_u16_row::<BITS>(&y, &uv, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    p_n_444_to_rgba_u16_row::<BITS>(&y, &uv, &mut rgba_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgba_scalar, rgba_simd,
    "AVX2 Pn4:4:4<{BITS}> → RGBA u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_yuv444p16_u16_avx2_rgba_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y = p16_plane_avx2(width, 37);
  let u = p16_plane_avx2(width, 53);
  let v = p16_plane_avx2(width, 71);
  let mut rgba_scalar = std::vec![0u16; width * 4];
  let mut rgba_simd = std::vec![0u16; width * 4];
  scalar::yuv_444p16_to_rgba_u16_row(&y, &u, &v, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    yuv_444p16_to_rgba_u16_row(&y, &u, &v, &mut rgba_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgba_scalar, rgba_simd,
    "AVX2 Yuv444p16 → RGBA u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_p_n_444_16_u16_avx2_rgba_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y = p16_plane_avx2(width, 37);
  let u = p16_plane_avx2(width, 53);
  let v = p16_plane_avx2(width, 71);
  let uv = interleave_uv_avx2(&u, &v);
  let mut rgba_scalar = std::vec![0u16; width * 4];
  let mut rgba_simd = std::vec![0u16; width * 4];
  scalar::p_n_444_16_to_rgba_u16_row(&y, &uv, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    p_n_444_16_to_rgba_u16_row(&y, &uv, &mut rgba_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgba_scalar, rgba_simd,
    "AVX2 P416 → RGBA u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[test]
fn avx2_yuv444p_n_rgba_u16_matches_scalar_all_bits() {
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
      check_yuv444p_n_u16_avx2_rgba_equivalence::<9>(32, m, full);
      check_yuv444p_n_u16_avx2_rgba_equivalence::<10>(32, m, full);
      check_yuv444p_n_u16_avx2_rgba_equivalence::<12>(32, m, full);
      check_yuv444p_n_u16_avx2_rgba_equivalence::<14>(32, m, full);
    }
  }
}

#[test]
fn avx2_yuv444p_n_rgba_u16_matches_scalar_tail_and_widths() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [17usize, 31, 47, 63, 1920, 1922] {
    check_yuv444p_n_u16_avx2_rgba_equivalence::<9>(w, ColorMatrix::Bt601, false);
    check_yuv444p_n_u16_avx2_rgba_equivalence::<10>(w, ColorMatrix::Bt709, true);
    check_yuv444p_n_u16_avx2_rgba_equivalence::<12>(w, ColorMatrix::Bt2020Ncl, false);
    check_yuv444p_n_u16_avx2_rgba_equivalence::<14>(w, ColorMatrix::YCgCo, true);
  }
}

#[test]
fn avx2_pn_444_rgba_u16_matches_scalar_all_bits() {
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
      check_pn_444_u16_avx2_rgba_equivalence::<10>(32, m, full);
      check_pn_444_u16_avx2_rgba_equivalence::<12>(32, m, full);
    }
  }
}

#[test]
fn avx2_pn_444_rgba_u16_matches_scalar_tail_and_widths() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [17usize, 31, 47, 63, 1920, 1922] {
    check_pn_444_u16_avx2_rgba_equivalence::<10>(w, ColorMatrix::Bt601, false);
    check_pn_444_u16_avx2_rgba_equivalence::<12>(w, ColorMatrix::Bt709, true);
  }
}

#[test]
fn avx2_yuv444p16_rgba_u16_matches_scalar_all_matrices() {
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
      check_yuv444p16_u16_avx2_rgba_equivalence(16, m, full);
    }
  }
  for w in [17usize, 31, 47, 63, 1920, 1922] {
    check_yuv444p16_u16_avx2_rgba_equivalence(w, ColorMatrix::Bt709, false);
  }
}

fn check_yuv444p16_u16_avx2_rgba_with_alpha_src_equivalence(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  alpha_seed: usize,
) {
  let y = p16_plane_avx2(width, 37);
  let u = p16_plane_avx2(width, 53);
  let v = p16_plane_avx2(width, 71);
  let a_src = p16_plane_avx2(width, alpha_seed);
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
    "AVX2 Yuva444p16 → RGBA u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range}, alpha_seed={alpha_seed})"
  );
}

#[test]
fn avx2_yuva444p16_rgba_u16_matches_scalar_all_matrices() {
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
      check_yuv444p16_u16_avx2_rgba_with_alpha_src_equivalence(16, m, full, 89);
    }
  }
}

#[test]
fn avx2_yuva444p16_rgba_u16_matches_scalar_widths_and_alpha() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [16usize, 17, 31, 47, 63, 1920, 1922] {
    check_yuv444p16_u16_avx2_rgba_with_alpha_src_equivalence(w, ColorMatrix::Bt709, true, 89);
  }
  for seed in [13usize, 41, 127, 211] {
    check_yuv444p16_u16_avx2_rgba_with_alpha_src_equivalence(16, ColorMatrix::Bt601, false, seed);
  }
}

#[test]
fn avx2_p416_rgba_u16_matches_scalar_all_matrices() {
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
      check_p_n_444_16_u16_avx2_rgba_equivalence(16, m, full);
    }
  }
  for w in [17usize, 31, 47, 63, 1920, 1922] {
    check_p_n_444_16_u16_avx2_rgba_equivalence(w, ColorMatrix::Bt709, false);
  }
}

// ---- YUVA 4:4:4 native-depth `u16` RGBA equivalence (Ship 8b‑1c) ----
//
// Mirrors the u8 RGBA alpha-source tests above for the u16 output
// path: per-pixel alpha element is loaded from the source plane,
// AND-masked with `bits_mask::<10>()`, and stored at native depth (no
// `>> (BITS - 8)` since both source alpha and output element are at
// the same bit depth). 32 px per iter → 16 alpha u16 per `__m256i`
// load × 2 halves; per-half splits into two `__m128i` quarters fed to
// the four `write_rgba_u16_8` calls per iter.

fn check_yuv444p_n_u16_avx2_rgba_with_alpha_src_equivalence<const BITS: u32>(
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
    "AVX2 Yuva444p<{BITS}> → RGBA u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range}, alpha_seed={alpha_seed})"
  );
}

#[test]
fn avx2_yuva444p10_rgba_u16_matches_scalar_all_matrices_32() {
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
      check_yuv444p_n_u16_avx2_rgba_with_alpha_src_equivalence::<10>(32, m, full, 89);
    }
  }
}

#[test]
fn avx2_yuva444p10_rgba_u16_matches_scalar_widths() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  // Natural width + tail widths forcing scalar-tail dispatch.
  for w in [32usize, 17, 31, 47, 63, 1920, 1922] {
    check_yuv444p_n_u16_avx2_rgba_with_alpha_src_equivalence::<10>(w, ColorMatrix::Bt709, true, 89);
  }
}

#[test]
fn avx2_yuva444p10_rgba_u16_matches_scalar_random_alpha() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  // Different alpha seeds — the 256-bit alpha load splits into two
  // 128-bit quarters via `_mm256_castsi256_si128` /
  // `_mm256_extracti128_si256::<1>`; the lane order through
  // `write_rgba_u16_8` must put alpha in the 4th channel.
  for seed in [13usize, 41, 89, 127, 211] {
    check_yuv444p_n_u16_avx2_rgba_with_alpha_src_equivalence::<10>(
      32,
      ColorMatrix::Bt601,
      false,
      seed,
    );
    check_yuv444p_n_u16_avx2_rgba_with_alpha_src_equivalence::<10>(
      63,
      ColorMatrix::Bt2020Ncl,
      true,
      seed,
    );
  }
}

#[test]
fn avx2_yuva444p_n_rgba_u16_matches_scalar_all_bits() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  // BITS = 9, 12, 14 (BITS = 10 covered above). Confirms the
  // AND-mask `mask_v` resolves correctly across the supported bit
  // depths.
  for full in [true, false] {
    check_yuv444p_n_u16_avx2_rgba_with_alpha_src_equivalence::<9>(32, ColorMatrix::Bt601, full, 53);
    check_yuv444p_n_u16_avx2_rgba_with_alpha_src_equivalence::<12>(
      32,
      ColorMatrix::Bt709,
      full,
      53,
    );
    check_yuv444p_n_u16_avx2_rgba_with_alpha_src_equivalence::<14>(
      32,
      ColorMatrix::Bt2020Ncl,
      full,
      53,
    );
  }
}

#[test]
fn avx2_yuva444p_n_rgba_u16_matches_scalar_all_bits_widths() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [33usize, 47, 1922] {
    check_yuv444p_n_u16_avx2_rgba_with_alpha_src_equivalence::<9>(
      w,
      ColorMatrix::Smpte240m,
      false,
      89,
    );
    check_yuv444p_n_u16_avx2_rgba_with_alpha_src_equivalence::<12>(w, ColorMatrix::Fcc, true, 89);
    check_yuv444p_n_u16_avx2_rgba_with_alpha_src_equivalence::<14>(
      w,
      ColorMatrix::YCgCo,
      false,
      89,
    );
  }
}

// ---- YUVA 4:2:0 native-depth `u16` RGBA equivalence (Ship 8b‑2c) ----

fn check_yuv420p_n_u16_avx2_rgba_with_alpha_src_equivalence<const BITS: u32>(
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
    "AVX2 Yuva420p<{BITS}> → RGBA u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range}, alpha_seed={alpha_seed})"
  );
}

fn check_yuv420p16_u16_avx2_rgba_with_alpha_src_equivalence(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  alpha_seed: usize,
) {
  let y = p16_plane_avx2(width, 37);
  let u = p16_plane_avx2(width / 2, 53);
  let v = p16_plane_avx2(width / 2, 71);
  let a_src = p16_plane_avx2(width, alpha_seed);
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
    "AVX2 Yuva420p16 → RGBA u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range}, alpha_seed={alpha_seed})"
  );
}

#[test]
fn avx2_yuva420p_n_rgba_u16_matches_scalar_all_bits() {
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
      check_yuv420p_n_u16_avx2_rgba_with_alpha_src_equivalence::<9>(32, m, full, 89);
      check_yuv420p_n_u16_avx2_rgba_with_alpha_src_equivalence::<10>(32, m, full, 89);
      check_yuv420p_n_u16_avx2_rgba_with_alpha_src_equivalence::<12>(32, m, full, 89);
    }
  }
}

#[test]
fn avx2_yuva420p_n_rgba_u16_matches_scalar_widths() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [32usize, 34, 46, 62, 1920, 1922] {
    check_yuv420p_n_u16_avx2_rgba_with_alpha_src_equivalence::<9>(w, ColorMatrix::Bt601, false, 89);
    check_yuv420p_n_u16_avx2_rgba_with_alpha_src_equivalence::<10>(w, ColorMatrix::Bt709, true, 89);
    check_yuv420p_n_u16_avx2_rgba_with_alpha_src_equivalence::<12>(w, ColorMatrix::Bt709, true, 89);
  }
}

#[test]
fn avx2_yuva420p16_rgba_u16_matches_scalar_all_matrices() {
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
      check_yuv420p16_u16_avx2_rgba_with_alpha_src_equivalence(16, m, full, 89);
    }
  }
}

#[test]
fn avx2_yuva420p16_rgba_u16_matches_scalar_widths_and_alpha() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [16usize, 18, 30, 34, 1920, 1922] {
    check_yuv420p16_u16_avx2_rgba_with_alpha_src_equivalence(w, ColorMatrix::Bt709, false, 89);
  }
  for seed in [13usize, 41, 127, 211] {
    check_yuv420p16_u16_avx2_rgba_with_alpha_src_equivalence(16, ColorMatrix::Bt601, true, seed);
  }
}

// ---- Ship 9b RGBA/BGRA shuffles -----------------------------------------

fn pseudo_random_rgba(width: usize) -> std::vec::Vec<u8> {
  (0..width * 4)
    .map(|i| ((i * 17 + 41) & 0xFF) as u8)
    .collect()
}

#[test]
fn avx2_rgba_to_rgb_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [1usize, 15, 31, 32, 33, 63, 64, 65, 1920, 1921] {
    let input = pseudo_random_rgba(w);
    let mut out_scalar = std::vec![0u8; w * 3];
    let mut out_avx = std::vec![0u8; w * 3];
    scalar::rgba_to_rgb_row(&input, &mut out_scalar, w);
    unsafe {
      rgba_to_rgb_row(&input, &mut out_avx, w);
    }
    assert_eq!(out_scalar, out_avx, "AVX2 rgba_to_rgb diverges (width={w})");
  }
}

#[test]
fn avx2_bgra_to_rgba_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [1usize, 15, 31, 32, 33, 63, 64, 65, 1920, 1921] {
    let input = pseudo_random_rgba(w);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_avx = std::vec![0u8; w * 4];
    scalar::bgra_to_rgba_row(&input, &mut out_scalar, w);
    unsafe {
      bgra_to_rgba_row(&input, &mut out_avx, w);
    }
    assert_eq!(
      out_scalar, out_avx,
      "AVX2 bgra_to_rgba diverges (width={w})"
    );
  }
}

#[test]
fn avx2_bgra_to_rgb_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [1usize, 15, 31, 32, 33, 63, 64, 65, 1920, 1921] {
    let input = pseudo_random_rgba(w);
    let mut out_scalar = std::vec![0u8; w * 3];
    let mut out_avx = std::vec![0u8; w * 3];
    scalar::bgra_to_rgb_row(&input, &mut out_scalar, w);
    unsafe {
      bgra_to_rgb_row(&input, &mut out_avx, w);
    }
    assert_eq!(out_scalar, out_avx, "AVX2 bgra_to_rgb diverges (width={w})");
  }
}

// ---- Ship 9c leading-alpha shuffles -----------------------------------

#[test]
fn avx2_argb_to_rgb_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [1usize, 15, 31, 32, 33, 63, 64, 65, 1920, 1921] {
    let input = pseudo_random_rgba(w);
    let mut out_scalar = std::vec![0u8; w * 3];
    let mut out_avx = std::vec![0u8; w * 3];
    scalar::argb_to_rgb_row(&input, &mut out_scalar, w);
    unsafe {
      argb_to_rgb_row(&input, &mut out_avx, w);
    }
    assert_eq!(out_scalar, out_avx, "AVX2 argb_to_rgb diverges (width={w})");
  }
}

#[test]
fn avx2_abgr_to_rgb_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [1usize, 15, 31, 32, 33, 63, 64, 65, 1920, 1921] {
    let input = pseudo_random_rgba(w);
    let mut out_scalar = std::vec![0u8; w * 3];
    let mut out_avx = std::vec![0u8; w * 3];
    scalar::abgr_to_rgb_row(&input, &mut out_scalar, w);
    unsafe {
      abgr_to_rgb_row(&input, &mut out_avx, w);
    }
    assert_eq!(out_scalar, out_avx, "AVX2 abgr_to_rgb diverges (width={w})");
  }
}

#[test]
fn avx2_argb_to_rgba_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [1usize, 15, 31, 32, 33, 63, 64, 65, 1920, 1921] {
    let input = pseudo_random_rgba(w);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_avx = std::vec![0u8; w * 4];
    scalar::argb_to_rgba_row(&input, &mut out_scalar, w);
    unsafe {
      argb_to_rgba_row(&input, &mut out_avx, w);
    }
    assert_eq!(
      out_scalar, out_avx,
      "AVX2 argb_to_rgba diverges (width={w})"
    );
  }
}

#[test]
fn avx2_abgr_to_rgba_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [1usize, 15, 31, 32, 33, 63, 64, 65, 1920, 1921] {
    let input = pseudo_random_rgba(w);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_avx = std::vec![0u8; w * 4];
    scalar::abgr_to_rgba_row(&input, &mut out_scalar, w);
    unsafe {
      abgr_to_rgba_row(&input, &mut out_avx, w);
    }
    assert_eq!(
      out_scalar, out_avx,
      "AVX2 abgr_to_rgba diverges (width={w})"
    );
  }
}

// ---- Ship 9d padding-byte shuffles -----------------------------------

#[test]
fn avx2_xrgb_to_rgba_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [1usize, 15, 31, 32, 33, 63, 64, 65, 1920, 1921] {
    let input = pseudo_random_rgba(w);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_avx = std::vec![0u8; w * 4];
    scalar::xrgb_to_rgba_row(&input, &mut out_scalar, w);
    unsafe {
      xrgb_to_rgba_row(&input, &mut out_avx, w);
    }
    assert_eq!(
      out_scalar, out_avx,
      "AVX2 xrgb_to_rgba diverges (width={w})"
    );
  }
}

#[test]
fn avx2_rgbx_to_rgba_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [1usize, 15, 31, 32, 33, 63, 64, 65, 1920, 1921] {
    let input = pseudo_random_rgba(w);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_avx = std::vec![0u8; w * 4];
    scalar::rgbx_to_rgba_row(&input, &mut out_scalar, w);
    unsafe {
      rgbx_to_rgba_row(&input, &mut out_avx, w);
    }
    assert_eq!(
      out_scalar, out_avx,
      "AVX2 rgbx_to_rgba diverges (width={w})"
    );
  }
}

#[test]
fn avx2_xbgr_to_rgba_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [1usize, 15, 31, 32, 33, 63, 64, 65, 1920, 1921] {
    let input = pseudo_random_rgba(w);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_avx = std::vec![0u8; w * 4];
    scalar::xbgr_to_rgba_row(&input, &mut out_scalar, w);
    unsafe {
      xbgr_to_rgba_row(&input, &mut out_avx, w);
    }
    assert_eq!(
      out_scalar, out_avx,
      "AVX2 xbgr_to_rgba diverges (width={w})"
    );
  }
}

#[test]
fn avx2_bgrx_to_rgba_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [1usize, 15, 31, 32, 33, 63, 64, 65, 1920, 1921] {
    let input = pseudo_random_rgba(w);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_avx = std::vec![0u8; w * 4];
    scalar::bgrx_to_rgba_row(&input, &mut out_scalar, w);
    unsafe {
      bgrx_to_rgba_row(&input, &mut out_avx, w);
    }
    assert_eq!(
      out_scalar, out_avx,
      "AVX2 bgrx_to_rgba diverges (width={w})"
    );
  }
}
