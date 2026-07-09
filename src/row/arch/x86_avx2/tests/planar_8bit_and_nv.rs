use super::super::*;

#[cfg(feature = "yuv-planar")]
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

#[cfg(feature = "yuv-planar")]
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

#[cfg(feature = "yuv-planar")]
#[test]
fn avx2_matches_scalar_width_64() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  check_equivalence(64, ColorMatrix::Bt601, true);
  check_equivalence(64, ColorMatrix::Bt709, false);
  check_equivalence(64, ColorMatrix::YCgCo, true);
}

#[cfg(feature = "yuv-planar")]
#[test]
fn avx2_matches_scalar_width_1920() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  check_equivalence(1920, ColorMatrix::Bt709, false);
}

#[cfg(feature = "yuv-planar")]
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

#[cfg(feature = "yuv-planar")]
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

#[cfg(feature = "yuv-planar")]
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

#[cfg(feature = "yuv-planar")]
#[test]
fn avx2_rgba_matches_scalar_width_64() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  check_rgba_equivalence(64, ColorMatrix::Bt601, true);
  check_rgba_equivalence(64, ColorMatrix::Bt709, false);
  check_rgba_equivalence(64, ColorMatrix::YCgCo, true);
}

#[cfg(feature = "yuv-planar")]
#[test]
fn avx2_rgba_matches_scalar_width_1920() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  check_rgba_equivalence(1920, ColorMatrix::Bt709, false);
}

#[cfg(feature = "yuv-planar")]
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

#[cfg(feature = "yuv-semi-planar")]
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

#[cfg(all(feature = "yuv-planar", feature = "yuv-semi-planar"))]
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

#[cfg(feature = "yuv-semi-planar")]
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

#[cfg(feature = "yuv-semi-planar")]
#[test]
fn avx2_nv12_matches_scalar_widths() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [64usize, 1920, 34, 46, 62, 1922] {
    check_nv12_equivalence(w, ColorMatrix::Bt709, false);
  }
}

#[cfg(all(feature = "yuv-planar", feature = "yuv-semi-planar"))]
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

#[cfg(feature = "yuv-semi-planar")]
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

#[cfg(feature = "yuv-semi-planar")]
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

#[cfg(feature = "yuv-semi-planar")]
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

#[cfg(feature = "yuv-semi-planar")]
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

#[cfg(feature = "yuv-semi-planar")]
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

#[cfg(feature = "yuv-semi-planar")]
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

#[cfg(feature = "yuv-semi-planar")]
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

#[cfg(feature = "yuv-semi-planar")]
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

#[cfg(feature = "yuv-semi-planar")]
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

#[cfg(feature = "yuv-semi-planar")]
#[test]
fn avx2_nv24_rgba_matches_scalar_widths() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [31usize, 32, 33, 63, 64, 65, 1920, 1921] {
    check_nv24_rgba_equivalence(w, ColorMatrix::Bt709, false);
  }
}

#[cfg(feature = "yuv-semi-planar")]
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

#[cfg(feature = "yuv-semi-planar")]
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

#[cfg(feature = "yuv-planar")]
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

#[cfg(feature = "yuv-planar")]
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

#[cfg(feature = "yuv-planar")]
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

#[cfg(feature = "yuv-planar")]
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

#[cfg(feature = "yuv-planar")]
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

#[cfg(feature = "yuv-planar")]
#[test]
fn avx2_yuv_444_rgba_matches_scalar_widths() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [31usize, 32, 33, 63, 64, 65, 1920, 1921] {
    check_yuv_444_rgba_equivalence(w, ColorMatrix::Bt709, false);
  }
}

#[cfg(feature = "yuva")]
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

#[cfg(feature = "yuva")]
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

#[cfg(feature = "yuva")]
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

#[cfg(feature = "yuv-planar")]
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

  scalar::yuv_444p_n_to_rgb_row::<BITS, false>(
    &y,
    &u,
    &v,
    &mut rgb_scalar,
    width,
    matrix,
    full_range,
  );
  scalar::yuv_444p_n_to_rgb_u16_row::<BITS, false>(
    &y,
    &u,
    &v,
    &mut u16_scalar,
    width,
    matrix,
    full_range,
  );
  unsafe {
    yuv_444p_n_to_rgb_row::<BITS, false>(&y, &u, &v, &mut rgb_avx2, width, matrix, full_range);
    yuv_444p_n_to_rgb_u16_row::<BITS, false>(&y, &u, &v, &mut u16_avx2, width, matrix, full_range);
  }
  assert_eq!(rgb_scalar, rgb_avx2, "AVX2 yuv_444p_n<{BITS}> u8 ≠ scalar");
  assert_eq!(u16_scalar, u16_avx2, "AVX2 yuv_444p_n<{BITS}> u16 ≠ scalar");
}

#[cfg(feature = "yuv-planar")]
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

#[cfg(feature = "yuv-planar")]
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

#[cfg(feature = "yuv-planar")]
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

#[cfg(feature = "yuv-planar")]
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

#[cfg(feature = "yuv-planar")]
#[test]
fn avx2_yuv_444p_n_matches_scalar_widths() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [1usize, 3, 15, 17, 31, 32, 33, 63, 64, 65, 1920, 1921] {
    check_yuv_444p_n_equivalence::<10>(w, ColorMatrix::Bt709, false);
  }
}

#[cfg(feature = "yuv-planar")]
fn check_yuv_444p16_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y: std::vec::Vec<u16> = (0..width).map(|i| (i * 2027 + 11) as u16).collect();
  let u: std::vec::Vec<u16> = (0..width).map(|i| (i * 2671 + 23) as u16).collect();
  let v: std::vec::Vec<u16> = (0..width).map(|i| (i * 3329 + 91) as u16).collect();
  let mut rgb_scalar = std::vec![0u8; width * 3];
  let mut rgb_avx2 = std::vec![0u8; width * 3];
  let mut u16_scalar = std::vec![0u16; width * 3];
  let mut u16_avx2 = std::vec![0u16; width * 3];

  scalar::yuv_444p16_to_rgb_row::<false>(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
  scalar::yuv_444p16_to_rgb_u16_row::<false>(
    &y,
    &u,
    &v,
    &mut u16_scalar,
    width,
    matrix,
    full_range,
  );
  unsafe {
    yuv_444p16_to_rgb_row::<false>(&y, &u, &v, &mut rgb_avx2, width, matrix, full_range);
    yuv_444p16_to_rgb_u16_row::<false>(&y, &u, &v, &mut u16_avx2, width, matrix, full_range);
  }
  assert_eq!(rgb_scalar, rgb_avx2, "AVX2 yuv_444p16 u8 ≠ scalar");
  assert_eq!(u16_scalar, u16_avx2, "AVX2 yuv_444p16 u16 ≠ scalar");
}

#[cfg(feature = "yuv-planar")]
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

#[cfg(feature = "yuv-planar")]
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

#[cfg(feature = "rgb")]
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

#[cfg(feature = "rgb")]
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

#[cfg(feature = "yuv-semi-planar")]
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

#[cfg(feature = "yuv-semi-planar")]
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

#[cfg(feature = "yuv-semi-planar")]
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

#[cfg(feature = "yuv-semi-planar")]
#[test]
fn nv21_avx2_matches_scalar_widths() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [32usize, 1920, 18, 30, 34, 1922] {
    check_nv21_equivalence(w, ColorMatrix::Bt709, false);
  }
}

#[cfg(feature = "yuv-semi-planar")]
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

#[cfg(feature = "yuv-semi-planar")]
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

#[cfg(feature = "yuv-semi-planar")]
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

#[cfg(all(feature = "yuv-planar", feature = "yuv-semi-planar"))]
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

#[cfg(feature = "yuv-semi-planar")]
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

#[cfg(feature = "yuv-semi-planar")]
#[test]
fn nv12_avx2_rgba_matches_scalar_widths() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [34usize, 46, 62, 1920, 1922] {
    check_nv12_rgba_equivalence(w, ColorMatrix::Bt601, false);
  }
}

#[cfg(all(feature = "yuv-planar", feature = "yuv-semi-planar"))]
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

#[cfg(feature = "yuv-semi-planar")]
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

#[cfg(feature = "yuv-semi-planar")]
#[test]
fn nv21_avx2_rgba_matches_scalar_widths() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [34usize, 46, 62, 1920, 1922] {
    check_nv21_rgba_equivalence(w, ColorMatrix::Bt601, false);
  }
}

// ---- yuv_410_to_rgb_row / yuv_410_to_rgba_row equivalence ------------
//
// AVX2 4:1:0 parity with scalar — 32 Y / 8 chroma per iter; chroma
// fan-out via per-128-bit-lane two-pass `_mm_unpack*_epi16` cascade
// (no lane-crossing needed because chroma split into low4/high4 is
// already lane-aligned). Width must be a multiple of 4.

#[cfg(feature = "yuv-planar")]
fn check_yuv_410_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
  let cw = width / 4;
  let u: std::vec::Vec<u8> = (0..cw).map(|i| ((i * 53 + 23) & 0xFF) as u8).collect();
  let v: std::vec::Vec<u8> = (0..cw).map(|i| ((i * 71 + 91) & 0xFF) as u8).collect();
  let mut rgb_scalar = std::vec![0u8; width * 3];
  let mut rgb_avx2 = std::vec![0u8; width * 3];

  scalar::yuv_410_to_rgb_row(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    yuv_410_to_rgb_row(&y, &u, &v, &mut rgb_avx2, width, matrix, full_range);
  }

  if rgb_scalar != rgb_avx2 {
    let first_diff = rgb_scalar
      .iter()
      .zip(rgb_avx2.iter())
      .position(|(a, b)| a != b)
      .unwrap();
    panic!(
      "AVX2 yuv_410 diverges from scalar at byte {first_diff} (width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} avx2={}",
      rgb_scalar[first_diff], rgb_avx2[first_diff]
    );
  }
}

#[cfg(feature = "yuv-planar")]
#[test]
fn yuv_410_avx2_matches_scalar_all_matrices_32() {
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
      check_yuv_410_equivalence(32, m, full);
    }
  }
}

#[cfg(feature = "yuv-planar")]
#[test]
fn yuv_410_avx2_matches_scalar_widths() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  // Cover: pure-scalar (4, 8, 12, 16, 20, 28 — all < 32), exactly one
  // SIMD iter (32), SIMD + tail (36, 60), multi-iter (64, 96, 1920).
  for &w in &[4usize, 8, 12, 16, 20, 28, 32, 36, 60, 64, 96, 128, 1920] {
    check_yuv_410_equivalence(w, ColorMatrix::Bt601, true);
    check_yuv_410_equivalence(w, ColorMatrix::Bt709, false);
  }
}

#[cfg(feature = "yuv-planar")]
#[test]
fn yuv_410_avx2_matches_scalar_bt2020() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for &w in &[32usize, 60, 64, 1920] {
    check_yuv_410_equivalence(w, ColorMatrix::Bt2020Ncl, false);
    check_yuv_410_equivalence(w, ColorMatrix::Bt2020Ncl, true);
  }
}

#[cfg(feature = "yuv-planar")]
fn check_yuv_410_rgba_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
  let cw = width / 4;
  let u: std::vec::Vec<u8> = (0..cw).map(|i| ((i * 53 + 23) & 0xFF) as u8).collect();
  let v: std::vec::Vec<u8> = (0..cw).map(|i| ((i * 71 + 91) & 0xFF) as u8).collect();
  let mut rgba_scalar = std::vec![0u8; width * 4];
  let mut rgba_avx2 = std::vec![0u8; width * 4];

  scalar::yuv_410_to_rgba_row(&y, &u, &v, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    yuv_410_to_rgba_row(&y, &u, &v, &mut rgba_avx2, width, matrix, full_range);
  }

  if rgba_scalar != rgba_avx2 {
    let first_diff = rgba_scalar
      .iter()
      .zip(rgba_avx2.iter())
      .position(|(a, b)| a != b)
      .unwrap();
    panic!(
      "AVX2 yuv_410 RGBA diverges from scalar at byte {first_diff} (width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} avx2={}",
      rgba_scalar[first_diff], rgba_avx2[first_diff]
    );
  }

  for (i, px) in rgba_avx2.chunks(4).enumerate() {
    assert_eq!(px[3], 0xFF, "alpha at pixel {i} must be 0xFF");
  }
}

#[cfg(feature = "yuv-planar")]
#[test]
fn yuv_410_avx2_rgba_matches_scalar_widths() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for &w in &[4usize, 16, 32, 36, 64, 128] {
    check_yuv_410_rgba_equivalence(w, ColorMatrix::Bt601, true);
    check_yuv_410_rgba_equivalence(w, ColorMatrix::YCgCo, false);
  }
}

// ---- yuv_411_to_rgb_row equivalence (AVX2 ↔ scalar) ------------------
//
// Direct backend test for the 4:1:1 path: bypasses the public dispatcher
// so the AVX2 1→4 chroma upsample is exercised regardless of what tier
// the dispatcher would pick on the current runner.

#[cfg(feature = "yuv-planar")]
fn check_yuv411_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  // FFmpeg `AV_PIX_FMT_YUV411P`: chroma row = `width.div_ceil(4)`.
  assert!(width > 0);
  let chroma_w = width.div_ceil(4);
  let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
  let u: std::vec::Vec<u8> = (0..chroma_w)
    .map(|i| ((i * 53 + 23) & 0xFF) as u8)
    .collect();
  let v: std::vec::Vec<u8> = (0..chroma_w)
    .map(|i| ((i * 71 + 91) & 0xFF) as u8)
    .collect();
  let mut rgb_scalar = std::vec![0u8; width * 3];
  let mut rgb_simd = std::vec![0u8; width * 3];

  scalar::yuv_411_to_rgb_row(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    yuv_411_to_rgb_row(&y, &u, &v, &mut rgb_simd, width, matrix, full_range);
  }

  if rgb_scalar != rgb_simd {
    let first_diff = rgb_scalar
      .iter()
      .zip(rgb_simd.iter())
      .position(|(a, b)| a != b)
      .unwrap();
    panic!(
      "AVX2 yuv_411 diverges from scalar at byte {first_diff} (width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} avx2={}",
      rgb_scalar[first_diff], rgb_simd[first_diff]
    );
  }
}

#[cfg(feature = "yuv-planar")]
#[test]
#[cfg_attr(miri, ignore = "AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_yuv411_matches_scalar_all_matrices_32() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  // Width 32 = exactly one AVX2 iteration with no scalar tail.
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_yuv411_equivalence(32, m, full);
    }
  }
}

#[cfg(feature = "yuv-planar")]
#[test]
#[cfg_attr(miri, ignore = "AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_yuv411_matches_scalar_tail_widths() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  // Widths that leave a non-trivial scalar tail (not multiple of 32
  // but multiple of 4).
  for w in [4usize, 8, 12, 20, 28, 36, 44, 60, 100, 132, 252] {
    check_yuv411_equivalence(w, ColorMatrix::Bt601, false);
  }
}

#[cfg(feature = "yuv-planar")]
#[test]
#[cfg_attr(miri, ignore = "AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_yuv411_matches_scalar_width_1920() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  check_yuv411_equivalence(1920, ColorMatrix::Bt709, false);
}

#[cfg(feature = "yuv-planar")]
#[test]
#[cfg_attr(miri, ignore = "AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_yuv411_matches_scalar_non_4_aligned_widths() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  // FFmpeg `AV_PIX_FMT_YUV411P` accepts any width via
  // `chroma_width = width.div_ceil(4)`. Widths < 32 stay entirely in the
  // scalar tail; larger non-4-aligned widths exercise the AVX2 32-pixel
  // SIMD body + partial-chroma scalar tail boundary.
  for w in [1usize, 2, 3, 5, 6, 7, 17, 31, 33, 47, 641] {
    check_yuv411_equivalence(w, ColorMatrix::Bt601, true);
    check_yuv411_equivalence(w, ColorMatrix::Bt709, false);
  }
}

#[cfg(feature = "yuv-planar")]
#[test]
#[cfg_attr(miri, ignore = "AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_yuv411_rgba_matches_scalar_non_4_aligned_widths() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [1usize, 2, 3, 5, 6, 7, 17, 641] {
    check_yuv411_rgba_equivalence(w, ColorMatrix::Bt601, true);
  }
}

// Direct regression for the AVX2 chroma fan-out lane bug: at width=32
// the kernel reads 8 chroma samples (c0..c7) and must fan c0..c3 across
// pixels 0..15 and c4..c7 across pixels 16..31. A previous version
// broadcast c0..c7 into both 128-bit lanes and relied on per-lane
// `_mm256_unpacklo_epi16`, which only consumes lanes 0..3 of each lane
// and therefore expanded c0..c3 in BOTH halves (pixels 16..31 silently
// reused the first half's chroma).
//
// Pick a fixture where c0..c3 are far from c4..c7 so any reuse is a
// large RGB delta. The Y plane is constant so any divergence comes
// from the chroma fan-out alone.
#[cfg(feature = "yuv-planar")]
#[test]
#[cfg_attr(miri, ignore = "AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_yuv411_chroma_fanout_low_high_split() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  // Width 32 = exactly one AVX2 iteration, no scalar tail.
  let width = 32usize;
  let y: std::vec::Vec<u8> = std::vec![128u8; width];
  // c0..c3 (low half, pixels 0..15) vs c4..c7 (high half, pixels 16..31)
  // chosen far apart in the chroma plane.
  let u: std::vec::Vec<u8> = std::vec![10, 20, 30, 40, 200, 210, 220, 230];
  let v: std::vec::Vec<u8> = std::vec![230, 220, 210, 200, 40, 30, 20, 10];

  for &m in &[
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      let mut rgb_scalar = std::vec![0u8; width * 3];
      let mut rgb_simd = std::vec![0u8; width * 3];
      scalar::yuv_411_to_rgb_row(&y, &u, &v, &mut rgb_scalar, width, m, full);
      unsafe {
        yuv_411_to_rgb_row(&y, &u, &v, &mut rgb_simd, width, m, full);
      }
      assert_eq!(
        rgb_scalar, rgb_simd,
        "AVX2 yuv_411 RGB chroma fan-out diverges (matrix={m:?}, full_range={full})"
      );

      let mut rgba_scalar = std::vec![0u8; width * 4];
      let mut rgba_simd = std::vec![0u8; width * 4];
      scalar::yuv_411_to_rgba_row(&y, &u, &v, &mut rgba_scalar, width, m, full);
      unsafe {
        yuv_411_to_rgba_row(&y, &u, &v, &mut rgba_simd, width, m, full);
      }
      assert_eq!(
        rgba_scalar, rgba_simd,
        "AVX2 yuv_411 RGBA chroma fan-out diverges (matrix={m:?}, full_range={full})"
      );
    }
  }
}

#[cfg(feature = "yuv-planar")]
fn check_yuv411_rgba_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  assert!(width > 0);
  let chroma_w = width.div_ceil(4);
  let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
  let u: std::vec::Vec<u8> = (0..chroma_w)
    .map(|i| ((i * 53 + 23) & 0xFF) as u8)
    .collect();
  let v: std::vec::Vec<u8> = (0..chroma_w)
    .map(|i| ((i * 71 + 91) & 0xFF) as u8)
    .collect();
  let mut rgba_scalar = std::vec![0u8; width * 4];
  let mut rgba_simd = std::vec![0u8; width * 4];

  scalar::yuv_411_to_rgba_row(&y, &u, &v, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    yuv_411_to_rgba_row(&y, &u, &v, &mut rgba_simd, width, matrix, full_range);
  }

  assert_eq!(
    rgba_scalar, rgba_simd,
    "AVX2 yuv_411 RGBA diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[cfg(feature = "yuv-planar")]
#[test]
#[cfg_attr(miri, ignore = "AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_yuv411_rgba_matches_scalar_widths() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for &m in &[
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      for &w in &[32usize, 64, 128, 256, 1920] {
        check_yuv411_rgba_equivalence(w, m, full);
      }
    }
  }
}
