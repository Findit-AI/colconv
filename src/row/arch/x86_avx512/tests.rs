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
  let mut rgb_avx512 = std::vec![0u8; width * 3];

  scalar::yuv_420_to_rgb_row(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    yuv_420_to_rgb_row(&y, &u, &v, &mut rgb_avx512, width, matrix, full_range);
  }

  if rgb_scalar != rgb_avx512 {
    let first_diff = rgb_scalar
      .iter()
      .zip(rgb_avx512.iter())
      .position(|(a, b)| a != b)
      .unwrap();
    panic!(
      "AVX‑512 diverges from scalar at byte {first_diff} (width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} avx512={}",
      rgb_scalar[first_diff], rgb_avx512[first_diff]
    );
  }
}

#[test]
fn avx512_matches_scalar_all_matrices_64() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
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
      check_equivalence(64, m, full);
    }
  }
}

#[test]
fn avx512_matches_scalar_width_128() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  check_equivalence(128, ColorMatrix::Bt601, true);
  check_equivalence(128, ColorMatrix::Bt709, false);
  check_equivalence(128, ColorMatrix::YCgCo, true);
}

#[test]
fn avx512_matches_scalar_width_1920() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  check_equivalence(1920, ColorMatrix::Bt709, false);
}

#[test]
fn avx512_matches_scalar_odd_tail_widths() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  // Widths that leave a non‑trivial scalar tail (non‑multiple of 64).
  for w in [66usize, 94, 126, 1922] {
    check_equivalence(w, ColorMatrix::Bt601, false);
  }
}

// ---- yuv_420_to_rgba_row equivalence --------------------------------
//
// Direct backend test for the new RGBA path: bypasses the public
// dispatcher so the AVX‑512 `write_rgba_64` path (four quarters
// through `write_rgba_16`) is exercised regardless of what tier
// the dispatcher would pick on the current runner.

fn check_rgba_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
  let u: std::vec::Vec<u8> = (0..width / 2)
    .map(|i| ((i * 53 + 23) & 0xFF) as u8)
    .collect();
  let v: std::vec::Vec<u8> = (0..width / 2)
    .map(|i| ((i * 71 + 91) & 0xFF) as u8)
    .collect();
  let mut rgba_scalar = std::vec![0u8; width * 4];
  let mut rgba_avx512 = std::vec![0u8; width * 4];

  scalar::yuv_420_to_rgba_row(&y, &u, &v, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    yuv_420_to_rgba_row(&y, &u, &v, &mut rgba_avx512, width, matrix, full_range);
  }

  if rgba_scalar != rgba_avx512 {
    let first_diff = rgba_scalar
      .iter()
      .zip(rgba_avx512.iter())
      .position(|(a, b)| a != b)
      .unwrap();
    let pixel = first_diff / 4;
    let channel = ["R", "G", "B", "A"][first_diff % 4];
    panic!(
      "AVX‑512 RGBA diverges from scalar at byte {first_diff} (px {pixel} {channel}, width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} avx512={}",
      rgba_scalar[first_diff], rgba_avx512[first_diff]
    );
  }
}

#[test]
fn avx512_rgba_matches_scalar_all_matrices_64() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
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
      check_rgba_equivalence(64, m, full);
    }
  }
}

#[test]
fn avx512_rgba_matches_scalar_width_128() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  check_rgba_equivalence(128, ColorMatrix::Bt601, true);
  check_rgba_equivalence(128, ColorMatrix::Bt709, false);
  check_rgba_equivalence(128, ColorMatrix::YCgCo, true);
}

#[test]
fn avx512_rgba_matches_scalar_width_1920() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  check_rgba_equivalence(1920, ColorMatrix::Bt709, false);
}

#[test]
fn avx512_rgba_matches_scalar_odd_tail_widths() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  // Widths that leave a non‑trivial scalar tail (non‑multiple of 64).
  for w in [66usize, 94, 126, 1922] {
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
  let mut rgb_avx512 = std::vec![0u8; width * 3];

  scalar::nv12_to_rgb_row(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    nv12_to_rgb_row(&y, &uv, &mut rgb_avx512, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_avx512,
    "AVX‑512 NV12 ≠ scalar (width={width}, matrix={matrix:?})"
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
    "AVX‑512 NV12 ≠ YUV420P for equivalent UV"
  );
}

#[test]
fn avx512_nv12_matches_scalar_all_matrices_64() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
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
      check_nv12_equivalence(64, m, full);
    }
  }
}

#[test]
fn avx512_nv12_matches_scalar_widths() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  for w in [128usize, 1920, 66, 94, 126, 1922] {
    check_nv12_equivalence(w, ColorMatrix::Bt709, false);
  }
}

#[test]
fn avx512_nv12_matches_yuv420p() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  for w in [64usize, 126, 256, 1920] {
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
  let mut rgb_avx512 = std::vec![0u8; width * 3];

  scalar::nv24_to_rgb_row(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    nv24_to_rgb_row(&y, &uv, &mut rgb_avx512, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_avx512,
    "AVX-512 NV24 ≠ scalar (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_nv42_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
  let vu: std::vec::Vec<u8> = (0..width)
    .flat_map(|i| [((i * 53 + 23) & 0xFF) as u8, ((i * 71 + 91) & 0xFF) as u8])
    .collect();
  let mut rgb_scalar = std::vec![0u8; width * 3];
  let mut rgb_avx512 = std::vec![0u8; width * 3];

  scalar::nv42_to_rgb_row(&y, &vu, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    nv42_to_rgb_row(&y, &vu, &mut rgb_avx512, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_avx512,
    "AVX-512 NV42 ≠ scalar (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[test]
fn avx512_nv24_matches_scalar_all_matrices_64() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
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
      check_nv24_equivalence(64, m, full);
    }
  }
}

#[test]
fn avx512_nv24_matches_scalar_widths() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  // 64 / 128 → main loop; 65 / 129 → main + 1-px tail; 63 → pure
  // scalar tail (< block size); 127 → main + 63-px tail; 1920 →
  // wide real-world baseline.
  for w in [63usize, 64, 65, 127, 128, 129, 1920, 1921] {
    check_nv24_equivalence(w, ColorMatrix::Bt709, false);
  }
}

#[test]
fn avx512_nv42_matches_scalar_all_matrices_64() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
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
      check_nv42_equivalence(64, m, full);
    }
  }
}

#[test]
fn avx512_nv42_matches_scalar_widths() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  for w in [63usize, 64, 65, 127, 128, 129, 1920, 1921] {
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
  let mut rgba_avx512 = std::vec![0u8; width * 4];

  scalar::nv24_to_rgba_row(&y, &uv, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    nv24_to_rgba_row(&y, &uv, &mut rgba_avx512, width, matrix, full_range);
  }

  if rgba_scalar != rgba_avx512 {
    let first_diff = rgba_scalar
      .iter()
      .zip(rgba_avx512.iter())
      .position(|(a, b)| a != b)
      .unwrap();
    let pixel = first_diff / 4;
    let channel = ["R", "G", "B", "A"][first_diff % 4];
    panic!(
      "AVX-512 NV24 RGBA diverges from scalar at byte {first_diff} (px {pixel} {channel}, width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} avx512={}",
      rgba_scalar[first_diff], rgba_avx512[first_diff]
    );
  }
}

fn check_nv42_rgba_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
  let vu: std::vec::Vec<u8> = (0..width)
    .flat_map(|i| [((i * 53 + 23) & 0xFF) as u8, ((i * 71 + 91) & 0xFF) as u8])
    .collect();
  let mut rgba_scalar = std::vec![0u8; width * 4];
  let mut rgba_avx512 = std::vec![0u8; width * 4];

  scalar::nv42_to_rgba_row(&y, &vu, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    nv42_to_rgba_row(&y, &vu, &mut rgba_avx512, width, matrix, full_range);
  }

  if rgba_scalar != rgba_avx512 {
    let first_diff = rgba_scalar
      .iter()
      .zip(rgba_avx512.iter())
      .position(|(a, b)| a != b)
      .unwrap();
    let pixel = first_diff / 4;
    let channel = ["R", "G", "B", "A"][first_diff % 4];
    panic!(
      "AVX-512 NV42 RGBA diverges from scalar at byte {first_diff} (px {pixel} {channel}, width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} avx512={}",
      rgba_scalar[first_diff], rgba_avx512[first_diff]
    );
  }
}

#[test]
fn avx512_nv24_rgba_matches_scalar_all_matrices_64() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
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
      check_nv24_rgba_equivalence(64, m, full);
    }
  }
}

#[test]
fn avx512_nv24_rgba_matches_scalar_widths() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  for w in [63usize, 64, 65, 127, 128, 129, 1920, 1921] {
    check_nv24_rgba_equivalence(w, ColorMatrix::Bt709, false);
  }
}

#[test]
fn avx512_nv42_rgba_matches_scalar_all_matrices_64() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
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
      check_nv42_rgba_equivalence(64, m, full);
    }
  }
}

#[test]
fn avx512_nv42_rgba_matches_scalar_widths() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  for w in [63usize, 64, 65, 127, 128, 129, 1920, 1921] {
    check_nv42_rgba_equivalence(w, ColorMatrix::Bt709, false);
  }
}

// ---- yuv_444_to_rgb_row equivalence ---------------------------------

fn check_yuv_444_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
  let u: std::vec::Vec<u8> = (0..width).map(|i| ((i * 53 + 23) & 0xFF) as u8).collect();
  let v: std::vec::Vec<u8> = (0..width).map(|i| ((i * 71 + 91) & 0xFF) as u8).collect();
  let mut rgb_scalar = std::vec![0u8; width * 3];
  let mut rgb_avx512 = std::vec![0u8; width * 3];

  scalar::yuv_444_to_rgb_row(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    yuv_444_to_rgb_row(&y, &u, &v, &mut rgb_avx512, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_avx512,
    "AVX-512 yuv_444 ≠ scalar (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[test]
fn avx512_yuv_444_matches_scalar_all_matrices_64() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
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
      check_yuv_444_equivalence(64, m, full);
    }
  }
}

#[test]
fn avx512_yuv_444_matches_scalar_widths() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  // Widths straddling the 64-pixel AVX-512 block, AVX2's 32-pixel
  // block, and SSE4.1's 16-pixel tail fallback. Odd widths validate
  // the 4:4:4 no-parity contract.
  for w in [63usize, 64, 65, 127, 128, 129, 1920, 1921] {
    check_yuv_444_equivalence(w, ColorMatrix::Bt709, false);
  }
}

// ---- yuv_444_to_rgba_row equivalence --------------------------------

fn check_yuv_444_rgba_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
  let u: std::vec::Vec<u8> = (0..width).map(|i| ((i * 53 + 23) & 0xFF) as u8).collect();
  let v: std::vec::Vec<u8> = (0..width).map(|i| ((i * 71 + 91) & 0xFF) as u8).collect();
  let mut rgba_scalar = std::vec![0u8; width * 4];
  let mut rgba_avx512 = std::vec![0u8; width * 4];

  scalar::yuv_444_to_rgba_row(&y, &u, &v, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    yuv_444_to_rgba_row(&y, &u, &v, &mut rgba_avx512, width, matrix, full_range);
  }

  if rgba_scalar != rgba_avx512 {
    let first_diff = rgba_scalar
      .iter()
      .zip(rgba_avx512.iter())
      .position(|(a, b)| a != b)
      .unwrap();
    let pixel = first_diff / 4;
    let channel = ["R", "G", "B", "A"][first_diff % 4];
    panic!(
      "AVX-512 yuv_444 RGBA diverges from scalar at byte {first_diff} (px {pixel} {channel}, width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} avx512={}",
      rgba_scalar[first_diff], rgba_avx512[first_diff]
    );
  }
}

#[test]
fn avx512_yuv_444_rgba_matches_scalar_all_matrices_64() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
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
      check_yuv_444_rgba_equivalence(64, m, full);
    }
  }
}

#[test]
fn avx512_yuv_444_rgba_matches_scalar_widths() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  for w in [63usize, 64, 65, 127, 128, 129, 1920, 1921] {
    check_yuv_444_rgba_equivalence(w, ColorMatrix::Bt709, false);
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
  let mut rgb_avx512 = std::vec![0u8; width * 3];
  let mut u16_scalar = std::vec![0u16; width * 3];
  let mut u16_avx512 = std::vec![0u16; width * 3];

  scalar::yuv_444p_n_to_rgb_row::<BITS>(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
  scalar::yuv_444p_n_to_rgb_u16_row::<BITS>(&y, &u, &v, &mut u16_scalar, width, matrix, full_range);
  unsafe {
    yuv_444p_n_to_rgb_row::<BITS>(&y, &u, &v, &mut rgb_avx512, width, matrix, full_range);
    yuv_444p_n_to_rgb_u16_row::<BITS>(&y, &u, &v, &mut u16_avx512, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_avx512,
    "AVX-512 yuv_444p_n<{BITS}> u8 ≠ scalar"
  );
  assert_eq!(
    u16_scalar, u16_avx512,
    "AVX-512 yuv_444p_n<{BITS}> u16 ≠ scalar"
  );
}

#[test]
fn avx512_yuv_444p9_matches_scalar_all_matrices() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  for m in [ColorMatrix::Bt709, ColorMatrix::Bt2020Ncl] {
    for full in [true, false] {
      check_yuv_444p_n_equivalence::<9>(64, m, full);
    }
  }
}

#[test]
fn avx512_yuv_444p10_matches_scalar_all_matrices() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
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
      check_yuv_444p_n_equivalence::<10>(64, m, full);
    }
  }
}

#[test]
fn avx512_yuv_444p12_matches_scalar_all_matrices() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  for m in [ColorMatrix::Bt709, ColorMatrix::Bt2020Ncl] {
    for full in [true, false] {
      check_yuv_444p_n_equivalence::<12>(64, m, full);
    }
  }
}

#[test]
fn avx512_yuv_444p14_matches_scalar_all_matrices() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  for m in [ColorMatrix::Bt709, ColorMatrix::Bt2020Ncl] {
    for full in [true, false] {
      check_yuv_444p_n_equivalence::<14>(64, m, full);
    }
  }
}

#[test]
fn avx512_yuv_444p_n_matches_scalar_widths() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  for w in [1usize, 3, 31, 32, 33, 63, 64, 65, 127, 128, 129, 1920, 1921] {
    check_yuv_444p_n_equivalence::<10>(w, ColorMatrix::Bt709, false);
  }
}

fn check_yuv_444p16_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y: std::vec::Vec<u16> = (0..width).map(|i| (i * 2027 + 11) as u16).collect();
  let u: std::vec::Vec<u16> = (0..width).map(|i| (i * 2671 + 23) as u16).collect();
  let v: std::vec::Vec<u16> = (0..width).map(|i| (i * 3329 + 91) as u16).collect();
  let mut rgb_scalar = std::vec![0u8; width * 3];
  let mut rgb_avx512 = std::vec![0u8; width * 3];
  let mut u16_scalar = std::vec![0u16; width * 3];
  let mut u16_avx512 = std::vec![0u16; width * 3];

  scalar::yuv_444p16_to_rgb_row(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
  scalar::yuv_444p16_to_rgb_u16_row(&y, &u, &v, &mut u16_scalar, width, matrix, full_range);
  unsafe {
    yuv_444p16_to_rgb_row(&y, &u, &v, &mut rgb_avx512, width, matrix, full_range);
    yuv_444p16_to_rgb_u16_row(&y, &u, &v, &mut u16_avx512, width, matrix, full_range);
  }
  assert_eq!(rgb_scalar, rgb_avx512, "AVX-512 yuv_444p16 u8 ≠ scalar");
  assert_eq!(u16_scalar, u16_avx512, "AVX-512 yuv_444p16 u16 ≠ scalar");
}

#[test]
fn avx512_yuv_444p16_matches_scalar_all_matrices() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
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
      check_yuv_444p16_equivalence(64, m, full);
    }
  }
}

#[test]
fn avx512_yuv_444p16_matches_scalar_widths() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  // The u16 kernel is 32-pixel per iter; the u8 kernel is 64.
  for w in [
    1usize, 15, 31, 32, 33, 63, 64, 65, 127, 128, 129, 1920, 1921,
  ] {
    check_yuv_444p16_equivalence(w, ColorMatrix::Bt709, false);
  }
}

// ---- bgr_rgb_swap_row equivalence -----------------------------------

fn check_swap_equivalence(width: usize) {
  let input: std::vec::Vec<u8> = (0..width * 3)
    .map(|i| ((i * 17 + 41) & 0xFF) as u8)
    .collect();
  let mut out_scalar = std::vec![0u8; width * 3];
  let mut out_avx512 = std::vec![0u8; width * 3];

  scalar::bgr_rgb_swap_row(&input, &mut out_scalar, width);
  unsafe {
    bgr_rgb_swap_row(&input, &mut out_avx512, width);
  }
  assert_eq!(out_scalar, out_avx512, "AVX‑512 swap diverges from scalar");
}

#[test]
fn avx512_swap_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  for w in [1usize, 31, 63, 64, 65, 95, 127, 128, 1920, 1921] {
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
  let mut rgb_avx512 = std::vec![0u8; width * 3];

  scalar::nv21_to_rgb_row(&y, &vu, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    nv21_to_rgb_row(&y, &vu, &mut rgb_avx512, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_avx512,
    "AVX-512 NV21 ≠ scalar (width={width}, matrix={matrix:?})"
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
    "AVX-512 NV21 ≠ NV12 with byte-swapped chroma"
  );
}

#[test]
fn nv21_avx512_matches_scalar_all_matrices_16() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
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
fn nv21_avx512_matches_scalar_widths() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  for w in [32usize, 1920, 18, 30, 34, 1922] {
    check_nv21_equivalence(w, ColorMatrix::Bt709, false);
  }
}

#[test]
fn nv21_avx512_matches_nv12_swapped() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
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
  let mut rgba_avx512 = std::vec![0u8; width * 4];

  scalar::nv12_to_rgba_row(&y, &uv, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    nv12_to_rgba_row(&y, &uv, &mut rgba_avx512, width, matrix, full_range);
  }

  if rgba_scalar != rgba_avx512 {
    let first_diff = rgba_scalar
      .iter()
      .zip(rgba_avx512.iter())
      .position(|(a, b)| a != b)
      .unwrap();
    let pixel = first_diff / 4;
    let channel = ["R", "G", "B", "A"][first_diff % 4];
    panic!(
      "AVX-512 NV12 RGBA diverges from scalar at byte {first_diff} (px {pixel} {channel}, width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} avx512={}",
      rgba_scalar[first_diff], rgba_avx512[first_diff]
    );
  }
}

fn check_nv21_rgba_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
  let vu: std::vec::Vec<u8> = (0..width / 2)
    .flat_map(|i| [((i * 53 + 23) & 0xFF) as u8, ((i * 71 + 91) & 0xFF) as u8])
    .collect();
  let mut rgba_scalar = std::vec![0u8; width * 4];
  let mut rgba_avx512 = std::vec![0u8; width * 4];

  scalar::nv21_to_rgba_row(&y, &vu, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    nv21_to_rgba_row(&y, &vu, &mut rgba_avx512, width, matrix, full_range);
  }

  if rgba_scalar != rgba_avx512 {
    let first_diff = rgba_scalar
      .iter()
      .zip(rgba_avx512.iter())
      .position(|(a, b)| a != b)
      .unwrap();
    let pixel = first_diff / 4;
    let channel = ["R", "G", "B", "A"][first_diff % 4];
    panic!(
      "AVX-512 NV21 RGBA diverges from scalar at byte {first_diff} (px {pixel} {channel}, width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} avx512={}",
      rgba_scalar[first_diff], rgba_avx512[first_diff]
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
    "AVX-512 NV12 RGBA must match Yuv420p RGBA for equivalent UV (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[test]
fn nv12_avx512_rgba_matches_scalar_all_matrices_64() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
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
      check_nv12_rgba_equivalence(64, m, full);
    }
  }
}

#[test]
fn nv12_avx512_rgba_matches_scalar_widths() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  for w in [66usize, 94, 126, 1920, 1922] {
    check_nv12_rgba_equivalence(w, ColorMatrix::Bt601, false);
  }
}

#[test]
fn nv12_avx512_rgba_matches_yuv420p_rgba_avx512() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  for w in [64usize, 128, 1920] {
    check_nv12_rgba_matches_yuv420p_rgba(w, ColorMatrix::Bt709, false);
    check_nv12_rgba_matches_yuv420p_rgba(w, ColorMatrix::YCgCo, true);
  }
}

#[test]
fn nv21_avx512_rgba_matches_scalar_all_matrices_64() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
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
      check_nv21_rgba_equivalence(64, m, full);
    }
  }
}

#[test]
fn nv21_avx512_rgba_matches_scalar_widths() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  for w in [66usize, 94, 126, 1920, 1922] {
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
fn avx512_hsv_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  let rgb: std::vec::Vec<u8> = (0..1921 * 3)
    .map(|i| ((i * 37 + 11) & 0xFF) as u8)
    .collect();
  for w in [1usize, 63, 64, 65, 127, 128, 1920, 1921] {
    check_hsv_equivalence(&rgb[..w * 3], w);
  }
}

// ---- yuv420p10 AVX-512 scalar-equivalence ---------------------------

fn p10_plane(n: usize, seed: usize) -> std::vec::Vec<u16> {
  (0..n)
    .map(|i| ((i * seed + seed * 3) & 0x3FF) as u16)
    .collect()
}

fn check_p10_u8_avx512_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
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
      "AVX-512 10→u8 diverges at byte {first_diff} (width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} simd={}",
      rgb_scalar[first_diff], rgb_simd[first_diff]
    );
  }
}

fn check_p10_u16_avx512_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
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
      "AVX-512 10→u16 diverges at elem {first_diff} (width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} simd={}",
      rgb_scalar[first_diff], rgb_simd[first_diff]
    );
  }
}

#[test]
fn avx512_p10_u8_matches_scalar_all_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_p10_u8_avx512_equivalence(64, m, full);
    }
  }
}

#[test]
fn avx512_p10_u16_matches_scalar_all_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_p10_u16_avx512_equivalence(64, m, full);
    }
  }
}

#[test]
fn avx512_p10_matches_scalar_odd_tail_widths() {
  for w in [66usize, 126, 130, 1922] {
    check_p10_u8_avx512_equivalence(w, ColorMatrix::Bt601, false);
    check_p10_u16_avx512_equivalence(w, ColorMatrix::Bt709, true);
  }
}

#[test]
fn avx512_p10_matches_scalar_1920() {
  check_p10_u8_avx512_equivalence(1920, ColorMatrix::Bt709, false);
  check_p10_u16_avx512_equivalence(1920, ColorMatrix::Bt2020Ncl, false);
}

// ---- yuv420p_n<BITS> AVX-512 scalar-equivalence (BITS=9 coverage) ---

fn p_n_plane_avx512<const BITS: u32>(n: usize, seed: usize) -> std::vec::Vec<u16> {
  let mask = ((1u32 << BITS) - 1) as u16;
  (0..n)
    .map(|i| ((i.wrapping_mul(seed).wrapping_add(seed * 3)) as u16) & mask)
    .collect()
}

fn check_p_n_u8_avx512_equivalence<const BITS: u32>(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  let y = p_n_plane_avx512::<BITS>(width, 37);
  let u = p_n_plane_avx512::<BITS>(width / 2, 53);
  let v = p_n_plane_avx512::<BITS>(width / 2, 71);
  let mut rgb_scalar = std::vec![0u8; width * 3];
  let mut rgb_simd = std::vec![0u8; width * 3];
  scalar::yuv_420p_n_to_rgb_row::<BITS>(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    yuv_420p_n_to_rgb_row::<BITS>(&y, &u, &v, &mut rgb_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_simd,
    "AVX-512 yuv_420p_n<{BITS}>→u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_p_n_u16_avx512_equivalence<const BITS: u32>(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  let y = p_n_plane_avx512::<BITS>(width, 37);
  let u = p_n_plane_avx512::<BITS>(width / 2, 53);
  let v = p_n_plane_avx512::<BITS>(width / 2, 71);
  let mut rgb_scalar = std::vec![0u16; width * 3];
  let mut rgb_simd = std::vec![0u16; width * 3];
  scalar::yuv_420p_n_to_rgb_u16_row::<BITS>(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    yuv_420p_n_to_rgb_u16_row::<BITS>(&y, &u, &v, &mut rgb_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_simd,
    "AVX-512 yuv_420p_n<{BITS}>→u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[test]
fn avx512_yuv420p9_matches_scalar_all_matrices_and_ranges() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_p_n_u8_avx512_equivalence::<9>(64, m, full);
      check_p_n_u16_avx512_equivalence::<9>(64, m, full);
    }
  }
}

#[test]
fn avx512_yuv420p9_matches_scalar_tail_and_large_widths() {
  // AVX-512 main loop is 64 px; widths chosen to stress tail handling
  // both below and above the SIMD lane width.
  for w in [18usize, 30, 34, 62, 66, 126, 130, 1922] {
    check_p_n_u8_avx512_equivalence::<9>(w, ColorMatrix::Bt601, false);
    check_p_n_u16_avx512_equivalence::<9>(w, ColorMatrix::Bt709, true);
  }
  check_p_n_u8_avx512_equivalence::<9>(1920, ColorMatrix::Bt709, false);
  check_p_n_u16_avx512_equivalence::<9>(1920, ColorMatrix::Bt2020Ncl, false);
}

// ---- P010 AVX-512 scalar-equivalence --------------------------------

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

fn check_p010_u8_avx512_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
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
  assert_eq!(rgb_scalar, rgb_simd, "AVX-512 P010→u8 diverges");
}

fn check_p010_u16_avx512_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
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
  assert_eq!(rgb_scalar, rgb_simd, "AVX-512 P010→u16 diverges");
}

#[test]
fn avx512_p010_u8_matches_scalar_all_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_p010_u8_avx512_equivalence(64, m, full);
    }
  }
}

#[test]
fn avx512_p010_u16_matches_scalar_all_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_p010_u16_avx512_equivalence(64, m, full);
    }
  }
}

#[test]
fn avx512_p010_matches_scalar_odd_tail_widths() {
  for w in [66usize, 126, 130, 1922] {
    check_p010_u8_avx512_equivalence(w, ColorMatrix::Bt601, false);
    check_p010_u16_avx512_equivalence(w, ColorMatrix::Bt709, true);
  }
}

#[test]
fn avx512_p010_matches_scalar_1920() {
  check_p010_u8_avx512_equivalence(1920, ColorMatrix::Bt709, false);
  check_p010_u16_avx512_equivalence(1920, ColorMatrix::Bt2020Ncl, false);
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

fn check_planar_u8_avx512_equivalence_n<const BITS: u32>(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
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
  assert_eq!(
    rgb_scalar, rgb_simd,
    "AVX-512 planar {BITS}-bit → u8 diverges"
  );
}

fn check_planar_u16_avx512_equivalence_n<const BITS: u32>(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
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
    "AVX-512 planar {BITS}-bit → u16 diverges"
  );
}

fn check_pn_u8_avx512_equivalence_n<const BITS: u32>(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
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
  assert_eq!(rgb_scalar, rgb_simd, "AVX-512 Pn {BITS}-bit → u8 diverges");
}

fn check_pn_u16_avx512_equivalence_n<const BITS: u32>(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
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
  assert_eq!(rgb_scalar, rgb_simd, "AVX-512 Pn {BITS}-bit → u16 diverges");
}

#[test]
fn avx512_p12_matches_scalar_all_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_planar_u8_avx512_equivalence_n::<12>(64, m, full);
      check_planar_u16_avx512_equivalence_n::<12>(64, m, full);
      check_pn_u8_avx512_equivalence_n::<12>(64, m, full);
      check_pn_u16_avx512_equivalence_n::<12>(64, m, full);
    }
  }
}

#[test]
fn avx512_p14_matches_scalar_all_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_planar_u8_avx512_equivalence_n::<14>(64, m, full);
      check_planar_u16_avx512_equivalence_n::<14>(64, m, full);
    }
  }
}

#[test]
fn avx512_p12_matches_scalar_tail_widths() {
  for w in [66usize, 126, 130, 1922] {
    check_planar_u8_avx512_equivalence_n::<12>(w, ColorMatrix::Bt601, false);
    check_planar_u16_avx512_equivalence_n::<12>(w, ColorMatrix::Bt709, true);
    check_pn_u8_avx512_equivalence_n::<12>(w, ColorMatrix::Bt601, false);
    check_pn_u16_avx512_equivalence_n::<12>(w, ColorMatrix::Bt2020Ncl, false);
  }
}

#[test]
fn avx512_p14_matches_scalar_tail_widths() {
  for w in [66usize, 126, 130, 1922] {
    check_planar_u8_avx512_equivalence_n::<14>(w, ColorMatrix::Bt601, false);
    check_planar_u16_avx512_equivalence_n::<14>(w, ColorMatrix::Bt709, true);
  }
}

// ---- 16-bit (full-range u16 samples) AVX-512 equivalence ------------

fn p16_plane_avx512(n: usize, seed: usize) -> std::vec::Vec<u16> {
  (0..n)
    .map(|i| ((i.wrapping_mul(seed).wrapping_add(seed * 3)) & 0xFFFF) as u16)
    .collect()
}

fn check_yuv420p16_u8_avx512_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  let y = p16_plane_avx512(width, 37);
  let u = p16_plane_avx512(width / 2, 53);
  let v = p16_plane_avx512(width / 2, 71);
  let mut rgb_scalar = std::vec![0u8; width * 3];
  let mut rgb_simd = std::vec![0u8; width * 3];
  scalar::yuv_420p16_to_rgb_row(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    yuv_420p16_to_rgb_row(&y, &u, &v, &mut rgb_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_simd,
    "AVX-512 yuv420p16→u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_yuv420p16_u16_avx512_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  let y = p16_plane_avx512(width, 37);
  let u = p16_plane_avx512(width / 2, 53);
  let v = p16_plane_avx512(width / 2, 71);
  let mut rgb_scalar = std::vec![0u16; width * 3];
  let mut rgb_simd = std::vec![0u16; width * 3];
  scalar::yuv_420p16_to_rgb_u16_row(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    yuv_420p16_to_rgb_u16_row(&y, &u, &v, &mut rgb_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_simd,
    "AVX-512 yuv420p16→u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_p16_u8_avx512_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  let y = p16_plane_avx512(width, 37);
  let u = p16_plane_avx512(width / 2, 53);
  let v = p16_plane_avx512(width / 2, 71);
  let uv = p010_uv_interleave(&u, &v);
  let mut rgb_scalar = std::vec![0u8; width * 3];
  let mut rgb_simd = std::vec![0u8; width * 3];
  scalar::p16_to_rgb_row(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    p16_to_rgb_row(&y, &uv, &mut rgb_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_simd,
    "AVX-512 p016→u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_p16_u16_avx512_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  let y = p16_plane_avx512(width, 37);
  let u = p16_plane_avx512(width / 2, 53);
  let v = p16_plane_avx512(width / 2, 71);
  let uv = p010_uv_interleave(&u, &v);
  let mut rgb_scalar = std::vec![0u16; width * 3];
  let mut rgb_simd = std::vec![0u16; width * 3];
  scalar::p16_to_rgb_u16_row(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    p16_to_rgb_u16_row(&y, &uv, &mut rgb_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_simd,
    "AVX-512 p016→u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[test]
fn avx512_p16_matches_scalar_all_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_yuv420p16_u8_avx512_equivalence(64, m, full);
      check_yuv420p16_u16_avx512_equivalence(64, m, full);
      check_p16_u8_avx512_equivalence(64, m, full);
      check_p16_u16_avx512_equivalence(64, m, full);
    }
  }
}

#[test]
fn avx512_p16_matches_scalar_tail_widths() {
  for w in [66usize, 126, 130, 1922] {
    check_yuv420p16_u8_avx512_equivalence(w, ColorMatrix::Bt601, false);
    check_yuv420p16_u16_avx512_equivalence(w, ColorMatrix::Bt709, true);
    check_p16_u8_avx512_equivalence(w, ColorMatrix::Bt601, false);
    check_p16_u16_avx512_equivalence(w, ColorMatrix::Bt2020Ncl, false);
  }
}

#[test]
fn avx512_p16_matches_scalar_1920() {
  check_yuv420p16_u8_avx512_equivalence(1920, ColorMatrix::Bt709, false);
  check_yuv420p16_u16_avx512_equivalence(1920, ColorMatrix::Bt2020Ncl, false);
  check_p16_u8_avx512_equivalence(1920, ColorMatrix::Bt709, false);
  check_p16_u16_avx512_equivalence(1920, ColorMatrix::Bt2020Ncl, false);
}

// ---- High-bit 4:2:0 RGBA equivalence (Ship 8 Tranche 5a) ----------
//
// RGBA wrappers share the math of their RGB siblings — only the store
// (and tail dispatch) branches on `ALPHA`. These tests pin that the
// SIMD RGBA path produces byte-identical output to the scalar RGBA
// reference.

fn check_planar_u8_avx512_rgba_equivalence_n<const BITS: u32>(
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
    "AVX-512 yuv_420p_n<{BITS}>→RGBA u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_pn_u8_avx512_rgba_equivalence_n<const BITS: u32>(
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
    "AVX-512 Pn<{BITS}>→RGBA u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_yuv420p16_u8_avx512_rgba_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y = p16_plane_avx512(width, 37);
  let u = p16_plane_avx512(width / 2, 53);
  let v = p16_plane_avx512(width / 2, 71);
  let mut rgba_scalar = std::vec![0u8; width * 4];
  let mut rgba_simd = std::vec![0u8; width * 4];
  scalar::yuv_420p16_to_rgba_row(&y, &u, &v, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    yuv_420p16_to_rgba_row(&y, &u, &v, &mut rgba_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgba_scalar, rgba_simd,
    "AVX-512 yuv_420p16→RGBA u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_p16_u8_avx512_rgba_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y = p16_plane_avx512(width, 37);
  let u = p16_plane_avx512(width / 2, 53);
  let v = p16_plane_avx512(width / 2, 71);
  let uv = p010_uv_interleave(&u, &v);
  let mut rgba_scalar = std::vec![0u8; width * 4];
  let mut rgba_simd = std::vec![0u8; width * 4];
  scalar::p16_to_rgba_row(&y, &uv, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    p16_to_rgba_row(&y, &uv, &mut rgba_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgba_scalar, rgba_simd,
    "AVX-512 P016→RGBA u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[test]
fn avx512_yuv420p_n_rgba_matches_scalar_all_bits() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_planar_u8_avx512_rgba_equivalence_n::<9>(64, m, full);
      check_planar_u8_avx512_rgba_equivalence_n::<10>(64, m, full);
      check_planar_u8_avx512_rgba_equivalence_n::<12>(64, m, full);
      check_planar_u8_avx512_rgba_equivalence_n::<14>(64, m, full);
    }
  }
}

#[test]
fn avx512_yuv420p_n_rgba_matches_scalar_tail_and_1920() {
  for w in [66usize, 96, 126, 1920, 1922] {
    check_planar_u8_avx512_rgba_equivalence_n::<9>(w, ColorMatrix::Bt601, false);
    check_planar_u8_avx512_rgba_equivalence_n::<10>(w, ColorMatrix::Bt709, true);
    check_planar_u8_avx512_rgba_equivalence_n::<12>(w, ColorMatrix::Bt2020Ncl, false);
    check_planar_u8_avx512_rgba_equivalence_n::<14>(w, ColorMatrix::YCgCo, true);
  }
}

#[test]
fn avx512_pn_rgba_matches_scalar_all_bits() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_pn_u8_avx512_rgba_equivalence_n::<10>(64, m, full);
      check_pn_u8_avx512_rgba_equivalence_n::<12>(64, m, full);
    }
  }
}

#[test]
fn avx512_pn_rgba_matches_scalar_tail_and_1920() {
  for w in [66usize, 96, 126, 1920, 1922] {
    check_pn_u8_avx512_rgba_equivalence_n::<10>(w, ColorMatrix::Bt601, false);
    check_pn_u8_avx512_rgba_equivalence_n::<12>(w, ColorMatrix::Bt709, true);
  }
}

#[test]
fn avx512_yuv420p16_rgba_matches_scalar_all_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_yuv420p16_u8_avx512_rgba_equivalence(64, m, full);
    }
  }
  for w in [66usize, 96, 126, 1920, 1922] {
    check_yuv420p16_u8_avx512_rgba_equivalence(w, ColorMatrix::Bt709, false);
  }
}

#[test]
fn avx512_p016_rgba_matches_scalar_all_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_p16_u8_avx512_rgba_equivalence(64, m, full);
    }
  }
  for w in [66usize, 96, 126, 1920, 1922] {
    check_p16_u8_avx512_rgba_equivalence(w, ColorMatrix::Bt709, false);
  }
}

// ---- Pn 4:4:4 (P410 / P412 / P416) AVX-512 equivalence -------------

fn high_bit_plane_avx512<const BITS: u32>(n: usize, seed: usize) -> std::vec::Vec<u16> {
  let mask = ((1u32 << BITS) - 1) as u16;
  let shift = 16 - BITS;
  (0..n)
    .map(|i| (((i.wrapping_mul(seed).wrapping_add(seed * 3)) as u16) & mask) << shift)
    .collect()
}

fn interleave_uv_avx512(u_full: &[u16], v_full: &[u16]) -> std::vec::Vec<u16> {
  debug_assert_eq!(u_full.len(), v_full.len());
  let mut out = std::vec::Vec::with_capacity(u_full.len() * 2);
  for i in 0..u_full.len() {
    out.push(u_full[i]);
    out.push(v_full[i]);
  }
  out
}

fn check_p_n_444_u8_avx512_equivalence<const BITS: u32>(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  let y = high_bit_plane_avx512::<BITS>(width, 37);
  let u = high_bit_plane_avx512::<BITS>(width, 53);
  let v = high_bit_plane_avx512::<BITS>(width, 71);
  let uv = interleave_uv_avx512(&u, &v);
  let mut rgb_scalar = std::vec![0u8; width * 3];
  let mut rgb_simd = std::vec![0u8; width * 3];
  scalar::p_n_444_to_rgb_row::<BITS>(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    p_n_444_to_rgb_row::<BITS>(&y, &uv, &mut rgb_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_simd,
    "AVX-512 Pn4:4:4 {BITS}-bit → u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_p_n_444_u16_avx512_equivalence<const BITS: u32>(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  let y = high_bit_plane_avx512::<BITS>(width, 37);
  let u = high_bit_plane_avx512::<BITS>(width, 53);
  let v = high_bit_plane_avx512::<BITS>(width, 71);
  let uv = interleave_uv_avx512(&u, &v);
  let mut rgb_scalar = std::vec![0u16; width * 3];
  let mut rgb_simd = std::vec![0u16; width * 3];
  scalar::p_n_444_to_rgb_u16_row::<BITS>(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    p_n_444_to_rgb_u16_row::<BITS>(&y, &uv, &mut rgb_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_simd,
    "AVX-512 Pn4:4:4 {BITS}-bit → u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_p_n_444_16_u8_avx512_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  let y = p16_plane_avx512(width, 37);
  let u = p16_plane_avx512(width, 53);
  let v = p16_plane_avx512(width, 71);
  let uv = interleave_uv_avx512(&u, &v);
  let mut rgb_scalar = std::vec![0u8; width * 3];
  let mut rgb_simd = std::vec![0u8; width * 3];
  scalar::p_n_444_16_to_rgb_row(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    p_n_444_16_to_rgb_row(&y, &uv, &mut rgb_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_simd,
    "AVX-512 P416 → u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_p_n_444_16_u16_avx512_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  let y = p16_plane_avx512(width, 37);
  let u = p16_plane_avx512(width, 53);
  let v = p16_plane_avx512(width, 71);
  let uv = interleave_uv_avx512(&u, &v);
  let mut rgb_scalar = std::vec![0u16; width * 3];
  let mut rgb_simd = std::vec![0u16; width * 3];
  scalar::p_n_444_16_to_rgb_u16_row(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    p_n_444_16_to_rgb_u16_row(&y, &uv, &mut rgb_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_simd,
    "AVX-512 P416 → u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[test]
fn avx512_p410_matches_scalar_all_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_p_n_444_u8_avx512_equivalence::<10>(64, m, full);
      check_p_n_444_u16_avx512_equivalence::<10>(64, m, full);
    }
  }
}

#[test]
fn avx512_p412_matches_scalar_all_matrices() {
  for m in [ColorMatrix::Bt709, ColorMatrix::Bt2020Ncl] {
    for full in [true, false] {
      check_p_n_444_u8_avx512_equivalence::<12>(64, m, full);
      check_p_n_444_u16_avx512_equivalence::<12>(64, m, full);
    }
  }
}

#[test]
fn avx512_p410_p412_matches_scalar_tail_widths() {
  // AVX-512 main loop is 64 px (or 32 px for the i64 u16 path);
  // tail widths force scalar fallback.
  for w in [1usize, 3, 33, 63, 65, 95, 127, 129, 1920, 1921] {
    check_p_n_444_u8_avx512_equivalence::<10>(w, ColorMatrix::Bt601, false);
    check_p_n_444_u16_avx512_equivalence::<10>(w, ColorMatrix::Bt709, true);
    check_p_n_444_u8_avx512_equivalence::<12>(w, ColorMatrix::Bt2020Ncl, false);
  }
}

#[test]
fn avx512_p416_matches_scalar_all_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_p_n_444_16_u8_avx512_equivalence(64, m, full);
      check_p_n_444_16_u16_avx512_equivalence(64, m, full);
    }
  }
}

#[test]
fn avx512_p416_matches_scalar_tail_widths() {
  for w in [1usize, 3, 31, 33, 63, 65, 95, 127, 129, 1920, 1921] {
    check_p_n_444_16_u8_avx512_equivalence(w, ColorMatrix::Bt709, false);
    check_p_n_444_16_u16_avx512_equivalence(w, ColorMatrix::Bt2020Ncl, true);
  }
}
