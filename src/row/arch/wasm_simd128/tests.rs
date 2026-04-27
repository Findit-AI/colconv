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
  let mut rgb_wasm = std::vec![0u8; width * 3];

  scalar::yuv_420_to_rgb_row(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    yuv_420_to_rgb_row(&y, &u, &v, &mut rgb_wasm, width, matrix, full_range);
  }

  assert_eq!(rgb_scalar, rgb_wasm, "simd128 diverges from scalar");
}

#[test]
fn simd128_matches_scalar_all_matrices_16() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_equivalence(16, m, full);
    }
  }
}

#[test]
fn simd128_matches_scalar_tail_widths() {
  for w in [18usize, 30, 34, 1922] {
    check_equivalence(w, ColorMatrix::Bt601, false);
  }
}

// ---- yuv_420_to_rgba_row equivalence --------------------------------
//
// Direct backend test for the new RGBA path: bypasses the public
// dispatcher so the wasm `write_rgba_16` swizzle (4-mask + 4
// store) is exercised on every wasm32+simd128 target.

fn check_rgba_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
  let u: std::vec::Vec<u8> = (0..width / 2)
    .map(|i| ((i * 53 + 23) & 0xFF) as u8)
    .collect();
  let v: std::vec::Vec<u8> = (0..width / 2)
    .map(|i| ((i * 71 + 91) & 0xFF) as u8)
    .collect();
  let mut rgba_scalar = std::vec![0u8; width * 4];
  let mut rgba_wasm = std::vec![0u8; width * 4];

  scalar::yuv_420_to_rgba_row(&y, &u, &v, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    yuv_420_to_rgba_row(&y, &u, &v, &mut rgba_wasm, width, matrix, full_range);
  }

  if rgba_scalar != rgba_wasm {
    let first_diff = rgba_scalar
      .iter()
      .zip(rgba_wasm.iter())
      .position(|(a, b)| a != b)
      .unwrap();
    let pixel = first_diff / 4;
    let channel = ["R", "G", "B", "A"][first_diff % 4];
    panic!(
      "wasm simd128 RGBA diverges from scalar at byte {first_diff} (px {pixel} {channel}, width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} wasm={}",
      rgba_scalar[first_diff], rgba_wasm[first_diff]
    );
  }
}

#[test]
fn simd128_rgba_matches_scalar_all_matrices_16() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_rgba_equivalence(16, m, full);
    }
  }
}

#[test]
fn simd128_rgba_matches_scalar_tail_widths() {
  for w in [18usize, 30, 34, 1922] {
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
  let mut rgb_wasm = std::vec![0u8; width * 3];

  scalar::nv12_to_rgb_row(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    nv12_to_rgb_row(&y, &uv, &mut rgb_wasm, width, matrix, full_range);
  }
  assert_eq!(rgb_scalar, rgb_wasm, "simd128 NV12 ≠ scalar");
}

#[test]
fn simd128_nv12_matches_scalar_all_matrices_16() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_nv12_equivalence(16, m, full);
    }
  }
}

#[test]
fn simd128_nv12_matches_scalar_widths() {
  for w in [32usize, 1920, 18, 30, 34, 1922] {
    check_nv12_equivalence(w, ColorMatrix::Bt709, false);
  }
}

// ---- nv24_to_rgb_row / nv42_to_rgb_row equivalence ------------------

fn check_nv24_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
  let uv: std::vec::Vec<u8> = (0..width)
    .flat_map(|i| [((i * 53 + 23) & 0xFF) as u8, ((i * 71 + 91) & 0xFF) as u8])
    .collect();
  let mut rgb_scalar = std::vec![0u8; width * 3];
  let mut rgb_wasm = std::vec![0u8; width * 3];

  scalar::nv24_to_rgb_row(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    nv24_to_rgb_row(&y, &uv, &mut rgb_wasm, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_wasm,
    "simd128 NV24 ≠ scalar (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_nv42_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
  let vu: std::vec::Vec<u8> = (0..width)
    .flat_map(|i| [((i * 53 + 23) & 0xFF) as u8, ((i * 71 + 91) & 0xFF) as u8])
    .collect();
  let mut rgb_scalar = std::vec![0u8; width * 3];
  let mut rgb_wasm = std::vec![0u8; width * 3];

  scalar::nv42_to_rgb_row(&y, &vu, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    nv42_to_rgb_row(&y, &vu, &mut rgb_wasm, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_wasm,
    "simd128 NV42 ≠ scalar (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[test]
fn simd128_nv24_matches_scalar_all_matrices_16() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_nv24_equivalence(16, m, full);
    }
  }
}

#[test]
fn simd128_nv24_matches_scalar_widths() {
  for w in [1usize, 3, 15, 17, 32, 33, 1920, 1921] {
    check_nv24_equivalence(w, ColorMatrix::Bt709, false);
  }
}

#[test]
fn simd128_nv42_matches_scalar_all_matrices_16() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_nv42_equivalence(16, m, full);
    }
  }
}

#[test]
fn simd128_nv42_matches_scalar_widths() {
  for w in [1usize, 3, 15, 17, 32, 33, 1920, 1921] {
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
  let mut rgba_simd = std::vec![0u8; width * 4];

  scalar::nv24_to_rgba_row(&y, &uv, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    nv24_to_rgba_row(&y, &uv, &mut rgba_simd, width, matrix, full_range);
  }

  if rgba_scalar != rgba_simd {
    let first_diff = rgba_scalar
      .iter()
      .zip(rgba_simd.iter())
      .position(|(a, b)| a != b)
      .unwrap();
    let pixel = first_diff / 4;
    let channel = ["R", "G", "B", "A"][first_diff % 4];
    panic!(
      "simd128 NV24 RGBA diverges from scalar at byte {first_diff} (px {pixel} {channel}, width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} simd={}",
      rgba_scalar[first_diff], rgba_simd[first_diff]
    );
  }
}

fn check_nv42_rgba_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
  let vu: std::vec::Vec<u8> = (0..width)
    .flat_map(|i| [((i * 53 + 23) & 0xFF) as u8, ((i * 71 + 91) & 0xFF) as u8])
    .collect();
  let mut rgba_scalar = std::vec![0u8; width * 4];
  let mut rgba_simd = std::vec![0u8; width * 4];

  scalar::nv42_to_rgba_row(&y, &vu, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    nv42_to_rgba_row(&y, &vu, &mut rgba_simd, width, matrix, full_range);
  }

  if rgba_scalar != rgba_simd {
    let first_diff = rgba_scalar
      .iter()
      .zip(rgba_simd.iter())
      .position(|(a, b)| a != b)
      .unwrap();
    let pixel = first_diff / 4;
    let channel = ["R", "G", "B", "A"][first_diff % 4];
    panic!(
      "simd128 NV42 RGBA diverges from scalar at byte {first_diff} (px {pixel} {channel}, width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} simd={}",
      rgba_scalar[first_diff], rgba_simd[first_diff]
    );
  }
}

#[test]
fn simd128_nv24_rgba_matches_scalar_all_matrices_16() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_nv24_rgba_equivalence(16, m, full);
    }
  }
}

#[test]
fn simd128_nv24_rgba_matches_scalar_widths() {
  for w in [1usize, 3, 15, 17, 32, 33, 1920, 1921] {
    check_nv24_rgba_equivalence(w, ColorMatrix::Bt709, false);
  }
}

#[test]
fn simd128_nv42_rgba_matches_scalar_all_matrices_16() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_nv42_rgba_equivalence(16, m, full);
    }
  }
}

#[test]
fn simd128_nv42_rgba_matches_scalar_widths() {
  for w in [1usize, 3, 15, 17, 32, 33, 1920, 1921] {
    check_nv42_rgba_equivalence(w, ColorMatrix::Bt709, false);
  }
}

// ---- yuv_444_to_rgb_row equivalence ---------------------------------

fn check_yuv_444_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
  let u: std::vec::Vec<u8> = (0..width).map(|i| ((i * 53 + 23) & 0xFF) as u8).collect();
  let v: std::vec::Vec<u8> = (0..width).map(|i| ((i * 71 + 91) & 0xFF) as u8).collect();
  let mut rgb_scalar = std::vec![0u8; width * 3];
  let mut rgb_wasm = std::vec![0u8; width * 3];

  scalar::yuv_444_to_rgb_row(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    yuv_444_to_rgb_row(&y, &u, &v, &mut rgb_wasm, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_wasm,
    "simd128 yuv_444 ≠ scalar (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[test]
fn simd128_yuv_444_matches_scalar_all_matrices_16() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_yuv_444_equivalence(16, m, full);
    }
  }
}

#[test]
fn simd128_yuv_444_matches_scalar_widths() {
  // Odd widths validate the 4:4:4 no-parity contract.
  for w in [1usize, 3, 15, 17, 32, 33, 1920, 1921] {
    check_yuv_444_equivalence(w, ColorMatrix::Bt709, false);
  }
}

// ---- yuv_444_to_rgba_row equivalence --------------------------------

fn check_yuv_444_rgba_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
  let u: std::vec::Vec<u8> = (0..width).map(|i| ((i * 53 + 23) & 0xFF) as u8).collect();
  let v: std::vec::Vec<u8> = (0..width).map(|i| ((i * 71 + 91) & 0xFF) as u8).collect();
  let mut rgba_scalar = std::vec![0u8; width * 4];
  let mut rgba_wasm = std::vec![0u8; width * 4];

  scalar::yuv_444_to_rgba_row(&y, &u, &v, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    yuv_444_to_rgba_row(&y, &u, &v, &mut rgba_wasm, width, matrix, full_range);
  }

  if rgba_scalar != rgba_wasm {
    let first_diff = rgba_scalar
      .iter()
      .zip(rgba_wasm.iter())
      .position(|(a, b)| a != b)
      .unwrap();
    let pixel = first_diff / 4;
    let channel = ["R", "G", "B", "A"][first_diff % 4];
    panic!(
      "wasm simd128 yuv_444 RGBA diverges from scalar at byte {first_diff} (px {pixel} {channel}, width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} wasm={}",
      rgba_scalar[first_diff], rgba_wasm[first_diff]
    );
  }
}

#[test]
fn simd128_yuv_444_rgba_matches_scalar_all_matrices_16() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_yuv_444_rgba_equivalence(16, m, full);
    }
  }
}

#[test]
fn simd128_yuv_444_rgba_matches_scalar_widths() {
  for w in [1usize, 3, 15, 17, 32, 33, 1920, 1921] {
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
  let mut rgb_wasm = std::vec![0u8; width * 3];
  let mut u16_scalar = std::vec![0u16; width * 3];
  let mut u16_wasm = std::vec![0u16; width * 3];

  scalar::yuv_444p_n_to_rgb_row::<BITS>(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
  scalar::yuv_444p_n_to_rgb_u16_row::<BITS>(&y, &u, &v, &mut u16_scalar, width, matrix, full_range);
  unsafe {
    yuv_444p_n_to_rgb_row::<BITS>(&y, &u, &v, &mut rgb_wasm, width, matrix, full_range);
    yuv_444p_n_to_rgb_u16_row::<BITS>(&y, &u, &v, &mut u16_wasm, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_wasm,
    "simd128 yuv_444p_n<{BITS}> u8 ≠ scalar"
  );
  assert_eq!(
    u16_scalar, u16_wasm,
    "simd128 yuv_444p_n<{BITS}> u16 ≠ scalar"
  );
}

#[test]
fn simd128_yuv_444p9_matches_scalar_all_matrices() {
  for m in [ColorMatrix::Bt709, ColorMatrix::Bt2020Ncl] {
    for full in [true, false] {
      check_yuv_444p_n_equivalence::<9>(16, m, full);
    }
  }
}

#[test]
fn simd128_yuv_444p10_matches_scalar_all_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_yuv_444p_n_equivalence::<10>(16, m, full);
    }
  }
}

#[test]
fn simd128_yuv_444p12_matches_scalar_all_matrices() {
  for m in [ColorMatrix::Bt709, ColorMatrix::Bt2020Ncl] {
    for full in [true, false] {
      check_yuv_444p_n_equivalence::<12>(16, m, full);
    }
  }
}

#[test]
fn simd128_yuv_444p14_matches_scalar_all_matrices() {
  for m in [ColorMatrix::Bt709, ColorMatrix::Bt2020Ncl] {
    for full in [true, false] {
      check_yuv_444p_n_equivalence::<14>(16, m, full);
    }
  }
}

#[test]
fn simd128_yuv_444p_n_matches_scalar_widths() {
  for w in [1usize, 3, 15, 17, 32, 33, 1920, 1921] {
    check_yuv_444p_n_equivalence::<10>(w, ColorMatrix::Bt709, false);
  }
}

fn check_yuv_444p16_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y: std::vec::Vec<u16> = (0..width).map(|i| (i * 2027 + 11) as u16).collect();
  let u: std::vec::Vec<u16> = (0..width).map(|i| (i * 2671 + 23) as u16).collect();
  let v: std::vec::Vec<u16> = (0..width).map(|i| (i * 3329 + 91) as u16).collect();
  let mut rgb_scalar = std::vec![0u8; width * 3];
  let mut rgb_wasm = std::vec![0u8; width * 3];

  scalar::yuv_444p16_to_rgb_row(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    yuv_444p16_to_rgb_row(&y, &u, &v, &mut rgb_wasm, width, matrix, full_range);
  }
  assert_eq!(rgb_scalar, rgb_wasm, "simd128 yuv_444p16 u8 ≠ scalar");
  // u16-output path delegates to scalar on wasm — no SIMD to compare
  // against beyond the direct passthrough.
}

#[test]
fn simd128_yuv_444p16_matches_scalar_all_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_yuv_444p16_equivalence(16, m, full);
    }
  }
}

#[test]
fn simd128_yuv_444p16_matches_scalar_widths() {
  for w in [1usize, 3, 15, 17, 32, 33, 1920, 1921] {
    check_yuv_444p16_equivalence(w, ColorMatrix::Bt709, false);
  }
}

// ---- bgr_rgb_swap_row equivalence -----------------------------------

fn check_swap_equivalence(width: usize) {
  let input: std::vec::Vec<u8> = (0..width * 3)
    .map(|i| ((i * 17 + 41) & 0xFF) as u8)
    .collect();
  let mut out_scalar = std::vec![0u8; width * 3];
  let mut out_wasm = std::vec![0u8; width * 3];

  scalar::bgr_rgb_swap_row(&input, &mut out_scalar, width);
  unsafe {
    bgr_rgb_swap_row(&input, &mut out_wasm, width);
  }
  assert_eq!(out_scalar, out_wasm, "simd128 swap diverges from scalar");
}

#[test]
fn simd128_swap_matches_scalar() {
  for w in [1usize, 15, 16, 17, 31, 32, 1920, 1921] {
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
  let mut rgb_wasm = std::vec![0u8; width * 3];

  scalar::nv21_to_rgb_row(&y, &vu, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    nv21_to_rgb_row(&y, &vu, &mut rgb_wasm, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_wasm,
    "simd128 NV21 ≠ scalar (width={width}, matrix={matrix:?})"
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
    "simd128 NV21 ≠ NV12 with byte-swapped chroma"
  );
}

#[test]
fn nv21_wasm_matches_scalar_all_matrices_16() {
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
fn nv21_wasm_matches_scalar_widths() {
  for w in [32usize, 1920, 18, 30, 34, 1922] {
    check_nv21_equivalence(w, ColorMatrix::Bt709, false);
  }
}

#[test]
fn nv21_wasm_matches_nv12_swapped() {
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
  let mut rgba_wasm = std::vec![0u8; width * 4];

  scalar::nv12_to_rgba_row(&y, &uv, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    nv12_to_rgba_row(&y, &uv, &mut rgba_wasm, width, matrix, full_range);
  }

  if rgba_scalar != rgba_wasm {
    let first_diff = rgba_scalar
      .iter()
      .zip(rgba_wasm.iter())
      .position(|(a, b)| a != b)
      .unwrap();
    let pixel = first_diff / 4;
    let channel = ["R", "G", "B", "A"][first_diff % 4];
    panic!(
      "wasm simd128 NV12 RGBA diverges from scalar at byte {first_diff} (px {pixel} {channel}, width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} wasm={}",
      rgba_scalar[first_diff], rgba_wasm[first_diff]
    );
  }
}

fn check_nv21_rgba_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y: std::vec::Vec<u8> = (0..width).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
  let vu: std::vec::Vec<u8> = (0..width / 2)
    .flat_map(|i| [((i * 53 + 23) & 0xFF) as u8, ((i * 71 + 91) & 0xFF) as u8])
    .collect();
  let mut rgba_scalar = std::vec![0u8; width * 4];
  let mut rgba_wasm = std::vec![0u8; width * 4];

  scalar::nv21_to_rgba_row(&y, &vu, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    nv21_to_rgba_row(&y, &vu, &mut rgba_wasm, width, matrix, full_range);
  }

  if rgba_scalar != rgba_wasm {
    let first_diff = rgba_scalar
      .iter()
      .zip(rgba_wasm.iter())
      .position(|(a, b)| a != b)
      .unwrap();
    let pixel = first_diff / 4;
    let channel = ["R", "G", "B", "A"][first_diff % 4];
    panic!(
      "wasm simd128 NV21 RGBA diverges from scalar at byte {first_diff} (px {pixel} {channel}, width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} wasm={}",
      rgba_scalar[first_diff], rgba_wasm[first_diff]
    );
  }
}

#[test]
fn nv12_wasm_rgba_matches_scalar_all_matrices_16() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_nv12_rgba_equivalence(16, m, full);
    }
  }
}

#[test]
fn nv12_wasm_rgba_matches_scalar_widths() {
  for w in [18usize, 30, 34, 1920, 1922] {
    check_nv12_rgba_equivalence(w, ColorMatrix::Bt601, false);
  }
}

#[test]
fn nv21_wasm_rgba_matches_scalar_all_matrices_16() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_nv21_rgba_equivalence(16, m, full);
    }
  }
}

#[test]
fn nv21_wasm_rgba_matches_scalar_widths() {
  for w in [18usize, 30, 34, 1920, 1922] {
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
fn simd128_hsv_matches_scalar() {
  let rgb: std::vec::Vec<u8> = (0..1921 * 3)
    .map(|i| ((i * 37 + 11) & 0xFF) as u8)
    .collect();
  for w in [1usize, 15, 16, 17, 31, 1920, 1921] {
    check_hsv_equivalence(&rgb[..w * 3], w);
  }
}

// ---- yuv420p10 scalar-equivalence -----------------------------------
//
// These tests compile only for `target_arch = "wasm32"` (via the
// outer `target_feature = "simd128"` gate on the module). CI
// executes them under wasmtime in the `test-wasm-simd128` job
// (see `.github/workflows/ci.yml`): the lib is compiled for
// `wasm32-wasip1` with `-C target-feature=+simd128` and
// `CARGO_TARGET_WASM32_WASIP1_RUNNER=wasmtime run --` passes each
// compiled `.wasm` test binary to wasmtime. Every scalar‑
// equivalence check below runs on real SIMD instructions, not
// just a compile check.

fn p10_plane(n: usize, seed: usize) -> std::vec::Vec<u16> {
  (0..n)
    .map(|i| ((i * seed + seed * 3) & 0x3FF) as u16)
    .collect()
}

fn check_p10_u8_simd128_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
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
      "simd128 10→u8 diverges at byte {first_diff} (width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} simd={}",
      rgb_scalar[first_diff], rgb_simd[first_diff]
    );
  }
}

fn check_p10_u16_simd128_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
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
      "simd128 10→u16 diverges at elem {first_diff} (width={width}, matrix={matrix:?}, full_range={full_range}): scalar={} simd={}",
      rgb_scalar[first_diff], rgb_simd[first_diff]
    );
  }
}

#[test]
fn simd128_p10_u8_matches_scalar_all_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_p10_u8_simd128_equivalence(16, m, full);
    }
  }
}

#[test]
fn simd128_p10_u16_matches_scalar_all_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_p10_u16_simd128_equivalence(16, m, full);
    }
  }
}

#[test]
fn simd128_p10_matches_scalar_tail_widths() {
  for w in [18usize, 30, 34, 1922] {
    check_p10_u8_simd128_equivalence(w, ColorMatrix::Bt601, false);
    check_p10_u16_simd128_equivalence(w, ColorMatrix::Bt709, true);
  }
}

#[test]
fn simd128_p10_matches_scalar_1920() {
  check_p10_u8_simd128_equivalence(1920, ColorMatrix::Bt709, false);
  check_p10_u16_simd128_equivalence(1920, ColorMatrix::Bt2020Ncl, false);
}

// ---- yuv420p_n<BITS> simd128 scalar-equivalence (BITS=9 coverage) ---

fn p_n_plane_simd128<const BITS: u32>(n: usize, seed: usize) -> std::vec::Vec<u16> {
  let mask = ((1u32 << BITS) - 1) as u16;
  (0..n)
    .map(|i| ((i.wrapping_mul(seed).wrapping_add(seed * 3)) as u16) & mask)
    .collect()
}

fn check_p_n_u8_simd128_equivalence<const BITS: u32>(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let y = p_n_plane_simd128::<BITS>(width, 37);
  let u = p_n_plane_simd128::<BITS>(width / 2, 53);
  let v = p_n_plane_simd128::<BITS>(width / 2, 71);
  let mut rgb_scalar = std::vec![0u8; width * 3];
  let mut rgb_simd = std::vec![0u8; width * 3];
  scalar::yuv_420p_n_to_rgb_row::<BITS>(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    yuv_420p_n_to_rgb_row::<BITS>(&y, &u, &v, &mut rgb_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_simd,
    "simd128 yuv_420p_n<{BITS}>→u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_p_n_u16_simd128_equivalence<const BITS: u32>(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let y = p_n_plane_simd128::<BITS>(width, 37);
  let u = p_n_plane_simd128::<BITS>(width / 2, 53);
  let v = p_n_plane_simd128::<BITS>(width / 2, 71);
  let mut rgb_scalar = std::vec![0u16; width * 3];
  let mut rgb_simd = std::vec![0u16; width * 3];
  scalar::yuv_420p_n_to_rgb_u16_row::<BITS>(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    yuv_420p_n_to_rgb_u16_row::<BITS>(&y, &u, &v, &mut rgb_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_simd,
    "simd128 yuv_420p_n<{BITS}>→u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[test]
fn simd128_yuv420p9_matches_scalar_all_matrices_and_ranges() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_p_n_u8_simd128_equivalence::<9>(16, m, full);
      check_p_n_u16_simd128_equivalence::<9>(16, m, full);
    }
  }
}

#[test]
fn simd128_yuv420p9_matches_scalar_tail_and_large_widths() {
  for w in [18usize, 30, 34, 1922] {
    check_p_n_u8_simd128_equivalence::<9>(w, ColorMatrix::Bt601, false);
    check_p_n_u16_simd128_equivalence::<9>(w, ColorMatrix::Bt709, true);
  }
  check_p_n_u8_simd128_equivalence::<9>(1920, ColorMatrix::Bt709, false);
  check_p_n_u16_simd128_equivalence::<9>(1920, ColorMatrix::Bt2020Ncl, false);
}

// ---- P010 simd128 scalar-equivalence --------------------------------

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

fn check_p010_u8_simd128_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
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
  assert_eq!(rgb_scalar, rgb_simd, "simd128 P010→u8 diverges");
}

fn check_p010_u16_simd128_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
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
  assert_eq!(rgb_scalar, rgb_simd, "simd128 P010→u16 diverges");
}

#[test]
fn simd128_p010_u8_matches_scalar_all_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_p010_u8_simd128_equivalence(16, m, full);
    }
  }
}

#[test]
fn simd128_p010_u16_matches_scalar_all_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_p010_u16_simd128_equivalence(16, m, full);
    }
  }
}

#[test]
fn simd128_p010_matches_scalar_tail_widths() {
  for w in [18usize, 30, 34, 1922] {
    check_p010_u8_simd128_equivalence(w, ColorMatrix::Bt601, false);
    check_p010_u16_simd128_equivalence(w, ColorMatrix::Bt709, true);
  }
}

#[test]
fn simd128_p010_matches_scalar_1920() {
  check_p010_u8_simd128_equivalence(1920, ColorMatrix::Bt709, false);
  check_p010_u16_simd128_equivalence(1920, ColorMatrix::Bt2020Ncl, false);
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

fn check_planar_u8_simd128_equivalence_n<const BITS: u32>(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
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
    "simd128 planar {BITS}-bit → u8 diverges"
  );
}

fn check_planar_u16_simd128_equivalence_n<const BITS: u32>(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
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
    "simd128 planar {BITS}-bit → u16 diverges"
  );
}

fn check_pn_u8_simd128_equivalence_n<const BITS: u32>(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
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
  assert_eq!(rgb_scalar, rgb_simd, "simd128 Pn {BITS}-bit → u8 diverges");
}

fn check_pn_u16_simd128_equivalence_n<const BITS: u32>(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
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
  assert_eq!(rgb_scalar, rgb_simd, "simd128 Pn {BITS}-bit → u16 diverges");
}

#[test]
fn simd128_p12_matches_scalar_all_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_planar_u8_simd128_equivalence_n::<12>(16, m, full);
      check_planar_u16_simd128_equivalence_n::<12>(16, m, full);
      check_pn_u8_simd128_equivalence_n::<12>(16, m, full);
      check_pn_u16_simd128_equivalence_n::<12>(16, m, full);
    }
  }
}

#[test]
fn simd128_p14_matches_scalar_all_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_planar_u8_simd128_equivalence_n::<14>(16, m, full);
      check_planar_u16_simd128_equivalence_n::<14>(16, m, full);
    }
  }
}

#[test]
fn simd128_p12_matches_scalar_tail_widths() {
  for w in [18usize, 30, 34, 1922] {
    check_planar_u8_simd128_equivalence_n::<12>(w, ColorMatrix::Bt601, false);
    check_planar_u16_simd128_equivalence_n::<12>(w, ColorMatrix::Bt709, true);
    check_pn_u8_simd128_equivalence_n::<12>(w, ColorMatrix::Bt601, false);
    check_pn_u16_simd128_equivalence_n::<12>(w, ColorMatrix::Bt2020Ncl, false);
  }
}

#[test]
fn simd128_p14_matches_scalar_tail_widths() {
  for w in [18usize, 30, 34, 1922] {
    check_planar_u8_simd128_equivalence_n::<14>(w, ColorMatrix::Bt601, false);
    check_planar_u16_simd128_equivalence_n::<14>(w, ColorMatrix::Bt709, true);
  }
}

// ---- 16-bit (full-range u16 samples) simd128 equivalence ------------

fn p16_plane_wasm(n: usize, seed: usize) -> std::vec::Vec<u16> {
  (0..n)
    .map(|i| ((i.wrapping_mul(seed).wrapping_add(seed * 3)) & 0xFFFF) as u16)
    .collect()
}

fn check_yuv420p16_u8_simd128_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y = p16_plane_wasm(width, 37);
  let u = p16_plane_wasm(width / 2, 53);
  let v = p16_plane_wasm(width / 2, 71);
  let mut rgb_scalar = std::vec![0u8; width * 3];
  let mut rgb_simd = std::vec![0u8; width * 3];
  scalar::yuv_420p16_to_rgb_row(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    yuv_420p16_to_rgb_row(&y, &u, &v, &mut rgb_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_simd,
    "simd128 yuv420p16→u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_p16_u8_simd128_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y = p16_plane_wasm(width, 37);
  let u = p16_plane_wasm(width / 2, 53);
  let v = p16_plane_wasm(width / 2, 71);
  let uv = p010_uv_interleave(&u, &v);
  let mut rgb_scalar = std::vec![0u8; width * 3];
  let mut rgb_simd = std::vec![0u8; width * 3];
  scalar::p16_to_rgb_row(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    p16_to_rgb_row(&y, &uv, &mut rgb_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_simd,
    "simd128 p016→u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

/// u16-output equivalence for the 4:2:0 16-bit kernel. Uses
/// **exact-sized** chroma planes (`width / 2` samples) to catch
/// any over-read past the end of tight chroma rows — the
/// specific failure mode that originally shipped as a `v128_load`
/// (16-byte) read for only 4 u16 needed, now fixed via
/// `v128_load64_zero`.
fn check_yuv420p16_u16_simd128_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y = p16_plane_wasm(width, 37);
  let u = p16_plane_wasm(width / 2, 53);
  let v = p16_plane_wasm(width / 2, 71);
  let mut rgb_scalar = std::vec![0u16; width * 3];
  let mut rgb_simd = std::vec![0u16; width * 3];
  scalar::yuv_420p16_to_rgb_u16_row(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    yuv_420p16_to_rgb_u16_row(&y, &u, &v, &mut rgb_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_simd,
    "simd128 yuv420p16→u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_p16_u16_simd128_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y = p16_plane_wasm(width, 37);
  let u = p16_plane_wasm(width / 2, 53);
  let v = p16_plane_wasm(width / 2, 71);
  let uv = p010_uv_interleave(&u, &v);
  let mut rgb_scalar = std::vec![0u16; width * 3];
  let mut rgb_simd = std::vec![0u16; width * 3];
  scalar::p16_to_rgb_u16_row(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    p16_to_rgb_u16_row(&y, &uv, &mut rgb_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_simd,
    "simd128 p016→u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_yuv444p16_u16_simd128_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y = p16_plane_wasm(width, 37);
  let u = p16_plane_wasm(width, 53);
  let v = p16_plane_wasm(width, 71);
  let mut rgb_scalar = std::vec![0u16; width * 3];
  let mut rgb_simd = std::vec![0u16; width * 3];
  scalar::yuv_444p16_to_rgb_u16_row(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    yuv_444p16_to_rgb_u16_row(&y, &u, &v, &mut rgb_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_simd,
    "simd128 yuv444p16→u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[test]
fn simd128_p16_matches_scalar_all_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_yuv420p16_u8_simd128_equivalence(16, m, full);
      check_p16_u8_simd128_equivalence(16, m, full);
    }
  }
}

#[test]
fn simd128_p16_matches_scalar_tail_widths() {
  for w in [18usize, 30, 34, 1922] {
    check_yuv420p16_u8_simd128_equivalence(w, ColorMatrix::Bt601, false);
    check_p16_u8_simd128_equivalence(w, ColorMatrix::Bt709, true);
  }
}

#[test]
fn simd128_p16_matches_scalar_1920() {
  check_yuv420p16_u8_simd128_equivalence(1920, ColorMatrix::Bt709, false);
  check_p16_u8_simd128_equivalence(1920, ColorMatrix::Bt2020Ncl, false);
}

#[test]
fn simd128_16bit_u16_matches_scalar_all_matrices() {
  // Every new 16-bit u16-output path at the smallest block-aligned
  // width (8 pixels for 420/p16, which matches the native SIMD
  // iteration; 4:4:4 uses 8 too).
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_yuv420p16_u16_simd128_equivalence(16, m, full);
      check_p16_u16_simd128_equivalence(16, m, full);
      check_yuv444p16_u16_simd128_equivalence(16, m, full);
    }
  }
}

#[test]
fn simd128_yuv420p16_u16_matches_scalar_tight_widths() {
  // Tight widths explicitly stress over-read of half-width chroma
  // planes (8 samples at width=16 → second 8-pixel SIMD iter reads
  // chroma from offset 4, formerly `v128_load` of 8 samples = 8
  // bytes past plane-end).
  for w in [8usize, 10, 16, 18, 24, 26, 1920, 1922] {
    check_yuv420p16_u16_simd128_equivalence(w, ColorMatrix::Bt709, false);
    check_p16_u16_simd128_equivalence(w, ColorMatrix::Bt2020Ncl, true);
    check_yuv444p16_u16_simd128_equivalence(w, ColorMatrix::Bt601, false);
  }
}

// ---- High-bit 4:2:0 RGBA equivalence (Ship 8 Tranche 5a) ----------
//
// RGBA wrappers share the math of their RGB siblings — only the store
// (and tail dispatch) branches on `ALPHA`. These tests pin that the
// SIMD RGBA path produces byte-identical output to the scalar RGBA
// reference.

fn check_planar_u8_simd128_rgba_equivalence_n<const BITS: u32>(
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
    "simd128 yuv_420p_n<{BITS}>→RGBA u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_pn_u8_simd128_rgba_equivalence_n<const BITS: u32>(
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
    "simd128 Pn<{BITS}>→RGBA u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_yuv420p16_u8_simd128_rgba_equivalence(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let y = p16_plane_wasm(width, 37);
  let u = p16_plane_wasm(width / 2, 53);
  let v = p16_plane_wasm(width / 2, 71);
  let mut rgba_scalar = std::vec![0u8; width * 4];
  let mut rgba_simd = std::vec![0u8; width * 4];
  scalar::yuv_420p16_to_rgba_row(&y, &u, &v, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    yuv_420p16_to_rgba_row(&y, &u, &v, &mut rgba_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgba_scalar, rgba_simd,
    "simd128 yuv_420p16→RGBA u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_p16_u8_simd128_rgba_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y = p16_plane_wasm(width, 37);
  let u = p16_plane_wasm(width / 2, 53);
  let v = p16_plane_wasm(width / 2, 71);
  let uv = p010_uv_interleave(&u, &v);
  let mut rgba_scalar = std::vec![0u8; width * 4];
  let mut rgba_simd = std::vec![0u8; width * 4];
  scalar::p16_to_rgba_row(&y, &uv, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    p16_to_rgba_row(&y, &uv, &mut rgba_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgba_scalar, rgba_simd,
    "simd128 P016→RGBA u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[test]
fn simd128_yuv420p_n_rgba_matches_scalar_all_bits() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_planar_u8_simd128_rgba_equivalence_n::<9>(16, m, full);
      check_planar_u8_simd128_rgba_equivalence_n::<10>(16, m, full);
      check_planar_u8_simd128_rgba_equivalence_n::<12>(16, m, full);
      check_planar_u8_simd128_rgba_equivalence_n::<14>(16, m, full);
    }
  }
}

#[test]
fn simd128_yuv420p_n_rgba_matches_scalar_tail_and_1920() {
  for w in [18usize, 30, 34, 1920, 1922] {
    check_planar_u8_simd128_rgba_equivalence_n::<9>(w, ColorMatrix::Bt601, false);
    check_planar_u8_simd128_rgba_equivalence_n::<10>(w, ColorMatrix::Bt709, true);
    check_planar_u8_simd128_rgba_equivalence_n::<12>(w, ColorMatrix::Bt2020Ncl, false);
    check_planar_u8_simd128_rgba_equivalence_n::<14>(w, ColorMatrix::YCgCo, true);
  }
}

#[test]
fn simd128_pn_rgba_matches_scalar_all_bits() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_pn_u8_simd128_rgba_equivalence_n::<10>(16, m, full);
      check_pn_u8_simd128_rgba_equivalence_n::<12>(16, m, full);
    }
  }
}

#[test]
fn simd128_pn_rgba_matches_scalar_tail_and_1920() {
  for w in [18usize, 30, 34, 1920, 1922] {
    check_pn_u8_simd128_rgba_equivalence_n::<10>(w, ColorMatrix::Bt601, false);
    check_pn_u8_simd128_rgba_equivalence_n::<12>(w, ColorMatrix::Bt709, true);
  }
}

#[test]
fn simd128_yuv420p16_rgba_matches_scalar_all_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_yuv420p16_u8_simd128_rgba_equivalence(16, m, full);
    }
  }
  for w in [18usize, 30, 34, 1920, 1922] {
    check_yuv420p16_u8_simd128_rgba_equivalence(w, ColorMatrix::Bt709, false);
  }
}

#[test]
fn simd128_p016_rgba_matches_scalar_all_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_p16_u8_simd128_rgba_equivalence(16, m, full);
    }
  }
  for w in [18usize, 30, 34, 1920, 1922] {
    check_p16_u8_simd128_rgba_equivalence(w, ColorMatrix::Bt709, false);
  }
}

// ---- High-bit 4:2:0 native-depth `u16` RGBA equivalence (Ship 8 Tranche 5b) ----

fn check_planar_u16_simd128_rgba_equivalence_n<const BITS: u32>(
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
    "simd128 yuv_420p_n<{BITS}>→RGBA u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_pn_u16_simd128_rgba_equivalence_n<const BITS: u32>(
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
    "simd128 Pn<{BITS}>→RGBA u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_yuv420p16_u16_simd128_rgba_equivalence(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let y = p16_plane_wasm(width, 37);
  let u = p16_plane_wasm(width / 2, 53);
  let v = p16_plane_wasm(width / 2, 71);
  let mut rgba_scalar = std::vec![0u16; width * 4];
  let mut rgba_simd = std::vec![0u16; width * 4];
  scalar::yuv_420p16_to_rgba_u16_row(&y, &u, &v, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    yuv_420p16_to_rgba_u16_row(&y, &u, &v, &mut rgba_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgba_scalar, rgba_simd,
    "simd128 yuv_420p16→RGBA u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_p16_u16_simd128_rgba_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y = p16_plane_wasm(width, 37);
  let u = p16_plane_wasm(width / 2, 53);
  let v = p16_plane_wasm(width / 2, 71);
  let uv = p010_uv_interleave(&u, &v);
  let mut rgba_scalar = std::vec![0u16; width * 4];
  let mut rgba_simd = std::vec![0u16; width * 4];
  scalar::p16_to_rgba_u16_row(&y, &uv, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    p16_to_rgba_u16_row(&y, &uv, &mut rgba_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgba_scalar, rgba_simd,
    "simd128 P016→RGBA u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[test]
fn simd128_yuv420p_n_rgba_u16_matches_scalar_all_bits() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_planar_u16_simd128_rgba_equivalence_n::<9>(16, m, full);
      check_planar_u16_simd128_rgba_equivalence_n::<10>(16, m, full);
      check_planar_u16_simd128_rgba_equivalence_n::<12>(16, m, full);
      check_planar_u16_simd128_rgba_equivalence_n::<14>(16, m, full);
    }
  }
}

#[test]
fn simd128_yuv420p_n_rgba_u16_matches_scalar_tail_and_1920() {
  for w in [18usize, 30, 34, 1920, 1922] {
    check_planar_u16_simd128_rgba_equivalence_n::<9>(w, ColorMatrix::Bt601, false);
    check_planar_u16_simd128_rgba_equivalence_n::<10>(w, ColorMatrix::Bt709, true);
    check_planar_u16_simd128_rgba_equivalence_n::<12>(w, ColorMatrix::Bt2020Ncl, false);
    check_planar_u16_simd128_rgba_equivalence_n::<14>(w, ColorMatrix::YCgCo, true);
  }
}

#[test]
fn simd128_pn_rgba_u16_matches_scalar_all_bits() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_pn_u16_simd128_rgba_equivalence_n::<10>(16, m, full);
      check_pn_u16_simd128_rgba_equivalence_n::<12>(16, m, full);
    }
  }
}

#[test]
fn simd128_pn_rgba_u16_matches_scalar_tail_and_1920() {
  for w in [18usize, 30, 34, 1920, 1922] {
    check_pn_u16_simd128_rgba_equivalence_n::<10>(w, ColorMatrix::Bt601, false);
    check_pn_u16_simd128_rgba_equivalence_n::<12>(w, ColorMatrix::Bt709, true);
  }
}

#[test]
fn simd128_yuv420p16_rgba_u16_matches_scalar_all_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_yuv420p16_u16_simd128_rgba_equivalence(16, m, full);
    }
  }
  for w in [18usize, 30, 34, 1920, 1922] {
    check_yuv420p16_u16_simd128_rgba_equivalence(w, ColorMatrix::Bt709, false);
  }
}

#[test]
fn simd128_p016_rgba_u16_matches_scalar_all_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_p16_u16_simd128_rgba_equivalence(16, m, full);
    }
  }
  for w in [18usize, 30, 34, 1920, 1922] {
    check_p16_u16_simd128_rgba_equivalence(w, ColorMatrix::Bt709, false);
  }
}

// ---- Pn 4:4:4 (P410 / P412 / P416) wasm simd128 equivalence ---------

fn high_bit_plane_wasm<const BITS: u32>(n: usize, seed: usize) -> std::vec::Vec<u16> {
  let mask = ((1u32 << BITS) - 1) as u16;
  let shift = 16 - BITS;
  (0..n)
    .map(|i| (((i.wrapping_mul(seed).wrapping_add(seed * 3)) as u16) & mask) << shift)
    .collect()
}

fn interleave_uv_wasm(u_full: &[u16], v_full: &[u16]) -> std::vec::Vec<u16> {
  debug_assert_eq!(u_full.len(), v_full.len());
  let mut out = std::vec::Vec::with_capacity(u_full.len() * 2);
  for i in 0..u_full.len() {
    out.push(u_full[i]);
    out.push(v_full[i]);
  }
  out
}

fn check_p_n_444_u8_simd128_equivalence<const BITS: u32>(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let y = high_bit_plane_wasm::<BITS>(width, 37);
  let u = high_bit_plane_wasm::<BITS>(width, 53);
  let v = high_bit_plane_wasm::<BITS>(width, 71);
  let uv = interleave_uv_wasm(&u, &v);
  let mut rgb_scalar = std::vec![0u8; width * 3];
  let mut rgb_simd = std::vec![0u8; width * 3];
  scalar::p_n_444_to_rgb_row::<BITS>(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    p_n_444_to_rgb_row::<BITS>(&y, &uv, &mut rgb_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_simd,
    "simd128 Pn4:4:4 {BITS}-bit → u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_p_n_444_u16_simd128_equivalence<const BITS: u32>(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let y = high_bit_plane_wasm::<BITS>(width, 37);
  let u = high_bit_plane_wasm::<BITS>(width, 53);
  let v = high_bit_plane_wasm::<BITS>(width, 71);
  let uv = interleave_uv_wasm(&u, &v);
  let mut rgb_scalar = std::vec![0u16; width * 3];
  let mut rgb_simd = std::vec![0u16; width * 3];
  scalar::p_n_444_to_rgb_u16_row::<BITS>(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    p_n_444_to_rgb_u16_row::<BITS>(&y, &uv, &mut rgb_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_simd,
    "simd128 Pn4:4:4 {BITS}-bit → u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_p_n_444_16_u8_simd128_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y = p16_plane_wasm(width, 37);
  let u = p16_plane_wasm(width, 53);
  let v = p16_plane_wasm(width, 71);
  let uv = interleave_uv_wasm(&u, &v);
  let mut rgb_scalar = std::vec![0u8; width * 3];
  let mut rgb_simd = std::vec![0u8; width * 3];
  scalar::p_n_444_16_to_rgb_row(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    p_n_444_16_to_rgb_row(&y, &uv, &mut rgb_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_simd,
    "simd128 P416 → u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_p_n_444_16_u16_simd128_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y = p16_plane_wasm(width, 37);
  let u = p16_plane_wasm(width, 53);
  let v = p16_plane_wasm(width, 71);
  let uv = interleave_uv_wasm(&u, &v);
  let mut rgb_scalar = std::vec![0u16; width * 3];
  let mut rgb_simd = std::vec![0u16; width * 3];
  scalar::p_n_444_16_to_rgb_u16_row(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    p_n_444_16_to_rgb_u16_row(&y, &uv, &mut rgb_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_simd,
    "simd128 P416 → u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[test]
fn simd128_p410_matches_scalar_all_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_p_n_444_u8_simd128_equivalence::<10>(16, m, full);
      check_p_n_444_u16_simd128_equivalence::<10>(16, m, full);
    }
  }
}

#[test]
fn simd128_p412_matches_scalar_all_matrices() {
  for m in [ColorMatrix::Bt709, ColorMatrix::Bt2020Ncl] {
    for full in [true, false] {
      check_p_n_444_u8_simd128_equivalence::<12>(16, m, full);
      check_p_n_444_u16_simd128_equivalence::<12>(16, m, full);
    }
  }
}

#[test]
fn simd128_p410_p412_matches_scalar_tail_widths() {
  for w in [1usize, 3, 7, 15, 17, 31, 33, 1920, 1921] {
    check_p_n_444_u8_simd128_equivalence::<10>(w, ColorMatrix::Bt601, false);
    check_p_n_444_u16_simd128_equivalence::<10>(w, ColorMatrix::Bt709, true);
    check_p_n_444_u8_simd128_equivalence::<12>(w, ColorMatrix::Bt2020Ncl, false);
  }
}

#[test]
fn simd128_p416_matches_scalar_all_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_p_n_444_16_u8_simd128_equivalence(16, m, full);
      check_p_n_444_16_u16_simd128_equivalence(16, m, full);
    }
  }
}

#[test]
fn simd128_p416_matches_scalar_tail_widths() {
  for w in [1usize, 3, 7, 8, 9, 15, 16, 17, 31, 33, 1920, 1921] {
    check_p_n_444_16_u8_simd128_equivalence(w, ColorMatrix::Bt709, false);
    check_p_n_444_16_u16_simd128_equivalence(w, ColorMatrix::Bt2020Ncl, true);
  }
}

// ---- High-bit 4:4:4 u8 RGBA equivalence (Ship 8 Tranche 7b) ---------
//
// Mirrors the 4:2:0 RGBA pattern in PR #25 (Tranche 5a). Each kernel
// family — Yuv444p_n (BITS-generic), Yuv444p16, Pn_444 (BITS-generic),
// Pn_444_16 — has its wasm simd128 RGBA kernel byte-pinned against the
// scalar reference at the natural width and a sweep of tail widths.
// (Module-level cfg gates these on `target_feature = "simd128"`, so no
// per-test feature guard is needed.)

fn check_yuv444p_n_u8_simd128_rgba_equivalence<const BITS: u32>(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let y = planar_n_plane::<BITS>(width, 37);
  let u = planar_n_plane::<BITS>(width, 53);
  let v = planar_n_plane::<BITS>(width, 71);
  let mut rgba_scalar = std::vec![0u8; width * 4];
  let mut rgba_wasm = std::vec![0u8; width * 4];
  scalar::yuv_444p_n_to_rgba_row::<BITS>(&y, &u, &v, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    yuv_444p_n_to_rgba_row::<BITS>(&y, &u, &v, &mut rgba_wasm, width, matrix, full_range);
  }
  assert_eq!(
    rgba_scalar, rgba_wasm,
    "wasm simd128 Yuv444p<{BITS}> → RGBA u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_pn_444_u8_simd128_rgba_equivalence<const BITS: u32>(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let y = high_bit_plane_wasm::<BITS>(width, 37);
  let u = high_bit_plane_wasm::<BITS>(width, 53);
  let v = high_bit_plane_wasm::<BITS>(width, 71);
  let uv = interleave_uv_wasm(&u, &v);
  let mut rgba_scalar = std::vec![0u8; width * 4];
  let mut rgba_wasm = std::vec![0u8; width * 4];
  scalar::p_n_444_to_rgba_row::<BITS>(&y, &uv, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    p_n_444_to_rgba_row::<BITS>(&y, &uv, &mut rgba_wasm, width, matrix, full_range);
  }
  assert_eq!(
    rgba_scalar, rgba_wasm,
    "wasm simd128 Pn4:4:4<{BITS}> → RGBA u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_yuv444p16_u8_simd128_rgba_equivalence(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let y = p16_plane_wasm(width, 37);
  let u = p16_plane_wasm(width, 53);
  let v = p16_plane_wasm(width, 71);
  let mut rgba_scalar = std::vec![0u8; width * 4];
  let mut rgba_wasm = std::vec![0u8; width * 4];
  scalar::yuv_444p16_to_rgba_row(&y, &u, &v, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    yuv_444p16_to_rgba_row(&y, &u, &v, &mut rgba_wasm, width, matrix, full_range);
  }
  assert_eq!(
    rgba_scalar, rgba_wasm,
    "wasm simd128 Yuv444p16 → RGBA u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_p_n_444_16_u8_simd128_rgba_equivalence(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let y = p16_plane_wasm(width, 37);
  let u = p16_plane_wasm(width, 53);
  let v = p16_plane_wasm(width, 71);
  let uv = interleave_uv_wasm(&u, &v);
  let mut rgba_scalar = std::vec![0u8; width * 4];
  let mut rgba_wasm = std::vec![0u8; width * 4];
  scalar::p_n_444_16_to_rgba_row(&y, &uv, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    p_n_444_16_to_rgba_row(&y, &uv, &mut rgba_wasm, width, matrix, full_range);
  }
  assert_eq!(
    rgba_scalar, rgba_wasm,
    "wasm simd128 P416 → RGBA u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[test]
fn simd128_yuv444p_n_rgba_matches_scalar_all_bits() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_yuv444p_n_u8_simd128_rgba_equivalence::<9>(16, m, full);
      check_yuv444p_n_u8_simd128_rgba_equivalence::<10>(16, m, full);
      check_yuv444p_n_u8_simd128_rgba_equivalence::<12>(16, m, full);
      check_yuv444p_n_u8_simd128_rgba_equivalence::<14>(16, m, full);
    }
  }
}

#[test]
fn simd128_yuv444p_n_rgba_matches_scalar_tail_and_widths() {
  for w in [17usize, 31, 47, 63, 1920, 1922] {
    check_yuv444p_n_u8_simd128_rgba_equivalence::<9>(w, ColorMatrix::Bt601, false);
    check_yuv444p_n_u8_simd128_rgba_equivalence::<10>(w, ColorMatrix::Bt709, true);
    check_yuv444p_n_u8_simd128_rgba_equivalence::<12>(w, ColorMatrix::Bt2020Ncl, false);
    check_yuv444p_n_u8_simd128_rgba_equivalence::<14>(w, ColorMatrix::YCgCo, true);
  }
}

#[test]
fn simd128_pn_444_rgba_matches_scalar_all_bits() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_pn_444_u8_simd128_rgba_equivalence::<10>(16, m, full);
      check_pn_444_u8_simd128_rgba_equivalence::<12>(16, m, full);
    }
  }
}

#[test]
fn simd128_pn_444_rgba_matches_scalar_tail_and_widths() {
  for w in [17usize, 31, 47, 63, 1920, 1922] {
    check_pn_444_u8_simd128_rgba_equivalence::<10>(w, ColorMatrix::Bt601, false);
    check_pn_444_u8_simd128_rgba_equivalence::<12>(w, ColorMatrix::Bt709, true);
  }
}

#[test]
fn simd128_yuv444p16_rgba_matches_scalar_all_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_yuv444p16_u8_simd128_rgba_equivalence(16, m, full);
    }
  }
  for w in [17usize, 31, 47, 63, 1920, 1922] {
    check_yuv444p16_u8_simd128_rgba_equivalence(w, ColorMatrix::Bt709, false);
  }
}

#[test]
fn simd128_p416_rgba_matches_scalar_all_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_p_n_444_16_u8_simd128_rgba_equivalence(16, m, full);
    }
  }
  for w in [17usize, 31, 47, 63, 1920, 1922] {
    check_p_n_444_16_u8_simd128_rgba_equivalence(w, ColorMatrix::Bt709, false);
  }
}
