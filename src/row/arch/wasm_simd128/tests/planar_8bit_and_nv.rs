use super::{
  super::*, high_bit_plane_wasm, interleave_uv_wasm, p_n_packed_plane, p010_uv_interleave,
  p16_plane_wasm, planar_n_plane,
};

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
  let mut rgba_simd = std::vec![0u8; width * 4];

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
      &mut rgba_simd,
      width,
      matrix,
      full_range,
    );
  }
  assert_eq!(
    rgba_scalar, rgba_simd,
    "wasm simd128 Yuva444p → RGBA u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range}, alpha_seed={alpha_seed})"
  );
}

#[test]
fn simd128_yuva444p_rgba_matches_scalar_all_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_yuv_444_rgba_with_alpha_src_equivalence(16, m, full, 89);
    }
  }
}

#[test]
fn simd128_yuva444p_rgba_matches_scalar_widths_and_alpha() {
  for w in [16usize, 17, 31, 47, 1920, 1922] {
    check_yuv_444_rgba_with_alpha_src_equivalence(w, ColorMatrix::Bt709, true, 89);
  }
  for seed in [13usize, 41, 127, 211] {
    check_yuv_444_rgba_with_alpha_src_equivalence(16, ColorMatrix::Bt601, false, seed);
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
