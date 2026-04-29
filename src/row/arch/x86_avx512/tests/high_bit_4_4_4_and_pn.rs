use super::{
  super::*, high_bit_plane_avx512, interleave_uv_avx512, p_n_packed_plane, p010_uv_interleave,
  p16_plane_avx512, planar_n_plane,
};

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
      check_planar_u8_avx512_rgba_equivalence_n::<9>(64, m, full);
      check_planar_u8_avx512_rgba_equivalence_n::<10>(64, m, full);
      check_planar_u8_avx512_rgba_equivalence_n::<12>(64, m, full);
      check_planar_u8_avx512_rgba_equivalence_n::<14>(64, m, full);
    }
  }
}

#[test]
fn avx512_yuv420p_n_rgba_matches_scalar_tail_and_1920() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  for w in [66usize, 96, 126, 1920, 1922] {
    check_planar_u8_avx512_rgba_equivalence_n::<9>(w, ColorMatrix::Bt601, false);
    check_planar_u8_avx512_rgba_equivalence_n::<10>(w, ColorMatrix::Bt709, true);
    check_planar_u8_avx512_rgba_equivalence_n::<12>(w, ColorMatrix::Bt2020Ncl, false);
    check_planar_u8_avx512_rgba_equivalence_n::<14>(w, ColorMatrix::YCgCo, true);
  }
}

#[test]
fn avx512_pn_rgba_matches_scalar_all_bits() {
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
      check_pn_u8_avx512_rgba_equivalence_n::<10>(64, m, full);
      check_pn_u8_avx512_rgba_equivalence_n::<12>(64, m, full);
    }
  }
}

#[test]
fn avx512_pn_rgba_matches_scalar_tail_and_1920() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  for w in [66usize, 96, 126, 1920, 1922] {
    check_pn_u8_avx512_rgba_equivalence_n::<10>(w, ColorMatrix::Bt601, false);
    check_pn_u8_avx512_rgba_equivalence_n::<12>(w, ColorMatrix::Bt709, true);
  }
}

#[test]
fn avx512_yuv420p16_rgba_matches_scalar_all_matrices() {
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
      check_yuv420p16_u8_avx512_rgba_equivalence(64, m, full);
    }
  }
  for w in [66usize, 96, 126, 1920, 1922] {
    check_yuv420p16_u8_avx512_rgba_equivalence(w, ColorMatrix::Bt709, false);
  }
}

#[test]
fn avx512_p016_rgba_matches_scalar_all_matrices() {
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
      check_p16_u8_avx512_rgba_equivalence(64, m, full);
    }
  }
  for w in [66usize, 96, 126, 1920, 1922] {
    check_p16_u8_avx512_rgba_equivalence(w, ColorMatrix::Bt709, false);
  }
}

// ---- High-bit 4:2:0 native-depth `u16` RGBA equivalence (Ship 8 Tranche 5b) ----

fn check_planar_u16_avx512_rgba_equivalence_n<const BITS: u32>(
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
    "AVX-512 yuv_420p_n<{BITS}>→RGBA u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_pn_u16_avx512_rgba_equivalence_n<const BITS: u32>(
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
    "AVX-512 Pn<{BITS}>→RGBA u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_yuv420p16_u16_avx512_rgba_equivalence(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let y = p16_plane_avx512(width, 37);
  let u = p16_plane_avx512(width / 2, 53);
  let v = p16_plane_avx512(width / 2, 71);
  let mut rgba_scalar = std::vec![0u16; width * 4];
  let mut rgba_simd = std::vec![0u16; width * 4];
  scalar::yuv_420p16_to_rgba_u16_row(&y, &u, &v, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    yuv_420p16_to_rgba_u16_row(&y, &u, &v, &mut rgba_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgba_scalar, rgba_simd,
    "AVX-512 yuv_420p16→RGBA u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_p16_u16_avx512_rgba_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y = p16_plane_avx512(width, 37);
  let u = p16_plane_avx512(width / 2, 53);
  let v = p16_plane_avx512(width / 2, 71);
  let uv = p010_uv_interleave(&u, &v);
  let mut rgba_scalar = std::vec![0u16; width * 4];
  let mut rgba_simd = std::vec![0u16; width * 4];
  scalar::p16_to_rgba_u16_row(&y, &uv, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    p16_to_rgba_u16_row(&y, &uv, &mut rgba_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgba_scalar, rgba_simd,
    "AVX-512 P016→RGBA u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[test]
fn avx512_yuv420p_n_rgba_u16_matches_scalar_all_bits() {
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
      check_planar_u16_avx512_rgba_equivalence_n::<9>(64, m, full);
      check_planar_u16_avx512_rgba_equivalence_n::<10>(64, m, full);
      check_planar_u16_avx512_rgba_equivalence_n::<12>(64, m, full);
      check_planar_u16_avx512_rgba_equivalence_n::<14>(64, m, full);
    }
  }
}

#[test]
fn avx512_yuv420p_n_rgba_u16_matches_scalar_tail_and_1920() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  for w in [66usize, 96, 126, 1920, 1922] {
    check_planar_u16_avx512_rgba_equivalence_n::<9>(w, ColorMatrix::Bt601, false);
    check_planar_u16_avx512_rgba_equivalence_n::<10>(w, ColorMatrix::Bt709, true);
    check_planar_u16_avx512_rgba_equivalence_n::<12>(w, ColorMatrix::Bt2020Ncl, false);
    check_planar_u16_avx512_rgba_equivalence_n::<14>(w, ColorMatrix::YCgCo, true);
  }
}

#[test]
fn avx512_pn_rgba_u16_matches_scalar_all_bits() {
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
      check_pn_u16_avx512_rgba_equivalence_n::<10>(64, m, full);
      check_pn_u16_avx512_rgba_equivalence_n::<12>(64, m, full);
    }
  }
}

#[test]
fn avx512_pn_rgba_u16_matches_scalar_tail_and_1920() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  for w in [66usize, 96, 126, 1920, 1922] {
    check_pn_u16_avx512_rgba_equivalence_n::<10>(w, ColorMatrix::Bt601, false);
    check_pn_u16_avx512_rgba_equivalence_n::<12>(w, ColorMatrix::Bt709, true);
  }
}

#[test]
fn avx512_yuv420p16_rgba_u16_matches_scalar_all_matrices() {
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
      check_yuv420p16_u16_avx512_rgba_equivalence(64, m, full);
    }
  }
  for w in [66usize, 96, 126, 1920, 1922] {
    check_yuv420p16_u16_avx512_rgba_equivalence(w, ColorMatrix::Bt709, false);
  }
}

#[test]
fn avx512_p016_rgba_u16_matches_scalar_all_matrices() {
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
      check_p16_u16_avx512_rgba_equivalence(64, m, full);
    }
  }
  for w in [66usize, 96, 126, 1920, 1922] {
    check_p16_u16_avx512_rgba_equivalence(w, ColorMatrix::Bt709, false);
  }
}

// ---- Pn 4:4:4 (P410 / P412 / P416) AVX-512 equivalence -------------

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

// ---- High-bit 4:4:4 u8 RGBA equivalence (Ship 8 Tranche 7b) ---------
//
// Mirrors the 4:2:0 RGBA pattern in PR #25 (Tranche 5a). Each kernel
// family — Yuv444p_n (BITS-generic), Yuv444p16, Pn_444 (BITS-generic),
// Pn_444_16 — has its AVX-512 RGBA kernel byte-pinned against the
// scalar reference at the natural width and a sweep of tail widths.
// Each test gates on `is_x86_feature_detected!("avx512bw")` to stay
// clean under sanitizer/Miri/non-feature-flagged CI runners.

fn check_yuv444p_n_u8_avx512_rgba_equivalence<const BITS: u32>(
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
    "AVX-512 Yuv444p<{BITS}> → RGBA u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_pn_444_u8_avx512_rgba_equivalence<const BITS: u32>(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let y = high_bit_plane_avx512::<BITS>(width, 37);
  let u = high_bit_plane_avx512::<BITS>(width, 53);
  let v = high_bit_plane_avx512::<BITS>(width, 71);
  let uv = interleave_uv_avx512(&u, &v);
  let mut rgba_scalar = std::vec![0u8; width * 4];
  let mut rgba_simd = std::vec![0u8; width * 4];
  scalar::p_n_444_to_rgba_row::<BITS>(&y, &uv, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    p_n_444_to_rgba_row::<BITS>(&y, &uv, &mut rgba_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgba_scalar, rgba_simd,
    "AVX-512 Pn4:4:4<{BITS}> → RGBA u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_yuv444p16_u8_avx512_rgba_equivalence(width: usize, matrix: ColorMatrix, full_range: bool) {
  let y = p16_plane_avx512(width, 37);
  let u = p16_plane_avx512(width, 53);
  let v = p16_plane_avx512(width, 71);
  let mut rgba_scalar = std::vec![0u8; width * 4];
  let mut rgba_simd = std::vec![0u8; width * 4];
  scalar::yuv_444p16_to_rgba_row(&y, &u, &v, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    yuv_444p16_to_rgba_row(&y, &u, &v, &mut rgba_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgba_scalar, rgba_simd,
    "AVX-512 Yuv444p16 → RGBA u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_p_n_444_16_u8_avx512_rgba_equivalence(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  let y = p16_plane_avx512(width, 37);
  let u = p16_plane_avx512(width, 53);
  let v = p16_plane_avx512(width, 71);
  let uv = interleave_uv_avx512(&u, &v);
  let mut rgba_scalar = std::vec![0u8; width * 4];
  let mut rgba_simd = std::vec![0u8; width * 4];
  scalar::p_n_444_16_to_rgba_row(&y, &uv, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    p_n_444_16_to_rgba_row(&y, &uv, &mut rgba_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgba_scalar, rgba_simd,
    "AVX-512 P416 → RGBA u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[test]
fn avx512_yuv444p_n_rgba_matches_scalar_all_bits() {
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
      check_yuv444p_n_u8_avx512_rgba_equivalence::<9>(64, m, full);
      check_yuv444p_n_u8_avx512_rgba_equivalence::<10>(64, m, full);
      check_yuv444p_n_u8_avx512_rgba_equivalence::<12>(64, m, full);
      check_yuv444p_n_u8_avx512_rgba_equivalence::<14>(64, m, full);
    }
  }
}

#[test]
fn avx512_yuv444p_n_rgba_matches_scalar_tail_and_widths() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  for w in [17usize, 31, 47, 63, 1920, 1922] {
    check_yuv444p_n_u8_avx512_rgba_equivalence::<9>(w, ColorMatrix::Bt601, false);
    check_yuv444p_n_u8_avx512_rgba_equivalence::<10>(w, ColorMatrix::Bt709, true);
    check_yuv444p_n_u8_avx512_rgba_equivalence::<12>(w, ColorMatrix::Bt2020Ncl, false);
    check_yuv444p_n_u8_avx512_rgba_equivalence::<14>(w, ColorMatrix::YCgCo, true);
  }
}

#[test]
fn avx512_pn_444_rgba_matches_scalar_all_bits() {
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
      check_pn_444_u8_avx512_rgba_equivalence::<10>(64, m, full);
      check_pn_444_u8_avx512_rgba_equivalence::<12>(64, m, full);
    }
  }
}

#[test]
fn avx512_pn_444_rgba_matches_scalar_tail_and_widths() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  for w in [17usize, 31, 47, 63, 1920, 1922] {
    check_pn_444_u8_avx512_rgba_equivalence::<10>(w, ColorMatrix::Bt601, false);
    check_pn_444_u8_avx512_rgba_equivalence::<12>(w, ColorMatrix::Bt709, true);
  }
}

#[test]
fn avx512_yuv444p16_rgba_matches_scalar_all_matrices() {
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
      check_yuv444p16_u8_avx512_rgba_equivalence(64, m, full);
    }
  }
  for w in [17usize, 31, 47, 63, 1920, 1922] {
    check_yuv444p16_u8_avx512_rgba_equivalence(w, ColorMatrix::Bt709, false);
  }
}

fn check_yuv444p16_u8_avx512_rgba_with_alpha_src_equivalence(
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  alpha_seed: usize,
) {
  let y = p16_plane_avx512(width, 37);
  let u = p16_plane_avx512(width, 53);
  let v = p16_plane_avx512(width, 71);
  let a_src = p16_plane_avx512(width, alpha_seed);
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
    "AVX-512 Yuva444p16 → RGBA u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range}, alpha_seed={alpha_seed})"
  );
}

#[test]
fn avx512_yuva444p16_rgba_matches_scalar_all_matrices() {
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
      check_yuv444p16_u8_avx512_rgba_with_alpha_src_equivalence(64, m, full, 89);
    }
  }
}

#[test]
fn avx512_yuva444p16_rgba_matches_scalar_widths_and_alpha() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  for w in [64usize, 65, 79, 95, 1920, 1922] {
    check_yuv444p16_u8_avx512_rgba_with_alpha_src_equivalence(w, ColorMatrix::Bt709, true, 89);
  }
  for seed in [13usize, 41, 127, 211] {
    check_yuv444p16_u8_avx512_rgba_with_alpha_src_equivalence(64, ColorMatrix::Bt601, false, seed);
  }
}

#[test]
fn avx512_p416_rgba_matches_scalar_all_matrices() {
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
      check_p_n_444_16_u8_avx512_rgba_equivalence(64, m, full);
    }
  }
  for w in [17usize, 31, 47, 63, 1920, 1922] {
    check_p_n_444_16_u8_avx512_rgba_equivalence(w, ColorMatrix::Bt709, false);
  }
}
