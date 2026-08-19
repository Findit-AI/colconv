use super::{super::*, p16_plane};
// Planar high-bit plane builder — used only by the `yuv-planar`-gated
// planar-vs-SSE4.1 equivalence helpers below.
#[cfg(feature = "yuv-planar")]
use super::planar_n_plane;
// Semi-planar (`p_n` / `p_n_444`) plane / UV-interleave builders — used
// only by the `yuv-semi-planar`-gated equivalence helpers / tests below.
#[cfg(feature = "yuv-semi-planar")]
use super::{high_bit_plane_sse41, interleave_uv_sse41, p_n_packed_plane, p010_uv_interleave};

// ---- High-bit 4:2:0 RGBA equivalence (Ship 8 Tranche 5a) ----------
//
// RGBA wrappers share the math of their RGB siblings — only the store
// (and tail dispatch) branches on `ALPHA`. These tests pin that the
// SIMD RGBA path produces byte-identical output to the scalar RGBA
// reference.

#[cfg(feature = "yuv-planar")]
fn check_planar_u8_sse41_rgba_equivalence_n<const BITS: u32>(
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  let y = planar_n_plane::<BITS>(width, 37);
  let u = planar_n_plane::<BITS>(width / 2, 53);
  let v = planar_n_plane::<BITS>(width / 2, 71);
  let mut rgba_scalar = std::vec![0u8; width * 4];
  let mut rgba_simd = std::vec![0u8; width * 4];
  scalar::yuv_420p_n_to_rgba_row::<BITS, false>(
    &y,
    &u,
    &v,
    &mut rgba_scalar,
    width,
    matrix,
    full_range,
  );
  unsafe {
    yuv_420p_n_to_rgba_row::<BITS, false>(&y, &u, &v, &mut rgba_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgba_scalar, rgba_simd,
    "SSE4.1 yuv_420p_n<{BITS}>→RGBA u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[cfg(feature = "yuv-semi-planar")]
fn check_pn_u8_sse41_rgba_equivalence_n<const BITS: u32>(
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  let y = p_n_packed_plane::<BITS>(width, 37);
  let u = p_n_packed_plane::<BITS>(width / 2, 53);
  let v = p_n_packed_plane::<BITS>(width / 2, 71);
  let uv = p010_uv_interleave(&u, &v);
  let mut rgba_scalar = std::vec![0u8; width * 4];
  let mut rgba_simd = std::vec![0u8; width * 4];
  scalar::p_n_to_rgba_row::<BITS, false>(&y, &uv, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    p_n_to_rgba_row::<BITS, false>(&y, &uv, &mut rgba_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgba_scalar, rgba_simd,
    "SSE4.1 Pn<{BITS}>→RGBA u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[cfg(feature = "yuv-planar")]
fn check_yuv420p16_u8_sse41_rgba_equivalence(width: usize, matrix: KernelMatrix, full_range: bool) {
  let y = p16_plane(width, 37);
  let u = p16_plane(width / 2, 53);
  let v = p16_plane(width / 2, 71);
  let mut rgba_scalar = std::vec![0u8; width * 4];
  let mut rgba_simd = std::vec![0u8; width * 4];
  scalar::yuv_420p16_to_rgba_row::<false>(&y, &u, &v, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    yuv_420p16_to_rgba_row::<false>(&y, &u, &v, &mut rgba_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgba_scalar, rgba_simd,
    "SSE4.1 yuv_420p16→RGBA u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[cfg(feature = "yuv-semi-planar")]
fn check_p16_u8_sse41_rgba_equivalence(width: usize, matrix: KernelMatrix, full_range: bool) {
  let y = p16_plane(width, 37);
  let u = p16_plane(width / 2, 53);
  let v = p16_plane(width / 2, 71);
  let uv = p010_uv_interleave(&u, &v);
  let mut rgba_scalar = std::vec![0u8; width * 4];
  let mut rgba_simd = std::vec![0u8; width * 4];
  scalar::p16_to_rgba_row::<false>(&y, &uv, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    p16_to_rgba_row::<false>(&y, &uv, &mut rgba_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgba_scalar, rgba_simd,
    "SSE4.1 P016→RGBA u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[cfg(feature = "yuv-planar")]
#[test]
fn sse41_yuv420p_n_rgba_matches_scalar_all_bits() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  for m in [
    KernelMatrix::Bt601,
    KernelMatrix::Bt709,
    KernelMatrix::Bt2020Ncl,
    KernelMatrix::Smpte240m,
    KernelMatrix::Fcc,
    KernelMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_planar_u8_sse41_rgba_equivalence_n::<9>(16, m, full);
      check_planar_u8_sse41_rgba_equivalence_n::<10>(16, m, full);
      check_planar_u8_sse41_rgba_equivalence_n::<12>(16, m, full);
      check_planar_u8_sse41_rgba_equivalence_n::<14>(16, m, full);
    }
  }
}

#[cfg(feature = "yuv-planar")]
#[test]
fn sse41_yuv420p_n_rgba_matches_scalar_tail_and_1920() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  for w in [18usize, 30, 34, 1920, 1922] {
    check_planar_u8_sse41_rgba_equivalence_n::<9>(w, KernelMatrix::Bt601, false);
    check_planar_u8_sse41_rgba_equivalence_n::<10>(w, KernelMatrix::Bt709, true);
    check_planar_u8_sse41_rgba_equivalence_n::<12>(w, KernelMatrix::Bt2020Ncl, false);
    check_planar_u8_sse41_rgba_equivalence_n::<14>(w, KernelMatrix::YCgCo, true);
  }
}

#[cfg(feature = "yuv-semi-planar")]
#[test]
fn sse41_pn_rgba_matches_scalar_all_bits() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  for m in [
    KernelMatrix::Bt601,
    KernelMatrix::Bt709,
    KernelMatrix::Bt2020Ncl,
    KernelMatrix::Smpte240m,
    KernelMatrix::Fcc,
    KernelMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_pn_u8_sse41_rgba_equivalence_n::<10>(16, m, full);
      check_pn_u8_sse41_rgba_equivalence_n::<12>(16, m, full);
    }
  }
}

#[cfg(feature = "yuv-semi-planar")]
#[test]
fn sse41_pn_rgba_matches_scalar_tail_and_1920() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  for w in [18usize, 30, 34, 1920, 1922] {
    check_pn_u8_sse41_rgba_equivalence_n::<10>(w, KernelMatrix::Bt601, false);
    check_pn_u8_sse41_rgba_equivalence_n::<12>(w, KernelMatrix::Bt709, true);
  }
}

#[cfg(feature = "yuv-planar")]
#[test]
fn sse41_yuv420p16_rgba_matches_scalar_all_matrices() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  for m in [
    KernelMatrix::Bt601,
    KernelMatrix::Bt709,
    KernelMatrix::Bt2020Ncl,
    KernelMatrix::Smpte240m,
    KernelMatrix::Fcc,
    KernelMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_yuv420p16_u8_sse41_rgba_equivalence(16, m, full);
    }
  }
  for w in [18usize, 30, 34, 1920, 1922] {
    check_yuv420p16_u8_sse41_rgba_equivalence(w, KernelMatrix::Bt709, false);
  }
}

#[cfg(feature = "yuv-semi-planar")]
#[test]
fn sse41_p016_rgba_matches_scalar_all_matrices() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  for m in [
    KernelMatrix::Bt601,
    KernelMatrix::Bt709,
    KernelMatrix::Bt2020Ncl,
    KernelMatrix::Smpte240m,
    KernelMatrix::Fcc,
    KernelMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_p16_u8_sse41_rgba_equivalence(16, m, full);
    }
  }
  for w in [18usize, 30, 34, 1920, 1922] {
    check_p16_u8_sse41_rgba_equivalence(w, KernelMatrix::Bt709, false);
  }
}

// ---- High-bit 4:2:0 native-depth `u16` RGBA equivalence (Ship 8 Tranche 5b) ----
//
// u16 RGBA wrappers share the math of their u16 RGB siblings — only
// the store (and tail dispatch) branches on `ALPHA`, with alpha set to
// `(1 << BITS) - 1` for BITS-generic kernels and `0xFFFF` for 16-bit.

#[cfg(feature = "yuv-planar")]
fn check_planar_u16_sse41_rgba_equivalence_n<const BITS: u32>(
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  let y = planar_n_plane::<BITS>(width, 37);
  let u = planar_n_plane::<BITS>(width / 2, 53);
  let v = planar_n_plane::<BITS>(width / 2, 71);
  let mut rgba_scalar = std::vec![0u16; width * 4];
  let mut rgba_simd = std::vec![0u16; width * 4];
  scalar::yuv_420p_n_to_rgba_u16_row::<BITS, false>(
    &y,
    &u,
    &v,
    &mut rgba_scalar,
    width,
    matrix,
    full_range,
  );
  unsafe {
    yuv_420p_n_to_rgba_u16_row::<BITS, false>(
      &y,
      &u,
      &v,
      &mut rgba_simd,
      width,
      matrix,
      full_range,
    );
  }
  assert_eq!(
    rgba_scalar, rgba_simd,
    "SSE4.1 yuv_420p_n<{BITS}>→RGBA u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[cfg(feature = "yuv-semi-planar")]
fn check_pn_u16_sse41_rgba_equivalence_n<const BITS: u32>(
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  let y = p_n_packed_plane::<BITS>(width, 37);
  let u = p_n_packed_plane::<BITS>(width / 2, 53);
  let v = p_n_packed_plane::<BITS>(width / 2, 71);
  let uv = p010_uv_interleave(&u, &v);
  let mut rgba_scalar = std::vec![0u16; width * 4];
  let mut rgba_simd = std::vec![0u16; width * 4];
  scalar::p_n_to_rgba_u16_row::<BITS, false>(&y, &uv, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    p_n_to_rgba_u16_row::<BITS, false>(&y, &uv, &mut rgba_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgba_scalar, rgba_simd,
    "SSE4.1 Pn<{BITS}>→RGBA u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[cfg(feature = "yuv-planar")]
fn check_yuv420p16_u16_sse41_rgba_equivalence(
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  let y = p16_plane(width, 37);
  let u = p16_plane(width / 2, 53);
  let v = p16_plane(width / 2, 71);
  let mut rgba_scalar = std::vec![0u16; width * 4];
  let mut rgba_simd = std::vec![0u16; width * 4];
  scalar::yuv_420p16_to_rgba_u16_row::<false>(
    &y,
    &u,
    &v,
    &mut rgba_scalar,
    width,
    matrix,
    full_range,
  );
  unsafe {
    yuv_420p16_to_rgba_u16_row::<false>(&y, &u, &v, &mut rgba_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgba_scalar, rgba_simd,
    "SSE4.1 yuv_420p16→RGBA u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[cfg(feature = "yuv-semi-planar")]
fn check_p16_u16_sse41_rgba_equivalence(width: usize, matrix: KernelMatrix, full_range: bool) {
  let y = p16_plane(width, 37);
  let u = p16_plane(width / 2, 53);
  let v = p16_plane(width / 2, 71);
  let uv = p010_uv_interleave(&u, &v);
  let mut rgba_scalar = std::vec![0u16; width * 4];
  let mut rgba_simd = std::vec![0u16; width * 4];
  scalar::p16_to_rgba_u16_row::<false>(&y, &uv, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    p16_to_rgba_u16_row::<false>(&y, &uv, &mut rgba_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgba_scalar, rgba_simd,
    "SSE4.1 P016→RGBA u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[cfg(feature = "yuv-planar")]
#[test]
fn sse41_yuv420p_n_rgba_u16_matches_scalar_all_bits() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  for m in [
    KernelMatrix::Bt601,
    KernelMatrix::Bt709,
    KernelMatrix::Bt2020Ncl,
    KernelMatrix::Smpte240m,
    KernelMatrix::Fcc,
    KernelMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_planar_u16_sse41_rgba_equivalence_n::<9>(16, m, full);
      check_planar_u16_sse41_rgba_equivalence_n::<10>(16, m, full);
      check_planar_u16_sse41_rgba_equivalence_n::<12>(16, m, full);
      check_planar_u16_sse41_rgba_equivalence_n::<14>(16, m, full);
    }
  }
}

#[cfg(feature = "yuv-planar")]
#[test]
fn sse41_yuv420p_n_rgba_u16_matches_scalar_tail_and_1920() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  for w in [18usize, 30, 34, 1920, 1922] {
    check_planar_u16_sse41_rgba_equivalence_n::<9>(w, KernelMatrix::Bt601, false);
    check_planar_u16_sse41_rgba_equivalence_n::<10>(w, KernelMatrix::Bt709, true);
    check_planar_u16_sse41_rgba_equivalence_n::<12>(w, KernelMatrix::Bt2020Ncl, false);
    check_planar_u16_sse41_rgba_equivalence_n::<14>(w, KernelMatrix::YCgCo, true);
  }
}

#[cfg(feature = "yuv-semi-planar")]
#[test]
fn sse41_pn_rgba_u16_matches_scalar_all_bits() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  for m in [
    KernelMatrix::Bt601,
    KernelMatrix::Bt709,
    KernelMatrix::Bt2020Ncl,
    KernelMatrix::Smpte240m,
    KernelMatrix::Fcc,
    KernelMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_pn_u16_sse41_rgba_equivalence_n::<10>(16, m, full);
      check_pn_u16_sse41_rgba_equivalence_n::<12>(16, m, full);
    }
  }
}

#[cfg(feature = "yuv-semi-planar")]
#[test]
fn sse41_pn_rgba_u16_matches_scalar_tail_and_1920() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  for w in [18usize, 30, 34, 1920, 1922] {
    check_pn_u16_sse41_rgba_equivalence_n::<10>(w, KernelMatrix::Bt601, false);
    check_pn_u16_sse41_rgba_equivalence_n::<12>(w, KernelMatrix::Bt709, true);
  }
}

#[cfg(feature = "yuv-planar")]
#[test]
fn sse41_yuv420p16_rgba_u16_matches_scalar_all_matrices() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  for m in [
    KernelMatrix::Bt601,
    KernelMatrix::Bt709,
    KernelMatrix::Bt2020Ncl,
    KernelMatrix::Smpte240m,
    KernelMatrix::Fcc,
    KernelMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_yuv420p16_u16_sse41_rgba_equivalence(16, m, full);
    }
  }
  for w in [18usize, 30, 34, 1920, 1922] {
    check_yuv420p16_u16_sse41_rgba_equivalence(w, KernelMatrix::Bt709, false);
  }
}

#[cfg(feature = "yuv-semi-planar")]
#[test]
fn sse41_p016_rgba_u16_matches_scalar_all_matrices() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  for m in [
    KernelMatrix::Bt601,
    KernelMatrix::Bt709,
    KernelMatrix::Bt2020Ncl,
    KernelMatrix::Smpte240m,
    KernelMatrix::Fcc,
    KernelMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_p16_u16_sse41_rgba_equivalence(16, m, full);
    }
  }
  for w in [18usize, 30, 34, 1920, 1922] {
    check_p16_u16_sse41_rgba_equivalence(w, KernelMatrix::Bt709, false);
  }
}

// ---- Pn 4:4:4 (P410 / P412 / P416) SSE4.1 equivalence ---------------

#[cfg(feature = "yuv-semi-planar")]
fn check_p_n_444_u8_sse41_equivalence<const BITS: u32>(
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  let y = high_bit_plane_sse41::<BITS>(width, 37);
  let u = high_bit_plane_sse41::<BITS>(width, 53);
  let v = high_bit_plane_sse41::<BITS>(width, 71);
  let uv = interleave_uv_sse41(&u, &v);
  let mut rgb_scalar = std::vec![0u8; width * 3];
  let mut rgb_simd = std::vec![0u8; width * 3];
  scalar::p_n_444_to_rgb_row::<BITS, false>(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    p_n_444_to_rgb_row::<BITS, false>(&y, &uv, &mut rgb_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_simd,
    "SSE4.1 Pn4:4:4 {BITS}-bit → u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[cfg(feature = "yuv-semi-planar")]
fn check_p_n_444_u16_sse41_equivalence<const BITS: u32>(
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  let y = high_bit_plane_sse41::<BITS>(width, 37);
  let u = high_bit_plane_sse41::<BITS>(width, 53);
  let v = high_bit_plane_sse41::<BITS>(width, 71);
  let uv = interleave_uv_sse41(&u, &v);
  let mut rgb_scalar = std::vec![0u16; width * 3];
  let mut rgb_simd = std::vec![0u16; width * 3];
  scalar::p_n_444_to_rgb_u16_row::<BITS, false>(
    &y,
    &uv,
    &mut rgb_scalar,
    width,
    matrix,
    full_range,
  );
  unsafe {
    p_n_444_to_rgb_u16_row::<BITS, false>(&y, &uv, &mut rgb_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_simd,
    "SSE4.1 Pn4:4:4 {BITS}-bit → u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[cfg(feature = "yuv-semi-planar")]
fn check_p_n_444_16_u8_sse41_equivalence(width: usize, matrix: KernelMatrix, full_range: bool) {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  let y = p16_plane(width, 37);
  let u = p16_plane(width, 53);
  let v = p16_plane(width, 71);
  let uv = interleave_uv_sse41(&u, &v);
  let mut rgb_scalar = std::vec![0u8; width * 3];
  let mut rgb_simd = std::vec![0u8; width * 3];
  scalar::p_n_444_16_to_rgb_row::<false>(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    p_n_444_16_to_rgb_row::<false>(&y, &uv, &mut rgb_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_simd,
    "SSE4.1 P416 → u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[cfg(feature = "yuv-semi-planar")]
fn check_p_n_444_16_u16_sse41_equivalence(width: usize, matrix: KernelMatrix, full_range: bool) {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  let y = p16_plane(width, 37);
  let u = p16_plane(width, 53);
  let v = p16_plane(width, 71);
  let uv = interleave_uv_sse41(&u, &v);
  let mut rgb_scalar = std::vec![0u16; width * 3];
  let mut rgb_simd = std::vec![0u16; width * 3];
  scalar::p_n_444_16_to_rgb_u16_row::<false>(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    p_n_444_16_to_rgb_u16_row::<false>(&y, &uv, &mut rgb_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_simd,
    "SSE4.1 P416 → u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[cfg(feature = "yuv-semi-planar")]
#[test]
fn sse41_p410_matches_scalar_all_matrices() {
  for m in [
    KernelMatrix::Bt601,
    KernelMatrix::Bt709,
    KernelMatrix::Bt2020Ncl,
    KernelMatrix::Smpte240m,
    KernelMatrix::Fcc,
    KernelMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_p_n_444_u8_sse41_equivalence::<10>(16, m, full);
      check_p_n_444_u16_sse41_equivalence::<10>(16, m, full);
    }
  }
}

#[cfg(feature = "yuv-semi-planar")]
#[test]
fn sse41_p412_matches_scalar_all_matrices() {
  for m in [KernelMatrix::Bt709, KernelMatrix::Bt2020Ncl] {
    for full in [true, false] {
      check_p_n_444_u8_sse41_equivalence::<12>(16, m, full);
      check_p_n_444_u16_sse41_equivalence::<12>(16, m, full);
    }
  }
}

#[cfg(feature = "yuv-semi-planar")]
#[test]
fn sse41_p410_p412_matches_scalar_tail_widths() {
  for w in [1usize, 3, 7, 15, 17, 31, 33, 1920, 1921] {
    check_p_n_444_u8_sse41_equivalence::<10>(w, KernelMatrix::Bt601, false);
    check_p_n_444_u16_sse41_equivalence::<10>(w, KernelMatrix::Bt709, true);
    check_p_n_444_u8_sse41_equivalence::<12>(w, KernelMatrix::Bt2020Ncl, false);
  }
}

// ---- NV20 low-bit-packed 4:4:4 (LOW_PACKED = true) dirty-bit parity -------
//
// ★ The low-packed de-pack masks (`& 0x03FF`) the LOW 10 bits and discards
// STRAY garbage in the top 6; feed garbage in the top 6 and assert every SSE4.1
// lane == scalar, so the mask (`depack_u16x8::<true>`) sanitizes identically.

/// Low-bit-packed plane: active 10-bit value in the LOW bits, STRAY non-zero
/// garbage in the top 6 (the de-pack must discard it).
#[cfg(feature = "yuv-semi-planar")]
fn nv20_dirty_plane_sse41(n: usize, seed: usize) -> std::vec::Vec<u16> {
  (0..n)
    .map(|i| {
      let logical = ((i.wrapping_mul(seed).wrapping_add(seed * 3)) as u16) & 0x03FF;
      let stray = (((i as u16).wrapping_mul(37).wrapping_add(1)) & 0x3F) | 1;
      logical | (stray << 10)
    })
    .collect()
}

#[cfg(feature = "yuv-semi-planar")]
fn check_nv20_444_lowpacked_sse41_equivalence(
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  let y = nv20_dirty_plane_sse41(width, 37);
  let u = nv20_dirty_plane_sse41(width, 53);
  let v = nv20_dirty_plane_sse41(width, 71);
  let uv = interleave_uv_sse41(&u, &v);

  let mut rgb_scalar = std::vec![0u8; width * 3];
  let mut rgb_simd = std::vec![0u8; width * 3];
  scalar::p_n_444_to_rgb_or_rgba_row::<10, false, false, true>(
    &y,
    &uv,
    &mut rgb_scalar,
    width,
    matrix,
    full_range,
  );
  unsafe {
    p_n_444_to_rgb_or_rgba_row::<10, false, false, true>(
      &y,
      &uv,
      &mut rgb_simd,
      width,
      matrix,
      full_range,
    );
  }
  assert_eq!(
    rgb_scalar, rgb_simd,
    "SSE4.1 NV20 4:4:4 low-packed → u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );

  let mut u16_scalar = std::vec![0u16; width * 3];
  let mut u16_simd = std::vec![0u16; width * 3];
  scalar::p_n_444_to_rgb_or_rgba_u16_row::<10, false, false, true>(
    &y,
    &uv,
    &mut u16_scalar,
    width,
    matrix,
    full_range,
  );
  unsafe {
    p_n_444_to_rgb_or_rgba_u16_row::<10, false, false, true>(
      &y,
      &uv,
      &mut u16_simd,
      width,
      matrix,
      full_range,
    );
  }
  assert_eq!(
    u16_scalar, u16_simd,
    "SSE4.1 NV20 4:4:4 low-packed → u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[cfg(feature = "yuv-semi-planar")]
#[test]
fn sse41_nv20_444_lowpacked_matches_scalar_dirty_widths() {
  for w in [1usize, 3, 7, 8, 15, 16, 17, 31, 32, 33, 48, 1920, 1921] {
    for m in [
      KernelMatrix::Bt601,
      KernelMatrix::Bt709,
      KernelMatrix::Bt2020Ncl,
    ] {
      for full in [true, false] {
        check_nv20_444_lowpacked_sse41_equivalence(w, m, full);
      }
    }
  }
}

#[cfg(feature = "yuv-semi-planar")]
#[test]
fn sse41_p416_matches_scalar_all_matrices() {
  for m in [
    KernelMatrix::Bt601,
    KernelMatrix::Bt709,
    KernelMatrix::Bt2020Ncl,
    KernelMatrix::Smpte240m,
    KernelMatrix::Fcc,
    KernelMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_p_n_444_16_u8_sse41_equivalence(16, m, full);
      check_p_n_444_16_u16_sse41_equivalence(16, m, full);
    }
  }
}

#[cfg(feature = "yuv-semi-planar")]
#[test]
fn sse41_p416_matches_scalar_tail_widths() {
  for w in [1usize, 3, 7, 8, 9, 15, 16, 17, 31, 33, 1920, 1921] {
    check_p_n_444_16_u8_sse41_equivalence(w, KernelMatrix::Bt709, false);
    check_p_n_444_16_u16_sse41_equivalence(w, KernelMatrix::Bt2020Ncl, true);
  }
}

// High-bit 4:4:4 u8 RGBA equivalence.
//
// Mirrors the 4:2:0 RGBA pattern. Each kernel family — Yuv444p_n
// (BITS-generic), Yuv444p16, Pn_444 (BITS-generic), Pn_444_16 — has its
// SSE4.1 RGBA kernel byte-pinned against the scalar reference at the
// natural width and a sweep of tail widths. Each test gates on
// `is_x86_feature_detected!("sse4.1")` to stay clean under
// sanitizer/Miri/non-feature-flagged CI runners.

#[cfg(feature = "yuv-planar")]
fn check_yuv444p_n_u8_sse41_rgba_equivalence<const BITS: u32>(
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  let y = planar_n_plane::<BITS>(width, 37);
  let u = planar_n_plane::<BITS>(width, 53);
  let v = planar_n_plane::<BITS>(width, 71);
  let mut rgba_scalar = std::vec![0u8; width * 4];
  let mut rgba_simd = std::vec![0u8; width * 4];
  scalar::yuv_444p_n_to_rgba_row::<BITS, false>(
    &y,
    &u,
    &v,
    &mut rgba_scalar,
    width,
    matrix,
    full_range,
  );
  unsafe {
    yuv_444p_n_to_rgba_row::<BITS, false>(&y, &u, &v, &mut rgba_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgba_scalar, rgba_simd,
    "SSE4.1 Yuv444p<{BITS}> → RGBA u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[cfg(feature = "yuv-semi-planar")]
fn check_pn_444_u8_sse41_rgba_equivalence<const BITS: u32>(
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  let y = high_bit_plane_sse41::<BITS>(width, 37);
  let u = high_bit_plane_sse41::<BITS>(width, 53);
  let v = high_bit_plane_sse41::<BITS>(width, 71);
  let uv = interleave_uv_sse41(&u, &v);
  let mut rgba_scalar = std::vec![0u8; width * 4];
  let mut rgba_simd = std::vec![0u8; width * 4];
  scalar::p_n_444_to_rgba_row::<BITS, false>(&y, &uv, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    p_n_444_to_rgba_row::<BITS, false>(&y, &uv, &mut rgba_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgba_scalar, rgba_simd,
    "SSE4.1 Pn4:4:4<{BITS}> → RGBA u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[cfg(feature = "yuv-planar")]
fn check_yuv444p16_u8_sse41_rgba_equivalence(width: usize, matrix: KernelMatrix, full_range: bool) {
  let y = p16_plane(width, 37);
  let u = p16_plane(width, 53);
  let v = p16_plane(width, 71);
  let mut rgba_scalar = std::vec![0u8; width * 4];
  let mut rgba_simd = std::vec![0u8; width * 4];
  scalar::yuv_444p16_to_rgba_row::<false>(&y, &u, &v, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    yuv_444p16_to_rgba_row::<false>(&y, &u, &v, &mut rgba_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgba_scalar, rgba_simd,
    "SSE4.1 Yuv444p16 → RGBA u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[cfg(feature = "yuv-semi-planar")]
fn check_p_n_444_16_u8_sse41_rgba_equivalence(
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
) {
  let y = p16_plane(width, 37);
  let u = p16_plane(width, 53);
  let v = p16_plane(width, 71);
  let uv = interleave_uv_sse41(&u, &v);
  let mut rgba_scalar = std::vec![0u8; width * 4];
  let mut rgba_simd = std::vec![0u8; width * 4];
  scalar::p_n_444_16_to_rgba_row::<false>(&y, &uv, &mut rgba_scalar, width, matrix, full_range);
  unsafe {
    p_n_444_16_to_rgba_row::<false>(&y, &uv, &mut rgba_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgba_scalar, rgba_simd,
    "SSE4.1 P416 → RGBA u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[cfg(feature = "yuv-planar")]
#[test]
fn sse41_yuv444p_n_rgba_matches_scalar_all_bits() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  for m in [
    KernelMatrix::Bt601,
    KernelMatrix::Bt709,
    KernelMatrix::Bt2020Ncl,
    KernelMatrix::Smpte240m,
    KernelMatrix::Fcc,
    KernelMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_yuv444p_n_u8_sse41_rgba_equivalence::<9>(16, m, full);
      check_yuv444p_n_u8_sse41_rgba_equivalence::<10>(16, m, full);
      check_yuv444p_n_u8_sse41_rgba_equivalence::<12>(16, m, full);
      check_yuv444p_n_u8_sse41_rgba_equivalence::<14>(16, m, full);
    }
  }
}

#[cfg(feature = "yuv-planar")]
#[test]
fn sse41_yuv444p_n_rgba_matches_scalar_tail_and_widths() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  for w in [17usize, 31, 47, 63, 1920, 1922] {
    check_yuv444p_n_u8_sse41_rgba_equivalence::<9>(w, KernelMatrix::Bt601, false);
    check_yuv444p_n_u8_sse41_rgba_equivalence::<10>(w, KernelMatrix::Bt709, true);
    check_yuv444p_n_u8_sse41_rgba_equivalence::<12>(w, KernelMatrix::Bt2020Ncl, false);
    check_yuv444p_n_u8_sse41_rgba_equivalence::<14>(w, KernelMatrix::YCgCo, true);
  }
}

#[cfg(feature = "yuv-semi-planar")]
#[test]
fn sse41_pn_444_rgba_matches_scalar_all_bits() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  for m in [
    KernelMatrix::Bt601,
    KernelMatrix::Bt709,
    KernelMatrix::Bt2020Ncl,
    KernelMatrix::Smpte240m,
    KernelMatrix::Fcc,
    KernelMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_pn_444_u8_sse41_rgba_equivalence::<10>(16, m, full);
      check_pn_444_u8_sse41_rgba_equivalence::<12>(16, m, full);
    }
  }
}

#[cfg(feature = "yuv-semi-planar")]
#[test]
fn sse41_pn_444_rgba_matches_scalar_tail_and_widths() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  for w in [17usize, 31, 47, 63, 1920, 1922] {
    check_pn_444_u8_sse41_rgba_equivalence::<10>(w, KernelMatrix::Bt601, false);
    check_pn_444_u8_sse41_rgba_equivalence::<12>(w, KernelMatrix::Bt709, true);
  }
}

#[cfg(feature = "yuv-planar")]
#[test]
fn sse41_yuv444p16_rgba_matches_scalar_all_matrices() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  for m in [
    KernelMatrix::Bt601,
    KernelMatrix::Bt709,
    KernelMatrix::Bt2020Ncl,
    KernelMatrix::Smpte240m,
    KernelMatrix::Fcc,
    KernelMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_yuv444p16_u8_sse41_rgba_equivalence(16, m, full);
    }
  }
  for w in [17usize, 31, 47, 63, 1920, 1922] {
    check_yuv444p16_u8_sse41_rgba_equivalence(w, KernelMatrix::Bt709, false);
  }
}

#[cfg(feature = "yuva")]
fn check_yuv444p16_u8_sse41_rgba_with_alpha_src_equivalence(
  width: usize,
  matrix: KernelMatrix,
  full_range: bool,
  alpha_seed: usize,
) {
  let y = p16_plane(width, 37);
  let u = p16_plane(width, 53);
  let v = p16_plane(width, 71);
  let a_src = p16_plane(width, alpha_seed);
  let mut rgba_scalar = std::vec![0u8; width * 4];
  let mut rgba_simd = std::vec![0u8; width * 4];
  scalar::yuv_444p16_to_rgba_with_alpha_src_row::<false>(
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
    yuv_444p16_to_rgba_with_alpha_src_row::<false>(
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
    "SSE4.1 Yuva444p16 → RGBA u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range}, alpha_seed={alpha_seed})"
  );
}

#[cfg(feature = "yuva")]
#[test]
fn sse41_yuva444p16_rgba_matches_scalar_all_matrices() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  for m in [
    KernelMatrix::Bt601,
    KernelMatrix::Bt709,
    KernelMatrix::Bt2020Ncl,
    KernelMatrix::Smpte240m,
    KernelMatrix::Fcc,
    KernelMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_yuv444p16_u8_sse41_rgba_with_alpha_src_equivalence(16, m, full, 89);
    }
  }
}

#[cfg(feature = "yuva")]
#[test]
fn sse41_yuva444p16_rgba_matches_scalar_widths_and_alpha() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  for w in [16usize, 17, 31, 47, 63, 1920, 1922] {
    check_yuv444p16_u8_sse41_rgba_with_alpha_src_equivalence(w, KernelMatrix::Bt709, true, 89);
  }
  for seed in [13usize, 41, 127, 211] {
    check_yuv444p16_u8_sse41_rgba_with_alpha_src_equivalence(16, KernelMatrix::Bt601, false, seed);
  }
}

#[cfg(feature = "yuv-semi-planar")]
#[test]
fn sse41_p416_rgba_matches_scalar_all_matrices() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  for m in [
    KernelMatrix::Bt601,
    KernelMatrix::Bt709,
    KernelMatrix::Bt2020Ncl,
    KernelMatrix::Smpte240m,
    KernelMatrix::Fcc,
    KernelMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_p_n_444_16_u8_sse41_rgba_equivalence(16, m, full);
    }
  }
  for w in [17usize, 31, 47, 63, 1920, 1922] {
    check_p_n_444_16_u8_sse41_rgba_equivalence(w, KernelMatrix::Bt709, false);
  }
}
