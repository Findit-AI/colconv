use super::{super::*, p16_plane_avx2};
// Planar high-bit plane builder — used only by the `yuv-planar`-gated
// planar-vs-NEON equivalence helpers below.
#[cfg(feature = "yuv-planar")]
use super::planar_n_plane;
// Semi-planar (`p_n`) plane builders — used only by the
// `yuv-semi-planar`-gated equivalence helpers / tests below.
#[cfg(feature = "yuv-semi-planar")]
use super::{p_n_packed_plane, p010_uv_interleave};

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

// ---- rgb_to_luma_row equivalence -------------------------------------

fn check_luma_equivalence(rgb: &[u8], width: usize, matrix: KernelMatrix, full_range: bool) {
  let mut y_s = std::vec![0u8; width];
  let mut y_k = std::vec![0u8; width];
  scalar::rgb_to_luma_row(rgb, &mut y_s, width, matrix, full_range);
  unsafe {
    rgb_to_luma_row(rgb, &mut y_k, width, matrix, full_range);
  }
  assert_eq!(
    y_s, y_k,
    "AVX2 rgb_to_luma diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[test]
#[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
fn avx2_rgb_to_luma_row_matches_scalar_widths() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  let rgb: std::vec::Vec<u8> = (0..1921 * 3)
    .map(|i| ((i * 37 + 11) & 0xFF) as u8)
    .collect();
  for &matrix in &[
    KernelMatrix::Bt601,
    KernelMatrix::Fcc,
    KernelMatrix::Bt709,
    KernelMatrix::Bt2020Ncl,
    KernelMatrix::Smpte240m,
    KernelMatrix::YCgCo,
  ] {
    for full_range in [true, false] {
      for &w in &[1usize, 7, 16, 31, 32, 33, 47, 63, 64, 65, 128, 130] {
        check_luma_equivalence(&rgb[..w * 3], w, matrix, full_range);
      }
    }
  }
}

// ---- yuv420p10 AVX2 scalar-equivalence ------------------------------

#[cfg(feature = "yuv-planar")]
fn p10_plane(n: usize, seed: usize) -> std::vec::Vec<u16> {
  (0..n)
    .map(|i| ((i * seed + seed * 3) & 0x3FF) as u16)
    .collect()
}

#[cfg(feature = "yuv-planar")]
fn check_p10_u8_avx2_equivalence(width: usize, matrix: KernelMatrix, full_range: bool) {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  let y = p10_plane(width, 37);
  let u = p10_plane(width / 2, 53);
  let v = p10_plane(width / 2, 71);
  let mut rgb_scalar = std::vec![0u8; width * 3];
  let mut rgb_simd = std::vec![0u8; width * 3];

  scalar::yuv_420p_n_to_rgb_row::<10, false>(
    &y,
    &u,
    &v,
    &mut rgb_scalar,
    width,
    matrix,
    full_range,
  );
  unsafe {
    yuv_420p_n_to_rgb_row::<10, false>(&y, &u, &v, &mut rgb_simd, width, matrix, full_range);
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

#[cfg(feature = "yuv-planar")]
fn check_p10_u16_avx2_equivalence(width: usize, matrix: KernelMatrix, full_range: bool) {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  let y = p10_plane(width, 37);
  let u = p10_plane(width / 2, 53);
  let v = p10_plane(width / 2, 71);
  let mut rgb_scalar = std::vec![0u16; width * 3];
  let mut rgb_simd = std::vec![0u16; width * 3];

  scalar::yuv_420p_n_to_rgb_u16_row::<10, false>(
    &y,
    &u,
    &v,
    &mut rgb_scalar,
    width,
    matrix,
    full_range,
  );
  unsafe {
    yuv_420p_n_to_rgb_u16_row::<10, false>(&y, &u, &v, &mut rgb_simd, width, matrix, full_range);
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

#[cfg(feature = "yuv-planar")]
#[test]
fn avx2_p10_u8_matches_scalar_all_matrices() {
  for m in [
    KernelMatrix::Bt601,
    KernelMatrix::Bt709,
    KernelMatrix::Bt2020Ncl,
    KernelMatrix::Smpte240m,
    KernelMatrix::Fcc,
    KernelMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_p10_u8_avx2_equivalence(32, m, full);
    }
  }
}

#[cfg(feature = "yuv-planar")]
#[test]
fn avx2_p10_u16_matches_scalar_all_matrices() {
  for m in [
    KernelMatrix::Bt601,
    KernelMatrix::Bt709,
    KernelMatrix::Bt2020Ncl,
    KernelMatrix::Smpte240m,
    KernelMatrix::Fcc,
    KernelMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_p10_u16_avx2_equivalence(32, m, full);
    }
  }
}

#[cfg(feature = "yuv-planar")]
#[test]
fn avx2_p10_matches_scalar_odd_tail_widths() {
  for w in [34usize, 62, 66, 1922] {
    check_p10_u8_avx2_equivalence(w, KernelMatrix::Bt601, false);
    check_p10_u16_avx2_equivalence(w, KernelMatrix::Bt709, true);
  }
}

#[cfg(feature = "yuv-planar")]
#[test]
fn avx2_p10_matches_scalar_1920() {
  check_p10_u8_avx2_equivalence(1920, KernelMatrix::Bt709, false);
  check_p10_u16_avx2_equivalence(1920, KernelMatrix::Bt2020Ncl, false);
}

// ---- yuv420p_n<BITS> AVX2 scalar-equivalence (BITS=9 coverage) ------

#[cfg(feature = "yuv-planar")]
fn p_n_plane_avx2<const BITS: u32>(n: usize, seed: usize) -> std::vec::Vec<u16> {
  let mask = ((1u32 << BITS) - 1) as u16;
  (0..n)
    .map(|i| ((i.wrapping_mul(seed).wrapping_add(seed * 3)) as u16) & mask)
    .collect()
}

#[cfg(feature = "yuv-planar")]
fn check_p_n_u8_avx2_equivalence<const BITS: u32>(
  width: usize,
  matrix: KernelMatrix,
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
  scalar::yuv_420p_n_to_rgb_row::<BITS, false>(
    &y,
    &u,
    &v,
    &mut rgb_scalar,
    width,
    matrix,
    full_range,
  );
  unsafe {
    yuv_420p_n_to_rgb_row::<BITS, false>(&y, &u, &v, &mut rgb_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_simd,
    "AVX2 yuv_420p_n<{BITS}>→u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[cfg(feature = "yuv-planar")]
fn check_p_n_u16_avx2_equivalence<const BITS: u32>(
  width: usize,
  matrix: KernelMatrix,
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
  scalar::yuv_420p_n_to_rgb_u16_row::<BITS, false>(
    &y,
    &u,
    &v,
    &mut rgb_scalar,
    width,
    matrix,
    full_range,
  );
  unsafe {
    yuv_420p_n_to_rgb_u16_row::<BITS, false>(&y, &u, &v, &mut rgb_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_simd,
    "AVX2 yuv_420p_n<{BITS}>→u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[cfg(feature = "yuv-planar")]
#[test]
fn avx2_yuv420p9_matches_scalar_all_matrices_and_ranges() {
  for m in [
    KernelMatrix::Bt601,
    KernelMatrix::Bt709,
    KernelMatrix::Bt2020Ncl,
    KernelMatrix::Smpte240m,
    KernelMatrix::Fcc,
    KernelMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_p_n_u8_avx2_equivalence::<9>(32, m, full);
      check_p_n_u16_avx2_equivalence::<9>(32, m, full);
    }
  }
}

#[cfg(feature = "yuv-planar")]
#[test]
fn avx2_yuv420p9_matches_scalar_tail_and_large_widths() {
  // AVX2 main loop is 32 px; widths chosen to stress tail handling
  // both below and above the SIMD lane width.
  for w in [18usize, 30, 34, 62, 66, 1922] {
    check_p_n_u8_avx2_equivalence::<9>(w, KernelMatrix::Bt601, false);
    check_p_n_u16_avx2_equivalence::<9>(w, KernelMatrix::Bt709, true);
  }
  check_p_n_u8_avx2_equivalence::<9>(1920, KernelMatrix::Bt709, false);
  check_p_n_u16_avx2_equivalence::<9>(1920, KernelMatrix::Bt2020Ncl, false);
}

// ---- P010 AVX2 scalar-equivalence -----------------------------------

#[cfg(feature = "yuv-semi-planar")]
fn p010_plane(n: usize, seed: usize) -> std::vec::Vec<u16> {
  (0..n)
    .map(|i| (((i * seed + seed * 3) & 0x3FF) as u16) << 6)
    .collect()
}

#[cfg(feature = "yuv-semi-planar")]
fn check_p010_u8_avx2_equivalence(width: usize, matrix: KernelMatrix, full_range: bool) {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  let y = p010_plane(width, 37);
  let u = p010_plane(width / 2, 53);
  let v = p010_plane(width / 2, 71);
  let uv = p010_uv_interleave(&u, &v);
  let mut rgb_scalar = std::vec![0u8; width * 3];
  let mut rgb_simd = std::vec![0u8; width * 3];
  scalar::p_n_to_rgb_row::<10, false>(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    p_n_to_rgb_row::<10, false>(&y, &uv, &mut rgb_simd, width, matrix, full_range);
  }
  assert_eq!(rgb_scalar, rgb_simd, "AVX2 P010→u8 diverges");
}

#[cfg(feature = "yuv-semi-planar")]
fn check_p010_u16_avx2_equivalence(width: usize, matrix: KernelMatrix, full_range: bool) {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  let y = p010_plane(width, 37);
  let u = p010_plane(width / 2, 53);
  let v = p010_plane(width / 2, 71);
  let uv = p010_uv_interleave(&u, &v);
  let mut rgb_scalar = std::vec![0u16; width * 3];
  let mut rgb_simd = std::vec![0u16; width * 3];
  scalar::p_n_to_rgb_u16_row::<10, false>(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    p_n_to_rgb_u16_row::<10, false>(&y, &uv, &mut rgb_simd, width, matrix, full_range);
  }
  assert_eq!(rgb_scalar, rgb_simd, "AVX2 P010→u16 diverges");
}

#[cfg(feature = "yuv-semi-planar")]
#[test]
fn avx2_p010_u8_matches_scalar_all_matrices() {
  for m in [
    KernelMatrix::Bt601,
    KernelMatrix::Bt709,
    KernelMatrix::Bt2020Ncl,
    KernelMatrix::Smpte240m,
    KernelMatrix::Fcc,
    KernelMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_p010_u8_avx2_equivalence(32, m, full);
    }
  }
}

#[cfg(feature = "yuv-semi-planar")]
#[test]
fn avx2_p010_u16_matches_scalar_all_matrices() {
  for m in [
    KernelMatrix::Bt601,
    KernelMatrix::Bt709,
    KernelMatrix::Bt2020Ncl,
    KernelMatrix::Smpte240m,
    KernelMatrix::Fcc,
    KernelMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_p010_u16_avx2_equivalence(32, m, full);
    }
  }
}

#[cfg(feature = "yuv-semi-planar")]
#[test]
fn avx2_p010_matches_scalar_odd_tail_widths() {
  for w in [34usize, 62, 66, 1922] {
    check_p010_u8_avx2_equivalence(w, KernelMatrix::Bt601, false);
    check_p010_u16_avx2_equivalence(w, KernelMatrix::Bt709, true);
  }
}

#[cfg(feature = "yuv-semi-planar")]
#[test]
fn avx2_p010_matches_scalar_1920() {
  check_p010_u8_avx2_equivalence(1920, KernelMatrix::Bt709, false);
  check_p010_u16_avx2_equivalence(1920, KernelMatrix::Bt2020Ncl, false);
}

// ---- Generic BITS equivalence (12/14-bit coverage) ------------------

#[cfg(feature = "yuv-planar")]
fn check_planar_u8_avx2_equivalence_n<const BITS: u32>(
  width: usize,
  matrix: KernelMatrix,
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
  scalar::yuv_420p_n_to_rgb_row::<BITS, false>(
    &y,
    &u,
    &v,
    &mut rgb_scalar,
    width,
    matrix,
    full_range,
  );
  unsafe {
    yuv_420p_n_to_rgb_row::<BITS, false>(&y, &u, &v, &mut rgb_simd, width, matrix, full_range);
  }
  assert_eq!(rgb_scalar, rgb_simd, "AVX2 planar {BITS}-bit → u8 diverges");
}

#[cfg(feature = "yuv-planar")]
fn check_planar_u16_avx2_equivalence_n<const BITS: u32>(
  width: usize,
  matrix: KernelMatrix,
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
  scalar::yuv_420p_n_to_rgb_u16_row::<BITS, false>(
    &y,
    &u,
    &v,
    &mut rgb_scalar,
    width,
    matrix,
    full_range,
  );
  unsafe {
    yuv_420p_n_to_rgb_u16_row::<BITS, false>(&y, &u, &v, &mut rgb_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_simd,
    "AVX2 planar {BITS}-bit → u16 diverges"
  );
}

#[cfg(feature = "yuv-semi-planar")]
fn check_pn_u8_avx2_equivalence_n<const BITS: u32>(
  width: usize,
  matrix: KernelMatrix,
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
  scalar::p_n_to_rgb_row::<BITS, false>(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    p_n_to_rgb_row::<BITS, false>(&y, &uv, &mut rgb_simd, width, matrix, full_range);
  }
  assert_eq!(rgb_scalar, rgb_simd, "AVX2 Pn {BITS}-bit → u8 diverges");
}

#[cfg(feature = "yuv-semi-planar")]
fn check_pn_u16_avx2_equivalence_n<const BITS: u32>(
  width: usize,
  matrix: KernelMatrix,
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
  scalar::p_n_to_rgb_u16_row::<BITS, false>(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    p_n_to_rgb_u16_row::<BITS, false>(&y, &uv, &mut rgb_simd, width, matrix, full_range);
  }
  assert_eq!(rgb_scalar, rgb_simd, "AVX2 Pn {BITS}-bit → u16 diverges");
}

#[test]
fn avx2_p12_matches_scalar_all_matrices() {
  for m in [
    KernelMatrix::Bt601,
    KernelMatrix::Bt709,
    KernelMatrix::Bt2020Ncl,
    KernelMatrix::Smpte240m,
    KernelMatrix::Fcc,
    KernelMatrix::YCgCo,
  ] {
    for full in [true, false] {
      #[cfg(feature = "yuv-planar")]
      check_planar_u8_avx2_equivalence_n::<12>(32, m, full);
      #[cfg(feature = "yuv-planar")]
      check_planar_u16_avx2_equivalence_n::<12>(32, m, full);
      #[cfg(feature = "yuv-semi-planar")]
      check_pn_u8_avx2_equivalence_n::<12>(32, m, full);
      #[cfg(feature = "yuv-semi-planar")]
      check_pn_u16_avx2_equivalence_n::<12>(32, m, full);
    }
  }
}

#[cfg(feature = "yuv-planar")]
#[test]
fn avx2_p14_matches_scalar_all_matrices() {
  for m in [
    KernelMatrix::Bt601,
    KernelMatrix::Bt709,
    KernelMatrix::Bt2020Ncl,
    KernelMatrix::Smpte240m,
    KernelMatrix::Fcc,
    KernelMatrix::YCgCo,
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
    #[cfg(feature = "yuv-planar")]
    check_planar_u8_avx2_equivalence_n::<12>(w, KernelMatrix::Bt601, false);
    #[cfg(feature = "yuv-planar")]
    check_planar_u16_avx2_equivalence_n::<12>(w, KernelMatrix::Bt709, true);
    #[cfg(feature = "yuv-semi-planar")]
    check_pn_u8_avx2_equivalence_n::<12>(w, KernelMatrix::Bt601, false);
    #[cfg(feature = "yuv-semi-planar")]
    check_pn_u16_avx2_equivalence_n::<12>(w, KernelMatrix::Bt2020Ncl, false);
  }
}

#[cfg(feature = "yuv-planar")]
#[test]
fn avx2_p14_matches_scalar_tail_widths() {
  for w in [34usize, 62, 66, 1922] {
    check_planar_u8_avx2_equivalence_n::<14>(w, KernelMatrix::Bt601, false);
    check_planar_u16_avx2_equivalence_n::<14>(w, KernelMatrix::Bt709, true);
  }
}

// ---- 16-bit (full-range u16 samples) AVX2 equivalence ---------------

#[cfg(feature = "yuv-planar")]
fn check_yuv420p16_u8_avx2_equivalence(width: usize, matrix: KernelMatrix, full_range: bool) {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  let y = p16_plane_avx2(width, 37);
  let u = p16_plane_avx2(width / 2, 53);
  let v = p16_plane_avx2(width / 2, 71);
  let mut rgb_scalar = std::vec![0u8; width * 3];
  let mut rgb_simd = std::vec![0u8; width * 3];
  scalar::yuv_420p16_to_rgb_row::<false>(&y, &u, &v, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    yuv_420p16_to_rgb_row::<false>(&y, &u, &v, &mut rgb_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_simd,
    "AVX2 yuv420p16→u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[cfg(feature = "yuv-planar")]
fn check_yuv420p16_u16_avx2_equivalence(width: usize, matrix: KernelMatrix, full_range: bool) {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  let y = p16_plane_avx2(width, 37);
  let u = p16_plane_avx2(width / 2, 53);
  let v = p16_plane_avx2(width / 2, 71);
  let mut rgb_scalar = std::vec![0u16; width * 3];
  let mut rgb_simd = std::vec![0u16; width * 3];
  scalar::yuv_420p16_to_rgb_u16_row::<false>(
    &y,
    &u,
    &v,
    &mut rgb_scalar,
    width,
    matrix,
    full_range,
  );
  unsafe {
    yuv_420p16_to_rgb_u16_row::<false>(&y, &u, &v, &mut rgb_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_simd,
    "AVX2 yuv420p16→u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[cfg(feature = "yuv-semi-planar")]
fn check_p16_u8_avx2_equivalence(width: usize, matrix: KernelMatrix, full_range: bool) {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  let y = p16_plane_avx2(width, 37);
  let u = p16_plane_avx2(width / 2, 53);
  let v = p16_plane_avx2(width / 2, 71);
  let uv = p010_uv_interleave(&u, &v);
  let mut rgb_scalar = std::vec![0u8; width * 3];
  let mut rgb_simd = std::vec![0u8; width * 3];
  scalar::p16_to_rgb_row::<false>(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    p16_to_rgb_row::<false>(&y, &uv, &mut rgb_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_simd,
    "AVX2 p016→u8 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[cfg(feature = "yuv-semi-planar")]
fn check_p16_u16_avx2_equivalence(width: usize, matrix: KernelMatrix, full_range: bool) {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  let y = p16_plane_avx2(width, 37);
  let u = p16_plane_avx2(width / 2, 53);
  let v = p16_plane_avx2(width / 2, 71);
  let uv = p010_uv_interleave(&u, &v);
  let mut rgb_scalar = std::vec![0u16; width * 3];
  let mut rgb_simd = std::vec![0u16; width * 3];
  scalar::p16_to_rgb_u16_row::<false>(&y, &uv, &mut rgb_scalar, width, matrix, full_range);
  unsafe {
    p16_to_rgb_u16_row::<false>(&y, &uv, &mut rgb_simd, width, matrix, full_range);
  }
  assert_eq!(
    rgb_scalar, rgb_simd,
    "AVX2 p016→u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[test]
fn avx2_p16_matches_scalar_all_matrices() {
  for m in [
    KernelMatrix::Bt601,
    KernelMatrix::Bt709,
    KernelMatrix::Bt2020Ncl,
    KernelMatrix::Smpte240m,
    KernelMatrix::Fcc,
    KernelMatrix::YCgCo,
  ] {
    for full in [true, false] {
      #[cfg(feature = "yuv-planar")]
      check_yuv420p16_u8_avx2_equivalence(32, m, full);
      #[cfg(feature = "yuv-planar")]
      check_yuv420p16_u16_avx2_equivalence(32, m, full);
      #[cfg(feature = "yuv-semi-planar")]
      check_p16_u8_avx2_equivalence(32, m, full);
      #[cfg(feature = "yuv-semi-planar")]
      check_p16_u16_avx2_equivalence(32, m, full);
    }
  }
}

#[test]
fn avx2_p16_matches_scalar_tail_widths() {
  for w in [34usize, 62, 66, 1922] {
    #[cfg(feature = "yuv-planar")]
    check_yuv420p16_u8_avx2_equivalence(w, KernelMatrix::Bt601, false);
    #[cfg(feature = "yuv-planar")]
    check_yuv420p16_u16_avx2_equivalence(w, KernelMatrix::Bt709, true);
    #[cfg(feature = "yuv-semi-planar")]
    check_p16_u8_avx2_equivalence(w, KernelMatrix::Bt601, false);
    #[cfg(feature = "yuv-semi-planar")]
    check_p16_u16_avx2_equivalence(w, KernelMatrix::Bt2020Ncl, false);
  }
}

#[test]
fn avx2_p16_matches_scalar_1920() {
  #[cfg(feature = "yuv-planar")]
  check_yuv420p16_u8_avx2_equivalence(1920, KernelMatrix::Bt709, false);
  #[cfg(feature = "yuv-planar")]
  check_yuv420p16_u16_avx2_equivalence(1920, KernelMatrix::Bt2020Ncl, false);
  #[cfg(feature = "yuv-semi-planar")]
  check_p16_u8_avx2_equivalence(1920, KernelMatrix::Bt709, false);
  #[cfg(feature = "yuv-semi-planar")]
  check_p16_u16_avx2_equivalence(1920, KernelMatrix::Bt2020Ncl, false);
}
