use super::super::*;

// ---- Tier 3 packed YUV 4:2:2 AVX2 scalar-equivalence ---------------

fn packed_yuv422_buffer(width: usize, seed: usize) -> std::vec::Vec<u8> {
  (0..width * 2)
    .map(|i| ((i.wrapping_mul(seed).wrapping_add(seed * 3)) & 0xFF) as u8)
    .collect()
}

fn check_yuyv422_rgb(width: usize, matrix: ColorMatrix, full_range: bool) {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  let p = packed_yuv422_buffer(width, 37);
  let mut s = std::vec![0u8; width * 3];
  let mut k = std::vec![0u8; width * 3];
  scalar::yuyv422_to_rgb_row(&p, &mut s, width, matrix, full_range);
  unsafe {
    yuyv422_to_rgb_row(&p, &mut k, width, matrix, full_range);
  }
  assert_eq!(s, k, "AVX2 yuyv422→RGB diverges (width={width})");
}

fn check_yuyv422_rgba(width: usize, matrix: ColorMatrix, full_range: bool) {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  let p = packed_yuv422_buffer(width, 37);
  let mut s = std::vec![0u8; width * 4];
  let mut k = std::vec![0u8; width * 4];
  scalar::yuyv422_to_rgba_row(&p, &mut s, width, matrix, full_range);
  unsafe {
    yuyv422_to_rgba_row(&p, &mut k, width, matrix, full_range);
  }
  assert_eq!(s, k, "AVX2 yuyv422→RGBA diverges (width={width})");
}

fn check_uyvy422_rgb(width: usize, matrix: ColorMatrix, full_range: bool) {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  let p = packed_yuv422_buffer(width, 37);
  let mut s = std::vec![0u8; width * 3];
  let mut k = std::vec![0u8; width * 3];
  scalar::uyvy422_to_rgb_row(&p, &mut s, width, matrix, full_range);
  unsafe {
    uyvy422_to_rgb_row(&p, &mut k, width, matrix, full_range);
  }
  assert_eq!(s, k, "AVX2 uyvy422→RGB diverges (width={width})");
}

fn check_uyvy422_rgba(width: usize, matrix: ColorMatrix, full_range: bool) {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  let p = packed_yuv422_buffer(width, 37);
  let mut s = std::vec![0u8; width * 4];
  let mut k = std::vec![0u8; width * 4];
  scalar::uyvy422_to_rgba_row(&p, &mut s, width, matrix, full_range);
  unsafe {
    uyvy422_to_rgba_row(&p, &mut k, width, matrix, full_range);
  }
  assert_eq!(s, k, "AVX2 uyvy422→RGBA diverges (width={width})");
}

fn check_yvyu422_rgb(width: usize, matrix: ColorMatrix, full_range: bool) {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  let p = packed_yuv422_buffer(width, 37);
  let mut s = std::vec![0u8; width * 3];
  let mut k = std::vec![0u8; width * 3];
  scalar::yvyu422_to_rgb_row(&p, &mut s, width, matrix, full_range);
  unsafe {
    yvyu422_to_rgb_row(&p, &mut k, width, matrix, full_range);
  }
  assert_eq!(s, k, "AVX2 yvyu422→RGB diverges (width={width})");
}

fn check_yvyu422_rgba(width: usize, matrix: ColorMatrix, full_range: bool) {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  let p = packed_yuv422_buffer(width, 37);
  let mut s = std::vec![0u8; width * 4];
  let mut k = std::vec![0u8; width * 4];
  scalar::yvyu422_to_rgba_row(&p, &mut s, width, matrix, full_range);
  unsafe {
    yvyu422_to_rgba_row(&p, &mut k, width, matrix, full_range);
  }
  assert_eq!(s, k, "AVX2 yvyu422→RGBA diverges (width={width})");
}

#[test]
fn avx2_packed_yuv422_rgb_matches_scalar_all_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_yuyv422_rgb(32, m, full);
      check_yuyv422_rgba(32, m, full);
      check_uyvy422_rgb(32, m, full);
      check_uyvy422_rgba(32, m, full);
      check_yvyu422_rgb(32, m, full);
      check_yvyu422_rgba(32, m, full);
    }
  }
}

#[test]
fn avx2_packed_yuv422_matches_scalar_widths() {
  // 32-pixel block boundary; mix of widths under, equal to, and over
  // the SIMD lane width to exercise main loop + scalar tail.
  for w in [
    2usize, 4, 14, 16, 30, 32, 34, 62, 64, 66, 126, 128, 1920, 1922,
  ] {
    check_yuyv422_rgb(w, ColorMatrix::Bt709, false);
    check_yuyv422_rgba(w, ColorMatrix::Bt709, true);
    check_uyvy422_rgb(w, ColorMatrix::Bt2020Ncl, true);
    check_uyvy422_rgba(w, ColorMatrix::Bt601, false);
    check_yvyu422_rgb(w, ColorMatrix::Smpte240m, false);
    check_yvyu422_rgba(w, ColorMatrix::YCgCo, true);
  }
}

fn check_luma(width: usize, format: char) {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  let p = packed_yuv422_buffer(width, 53);
  let mut s = std::vec![0u8; width];
  let mut k = std::vec![0u8; width];
  match format {
    'y' => unsafe {
      scalar::yuyv422_to_luma_row(&p, &mut s, width);
      yuyv422_to_luma_row(&p, &mut k, width);
    },
    'u' => unsafe {
      scalar::uyvy422_to_luma_row(&p, &mut s, width);
      uyvy422_to_luma_row(&p, &mut k, width);
    },
    'v' => unsafe {
      scalar::yvyu422_to_luma_row(&p, &mut s, width);
      yvyu422_to_luma_row(&p, &mut k, width);
    },
    _ => unreachable!(),
  }
  assert_eq!(s, k, "AVX2 luma diverges (format={format}, width={width})");
}

#[test]
fn avx2_packed_yuv422_luma_matches_scalar_widths() {
  for w in [
    2usize, 4, 14, 16, 30, 32, 34, 62, 64, 66, 126, 128, 1920, 1922,
  ] {
    check_luma(w, 'y');
    check_luma(w, 'u');
    check_luma(w, 'v');
  }
}
