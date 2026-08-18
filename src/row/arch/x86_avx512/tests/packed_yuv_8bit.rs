use super::super::*;

// ---- Tier 3 packed YUV 4:2:2 AVX-512 scalar-equivalence -----------

fn packed_yuv422_buffer(width: usize, seed: usize) -> std::vec::Vec<u8> {
  (0..width * 2)
    .map(|i| ((i.wrapping_mul(seed).wrapping_add(seed * 3)) & 0xFF) as u8)
    .collect()
}

fn check_yuyv422_rgb(width: usize, matrix: KernelMatrix, full_range: bool) {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  let p = packed_yuv422_buffer(width, 37);
  let mut s = std::vec![0u8; width * 3];
  let mut k = std::vec![0u8; width * 3];
  scalar::yuyv422_to_rgb_row(&p, &mut s, width, matrix, full_range);
  unsafe {
    yuyv422_to_rgb_row(&p, &mut k, width, matrix, full_range);
  }
  assert_eq!(s, k, "AVX-512 yuyv422→RGB diverges (width={width})");
}

fn check_yuyv422_rgba(width: usize, matrix: KernelMatrix, full_range: bool) {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  let p = packed_yuv422_buffer(width, 37);
  let mut s = std::vec![0u8; width * 4];
  let mut k = std::vec![0u8; width * 4];
  scalar::yuyv422_to_rgba_row(&p, &mut s, width, matrix, full_range);
  unsafe {
    yuyv422_to_rgba_row(&p, &mut k, width, matrix, full_range);
  }
  assert_eq!(s, k, "AVX-512 yuyv422→RGBA diverges (width={width})");
}

fn check_uyvy422_rgb(width: usize, matrix: KernelMatrix, full_range: bool) {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  let p = packed_yuv422_buffer(width, 37);
  let mut s = std::vec![0u8; width * 3];
  let mut k = std::vec![0u8; width * 3];
  scalar::uyvy422_to_rgb_row(&p, &mut s, width, matrix, full_range);
  unsafe {
    uyvy422_to_rgb_row(&p, &mut k, width, matrix, full_range);
  }
  assert_eq!(s, k, "AVX-512 uyvy422→RGB diverges (width={width})");
}

fn check_uyvy422_rgba(width: usize, matrix: KernelMatrix, full_range: bool) {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  let p = packed_yuv422_buffer(width, 37);
  let mut s = std::vec![0u8; width * 4];
  let mut k = std::vec![0u8; width * 4];
  scalar::uyvy422_to_rgba_row(&p, &mut s, width, matrix, full_range);
  unsafe {
    uyvy422_to_rgba_row(&p, &mut k, width, matrix, full_range);
  }
  assert_eq!(s, k, "AVX-512 uyvy422→RGBA diverges (width={width})");
}

fn check_yvyu422_rgb(width: usize, matrix: KernelMatrix, full_range: bool) {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  let p = packed_yuv422_buffer(width, 37);
  let mut s = std::vec![0u8; width * 3];
  let mut k = std::vec![0u8; width * 3];
  scalar::yvyu422_to_rgb_row(&p, &mut s, width, matrix, full_range);
  unsafe {
    yvyu422_to_rgb_row(&p, &mut k, width, matrix, full_range);
  }
  assert_eq!(s, k, "AVX-512 yvyu422→RGB diverges (width={width})");
}

fn check_yvyu422_rgba(width: usize, matrix: KernelMatrix, full_range: bool) {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  let p = packed_yuv422_buffer(width, 37);
  let mut s = std::vec![0u8; width * 4];
  let mut k = std::vec![0u8; width * 4];
  scalar::yvyu422_to_rgba_row(&p, &mut s, width, matrix, full_range);
  unsafe {
    yvyu422_to_rgba_row(&p, &mut k, width, matrix, full_range);
  }
  assert_eq!(s, k, "AVX-512 yvyu422→RGBA diverges (width={width})");
}

#[test]
fn avx512_packed_yuv422_rgb_matches_scalar_all_matrices() {
  for m in [
    KernelMatrix::Bt601,
    KernelMatrix::Bt709,
    KernelMatrix::Bt2020Ncl,
    KernelMatrix::Smpte240m,
    KernelMatrix::Fcc,
    KernelMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_yuyv422_rgb(64, m, full);
      check_yuyv422_rgba(64, m, full);
      check_uyvy422_rgb(64, m, full);
      check_uyvy422_rgba(64, m, full);
      check_yvyu422_rgb(64, m, full);
      check_yvyu422_rgba(64, m, full);
    }
  }
}

#[test]
fn avx512_packed_yuv422_matches_scalar_widths() {
  // 64-pixel block boundary; tail-width sweep covers main loop +
  // scalar fallback at every block-size boundary including
  // 16/32/64-pixel splits where lower-tier code paths would land.
  for w in [
    2usize, 4, 14, 16, 18, 30, 32, 34, 62, 64, 66, 126, 128, 130, 1920, 1922,
  ] {
    check_yuyv422_rgb(w, KernelMatrix::Bt709, false);
    check_yuyv422_rgba(w, KernelMatrix::Bt709, true);
    check_uyvy422_rgb(w, KernelMatrix::Bt2020Ncl, true);
    check_uyvy422_rgba(w, KernelMatrix::Bt601, false);
    check_yvyu422_rgb(w, KernelMatrix::Smpte240m, false);
    check_yvyu422_rgba(w, KernelMatrix::YCgCo, true);
  }
}

fn check_luma(width: usize, format: char) {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
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
  assert_eq!(
    s, k,
    "AVX-512 luma diverges (format={format}, width={width})"
  );
}

#[test]
fn avx512_packed_yuv422_luma_matches_scalar_widths() {
  for w in [
    2usize, 4, 14, 16, 30, 32, 34, 62, 64, 66, 126, 128, 130, 1920, 1922,
  ] {
    check_luma(w, 'y');
    check_luma(w, 'u');
    check_luma(w, 'v');
  }
}

// ===== u16 luma extraction (3 formats; YUYV / UYVY / YVYU) =====

fn check_luma_u16(width: usize, format: char) {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  let p = packed_yuv422_buffer(width, 71);
  let mut s = std::vec![0u16; width];
  let mut k = std::vec![0u16; width];
  match format {
    'y' => unsafe {
      scalar::yuyv422_to_luma_u16_row(&p, &mut s, width);
      yuyv422_to_luma_u16_row(&p, &mut k, width);
    },
    'u' => unsafe {
      scalar::uyvy422_to_luma_u16_row(&p, &mut s, width);
      uyvy422_to_luma_u16_row(&p, &mut k, width);
    },
    'v' => unsafe {
      scalar::yvyu422_to_luma_u16_row(&p, &mut s, width);
      yvyu422_to_luma_u16_row(&p, &mut k, width);
    },
    _ => unreachable!(),
  }
  assert_eq!(
    s, k,
    "AVX-512 luma_u16 diverges (format={format}, width={width})"
  );
}

#[test]
fn avx512_packed_yuv422_luma_u16_matches_scalar_widths() {
  // Sweep covers the 64-px AVX-512 main loop boundary plus tails on
  // both sides (63 / 65 / 95 / 127 / 129).
  for w in [
    1usize, 2, 4, 8, 14, 16, 30, 32, 34, 62, 63, 64, 65, 66, 95, 126, 127, 128, 129, 130, 1920,
    1922,
  ] {
    check_luma_u16(w, 'y');
    check_luma_u16(w, 'u');
    check_luma_u16(w, 'v');
  }
}
