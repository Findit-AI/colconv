use super::super::*;

// ---- Tier 3 packed YUV 4:2:2 SSE4.1 scalar-equivalence -------------

fn packed_yuv422_buffer(width: usize, seed: usize) -> std::vec::Vec<u8> {
  (0..width * 2)
    .map(|i| ((i.wrapping_mul(seed).wrapping_add(seed * 3)) & 0xFF) as u8)
    .collect()
}

// ===== YUYV422 =====

fn check_yuyv422_rgb(width: usize, matrix: KernelMatrix, full_range: bool) {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  let p = packed_yuv422_buffer(width, 37);
  let mut s = std::vec![0u8; width * 3];
  let mut k = std::vec![0u8; width * 3];
  scalar::yuyv422_to_rgb_row(&p, &mut s, width, matrix, full_range);
  unsafe {
    yuyv422_to_rgb_row(&p, &mut k, width, matrix, full_range);
  }
  assert_eq!(
    s, k,
    "SSE4.1 yuyv422→RGB diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_yuyv422_rgba(width: usize, matrix: KernelMatrix, full_range: bool) {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  let p = packed_yuv422_buffer(width, 37);
  let mut s = std::vec![0u8; width * 4];
  let mut k = std::vec![0u8; width * 4];
  scalar::yuyv422_to_rgba_row(&p, &mut s, width, matrix, full_range);
  unsafe {
    yuyv422_to_rgba_row(&p, &mut k, width, matrix, full_range);
  }
  assert_eq!(
    s, k,
    "SSE4.1 yuyv422→RGBA diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[test]
fn sse41_yuyv422_rgb_matches_scalar_all_matrices() {
  for m in [
    KernelMatrix::Bt601,
    KernelMatrix::Bt709,
    KernelMatrix::Bt2020Ncl,
    KernelMatrix::Smpte240m,
    KernelMatrix::Fcc,
    KernelMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_yuyv422_rgb(16, m, full);
      check_yuyv422_rgba(16, m, full);
    }
  }
}

#[test]
fn sse41_yuyv422_matches_scalar_widths() {
  for w in [2usize, 4, 14, 16, 18, 30, 32, 34, 62, 64, 66, 1920, 1922] {
    check_yuyv422_rgb(w, KernelMatrix::Bt709, false);
    check_yuyv422_rgba(w, KernelMatrix::Bt709, true);
  }
}

// ===== UYVY422 =====

fn check_uyvy422_rgb(width: usize, matrix: KernelMatrix, full_range: bool) {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  let p = packed_yuv422_buffer(width, 37);
  let mut s = std::vec![0u8; width * 3];
  let mut k = std::vec![0u8; width * 3];
  scalar::uyvy422_to_rgb_row(&p, &mut s, width, matrix, full_range);
  unsafe {
    uyvy422_to_rgb_row(&p, &mut k, width, matrix, full_range);
  }
  assert_eq!(
    s, k,
    "SSE4.1 uyvy422→RGB diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_uyvy422_rgba(width: usize, matrix: KernelMatrix, full_range: bool) {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  let p = packed_yuv422_buffer(width, 37);
  let mut s = std::vec![0u8; width * 4];
  let mut k = std::vec![0u8; width * 4];
  scalar::uyvy422_to_rgba_row(&p, &mut s, width, matrix, full_range);
  unsafe {
    uyvy422_to_rgba_row(&p, &mut k, width, matrix, full_range);
  }
  assert_eq!(
    s, k,
    "SSE4.1 uyvy422→RGBA diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[test]
fn sse41_uyvy422_rgb_matches_scalar_all_matrices() {
  for m in [
    KernelMatrix::Bt601,
    KernelMatrix::Bt709,
    KernelMatrix::Bt2020Ncl,
    KernelMatrix::Smpte240m,
    KernelMatrix::Fcc,
    KernelMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_uyvy422_rgb(16, m, full);
      check_uyvy422_rgba(16, m, full);
    }
  }
}

#[test]
fn sse41_uyvy422_matches_scalar_widths() {
  for w in [2usize, 4, 14, 16, 18, 30, 32, 34, 62, 64, 66, 1920, 1922] {
    check_uyvy422_rgb(w, KernelMatrix::Bt709, false);
    check_uyvy422_rgba(w, KernelMatrix::Bt709, true);
  }
}

// ===== YVYU422 =====

fn check_yvyu422_rgb(width: usize, matrix: KernelMatrix, full_range: bool) {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  let p = packed_yuv422_buffer(width, 37);
  let mut s = std::vec![0u8; width * 3];
  let mut k = std::vec![0u8; width * 3];
  scalar::yvyu422_to_rgb_row(&p, &mut s, width, matrix, full_range);
  unsafe {
    yvyu422_to_rgb_row(&p, &mut k, width, matrix, full_range);
  }
  assert_eq!(
    s, k,
    "SSE4.1 yvyu422→RGB diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_yvyu422_rgba(width: usize, matrix: KernelMatrix, full_range: bool) {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  let p = packed_yuv422_buffer(width, 37);
  let mut s = std::vec![0u8; width * 4];
  let mut k = std::vec![0u8; width * 4];
  scalar::yvyu422_to_rgba_row(&p, &mut s, width, matrix, full_range);
  unsafe {
    yvyu422_to_rgba_row(&p, &mut k, width, matrix, full_range);
  }
  assert_eq!(
    s, k,
    "SSE4.1 yvyu422→RGBA diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[test]
fn sse41_yvyu422_rgb_matches_scalar_all_matrices() {
  for m in [
    KernelMatrix::Bt601,
    KernelMatrix::Bt709,
    KernelMatrix::Bt2020Ncl,
    KernelMatrix::Smpte240m,
    KernelMatrix::Fcc,
    KernelMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_yvyu422_rgb(16, m, full);
      check_yvyu422_rgba(16, m, full);
    }
  }
}

#[test]
fn sse41_yvyu422_matches_scalar_widths() {
  for w in [2usize, 4, 14, 16, 18, 30, 32, 34, 62, 64, 66, 1920, 1922] {
    check_yvyu422_rgb(w, KernelMatrix::Bt709, false);
    check_yvyu422_rgba(w, KernelMatrix::Bt709, true);
  }
}

// ===== Luma extraction =====

fn check_luma(width: usize, format: char) {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
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
    "SSE4.1 luma diverges (format={format}, width={width})"
  );
}

#[test]
fn sse41_packed_yuv422_luma_matches_scalar_widths() {
  for w in [2usize, 4, 14, 16, 18, 30, 32, 34, 62, 64, 66, 1920, 1922] {
    check_luma(w, 'y');
    check_luma(w, 'u');
    check_luma(w, 'v');
  }
}

// ===== u16 luma extraction (3 formats; YUYV / UYVY / YVYU) =====

fn check_luma_u16(width: usize, format: char) {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
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
    "SSE4.1 luma_u16 diverges (format={format}, width={width})"
  );
}

#[test]
fn sse41_packed_yuv422_luma_u16_matches_scalar_widths() {
  // Sweep covers the 16-px SSE4.1 main loop boundary plus tails on
  // both sides (15 / 17 / 31 / 33).
  for w in [
    1usize, 2, 4, 8, 14, 15, 16, 17, 18, 30, 31, 32, 33, 34, 62, 63, 64, 66, 1920, 1922,
  ] {
    check_luma_u16(w, 'y');
    check_luma_u16(w, 'u');
    check_luma_u16(w, 'v');
  }
}
