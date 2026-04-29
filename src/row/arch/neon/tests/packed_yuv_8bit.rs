use super::super::*;

// ---- Tier 3 packed YUV 4:2:2 NEON scalar-equivalence ---------------
//
// 6 kernels (3 formats × {RGB, RGBA}) plus 3 luma extractors share a
// single deinterleave shape (`vld2q_u8` + `vuzp_u8`). The tests sweep
// every block-size boundary on the 16-pixel NEON main loop plus tail
// widths to exercise the scalar fallback dispatch.

/// Deterministic packed YUYV422-style buffer: `2 * width` bytes per
/// row, hash-like seed per byte position.
pub(super) fn packed_yuv422_buffer(width: usize, seed: usize) -> std::vec::Vec<u8> {
  (0..width * 2)
    .map(|i| ((i.wrapping_mul(seed).wrapping_add(seed * 3)) & 0xFF) as u8)
    .collect()
}

// ===== YUYV422 =====

fn check_yuyv422_rgb(width: usize, matrix: ColorMatrix, full_range: bool) {
  let p = packed_yuv422_buffer(width, 37);
  let mut s = std::vec![0u8; width * 3];
  let mut n = std::vec![0u8; width * 3];
  scalar::yuyv422_to_rgb_row(&p, &mut s, width, matrix, full_range);
  unsafe {
    yuyv422_to_rgb_row(&p, &mut n, width, matrix, full_range);
  }
  assert_eq!(
    s, n,
    "NEON yuyv422→RGB diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_yuyv422_rgba(width: usize, matrix: ColorMatrix, full_range: bool) {
  let p = packed_yuv422_buffer(width, 37);
  let mut s = std::vec![0u8; width * 4];
  let mut n = std::vec![0u8; width * 4];
  scalar::yuyv422_to_rgba_row(&p, &mut s, width, matrix, full_range);
  unsafe {
    yuyv422_to_rgba_row(&p, &mut n, width, matrix, full_range);
  }
  assert_eq!(
    s, n,
    "NEON yuyv422→RGBA diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_yuyv422_rgb_matches_scalar_all_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_yuyv422_rgb(16, m, full);
      check_yuyv422_rgba(16, m, full);
    }
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_yuyv422_matches_scalar_widths() {
  for w in [2usize, 4, 14, 16, 18, 30, 32, 34, 62, 64, 66, 1920, 1922] {
    check_yuyv422_rgb(w, ColorMatrix::Bt709, false);
    check_yuyv422_rgba(w, ColorMatrix::Bt709, true);
  }
}

// ===== UYVY422 =====

fn check_uyvy422_rgb(width: usize, matrix: ColorMatrix, full_range: bool) {
  let p = packed_yuv422_buffer(width, 37);
  let mut s = std::vec![0u8; width * 3];
  let mut n = std::vec![0u8; width * 3];
  scalar::uyvy422_to_rgb_row(&p, &mut s, width, matrix, full_range);
  unsafe {
    uyvy422_to_rgb_row(&p, &mut n, width, matrix, full_range);
  }
  assert_eq!(
    s, n,
    "NEON uyvy422→RGB diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_uyvy422_rgba(width: usize, matrix: ColorMatrix, full_range: bool) {
  let p = packed_yuv422_buffer(width, 37);
  let mut s = std::vec![0u8; width * 4];
  let mut n = std::vec![0u8; width * 4];
  scalar::uyvy422_to_rgba_row(&p, &mut s, width, matrix, full_range);
  unsafe {
    uyvy422_to_rgba_row(&p, &mut n, width, matrix, full_range);
  }
  assert_eq!(
    s, n,
    "NEON uyvy422→RGBA diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_uyvy422_rgb_matches_scalar_all_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_uyvy422_rgb(16, m, full);
      check_uyvy422_rgba(16, m, full);
    }
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_uyvy422_matches_scalar_widths() {
  for w in [2usize, 4, 14, 16, 18, 30, 32, 34, 62, 64, 66, 1920, 1922] {
    check_uyvy422_rgb(w, ColorMatrix::Bt709, false);
    check_uyvy422_rgba(w, ColorMatrix::Bt709, true);
  }
}

// ===== YVYU422 =====

fn check_yvyu422_rgb(width: usize, matrix: ColorMatrix, full_range: bool) {
  let p = packed_yuv422_buffer(width, 37);
  let mut s = std::vec![0u8; width * 3];
  let mut n = std::vec![0u8; width * 3];
  scalar::yvyu422_to_rgb_row(&p, &mut s, width, matrix, full_range);
  unsafe {
    yvyu422_to_rgb_row(&p, &mut n, width, matrix, full_range);
  }
  assert_eq!(
    s, n,
    "NEON yvyu422→RGB diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_yvyu422_rgba(width: usize, matrix: ColorMatrix, full_range: bool) {
  let p = packed_yuv422_buffer(width, 37);
  let mut s = std::vec![0u8; width * 4];
  let mut n = std::vec![0u8; width * 4];
  scalar::yvyu422_to_rgba_row(&p, &mut s, width, matrix, full_range);
  unsafe {
    yvyu422_to_rgba_row(&p, &mut n, width, matrix, full_range);
  }
  assert_eq!(
    s, n,
    "NEON yvyu422→RGBA diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_yvyu422_rgb_matches_scalar_all_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_yvyu422_rgb(16, m, full);
      check_yvyu422_rgba(16, m, full);
    }
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_yvyu422_matches_scalar_widths() {
  for w in [2usize, 4, 14, 16, 18, 30, 32, 34, 62, 64, 66, 1920, 1922] {
    check_yvyu422_rgb(w, ColorMatrix::Bt709, false);
    check_yvyu422_rgba(w, ColorMatrix::Bt709, true);
  }
}

// ===== Luma extraction (3 formats; YUYV / UYVY / YVYU) =====

fn check_luma(width: usize, format: char) {
  let p = packed_yuv422_buffer(width, 53);
  let mut s = std::vec![0u8; width];
  let mut n = std::vec![0u8; width];
  match format {
    'y' => unsafe {
      scalar::yuyv422_to_luma_row(&p, &mut s, width);
      yuyv422_to_luma_row(&p, &mut n, width);
    },
    'u' => unsafe {
      scalar::uyvy422_to_luma_row(&p, &mut s, width);
      uyvy422_to_luma_row(&p, &mut n, width);
    },
    'v' => unsafe {
      scalar::yvyu422_to_luma_row(&p, &mut s, width);
      yvyu422_to_luma_row(&p, &mut n, width);
    },
    _ => unreachable!(),
  }
  assert_eq!(s, n, "NEON {format}xyz luma diverges (width={width})");
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_packed_yuv422_luma_matches_scalar_widths() {
  for w in [2usize, 4, 14, 16, 18, 30, 32, 34, 62, 64, 66, 1920, 1922] {
    check_luma(w, 'y');
    check_luma(w, 'u');
    check_luma(w, 'v');
  }
}
