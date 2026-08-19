use super::{super::*, packed_yuv411_buffer};

// ---- Tier 5.25 packed YUV 4:1:1 NEON scalar-equivalence ------------

fn check_uyyvyy411_rgb(width: usize, matrix: KernelMatrix, full_range: bool) {
  let p = packed_yuv411_buffer(width, 37);
  let mut s = std::vec![0u8; width * 3];
  let mut n = std::vec![0u8; width * 3];
  scalar::uyyvyy411_to_rgb_row(&p, &mut s, width, matrix, full_range);
  unsafe {
    uyyvyy411_to_rgb_row(&p, &mut n, width, matrix, full_range);
  }
  assert_eq!(
    s, n,
    "NEON uyyvyy411→RGB diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_uyyvyy411_rgba(width: usize, matrix: KernelMatrix, full_range: bool) {
  let p = packed_yuv411_buffer(width, 37);
  let mut s = std::vec![0u8; width * 4];
  let mut n = std::vec![0u8; width * 4];
  scalar::uyyvyy411_to_rgba_row(&p, &mut s, width, matrix, full_range);
  unsafe {
    uyyvyy411_to_rgba_row(&p, &mut n, width, matrix, full_range);
  }
  assert_eq!(
    s, n,
    "NEON uyyvyy411→RGBA diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_uyyvyy411_rgb_matches_scalar_all_matrices() {
  for m in [
    KernelMatrix::Bt601,
    KernelMatrix::Bt709,
    KernelMatrix::Bt2020Ncl,
    KernelMatrix::Smpte240m,
    KernelMatrix::Fcc,
    KernelMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_uyyvyy411_rgb(32, m, full);
      check_uyyvyy411_rgba(32, m, full);
    }
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_uyyvyy411_matches_scalar_widths() {
  // Width must be a multiple of 4. Sweep small / boundary / wide.
  // 4..28 → all-tail (below 32-px NEON block); 32 → exact;
  // 36..60 → tail remainder; 64, 96, 128 → multi-iter; 1920 → typical
  // HD row.
  for w in [4usize, 8, 12, 16, 20, 28, 32, 36, 60, 64, 96, 128, 1920] {
    check_uyyvyy411_rgb(w, KernelMatrix::Bt709, false);
    check_uyyvyy411_rgba(w, KernelMatrix::Bt709, true);
    check_uyyvyy411_rgb(w, KernelMatrix::Bt2020Ncl, true);
    check_uyyvyy411_rgba(w, KernelMatrix::Bt2020Ncl, false);
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_uyyvyy411_luma_matches_scalar() {
  for w in [4usize, 8, 12, 16, 28, 32, 36, 60, 64, 128, 1920] {
    let p = packed_yuv411_buffer(w, 53);
    let mut s = std::vec![0u8; w];
    let mut n = std::vec![0u8; w];
    scalar::uyyvyy411_to_luma_row(&p, &mut s, w);
    unsafe {
      uyyvyy411_to_luma_row(&p, &mut n, w);
    }
    assert_eq!(s, n, "NEON uyyvyy411→luma diverges (width={w})");
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_uyyvyy411_luma_u16_matches_scalar() {
  for w in [4usize, 8, 12, 16, 28, 32, 36, 60, 64, 128, 1920] {
    let p = packed_yuv411_buffer(w, 71);
    let mut s = std::vec![0u16; w];
    let mut n = std::vec![0u16; w];
    scalar::uyyvyy411_to_luma_u16_row(&p, &mut s, w);
    unsafe {
      uyyvyy411_to_luma_u16_row(&p, &mut n, w);
    }
    assert_eq!(s, n, "NEON uyyvyy411→luma_u16 diverges (width={w})");
  }
}
