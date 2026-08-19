use super::{super::*, packed_yuv411_buffer};

// ---- Tier 5.25 packed YUV 4:1:1 AVX-512 scalar-equivalence ---------

fn check_uyyvyy411_rgb(width: usize, matrix: KernelMatrix, full_range: bool) {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  let p = packed_yuv411_buffer(width, 37);
  let mut s = std::vec![0u8; width * 3];
  let mut k = std::vec![0u8; width * 3];
  scalar::uyyvyy411_to_rgb_row(&p, &mut s, width, matrix, full_range);
  unsafe {
    uyyvyy411_to_rgb_row(&p, &mut k, width, matrix, full_range);
  }
  assert_eq!(
    s, k,
    "AVX-512 uyyvyy411→RGB diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_uyyvyy411_rgba(width: usize, matrix: KernelMatrix, full_range: bool) {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  let p = packed_yuv411_buffer(width, 37);
  let mut s = std::vec![0u8; width * 4];
  let mut k = std::vec![0u8; width * 4];
  scalar::uyyvyy411_to_rgba_row(&p, &mut s, width, matrix, full_range);
  unsafe {
    uyyvyy411_to_rgba_row(&p, &mut k, width, matrix, full_range);
  }
  assert_eq!(
    s, k,
    "AVX-512 uyyvyy411→RGBA diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[test]
#[cfg_attr(miri, ignore = "AVX-512 SIMD intrinsics unsupported by Miri")]
fn avx512_uyyvyy411_rgb_matches_scalar_all_matrices() {
  for m in [
    KernelMatrix::Bt601,
    KernelMatrix::Bt709,
    KernelMatrix::Bt2020Ncl,
    KernelMatrix::Smpte240m,
    KernelMatrix::Fcc,
    KernelMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_uyyvyy411_rgb(64, m, full);
      check_uyyvyy411_rgba(64, m, full);
    }
  }
}

#[test]
#[cfg_attr(miri, ignore = "AVX-512 SIMD intrinsics unsupported by Miri")]
fn avx512_uyyvyy411_matches_scalar_widths() {
  // 4..60 → all-tail (below 64-px AVX-512 block); 64 → exact;
  // 68..124 → tail remainder; 128, 192 → multi-iter; 1920 → typical
  // HD row (30 iters).
  for w in [4usize, 8, 16, 32, 60, 64, 68, 124, 128, 192, 256, 1920] {
    check_uyyvyy411_rgb(w, KernelMatrix::Bt709, false);
    check_uyyvyy411_rgba(w, KernelMatrix::Bt709, true);
    check_uyyvyy411_rgb(w, KernelMatrix::Bt2020Ncl, true);
    check_uyyvyy411_rgba(w, KernelMatrix::Bt2020Ncl, false);
  }
}

#[test]
#[cfg_attr(miri, ignore = "AVX-512 SIMD intrinsics unsupported by Miri")]
fn avx512_uyyvyy411_luma_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  for w in [4usize, 8, 16, 32, 60, 64, 68, 128, 192, 1920] {
    let p = packed_yuv411_buffer(w, 53);
    let mut s = std::vec![0u8; w];
    let mut k = std::vec![0u8; w];
    scalar::uyyvyy411_to_luma_row(&p, &mut s, w);
    unsafe {
      uyyvyy411_to_luma_row(&p, &mut k, w);
    }
    assert_eq!(s, k, "AVX-512 uyyvyy411→luma diverges (width={w})");
  }
}

#[test]
#[cfg_attr(miri, ignore = "AVX-512 SIMD intrinsics unsupported by Miri")]
fn avx512_uyyvyy411_luma_u16_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  for w in [4usize, 8, 16, 32, 60, 64, 68, 128, 192, 1920] {
    let p = packed_yuv411_buffer(w, 71);
    let mut s = std::vec![0u16; w];
    let mut k = std::vec![0u16; w];
    scalar::uyyvyy411_to_luma_u16_row(&p, &mut s, w);
    unsafe {
      uyyvyy411_to_luma_u16_row(&p, &mut k, w);
    }
    assert_eq!(s, k, "AVX-512 uyyvyy411→luma_u16 diverges (width={w})");
  }
}
