use super::*;

// ---- Ship 9b RGBA/BGRA shuffles -----------------------------------------

fn pseudo_random_rgba(width: usize) -> std::vec::Vec<u8> {
  (0..width * 4)
    .map(|i| ((i * 17 + 41) & 0xFF) as u8)
    .collect()
}

#[test]
fn avx512_rgba_to_rgb_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  for w in [1usize, 31, 63, 64, 65, 127, 128, 129, 1920, 1921] {
    let input = pseudo_random_rgba(w);
    let mut out_scalar = std::vec![0u8; w * 3];
    let mut out_avx = std::vec![0u8; w * 3];
    scalar::rgba_to_rgb_row(&input, &mut out_scalar, w);
    unsafe {
      rgba_to_rgb_row(&input, &mut out_avx, w);
    }
    assert_eq!(
      out_scalar, out_avx,
      "AVX-512 rgba_to_rgb diverges (width={w})"
    );
  }
}

#[test]
fn avx512_bgra_to_rgba_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  for w in [1usize, 31, 63, 64, 65, 127, 128, 129, 1920, 1921] {
    let input = pseudo_random_rgba(w);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_avx = std::vec![0u8; w * 4];
    scalar::bgra_to_rgba_row(&input, &mut out_scalar, w);
    unsafe {
      bgra_to_rgba_row(&input, &mut out_avx, w);
    }
    assert_eq!(
      out_scalar, out_avx,
      "AVX-512 bgra_to_rgba diverges (width={w})"
    );
  }
}

#[test]
fn avx512_bgra_to_rgb_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  for w in [1usize, 31, 63, 64, 65, 127, 128, 129, 1920, 1921] {
    let input = pseudo_random_rgba(w);
    let mut out_scalar = std::vec![0u8; w * 3];
    let mut out_avx = std::vec![0u8; w * 3];
    scalar::bgra_to_rgb_row(&input, &mut out_scalar, w);
    unsafe {
      bgra_to_rgb_row(&input, &mut out_avx, w);
    }
    assert_eq!(
      out_scalar, out_avx,
      "AVX-512 bgra_to_rgb diverges (width={w})"
    );
  }
}

// ---- Ship 9c leading-alpha shuffles -----------------------------------

#[test]
fn avx512_argb_to_rgb_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  for w in [1usize, 31, 63, 64, 65, 127, 128, 129, 1920, 1921] {
    let input = pseudo_random_rgba(w);
    let mut out_scalar = std::vec![0u8; w * 3];
    let mut out_avx = std::vec![0u8; w * 3];
    scalar::argb_to_rgb_row(&input, &mut out_scalar, w);
    unsafe {
      argb_to_rgb_row(&input, &mut out_avx, w);
    }
    assert_eq!(
      out_scalar, out_avx,
      "AVX-512 argb_to_rgb diverges (width={w})"
    );
  }
}

#[test]
fn avx512_abgr_to_rgb_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  for w in [1usize, 31, 63, 64, 65, 127, 128, 129, 1920, 1921] {
    let input = pseudo_random_rgba(w);
    let mut out_scalar = std::vec![0u8; w * 3];
    let mut out_avx = std::vec![0u8; w * 3];
    scalar::abgr_to_rgb_row(&input, &mut out_scalar, w);
    unsafe {
      abgr_to_rgb_row(&input, &mut out_avx, w);
    }
    assert_eq!(
      out_scalar, out_avx,
      "AVX-512 abgr_to_rgb diverges (width={w})"
    );
  }
}

#[test]
fn avx512_argb_to_rgba_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  for w in [1usize, 31, 63, 64, 65, 127, 128, 129, 1920, 1921] {
    let input = pseudo_random_rgba(w);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_avx = std::vec![0u8; w * 4];
    scalar::argb_to_rgba_row(&input, &mut out_scalar, w);
    unsafe {
      argb_to_rgba_row(&input, &mut out_avx, w);
    }
    assert_eq!(
      out_scalar, out_avx,
      "AVX-512 argb_to_rgba diverges (width={w})"
    );
  }
}

#[test]
fn avx512_abgr_to_rgba_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  for w in [1usize, 31, 63, 64, 65, 127, 128, 129, 1920, 1921] {
    let input = pseudo_random_rgba(w);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_avx = std::vec![0u8; w * 4];
    scalar::abgr_to_rgba_row(&input, &mut out_scalar, w);
    unsafe {
      abgr_to_rgba_row(&input, &mut out_avx, w);
    }
    assert_eq!(
      out_scalar, out_avx,
      "AVX-512 abgr_to_rgba diverges (width={w})"
    );
  }
}

// ---- Ship 9d padding-byte shuffles -----------------------------------

#[test]
fn avx512_xrgb_to_rgba_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  for w in [1usize, 31, 63, 64, 65, 127, 128, 129, 1920, 1921] {
    let input = pseudo_random_rgba(w);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_avx = std::vec![0u8; w * 4];
    scalar::xrgb_to_rgba_row(&input, &mut out_scalar, w);
    unsafe {
      xrgb_to_rgba_row(&input, &mut out_avx, w);
    }
    assert_eq!(
      out_scalar, out_avx,
      "AVX-512 xrgb_to_rgba diverges (width={w})"
    );
  }
}

#[test]
fn avx512_rgbx_to_rgba_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  for w in [1usize, 31, 63, 64, 65, 127, 128, 129, 1920, 1921] {
    let input = pseudo_random_rgba(w);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_avx = std::vec![0u8; w * 4];
    scalar::rgbx_to_rgba_row(&input, &mut out_scalar, w);
    unsafe {
      rgbx_to_rgba_row(&input, &mut out_avx, w);
    }
    assert_eq!(
      out_scalar, out_avx,
      "AVX-512 rgbx_to_rgba diverges (width={w})"
    );
  }
}

#[test]
fn avx512_xbgr_to_rgba_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  for w in [1usize, 31, 63, 64, 65, 127, 128, 129, 1920, 1921] {
    let input = pseudo_random_rgba(w);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_avx = std::vec![0u8; w * 4];
    scalar::xbgr_to_rgba_row(&input, &mut out_scalar, w);
    unsafe {
      xbgr_to_rgba_row(&input, &mut out_avx, w);
    }
    assert_eq!(
      out_scalar, out_avx,
      "AVX-512 xbgr_to_rgba diverges (width={w})"
    );
  }
}

#[test]
fn avx512_bgrx_to_rgba_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  for w in [1usize, 31, 63, 64, 65, 127, 128, 129, 1920, 1921] {
    let input = pseudo_random_rgba(w);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_avx = std::vec![0u8; w * 4];
    scalar::bgrx_to_rgba_row(&input, &mut out_scalar, w);
    unsafe {
      bgrx_to_rgba_row(&input, &mut out_avx, w);
    }
    assert_eq!(
      out_scalar, out_avx,
      "AVX-512 bgrx_to_rgba diverges (width={w})"
    );
  }
}

// ---- Ship 9e 10-bit packed RGB ---------------------------------------

#[test]
fn avx512_x2rgb10_to_rgb_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  for w in [1usize, 31, 63, 64, 65, 127, 128, 129, 1920, 1921] {
    let input = pseudo_random_rgba(w);
    let mut out_scalar = std::vec![0u8; w * 3];
    let mut out_avx = std::vec![0u8; w * 3];
    scalar::x2rgb10_to_rgb_row(&input, &mut out_scalar, w);
    unsafe {
      x2rgb10_to_rgb_row(&input, &mut out_avx, w);
    }
    assert_eq!(
      out_scalar, out_avx,
      "AVX-512 x2rgb10_to_rgb diverges (width={w})"
    );
  }
}

#[test]
fn avx512_x2rgb10_to_rgba_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  for w in [1usize, 31, 63, 64, 65, 127, 128, 129, 1920, 1921] {
    let input = pseudo_random_rgba(w);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_avx = std::vec![0u8; w * 4];
    scalar::x2rgb10_to_rgba_row(&input, &mut out_scalar, w);
    unsafe {
      x2rgb10_to_rgba_row(&input, &mut out_avx, w);
    }
    assert_eq!(
      out_scalar, out_avx,
      "AVX-512 x2rgb10_to_rgba diverges (width={w})"
    );
  }
}

#[test]
fn avx512_x2rgb10_to_rgb_u16_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  for w in [1usize, 7, 8, 9, 31, 32, 33, 63, 64, 65, 1920, 1921] {
    let input = pseudo_random_rgba(w);
    let mut out_scalar = std::vec![0u16; w * 3];
    let mut out_avx = std::vec![0u16; w * 3];
    scalar::x2rgb10_to_rgb_u16_row(&input, &mut out_scalar, w);
    unsafe {
      x2rgb10_to_rgb_u16_row(&input, &mut out_avx, w);
    }
    assert_eq!(
      out_scalar, out_avx,
      "AVX-512 x2rgb10_to_rgb_u16 diverges (width={w})"
    );
  }
}

#[test]
fn avx512_x2bgr10_to_rgb_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  for w in [1usize, 31, 63, 64, 65, 127, 128, 129, 1920, 1921] {
    let input = pseudo_random_rgba(w);
    let mut out_scalar = std::vec![0u8; w * 3];
    let mut out_avx = std::vec![0u8; w * 3];
    scalar::x2bgr10_to_rgb_row(&input, &mut out_scalar, w);
    unsafe {
      x2bgr10_to_rgb_row(&input, &mut out_avx, w);
    }
    assert_eq!(
      out_scalar, out_avx,
      "AVX-512 x2bgr10_to_rgb diverges (width={w})"
    );
  }
}

#[test]
fn avx512_x2bgr10_to_rgba_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  for w in [1usize, 31, 63, 64, 65, 127, 128, 129, 1920, 1921] {
    let input = pseudo_random_rgba(w);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_avx = std::vec![0u8; w * 4];
    scalar::x2bgr10_to_rgba_row(&input, &mut out_scalar, w);
    unsafe {
      x2bgr10_to_rgba_row(&input, &mut out_avx, w);
    }
    assert_eq!(
      out_scalar, out_avx,
      "AVX-512 x2bgr10_to_rgba diverges (width={w})"
    );
  }
}

#[test]
fn avx512_x2bgr10_to_rgb_u16_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  for w in [1usize, 7, 8, 9, 31, 32, 33, 63, 64, 65, 1920, 1921] {
    let input = pseudo_random_rgba(w);
    let mut out_scalar = std::vec![0u16; w * 3];
    let mut out_avx = std::vec![0u16; w * 3];
    scalar::x2bgr10_to_rgb_u16_row(&input, &mut out_scalar, w);
    unsafe {
      x2bgr10_to_rgb_u16_row(&input, &mut out_avx, w);
    }
    assert_eq!(
      out_scalar, out_avx,
      "AVX-512 x2bgr10_to_rgb_u16 diverges (width={w})"
    );
  }
}
