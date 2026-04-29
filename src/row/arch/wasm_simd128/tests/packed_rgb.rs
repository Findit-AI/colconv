use super::*;

// ---- Ship 9b RGBA/BGRA shuffles -----------------------------------------

fn pseudo_random_rgba(width: usize) -> std::vec::Vec<u8> {
  (0..width * 4)
    .map(|i| ((i * 17 + 41) & 0xFF) as u8)
    .collect()
}

#[test]
fn simd128_rgba_to_rgb_matches_scalar() {
  for w in [1usize, 7, 15, 16, 17, 31, 32, 33, 1920, 1921] {
    let input = pseudo_random_rgba(w);
    let mut out_scalar = std::vec![0u8; w * 3];
    let mut out_wasm = std::vec![0u8; w * 3];
    scalar::rgba_to_rgb_row(&input, &mut out_scalar, w);
    unsafe {
      rgba_to_rgb_row(&input, &mut out_wasm, w);
    }
    assert_eq!(
      out_scalar, out_wasm,
      "simd128 rgba_to_rgb diverges (width={w})"
    );
  }
}

#[test]
fn simd128_bgra_to_rgba_matches_scalar() {
  for w in [1usize, 7, 15, 16, 17, 31, 32, 33, 1920, 1921] {
    let input = pseudo_random_rgba(w);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_wasm = std::vec![0u8; w * 4];
    scalar::bgra_to_rgba_row(&input, &mut out_scalar, w);
    unsafe {
      bgra_to_rgba_row(&input, &mut out_wasm, w);
    }
    assert_eq!(
      out_scalar, out_wasm,
      "simd128 bgra_to_rgba diverges (width={w})"
    );
  }
}

#[test]
fn simd128_bgra_to_rgb_matches_scalar() {
  for w in [1usize, 7, 15, 16, 17, 31, 32, 33, 1920, 1921] {
    let input = pseudo_random_rgba(w);
    let mut out_scalar = std::vec![0u8; w * 3];
    let mut out_wasm = std::vec![0u8; w * 3];
    scalar::bgra_to_rgb_row(&input, &mut out_scalar, w);
    unsafe {
      bgra_to_rgb_row(&input, &mut out_wasm, w);
    }
    assert_eq!(
      out_scalar, out_wasm,
      "simd128 bgra_to_rgb diverges (width={w})"
    );
  }
}

// ---- Ship 9c leading-alpha shuffles -----------------------------------

#[test]
fn simd128_argb_to_rgb_matches_scalar() {
  for w in [1usize, 7, 15, 16, 17, 31, 32, 33, 1920, 1921] {
    let input = pseudo_random_rgba(w);
    let mut out_scalar = std::vec![0u8; w * 3];
    let mut out_wasm = std::vec![0u8; w * 3];
    scalar::argb_to_rgb_row(&input, &mut out_scalar, w);
    unsafe {
      argb_to_rgb_row(&input, &mut out_wasm, w);
    }
    assert_eq!(
      out_scalar, out_wasm,
      "simd128 argb_to_rgb diverges (width={w})"
    );
  }
}

#[test]
fn simd128_abgr_to_rgb_matches_scalar() {
  for w in [1usize, 7, 15, 16, 17, 31, 32, 33, 1920, 1921] {
    let input = pseudo_random_rgba(w);
    let mut out_scalar = std::vec![0u8; w * 3];
    let mut out_wasm = std::vec![0u8; w * 3];
    scalar::abgr_to_rgb_row(&input, &mut out_scalar, w);
    unsafe {
      abgr_to_rgb_row(&input, &mut out_wasm, w);
    }
    assert_eq!(
      out_scalar, out_wasm,
      "simd128 abgr_to_rgb diverges (width={w})"
    );
  }
}

#[test]
fn simd128_argb_to_rgba_matches_scalar() {
  for w in [1usize, 7, 15, 16, 17, 31, 32, 33, 1920, 1921] {
    let input = pseudo_random_rgba(w);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_wasm = std::vec![0u8; w * 4];
    scalar::argb_to_rgba_row(&input, &mut out_scalar, w);
    unsafe {
      argb_to_rgba_row(&input, &mut out_wasm, w);
    }
    assert_eq!(
      out_scalar, out_wasm,
      "simd128 argb_to_rgba diverges (width={w})"
    );
  }
}

#[test]
fn simd128_abgr_to_rgba_matches_scalar() {
  for w in [1usize, 7, 15, 16, 17, 31, 32, 33, 1920, 1921] {
    let input = pseudo_random_rgba(w);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_wasm = std::vec![0u8; w * 4];
    scalar::abgr_to_rgba_row(&input, &mut out_scalar, w);
    unsafe {
      abgr_to_rgba_row(&input, &mut out_wasm, w);
    }
    assert_eq!(
      out_scalar, out_wasm,
      "simd128 abgr_to_rgba diverges (width={w})"
    );
  }
}

// ---- Ship 9d padding-byte shuffles -----------------------------------

#[test]
fn simd128_xrgb_to_rgba_matches_scalar() {
  for w in [1usize, 7, 15, 16, 17, 31, 32, 33, 1920, 1921] {
    let input = pseudo_random_rgba(w);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_wasm = std::vec![0u8; w * 4];
    scalar::xrgb_to_rgba_row(&input, &mut out_scalar, w);
    unsafe {
      xrgb_to_rgba_row(&input, &mut out_wasm, w);
    }
    assert_eq!(
      out_scalar, out_wasm,
      "simd128 xrgb_to_rgba diverges (width={w})"
    );
  }
}

#[test]
fn simd128_rgbx_to_rgba_matches_scalar() {
  for w in [1usize, 7, 15, 16, 17, 31, 32, 33, 1920, 1921] {
    let input = pseudo_random_rgba(w);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_wasm = std::vec![0u8; w * 4];
    scalar::rgbx_to_rgba_row(&input, &mut out_scalar, w);
    unsafe {
      rgbx_to_rgba_row(&input, &mut out_wasm, w);
    }
    assert_eq!(
      out_scalar, out_wasm,
      "simd128 rgbx_to_rgba diverges (width={w})"
    );
  }
}

#[test]
fn simd128_xbgr_to_rgba_matches_scalar() {
  for w in [1usize, 7, 15, 16, 17, 31, 32, 33, 1920, 1921] {
    let input = pseudo_random_rgba(w);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_wasm = std::vec![0u8; w * 4];
    scalar::xbgr_to_rgba_row(&input, &mut out_scalar, w);
    unsafe {
      xbgr_to_rgba_row(&input, &mut out_wasm, w);
    }
    assert_eq!(
      out_scalar, out_wasm,
      "simd128 xbgr_to_rgba diverges (width={w})"
    );
  }
}

#[test]
fn simd128_bgrx_to_rgba_matches_scalar() {
  for w in [1usize, 7, 15, 16, 17, 31, 32, 33, 1920, 1921] {
    let input = pseudo_random_rgba(w);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_wasm = std::vec![0u8; w * 4];
    scalar::bgrx_to_rgba_row(&input, &mut out_scalar, w);
    unsafe {
      bgrx_to_rgba_row(&input, &mut out_wasm, w);
    }
    assert_eq!(
      out_scalar, out_wasm,
      "simd128 bgrx_to_rgba diverges (width={w})"
    );
  }
}

// ---- Ship 9e 10-bit packed RGB ---------------------------------------

#[test]
fn simd128_x2rgb10_to_rgb_matches_scalar() {
  for w in [1usize, 7, 15, 16, 17, 31, 32, 33, 1920, 1921] {
    let input = pseudo_random_rgba(w);
    let mut out_scalar = std::vec![0u8; w * 3];
    let mut out_wasm = std::vec![0u8; w * 3];
    scalar::x2rgb10_to_rgb_row(&input, &mut out_scalar, w);
    unsafe {
      x2rgb10_to_rgb_row(&input, &mut out_wasm, w);
    }
    assert_eq!(
      out_scalar, out_wasm,
      "simd128 x2rgb10_to_rgb diverges (width={w})"
    );
  }
}

#[test]
fn simd128_x2rgb10_to_rgba_matches_scalar() {
  for w in [1usize, 3, 4, 5, 7, 15, 16, 17, 31, 32, 33, 1920, 1921] {
    let input = pseudo_random_rgba(w);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_wasm = std::vec![0u8; w * 4];
    scalar::x2rgb10_to_rgba_row(&input, &mut out_scalar, w);
    unsafe {
      x2rgb10_to_rgba_row(&input, &mut out_wasm, w);
    }
    assert_eq!(
      out_scalar, out_wasm,
      "simd128 x2rgb10_to_rgba diverges (width={w})"
    );
  }
}

#[test]
fn simd128_x2rgb10_to_rgb_u16_matches_scalar() {
  for w in [1usize, 7, 8, 9, 15, 16, 17, 31, 32, 33, 1920, 1921] {
    let input = pseudo_random_rgba(w);
    let mut out_scalar = std::vec![0u16; w * 3];
    let mut out_wasm = std::vec![0u16; w * 3];
    scalar::x2rgb10_to_rgb_u16_row(&input, &mut out_scalar, w);
    unsafe {
      x2rgb10_to_rgb_u16_row(&input, &mut out_wasm, w);
    }
    assert_eq!(
      out_scalar, out_wasm,
      "simd128 x2rgb10_to_rgb_u16 diverges (width={w})"
    );
  }
}

#[test]
fn simd128_x2bgr10_to_rgb_matches_scalar() {
  for w in [1usize, 7, 15, 16, 17, 31, 32, 33, 1920, 1921] {
    let input = pseudo_random_rgba(w);
    let mut out_scalar = std::vec![0u8; w * 3];
    let mut out_wasm = std::vec![0u8; w * 3];
    scalar::x2bgr10_to_rgb_row(&input, &mut out_scalar, w);
    unsafe {
      x2bgr10_to_rgb_row(&input, &mut out_wasm, w);
    }
    assert_eq!(
      out_scalar, out_wasm,
      "simd128 x2bgr10_to_rgb diverges (width={w})"
    );
  }
}

#[test]
fn simd128_x2bgr10_to_rgba_matches_scalar() {
  for w in [1usize, 3, 4, 5, 7, 15, 16, 17, 31, 32, 33, 1920, 1921] {
    let input = pseudo_random_rgba(w);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_wasm = std::vec![0u8; w * 4];
    scalar::x2bgr10_to_rgba_row(&input, &mut out_scalar, w);
    unsafe {
      x2bgr10_to_rgba_row(&input, &mut out_wasm, w);
    }
    assert_eq!(
      out_scalar, out_wasm,
      "simd128 x2bgr10_to_rgba diverges (width={w})"
    );
  }
}

#[test]
fn simd128_x2bgr10_to_rgb_u16_matches_scalar() {
  for w in [1usize, 7, 8, 9, 15, 16, 17, 31, 32, 33, 1920, 1921] {
    let input = pseudo_random_rgba(w);
    let mut out_scalar = std::vec![0u16; w * 3];
    let mut out_wasm = std::vec![0u16; w * 3];
    scalar::x2bgr10_to_rgb_u16_row(&input, &mut out_scalar, w);
    unsafe {
      x2bgr10_to_rgb_u16_row(&input, &mut out_wasm, w);
    }
    assert_eq!(
      out_scalar, out_wasm,
      "simd128 x2bgr10_to_rgb_u16 diverges (width={w})"
    );
  }
}
