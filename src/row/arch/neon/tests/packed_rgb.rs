use super::*;

// ---- Ship 9b RGBA/BGRA shuffles -----------------------------------------

fn pseudo_random_rgba(width: usize) -> std::vec::Vec<u8> {
  let n = width * 4;
  let mut out = std::vec::Vec::with_capacity(n);
  let mut state: u32 = 0x6CCD_5C7B;
  for _ in 0..n {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    out.push((state >> 8) as u8);
  }
  out
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn rgba_to_rgb_neon_matches_scalar_widths() {
  for w in [1usize, 7, 15, 16, 17, 31, 32, 33, 1920, 1921] {
    let input = pseudo_random_rgba(w);
    let mut out_scalar = std::vec![0u8; w * 3];
    let mut out_neon = std::vec![0u8; w * 3];
    scalar::rgba_to_rgb_row(&input, &mut out_scalar, w);
    unsafe {
      rgba_to_rgb_row(&input, &mut out_neon, w);
    }
    assert_eq!(out_scalar, out_neon, "width {w}");
    // Direct semantic check: every output triple = first 3 of input quad.
    for x in 0..w {
      assert_eq!(out_neon[x * 3], input[x * 4], "R width {w} px {x}");
      assert_eq!(out_neon[x * 3 + 1], input[x * 4 + 1], "G width {w} px {x}");
      assert_eq!(out_neon[x * 3 + 2], input[x * 4 + 2], "B width {w} px {x}");
    }
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn bgra_to_rgba_neon_matches_scalar_widths() {
  for w in [1usize, 7, 15, 16, 17, 31, 32, 33, 1920, 1921] {
    let input = pseudo_random_rgba(w);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_neon = std::vec![0u8; w * 4];
    scalar::bgra_to_rgba_row(&input, &mut out_scalar, w);
    unsafe {
      bgra_to_rgba_row(&input, &mut out_neon, w);
    }
    assert_eq!(out_scalar, out_neon, "width {w}");
    // R↔B swap, G + alpha unchanged.
    for x in 0..w {
      assert_eq!(out_neon[x * 4], input[x * 4 + 2], "R width {w} px {x}");
      assert_eq!(out_neon[x * 4 + 1], input[x * 4 + 1], "G width {w} px {x}");
      assert_eq!(out_neon[x * 4 + 2], input[x * 4], "B width {w} px {x}");
      assert_eq!(out_neon[x * 4 + 3], input[x * 4 + 3], "A width {w} px {x}");
    }
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn bgra_to_rgb_neon_matches_scalar_widths() {
  for w in [1usize, 7, 15, 16, 17, 31, 32, 33, 1920, 1921] {
    let input = pseudo_random_rgba(w);
    let mut out_scalar = std::vec![0u8; w * 3];
    let mut out_neon = std::vec![0u8; w * 3];
    scalar::bgra_to_rgb_row(&input, &mut out_scalar, w);
    unsafe {
      bgra_to_rgb_row(&input, &mut out_neon, w);
    }
    assert_eq!(out_scalar, out_neon, "width {w}");
    // R↔B swap, G unchanged, alpha dropped.
    for x in 0..w {
      assert_eq!(out_neon[x * 3], input[x * 4 + 2], "R width {w} px {x}");
      assert_eq!(out_neon[x * 3 + 1], input[x * 4 + 1], "G width {w} px {x}");
      assert_eq!(out_neon[x * 3 + 2], input[x * 4], "B width {w} px {x}");
    }
  }
}

// ---- Ship 9c leading-alpha shuffles -----------------------------------

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn argb_to_rgb_neon_matches_scalar_widths() {
  for w in [1usize, 7, 15, 16, 17, 31, 32, 33, 1920, 1921] {
    let input = pseudo_random_rgba(w);
    let mut out_scalar = std::vec![0u8; w * 3];
    let mut out_neon = std::vec![0u8; w * 3];
    scalar::argb_to_rgb_row(&input, &mut out_scalar, w);
    unsafe {
      argb_to_rgb_row(&input, &mut out_neon, w);
    }
    assert_eq!(out_scalar, out_neon, "width {w}");
    // Drop leading alpha — output triple = input bytes 1, 2, 3.
    for x in 0..w {
      assert_eq!(out_neon[x * 3], input[x * 4 + 1], "R width {w} px {x}");
      assert_eq!(out_neon[x * 3 + 1], input[x * 4 + 2], "G width {w} px {x}");
      assert_eq!(out_neon[x * 3 + 2], input[x * 4 + 3], "B width {w} px {x}");
    }
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn abgr_to_rgb_neon_matches_scalar_widths() {
  for w in [1usize, 7, 15, 16, 17, 31, 32, 33, 1920, 1921] {
    let input = pseudo_random_rgba(w);
    let mut out_scalar = std::vec![0u8; w * 3];
    let mut out_neon = std::vec![0u8; w * 3];
    scalar::abgr_to_rgb_row(&input, &mut out_scalar, w);
    unsafe {
      abgr_to_rgb_row(&input, &mut out_neon, w);
    }
    assert_eq!(out_scalar, out_neon, "width {w}");
    // Reversed inner three bytes — output = input bytes 3, 2, 1.
    for x in 0..w {
      assert_eq!(out_neon[x * 3], input[x * 4 + 3], "R width {w} px {x}");
      assert_eq!(out_neon[x * 3 + 1], input[x * 4 + 2], "G width {w} px {x}");
      assert_eq!(out_neon[x * 3 + 2], input[x * 4 + 1], "B width {w} px {x}");
    }
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn argb_to_rgba_neon_matches_scalar_widths() {
  for w in [1usize, 7, 15, 16, 17, 31, 32, 33, 1920, 1921] {
    let input = pseudo_random_rgba(w);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_neon = std::vec![0u8; w * 4];
    scalar::argb_to_rgba_row(&input, &mut out_scalar, w);
    unsafe {
      argb_to_rgba_row(&input, &mut out_neon, w);
    }
    assert_eq!(out_scalar, out_neon, "width {w}");
    // Rotate alpha to trailing — output = input bytes 1, 2, 3, 0.
    for x in 0..w {
      assert_eq!(out_neon[x * 4], input[x * 4 + 1], "R width {w} px {x}");
      assert_eq!(out_neon[x * 4 + 1], input[x * 4 + 2], "G width {w} px {x}");
      assert_eq!(out_neon[x * 4 + 2], input[x * 4 + 3], "B width {w} px {x}");
      assert_eq!(out_neon[x * 4 + 3], input[x * 4], "A width {w} px {x}");
    }
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn abgr_to_rgba_neon_matches_scalar_widths() {
  for w in [1usize, 7, 15, 16, 17, 31, 32, 33, 1920, 1921] {
    let input = pseudo_random_rgba(w);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_neon = std::vec![0u8; w * 4];
    scalar::abgr_to_rgba_row(&input, &mut out_scalar, w);
    unsafe {
      abgr_to_rgba_row(&input, &mut out_neon, w);
    }
    assert_eq!(out_scalar, out_neon, "width {w}");
    // Full byte reverse — output = input bytes 3, 2, 1, 0.
    for x in 0..w {
      assert_eq!(out_neon[x * 4], input[x * 4 + 3], "R width {w} px {x}");
      assert_eq!(out_neon[x * 4 + 1], input[x * 4 + 2], "G width {w} px {x}");
      assert_eq!(out_neon[x * 4 + 2], input[x * 4 + 1], "B width {w} px {x}");
      assert_eq!(out_neon[x * 4 + 3], input[x * 4], "A width {w} px {x}");
    }
  }
}

// ---- Ship 9d padding-byte shuffles -----------------------------------

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn xrgb_to_rgba_neon_matches_scalar_widths() {
  for w in [1usize, 7, 15, 16, 17, 31, 32, 33, 1920, 1921] {
    let input = pseudo_random_rgba(w);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_neon = std::vec![0u8; w * 4];
    scalar::xrgb_to_rgba_row(&input, &mut out_scalar, w);
    unsafe {
      xrgb_to_rgba_row(&input, &mut out_neon, w);
    }
    assert_eq!(out_scalar, out_neon, "width {w}");
    // Drop leading byte; alpha forced to 0xFF.
    for x in 0..w {
      assert_eq!(out_neon[x * 4], input[x * 4 + 1], "R width {w} px {x}");
      assert_eq!(out_neon[x * 4 + 1], input[x * 4 + 2], "G width {w} px {x}");
      assert_eq!(out_neon[x * 4 + 2], input[x * 4 + 3], "B width {w} px {x}");
      assert_eq!(out_neon[x * 4 + 3], 0xFF, "A width {w} px {x}");
    }
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn rgbx_to_rgba_neon_matches_scalar_widths() {
  for w in [1usize, 7, 15, 16, 17, 31, 32, 33, 1920, 1921] {
    let input = pseudo_random_rgba(w);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_neon = std::vec![0u8; w * 4];
    scalar::rgbx_to_rgba_row(&input, &mut out_scalar, w);
    unsafe {
      rgbx_to_rgba_row(&input, &mut out_neon, w);
    }
    assert_eq!(out_scalar, out_neon, "width {w}");
    for x in 0..w {
      assert_eq!(out_neon[x * 4], input[x * 4], "R width {w} px {x}");
      assert_eq!(out_neon[x * 4 + 1], input[x * 4 + 1], "G width {w} px {x}");
      assert_eq!(out_neon[x * 4 + 2], input[x * 4 + 2], "B width {w} px {x}");
      assert_eq!(out_neon[x * 4 + 3], 0xFF, "A width {w} px {x}");
    }
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn xbgr_to_rgba_neon_matches_scalar_widths() {
  for w in [1usize, 7, 15, 16, 17, 31, 32, 33, 1920, 1921] {
    let input = pseudo_random_rgba(w);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_neon = std::vec![0u8; w * 4];
    scalar::xbgr_to_rgba_row(&input, &mut out_scalar, w);
    unsafe {
      xbgr_to_rgba_row(&input, &mut out_neon, w);
    }
    assert_eq!(out_scalar, out_neon, "width {w}");
    for x in 0..w {
      assert_eq!(out_neon[x * 4], input[x * 4 + 3], "R width {w} px {x}");
      assert_eq!(out_neon[x * 4 + 1], input[x * 4 + 2], "G width {w} px {x}");
      assert_eq!(out_neon[x * 4 + 2], input[x * 4 + 1], "B width {w} px {x}");
      assert_eq!(out_neon[x * 4 + 3], 0xFF, "A width {w} px {x}");
    }
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn bgrx_to_rgba_neon_matches_scalar_widths() {
  for w in [1usize, 7, 15, 16, 17, 31, 32, 33, 1920, 1921] {
    let input = pseudo_random_rgba(w);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_neon = std::vec![0u8; w * 4];
    scalar::bgrx_to_rgba_row(&input, &mut out_scalar, w);
    unsafe {
      bgrx_to_rgba_row(&input, &mut out_neon, w);
    }
    assert_eq!(out_scalar, out_neon, "width {w}");
    for x in 0..w {
      assert_eq!(out_neon[x * 4], input[x * 4 + 2], "R width {w} px {x}");
      assert_eq!(out_neon[x * 4 + 1], input[x * 4 + 1], "G width {w} px {x}");
      assert_eq!(out_neon[x * 4 + 2], input[x * 4], "B width {w} px {x}");
      assert_eq!(out_neon[x * 4 + 3], 0xFF, "A width {w} px {x}");
    }
  }
}

// ---- Ship 9e 10-bit packed RGB ---------------------------------------

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn x2rgb10_to_rgb_neon_matches_scalar_widths() {
  for w in [1usize, 7, 15, 16, 17, 31, 32, 33, 1920, 1921] {
    let input = pseudo_random_rgba(w);
    let mut out_scalar = std::vec![0u8; w * 3];
    let mut out_neon = std::vec![0u8; w * 3];
    scalar::x2rgb10_to_rgb_row(&input, &mut out_scalar, w);
    unsafe {
      x2rgb10_to_rgb_row(&input, &mut out_neon, w);
    }
    assert_eq!(out_scalar, out_neon, "width {w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn x2rgb10_to_rgba_neon_matches_scalar_widths() {
  for w in [1usize, 7, 15, 16, 17, 31, 32, 33, 1920, 1921] {
    let input = pseudo_random_rgba(w);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_neon = std::vec![0u8; w * 4];
    scalar::x2rgb10_to_rgba_row(&input, &mut out_scalar, w);
    unsafe {
      x2rgb10_to_rgba_row(&input, &mut out_neon, w);
    }
    assert_eq!(out_scalar, out_neon, "width {w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn x2rgb10_to_rgb_u16_neon_matches_scalar_widths() {
  for w in [1usize, 7, 8, 9, 15, 16, 17, 31, 32, 33, 1920, 1921] {
    let input = pseudo_random_rgba(w);
    let mut out_scalar = std::vec![0u16; w * 3];
    let mut out_neon = std::vec![0u16; w * 3];
    scalar::x2rgb10_to_rgb_u16_row(&input, &mut out_scalar, w);
    unsafe {
      x2rgb10_to_rgb_u16_row(&input, &mut out_neon, w);
    }
    assert_eq!(out_scalar, out_neon, "width {w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn x2bgr10_to_rgb_neon_matches_scalar_widths() {
  for w in [1usize, 7, 15, 16, 17, 31, 32, 33, 1920, 1921] {
    let input = pseudo_random_rgba(w);
    let mut out_scalar = std::vec![0u8; w * 3];
    let mut out_neon = std::vec![0u8; w * 3];
    scalar::x2bgr10_to_rgb_row(&input, &mut out_scalar, w);
    unsafe {
      x2bgr10_to_rgb_row(&input, &mut out_neon, w);
    }
    assert_eq!(out_scalar, out_neon, "width {w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn x2bgr10_to_rgba_neon_matches_scalar_widths() {
  for w in [1usize, 7, 15, 16, 17, 31, 32, 33, 1920, 1921] {
    let input = pseudo_random_rgba(w);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_neon = std::vec![0u8; w * 4];
    scalar::x2bgr10_to_rgba_row(&input, &mut out_scalar, w);
    unsafe {
      x2bgr10_to_rgba_row(&input, &mut out_neon, w);
    }
    assert_eq!(out_scalar, out_neon, "width {w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn x2bgr10_to_rgb_u16_neon_matches_scalar_widths() {
  for w in [1usize, 7, 8, 9, 15, 16, 17, 31, 32, 33, 1920, 1921] {
    let input = pseudo_random_rgba(w);
    let mut out_scalar = std::vec![0u16; w * 3];
    let mut out_neon = std::vec![0u16; w * 3];
    scalar::x2bgr10_to_rgb_u16_row(&input, &mut out_scalar, w);
    unsafe {
      x2bgr10_to_rgb_u16_row(&input, &mut out_neon, w);
    }
    assert_eq!(out_scalar, out_neon, "width {w}");
  }
}
