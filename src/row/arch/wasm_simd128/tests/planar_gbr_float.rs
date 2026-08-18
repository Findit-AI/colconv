//! wasm-simd128 SIMD-vs-scalar parity tests for planar GBR float kernels (Tier 10).
//!
//! Compile-time gated — these tests only compile when targeting wasm32
//! (via `row::arch::wasm_simd128::mod`). No runtime guard needed; the
//! `#[target_feature(enable = "simd128")]` on the kernel functions is
//! compile-time only on wasm (no runtime detection).
//!
//! Test pattern: Vec/assert_eq (no `for n in 0..N { assert_eq!(out[n], ...) }` —
//! `clippy::needless_range_loop` rejects).

use super::*;

// ---- Helpers -----------------------------------------------------------------

fn gbr_plane_f32(width: usize, seed: u32) -> std::vec::Vec<f32> {
  let mut state = seed;
  (0..width)
    .map(|_| {
      state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
      // Mix: ~60% in [0,1], ~20% HDR (>1), ~10% negative, ~10% boundary.
      let kind = (state >> 28) & 0b11;
      match kind {
        0 => ((state >> 8) & 0xFF) as f32 / 255.0,
        1 => 1.0 + ((state >> 16) & 0xF) as f32 * 0.1,
        2 => -(((state >> 4) & 0x3F) as f32) / 255.0,
        _ => (((state >> 12) & 0xFF) as f32) / 255.0,
      }
    })
    .collect()
}

fn gbr_plane_f16(width: usize, seed: u32) -> std::vec::Vec<half::f16> {
  gbr_plane_f32(width, seed)
    .iter()
    .map(|&v| half::f16::from_f32(v))
    .collect()
}

// ---- Gbrpf32 → u8 RGB --------------------------------------------------------

#[test]
fn wasm_gbrpf32_to_rgb_matches_scalar() {
  for w in [1usize, 3, 4, 5, 7, 8, 15, 16, 17, 31, 33, 1920, 1921] {
    let g = gbr_plane_f32(w, 0xA5A5_3C3C);
    let b = gbr_plane_f32(w, 0x12AB_34CD);
    let r = gbr_plane_f32(w, 0xDEAD_BEEF);
    let mut out_scalar = std::vec![0u8; w * 3];
    let mut out_simd = std::vec![0u8; w * 3];
    scalar::gbrpf32_to_rgb_row::<false>(&g, &b, &r, &mut out_scalar, w);
    unsafe {
      gbrpf32_to_rgb_row::<false>(&g, &b, &r, &mut out_simd, w);
    }
    assert_eq!(out_scalar, out_simd, "wasm gbrpf32_to_rgb width={w}");
  }
}

// ---- Gbrpf32 → u8 RGBA -------------------------------------------------------

#[test]
fn wasm_gbrpf32_to_rgba_matches_scalar() {
  for w in [1usize, 3, 4, 5, 7, 8, 15, 16, 17, 31, 33, 1920, 1921] {
    let g = gbr_plane_f32(w, 0xB1B2_C3C4);
    let b = gbr_plane_f32(w, 0x5A5B_6C6D);
    let r = gbr_plane_f32(w, 0xF0E1_D2C3);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_simd = std::vec![0u8; w * 4];
    scalar::gbrpf32_to_rgba_row::<false>(&g, &b, &r, &mut out_scalar, w);
    unsafe {
      gbrpf32_to_rgba_row::<false>(&g, &b, &r, &mut out_simd, w);
    }
    assert_eq!(out_scalar, out_simd, "wasm gbrpf32_to_rgba width={w}");
  }
}

// ---- Gbrpf32 → u16 RGB -------------------------------------------------------

#[test]
fn wasm_gbrpf32_to_rgb_u16_matches_scalar() {
  for w in [1usize, 3, 4, 5, 7, 8, 15, 16, 17, 31, 33, 1920, 1921] {
    let g = gbr_plane_f32(w, 0x1122_3344);
    let b = gbr_plane_f32(w, 0x5566_7788);
    let r = gbr_plane_f32(w, 0x99AA_BBCC);
    let mut out_scalar = std::vec![0u16; w * 3];
    let mut out_simd = std::vec![0u16; w * 3];
    scalar::gbrpf32_to_rgb_u16_row::<false>(&g, &b, &r, &mut out_scalar, w);
    unsafe {
      gbrpf32_to_rgb_u16_row::<false>(&g, &b, &r, &mut out_simd, w);
    }
    assert_eq!(out_scalar, out_simd, "wasm gbrpf32_to_rgb_u16 width={w}");
  }
}

// ---- Gbrpf32 → u16 RGBA ------------------------------------------------------

#[test]
fn wasm_gbrpf32_to_rgba_u16_matches_scalar() {
  for w in [1usize, 3, 4, 5, 7, 8, 15, 16, 17, 31, 33, 1920, 1921] {
    let g = gbr_plane_f32(w, 0xCAFE_BABE);
    let b = gbr_plane_f32(w, 0xDEAD_C0DE);
    let r = gbr_plane_f32(w, 0x0F0E_0D0C);
    let mut out_scalar = std::vec![0u16; w * 4];
    let mut out_simd = std::vec![0u16; w * 4];
    scalar::gbrpf32_to_rgba_u16_row::<false>(&g, &b, &r, &mut out_scalar, w);
    unsafe {
      gbrpf32_to_rgba_u16_row::<false>(&g, &b, &r, &mut out_simd, w);
    }
    assert_eq!(out_scalar, out_simd, "wasm gbrpf32_to_rgba_u16 width={w}");
  }
}

// ---- Gbrpf32 → f32 RGB (lossless) -------------------------------------------

#[test]
fn wasm_gbrpf32_to_rgb_f32_matches_scalar() {
  for w in [1usize, 3, 4, 5, 8, 16, 17, 33, 1921] {
    let g = gbr_plane_f32(w, 0x1111_2222);
    let b = gbr_plane_f32(w, 0x3333_4444);
    let r = gbr_plane_f32(w, 0x5555_6666);
    let mut out_scalar = std::vec![0.0f32; w * 3];
    let mut out_simd = std::vec![0.0f32; w * 3];
    scalar::gbrpf32_to_rgb_f32_row::<false>(&g, &b, &r, &mut out_scalar, w);
    unsafe {
      gbrpf32_to_rgb_f32_row::<false>(&g, &b, &r, &mut out_simd, w);
    }
    assert_eq!(out_scalar, out_simd, "wasm gbrpf32_to_rgb_f32 width={w}");
  }
}

// ---- Gbrpf32 → f32 RGBA (lossless) ------------------------------------------

#[test]
fn wasm_gbrpf32_to_rgba_f32_matches_scalar() {
  for w in [1usize, 3, 4, 5, 8, 16, 17, 33, 1921] {
    let g = gbr_plane_f32(w, 0x7777_8888);
    let b = gbr_plane_f32(w, 0x9999_AAAA);
    let r = gbr_plane_f32(w, 0xBBBB_CCCC);
    let mut out_scalar = std::vec![0.0f32; w * 4];
    let mut out_simd = std::vec![0.0f32; w * 4];
    scalar::gbrpf32_to_rgba_f32_row::<false>(&g, &b, &r, &mut out_scalar, w);
    unsafe {
      gbrpf32_to_rgba_f32_row::<false>(&g, &b, &r, &mut out_simd, w);
    }
    assert_eq!(out_scalar, out_simd, "wasm gbrpf32_to_rgba_f32 width={w}");
  }
}

// ---- Gbrpf32 → f16 RGB -------------------------------------------------------

#[test]
fn wasm_gbrpf32_to_rgb_f16_matches_scalar() {
  for w in [1usize, 3, 4, 5, 8, 16, 17, 33, 1921] {
    let g = gbr_plane_f32(w, 0xAABB_CCDD);
    let b = gbr_plane_f32(w, 0xEEFF_0011);
    let r = gbr_plane_f32(w, 0x2233_4455);
    let mut out_scalar = std::vec![half::f16::ZERO; w * 3];
    let mut out_simd = std::vec![half::f16::ZERO; w * 3];
    scalar::gbrpf32_to_rgb_f16_row::<false>(&g, &b, &r, &mut out_scalar, w);
    unsafe {
      gbrpf32_to_rgb_f16_row::<false>(&g, &b, &r, &mut out_simd, w);
    }
    assert_eq!(out_scalar, out_simd, "wasm gbrpf32_to_rgb_f16 width={w}");
  }
}

// ---- Gbrpf32 → f16 RGBA ------------------------------------------------------

#[test]
fn wasm_gbrpf32_to_rgba_f16_matches_scalar() {
  for w in [1usize, 3, 4, 5, 8, 16, 17, 33, 1921] {
    let g = gbr_plane_f32(w, 0x6677_8899);
    let b = gbr_plane_f32(w, 0xAABB_CCDD);
    let r = gbr_plane_f32(w, 0xEEFF_1122);
    let mut out_scalar = std::vec![half::f16::ZERO; w * 4];
    let mut out_simd = std::vec![half::f16::ZERO; w * 4];
    scalar::gbrpf32_to_rgba_f16_row::<false>(&g, &b, &r, &mut out_scalar, w);
    unsafe {
      gbrpf32_to_rgba_f16_row::<false>(&g, &b, &r, &mut out_simd, w);
    }
    assert_eq!(out_scalar, out_simd, "wasm gbrpf32_to_rgba_f16 width={w}");
  }
}

// ---- Gbrpf32 → u8 luma -------------------------------------------------------

#[test]
fn wasm_gbrpf32_to_luma_matches_scalar() {
  use crate::KernelMatrix;
  for w in [1usize, 3, 4, 5, 8, 16, 17, 33, 1921] {
    let g = gbr_plane_f32(w, 0x1A2B_3C4D);
    let b = gbr_plane_f32(w, 0x5E6F_7A8B);
    let r = gbr_plane_f32(w, 0x9CAD_BEEF);
    let mut out_scalar = std::vec![0u8; w];
    let mut out_simd = std::vec![0u8; w];
    scalar::gbrpf32_to_luma_row::<false>(&g, &b, &r, &mut out_scalar, w, KernelMatrix::Bt709, true);
    unsafe {
      gbrpf32_to_luma_row::<false>(&g, &b, &r, &mut out_simd, w, KernelMatrix::Bt709, true);
    }
    assert_eq!(out_scalar, out_simd, "wasm gbrpf32_to_luma width={w}");
  }
}

// ---- Gbrpf32 → u16 luma ------------------------------------------------------

#[test]
fn wasm_gbrpf32_to_luma_u16_matches_scalar() {
  use crate::KernelMatrix;
  for w in [1usize, 3, 4, 5, 8, 16, 17, 33, 1921] {
    let g = gbr_plane_f32(w, 0xF1E2_D3C4);
    let b = gbr_plane_f32(w, 0xB5A6_9788);
    let r = gbr_plane_f32(w, 0x7968_5748);
    let mut out_scalar = std::vec![0u16; w];
    let mut out_simd = std::vec![0u16; w];
    scalar::gbrpf32_to_luma_u16_row::<false>(
      &g,
      &b,
      &r,
      &mut out_scalar,
      w,
      KernelMatrix::Bt709,
      true,
    );
    unsafe {
      gbrpf32_to_luma_u16_row::<false>(&g, &b, &r, &mut out_simd, w, KernelMatrix::Bt709, true);
    }
    assert_eq!(out_scalar, out_simd, "wasm gbrpf32_to_luma_u16 width={w}");
  }
}

// ---- Gbrpf32 → HSV -----------------------------------------------------------

#[test]
fn wasm_gbrpf32_to_hsv_matches_scalar() {
  for w in [1usize, 3, 4, 5, 8, 16, 17, 33, 1921] {
    let g = gbr_plane_f32(w, 0x0102_0304);
    let b = gbr_plane_f32(w, 0x0506_0708);
    let r = gbr_plane_f32(w, 0x090A_0B0C);
    let mut h_scalar = std::vec![0u8; w];
    let mut s_scalar = std::vec![0u8; w];
    let mut v_scalar = std::vec![0u8; w];
    let mut h_simd = std::vec![0u8; w];
    let mut s_simd = std::vec![0u8; w];
    let mut v_simd = std::vec![0u8; w];
    scalar::gbrpf32_to_hsv_row::<false>(&g, &b, &r, &mut h_scalar, &mut s_scalar, &mut v_scalar, w);
    unsafe {
      gbrpf32_to_hsv_row::<false>(&g, &b, &r, &mut h_simd, &mut s_simd, &mut v_simd, w);
    }
    assert_eq!(h_scalar, h_simd, "wasm gbrpf32_to_hsv H width={w}");
    assert_eq!(s_scalar, s_simd, "wasm gbrpf32_to_hsv S width={w}");
    assert_eq!(v_scalar, v_simd, "wasm gbrpf32_to_hsv V width={w}");
  }
}

// ---- Gbrapf32 → u8 RGBA (source α) ------------------------------------------

#[test]
fn wasm_gbrapf32_to_rgba_matches_scalar() {
  for w in [1usize, 3, 4, 5, 7, 8, 15, 16, 17, 31, 33, 1920, 1921] {
    let g = gbr_plane_f32(w, 0xA1B2_C3D4);
    let b = gbr_plane_f32(w, 0xE5F6_0718);
    let r = gbr_plane_f32(w, 0x293A_4B5C);
    let a = gbr_plane_f32(w, 0x6D7E_8F90);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_simd = std::vec![0u8; w * 4];
    scalar::gbrapf32_to_rgba_row::<false>(&g, &b, &r, &a, &mut out_scalar, w);
    unsafe {
      gbrapf32_to_rgba_row::<false>(&g, &b, &r, &a, &mut out_simd, w);
    }
    assert_eq!(out_scalar, out_simd, "wasm gbrapf32_to_rgba width={w}");
  }
}

// ---- Gbrapf32 → u16 RGBA (source α) -----------------------------------------

#[test]
fn wasm_gbrapf32_to_rgba_u16_matches_scalar() {
  for w in [1usize, 3, 4, 5, 7, 8, 15, 16, 17, 31, 33, 1920, 1921] {
    let g = gbr_plane_f32(w, 0xABCD_EF01);
    let b = gbr_plane_f32(w, 0x2345_6789);
    let r = gbr_plane_f32(w, 0xABCD_EF01 ^ 0x5555_5555);
    let a = gbr_plane_f32(w, 0xFEDC_BA98);
    let mut out_scalar = std::vec![0u16; w * 4];
    let mut out_simd = std::vec![0u16; w * 4];
    scalar::gbrapf32_to_rgba_u16_row::<false>(&g, &b, &r, &a, &mut out_scalar, w);
    unsafe {
      gbrapf32_to_rgba_u16_row::<false>(&g, &b, &r, &a, &mut out_simd, w);
    }
    assert_eq!(out_scalar, out_simd, "wasm gbrapf32_to_rgba_u16 width={w}");
  }
}

// ---- Gbrapf32 → f32 RGBA (lossless) -----------------------------------------

#[test]
fn wasm_gbrapf32_to_rgba_f32_matches_scalar() {
  for w in [1usize, 3, 4, 5, 8, 16, 17, 33, 1921] {
    let g = gbr_plane_f32(w, 0x1357_9BDF);
    let b = gbr_plane_f32(w, 0x2468_ACE0);
    let r = gbr_plane_f32(w, 0x0F1E_2D3C);
    let a = gbr_plane_f32(w, 0x4B5A_6978);
    let mut out_scalar = std::vec![0.0f32; w * 4];
    let mut out_simd = std::vec![0.0f32; w * 4];
    scalar::gbrapf32_to_rgba_f32_row::<false>(&g, &b, &r, &a, &mut out_scalar, w);
    unsafe {
      gbrapf32_to_rgba_f32_row::<false>(&g, &b, &r, &a, &mut out_simd, w);
    }
    assert_eq!(out_scalar, out_simd, "wasm gbrapf32_to_rgba_f32 width={w}");
  }
}

// ---- Gbrapf32 → f16 RGBA (source α) -----------------------------------------

#[test]
fn wasm_gbrapf32_to_rgba_f16_matches_scalar() {
  for w in [1usize, 3, 4, 5, 8, 16, 17, 33, 1921] {
    let g = gbr_plane_f32(w, 0x8796_A5B4);
    let b = gbr_plane_f32(w, 0xC3D2_E1F0);
    let r = gbr_plane_f32(w, 0xFEDC_BA98);
    let a = gbr_plane_f32(w, 0x7654_3210);
    let mut out_scalar = std::vec![half::f16::ZERO; w * 4];
    let mut out_simd = std::vec![half::f16::ZERO; w * 4];
    scalar::gbrapf32_to_rgba_f16_row::<false>(&g, &b, &r, &a, &mut out_scalar, w);
    unsafe {
      gbrapf32_to_rgba_f16_row::<false>(&g, &b, &r, &a, &mut out_simd, w);
    }
    assert_eq!(out_scalar, out_simd, "wasm gbrapf32_to_rgba_f16 width={w}");
  }
}

// ---- Gbrpf16 → f16 RGB (lossless) -------------------------------------------

#[test]
fn wasm_gbrpf16_to_rgb_f16_matches_scalar() {
  for w in [1usize, 3, 4, 5, 8, 16, 17, 33, 1921] {
    let g = gbr_plane_f16(w, 0x1234_5678);
    let b = gbr_plane_f16(w, 0x9ABC_DEF0);
    let r = gbr_plane_f16(w, 0xFEDC_BA98);
    let mut out_scalar = std::vec![half::f16::ZERO; w * 3];
    let mut out_simd = std::vec![half::f16::ZERO; w * 3];
    scalar_f16::gbrpf16_to_rgb_f16_row::<false>(&g, &b, &r, &mut out_scalar, w);
    unsafe {
      gbrpf16_to_rgb_f16_row::<false>(&g, &b, &r, &mut out_simd, w);
    }
    assert_eq!(out_scalar, out_simd, "wasm gbrpf16_to_rgb_f16 width={w}");
  }
}

// ---- Gbrpf16 → f16 RGBA (lossless, opaque α = f16(1.0)) --------------------

#[test]
fn wasm_gbrpf16_to_rgba_f16_matches_scalar() {
  for w in [1usize, 3, 4, 5, 8, 16, 17, 33, 1921] {
    let g = gbr_plane_f16(w, 0xABCD_0101);
    let b = gbr_plane_f16(w, 0x2323_4545);
    let r = gbr_plane_f16(w, 0x6767_8989);
    let mut out_scalar = std::vec![half::f16::ZERO; w * 4];
    let mut out_simd = std::vec![half::f16::ZERO; w * 4];
    scalar_f16::gbrpf16_to_rgba_f16_row::<false>(&g, &b, &r, &mut out_scalar, w);
    unsafe {
      gbrpf16_to_rgba_f16_row::<false>(&g, &b, &r, &mut out_simd, w);
    }
    assert_eq!(out_scalar, out_simd, "wasm gbrpf16_to_rgba_f16 width={w}");
  }
}

// ---- Gbrapf16 → f16 RGBA (lossless, source α) --------------------------------

#[test]
fn wasm_gbrapf16_to_rgba_f16_matches_scalar() {
  for w in [1usize, 3, 4, 5, 8, 16, 17, 33, 1921] {
    let g = gbr_plane_f16(w, 0x1111_2222);
    let b = gbr_plane_f16(w, 0x3333_4444);
    let r = gbr_plane_f16(w, 0x5555_6666);
    let a = gbr_plane_f16(w, 0x7777_8888);
    let mut out_scalar = std::vec![half::f16::ZERO; w * 4];
    let mut out_simd = std::vec![half::f16::ZERO; w * 4];
    scalar_f16::gbrapf16_to_rgba_f16_row::<false>(&g, &b, &r, &a, &mut out_scalar, w);
    unsafe {
      gbrapf16_to_rgba_f16_row::<false>(&g, &b, &r, &a, &mut out_simd, w);
    }
    assert_eq!(out_scalar, out_simd, "wasm gbrapf16_to_rgba_f16 width={w}");
  }
}

// ---- Gbrpf16 → u8 RGB --------------------------------------------------------

#[test]
fn wasm_gbrpf16_to_rgb_matches_scalar() {
  for w in [1usize, 3, 4, 5, 7, 8, 15, 16, 17, 31, 33, 1920, 1921] {
    let g = gbr_plane_f16(w, 0xA1A2_B3B4);
    let b = gbr_plane_f16(w, 0xC5C6_D7D8);
    let r = gbr_plane_f16(w, 0xE9EA_FBFC);
    let mut out_scalar = std::vec![0u8; w * 3];
    let mut out_simd = std::vec![0u8; w * 3];
    scalar::gbrpf32_to_rgb_row::<false>(
      &g.iter().map(|v| v.to_f32()).collect::<std::vec::Vec<_>>(),
      &b.iter().map(|v| v.to_f32()).collect::<std::vec::Vec<_>>(),
      &r.iter().map(|v| v.to_f32()).collect::<std::vec::Vec<_>>(),
      &mut out_scalar,
      w,
    );
    unsafe {
      gbrpf16_to_rgb_row::<false>(&g, &b, &r, &mut out_simd, w);
    }
    assert_eq!(out_scalar, out_simd, "wasm gbrpf16_to_rgb width={w}");
  }
}

// ---- Gbrpf16 → u8 RGBA -------------------------------------------------------

#[test]
fn wasm_gbrpf16_to_rgba_matches_scalar() {
  for w in [1usize, 3, 4, 5, 7, 8, 15, 16, 17, 31, 33, 1920, 1921] {
    let g = gbr_plane_f16(w, 0x1234_ABCD);
    let b = gbr_plane_f16(w, 0xEF01_2345);
    let r = gbr_plane_f16(w, 0x6789_CDEF);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_simd = std::vec![0u8; w * 4];
    scalar::gbrpf32_to_rgba_row::<false>(
      &g.iter().map(|v| v.to_f32()).collect::<std::vec::Vec<_>>(),
      &b.iter().map(|v| v.to_f32()).collect::<std::vec::Vec<_>>(),
      &r.iter().map(|v| v.to_f32()).collect::<std::vec::Vec<_>>(),
      &mut out_scalar,
      w,
    );
    unsafe {
      gbrpf16_to_rgba_row::<false>(&g, &b, &r, &mut out_simd, w);
    }
    assert_eq!(out_scalar, out_simd, "wasm gbrpf16_to_rgba width={w}");
  }
}

// ---- Gbrpf16 → u16 RGB -------------------------------------------------------

#[test]
fn wasm_gbrpf16_to_rgb_u16_matches_scalar() {
  for w in [1usize, 3, 4, 5, 7, 8, 15, 16, 17, 31, 33, 1920, 1921] {
    let g = gbr_plane_f16(w, 0x0011_2233);
    let b = gbr_plane_f16(w, 0x4455_6677);
    let r = gbr_plane_f16(w, 0x8899_AABB);
    let mut out_scalar = std::vec![0u16; w * 3];
    let mut out_simd = std::vec![0u16; w * 3];
    scalar::gbrpf32_to_rgb_u16_row::<false>(
      &g.iter().map(|v| v.to_f32()).collect::<std::vec::Vec<_>>(),
      &b.iter().map(|v| v.to_f32()).collect::<std::vec::Vec<_>>(),
      &r.iter().map(|v| v.to_f32()).collect::<std::vec::Vec<_>>(),
      &mut out_scalar,
      w,
    );
    unsafe {
      gbrpf16_to_rgb_u16_row::<false>(&g, &b, &r, &mut out_simd, w);
    }
    assert_eq!(out_scalar, out_simd, "wasm gbrpf16_to_rgb_u16 width={w}");
  }
}

// ---- Gbrpf16 → u16 RGBA ------------------------------------------------------

#[test]
fn wasm_gbrpf16_to_rgba_u16_matches_scalar() {
  for w in [1usize, 3, 4, 5, 7, 8, 15, 16, 17, 31, 33, 1920, 1921] {
    let g = gbr_plane_f16(w, 0xCCDD_EEFF);
    let b = gbr_plane_f16(w, 0x0011_AABB);
    let r = gbr_plane_f16(w, 0xCC22_4466);
    let mut out_scalar = std::vec![0u16; w * 4];
    let mut out_simd = std::vec![0u16; w * 4];
    scalar::gbrpf32_to_rgba_u16_row::<false>(
      &g.iter().map(|v| v.to_f32()).collect::<std::vec::Vec<_>>(),
      &b.iter().map(|v| v.to_f32()).collect::<std::vec::Vec<_>>(),
      &r.iter().map(|v| v.to_f32()).collect::<std::vec::Vec<_>>(),
      &mut out_scalar,
      w,
    );
    unsafe {
      gbrpf16_to_rgba_u16_row::<false>(&g, &b, &r, &mut out_simd, w);
    }
    assert_eq!(out_scalar, out_simd, "wasm gbrpf16_to_rgba_u16 width={w}");
  }
}

// ---- Round-half-up correctness (wasm uses + 0.5 / trunc, not banker's) ------

#[test]
fn wasm_gbrpf32_to_rgb_round_half_up() {
  // 0.5/255 = exactly halfway between 0 and 1.
  // Banker's rounding → 0; round-half-up → 1.
  // 1.5/255 → 1 (banker's RNE rounds to 2); round-half-up → 2.
  // 2.5/255 → 2 (banker's RNE rounds to 2); round-half-up → 3.
  let vals = std::vec![0.5f32 / 255.0, 1.5 / 255.0, 2.5 / 255.0, 3.5 / 255.0];
  let g = vals.clone();
  let b = vals.clone();
  let r = vals.clone();
  let w = 4;
  let mut out_scalar = std::vec![0u8; w * 3];
  let mut out_simd = std::vec![0u8; w * 3];
  scalar::gbrpf32_to_rgb_row::<false>(&g, &b, &r, &mut out_scalar, w);
  unsafe {
    gbrpf32_to_rgb_row::<false>(&g, &b, &r, &mut out_simd, w);
  }
  // Both scalar and SIMD must use round-half-up (1, 2, 3, 4).
  assert_eq!(out_scalar, out_simd, "round-half-up: scalar vs SIMD");
  assert_eq!(out_simd[0], 1, "0.5/255 → 1 (round-half-up)");
  assert_eq!(out_simd[3], 2, "1.5/255 → 2 (round-half-up)");
  assert_eq!(out_simd[6], 3, "2.5/255 → 3 (round-half-up)");
  assert_eq!(out_simd[9], 4, "3.5/255 → 4 (round-half-up)");
}

// ---- BE parity helpers -------------------------------------------------------

fn be_encode_f32(src: &[f32]) -> std::vec::Vec<f32> {
  src
    .iter()
    .map(|v| f32::from_bits(v.to_bits().swap_bytes()))
    .collect()
}

fn be_encode_f16(src: &[half::f16]) -> std::vec::Vec<half::f16> {
  src
    .iter()
    .map(|v| half::f16::from_bits(v.to_bits().swap_bytes()))
    .collect()
}

// ---- BE parity: Gbrpf32 → u8 RGB -------------------------------------------

#[test]
fn wasm_gbrpf32_to_rgb_be_parity() {
  for w in [1usize, 4, 5, 7, 8, 16, 17, 32] {
    let g = gbr_plane_f32(w, 0xBE01_0001);
    let b = gbr_plane_f32(w, 0xBE01_0002);
    let r = gbr_plane_f32(w, 0xBE01_0003);
    let mut le_out = std::vec![0u8; w * 3];
    let mut be_out = std::vec![0u8; w * 3];
    unsafe {
      gbrpf32_to_rgb_row::<false>(&g, &b, &r, &mut le_out, w);
    }
    let g_be = be_encode_f32(&g);
    let b_be = be_encode_f32(&b);
    let r_be = be_encode_f32(&r);
    unsafe {
      gbrpf32_to_rgb_row::<true>(&g_be, &b_be, &r_be, &mut be_out, w);
    }
    assert_eq!(le_out, be_out, "wasm gbrpf32_to_rgb BE parity width={w}");
  }
}

// ---- BE parity: Gbrpf16 → f16 RGB (lossless) --------------------------------

#[test]
fn wasm_gbrpf16_to_rgb_f16_be_parity() {
  for w in [1usize, 4, 5, 7, 8, 16, 17, 32] {
    let g = gbr_plane_f16(w, 0xBE07_0001);
    let b = gbr_plane_f16(w, 0xBE07_0002);
    let r = gbr_plane_f16(w, 0xBE07_0003);
    let mut le_out = std::vec![half::f16::ZERO; w * 3];
    let mut be_out = std::vec![half::f16::ZERO; w * 3];
    unsafe {
      gbrpf16_to_rgb_f16_row::<false>(&g, &b, &r, &mut le_out, w);
    }
    let g_be = be_encode_f16(&g);
    let b_be = be_encode_f16(&b);
    let r_be = be_encode_f16(&r);
    unsafe {
      gbrpf16_to_rgb_f16_row::<true>(&g_be, &b_be, &r_be, &mut be_out, w);
    }
    assert_eq!(
      le_out, be_out,
      "wasm gbrpf16_to_rgb_f16 BE parity width={w}"
    );
  }
}

// ---- BE parity: Gbrapf16 → f16 RGBA (lossless) ------------------------------

#[test]
fn wasm_gbrapf16_to_rgba_f16_be_parity() {
  for w in [1usize, 4, 5, 7, 8, 16, 17, 32] {
    let g = gbr_plane_f16(w, 0xBE0F_0001);
    let b = gbr_plane_f16(w, 0xBE0F_0002);
    let r = gbr_plane_f16(w, 0xBE0F_0003);
    let a = gbr_plane_f16(w, 0xBE0F_0004);
    let mut le_out = std::vec![half::f16::ZERO; w * 4];
    let mut be_out = std::vec![half::f16::ZERO; w * 4];
    unsafe {
      gbrapf16_to_rgba_f16_row::<false>(&g, &b, &r, &a, &mut le_out, w);
    }
    let g_be = be_encode_f16(&g);
    let b_be = be_encode_f16(&b);
    let r_be = be_encode_f16(&r);
    let a_be = be_encode_f16(&a);
    unsafe {
      gbrapf16_to_rgba_f16_row::<true>(&g_be, &b_be, &r_be, &a_be, &mut be_out, w);
    }
    assert_eq!(
      le_out, be_out,
      "wasm gbrapf16_to_rgba_f16 BE parity width={w}"
    );
  }
}
