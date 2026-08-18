//! WASM simd128 parity tests for the Tier 12 (DCP / Xyz12) kernels.
//!
//! Every test asserts byte-identical output against the scalar
//! reference (0-ULP parity contract since SIMD calls the same scalar
//! `oetf_srgb` polynomial). simd128 is a compile-time gate (no runtime
//! detection), so tests run unconditionally inside a wasm build.

use crate::{
  KernelGamut,
  row::{
    arch::wasm_simd128::xyz12::{
      xyz12_to_rgb_f16_row, xyz12_to_rgb_f32_row, xyz12_to_rgb_row, xyz12_to_rgb_u16_row,
      xyz12_to_rgba_f16_row, xyz12_to_rgba_row, xyz12_to_rgba_u16_row, xyz12_to_xyz_f32_row,
    },
    scalar,
  },
};

const WIDTHS: &[usize] = &[1, 4, 7, 16, 33, 1920];

/// Pseudo-random 12-bit-active XYZ12 plane in the **high-bit-packed**
/// LE wire convention (FFmpeg `AV_PIX_FMT_XYZ12LE`: code in `[15:4]`,
/// low 4 bits zero).
fn xyz12_plane(width: usize, seed: u32) -> std::vec::Vec<u16> {
  let mut state = seed;
  (0..width * 3)
    .map(|_| {
      state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
      let code = (state & 0x0FFF) as u16;
      u16::from_ne_bytes((code << 4).to_le_bytes())
    })
    .collect()
}

fn xyz12_plane_dirty(width: usize, seed: u32) -> std::vec::Vec<u16> {
  let mut state = seed;
  (0..width * 3)
    .map(|i| {
      state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
      let code = (state & 0x0FFF) as u16;
      let clean = u16::from_ne_bytes((code << 4).to_le_bytes());
      let dirt: u16 = if i % 2 == 0 { 0x000F } else { 0x000A };
      clean | u16::from_ne_bytes(dirt.to_le_bytes())
    })
    .collect()
}

fn byte_swap_vec(v: &[u16]) -> std::vec::Vec<u16> {
  v.iter().map(|x| x.swap_bytes()).collect()
}

// ---- u8 RGB --------------------------------------------------------------

#[test]
fn wasm_xyz12_to_rgb_matches_scalar() {
  for &w in WIDTHS {
    for gamut in [
      KernelGamut::DciP3,
      KernelGamut::Rec709,
      KernelGamut::Rec2020,
    ] {
      let xyz = xyz12_plane(w, 0xC0FE_BABE);
      let mut out_scalar = std::vec![0u8; w * 3];
      let mut out_simd = std::vec![0u8; w * 3];
      scalar::xyz12::xyz12_to_rgb_row::<false>(&xyz, &mut out_scalar, w, gamut);
      unsafe {
        xyz12_to_rgb_row::<false>(&xyz, &mut out_simd, w, gamut);
      }
      assert_eq!(
        out_scalar, out_simd,
        "wasm xyz12_to_rgb diverges (w={w}, gamut={gamut:?})"
      );
    }
  }
}

#[test]
fn wasm_xyz12_to_rgb_dirty_input_matches_scalar() {
  for &w in WIDTHS {
    let xyz = xyz12_plane_dirty(w, 0xDEAD_F00D);
    let mut out_scalar = std::vec![0u8; w * 3];
    let mut out_simd = std::vec![0u8; w * 3];
    scalar::xyz12::xyz12_to_rgb_row::<false>(&xyz, &mut out_scalar, w, KernelGamut::DciP3);
    unsafe {
      xyz12_to_rgb_row::<false>(&xyz, &mut out_simd, w, KernelGamut::DciP3);
    }
    assert_eq!(
      out_scalar, out_simd,
      "wasm xyz12_to_rgb dirty-input diverges (w={w})"
    );
  }
}

#[test]
fn wasm_xyz12_to_rgb_be_matches_le() {
  for &w in WIDTHS {
    let xyz_le = xyz12_plane(w, 0x1234_5678);
    let xyz_be = byte_swap_vec(&xyz_le);
    let mut out_le = std::vec![0u8; w * 3];
    let mut out_be = std::vec![0u8; w * 3];
    unsafe {
      xyz12_to_rgb_row::<false>(&xyz_le, &mut out_le, w, KernelGamut::Rec709);
      xyz12_to_rgb_row::<true>(&xyz_be, &mut out_be, w, KernelGamut::Rec709);
    }
    assert_eq!(out_le, out_be, "wasm xyz12_to_rgb BE/LE mismatch (w={w})");
  }
}

// ---- u8 RGBA -------------------------------------------------------------

#[test]
fn wasm_xyz12_to_rgba_matches_scalar() {
  for &w in WIDTHS {
    let xyz = xyz12_plane(w, 0xAFAF_AFAF);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_simd = std::vec![0u8; w * 4];
    scalar::xyz12::xyz12_to_rgba_row::<false>(&xyz, &mut out_scalar, w, KernelGamut::Rec2020);
    unsafe {
      xyz12_to_rgba_row::<false>(&xyz, &mut out_simd, w, KernelGamut::Rec2020);
    }
    assert_eq!(out_scalar, out_simd, "wasm xyz12_to_rgba diverges (w={w})");
  }
}

// ---- u16 RGB / RGBA -----------------------------------------------------

#[test]
fn wasm_xyz12_to_rgb_u16_matches_scalar() {
  for &w in WIDTHS {
    let xyz = xyz12_plane(w, 0xFEED_FACE);
    let mut out_scalar = std::vec![0u16; w * 3];
    let mut out_simd = std::vec![0u16; w * 3];
    scalar::xyz12::xyz12_to_rgb_u16_row::<false>(&xyz, &mut out_scalar, w, KernelGamut::DciP3);
    unsafe {
      xyz12_to_rgb_u16_row::<false>(&xyz, &mut out_simd, w, KernelGamut::DciP3);
    }
    assert_eq!(
      out_scalar, out_simd,
      "wasm xyz12_to_rgb_u16 diverges (w={w})"
    );
  }
}

#[test]
fn wasm_xyz12_to_rgba_u16_matches_scalar() {
  for &w in WIDTHS {
    let xyz = xyz12_plane(w, 0xCAFE_F00D);
    let mut out_scalar = std::vec![0u16; w * 4];
    let mut out_simd = std::vec![0u16; w * 4];
    scalar::xyz12::xyz12_to_rgba_u16_row::<false>(&xyz, &mut out_scalar, w, KernelGamut::Rec709);
    unsafe {
      xyz12_to_rgba_u16_row::<false>(&xyz, &mut out_simd, w, KernelGamut::Rec709);
    }
    assert_eq!(
      out_scalar, out_simd,
      "wasm xyz12_to_rgba_u16 diverges (w={w})"
    );
  }
}

// ---- f32 outputs --------------------------------------------------------

#[test]
fn wasm_xyz12_to_rgb_f32_matches_scalar() {
  for &w in WIDTHS {
    let xyz = xyz12_plane(w, 0x600D_C0DE);
    let mut out_scalar = std::vec![0.0_f32; w * 3];
    let mut out_simd = std::vec![0.0_f32; w * 3];
    scalar::xyz12::xyz12_to_rgb_f32_row::<false>(&xyz, &mut out_scalar, w, KernelGamut::Rec2020);
    unsafe {
      xyz12_to_rgb_f32_row::<false>(&xyz, &mut out_simd, w, KernelGamut::Rec2020);
    }
    assert_eq!(
      out_scalar, out_simd,
      "wasm xyz12_to_rgb_f32 diverges (w={w})"
    );
  }
}

#[test]
fn wasm_xyz12_to_xyz_f32_matches_scalar() {
  for &w in WIDTHS {
    let xyz = xyz12_plane(w, 0xD00D_F00D);
    let mut out_scalar = std::vec![0.0_f32; w * 3];
    let mut out_simd = std::vec![0.0_f32; w * 3];
    scalar::xyz12::xyz12_to_xyz_f32_row::<false>(&xyz, &mut out_scalar, w);
    unsafe {
      xyz12_to_xyz_f32_row::<false>(&xyz, &mut out_simd, w);
    }
    assert_eq!(
      out_scalar, out_simd,
      "wasm xyz12_to_xyz_f32 diverges (w={w})"
    );
  }
}

// ---- f16 outputs --------------------------------------------------------

#[test]
fn wasm_xyz12_to_rgb_f16_matches_scalar() {
  for &w in WIDTHS {
    let xyz = xyz12_plane(w, 0xBEEF_CAFE);
    let zero_f16 = half::f16::from_f32(0.0);
    let mut out_scalar = std::vec![zero_f16; w * 3];
    let mut out_simd = std::vec![zero_f16; w * 3];
    scalar::xyz12::xyz12_to_rgb_f16_row::<false>(&xyz, &mut out_scalar, w, KernelGamut::DciP3);
    unsafe {
      xyz12_to_rgb_f16_row::<false>(&xyz, &mut out_simd, w, KernelGamut::DciP3);
    }
    assert_eq!(
      out_scalar, out_simd,
      "wasm xyz12_to_rgb_f16 diverges (w={w})"
    );
  }
}

#[test]
fn wasm_xyz12_to_rgba_f16_matches_scalar() {
  for &w in WIDTHS {
    let xyz = xyz12_plane(w, 0xF1F1_F1F1);
    let zero_f16 = half::f16::from_f32(0.0);
    let mut out_scalar = std::vec![zero_f16; w * 4];
    let mut out_simd = std::vec![zero_f16; w * 4];
    scalar::xyz12::xyz12_to_rgba_f16_row::<false>(&xyz, &mut out_scalar, w, KernelGamut::Rec709);
    unsafe {
      xyz12_to_rgba_f16_row::<false>(&xyz, &mut out_simd, w, KernelGamut::Rec709);
    }
    assert_eq!(
      out_scalar, out_simd,
      "wasm xyz12_to_rgba_f16 diverges (w={w})"
    );
  }
}
