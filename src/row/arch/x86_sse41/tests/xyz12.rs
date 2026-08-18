//! SSE4.1 parity tests for the Tier 12 (DCP / Xyz12) kernels.
//!
//! Every test asserts byte-identical output against the scalar
//! reference (0-ULP parity contract) and early-returns when SSE4.1 is
//! unavailable on the host (sanitizer / CI safety).

use super::super::*;
use crate::{
  KernelGamut,
  row::arch::x86_sse41::xyz12::{
    xyz12_to_rgb_f16_row, xyz12_to_rgb_f32_row, xyz12_to_rgb_row, xyz12_to_rgb_u16_row,
    xyz12_to_rgba_f16_row, xyz12_to_rgba_row, xyz12_to_rgba_u16_row, xyz12_to_xyz_f32_row,
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

/// Same as `xyz12_plane` but with the reserved low 4 bits dirtied —
/// every kernel must `>> 4` after the endian-aware load and discard
/// the garbage.
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
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn sse41_xyz12_to_rgb_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  for &w in WIDTHS {
    for gamut in [
      KernelGamut::DciP3,
      KernelGamut::Rec709,
      KernelGamut::Rec2020,
    ] {
      let xyz = xyz12_plane(w, 0xC0FE_BABE);
      let mut out_scalar = std::vec![0u8; w * 3];
      let mut out_sse = std::vec![0u8; w * 3];
      scalar::xyz12::xyz12_to_rgb_row::<false>(&xyz, &mut out_scalar, w, gamut);
      unsafe {
        xyz12_to_rgb_row::<false>(&xyz, &mut out_sse, w, gamut);
      }
      assert_eq!(
        out_scalar, out_sse,
        "SSE4.1 xyz12_to_rgb diverges (w={w}, gamut={gamut:?})"
      );
    }
  }
}

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn sse41_xyz12_to_rgb_dirty_input_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  for &w in WIDTHS {
    let xyz = xyz12_plane_dirty(w, 0xDEAD_F00D);
    let mut out_scalar = std::vec![0u8; w * 3];
    let mut out_sse = std::vec![0u8; w * 3];
    scalar::xyz12::xyz12_to_rgb_row::<false>(&xyz, &mut out_scalar, w, KernelGamut::DciP3);
    unsafe {
      xyz12_to_rgb_row::<false>(&xyz, &mut out_sse, w, KernelGamut::DciP3);
    }
    assert_eq!(
      out_scalar, out_sse,
      "SSE4.1 xyz12_to_rgb dirty-input diverges (w={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn sse41_xyz12_to_rgb_be_matches_le() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  for &w in WIDTHS {
    let xyz_le = xyz12_plane(w, 0x1234_5678);
    let xyz_be = byte_swap_vec(&xyz_le);
    let mut out_le = std::vec![0u8; w * 3];
    let mut out_be = std::vec![0u8; w * 3];
    unsafe {
      xyz12_to_rgb_row::<false>(&xyz_le, &mut out_le, w, KernelGamut::Rec709);
      xyz12_to_rgb_row::<true>(&xyz_be, &mut out_be, w, KernelGamut::Rec709);
    }
    assert_eq!(out_le, out_be, "SSE4.1 xyz12_to_rgb BE/LE mismatch (w={w})");
  }
}

// ---- u8 RGBA -------------------------------------------------------------

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn sse41_xyz12_to_rgba_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  for &w in WIDTHS {
    let xyz = xyz12_plane(w, 0xAFAF_AFAF);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_sse = std::vec![0u8; w * 4];
    scalar::xyz12::xyz12_to_rgba_row::<false>(&xyz, &mut out_scalar, w, KernelGamut::Rec2020);
    unsafe {
      xyz12_to_rgba_row::<false>(&xyz, &mut out_sse, w, KernelGamut::Rec2020);
    }
    assert_eq!(out_scalar, out_sse, "SSE4.1 xyz12_to_rgba diverges (w={w})");
  }
}

// In-register store regression coverage.
//
// The u8 RGB / RGBA store paths use in-register `_mm_shuffle_epi8` (RGB)
// / `_mm_unpacklo_epi8` (RGBA) interleave instead of a 3× stack-temp +
// per-pixel scalar scatter. These tests pin block-multiple widths
// (16, 32) that hit the SIMD fast path exclusively (no scalar tail),
// confirming byte-identical output against the scalar reference.

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn sse41_xyz12_to_rgb_in_register_store_parity() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  for &w in &[16usize, 32] {
    for gamut in [
      KernelGamut::DciP3,
      KernelGamut::Rec709,
      KernelGamut::Rec2020,
    ] {
      let xyz = xyz12_plane(w, 0x5101_5101);
      let mut out_scalar = std::vec![0u8; w * 3];
      let mut out_sse = std::vec![0u8; w * 3];
      scalar::xyz12::xyz12_to_rgb_row::<false>(&xyz, &mut out_scalar, w, gamut);
      unsafe {
        xyz12_to_rgb_row::<false>(&xyz, &mut out_sse, w, gamut);
      }
      assert_eq!(
        out_scalar, out_sse,
        "SSE4.1 xyz12_to_rgb in-register store parity (w={w}, gamut={gamut:?})"
      );
    }
  }
}

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn sse41_xyz12_to_rgba_in_register_store_parity() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  for &w in &[16usize, 32] {
    for gamut in [
      KernelGamut::DciP3,
      KernelGamut::Rec709,
      KernelGamut::Rec2020,
    ] {
      let xyz = xyz12_plane(w, 0x5202_5202);
      let mut out_scalar = std::vec![0u8; w * 4];
      let mut out_sse = std::vec![0u8; w * 4];
      scalar::xyz12::xyz12_to_rgba_row::<false>(&xyz, &mut out_scalar, w, gamut);
      unsafe {
        xyz12_to_rgba_row::<false>(&xyz, &mut out_sse, w, gamut);
      }
      assert_eq!(
        out_scalar, out_sse,
        "SSE4.1 xyz12_to_rgba in-register store parity (w={w}, gamut={gamut:?})"
      );
    }
  }
}

// ---- u16 RGB / RGBA -----------------------------------------------------

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn sse41_xyz12_to_rgb_u16_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  for &w in WIDTHS {
    let xyz = xyz12_plane(w, 0xFEED_FACE);
    let mut out_scalar = std::vec![0u16; w * 3];
    let mut out_sse = std::vec![0u16; w * 3];
    scalar::xyz12::xyz12_to_rgb_u16_row::<false>(&xyz, &mut out_scalar, w, KernelGamut::DciP3);
    unsafe {
      xyz12_to_rgb_u16_row::<false>(&xyz, &mut out_sse, w, KernelGamut::DciP3);
    }
    assert_eq!(
      out_scalar, out_sse,
      "SSE4.1 xyz12_to_rgb_u16 diverges (w={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn sse41_xyz12_to_rgba_u16_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  for &w in WIDTHS {
    let xyz = xyz12_plane(w, 0xCAFE_F00D);
    let mut out_scalar = std::vec![0u16; w * 4];
    let mut out_sse = std::vec![0u16; w * 4];
    scalar::xyz12::xyz12_to_rgba_u16_row::<false>(&xyz, &mut out_scalar, w, KernelGamut::Rec709);
    unsafe {
      xyz12_to_rgba_u16_row::<false>(&xyz, &mut out_sse, w, KernelGamut::Rec709);
    }
    assert_eq!(
      out_scalar, out_sse,
      "SSE4.1 xyz12_to_rgba_u16 diverges (w={w})"
    );
  }
}

// ---- f32 outputs --------------------------------------------------------

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn sse41_xyz12_to_rgb_f32_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  for &w in WIDTHS {
    let xyz = xyz12_plane(w, 0x600D_C0DE);
    let mut out_scalar = std::vec![0.0_f32; w * 3];
    let mut out_sse = std::vec![0.0_f32; w * 3];
    scalar::xyz12::xyz12_to_rgb_f32_row::<false>(&xyz, &mut out_scalar, w, KernelGamut::Rec2020);
    unsafe {
      xyz12_to_rgb_f32_row::<false>(&xyz, &mut out_sse, w, KernelGamut::Rec2020);
    }
    assert_eq!(
      out_scalar, out_sse,
      "SSE4.1 xyz12_to_rgb_f32 diverges (w={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn sse41_xyz12_to_xyz_f32_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  for &w in WIDTHS {
    let xyz = xyz12_plane(w, 0xD00D_F00D);
    let mut out_scalar = std::vec![0.0_f32; w * 3];
    let mut out_sse = std::vec![0.0_f32; w * 3];
    scalar::xyz12::xyz12_to_xyz_f32_row::<false>(&xyz, &mut out_scalar, w);
    unsafe {
      xyz12_to_xyz_f32_row::<false>(&xyz, &mut out_sse, w);
    }
    assert_eq!(
      out_scalar, out_sse,
      "SSE4.1 xyz12_to_xyz_f32 diverges (w={w})"
    );
  }
}

// ---- f16 outputs --------------------------------------------------------

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn sse41_xyz12_to_rgb_f16_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  for &w in WIDTHS {
    let xyz = xyz12_plane(w, 0xBEEF_CAFE);
    let zero_f16 = half::f16::from_f32(0.0);
    let mut out_scalar = std::vec![zero_f16; w * 3];
    let mut out_sse = std::vec![zero_f16; w * 3];
    scalar::xyz12::xyz12_to_rgb_f16_row::<false>(&xyz, &mut out_scalar, w, KernelGamut::DciP3);
    unsafe {
      xyz12_to_rgb_f16_row::<false>(&xyz, &mut out_sse, w, KernelGamut::DciP3);
    }
    assert_eq!(
      out_scalar, out_sse,
      "SSE4.1 xyz12_to_rgb_f16 diverges (w={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn sse41_xyz12_to_rgba_f16_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  for &w in WIDTHS {
    let xyz = xyz12_plane(w, 0xF1F1_F1F1);
    let zero_f16 = half::f16::from_f32(0.0);
    let mut out_scalar = std::vec![zero_f16; w * 4];
    let mut out_sse = std::vec![zero_f16; w * 4];
    scalar::xyz12::xyz12_to_rgba_f16_row::<false>(&xyz, &mut out_scalar, w, KernelGamut::Rec709);
    unsafe {
      xyz12_to_rgba_f16_row::<false>(&xyz, &mut out_sse, w, KernelGamut::Rec709);
    }
    assert_eq!(
      out_scalar, out_sse,
      "SSE4.1 xyz12_to_rgba_f16 diverges (w={w})"
    );
  }
}
