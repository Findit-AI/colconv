//! NEON parity tests for the Tier 12 (DCP / Xyz12) kernels.
//!
//! Every test asserts SIMD output is **byte-identical** to the scalar
//! reference. The 0-ULP parity contract relies on:
//!
//! 1. The per-lane scalar `smpte428_inverse_oetf` and `oetf_srgb`
//!    being *the same scalar function* both paths call, and
//! 2. The vectorized matmul + clamp + narrow being agnostic to lane
//!    arrangement (per-lane f32 ops, not horizontal reductions).
//!
//! Test matrix (per output × per gamut):
//! - widths: `[1, 4, 7, 16, 33, 1920]` (mix of small / SIMD-aligned /
//!   non-aligned / row-realistic).
//! - gamuts: `DciP3`, `Rec709`, `Rec2020` — at least one per output.
//! - BE: `false` and `true` parity test for the rgb-u8 path.

use super::*;
use crate::{
  KernelGamut,
  row::arch::neon::xyz12::{
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
      // Encode as `code << 4` per the FFmpeg high-bit-packed convention.
      // Host-independent: `from_ne_bytes(...to_le_bytes())` reinterprets
      // the LE wire bytes as a host-native `u16`, matching the BE
      // helper's `from_ne_bytes(...to_be_bytes())` pattern.
      u16::from_ne_bytes((code << 4).to_le_bytes())
    })
    .collect()
}

/// Mirrors the dirty-input pattern from the planar-gbr tests: a
/// non-spec-compliant producer that sets bits `[3:0]` (reserved zero
/// per the FFmpeg `AV_PIX_FMT_XYZ12LE` spec) must still produce the
/// same output as its clean counterpart — every kernel applies `>> 4`
/// after the endian-aware load.
fn xyz12_plane_dirty(width: usize, seed: u32) -> std::vec::Vec<u16> {
  let mut state = seed;
  (0..width * 3)
    .map(|i| {
      state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
      let code = (state & 0x0FFF) as u16;
      let clean = u16::from_ne_bytes((code << 4).to_le_bytes());
      // Set the reserved low 4 bits with arbitrary garbage on the wire.
      // Encode `dirt` as LE bytes reinterpreted as host-native so the
      // OR lands in the wire's low byte's low nibble on every host.
      let dirt: u16 = if i % 2 == 0 { 0x000F } else { 0x000A };
      clean | u16::from_ne_bytes(dirt.to_le_bytes())
    })
    .collect()
}

/// Returns the BE-wire counterpart of an LE-wire fixture: byte-swap
/// every `u16`. Host-independent: the kernel's `from_be` undoes the
/// swap to recover the original host-native LE encoding.
fn byte_swap_vec(v: &[u16]) -> std::vec::Vec<u16> {
  v.iter().map(|x| x.swap_bytes()).collect()
}

/// Encodes a 12-bit code in the BE-wire layout for direct host-
/// independent BE input fixtures (used by the BE-host parity test).
#[cfg_attr(not(tarpaulin), inline(always))]
fn pack12_be(code: u16) -> u16 {
  u16::from_ne_bytes((code << 4).to_be_bytes())
}

// ---- u8 RGB --------------------------------------------------------------

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_xyz12_to_rgb_matches_scalar() {
  for &w in WIDTHS {
    for gamut in [
      KernelGamut::DciP3,
      KernelGamut::Rec709,
      KernelGamut::Rec2020,
    ] {
      let xyz = xyz12_plane(w, 0xC0FE_BABE);
      let mut out_scalar = std::vec![0u8; w * 3];
      let mut out_neon = std::vec![0u8; w * 3];
      scalar::xyz12::xyz12_to_rgb_row::<false>(&xyz, &mut out_scalar, w, gamut);
      unsafe {
        xyz12_to_rgb_row::<false>(&xyz, &mut out_neon, w, gamut);
      }
      assert_eq!(
        out_scalar, out_neon,
        "NEON xyz12_to_rgb diverges (w={w}, gamut={gamut:?})"
      );
    }
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_xyz12_to_rgb_dirty_input_matches_scalar() {
  for &w in WIDTHS {
    let xyz = xyz12_plane_dirty(w, 0xDEAD_F00D);
    let mut out_scalar = std::vec![0u8; w * 3];
    let mut out_neon = std::vec![0u8; w * 3];
    scalar::xyz12::xyz12_to_rgb_row::<false>(&xyz, &mut out_scalar, w, KernelGamut::DciP3);
    unsafe {
      xyz12_to_rgb_row::<false>(&xyz, &mut out_neon, w, KernelGamut::DciP3);
    }
    assert_eq!(
      out_scalar, out_neon,
      "NEON xyz12_to_rgb dirty-input diverges (w={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_xyz12_to_rgb_be_matches_le() {
  for &w in WIDTHS {
    let xyz_le = xyz12_plane(w, 0x1234_5678);
    let xyz_be = byte_swap_vec(&xyz_le);
    let mut out_le = std::vec![0u8; w * 3];
    let mut out_be = std::vec![0u8; w * 3];
    unsafe {
      xyz12_to_rgb_row::<false>(&xyz_le, &mut out_le, w, KernelGamut::Rec709);
      xyz12_to_rgb_row::<true>(&xyz_be, &mut out_be, w, KernelGamut::Rec709);
    }
    assert_eq!(out_le, out_be, "NEON xyz12_to_rgb BE/LE mismatch (w={w})");
  }
}

/// Host-independent NEON-vs-scalar parity over BE-wire input. On an
/// LE host this exercises the new `BE != HOST_NATIVE_BE` scalar
/// fallback path; on a BE host (e.g. `s390x_unknown_linux_gnu` miri,
/// `aarch64_be`) it exercises the NEON SIMD body. Catches the prior
/// `if BE { swap }` bug (which corrupted both BE-host paths) by
/// asserting bit-exact scalar/SIMD parity regardless of host.
#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_xyz12_to_rgb_be_wire_matches_scalar() {
  for &w in WIDTHS {
    // Build a fixture of host-independent BE-wire u16 codes.
    let mut state: u32 = 0x9E37_79B9;
    let xyz_be: std::vec::Vec<u16> = (0..w * 3)
      .map(|_| {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        pack12_be((state & 0x0FFF) as u16)
      })
      .collect();
    let mut out_scalar = std::vec![0u8; w * 3];
    let mut out_neon = std::vec![0u8; w * 3];
    scalar::xyz12::xyz12_to_rgb_row::<true>(&xyz_be, &mut out_scalar, w, KernelGamut::DciP3);
    unsafe {
      xyz12_to_rgb_row::<true>(&xyz_be, &mut out_neon, w, KernelGamut::DciP3);
    }
    assert_eq!(
      out_scalar, out_neon,
      "NEON BE-wire xyz12_to_rgb diverges from scalar (w={w})"
    );
  }
}

// ---- u8 RGBA -------------------------------------------------------------

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_xyz12_to_rgba_matches_scalar() {
  for &w in WIDTHS {
    let xyz = xyz12_plane(w, 0xAFAF_AFAF);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_neon = std::vec![0u8; w * 4];
    scalar::xyz12::xyz12_to_rgba_row::<false>(&xyz, &mut out_scalar, w, KernelGamut::Rec2020);
    unsafe {
      xyz12_to_rgba_row::<false>(&xyz, &mut out_neon, w, KernelGamut::Rec2020);
    }
    assert_eq!(out_scalar, out_neon, "NEON xyz12_to_rgba diverges (w={w})");
  }
}

// ---- u16 RGB / RGBA -----------------------------------------------------

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_xyz12_to_rgb_u16_matches_scalar() {
  for &w in WIDTHS {
    let xyz = xyz12_plane(w, 0xFEED_FACE);
    let mut out_scalar = std::vec![0u16; w * 3];
    let mut out_neon = std::vec![0u16; w * 3];
    scalar::xyz12::xyz12_to_rgb_u16_row::<false>(&xyz, &mut out_scalar, w, KernelGamut::DciP3);
    unsafe {
      xyz12_to_rgb_u16_row::<false>(&xyz, &mut out_neon, w, KernelGamut::DciP3);
    }
    assert_eq!(
      out_scalar, out_neon,
      "NEON xyz12_to_rgb_u16 diverges (w={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_xyz12_to_rgba_u16_matches_scalar() {
  for &w in WIDTHS {
    let xyz = xyz12_plane(w, 0xCAFE_F00D);
    let mut out_scalar = std::vec![0u16; w * 4];
    let mut out_neon = std::vec![0u16; w * 4];
    scalar::xyz12::xyz12_to_rgba_u16_row::<false>(&xyz, &mut out_scalar, w, KernelGamut::Rec709);
    unsafe {
      xyz12_to_rgba_u16_row::<false>(&xyz, &mut out_neon, w, KernelGamut::Rec709);
    }
    assert_eq!(
      out_scalar, out_neon,
      "NEON xyz12_to_rgba_u16 diverges (w={w})"
    );
  }
}

// ---- f32 outputs --------------------------------------------------------

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_xyz12_to_rgb_f32_matches_scalar() {
  for &w in WIDTHS {
    let xyz = xyz12_plane(w, 0x600D_C0DE);
    let mut out_scalar = std::vec![0.0_f32; w * 3];
    let mut out_neon = std::vec![0.0_f32; w * 3];
    scalar::xyz12::xyz12_to_rgb_f32_row::<false>(&xyz, &mut out_scalar, w, KernelGamut::Rec2020);
    unsafe {
      xyz12_to_rgb_f32_row::<false>(&xyz, &mut out_neon, w, KernelGamut::Rec2020);
    }
    // 0-ULP parity contract: SIMD matmul uses plain mul+add (not FMA),
    // matching the scalar's exact rounding schedule, so f32 output
    // must be bit-exact identical lane-by-lane.
    assert_eq!(
      out_scalar, out_neon,
      "NEON xyz12_to_rgb_f32 diverges (w={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_xyz12_to_xyz_f32_matches_scalar() {
  for &w in WIDTHS {
    let xyz = xyz12_plane(w, 0xD00D_F00D);
    let mut out_scalar = std::vec![0.0_f32; w * 3];
    let mut out_neon = std::vec![0.0_f32; w * 3];
    scalar::xyz12::xyz12_to_xyz_f32_row::<false>(&xyz, &mut out_scalar, w);
    unsafe {
      xyz12_to_xyz_f32_row::<false>(&xyz, &mut out_neon, w);
    }
    // No matmul on this path — every f32 lane comes from the same
    // scalar `smpte428_inverse_oetf`, so equality is bit-exact.
    assert_eq!(
      out_scalar, out_neon,
      "NEON xyz12_to_xyz_f32 diverges (w={w})"
    );
  }
}

// ---- f16 outputs --------------------------------------------------------

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_xyz12_to_rgb_f16_matches_scalar() {
  for &w in WIDTHS {
    let xyz = xyz12_plane(w, 0xBEEF_CAFE);
    let zero_f16 = half::f16::from_f32(0.0);
    let mut out_scalar = std::vec![zero_f16; w * 3];
    let mut out_neon = std::vec![zero_f16; w * 3];
    scalar::xyz12::xyz12_to_rgb_f16_row::<false>(&xyz, &mut out_scalar, w, KernelGamut::DciP3);
    unsafe {
      xyz12_to_rgb_f16_row::<false>(&xyz, &mut out_neon, w, KernelGamut::DciP3);
    }
    assert_eq!(
      out_scalar, out_neon,
      "NEON xyz12_to_rgb_f16 diverges (w={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_xyz12_to_rgba_f16_matches_scalar() {
  for &w in WIDTHS {
    let xyz = xyz12_plane(w, 0xF1F1_F1F1);
    let zero_f16 = half::f16::from_f32(0.0);
    let mut out_scalar = std::vec![zero_f16; w * 4];
    let mut out_neon = std::vec![zero_f16; w * 4];
    scalar::xyz12::xyz12_to_rgba_f16_row::<false>(&xyz, &mut out_scalar, w, KernelGamut::Rec709);
    unsafe {
      xyz12_to_rgba_f16_row::<false>(&xyz, &mut out_neon, w, KernelGamut::Rec709);
    }
    assert_eq!(
      out_scalar, out_neon,
      "NEON xyz12_to_rgba_f16 diverges (w={w})"
    );
  }
}
