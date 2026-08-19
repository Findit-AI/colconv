//! BE parity tests for NEON high-bit YUV / P-format kernels.
//!
//! Each test constructs the LE and BE input buffers from raw bytes via
//! `to_le_bytes` / `to_be_bytes` (then `from_ne_bytes` to reinterpret
//! as host-native `u16`), so the byte-level encoding is host-independent:
//! on every host (LE or BE), `*_le` carries the intended u16 values as
//! LE-encoded bytes and `*_be` carries the same intended values as
//! BE-encoded bytes. Both kernels therefore decode to the intended
//! host-native u16 samples and must produce byte-identical output.
//!
//! The naive `swap_bytes` pattern is vacuous on BE hosts (both
//! `kernel::<false>` and `kernel::<true>` produce equal-but-wrong
//! outputs and the assert passes without exercising the BE-host
//! decode path).

use crate::row::neon_available;

use super::{super::*, p16_plane_neon};
// Planar high-bit plane builder — used only by the `yuv-planar`-gated
// planar BE-parity tests below.
#[cfg(feature = "yuv-planar")]
use super::planar_n_plane;
// Semi-planar (`p_n` / `p_n_444`) BE-parity helpers — used only by the
// `yuv-semi-planar`-gated tests below.
#[cfg(feature = "yuv-semi-planar")]
use super::{high_bit_plane, interleave_uv, p_n_packed_plane, p010_uv_interleave};

/// Reinterpret an intended-u16 buffer as host-native `u16` carrying
/// LE-encoded bytes. On LE hosts this is the identity; on BE hosts it
/// stores each value with its bytes swapped vs. host-native order.
fn as_le_u16_buf(buf: &[u16]) -> std::vec::Vec<u16> {
  buf
    .iter()
    .map(|v| u16::from_ne_bytes(v.to_le_bytes()))
    .collect()
}

/// Reinterpret an intended-u16 buffer as host-native `u16` carrying
/// BE-encoded bytes. On BE hosts this is the identity; on LE hosts it
/// stores each value with its bytes swapped vs. host-native order.
fn as_be_u16_buf(buf: &[u16]) -> std::vec::Vec<u16> {
  buf
    .iter()
    .map(|v| u16::from_ne_bytes(v.to_be_bytes()))
    .collect()
}

// ---- yuv_420p_n (planar 4:2:0 high-bit) -----------------------------

#[cfg(feature = "yuv-planar")]
#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_yuv_420p10_be_parity_u8() {
  if !neon_available() {
    return;
  }
  let width = 32;
  let y_intended = planar_n_plane::<10>(width, 13);
  let u_intended = planar_n_plane::<10>(width / 2, 17);
  let v_intended = planar_n_plane::<10>(width / 2, 19);
  let y_le = as_le_u16_buf(&y_intended);
  let u_le = as_le_u16_buf(&u_intended);
  let v_le = as_le_u16_buf(&v_intended);
  let y_be = as_be_u16_buf(&y_intended);
  let u_be = as_be_u16_buf(&u_intended);
  let v_be = as_be_u16_buf(&v_intended);

  for matrix in [KernelMatrix::Bt709, KernelMatrix::Bt2020Ncl] {
    for full_range in [true, false] {
      let mut out_le = std::vec![0u8; width * 3];
      let mut out_be = std::vec![0u8; width * 3];
      unsafe {
        yuv_420p_n_to_rgb_row::<10, false>(
          &y_le,
          &u_le,
          &v_le,
          &mut out_le,
          width,
          matrix,
          full_range,
        );
        yuv_420p_n_to_rgb_row::<10, true>(
          &y_be,
          &u_be,
          &v_be,
          &mut out_be,
          width,
          matrix,
          full_range,
        );
      }
      assert_eq!(out_le, out_be, "matrix={matrix:?} full_range={full_range}");
    }
  }
}

#[cfg(feature = "yuv-planar")]
#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_yuv_420p10_be_parity_u16() {
  if !neon_available() {
    return;
  }
  let width = 32;
  let y_intended = planar_n_plane::<10>(width, 23);
  let u_intended = planar_n_plane::<10>(width / 2, 29);
  let v_intended = planar_n_plane::<10>(width / 2, 31);
  let y_le = as_le_u16_buf(&y_intended);
  let u_le = as_le_u16_buf(&u_intended);
  let v_le = as_le_u16_buf(&v_intended);
  let y_be = as_be_u16_buf(&y_intended);
  let u_be = as_be_u16_buf(&u_intended);
  let v_be = as_be_u16_buf(&v_intended);

  let mut out_le = std::vec![0u16; width * 3];
  let mut out_be = std::vec![0u16; width * 3];
  unsafe {
    yuv_420p_n_to_rgb_u16_row::<10, false>(
      &y_le,
      &u_le,
      &v_le,
      &mut out_le,
      width,
      KernelMatrix::Bt709,
      true,
    );
    yuv_420p_n_to_rgb_u16_row::<10, true>(
      &y_be,
      &u_be,
      &v_be,
      &mut out_be,
      width,
      KernelMatrix::Bt709,
      true,
    );
  }
  assert_eq!(out_le, out_be);
}

// ---- yuv_444p_n (planar 4:4:4 high-bit) -----------------------------

#[cfg(feature = "yuv-planar")]
#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_yuv_444p12_be_parity_u8() {
  if !neon_available() {
    return;
  }
  let width = 32;
  let y_intended = planar_n_plane::<12>(width, 41);
  let u_intended = planar_n_plane::<12>(width, 43);
  let v_intended = planar_n_plane::<12>(width, 47);
  let y_le = as_le_u16_buf(&y_intended);
  let u_le = as_le_u16_buf(&u_intended);
  let v_le = as_le_u16_buf(&v_intended);
  let y_be = as_be_u16_buf(&y_intended);
  let u_be = as_be_u16_buf(&u_intended);
  let v_be = as_be_u16_buf(&v_intended);

  let mut out_le = std::vec![0u8; width * 3];
  let mut out_be = std::vec![0u8; width * 3];
  unsafe {
    yuv_444p_n_to_rgb_row::<12, false>(
      &y_le,
      &u_le,
      &v_le,
      &mut out_le,
      width,
      KernelMatrix::Bt709,
      true,
    );
    yuv_444p_n_to_rgb_row::<12, true>(
      &y_be,
      &u_be,
      &v_be,
      &mut out_be,
      width,
      KernelMatrix::Bt709,
      true,
    );
  }
  assert_eq!(out_le, out_be);
}

// ---- yuv_*p16 (16-bit planar) ----------------------------------------

#[cfg(feature = "yuv-planar")]
#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_yuv_420p16_be_parity_u8() {
  if !neon_available() {
    return;
  }
  let width = 32;
  let y_intended = p16_plane_neon(width, 53);
  let u_intended = p16_plane_neon(width / 2, 59);
  let v_intended = p16_plane_neon(width / 2, 61);
  let y_le = as_le_u16_buf(&y_intended);
  let u_le = as_le_u16_buf(&u_intended);
  let v_le = as_le_u16_buf(&v_intended);
  let y_be = as_be_u16_buf(&y_intended);
  let u_be = as_be_u16_buf(&u_intended);
  let v_be = as_be_u16_buf(&v_intended);

  let mut out_le = std::vec![0u8; width * 3];
  let mut out_be = std::vec![0u8; width * 3];
  unsafe {
    yuv_420p16_to_rgb_row::<false>(
      &y_le,
      &u_le,
      &v_le,
      &mut out_le,
      width,
      KernelMatrix::Bt709,
      true,
    );
    yuv_420p16_to_rgb_row::<true>(
      &y_be,
      &u_be,
      &v_be,
      &mut out_be,
      width,
      KernelMatrix::Bt709,
      true,
    );
  }
  assert_eq!(out_le, out_be);
}

#[cfg(feature = "yuv-planar")]
#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_yuv_444p16_be_parity_u16() {
  if !neon_available() {
    return;
  }
  let width = 32;
  let y_intended = p16_plane_neon(width, 67);
  let u_intended = p16_plane_neon(width, 71);
  let v_intended = p16_plane_neon(width, 73);
  let y_le = as_le_u16_buf(&y_intended);
  let u_le = as_le_u16_buf(&u_intended);
  let v_le = as_le_u16_buf(&v_intended);
  let y_be = as_be_u16_buf(&y_intended);
  let u_be = as_be_u16_buf(&u_intended);
  let v_be = as_be_u16_buf(&v_intended);

  let mut out_le = std::vec![0u16; width * 3];
  let mut out_be = std::vec![0u16; width * 3];
  unsafe {
    yuv_444p16_to_rgb_u16_row::<false>(
      &y_le,
      &u_le,
      &v_le,
      &mut out_le,
      width,
      KernelMatrix::Bt709,
      true,
    );
    yuv_444p16_to_rgb_u16_row::<true>(
      &y_be,
      &u_be,
      &v_be,
      &mut out_be,
      width,
      KernelMatrix::Bt709,
      true,
    );
  }
  assert_eq!(out_le, out_be);
}

// p_n / p_n_444 (semi-planar high-bit-packed).
//
// The 4:2:0 (`p_n_to_*`) and 4:4:4 (`p_n_444_to_*`) NEON kernels
// deinterleave UV via `vld2q_u16`, which materializes lanes in host-native
// order. Their per-lane byte-swap therefore must trigger on
// `BE != HOST_NATIVE_BE`, not just `BE`. These regression tests pin that
// gate against a BE-host miscompile.

#[cfg(feature = "yuv-semi-planar")]
#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_p010_be_parity_u8() {
  if !neon_available() {
    return;
  }
  let width = 32;
  let y_intended = p_n_packed_plane::<10>(width, 79);
  let u_half = p_n_packed_plane::<10>(width / 2, 83);
  let v_half = p_n_packed_plane::<10>(width / 2, 89);
  let uv_intended = p010_uv_interleave(&u_half, &v_half);
  let y_le = as_le_u16_buf(&y_intended);
  let uv_le = as_le_u16_buf(&uv_intended);
  let y_be = as_be_u16_buf(&y_intended);
  let uv_be = as_be_u16_buf(&uv_intended);

  let mut out_le = std::vec![0u8; width * 3];
  let mut out_be = std::vec![0u8; width * 3];
  unsafe {
    p_n_to_rgb_row::<10, false>(&y_le, &uv_le, &mut out_le, width, KernelMatrix::Bt709, true);
    p_n_to_rgb_row::<10, true>(&y_be, &uv_be, &mut out_be, width, KernelMatrix::Bt709, true);
  }
  assert_eq!(out_le, out_be);
}

#[cfg(feature = "yuv-semi-planar")]
#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_p012_be_parity_u16() {
  if !neon_available() {
    return;
  }
  // Native-depth u16 output exercises `p_n_to_rgb_or_rgba_u16_row`,
  // the second `vld2q_u16` + `deinterleave_endian::<BE>` site in
  // `subsampled_high_bit_pn_4_2_0.rs`.
  let width = 32;
  let y_intended = p_n_packed_plane::<12>(width, 149);
  let u_half = p_n_packed_plane::<12>(width / 2, 151);
  let v_half = p_n_packed_plane::<12>(width / 2, 157);
  let uv_intended = p010_uv_interleave(&u_half, &v_half);
  let y_le = as_le_u16_buf(&y_intended);
  let uv_le = as_le_u16_buf(&uv_intended);
  let y_be = as_be_u16_buf(&y_intended);
  let uv_be = as_be_u16_buf(&uv_intended);

  let mut out_le = std::vec![0u16; width * 3];
  let mut out_be = std::vec![0u16; width * 3];
  unsafe {
    p_n_to_rgb_u16_row::<12, false>(&y_le, &uv_le, &mut out_le, width, KernelMatrix::Bt709, true);
    p_n_to_rgb_u16_row::<12, true>(&y_be, &uv_be, &mut out_be, width, KernelMatrix::Bt709, true);
  }
  assert_eq!(out_le, out_be);
}

#[cfg(feature = "yuv-semi-planar")]
#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_p410_be_parity_u8() {
  if !neon_available() {
    return;
  }
  let width = 32;
  let y_intended = p_n_packed_plane::<10>(width, 97);
  let u_full = high_bit_plane::<10>(width, 101);
  let v_full = high_bit_plane::<10>(width, 103);
  let uv_intended = interleave_uv(&u_full, &v_full);
  let y_le = as_le_u16_buf(&y_intended);
  let uv_le = as_le_u16_buf(&uv_intended);
  let y_be = as_be_u16_buf(&y_intended);
  let uv_be = as_be_u16_buf(&uv_intended);

  let mut out_le = std::vec![0u8; width * 3];
  let mut out_be = std::vec![0u8; width * 3];
  unsafe {
    p_n_444_to_rgb_row::<10, false>(&y_le, &uv_le, &mut out_le, width, KernelMatrix::Bt709, true);
    p_n_444_to_rgb_row::<10, true>(&y_be, &uv_be, &mut out_be, width, KernelMatrix::Bt709, true);
  }
  assert_eq!(out_le, out_be);
}

#[cfg(feature = "yuv-semi-planar")]
#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_p412_be_parity_u16() {
  if !neon_available() {
    return;
  }
  // Native-depth u16 output exercises `p_n_444_to_rgb_or_rgba_u16_row`,
  // the second `vld2q_u16` + `deinterleave_endian::<BE>` site in
  // `subsampled_high_bit_pn_4_4_4.rs`.
  let width = 32;
  let y_intended = p_n_packed_plane::<12>(width, 163);
  let u_full = high_bit_plane::<12>(width, 167);
  let v_full = high_bit_plane::<12>(width, 173);
  let uv_intended = interleave_uv(&u_full, &v_full);
  let y_le = as_le_u16_buf(&y_intended);
  let uv_le = as_le_u16_buf(&uv_intended);
  let y_be = as_be_u16_buf(&y_intended);
  let uv_be = as_be_u16_buf(&uv_intended);

  let mut out_le = std::vec![0u16; width * 3];
  let mut out_be = std::vec![0u16; width * 3];
  unsafe {
    p_n_444_to_rgb_u16_row::<12, false>(
      &y_le,
      &uv_le,
      &mut out_le,
      width,
      KernelMatrix::Bt709,
      true,
    );
    p_n_444_to_rgb_u16_row::<12, true>(
      &y_be,
      &uv_be,
      &mut out_be,
      width,
      KernelMatrix::Bt709,
      true,
    );
  }
  assert_eq!(out_le, out_be);
}

#[cfg(feature = "yuv-semi-planar")]
#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_p016_be_parity_u8() {
  if !neon_available() {
    return;
  }
  let width = 32;
  let y_intended = p16_plane_neon(width, 107);
  let u_half = p16_plane_neon(width / 2, 109);
  let v_half = p16_plane_neon(width / 2, 113);
  let uv_intended = p010_uv_interleave(&u_half, &v_half);
  let y_le = as_le_u16_buf(&y_intended);
  let uv_le = as_le_u16_buf(&uv_intended);
  let y_be = as_be_u16_buf(&y_intended);
  let uv_be = as_be_u16_buf(&uv_intended);

  let mut out_le = std::vec![0u8; width * 3];
  let mut out_be = std::vec![0u8; width * 3];
  unsafe {
    p16_to_rgb_row::<false>(&y_le, &uv_le, &mut out_le, width, KernelMatrix::Bt709, true);
    p16_to_rgb_row::<true>(&y_be, &uv_be, &mut out_be, width, KernelMatrix::Bt709, true);
  }
  assert_eq!(out_le, out_be);
}

#[cfg(feature = "yuv-semi-planar")]
#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_p016_be_parity_u16() {
  if !neon_available() {
    return;
  }
  // Native-depth u16 output exercises `p16_to_rgb_or_rgba_u16_row`,
  // a third `vld2q_u16` + `deinterleave_endian::<BE>` site (line 716).
  let width = 32;
  let y_intended = p16_plane_neon(width, 179);
  let u_half = p16_plane_neon(width / 2, 181);
  let v_half = p16_plane_neon(width / 2, 191);
  let uv_intended = p010_uv_interleave(&u_half, &v_half);
  let y_le = as_le_u16_buf(&y_intended);
  let uv_le = as_le_u16_buf(&uv_intended);
  let y_be = as_be_u16_buf(&y_intended);
  let uv_be = as_be_u16_buf(&uv_intended);

  let mut out_le = std::vec![0u16; width * 3];
  let mut out_be = std::vec![0u16; width * 3];
  unsafe {
    p16_to_rgb_u16_row::<false>(&y_le, &uv_le, &mut out_le, width, KernelMatrix::Bt709, true);
    p16_to_rgb_u16_row::<true>(&y_be, &uv_be, &mut out_be, width, KernelMatrix::Bt709, true);
  }
  assert_eq!(out_le, out_be);
}

#[cfg(feature = "yuv-semi-planar")]
#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_p416_be_parity_u16() {
  if !neon_available() {
    return;
  }
  let width = 32;
  let y_intended = p16_plane_neon(width, 127);
  let u_full = p16_plane_neon(width, 131);
  let v_full = p16_plane_neon(width, 137);
  let uv_intended = interleave_uv(&u_full, &v_full);
  let y_le = as_le_u16_buf(&y_intended);
  let uv_le = as_le_u16_buf(&uv_intended);
  let y_be = as_be_u16_buf(&y_intended);
  let uv_be = as_be_u16_buf(&uv_intended);

  let mut out_le = std::vec![0u16; width * 3];
  let mut out_be = std::vec![0u16; width * 3];
  unsafe {
    p_n_444_16_to_rgb_u16_row::<false>(
      &y_le,
      &uv_le,
      &mut out_le,
      width,
      KernelMatrix::Bt709,
      true,
    );
    p_n_444_16_to_rgb_u16_row::<true>(&y_be, &uv_be, &mut out_be, width, KernelMatrix::Bt709, true);
  }
  assert_eq!(out_le, out_be);
}
