//! BE parity tests for SSE4.1 high-bit YUV / P-format kernels.
//!
//! Each test takes a randomized LE input buffer, byte-swaps every u16
//! element to produce a BE-encoded buffer, then asserts that
//! `kernel::<BE = true>(swapped_input)` produces byte-identical output
//! to `kernel::<BE = false>(original_input)`.

use super::{super::*, p16_plane};
// Planar high-bit plane builder — used only by the `yuv-planar`-gated
// planar BE-parity tests below.
#[cfg(feature = "yuv-planar")]
use super::planar_n_plane;
// Semi-planar (`p_n` / `p_n_444`) BE-parity helpers — used only by the
// `yuv-semi-planar`-gated tests below.
#[cfg(feature = "yuv-semi-planar")]
use super::{high_bit_plane_sse41, interleave_uv_sse41, p_n_packed_plane, p010_uv_interleave};

fn byteswap_u16_buf(buf: &[u16]) -> std::vec::Vec<u16> {
  buf.iter().map(|x| x.swap_bytes()).collect()
}

#[cfg(feature = "yuv-planar")]
#[test]
fn sse41_yuv_420p10_be_parity_u8() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  let width = 32;
  let y = planar_n_plane::<10>(width, 13);
  let u = planar_n_plane::<10>(width / 2, 17);
  let v = planar_n_plane::<10>(width / 2, 19);
  let y_be = byteswap_u16_buf(&y);
  let u_be = byteswap_u16_buf(&u);
  let v_be = byteswap_u16_buf(&v);

  let mut out_le = std::vec![0u8; width * 3];
  let mut out_be = std::vec![0u8; width * 3];
  unsafe {
    yuv_420p_n_to_rgb_row::<10, false>(&y, &u, &v, &mut out_le, width, KernelMatrix::Bt709, true);
    yuv_420p_n_to_rgb_row::<10, true>(
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
fn sse41_yuv_420p10_be_parity_u16() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  let width = 32;
  let y = planar_n_plane::<10>(width, 23);
  let u = planar_n_plane::<10>(width / 2, 29);
  let v = planar_n_plane::<10>(width / 2, 31);
  let y_be = byteswap_u16_buf(&y);
  let u_be = byteswap_u16_buf(&u);
  let v_be = byteswap_u16_buf(&v);

  let mut out_le = std::vec![0u16; width * 3];
  let mut out_be = std::vec![0u16; width * 3];
  unsafe {
    yuv_420p_n_to_rgb_u16_row::<10, false>(
      &y,
      &u,
      &v,
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

#[cfg(feature = "yuv-planar")]
#[test]
fn sse41_yuv_444p12_be_parity_u8() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  let width = 32;
  let y = planar_n_plane::<12>(width, 41);
  let u = planar_n_plane::<12>(width, 43);
  let v = planar_n_plane::<12>(width, 47);
  let y_be = byteswap_u16_buf(&y);
  let u_be = byteswap_u16_buf(&u);
  let v_be = byteswap_u16_buf(&v);

  let mut out_le = std::vec![0u8; width * 3];
  let mut out_be = std::vec![0u8; width * 3];
  unsafe {
    yuv_444p_n_to_rgb_row::<12, false>(&y, &u, &v, &mut out_le, width, KernelMatrix::Bt709, true);
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

#[cfg(feature = "yuv-planar")]
#[test]
fn sse41_yuv_420p16_be_parity_u8() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  let width = 32;
  let y = p16_plane(width, 53);
  let u = p16_plane(width / 2, 59);
  let v = p16_plane(width / 2, 61);
  let y_be = byteswap_u16_buf(&y);
  let u_be = byteswap_u16_buf(&u);
  let v_be = byteswap_u16_buf(&v);

  let mut out_le = std::vec![0u8; width * 3];
  let mut out_be = std::vec![0u8; width * 3];
  unsafe {
    yuv_420p16_to_rgb_row::<false>(&y, &u, &v, &mut out_le, width, KernelMatrix::Bt709, true);
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
fn sse41_yuv_444p16_be_parity_u16() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  let width = 32;
  let y = p16_plane(width, 67);
  let u = p16_plane(width, 71);
  let v = p16_plane(width, 73);
  let y_be = byteswap_u16_buf(&y);
  let u_be = byteswap_u16_buf(&u);
  let v_be = byteswap_u16_buf(&v);

  let mut out_le = std::vec![0u16; width * 3];
  let mut out_be = std::vec![0u16; width * 3];
  unsafe {
    yuv_444p16_to_rgb_u16_row::<false>(&y, &u, &v, &mut out_le, width, KernelMatrix::Bt709, true);
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

#[cfg(feature = "yuv-semi-planar")]
#[test]
fn sse41_p010_be_parity_u8() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  let width = 32;
  let y = p_n_packed_plane::<10>(width, 79);
  let u_half = p_n_packed_plane::<10>(width / 2, 83);
  let v_half = p_n_packed_plane::<10>(width / 2, 89);
  let uv_half = p010_uv_interleave(&u_half, &v_half);
  let y_be = byteswap_u16_buf(&y);
  let uv_be = byteswap_u16_buf(&uv_half);

  let mut out_le = std::vec![0u8; width * 3];
  let mut out_be = std::vec![0u8; width * 3];
  unsafe {
    p_n_to_rgb_row::<10, false>(&y, &uv_half, &mut out_le, width, KernelMatrix::Bt709, true);
    p_n_to_rgb_row::<10, true>(&y_be, &uv_be, &mut out_be, width, KernelMatrix::Bt709, true);
  }
  assert_eq!(out_le, out_be);
}

#[cfg(feature = "yuv-semi-planar")]
#[test]
fn sse41_p410_be_parity_u8() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  let width = 32;
  let y = p_n_packed_plane::<10>(width, 97);
  let u_full = high_bit_plane_sse41::<10>(width, 101);
  let v_full = high_bit_plane_sse41::<10>(width, 103);
  let uv_full = interleave_uv_sse41(&u_full, &v_full);
  let y_be = byteswap_u16_buf(&y);
  let uv_be = byteswap_u16_buf(&uv_full);

  let mut out_le = std::vec![0u8; width * 3];
  let mut out_be = std::vec![0u8; width * 3];
  unsafe {
    p_n_444_to_rgb_row::<10, false>(&y, &uv_full, &mut out_le, width, KernelMatrix::Bt709, true);
    p_n_444_to_rgb_row::<10, true>(&y_be, &uv_be, &mut out_be, width, KernelMatrix::Bt709, true);
  }
  assert_eq!(out_le, out_be);
}

#[cfg(feature = "yuv-semi-planar")]
#[test]
fn sse41_p016_be_parity_u8() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  let width = 32;
  let y = p16_plane(width, 107);
  let u_half = p16_plane(width / 2, 109);
  let v_half = p16_plane(width / 2, 113);
  let uv_half = p010_uv_interleave(&u_half, &v_half);
  let y_be = byteswap_u16_buf(&y);
  let uv_be = byteswap_u16_buf(&uv_half);

  let mut out_le = std::vec![0u8; width * 3];
  let mut out_be = std::vec![0u8; width * 3];
  unsafe {
    p16_to_rgb_row::<false>(&y, &uv_half, &mut out_le, width, KernelMatrix::Bt709, true);
    p16_to_rgb_row::<true>(&y_be, &uv_be, &mut out_be, width, KernelMatrix::Bt709, true);
  }
  assert_eq!(out_le, out_be);
}

#[cfg(feature = "yuv-semi-planar")]
#[test]
fn sse41_p416_be_parity_u16() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  let width = 32;
  let y = p16_plane(width, 127);
  let u_full = p16_plane(width, 131);
  let v_full = p16_plane(width, 137);
  let uv_full = interleave_uv_sse41(&u_full, &v_full);
  let y_be = byteswap_u16_buf(&y);
  let uv_be = byteswap_u16_buf(&uv_full);

  let mut out_le = std::vec![0u16; width * 3];
  let mut out_be = std::vec![0u16; width * 3];
  unsafe {
    p_n_444_16_to_rgb_u16_row::<false>(&y, &uv_full, &mut out_le, width, KernelMatrix::Bt709, true);
    p_n_444_16_to_rgb_u16_row::<true>(&y_be, &uv_be, &mut out_be, width, KernelMatrix::Bt709, true);
  }
  assert_eq!(out_le, out_be);
}

// BE-input SIMD-vs-scalar parity for X2RGB10 / X2BGR10.
//
// Without a BE-aware load (`x2_load_endian_u32x4::<BE>`) the SSE4.1 X2
// 10-bit kernels gate their SIMD body on `if !BE` and silently fall
// through to scalar for BE input. The width sweep below crosses the
// 16-pixel boundary so a regression to scalar fallback would still pass
// the parity check (since it would compare scalar to scalar) — but the
// test also asserts against a hand-built deterministic input where each
// byte has a distinct position, exercising every byte-swap lane.
//
// X2RGB10 / X2BGR10 are packed-RGB formats, so the helper and tests are
// gated on `rgb` (their `scalar::x2*` / SIMD `x2*` kernels live in the
// `rgb`-gated `packed_rgb` modules and are absent in a YUV-only build).

#[cfg(feature = "rgb")]
fn x2_packed_input(width: usize, seed: u32) -> std::vec::Vec<u8> {
  let mut state = seed;
  let mut out = std::vec::Vec::with_capacity(width * 4);
  for _ in 0..width * 4 {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    out.push((state >> 17) as u8);
  }
  out
}

#[cfg(feature = "rgb")]
#[test]
fn sse41_x2rgb10_to_rgb_be_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  for w in [1usize, 7, 15, 16, 17, 31, 32, 33, 1920, 1921] {
    let input = x2_packed_input(w, 0xC0DE_BEEF);
    let mut out_scalar = std::vec![0u8; w * 3];
    let mut out_simd = std::vec![0u8; w * 3];
    scalar::x2rgb10_to_rgb_row::<true>(&input, &mut out_scalar, w);
    unsafe {
      x2rgb10_to_rgb_row::<true>(&input, &mut out_simd, w);
    }
    assert_eq!(
      out_scalar, out_simd,
      "SSE4.1 x2rgb10_to_rgb<BE> diverges (width={w})"
    );
  }
}

#[cfg(feature = "rgb")]
#[test]
fn sse41_x2rgb10_to_rgba_be_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  for w in [1usize, 7, 15, 16, 17, 31, 32, 33, 1920, 1921] {
    let input = x2_packed_input(w, 0xFEED_FACE);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_simd = std::vec![0u8; w * 4];
    scalar::x2rgb10_to_rgba_row::<true>(&input, &mut out_scalar, w);
    unsafe {
      x2rgb10_to_rgba_row::<true>(&input, &mut out_simd, w);
    }
    assert_eq!(
      out_scalar, out_simd,
      "SSE4.1 x2rgb10_to_rgba<BE> diverges (width={w})"
    );
  }
}

#[cfg(feature = "rgb")]
#[test]
fn sse41_x2rgb10_to_rgb_u16_be_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  for w in [1usize, 7, 8, 9, 15, 16, 17, 31, 32, 33, 1920, 1921] {
    let input = x2_packed_input(w, 0xDEAD_C0DE);
    let mut out_scalar = std::vec![0u16; w * 3];
    let mut out_simd = std::vec![0u16; w * 3];
    scalar::x2rgb10_to_rgb_u16_row::<true>(&input, &mut out_scalar, w);
    unsafe {
      x2rgb10_to_rgb_u16_row::<true>(&input, &mut out_simd, w);
    }
    assert_eq!(
      out_scalar, out_simd,
      "SSE4.1 x2rgb10_to_rgb_u16<BE> diverges (width={w})"
    );
  }
}

#[cfg(feature = "rgb")]
#[test]
fn sse41_x2bgr10_to_rgb_be_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  for w in [1usize, 7, 15, 16, 17, 31, 32, 33, 1920, 1921] {
    let input = x2_packed_input(w, 0xCAFE_F00D);
    let mut out_scalar = std::vec![0u8; w * 3];
    let mut out_simd = std::vec![0u8; w * 3];
    scalar::x2bgr10_to_rgb_row::<true>(&input, &mut out_scalar, w);
    unsafe {
      x2bgr10_to_rgb_row::<true>(&input, &mut out_simd, w);
    }
    assert_eq!(
      out_scalar, out_simd,
      "SSE4.1 x2bgr10_to_rgb<BE> diverges (width={w})"
    );
  }
}

#[cfg(feature = "rgb")]
#[test]
fn sse41_x2bgr10_to_rgba_be_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  for w in [1usize, 7, 15, 16, 17, 31, 32, 33, 1920, 1921] {
    let input = x2_packed_input(w, 0xBA0B_AB1E);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_simd = std::vec![0u8; w * 4];
    scalar::x2bgr10_to_rgba_row::<true>(&input, &mut out_scalar, w);
    unsafe {
      x2bgr10_to_rgba_row::<true>(&input, &mut out_simd, w);
    }
    assert_eq!(
      out_scalar, out_simd,
      "SSE4.1 x2bgr10_to_rgba<BE> diverges (width={w})"
    );
  }
}

#[cfg(feature = "rgb")]
#[test]
fn sse41_x2bgr10_to_rgb_u16_be_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  for w in [1usize, 7, 8, 9, 15, 16, 17, 31, 32, 33, 1920, 1921] {
    let input = x2_packed_input(w, 0xACE0_FACE);
    let mut out_scalar = std::vec![0u16; w * 3];
    let mut out_simd = std::vec![0u16; w * 3];
    scalar::x2bgr10_to_rgb_u16_row::<true>(&input, &mut out_scalar, w);
    unsafe {
      x2bgr10_to_rgb_u16_row::<true>(&input, &mut out_simd, w);
    }
    assert_eq!(
      out_scalar, out_simd,
      "SSE4.1 x2bgr10_to_rgb_u16<BE> diverges (width={w})"
    );
  }
}
