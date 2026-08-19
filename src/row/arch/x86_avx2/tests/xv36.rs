use super::super::*;
use crate::{KernelMatrix, row::scalar};

/// Build a deterministic pseudo-random XV36 quadruple stream.
/// Each channel is a 12-bit value MSB-aligned in u16 (low 4 bits zero).
fn pseudo_random_xv36(width: usize, seed: usize) -> std::vec::Vec<u16> {
  (0..width * 4)
    .map(|i| {
      let s = i.wrapping_mul(seed).wrapping_add(seed * 3);
      ((s & 0xFFF) << 4) as u16 // 12-bit value, MSB-aligned
    })
    .collect()
}

fn check_rgb<const ALPHA: bool>(width: usize, matrix: KernelMatrix, full_range: bool) {
  let p = pseudo_random_xv36(width, 0xAA55);
  let bpp = if ALPHA { 4 } else { 3 };
  let mut s = std::vec![0u8; width * bpp];
  let mut k = std::vec![0u8; width * bpp];
  scalar::xv36_to_rgb_or_rgba_row::<ALPHA, false>(&p, &mut s, width, matrix, full_range);
  unsafe {
    xv36_to_rgb_or_rgba_row::<ALPHA, false>(&p, &mut k, width, matrix, full_range);
  }
  assert_eq!(
    s,
    k,
    "AVX2 xv36<ALPHA={ALPHA}>→{} diverges (width={width}, matrix={matrix:?}, full_range={full_range})",
    if ALPHA { "RGBA" } else { "RGB" }
  );
}

fn check_rgb_u16<const ALPHA: bool>(width: usize, matrix: KernelMatrix, full_range: bool) {
  let p = pseudo_random_xv36(width, 0xAA55);
  let bpp = if ALPHA { 4 } else { 3 };
  let mut s = std::vec![0u16; width * bpp];
  let mut k = std::vec![0u16; width * bpp];
  scalar::xv36_to_rgb_u16_or_rgba_u16_row::<ALPHA, false>(&p, &mut s, width, matrix, full_range);
  unsafe {
    xv36_to_rgb_u16_or_rgba_u16_row::<ALPHA, false>(&p, &mut k, width, matrix, full_range);
  }
  assert_eq!(
    s,
    k,
    "AVX2 xv36<ALPHA={ALPHA}>→{} u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})",
    if ALPHA { "RGBA u16" } else { "RGB u16" }
  );
}

fn check_luma(width: usize) {
  let p = pseudo_random_xv36(width, 0xC001);
  let mut s = std::vec![0u8; width];
  let mut k = std::vec![0u8; width];
  scalar::xv36_to_luma_row::<false>(&p, &mut s, width);
  unsafe {
    xv36_to_luma_row::<false>(&p, &mut k, width);
  }
  assert_eq!(s, k, "AVX2 xv36→luma diverges (width={width})");
}

fn check_luma_u16(width: usize) {
  let p = pseudo_random_xv36(width, 0xC001);
  let mut s = std::vec![0u16; width];
  let mut k = std::vec![0u16; width];
  scalar::xv36_to_luma_u16_row::<false>(&p, &mut s, width);
  unsafe {
    xv36_to_luma_u16_row::<false>(&p, &mut k, width);
  }
  assert_eq!(s, k, "AVX2 xv36→luma u16 diverges (width={width})");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn avx2_xv36_rgb_matches_scalar_all_matrices() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for m in [
    KernelMatrix::Bt601,
    KernelMatrix::Bt709,
    KernelMatrix::Bt2020Ncl,
    KernelMatrix::Smpte240m,
    KernelMatrix::Fcc,
    KernelMatrix::YCgCo,
  ] {
    for full in [true, false] {
      // Width 16 = one main-loop iteration (no tail).
      check_rgb::<false>(16, m, full);
      check_rgb::<true>(16, m, full);
      check_rgb_u16::<false>(16, m, full);
      check_rgb_u16::<true>(16, m, full);
    }
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn avx2_xv36_matches_scalar_widths() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  // Includes widths with SIMD main loop (multiples of 16), scalar tails
  // (1..15 — <16 pixels, no main loop), and large production widths
  // (1920p, 1921 = 1920+1 tail, 1923 = 1920+3 tail).
  for w in [1usize, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 1920, 1921, 1923] {
    check_rgb::<false>(w, KernelMatrix::Bt709, false);
    check_rgb::<true>(w, KernelMatrix::Bt709, true);
    check_rgb_u16::<false>(w, KernelMatrix::Bt2020Ncl, true);
    check_rgb_u16::<true>(w, KernelMatrix::Bt601, false);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn avx2_xv36_luma_matches_scalar_widths() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [1usize, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 1920, 1921, 1923] {
    check_luma(w);
    check_luma_u16(w);
  }
}

/// Build an XV36 packed buffer for `width` pixels where:
/// - Y[n] = (n + 1) << 4 (12-bit `n+1` MSB-aligned in slot 1)
/// - U[n] = (2n + 1) << 4 (12-bit `2n+1` MSB-aligned; XV36 is 4:4:4)
/// - V    = 0x800 << 4 = 0x8000 (neutral 12-bit midpoint, MSB-aligned)
/// - A    = 0 (slot 3 is padding for the X-prefix variant)
///
/// XV36 layout (per pixel, four contiguous u16):
///   slot 0 = U (12-bit value MSB-aligned: bits[15:4] = value, bits[3:0] = 0)
///   slot 1 = Y (12-bit value MSB-aligned)
///   slot 2 = V (12-bit value MSB-aligned)
///   slot 3 = A (padding for XV36 — read but discarded)
fn build_xv36_packed_y_n_plus_1_u_2n_plus_1_v_neutral_a_zero(width: usize) -> std::vec::Vec<u16> {
  let mut packed = std::vec::Vec::with_capacity(width * 4);
  for n in 0..width {
    let y = ((n as u16) + 1) << 4; // slot 1: Y = (n+1) MSB-aligned
    let u = ((2 * (n as u16)) + 1) << 4; // slot 0: U = (2n+1) MSB-aligned
    packed.push(u);
    packed.push(y);
    packed.push(0x8000u16); // slot 2: V = 0x800 << 4 (neutral)
    packed.push(0u16); // slot 3: A = padding
  }
  packed
}

/// Multi-channel lane-order regression — encodes pixel index in BOTH Y AND U
/// so we catch per-channel asymmetric mask bugs that the previous Y-only
/// test would miss. Pattern from Ship 12d AYUV64 backport.
///
/// Asserts:
/// - `luma_u16_row` output = [1..=W] (Y values direct after `>> 4`)
/// - SIMD u16 RGB output == scalar u16 RGB output (any lane-order or
///   per-channel mask bug diverges between SIMD and scalar)
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn avx2_xv36_lane_order_per_pixel_y_and_u() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  // W = 2 × SIMD entry threshold (16) so we exercise ≥2 main-loop iterations.
  const W: usize = 32;
  let packed = build_xv36_packed_y_n_plus_1_u_2n_plus_1_v_neutral_a_zero(W);

  // Part 1: Luma natural-order at u16 (drops the 4-bit padding to recover n+1)
  let mut luma_u16 = std::vec![0u16; W];
  unsafe {
    xv36_to_luma_u16_row::<false>(&packed, &mut luma_u16, W);
  }
  let expected_luma: std::vec::Vec<u16> = (1..=W as u16).collect();
  assert_eq!(luma_u16, expected_luma, "avx2 xv36 luma_u16 reorder bug");

  // Part 2: SIMD vs scalar parity at u16 RGB (catches U/Y channel swap bugs)
  let mut simd_rgb = std::vec![0u16; W * 3];
  let mut scalar_rgb = std::vec![0u16; W * 3];
  unsafe {
    xv36_to_rgb_u16_or_rgba_u16_row::<false, false>(
      &packed,
      &mut simd_rgb,
      W,
      KernelMatrix::Bt709,
      false,
    );
  }
  scalar::xv36_to_rgb_u16_or_rgba_u16_row::<false, false>(
    &packed,
    &mut scalar_rgb,
    W,
    KernelMatrix::Bt709,
    false,
  );
  assert_eq!(
    simd_rgb, scalar_rgb,
    "avx2 xv36 SIMD vs scalar diverges (u16 RGB) — lane-order bug"
  );
}

/// SIMD-level BE-vs-LE parity test — exercises the host-aware endian gate
/// in `endian::load_endian_u16x*::<BE>` for AVX2. See sibling v410 test for
/// rationale.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn avx2_xv36_be_le_simd_parity() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  // Construct LE/BE buffers from raw bytes via `to_le_bytes` / `to_be_bytes`
  // so semantics are host-independent. The earlier `swap_bytes` pattern only
  // validated this on LE hosts (on BE hosts both buffers degenerate to
  // equal-but-wrong values and the test passed vacuously).
  for w in [7usize, 8, 17, 33] {
    let intended = pseudo_random_xv36(w, 0xBEEF);
    let le_bytes: std::vec::Vec<u8> = intended.iter().flat_map(|v| v.to_le_bytes()).collect();
    let be_bytes: std::vec::Vec<u8> = intended.iter().flat_map(|v| v.to_be_bytes()).collect();
    let le: std::vec::Vec<u16> = le_bytes
      .chunks_exact(2)
      .map(|b| u16::from_ne_bytes([b[0], b[1]]))
      .collect();
    let be: std::vec::Vec<u16> = be_bytes
      .chunks_exact(2)
      .map(|b| u16::from_ne_bytes([b[0], b[1]]))
      .collect();

    for (alpha, bpp) in [(false, 3usize), (true, 4)] {
      let mut out_le = std::vec![0u8; w * bpp];
      let mut out_be = std::vec![0u8; w * bpp];
      unsafe {
        if alpha {
          xv36_to_rgb_or_rgba_row::<true, false>(&le, &mut out_le, w, KernelMatrix::Bt709, false);
          xv36_to_rgb_or_rgba_row::<true, true>(&be, &mut out_be, w, KernelMatrix::Bt709, false);
        } else {
          xv36_to_rgb_or_rgba_row::<false, false>(&le, &mut out_le, w, KernelMatrix::Bt709, false);
          xv36_to_rgb_or_rgba_row::<false, true>(&be, &mut out_be, w, KernelMatrix::Bt709, false);
        }
      }
      assert_eq!(
        out_le, out_be,
        "avx2 xv36 BE-vs-LE SIMD parity failed (alpha={alpha}, w={w})"
      );
    }

    for (alpha, bpp) in [(false, 3usize), (true, 4)] {
      let mut out_le = std::vec![0u16; w * bpp];
      let mut out_be = std::vec![0u16; w * bpp];
      unsafe {
        if alpha {
          xv36_to_rgb_u16_or_rgba_u16_row::<true, false>(
            &le,
            &mut out_le,
            w,
            KernelMatrix::Bt709,
            true,
          );
          xv36_to_rgb_u16_or_rgba_u16_row::<true, true>(
            &be,
            &mut out_be,
            w,
            KernelMatrix::Bt709,
            true,
          );
        } else {
          xv36_to_rgb_u16_or_rgba_u16_row::<false, false>(
            &le,
            &mut out_le,
            w,
            KernelMatrix::Bt709,
            true,
          );
          xv36_to_rgb_u16_or_rgba_u16_row::<false, true>(
            &be,
            &mut out_be,
            w,
            KernelMatrix::Bt709,
            true,
          );
        }
      }
      assert_eq!(
        out_le, out_be,
        "avx2 xv36 BE-vs-LE SIMD parity failed (u16, alpha={alpha}, w={w})"
      );
    }

    {
      let mut out_le = std::vec![0u8; w];
      let mut out_be = std::vec![0u8; w];
      unsafe {
        xv36_to_luma_row::<false>(&le, &mut out_le, w);
        xv36_to_luma_row::<true>(&be, &mut out_be, w);
      }
      assert_eq!(
        out_le, out_be,
        "avx2 xv36 BE-vs-LE SIMD parity failed (luma u8, w={w})"
      );
    }

    {
      let mut out_le = std::vec![0u16; w];
      let mut out_be = std::vec![0u16; w];
      unsafe {
        xv36_to_luma_u16_row::<false>(&le, &mut out_le, w);
        xv36_to_luma_u16_row::<true>(&be, &mut out_be, w);
      }
      assert_eq!(
        out_le, out_be,
        "avx2 xv36 BE-vs-LE SIMD parity failed (luma u16, w={w})"
      );
    }
  }
}
