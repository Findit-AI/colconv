use super::super::*;
use crate::{KernelMatrix, row::scalar};

/// Verify multi-channel Y+U lane order for Y2xx (BITS-generic Y210/Y212).
///
/// Y2xx YUYV-shape u16x2: `[Y0, U, Y1, V]` per 2 pixels (4:2:2).
/// MSB-aligned: low `(16 - BITS)` bits are zero, active value in high BITS.
/// - `Y[n] = ((n + 1) as u16) << shift`
/// - `U[k] = ((2k + 1) as u16) << shift`  (one U per pair)
/// - `V = 0x8000` (neutral midpoint, same for BITS=10 and BITS=12)
///
/// Part 1: luma u16 natural-order check.
/// Part 2: SIMD vs scalar parity on u16 RGB output.
///
/// NEON threshold: 8 px/iter. W=16 covers exactly 2 full SIMD iterations.
fn check_y2xx_lane_order_per_pixel_y_and_u<const BITS: u32>() {
  const W: usize = 16;
  let shift: u16 = (16 - BITS) as u16;
  let neutral_chroma: u16 = (1u16 << (BITS - 1)) << shift; // 0x8000 for both BITS=10,12

  // Build Y2xx YUYV-shape: [Y0, U, Y1, V] per 2-pixel pair.
  let mut packed = std::vec![0u16; W * 2];
  for k in 0..(W / 2) {
    let y0 = ((2 * k) as u16 + 1) << shift;
    let y1 = ((2 * k) as u16 + 2) << shift;
    let u = ((2 * k) as u16 + 1) << shift;
    packed[k * 4] = y0;
    packed[k * 4 + 1] = u;
    packed[k * 4 + 2] = y1;
    packed[k * 4 + 3] = neutral_chroma;
  }

  // Part 1: luma u16 natural-order (low-bit-packed: active BITS in low bits).
  let mut luma_u16 = std::vec![0u16; W];
  unsafe {
    y2xx_n_to_luma_u16_row::<BITS, false>(&packed, &mut luma_u16, W);
  }
  let expected_luma: std::vec::Vec<u16> = (1..=W as u16).collect();
  assert_eq!(
    luma_u16, expected_luma,
    "y2xx<BITS={BITS}> luma_u16 reorder bug"
  );

  // Part 2: SIMD vs scalar parity at u16 RGB.
  let mut simd_rgb = std::vec![0u16; W * 3];
  let mut scalar_rgb = std::vec![0u16; W * 3];
  unsafe {
    y2xx_n_to_rgb_u16_or_rgba_u16_row::<BITS, false, false>(
      &packed,
      &mut simd_rgb,
      W,
      KernelMatrix::Bt709,
      false,
    );
  }
  scalar::y2xx_n_to_rgb_u16_or_rgba_u16_row::<BITS, false, false>(
    &packed,
    &mut scalar_rgb,
    W,
    KernelMatrix::Bt709,
    false,
  );
  assert_eq!(
    simd_rgb, scalar_rgb,
    "y2xx<BITS={BITS}> SIMD vs scalar diverges (u16 RGB)"
  );
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_y2xx_lane_order_per_pixel_y_and_u_bits10() {
  check_y2xx_lane_order_per_pixel_y_and_u::<10>();
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_y2xx_lane_order_per_pixel_y_and_u_bits12() {
  check_y2xx_lane_order_per_pixel_y_and_u::<12>();
}

/// Builds a deterministic pseudo-random Y210-shaped u16 buffer with
/// `width * 2` u16 samples (one quadruple = 4 u16 = 2 pixels). Each
/// u16 sample has 10 active bits sitting in the high bits, low 6
/// bits zero (matches Y210's MSB-aligned encoding).
fn pseudo_random_y210(width: usize, seed: usize) -> std::vec::Vec<u16> {
  (0..width * 2)
    .map(|i| {
      let s = ((i.wrapping_mul(seed).wrapping_add(seed * 3)) & 0x3FF) as u16;
      s << 6
    })
    .collect()
}

fn check_rgb<const BITS: u32>(width: usize, matrix: KernelMatrix, full_range: bool) {
  let p = pseudo_random_y210(width, 0xAA55);
  let mut s = std::vec![0u8; width * 3];
  let mut k = std::vec![0u8; width * 3];
  scalar::y2xx_n_to_rgb_or_rgba_row::<BITS, false, false>(&p, &mut s, width, matrix, full_range);
  unsafe {
    y2xx_n_to_rgb_or_rgba_row::<BITS, false, false>(&p, &mut k, width, matrix, full_range);
  }
  assert_eq!(
    s, k,
    "NEON y2xx<{BITS}>→RGB diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_rgba<const BITS: u32>(width: usize, matrix: KernelMatrix, full_range: bool) {
  let p = pseudo_random_y210(width, 0xAA55);
  let mut s = std::vec![0u8; width * 4];
  let mut k = std::vec![0u8; width * 4];
  scalar::y2xx_n_to_rgb_or_rgba_row::<BITS, true, false>(&p, &mut s, width, matrix, full_range);
  unsafe {
    y2xx_n_to_rgb_or_rgba_row::<BITS, true, false>(&p, &mut k, width, matrix, full_range);
  }
  assert_eq!(
    s, k,
    "NEON y2xx<{BITS}>→RGBA diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_rgb_u16<const BITS: u32>(width: usize, matrix: KernelMatrix, full_range: bool) {
  let p = pseudo_random_y210(width, 0xAA55);
  let mut s = std::vec![0u16; width * 3];
  let mut k = std::vec![0u16; width * 3];
  scalar::y2xx_n_to_rgb_u16_or_rgba_u16_row::<BITS, false, false>(
    &p, &mut s, width, matrix, full_range,
  );
  unsafe {
    y2xx_n_to_rgb_u16_or_rgba_u16_row::<BITS, false, false>(&p, &mut k, width, matrix, full_range);
  }
  assert_eq!(
    s, k,
    "NEON y2xx<{BITS}>→RGB u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_rgba_u16<const BITS: u32>(width: usize, matrix: KernelMatrix, full_range: bool) {
  let p = pseudo_random_y210(width, 0xAA55);
  let mut s = std::vec![0u16; width * 4];
  let mut k = std::vec![0u16; width * 4];
  scalar::y2xx_n_to_rgb_u16_or_rgba_u16_row::<BITS, true, false>(
    &p, &mut s, width, matrix, full_range,
  );
  unsafe {
    y2xx_n_to_rgb_u16_or_rgba_u16_row::<BITS, true, false>(&p, &mut k, width, matrix, full_range);
  }
  assert_eq!(
    s, k,
    "NEON y2xx<{BITS}>→RGBA u16 diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_luma<const BITS: u32>(width: usize) {
  let p = pseudo_random_y210(width, 0xC001);
  let mut s = std::vec![0u8; width];
  let mut k = std::vec![0u8; width];
  scalar::y2xx_n_to_luma_row::<BITS, false>(&p, &mut s, width);
  unsafe {
    y2xx_n_to_luma_row::<BITS, false>(&p, &mut k, width);
  }
  assert_eq!(s, k, "NEON y2xx<{BITS}>→luma diverges (width={width})");
}

fn check_luma_u16<const BITS: u32>(width: usize) {
  let p = pseudo_random_y210(width, 0xC001);
  let mut s = std::vec![0u16; width];
  let mut k = std::vec![0u16; width];
  scalar::y2xx_n_to_luma_u16_row::<BITS, false>(&p, &mut s, width);
  unsafe {
    y2xx_n_to_luma_u16_row::<BITS, false>(&p, &mut k, width);
  }
  assert_eq!(s, k, "NEON y2xx<{BITS}>→luma u16 diverges (width={width})");
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_y210_rgb_matches_scalar_all_matrices() {
  for m in [
    KernelMatrix::Bt601,
    KernelMatrix::Bt709,
    KernelMatrix::Bt2020Ncl,
    KernelMatrix::Smpte240m,
    KernelMatrix::Fcc,
    KernelMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_rgb::<10>(16, m, full);
      check_rgba::<10>(16, m, full);
      check_rgb_u16::<10>(16, m, full);
      check_rgba_u16::<10>(16, m, full);
    }
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_y210_matches_scalar_widths() {
  for w in [2usize, 4, 14, 16, 18, 30, 32, 34, 62, 64, 66, 1920, 1922] {
    check_rgb::<10>(w, KernelMatrix::Bt709, false);
    check_rgba::<10>(w, KernelMatrix::Bt709, true);
    check_rgb_u16::<10>(w, KernelMatrix::Bt2020Ncl, true);
    check_rgba_u16::<10>(w, KernelMatrix::Bt601, false);
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_y210_luma_matches_scalar_widths() {
  for w in [2usize, 4, 14, 16, 18, 30, 32, 34, 62, 64, 66, 1920, 1922] {
    check_luma::<10>(w);
    check_luma_u16::<10>(w);
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_y212_matches_scalar_widths() {
  // 12-bit MSB-aligned generator: shift by 4 instead of 6.
  fn pseudo_random_y212(width: usize, seed: usize) -> std::vec::Vec<u16> {
    (0..width * 2)
      .map(|i| {
        let s = ((i.wrapping_mul(seed).wrapping_add(seed * 3)) & 0xFFF) as u16;
        s << 4
      })
      .collect()
  }
  for w in [2usize, 4, 14, 16, 18, 30, 32, 34, 62, 64, 66, 1920, 1922] {
    let p = pseudo_random_y212(w, 0xAA55);
    let mut s = std::vec![0u8; w * 3];
    let mut k = std::vec![0u8; w * 3];
    scalar::y2xx_n_to_rgb_or_rgba_row::<12, false, false>(
      &p,
      &mut s,
      w,
      KernelMatrix::Bt709,
      false,
    );
    unsafe {
      y2xx_n_to_rgb_or_rgba_row::<12, false, false>(&p, &mut k, w, KernelMatrix::Bt709, false);
    }
    assert_eq!(s, k, "NEON y2xx<12>→RGB diverges (width={w})");

    let mut s_u16 = std::vec![0u16; w * 4];
    let mut k_u16 = std::vec![0u16; w * 4];
    scalar::y2xx_n_to_rgb_u16_or_rgba_u16_row::<12, true, false>(
      &p,
      &mut s_u16,
      w,
      KernelMatrix::Bt2020Ncl,
      true,
    );
    unsafe {
      y2xx_n_to_rgb_u16_or_rgba_u16_row::<12, true, false>(
        &p,
        &mut k_u16,
        w,
        KernelMatrix::Bt2020Ncl,
        true,
      );
    }
    assert_eq!(s_u16, k_u16, "NEON y2xx<12>→RGBA u16 diverges (width={w})");

    let mut sl = std::vec![0u8; w];
    let mut kl = std::vec![0u8; w];
    scalar::y2xx_n_to_luma_row::<12, false>(&p, &mut sl, w);
    unsafe {
      y2xx_n_to_luma_row::<12, false>(&p, &mut kl, w);
    }
    assert_eq!(sl, kl, "NEON y2xx<12>→luma diverges (width={w})");

    let mut slu = std::vec![0u16; w];
    let mut klu = std::vec![0u16; w];
    scalar::y2xx_n_to_luma_u16_row::<12, false>(&p, &mut slu, w);
    unsafe {
      y2xx_n_to_luma_u16_row::<12, false>(&p, &mut klu, w);
    }
    assert_eq!(slu, klu, "NEON y2xx<12>→luma u16 diverges (width={w})");
  }
}

// Host-independent BE/LE SIMD parity tests.
//
// Constructs LE/BE buffers from raw bytes via `to_le_bytes` /
// `to_be_bytes` and reinterprets as host-native `u16` via `from_ne_bytes`.
// The byte-level encoding is host-independent — on every host the LE
// buffer carries the intended values as LE-encoded bytes and the BE
// buffer carries the same values as BE-encoded bytes — so both kernel
// monomorphizations decode to the same logical values and produce
// byte-identical output on both LE and BE hosts. Locks down the
// `BE == HOST_NATIVE_BE` host-endian gate on the NEON Y2xx SIMD bodies.

/// Builds intended Y2xx-shaped values then materializes both LE-encoded
/// and BE-encoded `&[u16]` planes from raw bytes (host-independent).
fn build_le_be_y2xx<const BITS: u32>(
  width: usize,
  seed: usize,
) -> (std::vec::Vec<u16>, std::vec::Vec<u16>) {
  let shift = 16 - BITS;
  let mask: u16 = (1u16 << BITS) - 1;
  let intended: std::vec::Vec<u16> = (0..width * 2)
    .map(|i| {
      let s = ((i.wrapping_mul(seed).wrapping_add(seed * 3)) & mask as usize) as u16;
      s << shift
    })
    .collect();
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
  (le, be)
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_y2xx_be_le_simd_parity_bits10() {
  // Widths covering the SIMD body (8 px), tail-only (< 8), and
  // body+tail (8 + tail) so both code paths are exercised on each host.
  for w in [4usize, 8, 14, 16, 22, 32, 1920] {
    let (le, be) = build_le_be_y2xx::<10>(w, 0xBEEF);
    // u8 RGB
    let mut le_rgb = std::vec![0u8; w * 3];
    let mut be_rgb = std::vec![0u8; w * 3];
    unsafe {
      y2xx_n_to_rgb_or_rgba_row::<10, false, false>(
        &le,
        &mut le_rgb,
        w,
        KernelMatrix::Bt709,
        false,
      );
      y2xx_n_to_rgb_or_rgba_row::<10, false, true>(&be, &mut be_rgb, w, KernelMatrix::Bt709, false);
    }
    assert_eq!(le_rgb, be_rgb, "y2xx<10> NEON LE vs BE RGB parity (w={w})");

    // u16 RGB
    let mut le_u16 = std::vec![0u16; w * 3];
    let mut be_u16 = std::vec![0u16; w * 3];
    unsafe {
      y2xx_n_to_rgb_u16_or_rgba_u16_row::<10, false, false>(
        &le,
        &mut le_u16,
        w,
        KernelMatrix::Bt709,
        false,
      );
      y2xx_n_to_rgb_u16_or_rgba_u16_row::<10, false, true>(
        &be,
        &mut be_u16,
        w,
        KernelMatrix::Bt709,
        false,
      );
    }
    assert_eq!(
      le_u16, be_u16,
      "y2xx<10> NEON LE vs BE RGB u16 parity (w={w})"
    );

    // luma u8
    let mut le_l = std::vec![0u8; w];
    let mut be_l = std::vec![0u8; w];
    unsafe {
      y2xx_n_to_luma_row::<10, false>(&le, &mut le_l, w);
      y2xx_n_to_luma_row::<10, true>(&be, &mut be_l, w);
    }
    assert_eq!(le_l, be_l, "y2xx<10> NEON LE vs BE luma u8 parity (w={w})");

    // luma u16
    let mut le_lu = std::vec![0u16; w];
    let mut be_lu = std::vec![0u16; w];
    unsafe {
      y2xx_n_to_luma_u16_row::<10, false>(&le, &mut le_lu, w);
      y2xx_n_to_luma_u16_row::<10, true>(&be, &mut be_lu, w);
    }
    assert_eq!(
      le_lu, be_lu,
      "y2xx<10> NEON LE vs BE luma u16 parity (w={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_y2xx_be_le_simd_parity_bits12() {
  for w in [4usize, 8, 14, 16, 22, 32, 1920] {
    let (le, be) = build_le_be_y2xx::<12>(w, 0xC0DE);

    let mut le_rgba = std::vec![0u8; w * 4];
    let mut be_rgba = std::vec![0u8; w * 4];
    unsafe {
      y2xx_n_to_rgb_or_rgba_row::<12, true, false>(
        &le,
        &mut le_rgba,
        w,
        KernelMatrix::Bt2020Ncl,
        true,
      );
      y2xx_n_to_rgb_or_rgba_row::<12, true, true>(
        &be,
        &mut be_rgba,
        w,
        KernelMatrix::Bt2020Ncl,
        true,
      );
    }
    assert_eq!(
      le_rgba, be_rgba,
      "y2xx<12> NEON LE vs BE RGBA parity (w={w})"
    );

    let mut le_lu = std::vec![0u16; w];
    let mut be_lu = std::vec![0u16; w];
    unsafe {
      y2xx_n_to_luma_u16_row::<12, false>(&le, &mut le_lu, w);
      y2xx_n_to_luma_u16_row::<12, true>(&be, &mut be_lu, w);
    }
    assert_eq!(
      le_lu, be_lu,
      "y2xx<12> NEON LE vs BE luma u16 parity (w={w})"
    );
  }
}
