//! Tests for `crate::row::scalar::planar_gbr_32bit`.

use super::*;
use crate::KernelMatrix;

/// Re-encode a host-native u32 slice as LE-encoded byte storage. Kernels
/// called with `BE = false` recover the intended logical values via
/// `u32::from_le` on both LE (no-op) and BE (byte-swap) hosts.
fn le_wire_u16(v: u16) -> u16 {
  u16::from_ne_bytes(v.to_le_bytes())
}

fn as_le_u32(host: &[u32]) -> std::vec::Vec<u32> {
  host
    .iter()
    .map(|v| u32::from_ne_bytes(v.to_le_bytes()))
    .collect()
}

/// Mirror of [`as_le_u32`] for kernels invoked with `BE = true`.
fn as_be_u32(host: &[u32]) -> std::vec::Vec<u32> {
  host
    .iter()
    .map(|v| u32::from_ne_bytes(v.to_be_bytes()))
    .collect()
}

// ---- Channel reorder + narrow (G/B/R/A → R, G, B[, A]) ----------------------

/// u8 RGB: planes reorder to R, G, B and narrow `>> 24`; alpha dropped.
#[test]
fn gbr32_to_rgb_reorders_and_narrows_u8() {
  let g = as_le_u32(&[0x6400_0000]); // 100 << 24
  let b = as_le_u32(&[0x3200_0000]); // 50  << 24
  let r = as_le_u32(&[0xC800_0000]); // 200 << 24
  let mut out = [0u8; 3];
  gbr32_to_rgb_row::<false>(&g, &b, &r, &mut out, 1);
  assert_eq!(out, [200, 100, 50], "packed order is R, G, B");
}

/// u16 RGB: planes reorder to R, G, B and narrow `>> 16`; alpha dropped.
#[test]
fn gbr32_to_rgb_u16_reorders_and_narrows() {
  let g = as_le_u32(&[0x1234_5678]);
  let b = as_le_u32(&[0x9ABC_DEF0]);
  let r = as_le_u32(&[0x0F0F_AAAA]);
  let mut out = [0u16; 3];
  gbr32_to_rgb_u16_row::<false>(&g, &b, &r, &mut out, 1);
  assert_eq!(out, [0x0F0F, 0x1234, 0x9ABC], "R, G, B narrowed >> 16");
}

/// u8 RGBA: real alpha narrowed `>> 24` into slot 3.
#[test]
fn gbra32_to_rgba_passes_alpha_u8() {
  let g = as_le_u32(&[0x6400_0000]);
  let b = as_le_u32(&[0x3200_0000]);
  let r = as_le_u32(&[0xC800_0000]);
  let a = as_le_u32(&[0x9000_0000]); // 0x90
  let mut out = [0u8; 4];
  gbra32_to_rgba_row::<false>(&g, &b, &r, &a, &mut out, 1);
  assert_eq!(out, [200, 100, 50, 0x90], "R, G, B, A");
}

/// u16 RGBA: real alpha narrowed `>> 16` into slot 3.
#[test]
fn gbra32_to_rgba_u16_passes_alpha() {
  let g = as_le_u32(&[0x1234_0000]);
  let b = as_le_u32(&[0x5678_0000]);
  let r = as_le_u32(&[0x9ABC_0000]);
  let a = as_le_u32(&[0xBEEF_0000]);
  let mut out = [0u16; 4];
  gbra32_to_rgba_u16_row::<false>(&g, &b, &r, &a, &mut out, 1);
  assert_eq!(out, [0x9ABC, 0x1234, 0x5678, 0xBEEF], "R, G, B, A");
}

/// All-max input saturates to 0xFF (u8) and 0xFFFF (u16).
#[test]
fn gbra32_all_max_saturates() {
  let p = std::vec![0xFFFF_FFFFu32; 4];
  let mut u8out = [0u8; 16];
  gbra32_to_rgba_row::<false>(&p, &p, &p, &p, &mut u8out, 4);
  assert!(u8out.iter().all(|&v| v == 0xFF));
  let mut u16out = [0u16; 16];
  gbra32_to_rgba_u16_row::<false>(&p, &p, &p, &p, &mut u16out, 4);
  assert!(u16out.iter().all(|&v| v == 0xFFFF));
}

// ---- BE parity (host-independent fixtures, pinned to absolute reference) ----

#[test]
fn gbra32_to_rgba_u16_be_parity() {
  let g_i: std::vec::Vec<u32> = std::vec![0x1234_5678, 0x0011_2233];
  let b_i: std::vec::Vec<u32> = std::vec![0x5678_9ABC, 0x4455_6677];
  let r_i: std::vec::Vec<u32> = std::vec![0x9ABC_DEF0, 0x8899_AABB];
  let a_i: std::vec::Vec<u32> = std::vec![0xCAFE_BABE, 0xDEAD_BEEF];

  let mut out_le = std::vec![0u16; 8];
  let mut out_be = std::vec![0u16; 8];
  gbra32_to_rgba_u16_row::<false>(
    &as_le_u32(&g_i),
    &as_le_u32(&b_i),
    &as_le_u32(&r_i),
    &as_le_u32(&a_i),
    &mut out_le,
    2,
  );
  gbra32_to_rgba_u16_row::<true>(
    &as_be_u32(&g_i),
    &as_be_u32(&b_i),
    &as_be_u32(&r_i),
    &as_be_u32(&a_i),
    &mut out_be,
    2,
  );
  let mut expected = std::vec![0u16; 8];
  for x in 0..2 {
    expected[x * 4] = (r_i[x] >> 16) as u16;
    expected[x * 4 + 1] = (g_i[x] >> 16) as u16;
    expected[x * 4 + 2] = (b_i[x] >> 16) as u16;
    expected[x * 4 + 3] = (a_i[x] >> 16) as u16;
  }
  assert_eq!(out_le, expected, "LE path must match scalar reference");
  assert_eq!(out_be, expected, "BE path must match scalar reference");
  assert_eq!(out_le, out_be, "BE and LE outputs must agree");
}

#[test]
fn gbr32_to_rgb_be_parity() {
  let g_i: std::vec::Vec<u32> = std::vec![0x1234_5678, 0x0011_2233];
  let b_i: std::vec::Vec<u32> = std::vec![0x5678_9ABC, 0x4455_6677];
  let r_i: std::vec::Vec<u32> = std::vec![0x9ABC_DEF0, 0x8899_AABB];

  let mut out_le = std::vec![0u8; 6];
  let mut out_be = std::vec![0u8; 6];
  gbr32_to_rgb_row::<false>(
    &as_le_u32(&g_i),
    &as_le_u32(&b_i),
    &as_le_u32(&r_i),
    &mut out_le,
    2,
  );
  gbr32_to_rgb_row::<true>(
    &as_be_u32(&g_i),
    &as_be_u32(&b_i),
    &as_be_u32(&r_i),
    &mut out_be,
    2,
  );
  let mut expected = std::vec![0u8; 6];
  for x in 0..2 {
    expected[x * 3] = (r_i[x] >> 24) as u8;
    expected[x * 3 + 1] = (g_i[x] >> 24) as u8;
    expected[x * 3 + 2] = (b_i[x] >> 24) as u8;
  }
  assert_eq!(out_le, expected, "LE path must match scalar reference");
  assert_eq!(out_be, expected, "BE path must match scalar reference");
  assert_eq!(out_le, out_be, "BE and LE outputs must agree");
}

// ---- Luma (Q15 native u16, full + limited range) ----------------------------

/// Neutral grey: G = B = R = mid produces Y' = mid (full-range, coeffs sum to
/// 1 in Q15). Narrow `>> 16` then Q15.
#[test]
fn gbr32_luma_full_range_neutral_grey() {
  let mid = 0x8000_FFFFu32; // narrows >> 16 to 0x8000 = 32768
  let g = as_le_u32(&[mid; 4]);
  let mut out = [0u16; 4];
  gbr32_to_luma_u16_row::<false>(&g, &g, &g, &mut out, 4, KernelMatrix::Bt709, true);
  for &y in &out {
    // R=G=B=32768 → Y' = 32768 (allow 1 LSB Q15 rounding slack).
    assert!((y as i32 - 32768).abs() <= 1, "neutral grey Y'={y}");
  }
}

/// Limited-range black floor: zero input maps to Y' = 16 << 8 = 4096.
#[test]
fn gbr32_luma_limited_range_black_floor() {
  let g = as_le_u32(&[0u32; 2]);
  let mut out = [0u16; 2];
  gbr32_to_luma_u16_row::<false>(&g, &g, &g, &mut out, 2, KernelMatrix::Bt709, false);
  assert!(out.iter().all(|&y| y == 4096), "limited black = 16 << 8");
}

/// Limited-range white ceiling: max input maps to Y' = 235 << 8 = 60160.
#[test]
fn gbr32_luma_limited_range_white_ceiling() {
  let g = as_le_u32(&[0xFFFF_FFFFu32; 2]);
  let mut out = [0u16; 2];
  gbr32_to_luma_u16_row::<false>(&g, &g, &g, &mut out, 2, KernelMatrix::Bt709, false);
  assert!(out.iter().all(|&y| y == 60160), "limited white = 235 << 8");
}

/// Luma is byte-identical to feeding the `>> 16`-narrowed planes through the
/// high-bit `gbr_to_luma_u16_high_bit_row::<16>` path (the staging equivalence
/// the resample tail relies on).
#[test]
fn gbr32_luma_matches_narrowed_high_bit_path() {
  let g_i: std::vec::Vec<u32> = (0..8).map(|i| (i as u32) * 0x1357_2468 + 0xABCD).collect();
  let b_i: std::vec::Vec<u32> = (0..8).map(|i| (i as u32) * 0x0246_8ACE + 0x1234).collect();
  let r_i: std::vec::Vec<u32> = (0..8).map(|i| (i as u32) * 0x0FED_CBA9 + 0x5678).collect();
  for full_range in [false, true] {
    for matrix in [KernelMatrix::Bt709, KernelMatrix::Bt601] {
      let mut out32 = std::vec![0u16; 8];
      gbr32_to_luma_u16_row::<false>(
        &as_le_u32(&g_i),
        &as_le_u32(&b_i),
        &as_le_u32(&r_i),
        &mut out32,
        8,
        matrix,
        full_range,
      );
      let gn: std::vec::Vec<u16> = g_i.iter().map(|&v| le_wire_u16((v >> 16) as u16)).collect();
      let bn: std::vec::Vec<u16> = b_i.iter().map(|&v| le_wire_u16((v >> 16) as u16)).collect();
      let rn: std::vec::Vec<u16> = r_i.iter().map(|&v| le_wire_u16((v >> 16) as u16)).collect();
      let mut out16 = std::vec![0u16; 8];
      super::super::planar_gbr_high_bit::gbr_to_luma_u16_high_bit_row::<16, false>(
        &gn, &bn, &rn, &mut out16, 8, matrix, full_range,
      );
      assert_eq!(out32, out16, "fr={full_range} matrix={matrix:?}");
    }
  }
}
