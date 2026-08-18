//! Tests for `crate::row::scalar::planar_gbr_msb`.

use super::*;
use crate::{KernelMatrix, row::scalar::planar_gbr_high_bit as low};

/// Re-encode a host-native u16 slice as LE-encoded byte storage. Kernels
/// called with `BE = false` recover the intended logical values via
/// `u16::from_le` on both LE (no-op) and BE (byte-swap) hosts.
fn as_le_u16(host: &[u16]) -> std::vec::Vec<u16> {
  host
    .iter()
    .map(|v| u16::from_ne_bytes(v.to_le_bytes()))
    .collect()
}

/// Re-encode a host-native u16 slice as BE-encoded byte storage (for `BE =
/// true` kernels).
fn as_be_u16(host: &[u16]) -> std::vec::Vec<u16> {
  host
    .iter()
    .map(|v| u16::from_ne_bytes(v.to_be_bytes()))
    .collect()
}

/// MSB-align a logical sample `s` (`0 ..= (1 << BITS) - 1`) into the high
/// `BITS` bits of a `u16` (the on-wire MSB representation, FFmpeg
/// `shift = 16 - BITS`).
fn msb<const BITS: u32>(samples: &[u16]) -> std::vec::Vec<u16> {
  samples.iter().map(|&s| s << (16 - BITS)).collect()
}

// ---- gbr_to_rgb_msb_row: u8 output, recover + downshift -----------------

#[test]
fn rgb_msb_bits10_channel_reorder() {
  // G=0, B=100, R=1000 (logical) → packed R,G,B = 1000>>2, 0>>2, 100>>2.
  let g = as_le_u16(&msb::<10>(&[0]));
  let b = as_le_u16(&msb::<10>(&[100]));
  let r = as_le_u16(&msb::<10>(&[1000]));
  let mut out = [0u8; 3];
  gbr_to_rgb_msb_row::<10, false>(&g, &b, &r, &mut out, 1);
  assert_eq!(out[0], 250); // R = 1000 >> 2
  assert_eq!(out[1], 0); // G
  assert_eq!(out[2], 25); // B = 100 >> 2
}

#[test]
fn rgb_msb_bits10_max_value_becomes_0xff() {
  let max = (1u16 << 10) - 1; // 1023 logical
  let g = as_le_u16(&msb::<10>(&[max; 4]));
  let b = as_le_u16(&msb::<10>(&[max; 4]));
  let r = as_le_u16(&msb::<10>(&[max; 4]));
  let mut out = [0u8; 12];
  gbr_to_rgb_msb_row::<10, false>(&g, &b, &r, &mut out, 4);
  assert!(out.iter().all(|&v| v == 0xFF), "all pixels must be 0xFF");
}

#[test]
fn rgb_msb_bits12_downshift_by_4() {
  // BITS=12: logical 4080 >> 4 = 255.
  let r = as_le_u16(&msb::<12>(&[4080]));
  let g = as_le_u16(&msb::<12>(&[0]));
  let b = as_le_u16(&msb::<12>(&[0]));
  let mut out = [0u8; 3];
  gbr_to_rgb_msb_row::<12, false>(&g, &b, &r, &mut out, 1);
  assert_eq!(out[0], 255); // R channel
}

#[test]
fn rgb_msb_bits10_high_byte_is_independent_of_align() {
  // The net u8 of an MSB sample is the high byte `raw >> 8`. Sample 0x2AB
  // (683) << 6 = 0xAAC0 → high byte 0xAA = 170 = 683 >> 2.
  let r = as_le_u16(&msb::<10>(&[683]));
  let g = as_le_u16(&msb::<10>(&[0]));
  let b = as_le_u16(&msb::<10>(&[0]));
  let mut out = [0u8; 3];
  gbr_to_rgb_msb_row::<10, false>(&g, &b, &r, &mut out, 1);
  assert_eq!(out[0], 170);
}

// ---- parity: MSB(s) == high_bit(s) for every output variant -------------

#[test]
fn rgb_msb_matches_high_bit_on_recovered_samples() {
  // For each output kernel, encoding the same logical samples MSB-aligned
  // (this family) vs low-aligned (the sibling) must produce byte-identical
  // output.
  let samples_r: [u16; 5] = [0, 1, 511, 1000, 1023];
  let samples_g: [u16; 5] = [1023, 512, 256, 7, 0];
  let samples_b: [u16; 5] = [333, 0, 1023, 64, 900];
  let w = 5;

  let r_lo = as_le_u16(&samples_r);
  let g_lo = as_le_u16(&samples_g);
  let b_lo = as_le_u16(&samples_b);
  let r_hi = as_le_u16(&msb::<10>(&samples_r));
  let g_hi = as_le_u16(&msb::<10>(&samples_g));
  let b_hi = as_le_u16(&msb::<10>(&samples_b));

  // u8 RGB
  let mut a = [0u8; 15];
  let mut e = [0u8; 15];
  gbr_to_rgb_msb_row::<10, false>(&g_hi, &b_hi, &r_hi, &mut a, w);
  low::gbr_to_rgb_high_bit_row::<10, false>(&g_lo, &b_lo, &r_lo, &mut e, w);
  assert_eq!(a, e, "rgb u8");

  // u16 RGB
  let mut a = [0u16; 15];
  let mut e = [0u16; 15];
  gbr_to_rgb_u16_msb_row::<10, false>(&g_hi, &b_hi, &r_hi, &mut a, w);
  low::gbr_to_rgb_u16_high_bit_row::<10, false>(&g_lo, &b_lo, &r_lo, &mut e, w);
  assert_eq!(a, e, "rgb u16");

  // u8 RGBA opaque
  let mut a = [0u8; 20];
  let mut e = [0u8; 20];
  gbr_to_rgba_opaque_msb_row::<10, false>(&g_hi, &b_hi, &r_hi, &mut a, w);
  low::gbr_to_rgba_opaque_high_bit_row::<10, false>(&g_lo, &b_lo, &r_lo, &mut e, w);
  assert_eq!(a, e, "rgba u8 opaque");

  // u16 RGBA opaque
  let mut a = [0u16; 20];
  let mut e = [0u16; 20];
  gbr_to_rgba_opaque_u16_msb_row::<10, false>(&g_hi, &b_hi, &r_hi, &mut a, w);
  low::gbr_to_rgba_opaque_u16_high_bit_row::<10, false>(&g_lo, &b_lo, &r_lo, &mut e, w);
  assert_eq!(a, e, "rgba u16 opaque");

  // luma (full + limited)
  for full in [true, false] {
    let mut a = [0u16; 5];
    let mut e = [0u16; 5];
    gbr_to_luma_u16_msb_row::<10, false>(&g_hi, &b_hi, &r_hi, &mut a, w, KernelMatrix::Bt709, full);
    low::gbr_to_luma_u16_high_bit_row::<10, false>(
      &g_lo,
      &b_lo,
      &r_lo,
      &mut e,
      w,
      KernelMatrix::Bt709,
      full,
    );
    assert_eq!(a, e, "luma full_range={full}");
  }
}

#[test]
fn rgb_msb_bits12_matches_high_bit() {
  let samples_r: [u16; 4] = [0, 17, 2048, 4095];
  let samples_g: [u16; 4] = [4095, 1, 1024, 0];
  let samples_b: [u16; 4] = [123, 4095, 0, 777];
  let w = 4;
  let r_lo = as_le_u16(&samples_r);
  let g_lo = as_le_u16(&samples_g);
  let b_lo = as_le_u16(&samples_b);
  let r_hi = as_le_u16(&msb::<12>(&samples_r));
  let g_hi = as_le_u16(&msb::<12>(&samples_g));
  let b_hi = as_le_u16(&msb::<12>(&samples_b));

  let mut a = [0u16; 12];
  let mut e = [0u16; 12];
  gbr_to_rgb_u16_msb_row::<12, false>(&g_hi, &b_hi, &r_hi, &mut a, w);
  low::gbr_to_rgb_u16_high_bit_row::<12, false>(&g_lo, &b_lo, &r_lo, &mut e, w);
  assert_eq!(a, e);
}

// ---- u16 native output --------------------------------------------------

#[test]
fn rgb_u16_msb_recovers_native_sample() {
  let g = as_le_u16(&msb::<10>(&[111]));
  let b = as_le_u16(&msb::<10>(&[222]));
  let r = as_le_u16(&msb::<10>(&[333]));
  let mut out = [0u16; 3];
  gbr_to_rgb_u16_msb_row::<10, false>(&g, &b, &r, &mut out, 1);
  assert_eq!(out[0], 333); // R
  assert_eq!(out[1], 111); // G
  assert_eq!(out[2], 222); // B
}

// ---- opaque alpha constants ---------------------------------------------

#[test]
fn rgba_opaque_msb_u8_alpha_is_0xff() {
  let g = as_le_u16(&msb::<10>(&[10]));
  let b = as_le_u16(&msb::<10>(&[20]));
  let r = as_le_u16(&msb::<10>(&[30]));
  let mut out = [0u8; 4];
  gbr_to_rgba_opaque_msb_row::<10, false>(&g, &b, &r, &mut out, 1);
  assert_eq!(out[3], 0xFF);
}

#[test]
fn rgba_opaque_msb_u16_alpha_is_native_max() {
  let g = as_le_u16(&msb::<12>(&[10]));
  let b = as_le_u16(&msb::<12>(&[20]));
  let r = as_le_u16(&msb::<12>(&[30]));
  let mut out = [0u16; 4];
  gbr_to_rgba_opaque_u16_msb_row::<12, false>(&g, &b, &r, &mut out, 1);
  assert_eq!(out[3], (1u16 << 12) - 1); // 4095
}

// ---- big-endian recovery ------------------------------------------------

#[test]
fn rgb_msb_be_matches_le() {
  let samples_r: [u16; 3] = [1000, 7, 1023];
  let samples_g: [u16; 3] = [0, 1023, 512];
  let samples_b: [u16; 3] = [100, 256, 0];
  let w = 3;
  let r_le = as_le_u16(&msb::<10>(&samples_r));
  let g_le = as_le_u16(&msb::<10>(&samples_g));
  let b_le = as_le_u16(&msb::<10>(&samples_b));
  let r_be = as_be_u16(&msb::<10>(&samples_r));
  let g_be = as_be_u16(&msb::<10>(&samples_g));
  let b_be = as_be_u16(&msb::<10>(&samples_b));

  let mut le = [0u8; 9];
  let mut be = [0u8; 9];
  gbr_to_rgb_msb_row::<10, false>(&g_le, &b_le, &r_le, &mut le, w);
  gbr_to_rgb_msb_row::<10, true>(&g_be, &b_be, &r_be, &mut be, w);
  assert_eq!(le, be, "BE recovery must match LE");
}

#[test]
fn luma_msb_be_matches_le() {
  let samples_r: [u16; 3] = [1000, 7, 1023];
  let samples_g: [u16; 3] = [0, 1023, 512];
  let samples_b: [u16; 3] = [100, 256, 0];
  let w = 3;
  let r_le = as_le_u16(&msb::<10>(&samples_r));
  let g_le = as_le_u16(&msb::<10>(&samples_g));
  let b_le = as_le_u16(&msb::<10>(&samples_b));
  let r_be = as_be_u16(&msb::<10>(&samples_r));
  let g_be = as_be_u16(&msb::<10>(&samples_g));
  let b_be = as_be_u16(&msb::<10>(&samples_b));

  let mut le = [0u16; 3];
  let mut be = [0u16; 3];
  gbr_to_luma_u16_msb_row::<10, false>(&g_le, &b_le, &r_le, &mut le, w, KernelMatrix::Bt709, true);
  gbr_to_luma_u16_msb_row::<10, true>(&g_be, &b_be, &r_be, &mut be, w, KernelMatrix::Bt709, true);
  assert_eq!(le, be);
}
