//! Tests for `crate::row::scalar::planar_gbr_high_bit`.

use super::*;
use crate::KernelMatrix;

/// Re-encode a host-native u16 slice as LE-encoded byte storage. Kernels
/// called with `BE = false` recover the intended logical values via
/// `u16::from_le` on both LE (no-op) and BE (byte-swap) hosts.
fn as_le_u16(host: &[u16]) -> std::vec::Vec<u16> {
  host
    .iter()
    .map(|v| u16::from_ne_bytes(v.to_le_bytes()))
    .collect()
}

// ---- gbr_to_rgb_high_bit_row: u8 output, downshift ----------------------

#[test]
fn rgb_high_bit_bits10_channel_reorder() {
  // G=0, B=100, R=1000 → packed R,G,B = 1000>>2, 0>>2, 100>>2 = 250, 0, 25
  let g = as_le_u16(&[0u16; 1]);
  let b = as_le_u16(&[100u16; 1]);
  let r = as_le_u16(&[1000u16; 1]);
  let mut out = [0u8; 3];
  gbr_to_rgb_high_bit_row::<10, false>(&g, &b, &r, &mut out, 1);
  assert_eq!(out[0], 250); // R
  assert_eq!(out[1], 0); // G
  assert_eq!(out[2], 25); // B
}

#[test]
fn rgb_high_bit_bits10_max_value_becomes_0xff() {
  let max = (1u16 << 10) - 1; // 1023
  let g = as_le_u16(&[max; 4]);
  let b = as_le_u16(&[max; 4]);
  let r = as_le_u16(&[max; 4]);
  let mut out = [0u8; 12];
  gbr_to_rgb_high_bit_row::<10, false>(&g, &b, &r, &mut out, 4);
  assert!(out.iter().all(|&v| v == 0xFF), "all pixels must be 0xFF");
}

#[test]
fn rgb_high_bit_bits16_max_value_becomes_0xff() {
  let max = u16::MAX;
  let g = as_le_u16(&[max; 2]);
  let b = as_le_u16(&[max; 2]);
  let r = as_le_u16(&[max; 2]);
  let mut out = [0u8; 6];
  gbr_to_rgb_high_bit_row::<16, false>(&g, &b, &r, &mut out, 2);
  assert!(out.iter().all(|&v| v == 0xFF));
}

#[test]
fn rgb_high_bit_bits10_zero_becomes_zero() {
  let g = as_le_u16(&[0u16; 2]);
  let b = as_le_u16(&[0u16; 2]);
  let r = as_le_u16(&[0u16; 2]);
  let mut out = [0xFFu8; 6];
  gbr_to_rgb_high_bit_row::<10, false>(&g, &b, &r, &mut out, 2);
  assert!(out.iter().all(|&v| v == 0));
}

#[test]
fn rgb_high_bit_bits9_downshift_by_1() {
  // BITS=9: shift = 1. Value 510 >> 1 = 255.
  let g = as_le_u16(&[510u16; 1]);
  let b = as_le_u16(&[0u16; 1]);
  let r = as_le_u16(&[0u16; 1]);
  let mut out = [0u8; 3];
  gbr_to_rgb_high_bit_row::<9, false>(&g, &b, &r, &mut out, 1);
  assert_eq!(out[1], 255); // G channel
}

#[test]
fn rgb_high_bit_bits12_downshift_by_4() {
  // BITS=12: shift = 4. Value 4080 >> 4 = 255.
  let r = as_le_u16(&[4080u16; 1]);
  let g = as_le_u16(&[0u16; 1]);
  let b = as_le_u16(&[0u16; 1]);
  let mut out = [0u8; 3];
  gbr_to_rgb_high_bit_row::<12, false>(&g, &b, &r, &mut out, 1);
  assert_eq!(out[0], 255); // R channel
}

#[test]
fn rgb_high_bit_multiple_pixels_correct_layout() {
  // 3 pixels: (R,G,B) = (100,200,300>>2=75), (200>>2=50,0,0), (0,150>>2=37,50>>2=12)
  // BITS=10, shift=2
  let r = as_le_u16(&[400u16, 200u16, 0u16]);
  let g = as_le_u16(&[800u16, 0u16, 600u16]);
  let b = as_le_u16(&[300u16, 0u16, 200u16]);
  let mut out = [0u8; 9];
  gbr_to_rgb_high_bit_row::<10, false>(&g, &b, &r, &mut out, 3);
  // pixel 0: R=400>>2=100, G=800>>2=200, B=300>>2=75
  assert_eq!(out[0], 100);
  assert_eq!(out[1], 200);
  assert_eq!(out[2], 75);
  // pixel 1: R=200>>2=50, G=0, B=0
  assert_eq!(out[3], 50);
  assert_eq!(out[4], 0);
  assert_eq!(out[5], 0);
  // pixel 2: R=0, G=600>>2=150, B=200>>2=50
  assert_eq!(out[6], 0);
  assert_eq!(out[7], 150);
  assert_eq!(out[8], 50);
}

// ---- gbr_to_rgb_u16_high_bit_row: u16 output, no shift ------------------

#[test]
fn rgb_u16_high_bit_channel_reorder() {
  let g = as_le_u16(&[111u16; 1]);
  let b = as_le_u16(&[222u16; 1]);
  let r = as_le_u16(&[333u16; 1]);
  let mut out = [0u16; 3];
  gbr_to_rgb_u16_high_bit_row::<10, false>(&g, &b, &r, &mut out, 1);
  assert_eq!(out[0], 333); // R
  assert_eq!(out[1], 111); // G
  assert_eq!(out[2], 222); // B
}

#[test]
fn rgb_u16_high_bit_bits10_max_preserved() {
  let max = (1u16 << 10) - 1; // 1023
  let g = as_le_u16(&[max; 4]);
  let b = as_le_u16(&[max; 4]);
  let r = as_le_u16(&[max; 4]);
  let mut out = [0u16; 12];
  gbr_to_rgb_u16_high_bit_row::<10, false>(&g, &b, &r, &mut out, 4);
  assert!(out.iter().all(|&v| v == max));
}

#[test]
fn rgb_u16_high_bit_bits16_max_preserved() {
  let max = u16::MAX;
  let g = as_le_u16(&[max; 2]);
  let b = as_le_u16(&[max; 2]);
  let r = as_le_u16(&[max; 2]);
  let mut out = [0u16; 6];
  gbr_to_rgb_u16_high_bit_row::<16, false>(&g, &b, &r, &mut out, 2);
  assert!(out.iter().all(|&v| v == max));
}

#[test]
fn rgb_u16_high_bit_values_not_shifted() {
  // Verify that u16 output does NOT shift values (unlike u8 output).
  let g = as_le_u16(&[1000u16; 1]);
  let b = as_le_u16(&[2000u16; 1]);
  let r = as_le_u16(&[3000u16; 1]);
  let mut out = [0u16; 3];
  gbr_to_rgb_u16_high_bit_row::<12, false>(&g, &b, &r, &mut out, 1);
  assert_eq!(out[0], 3000); // R — unchanged
  assert_eq!(out[1], 1000); // G — unchanged
  assert_eq!(out[2], 2000); // B — unchanged
}

// ---- gbr_to_rgba_opaque_high_bit_row: u8 RGBA with constant alpha --------

#[test]
fn rgba_opaque_high_bit_bits10_alpha_is_0xff() {
  let max = (1u16 << 10) - 1;
  let g = as_le_u16(&[max; 4]);
  let b = as_le_u16(&[max; 4]);
  let r = as_le_u16(&[max; 4]);
  let mut out = [0u8; 16];
  gbr_to_rgba_opaque_high_bit_row::<10, false>(&g, &b, &r, &mut out, 4);
  for i in 0..4 {
    assert_eq!(out[i * 4 + 3], 0xFF, "alpha must be 0xFF at pixel {i}");
    assert_eq!(out[i * 4], 0xFF, "R must be 0xFF at pixel {i}");
  }
}

#[test]
fn rgba_opaque_high_bit_bits9_downshift_correct() {
  // BITS=9, shift=1. Value 510 >> 1 = 255.
  let g = as_le_u16(&[510u16; 1]);
  let b = as_le_u16(&[0u16; 1]);
  let r = as_le_u16(&[0u16; 1]);
  let mut out = [0u8; 4];
  gbr_to_rgba_opaque_high_bit_row::<9, false>(&g, &b, &r, &mut out, 1);
  assert_eq!(out[1], 255); // G
  assert_eq!(out[3], 0xFF); // alpha
}

// ---- gbr_to_rgba_opaque_u16_high_bit_row: u16 RGBA with constant alpha ---

#[test]
fn rgba_opaque_u16_high_bit_bits10_alpha_is_1023() {
  let g = as_le_u16(&[500u16; 2]);
  let b = as_le_u16(&[200u16; 2]);
  let r = as_le_u16(&[800u16; 2]);
  let mut out = [0u16; 8];
  gbr_to_rgba_opaque_u16_high_bit_row::<10, false>(&g, &b, &r, &mut out, 2);
  let opaque = (1u16 << 10) - 1; // 1023
  assert_eq!(out[3], opaque); // pixel 0 alpha
  assert_eq!(out[7], opaque); // pixel 1 alpha
  assert_eq!(out[0], 800); // R
  assert_eq!(out[1], 500); // G
  assert_eq!(out[2], 200); // B
}

#[test]
fn rgba_opaque_u16_high_bit_bits16_alpha_is_65535() {
  let g = as_le_u16(&[0u16; 1]);
  let b = as_le_u16(&[0u16; 1]);
  let r = as_le_u16(&[0u16; 1]);
  let mut out = [0u16; 4];
  gbr_to_rgba_opaque_u16_high_bit_row::<16, false>(&g, &b, &r, &mut out, 1);
  assert_eq!(out[3], u16::MAX);
}

#[test]
fn rgba_opaque_u16_high_bit_bits9_alpha_is_511() {
  let g = as_le_u16(&[0u16; 1]);
  let b = as_le_u16(&[0u16; 1]);
  let r = as_le_u16(&[0u16; 1]);
  let mut out = [0u16; 4];
  gbr_to_rgba_opaque_u16_high_bit_row::<9, false>(&g, &b, &r, &mut out, 1);
  assert_eq!(out[3], (1u16 << 9) - 1); // 511
}

// ---- gbra_to_rgba_high_bit_row: u8 RGBA with source alpha ----------------

#[test]
fn gbra_rgba_high_bit_bits10_source_alpha_downshifted() {
  // BITS=10, shift=2. Alpha value 512 >> 2 = 128.
  let g = as_le_u16(&[0u16; 1]);
  let b = as_le_u16(&[0u16; 1]);
  let r = as_le_u16(&[0u16; 1]);
  let a = as_le_u16(&[512u16; 1]);
  let mut out = [0u8; 4];
  gbra_to_rgba_high_bit_row::<10, false>(&g, &b, &r, &a, &mut out, 1);
  assert_eq!(out[3], 128); // alpha = 512 >> 2
}

#[test]
fn gbra_rgba_high_bit_bits10_max_alpha_is_0xff() {
  let max = (1u16 << 10) - 1;
  let g = as_le_u16(&[max; 2]);
  let b = as_le_u16(&[max; 2]);
  let r = as_le_u16(&[max; 2]);
  let a = as_le_u16(&[max; 2]);
  let mut out = [0u8; 8];
  gbra_to_rgba_high_bit_row::<10, false>(&g, &b, &r, &a, &mut out, 2);
  for i in 0..2 {
    assert_eq!(out[i * 4 + 3], 0xFF, "alpha must be 0xFF at pixel {i}");
  }
}

#[test]
fn gbra_rgba_high_bit_bits14_channel_reorder_and_shift() {
  // BITS=14, shift=6. R=16320 >> 6 = 255, G=0, B=0, A=8192 >> 6 = 128.
  let g = as_le_u16(&[0u16; 1]);
  let b = as_le_u16(&[0u16; 1]);
  let r = as_le_u16(&[16320u16; 1]);
  let a = as_le_u16(&[8192u16; 1]);
  let mut out = [0u8; 4];
  gbra_to_rgba_high_bit_row::<14, false>(&g, &b, &r, &a, &mut out, 1);
  assert_eq!(out[0], 255); // R
  assert_eq!(out[1], 0); // G
  assert_eq!(out[2], 0); // B
  assert_eq!(out[3], 128); // A
}

// ---- gbra_to_rgba_u16_high_bit_row: u16 RGBA with source alpha -----------

#[test]
fn gbra_rgba_u16_high_bit_source_alpha_preserved() {
  let g = as_le_u16(&[100u16; 1]);
  let b = as_le_u16(&[200u16; 1]);
  let r = as_le_u16(&[300u16; 1]);
  let a = as_le_u16(&[777u16; 1]);
  let mut out = [0u16; 4];
  gbra_to_rgba_u16_high_bit_row::<10, false>(&g, &b, &r, &a, &mut out, 1);
  assert_eq!(out[0], 300); // R
  assert_eq!(out[1], 100); // G
  assert_eq!(out[2], 200); // B
  assert_eq!(out[3], 777); // A — preserved as-is
}

#[test]
fn gbra_rgba_u16_high_bit_bits16_all_channels_preserved() {
  let g = as_le_u16(&[10000u16; 2]);
  let b = as_le_u16(&[20000u16; 2]);
  let r = as_le_u16(&[30000u16; 2]);
  let a = as_le_u16(&[40000u16; 2]);
  let mut out = [0u16; 8];
  gbra_to_rgba_u16_high_bit_row::<16, false>(&g, &b, &r, &a, &mut out, 2);
  for i in 0..2 {
    assert_eq!(out[i * 4], 30000);
    assert_eq!(out[i * 4 + 1], 10000);
    assert_eq!(out[i * 4 + 2], 20000);
    assert_eq!(out[i * 4 + 3], 40000);
  }
}

// ---- Round-trip parity: high-bit u8 output matches 8-bit source ----------

#[test]
fn rgb_high_bit_bits10_parity_with_scaled_8bit() {
  // val=128 in 8-bit; in 10-bit: 128 << 2 = 512. 512 >> 2 = 128.
  let val: u16 = 128u16 << 2;
  let g = as_le_u16(&[val; 8]);
  let b = as_le_u16(&[val; 8]);
  let r = as_le_u16(&[val; 8]);
  let mut out = [0u8; 24];
  gbr_to_rgb_high_bit_row::<10, false>(&g, &b, &r, &mut out, 8);
  assert!(out.iter().all(|&v| v == 128));
}

#[test]
fn rgb_high_bit_bits12_parity_with_scaled_8bit() {
  // val=200 in 8-bit; in 12-bit: 200 << 4 = 3200. 3200 >> 4 = 200.
  let val: u16 = 200u16 << 4;
  let g = as_le_u16(&[val; 4]);
  let b = as_le_u16(&[val; 4]);
  let r = as_le_u16(&[val; 4]);
  let mut out = [0u8; 12];
  gbr_to_rgb_high_bit_row::<12, false>(&g, &b, &r, &mut out, 4);
  assert!(out.iter().all(|&v| v == 200));
}

// ---- Upper-bits masking tests --------------------------------------------
// These tests verify that samples with bits above BITS set are masked
// correctly before processing, ensuring scalar/SIMD produce identical output.

#[test]
fn gbr_to_rgb_high_bit_masks_upper_bits_bits10() {
  // BITS=10, mask=0x03FF. Input 0x0CFF has upper bits set.
  // masked = 0x0CFF & 0x03FF = 0x00FF = 255. 255 >> 2 = 63 as u8.
  let dirty: u16 = 0x0CFF;
  let clean = dirty & 0x03FF;
  let expected_u8 = (clean >> 2) as u8;
  let g = as_le_u16(&[dirty; 1]);
  let b = as_le_u16(&[dirty; 1]);
  let r = as_le_u16(&[dirty; 1]);
  let mut out = [0u8; 3];
  gbr_to_rgb_high_bit_row::<10, false>(&g, &b, &r, &mut out, 1);
  assert_eq!(
    out[0], expected_u8,
    "R must equal masked-then-shifted value"
  );
  assert_eq!(
    out[1], expected_u8,
    "G must equal masked-then-shifted value"
  );
  assert_eq!(
    out[2], expected_u8,
    "B must equal masked-then-shifted value"
  );
}

#[test]
fn gbr_to_rgb_high_bit_masks_upper_bits_multiple_widths_bits10() {
  // Width sweep: [1, 7, 8, 16, 17, 32, 33, 64, 128, 130].
  let dirty: u16 = 0x0500; // BITS=10: mask&0x0500 = 0x0100=256; 256>>2=64.
  let clean = dirty & 0x03FF;
  let expected_u8 = (clean >> 2) as u8;
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = as_le_u16(&std::vec![dirty; w]);
    let b = as_le_u16(&std::vec![dirty; w]);
    let r = as_le_u16(&std::vec![dirty; w]);
    let mut out = std::vec![0u8; w * 3];
    gbr_to_rgb_high_bit_row::<10, false>(&g, &b, &r, &mut out, w);
    for i in 0..w {
      assert_eq!(out[i * 3], expected_u8, "R pixel {i} wrong at width {w}");
      assert_eq!(
        out[i * 3 + 1],
        expected_u8,
        "G pixel {i} wrong at width {w}"
      );
      assert_eq!(
        out[i * 3 + 2],
        expected_u8,
        "B pixel {i} wrong at width {w}"
      );
    }
  }
}

#[test]
fn gbra_to_rgba_high_bit_masks_upper_bits_alpha_bits10() {
  // Verify that the alpha channel is also masked before shifting.
  // BITS=10: dirty_alpha = 0x0800 | 512 = 0x0A00 = 2560.
  // masked = 2560 & 0x03FF = 0x0200 = 512. 512 >> 2 = 128.
  let dirty_rgb: u16 = 0x0400; // masked = 0 (upper bit only). 0>>2=0.
  let dirty_alpha: u16 = 0x0A00; // masked = 0x0200 = 512. 512>>2=128.
  let g = as_le_u16(&[dirty_rgb; 1]);
  let b = as_le_u16(&[dirty_rgb; 1]);
  let r = as_le_u16(&[dirty_rgb; 1]);
  let a = as_le_u16(&[dirty_alpha; 1]);
  let mut out = [0u8; 4];
  gbra_to_rgba_high_bit_row::<10, false>(&g, &b, &r, &a, &mut out, 1);
  assert_eq!(out[0], 0, "R (dirty, masked to 0)");
  assert_eq!(out[1], 0, "G (dirty, masked to 0)");
  assert_eq!(out[2], 0, "B (dirty, masked to 0)");
  assert_eq!(out[3], 128, "alpha must be masked then shifted");
}

#[test]
fn gbr_to_rgb_u16_high_bit_masks_upper_bits_bits10() {
  // u16-output: verify that masked sample is in the output (not raw dirty value).
  let dirty: u16 = 0x0CFF;
  let clean = dirty & 0x03FF; // = 0x00FF = 255
  let g = as_le_u16(&[dirty; 1]);
  let b = as_le_u16(&[dirty; 1]);
  let r = as_le_u16(&[dirty; 1]);
  let mut out = [0u16; 3];
  gbr_to_rgb_u16_high_bit_row::<10, false>(&g, &b, &r, &mut out, 1);
  assert_eq!(out[0], clean, "R u16 must be masked value");
  assert_eq!(out[1], clean, "G u16 must be masked value");
  assert_eq!(out[2], clean, "B u16 must be masked value");
}

#[test]
fn gbra_to_rgba_u16_high_bit_masks_upper_bits_bits10() {
  // u16 RGBA output: all channels masked.
  let dirty: u16 = 0x0555; // BITS=10: masked = 0x0555 & 0x03FF = 0x0155 = 341.
  let clean = dirty & 0x03FF;
  let g = as_le_u16(&[dirty; 1]);
  let b = as_le_u16(&[dirty; 1]);
  let r = as_le_u16(&[dirty; 1]);
  let a = as_le_u16(&[dirty; 1]);
  let mut out = [0u16; 4];
  gbra_to_rgba_u16_high_bit_row::<10, false>(&g, &b, &r, &a, &mut out, 1);
  assert_eq!(out[0], clean, "R u16 must be masked");
  assert_eq!(out[1], clean, "G u16 must be masked");
  assert_eq!(out[2], clean, "B u16 must be masked");
  assert_eq!(out[3], clean, "A u16 must be masked");
}

#[test]
fn gbr_to_rgba_opaque_high_bit_masks_upper_bits_bits10() {
  // u8 RGBA opaque: RGB channels masked, alpha always 0xFF.
  let dirty: u16 = 0x0CFF; // masked & 0x03FF = 0x00FF = 255. 255>>2=63.
  let clean = dirty & 0x03FF;
  let expected_u8 = (clean >> 2) as u8;
  let g = as_le_u16(&[dirty; 1]);
  let b = as_le_u16(&[dirty; 1]);
  let r = as_le_u16(&[dirty; 1]);
  let mut out = [0u8; 4];
  gbr_to_rgba_opaque_high_bit_row::<10, false>(&g, &b, &r, &mut out, 1);
  assert_eq!(out[0], expected_u8, "R must be masked");
  assert_eq!(out[1], expected_u8, "G must be masked");
  assert_eq!(out[2], expected_u8, "B must be masked");
  assert_eq!(out[3], 0xFF, "alpha must be 0xFF");
}

#[test]
fn gbr_to_rgba_opaque_u16_high_bit_masks_upper_bits_bits10() {
  // u16 RGBA opaque: RGB masked, alpha is opaque mask value.
  let dirty: u16 = 0x0CFF; // masked = 0x00FF = 255.
  let clean = dirty & 0x03FF;
  let g = as_le_u16(&[dirty; 1]);
  let b = as_le_u16(&[dirty; 1]);
  let r = as_le_u16(&[dirty; 1]);
  let mut out = [0u16; 4];
  gbr_to_rgba_opaque_u16_high_bit_row::<10, false>(&g, &b, &r, &mut out, 1);
  assert_eq!(out[0], clean, "R u16 must be masked");
  assert_eq!(out[1], clean, "G u16 must be masked");
  assert_eq!(out[2], clean, "B u16 must be masked");
  assert_eq!(out[3], (1u16 << 10) - 1, "alpha must be opaque 1023");
}

#[test]
fn gbr_to_rgb_high_bit_bits16_mask_is_noop() {
  // For BITS=16, mask = 0xFFFF. The AND is a no-op; verify that u16::MAX
  // samples pass through correctly (masked == original).
  let val = u16::MAX;
  let g = as_le_u16(&[val; 2]);
  let b = as_le_u16(&[val; 2]);
  let r = as_le_u16(&[val; 2]);
  let mut out = [0u8; 6];
  gbr_to_rgb_high_bit_row::<16, false>(&g, &b, &r, &mut out, 2);
  assert!(
    out.iter().all(|&v| v == 0xFF),
    "BITS=16: max sample => 0xFF"
  );
}

// ---- Cross-path consistency: direct GBRA vs masked RGB + separate alpha ---

#[test]
fn gbra_to_rgba_high_bit_cross_path_consistency_bits10() {
  // With upper-bits-set alpha: direct gbra_to_rgba == manual masking.
  // BITS=10, dirty_alpha = 0x0800 | 0x0100 = 0x0900; masked=0x0100=256; 256>>2=64.
  let dirty_alpha: u16 = 0x0900;
  let clean_alpha = dirty_alpha & 0x03FF; // 256
  let expected_a_u8 = (clean_alpha >> 2) as u8; // 64

  let r = as_le_u16(&[400u16; 1]); // in-range sample: 400 >> 2 = 100
  let g = as_le_u16(&[200u16; 1]);
  let b = as_le_u16(&[100u16; 1]);
  let a = as_le_u16(&[dirty_alpha; 1]);

  // Direct path
  let mut out_direct = [0u8; 4];
  gbra_to_rgba_high_bit_row::<10, false>(&g, &b, &r, &a, &mut out_direct, 1);

  // Manual path: apply mask to alpha, call with clean value
  let a_clean = as_le_u16(&[clean_alpha; 1]);
  let mut out_manual = [0u8; 4];
  gbra_to_rgba_high_bit_row::<10, false>(&g, &b, &r, &a_clean, &mut out_manual, 1);

  assert_eq!(
    out_direct, out_manual,
    "direct GBRA path must match manually-masked alpha path"
  );
  assert_eq!(out_direct[3], expected_a_u8, "alpha channel value");
}

// ---- gbr_to_luma_u16_high_bit_row: native-depth luma --------------------

#[test]
fn luma_u16_high_bit_bits10_max_white_not_banded() {
  // BITS=10: max = 1023. Old path gave (255 as u16) << 2 = 1020, not 1023.
  // New kernel must produce a value near 1023 for all-white input.
  let max = (1u16 << 10) - 1; // 1023
  let g = as_le_u16(&[max; 1]);
  let b = as_le_u16(&[max; 1]);
  let r = as_le_u16(&[max; 1]);
  let mut out = [0u16; 1];
  gbr_to_luma_u16_high_bit_row::<10, false>(&g, &b, &r, &mut out, 1, KernelMatrix::Bt709, true);
  // For BT.709 full-range all-white: Y = round(Kr*max + Kg*max + Kb*max).
  // = round((6966 + 23436 + 2366) / 32768 * 1023) ≈ round(32768/32768 * 1023) = 1023.
  assert!(
    out[0] >= 1020,
    "max-white luma_u16 must be near 1023 (was {}, old banded path gives 1020)",
    out[0]
  );
  assert!(
    out[0] <= 1023,
    "max-white luma_u16 must not exceed native max"
  );
}

#[test]
fn luma_u16_high_bit_bits12_max_white_not_banded() {
  // BITS=12: max = 4095. Old path: (255 as u16) << 4 = 4080.
  // New kernel should give a value in [4090, 4095].
  let max = (1u16 << 12) - 1; // 4095
  let g = as_le_u16(&[max; 1]);
  let b = as_le_u16(&[max; 1]);
  let r = as_le_u16(&[max; 1]);
  let mut out = [0u16; 1];
  gbr_to_luma_u16_high_bit_row::<12, false>(&g, &b, &r, &mut out, 1, KernelMatrix::Bt601, true);
  assert!(
    out[0] >= 4090,
    "max-white luma_u16 bits12 must be near 4095 (was {})",
    out[0]
  );
  assert!(out[0] <= 4095, "must not exceed native max");
}

#[test]
fn luma_u16_high_bit_bits16_max_white_not_banded() {
  // BITS=16: max = 65535. Old path: (255 as u16) << 8 = 65280.
  // New kernel (i64 path) should give a value in [65520, 65535].
  let max = u16::MAX;
  let g = as_le_u16(&[max; 1]);
  let b = as_le_u16(&[max; 1]);
  let r = as_le_u16(&[max; 1]);
  let mut out = [0u16; 1];
  gbr_to_luma_u16_high_bit_row::<16, false>(&g, &b, &r, &mut out, 1, KernelMatrix::Bt709, true);
  assert!(
    out[0] >= 65520,
    "max-white luma_u16 bits16 must be near 65535 (was {}), old banded gives 65280",
    out[0]
  );
  // u16 is bounded by type; kernel clamp ensures value stays in [0, native_max].
}

#[test]
fn luma_u16_high_bit_bits10_neutral_gray_midrange() {
  // BITS=10: mid = 512. Luma of neutral gray ≈ 512.
  let mid = 512u16;
  let g = as_le_u16(&[mid; 1]);
  let b = as_le_u16(&[mid; 1]);
  let r = as_le_u16(&[mid; 1]);
  let mut out = [0u16; 1];
  gbr_to_luma_u16_high_bit_row::<10, false>(&g, &b, &r, &mut out, 1, KernelMatrix::Bt709, true);
  assert!(
    out[0] >= 510 && out[0] <= 514,
    "neutral gray luma_u16 must be ~512 (was {})",
    out[0]
  );
}

#[test]
fn luma_u16_high_bit_bits10_zero_gives_zero() {
  let g = as_le_u16(&[0u16; 2]);
  let b = as_le_u16(&[0u16; 2]);
  let r = as_le_u16(&[0u16; 2]);
  let mut out = [0xFFFFu16; 2];
  gbr_to_luma_u16_high_bit_row::<10, false>(&g, &b, &r, &mut out, 2, KernelMatrix::Bt709, true);
  assert!(out.iter().all(|&v| v == 0), "all-black must give zero luma");
}

#[test]
fn luma_u16_high_bit_bits10_full_range_vs_limited_range() {
  // For mid-gray input, limited-range luma should be in [16<<2, 235<<2] = [64, 940].
  let mid = 512u16;
  let g = as_le_u16(&[mid; 1]);
  let b = as_le_u16(&[mid; 1]);
  let r = as_le_u16(&[mid; 1]);
  let mut out_full = [0u16; 1];
  let mut out_lim = [0u16; 1];
  gbr_to_luma_u16_high_bit_row::<10, false>(
    &g,
    &b,
    &r,
    &mut out_full,
    1,
    KernelMatrix::Bt601,
    true,
  );
  gbr_to_luma_u16_high_bit_row::<10, false>(
    &g,
    &b,
    &r,
    &mut out_lim,
    1,
    KernelMatrix::Bt601,
    false,
  );
  let y_off = 16u16 << 2; // 64
  let y_max = 235u16 << 2; // 940
  assert!(
    out_full[0] >= out_lim[0],
    "limited-range luma <= full-range luma for mid gray"
  );
  assert!(
    out_lim[0] >= y_off,
    "limited-range must be >= 64 (was {})",
    out_lim[0]
  );
  assert!(
    out_lim[0] <= y_max,
    "limited-range must be <= 940 (was {})",
    out_lim[0]
  );
}

#[test]
fn luma_u16_high_bit_bits16_limited_range_black_gives_min_offset() {
  // BITS=16: all-black limited-range should give Y_off = 16 << 8 = 4096.
  let g = as_le_u16(&[0u16; 1]);
  let b = as_le_u16(&[0u16; 1]);
  let r = as_le_u16(&[0u16; 1]);
  let mut out = [0u16; 1];
  gbr_to_luma_u16_high_bit_row::<16, false>(&g, &b, &r, &mut out, 1, KernelMatrix::Bt709, false);
  let y_off = 16u16 << 8; // 4096
  assert_eq!(
    out[0], y_off,
    "all-black limited-range must give Y_off={y_off}"
  );
}

// Limited-range native-depth scaling boundary regression.
//
// An 8-bit `219/255` Q15 scale collapses the top ~250 input codes onto
// the y_max clamp at BITS=16 (e.g. y_full = 65300 → 60160 instead of
// ~59955), destroying highlight gradation. The current implementation
// uses native-depth scaling `(y_full x range) / native_max` where
// `range = 219 << (BITS - 8)` and `native_max = (1 << BITS) - 1`.

#[test]
fn luma_u16_high_bit_bits16_limited_range_max_white_maps_to_y_max() {
  // BITS=16, all-white in: y_full clamps to native_max=65535;
  // y_lim = 4096 + 65535 x 56064 / 65535 = 60160 = 235 << 8.
  let g = as_le_u16(&[u16::MAX; 1]);
  let b = as_le_u16(&[u16::MAX; 1]);
  let r = as_le_u16(&[u16::MAX; 1]);
  let mut out = [0u16; 1];
  gbr_to_luma_u16_high_bit_row::<16, false>(&g, &b, &r, &mut out, 1, KernelMatrix::Bt709, false);
  let y_max = 235u16 << 8; // 60160
  assert_eq!(
    out[0], y_max,
    "all-white limited-range must give Y_max={y_max}"
  );
}

#[test]
fn luma_u16_high_bit_bits16_limited_range_near_white_keeps_gradation() {
  // BITS=16, BT.709 luma weights ≈ Kr=0.2126, Kg=0.7152, Kb=0.0722.
  // Setting all 3 channels equal makes the matrix multiply produce
  // y_full ≈ input, so we can probe the limited-range scaling at
  // specific y_full values. y_full = 65000 / 65300 / 65500 must each
  // produce a distinct y_lim — an 8-bit ratio would clamp the top
  // two onto y_max=60160, destroying the gradation.
  for &v in &[65000u16, 65300, 65500] {
    let g = as_le_u16(&[v; 1]);
    let b = as_le_u16(&[v; 1]);
    let r = as_le_u16(&[v; 1]);
    let mut out = [0u16; 1];
    gbr_to_luma_u16_high_bit_row::<16, false>(&g, &b, &r, &mut out, 1, KernelMatrix::Bt709, false);
    // Native-depth limited-range: y_lim = 4096 + v x 56064 / 65535
    let expected = 4096 + ((v as u64 * 56064 + 65535 / 2) / 65535) as u16;
    // Allow ±1 LSB for matrix-multiply rounding (BT.709 weights aren't
    // exactly 1.0 even with all channels equal; tiny Q15 residue).
    let diff = (out[0] as i32 - expected as i32).abs();
    assert!(
      diff <= 1,
      "v={v} expected ≈{expected} got {} (diff {diff})",
      out[0]
    );
    // The bug-collapse value would be 60160 (y_max). Reject any
    // result above 60160 — that means we re-introduced the clamp.
    assert!(
      out[0] < 60160 || (v >= 65500),
      "limited-range must not clamp at v={v} (got {})",
      out[0]
    );
  }
}

#[test]
fn luma_u16_high_bit_bits10_limited_range_endpoints() {
  // BITS=10: y_off=64 (=16<<2), y_max=940 (=235<<2), native_max=1023.
  // BT.709 luma at all-equal channels passes y_full ≈ input through.
  // Test endpoint values: 0 → 64, 1023 → 940.
  let cases: &[(u16, u16)] = &[(0, 64), (1023, 940)];
  for &(input, expected) in cases {
    let g = as_le_u16(&[input; 1]);
    let b = as_le_u16(&[input; 1]);
    let r = as_le_u16(&[input; 1]);
    let mut out = [0u16; 1];
    gbr_to_luma_u16_high_bit_row::<10, false>(&g, &b, &r, &mut out, 1, KernelMatrix::Bt709, false);
    let diff = (out[0] as i32 - expected as i32).abs();
    assert!(
      diff <= 1,
      "BITS=10 input={input} expected ≈{expected} got {}",
      out[0]
    );
  }
}

// ---- BE vs LE parity: scalar<BITS, true> on BE-encoded storage must match -
// scalar<BITS, false> on LE-encoded storage. Each plane is built from the ---
// same host-native `intended` buffer, then re-encoded with `to_le_bytes` /  -
// `to_be_bytes` so the kernels' `from_le` / `from_be` decode it back to the -
// same logical values on every host. The previous helper (`swap_bytes` of  -
// host-native data) was vacuous: on BE the `<false>` path would byte-swap   -
// host-native into wrong logical values while the `<true>` path on the      -
// swapped buffer produced the same wrong values, so equality could pass on  -
// a corrupted decode. Each test now also pins the LE output to an absolute  -
// expected value computed independently from `intended`. -------------------

/// Re-encode host-native `u16` samples as BE byte storage. On a BE host this
/// is identity; on a LE host each element is byte-swapped so the kernel's
/// `from_be` recovers the original logical value. Mirror of the existing
/// `as_le_u16` helper above.
fn as_be_u16(host: &[u16]) -> std::vec::Vec<u16> {
  host
    .iter()
    .map(|v| u16::from_ne_bytes(v.to_be_bytes()))
    .collect()
}

fn rand_plane<const BITS: u32>(seed: u32, n: usize) -> std::vec::Vec<u16> {
  let mask = (1u32 << BITS) - 1;
  let mut s = seed;
  (0..n)
    .map(|_| {
      s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
      (s & mask) as u16
    })
    .collect()
}

/// Independent scalar reference for the `gbr_to_rgb_high_bit_row` kernel:
/// reorders the planes to packed R, G, B and applies the `>> (BITS - 8)`
/// downshift on host-native logical samples. Used to pin the LE path's
/// output absolutely (so equality cannot pass on equally corrupted decodes).
fn ref_gbr_to_rgb_high_bit<const BITS: u32>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  width: usize,
) -> std::vec::Vec<u8> {
  let shift = BITS - 8;
  let mut out = std::vec![0u8; width * 3];
  for x in 0..width {
    out[x * 3] = (r[x] >> shift) as u8;
    out[x * 3 + 1] = (g[x] >> shift) as u8;
    out[x * 3 + 2] = (b[x] >> shift) as u8;
  }
  out
}

/// Independent scalar reference for `gbr_to_rgba_opaque_high_bit_row`.
fn ref_gbr_to_rgba_opaque_high_bit<const BITS: u32>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  width: usize,
) -> std::vec::Vec<u8> {
  let shift = BITS - 8;
  let mut out = std::vec![0u8; width * 4];
  for x in 0..width {
    out[x * 4] = (r[x] >> shift) as u8;
    out[x * 4 + 1] = (g[x] >> shift) as u8;
    out[x * 4 + 2] = (b[x] >> shift) as u8;
    out[x * 4 + 3] = 0xFF;
  }
  out
}

/// Independent scalar reference for `gbra_to_rgba_high_bit_row`.
fn ref_gbra_to_rgba_high_bit<const BITS: u32>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  a: &[u16],
  width: usize,
) -> std::vec::Vec<u8> {
  let shift = BITS - 8;
  let mut out = std::vec![0u8; width * 4];
  for x in 0..width {
    out[x * 4] = (r[x] >> shift) as u8;
    out[x * 4 + 1] = (g[x] >> shift) as u8;
    out[x * 4 + 2] = (b[x] >> shift) as u8;
    out[x * 4 + 3] = (a[x] >> shift) as u8;
  }
  out
}

/// Independent scalar reference for `gbr_to_rgb_u16_high_bit_row`.
fn ref_gbr_to_rgb_u16_high_bit<const BITS: u32>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  width: usize,
) -> std::vec::Vec<u16> {
  let mask: u16 = ((1u32 << BITS) - 1) as u16;
  let mut out = std::vec![0u16; width * 3];
  for x in 0..width {
    out[x * 3] = r[x] & mask;
    out[x * 3 + 1] = g[x] & mask;
    out[x * 3 + 2] = b[x] & mask;
  }
  out
}

/// Independent scalar reference for `gbra_to_rgba_u16_high_bit_row`.
fn ref_gbra_to_rgba_u16_high_bit<const BITS: u32>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  a: &[u16],
  width: usize,
) -> std::vec::Vec<u16> {
  let mask: u16 = ((1u32 << BITS) - 1) as u16;
  let mut out = std::vec![0u16; width * 4];
  for x in 0..width {
    out[x * 4] = r[x] & mask;
    out[x * 4 + 1] = g[x] & mask;
    out[x * 4 + 2] = b[x] & mask;
    out[x * 4 + 3] = a[x] & mask;
  }
  out
}

#[test]
fn scalar_gbr_to_rgb_high_bit_be_parity_bits10() {
  for w in [1usize, 7, 8, 9, 17, 33, 65] {
    let g = rand_plane::<10>(0xAAAA, w);
    let b = rand_plane::<10>(0xBBBB, w);
    let r = rand_plane::<10>(0xCCCC, w);
    let g_le = as_le_u16(&g);
    let b_le = as_le_u16(&b);
    let r_le = as_le_u16(&r);
    let g_be = as_be_u16(&g);
    let b_be = as_be_u16(&b);
    let r_be = as_be_u16(&r);
    let mut out_le = std::vec![0u8; w * 3];
    let mut out_be = std::vec![0u8; w * 3];
    gbr_to_rgb_high_bit_row::<10, false>(&g_le, &b_le, &r_le, &mut out_le, w);
    gbr_to_rgb_high_bit_row::<10, true>(&g_be, &b_be, &r_be, &mut out_be, w);
    let expected = ref_gbr_to_rgb_high_bit::<10>(&g, &b, &r, w);
    assert_eq!(
      out_le, expected,
      "scalar LE path does not match independent reference (gbr_to_rgb bits10 w={w})"
    );
    assert_eq!(
      out_le, out_be,
      "scalar BE/LE mismatch gbr_to_rgb bits10 w={w}"
    );
  }
}

#[test]
fn scalar_gbr_to_rgb_high_bit_be_parity_bits16() {
  for w in [1usize, 7, 8, 9, 17, 33, 65] {
    let g = rand_plane::<16>(0xAAAA, w);
    let b = rand_plane::<16>(0xBBBB, w);
    let r = rand_plane::<16>(0xCCCC, w);
    let g_le = as_le_u16(&g);
    let b_le = as_le_u16(&b);
    let r_le = as_le_u16(&r);
    let g_be = as_be_u16(&g);
    let b_be = as_be_u16(&b);
    let r_be = as_be_u16(&r);
    let mut out_le = std::vec![0u8; w * 3];
    let mut out_be = std::vec![0u8; w * 3];
    gbr_to_rgb_high_bit_row::<16, false>(&g_le, &b_le, &r_le, &mut out_le, w);
    gbr_to_rgb_high_bit_row::<16, true>(&g_be, &b_be, &r_be, &mut out_be, w);
    let expected = ref_gbr_to_rgb_high_bit::<16>(&g, &b, &r, w);
    assert_eq!(
      out_le, expected,
      "scalar LE path does not match independent reference (gbr_to_rgb bits16 w={w})"
    );
    assert_eq!(
      out_le, out_be,
      "scalar BE/LE mismatch gbr_to_rgb bits16 w={w}"
    );
  }
}

#[test]
fn scalar_gbr_to_rgba_opaque_high_bit_be_parity_bits10() {
  for w in [1usize, 7, 8, 9, 17] {
    let g = rand_plane::<10>(0xAAAA, w);
    let b = rand_plane::<10>(0xBBBB, w);
    let r = rand_plane::<10>(0xCCCC, w);
    let g_le = as_le_u16(&g);
    let b_le = as_le_u16(&b);
    let r_le = as_le_u16(&r);
    let g_be = as_be_u16(&g);
    let b_be = as_be_u16(&b);
    let r_be = as_be_u16(&r);
    let mut out_le = std::vec![0u8; w * 4];
    let mut out_be = std::vec![0u8; w * 4];
    gbr_to_rgba_opaque_high_bit_row::<10, false>(&g_le, &b_le, &r_le, &mut out_le, w);
    gbr_to_rgba_opaque_high_bit_row::<10, true>(&g_be, &b_be, &r_be, &mut out_be, w);
    let expected = ref_gbr_to_rgba_opaque_high_bit::<10>(&g, &b, &r, w);
    assert_eq!(
      out_le, expected,
      "scalar LE path does not match independent reference (gbr_to_rgba_opaque bits10 w={w})"
    );
    assert_eq!(
      out_le, out_be,
      "scalar BE/LE mismatch gbr_to_rgba_opaque bits10 w={w}"
    );
  }
}

#[test]
fn scalar_gbra_to_rgba_high_bit_be_parity_bits10() {
  for w in [1usize, 7, 8, 9, 17] {
    let g = rand_plane::<10>(0xAAAA, w);
    let b = rand_plane::<10>(0xBBBB, w);
    let r = rand_plane::<10>(0xCCCC, w);
    let a = rand_plane::<10>(0xDDDD, w);
    let g_le = as_le_u16(&g);
    let b_le = as_le_u16(&b);
    let r_le = as_le_u16(&r);
    let a_le = as_le_u16(&a);
    let g_be = as_be_u16(&g);
    let b_be = as_be_u16(&b);
    let r_be = as_be_u16(&r);
    let a_be = as_be_u16(&a);
    let mut out_le = std::vec![0u8; w * 4];
    let mut out_be = std::vec![0u8; w * 4];
    gbra_to_rgba_high_bit_row::<10, false>(&g_le, &b_le, &r_le, &a_le, &mut out_le, w);
    gbra_to_rgba_high_bit_row::<10, true>(&g_be, &b_be, &r_be, &a_be, &mut out_be, w);
    let expected = ref_gbra_to_rgba_high_bit::<10>(&g, &b, &r, &a, w);
    assert_eq!(
      out_le, expected,
      "scalar LE path does not match independent reference (gbra_to_rgba bits10 w={w})"
    );
    assert_eq!(
      out_le, out_be,
      "scalar BE/LE mismatch gbra_to_rgba bits10 w={w}"
    );
  }
}

#[test]
fn scalar_gbr_to_rgb_u16_high_bit_be_parity_bits10() {
  for w in [1usize, 7, 8, 9, 17] {
    let g = rand_plane::<10>(0xAAAA, w);
    let b = rand_plane::<10>(0xBBBB, w);
    let r = rand_plane::<10>(0xCCCC, w);
    let g_le = as_le_u16(&g);
    let b_le = as_le_u16(&b);
    let r_le = as_le_u16(&r);
    let g_be = as_be_u16(&g);
    let b_be = as_be_u16(&b);
    let r_be = as_be_u16(&r);
    let mut out_le = std::vec![0u16; w * 3];
    let mut out_be = std::vec![0u16; w * 3];
    gbr_to_rgb_u16_high_bit_row::<10, false>(&g_le, &b_le, &r_le, &mut out_le, w);
    gbr_to_rgb_u16_high_bit_row::<10, true>(&g_be, &b_be, &r_be, &mut out_be, w);
    let expected = ref_gbr_to_rgb_u16_high_bit::<10>(&g, &b, &r, w);
    assert_eq!(
      out_le, expected,
      "scalar LE path does not match independent reference (gbr_to_rgb_u16 bits10 w={w})"
    );
    assert_eq!(
      out_le, out_be,
      "scalar BE/LE mismatch gbr_to_rgb_u16 bits10 w={w}"
    );
  }
}

#[test]
fn scalar_gbra_to_rgba_u16_high_bit_be_parity_bits10() {
  for w in [1usize, 7, 8, 9, 17] {
    let g = rand_plane::<10>(0xAAAA, w);
    let b = rand_plane::<10>(0xBBBB, w);
    let r = rand_plane::<10>(0xCCCC, w);
    let a = rand_plane::<10>(0xDDDD, w);
    let g_le = as_le_u16(&g);
    let b_le = as_le_u16(&b);
    let r_le = as_le_u16(&r);
    let a_le = as_le_u16(&a);
    let g_be = as_be_u16(&g);
    let b_be = as_be_u16(&b);
    let r_be = as_be_u16(&r);
    let a_be = as_be_u16(&a);
    let mut out_le = std::vec![0u16; w * 4];
    let mut out_be = std::vec![0u16; w * 4];
    gbra_to_rgba_u16_high_bit_row::<10, false>(&g_le, &b_le, &r_le, &a_le, &mut out_le, w);
    gbra_to_rgba_u16_high_bit_row::<10, true>(&g_be, &b_be, &r_be, &a_be, &mut out_be, w);
    let expected = ref_gbra_to_rgba_u16_high_bit::<10>(&g, &b, &r, &a, w);
    assert_eq!(
      out_le, expected,
      "scalar LE path does not match independent reference (gbra_to_rgba_u16 bits10 w={w})"
    );
    assert_eq!(
      out_le, out_be,
      "scalar BE/LE mismatch gbra_to_rgba_u16 bits10 w={w}"
    );
  }
}
