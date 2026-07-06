//! Tests for `crate::row::scalar::grayf16`.

use super::*;
use half::f16;

/// Re-encode a host-native f16 slice as LE-encoded byte storage (each element
/// stored with LE u16-bit byte layout). Kernels called with `BE = false`
/// recover the intended host-native value via `u16::from_le` on both LE
/// (no-op) and BE (byte-swap) hosts.
fn as_le_f16(host: &[f16]) -> std::vec::Vec<f16> {
  host
    .iter()
    .map(|v| f16::from_bits(u16::from_ne_bytes(v.to_bits().to_le_bytes())))
    .collect()
}

/// Mirror of `as_le_f16` for kernels invoked with `BE = true`.
fn as_be_f16(host: &[f16]) -> std::vec::Vec<f16> {
  host
    .iter()
    .map(|v| f16::from_bits(u16::from_ne_bytes(v.to_bits().to_be_bytes())))
    .collect()
}

fn h(v: f32) -> f16 {
  f16::from_f32(v)
}

// ---- grayf16_to_rgb_row --------------------------------------------------

#[cfg_attr(
  miri,
  ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
)]
#[test]
fn grayf16_to_rgb_zero() {
  let plane = as_le_f16(&[h(0.0)]);
  let mut out = [0xFFu8; 3];
  grayf16_to_rgb_row::<false>(&plane, &mut out, 1);
  assert_eq!(out, [0, 0, 0]);
}

#[cfg_attr(
  miri,
  ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
)]
#[test]
fn grayf16_to_rgb_max() {
  let plane = as_le_f16(&[h(1.0)]);
  let mut out = [0u8; 3];
  grayf16_to_rgb_row::<false>(&plane, &mut out, 1);
  assert_eq!(out, [255, 255, 255]);
}

#[cfg_attr(
  miri,
  ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
)]
#[test]
fn grayf16_to_rgb_mid() {
  // 0.5 is exactly representable in f16. 0.5 * 255 + 0.5 = 128.0 → 128.
  let plane = as_le_f16(&[h(0.5)]);
  let mut out = [0u8; 3];
  grayf16_to_rgb_row::<false>(&plane, &mut out, 1);
  assert_eq!(out, [128, 128, 128]);
}

#[cfg_attr(
  miri,
  ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
)]
#[test]
fn grayf16_to_rgb_saturates_high() {
  let plane = as_le_f16(&[h(1.5)]);
  let mut out = [0u8; 3];
  grayf16_to_rgb_row::<false>(&plane, &mut out, 1);
  assert_eq!(out, [255, 255, 255]);
}

#[cfg_attr(
  miri,
  ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
)]
#[test]
fn grayf16_to_rgb_saturates_low() {
  let plane = as_le_f16(&[h(-0.1)]);
  let mut out = [0xFFu8; 3];
  grayf16_to_rgb_row::<false>(&plane, &mut out, 1);
  assert_eq!(out, [0, 0, 0]);
}

// ---- grayf16_to_rgba_row -------------------------------------------------

#[cfg_attr(
  miri,
  ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
)]
#[test]
fn grayf16_to_rgba_zero_alpha_opaque() {
  let plane = as_le_f16(&[h(0.0)]);
  let mut out = [0u8; 4];
  grayf16_to_rgba_row::<false>(&plane, &mut out, 1);
  assert_eq!(out, [0, 0, 0, 0xFF]);
}

#[cfg_attr(
  miri,
  ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
)]
#[test]
fn grayf16_to_rgba_max_alpha_opaque() {
  let plane = as_le_f16(&[h(1.0)]);
  let mut out = [0u8; 4];
  grayf16_to_rgba_row::<false>(&plane, &mut out, 1);
  assert_eq!(out, [255, 255, 255, 0xFF]);
}

// ---- grayf16_to_rgb_u16_row ----------------------------------------------

#[cfg_attr(
  miri,
  ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
)]
#[test]
fn grayf16_to_rgb_u16_zero() {
  let plane = as_le_f16(&[h(0.0)]);
  let mut out = [0xFFFFu16; 3];
  grayf16_to_rgb_u16_row::<false>(&plane, &mut out, 1);
  assert_eq!(out, [0, 0, 0]);
}

#[cfg_attr(
  miri,
  ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
)]
#[test]
fn grayf16_to_rgb_u16_max() {
  let plane = as_le_f16(&[h(1.0)]);
  let mut out = [0u16; 3];
  grayf16_to_rgb_u16_row::<false>(&plane, &mut out, 1);
  assert_eq!(out, [65535, 65535, 65535]);
}

#[cfg_attr(
  miri,
  ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
)]
#[test]
fn grayf16_to_rgb_u16_saturates_high() {
  let plane = as_le_f16(&[h(2.0)]);
  let mut out = [0u16; 3];
  grayf16_to_rgb_u16_row::<false>(&plane, &mut out, 1);
  assert_eq!(out, [65535, 65535, 65535]);
}

// ---- grayf16_to_rgba_u16_row ---------------------------------------------

#[cfg_attr(
  miri,
  ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
)]
#[test]
fn grayf16_to_rgba_u16_opaque() {
  let plane = as_le_f16(&[h(1.0)]);
  let mut out = [0u16; 4];
  grayf16_to_rgba_u16_row::<false>(&plane, &mut out, 1);
  assert_eq!(out, [65535, 65535, 65535, 0xFFFF]);
}

// ---- grayf16_to_rgb_f32_row ----------------------------------------------

#[cfg_attr(
  miri,
  ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
)]
#[test]
fn grayf16_to_rgb_f32_lossless_replicate() {
  // 1.5 is exactly representable in f16; widening to f32 is exact.
  let plane = as_le_f16(&[h(1.5)]);
  let mut out = [0.0f32; 3];
  grayf16_to_rgb_f32_row::<false>(&plane, &mut out, 1);
  assert_eq!(out, [1.5, 1.5, 1.5]);
}

#[cfg_attr(
  miri,
  ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
)]
#[test]
fn grayf16_to_rgb_f32_negative_preserved() {
  let plane = as_le_f16(&[h(-0.5)]);
  let mut out = [0.0f32; 3];
  grayf16_to_rgb_f32_row::<false>(&plane, &mut out, 1);
  assert_eq!(out, [-0.5, -0.5, -0.5]);
}

// ---- grayf16_to_luma_row -------------------------------------------------

#[cfg_attr(
  miri,
  ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
)]
#[test]
fn grayf16_to_luma_zero() {
  let plane = as_le_f16(&[h(0.0)]);
  let mut out = [0xFFu8; 1];
  grayf16_to_luma_row::<false>(&plane, &mut out, 1);
  assert_eq!(out, [0]);
}

#[cfg_attr(
  miri,
  ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
)]
#[test]
fn grayf16_to_luma_max() {
  let plane = as_le_f16(&[h(1.0)]);
  let mut out = [0u8; 1];
  grayf16_to_luma_row::<false>(&plane, &mut out, 1);
  assert_eq!(out, [255]);
}

// ---- grayf16_to_luma_u16_row ---------------------------------------------

#[cfg_attr(
  miri,
  ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
)]
#[test]
fn grayf16_to_luma_u16_max() {
  let plane = as_le_f16(&[h(1.0)]);
  let mut out = [0u16; 1];
  grayf16_to_luma_u16_row::<false>(&plane, &mut out, 1);
  assert_eq!(out, [65535]);
}

// ---- grayf16_to_luma_f32_row ---------------------------------------------

#[cfg_attr(
  miri,
  ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
)]
#[test]
fn grayf16_to_luma_f32_identity() {
  let intended = [h(0.0), h(0.5), h(1.0), h(1.5), h(-0.1)];
  let plane = as_le_f16(&intended);
  let mut out = [99.0f32; 5];
  grayf16_to_luma_f32_row::<false>(&plane, &mut out, 5);
  // Lossless widen — each output equals the f16's f32 widening.
  let expected: std::vec::Vec<f32> = intended.iter().map(|v| v.to_f32()).collect();
  assert_eq!(out.as_slice(), expected.as_slice());
}

// ---- grayf16_to_hsv_row --------------------------------------------------

#[cfg_attr(
  miri,
  ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
)]
#[test]
fn grayf16_to_hsv_zero() {
  let plane = as_le_f16(&[h(0.0)]);
  let mut hue = [0xFFu8; 1];
  let mut s = [0xFFu8; 1];
  let mut v = [0u8; 1];
  grayf16_to_hsv_row::<false>(&plane, &mut hue, &mut s, &mut v, 1);
  assert_eq!(hue[0], 0, "H must be 0 for achromatic source");
  assert_eq!(s[0], 0, "S must be 0 for achromatic source");
  assert_eq!(v[0], 0);
}

#[cfg_attr(
  miri,
  ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
)]
#[test]
fn grayf16_to_hsv_max() {
  let plane = as_le_f16(&[h(1.0)]);
  let mut hue = [0u8; 1];
  let mut s = [0u8; 1];
  let mut v = [0u8; 1];
  grayf16_to_hsv_row::<false>(&plane, &mut hue, &mut s, &mut v, 1);
  assert_eq!(hue[0], 0);
  assert_eq!(s[0], 0);
  assert_eq!(v[0], 255);
}

#[cfg_attr(
  miri,
  ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
)]
#[test]
fn grayf16_to_hsv_mid() {
  let plane = as_le_f16(&[h(0.5)]);
  let mut hue = [0u8; 1];
  let mut s = [0u8; 1];
  let mut v = [0u8; 1];
  grayf16_to_hsv_row::<false>(&plane, &mut hue, &mut s, &mut v, 1);
  assert_eq!(hue[0], 0);
  assert_eq!(s[0], 0);
  assert_eq!(v[0], 128);
}

#[cfg_attr(
  miri,
  ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
)]
#[test]
fn grayf16_to_hsv_clamps_hdr() {
  let plane = as_le_f16(&[h(2.0)]);
  let mut hue = [0u8; 1];
  let mut s = [0u8; 1];
  let mut v = [0u8; 1];
  grayf16_to_hsv_row::<false>(&plane, &mut hue, &mut s, &mut v, 1);
  assert_eq!(v[0], 255);
}

#[cfg_attr(
  miri,
  ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
)]
#[test]
fn grayf16_to_rgb_multi_pixel() {
  let plane = as_le_f16(&[h(0.0), h(1.0), h(0.5)]);
  let mut out = [0u8; 9];
  grayf16_to_rgb_row::<false>(&plane, &mut out, 3);
  assert_eq!(&out[0..3], &[0, 0, 0]);
  assert_eq!(&out[3..6], &[255, 255, 255]);
  assert_eq!(&out[6..9], &[128, 128, 128]);
}

// ---- BE parity tests: grayf16 ---------------------------------------------
// Build a single host-native `intended` fixture, materialise it as LE-encoded
// and BE-encoded byte storage, run both `<false>` and `<true>` kernels, and
// pin each output against an absolute scalar reference so the parity assertion
// cannot pass on two equally corrupt decodes.

fn ref_grayf16_to_rgb(intended: &[f16], width: usize) -> std::vec::Vec<u8> {
  let mut out = std::vec![0u8; width * 3];
  for (x, &y) in intended[..width].iter().enumerate() {
    let v = f32_to_u8(y.to_f32());
    out[x * 3] = v;
    out[x * 3 + 1] = v;
    out[x * 3 + 2] = v;
  }
  out
}

fn ref_grayf16_to_luma(intended: &[f16], width: usize) -> std::vec::Vec<u8> {
  let mut out = std::vec![0u8; width];
  for (x, &y) in intended[..width].iter().enumerate() {
    out[x] = f32_to_u8(y.to_f32());
  }
  out
}

#[cfg_attr(
  miri,
  ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
)]
#[test]
fn grayf16_be_parity_rgb() {
  let intended = [h(0.5)];
  let le = as_le_f16(&intended);
  let be = as_be_f16(&intended);
  let mut out_le = [0u8; 3];
  let mut out_be = [0u8; 3];
  grayf16_to_rgb_row::<false>(&le, &mut out_le, 1);
  grayf16_to_rgb_row::<true>(&be, &mut out_be, 1);
  let expected = ref_grayf16_to_rgb(&intended, 1);
  assert_eq!(out_le.as_slice(), expected, "LE path must match reference");
  assert_eq!(out_be.as_slice(), expected, "BE path must match reference");
  assert_eq!(out_le, out_be, "BE and LE grayf16 rgb outputs must agree");
}

#[cfg_attr(
  miri,
  ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
)]
#[test]
fn grayf16_be_parity_luma() {
  let intended = [h(0.25)];
  let le = as_le_f16(&intended);
  let be = as_be_f16(&intended);
  let mut out_le = [0u8; 1];
  let mut out_be = [0u8; 1];
  grayf16_to_luma_row::<false>(&le, &mut out_le, 1);
  grayf16_to_luma_row::<true>(&be, &mut out_be, 1);
  let expected = ref_grayf16_to_luma(&intended, 1);
  assert_eq!(out_le.as_slice(), expected, "LE path must match reference");
  assert_eq!(out_be.as_slice(), expected, "BE path must match reference");
  assert_eq!(out_le, out_be, "BE and LE grayf16 luma outputs must agree");
}

#[cfg_attr(
  miri,
  ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
)]
#[test]
fn grayf16_to_luma_f32_row_be_le_parity_lossless() {
  // Mix of normal, HDR, negative, subnormal, and exact-zero f16 values.
  let intended: std::vec::Vec<f16> = std::vec![
    h(0.25),
    h(1.5),
    h(-0.5),
    f16::from_bits(0x0001),
    h(0.0),
    h(2048.0)
  ];
  let width = intended.len();
  let le = as_le_f16(&intended);
  let be = as_be_f16(&intended);
  let mut out_le = std::vec![0.0f32; width];
  let mut out_be = std::vec![0.0f32; width];
  grayf16_to_luma_f32_row::<false>(&le, &mut out_le, width);
  grayf16_to_luma_f32_row::<true>(&be, &mut out_be, width);
  let expected: std::vec::Vec<f32> = intended.iter().map(|v| v.to_f32()).collect();
  let bits_le: std::vec::Vec<u32> = out_le.iter().map(|v| v.to_bits()).collect();
  let bits_be: std::vec::Vec<u32> = out_be.iter().map(|v| v.to_bits()).collect();
  let bits_expected: std::vec::Vec<u32> = expected.iter().map(|v| v.to_bits()).collect();
  assert_eq!(bits_le, bits_expected, "LE path must match reference");
  assert_eq!(bits_be, bits_expected, "BE path must match reference");
  assert_eq!(bits_le, bits_be, "BE and LE grayf16 luma_f32 must match");
}
