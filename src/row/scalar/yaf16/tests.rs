//! Tests for `crate::row::scalar::yaf16`.

use super::*;

/// Build a packed `[Y, A]` f16 plane from host f32 values, stored LE. All test
/// values (`0.0, 0.25, 0.5, 0.75, 0.9, 1.0, 1.5, 2.0, 2.5, -0.1, ...`) round to
/// the same f16, and the widen back to f32 is lossless, so the integer outputs
/// match the `yaf32` scalar exactly.
fn as_le(host: &[f32]) -> std::vec::Vec<half::f16> {
  host
    .iter()
    .map(|&v| {
      let h = half::f16::from_f32(v);
      half::f16::from_bits(u16::from_ne_bytes(h.to_bits().to_le_bytes()))
    })
    .collect()
}

/// BE-encoded variant of [`as_le`], for the `BE = true` kernels.
fn as_be(host: &[f32]) -> std::vec::Vec<half::f16> {
  host
    .iter()
    .map(|&v| {
      let h = half::f16::from_f32(v);
      half::f16::from_bits(u16::from_ne_bytes(h.to_bits().to_be_bytes()))
    })
    .collect()
}

// ---- yaf16_to_rgb_row (Y broadcast, alpha dropped) -----------------------

#[cfg_attr(
  miri,
  ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
)]
#[test]
fn rgb_broadcasts_y_drops_alpha() {
  let packed = as_le(&[0.5, 0.0, 1.0, 0.25]);
  let mut out = [0u8; 6];
  yaf16_to_rgb_row::<false>(&packed, &mut out, 2);
  assert_eq!(out, [128, 128, 128, 255, 255, 255]);
}

#[cfg_attr(
  miri,
  ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
)]
#[test]
fn rgb_be_matches_le() {
  let host = [0.25f32, 0.9, 0.5, 0.1];
  let mut le_out = [0u8; 6];
  let mut be_out = [0u8; 6];
  yaf16_to_rgb_row::<false>(&as_le(&host), &mut le_out, 2);
  yaf16_to_rgb_row::<true>(&as_be(&host), &mut be_out, 2);
  assert_eq!(le_out, be_out);
}

// ---- yaf16_to_rgba_row (Y broadcast + real alpha) ------------------------

#[cfg_attr(
  miri,
  ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
)]
#[test]
fn rgba_carries_source_alpha() {
  let packed = as_le(&[0.5, 0.5, 1.0, 0.0]);
  let mut out = [9u8; 8];
  yaf16_to_rgba_row::<false>(&packed, &mut out, 2);
  assert_eq!(out, [128, 128, 128, 128, 255, 255, 255, 0]);
}

#[cfg_attr(
  miri,
  ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
)]
#[test]
fn rgba_alpha_saturates() {
  let packed = as_le(&[1.0, 2.0, 0.0, -0.5]);
  let mut out = [9u8; 8];
  yaf16_to_rgba_row::<false>(&packed, &mut out, 2);
  assert_eq!(out, [255, 255, 255, 255, 0, 0, 0, 0]);
}

#[cfg_attr(
  miri,
  ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
)]
#[test]
fn rgba_be_matches_le() {
  let host = [0.5f32, 0.25, 0.75, 1.0];
  let mut le_out = [0u8; 8];
  let mut be_out = [0u8; 8];
  yaf16_to_rgba_row::<false>(&as_le(&host), &mut le_out, 2);
  yaf16_to_rgba_row::<true>(&as_be(&host), &mut be_out, 2);
  assert_eq!(le_out, be_out);
}

// ---- u16 outputs ----------------------------------------------------------

#[cfg_attr(
  miri,
  ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
)]
#[test]
fn rgba_u16_carries_alpha() {
  let packed = as_le(&[1.0, 0.5]);
  let mut out = [0u16; 4];
  yaf16_to_rgba_u16_row::<false>(&packed, &mut out, 1);
  assert_eq!(out, [65535, 65535, 65535, 32768]);
}

#[cfg_attr(
  miri,
  ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
)]
#[test]
fn rgb_u16_broadcasts() {
  let packed = as_le(&[1.0, 0.0]);
  let mut out = [0u16; 3];
  yaf16_to_rgb_u16_row::<false>(&packed, &mut out, 1);
  assert_eq!(out, [65535, 65535, 65535]);
}

// ---- lossless f32 paths (widen, alpha dropped) ---------------------------

#[cfg_attr(
  miri,
  ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
)]
#[test]
fn rgb_f32_lossless_widen() {
  // 2.5 / -1.0 are exact in f16; widen is lossless.
  let packed = as_le(&[2.5, 0.3, -1.0, 0.7]);
  let mut out = [0.0f32; 6];
  yaf16_to_rgb_f32_row::<false>(&packed, &mut out, 2);
  assert_eq!(out, [2.5, 2.5, 2.5, -1.0, -1.0, -1.0]);
}

#[cfg_attr(
  miri,
  ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
)]
#[test]
fn luma_f32_lossless_widen() {
  let packed = as_le(&[3.0, 0.1, -2.0, 0.9]);
  let mut out = [0.0f32; 2];
  yaf16_to_luma_f32_row::<false>(&packed, &mut out, 2);
  assert_eq!(out, [3.0, -2.0]);
}

// ---- luma + hsv ----------------------------------------------------------

#[cfg_attr(
  miri,
  ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
)]
#[test]
fn luma_clamps_and_rounds() {
  let packed = as_le(&[0.5, 0.9, 1.5, 0.0, -0.2, 0.0]);
  let mut out = [9u8; 3];
  yaf16_to_luma_row::<false>(&packed, &mut out, 3);
  assert_eq!(out, [128, 255, 0]);
}

#[cfg_attr(
  miri,
  ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
)]
#[test]
fn luma_u16_clamps() {
  let packed = as_le(&[0.5, 0.0]);
  let mut out = [0u16; 1];
  yaf16_to_luma_u16_row::<false>(&packed, &mut out, 1);
  assert_eq!(out, [32768]);
}

#[cfg_attr(
  miri,
  ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
)]
#[test]
fn hsv_gray_fast_path() {
  let packed = as_le(&[0.5, 0.9, 1.0, 0.0]);
  let mut h = [9u8; 2];
  let mut s = [9u8; 2];
  let mut v = [9u8; 2];
  yaf16_to_hsv_row::<false>(&packed, &mut h, &mut s, &mut v, 2);
  assert_eq!(h, [0, 0]);
  assert_eq!(s, [0, 0]);
  assert_eq!(v, [128, 255]);
}
