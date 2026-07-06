//! Tests for `crate::row::scalar::planar_gbr_float`.

use super::*;
use crate::ColorMatrix;

// ---- helpers: host-independent f32 LE / BE byte-storage encoders -----------

/// Re-encode a host-native f32 slice as LE-encoded f32 storage. Kernels
/// called with `BE = false` recover the intended host-native value via
/// `u32::from_le` on both LE (no-op) and BE (byte-swap) hosts.
fn as_le_f32(host: &[f32]) -> std::vec::Vec<f32> {
  host
    .iter()
    .map(|v| f32::from_bits(u32::from_ne_bytes(v.to_bits().to_le_bytes())))
    .collect()
}

/// Mirror of `as_le_f32` for kernels invoked with `BE = true`. Combined
/// with `as_le_f32`, lets a single host-native `intended` fixture drive
/// both `<false>` and `<true>` kernel paths so they decode the same logical
/// bit pattern on every host.
fn as_be_f32(host: &[f32]) -> std::vec::Vec<f32> {
  host
    .iter()
    .map(|v| f32::from_bits(u32::from_ne_bytes(v.to_bits().to_be_bytes())))
    .collect()
}

// -- Scalar references for the BE-parity tests --
//
// Walk host-native `intended` G/B/R(/A) planes and reproduce each kernel's
// documented behaviour without going through any byte-order conversion.
// Pinning the LE / BE outputs against these absolute references prevents
// the parity assertion from passing in lock-step on two equally corrupt
// decode paths. The references mirror the in-source kernel logic bit-for-
// bit (clamp, round-half-up, channel reorder, fused narrow) so they can
// pin the kernel outputs absolutely.

fn ref_gbrpf32_to_rgb_u8(g: &[f32], b: &[f32], r: &[f32], width: usize) -> std::vec::Vec<u8> {
  let mut out = std::vec![0u8; width * 3];
  for x in 0..width {
    let dst = x * 3;
    out[dst] = f32_to_u8(r[x]);
    out[dst + 1] = f32_to_u8(g[x]);
    out[dst + 2] = f32_to_u8(b[x]);
  }
  out
}

fn ref_gbrpf32_to_rgba_u8(g: &[f32], b: &[f32], r: &[f32], width: usize) -> std::vec::Vec<u8> {
  let mut out = std::vec![0u8; width * 4];
  for x in 0..width {
    let dst = x * 4;
    out[dst] = f32_to_u8(r[x]);
    out[dst + 1] = f32_to_u8(g[x]);
    out[dst + 2] = f32_to_u8(b[x]);
    out[dst + 3] = 0xFF;
  }
  out
}

fn ref_gbrpf32_to_rgb_u16(g: &[f32], b: &[f32], r: &[f32], width: usize) -> std::vec::Vec<u16> {
  let mut out = std::vec![0u16; width * 3];
  for x in 0..width {
    let dst = x * 3;
    out[dst] = f32_to_u16(r[x]);
    out[dst + 1] = f32_to_u16(g[x]);
    out[dst + 2] = f32_to_u16(b[x]);
  }
  out
}

fn ref_gbrpf32_to_rgba_u16(g: &[f32], b: &[f32], r: &[f32], width: usize) -> std::vec::Vec<u16> {
  let mut out = std::vec![0u16; width * 4];
  for x in 0..width {
    let dst = x * 4;
    out[dst] = f32_to_u16(r[x]);
    out[dst + 1] = f32_to_u16(g[x]);
    out[dst + 2] = f32_to_u16(b[x]);
    out[dst + 3] = 0xFFFF;
  }
  out
}

fn ref_gbrpf32_to_rgb_f32(g: &[f32], b: &[f32], r: &[f32], width: usize) -> std::vec::Vec<f32> {
  let mut out = std::vec![0.0f32; width * 3];
  for x in 0..width {
    let dst = x * 3;
    out[dst] = r[x];
    out[dst + 1] = g[x];
    out[dst + 2] = b[x];
  }
  out
}

fn ref_gbrpf32_to_rgba_f32(g: &[f32], b: &[f32], r: &[f32], width: usize) -> std::vec::Vec<f32> {
  let mut out = std::vec![0.0f32; width * 4];
  for x in 0..width {
    let dst = x * 4;
    out[dst] = r[x];
    out[dst + 1] = g[x];
    out[dst + 2] = b[x];
    out[dst + 3] = 1.0;
  }
  out
}

fn ref_gbrpf32_to_rgb_f16(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  width: usize,
) -> std::vec::Vec<half::f16> {
  let mut out = std::vec![half::f16::ZERO; width * 3];
  for x in 0..width {
    let dst = x * 3;
    out[dst] = f32_to_f16(r[x]);
    out[dst + 1] = f32_to_f16(g[x]);
    out[dst + 2] = f32_to_f16(b[x]);
  }
  out
}

fn ref_gbrpf32_to_rgba_f16(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  width: usize,
) -> std::vec::Vec<half::f16> {
  let mut out = std::vec![half::f16::ZERO; width * 4];
  let one = half::f16::from_f32(1.0);
  for x in 0..width {
    let dst = x * 4;
    out[dst] = f32_to_f16(r[x]);
    out[dst + 1] = f32_to_f16(g[x]);
    out[dst + 2] = f32_to_f16(b[x]);
    out[dst + 3] = one;
  }
  out
}

/// Reference for `gbrpf32_to_luma_row`: stage through u8 RGB scratch in
/// chunks of 64, then `super::rgb_to_luma_row`. Mirrors the kernel exactly
/// so the staged-rounding behaviour is reproduced.
fn ref_gbrpf32_to_luma(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) -> std::vec::Vec<u8> {
  let mut luma = std::vec![0u8; width];
  const CHUNK: usize = 64;
  let mut scratch = [0u8; CHUNK * 3];
  let mut offset = 0;
  while offset < width {
    let n = (width - offset).min(CHUNK);
    let rgb = ref_gbrpf32_to_rgb_u8(&g[offset..], &b[offset..], &r[offset..], n);
    scratch[..n * 3].copy_from_slice(&rgb);
    super::super::rgb_to_luma_row(
      &scratch[..n * 3],
      &mut luma[offset..offset + n],
      n,
      matrix,
      full_range,
    );
    offset += n;
  }
  luma
}

fn ref_gbrpf32_to_luma_u16(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) -> std::vec::Vec<u16> {
  let mut luma = std::vec![0u16; width];
  const CHUNK: usize = 64;
  let mut scratch = [0u8; CHUNK * 3];
  let mut offset = 0;
  while offset < width {
    let n = (width - offset).min(CHUNK);
    let rgb = ref_gbrpf32_to_rgb_u8(&g[offset..], &b[offset..], &r[offset..], n);
    scratch[..n * 3].copy_from_slice(&rgb);
    super::super::rgb_to_luma_u16_row(
      &scratch[..n * 3],
      &mut luma[offset..offset + n],
      n,
      matrix,
      full_range,
    );
    offset += n;
  }
  luma
}

fn ref_gbrpf32_to_hsv(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  width: usize,
) -> (std::vec::Vec<u8>, std::vec::Vec<u8>, std::vec::Vec<u8>) {
  let mut h = std::vec![0u8; width];
  let mut s = std::vec![0u8; width];
  let mut v = std::vec![0u8; width];
  const CHUNK: usize = 64;
  let mut scratch = [0u8; CHUNK * 3];
  let mut offset = 0;
  while offset < width {
    let n = (width - offset).min(CHUNK);
    let rgb = ref_gbrpf32_to_rgb_u8(&g[offset..], &b[offset..], &r[offset..], n);
    scratch[..n * 3].copy_from_slice(&rgb);
    super::super::rgb_to_hsv_row(
      &scratch[..n * 3],
      &mut h[offset..offset + n],
      &mut s[offset..offset + n],
      &mut v[offset..offset + n],
      n,
    );
    offset += n;
  }
  (h, s, v)
}

fn ref_gbrapf32_to_rgba_u8(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  a: &[f32],
  width: usize,
) -> std::vec::Vec<u8> {
  let mut out = std::vec![0u8; width * 4];
  for x in 0..width {
    let dst = x * 4;
    out[dst] = f32_to_u8(r[x]);
    out[dst + 1] = f32_to_u8(g[x]);
    out[dst + 2] = f32_to_u8(b[x]);
    out[dst + 3] = f32_to_u8(a[x]);
  }
  out
}

fn ref_gbrapf32_to_rgba_u16(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  a: &[f32],
  width: usize,
) -> std::vec::Vec<u16> {
  let mut out = std::vec![0u16; width * 4];
  for x in 0..width {
    let dst = x * 4;
    out[dst] = f32_to_u16(r[x]);
    out[dst + 1] = f32_to_u16(g[x]);
    out[dst + 2] = f32_to_u16(b[x]);
    out[dst + 3] = f32_to_u16(a[x]);
  }
  out
}

fn ref_gbrapf32_to_rgba_f32(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  a: &[f32],
  width: usize,
) -> std::vec::Vec<f32> {
  let mut out = std::vec![0.0f32; width * 4];
  for x in 0..width {
    let dst = x * 4;
    out[dst] = r[x];
    out[dst + 1] = g[x];
    out[dst + 2] = b[x];
    out[dst + 3] = a[x];
  }
  out
}

fn ref_gbrapf32_to_rgba_f16(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  a: &[f32],
  width: usize,
) -> std::vec::Vec<half::f16> {
  let mut out = std::vec![half::f16::ZERO; width * 4];
  for x in 0..width {
    let dst = x * 4;
    out[dst] = f32_to_f16(r[x]);
    out[dst + 1] = f32_to_f16(g[x]);
    out[dst + 2] = f32_to_f16(b[x]);
    out[dst + 3] = f32_to_f16(a[x]);
  }
  out
}

// ---- gbrpf32_to_rgb_row --------------------------------------------------

#[test]
fn gbrpf32_to_rgb_clamps_and_scales() {
  // Values: 0.0, 0.5, 1.0, 1.5, -0.1, NaN → 0, 128, 255, 255, 0, 0
  // NaN passes through f32::clamp unchanged (Rust 1.50+); `NaN as u8` saturates to 0.
  // All three channels use the same value for simplicity.
  let vals = [0.0f32, 0.5, 1.0, 1.5, -0.1, f32::NAN];
  let expected = [0u8, 128, 255, 255, 0, 0];
  for (v, e) in vals.iter().zip(expected.iter()) {
    let g = as_le_f32(&[*v]);
    let b = as_le_f32(&[*v]);
    let r = as_le_f32(&[*v]);
    let mut out = [0u8; 3];
    gbrpf32_to_rgb_row::<false>(&g, &b, &r, &mut out, 1);
    assert_eq!(out[0], *e, "R: v={v}, expected={e}");
    assert_eq!(out[1], *e, "G: v={v}, expected={e}");
    assert_eq!(out[2], *e, "B: v={v}, expected={e}");
  }
}

#[test]
fn gbrpf32_to_rgb_channel_reorder() {
  // G=0.0, B=0.5, R=1.0 → packed R=255, G=0, B=128
  let g = as_le_f32(&[0.0f32]);
  let b = as_le_f32(&[0.5f32]);
  let r = as_le_f32(&[1.0f32]);
  let mut out = [0u8; 3];
  gbrpf32_to_rgb_row::<false>(&g, &b, &r, &mut out, 1);
  assert_eq!(out[0], 255, "R");
  assert_eq!(out[1], 0, "G");
  assert_eq!(out[2], 128, "B");
}

#[test]
fn gbrpf32_to_rgb_be_parity() {
  // Build host-native intended planes; materialise as LE / BE byte storage
  // so each kernel's `from_le` / `from_be` recovers the same logical bits
  // on every host. Pin both outputs against an absolute scalar reference.
  let g_intended = [0.0f32, 0.25, 0.5, 1.0];
  let b_intended = [0.1f32, 0.3, 0.7, 0.9];
  let r_intended = [0.5f32, 0.8, 0.2, 0.6];
  let g_le = as_le_f32(&g_intended);
  let b_le = as_le_f32(&b_intended);
  let r_le = as_le_f32(&r_intended);
  let g_be = as_be_f32(&g_intended);
  let b_be = as_be_f32(&b_intended);
  let r_be = as_be_f32(&r_intended);
  let mut le_out = std::vec![0u8; 4 * 3];
  let mut be_out = std::vec![0u8; 4 * 3];
  gbrpf32_to_rgb_row::<false>(&g_le, &b_le, &r_le, &mut le_out, 4);
  gbrpf32_to_rgb_row::<true>(&g_be, &b_be, &r_be, &mut be_out, 4);
  let expected = ref_gbrpf32_to_rgb_u8(&g_intended, &b_intended, &r_intended, 4);
  assert_eq!(le_out, expected, "LE path must match scalar reference");
  assert_eq!(be_out, expected, "BE path must match scalar reference");
  assert_eq!(be_out, le_out, "BE gbrpf32_to_rgb_row must match LE");
}

// ---- gbrpf32_to_rgba_row -------------------------------------------------

#[test]
fn gbrpf32_to_rgba_fills_alpha_max() {
  let g = [0.5f32];
  let b = [0.5f32];
  let r = [0.5f32];
  let mut out = [0u8; 4];
  gbrpf32_to_rgba_row::<false>(&g, &b, &r, &mut out, 1);
  assert_eq!(out[3], 0xFF, "alpha must be 0xFF");
}

#[test]
fn gbrpf32_to_rgba_clamps_and_scales() {
  let vals = [0.0f32, 0.5, 1.0, 1.5, -0.1];
  let expected = [0u8, 128, 255, 255, 0];
  for (v, e) in vals.iter().zip(expected.iter()) {
    let g = as_le_f32(&[*v]);
    let b = as_le_f32(&[*v]);
    let r = as_le_f32(&[*v]);
    let mut out = [0u8; 4];
    gbrpf32_to_rgba_row::<false>(&g, &b, &r, &mut out, 1);
    assert_eq!(out[0], *e, "R: v={v}");
    assert_eq!(out[3], 0xFF, "alpha must remain 0xFF");
  }
}

#[test]
fn gbrpf32_to_rgba_be_parity() {
  let g_intended = [0.0f32, 0.25, 0.5, 1.0];
  let b_intended = [0.1f32, 0.3, 0.7, 0.9];
  let r_intended = [0.5f32, 0.8, 0.2, 0.6];
  let g_le = as_le_f32(&g_intended);
  let b_le = as_le_f32(&b_intended);
  let r_le = as_le_f32(&r_intended);
  let g_be = as_be_f32(&g_intended);
  let b_be = as_be_f32(&b_intended);
  let r_be = as_be_f32(&r_intended);
  let mut le_out = std::vec![0u8; 4 * 4];
  let mut be_out = std::vec![0u8; 4 * 4];
  gbrpf32_to_rgba_row::<false>(&g_le, &b_le, &r_le, &mut le_out, 4);
  gbrpf32_to_rgba_row::<true>(&g_be, &b_be, &r_be, &mut be_out, 4);
  let expected = ref_gbrpf32_to_rgba_u8(&g_intended, &b_intended, &r_intended, 4);
  assert_eq!(le_out, expected, "LE path must match scalar reference");
  assert_eq!(be_out, expected, "BE path must match scalar reference");
  assert_eq!(be_out, le_out, "BE gbrpf32_to_rgba_row must match LE");
}

// ---- gbrpf32_to_rgb_u16_row ----------------------------------------------

#[test]
fn gbrpf32_to_rgb_u16_clamps_and_scales() {
  // NaN passes through f32::clamp unchanged (Rust 1.50+); `NaN as u16` saturates to 0.
  let vals = [0.0f32, 0.5, 1.0, 1.5, -0.1, f32::NAN];
  // 0.5 → (0.5 * 65535 + 0.5) as u16 = 32768; NaN → 0
  let expected = [0u16, 32768, 65535, 65535, 0, 0];
  for (v, e) in vals.iter().zip(expected.iter()) {
    let g = as_le_f32(&[*v]);
    let b = as_le_f32(&[*v]);
    let r = as_le_f32(&[*v]);
    let mut out = [0u16; 3];
    gbrpf32_to_rgb_u16_row::<false>(&g, &b, &r, &mut out, 1);
    assert_eq!(out[0], *e, "R u16: v={v}");
    assert_eq!(out[1], *e, "G u16: v={v}");
    assert_eq!(out[2], *e, "B u16: v={v}");
  }
}

#[test]
fn gbrpf32_to_rgb_u16_be_parity() {
  let g_intended = [0.0f32, 0.25, 0.5, 1.0];
  let b_intended = [0.1f32, 0.3, 0.7, 0.9];
  let r_intended = [0.5f32, 0.8, 0.2, 0.6];
  let g_le = as_le_f32(&g_intended);
  let b_le = as_le_f32(&b_intended);
  let r_le = as_le_f32(&r_intended);
  let g_be = as_be_f32(&g_intended);
  let b_be = as_be_f32(&b_intended);
  let r_be = as_be_f32(&r_intended);
  let mut le_out = std::vec![0u16; 4 * 3];
  let mut be_out = std::vec![0u16; 4 * 3];
  gbrpf32_to_rgb_u16_row::<false>(&g_le, &b_le, &r_le, &mut le_out, 4);
  gbrpf32_to_rgb_u16_row::<true>(&g_be, &b_be, &r_be, &mut be_out, 4);
  let expected = ref_gbrpf32_to_rgb_u16(&g_intended, &b_intended, &r_intended, 4);
  assert_eq!(le_out, expected, "LE path must match scalar reference");
  assert_eq!(be_out, expected, "BE path must match scalar reference");
  assert_eq!(be_out, le_out, "BE gbrpf32_to_rgb_u16_row must match LE");
}

// ---- gbrpf32_to_rgba_u16_row ---------------------------------------------

#[test]
fn gbrpf32_to_rgba_u16_fills_alpha_max() {
  let g = [0.5f32];
  let b = [0.5f32];
  let r = [0.5f32];
  let mut out = [0u16; 4];
  gbrpf32_to_rgba_u16_row::<false>(&g, &b, &r, &mut out, 1);
  assert_eq!(out[3], 0xFFFF, "alpha must be 0xFFFF");
}

#[test]
fn gbrpf32_to_rgba_u16_be_parity() {
  let g_intended = [0.0f32, 0.25, 0.5, 1.0];
  let b_intended = [0.1f32, 0.3, 0.7, 0.9];
  let r_intended = [0.5f32, 0.8, 0.2, 0.6];
  let g_le = as_le_f32(&g_intended);
  let b_le = as_le_f32(&b_intended);
  let r_le = as_le_f32(&r_intended);
  let g_be = as_be_f32(&g_intended);
  let b_be = as_be_f32(&b_intended);
  let r_be = as_be_f32(&r_intended);
  let mut le_out = std::vec![0u16; 4 * 4];
  let mut be_out = std::vec![0u16; 4 * 4];
  gbrpf32_to_rgba_u16_row::<false>(&g_le, &b_le, &r_le, &mut le_out, 4);
  gbrpf32_to_rgba_u16_row::<true>(&g_be, &b_be, &r_be, &mut be_out, 4);
  let expected = ref_gbrpf32_to_rgba_u16(&g_intended, &b_intended, &r_intended, 4);
  assert_eq!(le_out, expected, "LE path must match scalar reference");
  assert_eq!(be_out, expected, "BE path must match scalar reference");
  assert_eq!(be_out, le_out, "BE gbrpf32_to_rgba_u16_row must match LE");
}

// ---- gbrpf32_to_rgb_f32_row (lossless) ------------------------------------

#[test]
fn gbrpf32_to_rgb_f32_lossless_passthrough() {
  // HDR 2.5, NaN, Inf, negative all preserved bit-exact. Output is host-
  // native f32 after the kernel's `from_le` decode; compare against the
  // original host-native values rather than the LE-encoded inputs.
  let host_g = [2.5f32, f32::NAN, f32::INFINITY, -1.0];
  let host_b = [0.1f32, 0.2, 0.3, 0.4];
  let host_r = [0.5f32, 0.6, 0.7, 0.8];
  let g = as_le_f32(&host_g);
  let b = as_le_f32(&host_b);
  let r = as_le_f32(&host_r);
  let mut out = [0.0f32; 12];
  gbrpf32_to_rgb_f32_row::<false>(&g, &b, &r, &mut out, 4);
  // Check R channel (index 0, 3, 6, 9 in RGBA interleave = index 0, 3, 6, 9)
  assert_eq!(out[0], host_r[0]);
  assert_eq!(out[3], host_r[1]);
  assert_eq!(out[6], host_r[2]);
  assert_eq!(out[9], host_r[3]);
  // Check G channel (index 1, 4, 7, 10)
  assert_eq!(out[1], host_g[0], "G HDR preserved");
  assert!(out[4].is_nan(), "G NaN preserved");
  assert!(out[7].is_infinite() && out[7] > 0.0, "G +Inf preserved");
  assert_eq!(out[10], host_g[3], "G negative preserved");
}

#[test]
fn gbrpf32_to_rgb_f32_be_parity() {
  let g_intended = [0.0f32, 0.25, 0.5, 1.0];
  let b_intended = [0.1f32, 0.3, 0.7, 0.9];
  let r_intended = [0.5f32, 0.8, 0.2, 0.6];
  let g_le = as_le_f32(&g_intended);
  let b_le = as_le_f32(&b_intended);
  let r_le = as_le_f32(&r_intended);
  let g_be = as_be_f32(&g_intended);
  let b_be = as_be_f32(&b_intended);
  let r_be = as_be_f32(&r_intended);
  let mut le_out = std::vec![0.0f32; 4 * 3];
  let mut be_out = std::vec![0.0f32; 4 * 3];
  gbrpf32_to_rgb_f32_row::<false>(&g_le, &b_le, &r_le, &mut le_out, 4);
  gbrpf32_to_rgb_f32_row::<true>(&g_be, &b_be, &r_be, &mut be_out, 4);
  let expected = ref_gbrpf32_to_rgb_f32(&g_intended, &b_intended, &r_intended, 4);
  assert_eq!(le_out, expected, "LE path must match scalar reference");
  assert_eq!(be_out, expected, "BE path must match scalar reference");
  assert_eq!(be_out, le_out, "BE gbrpf32_to_rgb_f32_row must match LE");
}

// ---- gbrpf32_to_rgba_f32_row (lossless, α = 1.0) -------------------------

#[test]
fn gbrpf32_to_rgba_f32_alpha_is_one() {
  let g = [0.5f32];
  let b = [0.5f32];
  let r = [0.5f32];
  let mut out = [0.0f32; 4];
  gbrpf32_to_rgba_f32_row::<false>(&g, &b, &r, &mut out, 1);
  assert_eq!(out[3], 1.0, "alpha must be 1.0");
}

#[test]
fn gbrpf32_to_rgba_f32_lossless_passthrough() {
  let r = as_le_f32(&[2.5f32]);
  let g = as_le_f32(&[f32::NAN]);
  let b = as_le_f32(&[f32::NEG_INFINITY]);
  let mut out = [0.0f32; 4];
  gbrpf32_to_rgba_f32_row::<false>(&g, &b, &r, &mut out, 1);
  assert_eq!(out[0], 2.5, "R HDR preserved");
  assert!(out[1].is_nan(), "G NaN preserved");
  assert!(out[2].is_infinite() && out[2] < 0.0, "B -Inf preserved");
  assert_eq!(out[3], 1.0, "alpha = 1.0");
}

#[test]
fn gbrpf32_to_rgba_f32_be_parity() {
  let g_intended = [0.0f32, 0.25, 0.5, 1.0];
  let b_intended = [0.1f32, 0.3, 0.7, 0.9];
  let r_intended = [0.5f32, 0.8, 0.2, 0.6];
  let g_le = as_le_f32(&g_intended);
  let b_le = as_le_f32(&b_intended);
  let r_le = as_le_f32(&r_intended);
  let g_be = as_be_f32(&g_intended);
  let b_be = as_be_f32(&b_intended);
  let r_be = as_be_f32(&r_intended);
  let mut le_out = std::vec![0.0f32; 4 * 4];
  let mut be_out = std::vec![0.0f32; 4 * 4];
  gbrpf32_to_rgba_f32_row::<false>(&g_le, &b_le, &r_le, &mut le_out, 4);
  gbrpf32_to_rgba_f32_row::<true>(&g_be, &b_be, &r_be, &mut be_out, 4);
  let expected = ref_gbrpf32_to_rgba_f32(&g_intended, &b_intended, &r_intended, 4);
  assert_eq!(le_out, expected, "LE path must match scalar reference");
  assert_eq!(be_out, expected, "BE path must match scalar reference");
  assert_eq!(be_out, le_out, "BE gbrpf32_to_rgba_f32_row must match LE");
}

// ---- gbrpf32_to_rgb_f16_row ----------------------------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
)]
fn gbrpf32_to_rgb_f16_normal_values() {
  let g = [0.0f32, 0.5, 1.0];
  let b = [0.25f32, 0.75, 0.0];
  let r = [1.0f32, 0.0, 0.5];
  let mut out = vec![half::f16::ZERO; 9];
  gbrpf32_to_rgb_f16_row::<false>(&g, &b, &r, &mut out, 3);
  assert_eq!(out[0], half::f16::from_f32(1.0), "R[0]");
  assert_eq!(out[1], half::f16::from_f32(0.0), "G[0]");
  assert_eq!(out[2], half::f16::from_f32(0.25), "B[0]");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
)]
fn gbrpf32_to_rgb_f16_hdr_saturates_to_inf() {
  // Input 70000.0 > f16 max (~65504) → +Inf
  let g = [70_000.0f32];
  let b = [-70_000.0f32];
  let r = [0.5f32];
  let mut out = vec![half::f16::ZERO; 3];
  gbrpf32_to_rgb_f16_row::<false>(&g, &b, &r, &mut out, 1);
  // G maps to index 1
  assert!(out[1].is_infinite() && out[1].to_f32() > 0.0, "G +Inf");
  // B maps to index 2
  assert!(out[2].is_infinite() && out[2].to_f32() < 0.0, "B -Inf");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
)]
fn gbrpf32_to_rgb_f16_be_parity() {
  let g_intended = [0.0f32, 0.25, 0.5, 1.0];
  let b_intended = [0.1f32, 0.3, 0.7, 0.9];
  let r_intended = [0.5f32, 0.8, 0.2, 0.6];
  let g_le = as_le_f32(&g_intended);
  let b_le = as_le_f32(&b_intended);
  let r_le = as_le_f32(&r_intended);
  let g_be = as_be_f32(&g_intended);
  let b_be = as_be_f32(&b_intended);
  let r_be = as_be_f32(&r_intended);
  let mut le_out = vec![half::f16::ZERO; 4 * 3];
  let mut be_out = vec![half::f16::ZERO; 4 * 3];
  gbrpf32_to_rgb_f16_row::<false>(&g_le, &b_le, &r_le, &mut le_out, 4);
  gbrpf32_to_rgb_f16_row::<true>(&g_be, &b_be, &r_be, &mut be_out, 4);
  let expected = ref_gbrpf32_to_rgb_f16(&g_intended, &b_intended, &r_intended, 4);
  assert_eq!(le_out, expected, "LE path must match scalar reference");
  assert_eq!(be_out, expected, "BE path must match scalar reference");
  assert_eq!(be_out, le_out, "BE gbrpf32_to_rgb_f16_row must match LE");
}

// ---- gbrpf32_to_rgba_f16_row ---------------------------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
)]
fn gbrpf32_to_rgba_f16_alpha_is_one() {
  let g = [0.5f32];
  let b = [0.5f32];
  let r = [0.5f32];
  let mut out = vec![half::f16::ZERO; 4];
  gbrpf32_to_rgba_f16_row::<false>(&g, &b, &r, &mut out, 1);
  assert_eq!(out[3], half::f16::from_f32(1.0), "alpha must be f16(1.0)");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
)]
fn gbrpf32_to_rgba_f16_be_parity() {
  let g_intended = [0.0f32, 0.25, 0.5, 1.0];
  let b_intended = [0.1f32, 0.3, 0.7, 0.9];
  let r_intended = [0.5f32, 0.8, 0.2, 0.6];
  let g_le = as_le_f32(&g_intended);
  let b_le = as_le_f32(&b_intended);
  let r_le = as_le_f32(&r_intended);
  let g_be = as_be_f32(&g_intended);
  let b_be = as_be_f32(&b_intended);
  let r_be = as_be_f32(&r_intended);
  let mut le_out = vec![half::f16::ZERO; 4 * 4];
  let mut be_out = vec![half::f16::ZERO; 4 * 4];
  gbrpf32_to_rgba_f16_row::<false>(&g_le, &b_le, &r_le, &mut le_out, 4);
  gbrpf32_to_rgba_f16_row::<true>(&g_be, &b_be, &r_be, &mut be_out, 4);
  let expected = ref_gbrpf32_to_rgba_f16(&g_intended, &b_intended, &r_intended, 4);
  assert_eq!(le_out, expected, "LE path must match scalar reference");
  assert_eq!(be_out, expected, "BE path must match scalar reference");
  assert_eq!(be_out, le_out, "BE gbrpf32_to_rgba_f16_row must match LE");
}

// ---- gbrpf32_to_luma_row -------------------------------------------------

#[test]
fn gbrpf32_to_luma_zero_gives_zero() {
  let g = [0.0f32];
  let b = [0.0f32];
  let r = [0.0f32];
  let mut out = [0xFFu8; 1];
  gbrpf32_to_luma_row::<false>(&g, &b, &r, &mut out, 1, ColorMatrix::Bt709, true);
  assert_eq!(out[0], 0);
}

#[test]
fn gbrpf32_to_luma_max_gives_255() {
  let g = as_le_f32(&[1.0f32]);
  let b = as_le_f32(&[1.0f32]);
  let r = as_le_f32(&[1.0f32]);
  let mut out = [0u8; 1];
  gbrpf32_to_luma_row::<false>(&g, &b, &r, &mut out, 1, ColorMatrix::Bt709, true);
  assert_eq!(out[0], 255);
}

#[test]
fn gbrpf32_to_luma_be_parity() {
  let g_intended = [0.0f32, 0.25, 0.5, 1.0];
  let b_intended = [0.1f32, 0.3, 0.7, 0.9];
  let r_intended = [0.5f32, 0.8, 0.2, 0.6];
  let g_le = as_le_f32(&g_intended);
  let b_le = as_le_f32(&b_intended);
  let r_le = as_le_f32(&r_intended);
  let g_be = as_be_f32(&g_intended);
  let b_be = as_be_f32(&b_intended);
  let r_be = as_be_f32(&r_intended);
  let mut le_out = std::vec![0u8; 4];
  let mut be_out = std::vec![0u8; 4];
  gbrpf32_to_luma_row::<false>(
    &g_le,
    &b_le,
    &r_le,
    &mut le_out,
    4,
    ColorMatrix::Bt709,
    true,
  );
  gbrpf32_to_luma_row::<true>(
    &g_be,
    &b_be,
    &r_be,
    &mut be_out,
    4,
    ColorMatrix::Bt709,
    true,
  );
  let expected = ref_gbrpf32_to_luma(
    &g_intended,
    &b_intended,
    &r_intended,
    4,
    ColorMatrix::Bt709,
    true,
  );
  assert_eq!(le_out, expected, "LE path must match scalar reference");
  assert_eq!(be_out, expected, "BE path must match scalar reference");
  assert_eq!(be_out, le_out, "BE gbrpf32_to_luma_row must match LE");
}

// ---- gbrpf32_to_luma_u16_row ---------------------------------------------

#[test]
fn gbrpf32_to_luma_u16_zero_gives_zero() {
  let g = [0.0f32];
  let b = [0.0f32];
  let r = [0.0f32];
  let mut out = [0xFFFFu16; 1];
  gbrpf32_to_luma_u16_row::<false>(&g, &b, &r, &mut out, 1, ColorMatrix::Bt709, true);
  assert_eq!(out[0], 0);
}

#[test]
fn gbrpf32_to_luma_u16_max_gives_255_zero_extended() {
  let g = as_le_f32(&[1.0f32]);
  let b = as_le_f32(&[1.0f32]);
  let r = as_le_f32(&[1.0f32]);
  let mut out = [0u16; 1];
  gbrpf32_to_luma_u16_row::<false>(&g, &b, &r, &mut out, 1, ColorMatrix::Bt709, true);
  assert_eq!(out[0], 255, "luma_u16 is zero-extended u8 luma");
}

#[test]
fn gbrpf32_to_luma_u16_be_parity() {
  let g_intended = [0.0f32, 0.25, 0.5, 1.0];
  let b_intended = [0.1f32, 0.3, 0.7, 0.9];
  let r_intended = [0.5f32, 0.8, 0.2, 0.6];
  let g_le = as_le_f32(&g_intended);
  let b_le = as_le_f32(&b_intended);
  let r_le = as_le_f32(&r_intended);
  let g_be = as_be_f32(&g_intended);
  let b_be = as_be_f32(&b_intended);
  let r_be = as_be_f32(&r_intended);
  let mut le_out = std::vec![0u16; 4];
  let mut be_out = std::vec![0u16; 4];
  gbrpf32_to_luma_u16_row::<false>(
    &g_le,
    &b_le,
    &r_le,
    &mut le_out,
    4,
    ColorMatrix::Bt709,
    true,
  );
  gbrpf32_to_luma_u16_row::<true>(
    &g_be,
    &b_be,
    &r_be,
    &mut be_out,
    4,
    ColorMatrix::Bt709,
    true,
  );
  let expected = ref_gbrpf32_to_luma_u16(
    &g_intended,
    &b_intended,
    &r_intended,
    4,
    ColorMatrix::Bt709,
    true,
  );
  assert_eq!(le_out, expected, "LE path must match scalar reference");
  assert_eq!(be_out, expected, "BE path must match scalar reference");
  assert_eq!(be_out, le_out, "BE gbrpf32_to_luma_u16_row must match LE");
}

// ---- gbrpf32_to_hsv_row --------------------------------------------------

#[test]
fn gbrpf32_to_hsv_achromatic_black() {
  let g = [0.0f32];
  let b = [0.0f32];
  let r = [0.0f32];
  let mut h = [0xFFu8; 1];
  let mut s = [0xFFu8; 1];
  let mut v = [0xFFu8; 1];
  gbrpf32_to_hsv_row::<false>(&g, &b, &r, &mut h, &mut s, &mut v, 1);
  assert_eq!(v[0], 0, "V must be 0 for black");
  assert_eq!(s[0], 0, "S must be 0 for achromatic");
}

#[test]
fn gbrpf32_to_hsv_achromatic_white() {
  let g = as_le_f32(&[1.0f32]);
  let b = as_le_f32(&[1.0f32]);
  let r = as_le_f32(&[1.0f32]);
  let mut h = [0u8; 1];
  let mut s = [0u8; 1];
  let mut v = [0u8; 1];
  gbrpf32_to_hsv_row::<false>(&g, &b, &r, &mut h, &mut s, &mut v, 1);
  assert_eq!(v[0], 255, "V must be 255 for white");
  assert_eq!(s[0], 0, "S must be 0 for achromatic");
}

#[test]
fn gbrpf32_to_hsv_be_parity() {
  let g_intended = [0.0f32, 0.25, 0.5, 1.0];
  let b_intended = [0.1f32, 0.3, 0.7, 0.9];
  let r_intended = [0.5f32, 0.8, 0.2, 0.6];
  let g_le = as_le_f32(&g_intended);
  let b_le = as_le_f32(&b_intended);
  let r_le = as_le_f32(&r_intended);
  let g_be = as_be_f32(&g_intended);
  let b_be = as_be_f32(&b_intended);
  let r_be = as_be_f32(&r_intended);
  let mut le_h = std::vec![0u8; 4];
  let mut le_s = std::vec![0u8; 4];
  let mut le_v = std::vec![0u8; 4];
  let mut be_h = std::vec![0u8; 4];
  let mut be_s = std::vec![0u8; 4];
  let mut be_v = std::vec![0u8; 4];
  gbrpf32_to_hsv_row::<false>(&g_le, &b_le, &r_le, &mut le_h, &mut le_s, &mut le_v, 4);
  gbrpf32_to_hsv_row::<true>(&g_be, &b_be, &r_be, &mut be_h, &mut be_s, &mut be_v, 4);
  let (h_expected, s_expected, v_expected) =
    ref_gbrpf32_to_hsv(&g_intended, &b_intended, &r_intended, 4);
  assert_eq!(le_h, h_expected, "LE H must match scalar reference");
  assert_eq!(le_s, s_expected, "LE S must match scalar reference");
  assert_eq!(le_v, v_expected, "LE V must match scalar reference");
  assert_eq!(be_h, h_expected, "BE H must match scalar reference");
  assert_eq!(be_s, s_expected, "BE S must match scalar reference");
  assert_eq!(be_v, v_expected, "BE V must match scalar reference");
  assert_eq!(be_h, le_h, "BE hsv H must match LE");
  assert_eq!(be_s, le_s, "BE hsv S must match LE");
  assert_eq!(be_v, le_v, "BE hsv V must match LE");
}

// ---- gbrapf32_to_rgba_row ------------------------------------------------

#[test]
fn gbrapf32_to_rgba_source_alpha_passthrough() {
  let g = as_le_f32(&[0.5f32]);
  let b = as_le_f32(&[0.5f32]);
  let r = as_le_f32(&[0.5f32]);
  let a = as_le_f32(&[0.5f32]);
  let mut out = [0u8; 4];
  gbrapf32_to_rgba_row::<false>(&g, &b, &r, &a, &mut out, 1);
  // 0.5 → (0.5 * 255 + 0.5) as u8 = 128
  assert_eq!(out[3], 128, "alpha from source plane");
}

#[test]
fn gbrapf32_to_rgba_source_alpha_clamps() {
  let g = as_le_f32(&[0.5f32]);
  let b = as_le_f32(&[0.5f32]);
  let r = as_le_f32(&[0.5f32]);
  // Test α > 1.0 → 255 and α < 0.0 → 0
  let a_high = as_le_f32(&[1.5f32]);
  let a_low = as_le_f32(&[-0.1f32]);
  let mut out_high = [0u8; 4];
  let mut out_low = [0u8; 4];
  gbrapf32_to_rgba_row::<false>(&g, &b, &r, &a_high, &mut out_high, 1);
  gbrapf32_to_rgba_row::<false>(&g, &b, &r, &a_low, &mut out_low, 1);
  assert_eq!(out_high[3], 255, "alpha HDR clamps to 255");
  assert_eq!(out_low[3], 0, "alpha negative clamps to 0");
}

#[test]
fn gbrapf32_to_rgba_be_parity() {
  let g_intended = [0.0f32, 0.25, 0.5, 1.0];
  let b_intended = [0.1f32, 0.3, 0.7, 0.9];
  let r_intended = [0.5f32, 0.8, 0.2, 0.6];
  let a_intended = [0.2f32, 0.4, 0.6, 0.8];
  let g_le = as_le_f32(&g_intended);
  let b_le = as_le_f32(&b_intended);
  let r_le = as_le_f32(&r_intended);
  let a_le = as_le_f32(&a_intended);
  let g_be = as_be_f32(&g_intended);
  let b_be = as_be_f32(&b_intended);
  let r_be = as_be_f32(&r_intended);
  let a_be = as_be_f32(&a_intended);
  let mut le_out = std::vec![0u8; 4 * 4];
  let mut be_out = std::vec![0u8; 4 * 4];
  gbrapf32_to_rgba_row::<false>(&g_le, &b_le, &r_le, &a_le, &mut le_out, 4);
  gbrapf32_to_rgba_row::<true>(&g_be, &b_be, &r_be, &a_be, &mut be_out, 4);
  let expected = ref_gbrapf32_to_rgba_u8(&g_intended, &b_intended, &r_intended, &a_intended, 4);
  assert_eq!(le_out, expected, "LE path must match scalar reference");
  assert_eq!(be_out, expected, "BE path must match scalar reference");
  assert_eq!(be_out, le_out, "BE gbrapf32_to_rgba_row must match LE");
}

// ---- gbrapf32_to_rgba_u16_row --------------------------------------------

#[test]
fn gbrapf32_to_rgba_u16_source_alpha_passthrough() {
  let g = as_le_f32(&[0.5f32]);
  let b = as_le_f32(&[0.5f32]);
  let r = as_le_f32(&[0.5f32]);
  let a = as_le_f32(&[0.5f32]);
  let mut out = [0u16; 4];
  gbrapf32_to_rgba_u16_row::<false>(&g, &b, &r, &a, &mut out, 1);
  // 0.5 → (0.5 * 65535 + 0.5) as u16 = 32768
  assert_eq!(out[3], 32768, "u16 alpha from source plane");
}

#[test]
fn gbrapf32_to_rgba_u16_source_alpha_clamps() {
  let g = as_le_f32(&[0.5f32]);
  let b = as_le_f32(&[0.5f32]);
  let r = as_le_f32(&[0.5f32]);
  let a_high = as_le_f32(&[1.5f32]);
  let a_low = as_le_f32(&[-0.1f32]);
  let mut out_high = [0u16; 4];
  let mut out_low = [0u16; 4];
  gbrapf32_to_rgba_u16_row::<false>(&g, &b, &r, &a_high, &mut out_high, 1);
  gbrapf32_to_rgba_u16_row::<false>(&g, &b, &r, &a_low, &mut out_low, 1);
  assert_eq!(out_high[3], 65535, "u16 alpha HDR clamps to 65535");
  assert_eq!(out_low[3], 0, "u16 alpha negative clamps to 0");
}

#[test]
fn gbrapf32_to_rgba_u16_be_parity() {
  let g_intended = [0.0f32, 0.25, 0.5, 1.0];
  let b_intended = [0.1f32, 0.3, 0.7, 0.9];
  let r_intended = [0.5f32, 0.8, 0.2, 0.6];
  let a_intended = [0.2f32, 0.4, 0.6, 0.8];
  let g_le = as_le_f32(&g_intended);
  let b_le = as_le_f32(&b_intended);
  let r_le = as_le_f32(&r_intended);
  let a_le = as_le_f32(&a_intended);
  let g_be = as_be_f32(&g_intended);
  let b_be = as_be_f32(&b_intended);
  let r_be = as_be_f32(&r_intended);
  let a_be = as_be_f32(&a_intended);
  let mut le_out = std::vec![0u16; 4 * 4];
  let mut be_out = std::vec![0u16; 4 * 4];
  gbrapf32_to_rgba_u16_row::<false>(&g_le, &b_le, &r_le, &a_le, &mut le_out, 4);
  gbrapf32_to_rgba_u16_row::<true>(&g_be, &b_be, &r_be, &a_be, &mut be_out, 4);
  let expected = ref_gbrapf32_to_rgba_u16(&g_intended, &b_intended, &r_intended, &a_intended, 4);
  assert_eq!(le_out, expected, "LE path must match scalar reference");
  assert_eq!(be_out, expected, "BE path must match scalar reference");
  assert_eq!(be_out, le_out, "BE gbrapf32_to_rgba_u16_row must match LE");
}

// ---- gbrapf32_to_rgba_f32_row (lossless source α) -------------------------

#[test]
fn gbrapf32_to_rgba_f32_lossless_passthrough() {
  // HDR 2.5, NaN, Inf, negative all preserved — including in α
  let g = as_le_f32(&[0.5f32]);
  let b = as_le_f32(&[0.5f32]);
  let r = as_le_f32(&[0.5f32]);
  let a = as_le_f32(&[2.5f32]);
  let mut out = [0.0f32; 4];
  gbrapf32_to_rgba_f32_row::<false>(&g, &b, &r, &a, &mut out, 1);
  assert_eq!(out[3], 2.5, "HDR alpha preserved bit-exact");
}

#[test]
fn gbrapf32_to_rgba_f32_nan_alpha_preserved() {
  let g = as_le_f32(&[0.5f32]);
  let b = as_le_f32(&[0.5f32]);
  let r = as_le_f32(&[0.5f32]);
  let a = as_le_f32(&[f32::NAN]);
  let mut out = [0.0f32; 4];
  gbrapf32_to_rgba_f32_row::<false>(&g, &b, &r, &a, &mut out, 1);
  assert!(out[3].is_nan(), "NaN alpha preserved");
}

#[test]
fn gbrapf32_to_rgba_f32_be_parity() {
  let g_intended = [0.0f32, 0.25, 0.5, 1.0];
  let b_intended = [0.1f32, 0.3, 0.7, 0.9];
  let r_intended = [0.5f32, 0.8, 0.2, 0.6];
  let a_intended = [0.2f32, 0.4, 0.6, 0.8];
  let g_le = as_le_f32(&g_intended);
  let b_le = as_le_f32(&b_intended);
  let r_le = as_le_f32(&r_intended);
  let a_le = as_le_f32(&a_intended);
  let g_be = as_be_f32(&g_intended);
  let b_be = as_be_f32(&b_intended);
  let r_be = as_be_f32(&r_intended);
  let a_be = as_be_f32(&a_intended);
  let mut le_out = std::vec![0.0f32; 4 * 4];
  let mut be_out = std::vec![0.0f32; 4 * 4];
  gbrapf32_to_rgba_f32_row::<false>(&g_le, &b_le, &r_le, &a_le, &mut le_out, 4);
  gbrapf32_to_rgba_f32_row::<true>(&g_be, &b_be, &r_be, &a_be, &mut be_out, 4);
  let expected = ref_gbrapf32_to_rgba_f32(&g_intended, &b_intended, &r_intended, &a_intended, 4);
  assert_eq!(le_out, expected, "LE path must match scalar reference");
  assert_eq!(be_out, expected, "BE path must match scalar reference");
  assert_eq!(be_out, le_out, "BE gbrapf32_to_rgba_f32_row must match LE");
}

// ---- gbrapf32_to_rgba_f16_row --------------------------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
)]
fn gbrapf32_to_rgba_f16_source_alpha_passthrough() {
  let g = [0.5f32];
  let b = [0.5f32];
  let r = [0.5f32];
  let a = [0.75f32];
  let mut out = vec![half::f16::ZERO; 4];
  gbrapf32_to_rgba_f16_row::<false>(&g, &b, &r, &a, &mut out, 1);
  assert_eq!(out[3], half::f16::from_f32(0.75), "f16 alpha from source");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
)]
fn gbrapf32_to_rgba_f16_hdr_alpha_saturates() {
  let g = [0.5f32];
  let b = [0.5f32];
  let r = [0.5f32];
  let a = [70_000.0f32];
  let mut out = vec![half::f16::ZERO; 4];
  gbrapf32_to_rgba_f16_row::<false>(&g, &b, &r, &a, &mut out, 1);
  assert!(
    out[3].is_infinite() && out[3].to_f32() > 0.0,
    "HDR alpha saturates to +Inf"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
)]
fn gbrapf32_to_rgba_f16_be_parity() {
  let g_intended = [0.0f32, 0.25, 0.5, 1.0];
  let b_intended = [0.1f32, 0.3, 0.7, 0.9];
  let r_intended = [0.5f32, 0.8, 0.2, 0.6];
  let a_intended = [0.2f32, 0.4, 0.6, 0.8];
  let g_le = as_le_f32(&g_intended);
  let b_le = as_le_f32(&b_intended);
  let r_le = as_le_f32(&r_intended);
  let a_le = as_le_f32(&a_intended);
  let g_be = as_be_f32(&g_intended);
  let b_be = as_be_f32(&b_intended);
  let r_be = as_be_f32(&r_intended);
  let a_be = as_be_f32(&a_intended);
  let mut le_out = vec![half::f16::ZERO; 4 * 4];
  let mut be_out = vec![half::f16::ZERO; 4 * 4];
  gbrapf32_to_rgba_f16_row::<false>(&g_le, &b_le, &r_le, &a_le, &mut le_out, 4);
  gbrapf32_to_rgba_f16_row::<true>(&g_be, &b_be, &r_be, &a_be, &mut be_out, 4);
  let expected = ref_gbrapf32_to_rgba_f16(&g_intended, &b_intended, &r_intended, &a_intended, 4);
  assert_eq!(le_out, expected, "LE path must match scalar reference");
  assert_eq!(be_out, expected, "BE path must match scalar reference");
  assert_eq!(be_out, le_out, "BE gbrapf32_to_rgba_f16_row must match LE");
}
