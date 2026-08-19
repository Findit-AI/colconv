use super::*;
use crate::{
  frame::{Rgb96BeFrame, Rgb96Frame, Rgba128BeFrame, Rgba128Frame},
  sinker::mixed::MixedSinker,
  source::{Rgb96, Rgba128, rgb96_to, rgba128_to, rgba128_to_endian},
};

/// Re-encode a host-native u32 slice as LE-encoded byte storage. Sink kernels
/// recover the intended logical values via `u32::from_le` on both LE (no-op)
/// and BE (byte-swap) hosts.
fn as_le_u32(host: &[u32]) -> std::vec::Vec<u32> {
  host
    .iter()
    .map(|v| u32::from_ne_bytes(v.to_le_bytes()))
    .collect()
}

/// Mirror of [`as_le_u32`] for the BE-encoded source path.
fn as_be_u32(host: &[u32]) -> std::vec::Vec<u32> {
  host
    .iter()
    .map(|v| u32::from_ne_bytes(v.to_be_bytes()))
    .collect()
}

// ---- Rgb96 -----------------------------------------------------------------

#[test]
fn rgb96_with_rgb_u16_narrows_16() {
  // Each 32-bit channel narrowed >> 16.
  let src = as_le_u32(&[0x1234_5678, 0x5678_9ABC, 0x9ABC_DEF0]);
  let frame = Rgb96Frame::new(&src, 1, 1, 3);
  let mut out = vec![0u16; 3];
  let mut sinker = MixedSinker::<Rgb96>::new(1, 1)
    .with_rgb_u16(&mut out)
    .unwrap();
  rgb96_to(&frame, true, sinker.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
  assert_eq!(out, vec![0x1234u16, 0x5678, 0x9ABC]);
}

#[test]
fn rgb96_with_rgb_narrows_24() {
  // Each 32-bit channel narrowed >> 24.
  let src = as_le_u32(&[0xFF00_0000, 0x8000_0000, 0x0100_0000]);
  let frame = Rgb96Frame::new(&src, 1, 1, 3);
  let mut out = vec![0u8; 3];
  let mut sinker = MixedSinker::<Rgb96>::new(1, 1).with_rgb(&mut out).unwrap();
  rgb96_to(&frame, true, sinker.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
  assert_eq!(out, vec![0xFFu8, 0x80, 0x01]);
}

#[test]
fn rgb96_with_rgba_forces_0xff() {
  let src = as_le_u32(&[0xFFFF_FFFF, 0x8000_0000, 0x0000_0000]);
  let frame = Rgb96Frame::new(&src, 1, 1, 3);
  let mut out = vec![0u8; 4];
  let mut sinker = MixedSinker::<Rgb96>::new(1, 1).with_rgba(&mut out).unwrap();
  rgb96_to(&frame, true, sinker.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
  assert_eq!(out[3], 0xFF); // alpha forced to opaque
  assert_eq!(out[0], 0xFF);
}

#[test]
fn rgb96_with_rgba_u16_forces_0xffff() {
  let src = as_le_u32(&[0x1000_0000, 0x2000_0000, 0x3000_0000]);
  let frame = Rgb96Frame::new(&src, 1, 1, 3);
  let mut out = vec![0u16; 4];
  let mut sinker = MixedSinker::<Rgb96>::new(1, 1)
    .with_rgba_u16(&mut out)
    .unwrap();
  rgb96_to(&frame, true, sinker.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
  assert_eq!(out[0], 0x1000);
  assert_eq!(out[3], 0xFFFF);
}

#[test]
fn rgb96_with_rgb_u16_and_rgba_u16_both_correct() {
  let intended: [u32; 6] = [
    0x1234_0000,
    0x5678_0000,
    0x9ABC_0000,
    0xDEF0_0000,
    0x1357_0000,
    0x2468_0000,
  ];
  let src = as_le_u32(&intended);
  let frame = Rgb96Frame::new(&src, 2, 1, 6);
  let mut rgb_u16 = vec![0u16; 6];
  let mut rgba_u16 = vec![0u16; 8];
  let mut sinker = MixedSinker::<Rgb96>::new(2, 1)
    .with_rgb_u16(&mut rgb_u16)
    .unwrap()
    .with_rgba_u16(&mut rgba_u16)
    .unwrap();
  rgb96_to(&frame, true, sinker.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
  assert_eq!(
    rgb_u16,
    vec![0x1234u16, 0x5678, 0x9ABC, 0xDEF0, 0x1357, 0x2468]
  );
  assert_eq!(rgba_u16[3], 0xFFFF);
  assert_eq!(rgba_u16[7], 0xFFFF);
}

#[test]
fn rgb96_with_luma_non_zero_for_nonzero_input() {
  let src = as_le_u32(&[0x8000_0000, 0x8000_0000, 0x8000_0000]);
  let frame = Rgb96Frame::new(&src, 1, 1, 3);
  let mut luma = vec![0u8; 1];
  let mut sinker = MixedSinker::<Rgb96>::new(1, 1)
    .with_luma(&mut luma)
    .unwrap();
  rgb96_to(&frame, true, sinker.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
  assert_ne!(luma[0], 0);
}

#[test]
fn rgb96_with_luma_u16_non_zero_for_nonzero_input() {
  let src = as_le_u32(&[0x8000_0000, 0x8000_0000, 0x8000_0000]);
  let frame = Rgb96Frame::new(&src, 1, 1, 3);
  let mut luma_u16 = vec![0u16; 1];
  let mut sinker = MixedSinker::<Rgb96>::new(1, 1)
    .with_luma_u16(&mut luma_u16)
    .unwrap();
  rgb96_to(&frame, true, sinker.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
  assert_ne!(luma_u16[0], 0);
}

#[test]
fn rgb96_with_hsv_value_tracks_max_channel() {
  // Pure red at full intensity → HSV value (V) channel is max (255).
  let src = as_le_u32(&[0xFFFF_FFFF, 0x0000_0000, 0x0000_0000]);
  let frame = Rgb96Frame::new(&src, 1, 1, 3);
  let mut h = vec![0u8; 1];
  let mut s = vec![0u8; 1];
  let mut v = vec![0u8; 1];
  let mut sinker = MixedSinker::<Rgb96>::new(1, 1)
    .with_hsv(&mut h, &mut s, &mut v)
    .unwrap();
  rgb96_to(&frame, true, sinker.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
  assert_eq!(v[0], 0xFF, "value tracks max channel");
  assert_eq!(s[0], 0xFF, "fully saturated");
}

#[test]
fn rgb96_simd_matches_scalar() {
  // width=19: SIMD path (if available) vs forced scalar path, all outputs.
  let w: usize = 19;
  let src: Vec<u32> = (0..w * 3)
    .map(|i| (i as u32).wrapping_mul(0x0100_1001))
    .collect();
  let frame = Rgb96Frame::new(&src, w as u32, 1, w as u32 * 3);
  let mut rgb_simd = vec![0u8; w * 3];
  let mut rgba_simd = vec![0u8; w * 4];
  let mut rgb_u16_simd = vec![0u16; w * 3];
  let mut rgba_u16_simd = vec![0u16; w * 4];
  let mut s1 = MixedSinker::<Rgb96>::new(w, 1)
    .with_rgb(&mut rgb_simd)
    .unwrap()
    .with_rgba(&mut rgba_simd)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16_simd)
    .unwrap()
    .with_rgba_u16(&mut rgba_u16_simd)
    .unwrap();
  rgb96_to(&frame, true, s1.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();

  let mut rgb_scalar = vec![0u8; w * 3];
  let mut rgba_scalar = vec![0u8; w * 4];
  let mut rgb_u16_scalar = vec![0u16; w * 3];
  let mut rgba_u16_scalar = vec![0u16; w * 4];
  let mut s2 = MixedSinker::<Rgb96>::new(w, 1)
    .with_rgb(&mut rgb_scalar)
    .unwrap()
    .with_rgba(&mut rgba_scalar)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16_scalar)
    .unwrap()
    .with_rgba_u16(&mut rgba_u16_scalar)
    .unwrap();
  s2.set_simd(false);
  rgb96_to(&frame, true, s2.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();

  assert_eq!(rgb_simd, rgb_scalar, "rgb");
  assert_eq!(rgba_simd, rgba_scalar, "rgba");
  assert_eq!(rgb_u16_simd, rgb_u16_scalar, "rgb_u16");
  assert_eq!(rgba_u16_simd, rgba_u16_scalar, "rgba_u16");
}

#[test]
fn rgb96_be_le_decode_agree() {
  // A single host-native `intended` fixture re-encoded LE / BE must decode to
  // the same host-native u16 RGB through the matching-endian frame + walker.
  let intended: [u32; 6] = [
    0x1234_5678,
    0x5678_9ABC,
    0x9ABC_DEF0,
    0x0011_2233,
    0x4455_6677,
    0x8899_AABB,
  ];
  let src_le = as_le_u32(&intended);
  let src_be = as_be_u32(&intended);
  let le_frame = Rgb96Frame::new(&src_le, 2, 1, 6);
  let be_frame = Rgb96BeFrame::new(&src_be, 2, 1, 6);
  let mut out_le = vec![0u16; 6];
  let mut out_be = vec![0u16; 6];
  let mut s_le = MixedSinker::<Rgb96>::new(2, 1)
    .with_rgb_u16(&mut out_le)
    .unwrap();
  rgb96_to(&le_frame, true, s_le.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
  let mut s_be = MixedSinker::<Rgb96<true>>::new(2, 1)
    .with_rgb_u16(&mut out_be)
    .unwrap();
  crate::source::rgb96_to_endian::<_, true>(
    &be_frame,
    true,
    s_be.set_kernel_matrix(KernelMatrix::Bt709),
  )
  .unwrap();
  let expected: Vec<u16> = intended.iter().map(|&v| (v >> 16) as u16).collect();
  assert_eq!(out_le, expected, "LE decode");
  assert_eq!(out_be, expected, "BE decode");
  assert_eq!(out_le, out_be, "BE and LE agree");
}

// ---- Rgba128 ---------------------------------------------------------------

#[test]
fn rgba128_with_rgba_passes_source_alpha() {
  // All 4 channels narrowed >> 24; source alpha passes through (not forced).
  let src = as_le_u32(&[0x1100_0000, 0x2200_0000, 0x3300_0000, 0x4400_0000]);
  let frame = Rgba128Frame::new(&src, 1, 1, 4);
  let mut out = vec![0u8; 4];
  let mut sinker = MixedSinker::<Rgba128>::new(1, 1)
    .with_rgba(&mut out)
    .unwrap();
  rgba128_to(&frame, true, sinker.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
  assert_eq!(
    out,
    vec![0x11u8, 0x22, 0x33, 0x44],
    "source alpha preserved"
  );
}

#[test]
fn rgba128_with_rgb_drops_alpha() {
  let src = as_le_u32(&[0x1100_0000, 0x2200_0000, 0x3300_0000, 0x4400_0000]);
  let frame = Rgba128Frame::new(&src, 1, 1, 4);
  let mut out = vec![0u8; 3];
  let mut sinker = MixedSinker::<Rgba128>::new(1, 1)
    .with_rgb(&mut out)
    .unwrap();
  rgba128_to(&frame, true, sinker.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
  assert_eq!(out, vec![0x11u8, 0x22, 0x33]);
}

#[test]
fn rgba128_with_rgba_u16_narrows_alpha_16() {
  let src = as_le_u32(&[0x1122_0000, 0x3344_0000, 0x5566_0000, 0x7788_0000]);
  let frame = Rgba128Frame::new(&src, 1, 1, 4);
  let mut out = vec![0u16; 4];
  let mut sinker = MixedSinker::<Rgba128>::new(1, 1)
    .with_rgba_u16(&mut out)
    .unwrap();
  rgba128_to(&frame, true, sinker.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
  assert_eq!(out, vec![0x1122u16, 0x3344, 0x5566, 0x7788]);
}

#[test]
fn rgba128_strategy_a_plus_rgb_and_rgba_match_standalone() {
  // with_rgb + with_rgba combined (Strategy A+) must be byte-identical to the
  // standalone rgba128_to_rgba kernel — the α slot comes from the source, not
  // forced opaque.
  let w = 5;
  let src: Vec<u32> = (0..w * 4)
    .map(|i| (i as u32).wrapping_mul(0x0123_4567).wrapping_add(7))
    .collect();
  let frame = Rgba128Frame::new(&src, w as u32, 1, (w * 4) as u32);

  let mut rgb = vec![0u8; w * 3];
  let mut rgba_combined = vec![0u8; w * 4];
  let mut sink = MixedSinker::<Rgba128>::new(w, 1)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_rgba(&mut rgba_combined)
    .unwrap();
  rgba128_to(&frame, true, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();

  let mut rgba_standalone = vec![0u8; w * 4];
  let mut sink2 = MixedSinker::<Rgba128>::new(w, 1)
    .with_rgba(&mut rgba_standalone)
    .unwrap();
  rgba128_to(&frame, true, sink2.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();

  assert_eq!(
    rgba_combined, rgba_standalone,
    "Strategy A+ RGBA must match standalone"
  );
}

#[test]
fn rgba128_simd_matches_scalar() {
  let w = 19;
  let src: Vec<u32> = (0..w * 4)
    .map(|i| (i as u32).wrapping_mul(0x0100_1001).wrapping_add(0xABCD))
    .collect();
  let frame = Rgba128Frame::new(&src, w as u32, 1, (w * 4) as u32);
  let mut rgb_simd = vec![0u8; w * 3];
  let mut rgba_simd = vec![0u8; w * 4];
  let mut rgb_u16_simd = vec![0u16; w * 3];
  let mut rgba_u16_simd = vec![0u16; w * 4];
  let mut s1 = MixedSinker::<Rgba128>::new(w, 1)
    .with_rgb(&mut rgb_simd)
    .unwrap()
    .with_rgba(&mut rgba_simd)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16_simd)
    .unwrap()
    .with_rgba_u16(&mut rgba_u16_simd)
    .unwrap();
  rgba128_to(&frame, true, s1.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();

  let mut rgb_scalar = vec![0u8; w * 3];
  let mut rgba_scalar = vec![0u8; w * 4];
  let mut rgb_u16_scalar = vec![0u16; w * 3];
  let mut rgba_u16_scalar = vec![0u16; w * 4];
  let mut s2 = MixedSinker::<Rgba128>::new(w, 1)
    .with_rgb(&mut rgb_scalar)
    .unwrap()
    .with_rgba(&mut rgba_scalar)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16_scalar)
    .unwrap()
    .with_rgba_u16(&mut rgba_u16_scalar)
    .unwrap();
  s2.set_simd(false);
  rgba128_to(&frame, true, s2.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();

  assert_eq!(rgb_simd, rgb_scalar, "rgb");
  assert_eq!(rgba_simd, rgba_scalar, "rgba");
  assert_eq!(rgb_u16_simd, rgb_u16_scalar, "rgb_u16");
  assert_eq!(rgba_u16_simd, rgba_u16_scalar, "rgba_u16");
}

#[test]
fn rgba128_be_le_decode_agree() {
  let intended: [u32; 8] = [
    0x1234_5678,
    0x5678_9ABC,
    0x9ABC_DEF0,
    0x0011_2233,
    0x4455_6677,
    0x8899_AABB,
    0xCCDD_EEFF,
    0x1357_9BDF,
  ];
  let src_le = as_le_u32(&intended);
  let src_be = as_be_u32(&intended);
  let le_frame = Rgba128Frame::new(&src_le, 2, 1, 8);
  let be_frame = Rgba128BeFrame::new(&src_be, 2, 1, 8);
  let mut out_le = vec![0u16; 8];
  let mut out_be = vec![0u16; 8];
  let mut s_le = MixedSinker::<Rgba128>::new(2, 1)
    .with_rgba_u16(&mut out_le)
    .unwrap();
  rgba128_to(&le_frame, true, s_le.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
  let mut s_be = MixedSinker::<Rgba128<true>>::new(2, 1)
    .with_rgba_u16(&mut out_be)
    .unwrap();
  rgba128_to_endian::<_, true>(&be_frame, true, s_be.set_kernel_matrix(KernelMatrix::Bt709))
    .unwrap();
  let expected: Vec<u16> = intended.iter().map(|&v| (v >> 16) as u16).collect();
  assert_eq!(out_le, expected, "LE decode");
  assert_eq!(out_be, expected, "BE decode");
}
