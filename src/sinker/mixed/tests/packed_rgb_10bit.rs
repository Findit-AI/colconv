use super::*;

// ---- Tier 6 — 10-bit packed RGB family (Ship 9e) -----------------------

/// Builds a row of `width` X2RGB10 LE pixels with explicit
/// channel values. Padding bits are set to `0` (well-behaved input).
fn solid_x2rgb10_frame(width: u32, height: u32, r10: u32, g10: u32, b10: u32) -> Vec<u8> {
  let w = width as usize;
  let h = height as usize;
  let pix: u32 = ((r10 & 0x3FF) << 20) | ((g10 & 0x3FF) << 10) | (b10 & 0x3FF);
  let mut buf = std::vec![0u8; w * h * 4];
  for px in buf.chunks_mut(4) {
    px.copy_from_slice(&pix.to_le_bytes());
  }
  buf
}

fn solid_x2bgr10_frame(width: u32, height: u32, r10: u32, g10: u32, b10: u32) -> Vec<u8> {
  let w = width as usize;
  let h = height as usize;
  // X2BGR10: R at low 10, G mid, B high.
  let pix: u32 = ((b10 & 0x3FF) << 20) | ((g10 & 0x3FF) << 10) | (r10 & 0x3FF);
  let mut buf = std::vec![0u8; w * h * 4];
  for px in buf.chunks_mut(4) {
    px.copy_from_slice(&pix.to_le_bytes());
  }
  buf
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn x2rgb10_with_rgb_downshifts_to_8bit() {
  // 10-bit channel `0x3FC` = 1020. >> 2 = 0xFF. So channels 0x3FC,
  // 0x200, 0x080 → u8 (0xFF, 0x80, 0x20).
  let pix = solid_x2rgb10_frame(16, 4, 0x3FC, 0x200, 0x080);
  let src = X2Rgb10Frame::try_new(&pix, 16, 4, 64).unwrap();

  let mut rgb_out = std::vec![0u8; 16 * 4 * 3];
  let mut sink = MixedSinker::<X2Rgb10>::new(16, 4)
    .with_rgb(&mut rgb_out)
    .unwrap();
  x2rgb10_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

  for px in rgb_out.chunks(3) {
    assert_eq!(px, [0xFF, 0x80, 0x20]);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn x2rgb10_with_rgba_forces_alpha_to_ff() {
  let pix = solid_x2rgb10_frame(16, 4, 0x3FC, 0x200, 0x080);
  let src = X2Rgb10Frame::try_new(&pix, 16, 4, 64).unwrap();

  let mut rgba_out = std::vec![0u8; 16 * 4 * 4];
  let mut sink = MixedSinker::<X2Rgb10>::new(16, 4)
    .with_rgba(&mut rgba_out)
    .unwrap();
  x2rgb10_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

  for px in rgba_out.chunks(4) {
    assert_eq!(px, [0xFF, 0x80, 0x20, 0xFF]);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn x2rgb10_with_rgb_u16_preserves_native_precision() {
  let pix = solid_x2rgb10_frame(16, 4, 0x3FC, 0x200, 0x080);
  let src = X2Rgb10Frame::try_new(&pix, 16, 4, 64).unwrap();

  let mut rgb_out = std::vec![0u16; 16 * 4 * 3];
  let mut sink = MixedSinker::<X2Rgb10>::new(16, 4)
    .with_rgb_u16(&mut rgb_out)
    .unwrap();
  x2rgb10_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

  for px in rgb_out.chunks(3) {
    assert_eq!(px, [0x3FC, 0x200, 0x080]);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn x2bgr10_with_rgb_swaps_channels() {
  // X2BGR10 byte positions: R at low, B at high. Sinker output is
  // still (R, G, B). Same numerical channels as x2rgb10 test.
  let pix = solid_x2bgr10_frame(16, 4, 0x3FC, 0x200, 0x080);
  let src = X2Bgr10Frame::try_new(&pix, 16, 4, 64).unwrap();

  let mut rgb_out = std::vec![0u8; 16 * 4 * 3];
  let mut sink = MixedSinker::<X2Bgr10>::new(16, 4)
    .with_rgb(&mut rgb_out)
    .unwrap();
  x2bgr10_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

  for px in rgb_out.chunks(3) {
    assert_eq!(px, [0xFF, 0x80, 0x20]);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn x2bgr10_with_rgba_swaps_and_forces_alpha() {
  let pix = solid_x2bgr10_frame(16, 4, 0x3FC, 0x200, 0x080);
  let src = X2Bgr10Frame::try_new(&pix, 16, 4, 64).unwrap();

  let mut rgba_out = std::vec![0u8; 16 * 4 * 4];
  let mut sink = MixedSinker::<X2Bgr10>::new(16, 4)
    .with_rgba(&mut rgba_out)
    .unwrap();
  x2bgr10_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

  for px in rgba_out.chunks(4) {
    assert_eq!(px, [0xFF, 0x80, 0x20, 0xFF]);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn x2rgb10_simd_matches_scalar_with_random_input() {
  // Width 1921 forces both SIMD main loop AND scalar tail across
  // every backend block size. HSV omitted — see Ship 9b for the
  // ±1 LSB rationale.
  let w = 1921usize;
  let h = 4usize;
  let mut pix = std::vec![0u8; w * h * 4];
  pseudo_random_u8(&mut pix, 0x1234_5678);
  let src = X2Rgb10Frame::try_new(&pix, w as u32, h as u32, (w * 4) as u32).unwrap();

  let mut rgb_simd = std::vec![0u8; w * h * 3];
  let mut rgb_scalar = std::vec![0u8; w * h * 3];
  let mut rgba_simd = std::vec![0u8; w * h * 4];
  let mut rgba_scalar = std::vec![0u8; w * h * 4];
  let mut rgb_u16_simd = std::vec![0u16; w * h * 3];
  let mut rgb_u16_scalar = std::vec![0u16; w * h * 3];
  let mut luma_simd = std::vec![0u8; w * h];
  let mut luma_scalar = std::vec![0u8; w * h];

  let mut s_simd = MixedSinker::<X2Rgb10>::new(w, h)
    .with_rgb(&mut rgb_simd)
    .unwrap()
    .with_rgba(&mut rgba_simd)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16_simd)
    .unwrap()
    .with_luma(&mut luma_simd)
    .unwrap();
  x2rgb10_to(&src, true, ColorMatrix::Bt709, &mut s_simd).unwrap();

  let mut s_scalar = MixedSinker::<X2Rgb10>::new(w, h)
    .with_rgb(&mut rgb_scalar)
    .unwrap()
    .with_rgba(&mut rgba_scalar)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16_scalar)
    .unwrap()
    .with_luma(&mut luma_scalar)
    .unwrap();
  s_scalar.set_simd(false);
  x2rgb10_to(&src, true, ColorMatrix::Bt709, &mut s_scalar).unwrap();

  assert_eq!(rgb_simd, rgb_scalar, "RGB output diverges");
  assert_eq!(rgba_simd, rgba_scalar, "RGBA output diverges");
  assert_eq!(rgb_u16_simd, rgb_u16_scalar, "RGB u16 output diverges");
  assert_eq!(luma_simd, luma_scalar, "Luma output diverges");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn x2bgr10_simd_matches_scalar_with_random_input() {
  let w = 1921usize;
  let h = 4usize;
  let mut pix = std::vec![0u8; w * h * 4];
  pseudo_random_u8(&mut pix, 0xABCD_EF01);
  let src = X2Bgr10Frame::try_new(&pix, w as u32, h as u32, (w * 4) as u32).unwrap();

  let mut rgb_simd = std::vec![0u8; w * h * 3];
  let mut rgb_scalar = std::vec![0u8; w * h * 3];
  let mut rgba_simd = std::vec![0u8; w * h * 4];
  let mut rgba_scalar = std::vec![0u8; w * h * 4];
  let mut rgb_u16_simd = std::vec![0u16; w * h * 3];
  let mut rgb_u16_scalar = std::vec![0u16; w * h * 3];
  let mut luma_simd = std::vec![0u8; w * h];
  let mut luma_scalar = std::vec![0u8; w * h];

  let mut s_simd = MixedSinker::<X2Bgr10>::new(w, h)
    .with_rgb(&mut rgb_simd)
    .unwrap()
    .with_rgba(&mut rgba_simd)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16_simd)
    .unwrap()
    .with_luma(&mut luma_simd)
    .unwrap();
  x2bgr10_to(&src, true, ColorMatrix::Bt709, &mut s_simd).unwrap();

  let mut s_scalar = MixedSinker::<X2Bgr10>::new(w, h)
    .with_rgb(&mut rgb_scalar)
    .unwrap()
    .with_rgba(&mut rgba_scalar)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16_scalar)
    .unwrap()
    .with_luma(&mut luma_scalar)
    .unwrap();
  s_scalar.set_simd(false);
  x2bgr10_to(&src, true, ColorMatrix::Bt709, &mut s_scalar).unwrap();

  assert_eq!(rgb_simd, rgb_scalar, "RGB output diverges");
  assert_eq!(rgba_simd, rgba_scalar, "RGBA output diverges");
  assert_eq!(rgb_u16_simd, rgb_u16_scalar, "RGB u16 output diverges");
  assert_eq!(luma_simd, luma_scalar, "Luma output diverges");
}
