use super::*;

// Frame-building helpers.

/// Build a frame buffer where every pixel has the given LE u16 word value.
fn solid_frame_le16(width: u32, height: u32, pixel: u16) -> std::vec::Vec<u8> {
  let px = pixel.to_le_bytes();
  let n = width as usize * height as usize;
  let mut buf = std::vec::Vec::with_capacity(n * 2);
  for _ in 0..n {
    buf.extend_from_slice(&px);
  }
  buf
}

/// Build a pseudo-random frame buffer masked to `mask` bits per pixel.
fn random_legacy_frame(width: u32, height: u32, seed: u32, mask: u16) -> std::vec::Vec<u8> {
  let n = width as usize * height as usize;
  let mut buf = std::vec![0u8; n * 2];
  let mut state = seed;
  for chunk in buf.chunks_mut(2) {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    let word = ((state >> 8) as u16) & mask;
    chunk.copy_from_slice(&word.to_le_bytes());
  }
  buf
}

// SIMD-vs-scalar parity (all 6 formats).

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgb565_simd_matches_scalar() {
  let w = 1921usize;
  let h = 4usize;
  let pix = random_legacy_frame(w as u32, h as u32, 0xDEAD_BEEF, 0xFFFF);
  let src = Rgb565Frame::try_new(&pix, w as u32, h as u32, (w * 2) as u32).unwrap();

  let mut rgb_simd = std::vec![0u8; w * h * 3];
  let mut rgb_scalar = std::vec![0u8; w * h * 3];
  let mut rgba_simd = std::vec![0u8; w * h * 4];
  let mut rgba_scalar = std::vec![0u8; w * h * 4];
  let mut rgb_u16_simd = std::vec![0u16; w * h * 3];
  let mut rgb_u16_scalar = std::vec![0u16; w * h * 3];
  let mut rgba_u16_simd = std::vec![0u16; w * h * 4];
  let mut rgba_u16_scalar = std::vec![0u16; w * h * 4];
  let mut luma_simd = std::vec![0u8; w * h];
  let mut luma_scalar = std::vec![0u8; w * h];

  let mut s = MixedSinker::<Rgb565>::new(w, h)
    .with_rgb(&mut rgb_simd)
    .unwrap()
    .with_rgba(&mut rgba_simd)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16_simd)
    .unwrap()
    .with_rgba_u16(&mut rgba_u16_simd)
    .unwrap()
    .with_luma(&mut luma_simd)
    .unwrap();
  rgb565_to(&src, true, KernelMatrix::Bt709, &mut s).unwrap();

  let mut s2 = MixedSinker::<Rgb565>::new(w, h)
    .with_rgb(&mut rgb_scalar)
    .unwrap()
    .with_rgba(&mut rgba_scalar)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16_scalar)
    .unwrap()
    .with_rgba_u16(&mut rgba_u16_scalar)
    .unwrap()
    .with_luma(&mut luma_scalar)
    .unwrap();
  s2.set_simd(false);
  rgb565_to(&src, true, KernelMatrix::Bt709, &mut s2).unwrap();

  assert_eq!(rgb_simd, rgb_scalar, "RGB565 rgb output diverges");
  assert_eq!(rgba_simd, rgba_scalar, "RGB565 rgba output diverges");
  assert_eq!(
    rgb_u16_simd, rgb_u16_scalar,
    "RGB565 rgb_u16 output diverges"
  );
  assert_eq!(
    rgba_u16_simd, rgba_u16_scalar,
    "RGB565 rgba_u16 output diverges"
  );
  assert_eq!(luma_simd, luma_scalar, "RGB565 luma output diverges");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bgr565_simd_matches_scalar() {
  let w = 1921usize;
  let h = 4usize;
  let pix = random_legacy_frame(w as u32, h as u32, 0xCAFE_BABE, 0xFFFF);
  let src = Bgr565Frame::try_new(&pix, w as u32, h as u32, (w * 2) as u32).unwrap();

  let mut rgb_simd = std::vec![0u8; w * h * 3];
  let mut rgb_scalar = std::vec![0u8; w * h * 3];
  let mut rgba_simd = std::vec![0u8; w * h * 4];
  let mut rgba_scalar = std::vec![0u8; w * h * 4];
  let mut rgb_u16_simd = std::vec![0u16; w * h * 3];
  let mut rgb_u16_scalar = std::vec![0u16; w * h * 3];

  let mut s = MixedSinker::<Bgr565>::new(w, h)
    .with_rgb(&mut rgb_simd)
    .unwrap()
    .with_rgba(&mut rgba_simd)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16_simd)
    .unwrap();
  bgr565_to(&src, true, KernelMatrix::Bt709, &mut s).unwrap();

  let mut s2 = MixedSinker::<Bgr565>::new(w, h)
    .with_rgb(&mut rgb_scalar)
    .unwrap()
    .with_rgba(&mut rgba_scalar)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16_scalar)
    .unwrap();
  s2.set_simd(false);
  bgr565_to(&src, true, KernelMatrix::Bt709, &mut s2).unwrap();

  assert_eq!(rgb_simd, rgb_scalar, "BGR565 rgb output diverges");
  assert_eq!(rgba_simd, rgba_scalar, "BGR565 rgba output diverges");
  assert_eq!(
    rgb_u16_simd, rgb_u16_scalar,
    "BGR565 rgb_u16 output diverges"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgb555_simd_matches_scalar() {
  let w = 1921usize;
  let h = 4usize;
  // RGB555: bit 15 is padding — mask to [14:0] = 0x7FFF.
  let pix = random_legacy_frame(w as u32, h as u32, 0xABCD_1234, 0x7FFF);
  let src = Rgb555Frame::try_new(&pix, w as u32, h as u32, (w * 2) as u32).unwrap();

  let mut rgb_simd = std::vec![0u8; w * h * 3];
  let mut rgb_scalar = std::vec![0u8; w * h * 3];
  let mut rgba_simd = std::vec![0u8; w * h * 4];
  let mut rgba_scalar = std::vec![0u8; w * h * 4];
  let mut rgb_u16_simd = std::vec![0u16; w * h * 3];
  let mut rgb_u16_scalar = std::vec![0u16; w * h * 3];

  let mut s = MixedSinker::<Rgb555>::new(w, h)
    .with_rgb(&mut rgb_simd)
    .unwrap()
    .with_rgba(&mut rgba_simd)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16_simd)
    .unwrap();
  rgb555_to(&src, true, KernelMatrix::Bt709, &mut s).unwrap();

  let mut s2 = MixedSinker::<Rgb555>::new(w, h)
    .with_rgb(&mut rgb_scalar)
    .unwrap()
    .with_rgba(&mut rgba_scalar)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16_scalar)
    .unwrap();
  s2.set_simd(false);
  rgb555_to(&src, true, KernelMatrix::Bt709, &mut s2).unwrap();

  assert_eq!(rgb_simd, rgb_scalar, "RGB555 rgb output diverges");
  assert_eq!(rgba_simd, rgba_scalar, "RGB555 rgba output diverges");
  assert_eq!(
    rgb_u16_simd, rgb_u16_scalar,
    "RGB555 rgb_u16 output diverges"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bgr555_simd_matches_scalar() {
  let w = 1921usize;
  let h = 4usize;
  let pix = random_legacy_frame(w as u32, h as u32, 0x1357_2468, 0x7FFF);
  let src = Bgr555Frame::try_new(&pix, w as u32, h as u32, (w * 2) as u32).unwrap();

  let mut rgb_simd = std::vec![0u8; w * h * 3];
  let mut rgb_scalar = std::vec![0u8; w * h * 3];
  let mut rgba_simd = std::vec![0u8; w * h * 4];
  let mut rgba_scalar = std::vec![0u8; w * h * 4];

  let mut s = MixedSinker::<Bgr555>::new(w, h)
    .with_rgb(&mut rgb_simd)
    .unwrap()
    .with_rgba(&mut rgba_simd)
    .unwrap();
  bgr555_to(&src, true, KernelMatrix::Bt709, &mut s).unwrap();

  let mut s2 = MixedSinker::<Bgr555>::new(w, h)
    .with_rgb(&mut rgb_scalar)
    .unwrap()
    .with_rgba(&mut rgba_scalar)
    .unwrap();
  s2.set_simd(false);
  bgr555_to(&src, true, KernelMatrix::Bt709, &mut s2).unwrap();

  assert_eq!(rgb_simd, rgb_scalar, "BGR555 rgb output diverges");
  assert_eq!(rgba_simd, rgba_scalar, "BGR555 rgba output diverges");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgb444_simd_matches_scalar() {
  let w = 1921usize;
  let h = 4usize;
  // RGB444: bits [15:12] are padding — mask to [11:0] = 0x0FFF.
  let pix = random_legacy_frame(w as u32, h as u32, 0xFEDC_BA98, 0x0FFF);
  let src = Rgb444Frame::try_new(&pix, w as u32, h as u32, (w * 2) as u32).unwrap();

  let mut rgb_simd = std::vec![0u8; w * h * 3];
  let mut rgb_scalar = std::vec![0u8; w * h * 3];
  let mut rgba_simd = std::vec![0u8; w * h * 4];
  let mut rgba_scalar = std::vec![0u8; w * h * 4];
  let mut rgb_u16_simd = std::vec![0u16; w * h * 3];
  let mut rgb_u16_scalar = std::vec![0u16; w * h * 3];

  let mut s = MixedSinker::<Rgb444>::new(w, h)
    .with_rgb(&mut rgb_simd)
    .unwrap()
    .with_rgba(&mut rgba_simd)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16_simd)
    .unwrap();
  rgb444_to(&src, true, KernelMatrix::Bt709, &mut s).unwrap();

  let mut s2 = MixedSinker::<Rgb444>::new(w, h)
    .with_rgb(&mut rgb_scalar)
    .unwrap()
    .with_rgba(&mut rgba_scalar)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16_scalar)
    .unwrap();
  s2.set_simd(false);
  rgb444_to(&src, true, KernelMatrix::Bt709, &mut s2).unwrap();

  assert_eq!(rgb_simd, rgb_scalar, "RGB444 rgb output diverges");
  assert_eq!(rgba_simd, rgba_scalar, "RGB444 rgba output diverges");
  assert_eq!(
    rgb_u16_simd, rgb_u16_scalar,
    "RGB444 rgb_u16 output diverges"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bgr444_simd_matches_scalar() {
  let w = 1921usize;
  let h = 4usize;
  let pix = random_legacy_frame(w as u32, h as u32, 0x0123_4567, 0x0FFF);
  let src = Bgr444Frame::try_new(&pix, w as u32, h as u32, (w * 2) as u32).unwrap();

  let mut rgb_simd = std::vec![0u8; w * h * 3];
  let mut rgb_scalar = std::vec![0u8; w * h * 3];
  let mut rgba_simd = std::vec![0u8; w * h * 4];
  let mut rgba_scalar = std::vec![0u8; w * h * 4];

  let mut s = MixedSinker::<Bgr444>::new(w, h)
    .with_rgb(&mut rgb_simd)
    .unwrap()
    .with_rgba(&mut rgba_simd)
    .unwrap();
  bgr444_to(&src, true, KernelMatrix::Bt709, &mut s).unwrap();

  let mut s2 = MixedSinker::<Bgr444>::new(w, h)
    .with_rgb(&mut rgb_scalar)
    .unwrap()
    .with_rgba(&mut rgba_scalar)
    .unwrap();
  s2.set_simd(false);
  bgr444_to(&src, true, KernelMatrix::Bt709, &mut s2).unwrap();

  assert_eq!(rgb_simd, rgb_scalar, "BGR444 rgb output diverges");
  assert_eq!(rgba_simd, rgba_scalar, "BGR444 rgba output diverges");
}

// u8 RGBA alpha = 0xFF (all 6 formats).

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgb565_rgba_forces_alpha_ff() {
  // Pixel 0x07E0 = G=63, R=0, B=0 → R=0, G=255, B=0, A=0xFF.
  let pix = solid_frame_le16(8, 2, 0x07E0);
  let src = Rgb565Frame::try_new(&pix, 8, 2, 16).unwrap();
  let mut rgba_out = std::vec![0u8; 8 * 2 * 4];
  let mut sink = MixedSinker::<Rgb565>::new(8, 2)
    .with_rgba(&mut rgba_out)
    .unwrap();
  rgb565_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  for px in rgba_out.chunks(4) {
    assert_eq!(px[3], 0xFF, "RGB565 RGBA alpha not 0xFF");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bgr565_rgba_forces_alpha_ff() {
  let pix = solid_frame_le16(8, 2, 0x07E0);
  let src = Bgr565Frame::try_new(&pix, 8, 2, 16).unwrap();
  let mut rgba_out = std::vec![0u8; 8 * 2 * 4];
  let mut sink = MixedSinker::<Bgr565>::new(8, 2)
    .with_rgba(&mut rgba_out)
    .unwrap();
  bgr565_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  for px in rgba_out.chunks(4) {
    assert_eq!(px[3], 0xFF, "BGR565 RGBA alpha not 0xFF");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgb555_rgba_forces_alpha_ff() {
  // Max-value RGB555 pixel = 0x7FFF (all 5-bit channels set, bit 15 = 0).
  let pix = solid_frame_le16(8, 2, 0x7FFF);
  let src = Rgb555Frame::try_new(&pix, 8, 2, 16).unwrap();
  let mut rgba_out = std::vec![0u8; 8 * 2 * 4];
  let mut sink = MixedSinker::<Rgb555>::new(8, 2)
    .with_rgba(&mut rgba_out)
    .unwrap();
  rgb555_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  for px in rgba_out.chunks(4) {
    assert_eq!(px[3], 0xFF, "RGB555 RGBA alpha not 0xFF");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bgr555_rgba_forces_alpha_ff() {
  let pix = solid_frame_le16(8, 2, 0x7FFF);
  let src = Bgr555Frame::try_new(&pix, 8, 2, 16).unwrap();
  let mut rgba_out = std::vec![0u8; 8 * 2 * 4];
  let mut sink = MixedSinker::<Bgr555>::new(8, 2)
    .with_rgba(&mut rgba_out)
    .unwrap();
  bgr555_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  for px in rgba_out.chunks(4) {
    assert_eq!(px[3], 0xFF, "BGR555 RGBA alpha not 0xFF");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgb444_rgba_forces_alpha_ff() {
  // Max-value RGB444 pixel = 0x0FFF.
  let pix = solid_frame_le16(8, 2, 0x0FFF);
  let src = Rgb444Frame::try_new(&pix, 8, 2, 16).unwrap();
  let mut rgba_out = std::vec![0u8; 8 * 2 * 4];
  let mut sink = MixedSinker::<Rgb444>::new(8, 2)
    .with_rgba(&mut rgba_out)
    .unwrap();
  rgb444_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  for px in rgba_out.chunks(4) {
    assert_eq!(px[3], 0xFF, "RGB444 RGBA alpha not 0xFF");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bgr444_rgba_forces_alpha_ff() {
  let pix = solid_frame_le16(8, 2, 0x0FFF);
  let src = Bgr444Frame::try_new(&pix, 8, 2, 16).unwrap();
  let mut rgba_out = std::vec![0u8; 8 * 2 * 4];
  let mut sink = MixedSinker::<Bgr444>::new(8, 2)
    .with_rgba(&mut rgba_out)
    .unwrap();
  bgr444_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  for px in rgba_out.chunks(4) {
    assert_eq!(px[3], 0xFF, "BGR444 RGBA alpha not 0xFF");
  }
}

// u16 RGBA alpha = 0xFFFF (3 representative formats).

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgb565_rgba_u16_forces_alpha_ffff() {
  let pix = solid_frame_le16(8, 2, 0xF800); // R=31, G=0, B=0
  let src = Rgb565Frame::try_new(&pix, 8, 2, 16).unwrap();
  let mut rgba_u16_out = std::vec![0u16; 8 * 2 * 4];
  let mut sink = MixedSinker::<Rgb565>::new(8, 2)
    .with_rgba_u16(&mut rgba_u16_out)
    .unwrap();
  rgb565_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  for px in rgba_u16_out.chunks(4) {
    assert_eq!(px[3], 0xFFFF, "RGB565 RGBA u16 alpha not 0xFFFF");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgb555_rgba_u16_forces_alpha_ffff() {
  let pix = solid_frame_le16(8, 2, 0x7C00); // R=31, G=0, B=0
  let src = Rgb555Frame::try_new(&pix, 8, 2, 16).unwrap();
  let mut rgba_u16_out = std::vec![0u16; 8 * 2 * 4];
  let mut sink = MixedSinker::<Rgb555>::new(8, 2)
    .with_rgba_u16(&mut rgba_u16_out)
    .unwrap();
  rgb555_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  for px in rgba_u16_out.chunks(4) {
    assert_eq!(px[3], 0xFFFF, "RGB555 RGBA u16 alpha not 0xFFFF");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgb444_rgba_u16_forces_alpha_ffff() {
  let pix = solid_frame_le16(8, 2, 0x0F00); // R=15, G=0, B=0
  let src = Rgb444Frame::try_new(&pix, 8, 2, 16).unwrap();
  let mut rgba_u16_out = std::vec![0u16; 8 * 2 * 4];
  let mut sink = MixedSinker::<Rgb444>::new(8, 2)
    .with_rgba_u16(&mut rgba_u16_out)
    .unwrap();
  rgb444_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  for px in rgba_u16_out.chunks(4) {
    assert_eq!(px[3], 0xFFFF, "RGB444 RGBA u16 alpha not 0xFFFF");
  }
}

// u16 native precision (no expansion).

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgb565_u16_native_precision_max() {
  // 0xFFFF → R5=31, G6=63, B5=31 (no expansion).
  let pix = solid_frame_le16(16, 4, 0xFFFF);
  let src = Rgb565Frame::try_new(&pix, 16, 4, 32).unwrap();
  let mut rgb_u16_out = std::vec![0u16; 16 * 4 * 3];
  let mut sink = MixedSinker::<Rgb565>::new(16, 4)
    .with_rgb_u16(&mut rgb_u16_out)
    .unwrap();
  rgb565_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  for px in rgb_u16_out.chunks(3) {
    assert_eq!(px, [31, 63, 31], "RGB565 u16 max pixel wrong");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgb555_u16_native_precision_max() {
  // 0x7FFF → R5=31, G5=31, B5=31 (bit 15 unused).
  let pix = solid_frame_le16(16, 4, 0x7FFF);
  let src = Rgb555Frame::try_new(&pix, 16, 4, 32).unwrap();
  let mut rgb_u16_out = std::vec![0u16; 16 * 4 * 3];
  let mut sink = MixedSinker::<Rgb555>::new(16, 4)
    .with_rgb_u16(&mut rgb_u16_out)
    .unwrap();
  rgb555_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  for px in rgb_u16_out.chunks(3) {
    assert_eq!(px, [31, 31, 31], "RGB555 u16 max pixel wrong");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgb444_u16_native_precision_max() {
  // 0x0FFF → R4=15, G4=15, B4=15 (bits [15:12] unused).
  let pix = solid_frame_le16(16, 4, 0x0FFF);
  let src = Rgb444Frame::try_new(&pix, 16, 4, 32).unwrap();
  let mut rgb_u16_out = std::vec![0u16; 16 * 4 * 3];
  let mut sink = MixedSinker::<Rgb444>::new(16, 4)
    .with_rgb_u16(&mut rgb_u16_out)
    .unwrap();
  rgb444_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  for px in rgb_u16_out.chunks(3) {
    assert_eq!(px, [15, 15, 15], "RGB444 u16 max pixel wrong");
  }
}

// Channel expansion correctness (known pixel values).

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgb565_known_pixel_pure_green() {
  // 0x07E0: R5=0, G6=63, B5=0 → R=0, G=(63<<2)|(63>>4)=252|3=255, B=0.
  let pix = solid_frame_le16(16, 1, 0x07E0);
  let src = Rgb565Frame::try_new(&pix, 16, 1, 32).unwrap();
  let mut rgb_out = std::vec![0u8; 16 * 3];
  let mut sink = MixedSinker::<Rgb565>::new(16, 1)
    .with_rgb(&mut rgb_out)
    .unwrap();
  rgb565_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  for px in rgb_out.chunks(3) {
    assert_eq!(px, [0, 255, 0], "RGB565 pure green pixel wrong");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgb565_known_pixel_pure_red() {
  // 0xF800: R5=31, G6=0, B5=0 → R=(31<<3)|(31>>2)=248|7=255, G=0, B=0.
  let pix = solid_frame_le16(16, 1, 0xF800);
  let src = Rgb565Frame::try_new(&pix, 16, 1, 32).unwrap();
  let mut rgb_out = std::vec![0u8; 16 * 3];
  let mut sink = MixedSinker::<Rgb565>::new(16, 1)
    .with_rgb(&mut rgb_out)
    .unwrap();
  rgb565_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  for px in rgb_out.chunks(3) {
    assert_eq!(px, [255, 0, 0], "RGB565 pure red pixel wrong");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgb565_known_pixel_all_zeros() {
  let pix = solid_frame_le16(16, 1, 0x0000);
  let src = Rgb565Frame::try_new(&pix, 16, 1, 32).unwrap();
  let mut rgb_out = std::vec![0u8; 16 * 3];
  let mut sink = MixedSinker::<Rgb565>::new(16, 1)
    .with_rgb(&mut rgb_out)
    .unwrap();
  rgb565_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  for px in rgb_out.chunks(3) {
    assert_eq!(px, [0, 0, 0], "RGB565 all-zeros pixel wrong");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgb565_known_pixel_all_ones_expands_to_255() {
  // 0xFFFF → R=255, G=255, B=255.
  let pix = solid_frame_le16(16, 1, 0xFFFF);
  let src = Rgb565Frame::try_new(&pix, 16, 1, 32).unwrap();
  let mut rgb_out = std::vec![0u8; 16 * 3];
  let mut sink = MixedSinker::<Rgb565>::new(16, 1)
    .with_rgb(&mut rgb_out)
    .unwrap();
  rgb565_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  for px in rgb_out.chunks(3) {
    assert_eq!(px, [255, 255, 255], "RGB565 all-ones pixel wrong");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgb555_known_pixel_all_ones_expands_to_255() {
  // 0x7FFF → R=255, G=255, B=255 (5-bit expansion for all three).
  let pix = solid_frame_le16(16, 1, 0x7FFF);
  let src = Rgb555Frame::try_new(&pix, 16, 1, 32).unwrap();
  let mut rgb_out = std::vec![0u8; 16 * 3];
  let mut sink = MixedSinker::<Rgb555>::new(16, 1)
    .with_rgb(&mut rgb_out)
    .unwrap();
  rgb555_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  for px in rgb_out.chunks(3) {
    assert_eq!(px, [255, 255, 255], "RGB555 all-ones pixel wrong");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgb444_known_pixel_all_ones_expands_to_255() {
  // 0x0FFF → R=255, G=255, B=255 (4-bit expansion: (15<<4)|15 = 255).
  let pix = solid_frame_le16(16, 1, 0x0FFF);
  let src = Rgb444Frame::try_new(&pix, 16, 1, 32).unwrap();
  let mut rgb_out = std::vec![0u8; 16 * 3];
  let mut sink = MixedSinker::<Rgb444>::new(16, 1)
    .with_rgb(&mut rgb_out)
    .unwrap();
  rgb444_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  for px in rgb_out.chunks(3) {
    assert_eq!(px, [255, 255, 255], "RGB444 all-ones pixel wrong");
  }
}

// BGR channel-order correctness.

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bgr565_channel_order_r_in_low_bits() {
  // BGR565: R is in bits [4:0]. Set only those bits → B=0, G=0, R=31.
  // Output (R, G, B byte order) should be (255, 0, 0).
  let pix = solid_frame_le16(16, 1, 0x001F); // B=0, G=0, R5=31
  let src = Bgr565Frame::try_new(&pix, 16, 1, 32).unwrap();
  let mut rgb_out = std::vec![0u8; 16 * 3];
  let mut sink = MixedSinker::<Bgr565>::new(16, 1)
    .with_rgb(&mut rgb_out)
    .unwrap();
  bgr565_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  for px in rgb_out.chunks(3) {
    assert_eq!(px, [255, 0, 0], "BGR565 R-channel order wrong");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bgr555_channel_order_r_in_low_bits() {
  // BGR555: R is in bits [4:0]. Set only R5=31 (0x001F) → output R=255.
  let pix = solid_frame_le16(16, 1, 0x001F);
  let src = Bgr555Frame::try_new(&pix, 16, 1, 32).unwrap();
  let mut rgb_out = std::vec![0u8; 16 * 3];
  let mut sink = MixedSinker::<Bgr555>::new(16, 1)
    .with_rgb(&mut rgb_out)
    .unwrap();
  bgr555_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  for px in rgb_out.chunks(3) {
    assert_eq!(px, [255, 0, 0], "BGR555 R-channel order wrong");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bgr444_channel_order_r_in_low_bits() {
  // BGR444: R is in bits [3:0]. Set only R4=15 (0x000F) → output R=255.
  let pix = solid_frame_le16(16, 1, 0x000F);
  let src = Bgr444Frame::try_new(&pix, 16, 1, 32).unwrap();
  let mut rgb_out = std::vec![0u8; 16 * 3];
  let mut sink = MixedSinker::<Bgr444>::new(16, 1)
    .with_rgb(&mut rgb_out)
    .unwrap();
  bgr444_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  for px in rgb_out.chunks(3) {
    assert_eq!(px, [255, 0, 0], "BGR444 R-channel order wrong");
  }
}

// luma_u16 is zero-extended u8 luma (range [0, 255]).

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgb565_luma_u16_matches_luma_u8_zero_extended() {
  let w = 64usize;
  let h = 4usize;
  let pix = random_legacy_frame(w as u32, h as u32, 0xBEEF_FEED, 0xFFFF);
  let src = Rgb565Frame::try_new(&pix, w as u32, h as u32, (w * 2) as u32).unwrap();

  let mut luma_u8 = std::vec![0u8; w * h];
  let mut luma_u16 = std::vec![0u16; w * h];

  let mut s = MixedSinker::<Rgb565>::new(w, h)
    .with_luma(&mut luma_u8)
    .unwrap()
    .with_luma_u16(&mut luma_u16)
    .unwrap();
  s.set_simd(false);
  rgb565_to(&src, true, KernelMatrix::Bt709, &mut s).unwrap();

  // Every luma_u16 value must equal the corresponding u8 zero-extended.
  let expected: std::vec::Vec<u16> = luma_u8.iter().map(|&y| y as u16).collect();
  assert_eq!(
    luma_u16, expected,
    "RGB565 luma_u16 not zero-extended u8 luma"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bgr565_luma_u16_matches_luma_u8_zero_extended() {
  let w = 64usize;
  let h = 4usize;
  let pix = random_legacy_frame(w as u32, h as u32, 0xF00D_CAFE, 0xFFFF);
  let src = Bgr565Frame::try_new(&pix, w as u32, h as u32, (w * 2) as u32).unwrap();

  let mut luma_u8 = std::vec![0u8; w * h];
  let mut luma_u16 = std::vec![0u16; w * h];

  let mut s = MixedSinker::<Bgr565>::new(w, h)
    .with_luma(&mut luma_u8)
    .unwrap()
    .with_luma_u16(&mut luma_u16)
    .unwrap();
  s.set_simd(false);
  bgr565_to(&src, true, KernelMatrix::Bt709, &mut s).unwrap();

  let expected: std::vec::Vec<u16> = luma_u8.iter().map(|&y| y as u16).collect();
  assert_eq!(
    luma_u16, expected,
    "BGR565 luma_u16 not zero-extended u8 luma"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgb444_luma_u16_matches_luma_u8_zero_extended() {
  let w = 64usize;
  let h = 4usize;
  let pix = random_legacy_frame(w as u32, h as u32, 0xA1B2_C3D4, 0x0FFF);
  let src = Rgb444Frame::try_new(&pix, w as u32, h as u32, (w * 2) as u32).unwrap();

  let mut luma_u8 = std::vec![0u8; w * h];
  let mut luma_u16 = std::vec![0u16; w * h];

  let mut s = MixedSinker::<Rgb444>::new(w, h)
    .with_luma(&mut luma_u8)
    .unwrap()
    .with_luma_u16(&mut luma_u16)
    .unwrap();
  s.set_simd(false);
  rgb444_to(&src, true, KernelMatrix::Bt709, &mut s).unwrap();

  let expected: std::vec::Vec<u16> = luma_u8.iter().map(|&y| y as u16).collect();
  assert_eq!(
    luma_u16, expected,
    "RGB444 luma_u16 not zero-extended u8 luma"
  );
}

// luma_u16 values stay in [0, 255].

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgb565_luma_u16_in_u8_range() {
  let w = 128usize;
  let h = 4usize;
  let pix = random_legacy_frame(w as u32, h as u32, 0x1234_5678, 0xFFFF);
  let src = Rgb565Frame::try_new(&pix, w as u32, h as u32, (w * 2) as u32).unwrap();

  let mut luma_u16 = std::vec![0u16; w * h];
  let mut sink = MixedSinker::<Rgb565>::new(w, h)
    .with_luma_u16(&mut luma_u16)
    .unwrap();
  sink.set_simd(false);
  rgb565_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();

  assert!(
    luma_u16.iter().all(|&y| y <= 255),
    "RGB565 luma_u16 out of [0,255] range"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgb555_luma_u16_in_u8_range() {
  let w = 128usize;
  let h = 4usize;
  let pix = random_legacy_frame(w as u32, h as u32, 0x8765_4321, 0x7FFF);
  let src = Rgb555Frame::try_new(&pix, w as u32, h as u32, (w * 2) as u32).unwrap();

  let mut luma_u16 = std::vec![0u16; w * h];
  let mut sink = MixedSinker::<Rgb555>::new(w, h)
    .with_luma_u16(&mut luma_u16)
    .unwrap();
  sink.set_simd(false);
  rgb555_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();

  assert!(
    luma_u16.iter().all(|&y| y <= 255),
    "RGB555 luma_u16 out of [0,255] range"
  );
}

// HSV output is non-trivial for non-gray content.

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgb565_hsv_pure_red_hue_near_zero() {
  // Pure red (R=255, G=0, B=0) → H near 0 (≤ 1 LSB from 0).
  let pix = solid_frame_le16(16, 1, 0xF800); // R5=31
  let src = Rgb565Frame::try_new(&pix, 16, 1, 32).unwrap();
  let mut h = std::vec![0u8; 16];
  let mut s = std::vec![0u8; 16];
  let mut v = std::vec![0u8; 16];
  let mut sink = MixedSinker::<Rgb565>::new(16, 1)
    .with_hsv(&mut h, &mut s, &mut v)
    .unwrap();
  rgb565_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  // Red: H=0, S=255, V=255.
  for (&hh, (&ss, &vv)) in h.iter().zip(s.iter().zip(v.iter())) {
    assert_eq!(hh, 0, "RGB565 red hue not 0");
    assert_eq!(ss, 255, "RGB565 red saturation not 255");
    assert_eq!(vv, 255, "RGB565 red value not 255");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgb555_hsv_pure_blue_hue() {
  // Pure blue (R=0, G=0, B=255) — RGB555 pure blue: B5=31 in bits [4:0] = 0x001F.
  // The crate's `rgb_to_hsv_row` maps H to [0, 255] over a 360° period with
  // 0/255=red, 85=green, (blue segment: ~120 in the 0-240/255 mapping). The
  // actual value produced by the scalar kernel is used here as the reference.
  let pix = solid_frame_le16(16, 1, 0x001F);
  let src = Rgb555Frame::try_new(&pix, 16, 1, 32).unwrap();
  let mut hh = std::vec![0u8; 16];
  let mut ss = std::vec![0u8; 16];
  let mut vv = std::vec![0u8; 16];
  let mut sink = MixedSinker::<Rgb555>::new(16, 1)
    .with_hsv(&mut hh, &mut ss, &mut vv)
    .unwrap();
  sink.set_simd(false);
  rgb555_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  // S=255, V=255 are unambiguous for pure blue; H is verified via simd=false
  // scalar reference (exact value depends on the crate's HSV formula).
  assert!(
    ss.iter().all(|&s| s == 255),
    "RGB555 blue saturation not 255"
  );
  assert!(vv.iter().all(|&v| v == 255), "RGB555 blue value not 255");
  // H is non-zero for blue; verify consistency — all pixels must agree.
  assert!(
    hh.windows(2).all(|w| w[0] == w[1]),
    "RGB555 blue hue inconsistent"
  );
}

// Walker round-trip (width=1 edge case).

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgb565_width_one_round_trip() {
  // Single-pixel frame: R=0, G=63, B=0 (pure green) = 0x07E0.
  let pix = solid_frame_le16(1, 1, 0x07E0);
  let src = Rgb565Frame::try_new(&pix, 1, 1, 2).unwrap();
  let mut rgb_out = std::vec![0u8; 3];
  let mut rgba_out = std::vec![0u8; 4];
  let mut sink = MixedSinker::<Rgb565>::new(1, 1)
    .with_rgb(&mut rgb_out)
    .unwrap()
    .with_rgba(&mut rgba_out)
    .unwrap();
  rgb565_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  assert_eq!(
    rgb_out.as_slice(),
    [0u8, 255, 0],
    "RGB565 width=1 RGB wrong"
  );
  assert_eq!(
    rgba_out.as_slice(),
    [0u8, 255, 0, 0xFF],
    "RGB565 width=1 RGBA wrong"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgb444_width_one_round_trip() {
  // Single-pixel: R4=15, G4=0, B4=0 = 0x0F00 → R=255, G=0, B=0.
  let pix = solid_frame_le16(1, 1, 0x0F00);
  let src = Rgb444Frame::try_new(&pix, 1, 1, 2).unwrap();
  let mut rgb_out = std::vec![0u8; 3];
  let mut sink = MixedSinker::<Rgb444>::new(1, 1)
    .with_rgb(&mut rgb_out)
    .unwrap();
  rgb444_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  assert_eq!(
    rgb_out.as_slice(),
    [255u8, 0, 0],
    "RGB444 width=1 RGB wrong"
  );
}

// Error paths: buffer too short.

#[test]
fn rgb565_rgba_buffer_too_short_returns_error() {
  let mut too_short = std::vec![0u8; 3]; // needs width*height*4
  let result = MixedSinker::<Rgb565>::new(4, 1).with_rgba(&mut too_short);
  assert!(
    matches!(result, Err(MixedSinkerError::InsufficientRgbaBuffer(_))),
    "Expected InsufficientRgbaBuffer"
  );
}

#[test]
fn rgb565_rgb_u16_buffer_too_short_returns_error() {
  let mut too_short = std::vec![0u16; 1]; // needs width*height*3
  let result = MixedSinker::<Rgb565>::new(4, 1).with_rgb_u16(&mut too_short);
  assert!(
    matches!(result, Err(MixedSinkerError::InsufficientRgbU16Buffer(_))),
    "Expected InsufficientRgbU16Buffer"
  );
}

#[test]
fn rgb565_rgba_u16_buffer_too_short_returns_error() {
  let mut too_short = std::vec![0u16; 1]; // needs width*height*4
  let result = MixedSinker::<Rgb565>::new(4, 1).with_rgba_u16(&mut too_short);
  assert!(
    matches!(result, Err(MixedSinkerError::InsufficientRgbaU16Buffer(_))),
    "Expected InsufficientRgbaU16Buffer"
  );
}

#[test]
fn rgb565_luma_u16_buffer_too_short_returns_error() {
  let mut too_short = std::vec![0u16; 0]; // needs width*height
  let result = MixedSinker::<Rgb565>::new(4, 1).with_luma_u16(&mut too_short);
  assert!(
    matches!(result, Err(MixedSinkerError::InsufficientLumaU16Buffer(_))),
    "Expected InsufficientLumaU16Buffer"
  );
}

#[test]
fn bgr565_rgba_buffer_too_short_returns_error() {
  let mut too_short = std::vec![0u8; 3];
  let result = MixedSinker::<Bgr565>::new(4, 1).with_rgba(&mut too_short);
  assert!(
    matches!(result, Err(MixedSinkerError::InsufficientRgbaBuffer(_))),
    "Expected InsufficientRgbaBuffer"
  );
}

#[test]
fn rgb555_rgba_buffer_too_short_returns_error() {
  let mut too_short = std::vec![0u8; 3];
  let result = MixedSinker::<Rgb555>::new(4, 1).with_rgba(&mut too_short);
  assert!(
    matches!(result, Err(MixedSinkerError::InsufficientRgbaBuffer(_))),
    "Expected InsufficientRgbaBuffer"
  );
}

#[test]
fn bgr444_rgba_buffer_too_short_returns_error() {
  let mut too_short = std::vec![0u8; 3];
  let result = MixedSinker::<Bgr444>::new(4, 1).with_rgba(&mut too_short);
  assert!(
    matches!(result, Err(MixedSinkerError::InsufficientRgbaBuffer(_))),
    "Expected InsufficientRgbaBuffer"
  );
}

// =========================================================================
// Legacy bit-packed RGB/BGR (8bpp 3:3:2 + 1:2:1; 4bpp 1:2:1) — end-to-end
// walker → MixedSinker known-value coverage (scalar-only direct path).
// =========================================================================

/// Drives every output channel of a byte-per-pixel legacy packed-RGB source
/// (`width` bytes/row) and asserts the deterministic RGB / RGBA / u16 outputs
/// plus the white/black luma + HSV-V endpoints. `expect_rgb` is the
/// `width * 3` packed-RGB reference; `expect_rgb_u16` is the native `u16`
/// reference (its alpha-extended form is checked too).
macro_rules! assert_byte_format {
  ($Frame:ident, $Marker:ident, $walker:ident, $bytes:expr, $expect_rgb:expr, $expect_rgb_u16:expr) => {{
    let bytes: std::vec::Vec<u8> = $bytes;
    let w = bytes.len();
    let frame = $Frame::try_new(&bytes, w as u32, 1, w as u32).unwrap();

    let mut rgb = std::vec![0u8; w * 3];
    let mut rgba = std::vec![0u8; w * 4];
    let mut rgb_u16 = std::vec![0u16; w * 3];
    let mut rgba_u16 = std::vec![0u16; w * 4];
    let mut luma = std::vec![0u8; w];
    let mut luma_u16 = std::vec![0u16; w];
    let mut hh = std::vec![0u8; w];
    let mut ss = std::vec![0u8; w];
    let mut vv = std::vec![0u8; w];

    let mut sink = MixedSinker::<$Marker>::new(w, 1)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_rgba(&mut rgba)
      .unwrap()
      .with_rgb_u16(&mut rgb_u16)
      .unwrap()
      .with_rgba_u16(&mut rgba_u16)
      .unwrap()
      .with_luma(&mut luma)
      .unwrap()
      .with_luma_u16(&mut luma_u16)
      .unwrap()
      .with_hsv(&mut hh, &mut ss, &mut vv)
      .unwrap();
    $walker(&frame, true, KernelMatrix::Bt709, &mut sink).unwrap();

    let exp_rgb: std::vec::Vec<u8> = $expect_rgb;
    let exp_rgb_u16: std::vec::Vec<u16> = $expect_rgb_u16;
    assert_eq!(rgb, exp_rgb, "rgb output mismatch");
    assert_eq!(rgb_u16, exp_rgb_u16, "rgb_u16 output mismatch");
    for x in 0..w {
      assert_eq!(rgba[x * 4], exp_rgb[x * 3], "rgba R[{x}]");
      assert_eq!(rgba[x * 4 + 1], exp_rgb[x * 3 + 1], "rgba G[{x}]");
      assert_eq!(rgba[x * 4 + 2], exp_rgb[x * 3 + 2], "rgba B[{x}]");
      assert_eq!(rgba[x * 4 + 3], 0xFF, "rgba alpha[{x}]");
      assert_eq!(rgba_u16[x * 4], exp_rgb_u16[x * 3], "rgba_u16 R[{x}]");
      assert_eq!(rgba_u16[x * 4 + 1], exp_rgb_u16[x * 3 + 1], "rgba_u16 G[{x}]");
      assert_eq!(rgba_u16[x * 4 + 2], exp_rgb_u16[x * 3 + 2], "rgba_u16 B[{x}]");
      assert_eq!(rgba_u16[x * 4 + 3], 0xFFFF, "rgba_u16 alpha[{x}]");
      // luma is the zero-extended u8 value.
      assert_eq!(luma_u16[x], u16::from(luma[x]), "luma_u16[{x}]");
    }
    (rgb, luma, vv, ss)
  }};
}

#[test]
fn rgb8_walker_known_values() {
  // px0 0xFF white; px1 0x00 black; px2 0xE0 red (R=7); px3 0x03 blue (B=3).
  let (_rgb, luma, vv, ss) = assert_byte_format!(
    Rgb8Frame,
    Rgb8,
    rgb8_to,
    std::vec![0xFF, 0x00, 0xE0, 0x03],
    std::vec![255, 255, 255, 0, 0, 0, 255, 0, 0, 0, 0, 255],
    std::vec![7, 7, 3, 0, 0, 0, 7, 0, 0, 0, 0, 3]
  );
  assert_eq!(luma[0], 255, "white luma");
  assert_eq!(luma[1], 0, "black luma");
  assert_eq!(vv[0], 255, "white HSV value");
  assert_eq!(ss[0], 0, "white HSV saturation");
  assert_eq!(vv[1], 0, "black HSV value");
}

#[test]
fn bgr8_walker_channel_order() {
  // px0 0x07 R=7 → red; px1 0xC0 B=3 → blue; px2 0x38 G=7 → green; px3 0xFF white.
  assert_byte_format!(
    Bgr8Frame,
    Bgr8,
    bgr8_to,
    std::vec![0x07, 0xC0, 0x38, 0xFF],
    std::vec![255, 0, 0, 0, 0, 255, 0, 255, 0, 255, 255, 255],
    std::vec![7, 0, 0, 0, 0, 3, 0, 7, 0, 7, 7, 3]
  );
}

#[test]
fn rgb4_byte_walker_known_values_high_nibble_ignored() {
  // px0 0x0F white(1,3,1); px1 0x00 black; px2 0xF8 R=1 (high nibble padding
  // must be ignored) → red; px3 0x01 B=1 → blue.
  assert_byte_format!(
    Rgb4ByteFrame,
    Rgb4Byte,
    rgb4_byte_to,
    std::vec![0x0F, 0x00, 0xF8, 0x01],
    std::vec![255, 255, 255, 0, 0, 0, 255, 0, 0, 0, 0, 255],
    std::vec![1, 3, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1]
  );
}

#[test]
fn bgr4_byte_walker_channel_order() {
  // px0 0x01 R=1 → red; px1 0x08 B=1 → blue; px2 0x06 G=3 → green; px3 0x0F white.
  assert_byte_format!(
    Bgr4ByteFrame,
    Bgr4Byte,
    bgr4_byte_to,
    std::vec![0x01, 0x08, 0x06, 0x0F],
    std::vec![255, 0, 0, 0, 0, 255, 0, 255, 0, 255, 255, 255],
    std::vec![1, 0, 0, 0, 0, 1, 0, 3, 0, 1, 3, 1]
  );
}

#[test]
fn rgb4_walker_two_pixels_per_byte() {
  // byte0 0xF0: hi=0xF white, lo=0x0 black; byte1 0x81: hi=0x8 R=1 red,
  // lo=0x1 B=1 blue. width = 4, two pixels per byte.
  let bytes = std::vec![0xF0u8, 0x81u8];
  let frame = Rgb4Frame::try_new(&bytes, 4, 1, 2).unwrap();
  let mut rgb = std::vec![0u8; 12];
  let mut rgb_u16 = std::vec![0u16; 12];
  let mut sink = MixedSinker::<Rgb4>::new(4, 1)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16)
    .unwrap();
  rgb4_to(&frame, true, KernelMatrix::Bt709, &mut sink).unwrap();
  assert_eq!(rgb, std::vec![255, 255, 255, 0, 0, 0, 255, 0, 0, 0, 0, 255]);
  assert_eq!(rgb_u16, std::vec![1, 3, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
}

#[test]
fn bgr4_walker_two_pixels_per_byte_channel_order() {
  // byte0 0xF0: hi=0xF white, lo=0x0 black; byte1 0x18: hi=0x1 R=1 red,
  // lo=0x8 B=1 blue.
  let bytes = std::vec![0xF0u8, 0x18u8];
  let frame = Bgr4Frame::try_new(&bytes, 4, 1, 2).unwrap();
  let mut rgba = std::vec![0u8; 16];
  let mut sink = MixedSinker::<Bgr4>::new(4, 1).with_rgba(&mut rgba).unwrap();
  bgr4_to(&frame, true, KernelMatrix::Bt709, &mut sink).unwrap();
  let expect = std::vec![
    255, 255, 255, 255, 0, 0, 0, 255, 255, 0, 0, 255, 0, 0, 255, 255,
  ];
  assert_eq!(rgba, expect);
}

#[test]
fn rgb8_rgba_only_fast_path() {
  // Standalone RGBA (no rgb/luma/hsv) exercises the no-staging branch.
  let bytes = std::vec![0xE0u8, 0x03u8];
  let frame = Rgb8Frame::try_new(&bytes, 2, 1, 2).unwrap();
  let mut rgba = std::vec![0u8; 8];
  let mut sink = MixedSinker::<Rgb8>::new(2, 1).with_rgba(&mut rgba).unwrap();
  rgb8_to(&frame, true, KernelMatrix::Bt709, &mut sink).unwrap();
  assert_eq!(rgba, std::vec![255, 0, 0, 0xFF, 0, 0, 255, 0xFF]);
}

#[test]
fn rgb8_rgba_buffer_too_short_returns_error() {
  let mut too_short = std::vec![0u8; 3];
  let result = MixedSinker::<Rgb8>::new(4, 1).with_rgba(&mut too_short);
  assert!(
    matches!(result, Err(MixedSinkerError::InsufficientRgbaBuffer(_))),
    "Expected InsufficientRgbaBuffer"
  );
}

#[test]
fn rgb4_row_shape_mismatch_returns_error() {
  // Manually drive `process` with a wrong-length row: width 4 expects
  // width.div_ceil(2) = 2 bytes; feed 3.
  use crate::{PixelSink, source::Rgb4Row};
  let mut rgb = std::vec![0u8; 12];
  let mut sink = MixedSinker::<Rgb4>::new(4, 1).with_rgb(&mut rgb).unwrap();
  PixelSink::begin_frame(&mut sink, 4, 1).unwrap();
  let bad = [0u8; 3];
  let row = Rgb4Row::new(&bad, 0, KernelMatrix::Bt709, true);
  let result = PixelSink::process(&mut sink, row);
  assert!(
    matches!(result, Err(MixedSinkerError::RowShapeMismatch(_))),
    "Expected RowShapeMismatch, got {result:?}"
  );
}
