use super::*;

// ---- Yuv420p12 ---------------------------------------------------------
//
// Planar 12-bit, low-bit-packed. Mirrors the Yuv420p10 shape — same
// planar layout, wider sample range. `mid-gray` for 12-bit is
// Y=UV=2048; native-depth white (full-range) is 4095.

fn solid_yuv420p12_frame(
  width: u32,
  height: u32,
  y: u16,
  u: u16,
  v: u16,
) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
  let w = width as usize;
  let h = height as usize;
  let cw = w / 2;
  let ch = h / 2;
  (
    std::vec![y; w * h],
    std::vec![u; cw * ch],
    std::vec![v; cw * ch],
  )
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p12_rgb_u8_only_gray_is_gray() {
  let (yp, up, vp) = solid_yuv420p12_frame(16, 8, 2048, 2048, 2048);
  let src = Yuv420p12Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yuv420p12>::new(16, 8)
    .with_rgb(&mut rgb)
    .unwrap();
  yuv420p12_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgb.chunks(3) {
    assert!(px[0].abs_diff(128) <= 1);
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p12_rgb_u16_only_native_depth_gray() {
  let (yp, up, vp) = solid_yuv420p12_frame(16, 8, 2048, 2048, 2048);
  let src = Yuv420p12Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgb = std::vec![0u16; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yuv420p12>::new(16, 8)
    .with_rgb_u16(&mut rgb)
    .unwrap();
  yuv420p12_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgb.chunks(3) {
    assert!(px[0].abs_diff(2048) <= 1, "got {px:?}");
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    // Upper 4 bits must be zero — 12-bit low-packed convention.
    assert!(px[0] <= 4095);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p12_rgb_u8_and_u16_both_populated() {
  // Full-range white: Y=4095, UV=2048.
  let (yp, up, vp) = solid_yuv420p12_frame(16, 8, 4095, 2048, 2048);
  let src = Yuv420p12Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgb_u8 = std::vec![0u8; 16 * 8 * 3];
  let mut rgb_u16 = std::vec![0u16; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yuv420p12>::new(16, 8)
    .with_rgb(&mut rgb_u8)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16)
    .unwrap();
  yuv420p12_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  assert!(rgb_u8.iter().all(|&c| c == 255));
  assert!(rgb_u16.iter().all(|&c| c == 4095));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p12_luma_downshifts_to_8bit() {
  // Y=2048 at 12 bits → 2048 >> (12 - 8) = 128 at 8 bits.
  let (yp, up, vp) = solid_yuv420p12_frame(16, 8, 2048, 2048, 2048);
  let src = Yuv420p12Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut luma = std::vec![0u8; 16 * 8];
  let mut sink = MixedSinker::<Yuv420p12>::new(16, 8)
    .with_luma(&mut luma)
    .unwrap();
  yuv420p12_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  assert!(luma.iter().all(|&l| l == 128));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p12_hsv_from_gray_is_zero_hue_zero_sat() {
  let (yp, up, vp) = solid_yuv420p12_frame(16, 8, 2048, 2048, 2048);
  let src = Yuv420p12Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut h = std::vec![0xFFu8; 16 * 8];
  let mut s = std::vec![0xFFu8; 16 * 8];
  let mut v = std::vec![0xFFu8; 16 * 8];
  let mut sink = MixedSinker::<Yuv420p12>::new(16, 8)
    .with_hsv(&mut h, &mut s, &mut v)
    .unwrap();
  yuv420p12_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  assert!(h.iter().all(|&b| b == 0));
  assert!(s.iter().all(|&b| b == 0));
  assert!(v.iter().all(|&b| b.abs_diff(128) <= 1));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p12_rgb_u16_too_short_returns_err() {
  let mut rgb = std::vec![0u16; 10];
  let err = MixedSinker::<Yuv420p12>::new(16, 8)
    .with_rgb_u16(&mut rgb)
    .err()
    .unwrap();
  assert!(matches!(err, MixedSinkerError::RgbU16BufferTooShort { .. }));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p12_with_simd_false_matches_with_simd_true() {
  let (yp, up, vp) = solid_yuv420p12_frame(64, 16, 2400, 1600, 2800);
  let src = Yuv420p12Frame::new(&yp, &up, &vp, 64, 16, 64, 32, 32);

  let mut rgb_scalar = std::vec![0u8; 64 * 16 * 3];
  let mut rgb_u16_scalar = std::vec![0u16; 64 * 16 * 3];
  let mut s_scalar = MixedSinker::<Yuv420p12>::new(64, 16)
    .with_simd(false)
    .with_rgb(&mut rgb_scalar)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16_scalar)
    .unwrap();
  yuv420p12_to(&src, false, ColorMatrix::Bt709, &mut s_scalar).unwrap();

  let mut rgb_simd = std::vec![0u8; 64 * 16 * 3];
  let mut rgb_u16_simd = std::vec![0u16; 64 * 16 * 3];
  let mut s_simd = MixedSinker::<Yuv420p12>::new(64, 16)
    .with_rgb(&mut rgb_simd)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16_simd)
    .unwrap();
  yuv420p12_to(&src, false, ColorMatrix::Bt709, &mut s_simd).unwrap();

  assert_eq!(rgb_scalar, rgb_simd);
  assert_eq!(rgb_u16_scalar, rgb_u16_simd);
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p12_rgba_u8_only_gray_with_opaque_alpha() {
  // 12-bit mid-gray (Y=U=V=2048) → 8-bit RGBA ≈ (128, 128, 128, 255).
  let (yp, up, vp) = solid_yuv420p12_frame(16, 8, 2048, 2048, 2048);
  let src = Yuv420p12Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuv420p12>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  yuv420p12_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(128) <= 1);
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert_eq!(px[3], 0xFF, "alpha must be opaque");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p12_rgba_u16_only_native_depth_gray_with_opaque_alpha() {
  // 12-bit mid-gray → u16 RGBA: each color element ≈ 2048, alpha = 4095.
  let (yp, up, vp) = solid_yuv420p12_frame(16, 8, 2048, 2048, 2048);
  let src = Yuv420p12Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgba = std::vec![0u16; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuv420p12>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  yuv420p12_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(2048) <= 1, "got {px:?}");
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert_eq!(px[3], 4095, "alpha must equal (1 << 12) - 1");
  }
}

// ---- Yuv420p14 ---------------------------------------------------------

fn solid_yuv420p14_frame(
  width: u32,
  height: u32,
  y: u16,
  u: u16,
  v: u16,
) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
  let w = width as usize;
  let h = height as usize;
  let cw = w / 2;
  let ch = h / 2;
  (
    std::vec![y; w * h],
    std::vec![u; cw * ch],
    std::vec![v; cw * ch],
  )
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p14_rgb_u8_only_gray_is_gray() {
  // 14-bit mid-gray: Y=UV=8192.
  let (yp, up, vp) = solid_yuv420p14_frame(16, 8, 8192, 8192, 8192);
  let src = Yuv420p14Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yuv420p14>::new(16, 8)
    .with_rgb(&mut rgb)
    .unwrap();
  yuv420p14_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgb.chunks(3) {
    assert!(px[0].abs_diff(128) <= 1);
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p14_rgb_u16_only_native_depth_gray() {
  let (yp, up, vp) = solid_yuv420p14_frame(16, 8, 8192, 8192, 8192);
  let src = Yuv420p14Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgb = std::vec![0u16; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yuv420p14>::new(16, 8)
    .with_rgb_u16(&mut rgb)
    .unwrap();
  yuv420p14_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgb.chunks(3) {
    assert!(px[0].abs_diff(8192) <= 1, "got {px:?}");
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert!(px[0] <= 16383);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p14_luma_downshifts_to_8bit() {
  // Y=8192 at 14 bits → 8192 >> (14 - 8) = 128.
  let (yp, up, vp) = solid_yuv420p14_frame(16, 8, 8192, 8192, 8192);
  let src = Yuv420p14Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut luma = std::vec![0u8; 16 * 8];
  let mut sink = MixedSinker::<Yuv420p14>::new(16, 8)
    .with_luma(&mut luma)
    .unwrap();
  yuv420p14_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  assert!(luma.iter().all(|&l| l == 128));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p14_rgb_u8_and_u16_both_populated() {
  let (yp, up, vp) = solid_yuv420p14_frame(16, 8, 16383, 8192, 8192);
  let src = Yuv420p14Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgb_u8 = std::vec![0u8; 16 * 8 * 3];
  let mut rgb_u16 = std::vec![0u16; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yuv420p14>::new(16, 8)
    .with_rgb(&mut rgb_u8)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16)
    .unwrap();
  yuv420p14_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  assert!(rgb_u8.iter().all(|&c| c == 255));
  assert!(rgb_u16.iter().all(|&c| c == 16383));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p14_with_simd_false_matches_with_simd_true() {
  let (yp, up, vp) = solid_yuv420p14_frame(64, 16, 9600, 6400, 11200);
  let src = Yuv420p14Frame::new(&yp, &up, &vp, 64, 16, 64, 32, 32);

  let mut rgb_scalar = std::vec![0u8; 64 * 16 * 3];
  let mut rgb_u16_scalar = std::vec![0u16; 64 * 16 * 3];
  let mut s_scalar = MixedSinker::<Yuv420p14>::new(64, 16)
    .with_simd(false)
    .with_rgb(&mut rgb_scalar)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16_scalar)
    .unwrap();
  yuv420p14_to(&src, false, ColorMatrix::Bt709, &mut s_scalar).unwrap();

  let mut rgb_simd = std::vec![0u8; 64 * 16 * 3];
  let mut rgb_u16_simd = std::vec![0u16; 64 * 16 * 3];
  let mut s_simd = MixedSinker::<Yuv420p14>::new(64, 16)
    .with_rgb(&mut rgb_simd)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16_simd)
    .unwrap();
  yuv420p14_to(&src, false, ColorMatrix::Bt709, &mut s_simd).unwrap();

  assert_eq!(rgb_scalar, rgb_simd);
  assert_eq!(rgb_u16_scalar, rgb_u16_simd);
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p14_rgba_u8_only_gray_with_opaque_alpha() {
  // 14-bit mid-gray (Y=U=V=8192) → 8-bit RGBA ≈ (128, 128, 128, 255).
  let (yp, up, vp) = solid_yuv420p14_frame(16, 8, 8192, 8192, 8192);
  let src = Yuv420p14Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuv420p14>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  yuv420p14_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(128) <= 1);
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert_eq!(px[3], 0xFF, "alpha must be opaque");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p14_rgba_u16_only_native_depth_gray_with_opaque_alpha() {
  // 14-bit mid-gray → u16 RGBA: each color element ≈ 8192, alpha = 16383.
  let (yp, up, vp) = solid_yuv420p14_frame(16, 8, 8192, 8192, 8192);
  let src = Yuv420p14Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgba = std::vec![0u16; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuv420p14>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  yuv420p14_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(8192) <= 1, "got {px:?}");
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert_eq!(px[3], 16383, "alpha must equal (1 << 14) - 1");
  }
}

// ---- P012 --------------------------------------------------------------
//
// Semi-planar 12-bit, high-bit-packed (samples in high 12 of each
// u16). Mirrors the P010 test shape — UV interleaved, `value << 4`.

fn solid_p012_frame(
  width: u32,
  height: u32,
  y_12bit: u16,
  u_12bit: u16,
  v_12bit: u16,
) -> (Vec<u16>, Vec<u16>) {
  let w = width as usize;
  let h = height as usize;
  let cw = w / 2;
  let ch = h / 2;
  // Shift into the high 12 bits (P012 packing).
  let y = std::vec![y_12bit << 4; w * h];
  let uv: Vec<u16> = (0..cw * ch)
    .flat_map(|_| [u_12bit << 4, v_12bit << 4])
    .collect();
  (y, uv)
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p012_rgb_u8_only_gray_is_gray() {
  let (yp, uvp) = solid_p012_frame(16, 8, 2048, 2048, 2048);
  let src = P012Frame::new(&yp, &uvp, 16, 8, 16, 16);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<P012>::new(16, 8).with_rgb(&mut rgb).unwrap();
  p012_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgb.chunks(3) {
    assert!(px[0].abs_diff(128) <= 1);
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p012_rgb_u16_only_native_depth_gray() {
  // Output is low-bit-packed 12-bit (yuv420p12le convention).
  let (yp, uvp) = solid_p012_frame(16, 8, 2048, 2048, 2048);
  let src = P012Frame::new(&yp, &uvp, 16, 8, 16, 16);

  let mut rgb = std::vec![0u16; 16 * 8 * 3];
  let mut sink = MixedSinker::<P012>::new(16, 8)
    .with_rgb_u16(&mut rgb)
    .unwrap();
  p012_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgb.chunks(3) {
    assert!(px[0].abs_diff(2048) <= 1, "got {px:?}");
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert!(
      px[0] <= 4095,
      "output must stay within 12-bit low-packed range"
    );
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p012_rgb_u8_and_u16_both_populated() {
  let (yp, uvp) = solid_p012_frame(16, 8, 4095, 2048, 2048);
  let src = P012Frame::new(&yp, &uvp, 16, 8, 16, 16);

  let mut rgb_u8 = std::vec![0u8; 16 * 8 * 3];
  let mut rgb_u16 = std::vec![0u16; 16 * 8 * 3];
  let mut sink = MixedSinker::<P012>::new(16, 8)
    .with_rgb(&mut rgb_u8)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16)
    .unwrap();
  p012_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  assert!(rgb_u8.iter().all(|&c| c == 255));
  assert!(rgb_u16.iter().all(|&c| c == 4095));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p012_luma_downshifts_to_8bit() {
  // Y=2048 at 12 bits, P012-packed (2048 << 4 = 0x8000). After >> 8,
  // the 8-bit luma is 0x80 = 128 — same accessor as P010 since both
  // store active bits in the high positions.
  let (yp, uvp) = solid_p012_frame(16, 8, 2048, 2048, 2048);
  let src = P012Frame::new(&yp, &uvp, 16, 8, 16, 16);

  let mut luma = std::vec![0u8; 16 * 8];
  let mut sink = MixedSinker::<P012>::new(16, 8)
    .with_luma(&mut luma)
    .unwrap();
  p012_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  assert!(luma.iter().all(|&l| l == 128));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p012_matches_yuv420p12_mixed_sinker_with_shifted_samples() {
  // Logical equivalence — same 12-bit samples fed through both
  // layouts must produce byte-identical u8 RGB.
  let w = 16u32;
  let h = 8u32;
  let y = 2400u16;
  let u = 1600u16;
  let v = 2800u16;

  let (yp_p12, up_p12, vp_p12) = solid_yuv420p12_frame(w, h, y, u, v);
  let src_p12 = Yuv420p12Frame::new(&yp_p12, &up_p12, &vp_p12, w, h, w, w / 2, w / 2);

  let (yp_p012, uvp_p012) = solid_p012_frame(w, h, y, u, v);
  let src_p012 = P012Frame::new(&yp_p012, &uvp_p012, w, h, w, w);

  let mut rgb_yuv = std::vec![0u8; (w * h * 3) as usize];
  let mut rgb_p012 = std::vec![0u8; (w * h * 3) as usize];
  let mut s_yuv = MixedSinker::<Yuv420p12>::new(w as usize, h as usize)
    .with_rgb(&mut rgb_yuv)
    .unwrap();
  let mut s_p012 = MixedSinker::<P012>::new(w as usize, h as usize)
    .with_rgb(&mut rgb_p012)
    .unwrap();
  yuv420p12_to(&src_p12, true, ColorMatrix::Bt709, &mut s_yuv).unwrap();
  p012_to(&src_p012, true, ColorMatrix::Bt709, &mut s_p012).unwrap();
  assert_eq!(rgb_yuv, rgb_p012);
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p012_rgb_u16_too_short_returns_err() {
  let mut rgb = std::vec![0u16; 10];
  let err = MixedSinker::<P012>::new(16, 8)
    .with_rgb_u16(&mut rgb)
    .err()
    .unwrap();
  assert!(matches!(err, MixedSinkerError::RgbU16BufferTooShort { .. }));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p012_with_simd_false_matches_with_simd_true() {
  let (yp, uvp) = solid_p012_frame(64, 16, 2400, 1600, 2800);
  let src = P012Frame::new(&yp, &uvp, 64, 16, 64, 64);

  let mut rgb_scalar = std::vec![0u8; 64 * 16 * 3];
  let mut rgb_u16_scalar = std::vec![0u16; 64 * 16 * 3];
  let mut s_scalar = MixedSinker::<P012>::new(64, 16)
    .with_simd(false)
    .with_rgb(&mut rgb_scalar)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16_scalar)
    .unwrap();
  p012_to(&src, false, ColorMatrix::Bt709, &mut s_scalar).unwrap();

  let mut rgb_simd = std::vec![0u8; 64 * 16 * 3];
  let mut rgb_u16_simd = std::vec![0u16; 64 * 16 * 3];
  let mut s_simd = MixedSinker::<P012>::new(64, 16)
    .with_rgb(&mut rgb_simd)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16_simd)
    .unwrap();
  p012_to(&src, false, ColorMatrix::Bt709, &mut s_simd).unwrap();

  assert_eq!(rgb_scalar, rgb_simd);
  assert_eq!(rgb_u16_scalar, rgb_u16_simd);
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p012_rgba_u8_only_gray_with_opaque_alpha() {
  // P012 mid-gray (12-bit values shifted into the high 12): Y/U/V = 2048 << 4.
  // Output 8-bit RGBA ≈ (128, 128, 128, 255).
  let (yp, uvp) = solid_p012_frame(16, 8, 2048, 2048, 2048);
  let src = P012Frame::new(&yp, &uvp, 16, 8, 16, 16);

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<P012>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  p012_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(128) <= 1);
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert_eq!(px[3], 0xFF, "alpha must be opaque");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p012_rgba_u16_only_native_depth_gray_with_opaque_alpha() {
  // P012 mid-gray → u16 RGBA: each color element ≈ 2048 (low-bit-packed),
  // alpha = (1 << 12) - 1 = 4095.
  let (yp, uvp) = solid_p012_frame(16, 8, 2048, 2048, 2048);
  let src = P012Frame::new(&yp, &uvp, 16, 8, 16, 16);

  let mut rgba = std::vec![0u16; 16 * 8 * 4];
  let mut sink = MixedSinker::<P012>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  p012_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(2048) <= 1, "got {px:?}");
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert_eq!(px[3], 4095, "alpha must equal (1 << 12) - 1");
  }
}
