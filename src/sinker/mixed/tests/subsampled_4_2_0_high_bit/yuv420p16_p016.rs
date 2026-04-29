use super::*;

// ---- Yuv420p16 ---------------------------------------------------------
//
// Planar 16-bit, full u16 range. Mid-gray is Y=UV=32768; full-range
// white luma is 65535.

pub(in crate::sinker::mixed::tests) fn solid_yuv420p16_frame(
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
fn yuv420p16_rgb_u8_only_gray_is_gray() {
  let (yp, up, vp) = solid_yuv420p16_frame(16, 8, 32768, 32768, 32768);
  let src = Yuv420p16Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yuv420p16>::new(16, 8)
    .with_rgb(&mut rgb)
    .unwrap();
  yuv420p16_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn yuv420p16_rgb_u16_only_native_depth_gray() {
  let (yp, up, vp) = solid_yuv420p16_frame(16, 8, 32768, 32768, 32768);
  let src = Yuv420p16Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgb = std::vec![0u16; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yuv420p16>::new(16, 8)
    .with_rgb_u16(&mut rgb)
    .unwrap();
  yuv420p16_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgb.chunks(3) {
    assert!(px[0].abs_diff(32768) <= 1, "got {px:?}");
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p16_rgb_u8_and_u16_both_populated() {
  // Full-range white: Y=65535, UV=32768.
  let (yp, up, vp) = solid_yuv420p16_frame(16, 8, 65535, 32768, 32768);
  let src = Yuv420p16Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgb_u8 = std::vec![0u8; 16 * 8 * 3];
  let mut rgb_u16 = std::vec![0u16; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yuv420p16>::new(16, 8)
    .with_rgb(&mut rgb_u8)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16)
    .unwrap();
  yuv420p16_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  assert!(rgb_u8.iter().all(|&c| c == 255));
  assert!(rgb_u16.iter().all(|&c| c == 65535));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p16_luma_downshifts_to_8bit() {
  // Y=32768 at 16 bits → 32768 >> (16 - 8) = 128.
  let (yp, up, vp) = solid_yuv420p16_frame(16, 8, 32768, 32768, 32768);
  let src = Yuv420p16Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut luma = std::vec![0u8; 16 * 8];
  let mut sink = MixedSinker::<Yuv420p16>::new(16, 8)
    .with_luma(&mut luma)
    .unwrap();
  yuv420p16_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  assert!(luma.iter().all(|&l| l == 128));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p16_hsv_from_gray_is_zero_hue_zero_sat() {
  let (yp, up, vp) = solid_yuv420p16_frame(16, 8, 32768, 32768, 32768);
  let src = Yuv420p16Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut h = std::vec![0xFFu8; 16 * 8];
  let mut s = std::vec![0xFFu8; 16 * 8];
  let mut v = std::vec![0xFFu8; 16 * 8];
  let mut sink = MixedSinker::<Yuv420p16>::new(16, 8)
    .with_hsv(&mut h, &mut s, &mut v)
    .unwrap();
  yuv420p16_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  assert!(h.iter().all(|&b| b == 0));
  assert!(s.iter().all(|&b| b == 0));
  assert!(v.iter().all(|&b| b.abs_diff(128) <= 1));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p16_rgb_u16_too_short_returns_err() {
  let mut rgb = std::vec![0u16; 10];
  let err = MixedSinker::<Yuv420p16>::new(16, 8)
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
fn yuv420p16_with_simd_false_matches_with_simd_true() {
  let (yp, up, vp) = solid_yuv420p16_frame(64, 16, 40000, 20000, 45000);
  let src = Yuv420p16Frame::new(&yp, &up, &vp, 64, 16, 64, 32, 32);

  let mut rgb_scalar = std::vec![0u8; 64 * 16 * 3];
  let mut rgb_u16_scalar = std::vec![0u16; 64 * 16 * 3];
  let mut s_scalar = MixedSinker::<Yuv420p16>::new(64, 16)
    .with_simd(false)
    .with_rgb(&mut rgb_scalar)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16_scalar)
    .unwrap();
  yuv420p16_to(&src, false, ColorMatrix::Bt709, &mut s_scalar).unwrap();

  let mut rgb_simd = std::vec![0u8; 64 * 16 * 3];
  let mut rgb_u16_simd = std::vec![0u16; 64 * 16 * 3];
  let mut s_simd = MixedSinker::<Yuv420p16>::new(64, 16)
    .with_rgb(&mut rgb_simd)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16_simd)
    .unwrap();
  yuv420p16_to(&src, false, ColorMatrix::Bt709, &mut s_simd).unwrap();

  assert_eq!(rgb_scalar, rgb_simd);
  assert_eq!(rgb_u16_scalar, rgb_u16_simd);
}

// ---- Yuv420p16 RGBA (Ship 8 Tranche 5b) -------------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p16_rgba_u16_only_native_depth_gray_with_opaque_alpha() {
  // 16-bit mid-gray: Y=UV=32768. Output u16 RGBA: each color element ≈
  // 32768, alpha = 0xFFFF.
  let (yp, up, vp) = solid_yuv420p16_frame(16, 8, 32768, 32768, 32768);
  let src = Yuv420p16Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgba = std::vec![0u16; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuv420p16>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  yuv420p16_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(32768) <= 8, "got {px:?}");
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert_eq!(px[3], 0xFFFF, "alpha must equal 0xFFFF");
  }
}

// ---- P016 --------------------------------------------------------------

fn solid_p016_frame(width: u32, height: u32, y: u16, u: u16, v: u16) -> (Vec<u16>, Vec<u16>) {
  let w = width as usize;
  let h = height as usize;
  let cw = w / 2;
  let ch = h / 2;
  // At 16 bits there's no shift — samples go in raw.
  let y_plane = std::vec![y; w * h];
  let uv: Vec<u16> = (0..cw * ch).flat_map(|_| [u, v]).collect();
  (y_plane, uv)
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p016_rgb_u8_only_gray_is_gray() {
  let (yp, uvp) = solid_p016_frame(16, 8, 32768, 32768, 32768);
  let src = P016Frame::new(&yp, &uvp, 16, 8, 16, 16);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<P016>::new(16, 8).with_rgb(&mut rgb).unwrap();
  p016_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn p016_rgb_u16_only_native_depth_gray() {
  let (yp, uvp) = solid_p016_frame(16, 8, 32768, 32768, 32768);
  let src = P016Frame::new(&yp, &uvp, 16, 8, 16, 16);

  let mut rgb = std::vec![0u16; 16 * 8 * 3];
  let mut sink = MixedSinker::<P016>::new(16, 8)
    .with_rgb_u16(&mut rgb)
    .unwrap();
  p016_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgb.chunks(3) {
    assert!(px[0].abs_diff(32768) <= 1, "got {px:?}");
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p016_rgb_u8_and_u16_both_populated() {
  let (yp, uvp) = solid_p016_frame(16, 8, 65535, 32768, 32768);
  let src = P016Frame::new(&yp, &uvp, 16, 8, 16, 16);

  let mut rgb_u8 = std::vec![0u8; 16 * 8 * 3];
  let mut rgb_u16 = std::vec![0u16; 16 * 8 * 3];
  let mut sink = MixedSinker::<P016>::new(16, 8)
    .with_rgb(&mut rgb_u8)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16)
    .unwrap();
  p016_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  assert!(rgb_u8.iter().all(|&c| c == 255));
  assert!(rgb_u16.iter().all(|&c| c == 65535));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p016_luma_downshifts_to_8bit() {
  let (yp, uvp) = solid_p016_frame(16, 8, 32768, 32768, 32768);
  let src = P016Frame::new(&yp, &uvp, 16, 8, 16, 16);

  let mut luma = std::vec![0u8; 16 * 8];
  let mut sink = MixedSinker::<P016>::new(16, 8)
    .with_luma(&mut luma)
    .unwrap();
  p016_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  assert!(luma.iter().all(|&l| l == 128));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p016_matches_yuv420p16_mixed_sinker() {
  // At 16 bits P016 and yuv420p16 are numerically identical —
  // the packing distinction degenerates when every bit is active.
  // Only the plane count / interleave layout differs.
  let w = 16u32;
  let h = 8u32;
  let y = 40000u16;
  let u = 20000u16;
  let v = 45000u16;

  let (yp_p16, up_p16, vp_p16) = solid_yuv420p16_frame(w, h, y, u, v);
  let src_p16 = Yuv420p16Frame::new(&yp_p16, &up_p16, &vp_p16, w, h, w, w / 2, w / 2);

  let (yp_p016, uvp_p016) = solid_p016_frame(w, h, y, u, v);
  let src_p016 = P016Frame::new(&yp_p016, &uvp_p016, w, h, w, w);

  let mut rgb_yuv = std::vec![0u8; (w * h * 3) as usize];
  let mut rgb_p016 = std::vec![0u8; (w * h * 3) as usize];
  let mut s_yuv = MixedSinker::<Yuv420p16>::new(w as usize, h as usize)
    .with_rgb(&mut rgb_yuv)
    .unwrap();
  let mut s_p016 = MixedSinker::<P016>::new(w as usize, h as usize)
    .with_rgb(&mut rgb_p016)
    .unwrap();
  yuv420p16_to(&src_p16, true, ColorMatrix::Bt709, &mut s_yuv).unwrap();
  p016_to(&src_p016, true, ColorMatrix::Bt709, &mut s_p016).unwrap();
  assert_eq!(rgb_yuv, rgb_p016);
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p016_rgb_u16_too_short_returns_err() {
  let mut rgb = std::vec![0u16; 10];
  let err = MixedSinker::<P016>::new(16, 8)
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
fn p016_with_simd_false_matches_with_simd_true() {
  let (yp, uvp) = solid_p016_frame(64, 16, 40000, 20000, 45000);
  let src = P016Frame::new(&yp, &uvp, 64, 16, 64, 64);

  let mut rgb_scalar = std::vec![0u8; 64 * 16 * 3];
  let mut rgb_u16_scalar = std::vec![0u16; 64 * 16 * 3];
  let mut s_scalar = MixedSinker::<P016>::new(64, 16)
    .with_simd(false)
    .with_rgb(&mut rgb_scalar)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16_scalar)
    .unwrap();
  p016_to(&src, false, ColorMatrix::Bt709, &mut s_scalar).unwrap();

  let mut rgb_simd = std::vec![0u8; 64 * 16 * 3];
  let mut rgb_u16_simd = std::vec![0u16; 64 * 16 * 3];
  let mut s_simd = MixedSinker::<P016>::new(64, 16)
    .with_rgb(&mut rgb_simd)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16_simd)
    .unwrap();
  p016_to(&src, false, ColorMatrix::Bt709, &mut s_simd).unwrap();

  assert_eq!(rgb_scalar, rgb_simd);
  assert_eq!(rgb_u16_scalar, rgb_u16_simd);
}
