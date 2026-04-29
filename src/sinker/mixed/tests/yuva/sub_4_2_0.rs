use super::{
  super::{
    subsampled_4_2_0_high_bit::{solid_yuv420p10_frame, solid_yuv420p16_frame},
    yuv420p_8bit::solid_yuv420p_frame,
  },
  *,
};

// ---- Yuva420p (8-bit) (Ship 8b‑2a) ---------------------------------

fn solid_yuva420p_frame(
  width: u32,
  height: u32,
  y: u8,
  u: u8,
  v: u8,
  a: u8,
) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
  let w = width as usize;
  let h = height as usize;
  let cw = w / 2;
  let ch = h / 2;
  (
    std::vec![y; w * h],
    std::vec![u; cw * ch],
    std::vec![v; cw * ch],
    std::vec![a; w * h],
  )
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva420p_rgba_u8_with_source_alpha_passes_through() {
  // 8-bit mid-gray with mid-alpha: Y=U=V=128, A=128.
  let (yp, up, vp, ap) = solid_yuva420p_frame(16, 8, 128, 128, 128, 128);
  let src = Yuva420pFrame::try_new(&yp, &up, &vp, &ap, 16, 8, 16, 8, 8, 16).unwrap();

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuva420p>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  yuva420p_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(128) <= 1, "got {px:?}");
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert_eq!(px[3], 128, "alpha must equal source A directly (no shift)");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva420p_rgba_u8_fully_opaque_alpha_yields_0xff() {
  let (yp, up, vp, ap) = solid_yuva420p_frame(16, 8, 128, 128, 128, 0xFF);
  let src = Yuva420pFrame::try_new(&yp, &up, &vp, &ap, 16, 8, 16, 8, 8, 16).unwrap();

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuva420p>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  yuva420p_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert_eq!(px[3], 0xFF);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva420p_rgba_u8_zero_alpha_yields_0() {
  let (yp, up, vp, ap) = solid_yuva420p_frame(16, 8, 128, 128, 128, 0);
  let src = Yuva420pFrame::try_new(&yp, &up, &vp, &ap, 16, 8, 16, 8, 8, 16).unwrap();

  let mut rgba = std::vec![0xFFu8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuva420p>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  yuva420p_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert_eq!(px[3], 0);
  }
}

#[test]
fn yuva420p_rgba_buf_too_short_returns_err() {
  let mut rgba = std::vec![0u8; 10];
  let err = MixedSinker::<Yuva420p>::new(16, 8)
    .with_rgba(&mut rgba)
    .err()
    .expect("expected RgbaBufferTooShort");
  assert!(matches!(err, MixedSinkerError::RgbaBufferTooShort { .. }));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva420p_with_rgb_alpha_drop_matches_yuv420p() {
  // alpha-drop path: with_rgb on Yuva420p must equal with_rgb on
  // Yuv420p given the same Y/U/V data. Codex PR #32 review fix #1
  // applied upfront here.
  let (yp, up, vp) = solid_yuv420p_frame(16, 8, 180, 60, 200);
  let (yp_a, up_a, vp_a, ap) = solid_yuva420p_frame(16, 8, 180, 60, 200, 128);

  let yuv = Yuv420pFrame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);
  let yuva = Yuva420pFrame::try_new(&yp_a, &up_a, &vp_a, &ap, 16, 8, 16, 8, 8, 16).unwrap();

  let mut rgb_yuv = std::vec![0u8; 16 * 8 * 3];
  let mut s_yuv = MixedSinker::<Yuv420p>::new(16, 8)
    .with_rgb(&mut rgb_yuv)
    .unwrap();
  yuv420p_to(&yuv, true, ColorMatrix::Bt709, &mut s_yuv).unwrap();

  let mut rgb_yuva = std::vec![0u8; 16 * 8 * 3];
  let mut s_yuva = MixedSinker::<Yuva420p>::new(16, 8)
    .with_rgb(&mut rgb_yuva)
    .unwrap();
  yuva420p_to(&yuva, true, ColorMatrix::Bt709, &mut s_yuva).unwrap();

  assert_eq!(rgb_yuv, rgb_yuva);
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva420p_with_rgb_and_with_rgba_combine() {
  // RGB triples in both buffers must match (alpha-drop + alpha
  // source forks per buffer in Strategy B).
  let (yp, up, vp, ap) = solid_yuva420p_frame(16, 8, 180, 60, 200, 200);
  let src = Yuva420pFrame::try_new(&yp, &up, &vp, &ap, 16, 8, 16, 8, 8, 16).unwrap();

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuva420p>::new(16, 8)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap();
  yuva420p_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

  for (rgb_px, rgba_px) in rgb.chunks(3).zip(rgba.chunks(4)) {
    assert_eq!(rgb_px[0], rgba_px[0]);
    assert_eq!(rgb_px[1], rgba_px[1]);
    assert_eq!(rgb_px[2], rgba_px[2]);
    assert_eq!(rgba_px[3], 200);
  }
}

// ---- Yuva420p9 (Ship 8b‑2a) ----------------------------------------

fn solid_yuva420p9_frame(
  width: u32,
  height: u32,
  y: u16,
  u: u16,
  v: u16,
  a: u16,
) -> (Vec<u16>, Vec<u16>, Vec<u16>, Vec<u16>) {
  let w = width as usize;
  let h = height as usize;
  let cw = w / 2;
  let ch = h / 2;
  (
    std::vec![y; w * h],
    std::vec![u; cw * ch],
    std::vec![v; cw * ch],
    std::vec![a; w * h],
  )
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva420p9_rgba_u8_with_source_alpha_passes_through() {
  // 9-bit mid-gray (Y=U=V=256) and mid-alpha (A=128 → u8 alpha = 128 >> 1 = 64).
  let (yp, up, vp, ap) = solid_yuva420p9_frame(16, 8, 256, 256, 256, 128);
  let src = Yuva420p9Frame::try_new(&yp, &up, &vp, &ap, 16, 8, 16, 8, 8, 16).unwrap();

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuva420p9>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  yuva420p9_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(128) <= 1, "got {px:?}");
    assert_eq!(px[3], 64, "alpha = 128 >> (9-8) = 64");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva420p9_rgba_u16_native_depth() {
  let (yp, up, vp, ap) = solid_yuva420p9_frame(16, 8, 256, 256, 256, 128);
  let src = Yuva420p9Frame::try_new(&yp, &up, &vp, &ap, 16, 8, 16, 8, 8, 16).unwrap();

  let mut rgba = std::vec![0u16; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuva420p9>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  yuva420p9_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert_eq!(px[3], 128, "alpha at native depth");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva420p9_rgba_fully_opaque_max() {
  let (yp, up, vp, ap) = solid_yuva420p9_frame(16, 8, 256, 256, 256, 511);
  let src = Yuva420p9Frame::try_new(&yp, &up, &vp, &ap, 16, 8, 16, 8, 8, 16).unwrap();

  let mut rgba_u8 = std::vec![0u8; 16 * 8 * 4];
  let mut s_u8 = MixedSinker::<Yuva420p9>::new(16, 8)
    .with_rgba(&mut rgba_u8)
    .unwrap();
  yuva420p9_to(&src, true, ColorMatrix::Bt601, &mut s_u8).unwrap();
  for px in rgba_u8.chunks(4) {
    assert_eq!(px[3], 0xFF, "511 >> 1 = 255");
  }

  let mut rgba_u16 = std::vec![0u16; 16 * 8 * 4];
  let mut s_u16 = MixedSinker::<Yuva420p9>::new(16, 8)
    .with_rgba_u16(&mut rgba_u16)
    .unwrap();
  yuva420p9_to(&src, true, ColorMatrix::Bt601, &mut s_u16).unwrap();
  for px in rgba_u16.chunks(4) {
    assert_eq!(px[3], 511);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva420p9_rgba_zero_alpha() {
  let (yp, up, vp, ap) = solid_yuva420p9_frame(16, 8, 256, 256, 256, 0);
  let src = Yuva420p9Frame::try_new(&yp, &up, &vp, &ap, 16, 8, 16, 8, 8, 16).unwrap();

  let mut rgba = std::vec![0xFFu8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuva420p9>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  yuva420p9_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();
  for px in rgba.chunks(4) {
    assert_eq!(px[3], 0);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva420p9_rgba_overrange_alpha_masked() {
  // alpha = 0x0500 (1280): masked to low 9 bits = 0x100 (256).
  // u8: 256 >> 1 = 128. u16: 256.
  let (yp, up, vp, ap) = solid_yuva420p9_frame(16, 8, 256, 256, 256, 0x0500);
  let src = Yuva420p9Frame::try_new(&yp, &up, &vp, &ap, 16, 8, 16, 8, 8, 16).unwrap();

  let mut rgba_u8 = std::vec![0u8; 16 * 8 * 4];
  let mut s_u8 = MixedSinker::<Yuva420p9>::new(16, 8)
    .with_rgba(&mut rgba_u8)
    .unwrap();
  yuva420p9_to(&src, true, ColorMatrix::Bt601, &mut s_u8).unwrap();
  for px in rgba_u8.chunks(4) {
    assert_eq!(px[3], 128, "0x0500 & 0x1FF = 256, 256 >> 1 = 128");
  }

  let mut rgba_u16 = std::vec![0u16; 16 * 8 * 4];
  let mut s_u16 = MixedSinker::<Yuva420p9>::new(16, 8)
    .with_rgba_u16(&mut rgba_u16)
    .unwrap();
  yuva420p9_to(&src, true, ColorMatrix::Bt601, &mut s_u16).unwrap();
  for px in rgba_u16.chunks(4) {
    assert_eq!(px[3], 256);
  }
}

#[test]
fn yuva420p9_rgba_buf_too_short_returns_err() {
  let mut rgba = std::vec![0u8; 10];
  let err = MixedSinker::<Yuva420p9>::new(16, 8)
    .with_rgba(&mut rgba)
    .err()
    .expect("expected RgbaBufferTooShort");
  assert!(matches!(err, MixedSinkerError::RgbaBufferTooShort { .. }));
}

#[test]
fn yuva420p9_rgba_u16_buf_too_short_returns_err() {
  let mut rgba = std::vec![0u16; 10];
  let err = MixedSinker::<Yuva420p9>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .err()
    .expect("expected RgbaU16BufferTooShort");
  assert!(matches!(
    err,
    MixedSinkerError::RgbaU16BufferTooShort { .. }
  ));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva420p9_with_rgb_alpha_drop_matches_yuv420p9() {
  let (yp, up, vp) = solid_yuv420p10_frame(16, 8, 256, 256, 256);
  let (yp_a, up_a, vp_a, ap) = solid_yuva420p9_frame(16, 8, 256, 256, 256, 128);

  let yuv = Yuv420p9Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);
  let yuva = Yuva420p9Frame::try_new(&yp_a, &up_a, &vp_a, &ap, 16, 8, 16, 8, 8, 16).unwrap();

  let mut rgb_yuv = std::vec![0u8; 16 * 8 * 3];
  let mut s_yuv = MixedSinker::<Yuv420p9>::new(16, 8)
    .with_rgb(&mut rgb_yuv)
    .unwrap();
  yuv420p9_to(&yuv, true, ColorMatrix::Bt709, &mut s_yuv).unwrap();

  let mut rgb_yuva = std::vec![0u8; 16 * 8 * 3];
  let mut s_yuva = MixedSinker::<Yuva420p9>::new(16, 8)
    .with_rgb(&mut rgb_yuva)
    .unwrap();
  yuva420p9_to(&yuva, true, ColorMatrix::Bt709, &mut s_yuva).unwrap();

  assert_eq!(rgb_yuv, rgb_yuva);
}

// ---- Yuva420p10 (Ship 8b‑2a) ---------------------------------------

fn solid_yuva420p10_frame(
  width: u32,
  height: u32,
  y: u16,
  u: u16,
  v: u16,
  a: u16,
) -> (Vec<u16>, Vec<u16>, Vec<u16>, Vec<u16>) {
  let w = width as usize;
  let h = height as usize;
  let cw = w / 2;
  let ch = h / 2;
  (
    std::vec![y; w * h],
    std::vec![u; cw * ch],
    std::vec![v; cw * ch],
    std::vec![a; w * h],
  )
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva420p10_rgba_u8_with_source_alpha_passes_through() {
  // 10-bit mid-gray (Y=U=V=512), mid-alpha A=256 → u8 alpha = 256 >> 2 = 64.
  let (yp, up, vp, ap) = solid_yuva420p10_frame(16, 8, 512, 512, 512, 256);
  let src = Yuva420p10Frame::try_new(&yp, &up, &vp, &ap, 16, 8, 16, 8, 8, 16).unwrap();

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuva420p10>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  yuva420p10_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(128) <= 1, "got {px:?}");
    assert_eq!(px[3], 64);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva420p10_rgba_u16_native_depth() {
  let (yp, up, vp, ap) = solid_yuva420p10_frame(16, 8, 512, 512, 512, 256);
  let src = Yuva420p10Frame::try_new(&yp, &up, &vp, &ap, 16, 8, 16, 8, 8, 16).unwrap();

  let mut rgba = std::vec![0u16; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuva420p10>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  yuva420p10_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert_eq!(px[3], 256);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva420p10_rgba_fully_opaque_max() {
  let (yp, up, vp, ap) = solid_yuva420p10_frame(16, 8, 512, 512, 512, 1023);
  let src = Yuva420p10Frame::try_new(&yp, &up, &vp, &ap, 16, 8, 16, 8, 8, 16).unwrap();

  let mut rgba_u8 = std::vec![0u8; 16 * 8 * 4];
  let mut s_u8 = MixedSinker::<Yuva420p10>::new(16, 8)
    .with_rgba(&mut rgba_u8)
    .unwrap();
  yuva420p10_to(&src, true, ColorMatrix::Bt601, &mut s_u8).unwrap();
  for px in rgba_u8.chunks(4) {
    assert_eq!(px[3], 0xFF);
  }

  let mut rgba_u16 = std::vec![0u16; 16 * 8 * 4];
  let mut s_u16 = MixedSinker::<Yuva420p10>::new(16, 8)
    .with_rgba_u16(&mut rgba_u16)
    .unwrap();
  yuva420p10_to(&src, true, ColorMatrix::Bt601, &mut s_u16).unwrap();
  for px in rgba_u16.chunks(4) {
    assert_eq!(px[3], 1023);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva420p10_rgba_zero_alpha() {
  let (yp, up, vp, ap) = solid_yuva420p10_frame(16, 8, 512, 512, 512, 0);
  let src = Yuva420p10Frame::try_new(&yp, &up, &vp, &ap, 16, 8, 16, 8, 8, 16).unwrap();

  let mut rgba = std::vec![0xFFu8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuva420p10>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  yuva420p10_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();
  for px in rgba.chunks(4) {
    assert_eq!(px[3], 0);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva420p10_rgba_overrange_alpha_masked() {
  // alpha = 0xFFFF: low 10 bits = 0x3FF (1023). u8: 1023 >> 2 = 255.
  let (yp, up, vp, ap) = solid_yuva420p10_frame(16, 8, 512, 512, 512, 0xFFFF);
  let src = Yuva420p10Frame::try_new(&yp, &up, &vp, &ap, 16, 8, 16, 8, 8, 16).unwrap();

  let mut rgba_u16 = std::vec![0u16; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuva420p10>::new(16, 8)
    .with_rgba_u16(&mut rgba_u16)
    .unwrap();
  yuva420p10_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();
  for px in rgba_u16.chunks(4) {
    assert_eq!(px[3], 1023, "0xFFFF & 0x3FF = 1023");
  }
}

#[test]
fn yuva420p10_rgba_buf_too_short_returns_err() {
  let mut rgba = std::vec![0u8; 10];
  let err = MixedSinker::<Yuva420p10>::new(16, 8)
    .with_rgba(&mut rgba)
    .err()
    .expect("expected RgbaBufferTooShort");
  assert!(matches!(err, MixedSinkerError::RgbaBufferTooShort { .. }));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva420p10_with_rgb_alpha_drop_matches_yuv420p10() {
  let (yp, up, vp) = solid_yuv420p10_frame(16, 8, 600, 400, 700);
  let (yp_a, up_a, vp_a, ap) = solid_yuva420p10_frame(16, 8, 600, 400, 700, 256);

  let yuv = Yuv420p10Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);
  let yuva = Yuva420p10Frame::try_new(&yp_a, &up_a, &vp_a, &ap, 16, 8, 16, 8, 8, 16).unwrap();

  let mut rgb_yuv = std::vec![0u8; 16 * 8 * 3];
  let mut s_yuv = MixedSinker::<Yuv420p10>::new(16, 8)
    .with_rgb(&mut rgb_yuv)
    .unwrap();
  yuv420p10_to(&yuv, true, ColorMatrix::Bt709, &mut s_yuv).unwrap();

  let mut rgb_yuva = std::vec![0u8; 16 * 8 * 3];
  let mut s_yuva = MixedSinker::<Yuva420p10>::new(16, 8)
    .with_rgb(&mut rgb_yuva)
    .unwrap();
  yuva420p10_to(&yuva, true, ColorMatrix::Bt709, &mut s_yuva).unwrap();

  assert_eq!(rgb_yuv, rgb_yuva);
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva420p10_with_rgb_and_with_rgba_combine() {
  let (yp, up, vp, ap) = solid_yuva420p10_frame(16, 8, 600, 400, 700, 512);
  let src = Yuva420p10Frame::try_new(&yp, &up, &vp, &ap, 16, 8, 16, 8, 8, 16).unwrap();

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuva420p10>::new(16, 8)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap();
  yuva420p10_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

  for (rgb_px, rgba_px) in rgb.chunks(3).zip(rgba.chunks(4)) {
    assert_eq!(rgb_px[0], rgba_px[0]);
    assert_eq!(rgb_px[1], rgba_px[1]);
    assert_eq!(rgb_px[2], rgba_px[2]);
    assert_eq!(rgba_px[3], 128, "(512 >> 2) = 128");
  }
}

// ---- Yuva420p16 (Ship 8b‑2a) ---------------------------------------

fn solid_yuva420p16_frame(
  width: u32,
  height: u32,
  y: u16,
  u: u16,
  v: u16,
  a: u16,
) -> (Vec<u16>, Vec<u16>, Vec<u16>, Vec<u16>) {
  let w = width as usize;
  let h = height as usize;
  let cw = w / 2;
  let ch = h / 2;
  (
    std::vec![y; w * h],
    std::vec![u; cw * ch],
    std::vec![v; cw * ch],
    std::vec![a; w * h],
  )
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva420p16_rgba_u8_with_source_alpha_passes_through() {
  // 16-bit mid-gray (Y=U=V=0x8000), mid-alpha A=0x8000 → u8 alpha = 0x80.
  let (yp, up, vp, ap) = solid_yuva420p16_frame(16, 8, 0x8000, 0x8000, 0x8000, 0x8000);
  let src = Yuva420p16Frame::try_new(&yp, &up, &vp, &ap, 16, 8, 16, 8, 8, 16).unwrap();

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuva420p16>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  yuva420p16_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(0x80) <= 1, "got {px:?}");
    assert_eq!(px[3], 0x80, "alpha = 0x8000 >> 8 = 0x80");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva420p16_rgba_u16_native_depth() {
  let (yp, up, vp, ap) = solid_yuva420p16_frame(16, 8, 0x8000, 0x8000, 0x8000, 0x8000);
  let src = Yuva420p16Frame::try_new(&yp, &up, &vp, &ap, 16, 8, 16, 8, 8, 16).unwrap();

  let mut rgba = std::vec![0u16; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuva420p16>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  yuva420p16_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert_eq!(px[3], 0x8000, "alpha at native u16 depth (no shift)");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva420p16_rgba_fully_opaque_max() {
  let (yp, up, vp, ap) = solid_yuva420p16_frame(16, 8, 0x8000, 0x8000, 0x8000, 0xFFFF);
  let src = Yuva420p16Frame::try_new(&yp, &up, &vp, &ap, 16, 8, 16, 8, 8, 16).unwrap();

  let mut rgba_u8 = std::vec![0u8; 16 * 8 * 4];
  let mut s_u8 = MixedSinker::<Yuva420p16>::new(16, 8)
    .with_rgba(&mut rgba_u8)
    .unwrap();
  yuva420p16_to(&src, true, ColorMatrix::Bt601, &mut s_u8).unwrap();
  for px in rgba_u8.chunks(4) {
    assert_eq!(px[3], 0xFF);
  }

  let mut rgba_u16 = std::vec![0u16; 16 * 8 * 4];
  let mut s_u16 = MixedSinker::<Yuva420p16>::new(16, 8)
    .with_rgba_u16(&mut rgba_u16)
    .unwrap();
  yuva420p16_to(&src, true, ColorMatrix::Bt601, &mut s_u16).unwrap();
  for px in rgba_u16.chunks(4) {
    assert_eq!(px[3], 0xFFFF);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva420p16_rgba_zero_alpha() {
  let (yp, up, vp, ap) = solid_yuva420p16_frame(16, 8, 0x8000, 0x8000, 0x8000, 0);
  let src = Yuva420p16Frame::try_new(&yp, &up, &vp, &ap, 16, 8, 16, 8, 8, 16).unwrap();

  let mut rgba = std::vec![0xFFu8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuva420p16>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  yuva420p16_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();
  for px in rgba.chunks(4) {
    assert_eq!(px[3], 0);
  }
}

#[test]
fn yuva420p16_rgba_buf_too_short_returns_err() {
  let mut rgba = std::vec![0u8; 10];
  let err = MixedSinker::<Yuva420p16>::new(16, 8)
    .with_rgba(&mut rgba)
    .err()
    .expect("expected RgbaBufferTooShort");
  assert!(matches!(err, MixedSinkerError::RgbaBufferTooShort { .. }));
}

#[test]
fn yuva420p16_rgba_u16_buf_too_short_returns_err() {
  let mut rgba = std::vec![0u16; 10];
  let err = MixedSinker::<Yuva420p16>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .err()
    .expect("expected RgbaU16BufferTooShort");
  assert!(matches!(
    err,
    MixedSinkerError::RgbaU16BufferTooShort { .. }
  ));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva420p16_with_rgb_alpha_drop_matches_yuv420p16() {
  let (yp, up, vp) = solid_yuv420p16_frame(16, 8, 0x8000, 0x4000, 0xC000);
  let (yp_a, up_a, vp_a, ap) = solid_yuva420p16_frame(16, 8, 0x8000, 0x4000, 0xC000, 0x8000);

  let yuv = Yuv420p16Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);
  let yuva = Yuva420p16Frame::try_new(&yp_a, &up_a, &vp_a, &ap, 16, 8, 16, 8, 8, 16).unwrap();

  let mut rgb_yuv = std::vec![0u8; 16 * 8 * 3];
  let mut s_yuv = MixedSinker::<Yuv420p16>::new(16, 8)
    .with_rgb(&mut rgb_yuv)
    .unwrap();
  yuv420p16_to(&yuv, true, ColorMatrix::Bt709, &mut s_yuv).unwrap();

  let mut rgb_yuva = std::vec![0u8; 16 * 8 * 3];
  let mut s_yuva = MixedSinker::<Yuva420p16>::new(16, 8)
    .with_rgb(&mut rgb_yuva)
    .unwrap();
  yuva420p16_to(&yuva, true, ColorMatrix::Bt709, &mut s_yuva).unwrap();

  assert_eq!(rgb_yuv, rgb_yuva);
}
