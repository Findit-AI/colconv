use super::{yuv420p_8bit::solid_yuv420p_frame, *};

// ---- Ship 6: sanity tests for new 4:2:2 / 4:4:4 formats ---------------

pub(super) fn solid_yuv422p_frame(
  width: u32,
  height: u32,
  y: u8,
  u: u8,
  v: u8,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let w = width as usize;
  let h = height as usize;
  let cw = w / 2;
  // 4:2:2: chroma is half-width, FULL-height.
  (
    std::vec![y; w * h],
    std::vec![u; cw * h],
    std::vec![v; cw * h],
  )
}

pub(super) fn solid_yuv444p_frame(
  width: u32,
  height: u32,
  y: u8,
  u: u8,
  v: u8,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let w = width as usize;
  let h = height as usize;
  (
    std::vec![y; w * h],
    std::vec![u; w * h],
    std::vec![v; w * h],
  )
}

pub(super) fn solid_yuv422p_n_frame(
  width: u32,
  height: u32,
  y: u16,
  u: u16,
  v: u16,
) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
  let w = width as usize;
  let h = height as usize;
  let cw = w / 2;
  (
    std::vec![y; w * h],
    std::vec![u; cw * h],
    std::vec![v; cw * h],
  )
}

pub(super) fn solid_yuv444p_n_frame(
  width: u32,
  height: u32,
  y: u16,
  u: u16,
  v: u16,
) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
  let w = width as usize;
  let h = height as usize;
  (
    std::vec![y; w * h],
    std::vec![u; w * h],
    std::vec![v; w * h],
  )
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv422p_gray_to_gray() {
  let (yp, up, vp) = solid_yuv422p_frame(16, 8, 128, 128, 128);
  let src = Yuv422pFrame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yuv422p>::new(16, 8)
    .with_rgb(&mut rgb)
    .unwrap();
  yuv422p_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn yuv444p_gray_to_gray() {
  let (yp, up, vp) = solid_yuv444p_frame(16, 8, 128, 128, 128);
  let src = Yuv444pFrame::new(&yp, &up, &vp, 16, 8, 16, 16, 16);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yuv444p>::new(16, 8)
    .with_rgb(&mut rgb)
    .unwrap();
  yuv444p_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgb.chunks(3) {
    assert!(px[0].abs_diff(128) <= 1);
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
  }
}

// ---- Yuv444p RGBA (Ship 8 PR 4a) tests ----------------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv444p_rgba_only_converts_gray_to_gray_with_opaque_alpha() {
  let (yp, up, vp) = solid_yuv444p_frame(16, 8, 128, 128, 128);
  let src = Yuv444pFrame::new(&yp, &up, &vp, 16, 8, 16, 16, 16);

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuv444p>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  yuv444p_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(128) <= 1, "R");
    assert_eq!(px[0], px[1], "RGB monochromatic");
    assert_eq!(px[1], px[2], "RGB monochromatic");
    assert_eq!(px[3], 0xFF, "alpha must default to opaque");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv444p_with_rgb_and_with_rgba_produce_byte_identical_rgb_bytes() {
  let w = 32u32;
  let h = 16u32;
  let ws = w as usize;
  let hs = h as usize;
  let (yp, up, vp) = solid_yuv444p_frame(w, h, 180, 60, 200);
  let src = Yuv444pFrame::new(&yp, &up, &vp, w, h, w, w, w);

  let mut rgb = std::vec![0u8; ws * hs * 3];
  let mut rgba = std::vec![0u8; ws * hs * 4];
  let mut sink = MixedSinker::<Yuv444p>::new(ws, hs)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap();
  yuv444p_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for i in 0..(ws * hs) {
    assert_eq!(rgba[i * 4], rgb[i * 3], "R differs at pixel {i}");
    assert_eq!(rgba[i * 4 + 1], rgb[i * 3 + 1], "G differs at pixel {i}");
    assert_eq!(rgba[i * 4 + 2], rgb[i * 3 + 2], "B differs at pixel {i}");
    assert_eq!(rgba[i * 4 + 3], 0xFF, "A not opaque at pixel {i}");
  }
}

#[test]
fn yuv444p_rgba_buffer_too_short_returns_err() {
  let mut rgba_short = std::vec![0u8; 16 * 8 * 4 - 1];
  let result = MixedSinker::<Yuv444p>::new(16, 8).with_rgba(&mut rgba_short);
  let Err(err) = result else {
    panic!("expected RgbaBufferTooShort error");
  };
  assert!(matches!(
    err,
    MixedSinkerError::RgbaBufferTooShort {
      expected: 512,
      actual: 511,
    }
  ));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv444p_rgba_simd_matches_scalar_with_random_yuv() {
  // 4:4:4 has full-width chroma — U / V are width-sized per row.
  // Width 1922 forces both the SIMD main loop AND scalar tail
  // across every backend block size (16/32/64).
  let w = 1922usize;
  let h = 4usize;
  let mut yp = std::vec![0u8; w * h];
  let mut up = std::vec![0u8; w * h];
  let mut vp = std::vec![0u8; w * h];
  pseudo_random_u8(&mut yp, 0xC001_C0DE);
  pseudo_random_u8(&mut up, 0xCAFE_F00D);
  pseudo_random_u8(&mut vp, 0xDEAD_BEEF);
  let src = Yuv444pFrame::new(
    &yp, &up, &vp, w as u32, h as u32, w as u32, w as u32, w as u32,
  );

  for &matrix in &[
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::YCgCo,
  ] {
    for &full_range in &[true, false] {
      let mut rgba_simd = std::vec![0u8; w * h * 4];
      let mut rgba_scalar = std::vec![0u8; w * h * 4];

      let mut s_simd = MixedSinker::<Yuv444p>::new(w, h)
        .with_rgba(&mut rgba_simd)
        .unwrap();
      yuv444p_to(&src, full_range, matrix, &mut s_simd).unwrap();

      let mut s_scalar = MixedSinker::<Yuv444p>::new(w, h)
        .with_rgba(&mut rgba_scalar)
        .unwrap();
      s_scalar.set_simd(false);
      yuv444p_to(&src, full_range, matrix, &mut s_scalar).unwrap();

      if rgba_simd != rgba_scalar {
        let mismatch = rgba_simd
          .iter()
          .zip(rgba_scalar.iter())
          .position(|(a, b)| a != b)
          .unwrap();
        let pixel = mismatch / 4;
        let channel = ["R", "G", "B", "A"][mismatch % 4];
        panic!(
          "Yuv444p RGBA SIMD ≠ scalar at byte {mismatch} (px {pixel} {channel}) for matrix={matrix:?} full_range={full_range}: simd={} scalar={}",
          rgba_simd[mismatch], rgba_scalar[mismatch]
        );
      }
    }
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv422p10_gray_to_gray() {
  let (yp, up, vp) = solid_yuv422p_n_frame(16, 8, 512, 512, 512);
  let src = Yuv422p10Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yuv422p10>::new(16, 8)
    .with_rgb(&mut rgb)
    .unwrap();
  yuv422p10_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn yuv422p12_gray_to_gray() {
  let (yp, up, vp) = solid_yuv422p_n_frame(16, 8, 2048, 2048, 2048);
  let src = Yuv422p12Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yuv422p12>::new(16, 8)
    .with_rgb(&mut rgb)
    .unwrap();
  yuv422p12_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn yuv422p12_rgba_u8_only_gray_with_opaque_alpha() {
  // 12-bit mid-gray → 8-bit RGBA ≈ (128, 128, 128, 255).
  let (yp, up, vp) = solid_yuv422p_n_frame(16, 8, 2048, 2048, 2048);
  let src = Yuv422p12Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuv422p12>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  yuv422p12_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn yuv422p12_rgba_u16_only_native_depth_gray_with_opaque_alpha() {
  // 12-bit mid-gray → u16 RGBA: each color element ≈ 2048, alpha = 4095.
  let (yp, up, vp) = solid_yuv422p_n_frame(16, 8, 2048, 2048, 2048);
  let src = Yuv422p12Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgba = std::vec![0u16; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuv422p12>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  yuv422p12_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(2048) <= 1, "got {px:?}");
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert_eq!(px[3], 4095, "alpha must equal (1 << 12) - 1");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv422p14_gray_to_gray() {
  let (yp, up, vp) = solid_yuv422p_n_frame(16, 8, 8192, 8192, 8192);
  let src = Yuv422p14Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yuv422p14>::new(16, 8)
    .with_rgb(&mut rgb)
    .unwrap();
  yuv422p14_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn yuv422p14_rgba_u8_only_gray_with_opaque_alpha() {
  // 14-bit mid-gray → 8-bit RGBA ≈ (128, 128, 128, 255).
  let (yp, up, vp) = solid_yuv422p_n_frame(16, 8, 8192, 8192, 8192);
  let src = Yuv422p14Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuv422p14>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  yuv422p14_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn yuv422p14_rgba_u16_only_native_depth_gray_with_opaque_alpha() {
  // 14-bit mid-gray → u16 RGBA: each color element ≈ 8192, alpha = 16383.
  let (yp, up, vp) = solid_yuv422p_n_frame(16, 8, 8192, 8192, 8192);
  let src = Yuv422p14Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgba = std::vec![0u16; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuv422p14>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  yuv422p14_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(8192) <= 1, "got {px:?}");
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert_eq!(px[3], 16383, "alpha must equal (1 << 14) - 1");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv422p16_gray_to_gray_u16() {
  let (yp, up, vp) = solid_yuv422p_n_frame(16, 8, 32768, 32768, 32768);
  let src = Yuv422p16Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgb_u8 = std::vec![0u8; 16 * 8 * 3];
  let mut rgb_u16 = std::vec![0u16; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yuv422p16>::new(16, 8)
    .with_rgb(&mut rgb_u8)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16)
    .unwrap();
  yuv422p16_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgb_u8.chunks(3) {
    assert!(px[0].abs_diff(128) <= 1);
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
  }
  for px in rgb_u16.chunks(3) {
    assert!(px[0].abs_diff(32768) <= 256);
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv444p10_gray_to_gray() {
  let (yp, up, vp) = solid_yuv444p_n_frame(16, 8, 512, 512, 512);
  let src = Yuv444p10Frame::new(&yp, &up, &vp, 16, 8, 16, 16, 16);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yuv444p10>::new(16, 8)
    .with_rgb(&mut rgb)
    .unwrap();
  yuv444p10_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn yuv444p12_gray_to_gray() {
  let (yp, up, vp) = solid_yuv444p_n_frame(16, 8, 2048, 2048, 2048);
  let src = Yuv444p12Frame::new(&yp, &up, &vp, 16, 8, 16, 16, 16);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yuv444p12>::new(16, 8)
    .with_rgb(&mut rgb)
    .unwrap();
  yuv444p12_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn yuv444p12_rgba_u8_only_gray_with_opaque_alpha() {
  // 12-bit mid-gray → 8-bit RGBA ≈ (128, 128, 128, 255).
  let (yp, up, vp) = solid_yuv444p_n_frame(16, 8, 2048, 2048, 2048);
  let src = Yuv444p12Frame::new(&yp, &up, &vp, 16, 8, 16, 16, 16);

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuv444p12>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  yuv444p12_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn yuv444p12_rgba_u16_only_native_depth_gray_with_opaque_alpha() {
  // 12-bit mid-gray → u16 RGBA: each color element ≈ 2048, alpha = 4095.
  let (yp, up, vp) = solid_yuv444p_n_frame(16, 8, 2048, 2048, 2048);
  let src = Yuv444p12Frame::new(&yp, &up, &vp, 16, 8, 16, 16, 16);

  let mut rgba = std::vec![0u16; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuv444p12>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  yuv444p12_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(2048) <= 1, "got {px:?}");
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert_eq!(px[3], 4095, "alpha must equal (1 << 12) - 1");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv444p14_gray_to_gray() {
  let (yp, up, vp) = solid_yuv444p_n_frame(16, 8, 8192, 8192, 8192);
  let src = Yuv444p14Frame::new(&yp, &up, &vp, 16, 8, 16, 16, 16);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yuv444p14>::new(16, 8)
    .with_rgb(&mut rgb)
    .unwrap();
  yuv444p14_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn yuv444p14_rgba_u8_only_gray_with_opaque_alpha() {
  // 14-bit mid-gray → 8-bit RGBA ≈ (128, 128, 128, 255).
  let (yp, up, vp) = solid_yuv444p_n_frame(16, 8, 8192, 8192, 8192);
  let src = Yuv444p14Frame::new(&yp, &up, &vp, 16, 8, 16, 16, 16);

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuv444p14>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  yuv444p14_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn yuv444p14_rgba_u16_only_native_depth_gray_with_opaque_alpha() {
  // 14-bit mid-gray → u16 RGBA: each color element ≈ 8192, alpha = 16383.
  let (yp, up, vp) = solid_yuv444p_n_frame(16, 8, 8192, 8192, 8192);
  let src = Yuv444p14Frame::new(&yp, &up, &vp, 16, 8, 16, 16, 16);

  let mut rgba = std::vec![0u16; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuv444p14>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  yuv444p14_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(8192) <= 1, "got {px:?}");
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert_eq!(px[3], 16383, "alpha must equal (1 << 14) - 1");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv444p16_gray_to_gray_u16() {
  let (yp, up, vp) = solid_yuv444p_n_frame(16, 8, 32768, 32768, 32768);
  let src = Yuv444p16Frame::new(&yp, &up, &vp, 16, 8, 16, 16, 16);

  let mut rgb_u8 = std::vec![0u8; 16 * 8 * 3];
  let mut rgb_u16 = std::vec![0u16; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yuv444p16>::new(16, 8)
    .with_rgb(&mut rgb_u8)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16)
    .unwrap();
  yuv444p16_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgb_u8.chunks(3) {
    assert!(px[0].abs_diff(128) <= 1);
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
  }
  for px in rgb_u16.chunks(3) {
    assert!(px[0].abs_diff(32768) <= 256);
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv422p_matches_yuv420p_luma_when_chroma_matches() {
  // 4:2:2 and 4:2:0 differ only in vertical chroma walk. With solid
  // chroma planes they must produce identical RGB output — this is
  // the whole reason Yuv422p reuses the yuv_420 row kernel.
  let w = 32u32;
  let h = 8u32;
  let (yp, up422, vp422) = solid_yuv422p_frame(w, h, 140, 100, 160);
  let src422 = Yuv422pFrame::new(&yp, &up422, &vp422, w, h, w, w / 2, w / 2);

  let (yp420, up420, vp420) = solid_yuv420p_frame(w, h, 140, 100, 160);
  let src420 = Yuv420pFrame::new(&yp420, &up420, &vp420, w, h, w, w / 2, w / 2);

  let mut rgb422 = std::vec![0u8; (w * h * 3) as usize];
  let mut rgb420 = std::vec![0u8; (w * h * 3) as usize];
  let mut s422 = MixedSinker::<Yuv422p>::new(w as usize, h as usize)
    .with_rgb(&mut rgb422)
    .unwrap();
  let mut s420 = MixedSinker::<Yuv420p>::new(w as usize, h as usize)
    .with_rgb(&mut rgb420)
    .unwrap();
  yuv422p_to(&src422, true, ColorMatrix::Bt709, &mut s422).unwrap();
  yuv420p_to(&src420, true, ColorMatrix::Bt709, &mut s420).unwrap();
  assert_eq!(rgb422, rgb420);
}

// ---- Yuv422p RGBA (Ship 8 PR 3) tests -----------------------------------
//
// Yuv422p reuses the Yuv420p `_to_rgba_row` dispatcher (same row
// contract). Tests mirror the Yuv420p RGBA set; the cross-format
// invariant against Yuv420p (with solid chroma so 4:2:0 vertical
// upsample matches Yuv422p's per-row chroma) catches walker
// regressions specific to the Yuv422p RGBA wiring.

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv422p_rgba_only_converts_gray_to_gray_with_opaque_alpha() {
  let (yp, up, vp) = solid_yuv422p_frame(16, 8, 128, 128, 128);
  let src = Yuv422pFrame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuv422p>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  yuv422p_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(128) <= 1, "R");
    assert_eq!(px[0], px[1], "RGB monochromatic");
    assert_eq!(px[1], px[2], "RGB monochromatic");
    assert_eq!(px[3], 0xFF, "alpha must default to opaque");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv422p_with_rgb_and_with_rgba_produce_byte_identical_rgb_bytes() {
  let w = 32u32;
  let h = 16u32;
  let ws = w as usize;
  let hs = h as usize;
  let (yp, up, vp) = solid_yuv422p_frame(w, h, 180, 60, 200);
  let src = Yuv422pFrame::new(&yp, &up, &vp, w, h, w, w / 2, w / 2);

  let mut rgb = std::vec![0u8; ws * hs * 3];
  let mut rgba = std::vec![0u8; ws * hs * 4];
  let mut sink = MixedSinker::<Yuv422p>::new(ws, hs)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap();
  yuv422p_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for i in 0..(ws * hs) {
    assert_eq!(rgba[i * 4], rgb[i * 3], "R differs at pixel {i}");
    assert_eq!(rgba[i * 4 + 1], rgb[i * 3 + 1], "G differs at pixel {i}");
    assert_eq!(rgba[i * 4 + 2], rgb[i * 3 + 2], "B differs at pixel {i}");
    assert_eq!(rgba[i * 4 + 3], 0xFF, "A not opaque at pixel {i}");
  }
}

#[test]
fn yuv422p_rgba_buffer_too_short_returns_err() {
  let mut rgba_short = std::vec![0u8; 16 * 8 * 4 - 1];
  let result = MixedSinker::<Yuv422p>::new(16, 8).with_rgba(&mut rgba_short);
  let Err(err) = result else {
    panic!("expected RgbaBufferTooShort error");
  };
  assert!(matches!(
    err,
    MixedSinkerError::RgbaBufferTooShort {
      expected: 512,
      actual: 511,
    }
  ));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv422p_rgba_simd_matches_scalar_with_random_yuv() {
  // Random per-pixel YUV across all matrices × both ranges. Width
  // 1922 forces both the SIMD main loop AND a scalar tail across
  // every backend block size (16/32/64). 4:2:2 chroma is full-
  // height, so up/vp use `w/2 × h` instead of `w/2 × h/2`.
  let w = 1922usize;
  let h = 4usize;
  let mut yp = std::vec![0u8; w * h];
  let mut up = std::vec![0u8; (w / 2) * h];
  let mut vp = std::vec![0u8; (w / 2) * h];
  pseudo_random_u8(&mut yp, 0xC001_C0DE);
  pseudo_random_u8(&mut up, 0xCAFE_F00D);
  pseudo_random_u8(&mut vp, 0xDEAD_BEEF);
  let src = Yuv422pFrame::new(
    &yp,
    &up,
    &vp,
    w as u32,
    h as u32,
    w as u32,
    (w / 2) as u32,
    (w / 2) as u32,
  );

  for &matrix in &[
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::YCgCo,
  ] {
    for &full_range in &[true, false] {
      let mut rgba_simd = std::vec![0u8; w * h * 4];
      let mut rgba_scalar = std::vec![0u8; w * h * 4];

      let mut s_simd = MixedSinker::<Yuv422p>::new(w, h)
        .with_rgba(&mut rgba_simd)
        .unwrap();
      yuv422p_to(&src, full_range, matrix, &mut s_simd).unwrap();

      let mut s_scalar = MixedSinker::<Yuv422p>::new(w, h)
        .with_rgba(&mut rgba_scalar)
        .unwrap();
      s_scalar.set_simd(false);
      yuv422p_to(&src, full_range, matrix, &mut s_scalar).unwrap();

      if rgba_simd != rgba_scalar {
        let mismatch = rgba_simd
          .iter()
          .zip(rgba_scalar.iter())
          .position(|(a, b)| a != b)
          .unwrap();
        let pixel = mismatch / 4;
        let channel = ["R", "G", "B", "A"][mismatch % 4];
        panic!(
          "Yuv422p RGBA SIMD ≠ scalar at byte {mismatch} (px {pixel} {channel}) for matrix={matrix:?} full_range={full_range}: simd={} scalar={}",
          rgba_simd[mismatch], rgba_scalar[mismatch]
        );
      }
    }
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv422p_rgba_matches_yuv420p_rgba_when_chroma_matches() {
  // 4:2:2 and 4:2:0 differ only in vertical chroma walk. With
  // solid chroma planes they must produce identical RGBA — same
  // shape as the existing `yuv422p_matches_yuv420p_luma_when_chroma_matches`
  // RGB-path test for the new RGBA path.
  let w = 32u32;
  let h = 8u32;
  let (yp, up422, vp422) = solid_yuv422p_frame(w, h, 140, 100, 160);
  let src422 = Yuv422pFrame::new(&yp, &up422, &vp422, w, h, w, w / 2, w / 2);

  let (yp420, up420, vp420) = solid_yuv420p_frame(w, h, 140, 100, 160);
  let src420 = Yuv420pFrame::new(&yp420, &up420, &vp420, w, h, w, w / 2, w / 2);

  let mut rgba422 = std::vec![0u8; (w * h * 4) as usize];
  let mut rgba420 = std::vec![0u8; (w * h * 4) as usize];
  let mut s422 = MixedSinker::<Yuv422p>::new(w as usize, h as usize)
    .with_rgba(&mut rgba422)
    .unwrap();
  let mut s420 = MixedSinker::<Yuv420p>::new(w as usize, h as usize)
    .with_rgba(&mut rgba420)
    .unwrap();
  yuv422p_to(&src422, true, ColorMatrix::Bt709, &mut s422).unwrap();
  yuv420p_to(&src420, true, ColorMatrix::Bt709, &mut s420).unwrap();
  assert_eq!(rgba422, rgba420);
}

// ---- 9-bit family + 4:4:0 family sanity tests ------------------------

pub(super) fn solid_yuv440p_frame(
  width: u32,
  height: u32,
  y: u8,
  u: u8,
  v: u8,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let w = width as usize;
  let h = height as usize;
  let ch = (height as usize).div_ceil(2);
  (
    std::vec![y; w * h],
    std::vec![u; w * ch],
    std::vec![v; w * ch],
  )
}

pub(super) fn solid_yuv440p_n_frame(
  width: u32,
  height: u32,
  y: u16,
  u: u16,
  v: u16,
) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
  let w = width as usize;
  let h = height as usize;
  let ch = (height as usize).div_ceil(2);
  (
    std::vec![y; w * h],
    std::vec![u; w * ch],
    std::vec![v; w * ch],
  )
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p9_gray_to_gray() {
  let (yp, up, vp) = solid_yuv422p_n_frame(16, 8, 256, 256, 256);
  // 4:2:0 chroma is w/2 × h/2; reuse the 4:2:2 helper's `cw * h` and
  // truncate to the 4:2:0 layout (cw = 8, ch = 4).
  let up = up[..8 * 4].to_vec();
  let vp = vp[..8 * 4].to_vec();
  let src = Yuv420p9Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yuv420p9>::new(16, 8)
    .with_rgb(&mut rgb)
    .unwrap();
  yuv420p9_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn yuv420p9_rgba_u8_only_gray_with_opaque_alpha() {
  // 9-bit mid-gray (Y=U=V=256) → 8-bit RGBA ≈ (128, 128, 128, 255).
  let (yp, up, vp) = solid_yuv422p_n_frame(16, 8, 256, 256, 256);
  let up = up[..8 * 4].to_vec();
  let vp = vp[..8 * 4].to_vec();
  let src = Yuv420p9Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuv420p9>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  yuv420p9_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn yuv420p9_rgba_u16_only_native_depth_gray_with_opaque_alpha() {
  // 9-bit mid-gray → u16 RGBA: each color element ≈ 256, alpha = 511.
  let (yp, up, vp) = solid_yuv422p_n_frame(16, 8, 256, 256, 256);
  let up = up[..8 * 4].to_vec();
  let vp = vp[..8 * 4].to_vec();
  let src = Yuv420p9Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgba = std::vec![0u16; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuv420p9>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  yuv420p9_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(256) <= 1, "got {px:?}");
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert_eq!(px[3], 511, "alpha must equal (1 << 9) - 1");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv422p9_gray_to_gray() {
  let (yp, up, vp) = solid_yuv422p_n_frame(16, 8, 256, 256, 256);
  let src = Yuv422p9Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yuv422p9>::new(16, 8)
    .with_rgb(&mut rgb)
    .unwrap();
  yuv422p9_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn yuv422p9_rgba_u8_only_gray_with_opaque_alpha() {
  // 9-bit mid-gray → 8-bit RGBA ≈ (128, 128, 128, 255).
  let (yp, up, vp) = solid_yuv422p_n_frame(16, 8, 256, 256, 256);
  let src = Yuv422p9Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuv422p9>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  yuv422p9_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn yuv422p9_rgba_u16_only_native_depth_gray_with_opaque_alpha() {
  // 9-bit mid-gray → u16 RGBA: each color element ≈ 256, alpha = 511.
  let (yp, up, vp) = solid_yuv422p_n_frame(16, 8, 256, 256, 256);
  let src = Yuv422p9Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgba = std::vec![0u16; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuv422p9>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  yuv422p9_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(256) <= 1, "got {px:?}");
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert_eq!(px[3], 511, "alpha must equal (1 << 9) - 1");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv444p9_gray_to_gray() {
  let (yp, up, vp) = solid_yuv444p_n_frame(16, 8, 256, 256, 256);
  let src = Yuv444p9Frame::new(&yp, &up, &vp, 16, 8, 16, 16, 16);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yuv444p9>::new(16, 8)
    .with_rgb(&mut rgb)
    .unwrap();
  yuv444p9_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn yuv444p9_rgba_u8_only_gray_with_opaque_alpha() {
  // 9-bit mid-gray → 8-bit RGBA ≈ (128, 128, 128, 255).
  let (yp, up, vp) = solid_yuv444p_n_frame(16, 8, 256, 256, 256);
  let src = Yuv444p9Frame::new(&yp, &up, &vp, 16, 8, 16, 16, 16);

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuv444p9>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  yuv444p9_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn yuv444p9_rgba_u16_only_native_depth_gray_with_opaque_alpha() {
  // 9-bit mid-gray → u16 RGBA: each color element ≈ 256, alpha = 511.
  let (yp, up, vp) = solid_yuv444p_n_frame(16, 8, 256, 256, 256);
  let src = Yuv444p9Frame::new(&yp, &up, &vp, 16, 8, 16, 16, 16);

  let mut rgba = std::vec![0u16; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuv444p9>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  yuv444p9_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(256) <= 1, "got {px:?}");
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert_eq!(px[3], 511, "alpha must equal (1 << 9) - 1");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv440p_gray_to_gray() {
  let (yp, up, vp) = solid_yuv440p_frame(16, 8, 128, 128, 128);
  let src = Yuv440pFrame::new(&yp, &up, &vp, 16, 8, 16, 16, 16);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yuv440p>::new(16, 8)
    .with_rgb(&mut rgb)
    .unwrap();
  yuv440p_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgb.chunks(3) {
    assert!(px[0].abs_diff(128) <= 1);
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
  }
}

// ---- Yuv440p RGBA (Ship 8 PR 4c) tests --------------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv440p_rgba_only_converts_gray_to_gray_with_opaque_alpha() {
  let (yp, up, vp) = solid_yuv440p_frame(16, 8, 128, 128, 128);
  let src = Yuv440pFrame::new(&yp, &up, &vp, 16, 8, 16, 16, 16);

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuv440p>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  yuv440p_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(128) <= 1, "R");
    assert_eq!(px[0], px[1], "RGB monochromatic");
    assert_eq!(px[1], px[2], "RGB monochromatic");
    assert_eq!(px[3], 0xFF, "alpha must default to opaque");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv440p_with_rgb_and_with_rgba_produce_byte_identical_rgb_bytes() {
  let w = 32u32;
  let h = 16u32;
  let ws = w as usize;
  let hs = h as usize;
  let (yp, up, vp) = solid_yuv440p_frame(w, h, 180, 60, 200);
  let src = Yuv440pFrame::new(&yp, &up, &vp, w, h, w, w, w);

  let mut rgb = std::vec![0u8; ws * hs * 3];
  let mut rgba = std::vec![0u8; ws * hs * 4];
  let mut sink = MixedSinker::<Yuv440p>::new(ws, hs)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap();
  yuv440p_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for i in 0..(ws * hs) {
    assert_eq!(rgba[i * 4], rgb[i * 3], "R differs at pixel {i}");
    assert_eq!(rgba[i * 4 + 1], rgb[i * 3 + 1], "G differs at pixel {i}");
    assert_eq!(rgba[i * 4 + 2], rgb[i * 3 + 2], "B differs at pixel {i}");
    assert_eq!(rgba[i * 4 + 3], 0xFF, "A not opaque at pixel {i}");
  }
}

#[test]
fn yuv440p_rgba_buffer_too_short_returns_err() {
  let mut rgba_short = std::vec![0u8; 16 * 8 * 4 - 1];
  let result = MixedSinker::<Yuv440p>::new(16, 8).with_rgba(&mut rgba_short);
  let Err(err) = result else {
    panic!("expected RgbaBufferTooShort error");
  };
  assert!(matches!(
    err,
    MixedSinkerError::RgbaBufferTooShort {
      expected: 512,
      actual: 511,
    }
  ));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv440p_rgba_simd_matches_scalar_with_random_yuv() {
  // Width 1922 forces both the SIMD main loop AND scalar tail across
  // every backend block size (16/32/64). 4:4:0 chroma is full-width
  // but half-height, so chroma plane is `w * h/2`.
  let w = 1922usize;
  let h = 4usize;
  let ch = h / 2;
  let mut yp = std::vec![0u8; w * h];
  let mut up = std::vec![0u8; w * ch];
  let mut vp = std::vec![0u8; w * ch];
  pseudo_random_u8(&mut yp, 0xC001_C0DE);
  pseudo_random_u8(&mut up, 0xCAFE_F00D);
  pseudo_random_u8(&mut vp, 0xDEAD_BEEF);
  let src = Yuv440pFrame::new(
    &yp, &up, &vp, w as u32, h as u32, w as u32, w as u32, w as u32,
  );

  for &matrix in &[
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::YCgCo,
  ] {
    for &full_range in &[true, false] {
      let mut rgba_simd = std::vec![0u8; w * h * 4];
      let mut rgba_scalar = std::vec![0u8; w * h * 4];

      let mut s_simd = MixedSinker::<Yuv440p>::new(w, h)
        .with_rgba(&mut rgba_simd)
        .unwrap();
      yuv440p_to(&src, full_range, matrix, &mut s_simd).unwrap();

      let mut s_scalar = MixedSinker::<Yuv440p>::new(w, h)
        .with_rgba(&mut rgba_scalar)
        .unwrap();
      s_scalar.set_simd(false);
      yuv440p_to(&src, full_range, matrix, &mut s_scalar).unwrap();

      assert_eq!(
        rgba_simd, rgba_scalar,
        "Yuv440p RGBA SIMD ≠ scalar (matrix={matrix:?}, full_range={full_range})"
      );
    }
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv440p10_gray_to_gray() {
  let (yp, up, vp) = solid_yuv440p_n_frame(16, 8, 512, 512, 512);
  let src = Yuv440p10Frame::new(&yp, &up, &vp, 16, 8, 16, 16, 16);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yuv440p10>::new(16, 8)
    .with_rgb(&mut rgb)
    .unwrap();
  yuv440p10_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn yuv440p12_gray_to_gray() {
  let (yp, up, vp) = solid_yuv440p_n_frame(16, 8, 2048, 2048, 2048);
  let src = Yuv440p12Frame::new(&yp, &up, &vp, 16, 8, 16, 16, 16);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yuv440p12>::new(16, 8)
    .with_rgb(&mut rgb)
    .unwrap();
  yuv440p12_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn yuv440p12_rgba_u8_only_gray_with_opaque_alpha() {
  // 4:4:0 reuses the 4:4:4 dispatcher. 12-bit mid-gray → 8-bit RGBA
  // ≈ (128, 128, 128, 255).
  let (yp, up, vp) = solid_yuv440p_n_frame(16, 8, 2048, 2048, 2048);
  let src = Yuv440p12Frame::new(&yp, &up, &vp, 16, 8, 16, 16, 16);

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuv440p12>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  yuv440p12_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn yuv440p12_rgba_u16_only_native_depth_gray_with_opaque_alpha() {
  // 12-bit mid-gray → u16 RGBA: each color element ≈ 2048, alpha = 4095.
  let (yp, up, vp) = solid_yuv440p_n_frame(16, 8, 2048, 2048, 2048);
  let src = Yuv440p12Frame::new(&yp, &up, &vp, 16, 8, 16, 16, 16);

  let mut rgba = std::vec![0u16; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuv440p12>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  yuv440p12_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(2048) <= 1, "got {px:?}");
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert_eq!(px[3], 4095, "alpha must equal (1 << 12) - 1");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv440p_matches_yuv444p_when_chroma_constant_per_pair() {
  // 4:4:0 reuses the 4:4:4 row math; the only difference is the
  // walker reads chroma row r/2. With the same chroma value at every
  // (r, c), Yuv440p must produce identical RGB to Yuv444p with
  // duplicated chroma rows.
  let w = 32u32;
  let h = 8u32;
  let (yp, up440, vp440) = solid_yuv440p_frame(w, h, 140, 100, 160);
  let src440 = Yuv440pFrame::new(&yp, &up440, &vp440, w, h, w, w, w);

  // Yuv444p needs full-height chroma, so duplicate each of the 4 4:4:0
  // chroma rows into 2 rows.
  let mut up444 = std::vec::Vec::with_capacity((w * h) as usize);
  let mut vp444 = std::vec::Vec::with_capacity((w * h) as usize);
  for r in 0..h {
    let cr = (r / 2) as usize;
    let row_start = cr * w as usize;
    let row_end = row_start + w as usize;
    up444.extend_from_slice(&up440[row_start..row_end]);
    vp444.extend_from_slice(&vp440[row_start..row_end]);
  }
  let src444 = Yuv444pFrame::new(&yp, &up444, &vp444, w, h, w, w, w);

  let mut rgb440 = std::vec![0u8; (w * h * 3) as usize];
  let mut rgb444 = std::vec![0u8; (w * h * 3) as usize];
  let mut s440 = MixedSinker::<Yuv440p>::new(w as usize, h as usize)
    .with_rgb(&mut rgb440)
    .unwrap();
  let mut s444 = MixedSinker::<Yuv444p>::new(w as usize, h as usize)
    .with_rgb(&mut rgb444)
    .unwrap();
  yuv440p_to(&src440, true, ColorMatrix::Bt709, &mut s440).unwrap();
  yuv444p_to(&src444, true, ColorMatrix::Bt709, &mut s444).unwrap();
  assert_eq!(rgb440, rgb444);
}

// ---- Walker-level SIMD-vs-scalar equivalence for 9-bit 4:2:x --------
//
// Per-arch row-kernel tests cover the BITS=9 path with non-neutral
// chroma directly. These walker-level tests additionally pin the
// public dispatcher behavior — Yuv420p9 / Yuv422p9 read through the
// same `yuv_420p_n_to_rgb_*<9>` half-width kernels, so a backend
// bug here would silently corrupt user output. Width 1922 forces
// both the SIMD main loop and a scalar tail; chroma is non-neutral
// and limited-range parameters are exercised below.

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p9_walker_simd_matches_scalar_with_random_chroma() {
  let w = 1922u32; // forces tail handling on every backend
  let h = 4u32;
  let mut yp = std::vec![0u16; (w * h) as usize];
  let mut up = std::vec![0u16; ((w / 2) * (h / 2)) as usize];
  let mut vp = std::vec![0u16; ((w / 2) * (h / 2)) as usize];
  pseudo_random_u16_low_n_bits(&mut yp, 0x1111, 9);
  pseudo_random_u16_low_n_bits(&mut up, 0x2222, 9);
  pseudo_random_u16_low_n_bits(&mut vp, 0x3333, 9);
  let src = Yuv420p9Frame::new(&yp, &up, &vp, w, h, w, w / 2, w / 2);

  for &full_range in &[true, false] {
    let mut rgb_simd = std::vec![0u8; (w * h * 3) as usize];
    let mut rgb_scalar = std::vec![0u8; (w * h * 3) as usize];
    let mut rgb_u16_simd = std::vec![0u16; (w * h * 3) as usize];
    let mut rgb_u16_scalar = std::vec![0u16; (w * h * 3) as usize];

    let mut s_simd = MixedSinker::<Yuv420p9>::new(w as usize, h as usize)
      .with_rgb(&mut rgb_simd)
      .unwrap()
      .with_rgb_u16(&mut rgb_u16_simd)
      .unwrap();
    yuv420p9_to(&src, full_range, ColorMatrix::Bt709, &mut s_simd).unwrap();

    let mut s_scalar = MixedSinker::<Yuv420p9>::new(w as usize, h as usize)
      .with_rgb(&mut rgb_scalar)
      .unwrap()
      .with_rgb_u16(&mut rgb_u16_scalar)
      .unwrap();
    s_scalar.set_simd(false);
    yuv420p9_to(&src, full_range, ColorMatrix::Bt709, &mut s_scalar).unwrap();

    assert_eq!(rgb_simd, rgb_scalar, "Yuv420p9 SIMD u8 ≠ scalar u8");
    assert_eq!(
      rgb_u16_simd, rgb_u16_scalar,
      "Yuv420p9 SIMD u16 ≠ scalar u16"
    );
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv422p9_walker_simd_matches_scalar_with_random_chroma() {
  let w = 1922u32;
  let h = 4u32;
  let mut yp = std::vec![0u16; (w * h) as usize];
  let mut up = std::vec![0u16; ((w / 2) * h) as usize];
  let mut vp = std::vec![0u16; ((w / 2) * h) as usize];
  pseudo_random_u16_low_n_bits(&mut yp, 0x4444, 9);
  pseudo_random_u16_low_n_bits(&mut up, 0x5555, 9);
  pseudo_random_u16_low_n_bits(&mut vp, 0x6666, 9);
  let src = Yuv422p9Frame::new(&yp, &up, &vp, w, h, w, w / 2, w / 2);

  for &full_range in &[true, false] {
    let mut rgb_simd = std::vec![0u8; (w * h * 3) as usize];
    let mut rgb_scalar = std::vec![0u8; (w * h * 3) as usize];
    let mut rgb_u16_simd = std::vec![0u16; (w * h * 3) as usize];
    let mut rgb_u16_scalar = std::vec![0u16; (w * h * 3) as usize];

    let mut s_simd = MixedSinker::<Yuv422p9>::new(w as usize, h as usize)
      .with_rgb(&mut rgb_simd)
      .unwrap()
      .with_rgb_u16(&mut rgb_u16_simd)
      .unwrap();
    yuv422p9_to(&src, full_range, ColorMatrix::Bt2020Ncl, &mut s_simd).unwrap();

    let mut s_scalar = MixedSinker::<Yuv422p9>::new(w as usize, h as usize)
      .with_rgb(&mut rgb_scalar)
      .unwrap()
      .with_rgb_u16(&mut rgb_u16_scalar)
      .unwrap();
    s_scalar.set_simd(false);
    yuv422p9_to(&src, full_range, ColorMatrix::Bt2020Ncl, &mut s_scalar).unwrap();

    assert_eq!(rgb_simd, rgb_scalar, "Yuv422p9 SIMD u8 ≠ scalar u8");
    assert_eq!(
      rgb_u16_simd, rgb_u16_scalar,
      "Yuv422p9 SIMD u16 ≠ scalar u16"
    );
  }
}
