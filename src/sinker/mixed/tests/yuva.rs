use super::*;

// ---- Yuva444p10 (Ship 8b‑1a) ----------------------------------

fn solid_yuva444p10_frame(
  width: u32,
  height: u32,
  y: u16,
  u: u16,
  v: u16,
  a: u16,
) -> (Vec<u16>, Vec<u16>, Vec<u16>, Vec<u16>) {
  let n = (width * height) as usize;
  (
    std::vec![y; n],
    std::vec![u; n],
    std::vec![v; n],
    std::vec![a; n],
  )
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva444p10_rgba_u8_with_source_alpha_passes_through() {
  // 10-bit mid-gray with non-opaque alpha: Y=U=V=512, A=256.
  // u8 RGBA path: each color byte ≈ 128, alpha = 256 >> 2 = 64.
  let (yp, up, vp, ap) = solid_yuva444p10_frame(16, 8, 512, 512, 512, 256);
  let src = Yuva444p10Frame::try_new(&yp, &up, &vp, &ap, 16, 8, 16, 16, 16, 16).unwrap();

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuva444p10>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  yuva444p10_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(128) <= 1, "got {px:?}");
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert_eq!(px[3], 64, "alpha must equal 256 >> 2 = 64");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva444p10_rgba_u16_with_source_alpha_passes_through_native_depth() {
  // 10-bit mid-gray with non-opaque alpha: Y=U=V=512, A=256.
  // u16 RGBA path: each color element ≈ 512, alpha = 256 (native depth).
  let (yp, up, vp, ap) = solid_yuva444p10_frame(16, 8, 512, 512, 512, 256);
  let src = Yuva444p10Frame::try_new(&yp, &up, &vp, &ap, 16, 8, 16, 16, 16, 16).unwrap();

  let mut rgba = std::vec![0u16; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuva444p10>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  yuva444p10_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(512) <= 1, "got {px:?}");
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert_eq!(px[3], 256, "alpha must equal source A (native depth)");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva444p10_rgba_u8_fully_opaque_alpha_yields_0xff() {
  // Source A = (1 << 10) - 1 = 1023 → u8 alpha = 1023 >> 2 = 255 = 0xFF.
  let (yp, up, vp, ap) = solid_yuva444p10_frame(16, 8, 512, 512, 512, 1023);
  let src = Yuva444p10Frame::try_new(&yp, &up, &vp, &ap, 16, 8, 16, 16, 16, 16).unwrap();

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuva444p10>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  yuva444p10_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert_eq!(px[3], 0xFF, "fully-opaque source alpha must yield 0xFF");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva444p10_rgba_u16_fully_opaque_alpha_yields_native_max() {
  // Source A = 1023 → u16 alpha = 1023 (no depth conversion needed).
  let (yp, up, vp, ap) = solid_yuva444p10_frame(16, 8, 512, 512, 512, 1023);
  let src = Yuva444p10Frame::try_new(&yp, &up, &vp, &ap, 16, 8, 16, 16, 16, 16).unwrap();

  let mut rgba = std::vec![0u16; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuva444p10>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  yuva444p10_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert_eq!(px[3], 1023, "fully-opaque source alpha = (1 << 10) - 1");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva444p10_rgba_u8_zero_alpha_yields_0() {
  // Source A = 0 → u8 alpha = 0 (fully transparent).
  let (yp, up, vp, ap) = solid_yuva444p10_frame(16, 8, 512, 512, 512, 0);
  let src = Yuva444p10Frame::try_new(&yp, &up, &vp, &ap, 16, 8, 16, 16, 16, 16).unwrap();

  let mut rgba = std::vec![0xFFu8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuva444p10>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  yuva444p10_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert_eq!(px[3], 0, "zero source alpha must yield 0");
  }
}

#[test]
fn yuva444p10_rgba_buf_too_short_returns_err() {
  let mut rgba = std::vec![0u8; 10];
  let err = MixedSinker::<Yuva444p10>::new(16, 8)
    .with_rgba(&mut rgba)
    .err()
    .expect("expected RgbaBufferTooShort");
  assert!(matches!(err, MixedSinkerError::RgbaBufferTooShort { .. }));
}

#[test]
fn yuva444p10_rgba_u16_buf_too_short_returns_err() {
  let mut rgba = std::vec![0u16; 10];
  let err = MixedSinker::<Yuva444p10>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .err()
    .expect("expected RgbaU16BufferTooShort");
  assert!(matches!(
    err,
    MixedSinkerError::RgbaU16BufferTooShort { .. }
  ));
}

// ---- Yuva444p10 alpha-drop paths (Codex PR #32 review fix #1) ----
//
// `with_rgb` / `with_luma` / `with_hsv` are declared on the generic
// `MixedSinker<F>` impl, so attaching them to a `MixedSinker::<Yuva444p10>`
// is callable. Previously the `process` impl only wrote `rgba` /
// `rgba_u16` and silently returned Ok, leaving these buffers stale.
// These tests pin that the alpha-drop paths now write byte-identical
// output to what `MixedSinker::<Yuv444p10>` would produce on the same
// Y/U/V data.

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva444p10_with_rgb_writes_buffer_alpha_drop_matches_yuv444p10() {
  let (yp, up, vp) = solid_yuv444p_n_frame(16, 8, 600, 400, 700);
  let (yp_a, up_a, vp_a, ap) = solid_yuva444p10_frame(16, 8, 600, 400, 700, 256);

  let yuv = Yuv444p10Frame::new(&yp, &up, &vp, 16, 8, 16, 16, 16);
  let yuva = Yuva444p10Frame::try_new(&yp_a, &up_a, &vp_a, &ap, 16, 8, 16, 16, 16, 16).unwrap();

  let mut rgb_yuv = std::vec![0u8; 16 * 8 * 3];
  let mut s_yuv = MixedSinker::<Yuv444p10>::new(16, 8)
    .with_rgb(&mut rgb_yuv)
    .unwrap();
  yuv444p10_to(&yuv, true, ColorMatrix::Bt709, &mut s_yuv).unwrap();

  let mut rgb_yuva = std::vec![0u8; 16 * 8 * 3];
  let mut s_yuva = MixedSinker::<Yuva444p10>::new(16, 8)
    .with_rgb(&mut rgb_yuva)
    .unwrap();
  yuva444p10_to(&yuva, true, ColorMatrix::Bt709, &mut s_yuva).unwrap();

  assert_eq!(
    rgb_yuv, rgb_yuva,
    "Yuva444p10 with_rgb (alpha-drop) must equal Yuv444p10 with_rgb"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva444p10_with_rgb_u16_writes_buffer_alpha_drop_matches_yuv444p10() {
  let (yp, up, vp) = solid_yuv444p_n_frame(16, 8, 600, 400, 700);
  let (yp_a, up_a, vp_a, ap) = solid_yuva444p10_frame(16, 8, 600, 400, 700, 256);

  let yuv = Yuv444p10Frame::new(&yp, &up, &vp, 16, 8, 16, 16, 16);
  let yuva = Yuva444p10Frame::try_new(&yp_a, &up_a, &vp_a, &ap, 16, 8, 16, 16, 16, 16).unwrap();

  let mut rgb_yuv = std::vec![0u16; 16 * 8 * 3];
  let mut s_yuv = MixedSinker::<Yuv444p10>::new(16, 8)
    .with_rgb_u16(&mut rgb_yuv)
    .unwrap();
  yuv444p10_to(&yuv, true, ColorMatrix::Bt709, &mut s_yuv).unwrap();

  let mut rgb_yuva = std::vec![0u16; 16 * 8 * 3];
  let mut s_yuva = MixedSinker::<Yuva444p10>::new(16, 8)
    .with_rgb_u16(&mut rgb_yuva)
    .unwrap();
  yuva444p10_to(&yuva, true, ColorMatrix::Bt709, &mut s_yuva).unwrap();

  assert_eq!(
    rgb_yuv, rgb_yuva,
    "Yuva444p10 with_rgb_u16 (alpha-drop) must equal Yuv444p10 with_rgb_u16"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva444p10_with_luma_writes_buffer_y_downshift_8bit() {
  let (yp, up, vp, ap) = solid_yuva444p10_frame(16, 8, 512, 512, 512, 256);
  let src = Yuva444p10Frame::try_new(&yp, &up, &vp, &ap, 16, 8, 16, 16, 16, 16).unwrap();

  let mut luma = std::vec![0u8; 16 * 8];
  let mut sink = MixedSinker::<Yuva444p10>::new(16, 8)
    .with_luma(&mut luma)
    .unwrap();
  yuva444p10_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  assert!(luma.iter().all(|&l| l == 128));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva444p10_with_hsv_writes_buffer_gray_is_zero_hue_zero_sat() {
  let (yp, up, vp, ap) = solid_yuva444p10_frame(16, 8, 512, 512, 512, 256);
  let src = Yuva444p10Frame::try_new(&yp, &up, &vp, &ap, 16, 8, 16, 16, 16, 16).unwrap();

  let mut h = std::vec![0xFFu8; 16 * 8];
  let mut s = std::vec![0xFFu8; 16 * 8];
  let mut v = std::vec![0xFFu8; 16 * 8];
  let mut sink = MixedSinker::<Yuva444p10>::new(16, 8)
    .with_hsv(&mut h, &mut s, &mut v)
    .unwrap();
  yuva444p10_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  assert!(h.iter().all(|&b| b == 0), "neutral gray → H = 0");
  assert!(s.iter().all(|&b| b == 0), "neutral gray → S = 0");
  assert!(
    v.iter().all(|&b| b.abs_diff(128) <= 1),
    "neutral gray → V ≈ 128"
  );
}

// ---- Yuva444p10 RGB + RGBA combine (alpha-source forks per buffer) -

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva444p10_with_rgb_and_with_rgba_both_write_buffers() {
  // Both attached: RGB fills with alpha-drop bytes; RGBA fills with
  // source-derived alpha. RGBA quads' RGB triples must equal the RGB
  // buffer; alpha is `source >> 2` per the depth conversion.
  let (yp, up, vp, ap) = solid_yuva444p10_frame(16, 8, 600, 400, 700, 512);
  let src = Yuva444p10Frame::try_new(&yp, &up, &vp, &ap, 16, 8, 16, 16, 16, 16).unwrap();

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuva444p10>::new(16, 8)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap();
  yuva444p10_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

  for (rgb_px, rgba_px) in rgb.chunks(3).zip(rgba.chunks(4)) {
    assert_eq!(rgb_px[0], rgba_px[0]);
    assert_eq!(rgb_px[1], rgba_px[1]);
    assert_eq!(rgb_px[2], rgba_px[2]);
    assert_eq!(rgba_px[3], 128u8, "alpha = (512 >> 2) = 128");
  }
}

// ---- Yuva444p10 overrange alpha clamping (Codex PR #32 review fix #2) ----
//
// `Yuva444p10Frame::try_new` admits any `&[u16]` for the alpha plane
// without per-sample validation (only `try_new_checked` validates).
// The scalar templates now mask `a_src` reads with `bits_mask::<BITS>()`
// — without that mask an overrange `1024` at BITS=10 would shift to
// `256` and cast to u8 zero, silently turning over-range alpha into
// transparent output. These tests pin the masking behavior.

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva444p10_rgba_u8_overrange_alpha_is_masked_to_low_bits() {
  // alpha = 0x0500 (1280): bits beyond the low 10 are masked away,
  // leaving 0x100 (256). u8 conversion: 256 >> 2 = 64.
  let (yp, up, vp, ap) = solid_yuva444p10_frame(16, 8, 512, 512, 512, 0x0500);
  let src = Yuva444p10Frame::try_new(&yp, &up, &vp, &ap, 16, 8, 16, 16, 16, 16).unwrap();

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuva444p10>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  yuva444p10_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert_eq!(
      px[3], 64,
      "0x0500 masked to low 10 bits = 256, u8 = 256 >> 2 = 64"
    );
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva444p10_rgba_u16_overrange_alpha_is_masked_to_low_bits() {
  // alpha = 0xFFFF: low 10 bits = 0x3FF (1023). Without the mask the
  // raw u16 0xFFFF would leak straight to output, exceeding the
  // documented `[0, 1023]` native-depth range.
  let (yp, up, vp, ap) = solid_yuva444p10_frame(16, 8, 512, 512, 512, 0xFFFF);
  let src = Yuva444p10Frame::try_new(&yp, &up, &vp, &ap, 16, 8, 16, 16, 16, 16).unwrap();

  let mut rgba = std::vec![0u16; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuva444p10>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  yuva444p10_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert_eq!(
      px[3], 1023,
      "0xFFFF masked to low 10 bits = 1023 (max valid 10-bit value)"
    );
  }
}

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

// ---- Yuva422p family (Ship 8b‑3) -----------------------------------

fn solid_yuva422p_frame(
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
  // 4:2:2: chroma full-height (only horizontal subsampling).
  (
    std::vec![y; w * h],
    std::vec![u; cw * h],
    std::vec![v; cw * h],
    std::vec![a; w * h],
  )
}

fn solid_yuva422p_frame_u16(
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
  (
    std::vec![y; w * h],
    std::vec![u; cw * h],
    std::vec![v; cw * h],
    std::vec![a; w * h],
  )
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva422p_rgba_u8_with_source_alpha_passes_through() {
  let (yp, up, vp, ap) = solid_yuva422p_frame(16, 8, 128, 128, 128, 128);
  let src = Yuva422pFrame::try_new(&yp, &up, &vp, &ap, 16, 8, 16, 8, 8, 16).unwrap();

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuva422p>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  yuva422p_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(128) <= 1, "got {px:?}");
    assert_eq!(px[3], 128, "alpha pass-through");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva422p_with_rgb_alpha_drop_matches_yuv422p() {
  let (yp_a, up_a, vp_a, ap) = solid_yuva422p_frame(16, 8, 180, 60, 200, 200);
  let yuv = Yuv422pFrame::try_new(&yp_a, &up_a, &vp_a, 16, 8, 16, 8, 8).unwrap();
  let yuva = Yuva422pFrame::try_new(&yp_a, &up_a, &vp_a, &ap, 16, 8, 16, 8, 8, 16).unwrap();

  let mut rgb_yuv = std::vec![0u8; 16 * 8 * 3];
  let mut s_yuv = MixedSinker::<Yuv422p>::new(16, 8)
    .with_rgb(&mut rgb_yuv)
    .unwrap();
  yuv422p_to(&yuv, true, ColorMatrix::Bt709, &mut s_yuv).unwrap();

  let mut rgb_yuva = std::vec![0u8; 16 * 8 * 3];
  let mut s_yuva = MixedSinker::<Yuva422p>::new(16, 8)
    .with_rgb(&mut rgb_yuva)
    .unwrap();
  yuva422p_to(&yuva, true, ColorMatrix::Bt709, &mut s_yuva).unwrap();

  assert_eq!(rgb_yuv, rgb_yuva);
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva422p9_rgba_u8_with_source_alpha_passes_through() {
  // 9-bit mid-gray (Y=U=V=256) and mid-alpha (A=128 → u8 alpha = 128 >> 1 = 64).
  let (yp, up, vp, ap) = solid_yuva422p_frame_u16(16, 8, 256, 256, 256, 128);
  let src = Yuva422p9Frame::try_new(&yp, &up, &vp, &ap, 16, 8, 16, 8, 8, 16).unwrap();

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuva422p9>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  yuva422p9_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn yuva422p9_rgba_u16_native_depth() {
  let (yp, up, vp, ap) = solid_yuva422p_frame_u16(16, 8, 256, 256, 256, 128);
  let src = Yuva422p9Frame::try_new(&yp, &up, &vp, &ap, 16, 8, 16, 8, 8, 16).unwrap();

  let mut rgba = std::vec![0u16; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuva422p9>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  yuva422p9_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert_eq!(px[3], 128, "alpha at native depth");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva422p10_rgba_u8_with_source_alpha_passes_through() {
  let (yp, up, vp, ap) = solid_yuva422p_frame_u16(16, 8, 512, 512, 512, 512);
  let src = Yuva422p10Frame::try_new(&yp, &up, &vp, &ap, 16, 8, 16, 8, 8, 16).unwrap();

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuva422p10>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  yuva422p10_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert_eq!(px[3], 128, "alpha = 512 >> (10-8) = 128");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva422p10_rgba_u16_native_depth() {
  let (yp, up, vp, ap) = solid_yuva422p_frame_u16(16, 8, 512, 512, 512, 512);
  let src = Yuva422p10Frame::try_new(&yp, &up, &vp, &ap, 16, 8, 16, 8, 8, 16).unwrap();

  let mut rgba = std::vec![0u16; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuva422p10>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  yuva422p10_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert_eq!(px[3], 512, "alpha at native depth");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva422p16_rgba_u8_with_source_alpha_passes_through() {
  // 16-bit full-range Y=U=V=32768 (mid-gray) + alpha=32768 → u8 alpha = 32768 >> 8 = 128.
  let (yp, up, vp, ap) = solid_yuva422p_frame_u16(16, 8, 32768, 32768, 32768, 32768);
  let src = Yuva422p16Frame::try_new(&yp, &up, &vp, &ap, 16, 8, 16, 8, 8, 16).unwrap();

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuva422p16>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  yuva422p16_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert_eq!(px[3], 128, "alpha = 32768 >> 8 = 128");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva422p16_rgba_u16_native_depth_full_range() {
  let (yp, up, vp, ap) = solid_yuva422p_frame_u16(16, 8, 32768, 32768, 32768, 32768);
  let src = Yuva422p16Frame::try_new(&yp, &up, &vp, &ap, 16, 8, 16, 8, 8, 16).unwrap();

  let mut rgba = std::vec![0u16; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuva422p16>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  yuva422p16_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert_eq!(px[3], 32768, "alpha at native depth");
  }
}

// ---- Yuva444p9 (Ship 8b‑3) -----------------------------------------

fn solid_yuva444p9_frame(
  width: u32,
  height: u32,
  y: u16,
  u: u16,
  v: u16,
  a: u16,
) -> (Vec<u16>, Vec<u16>, Vec<u16>, Vec<u16>) {
  let w = width as usize;
  let h = height as usize;
  // 4:4:4: chroma full-width × full-height.
  (
    std::vec![y; w * h],
    std::vec![u; w * h],
    std::vec![v; w * h],
    std::vec![a; w * h],
  )
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva444p9_rgba_u8_with_source_alpha_passes_through() {
  let (yp, up, vp, ap) = solid_yuva444p9_frame(16, 8, 256, 256, 256, 128);
  let src = Yuva444p9Frame::try_new(&yp, &up, &vp, &ap, 16, 8, 16, 16, 16, 16).unwrap();

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuva444p9>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  yuva444p9_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn yuva444p9_rgba_u16_native_depth() {
  let (yp, up, vp, ap) = solid_yuva444p9_frame(16, 8, 256, 256, 256, 128);
  let src = Yuva444p9Frame::try_new(&yp, &up, &vp, &ap, 16, 8, 16, 16, 16, 16).unwrap();

  let mut rgba = std::vec![0u16; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuva444p9>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  yuva444p9_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert_eq!(px[3], 128, "alpha at native depth");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva444p9_with_rgb_alpha_drop_matches_yuv444p9() {
  let (yp, up, vp, ap) = solid_yuva444p9_frame(16, 8, 256, 100, 200, 256);
  let yuv = Yuv444p9Frame::try_new(&yp, &up, &vp, 16, 8, 16, 16, 16).unwrap();
  let yuva = Yuva444p9Frame::try_new(&yp, &up, &vp, &ap, 16, 8, 16, 16, 16, 16).unwrap();

  let mut rgb_yuv = std::vec![0u8; 16 * 8 * 3];
  let mut s_yuv = MixedSinker::<Yuv444p9>::new(16, 8)
    .with_rgb(&mut rgb_yuv)
    .unwrap();
  yuv444p9_to(&yuv, true, ColorMatrix::Bt709, &mut s_yuv).unwrap();

  let mut rgb_yuva = std::vec![0u8; 16 * 8 * 3];
  let mut s_yuva = MixedSinker::<Yuva444p9>::new(16, 8)
    .with_rgb(&mut rgb_yuva)
    .unwrap();
  yuva444p9_to(&yuva, true, ColorMatrix::Bt709, &mut s_yuva).unwrap();

  assert_eq!(rgb_yuv, rgb_yuva);
}

// ---- Yuva422p12 / Yuva444p12 / Yuva444p14 (Ship 8b‑4) --------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva422p12_rgba_u8_with_source_alpha_passes_through() {
  let (yp, up, vp, ap) = solid_yuva422p_frame_u16(16, 8, 2048, 2048, 2048, 2048);
  let src = Yuva422p12Frame::try_new(&yp, &up, &vp, &ap, 16, 8, 16, 8, 8, 16).unwrap();

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuva422p12>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  yuva422p12_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert_eq!(px[3], 128, "alpha = 2048 >> (12-8) = 128");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva422p12_rgba_u16_native_depth() {
  let (yp, up, vp, ap) = solid_yuva422p_frame_u16(16, 8, 2048, 2048, 2048, 2048);
  let src = Yuva422p12Frame::try_new(&yp, &up, &vp, &ap, 16, 8, 16, 8, 8, 16).unwrap();

  let mut rgba = std::vec![0u16; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuva422p12>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  yuva422p12_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert_eq!(px[3], 2048, "alpha at native depth");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva422p12_with_rgb_alpha_drop_matches_yuv422p12() {
  let (yp, up, vp, ap) = solid_yuva422p_frame_u16(16, 8, 2048, 1500, 2500, 2048);
  let yuv = Yuv422p12Frame::try_new(&yp, &up, &vp, 16, 8, 16, 8, 8).unwrap();
  let yuva = Yuva422p12Frame::try_new(&yp, &up, &vp, &ap, 16, 8, 16, 8, 8, 16).unwrap();

  let mut rgb_yuv = std::vec![0u8; 16 * 8 * 3];
  let mut s_yuv = MixedSinker::<Yuv422p12>::new(16, 8)
    .with_rgb(&mut rgb_yuv)
    .unwrap();
  yuv422p12_to(&yuv, true, ColorMatrix::Bt709, &mut s_yuv).unwrap();

  let mut rgb_yuva = std::vec![0u8; 16 * 8 * 3];
  let mut s_yuva = MixedSinker::<Yuva422p12>::new(16, 8)
    .with_rgb(&mut rgb_yuva)
    .unwrap();
  yuva422p12_to(&yuva, true, ColorMatrix::Bt709, &mut s_yuva).unwrap();

  assert_eq!(rgb_yuv, rgb_yuva);
}

fn solid_yuva444p_frame_u16(
  width: u32,
  height: u32,
  y: u16,
  u: u16,
  v: u16,
  a: u16,
) -> (Vec<u16>, Vec<u16>, Vec<u16>, Vec<u16>) {
  let w = width as usize;
  let h = height as usize;
  // 4:4:4: chroma full-width × full-height; alpha 1:1 with Y.
  (
    std::vec![y; w * h],
    std::vec![u; w * h],
    std::vec![v; w * h],
    std::vec![a; w * h],
  )
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva444p12_rgba_u8_with_source_alpha_passes_through() {
  let (yp, up, vp, ap) = solid_yuva444p_frame_u16(16, 8, 2048, 2048, 2048, 2048);
  let src = Yuva444p12Frame::try_new(&yp, &up, &vp, &ap, 16, 8, 16, 16, 16, 16).unwrap();

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuva444p12>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  yuva444p12_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert_eq!(px[3], 128, "alpha = 2048 >> (12-8) = 128");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva444p12_rgba_u16_native_depth() {
  let (yp, up, vp, ap) = solid_yuva444p_frame_u16(16, 8, 2048, 2048, 2048, 2048);
  let src = Yuva444p12Frame::try_new(&yp, &up, &vp, &ap, 16, 8, 16, 16, 16, 16).unwrap();

  let mut rgba = std::vec![0u16; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuva444p12>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  yuva444p12_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert_eq!(px[3], 2048, "alpha at native depth");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva444p12_with_rgb_alpha_drop_matches_yuv444p12() {
  let (yp, up, vp, ap) = solid_yuva444p_frame_u16(16, 8, 2048, 1500, 2500, 2048);
  let yuv = Yuv444p12Frame::try_new(&yp, &up, &vp, 16, 8, 16, 16, 16).unwrap();
  let yuva = Yuva444p12Frame::try_new(&yp, &up, &vp, &ap, 16, 8, 16, 16, 16, 16).unwrap();

  let mut rgb_yuv = std::vec![0u8; 16 * 8 * 3];
  let mut s_yuv = MixedSinker::<Yuv444p12>::new(16, 8)
    .with_rgb(&mut rgb_yuv)
    .unwrap();
  yuv444p12_to(&yuv, true, ColorMatrix::Bt709, &mut s_yuv).unwrap();

  let mut rgb_yuva = std::vec![0u8; 16 * 8 * 3];
  let mut s_yuva = MixedSinker::<Yuva444p12>::new(16, 8)
    .with_rgb(&mut rgb_yuva)
    .unwrap();
  yuva444p12_to(&yuva, true, ColorMatrix::Bt709, &mut s_yuva).unwrap();

  assert_eq!(rgb_yuv, rgb_yuva);
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva444p14_rgba_u8_with_source_alpha_passes_through() {
  let (yp, up, vp, ap) = solid_yuva444p_frame_u16(16, 8, 8192, 8192, 8192, 8192);
  let src = Yuva444p14Frame::try_new(&yp, &up, &vp, &ap, 16, 8, 16, 16, 16, 16).unwrap();

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuva444p14>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  yuva444p14_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert_eq!(px[3], 128, "alpha = 8192 >> (14-8) = 128");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva444p14_rgba_u16_native_depth() {
  let (yp, up, vp, ap) = solid_yuva444p_frame_u16(16, 8, 8192, 8192, 8192, 8192);
  let src = Yuva444p14Frame::try_new(&yp, &up, &vp, &ap, 16, 8, 16, 16, 16, 16).unwrap();

  let mut rgba = std::vec![0u16; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuva444p14>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  yuva444p14_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert_eq!(px[3], 8192, "alpha at native depth");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva444p14_with_rgb_alpha_drop_matches_yuv444p14() {
  let (yp, up, vp, ap) = solid_yuva444p_frame_u16(16, 8, 8192, 6000, 10000, 8192);
  let yuv = Yuv444p14Frame::try_new(&yp, &up, &vp, 16, 8, 16, 16, 16).unwrap();
  let yuva = Yuva444p14Frame::try_new(&yp, &up, &vp, &ap, 16, 8, 16, 16, 16, 16).unwrap();

  let mut rgb_yuv = std::vec![0u8; 16 * 8 * 3];
  let mut s_yuv = MixedSinker::<Yuv444p14>::new(16, 8)
    .with_rgb(&mut rgb_yuv)
    .unwrap();
  yuv444p14_to(&yuv, true, ColorMatrix::Bt709, &mut s_yuv).unwrap();

  let mut rgb_yuva = std::vec![0u8; 16 * 8 * 3];
  let mut s_yuva = MixedSinker::<Yuva444p14>::new(16, 8)
    .with_rgb(&mut rgb_yuva)
    .unwrap();
  yuva444p14_to(&yuva, true, ColorMatrix::Bt709, &mut s_yuva).unwrap();

  assert_eq!(rgb_yuv, rgb_yuva);
}

// ---- Yuva422p12 SIMD-vs-scalar parity (Ship 8b‑4) -----------------
//
// Yuva422p12 routes through the BITS-generic `yuv_420p_n_*<12>` row
// kernels via the new yuva420p12 dispatchers. Width 1922 enters and
// exits the main SIMD loop on every backend block size (NEON 16,
// AVX2 32, AVX-512 64) so a bad 12-bit alpha shift, chroma
// duplication, or RGBA interleave on any tier shows up here.

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva422p12_rgba_u8_simd_matches_scalar_with_random_yuva() {
  let w = 1922usize;
  let h = 4usize;
  let mut yp = std::vec![0u16; w * h];
  let mut up = std::vec![0u16; (w / 2) * h];
  let mut vp = std::vec![0u16; (w / 2) * h];
  let mut ap = std::vec![0u16; w * h];
  pseudo_random_u16_low_n_bits(&mut yp, 0xC001_C0DE, 12);
  pseudo_random_u16_low_n_bits(&mut up, 0xCAFE_F00D, 12);
  pseudo_random_u16_low_n_bits(&mut vp, 0xDEAD_BEEF, 12);
  pseudo_random_u16_low_n_bits(&mut ap, 0xA1FA_5EED, 12);
  let src = Yuva422p12Frame::try_new(
    &yp,
    &up,
    &vp,
    &ap,
    w as u32,
    h as u32,
    w as u32,
    (w / 2) as u32,
    (w / 2) as u32,
    w as u32,
  )
  .unwrap();

  for &matrix in &[
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::YCgCo,
  ] {
    for &full_range in &[true, false] {
      let mut rgba_simd = std::vec![0u8; w * h * 4];
      let mut rgba_scalar = std::vec![0u8; w * h * 4];

      let mut s_simd = MixedSinker::<Yuva422p12>::new(w, h)
        .with_rgba(&mut rgba_simd)
        .unwrap();
      yuva422p12_to(&src, full_range, matrix, &mut s_simd).unwrap();

      let mut s_scalar = MixedSinker::<Yuva422p12>::new(w, h)
        .with_rgba(&mut rgba_scalar)
        .unwrap();
      s_scalar.set_simd(false);
      yuva422p12_to(&src, full_range, matrix, &mut s_scalar).unwrap();

      if rgba_simd != rgba_scalar {
        let mismatch = rgba_simd
          .iter()
          .zip(rgba_scalar.iter())
          .position(|(a, b)| a != b)
          .unwrap();
        let pixel = mismatch / 4;
        let channel = ["R", "G", "B", "A"][mismatch % 4];
        panic!(
          "Yuva422p12 RGBA u8 SIMD ≠ scalar at byte {mismatch} (px {pixel} {channel}) for matrix={matrix:?} full_range={full_range}: simd={} scalar={}",
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
fn yuva422p12_rgba_u16_simd_matches_scalar_with_random_yuva() {
  let w = 1922usize;
  let h = 4usize;
  let mut yp = std::vec![0u16; w * h];
  let mut up = std::vec![0u16; (w / 2) * h];
  let mut vp = std::vec![0u16; (w / 2) * h];
  let mut ap = std::vec![0u16; w * h];
  pseudo_random_u16_low_n_bits(&mut yp, 0xC001_C0DE, 12);
  pseudo_random_u16_low_n_bits(&mut up, 0xCAFE_F00D, 12);
  pseudo_random_u16_low_n_bits(&mut vp, 0xDEAD_BEEF, 12);
  pseudo_random_u16_low_n_bits(&mut ap, 0xA1FA_5EED, 12);
  let src = Yuva422p12Frame::try_new(
    &yp,
    &up,
    &vp,
    &ap,
    w as u32,
    h as u32,
    w as u32,
    (w / 2) as u32,
    (w / 2) as u32,
    w as u32,
  )
  .unwrap();

  for &matrix in &[
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::YCgCo,
  ] {
    for &full_range in &[true, false] {
      let mut rgba_simd = std::vec![0u16; w * h * 4];
      let mut rgba_scalar = std::vec![0u16; w * h * 4];

      let mut s_simd = MixedSinker::<Yuva422p12>::new(w, h)
        .with_rgba_u16(&mut rgba_simd)
        .unwrap();
      yuva422p12_to(&src, full_range, matrix, &mut s_simd).unwrap();

      let mut s_scalar = MixedSinker::<Yuva422p12>::new(w, h)
        .with_rgba_u16(&mut rgba_scalar)
        .unwrap();
      s_scalar.set_simd(false);
      yuva422p12_to(&src, full_range, matrix, &mut s_scalar).unwrap();

      if rgba_simd != rgba_scalar {
        let mismatch = rgba_simd
          .iter()
          .zip(rgba_scalar.iter())
          .position(|(a, b)| a != b)
          .unwrap();
        let pixel = mismatch / 4;
        let channel = ["R", "G", "B", "A"][mismatch % 4];
        panic!(
          "Yuva422p12 RGBA u16 SIMD ≠ scalar at element {mismatch} (px {pixel} {channel}) for matrix={matrix:?} full_range={full_range}: simd={} scalar={}",
          rgba_simd[mismatch], rgba_scalar[mismatch]
        );
      }
    }
  }
}

// ---- Yuva444p16 (Ship 8b‑5a — scalar prep) ------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva444p16_rgba_u8_with_source_alpha_passes_through() {
  let (yp, up, vp, ap) = solid_yuva444p_frame_u16(16, 8, 32768, 32768, 32768, 32768);
  let src = Yuva444p16Frame::try_new(&yp, &up, &vp, &ap, 16, 8, 16, 16, 16, 16).unwrap();

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuva444p16>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  yuva444p16_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert_eq!(px[3], 128, "alpha = 32768 >> 8 = 128");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva444p16_rgba_u16_native_depth() {
  let (yp, up, vp, ap) = solid_yuva444p_frame_u16(16, 8, 32768, 32768, 32768, 32768);
  let src = Yuva444p16Frame::try_new(&yp, &up, &vp, &ap, 16, 8, 16, 16, 16, 16).unwrap();

  let mut rgba = std::vec![0u16; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuva444p16>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  yuva444p16_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert_eq!(px[3], 32768, "alpha at native depth");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva444p16_with_rgb_alpha_drop_matches_yuv444p16() {
  let (yp, up, vp, ap) = solid_yuva444p_frame_u16(16, 8, 32768, 24000, 40000, 32768);
  let yuv = Yuv444p16Frame::try_new(&yp, &up, &vp, 16, 8, 16, 16, 16).unwrap();
  let yuva = Yuva444p16Frame::try_new(&yp, &up, &vp, &ap, 16, 8, 16, 16, 16, 16).unwrap();

  let mut rgb_yuv = std::vec![0u8; 16 * 8 * 3];
  let mut s_yuv = MixedSinker::<Yuv444p16>::new(16, 8)
    .with_rgb(&mut rgb_yuv)
    .unwrap();
  yuv444p16_to(&yuv, true, ColorMatrix::Bt709, &mut s_yuv).unwrap();

  let mut rgb_yuva = std::vec![0u8; 16 * 8 * 3];
  let mut s_yuva = MixedSinker::<Yuva444p16>::new(16, 8)
    .with_rgb(&mut rgb_yuva)
    .unwrap();
  yuva444p16_to(&yuva, true, ColorMatrix::Bt709, &mut s_yuva).unwrap();

  assert_eq!(rgb_yuv, rgb_yuva);
}

// ---- Yuva444p16 SIMD-vs-scalar parity (Ship 8b‑5b) ----------------
//
// Yuva444p16 u8 RGBA path now goes through per-arch SIMD wrappers
// (yuv_444p16_to_rgba_with_alpha_src_row across NEON / SSE4.1 / AVX2 /
// AVX-512 / wasm simd128). Width 1922 enters and exits each backend's
// main loop (NEON 16, SSE4.1 16, AVX2 32, AVX-512 64) plus a scalar
// tail.

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva444p16_rgba_u8_simd_matches_scalar_with_random_yuva() {
  let w = 1922usize;
  let h = 4usize;
  let mut yp = std::vec![0u16; w * h];
  let mut up = std::vec![0u16; w * h];
  let mut vp = std::vec![0u16; w * h];
  let mut ap = std::vec![0u16; w * h];
  // 16-bit input is full-range u16 (no bits_mask).
  pseudo_random_u16_low_n_bits(&mut yp, 0xC001_C0DE, 16);
  pseudo_random_u16_low_n_bits(&mut up, 0xCAFE_F00D, 16);
  pseudo_random_u16_low_n_bits(&mut vp, 0xDEAD_BEEF, 16);
  pseudo_random_u16_low_n_bits(&mut ap, 0xA1FA_5EED, 16);
  let src = Yuva444p16Frame::try_new(
    &yp, &up, &vp, &ap, w as u32, h as u32, w as u32, w as u32, w as u32, w as u32,
  )
  .unwrap();

  for &matrix in &[
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::YCgCo,
  ] {
    for &full_range in &[true, false] {
      let mut rgba_simd = std::vec![0u8; w * h * 4];
      let mut rgba_scalar = std::vec![0u8; w * h * 4];

      let mut s_simd = MixedSinker::<Yuva444p16>::new(w, h)
        .with_rgba(&mut rgba_simd)
        .unwrap();
      yuva444p16_to(&src, full_range, matrix, &mut s_simd).unwrap();

      let mut s_scalar = MixedSinker::<Yuva444p16>::new(w, h)
        .with_rgba(&mut rgba_scalar)
        .unwrap();
      s_scalar.set_simd(false);
      yuva444p16_to(&src, full_range, matrix, &mut s_scalar).unwrap();

      if rgba_simd != rgba_scalar {
        let mismatch = rgba_simd
          .iter()
          .zip(rgba_scalar.iter())
          .position(|(a, b)| a != b)
          .unwrap();
        let pixel = mismatch / 4;
        let channel = ["R", "G", "B", "A"][mismatch % 4];
        panic!(
          "Yuva444p16 RGBA u8 SIMD ≠ scalar at byte {mismatch} (px {pixel} {channel}) for matrix={matrix:?} full_range={full_range}: simd={} scalar={}",
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
fn yuva444p16_rgba_u16_simd_matches_scalar_with_random_yuva() {
  // Yuva444p16 u16 path uses the i64 chroma kernel family. Block
  // sizes per backend: 8 / 8 / 16 / 32 / 8 px (NEON / SSE4.1 / AVX2 /
  // AVX-512 / wasm simd128). Width 1922 enters and exits every main
  // loop plus a scalar tail.
  let w = 1922usize;
  let h = 4usize;
  let mut yp = std::vec![0u16; w * h];
  let mut up = std::vec![0u16; w * h];
  let mut vp = std::vec![0u16; w * h];
  let mut ap = std::vec![0u16; w * h];
  pseudo_random_u16_low_n_bits(&mut yp, 0xC001_C0DE, 16);
  pseudo_random_u16_low_n_bits(&mut up, 0xCAFE_F00D, 16);
  pseudo_random_u16_low_n_bits(&mut vp, 0xDEAD_BEEF, 16);
  pseudo_random_u16_low_n_bits(&mut ap, 0xA1FA_5EED, 16);
  let src = Yuva444p16Frame::try_new(
    &yp, &up, &vp, &ap, w as u32, h as u32, w as u32, w as u32, w as u32, w as u32,
  )
  .unwrap();

  for &matrix in &[
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::YCgCo,
  ] {
    for &full_range in &[true, false] {
      let mut rgba_simd = std::vec![0u16; w * h * 4];
      let mut rgba_scalar = std::vec![0u16; w * h * 4];

      let mut s_simd = MixedSinker::<Yuva444p16>::new(w, h)
        .with_rgba_u16(&mut rgba_simd)
        .unwrap();
      yuva444p16_to(&src, full_range, matrix, &mut s_simd).unwrap();

      let mut s_scalar = MixedSinker::<Yuva444p16>::new(w, h)
        .with_rgba_u16(&mut rgba_scalar)
        .unwrap();
      s_scalar.set_simd(false);
      yuva444p16_to(&src, full_range, matrix, &mut s_scalar).unwrap();

      if rgba_simd != rgba_scalar {
        let mismatch = rgba_simd
          .iter()
          .zip(rgba_scalar.iter())
          .position(|(a, b)| a != b)
          .unwrap();
        let pixel = mismatch / 4;
        let channel = ["R", "G", "B", "A"][mismatch % 4];
        panic!(
          "Yuva444p16 RGBA u16 SIMD ≠ scalar at element {mismatch} (px {pixel} {channel}) for matrix={matrix:?} full_range={full_range}: simd={} scalar={}",
          rgba_simd[mismatch], rgba_scalar[mismatch]
        );
      }
    }
  }
}

// ---- Yuva444p (8-bit) tests (Ship 8b‑6) ---------------------------

fn solid_yuva444p_frame(
  width: u32,
  height: u32,
  y: u8,
  u: u8,
  v: u8,
  a: u8,
) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
  let w = width as usize;
  let h = height as usize;
  // 4:4:4: chroma full-width × full-height; alpha 1:1 with Y.
  (
    std::vec![y; w * h],
    std::vec![u; w * h],
    std::vec![v; w * h],
    std::vec![a; w * h],
  )
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva444p_rgba_u8_with_source_alpha_passes_through() {
  let (yp, up, vp, ap) = solid_yuva444p_frame(16, 8, 128, 128, 128, 128);
  let src = Yuva444pFrame::try_new(&yp, &up, &vp, &ap, 16, 8, 16, 16, 16, 16).unwrap();

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuva444p>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  yuva444p_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(128) <= 1, "got {px:?}");
    assert_eq!(px[3], 128, "alpha pass-through");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva444p_with_rgb_alpha_drop_matches_yuv444p() {
  let (yp, up, vp, ap) = solid_yuva444p_frame(16, 8, 180, 60, 200, 200);
  let yuv = Yuv444pFrame::try_new(&yp, &up, &vp, 16, 8, 16, 16, 16).unwrap();
  let yuva = Yuva444pFrame::try_new(&yp, &up, &vp, &ap, 16, 8, 16, 16, 16, 16).unwrap();

  let mut rgb_yuv = std::vec![0u8; 16 * 8 * 3];
  let mut s_yuv = MixedSinker::<Yuv444p>::new(16, 8)
    .with_rgb(&mut rgb_yuv)
    .unwrap();
  yuv444p_to(&yuv, true, ColorMatrix::Bt709, &mut s_yuv).unwrap();

  let mut rgb_yuva = std::vec![0u8; 16 * 8 * 3];
  let mut s_yuva = MixedSinker::<Yuva444p>::new(16, 8)
    .with_rgb(&mut rgb_yuva)
    .unwrap();
  yuva444p_to(&yuva, true, ColorMatrix::Bt709, &mut s_yuva).unwrap();

  assert_eq!(rgb_yuv, rgb_yuva);
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva444p_rgba_simd_matches_scalar_with_random_yuva() {
  // Yuva444p u8 RGBA goes through per-arch SIMD wrappers
  // (yuv_444_to_rgba_with_alpha_src_row across NEON / SSE4.1 / AVX2 /
  // AVX-512 / wasm simd128). Width 1922 enters and exits each
  // backend's main loop (NEON 16, SSE4.1 16, AVX2 32, AVX-512 64,
  // wasm 16) plus a scalar tail.
  let w = 1922usize;
  let h = 4usize;
  let mut yp = std::vec![0u8; w * h];
  let mut up = std::vec![0u8; w * h];
  let mut vp = std::vec![0u8; w * h];
  let mut ap = std::vec![0u8; w * h];
  pseudo_random_u8(&mut yp, 0xC001_C0DE);
  pseudo_random_u8(&mut up, 0xCAFE_F00D);
  pseudo_random_u8(&mut vp, 0xDEAD_BEEF);
  pseudo_random_u8(&mut ap, 0xA1FA_5EED);
  let src = Yuva444pFrame::try_new(
    &yp, &up, &vp, &ap, w as u32, h as u32, w as u32, w as u32, w as u32, w as u32,
  )
  .unwrap();

  for &matrix in &[
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::YCgCo,
  ] {
    for &full_range in &[true, false] {
      let mut rgba_simd = std::vec![0u8; w * h * 4];
      let mut rgba_scalar = std::vec![0u8; w * h * 4];

      let mut s_simd = MixedSinker::<Yuva444p>::new(w, h)
        .with_rgba(&mut rgba_simd)
        .unwrap();
      yuva444p_to(&src, full_range, matrix, &mut s_simd).unwrap();

      let mut s_scalar = MixedSinker::<Yuva444p>::new(w, h)
        .with_rgba(&mut rgba_scalar)
        .unwrap();
      s_scalar.set_simd(false);
      yuva444p_to(&src, full_range, matrix, &mut s_scalar).unwrap();

      if rgba_simd != rgba_scalar {
        let mismatch = rgba_simd
          .iter()
          .zip(rgba_scalar.iter())
          .position(|(a, b)| a != b)
          .unwrap();
        let pixel = mismatch / 4;
        let channel = ["R", "G", "B", "A"][mismatch % 4];
        panic!(
          "Yuva444p RGBA u8 SIMD ≠ scalar at byte {mismatch} (px {pixel} {channel}) for matrix={matrix:?} full_range={full_range}: simd={} scalar={}",
          rgba_simd[mismatch], rgba_scalar[mismatch]
        );
      }
    }
  }
}
