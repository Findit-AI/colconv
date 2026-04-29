use super::*;

// ---- Yuv420p10 --------------------------------------------------------

pub(in crate::sinker::mixed::tests) fn solid_yuv420p10_frame(
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
fn yuv420p10_rgb_u8_only_gray_is_gray() {
  // 10-bit mid-gray: Y=512, UV=512 → 8-bit RGB ≈ 128 on every channel.
  let (yp, up, vp) = solid_yuv420p10_frame(16, 8, 512, 512, 512);
  let src = Yuv420p10Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yuv420p10>::new(16, 8)
    .with_rgb(&mut rgb)
    .unwrap();
  yuv420p10_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn yuv420p10_rgb_u16_only_native_depth_gray() {
  // Same mid-gray frame → u16 RGB output in native 10-bit depth, so
  // each channel should be ≈ 512 (the 10-bit mid).
  let (yp, up, vp) = solid_yuv420p10_frame(16, 8, 512, 512, 512);
  let src = Yuv420p10Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgb = std::vec![0u16; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yuv420p10>::new(16, 8)
    .with_rgb_u16(&mut rgb)
    .unwrap();
  yuv420p10_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgb.chunks(3) {
    assert!(px[0].abs_diff(512) <= 1, "got {px:?}");
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    // Upper 6 bits of each u16 must be zero — 10-bit convention.
    assert!(px[0] <= 1023);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p10_rgb_u8_and_u16_both_populated() {
  // 10-bit full-range white: Y=1023, UV=512. Both buffers should
  // fill with their respective "white" values (255 for u8, 1023 for
  // u16) in the same call.
  let (yp, up, vp) = solid_yuv420p10_frame(16, 8, 1023, 512, 512);
  let src = Yuv420p10Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgb_u8 = std::vec![0u8; 16 * 8 * 3];
  let mut rgb_u16 = std::vec![0u16; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yuv420p10>::new(16, 8)
    .with_rgb(&mut rgb_u8)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16)
    .unwrap();
  yuv420p10_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  assert!(rgb_u8.iter().all(|&c| c == 255));
  assert!(rgb_u16.iter().all(|&c| c == 1023));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p10_luma_downshifts_to_8bit() {
  // Y=512 at 10 bits → 512 >> 2 = 128 at 8 bits.
  let (yp, up, vp) = solid_yuv420p10_frame(16, 8, 512, 512, 512);
  let src = Yuv420p10Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut luma = std::vec![0u8; 16 * 8];
  let mut sink = MixedSinker::<Yuv420p10>::new(16, 8)
    .with_luma(&mut luma)
    .unwrap();
  yuv420p10_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  assert!(luma.iter().all(|&l| l == 128));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p10_hsv_from_gray_is_zero_hue_zero_sat() {
  // HSV derived from the internal u8 RGB scratch: neutral gray →
  // H=0, S=0, V≈128. Exercises the "HSV without RGB" scratch path
  // on the 10-bit source.
  let (yp, up, vp) = solid_yuv420p10_frame(16, 8, 512, 512, 512);
  let src = Yuv420p10Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut h = std::vec![0xFFu8; 16 * 8];
  let mut s = std::vec![0xFFu8; 16 * 8];
  let mut v = std::vec![0xFFu8; 16 * 8];
  let mut sink = MixedSinker::<Yuv420p10>::new(16, 8)
    .with_hsv(&mut h, &mut s, &mut v)
    .unwrap();
  yuv420p10_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  assert!(h.iter().all(|&b| b == 0));
  assert!(s.iter().all(|&b| b == 0));
  assert!(v.iter().all(|&b| b.abs_diff(128) <= 1));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p10_rgb_u16_too_short_returns_err() {
  let mut rgb = std::vec![0u16; 10]; // Way too small.
  let err = MixedSinker::<Yuv420p10>::new(16, 8)
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
fn yuv420p10_with_simd_false_matches_with_simd_true() {
  // The SIMD toggle exercises scalar-vs-SIMD dispatch. Both paths
  // must produce byte-identical results on both outputs.
  let (yp, up, vp) = solid_yuv420p10_frame(64, 16, 600, 400, 700);
  let src = Yuv420p10Frame::new(&yp, &up, &vp, 64, 16, 64, 32, 32);

  let mut rgb_scalar = std::vec![0u8; 64 * 16 * 3];
  let mut rgb_u16_scalar = std::vec![0u16; 64 * 16 * 3];
  let mut s_scalar = MixedSinker::<Yuv420p10>::new(64, 16)
    .with_simd(false)
    .with_rgb(&mut rgb_scalar)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16_scalar)
    .unwrap();
  yuv420p10_to(&src, false, ColorMatrix::Bt709, &mut s_scalar).unwrap();

  let mut rgb_simd = std::vec![0u8; 64 * 16 * 3];
  let mut rgb_u16_simd = std::vec![0u16; 64 * 16 * 3];
  let mut s_simd = MixedSinker::<Yuv420p10>::new(64, 16)
    .with_rgb(&mut rgb_simd)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16_simd)
    .unwrap();
  yuv420p10_to(&src, false, ColorMatrix::Bt709, &mut s_simd).unwrap();

  assert_eq!(rgb_scalar, rgb_simd);
  assert_eq!(rgb_u16_scalar, rgb_u16_simd);
}

// ---- Yuv420p10 RGBA (Ship 8 Tranche 5b) -------------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p10_rgba_u8_only_gray_with_opaque_alpha() {
  // 10-bit mid-gray → 8-bit RGBA ≈ (128, 128, 128, 255) per pixel.
  let (yp, up, vp) = solid_yuv420p10_frame(16, 8, 512, 512, 512);
  let src = Yuv420p10Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuv420p10>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  yuv420p10_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn yuv420p10_rgba_u16_only_native_depth_gray_with_opaque_alpha() {
  // 10-bit mid-gray → u16 RGBA: each color element ≈ 512, alpha = 1023.
  let (yp, up, vp) = solid_yuv420p10_frame(16, 8, 512, 512, 512);
  let src = Yuv420p10Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgba = std::vec![0u16; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuv420p10>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  yuv420p10_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(512) <= 1, "got {px:?}");
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert_eq!(px[3], 1023, "alpha must equal (1 << 10) - 1");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p10_with_rgb_and_with_rgba_produce_byte_identical_rgb_bytes() {
  // Strategy A: when both rgb and rgba are attached, the rgb buffer is
  // populated by the RGB kernel and the rgba buffer is populated via a
  // cheap expand pass. RGB triples must be byte-identical to the
  // standalone RGB-only run.
  let (yp, up, vp) = solid_yuv420p10_frame(64, 16, 600, 400, 700);
  let src = Yuv420p10Frame::new(&yp, &up, &vp, 64, 16, 64, 32, 32);

  let mut rgb_solo = std::vec![0u8; 64 * 16 * 3];
  let mut s_solo = MixedSinker::<Yuv420p10>::new(64, 16)
    .with_rgb(&mut rgb_solo)
    .unwrap();
  yuv420p10_to(&src, true, ColorMatrix::Bt709, &mut s_solo).unwrap();

  let mut rgb_combined = std::vec![0u8; 64 * 16 * 3];
  let mut rgba = std::vec![0u8; 64 * 16 * 4];
  let mut s_combined = MixedSinker::<Yuv420p10>::new(64, 16)
    .with_rgb(&mut rgb_combined)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap();
  yuv420p10_to(&src, true, ColorMatrix::Bt709, &mut s_combined).unwrap();

  assert_eq!(rgb_solo, rgb_combined, "RGB bytes must match across runs");
  for (rgb_px, rgba_px) in rgb_combined.chunks(3).zip(rgba.chunks(4)) {
    assert_eq!(rgb_px[0], rgba_px[0]);
    assert_eq!(rgb_px[1], rgba_px[1]);
    assert_eq!(rgb_px[2], rgba_px[2]);
    assert_eq!(rgba_px[3], 0xFF);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p10_with_rgb_u16_and_with_rgba_u16_produce_byte_identical_rgb_elems() {
  // Strategy A on the u16 path: rgb_u16 buffer populated by the u16 RGB
  // kernel, rgba_u16 fanned out via expand_rgb_u16_to_rgba_u16_row<10>.
  let (yp, up, vp) = solid_yuv420p10_frame(64, 16, 600, 400, 700);
  let src = Yuv420p10Frame::new(&yp, &up, &vp, 64, 16, 64, 32, 32);

  let mut rgb_solo = std::vec![0u16; 64 * 16 * 3];
  let mut s_solo = MixedSinker::<Yuv420p10>::new(64, 16)
    .with_rgb_u16(&mut rgb_solo)
    .unwrap();
  yuv420p10_to(&src, true, ColorMatrix::Bt709, &mut s_solo).unwrap();

  let mut rgb_combined = std::vec![0u16; 64 * 16 * 3];
  let mut rgba = std::vec![0u16; 64 * 16 * 4];
  let mut s_combined = MixedSinker::<Yuv420p10>::new(64, 16)
    .with_rgb_u16(&mut rgb_combined)
    .unwrap()
    .with_rgba_u16(&mut rgba)
    .unwrap();
  yuv420p10_to(&src, true, ColorMatrix::Bt709, &mut s_combined).unwrap();

  assert_eq!(
    rgb_solo, rgb_combined,
    "RGB u16 elements must match across runs"
  );
  for (rgb_px, rgba_px) in rgb_combined.chunks(3).zip(rgba.chunks(4)) {
    assert_eq!(rgb_px[0], rgba_px[0]);
    assert_eq!(rgb_px[1], rgba_px[1]);
    assert_eq!(rgb_px[2], rgba_px[2]);
    assert_eq!(rgba_px[3], 1023, "alpha = (1 << 10) - 1");
  }
}

#[test]
fn yuv420p10_rgba_too_short_returns_err() {
  let mut rgba = std::vec![0u8; 10];
  let err = MixedSinker::<Yuv420p10>::new(16, 8)
    .with_rgba(&mut rgba)
    .err()
    .expect("expected RgbaBufferTooShort");
  assert!(matches!(err, MixedSinkerError::RgbaBufferTooShort { .. }));
}

#[test]
fn yuv420p10_rgba_u16_too_short_returns_err() {
  let mut rgba = std::vec![0u16; 10];
  let err = MixedSinker::<Yuv420p10>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .err()
    .expect("expected RgbaU16BufferTooShort");
  assert!(matches!(
    err,
    MixedSinkerError::RgbaU16BufferTooShort { .. }
  ));
}

// ---- P010 --------------------------------------------------------------
//
// Semi-planar 10-bit, high-bit-packed (samples in high 10 of each
// u16). Mirrors the Yuv420p10 test shape but with UV interleaved.

fn solid_p010_frame(
  width: u32,
  height: u32,
  y_10bit: u16,
  u_10bit: u16,
  v_10bit: u16,
) -> (Vec<u16>, Vec<u16>) {
  let w = width as usize;
  let h = height as usize;
  let cw = w / 2;
  let ch = h / 2;
  // Shift into the high 10 bits (P010 packing).
  let y = std::vec![y_10bit << 6; w * h];
  let uv: Vec<u16> = (0..cw * ch)
    .flat_map(|_| [u_10bit << 6, v_10bit << 6])
    .collect();
  (y, uv)
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p010_rgb_u8_only_gray_is_gray() {
  // 10-bit mid-gray Y=512, UV=512 → ~128 u8 RGB across the frame.
  let (yp, uvp) = solid_p010_frame(16, 8, 512, 512, 512);
  let src = P010Frame::new(&yp, &uvp, 16, 8, 16, 16);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<P010>::new(16, 8).with_rgb(&mut rgb).unwrap();
  p010_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn p010_rgb_u16_only_native_depth_gray() {
  // Output u16 is yuv420p10le-packed (10-bit in low 10) even though
  // the input is P010-packed.
  let (yp, uvp) = solid_p010_frame(16, 8, 512, 512, 512);
  let src = P010Frame::new(&yp, &uvp, 16, 8, 16, 16);

  let mut rgb = std::vec![0u16; 16 * 8 * 3];
  let mut sink = MixedSinker::<P010>::new(16, 8)
    .with_rgb_u16(&mut rgb)
    .unwrap();
  p010_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgb.chunks(3) {
    assert!(px[0].abs_diff(512) <= 1, "got {px:?}");
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert!(
      px[0] <= 1023,
      "output must stay within 10-bit low-packed range"
    );
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p010_rgb_u8_and_u16_both_populated() {
  // 10-bit full-range white: Y=1023, UV=512. Both buffers fill in
  // one call.
  let (yp, uvp) = solid_p010_frame(16, 8, 1023, 512, 512);
  let src = P010Frame::new(&yp, &uvp, 16, 8, 16, 16);

  let mut rgb_u8 = std::vec![0u8; 16 * 8 * 3];
  let mut rgb_u16 = std::vec![0u16; 16 * 8 * 3];
  let mut sink = MixedSinker::<P010>::new(16, 8)
    .with_rgb(&mut rgb_u8)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16)
    .unwrap();
  p010_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  assert!(rgb_u8.iter().all(|&c| c == 255));
  assert!(rgb_u16.iter().all(|&c| c == 1023));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p010_luma_downshifts_to_8bit() {
  // Y=512 at 10 bits, P010-packed (0x8000). After >> 8, the 8-bit
  // luma is 0x80 = 128.
  let (yp, uvp) = solid_p010_frame(16, 8, 512, 512, 512);
  let src = P010Frame::new(&yp, &uvp, 16, 8, 16, 16);

  let mut luma = std::vec![0u8; 16 * 8];
  let mut sink = MixedSinker::<P010>::new(16, 8)
    .with_luma(&mut luma)
    .unwrap();
  p010_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  assert!(luma.iter().all(|&l| l == 128));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p010_matches_yuv420p10_mixed_sinker_with_shifted_samples() {
  // Logical equivalence: same samples fed through the two formats
  // (low-packed as yuv420p10, high-packed as P010) must produce
  // byte-identical u8 RGB.
  let w = 16u32;
  let h = 8u32;
  let y = 600u16;
  let u = 400u16;
  let v = 700u16;

  let (yp_p10, up_p10, vp_p10) = solid_yuv420p10_frame(w, h, y, u, v);
  let src_p10 = Yuv420p10Frame::new(&yp_p10, &up_p10, &vp_p10, w, h, w, w / 2, w / 2);

  let (yp_p010, uvp_p010) = solid_p010_frame(w, h, y, u, v);
  let src_p010 = P010Frame::new(&yp_p010, &uvp_p010, w, h, w, w);

  let mut rgb_yuv = std::vec![0u8; (w * h * 3) as usize];
  let mut rgb_p010 = std::vec![0u8; (w * h * 3) as usize];
  let mut s_yuv = MixedSinker::<Yuv420p10>::new(w as usize, h as usize)
    .with_rgb(&mut rgb_yuv)
    .unwrap();
  let mut s_p010 = MixedSinker::<P010>::new(w as usize, h as usize)
    .with_rgb(&mut rgb_p010)
    .unwrap();
  yuv420p10_to(&src_p10, true, ColorMatrix::Bt709, &mut s_yuv).unwrap();
  p010_to(&src_p010, true, ColorMatrix::Bt709, &mut s_p010).unwrap();
  assert_eq!(rgb_yuv, rgb_p010);
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p010_rgb_u16_too_short_returns_err() {
  let mut rgb = std::vec![0u16; 10];
  let err = MixedSinker::<P010>::new(16, 8)
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
fn p010_with_simd_false_matches_with_simd_true() {
  // Stubs delegate to scalar so simd=true and simd=false produce
  // byte-identical output for now. Real SIMD backends will replace
  // the stubs — equivalence is preserved by design.
  let (yp, uvp) = solid_p010_frame(64, 16, 600, 400, 700);
  let src = P010Frame::new(&yp, &uvp, 64, 16, 64, 64);

  let mut rgb_scalar = std::vec![0u8; 64 * 16 * 3];
  let mut rgb_u16_scalar = std::vec![0u16; 64 * 16 * 3];
  let mut s_scalar = MixedSinker::<P010>::new(64, 16)
    .with_simd(false)
    .with_rgb(&mut rgb_scalar)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16_scalar)
    .unwrap();
  p010_to(&src, false, ColorMatrix::Bt709, &mut s_scalar).unwrap();

  let mut rgb_simd = std::vec![0u8; 64 * 16 * 3];
  let mut rgb_u16_simd = std::vec![0u16; 64 * 16 * 3];
  let mut s_simd = MixedSinker::<P010>::new(64, 16)
    .with_rgb(&mut rgb_simd)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16_simd)
    .unwrap();
  p010_to(&src, false, ColorMatrix::Bt709, &mut s_simd).unwrap();

  assert_eq!(rgb_scalar, rgb_simd);
  assert_eq!(rgb_u16_scalar, rgb_u16_simd);
}

// ---- P010 RGBA (Ship 8 Tranche 5b) ------------------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p010_rgba_u16_only_native_depth_gray_with_opaque_alpha() {
  // P010 mid-gray (10-bit values shifted into the high 10): Y/U/V = 512 << 6.
  // Output u16 RGBA: each color element ≈ 512, alpha = 1023.
  let (yp, uvp) = solid_p010_frame(16, 8, 512, 512, 512);
  let src = P010Frame::new(&yp, &uvp, 16, 8, 16, 16);

  let mut rgba = std::vec![0u16; 16 * 8 * 4];
  let mut sink = MixedSinker::<P010>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  p010_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(512) <= 1, "got {px:?}");
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert_eq!(px[3], 1023, "alpha = (1 << 10) - 1");
  }
}
