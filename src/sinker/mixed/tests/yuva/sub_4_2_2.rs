use super::*;

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

pub(super) fn solid_yuva422p_frame_u16(
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
