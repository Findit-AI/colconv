#[cfg(feature = "yuv-planar")]
use super::planar_other_8bit_9bit::{
  solid_yuv422p_n_frame, solid_yuv440p_n_frame, solid_yuv444p_n_frame,
};
use super::*;

// ---- P210 / P212 / P216 / P410 / P412 / P416 sanity tests --------------

/// 4:2:2 P-family solid frame helper. UV is `width` u16 elements per
/// row, **full-height** chroma. All samples are high-bit-packed
/// (shifted left by `16 - bits`).
fn solid_p2x0_frame(
  width: u32,
  height: u32,
  bits: u32,
  y_value: u16,
  u_value: u16,
  v_value: u16,
) -> (Vec<u16>, Vec<u16>) {
  let w = width as usize;
  let h = height as usize;
  let cw = w / 2;
  let shift = 16 - bits;
  let y = std::vec![y_value << shift; w * h];
  // 4:2:2: full-height chroma, half-width x 2 elements per pair.
  let uv: Vec<u16> = (0..cw * h)
    .flat_map(|_| [u_value << shift, v_value << shift])
    .collect();
  (y, uv)
}

/// 4:4:4 P-family solid frame helper. UV is `2 * width` u16 elements
/// per row, **full-height** chroma (one pair per pixel).
fn solid_p4x0_frame(
  width: u32,
  height: u32,
  bits: u32,
  y_value: u16,
  u_value: u16,
  v_value: u16,
) -> (Vec<u16>, Vec<u16>) {
  let w = width as usize;
  let h = height as usize;
  let shift = 16 - bits;
  let y = std::vec![y_value << shift; w * h];
  // 4:4:4: full-height x full-width x 2 elements per pair.
  let uv: Vec<u16> = (0..w * h)
    .flat_map(|_| [u_value << shift, v_value << shift])
    .collect();
  (y, uv)
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p210_gray_to_gray() {
  let (yp, uvp) = solid_p2x0_frame(16, 8, 10, 512, 512, 512);
  let src = P210Frame::new(&yp, &uvp, 16, 8, 16, 16);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<P210>::new(16, 8).with_rgb(&mut rgb).unwrap();
  p210_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();

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
fn p212_gray_to_gray() {
  let (yp, uvp) = solid_p2x0_frame(16, 8, 12, 2048, 2048, 2048);
  let src = P212Frame::new(&yp, &uvp, 16, 8, 16, 16);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<P212>::new(16, 8).with_rgb(&mut rgb).unwrap();
  p212_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();

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
fn p212_rgba_u8_only_gray_with_opaque_alpha() {
  // P212 mid-gray (12-bit values shifted into the high 12): Y/U/V = 2048 << 4.
  // Output 8-bit RGBA ≈ (128, 128, 128, 255).
  let (yp, uvp) = solid_p2x0_frame(16, 8, 12, 2048, 2048, 2048);
  let src = P212Frame::new(&yp, &uvp, 16, 8, 16, 16);

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<P212>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  p212_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();

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
fn p212_rgba_u16_only_native_depth_gray_with_opaque_alpha() {
  // P212 mid-gray → u16 RGBA: each color element ≈ 2048 (low-bit-packed),
  // alpha = (1 << 12) - 1 = 4095.
  let (yp, uvp) = solid_p2x0_frame(16, 8, 12, 2048, 2048, 2048);
  let src = P212Frame::new(&yp, &uvp, 16, 8, 16, 16);

  let mut rgba = std::vec![0u16; 16 * 8 * 4];
  let mut sink = MixedSinker::<P212>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  p212_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();

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
fn p216_gray_to_gray_u16() {
  let (yp, uvp) = solid_p2x0_frame(16, 8, 16, 32768, 32768, 32768);
  let src = P216Frame::new(&yp, &uvp, 16, 8, 16, 16);

  let mut rgb_u8 = std::vec![0u8; 16 * 8 * 3];
  let mut rgb_u16 = std::vec![0u16; 16 * 8 * 3];
  let mut sink = MixedSinker::<P216>::new(16, 8)
    .with_rgb(&mut rgb_u8)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16)
    .unwrap();
  p216_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();

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
fn p216_rgba_u8_only_gray_with_opaque_alpha() {
  // P216 mid-gray (16-bit, no shift): Y/U/V = 32768. Output 8-bit RGBA
  // ≈ (128, 128, 128, 255).
  let (yp, uvp) = solid_p2x0_frame(16, 8, 16, 32768, 32768, 32768);
  let src = P216Frame::new(&yp, &uvp, 16, 8, 16, 16);

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<P216>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  p216_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();

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
fn p216_rgba_u16_only_native_depth_gray_with_opaque_alpha() {
  // 16-bit mid-gray → u16 RGBA: each color element ≈ 32768, alpha = 0xFFFF.
  // Covers the 16-bit dedicated kernel family (no Q15 downshift).
  let (yp, uvp) = solid_p2x0_frame(16, 8, 16, 32768, 32768, 32768);
  let src = P216Frame::new(&yp, &uvp, 16, 8, 16, 16);

  let mut rgba = std::vec![0u16; 16 * 8 * 4];
  let mut sink = MixedSinker::<P216>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  p216_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(32768) <= 256, "got {px:?}");
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert_eq!(px[3], 0xFFFF, "alpha must equal 0xFFFF");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p410_gray_to_gray() {
  // 4:4:4: uv_stride = 2 * width = 32 (16 pairs x 2 elements).
  let (yp, uvp) = solid_p4x0_frame(16, 8, 10, 512, 512, 512);
  let src = P410Frame::new(&yp, &uvp, 16, 8, 16, 32);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<P410>::new(16, 8).with_rgb(&mut rgb).unwrap();
  p410_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();

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
fn p412_gray_to_gray() {
  let (yp, uvp) = solid_p4x0_frame(16, 8, 12, 2048, 2048, 2048);
  let src = P412Frame::new(&yp, &uvp, 16, 8, 16, 32);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<P412>::new(16, 8).with_rgb(&mut rgb).unwrap();
  p412_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();

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
fn p412_rgba_u8_only_gray_with_opaque_alpha() {
  // P412 mid-gray (12-bit values shifted into the high 12): Y/U/V = 2048 << 4.
  // Output 8-bit RGBA ≈ (128, 128, 128, 255).
  let (yp, uvp) = solid_p4x0_frame(16, 8, 12, 2048, 2048, 2048);
  let src = P412Frame::new(&yp, &uvp, 16, 8, 16, 32);

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<P412>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  p412_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();

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
fn p412_rgba_u16_only_native_depth_gray_with_opaque_alpha() {
  // P412 mid-gray → u16 RGBA: each color element ≈ 2048 (low-bit-packed),
  // alpha = (1 << 12) - 1 = 4095.
  let (yp, uvp) = solid_p4x0_frame(16, 8, 12, 2048, 2048, 2048);
  let src = P412Frame::new(&yp, &uvp, 16, 8, 16, 32);

  let mut rgba = std::vec![0u16; 16 * 8 * 4];
  let mut sink = MixedSinker::<P412>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  p412_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();

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
fn p416_gray_to_gray_u16() {
  let (yp, uvp) = solid_p4x0_frame(16, 8, 16, 32768, 32768, 32768);
  let src = P416Frame::new(&yp, &uvp, 16, 8, 16, 32);

  let mut rgb_u8 = std::vec![0u8; 16 * 8 * 3];
  let mut rgb_u16 = std::vec![0u16; 16 * 8 * 3];
  let mut sink = MixedSinker::<P416>::new(16, 8)
    .with_rgb(&mut rgb_u8)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16)
    .unwrap();
  p416_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();

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
fn p416_rgba_u8_only_gray_with_opaque_alpha() {
  // P416 mid-gray (16-bit, no shift): Y/U/V = 32768. Output 8-bit RGBA
  // ≈ (128, 128, 128, 255).
  let (yp, uvp) = solid_p4x0_frame(16, 8, 16, 32768, 32768, 32768);
  let src = P416Frame::new(&yp, &uvp, 16, 8, 16, 32);

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<P416>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  p416_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();

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
fn p416_rgba_u16_only_native_depth_gray_with_opaque_alpha() {
  // 16-bit mid-gray → u16 RGBA: each color element ≈ 32768, alpha = 0xFFFF.
  // Covers the 16-bit dedicated kernel family (no Q15 downshift).
  let (yp, uvp) = solid_p4x0_frame(16, 8, 16, 32768, 32768, 32768);
  let src = P416Frame::new(&yp, &uvp, 16, 8, 16, 32);

  let mut rgba = std::vec![0u16; 16 * 8 * 4];
  let mut sink = MixedSinker::<P416>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  p416_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(32768) <= 256, "got {px:?}");
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert_eq!(px[3], 0xFFFF, "alpha must equal 0xFFFF");
  }
}

// ---- Walker-level SIMD-vs-scalar equivalence for P410 (4:4:4 Pn) ------
//
// P410 is the only new format in Ship 7 that ships a genuinely new
// SIMD kernel family (`p_n_444_to_rgb_*<BITS>`). Validate the
// walker against scalar with non-neutral chroma and tail widths.
// P210/P212/P216 reuse 4:2:0 P-family kernels (already covered by
// earlier ships' tests).

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p410_walker_simd_matches_scalar_with_random_chroma() {
  let w = 1922u32; // forces tail handling on every backend
  let h = 4u32;
  let mut yp = std::vec![0u16; (w * h) as usize];
  let mut uvp = std::vec![0u16; (2 * w * h) as usize];

  // Seed pseudo-random samples in the high 10 bits.
  let mut state: u32 = 0x1111_2222;
  for s in yp.iter_mut().chain(uvp.iter_mut()) {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *s = ((state & 0x3FF) as u16) << 6;
  }

  let src = P410Frame::new(&yp, &uvp, w, h, w, 2 * w);

  for &full_range in &[true, false] {
    let mut rgb_simd = std::vec![0u8; (w * h * 3) as usize];
    let mut rgb_scalar = std::vec![0u8; (w * h * 3) as usize];
    let mut rgb_u16_simd = std::vec![0u16; (w * h * 3) as usize];
    let mut rgb_u16_scalar = std::vec![0u16; (w * h * 3) as usize];

    let mut s_simd = MixedSinker::<P410>::new(w as usize, h as usize)
      .with_rgb(&mut rgb_simd)
      .unwrap()
      .with_rgb_u16(&mut rgb_u16_simd)
      .unwrap();
    p410_to(
      &src,
      full_range,
      s_simd.set_kernel_matrix(KernelMatrix::Bt2020Ncl),
    )
    .unwrap();

    let mut s_scalar = MixedSinker::<P410>::new(w as usize, h as usize)
      .with_rgb(&mut rgb_scalar)
      .unwrap()
      .with_rgb_u16(&mut rgb_u16_scalar)
      .unwrap();
    s_scalar.set_simd(false);
    p410_to(
      &src,
      full_range,
      s_scalar.set_kernel_matrix(KernelMatrix::Bt2020Ncl),
    )
    .unwrap();

    assert_eq!(rgb_simd, rgb_scalar, "P410 SIMD u8 ≠ scalar u8");
    assert_eq!(rgb_u16_simd, rgb_u16_scalar, "P410 SIMD u16 ≠ scalar u16");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p412_walker_simd_matches_scalar_with_random_chroma() {
  let w = 1922u32;
  let h = 4u32;
  let mut yp = std::vec![0u16; (w * h) as usize];
  let mut uvp = std::vec![0u16; (2 * w * h) as usize];

  let mut state: u32 = 0x3333_4444;
  for s in yp.iter_mut().chain(uvp.iter_mut()) {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *s = ((state & 0xFFF) as u16) << 4;
  }

  let src = P412Frame::new(&yp, &uvp, w, h, w, 2 * w);

  for &full_range in &[true, false] {
    let mut rgb_simd = std::vec![0u8; (w * h * 3) as usize];
    let mut rgb_scalar = std::vec![0u8; (w * h * 3) as usize];

    let mut s_simd = MixedSinker::<P412>::new(w as usize, h as usize)
      .with_rgb(&mut rgb_simd)
      .unwrap();
    p412_to(
      &src,
      full_range,
      s_simd.set_kernel_matrix(KernelMatrix::Bt709),
    )
    .unwrap();

    let mut s_scalar = MixedSinker::<P412>::new(w as usize, h as usize)
      .with_rgb(&mut rgb_scalar)
      .unwrap();
    s_scalar.set_simd(false);
    p412_to(
      &src,
      full_range,
      s_scalar.set_kernel_matrix(KernelMatrix::Bt709),
    )
    .unwrap();

    assert_eq!(rgb_simd, rgb_scalar, "P412 SIMD u8 ≠ scalar u8");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p416_walker_simd_matches_scalar_with_random_chroma() {
  let w = 1922u32;
  let h = 4u32;
  let mut yp = std::vec![0u16; (w * h) as usize];
  let mut uvp = std::vec![0u16; (2 * w * h) as usize];

  let mut state: u32 = 0x5555_6666;
  for s in yp.iter_mut().chain(uvp.iter_mut()) {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *s = state as u16;
  }

  let src = P416Frame::new(&yp, &uvp, w, h, w, 2 * w);

  for &full_range in &[true, false] {
    let mut rgb_simd = std::vec![0u8; (w * h * 3) as usize];
    let mut rgb_scalar = std::vec![0u8; (w * h * 3) as usize];
    let mut rgb_u16_simd = std::vec![0u16; (w * h * 3) as usize];
    let mut rgb_u16_scalar = std::vec![0u16; (w * h * 3) as usize];

    let mut s_simd = MixedSinker::<P416>::new(w as usize, h as usize)
      .with_rgb(&mut rgb_simd)
      .unwrap()
      .with_rgb_u16(&mut rgb_u16_simd)
      .unwrap();
    p416_to(
      &src,
      full_range,
      s_simd.set_kernel_matrix(KernelMatrix::Bt2020Ncl),
    )
    .unwrap();

    let mut s_scalar = MixedSinker::<P416>::new(w as usize, h as usize)
      .with_rgb(&mut rgb_scalar)
      .unwrap()
      .with_rgb_u16(&mut rgb_u16_scalar)
      .unwrap();
    s_scalar.set_simd(false);
    p416_to(
      &src,
      full_range,
      s_scalar.set_kernel_matrix(KernelMatrix::Bt2020Ncl),
    )
    .unwrap();

    assert_eq!(rgb_simd, rgb_scalar, "P416 SIMD u8 ≠ scalar u8");
    assert_eq!(rgb_u16_simd, rgb_u16_scalar, "P416 SIMD u16 ≠ scalar u16");
  }
}

// High-bit 4:2:2 RGBA wiring.
//
// Strategy A combine for the eight 4:2:2 high-bit sinker formats wired
// in the 4:2:2 high-bit file. Mirrors the 4:2:0 test suite shape;
// covers Yuv422p10 (planar BITS-generic), Yuv422p16 (planar 16-bit
// dedicated kernel), and P210 (semi-planar BITS-generic) — the row
// layer is exhaustively tested elsewhere.

#[cfg(feature = "yuv-planar")]
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv422p10_rgba_u8_only_gray_with_opaque_alpha() {
  // 10-bit mid-gray → 8-bit RGBA ≈ (128, 128, 128, 255) per pixel.
  let (yp, up, vp) = solid_yuv422p_n_frame(16, 8, 512, 512, 512);
  let src = Yuv422p10Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuv422p10>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  yuv422p10_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(128) <= 1);
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert_eq!(px[3], 0xFF, "alpha must be opaque");
  }
}

#[cfg(feature = "yuv-planar")]
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv422p10_rgba_u16_only_native_depth_gray_with_opaque_alpha() {
  // 10-bit mid-gray → u16 RGBA: each color element ≈ 512, alpha = 1023.
  let (yp, up, vp) = solid_yuv422p_n_frame(16, 8, 512, 512, 512);
  let src = Yuv422p10Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgba = std::vec![0u16; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuv422p10>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  yuv422p10_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(512) <= 1, "got {px:?}");
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert_eq!(px[3], 1023, "alpha must equal (1 << 10) - 1");
  }
}

#[cfg(feature = "yuv-planar")]
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv422p10_with_rgb_and_with_rgba_produce_byte_identical_rgb_bytes() {
  // Strategy A: when both rgb and rgba are attached, the rgb buffer is
  // populated by the RGB kernel and the rgba buffer is populated via a
  // cheap expand pass. RGB triples must be byte-identical to the
  // standalone RGB-only run.
  let (yp, up, vp) = solid_yuv422p_n_frame(64, 16, 600, 400, 700);
  let src = Yuv422p10Frame::new(&yp, &up, &vp, 64, 16, 64, 32, 32);

  let mut rgb_solo = std::vec![0u8; 64 * 16 * 3];
  let mut s_solo = MixedSinker::<Yuv422p10>::new(64, 16)
    .with_rgb(&mut rgb_solo)
    .unwrap();
  yuv422p10_to(&src, true, s_solo.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();

  let mut rgb_combined = std::vec![0u8; 64 * 16 * 3];
  let mut rgba = std::vec![0u8; 64 * 16 * 4];
  let mut s_combined = MixedSinker::<Yuv422p10>::new(64, 16)
    .with_rgb(&mut rgb_combined)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap();
  yuv422p10_to(
    &src,
    true,
    s_combined.set_kernel_matrix(KernelMatrix::Bt709),
  )
  .unwrap();

  assert_eq!(rgb_solo, rgb_combined, "RGB bytes must match across runs");
  for (rgb_px, rgba_px) in rgb_combined.chunks(3).zip(rgba.chunks(4)) {
    assert_eq!(rgb_px[0], rgba_px[0]);
    assert_eq!(rgb_px[1], rgba_px[1]);
    assert_eq!(rgb_px[2], rgba_px[2]);
    assert_eq!(rgba_px[3], 0xFF);
  }
}

#[cfg(feature = "yuv-planar")]
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv422p10_with_rgb_u16_and_with_rgba_u16_produce_byte_identical_rgb_elems() {
  // Strategy A on the u16 path: rgb_u16 buffer populated by the u16 RGB
  // kernel, rgba_u16 fanned out via expand_rgb_u16_to_rgba_u16_row<10>.
  let (yp, up, vp) = solid_yuv422p_n_frame(64, 16, 600, 400, 700);
  let src = Yuv422p10Frame::new(&yp, &up, &vp, 64, 16, 64, 32, 32);

  let mut rgb_solo = std::vec![0u16; 64 * 16 * 3];
  let mut s_solo = MixedSinker::<Yuv422p10>::new(64, 16)
    .with_rgb_u16(&mut rgb_solo)
    .unwrap();
  yuv422p10_to(&src, true, s_solo.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();

  let mut rgb_combined = std::vec![0u16; 64 * 16 * 3];
  let mut rgba = std::vec![0u16; 64 * 16 * 4];
  let mut s_combined = MixedSinker::<Yuv422p10>::new(64, 16)
    .with_rgb_u16(&mut rgb_combined)
    .unwrap()
    .with_rgba_u16(&mut rgba)
    .unwrap();
  yuv422p10_to(
    &src,
    true,
    s_combined.set_kernel_matrix(KernelMatrix::Bt709),
  )
  .unwrap();

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

#[cfg(feature = "yuv-planar")]
#[test]
fn yuv422p10_rgba_too_short_returns_err() {
  let mut rgba = std::vec![0u8; 10];
  let err = MixedSinker::<Yuv422p10>::new(16, 8)
    .with_rgba(&mut rgba)
    .err()
    .expect("expected InsufficientRgbaBuffer");
  assert!(matches!(err, MixedSinkerError::InsufficientRgbaBuffer(_)));
}

#[cfg(feature = "yuv-planar")]
#[test]
fn yuv422p10_rgba_u16_too_short_returns_err() {
  let mut rgba = std::vec![0u16; 10];
  let err = MixedSinker::<Yuv422p10>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .err()
    .expect("expected InsufficientRgbaU16Buffer");
  assert!(matches!(
    err,
    MixedSinkerError::InsufficientRgbaU16Buffer(_)
  ));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p210_rgba_u16_only_native_depth_gray_with_opaque_alpha() {
  // P210 stores 10-bit samples high-bit-packed (`<< 6`). Mid-gray u16
  // RGBA elements ≈ 512 (low-bit-packed, yuv420p10le convention) and
  // alpha = (1 << 10) - 1 = 1023.
  let (yp, uvp) = solid_p2x0_frame(16, 8, 10, 512, 512, 512);
  let src = P210Frame::new(&yp, &uvp, 16, 8, 16, 16);

  let mut rgba = std::vec![0u16; 16 * 8 * 4];
  let mut sink = MixedSinker::<P210>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  p210_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(512) <= 1, "got {px:?}");
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert_eq!(px[3], 1023, "alpha must equal (1 << 10) - 1");
  }
}

#[cfg(feature = "yuv-planar")]
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv422p16_rgba_u16_only_native_depth_gray_with_opaque_alpha() {
  // 16-bit mid-gray → u16 RGBA: each color element ≈ 32768, alpha = 0xFFFF.
  // Covers the 16-bit dedicated kernel family (no Q15 downshift).
  let (yp, up, vp) = solid_yuv422p_n_frame(16, 8, 32768, 32768, 32768);
  let src = Yuv422p16Frame::new(&yp, &up, &vp, 16, 8, 16, 8, 8);

  let mut rgba = std::vec![0u16; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuv422p16>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  yuv422p16_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(32768) <= 256, "got {px:?}");
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert_eq!(px[3], 0xFFFF, "alpha must equal 0xFFFF");
  }
}

// High-bit 4:4:4 RGBA sinker tests.
//
// Representative coverage of the 4:4:4 high-bit family — not exhaustive
// per-format. Yuv444p10 covers the BITS-generic planar path; P410 covers
// the Pn semi-planar path; Yuv444p16 covers the 16-bit dedicated kernel;
// Yuv440p10 covers the 4:4:0 kernel-reuse path. The remaining 4:4:4
// high-bit formats (9/12/14, P412/P416, Yuv440p12) are exercised by
// row-layer tests + the compile-time guarantee that the sinker builders
// typecheck.

#[cfg(feature = "yuv-planar")]
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv444p10_rgba_u8_only_gray_with_opaque_alpha() {
  // 10-bit mid-gray (Y=512, U=512, V=512) → 8-bit RGBA ≈ (128, 128, 128, 255).
  let (yp, up, vp) = solid_yuv444p_n_frame(16, 8, 512, 512, 512);
  let src = Yuv444p10Frame::new(&yp, &up, &vp, 16, 8, 16, 16, 16);

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuv444p10>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  yuv444p10_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(128) <= 1);
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert_eq!(px[3], 0xFF, "alpha must be opaque");
  }
}

#[cfg(feature = "yuv-planar")]
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv444p10_rgba_u16_only_native_depth_gray_with_opaque_alpha() {
  // 10-bit mid-gray → u16 RGBA: each color element ≈ 512, alpha = 1023.
  let (yp, up, vp) = solid_yuv444p_n_frame(16, 8, 512, 512, 512);
  let src = Yuv444p10Frame::new(&yp, &up, &vp, 16, 8, 16, 16, 16);

  let mut rgba = std::vec![0u16; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuv444p10>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  yuv444p10_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(512) <= 1, "got {px:?}");
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert_eq!(px[3], 1023, "alpha must equal (1 << 10) - 1");
  }
}

#[cfg(feature = "yuv-planar")]
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv444p10_with_rgb_and_with_rgba_produce_byte_identical_rgb_bytes() {
  // Strategy A on the u8 path: rgb buffer populated by the RGB kernel,
  // rgba buffer populated via the cheap expand_rgb_to_rgba_row pass.
  // RGB triples must be byte-identical to the standalone RGB-only run.
  let (yp, up, vp) = solid_yuv444p_n_frame(64, 16, 600, 400, 700);
  let src = Yuv444p10Frame::new(&yp, &up, &vp, 64, 16, 64, 64, 64);

  let mut rgb_solo = std::vec![0u8; 64 * 16 * 3];
  let mut s_solo = MixedSinker::<Yuv444p10>::new(64, 16)
    .with_rgb(&mut rgb_solo)
    .unwrap();
  yuv444p10_to(&src, true, s_solo.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();

  let mut rgb_combined = std::vec![0u8; 64 * 16 * 3];
  let mut rgba = std::vec![0u8; 64 * 16 * 4];
  let mut s_combined = MixedSinker::<Yuv444p10>::new(64, 16)
    .with_rgb(&mut rgb_combined)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap();
  yuv444p10_to(
    &src,
    true,
    s_combined.set_kernel_matrix(KernelMatrix::Bt709),
  )
  .unwrap();

  assert_eq!(rgb_solo, rgb_combined, "RGB bytes must match across runs");
  for (rgb_px, rgba_px) in rgb_combined.chunks(3).zip(rgba.chunks(4)) {
    assert_eq!(rgb_px[0], rgba_px[0]);
    assert_eq!(rgb_px[1], rgba_px[1]);
    assert_eq!(rgb_px[2], rgba_px[2]);
    assert_eq!(rgba_px[3], 0xFF);
  }
}

#[cfg(feature = "yuv-planar")]
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv444p10_with_rgb_u16_and_with_rgba_u16_produce_byte_identical_rgb_elems() {
  // Strategy A on the u16 path: rgb_u16 buffer populated by the u16 RGB
  // kernel, rgba_u16 fanned out via expand_rgb_u16_to_rgba_u16_row<10>.
  let (yp, up, vp) = solid_yuv444p_n_frame(64, 16, 600, 400, 700);
  let src = Yuv444p10Frame::new(&yp, &up, &vp, 64, 16, 64, 64, 64);

  let mut rgb_solo = std::vec![0u16; 64 * 16 * 3];
  let mut s_solo = MixedSinker::<Yuv444p10>::new(64, 16)
    .with_rgb_u16(&mut rgb_solo)
    .unwrap();
  yuv444p10_to(&src, true, s_solo.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();

  let mut rgb_combined = std::vec![0u16; 64 * 16 * 3];
  let mut rgba = std::vec![0u16; 64 * 16 * 4];
  let mut s_combined = MixedSinker::<Yuv444p10>::new(64, 16)
    .with_rgb_u16(&mut rgb_combined)
    .unwrap()
    .with_rgba_u16(&mut rgba)
    .unwrap();
  yuv444p10_to(
    &src,
    true,
    s_combined.set_kernel_matrix(KernelMatrix::Bt709),
  )
  .unwrap();

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

#[cfg(feature = "yuv-planar")]
#[test]
fn yuv444p10_rgba_too_short_returns_err() {
  let mut rgba = std::vec![0u8; 10];
  let err = MixedSinker::<Yuv444p10>::new(16, 8)
    .with_rgba(&mut rgba)
    .err()
    .expect("expected InsufficientRgbaBuffer");
  assert!(matches!(err, MixedSinkerError::InsufficientRgbaBuffer(_)));
}

#[cfg(feature = "yuv-planar")]
#[test]
fn yuv444p10_rgba_u16_too_short_returns_err() {
  let mut rgba = std::vec![0u16; 10];
  let err = MixedSinker::<Yuv444p10>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .err()
    .expect("expected InsufficientRgbaU16Buffer");
  assert!(matches!(
    err,
    MixedSinkerError::InsufficientRgbaU16Buffer(_)
  ));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p410_rgba_u16_only_native_depth_gray_with_opaque_alpha() {
  // P410 (semi-planar 10-bit): mid-gray (high-bit-packed = 512 << 6).
  // u16 RGBA output ≈ 512, alpha = 1023.
  let (yp, uvp) = solid_p4x0_frame(16, 8, 10, 512, 512, 512);
  let src = P410Frame::new(&yp, &uvp, 16, 8, 16, 32);

  let mut rgba = std::vec![0u16; 16 * 8 * 4];
  let mut sink = MixedSinker::<P410>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  p410_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(512) <= 1, "got {px:?}");
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert_eq!(px[3], 1023, "alpha must equal (1 << 10) - 1");
  }
}

#[cfg(feature = "yuv-planar")]
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv444p16_rgba_u16_only_native_depth_gray_with_opaque_alpha() {
  // 16-bit mid-gray → u16 RGBA: each color element ≈ 32768, alpha = 0xFFFF.
  // Covers the 16-bit dedicated kernel family (no Q15 downshift).
  let (yp, up, vp) = solid_yuv444p_n_frame(16, 8, 32768, 32768, 32768);
  let src = Yuv444p16Frame::new(&yp, &up, &vp, 16, 8, 16, 16, 16);

  let mut rgba = std::vec![0u16; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuv444p16>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  yuv444p16_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(32768) <= 256, "got {px:?}");
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert_eq!(px[3], 0xFFFF, "alpha must equal 0xFFFF");
  }
}

#[cfg(feature = "yuv-planar")]
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv440p10_rgba_u16_only_native_depth_gray_with_opaque_alpha() {
  // 4:4:0 reuses the 4:4:4 dispatcher. Confirms the kernel-reuse path
  // wires through correctly at the sinker boundary.
  let (yp, up, vp) = solid_yuv440p_n_frame(16, 8, 512, 512, 512);
  let src = Yuv440p10Frame::new(&yp, &up, &vp, 16, 8, 16, 16, 16);

  let mut rgba = std::vec![0u16; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuv440p10>::new(16, 8)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  yuv440p10_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(512) <= 1, "got {px:?}");
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert_eq!(px[3], 1023, "alpha must equal (1 << 10) - 1");
  }
}

// ---- Atomicity (#308): high-bit 4:2:2 semi-planar P-formats ------------
//
// Each identity-path `process` hoists an up-front RGB-scratch preflight so an
// allocator refusal returns `AllocationFailed` BEFORE any output row — luma
// included — is written, leaving the output frame untouched. Triggering set:
// luma + RGBA + HSV with NO rgb output, so `want_hsv && want_rgba && !want_rgb`
// needs an RGB row but has no caller RGB buffer and would grow
// `rgb_row_buf_or_scratch`'s scratch arm — the only growable scratch on the
// identity path (the u16 RGB / RGBA outputs write straight into their caller
// buffers; the P-formats expose no luma_u16). Reuses the crate's `yuva`-gated
// RGB-scratch failpoint, so these tests are `yuva`-gated too.

#[cfg(feature = "yuva")]
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p210_rgb_scratch_alloc_failure_leaves_outputs_untouched() {
  use crate::resample::ResampleError;

  let (yp, uvp) = solid_p2x0_frame(16, 8, 10, 512, 512, 512);
  let src = P210Frame::new(&yp, &uvp, 16, 8, 16, 16);
  let mut luma = std::vec![0xABu8; 16 * 8];
  let mut rgba = std::vec![0xCDu8; 16 * 8 * 4];
  let (mut hh, mut ss, mut vv) = (
    std::vec![0xCDu8; 16 * 8],
    std::vec![0xCDu8; 16 * 8],
    std::vec![0xCDu8; 16 * 8],
  );
  let mut sink = MixedSinker::<P210>::new(16, 8)
    .with_luma(&mut luma)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap()
    .with_hsv(&mut hh, &mut ss, &mut vv)
    .unwrap();

  super::super::arm_rgb_scratch_alloc_failure();
  let err = p210_to(&src, false, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap_err();
  drop(sink);

  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
    ),
    "RGB-scratch refusal must surface as a recoverable AllocationFailed, got {err:?}"
  );
  assert!(
    luma.iter().all(|&b| b == 0xAB),
    "luma must be untouched on the rgb-scratch alloc-failure path"
  );
}

#[cfg(feature = "yuva")]
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p212_rgb_scratch_alloc_failure_leaves_outputs_untouched() {
  use crate::resample::ResampleError;

  let (yp, uvp) = solid_p2x0_frame(16, 8, 12, 2048, 2048, 2048);
  let src = P212Frame::new(&yp, &uvp, 16, 8, 16, 16);
  let mut luma = std::vec![0xABu8; 16 * 8];
  let mut rgba = std::vec![0xCDu8; 16 * 8 * 4];
  let (mut hh, mut ss, mut vv) = (
    std::vec![0xCDu8; 16 * 8],
    std::vec![0xCDu8; 16 * 8],
    std::vec![0xCDu8; 16 * 8],
  );
  let mut sink = MixedSinker::<P212>::new(16, 8)
    .with_luma(&mut luma)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap()
    .with_hsv(&mut hh, &mut ss, &mut vv)
    .unwrap();

  super::super::arm_rgb_scratch_alloc_failure();
  let err = p212_to(&src, false, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap_err();
  drop(sink);

  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
    ),
    "RGB-scratch refusal must surface as a recoverable AllocationFailed, got {err:?}"
  );
  assert!(
    luma.iter().all(|&b| b == 0xAB),
    "luma must be untouched on the rgb-scratch alloc-failure path"
  );
}

#[cfg(feature = "yuva")]
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p216_rgb_scratch_alloc_failure_leaves_outputs_untouched() {
  use crate::resample::ResampleError;

  let (yp, uvp) = solid_p2x0_frame(16, 8, 16, 32768, 32768, 32768);
  let src = P216Frame::new(&yp, &uvp, 16, 8, 16, 16);
  let mut luma = std::vec![0xABu8; 16 * 8];
  let mut rgba = std::vec![0xCDu8; 16 * 8 * 4];
  let (mut hh, mut ss, mut vv) = (
    std::vec![0xCDu8; 16 * 8],
    std::vec![0xCDu8; 16 * 8],
    std::vec![0xCDu8; 16 * 8],
  );
  let mut sink = MixedSinker::<P216>::new(16, 8)
    .with_luma(&mut luma)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap()
    .with_hsv(&mut hh, &mut ss, &mut vv)
    .unwrap();

  super::super::arm_rgb_scratch_alloc_failure();
  let err = p216_to(&src, false, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap_err();
  drop(sink);

  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
    ),
    "RGB-scratch refusal must surface as a recoverable AllocationFailed, got {err:?}"
  );
  assert!(
    luma.iter().all(|&b| b == 0xAB),
    "luma must be untouched on the rgb-scratch alloc-failure path"
  );
}

// ---- Atomicity (#308): high-bit semi-planar 4:4:4 P-format -------------
//
// The up-front RGB-scratch preflight must return `AllocationFailed` BEFORE
// any output row — luma included — is written, leaving the output frame
// untouched on an allocator refusal. Triggering set: luma + RGBA + HSV with
// NO rgb output — `want_hsv && want_rgba && !want_rgb` needs an RGB row but
// has no caller RGB buffer, so the decode would grow `rgb_row_buf_or_scratch`'s
// scratch arm (the only growable scratch on the identity path; at 4:4:4 the
// u16 RGB / RGBA outputs still write straight into their caller buffers). Reuses
// the crate's `yuva`-gated RGB-scratch failpoint, so these tests are
// `yuva`-gated too.

#[cfg(feature = "yuva")]
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p410_rgb_scratch_alloc_failure_leaves_outputs_untouched() {
  use crate::resample::ResampleError;

  let (yp, uvp) = solid_p4x0_frame(16, 8, 10, 512, 512, 512);
  let src = P410Frame::new(&yp, &uvp, 16, 8, 16, 32);
  let mut luma = std::vec![0xABu8; 16 * 8];
  let mut rgba = std::vec![0xCDu8; 16 * 8 * 4];
  let (mut hh, mut ss, mut vv) = (
    std::vec![0xCDu8; 16 * 8],
    std::vec![0xCDu8; 16 * 8],
    std::vec![0xCDu8; 16 * 8],
  );
  let mut sink = MixedSinker::<P410>::new(16, 8)
    .with_luma(&mut luma)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap()
    .with_hsv(&mut hh, &mut ss, &mut vv)
    .unwrap();

  super::super::arm_rgb_scratch_alloc_failure();
  let err = p410_to(&src, false, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap_err();
  drop(sink);

  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
    ),
    "RGB-scratch refusal must surface as a recoverable AllocationFailed, got {err:?}"
  );
  assert!(
    luma.iter().all(|&b| b == 0xAB),
    "luma must be untouched on the rgb-scratch alloc-failure path"
  );
}

#[cfg(feature = "yuva")]
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p412_rgb_scratch_alloc_failure_leaves_outputs_untouched() {
  use crate::resample::ResampleError;

  let (yp, uvp) = solid_p4x0_frame(16, 8, 12, 2048, 2048, 2048);
  let src = P412Frame::new(&yp, &uvp, 16, 8, 16, 32);
  let mut luma = std::vec![0xABu8; 16 * 8];
  let mut rgba = std::vec![0xCDu8; 16 * 8 * 4];
  let (mut hh, mut ss, mut vv) = (
    std::vec![0xCDu8; 16 * 8],
    std::vec![0xCDu8; 16 * 8],
    std::vec![0xCDu8; 16 * 8],
  );
  let mut sink = MixedSinker::<P412>::new(16, 8)
    .with_luma(&mut luma)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap()
    .with_hsv(&mut hh, &mut ss, &mut vv)
    .unwrap();

  super::super::arm_rgb_scratch_alloc_failure();
  let err = p412_to(&src, false, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap_err();
  drop(sink);

  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
    ),
    "RGB-scratch refusal must surface as a recoverable AllocationFailed, got {err:?}"
  );
  assert!(
    luma.iter().all(|&b| b == 0xAB),
    "luma must be untouched on the rgb-scratch alloc-failure path"
  );
}

#[cfg(feature = "yuva")]
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn p416_rgb_scratch_alloc_failure_leaves_outputs_untouched() {
  use crate::resample::ResampleError;

  let (yp, uvp) = solid_p4x0_frame(16, 8, 16, 32768, 32768, 32768);
  let src = P416Frame::new(&yp, &uvp, 16, 8, 16, 32);
  let mut luma = std::vec![0xABu8; 16 * 8];
  let mut rgba = std::vec![0xCDu8; 16 * 8 * 4];
  let (mut hh, mut ss, mut vv) = (
    std::vec![0xCDu8; 16 * 8],
    std::vec![0xCDu8; 16 * 8],
    std::vec![0xCDu8; 16 * 8],
  );
  let mut sink = MixedSinker::<P416>::new(16, 8)
    .with_luma(&mut luma)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap()
    .with_hsv(&mut hh, &mut ss, &mut vv)
    .unwrap();

  super::super::arm_rgb_scratch_alloc_failure();
  let err = p416_to(&src, false, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap_err();
  drop(sink);

  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
    ),
    "RGB-scratch refusal must surface as a recoverable AllocationFailed, got {err:?}"
  );
  assert!(
    luma.iter().all(|&b| b == 0xAB),
    "luma must be untouched on the rgb-scratch alloc-failure path"
  );
}
