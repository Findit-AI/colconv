use super::*;

// ---- Tier 9 — Rgbaf32 packed-float-RGBA source family (real alpha) ------
//
// The alpha-bearing twin of the `Rgbf32` conversion tests. Each pixel is
// `4 x f32` (`R, G, B, A`). The `with_rgb*` outputs drop alpha; the
// `with_rgba*` outputs carry the **source** alpha (clamp + scale), and the
// lossless `with_rgba_f32` preserves it bit-exact.

/// Tightly-packed Rgbaf32 frame filled with a constant `(R, G, B, A)`.
fn solid_rgbaf32_frame(
  width: u32,
  height: u32,
  r: f32,
  g: f32,
  b: f32,
  a: f32,
) -> std::vec::Vec<f32> {
  let w = width as usize;
  let h = height as usize;
  let mut buf = std::vec![0.0f32; w * h * 4];
  for px in buf.chunks_mut(4) {
    px[0] = r;
    px[1] = g;
    px[2] = b;
    px[3] = a;
  }
  buf
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgbaf32_with_rgb_drops_alpha_and_clamps() {
  // 1.0 → 255, 2.0 → 255 (HDR clamp), -0.5 → 0; alpha 0.5 is ignored.
  let pix = solid_rgbaf32_frame(16, 4, 1.0, 2.0, -0.5, 0.5);
  let src = Rgbaf32LeFrame::try_new(&pix, 16, 4, 16 * 4).unwrap();

  let mut rgb_out = std::vec![0u8; 16 * 4 * 3];
  let mut sink = MixedSinker::<Rgbaf32>::new(16, 4)
    .with_rgb(&mut rgb_out)
    .unwrap();
  rgbaf32_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();

  for px in rgb_out.chunks(3) {
    assert_eq!(px, [255, 255, 0]);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgbaf32_with_rgba_carries_source_alpha() {
  // Alpha 0.5 → 128 (round-to-even of 127.5 → 128), NOT a constant 0xFF.
  let pix = solid_rgbaf32_frame(16, 4, 1.0, 0.0, 0.25, 0.5);
  let src = Rgbaf32LeFrame::try_new(&pix, 16, 4, 16 * 4).unwrap();

  let mut rgba_out = std::vec![0u8; 16 * 4 * 4];
  let mut sink = MixedSinker::<Rgbaf32>::new(16, 4)
    .with_rgba(&mut rgba_out)
    .unwrap();
  rgbaf32_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();

  for px in rgba_out.chunks(4) {
    assert_eq!(px[0], 255);
    assert_eq!(px[1], 0);
    assert_eq!(px[2], 64); // 0.25*255 = 63.75 → 64
    assert_eq!(px[3], 128, "alpha must come from the source, not 0xFF");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgbaf32_with_rgba_u16_carries_source_alpha() {
  let pix = solid_rgbaf32_frame(16, 4, 1.0, 0.0, 0.0, 1.0);
  let src = Rgbaf32LeFrame::try_new(&pix, 16, 4, 16 * 4).unwrap();

  let mut rgba_out = std::vec![0u16; 16 * 4 * 4];
  let mut sink = MixedSinker::<Rgbaf32>::new(16, 4)
    .with_rgba_u16(&mut rgba_out)
    .unwrap();
  rgbaf32_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();

  for px in rgba_out.chunks(4) {
    assert_eq!(px, [65535, 0, 0, 65535]);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgbaf32_with_rgba_f32_is_lossless_4channel() {
  // HDR, negatives, fractional alpha — all preserved bit-exact (4 channels).
  let vals = [0.0f32, 1.0, -0.25, 1.5, 0.5, 100.0, 0.33, -2.0];
  let pix: std::vec::Vec<f32> = (0..16 * 4 * 4).map(|i| vals[i % vals.len()]).collect();
  let src = Rgbaf32LeFrame::try_new(&pix, 16, 4, 16 * 4).unwrap();

  let mut out = std::vec![0.0f32; 16 * 4 * 4];
  let mut sink = MixedSinker::<Rgbaf32>::new(16, 4)
    .with_rgba_f32(&mut out)
    .unwrap();
  rgbaf32_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();

  assert_eq!(out, pix, "rgba_f32 4-channel pass-through is not lossless");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgbaf32_with_rgb_f32_drops_alpha_losslessly() {
  let vals = [0.0f32, 1.0, -0.25, 1.5, 0.5, 100.0, 0.33, -2.0];
  let pix: std::vec::Vec<f32> = (0..16 * 4 * 4).map(|i| vals[i % vals.len()]).collect();
  let src = Rgbaf32LeFrame::try_new(&pix, 16, 4, 16 * 4).unwrap();

  let mut out = std::vec![0.0f32; 16 * 4 * 3];
  let mut sink = MixedSinker::<Rgbaf32>::new(16, 4)
    .with_rgb_f32(&mut out)
    .unwrap();
  rgbaf32_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();

  // Each output pixel's R/G/B equals the source's R/G/B (alpha skipped).
  for (o, s) in out.chunks(3).zip(pix.chunks(4)) {
    assert_eq!(o, &s[..3]);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgbaf32_with_luma_and_hsv_ignore_alpha() {
  // White with alpha 0.0 → luma 255, HSV (0,0,255). Alpha must not leak.
  let pix = solid_rgbaf32_frame(16, 4, 1.0, 1.0, 1.0, 0.0);
  let src = Rgbaf32LeFrame::try_new(&pix, 16, 4, 16 * 4).unwrap();

  let n = 16 * 4;
  let mut luma = std::vec![0u8; n];
  let mut h = std::vec![0u8; n];
  let mut s = std::vec![0u8; n];
  let mut v = std::vec![0u8; n];
  let mut sink = MixedSinker::<Rgbaf32>::new(16, 4)
    .with_luma(&mut luma)
    .unwrap()
    .with_hsv(&mut h, &mut s, &mut v)
    .unwrap();
  rgbaf32_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
  for i in 0..n {
    assert_eq!(luma[i], 255);
    assert_eq!(h[i], 0);
    assert_eq!(s[i], 0);
    assert_eq!(v[i], 255);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgbaf32_simd_matches_scalar_with_random_input() {
  // Width 1921 forces both SIMD main loop and scalar tail.
  let w = 1921usize;
  let h = 4usize;
  let n = w * h * 4;
  let mut pix = std::vec![0.0f32; n];
  let mut state: u32 = 0x1234_5678;
  for (i, v) in pix.iter_mut().enumerate() {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *v = match (state >> 28) & 0b11 {
      0 => ((state >> 8) & 0xFF) as f32 / 255.0,
      1 => (((i as u32 & 0x7F) as f32) + 0.5) / 255.0,
      2 => 1.0 + ((state >> 16) & 0xF) as f32 * 0.25,
      _ => -(((state >> 4) & 0xFF) as f32) / 255.0,
    };
  }
  let src = Rgbaf32LeFrame::try_new(&pix, w as u32, h as u32, (w * 4) as u32).unwrap();

  let mut rgb_s = std::vec![0u8; w * h * 3];
  let mut rgb_c = std::vec![0u8; w * h * 3];
  let mut rgba_s = std::vec![0u8; w * h * 4];
  let mut rgba_c = std::vec![0u8; w * h * 4];
  let mut ru16_s = std::vec![0u16; w * h * 3];
  let mut ru16_c = std::vec![0u16; w * h * 3];
  let mut rau16_s = std::vec![0u16; w * h * 4];
  let mut rau16_c = std::vec![0u16; w * h * 4];
  let mut rf32_s = std::vec![0.0f32; w * h * 3];
  let mut rf32_c = std::vec![0.0f32; w * h * 3];
  let mut raf32_s = std::vec![0.0f32; w * h * 4];
  let mut raf32_c = std::vec![0.0f32; w * h * 4];
  let mut luma_s = std::vec![0u8; w * h];
  let mut luma_c = std::vec![0u8; w * h];

  let mut s_simd = MixedSinker::<Rgbaf32>::new(w, h)
    .with_rgb(&mut rgb_s)
    .unwrap()
    .with_rgba(&mut rgba_s)
    .unwrap()
    .with_rgb_u16(&mut ru16_s)
    .unwrap()
    .with_rgba_u16(&mut rau16_s)
    .unwrap()
    .with_rgb_f32(&mut rf32_s)
    .unwrap()
    .with_rgba_f32(&mut raf32_s)
    .unwrap()
    .with_luma(&mut luma_s)
    .unwrap();
  rgbaf32_to(&src, true, s_simd.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();

  let mut s_scalar = MixedSinker::<Rgbaf32>::new(w, h)
    .with_rgb(&mut rgb_c)
    .unwrap()
    .with_rgba(&mut rgba_c)
    .unwrap()
    .with_rgb_u16(&mut ru16_c)
    .unwrap()
    .with_rgba_u16(&mut rau16_c)
    .unwrap()
    .with_rgb_f32(&mut rf32_c)
    .unwrap()
    .with_rgba_f32(&mut raf32_c)
    .unwrap()
    .with_luma(&mut luma_c)
    .unwrap();
  s_scalar.set_simd(false);
  rgbaf32_to(&src, true, s_scalar.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();

  assert_eq!(rgb_s, rgb_c, "rgb");
  assert_eq!(rgba_s, rgba_c, "rgba");
  assert_eq!(ru16_s, ru16_c, "rgb_u16");
  assert_eq!(rau16_s, rau16_c, "rgba_u16");
  assert_eq!(rf32_s, rf32_c, "rgb_f32");
  assert_eq!(raf32_s, raf32_c, "rgba_f32");
  assert_eq!(luma_s, luma_c, "luma");
}

/// LE/BE round-trip: the same logical samples in both plane orderings must
/// yield byte-identical output through the matching `Rgbaf32<BE>` sink.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgbaf32_le_be_roundtrip_byte_identical() {
  let vals = [0.5f32, 1.5, -0.25, 100.0, 0.0, 0.75, 2.0, -1.0];
  let intended: std::vec::Vec<f32> = (0..16 * 4 * 4).map(|i| vals[i % vals.len()]).collect();
  let pix_le: std::vec::Vec<f32> = intended
    .iter()
    .map(|&v| f32::from_bits(u32::from_ne_bytes(v.to_bits().to_le_bytes())))
    .collect();
  let pix_be: std::vec::Vec<f32> = intended
    .iter()
    .map(|&v| f32::from_bits(u32::from_ne_bytes(v.to_bits().to_be_bytes())))
    .collect();

  let frame_le = Rgbaf32LeFrame::try_new(&pix_le, 16, 4, 16 * 4).unwrap();
  let mut out_le = std::vec![0.0f32; 16 * 4 * 4];
  let mut sink_le = MixedSinker::<Rgbaf32>::new(16, 4)
    .with_simd(false)
    .with_rgba_f32(&mut out_le)
    .unwrap();
  rgbaf32_to(
    &frame_le,
    true,
    sink_le.set_kernel_matrix(KernelMatrix::Bt709),
  )
  .unwrap();

  let frame_be = Rgbaf32BeFrame::try_new(&pix_be, 16, 4, 16 * 4).unwrap();
  let mut out_be = std::vec![0.0f32; 16 * 4 * 4];
  let mut sink_be = MixedSinker::<Rgbaf32<true>>::new(16, 4)
    .with_simd(false)
    .with_rgba_f32(&mut out_be)
    .unwrap();
  rgbaf32_to_endian(
    &frame_be,
    true,
    sink_be.set_kernel_matrix(KernelMatrix::Bt709),
  )
  .unwrap();

  assert_eq!(out_le, intended, "LE plane decoded wrong");
  assert_eq!(out_be, intended, "BE plane decoded wrong");
  assert_eq!(out_le, out_be, "LE/BE outputs diverge");
}
