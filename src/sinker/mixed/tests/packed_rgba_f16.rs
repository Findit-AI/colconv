use super::*;

// ---- Tier 9 — Rgbaf16 packed-half-float-RGBA source family (real alpha) -
//
// The alpha-bearing twin of the `Rgbf16` conversion tests. Each pixel is
// `4 x half::f16` (`R, G, B, A`). `with_rgb*` drops alpha; `with_rgba*`
// carries the source alpha; the lossless `with_rgba_f16` / widening
// `with_rgba_f32` preserve it.

fn solid_rgbaf16_frame(
  width: u32,
  height: u32,
  r: half::f16,
  g: half::f16,
  b: half::f16,
  a: half::f16,
) -> std::vec::Vec<half::f16> {
  let w = width as usize;
  let h = height as usize;
  let mut buf = std::vec![half::f16::ZERO; w * h * 4];
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
  ignore = "half::f16::from_f32 uses inline asm (fcvt) unsupported by Miri"
)]
fn rgbaf16_with_rgba_carries_source_alpha() {
  let pix = solid_rgbaf16_frame(
    16,
    4,
    half::f16::from_f32(1.0),
    half::f16::from_f32(0.0),
    half::f16::from_f32(0.25),
    half::f16::from_f32(0.5),
  );
  let src = Rgbaf16LeFrame::try_new(&pix, 16, 4, 16 * 4).unwrap();

  let mut rgba_out = std::vec![0u8; 16 * 4 * 4];
  let mut sink = MixedSinker::<Rgbaf16>::new(16, 4)
    .with_rgba(&mut rgba_out)
    .unwrap();
  rgbaf16_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();

  for px in rgba_out.chunks(4) {
    assert_eq!(px[0], 255);
    assert_eq!(px[1], 0);
    assert_eq!(px[2], 64);
    assert_eq!(px[3], 128, "alpha must come from the source, not 0xFF");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "half::f16::from_f32 uses inline asm (fcvt) unsupported by Miri"
)]
fn rgbaf16_with_rgb_drops_alpha() {
  let pix = solid_rgbaf16_frame(
    16,
    4,
    half::f16::from_f32(1.0),
    half::f16::from_f32(2.0),
    half::f16::from_f32(-0.5),
    half::f16::from_f32(0.5),
  );
  let src = Rgbaf16LeFrame::try_new(&pix, 16, 4, 16 * 4).unwrap();

  let mut rgb_out = std::vec![0u8; 16 * 4 * 3];
  let mut sink = MixedSinker::<Rgbaf16>::new(16, 4)
    .with_rgb(&mut rgb_out)
    .unwrap();
  rgbaf16_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  for px in rgb_out.chunks(3) {
    assert_eq!(px, [255, 255, 0]);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "half::f16::from_f32 uses inline asm (fcvt) unsupported by Miri"
)]
fn rgbaf16_with_rgba_f16_is_lossless_4channel() {
  let vals = [0.0f32, 1.0, -0.25, 1.5, 0.5, 100.0, 0.33, -2.0];
  let pix: std::vec::Vec<half::f16> = (0..16 * 4 * 4)
    .map(|i| half::f16::from_f32(vals[i % vals.len()]))
    .collect();
  let src = Rgbaf16LeFrame::try_new(&pix, 16, 4, 16 * 4).unwrap();

  let mut out = std::vec![half::f16::ZERO; 16 * 4 * 4];
  let mut sink = MixedSinker::<Rgbaf16>::new(16, 4)
    .with_rgba_f16(&mut out)
    .unwrap();
  rgbaf16_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  assert_eq!(out, pix, "rgba_f16 4-channel pass-through is not lossless");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "half::f16::from_f32 uses inline asm (fcvt) unsupported by Miri"
)]
fn rgbaf16_with_rgba_f32_widens_losslessly() {
  let vals = [0.0f32, 1.0, -0.25, 1.5, 0.5, 100.0, 0.33, -2.0];
  let pix: std::vec::Vec<half::f16> = (0..16 * 4 * 4)
    .map(|i| half::f16::from_f32(vals[i % vals.len()]))
    .collect();
  let src = Rgbaf16LeFrame::try_new(&pix, 16, 4, 16 * 4).unwrap();

  let mut out = std::vec![0.0f32; 16 * 4 * 4];
  let mut sink = MixedSinker::<Rgbaf16>::new(16, 4)
    .with_rgba_f32(&mut out)
    .unwrap();
  rgbaf16_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();

  let expected: std::vec::Vec<f32> = pix.iter().map(|h| h.to_f32()).collect();
  assert_eq!(out, expected, "rgba_f32 widen is not lossless");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "half::f16::from_f32 uses inline asm (fcvt) unsupported by Miri"
)]
fn rgbaf16_with_rgb_f16_drops_alpha() {
  let vals = [0.0f32, 1.0, 0.5, 1.5, 0.25, 0.75, 0.33, 2.0];
  let pix: std::vec::Vec<half::f16> = (0..16 * 4 * 4)
    .map(|i| half::f16::from_f32(vals[i % vals.len()]))
    .collect();
  let src = Rgbaf16LeFrame::try_new(&pix, 16, 4, 16 * 4).unwrap();

  let mut out = std::vec![half::f16::ZERO; 16 * 4 * 3];
  let mut sink = MixedSinker::<Rgbaf16>::new(16, 4)
    .with_rgb_f16(&mut out)
    .unwrap();
  rgbaf16_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  for (o, s) in out.chunks(3).zip(pix.chunks(4)) {
    assert_eq!(o, &s[..3]);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "half::f16::from_f32 uses inline asm (fcvt) unsupported by Miri"
)]
fn rgbaf16_simd_matches_scalar_with_random_input() {
  let w = 1921usize;
  let h = 4usize;
  let n = w * h * 4;
  let mut pix = std::vec![half::f16::ZERO; n];
  let mut state: u32 = 0x0BAD_F00D;
  for (i, v) in pix.iter_mut().enumerate() {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    let f = match (state >> 28) & 0b11 {
      0 => ((state >> 8) & 0xFF) as f32 / 255.0,
      1 => (((i as u32 & 0x7F) as f32) + 0.5) / 255.0,
      2 => 1.0 + ((state >> 16) & 0xF) as f32 * 0.25,
      _ => -(((state >> 4) & 0xFF) as f32) / 255.0,
    };
    *v = half::f16::from_f32(f);
  }
  let src = Rgbaf16LeFrame::try_new(&pix, w as u32, h as u32, (w * 4) as u32).unwrap();

  let mut rgb_s = std::vec![0u8; w * h * 3];
  let mut rgb_c = std::vec![0u8; w * h * 3];
  let mut rgba_s = std::vec![0u8; w * h * 4];
  let mut rgba_c = std::vec![0u8; w * h * 4];
  let mut rau16_s = std::vec![0u16; w * h * 4];
  let mut rau16_c = std::vec![0u16; w * h * 4];
  let mut rf16_s = std::vec![half::f16::ZERO; w * h * 3];
  let mut rf16_c = std::vec![half::f16::ZERO; w * h * 3];
  let mut raf16_s = std::vec![half::f16::ZERO; w * h * 4];
  let mut raf16_c = std::vec![half::f16::ZERO; w * h * 4];
  let mut raf32_s = std::vec![0.0f32; w * h * 4];
  let mut raf32_c = std::vec![0.0f32; w * h * 4];
  let mut luma_s = std::vec![0u8; w * h];
  let mut luma_c = std::vec![0u8; w * h];

  let mut s_simd = MixedSinker::<Rgbaf16>::new(w, h)
    .with_rgb(&mut rgb_s)
    .unwrap()
    .with_rgba(&mut rgba_s)
    .unwrap()
    .with_rgba_u16(&mut rau16_s)
    .unwrap()
    .with_rgb_f16(&mut rf16_s)
    .unwrap()
    .with_rgba_f16(&mut raf16_s)
    .unwrap()
    .with_rgba_f32(&mut raf32_s)
    .unwrap()
    .with_luma(&mut luma_s)
    .unwrap();
  rgbaf16_to(&src, true, KernelMatrix::Bt709, &mut s_simd).unwrap();

  let mut s_scalar = MixedSinker::<Rgbaf16>::new(w, h)
    .with_rgb(&mut rgb_c)
    .unwrap()
    .with_rgba(&mut rgba_c)
    .unwrap()
    .with_rgba_u16(&mut rau16_c)
    .unwrap()
    .with_rgb_f16(&mut rf16_c)
    .unwrap()
    .with_rgba_f16(&mut raf16_c)
    .unwrap()
    .with_rgba_f32(&mut raf32_c)
    .unwrap()
    .with_luma(&mut luma_c)
    .unwrap();
  s_scalar.set_simd(false);
  rgbaf16_to(&src, true, KernelMatrix::Bt709, &mut s_scalar).unwrap();

  assert_eq!(rgb_s, rgb_c, "rgb");
  assert_eq!(rgba_s, rgba_c, "rgba");
  assert_eq!(rau16_s, rau16_c, "rgba_u16");
  assert_eq!(rf16_s, rf16_c, "rgb_f16");
  assert_eq!(raf16_s, raf16_c, "rgba_f16");
  assert_eq!(raf32_s, raf32_c, "rgba_f32");
  assert_eq!(luma_s, luma_c, "luma");
  assert_eq!(raf16_s, pix, "rgba_f16 not lossless");
}

/// LE/BE round-trip via the lossless 4-channel f16 path.
#[test]
#[cfg_attr(
  miri,
  ignore = "half::f16::from_f32 uses inline asm (fcvt) unsupported by Miri"
)]
fn rgbaf16_le_be_roundtrip_byte_identical() {
  let vals = [0.5f32, 1.5, -0.25, 100.0, 0.0, 0.75, 2.0, -1.0];
  let intended: std::vec::Vec<half::f16> = (0..16 * 4 * 4)
    .map(|i| half::f16::from_f32(vals[i % vals.len()]))
    .collect();
  let pix_le: std::vec::Vec<half::f16> = intended
    .iter()
    .map(|&v| half::f16::from_bits(u16::from_ne_bytes(v.to_bits().to_le_bytes())))
    .collect();
  let pix_be: std::vec::Vec<half::f16> = intended
    .iter()
    .map(|&v| half::f16::from_bits(u16::from_ne_bytes(v.to_bits().to_be_bytes())))
    .collect();

  let frame_le = Rgbaf16LeFrame::try_new(&pix_le, 16, 4, 16 * 4).unwrap();
  let mut out_le = std::vec![half::f16::ZERO; 16 * 4 * 4];
  let mut sink_le = MixedSinker::<Rgbaf16>::new(16, 4)
    .with_simd(false)
    .with_rgba_f16(&mut out_le)
    .unwrap();
  rgbaf16_to(&frame_le, true, KernelMatrix::Bt709, &mut sink_le).unwrap();

  let frame_be = Rgbaf16BeFrame::try_new(&pix_be, 16, 4, 16 * 4).unwrap();
  let mut out_be = std::vec![half::f16::ZERO; 16 * 4 * 4];
  let mut sink_be = MixedSinker::<Rgbaf16<true>>::new(16, 4)
    .with_simd(false)
    .with_rgba_f16(&mut out_be)
    .unwrap();
  rgbaf16_to_endian(&frame_be, true, KernelMatrix::Bt709, &mut sink_be).unwrap();

  assert_eq!(out_le, intended, "LE plane decoded wrong");
  assert_eq!(out_be, intended, "BE plane decoded wrong");
  assert_eq!(out_le, out_be, "LE/BE outputs diverge");
}
