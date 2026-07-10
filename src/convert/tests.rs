//! Byte-parity tests: [`Convert`] output must equal the hand-assembled
//! Tier-1 (`MixedSinker` + `{fmt}_to`) path, for the planar-YUV proof formats
//! and every output kind their sinks support.

use crate::{
  ChromaLocation, ColorInfo, ColorMatrix, ColorSpec, DynamicRange, PixelFormat, Primaries,
  Transfer, YuvOptions,
  convert::Convert,
  frame::{Yuv420pFrame, Yuv444pFrame},
  resample::AreaResampler,
  sinker::{MixedSinker, MixedSinkerError},
  source::{Yuv420p, Yuv444p, yuv420p_to, yuv444p_to},
};

/// A `Bt601` / limited / left-sited spec — the same shape
/// [`crate::sinker::mixed::tests::st428`] decodes against.
fn spec_601() -> ColorSpec {
  ColorSpec::from_info(
    PixelFormat::Yuv420p,
    ColorInfo::new(
      Primaries::Bt709,
      Transfer::Unspecified,
      ColorMatrix::Bt601,
      DynamicRange::Limited,
      ChromaLocation::Left,
    ),
  )
}

// ===== Yuv420p ==========================================================

/// 4x2 `Yuv420p` fixture (2 chroma samples).
fn y420() -> (std::vec::Vec<u8>, std::vec::Vec<u8>, std::vec::Vec<u8>) {
  (
    std::vec![40u8, 80, 120, 160, 200, 240, 30, 70],
    std::vec![100u8, 150],
    std::vec![150u8, 100],
  )
}

fn frame420<'a>(y: &'a [u8], u: &'a [u8], v: &'a [u8]) -> Yuv420pFrame<'a> {
  Yuv420pFrame::new(y, u, v, 4, 2, 4, 2, 2)
}

/// rgb / rgba / luma / hsv output kinds each match the manual decode.
#[test]
fn yuv420p_output_kinds_match_manual() {
  let (y, u, v) = y420();
  let spec = spec_601();
  let (w, h) = (4usize, 2usize);

  // rgb
  let (mut a, mut b) = (std::vec![0u8; w * h * 3], std::vec![0u8; w * h * 3]);
  Convert::from(&frame420(&y, &u, &v))
    .spec(spec)
    .rgb(&mut a)
    .run()
    .unwrap();
  {
    let mut sink = MixedSinker::<Yuv420p>::new(w, h)
      .with_rgb(&mut b)
      .unwrap()
      .with_color_spec(spec);
    yuv420p_to(
      &frame420(&y, &u, &v),
      spec.full_range(),
      spec.matrix(),
      &mut sink,
    )
    .unwrap();
  }
  assert_eq!(a, b, "Yuv420p rgb");

  // rgba
  let (mut a, mut b) = (std::vec![0u8; w * h * 4], std::vec![0u8; w * h * 4]);
  Convert::from(&frame420(&y, &u, &v))
    .spec(spec)
    .rgba(&mut a)
    .run()
    .unwrap();
  {
    let mut sink = MixedSinker::<Yuv420p>::new(w, h)
      .with_rgba(&mut b)
      .unwrap()
      .with_color_spec(spec);
    yuv420p_to(
      &frame420(&y, &u, &v),
      spec.full_range(),
      spec.matrix(),
      &mut sink,
    )
    .unwrap();
  }
  assert_eq!(a, b, "Yuv420p rgba");

  // luma
  let (mut a, mut b) = (std::vec![0u8; w * h], std::vec![0u8; w * h]);
  Convert::from(&frame420(&y, &u, &v))
    .spec(spec)
    .luma(&mut a)
    .run()
    .unwrap();
  {
    let mut sink = MixedSinker::<Yuv420p>::new(w, h)
      .with_luma(&mut b)
      .unwrap()
      .with_color_spec(spec);
    yuv420p_to(
      &frame420(&y, &u, &v),
      spec.full_range(),
      spec.matrix(),
      &mut sink,
    )
    .unwrap();
  }
  assert_eq!(a, b, "Yuv420p luma");

  // hsv
  let (mut ha, mut sa, mut va) = (
    std::vec![0u8; w * h],
    std::vec![0u8; w * h],
    std::vec![0u8; w * h],
  );
  let (mut hb, mut sb, mut vb) = (
    std::vec![0u8; w * h],
    std::vec![0u8; w * h],
    std::vec![0u8; w * h],
  );
  Convert::from(&frame420(&y, &u, &v))
    .spec(spec)
    .hsv(&mut ha, &mut sa, &mut va)
    .run()
    .unwrap();
  {
    let mut sink = MixedSinker::<Yuv420p>::new(w, h)
      .with_hsv(&mut hb, &mut sb, &mut vb)
      .unwrap()
      .with_color_spec(spec);
    yuv420p_to(
      &frame420(&y, &u, &v),
      spec.full_range(),
      spec.matrix(),
      &mut sink,
    )
    .unwrap();
  }
  assert_eq!((ha, sa, va), (hb, sb, vb), "Yuv420p hsv");
}

/// `.spec(spec)` alone derives the walk range + matrix — the same bytes as
/// passing them manually.
#[test]
fn yuv420p_spec_derivation_matches_manual_matrix_range() {
  let (y, u, v) = y420();
  let spec = spec_601();
  let (w, h) = (4usize, 2usize);

  let mut a = std::vec![0u8; w * h * 3];
  Convert::from(&frame420(&y, &u, &v))
    .spec(spec)
    .rgb(&mut a)
    .run()
    .unwrap();

  let mut b = std::vec![0u8; w * h * 3];
  {
    let mut sink = MixedSinker::<Yuv420p>::new(w, h)
      .with_rgb(&mut b)
      .unwrap()
      .with_color_spec(spec);
    // Manual: pull the exact range + matrix the spec resolves to.
    yuv420p_to(
      &frame420(&y, &u, &v),
      spec.full_range(),
      spec.matrix(),
      &mut sink,
    )
    .unwrap();
  }
  assert_eq!(a, b);
}

/// No spec == the manual path driven by `YuvOptions::default()`.
#[test]
fn yuv420p_no_spec_matches_default_options() {
  let (y, u, v) = y420();
  let (w, h) = (4usize, 2usize);
  let def = YuvOptions::default();

  let mut a = std::vec![0u8; w * h * 3];
  Convert::from(&frame420(&y, &u, &v))
    .rgb(&mut a)
    .run()
    .unwrap();

  let mut b = std::vec![0u8; w * h * 3];
  {
    // No `with_color_spec`: the no-spec Convert path never touches it either.
    let mut sink = MixedSinker::<Yuv420p>::new(w, h).with_rgb(&mut b).unwrap();
    yuv420p_to(
      &frame420(&y, &u, &v),
      def.full_range(),
      def.matrix(),
      &mut sink,
    )
    .unwrap();
  }
  assert_eq!(a, b);
}

/// `.format_options` overrides the spec-derived matrix.
#[test]
fn yuv420p_format_options_override_wins() {
  let (y, u, v) = y420();
  let spec = spec_601(); // matrix = Bt601
  let (w, h) = (4usize, 2usize);
  let override_opts = YuvOptions::new().with_matrix(ColorMatrix::Bt2020Ncl);
  assert_ne!(override_opts.matrix(), spec.matrix());

  let mut a = std::vec![0u8; w * h * 3];
  Convert::from(&frame420(&y, &u, &v))
    .spec(spec)
    .format_options(override_opts)
    .rgb(&mut a)
    .run()
    .unwrap();

  let mut b = std::vec![0u8; w * h * 3];
  {
    // Sink still carries the spec (siting/primaries), but the walk uses the
    // override's range + matrix.
    let mut sink = MixedSinker::<Yuv420p>::new(w, h)
      .with_rgb(&mut b)
      .unwrap()
      .with_color_spec(spec);
    yuv420p_to(
      &frame420(&y, &u, &v),
      override_opts.full_range(),
      override_opts.matrix(),
      &mut sink,
    )
    .unwrap();
  }
  assert_eq!(a, b);
}

/// `.simd(false)` == the manual `with_simd(false)` scalar path.
#[test]
fn yuv420p_simd_false_matches_manual() {
  let (y, u, v) = y420();
  let spec = spec_601();
  let (w, h) = (4usize, 2usize);

  let mut a = std::vec![0u8; w * h * 3];
  Convert::from(&frame420(&y, &u, &v))
    .spec(spec)
    .simd(false)
    .rgb(&mut a)
    .run()
    .unwrap();

  let mut b = std::vec![0u8; w * h * 3];
  {
    let mut sink = MixedSinker::<Yuv420p>::new(w, h)
      .with_rgb(&mut b)
      .unwrap()
      .with_color_spec(spec)
      .with_simd(false);
    yuv420p_to(
      &frame420(&y, &u, &v),
      spec.full_range(),
      spec.matrix(),
      &mut sink,
    )
    .unwrap();
  }
  assert_eq!(a, b);
}

/// Multiple outputs (rgb + luma) attach together and both match.
#[test]
fn yuv420p_multi_output_matches_manual() {
  let (y, u, v) = y420();
  let spec = spec_601();
  let (w, h) = (4usize, 2usize);

  let (mut rgb_a, mut luma_a) = (std::vec![0u8; w * h * 3], std::vec![0u8; w * h]);
  Convert::from(&frame420(&y, &u, &v))
    .spec(spec)
    .rgb(&mut rgb_a)
    .luma(&mut luma_a)
    .run()
    .unwrap();

  let (mut rgb_b, mut luma_b) = (std::vec![0u8; w * h * 3], std::vec![0u8; w * h]);
  {
    let mut sink = MixedSinker::<Yuv420p>::new(w, h)
      .with_rgb(&mut rgb_b)
      .unwrap()
      .with_luma(&mut luma_b)
      .unwrap()
      .with_color_spec(spec);
    yuv420p_to(
      &frame420(&y, &u, &v),
      spec.full_range(),
      spec.matrix(),
      &mut sink,
    )
    .unwrap();
  }
  assert_eq!(rgb_a, rgb_b, "rgb");
  assert_eq!(luma_a, luma_b, "luma");
}

/// `.resize(w/2, h/2)` == the manual `AreaResampler` downscale path.
#[test]
fn yuv420p_resize_area_matches_manual() {
  // 8x4 source (cw = 4, ch = 2), downscale to 4x2.
  let y: std::vec::Vec<u8> = (0..32u16).map(|i| (i * 7) as u8).collect();
  let u: std::vec::Vec<u8> = (0..8u16).map(|i| (i * 11 + 40) as u8).collect();
  let v: std::vec::Vec<u8> = (0..8u16).map(|i| (i * 13 + 20) as u8).collect();
  let make = || Yuv420pFrame::new(&y, &u, &v, 8, 4, 8, 4, 4);
  let spec = spec_601();
  let (ow, oh) = (4usize, 2usize);

  let mut a = std::vec![0u8; ow * oh * 3];
  Convert::from(&make())
    .spec(spec)
    .resize(ow, oh)
    .rgb(&mut a)
    .run()
    .unwrap();

  let mut b = std::vec![0u8; ow * oh * 3];
  {
    let mut sink =
      MixedSinker::<Yuv420p, AreaResampler>::with_resampler(8, 4, AreaResampler::to(ow, oh))
        .unwrap()
        .with_rgb(&mut b)
        .unwrap()
        .with_color_spec(spec);
    yuv420p_to(&make(), spec.full_range(), spec.matrix(), &mut sink).unwrap();
  }
  assert_eq!(a, b);
}

/// `.resize` upscale surfaces the resampler's upscale error from `run()`.
#[test]
fn yuv420p_resize_upscale_errors() {
  let (y, u, v) = y420();
  let mut rgb = std::vec![0u8; 8 * 4 * 3];
  let err = Convert::from(&frame420(&y, &u, &v))
    .resize(8, 4) // 4x2 -> 8x4 is an upscale
    .rgb(&mut rgb)
    .run()
    .expect_err("upscale must be rejected");
  assert!(
    matches!(err, MixedSinkerError::Resample(_)),
    "unexpected error: {err:?}"
  );
}

// ===== Yuv444p ==========================================================

/// 4x2 `Yuv444p` fixture (full-res chroma).
fn y444() -> (std::vec::Vec<u8>, std::vec::Vec<u8>, std::vec::Vec<u8>) {
  (
    std::vec![40u8, 80, 120, 160, 200, 240, 30, 70],
    std::vec![100u8, 110, 120, 130, 140, 150, 160, 170],
    std::vec![170u8, 160, 150, 140, 130, 120, 110, 100],
  )
}

fn frame444<'a>(y: &'a [u8], u: &'a [u8], v: &'a [u8]) -> Yuv444pFrame<'a> {
  Yuv444pFrame::new(y, u, v, 4, 2, 4, 4, 4)
}

/// rgb / rgba / luma / hsv output kinds each match the manual decode.
#[test]
fn yuv444p_output_kinds_match_manual() {
  let (y, u, v) = y444();
  let spec = spec_601();
  let (w, h) = (4usize, 2usize);

  // rgb
  let (mut a, mut b) = (std::vec![0u8; w * h * 3], std::vec![0u8; w * h * 3]);
  Convert::from(&frame444(&y, &u, &v))
    .spec(spec)
    .rgb(&mut a)
    .run()
    .unwrap();
  {
    let mut sink = MixedSinker::<Yuv444p>::new(w, h)
      .with_rgb(&mut b)
      .unwrap()
      .with_color_spec(spec);
    yuv444p_to(
      &frame444(&y, &u, &v),
      spec.full_range(),
      spec.matrix(),
      &mut sink,
    )
    .unwrap();
  }
  assert_eq!(a, b, "Yuv444p rgb");

  // rgba
  let (mut a, mut b) = (std::vec![0u8; w * h * 4], std::vec![0u8; w * h * 4]);
  Convert::from(&frame444(&y, &u, &v))
    .spec(spec)
    .rgba(&mut a)
    .run()
    .unwrap();
  {
    let mut sink = MixedSinker::<Yuv444p>::new(w, h)
      .with_rgba(&mut b)
      .unwrap()
      .with_color_spec(spec);
    yuv444p_to(
      &frame444(&y, &u, &v),
      spec.full_range(),
      spec.matrix(),
      &mut sink,
    )
    .unwrap();
  }
  assert_eq!(a, b, "Yuv444p rgba");

  // luma
  let (mut a, mut b) = (std::vec![0u8; w * h], std::vec![0u8; w * h]);
  Convert::from(&frame444(&y, &u, &v))
    .spec(spec)
    .luma(&mut a)
    .run()
    .unwrap();
  {
    let mut sink = MixedSinker::<Yuv444p>::new(w, h)
      .with_luma(&mut b)
      .unwrap()
      .with_color_spec(spec);
    yuv444p_to(
      &frame444(&y, &u, &v),
      spec.full_range(),
      spec.matrix(),
      &mut sink,
    )
    .unwrap();
  }
  assert_eq!(a, b, "Yuv444p luma");

  // hsv
  let (mut ha, mut sa, mut va) = (
    std::vec![0u8; w * h],
    std::vec![0u8; w * h],
    std::vec![0u8; w * h],
  );
  let (mut hb, mut sb, mut vb) = (
    std::vec![0u8; w * h],
    std::vec![0u8; w * h],
    std::vec![0u8; w * h],
  );
  Convert::from(&frame444(&y, &u, &v))
    .spec(spec)
    .hsv(&mut ha, &mut sa, &mut va)
    .run()
    .unwrap();
  {
    let mut sink = MixedSinker::<Yuv444p>::new(w, h)
      .with_hsv(&mut hb, &mut sb, &mut vb)
      .unwrap()
      .with_color_spec(spec);
    yuv444p_to(
      &frame444(&y, &u, &v),
      spec.full_range(),
      spec.matrix(),
      &mut sink,
    )
    .unwrap();
  }
  assert_eq!((ha, sa, va), (hb, sb, vb), "Yuv444p hsv");
}

/// `.spec(spec)` alone derives the walk range + matrix — the same bytes as
/// passing them manually.
#[test]
fn yuv444p_spec_derivation_matches_manual_matrix_range() {
  let (y, u, v) = y444();
  let spec = spec_601();
  let (w, h) = (4usize, 2usize);

  let mut a = std::vec![0u8; w * h * 3];
  Convert::from(&frame444(&y, &u, &v))
    .spec(spec)
    .rgb(&mut a)
    .run()
    .unwrap();

  let mut b = std::vec![0u8; w * h * 3];
  {
    let mut sink = MixedSinker::<Yuv444p>::new(w, h)
      .with_rgb(&mut b)
      .unwrap()
      .with_color_spec(spec);
    yuv444p_to(
      &frame444(&y, &u, &v),
      spec.full_range(),
      spec.matrix(),
      &mut sink,
    )
    .unwrap();
  }
  assert_eq!(a, b);
}

/// No spec == the manual path driven by `YuvOptions::default()`.
#[test]
fn yuv444p_no_spec_matches_default_options() {
  let (y, u, v) = y444();
  let (w, h) = (4usize, 2usize);
  let def = YuvOptions::default();

  let mut a = std::vec![0u8; w * h * 3];
  Convert::from(&frame444(&y, &u, &v))
    .rgb(&mut a)
    .run()
    .unwrap();

  let mut b = std::vec![0u8; w * h * 3];
  {
    // No `with_color_spec`: the no-spec Convert path never touches it either.
    let mut sink = MixedSinker::<Yuv444p>::new(w, h).with_rgb(&mut b).unwrap();
    yuv444p_to(
      &frame444(&y, &u, &v),
      def.full_range(),
      def.matrix(),
      &mut sink,
    )
    .unwrap();
  }
  assert_eq!(a, b);
}

/// `.format_options` overrides the spec-derived matrix.
#[test]
fn yuv444p_format_options_override_wins() {
  let (y, u, v) = y444();
  let spec = spec_601(); // matrix = Bt601
  let (w, h) = (4usize, 2usize);
  let override_opts = YuvOptions::new().with_matrix(ColorMatrix::Bt2020Ncl);
  assert_ne!(override_opts.matrix(), spec.matrix());

  let mut a = std::vec![0u8; w * h * 3];
  Convert::from(&frame444(&y, &u, &v))
    .spec(spec)
    .format_options(override_opts)
    .rgb(&mut a)
    .run()
    .unwrap();

  let mut b = std::vec![0u8; w * h * 3];
  {
    // Sink still carries the spec (siting/primaries), but the walk uses the
    // override's range + matrix.
    let mut sink = MixedSinker::<Yuv444p>::new(w, h)
      .with_rgb(&mut b)
      .unwrap()
      .with_color_spec(spec);
    yuv444p_to(
      &frame444(&y, &u, &v),
      override_opts.full_range(),
      override_opts.matrix(),
      &mut sink,
    )
    .unwrap();
  }
  assert_eq!(a, b);
}

/// `.simd(false)` == the manual `with_simd(false)` scalar path (no resize).
#[test]
fn yuv444p_simd_false_matches_manual() {
  let (y, u, v) = y444();
  let spec = spec_601();
  let (w, h) = (4usize, 2usize);

  let mut a = std::vec![0u8; w * h * 3];
  Convert::from(&frame444(&y, &u, &v))
    .spec(spec)
    .simd(false)
    .rgb(&mut a)
    .run()
    .unwrap();

  let mut b = std::vec![0u8; w * h * 3];
  {
    let mut sink = MixedSinker::<Yuv444p>::new(w, h)
      .with_rgb(&mut b)
      .unwrap()
      .with_color_spec(spec)
      .with_simd(false);
    yuv444p_to(
      &frame444(&y, &u, &v),
      spec.full_range(),
      spec.matrix(),
      &mut sink,
    )
    .unwrap();
  }
  assert_eq!(a, b);
}

/// Multiple outputs (rgb + luma) attach together and both match.
#[test]
fn yuv444p_multi_output_matches_manual() {
  let (y, u, v) = y444();
  let spec = spec_601();
  let (w, h) = (4usize, 2usize);

  let (mut rgb_a, mut luma_a) = (std::vec![0u8; w * h * 3], std::vec![0u8; w * h]);
  Convert::from(&frame444(&y, &u, &v))
    .spec(spec)
    .rgb(&mut rgb_a)
    .luma(&mut luma_a)
    .run()
    .unwrap();

  let (mut rgb_b, mut luma_b) = (std::vec![0u8; w * h * 3], std::vec![0u8; w * h]);
  {
    let mut sink = MixedSinker::<Yuv444p>::new(w, h)
      .with_rgb(&mut rgb_b)
      .unwrap()
      .with_luma(&mut luma_b)
      .unwrap()
      .with_color_spec(spec);
    yuv444p_to(
      &frame444(&y, &u, &v),
      spec.full_range(),
      spec.matrix(),
      &mut sink,
    )
    .unwrap();
  }
  assert_eq!(rgb_a, rgb_b, "rgb");
  assert_eq!(luma_a, luma_b, "luma");
}

/// `.resize(w/2, h/2)` == the manual `AreaResampler` downscale path (default
/// simd).
#[test]
fn yuv444p_resize_area_matches_manual() {
  // 8x4 source, downscale to 4x2.
  let y: std::vec::Vec<u8> = (0..32u16).map(|i| (i * 7) as u8).collect();
  let u: std::vec::Vec<u8> = (0..32u16).map(|i| (i * 3 + 20) as u8).collect();
  let v: std::vec::Vec<u8> = (0..32u16).map(|i| (i * 5 + 10) as u8).collect();
  let make = || Yuv444pFrame::new(&y, &u, &v, 8, 4, 8, 8, 8);
  let spec = spec_601();
  let (ow, oh) = (4usize, 2usize);

  let mut a = std::vec![0u8; ow * oh * 3];
  Convert::from(&make())
    .spec(spec)
    .resize(ow, oh)
    .rgb(&mut a)
    .run()
    .unwrap();

  let mut b = std::vec![0u8; ow * oh * 3];
  {
    let mut sink =
      MixedSinker::<Yuv444p, AreaResampler>::with_resampler(8, 4, AreaResampler::to(ow, oh))
        .unwrap()
        .with_rgb(&mut b)
        .unwrap()
        .with_color_spec(spec);
    yuv444p_to(&make(), spec.full_range(), spec.matrix(), &mut sink).unwrap();
  }
  assert_eq!(a, b);
}

/// `.resize` upscale surfaces the resampler's upscale error from `run()`.
#[test]
fn yuv444p_resize_upscale_errors() {
  let (y, u, v) = y444();
  let mut rgb = std::vec![0u8; 8 * 4 * 3];
  let err = Convert::from(&frame444(&y, &u, &v))
    .resize(8, 4) // 4x2 -> 8x4 is an upscale
    .rgb(&mut rgb)
    .run()
    .expect_err("upscale must be rejected");
  assert!(
    matches!(err, MixedSinkerError::Resample(_)),
    "unexpected error: {err:?}"
  );
}

/// `.resize` + `.simd(false)` both match the manual path for Yuv444p.
#[test]
fn yuv444p_resize_and_scalar_match_manual() {
  // 8x4 source, downscale to 4x2.
  let y: std::vec::Vec<u8> = (0..32u16).map(|i| (i * 7) as u8).collect();
  let u: std::vec::Vec<u8> = (0..32u16).map(|i| (i * 3 + 20) as u8).collect();
  let v: std::vec::Vec<u8> = (0..32u16).map(|i| (i * 5 + 10) as u8).collect();
  let make = || Yuv444pFrame::new(&y, &u, &v, 8, 4, 8, 8, 8);
  let spec = spec_601();
  let (ow, oh) = (4usize, 2usize);

  let mut a = std::vec![0u8; ow * oh * 3];
  Convert::from(&make())
    .spec(spec)
    .simd(false)
    .resize(ow, oh)
    .rgb(&mut a)
    .run()
    .unwrap();

  let mut b = std::vec![0u8; ow * oh * 3];
  {
    let mut sink =
      MixedSinker::<Yuv444p, AreaResampler>::with_resampler(8, 4, AreaResampler::to(ow, oh))
        .unwrap()
        .with_rgb(&mut b)
        .unwrap()
        .with_color_spec(spec)
        .with_simd(false);
    yuv444p_to(&make(), spec.full_range(), spec.matrix(), &mut sink).unwrap();
  }
  assert_eq!(a, b);
}
