//! `Source`-trait coverage across the whole `walker!` table.
//!
//! Two proofs the per-row `@convert` arms hold up:
//!
//! * [`every_sinker_format_implements_source`] is the compile-time
//!   *exhaustiveness* proof — it names every sinker-supported source frame
//!   (under its feature gate) and asserts it implements [`Source`]. Bayer
//!   (`Bayer` / `Bayer16`) is deliberately absent: its options have no
//!   `Default`, so it is Tier-1-only by design.
//! * The per-family `*_parity` tests spot-check one representative per family /
//!   macro arm, asserting the [`Convert`] output is byte-identical to the
//!   hand-assembled `MixedSinker` + `{fmt}_to` path (mirroring the planar-YUV
//!   proof in [`super::tests`]).

use crate::{
  ChromaLocation, ColorInfo, ColorMatrix, ColorSpec, DynamicRange, PixelFormat, Primaries,
  Transfer,
  convert::Convert,
  resample::NoopResampler,
  sinker::{MixedSinker, MixedSinkerError},
};

/// Compile-time assertion that `F: Source`; monomorphising the call is what
/// forces the bound. A no-op at run time.
fn assert_source<F: crate::convert::Source>() {}

/// Names every frame type in a comma-separated list and asserts each is a
/// [`Source`] — the body of the per-feature blocks below.
// Every call site is feature-gated; combos that activate no block (e.g. `std` +
// `rgb-float` alone, where every format is Walker/Tier-1-only) leave it uncalled.
#[allow(unused_macros)]
macro_rules! assert_sources {
  ($($ty:ty),+ $(,)?) => { $(assert_source::<$ty>();)+ };
}

/// Exhaustiveness proof: every source format the sinkers support implements
/// [`Source`], so [`Convert`] works for all of them. Grouped by feature and by
/// macro arm (plain / `@const` endian-generic / `@const_bits` high-bit).
///
/// `Bayer` / `Bayer16` are intentionally omitted — `BayerOptions` has no
/// `Default`, so they cannot satisfy `Source::Options: FromSpec` and stay
/// Tier-1-only.
#[test]
fn every_sinker_format_implements_source() {
  #[cfg(feature = "xyz")]
  {
    assert_sources!(crate::frame::Xyz12Frame<'static, false>);
  }

  #[cfg(feature = "mono")]
  {
    assert_sources!(
      crate::frame::Pal8Frame<'static>,
      crate::frame::MonoblackFrame<'static>,
      crate::frame::MonowhiteFrame<'static>,
    );
  }

  #[cfg(feature = "yuv-planar")]
  {
    assert_sources!(
      crate::frame::Yuv420pFrame<'static>,
      crate::frame::Yuv422pFrame<'static>,
      crate::frame::Yuv444pFrame<'static>,
      crate::frame::Yuv440pFrame<'static>,
      crate::frame::Yuv410pFrame<'static>,
      crate::frame::Yuv411pFrame<'static>,
      crate::frame::Yuv420pFrame16<'static, 9, false>,
      crate::frame::Yuv420pFrame16<'static, 10, false>,
      crate::frame::Yuv420pFrame16<'static, 12, false>,
      crate::frame::Yuv420pFrame16<'static, 14, false>,
      crate::frame::Yuv420pFrame16<'static, 16, false>,
      crate::frame::Yuv422pFrame16<'static, 9, false>,
      crate::frame::Yuv422pFrame16<'static, 10, false>,
      crate::frame::Yuv422pFrame16<'static, 12, false>,
      crate::frame::Yuv422pFrame16<'static, 14, false>,
      crate::frame::Yuv422pFrame16<'static, 16, false>,
      crate::frame::Yuv444pFrame16<'static, 9, false>,
      crate::frame::Yuv444pFrame16<'static, 10, false>,
      crate::frame::Yuv444pFrame16<'static, 12, false>,
      crate::frame::Yuv444pFrame16<'static, 14, false>,
      crate::frame::Yuv444pFrame16<'static, 16, false>,
      crate::frame::Yuv444pMsbFrame<'static, 10, false>,
      crate::frame::Yuv444pMsbFrame<'static, 12, false>,
      crate::frame::Yuv440pFrame16<'static, 10, false>,
      crate::frame::Yuv440pFrame16<'static, 12, false>,
    );
  }

  #[cfg(feature = "yuv-semi-planar")]
  {
    assert_sources!(
      crate::frame::Nv12Frame<'static>,
      crate::frame::Nv16Frame<'static>,
      crate::frame::Nv21Frame<'static>,
      crate::frame::Nv24Frame<'static>,
      crate::frame::Nv42Frame<'static>,
      crate::frame::Nv20Frame<'static, false>,
      crate::frame::PnFrame<'static, 10, false>,
      crate::frame::PnFrame<'static, 12, false>,
      crate::frame::PnFrame<'static, 16, false>,
      crate::frame::PnFrame422<'static, 10, false>,
      crate::frame::PnFrame422<'static, 12, false>,
      crate::frame::PnFrame422<'static, 16, false>,
      crate::frame::PnFrame444<'static, 10, false>,
      crate::frame::PnFrame444<'static, 12, false>,
      crate::frame::PnFrame444<'static, 16, false>,
    );
  }

  #[cfg(feature = "yuv-packed")]
  {
    assert_sources!(
      crate::frame::Yuyv422Frame<'static>,
      crate::frame::Uyvy422Frame<'static>,
      crate::frame::Yvyu422Frame<'static>,
      crate::frame::Uyyvyy411Frame<'static>,
    );
  }

  #[cfg(feature = "y2xx")]
  {
    assert_sources!(
      crate::frame::Y2xxFrame<'static, 10, false>,
      crate::frame::Y2xxFrame<'static, 12, false>,
      crate::frame::Y2xxFrame<'static, 16, false>,
    );
  }

  #[cfg(feature = "yuv-444-packed")]
  {
    assert_sources!(
      crate::frame::VuyaFrame<'static>,
      crate::frame::VuyxFrame<'static>,
      crate::frame::AyuvFrame<'static>,
      crate::frame::UyvaFrame<'static>,
      crate::frame::Vyu444Frame<'static>,
      crate::frame::V30XFrame<'static>,
      crate::frame::V410Frame<'static, false>,
      crate::frame::Xv36Frame<'static, false>,
      crate::frame::Xv48Frame<'static, false>,
      crate::frame::Ayuv64Frame<'static, false>,
    );
  }

  #[cfg(feature = "v210")]
  {
    assert_sources!(crate::frame::V210Frame<'static, false>);
  }

  #[cfg(feature = "yuva")]
  {
    assert_sources!(
      crate::frame::Yuva420pFrame<'static>,
      crate::frame::Yuva422pFrame<'static>,
      crate::frame::Yuva444pFrame<'static>,
      crate::frame::Yuva420pFrame16<'static, 9, false>,
      crate::frame::Yuva420pFrame16<'static, 10, false>,
      crate::frame::Yuva420pFrame16<'static, 12, false>,
      crate::frame::Yuva420pFrame16<'static, 16, false>,
      crate::frame::Yuva422pFrame16<'static, 9, false>,
      crate::frame::Yuva422pFrame16<'static, 10, false>,
      crate::frame::Yuva422pFrame16<'static, 12, false>,
      crate::frame::Yuva422pFrame16<'static, 16, false>,
      crate::frame::Yuva444pFrame16<'static, 9, false>,
      crate::frame::Yuva444pFrame16<'static, 10, false>,
      crate::frame::Yuva444pFrame16<'static, 12, false>,
      crate::frame::Yuva444pFrame16<'static, 14, false>,
      crate::frame::Yuva444pFrame16<'static, 16, false>,
    );
  }

  #[cfg(feature = "rgb")]
  {
    assert_sources!(
      crate::frame::Rgb24Frame<'static>,
      crate::frame::Bgr24Frame<'static>,
      crate::frame::RgbaFrame<'static>,
      crate::frame::BgraFrame<'static>,
      crate::frame::ArgbFrame<'static>,
      crate::frame::AbgrFrame<'static>,
      crate::frame::XrgbFrame<'static>,
      crate::frame::RgbxFrame<'static>,
      crate::frame::XbgrFrame<'static>,
      crate::frame::BgrxFrame<'static>,
      crate::frame::Rgb48Frame<'static, false>,
      crate::frame::Bgr48Frame<'static, false>,
      crate::frame::Rgba64Frame<'static, false>,
      crate::frame::Bgra64Frame<'static, false>,
      crate::frame::Rgb96Frame<'static, false>,
      crate::frame::Rgba128Frame<'static, false>,
      crate::frame::X2Rgb10Frame<'static, false>,
      crate::frame::X2Bgr10Frame<'static, false>,
    );
  }

  // The packed-float-RGB `Source` impls ride the `@convert @const(cfg: …)` arm:
  // their blanket sink impls (hence `walk_mixed<R>`) exist only under
  // `any(feature = "rgb", feature = "yuv-planar")`, so the exhaustiveness proof
  // carries the same extra gate. Under `rgb-float` alone they are
  // Walker/Tier-1-only and (correctly) not `Source`.
  #[cfg(all(feature = "rgb-float", any(feature = "rgb", feature = "yuv-planar")))]
  {
    assert_sources!(
      crate::frame::Rgbf16Frame<'static, false>,
      crate::frame::Rgbf32Frame<'static, false>,
      crate::frame::Rgbaf16Frame<'static, false>,
      crate::frame::Rgbaf32Frame<'static, false>,
    );
  }

  #[cfg(feature = "rgb-legacy")]
  {
    assert_sources!(
      crate::frame::Rgb565Frame<'static>,
      crate::frame::Bgr565Frame<'static>,
      crate::frame::Rgb555Frame<'static>,
      crate::frame::Bgr555Frame<'static>,
      crate::frame::Rgb444Frame<'static>,
      crate::frame::Bgr444Frame<'static>,
      crate::frame::Rgb8Frame<'static>,
      crate::frame::Bgr8Frame<'static>,
      crate::frame::Rgb4ByteFrame<'static>,
      crate::frame::Bgr4ByteFrame<'static>,
      crate::frame::Rgb4Frame<'static>,
      crate::frame::Bgr4Frame<'static>,
    );
  }

  #[cfg(feature = "gray")]
  {
    assert_sources!(
      crate::frame::Gray8Frame<'static>,
      crate::frame::Ya8Frame<'static>,
      crate::frame::GrayNFrame<'static, 9, false>,
      crate::frame::GrayNFrame<'static, 10, false>,
      crate::frame::GrayNFrame<'static, 12, false>,
      crate::frame::GrayNFrame<'static, 14, false>,
      crate::frame::Gray16Frame<'static, false>,
      crate::frame::Gray32Frame<'static, false>,
      crate::frame::Ya16Frame<'static, false>,
      crate::frame::Grayf16Frame<'static, false>,
      crate::frame::Grayf32Frame<'static, false>,
      crate::frame::Yaf16Frame<'static, false>,
      crate::frame::Yaf32Frame<'static, false>,
    );
  }

  #[cfg(feature = "gbr")]
  {
    assert_sources!(
      crate::frame::GbrpFrame<'static>,
      crate::frame::GbrapFrame<'static>,
      crate::frame::GbrpHighBitFrame<'static, 9, false>,
      crate::frame::GbrpHighBitFrame<'static, 10, false>,
      crate::frame::GbrpHighBitFrame<'static, 12, false>,
      crate::frame::GbrpHighBitFrame<'static, 14, false>,
      crate::frame::GbrpHighBitFrame<'static, 16, false>,
      crate::frame::GbrapHighBitFrame<'static, 10, false>,
      crate::frame::GbrapHighBitFrame<'static, 12, false>,
      crate::frame::GbrapHighBitFrame<'static, 14, false>,
      crate::frame::GbrapHighBitFrame<'static, 16, false>,
      crate::frame::GbrpMsbFrame<'static, 10, false>,
      crate::frame::GbrpMsbFrame<'static, 12, false>,
      crate::frame::Gbrap32Frame<'static, false>,
      crate::frame::Gbrpf16Frame<'static, false>,
      crate::frame::Gbrpf32Frame<'static, false>,
      crate::frame::Gbrapf16Frame<'static, false>,
      crate::frame::Gbrapf32Frame<'static, false>,
    );
  }
}

/// A `Bt601` / limited / left-sited spec, matching [`super::tests`].
#[allow(dead_code)] // unused under single-format powersets without a parity test
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

/// Asserts the `Convert` rgb + luma output equals the manual `MixedSinker` +
/// `walk` path for a `YuvOptions`-family source (its walk consumes the spec's
/// `full_range` + `matrix`, which is exactly what `.spec(spec)` derives).
#[allow(dead_code)] // unused under single-format powersets without a parity test
fn assert_rgb_luma_parity<Fr, W>(w: usize, h: usize, make: impl Fn() -> Fr, walk: W)
where
  Fr: crate::convert::Source,
  W: Fn(
    &Fr,
    bool,
    ColorMatrix,
    &mut MixedSinker<'_, Fr::Marker, NoopResampler>,
  ) -> Result<(), MixedSinkerError>,
{
  let spec = spec_601();

  let mut convert_rgb = std::vec![0u8; w * h * 3];
  Convert::from(&make())
    .spec(spec)
    .rgb(&mut convert_rgb)
    .run()
    .unwrap();
  let mut manual_rgb = std::vec![0u8; w * h * 3];
  {
    let mut sink = MixedSinker::<Fr::Marker>::new(w, h)
      .with_rgb(&mut manual_rgb)
      .unwrap();
    // Mirror `Convert::run`: the sink-side spec is a planar-decode concern, so
    // `run` applies it only under `yuv-planar`; elsewhere the spec reaches the
    // walk solely through the derived `full_range` / `matrix` arguments.
    #[cfg(feature = "yuv-planar")]
    sink.set_color_spec(spec);
    walk(&make(), spec.full_range(), spec.matrix(), &mut sink).unwrap();
  }
  assert_eq!(convert_rgb, manual_rgb, "rgb");

  let mut convert_luma = std::vec![0u8; w * h];
  Convert::from(&make())
    .spec(spec)
    .luma(&mut convert_luma)
    .run()
    .unwrap();
  let mut manual_luma = std::vec![0u8; w * h];
  {
    let mut sink = MixedSinker::<Fr::Marker>::new(w, h)
      .with_luma(&mut manual_luma)
      .unwrap();
    // Mirror `Convert::run` (see the rgb arm above).
    #[cfg(feature = "yuv-planar")]
    sink.set_color_spec(spec);
    walk(&make(), spec.full_range(), spec.matrix(), &mut sink).unwrap();
  }
  assert_eq!(convert_luma, manual_luma, "luma");
}

// ---- Plain arm -----------------------------------------------------------

/// RGB (packed 8-bit) — plain `@convert` arm.
#[cfg(feature = "rgb")]
#[test]
fn rgb24_parity() {
  let px = [10u8, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
  assert_rgb_luma_parity(
    2,
    2,
    || crate::frame::Rgb24Frame::new(&px, 2, 2, 6),
    |f, fr, m, s| crate::source::rgb24_to(f, fr, m, s),
  );
}

/// Gray (single luma) — plain `@convert` arm.
#[cfg(feature = "gray")]
#[test]
fn gray8_parity() {
  let px = [30u8, 90, 150, 210];
  assert_rgb_luma_parity(
    2,
    2,
    || crate::frame::Gray8Frame::new(&px, 2, 2, 2),
    |f, fr, m, s| crate::source::gray8_to(f, fr, m, s),
  );
}

/// Packed YUV 4:2:2 — plain `@convert` arm.
#[cfg(feature = "yuv-packed")]
#[test]
fn yuyv422_parity() {
  let px = [
    40u8, 128, 80, 128, 120, 110, 160, 140, 200, 120, 30, 130, 70, 150, 240, 100,
  ];
  assert_rgb_luma_parity(
    4,
    2,
    || crate::frame::Yuyv422Frame::new(&px, 4, 2, 8),
    |f, fr, m, s| crate::source::yuyv422_to(f, fr, m, s),
  );
}

/// Semi-planar YUV 4:2:0 — plain `@convert` arm.
#[cfg(feature = "yuv-semi-planar")]
#[test]
fn nv12_parity() {
  let y = [16u8, 60, 110, 160, 200, 240, 32, 90];
  let uv = [128u8, 110, 140, 120];
  assert_rgb_luma_parity(
    4,
    2,
    || crate::frame::Nv12Frame::new(&y, &uv, 4, 2, 4, 4),
    |f, fr, m, s| crate::source::nv12_to(f, fr, m, s),
  );
}

/// Planar YUVA 4:2:0 (rgb + luma) — plain `@convert` arm.
#[cfg(feature = "yuva")]
#[test]
fn yuva420p_parity() {
  let y = [40u8, 80, 120, 160, 200, 240, 30, 70];
  let u = [100u8, 150];
  let v = [150u8, 100];
  let a = [10u8, 20, 30, 40, 50, 60, 70, 80];
  assert_rgb_luma_parity(
    4,
    2,
    || crate::frame::Yuva420pFrame::new(&y, &u, &v, &a, 4, 2, 4, 2, 2, 4),
    |f, fr, m, s| crate::source::yuva420p_to(f, fr, m, s),
  );
}

/// Planar YUVA 4:2:0 rgba — exercises the real source-alpha `attach_rgba` hook.
#[cfg(feature = "yuva")]
#[test]
fn yuva420p_rgba_parity() {
  let y = [40u8, 80, 120, 160, 200, 240, 30, 70];
  let u = [100u8, 150];
  let v = [150u8, 100];
  let a = [10u8, 20, 30, 40, 50, 60, 70, 80];
  let make = || crate::frame::Yuva420pFrame::new(&y, &u, &v, &a, 4, 2, 4, 2, 2, 4);
  let spec = spec_601();

  let mut convert_rgba = std::vec![0u8; 4 * 2 * 4];
  Convert::from(&make())
    .spec(spec)
    .rgba(&mut convert_rgba)
    .run()
    .unwrap();
  let mut manual_rgba = std::vec![0u8; 4 * 2 * 4];
  {
    let mut sink = MixedSinker::<crate::source::Yuva420p>::new(4, 2)
      .with_rgba(&mut manual_rgba)
      .unwrap()
      .with_color_spec(spec);
    crate::source::yuva420p_to(&make(), spec.full_range(), spec.matrix(), &mut sink).unwrap();
  }
  assert_eq!(convert_rgba, manual_rgba, "yuva420p rgba");
}

/// Packed YUV 4:4:4 (`Vuyx`, α padding) — plain `@convert` arm.
#[cfg(feature = "yuv-444-packed")]
#[test]
fn vuyx_parity() {
  let px = [
    0x80u8, 0x80, 0xA0, 0xFF, 0x70, 0x60, 0xC0, 0xFF, 0x90, 0x40, 0xB0, 0xFF, 0x50, 0x30, 0xD0,
    0xFF,
  ];
  assert_rgb_luma_parity(
    2,
    2,
    || crate::frame::VuyxFrame::new(&px, 2, 2, 8),
    |f, fr, m, s| crate::source::vuyx_to(f, fr, m, s),
  );
}

/// Palette (unit `Options`, `FromSpec` for `()`) — plain `@convert` arm.
#[cfg(feature = "mono")]
#[test]
fn pal8_parity() {
  let idx = [0u8, 1, 2, 3];
  let mut palette = [[0u8; 4]; 256];
  palette[0] = [10, 20, 30, 255];
  palette[1] = [40, 50, 60, 255];
  palette[2] = [70, 80, 90, 255];
  palette[3] = [100, 110, 120, 255];
  let make = || crate::frame::Pal8Frame::new(&idx, &palette, 2, 2, 2);
  let spec = spec_601();

  let mut convert_rgb = std::vec![0u8; 2 * 2 * 3];
  Convert::from(&make())
    .spec(spec)
    .rgb(&mut convert_rgb)
    .run()
    .unwrap();
  let mut manual_rgb = std::vec![0u8; 2 * 2 * 3];
  {
    let mut sink = MixedSinker::<crate::source::Pal8>::new(2, 2)
      .with_rgb(&mut manual_rgb)
      .unwrap()
      .with_color_spec(spec);
    crate::source::pal8_to(&make(), &mut sink).unwrap();
  }
  assert_eq!(convert_rgb, manual_rgb, "pal8 rgb");
}

// ---- `@const` (endian-generic) arm ---------------------------------------

/// Packed RGB 16-bit (`Rgb48`) — the endian-const `@convert @const` arm.
#[cfg(feature = "rgb")]
#[test]
fn rgb48_parity() {
  let px = [
    1000u16, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000,
  ];
  assert_rgb_luma_parity(
    2,
    2,
    || crate::frame::Rgb48Frame::new(&px, 2, 2, 6),
    |f, fr, m, s| crate::source::rgb48_to_endian::<_, false>(f, fr, m, s),
  );
}

/// Packed float RGBA (`Rgbaf32`) — the endian-const `@convert @const(cfg: …)`
/// arm. Its `Source` impl (so this parity check) exists only where the blanket
/// sink impl does: `any(feature = "rgb", feature = "yuv-planar")`.
#[cfg(all(feature = "rgb-float", any(feature = "rgb", feature = "yuv-planar")))]
#[test]
fn rgbaf32_parity() {
  let px = [
    0.1f32, 0.2, 0.3, 1.0, 0.4, 0.5, 0.6, 1.0, 0.7, 0.8, 0.9, 1.0, 0.15, 0.25, 0.35, 0.5,
  ];
  assert_rgb_luma_parity(
    2,
    2,
    || crate::frame::Rgbaf32Frame::new(&px, 2, 2, 8),
    |f, fr, m, s| crate::source::rgbaf32_to_endian::<_, false>(f, fr, m, s),
  );
}

/// XYZ12 (`Xyz12Options`, `FromSpec` derives the default gamut) — the
/// endian-const `@convert @const` arm.
#[cfg(feature = "xyz")]
#[test]
fn xyz12_parity() {
  let px = [
    100u16, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200,
  ];
  let make = || crate::frame::Xyz12Frame::new(&px, 2, 2, 6);
  let spec = spec_601();
  let gamut = crate::Xyz12Options::new().target_gamut();

  let mut convert_rgb = std::vec![0u8; 2 * 2 * 3];
  Convert::from(&make())
    .spec(spec)
    .rgb(&mut convert_rgb)
    .run()
    .unwrap();
  let mut manual_rgb = std::vec![0u8; 2 * 2 * 3];
  {
    let mut sink = MixedSinker::<crate::source::Xyz12>::new(2, 2)
      .with_rgb(&mut manual_rgb)
      .unwrap()
      .with_color_spec(spec);
    crate::source::xyz12_to::<false, _>(&make(), gamut, &mut sink).unwrap();
  }
  assert_eq!(convert_rgb, manual_rgb, "xyz12 rgb");
}

// ---- `@const_bits` (high-bit) arm ----------------------------------------

/// High-bit planar YUV 4:2:0 (`Yuv420p10`) — the `@convert @const_bits` arm.
#[cfg(feature = "yuv-planar")]
#[test]
fn yuv420p10_parity() {
  let y = [100u16, 200, 300, 400, 500, 600, 700, 800];
  let u = [512u16, 300];
  let v = [400u16, 600];
  assert_rgb_luma_parity(
    4,
    2,
    || crate::frame::Yuv420p10Frame::new(&y, &u, &v, 4, 2, 4, 2, 2),
    |f, fr, m, s| crate::source::yuv420p10_to_endian::<_, false>(f, fr, m, s),
  );
}
