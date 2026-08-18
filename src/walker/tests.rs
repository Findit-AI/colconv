// These tests sweep `full_range ∈ {false, true}` to exercise both kernel
// ranges directly (kernel correctness, not range-pinning), so they keep
// using the deprecated raw `YuvOptions` setters on purpose. The
// range-safe `ColorSpec` path is covered by `color_spec_*` below.
#![allow(deprecated)]

use super::*;
use crate::ColorMatrix;

// ---- Structural round-trip: Options builders / getters ----------------

#[test]
fn xyz12_options_default_is_dcip3() {
  assert_eq!(Xyz12Options::default(), Xyz12Options::new());
  assert_eq!(Xyz12Options::new().target_gamut(), KernelGamut::DciP3);
  // The honest default delegates to mediaframe's own gamut default.
  assert_eq!(Xyz12Options::default().target_gamut(), KernelGamut::DciP3);
}

#[test]
fn xyz12_options_with_target_gamut_round_trips() {
  for g in [
    KernelGamut::DciP3,
    KernelGamut::Rec709,
    KernelGamut::Rec2020,
  ] {
    assert_eq!(Xyz12Options::new().with_target_gamut(g).target_gamut(), g);
  }
}

#[test]
fn yuv_options_default_is_limited_bt709() {
  assert_eq!(YuvOptions::default(), YuvOptions::new());
  assert!(!YuvOptions::new().full_range());
  assert_eq!(YuvOptions::new().matrix(), KernelMatrix::Bt709);
}

#[test]
fn yuv_options_builders_and_mutators_round_trip() {
  // with_matrix
  assert_eq!(
    YuvOptions::new().with_matrix(KernelMatrix::Bt601).matrix(),
    KernelMatrix::Bt601
  );

  // bool consuming builders
  assert!(YuvOptions::new().with_full_range().full_range());
  assert!(YuvOptions::new().maybe_full_range(true).full_range());
  assert!(!YuvOptions::new().maybe_full_range(false).full_range());

  // bool in-place mutators (chainable via &mut Self)
  let mut o = YuvOptions::new();
  o.set_full_range();
  assert!(o.full_range());
  o.clear_full_range();
  assert!(!o.full_range());
  o.update_full_range(true);
  assert!(o.full_range());
}

// ---- ColorSpec: format-pinned range resolution (golden alias safety) --

#[test]
fn color_spec_pins_yuvj_to_full_regardless_of_stream_range() {
  // Every `yuvj*` alias pins full-range; the stream's claimed range loses,
  // so "Yuvj as limited-range" is unconstructable.
  for fmt in [
    PixelFormat::Yuvj420p,
    PixelFormat::Yuvj422p,
    PixelFormat::Yuvj444p,
    PixelFormat::Yuvj440p,
    PixelFormat::Yuvj411p,
  ] {
    for stream_range in [
      DynamicRange::Limited,
      DynamicRange::Unspecified,
      DynamicRange::Full,
      DynamicRange::other("unassigned-7"),
    ] {
      let spec = ColorSpec::resolve(fmt.clone(), stream_range.clone(), ColorMatrix::Bt601);
      assert!(
        spec.full_range(),
        "{fmt:?} pins full-range even with stream_range={stream_range:?}"
      );
      // The matrix passes through untouched.
      assert_eq!(spec.matrix(), ColorMatrix::Bt601);
    }
  }
}

#[test]
fn color_spec_leaves_yuv420p_stream_driven() {
  // No pin → the range follows the stream.
  let limited = ColorSpec::resolve(
    PixelFormat::Yuv420p,
    DynamicRange::Limited,
    ColorMatrix::Bt709,
  );
  assert!(!limited.full_range());
  let unspecified = ColorSpec::resolve(
    PixelFormat::Yuv420p,
    DynamicRange::Unspecified,
    ColorMatrix::Bt709,
  );
  assert!(!unspecified.full_range());
  let full = ColorSpec::resolve(PixelFormat::Yuv420p, DynamicRange::Full, ColorMatrix::Bt709);
  assert!(full.full_range());
  // Unknown ranges fall back conservatively to limited (studio swing).
  let unknown = ColorSpec::resolve(
    PixelFormat::Yuv420p,
    DynamicRange::other("unassigned-42"),
    ColorMatrix::Bt709,
  );
  assert!(!unknown.full_range());
}

#[test]
fn color_spec_resolves_format_aliases() {
  // Pure format aliases (no range pin): the canonical decode format is
  // followed and the range stays stream-driven.
  let gray = ColorSpec::resolve(
    PixelFormat::Gray8a,
    DynamicRange::Limited,
    ColorMatrix::Bt709,
  );
  assert_eq!(gray.format(), PixelFormat::Ya8);
  assert!(!gray.full_range());

  let xv30 = ColorSpec::resolve(PixelFormat::Xv30Le, DynamicRange::Full, ColorMatrix::Bt709);
  assert_eq!(xv30.format(), PixelFormat::V410Le);
  assert!(xv30.full_range());

  // A non-alias format resolves to itself.
  let plain = ColorSpec::resolve(
    PixelFormat::Yuv420p,
    DynamicRange::Limited,
    ColorMatrix::Bt709,
  );
  assert_eq!(plain.format(), PixelFormat::Yuv420p);
}

#[test]
fn yuv_options_from_color_spec_bridges_range_and_matrix() {
  // A pinned source's resolved range/matrix flow straight into YuvOptions.
  let spec = ColorSpec::resolve(
    PixelFormat::Yuvj420p,
    DynamicRange::Limited,
    ColorMatrix::Bt601,
  );
  let opts = YuvOptions::from_color_spec(&spec).unwrap();
  assert!(opts.full_range());
  assert_eq!(opts.matrix(), KernelMatrix::Bt601);

  // The bridge equals a direct raw construction of the resolved values.
  assert_eq!(
    opts,
    YuvOptions::new()
      .maybe_full_range(true)
      .with_matrix(KernelMatrix::Bt601)
  );
}

// ---- ColorSpec::from_info: carries the full colour Info ----------------

#[test]
fn color_spec_from_info_pins_yuvj_full_and_carries_metadata() {
  // `from_info` pins the `yuvj*` aliases full-range exactly like `resolve`
  // (the stream's claimed range loses) while threading the primaries /
  // transfer / chroma_location straight through.
  for fmt in [
    PixelFormat::Yuvj420p,
    PixelFormat::Yuvj422p,
    PixelFormat::Yuvj444p,
    PixelFormat::Yuvj440p,
    PixelFormat::Yuvj411p,
  ] {
    for stream_range in [
      DynamicRange::Limited,
      DynamicRange::Unspecified,
      DynamicRange::Full,
      DynamicRange::other("unassigned-7"),
    ] {
      let info = ColorInfo::new(
        Primaries::Bt2020,
        Transfer::SmpteSt2084Pq,
        ColorMatrix::Bt601,
        stream_range.clone(),
        ChromaLocation::TopLeft,
      );
      let spec = ColorSpec::from_info(fmt.clone(), info);
      assert!(
        spec.full_range(),
        "{fmt:?} pins full-range even with stream_range={stream_range:?}"
      );
      // The kernel-facing matrix flows through…
      assert_eq!(spec.matrix(), ColorMatrix::Bt601);
      // …and the carried-for-completeness metadata round-trips untouched.
      assert_eq!(spec.primaries(), Primaries::Bt2020);
      assert_eq!(spec.transfer(), Transfer::SmpteSt2084Pq);
      assert_eq!(spec.chroma_location(), ChromaLocation::TopLeft);
    }
  }
}

#[test]
fn color_spec_from_info_leaves_plain_format_stream_driven() {
  // No range pin → the resolved range follows `stream.range()`, the
  // canonical format is the format itself, and the metadata still carries.
  for (stream_range, expect_full) in [
    (DynamicRange::Limited, false),
    (DynamicRange::Unspecified, false),
    (DynamicRange::other("unassigned-42"), false),
    (DynamicRange::Full, true),
  ] {
    let info = ColorInfo::new(
      Primaries::Bt709,
      Transfer::Bt709,
      ColorMatrix::Bt709,
      stream_range.clone(),
      ChromaLocation::Left,
    );
    let spec = ColorSpec::from_info(PixelFormat::Yuv420p, info);
    assert_eq!(
      spec.full_range(),
      expect_full,
      "Yuv420p range follows stream_range={stream_range:?}"
    );
    assert_eq!(spec.format(), PixelFormat::Yuv420p);
    assert_eq!(spec.primaries(), Primaries::Bt709);
    assert_eq!(spec.transfer(), Transfer::Bt709);
    assert_eq!(spec.chroma_location(), ChromaLocation::Left);
  }
}

#[test]
fn color_spec_from_info_follows_format_alias() {
  // A pure format alias (no range pin) is followed to its canonical decode
  // form, and the carried metadata is independent of that aliasing.
  let info = ColorInfo::new(
    Primaries::Bt2020,
    Transfer::AribStdB67Hlg,
    ColorMatrix::Bt709,
    DynamicRange::Full,
    ChromaLocation::Center,
  );
  let spec = ColorSpec::from_info(PixelFormat::Xv30Le, info);
  assert_eq!(spec.format(), PixelFormat::V410Le);
  assert!(spec.full_range());
  assert_eq!(spec.primaries(), Primaries::Bt2020);
  assert_eq!(spec.transfer(), Transfer::AribStdB67Hlg);
  assert_eq!(spec.chroma_location(), ChromaLocation::Center);
}

#[test]
fn color_spec_resolve_defaults_carried_metadata_to_unspecified() {
  // The range-only entry point leaves the three carried fields at their
  // `Unspecified` defaults, while range + matrix resolve as before.
  for fmt in [
    PixelFormat::Yuv420p,
    PixelFormat::Yuvj420p,
    PixelFormat::Xv30Le,
  ] {
    let spec = ColorSpec::resolve(fmt, DynamicRange::Full, ColorMatrix::Bt709);
    assert_eq!(spec.primaries(), Primaries::Unspecified);
    assert_eq!(spec.transfer(), Transfer::Unspecified);
    assert_eq!(spec.chroma_location(), ChromaLocation::Unspecified);
    assert_eq!(spec.matrix(), ColorMatrix::Bt709);
  }
}

#[test]
fn color_spec_resolve_equals_from_info_with_range_only_info() {
  // `resolve` is exactly `from_info` fed an otherwise-`Unspecified` Info
  // carrying just the stream range + matrix — the delegation invariant.
  for fmt in [
    PixelFormat::Yuv420p,
    PixelFormat::Yuvj420p,
    PixelFormat::Yuvj444p,
    PixelFormat::Xv30Le,
    PixelFormat::Gray8a,
  ] {
    for stream_range in [
      DynamicRange::Limited,
      DynamicRange::Unspecified,
      DynamicRange::Full,
      DynamicRange::other("unassigned-9"),
    ] {
      for matrix in [KernelMatrix::Bt709, KernelMatrix::Bt601] {
        let via_resolve =
          ColorSpec::resolve(fmt.clone(), stream_range.clone(), ColorMatrix::from(matrix));
        let via_info = ColorSpec::from_info(
          fmt.clone(),
          ColorInfo::UNSPECIFIED
            .with_range(stream_range.clone())
            .with_matrix(ColorMatrix::from(matrix)),
        );
        assert_eq!(
          via_resolve, via_info,
          "resolve vs from_info mismatch (fmt={fmt:?}, range={stream_range:?}, matrix={matrix:?})"
        );
      }
    }
  }
}

#[cfg(feature = "bayer")]
mod bayer_options {
  use super::*;
  use crate::raw::{BayerDemosaic, BayerPattern, ColorCorrectionMatrix, WhiteBalance};

  #[test]
  fn new_defaults_demosaic_wb_ccm() {
    let o = BayerOptions::new(BayerPattern::Rggb);
    assert_eq!(o.pattern(), BayerPattern::Rggb);
    assert_eq!(o.demosaic(), BayerDemosaic::Bilinear);
    assert_eq!(o.wb(), WhiteBalance::neutral());
    assert_eq!(o.ccm(), ColorCorrectionMatrix::identity());
  }

  #[test]
  fn builders_round_trip() {
    let wb = WhiteBalance::new(1.5, 1.0, 2.0);
    let ccm = ColorCorrectionMatrix::new([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]);
    let o = BayerOptions::new(BayerPattern::Bggr)
      .with_demosaic(BayerDemosaic::Bilinear)
      .with_wb(wb)
      .with_ccm(ccm);
    assert_eq!(o.pattern(), BayerPattern::Bggr);
    assert_eq!(o.demosaic(), BayerDemosaic::Bilinear);
    assert_eq!(o.wb(), wb);
    assert_eq!(o.ccm(), ccm);
  }
}

// ---- Parity: Walker::walk byte-identical to xyz12_to directly ---------

#[cfg(feature = "xyz")]
mod xyz12_parity {
  use super::*;
  use crate::{
    frame::{Xyz12BeFrame, Xyz12LeFrame},
    sinker::MixedSinker,
    source::{Xyz12Be, Xyz12Le, xyz12_to},
  };

  /// Packs a 12-bit code into the high-bit-packed LE wire `u16`
  /// (active bits in `[15:4]`), host-independent.
  fn pack12_le(code: u16) -> u16 {
    u16::from_ne_bytes((code << 4).to_le_bytes())
  }

  /// BE-wire variant of [`pack12_le`].
  fn pack12_be(code: u16) -> u16 {
    u16::from_ne_bytes((code << 4).to_be_bytes())
  }

  /// A small ramp frame so different rows / columns carry different
  /// XYZ triples (exercises a non-degenerate conversion).
  fn ramp_frame<F>(width: u32, height: u32, pack: F) -> std::vec::Vec<u16>
  where
    F: Fn(u16) -> u16,
  {
    let w = width as usize;
    let h = height as usize;
    let mut buf = std::vec![0u16; w * h * 3];
    for (i, px) in buf.chunks_mut(3).enumerate() {
      let base = ((i * 37) % 4096) as u16;
      px[0] = pack(base);
      px[1] = pack((base + 411) % 4096);
      px[2] = pack((base + 822) % 4096);
    }
    buf
  }

  const W: u32 = 8;
  const H: u32 = 4;

  /// Asserts the LE u8-RGB output of `Walker::walk` equals `xyz12_to`.
  fn assert_parity_rgb_u8_le(gamut: KernelGamut) {
    let pix = ramp_frame(W, H, pack12_le);
    let src = Xyz12LeFrame::try_new(&pix, W, H, W * 3).unwrap();
    let opts = Xyz12Options::new().with_target_gamut(gamut);

    let mut via_walker = std::vec![0u8; (W * H * 3) as usize];
    let mut via_direct = std::vec![0u8; (W * H * 3) as usize];

    let mut sink_w = MixedSinker::<Xyz12Le>::new(W as usize, H as usize)
      .with_rgb(&mut via_walker)
      .unwrap();
    <Xyz12<false> as Walker<_>>::walk(&src, &opts, &mut sink_w).unwrap();

    let mut sink_d = MixedSinker::<Xyz12Le>::new(W as usize, H as usize)
      .with_rgb(&mut via_direct)
      .unwrap();
    xyz12_to(&src, gamut, &mut sink_d).unwrap();

    assert_eq!(via_walker, via_direct, "rgb u8 LE parity (gamut {gamut:?})");
  }

  /// Asserts the BE u16-RGB output of `Walker::walk` equals `xyz12_to`.
  fn assert_parity_rgb_u16_be(gamut: KernelGamut) {
    let pix = ramp_frame(W, H, pack12_be);
    let src = Xyz12BeFrame::try_new(&pix, W, H, W * 3).unwrap();
    let opts = Xyz12Options::new().with_target_gamut(gamut);

    let mut via_walker = std::vec![0u16; (W * H * 3) as usize];
    let mut via_direct = std::vec![0u16; (W * H * 3) as usize];

    let mut sink_w = MixedSinker::<Xyz12Be>::new(W as usize, H as usize)
      .with_rgb_u16(&mut via_walker)
      .unwrap();
    <Xyz12<true> as Walker<_>>::walk(&src, &opts, &mut sink_w).unwrap();

    let mut sink_d = MixedSinker::<Xyz12Be>::new(W as usize, H as usize)
      .with_rgb_u16(&mut via_direct)
      .unwrap();
    xyz12_to(&src, gamut, &mut sink_d).unwrap();

    assert_eq!(
      via_walker, via_direct,
      "rgb u16 BE parity (gamut {gamut:?})"
    );
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn walk_rgb_u8_le_matches_direct() {
    // >=2 gamuts x LE + rgb u8.
    assert_parity_rgb_u8_le(KernelGamut::DciP3);
    assert_parity_rgb_u8_le(KernelGamut::Rec709);
    assert_parity_rgb_u8_le(KernelGamut::Rec2020);
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn walk_rgb_u16_be_matches_direct() {
    // >=2 gamuts x BE + rgb u16.
    assert_parity_rgb_u16_be(KernelGamut::DciP3);
    assert_parity_rgb_u16_be(KernelGamut::Rec709);
    assert_parity_rgb_u16_be(KernelGamut::Rec2020);
  }

  /// Cross-check: the LE u16 + BE u8 corners too, so the full
  /// 2-gamut x {LE,BE} x {u8,u16} matrix is covered.
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn walk_le_u16_and_be_u8_match_direct() {
    for gamut in [KernelGamut::DciP3, KernelGamut::Rec709] {
      // LE, u16.
      {
        let pix = ramp_frame(W, H, pack12_le);
        let src = Xyz12LeFrame::try_new(&pix, W, H, W * 3).unwrap();
        let opts = Xyz12Options::new().with_target_gamut(gamut);
        let mut via_walker = std::vec![0u16; (W * H * 3) as usize];
        let mut via_direct = std::vec![0u16; (W * H * 3) as usize];
        let mut sw = MixedSinker::<Xyz12Le>::new(W as usize, H as usize)
          .with_rgb_u16(&mut via_walker)
          .unwrap();
        <Xyz12<false> as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();
        let mut sd = MixedSinker::<Xyz12Le>::new(W as usize, H as usize)
          .with_rgb_u16(&mut via_direct)
          .unwrap();
        xyz12_to(&src, gamut, &mut sd).unwrap();
        assert_eq!(
          via_walker, via_direct,
          "rgb u16 LE parity (gamut {gamut:?})"
        );
      }
      // BE, u8.
      {
        let pix = ramp_frame(W, H, pack12_be);
        let src = Xyz12BeFrame::try_new(&pix, W, H, W * 3).unwrap();
        let opts = Xyz12Options::new().with_target_gamut(gamut);
        let mut via_walker = std::vec![0u8; (W * H * 3) as usize];
        let mut via_direct = std::vec![0u8; (W * H * 3) as usize];
        let mut sw = MixedSinker::<Xyz12Be>::new(W as usize, H as usize)
          .with_rgb(&mut via_walker)
          .unwrap();
        <Xyz12<true> as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();
        let mut sd = MixedSinker::<Xyz12Be>::new(W as usize, H as usize)
          .with_rgb(&mut via_direct)
          .unwrap();
        xyz12_to(&src, gamut, &mut sd).unwrap();
        assert_eq!(via_walker, via_direct, "rgb u8 BE parity (gamut {gamut:?})");
      }
    }
  }
}

// ---- Parity: Bayer 8-bit + Bayer16 ------------------------------------

#[cfg(feature = "bayer")]
mod bayer_parity {
  use super::*;
  use crate::{
    frame::{
      Bayer10Frame, Bayer12Frame, Bayer14Frame, Bayer16Frame, BayerDemosaic, BayerFrame,
      BayerPattern, ColorCorrectionMatrix, WhiteBalance, bayer_to, bayer16_to,
    },
    sinker::MixedSinker,
    source::{Bayer, Bayer10, Bayer12, Bayer14, Bayer16Bit},
  };

  const W: u32 = 8;
  const H: u32 = 6;

  /// A column/row-varying `u8` Bayer plane (non-degenerate mosaic).
  fn ramp8() -> std::vec::Vec<u8> {
    let mut data = std::vec![0u8; (W * H) as usize];
    for (i, p) in data.iter_mut().enumerate() {
      *p = ((i * 17 + 3) % 251) as u8;
    }
    data
  }

  /// A column/row-varying low-packed `u16` Bayer plane for `BITS`.
  fn ramp16(bits: u32) -> std::vec::Vec<u16> {
    let max = (1u32 << bits) - 1;
    let mut data = std::vec![0u16; (W * H) as usize];
    for (i, p) in data.iter_mut().enumerate() {
      *p = (((i as u32) * 1103 + 7) % (max + 1)) as u16;
    }
    data
  }

  /// A non-neutral white balance + non-identity CCM, so the fused 3×3
  /// is exercised (a neutral/identity pair would hide a mis-forwarded
  /// param).
  fn nontrivial_opts(pattern: BayerPattern) -> BayerOptions {
    BayerOptions::new(pattern)
      .with_demosaic(BayerDemosaic::Bilinear)
      .with_wb(WhiteBalance::new(1.5, 1.0, 1.75))
      .with_ccm(ColorCorrectionMatrix::new([
        [1.2, -0.1, -0.1],
        [-0.2, 1.3, -0.1],
        [-0.1, -0.2, 1.4],
      ]))
  }

  const PATTERNS: [BayerPattern; 4] = [
    BayerPattern::Rggb,
    BayerPattern::Bggr,
    BayerPattern::Grbg,
    BayerPattern::Gbrg,
  ];

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn walk_bayer8_matches_direct() {
    let plane = ramp8();
    for pattern in PATTERNS {
      let opts = nontrivial_opts(pattern);
      let src = BayerFrame::try_new(&plane, W, H, W).unwrap();

      let mut via_walker = std::vec![0u8; (W * H * 3) as usize];
      let mut via_direct = std::vec![0u8; (W * H * 3) as usize];

      let mut sw = MixedSinker::<Bayer>::new(W as usize, H as usize)
        .with_rgb(&mut via_walker)
        .unwrap();
      <Bayer as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();

      let mut sd = MixedSinker::<Bayer>::new(W as usize, H as usize)
        .with_rgb(&mut via_direct)
        .unwrap();
      bayer_to(
        &src,
        opts.pattern(),
        opts.demosaic(),
        opts.wb(),
        opts.ccm(),
        &mut sd,
      )
      .unwrap();

      assert_eq!(
        via_walker, via_direct,
        "bayer8 parity (pattern {pattern:?})"
      );
    }
  }

  /// Drives the Bayer16 parity for one concrete `BITS` marker `$marker`
  /// + its `$frame` alias, across all four patterns.
  macro_rules! bayer16_case {
    ($name:ident, $marker:ty, $frame:ident, $bits:expr) => {
      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn $name() {
        let plane = ramp16($bits);
        for pattern in PATTERNS {
          let opts = nontrivial_opts(pattern);
          let src = $frame::try_new(&plane, W, H, W).unwrap();

          let mut via_walker = std::vec![0u8; (W * H * 3) as usize];
          let mut via_direct = std::vec![0u8; (W * H * 3) as usize];

          let mut sw = MixedSinker::<$marker>::new(W as usize, H as usize)
            .with_rgb(&mut via_walker)
            .unwrap();
          <$marker as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();

          let mut sd = MixedSinker::<$marker>::new(W as usize, H as usize)
            .with_rgb(&mut via_direct)
            .unwrap();
          bayer16_to::<$bits, _>(
            &src,
            opts.pattern(),
            opts.demosaic(),
            opts.wb(),
            opts.ccm(),
            &mut sd,
          )
          .unwrap();

          assert_eq!(
            via_walker, via_direct,
            "bayer16<{}> parity (pattern {pattern:?})",
            $bits
          );
        }
      }
    };
  }

  bayer16_case!(walk_bayer10_matches_direct, Bayer10, Bayer10Frame, 10);
  bayer16_case!(walk_bayer12_matches_direct, Bayer12, Bayer12Frame, 12);
  bayer16_case!(walk_bayer14_matches_direct, Bayer14, Bayer14Frame, 14);
  bayer16_case!(walk_bayer16_matches_direct, Bayer16Bit, Bayer16Frame, 16);
}

// ---- Parity: Pal8 (palette is frame-intrinsic; Options = ()) ----------

#[cfg(feature = "mono")]
mod pal8_parity {
  use super::*;
  use crate::{
    frame::Pal8Frame,
    sinker::MixedSinker,
    source::{Pal8, pal8_to},
  };

  const W: u32 = 8;
  const H: u32 = 4;

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn walk_pal8_matches_direct() {
    // A non-degenerate BGRA palette + an index ramp that hits a spread
    // of entries.
    let mut palette = [[0u8; 4]; 256];
    for (i, e) in palette.iter_mut().enumerate() {
      let i = i as u8;
      *e = [i, i.wrapping_mul(3), i.wrapping_mul(7), 255];
    }
    let mut data = std::vec![0u8; (W * H) as usize];
    for (i, p) in data.iter_mut().enumerate() {
      *p = ((i * 29 + 5) % 256) as u8;
    }
    let src = Pal8Frame::try_new(&data, &palette, W, H, W).unwrap();

    let mut via_walker = std::vec![0u8; (W * H * 3) as usize];
    let mut via_direct = std::vec![0u8; (W * H * 3) as usize];

    let mut sw = MixedSinker::<Pal8>::new(W as usize, H as usize)
      .with_rgb(&mut via_walker)
      .unwrap();
    // `()` Options — the palette rides on the frame, not the knobs.
    <Pal8 as Walker<_>>::walk(&src, &(), &mut sw).unwrap();

    let mut sd = MixedSinker::<Pal8>::new(W as usize, H as usize)
      .with_rgb(&mut via_direct)
      .unwrap();
    pal8_to(&src, &mut sd).unwrap();

    assert_eq!(via_walker, via_direct, "pal8 rgb parity");
  }
}

// ---- Parity: Monoblack + Monowhite (reuse YuvOptions) -----------------

#[cfg(feature = "mono")]
mod mono_parity {
  use super::*;
  use crate::{
    PixelSink,
    frame::{MonoblackFrame, MonowhiteFrame},
    sinker::MixedSinker,
    source::{
      Monoblack, MonoblackRow, MonoblackSink, Monowhite, MonowhiteRow, MonowhiteSink, monoblack_to,
      monowhite_to,
    },
  };

  const W: u32 = 13; // not a byte multiple → exercises the tail bits
  const H: u32 = 5;

  /// MSB-first 1-bit packed plane, `div_ceil(8)` bytes per row.
  fn packed_1bpp() -> (std::vec::Vec<u8>, u32) {
    let stride = W.div_ceil(8);
    let mut data = std::vec![0u8; (stride * H) as usize];
    for (i, b) in data.iter_mut().enumerate() {
      *b = ((i * 53 + 9) % 256) as u8;
    }
    (data, stride)
  }

  const MATRICES: [KernelMatrix; 2] = [KernelMatrix::Bt709, KernelMatrix::Bt601];

  // An instrumented sink recording each row's forwarded `(full_range, matrix)`.
  // The mono luma path expands bits to 0/255 and ignores that metadata, so a
  // byte-parity test cannot see a dropped forward — this can.
  macro_rules! metadata_probe {
    ($probe:ident, $row:ident, $sink:ident) => {
      #[derive(Default)]
      struct $probe {
        seen: std::vec::Vec<(bool, KernelMatrix)>,
      }
      impl PixelSink for $probe {
        type Input<'r> = $row<'r>;
        type Error = core::convert::Infallible;
        fn begin_frame(&mut self, _w: u32, _h: u32) -> Result<(), Self::Error> {
          Ok(())
        }
        fn process(&mut self, row: $row<'_>) -> Result<(), Self::Error> {
          self.seen.push((row.full_range(), row.matrix()));
          Ok(())
        }
      }
      impl $sink for $probe {}
    };
  }
  metadata_probe!(MonoblackProbe, MonoblackRow, MonoblackSink);
  metadata_probe!(MonowhiteProbe, MonowhiteRow, MonowhiteSink);

  /// The luma path discards `full_range`/`matrix`, so byte parity can't prove
  /// the Walker forwards them; instrument the sink and assert every emitted row
  /// carries exactly the supplied `YuvOptions` values.
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn walk_mono_forwards_full_range_and_matrix() {
    let (data, stride) = packed_1bpp();
    for full_range in [false, true] {
      for matrix in MATRICES {
        let opts = YuvOptions::new()
          .maybe_full_range(full_range)
          .with_matrix(matrix);

        let mut bp = MonoblackProbe::default();
        let src = MonoblackFrame::try_new(&data, W, H, stride).unwrap();
        <Monoblack as Walker<_>>::walk(&src, &opts, &mut bp).unwrap();
        assert!(!bp.seen.is_empty(), "monoblack walked at least one row");
        for &(fr, m) in &bp.seen {
          assert_eq!(
            (fr, m),
            (full_range, matrix),
            "monoblack forwards full_range/matrix into the row"
          );
        }

        let mut wp = MonowhiteProbe::default();
        let src = MonowhiteFrame::try_new(&data, W, H, stride).unwrap();
        <Monowhite as Walker<_>>::walk(&src, &opts, &mut wp).unwrap();
        assert!(!wp.seen.is_empty(), "monowhite walked at least one row");
        for &(fr, m) in &wp.seen {
          assert_eq!(
            (fr, m),
            (full_range, matrix),
            "monowhite forwards full_range/matrix into the row"
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
  fn walk_monoblack_matches_direct() {
    let (data, stride) = packed_1bpp();
    for full_range in [false, true] {
      for matrix in MATRICES {
        let opts = YuvOptions::new()
          .maybe_full_range(full_range)
          .with_matrix(matrix);
        let src = MonoblackFrame::try_new(&data, W, H, stride).unwrap();

        let mut via_walker = std::vec![0u8; (W * H) as usize];
        let mut via_direct = std::vec![0u8; (W * H) as usize];

        let mut sw = MixedSinker::<Monoblack>::new(W as usize, H as usize)
          .with_luma(&mut via_walker)
          .unwrap();
        <Monoblack as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();

        let mut sd = MixedSinker::<Monoblack>::new(W as usize, H as usize)
          .with_luma(&mut via_direct)
          .unwrap();
        monoblack_to(&src, opts.full_range(), opts.matrix(), &mut sd).unwrap();

        assert_eq!(
          via_walker, via_direct,
          "monoblack parity (full_range={full_range}, matrix={matrix:?})"
        );
      }
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn walk_monowhite_matches_direct() {
    let (data, stride) = packed_1bpp();
    for full_range in [false, true] {
      for matrix in MATRICES {
        let opts = YuvOptions::new()
          .maybe_full_range(full_range)
          .with_matrix(matrix);
        let src = MonowhiteFrame::try_new(&data, W, H, stride).unwrap();

        let mut via_walker = std::vec![0u8; (W * H) as usize];
        let mut via_direct = std::vec![0u8; (W * H) as usize];

        let mut sw = MixedSinker::<Monowhite>::new(W as usize, H as usize)
          .with_luma(&mut via_walker)
          .unwrap();
        <Monowhite as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();

        let mut sd = MixedSinker::<Monowhite>::new(W as usize, H as usize)
          .with_luma(&mut via_direct)
          .unwrap();
        monowhite_to(&src, opts.full_range(), opts.matrix(), &mut sd).unwrap();

        assert_eq!(
          via_walker, via_direct,
          "monowhite parity (full_range={full_range}, matrix={matrix:?})"
        );
      }
    }
  }
}

// ---- Parity: uniform YUV families (reuse YuvOptions) ------------------
//
// Each test asserts `<Marker as Walker<_>>::walk` is byte-identical to a
// direct walker call — same `MixedSinker::with_rgb` sink, same walker fn
// on both sides — across full/limited × Bt709/Bt601. Coverage spans every
// family plus each const-generic axis: an 8-bit and a high-bit case for
// planar / semi-planar / YUVA, a packed case, and a Y2xx case.
//
// The 8-bit families have no byte-order axis — their walk delegates to
// the plain `{fmt}_to` and the "direct" side calls the same fn, so
// byte-identity is structural. The high-bit families are endian-generic:
// the [`Walker`] impl delegates to the const-generic
// `{fmt}_to_endian::<_, BE>` (the LE `{fmt}_to` is just its `BE = false`
// shim), so **both** ends of the matrix are proven — the LE cases below
// drive the impl at the marker's `<const BE = false>` default against the
// LE frame alias, and the dedicated BE cases drive `Marker<true>` against
// the `{Fmt}BeFrame` alias, each compared to a direct
// `{fmt}_to_endian::<_, BE>` call. The non-degenerate ramps still guard
// against a mis-forwarded `full_range` / `matrix` (a swapped pair changes
// the output).

#[cfg(feature = "yuv-planar")]
mod yuv_planar_parity {
  use super::*;
  use crate::sinker::MixedSinker;

  const MATRICES: [KernelMatrix; 2] = [KernelMatrix::Bt709, KernelMatrix::Bt601];

  /// A deterministic, column/row-varying `u8` plane of `n` samples.
  fn ramp8(n: usize) -> std::vec::Vec<u8> {
    (0..n).map(|i| ((i * 17 + 3) % 251) as u8).collect()
  }

  /// A deterministic low-packed `u16` plane of `n` samples bounded to
  /// `bits` (active bits in the low end, matching the LE wire layout on
  /// the test host).
  fn ramp16(n: usize, bits: u32) -> std::vec::Vec<u16> {
    let max = (1u32 << bits) - 1;
    (0..n)
      .map(|i| (((i as u32) * 1103 + 7) & max) as u16)
      .collect()
  }

  /// The range-safe [`ColorSpec`] path is **byte-identical** to the
  /// equivalent raw `full_range` walk — `from_color_spec` only feeds the
  /// resolved `(full_range, matrix)` into the same kernels (additive
  /// equivalence). Covers both a stream-driven `yuv420p` and a *pinned*
  /// `yuvj420p` source (which decodes full-range as `yuv420p`, matching a
  /// correct raw full-range walk byte for byte).
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn color_spec_walk_matches_raw_full_range() {
    const W: u32 = 16;
    const H: u32 = 4;
    let cw = (W as usize) / 2;
    let ch = (H as usize).div_ceil(2);
    let y = ramp8((W * H) as usize);
    let u = ramp8(cw * ch);
    let v = ramp8(cw * ch);

    // (format, stream_range, expected resolved full_range).
    let cases = [
      (PixelFormat::Yuv420p, DynamicRange::Limited, false),
      (PixelFormat::Yuv420p, DynamicRange::Full, true),
      (PixelFormat::Yuv420p, DynamicRange::Unspecified, false),
      (PixelFormat::Yuvj420p, DynamicRange::Limited, true),
      (PixelFormat::Yuvj420p, DynamicRange::Unspecified, true),
    ];

    for (fmt, stream_range, expected_fr) in cases {
      for matrix in MATRICES {
        let spec = ColorSpec::resolve(fmt.clone(), stream_range.clone(), ColorMatrix::from(matrix));
        assert_eq!(spec.full_range(), expected_fr);
        let spec_opts = YuvOptions::from_color_spec(&spec).unwrap();

        // The equivalent raw path sets the resolved range directly.
        let raw_opts = YuvOptions::new()
          .maybe_full_range(expected_fr)
          .with_matrix(matrix);
        assert_eq!(spec_opts, raw_opts);

        let src =
          crate::frame::Yuv420pFrame::try_new(&y, &u, &v, W, H, W, cw as u32, cw as u32).unwrap();

        let mut via_spec = std::vec![0u8; (W * H * 3) as usize];
        let mut via_raw = std::vec![0u8; (W * H * 3) as usize];

        let mut ss = MixedSinker::<crate::source::Yuv420p>::new(W as usize, H as usize)
          .with_rgb(&mut via_spec)
          .unwrap();
        <crate::source::Yuv420p as Walker<_>>::walk(&src, &spec_opts, &mut ss).unwrap();

        let mut sr = MixedSinker::<crate::source::Yuv420p>::new(W as usize, H as usize)
          .with_rgb(&mut via_raw)
          .unwrap();
        <crate::source::Yuv420p as Walker<_>>::walk(&src, &raw_opts, &mut sr).unwrap();

        assert_eq!(
          via_spec, via_raw,
          "ColorSpec walk byte-identical to raw (fmt={fmt:?}, stream_range={stream_range:?}, matrix={matrix:?})"
        );
      }
    }
  }

  /// Drives a 3-plane planar YUV family (8-bit or high-bit-LE). `$ramp`
  /// builds a plane of the right element type; `$try_new` is the frame
  /// ctor (8-arg `y,u,v,w,h,ys,us,vs`); `$walker` the direct walker fn.
  macro_rules! planar3_case {
    (
      $name:ident, $marker:ty, $try_new:path, $walker:path,
      ramp = $ramp:expr, cw_div = $cw_div:expr, ch_div = $ch_div:expr,
    ) => {
      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn $name() {
        const W: u32 = 16;
        const H: u32 = 4;
        let cw = (W as usize) / $cw_div;
        let ch = (H as usize).div_ceil($ch_div);
        let make = $ramp;
        let y = make((W * H) as usize);
        let u = make(cw * ch);
        let v = make(cw * ch);

        for full_range in [false, true] {
          for matrix in MATRICES {
            let opts = YuvOptions::new().maybe_full_range(full_range).with_matrix(matrix);
            let src = $try_new(&y, &u, &v, W, H, W, cw as u32, cw as u32).unwrap();

            let mut via_walker = std::vec![0u8; (W * H * 3) as usize];
            let mut via_direct = std::vec![0u8; (W * H * 3) as usize];

            let mut sw = MixedSinker::<$marker>::new(W as usize, H as usize)
              .with_rgb(&mut via_walker)
              .unwrap();
            <$marker as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();

            let mut sd = MixedSinker::<$marker>::new(W as usize, H as usize)
              .with_rgb(&mut via_direct)
              .unwrap();
            $walker(&src, full_range, matrix, &mut sd).unwrap();

            assert_eq!(
              via_walker, via_direct,
              "{} parity (full_range={full_range}, matrix={matrix:?})",
              stringify!($name)
            );
          }
        }
      }
    };
  }

  // 8-bit: 4:2:0 (half/half chroma).
  planar3_case!(
    walk_yuv420p_matches_direct,
    crate::source::Yuv420p,
    crate::frame::Yuv420pFrame::try_new,
    crate::source::yuv420p_to,
    ramp = ramp8,
    cw_div = 2,
    ch_div = 2,
  );
  // High-bit-LE: 4:2:2 10-bit (half-width, full-height chroma).
  planar3_case!(
    walk_yuv422p10_matches_direct,
    crate::source::Yuv422p10,
    crate::frame::Yuv422p10LeFrame::try_new,
    crate::source::yuv422p10_to,
    ramp = |n| ramp16(n, 10),
    cw_div = 2,
    ch_div = 1,
  );
  // High-bit-LE: 4:4:4 16-bit (full chroma; the i64 chroma_sum kernel).
  planar3_case!(
    walk_yuv444p16_matches_direct,
    crate::source::Yuv444p16,
    crate::frame::Yuv444p16LeFrame::try_new,
    crate::source::yuv444p16_to,
    ramp = |n| ramp16(n, 16),
    cw_div = 1,
    ch_div = 1,
  );

  /// BE sibling of [`planar3_case`]: drives the `@const_bits` impl at
  /// `Marker<true>` against a `{Fmt}BeFrame` and compares to a direct
  /// `{fmt}_to_endian::<_, true>` call. `$marker` is the BE-pinned marker
  /// (`Yuv422p10<true>`); `$walker_endian` the const-generic walker.
  macro_rules! planar3_be_case {
    (
      $name:ident, $marker:ty, $try_new:path, $walker_endian:path,
      ramp = $ramp:expr, cw_div = $cw_div:expr, ch_div = $ch_div:expr,
    ) => {
      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn $name() {
        const W: u32 = 16;
        const H: u32 = 4;
        let cw = (W as usize) / $cw_div;
        let ch = (H as usize).div_ceil($ch_div);
        let make = $ramp;
        let y = make((W * H) as usize);
        let u = make(cw * ch);
        let v = make(cw * ch);

        for full_range in [false, true] {
          for matrix in MATRICES {
            let opts = YuvOptions::new().maybe_full_range(full_range).with_matrix(matrix);
            let src = $try_new(&y, &u, &v, W, H, W, cw as u32, cw as u32).unwrap();

            let mut via_walker = std::vec![0u8; (W * H * 3) as usize];
            let mut via_direct = std::vec![0u8; (W * H * 3) as usize];

            let mut sw = MixedSinker::<$marker>::new(W as usize, H as usize)
              .with_rgb(&mut via_walker)
              .unwrap();
            <$marker as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();

            let mut sd = MixedSinker::<$marker>::new(W as usize, H as usize)
              .with_rgb(&mut via_direct)
              .unwrap();
            $walker_endian(&src, full_range, matrix, &mut sd).unwrap();

            assert_eq!(
              via_walker, via_direct,
              "{} BE parity (full_range={full_range}, matrix={matrix:?})",
              stringify!($name)
            );
          }
        }
      }
    };
  }

  // High-bit-BE: 4:2:2 10-bit (half-width, full-height chroma).
  planar3_be_case!(
    walk_yuv422p10_be_matches_direct,
    crate::source::Yuv422p10<true>,
    crate::frame::Yuv422p10BeFrame::try_new,
    crate::source::yuv422p10_to_endian::<_, true>,
    ramp = |n| ramp16(n, 10),
    cw_div = 2,
    ch_div = 1,
  );
}

#[cfg(feature = "yuv-semi-planar")]
mod yuv_semi_planar_parity {
  use super::*;
  use crate::sinker::MixedSinker;

  const MATRICES: [KernelMatrix; 2] = [KernelMatrix::Bt709, KernelMatrix::Bt601];

  fn ramp8(n: usize) -> std::vec::Vec<u8> {
    (0..n).map(|i| ((i * 23 + 5) % 251) as u8).collect()
  }
  fn ramp16(n: usize, bits: u32) -> std::vec::Vec<u16> {
    let max = (1u32 << bits) - 1;
    (0..n)
      .map(|i| (((i as u32) * 1399 + 11) & max) as u16)
      .collect()
  }

  /// Drives a semi-planar (Y + interleaved chroma) family. `$try_new`
  /// is the 6-arg ctor (`y,uv,w,h,ys,uvs`); `chroma_w_factor` is the UV
  /// row length in elements as a multiple of width; `ch_div` the chroma
  /// vertical divisor.
  macro_rules! semi_planar_case {
    (
      $name:ident, $marker:ty, $try_new:path, $walker:path,
      ramp = $ramp:expr, chroma_w_factor = $cwf:expr, ch_div = $ch_div:expr,
    ) => {
      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn $name() {
        const W: u32 = 16;
        const H: u32 = 4;
        let uv_row = ($cwf as usize) * (W as usize);
        let ch = (H as usize).div_ceil($ch_div);
        let make = $ramp;
        let y = make((W * H) as usize);
        let uv = make(uv_row * ch);

        for full_range in [false, true] {
          for matrix in MATRICES {
            let opts = YuvOptions::new().maybe_full_range(full_range).with_matrix(matrix);
            let src = $try_new(&y, &uv, W, H, W, uv_row as u32).unwrap();

            let mut via_walker = std::vec![0u8; (W * H * 3) as usize];
            let mut via_direct = std::vec![0u8; (W * H * 3) as usize];

            let mut sw = MixedSinker::<$marker>::new(W as usize, H as usize)
              .with_rgb(&mut via_walker)
              .unwrap();
            <$marker as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();

            let mut sd = MixedSinker::<$marker>::new(W as usize, H as usize)
              .with_rgb(&mut via_direct)
              .unwrap();
            $walker(&src, full_range, matrix, &mut sd).unwrap();

            assert_eq!(
              via_walker, via_direct,
              "{} parity (full_range={full_range}, matrix={matrix:?})",
              stringify!($name)
            );
          }
        }
      }
    };
  }

  // 8-bit: Nv12 (4:2:0, UV interleaved half-width/half-height).
  semi_planar_case!(
    walk_nv12_matches_direct,
    crate::source::Nv12,
    crate::frame::Nv12Frame::try_new,
    crate::source::nv12_to,
    ramp = ramp8,
    chroma_w_factor = 1,
    ch_div = 2,
  );
  // High-bit-LE: P010 (4:2:0 10-bit packed u16, MSB-justified).
  semi_planar_case!(
    walk_p010_matches_direct,
    crate::source::P010,
    crate::frame::P010LeFrame::try_new,
    crate::source::p010_to,
    ramp = |n| ramp16(n, 16),
    chroma_w_factor = 1,
    ch_div = 2,
  );

  /// BE sibling of [`semi_planar_case`]: drives the `@const_bits` impl at
  /// `Marker<true>` against a `{Fmt}BeFrame` and compares to a direct
  /// `{fmt}_to_endian::<_, true>` call.
  macro_rules! semi_planar_be_case {
    (
      $name:ident, $marker:ty, $try_new:path, $walker_endian:path,
      ramp = $ramp:expr, chroma_w_factor = $cwf:expr, ch_div = $ch_div:expr,
    ) => {
      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn $name() {
        const W: u32 = 16;
        const H: u32 = 4;
        let uv_row = ($cwf as usize) * (W as usize);
        let ch = (H as usize).div_ceil($ch_div);
        let make = $ramp;
        let y = make((W * H) as usize);
        let uv = make(uv_row * ch);

        for full_range in [false, true] {
          for matrix in MATRICES {
            let opts = YuvOptions::new().maybe_full_range(full_range).with_matrix(matrix);
            let src = $try_new(&y, &uv, W, H, W, uv_row as u32).unwrap();

            let mut via_walker = std::vec![0u8; (W * H * 3) as usize];
            let mut via_direct = std::vec![0u8; (W * H * 3) as usize];

            let mut sw = MixedSinker::<$marker>::new(W as usize, H as usize)
              .with_rgb(&mut via_walker)
              .unwrap();
            <$marker as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();

            let mut sd = MixedSinker::<$marker>::new(W as usize, H as usize)
              .with_rgb(&mut via_direct)
              .unwrap();
            $walker_endian(&src, full_range, matrix, &mut sd).unwrap();

            assert_eq!(
              via_walker, via_direct,
              "{} BE parity (full_range={full_range}, matrix={matrix:?})",
              stringify!($name)
            );
          }
        }
      }
    };
  }

  // High-bit-BE: P010 (4:2:0 10-bit packed u16, MSB-justified).
  semi_planar_be_case!(
    walk_p010_be_matches_direct,
    crate::source::P010<true>,
    crate::frame::P010BeFrame::try_new,
    crate::source::p010_to_endian::<_, true>,
    ramp = |n| ramp16(n, 16),
    chroma_w_factor = 1,
    ch_div = 2,
  );
}

#[cfg(feature = "yuv-packed")]
mod yuv_packed_parity {
  use super::*;
  use crate::{
    frame::Yuyv422Frame,
    sinker::MixedSinker,
    source::{Yuyv422, yuyv422_to},
  };

  const MATRICES: [KernelMatrix; 2] = [KernelMatrix::Bt709, KernelMatrix::Bt601];

  /// Packed YUYV 4:2:2 — single buffer, `width * 2` u8 per row.
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn walk_yuyv422_matches_direct() {
    const W: u32 = 16;
    const H: u32 = 4;
    let buf: std::vec::Vec<u8> = (0..(W * H * 2) as usize)
      .map(|i| ((i * 19 + 7) % 251) as u8)
      .collect();

    for full_range in [false, true] {
      for matrix in MATRICES {
        let opts = YuvOptions::new()
          .maybe_full_range(full_range)
          .with_matrix(matrix);
        let src = Yuyv422Frame::try_new(&buf, W, H, W * 2).unwrap();

        let mut via_walker = std::vec![0u8; (W * H * 3) as usize];
        let mut via_direct = std::vec![0u8; (W * H * 3) as usize];

        let mut sw = MixedSinker::<Yuyv422>::new(W as usize, H as usize)
          .with_rgb(&mut via_walker)
          .unwrap();
        <Yuyv422 as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();

        let mut sd = MixedSinker::<Yuyv422>::new(W as usize, H as usize)
          .with_rgb(&mut via_direct)
          .unwrap();
        yuyv422_to(&src, full_range, matrix, &mut sd).unwrap();

        assert_eq!(
          via_walker, via_direct,
          "yuyv422 parity (full_range={full_range}, matrix={matrix:?})"
        );
      }
    }
  }
}

#[cfg(feature = "y2xx")]
mod y2xx_parity {
  use super::*;
  use crate::{
    frame::{Y210BeFrame, Y210LeFrame},
    sinker::MixedSinker,
    source::{Y210, y210_to, y210_to_endian},
  };

  const MATRICES: [KernelMatrix; 2] = [KernelMatrix::Bt709, KernelMatrix::Bt601];

  /// MSB-justified 10-bit Y210 wire samples (low 6 bits zero).
  fn y210_ramp() -> std::vec::Vec<u16> {
    const W: u32 = 16;
    const H: u32 = 4;
    (0..(W * H * 2) as usize)
      .map(|i| (((i as u32 * 1103 + 7) & 0x03FF) << 6) as u16)
      .collect()
  }

  /// Y210 LE — packed 4:2:2 10-bit, single `u16` buffer, `width * 2` per
  /// row. Drives the `@const_bits` impl at the marker's `<const BE =
  /// false>` default against the LE `y210_to` wrapper.
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn walk_y210_matches_direct() {
    const W: u32 = 16;
    const H: u32 = 4;
    let buf = y210_ramp();

    for full_range in [false, true] {
      for matrix in MATRICES {
        let opts = YuvOptions::new()
          .maybe_full_range(full_range)
          .with_matrix(matrix);
        let src = Y210LeFrame::try_new(&buf, W, H, W * 2).unwrap();

        let mut via_walker = std::vec![0u8; (W * H * 3) as usize];
        let mut via_direct = std::vec![0u8; (W * H * 3) as usize];

        let mut sw = MixedSinker::<Y210>::new(W as usize, H as usize)
          .with_rgb(&mut via_walker)
          .unwrap();
        <Y210 as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();

        let mut sd = MixedSinker::<Y210>::new(W as usize, H as usize)
          .with_rgb(&mut via_direct)
          .unwrap();
        y210_to(&src, full_range, matrix, &mut sd).unwrap();

        assert_eq!(
          via_walker, via_direct,
          "y210 parity (full_range={full_range}, matrix={matrix:?})"
        );
      }
    }
  }

  /// Y210 BE — drives the `@const_bits` impl at `Y210<true>` against the
  /// `Y210BeFrame` alias, compared to a direct `y210_to_endian::<_, true>`.
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn walk_y210_be_matches_direct() {
    const W: u32 = 16;
    const H: u32 = 4;
    let buf = y210_ramp();

    for full_range in [false, true] {
      for matrix in MATRICES {
        let opts = YuvOptions::new()
          .maybe_full_range(full_range)
          .with_matrix(matrix);
        let src = Y210BeFrame::try_new(&buf, W, H, W * 2).unwrap();

        let mut via_walker = std::vec![0u8; (W * H * 3) as usize];
        let mut via_direct = std::vec![0u8; (W * H * 3) as usize];

        let mut sw = MixedSinker::<Y210<true>>::new(W as usize, H as usize)
          .with_rgb(&mut via_walker)
          .unwrap();
        <Y210<true> as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();

        let mut sd = MixedSinker::<Y210<true>>::new(W as usize, H as usize)
          .with_rgb(&mut via_direct)
          .unwrap();
        y210_to_endian::<_, true>(&src, full_range, matrix, &mut sd).unwrap();

        assert_eq!(
          via_walker, via_direct,
          "y210 BE parity (full_range={full_range}, matrix={matrix:?})"
        );
      }
    }
  }
}

// ---- Parity: packed YUV 4:4:4 families (reuse YuvOptions) -------------
//
// Packed 4:4:4 sources. Each test asserts `<Marker as Walker<_>>::walk` is
// byte-identical to a direct walker call into the same
// `MixedSinker::with_rgb` sink, across full/limited × Bt709/Bt601. The
// packed YUV → RGB output is matrix-weighted + full_range-scaled, so
// `with_rgb` genuinely exercises the `(full_range, matrix)` forwarding (a
// swapped pair changes the output). Coverage spans both topologies: the
// plain arm (8-bit Vuya / Vuyx, LE-only 10-bit V30X) and the `@const BE`
// arm (V410 10-bit, Xv36 12-bit, Ayuv64 16-bit + α) — for the latter,
// BOTH LE and BE, since the impl delegates to the const-generic
// `{fmt}_to_endian` (the LE `{fmt}_to` is its `BE = false` shim).

#[cfg(feature = "yuv-444-packed")]
mod yuv_444_packed_parity {
  use super::*;
  use crate::sinker::MixedSinker;

  const W: u32 = 16;
  const H: u32 = 4;
  const MATRICES: [KernelMatrix; 2] = [KernelMatrix::Bt709, KernelMatrix::Bt601];

  /// A deterministic, column/row-varying `u8` buffer of `n` bytes.
  fn ramp8(n: usize) -> std::vec::Vec<u8> {
    (0..n).map(|i| ((i * 19 + 7) % 251) as u8).collect()
  }

  /// A deterministic `u16` buffer of `n` samples (full 16-bit range).
  fn ramp16(n: usize) -> std::vec::Vec<u16> {
    (0..n).map(|i| ((i * 1103 + 7) & 0xFFFF) as u16).collect()
  }

  /// A deterministic `u32` buffer of `n` words (full 32-bit range).
  fn ramp32(n: usize) -> std::vec::Vec<u32> {
    (0..n)
      .map(|i| (i as u32).wrapping_mul(2654435761).wrapping_add(7))
      .collect()
  }

  /// Drives a plain-arm packed 4:4:4 family (no byte-order axis). `$ramp`
  /// builds the single packed buffer of `row_elems * H` elements;
  /// `$try_new` is the 4-arg ctor (`packed, w, h, stride`); `$walker` the
  /// direct walker fn; `$row_elems` the per-row element count.
  macro_rules! packed444 {
    ($name:ident, $marker:ty, $try_new:path, $walker:path, $ramp:expr, $row_elems:expr) => {
      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn $name() {
        let row_elems = $row_elems as usize;
        let buf = ($ramp)(row_elems * H as usize);
        for full_range in [false, true] {
          for matrix in MATRICES {
            let opts = YuvOptions::new()
              .maybe_full_range(full_range)
              .with_matrix(matrix);
            let src = $try_new(&buf, W, H, row_elems as u32).unwrap();

            let mut via_walker = std::vec![0u8; (W * H * 3) as usize];
            let mut via_direct = std::vec![0u8; (W * H * 3) as usize];

            let mut sw = MixedSinker::<$marker>::new(W as usize, H as usize)
              .with_rgb(&mut via_walker)
              .unwrap();
            <$marker as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();

            let mut sd = MixedSinker::<$marker>::new(W as usize, H as usize)
              .with_rgb(&mut via_direct)
              .unwrap();
            $walker(&src, full_range, matrix, &mut sd).unwrap();

            assert_eq!(
              via_walker, via_direct,
              "{} parity (full_range={full_range}, matrix={matrix:?})",
              stringify!($name)
            );
          }
        }
      }
    };
  }

  // Plain arm: Vuya (8-bit, V/U/Y/A; `width * 4` bytes per row).
  packed444!(
    walk_vuya_matches_direct,
    crate::source::Vuya,
    crate::frame::VuyaFrame::try_new,
    crate::source::vuya_to,
    ramp8,
    W * 4
  );
  // Plain arm: Vuyx (8-bit, α padding; `width * 4` bytes per row).
  packed444!(
    walk_vuyx_matches_direct,
    crate::source::Vuyx,
    crate::frame::VuyxFrame::try_new,
    crate::source::vuyx_to,
    ramp8,
    W * 4
  );
  // Plain arm: V30X (10-bit, one u32 word per pixel; `width` u32 per row).
  packed444!(
    walk_v30x_matches_direct,
    crate::source::V30X,
    crate::frame::V30XFrame::try_new,
    crate::source::v30x_to,
    ramp32,
    W
  );

  /// `@const BE` sibling: drives the LE impl at the marker's `<const BE =
  /// false>` default against the `{Fmt}LeFrame` + LE `{fmt}_to` wrapper,
  /// and (in the `_be` test) the BE impl at `Marker<true>` against the
  /// `{Fmt}BeFrame` + a direct `{fmt}_to_endian::<_, true>` call.
  macro_rules! packed444_be {
    (
      $name:ident, $marker:ty, $try_new:path, $walker:path,
      $ramp:expr, $row_elems:expr
    ) => {
      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn $name() {
        let row_elems = $row_elems as usize;
        let buf = ($ramp)(row_elems * H as usize);
        for full_range in [false, true] {
          for matrix in MATRICES {
            let opts = YuvOptions::new()
              .maybe_full_range(full_range)
              .with_matrix(matrix);
            let src = $try_new(&buf, W, H, row_elems as u32).unwrap();

            let mut via_walker = std::vec![0u8; (W * H * 3) as usize];
            let mut via_direct = std::vec![0u8; (W * H * 3) as usize];

            let mut sw = MixedSinker::<$marker>::new(W as usize, H as usize)
              .with_rgb(&mut via_walker)
              .unwrap();
            <$marker as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();

            let mut sd = MixedSinker::<$marker>::new(W as usize, H as usize)
              .with_rgb(&mut via_direct)
              .unwrap();
            $walker(&src, full_range, matrix, &mut sd).unwrap();

            assert_eq!(
              via_walker, via_direct,
              "{} parity (full_range={full_range}, matrix={matrix:?})",
              stringify!($name)
            );
          }
        }
      }
    };
  }

  // `@const BE` arm — V410 (10-bit, one u32 per pixel), LE + BE.
  packed444_be!(
    walk_v410_le_matches_direct,
    crate::source::V410,
    crate::frame::V410LeFrame::try_new,
    crate::source::v410_to,
    ramp32,
    W
  );
  packed444_be!(
    walk_v410_be_matches_direct,
    crate::source::V410<true>,
    crate::frame::V410BeFrame::try_new,
    crate::source::v410_to_endian::<_, true>,
    ramp32,
    W
  );
  // `@const BE` arm — Xv36 (12-bit, U/Y/V/A u16 quadruple), LE + BE.
  packed444_be!(
    walk_xv36_le_matches_direct,
    crate::source::Xv36,
    crate::frame::Xv36LeFrame::try_new,
    crate::source::xv36_to,
    ramp16,
    W * 4
  );
  packed444_be!(
    walk_xv36_be_matches_direct,
    crate::source::Xv36<true>,
    crate::frame::Xv36BeFrame::try_new,
    crate::source::xv36_to_endian::<_, true>,
    ramp16,
    W * 4
  );
  // `@const BE` arm — Ayuv64 (16-bit + source α, A/Y/U/V u16 quad), LE + BE.
  packed444_be!(
    walk_ayuv64_le_matches_direct,
    crate::source::Ayuv64,
    crate::frame::Ayuv64LeFrame::try_new,
    crate::source::ayuv64_to,
    ramp16,
    W * 4
  );
  packed444_be!(
    walk_ayuv64_be_matches_direct,
    crate::source::Ayuv64<true>,
    crate::frame::Ayuv64BeFrame::try_new,
    crate::source::ayuv64_to_endian::<_, true>,
    ramp16,
    W * 4
  );
}

// ---- Parity: packed YUV 4:2:2 10-bit V210 (reuse YuvOptions) ----------
//
// V210 packs 6 pixels per 16-byte block. Endian-generic: the `@const BE`
// impl delegates to the const-generic `v210_to_endian::<_, BE>` (the LE
// `v210_to` is its `BE = false` shim), so both halves of the matrix are
// proven — the LE case drives the impl at the marker's `<const BE =
// false>` default against the `V210LeFrame` alias, and the BE case drives
// `V210<true>` against `V210BeFrame`. The packed YUV → RGB output is
// matrix-weighted + full_range-scaled, so `with_rgb` exercises the
// `(full_range, matrix)` forwarding across full/limited × Bt709/Bt601.

#[cfg(feature = "v210")]
mod v210_parity {
  use super::*;
  use crate::{
    frame::{V210BeFrame, V210LeFrame},
    sinker::MixedSinker,
    source::{V210, v210_to, v210_to_endian},
  };

  // `width = 12` is a multiple of 6 (whole v210 blocks); stride is the
  // per-row byte count `(width / 6) * 16`.
  const W: u32 = 12;
  const H: u32 = 4;
  const STRIDE: u32 = W.div_ceil(6) * 16;
  const MATRICES: [KernelMatrix; 2] = [KernelMatrix::Bt709, KernelMatrix::Bt601];

  /// A deterministic, column/row-varying `u8` v210 buffer.
  fn ramp(n: usize) -> std::vec::Vec<u8> {
    (0..n).map(|i| ((i * 19 + 7) % 251) as u8).collect()
  }

  /// V210 LE — drives the `@const BE` impl at the marker's `<const BE =
  /// false>` default against the LE `v210_to` wrapper.
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn walk_v210_le_matches_direct() {
    let buf = ramp((STRIDE * H) as usize);
    for full_range in [false, true] {
      for matrix in MATRICES {
        let opts = YuvOptions::new()
          .maybe_full_range(full_range)
          .with_matrix(matrix);
        let src = V210LeFrame::try_new(&buf, W, H, STRIDE).unwrap();

        let mut via_walker = std::vec![0u8; (W * H * 3) as usize];
        let mut via_direct = std::vec![0u8; (W * H * 3) as usize];

        let mut sw = MixedSinker::<V210>::new(W as usize, H as usize)
          .with_rgb(&mut via_walker)
          .unwrap();
        <V210 as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();

        let mut sd = MixedSinker::<V210>::new(W as usize, H as usize)
          .with_rgb(&mut via_direct)
          .unwrap();
        v210_to(&src, full_range, matrix, &mut sd).unwrap();

        assert_eq!(
          via_walker, via_direct,
          "v210 LE parity (full_range={full_range}, matrix={matrix:?})"
        );
      }
    }
  }

  /// V210 BE — drives the `@const BE` impl at `V210<true>` against the
  /// `V210BeFrame` alias, compared to a direct `v210_to_endian::<_, true>`.
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn walk_v210_be_matches_direct() {
    let buf = ramp((STRIDE * H) as usize);
    for full_range in [false, true] {
      for matrix in MATRICES {
        let opts = YuvOptions::new()
          .maybe_full_range(full_range)
          .with_matrix(matrix);
        let src = V210BeFrame::try_new(&buf, W, H, STRIDE).unwrap();

        let mut via_walker = std::vec![0u8; (W * H * 3) as usize];
        let mut via_direct = std::vec![0u8; (W * H * 3) as usize];

        let mut sw = MixedSinker::<V210<true>>::new(W as usize, H as usize)
          .with_rgb(&mut via_walker)
          .unwrap();
        <V210<true> as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();

        let mut sd = MixedSinker::<V210<true>>::new(W as usize, H as usize)
          .with_rgb(&mut via_direct)
          .unwrap();
        v210_to_endian::<_, true>(&src, full_range, matrix, &mut sd).unwrap();

        assert_eq!(
          via_walker, via_direct,
          "v210 BE parity (full_range={full_range}, matrix={matrix:?})"
        );
      }
    }
  }
}

#[cfg(feature = "yuva")]
mod yuva_parity {
  use super::*;
  use crate::sinker::MixedSinker;

  const MATRICES: [KernelMatrix; 2] = [KernelMatrix::Bt709, KernelMatrix::Bt601];

  fn ramp8(n: usize) -> std::vec::Vec<u8> {
    (0..n).map(|i| ((i * 31 + 9) % 251) as u8).collect()
  }
  fn ramp16(n: usize, bits: u32) -> std::vec::Vec<u16> {
    let max = (1u32 << bits) - 1;
    (0..n)
      .map(|i| (((i as u32) * 911 + 13) & max) as u16)
      .collect()
  }

  /// Drives a 4-plane planar YUVA family. The alpha plane is read inside
  /// `{fmt}_to`, never an `Options` knob — so a [`YuvOptions`] is all the
  /// walk needs, and byte-identity to the direct call proves the alpha
  /// path is forwarded too. `$try_new` is the 10-arg ctor
  /// (`y,u,v,a,w,h,ys,us,vs,as`).
  macro_rules! planar4_case {
    (
      $name:ident, $marker:ty, $try_new:path, $walker:path,
      ramp = $ramp:expr, cw_div = $cw_div:expr, ch_div = $ch_div:expr,
    ) => {
      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn $name() {
        const W: u32 = 16;
        const H: u32 = 4;
        let cw = (W as usize) / $cw_div;
        let ch = (H as usize).div_ceil($ch_div);
        let make = $ramp;
        let y = make((W * H) as usize);
        let u = make(cw * ch);
        let v = make(cw * ch);
        let a = make((W * H) as usize);

        for full_range in [false, true] {
          for matrix in MATRICES {
            let opts = YuvOptions::new().maybe_full_range(full_range).with_matrix(matrix);
            let src = $try_new(&y, &u, &v, &a, W, H, W, cw as u32, cw as u32, W).unwrap();

            let mut via_walker = std::vec![0u8; (W * H * 3) as usize];
            let mut via_direct = std::vec![0u8; (W * H * 3) as usize];

            let mut sw = MixedSinker::<$marker>::new(W as usize, H as usize)
              .with_rgb(&mut via_walker)
              .unwrap();
            <$marker as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();

            let mut sd = MixedSinker::<$marker>::new(W as usize, H as usize)
              .with_rgb(&mut via_direct)
              .unwrap();
            $walker(&src, full_range, matrix, &mut sd).unwrap();

            assert_eq!(
              via_walker, via_direct,
              "{} parity (full_range={full_range}, matrix={matrix:?})",
              stringify!($name)
            );
          }
        }
      }
    };
  }

  // 8-bit: Yuva420p (half/half chroma + full-res alpha).
  planar4_case!(
    walk_yuva420p_matches_direct,
    crate::source::Yuva420p,
    crate::frame::Yuva420pFrame::try_new,
    crate::source::yuva420p_to,
    ramp = ramp8,
    cw_div = 2,
    ch_div = 2,
  );
  // High-bit-LE: Yuva444p12 (full chroma + full-res alpha).
  planar4_case!(
    walk_yuva444p12_matches_direct,
    crate::source::Yuva444p12,
    crate::frame::Yuva444p12LeFrame::try_new,
    crate::source::yuva444p12_to,
    ramp = |n| ramp16(n, 12),
    cw_div = 1,
    ch_div = 1,
  );

  /// BE sibling of [`planar4_case`]: drives the `@const_bits` impl at
  /// `Marker<true>` against a `{Fmt}BeFrame`, compared to a direct
  /// `{fmt}_to_endian::<_, true>` call (alpha read inside the walker).
  macro_rules! planar4_be_case {
    (
      $name:ident, $marker:ty, $try_new:path, $walker_endian:path,
      ramp = $ramp:expr, cw_div = $cw_div:expr, ch_div = $ch_div:expr,
    ) => {
      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn $name() {
        const W: u32 = 16;
        const H: u32 = 4;
        let cw = (W as usize) / $cw_div;
        let ch = (H as usize).div_ceil($ch_div);
        let make = $ramp;
        let y = make((W * H) as usize);
        let u = make(cw * ch);
        let v = make(cw * ch);
        let a = make((W * H) as usize);

        for full_range in [false, true] {
          for matrix in MATRICES {
            let opts = YuvOptions::new().maybe_full_range(full_range).with_matrix(matrix);
            let src = $try_new(&y, &u, &v, &a, W, H, W, cw as u32, cw as u32, W).unwrap();

            let mut via_walker = std::vec![0u8; (W * H * 3) as usize];
            let mut via_direct = std::vec![0u8; (W * H * 3) as usize];

            let mut sw = MixedSinker::<$marker>::new(W as usize, H as usize)
              .with_rgb(&mut via_walker)
              .unwrap();
            <$marker as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();

            let mut sd = MixedSinker::<$marker>::new(W as usize, H as usize)
              .with_rgb(&mut via_direct)
              .unwrap();
            $walker_endian(&src, full_range, matrix, &mut sd).unwrap();

            assert_eq!(
              via_walker, via_direct,
              "{} BE parity (full_range={full_range}, matrix={matrix:?})",
              stringify!($name)
            );
          }
        }
      }
    };
  }

  // High-bit-BE: Yuva444p12 (full chroma + full-res alpha).
  planar4_be_case!(
    walk_yuva444p12_be_matches_direct,
    crate::source::Yuva444p12<true>,
    crate::frame::Yuva444p12BeFrame::try_new,
    crate::source::yuva444p12_to_endian::<_, true>,
    ramp = |n| ramp16(n, 12),
    cw_div = 1,
    ch_div = 1,
  );
}

// ---- Parity: packed RGB families (already-RGB; reuse YuvOptions) ------
//
// RGB sources carry no chroma matrix, but the free `{fmt}_to` /
// `{fmt}_to_endian` walkers still take `(full_range, matrix)` (the
// RGB-input row threads them to the `with_luma` / `with_hsv` outputs), so
// the [`Walker`] impl forwards `YuvOptions`. Each test asserts
// `<Marker as Walker<_>>::walk` is byte-identical to a direct walker
// call into the same `MixedSinker::with_rgb` sink, across full/limited ×
// Bt709/Bt601 (a mis-forwarded `full_range`/`matrix` would only show in
// the luma/hsv path, not `with_rgb` — but forwarding both proves the
// signature is wired). Coverage spans every topology: an 8-bit packed
// (Rgb24, plain arm), a 16-bit (Rgb48, the `@const BE` arm — BOTH LE and
// BE, since the impl delegates to the const-generic `{fmt}_to_endian`),
// and a legacy 5/6/5 (Rgb565, byte-order-fixed LE plain arm).

#[cfg(feature = "rgb")]
mod rgb_parity {
  use super::*;
  use crate::{
    frame::{Rgb24Frame, Rgb48BeFrame, Rgb48Frame, Rgb96BeFrame, Rgb96Frame},
    sinker::MixedSinker,
    source::{Rgb24, Rgb48, Rgb96, rgb24_to, rgb48_to, rgb48_to_endian, rgb96_to, rgb96_to_endian},
  };

  const W: u32 = 8;
  const H: u32 = 4;
  const MATRICES: [KernelMatrix; 2] = [KernelMatrix::Bt709, KernelMatrix::Bt601];

  /// A deterministic, column/row-varying `u8` plane of `n` samples.
  fn ramp8(n: usize) -> std::vec::Vec<u8> {
    (0..n).map(|i| ((i * 17 + 3) % 251) as u8).collect()
  }

  /// A deterministic `u16` plane of `n` samples (full 16-bit range).
  fn ramp16(n: usize) -> std::vec::Vec<u16> {
    (0..n).map(|i| ((i * 1103 + 7) & 0xFFFF) as u16).collect()
  }

  /// A deterministic `u32` plane of `n` samples (full 32-bit range).
  fn ramp32(n: usize) -> std::vec::Vec<u32> {
    (0..n)
      .map(|i| (i as u32).wrapping_mul(0x0193_3C7F).wrapping_add(0x1357))
      .collect()
  }

  /// 8-bit packed Rgb24 — plain arm, `width * 3` u8 per row.
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn walk_rgb24_matches_direct() {
    let buf = ramp8((W * H * 3) as usize);
    for full_range in [false, true] {
      for matrix in MATRICES {
        let opts = YuvOptions::new()
          .maybe_full_range(full_range)
          .with_matrix(matrix);
        let src = Rgb24Frame::try_new(&buf, W, H, W * 3).unwrap();

        let mut via_walker = std::vec![0u8; (W * H * 3) as usize];
        let mut via_direct = std::vec![0u8; (W * H * 3) as usize];

        let mut sw = MixedSinker::<Rgb24>::new(W as usize, H as usize)
          .with_rgb(&mut via_walker)
          .unwrap();
        <Rgb24 as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();

        let mut sd = MixedSinker::<Rgb24>::new(W as usize, H as usize)
          .with_rgb(&mut via_direct)
          .unwrap();
        rgb24_to(&src, full_range, matrix, &mut sd).unwrap();

        assert_eq!(
          via_walker, via_direct,
          "rgb24 parity (full_range={full_range}, matrix={matrix:?})"
        );
      }
    }
  }

  /// 16-bit packed Rgb48 LE — `@const BE` arm at the marker's `<const BE
  /// = false>` default against the LE `rgb48_to` wrapper. `width * 3`
  /// u16 per row.
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn walk_rgb48_le_matches_direct() {
    let buf = ramp16((W * H * 3) as usize);
    for full_range in [false, true] {
      for matrix in MATRICES {
        let opts = YuvOptions::new()
          .maybe_full_range(full_range)
          .with_matrix(matrix);
        let src = Rgb48Frame::try_new(&buf, W, H, W * 3).unwrap();

        let mut via_walker = std::vec![0u8; (W * H * 3) as usize];
        let mut via_direct = std::vec![0u8; (W * H * 3) as usize];

        let mut sw = MixedSinker::<Rgb48>::new(W as usize, H as usize)
          .with_rgb(&mut via_walker)
          .unwrap();
        <Rgb48 as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();

        let mut sd = MixedSinker::<Rgb48>::new(W as usize, H as usize)
          .with_rgb(&mut via_direct)
          .unwrap();
        rgb48_to(&src, full_range, matrix, &mut sd).unwrap();

        assert_eq!(
          via_walker, via_direct,
          "rgb48 LE parity (full_range={full_range}, matrix={matrix:?})"
        );
      }
    }
  }

  /// 16-bit packed Rgb48 BE — drives the `@const BE` impl at `Rgb48<true>`
  /// against the `Rgb48BeFrame` alias, compared to a direct
  /// `rgb48_to_endian::<_, true>` call. Proves the BE half of the
  /// endian-generic impl.
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn walk_rgb48_be_matches_direct() {
    let buf = ramp16((W * H * 3) as usize);
    for full_range in [false, true] {
      for matrix in MATRICES {
        let opts = YuvOptions::new()
          .maybe_full_range(full_range)
          .with_matrix(matrix);
        let src = Rgb48BeFrame::try_new(&buf, W, H, W * 3).unwrap();

        let mut via_walker = std::vec![0u8; (W * H * 3) as usize];
        let mut via_direct = std::vec![0u8; (W * H * 3) as usize];

        let mut sw = MixedSinker::<Rgb48<true>>::new(W as usize, H as usize)
          .with_rgb(&mut via_walker)
          .unwrap();
        <Rgb48<true> as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();

        let mut sd = MixedSinker::<Rgb48<true>>::new(W as usize, H as usize)
          .with_rgb(&mut via_direct)
          .unwrap();
        rgb48_to_endian::<_, true>(&src, full_range, matrix, &mut sd).unwrap();

        assert_eq!(
          via_walker, via_direct,
          "rgb48 BE parity (full_range={full_range}, matrix={matrix:?})"
        );
      }
    }
  }

  /// 32-bit packed Rgb96 LE — `@const BE` arm at the `<const BE = false>`
  /// default against the LE `rgb96_to` wrapper. `width * 3` u32 per row.
  /// Uses a `with_luma` sink so a dropped `(full_range, matrix)` forward
  /// would diverge (luma is matrix-weighted + full-range-scaled).
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn walk_rgb96_le_matches_direct() {
    let buf = ramp32((W * H * 3) as usize);
    for full_range in [false, true] {
      for matrix in MATRICES {
        let opts = YuvOptions::new()
          .maybe_full_range(full_range)
          .with_matrix(matrix);
        let src = Rgb96Frame::try_new(&buf, W, H, W * 3).unwrap();

        let mut via_walker = std::vec![0u8; (W * H) as usize];
        let mut via_direct = std::vec![0u8; (W * H) as usize];

        let mut sw = MixedSinker::<Rgb96>::new(W as usize, H as usize)
          .with_luma(&mut via_walker)
          .unwrap();
        <Rgb96 as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();

        let mut sd = MixedSinker::<Rgb96>::new(W as usize, H as usize)
          .with_luma(&mut via_direct)
          .unwrap();
        rgb96_to(&src, full_range, matrix, &mut sd).unwrap();

        assert_eq!(
          via_walker, via_direct,
          "rgb96 LE parity (full_range={full_range}, matrix={matrix:?})"
        );
      }
    }
  }

  /// 32-bit packed Rgb96 BE — drives the `@const BE` impl at `Rgb96<true>`
  /// against the `Rgb96BeFrame` alias, compared to a direct
  /// `rgb96_to_endian::<_, true>` call.
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn walk_rgb96_be_matches_direct() {
    let buf = ramp32((W * H * 3) as usize);
    for full_range in [false, true] {
      for matrix in MATRICES {
        let opts = YuvOptions::new()
          .maybe_full_range(full_range)
          .with_matrix(matrix);
        let src = Rgb96BeFrame::try_new(&buf, W, H, W * 3).unwrap();

        let mut via_walker = std::vec![0u8; (W * H) as usize];
        let mut via_direct = std::vec![0u8; (W * H) as usize];

        let mut sw = MixedSinker::<Rgb96<true>>::new(W as usize, H as usize)
          .with_luma(&mut via_walker)
          .unwrap();
        <Rgb96<true> as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();

        let mut sd = MixedSinker::<Rgb96<true>>::new(W as usize, H as usize)
          .with_luma(&mut via_direct)
          .unwrap();
        rgb96_to_endian::<_, true>(&src, full_range, matrix, &mut sd).unwrap();

        assert_eq!(
          via_walker, via_direct,
          "rgb96 BE parity (full_range={full_range}, matrix={matrix:?})"
        );
      }
    }
  }
}

// ---- Parity: packed 10-bit 2-10-10-10 RGB (X2Rgb10 / X2Bgr10) ---------
//
// These are already-RGB sources, so `X2*10 → RGB` is an option-ignoring
// permute: a dropped or swapped `(full_range, matrix)` forward would not
// show in a `with_rgb` output. `X2*10 → luma` is matrix-weighted +
// full_range-scaled, so luma parity *does* catch it (exactly as the GBR
// families are tested). Each test asserts `<Marker as Walker<_>>::walk` is
// byte-identical to a direct walker call into the same
// `MixedSinker::with_luma` sink, across full/limited × Bt709/Bt601. Both
// families are endian-generic (`@const BE` arm): the LE case drives the
// impl at the marker's `<const BE = false>` default against the
// `{Fmt}LeFrame` + LE `{fmt}_to` wrapper, and the BE case drives
// `Marker<true>` against `{Fmt}BeFrame` + a direct
// `{fmt}_to_endian::<_, true>` call, so both halves of the impl are proven.

#[cfg(feature = "rgb")]
mod x2_packed_rgb_parity {
  use super::*;
  use crate::sinker::MixedSinker;

  const W: u32 = 8;
  const H: u32 = 4;
  const MATRICES: [KernelMatrix; 2] = [KernelMatrix::Bt709, KernelMatrix::Bt601];

  /// A deterministic, column/row-varying `u8` X2*10 buffer — `width * 4`
  /// bytes per row (one u32 word per pixel).
  fn ramp(n: usize) -> std::vec::Vec<u8> {
    (0..n).map(|i| ((i * 19 + 7) % 251) as u8).collect()
  }

  /// Drives one endianness of an X2*10 family against a luma sink.
  /// `$try_new` is the 4-arg ctor (`packed, w, h, stride`); `$walker` the
  /// direct walker fn (the LE wrapper or `{fmt}_to_endian::<_, true>`).
  macro_rules! x2_luma_case {
    ($name:ident, $marker:ty, $try_new:path, $walker:path, $label:literal) => {
      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn $name() {
        let buf = ramp((W * 4 * H) as usize);
        for full_range in [false, true] {
          for matrix in MATRICES {
            let opts = YuvOptions::new()
              .maybe_full_range(full_range)
              .with_matrix(matrix);
            let src = $try_new(&buf, W, H, W * 4).unwrap();

            let mut via_walker = std::vec![0u8; (W * H) as usize];
            let mut via_direct = std::vec![0u8; (W * H) as usize];

            let mut sw = MixedSinker::<$marker>::new(W as usize, H as usize)
              .with_luma(&mut via_walker)
              .unwrap();
            <$marker as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();

            let mut sd = MixedSinker::<$marker>::new(W as usize, H as usize)
              .with_luma(&mut via_direct)
              .unwrap();
            $walker(&src, full_range, matrix, &mut sd).unwrap();

            assert_eq!(
              via_walker, via_direct,
              concat!($label, " luma parity (full_range={}, matrix={:?})"),
              full_range, matrix
            );
          }
        }
      }
    };
  }

  x2_luma_case!(
    walk_x2rgb10_le_matches_direct,
    crate::source::X2Rgb10,
    crate::frame::X2Rgb10LeFrame::try_new,
    crate::source::x2rgb10_to,
    "x2rgb10 LE"
  );
  x2_luma_case!(
    walk_x2rgb10_be_matches_direct,
    crate::source::X2Rgb10<true>,
    crate::frame::X2Rgb10BeFrame::try_new,
    crate::source::x2rgb10_to_endian::<_, true>,
    "x2rgb10 BE"
  );
  x2_luma_case!(
    walk_x2bgr10_le_matches_direct,
    crate::source::X2Bgr10,
    crate::frame::X2Bgr10LeFrame::try_new,
    crate::source::x2bgr10_to,
    "x2bgr10 LE"
  );
  x2_luma_case!(
    walk_x2bgr10_be_matches_direct,
    crate::source::X2Bgr10<true>,
    crate::frame::X2Bgr10BeFrame::try_new,
    crate::source::x2bgr10_to_endian::<_, true>,
    "x2bgr10 BE"
  );
}

#[cfg(feature = "rgb-legacy")]
mod rgb_legacy_parity {
  use super::*;
  use crate::{
    frame::{
      Bgr4ByteFrame, Bgr4Frame, Bgr8Frame, Rgb4ByteFrame, Rgb4Frame, Rgb8Frame, Rgb565Frame,
    },
    sinker::MixedSinker,
    source::{
      Bgr4, Bgr4Byte, Bgr8, Rgb4, Rgb4Byte, Rgb8, Rgb565, bgr4_byte_to, bgr4_to, bgr8_to,
      rgb4_byte_to, rgb4_to, rgb8_to, rgb565_to,
    },
  };

  const W: u32 = 8;
  const H: u32 = 4;
  const MATRICES: [KernelMatrix; 2] = [KernelMatrix::Bt709, KernelMatrix::Bt601];

  /// Legacy 5/6/5 Rgb565 — byte-order-fixed LE, plain arm, `width * 2`
  /// bytes (`width` LE `u16` pixels) per row.
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn walk_rgb565_matches_direct() {
    let buf: std::vec::Vec<u8> = (0..(W * H * 2) as usize)
      .map(|i| ((i * 19 + 7) % 251) as u8)
      .collect();
    for full_range in [false, true] {
      for matrix in MATRICES {
        let opts = YuvOptions::new()
          .maybe_full_range(full_range)
          .with_matrix(matrix);
        let src = Rgb565Frame::try_new(&buf, W, H, W * 2).unwrap();

        let mut via_walker = std::vec![0u8; (W * H * 3) as usize];
        let mut via_direct = std::vec![0u8; (W * H * 3) as usize];

        let mut sw = MixedSinker::<Rgb565>::new(W as usize, H as usize)
          .with_rgb(&mut via_walker)
          .unwrap();
        <Rgb565 as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();

        let mut sd = MixedSinker::<Rgb565>::new(W as usize, H as usize)
          .with_rgb(&mut via_direct)
          .unwrap();
        rgb565_to(&src, full_range, matrix, &mut sd).unwrap();

        assert_eq!(
          via_walker, via_direct,
          "rgb565 parity (full_range={full_range}, matrix={matrix:?})"
        );
      }
    }
  }

  /// Legacy bit-packed RGB/BGR (8bpp `Rgb8`/`Bgr8`/`Rgb4Byte`/`Bgr4Byte`,
  /// 1 byte/pixel; 4bpp `Rgb4`/`Bgr4`, two pixels/byte): `Walker::walk` is
  /// byte-identical to the direct free `{fmt}_to` call across
  /// full/limited × Bt709/Bt601.
  #[test]
  fn walk_legacy_bit_packed_matches_direct() {
    macro_rules! check {
      ($Frame:ident, $Marker:ident, $free:path, $row_bytes:expr) => {{
        let rb: usize = $row_bytes;
        let buf: std::vec::Vec<u8> =
          (0..rb * H as usize).map(|i| ((i * 19 + 7) % 251) as u8).collect();
        for full_range in [false, true] {
          for matrix in MATRICES {
            let opts = YuvOptions::new()
              .maybe_full_range(full_range)
              .with_matrix(matrix);
            let src = $Frame::try_new(&buf, W, H, rb as u32).unwrap();

            let mut via_walker = std::vec![0u8; (W * H * 3) as usize];
            let mut via_direct = std::vec![0u8; (W * H * 3) as usize];

            let mut sw = MixedSinker::<$Marker>::new(W as usize, H as usize)
              .with_rgb(&mut via_walker)
              .unwrap();
            <$Marker as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();

            let mut sd = MixedSinker::<$Marker>::new(W as usize, H as usize)
              .with_rgb(&mut via_direct)
              .unwrap();
            $free(&src, full_range, matrix, &mut sd).unwrap();

            assert_eq!(
              via_walker,
              via_direct,
              "{} parity (full_range={full_range}, matrix={matrix:?})",
              stringify!($Marker)
            );
          }
        }
      }};
    }

    check!(Rgb8Frame, Rgb8, rgb8_to, W as usize);
    check!(Bgr8Frame, Bgr8, bgr8_to, W as usize);
    check!(Rgb4ByteFrame, Rgb4Byte, rgb4_byte_to, W as usize);
    check!(Bgr4ByteFrame, Bgr4Byte, bgr4_byte_to, W as usize);
    check!(Rgb4Frame, Rgb4, rgb4_to, (W as usize).div_ceil(2));
    check!(Bgr4Frame, Bgr4, bgr4_to, (W as usize).div_ceil(2));
  }
}

// ---- Parity: gray families (single-luma / luma+alpha; reuse YuvOptions) ---
//
// Gray sources carry no chroma matrix, but the free `{fmt}_to` /
// `{fmt}_to_endian` walkers take `(full_range, matrix)`: the RGB output
// rescales limited-range luma (so `full_range` is genuinely exercised by
// `with_rgb`), while `matrix` is carried through but unused by the
// chroma-free gray kernels. Each test asserts `<Marker as Walker<_>>::walk`
// is byte-identical to a direct walker call into the same
// `MixedSinker::with_rgb` sink, across full/limited × Bt709/Bt601.
// Coverage spans every topology: a single-luma 8-bit (Gray8, plain arm), a
// luma+alpha 8-bit (Ya8, plain arm), a high-bit GrayN (Gray10, the
// `@const_bits` arm — BOTH LE and BE), and a 16-bit (Gray16, the
// `@const BE` arm — BOTH LE and BE). The high-bit / 16-bit impls delegate
// to the const-generic `{fmt}_to_endian::<_, BE>` (the LE `{fmt}_to` is its
// `BE = false` shim), so both halves of the endian-generic impl are proven.

#[cfg(feature = "gray")]
mod gray_parity {
  use super::*;
  use crate::sinker::MixedSinker;

  const W: u32 = 16;
  const H: u32 = 4;
  const MATRICES: [KernelMatrix; 2] = [KernelMatrix::Bt709, KernelMatrix::Bt601];

  /// A deterministic, column/row-varying `u8` plane of `n` samples.
  fn ramp8(n: usize) -> std::vec::Vec<u8> {
    (0..n).map(|i| ((i * 17 + 3) % 251) as u8).collect()
  }

  /// A deterministic low-packed `u16` plane of `n` samples bounded to
  /// `bits` (active bits in the low end, matching the LE wire layout on the
  /// test host).
  fn ramp16(n: usize, bits: u32) -> std::vec::Vec<u16> {
    let max = (1u32 << bits) - 1;
    (0..n)
      .map(|i| (((i as u32) * 1103 + 7) & max) as u16)
      .collect()
  }

  /// A deterministic, full-range `u32` plane of `n` samples.
  fn ramp32(n: usize) -> std::vec::Vec<u32> {
    (0..n)
      .map(|i| (i as u32).wrapping_mul(2_654_435_761) ^ 0x55AA_33CC)
      .collect()
  }

  /// Gray8 — single `u8` luma plane, `width` u8 per row. Plain arm.
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn walk_gray8_matches_direct() {
    use crate::{
      frame::Gray8Frame,
      source::{Gray8, gray8_to},
    };
    let y = ramp8((W * H) as usize);
    for full_range in [false, true] {
      for matrix in MATRICES {
        let opts = YuvOptions::new()
          .maybe_full_range(full_range)
          .with_matrix(matrix);
        let src = Gray8Frame::try_new(&y, W, H, W).unwrap();

        let mut via_walker = std::vec![0u8; (W * H * 3) as usize];
        let mut via_direct = std::vec![0u8; (W * H * 3) as usize];

        let mut sw = MixedSinker::<Gray8>::new(W as usize, H as usize)
          .with_rgb(&mut via_walker)
          .unwrap();
        <Gray8 as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();

        let mut sd = MixedSinker::<Gray8>::new(W as usize, H as usize)
          .with_rgb(&mut via_direct)
          .unwrap();
        gray8_to(&src, full_range, matrix, &mut sd).unwrap();

        assert_eq!(
          via_walker, via_direct,
          "gray8 parity (full_range={full_range}, matrix={matrix:?})"
        );
      }
    }
  }

  /// Ya8 — packed `[Y, A]` u8, `width × 2` u8 per row. Plain arm.
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn walk_ya8_matches_direct() {
    use crate::{
      frame::Ya8Frame,
      source::{Ya8, ya8_to},
    };
    let packed = ramp8((W * H * 2) as usize);
    for full_range in [false, true] {
      for matrix in MATRICES {
        let opts = YuvOptions::new()
          .maybe_full_range(full_range)
          .with_matrix(matrix);
        let src = Ya8Frame::try_new(&packed, W, H, W * 2).unwrap();

        let mut via_walker = std::vec![0u8; (W * H * 3) as usize];
        let mut via_direct = std::vec![0u8; (W * H * 3) as usize];

        let mut sw = MixedSinker::<Ya8>::new(W as usize, H as usize)
          .with_rgb(&mut via_walker)
          .unwrap();
        <Ya8 as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();

        let mut sd = MixedSinker::<Ya8>::new(W as usize, H as usize)
          .with_rgb(&mut via_direct)
          .unwrap();
        ya8_to(&src, full_range, matrix, &mut sd).unwrap();

        assert_eq!(
          via_walker, via_direct,
          "ya8 parity (full_range={full_range}, matrix={matrix:?})"
        );
      }
    }
  }

  /// Gray10 LE — high-bit `GrayN` (10-bit low-packed u16). Drives the
  /// `@const_bits` impl at the marker's `<const BE = false>` default
  /// against the LE `gray10_to` wrapper.
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn walk_gray10_le_matches_direct() {
    use crate::{
      frame::Gray10LeFrame,
      source::{Gray10, gray10_to},
    };
    let y = ramp16((W * H) as usize, 10);
    for full_range in [false, true] {
      for matrix in MATRICES {
        let opts = YuvOptions::new()
          .maybe_full_range(full_range)
          .with_matrix(matrix);
        let src = Gray10LeFrame::try_new(&y, W, H, W).unwrap();

        let mut via_walker = std::vec![0u8; (W * H * 3) as usize];
        let mut via_direct = std::vec![0u8; (W * H * 3) as usize];

        let mut sw = MixedSinker::<Gray10>::new(W as usize, H as usize)
          .with_rgb(&mut via_walker)
          .unwrap();
        <Gray10 as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();

        let mut sd = MixedSinker::<Gray10>::new(W as usize, H as usize)
          .with_rgb(&mut via_direct)
          .unwrap();
        gray10_to(&src, full_range, matrix, &mut sd).unwrap();

        assert_eq!(
          via_walker, via_direct,
          "gray10 LE parity (full_range={full_range}, matrix={matrix:?})"
        );
      }
    }
  }

  /// Gray10 BE — drives the `@const_bits` impl at `Gray10<true>` against
  /// the `Gray10BeFrame` alias, compared to a direct
  /// `gray10_to_endian::<_, true>`.
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn walk_gray10_be_matches_direct() {
    use crate::{
      frame::Gray10BeFrame,
      source::{Gray10, gray10_to_endian},
    };
    let y = ramp16((W * H) as usize, 10);
    for full_range in [false, true] {
      for matrix in MATRICES {
        let opts = YuvOptions::new()
          .maybe_full_range(full_range)
          .with_matrix(matrix);
        let src = Gray10BeFrame::try_new(&y, W, H, W).unwrap();

        let mut via_walker = std::vec![0u8; (W * H * 3) as usize];
        let mut via_direct = std::vec![0u8; (W * H * 3) as usize];

        let mut sw = MixedSinker::<Gray10<true>>::new(W as usize, H as usize)
          .with_rgb(&mut via_walker)
          .unwrap();
        <Gray10<true> as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();

        let mut sd = MixedSinker::<Gray10<true>>::new(W as usize, H as usize)
          .with_rgb(&mut via_direct)
          .unwrap();
        gray10_to_endian::<_, true>(&src, full_range, matrix, &mut sd).unwrap();

        assert_eq!(
          via_walker, via_direct,
          "gray10 BE parity (full_range={full_range}, matrix={matrix:?})"
        );
      }
    }
  }

  /// Gray16 LE — `@const BE` arm at the marker's `<const BE = false>`
  /// default against the LE `gray16_to` wrapper.
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn walk_gray16_le_matches_direct() {
    use crate::{
      frame::Gray16LeFrame,
      source::{Gray16, gray16_to},
    };
    let y = ramp16((W * H) as usize, 16);
    for full_range in [false, true] {
      for matrix in MATRICES {
        let opts = YuvOptions::new()
          .maybe_full_range(full_range)
          .with_matrix(matrix);
        let src = Gray16LeFrame::try_new(&y, W, H, W).unwrap();

        let mut via_walker = std::vec![0u8; (W * H * 3) as usize];
        let mut via_direct = std::vec![0u8; (W * H * 3) as usize];

        let mut sw = MixedSinker::<Gray16>::new(W as usize, H as usize)
          .with_rgb(&mut via_walker)
          .unwrap();
        <Gray16 as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();

        let mut sd = MixedSinker::<Gray16>::new(W as usize, H as usize)
          .with_rgb(&mut via_direct)
          .unwrap();
        gray16_to(&src, full_range, matrix, &mut sd).unwrap();

        assert_eq!(
          via_walker, via_direct,
          "gray16 LE parity (full_range={full_range}, matrix={matrix:?})"
        );
      }
    }
  }

  /// Gray16 BE — drives the `@const BE` impl at `Gray16<true>` against the
  /// `Gray16BeFrame` alias, compared to a direct `gray16_to_endian::<_, true>`.
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn walk_gray16_be_matches_direct() {
    use crate::{
      frame::Gray16BeFrame,
      source::{Gray16, gray16_to_endian},
    };
    let y = ramp16((W * H) as usize, 16);
    for full_range in [false, true] {
      for matrix in MATRICES {
        let opts = YuvOptions::new()
          .maybe_full_range(full_range)
          .with_matrix(matrix);
        let src = Gray16BeFrame::try_new(&y, W, H, W).unwrap();

        let mut via_walker = std::vec![0u8; (W * H * 3) as usize];
        let mut via_direct = std::vec![0u8; (W * H * 3) as usize];

        let mut sw = MixedSinker::<Gray16<true>>::new(W as usize, H as usize)
          .with_rgb(&mut via_walker)
          .unwrap();
        <Gray16<true> as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();

        let mut sd = MixedSinker::<Gray16<true>>::new(W as usize, H as usize)
          .with_rgb(&mut via_direct)
          .unwrap();
        gray16_to_endian::<_, true>(&src, full_range, matrix, &mut sd).unwrap();

        assert_eq!(
          via_walker, via_direct,
          "gray16 BE parity (full_range={full_range}, matrix={matrix:?})"
        );
      }
    }
  }

  /// Gray32 LE — the full-bit integer twin of Gray16 on the `@const BE` arm at
  /// the `<const BE = false>` default against the LE `gray32_to` wrapper.
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn walk_gray32_le_matches_direct() {
    use crate::{
      frame::Gray32LeFrame,
      source::{Gray32, gray32_to},
    };
    let y = ramp32((W * H) as usize);
    for full_range in [false, true] {
      for matrix in MATRICES {
        let opts = YuvOptions::new()
          .maybe_full_range(full_range)
          .with_matrix(matrix);
        let src = Gray32LeFrame::try_new(&y, W, H, W).unwrap();

        let mut via_walker = std::vec![0u8; (W * H * 3) as usize];
        let mut via_direct = std::vec![0u8; (W * H * 3) as usize];

        let mut sw = MixedSinker::<Gray32>::new(W as usize, H as usize)
          .with_rgb(&mut via_walker)
          .unwrap();
        <Gray32 as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();

        let mut sd = MixedSinker::<Gray32>::new(W as usize, H as usize)
          .with_rgb(&mut via_direct)
          .unwrap();
        gray32_to(&src, full_range, matrix, &mut sd).unwrap();

        assert_eq!(
          via_walker, via_direct,
          "gray32 LE parity (full_range={full_range}, matrix={matrix:?})"
        );
      }
    }
  }

  /// Gray32 BE — drives the `@const BE` impl at `Gray32<true>` against the
  /// `Gray32BeFrame` alias, compared to a direct `gray32_to_endian::<_, true>`.
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn walk_gray32_be_matches_direct() {
    use crate::{
      frame::Gray32BeFrame,
      source::{Gray32, gray32_to_endian},
    };
    let y = ramp32((W * H) as usize);
    for full_range in [false, true] {
      for matrix in MATRICES {
        let opts = YuvOptions::new()
          .maybe_full_range(full_range)
          .with_matrix(matrix);
        let src = Gray32BeFrame::try_new(&y, W, H, W).unwrap();

        let mut via_walker = std::vec![0u8; (W * H * 3) as usize];
        let mut via_direct = std::vec![0u8; (W * H * 3) as usize];

        let mut sw = MixedSinker::<Gray32<true>>::new(W as usize, H as usize)
          .with_rgb(&mut via_walker)
          .unwrap();
        <Gray32<true> as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();

        let mut sd = MixedSinker::<Gray32<true>>::new(W as usize, H as usize)
          .with_rgb(&mut via_direct)
          .unwrap();
        gray32_to_endian::<_, true>(&src, full_range, matrix, &mut sd).unwrap();

        assert_eq!(
          via_walker, via_direct,
          "gray32 BE parity (full_range={full_range}, matrix={matrix:?})"
        );
      }
    }
  }
}

// ---- Parity: planar GBR families (already-RGB; reuse YuvOptions) -------
//
// GBR sources carry no chroma matrix, but the free `{fmt}_to` /
// `{fmt}_to_endian` walkers still take `(full_range, matrix)` (the
// RGB-input row threads them to the `with_luma` / `with_hsv` outputs), so
// the [`Walker`] impl forwards `YuvOptions`. Each test asserts
// `<Marker as Walker<_>>::walk` is byte-identical to a direct walker call
// into the same `MixedSinker::with_rgb` sink, across full/limited ×
// Bt709/Bt601. Coverage spans both topologies: an 8-bit planar (Gbrp,
// plain arm) and a high-bit (Gbrp10, the `@const_bits` arm — BOTH LE and
// BE, since the impl delegates to the const-generic `{fmt}_to_endian`).

#[cfg(feature = "gbr")]
mod gbr_parity {
  use super::*;
  use crate::sinker::MixedSinker;

  const W: u32 = 16;
  const H: u32 = 4;
  const MATRICES: [KernelMatrix; 2] = [KernelMatrix::Bt709, KernelMatrix::Bt601];

  /// Three deterministic, distinct `u8` G/B/R planes of `W × H` samples.
  fn planes8() -> (std::vec::Vec<u8>, std::vec::Vec<u8>, std::vec::Vec<u8>) {
    let n = (W * H) as usize;
    let g = (0..n).map(|i| ((i * 17 + 3) % 251) as u8).collect();
    let b = (0..n).map(|i| ((i * 23 + 5) % 251) as u8).collect();
    let r = (0..n).map(|i| ((i * 31 + 9) % 251) as u8).collect();
    (g, b, r)
  }

  /// Three deterministic low-packed `u16` G/B/R planes bounded to `bits`.
  fn planes16(bits: u32) -> (std::vec::Vec<u16>, std::vec::Vec<u16>, std::vec::Vec<u16>) {
    let n = (W * H) as usize;
    let max = (1u32 << bits) - 1;
    let g = (0..n)
      .map(|i| (((i as u32) * 1103 + 7) & max) as u16)
      .collect();
    let b = (0..n)
      .map(|i| (((i as u32) * 1399 + 11) & max) as u16)
      .collect();
    let r = (0..n)
      .map(|i| (((i as u32) * 911 + 13) & max) as u16)
      .collect();
    (g, b, r)
  }

  /// Gbrp — three full-width `u8` G/B/R planes. Plain arm.
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn walk_gbrp_matches_direct() {
    use crate::{
      frame::GbrpFrame,
      source::{Gbrp, gbrp_to},
    };
    let (g, b, r) = planes8();
    for full_range in [false, true] {
      for matrix in MATRICES {
        let opts = YuvOptions::new()
          .maybe_full_range(full_range)
          .with_matrix(matrix);
        let src = GbrpFrame::try_new(&g, &b, &r, W, H, W, W, W).unwrap();

        let mut via_walker = std::vec![0u8; (W * H * 3) as usize];
        let mut via_direct = std::vec![0u8; (W * H * 3) as usize];

        let mut sw = MixedSinker::<Gbrp>::new(W as usize, H as usize)
          .with_rgb(&mut via_walker)
          .unwrap();
        <Gbrp as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();

        let mut sd = MixedSinker::<Gbrp>::new(W as usize, H as usize)
          .with_rgb(&mut via_direct)
          .unwrap();
        gbrp_to(&src, full_range, matrix, &mut sd).unwrap();

        assert_eq!(
          via_walker, via_direct,
          "gbrp parity (full_range={full_range}, matrix={matrix:?})"
        );
      }
    }
  }

  /// Gbrp10 LE — high-bit planar GBR (10-bit low-packed u16). Drives the
  /// `@const_bits` impl at the marker's `<const BE = false>` default
  /// against the LE `gbrp10_to` wrapper.
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn walk_gbrp10_le_matches_direct() {
    use crate::{
      frame::Gbrp10LeFrame,
      source::{Gbrp10, gbrp10_to},
    };
    let (g, b, r) = planes16(10);
    for full_range in [false, true] {
      for matrix in MATRICES {
        let opts = YuvOptions::new()
          .maybe_full_range(full_range)
          .with_matrix(matrix);
        let src = Gbrp10LeFrame::try_new(&g, &b, &r, W, H, W, W, W).unwrap();

        let mut via_walker = std::vec![0u8; (W * H * 3) as usize];
        let mut via_direct = std::vec![0u8; (W * H * 3) as usize];

        let mut sw = MixedSinker::<Gbrp10>::new(W as usize, H as usize)
          .with_rgb(&mut via_walker)
          .unwrap();
        <Gbrp10 as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();

        let mut sd = MixedSinker::<Gbrp10>::new(W as usize, H as usize)
          .with_rgb(&mut via_direct)
          .unwrap();
        gbrp10_to(&src, full_range, matrix, &mut sd).unwrap();

        assert_eq!(
          via_walker, via_direct,
          "gbrp10 LE parity (full_range={full_range}, matrix={matrix:?})"
        );
      }
    }
  }

  /// Gbrp10 BE — drives the `@const_bits` impl at `Gbrp10<true>` against
  /// the `Gbrp10BeFrame` alias, compared to a direct
  /// `gbrp10_to_endian::<_, true>`.
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn walk_gbrp10_be_matches_direct() {
    use crate::{
      frame::Gbrp10BeFrame,
      source::{Gbrp10, gbrp10_to_endian},
    };
    let (g, b, r) = planes16(10);
    for full_range in [false, true] {
      for matrix in MATRICES {
        let opts = YuvOptions::new()
          .maybe_full_range(full_range)
          .with_matrix(matrix);
        let src = Gbrp10BeFrame::try_new(&g, &b, &r, W, H, W, W, W).unwrap();

        let mut via_walker = std::vec![0u8; (W * H * 3) as usize];
        let mut via_direct = std::vec![0u8; (W * H * 3) as usize];

        let mut sw = MixedSinker::<Gbrp10<true>>::new(W as usize, H as usize)
          .with_rgb(&mut via_walker)
          .unwrap();
        <Gbrp10<true> as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();

        let mut sd = MixedSinker::<Gbrp10<true>>::new(W as usize, H as usize)
          .with_rgb(&mut via_direct)
          .unwrap();
        gbrp10_to_endian::<_, true>(&src, full_range, matrix, &mut sd).unwrap();

        assert_eq!(
          via_walker, via_direct,
          "gbrp10 BE parity (full_range={full_range}, matrix={matrix:?})"
        );
      }
    }
  }

  /// Four deterministic distinct `u32` G/B/R/A planes of `W × H` samples
  /// (full 32-bit range, nonzero low bits), including a real alpha plane.
  #[allow(clippy::type_complexity)]
  fn planes32() -> (
    std::vec::Vec<u32>,
    std::vec::Vec<u32>,
    std::vec::Vec<u32>,
    std::vec::Vec<u32>,
  ) {
    let n = (W * H) as usize;
    let g = (0..n)
      .map(|i| (i as u32).wrapping_mul(0x0123_4567).wrapping_add(0x11))
      .collect();
    let b = (0..n)
      .map(|i| (i as u32).wrapping_mul(0x0246_8ACE).wrapping_add(0x22))
      .collect();
    let r = (0..n)
      .map(|i| (i as u32).wrapping_mul(0x0FED_CBA9).wrapping_add(0x33))
      .collect();
    let a = (0..n)
      .map(|i| (i as u32).wrapping_mul(0x0777_1333).wrapping_add(0x44))
      .collect();
    (g, b, r, a)
  }

  /// Gbrap32 LE — the full-bit `u32` twin of `Gbrap16` on the `@const BE` arm
  /// at the `<const BE = false>` default against the LE `gbrap32_to` wrapper.
  /// Drives `with_rgba` so the real alpha plane rides through the walker.
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn walk_gbrap32_le_matches_direct() {
    use crate::{
      frame::Gbrap32LeFrame,
      source::{Gbrap32, gbrap32_to},
    };
    let (g, b, r, a) = planes32();
    for full_range in [false, true] {
      for matrix in MATRICES {
        let opts = YuvOptions::new()
          .maybe_full_range(full_range)
          .with_matrix(matrix);
        let src = Gbrap32LeFrame::try_new(&g, &b, &r, &a, W, H, W, W, W, W).unwrap();

        let mut via_walker = std::vec![0u8; (W * H * 4) as usize];
        let mut via_direct = std::vec![0u8; (W * H * 4) as usize];

        let mut sw = MixedSinker::<Gbrap32>::new(W as usize, H as usize)
          .with_rgba(&mut via_walker)
          .unwrap();
        <Gbrap32 as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();

        let mut sd = MixedSinker::<Gbrap32>::new(W as usize, H as usize)
          .with_rgba(&mut via_direct)
          .unwrap();
        gbrap32_to(&src, full_range, matrix, &mut sd).unwrap();

        assert_eq!(
          via_walker, via_direct,
          "gbrap32 LE parity (full_range={full_range}, matrix={matrix:?})"
        );
      }
    }
  }

  /// Gbrap32 BE — drives the `@const BE` impl at `Gbrap32<true>` against the
  /// `Gbrap32BeFrame` alias, compared to a direct `gbrap32_to_endian::<_, true>`.
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn walk_gbrap32_be_matches_direct() {
    use crate::{
      frame::Gbrap32BeFrame,
      source::{Gbrap32, gbrap32_to_endian},
    };
    let (g, b, r, a) = planes32();
    for full_range in [false, true] {
      for matrix in MATRICES {
        let opts = YuvOptions::new()
          .maybe_full_range(full_range)
          .with_matrix(matrix);
        let src = Gbrap32BeFrame::try_new(&g, &b, &r, &a, W, H, W, W, W, W).unwrap();

        let mut via_walker = std::vec![0u8; (W * H * 4) as usize];
        let mut via_direct = std::vec![0u8; (W * H * 4) as usize];

        let mut sw = MixedSinker::<Gbrap32<true>>::new(W as usize, H as usize)
          .with_rgba(&mut via_walker)
          .unwrap();
        <Gbrap32<true> as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();

        let mut sd = MixedSinker::<Gbrap32<true>>::new(W as usize, H as usize)
          .with_rgba(&mut via_direct)
          .unwrap();
        gbrap32_to_endian::<_, true>(&src, full_range, matrix, &mut sd).unwrap();

        assert_eq!(
          via_walker, via_direct,
          "gbrap32 BE parity (full_range={full_range}, matrix={matrix:?})"
        );
      }
    }
  }

  /// GBR → RGB is a plane permute that ignores `full_range` / `matrix`, so the
  /// `with_rgb` parity above cannot prove the Walker forwards them. GBR → luma
  /// weights G/B/R through the matrix and scales by `full_range`, so luma
  /// parity does: a dropped or swapped forward yields byte-different luma.
  /// Covers the plain arm (Gbrp) and both endians of the `@const_bits` arm
  /// (Gbrp10 LE/BE).
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn walk_gbr_forwards_full_range_and_matrix_via_luma() {
    use crate::{
      frame::{Gbrp10BeFrame, Gbrp10LeFrame, GbrpFrame},
      source::{Gbrp, Gbrp10, gbrp_to, gbrp10_to, gbrp10_to_endian},
    };
    let (g8, b8, r8) = planes8();
    let (g16, b16, r16) = planes16(10);
    for full_range in [false, true] {
      for matrix in MATRICES {
        let opts = YuvOptions::new()
          .maybe_full_range(full_range)
          .with_matrix(matrix);

        let src = GbrpFrame::try_new(&g8, &b8, &r8, W, H, W, W, W).unwrap();
        let mut vw = std::vec![0u8; (W * H) as usize];
        let mut vd = std::vec![0u8; (W * H) as usize];
        let mut sw = MixedSinker::<Gbrp>::new(W as usize, H as usize)
          .with_luma(&mut vw)
          .unwrap();
        <Gbrp as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();
        let mut sd = MixedSinker::<Gbrp>::new(W as usize, H as usize)
          .with_luma(&mut vd)
          .unwrap();
        gbrp_to(&src, full_range, matrix, &mut sd).unwrap();
        assert_eq!(
          vw, vd,
          "gbrp luma parity (full_range={full_range}, matrix={matrix:?})"
        );

        let src = Gbrp10LeFrame::try_new(&g16, &b16, &r16, W, H, W, W, W).unwrap();
        let mut vw = std::vec![0u8; (W * H) as usize];
        let mut vd = std::vec![0u8; (W * H) as usize];
        let mut sw = MixedSinker::<Gbrp10>::new(W as usize, H as usize)
          .with_luma(&mut vw)
          .unwrap();
        <Gbrp10 as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();
        let mut sd = MixedSinker::<Gbrp10>::new(W as usize, H as usize)
          .with_luma(&mut vd)
          .unwrap();
        gbrp10_to(&src, full_range, matrix, &mut sd).unwrap();
        assert_eq!(
          vw, vd,
          "gbrp10 LE luma parity (full_range={full_range}, matrix={matrix:?})"
        );

        let src = Gbrp10BeFrame::try_new(&g16, &b16, &r16, W, H, W, W, W).unwrap();
        let mut vw = std::vec![0u8; (W * H) as usize];
        let mut vd = std::vec![0u8; (W * H) as usize];
        let mut sw = MixedSinker::<Gbrp10<true>>::new(W as usize, H as usize)
          .with_luma(&mut vw)
          .unwrap();
        <Gbrp10<true> as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();
        let mut sd = MixedSinker::<Gbrp10<true>>::new(W as usize, H as usize)
          .with_luma(&mut vd)
          .unwrap();
        gbrp10_to_endian::<_, true>(&src, full_range, matrix, &mut sd).unwrap();
        assert_eq!(
          vw, vd,
          "gbrp10 BE luma parity (full_range={full_range}, matrix={matrix:?})"
        );
      }
    }
  }

  /// MSB-align low-packed `u16` samples into the high `bits` of each element.
  fn msb_align(p: &[u16], bits: u32) -> std::vec::Vec<u16> {
    p.iter().map(|&s| s << (16 - bits)).collect()
  }

  /// Gbrp10Msb LE — MSB-aligned high-bit planar GBR (sample in high 10 bits).
  /// Drives the `@const_bits` impl at `<const BE = false>` against the LE
  /// `gbrp10_msb_to` wrapper.
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn walk_gbrp10_msb_le_matches_direct() {
    use crate::{
      frame::Gbrp10MsbLeFrame,
      source::{Gbrp10Msb, gbrp10_msb_to},
    };
    let (g, b, r) = planes16(10);
    let (g, b, r) = (msb_align(&g, 10), msb_align(&b, 10), msb_align(&r, 10));
    for full_range in [false, true] {
      for matrix in MATRICES {
        let opts = YuvOptions::new()
          .maybe_full_range(full_range)
          .with_matrix(matrix);
        let src = Gbrp10MsbLeFrame::new(&g, &b, &r, W, H, W, W, W);

        let mut via_walker = std::vec![0u8; (W * H * 3) as usize];
        let mut via_direct = std::vec![0u8; (W * H * 3) as usize];

        let mut sw = MixedSinker::<Gbrp10Msb>::new(W as usize, H as usize)
          .with_rgb(&mut via_walker)
          .unwrap();
        <Gbrp10Msb as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();

        let mut sd = MixedSinker::<Gbrp10Msb>::new(W as usize, H as usize)
          .with_rgb(&mut via_direct)
          .unwrap();
        gbrp10_msb_to(&src, full_range, matrix, &mut sd).unwrap();

        assert_eq!(
          via_walker, via_direct,
          "gbrp10msb LE parity (full_range={full_range}, matrix={matrix:?})"
        );
      }
    }
  }

  /// Gbrp12Msb BE — drives the `@const_bits` impl at `Gbrp12Msb<true>` against
  /// the `Gbrp12MsbBeFrame` alias, compared to a direct
  /// `gbrp12_msb_to_endian::<_, true>`.
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn walk_gbrp12_msb_be_matches_direct() {
    use crate::{
      frame::Gbrp12MsbBeFrame,
      source::{Gbrp12Msb, gbrp12_msb_to_endian},
    };
    let (g, b, r) = planes16(12);
    let (g, b, r) = (msb_align(&g, 12), msb_align(&b, 12), msb_align(&r, 12));
    for full_range in [false, true] {
      for matrix in MATRICES {
        let opts = YuvOptions::new()
          .maybe_full_range(full_range)
          .with_matrix(matrix);
        let src = Gbrp12MsbBeFrame::new(&g, &b, &r, W, H, W, W, W);

        let mut via_walker = std::vec![0u8; (W * H * 3) as usize];
        let mut via_direct = std::vec![0u8; (W * H * 3) as usize];

        let mut sw = MixedSinker::<Gbrp12Msb<true>>::new(W as usize, H as usize)
          .with_rgb(&mut via_walker)
          .unwrap();
        <Gbrp12Msb<true> as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();

        let mut sd = MixedSinker::<Gbrp12Msb<true>>::new(W as usize, H as usize)
          .with_rgb(&mut via_direct)
          .unwrap();
        gbrp12_msb_to_endian::<_, true>(&src, full_range, matrix, &mut sd).unwrap();

        assert_eq!(
          via_walker, via_direct,
          "gbrp12msb BE parity (full_range={full_range}, matrix={matrix:?})"
        );
      }
    }
  }

  /// Yuv444p10Msb LE — MSB-aligned high-bit planar YUV 4:4:4 (sample in high
  /// 10 bits). Drives the `@const_bits` impl at `<const BE = false>` against the
  /// LE `yuv444p10_msb_to` wrapper.
  #[cfg(feature = "yuv-planar")]
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn walk_yuv444p10_msb_le_matches_direct() {
    use crate::{
      frame::Yuv444p10MsbLeFrame,
      source::{Yuv444p10Msb, yuv444p10_msb_to},
    };
    let (y, u, v) = planes16(10);
    let (y, u, v) = (msb_align(&y, 10), msb_align(&u, 10), msb_align(&v, 10));
    for full_range in [false, true] {
      for matrix in MATRICES {
        let opts = YuvOptions::new()
          .maybe_full_range(full_range)
          .with_matrix(matrix);
        let src = Yuv444p10MsbLeFrame::new(&y, &u, &v, W, H, W, W, W);

        let mut via_walker = std::vec![0u8; (W * H * 3) as usize];
        let mut via_direct = std::vec![0u8; (W * H * 3) as usize];

        let mut sw = MixedSinker::<Yuv444p10Msb>::new(W as usize, H as usize)
          .with_rgb(&mut via_walker)
          .unwrap();
        <Yuv444p10Msb as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();

        let mut sd = MixedSinker::<Yuv444p10Msb>::new(W as usize, H as usize)
          .with_rgb(&mut via_direct)
          .unwrap();
        yuv444p10_msb_to(&src, full_range, matrix, &mut sd).unwrap();

        assert_eq!(
          via_walker, via_direct,
          "yuv444p10msb LE parity (full_range={full_range}, matrix={matrix:?})"
        );
      }
    }
  }

  /// Yuv444p12Msb BE — drives the `@const_bits` impl at `Yuv444p12Msb<true>`
  /// against the `Yuv444p12MsbBeFrame` alias, compared to a direct
  /// `yuv444p12_msb_to_endian::<_, true>`.
  #[cfg(feature = "yuv-planar")]
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn walk_yuv444p12_msb_be_matches_direct() {
    use crate::{
      frame::Yuv444p12MsbBeFrame,
      source::{Yuv444p12Msb, yuv444p12_msb_to_endian},
    };
    let (y, u, v) = planes16(12);
    let (y, u, v) = (msb_align(&y, 12), msb_align(&u, 12), msb_align(&v, 12));
    for full_range in [false, true] {
      for matrix in MATRICES {
        let opts = YuvOptions::new()
          .maybe_full_range(full_range)
          .with_matrix(matrix);
        let src = Yuv444p12MsbBeFrame::new(&y, &u, &v, W, H, W, W, W);

        let mut via_walker = std::vec![0u8; (W * H * 3) as usize];
        let mut via_direct = std::vec![0u8; (W * H * 3) as usize];

        let mut sw = MixedSinker::<Yuv444p12Msb<true>>::new(W as usize, H as usize)
          .with_rgb(&mut via_walker)
          .unwrap();
        <Yuv444p12Msb<true> as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();

        let mut sd = MixedSinker::<Yuv444p12Msb<true>>::new(W as usize, H as usize)
          .with_rgb(&mut via_direct)
          .unwrap();
        yuv444p12_msb_to_endian::<_, true>(&src, full_range, matrix, &mut sd).unwrap();

        assert_eq!(
          via_walker, via_direct,
          "yuv444p12msb BE parity (full_range={full_range}, matrix={matrix:?})"
        );
      }
    }
  }

  /// Like `walk_gbr_forwards_full_range_and_matrix_via_luma` but for the MSB
  /// markers: GBR → luma weights G/B/R through the matrix and scales by
  /// `full_range`, so luma parity proves the Walker forwards both knobs into
  /// `gbrp{10,12}_msb_to_endian`. Covers both endians of each MSB depth.
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn walk_gbr_msb_forwards_full_range_and_matrix_via_luma() {
    use crate::{
      frame::{Gbrp10MsbLeFrame, Gbrp12MsbBeFrame},
      source::{Gbrp10Msb, Gbrp12Msb, gbrp10_msb_to, gbrp12_msb_to_endian},
    };
    let (g10, b10, r10) = planes16(10);
    let (g10, b10, r10) = (
      msb_align(&g10, 10),
      msb_align(&b10, 10),
      msb_align(&r10, 10),
    );
    let (g12, b12, r12) = planes16(12);
    let (g12, b12, r12) = (
      msb_align(&g12, 12),
      msb_align(&b12, 12),
      msb_align(&r12, 12),
    );
    for full_range in [false, true] {
      for matrix in MATRICES {
        let opts = YuvOptions::new()
          .maybe_full_range(full_range)
          .with_matrix(matrix);

        let src = Gbrp10MsbLeFrame::new(&g10, &b10, &r10, W, H, W, W, W);
        let mut vw = std::vec![0u16; (W * H) as usize];
        let mut vd = std::vec![0u16; (W * H) as usize];
        let mut sw = MixedSinker::<Gbrp10Msb>::new(W as usize, H as usize)
          .with_luma_u16(&mut vw)
          .unwrap();
        <Gbrp10Msb as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();
        let mut sd = MixedSinker::<Gbrp10Msb>::new(W as usize, H as usize)
          .with_luma_u16(&mut vd)
          .unwrap();
        gbrp10_msb_to(&src, full_range, matrix, &mut sd).unwrap();
        assert_eq!(
          vw, vd,
          "gbrp10msb LE luma parity (full_range={full_range}, matrix={matrix:?})"
        );

        let src = Gbrp12MsbBeFrame::new(&g12, &b12, &r12, W, H, W, W, W);
        let mut vw = std::vec![0u16; (W * H) as usize];
        let mut vd = std::vec![0u16; (W * H) as usize];
        let mut sw = MixedSinker::<Gbrp12Msb<true>>::new(W as usize, H as usize)
          .with_luma_u16(&mut vw)
          .unwrap();
        <Gbrp12Msb<true> as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();
        let mut sd = MixedSinker::<Gbrp12Msb<true>>::new(W as usize, H as usize)
          .with_luma_u16(&mut vd)
          .unwrap();
        gbrp12_msb_to_endian::<_, true>(&src, full_range, matrix, &mut sd).unwrap();
        assert_eq!(
          vw, vd,
          "gbrp12msb BE luma parity (full_range={full_range}, matrix={matrix:?})"
        );
      }
    }
  }
}

// ---- Parity: packed float RGB (Rgbf16 / Rgbf32; reuse YuvOptions) ------
//
// These are already-RGB float sources, so `Rgbf* → RGB` is an
// option-ignoring clamp+scale: a dropped or swapped `(full_range, matrix)`
// forward would not show in a `with_rgb` output. `Rgbf* → luma` weights
// R/G/B through the matrix and scales by `full_range`, so luma parity does
// catch it (exactly as the GBR / X2*10 families are tested). Each test
// asserts `<Marker as Walker<_>>::walk` is byte-identical to a direct walker
// call into the same `MixedSinker::with_luma` sink, across full/limited ×
// Bt709/Bt601. Both families are endian-generic (`@const BE` arm): the LE
// case drives the impl at the marker's `<const BE = false>` default against
// the `{Fmt}LeFrame` + LE `{fmt}_to` wrapper, and the BE case drives
// `Marker<true>` against `{Fmt}BeFrame` + a direct `{fmt}_to_endian::<_,
// true>` call, so both halves of the impl are proven. Inputs are finite,
// in-range half/single-precision values so the integer output is
// well-defined (both sides run the same kernel, so byte-identity holds
// regardless, but finite inputs keep the fixture honest).

#[cfg(feature = "rgb-float")]
mod rgbf_parity {
  use super::*;
  use crate::{
    frame::{Rgbf16BeFrame, Rgbf16Frame, Rgbf32BeFrame, Rgbf32Frame},
    sinker::MixedSinker,
    source::{Rgbf16, Rgbf32, rgbf16_to, rgbf16_to_endian, rgbf32_to, rgbf32_to_endian},
  };

  const W: u32 = 8;
  const H: u32 = 4;
  const MATRICES: [KernelMatrix; 2] = [KernelMatrix::Bt709, KernelMatrix::Bt601];

  /// Deterministic finite `f32` ramp in `[0, 1]` of `n` packed R/G/B
  /// samples (no NaN/inf — the integer output stays well-defined).
  fn ramp_f32(n: usize) -> std::vec::Vec<f32> {
    (0..n)
      .map(|i| ((i * 37 + 11) % 257) as f32 / 256.0)
      .collect()
  }

  /// Same ramp narrowed to `half::f16` (IEEE-754 round-to-nearest-even).
  fn ramp_f16(n: usize) -> std::vec::Vec<half::f16> {
    ramp_f32(n).into_iter().map(half::f16::from_f32).collect()
  }

  /// Rgbf16 LE — `@const BE` arm at the marker's `<const BE = false>`
  /// default against the LE `rgbf16_to` wrapper. `width * 3` `f16` per row.
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn walk_rgbf16_le_matches_direct() {
    let buf = ramp_f16((W * H * 3) as usize);
    for full_range in [false, true] {
      for matrix in MATRICES {
        let opts = YuvOptions::new()
          .maybe_full_range(full_range)
          .with_matrix(matrix);
        let src = Rgbf16Frame::try_new(&buf, W, H, W * 3).unwrap();

        let mut via_walker = std::vec![0u8; (W * H) as usize];
        let mut via_direct = std::vec![0u8; (W * H) as usize];

        let mut sw = MixedSinker::<Rgbf16>::new(W as usize, H as usize)
          .with_luma(&mut via_walker)
          .unwrap();
        <Rgbf16 as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();

        let mut sd = MixedSinker::<Rgbf16>::new(W as usize, H as usize)
          .with_luma(&mut via_direct)
          .unwrap();
        rgbf16_to(&src, full_range, matrix, &mut sd).unwrap();

        assert_eq!(
          via_walker, via_direct,
          "rgbf16 LE luma parity (full_range={full_range}, matrix={matrix:?})"
        );
      }
    }
  }

  /// Rgbf16 BE — drives the `@const BE` impl at `Rgbf16<true>` against the
  /// `Rgbf16BeFrame` alias, compared to a direct `rgbf16_to_endian::<_,
  /// true>`.
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn walk_rgbf16_be_matches_direct() {
    let buf = ramp_f16((W * H * 3) as usize);
    for full_range in [false, true] {
      for matrix in MATRICES {
        let opts = YuvOptions::new()
          .maybe_full_range(full_range)
          .with_matrix(matrix);
        let src = Rgbf16BeFrame::try_new(&buf, W, H, W * 3).unwrap();

        let mut via_walker = std::vec![0u8; (W * H) as usize];
        let mut via_direct = std::vec![0u8; (W * H) as usize];

        let mut sw = MixedSinker::<Rgbf16<true>>::new(W as usize, H as usize)
          .with_luma(&mut via_walker)
          .unwrap();
        <Rgbf16<true> as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();

        let mut sd = MixedSinker::<Rgbf16<true>>::new(W as usize, H as usize)
          .with_luma(&mut via_direct)
          .unwrap();
        rgbf16_to_endian::<_, true>(&src, full_range, matrix, &mut sd).unwrap();

        assert_eq!(
          via_walker, via_direct,
          "rgbf16 BE luma parity (full_range={full_range}, matrix={matrix:?})"
        );
      }
    }
  }

  /// Rgbf32 LE — `@const BE` arm at the marker's `<const BE = false>`
  /// default against the LE `rgbf32_to` wrapper. `width * 3` `f32` per row.
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn walk_rgbf32_le_matches_direct() {
    let buf = ramp_f32((W * H * 3) as usize);
    for full_range in [false, true] {
      for matrix in MATRICES {
        let opts = YuvOptions::new()
          .maybe_full_range(full_range)
          .with_matrix(matrix);
        let src = Rgbf32Frame::try_new(&buf, W, H, W * 3).unwrap();

        let mut via_walker = std::vec![0u8; (W * H) as usize];
        let mut via_direct = std::vec![0u8; (W * H) as usize];

        let mut sw = MixedSinker::<Rgbf32>::new(W as usize, H as usize)
          .with_luma(&mut via_walker)
          .unwrap();
        <Rgbf32 as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();

        let mut sd = MixedSinker::<Rgbf32>::new(W as usize, H as usize)
          .with_luma(&mut via_direct)
          .unwrap();
        rgbf32_to(&src, full_range, matrix, &mut sd).unwrap();

        assert_eq!(
          via_walker, via_direct,
          "rgbf32 LE luma parity (full_range={full_range}, matrix={matrix:?})"
        );
      }
    }
  }

  /// Rgbf32 BE — drives the `@const BE` impl at `Rgbf32<true>` against the
  /// `Rgbf32BeFrame` alias, compared to a direct `rgbf32_to_endian::<_,
  /// true>`.
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn walk_rgbf32_be_matches_direct() {
    let buf = ramp_f32((W * H * 3) as usize);
    for full_range in [false, true] {
      for matrix in MATRICES {
        let opts = YuvOptions::new()
          .maybe_full_range(full_range)
          .with_matrix(matrix);
        let src = Rgbf32BeFrame::try_new(&buf, W, H, W * 3).unwrap();

        let mut via_walker = std::vec![0u8; (W * H) as usize];
        let mut via_direct = std::vec![0u8; (W * H) as usize];

        let mut sw = MixedSinker::<Rgbf32<true>>::new(W as usize, H as usize)
          .with_luma(&mut via_walker)
          .unwrap();
        <Rgbf32<true> as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();

        let mut sd = MixedSinker::<Rgbf32<true>>::new(W as usize, H as usize)
          .with_luma(&mut via_direct)
          .unwrap();
        rgbf32_to_endian::<_, true>(&src, full_range, matrix, &mut sd).unwrap();

        assert_eq!(
          via_walker, via_direct,
          "rgbf32 BE luma parity (full_range={full_range}, matrix={matrix:?})"
        );
      }
    }
  }

  /// The `with_rgb` corner too (clamp+scale to u8), so the full-channel
  /// output path is exercised alongside the forwarding-sensitive luma
  /// tests. Drives Rgbf32 LE at the marker default against `rgbf32_to`.
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn walk_rgbf32_rgb_matches_direct() {
    let buf = ramp_f32((W * H * 3) as usize);
    for full_range in [false, true] {
      for matrix in MATRICES {
        let opts = YuvOptions::new()
          .maybe_full_range(full_range)
          .with_matrix(matrix);
        let src = Rgbf32Frame::try_new(&buf, W, H, W * 3).unwrap();

        let mut via_walker = std::vec![0u8; (W * H * 3) as usize];
        let mut via_direct = std::vec![0u8; (W * H * 3) as usize];

        let mut sw = MixedSinker::<Rgbf32>::new(W as usize, H as usize)
          .with_rgb(&mut via_walker)
          .unwrap();
        <Rgbf32 as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();

        let mut sd = MixedSinker::<Rgbf32>::new(W as usize, H as usize)
          .with_rgb(&mut via_direct)
          .unwrap();
        rgbf32_to(&src, full_range, matrix, &mut sd).unwrap();

        assert_eq!(
          via_walker, via_direct,
          "rgbf32 rgb parity (full_range={full_range}, matrix={matrix:?})"
        );
      }
    }
  }
}

// ---- Parity: packed float RGBA (Rgbaf16 / Rgbaf32; reuse YuvOptions) ---
//
// The alpha-bearing twins. Luma parity catches a dropped/swapped
// `(full_range, matrix)` forward exactly as for `Rgbf*`; an extra `with_rgba`
// test exercises the real-alpha output path through the walker. Both ride the
// `@const BE` arm, so the LE case drives the `<const BE = false>` default and
// the BE case drives `Marker<true>` against the `*BeFrame` alias.

#[cfg(feature = "rgb-float")]
mod rgbaf_parity {
  use super::*;
  use crate::{
    frame::{Rgbaf16BeFrame, Rgbaf16Frame, Rgbaf32BeFrame, Rgbaf32Frame},
    sinker::MixedSinker,
    source::{Rgbaf16, Rgbaf32, rgbaf16_to, rgbaf16_to_endian, rgbaf32_to, rgbaf32_to_endian},
  };

  const W: u32 = 8;
  const H: u32 = 4;
  const MATRICES: [KernelMatrix; 2] = [KernelMatrix::Bt709, KernelMatrix::Bt601];

  fn ramp_f32(n: usize) -> std::vec::Vec<f32> {
    (0..n)
      .map(|i| ((i * 37 + 11) % 257) as f32 / 256.0)
      .collect()
  }
  fn ramp_f16(n: usize) -> std::vec::Vec<half::f16> {
    ramp_f32(n).into_iter().map(half::f16::from_f32).collect()
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn walk_rgbaf16_le_matches_direct() {
    let buf = ramp_f16((W * H * 4) as usize);
    for full_range in [false, true] {
      for matrix in MATRICES {
        let opts = YuvOptions::new()
          .maybe_full_range(full_range)
          .with_matrix(matrix);
        let src = Rgbaf16Frame::try_new(&buf, W, H, W * 4).unwrap();
        let mut via_walker = std::vec![0u8; (W * H) as usize];
        let mut via_direct = std::vec![0u8; (W * H) as usize];
        let mut sw = MixedSinker::<Rgbaf16>::new(W as usize, H as usize)
          .with_luma(&mut via_walker)
          .unwrap();
        <Rgbaf16 as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();
        let mut sd = MixedSinker::<Rgbaf16>::new(W as usize, H as usize)
          .with_luma(&mut via_direct)
          .unwrap();
        rgbaf16_to(&src, full_range, matrix, &mut sd).unwrap();
        assert_eq!(via_walker, via_direct, "rgbaf16 LE luma parity");
      }
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn walk_rgbaf16_be_matches_direct() {
    let buf = ramp_f16((W * H * 4) as usize);
    for full_range in [false, true] {
      for matrix in MATRICES {
        let opts = YuvOptions::new()
          .maybe_full_range(full_range)
          .with_matrix(matrix);
        let src = Rgbaf16BeFrame::try_new(&buf, W, H, W * 4).unwrap();
        let mut via_walker = std::vec![0u8; (W * H) as usize];
        let mut via_direct = std::vec![0u8; (W * H) as usize];
        let mut sw = MixedSinker::<Rgbaf16<true>>::new(W as usize, H as usize)
          .with_luma(&mut via_walker)
          .unwrap();
        <Rgbaf16<true> as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();
        let mut sd = MixedSinker::<Rgbaf16<true>>::new(W as usize, H as usize)
          .with_luma(&mut via_direct)
          .unwrap();
        rgbaf16_to_endian::<_, true>(&src, full_range, matrix, &mut sd).unwrap();
        assert_eq!(via_walker, via_direct, "rgbaf16 BE luma parity");
      }
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn walk_rgbaf32_le_matches_direct() {
    let buf = ramp_f32((W * H * 4) as usize);
    for full_range in [false, true] {
      for matrix in MATRICES {
        let opts = YuvOptions::new()
          .maybe_full_range(full_range)
          .with_matrix(matrix);
        let src = Rgbaf32Frame::try_new(&buf, W, H, W * 4).unwrap();
        let mut via_walker = std::vec![0u8; (W * H) as usize];
        let mut via_direct = std::vec![0u8; (W * H) as usize];
        let mut sw = MixedSinker::<Rgbaf32>::new(W as usize, H as usize)
          .with_luma(&mut via_walker)
          .unwrap();
        <Rgbaf32 as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();
        let mut sd = MixedSinker::<Rgbaf32>::new(W as usize, H as usize)
          .with_luma(&mut via_direct)
          .unwrap();
        rgbaf32_to(&src, full_range, matrix, &mut sd).unwrap();
        assert_eq!(via_walker, via_direct, "rgbaf32 LE luma parity");
      }
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn walk_rgbaf32_be_matches_direct() {
    let buf = ramp_f32((W * H * 4) as usize);
    for full_range in [false, true] {
      for matrix in MATRICES {
        let opts = YuvOptions::new()
          .maybe_full_range(full_range)
          .with_matrix(matrix);
        let src = Rgbaf32BeFrame::try_new(&buf, W, H, W * 4).unwrap();
        let mut via_walker = std::vec![0u8; (W * H) as usize];
        let mut via_direct = std::vec![0u8; (W * H) as usize];
        let mut sw = MixedSinker::<Rgbaf32<true>>::new(W as usize, H as usize)
          .with_luma(&mut via_walker)
          .unwrap();
        <Rgbaf32<true> as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();
        let mut sd = MixedSinker::<Rgbaf32<true>>::new(W as usize, H as usize)
          .with_luma(&mut via_direct)
          .unwrap();
        rgbaf32_to_endian::<_, true>(&src, full_range, matrix, &mut sd).unwrap();
        assert_eq!(via_walker, via_direct, "rgbaf32 BE luma parity");
      }
    }
  }

  /// The real-alpha `with_rgba` output through the walker (exercises the
  /// source alpha channel end-to-end).
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn walk_rgbaf32_rgba_matches_direct() {
    let buf = ramp_f32((W * H * 4) as usize);
    let opts = YuvOptions::new();
    let src = Rgbaf32Frame::try_new(&buf, W, H, W * 4).unwrap();
    let mut via_walker = std::vec![0u8; (W * H * 4) as usize];
    let mut via_direct = std::vec![0u8; (W * H * 4) as usize];
    let mut sw = MixedSinker::<Rgbaf32>::new(W as usize, H as usize)
      .with_rgba(&mut via_walker)
      .unwrap();
    <Rgbaf32 as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();
    let mut sd = MixedSinker::<Rgbaf32>::new(W as usize, H as usize)
      .with_rgba(&mut via_direct)
      .unwrap();
    rgbaf32_to(&src, opts.full_range(), opts.matrix(), &mut sd).unwrap();
    assert_eq!(via_walker, via_direct, "rgbaf32 rgba parity");
  }
}

// ---- Parity: float-luma Grayf32 (reuse YuvOptions) --------------------
//
// A gray source carries no chroma matrix, but the free `grayf32_to` /
// `grayf32_to_endian` walkers take `(full_range, matrix)`: the RGB output
// rescales limited-range luma (so `full_range` is genuinely exercised by
// `with_rgb`), while `matrix` is carried through but unused by the
// chroma-free gray kernel. Each test asserts `<Marker as Walker<_>>::walk`
// is byte-identical to a direct walker call into the same
// `MixedSinker::with_rgb` sink, across full/limited × Bt709/Bt601. Grayf32
// is endian-generic (`@const BE` arm): the LE case drives the impl at the
// marker's `<const BE = false>` default against the LE `grayf32_to` wrapper,
// and the BE case drives `Grayf32<true>` against `Grayf32BeFrame` + a direct
// `grayf32_to_endian::<_, true>`, so both halves of the impl are proven.

#[cfg(feature = "gray")]
mod grayf32_parity {
  use super::*;
  use crate::{
    PixelSink,
    frame::{Grayf32BeFrame, Grayf32Frame},
    sinker::MixedSinker,
    source::{Grayf32, Grayf32Row, Grayf32Sink, grayf32_to, grayf32_to_endian},
  };

  const W: u32 = 16;
  const H: u32 = 4;
  const MATRICES: [KernelMatrix; 2] = [KernelMatrix::Bt709, KernelMatrix::Bt601];

  /// Deterministic finite `f32` luma ramp in `[0, 1]` of `n` samples.
  fn ramp_f32(n: usize) -> std::vec::Vec<f32> {
    (0..n)
      .map(|i| ((i * 23 + 7) % 257) as f32 / 256.0)
      .collect()
  }

  /// An instrumented sink recording each row's forwarded `(full_range,
  /// matrix)`. The Grayf32 → RGB path clamps the f32 luma directly and never
  /// consumes those fields, so byte parity can't see a dropped forward — this
  /// can. `Grayf32Sink<BE>` is endian-parameterised, so the probe blanket-impls
  /// it for every `BE`.
  #[derive(Default)]
  struct Grayf32Probe {
    seen: std::vec::Vec<(bool, KernelMatrix)>,
  }
  impl PixelSink for Grayf32Probe {
    type Input<'r> = Grayf32Row<'r>;
    type Error = core::convert::Infallible;
    fn begin_frame(&mut self, _w: u32, _h: u32) -> Result<(), Self::Error> {
      Ok(())
    }
    fn process(&mut self, row: Grayf32Row<'_>) -> Result<(), Self::Error> {
      self.seen.push((row.full_range(), row.matrix()));
      Ok(())
    }
  }
  impl<const BE: bool> Grayf32Sink<BE> for Grayf32Probe {}

  /// Grayf32 → RGB clamps the f32 luma directly and discards `full_range` /
  /// `matrix` (carried on the row for sinks that observe them), so the RGB
  /// parity below can't prove the Walker forwards them. Instrument the sink and
  /// assert every emitted row carries exactly the supplied `YuvOptions`, for
  /// both the LE default and the BE marker.
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn walk_grayf32_forwards_full_range_and_matrix() {
    let y = ramp_f32((W * H) as usize);
    for full_range in [false, true] {
      for matrix in MATRICES {
        let opts = YuvOptions::new()
          .maybe_full_range(full_range)
          .with_matrix(matrix);

        let src = Grayf32Frame::try_new(&y, W, H, W).unwrap();
        let mut le = Grayf32Probe::default();
        <Grayf32 as Walker<_>>::walk(&src, &opts, &mut le).unwrap();
        assert!(!le.seen.is_empty(), "grayf32 LE walked at least one row");
        for &(fr, m) in &le.seen {
          assert_eq!(
            (fr, m),
            (full_range, matrix),
            "grayf32 LE forwards full_range/matrix into the row"
          );
        }

        let src = Grayf32BeFrame::try_new(&y, W, H, W).unwrap();
        let mut be = Grayf32Probe::default();
        <Grayf32<true> as Walker<_>>::walk(&src, &opts, &mut be).unwrap();
        assert!(!be.seen.is_empty(), "grayf32 BE walked at least one row");
        for &(fr, m) in &be.seen {
          assert_eq!(
            (fr, m),
            (full_range, matrix),
            "grayf32 BE forwards full_range/matrix into the row"
          );
        }
      }
    }
  }

  /// Grayf32 LE — `@const BE` arm at the marker's `<const BE = false>`
  /// default against the LE `grayf32_to` wrapper. `width` `f32` per row.
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn walk_grayf32_le_matches_direct() {
    let y = ramp_f32((W * H) as usize);
    for full_range in [false, true] {
      for matrix in MATRICES {
        let opts = YuvOptions::new()
          .maybe_full_range(full_range)
          .with_matrix(matrix);
        let src = Grayf32Frame::try_new(&y, W, H, W).unwrap();

        let mut via_walker = std::vec![0u8; (W * H * 3) as usize];
        let mut via_direct = std::vec![0u8; (W * H * 3) as usize];

        let mut sw = MixedSinker::<Grayf32>::new(W as usize, H as usize)
          .with_rgb(&mut via_walker)
          .unwrap();
        <Grayf32 as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();

        let mut sd = MixedSinker::<Grayf32>::new(W as usize, H as usize)
          .with_rgb(&mut via_direct)
          .unwrap();
        grayf32_to(&src, full_range, matrix, &mut sd).unwrap();

        assert_eq!(
          via_walker, via_direct,
          "grayf32 LE parity (full_range={full_range}, matrix={matrix:?})"
        );
      }
    }
  }

  /// Grayf32 BE — drives the `@const BE` impl at `Grayf32<true>` against the
  /// `Grayf32BeFrame` alias, compared to a direct `grayf32_to_endian::<_,
  /// true>`.
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn walk_grayf32_be_matches_direct() {
    let y = ramp_f32((W * H) as usize);
    for full_range in [false, true] {
      for matrix in MATRICES {
        let opts = YuvOptions::new()
          .maybe_full_range(full_range)
          .with_matrix(matrix);
        let src = Grayf32BeFrame::try_new(&y, W, H, W).unwrap();

        let mut via_walker = std::vec![0u8; (W * H * 3) as usize];
        let mut via_direct = std::vec![0u8; (W * H * 3) as usize];

        let mut sw = MixedSinker::<Grayf32<true>>::new(W as usize, H as usize)
          .with_rgb(&mut via_walker)
          .unwrap();
        <Grayf32<true> as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();

        let mut sd = MixedSinker::<Grayf32<true>>::new(W as usize, H as usize)
          .with_rgb(&mut via_direct)
          .unwrap();
        grayf32_to_endian::<_, true>(&src, full_range, matrix, &mut sd).unwrap();

        assert_eq!(
          via_walker, via_direct,
          "grayf32 BE parity (full_range={full_range}, matrix={matrix:?})"
        );
      }
    }
  }
}

// ---- Parity: float-luma Grayf16 (reuse YuvOptions) --------------------
//
// The half-float twin of the Grayf32 parity tests: the free `grayf16_to` /
// `grayf16_to_endian` walkers take `(full_range, matrix)` (the RGB output
// rescales limited-range luma; `matrix` is carried but unused by the chroma-
// free gray kernel). Grayf16 is endian-generic (`@const BE` arm): the LE case
// drives the marker's `<const BE = false>` default against `grayf16_to`, and
// the BE case drives `Grayf16<true>` against `Grayf16BeFrame` + a direct
// `grayf16_to_endian::<_, true>`.

#[cfg(feature = "gray")]
mod grayf16_parity {
  use super::*;
  use crate::{
    PixelSink,
    frame::{Grayf16BeFrame, Grayf16Frame},
    sinker::MixedSinker,
    source::{Grayf16, Grayf16Row, Grayf16Sink, grayf16_to, grayf16_to_endian},
  };
  use half::f16;

  const W: u32 = 16;
  const H: u32 = 4;
  const MATRICES: [KernelMatrix; 2] = [KernelMatrix::Bt709, KernelMatrix::Bt601];

  /// Deterministic finite `f16` luma ramp in `[0, 1]` of `n` samples.
  fn ramp_f16(n: usize) -> std::vec::Vec<f16> {
    (0..n)
      .map(|i| f16::from_f32(((i * 23 + 7) % 257) as f32 / 256.0))
      .collect()
  }

  /// An instrumented sink recording each row's forwarded `(full_range,
  /// matrix)`. `Grayf16Sink<BE>` is endian-parameterised, so the probe
  /// blanket-impls it for every `BE`.
  #[derive(Default)]
  struct Grayf16Probe {
    seen: std::vec::Vec<(bool, KernelMatrix)>,
  }
  impl PixelSink for Grayf16Probe {
    type Input<'r> = Grayf16Row<'r>;
    type Error = core::convert::Infallible;
    fn begin_frame(&mut self, _w: u32, _h: u32) -> Result<(), Self::Error> {
      Ok(())
    }
    fn process(&mut self, row: Grayf16Row<'_>) -> Result<(), Self::Error> {
      self.seen.push((row.full_range(), row.matrix()));
      Ok(())
    }
  }
  impl<const BE: bool> Grayf16Sink<BE> for Grayf16Probe {}

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn walk_grayf16_forwards_full_range_and_matrix() {
    let y = ramp_f16((W * H) as usize);
    for full_range in [false, true] {
      for matrix in MATRICES {
        let opts = YuvOptions::new()
          .maybe_full_range(full_range)
          .with_matrix(matrix);

        let src = Grayf16Frame::try_new(&y, W, H, W).unwrap();
        let mut le = Grayf16Probe::default();
        <Grayf16 as Walker<_>>::walk(&src, &opts, &mut le).unwrap();
        assert!(!le.seen.is_empty(), "grayf16 LE walked at least one row");
        for &(fr, m) in &le.seen {
          assert_eq!(
            (fr, m),
            (full_range, matrix),
            "grayf16 LE forwards full_range/matrix into the row"
          );
        }

        let src = Grayf16BeFrame::try_new(&y, W, H, W).unwrap();
        let mut be = Grayf16Probe::default();
        <Grayf16<true> as Walker<_>>::walk(&src, &opts, &mut be).unwrap();
        assert!(!be.seen.is_empty(), "grayf16 BE walked at least one row");
        for &(fr, m) in &be.seen {
          assert_eq!(
            (fr, m),
            (full_range, matrix),
            "grayf16 BE forwards full_range/matrix into the row"
          );
        }
      }
    }
  }

  /// Grayf16 LE — `@const BE` arm at the marker's `<const BE = false>` default
  /// against the LE `grayf16_to` wrapper.
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn walk_grayf16_le_matches_direct() {
    let y = ramp_f16((W * H) as usize);
    for full_range in [false, true] {
      for matrix in MATRICES {
        let opts = YuvOptions::new()
          .maybe_full_range(full_range)
          .with_matrix(matrix);
        let src = Grayf16Frame::try_new(&y, W, H, W).unwrap();

        let mut via_walker = std::vec![0u8; (W * H * 3) as usize];
        let mut via_direct = std::vec![0u8; (W * H * 3) as usize];

        let mut sw = MixedSinker::<Grayf16>::new(W as usize, H as usize)
          .with_rgb(&mut via_walker)
          .unwrap();
        <Grayf16 as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();

        let mut sd = MixedSinker::<Grayf16>::new(W as usize, H as usize)
          .with_rgb(&mut via_direct)
          .unwrap();
        grayf16_to(&src, full_range, matrix, &mut sd).unwrap();

        assert_eq!(
          via_walker, via_direct,
          "grayf16 LE parity (full_range={full_range}, matrix={matrix:?})"
        );
      }
    }
  }

  /// Grayf16 BE — drives the `@const BE` impl at `Grayf16<true>` against the
  /// `Grayf16BeFrame` alias, compared to a direct `grayf16_to_endian::<_, true>`.
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn walk_grayf16_be_matches_direct() {
    let y = ramp_f16((W * H) as usize);
    for full_range in [false, true] {
      for matrix in MATRICES {
        let opts = YuvOptions::new()
          .maybe_full_range(full_range)
          .with_matrix(matrix);
        let src = Grayf16BeFrame::try_new(&y, W, H, W).unwrap();

        let mut via_walker = std::vec![0u8; (W * H * 3) as usize];
        let mut via_direct = std::vec![0u8; (W * H * 3) as usize];

        let mut sw = MixedSinker::<Grayf16<true>>::new(W as usize, H as usize)
          .with_rgb(&mut via_walker)
          .unwrap();
        <Grayf16<true> as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();

        let mut sd = MixedSinker::<Grayf16<true>>::new(W as usize, H as usize)
          .with_rgb(&mut via_direct)
          .unwrap();
        grayf16_to_endian::<_, true>(&src, full_range, matrix, &mut sd).unwrap();

        assert_eq!(
          via_walker, via_direct,
          "grayf16 BE parity (full_range={full_range}, matrix={matrix:?})"
        );
      }
    }
  }
}

// ---- Parity: planar float GBR (Gbrpf16/32, Gbrapf16/32; Options = ()) --
//
// The float GBR walkers take only `(src, sink)` — they carry **no**
// `full_range` / `matrix` knobs — so the [`Walker`] [`Options`] is the unit
// type `()` and there is nothing to forward: byte-identity to a direct call
// into the same sink fully proves the wiring. Each test asserts
// `<Marker as Walker<_>>::walk` (with `&()` opts) is byte-identical to the
// direct `{fmt}_to` / `{fmt}_to_endian` call into the same
// `MixedSinker::with_rgb` sink. Both endians of each endian-generic family
// are proven: the LE case drives the impl at the marker's `<const BE =
// false>` default against the `{Fmt}LeFrame` + LE `{fmt}_to` wrapper, and
// the BE case drives `Marker<true>` against `{Fmt}BeFrame` + a direct
// `{fmt}_to_endian::<_, true>`. Coverage spans both topologies (single
// G/B/R `Gbrpf*` and + alpha `Gbrapf*`) at both precisions (f16 / f32).

#[cfg(feature = "gbr")]
mod gbr_float_parity {
  use super::*;
  use crate::sinker::MixedSinker;

  const W: u32 = 16;
  const H: u32 = 4;

  /// Three deterministic finite `f32` G/B/R planes in `[0, 1]`.
  fn planes_f32() -> (std::vec::Vec<f32>, std::vec::Vec<f32>, std::vec::Vec<f32>) {
    let n = (W * H) as usize;
    let g = (0..n)
      .map(|i| ((i * 17 + 3) % 257) as f32 / 256.0)
      .collect();
    let b = (0..n)
      .map(|i| ((i * 23 + 5) % 257) as f32 / 256.0)
      .collect();
    let r = (0..n)
      .map(|i| ((i * 31 + 9) % 257) as f32 / 256.0)
      .collect();
    (g, b, r)
  }

  /// Narrows an `f32` plane to `half::f16` (IEEE-754 round-to-nearest-even).
  fn to_f16(p: &[f32]) -> std::vec::Vec<half::f16> {
    p.iter().copied().map(half::f16::from_f32).collect()
  }

  /// Drives a 3-plane float GBR family (`Gbrpf16` / `Gbrpf32`). `$elem` is
  /// the plane element type ctor closure; `$try_new` the 8-arg frame ctor
  /// (`g,b,r,w,h,gs,bs,rs`); `$walker` the direct walker fn (the LE wrapper
  /// or `{fmt}_to_endian::<_, true>`); `$marker` the (possibly BE-pinned)
  /// marker.
  macro_rules! gbrp_float_case {
    ($name:ident, $marker:ty, $try_new:path, $walker:path, $mk:expr) => {
      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn $name() {
        let (g, b, r) = planes_f32();
        let mk = $mk;
        let (g, b, r) = (mk(&g), mk(&b), mk(&r));
        let src = $try_new(&g, &b, &r, W, H, W, W, W).unwrap();

        let mut via_walker = std::vec![0u8; (W * H * 3) as usize];
        let mut via_direct = std::vec![0u8; (W * H * 3) as usize];

        let mut sw = MixedSinker::<$marker>::new(W as usize, H as usize)
          .with_rgb(&mut via_walker)
          .unwrap();
        // `()` Options — the float GBR walker takes no full_range/matrix.
        <$marker as Walker<_>>::walk(&src, &(), &mut sw).unwrap();

        let mut sd = MixedSinker::<$marker>::new(W as usize, H as usize)
          .with_rgb(&mut via_direct)
          .unwrap();
        $walker(&src, &mut sd).unwrap();

        assert_eq!(
          via_walker, via_direct,
          "{} rgb parity",
          stringify!($name)
        );
      }
    };
  }

  /// Drives a 4-plane float GBRA family (`Gbrapf16` / `Gbrapf32`). Same as
  /// [`gbrp_float_case`] but the frame ctor is 10-arg
  /// (`g,b,r,a,w,h,gs,bs,rs,as`) and the alpha plane is read inside the
  /// walker.
  macro_rules! gbrap_float_case {
    ($name:ident, $marker:ty, $try_new:path, $walker:path, $mk:expr) => {
      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn $name() {
        let (g, b, r) = planes_f32();
        let a: std::vec::Vec<f32> = (0..(W * H) as usize)
          .map(|i| ((i * 13 + 1) % 257) as f32 / 256.0)
          .collect();
        let mk = $mk;
        let (g, b, r, a) = (mk(&g), mk(&b), mk(&r), mk(&a));
        let src = $try_new(&g, &b, &r, &a, W, H, W, W, W, W).unwrap();

        let mut via_walker = std::vec![0u8; (W * H * 3) as usize];
        let mut via_direct = std::vec![0u8; (W * H * 3) as usize];

        let mut sw = MixedSinker::<$marker>::new(W as usize, H as usize)
          .with_rgb(&mut via_walker)
          .unwrap();
        <$marker as Walker<_>>::walk(&src, &(), &mut sw).unwrap();

        let mut sd = MixedSinker::<$marker>::new(W as usize, H as usize)
          .with_rgb(&mut via_direct)
          .unwrap();
        $walker(&src, &mut sd).unwrap();

        assert_eq!(
          via_walker, via_direct,
          "{} rgb parity",
          stringify!($name)
        );
      }
    };
  }

  // f16 (narrowing closure); f32 (identity copy closure).
  gbrp_float_case!(
    walk_gbrpf16_le_matches_direct,
    crate::source::Gbrpf16,
    crate::frame::Gbrpf16LeFrame::try_new,
    crate::source::gbrpf16_to,
    |p: &[f32]| to_f16(p)
  );
  gbrp_float_case!(
    walk_gbrpf16_be_matches_direct,
    crate::source::Gbrpf16<true>,
    crate::frame::Gbrpf16BeFrame::try_new,
    crate::source::gbrpf16_to_endian::<_, true>,
    |p: &[f32]| to_f16(p)
  );
  gbrp_float_case!(
    walk_gbrpf32_le_matches_direct,
    crate::source::Gbrpf32,
    crate::frame::Gbrpf32LeFrame::try_new,
    crate::source::gbrpf32_to,
    |p: &[f32]| p.to_vec()
  );
  gbrp_float_case!(
    walk_gbrpf32_be_matches_direct,
    crate::source::Gbrpf32<true>,
    crate::frame::Gbrpf32BeFrame::try_new,
    crate::source::gbrpf32_to_endian::<_, true>,
    |p: &[f32]| p.to_vec()
  );

  gbrap_float_case!(
    walk_gbrapf16_le_matches_direct,
    crate::source::Gbrapf16,
    crate::frame::Gbrapf16LeFrame::try_new,
    crate::source::gbrapf16_to,
    |p: &[f32]| to_f16(p)
  );
  gbrap_float_case!(
    walk_gbrapf16_be_matches_direct,
    crate::source::Gbrapf16<true>,
    crate::frame::Gbrapf16BeFrame::try_new,
    crate::source::gbrapf16_to_endian::<_, true>,
    |p: &[f32]| to_f16(p)
  );
  gbrap_float_case!(
    walk_gbrapf32_le_matches_direct,
    crate::source::Gbrapf32,
    crate::frame::Gbrapf32LeFrame::try_new,
    crate::source::gbrapf32_to,
    |p: &[f32]| p.to_vec()
  );
  gbrap_float_case!(
    walk_gbrapf32_be_matches_direct,
    crate::source::Gbrapf32<true>,
    crate::frame::Gbrapf32BeFrame::try_new,
    crate::source::gbrapf32_to_endian::<_, true>,
    |p: &[f32]| p.to_vec()
  );
}

// ---- Parity: float-luma + alpha Yaf16 / Yaf32 (reuse YuvOptions) ------------
//
// The gray+alpha twins of the Grayf16 / Grayf32 walker parity tests: each free
// `yaf{16,32}_to` / `_to_endian` walker takes `(full_range, matrix)` (carried
// but unused by the chroma-free gray+alpha kernels). The `rgba` output exercises
// the Y broadcast and the real source alpha through the walker dispatch, and the
// BE case drives `Fmt<true>` against the `FmtBeFrame` alias.

#[cfg(feature = "gray")]
mod yaf_parity {
  use super::*;
  use crate::{
    frame::{Yaf16BeFrame, Yaf16Frame, Yaf32BeFrame, Yaf32Frame},
    sinker::MixedSinker,
    source::{Yaf16, Yaf32, yaf16_to, yaf16_to_endian, yaf32_to, yaf32_to_endian},
  };
  use half::f16;

  const W: u32 = 16;
  const H: u32 = 4;
  const MATRICES: [KernelMatrix; 2] = [KernelMatrix::Bt709, KernelMatrix::Bt601];

  /// Deterministic finite packed `[Y, A]` f32 ramp (`W*H*2` elements).
  fn ramp_f32_packed(n: usize) -> std::vec::Vec<f32> {
    (0..n)
      .map(|i| ((i * 17 + 3) % 211) as f32 / 210.0)
      .collect()
  }

  /// Deterministic finite packed `[Y, A]` f16 ramp.
  fn ramp_f16_packed(n: usize) -> std::vec::Vec<f16> {
    (0..n)
      .map(|i| f16::from_f32(((i * 23 + 7) % 257) as f32 / 256.0))
      .collect()
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn walk_yaf32_le_matches_direct() {
    let packed = ramp_f32_packed((W * H * 2) as usize);
    for full_range in [false, true] {
      for matrix in MATRICES {
        let opts = YuvOptions::new()
          .maybe_full_range(full_range)
          .with_matrix(matrix);
        let src = Yaf32Frame::try_new(&packed, W, H, W * 2).unwrap();

        let mut via_walker = std::vec![0u8; (W * H * 4) as usize];
        let mut via_direct = std::vec![0u8; (W * H * 4) as usize];

        let mut sw = MixedSinker::<Yaf32>::new(W as usize, H as usize)
          .with_rgba(&mut via_walker)
          .unwrap();
        <Yaf32 as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();

        let mut sd = MixedSinker::<Yaf32>::new(W as usize, H as usize)
          .with_rgba(&mut via_direct)
          .unwrap();
        yaf32_to(&src, full_range, matrix, &mut sd).unwrap();

        assert_eq!(via_walker, via_direct, "yaf32 LE parity");
      }
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn walk_yaf32_be_matches_direct() {
    let packed = ramp_f32_packed((W * H * 2) as usize);
    let opts = YuvOptions::new();
    let src = Yaf32BeFrame::try_new(&packed, W, H, W * 2).unwrap();

    let mut via_walker = std::vec![0u8; (W * H * 4) as usize];
    let mut via_direct = std::vec![0u8; (W * H * 4) as usize];

    let mut sw = MixedSinker::<Yaf32<true>>::new(W as usize, H as usize)
      .with_rgba(&mut via_walker)
      .unwrap();
    <Yaf32<true> as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();

    let mut sd = MixedSinker::<Yaf32<true>>::new(W as usize, H as usize)
      .with_rgba(&mut via_direct)
      .unwrap();
    yaf32_to_endian::<_, true>(&src, opts.full_range(), opts.matrix(), &mut sd).unwrap();

    assert_eq!(via_walker, via_direct, "yaf32 BE parity");
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn walk_yaf16_le_matches_direct() {
    let packed = ramp_f16_packed((W * H * 2) as usize);
    for full_range in [false, true] {
      for matrix in MATRICES {
        let opts = YuvOptions::new()
          .maybe_full_range(full_range)
          .with_matrix(matrix);
        let src = Yaf16Frame::try_new(&packed, W, H, W * 2).unwrap();

        let mut via_walker = std::vec![0u8; (W * H * 4) as usize];
        let mut via_direct = std::vec![0u8; (W * H * 4) as usize];

        let mut sw = MixedSinker::<Yaf16>::new(W as usize, H as usize)
          .with_rgba(&mut via_walker)
          .unwrap();
        <Yaf16 as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();

        let mut sd = MixedSinker::<Yaf16>::new(W as usize, H as usize)
          .with_rgba(&mut via_direct)
          .unwrap();
        yaf16_to(&src, full_range, matrix, &mut sd).unwrap();

        assert_eq!(via_walker, via_direct, "yaf16 LE parity");
      }
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn walk_yaf16_be_matches_direct() {
    let packed = ramp_f16_packed((W * H * 2) as usize);
    let opts = YuvOptions::new();
    let src = Yaf16BeFrame::try_new(&packed, W, H, W * 2).unwrap();

    let mut via_walker = std::vec![0u8; (W * H * 4) as usize];
    let mut via_direct = std::vec![0u8; (W * H * 4) as usize];

    let mut sw = MixedSinker::<Yaf16<true>>::new(W as usize, H as usize)
      .with_rgba(&mut via_walker)
      .unwrap();
    <Yaf16<true> as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();

    let mut sd = MixedSinker::<Yaf16<true>>::new(W as usize, H as usize)
      .with_rgba(&mut via_direct)
      .unwrap();
    yaf16_to_endian::<_, true>(&src, opts.full_range(), opts.matrix(), &mut sd).unwrap();

    assert_eq!(via_walker, via_direct, "yaf16 BE parity");
  }
}

/// The P-knife: the exchange between the open descriptor vocabulary and the
/// closed coefficient selector, which is the one place a matrix pixon cannot
/// decode is refused.
///
/// Through 0.1 the kernels carried the open `Matrix` and ended their
/// coefficient tables with `_ => BT.709`, so every matrix below decoded as
/// BT.709 and returned a wrong picture with no diagnostic. These assertions
/// pin the replacement policy arm by arm.
mod kernel_matrix_exchange {
  use super::*;
  use crate::KernelMatrix;

  fn spec_of(matrix: ColorMatrix) -> ColorSpec {
    ColorSpec::resolve(PixelFormat::Yuv420p, DynamicRange::Limited, matrix)
  }

  #[test]
  fn tabulated_matrices_map_to_themselves() {
    for k in [
      KernelMatrix::Bt601,
      KernelMatrix::Bt709,
      KernelMatrix::Unspecified,
      KernelMatrix::Fcc,
      KernelMatrix::Bt470Bg,
      KernelMatrix::Smpte170M,
      KernelMatrix::Smpte240m,
      KernelMatrix::YCgCo,
      KernelMatrix::Bt2020Ncl,
      KernelMatrix::ChromaDerivedNcl,
    ] {
      assert_eq!(
        spec_of(ColorMatrix::from(k)).kernel_matrix(),
        Ok(k),
        "{k:?} must survive the exchange unchanged",
      );
    }
  }

  #[test]
  fn non_affine_and_gbr_identity_take_the_documented_affine_fallback() {
    // pixon decodes these four non-affinely from the sink's transfer /
    // primaries (#303); the affine selector is consulted only when that tag
    // does not resolve, and BT.709 is the fallback #303 already shipped.
    // `Rgb` names no YCbCr basis at all — the GBR luma derivation is pinned to
    // BT.709 regardless (see the `GBR_*_LUMA_MATRIX` constants).
    for m in [
      ColorMatrix::Ictcp,
      ColorMatrix::ChromaDerivedCl,
      ColorMatrix::IptC2,
      ColorMatrix::Smpte2085,
      ColorMatrix::Rgb,
    ] {
      assert_eq!(
        spec_of(m.clone()).kernel_matrix(),
        Ok(KernelMatrix::Bt709),
        "{m:?} must take the documented BT.709 affine fallback",
      );
    }
  }

  #[test]
  fn matrices_pixon_cannot_decode_are_refused() {
    // Constant-luminance BT.2020, the two reversible YCgCo variants, and any
    // matrix this build does not name. Each of these decoded as BT.709 through
    // 0.1; an error is the only honest answer.
    for m in [
      ColorMatrix::Bt2020Cl,
      ColorMatrix::YCgCoRe,
      ColorMatrix::YCgCoRo,
      ColorMatrix::other("acescct"),
    ] {
      assert!(
        spec_of(m.clone()).kernel_matrix().is_err(),
        "{m:?} must be refused, not decoded as BT.709",
      );
      assert!(
        YuvOptions::from_color_spec(&spec_of(m.clone())).is_err(),
        "{m:?} must be refused at the walker door too",
      );
    }
  }
}
