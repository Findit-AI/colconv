//! Fused-downscale coverage for the planar `Gbrp` source (Tier 10).
//!
//! `Gbrp` scatters its G/B/R planes into a source-width packed-RGB row
//! and feeds the shared 3-channel packed-RGB resample tail, so binning
//! IS the whole fused job and every output (rgb, rgba, luma, luma_u16,
//! hsv) must match a **direct** full-resolution `Gbrp` conversion of the
//! pre-binned frame — the parity oracle.
//!
//! The out-of-sequence / mid-frame contract is exercised by the shared
//! tail's `resample_rgb24` / `resample_padding_byte` suites against the
//! exact same shared `resample_preflight_check_only` / `frozen_outputs_check`
//! transactional preflight;
//! `GbrpRow::new` is `pub(crate)` in `mediaframe`, so a `Gbrp` row can
//! only reach `process` through the in-order `gbrp_to` walker and a
//! direct out-of-order `process` call cannot be constructed here.

use crate::{
  KernelMatrix,
  resample::{AreaResampler, FilteredResampler, Triangle},
  sinker::MixedSinker,
  source::{Gbrp, gbrp_to},
};
use mediaframe::frame::GbrpFrame;
// The Rgb24-equivalence oracle below needs the packed-RGB source; gate it on
// `rgb` so a `gbr`-only build still compiles (the gbr-only acceptance test
// below covers that build).
#[cfg(feature = "rgb")]
use crate::{
  resample::{CatmullRom, Lanczos3},
  source::{Rgb24, rgb24_to},
};
#[cfg(feature = "rgb")]
use mediaframe::frame::Rgb24Frame;

const SRC: usize = 8;
const OUT: usize = 4;
/// Upscale target — exercises the filter engine's enlarge path (PIL keeps
/// the support native when enlarging) in addition to the reduce path. Used
/// only by the rgb-gated Rgb24-equivalence upscale test, so gate it too.
#[cfg(feature = "rgb")]
const UP: usize = 13;
const MATRIX: KernelMatrix = KernelMatrix::Bt709;

/// `(r, g, b)` ramp for source pixel `i` — interior values so the
/// derived luma / HSV kernels see real math.
fn rgb_px(i: usize) -> [u8; 3] {
  [
    40 + (i as u8) * 2,
    200 - (i as u8) * 2,
    60 + ((i % 8) as u8) * 10,
  ]
}

/// Source-width packed RGB ramp (`SRC * SRC * 3` bytes).
fn rgb_ramp() -> Vec<u8> {
  let mut buf = std::vec![0u8; SRC * SRC * 3];
  for (i, px) in buf.chunks_exact_mut(3).enumerate() {
    px.copy_from_slice(&rgb_px(i));
  }
  buf
}

/// Scatter a packed-RGB buffer into `(g, b, r)` planes — the inverse of
/// `gbr_to_rgb_row`. Each plane has `width * height` bytes.
fn planes_from_packed_rgb(rgb: &[u8], n: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let (mut g, mut b, mut r) = (std::vec![0u8; n], std::vec![0u8; n], std::vec![0u8; n]);
  for i in 0..n {
    r[i] = rgb[i * 3];
    g[i] = rgb[i * 3 + 1];
    b[i] = rgb[i * 3 + 2];
  }
  (g, b, r)
}

fn gbrp_frame<'a>(g: &'a [u8], b: &'a [u8], r: &'a [u8], w: usize, h: usize) -> GbrpFrame<'a> {
  GbrpFrame::try_new(g, b, r, w as u32, h as u32, w as u32, w as u32, w as u32).unwrap()
}

/// Exact 2x2 block mean with round-half-up for the source-width RGB
/// ramp — the integer-area-mean contract for a 2:1 downscale.
fn expected_block_mean(rgb: &[u8], ox: usize, oy: usize, c: usize) -> u8 {
  let mut acc = 0u32;
  for dy in 0..2 {
    for dx in 0..2 {
      acc += rgb[((oy * 2 + dy) * SRC + ox * 2 + dx) * 3 + c] as u32;
    }
  }
  ((acc + 2) / 4) as u8
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrp_downscale_rgb_is_exact_area_mean() {
  let rgb = rgb_ramp();
  let (g, b, r) = planes_from_packed_rgb(&rgb, SRC * SRC);
  let src = gbrp_frame(&g, &b, &r, SRC, SRC);

  let mut out = std::vec![0u8; OUT * OUT * 3];
  {
    let mut sink =
      MixedSinker::<Gbrp, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb(&mut out)
        .unwrap();
    gbrp_to(&src, true, MATRIX, &mut sink).unwrap();
  }
  for oy in 0..OUT {
    for ox in 0..OUT {
      for c in 0..3 {
        assert_eq!(
          out[(oy * OUT + ox) * 3 + c],
          expected_block_mean(&rgb, ox, oy, c),
          "({ox},{oy}) c{c}"
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
fn gbrp_all_outputs_match_direct_conversion_of_prebinned_frame() {
  // Resample SRC->OUT with every output attached, then compare against a
  // full-resolution Gbrp conversion of the pre-binned (block-mean) frame
  // — the parity oracle. The fused path must produce exactly what the
  // direct path would over the already-downscaled image.
  let rgb = rgb_ramp();
  let (g, b, r) = planes_from_packed_rgb(&rgb, SRC * SRC);
  let src = gbrp_frame(&g, &b, &r, SRC, SRC);

  let make = || {
    (
      std::vec![0u8; OUT * OUT * 3], // rgb
      std::vec![0u8; OUT * OUT * 4], // rgba
      std::vec![0u8; OUT * OUT],     // luma
      std::vec![0u16; OUT * OUT],    // luma_u16
      std::vec![0u8; OUT * OUT],     // h
      std::vec![0u8; OUT * OUT],     // s
      std::vec![0u8; OUT * OUT],     // v
    )
  };

  let (mut rgb_o, mut rgba_o, mut luma_o, mut lu16_o, mut h_o, mut s_o, mut v_o) = make();
  {
    let mut sink =
      MixedSinker::<Gbrp, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb(&mut rgb_o)
        .unwrap()
        .with_rgba(&mut rgba_o)
        .unwrap()
        .with_luma(&mut luma_o)
        .unwrap()
        .with_luma_u16(&mut lu16_o)
        .unwrap()
        .with_hsv(&mut h_o, &mut s_o, &mut v_o)
        .unwrap();
    gbrp_to(&src, true, MATRIX, &mut sink).unwrap();
  }

  // Oracle: build the binned RGB by exact block-mean, scatter it back to
  // planes, and run a direct (identity) full-res Gbrp sink over it.
  let mut binned = std::vec![0u8; OUT * OUT * 3];
  for oy in 0..OUT {
    for ox in 0..OUT {
      for c in 0..3 {
        binned[(oy * OUT + ox) * 3 + c] = expected_block_mean(&rgb, ox, oy, c);
      }
    }
  }
  // The resample RGB output is itself the binned RGB — assert that link
  // explicitly, then drive the oracle from the same bytes.
  assert_eq!(rgb_o, binned, "resample rgb == exact block-mean");

  let (bg, bb, br) = planes_from_packed_rgb(&binned, OUT * OUT);
  let binned_src = gbrp_frame(&bg, &bb, &br, OUT, OUT);
  let (mut rgba_ref, mut luma_ref, mut lu16_ref, mut h_ref, mut s_ref, mut v_ref) = (
    std::vec![0u8; OUT * OUT * 4],
    std::vec![0u8; OUT * OUT],
    std::vec![0u16; OUT * OUT],
    std::vec![0u8; OUT * OUT],
    std::vec![0u8; OUT * OUT],
    std::vec![0u8; OUT * OUT],
  );
  {
    let mut sink = MixedSinker::<Gbrp>::new(OUT, OUT)
      .with_rgba(&mut rgba_ref)
      .unwrap()
      .with_luma(&mut luma_ref)
      .unwrap()
      .with_luma_u16(&mut lu16_ref)
      .unwrap()
      .with_hsv(&mut h_ref, &mut s_ref, &mut v_ref)
      .unwrap();
    gbrp_to(&binned_src, true, MATRIX, &mut sink).unwrap();
  }

  assert_eq!(rgba_o, rgba_ref, "rgba (alpha forced 0xFF)");
  assert_eq!(luma_o, luma_ref, "luma");
  assert_eq!(lu16_o, lu16_ref, "luma_u16");
  assert_eq!(h_o, h_ref, "hsv H");
  assert_eq!(s_o, s_ref, "hsv S");
  assert_eq!(v_o, v_ref, "hsv V");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrp_identity_plan_matches_new_sink() {
  let rgb = rgb_ramp();
  let (g, b, r) = planes_from_packed_rgb(&rgb, SRC * SRC);
  let src = gbrp_frame(&g, &b, &r, SRC, SRC);

  let mut direct = std::vec![0u8; SRC * SRC * 3];
  {
    let mut sink = MixedSinker::<Gbrp>::new(SRC, SRC)
      .with_rgb(&mut direct)
      .unwrap();
    gbrp_to(&src, true, MATRIX, &mut sink).unwrap();
  }
  let mut via_area = std::vec![0u8; SRC * SRC * 3];
  {
    let mut sink =
      MixedSinker::<Gbrp, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(SRC, SRC))
        .unwrap()
        .with_rgb(&mut via_area)
        .unwrap();
    gbrp_to(&src, true, MATRIX, &mut sink).unwrap();
  }
  assert_eq!(direct, via_area, "identity-plan resample == direct sink");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrp_resample_no_outputs_is_a_no_op() {
  // A resampling sink with no attached outputs is the documented legal
  // no-op: it walks every row and returns Ok without touching any
  // caller buffer (there is none to touch).
  let rgb = rgb_ramp();
  let (g, b, r) = planes_from_packed_rgb(&rgb, SRC * SRC);
  let src = gbrp_frame(&g, &b, &r, SRC, SRC);

  let mut sink =
    MixedSinker::<Gbrp, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap();
  gbrp_to(&src, true, MATRIX, &mut sink).unwrap();
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrp_resample_reuses_stream_across_frames() {
  // begin_frame resets the area stream + frozen output set, so frame 2's
  // row 0 is accepted (not rejected as out-of-sequence) and the output
  // reflects frame 2's input — without the reset it would still show
  // frame 1. Both frames share one output buffer; only the input data
  // changes.
  let rgb1 = rgb_ramp();
  let rgb2: Vec<u8> = rgb1.iter().map(|&p| 255 - p).collect();
  let (g1, b1, r1) = planes_from_packed_rgb(&rgb1, SRC * SRC);
  let (g2, b2, r2) = planes_from_packed_rgb(&rgb2, SRC * SRC);

  let mut out = std::vec![0u8; OUT * OUT * 3];
  {
    let mut sink =
      MixedSinker::<Gbrp, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb(&mut out)
        .unwrap();
    gbrp_to(
      &gbrp_frame(&g1, &b1, &r1, SRC, SRC),
      true,
      MATRIX,
      &mut sink,
    )
    .unwrap();
    gbrp_to(
      &gbrp_frame(&g2, &b2, &r2, SRC, SRC),
      true,
      MATRIX,
      &mut sink,
    )
    .unwrap();
  }

  let mut expected = std::vec![0u8; OUT * OUT * 3];
  for oy in 0..OUT {
    for ox in 0..OUT {
      for c in 0..3 {
        expected[(oy * OUT + ox) * 3 + c] = expected_block_mean(&rgb2, ox, oy, c);
      }
    }
  }
  assert_eq!(out, expected, "frame 2 output must area-downscale frame 2");
}

// ---- Filter-resample routing (Tier 10 -> separable filter engine) -------
//
// `Gbrp` scatters its G/B/R planes into a source-width packed-RGB row and
// then runs the SAME shared packed-RGB filter tail `Rgb24` runs — so a
// `Gbrp` filter resample MUST be byte-identical to the equivalent `Rgb24`
// filter resample of the same pixels (rgb, rgba, and luma alike). That
// equivalence is the oracle: it pins the scatter + the filter routing in
// one shot, for both an enlarge and a reduce and across all three kernels.

/// Build the packed `R, G, B` source frame backing both sinks (interior
/// ramps so the signed-lobe kernels see real math).
fn packed_rgb_for_filter() -> Vec<u8> {
  rgb_ramp()
}

/// Resample the same pixels through a `Gbrp` filter sink and the
/// equivalent `Rgb24` filter sink, asserting rgb / rgba / luma outputs are
/// byte-identical. Generic over the kernel; covers any `out_w x out_h`.
#[cfg(feature = "rgb")]
fn assert_gbrp_filter_matches_rgb24<K>(kernel: K, out_w: usize, out_h: usize, label: &str)
where
  K: crate::resample::FilterKernel + Copy,
{
  let rgb = packed_rgb_for_filter();
  let (g, b, r) = planes_from_packed_rgb(&rgb, SRC * SRC);
  let gbrp_src = gbrp_frame(&g, &b, &r, SRC, SRC);
  let rgb_src = Rgb24Frame::new(&rgb, SRC as u32, SRC as u32, (SRC * 3) as u32);

  let n = out_w * out_h;
  let (mut rgb_a, mut rgba_a, mut luma_a) = (
    std::vec![0u8; n * 3],
    std::vec![0u8; n * 4],
    std::vec![0u8; n],
  );
  {
    let mut sink = MixedSinker::<Gbrp, FilteredResampler<K>>::with_resampler(
      SRC,
      SRC,
      FilteredResampler::new(out_w, out_h, kernel),
    )
    .unwrap()
    .with_rgb(&mut rgb_a)
    .unwrap()
    .with_rgba(&mut rgba_a)
    .unwrap()
    .with_luma(&mut luma_a)
    .unwrap();
    gbrp_to(&gbrp_src, true, MATRIX, &mut sink).unwrap();
  }

  let (mut rgb_b, mut rgba_b, mut luma_b) = (
    std::vec![0u8; n * 3],
    std::vec![0u8; n * 4],
    std::vec![0u8; n],
  );
  {
    let mut sink = MixedSinker::<Rgb24, FilteredResampler<K>>::with_resampler(
      SRC,
      SRC,
      FilteredResampler::new(out_w, out_h, kernel),
    )
    .unwrap()
    .with_rgb(&mut rgb_b)
    .unwrap()
    .with_rgba(&mut rgba_b)
    .unwrap()
    .with_luma(&mut luma_b)
    .unwrap();
    rgb24_to(&rgb_src, true, MATRIX, &mut sink).unwrap();
  }

  assert_eq!(rgb_a, rgb_b, "{label}: rgb");
  assert_eq!(rgba_a, rgba_b, "{label}: rgba (alpha forced 0xFF)");
  assert_eq!(luma_a, luma_b, "{label}: luma");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
#[cfg(feature = "rgb")]
fn gbrp_filter_downscale_matches_rgb24() {
  assert_gbrp_filter_matches_rgb24(Triangle, OUT, OUT, "triangle/down");
  assert_gbrp_filter_matches_rgb24(CatmullRom, OUT, OUT, "catmullrom/down");
  assert_gbrp_filter_matches_rgb24(Lanczos3, OUT, OUT, "lanczos3/down");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
#[cfg(feature = "rgb")]
fn gbrp_filter_upscale_matches_rgb24() {
  assert_gbrp_filter_matches_rgb24(Triangle, UP, UP, "triangle/up");
  assert_gbrp_filter_matches_rgb24(CatmullRom, UP, UP, "catmullrom/up");
  assert_gbrp_filter_matches_rgb24(Lanczos3, UP, UP, "lanczos3/up");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrp_filter_plan_no_longer_rejected() {
  // Regression: before routing, a `Gbrp` filter plan fell through to the
  // shared tail's fence and was rejected with `UnsupportedFilter`. It must
  // now run and populate the output (non-empty, non-stale).
  let rgb = packed_rgb_for_filter();
  let (g, b, r) = planes_from_packed_rgb(&rgb, SRC * SRC);
  let src = gbrp_frame(&g, &b, &r, SRC, SRC);

  let mut out = std::vec![0xABu8; OUT * OUT * 3];
  let mut sink = MixedSinker::<Gbrp, FilteredResampler<Triangle>>::with_resampler(
    SRC,
    SRC,
    FilteredResampler::new(OUT, OUT, Triangle),
  )
  .unwrap()
  .with_rgb(&mut out)
  .unwrap();
  gbrp_to(&src, true, MATRIX, &mut sink).expect("Gbrp filter plan must be accepted (routed)");
  assert!(
    out.iter().any(|&px| px != 0xAB),
    "routed filter plan must write the output buffer"
  );
}
