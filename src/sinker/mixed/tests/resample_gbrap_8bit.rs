//! Alpha-aware fused-downscale coverage for the 8-bit planar GBR+alpha
//! source (`Gbrap`).
//!
//! `Gbrap` de-interleaves its G/B/R/A planes into the canonical
//! source-width `R, G, B, A` row the packed-RGBA sources stage
//! (`gbra_to_rgba_row`) and feeds the **same** 4-channel packed-RGBA
//! resample tail (`packed_rgba_resample`) — so this suite is the planar
//! twin of `resample_packed_rgba_8bit`, asserting the identical alpha
//! contract: straight RGBA is the 2x2 block mean of a direct full-res
//! conversion (alpha averaged, not forced opaque); premultiplied bins
//! premultiplied color and un-premultiplies (transparent pixels never
//! bleed); the rgb / luma / hsv outputs derive from the binned color;
//! the alpha mode is frozen per frame.

use super::*;
use crate::{
  KernelMatrix,
  resample::AreaResampler,
  sinker::{AlphaMode, MixedSinker},
  source::{Gbrap, gbrap_to},
};
use mediaframe::frame::GbrapFrame;

const SRC: usize = 8;
const OUT: usize = 4;

/// Pseudo-random canonical `R, G, B, A` plane (`SRC * SRC * 4` bytes),
/// alpha included (varying, not all-opaque).
fn canonical_frame(seed: u32) -> Vec<u8> {
  let mut buf = std::vec![0u8; SRC * SRC * 4];
  pseudo_random_u8(&mut buf, seed);
  buf
}

/// Scatter a canonical interleaved `R, G, B, A` plane into the four
/// `(g, b, r, a)` planes a `GbrapFrame` carries — the inverse of
/// `gbra_to_rgba_row`. Each plane is `width * height` bytes.
fn planes_from_canonical(rgba: &[u8], n: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
  let (mut g, mut b, mut r, mut a) = (
    std::vec![0u8; n],
    std::vec![0u8; n],
    std::vec![0u8; n],
    std::vec![0u8; n],
  );
  for i in 0..n {
    r[i] = rgba[i * 4];
    g[i] = rgba[i * 4 + 1];
    b[i] = rgba[i * 4 + 2];
    a[i] = rgba[i * 4 + 3];
  }
  (g, b, r, a)
}

fn gbrap_frame<'a>(
  g: &'a [u8],
  b: &'a [u8],
  r: &'a [u8],
  a: &'a [u8],
  w: usize,
  h: usize,
) -> GbrapFrame<'a> {
  GbrapFrame::try_new(
    g, b, r, a, w as u32, h as u32, w as u32, w as u32, w as u32, w as u32,
  )
  .unwrap()
}

/// Round-half-up 2x2 block mean of a canonical RGBA plane (every
/// channel, alpha included) — the contract for integer-ratio area
/// downscale.
fn block_mean_rgba(src: &[u8]) -> Vec<u8> {
  let mut out = std::vec![0u8; OUT * OUT * 4];
  for oy in 0..OUT {
    for ox in 0..OUT {
      for c in 0..4 {
        let mut acc = 0u32;
        for dy in 0..2 {
          for dx in 0..2 {
            acc += src[((oy * 2 + dy) * SRC + ox * 2 + dx) * 4 + c] as u32;
          }
        }
        out[(oy * OUT + ox) * 4 + c] = ((acc + 2) / 4) as u8;
      }
    }
  }
  out
}

/// Premultiply one canonical RGBA plane in place — `round(c * a / 255)`
/// per color channel, alpha untouched. Mirrors the impl's exact op.
fn premultiply(plane: &mut [u8]) {
  for px in plane.chunks_exact_mut(4) {
    let a = px[3] as u32;
    for c in &mut px[..3] {
      *c = ((*c as u32 * a + 127) / 255) as u8;
    }
  }
}

/// Un-premultiply one binned canonical RGBA plane — `round(c' * 255 /
/// a)` clamped to 255, color 0 when `a == 0`; alpha copied. Mirrors the
/// impl's exact op.
fn unpremultiply(plane: &[u8]) -> Vec<u8> {
  let mut out = std::vec![0u8; plane.len()];
  for (o, i) in out.chunks_exact_mut(4).zip(plane.chunks_exact(4)) {
    let a = i[3] as u32;
    for c in 0..3 {
      o[c] = (i[c] as u32 * 255 + a / 2)
        .checked_div(a)
        .map_or(0, |q| q.min(255)) as u8;
    }
    o[3] = i[3];
  }
  out
}

/// Drop alpha from a canonical RGBA plane → packed RGB.
fn drop_alpha(rgba: &[u8]) -> Vec<u8> {
  let mut out = std::vec![0u8; rgba.len() / 4 * 3];
  for (o, i) in out.chunks_exact_mut(3).zip(rgba.chunks_exact(4)) {
    o.copy_from_slice(&i[..3]);
  }
  out
}

/// Full-resolution canonical RGBA of the source — a direct (identity)
/// `Gbrap` conversion. The oracles bin / premultiply this.
fn direct_rgba(canonical: &[u8]) -> Vec<u8> {
  let (g, b, r, a) = planes_from_canonical(canonical, SRC * SRC);
  let src = gbrap_frame(&g, &b, &r, &a, SRC, SRC);
  let mut rgba = std::vec![0u8; SRC * SRC * 4];
  let mut sink = MixedSinker::<Gbrap>::new(SRC, SRC)
    .with_rgba(&mut rgba)
    .unwrap();
  gbrap_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  rgba
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn straight_rgba_is_block_mean_of_direct_rgba() {
  let canonical = canonical_frame(0x51A1);
  let (g, b, r, a) = planes_from_canonical(&canonical, SRC * SRC);
  let src = gbrap_frame(&g, &b, &r, &a, SRC, SRC);

  let mut rgba = std::vec![0u8; OUT * OUT * 4];
  {
    let mut sink =
      MixedSinker::<Gbrap, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap();
    gbrap_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  }
  assert_eq!(rgba, block_mean_rgba(&direct_rgba(&canonical)));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn straight_alpha_is_averaged_not_forced_opaque() {
  // Alpha varies across the frame; the resampled alpha must be the area
  // mean of the source alpha, never silently forced to 0xFF.
  let mut canonical = canonical_frame(0x9E37);
  for (i, px) in canonical.chunks_exact_mut(4).enumerate() {
    px[3] = (i as u8).wrapping_mul(3);
  }
  let (g, b, r, a) = planes_from_canonical(&canonical, SRC * SRC);
  let src = gbrap_frame(&g, &b, &r, &a, SRC, SRC);

  let mut rgba = std::vec![0u8; OUT * OUT * 4];
  {
    let mut sink =
      MixedSinker::<Gbrap, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap();
    gbrap_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  }
  let oracle = block_mean_rgba(&direct_rgba(&canonical));
  assert_eq!(rgba, oracle, "straight rgba == block mean");
  assert!(
    rgba.chunks_exact(4).any(|px| px[3] != 0xFF),
    "resampled alpha was forced opaque — area-mean alpha lost"
  );
  // Independently recompute each output alpha as the explicit area mean
  // of the source alpha plane.
  for oy in 0..OUT {
    for ox in 0..OUT {
      let mut acc = 0u32;
      for dy in 0..2 {
        for dx in 0..2 {
          acc += a[(oy * 2 + dy) * SRC + ox * 2 + dx] as u32;
        }
      }
      assert_eq!(
        rgba[(oy * OUT + ox) * 4 + 3],
        ((acc + 2) / 4) as u8,
        "alpha mean ({ox},{oy})"
      );
    }
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn straight_rgb_luma_hsv_derive_from_binned_color() {
  // rgb == drop-alpha of the binned color; luma / hsv match a direct
  // full-res conversion of the binned RGB (GBR->luma == RGB->luma).
  let canonical = canonical_frame(0xBEEF);
  let (g, b, r, a) = planes_from_canonical(&canonical, SRC * SRC);
  let src = gbrap_frame(&g, &b, &r, &a, SRC, SRC);

  let mut rgb = std::vec![0u8; OUT * OUT * 3];
  let mut rgba = std::vec![0u8; OUT * OUT * 4];
  let mut luma = std::vec![0u8; OUT * OUT];
  let mut h = std::vec![0u8; OUT * OUT];
  let mut s = std::vec![0u8; OUT * OUT];
  let mut v = std::vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Gbrap, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb(&mut rgb)
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap()
        .with_luma(&mut luma)
        .unwrap()
        .with_hsv(&mut h, &mut s, &mut v)
        .unwrap();
    gbrap_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  }

  let binned = block_mean_rgba(&direct_rgba(&canonical));
  assert_eq!(rgb, drop_alpha(&binned), "rgb == drop-alpha(binned)");

  // Reference luma / hsv from a direct `Gbrap` conversion of the binned
  // color (alpha is dropped for luma / hsv, so any α works) — keeps the
  // oracle within the `gbr` feature, byte-identical to an RGB-source
  // derivation since GBR->luma == RGB->luma.
  let binned_rgb = drop_alpha(&binned);
  let (rg, rb, rr, ra) = {
    let n = OUT * OUT;
    let (mut g, mut b, mut r, a) = (
      std::vec![0u8; n],
      std::vec![0u8; n],
      std::vec![0u8; n],
      std::vec![0xFFu8; n],
    );
    for i in 0..n {
      r[i] = binned_rgb[i * 3];
      g[i] = binned_rgb[i * 3 + 1];
      b[i] = binned_rgb[i * 3 + 2];
    }
    (g, b, r, a)
  };
  let binned_src = gbrap_frame(&rg, &rb, &rr, &ra, OUT, OUT);
  let mut luma_ref = std::vec![0u8; OUT * OUT];
  let mut h_ref = std::vec![0u8; OUT * OUT];
  let mut s_ref = std::vec![0u8; OUT * OUT];
  let mut v_ref = std::vec![0u8; OUT * OUT];
  {
    let mut sink = MixedSinker::<Gbrap>::new(OUT, OUT)
      .with_luma(&mut luma_ref)
      .unwrap()
      .with_hsv(&mut h_ref, &mut s_ref, &mut v_ref)
      .unwrap();
    gbrap_to(&binned_src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  }
  assert_eq!(luma, luma_ref, "luma");
  assert_eq!(h, h_ref, "hsv H");
  assert_eq!(s, s_ref, "hsv S");
  assert_eq!(v, v_ref, "hsv V");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn premultiplied_rgba_matches_premult_bin_unpremult_oracle() {
  let canonical = canonical_frame(0x1234);
  let (g, b, r, a) = planes_from_canonical(&canonical, SRC * SRC);
  let src = gbrap_frame(&g, &b, &r, &a, SRC, SRC);

  let mut rgba = std::vec![0u8; OUT * OUT * 4];
  let mut rgb = std::vec![0u8; OUT * OUT * 3];
  {
    let mut sink =
      MixedSinker::<Gbrap, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_alpha_mode(AlphaMode::Premultiplied)
        .with_rgba(&mut rgba)
        .unwrap()
        .with_rgb(&mut rgb)
        .unwrap();
    gbrap_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  }

  let mut pm = direct_rgba(&canonical);
  premultiply(&mut pm);
  let binned = block_mean_rgba(&pm);
  let oracle = unpremultiply(&binned);
  assert_eq!(rgba, oracle, "premult rgba");
  assert_eq!(rgb, drop_alpha(&oracle), "premult rgb");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn premultiplied_transparent_block_does_not_bleed() {
  // A fully-transparent 2x2 source block carries arbitrary stored color;
  // under premultiplied binning it must not bleed into the output.
  let mut canonical = canonical_frame(0xABCD);
  for off in [(0, 0), (1, 0), (0, 1), (1, 1)] {
    let i = off.1 * SRC + off.0;
    canonical[i * 4] = 250;
    canonical[i * 4 + 1] = 240;
    canonical[i * 4 + 2] = 230;
    canonical[i * 4 + 3] = 0;
  }
  let (g, b, r, a) = planes_from_canonical(&canonical, SRC * SRC);
  let src = gbrap_frame(&g, &b, &r, &a, SRC, SRC);

  let mut rgba = std::vec![0u8; OUT * OUT * 4];
  {
    let mut sink =
      MixedSinker::<Gbrap, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_alpha_mode(AlphaMode::Premultiplied)
        .with_rgba(&mut rgba)
        .unwrap();
    gbrap_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  }
  assert_eq!(&rgba[..4], &[0, 0, 0, 0], "transparent block bled color");

  let mut pm = direct_rgba(&canonical);
  premultiply(&mut pm);
  let oracle = unpremultiply(&block_mean_rgba(&pm));
  assert_eq!(rgba, oracle, "premult output != oracle");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn straight_and_premult_differ_under_varying_alpha() {
  // Guard that the mode flag actually changes behaviour: with non-trivial
  // alpha the two oracles produce different RGBA.
  let mut canonical = canonical_frame(0x77AA);
  for (i, px) in canonical.chunks_exact_mut(4).enumerate() {
    px[3] = 16u8.wrapping_add((i as u8).wrapping_mul(5));
  }
  let (g, b, r, a) = planes_from_canonical(&canonical, SRC * SRC);
  let render = |mode: AlphaMode| {
    let src = gbrap_frame(&g, &b, &r, &a, SRC, SRC);
    let mut rgba = std::vec![0u8; OUT * OUT * 4];
    let mut sink =
      MixedSinker::<Gbrap, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_alpha_mode(mode)
        .with_rgba(&mut rgba)
        .unwrap();
    gbrap_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
    rgba
  };
  assert_ne!(
    render(AlphaMode::Straight),
    render(AlphaMode::Premultiplied),
    "alpha mode had no effect"
  );
}

#[test]
fn default_alpha_mode_is_straight() {
  let sink = MixedSinker::<Gbrap>::new(SRC, SRC);
  assert_eq!(sink.alpha_mode(), AlphaMode::Straight);
  assert!(sink.alpha_mode().is_straight());
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn identity_plan_matches_direct_rgba() {
  let canonical = canonical_frame(0x0F0F);
  let (g, b, r, a) = planes_from_canonical(&canonical, SRC * SRC);
  let src = gbrap_frame(&g, &b, &r, &a, SRC, SRC);

  let mut rgba = std::vec![0u8; SRC * SRC * 4];
  {
    let mut sink =
      MixedSinker::<Gbrap, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(SRC, SRC))
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap();
    gbrap_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  }
  assert_eq!(rgba, direct_rgba(&canonical), "identity plan == direct");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn no_output_sink_is_a_noop() {
  // A resampling sink with no attached outputs walks every row and
  // returns Ok without touching any caller buffer (there is none).
  let canonical = canonical_frame(0x4242);
  let (g, b, r, a) = planes_from_canonical(&canonical, SRC * SRC);
  let src = gbrap_frame(&g, &b, &r, &a, SRC, SRC);
  let mut sink =
    MixedSinker::<Gbrap, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap();
  gbrap_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn cross_frame_reset_reuses_streams() {
  // begin_frame resets the stream + frozen output set, so a second frame
  // through the same sink area-downscales the second frame's input.
  let canonical = canonical_frame(0x5151);
  let (g, b, r, a) = planes_from_canonical(&canonical, SRC * SRC);
  let src = gbrap_frame(&g, &b, &r, &a, SRC, SRC);

  let mut rgba = std::vec![0u8; OUT * OUT * 4];
  {
    let mut sink =
      MixedSinker::<Gbrap, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap();
    gbrap_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
    gbrap_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  }
  assert_eq!(rgba, block_mean_rgba(&direct_rgba(&canonical)));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn accepts_alpha_mode_change_across_frames() {
  // begin_frame (re-run by the walker on each frame) re-arms the frozen
  // alpha mode, so a fresh frame may use a different mode without a false
  // ResampleOutputsChanged. The walker drives both frames in order; the
  // mode is flipped between the two walker calls.
  let canonical = canonical_frame(0xB2B2);
  let (g, b, r, a) = planes_from_canonical(&canonical, SRC * SRC);
  let src = gbrap_frame(&g, &b, &r, &a, SRC, SRC);

  let mut rgba = std::vec![0u8; OUT * OUT * 4];
  {
    let mut sink =
      MixedSinker::<Gbrap, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap();
    // Frame 1 under the default Straight.
    gbrap_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
    // Frame 2 under Premultiplied must be accepted (the freeze re-arms).
    sink.set_alpha_mode(AlphaMode::Premultiplied);
    gbrap_to(&src, true, KernelMatrix::Bt709, &mut sink)
      .expect("a fresh frame must accept a different alpha mode");
  }
  // Frame 2's output is the premultiplied oracle.
  let mut pm = direct_rgba(&canonical);
  premultiply(&mut pm);
  let oracle = unpremultiply(&block_mean_rgba(&pm));
  assert_eq!(rgba, oracle, "premult frame 2 output");
}

// NOTE on the mid-frame freeze contract.
//
// The packed-RGBA suites (`resample_packed_rgba_8bit` /
// `resample_packed_rgba_16bit`) drive single rows out of order and flip
// `set_alpha_mode` mid-frame to assert the same-frame alpha-mode freeze,
// the route-switch rejections, and the out-of-sequence-first-row retry —
// all against the **exact** shared functions `Gbrap` now reuses
// (`check_frozen_alpha_mode`, `packed_rgba_resample`,
// `packed_rgba_u16_resample`, `packed_rgb_resample_*`). A `GbrapRow` can
// only be constructed by `mediaframe`'s in-order `gbrap_to` walker
// (`GbrapRow::new` is `pub(crate)`), so a pixon test cannot inject an
// out-of-sequence row or a mid-frame mutation here. The cross-frame
// re-arm above is the part observable through the walker; the mid-frame
// rejections are covered by the shared-tail packed-RGBA suites (mirroring
// the `resample_gbrp` suite's same documented split).

// ---- Filter-resample routing (Tier 10 -> separable filter engine) -------
//
// Straight-alpha `Gbrap` scatters its G/B/R/A planes into the canonical
// `R, G, B, A` row and runs the SAME shared 4-channel packed-RGBA filter
// tail (`packed_rgba_filter_resample`) the straight `Rgba` source runs —
// so a `Gbrap` filter resample MUST be byte-identical to the equivalent
// `Rgba` filter resample of the same pixels (rgba, rgb, luma, luma_u16
// alike). That equivalence is the oracle: it pins the scatter + the filter
// routing in one shot, for an enlarge and a reduce across all three kernels.
// PIL filters R, G, B, A independently with no premultiplication, so the
// filter path is reached only for straight alpha.

/// `Gbrap` exposes a u16 luma output that the packed-RGBA `Rgba` source
/// does not, so the per-channel u16-luma oracle is the u8-luma zero-extend.
#[cfg(feature = "rgb")]
fn luma_u8_to_u16(luma: &[u8]) -> Vec<u16> {
  luma.iter().map(|&v| v as u16).collect()
}

/// Resample the same canonical RGBA pixels through a `Gbrap` filter sink
/// and the equivalent straight `Rgba` filter sink, asserting rgba / rgb /
/// luma are byte-identical and `Gbrap`'s u16 luma is the zero-extend of the
/// shared u8 luma. Generic over the kernel; covers any `out_w x out_h`.
#[cfg(feature = "rgb")]
fn assert_gbrap_filter_matches_rgba<K>(kernel: K, out_w: usize, out_h: usize, label: &str)
where
  K: crate::resample::FilterKernel + Copy,
{
  use crate::source::{Rgba, rgba_to};
  use mediaframe::frame::RgbaFrame;

  let canonical = canonical_frame(0xA15F ^ (out_w as u32) ^ ((out_h as u32) << 8));
  let (g, b, r, a) = planes_from_canonical(&canonical, SRC * SRC);
  let gbrap_src = gbrap_frame(&g, &b, &r, &a, SRC, SRC);
  let rgba_src = RgbaFrame::new(&canonical, SRC as u32, SRC as u32, (SRC * 4) as u32);

  let n = out_w * out_h;
  let (mut rgba_a, mut rgb_a, mut luma_a, mut luma_u16_a) = (
    std::vec![0u8; n * 4],
    std::vec![0u8; n * 3],
    std::vec![0u8; n],
    std::vec![0u16; n],
  );
  {
    let mut sink = MixedSinker::<Gbrap, crate::resample::FilteredResampler<K>>::with_resampler(
      SRC,
      SRC,
      crate::resample::FilteredResampler::new(out_w, out_h, kernel),
    )
    .unwrap()
    .with_rgba(&mut rgba_a)
    .unwrap()
    .with_rgb(&mut rgb_a)
    .unwrap()
    .with_luma(&mut luma_a)
    .unwrap()
    .with_luma_u16(&mut luma_u16_a)
    .unwrap();
    gbrap_to(&gbrap_src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  }

  let (mut rgba_b, mut rgb_b, mut luma_b) = (
    std::vec![0u8; n * 4],
    std::vec![0u8; n * 3],
    std::vec![0u8; n],
  );
  {
    let mut sink = MixedSinker::<Rgba, crate::resample::FilteredResampler<K>>::with_resampler(
      SRC,
      SRC,
      crate::resample::FilteredResampler::new(out_w, out_h, kernel),
    )
    .unwrap()
    .with_rgba(&mut rgba_b)
    .unwrap()
    .with_rgb(&mut rgb_b)
    .unwrap()
    .with_luma(&mut luma_b)
    .unwrap();
    rgba_to(&rgba_src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  }

  assert_eq!(rgba_a, rgba_b, "{label}: rgba (real alpha filtered)");
  assert_eq!(rgb_a, rgb_b, "{label}: rgb (alpha-dropped)");
  assert_eq!(luma_a, luma_b, "{label}: luma");
  assert_eq!(
    luma_u16_a,
    luma_u8_to_u16(&luma_b),
    "{label}: luma_u16 is the zero-extend of the shared u8 luma"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
#[cfg(feature = "rgb")]
fn gbrap_filter_downscale_matches_rgba() {
  use crate::resample::{CatmullRom, Lanczos3, Triangle};
  assert_gbrap_filter_matches_rgba(Triangle, OUT, OUT, "triangle/down");
  assert_gbrap_filter_matches_rgba(CatmullRom, OUT, OUT, "catmullrom/down");
  assert_gbrap_filter_matches_rgba(Lanczos3, OUT, OUT, "lanczos3/down");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
#[cfg(feature = "rgb")]
fn gbrap_filter_upscale_matches_rgba() {
  use crate::resample::{CatmullRom, Lanczos3, Triangle};
  // 8 -> 7 enlarge — exercises the filter engine's grow path.
  const UP: usize = 7;
  assert_gbrap_filter_matches_rgba(Triangle, UP, UP, "triangle/up");
  assert_gbrap_filter_matches_rgba(CatmullRom, UP, UP, "catmullrom/up");
  assert_gbrap_filter_matches_rgba(Lanczos3, UP, UP, "lanczos3/up");
}

/// A `Gbrap` filter plan must be accepted and produce a real (non-sentinel)
/// output — feature-independent (no `Rgba` reference), so this also guards
/// the `gbr`-solo build, where the rgb-gated oracle tests above compile out.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrap_filter_plan_is_accepted() {
  use crate::resample::{FilteredResampler, Triangle};
  // Vary alpha so the 4-channel filter path (rgba attached) is exercised.
  let mut canonical = canonical_frame(0x7A1C);
  for (i, px) in canonical.chunks_exact_mut(4).enumerate() {
    px[3] = (i as u8).wrapping_mul(5);
  }
  let (g, b, r, a) = planes_from_canonical(&canonical, SRC * SRC);
  let src = gbrap_frame(&g, &b, &r, &a, SRC, SRC);

  let mut rgba = std::vec![0xABu8; OUT * OUT * 4];
  let mut sink = MixedSinker::<Gbrap, FilteredResampler<Triangle>>::with_resampler(
    SRC,
    SRC,
    FilteredResampler::new(OUT, OUT, Triangle),
  )
  .unwrap()
  .with_rgba(&mut rgba)
  .unwrap();
  gbrap_to(&src, true, KernelMatrix::Bt709, &mut sink)
    .expect("Gbrap filter plan must be accepted (routed)");
  assert!(
    rgba.iter().any(|&px| px != 0xAB),
    "routed filter plan must write the rgba buffer"
  );
}
