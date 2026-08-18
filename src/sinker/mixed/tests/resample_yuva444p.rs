//! Alpha-aware fused-downscale coverage for the 8-bit planar 4:4:4 YUV
//! source with a real full-resolution source alpha plane, `Yuva444p`.
//!
//! `Yuva444p` routes through the packed-YUVA tail
//! ([`packed_yuva444_resample`](super::super::packed_yuva444_resample)) at
//! `SRC_BITS = 8`: no chroma subsampling — every pixel carries its own
//! U / V. The u8 colour stream bins the converted u8 RGBA row
//! (`yuva444p_to_rgba_row` — full-width chroma, real source α), and the
//! native-Y luma stream bins the Y plane directly. Each output is
//! byte-identical to a direct convert-then-bin; `Yuva444p` exposes no u16
//! colour outputs.

use crate::{
  KernelMatrix, PixelSink,
  frame::Yuva444pFrame,
  resample::{AreaResampler, ResampleError},
  sinker::{AlphaMode, MixedSinker, MixedSinkerError},
  source::{Yuva444p, Yuva444pRow, yuva444p_to},
};

const SRC: usize = 8;
const OUT: usize = 4;
const M: KernelMatrix = KernelMatrix::Bt709;
const FR: bool = true;
const FR_LIMITED: bool = false;

/// Pseudo-random Y / U / V / A planes; all four are full-resolution
/// (`SRC * SRC`) in 4:4:4. Alpha varies (not all-opaque).
fn planes(seed: u32) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
  let mut y = std::vec![0u8; SRC * SRC];
  let mut u = std::vec![0u8; SRC * SRC];
  let mut v = std::vec![0u8; SRC * SRC];
  let mut a = std::vec![0u8; SRC * SRC];
  super::pseudo_random_u8(&mut y, seed);
  super::pseudo_random_u8(&mut u, seed ^ 0x1111_1111);
  super::pseudo_random_u8(&mut v, seed ^ 0x2222_2222);
  super::pseudo_random_u8(&mut a, seed ^ 0x3333_3333);
  (y, u, v, a)
}

fn frame<'a>(y: &'a [u8], u: &'a [u8], v: &'a [u8], a: &'a [u8]) -> Yuva444pFrame<'a> {
  Yuva444pFrame::try_new(
    y, u, v, a, SRC as u32, SRC as u32, SRC as u32, SRC as u32, SRC as u32, SRC as u32,
  )
  .unwrap()
}

/// Full-resolution canonical RGBA of the source — a direct (identity)
/// `Yuva444p` conversion. The oracles bin / premultiply this.
fn direct_rgba(y: &[u8], u: &[u8], v: &[u8], a: &[u8], full_range: bool) -> Vec<u8> {
  let mut rgba = std::vec![0u8; SRC * SRC * 4];
  {
    let mut sink = MixedSinker::<Yuva444p>::new(SRC, SRC)
      .with_rgba(&mut rgba)
      .unwrap();
    yuva444p_to(&frame(y, u, v, a), full_range, M, &mut sink).unwrap();
  }
  rgba
}

/// Round-half-up 2x2 block mean of a canonical RGBA plane (alpha included).
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

/// Round-half-up 2x2 block mean of the native Y plane.
fn block_mean_native_y(y: &[u8]) -> Vec<u8> {
  let mut out = std::vec![0u8; OUT * OUT];
  for oy in 0..OUT {
    for ox in 0..OUT {
      let mut acc = 0u32;
      for dy in 0..2 {
        for dx in 0..2 {
          acc += y[(oy * 2 + dy) * SRC + ox * 2 + dx] as u32;
        }
      }
      out[oy * OUT + ox] = ((acc + 2) / 4) as u8;
    }
  }
  out
}

fn premultiply(plane: &mut [u8]) {
  for px in plane.chunks_exact_mut(4) {
    let a = px[3] as u32;
    for c in &mut px[..3] {
      *c = ((*c as u32 * a + 127) / 255) as u8;
    }
  }
}

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

fn drop_alpha(rgba: &[u8]) -> Vec<u8> {
  let mut out = std::vec![0u8; rgba.len() / 4 * 3];
  for (o, i) in out.chunks_exact_mut(3).zip(rgba.chunks_exact(4)) {
    o.copy_from_slice(&i[..3]);
  }
  out
}

/// Direct (identity) `Yuva444p` luma / luma_u16 of a binned-Y plane at the
/// given range — the byte-exact native-Y oracle (neutral chroma, opaque
/// alpha; both irrelevant to luma).
fn direct_luma_of_binned_y(binned_y: &[u8], full_range: bool) -> (Vec<u8>, Vec<u16>) {
  let n = binned_y.len();
  let u = std::vec![128u8; n];
  let v = std::vec![128u8; n];
  let a = std::vec![0xFFu8; n];
  let src = Yuva444pFrame::try_new(
    binned_y, &u, &v, &a, OUT as u32, OUT as u32, OUT as u32, OUT as u32, OUT as u32, OUT as u32,
  )
  .unwrap();
  let mut luma = std::vec![0u8; n];
  let mut lu16 = std::vec![0u16; n];
  {
    let mut sink = MixedSinker::<Yuva444p>::new(OUT, OUT)
      .with_luma(&mut luma)
      .unwrap()
      .with_luma_u16(&mut lu16)
      .unwrap();
    yuva444p_to(&src, full_range, M, &mut sink).unwrap();
  }
  (luma, lu16)
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva444p_straight_rgba_is_block_mean_of_direct() {
  let (y, u, v, a) = planes(0x51A1);
  let mut rgba = std::vec![0u8; OUT * OUT * 4];
  {
    let mut sink =
      MixedSinker::<Yuva444p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap();
    yuva444p_to(&frame(&y, &u, &v, &a), FR, M, &mut sink).unwrap();
  }
  let oracle = block_mean_rgba(&direct_rgba(&y, &u, &v, &a, FR));
  assert_eq!(rgba, oracle, "straight rgba == block mean");
  assert!(
    rgba.chunks_exact(4).any(|px| px[3] != 0xFF),
    "resampled alpha was forced opaque — area-mean alpha lost"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva444p_straight_all_outputs_derive_correctly() {
  let (y, u, v, a) = planes(0xBEEF);

  let mut rgb = std::vec![0u8; OUT * OUT * 3];
  let mut rgba = std::vec![0u8; OUT * OUT * 4];
  let mut luma = std::vec![0u8; OUT * OUT];
  let mut lu16 = std::vec![0u16; OUT * OUT];
  let mut h = std::vec![0u8; OUT * OUT];
  let mut s = std::vec![0u8; OUT * OUT];
  let mut v_hsv = std::vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Yuva444p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb(&mut rgb)
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap()
        .with_luma(&mut luma)
        .unwrap()
        .with_luma_u16(&mut lu16)
        .unwrap()
        .with_hsv(&mut h, &mut s, &mut v_hsv)
        .unwrap();
    yuva444p_to(&frame(&y, &u, &v, &a), FR, M, &mut sink).unwrap();
  }

  let binned = block_mean_rgba(&direct_rgba(&y, &u, &v, &a, FR));
  assert_eq!(rgba, binned, "rgba == block mean");
  let binned_rgb = drop_alpha(&binned);
  assert_eq!(rgb, binned_rgb, "rgb == drop-alpha(binned)");

  let y_binned = block_mean_native_y(&y);
  let (luma_ref, lu16_ref) = direct_luma_of_binned_y(&y_binned, FR);
  assert_eq!(luma, luma_ref, "luma (native Y)");
  assert_eq!(luma, y_binned, "luma == native-Y block mean");
  assert_eq!(lu16, lu16_ref, "luma_u16 (native Y zero-extended)");

  let mut h_ref = std::vec![0u8; OUT * OUT];
  let mut s_ref = std::vec![0u8; OUT * OUT];
  let mut v_ref = std::vec![0u8; OUT * OUT];
  crate::row::rgb_to_hsv_row(
    &binned_rgb,
    &mut h_ref,
    &mut s_ref,
    &mut v_ref,
    OUT * OUT,
    false,
  );
  assert_eq!(h, h_ref, "hsv H");
  assert_eq!(s, s_ref, "hsv S");
  assert_eq!(v_hsv, v_ref, "hsv V");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva444p_premultiplied_matches_premult_bin_unpremult_oracle() {
  let (y, u, v, a) = planes(0x1234);

  let mut rgba = std::vec![0u8; OUT * OUT * 4];
  let mut rgb = std::vec![0u8; OUT * OUT * 3];
  let mut lu16 = std::vec![0u16; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Yuva444p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_alpha_mode(AlphaMode::Premultiplied)
        .with_rgba(&mut rgba)
        .unwrap()
        .with_rgb(&mut rgb)
        .unwrap()
        .with_luma_u16(&mut lu16)
        .unwrap();
    yuva444p_to(&frame(&y, &u, &v, &a), FR, M, &mut sink).unwrap();
  }

  let mut pm = direct_rgba(&y, &u, &v, &a, FR);
  premultiply(&mut pm);
  let binned = block_mean_rgba(&pm);
  let oracle = unpremultiply(&binned);
  assert_eq!(rgba, oracle, "premult rgba");
  assert_eq!(rgb, drop_alpha(&oracle), "premult rgb");
  let y_binned = block_mean_native_y(&y);
  assert_eq!(
    lu16,
    y_binned.iter().map(|&p| p as u16).collect::<Vec<_>>(),
    "premult luma_u16 == native-Y bin zero-extended"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva444p_premultiplied_transparent_block_does_not_bleed() {
  let (mut y, u, v, mut a) = planes(0xABCD);
  for off in [(0, 0), (1, 0), (0, 1), (1, 1)] {
    let i = off.1 * SRC + off.0;
    y[i] = 250;
    a[i] = 0;
  }
  let mut rgba = std::vec![0u8; OUT * OUT * 4];
  {
    let mut sink =
      MixedSinker::<Yuva444p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_alpha_mode(AlphaMode::Premultiplied)
        .with_rgba(&mut rgba)
        .unwrap();
    yuva444p_to(&frame(&y, &u, &v, &a), FR, M, &mut sink).unwrap();
  }
  assert_eq!(&rgba[..4], &[0, 0, 0, 0], "transparent block bled colour");
  let mut pm = direct_rgba(&y, &u, &v, &a, FR);
  premultiply(&mut pm);
  let oracle = unpremultiply(&block_mean_rgba(&pm));
  assert_eq!(rgba, oracle, "premult output != oracle");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva444p_premultiplied_nonuniform_alpha_luma_is_native_y_not_colour() {
  // (Y, A) = (0, 255), (255, 0) alternating columns → native-Y mean 128,
  // but premult colour R collapses to 0.
  let mut y = std::vec![0u8; SRC * SRC];
  let mut a = std::vec![0u8; SRC * SRC];
  for i in 0..SRC * SRC {
    let odd = !(i % SRC).is_multiple_of(2);
    y[i] = if odd { 255 } else { 0 };
    a[i] = if odd { 0 } else { 255 };
  }
  let u = std::vec![128u8; SRC * SRC];
  let v = std::vec![128u8; SRC * SRC];

  let mut luma = std::vec![0u8; OUT * OUT];
  let mut lu16 = std::vec![0u16; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Yuva444p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_alpha_mode(AlphaMode::Premultiplied)
        .with_luma(&mut luma)
        .unwrap()
        .with_luma_u16(&mut lu16)
        .unwrap();
    yuva444p_to(&frame(&y, &u, &v, &a), FR, M, &mut sink).unwrap();
  }
  assert!(
    luma.iter().all(|&p| p == 128),
    "premult luma must be 128, got {luma:?}"
  );
  assert!(
    lu16.iter().all(|&p| p == 128),
    "premult luma_u16 must be 128, got {lu16:?}"
  );

  let y_binned = block_mean_native_y(&y);
  let (luma_ref, lu16_ref) = direct_luma_of_binned_y(&y_binned, FR);
  assert_eq!(luma, luma_ref, "premult luma == native-Y bin oracle");
  assert_eq!(lu16, lu16_ref, "premult luma_u16 == native-Y bin oracle");

  let mut pm = direct_rgba(&y, &u, &v, &a, FR);
  premultiply(&mut pm);
  let color_oracle = unpremultiply(&block_mean_rgba(&pm));
  let color_r: Vec<u8> = color_oracle.chunks_exact(4).map(|px| px[0]).collect();
  assert!(
    color_r.iter().all(|&r| r == 0),
    "fixture failed to exercise the divergence"
  );
  assert_ne!(luma, color_r, "luma must NOT be the colour-derived R");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva444p_straight_and_premult_differ_under_varying_alpha() {
  let (y, u, v, mut a) = planes(0x77AA);
  for (i, px) in a.iter_mut().enumerate() {
    *px = 16u8.wrapping_add((i as u8).wrapping_mul(5));
  }
  let render = |mode: AlphaMode| {
    let mut rgba = std::vec![0u8; OUT * OUT * 4];
    let mut sink =
      MixedSinker::<Yuva444p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_alpha_mode(mode)
        .with_rgba(&mut rgba)
        .unwrap();
    yuva444p_to(&frame(&y, &u, &v, &a), FR, M, &mut sink).unwrap();
    rgba
  };
  assert_ne!(
    render(AlphaMode::Straight),
    render(AlphaMode::Premultiplied),
    "alpha mode had no effect"
  );
}

#[test]
fn yuva444p_default_alpha_mode_is_straight() {
  let sink = MixedSinker::<Yuva444p>::new(SRC, SRC);
  assert_eq!(sink.alpha_mode(), AlphaMode::Straight);
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva444p_chroma_full_res_rgb_is_block_mean_of_direct_under_saturated_chroma() {
  // 4:4:4 carries per-pixel chroma, so a high-frequency chroma pattern must
  // still bin to the exact 2x2 mean of the direct full-res conversion (no
  // chroma upsampling approximation can creep in).
  let y = std::vec![128u8; SRC * SRC];
  let mut u = std::vec![0u8; SRC * SRC];
  let mut v = std::vec![0u8; SRC * SRC];
  let a = std::vec![200u8; SRC * SRC];
  for i in 0..SRC * SRC {
    // Per-pixel checkerboard chroma — adjacent pixels saturate opposite ways.
    let cb = ((i % SRC) + (i / SRC)).is_multiple_of(2);
    u[i] = if cb { 240 } else { 16 };
    v[i] = if cb { 16 } else { 240 };
  }
  let mut rgb = std::vec![0u8; OUT * OUT * 3];
  {
    let mut sink =
      MixedSinker::<Yuva444p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb(&mut rgb)
        .unwrap();
    yuva444p_to(&frame(&y, &u, &v, &a), FR, M, &mut sink).unwrap();
  }
  let oracle = drop_alpha(&block_mean_rgba(&direct_rgba(&y, &u, &v, &a, FR)));
  assert_eq!(
    rgb, oracle,
    "4:4:4 chroma rgb == block mean of direct convert"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva444p_identity_plan_matches_direct() {
  let (y, u, v, a) = planes(0x0F0F);
  let mut rgba = std::vec![0u8; SRC * SRC * 4];
  {
    let mut sink =
      MixedSinker::<Yuva444p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(SRC, SRC))
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap();
    yuva444p_to(&frame(&y, &u, &v, &a), FR, M, &mut sink).unwrap();
  }
  assert_eq!(
    rgba,
    direct_rgba(&y, &u, &v, &a, FR),
    "identity plan == direct"
  );
}

// Fractional-ratio coverage cross-references the resampler's straight `Rgba`
// area-bin, so it is gated to the `rgb` feature (the `Rgba` source).
#[cfg(feature = "rgb")]
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva444p_fractional_ratio_rgba_matches_oracle() {
  use crate::{
    frame::RgbaFrame,
    source::{Rgba, rgba_to},
  };
  const S: usize = 6;
  const O: usize = 4;
  let mut y = std::vec![0u8; S * S];
  let mut u = std::vec![0u8; S * S];
  let mut v = std::vec![0u8; S * S];
  let mut a = std::vec![0u8; S * S];
  super::pseudo_random_u8(&mut y, 0xF00D);
  super::pseudo_random_u8(&mut u, 0xBEEF);
  super::pseudo_random_u8(&mut v, 0xCAFE);
  super::pseudo_random_u8(&mut a, 0xD00D);
  let src = Yuva444pFrame::try_new(
    &y, &u, &v, &a, S as u32, S as u32, S as u32, S as u32, S as u32, S as u32,
  )
  .unwrap();

  let mut rgba = std::vec![0u8; O * O * 4];
  {
    let mut sink =
      MixedSinker::<Yuva444p, AreaResampler>::with_resampler(S, S, AreaResampler::to(O, O))
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap();
    yuva444p_to(&src, FR, M, &mut sink).unwrap();
  }

  let mut full = std::vec![0u8; S * S * 4];
  {
    let mut sink = MixedSinker::<Yuva444p>::new(S, S)
      .with_rgba(&mut full)
      .unwrap();
    yuva444p_to(&src, FR, M, &mut sink).unwrap();
  }
  let mut oracle = std::vec![0u8; O * O * 4];
  {
    let rgba_frame = RgbaFrame::new(&full, S as u32, S as u32, (S * 4) as u32);
    let mut sink =
      MixedSinker::<Rgba, AreaResampler>::with_resampler(S, S, AreaResampler::to(O, O))
        .unwrap()
        .with_rgba(&mut oracle)
        .unwrap();
    rgba_to(&rgba_frame, FR, M, &mut sink).unwrap();
  }
  assert_eq!(rgba, oracle, "fractional-ratio rgba == convert-then-bin");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva444p_limited_range_luma_is_native_y() {
  let (y, u, v, a) = planes(0xCAFE);
  let render = |full_range: bool| {
    let mut luma = std::vec![0u8; OUT * OUT];
    let mut lu16 = std::vec![0u16; OUT * OUT];
    let mut sink =
      MixedSinker::<Yuva444p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_luma(&mut luma)
        .unwrap()
        .with_luma_u16(&mut lu16)
        .unwrap();
    yuva444p_to(&frame(&y, &u, &v, &a), full_range, M, &mut sink).unwrap();
    (luma, lu16)
  };
  let (luma_lim, lu16_lim) = render(FR_LIMITED);
  let (luma_full, lu16_full) = render(FR);
  let y_binned = block_mean_native_y(&y);
  assert_eq!(
    luma_lim, y_binned,
    "limited-range luma == native-Y block mean"
  );
  assert_eq!(
    luma_lim, luma_full,
    "native-Y luma must be range-independent"
  );
  assert_eq!(
    lu16_lim, lu16_full,
    "native-Y luma_u16 must be range-independent"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva444p_cross_frame_reset_reuses_streams() {
  let (y, u, v, a) = planes(0x5151);
  let mut rgba = std::vec![0u8; OUT * OUT * 4];
  {
    let mut sink =
      MixedSinker::<Yuva444p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap();
    yuva444p_to(&frame(&y, &u, &v, &a), FR, M, &mut sink).unwrap();
    yuva444p_to(&frame(&y, &u, &v, &a), FR, M, &mut sink).unwrap();
  }
  assert_eq!(rgba, block_mean_rgba(&direct_rgba(&y, &u, &v, &a, FR)));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva444p_accepts_alpha_mode_change_across_frames() {
  let (y, u, v, a) = planes(0xB2B2);
  let mut rgba = std::vec![0u8; OUT * OUT * 4];
  {
    let mut sink =
      MixedSinker::<Yuva444p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap();
    yuva444p_to(&frame(&y, &u, &v, &a), FR, M, &mut sink).unwrap();
    sink.set_alpha_mode(AlphaMode::Premultiplied);
    yuva444p_to(&frame(&y, &u, &v, &a), FR, M, &mut sink)
      .expect("a fresh frame must accept a different alpha mode");
  }
  let mut pm = direct_rgba(&y, &u, &v, &a, FR);
  premultiply(&mut pm);
  let oracle = unpremultiply(&block_mean_rgba(&pm));
  assert_eq!(rgba, oracle, "premult frame 2 output");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva444p_mid_frame_alpha_mode_flip_is_rejected() {
  let (y, u, v, a) = planes(0x33AA);
  let mut rgba = std::vec![0u8; OUT * OUT * 4];
  let mut sink =
    MixedSinker::<Yuva444p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_rgba(&mut rgba)
      .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  sink
    .process(Yuva444pRow::new(
      &y[..SRC],
      &u[..SRC],
      &v[..SRC],
      &a[..SRC],
      0,
      M,
      FR,
    ))
    .unwrap();
  sink.set_alpha_mode(AlphaMode::Premultiplied);
  let err = sink
    .process(Yuva444pRow::new(
      &y[SRC..2 * SRC],
      &u[SRC..2 * SRC],
      &v[SRC..2 * SRC],
      &a[SRC..2 * SRC],
      1,
      M,
      FR,
    ))
    .unwrap_err();
  assert!(
    matches!(err, MixedSinkerError::ResampleOutputsChanged(_)),
    "mid-frame alpha flip not rejected: {err:?}"
  );
}

#[test]
fn yuva444p_out_of_sequence_first_row_is_rejected() {
  let (y, u, v, a) = planes(0x44BB);
  let mut rgba = std::vec![0u8; OUT * OUT * 4];
  let mut sink =
    MixedSinker::<Yuva444p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_rgba(&mut rgba)
      .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  let err = sink
    .process(Yuva444pRow::new(
      &y[2 * SRC..3 * SRC],
      &u[2 * SRC..3 * SRC],
      &v[2 * SRC..3 * SRC],
      &a[2 * SRC..3 * SRC],
      2,
      M,
      FR,
    ))
    .unwrap_err();
  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
    ),
    "out-of-sequence first row not rejected: {err:?}"
  );
  assert!(rgba.iter().all(|&b| b == 0), "rejected row mutated output");
}

#[test]
fn yuva444p_no_output_sink_is_a_noop() {
  let (y, u, v, a) = planes(0x4242);
  let mut sink =
    MixedSinker::<Yuva444p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap();
  yuva444p_to(&frame(&y, &u, &v, &a), FR, M, &mut sink).unwrap();
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuva444p_resample_simd_matches_scalar() {
  let (y, u, v, a) = planes(0x1357);
  let run = |simd: bool| {
    let mut rgba = std::vec![0u8; OUT * OUT * 4];
    let mut luma = std::vec![0u8; OUT * OUT];
    let mut sink =
      MixedSinker::<Yuva444p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_simd(simd)
        .with_rgba(&mut rgba)
        .unwrap()
        .with_luma(&mut luma)
        .unwrap();
    yuva444p_to(&frame(&y, &u, &v, &a), FR, M, &mut sink).unwrap();
    (rgba, luma)
  };
  assert_eq!(run(true), run(false), "Yuva444p resample SIMD != scalar");
}

#[test]
fn yuva444p_direct_hsv_only_is_rgb_free_and_infallible() {
  // #263 PR 8: on the direct (identity) path, `with_luma` +
  // `with_luma_u16` + `with_hsv` with NO rgb / rgba plane attached now
  // routes HSV through the direct `yuv_444_to_hsv_row` kernel — it does
  // NOT touch the growing rgb scratch. Proof: arm the rgb-scratch
  // allocation failpoint (which would surface `AllocationFailed` if the
  // path still grew the scratch); the row must instead SUCCEED, leave the
  // scratch unallocated, and write every output. The failpoint is
  // take-on-read, so disarm it after to avoid leaking into a later
  // same-thread test.
  let (y, u, v, a) = planes(0x7E57);
  let mut luma = std::vec![0u8; SRC * SRC];
  let mut lu16 = std::vec![0u16; SRC * SRC];
  let mut hh = std::vec![0u8; SRC * SRC];
  let mut ss = std::vec![0u8; SRC * SRC];
  let mut vv = std::vec![0u8; SRC * SRC];
  {
    let mut sink = MixedSinker::<Yuva444p>::new(SRC, SRC)
      .with_luma(&mut luma)
      .unwrap()
      .with_luma_u16(&mut lu16)
      .unwrap()
      .with_hsv(&mut hh, &mut ss, &mut vv)
      .unwrap();
    sink.begin_frame(SRC as u32, SRC as u32).unwrap();
    super::super::arm_rgb_scratch_alloc_failure();
    sink
      .process(Yuva444pRow::new(
        &y[..SRC],
        &u[..SRC],
        &v[..SRC],
        &a[..SRC],
        0,
        M,
        FR,
      ))
      .expect("HSV-only direct row must be RGB-free (no scratch alloc)");
    assert_eq!(
      sink.rgb_scratch_capacity(),
      0,
      "HSV-only direct path must not allocate the rgb scratch"
    );
  }
  super::super::disarm_rgb_scratch_alloc_failure();
  let lu16_ref: std::vec::Vec<u16> = y[..SRC].iter().map(|&b| b as u16).collect();
  assert_eq!(
    &lu16[..SRC],
    &lu16_ref[..],
    "direct luma_u16 == zero-extended Y"
  );
  assert_eq!(&luma[..SRC], &y[..SRC], "direct luma == Y verbatim");
  // HSV row 0 matches the explicit YUV→RGB→HSV reference (4:4:4 kernel).
  let mut rgb0 = std::vec![0u8; SRC * 3];
  crate::row::yuv_444_to_rgb_row(&y[..SRC], &u[..SRC], &v[..SRC], &mut rgb0, SRC, M, FR, true);
  let (mut rh, mut rs, mut rv) = (
    std::vec![0u8; SRC],
    std::vec![0u8; SRC],
    std::vec![0u8; SRC],
  );
  crate::row::rgb_to_hsv_row(&rgb0, &mut rh, &mut rs, &mut rv, SRC, true);
  assert_eq!(&hh[..SRC], &rh[..], "row 0 H");
  assert_eq!(&ss[..SRC], &rs[..], "row 0 S");
  assert_eq!(&vv[..SRC], &rv[..], "row 0 V");
}
