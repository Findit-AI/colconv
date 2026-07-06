//! Alpha-aware fused-downscale coverage for the half-float planar-GBR+alpha
//! family ([`Gbrapf16`]).
//!
//! `Gbrapf16` widens its G/B/R/A `half::f16` planes to host-native f32,
//! scatters them into a source-width packed `R, G, B, A` f32 row, and bins
//! all four channels in float (there is no `AreaStream<f16>`). Per finalized
//! output row the binned packed row is de-interleaved into G/B/R/A
//! `half::f16` planes (each element **rounded** with `half::f16::from_f32`)
//! and the exact direct `gbrapf16_*` / `gbrpf16_*` kernels run, so every
//! output is byte-identical to a direct `Gbrapf16` conversion of the frame
//! whose per-pixel f16 G/B/R/A is the f32 area mean rounded to f16 — the
//! parity oracle. The f32-derived outputs (rgb_f32 / rgba_f32 / luma /
//! luma_u16 / hsv) widen the **rounded** f16 planes back to f32, exactly as
//! the direct path widens its f16 source.
//!
//! Premultiplied mode premultiplies in f32 (R' = R*A) before binning and
//! un-premultiplies (R = mean(R*A)/mean(A), A == 0 -> RGB = 0) before the
//! per-output round-to-f16, the f16 twin of the integer premult oracle.
//!
//! `Gbrapf16Row::new` is `pub(crate)` in `mediaframe`, so a row only reaches
//! `process` through the in-order walker; the mid-frame alpha-mode-freeze /
//! out-of-sequence rejections are covered by the shared-tail
//! `resample_packed_rgba_8bit` suite against the same `check_frozen_alpha_mode`
//! ordering and the same f32 premult math.

use super::*;
use crate::{
  resample::AreaResampler,
  sinker::{AlphaMode, MixedSinker},
  source::{Gbrapf16, gbrapf16_to},
};
use half::f16;

const SRC: usize = 8;
const OUT: usize = 4;

/// Re-encode host-native `half::f16` to wire byte storage per endianness,
/// so a fixture reads back identically on LE/BE hosts.
fn as_wire_f16(host: &[f16], be: bool) -> Vec<f16> {
  host
    .iter()
    .map(|&v| {
      let bits = v.to_bits();
      f16::from_bits(if be { bits.to_be() } else { bits.to_le() })
    })
    .collect()
}

/// Per-plane f16 planes with values exactly representable in f16 (small
/// integers / simple fractions) so the f32 block mean rounds to f16
/// deterministically. α varies and includes 0 (bleed guard). Returns
/// `(g, b, r, a)` host-native f16 planes.
fn gbra_planes() -> (Vec<f16>, Vec<f16>, Vec<f16>, Vec<f16>) {
  let n = SRC * SRC;
  let mut g = std::vec![f16::ZERO; n];
  let mut b = std::vec![f16::ZERO; n];
  let mut r = std::vec![f16::ZERO; n];
  let mut a = std::vec![f16::ZERO; n];
  for i in 0..n {
    let ii = i as i32;
    r[i] = f16::from_f32((ii % 5) as f32);
    g[i] = f16::from_f32((ii % 9) as f32);
    b[i] = f16::from_f32(((ii % 4) as f32) * 0.25);
    a[i] = f16::from_f32(((ii % 5) as f32) * 0.25);
  }
  (g, b, r, a)
}

fn frame<'a>(
  g: &'a [f16],
  b: &'a [f16],
  r: &'a [f16],
  a: &'a [f16],
  w: usize,
  h: usize,
) -> crate::frame::Gbrapf16Frame<'a> {
  crate::frame::Gbrapf16Frame::try_new(
    g, b, r, a, w as u32, h as u32, w as u32, w as u32, w as u32, w as u32,
  )
  .unwrap()
}

/// Exact 2x2 block mean (in f32) of host f16 plane values, then rounded to
/// f16 — the per-pixel value the parity oracle's binned frame carries.
fn block_mean_f16(plane: &[f16], ox: usize, oy: usize) -> f16 {
  let mut acc = 0.0f32;
  for dy in 0..2 {
    for dx in 0..2 {
      acc += plane[(oy * 2 + dy) * SRC + ox * 2 + dx].to_f32();
    }
  }
  f16::from_f32(acc / 4.0)
}

/// De-interleave a packed `R, G, B, A` f16 buffer into `(g, b, r, a)` planes.
fn planes_from_packed_rgba(rgba: &[f16], n: usize) -> (Vec<f16>, Vec<f16>, Vec<f16>, Vec<f16>) {
  let (mut g, mut b, mut r, mut a) = (
    std::vec![f16::ZERO; n],
    std::vec![f16::ZERO; n],
    std::vec![f16::ZERO; n],
    std::vec![f16::ZERO; n],
  );
  for i in 0..n {
    r[i] = rgba[i * 4];
    g[i] = rgba[i * 4 + 1];
    b[i] = rgba[i * 4 + 2];
    a[i] = rgba[i * 4 + 3];
  }
  (g, b, r, a)
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrapf16_downscale_rgba_f16_is_rounded_area_mean() {
  // The lossless-native `rgba_f16` output is the f32 2x2 block mean of every
  // channel (α included) rounded to f16.
  let (g, b, r, a) = gbra_planes();
  let (gw, bw, rw, aw) = (
    as_wire_f16(&g, false),
    as_wire_f16(&b, false),
    as_wire_f16(&r, false),
    as_wire_f16(&a, false),
  );
  let src = frame(&gw, &bw, &rw, &aw, SRC, SRC);

  let mut rgba_f16 = std::vec![f16::ZERO; OUT * OUT * 4];
  {
    let mut sink =
      MixedSinker::<Gbrapf16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgba_f16(&mut rgba_f16)
        .unwrap();
    gbrapf16_to(&src, &mut sink).unwrap();
  }
  for oy in 0..OUT {
    for ox in 0..OUT {
      let base = (oy * OUT + ox) * 4;
      assert_eq!(rgba_f16[base], block_mean_f16(&r, ox, oy), "R ({ox},{oy})");
      assert_eq!(
        rgba_f16[base + 1],
        block_mean_f16(&g, ox, oy),
        "G ({ox},{oy})"
      );
      assert_eq!(
        rgba_f16[base + 2],
        block_mean_f16(&b, ox, oy),
        "B ({ox},{oy})"
      );
      assert_eq!(
        rgba_f16[base + 3],
        block_mean_f16(&a, ox, oy),
        "A ({ox},{oy})"
      );
    }
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrapf16_all_outputs_match_direct_conversion_of_prebinned_frame() {
  // Every output at once, compared against a direct Gbrapf16 conversion of
  // the f16-rounded block-mean frame — the parity oracle.
  let (g, b, r, a) = gbra_planes();
  let (gw, bw, rw, aw) = (
    as_wire_f16(&g, false),
    as_wire_f16(&b, false),
    as_wire_f16(&r, false),
    as_wire_f16(&a, false),
  );
  let src = frame(&gw, &bw, &rw, &aw, SRC, SRC);

  let mut rgb = std::vec![0u8; OUT * OUT * 3];
  let mut rgb_u16 = std::vec![0u16; OUT * OUT * 3];
  let mut rgb_f32 = std::vec![0.0f32; OUT * OUT * 3];
  let mut rgba = std::vec![0u8; OUT * OUT * 4];
  let mut rgba_u16 = std::vec![0u16; OUT * OUT * 4];
  let mut rgba_f32 = std::vec![0.0f32; OUT * OUT * 4];
  let mut rgb_f16 = std::vec![f16::ZERO; OUT * OUT * 3];
  let mut rgba_f16 = std::vec![f16::ZERO; OUT * OUT * 4];
  let mut luma = std::vec![0u8; OUT * OUT];
  let mut luma_u16 = std::vec![0u16; OUT * OUT];
  let mut h = std::vec![0u8; OUT * OUT];
  let mut s_ = std::vec![0u8; OUT * OUT];
  let mut v_ = std::vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Gbrapf16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb(&mut rgb)
        .unwrap()
        .with_rgb_u16(&mut rgb_u16)
        .unwrap()
        .with_rgb_f32(&mut rgb_f32)
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap()
        .with_rgba_u16(&mut rgba_u16)
        .unwrap()
        .with_rgba_f32(&mut rgba_f32)
        .unwrap()
        .with_rgb_f16(&mut rgb_f16)
        .unwrap()
        .with_rgba_f16(&mut rgba_f16)
        .unwrap()
        .with_luma(&mut luma)
        .unwrap()
        .with_luma_u16(&mut luma_u16)
        .unwrap()
        .with_hsv(&mut h, &mut s_, &mut v_)
        .unwrap();
    gbrapf16_to(&src, &mut sink).unwrap();
  }

  // Reference: direct sink over the f16-rounded binned RGBA, split back into
  // LE-wire G/B/R/A f16 planes. `rgba_f16` holds exactly that rounded frame.
  let (bg, bb, br, ba) = planes_from_packed_rgba(&rgba_f16, OUT * OUT);
  let (bgw, bbw, brw, baw) = (
    as_wire_f16(&bg, false),
    as_wire_f16(&bb, false),
    as_wire_f16(&br, false),
    as_wire_f16(&ba, false),
  );
  let mut ref_rgb = std::vec![0u8; OUT * OUT * 3];
  let mut ref_rgb_u16 = std::vec![0u16; OUT * OUT * 3];
  let mut ref_rgb_f32 = std::vec![0.0f32; OUT * OUT * 3];
  let mut ref_rgba = std::vec![0u8; OUT * OUT * 4];
  let mut ref_rgba_u16 = std::vec![0u16; OUT * OUT * 4];
  let mut ref_rgba_f32 = std::vec![0.0f32; OUT * OUT * 4];
  let mut ref_rgb_f16 = std::vec![f16::ZERO; OUT * OUT * 3];
  let mut ref_luma = std::vec![0u8; OUT * OUT];
  let mut ref_luma_u16 = std::vec![0u16; OUT * OUT];
  let mut ref_h = std::vec![0u8; OUT * OUT];
  let mut ref_s = std::vec![0u8; OUT * OUT];
  let mut ref_v = std::vec![0u8; OUT * OUT];
  {
    let binned = frame(&bgw, &bbw, &brw, &baw, OUT, OUT);
    let mut sink = MixedSinker::<Gbrapf16>::new(OUT, OUT)
      .with_rgb(&mut ref_rgb)
      .unwrap()
      .with_rgb_u16(&mut ref_rgb_u16)
      .unwrap()
      .with_rgb_f32(&mut ref_rgb_f32)
      .unwrap()
      .with_rgba(&mut ref_rgba)
      .unwrap()
      .with_rgba_u16(&mut ref_rgba_u16)
      .unwrap()
      .with_rgba_f32(&mut ref_rgba_f32)
      .unwrap()
      .with_rgb_f16(&mut ref_rgb_f16)
      .unwrap()
      .with_luma(&mut ref_luma)
      .unwrap()
      .with_luma_u16(&mut ref_luma_u16)
      .unwrap()
      .with_hsv(&mut ref_h, &mut ref_s, &mut ref_v)
      .unwrap();
    gbrapf16_to(&binned, &mut sink).unwrap();
  }

  assert_eq!(rgb, ref_rgb, "rgb");
  assert_eq!(rgb_u16, ref_rgb_u16, "rgb_u16");
  assert_eq!(rgb_f32, ref_rgb_f32, "rgb_f32 (widened rounded planes)");
  assert_eq!(rgba, ref_rgba, "rgba");
  assert_eq!(rgba_u16, ref_rgba_u16, "rgba_u16");
  assert_eq!(rgba_f32, ref_rgba_f32, "rgba_f32 (widened rounded planes)");
  assert_eq!(rgb_f16, ref_rgb_f16, "rgb_f16");
  assert_eq!(luma, ref_luma, "luma");
  assert_eq!(luma_u16, ref_luma_u16, "luma_u16 (narrowed)");
  assert_eq!(h, ref_h, "hsv H");
  assert_eq!(s_, ref_s, "hsv S");
  assert_eq!(v_, ref_v, "hsv V");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrapf16_premultiplied_matches_premult_bin_unpremult_oracle() {
  // Premultiplied: premultiply in f32, bin, un-premultiply, then round to
  // f16. The oracle resolves the straight color in f32 and rounds.
  let (g, b, r, a) = gbra_planes();
  let (gw, bw, rw, aw) = (
    as_wire_f16(&g, false),
    as_wire_f16(&b, false),
    as_wire_f16(&r, false),
    as_wire_f16(&a, false),
  );
  let src = frame(&gw, &bw, &rw, &aw, SRC, SRC);

  let mut rgba_f16 = std::vec![f16::ZERO; OUT * OUT * 4];
  {
    let mut sink =
      MixedSinker::<Gbrapf16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_alpha_mode(AlphaMode::Premultiplied)
        .with_rgba_f16(&mut rgba_f16)
        .unwrap();
    gbrapf16_to(&src, &mut sink).unwrap();
  }

  for oy in 0..OUT {
    for ox in 0..OUT {
      // mean(α) in f32 over the block.
      let mut mean_a = 0.0f32;
      for dy in 0..2 {
        for dx in 0..2 {
          mean_a += a[(oy * 2 + dy) * SRC + ox * 2 + dx].to_f32();
        }
      }
      mean_a /= 4.0;
      let chans = [&r, &g, &b];
      let mut straight = [0.0f32; 3];
      for (c, plane) in chans.iter().enumerate() {
        let mut pm = 0.0f32;
        for dy in 0..2 {
          for dx in 0..2 {
            let idx = (oy * 2 + dy) * SRC + ox * 2 + dx;
            pm += plane[idx].to_f32() * a[idx].to_f32();
          }
        }
        pm /= 4.0;
        straight[c] = if mean_a == 0.0 { 0.0 } else { pm / mean_a };
      }
      let base = (oy * OUT + ox) * 4;
      assert_eq!(
        rgba_f16[base],
        f16::from_f32(straight[0]),
        "premult R ({ox},{oy})"
      );
      assert_eq!(
        rgba_f16[base + 1],
        f16::from_f32(straight[1]),
        "premult G ({ox},{oy})"
      );
      assert_eq!(
        rgba_f16[base + 2],
        f16::from_f32(straight[2]),
        "premult B ({ox},{oy})"
      );
      assert_eq!(
        rgba_f16[base + 3],
        f16::from_f32(mean_a),
        "premult A ({ox},{oy})"
      );
    }
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrapf16_premultiplied_transparent_block_does_not_bleed() {
  let (mut g, mut b, mut r, mut a) = gbra_planes();
  for off in [(0usize, 0usize), (1, 0), (0, 1), (1, 1)] {
    let i = off.1 * SRC + off.0;
    r[i] = f16::from_f32(7.0);
    g[i] = f16::from_f32(3.0);
    b[i] = f16::from_f32(5.0);
    a[i] = f16::ZERO;
  }
  let (gw, bw, rw, aw) = (
    as_wire_f16(&g, false),
    as_wire_f16(&b, false),
    as_wire_f16(&r, false),
    as_wire_f16(&a, false),
  );
  let src = frame(&gw, &bw, &rw, &aw, SRC, SRC);

  let mut rgba_f16 = std::vec![f16::ZERO; OUT * OUT * 4];
  {
    let mut sink =
      MixedSinker::<Gbrapf16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_alpha_mode(AlphaMode::Premultiplied)
        .with_rgba_f16(&mut rgba_f16)
        .unwrap();
    gbrapf16_to(&src, &mut sink).unwrap();
  }
  assert_eq!(
    &rgba_f16[..4],
    &[f16::ZERO, f16::ZERO, f16::ZERO, f16::ZERO],
    "transparent block bled color"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrapf16_fractional_ratio_matches_convert_then_bin() {
  // Fractional downscale (8 -> 3). The f32 stream is ±tolerance, so assert
  // the fused identity: output == direct convert of the rounded binned RGBA.
  const FOUT: usize = 3;
  let (g, b, r, a) = gbra_planes();
  let (gw, bw, rw, aw) = (
    as_wire_f16(&g, false),
    as_wire_f16(&b, false),
    as_wire_f16(&r, false),
    as_wire_f16(&a, false),
  );
  let src = frame(&gw, &bw, &rw, &aw, SRC, SRC);

  let mut rgba_u16 = std::vec![0u16; FOUT * FOUT * 4];
  let mut rgba_f16 = std::vec![f16::ZERO; FOUT * FOUT * 4];
  {
    let mut sink = MixedSinker::<Gbrapf16, AreaResampler>::with_resampler(
      SRC,
      SRC,
      AreaResampler::to(FOUT, FOUT),
    )
    .unwrap()
    .with_rgba_u16(&mut rgba_u16)
    .unwrap()
    .with_rgba_f16(&mut rgba_f16)
    .unwrap();
    gbrapf16_to(&src, &mut sink).unwrap();
  }
  let (bg, bb, br, ba) = planes_from_packed_rgba(&rgba_f16, FOUT * FOUT);
  let (bgw, bbw, brw, baw) = (
    as_wire_f16(&bg, false),
    as_wire_f16(&bb, false),
    as_wire_f16(&br, false),
    as_wire_f16(&ba, false),
  );
  let mut ref_rgba_u16 = std::vec![0u16; FOUT * FOUT * 4];
  {
    let binned = frame(&bgw, &bbw, &brw, &baw, FOUT, FOUT);
    let mut sink = MixedSinker::<Gbrapf16>::new(FOUT, FOUT)
      .with_rgba_u16(&mut ref_rgba_u16)
      .unwrap();
    gbrapf16_to(&binned, &mut sink).unwrap();
  }
  assert_eq!(
    rgba_u16, ref_rgba_u16,
    "fractional rgba_u16 == convert-then-bin"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrapf16_le_be_parity() {
  let (g, b, r, a) = gbra_planes();

  let render = |be: bool| {
    let (gw, bw, rw, aw) = (
      as_wire_f16(&g, be),
      as_wire_f16(&b, be),
      as_wire_f16(&r, be),
      as_wire_f16(&a, be),
    );
    let mut rgba = std::vec![0u8; OUT * OUT * 4];
    let mut rgba_u16 = std::vec![0u16; OUT * OUT * 4];
    let mut rgba_f16 = std::vec![f16::ZERO; OUT * OUT * 4];
    let mut rgba_f32 = std::vec![0.0f32; OUT * OUT * 4];
    if be {
      let src = crate::frame::Gbrapf16BeFrame::try_new(
        &gw, &bw, &rw, &aw, SRC as u32, SRC as u32, SRC as u32, SRC as u32, SRC as u32, SRC as u32,
      )
      .unwrap();
      let mut sink = MixedSinker::<crate::source::Gbrapf16<true>, AreaResampler>::with_resampler(
        SRC,
        SRC,
        AreaResampler::to(OUT, OUT),
      )
      .unwrap()
      .with_rgba(&mut rgba)
      .unwrap()
      .with_rgba_u16(&mut rgba_u16)
      .unwrap()
      .with_rgba_f16(&mut rgba_f16)
      .unwrap()
      .with_rgba_f32(&mut rgba_f32)
      .unwrap();
      crate::source::gbrapf16_to_endian(&src, &mut sink).unwrap();
    } else {
      let src = frame(&gw, &bw, &rw, &aw, SRC, SRC);
      let mut sink = MixedSinker::<Gbrapf16, AreaResampler>::with_resampler(
        SRC,
        SRC,
        AreaResampler::to(OUT, OUT),
      )
      .unwrap()
      .with_rgba(&mut rgba)
      .unwrap()
      .with_rgba_u16(&mut rgba_u16)
      .unwrap()
      .with_rgba_f16(&mut rgba_f16)
      .unwrap()
      .with_rgba_f32(&mut rgba_f32)
      .unwrap();
      gbrapf16_to(&src, &mut sink).unwrap();
    }
    (rgba, rgba_u16, rgba_f16, rgba_f32)
  };
  assert_eq!(render(false), render(true), "LE/BE outputs diverge");
}

#[test]
fn gbrapf16_default_alpha_mode_is_straight() {
  let sink = MixedSinker::<Gbrapf16>::new(SRC, SRC);
  assert_eq!(sink.alpha_mode(), AlphaMode::Straight);
  assert!(sink.alpha_mode().is_straight());
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrapf16_identity_plan_matches_new_sink() {
  let (g, b, r, a) = gbra_planes();
  let (gw, bw, rw, aw) = (
    as_wire_f16(&g, false),
    as_wire_f16(&b, false),
    as_wire_f16(&r, false),
    as_wire_f16(&a, false),
  );
  let src = frame(&gw, &bw, &rw, &aw, SRC, SRC);

  let mut direct = std::vec![f16::ZERO; SRC * SRC * 4];
  {
    let mut sink = MixedSinker::<Gbrapf16>::new(SRC, SRC)
      .with_rgba_f16(&mut direct)
      .unwrap();
    gbrapf16_to(&src, &mut sink).unwrap();
  }
  let mut via_area = std::vec![f16::ZERO; SRC * SRC * 4];
  {
    let mut sink =
      MixedSinker::<Gbrapf16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(SRC, SRC))
        .unwrap()
        .with_rgba_f16(&mut via_area)
        .unwrap();
    gbrapf16_to(&src, &mut sink).unwrap();
  }
  assert_eq!(direct, via_area, "identity-plan resample == direct sink");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrapf16_no_output_sink_is_a_noop() {
  let (g, b, r, a) = gbra_planes();
  let (gw, bw, rw, aw) = (
    as_wire_f16(&g, false),
    as_wire_f16(&b, false),
    as_wire_f16(&r, false),
    as_wire_f16(&a, false),
  );
  let src = frame(&gw, &bw, &rw, &aw, SRC, SRC);

  let mut sink =
    MixedSinker::<Gbrapf16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap();
  gbrapf16_to(&src, &mut sink).unwrap();
  assert!(
    !sink.rgba_stream_f32_allocated(),
    "no-output sink allocated the 4-channel f32 stream"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrapf16_accepts_alpha_mode_change_across_frames() {
  let (g, b, r, a) = gbra_planes();
  let (gw, bw, rw, aw) = (
    as_wire_f16(&g, false),
    as_wire_f16(&b, false),
    as_wire_f16(&r, false),
    as_wire_f16(&a, false),
  );
  let src = frame(&gw, &bw, &rw, &aw, SRC, SRC);

  let mut rgba_f16 = std::vec![f16::ZERO; OUT * OUT * 4];
  {
    let mut sink =
      MixedSinker::<Gbrapf16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgba_f16(&mut rgba_f16)
        .unwrap();
    gbrapf16_to(&src, &mut sink).unwrap();
    sink.set_alpha_mode(AlphaMode::Premultiplied);
    gbrapf16_to(&src, &mut sink).expect("a fresh frame must accept a different alpha mode");
  }
  // Frame 2 (premult) oracle for block (0,0), α channel.
  let mut mean_a = 0.0f32;
  for dy in 0..2 {
    for dx in 0..2 {
      mean_a += a[dy * SRC + dx].to_f32();
    }
  }
  mean_a /= 4.0;
  assert_eq!(rgba_f16[3], f16::from_f32(mean_a), "premult frame2 A");
}
