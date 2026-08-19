//! Alpha-aware fused-downscale coverage for the float planar-GBR+alpha
//! family ([`Gbrapf32`]).
//!
//! `Gbrapf32` scatters its G/B/R/A planes into a source-width packed
//! `R, G, B, A` f32 row and bins all four channels in float on a dedicated
//! 4-channel `AreaStream<f32>`. Per finalized output row the binned packed
//! row is de-interleaved back into G/B/R/A planes and the exact direct
//! `gbrapf32_*` (RGBA) / `gbrpf32_*` (RGB / luma / hsv, α dropped) kernels
//! run, so every output matches a **direct** full-resolution `Gbrapf32`
//! conversion of the pre-binned frame — the parity oracle. Because the f32
//! stream is a ±tolerance bin per the #143 f32 contract (not 0-ULP), the
//! integer-ratio fixtures use integer-valued samples (exact 2x2 means) for
//! byte-exact assertions and the fractional-ratio fixture asserts the
//! convert-then-bin oracle equality (the fused identity), not raw means.
//!
//! Straight bins R/G/B/A independently; premultiplied premultiplies in f32
//! (α the raw plane value, `R' = R * A`), bins, then un-premultiplies
//! (`R = mean(R*A) / mean(A)`, `A == 0 -> RGB = 0`).
//!
//! `Gbrapf32Row::new` is `pub(crate)` in `mediaframe` and the float-GBR rows
//! carry no `for_tests` door, so a row can only
//! reach `process` through the in-order walker; the mid-frame
//! alpha-mode-freeze / out-of-sequence rejections are covered by the
//! shared-tail `resample_packed_rgba_8bit` / `resample_packed_rgba_16bit`
//! suites against the exact same `check_frozen_alpha_mode` ordering, and the
//! premult math is the f32 twin of `packed_rgba_resample`'s.

use super::*;
use crate::{
  resample::AreaResampler,
  sinker::{AlphaMode, MixedSinker},
  source::{Gbrapf32, gbrapf32_to},
};

const SRC: usize = 8;
const OUT: usize = 4;

/// LE-encode a host-native `f32` slice as the `*LE` Frame contract
/// requires, so a fixture reads back identically on LE (no-op) and BE
/// (byte-swap) hosts.
fn as_le_f32(host: &[f32]) -> Vec<f32> {
  host
    .iter()
    .map(|&v| f32::from_bits(v.to_bits().to_le()))
    .collect()
}

/// BE-encode a host-native `f32` slice (the `*BE` Frame contract).
fn as_be_f32(host: &[f32]) -> Vec<f32> {
  host
    .iter()
    .map(|&v| f32::from_bits(v.to_bits().to_be()))
    .collect()
}

/// Per-plane f32 ramps with **integer-valued** samples (so the 2x2 area
/// mean is exact in f32) spanning HDR (> 1.0) and negative values, plus an
/// α plane that varies (and includes 0 and HDR α). Returns `(g, b, r, a)`
/// host-native planes.
fn gbra_planes() -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
  let n = SRC * SRC;
  let mut g = std::vec![0.0f32; n];
  let mut b = std::vec![0.0f32; n];
  let mut r = std::vec![0.0f32; n];
  let mut a = std::vec![0.0f32; n];
  for i in 0..n {
    let ii = i as i32;
    r[i] = (ii % 5) as f32;
    g[i] = (100 - ii) as f32;
    b[i] = -((ii % 7) as f32);
    // α: integer-valued ramp through {0, 1, 2, 3} so premult products and
    // means stay exactly representable; a==0 exercises the bleed guard.
    a[i] = (ii % 4) as f32;
  }
  (g, b, r, a)
}

fn frame<'a>(
  g: &'a [f32],
  b: &'a [f32],
  r: &'a [f32],
  a: &'a [f32],
  w: usize,
  h: usize,
) -> crate::frame::Gbrapf32Frame<'a> {
  crate::frame::Gbrapf32Frame::try_new(
    g, b, r, a, w as u32, h as u32, w as u32, w as u32, w as u32, w as u32,
  )
  .unwrap()
}

/// Exact 2x2 block mean over host f32 values — integer-valued samples
/// divided by 4 are exactly representable.
fn block_mean_plane(plane: &[f32], ox: usize, oy: usize) -> f32 {
  let mut acc = 0.0f64;
  for dy in 0..2 {
    for dx in 0..2 {
      acc += plane[(oy * 2 + dy) * SRC + ox * 2 + dx] as f64;
    }
  }
  (acc / 4.0) as f32
}

/// De-interleave a packed `R, G, B, A` f32 row buffer into `(g, b, r, a)`
/// planes — the inverse of the source scatter, used to drive the oracle.
fn planes_from_packed_rgba(rgba: &[f32], n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
  let (mut g, mut b, mut r, mut a) = (
    std::vec![0.0f32; n],
    std::vec![0.0f32; n],
    std::vec![0.0f32; n],
    std::vec![0.0f32; n],
  );
  for i in 0..n {
    r[i] = rgba[i * 4];
    g[i] = rgba[i * 4 + 1];
    b[i] = rgba[i * 4 + 2];
    a[i] = rgba[i * 4 + 3];
  }
  (g, b, r, a)
}

/// De-interleave a packed `R, G, B` f32 row buffer into `(g, b, r)` planes.
fn planes_from_packed_rgb(rgb: &[f32], n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
  let (mut g, mut b, mut r) = (
    std::vec![0.0f32; n],
    std::vec![0.0f32; n],
    std::vec![0.0f32; n],
  );
  for i in 0..n {
    r[i] = rgb[i * 3];
    g[i] = rgb[i * 3 + 1];
    b[i] = rgb[i * 3 + 2];
  }
  (g, b, r)
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrapf32_downscale_rgba_f32_is_exact_area_mean() {
  // The lossless `rgba_f32` output is the exact native 2x2 block mean of
  // every channel including α (integer-valued samples → exact in f32).
  let (g, b, r, a) = gbra_planes();
  let (gw, bw, rw, aw) = (as_le_f32(&g), as_le_f32(&b), as_le_f32(&r), as_le_f32(&a));
  let src = frame(&gw, &bw, &rw, &aw, SRC, SRC);

  let mut rgba_f32 = std::vec![0.0f32; OUT * OUT * 4];
  {
    let mut sink =
      MixedSinker::<Gbrapf32, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgba_f32(&mut rgba_f32)
        .unwrap();
    gbrapf32_to(&src, &mut sink).unwrap();
  }
  for oy in 0..OUT {
    for ox in 0..OUT {
      let base = (oy * OUT + ox) * 4;
      assert_eq!(
        rgba_f32[base],
        block_mean_plane(&r, ox, oy),
        "R ({ox},{oy})"
      );
      assert_eq!(
        rgba_f32[base + 1],
        block_mean_plane(&g, ox, oy),
        "G ({ox},{oy})"
      );
      assert_eq!(
        rgba_f32[base + 2],
        block_mean_plane(&b, ox, oy),
        "B ({ox},{oy})"
      );
      assert_eq!(
        rgba_f32[base + 3],
        block_mean_plane(&a, ox, oy),
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
fn gbrapf32_straight_alpha_is_averaged_not_forced_opaque() {
  let (g, b, r, a) = gbra_planes();
  let (gw, bw, rw, aw) = (as_le_f32(&g), as_le_f32(&b), as_le_f32(&r), as_le_f32(&a));
  let src = frame(&gw, &bw, &rw, &aw, SRC, SRC);

  let mut rgba_f32 = std::vec![0.0f32; OUT * OUT * 4];
  {
    let mut sink =
      MixedSinker::<Gbrapf32, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgba_f32(&mut rgba_f32)
        .unwrap();
    gbrapf32_to(&src, &mut sink).unwrap();
  }
  // Some output α must differ from the per-pixel α (area mean really ran).
  assert!(
    rgba_f32
      .chunks_exact(4)
      .any(|px| px[3] != 0.0 && px[3] != 1.0),
    "resampled α was forced to a constant — area-mean α lost"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrapf32_all_outputs_match_direct_conversion_of_prebinned_frame() {
  // Resample SRC->OUT with every output attached, then compare against a
  // full-resolution direct Gbrapf32 conversion of the pre-binned frame —
  // the parity oracle. Every output matches the direct path, including
  // luma_u16 at the direct path's narrowed (8-bit-in-u16) precision.
  let (g, b, r, a) = gbra_planes();
  let (gw, bw, rw, aw) = (as_le_f32(&g), as_le_f32(&b), as_le_f32(&r), as_le_f32(&a));
  let src = frame(&gw, &bw, &rw, &aw, SRC, SRC);

  let mut rgb = std::vec![0u8; OUT * OUT * 3];
  let mut rgb_u16 = std::vec![0u16; OUT * OUT * 3];
  let mut rgb_f32 = std::vec![0.0f32; OUT * OUT * 3];
  let mut rgba = std::vec![0u8; OUT * OUT * 4];
  let mut rgba_u16 = std::vec![0u16; OUT * OUT * 4];
  let mut rgba_f32 = std::vec![0.0f32; OUT * OUT * 4];
  let mut rgb_f16 = std::vec![half::f16::ZERO; OUT * OUT * 3];
  let mut rgba_f16 = std::vec![half::f16::ZERO; OUT * OUT * 4];
  let mut luma = std::vec![0u8; OUT * OUT];
  let mut luma_u16 = std::vec![0u16; OUT * OUT];
  let mut h = std::vec![0u8; OUT * OUT];
  let mut s_ = std::vec![0u8; OUT * OUT];
  let mut v_ = std::vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Gbrapf32, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
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
    gbrapf32_to(&src, &mut sink).unwrap();
  }

  // Reference: the full-res direct sink over the (exact) binned RGBA f32,
  // split back into LE-wire G/B/R/A planes exactly as the source arrived.
  let (bg, bb, br, ba) = planes_from_packed_rgba(&rgba_f32, OUT * OUT);
  let (bgw, bbw, brw, baw) = (
    as_le_f32(&bg),
    as_le_f32(&bb),
    as_le_f32(&br),
    as_le_f32(&ba),
  );
  let mut ref_rgb = std::vec![0u8; OUT * OUT * 3];
  let mut ref_rgb_u16 = std::vec![0u16; OUT * OUT * 3];
  let mut ref_rgb_f32 = std::vec![0.0f32; OUT * OUT * 3];
  let mut ref_rgba = std::vec![0u8; OUT * OUT * 4];
  let mut ref_rgba_u16 = std::vec![0u16; OUT * OUT * 4];
  let mut ref_rgb_f16 = std::vec![half::f16::ZERO; OUT * OUT * 3];
  let mut ref_rgba_f16 = std::vec![half::f16::ZERO; OUT * OUT * 4];
  let mut ref_luma = std::vec![0u8; OUT * OUT];
  let mut ref_luma_u16 = std::vec![0u16; OUT * OUT];
  let mut ref_h = std::vec![0u8; OUT * OUT];
  let mut ref_s = std::vec![0u8; OUT * OUT];
  let mut ref_v = std::vec![0u8; OUT * OUT];
  {
    let binned = frame(&bgw, &bbw, &brw, &baw, OUT, OUT);
    let mut sink = MixedSinker::<Gbrapf32>::new(OUT, OUT)
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
      .with_rgb_f16(&mut ref_rgb_f16)
      .unwrap()
      .with_rgba_f16(&mut ref_rgba_f16)
      .unwrap()
      .with_luma(&mut ref_luma)
      .unwrap()
      .with_luma_u16(&mut ref_luma_u16)
      .unwrap()
      .with_hsv(&mut ref_h, &mut ref_s, &mut ref_v)
      .unwrap();
    gbrapf32_to(&binned, &mut sink).unwrap();
  }

  assert_eq!(rgb, ref_rgb, "rgb");
  assert_eq!(rgb_u16, ref_rgb_u16, "rgb_u16");
  assert_eq!(
    rgb_f32, ref_rgb_f32,
    "rgb_f32 (lossless drop-α, full parity)"
  );
  assert_eq!(rgba, ref_rgba, "rgba");
  assert_eq!(rgba_u16, ref_rgba_u16, "rgba_u16");
  assert_eq!(rgb_f16, ref_rgb_f16, "rgb_f16");
  assert_eq!(rgba_f16, ref_rgba_f16, "rgba_f16");
  assert_eq!(luma, ref_luma, "luma");
  // luma_u16 on the fused path is the direct path's narrowed (8-bit, in a
  // u16 carrier) value — `gbrpf32_to_luma_u16_row` stages through u8.
  assert_eq!(luma_u16, ref_luma_u16, "luma_u16 (narrowed, full parity)");
  assert_eq!(h, ref_h, "hsv H");
  assert_eq!(s_, ref_s, "hsv S");
  assert_eq!(v_, ref_v, "hsv V");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrapf32_premultiplied_matches_premult_bin_unpremult_oracle() {
  // Premultiplied mode: premultiply in f32 (R' = R*A), bin, un-premultiply
  // (R = mean(R*A)/mean(A)). The oracle premultiplies the host planes, bins
  // exactly, un-premultiplies, then drives the direct path over the result.
  let (g, b, r, a) = gbra_planes();
  let (gw, bw, rw, aw) = (as_le_f32(&g), as_le_f32(&b), as_le_f32(&r), as_le_f32(&a));
  let src = frame(&gw, &bw, &rw, &aw, SRC, SRC);

  let mut rgba_f32 = std::vec![0.0f32; OUT * OUT * 4];
  let mut rgb_f32 = std::vec![0.0f32; OUT * OUT * 3];
  {
    let mut sink =
      MixedSinker::<Gbrapf32, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_alpha_mode(AlphaMode::Premultiplied)
        .with_rgba_f32(&mut rgba_f32)
        .unwrap()
        .with_rgb_f32(&mut rgb_f32)
        .unwrap();
    gbrapf32_to(&src, &mut sink).unwrap();
  }

  // Oracle: bin premultiplied color & raw α per output block, then resolve.
  for oy in 0..OUT {
    for ox in 0..OUT {
      let mean_a = block_mean_plane(&a, ox, oy);
      // mean(channel * α) over the 2x2 block.
      let mut prem = [0.0f32; 3]; // R', G', B'
      let chans = [&r, &g, &b];
      for (c, plane) in chans.iter().enumerate() {
        let mut acc = 0.0f64;
        for dy in 0..2 {
          for dx in 0..2 {
            let idx = (oy * 2 + dy) * SRC + ox * 2 + dx;
            acc += (plane[idx] * a[idx]) as f64;
          }
        }
        prem[c] = (acc / 4.0) as f32;
      }
      let straight = |pm: f32| if mean_a == 0.0 { 0.0 } else { pm / mean_a };
      let base = (oy * OUT + ox) * 4;
      assert_eq!(rgba_f32[base], straight(prem[0]), "premult R ({ox},{oy})");
      assert_eq!(
        rgba_f32[base + 1],
        straight(prem[1]),
        "premult G ({ox},{oy})"
      );
      assert_eq!(
        rgba_f32[base + 2],
        straight(prem[2]),
        "premult B ({ox},{oy})"
      );
      assert_eq!(rgba_f32[base + 3], mean_a, "premult A ({ox},{oy})");
      // rgb_f32 is the un-premultiplied color with α dropped.
      let rbase = (oy * OUT + ox) * 3;
      assert_eq!(
        rgb_f32[rbase],
        straight(prem[0]),
        "premult rgb R ({ox},{oy})"
      );
      assert_eq!(
        rgb_f32[rbase + 1],
        straight(prem[1]),
        "premult rgb G ({ox},{oy})"
      );
      assert_eq!(
        rgb_f32[rbase + 2],
        straight(prem[2]),
        "premult rgb B ({ox},{oy})"
      );
    }
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrapf32_premultiplied_transparent_block_does_not_bleed() {
  // A fully-transparent (α = 0) 2x2 block with arbitrary stored color must
  // produce RGB = 0 under premultiplied resampling (no color bleed).
  let (mut g, mut b, mut r, mut a) = gbra_planes();
  for off in [(0usize, 0usize), (1, 0), (0, 1), (1, 1)] {
    let i = off.1 * SRC + off.0;
    r[i] = 999.0;
    g[i] = -999.0;
    b[i] = 999.0;
    a[i] = 0.0;
  }
  let (gw, bw, rw, aw) = (as_le_f32(&g), as_le_f32(&b), as_le_f32(&r), as_le_f32(&a));
  let src = frame(&gw, &bw, &rw, &aw, SRC, SRC);

  let mut rgba_f32 = std::vec![0.0f32; OUT * OUT * 4];
  {
    let mut sink =
      MixedSinker::<Gbrapf32, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_alpha_mode(AlphaMode::Premultiplied)
        .with_rgba_f32(&mut rgba_f32)
        .unwrap();
    gbrapf32_to(&src, &mut sink).unwrap();
  }
  assert_eq!(
    &rgba_f32[..4],
    &[0.0, 0.0, 0.0, 0.0],
    "transparent block bled color"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrapf32_rgb_only_straight_drops_alpha_via_3channel_path() {
  // A straight sink with only RGB-flavoured outputs (no rgba*) must drop α
  // and match a direct Gbrpf32-equivalent conversion of the binned RGB —
  // identical to the 3-channel `resample_gbrpf32` oracle.
  let (g, b, r, a) = gbra_planes();
  let (gw, bw, rw, aw) = (as_le_f32(&g), as_le_f32(&b), as_le_f32(&r), as_le_f32(&a));
  let src = frame(&gw, &bw, &rw, &aw, SRC, SRC);

  let mut rgb_f32 = std::vec![0.0f32; OUT * OUT * 3];
  let mut luma = std::vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Gbrapf32, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb_f32(&mut rgb_f32)
        .unwrap()
        .with_luma(&mut luma)
        .unwrap();
    gbrapf32_to(&src, &mut sink).unwrap();
  }
  // rgb_f32 is the exact RGB block mean (α dropped, integer-valued samples).
  for oy in 0..OUT {
    for ox in 0..OUT {
      let base = (oy * OUT + ox) * 3;
      assert_eq!(rgb_f32[base], block_mean_plane(&r, ox, oy), "R ({ox},{oy})");
      assert_eq!(
        rgb_f32[base + 1],
        block_mean_plane(&g, ox, oy),
        "G ({ox},{oy})"
      );
      assert_eq!(
        rgb_f32[base + 2],
        block_mean_plane(&b, ox, oy),
        "B ({ox},{oy})"
      );
    }
  }
  // luma matches a direct Gbrpf32 conversion of the binned RGB.
  let (bg, bb, br) = planes_from_packed_rgb(&rgb_f32, OUT * OUT);
  let (bgw, bbw, brw) = (as_le_f32(&bg), as_le_f32(&bb), as_le_f32(&br));
  let mut luma_ref = std::vec![0u8; OUT * OUT];
  {
    let binned = crate::frame::Gbrpf32Frame::try_new(
      &bgw, &bbw, &brw, OUT as u32, OUT as u32, OUT as u32, OUT as u32, OUT as u32,
    )
    .unwrap();
    let mut sink = MixedSinker::<crate::source::Gbrpf32>::new(OUT, OUT)
      .with_luma(&mut luma_ref)
      .unwrap();
    crate::source::gbrpf32_to(&binned, &mut sink).unwrap();
  }
  assert_eq!(luma, luma_ref, "luma (drop-α == 3-channel path)");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrapf32_fractional_ratio_matches_convert_then_bin() {
  // Fractional downscale ratio (8 -> 3, non-integer). The f32 stream is a
  // ±tolerance bin (#143 f32 contract, not 0-ULP), so assert the *fused
  // identity*: the resampled output equals a direct conversion of the
  // (lossless) binned RGBA, NOT a hand-rolled mean.
  const FOUT: usize = 3;
  let (g, b, r, a) = gbra_planes();
  let (gw, bw, rw, aw) = (as_le_f32(&g), as_le_f32(&b), as_le_f32(&r), as_le_f32(&a));
  let src = frame(&gw, &bw, &rw, &aw, SRC, SRC);

  let mut rgba_u16 = std::vec![0u16; FOUT * FOUT * 4];
  let mut rgba_f32 = std::vec![0.0f32; FOUT * FOUT * 4];
  {
    let mut sink = MixedSinker::<Gbrapf32, AreaResampler>::with_resampler(
      SRC,
      SRC,
      AreaResampler::to(FOUT, FOUT),
    )
    .unwrap()
    .with_rgba_u16(&mut rgba_u16)
    .unwrap()
    .with_rgba_f32(&mut rgba_f32)
    .unwrap();
    gbrapf32_to(&src, &mut sink).unwrap();
  }

  // Oracle: direct convert of the binned (lossless) RGBA f32.
  let (bg, bb, br, ba) = planes_from_packed_rgba(&rgba_f32, FOUT * FOUT);
  let (bgw, bbw, brw, baw) = (
    as_le_f32(&bg),
    as_le_f32(&bb),
    as_le_f32(&br),
    as_le_f32(&ba),
  );
  let mut ref_rgba_u16 = std::vec![0u16; FOUT * FOUT * 4];
  {
    let binned = frame(&bgw, &bbw, &brw, &baw, FOUT, FOUT);
    let mut sink = MixedSinker::<Gbrapf32>::new(FOUT, FOUT)
      .with_rgba_u16(&mut ref_rgba_u16)
      .unwrap();
    gbrapf32_to(&binned, &mut sink).unwrap();
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
fn gbrapf32_le_be_parity() {
  // The same host values through LE and BE wire frames must produce
  // byte-identical resampled output (every flavour).
  let (g, b, r, a) = gbra_planes();

  let render = |be: bool| {
    let (gw, bw, rw, aw) = if be {
      (as_be_f32(&g), as_be_f32(&b), as_be_f32(&r), as_be_f32(&a))
    } else {
      (as_le_f32(&g), as_le_f32(&b), as_le_f32(&r), as_le_f32(&a))
    };
    let mut rgba = std::vec![0u8; OUT * OUT * 4];
    let mut rgba_u16 = std::vec![0u16; OUT * OUT * 4];
    let mut rgba_f32 = std::vec![0.0f32; OUT * OUT * 4];
    let mut rgba_f16 = std::vec![half::f16::ZERO; OUT * OUT * 4];
    let mut luma_u16 = std::vec![0u16; OUT * OUT];
    if be {
      let src = crate::frame::Gbrapf32BeFrame::try_new(
        &gw, &bw, &rw, &aw, SRC as u32, SRC as u32, SRC as u32, SRC as u32, SRC as u32, SRC as u32,
      )
      .unwrap();
      let mut sink = MixedSinker::<crate::source::Gbrapf32<true>, AreaResampler>::with_resampler(
        SRC,
        SRC,
        AreaResampler::to(OUT, OUT),
      )
      .unwrap()
      .with_rgba(&mut rgba)
      .unwrap()
      .with_rgba_u16(&mut rgba_u16)
      .unwrap()
      .with_rgba_f32(&mut rgba_f32)
      .unwrap()
      .with_rgba_f16(&mut rgba_f16)
      .unwrap()
      .with_luma_u16(&mut luma_u16)
      .unwrap();
      crate::source::gbrapf32_to_endian(&src, &mut sink).unwrap();
    } else {
      let src = frame(&gw, &bw, &rw, &aw, SRC, SRC);
      let mut sink = MixedSinker::<Gbrapf32, AreaResampler>::with_resampler(
        SRC,
        SRC,
        AreaResampler::to(OUT, OUT),
      )
      .unwrap()
      .with_rgba(&mut rgba)
      .unwrap()
      .with_rgba_u16(&mut rgba_u16)
      .unwrap()
      .with_rgba_f32(&mut rgba_f32)
      .unwrap()
      .with_rgba_f16(&mut rgba_f16)
      .unwrap()
      .with_luma_u16(&mut luma_u16)
      .unwrap();
      gbrapf32_to(&src, &mut sink).unwrap();
    }
    (rgba, rgba_u16, rgba_f32, rgba_f16, luma_u16)
  };
  assert_eq!(render(false), render(true), "LE/BE outputs diverge");
}

#[test]
fn gbrapf32_default_alpha_mode_is_straight() {
  let sink = MixedSinker::<Gbrapf32>::new(SRC, SRC);
  assert_eq!(sink.alpha_mode(), AlphaMode::Straight);
  assert!(sink.alpha_mode().is_straight());
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrapf32_identity_plan_matches_new_sink() {
  let (g, b, r, a) = gbra_planes();
  let (gw, bw, rw, aw) = (as_le_f32(&g), as_le_f32(&b), as_le_f32(&r), as_le_f32(&a));
  let src = frame(&gw, &bw, &rw, &aw, SRC, SRC);

  let mut direct = std::vec![0.0f32; SRC * SRC * 4];
  {
    let mut sink = MixedSinker::<Gbrapf32>::new(SRC, SRC)
      .with_rgba_f32(&mut direct)
      .unwrap();
    gbrapf32_to(&src, &mut sink).unwrap();
  }
  let mut via_area = std::vec![0.0f32; SRC * SRC * 4];
  {
    let mut sink =
      MixedSinker::<Gbrapf32, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(SRC, SRC))
        .unwrap()
        .with_rgba_f32(&mut via_area)
        .unwrap();
    gbrapf32_to(&src, &mut sink).unwrap();
  }
  assert_eq!(direct, via_area, "identity-plan resample == direct sink");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrapf32_no_output_sink_is_a_noop() {
  let (g, b, r, a) = gbra_planes();
  let (gw, bw, rw, aw) = (as_le_f32(&g), as_le_f32(&b), as_le_f32(&r), as_le_f32(&a));
  let src = frame(&gw, &bw, &rw, &aw, SRC, SRC);

  let mut sink =
    MixedSinker::<Gbrapf32, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap();
  gbrapf32_to(&src, &mut sink).unwrap();
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
fn gbrapf32_resample_reuses_stream_across_frames() {
  // begin_frame resets the f32 area stream + frozen output/alpha snapshots,
  // so frame 2's row 0 is accepted and the output reflects frame 2's input.
  let (g1, b1, r1, a1) = gbra_planes();
  let g2: Vec<f32> = g1.iter().map(|&v| -v).collect();
  let b2: Vec<f32> = b1.iter().map(|&v| -v).collect();
  let r2: Vec<f32> = r1.iter().map(|&v| -v).collect();
  let a2 = a1.clone();
  let (g1w, b1w, r1w, a1w) = (
    as_le_f32(&g1),
    as_le_f32(&b1),
    as_le_f32(&r1),
    as_le_f32(&a1),
  );
  let (g2w, b2w, r2w, a2w) = (
    as_le_f32(&g2),
    as_le_f32(&b2),
    as_le_f32(&r2),
    as_le_f32(&a2),
  );

  let mut out = std::vec![0.0f32; OUT * OUT * 4];
  {
    let mut sink =
      MixedSinker::<Gbrapf32, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgba_f32(&mut out)
        .unwrap();
    gbrapf32_to(&frame(&g1w, &b1w, &r1w, &a1w, SRC, SRC), &mut sink).unwrap();
    gbrapf32_to(&frame(&g2w, &b2w, &r2w, &a2w, SRC, SRC), &mut sink).unwrap();
  }
  for oy in 0..OUT {
    for ox in 0..OUT {
      let base = (oy * OUT + ox) * 4;
      assert_eq!(out[base], block_mean_plane(&r2, ox, oy), "frame2 R");
      assert_eq!(out[base + 1], block_mean_plane(&g2, ox, oy), "frame2 G");
      assert_eq!(out[base + 2], block_mean_plane(&b2, ox, oy), "frame2 B");
      assert_eq!(out[base + 3], block_mean_plane(&a2, ox, oy), "frame2 A");
    }
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrapf32_accepts_alpha_mode_change_across_frames() {
  // begin_frame (re-run by the walker each frame) re-arms the frozen mode,
  // so frame 2 may flip to Premultiplied without a false
  // ResampleOutputsChanged.
  let (g, b, r, a) = gbra_planes();
  let (gw, bw, rw, aw) = (as_le_f32(&g), as_le_f32(&b), as_le_f32(&r), as_le_f32(&a));
  let src = frame(&gw, &bw, &rw, &aw, SRC, SRC);

  let mut rgba_f32 = std::vec![0.0f32; OUT * OUT * 4];
  {
    let mut sink =
      MixedSinker::<Gbrapf32, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgba_f32(&mut rgba_f32)
        .unwrap();
    gbrapf32_to(&src, &mut sink).unwrap();
    sink.set_alpha_mode(AlphaMode::Premultiplied);
    gbrapf32_to(&src, &mut sink).expect("a fresh frame must accept a different alpha mode");
  }
  // Frame 2 (premult) oracle for block (0,0).
  let mean_a = block_mean_plane(&a, 0, 0);
  let straight = |plane: &[f32]| {
    let mut acc = 0.0f64;
    for dy in 0..2 {
      for dx in 0..2 {
        let idx = dy * SRC + dx;
        acc += (plane[idx] * a[idx]) as f64;
      }
    }
    let pm = (acc / 4.0) as f32;
    if mean_a == 0.0 { 0.0 } else { pm / mean_a }
  };
  assert_eq!(rgba_f32[0], straight(&r), "premult frame2 R");
  assert_eq!(rgba_f32[3], mean_a, "premult frame2 A");
}
