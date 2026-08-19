//! Fused-downscale coverage for the 8-bit palette-indexed source `Pal8`.
//!
//! Averaging palette *indices* is meaningless (index 5 and index 200 averaged
//! is an unrelated color), so the only sensible area-resample is to expand
//! each pixel to its palette color and bin THAT. `Pal8` therefore routes like
//! any RGBA-producing format: each index is looked up to its `[R, G, B, A]`
//! (real per-entry alpha, FFmpeg `[B, G, R, A]` palette order) and the
//! canonical RGBA is fed through the 4-channel packed-RGBA-style tail
//! (`pal8_rgba_resample`). A resampled frame is byte-identical to a direct
//! full-res `Pal8` -> RGBA conversion followed by an area-bin of that color.
//!
//! The oracles bin a DIRECT full-res `Pal8` conversion. Crucially:
//! - **color-average, not index-average**: a palette where two adjacent
//!   pixels' indices map to colors whose mean differs from the indices' mean
//!   pins that the bin happens AFTER the palette lookup
//!   (`color_average_not_index_average`);
//! - the per-output derivations (rgb / rgb_u16 / rgba_u16 / luma / luma_u16 /
//!   hsv) are validated against a direct `Pal8` conversion of the binned
//!   color, so luma uses the SAME Q8 BT.709 coefficients and the u16 outputs
//!   the SAME `(x << 8) | x` full-range widening the identity path uses (NOT
//!   the matrix-luma / zero-extension the `Ya8` tail uses);
//! - straight alpha is an area mean (not forced opaque); premultiplied bins
//!   premultiplied color and un-premultiplies (transparent pixels never
//!   bleed); the alpha mode is frozen per frame; A == 0 -> RGB == 0;
//! - integer-ratio (2:1) and fractional-ratio (8->3) downscales match.

use crate::{
  PixelSink,
  frame::Pal8Frame,
  resample::{AreaResampler, ResampleError},
  sinker::{AlphaMode, MixedSinker, MixedSinkerError},
  source::{Pal8, Pal8Row, pal8_to},
};

const SRC: usize = 8;
const OUT: usize = 4;

/// Local LCG (the shared `pseudo_random_u8` helper is gated to feature sets
/// that exclude `mono`, so a `mono`-solo build needs its own).
fn fill_pseudo_random(buf: &mut [u8], seed: u32) {
  let mut state = seed;
  for b in buf {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *b = (state >> 16) as u8;
  }
}

/// A palette whose every entry maps to a DISTINCT, non-trivial `[R, G, B, A]`
/// — so the expand-then-bin color genuinely depends on the lookup (an
/// index-average would land on a wholly different color).
fn varied_palette(seed: u32) -> [[u8; 4]; 256] {
  let mut p = [[0u8; 4]; 256];
  for (i, entry) in p.iter_mut().enumerate() {
    let i = i as u32;
    // FFmpeg PAL8 byte order is [B, G, R, A].
    entry[0] = ((i.wrapping_mul(97).wrapping_add(seed)) ^ 0x5A) as u8; // B
    entry[1] = ((i.wrapping_mul(57).wrapping_add(seed)) ^ 0x3C) as u8; // G
    entry[2] = ((i.wrapping_mul(193).wrapping_add(seed)) ^ 0xA5) as u8; // R
    entry[3] = ((i.wrapping_mul(151).wrapping_add(seed)) ^ 0xF0) as u8; // A
  }
  p
}

/// A pseudo-random `SRC * SRC` index plane.
fn index_plane(seed: u32) -> Vec<u8> {
  let mut buf = std::vec![0u8; SRC * SRC];
  fill_pseudo_random(&mut buf, seed);
  buf
}

/// Full-resolution canonical RGBA of the source — a DIRECT (identity) `Pal8`
/// conversion (palette lookup, `[B, G, R, A]` -> `[R, G, B, A]`). The oracles
/// bin / premultiply this.
fn direct_rgba(indices: &[u8], palette: &[[u8; 4]; 256]) -> Vec<u8> {
  let frame = Pal8Frame::new(indices, palette, SRC as u32, SRC as u32, SRC as u32);
  let mut rgba = std::vec![0u8; SRC * SRC * 4];
  let mut sink = MixedSinker::<Pal8>::new(SRC, SRC)
    .with_rgba(&mut rgba)
    .unwrap();
  pal8_to(&frame, &mut sink).unwrap();
  rgba
}

/// Round-half-up 2x2 block mean of a canonical RGBA plane (every channel,
/// alpha included) — the integer-ratio area-downscale contract.
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

/// Premultiply one canonical RGBA plane in place — `round(c * a / 255)` per
/// color channel, alpha untouched.
fn premultiply(plane: &mut [u8]) {
  for px in plane.chunks_exact_mut(4) {
    let a = px[3] as u32;
    for c in &mut px[..3] {
      *c = ((*c as u32 * a + 127) / 255) as u8;
    }
  }
}

/// Un-premultiply one binned canonical RGBA plane — `round(c' * 255 / a)`
/// clamped to 255, color 0 when `a == 0`; alpha copied.
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

/// Drop alpha from a canonical RGBA plane -> packed RGB.
fn drop_alpha(rgba: &[u8]) -> Vec<u8> {
  let mut out = std::vec![0u8; rgba.len() / 4 * 3];
  for (o, i) in out.chunks_exact_mut(3).zip(rgba.chunks_exact(4)) {
    o.copy_from_slice(&i[..3]);
  }
  out
}

/// Full-range `(x << 8) | x` widening of a u8 plane to u16 — the `Pal8` u16
/// output convention (NOT the `Ya8` zero-extension).
fn expand_u16(plane: &[u8]) -> Vec<u16> {
  plane
    .iter()
    .map(|&v| {
      let v = v as u16;
      (v << 8) | v
    })
    .collect()
}

/// Builds a 1-row `Pal8Frame` (one palette entry per pixel) that reproduces a
/// canonical RGBA plane exactly, so a DIRECT `Pal8` conversion of it runs the
/// identity-path kernels (Q8 BT.709 luma, `(x << 8) | x` luma_u16, OpenCV HSV)
/// byte-for-byte over the binned color — the parity source of truth for the
/// derived outputs. Returns `(indices, palette)`.
fn per_pixel_palette(binned_rgba: &[u8]) -> (Vec<u8>, [[u8; 4]; 256]) {
  let n = binned_rgba.len() / 4;
  let mut palette = [[0u8; 4]; 256];
  let mut indices = std::vec![0u8; n];
  for (i, px) in binned_rgba.chunks_exact(4).enumerate() {
    // px = [R, G, B, A]; palette entry is [B, G, R, A].
    palette[i] = [px[2], px[1], px[0], px[3]];
    indices[i] = i as u8;
  }
  (indices, palette)
}

/// Direct `Pal8` luma / luma_u16 of a binned canonical RGBA plane.
fn direct_luma_of_binned(binned_rgba: &[u8]) -> (Vec<u8>, Vec<u16>) {
  let n = binned_rgba.len() / 4;
  let (indices, palette) = per_pixel_palette(binned_rgba);
  let frame = Pal8Frame::new(&indices, &palette, n as u32, 1, n as u32);
  let mut luma = std::vec![0u8; n];
  let mut lu16 = std::vec![0u16; n];
  let mut sink = MixedSinker::<Pal8>::new(n, 1)
    .with_luma(&mut luma)
    .unwrap()
    .with_luma_u16(&mut lu16)
    .unwrap();
  pal8_to(&frame, &mut sink).unwrap();
  (luma, lu16)
}

/// Direct `Pal8` HSV of a binned canonical RGBA plane.
fn direct_hsv_of_binned(binned_rgba: &[u8]) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let n = binned_rgba.len() / 4;
  let (indices, palette) = per_pixel_palette(binned_rgba);
  let frame = Pal8Frame::new(&indices, &palette, n as u32, 1, n as u32);
  let mut h = std::vec![0u8; n];
  let mut s = std::vec![0u8; n];
  let mut v = std::vec![0u8; n];
  let mut sink = MixedSinker::<Pal8>::new(n, 1)
    .with_hsv(&mut h, &mut s, &mut v)
    .unwrap();
  pal8_to(&frame, &mut sink).unwrap();
  (h, s, v)
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn straight_rgba_is_block_mean_of_direct() {
  let palette = varied_palette(0x11);
  let indices = index_plane(0x51A1);
  let frame = Pal8Frame::new(&indices, &palette, SRC as u32, SRC as u32, SRC as u32);

  let mut rgba = std::vec![0u8; OUT * OUT * 4];
  {
    let mut sink =
      MixedSinker::<Pal8, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap();
    pal8_to(&frame, &mut sink).unwrap();
  }
  assert_eq!(rgba, block_mean_rgba(&direct_rgba(&indices, &palette)));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn color_average_not_index_average() {
  // THE defining property of routing Pal8: the bin happens AFTER the palette
  // lookup. Construct a palette where two adjacent indices map to colors whose
  // average differs sharply from what the *indices* would average to, then
  // assert the binned result is the COLOR-average.
  //
  // Index 0 -> R=0,   G=0,   B=0
  // Index 255 -> R=254, G=254, B=254
  // Index 100 (the index-average of 0 and 200... but we use 0/255 indices) is
  // an unrelated mid-gray-ish entry, so an index-average path would read a
  // wholly different palette entry than the color-average 127.
  let mut palette = [[0u8; 4]; 256];
  for entry in palette.iter_mut() {
    *entry = [0, 0, 0, 255];
  }
  // [B, G, R, A]
  palette[0] = [0, 0, 0, 255]; // black
  palette[255] = [254, 254, 254, 255]; // near-white
  // A booby-trap entry at the index that an index-average of 0 and 254 would
  // land on (127): a saturated red. If anything averaged indices then looked
  // up, red would leak in.
  palette[127] = [0, 0, 255, 255]; // pure red (R=255)

  // Every 2x2 block is two index-0 and two index-255 pixels (checkerboard by
  // column), so the COLOR mean per block is ((0+0+254+254)/4) = 127 on each
  // of R/G/B. The INDEX mean would be (0+0+255+255)/4 = 127 -> palette[127]
  // -> pure red (R=255, G=0, B=0). These differ, so the test discriminates.
  let mut indices = std::vec![0u8; SRC * SRC];
  for (i, ix) in indices.iter_mut().enumerate() {
    let col = i % SRC;
    *ix = if col % 2 == 1 { 255 } else { 0 };
  }
  let frame = Pal8Frame::new(&indices, &palette, SRC as u32, SRC as u32, SRC as u32);

  let mut rgba = std::vec![0u8; OUT * OUT * 4];
  {
    let mut sink =
      MixedSinker::<Pal8, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap();
    pal8_to(&frame, &mut sink).unwrap();
  }

  // Color-average oracle (expand-then-bin): every output pixel is gray 127.
  let oracle = block_mean_rgba(&direct_rgba(&indices, &palette));
  assert_eq!(rgba, oracle, "binned color != expand-then-bin oracle");
  for px in rgba.chunks_exact(4) {
    assert_eq!(
      px,
      [127, 127, 127, 255],
      "expected the COLOR-average gray 127, got {px:?}"
    );
  }
  // And it must NOT be the index-average lookup (pure red R=255). Guard the
  // fixture so the discriminator is real.
  assert_eq!(palette[127], [0, 0, 255, 255], "fixture: index-avg entry");
  assert!(
    rgba.chunks_exact(4).all(|px| px[0] != 255 || px[1] != 0),
    "binned result is the INDEX-average (pure red) — bin-then-expand bug"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn straight_alpha_is_averaged_not_forced_opaque() {
  // Palette alpha varies across entries, so the binned alpha is a real area
  // mean (not forced to 0xFF).
  let palette = varied_palette(0x7E);
  let indices = index_plane(0x9E37);
  let frame = Pal8Frame::new(&indices, &palette, SRC as u32, SRC as u32, SRC as u32);

  let mut rgba = std::vec![0u8; OUT * OUT * 4];
  {
    let mut sink =
      MixedSinker::<Pal8, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap();
    pal8_to(&frame, &mut sink).unwrap();
  }
  let oracle = block_mean_rgba(&direct_rgba(&indices, &palette));
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
fn straight_all_outputs_derive_from_binned_color() {
  // Every output flavour at once: rgba is the binned color, rgb drops alpha,
  // rgb_u16 / rgba_u16 are its `(x << 8) | x` widening, and luma / luma_u16 /
  // hsv match a direct full-res `Pal8` conversion of the binned color.
  let palette = varied_palette(0x42);
  let indices = index_plane(0xBEEF);
  let frame = Pal8Frame::new(&indices, &palette, SRC as u32, SRC as u32, SRC as u32);

  let mut rgb = std::vec![0u8; OUT * OUT * 3];
  let mut rgba = std::vec![0u8; OUT * OUT * 4];
  let mut rgb_u16 = std::vec![0u16; OUT * OUT * 3];
  let mut rgba_u16 = std::vec![0u16; OUT * OUT * 4];
  let mut luma = std::vec![0u8; OUT * OUT];
  let mut lu16 = std::vec![0u16; OUT * OUT];
  let mut h = std::vec![0u8; OUT * OUT];
  let mut s = std::vec![0u8; OUT * OUT];
  let mut v = std::vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Pal8, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb(&mut rgb)
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap()
        .with_rgb_u16(&mut rgb_u16)
        .unwrap()
        .with_rgba_u16(&mut rgba_u16)
        .unwrap()
        .with_luma(&mut luma)
        .unwrap()
        .with_luma_u16(&mut lu16)
        .unwrap()
        .with_hsv(&mut h, &mut s, &mut v)
        .unwrap();
    pal8_to(&frame, &mut sink).unwrap();
  }

  let binned = block_mean_rgba(&direct_rgba(&indices, &palette));
  assert_eq!(rgba, binned, "rgba == block mean");
  assert_eq!(rgb, drop_alpha(&binned), "rgb == drop-alpha(binned)");
  assert_eq!(
    rgba_u16,
    expand_u16(&binned),
    "rgba_u16 == (x<<8)|x of binned"
  );
  assert_eq!(
    rgb_u16,
    expand_u16(&drop_alpha(&binned)),
    "rgb_u16 == (x<<8)|x of drop-alpha(binned)"
  );

  let (luma_ref, lu16_ref) = direct_luma_of_binned(&binned);
  let (h_ref, s_ref, v_ref) = direct_hsv_of_binned(&binned);
  assert_eq!(luma, luma_ref, "luma (Q8 BT.709 of binned RGB)");
  assert_eq!(lu16, lu16_ref, "luma_u16 ((y<<8)|y of binned RGB)");
  assert_eq!(h, h_ref, "hsv H");
  assert_eq!(s, s_ref, "hsv S");
  assert_eq!(v, v_ref, "hsv V");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn premultiplied_matches_premult_bin_unpremult_oracle() {
  let palette = varied_palette(0x33);
  let indices = index_plane(0x1234);
  let frame = Pal8Frame::new(&indices, &palette, SRC as u32, SRC as u32, SRC as u32);

  let mut rgba = std::vec![0u8; OUT * OUT * 4];
  let mut rgb = std::vec![0u8; OUT * OUT * 3];
  let mut rgba_u16 = std::vec![0u16; OUT * OUT * 4];
  let mut luma = std::vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Pal8, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_alpha_mode(AlphaMode::Premultiplied)
        .with_rgba(&mut rgba)
        .unwrap()
        .with_rgb(&mut rgb)
        .unwrap()
        .with_rgba_u16(&mut rgba_u16)
        .unwrap()
        .with_luma(&mut luma)
        .unwrap();
    pal8_to(&frame, &mut sink).unwrap();
  }

  let mut pm = direct_rgba(&indices, &palette);
  premultiply(&mut pm);
  let binned = block_mean_rgba(&pm);
  let oracle = unpremultiply(&binned);
  assert_eq!(rgba, oracle, "premult rgba");
  assert_eq!(rgb, drop_alpha(&oracle), "premult rgb");
  assert_eq!(rgba_u16, expand_u16(&oracle), "premult rgba_u16");
  // luma is Q8 BT.709 over the un-premultiplied straight RGB — the direct
  // `Pal8` luma of the un-premultiplied binned color.
  let (luma_ref, _) = direct_luma_of_binned(&oracle);
  assert_eq!(luma, luma_ref, "premult luma (Q8 of un-premult color)");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn premultiplied_transparent_block_does_not_bleed() {
  let mut palette = varied_palette(0xAB);
  let mut indices = index_plane(0xABCD);
  // Force the top-left 2x2 block to a fully-transparent palette entry whose
  // stored color is bright — un-premultiplied straight color must be 0.
  palette[7] = [200, 210, 220, 0]; // [B, G, R, A=0] — opaque-bright, alpha 0
  for off in [(0, 0), (1, 0), (0, 1), (1, 1)] {
    indices[off.1 * SRC + off.0] = 7;
  }
  let frame = Pal8Frame::new(&indices, &palette, SRC as u32, SRC as u32, SRC as u32);

  let mut rgba = std::vec![0u8; OUT * OUT * 4];
  {
    let mut sink =
      MixedSinker::<Pal8, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_alpha_mode(AlphaMode::Premultiplied)
        .with_rgba(&mut rgba)
        .unwrap();
    pal8_to(&frame, &mut sink).unwrap();
  }
  assert_eq!(&rgba[..4], &[0, 0, 0, 0], "transparent block bled color");

  let mut pm = direct_rgba(&indices, &palette);
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
  let palette = varied_palette(0x77);
  let indices = index_plane(0x77AA);
  let render = |mode: AlphaMode| {
    let frame = Pal8Frame::new(&indices, &palette, SRC as u32, SRC as u32, SRC as u32);
    let mut rgba = std::vec![0u8; OUT * OUT * 4];
    let mut sink =
      MixedSinker::<Pal8, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_alpha_mode(mode)
        .with_rgba(&mut rgba)
        .unwrap();
    pal8_to(&frame, &mut sink).unwrap();
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
  let sink = MixedSinker::<Pal8>::new(SRC, SRC);
  assert_eq!(sink.alpha_mode(), AlphaMode::Straight);
  assert!(sink.alpha_mode().is_straight());
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn identity_plan_matches_direct() {
  let palette = varied_palette(0x0F);
  let indices = index_plane(0x0F0F);
  let frame = Pal8Frame::new(&indices, &palette, SRC as u32, SRC as u32, SRC as u32);

  let mut rgba = std::vec![0u8; SRC * SRC * 4];
  {
    let mut sink =
      MixedSinker::<Pal8, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(SRC, SRC))
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap();
    pal8_to(&frame, &mut sink).unwrap();
  }
  assert_eq!(
    rgba,
    direct_rgba(&indices, &palette),
    "identity plan == direct"
  );
}

// The fractional-ratio reference reuses the packed-RGBA 8-bit source as an
// independent area-engine oracle, so it is gated on `rgb` (its frame / walker
// live there); a `mono`-solo build covers fractional ratios via the shared
// `resample_packed_rgba_8bit` suite plus the integer-ratio oracles above.
#[cfg(feature = "rgb")]
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn fractional_ratio_matches_direct_then_bin() {
  // 8 -> 3 fractional downscale: assert the resampled rgba equals a direct
  // full-res convert fed through the SAME AreaResampler at OUT=3 over the
  // canonical RGBA frame (the area engine is the source of truth; this guards
  // the Pal8 decode + routing, not the resampler arithmetic).
  const F: usize = 3;
  let palette = varied_palette(0xAC);
  let indices = index_plane(0xF2AC);
  let frame = Pal8Frame::new(&indices, &palette, SRC as u32, SRC as u32, SRC as u32);

  let mut rgba = std::vec![0u8; F * F * 4];
  {
    let mut sink =
      MixedSinker::<Pal8, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(F, F))
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap();
    pal8_to(&frame, &mut sink).unwrap();
  }

  // Reference: feed the canonical RGBA (the direct full-res Pal8 lookup)
  // through the packed-RGBA 8-bit source at the same plan. That path is
  // already covered/trusted; Pal8 must match it exactly because its decode
  // produces that very row.
  let canonical = direct_rgba(&indices, &palette);
  let mut rgba_ref = std::vec![0u8; F * F * 4];
  {
    use crate::{KernelMatrix, frame::RgbaFrame, source::rgba_to};
    let rsrc = RgbaFrame::new(&canonical, SRC as u32, SRC as u32, (SRC * 4) as u32);
    let mut sink = MixedSinker::<crate::source::Rgba, AreaResampler>::with_resampler(
      SRC,
      SRC,
      AreaResampler::to(F, F),
    )
    .unwrap()
    .with_rgba(&mut rgba_ref)
    .unwrap();
    rgba_to(&rsrc, true, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
  }
  assert_eq!(rgba, rgba_ref, "Pal8 8->3 != packed-RGBA 8->3 of canonical");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn cross_frame_reset_reuses_streams() {
  let palette = varied_palette(0x51);
  let indices = index_plane(0x5151);
  let frame = Pal8Frame::new(&indices, &palette, SRC as u32, SRC as u32, SRC as u32);

  let mut rgba = std::vec![0u8; OUT * OUT * 4];
  {
    let mut sink =
      MixedSinker::<Pal8, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap();
    pal8_to(&frame, &mut sink).unwrap();
    pal8_to(&frame, &mut sink).unwrap();
  }
  assert_eq!(rgba, block_mean_rgba(&direct_rgba(&indices, &palette)));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn accepts_alpha_mode_change_across_frames() {
  let palette = varied_palette(0xB2);
  let indices = index_plane(0xB2B2);
  let frame = Pal8Frame::new(&indices, &palette, SRC as u32, SRC as u32, SRC as u32);

  let mut rgba = std::vec![0u8; OUT * OUT * 4];
  {
    let mut sink =
      MixedSinker::<Pal8, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap();
    pal8_to(&frame, &mut sink).unwrap();
    sink.set_alpha_mode(AlphaMode::Premultiplied);
    pal8_to(&frame, &mut sink).expect("a fresh frame must accept a different alpha mode");
  }
  let mut pm = direct_rgba(&indices, &palette);
  premultiply(&mut pm);
  let oracle = unpremultiply(&block_mean_rgba(&pm));
  assert_eq!(rgba, oracle, "premult frame 2 output");
}

// ---- direct-row freeze / sequencing (Pal8Row is publicly constructible) ----

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn mid_frame_alpha_mode_flip_is_rejected() {
  // Feed row 0 under Straight, flip to Premultiplied, then feed row 1: the
  // frozen mode (snapshotted in begin_frame) must reject the changed mode
  // before any further binning.
  let palette = varied_palette(0x3A);
  let indices = index_plane(0x33AA);
  let mut rgba = std::vec![0u8; OUT * OUT * 4];
  let mut sink =
    MixedSinker::<Pal8, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_rgba(&mut rgba)
      .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  sink
    .process(Pal8Row::for_tests(&indices[..SRC], &palette, 0))
    .unwrap();
  sink.set_alpha_mode(AlphaMode::Premultiplied);
  let err = sink
    .process(Pal8Row::for_tests(&indices[SRC..2 * SRC], &palette, 1))
    .unwrap_err();
  assert!(
    matches!(err, MixedSinkerError::ResampleOutputsChanged(_)),
    "mid-frame alpha flip not rejected: {err:?}"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn out_of_sequence_first_row_is_rejected() {
  // A fresh resampling sink expects row 0 first; feeding row 1 first trips
  // OutOfSequenceRow (before any snapshot is stored).
  let palette = varied_palette(0x4B);
  let indices = index_plane(0x44BB);
  let mut rgba = std::vec![0u8; OUT * OUT * 4];
  let mut sink =
    MixedSinker::<Pal8, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_rgba(&mut rgba)
      .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  let err = sink
    .process(Pal8Row::for_tests(&indices[SRC..2 * SRC], &palette, 1))
    .unwrap_err();
  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
    ),
    "out-of-sequence first row not rejected: {err:?}"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn no_output_sink_is_a_noop() {
  let palette = varied_palette(0x24);
  let indices = index_plane(0x4242);
  let frame = Pal8Frame::new(&indices, &palette, SRC as u32, SRC as u32, SRC as u32);
  let mut sink =
    MixedSinker::<Pal8, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap();
  pal8_to(&frame, &mut sink).unwrap();
}

// RFC #238 tail-collapse atomicity: the drop-alpha RGB scratch (`source_rgb_scratch`,
// grown here because hsv is attached) is grown AFTER the RGBA stream builds into a
// LOCAL and BEFORE the field insert + freeze, so a scratch OOM leaves the stream
// field `None` AND `resample_outputs` uncommitted — a retryable first row. The retry
// respects the frozen alpha mode: `begin_frame` snapshots it once and the replay
// never re-begins, so each retried row still passes the frozen alpha-mode guard.
#[test]
#[cfg(feature = "std")]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn area_first_row_scratch_oom_leaves_stream_and_freeze_uncommitted_for_retry() {
  let palette = varied_palette(0x71);
  let indices = index_plane(0x5C0A);
  let mut rgba = std::vec![0u8; OUT * OUT * 4];
  let mut h = std::vec![0u8; OUT * OUT];
  let mut s = std::vec![0u8; OUT * OUT];
  let mut v = std::vec![0u8; OUT * OUT];
  let mut luma = std::vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Pal8, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap();
    // Attach hsv so the drop-alpha RGB scratch (`source_rgb_scratch`) is grown.
    sink.set_hsv(&mut h, &mut s, &mut v).unwrap();
    sink.begin_frame(SRC as u32, SRC as u32).unwrap();
    // Arm the single-shot source-RGB scratch grow failpoint: the row-0 RGBA stream box
    // builds OK (into a LOCAL), then the drop-alpha RGB scratch grow refuses —
    // surfacing AllocationFailed with BOTH the stream field `None` and the freeze
    // uncommitted (the commit-together atomic shape).
    crate::sinker::mixed::arm_source_rgb_scratch_failure();
    let err = sink
      .process(Pal8Row::for_tests(&indices[..SRC], &palette, 0))
      .unwrap_err();
    assert!(
      matches!(
        err,
        MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
      ),
      "first-row drop-alpha scratch OOM must surface AllocationFailed, got {err:?}"
    );
    // DIRECT stream-rollback proof (the non-vacuous part): the RGBA stream field must
    // be `None` — the fix builds it into a LOCAL and inserts only after the scratch
    // grows. The insert-before-scratch shape would leave it `Some` here, so this
    // assertion FAILS against that shape and PASSES against the fix.
    assert!(
      !sink.rgba_stream_allocated(),
      "a scratch OOM must leave the rgba_stream field None (no partial commit)"
    );
    // DIRECT freeze-rollback proof: the output-set snapshot must NOT be committed. A
    // freeze-before-scratch regression would commit it here even though the stream
    // insertion is still delayed.
    assert!(
      !sink.resample_outputs_frozen(),
      "a first-row scratch OOM must leave resample_outputs uncommitted (no partial commit)"
    );
    // The retry drives the same frame ROW-LEVEL (no re-begin), so the freeze snapshot
    // and the alpha mode both stay frozen — attaching luma (a CHANGED output set) and
    // replaying from row 0 is ACCEPTED (not ResampleOutputsChanged), and each row
    // still passes the frozen alpha-mode guard.
    sink.set_luma(&mut luma).unwrap();
    for r in 0..SRC {
      sink
        .process(Pal8Row::for_tests(
          &indices[r * SRC..(r + 1) * SRC],
          &palette,
          r,
        ))
        .expect("frame replay after a first-row scratch OOM must succeed");
    }
  }
  assert!(
    rgba.iter().any(|&b| b != 0),
    "the recovered frame produced real RGBA output (scratch OOM was recoverable)"
  );
  assert!(
    luma.iter().any(|&b| b != 0),
    "the changed-output retry drove the newly-attached luma"
  );
}
