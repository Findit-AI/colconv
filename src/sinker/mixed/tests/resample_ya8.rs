//! Alpha-aware fused-downscale coverage for the packed 8-bit gray+alpha
//! source (`Ya8`).
//!
//! `Ya8` decodes each packed `[Y, A]` row into the canonical source-width
//! `R, G, B, A` row with `R = G = B = Y` (`ya8_to_rgba_row`) and feeds the
//! **same** 4-channel packed-RGBA resample tail (`packed_rgba_resample`)
//! the `Rgba` / `Bgra` / `Gbrap` sources take — so binning the direct
//! RGBA yields `(binY, binY, binY, binA)`, byte-identical to binning Y once
//! and duplicating. This suite asserts the alpha contract at u8 depth:
//! - straight RGBA is the 2x2 block mean of a direct full-res conversion
//!   (alpha averaged, not forced opaque);
//! - premultiplied bins premultiplied color and un-premultiplies
//!   (transparent pixels never bleed);
//! - rgb / hsv derive from the binned color, and the u16 RGB outputs
//!   zero-extend it;
//! - native-Y luma: `luma` is the binned Y byte and `luma_u16` is its
//!   zero-extension, taken from an INDEPENDENT native-Y area bin (the Y
//!   plane fed through its own single-channel stream), NEVER from the
//!   alpha- or range-affected color. This is byte-exact to the direct
//!   `ya8_to_luma*` kernels for every color matrix, both ranges, AND every
//!   alpha mode — under premultiplied the color collapses to
//!   `mean(Y*A)/mean(A)`, but native Y stays `mean(Y)`. The
//!   `limited_range_*` tests pin range-independence and
//!   `premultiplied_nonuniform_alpha_*` pins alpha-independence (incl. the
//!   `(0,255),(255,0),(0,255),(255,0) -> 128` case);
//! - the alpha mode is frozen per frame and re-armed across frames;
//! - integer-ratio (2:1) and fractional-ratio (8->3) downscales match
//!   their respective direct-then-area-bin references.
//!
//! Unlike `Gbrap`, a `Ya8Row` is publicly constructible, so the mid-frame
//! alpha-mode-freeze and out-of-sequence-row rejections are driven
//! directly here (not only via the shared-tail packed-RGBA suite).

use crate::{
  KernelMatrix, PixelSink,
  frame::Ya8Frame,
  resample::{AreaResampler, ResampleError},
  sinker::{AlphaMode, MixedSinker, MixedSinkerError},
  source::{Ya8, Ya8Row, ya8_to},
};

const SRC: usize = 8;
const OUT: usize = 4;
const FR: bool = true;
/// Limited (studio) range — exercises the native-Y luma path against the
/// range-dependent `rgb_to_luma*` it must NOT use.
const FR_LIMITED: bool = false;
const M: KernelMatrix = KernelMatrix::Bt709;

/// Pseudo-random packed `[Y, A]` plane (`SRC * SRC * 2` bytes); alpha
/// varies (not all-opaque).
fn packed_frame(seed: u32) -> Vec<u8> {
  let mut buf = std::vec![0u8; SRC * SRC * 2];
  super::pseudo_random_u8(&mut buf, seed);
  buf
}

/// Canonical `R, G, B, A` of one packed `[Y, A]` plane: `R = G = B = Y`,
/// `A` passed through — the exact `ya8_to_rgba_row` mapping. Only the
/// `rgb`-gated fractional-ratio oracle consumes it.
#[cfg(feature = "rgb")]
fn canonical_from_packed(packed: &[u8], n: usize) -> Vec<u8> {
  let mut out = std::vec![0u8; n * 4];
  for i in 0..n {
    let y = packed[i * 2];
    let a = packed[i * 2 + 1];
    out[i * 4] = y;
    out[i * 4 + 1] = y;
    out[i * 4 + 2] = y;
    out[i * 4 + 3] = a;
  }
  out
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

/// Round-half-up 2x2 block mean of the **native Y plane** of a packed
/// `[Y, A]` source (`Y = packed[2*i]`) — the alpha-independent native-Y
/// area-downscale oracle. This is `mean(Y)`, NOT the color path's
/// `mean(Y*A)/mean(A)`, so it is the correct `luma` source under every
/// alpha mode.
fn block_mean_native_y(packed: &[u8]) -> Vec<u8> {
  let mut out = std::vec![0u8; OUT * OUT];
  for oy in 0..OUT {
    for ox in 0..OUT {
      let mut acc = 0u32;
      for dy in 0..2 {
        for dx in 0..2 {
          acc += packed[((oy * 2 + dy) * SRC + ox * 2 + dx) * 2] as u32;
        }
      }
      out[oy * OUT + ox] = ((acc + 2) / 4) as u8;
    }
  }
  out
}

/// Premultiply one canonical RGBA plane in place — `round(c * a / 255)`
/// per color channel, alpha untouched.
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

/// Drop alpha from a canonical RGBA plane → packed RGB.
fn drop_alpha(rgba: &[u8]) -> Vec<u8> {
  let mut out = std::vec![0u8; rgba.len() / 4 * 3];
  for (o, i) in out.chunks_exact_mut(3).zip(rgba.chunks_exact(4)) {
    o.copy_from_slice(&i[..3]);
  }
  out
}

/// Zero-extend a u8 plane to u16.
fn zero_extend(plane: &[u8]) -> Vec<u16> {
  plane.iter().map(|&v| v as u16).collect()
}

/// Full-resolution canonical RGBA of the source — a direct (identity)
/// `Ya8` conversion. The oracles bin / premultiply this.
fn direct_rgba(packed: &[u8]) -> Vec<u8> {
  let frame = Ya8Frame::new(packed, SRC as u32, SRC as u32, (SRC * 2) as u32);
  let mut rgba = std::vec![0u8; SRC * SRC * 4];
  let mut sink = MixedSinker::<Ya8>::new(SRC, SRC)
    .with_rgba(&mut rgba)
    .unwrap();
  ya8_to(&frame, FR, sink.set_kernel_matrix(M)).unwrap();
  rgba
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn straight_rgba_is_block_mean_of_direct() {
  let packed = packed_frame(0x51A1);
  let frame = Ya8Frame::new(&packed, SRC as u32, SRC as u32, (SRC * 2) as u32);

  let mut rgba = std::vec![0u8; OUT * OUT * 4];
  {
    let mut sink =
      MixedSinker::<Ya8, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap();
    ya8_to(&frame, FR, sink.set_kernel_matrix(M)).unwrap();
  }
  assert_eq!(rgba, block_mean_rgba(&direct_rgba(&packed)));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn straight_alpha_is_averaged_not_forced_opaque() {
  let mut packed = packed_frame(0x9E37);
  for (i, px) in packed.chunks_exact_mut(2).enumerate() {
    px[1] = (i as u8).wrapping_mul(3);
  }
  let frame = Ya8Frame::new(&packed, SRC as u32, SRC as u32, (SRC * 2) as u32);

  let mut rgba = std::vec![0u8; OUT * OUT * 4];
  {
    let mut sink =
      MixedSinker::<Ya8, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap();
    ya8_to(&frame, FR, sink.set_kernel_matrix(M)).unwrap();
  }
  let oracle = block_mean_rgba(&direct_rgba(&packed));
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
  // Every output flavour at once: rgba is the binned color, rgb drops
  // alpha, rgb_u16 / rgba_u16 zero-extend it, and luma / luma_u16 / hsv
  // match a direct full-res `Ya8` conversion of the binned Y.
  let packed = packed_frame(0xBEEF);
  let frame = Ya8Frame::new(&packed, SRC as u32, SRC as u32, (SRC * 2) as u32);

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
      MixedSinker::<Ya8, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
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
    ya8_to(&frame, FR, sink.set_kernel_matrix(M)).unwrap();
  }

  let binned = block_mean_rgba(&direct_rgba(&packed));
  assert_eq!(rgba, binned, "rgba == block mean");
  assert_eq!(rgb, drop_alpha(&binned), "rgb == drop-alpha(binned)");
  assert_eq!(
    rgba_u16,
    zero_extend(&binned),
    "rgba_u16 == zero-extend(binned)"
  );
  assert_eq!(
    rgb_u16,
    zero_extend(&drop_alpha(&binned)),
    "rgb_u16 == zero-extend(drop-alpha(binned))"
  );

  // luma / luma_u16: native Y. The binned color's R channel IS the binned
  // Y (R = Y at every source pixel), so a direct `Ya8` conversion of the
  // binned Y is the byte-exact reference for every matrix.
  let binned_rgb = drop_alpha(&binned);
  let mut binned_packed = std::vec![0u8; OUT * OUT * 2];
  for i in 0..OUT * OUT {
    binned_packed[i * 2] = binned_rgb[i * 3]; // R == Y
    binned_packed[i * 2 + 1] = 0xFF;
  }
  let binned_frame = Ya8Frame::new(&binned_packed, OUT as u32, OUT as u32, (OUT * 2) as u32);
  let mut luma_ref = std::vec![0u8; OUT * OUT];
  let mut lu16_ref = std::vec![0u16; OUT * OUT];
  let mut h_ref = std::vec![0u8; OUT * OUT];
  let mut s_ref = std::vec![0u8; OUT * OUT];
  let mut v_ref = std::vec![0u8; OUT * OUT];
  {
    let mut sink = MixedSinker::<Ya8>::new(OUT, OUT)
      .with_luma(&mut luma_ref)
      .unwrap()
      .with_luma_u16(&mut lu16_ref)
      .unwrap()
      .with_hsv(&mut h_ref, &mut s_ref, &mut v_ref)
      .unwrap();
    ya8_to(&binned_frame, FR, sink.set_kernel_matrix(M)).unwrap();
  }
  assert_eq!(luma, luma_ref, "luma (native Y)");
  assert_eq!(lu16, lu16_ref, "luma_u16 (native Y zero-extended)");
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
  let packed = packed_frame(0x1234);
  let frame = Ya8Frame::new(&packed, SRC as u32, SRC as u32, (SRC * 2) as u32);

  let mut rgba = std::vec![0u8; OUT * OUT * 4];
  let mut rgb = std::vec![0u8; OUT * OUT * 3];
  let mut rgba_u16 = std::vec![0u16; OUT * OUT * 4];
  let mut lu16 = std::vec![0u16; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Ya8, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_alpha_mode(AlphaMode::Premultiplied)
        .with_rgba(&mut rgba)
        .unwrap()
        .with_rgb(&mut rgb)
        .unwrap()
        .with_rgba_u16(&mut rgba_u16)
        .unwrap()
        .with_luma_u16(&mut lu16)
        .unwrap();
    ya8_to(&frame, FR, sink.set_kernel_matrix(M)).unwrap();
  }

  let mut pm = direct_rgba(&packed);
  premultiply(&mut pm);
  let binned = block_mean_rgba(&pm);
  let oracle = unpremultiply(&binned);
  assert_eq!(rgba, oracle, "premult rgba");
  assert_eq!(rgb, drop_alpha(&oracle), "premult rgb");
  assert_eq!(rgba_u16, zero_extend(&oracle), "premult rgba_u16");
  // luma_u16 under premult is the area-mean of the NATIVE Y plane
  // (`mean(Y)`), zero-extended — alpha-INDEPENDENT, NOT the color path's
  // `mean(Y*A)/mean(A)` (the un-premultiplied straight R). Compare to the
  // native-Y bin oracle, which equals the direct `ya8_to_luma_u16_row`.
  let binned_y = block_mean_native_y(&packed);
  let (_, lu16_ref) = direct_luma_of_binned_y(&binned_y, FR);
  assert_eq!(lu16, lu16_ref, "premult luma_u16 (native-Y bin oracle)");
  assert_eq!(
    lu16,
    zero_extend(&binned_y),
    "premult luma_u16 == native-Y bin zero-extended"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn premultiplied_transparent_block_does_not_bleed() {
  let mut packed = packed_frame(0xABCD);
  for off in [(0, 0), (1, 0), (0, 1), (1, 1)] {
    let i = off.1 * SRC + off.0;
    packed[i * 2] = 250; // Y
    packed[i * 2 + 1] = 0; // A
  }
  let frame = Ya8Frame::new(&packed, SRC as u32, SRC as u32, (SRC * 2) as u32);

  let mut rgba = std::vec![0u8; OUT * OUT * 4];
  {
    let mut sink =
      MixedSinker::<Ya8, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_alpha_mode(AlphaMode::Premultiplied)
        .with_rgba(&mut rgba)
        .unwrap();
    ya8_to(&frame, FR, sink.set_kernel_matrix(M)).unwrap();
  }
  assert_eq!(&rgba[..4], &[0, 0, 0, 0], "transparent block bled color");

  let mut pm = direct_rgba(&packed);
  premultiply(&mut pm);
  let oracle = unpremultiply(&block_mean_rgba(&pm));
  assert_eq!(rgba, oracle, "premult output != oracle");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn premultiplied_nonuniform_alpha_luma_is_native_y_bin_not_color() {
  // The cited counterexample, tiled over every 2x2 block:
  //   (Y, A) = (0, 255), (255, 0), (0, 255), (255, 0)
  // Native-Y mean = (0 + 255 + 0 + 255) / 4 = 128 (round-half-up).
  // The premultiplied color collapses to mean(Y*A)/mean(A): premult color
  // is (0,0,0,255),(0,0,0,0),(0,0,0,255),(0,0,0,0); the binned alpha is
  // 128 and the binned premult-color R is 0, so an un-premultiplied
  // (color-derived) luma would be 0 — the bug. Native-Y luma must be 128.
  let mut packed = std::vec![0u8; SRC * SRC * 2];
  for (i, px) in packed.chunks_exact_mut(2).enumerate() {
    let col = i % SRC;
    // Y = 0 on even columns, 255 on odd; A = the complement, so every pixel
    // with Y = 255 is fully transparent and every Y = 0 is fully opaque.
    let odd = col % 2 == 1;
    px[0] = if odd { 255 } else { 0 }; // Y
    px[1] = if odd { 0 } else { 255 }; // A
  }
  let frame = Ya8Frame::new(&packed, SRC as u32, SRC as u32, (SRC * 2) as u32);

  let mut luma = std::vec![0u8; OUT * OUT];
  let mut lu16 = std::vec![0u16; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Ya8, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_alpha_mode(AlphaMode::Premultiplied)
        .with_luma(&mut luma)
        .unwrap()
        .with_luma_u16(&mut lu16)
        .unwrap();
    ya8_to(&frame, FR, sink.set_kernel_matrix(M)).unwrap();
  }

  // Every output pixel's native-Y mean is 128.
  assert!(
    luma.iter().all(|&y| y == 128),
    "premult non-uniform-alpha luma must be native-Y mean 128, got {luma:?}"
  );
  assert!(
    lu16.iter().all(|&y| y == 128),
    "premult non-uniform-alpha luma_u16 must be native-Y mean 128, got {lu16:?}"
  );

  // And it equals the native-Y bin oracle (the direct `ya8_to_luma*` of the
  // block-meaned native Y plane) — NOT the un-premultiplied color R, which
  // would be 0 here.
  let binned_y = block_mean_native_y(&packed);
  assert!(binned_y.iter().all(|&y| y == 128), "native-Y bin sanity");
  let (luma_ref, lu16_ref) = direct_luma_of_binned_y(&binned_y, FR);
  assert_eq!(luma, luma_ref, "premult luma == native-Y bin oracle");
  assert_eq!(lu16, lu16_ref, "premult luma_u16 == native-Y bin oracle");

  // Guard the test: the color-derived (un-premultiplied straight R) luma
  // really would be 0 here, so this pins the divergence the bug produced.
  let mut pm = direct_rgba(&packed);
  premultiply(&mut pm);
  let color_oracle = unpremultiply(&block_mean_rgba(&pm));
  let color_luma_r: Vec<u8> = color_oracle.chunks_exact(4).map(|px| px[0]).collect();
  assert!(
    color_luma_r.iter().all(|&r| r == 0),
    "fixture failed to exercise the color-vs-native-Y divergence"
  );
  assert_ne!(luma, color_luma_r, "luma must NOT be the color-derived R");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn straight_and_premult_differ_under_varying_alpha() {
  let mut packed = packed_frame(0x77AA);
  for (i, px) in packed.chunks_exact_mut(2).enumerate() {
    px[1] = 16u8.wrapping_add((i as u8).wrapping_mul(5));
  }
  let render = |mode: AlphaMode| {
    let frame = Ya8Frame::new(&packed, SRC as u32, SRC as u32, (SRC * 2) as u32);
    let mut rgba = std::vec![0u8; OUT * OUT * 4];
    let mut sink =
      MixedSinker::<Ya8, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_alpha_mode(mode)
        .with_rgba(&mut rgba)
        .unwrap();
    ya8_to(&frame, FR, sink.set_kernel_matrix(M)).unwrap();
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
  let sink = MixedSinker::<Ya8>::new(SRC, SRC);
  assert_eq!(sink.alpha_mode(), AlphaMode::Straight);
  assert!(sink.alpha_mode().is_straight());
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn identity_plan_matches_direct() {
  let packed = packed_frame(0x0F0F);
  let frame = Ya8Frame::new(&packed, SRC as u32, SRC as u32, (SRC * 2) as u32);

  let mut rgba = std::vec![0u8; SRC * SRC * 4];
  {
    let mut sink =
      MixedSinker::<Ya8, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(SRC, SRC))
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap();
    ya8_to(&frame, FR, sink.set_kernel_matrix(M)).unwrap();
  }
  assert_eq!(rgba, direct_rgba(&packed), "identity plan == direct");
}

// The fractional-ratio reference reuses the packed-RGBA 8-bit source as an
// independent area-engine oracle, so it is gated on `rgb` (its frame /
// walker live there); a `gray`-solo build covers fractional ratios via the
// shared `resample_packed_rgba_8bit` suite.
#[cfg(feature = "rgb")]
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn fractional_ratio_matches_direct_then_bin() {
  // 8 -> 3 fractional downscale: assert the resampled rgba equals a direct
  // full-res convert fed through the SAME AreaResampler at OUT=3 over the
  // canonical RGBA frame (the area engine is the source of truth; this
  // guards the Ya8 decode + routing, not the resampler arithmetic).
  const F: usize = 3;
  let packed = packed_frame(0xF2AC);
  let frame = Ya8Frame::new(&packed, SRC as u32, SRC as u32, (SRC * 2) as u32);

  let mut rgba = std::vec![0u8; F * F * 4];
  {
    let mut sink =
      MixedSinker::<Ya8, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(F, F))
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap();
    ya8_to(&frame, FR, sink.set_kernel_matrix(M)).unwrap();
  }

  // Reference: feed the canonical RGBA (R=G=B=Y, A) through the packed-RGBA
  // 8-bit source at the same plan. That path is already covered/trusted;
  // Ya8 must match it exactly because its decode produces that very row.
  let canonical = canonical_from_packed(&packed, SRC * SRC);
  let mut rgba_ref = std::vec![0u8; F * F * 4];
  {
    use crate::{frame::RgbaFrame, source::rgba_to};
    let rsrc = RgbaFrame::new(&canonical, SRC as u32, SRC as u32, (SRC * 4) as u32);
    let mut sink = MixedSinker::<crate::source::Rgba, AreaResampler>::with_resampler(
      SRC,
      SRC,
      AreaResampler::to(F, F),
    )
    .unwrap()
    .with_rgba(&mut rgba_ref)
    .unwrap();
    rgba_to(&rsrc, FR, sink.set_kernel_matrix(M)).unwrap();
  }
  assert_eq!(rgba, rgba_ref, "Ya8 8->3 != packed-RGBA 8->3 of canonical");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn cross_frame_reset_reuses_streams() {
  let packed = packed_frame(0x5151);
  let frame = Ya8Frame::new(&packed, SRC as u32, SRC as u32, (SRC * 2) as u32);

  let mut rgba = std::vec![0u8; OUT * OUT * 4];
  {
    let mut sink =
      MixedSinker::<Ya8, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap();
    ya8_to(&frame, FR, sink.set_kernel_matrix(M)).unwrap();
    ya8_to(&frame, FR, sink.set_kernel_matrix(M)).unwrap();
  }
  assert_eq!(rgba, block_mean_rgba(&direct_rgba(&packed)));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn accepts_alpha_mode_change_across_frames() {
  let packed = packed_frame(0xB2B2);
  let frame = Ya8Frame::new(&packed, SRC as u32, SRC as u32, (SRC * 2) as u32);

  let mut rgba = std::vec![0u8; OUT * OUT * 4];
  {
    let mut sink =
      MixedSinker::<Ya8, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap();
    ya8_to(&frame, FR, sink.set_kernel_matrix(M)).unwrap();
    sink.set_alpha_mode(AlphaMode::Premultiplied);
    ya8_to(&frame, FR, sink.set_kernel_matrix(M))
      .expect("a fresh frame must accept a different alpha mode");
  }
  let mut pm = direct_rgba(&packed);
  premultiply(&mut pm);
  let oracle = unpremultiply(&block_mean_rgba(&pm));
  assert_eq!(rgba, oracle, "premult frame 2 output");
}

// ---- direct-row freeze / sequencing (Ya8Row is publicly constructible) ----

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn mid_frame_alpha_mode_flip_is_rejected() {
  // Feed row 0 under Straight, flip to Premultiplied, then feed row 1: the
  // frozen mode (snapshotted in begin_frame) must reject the changed mode
  // before any further binning.
  let packed = packed_frame(0x33AA);
  let row_bytes = SRC * 2;
  let mut rgba = std::vec![0u8; OUT * OUT * 4];
  let mut sink =
    MixedSinker::<Ya8, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_rgba(&mut rgba)
      .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  sink
    .process(Ya8Row::for_tests(&packed[..row_bytes], 0, M, FR))
    .unwrap();
  sink.set_alpha_mode(AlphaMode::Premultiplied);
  let err = sink
    .process(Ya8Row::for_tests(
      &packed[row_bytes..2 * row_bytes],
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
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn out_of_sequence_first_row_is_rejected() {
  // A fresh resampling sink expects row 0 first; feeding row 1 first trips
  // OutOfSequenceRow (before any snapshot is stored).
  let packed = packed_frame(0x44BB);
  let row_bytes = SRC * 2;
  let mut rgba = std::vec![0u8; OUT * OUT * 4];
  let mut sink =
    MixedSinker::<Ya8, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_rgba(&mut rgba)
      .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  let err = sink
    .process(Ya8Row::for_tests(
      &packed[row_bytes..2 * row_bytes],
      1,
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
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn no_output_sink_is_a_noop() {
  let packed = packed_frame(0x4242);
  let frame = Ya8Frame::new(&packed, SRC as u32, SRC as u32, (SRC * 2) as u32);
  let mut sink =
    MixedSinker::<Ya8, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap();
  ya8_to(&frame, FR, sink.set_kernel_matrix(M)).unwrap();
}

// ---- limited-range (full_range = false) native-Y luma regression --------------
//
// The direct `Ya8` luma is the native Y byte pass-through and luma_u16 its
// zero-extension (`ya8_to_luma_row` / `ya8_to_luma_u16_row`): neither applies
// the matrix or range. The resample tail must reproduce that for BOTH ranges.
// A prior version derived luma from the binned RGB via `rgb_to_luma*`, which
// is byte-identical to native Y only at `full_range = true` (a full-range gray
// maps to itself) — so a `full_range = false` row corrupted the grayscale (a
// valid Y = 16 became ≈30). These tests pin the limited-range case that the
// FR=true suite could not catch.

/// Direct (identity) `Ya8` luma / luma_u16 of a binned-Y plane at the given
/// range — the byte-exact native-Y oracle. `binned_y[i]` is the binned Y byte
/// (alpha forced opaque, irrelevant to luma).
fn direct_luma_of_binned_y(binned_y: &[u8], full_range: bool) -> (Vec<u8>, Vec<u16>) {
  let n = binned_y.len();
  let mut packed = std::vec![0u8; n * 2];
  for (i, &y) in binned_y.iter().enumerate() {
    packed[i * 2] = y;
    packed[i * 2 + 1] = 0xFF;
  }
  let frame = Ya8Frame::new(&packed, n as u32, 1, (n * 2) as u32);
  let mut luma = std::vec![0u8; n];
  let mut lu16 = std::vec![0u16; n];
  let mut sink = MixedSinker::<Ya8>::new(n, 1)
    .with_luma(&mut luma)
    .unwrap()
    .with_luma_u16(&mut lu16)
    .unwrap();
  ya8_to(&frame, full_range, sink.set_kernel_matrix(M)).unwrap();
  (luma, lu16)
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn limited_range_luma_is_native_y_not_rgb_derived() {
  let packed = packed_frame(0xCAFE);
  let frame = Ya8Frame::new(&packed, SRC as u32, SRC as u32, (SRC * 2) as u32);

  let mut luma = std::vec![0u8; OUT * OUT];
  let mut lu16 = std::vec![0u16; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Ya8, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_luma(&mut luma)
        .unwrap()
        .with_luma_u16(&mut lu16)
        .unwrap();
    ya8_to(&frame, FR_LIMITED, sink.set_kernel_matrix(M)).unwrap();
  }

  // Native-Y oracle: the area-mean of the native Y plane (alpha-independent
  // by construction).
  let binned_y = block_mean_native_y(&packed);
  let (luma_ref, lu16_ref) = direct_luma_of_binned_y(&binned_y, FR_LIMITED);
  assert_eq!(luma, luma_ref, "limited-range luma must be native Y");
  assert_eq!(luma, binned_y, "limited-range luma == binned Y byte");
  assert_eq!(
    lu16, lu16_ref,
    "limited-range luma_u16 must be native Y zero-extended"
  );
  assert_eq!(
    lu16,
    zero_extend(&binned_y),
    "limited-range luma_u16 == binned Y zero-extended"
  );

  // Native Y is range-independent: the SAME source at full range yields the
  // identical luma. A range-derived (`rgb_to_luma*`) luma would differ here —
  // this is the divergence the FR=true suite could not surface.
  let mut luma_fr = std::vec![0u8; OUT * OUT];
  let mut lu16_fr = std::vec![0u16; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Ya8, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_luma(&mut luma_fr)
        .unwrap()
        .with_luma_u16(&mut lu16_fr)
        .unwrap();
    ya8_to(&frame, FR, sink.set_kernel_matrix(M)).unwrap();
  }
  assert_eq!(luma, luma_fr, "native-Y luma must be range-independent");
  assert_eq!(lu16, lu16_fr, "native-Y luma_u16 must be range-independent");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn limited_range_y16_luma_is_16_not_rgb_scaled() {
  // The cited counterexample: a uniform Y = 16 limited-range gray. The direct
  // `Ya8` luma is 16 (native Y); a limited-range `rgb_to_luma_row` of
  // (16,16,16) scales up to ≈30. Pin native 16.
  let mut packed = std::vec![0u8; SRC * SRC * 2];
  for px in packed.chunks_exact_mut(2) {
    px[0] = 16; // Y
    px[1] = 0xFF; // A (opaque)
  }
  let frame = Ya8Frame::new(&packed, SRC as u32, SRC as u32, (SRC * 2) as u32);

  let mut luma = std::vec![0u8; OUT * OUT];
  let mut lu16 = std::vec![0u16; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Ya8, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_luma(&mut luma)
        .unwrap()
        .with_luma_u16(&mut lu16)
        .unwrap();
    ya8_to(&frame, FR_LIMITED, sink.set_kernel_matrix(M)).unwrap();
  }
  assert!(
    luma.iter().all(|&y| y == 16),
    "limited-range Y=16 luma must stay native 16, got {luma:?}"
  );
  assert!(
    lu16.iter().all(|&y| y == 16),
    "limited-range Y=16 luma_u16 must stay native 16, got {lu16:?}"
  );
}
