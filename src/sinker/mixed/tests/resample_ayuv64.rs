//! Alpha-aware fused-downscale coverage for the packed 16-bit 4:4:4 YUV
//! source with real source alpha, `Ayuv64` (`[A, Y, U, V]` u16 quadruple).
//!
//! `Ayuv64` is the most demanding alpha family: four independently-rounding
//! outputs. It routes through the packed-YUVA tail
//! ([`packed_yuva444_resample`](super::super::packed_yuva444_resample)) at
//! `SRC_BITS = 16` with THREE independent binnings:
//! - u8 colour bins the converted u8 RGBA row (`ayuv64_to_rgba_row`, α
//!   `>> 8`) → rgb / rgba / hsv;
//! - u16 colour bins the INDEPENDENT native u16 RGBA row
//!   (`ayuv64_to_rgba_u16_row`, α direct) → rgb_u16 / rgba_u16 — never a
//!   narrowing of the u8 bin (the u8 and u16 `YUV→RGB` kernels round
//!   independently, the uniform-gray + saturated-chroma counterexamples);
//! - native Y (slot 1) bins through the 1-channel u16 luma stream → luma_u16
//!   (native) / luma (`>> 8`) — alpha- and range-independent.
//!
//! Each colour binning is straight in [`AlphaMode::Straight`] and
//! premultiply-bin-unpremultiply (at its own depth max) in
//! [`AlphaMode::Premultiplied`]. Oracles are built from `Ayuv64`'s own
//! direct kernels; LE/BE parity exercises the `<const BE>` propagation.

use crate::{
  KernelMatrix, PixelSink,
  frame::{Ayuv64BeFrame, Ayuv64Frame, Ayuv64LeFrame},
  resample::{AreaResampler, ResampleError},
  sinker::{AlphaMode, MixedSinker, MixedSinkerError},
  source::{Ayuv64, Ayuv64Row, ayuv64_to, ayuv64_to_endian},
};

const SRC: usize = 8;
const OUT: usize = 4;
const M: KernelMatrix = KernelMatrix::Bt709;
const FR: bool = true;
const FR_LIMITED: bool = false;
const MAX16: u32 = 65535;

/// Pseudo-random packed `[A, Y, U, V]` u16 plane (`SRC * SRC * 4` elems);
/// alpha varies (not all-opaque). Logical (host-native) values — fed to an
/// LE `Ayuv64<false>` sink whose loader is identity on an LE host (the CI
/// hosts), matching the direct-kernel oracles built from the same values.
fn packed_frame(seed: u32) -> Vec<u16> {
  let mut buf = std::vec![0u16; SRC * SRC * 4];
  super::pseudo_random_u16_low_n_bits(&mut buf, seed, 16);
  buf
}

fn ayuv64_frame(buf: &[u16]) -> Ayuv64Frame<'_> {
  Ayuv64Frame::try_new(buf, SRC as u32, SRC as u32, (SRC * 4) as u32).unwrap()
}

/// Full-resolution direct (identity) `Ayuv64` u8 RGBA via
/// `ayuv64_to_rgba_row` (α `>> 8`). The u8 oracles bin / premultiply this.
fn direct_rgba_u8(packed: &[u16], full_range: bool) -> Vec<u8> {
  let frame = ayuv64_frame(packed);
  let mut rgba = std::vec![0u8; SRC * SRC * 4];
  {
    let mut sink = MixedSinker::<Ayuv64>::new(SRC, SRC)
      .with_rgba(&mut rgba)
      .unwrap();
    ayuv64_to(&frame, full_range, sink.set_kernel_matrix(M)).unwrap();
  }
  rgba
}

/// Full-resolution direct (identity) `Ayuv64` u16 RGBA via
/// `ayuv64_to_rgba_u16_row` (α direct). The u16 oracles bin / premultiply
/// this.
fn direct_rgba_u16(packed: &[u16], full_range: bool) -> Vec<u16> {
  let frame = ayuv64_frame(packed);
  let mut rgba = std::vec![0u16; SRC * SRC * 4];
  {
    let mut sink = MixedSinker::<Ayuv64>::new(SRC, SRC)
      .with_rgba_u16(&mut rgba)
      .unwrap();
    ayuv64_to(&frame, full_range, sink.set_kernel_matrix(M)).unwrap();
  }
  rgba
}

/// Round-half-up 2x2 block mean of a canonical RGBA plane.
fn block_mean_rgba<T: Copy + Into<u64>>(src: &[T], round_div_to: fn(u64) -> u64) -> Vec<u64> {
  let mut out = std::vec![0u64; OUT * OUT * 4];
  for oy in 0..OUT {
    for ox in 0..OUT {
      for c in 0..4 {
        let mut acc = 0u64;
        for dy in 0..2 {
          for dx in 0..2 {
            acc += src[((oy * 2 + dy) * SRC + ox * 2 + dx) * 4 + c].into();
          }
        }
        out[(oy * OUT + ox) * 4 + c] = round_div_to(acc);
      }
    }
  }
  out
}

fn bin4(acc: u64) -> u64 {
  (acc + 2) / 4
}

/// Round-half-up 2x2 block mean of a u8 RGBA plane → u8.
fn block_mean_rgba_u8(src: &[u8]) -> Vec<u8> {
  block_mean_rgba(src, bin4)
    .iter()
    .map(|&v| v as u8)
    .collect()
}

/// Round-half-up 2x2 block mean of a u16 RGBA plane → u16.
fn block_mean_rgba_u16(src: &[u16]) -> Vec<u16> {
  block_mean_rgba(src, bin4)
    .iter()
    .map(|&v| v as u16)
    .collect()
}

/// Round-half-up 2x2 block mean of the native Y plane (`Y = packed[4*i+1]`)
/// → u16 — the alpha-independent native-Y oracle.
fn block_mean_native_y(packed: &[u16]) -> Vec<u16> {
  let mut out = std::vec![0u16; OUT * OUT];
  for oy in 0..OUT {
    for ox in 0..OUT {
      let mut acc = 0u64;
      for dy in 0..2 {
        for dx in 0..2 {
          acc += packed[((oy * 2 + dy) * SRC + ox * 2 + dx) * 4 + 1] as u64;
        }
      }
      out[oy * OUT + ox] = ((acc + 2) / 4) as u16;
    }
  }
  out
}

fn premultiply<T: Copy + Into<u32> + TryFrom<u32>>(plane: &mut [T], max: u32) {
  let half = max / 2;
  for px in plane.chunks_exact_mut(4) {
    let a: u32 = px[3].into();
    for c in &mut px[..3] {
      let v: u32 = (*c).into();
      let pm = (v * a + half) / max;
      *c = T::try_from(pm).unwrap_or_else(|_| unreachable!());
    }
  }
}

fn unpremultiply<T: Copy + Into<u32> + TryFrom<u32>>(plane: &[T], max: u32) -> Vec<T> {
  let mut out: Vec<T> = plane.to_vec();
  for (o, i) in out.chunks_exact_mut(4).zip(plane.chunks_exact(4)) {
    let a: u32 = i[3].into();
    for c in 0..3 {
      let pm: u32 = i[c].into();
      let straight = (pm * max + a / 2).checked_div(a).map_or(0, |q| q.min(max));
      o[c] = T::try_from(straight).unwrap_or_else(|_| unreachable!());
    }
    o[3] = i[3];
  }
  out
}

fn drop_alpha_u8(rgba: &[u8]) -> Vec<u8> {
  let mut out = std::vec![0u8; rgba.len() / 4 * 3];
  for (o, i) in out.chunks_exact_mut(3).zip(rgba.chunks_exact(4)) {
    o.copy_from_slice(&i[..3]);
  }
  out
}

fn drop_alpha_u16(rgba: &[u16]) -> Vec<u16> {
  let mut out = std::vec![0u16; rgba.len() / 4 * 3];
  for (o, i) in out.chunks_exact_mut(3).zip(rgba.chunks_exact(4)) {
    o.copy_from_slice(&i[..3]);
  }
  out
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn ayuv64_straight_all_outputs_match_their_own_block_mean() {
  // Every output attached at once: each must match the block mean of its
  // OWN direct conversion at its own depth, proving the three streams
  // coexist and none is derived by narrowing another.
  let packed = packed_frame(0xBEEF);
  let frame = ayuv64_frame(&packed);

  let mut rgb = std::vec![0u8; OUT * OUT * 3];
  let mut rgba = std::vec![0u8; OUT * OUT * 4];
  let mut rgb_u16 = std::vec![0u16; OUT * OUT * 3];
  let mut rgba_u16 = std::vec![0u16; OUT * OUT * 4];
  let mut luma = std::vec![0u8; OUT * OUT];
  let mut luma_u16 = std::vec![0u16; OUT * OUT];
  let mut hh = std::vec![0u8; OUT * OUT];
  let mut ss = std::vec![0u8; OUT * OUT];
  let mut vv = std::vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Ayuv64, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
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
        .with_luma_u16(&mut luma_u16)
        .unwrap()
        .with_hsv(&mut hh, &mut ss, &mut vv)
        .unwrap();
    ayuv64_to(&frame, FR, sink.set_kernel_matrix(M)).unwrap();
  }

  let rgba_ref = block_mean_rgba_u8(&direct_rgba_u8(&packed, FR));
  let rgb_ref = drop_alpha_u8(&rgba_ref);
  assert_eq!(rgba, rgba_ref, "rgba == u8 block mean (alpha averaged)");
  assert_eq!(rgb, rgb_ref, "rgb == drop-alpha(u8 block mean)");

  let rgba_u16_ref = block_mean_rgba_u16(&direct_rgba_u16(&packed, FR));
  assert_eq!(rgba_u16, rgba_u16_ref, "rgba_u16 == native u16 block mean");
  assert_eq!(
    rgb_u16,
    drop_alpha_u16(&rgba_u16_ref),
    "rgb_u16 == drop-alpha(u16 block mean)"
  );

  let y_binned = block_mean_native_y(&packed);
  assert_eq!(luma_u16, y_binned, "luma_u16 == native-Y block mean");
  let luma_ref: Vec<u8> = y_binned.iter().map(|&p| (p >> 8) as u8).collect();
  assert_eq!(luma, luma_ref, "luma == native Y >> 8");

  let mut h_ref = std::vec![0u8; OUT * OUT];
  let mut s_ref = std::vec![0u8; OUT * OUT];
  let mut v_ref = std::vec![0u8; OUT * OUT];
  crate::row::rgb_to_hsv_row(
    &rgb_ref,
    &mut h_ref,
    &mut s_ref,
    &mut v_ref,
    OUT * OUT,
    false,
  );
  assert_eq!(hh, h_ref, "hsv H");
  assert_eq!(ss, s_ref, "hsv S");
  assert_eq!(vv, v_ref, "hsv V");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn ayuv64_straight_alpha_is_averaged_not_forced_opaque() {
  let mut packed = packed_frame(0x9E37);
  for (i, px) in packed.chunks_exact_mut(4).enumerate() {
    px[0] = (i as u16).wrapping_mul(700); // A varies
  }
  let frame = ayuv64_frame(&packed);
  let mut rgba_u16 = std::vec![0u16; OUT * OUT * 4];
  {
    let mut sink =
      MixedSinker::<Ayuv64, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgba_u16(&mut rgba_u16)
        .unwrap();
    ayuv64_to(&frame, FR, sink.set_kernel_matrix(M)).unwrap();
  }
  assert_eq!(rgba_u16, block_mean_rgba_u16(&direct_rgba_u16(&packed, FR)));
  assert!(
    rgba_u16.chunks_exact(4).any(|px| px[3] != MAX16 as u16),
    "resampled u16 alpha was forced opaque"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn ayuv64_uniform_gray_independent_u8_vs_u16_colour() {
  // A uniform frame downscaled MUST NOT change any colour output: each
  // equals the direct conversion of the same uniform frame. Deriving u8
  // colour by narrowing the binned u16 (rather than binning the direct u8
  // conversion) would shift the u8 RGB under the two paths' independent
  // rounding — the regression this pins.
  let packed = std::vec![[40000u16, 38000, 32768, 32768]; SRC * SRC]
    .into_iter()
    .flatten()
    .collect::<Vec<u16>>();
  let full_u8 = direct_rgba_u8(&packed, FR);
  let full_u16 = direct_rgba_u16(&packed, FR);

  let mut rgb = std::vec![0u8; OUT * OUT * 3];
  let mut rgb_u16 = std::vec![0u16; OUT * OUT * 3];
  {
    let mut sink =
      MixedSinker::<Ayuv64, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb(&mut rgb)
        .unwrap()
        .with_rgb_u16(&mut rgb_u16)
        .unwrap();
    ayuv64_to(&ayuv64_frame(&packed), FR, sink.set_kernel_matrix(M)).unwrap();
  }
  let gray_u8 = &full_u8[..3];
  for px in rgb.chunks_exact(3) {
    assert_eq!(
      px, gray_u8,
      "uniform-gray u8 RGB changed (narrowed from u16?)"
    );
  }
  let gray_u16 = &full_u16[..3];
  for px in rgb_u16.chunks_exact(3) {
    assert_eq!(px, gray_u16, "uniform-gray u16 RGB changed under downscale");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn ayuv64_saturated_chroma_u8_is_not_a_narrowing_of_u16() {
  // Saturated chroma drives the YUV→RGB clamps; with a chroma ramp the u8
  // and u16 conversions round/clamp independently, so block-meaning each at
  // its own depth must match the respective direct kernel — a u8 path
  // derived by `>> 8` of the binned u16 would diverge here.
  let mut packed = std::vec![0u16; SRC * SRC * 4];
  for (i, px) in packed.chunks_exact_mut(4).enumerate() {
    px[0] = 0xFFFF; // A opaque
    px[1] = 30000 + (i as u16) * 100; // Y ramp
    px[2] = (i as u16).wrapping_mul(2000); // U — drives saturation
    px[3] = 60000u16.wrapping_sub((i as u16).wrapping_mul(1500)); // V
  }
  let frame = ayuv64_frame(&packed);
  let mut rgb = std::vec![0u8; OUT * OUT * 3];
  let mut rgb_u16 = std::vec![0u16; OUT * OUT * 3];
  {
    let mut sink =
      MixedSinker::<Ayuv64, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb(&mut rgb)
        .unwrap()
        .with_rgb_u16(&mut rgb_u16)
        .unwrap();
    ayuv64_to(&frame, FR, sink.set_kernel_matrix(M)).unwrap();
  }
  let rgb_ref = drop_alpha_u8(&block_mean_rgba_u8(&direct_rgba_u8(&packed, FR)));
  let rgb_u16_ref = drop_alpha_u16(&block_mean_rgba_u16(&direct_rgba_u16(&packed, FR)));
  assert_eq!(rgb, rgb_ref, "u8 RGB must bin the direct u8 conversion");
  assert_eq!(
    rgb_u16, rgb_u16_ref,
    "u16 RGB must bin the direct u16 conversion"
  );
  // The independent rounding really diverges: narrowing the u16 bin to u8
  // is not byte-identical to the u8 bin (guards the counterexample's bite).
  let narrowed: Vec<u8> = rgb_u16_ref.iter().map(|&v| (v >> 8) as u8).collect();
  assert_ne!(
    rgb_ref, narrowed,
    "fixture failed to exercise u8-vs-u16 divergence"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn ayuv64_premultiplied_independent_u8_and_u16() {
  let packed = packed_frame(0x1234);
  let frame = ayuv64_frame(&packed);

  let mut rgba = std::vec![0u8; OUT * OUT * 4];
  let mut rgba_u16 = std::vec![0u16; OUT * OUT * 4];
  let mut luma_u16 = std::vec![0u16; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Ayuv64, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_alpha_mode(AlphaMode::Premultiplied)
        .with_rgba(&mut rgba)
        .unwrap()
        .with_rgba_u16(&mut rgba_u16)
        .unwrap()
        .with_luma_u16(&mut luma_u16)
        .unwrap();
    ayuv64_to(&frame, FR, sink.set_kernel_matrix(M)).unwrap();
  }

  // u8 premult: premult at 255 → bin → un-premult at 255.
  let mut pm8 = direct_rgba_u8(&packed, FR);
  premultiply(&mut pm8, 255);
  let oracle8 = unpremultiply(&block_mean_rgba_u8(&pm8), 255);
  assert_eq!(rgba, oracle8, "premult u8 rgba");

  // u16 premult: premult at 65535 → bin → un-premult at 65535 (independent).
  let mut pm16 = direct_rgba_u16(&packed, FR);
  premultiply(&mut pm16, MAX16);
  let oracle16 = unpremultiply(&block_mean_rgba_u16(&pm16), MAX16);
  assert_eq!(rgba_u16, oracle16, "premult u16 rgba");

  // luma_u16 under premult is the native-Y bin (alpha-independent).
  assert_eq!(
    luma_u16,
    block_mean_native_y(&packed),
    "premult luma_u16 == native-Y bin"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn ayuv64_premultiplied_transparent_block_does_not_bleed() {
  let mut packed = packed_frame(0xABCD);
  for off in [(0, 0), (1, 0), (0, 1), (1, 1)] {
    let i = off.1 * SRC + off.0;
    packed[i * 4] = 0; // A
    packed[i * 4 + 1] = 60000; // Y
  }
  let frame = ayuv64_frame(&packed);
  let mut rgba_u16 = std::vec![0u16; OUT * OUT * 4];
  {
    let mut sink =
      MixedSinker::<Ayuv64, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_alpha_mode(AlphaMode::Premultiplied)
        .with_rgba_u16(&mut rgba_u16)
        .unwrap();
    ayuv64_to(&frame, FR, sink.set_kernel_matrix(M)).unwrap();
  }
  assert_eq!(
    &rgba_u16[..4],
    &[0, 0, 0, 0],
    "transparent block bled colour"
  );
  let mut pm16 = direct_rgba_u16(&packed, FR);
  premultiply(&mut pm16, MAX16);
  let oracle16 = unpremultiply(&block_mean_rgba_u16(&pm16), MAX16);
  assert_eq!(rgba_u16, oracle16, "premult u16 output != oracle");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn ayuv64_premultiplied_nonuniform_alpha_luma_is_native_y_not_colour() {
  // The Y/alpha anti-correlated counterexample tiled over every 2x2 block:
  //   (Y, A) = (0, MAX), (MAX, 0), (0, MAX), (MAX, 0); native-Y mean ≈ MAX/2.
  // The premultiplied colour collapses to mean(Y*A)/mean(A) ≈ 0, so a
  // colour-derived luma would be ≈0 (the bug). Native-Y luma must be MAX/2.
  let mut packed = std::vec![0u16; SRC * SRC * 4];
  for (i, px) in packed.chunks_exact_mut(4).enumerate() {
    let odd = (i % SRC) % 2 == 1;
    px[0] = if odd { 0 } else { 0xFFFF }; // A complement
    px[1] = if odd { 0xFFFF } else { 0 }; // Y
    px[2] = 32768; // U neutral
    px[3] = 32768; // V neutral
  }
  let frame = ayuv64_frame(&packed);
  let mut luma = std::vec![0u8; OUT * OUT];
  let mut luma_u16 = std::vec![0u16; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Ayuv64, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_alpha_mode(AlphaMode::Premultiplied)
        .with_luma(&mut luma)
        .unwrap()
        .with_luma_u16(&mut luma_u16)
        .unwrap();
    ayuv64_to(&frame, FR, sink.set_kernel_matrix(M)).unwrap();
  }
  let y_binned = block_mean_native_y(&packed);
  // (0 + 65535 + 0 + 65535 + 2) / 4 = 32768.
  assert!(y_binned.iter().all(|&y| y == 32768), "native-Y bin sanity");
  assert_eq!(luma_u16, y_binned, "premult luma_u16 == native-Y bin");
  let luma_ref: Vec<u8> = y_binned.iter().map(|&p| (p >> 8) as u8).collect();
  assert_eq!(luma, luma_ref, "premult luma == native-Y >> 8");

  // Guard: the colour-derived luma (un-premultiplied straight R) is ≈0.
  let mut pm16 = direct_rgba_u16(&packed, FR);
  premultiply(&mut pm16, MAX16);
  let color_oracle = unpremultiply(&block_mean_rgba_u16(&pm16), MAX16);
  let color_r: Vec<u16> = color_oracle.chunks_exact(4).map(|px| px[0]).collect();
  assert!(
    color_r.iter().all(|&r| r < 1000),
    "fixture failed to exercise divergence: {color_r:?}"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn ayuv64_limited_range_luma_is_native_y() {
  // Native Y is range-independent: luma at limited range equals luma at full
  // range (both the native Y), unlike a range-derived `rgb_to_luma`.
  let packed = packed_frame(0xCAFE);
  let frame = ayuv64_frame(&packed);
  let render = |full_range: bool| {
    let mut luma_u16 = std::vec![0u16; OUT * OUT];
    let mut sink =
      MixedSinker::<Ayuv64, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_luma_u16(&mut luma_u16)
        .unwrap();
    ayuv64_to(&frame, full_range, sink.set_kernel_matrix(M)).unwrap();
    luma_u16
  };
  let lim = render(FR_LIMITED);
  let full = render(FR);
  assert_eq!(
    lim,
    block_mean_native_y(&packed),
    "limited-range luma_u16 == native-Y bin"
  );
  assert_eq!(lim, full, "native-Y luma_u16 must be range-independent");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn ayuv64_straight_and_premult_differ_under_varying_alpha() {
  let mut packed = packed_frame(0x77AA);
  for (i, px) in packed.chunks_exact_mut(4).enumerate() {
    px[0] = (i as u16).wrapping_mul(900).wrapping_add(1000); // A varies
  }
  let render = |mode: AlphaMode| {
    let frame = ayuv64_frame(&packed);
    let mut rgba_u16 = std::vec![0u16; OUT * OUT * 4];
    let mut sink =
      MixedSinker::<Ayuv64, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_alpha_mode(mode)
        .with_rgba_u16(&mut rgba_u16)
        .unwrap();
    ayuv64_to(&frame, FR, sink.set_kernel_matrix(M)).unwrap();
    rgba_u16
  };
  assert_ne!(
    render(AlphaMode::Straight),
    render(AlphaMode::Premultiplied),
    "alpha mode had no effect"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn ayuv64_le_be_resample_byte_identical() {
  // The same logical samples encoded LE vs BE, fed through `Ayuv64<false>`
  // vs `Ayuv64<true>`, must produce byte-identical resampled output for the
  // u8 + u16 colour AND luma paths (the `<const BE>` propagation through the
  // resample decode closures).
  let logical: Vec<u16> = (0..SRC * SRC * 4)
    .map(|i| match i % 4 {
      0 => 0xABCDu16, // A
      1 => 0x8000u16, // Y
      2 => 0x4000u16, // U
      _ => 0xC000u16, // V
    })
    .collect();
  let pix_le: Vec<u16> = logical.iter().map(|&v| super::as_le_u16(v)).collect();
  let pix_be: Vec<u16> = logical.iter().map(|&v| super::as_be_u16(v)).collect();

  let render_le = || {
    let frame = Ayuv64LeFrame::try_new(&pix_le, SRC as u32, SRC as u32, (SRC * 4) as u32).unwrap();
    let mut rgba = std::vec![0u8; OUT * OUT * 4];
    let mut rgba_u16 = std::vec![0u16; OUT * OUT * 4];
    let mut luma_u16 = std::vec![0u16; OUT * OUT];
    let mut sink =
      MixedSinker::<Ayuv64, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap()
        .with_rgba_u16(&mut rgba_u16)
        .unwrap()
        .with_luma_u16(&mut luma_u16)
        .unwrap();
    ayuv64_to(&frame, FR, sink.set_kernel_matrix(M)).unwrap();
    (rgba, rgba_u16, luma_u16)
  };
  let render_be = || {
    let frame = Ayuv64BeFrame::try_new(&pix_be, SRC as u32, SRC as u32, (SRC * 4) as u32).unwrap();
    let mut rgba = std::vec![0u8; OUT * OUT * 4];
    let mut rgba_u16 = std::vec![0u16; OUT * OUT * 4];
    let mut luma_u16 = std::vec![0u16; OUT * OUT];
    let mut sink = MixedSinker::<Ayuv64<true>, AreaResampler>::with_resampler(
      SRC,
      SRC,
      AreaResampler::to(OUT, OUT),
    )
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap()
    .with_rgba_u16(&mut rgba_u16)
    .unwrap()
    .with_luma_u16(&mut luma_u16)
    .unwrap();
    ayuv64_to_endian(&frame, FR, sink.set_kernel_matrix(M)).unwrap();
    (rgba, rgba_u16, luma_u16)
  };
  assert_eq!(
    render_le(),
    render_be(),
    "AYUV64 LE/BE resample outputs diverge"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn ayuv64_identity_plan_matches_direct() {
  let packed = packed_frame(0x0F0F);
  let mut rgba_u16 = std::vec![0u16; SRC * SRC * 4];
  {
    let mut sink =
      MixedSinker::<Ayuv64, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(SRC, SRC))
        .unwrap()
        .with_rgba_u16(&mut rgba_u16)
        .unwrap();
    ayuv64_to(&ayuv64_frame(&packed), FR, sink.set_kernel_matrix(M)).unwrap();
  }
  assert_eq!(
    rgba_u16,
    direct_rgba_u16(&packed, FR),
    "identity plan == direct"
  );
}

#[test]
fn ayuv64_default_alpha_mode_is_straight() {
  let sink = MixedSinker::<Ayuv64>::new(SRC, SRC);
  assert_eq!(sink.alpha_mode(), AlphaMode::Straight);
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn ayuv64_cross_frame_reset_reuses_streams() {
  let packed = packed_frame(0x5151);
  let frame = ayuv64_frame(&packed);
  let mut rgba_u16 = std::vec![0u16; OUT * OUT * 4];
  {
    let mut sink =
      MixedSinker::<Ayuv64, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgba_u16(&mut rgba_u16)
        .unwrap();
    ayuv64_to(&frame, FR, sink.set_kernel_matrix(M)).unwrap();
    ayuv64_to(&frame, FR, sink.set_kernel_matrix(M)).unwrap();
  }
  assert_eq!(rgba_u16, block_mean_rgba_u16(&direct_rgba_u16(&packed, FR)));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn ayuv64_accepts_alpha_mode_change_across_frames() {
  let packed = packed_frame(0xB2B2);
  let frame = ayuv64_frame(&packed);
  let mut rgba_u16 = std::vec![0u16; OUT * OUT * 4];
  {
    let mut sink =
      MixedSinker::<Ayuv64, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgba_u16(&mut rgba_u16)
        .unwrap();
    ayuv64_to(&frame, FR, sink.set_kernel_matrix(M)).unwrap();
    sink.set_alpha_mode(AlphaMode::Premultiplied);
    ayuv64_to(&frame, FR, sink.set_kernel_matrix(M))
      .expect("a fresh frame must accept a different alpha mode");
  }
  let mut pm16 = direct_rgba_u16(&packed, FR);
  premultiply(&mut pm16, MAX16);
  let oracle16 = unpremultiply(&block_mean_rgba_u16(&pm16), MAX16);
  assert_eq!(rgba_u16, oracle16, "premult frame 2 output");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn ayuv64_mid_frame_alpha_mode_flip_is_rejected() {
  let packed = packed_frame(0x33AA);
  let row_elems = SRC * 4;
  let mut rgba_u16 = std::vec![0u16; OUT * OUT * 4];
  let mut sink =
    MixedSinker::<Ayuv64, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_rgba_u16(&mut rgba_u16)
      .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  sink
    .process(Ayuv64Row::for_tests(&packed[..row_elems], 0, M, FR))
    .unwrap();
  sink.set_alpha_mode(AlphaMode::Premultiplied);
  let err = sink
    .process(Ayuv64Row::for_tests(
      &packed[row_elems..2 * row_elems],
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
fn ayuv64_out_of_sequence_first_row_is_rejected() {
  let packed = packed_frame(0x44BB);
  let row_elems = SRC * 4;
  let mut rgba_u16 = std::vec![0u16; OUT * OUT * 4];
  let mut sink =
    MixedSinker::<Ayuv64, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_rgba_u16(&mut rgba_u16)
      .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  let err = sink
    .process(Ayuv64Row::for_tests(
      &packed[row_elems..2 * row_elems],
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
  assert!(
    rgba_u16.iter().all(|&p| p == 0),
    "rejected row mutated output"
  );
}

#[test]
fn ayuv64_no_output_sink_is_a_noop() {
  let packed = packed_frame(0x4242);
  let frame = ayuv64_frame(&packed);
  let mut sink =
    MixedSinker::<Ayuv64, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap();
  ayuv64_to(&frame, FR, sink.set_kernel_matrix(M)).unwrap();
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn ayuv64_resample_simd_matches_scalar() {
  let packed = packed_frame(0x2468);
  let run = |simd: bool| {
    let frame = ayuv64_frame(&packed);
    let mut rgb = std::vec![0u8; OUT * OUT * 3];
    let mut rgb_u16 = std::vec![0u16; OUT * OUT * 3];
    let mut luma_u16 = std::vec![0u16; OUT * OUT];
    let mut sink =
      MixedSinker::<Ayuv64, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_simd(simd)
        .with_rgb(&mut rgb)
        .unwrap()
        .with_rgb_u16(&mut rgb_u16)
        .unwrap()
        .with_luma_u16(&mut luma_u16)
        .unwrap();
    ayuv64_to(&frame, FR, sink.set_kernel_matrix(M)).unwrap();
    (rgb, rgb_u16, luma_u16)
  };
  assert_eq!(run(true), run(false), "Ayuv64 resample SIMD != scalar");
}
