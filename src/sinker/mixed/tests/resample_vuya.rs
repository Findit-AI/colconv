//! Alpha-aware fused-downscale coverage for the packed 8-bit 4:4:4 YUV
//! source with real source alpha, `Vuya` (`[V, U, Y, A]` byte quadruple,
//! the A byte real alpha).
//!
//! `Vuya` routes through the packed-YUVA tail
//! ([`packed_yuva444_resample`](super::super::packed_yuva444_resample)) at
//! `SRC_BITS = 8`: the u8 colour stream bins the converted u8 RGBA row
//! (`vuya_to_rgba_row` — real source α), and the native-Y luma stream bins
//! the Y bytes. Each output is byte-identical to a direct convert-then-bin:
//! - straight rgba is the 2x2 block mean of a direct full-res
//!   `vuya_to_rgba_row` (alpha averaged, NOT forced opaque);
//! - premultiplied bins premultiplied colour and un-premultiplies
//!   (transparent pixels never bleed);
//! - rgb drops alpha and hsv derives from the binned colour;
//! - native-Y luma: `luma` is the binned Y byte and `luma_u16` its
//!   zero-extension, from an INDEPENDENT native-Y area bin — NEVER the
//!   alpha- / range-affected colour (byte-exact for every matrix, both
//!   ranges, AND every alpha mode). `Vuya` exposes no u16 colour outputs.

use crate::{
  KernelMatrix, PixelSink,
  frame::VuyaFrame,
  resample::{AreaResampler, ResampleError},
  sinker::{AlphaMode, MixedSinker, MixedSinkerError},
  source::{Vuya, VuyaRow, vuya_to},
};

const SRC: usize = 8;
const OUT: usize = 4;
const M: KernelMatrix = KernelMatrix::Bt709;
const FR: bool = true;
const FR_LIMITED: bool = false;

/// Pseudo-random packed `[V, U, Y, A]` plane (`SRC * SRC * 4` bytes);
/// alpha varies (not all-opaque).
fn packed_frame(seed: u32) -> Vec<u8> {
  let mut buf = std::vec![0u8; SRC * SRC * 4];
  super::pseudo_random_u8(&mut buf, seed);
  buf
}

/// Full-resolution canonical RGBA of the source — a direct (identity)
/// `Vuya` conversion via `vuya_to_rgba_row`. The oracles bin / premultiply
/// this.
fn direct_rgba(packed: &[u8]) -> Vec<u8> {
  let frame = VuyaFrame::try_new(packed, SRC as u32, SRC as u32, (SRC * 4) as u32).unwrap();
  let mut rgba = std::vec![0u8; SRC * SRC * 4];
  {
    let mut sink = MixedSinker::<Vuya>::new(SRC, SRC)
      .with_rgba(&mut rgba)
      .unwrap();
    vuya_to(&frame, FR, sink.set_kernel_matrix(M)).unwrap();
  }
  rgba
}

/// Round-half-up 2x2 block mean of a canonical RGBA plane (every channel,
/// alpha included).
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

/// Round-half-up 2x2 block mean of the native Y plane (`Y = packed[4*i+2]`)
/// — the alpha-independent native-Y oracle (`mean(Y)`, NOT the colour
/// path's `mean(Y*A)/mean(A)`).
fn block_mean_native_y(packed: &[u8]) -> Vec<u8> {
  let mut out = std::vec![0u8; OUT * OUT];
  for oy in 0..OUT {
    for ox in 0..OUT {
      let mut acc = 0u32;
      for dy in 0..2 {
        for dx in 0..2 {
          acc += packed[((oy * 2 + dy) * SRC + ox * 2 + dx) * 4 + 2] as u32;
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

/// Direct (identity) `Vuya` luma / luma_u16 of a binned-Y plane at the given
/// range — the byte-exact native-Y oracle. Y is range-independent, so this
/// is `binned_y` and its zero-extension; built via the real `Vuya` kernels.
fn direct_luma_of_binned_y(binned_y: &[u8], full_range: bool) -> (Vec<u8>, Vec<u16>) {
  let n = binned_y.len();
  let mut packed = std::vec![0u8; n * 4];
  for (i, &y) in binned_y.iter().enumerate() {
    packed[i * 4 + 2] = y; // Y at offset 2
    packed[i * 4 + 3] = 0xFF; // A (irrelevant to luma)
  }
  let frame = VuyaFrame::try_new(&packed, n as u32, 1, (n * 4) as u32).unwrap();
  let mut luma = std::vec![0u8; n];
  let mut lu16 = std::vec![0u16; n];
  {
    let mut sink = MixedSinker::<Vuya>::new(n, 1)
      .with_luma(&mut luma)
      .unwrap()
      .with_luma_u16(&mut lu16)
      .unwrap();
    vuya_to(&frame, full_range, sink.set_kernel_matrix(M)).unwrap();
  }
  (luma, lu16)
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn vuya_straight_rgba_is_block_mean_of_direct() {
  let packed = packed_frame(0x51A1);
  let frame = VuyaFrame::try_new(&packed, SRC as u32, SRC as u32, (SRC * 4) as u32).unwrap();
  let mut rgba = std::vec![0u8; OUT * OUT * 4];
  {
    let mut sink =
      MixedSinker::<Vuya, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap();
    vuya_to(&frame, FR, sink.set_kernel_matrix(M)).unwrap();
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
fn vuya_straight_all_outputs_derive_correctly() {
  let packed = packed_frame(0xBEEF);
  let frame = VuyaFrame::try_new(&packed, SRC as u32, SRC as u32, (SRC * 4) as u32).unwrap();

  let mut rgb = std::vec![0u8; OUT * OUT * 3];
  let mut rgba = std::vec![0u8; OUT * OUT * 4];
  let mut luma = std::vec![0u8; OUT * OUT];
  let mut lu16 = std::vec![0u16; OUT * OUT];
  let mut h = std::vec![0u8; OUT * OUT];
  let mut s = std::vec![0u8; OUT * OUT];
  let mut v = std::vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Vuya, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb(&mut rgb)
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap()
        .with_luma(&mut luma)
        .unwrap()
        .with_luma_u16(&mut lu16)
        .unwrap()
        .with_hsv(&mut h, &mut s, &mut v)
        .unwrap();
    vuya_to(&frame, FR, sink.set_kernel_matrix(M)).unwrap();
  }

  let binned = block_mean_rgba(&direct_rgba(&packed));
  assert_eq!(rgba, binned, "rgba == block mean");
  let binned_rgb = drop_alpha(&binned);
  assert_eq!(rgb, binned_rgb, "rgb == drop-alpha(binned)");

  // luma / luma_u16: native Y from the Y bytes — independent of colour.
  let y_binned = block_mean_native_y(&packed);
  let (luma_ref, lu16_ref) = direct_luma_of_binned_y(&y_binned, FR);
  assert_eq!(luma, luma_ref, "luma (native Y)");
  assert_eq!(luma, y_binned, "luma == native-Y block mean");
  assert_eq!(lu16, lu16_ref, "luma_u16 (native Y zero-extended)");

  // HSV from the binned u8 RGB.
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
  assert_eq!(v, v_ref, "hsv V");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn vuya_premultiplied_matches_premult_bin_unpremult_oracle() {
  let packed = packed_frame(0x1234);
  let frame = VuyaFrame::try_new(&packed, SRC as u32, SRC as u32, (SRC * 4) as u32).unwrap();

  let mut rgba = std::vec![0u8; OUT * OUT * 4];
  let mut rgb = std::vec![0u8; OUT * OUT * 3];
  let mut lu16 = std::vec![0u16; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Vuya, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_alpha_mode(AlphaMode::Premultiplied)
        .with_rgba(&mut rgba)
        .unwrap()
        .with_rgb(&mut rgb)
        .unwrap()
        .with_luma_u16(&mut lu16)
        .unwrap();
    vuya_to(&frame, FR, sink.set_kernel_matrix(M)).unwrap();
  }

  let mut pm = direct_rgba(&packed);
  premultiply(&mut pm);
  let binned = block_mean_rgba(&pm);
  let oracle = unpremultiply(&binned);
  assert_eq!(rgba, oracle, "premult rgba");
  assert_eq!(rgb, drop_alpha(&oracle), "premult rgb");
  // luma_u16 under premult is the area-mean of the NATIVE Y plane,
  // zero-extended — alpha-INDEPENDENT, NOT mean(Y*A)/mean(A).
  let y_binned = block_mean_native_y(&packed);
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
fn vuya_premultiplied_transparent_block_does_not_bleed() {
  let mut packed = packed_frame(0xABCD);
  for off in [(0, 0), (1, 0), (0, 1), (1, 1)] {
    let i = off.1 * SRC + off.0;
    packed[i * 4 + 2] = 250; // Y
    packed[i * 4 + 3] = 0; // A
  }
  let frame = VuyaFrame::try_new(&packed, SRC as u32, SRC as u32, (SRC * 4) as u32).unwrap();
  let mut rgba = std::vec![0u8; OUT * OUT * 4];
  {
    let mut sink =
      MixedSinker::<Vuya, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_alpha_mode(AlphaMode::Premultiplied)
        .with_rgba(&mut rgba)
        .unwrap();
    vuya_to(&frame, FR, sink.set_kernel_matrix(M)).unwrap();
  }
  assert_eq!(&rgba[..4], &[0, 0, 0, 0], "transparent block bled colour");
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
fn vuya_premultiplied_nonuniform_alpha_luma_is_native_y_not_colour() {
  // The Y/alpha anti-correlated counterexample, tiled over every 2x2 block:
  //   (Y, A) = (0, 255), (255, 0), (0, 255), (255, 0)
  // Native-Y mean = 128. The premultiplied colour R collapses to
  // mean(Y*A)/mean(A) = 0, so a colour-derived luma would be 0 (the bug).
  let mut packed = std::vec![0u8; SRC * SRC * 4];
  for (i, px) in packed.chunks_exact_mut(4).enumerate() {
    let odd = (i % SRC) % 2 == 1;
    px[0] = 128; // V neutral
    px[1] = 128; // U neutral
    px[2] = if odd { 255 } else { 0 }; // Y
    px[3] = if odd { 0 } else { 255 }; // A (complement)
  }
  let frame = VuyaFrame::try_new(&packed, SRC as u32, SRC as u32, (SRC * 4) as u32).unwrap();

  let mut luma = std::vec![0u8; OUT * OUT];
  let mut lu16 = std::vec![0u16; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Vuya, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_alpha_mode(AlphaMode::Premultiplied)
        .with_luma(&mut luma)
        .unwrap()
        .with_luma_u16(&mut lu16)
        .unwrap();
    vuya_to(&frame, FR, sink.set_kernel_matrix(M)).unwrap();
  }
  assert!(
    luma.iter().all(|&y| y == 128),
    "premult luma must be native-Y 128, got {luma:?}"
  );
  assert!(
    lu16.iter().all(|&y| y == 128),
    "premult luma_u16 must be 128, got {lu16:?}"
  );

  let y_binned = block_mean_native_y(&packed);
  assert!(y_binned.iter().all(|&y| y == 128), "native-Y bin sanity");
  let (luma_ref, lu16_ref) = direct_luma_of_binned_y(&y_binned, FR);
  assert_eq!(luma, luma_ref, "premult luma == native-Y bin oracle");
  assert_eq!(lu16, lu16_ref, "premult luma_u16 == native-Y bin oracle");

  // Guard: the colour-derived (un-premultiplied straight R) luma is 0 here.
  let mut pm = direct_rgba(&packed);
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
fn vuya_straight_and_premult_differ_under_varying_alpha() {
  let mut packed = packed_frame(0x77AA);
  for (i, px) in packed.chunks_exact_mut(4).enumerate() {
    px[3] = 16u8.wrapping_add((i as u8).wrapping_mul(5));
  }
  let render = |mode: AlphaMode| {
    let frame = VuyaFrame::try_new(&packed, SRC as u32, SRC as u32, (SRC * 4) as u32).unwrap();
    let mut rgba = std::vec![0u8; OUT * OUT * 4];
    let mut sink =
      MixedSinker::<Vuya, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_alpha_mode(mode)
        .with_rgba(&mut rgba)
        .unwrap();
    vuya_to(&frame, FR, sink.set_kernel_matrix(M)).unwrap();
    rgba
  };
  assert_ne!(
    render(AlphaMode::Straight),
    render(AlphaMode::Premultiplied),
    "alpha mode had no effect"
  );
}

#[test]
fn vuya_default_alpha_mode_is_straight() {
  let sink = MixedSinker::<Vuya>::new(SRC, SRC);
  assert_eq!(sink.alpha_mode(), AlphaMode::Straight);
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn vuya_identity_plan_matches_direct() {
  let packed = packed_frame(0x0F0F);
  let frame = VuyaFrame::try_new(&packed, SRC as u32, SRC as u32, (SRC * 4) as u32).unwrap();
  let mut rgba = std::vec![0u8; SRC * SRC * 4];
  {
    let mut sink =
      MixedSinker::<Vuya, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(SRC, SRC))
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap();
    vuya_to(&frame, FR, sink.set_kernel_matrix(M)).unwrap();
  }
  assert_eq!(rgba, direct_rgba(&packed), "identity plan == direct");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn vuya_limited_range_luma_is_native_y() {
  // Native Y is range-independent: the luma at limited range equals the luma
  // at full range (both the native Y), unlike a range-derived `rgb_to_luma`.
  let packed = packed_frame(0xCAFE);
  let frame = VuyaFrame::try_new(&packed, SRC as u32, SRC as u32, (SRC * 4) as u32).unwrap();

  let render = |full_range: bool| {
    let mut luma = std::vec![0u8; OUT * OUT];
    let mut lu16 = std::vec![0u16; OUT * OUT];
    let mut sink =
      MixedSinker::<Vuya, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_luma(&mut luma)
        .unwrap()
        .with_luma_u16(&mut lu16)
        .unwrap();
    vuya_to(&frame, full_range, sink.set_kernel_matrix(M)).unwrap();
    (luma, lu16)
  };
  let (luma_lim, lu16_lim) = render(FR_LIMITED);
  let (luma_full, lu16_full) = render(FR);
  let y_binned = block_mean_native_y(&packed);
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
fn vuya_cross_frame_reset_reuses_streams() {
  let packed = packed_frame(0x5151);
  let frame = VuyaFrame::try_new(&packed, SRC as u32, SRC as u32, (SRC * 4) as u32).unwrap();
  let mut rgba = std::vec![0u8; OUT * OUT * 4];
  {
    let mut sink =
      MixedSinker::<Vuya, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap();
    vuya_to(&frame, FR, sink.set_kernel_matrix(M)).unwrap();
    vuya_to(&frame, FR, sink.set_kernel_matrix(M)).unwrap();
  }
  assert_eq!(rgba, block_mean_rgba(&direct_rgba(&packed)));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn vuya_accepts_alpha_mode_change_across_frames() {
  let packed = packed_frame(0xB2B2);
  let frame = VuyaFrame::try_new(&packed, SRC as u32, SRC as u32, (SRC * 4) as u32).unwrap();
  let mut rgba = std::vec![0u8; OUT * OUT * 4];
  {
    let mut sink =
      MixedSinker::<Vuya, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap();
    vuya_to(&frame, FR, sink.set_kernel_matrix(M)).unwrap();
    sink.set_alpha_mode(AlphaMode::Premultiplied);
    vuya_to(&frame, FR, sink.set_kernel_matrix(M))
      .expect("a fresh frame must accept a different alpha mode");
  }
  let mut pm = direct_rgba(&packed);
  premultiply(&mut pm);
  let oracle = unpremultiply(&block_mean_rgba(&pm));
  assert_eq!(rgba, oracle, "premult frame 2 output");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn vuya_mid_frame_alpha_mode_flip_is_rejected() {
  let packed = packed_frame(0x33AA);
  let row_bytes = SRC * 4;
  let mut rgba = std::vec![0u8; OUT * OUT * 4];
  let mut sink =
    MixedSinker::<Vuya, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_rgba(&mut rgba)
      .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  sink
    .process(VuyaRow::for_tests(&packed[..row_bytes], 0, M, FR))
    .unwrap();
  sink.set_alpha_mode(AlphaMode::Premultiplied);
  let err = sink
    .process(VuyaRow::for_tests(
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
fn vuya_out_of_sequence_first_row_is_rejected() {
  let packed = packed_frame(0x44BB);
  let row_bytes = SRC * 4;
  let mut rgba = std::vec![0u8; OUT * OUT * 4];
  let mut sink =
    MixedSinker::<Vuya, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_rgba(&mut rgba)
      .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  let err = sink
    .process(VuyaRow::for_tests(
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
  assert!(rgba.iter().all(|&b| b == 0), "rejected row mutated output");
}

#[test]
fn vuya_no_output_sink_is_a_noop() {
  let packed = packed_frame(0x4242);
  let frame = VuyaFrame::try_new(&packed, SRC as u32, SRC as u32, (SRC * 4) as u32).unwrap();
  let mut sink =
    MixedSinker::<Vuya, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap();
  vuya_to(&frame, FR, sink.set_kernel_matrix(M)).unwrap();
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn vuya_resample_simd_matches_scalar() {
  let packed = packed_frame(0x1357);
  let run = |simd: bool| {
    let frame = VuyaFrame::try_new(&packed, SRC as u32, SRC as u32, (SRC * 4) as u32).unwrap();
    let mut rgba = std::vec![0u8; OUT * OUT * 4];
    let mut luma = std::vec![0u8; OUT * OUT];
    let mut sink =
      MixedSinker::<Vuya, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_simd(simd)
        .with_rgba(&mut rgba)
        .unwrap()
        .with_luma(&mut luma)
        .unwrap();
    vuya_to(&frame, FR, sink.set_kernel_matrix(M)).unwrap();
    (rgba, luma)
  };
  assert_eq!(run(true), run(false), "Vuya resample SIMD != scalar");
}
