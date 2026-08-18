//! Fused-downscale coverage for the packed 8-bit 4:4:4 YUV source `Vuyx`
//! (`[V, U, Y, X]` byte quadruple, the X byte padding / forced-opaque α).
//!
//! `Vuyx` has NO alpha, so it routes exactly like its no-alpha siblings
//! (`V30X` / `V410` / `Xv36`) through the three-stream tail
//! ([`packed_yuv444_triple_resample`](super::super::packed_yuv444_triple_resample))
//! at `SRC_BITS = 8` — u8 colour bins the converted u8 RGB row and luma
//! bins the de-interleaved native Y (the X padding byte is never read).
//! Each output is byte-identical to a direct conversion of the
//! area-downscaled frame:
//! - rgb / rgba / hsv == the round-half-up 2x2 block mean of the direct u8
//!   `Vuyx → RGB` conversion (rgba α forced to `0xFF`);
//! - luma_u16 == the block mean of the native Y zero-extended to u16; luma
//!   is that binned Y narrowed `>> 0` (an 8-bit source — a pass-through).
//!
//! `Vuyx` exposes no u16 colour outputs, so the u16 colour binning is never
//! active here. Oracles are built from `Vuyx`'s own direct kernels only.

use crate::{
  KernelMatrix, PixelSink,
  frame::VuyxFrame,
  resample::{AreaResampler, ResampleError},
  sinker::{MixedSinker, MixedSinkerError},
  source::{Vuyx, VuyxRow, vuyx_to},
};

use super::force_row_stage;

const SRC: usize = 8;
const OUT: usize = 4;
const M: KernelMatrix = KernelMatrix::Bt601;
const FR: bool = true;

/// Packs logical `(V, U, Y, X)` byte planes into a `Vuyx` row (X is
/// padding; set to a recognizable non-`0xFF` value so a path that
/// mistakenly read it as alpha would diverge).
fn pack_vuyx(v: &[u8], u: &[u8], y: &[u8]) -> Vec<u8> {
  let mut out = std::vec![0u8; v.len() * 4];
  for i in 0..v.len() {
    out[i * 4] = v[i];
    out[i * 4 + 1] = u[i];
    out[i * 4 + 2] = y[i];
    out[i * 4 + 3] = 0x5A; // padding — must be ignored
  }
  out
}

fn vuyx_frame(buf: &[u8]) -> VuyxFrame<'_> {
  VuyxFrame::try_new(buf, SRC as u32, SRC as u32, (SRC * 4) as u32).unwrap()
}

/// A per-channel ramp (interior values so the conversions see real math).
fn yuv_ramp() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let mut v = std::vec![0u8; SRC * SRC];
  let mut u = std::vec![0u8; SRC * SRC];
  let mut y = std::vec![0u8; SRC * SRC];
  for i in 0..SRC * SRC {
    y[i] = 40u8.wrapping_add((i as u8).wrapping_mul(3));
    u[i] = 90u8.wrapping_add((i as u8).wrapping_mul(2));
    v[i] = 200u8.wrapping_sub(i as u8);
  }
  (v, u, y)
}

/// Round-half-up 2x2 block mean of an `SRC`-grid 3-channel `u8` RGB plane.
fn block_mean_rgb_u8(rgb: &[u8]) -> Vec<u8> {
  let mut out = std::vec![0u8; OUT * OUT * 3];
  for oy in 0..OUT {
    for ox in 0..OUT {
      for c in 0..3 {
        let mut s = 0u32;
        for dy in 0..2 {
          for dx in 0..2 {
            s += rgb[((oy * 2 + dy) * SRC + ox * 2 + dx) * 3 + c] as u32;
          }
        }
        out[(oy * OUT + ox) * 3 + c] = ((s + 2) / 4) as u8;
      }
    }
  }
  out
}

/// Round-half-up 2x2 block mean of the native Y plane (`Y = packed[4*i+2]`).
fn block_mean_native_y(packed: &[u8]) -> Vec<u8> {
  let mut out = std::vec![0u8; OUT * OUT];
  for oy in 0..OUT {
    for ox in 0..OUT {
      let mut s = 0u32;
      for dy in 0..2 {
        for dx in 0..2 {
          s += packed[((oy * 2 + dy) * SRC + ox * 2 + dx) * 4 + 2] as u32;
        }
      }
      out[oy * OUT + ox] = ((s + 2) / 4) as u8;
    }
  }
  out
}

/// Direct full-res `Vuyx` u8 RGB of a packed frame.
fn direct_rgb_u8(packed: &[u8]) -> Vec<u8> {
  let src = vuyx_frame(packed);
  let mut rgb = std::vec![0u8; SRC * SRC * 3];
  {
    let mut sink = MixedSinker::<Vuyx>::new(SRC, SRC)
      .with_rgb(&mut rgb)
      .unwrap();
    vuyx_to(&src, FR, M, &mut sink).unwrap();
  }
  rgb
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn vuyx_uniform_gray_downscale_leaves_colour_unchanged() {
  // The area mean of a constant frame is the constant: every colour output
  // must equal the direct conversion of the same uniform frame.
  let v = std::vec![140u8; SRC * SRC];
  let u = std::vec![120u8; SRC * SRC];
  let y = std::vec![180u8; SRC * SRC];
  let packed = pack_vuyx(&v, &u, &y);
  let full_rgb = direct_rgb_u8(&packed);

  let mut rgb = std::vec![0u8; OUT * OUT * 3];
  let mut rgba = std::vec![0u8; OUT * OUT * 4];
  let mut hh = std::vec![0u8; OUT * OUT];
  let mut ss = std::vec![0u8; OUT * OUT];
  let mut vv = std::vec![0u8; OUT * OUT];
  {
    let mut sink = force_row_stage(
      MixedSinker::<Vuyx, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap(),
    )
    .with_rgb(&mut rgb)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap()
    .with_hsv(&mut hh, &mut ss, &mut vv)
    .unwrap();
    vuyx_to(&vuyx_frame(&packed), FR, M, &mut sink).unwrap();
  }
  let gray_px = &full_rgb[..3];
  for px in rgb.chunks_exact(3) {
    assert_eq!(px, gray_px, "uniform-gray u8 RGB changed under downscale");
  }
  for px in rgba.chunks_exact(4) {
    assert_eq!(&px[..3], gray_px, "uniform-gray rgba colour changed");
    assert_eq!(px[3], 0xFF, "rgba alpha must be forced opaque");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn vuyx_all_outputs_match_their_own_block_mean() {
  let (v, u, y) = yuv_ramp();
  let packed = pack_vuyx(&v, &u, &y);
  let full_rgb = direct_rgb_u8(&packed);

  let mut rgb = std::vec![0u8; OUT * OUT * 3];
  let mut rgba = std::vec![0u8; OUT * OUT * 4];
  let mut luma = std::vec![0u8; OUT * OUT];
  let mut luma_u16 = std::vec![0u16; OUT * OUT];
  let mut hh = std::vec![0u8; OUT * OUT];
  let mut ss = std::vec![0u8; OUT * OUT];
  let mut vv = std::vec![0u8; OUT * OUT];
  {
    let mut sink = force_row_stage(
      MixedSinker::<Vuyx, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap(),
    )
    .with_rgb(&mut rgb)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap()
    .with_luma(&mut luma)
    .unwrap()
    .with_luma_u16(&mut luma_u16)
    .unwrap()
    .with_hsv(&mut hh, &mut ss, &mut vv)
    .unwrap();
    vuyx_to(&vuyx_frame(&packed), FR, M, &mut sink).unwrap();
  }

  let rgb_ref = block_mean_rgb_u8(&full_rgb);
  assert_eq!(rgb, rgb_ref, "rgb");
  for (px, c) in rgba.chunks_exact(4).zip(rgb_ref.chunks_exact(3)) {
    assert_eq!(&px[..3], c, "rgba colour");
    assert_eq!(px[3], 0xFF, "rgba alpha forced opaque");
  }
  // luma == native-Y bin; luma_u16 == zero-extension of the same.
  let y_binned = block_mean_native_y(&packed);
  assert_eq!(luma, y_binned, "luma == native-Y block mean");
  let lu16_ref: Vec<u16> = y_binned.iter().map(|&p| p as u16).collect();
  assert_eq!(luma_u16, lu16_ref, "luma_u16 == native-Y zero-extended");

  // HSV from the binned u8 RGB.
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
fn vuyx_luma_taken_from_native_y_under_saturated_chroma() {
  // Constant Y, extreme U/V: the area-downscaled native Y is the constant
  // Y. RGB-derived luma would clamp away from it; native-Y luma must stay
  // exactly the constant.
  let yc = 128u8;
  let v = std::vec![250u8; SRC * SRC];
  let u = std::vec![5u8; SRC * SRC];
  let y = std::vec![yc; SRC * SRC];
  let packed = pack_vuyx(&v, &u, &y);
  let mut luma = std::vec![0u8; OUT * OUT];
  let mut luma_u16 = std::vec![0u16; OUT * OUT];
  {
    let mut sink = force_row_stage(
      MixedSinker::<Vuyx, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap(),
    )
    .with_luma(&mut luma)
    .unwrap()
    .with_luma_u16(&mut luma_u16)
    .unwrap();
    vuyx_to(&vuyx_frame(&packed), FR, M, &mut sink).unwrap();
  }
  assert!(
    luma.iter().all(|&p| p == yc),
    "luma must be native Y {yc}, got {luma:?}"
  );
  assert!(
    luma_u16.iter().all(|&p| p == yc as u16),
    "luma_u16 must be native Y {yc}, got {luma_u16:?}"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn vuyx_limited_range_luma_is_native_y_not_rgb_scaled() {
  // A uniform Y = 16 limited-range gray. The direct `Vuyx` luma is the
  // native Y byte (16); a limited-range `rgb_to_luma_row` of (16,16,16)
  // would scale it up. Native-Y luma must be range-independent.
  let v = std::vec![128u8; SRC * SRC];
  let u = std::vec![128u8; SRC * SRC];
  let y = std::vec![16u8; SRC * SRC];
  let packed = pack_vuyx(&v, &u, &y);
  let mut luma = std::vec![0u8; OUT * OUT];
  {
    let mut sink = force_row_stage(
      MixedSinker::<Vuyx, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap(),
    )
    .with_luma(&mut luma)
    .unwrap();
    vuyx_to(&vuyx_frame(&packed), false, M, &mut sink).unwrap();
  }
  assert!(
    luma.iter().all(|&p| p == 16),
    "limited-range Y=16 luma must stay 16, got {luma:?}"
  );
}

// The fractional-ratio reference reuses the packed-RGB source as an
// independent area-engine oracle, so it is gated on `rgb` (its frame /
// walker live there). The integer-ratio block-mean tests above cover the
// fused arithmetic in a `yuv-444-packed`-solo build.
#[cfg(feature = "rgb")]
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn vuyx_fractional_ratio_matches_direct_then_bin() {
  // 8 -> 3 fractional downscale: the resampled rgb must equal a direct
  // full-res convert fed through the SAME AreaResampler over the u8 RGB.
  const F: usize = 3;
  let (v, u, y) = yuv_ramp();
  let packed = pack_vuyx(&v, &u, &y);

  let mut rgb = std::vec![0u8; F * F * 3];
  {
    let mut sink = force_row_stage(
      MixedSinker::<Vuyx, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(F, F))
        .unwrap(),
    )
    .with_rgb(&mut rgb)
    .unwrap();
    vuyx_to(&vuyx_frame(&packed), FR, M, &mut sink).unwrap();
  }
  // Reference: feed the direct full-res RGB through the packed-RGB path at
  // the same plan (the area engine is the trusted source of truth here).
  let full_rgb = direct_rgb_u8(&packed);
  let mut rgb_ref = std::vec![0u8; F * F * 3];
  {
    use crate::{frame::Rgb24Frame, source::rgb24_to};
    let rsrc = Rgb24Frame::try_new(&full_rgb, SRC as u32, SRC as u32, (SRC * 3) as u32).unwrap();
    let mut sink = force_row_stage(
      MixedSinker::<crate::source::Rgb24, AreaResampler>::with_resampler(
        SRC,
        SRC,
        AreaResampler::to(F, F),
      )
      .unwrap(),
    )
    .with_rgb(&mut rgb_ref)
    .unwrap();
    rgb24_to(&rsrc, FR, M, &mut sink).unwrap();
  }
  assert_eq!(
    rgb, rgb_ref,
    "Vuyx 8->3 != packed-RGB 8->3 of the direct RGB"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn vuyx_identity_plan_matches_direct() {
  let (v, u, y) = yuv_ramp();
  let packed = pack_vuyx(&v, &u, &y);
  let direct = direct_rgb_u8(&packed);
  let mut via_area = std::vec![0u8; SRC * SRC * 3];
  {
    let mut sink = force_row_stage(
      MixedSinker::<Vuyx, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(SRC, SRC))
        .unwrap(),
    )
    .with_rgb(&mut via_area)
    .unwrap();
    vuyx_to(&vuyx_frame(&packed), FR, M, &mut sink).unwrap();
  }
  assert_eq!(direct, via_area, "identity plan must match the direct sink");
}

#[test]
fn vuyx_no_outputs_is_a_no_op() {
  let (v, u, y) = yuv_ramp();
  let packed = pack_vuyx(&v, &u, &y);
  let mut sink =
    MixedSinker::<Vuyx, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap();
  vuyx_to(&vuyx_frame(&packed), FR, M, &mut sink).unwrap();
  assert!(
    !sink.rgb_stream_allocated(),
    "u8 stream allocated for a no-op"
  );
  assert!(
    !sink.luma_stream_u16_allocated(),
    "luma stream allocated for a no-op"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn vuyx_resets_streams_across_frames() {
  let (v, u, y1) = yuv_ramp();
  let mut y2 = y1.clone();
  for p in y2.iter_mut() {
    *p = 255 - *p;
  }
  let p1 = pack_vuyx(&v, &u, &y1);
  let p2 = pack_vuyx(&v, &u, &y2);
  let mut luma_u16 = std::vec![0u16; OUT * OUT];
  {
    let mut sink = force_row_stage(
      MixedSinker::<Vuyx, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap(),
    )
    .with_luma_u16(&mut luma_u16)
    .unwrap();
    vuyx_to(&vuyx_frame(&p1), FR, M, &mut sink).unwrap();
    vuyx_to(&vuyx_frame(&p2), FR, M, &mut sink).unwrap();
  }
  let y_binned = block_mean_native_y(&p2);
  let lu16_ref: Vec<u16> = y_binned.iter().map(|&p| p as u16).collect();
  assert_eq!(
    luma_u16, lu16_ref,
    "frame 2 luma_u16 must area-downscale frame 2's Y"
  );
}

#[test]
fn vuyx_out_of_sequence_first_row_rejected_before_allocation() {
  let (v, u, y) = yuv_ramp();
  let packed = pack_vuyx(&v, &u, &y);
  let row3 = &packed[3 * SRC * 4..4 * SRC * 4];
  let mut rgb = std::vec![0u8; OUT * OUT * 3];
  let mut luma_u16 = std::vec![0u16; OUT * OUT];
  let mut sink = force_row_stage(
    MixedSinker::<Vuyx, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap(),
  )
  .with_rgb(&mut rgb)
  .unwrap()
  .with_luma_u16(&mut luma_u16)
  .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  let err = sink.process(VuyxRow::new(row3, 3, M, FR)).unwrap_err();
  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
    ),
    "expected OutOfSequenceRow, got {err:?}"
  );
  assert!(
    !sink.rgb_stream_allocated() && !sink.luma_stream_u16_allocated(),
    "stream allocated for a rejected first row"
  );
  assert!(
    rgb.iter().all(|&b| b == 0) && luma_u16.iter().all(|&p| p == 0),
    "rejected row mutated output"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn vuyx_rejects_mid_frame_output_change() {
  let (v, u, y) = yuv_ramp();
  let packed = pack_vuyx(&v, &u, &y);
  let mut rgb = std::vec![0u8; OUT * OUT * 3];
  let mut luma_u16 = std::vec![0u16; OUT * OUT];
  let mut sink = force_row_stage(
    MixedSinker::<Vuyx, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap(),
  )
  .with_rgb(&mut rgb)
  .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  sink
    .process(VuyxRow::new(&packed[..SRC * 4], 0, M, FR))
    .unwrap();
  sink.set_luma_u16(&mut luma_u16).unwrap();
  let err = sink
    .process(VuyxRow::new(&packed[SRC * 4..2 * SRC * 4], 1, M, FR))
    .unwrap_err();
  assert!(
    matches!(err, MixedSinkerError::ResampleOutputsChanged(_)),
    "expected ResampleOutputsChanged, got {err:?}"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn vuyx_resample_simd_matches_scalar() {
  let (v, u, y) = yuv_ramp();
  let packed = pack_vuyx(&v, &u, &y);
  let run = |simd: bool| {
    let mut rgb = std::vec![0u8; OUT * OUT * 3];
    let mut luma_u16 = std::vec![0u16; OUT * OUT];
    let mut sink = force_row_stage(
      MixedSinker::<Vuyx, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap(),
    )
    .with_simd(simd)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_luma_u16(&mut luma_u16)
    .unwrap();
    vuyx_to(&vuyx_frame(&packed), FR, M, &mut sink).unwrap();
    (rgb, luma_u16)
  };
  assert_eq!(run(true), run(false), "Vuyx resample SIMD != scalar");
}
