//! Fused-downscale coverage for the high-bit packed 4:4:4 YUV source
//! `V30X` (10-bit, one u32 word per pixel, LSB padding, LE-only).
//!
//! `V30X` routes through the three-stream tail
//! ([`packed_yuv444_triple_resample`](super::super::packed_yuv444_triple_resample)):
//! u8 colour bins the converted u8 RGB row, u16 colour bins the
//! converted native-u16 RGB row, and luma bins the de-interleaved native
//! Y. Each output is therefore byte-identical to a direct conversion of
//! the area-downscaled native frame, where every output group is binned
//! at its OWN precision:
//! - rgb / rgba / hsv == the round-half-up 2x2 block mean of the direct
//!   **u8** YUV→RGB conversion (NOT a narrowing of the binned u16 — the
//!   independent u8/u16 rounding makes a single narrowed binning diverge,
//!   the regression the uniform-gray counterexample below pins).
//! - rgb_u16 / rgba_u16 == the block mean of the direct **u16** YUV→RGB
//!   conversion, taken at native 10-bit depth.
//! - luma_u16 == the block mean of the direct native **Y**; luma is that
//!   binned Y narrowed `>> 2`.
//!
//! Oracles are built from `V30X`'s own direct kernels only (no `rgb` /
//! `gray` features), plus the always-available `rgb_to_hsv_row` /
//! `expand_rgb_u16_to_rgba_u16_row` helpers.

use crate::{
  KernelMatrix, PixelSink,
  frame::V30XFrame,
  resample::{AreaResampler, ResampleError},
  sinker::{MixedSinker, MixedSinkerError},
  source::{V30X, V30XRow, v30x_to},
};

const SRC: usize = 8;
const OUT: usize = 4;
const M: KernelMatrix = KernelMatrix::Bt601;
const FR: bool = true;
const SHIFT: u32 = 2; // 10-bit native → u8.

/// Packs a logical `(U, Y, V)` 10-bit plane into V30X words:
/// `(V << 22) | (Y << 12) | (U << 2)` (2-bit LSB padding).
fn pack_v30x(u: &[u16], y: &[u16], v: &[u16]) -> Vec<u32> {
  (0..u.len())
    .map(|i| {
      let u = (u[i] & 0x3FF) as u32;
      let y = (y[i] & 0x3FF) as u32;
      let v = (v[i] & 0x3FF) as u32;
      (v << 22) | (y << 12) | (u << 2)
    })
    .collect()
}

fn v30x_frame(buf: &[u32]) -> V30XFrame<'_> {
  V30XFrame::new(buf, SRC as u32, SRC as u32, SRC as u32)
}

/// A per-channel 10-bit ramp (every value interior so the conversions
/// see real math and the wide accumulator carries low bits a u8 path
/// would drop).
fn yuv_ramp() -> (Vec<u16>, Vec<u16>, Vec<u16>) {
  let mut u = vec![0u16; SRC * SRC];
  let mut y = vec![0u16; SRC * SRC];
  let mut v = vec![0u16; SRC * SRC];
  for i in 0..SRC * SRC {
    y[i] = 200 + (i as u16) * 9;
    u[i] = 350 + (i as u16) * 5;
    v[i] = 800 - (i as u16) * 4;
  }
  (u, y, v)
}

/// Round-half-up 2x2 block mean of an `SRC`-grid 3-channel `u8` RGB
/// plane to the `OUT` grid.
fn block_mean_rgb_u8(rgb: &[u8]) -> Vec<u8> {
  let mut out = vec![0u8; OUT * OUT * 3];
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

/// Round-half-up 2x2 block mean of an `SRC`-grid 3-channel `u16` RGB
/// plane to the `OUT` grid (native-depth accumulator).
fn block_mean_rgb_u16(rgb: &[u16]) -> Vec<u16> {
  let mut out = vec![0u16; OUT * OUT * 3];
  for oy in 0..OUT {
    for ox in 0..OUT {
      for c in 0..3 {
        let mut s = 0u64;
        for dy in 0..2 {
          for dx in 0..2 {
            s += rgb[((oy * 2 + dy) * SRC + ox * 2 + dx) * 3 + c] as u64;
          }
        }
        out[(oy * OUT + ox) * 3 + c] = ((s + 2) / 4) as u16;
      }
    }
  }
  out
}

/// Round-half-up 2x2 block mean of an `SRC`-grid single-channel `u16`
/// plane to the `OUT` grid (native-depth accumulator).
fn block_mean_u16(plane: &[u16]) -> Vec<u16> {
  let mut out = vec![0u16; OUT * OUT];
  for oy in 0..OUT {
    for ox in 0..OUT {
      let mut s = 0u64;
      for dy in 0..2 {
        for dx in 0..2 {
          s += plane[(oy * 2 + dy) * SRC + ox * 2 + dx] as u64;
        }
      }
      out[oy * OUT + ox] = ((s + 2) / 4) as u16;
    }
  }
  out
}

/// Direct full-res `V30X` conversions of a packed frame: u8 RGB, u16
/// RGB, and native Y.
fn direct_full(packed: &[u32]) -> (Vec<u8>, Vec<u16>, Vec<u16>) {
  let src = v30x_frame(packed);
  let mut rgb_u8 = vec![0u8; SRC * SRC * 3];
  let mut rgb_u16 = vec![0u16; SRC * SRC * 3];
  let mut y_u16 = vec![0u16; SRC * SRC];
  {
    let mut sink = MixedSinker::<V30X>::new(SRC, SRC)
      .with_rgb(&mut rgb_u8)
      .unwrap()
      .with_rgb_u16(&mut rgb_u16)
      .unwrap()
      .with_luma_u16(&mut y_u16)
      .unwrap();
    v30x_to(&src, FR, M, &mut sink).unwrap();
  }
  (rgb_u8, rgb_u16, y_u16)
}

// ---- The mandatory uniform-gray counterexample ------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn v30x_uniform_gray_downscale_leaves_colour_outputs_unchanged() {
  // A uniform (constant) frame downscaled MUST NOT change any colour
  // output: the area mean of a constant is the constant, so each output
  // equals the direct conversion of the same uniform frame. Deriving u8
  // colour by narrowing the binned u16 (rather than binning the direct
  // u8 conversion) would shift the u8 RGB / rgba / hsv away from the
  // direct u8 conversion under the two paths' independent rounding.
  let u = vec![400u16; SRC * SRC];
  let y = vec![620u16; SRC * SRC];
  let v = vec![560u16; SRC * SRC];
  let packed = pack_v30x(&u, &y, &v);
  let (full_rgb, full_rgb_u16, _full_y) = direct_full(&packed);

  let mut rgb = vec![0u8; OUT * OUT * 3];
  let mut rgba = vec![0u8; OUT * OUT * 4];
  let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
  let mut hh = vec![0u8; OUT * OUT];
  let mut ss = vec![0u8; OUT * OUT];
  let mut vv = vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<V30X, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb(&mut rgb)
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap()
        .with_rgb_u16(&mut rgb_u16)
        .unwrap()
        .with_hsv(&mut hh, &mut ss, &mut vv)
        .unwrap();
    v30x_to(&v30x_frame(&packed), FR, M, &mut sink).unwrap();
  }

  // Every direct u8 pixel is identical; the resampled u8 RGB must equal
  // that exact pixel — NOT a narrowing of the u16 bin.
  let gray_px = &full_rgb[..3];
  for px in rgb.chunks_exact(3) {
    assert_eq!(px, gray_px, "uniform-gray u8 RGB changed under downscale");
  }
  for px in rgba.chunks_exact(4) {
    assert_eq!(&px[..3], gray_px, "uniform-gray rgba colour changed");
    assert_eq!(px[3], 0xFF, "rgba alpha");
  }
  // u16 RGB likewise unchanged.
  let gray_px_u16 = &full_rgb_u16[..3];
  for px in rgb_u16.chunks_exact(3) {
    assert_eq!(
      px, gray_px_u16,
      "uniform-gray u16 RGB changed under downscale"
    );
  }
  // HSV of the unchanged u8 gray.
  let mut h_ref = vec![0u8; OUT * OUT];
  let mut s_ref = vec![0u8; OUT * OUT];
  let mut v_ref = vec![0u8; OUT * OUT];
  let gray_rgb_out: Vec<u8> = gray_px
    .iter()
    .cloned()
    .cycle()
    .take(OUT * OUT * 3)
    .collect();
  crate::row::rgb_to_hsv_row(
    &gray_rgb_out,
    &mut h_ref,
    &mut s_ref,
    &mut v_ref,
    OUT * OUT,
    false,
  );
  assert_eq!(hh, h_ref, "uniform-gray hsv H changed");
  assert_eq!(ss, s_ref, "uniform-gray hsv S changed");
  assert_eq!(vv, v_ref, "uniform-gray hsv V changed");
}

// ---- Native-depth block-mean (u16 RGB + native Y) ---------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn v30x_downscale_rgb_u16_is_native_depth_block_mean() {
  let (u, y, v) = yuv_ramp();
  let packed = pack_v30x(&u, &y, &v);
  let (_full_rgb, full_rgb_u16, _full_y) = direct_full(&packed);

  let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
  {
    let mut sink =
      MixedSinker::<V30X, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb_u16(&mut rgb_u16)
        .unwrap();
    v30x_to(&v30x_frame(&packed), FR, M, &mut sink).unwrap();
  }
  assert_eq!(
    rgb_u16,
    block_mean_rgb_u16(&full_rgb_u16),
    "rgb_u16 must be the native-depth area mean of the direct u16 RGB"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn v30x_downscale_luma_is_native_depth_block_mean_of_y() {
  let (u, y, v) = yuv_ramp();
  let packed = pack_v30x(&u, &y, &v);
  let (_full_rgb, _full_rgb_u16, full_y) = direct_full(&packed);

  let mut luma = vec![0u8; OUT * OUT];
  let mut luma_u16 = vec![0u16; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<V30X, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_luma(&mut luma)
        .unwrap()
        .with_luma_u16(&mut luma_u16)
        .unwrap();
    v30x_to(&v30x_frame(&packed), FR, M, &mut sink).unwrap();
  }
  let y_binned = block_mean_u16(&full_y);
  assert_eq!(
    luma_u16, y_binned,
    "luma_u16 must be the area-downscaled native Y"
  );
  let luma_ref: Vec<u8> = y_binned.iter().map(|&p| (p >> SHIFT) as u8).collect();
  assert_eq!(
    luma, luma_ref,
    "luma must be the binned native Y narrowed >> 2"
  );
}

// ---- All-outputs parity vs the direct per-output kernel ---------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn v30x_all_outputs_match_their_own_native_depth_block_mean() {
  // Every output attached at once: each must match the block mean of its
  // OWN direct conversion, proving the three streams coexist and none is
  // derived by narrowing another.
  let (u, y, v) = yuv_ramp();
  let packed = pack_v30x(&u, &y, &v);
  let (full_rgb, full_rgb_u16, full_y) = direct_full(&packed);

  let mut rgb = vec![0u8; OUT * OUT * 3];
  let mut rgba = vec![0u8; OUT * OUT * 4];
  let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
  let mut rgba_u16 = vec![0u16; OUT * OUT * 4];
  let mut luma = vec![0u8; OUT * OUT];
  let mut luma_u16 = vec![0u16; OUT * OUT];
  let mut hh = vec![0u8; OUT * OUT];
  let mut ss = vec![0u8; OUT * OUT];
  let mut vv = vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<V30X, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
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
    v30x_to(&v30x_frame(&packed), FR, M, &mut sink).unwrap();
  }

  let rgb_ref = block_mean_rgb_u8(&full_rgb);
  let rgb_u16_ref = block_mean_rgb_u16(&full_rgb_u16);
  let y_binned = block_mean_u16(&full_y);

  assert_eq!(rgb, rgb_ref, "all-outputs rgb");
  for (px, c) in rgba.chunks_exact(4).zip(rgb_ref.chunks_exact(3)) {
    assert_eq!(&px[..3], c, "all-outputs rgba colour");
    assert_eq!(px[3], 0xFF, "all-outputs rgba alpha");
  }
  assert_eq!(rgb_u16, rgb_u16_ref, "all-outputs rgb_u16");
  // rgba_u16 colour == rgb_u16, alpha == 10-bit max (0x3FF).
  let mut rgba_u16_ref = vec![0u16; OUT * OUT * 4];
  crate::row::expand_rgb_u16_to_rgba_u16_row::<10>(&rgb_u16_ref, &mut rgba_u16_ref, OUT * OUT);
  assert_eq!(rgba_u16, rgba_u16_ref, "all-outputs rgba_u16");
  assert_eq!(luma_u16, y_binned, "all-outputs luma_u16");
  let luma_ref: Vec<u8> = y_binned.iter().map(|&p| (p >> SHIFT) as u8).collect();
  assert_eq!(luma, luma_ref, "all-outputs luma");
  // HSV from the binned u8 RGB.
  let mut h_ref = vec![0u8; OUT * OUT];
  let mut s_ref = vec![0u8; OUT * OUT];
  let mut v_ref = vec![0u8; OUT * OUT];
  crate::row::rgb_to_hsv_row(
    &rgb_ref,
    &mut h_ref,
    &mut s_ref,
    &mut v_ref,
    OUT * OUT,
    false,
  );
  assert_eq!(hh, h_ref, "all-outputs hsv H");
  assert_eq!(ss, s_ref, "all-outputs hsv S");
  assert_eq!(vv, v_ref, "all-outputs hsv V");
}

// ---- Luma-from-native-Y counterexample --------------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn v30x_luma_taken_from_native_y_under_saturated_chroma() {
  // Constant Y, extreme U/V: the area-downscaled native Y is the constant
  // Y. RGB-derived luma would clamp away from it; luma-from-Y must stay
  // exactly the constant.
  let yc = 512u16;
  let u = vec![1000u16; SRC * SRC];
  let y = vec![yc; SRC * SRC];
  let v = vec![20u16; SRC * SRC];
  let packed = pack_v30x(&u, &y, &v);
  let mut luma_u16 = vec![0u16; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<V30X, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_luma_u16(&mut luma_u16)
        .unwrap();
    v30x_to(&v30x_frame(&packed), FR, M, &mut sink).unwrap();
  }
  assert!(
    luma_u16.iter().all(|&p| p == yc),
    "luma_u16 must be the native Y ({yc}), not RGB-derived; got {luma_u16:?}"
  );
}

// ---- Identity / no-op / reset -----------------------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn v30x_identity_plan_matches_new_sink() {
  let (u, y, v) = yuv_ramp();
  let packed = pack_v30x(&u, &y, &v);
  let mut direct = vec![0u16; SRC * SRC * 3];
  {
    let mut sink = MixedSinker::<V30X>::new(SRC, SRC)
      .with_rgb_u16(&mut direct)
      .unwrap();
    v30x_to(&v30x_frame(&packed), FR, M, &mut sink).unwrap();
  }
  let mut via_area = vec![0u16; SRC * SRC * 3];
  {
    let mut sink =
      MixedSinker::<V30X, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(SRC, SRC))
        .unwrap()
        .with_rgb_u16(&mut via_area)
        .unwrap();
    v30x_to(&v30x_frame(&packed), FR, M, &mut sink).unwrap();
  }
  assert_eq!(direct, via_area, "identity plan must match the direct sink");
}

#[test]
fn v30x_no_outputs_is_a_no_op() {
  let (u, y, v) = yuv_ramp();
  let packed = pack_v30x(&u, &y, &v);
  let mut sink =
    MixedSinker::<V30X, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap();
  v30x_to(&v30x_frame(&packed), FR, M, &mut sink).unwrap();
  assert!(
    !sink.rgb_stream_allocated(),
    "u8 stream allocated for a no-op"
  );
  assert!(
    !sink.rgb_stream_u16_allocated(),
    "u16 stream allocated for a no-op"
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
fn v30x_resets_streams_across_frames() {
  let (u, y1, v) = yuv_ramp();
  let mut y2 = y1.clone();
  for p in y2.iter_mut() {
    *p = 1023 - *p;
  }
  let p1 = pack_v30x(&u, &y1, &v);
  let p2 = pack_v30x(&u, &y2, &v);
  let mut luma_u16 = vec![0u16; OUT * OUT];
  let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
  {
    let mut sink =
      MixedSinker::<V30X, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_luma_u16(&mut luma_u16)
        .unwrap()
        .with_rgb_u16(&mut rgb_u16)
        .unwrap();
    v30x_to(&v30x_frame(&p1), FR, M, &mut sink).unwrap();
    v30x_to(&v30x_frame(&p2), FR, M, &mut sink).unwrap();
  }
  let (_r, _r16, full_y2) = direct_full(&p2);
  assert_eq!(
    luma_u16,
    block_mean_u16(&full_y2),
    "frame 2 luma_u16 must area-downscale frame 2's Y"
  );
}

// ---- Sequence / freeze ordering ---------------------------------------

#[test]
fn v30x_out_of_sequence_first_row_rejected_before_allocation() {
  let (u, y, v) = yuv_ramp();
  let packed = pack_v30x(&u, &y, &v);
  let row3 = &packed[3 * SRC..4 * SRC];
  let mut luma_u16 = vec![0u16; OUT * OUT];
  let mut rgb = vec![0u8; OUT * OUT * 3];
  let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
  let mut sink =
    MixedSinker::<V30X, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_luma_u16(&mut luma_u16)
      .unwrap()
      .with_rgb(&mut rgb)
      .unwrap()
      .with_rgb_u16(&mut rgb_u16)
      .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  let err = sink.process(V30XRow::new(row3, 3, M, FR)).unwrap_err();
  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
    ),
    "expected OutOfSequenceRow, got {err:?}"
  );
  assert!(
    !sink.rgb_stream_allocated()
      && !sink.rgb_stream_u16_allocated()
      && !sink.luma_stream_u16_allocated(),
    "stream allocated for a rejected first row"
  );
  assert_eq!(sink.rgb_scratch_capacity(), 0, "u8 scratch grown on reject");
  assert_eq!(
    sink.rgb_scratch_u16_capacity(),
    0,
    "u16 scratch grown on reject"
  );
  assert_eq!(
    sink.luma_scratch_u16_capacity(),
    0,
    "Y scratch grown on reject"
  );
  assert!(
    luma_u16.iter().all(|&p| p == 0)
      && rgb.iter().all(|&b| b == 0)
      && rgb_u16.iter().all(|&p| p == 0),
    "rejected row mutated output"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn v30x_rejects_mid_frame_out_of_sequence() {
  let (u, y, v) = yuv_ramp();
  let packed = pack_v30x(&u, &y, &v);
  let mut luma_u16 = vec![0u16; OUT * OUT];
  let mut sink =
    MixedSinker::<V30X, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_luma_u16(&mut luma_u16)
      .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  sink
    .process(V30XRow::new(&packed[..SRC], 0, M, FR))
    .unwrap();
  let err = sink
    .process(V30XRow::new(&packed[2 * SRC..3 * SRC], 2, M, FR))
    .unwrap_err();
  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
    ),
    "expected OutOfSequenceRow, got {err:?}"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn v30x_rejects_mid_frame_output_change() {
  let (u, y, v) = yuv_ramp();
  let packed = pack_v30x(&u, &y, &v);
  let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
  let mut luma_u16 = vec![0u16; OUT * OUT];
  let mut sink =
    MixedSinker::<V30X, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_rgb_u16(&mut rgb_u16)
      .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  sink
    .process(V30XRow::new(&packed[..SRC], 0, M, FR))
    .unwrap();
  sink.set_luma_u16(&mut luma_u16).unwrap();
  let err = sink
    .process(V30XRow::new(&packed[SRC..2 * SRC], 1, M, FR))
    .unwrap_err();
  assert!(
    matches!(err, MixedSinkerError::ResampleOutputsChanged(_)),
    "expected ResampleOutputsChanged, got {err:?}"
  );
  assert!(
    luma_u16.iter().all(|&p| p == 0),
    "rejected row mutated the new output"
  );
}

// ---- SIMD vs scalar parity --------------------------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn v30x_resample_simd_matches_scalar() {
  let (u, y, v) = yuv_ramp();
  let packed = pack_v30x(&u, &y, &v);

  let run = |simd: bool| {
    let mut rgb = vec![0u8; OUT * OUT * 3];
    let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
    let mut luma_u16 = vec![0u16; OUT * OUT];
    let mut sink =
      MixedSinker::<V30X, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_simd(simd)
        .with_rgb(&mut rgb)
        .unwrap()
        .with_rgb_u16(&mut rgb_u16)
        .unwrap()
        .with_luma_u16(&mut luma_u16)
        .unwrap();
    v30x_to(&v30x_frame(&packed), FR, M, &mut sink).unwrap();
    (rgb, rgb_u16, luma_u16)
  };
  assert_eq!(run(true), run(false), "V30X resample SIMD != scalar");
}
