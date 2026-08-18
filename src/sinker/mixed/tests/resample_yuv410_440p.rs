//! Fused-downscale coverage for `Yuv410p` / `Yuv440p` — routed through
//! the planar dual-stream resample: **luma / luma_u16 area-resample
//! the Y plane directly** (the YUV luma contract), while RGB / RGBA /
//! HSV bin a converted source-width RGB row. So RGB equals an `Rgb24`
//! resample of the identity-converted frame, and luma equals the
//! area-downscaled Y plane — *not* RGB-derived luma, pinned under
//! saturated chroma where the two diverge.
//!
//! These are the remaining 8-bit planar subsampling variants: 4:1:0
//! (quarter-width *and* quarter-height chroma) and 4:4:0 (full-width,
//! half-height chroma). They route identically to Yuv411p / Yuv444p;
//! only the chroma-subsampling convert kernel differs.

use crate::{
  KernelMatrix, PixelSink,
  resample::{AreaResampler, ResampleError},
  sinker::{MixedSinker, MixedSinkerError},
  source::{Rgb24, Yuv410p, Yuv410pRow, Yuv440p, Yuv440pRow, rgb24_to, yuv410p_to, yuv440p_to},
};
use mediaframe::frame::{Rgb24Frame, Yuv410pFrame, Yuv440pFrame};

const SRC: usize = 8;
const OUT: usize = 4;

/// Exact 2x2-block area mean (round-half-up) of an `SRC`-grid plane to
/// the `OUT` grid — the integer-ratio (2:1) area-downscale reference.
fn block_mean_2x2(plane: &[u8]) -> Vec<u8> {
  let mut out = vec![0u8; OUT * OUT];
  for oy in 0..OUT {
    for ox in 0..OUT {
      let mut s = 0u32;
      for dy in 0..2 {
        for dx in 0..2 {
          s += plane[(oy * 2 + dy) * SRC + ox * 2 + dx] as u32;
        }
      }
      out[oy * OUT + ox] = ((s + 2) / 4) as u8;
    }
  }
  out
}

/// `Rgb24` resample of a full-res converted RGB frame — the colour
/// reference both formats must match.
fn rgb24_rgb_reference(converted: &[u8]) -> Vec<u8> {
  let src = Rgb24Frame::new(converted, SRC as u32, SRC as u32, (SRC * 3) as u32);
  let mut rgb = vec![0u8; OUT * OUT * 3];
  {
    let mut sink =
      MixedSinker::<Rgb24, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb(&mut rgb)
        .unwrap();
    rgb24_to(&src, true, KernelMatrix::Bt601, &mut sink).unwrap();
  }
  rgb
}

// ---- Yuv410p (quarter-width, quarter-height chroma) ----

fn yuv410p_ramp() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let cw = SRC / 4;
  let ch = SRC / 4;
  let mut y = vec![0u8; SRC * SRC];
  let mut u = vec![0u8; cw * ch];
  let mut v = vec![0u8; cw * ch];
  for (i, p) in y.iter_mut().enumerate() {
    *p = 40 + (i as u8) * 2;
  }
  for (i, p) in u.iter_mut().enumerate() {
    *p = 70 + ((i % cw) as u8) * 5;
  }
  for (i, p) in v.iter_mut().enumerate() {
    *p = 200 - ((i % cw) as u8) * 4;
  }
  (y, u, v)
}

fn yuv410p_frame<'a>(y: &'a [u8], u: &'a [u8], v: &'a [u8]) -> Yuv410pFrame<'a> {
  let cw = (SRC / 4) as u32;
  Yuv410pFrame::new(y, u, v, SRC as u32, SRC as u32, SRC as u32, cw, cw)
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv410p_resample_rgb_matches_rgb24_of_converted_frame() {
  let (y, u, v) = yuv410p_ramp();
  let mut full_rgb = vec![0u8; SRC * SRC * 3];
  {
    let mut sink = MixedSinker::<Yuv410p>::new(SRC, SRC)
      .with_rgb(&mut full_rgb)
      .unwrap();
    yuv410p_to(
      &yuv410p_frame(&y, &u, &v),
      true,
      KernelMatrix::Bt601,
      &mut sink,
    )
    .unwrap();
  }
  let mut rgb = vec![0u8; OUT * OUT * 3];
  {
    let mut sink =
      MixedSinker::<Yuv410p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb(&mut rgb)
        .unwrap();
    yuv410p_to(
      &yuv410p_frame(&y, &u, &v),
      true,
      KernelMatrix::Bt601,
      &mut sink,
    )
    .unwrap();
  }
  assert_eq!(
    rgb,
    rgb24_rgb_reference(&full_rgb),
    "rgb: row-stage == convert-then-bin"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv410p_resample_luma_is_area_downscaled_y_plane() {
  let (y, u, v) = yuv410p_ramp();
  let (mut luma, mut luma_u16) = (vec![0u8; OUT * OUT], vec![0u16; OUT * OUT]);
  {
    let mut sink =
      MixedSinker::<Yuv410p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_luma(&mut luma)
        .unwrap()
        .with_luma_u16(&mut luma_u16)
        .unwrap();
    yuv410p_to(
      &yuv410p_frame(&y, &u, &v),
      true,
      KernelMatrix::Bt601,
      &mut sink,
    )
    .unwrap();
  }
  let y_ref = block_mean_2x2(&y);
  assert_eq!(luma, y_ref, "luma must be the area-downscaled Y plane");
  let y_ref_u16: Vec<u16> = y_ref.iter().map(|&b| b as u16).collect();
  assert_eq!(
    luma_u16, y_ref_u16,
    "luma_u16 must be the area-downscaled Y, zero-extended"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv410p_resample_luma_from_y_not_rgb_under_saturated_chroma() {
  let cw = SRC / 4;
  let ch = SRC / 4;
  let y = vec![16u8; SRC * SRC];
  let u = vec![240u8; cw * ch];
  let v = vec![16u8; cw * ch];
  let mut luma = vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Yuv410p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_luma(&mut luma)
        .unwrap();
    yuv410p_to(
      &yuv410p_frame(&y, &u, &v),
      true,
      KernelMatrix::Bt601,
      &mut sink,
    )
    .unwrap();
  }
  assert!(
    luma.iter().all(|&b| b == 16),
    "luma must be the Y plane (16), not RGB-derived; got {luma:?}"
  );
}

// ---- Yuv440p (full-width, half-height chroma) ----

fn yuv440p_ramp() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let ch = SRC / 2;
  let mut y = vec![0u8; SRC * SRC];
  let mut u = vec![0u8; SRC * ch];
  let mut v = vec![0u8; SRC * ch];
  for (i, p) in y.iter_mut().enumerate() {
    *p = 40 + (i as u8) * 2;
  }
  for (i, p) in u.iter_mut().enumerate() {
    *p = 70 + ((i % SRC) as u8) * 5;
  }
  for (i, p) in v.iter_mut().enumerate() {
    *p = 200 - ((i % SRC) as u8) * 3;
  }
  (y, u, v)
}

fn yuv440p_frame<'a>(y: &'a [u8], u: &'a [u8], v: &'a [u8]) -> Yuv440pFrame<'a> {
  Yuv440pFrame::new(
    y, u, v, SRC as u32, SRC as u32, SRC as u32, SRC as u32, SRC as u32,
  )
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv440p_resample_rgb_matches_rgb24_of_converted_frame() {
  let (y, u, v) = yuv440p_ramp();
  let mut full_rgb = vec![0u8; SRC * SRC * 3];
  {
    let mut sink = MixedSinker::<Yuv440p>::new(SRC, SRC)
      .with_rgb(&mut full_rgb)
      .unwrap();
    yuv440p_to(
      &yuv440p_frame(&y, &u, &v),
      true,
      KernelMatrix::Bt601,
      &mut sink,
    )
    .unwrap();
  }
  let mut rgb = vec![0u8; OUT * OUT * 3];
  {
    // Pin the row-stage tier: this oracle is convert-then-bin (the default
    // tier is now the bin-then-convert native fast tier, covered for parity
    // in `resample_yuv_planar_8bit_native`).
    let mut sink =
      MixedSinker::<Yuv440p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_native(false)
        .with_rgb(&mut rgb)
        .unwrap();
    yuv440p_to(
      &yuv440p_frame(&y, &u, &v),
      true,
      KernelMatrix::Bt601,
      &mut sink,
    )
    .unwrap();
  }
  assert_eq!(
    rgb,
    rgb24_rgb_reference(&full_rgb),
    "rgb: row-stage == convert-then-bin"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv440p_resample_luma_is_area_downscaled_y_plane() {
  let (y, u, v) = yuv440p_ramp();
  let (mut luma, mut luma_u16) = (vec![0u8; OUT * OUT], vec![0u16; OUT * OUT]);
  {
    let mut sink =
      MixedSinker::<Yuv440p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_luma(&mut luma)
        .unwrap()
        .with_luma_u16(&mut luma_u16)
        .unwrap();
    yuv440p_to(
      &yuv440p_frame(&y, &u, &v),
      true,
      KernelMatrix::Bt601,
      &mut sink,
    )
    .unwrap();
  }
  let y_ref = block_mean_2x2(&y);
  assert_eq!(luma, y_ref, "luma must be the area-downscaled Y plane");
  let y_ref_u16: Vec<u16> = y_ref.iter().map(|&b| b as u16).collect();
  assert_eq!(
    luma_u16, y_ref_u16,
    "luma_u16 must be the area-downscaled Y, zero-extended"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv440p_resample_luma_from_y_not_rgb_under_saturated_chroma() {
  let ch = SRC / 2;
  let y = vec![16u8; SRC * SRC];
  let u = vec![240u8; SRC * ch];
  let v = vec![16u8; SRC * ch];
  let mut luma = vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Yuv440p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_luma(&mut luma)
        .unwrap();
    yuv440p_to(
      &yuv440p_frame(&y, &u, &v),
      true,
      KernelMatrix::Bt601,
      &mut sink,
    )
    .unwrap();
  }
  assert!(
    luma.iter().all(|&b| b == 16),
    "luma must be the Y plane (16), not RGB-derived; got {luma:?}"
  );
}

// ---- Cross-frame stream reset + identity-plan + sequencing ----

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv410p_resample_reuses_luma_stream_across_frames() {
  // Reset coverage: a reused sink must reset the Y-plane luma stream
  // each frame, else frame 2 row 0 is rejected as out-of-sequence.
  let (y1, u, v) = yuv410p_ramp();
  let mut y2 = y1.clone();
  for p in y2.iter_mut() {
    *p = 255 - *p;
  }
  let mut luma = vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Yuv410p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_luma(&mut luma)
        .unwrap();
    yuv410p_to(
      &yuv410p_frame(&y1, &u, &v),
      true,
      KernelMatrix::Bt601,
      &mut sink,
    )
    .unwrap();
    yuv410p_to(
      &yuv410p_frame(&y2, &u, &v),
      true,
      KernelMatrix::Bt601,
      &mut sink,
    )
    .unwrap();
  }
  assert_eq!(
    luma,
    block_mean_2x2(&y2),
    "frame 2 luma must area-downscale frame 2's Y"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv440p_resample_reuses_luma_stream_across_frames() {
  let (y1, u, v) = yuv440p_ramp();
  let mut y2 = y1.clone();
  for p in y2.iter_mut() {
    *p = 255 - *p;
  }
  let mut luma = vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Yuv440p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_luma(&mut luma)
        .unwrap();
    yuv440p_to(
      &yuv440p_frame(&y1, &u, &v),
      true,
      KernelMatrix::Bt601,
      &mut sink,
    )
    .unwrap();
    yuv440p_to(
      &yuv440p_frame(&y2, &u, &v),
      true,
      KernelMatrix::Bt601,
      &mut sink,
    )
    .unwrap();
  }
  assert_eq!(
    luma,
    block_mean_2x2(&y2),
    "frame 2 luma must area-downscale frame 2's Y"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv410p_identity_plan_matches_new_sink() {
  let (y, u, v) = yuv410p_ramp();
  let mut direct = vec![0u8; SRC * SRC * 3];
  {
    let mut sink = MixedSinker::<Yuv410p>::new(SRC, SRC)
      .with_rgb(&mut direct)
      .unwrap();
    yuv410p_to(
      &yuv410p_frame(&y, &u, &v),
      true,
      KernelMatrix::Bt601,
      &mut sink,
    )
    .unwrap();
  }
  let mut via_area = vec![0u8; SRC * SRC * 3];
  {
    let mut sink =
      MixedSinker::<Yuv410p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(SRC, SRC))
        .unwrap()
        .with_rgb(&mut via_area)
        .unwrap();
    yuv410p_to(
      &yuv410p_frame(&y, &u, &v),
      true,
      KernelMatrix::Bt601,
      &mut sink,
    )
    .unwrap();
  }
  assert_eq!(direct, via_area);
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv440p_identity_plan_matches_new_sink() {
  let (y, u, v) = yuv440p_ramp();
  let mut direct = vec![0u8; SRC * SRC * 3];
  {
    let mut sink = MixedSinker::<Yuv440p>::new(SRC, SRC)
      .with_rgb(&mut direct)
      .unwrap();
    yuv440p_to(
      &yuv440p_frame(&y, &u, &v),
      true,
      KernelMatrix::Bt601,
      &mut sink,
    )
    .unwrap();
  }
  let mut via_area = vec![0u8; SRC * SRC * 3];
  {
    let mut sink =
      MixedSinker::<Yuv440p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(SRC, SRC))
        .unwrap()
        .with_rgb(&mut via_area)
        .unwrap();
    yuv440p_to(
      &yuv440p_frame(&y, &u, &v),
      true,
      KernelMatrix::Bt601,
      &mut sink,
    )
    .unwrap();
  }
  assert_eq!(direct, via_area);
}

#[test]
fn yuv410p_resample_no_outputs_is_a_no_op() {
  let (y, u, v) = yuv410p_ramp();
  let mut sink =
    MixedSinker::<Yuv410p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap();
  // No outputs attached: a legal no-op, accepted without error.
  yuv410p_to(
    &yuv410p_frame(&y, &u, &v),
    true,
    KernelMatrix::Bt601,
    &mut sink,
  )
  .unwrap();
}

#[test]
fn yuv440p_resample_no_outputs_is_a_no_op() {
  let (y, u, v) = yuv440p_ramp();
  let mut sink =
    MixedSinker::<Yuv440p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap();
  yuv440p_to(
    &yuv440p_frame(&y, &u, &v),
    true,
    KernelMatrix::Bt601,
    &mut sink,
  )
  .unwrap();
}

#[test]
fn yuv410p_resample_rejects_out_of_sequence_rows() {
  let (y, u, v) = yuv410p_ramp();
  let cw = SRC / 4;
  let mut rgb = vec![0u8; OUT * OUT * 3];
  let mut sink =
    MixedSinker::<Yuv410p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_rgb(&mut rgb)
      .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  // Feed row 2 first — the stream expects strict sequencing from 0. In
  // 4:1:0 the chroma row index is `row / 4`, so row 2 reads chroma 0.
  let row2 = Yuv410pRow::new(
    &y[SRC * 2..SRC * 3],
    &u[0..cw],
    &v[0..cw],
    2,
    KernelMatrix::Bt601,
    true,
  );
  let err = sink.process(row2).unwrap_err();
  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
    ),
    "got {err:?}"
  );
}

#[test]
fn yuv440p_resample_rejects_out_of_sequence_rows() {
  let (y, u, v) = yuv440p_ramp();
  let mut rgb = vec![0u8; OUT * OUT * 3];
  let mut sink =
    MixedSinker::<Yuv440p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_rgb(&mut rgb)
      .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  // Feed row 2 first — the stream expects strict sequencing from 0. In
  // 4:4:0 the chroma row index is `row / 2`, so row 2 reads chroma 1.
  let row2 = Yuv440pRow::new(
    &y[SRC * 2..SRC * 3],
    &u[SRC..SRC * 2],
    &v[SRC..SRC * 2],
    2,
    KernelMatrix::Bt601,
    true,
  );
  let err = sink.process(row2).unwrap_err();
  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
    ),
    "got {err:?}"
  );
}
