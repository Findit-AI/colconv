//! Fused-downscale coverage for `Yuv411p` — routed through the planar
//! dual-stream resample: **luma / luma_u16 area-resample the Y plane
//! directly** (the YUV luma contract), while RGB / RGBA / HSV bin a
//! converted source-width RGB row. So RGB equals an `Rgb24` resample
//! of the identity-converted frame, and luma equals the
//! area-downscaled Y plane — *not* RGB-derived luma. The latter is
//! pinned under saturated chroma, where converting Y/U/V to RGB and
//! back to luma would clip far away from the true Y.

use crate::{
  KernelMatrix, PixelSink,
  resample::{AreaResampler, ResampleError},
  sinker::{MixedSinker, MixedSinkerError},
  source::{Rgb24, Yuv411p, Yuv411pRow, rgb24_to, yuv411p_to},
};
use mediaframe::frame::{Rgb24Frame, Yuv411pFrame};

const SRC_W: usize = 8;
const SRC_H: usize = 8;
const OUT_W: usize = 4;
const OUT_H: usize = 4;
// FFmpeg `AV_PIX_FMT_YUV411P`: chroma row width is `width.div_ceil(4)`.
const CW: usize = SRC_W.div_ceil(4);

fn yuv411p_frame<'a>(y: &'a [u8], u: &'a [u8], v: &'a [u8]) -> Yuv411pFrame<'a> {
  Yuv411pFrame::new(
    y,
    u,
    v,
    SRC_W as u32,
    SRC_H as u32,
    SRC_W as u32,
    CW as u32,
    CW as u32,
  )
}

/// Interior ramps across Y / U / V so the chroma upsample and the
/// area mean both see real variation.
fn ramp() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let mut y = vec![0u8; SRC_W * SRC_H];
  for (i, p) in y.iter_mut().enumerate() {
    *p = 32 + (i % 64) as u8;
  }
  let mut u = vec![0u8; CW * SRC_H];
  for (i, p) in u.iter_mut().enumerate() {
    *p = 100 + (i % 40) as u8;
  }
  let mut v = vec![0u8; CW * SRC_H];
  for (i, p) in v.iter_mut().enumerate() {
    *p = 140 - (i % 40) as u8;
  }
  (y, u, v)
}

/// Exact 2x2-block area mean (round-half-up) of an `SRC`-grid plane to
/// the `OUT` grid — the integer-ratio (2:1) area-downscale reference.
fn block_mean_2x2(plane: &[u8]) -> Vec<u8> {
  let mut out = vec![0u8; OUT_W * OUT_H];
  for oy in 0..OUT_H {
    for ox in 0..OUT_W {
      let mut s = 0u32;
      for dy in 0..2 {
        for dx in 0..2 {
          s += plane[(oy * 2 + dy) * SRC_W + ox * 2 + dx] as u32;
        }
      }
      out[oy * OUT_W + ox] = ((s + 2) / 4) as u8;
    }
  }
  out
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv411p_resample_rgb_matches_rgb24_of_converted_frame() {
  let (yp, up, vp) = ramp();
  let src = yuv411p_frame(&yp, &up, &vp);

  let mut rgb_a = vec![0u8; OUT_W * OUT_H * 3];
  {
    let mut sink = MixedSinker::<Yuv411p, AreaResampler>::with_resampler(
      SRC_W,
      SRC_H,
      AreaResampler::to(OUT_W, OUT_H),
    )
    .unwrap()
    .with_rgb(&mut rgb_a)
    .unwrap();
    yuv411p_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
  }

  let mut full_rgb = vec![0u8; SRC_W * SRC_H * 3];
  {
    let mut sink = MixedSinker::<Yuv411p>::new(SRC_W, SRC_H)
      .with_rgb(&mut full_rgb)
      .unwrap();
    yuv411p_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
  }
  let rgb_src = Rgb24Frame::new(&full_rgb, SRC_W as u32, SRC_H as u32, (SRC_W * 3) as u32);
  let mut rgb_b = vec![0u8; OUT_W * OUT_H * 3];
  {
    let mut sink = MixedSinker::<Rgb24, AreaResampler>::with_resampler(
      SRC_W,
      SRC_H,
      AreaResampler::to(OUT_W, OUT_H),
    )
    .unwrap()
    .with_rgb(&mut rgb_b)
    .unwrap();
    rgb24_to(&rgb_src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
  }
  assert_eq!(rgb_a, rgb_b, "rgb: row-stage must equal convert-then-bin");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv411p_resample_luma_is_area_downscaled_y_plane() {
  let (yp, up, vp) = ramp();
  let src = yuv411p_frame(&yp, &up, &vp);

  let (mut luma, mut luma_u16) = (vec![0u8; OUT_W * OUT_H], vec![0u16; OUT_W * OUT_H]);
  {
    let mut sink = MixedSinker::<Yuv411p, AreaResampler>::with_resampler(
      SRC_W,
      SRC_H,
      AreaResampler::to(OUT_W, OUT_H),
    )
    .unwrap()
    .with_luma(&mut luma)
    .unwrap()
    .with_luma_u16(&mut luma_u16)
    .unwrap();
    yuv411p_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
  }
  let y_ref = block_mean_2x2(&yp);
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
fn yuv411p_resample_luma_comes_from_y_not_rgb_under_saturated_chroma() {
  // Uniform low Y with saturated chroma: converting to RGB and back to
  // luma would clip far above the true Y. The resampled luma must be
  // the (uniform) Y mean, proving it area-resamples the Y plane.
  let yp = vec![16u8; SRC_W * SRC_H];
  let up = vec![240u8; CW * SRC_H];
  let vp = vec![16u8; CW * SRC_H];
  let src = yuv411p_frame(&yp, &up, &vp);

  let mut luma = vec![0u8; OUT_W * OUT_H];
  {
    let mut sink = MixedSinker::<Yuv411p, AreaResampler>::with_resampler(
      SRC_W,
      SRC_H,
      AreaResampler::to(OUT_W, OUT_H),
    )
    .unwrap()
    .with_luma(&mut luma)
    .unwrap();
    yuv411p_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
  }
  assert!(
    luma.iter().all(|&b| b == 16),
    "luma must area-resample the Y plane (16), not RGB-derived luma; got {luma:?}"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv411p_identity_plan_matches_new_sink() {
  let (yp, up, vp) = ramp();
  let src = yuv411p_frame(&yp, &up, &vp);

  let mut direct = vec![0u8; SRC_W * SRC_H * 3];
  {
    let mut sink = MixedSinker::<Yuv411p>::new(SRC_W, SRC_H)
      .with_rgb(&mut direct)
      .unwrap();
    yuv411p_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
  }
  let mut via_area = vec![0u8; SRC_W * SRC_H * 3];
  {
    let mut sink = MixedSinker::<Yuv411p, AreaResampler>::with_resampler(
      SRC_W,
      SRC_H,
      AreaResampler::to(SRC_W, SRC_H),
    )
    .unwrap()
    .with_rgb(&mut via_area)
    .unwrap();
    yuv411p_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
  }
  assert_eq!(direct, via_area);
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv411p_resample_reuses_luma_stream_across_frames() {
  // A reused sink must reset the Y-plane luma stream each frame; without
  // the reset, frame 2's row 0 is rejected as out-of-sequence and the
  // luma never reflects frame 2.
  let (y1, up, vp) = ramp();
  let mut y2 = y1.clone();
  for p in y2.iter_mut() {
    *p = 255 - *p;
  }
  let (mut luma, mut luma_u16) = (vec![0u8; OUT_W * OUT_H], vec![0u16; OUT_W * OUT_H]);
  {
    let mut sink = MixedSinker::<Yuv411p, AreaResampler>::with_resampler(
      SRC_W,
      SRC_H,
      AreaResampler::to(OUT_W, OUT_H),
    )
    .unwrap()
    .with_luma(&mut luma)
    .unwrap()
    .with_luma_u16(&mut luma_u16)
    .unwrap();
    yuv411p_to(
      &yuv411p_frame(&y1, &up, &vp),
      true,
      sink.set_kernel_matrix(KernelMatrix::Bt601),
    )
    .unwrap();
    yuv411p_to(
      &yuv411p_frame(&y2, &up, &vp),
      true,
      sink.set_kernel_matrix(KernelMatrix::Bt601),
    )
    .unwrap();
  }
  let y2_ref = block_mean_2x2(&y2);
  assert_eq!(luma, y2_ref, "frame 2 luma must area-downscale frame 2's Y");
  let y2_ref_u16: Vec<u16> = y2_ref.iter().map(|&b| b as u16).collect();
  assert_eq!(
    luma_u16, y2_ref_u16,
    "frame 2 luma_u16 must area-downscale frame 2's Y"
  );
}

#[test]
fn yuv411p_resample_no_outputs_is_a_no_op() {
  let (yp, up, vp) = ramp();
  let src = yuv411p_frame(&yp, &up, &vp);
  let mut sink = MixedSinker::<Yuv411p, AreaResampler>::with_resampler(
    SRC_W,
    SRC_H,
    AreaResampler::to(OUT_W, OUT_H),
  )
  .unwrap();
  // No outputs attached: a legal no-op, accepted without error.
  yuv411p_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
}

#[test]
fn yuv411p_resample_rejects_out_of_sequence_rows() {
  let (yp, up, vp) = ramp();
  let mut rgb = vec![0u8; OUT_W * OUT_H * 3];
  let mut sink = MixedSinker::<Yuv411p, AreaResampler>::with_resampler(
    SRC_W,
    SRC_H,
    AreaResampler::to(OUT_W, OUT_H),
  )
  .unwrap()
  .with_rgb(&mut rgb)
  .unwrap();
  sink.begin_frame(SRC_W as u32, SRC_H as u32).unwrap();
  // Feed row 2 first — the stream expects strict sequencing from 0.
  let row2 = Yuv411pRow::for_tests(
    &yp[SRC_W * 2..SRC_W * 3],
    &up[CW * 2..CW * 3],
    &vp[CW * 2..CW * 3],
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
