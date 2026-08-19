//! Fused-downscale coverage for `Yuv422p` / `Yuv444p` — routed through
//! the planar dual-stream resample: **luma / luma_u16 area-resample
//! the Y plane directly** (the YUV luma contract), while RGB / RGBA /
//! HSV bin a converted source-width RGB row. So RGB equals an `Rgb24`
//! resample of the identity-converted frame, and luma equals the
//! area-downscaled Y plane — *not* RGB-derived luma, pinned under
//! saturated chroma where the two diverge.

use crate::{
  KernelMatrix,
  resample::AreaResampler,
  sinker::MixedSinker,
  source::{Rgb24, Yuv422p, Yuv444p, rgb24_to, yuv422p_to, yuv444p_to},
};
use mediaframe::frame::{Rgb24Frame, Yuv422pFrame, Yuv444pFrame};

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
    rgb24_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
  }
  rgb
}

// ---- Yuv422p (half-width, full-height chroma) ----

fn yuv422p_ramp() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let cw = SRC / 2;
  let mut y = vec![0u8; SRC * SRC];
  let mut u = vec![0u8; cw * SRC];
  let mut v = vec![0u8; cw * SRC];
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

fn yuv422p_frame<'a>(y: &'a [u8], u: &'a [u8], v: &'a [u8]) -> Yuv422pFrame<'a> {
  let cw = (SRC / 2) as u32;
  Yuv422pFrame::new(y, u, v, SRC as u32, SRC as u32, SRC as u32, cw, cw)
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv422p_resample_rgb_matches_rgb24_of_converted_frame() {
  let (y, u, v) = yuv422p_ramp();
  let mut full_rgb = vec![0u8; SRC * SRC * 3];
  {
    let mut sink = MixedSinker::<Yuv422p>::new(SRC, SRC)
      .with_rgb(&mut full_rgb)
      .unwrap();
    yuv422p_to(
      &yuv422p_frame(&y, &u, &v),
      true,
      sink.set_kernel_matrix(KernelMatrix::Bt601),
    )
    .unwrap();
  }
  let mut rgb = vec![0u8; OUT * OUT * 3];
  {
    // Pin the row-stage tier: this oracle is convert-then-bin (the default
    // tier is now the bin-then-convert native fast tier, covered for parity
    // in `resample_yuv_planar_8bit_native`).
    let mut sink =
      MixedSinker::<Yuv422p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_native(false)
        .with_rgb(&mut rgb)
        .unwrap();
    yuv422p_to(
      &yuv422p_frame(&y, &u, &v),
      true,
      sink.set_kernel_matrix(KernelMatrix::Bt601),
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
fn yuv422p_resample_luma_is_area_downscaled_y_plane() {
  let (y, u, v) = yuv422p_ramp();
  let (mut luma, mut luma_u16) = (vec![0u8; OUT * OUT], vec![0u16; OUT * OUT]);
  {
    let mut sink =
      MixedSinker::<Yuv422p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_luma(&mut luma)
        .unwrap()
        .with_luma_u16(&mut luma_u16)
        .unwrap();
    yuv422p_to(
      &yuv422p_frame(&y, &u, &v),
      true,
      sink.set_kernel_matrix(KernelMatrix::Bt601),
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
fn yuv422p_resample_luma_from_y_not_rgb_under_saturated_chroma() {
  let cw = SRC / 2;
  let y = vec![16u8; SRC * SRC];
  let u = vec![240u8; cw * SRC];
  let v = vec![16u8; cw * SRC];
  let mut luma = vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Yuv422p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_luma(&mut luma)
        .unwrap();
    yuv422p_to(
      &yuv422p_frame(&y, &u, &v),
      true,
      sink.set_kernel_matrix(KernelMatrix::Bt601),
    )
    .unwrap();
  }
  assert!(
    luma.iter().all(|&b| b == 16),
    "luma must be the Y plane (16), not RGB-derived; got {luma:?}"
  );
}

// ---- Yuv444p (full-width, full-height chroma) ----

fn yuv444p_ramp() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let mut y = vec![0u8; SRC * SRC];
  let mut u = vec![0u8; SRC * SRC];
  let mut v = vec![0u8; SRC * SRC];
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

fn yuv444p_frame<'a>(y: &'a [u8], u: &'a [u8], v: &'a [u8]) -> Yuv444pFrame<'a> {
  Yuv444pFrame::new(
    y, u, v, SRC as u32, SRC as u32, SRC as u32, SRC as u32, SRC as u32,
  )
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv444p_resample_rgb_matches_rgb24_of_converted_frame() {
  let (y, u, v) = yuv444p_ramp();
  let mut full_rgb = vec![0u8; SRC * SRC * 3];
  {
    let mut sink = MixedSinker::<Yuv444p>::new(SRC, SRC)
      .with_rgb(&mut full_rgb)
      .unwrap();
    yuv444p_to(
      &yuv444p_frame(&y, &u, &v),
      true,
      sink.set_kernel_matrix(KernelMatrix::Bt601),
    )
    .unwrap();
  }
  let mut rgb = vec![0u8; OUT * OUT * 3];
  {
    // Pin the row-stage tier: this oracle is convert-then-bin (the default
    // tier is now the bin-then-convert native fast tier, covered for parity
    // in `resample_yuv_planar_8bit_native`).
    let mut sink =
      MixedSinker::<Yuv444p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_native(false)
        .with_rgb(&mut rgb)
        .unwrap();
    yuv444p_to(
      &yuv444p_frame(&y, &u, &v),
      true,
      sink.set_kernel_matrix(KernelMatrix::Bt601),
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
fn yuv444p_resample_luma_is_area_downscaled_y_plane() {
  let (y, u, v) = yuv444p_ramp();
  let (mut luma, mut luma_u16) = (vec![0u8; OUT * OUT], vec![0u16; OUT * OUT]);
  {
    let mut sink =
      MixedSinker::<Yuv444p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_luma(&mut luma)
        .unwrap()
        .with_luma_u16(&mut luma_u16)
        .unwrap();
    yuv444p_to(
      &yuv444p_frame(&y, &u, &v),
      true,
      sink.set_kernel_matrix(KernelMatrix::Bt601),
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
fn yuv444p_resample_luma_from_y_not_rgb_under_saturated_chroma() {
  let y = vec![16u8; SRC * SRC];
  let u = vec![240u8; SRC * SRC];
  let v = vec![16u8; SRC * SRC];
  let mut luma = vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Yuv444p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_luma(&mut luma)
        .unwrap();
    yuv444p_to(
      &yuv444p_frame(&y, &u, &v),
      true,
      sink.set_kernel_matrix(KernelMatrix::Bt601),
    )
    .unwrap();
  }
  assert!(
    luma.iter().all(|&b| b == 16),
    "luma must be the Y plane (16), not RGB-derived; got {luma:?}"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv422p_resample_reuses_luma_stream_across_frames() {
  // Reset coverage: a reused sink must reset the Y-plane luma stream
  // each frame, else frame 2 row 0 is rejected as out-of-sequence.
  let (y1, u, v) = yuv422p_ramp();
  let mut y2 = y1.clone();
  for p in y2.iter_mut() {
    *p = 255 - *p;
  }
  let mut luma = vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Yuv422p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_luma(&mut luma)
        .unwrap();
    yuv422p_to(
      &yuv422p_frame(&y1, &u, &v),
      true,
      sink.set_kernel_matrix(KernelMatrix::Bt601),
    )
    .unwrap();
    yuv422p_to(
      &yuv422p_frame(&y2, &u, &v),
      true,
      sink.set_kernel_matrix(KernelMatrix::Bt601),
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
fn yuv444p_resample_reuses_luma_stream_across_frames() {
  let (y1, u, v) = yuv444p_ramp();
  let mut y2 = y1.clone();
  for p in y2.iter_mut() {
    *p = 255 - *p;
  }
  let mut luma = vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Yuv444p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_luma(&mut luma)
        .unwrap();
    yuv444p_to(
      &yuv444p_frame(&y1, &u, &v),
      true,
      sink.set_kernel_matrix(KernelMatrix::Bt601),
    )
    .unwrap();
    yuv444p_to(
      &yuv444p_frame(&y2, &u, &v),
      true,
      sink.set_kernel_matrix(KernelMatrix::Bt601),
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
fn yuv422p_identity_plan_matches_new_sink() {
  let (y, u, v) = yuv422p_ramp();
  let mut direct = vec![0u8; SRC * SRC * 3];
  {
    let mut sink = MixedSinker::<Yuv422p>::new(SRC, SRC)
      .with_rgb(&mut direct)
      .unwrap();
    yuv422p_to(
      &yuv422p_frame(&y, &u, &v),
      true,
      sink.set_kernel_matrix(KernelMatrix::Bt601),
    )
    .unwrap();
  }
  let mut via_area = vec![0u8; SRC * SRC * 3];
  {
    let mut sink =
      MixedSinker::<Yuv422p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(SRC, SRC))
        .unwrap()
        .with_rgb(&mut via_area)
        .unwrap();
    yuv422p_to(
      &yuv422p_frame(&y, &u, &v),
      true,
      sink.set_kernel_matrix(KernelMatrix::Bt601),
    )
    .unwrap();
  }
  assert_eq!(direct, via_area);
}
