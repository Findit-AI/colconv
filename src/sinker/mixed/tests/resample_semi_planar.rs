//! Fused-downscale coverage for the 8-bit semi-planar YUV family
//! (NV12 / NV21 / NV16 / NV24 / NV42). Each member bins the Y plane
//! directly for luma (the YUV luma contract). For colour every member
//! defaults to the native bin-then-convert tier (see the `native_tier`
//! module below) — the 4:2:0 members (NV12 / NV21) via the planar 4:2:0
//! join, the 4:2:2 / 4:4:4 members (NV16 / NV24 / NV42) via the non-4:2:0
//! planar join, both on the de-interleaved chroma planes. The
//! convert-then-bin row-stage contract — RGB equals an `Rgb24` resample of
//! the identity-converted frame, byte-identical to the [`Yuv420p`] /
//! [`Yuv422p`] / [`Yuv444p`] row-stage resample of the de-interleaved
//! planes — is exercised under `with_native(false)`.

use crate::{
  KernelMatrix, PixelSink,
  resample::{AreaResampler, ResampleError},
  sinker::{MixedSinker, MixedSinkerError},
  source::{
    Nv12, Nv12Row, Nv16, Nv21, Nv24, Nv42, Rgb24, nv12_to, nv16_to, nv21_to, nv24_to, nv42_to,
    rgb24_to,
  },
};
// `Nv21Row` is consumed only by the native-tier 4:2:0 coverage (NV21 shares the
// planar 4:2:0 join), which is itself `yuv-planar`-gated.
#[cfg(feature = "yuv-planar")]
use crate::source::Nv21Row;
use mediaframe::frame::{Nv12Frame, Nv16Frame, Nv21Frame, Nv24Frame, Nv42Frame, Rgb24Frame};

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
/// reference every member must match (convert-then-bin semantics).
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

// ---- Plane / interleaved-chroma fixtures --------------------------------

/// A deterministic Y ramp over the `SRC` grid.
fn y_ramp() -> Vec<u8> {
  (0..SRC * SRC)
    .map(|i| 40u8.wrapping_add((i as u8).wrapping_mul(2)))
    .collect()
}

/// Separate U / V planes at chroma width `cw`, chroma height `ch`.
fn uv_planes(cw: usize, ch: usize) -> (Vec<u8>, Vec<u8>) {
  let mut u = vec![0u8; cw * ch];
  let mut v = vec![0u8; cw * ch];
  for (i, p) in u.iter_mut().enumerate() {
    *p = 70 + ((i % cw) as u8).wrapping_mul(5);
  }
  for (i, p) in v.iter_mut().enumerate() {
    *p = 200u8.wrapping_sub(((i % cw) as u8).wrapping_mul(4));
  }
  (u, v)
}

/// Interleave separate U / V planes into a semi-planar chroma plane.
/// `swap_uv = false` writes `U0 V0 U1 V1 …` (NV12 / NV16 / NV24);
/// `swap_uv = true` writes `V0 U0 …` (NV21 / NV42).
fn interleave(u: &[u8], v: &[u8], swap_uv: bool) -> Vec<u8> {
  let mut out = vec![0u8; u.len() * 2];
  for (i, (&uu, &vv)) in u.iter().zip(v).enumerate() {
    let (a, b) = if swap_uv { (vv, uu) } else { (uu, vv) };
    out[i * 2] = a;
    out[i * 2 + 1] = b;
  }
  out
}

// =========================================================================
// NV12 (4:2:0, UV)
// =========================================================================

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv12_resample_rgb_matches_rgb24_of_converted_frame() {
  let y = y_ramp();
  let (u, v) = uv_planes(SRC / 2, SRC / 2);
  let uv = interleave(&u, &v, false);
  let mut full_rgb = vec![0u8; SRC * SRC * 3];
  {
    let frame = Nv12Frame::new(&y, &uv, SRC as u32, SRC as u32, SRC as u32, SRC as u32);
    let mut sink = MixedSinker::<Nv12>::new(SRC, SRC)
      .with_rgb(&mut full_rgb)
      .unwrap();
    nv12_to(&frame, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
  }
  let mut rgb = vec![0u8; OUT * OUT * 3];
  {
    let frame = Nv12Frame::new(&y, &uv, SRC as u32, SRC as u32, SRC as u32, SRC as u32);
    let mut sink = MixedSinker::<Nv12, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        // The convert-then-bin RGB contract is the ROW-STAGE tier; the
        // native default averages in the YUV domain (see `native_tier`).
        .with_native(false)
        .with_rgb(&mut rgb)
        .unwrap();
    nv12_to(&frame, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
  }
  assert_eq!(
    rgb,
    rgb24_rgb_reference(&full_rgb),
    "nv12 rgb: convert-then-bin"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv12_resample_luma_is_area_downscaled_y_plane() {
  let y = y_ramp();
  let (u, v) = uv_planes(SRC / 2, SRC / 2);
  let uv = interleave(&u, &v, false);
  let (mut luma, mut luma_u16) = (vec![0u8; OUT * OUT], vec![0u16; OUT * OUT]);
  {
    let frame = Nv12Frame::new(&y, &uv, SRC as u32, SRC as u32, SRC as u32, SRC as u32);
    let mut sink =
      MixedSinker::<Nv12, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_luma(&mut luma)
        .unwrap()
        .with_luma_u16(&mut luma_u16)
        .unwrap();
    nv12_to(&frame, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
  }
  let y_ref = block_mean_2x2(&y);
  assert_eq!(luma, y_ref, "nv12 luma must be the area-downscaled Y plane");
  let y_ref_u16: Vec<u16> = y_ref.iter().map(|&b| b as u16).collect();
  assert_eq!(luma_u16, y_ref_u16, "nv12 luma_u16 = Y, zero-extended");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv12_resample_luma_from_y_not_rgb_under_saturated_chroma() {
  let y = vec![16u8; SRC * SRC];
  let cw = SRC / 2;
  let ch = SRC / 2;
  let u = vec![240u8; cw * ch];
  let v = vec![16u8; cw * ch];
  let uv = interleave(&u, &v, false);
  let mut luma = vec![0u8; OUT * OUT];
  {
    let frame = Nv12Frame::new(&y, &uv, SRC as u32, SRC as u32, SRC as u32, SRC as u32);
    let mut sink =
      MixedSinker::<Nv12, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_luma(&mut luma)
        .unwrap();
    nv12_to(&frame, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
  }
  assert!(
    luma.iter().all(|&b| b == 16),
    "nv12 luma must be Y (16); got {luma:?}"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv12_identity_plan_matches_new_sink() {
  let y = y_ramp();
  let (u, v) = uv_planes(SRC / 2, SRC / 2);
  let uv = interleave(&u, &v, false);
  let mut direct = vec![0u8; SRC * SRC * 3];
  {
    let frame = Nv12Frame::new(&y, &uv, SRC as u32, SRC as u32, SRC as u32, SRC as u32);
    let mut sink = MixedSinker::<Nv12>::new(SRC, SRC)
      .with_rgb(&mut direct)
      .unwrap();
    nv12_to(&frame, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
  }
  let mut via_area = vec![0u8; SRC * SRC * 3];
  {
    let frame = Nv12Frame::new(&y, &uv, SRC as u32, SRC as u32, SRC as u32, SRC as u32);
    let mut sink =
      MixedSinker::<Nv12, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(SRC, SRC))
        .unwrap()
        .with_rgb(&mut via_area)
        .unwrap();
    nv12_to(&frame, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
  }
  assert_eq!(direct, via_area, "nv12 identity plan == new sink");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv12_resample_no_output_is_noop() {
  let y = y_ramp();
  let (u, v) = uv_planes(SRC / 2, SRC / 2);
  let uv = interleave(&u, &v, false);
  let frame = Nv12Frame::new(&y, &uv, SRC as u32, SRC as u32, SRC as u32, SRC as u32);
  let mut sink =
    MixedSinker::<Nv12, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap();
  // No attached outputs: every row is a legal no-op (no alloc, no
  // sequencing), even out of order.
  nv12_to(&frame, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv12_resample_reuses_streams_across_frames() {
  let y1 = y_ramp();
  let mut y2 = y1.clone();
  for p in y2.iter_mut() {
    *p = 255 - *p;
  }
  let (u, v) = uv_planes(SRC / 2, SRC / 2);
  let uv = interleave(&u, &v, false);
  let mut luma = vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Nv12, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_luma(&mut luma)
        .unwrap();
    let f1 = Nv12Frame::new(&y1, &uv, SRC as u32, SRC as u32, SRC as u32, SRC as u32);
    let f2 = Nv12Frame::new(&y2, &uv, SRC as u32, SRC as u32, SRC as u32, SRC as u32);
    nv12_to(&f1, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
    nv12_to(&f2, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
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
fn nv12_resample_out_of_order_row_rejected() {
  let y = y_ramp();
  let (u, v) = uv_planes(SRC / 2, SRC / 2);
  let uv = interleave(&u, &v, false);
  let mut luma = vec![0u8; OUT * OUT];
  let mut sink =
    MixedSinker::<Nv12, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_luma(&mut luma)
      .unwrap();
  // Feed row 3 first: the luma stream expects row 0.
  let err = sink
    .process(Nv12Row::for_tests(
      &y[3 * SRC..4 * SRC],
      &uv[SRC..2 * SRC],
      3,
      KernelMatrix::Bt601,
      true,
    ))
    .unwrap_err();
  assert!(matches!(
    err,
    MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
  ));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv12_rejected_first_row_does_not_poison_output_retry() {
  // A rejected out-of-sequence FIRST row must store no frozen-output
  // snapshot, so retrying row 0 after attaching a NEW output succeeds
  // instead of tripping ResampleOutputsChanged (the shared
  // `planar_dual_resample` semi-planar path).
  let y = y_ramp();
  let (u, v) = uv_planes(SRC / 2, SRC / 2);
  let uv = interleave(&u, &v, false);
  let mut luma = vec![0u8; OUT * OUT];
  let mut sink =
    MixedSinker::<Nv12, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_luma(&mut luma)
      .unwrap();
  let err = sink
    .process(Nv12Row::for_tests(
      &y[3 * SRC..4 * SRC],
      &uv[SRC..2 * SRC],
      3,
      KernelMatrix::Bt601,
      true,
    ))
    .unwrap_err();
  assert!(matches!(
    err,
    MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
  ));
  let mut rgb = vec![0u8; OUT * OUT * 3];
  sink.set_rgb(&mut rgb).unwrap();
  sink
    .process(Nv12Row::for_tests(
      &y[0..SRC],
      &uv[0..SRC],
      0,
      KernelMatrix::Bt601,
      true,
    ))
    .expect("row 0 must succeed after a rejected out-of-sequence first row");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv12_resample_mid_frame_output_change_rejected() {
  let y = y_ramp();
  let (u, v) = uv_planes(SRC / 2, SRC / 2);
  let uv = interleave(&u, &v, false);
  let mut luma = vec![0u8; OUT * OUT];
  let mut luma2 = vec![0u8; OUT * OUT];
  let mut sink =
    MixedSinker::<Nv12, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_luma(&mut luma)
      .unwrap();
  sink
    .process(Nv12Row::for_tests(
      &y[0..SRC],
      &uv[0..SRC],
      0,
      KernelMatrix::Bt601,
      true,
    ))
    .unwrap();
  // Swap the luma buffer mid-frame: the frozen-output check must reject.
  sink.set_luma(&mut luma2).unwrap();
  let err = sink
    .process(Nv12Row::for_tests(
      &y[SRC..2 * SRC],
      &uv[0..SRC],
      1,
      KernelMatrix::Bt601,
      true,
    ))
    .unwrap_err();
  assert!(matches!(err, MixedSinkerError::ResampleOutputsChanged(_)));
}

// =========================================================================
// NV21 (4:2:0, VU) — must equal NV12 on the same logical chroma
// =========================================================================

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv21_resample_rgb_matches_rgb24_of_converted_frame() {
  let y = y_ramp();
  let (u, v) = uv_planes(SRC / 2, SRC / 2);
  let vu = interleave(&u, &v, true);
  let mut full_rgb = vec![0u8; SRC * SRC * 3];
  {
    let frame = Nv21Frame::new(&y, &vu, SRC as u32, SRC as u32, SRC as u32, SRC as u32);
    let mut sink = MixedSinker::<Nv21>::new(SRC, SRC)
      .with_rgb(&mut full_rgb)
      .unwrap();
    nv21_to(&frame, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
  }
  let mut rgb = vec![0u8; OUT * OUT * 3];
  {
    let frame = Nv21Frame::new(&y, &vu, SRC as u32, SRC as u32, SRC as u32, SRC as u32);
    let mut sink = MixedSinker::<Nv21, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        // Convert-then-bin is the row-stage tier (native default differs
        // by conversion-order rounding — covered in `native_tier`).
        .with_native(false)
        .with_rgb(&mut rgb)
        .unwrap();
    nv21_to(&frame, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
  }
  assert_eq!(
    rgb,
    rgb24_rgb_reference(&full_rgb),
    "nv21 rgb: convert-then-bin"
  );
}

// =========================================================================
// NV16 (4:2:2, UV)
// =========================================================================

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv16_resample_rgb_matches_rgb24_of_converted_frame() {
  let y = y_ramp();
  let (u, v) = uv_planes(SRC / 2, SRC);
  let uv = interleave(&u, &v, false);
  let mut full_rgb = vec![0u8; SRC * SRC * 3];
  {
    let frame = Nv16Frame::new(&y, &uv, SRC as u32, SRC as u32, SRC as u32, SRC as u32);
    let mut sink = MixedSinker::<Nv16>::new(SRC, SRC)
      .with_rgb(&mut full_rgb)
      .unwrap();
    nv16_to(&frame, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
  }
  let mut rgb = vec![0u8; OUT * OUT * 3];
  {
    let frame = Nv16Frame::new(&y, &uv, SRC as u32, SRC as u32, SRC as u32, SRC as u32);
    let mut sink = MixedSinker::<Nv16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        // The convert-then-bin RGB contract is the ROW-STAGE tier; the
        // native default averages in the YUV domain (see `native_tier`).
        .with_native(false)
        .with_rgb(&mut rgb)
        .unwrap();
    nv16_to(&frame, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
  }
  assert_eq!(
    rgb,
    rgb24_rgb_reference(&full_rgb),
    "nv16 rgb: convert-then-bin"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv16_resample_luma_is_area_downscaled_y_plane() {
  let y = y_ramp();
  let (u, v) = uv_planes(SRC / 2, SRC);
  let uv = interleave(&u, &v, false);
  let mut luma = vec![0u8; OUT * OUT];
  {
    let frame = Nv16Frame::new(&y, &uv, SRC as u32, SRC as u32, SRC as u32, SRC as u32);
    let mut sink =
      MixedSinker::<Nv16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_luma(&mut luma)
        .unwrap();
    nv16_to(&frame, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
  }
  assert_eq!(luma, block_mean_2x2(&y), "nv16 luma = area-downscaled Y");
}

// =========================================================================
// NV24 (4:4:4, UV) and NV42 (4:4:4, VU)
// =========================================================================

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv24_resample_rgb_matches_rgb24_of_converted_frame() {
  let y = y_ramp();
  let (u, v) = uv_planes(SRC, SRC);
  let uv = interleave(&u, &v, false);
  let mut full_rgb = vec![0u8; SRC * SRC * 3];
  {
    let frame = Nv24Frame::new(
      &y,
      &uv,
      SRC as u32,
      SRC as u32,
      SRC as u32,
      (SRC * 2) as u32,
    );
    let mut sink = MixedSinker::<Nv24>::new(SRC, SRC)
      .with_rgb(&mut full_rgb)
      .unwrap();
    nv24_to(&frame, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
  }
  let mut rgb = vec![0u8; OUT * OUT * 3];
  {
    let frame = Nv24Frame::new(
      &y,
      &uv,
      SRC as u32,
      SRC as u32,
      SRC as u32,
      (SRC * 2) as u32,
    );
    let mut sink = MixedSinker::<Nv24, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        // The convert-then-bin RGB contract is the ROW-STAGE tier; the
        // native default averages in the YUV domain (see `native_tier`).
        .with_native(false)
        .with_rgb(&mut rgb)
        .unwrap();
    nv24_to(&frame, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
  }
  assert_eq!(
    rgb,
    rgb24_rgb_reference(&full_rgb),
    "nv24 rgb: convert-then-bin"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv24_resample_luma_is_area_downscaled_y_plane() {
  let y = y_ramp();
  let (u, v) = uv_planes(SRC, SRC);
  let uv = interleave(&u, &v, false);
  let mut luma = vec![0u8; OUT * OUT];
  {
    let frame = Nv24Frame::new(
      &y,
      &uv,
      SRC as u32,
      SRC as u32,
      SRC as u32,
      (SRC * 2) as u32,
    );
    let mut sink =
      MixedSinker::<Nv24, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_luma(&mut luma)
        .unwrap();
    nv24_to(&frame, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
  }
  assert_eq!(luma, block_mean_2x2(&y), "nv24 luma = area-downscaled Y");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv42_resample_rgb_matches_rgb24_of_converted_frame() {
  let y = y_ramp();
  let (u, v) = uv_planes(SRC, SRC);
  let vu = interleave(&u, &v, true);
  let mut full_rgb = vec![0u8; SRC * SRC * 3];
  {
    let frame = Nv42Frame::new(
      &y,
      &vu,
      SRC as u32,
      SRC as u32,
      SRC as u32,
      (SRC * 2) as u32,
    );
    let mut sink = MixedSinker::<Nv42>::new(SRC, SRC)
      .with_rgb(&mut full_rgb)
      .unwrap();
    nv42_to(&frame, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
  }
  let mut rgb = vec![0u8; OUT * OUT * 3];
  {
    let frame = Nv42Frame::new(
      &y,
      &vu,
      SRC as u32,
      SRC as u32,
      SRC as u32,
      (SRC * 2) as u32,
    );
    let mut sink = MixedSinker::<Nv42, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        // The convert-then-bin RGB contract is the ROW-STAGE tier; the
        // native default averages in the YUV domain (see `native_tier`).
        .with_native(false)
        .with_rgb(&mut rgb)
        .unwrap();
    nv42_to(&frame, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
  }
  assert_eq!(
    rgb,
    rgb24_rgb_reference(&full_rgb),
    "nv42 rgb: convert-then-bin"
  );
}

// =========================================================================
// All-outputs combo (rgb + rgba + luma + luma_u16 + hsv) in one frame —
// each output equals its standalone single-output resample.
// =========================================================================

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv12_resample_all_outputs_match_standalone() {
  let y = y_ramp();
  let (u, v) = uv_planes(SRC / 2, SRC / 2);
  let uv = interleave(&u, &v, false);
  let mk = || Nv12Frame::new(&y, &uv, SRC as u32, SRC as u32, SRC as u32, SRC as u32);

  // Combined sink: every output attached.
  let (mut rgb, mut rgba) = (vec![0u8; OUT * OUT * 3], vec![0u8; OUT * OUT * 4]);
  let (mut luma, mut luma_u16) = (vec![0u8; OUT * OUT], vec![0u16; OUT * OUT]);
  let (mut hh, mut ss, mut vv) = (
    vec![0u8; OUT * OUT],
    vec![0u8; OUT * OUT],
    vec![0u8; OUT * OUT],
  );
  {
    let mut sink =
      MixedSinker::<Nv12, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
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
    nv12_to(&mk(), true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
  }

  // Standalone references.
  let mut rgb_ref = vec![0u8; OUT * OUT * 3];
  {
    let mut s =
      MixedSinker::<Nv12, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb(&mut rgb_ref)
        .unwrap();
    nv12_to(&mk(), true, s.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
  }
  let mut rgba_ref = vec![0u8; OUT * OUT * 4];
  {
    let mut s =
      MixedSinker::<Nv12, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgba(&mut rgba_ref)
        .unwrap();
    nv12_to(&mk(), true, s.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
  }
  let (mut h_ref, mut s_ref, mut v_ref) = (
    vec![0u8; OUT * OUT],
    vec![0u8; OUT * OUT],
    vec![0u8; OUT * OUT],
  );
  {
    let mut s =
      MixedSinker::<Nv12, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_hsv(&mut h_ref, &mut s_ref, &mut v_ref)
        .unwrap();
    nv12_to(&mk(), true, s.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
  }

  assert_eq!(rgb, rgb_ref, "combined rgb == standalone");
  assert_eq!(rgba, rgba_ref, "combined rgba == standalone");
  assert_eq!(
    luma,
    block_mean_2x2(&y),
    "combined luma == area-downscaled Y"
  );
  let y_ref_u16: Vec<u16> = block_mean_2x2(&y).iter().map(|&b| b as u16).collect();
  assert_eq!(luma_u16, y_ref_u16, "combined luma_u16 == zero-extended Y");
  assert_eq!(
    (hh, ss, vv),
    (h_ref, s_ref, v_ref),
    "combined hsv == standalone"
  );
}

// =========================================================================
// Byte-identical parity with the planar twin on the de-interleaved planes
// (the strongest check — requires the yuv-planar family compiled in).
// =========================================================================

#[cfg(feature = "yuv-planar")]
mod twin_parity {
  use super::*;
  use crate::source::{Yuv420p, Yuv444p, yuv420p_to, yuv444p_to};
  use mediaframe::frame::{Yuv420pFrame, Yuv444pFrame};

  /// NV12 row-stage resample must be byte-identical to a Yuv420p
  /// row-stage resample of the de-interleaved planes, for every output.
  /// (The native-tier twin parity lives in the `native_tier` module.)
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn nv12_equals_yuv420p_on_deinterleaved_planes() {
    let y = y_ramp();
    let (u, v) = uv_planes(SRC / 2, SRC / 2);
    let uv = interleave(&u, &v, false);
    let cw = (SRC / 2) as u32;

    let mut nv_rgb = vec![0u8; OUT * OUT * 3];
    let mut nv_luma = vec![0u8; OUT * OUT];
    {
      let frame = Nv12Frame::new(&y, &uv, SRC as u32, SRC as u32, SRC as u32, SRC as u32);
      let mut sink = MixedSinker::<Nv12, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
          .unwrap()
          // Both sides on the row-stage tier so the comparison is the
          // RGB-domain convert-then-bin contract (native-vs-native parity
          // is covered separately in `native_tier`).
          .with_native(false)
          .with_rgb(&mut nv_rgb)
          .unwrap()
          .with_luma(&mut nv_luma)
          .unwrap();
      nv12_to(&frame, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
    }
    let mut p_rgb = vec![0u8; OUT * OUT * 3];
    let mut p_luma = vec![0u8; OUT * OUT];
    {
      let frame = Yuv420pFrame::new(&y, &u, &v, SRC as u32, SRC as u32, SRC as u32, cw, cw);
      let mut sink = MixedSinker::<Yuv420p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
          .unwrap()
          // Force the row-stage tier — the semi-planar path is row-stage.
          .with_native(false)
          .with_rgb(&mut p_rgb)
          .unwrap()
          .with_luma(&mut p_luma)
          .unwrap();
      yuv420p_to(&frame, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
    }
    assert_eq!(nv_rgb, p_rgb, "nv12 rgb == yuv420p(row-stage) rgb");
    assert_eq!(nv_luma, p_luma, "nv12 luma == yuv420p luma");
  }

  /// NV24 ROW-STAGE resample must be byte-identical to a Yuv444p row-stage
  /// resample of the de-interleaved planes. Both sides pin the row-stage
  /// tier (NV24 and Yuv444p now both default to their native fast tier) so
  /// the comparison is the RGB-domain convert-then-bin contract. (The NV24
  /// native-vs-Yuv444p-native parity is covered in the `native_tier`
  /// module's `nv24_native_equals_yuv444p_native_on_deinterleaved_planes`.)
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn nv24_equals_yuv444p_on_deinterleaved_planes() {
    let y = y_ramp();
    let (u, v) = uv_planes(SRC, SRC);
    let uv = interleave(&u, &v, false);

    let mut nv_rgb = vec![0u8; OUT * OUT * 3];
    {
      let frame = Nv24Frame::new(
        &y,
        &uv,
        SRC as u32,
        SRC as u32,
        SRC as u32,
        (SRC * 2) as u32,
      );
      let mut sink =
        MixedSinker::<Nv24, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
          .unwrap()
          .with_native(false)
          .with_rgb(&mut nv_rgb)
          .unwrap();
      nv24_to(&frame, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
    }
    let mut p_rgb = vec![0u8; OUT * OUT * 3];
    {
      let frame = Yuv444pFrame::new(
        &y, &u, &v, SRC as u32, SRC as u32, SRC as u32, SRC as u32, SRC as u32,
      );
      let mut sink = MixedSinker::<Yuv444p, AreaResampler>::with_resampler(
        SRC,
        SRC,
        AreaResampler::to(OUT, OUT),
      )
      .unwrap()
      .with_native(false)
      .with_rgb(&mut p_rgb)
      .unwrap();
      yuv444p_to(&frame, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
    }
    assert_eq!(nv_rgb, p_rgb, "nv24 rgb == yuv444p rgb");
  }
}

// =========================================================================
// Native fast-tier (P2 bin-then-convert) for the semi-planar family — gated
// on the planar family, whose joins the semi-planar native path reuses:
// NV12 / NV21 via the 4:2:0 join, NV16 / NV24 / NV42 via the non-4:2:0
// (4:2:2 / 4:4:4) join on the de-interleaved chroma planes.
//
// The bar (per member) is: (a) BYTE-IDENTICAL to the matching planar NATIVE
// conversion of the de-interleaved planes (the de-interleave-then-reuse
// claim — NV12/NV21 vs Yuv420p, NV16 vs Yuv422p, NV24/NV42 vs Yuv444p);
// (b) within rounding tolerance of the semi-planar ROW-STAGE tier on in-gamut
// content (the conversion-order caveat the planar tiers carry); (c) EXACT
// (full-res conversion) on constant planes; plus the chroma-row,
// odd/tail-width, and recoverable-allocation / atomicity contracts.
// =========================================================================

#[cfg(feature = "yuv-planar")]
mod native_tier {
  use super::*;
  use crate::source::{Yuv420p, yuv420p_to};
  use mediaframe::frame::Yuv420pFrame;

  /// A wider textured fixture (12x10 luma, 6x5 chroma) so the native and
  /// row-stage tiers exercise real conversion math at several geometries.
  /// Values stay interior to the limited-range gamut: near the clamp
  /// boundary the two tiers diverge by more than rounding (the documented
  /// out-of-gamut caveat), which would mask a genuine regression.
  fn textured(w: usize, h: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let cw = w / 2;
    let ch = h.div_ceil(2);
    let y: Vec<u8> = (0..w * h).map(|i| 60 + (i % 64) as u8).collect();
    let u: Vec<u8> = (0..cw * ch).map(|i| 110 + (i % 24) as u8).collect();
    let v: Vec<u8> = (0..cw * ch).map(|i| 120 + (i % 24) as u8).collect();
    (y, u, v)
  }

  /// `(rgb, rgba, luma, luma_u16, hsv_h, hsv_s, hsv_v)` of one NV12 native
  /// or row-stage downscale of the interleaved frame.
  type Outs = (
    Vec<u8>,
    Vec<u8>,
    Vec<u8>,
    Vec<u16>,
    Vec<u8>,
    Vec<u8>,
    Vec<u8>,
  );

  #[allow(clippy::too_many_arguments)]
  fn run_nv12(
    y: &[u8],
    uv: &[u8],
    w: usize,
    h: usize,
    ow: usize,
    oh: usize,
    full_range: bool,
    matrix: KernelMatrix,
    native: bool,
  ) -> Outs {
    let n = ow * oh;
    let mut rgb = vec![0u8; n * 3];
    let mut rgba = vec![0u8; n * 4];
    let mut luma = vec![0u8; n];
    let mut luma_u16 = vec![0u16; n];
    let (mut hh, mut ss, mut vv) = (vec![0u8; n], vec![0u8; n], vec![0u8; n]);
    {
      let frame = Nv12Frame::new(y, uv, w as u32, h as u32, w as u32, w as u32);
      let mut sink =
        MixedSinker::<Nv12, AreaResampler>::with_resampler(w, h, AreaResampler::to(ow, oh))
          .unwrap()
          .with_native(native)
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
      nv12_to(&frame, full_range, sink.set_kernel_matrix(matrix)).unwrap();
    }
    (rgb, rgba, luma, luma_u16, hh, ss, vv)
  }

  /// `with_native(true)` must be the builder default for the semi-planar
  /// family, just as for the planar twin.
  #[test]
  fn native_is_default_on() {
    let sink =
      MixedSinker::<Nv12, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4)).unwrap();
    assert!(sink.native(), "with_native must default to true");
    assert!(
      !sink.with_native(false).native(),
      "with_native(false) must disable the tier"
    );
  }

  /// (a) The strongest check: NV12 native is byte-identical to a Yuv420p
  /// NATIVE conversion of the de-interleaved planes, for every output.
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn nv12_native_equals_yuv420p_native_on_deinterleaved_planes() {
    for (w, h, ow, oh) in [
      (8usize, 8usize, 4usize, 4usize),
      (12, 10, 5, 4),
      (12, 10, 7, 6),
      (8, 9, 3, 4),
    ] {
      let (u, v) = {
        let cw = w / 2;
        let ch = h.div_ceil(2);
        let mut u = vec![0u8; cw * ch];
        let mut v = vec![0u8; cw * ch];
        for (i, p) in u.iter_mut().enumerate() {
          *p = 70 + ((i % cw) as u8).wrapping_mul(5);
        }
        for (i, p) in v.iter_mut().enumerate() {
          *p = 200u8.wrapping_sub(((i % cw) as u8).wrapping_mul(4));
        }
        (u, v)
      };
      let y: Vec<u8> = (0..w * h)
        .map(|i| 40u8.wrapping_add((i as u8).wrapping_mul(2)))
        .collect();
      let uv = interleave(&u, &v, false);
      let cw = (w / 2) as u32;

      let nv = run_nv12(&y, &uv, w, h, ow, oh, true, KernelMatrix::Bt601, true);

      let n = ow * oh;
      let mut p_rgb = vec![0u8; n * 3];
      let mut p_luma = vec![0u8; n];
      {
        let frame = Yuv420pFrame::new(&y, &u, &v, w as u32, h as u32, w as u32, cw, cw);
        let mut sink =
          MixedSinker::<Yuv420p, AreaResampler>::with_resampler(w, h, AreaResampler::to(ow, oh))
            .unwrap()
            .with_native(true)
            .with_rgb(&mut p_rgb)
            .unwrap()
            .with_luma(&mut p_luma)
            .unwrap();
        yuv420p_to(&frame, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
      }
      assert_eq!(
        nv.0, p_rgb,
        "nv12 native rgb == yuv420p native rgb ({w}x{h}->{ow}x{oh})"
      );
      assert_eq!(
        nv.2, p_luma,
        "nv12 native luma == yuv420p native luma ({w}x{h}->{ow}x{oh})"
      );
    }
  }

  /// NV21 native (VU order) de-interleaves to the SAME logical U / V, so
  /// it too must equal the Yuv420p native conversion of those planes.
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn nv21_native_equals_yuv420p_native_on_deinterleaved_planes() {
    let (w, h, ow, oh) = (12, 10, 5, 4);
    let (u, v) = uv_planes(w / 2, h / 2);
    let y: Vec<u8> = (0..w * h)
      .map(|i| 40u8.wrapping_add((i as u8).wrapping_mul(2)))
      .collect();
    let vu = interleave(&u, &v, true);
    let cw = (w / 2) as u32;

    let n = ow * oh;
    let mut nv_rgb = vec![0u8; n * 3];
    {
      let frame = Nv21Frame::new(&y, &vu, w as u32, h as u32, w as u32, w as u32);
      let mut sink =
        MixedSinker::<Nv21, AreaResampler>::with_resampler(w, h, AreaResampler::to(ow, oh))
          .unwrap()
          .with_native(true)
          .with_rgb(&mut nv_rgb)
          .unwrap();
      nv21_to(&frame, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
    }
    let mut p_rgb = vec![0u8; n * 3];
    {
      let frame = Yuv420pFrame::new(&y, &u, &v, w as u32, h as u32, w as u32, cw, cw);
      let mut sink =
        MixedSinker::<Yuv420p, AreaResampler>::with_resampler(w, h, AreaResampler::to(ow, oh))
          .unwrap()
          .with_native(true)
          .with_rgb(&mut p_rgb)
          .unwrap();
      yuv420p_to(&frame, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
    }
    assert_eq!(nv_rgb, p_rgb, "nv21 native rgb == yuv420p native rgb");
  }

  /// (b) Native vs row-stage: luma is bit-identical (both bin the same Y
  /// plane), and in-gamut colour diverges only by per-pixel rounding /
  /// clamping inside the affine conversion. Bound matches the planar twin
  /// (`native_and_row_stage_color_within_tolerance`) because NV12 equals
  /// Yuv420p on BOTH tiers, so the cross-tier delta is identical. Swept
  /// over matrices and the limited/full range flag.
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn nv12_native_within_tolerance_of_row_stage() {
    let (w, h) = (12, 10);
    let (yp, up, vp) = textured(w, h);
    let uv = interleave(&up, &vp, false);
    for (ow, oh) in [(6, 5), (4, 4), (7, 6), (5, 3)] {
      for full_range in [false, true] {
        for matrix in [
          KernelMatrix::Bt601,
          KernelMatrix::Bt709,
          KernelMatrix::Bt2020Ncl,
        ] {
          let native = run_nv12(&yp, &uv, w, h, ow, oh, full_range, matrix, true);
          let row = run_nv12(&yp, &uv, w, h, ow, oh, full_range, matrix, false);
          assert_eq!(
            native.2, row.2,
            "luma bit-identical {ow}x{oh} fr={full_range} {matrix:?}"
          );
          assert_eq!(
            native.3, row.3,
            "luma_u16 bit-identical {ow}x{oh} fr={full_range} {matrix:?}"
          );
          for (name, a, b) in [
            ("rgb", &native.0, &row.0),
            ("rgba", &native.1, &row.1),
            ("hsv-v", &native.6, &row.6),
          ] {
            for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
              assert!(
                x.abs_diff(*y) <= 3,
                "{name} {ow}x{oh} fr={full_range} {matrix:?} idx {i}: native {x} vs row {y}"
              );
            }
          }
        }
      }
    }
  }

  /// (c) Constant planes bin exactly on both grids, so native reproduces
  /// the full-resolution conversion EXACTLY (the true 0-LSB case) — the
  /// cv2 INTER_AREA analogue of the planar `native_solid_frame_exact`.
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn nv12_native_solid_frame_exact() {
    let (w, h) = (8, 8);
    let yp = vec![120u8; w * h];
    let up = vec![90u8; (w / 2) * (h / 2)];
    let vp = vec![170u8; (w / 2) * (h / 2)];
    let uv = interleave(&up, &vp, false);

    let mut full_rgb = vec![0u8; w * h * 3];
    {
      let frame = Nv12Frame::new(&yp, &uv, w as u32, h as u32, w as u32, w as u32);
      let mut sink = MixedSinker::<Nv12>::new(w, h)
        .with_rgb(&mut full_rgb)
        .unwrap();
      nv12_to(&frame, false, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
    }
    let out = run_nv12(&yp, &uv, w, h, 4, 4, false, KernelMatrix::Bt709, true);
    for px in out.0.chunks_exact(3) {
      assert_eq!(
        (px[0], px[1], px[2]),
        (full_rgb[0], full_rgb[1], full_rgb[2]),
        "native solid rgb == full-res conversion"
      );
    }
    assert!(out.2.iter().all(|&l| l == 120), "native solid luma == Y");
  }

  /// Native and row-stage agree even when only luma is attached (the
  /// chroma stream is absent — the join must never touch the empty U / V
  /// scratch). Exercises the luma-only native fast path.
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn nv12_native_luma_only_matches_row_stage() {
    let (w, h) = (12, 10);
    let (yp, up, vp) = textured(w, h);
    let uv = interleave(&up, &vp, false);
    let run = |native: bool| {
      let mut luma = vec![0u8; 6 * 5];
      let frame = Nv12Frame::new(&yp, &uv, w as u32, h as u32, w as u32, w as u32);
      let mut sink =
        MixedSinker::<Nv12, AreaResampler>::with_resampler(w, h, AreaResampler::to(6, 5))
          .unwrap()
          .with_native(native)
          .with_luma(&mut luma)
          .unwrap();
      nv12_to(&frame, false, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
      luma
    };
    assert_eq!(run(true), run(false), "luma-only native == row-stage");
  }

  /// A rejected out-of-sequence FIRST row on the native tier must store no
  /// frozen-output snapshot, so retrying row 0 after attaching a NEW output
  /// succeeds (the join's preflight, reused verbatim). Also proves the
  /// de-interleave touched no caller output on the rejected call.
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn nv12_native_rejected_first_row_does_not_poison_retry() {
    let (w, h) = (8, 8);
    let y = y_ramp_for(w, h);
    let (u, v) = uv_planes(w / 2, h / 2);
    let uv = interleave(&u, &v, false);
    let mut luma = vec![0u8; 4 * 4];
    let mut sink =
      MixedSinker::<Nv12, AreaResampler>::with_resampler(w, h, AreaResampler::to(4, 4))
        .unwrap()
        .with_native(true)
        .with_luma(&mut luma)
        .unwrap();
    let err = sink
      .process(Nv12Row::for_tests(
        &y[3 * w..4 * w],
        &uv[w..2 * w],
        3,
        KernelMatrix::Bt601,
        true,
      ))
      .unwrap_err();
    assert!(matches!(
      err,
      MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
    ));
    let mut rgb = vec![0u8; 4 * 4 * 3];
    sink.set_rgb(&mut rgb).unwrap();
    sink
      .process(Nv12Row::for_tests(
        &y[0..w],
        &uv[0..w],
        0,
        KernelMatrix::Bt601,
        true,
      ))
      .expect("row 0 must succeed after a rejected out-of-sequence first row");
  }

  /// An out-of-sequence ODD first row with COLOUR attached and no
  /// `begin_frame` (so the U / V scratch is still empty): the de-interleave
  /// is skipped on odd rows, so the join must receive empty chroma slices
  /// and reject the row rather than the slice indexing past the empty
  /// scratch. Regression guard for the no-panic contract on direct callers.
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn nv12_native_color_odd_first_row_rejected_without_panic() {
    let (w, h) = (8, 8);
    let y = y_ramp_for(w, h);
    let (u, v) = uv_planes(w / 2, h / 2);
    let uv = interleave(&u, &v, false);
    let mut rgb = vec![0u8; 4 * 4 * 3];
    let mut sink =
      MixedSinker::<Nv12, AreaResampler>::with_resampler(w, h, AreaResampler::to(4, 4))
        .unwrap()
        .with_native(true)
        .with_rgb(&mut rgb)
        .unwrap();
    let err = sink
      .process(Nv12Row::for_tests(
        &y[w..2 * w],
        &uv[w..2 * w],
        1,
        KernelMatrix::Bt601,
        true,
      ))
      .unwrap_err();
    assert!(matches!(
      err,
      MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
    ));
    assert!(
      rgb.iter().all(|&b| b == 0),
      "rejected row touched no output"
    );
  }

  /// An out-of-sequence EVEN (chroma-bearing) first row with COLOUR
  /// attached must be rejected by the join's first-row preflight BEFORE the
  /// U / V de-interleave scratch is reserved — so under allocation pressure
  /// it returns the deterministic `OutOfSequenceRow`, never
  /// `AllocationFailed`, and grows no sink state (the preflight-atomicity /
  /// recoverable-allocation contract). The de-interleave alloc failpoint is
  /// armed on the reserve that an even colour row WOULD reach: with the
  /// wrapper ordered correctly the preflight fires first and the failpoint
  /// is never consumed; with the de-interleave ordered ahead of the
  /// preflight (the bug) the armed reserve refuses and the call surfaces
  /// `AllocationFailed` instead — which this test forbids. (A plain
  /// capacity check would pass even with the bug, since the reserve
  /// succeeds under normal allocation and `OutOfSequenceRow` is still
  /// returned afterwards — hence forcing the failure.)
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn nv12_native_color_even_first_row_rejected_before_scratch_alloc() {
    let (w, h) = (8, 8);
    let y = y_ramp_for(w, h);
    let (u, v) = uv_planes(w / 2, h / 2);
    let uv = interleave(&u, &v, false);
    let mut rgb = vec![0u8; 4 * 4 * 3];
    let mut sink =
      MixedSinker::<Nv12, AreaResampler>::with_resampler(w, h, AreaResampler::to(4, 4))
        .unwrap()
        .with_native(true)
        .with_rgb(&mut rgb)
        .unwrap();
    // Feed an EVEN out-of-sequence first row (idx 2; the join expects 0)
    // with colour attached, so the chroma-bearing de-interleave path is
    // live. No `begin_frame`, so the U / V scratch starts empty and the
    // reserve below WOULD run — but only if the preflight let it.
    crate::sinker::mixed::semi_planar_8bit::arm_deinterleave_alloc_failure();
    let err = sink
      .process(Nv12Row::for_tests(
        &y[2 * w..3 * w],
        &uv[w..2 * w],
        2,
        KernelMatrix::Bt601,
        true,
      ))
      .unwrap_err();
    assert!(
      matches!(
        err,
        MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
      ),
      "even colour first row must reject as OutOfSequenceRow before the \
       de-interleave scratch alloc, got {err:?}"
    );
    assert!(
      rgb.iter().all(|&b| b == 0),
      "rejected even colour first row touched no output"
    );
    // The failpoint is single-shot (take-on-read). It must NOT have been
    // consumed: the preflight rejected the row before the reserve, so the
    // next chroma-bearing reserve is the first to see it. Prove that by
    // running a valid frame and asserting it now refuses with the still-set
    // flag — confirming the rejected row never reached the reserve.
    let mut rgb2 = vec![0u8; 4 * 4 * 3];
    let mut sink2 =
      MixedSinker::<Nv12, AreaResampler>::with_resampler(w, h, AreaResampler::to(4, 4))
        .unwrap()
        .with_native(true)
        .with_rgb(&mut rgb2)
        .unwrap();
    let err2 = sink2
      .process(Nv12Row::for_tests(
        &y[0..w],
        &uv[0..w],
        0,
        KernelMatrix::Bt601,
        true,
      ))
      .unwrap_err();
    assert!(
      matches!(
        err2,
        MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
      ),
      "armed failpoint must still be live (unconsumed by the rejected \
       row) and fire on the first in-sequence even colour reserve, got {err2:?}"
    );
  }

  /// A mid-frame output-set change on a chroma-bearing even row must be
  /// rejected by the wrapper's FULL preflight (the frozen-output check)
  /// BEFORE the U / V de-interleave scratch is reserved — so under
  /// allocation pressure it returns the deterministic
  /// `ResampleOutputsChanged`, never `AllocationFailed`, and grows no sink
  /// state. A luma-only first frame (rows 0, 1 — no colour) freezes a
  /// luma-only output set; RGB is then attached and an even row 2 is fed
  /// with colour, so the chroma-bearing de-interleave path is live. With
  /// the wrapper running only the partial preflight (or the de-interleave
  /// ordered ahead of the frozen check) the armed reserve refuses and the
  /// call surfaces `AllocationFailed` — which this test forbids. (The
  /// failpoint is forced because a plain capacity check would succeed under
  /// normal allocation and `ResampleOutputsChanged` would still follow,
  /// masking the ordering bug.)
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn nv12_native_frozen_mid_frame_change_rejected_before_scratch_alloc() {
    let (w, h) = (8, 8);
    let y = y_ramp_for(w, h);
    let (u, v) = uv_planes(w / 2, h / 2);
    let uv = interleave(&u, &v, false);
    let mut luma = vec![0u8; 4 * 4];
    let mut rgb = vec![0u8; 4 * 4 * 3];
    let mut sink =
      MixedSinker::<Nv12, AreaResampler>::with_resampler(w, h, AreaResampler::to(4, 4))
        .unwrap()
        .with_native(true)
        .with_luma(&mut luma)
        .unwrap();
    // Luma-only first frame: rows 0 and 1 freeze a luma-only output set
    // (no colour, so no de-interleave runs yet).
    for r in 0..2 {
      sink
        .process(Nv12Row::for_tests(
          &y[r * w..(r + 1) * w],
          &uv[r * w..(r + 1) * w],
          r,
          KernelMatrix::Bt601,
          true,
        ))
        .expect("luma-only rows freeze a luma-only output set");
    }
    // Attach RGB mid-frame, changing the output set. Arm the de-interleave
    // failpoint on the reserve the chroma-bearing even row WOULD reach:
    // with the full preflight first the frozen check fires and the
    // failpoint is never consumed; with the de-interleave ahead of the
    // frozen check (the bug) the armed reserve refuses as AllocationFailed.
    sink.set_rgb(&mut rgb).unwrap();
    crate::sinker::mixed::semi_planar_8bit::arm_deinterleave_alloc_failure();
    let err = sink
      .process(Nv12Row::for_tests(
        &y[2 * w..3 * w],
        &uv[2 * w..3 * w],
        2,
        KernelMatrix::Bt601,
        true,
      ))
      .unwrap_err();
    assert!(
      matches!(err, MixedSinkerError::ResampleOutputsChanged(_)),
      "mid-frame output change on an even colour row must reject as \
       ResampleOutputsChanged before the de-interleave scratch alloc, got {err:?}"
    );
    assert!(
      rgb.iter().all(|&b| b == 0),
      "rejected mid-frame-change row touched no colour output"
    );
    // The failpoint is single-shot (take-on-read). It must NOT have been
    // consumed: the frozen check rejected the row before the reserve, so
    // the next chroma-bearing reserve is the first to see it. Prove that by
    // running a fresh in-sequence even colour row and asserting it now
    // refuses with the still-set flag — confirming the rejected row never
    // reached the reserve.
    let mut rgb2 = vec![0u8; 4 * 4 * 3];
    let mut sink2 =
      MixedSinker::<Nv12, AreaResampler>::with_resampler(w, h, AreaResampler::to(4, 4))
        .unwrap()
        .with_native(true)
        .with_rgb(&mut rgb2)
        .unwrap();
    let err2 = sink2
      .process(Nv12Row::for_tests(
        &y[0..w],
        &uv[0..w],
        0,
        KernelMatrix::Bt601,
        true,
      ))
      .unwrap_err();
    assert!(
      matches!(
        err2,
        MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
      ),
      "armed failpoint must still be live (unconsumed by the rejected \
       mid-frame-change row) and fire on the first in-sequence even colour \
       reserve, got {err2:?}"
    );
  }

  /// Sequence-rejection after a RECOVERABLE de-interleave allocation failure:
  /// because the wrapper now runs the COMPARE-ONLY preflight (the commit is
  /// owned by the folded `yuv420p_process_native`), a de-interleave OOM on a
  /// chroma-bearing even row 0 leaves `resample_outputs` UNFROZEN and the join
  /// unbuilt (`native_420 == None`, since the join is created inside the delegate
  /// AFTER the de-interleave) — the de-interleave stays a genuine pre-commit
  /// step. A later OUT-OF-SEQUENCE even row must still reject as the
  /// deterministic `OutOfSequenceRow`, never `AllocationFailed`: outputs are not
  /// frozen, so the preflight's pre-compare first-row sequence check stands
  /// between the out-of-sequence row and the wrapper's fallible de-interleave
  /// reserve; without it the re-armed failpoint would fire first and surface
  /// `AllocationFailed`. The re-arm is proven unconsumed by a subsequent
  /// in-sequence even colour row that DOES fire it.
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn nv12_native_oos_after_recoverable_dealloc_failure_rejected_before_scratch_alloc() {
    let (w, h) = (8, 8);
    let y = y_ramp_for(w, h);
    let (u, v) = uv_planes(w / 2, h / 2);
    let uv = interleave(&u, &v, false);
    let mut rgb = vec![0u8; 4 * 4 * 3];
    let mut sink =
      MixedSinker::<Nv12, AreaResampler>::with_resampler(w, h, AreaResampler::to(4, 4))
        .unwrap()
        .with_native(true)
        .with_rgb(&mut rgb)
        .unwrap();
    // Step 1 — a RECOVERABLE de-interleave failure on the in-sequence even
    // colour row 0. The compare-only preflight clears WITHOUT freezing, then the
    // armed de-interleave reserve refuses: AllocationFailed. This leaves
    // `resample_outputs = None` (unfrozen — the delegate owns the commit, which
    // the pre-commit de-interleave failure never reaches) AND `native_420 = None`
    // (the join is built inside the delegate, after the de-interleave) — the
    // exact recoverable state the sequence check must defend.
    crate::sinker::mixed::semi_planar_8bit::arm_deinterleave_alloc_failure();
    let err0 = sink
      .process(Nv12Row::for_tests(
        &y[0..w],
        &uv[0..w],
        0,
        KernelMatrix::Bt601,
        true,
      ))
      .unwrap_err();
    assert!(
      matches!(
        err0,
        MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
      ),
      "the recoverable de-interleave failure on even row 0 must surface \
       AllocationFailed (leaving outputs unfrozen and the join unbuilt), got {err0:?}"
    );
    // (`rgb` stays borrowed by `sink` across step 2; the no-output contract
    // on a rejected row is already covered by the two tests above — this
    // test isolates the OutOfSequenceRow-vs-AllocationFailed precedence and
    // the failpoint's survival. The final buffer is checked after step 2.)
    // Step 2 — RE-ARM the failpoint, then feed an OUT-OF-SEQUENCE even row
    // (idx 2; the unbuilt join still expects 0) with colour. Outputs are unfrozen
    // (step 1 never committed), so the preflight's pre-compare first-row sequence
    // check is the gate; it must reject as OutOfSequenceRow BEFORE the wrapper
    // reaches the (re-armed) de-interleave reserve.
    crate::sinker::mixed::semi_planar_8bit::arm_deinterleave_alloc_failure();
    let err2 = sink
      .process(Nv12Row::for_tests(
        &y[2 * w..3 * w],
        &uv[2 * w..3 * w],
        2,
        KernelMatrix::Bt601,
        true,
      ))
      .unwrap_err();
    assert!(
      matches!(
        err2,
        MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
      ),
      "an out-of-sequence even row after a recoverable de-interleave failure \
       must reject as OutOfSequenceRow (the pre-compare sequence check), never \
       AllocationFailed, got {err2:?}"
    );
    // Neither the recoverable-failure row 0 nor the rejected out-of-sequence
    // row touched the colour output (`sink` is done with `rgb` here).
    assert!(
      rgb.iter().all(|&b| b == 0),
      "neither the recoverable-failure nor the out-of-sequence row touched \
       the colour output"
    );
    // Step 3 — the failpoint armed in step 2 must NOT have been consumed:
    // the post-freeze check rejected the out-of-sequence row before the
    // de-interleave reserve. Prove it by feeding an in-sequence even colour
    // row 0 (a fresh sink, same frozen RGB shape) — the still-armed flag
    // fires on the FIRST reserve it reaches, confirming step 2 never did.
    let mut rgb2 = vec![0u8; 4 * 4 * 3];
    let mut sink2 =
      MixedSinker::<Nv12, AreaResampler>::with_resampler(w, h, AreaResampler::to(4, 4))
        .unwrap()
        .with_native(true)
        .with_rgb(&mut rgb2)
        .unwrap();
    let err3 = sink2
      .process(Nv12Row::for_tests(
        &y[0..w],
        &uv[0..w],
        0,
        KernelMatrix::Bt601,
        true,
      ))
      .unwrap_err();
    assert!(
      matches!(
        err3,
        MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
      ),
      "the failpoint re-armed in step 2 must still be live (the rejected \
       out-of-sequence row never reached the reserve) and fire on the first \
       in-sequence even colour reserve, got {err3:?}"
    );
  }

  /// A no-output native call must be a no-op `Ok` regardless of row index
  /// (no freeze, no Y feed, no de-interleave alloc), so a later row 0 after
  /// attaching an output succeeds — the join's no-output short-circuit.
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn nv12_native_no_output_row_does_not_poison_retry() {
    let (w, h) = (8, 8);
    let y = y_ramp_for(w, h);
    let (u, v) = uv_planes(w / 2, h / 2);
    let uv = interleave(&u, &v, false);
    let mut sink =
      MixedSinker::<Nv12, AreaResampler>::with_resampler(w, h, AreaResampler::to(4, 4))
        .unwrap()
        .with_native(true);
    sink
      .process(Nv12Row::for_tests(
        &y[0..w],
        &uv[0..w],
        0,
        KernelMatrix::Bt601,
        true,
      ))
      .expect("a no-output native row must be a no-op Ok");
    let mut luma = vec![0u8; 4 * 4];
    sink.set_luma(&mut luma).unwrap();
    sink
      .process(Nv12Row::for_tests(
        &y[0..w],
        &uv[0..w],
        0,
        KernelMatrix::Bt601,
        true,
      ))
      .expect("row 0 must succeed after a no-output native call");
  }

  /// Swapping an output buffer mid-frame on the native tier desyncs the
  /// frozen-output set and must be rejected atomically (no caller mutation).
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn nv12_native_mid_frame_output_change_rejected() {
    let (w, h) = (8, 8);
    let y = y_ramp_for(w, h);
    let (u, v) = uv_planes(w / 2, h / 2);
    let uv = interleave(&u, &v, false);
    let mut luma = vec![0u8; 4 * 4];
    let mut luma2 = vec![0u8; 4 * 4];
    let mut sink =
      MixedSinker::<Nv12, AreaResampler>::with_resampler(w, h, AreaResampler::to(4, 4))
        .unwrap()
        .with_native(true)
        .with_luma(&mut luma)
        .unwrap();
    sink
      .process(Nv12Row::for_tests(
        &y[0..w],
        &uv[0..w],
        0,
        KernelMatrix::Bt601,
        true,
      ))
      .unwrap();
    sink.set_luma(&mut luma2).unwrap();
    let err = sink
      .process(Nv12Row::for_tests(
        &y[w..2 * w],
        &uv[w..2 * w],
        1,
        KernelMatrix::Bt601,
        true,
      ))
      .unwrap_err();
    assert!(matches!(err, MixedSinkerError::ResampleOutputsChanged(_)));
    assert!(luma2.iter().all(|&l| l == 0), "swapped buffer untouched");
  }

  /// Native survives a frame restart on a reused sink: `begin_frame` resets
  /// the join, so a second frame downscales its own planes correctly.
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn nv12_native_reuses_join_across_frames() {
    let (w, h) = (8, 8);
    let y1 = y_ramp_for(w, h);
    let mut y2 = y1.clone();
    for p in y2.iter_mut() {
      *p = 255 - *p;
    }
    let (u, v) = uv_planes(w / 2, h / 2);
    let uv = interleave(&u, &v, false);
    let mut luma = vec![0u8; 4 * 4];
    {
      let mut sink =
        MixedSinker::<Nv12, AreaResampler>::with_resampler(w, h, AreaResampler::to(4, 4))
          .unwrap()
          .with_native(true)
          .with_luma(&mut luma)
          .unwrap();
      let f1 = Nv12Frame::new(&y1, &uv, w as u32, h as u32, w as u32, w as u32);
      let f2 = Nv12Frame::new(&y2, &uv, w as u32, h as u32, w as u32, w as u32);
      nv12_to(&f1, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
      nv12_to(&f2, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
    }
    // Frame 2's luma must area-downscale frame 2's Y (2x2 block mean).
    let mut expect = vec![0u8; 4 * 4];
    for oy in 0..4 {
      for ox in 0..4 {
        let mut s = 0u32;
        for dy in 0..2 {
          for dx in 0..2 {
            s += y2[(oy * 2 + dy) * w + ox * 2 + dx] as u32;
          }
        }
        expect[oy * 4 + ox] = ((s + 2) / 4) as u8;
      }
    }
    assert_eq!(luma, expect, "frame 2 native luma == area-downscaled Y2");
  }

  /// A deterministic Y ramp over a `w`x`h` grid (the fixed-`SRC` `y_ramp`
  /// helper above only covers the 8x8 module geometry).
  fn y_ramp_for(w: usize, h: usize) -> Vec<u8> {
    (0..w * h)
      .map(|i| 40u8.wrapping_add((i as u8).wrapping_mul(2)))
      .collect()
  }

  // ---- frozen native-vs-row-stage route (issue #186) --------------------
  //
  // NV12 / NV21 share the planar twin's native join via
  // `semi_planar_process_native`; the row-stage tier is
  // `planar_dual_resample`. The two carry independent, in-order, once-only
  // stream state, so a mid-frame `set_native` flip must reject as the
  // deterministic `NativeRouteChanged` — CHECKED before and frozen after
  // dispatch, both gated on whether the call bears output (the P0xx
  // template).

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn nv12_native_to_rowstage_route_flip_mid_frame_rejected() {
    let y = y_ramp();
    let (u, v) = uv_planes(SRC / 2, SRC / 2);
    let uv = interleave(&u, &v, false);
    let mut luma = vec![0u8; OUT * OUT];
    let mut sink =
      MixedSinker::<Nv12, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_native(true)
        .with_luma(&mut luma)
        .unwrap();
    sink.begin_frame(SRC as u32, SRC as u32).unwrap();
    // Row 0 freezes the route = native.
    sink
      .process(Nv12Row::for_tests(
        &y[0..SRC],
        &uv[0..SRC],
        0,
        KernelMatrix::Bt601,
        true,
      ))
      .expect("native row 0 freezes the route and succeeds");
    // Flip to the row-stage tier and feed the next in-sequence row.
    sink.set_native(false);
    let err = sink
      .process(Nv12Row::for_tests(
        &y[SRC..2 * SRC],
        &uv[0..SRC],
        1,
        KernelMatrix::Bt601,
        true,
      ))
      .unwrap_err();
    assert!(
      matches!(err, MixedSinkerError::NativeRouteChanged(_)),
      "a native -> row-stage mid-frame route flip must reject as \
       NativeRouteChanged, got {err:?}"
    );
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn nv12_rowstage_to_native_route_flip_mid_frame_rejected() {
    let y = y_ramp();
    let (u, v) = uv_planes(SRC / 2, SRC / 2);
    let uv = interleave(&u, &v, false);
    let mut luma = vec![0u8; OUT * OUT];
    let mut sink =
      MixedSinker::<Nv12, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_native(false)
        .with_luma(&mut luma)
        .unwrap();
    sink.begin_frame(SRC as u32, SRC as u32).unwrap();
    // Row 0 freezes the route = row-stage.
    sink
      .process(Nv12Row::for_tests(
        &y[0..SRC],
        &uv[0..SRC],
        0,
        KernelMatrix::Bt601,
        true,
      ))
      .expect("row-stage row 0 freezes the route and succeeds");
    // Flip to the native tier and feed the next in-sequence row.
    sink.set_native(true);
    let err = sink
      .process(Nv12Row::for_tests(
        &y[SRC..2 * SRC],
        &uv[0..SRC],
        1,
        KernelMatrix::Bt601,
        true,
      ))
      .unwrap_err();
    assert!(
      matches!(err, MixedSinkerError::NativeRouteChanged(_)),
      "a row-stage -> native mid-frame route flip must reject as \
       NativeRouteChanged, got {err:?}"
    );
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn nv21_native_to_rowstage_route_flip_mid_frame_rejected() {
    // The NV21 VU-order twin must guard identically.
    let y = y_ramp();
    let (u, v) = uv_planes(SRC / 2, SRC / 2);
    let vu = interleave(&u, &v, true);
    let mut luma = vec![0u8; OUT * OUT];
    let mut sink =
      MixedSinker::<Nv21, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_native(true)
        .with_luma(&mut luma)
        .unwrap();
    sink.begin_frame(SRC as u32, SRC as u32).unwrap();
    sink
      .process(Nv21Row::for_tests(
        &y[0..SRC],
        &vu[0..SRC],
        0,
        KernelMatrix::Bt601,
        true,
      ))
      .expect("native row 0 freezes the route and succeeds");
    sink.set_native(false);
    let err = sink
      .process(Nv21Row::for_tests(
        &y[SRC..2 * SRC],
        &vu[0..SRC],
        1,
        KernelMatrix::Bt601,
        true,
      ))
      .unwrap_err();
    assert!(
      matches!(err, MixedSinkerError::NativeRouteChanged(_)),
      "nv21: a native -> row-stage mid-frame route flip must reject as \
       NativeRouteChanged, got {err:?}"
    );
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn nv21_rowstage_to_native_route_flip_mid_frame_rejected() {
    // The NV21 VU-order twin must guard identically in the other direction.
    let y = y_ramp();
    let (u, v) = uv_planes(SRC / 2, SRC / 2);
    let vu = interleave(&u, &v, true);
    let mut luma = vec![0u8; OUT * OUT];
    let mut sink =
      MixedSinker::<Nv21, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_native(false)
        .with_luma(&mut luma)
        .unwrap();
    sink.begin_frame(SRC as u32, SRC as u32).unwrap();
    // Row 0 freezes the route = row-stage.
    sink
      .process(Nv21Row::for_tests(
        &y[0..SRC],
        &vu[0..SRC],
        0,
        KernelMatrix::Bt601,
        true,
      ))
      .expect("row-stage row 0 freezes the route and succeeds");
    // Flip to the native tier and feed the next in-sequence row.
    sink.set_native(true);
    let err = sink
      .process(Nv21Row::for_tests(
        &y[SRC..2 * SRC],
        &vu[0..SRC],
        1,
        KernelMatrix::Bt601,
        true,
      ))
      .unwrap_err();
    assert!(
      matches!(err, MixedSinkerError::NativeRouteChanged(_)),
      "nv21: a row-stage -> native mid-frame route flip must reject as \
       NativeRouteChanged, got {err:?}"
    );
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn nv21_route_constant_succeeds_and_resets_across_frames() {
    // The NV21 twin: a constant-route frame runs to completion, and the
    // per-frame reset (in `begin_frame`) lets the NEXT frame pick the
    // OTHER tier.
    let y = y_ramp();
    let (u, v) = uv_planes(SRC / 2, SRC / 2);
    let vu = interleave(&u, &v, true);
    let mut luma = vec![0u8; OUT * OUT];
    let mut sink =
      MixedSinker::<Nv21, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_native(true)
        .with_luma(&mut luma)
        .unwrap();
    let frame = Nv21Frame::new(&y, &vu, SRC as u32, SRC as u32, SRC as u32, SRC as u32);
    // Frame 1: native, route constant across every row — no false rejection.
    nv21_to(&frame, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
    // Frame 2: flip to row-stage for the WHOLE frame. The walker's
    // `begin_frame` cleared the frozen route, so this is allowed.
    sink.set_native(false);
    nv21_to(&frame, true, sink.set_kernel_matrix(KernelMatrix::Bt601))
      .expect("a new frame may pick the other tier; the route reset per frame");
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn nv21_no_output_call_after_frozen_route_is_a_noop() {
    // The NV21 twin: a NO-OUTPUT call after an output-bearing row froze the
    // route must be a TRUE no-op — route-invisible — even when `set_native`
    // is FLIPPED: it returns `Ok` (not `NativeRouteChanged`) and leaves the
    // frozen route untouched (both the CHECK and the SET gate on
    // `need_output`). No public API detaches an output, so we set
    // `frozen_native_route` directly to the value an accepted output-bearing
    // native first row stores (`Some(true)` = native), the same white-box
    // reach the atomicity tests use.
    let y = y_ramp();
    let (u, v) = uv_planes(SRC / 2, SRC / 2);
    let vu = interleave(&u, &v, true);
    let mut sink =
      MixedSinker::<Nv21, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_native(true);
    sink.begin_frame(SRC as u32, SRC as u32).unwrap();
    sink.frozen_native_route = Some(true);
    // No-output row (no outputs -> `need_output` false), route flipped to
    // row-stage. The CHECK is skipped, so this is a true no-op.
    sink.set_native(false);
    sink
      .process(Nv21Row::for_tests(
        &y[0..SRC],
        &vu[0..SRC],
        0,
        KernelMatrix::Bt601,
        true,
      ))
      .expect("a no-output call after a frozen route must be a true no-op, not NativeRouteChanged");
    assert_eq!(
      sink.frozen_native_route,
      Some(true),
      "a no-output call must leave the frozen route unchanged"
    );
    // The route is STILL native and consumed no stream state: an
    // output-bearing native row 0 succeeds...
    let mut luma = vec![0u8; OUT * OUT];
    sink.set_native(true);
    sink.set_luma(&mut luma).unwrap();
    sink
      .process(Nv21Row::for_tests(
        &y[0..SRC],
        &vu[0..SRC],
        0,
        KernelMatrix::Bt601,
        true,
      ))
      .expect("an output-bearing row under the original native route succeeds");
    // ...while an output-bearing flip to the OTHER route now rejects.
    sink.set_native(false);
    let err = sink
      .process(Nv21Row::for_tests(
        &y[SRC..2 * SRC],
        &vu[0..SRC],
        1,
        KernelMatrix::Bt601,
        true,
      ))
      .unwrap_err();
    assert!(
      matches!(err, MixedSinkerError::NativeRouteChanged(_)),
      "nv21: an output-bearing flip after the frozen route stayed native \
       must reject as NativeRouteChanged, got {err:?}"
    );
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn nv12_route_constant_succeeds_and_resets_across_frames() {
    // A constant-route frame runs to completion, and the per-frame reset
    // (in `begin_frame`) lets the NEXT frame pick the OTHER tier.
    let y = y_ramp();
    let (u, v) = uv_planes(SRC / 2, SRC / 2);
    let uv = interleave(&u, &v, false);
    let mut luma = vec![0u8; OUT * OUT];
    let mut sink =
      MixedSinker::<Nv12, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_native(true)
        .with_luma(&mut luma)
        .unwrap();
    let frame = Nv12Frame::new(&y, &uv, SRC as u32, SRC as u32, SRC as u32, SRC as u32);
    // Frame 1: native, route constant across every row — no false rejection.
    nv12_to(&frame, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
    // Frame 2: flip to row-stage for the WHOLE frame. The walker's
    // `begin_frame` cleared the frozen route, so this is allowed.
    sink.set_native(false);
    nv12_to(&frame, true, sink.set_kernel_matrix(KernelMatrix::Bt601))
      .expect("a new frame may pick the other tier; the route reset per frame");
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn nv12_no_output_call_after_frozen_route_is_a_noop() {
    // A NO-OUTPUT call after an output-bearing row froze the route must be
    // a TRUE no-op — route-invisible — even when `set_native` is FLIPPED:
    // it returns `Ok` (not `NativeRouteChanged`) and leaves the frozen
    // route untouched (both the CHECK and the SET gate on `need_output`).
    // No public API detaches an output, so we set `frozen_native_route`
    // directly to the value an accepted output-bearing native first row
    // stores (`Some(true)` = native), the same white-box reach the
    // atomicity tests use.
    let y = y_ramp();
    let (u, v) = uv_planes(SRC / 2, SRC / 2);
    let uv = interleave(&u, &v, false);
    let mut sink =
      MixedSinker::<Nv12, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_native(true);
    sink.begin_frame(SRC as u32, SRC as u32).unwrap();
    sink.frozen_native_route = Some(true);
    // No-output row (no outputs -> `need_output` false), route flipped to
    // row-stage. The CHECK is skipped, so this is a true no-op.
    sink.set_native(false);
    sink
      .process(Nv12Row::for_tests(
        &y[0..SRC],
        &uv[0..SRC],
        0,
        KernelMatrix::Bt601,
        true,
      ))
      .expect("a no-output call after a frozen route must be a true no-op, not NativeRouteChanged");
    assert_eq!(
      sink.frozen_native_route,
      Some(true),
      "a no-output call must leave the frozen route unchanged"
    );
    // The route is STILL native and consumed no stream state: an
    // output-bearing native row 0 succeeds...
    let mut luma = vec![0u8; OUT * OUT];
    sink.set_native(true);
    sink.set_luma(&mut luma).unwrap();
    sink
      .process(Nv12Row::for_tests(
        &y[0..SRC],
        &uv[0..SRC],
        0,
        KernelMatrix::Bt601,
        true,
      ))
      .expect("an output-bearing row under the original native route succeeds");
    // ...while an output-bearing flip to the OTHER route now rejects.
    sink.set_native(false);
    let err = sink
      .process(Nv12Row::for_tests(
        &y[SRC..2 * SRC],
        &uv[0..SRC],
        1,
        KernelMatrix::Bt601,
        true,
      ))
      .unwrap_err();
    assert!(
      matches!(err, MixedSinkerError::NativeRouteChanged(_)),
      "an output-bearing flip after the frozen route stayed native must \
       reject as NativeRouteChanged, got {err:?}"
    );
  }

  // ---- Non-4:2:0 native fast tier (NV16 4:2:2 / NV24 4:4:4 UV /
  // NV42 4:4:4 VU) — issue #123 -------------------------------------------
  //
  // These reuse the non-4:2:0 planar join (`yuv_planar_process_native`) on
  // the de-interleaved chroma planes (chroma_vsub == 1: a chroma row per Y
  // row, vs the 4:2:0 even-only cadence). Same bar as the 4:2:0 members,
  // re-pointed at the matching planar twin: NV16 vs Yuv422p, NV24 / NV42 vs
  // Yuv444p; NV42 de-interleaves its VU chroma with U / V swapped.

  use crate::source::{Yuv422p, Yuv444p, yuv422p_to, yuv444p_to};
  use mediaframe::frame::{Yuv422pFrame, Yuv444pFrame};

  /// Textured U / V planes at chroma width `cw`, FULL chroma height `h`
  /// (4:2:2 / 4:4:4 have a chroma row per Y row). Interior to the
  /// limited-range gamut so the two tiers diverge only by rounding.
  fn uv_full_height(cw: usize, h: usize) -> (Vec<u8>, Vec<u8>) {
    let u: Vec<u8> = (0..cw * h).map(|i| 110 + (i % 24) as u8).collect();
    let v: Vec<u8> = (0..cw * h).map(|i| 120 + (i % 24) as u8).collect();
    (u, v)
  }

  /// One all-outputs NV16 downscale of the interleaved (UV) frame.
  #[allow(clippy::too_many_arguments)]
  fn run_nv16(
    y: &[u8],
    uv: &[u8],
    w: usize,
    h: usize,
    ow: usize,
    oh: usize,
    full_range: bool,
    matrix: KernelMatrix,
    native: bool,
  ) -> Outs {
    let n = ow * oh;
    let mut rgb = vec![0u8; n * 3];
    let mut rgba = vec![0u8; n * 4];
    let mut luma = vec![0u8; n];
    let mut luma_u16 = vec![0u16; n];
    let (mut hh, mut ss, mut vv) = (vec![0u8; n], vec![0u8; n], vec![0u8; n]);
    {
      let frame = Nv16Frame::new(y, uv, w as u32, h as u32, w as u32, w as u32);
      let mut sink =
        MixedSinker::<Nv16, AreaResampler>::with_resampler(w, h, AreaResampler::to(ow, oh))
          .unwrap()
          .with_native(native)
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
      nv16_to(&frame, full_range, sink.set_kernel_matrix(matrix)).unwrap();
    }
    (rgb, rgba, luma, luma_u16, hh, ss, vv)
  }

  /// One all-outputs NV24 downscale of the interleaved (UV) frame.
  #[allow(clippy::too_many_arguments)]
  fn run_nv24(
    y: &[u8],
    uv: &[u8],
    w: usize,
    h: usize,
    ow: usize,
    oh: usize,
    full_range: bool,
    matrix: KernelMatrix,
    native: bool,
  ) -> Outs {
    let n = ow * oh;
    let mut rgb = vec![0u8; n * 3];
    let mut rgba = vec![0u8; n * 4];
    let mut luma = vec![0u8; n];
    let mut luma_u16 = vec![0u16; n];
    let (mut hh, mut ss, mut vv) = (vec![0u8; n], vec![0u8; n], vec![0u8; n]);
    {
      let frame = Nv24Frame::new(y, uv, w as u32, h as u32, w as u32, (w * 2) as u32);
      let mut sink =
        MixedSinker::<Nv24, AreaResampler>::with_resampler(w, h, AreaResampler::to(ow, oh))
          .unwrap()
          .with_native(native)
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
      nv24_to(&frame, full_range, sink.set_kernel_matrix(matrix)).unwrap();
    }
    (rgb, rgba, luma, luma_u16, hh, ss, vv)
  }

  /// (a) The strongest check: NV16 native is byte-identical to a Yuv422p
  /// NATIVE conversion of the de-interleaved planes, for every output, at
  /// several geometries.
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn nv16_native_equals_yuv422p_native_on_deinterleaved_planes() {
    for (w, h, ow, oh) in [
      (8usize, 8usize, 4usize, 4usize),
      (12, 10, 5, 4),
      (12, 10, 7, 6),
    ] {
      let cw = w / 2;
      let (u, v) = uv_full_height(cw, h);
      let y: Vec<u8> = (0..w * h)
        .map(|i| 40u8.wrapping_add((i as u8).wrapping_mul(2)))
        .collect();
      let uv = interleave(&u, &v, false);

      let nv = run_nv16(&y, &uv, w, h, ow, oh, true, KernelMatrix::Bt601, true);

      let n = ow * oh;
      let mut p_rgb = vec![0u8; n * 3];
      let mut p_luma = vec![0u8; n];
      {
        let frame = Yuv422pFrame::new(
          &y, &u, &v, w as u32, h as u32, w as u32, cw as u32, cw as u32,
        );
        let mut sink =
          MixedSinker::<Yuv422p, AreaResampler>::with_resampler(w, h, AreaResampler::to(ow, oh))
            .unwrap()
            .with_native(true)
            .with_rgb(&mut p_rgb)
            .unwrap()
            .with_luma(&mut p_luma)
            .unwrap();
        yuv422p_to(&frame, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
      }
      assert_eq!(
        nv.0, p_rgb,
        "nv16 native rgb == yuv422p native rgb ({w}x{h}->{ow}x{oh})"
      );
      assert_eq!(
        nv.2, p_luma,
        "nv16 native luma == yuv422p native luma ({w}x{h}->{ow}x{oh})"
      );
    }
  }

  /// (a) NV24 native is byte-identical to a Yuv444p NATIVE conversion of the
  /// de-interleaved planes, for every output.
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn nv24_native_equals_yuv444p_native_on_deinterleaved_planes() {
    for (w, h, ow, oh) in [
      (8usize, 8usize, 4usize, 4usize),
      (12, 10, 5, 4),
      (12, 10, 7, 6),
    ] {
      let (u, v) = uv_full_height(w, h);
      let y: Vec<u8> = (0..w * h)
        .map(|i| 40u8.wrapping_add((i as u8).wrapping_mul(2)))
        .collect();
      let uv = interleave(&u, &v, false);

      let nv = run_nv24(&y, &uv, w, h, ow, oh, true, KernelMatrix::Bt601, true);

      let n = ow * oh;
      let mut p_rgb = vec![0u8; n * 3];
      let mut p_luma = vec![0u8; n];
      {
        let frame = Yuv444pFrame::new(&y, &u, &v, w as u32, h as u32, w as u32, w as u32, w as u32);
        let mut sink =
          MixedSinker::<Yuv444p, AreaResampler>::with_resampler(w, h, AreaResampler::to(ow, oh))
            .unwrap()
            .with_native(true)
            .with_rgb(&mut p_rgb)
            .unwrap()
            .with_luma(&mut p_luma)
            .unwrap();
        yuv444p_to(&frame, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
      }
      assert_eq!(
        nv.0, p_rgb,
        "nv24 native rgb == yuv444p native rgb ({w}x{h}->{ow}x{oh})"
      );
      assert_eq!(
        nv.2, p_luma,
        "nv24 native luma == yuv444p native luma ({w}x{h}->{ow}x{oh})"
      );
    }
  }

  /// (a) NV42 native (VU order) de-interleaves to the SAME logical U / V, so
  /// it too must equal the Yuv444p native conversion of those planes — the
  /// VU-order regression guard (a wrong swap would map U<->V and diverge).
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn nv42_native_equals_yuv444p_native_on_deinterleaved_planes() {
    let (w, h, ow, oh) = (12, 10, 5, 4);
    let (u, v) = uv_full_height(w, h);
    let y: Vec<u8> = (0..w * h)
      .map(|i| 40u8.wrapping_add((i as u8).wrapping_mul(2)))
      .collect();
    // VU-order interleave (V before U) — the wire layout NV42 carries.
    let vu = interleave(&u, &v, true);

    let n = ow * oh;
    let mut nv_rgb = vec![0u8; n * 3];
    {
      let frame = Nv42Frame::new(&y, &vu, w as u32, h as u32, w as u32, (w * 2) as u32);
      let mut sink =
        MixedSinker::<Nv42, AreaResampler>::with_resampler(w, h, AreaResampler::to(ow, oh))
          .unwrap()
          .with_native(true)
          .with_rgb(&mut nv_rgb)
          .unwrap();
      nv42_to(&frame, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
    }
    let mut p_rgb = vec![0u8; n * 3];
    {
      let frame = Yuv444pFrame::new(&y, &u, &v, w as u32, h as u32, w as u32, w as u32, w as u32);
      let mut sink =
        MixedSinker::<Yuv444p, AreaResampler>::with_resampler(w, h, AreaResampler::to(ow, oh))
          .unwrap()
          .with_native(true)
          .with_rgb(&mut p_rgb)
          .unwrap();
      yuv444p_to(&frame, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
    }
    assert_eq!(
      nv_rgb, p_rgb,
      "nv42 native rgb == yuv444p native rgb (VU de-interleaved to the same U/V)"
    );
  }

  /// (b) Native vs row-stage: luma is bit-identical (both bin the same Y
  /// plane), and in-gamut colour diverges only by per-pixel rounding inside
  /// the affine conversion (bound 3, matching the 4:2:0 sweep). Run for both
  /// NV16 (4:2:2) and NV24 (4:4:4) across matrices and the range flag.
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn non420_native_within_tolerance_of_row_stage() {
    let (w, h) = (12, 10);
    let y: Vec<u8> = (0..w * h).map(|i| 60 + (i % 64) as u8).collect();
    let (u16p, v16p) = uv_full_height(w / 2, h);
    let uv16 = interleave(&u16p, &v16p, false);
    let (u24p, v24p) = uv_full_height(w, h);
    let uv24 = interleave(&u24p, &v24p, false);
    for (ow, oh) in [(6, 5), (4, 4), (7, 6), (5, 3)] {
      for full_range in [false, true] {
        for matrix in [
          KernelMatrix::Bt601,
          KernelMatrix::Bt709,
          KernelMatrix::Bt2020Ncl,
        ] {
          for (tag, native, row) in [
            (
              "nv16",
              run_nv16(&y, &uv16, w, h, ow, oh, full_range, matrix, true),
              run_nv16(&y, &uv16, w, h, ow, oh, full_range, matrix, false),
            ),
            (
              "nv24",
              run_nv24(&y, &uv24, w, h, ow, oh, full_range, matrix, true),
              run_nv24(&y, &uv24, w, h, ow, oh, full_range, matrix, false),
            ),
          ] {
            assert_eq!(
              native.2, row.2,
              "{tag} luma bit-identical {ow}x{oh} fr={full_range} {matrix:?}"
            );
            assert_eq!(
              native.3, row.3,
              "{tag} luma_u16 bit-identical {ow}x{oh} fr={full_range} {matrix:?}"
            );
            for (name, a, b) in [
              ("rgb", &native.0, &row.0),
              ("rgba", &native.1, &row.1),
              ("hsv-v", &native.6, &row.6),
            ] {
              for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                assert!(
                  x.abs_diff(*y) <= 3,
                  "{tag} {name} {ow}x{oh} fr={full_range} {matrix:?} idx {i}: \
                   native {x} vs row {y}"
                );
              }
            }
          }
        }
      }
    }
  }

  /// (c) Constant planes bin exactly on both grids, so NV16 / NV24 native
  /// reproduce the full-resolution conversion EXACTLY (the true 0-LSB case).
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn non420_native_solid_frame_exact() {
    let (w, h) = (8, 8);
    let yp = vec![120u8; w * h];
    // NV16: chroma w/2 x h. NV24: chroma w x h. Both constant.
    let up16 = vec![90u8; (w / 2) * h];
    let vp16 = vec![170u8; (w / 2) * h];
    let uv16 = interleave(&up16, &vp16, false);
    let up24 = vec![90u8; w * h];
    let vp24 = vec![170u8; w * h];
    let uv24 = interleave(&up24, &vp24, false);

    // The full-res converted top-left pixel — identical for both subsamplings
    // since chroma is constant (90, 170) everywhere.
    let mut full_rgb = vec![0u8; w * h * 3];
    {
      let frame = Nv24Frame::new(&yp, &uv24, w as u32, h as u32, w as u32, (w * 2) as u32);
      let mut sink = MixedSinker::<Nv24>::new(w, h)
        .with_rgb(&mut full_rgb)
        .unwrap();
      nv24_to(&frame, false, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
    }
    let want = (full_rgb[0], full_rgb[1], full_rgb[2]);

    let nv16 = run_nv16(&yp, &uv16, w, h, 4, 4, false, KernelMatrix::Bt709, true);
    for px in nv16.0.chunks_exact(3) {
      assert_eq!(
        (px[0], px[1], px[2]),
        want,
        "nv16 native solid rgb == full-res conversion"
      );
    }
    assert!(
      nv16.2.iter().all(|&l| l == 120),
      "nv16 native solid luma == Y"
    );

    let nv24 = run_nv24(&yp, &uv24, w, h, 4, 4, false, KernelMatrix::Bt709, true);
    for px in nv24.0.chunks_exact(3) {
      assert_eq!(
        (px[0], px[1], px[2]),
        want,
        "nv24 native solid rgb == full-res conversion"
      );
    }
    assert!(
      nv24.2.iter().all(|&l| l == 120),
      "nv24 native solid luma == Y"
    );
  }

  /// `with_native(true)` is the builder default for the non-4:2:0 members
  /// too (NV16 / NV24 / NV42), just as for the 4:2:0 twins.
  #[test]
  fn non420_native_is_default_on() {
    assert!(
      MixedSinker::<Nv16, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
        .unwrap()
        .native(),
      "nv16 with_native must default to true"
    );
    assert!(
      MixedSinker::<Nv24, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
        .unwrap()
        .native(),
      "nv24 with_native must default to true"
    );
    assert!(
      MixedSinker::<Nv42, AreaResampler>::with_resampler(8, 8, AreaResampler::to(4, 4))
        .unwrap()
        .native(),
      "nv42 with_native must default to true"
    );
  }

  /// Native and row-stage agree even when only luma is attached (the chroma
  /// stream is absent — the join must never touch the empty U / V scratch).
  /// Exercises the luma-only non-4:2:0 native fast path (NV24, chroma w x h).
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn nv24_native_luma_only_matches_row_stage() {
    let (w, h) = (12, 10);
    let y: Vec<u8> = (0..w * h).map(|i| 60 + (i % 64) as u8).collect();
    let (u, v) = uv_full_height(w, h);
    let uv = interleave(&u, &v, false);
    let run = |native: bool| {
      let mut luma = vec![0u8; 6 * 5];
      let frame = Nv24Frame::new(&y, &uv, w as u32, h as u32, w as u32, (w * 2) as u32);
      let mut sink =
        MixedSinker::<Nv24, AreaResampler>::with_resampler(w, h, AreaResampler::to(6, 5))
          .unwrap()
          .with_native(native)
          .with_luma(&mut luma)
          .unwrap();
      nv24_to(&frame, false, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
      luma
    };
    assert_eq!(run(true), run(false), "nv24 luma-only native == row-stage");
  }

  /// The non-4:2:0 members share the planar twin's non-4:2:0 native join via
  /// `semi_planar_process_native_non420`; the row-stage tier is
  /// `planar_dual_resample`. A mid-frame `set_native` flip must reject as the
  /// deterministic `NativeRouteChanged` — CHECKED before and frozen after
  /// dispatch (the #186 template, here on NV16).
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn nv16_native_to_rowstage_route_flip_mid_frame_rejected() {
    let y = y_ramp();
    let (u, v) = uv_planes(SRC / 2, SRC);
    let uv = interleave(&u, &v, false);
    let mut luma = vec![0u8; OUT * OUT];
    let mut sink =
      MixedSinker::<Nv16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_native(true)
        .with_luma(&mut luma)
        .unwrap();
    sink.begin_frame(SRC as u32, SRC as u32).unwrap();
    // Row 0 freezes the route = native.
    sink
      .process(crate::source::Nv16Row::for_tests(
        &y[0..SRC],
        &uv[0..SRC],
        0,
        KernelMatrix::Bt601,
        true,
      ))
      .expect("native row 0 freezes the route and succeeds");
    sink.set_native(false);
    let err = sink
      .process(crate::source::Nv16Row::for_tests(
        &y[SRC..2 * SRC],
        &uv[SRC..2 * SRC],
        1,
        KernelMatrix::Bt601,
        true,
      ))
      .unwrap_err();
    assert!(
      matches!(err, MixedSinkerError::NativeRouteChanged(_)),
      "nv16: a native -> row-stage mid-frame route flip must reject as \
       NativeRouteChanged, got {err:?}"
    );
  }

  /// NV16 native survives a frame restart on a reused sink: `begin_frame`
  /// resets the join + the frozen route, so a second frame (the OTHER tier)
  /// downscales its own planes correctly. Guards the new `native_planar` /
  /// `frozen_native_route` reset wiring in the NV16 `begin_frame`.
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn nv16_native_reuses_join_and_resets_route_across_frames() {
    let y = y_ramp();
    let (u, v) = uv_planes(SRC / 2, SRC);
    let uv = interleave(&u, &v, false);
    let mut luma = vec![0u8; OUT * OUT];
    let mut sink =
      MixedSinker::<Nv16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_native(true)
        .with_luma(&mut luma)
        .unwrap();
    let frame = Nv16Frame::new(&y, &uv, SRC as u32, SRC as u32, SRC as u32, SRC as u32);
    // Frame 1: native, route constant across every row — no false rejection.
    nv16_to(&frame, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
    // Frame 2: flip to row-stage for the WHOLE frame; the per-frame reset
    // (in `begin_frame`) cleared the frozen route, so this is allowed.
    sink.set_native(false);
    nv16_to(&frame, true, sink.set_kernel_matrix(KernelMatrix::Bt601))
      .expect("a new frame may pick the other tier; the route reset per frame");
    assert_eq!(
      luma,
      block_mean_2x2(&y),
      "frame 2 luma == area-downscaled Y"
    );
  }

  /// A luma-only non-4:2:0 semi-planar native sink must NOT plan or allocate
  /// any chroma state — else luma-only Nv16/Nv24/Nv42 resampling depends on an
  /// unused chroma allocation and can fail under memory pressure before
  /// producing luma (a regression vs the Y-only row-stage path). Armed with
  /// the planar-native chroma-planning failpoint (the join is shared with the
  /// planar twin): a luma-only row leaves it unconsumed (so the run succeeds),
  /// while a colour row reaches chroma planning and the failpoint fires.
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn luma_only_non420_native_skips_chroma_planning() {
    let y = y_ramp();
    let (u, v) = uv_planes(SRC / 2, SRC);
    let uv = interleave(&u, &v, false);
    let frame = Nv16Frame::new(&y, &uv, SRC as u32, SRC as u32, SRC as u32, SRC as u32);

    crate::sinker::mixed::arm_planar_native_chroma_failure();

    // Luma-only: the armed chroma failpoint is never reached -> Ok.
    let mut luma = vec![0u8; OUT * OUT];
    {
      let mut sink =
        MixedSinker::<Nv16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
          .unwrap()
          .with_native(true)
          .with_luma(&mut luma)
          .unwrap();
      nv16_to(&frame, true, sink.set_kernel_matrix(KernelMatrix::Bt601))
        .expect("luma-only native must not plan chroma");
    }
    assert_eq!(
      luma,
      block_mean_2x2(&y),
      "luma-only native == area-downscaled Y"
    );

    // Colour: the still-armed failpoint fires at chroma planning -> Err. This
    // both proves the failpoint is wired to chroma planning and consumes the
    // arm so it cannot leak to another test on this thread.
    let mut rgb = vec![0u8; OUT * OUT * 3];
    let mut sink =
      MixedSinker::<Nv16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_native(true)
        .with_rgb(&mut rgb)
        .unwrap();
    assert!(
      nv16_to(&frame, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).is_err(),
      "colour native must reach chroma planning (the armed failpoint fires)"
    );
  }
}
