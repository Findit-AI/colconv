//! Fused-downscale coverage for `Grayf16` — routed through a single 1-channel
//! `AreaStream<f32>` that widens the source `f16` luma plane to `f32` and bins
//! it at f32 precision (Grayf16 *is* an f16 luma plane). The wire row widens to
//! a host-native f32 luma plane first (the same kernel the direct `luma_f32`
//! path uses), then every attached output derives from each finalized binned
//! f32 luma row using the `grayf32` kernels — so every resampled output equals
//! the direct `Grayf32` sink run over a frame that already holds the binned f32
//! luma plane (the binned domain is f32, not f16). The half-float twin of
//! `resample_grayf32`.

use crate::{
  KernelMatrix, PixelSink,
  frame::{Grayf16Frame, Grayf32Frame},
  resample::{AreaResampler, ResampleError},
  sinker::{MixedSinker, MixedSinkerError},
  source::{Grayf16, Grayf16Row, Grayf32, grayf16_to, grayf16_to_endian},
};
use half::f16;

const SRC: usize = 8;
const OUT: usize = 4;
const FR: bool = true;
const M: KernelMatrix = KernelMatrix::Bt709;

/// Re-encode a host-native f16 slice as LE-encoded byte storage (the `grayf16le`
/// plane contract), recovered via `u16::from_le`.
fn as_le_f16(host: &[f16]) -> Vec<f16> {
  host
    .iter()
    .map(|v| f16::from_bits(u16::from_ne_bytes(v.to_bits().to_le_bytes())))
    .collect()
}

/// Re-encode a host-native f16 slice as BE-encoded byte storage (`grayf16be`),
/// recovered via `u16::from_be`.
fn as_be_f16(host: &[f16]) -> Vec<f16> {
  host
    .iter()
    .map(|v| f16::from_bits(u16::from_ne_bytes(v.to_bits().to_be_bytes())))
    .collect()
}

/// Re-encode a host-native f32 slice as LE-encoded byte storage (the `grayf32le`
/// plane contract) for the `Grayf32` reference sink over the binned luma.
fn as_le_f32(host: &[f32]) -> Vec<f32> {
  host
    .iter()
    .map(|&v| f32::from_bits(v.to_bits().to_le()))
    .collect()
}

/// Interior f16 luma ramp mixing in-range, HDR (> 1.0), and negative values so
/// the area mean sees real variation per 2x2 block.
fn ramp() -> Vec<f16> {
  let mut y = vec![f16::ZERO; SRC * SRC];
  for (i, p) in y.iter_mut().enumerate() {
    *p = f16::from_f32(match i % 5 {
      0 => i as f32 / 64.0,
      1 => 1.0 + i as f32 / 8.0,
      2 => -(i as f32) / 100.0,
      3 => 0.5,
      _ => 2.5,
    });
  }
  y
}

/// Exact 2x2-block area mean of the *widened* f16 plane (each f16 → f32) to the
/// `OUT` grid. The engine accumulates in `f64` and finalizes `(acc / 4) as f32`
/// — bit-identical to the stream for the uniform 2:1 box.
fn block_mean_2x2(plane: &[f16]) -> Vec<f32> {
  let mut out = vec![0.0f32; OUT * OUT];
  for oy in 0..OUT {
    for ox in 0..OUT {
      let mut s = 0.0f64;
      for dy in 0..2 {
        for dx in 0..2 {
          s += plane[(oy * 2 + dy) * SRC + ox * 2 + dx].to_f32() as f64;
        }
      }
      out[oy * OUT + ox] = (s / 4.0) as f32;
    }
  }
  out
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn grayf16_downscale_luma_f32_is_exact_area_mean() {
  let plane = ramp();
  let pix = as_le_f16(&plane);
  let src = Grayf16Frame::new(&pix, SRC as u32, SRC as u32, SRC as u32);

  let mut luma_f32 = vec![0.0f32; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Grayf16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_luma_f32(&mut luma_f32)
        .unwrap();
    grayf16_to(&src, FR, sink.set_kernel_matrix(M)).unwrap();
  }
  assert_eq!(
    luma_f32,
    block_mean_2x2(&plane),
    "luma_f32 must be the exact 2x2 f32 block mean of the widened f16 plane"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn grayf16_all_outputs_match_direct_grayf32_over_binned_luma() {
  // Every attached output must be exactly what the direct `Grayf32` sink
  // produces over the (exact) binned f32 luma plane — the Grayf16 emit applies
  // the identical grayf32 kernels per finalized binned f32 row.
  let plane = ramp();
  let pix = as_le_f16(&plane);
  let src = Grayf16Frame::new(&pix, SRC as u32, SRC as u32, SRC as u32);

  let mut rgb = vec![0u8; OUT * OUT * 3];
  let mut rgba = vec![0u8; OUT * OUT * 4];
  let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
  let mut rgba_u16 = vec![0u16; OUT * OUT * 4];
  let mut luma = vec![0u8; OUT * OUT];
  let mut luma_u16 = vec![0u16; OUT * OUT];
  let mut rgb_f32 = vec![0.0f32; OUT * OUT * 3];
  let mut luma_f32 = vec![0.0f32; OUT * OUT];
  let mut h = vec![0u8; OUT * OUT];
  let mut s_ = vec![0u8; OUT * OUT];
  let mut v_ = vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Grayf16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
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
        .with_rgb_f32(&mut rgb_f32)
        .unwrap()
        .with_luma_f32(&mut luma_f32)
        .unwrap()
        .with_hsv(&mut h, &mut s_, &mut v_)
        .unwrap();
    grayf16_to(&src, FR, sink.set_kernel_matrix(M)).unwrap();
  }

  // Reference: the direct Grayf32 sink over the exact binned f32 luma plane.
  let binned = block_mean_2x2(&plane);
  let binned_pix = as_le_f32(&binned);
  let mut ref_rgb = vec![0u8; OUT * OUT * 3];
  let mut ref_rgba = vec![0u8; OUT * OUT * 4];
  let mut ref_rgb_u16 = vec![0u16; OUT * OUT * 3];
  let mut ref_rgba_u16 = vec![0u16; OUT * OUT * 4];
  let mut ref_luma = vec![0u8; OUT * OUT];
  let mut ref_luma_u16 = vec![0u16; OUT * OUT];
  let mut ref_rgb_f32 = vec![0.0f32; OUT * OUT * 3];
  let mut ref_luma_f32 = vec![0.0f32; OUT * OUT];
  let mut ref_h = vec![0u8; OUT * OUT];
  let mut ref_s = vec![0u8; OUT * OUT];
  let mut ref_v = vec![0u8; OUT * OUT];
  {
    let binned_frame = Grayf32Frame::new(&binned_pix, OUT as u32, OUT as u32, OUT as u32);
    let mut sink = MixedSinker::<Grayf32>::new(OUT, OUT)
      .with_rgb(&mut ref_rgb)
      .unwrap()
      .with_rgba(&mut ref_rgba)
      .unwrap()
      .with_rgb_u16(&mut ref_rgb_u16)
      .unwrap()
      .with_rgba_u16(&mut ref_rgba_u16)
      .unwrap()
      .with_luma(&mut ref_luma)
      .unwrap()
      .with_luma_u16(&mut ref_luma_u16)
      .unwrap()
      .with_rgb_f32(&mut ref_rgb_f32)
      .unwrap()
      .with_luma_f32(&mut ref_luma_f32)
      .unwrap()
      .with_hsv(&mut ref_h, &mut ref_s, &mut ref_v)
      .unwrap();
    crate::source::grayf32_to(&binned_frame, FR, sink.set_kernel_matrix(M)).unwrap();
  }
  assert_eq!(luma_f32, ref_luma_f32, "luma_f32");
  assert_eq!(rgb_f32, ref_rgb_f32, "rgb_f32");
  assert_eq!(luma, ref_luma, "luma");
  assert_eq!(luma_u16, ref_luma_u16, "luma_u16");
  assert_eq!(rgb, ref_rgb, "rgb");
  assert_eq!(rgba, ref_rgba, "rgba");
  assert_eq!(rgb_u16, ref_rgb_u16, "rgb_u16");
  assert_eq!(rgba_u16, ref_rgba_u16, "rgba_u16");
  assert_eq!(h, ref_h, "h");
  assert_eq!(s_, ref_s, "s");
  assert_eq!(v_, ref_v, "v");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn grayf16_le_be_resample_outputs_identical() {
  // The binned f32 luma is host-native, so the derive kernels run with
  // `HOST_NATIVE_BE`. LE and BE wire encodings of the same logical plane must
  // resample to identical outputs.
  let plane = ramp();
  let pix_le = as_le_f16(&plane);
  let pix_be = as_be_f16(&plane);

  let mut le_luma_f32 = vec![0.0f32; OUT * OUT];
  let mut le_rgb_u16 = vec![0u16; OUT * OUT * 3];
  let mut le_rgba = vec![0u8; OUT * OUT * 4];
  {
    let frame = Grayf16Frame::new(&pix_le, SRC as u32, SRC as u32, SRC as u32);
    let mut sink =
      MixedSinker::<Grayf16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_luma_f32(&mut le_luma_f32)
        .unwrap()
        .with_rgb_u16(&mut le_rgb_u16)
        .unwrap()
        .with_rgba(&mut le_rgba)
        .unwrap();
    grayf16_to(&frame, FR, sink.set_kernel_matrix(M)).unwrap();
  }

  let mut be_luma_f32 = vec![0.0f32; OUT * OUT];
  let mut be_rgb_u16 = vec![0u16; OUT * OUT * 3];
  let mut be_rgba = vec![0u8; OUT * OUT * 4];
  {
    let frame = Grayf16Frame::<true>::new(&pix_be, SRC as u32, SRC as u32, SRC as u32);
    let mut sink = MixedSinker::<Grayf16<true>, AreaResampler>::with_resampler(
      SRC,
      SRC,
      AreaResampler::to(OUT, OUT),
    )
    .unwrap()
    .with_luma_f32(&mut be_luma_f32)
    .unwrap()
    .with_rgb_u16(&mut be_rgb_u16)
    .unwrap()
    .with_rgba(&mut be_rgba)
    .unwrap();
    grayf16_to_endian::<_, true>(&frame, FR, sink.set_kernel_matrix(M)).unwrap();
  }

  assert_eq!(le_luma_f32, be_luma_f32, "luma_f32 LE/BE diverge");
  assert_eq!(le_rgb_u16, be_rgb_u16, "rgb_u16 LE/BE diverge");
  assert_eq!(le_rgba, be_rgba, "rgba LE/BE diverge");
  assert_eq!(
    le_luma_f32,
    block_mean_2x2(&plane),
    "luma_f32 not area mean"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn grayf16_standalone_rgba_matches_direct_over_binned_luma() {
  let plane = ramp();
  let pix = as_le_f16(&plane);
  let src = Grayf16Frame::new(&pix, SRC as u32, SRC as u32, SRC as u32);

  let mut rgba = vec![0u8; OUT * OUT * 4];
  {
    let mut sink =
      MixedSinker::<Grayf16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap();
    grayf16_to(&src, FR, sink.set_kernel_matrix(M)).unwrap();
  }
  let binned_pix = as_le_f32(&block_mean_2x2(&plane));
  let mut ref_rgba = vec![0u8; OUT * OUT * 4];
  {
    let binned = Grayf32Frame::new(&binned_pix, OUT as u32, OUT as u32, OUT as u32);
    let mut sink = MixedSinker::<Grayf32>::new(OUT, OUT)
      .with_rgba(&mut ref_rgba)
      .unwrap();
    crate::source::grayf32_to(&binned, FR, sink.set_kernel_matrix(M)).unwrap();
  }
  assert_eq!(rgba, ref_rgba, "standalone rgba");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn grayf16_identity_plan_matches_new_sink() {
  let plane = ramp();
  let pix = as_le_f16(&plane);
  let src = Grayf16Frame::new(&pix, SRC as u32, SRC as u32, SRC as u32);

  let mut direct = vec![0.0f32; SRC * SRC * 3];
  {
    let mut sink = MixedSinker::<Grayf16>::new(SRC, SRC)
      .with_rgb_f32(&mut direct)
      .unwrap();
    grayf16_to(&src, FR, sink.set_kernel_matrix(M)).unwrap();
  }
  let mut via_area = vec![0.0f32; SRC * SRC * 3];
  {
    let mut sink =
      MixedSinker::<Grayf16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(SRC, SRC))
        .unwrap()
        .with_rgb_f32(&mut via_area)
        .unwrap();
    grayf16_to(&src, FR, sink.set_kernel_matrix(M)).unwrap();
  }
  assert_eq!(direct, via_area, "identity plan must match the direct sink");
}

#[cfg_attr(
  miri,
  ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
)]
#[test]
fn grayf16_resample_no_outputs_is_a_no_op() {
  let plane = ramp();
  let pix = as_le_f16(&plane);
  let src = Grayf16Frame::new(&pix, SRC as u32, SRC as u32, SRC as u32);
  let mut sink =
    MixedSinker::<Grayf16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap();
  grayf16_to(&src, FR, sink.set_kernel_matrix(M)).unwrap();
  assert!(
    !sink.luma_stream_f32_allocated(),
    "no-output sink allocated an f32 luma stream"
  );
}

#[cfg_attr(
  miri,
  ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
)]
#[test]
fn grayf16_out_of_sequence_first_row_rejected_before_allocation() {
  let plane = ramp();
  let pix = as_le_f16(&plane);
  let row3 = &pix[3 * SRC..4 * SRC];

  let mut luma_f32 = vec![0.0f32; OUT * OUT];
  let mut sink =
    MixedSinker::<Grayf16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_luma_f32(&mut luma_f32)
      .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  let err = sink
    .process(Grayf16Row::for_tests(row3, 3, M, FR))
    .unwrap_err();
  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
    ),
    "expected OutOfSequenceRow, got {err:?}"
  );
  assert!(
    !sink.luma_stream_f32_allocated(),
    "stream allocated for a rejected row"
  );
  assert_eq!(
    sink.luma_scratch_f32_capacity(),
    0,
    "source-luma staging grown for a rejected row"
  );
  assert!(
    luma_f32.iter().all(|&b| b == 0.0),
    "rejected row mutated output"
  );
}

#[cfg_attr(
  miri,
  ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
)]
#[test]
fn grayf16_resample_reuses_luma_stream_across_frames() {
  let y1 = ramp();
  let mut y2 = y1.clone();
  for p in y2.iter_mut() {
    *p = f16::from_f32(3.0 - p.to_f32());
  }
  let pix1 = as_le_f16(&y1);
  let pix2 = as_le_f16(&y2);
  let mut luma_f32 = vec![0.0f32; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Grayf16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_luma_f32(&mut luma_f32)
        .unwrap();
    grayf16_to(
      &Grayf16Frame::new(&pix1, SRC as u32, SRC as u32, SRC as u32),
      FR,
      sink.set_kernel_matrix(M),
    )
    .unwrap();
    grayf16_to(
      &Grayf16Frame::new(&pix2, SRC as u32, SRC as u32, SRC as u32),
      FR,
      sink.set_kernel_matrix(M),
    )
    .unwrap();
  }
  assert_eq!(
    luma_f32,
    block_mean_2x2(&y2),
    "frame 2 luma_f32 must area-downscale frame 2's widened luma"
  );
}

#[cfg_attr(
  miri,
  ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
)]
#[test]
fn grayf16_resample_rejects_mid_frame_output_change() {
  let plane = ramp();
  let pix = as_le_f16(&plane);
  let mut rgb_f32 = vec![0.0f32; OUT * OUT * 3];
  let mut luma_f32 = vec![0.0f32; OUT * OUT];
  let mut sink =
    MixedSinker::<Grayf16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_rgb_f32(&mut rgb_f32)
      .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  sink
    .process(Grayf16Row::for_tests(&pix[..SRC], 0, M, FR))
    .unwrap();
  sink.set_luma_f32(&mut luma_f32).unwrap();
  let err = sink
    .process(Grayf16Row::for_tests(&pix[SRC..2 * SRC], 1, M, FR))
    .unwrap_err();
  assert!(
    matches!(err, MixedSinkerError::ResampleOutputsChanged(_)),
    "expected ResampleOutputsChanged, got {err:?}"
  );
  assert!(
    luma_f32.iter().all(|&b| b == 0.0),
    "rejected row mutated the new output"
  );
}
