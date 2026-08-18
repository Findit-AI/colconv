//! Fused-downscale coverage for the high-bit packed 4:4:4 YUV source
//! `V410` (10-bit, one u32 word per pixel, MSB padding, `<const BE>`).
//!
//! `V410` routes through the same three-stream tail as `V30X`; the only
//! differences are the packing (`(V << 20) | (Y << 10) | U`, MSB pad) and
//! the source decode wire endianness (`<const BE>`). See
//! [`resample_v30x`](super::resample_v30x) for the oracle rationale. This
//! suite additionally pins **LE/BE parity** (scalar + SIMD): the binned
//! rows are host-native, so the LE and BE wire encodings of the same
//! logical frame must produce byte-identical resampled output.

use crate::{
  KernelMatrix, PixelSink,
  frame::{V410BeFrame, V410Frame, V410LeFrame},
  resample::{AreaResampler, ResampleError},
  sinker::{MixedSinker, MixedSinkerError},
  source::{V410, V410Row, v410_to, v410_to_endian},
};

use super::{as_be_u32, as_le_u32, force_row_stage};

const SRC: usize = 8;
const OUT: usize = 4;
const M: KernelMatrix = KernelMatrix::Bt709;
const FR: bool = true;
const SHIFT: u32 = 2; // 10-bit native → u8.

/// Packs a logical `(U, Y, V)` 10-bit plane into V410 words:
/// `(V << 20) | (Y << 10) | U` (2-bit MSB padding).
fn pack_v410(u: &[u16], y: &[u16], v: &[u16]) -> Vec<u32> {
  (0..u.len())
    .map(|i| {
      let u = (u[i] & 0x3FF) as u32;
      let y = (y[i] & 0x3FF) as u32;
      let v = (v[i] & 0x3FF) as u32;
      (v << 20) | (y << 10) | u
    })
    .collect()
}

fn v410_frame(buf: &[u32]) -> V410Frame<'_> {
  V410Frame::new(buf, SRC as u32, SRC as u32, SRC as u32)
}

fn yuv_ramp() -> (Vec<u16>, Vec<u16>, Vec<u16>) {
  let mut u = vec![0u16; SRC * SRC];
  let mut y = vec![0u16; SRC * SRC];
  let mut v = vec![0u16; SRC * SRC];
  for i in 0..SRC * SRC {
    y[i] = 180 + (i as u16) * 11;
    u[i] = 320 + (i as u16) * 6;
    v[i] = 820 - (i as u16) * 5;
  }
  (u, y, v)
}

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

fn direct_full(packed: &[u32]) -> (Vec<u8>, Vec<u16>, Vec<u16>) {
  let src = v410_frame(packed);
  let mut rgb_u8 = vec![0u8; SRC * SRC * 3];
  let mut rgb_u16 = vec![0u16; SRC * SRC * 3];
  let mut y_u16 = vec![0u16; SRC * SRC];
  {
    let mut sink = MixedSinker::<V410>::new(SRC, SRC)
      .with_rgb(&mut rgb_u8)
      .unwrap()
      .with_rgb_u16(&mut rgb_u16)
      .unwrap()
      .with_luma_u16(&mut y_u16)
      .unwrap();
    v410_to(&src, FR, M, &mut sink).unwrap();
  }
  (rgb_u8, rgb_u16, y_u16)
}

// ---- The mandatory uniform-gray counterexample ------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn v410_uniform_gray_downscale_leaves_colour_outputs_unchanged() {
  let u = vec![380u16; SRC * SRC];
  let y = vec![640u16; SRC * SRC];
  let v = vec![540u16; SRC * SRC];
  let packed = pack_v410(&u, &y, &v);
  let (full_rgb, full_rgb_u16, _full_y) = direct_full(&packed);

  let mut rgb = vec![0u8; OUT * OUT * 3];
  let mut rgba = vec![0u8; OUT * OUT * 4];
  let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
  let mut hh = vec![0u8; OUT * OUT];
  let mut ss = vec![0u8; OUT * OUT];
  let mut vv = vec![0u8; OUT * OUT];
  {
    let mut sink = force_row_stage(
      MixedSinker::<V410, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap(),
    )
    .with_rgb(&mut rgb)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16)
    .unwrap()
    .with_hsv(&mut hh, &mut ss, &mut vv)
    .unwrap();
    v410_to(&v410_frame(&packed), FR, M, &mut sink).unwrap();
  }

  let gray_px = &full_rgb[..3];
  for px in rgb.chunks_exact(3) {
    assert_eq!(px, gray_px, "uniform-gray u8 RGB changed under downscale");
  }
  for px in rgba.chunks_exact(4) {
    assert_eq!(&px[..3], gray_px, "uniform-gray rgba colour changed");
    assert_eq!(px[3], 0xFF, "rgba alpha");
  }
  let gray_px_u16 = &full_rgb_u16[..3];
  for px in rgb_u16.chunks_exact(3) {
    assert_eq!(
      px, gray_px_u16,
      "uniform-gray u16 RGB changed under downscale"
    );
  }
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
fn v410_downscale_rgb_u16_is_native_depth_block_mean() {
  let (u, y, v) = yuv_ramp();
  let packed = pack_v410(&u, &y, &v);
  let (_full_rgb, full_rgb_u16, _full_y) = direct_full(&packed);

  let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
  {
    let mut sink = force_row_stage(
      MixedSinker::<V410, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap(),
    )
    .with_rgb_u16(&mut rgb_u16)
    .unwrap();
    v410_to(&v410_frame(&packed), FR, M, &mut sink).unwrap();
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
fn v410_downscale_luma_is_native_depth_block_mean_of_y() {
  let (u, y, v) = yuv_ramp();
  let packed = pack_v410(&u, &y, &v);
  let (_full_rgb, _full_rgb_u16, full_y) = direct_full(&packed);

  let mut luma = vec![0u8; OUT * OUT];
  let mut luma_u16 = vec![0u16; OUT * OUT];
  {
    let mut sink = force_row_stage(
      MixedSinker::<V410, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap(),
    )
    .with_luma(&mut luma)
    .unwrap()
    .with_luma_u16(&mut luma_u16)
    .unwrap();
    v410_to(&v410_frame(&packed), FR, M, &mut sink).unwrap();
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
fn v410_all_outputs_match_their_own_native_depth_block_mean() {
  let (u, y, v) = yuv_ramp();
  let packed = pack_v410(&u, &y, &v);
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
    let mut sink = force_row_stage(
      MixedSinker::<V410, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap(),
    )
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
    v410_to(&v410_frame(&packed), FR, M, &mut sink).unwrap();
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
  let mut rgba_u16_ref = vec![0u16; OUT * OUT * 4];
  crate::row::expand_rgb_u16_to_rgba_u16_row::<10>(&rgb_u16_ref, &mut rgba_u16_ref, OUT * OUT);
  assert_eq!(rgba_u16, rgba_u16_ref, "all-outputs rgba_u16");
  assert_eq!(luma_u16, y_binned, "all-outputs luma_u16");
  let luma_ref: Vec<u8> = y_binned.iter().map(|&p| (p >> SHIFT) as u8).collect();
  assert_eq!(luma, luma_ref, "all-outputs luma");
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
fn v410_luma_taken_from_native_y_under_saturated_chroma() {
  let yc = 512u16;
  let u = vec![1000u16; SRC * SRC];
  let y = vec![yc; SRC * SRC];
  let v = vec![20u16; SRC * SRC];
  let packed = pack_v410(&u, &y, &v);
  let mut luma_u16 = vec![0u16; OUT * OUT];
  {
    let mut sink = force_row_stage(
      MixedSinker::<V410, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap(),
    )
    .with_luma_u16(&mut luma_u16)
    .unwrap();
    v410_to(&v410_frame(&packed), FR, M, &mut sink).unwrap();
  }
  assert!(
    luma_u16.iter().all(|&p| p == yc),
    "luma_u16 must be the native Y ({yc}), not RGB-derived; got {luma_u16:?}"
  );
}

// ---- LE/BE parity (scalar + SIMD) -------------------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn v410_resample_le_be_parity() {
  let (u, y, v) = yuv_ramp();
  let host = pack_v410(&u, &y, &v);
  let pix_le: Vec<u32> = host.iter().map(|&w| as_le_u32(w)).collect();
  let pix_be: Vec<u32> = host.iter().map(|&w| as_be_u32(w)).collect();

  for simd in [true, false] {
    let mut le_rgb = vec![0u8; OUT * OUT * 3];
    let mut le_rgb_u16 = vec![0u16; OUT * OUT * 3];
    let mut le_luma_u16 = vec![0u16; OUT * OUT];
    {
      let frame = V410LeFrame::try_new(&pix_le, SRC as u32, SRC as u32, SRC as u32).unwrap();
      let mut sink = force_row_stage(
        MixedSinker::<V410, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
          .unwrap(),
      )
      .with_simd(simd)
      .with_rgb(&mut le_rgb)
      .unwrap()
      .with_rgb_u16(&mut le_rgb_u16)
      .unwrap()
      .with_luma_u16(&mut le_luma_u16)
      .unwrap();
      v410_to(&frame, FR, M, &mut sink).unwrap();
    }

    let mut be_rgb = vec![0u8; OUT * OUT * 3];
    let mut be_rgb_u16 = vec![0u16; OUT * OUT * 3];
    let mut be_luma_u16 = vec![0u16; OUT * OUT];
    {
      let frame = V410BeFrame::try_new(&pix_be, SRC as u32, SRC as u32, SRC as u32).unwrap();
      let mut sink = force_row_stage(
        MixedSinker::<V410<true>, AreaResampler>::with_resampler(
          SRC,
          SRC,
          AreaResampler::to(OUT, OUT),
        )
        .unwrap(),
      )
      .with_simd(simd)
      .with_rgb(&mut be_rgb)
      .unwrap()
      .with_rgb_u16(&mut be_rgb_u16)
      .unwrap()
      .with_luma_u16(&mut be_luma_u16)
      .unwrap();
      v410_to_endian(&frame, FR, M, &mut sink).unwrap();
    }

    assert_eq!(
      le_rgb, be_rgb,
      "V410 LE/BE resample rgb diverge (simd={simd})"
    );
    assert_eq!(
      le_rgb_u16, be_rgb_u16,
      "V410 LE/BE resample rgb_u16 diverge (simd={simd})"
    );
    assert_eq!(
      le_luma_u16, be_luma_u16,
      "V410 LE/BE resample luma_u16 diverge (simd={simd})"
    );
  }
}

// ---- Identity / no-op / reset -----------------------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn v410_identity_plan_matches_new_sink() {
  let (u, y, v) = yuv_ramp();
  let packed = pack_v410(&u, &y, &v);
  let mut direct = vec![0u16; SRC * SRC * 3];
  {
    let mut sink = MixedSinker::<V410>::new(SRC, SRC)
      .with_rgb_u16(&mut direct)
      .unwrap();
    v410_to(&v410_frame(&packed), FR, M, &mut sink).unwrap();
  }
  let mut via_area = vec![0u16; SRC * SRC * 3];
  {
    let mut sink = force_row_stage(
      MixedSinker::<V410, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(SRC, SRC))
        .unwrap(),
    )
    .with_rgb_u16(&mut via_area)
    .unwrap();
    v410_to(&v410_frame(&packed), FR, M, &mut sink).unwrap();
  }
  assert_eq!(direct, via_area, "identity plan must match the direct sink");
}

#[test]
fn v410_no_outputs_is_a_no_op() {
  let (u, y, v) = yuv_ramp();
  let packed = pack_v410(&u, &y, &v);
  let mut sink =
    MixedSinker::<V410, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap();
  v410_to(&v410_frame(&packed), FR, M, &mut sink).unwrap();
  assert!(!sink.rgb_stream_allocated());
  assert!(!sink.rgb_stream_u16_allocated());
  assert!(!sink.luma_stream_u16_allocated());
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn v410_resets_streams_across_frames() {
  let (u, y1, v) = yuv_ramp();
  let mut y2 = y1.clone();
  for p in y2.iter_mut() {
    *p = 1023 - *p;
  }
  let p1 = pack_v410(&u, &y1, &v);
  let p2 = pack_v410(&u, &y2, &v);
  let mut luma_u16 = vec![0u16; OUT * OUT];
  let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
  {
    let mut sink = force_row_stage(
      MixedSinker::<V410, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap(),
    )
    .with_luma_u16(&mut luma_u16)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16)
    .unwrap();
    v410_to(&v410_frame(&p1), FR, M, &mut sink).unwrap();
    v410_to(&v410_frame(&p2), FR, M, &mut sink).unwrap();
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
fn v410_out_of_sequence_first_row_rejected_before_allocation() {
  let (u, y, v) = yuv_ramp();
  let packed = pack_v410(&u, &y, &v);
  let row3 = &packed[3 * SRC..4 * SRC];
  let mut luma_u16 = vec![0u16; OUT * OUT];
  let mut rgb = vec![0u8; OUT * OUT * 3];
  let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
  let mut sink = force_row_stage(
    MixedSinker::<V410, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap(),
  )
  .with_luma_u16(&mut luma_u16)
  .unwrap()
  .with_rgb(&mut rgb)
  .unwrap()
  .with_rgb_u16(&mut rgb_u16)
  .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  let err = sink.process(V410Row::new(row3, 3, M, FR)).unwrap_err();
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
fn v410_rejects_mid_frame_output_change() {
  let (u, y, v) = yuv_ramp();
  let packed = pack_v410(&u, &y, &v);
  let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
  let mut luma_u16 = vec![0u16; OUT * OUT];
  let mut sink = force_row_stage(
    MixedSinker::<V410, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap(),
  )
  .with_rgb_u16(&mut rgb_u16)
  .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  sink
    .process(V410Row::new(&packed[..SRC], 0, M, FR))
    .unwrap();
  sink.set_luma_u16(&mut luma_u16).unwrap();
  let err = sink
    .process(V410Row::new(&packed[SRC..2 * SRC], 1, M, FR))
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
