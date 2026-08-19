//! Fused-downscale coverage for the high-bit packed 4:4:4 YUV source
//! `Xv48` (16-bit, four u16 slots per pixel `[U, Y, V, X]`, full-depth,
//! `<const BE>`).
//!
//! `Xv48` routes through the same three-stream tail as `V30X` / `V410` /
//! `Xv36`; the differences are the layout (4 u16 per pixel, full 16-bit
//! native — no MSB shift), `BITS = 16` (so the binned native Y narrows
//! `>> 8` to u8, the native alpha max is `0xFFFF`, and the u16 RGB stream
//! uses i64 chroma), and the `<const BE>` decode wire. See
//! [`resample_xv36`](super::resample_xv36) for the oracle rationale; this
//! suite additionally pins LE/BE parity (scalar + SIMD).

use crate::{
  KernelMatrix, PixelSink,
  frame::{Xv48BeFrame, Xv48Frame, Xv48LeFrame},
  resample::{AreaResampler, ResampleError},
  sinker::{MixedSinker, MixedSinkerError},
  source::{Xv48, Xv48Row, xv48_to, xv48_to_endian},
};

use super::{as_be_u16, as_le_u16, force_row_stage};

const SRC: usize = 8;
const OUT: usize = 4;
const M: KernelMatrix = KernelMatrix::Bt709;
const FR: bool = true;
const SHIFT: u32 = 8; // 16-bit native → u8.

/// Packs a logical `(U, Y, V)` 16-bit plane (X is padding) into XV48
/// `[U, Y, V, X]` u16 quadruples (full 16-bit native — no shift).
fn pack_xv48(u: &[u16], y: &[u16], v: &[u16]) -> Vec<u16> {
  let mut out = vec![0u16; u.len() * 4];
  for i in 0..u.len() {
    out[i * 4] = u[i];
    out[i * 4 + 1] = y[i];
    out[i * 4 + 2] = v[i];
    out[i * 4 + 3] = 0;
  }
  out
}

fn xv48_frame(buf: &[u16]) -> Xv48Frame<'_> {
  Xv48Frame::new(buf, SRC as u32, SRC as u32, (SRC * 4) as u32)
}

fn yuv_ramp() -> (Vec<u16>, Vec<u16>, Vec<u16>) {
  let mut u = vec![0u16; SRC * SRC];
  let mut y = vec![0u16; SRC * SRC];
  let mut v = vec![0u16; SRC * SRC];
  for i in 0..SRC * SRC {
    y[i] = 11200 + (i as u16) * 640;
    u[i] = 22400 + (i as u16) * 320;
    v[i] = 51200 - (i as u16) * 288;
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

fn direct_full(packed: &[u16]) -> (Vec<u8>, Vec<u16>, Vec<u16>) {
  let src = xv48_frame(packed);
  let mut rgb_u8 = vec![0u8; SRC * SRC * 3];
  let mut rgb_u16 = vec![0u16; SRC * SRC * 3];
  let mut y_u16 = vec![0u16; SRC * SRC];
  {
    let mut sink = MixedSinker::<Xv48>::new(SRC, SRC)
      .with_rgb(&mut rgb_u8)
      .unwrap()
      .with_rgb_u16(&mut rgb_u16)
      .unwrap()
      .with_luma_u16(&mut y_u16)
      .unwrap();
    xv48_to(&src, FR, sink.set_kernel_matrix(M)).unwrap();
  }
  (rgb_u8, rgb_u16, y_u16)
}

// ---- The mandatory uniform-gray counterexample ------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn xv48_uniform_gray_downscale_leaves_colour_outputs_unchanged() {
  let u = vec![25600u16; SRC * SRC];
  let y = vec![40960u16; SRC * SRC];
  let v = vec![35200u16; SRC * SRC];
  let packed = pack_xv48(&u, &y, &v);
  let (full_rgb, full_rgb_u16, _full_y) = direct_full(&packed);

  let mut rgb = vec![0u8; OUT * OUT * 3];
  let mut rgba = vec![0u8; OUT * OUT * 4];
  let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
  let mut hh = vec![0u8; OUT * OUT];
  let mut ss = vec![0u8; OUT * OUT];
  let mut vv = vec![0u8; OUT * OUT];
  {
    let mut sink = force_row_stage(
      MixedSinker::<Xv48, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
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
    xv48_to(&xv48_frame(&packed), FR, sink.set_kernel_matrix(M)).unwrap();
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
fn xv48_downscale_rgb_u16_is_native_depth_block_mean() {
  let (u, y, v) = yuv_ramp();
  let packed = pack_xv48(&u, &y, &v);
  let (_full_rgb, full_rgb_u16, _full_y) = direct_full(&packed);

  let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
  {
    let mut sink = force_row_stage(
      MixedSinker::<Xv48, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap(),
    )
    .with_rgb_u16(&mut rgb_u16)
    .unwrap();
    xv48_to(&xv48_frame(&packed), FR, sink.set_kernel_matrix(M)).unwrap();
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
fn xv48_downscale_luma_is_native_depth_block_mean_of_y() {
  let (u, y, v) = yuv_ramp();
  let packed = pack_xv48(&u, &y, &v);
  let (_full_rgb, _full_rgb_u16, full_y) = direct_full(&packed);

  let mut luma = vec![0u8; OUT * OUT];
  let mut luma_u16 = vec![0u16; OUT * OUT];
  {
    let mut sink = force_row_stage(
      MixedSinker::<Xv48, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap(),
    )
    .with_luma(&mut luma)
    .unwrap()
    .with_luma_u16(&mut luma_u16)
    .unwrap();
    xv48_to(&xv48_frame(&packed), FR, sink.set_kernel_matrix(M)).unwrap();
  }
  let y_binned = block_mean_u16(&full_y);
  assert_eq!(
    luma_u16, y_binned,
    "luma_u16 must be the area-downscaled native Y"
  );
  let luma_ref: Vec<u8> = y_binned.iter().map(|&p| (p >> SHIFT) as u8).collect();
  assert_eq!(
    luma, luma_ref,
    "luma must be the binned native Y narrowed >> 8"
  );
}

// ---- All-outputs parity vs the direct per-output kernel ---------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn xv48_all_outputs_match_their_own_native_depth_block_mean() {
  let (u, y, v) = yuv_ramp();
  let packed = pack_xv48(&u, &y, &v);
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
      MixedSinker::<Xv48, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
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
    xv48_to(&xv48_frame(&packed), FR, sink.set_kernel_matrix(M)).unwrap();
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
  crate::row::expand_rgb_u16_to_rgba_u16_row::<16>(&rgb_u16_ref, &mut rgba_u16_ref, OUT * OUT);
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
fn xv48_luma_taken_from_native_y_under_saturated_chroma() {
  let yc = 32768u16;
  let u = vec![64000u16; SRC * SRC];
  let y = vec![yc; SRC * SRC];
  let v = vec![1280u16; SRC * SRC];
  let packed = pack_xv48(&u, &y, &v);
  let mut luma_u16 = vec![0u16; OUT * OUT];
  {
    let mut sink = force_row_stage(
      MixedSinker::<Xv48, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap(),
    )
    .with_luma_u16(&mut luma_u16)
    .unwrap();
    xv48_to(&xv48_frame(&packed), FR, sink.set_kernel_matrix(M)).unwrap();
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
fn xv48_resample_le_be_parity() {
  let (u, y, v) = yuv_ramp();
  let host = pack_xv48(&u, &y, &v);
  let pix_le: Vec<u16> = host.iter().map(|&w| as_le_u16(w)).collect();
  let pix_be: Vec<u16> = host.iter().map(|&w| as_be_u16(w)).collect();

  for simd in [true, false] {
    let mut le_rgb = vec![0u8; OUT * OUT * 3];
    let mut le_rgb_u16 = vec![0u16; OUT * OUT * 3];
    let mut le_luma_u16 = vec![0u16; OUT * OUT];
    {
      let frame = Xv48LeFrame::try_new(&pix_le, SRC as u32, SRC as u32, (SRC * 4) as u32).unwrap();
      let mut sink = force_row_stage(
        MixedSinker::<Xv48, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
          .unwrap(),
      )
      .with_simd(simd)
      .with_rgb(&mut le_rgb)
      .unwrap()
      .with_rgb_u16(&mut le_rgb_u16)
      .unwrap()
      .with_luma_u16(&mut le_luma_u16)
      .unwrap();
      xv48_to(&frame, FR, sink.set_kernel_matrix(M)).unwrap();
    }

    let mut be_rgb = vec![0u8; OUT * OUT * 3];
    let mut be_rgb_u16 = vec![0u16; OUT * OUT * 3];
    let mut be_luma_u16 = vec![0u16; OUT * OUT];
    {
      let frame = Xv48BeFrame::try_new(&pix_be, SRC as u32, SRC as u32, (SRC * 4) as u32).unwrap();
      let mut sink = force_row_stage(
        MixedSinker::<Xv48<true>, AreaResampler>::with_resampler(
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
      xv48_to_endian(&frame, FR, sink.set_kernel_matrix(M)).unwrap();
    }

    assert_eq!(
      le_rgb, be_rgb,
      "Xv48 LE/BE resample rgb diverge (simd={simd})"
    );
    assert_eq!(
      le_rgb_u16, be_rgb_u16,
      "Xv48 LE/BE resample rgb_u16 diverge (simd={simd})"
    );
    assert_eq!(
      le_luma_u16, be_luma_u16,
      "Xv48 LE/BE resample luma_u16 diverge (simd={simd})"
    );
  }
}

// ---- Identity / no-op / reset -----------------------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn xv48_identity_plan_matches_new_sink() {
  let (u, y, v) = yuv_ramp();
  let packed = pack_xv48(&u, &y, &v);
  let mut direct = vec![0u16; SRC * SRC * 3];
  {
    let mut sink = MixedSinker::<Xv48>::new(SRC, SRC)
      .with_rgb_u16(&mut direct)
      .unwrap();
    xv48_to(&xv48_frame(&packed), FR, sink.set_kernel_matrix(M)).unwrap();
  }
  let mut via_area = vec![0u16; SRC * SRC * 3];
  {
    let mut sink = force_row_stage(
      MixedSinker::<Xv48, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(SRC, SRC))
        .unwrap(),
    )
    .with_rgb_u16(&mut via_area)
    .unwrap();
    xv48_to(&xv48_frame(&packed), FR, sink.set_kernel_matrix(M)).unwrap();
  }
  assert_eq!(direct, via_area, "identity plan must match the direct sink");
}

#[test]
fn xv48_no_outputs_is_a_no_op() {
  let (u, y, v) = yuv_ramp();
  let packed = pack_xv48(&u, &y, &v);
  let mut sink =
    MixedSinker::<Xv48, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap();
  xv48_to(&xv48_frame(&packed), FR, sink.set_kernel_matrix(M)).unwrap();
  assert!(!sink.rgb_stream_allocated());
  assert!(!sink.rgb_stream_u16_allocated());
  assert!(!sink.luma_stream_u16_allocated());
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn xv48_resets_streams_across_frames() {
  let (u, y1, v) = yuv_ramp();
  let mut y2 = y1.clone();
  for p in y2.iter_mut() {
    *p = 65535 - *p;
  }
  let p1 = pack_xv48(&u, &y1, &v);
  let p2 = pack_xv48(&u, &y2, &v);
  let mut luma_u16 = vec![0u16; OUT * OUT];
  let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
  {
    let mut sink = force_row_stage(
      MixedSinker::<Xv48, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap(),
    )
    .with_luma_u16(&mut luma_u16)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16)
    .unwrap();
    xv48_to(&xv48_frame(&p1), FR, sink.set_kernel_matrix(M)).unwrap();
    xv48_to(&xv48_frame(&p2), FR, sink.set_kernel_matrix(M)).unwrap();
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
fn xv48_out_of_sequence_first_row_rejected_before_allocation() {
  let (u, y, v) = yuv_ramp();
  let packed = pack_xv48(&u, &y, &v);
  let stride = SRC * 4;
  let row3 = &packed[3 * stride..4 * stride];
  let mut luma_u16 = vec![0u16; OUT * OUT];
  let mut rgb = vec![0u8; OUT * OUT * 3];
  let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
  let mut sink = force_row_stage(
    MixedSinker::<Xv48, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap(),
  )
  .with_luma_u16(&mut luma_u16)
  .unwrap()
  .with_rgb(&mut rgb)
  .unwrap()
  .with_rgb_u16(&mut rgb_u16)
  .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  let err = sink
    .process(Xv48Row::for_tests(row3, 3, M, FR))
    .unwrap_err();
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
fn xv48_rejects_mid_frame_output_change() {
  let (u, y, v) = yuv_ramp();
  let packed = pack_xv48(&u, &y, &v);
  let stride = SRC * 4;
  let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
  let mut luma_u16 = vec![0u16; OUT * OUT];
  let mut sink = force_row_stage(
    MixedSinker::<Xv48, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap(),
  )
  .with_rgb_u16(&mut rgb_u16)
  .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  sink
    .process(Xv48Row::for_tests(&packed[..stride], 0, M, FR))
    .unwrap();
  sink.set_luma_u16(&mut luma_u16).unwrap();
  let err = sink
    .process(Xv48Row::for_tests(&packed[stride..2 * stride], 1, M, FR))
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
