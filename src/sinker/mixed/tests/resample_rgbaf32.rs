//! Fused-downscale + filter coverage for the packed-float-RGBA family
//! (`Rgbaf32`): the wire row converts to source-width host packed RGBA
//! f32, binning runs in float over all four channels (straight alpha), the
//! `rgba_f32` output is the exact area mean, and every other output mirrors
//! the direct Rgbaf32 path's kernels run over the binned row.

use crate::{
  KernelMatrix, PixelSink,
  frame::Rgbaf32LeFrame,
  resample::{
    AreaResampler, CatmullRom, FilterKernel, FilterStream, FilteredResampler, Lanczos3,
    ResampleError, Resampler, Triangle,
  },
  sinker::{MixedSinker, MixedSinkerError},
  source::{Rgbaf32, Rgbaf32Row, rgbaf32_to},
};

const SRC: usize = 8;
const OUT: usize = 4;

fn as_le_rgbaf32(host: &[f32]) -> Vec<f32> {
  host
    .iter()
    .map(|&v| f32::from_bits(u32::from_ne_bytes(v.to_bits().to_le_bytes())))
    .collect()
}

/// Per-channel integer-valued f32 ramp (so the 2x2 area mean is exact)
/// spanning HDR and negatives — alpha gets its own distinct ramp so an
/// alpha mix-up diverges immediately.
fn packed_frame_rgbaf32() -> Vec<f32> {
  let mut buf = vec![0.0f32; SRC * SRC * 4];
  for (i, px) in buf.chunks_exact_mut(4).enumerate() {
    let i = i as i32;
    px[0] = (i % 5) as f32;
    px[1] = (100 - i) as f32;
    px[2] = -((i % 7) as f32);
    px[3] = (i % 4) as f32; // alpha: 0,1,2,3 — real per-pixel
  }
  buf
}

fn block_mean(src: &[f32], ox: usize, oy: usize, c: usize) -> f32 {
  let mut acc = 0.0f64;
  for dy in 0..2 {
    for dx in 0..2 {
      acc += src[((oy * 2 + dy) * SRC + ox * 2 + dx) * 4 + c] as f64;
    }
  }
  (acc / 4.0) as f32
}

#[test]
fn rgbaf32_downscale_rgba_f32_is_exact_area_mean() {
  let host = packed_frame_rgbaf32();
  let wire = as_le_rgbaf32(&host);
  let src = Rgbaf32LeFrame::try_new(&wire, SRC as u32, SRC as u32, (SRC * 4) as u32).unwrap();

  let mut rgba_f32 = vec![0.0f32; OUT * OUT * 4];
  {
    let mut sink =
      MixedSinker::<Rgbaf32, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgba_f32(&mut rgba_f32)
        .unwrap();
    rgbaf32_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  }
  for oy in 0..OUT {
    for ox in 0..OUT {
      for c in 0..4 {
        let got = rgba_f32[(oy * OUT + ox) * 4 + c];
        let want = block_mean(&host, ox, oy, c);
        assert_eq!(got, want, "({ox},{oy}) c{c}: {got} != {want}");
      }
    }
  }
}

#[test]
fn rgbaf32_derived_outputs_come_from_binned_rgba() {
  // Every attached output must be exactly what the direct full-res Rgbaf32
  // sink produces over a frame that already holds the binned f32 RGBA.
  let host = packed_frame_rgbaf32();
  let wire = as_le_rgbaf32(&host);
  let src = Rgbaf32LeFrame::try_new(&wire, SRC as u32, SRC as u32, (SRC * 4) as u32).unwrap();

  let mut rgb = vec![0u8; OUT * OUT * 3];
  let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
  let mut rgb_f32 = vec![0.0f32; OUT * OUT * 3];
  let mut rgba = vec![0u8; OUT * OUT * 4];
  let mut rgba_u16 = vec![0u16; OUT * OUT * 4];
  let mut rgba_f32 = vec![0.0f32; OUT * OUT * 4];
  let mut luma = vec![0u8; OUT * OUT];
  let mut luma_u16 = vec![0u16; OUT * OUT];
  let mut h = vec![0u8; OUT * OUT];
  let mut s_ = vec![0u8; OUT * OUT];
  let mut v_ = vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Rgbaf32, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb(&mut rgb)
        .unwrap()
        .with_rgb_u16(&mut rgb_u16)
        .unwrap()
        .with_rgb_f32(&mut rgb_f32)
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap()
        .with_rgba_u16(&mut rgba_u16)
        .unwrap()
        .with_rgba_f32(&mut rgba_f32)
        .unwrap()
        .with_luma(&mut luma)
        .unwrap()
        .with_luma_u16(&mut luma_u16)
        .unwrap()
        .with_hsv(&mut h, &mut s_, &mut v_)
        .unwrap();
    rgbaf32_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  }

  // Reference: the full-res sink over the (exact) binned f32 RGBA.
  let binned_wire = as_le_rgbaf32(&rgba_f32);
  let mut ref_rgb = vec![0u8; OUT * OUT * 3];
  let mut ref_rgb_u16 = vec![0u16; OUT * OUT * 3];
  let mut ref_rgb_f32 = vec![0.0f32; OUT * OUT * 3];
  let mut ref_rgba = vec![0u8; OUT * OUT * 4];
  let mut ref_rgba_u16 = vec![0u16; OUT * OUT * 4];
  let mut ref_luma = vec![0u8; OUT * OUT];
  let mut ref_luma_u16 = vec![0u16; OUT * OUT];
  let mut ref_h = vec![0u8; OUT * OUT];
  let mut ref_s = vec![0u8; OUT * OUT];
  let mut ref_v = vec![0u8; OUT * OUT];
  {
    let binned =
      Rgbaf32LeFrame::try_new(&binned_wire, OUT as u32, OUT as u32, (OUT * 4) as u32).unwrap();
    let mut sink = MixedSinker::<Rgbaf32>::new(OUT, OUT)
      .with_rgb(&mut ref_rgb)
      .unwrap()
      .with_rgb_u16(&mut ref_rgb_u16)
      .unwrap()
      .with_rgb_f32(&mut ref_rgb_f32)
      .unwrap()
      .with_rgba(&mut ref_rgba)
      .unwrap()
      .with_rgba_u16(&mut ref_rgba_u16)
      .unwrap()
      .with_luma(&mut ref_luma)
      .unwrap()
      .with_luma_u16(&mut ref_luma_u16)
      .unwrap()
      .with_hsv(&mut ref_h, &mut ref_s, &mut ref_v)
      .unwrap();
    rgbaf32_to(&binned, true, KernelMatrix::Bt709, &mut sink).unwrap();
  }
  assert_eq!(rgb, ref_rgb, "rgb");
  assert_eq!(rgb_u16, ref_rgb_u16, "rgb_u16");
  assert_eq!(rgb_f32, ref_rgb_f32, "rgb_f32");
  assert_eq!(rgba, ref_rgba, "rgba");
  assert_eq!(rgba_u16, ref_rgba_u16, "rgba_u16");
  assert_eq!(luma, ref_luma, "luma");
  assert_eq!(luma_u16, ref_luma_u16, "luma_u16");
  assert_eq!(h, ref_h, "h");
  assert_eq!(s_, ref_s, "s");
  assert_eq!(v_, ref_v, "v");
}

#[test]
fn rgbaf32_identity_plan_matches_new_sink() {
  let host = packed_frame_rgbaf32();
  let wire = as_le_rgbaf32(&host);
  let src = Rgbaf32LeFrame::try_new(&wire, SRC as u32, SRC as u32, (SRC * 4) as u32).unwrap();

  let mut direct = vec![0.0f32; SRC * SRC * 4];
  {
    let mut sink = MixedSinker::<Rgbaf32>::new(SRC, SRC)
      .with_rgba_f32(&mut direct)
      .unwrap();
    rgbaf32_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  }
  let mut via_area = vec![0.0f32; SRC * SRC * 4];
  {
    let mut sink =
      MixedSinker::<Rgbaf32, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(SRC, SRC))
        .unwrap()
        .with_rgba_f32(&mut via_area)
        .unwrap();
    rgbaf32_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  }
  assert_eq!(direct, via_area);
}

// ---- Filter path (PIL F-mode, straight alpha) -------------------------

/// Single-channel filter resample of channel `c` of a packed RGBA f32
/// plane — the per-channel oracle. The 4-channel `Rgbaf32` filter's
/// channel `c` must equal this bit-for-bit (same engine, run per plane).
#[allow(clippy::too_many_arguments)]
fn channel_plane_filter<K: FilterKernel>(
  kernel: K,
  packed: &[f32],
  c: usize,
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
) -> Vec<f32> {
  let mut plane = vec![0.0f32; sw * sh];
  for (dst, px) in plane.iter_mut().zip(packed.chunks_exact(4)) {
    *dst = px[c];
  }
  let plan = FilteredResampler::new(ow, oh, kernel)
    .plan(sw, sh)
    .expect("valid")
    .expect("non-identity");
  let fh = plan.filter_h().expect("h");
  let fv = plan.filter_v().expect("v");
  let mut stream = FilterStream::<f32>::new(fh, fv, sw, sh, 1).expect("geometry");
  let mut out = vec![0.0f32; ow * oh];
  for y in 0..sh {
    stream
      .feed_row(y, &plane[y * sw..(y + 1) * sw], true, |oy, fin| {
        out[oy * ow..(oy + 1) * ow].copy_from_slice(fin);
      })
      .expect("rows in order");
  }
  out
}

fn assert_rgba_f32_is_per_channel_filter<K: FilterKernel + Copy>(
  kernel: K,
  host: &[f32],
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
  ctx: &str,
) {
  let wire = as_le_rgbaf32(host);
  let src = Rgbaf32LeFrame::try_new(&wire, sw as u32, sh as u32, (sw * 4) as u32).unwrap();
  let mut rgba_f32 = vec![0.0f32; ow * oh * 4];
  {
    let mut sink = MixedSinker::<Rgbaf32, FilteredResampler<K>>::with_resampler(
      sw,
      sh,
      FilteredResampler::new(ow, oh, kernel),
    )
    .unwrap()
    .with_rgba_f32(&mut rgba_f32)
    .unwrap();
    rgbaf32_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  }
  for c in 0..4 {
    let plane = channel_plane_filter(kernel, host, c, sw, sh, ow, oh);
    for (i, &want) in plane.iter().enumerate() {
      assert_eq!(
        rgba_f32[i * 4 + c].to_bits(),
        want.to_bits(),
        "{ctx} channel {c} px {i}",
      );
    }
  }
}

fn ramp_frame(w: usize, h: usize) -> Vec<f32> {
  let mut buf = vec![0.0f32; w * h * 4];
  for (i, px) in buf.chunks_exact_mut(4).enumerate() {
    let i = i as f32;
    px[0] = 0.1 + i * 0.05;
    px[1] = 2.5 - i * 0.07;
    px[2] = -0.4 + i * 0.03;
    px[3] = 0.2 + i * 0.04; // alpha ramp — a real filtered channel
  }
  buf
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgbaf32_filter_rgba_f32_is_per_channel() {
  let down = ramp_frame(8, 8);
  assert_rgba_f32_is_per_channel_filter(Triangle, &down, 8, 8, 4, 4, "triangle down");
  assert_rgba_f32_is_per_channel_filter(CatmullRom, &down, 8, 8, 4, 4, "catmullrom down");
  assert_rgba_f32_is_per_channel_filter(Lanczos3, &down, 8, 8, 4, 4, "lanczos3 down");
  let up = ramp_frame(4, 4);
  assert_rgba_f32_is_per_channel_filter(Triangle, &up, 4, 4, 7, 7, "triangle up");
  assert_rgba_f32_is_per_channel_filter(CatmullRom, &up, 4, 4, 7, 7, "catmullrom up");
}

#[test]
fn rgbaf32_no_output_sink_is_a_noop() {
  let host = packed_frame_rgbaf32();
  let wire = as_le_rgbaf32(&host);
  let src = Rgbaf32LeFrame::try_new(&wire, SRC as u32, SRC as u32, (SRC * 4) as u32).unwrap();
  let mut sink =
    MixedSinker::<Rgbaf32, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap();
  rgbaf32_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  assert!(
    !sink.rgba_stream_f32_allocated(),
    "no-output sink allocated the stream"
  );
}

#[test]
fn rgbaf32_out_of_sequence_first_row_rejected_before_allocation() {
  let host = packed_frame_rgbaf32();
  let wire = as_le_rgbaf32(&host);
  let row3 = &wire[3 * SRC * 4..4 * SRC * 4];

  let mut rgba_f32 = vec![0.0f32; OUT * OUT * 4];
  let mut sink =
    MixedSinker::<Rgbaf32, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_rgba_f32(&mut rgba_f32)
      .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  let err = sink
    .process(Rgbaf32Row::new(row3, 3, KernelMatrix::Bt709, true))
    .unwrap_err();
  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
    ),
    "expected OutOfSequenceRow, got {err:?}"
  );
  assert!(
    !sink.rgba_stream_f32_allocated(),
    "stream allocated for a rejected row"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgbaf32_first_row_scratch_oom_leaves_stream_and_freeze_uncommitted_for_retry() {
  let host = packed_frame_rgbaf32();
  let wire = as_le_rgbaf32(&host);
  let mut rgba_u16 = vec![0u16; OUT * OUT * 4];
  let mut luma = vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Rgbaf32, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgba_u16(&mut rgba_u16)
        .unwrap();
    sink.begin_frame(SRC as u32, SRC as u32).unwrap();
    // Arm the single-shot 4-channel `f32` SCRATCH grow failpoint: the row-0 area
    // stream box builds OK (into a local), then the `source_rgba_f32_scratch` grow —
    // which runs AFTER the stream build and BEFORE the field insert — refuses,
    // surfacing AllocationFailed with the rgba_stream_f32 field `None` and the freeze
    // uncommitted (the commit-together atomic shape). This exercises the exact
    // scratch-after-stream ordering boundary a box failure cannot pin.
    crate::sinker::mixed::arm_source_rgba_f32_scratch_failure();
    let err = sink
      .process(Rgbaf32Row::new(
        &wire[..SRC * 4],
        0,
        KernelMatrix::Bt709,
        true,
      ))
      .unwrap_err();
    assert!(
      matches!(
        err,
        MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
      ),
      "first-row scratch OOM must surface AllocationFailed, got {err:?}"
    );
    // DIRECT stream-rollback proof (the non-vacuous part): the 4-channel `f32` area
    // stream field must be `None` — built into a LOCAL and inserted only after the
    // scratch grows, so a failed scratch leaves nothing committed. The
    // insert-before-scratch shape would leave it `Some` here, so this assertion FAILS
    // against that shape and PASSES against the fix (a `try_box` box failure cannot
    // make this distinction).
    assert!(
      !sink.rgba_stream_f32_allocated(),
      "a scratch OOM must leave the rgba_stream_f32 field None (no partial commit)"
    );
    // The freeze was likewise never committed — attaching luma (a CHANGED output
    // set) and replaying from row 0 is ACCEPTED, not ResampleOutputsChanged.
    sink.set_luma(&mut luma).unwrap();
    for r in 0..SRC {
      sink
        .process(Rgbaf32Row::new(
          &wire[r * SRC * 4..(r + 1) * SRC * 4],
          r,
          KernelMatrix::Bt709,
          true,
        ))
        .expect("frame replay after a first-row scratch OOM must succeed");
    }
  }
  assert!(
    rgba_u16.iter().any(|&b| b != 0),
    "the recovered frame produced real rgba_u16 output (scratch OOM was recoverable)"
  );
  assert!(
    luma.iter().any(|&b| b != 0),
    "the newly-attached luma output was driven on the accepted retry"
  );
}

#[test]
fn rgbaf32_mid_frame_out_of_sequence_rejected() {
  let host = packed_frame_rgbaf32();
  let wire = as_le_rgbaf32(&host);
  let mut rgba_f32 = vec![0.0f32; OUT * OUT * 4];
  let mut sink =
    MixedSinker::<Rgbaf32, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_rgba_f32(&mut rgba_f32)
      .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  sink
    .process(Rgbaf32Row::new(
      &wire[..SRC * 4],
      0,
      KernelMatrix::Bt709,
      true,
    ))
    .unwrap();
  let err = sink
    .process(Rgbaf32Row::new(
      &wire[2 * SRC * 4..3 * SRC * 4],
      2,
      KernelMatrix::Bt709,
      true,
    ))
    .unwrap_err();
  assert!(matches!(
    err,
    MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
  ));
}
