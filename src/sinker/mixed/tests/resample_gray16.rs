//! Fused-downscale coverage for `Gray16` — routed through a single
//! 1-channel `AreaStream<u16>` that bins the source `u16` luma plane at
//! u16 precision (Gray16 *is* a u16 luma plane). The wire row converts
//! to a host-native u16 luma plane first (the same kernel the direct
//! `luma_u16` path uses), then every attached output derives from each
//! finalized binned u16 luma row exactly as the direct path does:
//! `luma_u16` is a host-native pass-through, `luma` is `>> 8`, `rgb` /
//! `rgba` broadcast the `>> 8` byte (α = 0xFF), `rgb_u16` / `rgba_u16`
//! broadcast the native u16 (α = 0xFFFF), and `hsv` is `H=0 / S=0 /
//! V=Y>>8`. So every resampled output equals the direct Gray16 sink run
//! over a frame that already holds the binned u16 luma plane.
//!
//! The binning is at full u16 precision: a u8 luma stream would bin the
//! high byte only, losing the low 8 bits of every sample.

use crate::{
  KernelMatrix, PixelSink,
  frame::Gray16Frame,
  resample::{AreaResampler, ResampleError},
  sinker::{MixedSinker, MixedSinkerError},
  source::{Gray16, Gray16Row, gray16_to, gray16_to_endian},
};

const SRC: usize = 8;
const OUT: usize = 4;
// Gray is luma-only; the walker still threads a matrix / range through.
const FR: bool = true;
const M: KernelMatrix = KernelMatrix::Bt709;

/// Re-encode a host-native u16 slice as LE-encoded byte storage (the
/// `gray16le` plane contract). `gray16_to` recovers the logical values
/// via `u16::from_le` — a no-op on LE hosts, a byte-swap on BE.
fn as_le_u16(host: &[u16]) -> Vec<u16> {
  host
    .iter()
    .map(|v| u16::from_ne_bytes(v.to_le_bytes()))
    .collect()
}

/// Re-encode a host-native u16 slice as BE-encoded byte storage (the
/// `gray16be` plane contract), recovered via `u16::from_be`.
fn as_be_u16(host: &[u16]) -> Vec<u16> {
  host
    .iter()
    .map(|v| u16::from_ne_bytes(v.to_be_bytes()))
    .collect()
}

/// Interior 16-bit luma ramp so the area mean sees real variation per
/// 2x2 block, exercising the low byte the u16 stream must preserve.
fn ramp() -> Vec<u16> {
  let mut y = vec![0u16; SRC * SRC];
  for (i, p) in y.iter_mut().enumerate() {
    // Spread across the full 16-bit range with a non-trivial low byte.
    *p = ((i * 1031) % 65536) as u16;
  }
  y
}

/// Exact 2x2-block area mean (round-half-up) of an `SRC`-grid `u16`
/// plane to the `OUT` grid — the integer-ratio (2:1) area-downscale
/// reference, computed at u16 precision (sum of four u16 fits in u32).
fn block_mean_2x2(plane: &[u16]) -> Vec<u16> {
  let mut out = vec![0u16; OUT * OUT];
  for oy in 0..OUT {
    for ox in 0..OUT {
      let mut s = 0u32;
      for dy in 0..2 {
        for dx in 0..2 {
          s += plane[(oy * 2 + dy) * SRC + ox * 2 + dx] as u32;
        }
      }
      out[oy * OUT + ox] = ((s + 2) / 4) as u16;
    }
  }
  out
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gray16_downscale_luma_u16_is_exact_area_mean() {
  let plane = ramp();
  let pix = as_le_u16(&plane);
  let src = Gray16Frame::new(&pix, SRC as u32, SRC as u32, SRC as u32);

  let mut luma_u16 = vec![0u16; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Gray16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_luma_u16(&mut luma_u16)
        .unwrap();
    gray16_to(&src, FR, M, &mut sink).unwrap();
  }
  assert_eq!(
    luma_u16,
    block_mean_2x2(&plane),
    "luma_u16 must be the exact 2x2 block mean of the u16 luma plane"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gray16_all_outputs_match_direct_over_binned_luma() {
  // Every attached output — luma / luma_u16 / rgb / rgba / rgb_u16 /
  // rgba_u16 / hsv — must be exactly what the direct Gray16 sink
  // produces over the (exact) binned u16 luma plane. The binned luma is
  // the area mean, so we feed that mean as a full-resolution `OUT`-grid
  // Gray16 frame to the reference sink.
  let plane = ramp();
  let pix = as_le_u16(&plane);
  let src = Gray16Frame::new(&pix, SRC as u32, SRC as u32, SRC as u32);

  let mut rgb = vec![0u8; OUT * OUT * 3];
  let mut rgba = vec![0u8; OUT * OUT * 4];
  let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
  let mut rgba_u16 = vec![0u16; OUT * OUT * 4];
  let mut luma = vec![0u8; OUT * OUT];
  let mut luma_u16 = vec![0u16; OUT * OUT];
  let mut h = vec![0u8; OUT * OUT];
  let mut s_ = vec![0u8; OUT * OUT];
  let mut v_ = vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Gray16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
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
        .with_hsv(&mut h, &mut s_, &mut v_)
        .unwrap();
    gray16_to(&src, FR, M, &mut sink).unwrap();
  }

  // Reference: the direct sink over the exact binned u16 luma plane.
  let binned = block_mean_2x2(&plane);
  let binned_pix = as_le_u16(&binned);
  let mut ref_rgb = vec![0u8; OUT * OUT * 3];
  let mut ref_rgba = vec![0u8; OUT * OUT * 4];
  let mut ref_rgb_u16 = vec![0u16; OUT * OUT * 3];
  let mut ref_rgba_u16 = vec![0u16; OUT * OUT * 4];
  let mut ref_luma = vec![0u8; OUT * OUT];
  let mut ref_luma_u16 = vec![0u16; OUT * OUT];
  let mut ref_h = vec![0u8; OUT * OUT];
  let mut ref_s = vec![0u8; OUT * OUT];
  let mut ref_v = vec![0u8; OUT * OUT];
  {
    let binned_frame = Gray16Frame::new(&binned_pix, OUT as u32, OUT as u32, OUT as u32);
    let mut sink = MixedSinker::<Gray16>::new(OUT, OUT)
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
      .with_hsv(&mut ref_h, &mut ref_s, &mut ref_v)
      .unwrap();
    gray16_to(&binned_frame, FR, M, &mut sink).unwrap();
  }
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
fn gray16_le_be_resample_outputs_identical() {
  // The binned luma is host-native, so the derive kernels must run with
  // `HOST_NATIVE_BE`, not `<false>`. On an LE dev/CI host `<false>`
  // masks the bug; the LE-vs-BE parity check catches a wrong const on
  // either host (LE and BE wire encodings of the same logical plane must
  // resample to identical outputs).
  let plane = ramp();
  let pix_le = as_le_u16(&plane);
  let pix_be = as_be_u16(&plane);

  let mut le_luma_u16 = vec![0u16; OUT * OUT];
  let mut le_rgb_u16 = vec![0u16; OUT * OUT * 3];
  let mut le_rgba = vec![0u8; OUT * OUT * 4];
  {
    let frame = Gray16Frame::new(&pix_le, SRC as u32, SRC as u32, SRC as u32);
    let mut sink =
      MixedSinker::<Gray16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_luma_u16(&mut le_luma_u16)
        .unwrap()
        .with_rgb_u16(&mut le_rgb_u16)
        .unwrap()
        .with_rgba(&mut le_rgba)
        .unwrap();
    gray16_to(&frame, FR, M, &mut sink).unwrap();
  }

  let mut be_luma_u16 = vec![0u16; OUT * OUT];
  let mut be_rgb_u16 = vec![0u16; OUT * OUT * 3];
  let mut be_rgba = vec![0u8; OUT * OUT * 4];
  {
    let frame = Gray16Frame::<true>::new(&pix_be, SRC as u32, SRC as u32, SRC as u32);
    let mut sink = MixedSinker::<Gray16<true>, AreaResampler>::with_resampler(
      SRC,
      SRC,
      AreaResampler::to(OUT, OUT),
    )
    .unwrap()
    .with_luma_u16(&mut be_luma_u16)
    .unwrap()
    .with_rgb_u16(&mut be_rgb_u16)
    .unwrap()
    .with_rgba(&mut be_rgba)
    .unwrap();
    gray16_to_endian::<_, true>(&frame, FR, M, &mut sink).unwrap();
  }

  assert_eq!(le_luma_u16, be_luma_u16, "luma_u16 LE/BE diverge");
  assert_eq!(le_rgb_u16, be_rgb_u16, "rgb_u16 LE/BE diverge");
  assert_eq!(le_rgba, be_rgba, "rgba LE/BE diverge");
  // And the binned u16 luma is the exact area mean regardless of wire.
  assert_eq!(
    le_luma_u16,
    block_mean_2x2(&plane),
    "luma_u16 not area mean"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gray16_standalone_rgba_matches_direct_over_binned_luma() {
  // u8 RGBA-only exercises the dedicated fast path (no RGB scratch).
  let plane = ramp();
  let pix = as_le_u16(&plane);
  let src = Gray16Frame::new(&pix, SRC as u32, SRC as u32, SRC as u32);

  let mut rgba = vec![0u8; OUT * OUT * 4];
  {
    let mut sink =
      MixedSinker::<Gray16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap();
    gray16_to(&src, FR, M, &mut sink).unwrap();
  }
  let binned_pix = as_le_u16(&block_mean_2x2(&plane));
  let mut ref_rgba = vec![0u8; OUT * OUT * 4];
  {
    let binned = Gray16Frame::new(&binned_pix, OUT as u32, OUT as u32, OUT as u32);
    let mut sink = MixedSinker::<Gray16>::new(OUT, OUT)
      .with_rgba(&mut ref_rgba)
      .unwrap();
    gray16_to(&binned, FR, M, &mut sink).unwrap();
  }
  assert_eq!(rgba, ref_rgba, "standalone rgba");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gray16_standalone_rgba_u16_matches_direct_over_binned_luma() {
  // u16 RGBA-only exercises the native rgba_u16 fast path.
  let plane = ramp();
  let pix = as_le_u16(&plane);
  let src = Gray16Frame::new(&pix, SRC as u32, SRC as u32, SRC as u32);

  let mut rgba_u16 = vec![0u16; OUT * OUT * 4];
  {
    let mut sink =
      MixedSinker::<Gray16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgba_u16(&mut rgba_u16)
        .unwrap();
    gray16_to(&src, FR, M, &mut sink).unwrap();
  }
  let binned_pix = as_le_u16(&block_mean_2x2(&plane));
  let mut ref_rgba_u16 = vec![0u16; OUT * OUT * 4];
  {
    let binned = Gray16Frame::new(&binned_pix, OUT as u32, OUT as u32, OUT as u32);
    let mut sink = MixedSinker::<Gray16>::new(OUT, OUT)
      .with_rgba_u16(&mut ref_rgba_u16)
      .unwrap();
    gray16_to(&binned, FR, M, &mut sink).unwrap();
  }
  assert_eq!(rgba_u16, ref_rgba_u16, "standalone rgba_u16");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gray16_hsv_plus_rgba_matches_direct_over_binned_luma() {
  // HSV + u8 RGBA without RGB: both derive directly from the binned luma
  // (no RGB kernel, no RGB scratch — asserted below), byte-identical to
  // the direct path's output.
  let plane = ramp();
  let pix = as_le_u16(&plane);
  let src = Gray16Frame::new(&pix, SRC as u32, SRC as u32, SRC as u32);

  let mut rgba = vec![0u8; OUT * OUT * 4];
  let mut h = vec![0u8; OUT * OUT];
  let mut s_ = vec![0u8; OUT * OUT];
  let mut v_ = vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Gray16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap()
        .with_hsv(&mut h, &mut s_, &mut v_)
        .unwrap();
    gray16_to(&src, FR, M, &mut sink).unwrap();
    // Regression: this case must not reserve RGB scratch — it derives
    // HSV+RGBA from luma and never reads the scratch, so reserving it
    // could spuriously AllocationFail under memory pressure.
    assert_eq!(
      sink.rgb_scratch_capacity(),
      0,
      "gray16 HSV+RGBA must not reserve RGB scratch"
    );
  }
  let binned_pix = as_le_u16(&block_mean_2x2(&plane));
  let mut ref_rgba = vec![0u8; OUT * OUT * 4];
  let mut ref_h = vec![0u8; OUT * OUT];
  let mut ref_s = vec![0u8; OUT * OUT];
  let mut ref_v = vec![0u8; OUT * OUT];
  {
    let binned = Gray16Frame::new(&binned_pix, OUT as u32, OUT as u32, OUT as u32);
    let mut sink = MixedSinker::<Gray16>::new(OUT, OUT)
      .with_rgba(&mut ref_rgba)
      .unwrap()
      .with_hsv(&mut ref_h, &mut ref_s, &mut ref_v)
      .unwrap();
    gray16_to(&binned, FR, M, &mut sink).unwrap();
  }
  assert_eq!(rgba, ref_rgba, "hsv+rgba: rgba");
  assert_eq!(h, ref_h, "hsv+rgba: h");
  assert_eq!(s_, ref_s, "hsv+rgba: s");
  assert_eq!(v_, ref_v, "hsv+rgba: v");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gray16_identity_plan_matches_new_sink() {
  let plane = ramp();
  let pix = as_le_u16(&plane);
  let src = Gray16Frame::new(&pix, SRC as u32, SRC as u32, SRC as u32);

  let mut direct = vec![0u16; SRC * SRC * 3];
  {
    let mut sink = MixedSinker::<Gray16>::new(SRC, SRC)
      .with_rgb_u16(&mut direct)
      .unwrap();
    gray16_to(&src, FR, M, &mut sink).unwrap();
  }
  let mut via_area = vec![0u16; SRC * SRC * 3];
  {
    let mut sink =
      MixedSinker::<Gray16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(SRC, SRC))
        .unwrap()
        .with_rgb_u16(&mut via_area)
        .unwrap();
    gray16_to(&src, FR, M, &mut sink).unwrap();
  }
  assert_eq!(direct, via_area, "identity plan must match the direct sink");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gray16_resample_reuses_luma_stream_across_frames() {
  // A reused sink must reset the u16 luma stream each frame; without the
  // reset, frame 2's row 0 is rejected as out-of-sequence.
  let y1 = ramp();
  let mut y2 = y1.clone();
  for p in y2.iter_mut() {
    *p = 0xFFFF - *p;
  }
  let pix1 = as_le_u16(&y1);
  let pix2 = as_le_u16(&y2);
  let mut luma_u16 = vec![0u16; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Gray16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_luma_u16(&mut luma_u16)
        .unwrap();
    gray16_to(
      &Gray16Frame::new(&pix1, SRC as u32, SRC as u32, SRC as u32),
      FR,
      M,
      &mut sink,
    )
    .unwrap();
    gray16_to(
      &Gray16Frame::new(&pix2, SRC as u32, SRC as u32, SRC as u32),
      FR,
      M,
      &mut sink,
    )
    .unwrap();
  }
  assert_eq!(
    luma_u16,
    block_mean_2x2(&y2),
    "frame 2 luma_u16 must area-downscale frame 2's luma"
  );
}

#[test]
fn gray16_resample_no_outputs_is_a_no_op() {
  let plane = ramp();
  let pix = as_le_u16(&plane);
  let src = Gray16Frame::new(&pix, SRC as u32, SRC as u32, SRC as u32);
  let mut sink =
    MixedSinker::<Gray16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap();
  // No outputs attached: a legal no-op, accepted without error.
  gray16_to(&src, FR, M, &mut sink).unwrap();
  // A no-output call has no stream to sequence and never allocates.
  assert!(
    !sink.luma_stream_u16_allocated(),
    "no-output sink allocated a u16 luma stream"
  );
}

#[test]
fn gray16_out_of_sequence_first_row_rejected_before_allocation() {
  let plane = ramp();
  let pix = as_le_u16(&plane);
  let row3 = &pix[3 * SRC..4 * SRC];

  let mut luma_u16 = vec![0u16; OUT * OUT];
  let mut sink =
    MixedSinker::<Gray16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_luma_u16(&mut luma_u16)
      .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  // Feed row 3 first — the stream expects strict sequencing from 0.
  let err = sink.process(Gray16Row::new(row3, 3, M, FR)).unwrap_err();
  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
    ),
    "expected OutOfSequenceRow, got {err:?}"
  );
  // The out-of-sequence first row must be rejected before the u16 luma
  // stream is allocated.
  assert!(
    !sink.luma_stream_u16_allocated(),
    "stream allocated for a rejected row"
  );
  assert!(
    luma_u16.iter().all(|&b| b == 0),
    "rejected row mutated output"
  );
}

#[test]
fn gray16_rejected_first_row_does_not_poison_output_retry() {
  // A rejected out-of-sequence FIRST row must store no frozen-output
  // snapshot, so retrying row 0 after reconfiguring the output set
  // succeeds instead of tripping ResampleOutputsChanged.
  let plane = ramp();
  let pix = as_le_u16(&plane);
  let mut luma_u16 = vec![0u16; OUT * OUT];
  let mut sink =
    MixedSinker::<Gray16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_luma_u16(&mut luma_u16)
      .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  let err = sink
    .process(Gray16Row::new(&pix[3 * SRC..4 * SRC], 3, M, FR))
    .unwrap_err();
  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
    ),
    "expected OutOfSequenceRow, got {err:?}"
  );
  let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
  sink.set_rgb_u16(&mut rgb_u16).unwrap();
  sink
    .process(Gray16Row::new(&pix[..SRC], 0, M, FR))
    .expect("row 0 must succeed after a rejected out-of-sequence first row");
}

#[test]
fn gray16_resample_rejects_mid_frame_out_of_sequence() {
  let plane = ramp();
  let pix = as_le_u16(&plane);
  let mut luma_u16 = vec![0u16; OUT * OUT];
  let mut sink =
    MixedSinker::<Gray16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_luma_u16(&mut luma_u16)
      .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  sink.process(Gray16Row::new(&pix[..SRC], 0, M, FR)).unwrap();
  // Skip row 1 — feeding row 2 next is out of sequence.
  let err = sink
    .process(Gray16Row::new(&pix[2 * SRC..3 * SRC], 2, M, FR))
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
fn gray16_resample_rejects_mid_frame_output_change() {
  let plane = ramp();
  let pix = as_le_u16(&plane);
  let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
  let mut luma_u16 = vec![0u16; OUT * OUT];
  let mut sink =
    MixedSinker::<Gray16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_rgb_u16(&mut rgb_u16)
      .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  sink.process(Gray16Row::new(&pix[..SRC], 0, M, FR)).unwrap();
  // Attaching a new output mid-frame trips the frozen-output check.
  sink.set_luma_u16(&mut luma_u16).unwrap();
  let err = sink
    .process(Gray16Row::new(&pix[SRC..2 * SRC], 1, M, FR))
    .unwrap_err();
  assert!(
    matches!(err, MixedSinkerError::ResampleOutputsChanged(_)),
    "expected ResampleOutputsChanged, got {err:?}"
  );
  assert!(
    luma_u16.iter().all(|&b| b == 0),
    "rejected row mutated the new output"
  );
}

// The source-width host-native u16 luma staging scratch — grown for every attached
// output (Gray16 always stages the source luma before binning), pregrown BEFORE the
// freeze via the shared driver's `None`-stream pass.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gray16_first_row_scratch_oom_leaves_stream_and_freeze_uncommitted_for_retry() {
  let plane = ramp();
  let pix = as_le_u16(&plane);
  let mut luma_u16 = vec![0u16; OUT * OUT];
  let mut rgb = vec![0u8; OUT * OUT * 3];
  {
    let mut sink =
      MixedSinker::<Gray16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_luma_u16(&mut luma_u16)
        .unwrap();
    sink.begin_frame(SRC as u32, SRC as u32).unwrap();
    // Arm the source-luma-u16 scratch grow failpoint: the row-0 stream box builds OK
    // (into a local) and the source-luma staging grow — which the emit pregrows BEFORE
    // the field insert + freeze — refuses, surfacing AllocationFailed with BOTH the
    // stream field `None` and the freeze uncommitted (the commit-together atomic shape;
    // the insert-before-scratch shape would have left the stream committed).
    crate::sinker::mixed::arm_source_luma_u16_scratch_failure();
    let err = sink
      .process(Gray16Row::new(&pix[..SRC], 0, M, FR))
      .unwrap_err();
    assert!(
      matches!(
        err,
        MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
      ),
      "first-row scratch OOM must surface AllocationFailed, got {err:?}"
    );
    // DIRECT stream-rollback proof: the u16 luma stream field must be `None` — the
    // driver builds it into a LOCAL and inserts only after the scratch grows, so a
    // failed scratch leaves nothing committed. The insert-before-scratch shape would
    // leave it `Some` here, so this FAILS against that shape and PASSES against the fix.
    assert!(
      !sink.luma_stream_u16_allocated(),
      "a scratch OOM must leave the luma_stream_u16 field None (no partial commit)"
    );
    // The freeze was likewise never committed — attaching rgb (a CHANGED output set)
    // and replaying from row 0 is ACCEPTED, not ResampleOutputsChanged.
    sink.set_rgb(&mut rgb).unwrap();
    for r in 0..SRC {
      sink
        .process(Gray16Row::new(&pix[r * SRC..(r + 1) * SRC], r, M, FR))
        .expect("frame replay after a first-row scratch OOM must succeed");
    }
  }
  assert!(
    luma_u16.iter().any(|&b| b != 0),
    "the recovered frame produced real luma_u16 output (scratch OOM was recoverable)"
  );
  assert!(
    rgb.iter().any(|&b| b != 0),
    "the newly-attached rgb output was driven on the accepted retry"
  );
}

// ---- Filter-plan routing ----------------------------------------------------
//
// Gray16 *is* a u16 luma plane: the area path converts the wire row to a
// host-native u16 luma plane and bins it through a 1-channel `AreaStream<u16>`;
// the filter path converts identically and resamples that plane through the
// signed-coefficient single-channel `FilterStream<u16>` (the filter twin of the
// bin). Gray16 is full 16-bit (native max == u16 max), so the `FilterStream`'s
// `0..=65535` clamp *is* the native clamp — no extra clamp. So the filter
// `luma_u16` must equal a single-channel `FilterStream<u16>` resample of the
// host-native source Y plane **value for value**, and every derived output
// (luma >> 8 / rgb / rgba / rgb_u16 / rgba_u16 / hsv) follows from that
// resampled luma exactly as the area path derives from its binned luma.

use crate::resample::{
  CatmullRom, FilterKernel, FilterStream, FilteredResampler, Lanczos3, Resampler, Triangle,
};

/// A larger, full-u16-wide grid than the 2:1 area fixture so a downscale
/// (`FW`->`FOUT_DOWN`) and an upscale (`FOUT_DOWN`->`FUP`) both run real,
/// non-trivial windows.
const FW: usize = 8;
const FH: usize = 8;
const FOUT_DOWN: usize = 4;
const FUP: usize = 7;

/// A host-native u16 Y ramp with a hard mid-column edge per row (plus two
/// textured interior rows) so filter windows straddling the edge produce real
/// intermediate values, exercising the low byte the u16 stream must preserve.
fn filter_ramp() -> Vec<u16> {
  let mut y = vec![0u16; FW * FH];
  for row in 0..FH {
    for col in 0..FW {
      y[row * FW + col] = if col < FW / 2 { 0x1234 } else { 0xC0DE };
    }
  }
  for col in 0..FW {
    y[4 * FW + col] = (col * 8000) as u16;
    y[5 * FW + col] = (60000 - col * 8000) as u16;
  }
  y
}

/// Single-channel filter resample of a host-native u16 luma plane via the
/// merged engine's [`FilterStream<u16>`] (channels = 1) — the Gray16 luma
/// oracle. Full 16-bit, so the engine's `0..=65535` clamp is the native clamp.
fn native_luma_filter<K: FilterKernel>(
  kernel: K,
  luma_plane: &[u16],
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
) -> Vec<u16> {
  let plan = FilteredResampler::new(ow, oh, kernel)
    .plan(sw, sh)
    .expect("valid filter plan")
    .expect("non-identity");
  let fh = plan.filter_h().expect("h windows");
  let fv = plan.filter_v().expect("v windows");
  let mut stream = FilterStream::<u16>::new(fh, fv, sw, sh, 1).expect("geometry");
  let mut out = vec![0u16; ow * oh];
  for row in 0..sh {
    stream
      .feed_row(
        row,
        &luma_plane[row * sw..(row + 1) * sw],
        true,
        |oy, fin| {
          out[oy * ow..(oy + 1) * ow].copy_from_slice(fin);
        },
      )
      .expect("rows in order");
  }
  out
}

/// Every resampled output a Gray16 filter equivalence asserts on.
struct Gray16FilterOutputs {
  luma: Vec<u8>,
  luma_u16: Vec<u16>,
  rgb: Vec<u8>,
  rgba: Vec<u8>,
  rgb_u16: Vec<u16>,
  rgba_u16: Vec<u16>,
  hp: Vec<u8>,
  sp: Vec<u8>,
  vp: Vec<u8>,
}

/// Run a `Gray16` filter sink over the LE-encoded `filter_ramp()` at `ow x oh`
/// under `kernel`, attaching every output the equivalence asserts on.
fn gray16_filter_outputs<K: FilterKernel + Copy>(
  ow: usize,
  oh: usize,
  kernel: K,
) -> Gray16FilterOutputs {
  let plane = filter_ramp();
  let pix = as_le_u16(&plane);
  let src = Gray16Frame::new(&pix, FW as u32, FH as u32, FW as u32);
  let mut o = Gray16FilterOutputs {
    luma: vec![0u8; ow * oh],
    luma_u16: vec![0u16; ow * oh],
    rgb: vec![0u8; ow * oh * 3],
    rgba: vec![0u8; ow * oh * 4],
    rgb_u16: vec![0u16; ow * oh * 3],
    rgba_u16: vec![0u16; ow * oh * 4],
    hp: vec![0u8; ow * oh],
    sp: vec![0u8; ow * oh],
    vp: vec![0u8; ow * oh],
  };
  {
    let mut sink = MixedSinker::<Gray16, FilteredResampler<K>>::with_resampler(
      FW,
      FH,
      FilteredResampler::new(ow, oh, kernel),
    )
    .unwrap()
    .with_luma(&mut o.luma)
    .unwrap()
    .with_luma_u16(&mut o.luma_u16)
    .unwrap()
    .with_rgb(&mut o.rgb)
    .unwrap()
    .with_rgba(&mut o.rgba)
    .unwrap()
    .with_rgb_u16(&mut o.rgb_u16)
    .unwrap()
    .with_rgba_u16(&mut o.rgba_u16)
    .unwrap()
    .with_hsv(&mut o.hp, &mut o.sp, &mut o.vp)
    .unwrap();
    gray16_to(&src, FR, M, &mut sink).unwrap();
  }
  o
}

/// Asserts a `Gray16` filter resample's every output is derived from the
/// single-channel native-luma oracle exactly as the area emit derives from its
/// binned luma, and returns the max per-sample `luma_u16` diff (exactly 0 —
/// same engine, full 16-bit so the engine clamp is the native clamp).
fn assert_gray16_filter_matches_oracle<K: FilterKernel + Copy>(
  kernel: K,
  ow: usize,
  oh: usize,
  ctx: &str,
) -> u16 {
  let plane = filter_ramp();
  let got = gray16_filter_outputs(ow, oh, kernel);
  let y_ref = native_luma_filter(kernel, &plane, FW, FH, ow, oh);

  let mut max_diff = 0u16;
  for (i, (&g, &w)) in got.luma_u16.iter().zip(y_ref.iter()).enumerate() {
    max_diff = max_diff.max(g.abs_diff(w));
    assert_eq!(
      g, w,
      "{ctx} luma_u16[{i}]: {g} vs single-channel native-luma filter {w}"
    );
  }
  // Every derived output mirrors the area emit applied to the resampled luma:
  // the reference is the direct Gray16 sink run over the resampled-Y frame
  // (re-encoded LE, the host-native u16 plane recovered by the loader).
  let ref_pix = as_le_u16(&y_ref);
  let mut ref_luma = vec![0u8; ow * oh];
  let mut ref_luma_u16 = vec![0u16; ow * oh];
  let mut ref_rgb = vec![0u8; ow * oh * 3];
  let mut ref_rgba = vec![0u8; ow * oh * 4];
  let mut ref_rgb_u16 = vec![0u16; ow * oh * 3];
  let mut ref_rgba_u16 = vec![0u16; ow * oh * 4];
  let mut ref_h = vec![0u8; ow * oh];
  let mut ref_s = vec![0u8; ow * oh];
  let mut ref_v = vec![0u8; ow * oh];
  {
    let binned = Gray16Frame::new(&ref_pix, ow as u32, oh as u32, ow as u32);
    let mut sink = MixedSinker::<Gray16>::new(ow, oh)
      .with_luma(&mut ref_luma)
      .unwrap()
      .with_luma_u16(&mut ref_luma_u16)
      .unwrap()
      .with_rgb(&mut ref_rgb)
      .unwrap()
      .with_rgba(&mut ref_rgba)
      .unwrap()
      .with_rgb_u16(&mut ref_rgb_u16)
      .unwrap()
      .with_rgba_u16(&mut ref_rgba_u16)
      .unwrap()
      .with_hsv(&mut ref_h, &mut ref_s, &mut ref_v)
      .unwrap();
    gray16_to(&binned, FR, M, &mut sink).unwrap();
  }
  assert_eq!(got.luma, ref_luma, "{ctx} luma (>> 8)");
  assert_eq!(got.rgb, ref_rgb, "{ctx} rgb");
  assert_eq!(got.rgba, ref_rgba, "{ctx} rgba");
  assert_eq!(got.rgb_u16, ref_rgb_u16, "{ctx} rgb_u16");
  assert_eq!(got.rgba_u16, ref_rgba_u16, "{ctx} rgba_u16");
  assert_eq!(got.hp, ref_h, "{ctx} hsv H");
  assert_eq!(got.sp, ref_s, "{ctx} hsv S");
  assert_eq!(got.vp, ref_v, "{ctx} hsv V");
  max_diff
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gray16_filter_luma_is_single_channel_native_luma() {
  // Downscale 8->4 and upscale 4->7 under all three kernels; luma_u16 is the
  // single-channel `FilterStream<u16>` resample of the host-native source Y.
  for (ow, oh, tag) in [(FOUT_DOWN, FOUT_DOWN, "down"), (FUP, FUP, "up")] {
    assert_eq!(
      assert_gray16_filter_matches_oracle(Triangle, ow, oh, &format!("gray16 triangle {tag}")),
      0,
      "triangle {tag} luma_u16 diff must be 0"
    );
    assert_eq!(
      assert_gray16_filter_matches_oracle(CatmullRom, ow, oh, &format!("gray16 catmullrom {tag}")),
      0,
      "catmullrom {tag} luma_u16 diff must be 0"
    );
    assert_eq!(
      assert_gray16_filter_matches_oracle(Lanczos3, ow, oh, &format!("gray16 lanczos3 {tag}")),
      0,
      "lanczos3 {tag} luma_u16 diff must be 0"
    );
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gray16_filter_plan_is_accepted() {
  // A FilteredResampler plan is now ACCEPTED (Gray16 is routed to the filter
  // path) — it must NOT return UnsupportedFilter, and it must write the output.
  let plane = filter_ramp();
  let pix = as_le_u16(&plane);
  let src = Gray16Frame::new(&pix, FW as u32, FH as u32, FW as u32);
  let mut luma_u16 = vec![0xA5A5u16; FOUT_DOWN * FOUT_DOWN];
  {
    let mut sink = MixedSinker::<Gray16, FilteredResampler<Triangle>>::with_resampler(
      FW,
      FH,
      FilteredResampler::new(FOUT_DOWN, FOUT_DOWN, Triangle),
    )
    .unwrap()
    .with_luma_u16(&mut luma_u16)
    .unwrap();
    gray16_to(&src, FR, M, &mut sink).expect("filter plan must be accepted");
  }
  let y_ref = native_luma_filter(Triangle, &plane, FW, FH, FOUT_DOWN, FOUT_DOWN);
  assert_eq!(
    luma_u16, y_ref,
    "accepted filter luma_u16 = single-channel oracle"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gray16_filter_le_be_parity() {
  // The binned u16 luma is host-native; the derive kernels must use
  // `HOST_NATIVE_BE`, not `<false>`. The LE-vs-BE wire encodings of the same
  // logical plane must produce identical filter output on either host.
  let plane = filter_ramp();
  let pix_le = as_le_u16(&plane);
  let pix_be = as_be_u16(&plane);

  let mut le_luma_u16 = vec![0u16; FOUT_DOWN * FOUT_DOWN];
  let mut le_rgb_u16 = vec![0u16; FOUT_DOWN * FOUT_DOWN * 3];
  {
    let frame = Gray16Frame::new(&pix_le, FW as u32, FH as u32, FW as u32);
    let mut sink = MixedSinker::<Gray16, FilteredResampler<Triangle>>::with_resampler(
      FW,
      FH,
      FilteredResampler::new(FOUT_DOWN, FOUT_DOWN, Triangle),
    )
    .unwrap()
    .with_luma_u16(&mut le_luma_u16)
    .unwrap()
    .with_rgb_u16(&mut le_rgb_u16)
    .unwrap();
    gray16_to(&frame, FR, M, &mut sink).unwrap();
  }
  let mut be_luma_u16 = vec![0u16; FOUT_DOWN * FOUT_DOWN];
  let mut be_rgb_u16 = vec![0u16; FOUT_DOWN * FOUT_DOWN * 3];
  {
    let frame = Gray16Frame::<true>::new(&pix_be, FW as u32, FH as u32, FW as u32);
    let mut sink = MixedSinker::<Gray16<true>, FilteredResampler<Triangle>>::with_resampler(
      FW,
      FH,
      FilteredResampler::new(FOUT_DOWN, FOUT_DOWN, Triangle),
    )
    .unwrap()
    .with_luma_u16(&mut be_luma_u16)
    .unwrap()
    .with_rgb_u16(&mut be_rgb_u16)
    .unwrap();
    gray16_to_endian::<_, true>(&frame, FR, M, &mut sink).unwrap();
  }
  assert_eq!(le_luma_u16, be_luma_u16, "filter luma_u16 LE/BE diverge");
  assert_eq!(le_rgb_u16, be_rgb_u16, "filter rgb_u16 LE/BE diverge");
}
