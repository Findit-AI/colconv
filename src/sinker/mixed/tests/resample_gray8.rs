//! Fused-downscale coverage for `Gray8` — routed through a single
//! 1-channel `AreaStream<u8>` that bins the source Y plane (Gray *is* a
//! luma plane). Every attached output then derives from the binned Y
//! exactly as the direct path does: `luma` copies it, `luma_u16`
//! zero-extends it, `rgb` broadcasts Y to `[Y, Y, Y]`, `rgba` broadcasts
//! and pads alpha to `0xFF`, and `hsv` is `H=0 / S=0 / V=Y`. So every
//! resampled output equals the direct Gray8 sink run over a frame that
//! already holds the binned Y plane.

use crate::{
  KernelMatrix, PixelSink,
  resample::{AreaResampler, FilteredResampler, ResampleError, Triangle},
  sinker::{MixedSinker, MixedSinkerError},
  source::{Gray8, Gray8Row, gray8_to},
};
use mediaframe::frame::Gray8Frame;

const SRC: usize = 8;
const OUT: usize = 4;
// Gray is luma-only; the walker still threads a matrix / range through.
const FR: bool = true;
const M: KernelMatrix = KernelMatrix::Bt709;

fn gray8_frame(plane: &[u8]) -> Gray8Frame<'_> {
  Gray8Frame::new(plane, SRC as u32, SRC as u32, SRC as u32)
}

/// Interior Y ramp so the area mean sees real variation per 2x2 block.
fn ramp() -> Vec<u8> {
  let mut y = vec![0u8; SRC * SRC];
  for (i, p) in y.iter_mut().enumerate() {
    *p = 16 + (i % 200) as u8;
  }
  y
}

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

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gray8_downscale_luma_is_exact_area_mean() {
  let plane = ramp();
  let src = gray8_frame(&plane);

  let mut luma = vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Gray8, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_luma(&mut luma)
        .unwrap();
    gray8_to(&src, FR, M, &mut sink).unwrap();
  }
  assert_eq!(
    luma,
    block_mean_2x2(&plane),
    "luma must be the exact 2x2 block mean of the Y plane"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gray8_all_outputs_match_direct_over_binned_y() {
  // Every attached output — luma / luma_u16 / rgb / rgba / hsv — must be
  // exactly what the direct Gray8 sink produces over the (exact) binned
  // Y plane. The binned Y is the area mean, so we feed that mean as a
  // full-resolution `OUT`-grid Gray8 frame to the reference sink.
  let plane = ramp();
  let src = gray8_frame(&plane);

  let mut rgb = vec![0u8; OUT * OUT * 3];
  let mut rgba = vec![0u8; OUT * OUT * 4];
  let mut luma = vec![0u8; OUT * OUT];
  let mut luma_u16 = vec![0u16; OUT * OUT];
  let mut h = vec![0u8; OUT * OUT];
  let mut s_ = vec![0u8; OUT * OUT];
  let mut v_ = vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Gray8, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb(&mut rgb)
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap()
        .with_luma(&mut luma)
        .unwrap()
        .with_luma_u16(&mut luma_u16)
        .unwrap()
        .with_hsv(&mut h, &mut s_, &mut v_)
        .unwrap();
    gray8_to(&src, FR, M, &mut sink).unwrap();
  }

  // Reference: the direct sink over the exact binned Y plane.
  let binned_y = block_mean_2x2(&plane);
  let mut ref_rgb = vec![0u8; OUT * OUT * 3];
  let mut ref_rgba = vec![0u8; OUT * OUT * 4];
  let mut ref_luma = vec![0u8; OUT * OUT];
  let mut ref_luma_u16 = vec![0u16; OUT * OUT];
  let mut ref_h = vec![0u8; OUT * OUT];
  let mut ref_s = vec![0u8; OUT * OUT];
  let mut ref_v = vec![0u8; OUT * OUT];
  {
    let binned = Gray8Frame::new(&binned_y, OUT as u32, OUT as u32, OUT as u32);
    let mut sink = MixedSinker::<Gray8>::new(OUT, OUT)
      .with_rgb(&mut ref_rgb)
      .unwrap()
      .with_rgba(&mut ref_rgba)
      .unwrap()
      .with_luma(&mut ref_luma)
      .unwrap()
      .with_luma_u16(&mut ref_luma_u16)
      .unwrap()
      .with_hsv(&mut ref_h, &mut ref_s, &mut ref_v)
      .unwrap();
    gray8_to(&binned, FR, M, &mut sink).unwrap();
  }
  assert_eq!(luma, ref_luma, "luma");
  assert_eq!(luma_u16, ref_luma_u16, "luma_u16");
  assert_eq!(rgb, ref_rgb, "rgb");
  assert_eq!(rgba, ref_rgba, "rgba");
  assert_eq!(h, ref_h, "h");
  assert_eq!(s_, ref_s, "s");
  assert_eq!(v_, ref_v, "v");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gray8_standalone_rgba_matches_direct_over_binned_y() {
  // RGBA-only exercises the dedicated fast path (no RGB scratch).
  let plane = ramp();
  let src = gray8_frame(&plane);

  let mut rgba = vec![0u8; OUT * OUT * 4];
  {
    let mut sink =
      MixedSinker::<Gray8, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap();
    gray8_to(&src, FR, M, &mut sink).unwrap();
  }
  let binned_y = block_mean_2x2(&plane);
  let mut ref_rgba = vec![0u8; OUT * OUT * 4];
  {
    let binned = Gray8Frame::new(&binned_y, OUT as u32, OUT as u32, OUT as u32);
    let mut sink = MixedSinker::<Gray8>::new(OUT, OUT)
      .with_rgba(&mut ref_rgba)
      .unwrap();
    gray8_to(&binned, FR, M, &mut sink).unwrap();
  }
  assert_eq!(rgba, ref_rgba, "standalone rgba");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gray8_hsv_plus_rgba_matches_direct_over_binned_y() {
  // HSV + RGBA without RGB forces the RGB-kernel-into-scratch branch
  // (HSV derived from the scratch RGB, RGBA fanned out from it).
  let plane = ramp();
  let src = gray8_frame(&plane);

  let mut rgba = vec![0u8; OUT * OUT * 4];
  let mut h = vec![0u8; OUT * OUT];
  let mut s_ = vec![0u8; OUT * OUT];
  let mut v_ = vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Gray8, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap()
        .with_hsv(&mut h, &mut s_, &mut v_)
        .unwrap();
    gray8_to(&src, FR, M, &mut sink).unwrap();
  }
  let binned_y = block_mean_2x2(&plane);
  let mut ref_rgba = vec![0u8; OUT * OUT * 4];
  let mut ref_h = vec![0u8; OUT * OUT];
  let mut ref_s = vec![0u8; OUT * OUT];
  let mut ref_v = vec![0u8; OUT * OUT];
  {
    let binned = Gray8Frame::new(&binned_y, OUT as u32, OUT as u32, OUT as u32);
    let mut sink = MixedSinker::<Gray8>::new(OUT, OUT)
      .with_rgba(&mut ref_rgba)
      .unwrap()
      .with_hsv(&mut ref_h, &mut ref_s, &mut ref_v)
      .unwrap();
    gray8_to(&binned, FR, M, &mut sink).unwrap();
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
fn gray8_identity_plan_matches_new_sink() {
  let plane = ramp();
  let src = gray8_frame(&plane);

  let mut direct = vec![0u8; SRC * SRC * 3];
  {
    let mut sink = MixedSinker::<Gray8>::new(SRC, SRC)
      .with_rgb(&mut direct)
      .unwrap();
    gray8_to(&src, FR, M, &mut sink).unwrap();
  }
  let mut via_area = vec![0u8; SRC * SRC * 3];
  {
    let mut sink =
      MixedSinker::<Gray8, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(SRC, SRC))
        .unwrap()
        .with_rgb(&mut via_area)
        .unwrap();
    gray8_to(&src, FR, M, &mut sink).unwrap();
  }
  assert_eq!(direct, via_area, "identity plan must match the direct sink");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gray8_resample_reuses_luma_stream_across_frames() {
  // A reused sink must reset the luma stream each frame; without the
  // reset, frame 2's row 0 is rejected as out-of-sequence.
  let y1 = ramp();
  let mut y2 = y1.clone();
  for p in y2.iter_mut() {
    *p = 255 - *p;
  }
  let mut luma = vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Gray8, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_luma(&mut luma)
        .unwrap();
    gray8_to(&gray8_frame(&y1), FR, M, &mut sink).unwrap();
    gray8_to(&gray8_frame(&y2), FR, M, &mut sink).unwrap();
  }
  assert_eq!(
    luma,
    block_mean_2x2(&y2),
    "frame 2 luma must area-downscale frame 2's Y"
  );
}

#[test]
fn gray8_resample_no_outputs_is_a_no_op() {
  let plane = ramp();
  let src = gray8_frame(&plane);
  let mut sink =
    MixedSinker::<Gray8, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap();
  // No outputs attached: a legal no-op, accepted without error.
  gray8_to(&src, FR, M, &mut sink).unwrap();
  // A no-output call has no stream to sequence and never allocates.
  assert!(
    !sink.luma_stream_allocated(),
    "no-output sink allocated a stream"
  );
}

#[test]
fn gray8_out_of_sequence_first_row_rejected_before_allocation() {
  let plane = ramp();
  let row3 = &plane[3 * SRC..4 * SRC];

  let mut luma = vec![0u8; OUT * OUT];
  let mut sink =
    MixedSinker::<Gray8, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_luma(&mut luma)
      .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  // Feed row 3 first — the stream expects strict sequencing from 0.
  let err = sink.process(Gray8Row::new(row3, 3, M, FR)).unwrap_err();
  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
    ),
    "expected OutOfSequenceRow, got {err:?}"
  );
  // The out-of-sequence first row must be rejected before the luma
  // stream is allocated.
  assert!(
    !sink.luma_stream_allocated(),
    "stream allocated for a rejected row"
  );
  assert!(luma.iter().all(|&b| b == 0), "rejected row mutated output");
}

#[test]
fn gray8_rejected_first_row_does_not_poison_output_retry() {
  // A rejected out-of-sequence FIRST row must store no frozen-output
  // snapshot, so retrying row 0 after reconfiguring the output set
  // succeeds instead of tripping ResampleOutputsChanged against a
  // snapshot the rejected row should never have committed.
  let plane = ramp();
  let mut luma = vec![0u8; OUT * OUT];
  let mut sink =
    MixedSinker::<Gray8, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_luma(&mut luma)
      .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  let row3 = &plane[3 * SRC..4 * SRC];
  let err = sink.process(Gray8Row::new(row3, 3, M, FR)).unwrap_err();
  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
    ),
    "expected OutOfSequenceRow, got {err:?}"
  );
  let mut rgb = vec![0u8; OUT * OUT * 3];
  sink.set_rgb(&mut rgb).unwrap();
  sink
    .process(Gray8Row::new(&plane[..SRC], 0, M, FR))
    .expect("row 0 must succeed after a rejected out-of-sequence first row");
}

#[test]
fn gray8_resample_rejects_mid_frame_out_of_sequence() {
  let plane = ramp();
  let mut luma = vec![0u8; OUT * OUT];
  let mut sink =
    MixedSinker::<Gray8, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_luma(&mut luma)
      .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  sink
    .process(Gray8Row::new(&plane[..SRC], 0, M, FR))
    .unwrap();
  // Skip row 1 — feeding row 2 next is out of sequence.
  let err = sink
    .process(Gray8Row::new(&plane[2 * SRC..3 * SRC], 2, M, FR))
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
fn gray8_resample_rejects_mid_frame_output_change() {
  let plane = ramp();
  let mut rgb = vec![0u8; OUT * OUT * 3];
  let mut luma = vec![0u8; OUT * OUT];
  let mut sink =
    MixedSinker::<Gray8, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_rgb(&mut rgb)
      .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  sink
    .process(Gray8Row::new(&plane[..SRC], 0, M, FR))
    .unwrap();
  // Attaching a new output mid-frame trips the frozen-output check.
  sink.set_luma(&mut luma).unwrap();
  let err = sink
    .process(Gray8Row::new(&plane[SRC..2 * SRC], 1, M, FR))
    .unwrap_err();
  assert!(
    matches!(err, MixedSinkerError::ResampleOutputsChanged(_)),
    "expected ResampleOutputsChanged, got {err:?}"
  );
  assert!(
    luma.iter().all(|&b| b == 0),
    "rejected row mutated the new output"
  );
}

// The EMIT-owned out-width u8 RGB scratch — grown ONLY for the `hsv + rgba` shape
// with no `rgb` attached (`need_rgb_kernel && !want_rgb`), pregrown BEFORE the freeze
// via the shared driver's `None`-stream pass. Uses the crate's RGB-scratch
// failpoint (widened to `gray`), and attaches `rgba` + `hsv` outputs, so this
// test is `rgb`-gated (the module already provides `gray`).
#[cfg(feature = "rgb")]
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gray8_first_row_emit_scratch_oom_leaves_stream_and_freeze_uncommitted_for_retry() {
  let plane = ramp();
  let mut rgba = vec![0u8; OUT * OUT * 4];
  let mut hh = vec![0u8; OUT * OUT];
  let mut ss = vec![0u8; OUT * OUT];
  let mut vv = vec![0u8; OUT * OUT];
  let mut luma = vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Gray8, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap()
        .with_hsv(&mut hh, &mut ss, &mut vv)
        .unwrap();
    sink.begin_frame(SRC as u32, SRC as u32).unwrap();
    // Arm the RGB-scratch grow failpoint — the out-width u8 scratch the Gray8 emit
    // pregrows for its `hsv + rgba` (no `rgb`) shape, AFTER the row-0 stream box builds
    // into a local. The pregrow runs it BEFORE the freeze, so this OOM must leave BOTH
    // the luma stream field `None` and the freeze uncommitted (the post-freeze-grow
    // shape would have left the stream committed).
    crate::sinker::mixed::arm_rgb_scratch_alloc_failure();
    let err = sink
      .process(Gray8Row::new(&plane[..SRC], 0, M, FR))
      .unwrap_err();
    assert!(
      matches!(
        err,
        MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
      ),
      "first-row emit-scratch OOM must surface AllocationFailed, got {err:?}"
    );
    // DIRECT stream-rollback proof (the non-vacuous part): the luma stream field must
    // be `None` — the driver builds it into a LOCAL and inserts only after the emit
    // scratch pregrows, so a failed scratch leaves nothing committed. The
    // insert-before-scratch shape would leave it `Some` here, so this FAILS against
    // that shape and PASSES against the fix.
    assert!(
      !sink.luma_stream_allocated(),
      "an emit-owned scratch OOM must leave the luma stream field None (no partial commit)"
    );
    // The freeze was likewise never committed — attaching luma (a CHANGED output set)
    // and replaying from row 0 is ACCEPTED, not ResampleOutputsChanged.
    sink.set_luma(&mut luma).unwrap();
    for r in 0..SRC {
      sink
        .process(Gray8Row::new(&plane[r * SRC..(r + 1) * SRC], r, M, FR))
        .expect("frame replay after a first-row emit-scratch OOM must succeed");
    }
  }
  assert!(
    rgba.iter().any(|&b| b != 0),
    "the recovered frame produced real rgba output (emit-scratch OOM was recoverable)"
  );
  assert!(
    luma.iter().any(|&b| b != 0),
    "the newly-attached luma output was driven on the accepted retry"
  );
}

// ---- Filter-plan routing ----------------------------------------------------
//
// Gray8 *is* a luma plane: the area path bins the source Y through a 1-channel
// `AreaStream<u8>`; the filter path resamples the same Y through the
// signed-coefficient single-channel `FilterStream<u8>` (the filter twin of the
// bin). So the filter `luma` must equal a single-channel `FilterStream<u8>`
// resample of the source Y plane **byte for byte** (same engine, same
// coefficients, full-range u8 so no clamp on either), and every derived output
// (luma_u16 / rgb / rgba / hsv) follows from that resampled luma exactly as the
// area path derives from its binned luma.

use crate::resample::{CatmullRom, FilterKernel, FilterStream, Lanczos3, Resampler};

/// A larger, full-byte-wide grid than the 2:1 area fixture so a downscale
/// (`FW`->`FOUT_DOWN`) and an upscale (`FOUT_DOWN`->`FUP`) both run real,
/// non-trivial windows.
const FW: usize = 8;
const FH: usize = 8;
const FOUT_DOWN: usize = 4;
const FUP: usize = 7;

fn gray8_filter_frame(plane: &[u8]) -> Gray8Frame<'_> {
  Gray8Frame::new(plane, FW as u32, FH as u32, FW as u32)
}

/// A Y ramp with a hard mid-column edge per row so a filter window straddling
/// the edge produces intermediate grays (real antialiasing) rather than only
/// the endpoints.
fn filter_ramp() -> Vec<u8> {
  let mut y = vec![0u8; FW * FH];
  for row in 0..FH {
    for col in 0..FW {
      y[row * FW + col] = if col < FW / 2 { 32 } else { 220 };
    }
  }
  // Texture two interior rows so the vertical window varies too.
  for col in 0..FW {
    y[4 * FW + col] = (col * 30) as u8;
    y[5 * FW + col] = (255 - col * 30) as u8;
  }
  y
}

/// Single-channel filter resample of a u8 luma plane via the merged engine's
/// [`FilterStream<u8>`] (channels = 1) — the Gray8 luma oracle. Full-range u8,
/// so no native-depth clamp.
fn native_luma_filter<K: FilterKernel>(
  kernel: K,
  luma_plane: &[u8],
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
) -> Vec<u8> {
  let plan = FilteredResampler::new(ow, oh, kernel)
    .plan(sw, sh)
    .expect("valid filter plan")
    .expect("non-identity");
  let fh = plan.filter_h().expect("h windows");
  let fv = plan.filter_v().expect("v windows");
  let mut stream = FilterStream::<u8>::new(fh, fv, sw, sh, 1).expect("geometry");
  let mut out = vec![0u8; ow * oh];
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

/// Every resampled output a Gray8 filter equivalence asserts on.
struct Gray8FilterOutputs {
  luma: Vec<u8>,
  luma_u16: Vec<u16>,
  rgb: Vec<u8>,
  rgba: Vec<u8>,
  hp: Vec<u8>,
  sp: Vec<u8>,
  vp: Vec<u8>,
}

/// Run a `Gray8` filter sink over `filter_ramp()` at `ow x oh` under `kernel`,
/// attaching every output the equivalence asserts on.
fn gray8_filter_outputs<K: FilterKernel + Copy>(
  ow: usize,
  oh: usize,
  kernel: K,
) -> Gray8FilterOutputs {
  let plane = filter_ramp();
  let src = gray8_filter_frame(&plane);
  let mut o = Gray8FilterOutputs {
    luma: vec![0u8; ow * oh],
    luma_u16: vec![0u16; ow * oh],
    rgb: vec![0u8; ow * oh * 3],
    rgba: vec![0u8; ow * oh * 4],
    hp: vec![0u8; ow * oh],
    sp: vec![0u8; ow * oh],
    vp: vec![0u8; ow * oh],
  };
  {
    let mut sink = MixedSinker::<Gray8, FilteredResampler<K>>::with_resampler(
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
    .with_hsv(&mut o.hp, &mut o.sp, &mut o.vp)
    .unwrap();
    gray8_to(&src, FR, M, &mut sink).unwrap();
  }
  o
}

/// Asserts a `Gray8` filter resample's every output is derived from the
/// single-channel native-luma oracle exactly as the area emit derives from its
/// binned luma, and returns the max per-sample `luma` diff (exactly 0 — same
/// engine, no clamp).
fn assert_gray8_filter_matches_oracle<K: FilterKernel + Copy>(
  kernel: K,
  ow: usize,
  oh: usize,
  ctx: &str,
) -> u8 {
  let plane = filter_ramp();
  let got = gray8_filter_outputs(ow, oh, kernel);
  let y_ref = native_luma_filter(kernel, &plane, FW, FH, ow, oh);

  let mut max_diff = 0u8;
  for (i, (&g, &w)) in got.luma.iter().zip(y_ref.iter()).enumerate() {
    max_diff = max_diff.max(g.abs_diff(w));
    assert_eq!(
      g, w,
      "{ctx} luma[{i}]: {g} vs single-channel native-luma filter {w}"
    );
  }
  // Every derived output mirrors the area emit applied to the resampled luma:
  // the reference is the direct Gray8 sink run over the resampled-Y frame.
  let mut ref_luma_u16 = vec![0u16; ow * oh];
  let mut ref_rgb = vec![0u8; ow * oh * 3];
  let mut ref_rgba = vec![0u8; ow * oh * 4];
  let mut ref_h = vec![0u8; ow * oh];
  let mut ref_s = vec![0u8; ow * oh];
  let mut ref_v = vec![0u8; ow * oh];
  {
    let binned = Gray8Frame::new(&y_ref, ow as u32, oh as u32, ow as u32);
    let mut sink = MixedSinker::<Gray8>::new(ow, oh)
      .with_luma_u16(&mut ref_luma_u16)
      .unwrap()
      .with_rgb(&mut ref_rgb)
      .unwrap()
      .with_rgba(&mut ref_rgba)
      .unwrap()
      .with_hsv(&mut ref_h, &mut ref_s, &mut ref_v)
      .unwrap();
    gray8_to(&binned, FR, M, &mut sink).unwrap();
  }
  assert_eq!(got.luma_u16, ref_luma_u16, "{ctx} luma_u16");
  assert_eq!(got.rgb, ref_rgb, "{ctx} rgb");
  assert_eq!(got.rgba, ref_rgba, "{ctx} rgba");
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
fn gray8_filter_luma_is_single_channel_native_luma() {
  // Downscale 8->4 and upscale 4->7 under all three kernels; luma is the
  // single-channel `FilterStream<u8>` resample of the source Y, byte for byte.
  for (ow, oh, tag) in [(FOUT_DOWN, FOUT_DOWN, "down"), (FUP, FUP, "up")] {
    assert_eq!(
      assert_gray8_filter_matches_oracle(Triangle, ow, oh, &format!("gray8 triangle {tag}")),
      0,
      "triangle {tag} luma diff must be 0"
    );
    assert_eq!(
      assert_gray8_filter_matches_oracle(CatmullRom, ow, oh, &format!("gray8 catmullrom {tag}")),
      0,
      "catmullrom {tag} luma diff must be 0"
    );
    assert_eq!(
      assert_gray8_filter_matches_oracle(Lanczos3, ow, oh, &format!("gray8 lanczos3 {tag}")),
      0,
      "lanczos3 {tag} luma diff must be 0"
    );
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gray8_filter_plan_is_accepted() {
  // A FilteredResampler plan is now ACCEPTED (Gray8 is routed to the filter
  // path) — it must NOT return UnsupportedFilter, and it must write the output.
  let plane = filter_ramp();
  let src = gray8_filter_frame(&plane);
  let mut luma = vec![0x5Au8; FOUT_DOWN * FOUT_DOWN];
  {
    let mut sink = MixedSinker::<Gray8, FilteredResampler<Triangle>>::with_resampler(
      FW,
      FH,
      FilteredResampler::new(FOUT_DOWN, FOUT_DOWN, Triangle),
    )
    .unwrap()
    .with_luma(&mut luma)
    .unwrap();
    gray8_to(&src, FR, M, &mut sink).expect("filter plan must be accepted");
  }
  let y_ref = native_luma_filter(Triangle, &plane, FW, FH, FOUT_DOWN, FOUT_DOWN);
  assert_eq!(luma, y_ref, "accepted filter luma = single-channel oracle");
}
