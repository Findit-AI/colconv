//! Fused-downscale coverage for the packed-float-RGB family
//! (`Rgbf32`): the wire row converts to source-width host f32 RGB,
//! binning runs in float, the `rgb_f32` output is the exact area mean,
//! and every integer output (`rgb` / `rgba` / `rgb_u16` / `rgba_u16` /
//! `luma` / `luma_u16` / `hsv`) mirrors the direct Rgbf32 path's
//! clamp+scale kernels run over the binned row — the canonical oracle
//! being the direct full-res conversion over a pre-binned frame.

use crate::{
  KernelMatrix, PixelSink,
  frame::Rgbf32LeFrame,
  resample::{AreaResampler, ResampleError},
  sinker::{MixedSinker, MixedSinkerError},
  source::{Rgbf32, Rgbf32Row, rgbf32_to},
};

const SRC: usize = 8;
const OUT: usize = 4;

/// Re-encode a host-native f32 slice as **LE-encoded** byte storage,
/// so a fixture reads back identically on LE (no-op) and BE (byte-swap)
/// hosts. Mirrors `as_le_rgbf32` in the direct Rgbf32 tests.
fn as_le_rgbf32(host: &[f32]) -> Vec<f32> {
  host
    .iter()
    .map(|&v| f32::from_bits(u32::from_ne_bytes(v.to_bits().to_le_bytes())))
    .collect()
}

/// Per-channel f32 ramp with **integer-valued** samples (so the 2x2
/// area mean is exact in f32) that deliberately spans HDR (> 1.0) and
/// negative values — the float path must carry both losslessly into
/// `rgb_f32`, while the integer outputs saturate them per the direct
/// path's clamp.
fn packed_frame_f32() -> Vec<f32> {
  let mut buf = vec![0.0f32; SRC * SRC * 3];
  for (i, px) in buf.chunks_exact_mut(3).enumerate() {
    let i = i as i32;
    // R: small in-range integers and HDR (e.g. 2..) — saturates to the
    // integer max on the u8/u16 paths, preserved exactly in rgb_f32.
    px[0] = (i % 5) as f32; // 0..4, includes HDR (>1) values
    // G: large HDR values to exercise saturation.
    px[1] = (100 - i) as f32;
    // B: negative samples — clamp to 0 on integer outputs, preserved
    // exactly (with sign) in rgb_f32.
    px[2] = -((i % 7) as f32);
  }
  buf
}

/// Exact 2x2 block mean over host f32 values — integer-valued samples
/// divided by 4 (a power of two) are exactly representable, so this is
/// the bit-exact contract for the float area downscale.
fn expected_block_mean_f32(src: &[f32], ox: usize, oy: usize, c: usize) -> f32 {
  let mut acc = 0.0f64;
  for dy in 0..2 {
    for dx in 0..2 {
      acc += src[((oy * 2 + dy) * SRC + ox * 2 + dx) * 3 + c] as f64;
    }
  }
  (acc / 4.0) as f32
}

#[test]
fn rgbf32_downscale_rgb_f32_is_exact_area_mean() {
  let host = packed_frame_f32();
  let wire = as_le_rgbf32(&host);
  let src = Rgbf32LeFrame::try_new(&wire, SRC as u32, SRC as u32, (SRC * 3) as u32).unwrap();

  let mut rgb_f32 = vec![0.0f32; OUT * OUT * 3];
  {
    let mut sink =
      MixedSinker::<Rgbf32, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb_f32(&mut rgb_f32)
        .unwrap();
    rgbf32_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  }
  for oy in 0..OUT {
    for ox in 0..OUT {
      for c in 0..3 {
        let got = rgb_f32[(oy * OUT + ox) * 3 + c];
        let want = expected_block_mean_f32(&host, ox, oy, c);
        assert_eq!(got, want, "({ox},{oy}) c{c}: {got} != {want}");
      }
    }
  }
}

#[test]
fn rgbf32_downscale_rgb_f32_preserves_hdr_and_negative() {
  // The binned mean of an HDR-or-negative 2x2 block must itself be HDR
  // / negative — proving the float path never clamps `rgb_f32`. The
  // fixture's G channel is `100 - i` (always > 1) and B is `-(i % 7)`
  // (<= 0), so at least one binned cell is > 1 and at least one < 0.
  let host = packed_frame_f32();
  let wire = as_le_rgbf32(&host);
  let src = Rgbf32LeFrame::try_new(&wire, SRC as u32, SRC as u32, (SRC * 3) as u32).unwrap();

  let mut rgb_f32 = vec![0.0f32; OUT * OUT * 3];
  {
    let mut sink =
      MixedSinker::<Rgbf32, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb_f32(&mut rgb_f32)
        .unwrap();
    rgbf32_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  }
  let any_hdr = rgb_f32.iter().any(|&v| v > 1.0);
  let any_negative = rgb_f32.iter().any(|&v| v < 0.0);
  assert!(any_hdr, "binned rgb_f32 lost all HDR (> 1.0) values");
  assert!(any_negative, "binned rgb_f32 clamped away negative values");
}

#[test]
fn rgbf32_derived_outputs_come_from_binned_rgb() {
  // Every attached output — lossless f32, native-depth u16, and u8 —
  // must be exactly what the direct full-res Rgbf32 sink produces over
  // a frame that already holds the binned f32 RGB.
  let host = packed_frame_f32();
  let wire = as_le_rgbf32(&host);
  let src = Rgbf32LeFrame::try_new(&wire, SRC as u32, SRC as u32, (SRC * 3) as u32).unwrap();

  let mut rgb = vec![0u8; OUT * OUT * 3];
  let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
  let mut rgb_f32 = vec![0.0f32; OUT * OUT * 3];
  let mut rgba = vec![0u8; OUT * OUT * 4];
  let mut rgba_u16 = vec![0u16; OUT * OUT * 4];
  let mut luma = vec![0u8; OUT * OUT];
  let mut luma_u16 = vec![0u16; OUT * OUT];
  let mut h = vec![0u8; OUT * OUT];
  let mut s_ = vec![0u8; OUT * OUT];
  let mut v_ = vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Rgbf32, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
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
        .with_luma(&mut luma)
        .unwrap()
        .with_luma_u16(&mut luma_u16)
        .unwrap()
        .with_hsv(&mut h, &mut s_, &mut v_)
        .unwrap();
    rgbf32_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  }

  // Reference: the full-res sink over the (exact) binned f32 RGB,
  // re-encoded as an LE-wire plane exactly as the source arrived.
  let binned_wire = as_le_rgbf32(&rgb_f32);
  let mut ref_rgb = vec![0u8; OUT * OUT * 3];
  let mut ref_rgb_u16 = vec![0u16; OUT * OUT * 3];
  let mut ref_rgba = vec![0u8; OUT * OUT * 4];
  let mut ref_rgba_u16 = vec![0u16; OUT * OUT * 4];
  let mut ref_luma = vec![0u8; OUT * OUT];
  let mut ref_luma_u16 = vec![0u16; OUT * OUT];
  let mut ref_h = vec![0u8; OUT * OUT];
  let mut ref_s = vec![0u8; OUT * OUT];
  let mut ref_v = vec![0u8; OUT * OUT];
  {
    let binned =
      Rgbf32LeFrame::try_new(&binned_wire, OUT as u32, OUT as u32, (OUT * 3) as u32).unwrap();
    let mut sink = MixedSinker::<Rgbf32>::new(OUT, OUT)
      .with_rgb(&mut ref_rgb)
      .unwrap()
      .with_rgb_u16(&mut ref_rgb_u16)
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
    rgbf32_to(&binned, true, KernelMatrix::Bt709, &mut sink).unwrap();
  }
  assert_eq!(rgb, ref_rgb, "rgb");
  assert_eq!(rgb_u16, ref_rgb_u16, "rgb_u16");
  assert_eq!(rgba, ref_rgba, "rgba");
  assert_eq!(rgba_u16, ref_rgba_u16, "rgba_u16");
  assert_eq!(luma, ref_luma, "luma");
  assert_eq!(luma_u16, ref_luma_u16, "luma_u16");
  assert_eq!(h, ref_h, "h");
  assert_eq!(s_, ref_s, "s");
  assert_eq!(v_, ref_v, "v");
}

#[test]
fn rgbf32_identity_plan_matches_new_sink() {
  let host = packed_frame_f32();
  let wire = as_le_rgbf32(&host);
  let src = Rgbf32LeFrame::try_new(&wire, SRC as u32, SRC as u32, (SRC * 3) as u32).unwrap();

  let mut direct = vec![0.0f32; SRC * SRC * 3];
  {
    let mut sink = MixedSinker::<Rgbf32>::new(SRC, SRC)
      .with_rgb_f32(&mut direct)
      .unwrap();
    rgbf32_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  }
  let mut via_area = vec![0.0f32; SRC * SRC * 3];
  {
    let mut sink =
      MixedSinker::<Rgbf32, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(SRC, SRC))
        .unwrap()
        .with_rgb_f32(&mut via_area)
        .unwrap();
    rgbf32_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  }
  assert_eq!(direct, via_area);
}

#[test]
fn rgbf32_no_output_sink_is_a_noop() {
  // A resampled sink with no attached output neither allocates the
  // stream nor enforces sequencing — the documented legal no-op.
  let host = packed_frame_f32();
  let wire = as_le_rgbf32(&host);
  let src = Rgbf32LeFrame::try_new(&wire, SRC as u32, SRC as u32, (SRC * 3) as u32).unwrap();

  let mut sink =
    MixedSinker::<Rgbf32, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap();
  rgbf32_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  assert!(
    !sink.rgb_stream_f32_allocated(),
    "no-output sink allocated the f32 stream"
  );
  assert_eq!(
    sink.rgb_scratch_capacity(),
    0,
    "no-output sink grew the narrow scratch"
  );
}

#[test]
fn rgbf32_f32_only_downscale_does_not_size_the_narrow_scratch() {
  let host = packed_frame_f32();
  let wire = as_le_rgbf32(&host);
  let src = Rgbf32LeFrame::try_new(&wire, SRC as u32, SRC as u32, (SRC * 3) as u32).unwrap();

  // f32-only sink (only rgb_f32 attached): the binned row is copied
  // losslessly, so the u8 narrow scratch is never sized.
  let mut rgb_f32 = vec![0.0f32; OUT * OUT * 3];
  let mut sink =
    MixedSinker::<Rgbf32, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_rgb_f32(&mut rgb_f32)
      .unwrap();
  rgbf32_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  assert_eq!(
    sink.rgb_scratch_capacity(),
    0,
    "f32-only sink grew the u8 narrow scratch"
  );

  // u16-only sink likewise stays clear of the u8 narrow scratch
  // (rgb_u16 derives directly from the float binned row).
  let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
  let mut sink_u16 =
    MixedSinker::<Rgbf32, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_rgb_u16(&mut rgb_u16)
      .unwrap();
  rgbf32_to(&src, true, KernelMatrix::Bt709, &mut sink_u16).unwrap();
  assert_eq!(
    sink_u16.rgb_scratch_capacity(),
    0,
    "u16-only sink grew the u8 narrow scratch"
  );

  // Positive control: attaching a u8 output sizes it to the out-width
  // RGB row.
  let mut rgb = vec![0u8; OUT * OUT * 3];
  let mut sink2 =
    MixedSinker::<Rgbf32, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_rgb(&mut rgb)
      .unwrap();
  rgbf32_to(&src, true, KernelMatrix::Bt709, &mut sink2).unwrap();
  assert!(
    sink2.rgb_scratch_capacity() >= OUT * 3,
    "u8 output did not size the narrow scratch"
  );
}

#[test]
fn rgbf32_out_of_sequence_first_row_rejected_before_allocation() {
  let host = packed_frame_f32();
  let wire = as_le_rgbf32(&host);
  let row3 = &wire[3 * SRC * 3..4 * SRC * 3];

  let mut rgb_f32 = vec![0.0f32; OUT * OUT * 3];
  let mut sink =
    MixedSinker::<Rgbf32, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_rgb_f32(&mut rgb_f32)
      .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  let err = sink
    .process(Rgbf32Row::new(row3, 3, KernelMatrix::Bt709, true))
    .unwrap_err();
  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
    ),
    "expected OutOfSequenceRow, got {err:?}"
  );
  // The out-of-sequence first row must be rejected before the f32
  // stream or the narrow scratch is allocated.
  assert!(
    !sink.rgb_stream_f32_allocated(),
    "stream allocated for a rejected row"
  );
  assert_eq!(
    sink.rgb_scratch_capacity(),
    0,
    "scratch grown for a rejected row"
  );
}

#[test]
fn rgbf32_rejected_first_row_does_not_poison_output_retry() {
  // A rejected out-of-sequence FIRST row must store no frozen-output
  // snapshot (the split packed-float-RGB preflight rejects the OOS first
  // row before its freeze), so retrying row 0 after attaching a new output
  // succeeds instead of tripping ResampleOutputsChanged.
  let host = packed_frame_f32();
  let wire = as_le_rgbf32(&host);
  let mut rgb_f32 = vec![0.0f32; OUT * OUT * 3];
  let mut sink =
    MixedSinker::<Rgbf32, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_rgb_f32(&mut rgb_f32)
      .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  let err = sink
    .process(Rgbf32Row::new(
      &wire[3 * SRC * 3..4 * SRC * 3],
      3,
      KernelMatrix::Bt709,
      true,
    ))
    .unwrap_err();
  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
    ),
    "expected OutOfSequenceRow, got {err:?}"
  );
  let mut luma = vec![0u8; OUT * OUT];
  sink.set_luma(&mut luma).unwrap();
  sink
    .process(Rgbf32Row::new(
      &wire[..SRC * 3],
      0,
      KernelMatrix::Bt709,
      true,
    ))
    .expect("row 0 must succeed after a rejected out-of-sequence first row");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgbf32_first_row_scratch_oom_leaves_stream_and_freeze_uncommitted_for_retry() {
  let host = packed_frame_f32();
  let wire = as_le_rgbf32(&host);
  let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
  let mut luma = vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Rgbf32, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb_u16(&mut rgb_u16)
        .unwrap();
    sink.begin_frame(SRC as u32, SRC as u32).unwrap();
    // Arm the single-shot source-RGB-`f32`-scratch grow failpoint: `Rgbf32` stages
    // its wire row into the host-`f32` scratch, so the row-0 area stream box builds
    // OK (into a local) and the scratch grow — which runs AFTER the stream build and
    // BEFORE the field insert — refuses, surfacing AllocationFailed with BOTH the
    // stream field `None` and the freeze uncommitted (the commit-together atomic
    // shape; the pre-fix ordering would have left the stream committed).
    crate::sinker::mixed::arm_source_rgb_f32_scratch_failure();
    let err = sink
      .process(Rgbf32Row::new(
        &wire[..SRC * 3],
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
    // DIRECT stream-rollback proof (the non-vacuous part): the `f32` area stream
    // field must be `None` — the fix builds it into a LOCAL and inserts only after
    // the scratch grows, so a failed scratch leaves nothing committed. The
    // insert-before-scratch shape would leave it `Some` here, so this assertion FAILS
    // against that shape and PASSES against the fix (a `try_box` box failure cannot
    // make this distinction — it fails before any stream exists to insert).
    assert!(
      !sink.rgb_stream_f32_allocated(),
      "a scratch OOM must leave the rgb_stream_f32 field None (no partial commit)"
    );
    // The freeze was likewise never committed — attaching luma (a CHANGED output
    // set) and replaying from row 0 is ACCEPTED, not ResampleOutputsChanged.
    sink.set_luma(&mut luma).unwrap();
    for r in 0..SRC {
      sink
        .process(Rgbf32Row::new(
          &wire[r * SRC * 3..(r + 1) * SRC * 3],
          r,
          KernelMatrix::Bt709,
          true,
        ))
        .expect("frame replay after a first-row scratch OOM must succeed");
    }
  }
  assert!(
    rgb_u16.iter().any(|&b| b != 0),
    "the recovered frame produced real rgb_u16 output (scratch OOM was recoverable)"
  );
  assert!(
    luma.iter().any(|&b| b != 0),
    "the newly-attached luma output was driven on the accepted retry"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgbf32_first_row_emit_scratch_oom_leaves_stream_and_freeze_uncommitted_for_retry() {
  // The EMIT-owned out-width u8 narrow scratch (grown for a u8 `rgb` output), NOT
  // the source f32 staging scratch. The combined helper pre-grows every emit-owned
  // scratch via an `emit(None)` pregrow call BEFORE the freeze (mirroring the u16 /
  // u32 helpers), so a first-row OOM in it surfaces with the stream field `None` and
  // `resample_outputs` uncommitted — never a partial commit.
  let host = packed_frame_f32();
  let wire = as_le_rgbf32(&host);
  let mut rgb = vec![0u8; OUT * OUT * 3];
  let mut luma = vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Rgbf32, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb(&mut rgb)
        .unwrap();
    sink.begin_frame(SRC as u32, SRC as u32).unwrap();
    // Arm the u8 narrow-scratch (`source_rgb_scratch`) grow failpoint — the scratch
    // the `Rgbf32` emit grows for its u8 `rgb` output, AFTER the stream box builds.
    // The pregrow call runs it before the freeze, so this OOM must leave BOTH the
    // stream field `None` and the freeze uncommitted (the post-freeze-grow shape
    // would have left the stream committed).
    crate::sinker::mixed::arm_source_rgb_scratch_failure();
    let err = sink
      .process(Rgbf32Row::new(
        &wire[..SRC * 3],
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
      "first-row emit-scratch OOM must surface AllocationFailed, got {err:?}"
    );
    // DIRECT stream-rollback proof: an emit-OWNED scratch OOM must ALSO leave the
    // field `None` — the helper pregrows the emit scratch before the stream insert +
    // freeze, so nothing is committed. The post-freeze-grow shape would leave it
    // `Some` here (freeze committed, stream inserted), so this assertion FAILS
    // against that shape and PASSES against the fix.
    assert!(
      !sink.rgb_stream_f32_allocated(),
      "an emit-owned scratch OOM must leave the rgb_stream_f32 field None (no partial commit)"
    );
    // The freeze was likewise never committed — attaching luma (a CHANGED output
    // set) and replaying from row 0 is ACCEPTED, not ResampleOutputsChanged.
    sink.set_luma(&mut luma).unwrap();
    for r in 0..SRC {
      sink
        .process(Rgbf32Row::new(
          &wire[r * SRC * 3..(r + 1) * SRC * 3],
          r,
          KernelMatrix::Bt709,
          true,
        ))
        .expect("frame replay after a first-row emit-scratch OOM must succeed");
    }
  }
  assert!(
    rgb.iter().any(|&b| b != 0),
    "the recovered frame produced real rgb output (emit-scratch OOM was recoverable)"
  );
  assert!(
    luma.iter().any(|&b| b != 0),
    "the newly-attached luma output was driven on the accepted retry"
  );
}

#[test]
fn rgbf32_mid_frame_out_of_sequence_rejected() {
  let host = packed_frame_f32();
  let wire = as_le_rgbf32(&host);

  let mut rgb_f32 = vec![0.0f32; OUT * OUT * 3];
  let mut sink =
    MixedSinker::<Rgbf32, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_rgb_f32(&mut rgb_f32)
      .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  sink
    .process(Rgbf32Row::new(
      &wire[..SRC * 3],
      0,
      KernelMatrix::Bt709,
      true,
    ))
    .unwrap();
  // Skip row 1 — feeding row 2 next must be rejected.
  let err = sink
    .process(Rgbf32Row::new(
      &wire[2 * SRC * 3..3 * SRC * 3],
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

#[test]
fn rgbf32_mid_frame_output_change_rejected() {
  let host = packed_frame_f32();
  let wire = as_le_rgbf32(&host);

  let mut rgb_f32 = vec![0.0f32; OUT * OUT * 3];
  let mut luma = vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Rgbf32, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb_f32(&mut rgb_f32)
        .unwrap();
    sink.begin_frame(SRC as u32, SRC as u32).unwrap();
    sink
      .process(Rgbf32Row::new(
        &wire[..SRC * 3],
        0,
        KernelMatrix::Bt709,
        true,
      ))
      .unwrap();
    sink.set_luma(&mut luma).unwrap();
    let err = sink
      .process(Rgbf32Row::new(
        &wire[SRC * 3..SRC * 6],
        1,
        KernelMatrix::Bt709,
        true,
      ))
      .unwrap_err();
    assert!(matches!(err, MixedSinkerError::ResampleOutputsChanged(_)));
  }
  assert!(luma.iter().all(|&l| l == 0));
}
