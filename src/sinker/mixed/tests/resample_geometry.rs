//! Geometry-split contract: output buffers validate against the
//! resampler's output geometry while `begin_frame` keeps validating the
//! walker against the source geometry.

use crate::{
  PixelSink,
  resample::{AreaResampler, NoopResampler, ResampleError},
  sinker::{MixedSinker, MixedSinkerError, mixed::HsvPlane},
  source::Yuv420p,
};

const SRC: usize = 8;
const OUT: usize = 4;

fn downscaled<'a>() -> MixedSinker<'a, Yuv420p, AreaResampler> {
  MixedSinker::<Yuv420p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
    .expect("8x8 -> 4x4 area plan never fails")
}

#[test]
fn new_keeps_output_geometry_equal_to_source() {
  let sink = MixedSinker::<Yuv420p>::new(SRC, SRC);
  assert_eq!((sink.width(), sink.height()), (SRC, SRC));
  assert_eq!((sink.out_width(), sink.out_height()), (SRC, SRC));
}

#[test]
fn with_resampler_noop_matches_new() {
  let sink = MixedSinker::<Yuv420p, NoopResampler>::with_resampler(SRC, SRC, NoopResampler)
    .expect("identity plan never fails");
  assert_eq!((sink.width(), sink.height()), (SRC, SRC));
  assert_eq!((sink.out_width(), sink.out_height()), (SRC, SRC));
}

#[test]
fn with_resampler_area_identity_matches_new() {
  let sink =
    MixedSinker::<Yuv420p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(SRC, SRC))
      .expect("identity area plan never fails");
  assert_eq!((sink.out_width(), sink.out_height()), (SRC, SRC));
}

#[test]
fn with_resampler_downscale_shrinks_output_geometry() {
  let sink = downscaled();
  assert_eq!((sink.width(), sink.height()), (SRC, SRC));
  assert_eq!((sink.out_width(), sink.out_height()), (OUT, OUT));
}

#[test]
fn rgb_buffer_validates_against_output_geometry() {
  let mut short = vec![0u8; OUT * OUT * 3 - 1];
  let err = downscaled().with_rgb(&mut short).map(|_| ()).unwrap_err();
  match err {
    MixedSinkerError::InsufficientRgbBuffer(e) => {
      assert_eq!(e.expected(), OUT * OUT * 3);
      assert_eq!(e.actual(), OUT * OUT * 3 - 1);
    }
    other => panic!("expected InsufficientRgbBuffer, got {other:?}"),
  }

  let mut exact = vec![0u8; OUT * OUT * 3];
  assert!(downscaled().with_rgb(&mut exact).is_ok());
}

#[test]
fn luma_buffer_validates_against_output_geometry() {
  let mut short = vec![0u8; OUT * OUT - 1];
  let err = downscaled().with_luma(&mut short).map(|_| ()).unwrap_err();
  match err {
    MixedSinkerError::InsufficientLumaBuffer(e) => assert_eq!(e.expected(), OUT * OUT),
    other => panic!("expected InsufficientLumaBuffer, got {other:?}"),
  }

  let mut exact = vec![0u8; OUT * OUT];
  assert!(downscaled().with_luma(&mut exact).is_ok());
}

#[test]
fn luma_u16_buffer_validates_against_output_geometry() {
  let mut short = vec![0u16; OUT * OUT - 1];
  let err = downscaled()
    .with_luma_u16(&mut short)
    .map(|_| ())
    .unwrap_err();
  match err {
    MixedSinkerError::InsufficientLumaU16Buffer(e) => assert_eq!(e.expected(), OUT * OUT),
    other => panic!("expected InsufficientLumaU16Buffer, got {other:?}"),
  }

  let mut exact = vec![0u16; OUT * OUT];
  assert!(downscaled().with_luma_u16(&mut exact).is_ok());
}

#[test]
fn rgba_buffer_validates_against_output_geometry() {
  let mut short = vec![0u8; OUT * OUT * 4 - 1];
  let err = downscaled().with_rgba(&mut short).map(|_| ()).unwrap_err();
  match err {
    MixedSinkerError::InsufficientRgbaBuffer(e) => assert_eq!(e.expected(), OUT * OUT * 4),
    other => panic!("expected InsufficientRgbaBuffer, got {other:?}"),
  }

  let mut exact = vec![0u8; OUT * OUT * 4];
  assert!(downscaled().with_rgba(&mut exact).is_ok());
}

#[test]
fn hsv_planes_validate_against_output_geometry() {
  let mut h = vec![0u8; OUT * OUT - 1];
  let mut s = vec![0u8; OUT * OUT];
  let mut v = vec![0u8; OUT * OUT];
  let err = downscaled()
    .with_hsv(&mut h, &mut s, &mut v)
    .map(|_| ())
    .unwrap_err();
  match err {
    MixedSinkerError::InsufficientHsvPlane(e) => {
      assert_eq!(e.which(), HsvPlane::H);
      assert_eq!(e.expected(), OUT * OUT);
    }
    other => panic!("expected InsufficientHsvPlane, got {other:?}"),
  }

  let mut h = vec![0u8; OUT * OUT];
  assert!(downscaled().with_hsv(&mut h, &mut s, &mut v).is_ok());
}

#[test]
fn begin_frame_still_validates_source_geometry() {
  // The walker contract is unchanged under a non-identity plan: frames
  // validate against the SOURCE geometry, never the output geometry.
  let mut sink = downscaled();
  assert!(sink.begin_frame(SRC as u32, SRC as u32).is_ok());
  assert!(matches!(
    sink.begin_frame(OUT as u32, OUT as u32),
    Err(MixedSinkerError::DimensionMismatch(_))
  ));
}

#[test]
fn plan_error_surfaces_as_mixed_sinker_error() {
  let err =
    MixedSinker::<Yuv420p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(0, 0))
      .map(|_| ())
      .unwrap_err();
  assert!(matches!(
    err,
    MixedSinkerError::Resample(ResampleError::ZeroOutputDimension(_))
  ));
}

// ---- End-to-end fused downscale (Yuv420p row-stage tier) ----------------

use crate::{KernelMatrix, frame::Yuv420pFrame, source::yuv420p_to};

/// 8x8 frame whose Y plane is the row-major ramp `8*row + col` with
/// neutral chroma: full-range luma equals the Y plane verbatim, so
/// area means are hand-computable.
fn gradient_frame_planes() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let y: Vec<u8> = (0..64u8).collect();
  (y, vec![128u8; 16], vec![128u8; 16])
}

#[test]
fn downscale_yuv420p_gradient_luma_integer_ratio() {
  let (yp, up, vp) = gradient_frame_planes();
  let src = Yuv420pFrame::new(&yp, &up, &vp, 8, 8, 8, 4, 4);

  let mut luma = vec![0u8; OUT * OUT];
  let mut sink =
    MixedSinker::<Yuv420p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_luma(&mut luma)
      .unwrap();
  yuv420p_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();

  // 2x2 block mean of the ramp is `16r + 2c + 4.5`; round-half-up.
  for r in 0..OUT {
    for c in 0..OUT {
      assert_eq!(luma[r * OUT + c], (16 * r + 2 * c + 5) as u8, "({r},{c})");
    }
  }
}

#[test]
fn downscale_yuv420p_gradient_luma_fractional_ratio() {
  let (yp, up, vp) = gradient_frame_planes();
  let src = Yuv420pFrame::new(&yp, &up, &vp, 8, 8, 8, 4, 4);

  let mut luma = vec![0u8; 3 * 3];
  let mut sink =
    MixedSinker::<Yuv420p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(3, 3))
      .unwrap()
      .with_luma(&mut luma)
      .unwrap();
  yuv420p_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();

  // Independent reference: direct 2-D area mean over the ramp with the
  // exact coverage weights (x3 grid), denominator 64, round-half-up.
  let spans = [
    (0usize, [3u64, 3, 2].as_slice()),
    (2, &[1, 3, 3, 1]),
    (5, &[2, 3, 3]),
  ];
  for (r, &(vy, vw)) in spans.iter().enumerate() {
    for (c, &(hx, hw)) in spans.iter().enumerate() {
      let mut acc = 0u64;
      for (dy, &wy) in vw.iter().enumerate() {
        for (dx, &wx) in hw.iter().enumerate() {
          acc += wy * wx * (8 * (vy + dy) + hx + dx) as u64;
        }
      }
      let expected = ((acc + 32) / 64) as u8;
      assert_eq!(luma[r * 3 + c], expected, "({r},{c})");
    }
  }
}

#[test]
fn downscale_yuv420p_solid_matches_full_res_conversion() {
  // A solid frame's downscale must equal the full-res conversion's
  // solid value on every output channel.
  let yp = vec![120u8; 64];
  let up = vec![90u8; 16];
  let vp = vec![170u8; 16];
  let src = Yuv420pFrame::new(&yp, &up, &vp, 8, 8, 8, 4, 4);

  let mut full_rgb = vec![0u8; SRC * SRC * 3];
  let mut full_h = vec![0u8; SRC * SRC];
  let mut full_s = vec![0u8; SRC * SRC];
  let mut full_v = vec![0u8; SRC * SRC];
  let mut full = MixedSinker::<Yuv420p>::new(SRC, SRC)
    .with_rgb(&mut full_rgb)
    .unwrap()
    .with_hsv(&mut full_h, &mut full_s, &mut full_v)
    .unwrap();
  yuv420p_to(&src, false, full.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();

  let mut rgb = vec![0u8; OUT * OUT * 3];
  let mut rgba = vec![0u8; OUT * OUT * 4];
  let mut luma = vec![0u8; OUT * OUT];
  let mut luma_u16 = vec![0u16; OUT * OUT];
  let mut h = vec![0u8; OUT * OUT];
  let mut s = vec![0u8; OUT * OUT];
  let mut v = vec![0u8; OUT * OUT];
  let mut sink =
    MixedSinker::<Yuv420p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_rgb(&mut rgb)
      .unwrap()
      .with_rgba(&mut rgba)
      .unwrap()
      .with_luma(&mut luma)
      .unwrap()
      .with_luma_u16(&mut luma_u16)
      .unwrap()
      .with_hsv(&mut h, &mut s, &mut v)
      .unwrap();
  yuv420p_to(&src, false, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();

  let (er, eg, eb) = (full_rgb[0], full_rgb[1], full_rgb[2]);
  for px in rgb.chunks_exact(3) {
    assert_eq!((px[0], px[1], px[2]), (er, eg, eb));
  }
  for px in rgba.chunks_exact(4) {
    assert_eq!((px[0], px[1], px[2], px[3]), (er, eg, eb, 0xFF));
  }
  assert!(luma.iter().all(|&l| l == 120));
  assert!(luma_u16.iter().all(|&l| l == 120));
  assert!(h.iter().all(|&x| x == full_h[0]));
  assert!(s.iter().all(|&x| x == full_s[0]));
  assert!(v.iter().all(|&x| x == full_v[0]));
}

#[test]
fn downscale_luma_only_works_without_rgb_buffers() {
  let (yp, up, vp) = gradient_frame_planes();
  let src = Yuv420pFrame::new(&yp, &up, &vp, 8, 8, 8, 4, 4);

  let mut luma = vec![0u8; OUT * OUT];
  let mut sink =
    MixedSinker::<Yuv420p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_luma(&mut luma)
      .unwrap();
  yuv420p_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
  assert_eq!(luma[0], 5);
}

#[test]
fn downscale_state_resets_between_frames() {
  let (yp, up, vp) = gradient_frame_planes();
  let src1 = Yuv420pFrame::new(&yp, &up, &vp, 8, 8, 8, 4, 4);
  let solid = vec![40u8; 64];
  let src2 = Yuv420pFrame::new(&solid, &up, &vp, 8, 8, 8, 4, 4);

  let mut luma = vec![0u8; OUT * OUT];
  let mut sink =
    MixedSinker::<Yuv420p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_luma(&mut luma)
      .unwrap();
  yuv420p_to(&src1, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
  yuv420p_to(&src2, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
  assert!(
    luma.iter().all(|&l| l == 40),
    "second frame must not inherit state"
  );
}

#[test]
fn direct_out_of_order_process_rejected_under_resampling() {
  use crate::source::Yuv420pRow;

  let mut luma = vec![0u8; OUT * OUT];
  let mut sink = downscaled().with_luma(&mut luma).unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();

  let y = [0u8; SRC];
  let u = [128u8; SRC / 2];
  let v = [128u8; SRC / 2];
  // Row 3 before rows 0..3: the stream must reject, not corrupt.
  let err = sink
    .process(Yuv420pRow::for_tests(
      &y,
      &u,
      &v,
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
fn rejected_first_row_does_not_poison_output_retry_row_stage() {
  use crate::source::Yuv420pRow;

  // A rejected out-of-sequence FIRST row must store no frozen-output
  // snapshot, so retrying row 0 after attaching a NEW output succeeds
  // instead of tripping ResampleOutputsChanged (the row-stage 4:2:0 path).
  let y = [50u8; SRC];
  let u = [128u8; SRC / 2];
  let v = [128u8; SRC / 2];
  let mut luma = vec![0u8; OUT * OUT];
  let mut sink = downscaled().with_luma(&mut luma).unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  let err = sink
    .process(Yuv420pRow::for_tests(
      &y,
      &u,
      &v,
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
    .process(Yuv420pRow::for_tests(
      &y,
      &u,
      &v,
      0,
      KernelMatrix::Bt601,
      true,
    ))
    .expect("row 0 must succeed after a rejected out-of-sequence first row");
}

#[test]
fn rejected_first_row_does_not_poison_output_retry_native() {
  use crate::source::Yuv420pRow;

  // Same poison contract for the NATIVE 4:2:0 tier (with_native(true)):
  // a rejected out-of-sequence first row stores no snapshot, so retrying
  // row 0 after attaching a NEW output succeeds.
  let y = [50u8; SRC];
  let u = [128u8; SRC / 2];
  let v = [128u8; SRC / 2];
  let mut luma = vec![0u8; OUT * OUT];
  let mut sink = downscaled().with_native(true).with_luma(&mut luma).unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  let err = sink
    .process(Yuv420pRow::for_tests(
      &y,
      &u,
      &v,
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
    .process(Yuv420pRow::for_tests(
      &y,
      &u,
      &v,
      0,
      KernelMatrix::Bt601,
      true,
    ))
    .expect("row 0 must succeed after a rejected out-of-sequence first row");
}

#[test]
#[cfg(feature = "rgb")]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn native_first_build_scratch_oom_leaves_freeze_unfrozen_for_retry() {
  use crate::source::Yuv420pRow;

  // The 4:2:0 native delegate grows the output RGB scratch as a PRE-FEED step,
  // BEFORE it commits the output-set freeze. A scratch OOM on an ordinary
  // first-ever row-0 build therefore leaves `resample_outputs` uncommitted (the
  // delegate never freezes on a pre-feed failure), so a retry of row 0 with a
  // CHANGED output attachment is accepted, not mis-rejected as
  // ResampleOutputsChanged.
  let y = [50u8; SRC];
  let u = [128u8; SRC / 2];
  let v = [128u8; SRC / 2];
  let mut rgb = vec![0u8; OUT * OUT * 3];
  let mut luma = vec![0u8; OUT * OUT];
  let mut sink = downscaled().with_native(true).with_rgb(&mut rgb).unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  crate::sinker::mixed::arm_native_rgb_scratch_failure();
  let err = sink
    .process(Yuv420pRow::for_tests(
      &y,
      &u,
      &v,
      0,
      KernelMatrix::Bt601,
      true,
    ))
    .unwrap_err();
  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
    ),
    "first-build RGB-scratch OOM must surface AllocationFailed, got {err:?}"
  );
  // The freeze was never committed on the failed build — a retry of row 0 with
  // luma added (changed output set) is ACCEPTED, not ResampleOutputsChanged.
  sink.set_luma(&mut luma).unwrap();
  sink
    .process(Yuv420pRow::for_tests(
      &y,
      &u,
      &v,
      0,
      KernelMatrix::Bt601,
      true,
    ))
    .expect("row 0 must succeed after a first-build scratch OOM (freeze uncommitted)");
}

#[test]
#[cfg(feature = "rgb")]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn native_colour_capability_rebuild_scratch_oom_leaves_freeze_unfrozen_for_retry() {
  use crate::source::Yuv420pRow;

  // A cross-frame COLOUR-CAPABILITY rebuild: frame 1 builds a luma-only native
  // join (chroma absent, no output RGB scratch). Frame 2 attaches RGB, so the
  // delegate rebuilds the join WITH chroma and grows the (empty) output RGB
  // scratch — both before the row is accepted. A scratch OOM on that rebuild
  // must leave the cached luma-only join intact AND `resample_outputs`
  // uncommitted (the delegate builds into a local and freezes only after every
  // pre-feed allocation succeeds), so a row-0 retry with ANOTHER output attached
  // is accepted, not mis-rejected as ResampleOutputsChanged.
  let y = [50u8; SRC];
  let u = [128u8; SRC / 2];
  let v = [128u8; SRC / 2];
  let mut luma = vec![0u8; OUT * OUT];
  let mut rgb = vec![0u8; OUT * OUT * 3];
  let (mut hh, mut ss, mut vv) = (
    vec![0u8; OUT * OUT],
    vec![0u8; OUT * OUT],
    vec![0u8; OUT * OUT],
  );
  let mut sink = downscaled().with_native(true).with_luma(&mut luma).unwrap();
  // Frame 1: a luma-only native join (chroma absent, reserves no RGB scratch).
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  for r in 0..SRC {
    sink
      .process(Yuv420pRow::for_tests(
        &y,
        &u,
        &v,
        r,
        KernelMatrix::Bt601,
        true,
      ))
      .expect("luma-only frame builds the native join");
  }
  // Frame 2: attach RGB → the row-0 rebuild must build the chroma half and grow
  // the (now-needed, empty) output RGB scratch; refuse that grow.
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  sink.set_rgb(&mut rgb).unwrap();
  crate::sinker::mixed::arm_native_rgb_scratch_failure();
  let err = sink
    .process(Yuv420pRow::for_tests(
      &y,
      &u,
      &v,
      0,
      KernelMatrix::Bt601,
      true,
    ))
    .unwrap_err();
  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
    ),
    "the colour-capability rebuild scratch OOM must surface AllocationFailed, got {err:?}"
  );
  // The output-set freeze was never committed (the delegate commits only after
  // the scratch grows), so attaching ANOTHER output (hsv) and retrying row 0 is
  // ACCEPTED — the changed output set proves the freeze was rolled back. The
  // pre-fold shape would have frozen {luma, rgb} before the scratch failure and
  // rejected this retry as ResampleOutputsChanged.
  sink.set_hsv(&mut hh, &mut ss, &mut vv).unwrap();
  sink
    .process(Yuv420pRow::for_tests(
      &y,
      &u,
      &v,
      0,
      KernelMatrix::Bt601,
      true,
    ))
    .expect("row 0 with a changed output set must succeed after a rebuild scratch OOM");
}

#[test]
fn no_output_first_row_does_not_poison_output_retry_native() {
  use crate::source::Yuv420pRow;

  // The NATIVE 4:2:0 tier always feeds the Y plane on an output-bearing
  // row (luma is implicit), so its sequence counter is the Y stream — which
  // has no no-output condition. A no-output call must therefore be gated on
  // has-output and short-circuit BEFORE the freeze / Y feed / join
  // allocation; otherwise it snapshots the all-None output set, advances Y,
  // and a later attach-then-retry of row 0 trips ResampleOutputsChanged (the
  // frozen all-None set no longer matches) or OutOfSequenceRow (Y already at
  // row 1). With no output attached, processing a row must be a no-op Ok,
  // and a subsequent row 0 after attaching an output must succeed.
  let y = [50u8; SRC];
  let u = [128u8; SRC / 2];
  let v = [128u8; SRC / 2];
  let mut sink = downscaled().with_native(true);
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  sink
    .process(Yuv420pRow::for_tests(
      &y,
      &u,
      &v,
      0,
      KernelMatrix::Bt601,
      true,
    ))
    .expect("a no-output native row must be a no-op Ok");
  let mut luma = vec![0u8; OUT * OUT];
  sink.set_luma(&mut luma).unwrap();
  sink
    .process(Yuv420pRow::for_tests(
      &y,
      &u,
      &v,
      0,
      KernelMatrix::Bt601,
      true,
    ))
    .expect("row 0 must succeed after a no-output native call (no poisoned snapshot)");
}

#[test]
fn mid_frame_output_reconfiguration_rejected_atomically() {
  use crate::source::Yuv420pRow;

  let y = [50u8; SRC];
  let u = [128u8; SRC / 2];
  let v = [128u8; SRC / 2];
  let mut rgb = vec![0u8; OUT * OUT * 3];

  // Attaching a new output mid-frame desyncs the (fresh) color stream
  // from the in-flight luma stream: the call must fail BEFORE any
  // stream mutates caller output.
  let mut luma = vec![0u8; OUT * OUT];
  {
    let mut sink = downscaled().with_luma(&mut luma).unwrap();
    sink.begin_frame(SRC as u32, SRC as u32).unwrap();
    sink
      .process(Yuv420pRow::for_tests(
        &y,
        &u,
        &v,
        0,
        KernelMatrix::Bt601,
        true,
      ))
      .unwrap();
    sink.set_rgb(&mut rgb).unwrap();
    // Row 1 would have completed luma output row 0.
    let err = sink
      .process(Yuv420pRow::for_tests(
        &y,
        &u,
        &v,
        1,
        KernelMatrix::Bt601,
        true,
      ))
      .unwrap_err();
    assert!(matches!(err, MixedSinkerError::ResampleOutputsChanged(_)));
  }
  assert!(
    luma.iter().all(|&l| l == 0),
    "luma mutated on a failed call"
  );

  // begin_frame restarts every stream on the SAME sink: a full ordered
  // frame after the failed call succeeds.
  let mut luma = vec![0u8; OUT * OUT];
  {
    let mut rgb = vec![0u8; OUT * OUT * 3];
    let mut sink = downscaled().with_luma(&mut luma).unwrap();
    sink.begin_frame(SRC as u32, SRC as u32).unwrap();
    sink
      .process(Yuv420pRow::for_tests(
        &y,
        &u,
        &v,
        0,
        KernelMatrix::Bt601,
        true,
      ))
      .unwrap();
    sink.set_rgb(&mut rgb).unwrap();
    sink
      .process(Yuv420pRow::for_tests(
        &y,
        &u,
        &v,
        1,
        KernelMatrix::Bt601,
        true,
      ))
      .unwrap_err();
    sink.begin_frame(SRC as u32, SRC as u32).unwrap();
    for row in 0..SRC {
      sink
        .process(Yuv420pRow::for_tests(
          &y,
          &u,
          &v,
          row,
          KernelMatrix::Bt601,
          true,
        ))
        .unwrap();
    }
  }
  assert!(luma.iter().all(|&l| l == 50));
}

#[test]
fn same_group_mid_frame_attachment_rejected_atomically() {
  use crate::source::Yuv420pRow;

  let y = [50u8; SRC];
  let u = [128u8; SRC / 2];
  let v = [128u8; SRC / 2];

  // Color group: HSV joining after RGB has already emitted rows.
  let mut rgb = vec![0u8; OUT * OUT * 3];
  let mut h = vec![0u8; OUT * OUT];
  let mut s_ = vec![0u8; OUT * OUT];
  let mut v_ = vec![0u8; OUT * OUT];
  {
    let mut sink = downscaled().with_rgb(&mut rgb).unwrap();
    sink.begin_frame(SRC as u32, SRC as u32).unwrap();
    for row in 0..2 {
      sink
        .process(Yuv420pRow::for_tests(
          &y,
          &u,
          &v,
          row,
          KernelMatrix::Bt601,
          true,
        ))
        .unwrap();
    }
    sink.set_hsv(&mut h, &mut s_, &mut v_).unwrap();
    let err = sink
      .process(Yuv420pRow::for_tests(
        &y,
        &u,
        &v,
        2,
        KernelMatrix::Bt601,
        true,
      ))
      .unwrap_err();
    assert!(matches!(err, MixedSinkerError::ResampleOutputsChanged(_)));
  }
  assert!(h.iter().all(|&x| x == 0), "late HSV must stay untouched");

  // Luma group: luma_u16 joining after luma has emitted rows.
  let mut luma = vec![0u8; OUT * OUT];
  let mut luma16 = vec![0u16; OUT * OUT];
  {
    let mut sink = downscaled().with_luma(&mut luma).unwrap();
    sink.begin_frame(SRC as u32, SRC as u32).unwrap();
    for row in 0..2 {
      sink
        .process(Yuv420pRow::for_tests(
          &y,
          &u,
          &v,
          row,
          KernelMatrix::Bt601,
          true,
        ))
        .unwrap();
    }
    sink.set_luma_u16(&mut luma16).unwrap();
    let err = sink
      .process(Yuv420pRow::for_tests(
        &y,
        &u,
        &v,
        2,
        KernelMatrix::Bt601,
        true,
      ))
      .unwrap_err();
    assert!(matches!(err, MixedSinkerError::ResampleOutputsChanged(_)));
  }
  assert!(luma16.iter().all(|&x| x == 0));
}

#[test]
fn same_channel_buffer_replacement_rejected_atomically() {
  use crate::source::Yuv420pRow;

  let y = [50u8; SRC];
  let u = [128u8; SRC / 2];
  let v = [128u8; SRC / 2];

  // RGB replaced by a different buffer mid-frame: presence is
  // unchanged, identity is not — the frame must not split across two
  // caller buffers.
  let mut rgb_a = vec![0u8; OUT * OUT * 3];
  let mut rgb_b = vec![0u8; OUT * OUT * 3];
  {
    let mut sink = downscaled().with_rgb(&mut rgb_a).unwrap();
    sink.begin_frame(SRC as u32, SRC as u32).unwrap();
    for row in 0..2 {
      sink
        .process(Yuv420pRow::for_tests(
          &y,
          &u,
          &v,
          row,
          KernelMatrix::Bt601,
          true,
        ))
        .unwrap();
    }
    sink.set_rgb(&mut rgb_b).unwrap();
    let err = sink
      .process(Yuv420pRow::for_tests(
        &y,
        &u,
        &v,
        2,
        KernelMatrix::Bt601,
        true,
      ))
      .unwrap_err();
    assert!(matches!(err, MixedSinkerError::ResampleOutputsChanged(_)));
  }
  assert!(
    rgb_b.iter().all(|&x| x == 0),
    "replacement buffer must stay untouched"
  );
  assert!(
    rgb_a[..OUT * 3].iter().all(|&x| x != 0),
    "original buffer keeps its emitted row"
  );

  // HSV planes replaced mid-frame: same contract.
  let mut h_a = vec![0u8; OUT * OUT];
  let mut s_a = vec![0u8; OUT * OUT];
  let mut v_a = vec![0u8; OUT * OUT];
  let mut h_b = vec![0u8; OUT * OUT];
  let mut s_b = vec![0u8; OUT * OUT];
  let mut v_b = vec![0u8; OUT * OUT];
  {
    let mut sink = downscaled().with_hsv(&mut h_a, &mut s_a, &mut v_a).unwrap();
    sink.begin_frame(SRC as u32, SRC as u32).unwrap();
    for row in 0..2 {
      sink
        .process(Yuv420pRow::for_tests(
          &y,
          &u,
          &v,
          row,
          KernelMatrix::Bt601,
          true,
        ))
        .unwrap();
    }
    sink.set_hsv(&mut h_b, &mut s_b, &mut v_b).unwrap();
    let err = sink
      .process(Yuv420pRow::for_tests(
        &y,
        &u,
        &v,
        2,
        KernelMatrix::Bt601,
        true,
      ))
      .unwrap_err();
    assert!(matches!(err, MixedSinkerError::ResampleOutputsChanged(_)));
  }
  assert!(v_b.iter().all(|&x| x == 0));
}

/// Frame with luma gradient and spatially varying chroma — exercises
/// real conversion math on both tiers.
fn textured_frame_planes() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  // Values stay interior to the limited-range gamut: near clamp
  // boundaries the two tiers diverge by more than rounding
  // (avg-then-clamp vs clamp-then-avg) — the documented out-of-gamut
  // caveat, not a defect.
  let y: Vec<u8> = (0..64u8).map(|i| 60 + i * 2).collect();
  let u: Vec<u8> = (0..16u8).map(|i| 118 + i).collect();
  let v: Vec<u8> = (0..16u8).map(|i| 120 + i).collect();
  (y, u, v)
}

/// `(rgb, rgba, luma, hsv_h, hsv_v)` planes of one downscale run.
type DownscaleOutputs = (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>);

fn run_downscale(out_w: usize, out_h: usize, native: bool) -> DownscaleOutputs {
  let (yp, up, vp) = textured_frame_planes();
  let src = Yuv420pFrame::new(&yp, &up, &vp, 8, 8, 8, 4, 4);
  let n = out_w * out_h;
  let mut rgb = vec![0u8; n * 3];
  let mut rgba = vec![0u8; n * 4];
  let mut luma = vec![0u8; n];
  let mut h = vec![0u8; n];
  let mut s_ = vec![0u8; n];
  let mut v_ = vec![0u8; n];
  {
    let mut sink = MixedSinker::<Yuv420p, AreaResampler>::with_resampler(
      SRC,
      SRC,
      AreaResampler::to(out_w, out_h),
    )
    .unwrap()
    .with_native(native)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap()
    .with_luma(&mut luma)
    .unwrap()
    .with_hsv(&mut h, &mut s_, &mut v_)
    .unwrap();
    yuv420p_to(&src, false, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
  }
  (rgb, rgba, luma, h, v_)
}

#[test]
fn native_and_row_stage_luma_bit_identical() {
  // Both tiers bin the SAME Y plane with the SAME spans and exact
  // arithmetic: luma must match bit-for-bit, every geometry.
  for (ow, oh) in [(4, 4), (3, 3), (6, 6), (4, 3)] {
    let native = run_downscale(ow, oh, true);
    let row_stage = run_downscale(ow, oh, false);
    assert_eq!(native.2, row_stage.2, "luma {ow}x{oh}");
  }
}

#[test]
fn native_and_row_stage_color_within_tolerance() {
  // Native converts AFTER binning (output res); row-stage converts
  // BEFORE (source res). The orders differ only by per-pixel rounding
  // and clamping inside the affine conversion: bounded divergence on
  // in-gamut content. 6x6 exercises the chroma-upsample direction
  // (4 -> 6 on the chroma grid).
  for (ow, oh) in [(4, 4), (3, 3), (6, 6), (4, 3)] {
    let native = run_downscale(ow, oh, true);
    let row_stage = run_downscale(ow, oh, false);
    for (name, a, b) in [
      ("rgb", &native.0, &row_stage.0),
      ("rgba", &native.1, &row_stage.1),
      ("hsv-v", &native.4, &row_stage.4),
    ] {
      for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        // Bound: each binned plane rounds within half a step; the
        // affine conversion amplifies by at most ~1.8 per
        // coefficient, plus the final per-channel round.
        let d = x.abs_diff(*y);
        assert!(
          d <= 3,
          "{name} {ow}x{oh} idx {i}: native {x} vs row-stage {y}"
        );
      }
    }
  }
}

#[test]
fn native_solid_frame_exact_all_outputs() {
  // Constant planes: binning is exact on both grids, so the native
  // tier must reproduce the full-res conversion exactly.
  let yp = vec![120u8; 64];
  let up = vec![90u8; 16];
  let vp = vec![170u8; 16];
  let src = Yuv420pFrame::new(&yp, &up, &vp, 8, 8, 8, 4, 4);

  let mut full_rgb = vec![0u8; SRC * SRC * 3];
  let mut full = MixedSinker::<Yuv420p>::new(SRC, SRC)
    .with_rgb(&mut full_rgb)
    .unwrap();
  yuv420p_to(&src, false, full.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();

  let mut rgb = vec![0u8; OUT * OUT * 3];
  let mut luma = vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Yuv420p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_native(true)
        .with_rgb(&mut rgb)
        .unwrap()
        .with_luma(&mut luma)
        .unwrap();
    yuv420p_to(&src, false, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
  }
  for px in rgb.chunks_exact(3) {
    assert_eq!(
      (px[0], px[1], px[2]),
      (full_rgb[0], full_rgb[1], full_rgb[2])
    );
  }
  assert!(luma.iter().all(|&l| l == 120));
}

#[test]
fn native_misaligned_chroma_upsample_matches_row_stage() {
  // 6-wide source: chroma width 3 against output width 5 is a
  // misaligned upsample ratio in (1, 2) — a chroma cell sits under
  // three spans, the densest coverage the chroma grid can produce.
  let yp: Vec<u8> = (0..36u8).map(|i| 60 + i * 2).collect();
  let up: Vec<u8> = (0..9u8).map(|i| 118 + i).collect();
  let vp: Vec<u8> = (0..9u8).map(|i| 120 + i).collect();
  let src = Yuv420pFrame::new(&yp, &up, &vp, 6, 6, 6, 3, 3);

  let run = |native: bool| -> Vec<u8> {
    let mut rgb = vec![0u8; 5 * 5 * 3];
    let mut sink =
      MixedSinker::<Yuv420p, AreaResampler>::with_resampler(6, 6, AreaResampler::to(5, 5))
        .unwrap()
        .with_native(native)
        .with_rgb(&mut rgb)
        .unwrap();
    yuv420p_to(&src, false, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
    rgb
  };
  let native = run(true);
  let row_stage = run(false);
  for (i, (a, b)) in native.iter().zip(row_stage.iter()).enumerate() {
    assert!(a.abs_diff(*b) <= 3, "idx {i}: native {a} vs row-stage {b}");
  }
}

#[test]
fn native_odd_height_color_matches_row_stage() {
  // Odd source heights: the final chroma row covers ONE luma row, and
  // the native tier must weight it accordingly. Row-stage is ground
  // truth here (it bins converted rows over the uniform luma grid).
  let h_src = 9usize;
  let yp: Vec<u8> = (0..(8 * h_src) as u32)
    .map(|i| 60 + (i % 64) as u8 * 2)
    .collect();
  let up: Vec<u8> = (0..(4 * 5)).map(|i| 118 + (i % 16) as u8).collect();
  let vp: Vec<u8> = (0..(4 * 5)).map(|i| 120 + (i % 16) as u8).collect();
  let src = Yuv420pFrame::new(&yp, &up, &vp, 8, h_src as u32, 8, 4, 4);

  let run = |native: bool| -> Vec<u8> {
    let mut rgb = vec![0u8; 2 * 3 * 3];
    let mut sink =
      MixedSinker::<Yuv420p, AreaResampler>::with_resampler(8, h_src, AreaResampler::to(2, 3))
        .unwrap()
        .with_native(native)
        .with_rgb(&mut rgb)
        .unwrap();
    yuv420p_to(&src, false, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
    rgb
  };
  let native = run(true);
  let row_stage = run(false);
  for (i, (a, b)) in native.iter().zip(row_stage.iter()).enumerate() {
    assert!(
      a.abs_diff(*b) <= 3,
      "odd-height idx {i}: native {a} vs row-stage {b}"
    );
  }
}

#[test]
fn native_saturated_divergence_is_characterized() {
  // The two tiers differ in conversion ORDER: native averages in the
  // source (YUV) domain and clamps once after the mean; row-stage
  // clamps per pixel before averaging. On in-gamut content that is
  // rounding noise; on out-of-gamut content the divergence is
  // unbounded in principle — these two measured cases pin a mild and
  // a crafted example so the docs cannot understate it. Luma stays
  // bit-identical everywhere (both tiers bin the same Y plane).

  // Mild: alternating super-black/super-white extreme-chroma
  // checkerboard, 8x8 -> 4x4 Bt709 limited. Measured max 34.
  let y: Vec<u8> = (0..64u32)
    .map(|i| if i % 2 == 0 { 2 } else { 250 })
    .collect();
  let u: Vec<u8> = (0..16u32)
    .map(|i| if i % 2 == 0 { 10 } else { 245 })
    .collect();
  let v: Vec<u8> = (0..16u32)
    .map(|i| if i % 2 == 0 { 240 } else { 12 })
    .collect();
  let src = Yuv420pFrame::new(&y, &u, &v, 8, 8, 8, 4, 4);
  let run = |native: bool| {
    let mut rgb = vec![0u8; OUT * OUT * 3];
    let mut luma = vec![0u8; OUT * OUT];
    let mut sink =
      MixedSinker::<Yuv420p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_native(native)
        .with_rgb(&mut rgb)
        .unwrap()
        .with_luma(&mut luma)
        .unwrap();
    yuv420p_to(&src, false, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
    (rgb, luma)
  };
  let (n_rgb, n_luma) = run(true);
  let (r_rgb, r_luma) = run(false);
  assert_eq!(n_luma, r_luma, "luma must stay bit-identical");
  let max = n_rgb
    .iter()
    .zip(r_rgb.iter())
    .map(|(a, b)| a.abs_diff(*b))
    .max()
    .unwrap();
  assert!((4..=48).contains(&max), "mild case drifted: {max}");

  // Crafted: 4x4 -> 1x1, Bt2020Ncl limited, planes built to push the
  // mean far from the per-pixel clamps. Measured max 117.
  let y: Vec<u8> = vec![
    64, 32, 245, 240, 2, 10, 128, 224, 235, 240, 245, 250, 245, 255, 245, 224,
  ];
  let u: Vec<u8> = vec![128, 250, 16, 240];
  let v: Vec<u8> = vec![192, 10, 64, 10];
  let src = Yuv420pFrame::new(&y, &u, &v, 4, 4, 4, 2, 2);
  let run = |native: bool| {
    let mut rgb = vec![0u8; 3];
    let mut sink =
      MixedSinker::<Yuv420p, AreaResampler>::with_resampler(4, 4, AreaResampler::to(1, 1))
        .unwrap()
        .with_native(native)
        .with_rgb(&mut rgb)
        .unwrap();
    yuv420p_to(&src, false, sink.set_kernel_matrix(KernelMatrix::Bt2020Ncl)).unwrap();
    rgb
  };
  let n = run(true);
  let r = run(false);
  let max = n
    .iter()
    .zip(r.iter())
    .map(|(a, b)| a.abs_diff(*b))
    .max()
    .unwrap();
  assert!((64..=140).contains(&max), "crafted case drifted: {max}");
}

#[test]
fn native_join_upgrades_when_color_attaches_next_frame() {
  // Frame 1 runs luma-only (native join created WITHOUT its chroma
  // half); RGB+HSV attach before frame 2. The join must rebuild with
  // chroma for the new frame — not silently skip the color outputs.
  let yp = vec![120u8; 64];
  let up = vec![90u8; 16];
  let vp = vec![170u8; 16];
  let src = Yuv420pFrame::new(&yp, &up, &vp, 8, 8, 8, 4, 4);

  let mut full_rgb = vec![0u8; SRC * SRC * 3];
  let mut full = MixedSinker::<Yuv420p>::new(SRC, SRC)
    .with_rgb(&mut full_rgb)
    .unwrap();
  yuv420p_to(&src, false, full.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();

  let mut luma = vec![0u8; OUT * OUT];
  let mut rgb = vec![0u8; OUT * OUT * 3];
  let mut h = vec![0u8; OUT * OUT];
  let mut s_ = vec![0u8; OUT * OUT];
  let mut v_ = vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Yuv420p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_luma(&mut luma)
        .unwrap();
    yuv420p_to(&src, false, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();

    sink.set_rgb(&mut rgb).unwrap();
    sink.set_hsv(&mut h, &mut s_, &mut v_).unwrap();
    yuv420p_to(&src, false, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
  }
  assert!(luma.iter().all(|&l| l == 120));
  for px in rgb.chunks_exact(3) {
    assert_eq!(
      (px[0], px[1], px[2]),
      (full_rgb[0], full_rgb[1], full_rgb[2]),
      "color silently skipped after attaching mid-stream"
    );
  }
  assert!(v_.iter().all(|&x| x != 0), "hsv V plane untouched");
}

#[test]
fn identity_area_full_pipeline_matches_new_sink() {
  let (yp, up, vp) = gradient_frame_planes();
  let src = Yuv420pFrame::new(&yp, &up, &vp, 8, 8, 8, 4, 4);

  let mut direct = vec![0u8; SRC * SRC * 3];
  let mut sink = MixedSinker::<Yuv420p>::new(SRC, SRC)
    .with_rgb(&mut direct)
    .unwrap();
  yuv420p_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();

  let mut via_area = vec![0u8; SRC * SRC * 3];
  let mut sink =
    MixedSinker::<Yuv420p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(SRC, SRC))
      .unwrap()
      .with_rgb(&mut via_area)
      .unwrap();
  yuv420p_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();

  assert_eq!(direct, via_area);
}

// ---- frozen native-vs-row-stage route (issue #186) ----------------------
//
// The 8-bit planar 4:2:0 native (`yuv420p_process_native`) and row-stage
// (`yuv420p_process_resampled`) tiers carry independent, in-order,
// once-only stream state, so a mid-frame `set_native` flip must reject as
// the deterministic `NativeRouteChanged` rather than silently mix the two
// streams. The route is CHECKED before and frozen after dispatch, both
// gated on whether the call bears output — mirroring the P0xx template.

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn native_to_rowstage_route_flip_mid_frame_rejected() {
  use crate::source::Yuv420pRow;
  let y = [50u8; SRC];
  let u = [128u8; SRC / 2];
  let v = [128u8; SRC / 2];
  let mut luma = vec![0u8; OUT * OUT];
  let mut sink = downscaled().with_native(true).with_luma(&mut luma).unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  // Row 0 freezes the route = native.
  sink
    .process(Yuv420pRow::for_tests(
      &y,
      &u,
      &v,
      0,
      KernelMatrix::Bt601,
      true,
    ))
    .expect("native row 0 freezes the route and succeeds");
  // Flip to the row-stage tier and feed the next in-sequence row.
  sink.set_native(false);
  let err = sink
    .process(Yuv420pRow::for_tests(
      &y,
      &u,
      &v,
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
fn rowstage_to_native_route_flip_mid_frame_rejected() {
  use crate::source::Yuv420pRow;
  let y = [50u8; SRC];
  let u = [128u8; SRC / 2];
  let v = [128u8; SRC / 2];
  let mut luma = vec![0u8; OUT * OUT];
  let mut sink = downscaled()
    .with_native(false)
    .with_luma(&mut luma)
    .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  // Row 0 freezes the route = row-stage.
  sink
    .process(Yuv420pRow::for_tests(
      &y,
      &u,
      &v,
      0,
      KernelMatrix::Bt601,
      true,
    ))
    .expect("row-stage row 0 freezes the route and succeeds");
  // Flip to the native tier and feed the next in-sequence row.
  sink.set_native(true);
  let err = sink
    .process(Yuv420pRow::for_tests(
      &y,
      &u,
      &v,
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
fn route_constant_succeeds_and_resets_across_frames() {
  // A frame whose route stays constant runs to completion, and the
  // per-frame reset (in `begin_frame`) lets the NEXT frame pick the OTHER
  // tier — the route guard is reset, not sticky across frames.
  let (yp, up, vp) = gradient_frame_planes();
  let src = Yuv420pFrame::new(&yp, &up, &vp, 8, 8, 8, 4, 4);
  let mut luma = vec![0u8; OUT * OUT];
  let mut sink = downscaled().with_native(true).with_luma(&mut luma).unwrap();
  // Frame 1: native, route constant across every row — no false rejection.
  yuv420p_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
  // Frame 2: flip to row-stage for the WHOLE frame. `begin_frame` (driven
  // by the walker) cleared the frozen route, so this is allowed.
  sink.set_native(false);
  yuv420p_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601))
    .expect("a new frame may pick the other tier; the route reset per frame");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn no_output_call_after_frozen_route_is_a_noop() {
  use crate::source::Yuv420pRow;
  // A NO-OUTPUT call arriving AFTER an output-bearing row froze the route
  // must be a TRUE no-op — route-invisible — even when its `set_native` is
  // FLIPPED relative to the frozen route: it must return `Ok` (not
  // `NativeRouteChanged`) and leave the frozen route untouched. The route
  // CHECK and the SET both gate on `need_output`. With the CHECK's
  // `need_output` gate removed the ungated CHECK sees `frozen(native) !=
  // row-stage` and wrongly rejects.
  //
  // There is no public API to DETACH an output (buffers only transition
  // `None -> Some`), so we build the faithful equivalent by setting the
  // per-sink `frozen_native_route` field directly to the value an accepted
  // output-bearing native first row stores (`Some(true)` = native), then
  // issuing the flipped no-output call — the same white-box reach the
  // atomicity tests use.
  let y = [50u8; SRC];
  let u = [128u8; SRC / 2];
  let v = [128u8; SRC / 2];
  let mut sink = downscaled().with_native(true);
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  // Freeze the route to native exactly as an accepted output-bearing native
  // first row does (the SET site stores `Some(true)`).
  sink.frozen_native_route = Some(true);
  // A NO-OUTPUT row (no outputs attached -> `need_output` false) with the
  // route FLIPPED to row-stage. Gated on `need_output`, the CHECK is
  // skipped, so this is a true no-op (Ok); a no-output call is
  // route-invisible.
  sink.set_native(false);
  sink
    .process(Yuv420pRow::for_tests(
      &y,
      &u,
      &v,
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
  // Proof the route is STILL frozen to native and the no-output call
  // consumed no stream state: an OUTPUT-bearing row 0 under the original
  // native route is in-sequence and succeeds...
  let mut luma = vec![0u8; OUT * OUT];
  sink.set_native(true);
  sink.set_luma(&mut luma).unwrap();
  sink
    .process(Yuv420pRow::for_tests(
      &y,
      &u,
      &v,
      0,
      KernelMatrix::Bt601,
      true,
    ))
    .expect("an output-bearing row under the original native route succeeds");
  // ...while an OUTPUT-bearing flip to the OTHER route now rejects,
  // confirming the frozen route stayed native.
  sink.set_native(false);
  let err = sink
    .process(Yuv420pRow::for_tests(
      &y,
      &u,
      &v,
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
