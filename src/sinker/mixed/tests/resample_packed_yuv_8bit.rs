//! Fused-downscale coverage for the packed YUV 4:2:2 (8-bit) sources —
//! `Yuyv422` (YUY2), `Uyvy422` (UYVY), `Yvyu422` (YVYU).
//!
//! These route through the packed-YUV dual-stream resample, mirroring
//! the planar YUV row-stage tier: **luma / luma_u16 area-resample the
//! de-interleaved Y bytes** (the YUV luma contract — luma is taken from
//! Y, *not* re-derived from converted RGB), while RGB / RGBA / HSV bin a
//! converted source-width RGB row (the format's own `*_to_rgb_row`
//! kernel does the chroma de-interleave + 4:2:2 upsample in-register).
//!
//! Oracles, built without the `rgb` feature (these tests run under
//! `yuv-packed` alone):
//! - **RGB**: the direct full-res conversion of the packed frame, area-
//!   binned by a round-half-up 2x2 block mean (the engine's area-bin at
//!   the integer 2:1 ratio). The RGB a resampled frame bins is
//!   byte-identical to the direct path's RGB output, so binning it
//!   reproduces the resampled RGB exactly.
//! - **luma**: the area-downscaled Y plane (a 2x2 block mean of the
//!   de-interleaved Y bytes) — pinned under saturated chroma, where
//!   RGB-derived luma would diverge.

use crate::{
  KernelMatrix, PixelSink,
  frame::{Uyvy422Frame, Yuyv422Frame, Yvyu422Frame},
  resample::{AreaResampler, ResampleError},
  sinker::{MixedSinker, MixedSinkerError},
  source::{
    Uyvy422, Uyvy422Row, Yuyv422, Yuyv422Row, Yvyu422, Yvyu422Row, uyvy422_to, yuyv422_to,
    yvyu422_to,
  },
};

use super::force_row_stage;

const SRC: usize = 8;
const OUT: usize = 4;
const M: KernelMatrix = KernelMatrix::Bt601;
const FR: bool = true;

/// Exact 2x2-block area mean (round-half-up) of an `SRC`-grid plane to
/// the `OUT` grid — the integer-ratio (2:1) area-downscale reference for
/// a single channel.
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

/// Exact 2x2-block area mean (round-half-up) of an `SRC`-grid **RGB**
/// (3-channel) plane to the `OUT` grid — the colour reference.
fn block_mean_2x2_rgb(rgb: &[u8]) -> Vec<u8> {
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

/// Per-pixel `(Y, U, V)` ramp shared by the three formats so the packed
/// builders below differ only in byte permutation. Chroma is sampled at
/// the even column of each 2-pixel pair (4:2:2). Returns the Y plane
/// (source-grid) plus a `(u, v)` pair indexed per chroma column.
fn yuv_ramp() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let mut y = vec![0u8; SRC * SRC];
  let mut u = vec![0u8; (SRC / 2) * SRC];
  let mut v = vec![0u8; (SRC / 2) * SRC];
  for (i, p) in y.iter_mut().enumerate() {
    *p = 40 + (i as u8) * 2;
  }
  for row in 0..SRC {
    for cx in 0..SRC / 2 {
      u[row * (SRC / 2) + cx] = 70 + (cx as u8) * 5 + (row as u8);
      v[row * (SRC / 2) + cx] = 200 - (cx as u8) * 4 - (row as u8);
    }
  }
  (y, u, v)
}

/// Builds a YUYV422 packed plane (`Y0, U0, Y1, V0` per pair) from the
/// shared ramp.
fn yuyv_from(y: &[u8], u: &[u8], v: &[u8]) -> Vec<u8> {
  let mut buf = vec![0u8; 2 * SRC * SRC];
  for row in 0..SRC {
    for cx in 0..SRC / 2 {
      let base = row * 2 * SRC + cx * 4;
      buf[base] = y[row * SRC + cx * 2];
      buf[base + 1] = u[row * (SRC / 2) + cx];
      buf[base + 2] = y[row * SRC + cx * 2 + 1];
      buf[base + 3] = v[row * (SRC / 2) + cx];
    }
  }
  buf
}

/// Builds a UYVY422 packed plane (`U0, Y0, V0, Y1` per pair).
fn uyvy_from(y: &[u8], u: &[u8], v: &[u8]) -> Vec<u8> {
  let mut buf = vec![0u8; 2 * SRC * SRC];
  for row in 0..SRC {
    for cx in 0..SRC / 2 {
      let base = row * 2 * SRC + cx * 4;
      buf[base] = u[row * (SRC / 2) + cx];
      buf[base + 1] = y[row * SRC + cx * 2];
      buf[base + 2] = v[row * (SRC / 2) + cx];
      buf[base + 3] = y[row * SRC + cx * 2 + 1];
    }
  }
  buf
}

/// Builds a YVYU422 packed plane (`Y0, V0, Y1, U0` per pair).
fn yvyu_from(y: &[u8], u: &[u8], v: &[u8]) -> Vec<u8> {
  let mut buf = vec![0u8; 2 * SRC * SRC];
  for row in 0..SRC {
    for cx in 0..SRC / 2 {
      let base = row * 2 * SRC + cx * 4;
      buf[base] = y[row * SRC + cx * 2];
      buf[base + 1] = v[row * (SRC / 2) + cx];
      buf[base + 2] = y[row * SRC + cx * 2 + 1];
      buf[base + 3] = u[row * (SRC / 2) + cx];
    }
  }
  buf
}

fn yuyv_frame(buf: &[u8]) -> Yuyv422Frame<'_> {
  Yuyv422Frame::new(buf, SRC as u32, SRC as u32, (2 * SRC) as u32)
}
fn uyvy_frame(buf: &[u8]) -> Uyvy422Frame<'_> {
  Uyvy422Frame::new(buf, SRC as u32, SRC as u32, (2 * SRC) as u32)
}
fn yvyu_frame(buf: &[u8]) -> Yvyu422Frame<'_> {
  Yvyu422Frame::new(buf, SRC as u32, SRC as u32, (2 * SRC) as u32)
}

/// Saturated-chroma packed frame builders (constant Y, extreme U/V) —
/// the case where RGB-derived luma would diverge from the Y plane.
fn yuyv_saturated() -> Vec<u8> {
  let y = vec![16u8; SRC * SRC];
  let u = vec![240u8; (SRC / 2) * SRC];
  let v = vec![16u8; (SRC / 2) * SRC];
  yuyv_from(&y, &u, &v)
}
fn uyvy_saturated() -> Vec<u8> {
  let y = vec![16u8; SRC * SRC];
  let u = vec![240u8; (SRC / 2) * SRC];
  let v = vec![16u8; (SRC / 2) * SRC];
  uyvy_from(&y, &u, &v)
}
fn yvyu_saturated() -> Vec<u8> {
  let y = vec![16u8; SRC * SRC];
  let u = vec![240u8; (SRC / 2) * SRC];
  let v = vec![16u8; (SRC / 2) * SRC];
  yvyu_from(&y, &u, &v)
}

// A per-format macro keeps the three near-identical suites in lockstep
// while still naming each test after its format (so a failure points at
// the exact byte order — UYVY vs YUYV vs YVYU). `$drv` is the source
// walker, `$frame` the frame builder, `$build` the packed-plane builder,
// and `$sat` the saturated-chroma plane.
macro_rules! packed_yuv_resample_suite {
  (
    $fmt:ident, $row:ident, $drv:ident, $mk_frame:ident, $build:ident, $sat:ident,
    rgb: $rgb_name:ident,
    luma: $luma_name:ident,
    sat: $sat_name:ident,
    all: $all_name:ident,
    identity: $id_name:ident,
    no_op: $noop_name:ident,
    reset: $reset_name:ident,
    seq_first: $seq_first_name:ident,
    seq_mid: $seq_mid_name:ident,
    freeze: $freeze_name:ident,
    poison: $poison_name:ident,
  ) => {
    #[test]
    #[cfg_attr(
      miri,
      ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
    )]
    fn $rgb_name() {
      let (y, u, v) = yuv_ramp();
      let packed = $build(&y, &u, &v);
      // Direct full-res RGB conversion of the packed frame.
      let mut full_rgb = vec![0u8; SRC * SRC * 3];
      {
        let mut sink = MixedSinker::<$fmt>::new(SRC, SRC)
          .with_rgb(&mut full_rgb)
          .unwrap();
        $drv(&$mk_frame(&packed), FR, M, &mut sink).unwrap();
      }
      // Resampled RGB (pinned to the row-stage tier — see `force_row_stage`).
      let mut rgb = vec![0u8; OUT * OUT * 3];
      {
        let mut sink = force_row_stage(
          MixedSinker::<$fmt, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
            .unwrap(),
        )
        .with_rgb(&mut rgb)
        .unwrap();
        $drv(&$mk_frame(&packed), FR, M, &mut sink).unwrap();
      }
      assert_eq!(
        rgb,
        block_mean_2x2_rgb(&full_rgb),
        "rgb: row-stage == area-bin of the direct conversion"
      );
    }

    #[test]
    #[cfg_attr(
      miri,
      ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
    )]
    fn $luma_name() {
      let (y, u, v) = yuv_ramp();
      let packed = $build(&y, &u, &v);
      let (mut luma, mut luma_u16) = (vec![0u8; OUT * OUT], vec![0u16; OUT * OUT]);
      {
        let mut sink =
          MixedSinker::<$fmt, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
            .unwrap()
            .with_luma(&mut luma)
            .unwrap()
            .with_luma_u16(&mut luma_u16)
            .unwrap();
        $drv(&$mk_frame(&packed), FR, M, &mut sink).unwrap();
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
    fn $sat_name() {
      // Saturated chroma: Y is constant 16, so the area-downscaled Y is
      // all 16. RGB-derived luma would clamp away from 16; luma-from-Y
      // must stay exactly 16.
      let packed = $sat();
      let mut luma = vec![0u8; OUT * OUT];
      {
        let mut sink =
          MixedSinker::<$fmt, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
            .unwrap()
            .with_luma(&mut luma)
            .unwrap();
        $drv(&$mk_frame(&packed), FR, M, &mut sink).unwrap();
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
    fn $all_name() {
      // Every output attached at once: each must match its own oracle,
      // proving the two streams (luma-from-Y, colour-from-RGB) coexist.
      let (y, u, v) = yuv_ramp();
      let packed = $build(&y, &u, &v);
      let mut full_rgb = vec![0u8; SRC * SRC * 3];
      {
        let mut sink = MixedSinker::<$fmt>::new(SRC, SRC)
          .with_rgb(&mut full_rgb)
          .unwrap();
        $drv(&$mk_frame(&packed), FR, M, &mut sink).unwrap();
      }
      let rgb_ref = block_mean_2x2_rgb(&full_rgb);

      let mut rgb = vec![0u8; OUT * OUT * 3];
      let mut rgba = vec![0u8; OUT * OUT * 4];
      let mut luma = vec![0u8; OUT * OUT];
      let mut luma_u16 = vec![0u16; OUT * OUT];
      let mut hh = vec![0u8; OUT * OUT];
      let mut ss = vec![0u8; OUT * OUT];
      let mut vv = vec![0u8; OUT * OUT];
      {
        let mut sink = force_row_stage(
          MixedSinker::<$fmt, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
            .unwrap(),
        )
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
        $drv(&$mk_frame(&packed), FR, M, &mut sink).unwrap();
      }
      assert_eq!(rgb, rgb_ref, "all-outputs rgb");
      // RGBA is the RGB row with a 0xFF alpha pad.
      for (px, rgb_px) in rgba.chunks_exact(4).zip(rgb_ref.chunks_exact(3)) {
        assert_eq!(&px[..3], rgb_px, "all-outputs rgba colour");
        assert_eq!(px[3], 0xFF, "all-outputs rgba alpha");
      }
      // luma from the area-downscaled Y plane.
      let y_ref = block_mean_2x2(&y);
      assert_eq!(luma, y_ref, "all-outputs luma");
      let y_ref_u16: Vec<u16> = y_ref.iter().map(|&b| b as u16).collect();
      assert_eq!(luma_u16, y_ref_u16, "all-outputs luma_u16");
      // HSV value channel equals the binned-RGB-derived HSV; recompute
      // from the colour oracle and compare against the binned output.
      let mut hh_ref = vec![0u8; OUT * OUT];
      let mut ss_ref = vec![0u8; OUT * OUT];
      let mut vv_ref = vec![0u8; OUT * OUT];
      crate::row::rgb_to_hsv_row(
        &rgb_ref,
        &mut hh_ref,
        &mut ss_ref,
        &mut vv_ref,
        OUT * OUT,
        false,
      );
      assert_eq!(hh, hh_ref, "all-outputs hsv H");
      assert_eq!(ss, ss_ref, "all-outputs hsv S");
      assert_eq!(vv, vv_ref, "all-outputs hsv V");
    }

    #[test]
    #[cfg_attr(
      miri,
      ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
    )]
    fn $id_name() {
      // Identity plan must reproduce the direct sink byte for byte.
      let (y, u, v) = yuv_ramp();
      let packed = $build(&y, &u, &v);
      let mut direct = vec![0u8; SRC * SRC * 3];
      {
        let mut sink = MixedSinker::<$fmt>::new(SRC, SRC)
          .with_rgb(&mut direct)
          .unwrap();
        $drv(&$mk_frame(&packed), FR, M, &mut sink).unwrap();
      }
      let mut via_area = vec![0u8; SRC * SRC * 3];
      {
        let mut sink = force_row_stage(
          MixedSinker::<$fmt, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(SRC, SRC))
            .unwrap(),
        )
        .with_rgb(&mut via_area)
        .unwrap();
        $drv(&$mk_frame(&packed), FR, M, &mut sink).unwrap();
      }
      assert_eq!(direct, via_area, "identity plan must match the direct sink");
    }

    #[test]
    fn $noop_name() {
      // No outputs attached: a legal no-op that allocates nothing.
      let (y, u, v) = yuv_ramp();
      let packed = $build(&y, &u, &v);
      let mut sink =
        MixedSinker::<$fmt, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
          .unwrap();
      $drv(&$mk_frame(&packed), FR, M, &mut sink).unwrap();
      assert!(
        !sink.luma_stream_allocated(),
        "no-output sink allocated a luma stream"
      );
      assert!(
        !sink.rgb_stream_allocated(),
        "no-output sink allocated an rgb stream"
      );
    }

    #[test]
    #[cfg_attr(
      miri,
      ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
    )]
    fn $reset_name() {
      // A reused sink must reset both streams each frame; without the
      // reset, frame 2's row 0 is rejected as out-of-sequence.
      let (y1, u, v) = yuv_ramp();
      let mut y2 = y1.clone();
      for p in y2.iter_mut() {
        *p = 255 - *p;
      }
      let p1 = $build(&y1, &u, &v);
      let p2 = $build(&y2, &u, &v);
      let mut luma = vec![0u8; OUT * OUT];
      let mut rgb = vec![0u8; OUT * OUT * 3];
      {
        let mut sink =
          MixedSinker::<$fmt, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
            .unwrap()
            .with_luma(&mut luma)
            .unwrap()
            .with_rgb(&mut rgb)
            .unwrap();
        $drv(&$mk_frame(&p1), FR, M, &mut sink).unwrap();
        $drv(&$mk_frame(&p2), FR, M, &mut sink).unwrap();
      }
      assert_eq!(
        luma,
        block_mean_2x2(&y2),
        "frame 2 luma must area-downscale frame 2's Y"
      );
    }

    #[test]
    fn $seq_first_name() {
      // Out-of-sequence first row: rejected before any allocation.
      let (y, u, v) = yuv_ramp();
      let packed = $build(&y, &u, &v);
      let row3 = &packed[3 * 2 * SRC..4 * 2 * SRC];
      let mut luma = vec![0u8; OUT * OUT];
      let mut rgb = vec![0u8; OUT * OUT * 3];
      let mut sink =
        MixedSinker::<$fmt, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
          .unwrap()
          .with_luma(&mut luma)
          .unwrap()
          .with_rgb(&mut rgb)
          .unwrap();
      sink.begin_frame(SRC as u32, SRC as u32).unwrap();
      let err = sink.process($row::new(row3, 3, M, FR)).unwrap_err();
      assert!(
        matches!(
          err,
          MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
        ),
        "expected OutOfSequenceRow, got {err:?}"
      );
      assert!(
        !sink.luma_stream_allocated() && !sink.rgb_stream_allocated(),
        "stream allocated for a rejected row"
      );
      assert_eq!(sink.luma_scratch_capacity(), 0, "Y scratch grown on reject");
      assert_eq!(
        sink.rgb_scratch_capacity(),
        0,
        "RGB scratch grown on reject"
      );
      assert!(
        luma.iter().all(|&b| b == 0) && rgb.iter().all(|&b| b == 0),
        "rejected row mutated output"
      );
    }

    #[test]
    #[cfg_attr(
      miri,
      ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
    )]
    fn $seq_mid_name() {
      // Skipping a row mid-frame is out of sequence.
      let (y, u, v) = yuv_ramp();
      let packed = $build(&y, &u, &v);
      let mut luma = vec![0u8; OUT * OUT];
      let mut sink =
        MixedSinker::<$fmt, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
          .unwrap()
          .with_luma(&mut luma)
          .unwrap();
      sink.begin_frame(SRC as u32, SRC as u32).unwrap();
      sink
        .process($row::new(&packed[..2 * SRC], 0, M, FR))
        .unwrap();
      let err = sink
        .process($row::new(&packed[2 * 2 * SRC..3 * 2 * SRC], 2, M, FR))
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
    #[cfg_attr(
      miri,
      ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
    )]
    fn $freeze_name() {
      // Attaching a new output mid-frame trips the frozen-output check.
      let (y, u, v) = yuv_ramp();
      let packed = $build(&y, &u, &v);
      let mut rgb = vec![0u8; OUT * OUT * 3];
      let mut luma = vec![0u8; OUT * OUT];
      let mut sink =
        MixedSinker::<$fmt, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
          .unwrap()
          .with_rgb(&mut rgb)
          .unwrap();
      sink.begin_frame(SRC as u32, SRC as u32).unwrap();
      sink
        .process($row::new(&packed[..2 * SRC], 0, M, FR))
        .unwrap();
      sink.set_luma(&mut luma).unwrap();
      let err = sink
        .process($row::new(&packed[2 * SRC..2 * 2 * SRC], 1, M, FR))
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

    #[test]
    #[cfg_attr(
      miri,
      ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
    )]
    fn $poison_name() {
      // A rejected out-of-sequence FIRST row must store no frozen-output
      // snapshot, so retrying row 0 after attaching a NEW output succeeds
      // instead of tripping ResampleOutputsChanged.
      let (y, u, v) = yuv_ramp();
      let packed = $build(&y, &u, &v);
      let mut luma = vec![0u8; OUT * OUT];
      let mut sink =
        MixedSinker::<$fmt, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
          .unwrap()
          .with_luma(&mut luma)
          .unwrap();
      sink.begin_frame(SRC as u32, SRC as u32).unwrap();
      let err = sink
        .process($row::new(&packed[3 * 2 * SRC..4 * 2 * SRC], 3, M, FR))
        .unwrap_err();
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
        .process($row::new(&packed[..2 * SRC], 0, M, FR))
        .expect("row 0 must succeed after a rejected out-of-sequence first row");
    }
  };
}

packed_yuv_resample_suite!(
  Yuyv422, Yuyv422Row, yuyv422_to, yuyv_frame, yuyv_from, yuyv_saturated,
  rgb: yuyv422_resample_rgb_matches_direct_conversion,
  luma: yuyv422_resample_luma_is_area_downscaled_y,
  sat: yuyv422_resample_luma_from_y_under_saturated_chroma,
  all: yuyv422_resample_all_outputs_combo,
  identity: yuyv422_identity_plan_matches_new_sink,
  no_op: yuyv422_resample_no_outputs_is_a_no_op,
  reset: yuyv422_resample_resets_streams_across_frames,
  seq_first: yuyv422_out_of_sequence_first_row_rejected_before_allocation,
  seq_mid: yuyv422_resample_rejects_mid_frame_out_of_sequence,
  freeze: yuyv422_resample_rejects_mid_frame_output_change,
  poison: yuyv422_rejected_first_row_does_not_poison_output_retry,
);

packed_yuv_resample_suite!(
  Uyvy422, Uyvy422Row, uyvy422_to, uyvy_frame, uyvy_from, uyvy_saturated,
  rgb: uyvy422_resample_rgb_matches_direct_conversion,
  luma: uyvy422_resample_luma_is_area_downscaled_y,
  sat: uyvy422_resample_luma_from_y_under_saturated_chroma,
  all: uyvy422_resample_all_outputs_combo,
  identity: uyvy422_identity_plan_matches_new_sink,
  no_op: uyvy422_resample_no_outputs_is_a_no_op,
  reset: uyvy422_resample_resets_streams_across_frames,
  seq_first: uyvy422_out_of_sequence_first_row_rejected_before_allocation,
  seq_mid: uyvy422_resample_rejects_mid_frame_out_of_sequence,
  freeze: uyvy422_resample_rejects_mid_frame_output_change,
  poison: uyvy422_rejected_first_row_does_not_poison_output_retry,
);

packed_yuv_resample_suite!(
  Yvyu422, Yvyu422Row, yvyu422_to, yvyu_frame, yvyu_from, yvyu_saturated,
  rgb: yvyu422_resample_rgb_matches_direct_conversion,
  luma: yvyu422_resample_luma_is_area_downscaled_y,
  sat: yvyu422_resample_luma_from_y_under_saturated_chroma,
  all: yvyu422_resample_all_outputs_combo,
  identity: yvyu422_identity_plan_matches_new_sink,
  no_op: yvyu422_resample_no_outputs_is_a_no_op,
  reset: yvyu422_resample_resets_streams_across_frames,
  seq_first: yvyu422_out_of_sequence_first_row_rejected_before_allocation,
  seq_mid: yvyu422_resample_rejects_mid_frame_out_of_sequence,
  freeze: yvyu422_resample_rejects_mid_frame_output_change,
  poison: yvyu422_rejected_first_row_does_not_poison_output_retry,
);
