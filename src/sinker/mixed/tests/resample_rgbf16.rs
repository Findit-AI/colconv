//! Fused-downscale coverage for the packed-half-float-RGB family
//! ([`Rgbf16`]).
//!
//! There is no `AreaStream<f16>`, so `Rgbf16` widens its packed `R, G, B`
//! `half::f16` wire row to source-width host-native f32 RGB and bins in
//! **float** on the shared `AreaStream<f32>`. Per finalized output row the
//! tail **rounds the binned packed f32 row to `half::f16`** (it stays
//! packed — no de-interleave) and runs the exact direct `rgbf16_*`
//! kernels. Therefore every output (rgb / rgba / rgb_u16 / rgba_u16 /
//! rgb_f32 / rgb_f16 / luma / luma_u16 / hsv) is **byte-identical** to a
//! direct full-resolution `Rgbf16` conversion of the pre-binned frame —
//! the frame whose per-pixel f16 `R, G, B` is the f32 area mean rounded to
//! f16 (the oracle). For an integer downscale ratio the area mean is the
//! simple block average.

use crate::{
  KernelMatrix, PixelSink,
  frame::Rgbf16LeFrame,
  resample::{AreaResampler, ResampleError},
  sinker::{MixedSinker, MixedSinkerError},
  source::{Rgbf16, Rgbf16Row, rgbf16_to},
};

const SRC: usize = 8;
const OUT: usize = 4;

/// LE-encode a host-native `half::f16` slice as the `*LE` Frame contract
/// requires, so a fixture reads back identically on LE (no-op) and BE
/// (byte-swap) hosts. Mirrors `as_le_f16` in `resample_gbrpf16`.
fn as_le_f16(host: &[half::f16]) -> Vec<half::f16> {
  host
    .iter()
    .map(|&v| half::f16::from_bits(v.to_bits().to_le()))
    .collect()
}

/// Per-channel packed f16 ramp with **integer-valued** samples (so the
/// f32 2x2 area mean is exact and its round to f16 is deterministic) that
/// deliberately spans HDR (> 1.0) and negative values — the float path
/// carries both into the binned row, then the round to f16 + direct
/// `rgbf16_*` kernels reproduce the direct path's saturation on the
/// integer/u8 outputs. Returns a host-native packed `R, G, B` f16 buffer.
fn packed_frame_f16() -> Vec<half::f16> {
  let mut buf = std::vec![half::f16::ZERO; SRC * SRC * 3];
  for (i, px) in buf.chunks_exact_mut(3).enumerate() {
    let i = i as i32;
    // R: small in-range integers and HDR — saturates on u8/u16 outputs,
    // preserved (post-round) in rgb_f32.
    px[0] = half::f16::from_f32((i % 5) as f32);
    // G: large HDR values to exercise saturation.
    px[1] = half::f16::from_f32((100 - i) as f32);
    // B: negative samples — clamp to 0 on integer outputs, preserved
    // (with sign, post-round) in rgb_f32.
    px[2] = half::f16::from_f32(-((i % 7) as f32));
  }
  buf
}

/// Build the pre-binned packed f16 frame: average the f16 source **in
/// f32** over each 2x2 block, then round the mean to `half::f16`. This is
/// the oracle's pre-binned per-pixel value — the f32 block-mean rounded to
/// f16 — laid out packed `R, G, B`.
fn prebinned_packed_f16(src: &[half::f16]) -> Vec<half::f16> {
  let mut out = std::vec![half::f16::ZERO; OUT * OUT * 3];
  for oy in 0..OUT {
    for ox in 0..OUT {
      for c in 0..3 {
        let mut acc = 0.0f64;
        for dy in 0..2 {
          for dx in 0..2 {
            acc += src[((oy * 2 + dy) * SRC + ox * 2 + dx) * 3 + c].to_f32() as f64;
          }
        }
        out[(oy * OUT + ox) * 3 + c] = half::f16::from_f32((acc / 4.0) as f32);
      }
    }
  }
  out
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgbf16_downscale_rgb_f32_is_f16_rounded_area_mean() {
  // The direct `Rgbf16` `rgb_f32` path widens the f16 source to f32, so the
  // fused `rgb_f32` is the f32 area mean rounded to f16, then widened back
  // to f32 — NOT the raw f32 bin. Assert exactly that.
  let host = packed_frame_f16();
  let wire = as_le_f16(&host);
  let src = Rgbf16LeFrame::try_new(&wire, SRC as u32, SRC as u32, (SRC * 3) as u32).unwrap();

  let mut rgb_f32 = std::vec![0.0f32; OUT * OUT * 3];
  {
    let mut sink =
      MixedSinker::<Rgbf16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb_f32(&mut rgb_f32)
        .unwrap();
    rgbf16_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  }
  let prebinned = prebinned_packed_f16(&host);
  for oy in 0..OUT {
    for ox in 0..OUT {
      for c in 0..3 {
        let i = (oy * OUT + ox) * 3 + c;
        assert_eq!(rgb_f32[i], prebinned[i].to_f32(), "({ox},{oy}) c{c}");
      }
    }
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgbf16_downscale_rgb_f32_preserves_hdr_and_negative() {
  // The binned-then-rounded mean of an HDR-or-negative 2x2 block must
  // itself be HDR / negative — proving the float path never clamps
  // `rgb_f32`. The fixture's G channel is `100 - i` (always > 1, and well
  // within f16 range) and B is `-(i % 7)` (<= 0).
  let host = packed_frame_f16();
  let wire = as_le_f16(&host);
  let src = Rgbf16LeFrame::try_new(&wire, SRC as u32, SRC as u32, (SRC * 3) as u32).unwrap();

  let mut rgb_f32 = std::vec![0.0f32; OUT * OUT * 3];
  {
    let mut sink =
      MixedSinker::<Rgbf16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb_f32(&mut rgb_f32)
        .unwrap();
    rgbf16_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  }
  assert!(
    rgb_f32.iter().any(|&v| v > 1.0),
    "binned rgb_f32 lost all HDR (> 1.0) values"
  );
  assert!(
    rgb_f32.iter().any(|&v| v < 0.0),
    "binned rgb_f32 clamped away negative values"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgbf16_all_outputs_match_direct_conversion_of_prebinned_frame() {
  // Resample SRC->OUT with every output attached, then compare against a
  // full-resolution direct Rgbf16 conversion of the pre-binned frame (the
  // f32 area mean rounded to f16) — the parity oracle. Every output is
  // byte-identical to the direct path, luma_u16 at the direct path's
  // narrowed (8-bit-in-u16) precision (its kernel stages through u8).
  let host = packed_frame_f16();
  let wire = as_le_f16(&host);
  let src = Rgbf16LeFrame::try_new(&wire, SRC as u32, SRC as u32, (SRC * 3) as u32).unwrap();

  let mut rgb = std::vec![0u8; OUT * OUT * 3];
  let mut rgb_u16 = std::vec![0u16; OUT * OUT * 3];
  let mut rgb_f32 = std::vec![0.0f32; OUT * OUT * 3];
  let mut rgb_f16 = std::vec![half::f16::ZERO; OUT * OUT * 3];
  let mut rgba = std::vec![0u8; OUT * OUT * 4];
  let mut rgba_u16 = std::vec![0u16; OUT * OUT * 4];
  let mut luma = std::vec![0u8; OUT * OUT];
  let mut luma_u16 = std::vec![0u16; OUT * OUT];
  let mut h = std::vec![0u8; OUT * OUT];
  let mut s_ = std::vec![0u8; OUT * OUT];
  let mut v_ = std::vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Rgbf16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb(&mut rgb)
        .unwrap()
        .with_rgb_u16(&mut rgb_u16)
        .unwrap()
        .with_rgb_f32(&mut rgb_f32)
        .unwrap()
        .with_rgb_f16(&mut rgb_f16)
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
    rgbf16_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  }

  // Reference: the full-res direct sink over the pre-binned packed f16
  // frame (f32 area mean rounded to f16), LE-encoded exactly as the source.
  let prebinned = prebinned_packed_f16(&host);
  let prebinned_wire = as_le_f16(&prebinned);
  let mut ref_rgb = std::vec![0u8; OUT * OUT * 3];
  let mut ref_rgb_u16 = std::vec![0u16; OUT * OUT * 3];
  let mut ref_rgb_f32 = std::vec![0.0f32; OUT * OUT * 3];
  let mut ref_rgb_f16 = std::vec![half::f16::ZERO; OUT * OUT * 3];
  let mut ref_rgba = std::vec![0u8; OUT * OUT * 4];
  let mut ref_rgba_u16 = std::vec![0u16; OUT * OUT * 4];
  let mut ref_luma = std::vec![0u8; OUT * OUT];
  let mut ref_luma_u16 = std::vec![0u16; OUT * OUT];
  let mut ref_h = std::vec![0u8; OUT * OUT];
  let mut ref_s = std::vec![0u8; OUT * OUT];
  let mut ref_v = std::vec![0u8; OUT * OUT];
  {
    let binned =
      Rgbf16LeFrame::try_new(&prebinned_wire, OUT as u32, OUT as u32, (OUT * 3) as u32).unwrap();
    let mut sink = MixedSinker::<Rgbf16>::new(OUT, OUT)
      .with_rgb(&mut ref_rgb)
      .unwrap()
      .with_rgb_u16(&mut ref_rgb_u16)
      .unwrap()
      .with_rgb_f32(&mut ref_rgb_f32)
      .unwrap()
      .with_rgb_f16(&mut ref_rgb_f16)
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
    rgbf16_to(&binned, true, KernelMatrix::Bt709, &mut sink).unwrap();
  }

  assert_eq!(rgb, ref_rgb, "rgb");
  assert_eq!(rgb_u16, ref_rgb_u16, "rgb_u16");
  assert_eq!(rgb_f32, ref_rgb_f32, "rgb_f32 (f16-rounded, full parity)");
  assert_eq!(rgb_f16, ref_rgb_f16, "rgb_f16 (full parity)");
  assert_eq!(rgba, ref_rgba, "rgba");
  assert_eq!(rgba_u16, ref_rgba_u16, "rgba_u16");
  assert_eq!(luma, ref_luma, "luma");
  // luma_u16 on the fused path is the direct path's narrowed (8-bit, in a
  // u16 carrier) value — `rgbf16_to_rgb_row` stages through u8, then
  // `rgb_to_luma_u16_row` runs over it — so it is byte-identical to the
  // direct Rgbf16 `with_luma_u16`.
  assert_eq!(luma_u16, ref_luma_u16, "luma_u16 (narrowed, full parity)");
  assert_eq!(h, ref_h, "hsv H");
  assert_eq!(s_, ref_s, "hsv S");
  assert_eq!(v_, ref_v, "hsv V");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgbf16_identity_plan_matches_new_sink() {
  // An identity (SRC->SRC) plan still routes through the f32 bin + f16
  // round emit; an identity bin is a copy, so each output must equal the
  // direct `new` sink (which widens the f16 source to f32). rgb_f32 is the
  // f16 source widened to f32 — i.e. the identity-binned-then-rounded value
  // equals the source f16 widened, since rounding an exact f16 value is
  // itself.
  let host = packed_frame_f16();
  let wire = as_le_f16(&host);
  let src = Rgbf16LeFrame::try_new(&wire, SRC as u32, SRC as u32, (SRC * 3) as u32).unwrap();

  let mut direct = std::vec![0.0f32; SRC * SRC * 3];
  {
    let mut sink = MixedSinker::<Rgbf16>::new(SRC, SRC)
      .with_rgb_f32(&mut direct)
      .unwrap();
    rgbf16_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  }
  let mut via_area = std::vec![0.0f32; SRC * SRC * 3];
  {
    let mut sink =
      MixedSinker::<Rgbf16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(SRC, SRC))
        .unwrap()
        .with_rgb_f32(&mut via_area)
        .unwrap();
    rgbf16_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  }
  assert_eq!(direct, via_area, "identity-plan resample == direct sink");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgbf16_no_output_sink_is_a_noop() {
  // A resampled sink with no attached output neither allocates the
  // stream nor enforces sequencing — the documented legal no-op.
  let host = packed_frame_f16();
  let wire = as_le_f16(&host);
  let src = Rgbf16LeFrame::try_new(&wire, SRC as u32, SRC as u32, (SRC * 3) as u32).unwrap();

  let mut sink =
    MixedSinker::<Rgbf16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap();
  rgbf16_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  assert!(
    !sink.rgb_stream_f32_allocated(),
    "no-output sink allocated the f32 stream"
  );
  assert_eq!(
    sink.rgb_packed_scratch_f16_capacity(),
    0,
    "no-output sink grew the packed f16 scratch"
  );
  assert_eq!(
    sink.rgb_scratch_capacity(),
    0,
    "no-output sink grew the u8 narrow scratch"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgbf16_scratch_sizing_is_output_gated() {
  // The packed f16 row is sized whenever any output is attached (even an
  // f32-only sink, because rgb_f32 widens the rounded f16 row); the u8
  // narrow scratch only when a u8-staged output (rgb / luma / luma_u16 /
  // hsv) is attached.
  let host = packed_frame_f16();
  let wire = as_le_f16(&host);
  let src = Rgbf16LeFrame::try_new(&wire, SRC as u32, SRC as u32, (SRC * 3) as u32).unwrap();

  // f32-only sink: rounds the bin to the packed f16 row then widens it, so
  // the packed f16 row is sized but the u8 narrow scratch is not.
  let mut rgb_f32 = std::vec![0.0f32; OUT * OUT * 3];
  let mut sink =
    MixedSinker::<Rgbf16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_rgb_f32(&mut rgb_f32)
      .unwrap();
  rgbf16_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  assert!(
    sink.rgb_packed_scratch_f16_capacity() >= OUT * 3,
    "f32 output did not size the packed f16 scratch"
  );
  assert_eq!(
    sink.rgb_scratch_capacity(),
    0,
    "f32-only sink grew the u8 narrow scratch"
  );

  // u16-only sink likewise stays clear of the u8 narrow scratch (rgb_u16
  // derives directly from the rounded packed f16 row).
  let mut rgb_u16 = std::vec![0u16; OUT * OUT * 3];
  let mut sink_u16 =
    MixedSinker::<Rgbf16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_rgb_u16(&mut rgb_u16)
      .unwrap();
  rgbf16_to(&src, true, KernelMatrix::Bt709, &mut sink_u16).unwrap();
  assert_eq!(
    sink_u16.rgb_scratch_capacity(),
    0,
    "u16-only sink grew the u8 narrow scratch"
  );

  // Positive control: attaching a u8 output sizes the u8 narrow scratch to
  // the out-width RGB row.
  let mut rgb = std::vec![0u8; OUT * OUT * 3];
  let mut sink2 =
    MixedSinker::<Rgbf16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_rgb(&mut rgb)
      .unwrap();
  rgbf16_to(&src, true, KernelMatrix::Bt709, &mut sink2).unwrap();
  assert!(
    sink2.rgb_scratch_capacity() >= OUT * 3,
    "u8 output did not size the narrow scratch"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgbf16_resample_reuses_stream_across_frames() {
  // begin_frame resets the f32 area stream + frozen output set, so frame
  // 2's row 0 is accepted (not rejected as out-of-sequence) and the output
  // reflects frame 2's input. Both frames share one output buffer; only
  // the input data changes.
  let host1 = packed_frame_f16();
  // Frame 2: negate every channel so its block mean differs from frame 1.
  let host2: Vec<half::f16> = host1
    .iter()
    .map(|&v| half::f16::from_f32(-v.to_f32()))
    .collect();
  let wire1 = as_le_f16(&host1);
  let wire2 = as_le_f16(&host2);
  let src1 = Rgbf16LeFrame::try_new(&wire1, SRC as u32, SRC as u32, (SRC * 3) as u32).unwrap();
  let src2 = Rgbf16LeFrame::try_new(&wire2, SRC as u32, SRC as u32, (SRC * 3) as u32).unwrap();

  let mut out = std::vec![0.0f32; OUT * OUT * 3];
  {
    let mut sink =
      MixedSinker::<Rgbf16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb_f32(&mut out)
        .unwrap();
    rgbf16_to(&src1, true, KernelMatrix::Bt709, &mut sink).unwrap();
    rgbf16_to(&src2, true, KernelMatrix::Bt709, &mut sink).unwrap();
  }

  let prebinned2 = prebinned_packed_f16(&host2);
  for i in 0..OUT * OUT * 3 {
    assert_eq!(out[i], prebinned2[i].to_f32(), "frame2 elem {i}");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgbf16_out_of_sequence_first_row_rejected_before_allocation() {
  let host = packed_frame_f16();
  let wire = as_le_f16(&host);
  let row3 = &wire[3 * SRC * 3..4 * SRC * 3];

  let mut rgb_f32 = std::vec![0.0f32; OUT * OUT * 3];
  let mut sink =
    MixedSinker::<Rgbf16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_rgb_f32(&mut rgb_f32)
      .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  let err = sink
    .process(Rgbf16Row::new(row3, 3, KernelMatrix::Bt709, true))
    .unwrap_err();
  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
    ),
    "expected OutOfSequenceRow, got {err:?}"
  );
  // The out-of-sequence first row must be rejected before the f32 stream or
  // either scratch is allocated.
  assert!(
    !sink.rgb_stream_f32_allocated(),
    "stream allocated for a rejected row"
  );
  assert_eq!(
    sink.rgb_packed_scratch_f16_capacity(),
    0,
    "packed f16 scratch grown for a rejected row"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgbf16_mid_frame_out_of_sequence_rejected() {
  let host = packed_frame_f16();
  let wire = as_le_f16(&host);

  let mut rgb_f32 = std::vec![0.0f32; OUT * OUT * 3];
  let mut sink =
    MixedSinker::<Rgbf16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_rgb_f32(&mut rgb_f32)
      .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  sink
    .process(Rgbf16Row::new(
      &wire[..SRC * 3],
      0,
      KernelMatrix::Bt709,
      true,
    ))
    .unwrap();
  // Skip row 1 — feeding row 2 next must be rejected.
  let err = sink
    .process(Rgbf16Row::new(
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
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgbf16_mid_frame_output_change_rejected() {
  let host = packed_frame_f16();
  let wire = as_le_f16(&host);

  let mut rgb_f32 = std::vec![0.0f32; OUT * OUT * 3];
  let mut luma = std::vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Rgbf16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb_f32(&mut rgb_f32)
        .unwrap();
    sink.begin_frame(SRC as u32, SRC as u32).unwrap();
    sink
      .process(Rgbf16Row::new(
        &wire[..SRC * 3],
        0,
        KernelMatrix::Bt709,
        true,
      ))
      .unwrap();
    sink.set_luma(&mut luma).unwrap();
    let err = sink
      .process(Rgbf16Row::new(
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

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgbf16_first_row_emit_scratch_oom_leaves_stream_and_freeze_uncommitted_for_retry() {
  // The EMIT-owned out-width u8 narrow scratch (grown for a u8 `rgb` output by the
  // `Rgbf16` f16-round emit), NOT the source f32 staging scratch. The shared
  // combined helper pre-grows every emit-owned scratch via an `emit(None)` pregrow
  // call BEFORE the freeze, so a first-row OOM in it leaves the stream field `None`
  // and `resample_outputs` uncommitted — never a partial commit (the f16 arm reuses
  // the f32 stream, so this is the `AreaStream<f32>` field).
  let host = packed_frame_f16();
  let wire = as_le_f16(&host);
  let mut rgb = std::vec![0u8; OUT * OUT * 3];
  let mut luma = std::vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Rgbf16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb(&mut rgb)
        .unwrap();
    sink.begin_frame(SRC as u32, SRC as u32).unwrap();
    // Arm the u8 narrow-scratch (`source_rgb_scratch`) grow failpoint — the scratch
    // the f16 emit grows for its u8 `rgb` output, AFTER the stream box builds. The
    // pregrow call runs it before the freeze, so this OOM must leave BOTH the stream
    // field `None` and the freeze uncommitted.
    crate::sinker::mixed::arm_source_rgb_scratch_failure();
    let err = sink
      .process(Rgbf16Row::new(
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
    // field `None`. The post-freeze-grow shape would leave it `Some` here, so this
    // assertion FAILS against that shape and PASSES against the fix.
    assert!(
      !sink.rgb_stream_f32_allocated(),
      "an emit-owned scratch OOM must leave the rgb_stream_f32 field None (no partial commit)"
    );
    // The freeze was likewise never committed — attaching luma (a CHANGED output
    // set) and replaying from row 0 is ACCEPTED, not ResampleOutputsChanged.
    sink.set_luma(&mut luma).unwrap();
    for r in 0..SRC {
      sink
        .process(Rgbf16Row::new(
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
