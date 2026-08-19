//! Fused-downscale coverage for the high-bit packed RGB family
//! (`Rgb48` / `Bgr48`): the wire row converts to source-width host u16
//! RGB, binning runs at native 16-bit depth, the native-depth
//! `rgb_u16` / `rgba_u16` outputs are exact area means, and the u8 /
//! `luma_u16` outputs derive from a single `>> 8` narrowing — the same
//! source-of-truth ordering the direct path uses.

use crate::{
  KernelMatrix, PixelSink,
  resample::{AreaResampler, ResampleError},
  sinker::{MixedSinker, MixedSinkerError},
  source::{Bgr48, Bgr48Row, Rgb48, Rgb48Row, bgr48_to, rgb48_to},
};
use mediaframe::frame::{Bgr48Frame, Rgb48Frame};

const SRC: usize = 8;
const OUT: usize = 4;

/// Re-encode a host-native u16 slice as LE-wire byte storage, so a
/// fixture reads back identically on LE (no-op) and BE (byte-swap)
/// hosts.
fn as_le_u16(host: &[u16]) -> Vec<u16> {
  host
    .iter()
    .map(|v| u16::from_ne_bytes(v.to_le_bytes()))
    .collect()
}

/// Per-channel full-range u16 ramps; every value interior so the
/// derived kernels (luma / hsv) see real math and the wide
/// accumulator carries bits a u8 path would drop.
fn packed_frame_u16() -> Vec<u16> {
  let mut buf = vec![0u16; SRC * SRC * 3];
  for (i, px) in buf.chunks_exact_mut(3).enumerate() {
    px[0] = 4000 + (i as u16) * 600;
    px[1] = 60000 - (i as u16) * 700;
    px[2] = 1000 + ((i % 8) as u16) * 5000;
  }
  buf
}

/// Exact 2x2 block mean with round-half-up over host u16 values — the
/// contract for integer-ratio area downscale at native depth.
fn expected_block_mean_u16(src: &[u16], ox: usize, oy: usize, c: usize) -> u16 {
  let mut acc = 0u64;
  for dy in 0..2 {
    for dx in 0..2 {
      acc += src[((oy * 2 + dy) * SRC + ox * 2 + dx) * 3 + c] as u64;
    }
  }
  ((acc + 2) / 4) as u16
}

#[test]
fn rgb48_downscale_rgb_u16_is_exact_area_mean() {
  let host = packed_frame_u16();
  let wire = as_le_u16(&host);
  let src = Rgb48Frame::new(&wire, SRC as u32, SRC as u32, (SRC * 3) as u32);

  let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
  {
    let mut sink =
      MixedSinker::<Rgb48, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb_u16(&mut rgb_u16)
        .unwrap();
    rgb48_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
  }
  for oy in 0..OUT {
    for ox in 0..OUT {
      for c in 0..3 {
        assert_eq!(
          rgb_u16[(oy * OUT + ox) * 3 + c],
          expected_block_mean_u16(&host, ox, oy, c),
          "({ox},{oy}) c{c}"
        );
      }
    }
  }
}

#[test]
fn rgb48_derived_outputs_come_from_binned_rgb() {
  // Every attached output — native-depth u16 and narrowed u8 — must be
  // exactly what the direct full-res Rgb48 sink produces over a frame
  // that already holds the binned u16 RGB.
  let host = packed_frame_u16();
  let wire = as_le_u16(&host);
  let src = Rgb48Frame::new(&wire, SRC as u32, SRC as u32, (SRC * 3) as u32);

  let mut rgb = vec![0u8; OUT * OUT * 3];
  let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
  let mut rgba = vec![0u8; OUT * OUT * 4];
  let mut rgba_u16 = vec![0u16; OUT * OUT * 4];
  let mut luma = vec![0u8; OUT * OUT];
  let mut luma_u16 = vec![0u16; OUT * OUT];
  let mut h = vec![0u8; OUT * OUT];
  let mut s_ = vec![0u8; OUT * OUT];
  let mut v_ = vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Rgb48, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb(&mut rgb)
        .unwrap()
        .with_rgb_u16(&mut rgb_u16)
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
    rgb48_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
  }

  // Reference: the full-res sink over the (exact) binned u16 RGB.
  let binned_wire = as_le_u16(&rgb_u16);
  let mut ref_rgb = vec![0u8; OUT * OUT * 3];
  let mut ref_rgba = vec![0u8; OUT * OUT * 4];
  let mut ref_rgba_u16 = vec![0u16; OUT * OUT * 4];
  let mut ref_luma = vec![0u8; OUT * OUT];
  let mut ref_luma_u16 = vec![0u16; OUT * OUT];
  let mut ref_h = vec![0u8; OUT * OUT];
  let mut ref_s = vec![0u8; OUT * OUT];
  let mut ref_v = vec![0u8; OUT * OUT];
  {
    let binned = Rgb48Frame::new(&binned_wire, OUT as u32, OUT as u32, (OUT * 3) as u32);
    let mut sink = MixedSinker::<Rgb48>::new(OUT, OUT)
      .with_rgb(&mut ref_rgb)
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
    rgb48_to(&binned, true, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
  }
  assert_eq!(rgb, ref_rgb, "rgb");
  assert_eq!(rgba, ref_rgba, "rgba");
  assert_eq!(rgba_u16, ref_rgba_u16, "rgba_u16");
  assert_eq!(luma, ref_luma, "luma");
  assert_eq!(luma_u16, ref_luma_u16, "luma_u16");
  assert_eq!(h, ref_h, "h");
  assert_eq!(s_, ref_s, "s");
  assert_eq!(v_, ref_v, "v");
}

#[test]
fn rgb48_identity_plan_matches_new_sink() {
  let host = packed_frame_u16();
  let wire = as_le_u16(&host);
  let src = Rgb48Frame::new(&wire, SRC as u32, SRC as u32, (SRC * 3) as u32);

  let mut direct = vec![0u16; SRC * SRC * 3];
  {
    let mut sink = MixedSinker::<Rgb48>::new(SRC, SRC)
      .with_rgb_u16(&mut direct)
      .unwrap();
    rgb48_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
  }
  let mut via_area = vec![0u16; SRC * SRC * 3];
  {
    let mut sink =
      MixedSinker::<Rgb48, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(SRC, SRC))
        .unwrap()
        .with_rgb_u16(&mut via_area)
        .unwrap();
    rgb48_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
  }
  assert_eq!(direct, via_area);
}

#[test]
fn rgb48_contracts_hold_on_the_fused_path() {
  let host = packed_frame_u16();
  let wire = as_le_u16(&host);
  let row0 = &wire[..SRC * 3];

  // Out-of-order direct process.
  let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
  {
    let mut sink =
      MixedSinker::<Rgb48, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb_u16(&mut rgb_u16)
        .unwrap();
    sink.begin_frame(SRC as u32, SRC as u32).unwrap();
    let err = sink
      .process(Rgb48Row::for_tests(row0, 3, KernelMatrix::Bt709, true))
      .unwrap_err();
    assert!(matches!(
      err,
      MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
    ));
  }

  // Mid-frame output change.
  let mut luma = vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Rgb48, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb_u16(&mut rgb_u16)
        .unwrap();
    sink.begin_frame(SRC as u32, SRC as u32).unwrap();
    sink
      .process(Rgb48Row::for_tests(row0, 0, KernelMatrix::Bt709, true))
      .unwrap();
    sink.set_luma(&mut luma).unwrap();
    let err = sink
      .process(Rgb48Row::for_tests(
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
fn bgr48_downscale_rgb_u16_is_exact_area_mean() {
  // Bgr48 stores B, G, R; the converted host RGB swaps channel 0 and 2,
  // so the binned rgb_u16 output equals the block mean of the source
  // with the same swap applied.
  let host = packed_frame_u16();
  let wire = as_le_u16(&host);
  let src = Bgr48Frame::new(&wire, SRC as u32, SRC as u32, (SRC * 3) as u32);

  let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
  {
    let mut sink =
      MixedSinker::<Bgr48, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb_u16(&mut rgb_u16)
        .unwrap();
    bgr48_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
  }
  let swap = [2usize, 1, 0];
  for oy in 0..OUT {
    for ox in 0..OUT {
      for c in 0..3 {
        assert_eq!(
          rgb_u16[(oy * OUT + ox) * 3 + c],
          expected_block_mean_u16(&host, ox, oy, swap[c]),
          "({ox},{oy}) c{c}"
        );
      }
    }
  }
}

#[test]
fn rgb48_u16_only_downscale_does_not_size_the_narrow_scratch() {
  let host = packed_frame_u16();
  let wire = as_le_u16(&host);
  let src = Rgb48Frame::new(&wire, SRC as u32, SRC as u32, (SRC * 3) as u32);

  // Native-u16-only sink (only rgb_u16 attached): the binned row is
  // copied at native depth, so the u8 narrow scratch is never sized.
  let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
  let mut sink =
    MixedSinker::<Rgb48, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_rgb_u16(&mut rgb_u16)
      .unwrap();
  rgb48_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
  assert_eq!(
    sink.rgb_scratch_capacity(),
    0,
    "native-u16-only sink grew the u8 narrow scratch"
  );

  // Positive control: attaching a u8 output sizes it to the out-width
  // RGB row.
  let mut rgb_u16b = vec![0u16; OUT * OUT * 3];
  let mut rgb = vec![0u8; OUT * OUT * 3];
  let mut sink2 =
    MixedSinker::<Rgb48, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_rgb_u16(&mut rgb_u16b)
      .unwrap()
      .with_rgb(&mut rgb)
      .unwrap();
  rgb48_to(&src, true, sink2.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
  assert!(
    sink2.rgb_scratch_capacity() >= OUT * 3,
    "u8 output did not size the narrow scratch"
  );
}

#[test]
fn rgb48_out_of_sequence_first_row_rejected_before_allocation() {
  let host = packed_frame_u16();
  let wire = as_le_u16(&host);
  let row3 = &wire[3 * SRC * 3..4 * SRC * 3];

  let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
  let mut sink =
    MixedSinker::<Rgb48, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_rgb_u16(&mut rgb_u16)
      .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  let err = sink
    .process(Rgb48Row::for_tests(row3, 3, KernelMatrix::Bt709, true))
    .unwrap_err();
  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
    ),
    "expected OutOfSequenceRow, got {err:?}"
  );
  // The out-of-sequence first row must be rejected before the u16
  // stream or the narrow scratch is allocated.
  assert!(
    !sink.rgb_stream_u16_allocated(),
    "stream allocated for a rejected row"
  );
  assert_eq!(
    sink.rgb_scratch_capacity(),
    0,
    "scratch grown for a rejected row"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bgr48_first_row_scratch_oom_leaves_stream_and_freeze_uncommitted_for_retry() {
  let host = packed_frame_u16();
  let wire = as_le_u16(&host);
  let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
  let mut luma = vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Bgr48, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb_u16(&mut rgb_u16)
        .unwrap();
    sink.begin_frame(SRC as u32, SRC as u32).unwrap();
    // Arm the single-shot source-RGB-u16-scratch grow failpoint: Bgr48 stages its
    // wire row into the host-u16 scratch, so the row-0 stream box builds OK and the
    // scratch grow then refuses — surfacing AllocationFailed with BOTH the stream
    // field `None` and the freeze uncommitted (the commit-together atomic shape; the
    // pre-fix ordering would have left the stream committed).
    crate::sinker::mixed::arm_source_rgb_u16_scratch_failure();
    let err = sink
      .process(Bgr48Row::for_tests(
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
    // DIRECT stream-rollback proof (the non-vacuous part): the u16 area stream field
    // must be `None` — the fix builds it into a LOCAL and inserts only after the
    // scratch grows, so a failed scratch leaves nothing committed. The insert-before-
    // scratch shape would leave it `Some` here, so this assertion FAILS against that
    // shape and PASSES against the fix.
    assert!(
      !sink.rgb_stream_u16_allocated(),
      "a scratch OOM must leave the rgb_stream_u16 field None (no partial commit)"
    );
    // The freeze was likewise never committed — attaching luma (a CHANGED output
    // set) and replaying from row 0 is ACCEPTED, not ResampleOutputsChanged.
    sink.set_luma(&mut luma).unwrap();
    for r in 0..SRC {
      sink
        .process(Bgr48Row::for_tests(
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
fn rgb48_rejected_first_row_does_not_poison_output_retry() {
  // A rejected out-of-sequence FIRST row must store no frozen-output
  // snapshot (the split high-bit packed-RGB preflight rejects the OOS
  // first row before its freeze), so retrying row 0 after attaching a new
  // output succeeds instead of tripping ResampleOutputsChanged.
  let host = packed_frame_u16();
  let wire = as_le_u16(&host);
  let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
  let mut sink =
    MixedSinker::<Rgb48, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_rgb_u16(&mut rgb_u16)
      .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  let err = sink
    .process(Rgb48Row::for_tests(
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
    .process(Rgb48Row::for_tests(
      &wire[..SRC * 3],
      0,
      KernelMatrix::Bt709,
      true,
    ))
    .expect("row 0 must succeed after a rejected out-of-sequence first row");
}
