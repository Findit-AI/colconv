//! Fused-downscale coverage for `Bgr24` — the second format routed
//! through the shared packed-RGB resample tail. The source row is
//! swapped to RGB at source width in the shared scratch, then fed to
//! the same area stream and emit derivations as `Rgb24`, so these
//! tests pin both the swap and the shared tail.

use crate::{
  ColorMatrix, PixelSink,
  resample::{AreaResampler, FilteredResampler, ResampleError, Triangle},
  sinker::{MixedSinker, MixedSinkerError},
  source::{Bgr24, Bgr24Row, Rgb24, bgr24_to, rgb24_to},
};
use mediaframe::frame::{Bgr24Frame, Rgb24Frame};

const SRC: usize = 8;
const OUT: usize = 4;

/// Per-channel ramps in `B, G, R` order; every value interior so the
/// derived kernels see real math.
fn packed_bgr_frame() -> Vec<u8> {
  let mut buf = vec![0u8; SRC * SRC * 3];
  for (i, px) in buf.chunks_exact_mut(3).enumerate() {
    px[0] = 60 + ((i % 8) as u8) * 10; // B
    px[1] = 200 - (i as u8) * 2; // G
    px[2] = 40 + (i as u8) * 2; // R
  }
  buf
}

/// The same pixels with channels swapped to `R, G, B` — the canonical
/// form the shared tail bins.
fn swapped_to_rgb(bgr: &[u8]) -> Vec<u8> {
  let mut rgb = bgr.to_vec();
  for px in rgb.chunks_exact_mut(3) {
    px.swap(0, 2);
  }
  rgb
}

/// Direct 2x2 block mean (round-half-up) of the swapped (RGB) source —
/// the exact contract for integer-ratio area downscale.
fn expected_block_mean(rgb: &[u8], ox: usize, oy: usize, c: usize) -> u8 {
  let mut acc = 0u32;
  for dy in 0..2 {
    for dx in 0..2 {
      acc += rgb[((oy * 2 + dy) * SRC + ox * 2 + dx) * 3 + c] as u32;
    }
  }
  ((acc + 2) / 4) as u8
}

#[test]
fn bgr24_downscale_is_exact_area_mean_of_swapped_source() {
  let buf = packed_bgr_frame();
  let rgb_ref = swapped_to_rgb(&buf);
  let src = Bgr24Frame::try_new(&buf, SRC as u32, SRC as u32, (SRC * 3) as u32).unwrap();

  let mut rgb = vec![0u8; OUT * OUT * 3];
  {
    let mut sink =
      MixedSinker::<Bgr24, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb(&mut rgb)
        .unwrap();
    bgr24_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();
  }
  for oy in 0..OUT {
    for ox in 0..OUT {
      for c in 0..3 {
        assert_eq!(
          rgb[(oy * OUT + ox) * 3 + c],
          expected_block_mean(&rgb_ref, ox, oy, c),
          "({ox},{oy}) c{c}"
        );
      }
    }
  }
}

#[test]
fn bgr24_resample_matches_rgb24_of_swapped_frame() {
  // The shared tail is format-agnostic past the swap: a Bgr24 frame
  // and the channel-swapped Rgb24 frame must produce identical RGB,
  // RGBA, luma and HSV outputs.
  let buf = packed_bgr_frame();
  let rgb_buf = swapped_to_rgb(&buf);
  let bgr_src = Bgr24Frame::try_new(&buf, SRC as u32, SRC as u32, (SRC * 3) as u32).unwrap();
  let rgb_src = Rgb24Frame::new(&rgb_buf, SRC as u32, SRC as u32, (SRC * 3) as u32);

  let (mut rgb_a, mut rgba_a, mut luma_a) = (
    vec![0u8; OUT * OUT * 3],
    vec![0u8; OUT * OUT * 4],
    vec![0u8; OUT * OUT],
  );
  {
    let mut sink =
      MixedSinker::<Bgr24, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb(&mut rgb_a)
        .unwrap()
        .with_rgba(&mut rgba_a)
        .unwrap()
        .with_luma(&mut luma_a)
        .unwrap();
    bgr24_to(&bgr_src, true, ColorMatrix::Bt709, &mut sink).unwrap();
  }

  let (mut rgb_b, mut rgba_b, mut luma_b) = (
    vec![0u8; OUT * OUT * 3],
    vec![0u8; OUT * OUT * 4],
    vec![0u8; OUT * OUT],
  );
  {
    let mut sink =
      MixedSinker::<Rgb24, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb(&mut rgb_b)
        .unwrap()
        .with_rgba(&mut rgba_b)
        .unwrap()
        .with_luma(&mut luma_b)
        .unwrap();
    rgb24_to(&rgb_src, true, ColorMatrix::Bt709, &mut sink).unwrap();
  }

  assert_eq!(rgb_a, rgb_b, "rgb");
  assert_eq!(rgba_a, rgba_b, "rgba");
  assert_eq!(luma_a, luma_b, "luma");
}

#[test]
fn bgr24_identity_plan_matches_new_sink() {
  let buf = packed_bgr_frame();
  let src = Bgr24Frame::try_new(&buf, SRC as u32, SRC as u32, (SRC * 3) as u32).unwrap();

  let mut direct = vec![0u8; SRC * SRC * 3];
  {
    let mut sink = MixedSinker::<Bgr24>::new(SRC, SRC)
      .with_rgb(&mut direct)
      .unwrap();
    bgr24_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();
  }

  let mut via_area = vec![0u8; SRC * SRC * 3];
  {
    let mut sink =
      MixedSinker::<Bgr24, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(SRC, SRC))
        .unwrap()
        .with_rgb(&mut via_area)
        .unwrap();
    bgr24_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();
  }
  assert_eq!(direct, via_area);
}

#[test]
fn bgr24_resample_no_outputs_is_a_no_op() {
  // Attaching no outputs is legal and must write nothing. The
  // resample route freezes the (empty) output set, then returns
  // before allocating a stream or enforcing sequencing — so even an
  // out-of-order row is accepted, which a stream-backed path would
  // reject. (Without the output-presence preflight this allocates and
  // sequence-checks for no observable output.)
  let buf = packed_bgr_frame();
  let mut sink =
    MixedSinker::<Bgr24, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  let row2 = &buf[SRC * 3 * 2..SRC * 3 * 3];
  sink
    .process(Bgr24Row::new(row2, 2, ColorMatrix::Bt709, true))
    .unwrap();
}

#[test]
fn bgr24_resample_rejects_out_of_sequence_before_staging() {
  // An out-of-sequence row must be rejected before the source-width
  // scratch is grown or the BGR->RGB swap runs — sequencing is
  // checked ahead of staging (matching Rgb24 / YUV). The scratch
  // capacity staying zero after the rejection proves no staging
  // allocation happened; without the ordering it would grow to
  // source-width * 3.
  let buf = packed_bgr_frame();
  let mut rgb = vec![0u8; OUT * OUT * 3];
  let mut sink =
    MixedSinker::<Bgr24, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_rgb(&mut rgb)
      .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  let row2 = &buf[SRC * 3 * 2..SRC * 3 * 3];
  let err = sink
    .process(Bgr24Row::new(row2, 2, ColorMatrix::Bt709, true))
    .unwrap_err();
  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
    ),
    "got {err:?}"
  );
  assert_eq!(
    sink.rgb_scratch_capacity(),
    0,
    "out-of-sequence row must not stage into the scratch"
  );
}

#[test]
fn bgr24_filter_plan_rejected_with_typed_error() {
  // Bgr24 is NOT routed to the filter path (only Rgb24 / Rgb48 / Grayf32
  // are). A FilteredResampler plan must be REJECTED with the typed
  // UnsupportedFilter error at the first processed row — never silently
  // feed the plan's empty area spans to an area stream (which emits no
  // output and leaves the attached buffer stale).
  let buf = packed_bgr_frame();
  let src = Bgr24Frame::try_new(&buf, SRC as u32, SRC as u32, (SRC * 3) as u32).unwrap();

  let mut rgb = vec![0xABu8; OUT * OUT * 3];
  let mut sink = MixedSinker::<Bgr24, FilteredResampler<Triangle>>::with_resampler(
    SRC,
    SRC,
    FilteredResampler::new(OUT, OUT, Triangle),
  )
  .unwrap()
  .with_rgb(&mut rgb)
  .unwrap();
  let err = bgr24_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap_err();
  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::UnsupportedFilter(_))
    ),
    "Bgr24 filter plan must reject with UnsupportedFilter, got {err:?}"
  );
  // The output buffer is untouched — the rejection happened before any
  // emit, so the sentinel survives (no silent stale-output bug).
  assert!(
    rgb.iter().all(|&b| b == 0xAB),
    "rejected filter plan must not write the output buffer"
  );
}

#[test]
fn bgr24_resample_rejects_out_of_sequence_rows() {
  let buf = packed_bgr_frame();
  let mut rgb = vec![0u8; OUT * OUT * 3];
  let mut sink =
    MixedSinker::<Bgr24, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_rgb(&mut rgb)
      .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  // Skip row 0 — the stream expects strict sequencing from row 0.
  let row1 = &buf[SRC * 3..SRC * 3 * 2];
  let err = sink
    .process(Bgr24Row::new(row1, 1, ColorMatrix::Bt709, true))
    .unwrap_err();
  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
    ),
    "got {err:?}"
  );
}

/// RFC #238 tail-collapse atomicity (3-channel, scratch OOM): `packed_rgb_resample`
/// builds the area stream into a LOCAL and grows the source-width scratch BEFORE
/// committing the stream field + the output-set freeze TOGETHER. A scratch-grow
/// OOM AFTER the (successful) stream construction therefore leaves the stream
/// field `None` AND `resample_outputs` uncommitted (no partial commit), so a
/// row-0 retry with a CHANGED output set is ACCEPTED and drives real output.
/// (`Rgb24` is zero-copy/direct with no scratch to fail; `Bgr24`, its staged
/// sibling, grows `source_rgb_scratch`, so it exercises this path.)
#[test]
#[cfg(feature = "std")]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bgr24_first_row_scratch_oom_leaves_stream_and_freeze_uncommitted_for_retry() {
  let buf = packed_bgr_frame();
  let mut rgb = vec![0u8; OUT * OUT * 3];
  let mut luma = vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Bgr24, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb(&mut rgb)
        .unwrap();
    sink.begin_frame(SRC as u32, SRC as u32).unwrap();
    // Arm the single-shot source-RGB-scratch grow failpoint: the row-0 stream box
    // builds OK, then the scratch grow refuses — surfacing AllocationFailed with
    // BOTH the stream field `None` and the freeze uncommitted (the commit-together
    // atomic shape; the pre-fix ordering would have left the stream committed).
    crate::sinker::mixed::arm_source_rgb_scratch_failure();
    let err = sink
      .process(Bgr24Row::new(&buf[..SRC * 3], 0, ColorMatrix::Bt709, true))
      .unwrap_err();
    assert!(
      matches!(
        err,
        MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
      ),
      "first-row scratch OOM must surface AllocationFailed, got {err:?}"
    );
    // DIRECT stream-rollback proof (the non-vacuous part): the area stream field
    // must be `None` — the fix builds it into a LOCAL and inserts only after the
    // scratch grows, so the failed scratch leaves nothing committed. The R1-buggy
    // shape (insert the stream BEFORE the scratch grow) would leave it `Some` here,
    // so this assertion FAILS against the buggy shape and PASSES against the fix.
    assert!(
      !sink.rgb_stream_allocated(),
      "a scratch OOM must leave the rgb_stream field None (no partial commit)"
    );
    // The freeze was likewise never committed — attaching luma (a CHANGED output
    // set) and replaying from row 0 is ACCEPTED, not ResampleOutputsChanged.
    sink.set_luma(&mut luma).unwrap();
    for r in 0..SRC {
      sink
        .process(Bgr24Row::new(
          &buf[r * SRC * 3..(r + 1) * SRC * 3],
          r,
          ColorMatrix::Bt709,
          true,
        ))
        .expect("frame replay after a first-row scratch OOM must succeed");
    }
  }
  assert!(
    rgb.iter().any(|&b| b != 0),
    "the recovered frame produced real RGB output (scratch OOM was recoverable)"
  );
  assert!(
    luma.iter().any(|&b| b != 0),
    "the newly-attached luma output was driven on the accepted retry"
  );
}
