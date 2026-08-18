//! Fused-downscale coverage for the 8-bit **Bayer** CFA source.
//!
//! Bayer routes the same freeze/stream/scratch plumbing the packed-RGB
//! tail uses, but with a Bayer-flavored emit: each CFA source row is
//! demosaiced into the source-width RGB scratch (the exact kernel the
//! direct path runs), those RGB rows are area-binned, and every output
//! is then derived from the finalized RGB row the **Bayer** way —
//! Q8-coefficient luma (`LumaCoefficients`, not a `KernelMatrix`) and
//! the OpenCV HSV kernel. Bayer carries no `KernelMatrix` / `full_range`
//! on its row, so it cannot share the packed-RGB tail's `KernelMatrix`
//! emit; only the plumbing is shared.
//!
//! The parity oracle is therefore "direct demosaic of the full source
//! → area-bin that RGB → Bayer-derive each output": the demosaiced RGB
//! a resampled frame bins is byte-identical to the direct path's RGB
//! output, so binning it (a round-half-up 2x2 block mean for the
//! integer 2:1 ratio) and re-deriving luma / HSV reproduces the
//! resampled outputs exactly.

use crate::{
  PixelSink,
  frame::BayerFrame,
  raw::{BayerDemosaic, BayerPattern, BayerRow, ColorCorrectionMatrix, WhiteBalance, bayer_to},
  resample::{AreaResampler, ResampleError},
  sinker::{MixedSinker, MixedSinkerError},
  source::Bayer,
};

const SRC: usize = 8;
const OUT: usize = 4;

/// Bt709 luma Q8 coefficients — the `LumaCoefficients::Bt709` default a
/// Bayer sink derives `with_luma` from. Mirrors
/// `LumaCoefficients::to_q8()` so the oracle does not need the
/// `pub(super)` `rgb_row_to_luma_row`.
const LUMA_Q8: (u32, u32, u32) = (54, 183, 19);

/// Deterministic Bayer plane — interior, non-uniform values so the
/// demosaic and the area-bin both see real math.
fn bayer_plane(seed: u32) -> Vec<u8> {
  let mut buf = vec![0u8; SRC * SRC];
  let mut state = seed;
  for b in buf.iter_mut() {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    // Keep samples in [32, 223] so demosaiced channels stay interior
    // (no clamping at 0 / 255 masking a kernel difference).
    *b = 32 + ((state >> 17) as u8 % 192);
  }
  buf
}

/// The full-resolution demosaiced RGB the direct Bayer path produces
/// for `plane` under `pattern` (neutral WB, identity CCM). This is the
/// frame the resampled path area-bins.
fn direct_demosaic_rgb(plane: &[u8], pattern: BayerPattern) -> Vec<u8> {
  let frame = BayerFrame::try_new(plane, SRC as u32, SRC as u32, SRC as u32).unwrap();
  let mut rgb = vec![0u8; SRC * SRC * 3];
  let mut sink = MixedSinker::<Bayer>::new(SRC, SRC)
    .with_rgb(&mut rgb)
    .unwrap();
  bayer_to(
    &frame,
    pattern,
    BayerDemosaic::Bilinear,
    WhiteBalance::neutral(),
    ColorCorrectionMatrix::identity(),
    &mut sink,
  )
  .unwrap();
  rgb
}

/// Area-bin a full-resolution RGB frame to `OUT x OUT` — for an
/// integer 2:1 ratio the area resampler is exactly a round-half-up 2x2
/// block mean (the `resample_bgr24` tests pin this equivalence against
/// the `AreaResampler` output). Computing it directly keeps the Bayer
/// oracle self-contained, with no `rgb`-feature dependency.
fn area_bin_rgb(full_rgb: &[u8]) -> Vec<u8> {
  let mut out = vec![0u8; OUT * OUT * 3];
  for oy in 0..OUT {
    for ox in 0..OUT {
      for c in 0..3 {
        let mut acc = 0u32;
        for dy in 0..2 {
          for dx in 0..2 {
            acc += full_rgb[((oy * 2 + dy) * SRC + ox * 2 + dx) * 3 + c] as u32;
          }
        }
        out[(oy * OUT + ox) * 3 + c] = ((acc + 2) / 4) as u8;
      }
    }
  }
  out
}

/// Bayer-way luma of a binned RGB frame: `Y = (54R + 183G + 19B + 128) >> 8`,
/// clamped to 255 — the `LumaCoefficients::Bt709` Q8 derivation.
fn bayer_luma(binned_rgb: &[u8]) -> Vec<u8> {
  let (cr, cg, cb) = LUMA_Q8;
  binned_rgb
    .chunks_exact(3)
    .map(|px| {
      let (r, g, b) = (px[0] as u32, px[1] as u32, px[2] as u32);
      ((cr * r + cg * g + cb * b + 128) >> 8).min(255) as u8
    })
    .collect()
}

/// Bayer-way HSV of a binned RGB frame via the public OpenCV kernel.
fn bayer_hsv(binned_rgb: &[u8]) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let n = binned_rgb.len() / 3;
  let (mut h, mut s, mut v) = (vec![0u8; n], vec![0u8; n], vec![0u8; n]);
  crate::row::rgb_to_hsv_row(binned_rgb, &mut h, &mut s, &mut v, n, false);
  (h, s, v)
}

/// Every output of a resampled Bayer frame (`OUT x OUT`).
struct BayerOutputs {
  rgb: Vec<u8>,
  luma: Vec<u8>,
  h: Vec<u8>,
  s: Vec<u8>,
  v: Vec<u8>,
}

/// Run the resampled Bayer sink with rgb + luma + hsv all attached.
fn resample_bayer_all(plane: &[u8], pattern: BayerPattern) -> BayerOutputs {
  let frame = BayerFrame::try_new(plane, SRC as u32, SRC as u32, SRC as u32).unwrap();
  let mut out = BayerOutputs {
    rgb: vec![0u8; OUT * OUT * 3],
    luma: vec![0u8; OUT * OUT],
    h: vec![0u8; OUT * OUT],
    s: vec![0u8; OUT * OUT],
    v: vec![0u8; OUT * OUT],
  };
  {
    let mut sink =
      MixedSinker::<Bayer, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb(&mut out.rgb)
        .unwrap()
        .with_luma(&mut out.luma)
        .unwrap()
        .with_hsv(&mut out.h, &mut out.s, &mut out.v)
        .unwrap();
    bayer_to(
      &frame,
      pattern,
      BayerDemosaic::Bilinear,
      WhiteBalance::neutral(),
      ColorCorrectionMatrix::identity(),
      &mut sink,
    )
    .unwrap();
  }
  out
}

#[test]
fn bayer8_downscale_is_exact_area_mean_of_uniform_demosaic() {
  // A uniform Bayer plane demosaics to a uniform RGB frame, so every
  // 2x2 output block-mean is exactly that uniform pixel — no rounding
  // ambiguity. Pins the bin path end to end.
  let plane = vec![200u8; SRC * SRC];
  let full_rgb = direct_demosaic_rgb(&plane, BayerPattern::Rggb);
  // Uniform input ⇒ uniform demosaic.
  let px0 = [full_rgb[0], full_rgb[1], full_rgb[2]];
  for chunk in full_rgb.chunks_exact(3) {
    assert_eq!([chunk[0], chunk[1], chunk[2]], px0, "demosaic not uniform");
  }
  let out = resample_bayer_all(&plane, BayerPattern::Rggb);
  for block in out.rgb.chunks_exact(3) {
    assert_eq!(
      [block[0], block[1], block[2]],
      px0,
      "binned block != uniform"
    );
  }
}

#[test]
fn bayer8_downscale_block_mean_matches_direct_demosaic() {
  // Non-uniform plane: the binned RGB must equal a direct 2x2
  // round-half-up block mean of the directly-demosaiced RGB.
  let plane = bayer_plane(0x51A9);
  let full_rgb = direct_demosaic_rgb(&plane, BayerPattern::Rggb);
  let want = area_bin_rgb(&full_rgb);
  let out = resample_bayer_all(&plane, BayerPattern::Rggb);
  assert_eq!(out.rgb, want);
}

/// Every supported output of a resampled Bayer frame matches the direct
/// demosaic-then-bin oracle, for **every** CFA pattern the type carries.
#[test]
fn bayer8_resample_all_outputs_match_direct_demosaic_then_bin() {
  for pattern in [
    BayerPattern::Rggb,
    BayerPattern::Bggr,
    BayerPattern::Grbg,
    BayerPattern::Gbrg,
  ] {
    let plane = bayer_plane(0xC0DE ^ pattern as u32);
    let full_rgb = direct_demosaic_rgb(&plane, pattern);
    let binned = area_bin_rgb(&full_rgb);
    let luma_ref = bayer_luma(&binned);
    let (h_ref, s_ref, v_ref) = bayer_hsv(&binned);

    let out = resample_bayer_all(&plane, pattern);
    assert_eq!(out.rgb, binned, "rgb [{pattern}]");
    assert_eq!(out.luma, luma_ref, "luma [{pattern}]");
    assert_eq!(out.h, h_ref, "hsv H [{pattern}]");
    assert_eq!(out.s, s_ref, "hsv S [{pattern}]");
    assert_eq!(out.v, v_ref, "hsv V [{pattern}]");
  }
}

#[test]
fn bayer8_identity_plan_matches_new_sink() {
  // An identity (no-downscale) plan must be byte-identical to the
  // direct `MixedSinker::<Bayer>::new` path for every output.
  let plane = bayer_plane(0x7777);
  let frame = BayerFrame::try_new(&plane, SRC as u32, SRC as u32, SRC as u32).unwrap();

  let mut direct = vec![0u8; SRC * SRC * 3];
  {
    let mut sink = MixedSinker::<Bayer>::new(SRC, SRC)
      .with_rgb(&mut direct)
      .unwrap();
    bayer_to(
      &frame,
      BayerPattern::Grbg,
      BayerDemosaic::Bilinear,
      WhiteBalance::neutral(),
      ColorCorrectionMatrix::identity(),
      &mut sink,
    )
    .unwrap();
  }

  let mut via_area = vec![0u8; SRC * SRC * 3];
  {
    let mut sink =
      MixedSinker::<Bayer, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(SRC, SRC))
        .unwrap()
        .with_rgb(&mut via_area)
        .unwrap();
    bayer_to(
      &frame,
      BayerPattern::Grbg,
      BayerDemosaic::Bilinear,
      WhiteBalance::neutral(),
      ColorCorrectionMatrix::identity(),
      &mut sink,
    )
    .unwrap();
  }
  assert_eq!(direct, via_area);
}

#[test]
fn bayer8_resample_no_outputs_is_a_no_op() {
  // No attached outputs: the resample route freezes the (empty) output
  // set and returns before allocating a stream or enforcing
  // sequencing — so even an out-of-order row is accepted.
  let plane = bayer_plane(0x1234);
  let mut sink =
    MixedSinker::<Bayer, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  let row2 = &plane[SRC * 2..SRC * 3];
  sink
    .process(BayerRow::new(
      row2,
      row2,
      row2,
      2,
      BayerPattern::Rggb,
      BayerDemosaic::Bilinear,
      [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    ))
    .unwrap();
}

#[test]
fn bayer8_resample_rejects_out_of_sequence_before_staging() {
  // An out-of-sequence first row must be rejected before the
  // source-width scratch is grown or the demosaic runs — sequencing is
  // checked ahead of staging. The scratch staying at zero capacity
  // proves no staging allocation happened.
  let plane = bayer_plane(0x2345);
  let mut rgb = vec![0u8; OUT * OUT * 3];
  let mut sink =
    MixedSinker::<Bayer, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_rgb(&mut rgb)
      .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  let row2 = &plane[SRC * 2..SRC * 3];
  let err = sink
    .process(BayerRow::new(
      row2,
      row2,
      row2,
      2,
      BayerPattern::Rggb,
      BayerDemosaic::Bilinear,
      [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    ))
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
fn bayer8_resample_rejects_out_of_sequence_rows() {
  // Skipping row 0 mid-stream is rejected — the stream expects strict
  // sequencing from row 0.
  let plane = bayer_plane(0x3456);
  let mut rgb = vec![0u8; OUT * OUT * 3];
  let mut sink =
    MixedSinker::<Bayer, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_rgb(&mut rgb)
      .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  let row1 = &plane[SRC..SRC * 2];
  let err = sink
    .process(BayerRow::new(
      row1,
      row1,
      row1,
      1,
      BayerPattern::Rggb,
      BayerDemosaic::Bilinear,
      [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    ))
    .unwrap_err();
  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
    ),
    "got {err:?}"
  );
}

/// RFC #238 tail-collapse atomicity (Bayer, scratch OOM): the Bayer8 arm builds
/// its area stream into a LOCAL, grows the source-width demosaic scratch, then
/// inserts the stream + freezes together (keeping its BESPOKE Q8-luma / OpenCV-HSV
/// emit). A scratch-grow OOM AFTER the stream build therefore leaves the stream
/// field `None` AND `resample_outputs` uncommitted, so a row-0 retry with a
/// CHANGED output set is accepted. The DIRECT `rgb_stream`-None assertion is
/// non-vacuous — it FAILS against the R1-buggy insert-before-scratch shape.
#[test]
#[cfg(feature = "std")]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bayer8_first_row_scratch_oom_leaves_stream_and_freeze_uncommitted_for_retry() {
  let plane = bayer_plane(0x7A9E);
  let matrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
  let mut rgb = vec![0u8; OUT * OUT * 3];
  let mut luma = vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Bayer, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb(&mut rgb)
        .unwrap();
    sink.begin_frame(SRC as u32, SRC as u32).unwrap();
    // Arm the source-RGB-scratch grow failpoint: the row-0 stream box builds OK,
    // then the demosaic scratch grow refuses — AllocationFailed with BOTH the
    // stream field `None` and the freeze uncommitted.
    crate::sinker::mixed::arm_source_rgb_scratch_failure();
    let row0 = &plane[0..SRC];
    let err = sink
      .process(BayerRow::new(
        row0,
        row0,
        row0,
        0,
        BayerPattern::Rggb,
        BayerDemosaic::Bilinear,
        matrix,
      ))
      .unwrap_err();
    assert!(
      matches!(
        err,
        MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
      ),
      "first-row Bayer scratch OOM must surface AllocationFailed, got {err:?}"
    );
    // DIRECT stream-rollback proof: the area stream field must be `None` (the fix
    // builds it into a local and inserts only after the scratch grows). Fails
    // against the R1-buggy insert-before-scratch shape.
    assert!(
      !sink.rgb_stream_allocated(),
      "a Bayer scratch OOM must leave the rgb_stream field None (no partial commit)"
    );
    // The freeze was likewise never committed — attaching luma (a CHANGED output
    // set) and replaying from row 0 is ACCEPTED, not ResampleOutputsChanged.
    sink.set_luma(&mut luma).unwrap();
    for r in 0..SRC {
      let above = r.saturating_sub(1);
      let below = (r + 1).min(SRC - 1);
      sink
        .process(BayerRow::new(
          &plane[above * SRC..above * SRC + SRC],
          &plane[r * SRC..r * SRC + SRC],
          &plane[below * SRC..below * SRC + SRC],
          r,
          BayerPattern::Rggb,
          BayerDemosaic::Bilinear,
          matrix,
        ))
        .expect("frame replay after a first-row Bayer scratch OOM must succeed");
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
