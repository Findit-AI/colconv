//! Fused-downscale coverage for the 1-bit bilevel sources `Monoblack`
//! and `Monowhite` through the single-channel luma resample.
//!
//! Both formats are achromatic: each source bit expands to a 0/255 u8
//! luma (Monoblack: bit=1 → 255; Monowhite: bit=0 → 255), and every
//! output is a broadcast of that luma. The resample oracle is therefore
//! "expand the source bits to a 0/255 luma plane, area-bin that plane,
//! then derive each output from the binned luma" — exactly what the
//! sinker does, with the binned mean standing in for the per-pixel
//! expanded value of the direct path.

use crate::{
  ColorMatrix,
  resample::AreaResampler,
  sinker::MixedSinker,
  source::{Monoblack, Monowhite, monoblack_to, monowhite_to},
};
use mediaframe::frame::{MonoblackFrame, MonowhiteFrame};

const SRC_W: usize = 8;
const SRC_H: usize = 4;
const OUT_W: usize = 4;
const OUT_H: usize = 2;
// One byte per row covers all 8 pixels (MSB first).
const STRIDE: usize = SRC_W / 8;

/// Bit pattern (one byte per row) chosen so the 2x2 block means span a
/// spread of values (0, 64, 128, 191, 255) rather than only the
/// saturated endpoints.
///
/// Rows (bit = Monoblack luma 255):
///   row0: 1111_0000
///   row1: 1100_0000
///   row2: 1010_1010
///   row3: 0000_1111
const PATTERN: [u8; SRC_H * STRIDE] = [0b1111_0000, 0b1100_0000, 0b1010_1010, 0b0000_1111];

/// Expand a 1-bit `PATTERN`-shaped buffer to a source-width 0/255 luma
/// plane, applying polarity (`invert` = Monowhite).
fn expand_luma(data: &[u8], invert: bool) -> Vec<u8> {
  let mut out = vec![0u8; SRC_W * SRC_H];
  for row in 0..SRC_H {
    let byte = data[row * STRIDE];
    for col in 0..SRC_W {
      let raw = (byte >> (7 - col)) & 1;
      let bit = if invert { 1 - raw } else { raw };
      out[row * SRC_W + col] = bit * 255;
    }
  }
  out
}

/// Exact 2x2-block area mean (round-half-up) of an `SRC`-grid plane to
/// the `OUT` grid — the integer-ratio (2:1) area-downscale reference.
fn block_mean_2x2(plane: &[u8]) -> Vec<u8> {
  let mut out = vec![0u8; OUT_W * OUT_H];
  for oy in 0..OUT_H {
    for ox in 0..OUT_W {
      let mut s = 0u32;
      for dy in 0..2 {
        for dx in 0..2 {
          s += plane[(oy * 2 + dy) * SRC_W + ox * 2 + dx] as u32;
        }
      }
      out[oy * OUT_W + ox] = ((s + 2) / 4) as u8;
    }
  }
  out
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn monoblack_resample_luma_is_block_mean_of_expanded_bits() {
  let src = MonoblackFrame::try_new(&PATTERN, SRC_W as u32, SRC_H as u32, STRIDE as u32).unwrap();

  let mut luma = vec![0u8; OUT_W * OUT_H];
  {
    let mut sink = MixedSinker::<Monoblack, AreaResampler>::with_resampler(
      SRC_W,
      SRC_H,
      AreaResampler::to(OUT_W, OUT_H),
    )
    .unwrap()
    .with_luma(&mut luma)
    .unwrap();
    monoblack_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();
  }

  // Expanded 0/255 luma plane (Monoblack: bit=1 → 255):
  //   row0: 255 255 255 255   0   0   0   0
  //   row1: 255 255   0   0   0   0   0   0
  //   row2: 255   0 255   0 255   0 255   0
  //   row3:   0   0   0   0 255 255 255 255
  // 2x2 block means (round-half-up), e.g.
  //   out(0,0)=mean(255,255,255,255)=255   out(0,1)=mean(255,255,0,0)=128
  //   out(1,0)=mean(255,255,255,0)... = rows{2,3} cols{0,1} = mean(255,0,0,0)=64
  //   out(1,2)= rows{2,3} cols{4,5}     = mean(255,0,255,255)=191
  let expected = block_mean_2x2(&expand_luma(&PATTERN, false));
  assert_eq!(expected, vec![255, 128, 0, 0, 64, 64, 191, 191]);
  assert_eq!(
    luma, expected,
    "luma must be the 2x2 block mean of the expanded bits"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn monoblack_resample_all_outputs_match_direct_mono_over_binned_luma() {
  let src = MonoblackFrame::try_new(&PATTERN, SRC_W as u32, SRC_H as u32, STRIDE as u32).unwrap();

  let mut luma = vec![0u8; OUT_W * OUT_H];
  let mut luma_u16 = vec![0u16; OUT_W * OUT_H];
  let mut rgb = vec![0u8; OUT_W * OUT_H * 3];
  let mut rgba = vec![0u8; OUT_W * OUT_H * 4];
  let mut rgb_u16 = vec![0u16; OUT_W * OUT_H * 3];
  let mut rgba_u16 = vec![0u16; OUT_W * OUT_H * 4];
  let mut hp = vec![0u8; OUT_W * OUT_H];
  let mut sp = vec![0u8; OUT_W * OUT_H];
  let mut vp = vec![0u8; OUT_W * OUT_H];
  {
    let mut sink = MixedSinker::<Monoblack, AreaResampler>::with_resampler(
      SRC_W,
      SRC_H,
      AreaResampler::to(OUT_W, OUT_H),
    )
    .unwrap()
    .with_luma(&mut luma)
    .unwrap()
    .with_luma_u16(&mut luma_u16)
    .unwrap()
    .with_rgb(&mut rgb)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16)
    .unwrap()
    .with_rgba_u16(&mut rgba_u16)
    .unwrap()
    .with_hsv(&mut hp, &mut sp, &mut vp)
    .unwrap();
    monoblack_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();
  }

  // Oracle: the direct mono path broadcasts each (binned) luma to
  // R=G=B (alpha opaque), copies it to luma, zero-extends to u16, and
  // emits H=0/S=0/V=Y. Apply that to the area-binned luma reference.
  let y_ref = block_mean_2x2(&expand_luma(&PATTERN, false));

  assert_eq!(luma, y_ref);
  let y_ref_u16: Vec<u16> = y_ref.iter().map(|&y| y as u16).collect();
  assert_eq!(luma_u16, y_ref_u16);

  for (px, &y) in rgb.chunks_exact(3).zip(&y_ref) {
    assert_eq!(px, [y, y, y]);
  }
  for (px, &y) in rgba.chunks_exact(4).zip(&y_ref) {
    assert_eq!(px, [y, y, y, 0xFF]);
  }
  for (px, &y) in rgb_u16.chunks_exact(3).zip(&y_ref) {
    let y16 = y as u16;
    assert_eq!(px, [y16, y16, y16]);
  }
  for (px, &y) in rgba_u16.chunks_exact(4).zip(&y_ref) {
    let y16 = y as u16;
    assert_eq!(px, [y16, y16, y16, 0x00FF]);
  }
  assert_eq!(vp, y_ref, "HSV V must be the binned luma");
  assert!(hp.iter().all(|&h| h == 0), "HSV H must be 0 (achromatic)");
  assert!(sp.iter().all(|&s| s == 0), "HSV S must be 0 (achromatic)");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn monowhite_resample_all_outputs_match_direct_mono_over_binned_luma() {
  let src = MonowhiteFrame::try_new(&PATTERN, SRC_W as u32, SRC_H as u32, STRIDE as u32).unwrap();

  let mut luma = vec![0u8; OUT_W * OUT_H];
  let mut rgb = vec![0u8; OUT_W * OUT_H * 3];
  let mut rgba = vec![0u8; OUT_W * OUT_H * 4];
  {
    let mut sink = MixedSinker::<Monowhite, AreaResampler>::with_resampler(
      SRC_W,
      SRC_H,
      AreaResampler::to(OUT_W, OUT_H),
    )
    .unwrap()
    .with_luma(&mut luma)
    .unwrap()
    .with_rgb(&mut rgb)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap();
    monowhite_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();
  }

  // Monowhite inverts polarity: the expanded plane is the complement of
  // Monoblack's, so the binned reference differs.
  let y_ref = block_mean_2x2(&expand_luma(&PATTERN, true));
  assert_eq!(luma, y_ref);
  for (px, &y) in rgb.chunks_exact(3).zip(&y_ref) {
    assert_eq!(px, [y, y, y]);
  }
  for (px, &y) in rgba.chunks_exact(4).zip(&y_ref) {
    assert_eq!(px, [y, y, y, 0xFF]);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn monoblack_identity_plan_matches_new_sink() {
  let src = MonoblackFrame::try_new(&PATTERN, SRC_W as u32, SRC_H as u32, STRIDE as u32).unwrap();

  let mut direct_rgb = vec![0u8; SRC_W * SRC_H * 3];
  let mut direct_rgba = vec![0u8; SRC_W * SRC_H * 4];
  {
    let mut sink = MixedSinker::<Monoblack>::new(SRC_W, SRC_H)
      .with_rgb(&mut direct_rgb)
      .unwrap()
      .with_rgba(&mut direct_rgba)
      .unwrap();
    monoblack_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();
  }

  let mut area_rgb = vec![0u8; SRC_W * SRC_H * 3];
  let mut area_rgba = vec![0u8; SRC_W * SRC_H * 4];
  {
    // An identity (no-op) resampler plan must take the direct path and
    // stay byte-identical.
    let mut sink = MixedSinker::<Monoblack, AreaResampler>::with_resampler(
      SRC_W,
      SRC_H,
      AreaResampler::to(SRC_W, SRC_H),
    )
    .unwrap()
    .with_rgb(&mut area_rgb)
    .unwrap()
    .with_rgba(&mut area_rgba)
    .unwrap();
    monoblack_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();
  }

  assert_eq!(
    direct_rgb, area_rgb,
    "identity rgb must match the direct path"
  );
  assert_eq!(
    direct_rgba, area_rgba,
    "identity rgba must match the direct path"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn monoblack_resample_reuses_luma_stream_across_frames() {
  // A reused sink must reset the luma stream each frame; without the
  // reset, frame 2's row 0 is rejected as out-of-sequence and the luma
  // never reflects frame 2.
  let frame1 = PATTERN;
  let frame2: [u8; SRC_H * STRIDE] = [!PATTERN[0], !PATTERN[1], !PATTERN[2], !PATTERN[3]];

  let mut luma = vec![0u8; OUT_W * OUT_H];
  {
    let mut sink = MixedSinker::<Monoblack, AreaResampler>::with_resampler(
      SRC_W,
      SRC_H,
      AreaResampler::to(OUT_W, OUT_H),
    )
    .unwrap()
    .with_luma(&mut luma)
    .unwrap();
    let f1 = MonoblackFrame::try_new(&frame1, SRC_W as u32, SRC_H as u32, STRIDE as u32).unwrap();
    monoblack_to(&f1, true, ColorMatrix::Bt709, &mut sink).unwrap();
    let f2 = MonoblackFrame::try_new(&frame2, SRC_W as u32, SRC_H as u32, STRIDE as u32).unwrap();
    monoblack_to(&f2, true, ColorMatrix::Bt709, &mut sink).unwrap();
  }

  let y2_ref = block_mean_2x2(&expand_luma(&frame2, false));
  assert_eq!(
    luma, y2_ref,
    "frame 2 luma must area-downscale frame 2's bits"
  );
}

#[test]
fn monoblack_resample_no_outputs_is_a_no_op() {
  let src = MonoblackFrame::try_new(&PATTERN, SRC_W as u32, SRC_H as u32, STRIDE as u32).unwrap();
  let mut sink = MixedSinker::<Monoblack, AreaResampler>::with_resampler(
    SRC_W,
    SRC_H,
    AreaResampler::to(OUT_W, OUT_H),
  )
  .unwrap();
  // No outputs attached: a legal no-op, accepted without error. The
  // sequence guard's no-output branch returns before any stream is
  // created, so the luma stream is never allocated.
  monoblack_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();
  assert!(
    !sink.luma_stream_allocated(),
    "a no-output frame must not allocate the luma stream"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn monoblack_resample_begin_frame_resets_stream_sequencing() {
  // `MonoblackRow::new` is crate-private to mediaframe, so a misordered
  // first row cannot be hand-fed to `process` — the out-of-sequence
  // *rejection* (OutOfSequenceRow), and the conditional-ordering
  // guarantee that a rejected first row stores no frozen-output snapshot
  // that would poison a row-0 retry, are reachable only through a public
  // row ctor (the macro-generated formats, e.g. the Yuv411p suite, which
  // shares `mono_luma_resample`'s sequence-then-freeze body verbatim; the
  // `..._rejected_first_row_does_not_poison_output_retry` tests on the
  // single-stream gray paths cover the identical fixed ordering). What is
  // reachable for mono is the positive guarantee the rejection protects:
  // after a full frame advances the stream to `next_y == SRC_H`,
  // `begin_frame` must reset it so the next frame's row 0 is back in
  // sequence rather than rejected.
  let f1 = MonoblackFrame::try_new(&PATTERN, SRC_W as u32, SRC_H as u32, STRIDE as u32).unwrap();
  let mut luma = vec![0u8; OUT_W * OUT_H];
  let mut sink = MixedSinker::<Monoblack, AreaResampler>::with_resampler(
    SRC_W,
    SRC_H,
    AreaResampler::to(OUT_W, OUT_H),
  )
  .unwrap()
  .with_luma(&mut luma)
  .unwrap();
  monoblack_to(&f1, true, ColorMatrix::Bt709, &mut sink).unwrap();
  assert!(
    sink.luma_stream_allocated(),
    "the first frame must have created the luma stream"
  );
  // A second walk re-enters begin_frame (reset to next_y == 0) and must
  // succeed in strict order on the SAME stream.
  monoblack_to(&f1, true, ColorMatrix::Bt709, &mut sink).unwrap();
  let y_ref = block_mean_2x2(&expand_luma(&PATTERN, false));
  assert_eq!(
    luma, y_ref,
    "frame 2 must re-bin after the begin_frame reset"
  );
}

// The source-width `u8` expanded-luma staging scratch — grown for every attached
// output (mono expands each 1-bit row to a source-width 0/255 luma plane before
// binning), pregrown BEFORE the freeze via the transactional preflight's build-local
// / grow-scratch / commit-together shape.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn mono_first_row_scratch_oom_leaves_stream_uncommitted_for_retry() {
  let src = MonoblackFrame::try_new(&PATTERN, SRC_W as u32, SRC_H as u32, STRIDE as u32).unwrap();
  let mut luma = vec![0u8; OUT_W * OUT_H];
  let mut rgb = vec![0u8; OUT_W * OUT_H * 3];
  {
    let mut sink = MixedSinker::<Monoblack, AreaResampler>::with_resampler(
      SRC_W,
      SRC_H,
      AreaResampler::to(OUT_W, OUT_H),
    )
    .unwrap()
    .with_luma(&mut luma)
    .unwrap();
    // Arm the source-luma scratch grow failpoint: the row-0 stream box builds OK (into
    // a local) and the expanded-luma staging grow — pregrown BEFORE the field insert +
    // freeze — refuses, surfacing AllocationFailed with the luma stream field `None`
    // (the commit-together atomic shape; the insert-before-scratch shape would have
    // left the stream committed).
    crate::sinker::mixed::arm_source_luma_scratch_failure();
    let err = monoblack_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap_err();
    assert!(
      matches!(
        err,
        crate::sinker::MixedSinkerError::Resample(
          crate::resample::ResampleError::AllocationFailed(_)
        )
      ),
      "first-row scratch OOM must surface AllocationFailed, got {err:?}"
    );
    // DIRECT stream-rollback proof (the non-vacuous part): the luma stream field must
    // be `None` — the fix builds it into a LOCAL and inserts only after the scratch
    // grows, so a failed scratch leaves nothing committed. The insert-before-scratch
    // shape would leave it `Some` here, so this assertion FAILS against that shape and
    // PASSES against the fix.
    assert!(
      !sink.luma_stream_allocated(),
      "a scratch OOM must leave the luma stream field None (no partial commit)"
    );
    // The frame is recoverable: attaching rgb (a CHANGED output set) and re-walking
    // (begin_frame re-sequences from row 0) succeeds and drives both outputs.
    sink.set_rgb(&mut rgb).unwrap();
    monoblack_to(&src, true, ColorMatrix::Bt709, &mut sink)
      .expect("re-walk after a first-row scratch OOM must succeed");
  }
  assert!(
    luma.iter().any(|&b| b != 0),
    "the recovered frame produced real luma output (scratch OOM was recoverable)"
  );
  assert!(
    rgb.iter().any(|&b| b != 0),
    "the newly-attached rgb output was driven on the accepted re-walk"
  );
}

// ---- Separable-filter resample ----------------------------------------
//
// A 1-bit bilevel image filtered to continuous grayscale *is* antialiasing
// it: the area path expands each bit to a 0/255 u8 luma and bins that single
// plane; the filter path expands identically and resamples the same plane
// through the signed-coefficient single-channel `FilterStream<u8>` (the
// filter twin of the bin). So the filter `luma` must equal a single-channel
// `FilterStream<u8>` resample of the same expanded 0/255 plane **byte for
// bit** (same engine, same coefficients, full-range u8 so no clamp on
// either), and every derived output (luma_u16 / rgb / rgba / rgb_u16 /
// rgba_u16 / hsv) must follow from that resampled luma exactly as the area
// path derives from its binned luma. A hard 0/255 edge resampled with a
// Triangle / Catmull-Rom / Lanczos window becomes a smooth gray ramp — the
// intermediate grays are the antialiasing.

use crate::resample::{
  CatmullRom, FilterKernel, FilterStream, FilteredResampler, Lanczos3, Resampler, Triangle,
};

/// Source dimensions for the filter cases — a larger, full-byte-wide grid
/// than the 2:1 area fixture so a downscale (`FW`->`FOUT_DOWN`) and an
/// upscale (`FOUT_DOWN`->`FUP`) both run real, non-trivial windows.
const FW: usize = 8;
const FH: usize = 8;
const FSTRIDE: usize = FW / 8;
const FOUT_DOWN: usize = 4;
const FUP: usize = 7;

/// A bilevel pattern with a clean vertical edge in every row (left half set,
/// right half clear) plus a couple of textured rows, so a horizontal filter
/// window straddling the mid-column produces intermediate grays (the
/// antialiasing) rather than only the saturated endpoints. One byte per row.
///
/// Rows (bit = Monoblack luma 255):
///   0..3: 1111_0000   (a hard mid-column edge)
///   4:    1100_1100
///   5:    1010_1010
///   6,7:  1111_0000
const FPATTERN: [u8; FH * FSTRIDE] = [
  0b1111_0000,
  0b1111_0000,
  0b1111_0000,
  0b1111_0000,
  0b1100_1100,
  0b1010_1010,
  0b1111_0000,
  0b1111_0000,
];

/// Expand an `FPATTERN`-shaped 1-bit buffer (`FW x FH`) to a source-width
/// 0/255 luma plane, applying polarity (`invert` = Monowhite) — the same
/// expansion the mono filter path stages before feeding its single-channel
/// stream.
fn expand_luma_grid(data: &[u8], invert: bool) -> Vec<u8> {
  let mut out = vec![0u8; FW * FH];
  for row in 0..FH {
    let byte = data[row * FSTRIDE];
    for col in 0..FW {
      let raw = (byte >> (7 - col)) & 1;
      let bit = if invert { 1 - raw } else { raw };
      out[row * FW + col] = bit * 255;
    }
  }
  out
}

/// Single-channel filter resample of a u8 luma plane via the merged engine's
/// [`FilterStream<u8>`] (channels = 1) — the mono luma oracle. A mono filter
/// path's `luma` must equal this **byte for bit** (same engine, same
/// coefficients, the expanded 0/255 plane resampled directly). Full-range
/// u8, so no native-depth clamp.
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

/// Every resampled output a mono filter equivalence asserts on.
struct MonoFilterOutputs {
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

/// Run a `Monoblack` filter sink over `FPATTERN` at `ow x oh` under
/// `kernel`, attaching every output the equivalence asserts on.
fn monoblack_filter_outputs<K: FilterKernel + Copy>(
  ow: usize,
  oh: usize,
  kernel: K,
) -> MonoFilterOutputs {
  let src = MonoblackFrame::try_new(&FPATTERN, FW as u32, FH as u32, FSTRIDE as u32).unwrap();
  let mut o = MonoFilterOutputs {
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
    let mut sink = MixedSinker::<Monoblack, FilteredResampler<K>>::with_resampler(
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
    monoblack_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();
  }
  o
}

/// Asserts a `Monoblack` filter resample's every output is derived from the
/// single-channel native-luma oracle exactly as the area emit derives from
/// its binned luma, and returns the max per-sample `luma` diff (exactly 0 —
/// same engine, no clamp).
fn assert_monoblack_filter_matches_oracle<K: FilterKernel + Copy>(
  kernel: K,
  ow: usize,
  oh: usize,
  ctx: &str,
) -> u8 {
  let got = monoblack_filter_outputs(ow, oh, kernel);
  let y_ref = native_luma_filter(kernel, &expand_luma_grid(&FPATTERN, false), FW, FH, ow, oh);

  let mut max_diff = 0u8;
  for (i, (&g, &w)) in got.luma.iter().zip(y_ref.iter()).enumerate() {
    max_diff = max_diff.max(g.abs_diff(w));
    assert_eq!(
      g, w,
      "{ctx} luma[{i}]: {g} vs single-channel native-luma filter {w}"
    );
  }
  // Every derived output mirrors the area emit applied to the resampled luma.
  let y_ref_u16: Vec<u16> = y_ref.iter().map(|&y| y as u16).collect();
  assert_eq!(
    got.luma_u16, y_ref_u16,
    "{ctx} luma_u16 = resampled luma zero-extended"
  );
  for (px, &y) in got.rgb.chunks_exact(3).zip(&y_ref) {
    assert_eq!(px, [y, y, y], "{ctx} rgb = broadcast luma");
  }
  for (px, &y) in got.rgba.chunks_exact(4).zip(&y_ref) {
    assert_eq!(px, [y, y, y, 0xFF], "{ctx} rgba = broadcast luma, opaque");
  }
  for (px, &y) in got.rgb_u16.chunks_exact(3).zip(&y_ref) {
    let y16 = y as u16;
    assert_eq!(px, [y16, y16, y16], "{ctx} rgb_u16 = broadcast luma");
  }
  for (px, &y) in got.rgba_u16.chunks_exact(4).zip(&y_ref) {
    let y16 = y as u16;
    assert_eq!(
      px,
      [y16, y16, y16, 0x00FF],
      "{ctx} rgba_u16 = broadcast luma, opaque"
    );
  }
  assert_eq!(got.vp, y_ref, "{ctx} HSV V = resampled luma");
  assert!(
    got.hp.iter().all(|&h| h == 0),
    "{ctx} HSV H = 0 (achromatic)"
  );
  assert!(
    got.sp.iter().all(|&s| s == 0),
    "{ctx} HSV S = 0 (achromatic)"
  );
  max_diff
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn monoblack_filter_luma_is_single_channel_native_luma() {
  // Downscale 8 -> 4 and upscale 4 -> 7, every kernel; luma must be the
  // single-channel native-luma filter of the expanded 0/255 plane (max diff
  // 0), and every derived output follows from it.
  assert_eq!(
    0,
    assert_monoblack_filter_matches_oracle(
      Triangle,
      FOUT_DOWN,
      FOUT_DOWN,
      "monoblack triangle down"
    )
  );
  assert_eq!(
    0,
    assert_monoblack_filter_matches_oracle(
      CatmullRom,
      FOUT_DOWN,
      FOUT_DOWN,
      "monoblack catmullrom down"
    )
  );
  assert_eq!(
    0,
    assert_monoblack_filter_matches_oracle(
      Lanczos3,
      FOUT_DOWN,
      FOUT_DOWN,
      "monoblack lanczos3 down"
    )
  );
  assert_eq!(
    0,
    assert_monoblack_filter_matches_oracle(Triangle, FUP, FUP, "monoblack triangle up")
  );
  assert_eq!(
    0,
    assert_monoblack_filter_matches_oracle(CatmullRom, FUP, FUP, "monoblack catmullrom up")
  );
  assert_eq!(
    0,
    assert_monoblack_filter_matches_oracle(Lanczos3, FUP, FUP, "monoblack lanczos3 up")
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn monowhite_filter_luma_is_single_channel_native_luma() {
  // Monowhite inverts polarity: the expanded plane is the complement of
  // Monoblack's, so the resampled reference differs. luma must still be the
  // single-channel native-luma filter of *that* plane.
  let src = MonowhiteFrame::try_new(&FPATTERN, FW as u32, FH as u32, FSTRIDE as u32).unwrap();
  for (kernel_name, run) in [("triangle down", FOUT_DOWN), ("triangle up", FUP)] {
    let (ow, oh) = (run, run);
    let mut luma = vec![0u8; ow * oh];
    let mut rgb = vec![0u8; ow * oh * 3];
    let mut rgba = vec![0u8; ow * oh * 4];
    {
      let mut sink = MixedSinker::<Monowhite, FilteredResampler<Triangle>>::with_resampler(
        FW,
        FH,
        FilteredResampler::new(ow, oh, Triangle),
      )
      .unwrap()
      .with_luma(&mut luma)
      .unwrap()
      .with_rgb(&mut rgb)
      .unwrap()
      .with_rgba(&mut rgba)
      .unwrap();
      monowhite_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();
    }
    let y_ref = native_luma_filter(Triangle, &expand_luma_grid(&FPATTERN, true), FW, FH, ow, oh);
    assert_eq!(
      luma, y_ref,
      "monowhite {kernel_name}: luma = native-luma filter of inverted plane"
    );
    for (px, &y) in rgb.chunks_exact(3).zip(&y_ref) {
      assert_eq!(
        px,
        [y, y, y],
        "monowhite {kernel_name}: rgb = broadcast luma"
      );
    }
    for (px, &y) in rgba.chunks_exact(4).zip(&y_ref) {
      assert_eq!(
        px,
        [y, y, y, 0xFF],
        "monowhite {kernel_name}: rgba = broadcast luma, opaque"
      );
    }
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn monoblack_filter_antialiases_bilevel_edge_to_intermediate_grays() {
  // The whole point of filtering a bilevel image: a hard 0/255 edge must
  // resample to a smooth gray ramp. Downscale the mid-column-edge pattern
  // and assert the luma contains values strictly between 0 and 255 — the
  // antialiasing the area/bin path also produces, here via the filter
  // kernel. (A filter plan that was rejected, or one that snapped to 0/255,
  // would fail this.)
  let got = monoblack_filter_outputs(FOUT_DOWN, FOUT_DOWN, Triangle);
  assert!(
    got.luma.iter().any(|&y| y > 0 && y < 255),
    "a filtered bilevel edge must yield intermediate grays (antialiasing); got {:?}",
    got.luma
  );
  // And those grays broadcast into the RGB outputs (still achromatic).
  assert!(
    got
      .rgb
      .chunks_exact(3)
      .any(|px| px[0] > 0 && px[0] < 255 && px[0] == px[1] && px[1] == px[2]),
    "intermediate grays must broadcast to R=G=B"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn mono_filter_plans_are_accepted() {
  // Before this routing a filter plan was rejected with `UnsupportedFilter`;
  // now both mono variants produce real (non-sentinel) populated output.
  let got = monoblack_filter_outputs(FOUT_DOWN, FOUT_DOWN, Triangle);
  assert!(
    got.luma.iter().any(|&v| v != 0),
    "monoblack: filter resample must populate luma (no UnsupportedFilter)"
  );
  assert!(
    got.rgb.iter().any(|&v| v != 0),
    "monoblack: filter resample must populate rgb (no UnsupportedFilter)"
  );

  let src = MonowhiteFrame::try_new(&FPATTERN, FW as u32, FH as u32, FSTRIDE as u32).unwrap();
  let mut luma = vec![0u8; FOUT_DOWN * FOUT_DOWN];
  {
    let mut sink = MixedSinker::<Monowhite, FilteredResampler<Triangle>>::with_resampler(
      FW,
      FH,
      FilteredResampler::new(FOUT_DOWN, FOUT_DOWN, Triangle),
    )
    .unwrap()
    .with_luma(&mut luma)
    .unwrap();
    monowhite_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();
  }
  assert!(
    luma.iter().any(|&v| v != 0),
    "monowhite: filter resample must populate luma (no UnsupportedFilter)"
  );
}

#[test]
fn mono_filter_no_outputs_is_a_no_op_without_allocating() {
  // A no-output filter sink stays a legal no-op: the sequence guard's
  // no-output branch returns before any stream is created, so the
  // single-channel filter stream is never allocated.
  let src = MonoblackFrame::try_new(&FPATTERN, FW as u32, FH as u32, FSTRIDE as u32).unwrap();
  let mut sink = MixedSinker::<Monoblack, FilteredResampler<Triangle>>::with_resampler(
    FW,
    FH,
    FilteredResampler::new(FOUT_DOWN, FOUT_DOWN, Triangle),
  )
  .unwrap();
  monoblack_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();
  assert!(
    !sink.luma_filter_stream_allocated(),
    "a no-output filter frame must not allocate the filter stream"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn monoblack_filter_reuses_stream_across_frames() {
  // A reused filter sink must reset its filter stream each frame, else frame
  // 2's row 0 is rejected as out-of-sequence (the filter twin of the area
  // `*_reuses_luma_stream_across_frames` coverage). Confirms the
  // `begin_frame` reset of `luma_filter_stream`.
  let frame1 = FPATTERN;
  let frame2: [u8; FH * FSTRIDE] = {
    let mut f = FPATTERN;
    for b in f.iter_mut() {
      *b = !*b;
    }
    f
  };
  let mut luma = vec![0u8; FOUT_DOWN * FOUT_DOWN];
  {
    let mut sink = MixedSinker::<Monoblack, FilteredResampler<Triangle>>::with_resampler(
      FW,
      FH,
      FilteredResampler::new(FOUT_DOWN, FOUT_DOWN, Triangle),
    )
    .unwrap()
    .with_luma(&mut luma)
    .unwrap();
    let f1 = MonoblackFrame::try_new(&frame1, FW as u32, FH as u32, FSTRIDE as u32).unwrap();
    monoblack_to(&f1, true, ColorMatrix::Bt709, &mut sink).unwrap();
    let f2 = MonoblackFrame::try_new(&frame2, FW as u32, FH as u32, FSTRIDE as u32).unwrap();
    monoblack_to(&f2, true, ColorMatrix::Bt709, &mut sink).unwrap();
  }
  let want = native_luma_filter(
    Triangle,
    &expand_luma_grid(&frame2, false),
    FW,
    FH,
    FOUT_DOWN,
    FOUT_DOWN,
  );
  assert_eq!(
    luma, want,
    "frame 2 luma must be the native-luma filter of frame 2's bits (stream reset each frame)"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn monoblack_filter_identity_plan_matches_new_sink() {
  // An identity (no-op) filter plan must still take the direct path and stay
  // byte-identical to a fresh direct sink — the filter routing must not
  // perturb the identity case.
  let src = MonoblackFrame::try_new(&FPATTERN, FW as u32, FH as u32, FSTRIDE as u32).unwrap();
  let mut direct_rgb = vec![0u8; FW * FH * 3];
  {
    let mut sink = MixedSinker::<Monoblack>::new(FW, FH)
      .with_rgb(&mut direct_rgb)
      .unwrap();
    monoblack_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();
  }
  let mut filt_rgb = vec![0u8; FW * FH * 3];
  {
    let mut sink = MixedSinker::<Monoblack, FilteredResampler<Triangle>>::with_resampler(
      FW,
      FH,
      FilteredResampler::new(FW, FH, Triangle),
    )
    .unwrap()
    .with_rgb(&mut filt_rgb)
    .unwrap();
    monoblack_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();
  }
  assert_eq!(
    direct_rgb, filt_rgb,
    "identity filter rgb must match the direct path"
  );
}
