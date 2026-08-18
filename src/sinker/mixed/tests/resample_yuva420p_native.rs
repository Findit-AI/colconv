//! RFC #238 Phase 5 — STRAIGHT-alpha NATIVE fast tier for the 8-bit planar
//! 4:2:0 YUV-with-alpha source `Yuva420p` (the #235 alpha resolution).
//!
//! The native campaign excluded every alpha format because PREMULTIPLIED
//! alpha is bin-then-convert-incompatible (colour has been multiplied by
//! α). But STRAIGHT alpha — colour independent of α — IS compatible: bin
//! Y / U / V / A independently, convert Y / U / V → RGB once per output
//! pixel, attach the binned A → straight RGBA. This is the alpha-bearing
//! sibling of
//! [`yuv420p_process_native`](crate::sinker::mixed::planar_8bit::yuv420p_process_native):
//! Y / U / V bin and convert exactly as the no-alpha native tier (so rgb /
//! luma / hsv are byte-identical to it), and the alpha plane bins on the
//! LUMA grid through its own `AreaStream<u8>` (α is full-resolution in
//! Yuva420p, like Y), substituted into the RGBA output's α slot.
//!
//! Native-eligibility is STRAIGHT-only: `AlphaMode::Premultiplied` is NOT
//! native-eligible (it must convert-at-source, premultiply, bin, then
//! un-premultiply late — mathematically required), so a premultiplied
//! Yuva420p sink stays on the existing packed-YUVA area tail
//! ([`packed_yuva444_resample`](super::super::packed_yuva444_resample))
//! BYTE-IDENTICALLY, and the `with_native` flag does not change it.
//!
//! The native tier averages in the YUV domain then converts; the row-stage
//! tier converts then averages in RGB, so the colour outputs are NOT
//! byte-identical across tiers — only within a small in-gamut tolerance
//! (luma / alpha are bit-identical: both bin the same native plane).

use crate::{
  KernelMatrix, PixelSink,
  resample::{AreaResampler, ResampleError},
  sinker::{AlphaMode, MixedSinker, MixedSinkerError},
  source::{Yuva420p, Yuva420pRow, yuva420p_to},
};
use mediaframe::frame::{Yuva420pFrame, Yuva444pFrame};

const SRC: usize = 8;
const OUT: usize = 4;
const CW: usize = SRC / 2; // chroma width (half)
const CH: usize = SRC / 2; // chroma height (half)
const M: KernelMatrix = KernelMatrix::Bt709;
const FR: bool = true;

/// In-gamut per-channel tolerance between the native (average-in-YUV) and
/// row-stage (convert-then-average) colour outputs. The two average in
/// different domains and round independently per output pixel; this bound
/// only documents the in-gamut gap — the byte-exact correctness check is
/// `straight_native_equals_independent_oracle` against the bin-then-convert
/// oracle.
const TOL_U8: u8 = 5;

/// Mid-range Y + chroma ramp with a varying α ramp — every YUV code in
/// gamut, so the native-vs-rowstage colour delta is the per-pixel rounding
/// difference (convert-order), not an out-of-gamut clamp divergence. Used by
/// the tolerance check; α still varies so the binned α is a real area mean.
fn ramp() -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
  let mut y = std::vec![0u8; SRC * SRC];
  let mut u = std::vec![0u8; CW * CH];
  let mut v = std::vec![0u8; CW * CH];
  let mut a = std::vec![0u8; SRC * SRC];
  for (i, p) in y.iter_mut().enumerate() {
    *p = 40 + ((i as u32 * 3) % 160) as u8;
  }
  for (i, p) in u.iter_mut().enumerate() {
    *p = 100 + ((i as u32 * 7) % 56) as u8;
  }
  for (i, p) in v.iter_mut().enumerate() {
    *p = 150 - ((i as u32 * 5) % 56) as u8;
  }
  for (i, p) in a.iter_mut().enumerate() {
    *p = 20 + ((i as u32 * 11) % 220) as u8;
  }
  (y, u, v, a)
}

/// Pseudo-random Y / U / V / A planes; alpha varies (not all-opaque) so the
/// binned α is a genuine area mean. Y / A are full-resolution, U / V are
/// half x half.
fn planes(seed: u32) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
  let mut y = std::vec![0u8; SRC * SRC];
  let mut u = std::vec![0u8; CW * CH];
  let mut v = std::vec![0u8; CW * CH];
  let mut a = std::vec![0u8; SRC * SRC];
  super::pseudo_random_u8(&mut y, seed);
  super::pseudo_random_u8(&mut u, seed ^ 0x1111_1111);
  super::pseudo_random_u8(&mut v, seed ^ 0x2222_2222);
  super::pseudo_random_u8(&mut a, seed ^ 0x3333_3333);
  (y, u, v, a)
}

fn frame<'a>(y: &'a [u8], u: &'a [u8], v: &'a [u8], a: &'a [u8]) -> Yuva420pFrame<'a> {
  Yuva420pFrame::try_new(
    y, u, v, a, SRC as u32, SRC as u32, SRC as u32, CW as u32, CW as u32, SRC as u32,
  )
  .unwrap()
}

/// Round-half-up integer area mean of an `in_w x in_h` u8 plane down to
/// `OUT x OUT`, binning each axis by its own ratio. Y / A bin 2:1x2:1 from
/// `SRC x SRC`; U / V bin 2:1x2:1 from `CW x CH` (4:2:0). Reproduces the
/// native tier's per-plane binning to full output resolution.
fn bin_to_out(plane: &[u8], in_w: usize, in_h: usize) -> Vec<u8> {
  let (rx, ry) = (in_w / OUT, in_h / OUT);
  let denom = (rx * ry) as u32;
  let mut out = std::vec![0u8; OUT * OUT];
  for oy in 0..OUT {
    for ox in 0..OUT {
      let mut s = 0u32;
      for dy in 0..ry {
        for dx in 0..rx {
          s += plane[(oy * ry + dy) * in_w + ox * rx + dx] as u32;
        }
      }
      out[oy * OUT + ox] = ((s + denom / 2) / denom) as u8;
    }
  }
  out
}

/// Drive the `Yuva420p` resample for the full output set. `native` toggles
/// the straight-alpha native fast tier vs the row-stage (packed-YUVA area
/// tail) tier. `mode` is the alpha interpretation.
#[allow(clippy::type_complexity)]
fn run(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  a: &[u8],
  native: bool,
  mode: AlphaMode,
) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u16>) {
  let mut rgb = std::vec![0u8; OUT * OUT * 3];
  let mut rgba = std::vec![0u8; OUT * OUT * 4];
  let mut luma = std::vec![0u8; OUT * OUT];
  let mut lu16 = std::vec![0u16; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Yuva420p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_native(native)
        .with_alpha_mode(mode)
        .with_rgb(&mut rgb)
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap()
        .with_luma(&mut luma)
        .unwrap()
        .with_luma_u16(&mut lu16)
        .unwrap();
    yuva420p_to(&frame(y, u, v, a), FR, M, &mut sink).unwrap();
  }
  (rgb, rgba, luma, lu16)
}

/// The INDEPENDENT bin-codes-then-convert oracle for the STRAIGHT-alpha
/// native tier: area-bin Y / U / V / A each to OUTPUT resolution by its own
/// subsample ratio, then convert the full-output-width binned planes ONCE
/// through an identity `Yuva444p` sink (4:4:4 convert with the binned α as
/// straight alpha). This is the exact ground truth the native tier must
/// reproduce — RGB from the YUV convert, α from the independent native α
/// bin — derived with NO reference to the row-stage tier.
#[allow(clippy::type_complexity)]
fn oracle(y: &[u8], u: &[u8], v: &[u8], a: &[u8]) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u16>) {
  let yb = bin_to_out(y, SRC, SRC);
  let ub = bin_to_out(u, CW, CH);
  let vb = bin_to_out(v, CW, CH);
  let ab = bin_to_out(a, SRC, SRC);
  let mut rgb = std::vec![0u8; OUT * OUT * 3];
  let mut rgba = std::vec![0u8; OUT * OUT * 4];
  let mut luma = std::vec![0u8; OUT * OUT];
  let mut lu16 = std::vec![0u16; OUT * OUT];
  {
    let mut sink = MixedSinker::<crate::source::Yuva444p>::new(OUT, OUT)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_rgba(&mut rgba)
      .unwrap()
      .with_luma(&mut luma)
      .unwrap()
      .with_luma_u16(&mut lu16)
      .unwrap();
    let f = Yuva444pFrame::try_new(
      &yb, &ub, &vb, &ab, OUT as u32, OUT as u32, OUT as u32, OUT as u32, OUT as u32, OUT as u32,
    )
    .unwrap();
    crate::source::yuva444p_to(&f, FR, M, &mut sink).unwrap();
  }
  (rgb, rgba, luma, lu16)
}

fn max_delta(a: &[u8], b: &[u8]) -> u8 {
  a.iter()
    .zip(b)
    .map(|(&x, &y)| x.abs_diff(y))
    .max()
    .unwrap_or(0)
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn straight_alpha_native_equals_independent_oracle() {
  // Ground truth: the straight-alpha native tier IS bin-Y/U/V/A-then-convert.
  // Every output must match the INDEPENDENT oracle exactly — RGB from the
  // YUV convert, α from the independent native α bin, luma the native Y bin.
  for &seed in &[0x51A1u32, 0xBEEF, 0x0F0F, 0x77AA] {
    let (y, u, v, a) = planes(seed);
    let n = run(&y, &u, &v, &a, true, AlphaMode::Straight);
    let o = oracle(&y, &u, &v, &a);
    assert_eq!(
      n.0, o.0,
      "rgb must equal the bin-then-convert oracle (seed {seed:#x})"
    );
    assert_eq!(
      n.1, o.1,
      "straight rgba must equal the bin-then-convert oracle (seed {seed:#x})"
    );
    assert_eq!(
      n.2, o.2,
      "luma must equal the binned native Y (seed {seed:#x})"
    );
    assert_eq!(
      n.3, o.3,
      "luma_u16 must equal the binned native Y (seed {seed:#x})"
    );
    // The α slot of the native RGBA is the independent native α bin.
    let ab = bin_to_out(&a, SRC, SRC);
    let native_alpha: Vec<u8> = n.1.chunks_exact(4).map(|px| px[3]).collect();
    assert_eq!(
      native_alpha, ab,
      "native RGBA α must equal the native α bin (seed {seed:#x})"
    );
    assert!(
      n.1.chunks_exact(4).any(|px| px[3] != 0xFF),
      "native α was forced opaque — area-mean alpha lost (seed {seed:#x})"
    );
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn straight_native_vs_rowstage_within_tol() {
  // The row-stage tier (convert-then-bin in RGB) is the cv2 INTER_AREA
  // oracle; the native tier (bin-then-convert in YUV) tracks it within a
  // small in-gamut tolerance. Luma AND alpha are bit-identical (both bin the
  // same native plane). An in-gamut ramp fixture is required: random chroma
  // straddles the RGB clamp, where the two averaging domains diverge
  // unboundedly (the documented out-of-gamut behaviour).
  let (y, u, v, a) = ramp();
  let (n_rgb, n_rgba, n_luma, n_lu16) = run(&y, &u, &v, &a, true, AlphaMode::Straight);
  let (r_rgb, r_rgba, r_luma, r_lu16) = run(&y, &u, &v, &a, false, AlphaMode::Straight);

  assert_eq!(n_luma, r_luma, "luma must be bit-identical across tiers");
  assert_eq!(
    n_lu16, r_lu16,
    "luma_u16 must be bit-identical across tiers"
  );
  // Alpha (the 4th RGBA channel) is the same native-plane area bin in both.
  let n_alpha: Vec<u8> = n_rgba.chunks_exact(4).map(|px| px[3]).collect();
  let r_alpha: Vec<u8> = r_rgba.chunks_exact(4).map(|px| px[3]).collect();
  assert_eq!(n_alpha, r_alpha, "alpha must be bit-identical across tiers");

  let d_rgb = max_delta(&n_rgb, &r_rgb);
  assert!(
    d_rgb <= TOL_U8,
    "rgb native-vs-rowstage max delta {d_rgb} exceeds tolerance {TOL_U8}"
  );
  // RGB channels of RGBA compared (alpha already proven identical).
  let d_rgba = max_delta(&n_rgba, &r_rgba);
  assert!(
    d_rgba <= TOL_U8,
    "rgba native-vs-rowstage max delta {d_rgba} exceeds tolerance {TOL_U8}"
  );
}

#[test]
fn straight_native_default_matches_explicit_true() {
  // `with_native` defaults to true, so a default straight sink and an
  // explicit `with_native(true)` straight sink agree byte-for-byte.
  let (y, u, v, a) = planes(0x2468);
  let explicit = run(&y, &u, &v, &a, true, AlphaMode::Straight);

  let mut rgba = std::vec![0u8; OUT * OUT * 4];
  {
    let mut sink =
      MixedSinker::<Yuva420p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap();
    yuva420p_to(&frame(&y, &u, &v, &a), FR, M, &mut sink).unwrap();
  }
  assert_eq!(
    rgba, explicit.1,
    "default tier must equal explicit with_native(true)"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn premultiplied_byte_identical_to_current() {
  // Premultiplied is NOT native-eligible: `with_native(true)` and
  // `with_native(false)` produce the IDENTICAL packed-YUVA premultiplied
  // output (the native tier is never taken). This is the byte-identity
  // anchor — the new straight-alpha native path must not perturb premult.
  let (y, u, v, a) = planes(0x1234);
  let with_native = run(&y, &u, &v, &a, true, AlphaMode::Premultiplied);
  let without_native = run(&y, &u, &v, &a, false, AlphaMode::Premultiplied);
  assert_eq!(
    with_native, without_native,
    "with_native must not change Premultiplied output (byte-identical)"
  );

  // And it equals the premult-bin-unpremult oracle (the pre-Phase-5 path).
  let mut pm = {
    let mut full = std::vec![0u8; SRC * SRC * 4];
    let mut sink = MixedSinker::<Yuva420p>::new(SRC, SRC)
      .with_rgba(&mut full)
      .unwrap();
    yuva420p_to(&frame(&y, &u, &v, &a), FR, M, &mut sink).unwrap();
    full
  };
  for px in pm.chunks_exact_mut(4) {
    let alpha = px[3] as u32;
    for c in &mut px[..3] {
      *c = ((*c as u32 * alpha + 127) / 255) as u8;
    }
  }
  let mut binned = std::vec![0u8; OUT * OUT * 4];
  for oy in 0..OUT {
    for ox in 0..OUT {
      for c in 0..4 {
        let mut acc = 0u32;
        for dy in 0..2 {
          for dx in 0..2 {
            acc += pm[((oy * 2 + dy) * SRC + ox * 2 + dx) * 4 + c] as u32;
          }
        }
        binned[(oy * OUT + ox) * 4 + c] = ((acc + 2) / 4) as u8;
      }
    }
  }
  let mut oracle = std::vec![0u8; OUT * OUT * 4];
  for (o, i) in oracle.chunks_exact_mut(4).zip(binned.chunks_exact(4)) {
    let alpha = i[3] as u32;
    for c in 0..3 {
      o[c] = (i[c] as u32 * 255 + alpha / 2)
        .checked_div(alpha)
        .map_or(0, |q| q.min(255)) as u8;
    }
    o[3] = i[3];
  }
  assert_eq!(
    with_native.1, oracle,
    "premult rgba (native flag set) == premult-bin-unpremult oracle"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn straight_native_simd_matches_scalar() {
  // All five SIMD backends (the α `AreaStream<u8>` bin + the YUV→RGB convert
  // + the α-scatter) must agree with the scalar reference, byte-for-byte.
  let (y, u, v, a) = planes(0x9753);
  let render = |simd: bool| {
    let mut rgba = std::vec![0u8; OUT * OUT * 4];
    let mut luma = std::vec![0u8; OUT * OUT];
    let mut sink =
      MixedSinker::<Yuva420p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_simd(simd)
        .with_rgba(&mut rgba)
        .unwrap()
        .with_luma(&mut luma)
        .unwrap();
    yuva420p_to(&frame(&y, &u, &v, &a), FR, M, &mut sink).unwrap();
    (rgba, luma)
  };
  assert_eq!(
    render(true),
    render(false),
    "straight native SIMD != scalar"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn straight_native_cross_frame_reset_reuses_streams() {
  // The native join (Y/U/V + α streams) must reset between frames so a
  // second frame reuses the streams and reproduces the bin-then-convert
  // oracle.
  let (y, u, v, a) = planes(0x5151);
  let mut rgba = std::vec![0u8; OUT * OUT * 4];
  {
    let mut sink =
      MixedSinker::<Yuva420p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap();
    yuva420p_to(&frame(&y, &u, &v, &a), FR, M, &mut sink).unwrap();
    yuva420p_to(&frame(&y, &u, &v, &a), FR, M, &mut sink).unwrap();
  }
  assert_eq!(rgba, oracle(&y, &u, &v, &a).1, "second frame != oracle");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn straight_native_mid_frame_alpha_mode_flip_is_rejected() {
  // The AlphaMode freeze (check_frozen_alpha_mode) must reject a mid-frame
  // Straight -> Premultiplied flip on the native path, just like the
  // row-stage path.
  let (y, u, v, a) = planes(0x33AA);
  let mut rgba = std::vec![0u8; OUT * OUT * 4];
  let mut sink =
    MixedSinker::<Yuva420p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_rgba(&mut rgba)
      .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  sink
    .process(Yuva420pRow::new(
      &y[..SRC],
      &u[..CW],
      &v[..CW],
      &a[..SRC],
      0,
      M,
      FR,
    ))
    .unwrap();
  sink.set_alpha_mode(AlphaMode::Premultiplied);
  let err = sink
    .process(Yuva420pRow::new(
      &y[SRC..2 * SRC],
      &u[..CW],
      &v[..CW],
      &a[SRC..2 * SRC],
      1,
      M,
      FR,
    ))
    .unwrap_err();
  assert!(
    matches!(err, MixedSinkerError::ResampleOutputsChanged(_)),
    "mid-frame alpha flip not rejected on native path: {err:?}"
  );
}

#[test]
fn straight_native_out_of_sequence_first_row_is_rejected() {
  // The native preflight rejects an out-of-sequence first row BEFORE any
  // allocation, leaving the output untouched (atomicity).
  let (y, u, v, a) = planes(0x44BB);
  let mut rgba = std::vec![0u8; OUT * OUT * 4];
  let mut sink =
    MixedSinker::<Yuva420p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_rgba(&mut rgba)
      .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  let err = sink
    .process(Yuva420pRow::new(
      &y[2 * SRC..3 * SRC],
      &u[CW..2 * CW],
      &v[CW..2 * CW],
      &a[2 * SRC..3 * SRC],
      2,
      M,
      FR,
    ))
    .unwrap_err();
  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
    ),
    "out-of-sequence first row not rejected on native path: {err:?}"
  );
  assert!(rgba.iter().all(|&b| b == 0), "rejected row mutated output");
}

/// The straight-alpha native delegate grows the output RGB scratch as a
/// PRE-FEED step, BEFORE it commits the output-set freeze. A scratch OOM on an
/// ordinary first-ever row-0 build therefore leaves `resample_outputs`
/// uncommitted (the delegate builds the join into a local and freezes only after
/// every pre-feed allocation succeeds), so a retry of row 0 with a CHANGED output
/// attachment is accepted, not mis-rejected as ResampleOutputsChanged.
#[test]
#[cfg(feature = "rgb")]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn straight_native_first_build_scratch_oom_leaves_freeze_unfrozen_for_retry() {
  let (y, u, v, a) = planes(0x51A7);
  let mut rgb = std::vec![0u8; OUT * OUT * 3];
  let mut luma = std::vec![0u8; OUT * OUT];
  let mut sink =
    MixedSinker::<Yuva420p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_rgb(&mut rgb)
      .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  crate::sinker::mixed::arm_native_rgb_scratch_failure();
  let err = sink
    .process(Yuva420pRow::new(
      &y[0..SRC],
      &u[0..CW],
      &v[0..CW],
      &a[0..SRC],
      0,
      M,
      FR,
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
    .process(Yuva420pRow::new(
      &y[0..SRC],
      &u[0..CW],
      &v[0..CW],
      &a[0..SRC],
      0,
      M,
      FR,
    ))
    .expect("row 0 must succeed after a first-build scratch OOM (freeze uncommitted)");
}

/// A cross-frame COLOUR-CAPABILITY rebuild for the straight-alpha native
/// delegate: frame 1 builds a luma-only join (chroma absent, no output RGB
/// scratch); frame 2 attaches RGB, so the delegate rebuilds the join WITH chroma
/// into a local and grows the (empty) output RGB scratch before committing. A
/// scratch OOM on that rebuild must leave the cached luma-only join intact AND
/// `resample_outputs` uncommitted, so a row-0 retry is accepted, not
/// mis-rejected as ResampleOutputsChanged.
#[test]
#[cfg(feature = "rgb")]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn straight_native_colour_capability_rebuild_scratch_oom_leaves_freeze_unfrozen_for_retry() {
  let (y, u, v, a) = planes(0x51A8);
  let mut luma = std::vec![0u8; OUT * OUT];
  let mut rgb = std::vec![0u8; OUT * OUT * 3];
  let (mut hh, mut ss, mut vv) = (
    std::vec![0u8; OUT * OUT],
    std::vec![0u8; OUT * OUT],
    std::vec![0u8; OUT * OUT],
  );
  let mut sink =
    MixedSinker::<Yuva420p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_luma(&mut luma)
      .unwrap();
  // Frame 1: a luma-only join (chroma / α absent, reserves no RGB scratch).
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  for r in 0..SRC {
    sink
      .process(Yuva420pRow::new(
        &y[r * SRC..(r + 1) * SRC],
        &u[(r / 2) * CW..(r / 2) * CW + CW],
        &v[(r / 2) * CW..(r / 2) * CW + CW],
        &a[r * SRC..(r + 1) * SRC],
        r,
        M,
        FR,
      ))
      .expect("luma-only frame builds the native join");
  }
  // Frame 2: attach RGB → the row-0 rebuild must build the chroma half and grow
  // the (now-needed, empty) output RGB scratch; refuse that grow.
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  sink.set_rgb(&mut rgb).unwrap();
  crate::sinker::mixed::arm_native_rgb_scratch_failure();
  let err = sink
    .process(Yuva420pRow::new(
      &y[0..SRC],
      &u[0..CW],
      &v[0..CW],
      &a[0..SRC],
      0,
      M,
      FR,
    ))
    .unwrap_err();
  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
    ),
    "the colour-capability rebuild scratch OOM must surface AllocationFailed, got {err:?}"
  );
  // The freeze was never committed (the delegate commits only after the scratch
  // grows), so attaching ANOTHER output (hsv) and retrying row 0 is ACCEPTED —
  // the changed output set proves the freeze was rolled back (the pre-fold shape
  // would have frozen {luma, rgb} and rejected this as ResampleOutputsChanged).
  sink.set_hsv(&mut hh, &mut ss, &mut vv).unwrap();
  sink
    .process(Yuva420pRow::new(
      &y[0..SRC],
      &u[0..CW],
      &v[0..CW],
      &a[0..SRC],
      0,
      M,
      FR,
    ))
    .expect("row 0 must succeed after a rebuild scratch OOM (freeze uncommitted)");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn straight_native_alpha_drop_paths_match_no_alpha_native() {
  // rgb / luma alpha-drop outputs on the straight native path must be
  // byte-identical to the no-alpha Yuv420p native tier given the same
  // Y/U/V data (α is ignored for those outputs). Cross-checks that the
  // alpha plane never perturbs the colour/luma binning.
  use crate::{
    frame::Yuv420pFrame as Yuv420pFrameAlias,
    source::{Yuv420p, yuv420p_to},
  };
  let (y, u, v, a) = planes(0xC0DE);
  let (n_rgb, _n_rgba, n_luma, n_lu16) = run(&y, &u, &v, &a, true, AlphaMode::Straight);

  let mut rgb = std::vec![0u8; OUT * OUT * 3];
  let mut luma = std::vec![0u8; OUT * OUT];
  let mut lu16 = std::vec![0u16; OUT * OUT];
  {
    let f = Yuv420pFrameAlias::new(
      &y, &u, &v, SRC as u32, SRC as u32, SRC as u32, CW as u32, CW as u32,
    );
    let mut sink =
      MixedSinker::<Yuv420p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb(&mut rgb)
        .unwrap()
        .with_luma(&mut luma)
        .unwrap()
        .with_luma_u16(&mut lu16)
        .unwrap();
    yuv420p_to(&f, FR, M, &mut sink).unwrap();
  }
  assert_eq!(
    n_rgb, rgb,
    "straight native rgb must equal no-alpha native rgb"
  );
  assert_eq!(
    n_luma, luma,
    "straight native luma must equal no-alpha native luma"
  );
  assert_eq!(
    n_lu16, lu16,
    "straight native luma_u16 must equal no-alpha native luma_u16"
  );
}

// ---- frozen native-vs-row-stage route (the #186 guard, threaded onto the
// Phase 5 straight-alpha Yuva420p native dispatch) ----------------------

/// Flipping `set_native(true) -> false` mid-frame must reject as the
/// deterministic `NativeRouteChanged` BEFORE either tier consumes the row:
/// the native tier and the packed-YUVA row-stage tail carry independent,
/// once-only stream state, so splitting a frame across them is rejected, not
/// silently mixed. The straight-alpha sibling of the planar 4:2:0 guard.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn native_to_rowstage_route_flip_mid_frame_rejected() {
  let (y, u, v, a) = ramp();
  let mut luma = std::vec![0u8; OUT * OUT];
  let mut sink =
    MixedSinker::<Yuva420p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_native(true)
      .with_luma(&mut luma)
      .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  // Row 0 freezes the route = native.
  sink
    .process(Yuva420pRow::new(
      &y[..SRC],
      &u[..CW],
      &v[..CW],
      &a[..SRC],
      0,
      M,
      FR,
    ))
    .expect("native row 0 freezes the route and succeeds");
  // Flip to the row-stage tier and feed the next in-sequence row.
  sink.set_native(false);
  let err = sink
    .process(Yuva420pRow::new(
      &y[SRC..2 * SRC],
      &u[..CW],
      &v[..CW],
      &a[SRC..2 * SRC],
      1,
      M,
      FR,
    ))
    .unwrap_err();
  assert!(
    matches!(err, MixedSinkerError::NativeRouteChanged(_)),
    "a native -> row-stage mid-frame route flip must reject as \
     NativeRouteChanged, got {err:?}"
  );
}

/// The reverse flip `set_native(false) -> true` mid-frame must reject
/// identically — the guard catches BOTH directions.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rowstage_to_native_route_flip_mid_frame_rejected() {
  let (y, u, v, a) = ramp();
  let mut luma = std::vec![0u8; OUT * OUT];
  let mut sink =
    MixedSinker::<Yuva420p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_native(false)
      .with_luma(&mut luma)
      .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  // Row 0 freezes the route = row-stage.
  sink
    .process(Yuva420pRow::new(
      &y[..SRC],
      &u[..CW],
      &v[..CW],
      &a[..SRC],
      0,
      M,
      FR,
    ))
    .expect("row-stage row 0 freezes the route and succeeds");
  sink.set_native(true);
  let err = sink
    .process(Yuva420pRow::new(
      &y[SRC..2 * SRC],
      &u[..CW],
      &v[..CW],
      &a[SRC..2 * SRC],
      1,
      M,
      FR,
    ))
    .unwrap_err();
  assert!(
    matches!(err, MixedSinkerError::NativeRouteChanged(_)),
    "a row-stage -> native mid-frame route flip must reject as \
     NativeRouteChanged, got {err:?}"
  );
}

/// A constant-route frame runs to completion, and the per-frame reset (in
/// `begin_frame`) lets the NEXT frame pick the OTHER tier — the route guard
/// is reset, not sticky.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn route_constant_succeeds_and_resets_across_frames() {
  let (y, u, v, a) = planes(0x6161);
  let mut rgba = std::vec![0u8; OUT * OUT * 4];
  let mut sink =
    MixedSinker::<Yuva420p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_native(true)
      .with_rgba(&mut rgba)
      .unwrap();
  // Frame 1: native, route constant across every row.
  yuva420p_to(&frame(&y, &u, &v, &a), FR, M, &mut sink).unwrap();
  // Frame 2: flip to row-stage for the WHOLE frame after begin_frame — a new
  // frame may pick the other tier because the route resets per frame.
  sink.set_native(false);
  yuva420p_to(&frame(&y, &u, &v, &a), FR, M, &mut sink)
    .expect("a new frame may pick the other tier; the route resets per frame");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn straight_native_box_alloc_failure_recoverable() {
  // The outer `Box<NativeYuva420>` allocation must be recoverable: a refusal
  // (armed failpoint, standing in for host OOM) on the first native row
  // returns the typed `AllocationFailed`, NOT an abort, and leaves the output
  // untouched + the native field empty so the call is retryable
  // (first-row-transactional).
  let (y, u, v, a) = planes(0x70B0);
  // Scope 1: arm the failpoint, feed the first native row, and prove the box
  // refusal surfaces AllocationFailed (not an abort) with the output
  // untouched. The sink is dropped at the scope end so the buffer can be read.
  let mut rgba = std::vec![0u8; OUT * OUT * 4];
  {
    let mut sink =
      MixedSinker::<Yuva420p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap();
    sink.begin_frame(SRC as u32, SRC as u32).unwrap();
    crate::resample::arm_box_failure();
    let err = sink
      .process(Yuva420pRow::new(
        &y[..SRC],
        &u[..CW],
        &v[..CW],
        &a[..SRC],
        0,
        M,
        FR,
      ))
      .unwrap_err();
    assert!(
      matches!(
        err,
        MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
      ),
      "first-row box-alloc refusal must surface AllocationFailed (not abort), \
       got {err:?}"
    );
  }
  assert!(
    rgba.iter().all(|&b| b == 0),
    "a box-alloc-refused first row must not touch the output"
  );

  // Scope 2: a fresh frame after the single-shot failpoint is consumed must
  // resample cleanly and equal the bin-then-convert oracle — the refusal was
  // recoverable, not a one-way poison (the native field is left `None`, so the
  // join allocates afresh).
  let mut rgba2 = std::vec![0u8; OUT * OUT * 4];
  {
    let mut sink =
      MixedSinker::<Yuva420p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgba(&mut rgba2)
        .unwrap();
    yuva420p_to(&frame(&y, &u, &v, &a), FR, M, &mut sink)
      .expect("a fresh frame after the consumed failpoint resamples cleanly");
  }
  assert_eq!(
    rgba2,
    oracle(&y, &u, &v, &a).1,
    "the recovered frame must equal the bin-then-convert oracle"
  );
}
