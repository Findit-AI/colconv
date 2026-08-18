//! RFC #238 Phase 2 — [`AveragingDomain::Linear`] coverage for the planar
//! 8-bit YUV family (`Yuv420p` / `Yuv422p` / `Yuv444p` / `Yuv440p`).
//!
//! The Linear domain decodes each source pixel to RGB, linearises it via a
//! [`TransferFunction`] EOTF, area-bins the linear RGB, and re-encodes via
//! the OETF. These tests pin:
//!
//! - **`*_linear_domain_equals_independent_linear_light_oracle`** — the
//!   Linear RGB output equals a from-scratch oracle that decodes the same
//!   full-resolution RGB, EOTFs, 2x2-block-means in linear light, and OETFs,
//!   within a documented 1-LSB f32-rounding tolerance.
//! - **`linear_and_encoded_domains_differ`** — the two domains give
//!   materially different RGB (the affine convert makes linear-light mixing
//!   observably distinct), so the domain choice is meaningful.
//! - **`transfer_function_caller_override_changes_output` /
//!   `per_color_matrix_default_transfer_resolves`** — a caller
//!   `with_transfer_function` override changes the Linear output, and a sink
//!   with no override resolves the curve from its `KernelMatrix`.
//! - **`encoded_default_is_byte_identical_to_unset`** — leaving the domain
//!   at its default (`Encoded`) is byte-identical to never touching the
//!   builder; the encoded path is unchanged.

use crate::{
  KernelMatrix, PixelSink,
  resample::{
    AreaResampler, AveragingDomain, FilteredResampler, ResampleError, TransferFunction, Triangle,
  },
  sinker::{MixedSinker, MixedSinkerError},
  source::{
    Yuv420p, Yuv420pRow, Yuv422p, Yuv440p, Yuv444p, Yuv444pRow, yuv420p_to, yuv422p_to, yuv440p_to,
    yuv444p_to,
  },
};
use mediaframe::frame::{Yuv420pFrame, Yuv422pFrame, Yuv440pFrame, Yuv444pFrame};

const SRC: usize = 8;
const OUT: usize = 4;

// ---- shared fixtures -----------------------------------------------------

fn y_ramp() -> Vec<u8> {
  let mut y = vec![0u8; SRC * SRC];
  for (i, p) in y.iter_mut().enumerate() {
    *p = 24 + (i as u8) % 200;
  }
  y
}

/// A chroma plane of `cw x ch` with a saturated-ish ramp (drives the U/V
/// far from neutral so the convert produces vivid RGB — where linear vs
/// encoded mixing diverges most).
fn chroma(cw: usize, ch: usize, base: u8, step: u8) -> Vec<u8> {
  let mut c = vec![0u8; cw * ch];
  for (i, p) in c.iter_mut().enumerate() {
    *p = base.wrapping_add(((i % cw) as u8).wrapping_mul(step));
  }
  c
}

/// Decode a YUV frame to a **full-resolution encoded RGB** buffer
/// (`SRC x SRC x 3`) via the format's own identity conversion — the same
/// kernel the Linear path decodes through.
fn full_res_rgb_420(y: &[u8], u: &[u8], v: &[u8], matrix: KernelMatrix) -> Vec<u8> {
  let cw = SRC / 2;
  let src = Yuv420pFrame::new(
    y, u, v, SRC as u32, SRC as u32, SRC as u32, cw as u32, cw as u32,
  );
  let mut rgb = vec![0u8; SRC * SRC * 3];
  {
    let mut sink = MixedSinker::<Yuv420p>::new(SRC, SRC)
      .with_rgb(&mut rgb)
      .unwrap();
    yuv420p_to(&src, true, matrix, &mut sink).unwrap();
  }
  rgb
}

fn full_res_rgb_422(y: &[u8], u: &[u8], v: &[u8], matrix: KernelMatrix) -> Vec<u8> {
  let cw = SRC / 2;
  let src = Yuv422pFrame::new(
    y, u, v, SRC as u32, SRC as u32, SRC as u32, cw as u32, cw as u32,
  );
  let mut rgb = vec![0u8; SRC * SRC * 3];
  {
    let mut sink = MixedSinker::<Yuv422p>::new(SRC, SRC)
      .with_rgb(&mut rgb)
      .unwrap();
    yuv422p_to(&src, true, matrix, &mut sink).unwrap();
  }
  rgb
}

fn full_res_rgb_444(y: &[u8], u: &[u8], v: &[u8], matrix: KernelMatrix) -> Vec<u8> {
  let src = Yuv444pFrame::new(
    y, u, v, SRC as u32, SRC as u32, SRC as u32, SRC as u32, SRC as u32,
  );
  let mut rgb = vec![0u8; SRC * SRC * 3];
  {
    let mut sink = MixedSinker::<Yuv444p>::new(SRC, SRC)
      .with_rgb(&mut rgb)
      .unwrap();
    yuv444p_to(&src, true, matrix, &mut sink).unwrap();
  }
  rgb
}

fn full_res_rgb_440(y: &[u8], u: &[u8], v: &[u8], matrix: KernelMatrix) -> Vec<u8> {
  let src = Yuv440pFrame::new(
    y, u, v, SRC as u32, SRC as u32, SRC as u32, SRC as u32, SRC as u32,
  );
  let mut rgb = vec![0u8; SRC * SRC * 3];
  {
    let mut sink = MixedSinker::<Yuv440p>::new(SRC, SRC)
      .with_rgb(&mut rgb)
      .unwrap();
    yuv440p_to(&src, true, matrix, &mut sink).unwrap();
  }
  rgb
}

/// The independent linear-light oracle: from a full-resolution encoded RGB
/// buffer, EOTF each `SRC x SRC` pixel, take each 2x2-block linear mean, and
/// OETF back to encoded RGB at `OUT x OUT`. This is the
/// decode -> linearise -> bin -> encode reference, computed without touching
/// the production binning stream.
fn linear_light_oracle(full_rgb: &[u8], tf: TransferFunction) -> Vec<u8> {
  let mut out = vec![0u8; OUT * OUT * 3];
  for oy in 0..OUT {
    for ox in 0..OUT {
      for c in 0..3 {
        let mut acc = 0.0f32;
        for dy in 0..2 {
          for dx in 0..2 {
            let sy = oy * 2 + dy;
            let sx = ox * 2 + dx;
            let e = full_rgb[(sy * SRC + sx) * 3 + c] as f32 / 255.0;
            acc += tf.eotf(e);
          }
        }
        let mean = acc / 4.0;
        let enc = (tf.oetf(mean) * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
        out[(oy * OUT + ox) * 3 + c] = enc;
      }
    }
  }
  out
}

/// Per-channel max absolute difference between two equal-length `u8` RGB
/// buffers.
fn max_abs_diff(a: &[u8], b: &[u8]) -> u8 {
  a.iter()
    .zip(b.iter())
    .map(|(&x, &y)| x.abs_diff(y))
    .max()
    .unwrap_or(0)
}

// ---- linear == independent linear-light oracle (per format) --------------
//
// The Linear RGB output matches the from-scratch oracle within 1 LSB. Both
// EOTF the same full-resolution RGB and box-average in linear light; the
// only difference is the f32 accumulation order (the production
// `AreaStream<f32>` H-reduce-then-V-accumulate vs the oracle's flat 2x2
// sum), which can perturb the re-encoded byte by at most 1 — pinned as
// `<= 1` and documented here.

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv420p_linear_domain_equals_independent_linear_light_oracle() {
  let matrix = KernelMatrix::Bt709;
  let tf = TransferFunction::for_matrix(&crate::ColorMatrix::from(matrix));
  let y = y_ramp();
  let cw = SRC / 2;
  let u = chroma(cw, cw, 200, 6);
  let v = chroma(cw, cw, 40, 7);

  let oracle = linear_light_oracle(&full_res_rgb_420(&y, &u, &v, matrix), tf);

  let mut rgb = vec![0u8; OUT * OUT * 3];
  {
    let src = Yuv420pFrame::new(
      &y, &u, &v, SRC as u32, SRC as u32, SRC as u32, cw as u32, cw as u32,
    );
    let mut sink =
      MixedSinker::<Yuv420p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_averaging_domain(AveragingDomain::Linear)
        .with_rgb(&mut rgb)
        .unwrap();
    yuv420p_to(&src, true, matrix, &mut sink).unwrap();
  }
  assert!(
    max_abs_diff(&rgb, &oracle) <= 1,
    "Yuv420p linear vs oracle: max diff {} (rgb={rgb:?} oracle={oracle:?})",
    max_abs_diff(&rgb, &oracle),
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv422p_linear_domain_equals_independent_linear_light_oracle() {
  let matrix = KernelMatrix::Bt709;
  let tf = TransferFunction::for_matrix(&crate::ColorMatrix::from(matrix));
  let y = y_ramp();
  let cw = SRC / 2;
  let u = chroma(cw, SRC, 200, 6);
  let v = chroma(cw, SRC, 40, 7);

  let oracle = linear_light_oracle(&full_res_rgb_422(&y, &u, &v, matrix), tf);

  let mut rgb = vec![0u8; OUT * OUT * 3];
  {
    let src = Yuv422pFrame::new(
      &y, &u, &v, SRC as u32, SRC as u32, SRC as u32, cw as u32, cw as u32,
    );
    let mut sink =
      MixedSinker::<Yuv422p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_averaging_domain(AveragingDomain::Linear)
        .with_rgb(&mut rgb)
        .unwrap();
    yuv422p_to(&src, true, matrix, &mut sink).unwrap();
  }
  assert!(
    max_abs_diff(&rgb, &oracle) <= 1,
    "Yuv422p linear vs oracle: max diff {}",
    max_abs_diff(&rgb, &oracle),
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv444p_linear_domain_equals_independent_linear_light_oracle() {
  let matrix = KernelMatrix::Bt709;
  let tf = TransferFunction::for_matrix(&crate::ColorMatrix::from(matrix));
  let y = y_ramp();
  let u = chroma(SRC, SRC, 200, 6);
  let v = chroma(SRC, SRC, 40, 7);

  let oracle = linear_light_oracle(&full_res_rgb_444(&y, &u, &v, matrix), tf);

  let mut rgb = vec![0u8; OUT * OUT * 3];
  {
    let src = Yuv444pFrame::new(
      &y, &u, &v, SRC as u32, SRC as u32, SRC as u32, SRC as u32, SRC as u32,
    );
    let mut sink =
      MixedSinker::<Yuv444p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_averaging_domain(AveragingDomain::Linear)
        .with_rgb(&mut rgb)
        .unwrap();
    yuv444p_to(&src, true, matrix, &mut sink).unwrap();
  }
  assert!(
    max_abs_diff(&rgb, &oracle) <= 1,
    "Yuv444p linear vs oracle: max diff {}",
    max_abs_diff(&rgb, &oracle),
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv440p_linear_domain_equals_independent_linear_light_oracle() {
  let matrix = KernelMatrix::Bt709;
  let tf = TransferFunction::for_matrix(&crate::ColorMatrix::from(matrix));
  let y = y_ramp();
  let ch = SRC / 2;
  let u = chroma(SRC, ch, 200, 6);
  let v = chroma(SRC, ch, 40, 7);

  let oracle = linear_light_oracle(&full_res_rgb_440(&y, &u, &v, matrix), tf);

  let mut rgb = vec![0u8; OUT * OUT * 3];
  {
    let src = Yuv440pFrame::new(
      &y, &u, &v, SRC as u32, SRC as u32, SRC as u32, SRC as u32, SRC as u32,
    );
    let mut sink =
      MixedSinker::<Yuv440p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_averaging_domain(AveragingDomain::Linear)
        .with_rgb(&mut rgb)
        .unwrap();
    yuv440p_to(&src, true, matrix, &mut sink).unwrap();
  }
  assert!(
    max_abs_diff(&rgb, &oracle) <= 1,
    "Yuv440p linear vs oracle: max diff {}",
    max_abs_diff(&rgb, &oracle),
  );
}

// ---- linear vs encoded materially differ ---------------------------------

/// Runs a `Yuv420p` area downscale to RGB under the given domain.
fn run_420(y: &[u8], u: &[u8], v: &[u8], matrix: KernelMatrix, domain: AveragingDomain) -> Vec<u8> {
  let cw = SRC / 2;
  let src = Yuv420pFrame::new(
    y, u, v, SRC as u32, SRC as u32, SRC as u32, cw as u32, cw as u32,
  );
  let mut rgb = vec![0u8; OUT * OUT * 3];
  {
    let mut sink = MixedSinker::<Yuv420p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_averaging_domain(domain)
        // Pin the row-stage encoded tier so the encoded baseline is the
        // convert-then-bin (RGB-domain) average — the closest encoded
        // analogue to the linear path, isolating the domain difference.
        .with_native(false)
        .with_rgb(&mut rgb)
        .unwrap();
    yuv420p_to(&src, true, matrix, &mut sink).unwrap();
  }
  rgb
}

/// High-contrast in-gamut Y: each 2x2 block mixes near-black (16) and
/// near-white (235) luma. Averaging that contrast in gamma-encoded space
/// vs linear light diverges by tens of codes — and with neutral chroma the
/// RGB stays grey and in-gamut (no clamp to hide the difference).
fn y_checker() -> Vec<u8> {
  let mut y = vec![0u8; SRC * SRC];
  for (i, p) in y.iter_mut().enumerate() {
    let (r, c) = (i / SRC, i % SRC);
    *p = if (r + c) % 2 == 0 { 16 } else { 235 };
  }
  y
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn linear_and_encoded_domains_differ() {
  // Neutral chroma keeps RGB grey and in-gamut; the checkerboard luma gives
  // each output pixel a dark/bright mix whose linear-light mean is far from
  // its gamma-encoded mean. (Saturated chroma instead clamps every channel
  // to 0/255, which would hide the domain difference behind the clamp.)
  let matrix = KernelMatrix::Bt709;
  let y = y_checker();
  let cw = SRC / 2;
  let u = vec![128u8; cw * cw];
  let v = vec![128u8; cw * cw];

  let encoded = run_420(&y, &u, &v, matrix, AveragingDomain::Encoded);
  let linear = run_420(&y, &u, &v, matrix, AveragingDomain::Linear);
  assert!(
    max_abs_diff(&encoded, &linear) > 10,
    "linear must differ materially from encoded (max diff {}, encoded={encoded:?} linear={linear:?})",
    max_abs_diff(&encoded, &linear),
  );
}

// ---- caller transfer override + per-matrix default -----------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn transfer_function_caller_override_changes_output() {
  let matrix = KernelMatrix::Bt709;
  // In-gamut grey mid-tones (checker luma + neutral chroma): the sRGB and
  // BT.1886 curves linearise mid-grey differently, so the re-encoded mean
  // differs — without the clamp that saturated content would impose.
  let y = y_checker();
  let cw = SRC / 2;
  let u = vec![128u8; cw * cw];
  let v = vec![128u8; cw * cw];

  let run = |tf: TransferFunction| {
    let src = Yuv420pFrame::new(
      &y, &u, &v, SRC as u32, SRC as u32, SRC as u32, cw as u32, cw as u32,
    );
    let mut rgb = vec![0u8; OUT * OUT * 3];
    {
      let mut sink = MixedSinker::<Yuv420p, AreaResampler>::with_resampler(
        SRC,
        SRC,
        AreaResampler::to(OUT, OUT),
      )
      .unwrap()
      .with_averaging_domain(AveragingDomain::Linear)
      .with_transfer_function(tf)
      .with_rgb(&mut rgb)
      .unwrap();
      yuv420p_to(&src, true, matrix, &mut sink).unwrap();
    }
    rgb
  };

  let srgb = run(TransferFunction::Srgb);
  let bt1886 = run(TransferFunction::Bt1886);
  assert!(
    max_abs_diff(&srgb, &bt1886) > 1,
    "Srgb vs Bt1886 override must change the Linear output (max diff {})",
    max_abs_diff(&srgb, &bt1886),
  );

  // Each override matches its own oracle exactly (within 1 LSB), proving the
  // override is the curve actually applied — not silently ignored.
  for tf in [TransferFunction::Srgb, TransferFunction::Bt1886] {
    let oracle = linear_light_oracle(&full_res_rgb_420(&y, &u, &v, matrix), tf);
    let got = run(tf);
    assert!(
      max_abs_diff(&got, &oracle) <= 1,
      "override {} must match its oracle (max diff {})",
      tf.as_str(),
      max_abs_diff(&got, &oracle),
    );
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn per_color_matrix_default_transfer_resolves() {
  // With no caller override, a video-matrix sink resolves to BT.1886 — so
  // the un-overridden Linear output equals the BT.1886-override output.
  let matrix = KernelMatrix::Bt709;
  let y = y_ramp();
  let cw = SRC / 2;
  let u = chroma(cw, cw, 200, 6);
  let v = chroma(cw, cw, 40, 7);

  let default = run_420(&y, &u, &v, matrix, AveragingDomain::Linear);

  let mut overridden = vec![0u8; OUT * OUT * 3];
  {
    let src = Yuv420pFrame::new(
      &y, &u, &v, SRC as u32, SRC as u32, SRC as u32, cw as u32, cw as u32,
    );
    let mut sink =
      MixedSinker::<Yuv420p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_averaging_domain(AveragingDomain::Linear)
        .with_transfer_function(TransferFunction::Bt1886)
        .with_native(false)
        .with_rgb(&mut overridden)
        .unwrap();
    yuv420p_to(&src, true, matrix, &mut sink).unwrap();
  }
  assert_eq!(
    default, overridden,
    "a video-matrix sink with no override must resolve to BT.1886",
  );
}

// ---- encoded default byte-identical to never touching the builder --------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn encoded_default_is_byte_identical_to_unset() {
  // The default domain is Encoded; constructing the sink WITHOUT calling
  // `with_averaging_domain` must produce byte-identical output to explicitly
  // setting Encoded — i.e. the new field is inert on the default path. Cover
  // both encoded tiers (native + row-stage) and an RGBA output.
  let matrix = KernelMatrix::Bt601;
  let y = y_ramp();
  let cw = SRC / 2;
  let u = chroma(cw, cw, 110, 6);
  let v = chroma(cw, cw, 150, 7);

  for native in [true, false] {
    let render = |set_encoded: bool| {
      let src = Yuv420pFrame::new(
        &y, &u, &v, SRC as u32, SRC as u32, SRC as u32, cw as u32, cw as u32,
      );
      let mut rgba = vec![0u8; OUT * OUT * 4];
      {
        let base = MixedSinker::<Yuv420p, AreaResampler>::with_resampler(
          SRC,
          SRC,
          AreaResampler::to(OUT, OUT),
        )
        .unwrap()
        .with_native(native);
        let base = if set_encoded {
          base.with_averaging_domain(AveragingDomain::Encoded)
        } else {
          base
        };
        let mut sink = base.with_rgba(&mut rgba).unwrap();
        yuv420p_to(&src, true, matrix, &mut sink).unwrap();
      }
      rgba
    };
    assert_eq!(
      render(false),
      render(true),
      "default (unset) domain must be byte-identical to explicit Encoded (native={native})",
    );
  }
}

// ---- Linear is area-only: a filter plan is rejected, never silently encoded ----

/// The Linear domain only bins in the integer-area engine. Pairing it with a
/// [`FilteredResampler`] must reject at preflight with the typed
/// `UnsupportedFilter` — NOT silently fall through to the Encoded-semantics
/// filter path and return that (wrong) result. Verified two ways: the typed
/// error surfaces, AND the caller's pre-seeded output buffer is left exactly
/// as it was (no encoded filter result was written behind the caller's back).
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn linear_domain_rejects_filter_plan() {
  let matrix = KernelMatrix::Bt709;
  let y = y_ramp();
  let cw = SRC / 2;
  let u = chroma(cw, cw, 200, 6);
  let v = chroma(cw, cw, 40, 7);
  let src = Yuv420pFrame::new(
    &y, &u, &v, SRC as u32, SRC as u32, SRC as u32, cw as u32, cw as u32,
  );

  // Distinct, non-zero sentinel: if the reject ever silently produced the
  // encoded filter result, these bytes would be overwritten.
  const SENTINEL: u8 = 0xAB;
  let mut rgb = vec![SENTINEL; OUT * OUT * 3];
  {
    let mut sink = MixedSinker::<Yuv420p, FilteredResampler<Triangle>>::with_resampler(
      SRC,
      SRC,
      FilteredResampler::new(OUT, OUT, Triangle),
    )
    .unwrap()
    .with_averaging_domain(AveragingDomain::Linear)
    .with_rgb(&mut rgb)
    .unwrap();
    let err = yuv420p_to(&src, true, matrix, &mut sink).unwrap_err();
    assert!(
      matches!(
        err,
        MixedSinkerError::Resample(ResampleError::UnsupportedFilter(_))
      ),
      "Linear + filter plan must reject with UnsupportedFilter, got {err:?}",
    );
  }
  assert!(
    rgb.iter().all(|&b| b == SENTINEL),
    "rejected Linear filter plan must NOT write the output (silent encoded result leaked)",
  );
}

// ---- Linear tail is failure-atomic (reject before any output mutation) ----

/// The Linear tail follows the crate's reject-before-emit contract: a row
/// rejected at preflight (here an out-of-sequence row) must leave every
/// attached output byte untouched and surface the typed `OutOfSequenceRow`,
/// never a downstream `AllocationFailed` or a half-written frame.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn linear_domain_out_of_sequence_row_is_atomic() {
  const SENTINEL: u8 = 0xCD;
  let mut rgb = vec![SENTINEL; OUT * OUT * 3];
  {
    let mut sink =
      MixedSinker::<Yuv420p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_averaging_domain(AveragingDomain::Linear)
        .with_rgb(&mut rgb)
        .unwrap();
    sink.begin_frame(SRC as u32, SRC as u32).unwrap();

    let y = [50u8; SRC];
    let u = [128u8; SRC / 2];
    let v = [128u8; SRC / 2];
    // Row 3 before rows 0..3: rejected before the frame buffer is allocated
    // and before any output is written.
    let err = sink
      .process(Yuv420pRow::new(&y, &u, &v, 3, KernelMatrix::Bt709, true))
      .unwrap_err();
    assert!(
      matches!(
        err,
        MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
      ),
      "out-of-sequence Linear row must reject with OutOfSequenceRow (not AllocationFailed), got {err:?}",
    );
  }
  assert!(
    rgb.iter().all(|&b| b == SENTINEL),
    "a rejected Linear row must leave the output unmutated",
  );
}

/// Changing the attached output set on the FINAL source row — the row that
/// triggers the area bin, the tail allocations, and every output write — must
/// be rejected by `frozen_outputs_check` BEFORE any of that runs. This pins
/// that the whole final-row tail is gated behind the preflight: the typed
/// `ResampleOutputsChanged` surfaces and the freshly-attached output is left
/// untouched (no half-binned frame, no `AllocationFailed`).
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn linear_domain_final_row_output_change_is_atomic() {
  const SENTINEL: u8 = 0xEF;
  let y = y_ramp();
  let cw = SRC / 2;
  let u = chroma(cw, cw, 200, 6);
  let v = chroma(cw, cw, 40, 7);

  let mut rgb = vec![0u8; OUT * OUT * 3];
  // Attached only on the final row to flip the frozen output set; pre-seeded so
  // a tail that wrote behind the preflight would be detectable.
  let mut luma = vec![SENTINEL; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Yuv420p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_averaging_domain(AveragingDomain::Linear)
        .with_rgb(&mut rgb)
        .unwrap();
    sink.begin_frame(SRC as u32, SRC as u32).unwrap();
    // Rows 0..SRC-1 buffer cleanly under the rgb-only output set.
    for r in 0..SRC - 1 {
      let yr = &y[r * SRC..(r + 1) * SRC];
      let cr = r / 2;
      let ur = &u[cr * cw..(cr + 1) * cw];
      let vr = &v[cr * cw..(cr + 1) * cw];
      sink
        .process(Yuv420pRow::new(yr, ur, vr, r, KernelMatrix::Bt709, true))
        .unwrap();
    }
    // Final row: attach a new output (changing the frozen set) — must reject
    // before the bin / allocations / writes.
    sink.set_luma(&mut luma).unwrap();
    let r = SRC - 1;
    let yr = &y[r * SRC..(r + 1) * SRC];
    let cr = r / 2;
    let ur = &u[cr * cw..(cr + 1) * cw];
    let vr = &v[cr * cw..(cr + 1) * cw];
    let err = sink
      .process(Yuv420pRow::new(yr, ur, vr, r, KernelMatrix::Bt709, true))
      .unwrap_err();
    assert!(
      matches!(err, MixedSinkerError::ResampleOutputsChanged(_)),
      "mid-frame output-set change on the final Linear row must reject with \
       ResampleOutputsChanged (not AllocationFailed), got {err:?}",
    );
  }
  assert!(
    luma.iter().all(|&b| b == SENTINEL),
    "a rejected final-row output change must leave the new output unmutated",
  );
}

/// An allocation failure on the FINAL source row — the row that triggers the
/// area-bin tail and its f32 / luma / re-encode allocations — must leave the
/// persistent frame accumulator UNCHANGED, not just the output bytes. The
/// reject-before-emit fix made the *outputs* atomic; this pins the stronger
/// state-atomicity contract: a tail allocation that fails must not advance
/// `next_y` or consume the buffered frame, so the SAME sink retries the final
/// row cleanly (no `begin_frame`) once the pressure clears and produces the
/// correct binned result. Without the fix the failed final row advances
/// `next_y` to `h`, poisoning the accumulator so the retry is rejected as
/// out-of-sequence.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn linear_domain_final_row_alloc_failure_leaves_frame_retryable() {
  let matrix = KernelMatrix::Bt709;
  let tf = TransferFunction::for_matrix(&crate::ColorMatrix::from(matrix));
  let y = y_ramp();
  let cw = SRC / 2;
  let u = chroma(cw, cw, 200, 6);
  let v = chroma(cw, cw, 40, 7);

  // The correct result, computed independently — what a clean (un-failed) run
  // must produce, and therefore what the post-failure retry must also produce.
  let oracle = linear_light_oracle(&full_res_rgb_420(&y, &u, &v, matrix), tf);

  const SENTINEL: u8 = 0x5A;
  let mut rgb = vec![SENTINEL; OUT * OUT * 3];
  {
    let mut sink =
      MixedSinker::<Yuv420p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_averaging_domain(AveragingDomain::Linear)
        .with_rgb(&mut rgb)
        .unwrap();
    sink.begin_frame(SRC as u32, SRC as u32).unwrap();

    let feed = |sink: &mut MixedSinker<'_, Yuv420p, AreaResampler>, r: usize| {
      let yr = &y[r * SRC..(r + 1) * SRC];
      let cr = r / 2;
      let ur = &u[cr * cw..(cr + 1) * cw];
      let vr = &v[cr * cw..(cr + 1) * cw];
      sink.process(Yuv420pRow::new(yr, ur, vr, r, KernelMatrix::Bt709, true))
    };

    // Rows 0..SRC-1 buffer cleanly.
    for r in 0..SRC - 1 {
      feed(&mut sink, r).unwrap();
    }

    // Final row with the tail allocation armed to fail: the allocation tail's
    // refusal must surface as the typed `AllocationFailed` — an out-of-memory
    // condition is NOT a geometry overflow (the `ow`/`oh` products were already
    // validated by the plan). This pins the alloc-vs-overflow typing fix: the
    // failpoint and the real `try_zeroed` / `try_reserve` tail allocations both
    // map to `AllocationFailed`, never `GeometryOverflow`. (The output stays
    // unmutated — the separate `..._output_change_is_atomic` test pins the
    // output-byte half of the reject-before-emit contract; here we pin the
    // frame-STATE half.)
    crate::sinker::mixed::linear_light::arm_linear_tail_alloc_failure();
    let err = feed(&mut sink, SRC - 1).unwrap_err();
    assert!(
      matches!(
        err,
        MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
      ),
      "armed final-row tail alloc must surface AllocationFailed (an allocator \
       refusal is not a GeometryOverflow), got {err:?}",
    );

    // The failpoint is one-shot (taken on the armed call); the SAME sink must
    // now retry the final row cleanly — proving the failure did not advance
    // `next_y` or consume the buffered frame.
    feed(&mut sink, SRC - 1).unwrap();
  }
  assert!(
    max_abs_diff(&rgb, &oracle) <= 1,
    "the post-failure retry must produce the correct binned result (max diff {})",
    max_abs_diff(&rgb, &oracle),
  );
}

// ---- Premultiplied is a category error on these non-alpha YUV formats ----

/// `AveragingDomain::Premultiplied` scales each colour sample by its own alpha
/// before averaging, so it is only meaningful for an alpha-bearing format.
/// `Yuv420p` / `Yuv422p` / `Yuv444p` / `Yuv440p` carry no alpha plane, so the
/// domain is a category error: the dispatch must reject it with the typed
/// `PremultipliedDomainUnsupported` rather than silently downgrade to the
/// Encoded average (which would resample in a different domain than the caller
/// asked for, behind their back). This is the structural sibling of
/// `linear_domain_rejects_filter_plan`, pinned the same two ways for every one
/// of the four dispatch sites: the typed error surfaces, AND the caller's
/// pre-seeded output buffer is left exactly as it was (no encoded result
/// leaked). The reject is unconditional — it does not depend on the `rgb`
/// feature — because the format lacking alpha is a fact of the format, not of
/// the build. (Phase 5 wires Premultiplied for actual alpha formats; there it
/// will be honoured. Here, on non-alpha formats, rejecting is correct.)
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn linear_domain_premultiplied_on_non_alpha_rejects() {
  const SENTINEL: u8 = 0x9E;
  let matrix = KernelMatrix::Bt709;
  let y = y_ramp();
  let cw = SRC / 2;

  // Asserts the shared rejection contract for one format's `process` closure:
  // the typed `PremultipliedDomainUnsupported` surfaces and the pre-seeded
  // output stays untouched.
  fn assert_rejected(format: &str, run: impl FnOnce(&mut [u8]) -> Result<(), MixedSinkerError>) {
    let mut rgb = vec![SENTINEL; OUT * OUT * 3];
    let err = run(&mut rgb).unwrap_err();
    assert!(
      matches!(
        err,
        MixedSinkerError::Resample(ResampleError::PremultipliedDomainUnsupported(_))
      ),
      "{format} + Premultiplied must reject with PremultipliedDomainUnsupported, got {err:?}",
    );
    assert!(
      rgb.iter().all(|&b| b == SENTINEL),
      "{format} rejected Premultiplied row must NOT write the output (silent encoded result leaked)",
    );
  }

  // Yuv420p — chroma w/2 x h/2. Cover both encoded tiers (native + row-stage):
  // the reject must precede the tier split entirely, so neither produces a
  // result under Premultiplied.
  let u420 = chroma(cw, cw, 200, 6);
  let v420 = chroma(cw, cw, 40, 7);
  for native in [true, false] {
    assert_rejected("Yuv420p", |rgb| {
      let src = Yuv420pFrame::new(
        &y, &u420, &v420, SRC as u32, SRC as u32, SRC as u32, cw as u32, cw as u32,
      );
      let mut sink = MixedSinker::<Yuv420p, AreaResampler>::with_resampler(
        SRC,
        SRC,
        AreaResampler::to(OUT, OUT),
      )
      .unwrap()
      .with_native(native)
      .with_averaging_domain(AveragingDomain::Premultiplied)
      .with_rgb(rgb)
      .unwrap();
      yuv420p_to(&src, true, matrix, &mut sink)
    });
  }

  // Yuv422p — chroma w/2 x h.
  let u422 = chroma(cw, SRC, 200, 6);
  let v422 = chroma(cw, SRC, 40, 7);
  assert_rejected("Yuv422p", |rgb| {
    let src = Yuv422pFrame::new(
      &y, &u422, &v422, SRC as u32, SRC as u32, SRC as u32, cw as u32, cw as u32,
    );
    let mut sink =
      MixedSinker::<Yuv422p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_averaging_domain(AveragingDomain::Premultiplied)
        .with_rgb(rgb)
        .unwrap();
    yuv422p_to(&src, true, matrix, &mut sink)
  });

  // Yuv444p — chroma w x h.
  let u444 = chroma(SRC, SRC, 200, 6);
  let v444 = chroma(SRC, SRC, 40, 7);
  assert_rejected("Yuv444p", |rgb| {
    let src = Yuv444pFrame::new(
      &y, &u444, &v444, SRC as u32, SRC as u32, SRC as u32, SRC as u32, SRC as u32,
    );
    let mut sink =
      MixedSinker::<Yuv444p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_averaging_domain(AveragingDomain::Premultiplied)
        .with_rgb(rgb)
        .unwrap();
    yuv444p_to(&src, true, matrix, &mut sink)
  });

  // Yuv440p — chroma w x h/2.
  let ch = SRC / 2;
  let u440 = chroma(SRC, ch, 200, 6);
  let v440 = chroma(SRC, ch, 40, 7);
  assert_rejected("Yuv440p", |rgb| {
    let src = Yuv440pFrame::new(
      &y, &u440, &v440, SRC as u32, SRC as u32, SRC as u32, SRC as u32, SRC as u32,
    );
    let mut sink =
      MixedSinker::<Yuv440p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_averaging_domain(AveragingDomain::Premultiplied)
        .with_rgb(rgb)
        .unwrap();
    yuv440p_to(&src, true, matrix, &mut sink)
  });
}

/// The [`TransferFunction`] resolved on the first output-bearing Linear row is
/// frozen for the frame: a caller flipping [`MixedSinker::set_transfer_function`]
/// mid-frame must be rejected with [`MixedSinkerError::TransferFunctionChanged`]
/// BEFORE any state mutation (every buffered row is already linearised under
/// the first curve), and the accumulator must stay retryable — restoring the
/// transfer lets the SAME sink resume the row with no `begin_frame`.
#[test]
fn linear_domain_mid_frame_transfer_change_is_rejected_and_retryable() {
  const SENTINEL: u8 = 0xEF;
  let y = y_ramp();
  let cw = SRC / 2;
  let u = chroma(cw, cw, 200, 6);
  let v = chroma(cw, cw, 40, 7);
  let row = |r: usize| {
    let yr = &y[r * SRC..(r + 1) * SRC];
    let cr = r / 2;
    (yr, &u[cr * cw..(cr + 1) * cw], &v[cr * cw..(cr + 1) * cw])
  };

  let mut rgb = vec![SENTINEL; OUT * OUT * 3];
  {
    let mut sink =
      MixedSinker::<Yuv420p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_averaging_domain(AveragingDomain::Linear)
        .with_transfer_function(TransferFunction::Srgb)
        .with_rgb(&mut rgb)
        .unwrap();
    sink.begin_frame(SRC as u32, SRC as u32).unwrap();

    // Row 0 freezes the resolved transfer (Srgb) on the lazily-created frame.
    let (yr, ur, vr) = row(0);
    sink
      .process(Yuv420pRow::new(yr, ur, vr, 0, KernelMatrix::Bt709, true))
      .unwrap();

    // Flip the transfer mid-frame, then feed row 1 — must reject before the
    // frame buffer is touched (no allocation, no `next_y` advance).
    sink.set_transfer_function(TransferFunction::Bt1886);
    let (yr, ur, vr) = row(1);
    let err = sink
      .process(Yuv420pRow::new(yr, ur, vr, 1, KernelMatrix::Bt709, true))
      .unwrap_err();
    assert!(
      matches!(err, MixedSinkerError::TransferFunctionChanged(_)),
      "a mid-frame transfer-function change must reject with \
       TransferFunctionChanged, got {err:?}",
    );

    // Restore the frozen transfer: the SAME sink resumes row 1 (proving the
    // rejected call left `next_y` unadvanced — no poisoning, no `begin_frame`).
    sink.set_transfer_function(TransferFunction::Srgb);
    for r in 1..SRC {
      let (yr, ur, vr) = row(r);
      sink
        .process(Yuv420pRow::new(yr, ur, vr, r, KernelMatrix::Bt709, true))
        .unwrap();
    }
  }
  assert!(
    rgb.iter().any(|&b| b != SENTINEL),
    "the resumed frame must produce real output once completed",
  );
}

/// The [`AveragingDomain`] chosen on the first output-bearing row is frozen for
/// the frame, parallel to the frozen native route / output set / transfer: a
/// caller flipping [`MixedSinker::set_averaging_domain`] mid-frame must be
/// rejected with the specific [`MixedSinkerError::AveragingDomainChanged`] (NOT
/// the less-specific `OutOfSequenceRow` the per-path sequencing surfaces only
/// incidentally) BEFORE any state mutation, leaving every attached output byte
/// untouched, and the frame must stay retryable — restoring the domain lets the
/// SAME sink resume the row with no `begin_frame`.
#[test]
fn linear_domain_mid_frame_domain_change_is_rejected() {
  const SENTINEL: u8 = 0xEF;
  let y = y_ramp();
  let cw = SRC / 2;
  let u = chroma(cw, cw, 200, 6);
  let v = chroma(cw, cw, 40, 7);
  let row = |r: usize| {
    let yr = &y[r * SRC..(r + 1) * SRC];
    let cr = r / 2;
    (yr, &u[cr * cw..(cr + 1) * cw], &v[cr * cw..(cr + 1) * cw])
  };

  let mut rgb = vec![SENTINEL; OUT * OUT * 3];
  {
    let mut sink =
      MixedSinker::<Yuv420p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_averaging_domain(AveragingDomain::Linear)
        .with_rgb(&mut rgb)
        .unwrap();
    sink.begin_frame(SRC as u32, SRC as u32).unwrap();

    // Row 0 freezes the domain (Linear) on its first output-bearing row.
    let (yr, ur, vr) = row(0);
    sink
      .process(Yuv420pRow::new(yr, ur, vr, 0, KernelMatrix::Bt709, true))
      .unwrap();

    // Flip the domain mid-frame, then feed row 1 — the freeze guards the domain
    // choice itself (BEFORE the dispatch match and before any state mutation),
    // so it must reject with the SPECIFIC AveragingDomainChanged, never the
    // incidental OutOfSequenceRow that the per-path sequencing would otherwise
    // surface (Encoded's area stream is at row 0 while the Linear accumulator
    // advanced to row 1).
    sink.set_averaging_domain(AveragingDomain::Encoded);
    let (yr, ur, vr) = row(1);
    let err = sink
      .process(Yuv420pRow::new(yr, ur, vr, 1, KernelMatrix::Bt709, true))
      .unwrap_err();
    assert!(
      matches!(err, MixedSinkerError::AveragingDomainChanged(_)),
      "a mid-frame averaging-domain change must reject with the specific \
       AveragingDomainChanged, got {err:?}",
    );
    assert!(
      !matches!(
        err,
        MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
      ),
      "the domain-change rejection must be AveragingDomainChanged, never the \
       incidental OutOfSequenceRow, got {err:?}",
    );

    // Restore the frozen domain: the SAME sink resumes row 1 (proving the
    // rejected call left the accumulator unadvanced — no poisoning, no
    // `begin_frame`; the Linear tail emits only on the final row, so the
    // rejected non-final row also wrote no output byte) and runs the frame to
    // completion.
    sink.set_averaging_domain(AveragingDomain::Linear);
    for r in 1..SRC {
      let (yr, ur, vr) = row(r);
      sink
        .process(Yuv420pRow::new(yr, ur, vr, r, KernelMatrix::Bt709, true))
        .unwrap();
    }
  }
  // The frame still completed and wrote real output once resumed — the
  // rejected mid-frame domain flip neither poisoned the accumulator nor left a
  // half-written frame.
  assert!(
    rgb.iter().any(|&b| b != SENTINEL),
    "the resumed frame must produce real output once completed",
  );
}

/// A row REJECTED by the selected domain path must NOT commit the per-frame
/// domain freeze — the freeze is set-AFTER-accept (mirroring `frozen_native_route`),
/// not set-before-dispatch. Here a Linear sink is paired with a
/// [`FilteredResampler`] (the Linear domain is area-only, so the Linear tail
/// rejects the filter plan with [`ResampleError::UnsupportedFilter`]). That
/// rejection consumed no row, so it must leave `frozen_domain` unset: correcting
/// the domain to `Encoded` and retrying the SAME row 0 on the SAME sink must
/// SUCCEED (the Encoded filter path is valid). With the set-before-dispatch bug
/// the row-0 rejection would have stuck the freeze at Linear, and the corrected
/// retry would be wrongly rejected as [`MixedSinkerError::AveragingDomainChanged`].
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn linear_domain_filter_reject_does_not_poison_domain_freeze() {
  let y = y_ramp();
  let cw = SRC / 2;
  let u = chroma(cw, cw, 200, 6);
  let v = chroma(cw, cw, 40, 7);
  let row = |r: usize| {
    let yr = &y[r * SRC..(r + 1) * SRC];
    let cr = r / 2;
    Yuv420pRow::new(
      yr,
      &u[cr * cw..(cr + 1) * cw],
      &v[cr * cw..(cr + 1) * cw],
      r,
      KernelMatrix::Bt709,
      true,
    )
  };

  const SENTINEL: u8 = 0x3C;
  let mut rgb = vec![SENTINEL; OUT * OUT * 3];
  {
    let mut sink = MixedSinker::<Yuv420p, FilteredResampler<Triangle>>::with_resampler(
      SRC,
      SRC,
      FilteredResampler::new(OUT, OUT, Triangle),
    )
    .unwrap()
    .with_averaging_domain(AveragingDomain::Linear)
    .with_rgb(&mut rgb)
    .unwrap();
    sink.begin_frame(SRC as u32, SRC as u32).unwrap();

    // Row 0 under Linear + a filter plan: the Linear tail rejects the filter
    // with the typed UnsupportedFilter, consuming no row.
    let err = sink.process(row(0)).unwrap_err();
    assert!(
      matches!(
        err,
        MixedSinkerError::Resample(ResampleError::UnsupportedFilter(_))
      ),
      "Linear + filter plan must reject with UnsupportedFilter, got {err:?}",
    );

    // Correct the domain to Encoded and retry the SAME row 0 on the SAME sink.
    // The rejected row left `frozen_domain` unset (set-after-accept), so this
    // must NOT trip AveragingDomainChanged — the Encoded filter path is valid.
    // (Without the fix this `process` is rejected as AveragingDomainChanged:
    // the rejected row 0 would have stuck the freeze at Linear.)
    sink.set_averaging_domain(AveragingDomain::Encoded);
    sink.process(row(0)).expect(
      "after a rejected row the corrected-domain retry of the SAME row must \
       succeed — the rejected row must not have committed the domain freeze",
    );
    // Feed the rest of the frame so the Encoded filter path emits real output
    // (a Triangle downscale needs more than one source row before any output
    // row's support is complete).
    for r in 1..SRC {
      sink.process(row(r)).unwrap();
    }
  }
  // The corrected-domain frame ran to completion and produced real output —
  // the rejected row 0 neither poisoned the freeze nor left a dead sink.
  assert!(
    rgb.iter().any(|&b| b != SENTINEL),
    "the corrected-domain retry must produce real output",
  );
}

/// A fallible op on the row that *would* CREATE the frame — here the per-row
/// decode-scratch reserve, which runs AFTER the frame is built but BEFORE that
/// frame is committed to the sink — must leave `*frame` `None`, so the frozen
/// transfer is NOT captured. The structural invariant: `*frame` is `Some` only
/// after at least one output-bearing row was *fully* accepted.
///
/// Concretely: arm the scratch failpoint, feed row 0 (the first output-bearing
/// row) → it returns `AllocationFailed` having committed nothing. Then change
/// the transfer function and retry the SAME row 0 on the SAME sink — it must
/// SUCCEED and bin the frame under the *new* curve, with no `begin_frame`. The
/// retry NOT tripping [`TransferFunctionChanged`] is the whole point: before the
/// fix, the failed row 0 created the frame (freezing the *old* transfer) before
/// the scratch reserve ran, so the corrected-curve retry was wrongly rejected as
/// a mid-frame transfer change against a frame that consumed no row.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn linear_domain_first_row_scratch_failure_leaves_frame_unset() {
  let matrix = KernelMatrix::Bt709;
  let y = y_ramp();
  let cw = SRC / 2;
  let u = chroma(cw, cw, 200, 6);
  let v = chroma(cw, cw, 40, 7);

  // The retry resolves the transfer to Bt1886 (set below), so the correct
  // post-retry output is the Bt1886 oracle — proving the frame bound the NEW
  // curve, not the stale Srgb the failed first row would have frozen.
  let oracle = linear_light_oracle(
    &full_res_rgb_420(&y, &u, &v, matrix),
    TransferFunction::Bt1886,
  );

  const SENTINEL: u8 = 0x6B;
  let mut rgb = vec![SENTINEL; OUT * OUT * 3];
  {
    let mut sink = MixedSinker::<Yuv420p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_averaging_domain(AveragingDomain::Linear)
        // The curve the failed first row would have frozen — must NOT stick,
        // since that row consumed nothing.
        .with_transfer_function(TransferFunction::Srgb)
        .with_rgb(&mut rgb)
        .unwrap();
    sink.begin_frame(SRC as u32, SRC as u32).unwrap();

    let feed = |sink: &mut MixedSinker<'_, Yuv420p, AreaResampler>, r: usize| {
      let yr = &y[r * SRC..(r + 1) * SRC];
      let cr = r / 2;
      let ur = &u[cr * cw..(cr + 1) * cw];
      let vr = &v[cr * cw..(cr + 1) * cw];
      sink.process(Yuv420pRow::new(yr, ur, vr, r, KernelMatrix::Bt709, true))
    };

    // Row 0 with the scratch reserve armed to fail: the frame was built into a
    // local and the failure precedes its commit, so the typed `AllocationFailed`
    // surfaces with `*frame` still `None` (no frozen transfer captured).
    crate::sinker::mixed::linear_light::arm_linear_scratch_failure();
    let err = feed(&mut sink, 0).unwrap_err();
    assert!(
      matches!(
        err,
        MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
      ),
      "armed first-row scratch reserve must surface AllocationFailed, got {err:?}",
    );

    // Change the transfer, then retry the SAME row 0. Because the failed row
    // committed no frame, this is the FIRST accepted row: it freezes Bt1886 and
    // must NOT be rejected as TransferFunctionChanged. (The failpoint is
    // one-shot — already taken — so the scratch reserve now succeeds.)
    sink.set_transfer_function(TransferFunction::Bt1886);
    feed(&mut sink, 0).expect(
      "after a first-row scratch failure the corrected-transfer retry of the \
       SAME row 0 must succeed — the failed row must not have committed the frame \
       (which would have frozen the stale transfer)",
    );
    for r in 1..SRC {
      feed(&mut sink, r).unwrap();
    }
  }
  assert!(
    max_abs_diff(&rgb, &oracle) <= 1,
    "the post-failure retry must bin under the corrected (Bt1886) transfer \
     (max diff {})",
    max_abs_diff(&rgb, &oracle),
  );
}

/// The OUTPUT-SET sibling of the first-row poison: the output freeze
/// (`resample_outputs`) is the LAST first-row frozen-state write, and it must
/// commit transactionally with `*frame`, not before the fallible phase. A
/// fallible op on the row that *would* CREATE the frame — here the per-row
/// decode-scratch reserve — must therefore leave `resample_outputs` `None`
/// (alongside `*frame`), so a corrected retry of the SAME row 0 with a CHANGED
/// output set is accepted, NOT mis-rejected as [`ResampleOutputsChanged`].
///
/// Concretely: arm the scratch failpoint, feed row 0 with only `rgb` attached →
/// it returns `AllocationFailed` having committed nothing (no frozen output set).
/// Then ADD a second output (`luma`) — changing the output set — and retry the
/// SAME row 0 on the SAME sink: it must SUCCEED and freeze the *new* set, binning
/// the frame to completion under it, with no `begin_frame`. The retry NOT tripping
/// [`ResampleOutputsChanged`] is the whole point: before the fix, the shared
/// `frozen_outputs_check` committed `resample_outputs` (freezing the rgb-only set)
/// the instant it ran — BEFORE the fallible scratch reserve — so the failed row 0
/// poisoned the freeze and the changed-set retry was wrongly rejected against a
/// frame that consumed no row.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn linear_domain_first_row_failure_allows_output_set_retry() {
  let matrix = KernelMatrix::Bt709;
  let tf = TransferFunction::for_matrix(&crate::ColorMatrix::from(matrix));
  let y = y_ramp();
  let cw = SRC / 2;
  let u = chroma(cw, cw, 200, 6);
  let v = chroma(cw, cw, 40, 7);

  // The clean-run rgb result (what the post-retry frame must also produce):
  // the failed row 0 committed no output set, so the changed-set retry simply
  // binds {rgb, luma} and bins normally.
  let oracle = linear_light_oracle(&full_res_rgb_420(&y, &u, &v, matrix), tf);

  const SENTINEL: u8 = 0x4D;
  let mut rgb = vec![SENTINEL; OUT * OUT * 3];
  // Attached only AFTER the first-row failure, to change the frozen output set
  // on the retry; pre-seeded so a write proves the new set was honoured.
  let mut luma = vec![SENTINEL; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Yuv420p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_averaging_domain(AveragingDomain::Linear)
        .with_rgb(&mut rgb)
        .unwrap();
    sink.begin_frame(SRC as u32, SRC as u32).unwrap();

    let row = |r: usize| {
      let yr = &y[r * SRC..(r + 1) * SRC];
      let cr = r / 2;
      let ur = &u[cr * cw..(cr + 1) * cw];
      let vr = &v[cr * cw..(cr + 1) * cw];
      Yuv420pRow::new(yr, ur, vr, r, KernelMatrix::Bt709, true)
    };

    // Row 0 (rgb-only) with the scratch reserve armed to fail: the output-set
    // snapshot is only COMPARED (not committed) before the fallible phase, so
    // the failure surfaces `AllocationFailed` with `resample_outputs` still
    // `None` (no frozen set captured).
    crate::sinker::mixed::linear_light::arm_linear_scratch_failure();
    let err = sink.process(row(0)).unwrap_err();
    assert!(
      matches!(
        err,
        MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
      ),
      "armed first-row scratch reserve must surface AllocationFailed, got {err:?}",
    );

    // CHANGE the output set (add luma), then retry the SAME row 0. Because the
    // failed row committed no output freeze, this is the FIRST accepted row: it
    // freezes the NEW {rgb, luma} set and must NOT be rejected as
    // ResampleOutputsChanged. (The failpoint is one-shot — already taken — so
    // the scratch reserve now succeeds.)
    sink.set_luma(&mut luma).unwrap();
    sink.process(row(0)).expect(
      "after a first-row scratch failure the changed-output-set retry of the \
       SAME row 0 must succeed — the failed row must not have committed the \
       output freeze (which would have frozen the stale rgb-only set)",
    );
    for r in 1..SRC {
      sink.process(row(r)).unwrap();
    }
  }
  assert!(
    max_abs_diff(&rgb, &oracle) <= 1,
    "the post-failure retry must bin the rgb output correctly under the changed \
     output set (max diff {})",
    max_abs_diff(&rgb, &oracle),
  );
  assert!(
    luma.iter().any(|&b| b != SENTINEL),
    "the newly-attached luma output must be written on the retry — proving the \
     changed output set was honoured (not rejected against a poisoned freeze)",
  );
}

/// The single-row-frame variant of the first-row poison: when the FIRST row is
/// ALSO the FINAL row (`h == 1`), the bin tail's allocations run on that row —
/// still BEFORE the `*frame` commit under the structural fix. A tail-alloc
/// refusal there must therefore also leave `*frame` `None`, so a corrected-curve
/// retry of row 0 succeeds rather than tripping [`TransferFunctionChanged`].
///
/// Uses a `2x1` `Yuv444p` source (`h == 1`, so `idx == 0` is the final row) and
/// reuses the [`arm_linear_tail_alloc_failure`] failpoint. Before the fix the
/// frame was committed (freezing the transfer) before the tail ran, so the
/// post-failure corrected-curve retry was wrongly rejected.
///
/// [`arm_linear_tail_alloc_failure`]: crate::sinker::mixed::linear_light::arm_linear_tail_alloc_failure
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn linear_domain_single_row_frame_tail_failure_leaves_frame_unset() {
  // A 2x1 444 frame: one source row, two columns → 1x1 out. `h == 1`, so the
  // first row is the final row and its tail runs pre-commit.
  const W: usize = 2;
  let y = [80u8, 200u8];
  let u = [150u8, 90u8];
  let v = [60u8, 170u8];

  const SENTINEL: u8 = 0x71;
  let mut rgb = vec![SENTINEL; 3];
  {
    let mut sink =
      MixedSinker::<Yuv444p, AreaResampler>::with_resampler(W, 1, AreaResampler::to(1, 1))
        .unwrap()
        .with_averaging_domain(AveragingDomain::Linear)
        .with_transfer_function(TransferFunction::Srgb)
        .with_rgb(&mut rgb)
        .unwrap();
    sink.begin_frame(W as u32, 1).unwrap();

    // Row 0 (== final row) with the tail alloc armed to fail: the tail runs
    // before the frame is committed, so `AllocationFailed` surfaces with
    // `*frame` still `None`.
    crate::sinker::mixed::linear_light::arm_linear_tail_alloc_failure();
    let err = sink
      .process(Yuv444pRow::new(&y, &u, &v, 0, KernelMatrix::Bt709, true))
      .unwrap_err();
    assert!(
      matches!(
        err,
        MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
      ),
      "armed single-row tail alloc must surface AllocationFailed, got {err:?}",
    );

    // Change the transfer and retry the SAME row 0: it is the first accepted
    // row, so it freezes the NEW curve and must not trip TransferFunctionChanged
    // (the one-shot failpoint is already taken, so the tail now allocates).
    sink.set_transfer_function(TransferFunction::Bt1886);
    sink
      .process(Yuv444pRow::new(&y, &u, &v, 0, KernelMatrix::Bt709, true))
      .expect(
        "after a single-row tail failure the corrected-transfer retry of row 0 \
         must succeed — the failed row must not have committed the frame",
      );
  }
  assert!(
    rgb.iter().any(|&b| b != SENTINEL),
    "the post-failure single-row retry must produce real output",
  );
}
