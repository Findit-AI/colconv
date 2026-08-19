//! RFC #238 Phase 2 — the [`AveragingDomain::Linear`] silent-fallback guard
//! under `yuv-planar` WITHOUT `rgb`.
//!
//! `with_averaging_domain` is gated on `yuv-planar` alone, but the linear-light
//! tail decodes to RGB and so is only compiled under `rgb`. With `yuv-planar`
//! but not `rgb`, a `Linear` sink therefore cannot honour the domain — and it
//! must REJECT with a typed error, never silently fall through to the Encoded
//! path (which would resample in the wrong colour domain behind the caller's
//! back). This pins the always-compiled choke-point guard at each planar 8-bit
//! YUV dispatch site for the `rgb`-off build.

use crate::{
  KernelMatrix, PixelSink,
  resample::{AreaResampler, AveragingDomain, ResampleError},
  sinker::{MixedSinker, MixedSinkerError},
  source::{Yuv420p, Yuv420pRow},
};

const SRC: usize = 8;
const OUT: usize = 4;

/// Under `yuv-planar` without `rgb`, a `Linear` sink rejects at `process` with
/// the typed `LinearDomainUnsupported` (the domain needs the `rgb` decode path
/// this build omits) and leaves every attached output untouched — it must NOT
/// silently downgrade to the Encoded average.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn linear_domain_without_rgb_rejects_and_does_not_silently_encode() {
  const SENTINEL: u8 = 0x73;
  // Luma is the rgb-independent output available under `yuv-planar` alone.
  let mut luma = vec![SENTINEL; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Yuv420p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_averaging_domain(AveragingDomain::Linear)
        .with_luma(&mut luma)
        .unwrap();
    sink.begin_frame(SRC as u32, SRC as u32).unwrap();

    let y = [50u8; SRC];
    let u = [128u8; SRC / 2];
    let v = [128u8; SRC / 2];
    let err = sink
      .process(Yuv420pRow::for_tests(
        &y,
        &u,
        &v,
        0,
        KernelMatrix::Bt709,
        true,
      ))
      .unwrap_err();
    assert!(
      matches!(
        err,
        MixedSinkerError::Resample(ResampleError::LinearDomainUnsupported(_))
      ),
      "Linear sink without rgb must reject with LinearDomainUnsupported \
       (not silently encode), got {err:?}",
    );
  }
  assert!(
    luma.iter().all(|&b| b == SENTINEL),
    "a rejected Linear-without-rgb row must leave the output unmutated",
  );
}
