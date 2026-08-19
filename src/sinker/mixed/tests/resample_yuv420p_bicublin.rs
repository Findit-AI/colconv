//! BICUBLIN per-plane filter coverage for `Yuv420p` — swscale's
//! `SWS_BICUBLIN`: a **cubic** luma plane and a **linear** chroma plane,
//! each filtered in YUV-plane space at its own resolution and then
//! converted to RGB at the output grid.
//!
//! Unlike the single-kernel filter path
//! ([`planar_dual_filter_resample`](super::super::planar_resample::planar_dual_filter_resample)),
//! which converts the upsampled YUV to a source-width RGB row and applies
//! ONE kernel to the RGB, BICUBLIN filters the three planes **separately**
//! (Y at luma resolution with [`SwscaleBicubic`], U / V at chroma resolution
//! with [`Triangle`]) and converts the filtered Y / U / V to RGB at the
//! output resolution via the 4:4:4 kernel — the filter analog of the native
//! area tier (which bins the planes, then converts). So a single kernel
//! cannot express it.
//!
//! Oracles:
//! - **`bicublin_equals_independent_per_plane_oracle`** — the Bicublin
//!   colour / luma outputs equal an INDEPENDENT from-scratch oracle that
//!   filters Y with the cubic [`FilterStream`] and U / V with the linear
//!   [`FilterStream`] (each plane its own `cw x ch -> out_w x out_h` image),
//!   then converts the filtered planes through the SAME `yuv_444_to_rgb_row`
//!   kernel — **bit-identical** (same engine, same coefficients, same
//!   convert). `luma` equals the cubic single-channel Y filter; `luma_u16`
//!   is that zero-extended.
//! - **`bicublin_luma_chroma_kernels_differ_from_single_kernel`** — the
//!   Bicublin output differs from BOTH a single-cubic and a single-linear
//!   filter, proving the per-plane luma-vs-chroma distinction is real.
//! - atomicity — a mid-frame output-set change, an out-of-sequence row, and
//!   an allocation failure each return a typed error with the outputs
//!   unmutated and the row retryable (the filter-path atomicity contract).

use crate::{
  KernelMatrix, PixelSink,
  resample::{AreaResampler, Bicublin, FilterStream, ResampleError, SwscaleBicubic, Triangle},
  sinker::{MixedSinker, MixedSinkerError},
  source::{Yuv420p, Yuv420pRow, Yuv422p, yuv420p_to, yuv422p_to},
};
use mediaframe::frame::{Yuv420pFrame, Yuv422pFrame};

const M: KernelMatrix = KernelMatrix::Bt601;
const FR: bool = true;

/// Per-channel ramps so every filter window sees distinct neighbours (a
/// channel mix-up or a row/column transpose diverges immediately). All
/// samples interior so the conversions see real math. 4:2:0 chroma is
/// `cw = sw / 2`, `ch = sh / 2`.
fn ramp_420(sw: usize, sh: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let (cw, ch) = (sw / 2, sh / 2);
  let mut y = vec![0u8; sw * sh];
  let mut u = vec![0u8; cw * ch];
  let mut v = vec![0u8; cw * ch];
  for (i, p) in y.iter_mut().enumerate() {
    *p = (40 + (i % 100) * 2) as u8;
  }
  for (i, p) in u.iter_mut().enumerate() {
    *p = (70 + (i % 30) * 5) as u8;
  }
  for (i, p) in v.iter_mut().enumerate() {
    *p = (200u8).wrapping_sub(((i % 40) * 4) as u8);
  }
  (y, u, v)
}

/// Every resampled output a Bicublin assertion inspects.
struct Outputs {
  rgb: Vec<u8>,
  rgba: Vec<u8>,
  luma: Vec<u8>,
  luma_u16: Vec<u16>,
}

/// Runs the `Yuv420p` Bicublin sink over the planes (`sw x sh`) at
/// `ow x oh`, attaching every output the assertions inspect.
fn bicublin_outputs(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
) -> Outputs {
  let cw = (sw / 2) as u32;
  let src = Yuv420pFrame::new(y, u, v, sw as u32, sh as u32, sw as u32, cw, cw);
  let mut rgb = vec![0u8; ow * oh * 3];
  let mut rgba = vec![0u8; ow * oh * 4];
  let mut luma = vec![0u8; ow * oh];
  let mut luma_u16 = vec![0u16; ow * oh];
  {
    let mut sink = MixedSinker::<Yuv420p, Bicublin>::with_resampler(sw, sh, Bicublin::to(ow, oh))
      .unwrap()
      .with_rgb(&mut rgb)
      .unwrap()
      .with_rgba(&mut rgba)
      .unwrap()
      .with_luma(&mut luma)
      .unwrap()
      .with_luma_u16(&mut luma_u16)
      .unwrap();
    yuv420p_to(&src, FR, sink.set_kernel_matrix(M)).unwrap();
  }
  Outputs {
    rgb,
    rgba,
    luma,
    luma_u16,
  }
}

/// Single-plane filter of a `pw x ph -> ow x oh` u8 plane through the merged
/// engine's [`FilterStream<u8>`] (channels = 1) under `kernel` — the
/// per-plane oracle primitive. Built from [`ResamplePlan::filter`] directly
/// (NOT the [`FilteredResampler`] strategy, which short-circuits the
/// `in == out` identity to `None`), so the windows match
/// [`ResamplePlan::bicublin`]'s per-plane `FilterAxis::build` byte-for-byte
/// even when a plane's resolution equals the output (e.g. 4:2:0 chroma `4x4`
/// to an `4x4` output, where the chroma filter is a unit-weight identity the
/// engine still evaluates rather than skips).
fn plane_filter<K: crate::resample::FilterKernel>(
  kernel: K,
  plane: &[u8],
  pw: usize,
  ph: usize,
  ow: usize,
  oh: usize,
) -> Vec<u8> {
  let plan = crate::resample::ResamplePlan::filter(pw, ph, ow, oh, &kernel).expect("filter plan");
  let fh = plan.filter_h().expect("h windows");
  let fv = plan.filter_v().expect("v windows");
  let mut stream = FilterStream::<u8>::new(fh, fv, pw, ph, 1).expect("geometry");
  let mut out = vec![0u8; ow * oh];
  for row in 0..ph {
    stream
      .feed_row(row, &plane[row * pw..(row + 1) * pw], true, |oy, fin| {
        out[oy * ow..(oy + 1) * ow].copy_from_slice(fin);
      })
      .expect("rows in order");
  }
  out
}

/// The INDEPENDENT per-plane Bicublin oracle: filter Y with the cubic kernel
/// (`sw x sh -> ow x oh`), filter U / V with the linear kernel
/// (`cw x ch -> ow x oh`), then convert the filtered Y / U / V to RGB at the
/// output grid through `yuv_444_to_rgb_row`. Returns `(rgb, luma)`.
fn per_plane_oracle(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
) -> (Vec<u8>, Vec<u8>) {
  let (cw, ch) = (sw / 2, sh / 2);
  let y_f = plane_filter(SwscaleBicubic, y, sw, sh, ow, oh);
  let u_f = plane_filter(Triangle, u, cw, ch, ow, oh);
  let v_f = plane_filter(Triangle, v, cw, ch, ow, oh);
  let mut rgb = vec![0u8; ow * oh * 3];
  for oy in 0..oh {
    crate::row::yuv_444_to_rgb_row(
      &y_f[oy * ow..(oy + 1) * ow],
      &u_f[oy * ow..(oy + 1) * ow],
      &v_f[oy * ow..(oy + 1) * ow],
      &mut rgb[oy * ow * 3..(oy + 1) * ow * 3],
      ow,
      M,
      FR,
      true,
    );
  }
  (rgb, y_f)
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bicublin_equals_independent_per_plane_oracle() {
  for &(sw, sh, ow, oh) in &[
    (8usize, 8usize, 4usize, 4usize),
    (4, 4, 7, 7),
    (16, 12, 6, 5),
  ] {
    let (y, u, v) = ramp_420(sw, sh);
    let got = bicublin_outputs(&y, &u, &v, sw, sh, ow, oh);
    let (want_rgb, want_luma) = per_plane_oracle(&y, &u, &v, sw, sh, ow, oh);

    for (i, (&g, &w)) in got.rgb.iter().zip(want_rgb.iter()).enumerate() {
      assert_eq!(
        g, w,
        "bicublin {sw}x{sh}->{ow}x{oh} rgb[{i}]: {g} vs per-plane oracle {w}"
      );
    }
    for (i, (&g, &w)) in got.luma.iter().zip(want_luma.iter()).enumerate() {
      assert_eq!(
        g, w,
        "bicublin {sw}x{sh}->{ow}x{oh} luma[{i}]: {g} vs cubic native-Y {w}"
      );
    }
    // rgba colour == rgb, opaque alpha (0xFF).
    for (px, c) in got.rgba.chunks_exact(4).zip(want_rgb.chunks_exact(3)) {
      assert_eq!(&px[..3], c, "bicublin rgba colour == rgb");
      assert_eq!(px[3], 0xFF, "bicublin rgba opaque alpha");
    }
    // luma_u16 is the resampled cubic Y zero-extended.
    for (&lo, &hi) in got.luma.iter().zip(got.luma_u16.iter()) {
      assert_eq!(hi, lo as u16, "bicublin luma_u16 == luma zero-extended");
    }
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bicublin_luma_chroma_kernels_differ_from_single_kernel() {
  // A geometry where cubic and linear windows genuinely diverge.
  let (sw, sh, ow, oh) = (16usize, 12usize, 6usize, 5usize);
  let (y, u, v) = ramp_420(sw, sh);
  let got = bicublin_outputs(&y, &u, &v, sw, sh, ow, oh);

  // Single-CUBIC oracle: BOTH planes filtered with the cubic kernel, then
  // converted. Differs from Bicublin only in the chroma kernel, so any
  // difference proves the chroma plane really uses the linear kernel.
  let single_cubic = {
    let (cw, ch) = (sw / 2, sh / 2);
    let y_f = plane_filter(SwscaleBicubic, &y, sw, sh, ow, oh);
    let u_f = plane_filter(SwscaleBicubic, &u, cw, ch, ow, oh);
    let v_f = plane_filter(SwscaleBicubic, &v, cw, ch, ow, oh);
    let mut rgb = vec![0u8; ow * oh * 3];
    for oy in 0..oh {
      crate::row::yuv_444_to_rgb_row(
        &y_f[oy * ow..(oy + 1) * ow],
        &u_f[oy * ow..(oy + 1) * ow],
        &v_f[oy * ow..(oy + 1) * ow],
        &mut rgb[oy * ow * 3..(oy + 1) * ow * 3],
        ow,
        M,
        FR,
        true,
      );
    }
    rgb
  };

  // Single-LINEAR oracle: BOTH planes filtered with the linear kernel.
  // Differs from Bicublin only in the luma kernel.
  let single_linear = {
    let (cw, ch) = (sw / 2, sh / 2);
    let y_f = plane_filter(Triangle, &y, sw, sh, ow, oh);
    let u_f = plane_filter(Triangle, &u, cw, ch, ow, oh);
    let v_f = plane_filter(Triangle, &v, cw, ch, ow, oh);
    let mut rgb = vec![0u8; ow * oh * 3];
    for oy in 0..oh {
      crate::row::yuv_444_to_rgb_row(
        &y_f[oy * ow..(oy + 1) * ow],
        &u_f[oy * ow..(oy + 1) * ow],
        &v_f[oy * ow..(oy + 1) * ow],
        &mut rgb[oy * ow * 3..(oy + 1) * ow * 3],
        ow,
        M,
        FR,
        true,
      );
    }
    rgb
  };

  assert_ne!(
    got.rgb, single_cubic,
    "bicublin must differ from a single-cubic filter (chroma is linear, not cubic)"
  );
  assert_ne!(
    got.rgb, single_linear,
    "bicublin must differ from a single-linear filter (luma is cubic, not linear)"
  );
}

// ---- Atomicity (mirrors the single-kernel filter-path atomicity) ------
//
// Rows are constructed directly via `Yuv420pRow::for_tests(&y, &u, &v, idx, …)`
// (the `u` / `v` slices are the half-width chroma row for that Y row's pair,
// read only on even rows), so a single `process` call can be replayed and a
// mid-stream output swap exercised — the same pattern the geometry tests use.

/// The half-width chroma row for Y row `idx`'s pair: chroma row `idx / 2`.
fn chroma_row(plane: &[u8], cw: usize, idx: usize) -> &[u8] {
  let c = idx / 2;
  &plane[c * cw..(c + 1) * cw]
}

/// A mid-frame output-set change is rejected with a typed error and leaves
/// the already-written outputs unmutated, retryable.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bicublin_mid_frame_output_change_is_rejected() {
  const SW: usize = 8;
  const SH: usize = 8;
  const OW: usize = 4;
  const OH: usize = 4;
  let (y, u, v) = ramp_420(SW, SH);
  let cw = SW / 2;

  let mut luma_a = vec![0u8; OW * OH];
  let mut luma_b = vec![0u8; OW * OH];
  let mut sink = MixedSinker::<Yuv420p, Bicublin>::with_resampler(SW, SH, Bicublin::to(OW, OH))
    .unwrap()
    .with_luma(&mut luma_a)
    .unwrap();
  sink.begin_frame(SW as u32, SH as u32).unwrap();
  let yr = |i: usize| &y[i * SW..(i + 1) * SW];
  // Feed row 0, then swap the luma buffer and feed row 1 — the frozen-output
  // contract must reject the changed output set.
  sink
    .process(Yuv420pRow::for_tests(
      yr(0),
      chroma_row(&u, cw, 0),
      chroma_row(&v, cw, 0),
      0,
      M,
      FR,
    ))
    .unwrap();
  sink.set_luma(&mut luma_b).unwrap();
  let err = sink
    .process(Yuv420pRow::for_tests(
      yr(1),
      chroma_row(&u, cw, 1),
      chroma_row(&v, cw, 1),
      1,
      M,
      FR,
    ))
    .unwrap_err();
  assert!(
    matches!(err, MixedSinkerError::ResampleOutputsChanged(_)),
    "mid-frame output change must be ResampleOutputsChanged, got {err:?}"
  );
}

/// An out-of-sequence row is rejected with `OutOfSequenceRow` and the
/// outputs are unmutated — the same row can then be retried in order.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bicublin_out_of_sequence_row_is_rejected() {
  const SW: usize = 8;
  const SH: usize = 8;
  const OW: usize = 4;
  const OH: usize = 4;
  let (y, u, v) = ramp_420(SW, SH);
  let cw = SW / 2;
  let yr = |i: usize| &y[i * SW..(i + 1) * SW];

  let mut luma = vec![0u8; OW * OH];
  {
    let mut sink = MixedSinker::<Yuv420p, Bicublin>::with_resampler(SW, SH, Bicublin::to(OW, OH))
      .unwrap()
      .with_luma(&mut luma)
      .unwrap();
    sink.begin_frame(SW as u32, SH as u32).unwrap();
    // Skip row 0, feed row 1 first: rejected out of sequence, mutating nothing.
    let err = sink
      .process(Yuv420pRow::for_tests(
        yr(1),
        chroma_row(&u, cw, 1),
        chroma_row(&v, cw, 1),
        1,
        M,
        FR,
      ))
      .unwrap_err();
    assert!(
      matches!(
        err,
        MixedSinkerError::Resample(crate::resample::ResampleError::OutOfSequenceRow(_))
      ),
      "out-of-sequence row must be OutOfSequenceRow, got {err:?}"
    );
    // The rejected row poisoned no state: the SAME frame replays in order and
    // produces the correct cubic native-Y result below.
    for i in 0..SH {
      sink
        .process(Yuv420pRow::for_tests(
          yr(i),
          chroma_row(&u, cw, i),
          chroma_row(&v, cw, i),
          i,
          M,
          FR,
        ))
        .unwrap();
    }
  }
  let want = plane_filter(SwscaleBicubic, &y, SW, SH, OW, OH);
  assert_eq!(
    luma, want,
    "retried-in-order luma == cubic native-Y filter (rejected OOS row did not poison state)"
  );
}

// ---- Finding 2: same-size BICUBLIN still filters all four axes -----------

/// A SAME-SIZE BICUBLIN (`in_w == out_w`, `in_h == out_h`) is NOT the identity:
/// the half-resolution chroma plane is still UPSAMPLED to the output grid with
/// the linear kernel, and the luma plane is a same-size cubic convolution. The
/// pre-fix bug returned `Ok(None)` for `in == out`, which dropped the per-plane
/// filters entirely and took the direct conversion path. This proves the
/// same-size path now (a) equals the per-plane oracle bit-for-bit and (b)
/// differs from that direct/identity path — the chroma was genuinely
/// linear-filter-upsampled, not skipped.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bicublin_same_size_still_filters_all_planes() {
  for &(sw, sh) in &[(8usize, 8usize), (16, 12), (6, 10)] {
    // Same-size: output geometry equals the source luma geometry.
    let (ow, oh) = (sw, sh);
    let (y, u, v) = ramp_420(sw, sh);
    let got = bicublin_outputs(&y, &u, &v, sw, sh, ow, oh);
    let (want_rgb, want_luma) = per_plane_oracle(&y, &u, &v, sw, sh, ow, oh);

    // The same-size BICUBLIN matches the per-plane oracle bit-for-bit — the
    // chroma plane was upsampled with the linear kernel and the luma plane was
    // convolved with the cubic kernel; neither filter was skipped.
    assert_eq!(
      got.rgb, want_rgb,
      "same-size {sw}x{sh} bicublin rgb == per-plane oracle (chroma still linear-filtered)"
    );
    assert_eq!(
      got.luma, want_luma,
      "same-size {sw}x{sh} bicublin luma == cubic native-Y (luma still cubic-filtered)"
    );

    // Non-identity proof: the same-size BICUBLIN rgb differs from the direct
    // identity conversion (the pre-fix `Ok(None)` path), whose chroma upsample
    // is the row kernel's register interpolation, NOT the linear filter. A
    // skipped chroma filter would reproduce that identity rgb; a real linear
    // filter does not.
    let cw_u = (sw / 2) as u32;
    let src = Yuv420pFrame::new(&y, &u, &v, sw as u32, sh as u32, sw as u32, cw_u, cw_u);
    let mut identity_rgb = vec![0u8; ow * oh * 3];
    {
      let mut sink = MixedSinker::<Yuv420p>::new(sw, sh)
        .with_rgb(&mut identity_rgb)
        .unwrap();
      yuv420p_to(&src, FR, sink.set_kernel_matrix(M)).unwrap();
    }
    assert_ne!(
      got.rgb, identity_rgb,
      "same-size {sw}x{sh} bicublin rgb must differ from the identity convert (chroma is \
       linear-filter-upsampled, not the dropped-filter pass-through)"
    );
  }
}

// ---- Finding 1: a BICUBLIN plan is rejected by a non-Yuv420p filter sink -

/// A BICUBLIN plan keeps `kind() == Filter` (no dedicated span-kind variant)
/// and is reachable ONLY by the `Yuv420p` per-plane route. Handed to any OTHER
/// format's single-kernel filter path it carries a chroma window set that path
/// would silently ignore (mis-filtering every plane with the luma cubic
/// kernel), so the path must REJECT it with the typed `UnsupportedFilter`
/// rather than produce a silently-wrong single-kernel result. Here a `Yuv422p`
/// sink (4:2:2 — its filter dispatch routes through the shared
/// `planar_dual_filter_resample`) carries a `Bicublin` resampler; processing a
/// frame must surface `UnsupportedFilter`, never run.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bicublin_plan_rejected_by_non_yuv420p_filter_sink() {
  const SW: usize = 8;
  const SH: usize = 8;
  const OW: usize = 4;
  const OH: usize = 4;
  // 4:2:2 chroma: half width, FULL height.
  let (cw, ch) = (SW / 2, SH);
  let y = vec![60u8; SW * SH];
  let u = vec![90u8; cw * ch];
  let v = vec![150u8; cw * ch];
  let src = Yuv422pFrame::new(
    &y, &u, &v, SW as u32, SH as u32, SW as u32, cw as u32, cw as u32,
  );
  let mut rgb = vec![0u8; OW * OH * 3];
  let mut sink = MixedSinker::<Yuv422p, Bicublin>::with_resampler(SW, SH, Bicublin::to(OW, OH))
    .expect("plan builds (the format only matters at process time)")
    .with_rgb(&mut rgb)
    .unwrap();
  let err = yuv422p_to(&src, FR, sink.set_kernel_matrix(M)).unwrap_err();
  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::UnsupportedFilter(_))
    ),
    "a bicublin plan on a Yuv422p filter sink must be UnsupportedFilter, got {err:?}"
  );
  // The reject ran before any emit, so the output buffer is untouched.
  assert!(
    rgb.iter().all(|&b| b == 0),
    "a rejected bicublin plan must not touch the Yuv422p sink's output"
  );
}

// ---- Finding 3: the BICUBLIN join box-alloc is recoverable ---------------

/// The lazy `Box<BicublinYuv420>` is taken through the recoverable `try_box`
/// (not `Box::new`, which aborts on OOM): an allocator refusal surfaces the
/// typed `AllocationFailed` with the `bicublin_420` field left `None` (the
/// first-row-transactional contract), so a fresh frame after the single-shot
/// failpoint is consumed resamples cleanly.
#[test]
#[cfg(feature = "std")]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bicublin_box_alloc_failure_is_recoverable() {
  const SW: usize = 8;
  const SH: usize = 8;
  const OW: usize = 4;
  const OH: usize = 4;
  let (y, u, v) = ramp_420(SW, SH);
  let cw = SW / 2;
  let yr = |i: usize| &y[i * SW..(i + 1) * SW];

  // Scope 1: arm the single-shot box-alloc failpoint, then feed row 0. The join
  // build's outer box refusal must surface AllocationFailed (not abort), with
  // the output untouched and the field left None (retryable).
  let mut rgb = vec![0u8; OW * OH * 3];
  {
    let mut sink = MixedSinker::<Yuv420p, Bicublin>::with_resampler(SW, SH, Bicublin::to(OW, OH))
      .unwrap()
      .with_rgb(&mut rgb)
      .unwrap();
    sink.begin_frame(SW as u32, SH as u32).unwrap();
    crate::resample::arm_box_failure();
    let err = sink
      .process(Yuv420pRow::for_tests(
        yr(0),
        chroma_row(&u, cw, 0),
        chroma_row(&v, cw, 0),
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
      "first-row box-alloc refusal must surface AllocationFailed (not abort), got {err:?}"
    );
  }
  assert!(
    rgb.iter().all(|&b| b == 0),
    "a box-alloc-refused first row must not touch the output"
  );

  // Scope 2: a fresh frame after the single-shot failpoint is consumed must
  // resample cleanly and equal the per-plane oracle — the refusal was
  // recoverable, not a one-way poison (the field is left None, so the join
  // allocates afresh).
  let mut rgb2 = vec![0u8; OW * OH * 3];
  {
    let cw_u = (SW / 2) as u32;
    let src = Yuv420pFrame::new(&y, &u, &v, SW as u32, SH as u32, SW as u32, cw_u, cw_u);
    let mut sink = MixedSinker::<Yuv420p, Bicublin>::with_resampler(SW, SH, Bicublin::to(OW, OH))
      .unwrap()
      .with_rgb(&mut rgb2)
      .unwrap();
    yuv420p_to(&src, FR, sink.set_kernel_matrix(M))
      .expect("a fresh frame after the consumed failpoint resamples cleanly");
  }
  let (want_rgb, _) = per_plane_oracle(&y, &u, &v, SW, SH, OW, OH);
  assert_eq!(
    rgb2, want_rgb,
    "the recovered frame equals the per-plane oracle (box-alloc refusal was recoverable)"
  );
}

/// The BICUBLIN delegate grows the output RGB scratch as a PRE-FEED step,
/// BEFORE it commits the output-set freeze. A scratch OOM on an ordinary
/// first-ever row-0 build therefore leaves `resample_outputs` uncommitted (the
/// delegate never freezes on a pre-feed failure), so a retry of row 0 with a
/// CHANGED output attachment is accepted, not mis-rejected as
/// ResampleOutputsChanged. Mirrors the native area tier's fold.
#[test]
#[cfg(all(feature = "std", feature = "rgb"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bicublin_first_build_scratch_oom_leaves_freeze_unfrozen_for_retry() {
  const SW: usize = 8;
  const SH: usize = 8;
  const OW: usize = 4;
  const OH: usize = 4;
  let (y, u, v) = ramp_420(SW, SH);
  let cw = SW / 2;
  let yr = |i: usize| &y[i * SW..(i + 1) * SW];

  let mut rgb = vec![0u8; OW * OH * 3];
  let mut luma = vec![0u8; OW * OH];
  let mut sink = MixedSinker::<Yuv420p, Bicublin>::with_resampler(SW, SH, Bicublin::to(OW, OH))
    .unwrap()
    .with_rgb(&mut rgb)
    .unwrap();
  sink.begin_frame(SW as u32, SH as u32).unwrap();
  crate::sinker::mixed::arm_native_rgb_scratch_failure();
  let err = sink
    .process(Yuv420pRow::for_tests(
      yr(0),
      chroma_row(&u, cw, 0),
      chroma_row(&v, cw, 0),
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
    .process(Yuv420pRow::for_tests(
      yr(0),
      chroma_row(&u, cw, 0),
      chroma_row(&v, cw, 0),
      0,
      M,
      FR,
    ))
    .expect("row 0 must succeed after a first-build scratch OOM (freeze uncommitted)");
}

/// A cross-frame COLOUR-CAPABILITY rebuild for the BICUBLIN delegate: frame 1
/// builds a luma-only join (chroma absent, no output RGB scratch); frame 2
/// attaches RGB, so the delegate rebuilds the join WITH chroma into a local and
/// grows the (empty) output RGB scratch before committing. A scratch OOM on that
/// rebuild must leave the cached luma-only join intact AND `resample_outputs`
/// uncommitted, so a row-0 retry is accepted, not mis-rejected as
/// ResampleOutputsChanged.
#[test]
#[cfg(all(feature = "std", feature = "rgb"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn bicublin_colour_capability_rebuild_scratch_oom_leaves_freeze_unfrozen_for_retry() {
  const SW: usize = 8;
  const SH: usize = 8;
  const OW: usize = 4;
  const OH: usize = 4;
  let (y, u, v) = ramp_420(SW, SH);
  let cw = SW / 2;
  let yr = |i: usize| &y[i * SW..(i + 1) * SW];

  let mut luma = vec![0u8; OW * OH];
  let mut rgb = vec![0u8; OW * OH * 3];
  let (mut hh, mut ss, mut vv) = (vec![0u8; OW * OH], vec![0u8; OW * OH], vec![0u8; OW * OH]);
  let mut sink = MixedSinker::<Yuv420p, Bicublin>::with_resampler(SW, SH, Bicublin::to(OW, OH))
    .unwrap()
    .with_luma(&mut luma)
    .unwrap();
  // Frame 1: a luma-only join (chroma absent, reserves no RGB scratch).
  sink.begin_frame(SW as u32, SH as u32).unwrap();
  for i in 0..SH {
    sink
      .process(Yuv420pRow::for_tests(
        yr(i),
        chroma_row(&u, cw, i),
        chroma_row(&v, cw, i),
        i,
        M,
        FR,
      ))
      .expect("luma-only frame builds the bicublin join");
  }
  // Frame 2: attach RGB → the row-0 rebuild must build the chroma half and grow
  // the (now-needed, empty) output RGB scratch; refuse that grow.
  sink.begin_frame(SW as u32, SH as u32).unwrap();
  sink.set_rgb(&mut rgb).unwrap();
  crate::sinker::mixed::arm_native_rgb_scratch_failure();
  let err = sink
    .process(Yuv420pRow::for_tests(
      yr(0),
      chroma_row(&u, cw, 0),
      chroma_row(&v, cw, 0),
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
    .process(Yuv420pRow::for_tests(
      yr(0),
      chroma_row(&u, cw, 0),
      chroma_row(&v, cw, 0),
      0,
      M,
      FR,
    ))
    .expect("row 0 with a changed output set must succeed after a rebuild scratch OOM");
}

/// The directly-boxed per-plane `*_stream` fields take the same recoverable
/// `try_box` path as the native joins: a box-alloc refusal on the first row of
/// the row-stage area route (`Yuv420p` + `AreaResampler`, native tier off, so
/// `planar_dual_resample` boxes the `luma_stream` / `rgb_stream` fields) must
/// surface the typed `AllocationFailed` (NOT abort) with the output untouched
/// and the field left `None` (the first-row-transactional contract), so a fresh
/// frame after the single-shot failpoint resamples cleanly.
#[test]
#[cfg(feature = "std")]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn area_stream_box_alloc_failure_is_recoverable() {
  const SW: usize = 8;
  const SH: usize = 8;
  const OW: usize = 4;
  const OH: usize = 4;
  let (y, u, v) = ramp_420(SW, SH);
  let cw = SW / 2;
  let yr = |i: usize| &y[i * SW..(i + 1) * SW];

  // Scope 1: arm the single-shot box-alloc failpoint, then feed row 0. The
  // boxed-stream refusal must surface AllocationFailed (not abort), with the
  // output untouched and the stream field left None (retryable).
  let mut rgb = vec![0u8; OW * OH * 3];
  {
    let mut sink =
      MixedSinker::<Yuv420p, AreaResampler>::with_resampler(SW, SH, AreaResampler::to(OW, OH))
        .unwrap()
        .with_native(false)
        .with_rgb(&mut rgb)
        .unwrap();
    sink.begin_frame(SW as u32, SH as u32).unwrap();
    crate::resample::arm_box_failure();
    let err = sink
      .process(Yuv420pRow::for_tests(
        yr(0),
        chroma_row(&u, cw, 0),
        chroma_row(&v, cw, 0),
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
      "first-row stream box-alloc refusal must surface AllocationFailed (not abort), got {err:?}"
    );
  }
  assert!(
    rgb.iter().all(|&b| b == 0),
    "a box-alloc-refused first row must not touch the output"
  );

  // Scope 2: a fresh frame after the single-shot failpoint is consumed must
  // resample cleanly — the refusal was recoverable, not a one-way poison (the
  // stream field is left None, so it allocates afresh on the next frame).
  let mut rgb2 = vec![0u8; OW * OH * 3];
  {
    let cw_u = (SW / 2) as u32;
    let src = Yuv420pFrame::new(&y, &u, &v, SW as u32, SH as u32, SW as u32, cw_u, cw_u);
    let mut sink =
      MixedSinker::<Yuv420p, AreaResampler>::with_resampler(SW, SH, AreaResampler::to(OW, OH))
        .unwrap()
        .with_native(false)
        .with_rgb(&mut rgb2)
        .unwrap();
    yuv420p_to(&src, FR, sink.set_kernel_matrix(M))
      .expect("a fresh frame after the consumed failpoint resamples cleanly");
  }
  assert!(
    rgb2.iter().any(|&b| b != 0),
    "the recovered frame produced real output (box-alloc refusal was recoverable)"
  );
}
