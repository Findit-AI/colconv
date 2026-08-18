//! Sinker integration tests for `MixedSinker<Vyu444>`.
//!
//! VYU444 is the no-alpha, 3-byte (`V, Y, U`) sibling of `Vuyx`. The
//! headline oracle packs the SAME logical V/U/Y samples into both the
//! 3-byte VYU444 and the 4-byte VUYX (padding) byte streams and asserts
//! the sink outputs (RGB / RGBA-with-α=0xFF / luma / luma_u16 / HSV, plus
//! resampled outputs through the native and row-stage tiers) are
//! byte-identical. Plus smoke, buffer-shape errors, and SIMD-vs-scalar
//! parity.

#[cfg(all(test, feature = "std"))]
use super::*;

/// Pack logical V/U/Y samples into a VYU444 byte stream (`[V, Y, U]`,
/// 3 bytes per pixel).
#[cfg(all(test, feature = "std"))]
fn pack_vyu444(v: &[u8], u: &[u8], y: &[u8], n: usize) -> Vec<u8> {
  let mut out = std::vec![0u8; n * 3];
  for i in 0..n {
    out[i * 3] = v[i];
    out[i * 3 + 1] = y[i];
    out[i * 3 + 2] = u[i];
  }
  out
}

/// Pack the same samples into a VUYX byte stream (`[V, U, Y, X]`, X is
/// padding). The padding byte is set to a non-zero sentinel so that any
/// accidental read surfaces as a mismatch.
#[cfg(all(test, feature = "std"))]
fn pack_vuyx(v: &[u8], u: &[u8], y: &[u8], n: usize) -> Vec<u8> {
  let mut out = std::vec![0u8; n * 4];
  for i in 0..n {
    out[i * 4] = v[i];
    out[i * 4 + 1] = u[i];
    out[i * 4 + 2] = y[i];
    out[i * 4 + 3] = 0x5A; // padding sentinel
  }
  out
}

/// Headline cross-format oracle: VYU444 ↔ VUYX byte-identity for the same
/// logical samples across every standard output channel and both ranges.
#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn vyu444_matches_vuyx_reference_all_outputs() {
  let width = 64usize;
  let height = 4usize;
  let n = width * height;
  let (mut yp, mut up, mut vp) = (std::vec![0u8; n], std::vec![0u8; n], std::vec![0u8; n]);
  pseudo_random_u8(&mut yp, 0x11);
  pseudo_random_u8(&mut up, 0x22);
  pseudo_random_u8(&mut vp, 0x33);

  let vyu_buf = pack_vyu444(&vp, &up, &yp, n);
  let vuyx_buf = pack_vuyx(&vp, &up, &yp, n);
  let vy = Vyu444Frame::try_new(&vyu_buf, width as u32, height as u32, (width * 3) as u32).unwrap();
  let vx = VuyxFrame::try_new(&vuyx_buf, width as u32, height as u32, (width * 4) as u32).unwrap();

  for full_range in [true, false] {
    let m = KernelMatrix::Bt709;
    // RGB
    let (mut y_rgb, mut x_rgb) = (std::vec![0u8; n * 3], std::vec![0u8; n * 3]);
    {
      let mut a = MixedSinker::<Vyu444>::new(width, height)
        .with_rgb(&mut y_rgb)
        .unwrap();
      vyu444_to(&vy, full_range, m, &mut a).unwrap();
      let mut b = MixedSinker::<Vuyx>::new(width, height)
        .with_rgb(&mut x_rgb)
        .unwrap();
      vuyx_to(&vx, full_range, m, &mut b).unwrap();
    }
    assert_eq!(y_rgb, x_rgb, "VYU444↔VUYX RGB (full_range={full_range})");

    // RGBA (α forced 0xFF for both)
    let (mut y_rgba, mut x_rgba) = (std::vec![0u8; n * 4], std::vec![0u8; n * 4]);
    {
      let mut a = MixedSinker::<Vyu444>::new(width, height)
        .with_rgba(&mut y_rgba)
        .unwrap();
      vyu444_to(&vy, full_range, m, &mut a).unwrap();
      let mut b = MixedSinker::<Vuyx>::new(width, height)
        .with_rgba(&mut x_rgba)
        .unwrap();
      vuyx_to(&vx, full_range, m, &mut b).unwrap();
    }
    assert_eq!(y_rgba, x_rgba, "VYU444↔VUYX RGBA (full_range={full_range})");
    for i in 0..n {
      assert_eq!(y_rgba[i * 4 + 3], 0xFF, "VYU444 α at {i} must be 0xFF");
    }
  }

  // Luma / luma_u16 / HSV.
  let m = KernelMatrix::Bt709;
  let (mut y_l, mut x_l) = (std::vec![0u8; n], std::vec![0u8; n]);
  {
    let mut a = MixedSinker::<Vyu444>::new(width, height)
      .with_luma(&mut y_l)
      .unwrap();
    vyu444_to(&vy, false, m, &mut a).unwrap();
    let mut b = MixedSinker::<Vuyx>::new(width, height)
      .with_luma(&mut x_l)
      .unwrap();
    vuyx_to(&vx, false, m, &mut b).unwrap();
  }
  assert_eq!(y_l, x_l, "VYU444↔VUYX luma");
  for i in 0..n {
    assert_eq!(y_l[i], yp[i], "VYU444 luma at {i}");
  }

  let (mut y_l16, mut x_l16) = (std::vec![0u16; n], std::vec![0u16; n]);
  {
    let mut a = MixedSinker::<Vyu444>::new(width, height)
      .with_luma_u16(&mut y_l16)
      .unwrap();
    vyu444_to(&vy, false, m, &mut a).unwrap();
    let mut b = MixedSinker::<Vuyx>::new(width, height)
      .with_luma_u16(&mut x_l16)
      .unwrap();
    vuyx_to(&vx, false, m, &mut b).unwrap();
  }
  assert_eq!(y_l16, x_l16, "VYU444↔VUYX luma_u16");

  let (mut yh, mut ys, mut yv) = (std::vec![0u8; n], std::vec![0u8; n], std::vec![0u8; n]);
  let (mut xh, mut xs, mut xv) = (std::vec![0u8; n], std::vec![0u8; n], std::vec![0u8; n]);
  {
    let mut a = MixedSinker::<Vyu444>::new(width, height)
      .with_hsv(&mut yh, &mut ys, &mut yv)
      .unwrap();
    vyu444_to(&vy, true, m, &mut a).unwrap();
    let mut b = MixedSinker::<Vuyx>::new(width, height)
      .with_hsv(&mut xh, &mut xs, &mut xv)
      .unwrap();
    vuyx_to(&vx, true, m, &mut b).unwrap();
  }
  assert_eq!((yh, ys, yv), (xh, xs, xv), "VYU444↔VUYX HSV");
}

/// Resample (2:1 downscale) parity: VYU444 and VUYX must produce
/// byte-identical RGB and luma through the area resample tier (this
/// exercises VYU444's 3-byte native-tier de-pack against VUYX's 4-byte
/// de-pack on the shared `Yuv444p` planar join).
#[cfg(all(test, feature = "std", any(feature = "std", feature = "alloc")))]
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn vyu444_downscale_matches_vuyx() {
  let in_w = 32usize;
  let in_h = 8usize;
  let out_w = 16usize;
  let out_h = 4usize;
  let n = in_w * in_h;
  let (mut yp, mut up, mut vp) = (std::vec![0u8; n], std::vec![0u8; n], std::vec![0u8; n]);
  pseudo_random_u8(&mut yp, 0x77);
  pseudo_random_u8(&mut up, 0x88);
  pseudo_random_u8(&mut vp, 0x99);

  let vyu_buf = pack_vyu444(&vp, &up, &yp, n);
  let vuyx_buf = pack_vuyx(&vp, &up, &yp, n);
  let vy = Vyu444Frame::try_new(&vyu_buf, in_w as u32, in_h as u32, (in_w * 3) as u32).unwrap();
  let vx = VuyxFrame::try_new(&vuyx_buf, in_w as u32, in_h as u32, (in_w * 4) as u32).unwrap();

  let m = KernelMatrix::Bt709;
  // RGB + luma through a 2:1 area downscale.
  let (mut y_rgb, mut x_rgb) = (
    std::vec![0u8; out_w * out_h * 3],
    std::vec![0u8; out_w * out_h * 3],
  );
  let (mut y_l, mut x_l) = (std::vec![0u8; out_w * out_h], std::vec![0u8; out_w * out_h]);
  {
    let mut a = MixedSinker::<Vyu444, crate::resample::AreaResampler>::with_resampler(
      in_w,
      in_h,
      crate::resample::AreaResampler::to(out_w, out_h),
    )
    .unwrap()
    .with_rgb(&mut y_rgb)
    .unwrap()
    .with_luma(&mut y_l)
    .unwrap();
    vyu444_to(&vy, false, m, &mut a).unwrap();
    let mut b = MixedSinker::<Vuyx, crate::resample::AreaResampler>::with_resampler(
      in_w,
      in_h,
      crate::resample::AreaResampler::to(out_w, out_h),
    )
    .unwrap()
    .with_rgb(&mut x_rgb)
    .unwrap()
    .with_luma(&mut x_l)
    .unwrap();
    vuyx_to(&vx, false, m, &mut b).unwrap();
  }
  assert_eq!(y_l, x_l, "VYU444↔VUYX downscaled luma");
  assert_eq!(y_rgb, x_rgb, "VYU444↔VUYX downscaled RGB");
}

/// SIMD vs scalar parity at a width spanning every backend's block + tail.
#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn vyu444_simd_vs_scalar_parity_at_1922() {
  let w = 1922usize;
  let h = 2usize;
  let mut buf = std::vec![0u8; w * h * 3];
  pseudo_random_u8(&mut buf, 0xFADE);
  let src = Vyu444Frame::try_new(&buf, w as u32, h as u32, (w * 3) as u32).unwrap();

  let mut simd = std::vec![0u8; w * h * 4];
  let mut scalar = std::vec![0u8; w * h * 4];
  {
    let mut s = MixedSinker::<Vyu444>::new(w, h)
      .with_rgba(&mut simd)
      .unwrap();
    vyu444_to(&src, false, KernelMatrix::Bt709, &mut s).unwrap();
  }
  {
    let mut s = MixedSinker::<Vyu444>::new(w, h)
      .with_rgba(&mut scalar)
      .unwrap()
      .with_simd(false);
    vyu444_to(&src, false, KernelMatrix::Bt709, &mut s).unwrap();
  }
  assert_eq!(simd, scalar, "VYU444 SIMD ≠ scalar at width {w}");
}

/// A malformed packed row (length ≠ `width × 3`) surfaces the typed
/// `RowShapeMismatch` carrying the `Vyu444Packed` slice tag (3-byte stride).
#[test]
#[cfg(all(test, feature = "std"))]
fn vyu444_malformed_row_returns_row_shape_mismatch() {
  let mut rgb = std::vec![0u8; 4 * 3];
  let mut sink = MixedSinker::<Vyu444>::new(4, 1).with_rgb(&mut rgb).unwrap();
  sink.begin_frame(4, 1).unwrap();
  // Width 4 needs 12 packed bytes; hand 10.
  let short = std::vec![0u8; 10];
  let row = Vyu444Row::new(&short, 0, KernelMatrix::Bt709, false);
  let err = sink.process(row).unwrap_err();
  assert!(
    matches!(err, MixedSinkerError::RowShapeMismatch(e)
      if e.which() == RowSlice::Vyu444Packed && e.expected() == 12 && e.actual() == 10),
    "unexpected error: {err:?}"
  );
}

// ---- Atomicity (#308) --------------------------------------------------
//
// The packed 4:4:4 identity-path `process` must run the up-front RGB-scratch
// preflight BEFORE any output row (luma / luma_u16) is written, so an allocator
// refusal returns a recoverable `AllocationFailed` leaving the output frame
// untouched rather than partially mutated. The `luma + RGBA + HSV, no RGB`
// combo is the identity-path shape that reaches `rgb_row_buf_or_scratch`'s
// scratch arm (`want_hsv && want_rgba && !want_rgb`); standalone RGBA uses the
// dedicated `vyu444_to_rgba_row` kernel (α forced opaque) and never grows the
// scratch. Reuses the crate's `yuva`-gated RGB-scratch failpoint, so this test
// is too (under `--all-features` both `yuv-444-packed` and `yuva` are on).

#[cfg(feature = "yuva")]
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn vyu444_rgb_scratch_alloc_failure_leaves_outputs_untouched() {
  use crate::resample::ResampleError;

  let n = 16 * 8;
  let (yp, up, vp) = (
    std::vec![128u8; n],
    std::vec![128u8; n],
    std::vec![128u8; n],
  );
  let buf = pack_vyu444(&vp, &up, &yp, n);
  let src = Vyu444Frame::try_new(&buf, 16, 8, 16 * 3).unwrap();
  let mut luma = std::vec![0xABu8; n];
  let mut rgba = std::vec![0xCDu8; n * 4];
  let (mut hh, mut ss, mut vv) = (
    std::vec![0xCDu8; n],
    std::vec![0xCDu8; n],
    std::vec![0xCDu8; n],
  );
  let mut sink = MixedSinker::<Vyu444>::new(16, 8)
    .with_luma(&mut luma)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap()
    .with_hsv(&mut hh, &mut ss, &mut vv)
    .unwrap();

  super::super::arm_rgb_scratch_alloc_failure();
  let err = vyu444_to(&src, false, KernelMatrix::Bt601, &mut sink).unwrap_err();
  drop(sink);

  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
    ),
    "RGB-scratch refusal must surface as a recoverable AllocationFailed, got {err:?}"
  );
  assert!(
    luma.iter().all(|&b| b == 0xAB),
    "luma must be untouched on the rgb-scratch alloc-failure path"
  );
}
