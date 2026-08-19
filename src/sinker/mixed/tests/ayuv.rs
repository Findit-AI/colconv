//! Sinker integration tests for `MixedSinker<Ayuv>`.
//!
//! AYUV is the alpha-first (`A, Y, U, V`) channel re-ordering of `Vuya`.
//! The headline oracle packs the SAME logical V/U/Y/A samples in both the
//! AYUV and VUYA byte orders and asserts the sink outputs (RGB / RGBA /
//! luma / luma_u16 / HSV, plus resampled outputs) are byte-identical — so
//! AYUV inherits all of VUYA's validated behaviour. Plus smoke,
//! buffer-shape errors, and SIMD-vs-scalar parity.

#[cfg(all(test, feature = "std"))]
use super::*;

/// Pack logical V/U/Y/A samples into an AYUV byte stream (`[A, Y, U, V]`).
#[cfg(all(test, feature = "std"))]
fn pack_ayuv(v: &[u8], u: &[u8], y: &[u8], a: &[u8], n: usize) -> Vec<u8> {
  let mut out = std::vec![0u8; n * 4];
  for i in 0..n {
    out[i * 4] = a[i];
    out[i * 4 + 1] = y[i];
    out[i * 4 + 2] = u[i];
    out[i * 4 + 3] = v[i];
  }
  out
}

/// Pack the same samples into a VUYA byte stream (`[V, U, Y, A]`).
#[cfg(all(test, feature = "std"))]
fn pack_vuya(v: &[u8], u: &[u8], y: &[u8], a: &[u8], n: usize) -> Vec<u8> {
  let mut out = std::vec![0u8; n * 4];
  for i in 0..n {
    out[i * 4] = v[i];
    out[i * 4 + 1] = u[i];
    out[i * 4 + 2] = y[i];
    out[i * 4 + 3] = a[i];
  }
  out
}

/// Headline cross-format oracle: AYUV ↔ VUYA byte-identity for the same
/// logical samples across every standard output channel and both ranges.
#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn ayuv_matches_vuya_reference_all_outputs() {
  let width = 64usize;
  let height = 4usize;
  let n = width * height;
  let (mut yp, mut up, mut vp, mut ap) = (
    std::vec![0u8; n],
    std::vec![0u8; n],
    std::vec![0u8; n],
    std::vec![0u8; n],
  );
  pseudo_random_u8(&mut yp, 0xA1);
  pseudo_random_u8(&mut up, 0xB2);
  pseudo_random_u8(&mut vp, 0xC3);
  pseudo_random_u8(&mut ap, 0xD4);

  let ayuv_buf = pack_ayuv(&vp, &up, &yp, &ap, n);
  let vuya_buf = pack_vuya(&vp, &up, &yp, &ap, n);
  let ay = AyuvFrame::try_new(&ayuv_buf, width as u32, height as u32, (width * 4) as u32).unwrap();
  let vu = VuyaFrame::try_new(&vuya_buf, width as u32, height as u32, (width * 4) as u32).unwrap();

  for full_range in [true, false] {
    let m = KernelMatrix::Bt709;
    // RGB
    let (mut a_rgb, mut v_rgb) = (std::vec![0u8; n * 3], std::vec![0u8; n * 3]);
    {
      let mut a = MixedSinker::<Ayuv>::new(width, height)
        .with_rgb(&mut a_rgb)
        .unwrap();
      ayuv_to(&ay, full_range, a.set_kernel_matrix(m)).unwrap();
      let mut v = MixedSinker::<Vuya>::new(width, height)
        .with_rgb(&mut v_rgb)
        .unwrap();
      vuya_to(&vu, full_range, v.set_kernel_matrix(m)).unwrap();
    }
    assert_eq!(a_rgb, v_rgb, "AYUV↔VUYA RGB (full_range={full_range})");

    // RGBA (standalone — source α through the kernel)
    let (mut a_rgba, mut v_rgba) = (std::vec![0u8; n * 4], std::vec![0u8; n * 4]);
    {
      let mut a = MixedSinker::<Ayuv>::new(width, height)
        .with_rgba(&mut a_rgba)
        .unwrap();
      ayuv_to(&ay, full_range, a.set_kernel_matrix(m)).unwrap();
      let mut v = MixedSinker::<Vuya>::new(width, height)
        .with_rgba(&mut v_rgba)
        .unwrap();
      vuya_to(&vu, full_range, v.set_kernel_matrix(m)).unwrap();
    }
    assert_eq!(a_rgba, v_rgba, "AYUV↔VUYA RGBA (full_range={full_range})");
    // Alpha bytes equal source A.
    for i in 0..n {
      assert_eq!(a_rgba[i * 4 + 3], ap[i], "AYUV α at {i}");
    }

    // RGB + RGBA combo (Strategy path).
    let (mut a_crgb, mut a_crgba) = (std::vec![0u8; n * 3], std::vec![0u8; n * 4]);
    let (mut v_crgb, mut v_crgba) = (std::vec![0u8; n * 3], std::vec![0u8; n * 4]);
    {
      let mut a = MixedSinker::<Ayuv>::new(width, height)
        .with_rgb(&mut a_crgb)
        .unwrap()
        .with_rgba(&mut a_crgba)
        .unwrap();
      ayuv_to(&ay, full_range, a.set_kernel_matrix(m)).unwrap();
      let mut v = MixedSinker::<Vuya>::new(width, height)
        .with_rgb(&mut v_crgb)
        .unwrap()
        .with_rgba(&mut v_crgba)
        .unwrap();
      vuya_to(&vu, full_range, v.set_kernel_matrix(m)).unwrap();
    }
    assert_eq!(
      a_crgb, v_crgb,
      "AYUV↔VUYA combo RGB (full_range={full_range})"
    );
    assert_eq!(
      a_crgba, v_crgba,
      "AYUV↔VUYA combo RGBA (full_range={full_range})"
    );
  }

  // Luma / luma_u16 / HSV.
  let m = KernelMatrix::Bt709;
  let (mut a_l, mut v_l) = (std::vec![0u8; n], std::vec![0u8; n]);
  {
    let mut a = MixedSinker::<Ayuv>::new(width, height)
      .with_luma(&mut a_l)
      .unwrap();
    ayuv_to(&ay, false, a.set_kernel_matrix(m)).unwrap();
    let mut v = MixedSinker::<Vuya>::new(width, height)
      .with_luma(&mut v_l)
      .unwrap();
    vuya_to(&vu, false, v.set_kernel_matrix(m)).unwrap();
  }
  assert_eq!(a_l, v_l, "AYUV↔VUYA luma");
  // luma == source Y at offset 1.
  for i in 0..n {
    assert_eq!(a_l[i], yp[i], "AYUV luma at {i}");
  }

  let (mut a_l16, mut v_l16) = (std::vec![0u16; n], std::vec![0u16; n]);
  {
    let mut a = MixedSinker::<Ayuv>::new(width, height)
      .with_luma_u16(&mut a_l16)
      .unwrap();
    ayuv_to(&ay, false, a.set_kernel_matrix(m)).unwrap();
    let mut v = MixedSinker::<Vuya>::new(width, height)
      .with_luma_u16(&mut v_l16)
      .unwrap();
    vuya_to(&vu, false, v.set_kernel_matrix(m)).unwrap();
  }
  assert_eq!(a_l16, v_l16, "AYUV↔VUYA luma_u16");

  let (mut ah, mut as_, mut av) = (std::vec![0u8; n], std::vec![0u8; n], std::vec![0u8; n]);
  let (mut vh, mut vs, mut vv) = (std::vec![0u8; n], std::vec![0u8; n], std::vec![0u8; n]);
  {
    let mut a = MixedSinker::<Ayuv>::new(width, height)
      .with_hsv(&mut ah, &mut as_, &mut av)
      .unwrap();
    ayuv_to(&ay, true, a.set_kernel_matrix(m)).unwrap();
    let mut v = MixedSinker::<Vuya>::new(width, height)
      .with_hsv(&mut vh, &mut vs, &mut vv)
      .unwrap();
    vuya_to(&vu, true, v.set_kernel_matrix(m)).unwrap();
  }
  assert_eq!((ah, as_, av), (vh, vs, vv), "AYUV↔VUYA HSV");
}

/// SIMD vs scalar parity at a width spanning every backend's block + tail.
#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn ayuv_simd_vs_scalar_parity_at_1922() {
  let w = 1922usize;
  let h = 2usize;
  let mut buf = std::vec![0u8; w * h * 4];
  pseudo_random_u8(&mut buf, 0x0BAD_F00D);
  let src = AyuvFrame::try_new(&buf, w as u32, h as u32, (w * 4) as u32).unwrap();

  let mut simd = std::vec![0u8; w * h * 4];
  let mut scalar = std::vec![0u8; w * h * 4];
  {
    let mut s = MixedSinker::<Ayuv>::new(w, h).with_rgba(&mut simd).unwrap();
    ayuv_to(&src, false, s.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
  }
  {
    let mut s = MixedSinker::<Ayuv>::new(w, h)
      .with_rgba(&mut scalar)
      .unwrap()
      .with_simd(false);
    ayuv_to(&src, false, s.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
  }
  assert_eq!(simd, scalar, "AYUV SIMD ≠ scalar at width {w}");
}

/// A malformed packed row (length ≠ `width × 4`) surfaces the typed
/// `RowShapeMismatch` carrying the `AyuvPacked` slice tag. Fed directly to
/// `process` with matching frame dimensions so the dimension guard passes
/// and the row-shape guard is the one that fires.
#[test]
#[cfg(all(test, feature = "std"))]
fn ayuv_malformed_row_returns_row_shape_mismatch() {
  let mut rgb = std::vec![0u8; 4 * 3];
  let mut sink = MixedSinker::<Ayuv>::new(4, 1).with_rgb(&mut rgb).unwrap();
  sink.begin_frame(4, 1).unwrap();
  // Width 4 needs 16 packed bytes; hand 12.
  let short = std::vec![0u8; 12];
  let row = AyuvRow::for_tests(&short, 0, KernelMatrix::Bt709, false);
  let err = sink.process(row).unwrap_err();
  assert!(
    matches!(err, MixedSinkerError::RowShapeMismatch(e)
      if e.which() == RowSlice::AyuvPacked && e.expected() == 16 && e.actual() == 12),
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
// dedicated `ayuv_to_rgba_row` kernel and never grows the scratch. Reuses the
// crate's `yuva`-gated RGB-scratch failpoint, so this test is too (under
// `--all-features` both `yuv-444-packed` and `yuva` are on).

#[cfg(feature = "yuva")]
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn ayuv_rgb_scratch_alloc_failure_leaves_outputs_untouched() {
  use crate::resample::ResampleError;

  let n = 16 * 8;
  let (yp, up, vp, ap) = (
    std::vec![128u8; n],
    std::vec![128u8; n],
    std::vec![128u8; n],
    std::vec![200u8; n],
  );
  let buf = pack_ayuv(&vp, &up, &yp, &ap, n);
  let src = AyuvFrame::try_new(&buf, 16, 8, 16 * 4).unwrap();
  let mut luma = std::vec![0xABu8; n];
  let mut rgba = std::vec![0xCDu8; n * 4];
  let (mut hh, mut ss, mut vv) = (
    std::vec![0xCDu8; n],
    std::vec![0xCDu8; n],
    std::vec![0xCDu8; n],
  );
  let mut sink = MixedSinker::<Ayuv>::new(16, 8)
    .with_luma(&mut luma)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap()
    .with_hsv(&mut hh, &mut ss, &mut vv)
    .unwrap();

  super::super::arm_rgb_scratch_alloc_failure();
  let err = ayuv_to(&src, false, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap_err();
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
