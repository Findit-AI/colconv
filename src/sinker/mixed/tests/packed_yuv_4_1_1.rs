//! Tier 5.25 packed YUV 4:1:1 sinker tests — UYYVYY411 (DV legacy).

use super::*;

// ---- Solid-color builder ------------------------------------------------

/// Builds a solid UYYVYY411 packed plane with one (Y, U, V) repeated
/// across `width x height`. Layout per 6-byte / 4-pixel block:
/// `U, Y, Y, V, Y, Y`. Stride equals `width * 3 / 2` (no padding).
pub(super) fn solid_uyyvyy411_frame(width: u32, height: u32, y: u8, u: u8, v: u8) -> Vec<u8> {
  let w = width as usize;
  let h = height as usize;
  assert_eq!(w & 3, 0, "uyyvyy411 width must be multiple of 4");
  let mut buf = std::vec![0u8; w * 3 / 2 * h];
  for row in 0..h {
    let base = row * w * 3 / 2;
    for col in (0..w).step_by(4) {
      let blk = base + (col / 4) * 6;
      buf[blk] = u;
      buf[blk + 1] = y;
      buf[blk + 2] = y;
      buf[blk + 3] = v;
      buf[blk + 4] = y;
      buf[blk + 5] = y;
    }
  }
  buf
}

// ---- Uyyvyy411 MixedSinker ---------------------------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn uyyvyy411_luma_only_extracts_y_bytes() {
  let buf = solid_uyyvyy411_frame(16, 8, 42, 128, 128);
  let src = Uyyvyy411Frame::new(&buf, 16, 8, 24);

  let mut luma = std::vec![0u8; 16 * 8];
  let mut sink = MixedSinker::<Uyyvyy411>::new(16, 8)
    .with_luma(&mut luma)
    .unwrap();
  uyyvyy411_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();

  assert!(luma.iter().all(|&y| y == 42));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn uyyvyy411_luma_u16_zero_extends_y_bytes() {
  let buf = solid_uyyvyy411_frame(16, 8, 200, 128, 128);
  let src = Uyyvyy411Frame::new(&buf, 16, 8, 24);

  let mut luma = std::vec![0u16; 16 * 8];
  let mut sink = MixedSinker::<Uyyvyy411>::new(16, 8)
    .with_luma_u16(&mut luma)
    .unwrap();
  uyyvyy411_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();

  assert!(luma.iter().all(|&y| y == 200));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn uyyvyy411_rgb_only_converts_gray_to_gray() {
  let buf = solid_uyyvyy411_frame(16, 8, 128, 128, 128);
  let src = Uyyvyy411Frame::new(&buf, 16, 8, 24);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Uyyvyy411>::new(16, 8)
    .with_rgb(&mut rgb)
    .unwrap();
  uyyvyy411_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();

  for px in rgb.chunks(3) {
    assert!(px[0].abs_diff(128) <= 1);
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn uyyvyy411_rgba_only_converts_gray_to_gray_with_opaque_alpha() {
  let buf = solid_uyyvyy411_frame(16, 8, 128, 128, 128);
  let src = Uyyvyy411Frame::new(&buf, 16, 8, 24);

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Uyyvyy411>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  uyyvyy411_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(128) <= 1, "R");
    assert_eq!(px[0], px[1], "RGB monochromatic");
    assert_eq!(px[1], px[2], "RGB monochromatic");
    assert_eq!(px[3], 0xFF, "alpha defaults to opaque");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn uyyvyy411_mixed_all_outputs_populated() {
  let buf = solid_uyyvyy411_frame(16, 8, 200, 128, 128);
  let src = Uyyvyy411Frame::new(&buf, 16, 8, 24);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut luma = std::vec![0u8; 16 * 8];
  let mut h = std::vec![0u8; 16 * 8];
  let mut s = std::vec![0u8; 16 * 8];
  let mut v = std::vec![0u8; 16 * 8];
  let mut sink = MixedSinker::<Uyyvyy411>::new(16, 8)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_luma(&mut luma)
    .unwrap()
    .with_hsv(&mut h, &mut s, &mut v)
    .unwrap();
  uyyvyy411_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();

  assert!(luma.iter().all(|&y| y == 200));
  for px in rgb.chunks(3) {
    assert!(px[0].abs_diff(200) <= 1);
  }
  assert!(h.iter().all(|&b| b == 0));
  assert!(s.iter().all(|&b| b == 0));
  assert!(v.iter().all(|&b| b.abs_diff(200) <= 1));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn uyyvyy411_with_rgb_and_with_rgba_produce_byte_identical_rgb_bytes() {
  // Strategy A invariant: when both RGB and RGBA are attached, RGBA
  // bytes must equal the RGB row bytes + 0xFF alpha.
  let w = 32u32;
  let h = 16u32;
  let ws = w as usize;
  let hs = h as usize;
  let buf = solid_uyyvyy411_frame(w, h, 180, 60, 200);
  let src = Uyyvyy411Frame::new(&buf, w, h, w * 3 / 2);

  let mut rgb = std::vec![0u8; ws * hs * 3];
  let mut rgba = std::vec![0u8; ws * hs * 4];
  let mut sink = MixedSinker::<Uyyvyy411>::new(ws, hs)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap();
  uyyvyy411_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();

  for i in 0..(ws * hs) {
    assert_eq!(rgba[i * 4], rgb[i * 3], "R differs at pixel {i}");
    assert_eq!(rgba[i * 4 + 1], rgb[i * 3 + 1], "G differs at pixel {i}");
    assert_eq!(rgba[i * 4 + 2], rgb[i * 3 + 2], "B differs at pixel {i}");
    assert_eq!(rgba[i * 4 + 3], 0xFF, "A not opaque at pixel {i}");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn uyyvyy411_with_simd_false_matches_with_simd_true() {
  for &w in &[16usize, 32, 64, 128, 1920] {
    let h = 4usize;
    let row_bytes = w * 3 / 2;
    let mut packed = std::vec![0u8; row_bytes * h];
    pseudo_random_u8(&mut packed, 0xC001_C0DE);
    let src = Uyyvyy411Frame::new(&packed, w as u32, h as u32, row_bytes as u32);

    let mut rgb_simd = std::vec![0u8; w * h * 3];
    let mut rgb_scalar = std::vec![0u8; w * h * 3];
    let mut sink_simd = MixedSinker::<Uyyvyy411>::new(w, h)
      .with_rgb(&mut rgb_simd)
      .unwrap();
    let mut sink_scalar = MixedSinker::<Uyyvyy411>::new(w, h)
      .with_rgb(&mut rgb_scalar)
      .unwrap()
      .with_simd(false);
    uyyvyy411_to(
      &src,
      false,
      sink_simd.set_kernel_matrix(KernelMatrix::Bt709),
    )
    .unwrap();
    uyyvyy411_to(
      &src,
      false,
      sink_scalar.set_kernel_matrix(KernelMatrix::Bt709),
    )
    .unwrap();

    assert_eq!(rgb_simd, rgb_scalar, "Uyyvyy411 SIMD≠scalar at width {w}");
  }
}

#[test]
fn uyyvyy411_width_mismatch_returns_err() {
  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Uyyvyy411>::new(16, 8)
    .with_rgb(&mut rgb)
    .unwrap();
  let buf = solid_uyyvyy411_frame(20, 8, 0, 0, 0);
  let src = Uyyvyy411Frame::new(&buf, 20, 8, 30);
  let err = uyyvyy411_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap_err();
  assert!(matches!(err, MixedSinkerError::DimensionMismatch(_)));
}

#[test]
fn uyyvyy411_begin_frame_rejects_width_not_multiple_of_4() {
  // Sinker configured with a width that's not a multiple of 4 — the
  // begin_frame guard surfaces it before any row primitive runs.
  // Call begin_frame directly with matching dimensions so the
  // dimension-match check passes and the width-alignment check fires.
  // (Going through `uyyvyy411_to(...)` with a 16x8 frame against an
  // 18-wide sink would short-circuit on DimensionMismatch and never
  // exercise the alignment guard.)
  let mut rgb = std::vec![0u8; 18 * 8 * 3];
  let mut sink = MixedSinker::<Uyyvyy411>::new(18, 8)
    .with_rgb(&mut rgb)
    .unwrap();
  let err = sink.begin_frame(18, 8).unwrap_err();
  assert_eq!(
    err,
    MixedSinkerError::WidthAlignment(WidthAlignment::multiple_of_four(18,)),
    "expected WidthAlignment {{ width: 18, required: MultipleOfFour }}, got {err:?}"
  );
}

// Atomicity (#308): the packed 4:1:1 identity-path `process` must run the
// up-front RGB-scratch preflight BEFORE any output row (luma / luma_u16) is
// written, so an allocator refusal returns a recoverable `AllocationFailed`
// leaving the output frame untouched rather than partially mutated. Mirrors the
// packed 4:2:2 / planar / semi-planar siblings. Reuses the crate's RGB-scratch
// failpoint (`yuva`-gated, so this test is too; under `--all-features` both
// `yuv-packed` and `yuva` are on). The `luma + RGBA + HSV, no RGB` combo is the
// one identity-path shape that reaches `rgb_row_buf_or_scratch`'s scratch arm
// (`want_hsv && want_rgba && !want_rgb`).
#[cfg(feature = "yuva")]
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn uyyvyy411_rgb_scratch_alloc_failure_leaves_outputs_untouched() {
  use crate::resample::ResampleError;

  let buf = solid_uyyvyy411_frame(16, 8, 42, 128, 128);
  let src = Uyyvyy411Frame::new(&buf, 16, 8, 24);
  let mut luma = std::vec![0xABu8; 16 * 8];
  let mut rgba = std::vec![0xCDu8; 16 * 8 * 4];
  let (mut hh, mut ss, mut vv) = (
    std::vec![0xCDu8; 16 * 8],
    std::vec![0xCDu8; 16 * 8],
    std::vec![0xCDu8; 16 * 8],
  );
  let mut sink = MixedSinker::<Uyyvyy411>::new(16, 8)
    .with_luma(&mut luma)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap()
    .with_hsv(&mut hh, &mut ss, &mut vv)
    .unwrap();

  super::super::arm_rgb_scratch_alloc_failure();
  let err = uyyvyy411_to(&src, false, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap_err();
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
