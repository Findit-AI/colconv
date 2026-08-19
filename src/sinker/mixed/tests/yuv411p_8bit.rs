//! `MixedSinker<Yuv411p>` integration tests — covers the full output
//! set (rgb / rgba / luma / luma_u16 / hsv) plus Strategy A
//! (RGB + RGBA both attached → run RGB kernel once, fan out via
//! `expand_rgb_to_rgba_row`).
//!
//! 4:1:1 is DV-NTSC legacy with quarter-width chroma. FFmpeg's
//! `AV_PIX_FMT_YUV411P` defines chroma row width as
//! `width.div_ceil(4)`, so non-4-aligned widths are accepted; the
//! scalar tail handles the partial 1..3-pixel final chroma group.

use super::*;

/// Build a solid 4:1:1 frame with the given Y / U / V byte values.
/// Uses contiguous strides (`y_stride = w`, `u_stride = v_stride =
/// w.div_ceil(4)` per FFmpeg's `AV_PIX_FMT_YUV411P` ceil-shift rule).
fn solid_yuv411p_frame(
  width: u32,
  height: u32,
  y: u8,
  u: u8,
  v: u8,
) -> (std::vec::Vec<u8>, std::vec::Vec<u8>, std::vec::Vec<u8>) {
  let w = width as usize;
  let h = height as usize;
  let cw = w.div_ceil(4);
  (
    std::vec![y; w * h],
    std::vec![u; cw * h],
    std::vec![v; cw * h],
  )
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv411p_luma_only_copies_y_plane() {
  let (yp, up, vp) = solid_yuv411p_frame(16, 8, 42, 128, 128);
  let src = Yuv411pFrame::new(&yp, &up, &vp, 16, 8, 16, 4, 4);

  let mut luma = std::vec![0u8; 16 * 8];
  let mut sink = MixedSinker::<Yuv411p>::new(16, 8)
    .with_luma(&mut luma)
    .unwrap();
  yuv411p_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();

  assert!(luma.iter().all(|&y| y == 42), "luma should be solid 42");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv411p_luma_u16_zero_extends_y_plane() {
  let (yp, up, vp) = solid_yuv411p_frame(16, 8, 200, 128, 128);
  let src = Yuv411pFrame::new(&yp, &up, &vp, 16, 8, 16, 4, 4);

  let mut luma_u16 = std::vec![0u16; 16 * 8];
  let mut sink = MixedSinker::<Yuv411p>::new(16, 8)
    .with_luma_u16(&mut luma_u16)
    .unwrap();
  yuv411p_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();

  // 8-bit Y zero-extends into u16.
  assert!(luma_u16.iter().all(|&y| y == 200));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv411p_rgb_only_converts_gray_to_gray() {
  // Neutral chroma → gray RGB; solid Y=128 → ~128 in every RGB byte.
  let (yp, up, vp) = solid_yuv411p_frame(16, 8, 128, 128, 128);
  let src = Yuv411pFrame::new(&yp, &up, &vp, 16, 8, 16, 4, 4);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yuv411p>::new(16, 8)
    .with_rgb(&mut rgb)
    .unwrap();
  yuv411p_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();

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
fn yuv411p_rgba_only_with_opaque_alpha() {
  let (yp, up, vp) = solid_yuv411p_frame(16, 8, 200, 128, 128);
  let src = Yuv411pFrame::new(&yp, &up, &vp, 16, 8, 16, 4, 4);

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuv411p>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  yuv411p_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(200) <= 1);
    assert_eq!(px[3], 0xFF, "alpha must be opaque");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv411p_strategy_a_rgb_and_rgba_match_byte_for_byte() {
  // Strategy A: when both RGB and RGBA are attached, the RGB kernel
  // runs once and `expand_rgb_to_rgba_row` fans out into RGBA.
  // The first three bytes per pixel must match the dedicated
  // RGB-only output, with alpha = 0xFF.
  let w: u32 = 16;
  let h: u32 = 4;
  let (yp, up, vp) = solid_yuv411p_frame(w, h, 180, 60, 200);
  let src = Yuv411pFrame::new(&yp, &up, &vp, w, h, w, w / 4, w / 4);

  let ws = w as usize;
  let hs = h as usize;

  let mut rgb_only = std::vec![0u8; ws * hs * 3];
  let mut sink_rgb = MixedSinker::<Yuv411p>::new(ws, hs)
    .with_rgb(&mut rgb_only)
    .unwrap();
  yuv411p_to(&src, true, sink_rgb.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();

  let mut rgb_combo = std::vec![0u8; ws * hs * 3];
  let mut rgba_combo = std::vec![0u8; ws * hs * 4];
  let mut sink_combo = MixedSinker::<Yuv411p>::new(ws, hs)
    .with_rgb(&mut rgb_combo)
    .unwrap()
    .with_rgba(&mut rgba_combo)
    .unwrap();
  yuv411p_to(
    &src,
    true,
    sink_combo.set_kernel_matrix(KernelMatrix::Bt601),
  )
  .unwrap();

  assert_eq!(rgb_only, rgb_combo, "RGB-only and combo RGB must match");
  for px in 0..(ws * hs) {
    assert_eq!(rgba_combo[px * 4], rgb_only[px * 3]);
    assert_eq!(rgba_combo[px * 4 + 1], rgb_only[px * 3 + 1]);
    assert_eq!(rgba_combo[px * 4 + 2], rgb_only[px * 3 + 2]);
    assert_eq!(rgba_combo[px * 4 + 3], 0xFF);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv411p_hsv_only_allocates_scratch_and_produces_gray_hsv() {
  // Neutral gray → H=0, S=0, V=~128. No RGB buffer provided.
  let (yp, up, vp) = solid_yuv411p_frame(16, 8, 128, 128, 128);
  let src = Yuv411pFrame::new(&yp, &up, &vp, 16, 8, 16, 4, 4);

  let mut h = std::vec![0xFFu8; 16 * 8];
  let mut s = std::vec![0xFFu8; 16 * 8];
  let mut v = std::vec![0xFFu8; 16 * 8];
  let mut sink = MixedSinker::<Yuv411p>::new(16, 8)
    .with_hsv(&mut h, &mut s, &mut v)
    .unwrap();
  yuv411p_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();

  assert!(h.iter().all(|&b| b == 0));
  assert!(s.iter().all(|&b| b == 0));
  assert!(v.iter().all(|&b| b.abs_diff(128) <= 1));
}

#[test]
fn yuv411p_accepts_non_4_aligned_widths_via_div_ceil() {
  // FFmpeg `AV_PIX_FMT_YUV411P` defines chroma width as
  // `width.div_ceil(4)`. Width=15 → chroma_width=4 (3 full 4-pixel
  // groups + 1 partial 3-pixel group sharing the last chroma sample).
  // Both frame construction and `begin_frame` must accept it.
  let yp = std::vec![0u8; 15 * 4];
  let up = std::vec![128u8; 4 * 4];
  let vp = std::vec![128u8; 4 * 4];
  let frame = Yuv411pFrame::try_new(&yp, &up, &vp, 15, 4, 15, 4, 4)
    .expect("width=15 valid: chroma row carries 4 samples per FFmpeg semantics");
  assert_eq!(frame.width(), 15);

  let mut sink: MixedSinker<'_, Yuv411p> = MixedSinker::new(15usize, 4usize);
  <MixedSinker<'_, Yuv411p> as crate::PixelSink>::begin_frame(&mut sink, 15, 4)
    .expect("begin_frame should accept non-4-aligned width");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv411p_simd_matches_scalar_with_random_yuv() {
  // End-to-end SIMD vs scalar parity for the sinker pipeline.
  let w: u32 = 64;
  let h: u32 = 8;
  let ws = w as usize;
  let hs = h as usize;

  let mut yp = std::vec![0u8; ws * hs];
  let mut up = std::vec![0u8; (ws / 4) * hs];
  let mut vp = std::vec![0u8; (ws / 4) * hs];
  pseudo_random_u8(&mut yp, 0x1111);
  pseudo_random_u8(&mut up, 0x2222);
  pseudo_random_u8(&mut vp, 0x3333);
  let src = Yuv411pFrame::new(&yp, &up, &vp, w, h, w, w / 4, w / 4);

  for &matrix in &[
    KernelMatrix::Bt601,
    KernelMatrix::Bt709,
    KernelMatrix::YCgCo,
  ] {
    for full_range in [true, false] {
      let mut rgb_simd = std::vec![0u8; ws * hs * 3];
      let mut rgb_scalar = std::vec![0u8; ws * hs * 3];

      let mut s_simd = MixedSinker::<Yuv411p>::new(ws, hs)
        .with_rgb(&mut rgb_simd)
        .unwrap();
      yuv411p_to(&src, full_range, s_simd.set_kernel_matrix(matrix)).unwrap();

      let mut s_scalar = MixedSinker::<Yuv411p>::new(ws, hs)
        .with_rgb(&mut rgb_scalar)
        .unwrap();
      s_scalar.set_simd(false);
      yuv411p_to(&src, full_range, s_scalar.set_kernel_matrix(matrix)).unwrap();

      assert_eq!(
        rgb_simd, rgb_scalar,
        "SIMD vs scalar diverges at matrix={matrix:?} full_range={full_range}"
      );
    }
  }
}

#[test]
fn yuv411p_luma_u16_buffer_too_short_returns_err() {
  // Regression: validation must measure `buf.len()` in u16 elements,
  // not bytes. A buffer one element short of `width x height` u16s
  // must be rejected — this would have slipped through if the check
  // had compared `buf.len()` (u16 count) against a byte count.
  let mut buf = std::vec![0u16; 16 * 8 - 1];
  let err = MixedSinker::<Yuv411p>::new(16, 8)
    .with_luma_u16(&mut buf)
    .err()
    .unwrap();
  assert_eq!(
    err,
    MixedSinkerError::InsufficientLumaU16Buffer(InsufficientBuffer::new(128, 127))
  );
}

#[test]
fn yuv411p_luma_u16_buffer_exactly_sized_accepts() {
  // Companion to the negative test above: an exactly-sized buffer
  // (`width x height` u16 elements) must be accepted. Pins down the
  // boundary condition so the negative test can't pass for the wrong
  // reason (e.g. an off-by-one in the opposite direction).
  let mut buf = std::vec![0u16; 16 * 8];
  let result = MixedSinker::<Yuv411p>::new(16, 8).with_luma_u16(&mut buf);
  assert!(result.is_ok());
}

// Atomicity (#308): the up-front RGB-scratch preflight must return
// `AllocationFailed` BEFORE any output row is written, so an allocator refusal
// leaves the output frame untouched. Mirrors the Yuv420p sibling. Reuses the
// crate's RGB-scratch failpoint (`yuva`-gated, so this test is too; under
// `--all-features` both `yuv-planar` and `yuva` are on).
#[cfg(feature = "yuva")]
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv411p_rgb_scratch_alloc_failure_leaves_outputs_untouched() {
  use crate::resample::ResampleError;

  // luma + RGBA + HSV with NO rgb output: `want_hsv && want_rgba && !want_rgb`
  // needs an RGB row but has no caller RGB buffer, so the decode grows the RGB
  // row scratch (`rgb_row_buf_or_scratch`'s scratch arm). With that allocation
  // armed to fail, the preflight returns AllocationFailed BEFORE any output
  // row — luma included — is written.
  let (yp, up, vp) = solid_yuv411p_frame(16, 8, 42, 128, 128);
  let src = Yuv411pFrame::new(&yp, &up, &vp, 16, 8, 16, 4, 4);
  let mut luma = std::vec![0xABu8; 16 * 8];
  let mut rgba = std::vec![0xCDu8; 16 * 8 * 4];
  let (mut hh, mut ss, mut vv) = (
    std::vec![0xCDu8; 16 * 8],
    std::vec![0xCDu8; 16 * 8],
    std::vec![0xCDu8; 16 * 8],
  );
  let mut sink = MixedSinker::<Yuv411p>::new(16, 8)
    .with_luma(&mut luma)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap()
    .with_hsv(&mut hh, &mut ss, &mut vv)
    .unwrap();

  super::super::arm_rgb_scratch_alloc_failure();
  let err = yuv411p_to(&src, false, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap_err();
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
