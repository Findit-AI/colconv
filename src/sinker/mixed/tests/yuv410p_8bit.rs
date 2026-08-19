use super::*;

fn solid_yuv410p_frame(
  width: u32,
  height: u32,
  y: u8,
  u: u8,
  v: u8,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let w = width as usize;
  let h = height as usize;
  let cw = w / 4;
  let ch = h / 4;
  (
    std::vec![y; w * h],
    std::vec![u; cw * ch],
    std::vec![v; cw * ch],
  )
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn luma_only_copies_y_plane() {
  let (yp, up, vp) = solid_yuv410p_frame(16, 8, 42, 128, 128);
  let src = Yuv410pFrame::new(&yp, &up, &vp, 16, 8, 16, 4, 4);

  let mut luma = std::vec![0u8; 16 * 8];
  let mut sink = MixedSinker::<Yuv410p>::new(16, 8)
    .with_luma(&mut luma)
    .unwrap();
  yuv410p_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();

  assert!(luma.iter().all(|&y| y == 42), "luma should be solid 42");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn luma_u16_zero_extends_y_plane() {
  let (yp, up, vp) = solid_yuv410p_frame(16, 8, 200, 128, 128);
  let src = Yuv410pFrame::new(&yp, &up, &vp, 16, 8, 16, 4, 4);

  let mut luma = std::vec![0u16; 16 * 8];
  let mut sink = MixedSinker::<Yuv410p>::new(16, 8)
    .with_luma_u16(&mut luma)
    .unwrap();
  yuv410p_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();

  assert!(luma.iter().all(|&y| y == 200u16));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgb_only_converts_gray_to_gray() {
  // Neutral chroma → gray RGB; solid Y=128 → ~128 in every RGB byte.
  let (yp, up, vp) = solid_yuv410p_frame(16, 8, 128, 128, 128);
  let src = Yuv410pFrame::new(&yp, &up, &vp, 16, 8, 16, 4, 4);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yuv410p>::new(16, 8)
    .with_rgb(&mut rgb)
    .unwrap();
  yuv410p_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();

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
fn rgba_only_converts_gray_to_gray_with_opaque_alpha() {
  let (yp, up, vp) = solid_yuv410p_frame(16, 8, 128, 128, 128);
  let src = Yuv410pFrame::new(&yp, &up, &vp, 16, 8, 16, 4, 4);

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuv410p>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  yuv410p_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(128) <= 1, "R");
    assert_eq!(px[0], px[1], "RGB monochromatic");
    assert_eq!(px[1], px[2], "RGB monochromatic");
    assert_eq!(px[3], 0xFF, "alpha must default to opaque");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn hsv_only_produces_gray_hsv() {
  let (yp, up, vp) = solid_yuv410p_frame(16, 8, 128, 128, 128);
  let src = Yuv410pFrame::new(&yp, &up, &vp, 16, 8, 16, 4, 4);

  let mut h = std::vec![0xFFu8; 16 * 8];
  let mut s = std::vec![0xFFu8; 16 * 8];
  let mut v = std::vec![0xFFu8; 16 * 8];
  let mut sink = MixedSinker::<Yuv410p>::new(16, 8)
    .with_hsv(&mut h, &mut s, &mut v)
    .unwrap();
  yuv410p_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();

  assert!(h.iter().all(|&b| b == 0));
  assert!(s.iter().all(|&b| b == 0));
  assert!(v.iter().all(|&b| b.abs_diff(128) <= 1));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn mixed_all_outputs_populated() {
  let (yp, up, vp) = solid_yuv410p_frame(16, 8, 200, 128, 128);
  let src = Yuv410pFrame::new(&yp, &up, &vp, 16, 8, 16, 4, 4);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut luma = std::vec![0u8; 16 * 8];
  let mut luma_u16 = std::vec![0u16; 16 * 8];
  let mut h = std::vec![0u8; 16 * 8];
  let mut s = std::vec![0u8; 16 * 8];
  let mut v = std::vec![0u8; 16 * 8];
  let mut sink = MixedSinker::<Yuv410p>::new(16, 8)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap()
    .with_luma(&mut luma)
    .unwrap()
    .with_luma_u16(&mut luma_u16)
    .unwrap()
    .with_hsv(&mut h, &mut s, &mut v)
    .unwrap();
  yuv410p_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();

  // Luma = Y plane verbatim.
  assert!(luma.iter().all(|&y| y == 200));
  assert!(luma_u16.iter().all(|&y| y == 200u16));
  // RGB monochrome at ~200.
  for px in rgb.chunks(3) {
    assert!(px[0].abs_diff(200) <= 1);
  }
  // Strategy A fan-out: RGBA's RGB bytes should match RGB output, alpha = 0xFF.
  for (rgb_px, rgba_px) in rgb.chunks(3).zip(rgba.chunks(4)) {
    assert_eq!(rgb_px[0], rgba_px[0]);
    assert_eq!(rgb_px[1], rgba_px[1]);
    assert_eq!(rgb_px[2], rgba_px[2]);
    assert_eq!(rgba_px[3], 0xFF);
  }
  // HSV of gray.
  assert!(h.iter().all(|&b| b == 0));
  assert!(s.iter().all(|&b| b == 0));
  assert!(v.iter().all(|&b| b.abs_diff(200) <= 1));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn vertical_chroma_subsampling_4x_each_chroma_row_covers_4_y_rows() {
  // Build a 16x8 frame with two distinct chroma blocks vertically:
  // chroma row 0 covers Y rows 0..3, chroma row 1 covers Y rows 4..7.
  // Set chroma block 0 to (U=80, V=180) and block 1 to (U=160, V=80) —
  // distinct enough to produce visibly different RGB across the
  // horizontal divide.
  let w = 16usize;
  let h = 8usize;
  let cw = w / 4;
  // Two chroma rows: row 0 (covers Y rows 0..3) and row 1 (covers Y rows 4..7).
  let mut up = std::vec![0u8; cw * 2];
  let mut vp = std::vec![0u8; cw * 2];
  for x in 0..cw {
    up[x] = 80;
    up[cw + x] = 160;
    vp[x] = 180;
    vp[cw + x] = 80;
  }
  let yp = std::vec![128u8; w * h];

  let src = Yuv410pFrame::new(
    &yp, &up, &vp, w as u32, h as u32, w as u32, cw as u32, cw as u32,
  );

  let mut rgb = std::vec![0u8; w * h * 3];
  let mut sink = MixedSinker::<Yuv410p>::new(w, h)
    .with_rgb(&mut rgb)
    .unwrap();
  yuv410p_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();

  // Y rows 0..=3 should all share the same RGB color (chroma block 0).
  let row_bytes = w * 3;
  let r0 = &rgb[..row_bytes];
  for y_idx in 1..4 {
    let row = &rgb[y_idx * row_bytes..(y_idx + 1) * row_bytes];
    assert_eq!(
      r0, row,
      "Y row 0 and row {y_idx} should share chroma — same RGB"
    );
  }
  // Y rows 4..=7 should all share the same (different) RGB color.
  let r4 = &rgb[4 * row_bytes..5 * row_bytes];
  for y_idx in 5..8 {
    let row = &rgb[y_idx * row_bytes..(y_idx + 1) * row_bytes];
    assert_eq!(
      r4, row,
      "Y row 4 and row {y_idx} should share chroma — same RGB"
    );
  }
  // The two chroma blocks produce DIFFERENT RGB.
  assert_ne!(
    r0, r4,
    "distinct chroma blocks should produce distinct RGB output"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn horizontal_chroma_subsampling_4x_each_chroma_sample_covers_4_y_columns() {
  // Build a 16x4 frame with two distinct chroma samples per row:
  // sample 0 covers Y columns 0..3, sample 1 covers Y columns 4..7,
  // sample 2 covers Y columns 8..11, sample 3 covers Y columns 12..15.
  let w = 16usize;
  let h = 4usize;
  let cw = w / 4;
  let mut up = std::vec![0u8; cw];
  let mut vp = std::vec![0u8; cw];
  // Different chroma in each of the 4 horizontal blocks.
  for x in 0..cw {
    up[x] = (60 + x * 30) as u8; // 60, 90, 120, 150
    vp[x] = (200 - x * 30) as u8; // 200, 170, 140, 110
  }
  let yp = std::vec![128u8; w * h];

  let src = Yuv410pFrame::new(
    &yp, &up, &vp, w as u32, h as u32, w as u32, cw as u32, cw as u32,
  );

  let mut rgb = std::vec![0u8; w * h * 3];
  let mut sink = MixedSinker::<Yuv410p>::new(w, h)
    .with_rgb(&mut rgb)
    .unwrap();
  yuv410p_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();

  // Within row 0, columns 0..=3 should all share RGB; columns 4..=7
  // should share a different RGB; etc.
  let row0 = &rgb[..w * 3];
  for block in 0..cw {
    let block_start = block * 4;
    let p0 = &row0[block_start * 3..(block_start + 1) * 3];
    for x in 1..4 {
      let p = &row0[(block_start + x) * 3..(block_start + x + 1) * 3];
      assert_eq!(
        p0, p,
        "block {block}, col {x} should share chroma with col 0"
      );
    }
  }
  // Adjacent blocks must produce different RGB (chroma differs).
  for block in 0..cw - 1 {
    let p_a = &row0[block * 4 * 3..(block * 4 + 1) * 3];
    let p_b = &row0[(block + 1) * 4 * 3..((block + 1) * 4 + 1) * 3];
    assert_ne!(
      p_a,
      p_b,
      "block {block} and {} have distinct chroma → distinct RGB",
      block + 1
    );
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgb_simd_matches_scalar_pseudo_random() {
  // SIMD ↔ scalar parity at the sinker level. Width 64 exercises both
  // the NEON 16-lane main loop and the trailing scalar-fallback path
  // (no tail at 64 since 64 is a multiple of 16, but tests that the
  // routing produces the same result either way).
  let w = 64u32;
  let h = 8u32;
  let cw = (w / 4) as usize;
  let ch = (h / 4) as usize;
  let mut yp = std::vec![0u8; (w * h) as usize];
  let mut up = std::vec![0u8; cw * ch];
  let mut vp = std::vec![0u8; cw * ch];
  pseudo_random_u8(&mut yp, 0xC0FFEE);
  pseudo_random_u8(&mut up, 0xDECAFBAD);
  pseudo_random_u8(&mut vp, 0xBADF00D);
  let src = Yuv410pFrame::new(&yp, &up, &vp, w, h, w, cw as u32, cw as u32);

  let mut rgb_simd = std::vec![0u8; (w * h) as usize * 3];
  let mut rgb_scalar = std::vec![0u8; (w * h) as usize * 3];

  let mut sink_simd = MixedSinker::<Yuv410p>::new(w as usize, h as usize)
    .with_rgb(&mut rgb_simd)
    .unwrap();
  yuv410p_to(&src, true, sink_simd.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();

  let mut sink_scalar = MixedSinker::<Yuv410p>::new(w as usize, h as usize)
    .with_rgb(&mut rgb_scalar)
    .unwrap();
  sink_scalar.set_simd(false);
  yuv410p_to(
    &src,
    true,
    sink_scalar.set_kernel_matrix(KernelMatrix::Bt709),
  )
  .unwrap();

  assert_eq!(
    rgb_simd, rgb_scalar,
    "SIMD and scalar must produce byte-identical RGB"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn non_4_aligned_height_reuses_trailing_chroma_row() {
  // Height = 6 → chroma height = ceil(6/4) = 2. Y rows 0..=3 read
  // chroma row 0; Y rows 4..=5 read chroma row 1. Distinct chroma
  // values per chroma row let us assert the walker mapped the
  // trailing partial group correctly.
  let w: u32 = 16;
  let h: u32 = 6;
  let cw = (w / 4) as usize; // 4
  let ch = h.div_ceil(4) as usize; // 2

  // Distinct chroma per row: row 0 = (40, 200), row 1 = (200, 40).
  let mut up = std::vec![0u8; cw * ch];
  let mut vp = std::vec![0u8; cw * ch];
  for x in 0..cw {
    up[x] = 40;
    vp[x] = 200;
    up[cw + x] = 200;
    vp[cw + x] = 40;
  }
  // Mid-gray Y so the chroma swing is what dominates color output.
  let yp = std::vec![128u8; (w * h) as usize];

  let src = Yuv410pFrame::try_new(&yp, &up, &vp, w, h, w, cw as u32, cw as u32).expect("valid");

  let mut rgb_simd = std::vec![0u8; (w * h) as usize * 3];
  let mut rgb_scalar = std::vec![0u8; (w * h) as usize * 3];

  let mut sink_simd = MixedSinker::<Yuv410p>::new(w as usize, h as usize)
    .with_rgb(&mut rgb_simd)
    .unwrap();
  yuv410p_to(&src, true, sink_simd.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();

  let mut sink_scalar = MixedSinker::<Yuv410p>::new(w as usize, h as usize)
    .with_rgb(&mut rgb_scalar)
    .unwrap();
  sink_scalar.set_simd(false);
  yuv410p_to(
    &src,
    true,
    sink_scalar.set_kernel_matrix(KernelMatrix::Bt709),
  )
  .unwrap();

  assert_eq!(
    rgb_simd, rgb_scalar,
    "SIMD and scalar must agree at non-4-aligned heights"
  );

  // Sanity: the top half (chroma row 0, U=40 V=200) and bottom strip
  // (chroma row 1, U=200 V=40) must produce visibly different colors.
  let row_stride = (w * 3) as usize;
  let top_red = rgb_scalar[0];
  let bot_red = rgb_scalar[5 * row_stride];
  assert_ne!(
    top_red, bot_red,
    "row 0 and row 5 should derive from different chroma rows"
  );
}

#[test]
fn non_multiple_of_4_width_surfaces_width_alignment_multiple_of_four_error() {
  // Yuv410p subsamples chroma 4:1:0, so widths not divisible by 4
  // can't form a complete chroma group. The sinker must return
  // `WidthAlignment` with `WidthAlignmentRequirement::MultipleOfFour`
  // — matching the format-specific `Yuv410pFrameError::WidthNotMultipleOf4`
  // returned by `Yuv410pFrame::try_new` and the Uyyvyy411 sinker
  // convention. The pre-refactor variant `OddWidth` (whose message was
  // scoped to 4:2:0) is no longer applicable here.
  let mut rgb = std::vec![0u8; 18 * 8 * 3];
  let mut sink = MixedSinker::<Yuv410p>::new(18, 8)
    .with_rgb(&mut rgb)
    .unwrap();
  let err = sink.begin_frame(18, 8).unwrap_err();
  assert_eq!(
    err,
    MixedSinkerError::WidthAlignment(WidthAlignment::multiple_of_four(18,)),
    "expected WidthAlignment {{ width: 18, required: MultipleOfFour }}, got {err:?}"
  );
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
fn yuv410p_rgb_scratch_alloc_failure_leaves_outputs_untouched() {
  use crate::resample::ResampleError;

  // luma + RGBA + HSV with NO rgb output: `want_hsv && want_rgba && !want_rgb`
  // needs an RGB row but has no caller RGB buffer, so the decode grows the RGB
  // row scratch (`rgb_row_buf_or_scratch`'s scratch arm). With that allocation
  // armed to fail, the preflight returns AllocationFailed BEFORE any output
  // row — luma included — is written.
  let (yp, up, vp) = solid_yuv410p_frame(16, 8, 42, 128, 128);
  let src = Yuv410pFrame::new(&yp, &up, &vp, 16, 8, 16, 4, 4);
  let mut luma = std::vec![0xABu8; 16 * 8];
  let mut rgba = std::vec![0xCDu8; 16 * 8 * 4];
  let (mut hh, mut ss, mut vv) = (
    std::vec![0xCDu8; 16 * 8],
    std::vec![0xCDu8; 16 * 8],
    std::vec![0xCDu8; 16 * 8],
  );
  let mut sink = MixedSinker::<Yuv410p>::new(16, 8)
    .with_luma(&mut luma)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap()
    .with_hsv(&mut hh, &mut ss, &mut vv)
    .unwrap();

  super::super::arm_rgb_scratch_alloc_failure();
  let err = yuv410p_to(&src, false, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap_err();
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
