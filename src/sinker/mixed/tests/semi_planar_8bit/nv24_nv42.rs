// Cross-family fixture oracles for the `strategy_a_*_for_all_wired_families`
// tests below — each lives in its own per-family-gated module, so this
// import (and those two tests) only compiles when every wired family is on.
#[cfg(all(
  feature = "yuv-planar",
  feature = "yuv-packed",
  feature = "yuv-444-packed",
  feature = "v210",
  feature = "y2xx",
))]
use super::super::{
  packed_yuv_8bit::{solid_uyvy422_frame, solid_yuyv422_frame, solid_yvyu422_frame},
  planar_other_8bit_9bit::{solid_yuv422p_frame, solid_yuv440p_frame, solid_yuv444p_frame},
  v30x::solid_v30x_frame,
  v210::solid_v210_frame,
  v410::solid_v410_frame,
  vuyx::solid_vuyx_frame,
  xv36::solid_xv36_frame,
  y210::solid_y210_frame,
  y212::solid_y212_frame,
  y216::solid_y216_frame,
};
// `solid_nv{12,16,21}_frame` feed the cross-family `strategy_a_*` tests
// only, so they ride the same all-wired-families gate as the import above.
use super::*;
#[cfg(all(
  feature = "yuv-planar",
  feature = "yuv-packed",
  feature = "yuv-444-packed",
  feature = "v210",
  feature = "y2xx",
))]
use super::{nv12::solid_nv12_frame, nv16::solid_nv16_frame, nv21::solid_nv21_frame};

// ---- NV24 MixedSinker ---------------------------------------------------
//
// 4:4:4 semi-planar: UV row is `2 * width` bytes (one UV pair per
// Y pixel). Tests mirror the NV12 set plus one cross-format parity
// check against a synthetic NV42 frame (byte-swap the interleaved
// chroma → identical RGB output).

fn solid_nv24_frame(width: u32, height: u32, y: u8, u: u8, v: u8) -> (Vec<u8>, Vec<u8>) {
  let w = width as usize;
  let h = height as usize;
  // UV row payload = `2 * width` bytes = `width` interleaved U/V pairs.
  let mut uv = std::vec![0u8; 2 * w * h];
  for row in 0..h {
    for i in 0..w {
      uv[row * 2 * w + i * 2] = u;
      uv[row * 2 * w + i * 2 + 1] = v;
    }
  }
  (std::vec![y; w * h], uv)
}

#[test]
fn nv24_luma_only_copies_y_plane() {
  let (yp, uvp) = solid_nv24_frame(16, 8, 42, 128, 128);
  let src = Nv24Frame::new(&yp, &uvp, 16, 8, 16, 32);

  let mut luma = std::vec![0u8; 16 * 8];
  let mut sink = MixedSinker::<Nv24>::new(16, 8)
    .with_luma(&mut luma)
    .unwrap();
  nv24_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();

  assert!(luma.iter().all(|&y| y == 42));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv24_rgb_only_converts_gray_to_gray() {
  let (yp, uvp) = solid_nv24_frame(16, 8, 128, 128, 128);
  let src = Nv24Frame::new(&yp, &uvp, 16, 8, 16, 32);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Nv24>::new(16, 8).with_rgb(&mut rgb).unwrap();
  nv24_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();

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
fn nv24_mixed_all_three_outputs_populated() {
  let (yp, uvp) = solid_nv24_frame(16, 8, 200, 128, 128);
  let src = Nv24Frame::new(&yp, &uvp, 16, 8, 16, 32);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut luma = std::vec![0u8; 16 * 8];
  let mut h = std::vec![0u8; 16 * 8];
  let mut s = std::vec![0u8; 16 * 8];
  let mut v = std::vec![0u8; 16 * 8];
  let mut sink = MixedSinker::<Nv24>::new(16, 8)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_luma(&mut luma)
    .unwrap()
    .with_hsv(&mut h, &mut s, &mut v)
    .unwrap();
  nv24_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();

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
fn nv24_accepts_odd_width() {
  // 4:4:4 removes the width parity constraint. A 17-wide frame
  // should round-trip cleanly.
  let (yp, uvp) = solid_nv24_frame(17, 8, 200, 128, 128);
  let src = Nv24Frame::new(&yp, &uvp, 17, 8, 17, 34);

  let mut rgb = std::vec![0u8; 17 * 8 * 3];
  let mut sink = MixedSinker::<Nv24>::new(17, 8).with_rgb(&mut rgb).unwrap();
  nv24_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();

  for px in rgb.chunks(3) {
    assert!(px[0].abs_diff(200) <= 1);
  }
}

// ---- NV42 MixedSinker ---------------------------------------------------

fn solid_nv42_frame(width: u32, height: u32, y: u8, u: u8, v: u8) -> (Vec<u8>, Vec<u8>) {
  let w = width as usize;
  let h = height as usize;
  // VU row payload = `2 * width` bytes = `width` interleaved V/U pairs
  // (byte-swapped relative to NV24).
  let mut vu = std::vec![0u8; 2 * w * h];
  for row in 0..h {
    for i in 0..w {
      vu[row * 2 * w + i * 2] = v;
      vu[row * 2 * w + i * 2 + 1] = u;
    }
  }
  (std::vec![y; w * h], vu)
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv42_rgb_only_converts_gray_to_gray() {
  let (yp, vup) = solid_nv42_frame(16, 8, 128, 128, 128);
  let src = Nv42Frame::new(&yp, &vup, 16, 8, 16, 32);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Nv42>::new(16, 8).with_rgb(&mut rgb).unwrap();
  nv42_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();

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
fn nv42_matches_nv24_mixed_sinker_with_swapped_chroma() {
  // Cross-format parity: for the same Y plane and byte-swapped
  // interleaved chroma, NV24 and NV42 must produce identical RGB
  // output. Mirrors the NV21↔NV12 test.
  let w = 33usize; // deliberately odd to exercise the no-parity-constraint path
  let h = 8usize;
  let yp: Vec<u8> = (0..w * h).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
  let uv_nv24: Vec<u8> = (0..2 * w * h)
    .map(|i| ((i * 53 + 23) & 0xFF) as u8)
    .collect();
  // Build NV42 chroma by swapping each (U, V) pair.
  let mut vu_nv42 = std::vec![0u8; 2 * w * h];
  for i in 0..w * h {
    vu_nv42[i * 2] = uv_nv24[i * 2 + 1];
    vu_nv42[i * 2 + 1] = uv_nv24[i * 2];
  }
  let nv24_src = Nv24Frame::new(&yp, &uv_nv24, w as u32, h as u32, w as u32, (2 * w) as u32);
  let nv42_src = Nv42Frame::new(&yp, &vu_nv42, w as u32, h as u32, w as u32, (2 * w) as u32);

  let mut rgb_nv24 = std::vec![0u8; w * h * 3];
  let mut rgb_nv42 = std::vec![0u8; w * h * 3];
  let mut s_nv24 = MixedSinker::<Nv24>::new(w, h)
    .with_rgb(&mut rgb_nv24)
    .unwrap();
  let mut s_nv42 = MixedSinker::<Nv42>::new(w, h)
    .with_rgb(&mut rgb_nv42)
    .unwrap();
  nv24_to(
    &nv24_src,
    false,
    s_nv24.set_kernel_matrix(KernelMatrix::Bt709),
  )
  .unwrap();
  nv42_to(
    &nv42_src,
    false,
    s_nv42.set_kernel_matrix(KernelMatrix::Bt709),
  )
  .unwrap();

  assert_eq!(rgb_nv24, rgb_nv42);
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv24_with_simd_false_matches_with_simd_true() {
  // Widths chosen to force each backend's main loop AND its
  // scalar-tail path:
  // - 16, 17 → NEON/SSE4.1/wasm main (16-Y block), AVX2 + AVX-512 no main.
  // - 32, 33 → AVX2 main (32-Y block), AVX-512 no main.
  // - 64, 65 → AVX-512 main (64-Y block) once + optional 1-px tail.
  // - 127, 128 → AVX-512 main twice, 127 also forces a 63-px tail.
  // - 1920 → wide real-world baseline.
  for &w in &[16usize, 17, 32, 33, 64, 65, 127, 128, 1920] {
    let h = 4usize;
    let yp: Vec<u8> = (0..w * h).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
    let uvp: Vec<u8> = (0..2 * w * h)
      .map(|i| ((i * 53 + 23) & 0xFF) as u8)
      .collect();
    let src = Nv24Frame::new(&yp, &uvp, w as u32, h as u32, w as u32, (2 * w) as u32);

    let mut rgb_simd = std::vec![0u8; w * h * 3];
    let mut rgb_scalar = std::vec![0u8; w * h * 3];
    let mut sink_simd = MixedSinker::<Nv24>::new(w, h)
      .with_rgb(&mut rgb_simd)
      .unwrap();
    let mut sink_scalar = MixedSinker::<Nv24>::new(w, h)
      .with_rgb(&mut rgb_scalar)
      .unwrap()
      .with_simd(false);
    nv24_to(
      &src,
      false,
      sink_simd.set_kernel_matrix(KernelMatrix::Bt709),
    )
    .unwrap();
    nv24_to(
      &src,
      false,
      sink_scalar.set_kernel_matrix(KernelMatrix::Bt709),
    )
    .unwrap();

    assert_eq!(rgb_simd, rgb_scalar, "NV24 SIMD≠scalar at width {w}");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv42_with_simd_false_matches_with_simd_true() {
  // Same width coverage as the NV24 variant — exercises every
  // backend's main loop + scalar tail for the `SWAP_UV = true`
  // monomorphization.
  for &w in &[16usize, 17, 32, 33, 64, 65, 127, 128, 1920] {
    let h = 4usize;
    let yp: Vec<u8> = (0..w * h).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
    let vup: Vec<u8> = (0..2 * w * h)
      .map(|i| ((i * 53 + 23) & 0xFF) as u8)
      .collect();
    let src = Nv42Frame::new(&yp, &vup, w as u32, h as u32, w as u32, (2 * w) as u32);

    let mut rgb_simd = std::vec![0u8; w * h * 3];
    let mut rgb_scalar = std::vec![0u8; w * h * 3];
    let mut sink_simd = MixedSinker::<Nv42>::new(w, h)
      .with_rgb(&mut rgb_simd)
      .unwrap();
    let mut sink_scalar = MixedSinker::<Nv42>::new(w, h)
      .with_rgb(&mut rgb_scalar)
      .unwrap()
      .with_simd(false);
    nv42_to(
      &src,
      false,
      sink_simd.set_kernel_matrix(KernelMatrix::Bt709),
    )
    .unwrap();
    nv42_to(
      &src,
      false,
      sink_scalar.set_kernel_matrix(KernelMatrix::Bt709),
    )
    .unwrap();

    assert_eq!(rgb_simd, rgb_scalar, "NV42 SIMD≠scalar at width {w}");
  }
}

#[test]
fn nv24_width_mismatch_returns_err() {
  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Nv24>::new(16, 8).with_rgb(&mut rgb).unwrap();
  // 8-tall src matches the sink; width 17 vs sink's 16 triggers the
  // mismatch in `begin_frame`.
  let (yp, uvp) = solid_nv24_frame(17, 8, 0, 0, 0);
  let src = Nv24Frame::new(&yp, &uvp, 17, 8, 17, 34);
  let err = nv24_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap_err();
  assert!(matches!(err, MixedSinkerError::DimensionMismatch(_)));
}

// Nv24/Nv42 RGBA tests.

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv24_rgba_only_converts_gray_to_gray_with_opaque_alpha() {
  let (yp, uvp) = solid_nv24_frame(16, 8, 128, 128, 128);
  let src = Nv24Frame::new(&yp, &uvp, 16, 8, 16, 32);

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Nv24>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  nv24_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();

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
fn nv24_with_rgb_and_with_rgba_produce_byte_identical_rgb_bytes() {
  // Strategy A invariant: when both RGB and RGBA are attached, the
  // RGBA bytes must be byte-for-byte identical to the RGB row +
  // 0xFF alpha. This is the cross-format guarantee that holds even
  // after we replaced the dual-kernel path with the
  // expand_rgb_to_rgba_row fan-out.
  let w = 32u32;
  let h = 16u32;
  let ws = w as usize;
  let hs = h as usize;
  let (yp, uvp) = solid_nv24_frame(w, h, 180, 60, 200);
  let src = Nv24Frame::new(&yp, &uvp, w, h, w, 2 * w);

  let mut rgb = std::vec![0u8; ws * hs * 3];
  let mut rgba = std::vec![0u8; ws * hs * 4];
  let mut sink = MixedSinker::<Nv24>::new(ws, hs)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap();
  nv24_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();

  for i in 0..(ws * hs) {
    assert_eq!(rgba[i * 4], rgb[i * 3], "R differs at pixel {i}");
    assert_eq!(rgba[i * 4 + 1], rgb[i * 3 + 1], "G differs at pixel {i}");
    assert_eq!(rgba[i * 4 + 2], rgb[i * 3 + 2], "B differs at pixel {i}");
    assert_eq!(rgba[i * 4 + 3], 0xFF, "A not opaque at pixel {i}");
  }
}

#[test]
fn nv24_rgba_buffer_too_short_returns_err() {
  let mut rgba_short = std::vec![0u8; 16 * 8 * 4 - 1];
  let result = MixedSinker::<Nv24>::new(16, 8).with_rgba(&mut rgba_short);
  let Err(err) = result else {
    panic!("expected InsufficientRgbaBuffer error");
  };
  assert_eq!(
    err,
    MixedSinkerError::InsufficientRgbaBuffer(InsufficientBuffer::new(512, 511))
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv24_rgba_simd_matches_scalar_with_random_yuv() {
  // Width 1922 forces both the SIMD main loop AND scalar tail across
  // every backend block size (16/32/64).
  let w = 1922usize;
  let h = 4usize;
  let mut yp = std::vec![0u8; w * h];
  let mut uvp = std::vec![0u8; 2 * w * h];
  pseudo_random_u8(&mut yp, 0xC001_C0DE);
  pseudo_random_u8(&mut uvp, 0xCAFE_F00D);
  let src = Nv24Frame::new(&yp, &uvp, w as u32, h as u32, w as u32, (2 * w) as u32);

  for &matrix in &[
    KernelMatrix::Bt601,
    KernelMatrix::Bt709,
    KernelMatrix::Bt2020Ncl,
    KernelMatrix::YCgCo,
  ] {
    for &full_range in &[true, false] {
      let mut rgba_simd = std::vec![0u8; w * h * 4];
      let mut rgba_scalar = std::vec![0u8; w * h * 4];

      let mut s_simd = MixedSinker::<Nv24>::new(w, h)
        .with_rgba(&mut rgba_simd)
        .unwrap();
      nv24_to(&src, full_range, s_simd.set_kernel_matrix(matrix)).unwrap();

      let mut s_scalar = MixedSinker::<Nv24>::new(w, h)
        .with_rgba(&mut rgba_scalar)
        .unwrap();
      s_scalar.set_simd(false);
      nv24_to(&src, full_range, s_scalar.set_kernel_matrix(matrix)).unwrap();

      assert_eq!(
        rgba_simd, rgba_scalar,
        "Nv24 RGBA SIMD ≠ scalar (matrix={matrix:?}, full_range={full_range})"
      );
    }
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv42_rgba_only_converts_gray_to_gray_with_opaque_alpha() {
  let (yp, vup) = solid_nv42_frame(16, 8, 128, 128, 128);
  let src = Nv42Frame::new(&yp, &vup, 16, 8, 16, 32);

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Nv42>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  nv42_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(128) <= 1);
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
    assert_eq!(px[3], 0xFF);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv42_with_rgb_and_with_rgba_produce_byte_identical_rgb_bytes() {
  let w = 32u32;
  let h = 16u32;
  let ws = w as usize;
  let hs = h as usize;
  let (yp, vup) = solid_nv42_frame(w, h, 180, 60, 200);
  let src = Nv42Frame::new(&yp, &vup, w, h, w, 2 * w);

  let mut rgb = std::vec![0u8; ws * hs * 3];
  let mut rgba = std::vec![0u8; ws * hs * 4];
  let mut sink = MixedSinker::<Nv42>::new(ws, hs)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap();
  nv42_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();

  for i in 0..(ws * hs) {
    assert_eq!(rgba[i * 4], rgb[i * 3]);
    assert_eq!(rgba[i * 4 + 1], rgb[i * 3 + 1]);
    assert_eq!(rgba[i * 4 + 2], rgb[i * 3 + 2]);
    assert_eq!(rgba[i * 4 + 3], 0xFF);
  }
}

#[test]
fn nv42_rgba_buffer_too_short_returns_err() {
  let mut rgba_short = std::vec![0u8; 16 * 8 * 4 - 1];
  let result = MixedSinker::<Nv42>::new(16, 8).with_rgba(&mut rgba_short);
  let Err(err) = result else {
    panic!("expected InsufficientRgbaBuffer error");
  };
  assert_eq!(
    err,
    MixedSinkerError::InsufficientRgbaBuffer(InsufficientBuffer::new(512, 511))
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv42_rgba_simd_matches_scalar_with_random_yuv() {
  let w = 1922usize;
  let h = 4usize;
  let mut yp = std::vec![0u8; w * h];
  let mut vup = std::vec![0u8; 2 * w * h];
  pseudo_random_u8(&mut yp, 0xC001_C0DE);
  pseudo_random_u8(&mut vup, 0xCAFE_F00D);
  let src = Nv42Frame::new(&yp, &vup, w as u32, h as u32, w as u32, (2 * w) as u32);

  for &matrix in &[
    KernelMatrix::Bt601,
    KernelMatrix::Bt709,
    KernelMatrix::Bt2020Ncl,
    KernelMatrix::YCgCo,
  ] {
    for &full_range in &[true, false] {
      let mut rgba_simd = std::vec![0u8; w * h * 4];
      let mut rgba_scalar = std::vec![0u8; w * h * 4];

      let mut s_simd = MixedSinker::<Nv42>::new(w, h)
        .with_rgba(&mut rgba_simd)
        .unwrap();
      nv42_to(&src, full_range, s_simd.set_kernel_matrix(matrix)).unwrap();

      let mut s_scalar = MixedSinker::<Nv42>::new(w, h)
        .with_rgba(&mut rgba_scalar)
        .unwrap();
      s_scalar.set_simd(false);
      nv42_to(&src, full_range, s_scalar.set_kernel_matrix(matrix)).unwrap();

      assert_eq!(
        rgba_simd, rgba_scalar,
        "Nv42 RGBA SIMD ≠ scalar (matrix={matrix:?}, full_range={full_range})"
      );
    }
  }
}

// Cross-format Strategy A invariant: when both RGB+RGBA are
// attached, all 9 wired families derive RGBA from the RGB row via
// expand_rgb_to_rgba_row. This test runs all 9 process methods with
// the same gray input and asserts every RGBA sample equals the RGB
// sample with alpha = 0xFF — proving the fan-out shape never
// diverges from the kernel output.
#[cfg(all(
  feature = "yuv-planar",
  feature = "yuv-packed",
  feature = "yuv-444-packed",
  feature = "v210",
  feature = "y2xx",
))]
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn strategy_a_rgb_and_rgba_byte_identical_for_all_wired_families() {
  // Width chosen as a common multiple of 2 / 4 / 6 so all 4:2:0,
  // 4:2:2, 4:4:4 and v210 layouts work with the same dimensions.
  let w: u32 = 24;
  let h: u32 = 8;
  let ws = w as usize;
  let hs = h as usize;

  let assert_match = |rgb: &[u8], rgba: &[u8], who: &str| {
    for i in 0..(ws * hs) {
      assert_eq!(rgba[i * 4], rgb[i * 3], "{who}: R differs at px {i}");
      assert_eq!(
        rgba[i * 4 + 1],
        rgb[i * 3 + 1],
        "{who}: G differs at px {i}"
      );
      assert_eq!(
        rgba[i * 4 + 2],
        rgb[i * 3 + 2],
        "{who}: B differs at px {i}"
      );
      assert_eq!(rgba[i * 4 + 3], 0xFF, "{who}: alpha not opaque at px {i}");
    }
  };

  {
    let (yp, up, vp) = solid_yuv420p_frame(w, h, 200, 128, 128);
    let src = Yuv420pFrame::new(&yp, &up, &vp, w, h, w, w / 2, w / 2);
    let mut rgb = std::vec![0u8; ws * hs * 3];
    let mut rgba = std::vec![0u8; ws * hs * 4];
    let mut sink = MixedSinker::<Yuv420p>::new(ws, hs)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_rgba(&mut rgba)
      .unwrap();
    yuv420p_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
    assert_match(&rgb, &rgba, "Yuv420p");
  }

  {
    let (yp, up, vp) = solid_yuv422p_frame(w, h, 200, 128, 128);
    let src = Yuv422pFrame::new(&yp, &up, &vp, w, h, w, w / 2, w / 2);
    let mut rgb = std::vec![0u8; ws * hs * 3];
    let mut rgba = std::vec![0u8; ws * hs * 4];
    let mut sink = MixedSinker::<Yuv422p>::new(ws, hs)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_rgba(&mut rgba)
      .unwrap();
    yuv422p_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
    assert_match(&rgb, &rgba, "Yuv422p");
  }

  {
    let (yp, up, vp) = solid_yuv444p_frame(w, h, 200, 128, 128);
    let src = Yuv444pFrame::new(&yp, &up, &vp, w, h, w, w, w);
    let mut rgb = std::vec![0u8; ws * hs * 3];
    let mut rgba = std::vec![0u8; ws * hs * 4];
    let mut sink = MixedSinker::<Yuv444p>::new(ws, hs)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_rgba(&mut rgba)
      .unwrap();
    yuv444p_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
    assert_match(&rgb, &rgba, "Yuv444p");
  }

  {
    let (yp, uvp) = solid_nv12_frame(w, h, 200, 128, 128);
    let src = Nv12Frame::new(&yp, &uvp, w, h, w, w);
    let mut rgb = std::vec![0u8; ws * hs * 3];
    let mut rgba = std::vec![0u8; ws * hs * 4];
    let mut sink = MixedSinker::<Nv12>::new(ws, hs)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_rgba(&mut rgba)
      .unwrap();
    nv12_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
    assert_match(&rgb, &rgba, "Nv12");
  }

  {
    let (yp, vup) = solid_nv21_frame(w, h, 200, 128, 128);
    let src = Nv21Frame::new(&yp, &vup, w, h, w, w);
    let mut rgb = std::vec![0u8; ws * hs * 3];
    let mut rgba = std::vec![0u8; ws * hs * 4];
    let mut sink = MixedSinker::<Nv21>::new(ws, hs)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_rgba(&mut rgba)
      .unwrap();
    nv21_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
    assert_match(&rgb, &rgba, "Nv21");
  }

  {
    let (yp, uvp) = solid_nv16_frame(w, h, 200, 128, 128);
    let src = Nv16Frame::new(&yp, &uvp, w, h, w, w);
    let mut rgb = std::vec![0u8; ws * hs * 3];
    let mut rgba = std::vec![0u8; ws * hs * 4];
    let mut sink = MixedSinker::<Nv16>::new(ws, hs)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_rgba(&mut rgba)
      .unwrap();
    nv16_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
    assert_match(&rgb, &rgba, "Nv16");
  }

  {
    let (yp, uvp) = solid_nv24_frame(w, h, 200, 128, 128);
    let src = Nv24Frame::new(&yp, &uvp, w, h, w, 2 * w);
    let mut rgb = std::vec![0u8; ws * hs * 3];
    let mut rgba = std::vec![0u8; ws * hs * 4];
    let mut sink = MixedSinker::<Nv24>::new(ws, hs)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_rgba(&mut rgba)
      .unwrap();
    nv24_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
    assert_match(&rgb, &rgba, "Nv24");
  }

  {
    let (yp, vup) = solid_nv42_frame(w, h, 200, 128, 128);
    let src = Nv42Frame::new(&yp, &vup, w, h, w, 2 * w);
    let mut rgb = std::vec![0u8; ws * hs * 3];
    let mut rgba = std::vec![0u8; ws * hs * 4];
    let mut sink = MixedSinker::<Nv42>::new(ws, hs)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_rgba(&mut rgba)
      .unwrap();
    nv42_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
    assert_match(&rgb, &rgba, "Nv42");
  }

  {
    let (yp, up, vp) = solid_yuv440p_frame(w, h, 200, 128, 128);
    let src = Yuv440pFrame::new(&yp, &up, &vp, w, h, w, w, w);
    let mut rgb = std::vec![0u8; ws * hs * 3];
    let mut rgba = std::vec![0u8; ws * hs * 4];
    let mut sink = MixedSinker::<Yuv440p>::new(ws, hs)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_rgba(&mut rgba)
      .unwrap();
    yuv440p_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
    assert_match(&rgb, &rgba, "Yuv440p");
  }

  {
    let buf = solid_yuyv422_frame(w, h, 200, 128, 128);
    let src = Yuyv422Frame::new(&buf, w, h, 2 * w);
    let mut rgb = std::vec![0u8; ws * hs * 3];
    let mut rgba = std::vec![0u8; ws * hs * 4];
    let mut sink = MixedSinker::<Yuyv422>::new(ws, hs)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_rgba(&mut rgba)
      .unwrap();
    yuyv422_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
    assert_match(&rgb, &rgba, "Yuyv422");
  }

  {
    let buf = solid_uyvy422_frame(w, h, 200, 128, 128);
    let src = Uyvy422Frame::new(&buf, w, h, 2 * w);
    let mut rgb = std::vec![0u8; ws * hs * 3];
    let mut rgba = std::vec![0u8; ws * hs * 4];
    let mut sink = MixedSinker::<Uyvy422>::new(ws, hs)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_rgba(&mut rgba)
      .unwrap();
    uyvy422_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
    assert_match(&rgb, &rgba, "Uyvy422");
  }

  {
    let buf = solid_yvyu422_frame(w, h, 200, 128, 128);
    let src = Yvyu422Frame::new(&buf, w, h, 2 * w);
    let mut rgb = std::vec![0u8; ws * hs * 3];
    let mut rgba = std::vec![0u8; ws * hs * 4];
    let mut sink = MixedSinker::<Yvyu422>::new(ws, hs)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_rgba(&mut rgba)
      .unwrap();
    yvyu422_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
    assert_match(&rgb, &rgba, "Yvyu422");
  }

  {
    // V210 carries 10-bit samples; pick mid-range values inside 0..1024
    // so the gray-derived RGB stays sensible on the u8 path.
    let buf = solid_v210_frame(w, h, 700, 512, 512);
    let src = V210Frame::new(&buf, w, h, (w / 6) * 16);
    let mut rgb = std::vec![0u8; ws * hs * 3];
    let mut rgba = std::vec![0u8; ws * hs * 4];
    let mut sink = MixedSinker::<V210>::new(ws, hs)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_rgba(&mut rgba)
      .unwrap();
    v210_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
    assert_match(&rgb, &rgba, "V210");
  }

  {
    let buf = solid_y210_frame(w, h, 700, 512, 512);
    let src = Y210Frame::new(&buf, w, h, w * 2);
    let mut rgb = std::vec![0u8; ws * hs * 3];
    let mut rgba = std::vec![0u8; ws * hs * 4];
    let mut sink = MixedSinker::<Y210>::new(ws, hs)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_rgba(&mut rgba)
      .unwrap();
    y210_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
    assert_match(&rgb, &rgba, "Y210");
  }

  {
    let buf = solid_y212_frame(w, h, 700, 512, 512);
    let src = Y212Frame::new(&buf, w, h, w * 2);
    let mut rgb = std::vec![0u8; ws * hs * 3];
    let mut rgba = std::vec![0u8; ws * hs * 4];
    let mut sink = MixedSinker::<Y212>::new(ws, hs)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_rgba(&mut rgba)
      .unwrap();
    y212_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
    assert_match(&rgb, &rgba, "Y212");
  }

  {
    let buf = solid_y216_frame(w, h, 45000, 32768, 32768);
    let src = Y216Frame::new(&buf, w, h, w * 2);
    let mut rgb = std::vec![0u8; ws * hs * 3];
    let mut rgba = std::vec![0u8; ws * hs * 4];
    let mut sink = MixedSinker::<Y216>::new(ws, hs)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_rgba(&mut rgba)
      .unwrap();
    y216_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
    assert_match(&rgb, &rgba, "Y216");
  }

  {
    // V410 carries 10-bit samples; pick mid-range values inside 0..1024
    // so the gray-derived RGB stays sensible on the u8 path.
    let buf = solid_v410_frame(w, h, 200, 700, 400);
    let src = V410Frame::new(&buf, w, h, w);
    let mut rgb = std::vec![0u8; ws * hs * 3];
    let mut rgba = std::vec![0u8; ws * hs * 4];
    let mut sink = MixedSinker::<V410>::new(ws, hs)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_rgba(&mut rgba)
      .unwrap();
    v410_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
    assert_match(&rgb, &rgba, "V410");
  }

  {
    // V30X carries 10-bit samples (padding at LSB instead of MSB vs V410);
    // pick mid-range values inside 0..1024 so the gray-derived RGB stays
    // sensible on the u8 path.
    let buf = solid_v30x_frame(w, h, 200, 700, 400);
    let src = V30XFrame::new(&buf, w, h, w);
    let mut rgb = std::vec![0u8; ws * hs * 3];
    let mut rgba = std::vec![0u8; ws * hs * 4];
    let mut sink = MixedSinker::<V30X>::new(ws, hs)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_rgba(&mut rgba)
      .unwrap();
    v30x_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
    assert_match(&rgb, &rgba, "V30X");
  }

  {
    // XV36 carries 12-bit samples (MSB-aligned u16 quadruples);
    // pick mid-range 12-bit values so the gray-derived RGB stays
    // sensible on the u8 path. alpha_max = 0xFF for the u8 path.
    let buf = solid_xv36_frame(w, h, 0x300, 0x700, 0x500, 0);
    let src = Xv36Frame::new(&buf, w, h, w * 4);
    let mut rgb = std::vec![0u8; ws * hs * 3];
    let mut rgba = std::vec![0u8; ws * hs * 4];
    let mut sink = MixedSinker::<Xv36>::new(ws, hs)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_rgba(&mut rgba)
      .unwrap();
    xv36_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
    assert_match(&rgb, &rgba, "Xv36");
  }

  {
    // VUYX carries 8-bit samples; the A (padding) byte is ignored and
    // output α is always 0xFF. Strategy A applies: RGBA is derived from
    // the RGB row via expand_rgb_to_rgba_row (α=0xFF — no semantic
    // conflict). Use mid-range Y so the gray-derived RGB is sensible.
    let buf = solid_vuyx_frame(w, h, 128, 128, 200, 0x42);
    let src = VuyxFrame::try_new(&buf, w, h, w * 4).unwrap();
    let mut rgb = std::vec![0u8; ws * hs * 3];
    let mut rgba = std::vec![0u8; ws * hs * 4];
    let mut sink = MixedSinker::<Vuyx>::new(ws, hs)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_rgba(&mut rgba)
      .unwrap();
    vuyx_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
    assert_match(&rgb, &rgba, "Vuyx");
  }
}

// Cross-format Strategy A invariant on the **u16 RGB / u16 RGBA** path.
// First u16-output umbrella in the crate; future tranches will extend
// this with Y210 / Y212 / Y216 / Yuv420p10 etc. Initially V210 is the
// only entry — it is the first packed-source u16 sinker family wired
// in Ship 11a.
#[cfg(all(
  feature = "yuv-planar",
  feature = "yuv-packed",
  feature = "yuv-444-packed",
  feature = "v210",
  feature = "y2xx",
))]
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn strategy_a_rgb_u16_and_rgba_u16_byte_identical_for_all_wired_families() {
  let w: u32 = 12;
  let h: u32 = 8;
  let ws = w as usize;
  let hs = h as usize;
  let assert_match_u16 = |rgb: &[u16], rgba: &[u16], who: &str, alpha_max: u16| {
    for i in 0..(ws * hs) {
      assert_eq!(rgba[i * 4], rgb[i * 3], "{who}: R differs at px {i}");
      assert_eq!(
        rgba[i * 4 + 1],
        rgb[i * 3 + 1],
        "{who}: G differs at px {i}"
      );
      assert_eq!(
        rgba[i * 4 + 2],
        rgb[i * 3 + 2],
        "{who}: B differs at px {i}"
      );
      assert_eq!(rgba[i * 4 + 3], alpha_max, "{who}: alpha not max at px {i}");
    }
  };

  {
    let buf = solid_v210_frame(w, h, 700, 400, 600);
    let src = V210Frame::new(&buf, w, h, (w / 6) * 16);
    let mut rgb = std::vec![0u16; ws * hs * 3];
    let mut rgba = std::vec![0u16; ws * hs * 4];
    let mut sink = MixedSinker::<V210>::new(ws, hs)
      .with_rgb_u16(&mut rgb)
      .unwrap()
      .with_rgba_u16(&mut rgba)
      .unwrap();
    v210_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
    assert_match_u16(&rgb, &rgba, "V210", 1023);
  }

  {
    let buf = solid_y210_frame(w, h, 700, 400, 600);
    let src = Y210Frame::new(&buf, w, h, w * 2);
    let mut rgb = std::vec![0u16; ws * hs * 3];
    let mut rgba = std::vec![0u16; ws * hs * 4];
    let mut sink = MixedSinker::<Y210>::new(ws, hs)
      .with_rgb_u16(&mut rgb)
      .unwrap()
      .with_rgba_u16(&mut rgba)
      .unwrap();
    y210_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
    assert_match_u16(&rgb, &rgba, "Y210", 1023);
  }

  {
    let buf = solid_y212_frame(w, h, 700, 400, 600);
    let src = Y212Frame::new(&buf, w, h, w * 2);
    let mut rgb = std::vec![0u16; ws * hs * 3];
    let mut rgba = std::vec![0u16; ws * hs * 4];
    let mut sink = MixedSinker::<Y212>::new(ws, hs)
      .with_rgb_u16(&mut rgb)
      .unwrap()
      .with_rgba_u16(&mut rgba)
      .unwrap();
    y212_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
    assert_match_u16(&rgb, &rgba, "Y212", 4095);
  }

  {
    let buf = solid_y216_frame(w, h, 45000, 20000, 50000);
    let src = Y216Frame::new(&buf, w, h, w * 2);
    let mut rgb = std::vec![0u16; ws * hs * 3];
    let mut rgba = std::vec![0u16; ws * hs * 4];
    let mut sink = MixedSinker::<Y216>::new(ws, hs)
      .with_rgb_u16(&mut rgb)
      .unwrap()
      .with_rgba_u16(&mut rgba)
      .unwrap();
    y216_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
    assert_match_u16(&rgb, &rgba, "Y216", 0xFFFF);
  }

  {
    // V410 is 10-bit low-bit-packed; alpha_max = 0x3FF.
    let buf = solid_v410_frame(w, h, 200, 700, 400);
    let src = V410Frame::new(&buf, w, h, w);
    let mut rgb = std::vec![0u16; ws * hs * 3];
    let mut rgba = std::vec![0u16; ws * hs * 4];
    let mut sink = MixedSinker::<V410>::new(ws, hs)
      .with_rgb_u16(&mut rgb)
      .unwrap()
      .with_rgba_u16(&mut rgba)
      .unwrap();
    v410_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
    assert_match_u16(&rgb, &rgba, "V410", 0x3FF);
  }

  {
    // V30X is 10-bit low-bit-packed (padding at LSB); alpha_max = 0x3FF.
    let buf = solid_v30x_frame(w, h, 200, 700, 400);
    let src = V30XFrame::new(&buf, w, h, w);
    let mut rgb = std::vec![0u16; ws * hs * 3];
    let mut rgba = std::vec![0u16; ws * hs * 4];
    let mut sink = MixedSinker::<V30X>::new(ws, hs)
      .with_rgb_u16(&mut rgb)
      .unwrap()
      .with_rgba_u16(&mut rgba)
      .unwrap();
    v30x_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
    assert_match_u16(&rgb, &rgba, "V30X", 0x3FF);
  }

  {
    // XV36 is 12-bit low-bit-packed in the output; alpha_max = 0x0FFF.
    let buf = solid_xv36_frame(w, h, 0x300, 0x700, 0x500, 0);
    let src = Xv36Frame::new(&buf, w, h, w * 4);
    let mut rgb = std::vec![0u16; ws * hs * 3];
    let mut rgba = std::vec![0u16; ws * hs * 4];
    let mut sink = MixedSinker::<Xv36>::new(ws, hs)
      .with_rgb_u16(&mut rgb)
      .unwrap()
      .with_rgba_u16(&mut rgba)
      .unwrap();
    xv36_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
    assert_match_u16(&rgb, &rgba, "Xv36", 0x0FFF);
  }
}

// ---- with_luma_u16 (Task 2) -----------------------------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv24_with_luma_u16_extracts_y_zero_extended() {
  let width = 64usize;
  let height = 4usize;
  let n = width * height;

  let mut yp = std::vec![0u8; n];
  // NV24: UV row is 2 * width bytes (one U/V pair per Y pixel).
  let mut uvp = std::vec![0u8; 2 * width * height];
  pseudo_random_u8(&mut yp, 0xC0FFEE);
  pseudo_random_u8(&mut uvp, 0xBADF00D);

  let src = Nv24Frame::new(
    &yp,
    &uvp,
    width as u32,
    height as u32,
    width as u32,
    (2 * width) as u32,
  );

  let mut luma_out = std::vec![0u16; n];
  let mut sink = MixedSinker::<Nv24>::new(width, height)
    .with_luma_u16(&mut luma_out)
    .unwrap();
  nv24_to(&src, false, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();

  let expected: std::vec::Vec<u16> = yp.iter().map(|&y| y as u16).collect();
  assert_eq!(luma_out, expected, "Nv24 luma_u16 mismatch");
}

#[test]
fn nv24_luma_u16_buffer_too_short_returns_err() {
  let mut buf = std::vec![0u16; 16 * 8 - 1];
  let err = MixedSinker::<Nv24>::new(16, 8)
    .with_luma_u16(&mut buf)
    .err()
    .unwrap();
  assert_eq!(
    err,
    MixedSinkerError::InsufficientLumaU16Buffer(InsufficientBuffer::new(128, 127))
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv42_with_luma_u16_extracts_y_zero_extended() {
  let width = 64usize;
  let height = 4usize;
  let n = width * height;

  let mut yp = std::vec![0u8; n];
  // NV42: VU row is 2 * width bytes (one V/U pair per Y pixel).
  let mut vup = std::vec![0u8; 2 * width * height];
  pseudo_random_u8(&mut yp, 0xC0FFEE);
  pseudo_random_u8(&mut vup, 0xBADF00D);

  let src = Nv42Frame::new(
    &yp,
    &vup,
    width as u32,
    height as u32,
    width as u32,
    (2 * width) as u32,
  );

  let mut luma_out = std::vec![0u16; n];
  let mut sink = MixedSinker::<Nv42>::new(width, height)
    .with_luma_u16(&mut luma_out)
    .unwrap();
  nv42_to(&src, false, sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();

  let expected: std::vec::Vec<u16> = yp.iter().map(|&y| y as u16).collect();
  assert_eq!(luma_out, expected, "Nv42 luma_u16 mismatch");
}

#[test]
fn nv42_luma_u16_buffer_too_short_returns_err() {
  let mut buf = std::vec![0u16; 16 * 8 - 1];
  let err = MixedSinker::<Nv42>::new(16, 8)
    .with_luma_u16(&mut buf)
    .err()
    .unwrap();
  assert_eq!(
    err,
    MixedSinkerError::InsufficientLumaU16Buffer(InsufficientBuffer::new(128, 127))
  );
}

// Atomicity (#308): the up-front RGB-scratch preflight must return
// `AllocationFailed` BEFORE any output row is written, so an allocator refusal
// leaves the output frame untouched. Mirrors the planar_8bit / Yuv420p
// siblings. Reuses the crate's RGB-scratch failpoint (`yuva`-gated, so this
// test is too; under `--all-features` both `yuv-semi-planar` and `yuva` are
// on).
#[cfg(feature = "yuva")]
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv24_rgb_scratch_alloc_failure_leaves_outputs_untouched() {
  use crate::resample::ResampleError;

  // luma + RGBA + HSV with NO rgb output: `want_hsv && want_rgba && !want_rgb`
  // needs an RGB row but has no caller RGB buffer, so the decode grows the RGB
  // row scratch (`rgb_row_buf_or_scratch`'s scratch arm). With that allocation
  // armed to fail, the preflight returns AllocationFailed BEFORE any output
  // row — luma included — is written.
  let (yp, uvp) = solid_nv24_frame(16, 8, 42, 128, 128);
  let src = Nv24Frame::new(&yp, &uvp, 16, 8, 16, 32);
  let mut luma = std::vec![0xABu8; 16 * 8];
  let mut rgba = std::vec![0xCDu8; 16 * 8 * 4];
  let (mut hh, mut ss, mut vv) = (
    std::vec![0xCDu8; 16 * 8],
    std::vec![0xCDu8; 16 * 8],
    std::vec![0xCDu8; 16 * 8],
  );
  let mut sink = MixedSinker::<Nv24>::new(16, 8)
    .with_luma(&mut luma)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap()
    .with_hsv(&mut hh, &mut ss, &mut vv)
    .unwrap();

  super::super::super::arm_rgb_scratch_alloc_failure();
  let err = nv24_to(&src, false, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap_err();
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

// Nv42 twin of `nv24_rgb_scratch_alloc_failure_leaves_outputs_untouched`
// (#308) — same scratch arm (`want_hsv && want_rgba && !want_rgb`), VU-order
// chroma.
#[cfg(feature = "yuva")]
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv42_rgb_scratch_alloc_failure_leaves_outputs_untouched() {
  use crate::resample::ResampleError;

  let (yp, vup) = solid_nv42_frame(16, 8, 42, 128, 128);
  let src = Nv42Frame::new(&yp, &vup, 16, 8, 16, 32);
  let mut luma = std::vec![0xABu8; 16 * 8];
  let mut rgba = std::vec![0xCDu8; 16 * 8 * 4];
  let (mut hh, mut ss, mut vv) = (
    std::vec![0xCDu8; 16 * 8],
    std::vec![0xCDu8; 16 * 8],
    std::vec![0xCDu8; 16 * 8],
  );
  let mut sink = MixedSinker::<Nv42>::new(16, 8)
    .with_luma(&mut luma)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap()
    .with_hsv(&mut hh, &mut ss, &mut vv)
    .unwrap();

  super::super::super::arm_rgb_scratch_alloc_failure();
  let err = nv42_to(&src, false, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap_err();
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
