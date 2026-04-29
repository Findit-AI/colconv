use super::*;

// ---- NV16 MixedSinker ---------------------------------------------------
//
// 4:2:2: chroma is half-width, full-height. Per-row math is
// identical to NV12 (the impl calls `nv12_to_rgb_row`), so the
// tests mirror the NV12 set and add a cross-layout parity check
// against an NV12-shaped frame whose chroma rows are each
// duplicated (simulating 4:2:0 from 4:2:2 by vertical downsampling).

pub(super) fn solid_nv16_frame(width: u32, height: u32, y: u8, u: u8, v: u8) -> (Vec<u8>, Vec<u8>) {
  let w = width as usize;
  let h = height as usize;
  // NV16 UV is full-height (h rows, not h/2).
  let mut uv = std::vec![0u8; w * h];
  for row in 0..h {
    for i in 0..w / 2 {
      uv[row * w + i * 2] = u;
      uv[row * w + i * 2 + 1] = v;
    }
  }
  (std::vec![y; w * h], uv)
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv16_luma_only_copies_y_plane() {
  let (yp, uvp) = solid_nv16_frame(16, 8, 42, 128, 128);
  let src = Nv16Frame::new(&yp, &uvp, 16, 8, 16, 16);

  let mut luma = std::vec![0u8; 16 * 8];
  let mut sink = MixedSinker::<Nv16>::new(16, 8)
    .with_luma(&mut luma)
    .unwrap();
  nv16_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  assert!(luma.iter().all(|&y| y == 42));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv16_rgb_only_converts_gray_to_gray() {
  let (yp, uvp) = solid_nv16_frame(16, 8, 128, 128, 128);
  let src = Nv16Frame::new(&yp, &uvp, 16, 8, 16, 16);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Nv16>::new(16, 8).with_rgb(&mut rgb).unwrap();
  nv16_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn nv16_mixed_all_three_outputs_populated() {
  let (yp, uvp) = solid_nv16_frame(16, 8, 200, 128, 128);
  let src = Nv16Frame::new(&yp, &uvp, 16, 8, 16, 16);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut luma = std::vec![0u8; 16 * 8];
  let mut h = std::vec![0u8; 16 * 8];
  let mut s = std::vec![0u8; 16 * 8];
  let mut v = std::vec![0u8; 16 * 8];
  let mut sink = MixedSinker::<Nv16>::new(16, 8)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_luma(&mut luma)
    .unwrap()
    .with_hsv(&mut h, &mut s, &mut v)
    .unwrap();
  nv16_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn nv16_with_simd_false_matches_with_simd_true() {
  let w = 32usize;
  let h = 16usize;
  let yp: Vec<u8> = (0..w * h).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
  let uvp: Vec<u8> = (0..w * h).map(|i| ((i * 53 + 23) & 0xFF) as u8).collect();
  let src = Nv16Frame::new(&yp, &uvp, w as u32, h as u32, w as u32, w as u32);

  let mut rgb_simd = std::vec![0u8; w * h * 3];
  let mut rgb_scalar = std::vec![0u8; w * h * 3];
  let mut sink_simd = MixedSinker::<Nv16>::new(w, h)
    .with_rgb(&mut rgb_simd)
    .unwrap();
  let mut sink_scalar = MixedSinker::<Nv16>::new(w, h)
    .with_rgb(&mut rgb_scalar)
    .unwrap()
    .with_simd(false);
  nv16_to(&src, false, ColorMatrix::Bt709, &mut sink_simd).unwrap();
  nv16_to(&src, false, ColorMatrix::Bt709, &mut sink_scalar).unwrap();

  assert_eq!(rgb_simd, rgb_scalar);
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv16_matches_nv12_mixed_sinker_with_duplicated_chroma() {
  // Cross-layout parity: if we build an NV12 frame whose `uv_half`
  // plane contains only the even NV16 chroma rows (row 0, 2, 4, …),
  // the two frames must produce identical RGB output at every Y
  // row. This validates that NV16's walker + NV12's row primitive
  // yield the right 4:2:2 semantics (one UV row per Y row) on a
  // 4:2:0 reference that shares chroma across row pairs.
  let w = 32usize;
  let h = 16usize;
  let yp: Vec<u8> = (0..w * h).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
  let uv_nv16: Vec<u8> = (0..w * h).map(|i| ((i * 53 + 23) & 0xFF) as u8).collect();
  // Build NV12 chroma by sampling only even NV16 chroma rows.
  let mut uv_nv12 = std::vec![0u8; w * h / 2];
  for c_row in 0..h / 2 {
    let src_row = c_row * 2; // even NV16 chroma rows
    uv_nv12[c_row * w..(c_row + 1) * w].copy_from_slice(&uv_nv16[src_row * w..(src_row + 1) * w]);
  }
  // …and make the NV16 odd chroma rows match their even neighbors so
  // the 4:2:0 vertical upsample (same chroma for row pairs) matches
  // what NV16 carries through.
  let mut uv_nv16_aligned = uv_nv16.clone();
  for c_row in 0..h / 2 {
    let even_row = c_row * 2;
    let odd_row = even_row + 1;
    let (even, odd) = uv_nv16_aligned.split_at_mut(odd_row * w);
    odd[..w].copy_from_slice(&even[even_row * w..even_row * w + w]);
  }
  let nv16_src = Nv16Frame::new(
    &yp,
    &uv_nv16_aligned,
    w as u32,
    h as u32,
    w as u32,
    w as u32,
  );
  let nv12_src = Nv12Frame::new(&yp, &uv_nv12, w as u32, h as u32, w as u32, w as u32);

  let mut rgb_nv16 = std::vec![0u8; w * h * 3];
  let mut rgb_nv12 = std::vec![0u8; w * h * 3];
  let mut s_nv16 = MixedSinker::<Nv16>::new(w, h)
    .with_rgb(&mut rgb_nv16)
    .unwrap();
  let mut s_nv12 = MixedSinker::<Nv12>::new(w, h)
    .with_rgb(&mut rgb_nv12)
    .unwrap();
  nv16_to(&nv16_src, false, ColorMatrix::Bt709, &mut s_nv16).unwrap();
  nv12_to(&nv12_src, false, ColorMatrix::Bt709, &mut s_nv12).unwrap();

  assert_eq!(rgb_nv16, rgb_nv12);
}

// ---- NV16 RGBA (Ship 8 PR 3) tests --------------------------------------
//
// NV16 reuses the NV12 `_to_rgba_row` dispatcher (4:2:2's row
// contract is identical to NV12's). Tests mirror the NV12 set;
// the cross-format invariant against NV12 (with duplicated
// chroma rows so 4:2:0 vertical upsample matches NV16's per-row
// chroma) catches any wiring regression specific to the NV16
// walker that the kernel-level tests don't cover.

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv16_rgba_only_converts_gray_to_gray_with_opaque_alpha() {
  let (yp, uvp) = solid_nv16_frame(16, 8, 128, 128, 128);
  let src = Nv16Frame::new(&yp, &uvp, 16, 8, 16, 16);

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Nv16>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  nv16_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn nv16_with_rgb_and_with_rgba_produce_byte_identical_rgb_bytes() {
  let w = 32usize;
  let h = 16usize;
  let (yp, uvp) = solid_nv16_frame(w as u32, h as u32, 180, 60, 200);
  let src = Nv16Frame::new(&yp, &uvp, w as u32, h as u32, w as u32, w as u32);

  let mut rgb = std::vec![0u8; w * h * 3];
  let mut rgba = std::vec![0u8; w * h * 4];
  let mut sink = MixedSinker::<Nv16>::new(w, h)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap();
  nv16_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for i in 0..(w * h) {
    assert_eq!(rgba[i * 4], rgb[i * 3], "R differs at pixel {i}");
    assert_eq!(rgba[i * 4 + 1], rgb[i * 3 + 1], "G differs at pixel {i}");
    assert_eq!(rgba[i * 4 + 2], rgb[i * 3 + 2], "B differs at pixel {i}");
    assert_eq!(rgba[i * 4 + 3], 0xFF, "A not opaque at pixel {i}");
  }
}

#[test]
fn nv16_rgba_buffer_too_short_returns_err() {
  let mut rgba_short = std::vec![0u8; 16 * 8 * 4 - 1];
  let result = MixedSinker::<Nv16>::new(16, 8).with_rgba(&mut rgba_short);
  let Err(err) = result else {
    panic!("expected RgbaBufferTooShort error");
  };
  assert!(matches!(
    err,
    MixedSinkerError::RgbaBufferTooShort {
      expected: 512,
      actual: 511,
    }
  ));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv16_rgba_simd_matches_scalar_with_random_yuv() {
  // NV16 reuses the NV12 RGBA kernel; this test pins the wiring
  // regardless of which tier the dispatcher picks. Width 1922 +
  // height 4 to exercise both main loop and tail per backend.
  let w = 1922usize;
  let h = 4usize;
  let mut yp = std::vec![0u8; w * h];
  let mut uvp = std::vec![0u8; w * h]; // NV16 UV is full-height
  pseudo_random_u8(&mut yp, 0xC001_C0DE);
  pseudo_random_u8(&mut uvp, 0xCAFE_F00D);
  let src = Nv16Frame::new(&yp, &uvp, w as u32, h as u32, w as u32, w as u32);

  for &matrix in &[
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::YCgCo,
  ] {
    for &full_range in &[true, false] {
      let mut rgba_simd = std::vec![0u8; w * h * 4];
      let mut rgba_scalar = std::vec![0u8; w * h * 4];

      let mut s_simd = MixedSinker::<Nv16>::new(w, h)
        .with_rgba(&mut rgba_simd)
        .unwrap();
      nv16_to(&src, full_range, matrix, &mut s_simd).unwrap();

      let mut s_scalar = MixedSinker::<Nv16>::new(w, h)
        .with_rgba(&mut rgba_scalar)
        .unwrap();
      s_scalar.set_simd(false);
      nv16_to(&src, full_range, matrix, &mut s_scalar).unwrap();

      if rgba_simd != rgba_scalar {
        let mismatch = rgba_simd
          .iter()
          .zip(rgba_scalar.iter())
          .position(|(a, b)| a != b)
          .unwrap();
        let pixel = mismatch / 4;
        let channel = ["R", "G", "B", "A"][mismatch % 4];
        panic!(
          "NV16 RGBA SIMD ≠ scalar at byte {mismatch} (px {pixel} {channel}) for matrix={matrix:?} full_range={full_range}: simd={} scalar={}",
          rgba_simd[mismatch], rgba_scalar[mismatch]
        );
      }
    }
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv16_rgba_matches_nv12_rgba_with_duplicated_chroma() {
  // Cross-format invariant on the RGBA path. Mirrors the existing
  // `nv16_matches_nv12_mixed_sinker_with_duplicated_chroma` for
  // RGB: duplicating NV16 chroma rows pairwise so the 4:2:0
  // vertical upsample matches NV16's per-row chroma must yield
  // byte-identical RGBA. Catches NV16-vs-NV12 wiring regressions
  // specific to the new RGBA path.
  let w = 32usize;
  let h = 16usize;
  let yp: Vec<u8> = (0..w * h).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
  let uv_nv16: Vec<u8> = (0..w * h).map(|i| ((i * 53 + 23) & 0xFF) as u8).collect();
  let mut uv_nv12 = std::vec![0u8; w * h / 2];
  for c_row in 0..h / 2 {
    let src_row = c_row * 2;
    uv_nv12[c_row * w..(c_row + 1) * w].copy_from_slice(&uv_nv16[src_row * w..(src_row + 1) * w]);
  }
  let mut uv_nv16_aligned = uv_nv16.clone();
  for c_row in 0..h / 2 {
    let even_row = c_row * 2;
    let odd_row = even_row + 1;
    let (even, odd) = uv_nv16_aligned.split_at_mut(odd_row * w);
    odd[..w].copy_from_slice(&even[even_row * w..even_row * w + w]);
  }
  let nv16_src = Nv16Frame::new(
    &yp,
    &uv_nv16_aligned,
    w as u32,
    h as u32,
    w as u32,
    w as u32,
  );
  let nv12_src = Nv12Frame::new(&yp, &uv_nv12, w as u32, h as u32, w as u32, w as u32);

  let mut rgba_nv16 = std::vec![0u8; w * h * 4];
  let mut rgba_nv12 = std::vec![0u8; w * h * 4];
  let mut s_nv16 = MixedSinker::<Nv16>::new(w, h)
    .with_rgba(&mut rgba_nv16)
    .unwrap();
  let mut s_nv12 = MixedSinker::<Nv12>::new(w, h)
    .with_rgba(&mut rgba_nv12)
    .unwrap();
  nv16_to(&nv16_src, false, ColorMatrix::Bt709, &mut s_nv16).unwrap();
  nv12_to(&nv12_src, false, ColorMatrix::Bt709, &mut s_nv12).unwrap();

  assert_eq!(rgba_nv16, rgba_nv12);
}

#[test]
fn nv16_odd_width_sink_returns_err_at_begin_frame() {
  let mut rgb = std::vec![0u8; 15 * 8 * 3];
  let mut sink = MixedSinker::<Nv16>::new(15, 8).with_rgb(&mut rgb).unwrap();
  let (yp, uvp) = solid_nv16_frame(16, 8, 0, 0, 0); // dummy 16-wide frame
  let src = Nv16Frame::new(&yp, &uvp, 16, 8, 16, 16);
  let err = nv16_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap_err();
  assert!(matches!(err, MixedSinkerError::OddWidth { width: 15 }));
}
