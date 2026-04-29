//! Tier 3 packed YUV 4:2:2 sinker tests — Ship 10.

use super::*;

// ---- Solid-color builders ----------------------------------------------

/// Builds a solid YUYV422 packed plane with one (Y, U, V) repeated
/// across `width × height`. Layout per 2-pixel block:
/// `Y0, U0, Y1, V0`. Stride equals `2 * width` (no padding).
pub(super) fn solid_yuyv422_frame(width: u32, height: u32, y: u8, u: u8, v: u8) -> Vec<u8> {
  let w = width as usize;
  let h = height as usize;
  let mut buf = std::vec![0u8; 2 * w * h];
  for row in 0..h {
    let base = row * 2 * w;
    for col in (0..w).step_by(2) {
      buf[base + col * 2] = y;
      buf[base + col * 2 + 1] = u;
      buf[base + col * 2 + 2] = y;
      buf[base + col * 2 + 3] = v;
    }
  }
  buf
}

/// Builds a solid UYVY422 packed plane with one (Y, U, V) repeated.
/// Layout per 2-pixel block: `U0, Y0, V0, Y1`.
pub(super) fn solid_uyvy422_frame(width: u32, height: u32, y: u8, u: u8, v: u8) -> Vec<u8> {
  let w = width as usize;
  let h = height as usize;
  let mut buf = std::vec![0u8; 2 * w * h];
  for row in 0..h {
    let base = row * 2 * w;
    for col in (0..w).step_by(2) {
      buf[base + col * 2] = u;
      buf[base + col * 2 + 1] = y;
      buf[base + col * 2 + 2] = v;
      buf[base + col * 2 + 3] = y;
    }
  }
  buf
}

/// Builds a solid YVYU422 packed plane with one (Y, U, V) repeated.
/// Layout per 2-pixel block: `Y0, V0, Y1, U0`.
pub(super) fn solid_yvyu422_frame(width: u32, height: u32, y: u8, u: u8, v: u8) -> Vec<u8> {
  let w = width as usize;
  let h = height as usize;
  let mut buf = std::vec![0u8; 2 * w * h];
  for row in 0..h {
    let base = row * 2 * w;
    for col in (0..w).step_by(2) {
      buf[base + col * 2] = y;
      buf[base + col * 2 + 1] = v;
      buf[base + col * 2 + 2] = y;
      buf[base + col * 2 + 3] = u;
    }
  }
  buf
}

// ---- Yuyv422 MixedSinker -----------------------------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuyv422_luma_only_extracts_y_bytes() {
  let buf = solid_yuyv422_frame(16, 8, 42, 128, 128);
  let src = Yuyv422Frame::new(&buf, 16, 8, 32);

  let mut luma = std::vec![0u8; 16 * 8];
  let mut sink = MixedSinker::<Yuyv422>::new(16, 8)
    .with_luma(&mut luma)
    .unwrap();
  yuyv422_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  assert!(luma.iter().all(|&y| y == 42));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuyv422_rgb_only_converts_gray_to_gray() {
  let buf = solid_yuyv422_frame(16, 8, 128, 128, 128);
  let src = Yuyv422Frame::new(&buf, 16, 8, 32);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yuyv422>::new(16, 8)
    .with_rgb(&mut rgb)
    .unwrap();
  yuyv422_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn yuyv422_rgba_only_converts_gray_to_gray_with_opaque_alpha() {
  let buf = solid_yuyv422_frame(16, 8, 128, 128, 128);
  let src = Yuyv422Frame::new(&buf, 16, 8, 32);

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuyv422>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  yuyv422_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn yuyv422_mixed_all_outputs_populated() {
  let buf = solid_yuyv422_frame(16, 8, 200, 128, 128);
  let src = Yuyv422Frame::new(&buf, 16, 8, 32);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut luma = std::vec![0u8; 16 * 8];
  let mut h = std::vec![0u8; 16 * 8];
  let mut s = std::vec![0u8; 16 * 8];
  let mut v = std::vec![0u8; 16 * 8];
  let mut sink = MixedSinker::<Yuyv422>::new(16, 8)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_luma(&mut luma)
    .unwrap()
    .with_hsv(&mut h, &mut s, &mut v)
    .unwrap();
  yuyv422_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn yuyv422_with_rgb_and_with_rgba_produce_byte_identical_rgb_bytes() {
  // Strategy A invariant: when both RGB and RGBA are attached, RGBA
  // bytes must equal the RGB row bytes + 0xFF alpha.
  let w = 32u32;
  let h = 16u32;
  let ws = w as usize;
  let hs = h as usize;
  let buf = solid_yuyv422_frame(w, h, 180, 60, 200);
  let src = Yuyv422Frame::new(&buf, w, h, 2 * w);

  let mut rgb = std::vec![0u8; ws * hs * 3];
  let mut rgba = std::vec![0u8; ws * hs * 4];
  let mut sink = MixedSinker::<Yuyv422>::new(ws, hs)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap();
  yuyv422_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn yuyv422_with_simd_false_matches_with_simd_true() {
  // Widths chosen to cover each backend's main loop AND scalar tail:
  // 16, 18 → NEON/SSE4.1/wasm main; 32, 34 → AVX2 main; 64, 66 →
  // AVX-512 main; 1920, 1922 → wide/baseline.
  for &w in &[16usize, 18, 32, 34, 64, 66, 128, 1920, 1922] {
    let h = 4usize;
    let mut packed = std::vec![0u8; 2 * w * h];
    pseudo_random_u8(&mut packed, 0xC001_C0DE);
    let src = Yuyv422Frame::new(&packed, w as u32, h as u32, (2 * w) as u32);

    let mut rgb_simd = std::vec![0u8; w * h * 3];
    let mut rgb_scalar = std::vec![0u8; w * h * 3];
    let mut sink_simd = MixedSinker::<Yuyv422>::new(w, h)
      .with_rgb(&mut rgb_simd)
      .unwrap();
    let mut sink_scalar = MixedSinker::<Yuyv422>::new(w, h)
      .with_rgb(&mut rgb_scalar)
      .unwrap()
      .with_simd(false);
    yuyv422_to(&src, false, ColorMatrix::Bt709, &mut sink_simd).unwrap();
    yuyv422_to(&src, false, ColorMatrix::Bt709, &mut sink_scalar).unwrap();

    assert_eq!(rgb_simd, rgb_scalar, "Yuyv422 SIMD≠scalar at width {w}");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuyv422_rgba_simd_matches_scalar_with_random_yuv() {
  let w = 1922usize;
  let h = 4usize;
  let mut packed = std::vec![0u8; 2 * w * h];
  pseudo_random_u8(&mut packed, 0xC001_C0DE);
  let src = Yuyv422Frame::new(&packed, w as u32, h as u32, (2 * w) as u32);

  for &matrix in &[
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::YCgCo,
  ] {
    for &full_range in &[true, false] {
      let mut rgba_simd = std::vec![0u8; w * h * 4];
      let mut rgba_scalar = std::vec![0u8; w * h * 4];

      let mut s_simd = MixedSinker::<Yuyv422>::new(w, h)
        .with_rgba(&mut rgba_simd)
        .unwrap();
      yuyv422_to(&src, full_range, matrix, &mut s_simd).unwrap();

      let mut s_scalar = MixedSinker::<Yuyv422>::new(w, h)
        .with_rgba(&mut rgba_scalar)
        .unwrap();
      s_scalar.set_simd(false);
      yuyv422_to(&src, full_range, matrix, &mut s_scalar).unwrap();

      assert_eq!(
        rgba_simd, rgba_scalar,
        "Yuyv422 RGBA SIMD ≠ scalar (matrix={matrix:?}, full_range={full_range})"
      );
    }
  }
}

#[test]
fn yuyv422_width_mismatch_returns_err() {
  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yuyv422>::new(16, 8)
    .with_rgb(&mut rgb)
    .unwrap();
  let buf = solid_yuyv422_frame(18, 8, 0, 0, 0);
  let src = Yuyv422Frame::new(&buf, 18, 8, 36);
  let err = yuyv422_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap_err();
  assert!(matches!(err, MixedSinkerError::DimensionMismatch { .. }));
}

#[test]
fn yuyv422_process_rejects_short_packed_slice() {
  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yuyv422>::new(16, 8)
    .with_rgb(&mut rgb)
    .unwrap();
  let packed = [0u8; 31]; // expected 2 * 16 = 32
  let row = Yuyv422Row::new(&packed, 0, ColorMatrix::Bt601, true);
  let err = sink.process(row).err().unwrap();
  assert_eq!(
    err,
    MixedSinkerError::RowShapeMismatch {
      which: RowSlice::Yuyv422Packed,
      row: 0,
      expected: 32,
      actual: 31,
    }
  );
}

#[test]
fn yuyv422_process_rejects_out_of_range_row_idx() {
  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yuyv422>::new(16, 8)
    .with_rgb(&mut rgb)
    .unwrap();
  let packed = [128u8; 32];
  let row = Yuyv422Row::new(&packed, 8, ColorMatrix::Bt601, true);
  let err = sink.process(row).err().unwrap();
  assert_eq!(
    err,
    MixedSinkerError::RowIndexOutOfRange {
      row: 8,
      configured_height: 8,
    }
  );
}

#[test]
fn yuyv422_odd_width_rejected_in_begin_frame() {
  let mut rgb = std::vec![0u8; 17 * 8 * 3];
  let mut sink = MixedSinker::<Yuyv422>::new(17, 8)
    .with_rgb(&mut rgb)
    .unwrap();
  let err = sink.begin_frame(17, 8).err().unwrap();
  assert_eq!(err, MixedSinkerError::OddWidth { width: 17 });
}

#[test]
fn yuyv422_rgba_buffer_too_short_returns_err() {
  let mut rgba_short = std::vec![0u8; 16 * 8 * 4 - 1];
  let result = MixedSinker::<Yuyv422>::new(16, 8).with_rgba(&mut rgba_short);
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

// ---- Uyvy422 MixedSinker -----------------------------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn uyvy422_rgb_only_converts_gray_to_gray() {
  let buf = solid_uyvy422_frame(16, 8, 128, 128, 128);
  let src = Uyvy422Frame::new(&buf, 16, 8, 32);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Uyvy422>::new(16, 8)
    .with_rgb(&mut rgb)
    .unwrap();
  uyvy422_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn uyvy422_rgba_only_converts_gray_to_gray_with_opaque_alpha() {
  let buf = solid_uyvy422_frame(16, 8, 128, 128, 128);
  let src = Uyvy422Frame::new(&buf, 16, 8, 32);

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Uyvy422>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  uyvy422_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn uyvy422_with_rgb_and_with_rgba_produce_byte_identical_rgb_bytes() {
  let w = 32u32;
  let h = 16u32;
  let ws = w as usize;
  let hs = h as usize;
  let buf = solid_uyvy422_frame(w, h, 180, 60, 200);
  let src = Uyvy422Frame::new(&buf, w, h, 2 * w);

  let mut rgb = std::vec![0u8; ws * hs * 3];
  let mut rgba = std::vec![0u8; ws * hs * 4];
  let mut sink = MixedSinker::<Uyvy422>::new(ws, hs)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap();
  uyvy422_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for i in 0..(ws * hs) {
    assert_eq!(rgba[i * 4], rgb[i * 3]);
    assert_eq!(rgba[i * 4 + 1], rgb[i * 3 + 1]);
    assert_eq!(rgba[i * 4 + 2], rgb[i * 3 + 2]);
    assert_eq!(rgba[i * 4 + 3], 0xFF);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn uyvy422_with_simd_false_matches_with_simd_true() {
  for &w in &[16usize, 18, 32, 34, 64, 66, 128, 1920, 1922] {
    let h = 4usize;
    let mut packed = std::vec![0u8; 2 * w * h];
    pseudo_random_u8(&mut packed, 0xCAFE_F00D);
    let src = Uyvy422Frame::new(&packed, w as u32, h as u32, (2 * w) as u32);

    let mut rgb_simd = std::vec![0u8; w * h * 3];
    let mut rgb_scalar = std::vec![0u8; w * h * 3];
    let mut sink_simd = MixedSinker::<Uyvy422>::new(w, h)
      .with_rgb(&mut rgb_simd)
      .unwrap();
    let mut sink_scalar = MixedSinker::<Uyvy422>::new(w, h)
      .with_rgb(&mut rgb_scalar)
      .unwrap()
      .with_simd(false);
    uyvy422_to(&src, false, ColorMatrix::Bt709, &mut sink_simd).unwrap();
    uyvy422_to(&src, false, ColorMatrix::Bt709, &mut sink_scalar).unwrap();

    assert_eq!(rgb_simd, rgb_scalar, "Uyvy422 SIMD≠scalar at width {w}");
  }
}

#[test]
fn uyvy422_process_rejects_short_packed_slice() {
  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Uyvy422>::new(16, 8)
    .with_rgb(&mut rgb)
    .unwrap();
  let packed = [0u8; 31];
  let row = Uyvy422Row::new(&packed, 0, ColorMatrix::Bt601, true);
  let err = sink.process(row).err().unwrap();
  assert_eq!(
    err,
    MixedSinkerError::RowShapeMismatch {
      which: RowSlice::Uyvy422Packed,
      row: 0,
      expected: 32,
      actual: 31,
    }
  );
}

// ---- Yvyu422 MixedSinker -----------------------------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yvyu422_rgb_only_converts_gray_to_gray() {
  let buf = solid_yvyu422_frame(16, 8, 128, 128, 128);
  let src = Yvyu422Frame::new(&buf, 16, 8, 32);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yvyu422>::new(16, 8)
    .with_rgb(&mut rgb)
    .unwrap();
  yvyu422_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn yvyu422_rgba_only_converts_gray_to_gray_with_opaque_alpha() {
  let buf = solid_yvyu422_frame(16, 8, 128, 128, 128);
  let src = Yvyu422Frame::new(&buf, 16, 8, 32);

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yvyu422>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  yvyu422_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

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
fn yvyu422_with_rgb_and_with_rgba_produce_byte_identical_rgb_bytes() {
  let w = 32u32;
  let h = 16u32;
  let ws = w as usize;
  let hs = h as usize;
  let buf = solid_yvyu422_frame(w, h, 180, 60, 200);
  let src = Yvyu422Frame::new(&buf, w, h, 2 * w);

  let mut rgb = std::vec![0u8; ws * hs * 3];
  let mut rgba = std::vec![0u8; ws * hs * 4];
  let mut sink = MixedSinker::<Yvyu422>::new(ws, hs)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap();
  yvyu422_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for i in 0..(ws * hs) {
    assert_eq!(rgba[i * 4], rgb[i * 3]);
    assert_eq!(rgba[i * 4 + 1], rgb[i * 3 + 1]);
    assert_eq!(rgba[i * 4 + 2], rgb[i * 3 + 2]);
    assert_eq!(rgba[i * 4 + 3], 0xFF);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yvyu422_with_simd_false_matches_with_simd_true() {
  for &w in &[16usize, 18, 32, 34, 64, 66, 128, 1920, 1922] {
    let h = 4usize;
    let mut packed = std::vec![0u8; 2 * w * h];
    pseudo_random_u8(&mut packed, 0xDEAD_BEEF);
    let src = Yvyu422Frame::new(&packed, w as u32, h as u32, (2 * w) as u32);

    let mut rgb_simd = std::vec![0u8; w * h * 3];
    let mut rgb_scalar = std::vec![0u8; w * h * 3];
    let mut sink_simd = MixedSinker::<Yvyu422>::new(w, h)
      .with_rgb(&mut rgb_simd)
      .unwrap();
    let mut sink_scalar = MixedSinker::<Yvyu422>::new(w, h)
      .with_rgb(&mut rgb_scalar)
      .unwrap()
      .with_simd(false);
    yvyu422_to(&src, false, ColorMatrix::Bt709, &mut sink_simd).unwrap();
    yvyu422_to(&src, false, ColorMatrix::Bt709, &mut sink_scalar).unwrap();

    assert_eq!(rgb_simd, rgb_scalar, "Yvyu422 SIMD≠scalar at width {w}");
  }
}

#[test]
fn yvyu422_process_rejects_short_packed_slice() {
  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yvyu422>::new(16, 8)
    .with_rgb(&mut rgb)
    .unwrap();
  let packed = [0u8; 31];
  let row = Yvyu422Row::new(&packed, 0, ColorMatrix::Bt601, true);
  let err = sink.process(row).err().unwrap();
  assert_eq!(
    err,
    MixedSinkerError::RowShapeMismatch {
      which: RowSlice::Yvyu422Packed,
      row: 0,
      expected: 32,
      actual: 31,
    }
  );
}

// ---- Cross-format byte-permutation invariants --------------------------
//
// The three packed YUV 4:2:2 formats differ only in byte position
// within each 4-byte / 2-pixel block. Re-permuting one buffer into
// another's layout must produce identical RGB output — proves the
// kernels' deinterleave logic correctly threads byte positions
// through to the same Q15 chroma pipeline.

/// Permutes a yuyv422 buffer (`Y0 U Y1 V`) into a uyvy422 buffer
/// (`U Y0 V Y1`) by swapping bytes within each adjacent pair.
fn yuyv_to_uyvy_perm(yuyv: &[u8]) -> Vec<u8> {
  let mut out = std::vec![0u8; yuyv.len()];
  for i in (0..yuyv.len()).step_by(2) {
    out[i] = yuyv[i + 1];
    out[i + 1] = yuyv[i];
  }
  out
}

/// Permutes a yuyv422 buffer (`Y0 U Y1 V`) into a yvyu422 buffer
/// (`Y0 V Y1 U`) by swapping bytes 1 and 3 in each 4-byte block.
fn yuyv_to_yvyu_perm(yuyv: &[u8]) -> Vec<u8> {
  let mut out = std::vec![0u8; yuyv.len()];
  for i in (0..yuyv.len()).step_by(4) {
    out[i] = yuyv[i];
    out[i + 1] = yuyv[i + 3];
    out[i + 2] = yuyv[i + 2];
    out[i + 3] = yuyv[i + 1];
  }
  out
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuyv422_byte_swap_matches_uyvy422_rgb_output() {
  // YUYV ↔ UYVY differ only in Y/chroma byte position; re-permuting
  // one buffer's bytes into the other's layout must yield identical
  // RGB for every (Y0, Y1, U, V) tuple.
  let w = 32usize;
  let h = 4usize;
  let mut yuyv = std::vec![0u8; 2 * w * h];
  pseudo_random_u8(&mut yuyv, 0xDEADC0DE);
  let uyvy = yuyv_to_uyvy_perm(&yuyv);

  let yuyv_src = Yuyv422Frame::new(&yuyv, w as u32, h as u32, (2 * w) as u32);
  let uyvy_src = Uyvy422Frame::new(&uyvy, w as u32, h as u32, (2 * w) as u32);

  let mut rgb_yuyv = std::vec![0u8; w * h * 3];
  let mut rgb_uyvy = std::vec![0u8; w * h * 3];
  let mut s_yuyv = MixedSinker::<Yuyv422>::new(w, h)
    .with_rgb(&mut rgb_yuyv)
    .unwrap();
  let mut s_uyvy = MixedSinker::<Uyvy422>::new(w, h)
    .with_rgb(&mut rgb_uyvy)
    .unwrap();
  yuyv422_to(&yuyv_src, true, ColorMatrix::Bt601, &mut s_yuyv).unwrap();
  uyvy422_to(&uyvy_src, true, ColorMatrix::Bt601, &mut s_uyvy).unwrap();

  assert_eq!(rgb_yuyv, rgb_uyvy);
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuyv422_uv_swap_matches_yvyu422_rgb_output() {
  // YUYV ↔ YVYU differ only in chroma byte order. Swapping bytes 1
  // and 3 within each 4-byte block converts a YUYV buffer into the
  // matching YVYU buffer; both must produce identical RGB output.
  let w = 32usize;
  let h = 4usize;
  let mut yuyv = std::vec![0u8; 2 * w * h];
  pseudo_random_u8(&mut yuyv, 0xCAFEBABE);
  let yvyu = yuyv_to_yvyu_perm(&yuyv);

  let yuyv_src = Yuyv422Frame::new(&yuyv, w as u32, h as u32, (2 * w) as u32);
  let yvyu_src = Yvyu422Frame::new(&yvyu, w as u32, h as u32, (2 * w) as u32);

  let mut rgb_yuyv = std::vec![0u8; w * h * 3];
  let mut rgb_yvyu = std::vec![0u8; w * h * 3];
  let mut s_yuyv = MixedSinker::<Yuyv422>::new(w, h)
    .with_rgb(&mut rgb_yuyv)
    .unwrap();
  let mut s_yvyu = MixedSinker::<Yvyu422>::new(w, h)
    .with_rgb(&mut rgb_yvyu)
    .unwrap();
  yuyv422_to(&yuyv_src, true, ColorMatrix::Bt709, &mut s_yuyv).unwrap();
  yvyu422_to(&yvyu_src, true, ColorMatrix::Bt709, &mut s_yvyu).unwrap();

  assert_eq!(rgb_yuyv, rgb_yvyu);
}

// ---- Planar parity oracle ----------------------------------------------
//
// `Yuv422p` and packed 4:2:2 share the same chroma topology
// (half-width U/V, full-height). Re-packing planar samples into a
// packed-format buffer must produce identical RGB — proves the
// packed kernels match the well-validated planar kernels modulo
// byte permutation.

fn pack_yuv422p_to_yuyv422(y: &[u8], u: &[u8], v: &[u8], w: usize, h: usize) -> Vec<u8> {
  let cw = w / 2;
  let mut out = std::vec![0u8; 2 * w * h];
  for row in 0..h {
    for c in 0..cw {
      let dst = row * 2 * w + c * 4;
      out[dst] = y[row * w + 2 * c];
      out[dst + 1] = u[row * cw + c];
      out[dst + 2] = y[row * w + 2 * c + 1];
      out[dst + 3] = v[row * cw + c];
    }
  }
  out
}

fn pack_yuv422p_to_uyvy422(y: &[u8], u: &[u8], v: &[u8], w: usize, h: usize) -> Vec<u8> {
  let cw = w / 2;
  let mut out = std::vec![0u8; 2 * w * h];
  for row in 0..h {
    for c in 0..cw {
      let dst = row * 2 * w + c * 4;
      out[dst] = u[row * cw + c];
      out[dst + 1] = y[row * w + 2 * c];
      out[dst + 2] = v[row * cw + c];
      out[dst + 3] = y[row * w + 2 * c + 1];
    }
  }
  out
}

fn pack_yuv422p_to_yvyu422(y: &[u8], u: &[u8], v: &[u8], w: usize, h: usize) -> Vec<u8> {
  let cw = w / 2;
  let mut out = std::vec![0u8; 2 * w * h];
  for row in 0..h {
    for c in 0..cw {
      let dst = row * 2 * w + c * 4;
      out[dst] = y[row * w + 2 * c];
      out[dst + 1] = v[row * cw + c];
      out[dst + 2] = y[row * w + 2 * c + 1];
      out[dst + 3] = u[row * cw + c];
    }
  }
  out
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuyv422_reconstructed_from_yuv422p_matches_yuv422p_to_rgb() {
  let w = 32usize;
  let h = 4usize;
  let mut y_plane = std::vec![0u8; w * h];
  let mut u_plane = std::vec![0u8; (w / 2) * h];
  let mut v_plane = std::vec![0u8; (w / 2) * h];
  pseudo_random_u8(&mut y_plane, 0xC0FFEE);
  pseudo_random_u8(&mut u_plane, 0xBADF00D);
  pseudo_random_u8(&mut v_plane, 0xFEEDFACE);

  let planar = Yuv422pFrame::new(
    &y_plane,
    &u_plane,
    &v_plane,
    w as u32,
    h as u32,
    w as u32,
    (w / 2) as u32,
    (w / 2) as u32,
  );
  let packed = pack_yuv422p_to_yuyv422(&y_plane, &u_plane, &v_plane, w, h);
  let yuyv = Yuyv422Frame::new(&packed, w as u32, h as u32, (2 * w) as u32);

  let mut rgb_planar = std::vec![0u8; w * h * 3];
  let mut rgb_packed = std::vec![0u8; w * h * 3];
  let mut s_planar = MixedSinker::<Yuv422p>::new(w, h)
    .with_rgb(&mut rgb_planar)
    .unwrap();
  let mut s_packed = MixedSinker::<Yuyv422>::new(w, h)
    .with_rgb(&mut rgb_packed)
    .unwrap();
  yuv422p_to(&planar, false, ColorMatrix::Bt709, &mut s_planar).unwrap();
  yuyv422_to(&yuyv, false, ColorMatrix::Bt709, &mut s_packed).unwrap();

  assert_eq!(rgb_planar, rgb_packed);
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn uyvy422_reconstructed_from_yuv422p_matches_yuv422p_to_rgb() {
  let w = 32usize;
  let h = 4usize;
  let mut y_plane = std::vec![0u8; w * h];
  let mut u_plane = std::vec![0u8; (w / 2) * h];
  let mut v_plane = std::vec![0u8; (w / 2) * h];
  pseudo_random_u8(&mut y_plane, 0x1234567);
  pseudo_random_u8(&mut u_plane, 0x89ABCDE);
  pseudo_random_u8(&mut v_plane, 0xFEDCBA9);

  let planar = Yuv422pFrame::new(
    &y_plane,
    &u_plane,
    &v_plane,
    w as u32,
    h as u32,
    w as u32,
    (w / 2) as u32,
    (w / 2) as u32,
  );
  let packed = pack_yuv422p_to_uyvy422(&y_plane, &u_plane, &v_plane, w, h);
  let uyvy = Uyvy422Frame::new(&packed, w as u32, h as u32, (2 * w) as u32);

  let mut rgb_planar = std::vec![0u8; w * h * 3];
  let mut rgb_packed = std::vec![0u8; w * h * 3];
  let mut s_planar = MixedSinker::<Yuv422p>::new(w, h)
    .with_rgb(&mut rgb_planar)
    .unwrap();
  let mut s_packed = MixedSinker::<Uyvy422>::new(w, h)
    .with_rgb(&mut rgb_packed)
    .unwrap();
  yuv422p_to(&planar, false, ColorMatrix::Bt2020Ncl, &mut s_planar).unwrap();
  uyvy422_to(&uyvy, false, ColorMatrix::Bt2020Ncl, &mut s_packed).unwrap();

  assert_eq!(rgb_planar, rgb_packed);
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yvyu422_reconstructed_from_yuv422p_matches_yuv422p_to_rgb() {
  let w = 32usize;
  let h = 4usize;
  let mut y_plane = std::vec![0u8; w * h];
  let mut u_plane = std::vec![0u8; (w / 2) * h];
  let mut v_plane = std::vec![0u8; (w / 2) * h];
  pseudo_random_u8(&mut y_plane, 0x1A2B3C4D);
  pseudo_random_u8(&mut u_plane, 0x5E6F7081);
  pseudo_random_u8(&mut v_plane, 0x9203F4E5);

  let planar = Yuv422pFrame::new(
    &y_plane,
    &u_plane,
    &v_plane,
    w as u32,
    h as u32,
    w as u32,
    (w / 2) as u32,
    (w / 2) as u32,
  );
  let packed = pack_yuv422p_to_yvyu422(&y_plane, &u_plane, &v_plane, w, h);
  let yvyu = Yvyu422Frame::new(&packed, w as u32, h as u32, (2 * w) as u32);

  let mut rgb_planar = std::vec![0u8; w * h * 3];
  let mut rgb_packed = std::vec![0u8; w * h * 3];
  let mut s_planar = MixedSinker::<Yuv422p>::new(w, h)
    .with_rgb(&mut rgb_planar)
    .unwrap();
  let mut s_packed = MixedSinker::<Yvyu422>::new(w, h)
    .with_rgb(&mut rgb_packed)
    .unwrap();
  yuv422p_to(&planar, true, ColorMatrix::Bt601, &mut s_planar).unwrap();
  yvyu422_to(&yvyu, true, ColorMatrix::Bt601, &mut s_packed).unwrap();

  assert_eq!(rgb_planar, rgb_packed);
}
