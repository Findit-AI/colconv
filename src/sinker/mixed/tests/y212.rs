//! Tier 4 Y212 sinker tests — Ship 11c.
//!
//! Coverage matrix:
//! - Single-output paths (luma u8, luma u16, rgb, rgba, rgb_u16,
//!   rgba_u16) on solid-gray frames.
//! - Strategy A invariant (`with_rgb` + `with_rgba` byte-identical;
//!   same for the u16 variants).
//! - SIMD-vs-scalar parity across multiple widths covering the main
//!   loop + scalar tail of every backend block size.
//! - Three error-path tests: odd width, short packed slice, short
//!   luma_u16 buffer.

use super::*;

// ---- Solid-color Y212 builder -----------------------------------------

/// Builds a solid-color Y212 plane with one (Y, U, V) repeated. Each
/// row is `width × 2` u16 elements (`Y₀, U, Y₁, V` quadruples). All
/// samples are MSB-aligned 12-bit: each `u16` value's active 12 bits
/// occupy bits[15:4] and bits[3:0] are zero.
///
/// Width must be even (4:2:2 chroma pair).
pub(super) fn solid_y212_frame(width: u32, height: u32, y: u16, u: u16, v: u16) -> Vec<u16> {
  assert!(width.is_multiple_of(2), "Y212 requires even width");
  let row_elems = (width as usize) * 2;
  let mut buf = std::vec![0u16; row_elems * height as usize];
  let y_msb = (y & 0xFFF) << 4;
  let u_msb = (u & 0xFFF) << 4;
  let v_msb = (v & 0xFFF) << 4;
  for row in 0..height as usize {
    let off = row * row_elems;
    for q in 0..(width as usize / 2) {
      let base = off + q * 4;
      buf[base] = y_msb;
      buf[base + 1] = u_msb;
      buf[base + 2] = y_msb;
      buf[base + 3] = v_msb;
    }
  }
  buf
}

// ---- Single-output gray-to-gray tests ---------------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn y212_luma_only_extracts_y_bytes_downshifted() {
  let buf = solid_y212_frame(6, 8, 200, 2048, 2048); // Y=200 (12-bit native).
  let src = Y212Frame::new(&buf, 6, 8, 12);
  let mut luma = std::vec![0u8; 6 * 8];
  let mut sink = MixedSinker::<Y212>::new(6, 8).with_luma(&mut luma).unwrap();
  y212_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();
  // 12-bit Y=200 → 8-bit (200 >> 4) = 12.
  assert!(luma.iter().all(|&y| y == 12), "luma {luma:?}");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn y212_luma_u16_only_extracts_y_native_depth() {
  let buf = solid_y212_frame(6, 8, 200, 2048, 2048);
  let src = Y212Frame::new(&buf, 6, 8, 12);
  let mut luma = std::vec![0u16; 6 * 8];
  let mut sink = MixedSinker::<Y212>::new(6, 8)
    .with_luma_u16(&mut luma)
    .unwrap();
  y212_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();
  assert!(luma.iter().all(|&y| y == 200), "luma_u16 {:?}", &luma[..16]);
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn y212_rgb_only_converts_gray_to_gray() {
  let buf = solid_y212_frame(12, 4, 2048, 2048, 2048);
  let src = Y212Frame::new(&buf, 12, 4, 24);
  let mut rgb = std::vec![0u8; 12 * 4 * 3];
  let mut sink = MixedSinker::<Y212>::new(12, 4).with_rgb(&mut rgb).unwrap();
  y212_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();
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
fn y212_rgba_only_converts_gray_to_gray_with_opaque_alpha() {
  let buf = solid_y212_frame(12, 4, 2048, 2048, 2048);
  let src = Y212Frame::new(&buf, 12, 4, 24);
  let mut rgba = std::vec![0u8; 12 * 4 * 4];
  let mut sink = MixedSinker::<Y212>::new(12, 4)
    .with_rgba(&mut rgba)
    .unwrap();
  y212_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();
  for px in rgba.chunks(4) {
    assert_eq!(px[3], 0xFF);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn y212_rgb_u16_only_converts_gray_to_gray_native_depth() {
  let buf = solid_y212_frame(12, 4, 2048, 2048, 2048);
  let src = Y212Frame::new(&buf, 12, 4, 24);
  let mut rgb = std::vec![0u16; 12 * 4 * 3];
  let mut sink = MixedSinker::<Y212>::new(12, 4)
    .with_rgb_u16(&mut rgb)
    .unwrap();
  y212_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();
  for px in rgb.chunks(3) {
    assert!(px[0].abs_diff(2048) <= 2, "expected ~2048, got {}", px[0]);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn y212_rgba_u16_alpha_is_max() {
  let buf = solid_y212_frame(12, 4, 2048, 2048, 2048);
  let src = Y212Frame::new(&buf, 12, 4, 24);
  let mut rgba = std::vec![0u16; 12 * 4 * 4];
  let mut sink = MixedSinker::<Y212>::new(12, 4)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  y212_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();
  for px in rgba.chunks(4) {
    assert_eq!(px[3], 4095);
  }
}

// ---- Strategy A invariant tests ---------------------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn y212_with_rgb_and_with_rgba_byte_identical_u8() {
  // Strategy A invariant on u8 path — calling both `with_rgb` and
  // `with_rgba` must produce the same RGB bytes in both buffers, with
  // alpha = 0xFF in the RGBA buffer.
  let w = 12u32;
  let h = 4u32;
  let buf = solid_y212_frame(w, h, 700, 400, 600);
  let src = Y212Frame::new(&buf, w, h, w * 2);
  let mut rgb = std::vec![0u8; (w * h) as usize * 3];
  let mut rgba = std::vec![0u8; (w * h) as usize * 4];
  let mut sink = MixedSinker::<Y212>::new(w as usize, h as usize)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap();
  y212_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();
  for i in 0..(w * h) as usize {
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
fn y212_with_rgb_u16_and_with_rgba_u16_byte_identical() {
  // Strategy A invariant on u16 path.
  let w = 12u32;
  let h = 4u32;
  let buf = solid_y212_frame(w, h, 700, 400, 600);
  let src = Y212Frame::new(&buf, w, h, w * 2);
  let mut rgb = std::vec![0u16; (w * h) as usize * 3];
  let mut rgba = std::vec![0u16; (w * h) as usize * 4];
  let mut sink = MixedSinker::<Y212>::new(w as usize, h as usize)
    .with_rgb_u16(&mut rgb)
    .unwrap()
    .with_rgba_u16(&mut rgba)
    .unwrap();
  y212_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();
  for i in 0..(w * h) as usize {
    assert_eq!(rgba[i * 4], rgb[i * 3]);
    assert_eq!(rgba[i * 4 + 1], rgb[i * 3 + 1]);
    assert_eq!(rgba[i * 4 + 2], rgb[i * 3 + 2]);
    assert_eq!(rgba[i * 4 + 3], 4095);
  }
}

// ---- SIMD-vs-scalar parity --------------------------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn y212_with_simd_false_matches_with_simd_true() {
  // Pseudo-random Y212 across multiple widths covering the main loop
  // + scalar tail of every backend block size. Each u16 sample is
  // generated with 12 active bits, then shifted `<< 4` to become
  // MSB-aligned per the Y212 spec.
  for w in [2usize, 4, 14, 16, 18, 30, 32, 34, 1920, 1922] {
    let h = 2usize;
    let mut buf = std::vec![0u16; w * 2 * h];
    pseudo_random_u16_low_n_bits(&mut buf, 0xC0FFEE, 12);
    // MSB-align: each sample's active 12 bits move to bits[15:4].
    for s in &mut buf {
      *s <<= 4;
    }
    let src = Y212Frame::new(&buf, w as u32, h as u32, (w * 2) as u32);

    let mut rgb_simd = std::vec![0u8; w * h * 3];
    let mut rgb_scalar = std::vec![0u8; w * h * 3];
    let mut sink_simd = MixedSinker::<Y212>::new(w, h)
      .with_rgb(&mut rgb_simd)
      .unwrap();
    let mut sink_scalar = MixedSinker::<Y212>::new(w, h)
      .with_rgb(&mut rgb_scalar)
      .unwrap()
      .with_simd(false);
    y212_to(&src, false, ColorMatrix::Bt709, &mut sink_simd).unwrap();
    y212_to(&src, false, ColorMatrix::Bt709, &mut sink_scalar).unwrap();
    assert_eq!(rgb_simd, rgb_scalar, "Y212 SIMD≠scalar at width {w}");
  }
}

// ---- Error-path tests --------------------------------------------------

#[test]
fn y212_odd_width_returns_err() {
  // Direct `process()` call with a sink configured at width=3 (odd —
  // violates 4:2:2 chroma-pair constraint). The width check fires
  // *before* any kernel runs, preserving the no-panic contract — even
  // if the caller bypasses the walker (which would catch this in
  // `begin_frame`).
  let mut rgb = std::vec![0u8; 4 * 3];
  let mut sink = MixedSinker::<Y212>::new(3, 1).with_rgb(&mut rgb).unwrap();
  let buf = std::vec![0u16; 6];
  let row = Y212Row::new(&buf, 0, ColorMatrix::Bt601, true);
  let err = sink.process(row).err().unwrap();
  assert!(matches!(err, MixedSinkerError::OddWidth { width: 3 }));
}

#[test]
fn y212_process_rejects_short_packed_slice() {
  // 6-pixel-wide sink expects 12 u16 elements per row; an 11-element
  // slice surfaces as `RowShapeMismatch { which: Y212Packed, .. }`.
  let mut rgb = std::vec![0u8; 6 * 3];
  let mut sink = MixedSinker::<Y212>::new(6, 1).with_rgb(&mut rgb).unwrap();
  let packed = [0u16; 11];
  let row = Y212Row::new(&packed, 0, ColorMatrix::Bt601, true);
  let err = sink.process(row).err().unwrap();
  assert_eq!(
    err,
    MixedSinkerError::RowShapeMismatch {
      which: RowSlice::Y212Packed,
      row: 0,
      expected: 12,
      actual: 11,
    }
  );
}

#[test]
fn y212_luma_u16_buffer_too_short_returns_err() {
  // Buffer holds 6×7 = 42 elements; a 6×8 frame needs 48.
  let mut luma = std::vec![0u16; 6 * 7];
  let result = MixedSinker::<Y212>::new(6, 8).with_luma_u16(&mut luma);
  let Err(err) = result else {
    panic!("expected LumaU16BufferTooShort");
  };
  assert!(matches!(
    err,
    MixedSinkerError::LumaU16BufferTooShort {
      expected: 48,
      actual: 42,
    }
  ));
}

// ---- Cross-format oracles ---------------------------------------------
//
// Planar parity: re-pack a Yuv422p12 source into Y212 layout and verify
// both produce byte-identical RGB. Y212 is just an MSB-aligned u16
// packing of 4:2:2 12-bit planar data, so the converted output MUST
// match Yuv422p12's output for the same samples.

/// Pack three 12-bit planes (Y / U / V at 4:2:2 subsampling) into the
/// Y212 layout — each row is `width × 2` u16 elements laid out as
/// `(Y₀, U, Y₁, V)` quadruples with each sample MSB-aligned (active 12
/// bits in bits[15:4]). Width must be even.
fn pack_yuv422p12_to_y212(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  width: usize,
  height: usize,
) -> Vec<u16> {
  let cw = width / 2;
  let mut out = std::vec![0u16; width * 2 * height];
  for row in 0..height {
    for c in 0..cw {
      let off = row * width * 2 + c * 4;
      out[off] = (y[row * width + 2 * c] << 4) & 0xFFF0;
      out[off + 1] = (u[row * cw + c] << 4) & 0xFFF0;
      out[off + 2] = (y[row * width + 2 * c + 1] << 4) & 0xFFF0;
      out[off + 3] = (v[row * cw + c] << 4) & 0xFFF0;
    }
  }
  out
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn y212_reconstructed_from_yuv422p12_matches_yuv422p12_to_rgb() {
  let w = 16usize;
  let h = 4usize;
  let mut y_plane = std::vec![0u16; w * h];
  let mut u_plane = std::vec![0u16; (w / 2) * h];
  let mut v_plane = std::vec![0u16; (w / 2) * h];
  pseudo_random_u16_low_n_bits(&mut y_plane, 0xC0FFEE, 12);
  pseudo_random_u16_low_n_bits(&mut u_plane, 0xBADF00D, 12);
  pseudo_random_u16_low_n_bits(&mut v_plane, 0xFEEDFACE, 12);

  let planar = Yuv422p12Frame::new(
    &y_plane,
    &u_plane,
    &v_plane,
    w as u32,
    h as u32,
    w as u32,
    (w / 2) as u32,
    (w / 2) as u32,
  );
  let packed = pack_yuv422p12_to_y212(&y_plane, &u_plane, &v_plane, w, h);
  let y212 = Y212Frame::new(&packed, w as u32, h as u32, (w * 2) as u32);

  let mut rgb_planar = std::vec![0u8; w * h * 3];
  let mut rgb_packed = std::vec![0u8; w * h * 3];
  let mut s_planar = MixedSinker::<Yuv422p12>::new(w, h)
    .with_rgb(&mut rgb_planar)
    .unwrap();
  let mut s_packed = MixedSinker::<Y212>::new(w, h)
    .with_rgb(&mut rgb_packed)
    .unwrap();
  yuv422p12_to(&planar, false, ColorMatrix::Bt709, &mut s_planar).unwrap();
  y212_to(&y212, false, ColorMatrix::Bt709, &mut s_packed).unwrap();

  assert_eq!(rgb_planar, rgb_packed);
}
