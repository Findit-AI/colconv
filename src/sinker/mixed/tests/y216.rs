//! Tier 4 Y216 sinker tests — Ship 11d.
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

// ---- Solid-color Y216 builder -----------------------------------------

/// Builds a solid-color Y216 plane with one (Y, U, V) repeated. Each
/// row is `width × 2` u16 elements (`Y₀, U, Y₁, V` quadruples). All
/// samples are direct 16-bit values — Y216 uses the full u16 range
/// with no MSB-alignment shift (bits[15:0] all active).
///
/// Width must be even (4:2:2 chroma pair).
pub(super) fn solid_y216_frame(width: u32, height: u32, y: u16, u: u16, v: u16) -> Vec<u16> {
  assert!(width.is_multiple_of(2), "Y216 requires even width");
  let row_elems = (width as usize) * 2;
  let mut buf = std::vec![0u16; row_elems * height as usize];
  for row in 0..height as usize {
    let off = row * row_elems;
    for q in 0..(width as usize / 2) {
      let base = off + q * 4;
      buf[base] = y;
      buf[base + 1] = u;
      buf[base + 2] = y;
      buf[base + 3] = v;
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
fn y216_luma_only_extracts_y_bytes_downshifted() {
  // Y=51200 (16-bit direct) → 8-bit (51200 >> 8) = 200.
  let buf = solid_y216_frame(6, 8, 51200, 32768, 32768);
  let src = Y216Frame::new(&buf, 6, 8, 12);
  let mut luma = std::vec![0u8; 6 * 8];
  let mut sink = MixedSinker::<Y216>::new(6, 8).with_luma(&mut luma).unwrap();
  y216_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();
  // 16-bit Y=51200 → 8-bit (51200 >> 8) = 200.
  assert!(luma.iter().all(|&y| y == 200), "luma {luma:?}");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn y216_luma_u16_only_extracts_y_native_depth() {
  // Y216 luma_u16 is a direct extract — no shift.
  let buf = solid_y216_frame(6, 8, 51200, 32768, 32768);
  let src = Y216Frame::new(&buf, 6, 8, 12);
  let mut luma = std::vec![0u16; 6 * 8];
  let mut sink = MixedSinker::<Y216>::new(6, 8)
    .with_luma_u16(&mut luma)
    .unwrap();
  y216_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();
  assert!(
    luma.iter().all(|&y| y == 51200),
    "luma_u16 {:?}",
    &luma[..16]
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn y216_rgb_only_converts_gray_to_gray() {
  // Y=32896 (≈ 128/255 × 65535), U=V=32768 (neutral chroma, ≈ 0.5×65535).
  let buf = solid_y216_frame(12, 4, 32896, 32768, 32768);
  let src = Y216Frame::new(&buf, 12, 4, 24);
  let mut rgb = std::vec![0u8; 12 * 4 * 3];
  let mut sink = MixedSinker::<Y216>::new(12, 4).with_rgb(&mut rgb).unwrap();
  y216_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();
  for px in rgb.chunks(3) {
    assert!(px[0].abs_diff(128) <= 2);
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn y216_rgba_only_converts_gray_to_gray_with_opaque_alpha() {
  let buf = solid_y216_frame(12, 4, 32896, 32768, 32768);
  let src = Y216Frame::new(&buf, 12, 4, 24);
  let mut rgba = std::vec![0u8; 12 * 4 * 4];
  let mut sink = MixedSinker::<Y216>::new(12, 4)
    .with_rgba(&mut rgba)
    .unwrap();
  y216_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();
  for px in rgba.chunks(4) {
    assert_eq!(px[3], 0xFF);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn y216_rgb_u16_only_converts_gray_to_gray_native_depth() {
  let buf = solid_y216_frame(12, 4, 32896, 32768, 32768);
  let src = Y216Frame::new(&buf, 12, 4, 24);
  let mut rgb = std::vec![0u16; 12 * 4 * 3];
  let mut sink = MixedSinker::<Y216>::new(12, 4)
    .with_rgb_u16(&mut rgb)
    .unwrap();
  y216_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();
  for px in rgb.chunks(3) {
    // Mid-gray at 16-bit depth ≈ 32768; allow ±256 tolerance for
    // Q15 rounding on the 16-bit pipeline.
    assert!(
      px[0].abs_diff(32768) <= 512,
      "expected ~32768, got {}",
      px[0]
    );
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn y216_rgba_u16_alpha_is_max() {
  let buf = solid_y216_frame(12, 4, 32896, 32768, 32768);
  let src = Y216Frame::new(&buf, 12, 4, 24);
  let mut rgba = std::vec![0u16; 12 * 4 * 4];
  let mut sink = MixedSinker::<Y216>::new(12, 4)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  y216_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();
  for px in rgba.chunks(4) {
    assert_eq!(px[3], 0xFFFF);
  }
}

// ---- Strategy A invariant tests ---------------------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn y216_with_rgb_and_with_rgba_byte_identical_u8() {
  // Strategy A invariant on u8 path — calling both `with_rgb` and
  // `with_rgba` must produce the same RGB bytes in both buffers, with
  // alpha = 0xFF in the RGBA buffer.
  let w = 12u32;
  let h = 4u32;
  let buf = solid_y216_frame(w, h, 45000, 20000, 50000);
  let src = Y216Frame::new(&buf, w, h, w * 2);
  let mut rgb = std::vec![0u8; (w * h) as usize * 3];
  let mut rgba = std::vec![0u8; (w * h) as usize * 4];
  let mut sink = MixedSinker::<Y216>::new(w as usize, h as usize)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap();
  y216_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();
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
fn y216_with_rgb_u16_and_with_rgba_u16_byte_identical() {
  // Strategy A invariant on u16 path — alpha must be 0xFFFF (full
  // 16-bit opaque).
  let w = 12u32;
  let h = 4u32;
  let buf = solid_y216_frame(w, h, 45000, 20000, 50000);
  let src = Y216Frame::new(&buf, w, h, w * 2);
  let mut rgb = std::vec![0u16; (w * h) as usize * 3];
  let mut rgba = std::vec![0u16; (w * h) as usize * 4];
  let mut sink = MixedSinker::<Y216>::new(w as usize, h as usize)
    .with_rgb_u16(&mut rgb)
    .unwrap()
    .with_rgba_u16(&mut rgba)
    .unwrap();
  y216_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();
  for i in 0..(w * h) as usize {
    assert_eq!(rgba[i * 4], rgb[i * 3]);
    assert_eq!(rgba[i * 4 + 1], rgb[i * 3 + 1]);
    assert_eq!(rgba[i * 4 + 2], rgb[i * 3 + 2]);
    assert_eq!(rgba[i * 4 + 3], 0xFFFF);
  }
}

// ---- SIMD-vs-scalar parity --------------------------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn y216_with_simd_false_matches_with_simd_true() {
  // Pseudo-random Y216 across multiple widths covering the main loop
  // + scalar tail of every backend block size. Each u16 sample uses
  // the full 16-bit range — no MSB alignment needed for Y216.
  for w in [2usize, 4, 14, 16, 18, 30, 32, 34, 1920, 1922] {
    let h = 2usize;
    let mut buf = std::vec![0u16; w * 2 * h];
    pseudo_random_u16_low_n_bits(&mut buf, 0xC0FFEE, 16);
    let src = Y216Frame::new(&buf, w as u32, h as u32, (w * 2) as u32);

    let mut rgb_simd = std::vec![0u8; w * h * 3];
    let mut rgb_scalar = std::vec![0u8; w * h * 3];
    let mut sink_simd = MixedSinker::<Y216>::new(w, h)
      .with_rgb(&mut rgb_simd)
      .unwrap();
    let mut sink_scalar = MixedSinker::<Y216>::new(w, h)
      .with_rgb(&mut rgb_scalar)
      .unwrap()
      .with_simd(false);
    y216_to(&src, false, ColorMatrix::Bt709, &mut sink_simd).unwrap();
    y216_to(&src, false, ColorMatrix::Bt709, &mut sink_scalar).unwrap();
    assert_eq!(rgb_simd, rgb_scalar, "Y216 SIMD≠scalar at width {w}");
  }
}

// ---- Error-path tests --------------------------------------------------

#[test]
fn y216_odd_width_returns_err() {
  // Direct `process()` call with a sink configured at width=3 (odd —
  // violates 4:2:2 chroma-pair constraint). The width check fires
  // *before* any kernel runs, preserving the no-panic contract — even
  // if the caller bypasses the walker (which would catch this in
  // `begin_frame`).
  let mut rgb = std::vec![0u8; 4 * 3];
  let mut sink = MixedSinker::<Y216>::new(3, 1).with_rgb(&mut rgb).unwrap();
  let buf = std::vec![0u16; 6];
  let row = Y216Row::new(&buf, 0, ColorMatrix::Bt601, true);
  let err = sink.process(row).err().unwrap();
  assert!(matches!(err, MixedSinkerError::OddWidth { width: 3 }));
}

#[test]
fn y216_process_rejects_short_packed_slice() {
  // 6-pixel-wide sink expects 12 u16 elements per row; an 11-element
  // slice surfaces as `RowShapeMismatch { which: Y216Packed, .. }`.
  let mut rgb = std::vec![0u8; 6 * 3];
  let mut sink = MixedSinker::<Y216>::new(6, 1).with_rgb(&mut rgb).unwrap();
  let packed = [0u16; 11];
  let row = Y216Row::new(&packed, 0, ColorMatrix::Bt601, true);
  let err = sink.process(row).err().unwrap();
  assert_eq!(
    err,
    MixedSinkerError::RowShapeMismatch {
      which: RowSlice::Y216Packed,
      row: 0,
      expected: 12,
      actual: 11,
    }
  );
}

#[test]
fn y216_luma_u16_buffer_too_short_returns_err() {
  // Buffer holds 6×7 = 42 elements; a 6×8 frame needs 48.
  let mut luma = std::vec![0u16; 6 * 7];
  let result = MixedSinker::<Y216>::new(6, 8).with_luma_u16(&mut luma);
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
