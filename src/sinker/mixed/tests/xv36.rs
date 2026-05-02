//! Tier 5 XV36 sinker tests — Ship 12b.
//!
//! Coverage matrix:
//! - Single-output paths (rgb, rgba, rgb_u16, rgba_u16, luma u8,
//!   luma u16) on solid-gray frames.
//! - Strategy A invariant (`with_rgb` + `with_rgba` byte-identical;
//!   same for the u16 variants with BITS=12).
//! - SIMD-vs-scalar parity across multiple widths covering the main
//!   loop + scalar tail of every backend block size.
//! - Three error-path tests: row index out of range, short packed
//!   slice, and short rgba_u16 buffer.
//!
//! XV36 specifics vs V410:
//! - Source buffer is `&[u16]` (4 × u16 per pixel), not `&[u32]`.
//! - Channels are 12-bit MSB-aligned (low 4 bits zero per sample).
//! - `with_luma_u16` is supported natively (Y >> 4 → low-bit-packed
//!   12-bit value).
//! - u16 alpha max = `0x0FFF` (12-bit max, not 0x3FF / 0xFFFF).

#[cfg(all(test, feature = "std"))]
use super::*;

// ---- Solid-color XV36 builder -----------------------------------------

/// Builds a solid-color XV36 plane with one (U, Y, V, A) quadruple
/// repeated. Each pixel is stored as four u16 words:
///   word 0: U  (12-bit, MSB-aligned: `u_val << 4`)
///   word 1: Y  (12-bit, MSB-aligned: `y_val << 4`)
///   word 2: V  (12-bit, MSB-aligned: `v_val << 4`)
///   word 3: A  (12-bit, MSB-aligned: `a_val << 4`, padding)
///
/// The parameters `u`, `y`, `v`, `a` are raw 12-bit values (`[0, 4095]`);
/// they are left-shifted by 4 internally to produce the MSB-aligned
/// representation stored in the buffer.
///
/// Row stride equals `width × 4` u16 elements (no padding between rows).
#[cfg(all(test, feature = "std"))]
pub(super) fn solid_xv36_frame(
  width: u32,
  height: u32,
  u: u16,
  y: u16,
  v: u16,
  a: u16,
) -> Vec<u16> {
  debug_assert!(u <= 0xFFF && y <= 0xFFF && v <= 0xFFF && a <= 0xFFF);
  let quad = [u << 4, y << 4, v << 4, a << 4];
  (0..(width as usize) * (height as usize))
    .flat_map(|_| quad)
    .collect()
}

// ---- Single-output gray-to-gray tests ---------------------------------

#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn xv36_rgb_only_converts_gray_to_gray() {
  // Y=U=V=0x800 (12-bit midpoint ≈ 0.5×4095). After YUV→RGB at 12-bit
  // depth the per-channel u8 value should be near 128 ± tolerance.
  let buf = solid_xv36_frame(12, 4, 0x800, 0x800, 0x800, 0);
  let src = Xv36Frame::new(&buf, 12, 4, 48);
  let mut rgb = std::vec![0u8; 12 * 4 * 3];
  let mut sink = MixedSinker::<Xv36>::new(12, 4).with_rgb(&mut rgb).unwrap();
  xv36_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();
  for px in rgb.chunks(3) {
    assert!(px[0].abs_diff(128) <= 4, "expected ~128, got {}", px[0]);
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
  }
}

#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn xv36_rgba_only_converts_gray_to_gray_with_opaque_alpha() {
  // Alpha must be forced to 0xFF (XV36 A slot is padding).
  let buf = solid_xv36_frame(12, 4, 0x800, 0x800, 0x800, 0);
  let src = Xv36Frame::new(&buf, 12, 4, 48);
  let mut rgba = std::vec![0u8; 12 * 4 * 4];
  let mut sink = MixedSinker::<Xv36>::new(12, 4)
    .with_rgba(&mut rgba)
    .unwrap();
  xv36_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();
  for px in rgba.chunks(4) {
    assert_eq!(px[3], 0xFF, "alpha must be 0xFF for XV36");
  }
}

#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn xv36_rgb_u16_only_converts_gray_to_gray_native_depth() {
  // Y=U=V=0x800 (12-bit midpoint = 2048). After YUV→RGB at 12-bit depth
  // the per-channel u16 value should be near 2048; allow ±0x10 for Q15
  // rounding.
  let buf = solid_xv36_frame(12, 4, 0x800, 0x800, 0x800, 0);
  let src = Xv36Frame::new(&buf, 12, 4, 48);
  let mut rgb = std::vec![0u16; 12 * 4 * 3];
  let mut sink = MixedSinker::<Xv36>::new(12, 4)
    .with_rgb_u16(&mut rgb)
    .unwrap();
  xv36_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();
  for px in rgb.chunks(3) {
    assert!(
      px[0].abs_diff(0x800) <= 0x10,
      "expected ~0x800, got {:#X}",
      px[0]
    );
  }
}

#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn xv36_rgba_u16_alpha_is_max() {
  // 12-bit alpha max = 0x0FFF = 4095 (NOT 0x3FF or 0xFFFF).
  let buf = solid_xv36_frame(12, 4, 0x800, 0x800, 0x800, 0);
  let src = Xv36Frame::new(&buf, 12, 4, 48);
  let mut rgba = std::vec![0u16; 12 * 4 * 4];
  let mut sink = MixedSinker::<Xv36>::new(12, 4)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  xv36_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();
  for px in rgba.chunks(4) {
    assert_eq!(px[3], 0x0FFF, "u16 alpha must be 0x0FFF for XV36");
  }
}

#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn xv36_luma_only_extracts_y_bytes_downshifted() {
  // Y=0xF00 (12-bit value MSB-aligned: 0xF00 << 4 = 0xF000 in u16).
  // u8 luma: MSB-aligned u16 >> 8 = 0xF000 >> 8 = 0xF0 = 240.
  let buf = solid_xv36_frame(6, 8, 0x800, 0xF00, 0x800, 0);
  let src = Xv36Frame::new(&buf, 6, 8, 24);
  let mut luma = std::vec![0u8; 6 * 8];
  let mut sink = MixedSinker::<Xv36>::new(6, 8).with_luma(&mut luma).unwrap();
  xv36_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();
  // 0xF00 << 4 = 0xF000; 0xF000 >> 8 = 0xF0 = 240.
  assert!(luma.iter().all(|&y| y == 0xF0), "luma {luma:?}");
}

#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn xv36_luma_u16_only_extracts_y_native_depth() {
  // Y=0xABC (12-bit value). MSB-aligned u16 value: 0xABC << 4 = 0xABC0.
  // u16 luma_u16: MSB-aligned >> 4 = 0xABC0 >> 4 = 0xABC = 2748.
  let buf = solid_xv36_frame(6, 8, 0x800, 0xABC, 0x800, 0);
  let src = Xv36Frame::new(&buf, 6, 8, 24);
  let mut luma = std::vec![0u16; 6 * 8];
  let mut sink = MixedSinker::<Xv36>::new(6, 8)
    .with_luma_u16(&mut luma)
    .unwrap();
  xv36_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();
  // 0xABC0 >> 4 = 0xABC.
  assert!(
    luma.iter().all(|&y| y == 0xABC),
    "luma_u16 expected 0xABC, got {:?}",
    &luma[..8]
  );
}

// ---- Strategy A invariant tests ---------------------------------------

#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn xv36_with_rgb_and_with_rgba_byte_identical_u8() {
  // Strategy A invariant on u8 path — calling both `with_rgb` and
  // `with_rgba` must produce the same RGB bytes in both buffers, with
  // alpha = 0xFF in the RGBA buffer.
  let w = 12u32;
  let h = 4u32;
  let buf = solid_xv36_frame(w, h, 0x300, 0x700, 0x500, 0);
  let src = Xv36Frame::new(&buf, w, h, w * 4);
  let mut rgb = std::vec![0u8; (w * h) as usize * 3];
  let mut rgba = std::vec![0u8; (w * h) as usize * 4];
  let mut sink = MixedSinker::<Xv36>::new(w as usize, h as usize)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap();
  xv36_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();
  for i in 0..(w * h) as usize {
    assert_eq!(rgba[i * 4], rgb[i * 3], "R mismatch at pixel {i}");
    assert_eq!(rgba[i * 4 + 1], rgb[i * 3 + 1], "G mismatch at pixel {i}");
    assert_eq!(rgba[i * 4 + 2], rgb[i * 3 + 2], "B mismatch at pixel {i}");
    assert_eq!(rgba[i * 4 + 3], 0xFF, "alpha must be 0xFF at pixel {i}");
  }
}

#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn xv36_with_rgb_u16_and_with_rgba_u16_byte_identical() {
  // Strategy A invariant on u16 path — alpha must be 0x0FFF (12-bit
  // max, low-bit-packed).
  let w = 12u32;
  let h = 4u32;
  let buf = solid_xv36_frame(w, h, 0x300, 0x700, 0x500, 0);
  let src = Xv36Frame::new(&buf, w, h, w * 4);
  let mut rgb = std::vec![0u16; (w * h) as usize * 3];
  let mut rgba = std::vec![0u16; (w * h) as usize * 4];
  let mut sink = MixedSinker::<Xv36>::new(w as usize, h as usize)
    .with_rgb_u16(&mut rgb)
    .unwrap()
    .with_rgba_u16(&mut rgba)
    .unwrap();
  xv36_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();
  for i in 0..(w * h) as usize {
    assert_eq!(rgba[i * 4], rgb[i * 3], "R u16 mismatch at pixel {i}");
    assert_eq!(
      rgba[i * 4 + 1],
      rgb[i * 3 + 1],
      "G u16 mismatch at pixel {i}"
    );
    assert_eq!(
      rgba[i * 4 + 2],
      rgb[i * 3 + 2],
      "B u16 mismatch at pixel {i}"
    );
    assert_eq!(rgba[i * 4 + 3], 0x0FFF, "alpha must be 0x0FFF at pixel {i}");
  }
}

// ---- SIMD-vs-scalar parity --------------------------------------------

#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn xv36_with_simd_false_matches_with_simd_true() {
  // Pseudo-random XV36 across multiple widths covering the main loop
  // + scalar tail of every backend block size. XV36 samples are 12-bit
  // MSB-aligned in u16 (low 4 bits zero); pseudo-random fill uses the
  // low 12 bits of each channel slot, then shifts left by 4.
  for w in [1usize, 2, 4, 7, 8, 15, 16, 17, 31, 32, 33, 1920, 1922] {
    let h = 2usize;
    let mut buf = std::vec![0u16; w * h * 4];
    // Fill each u16 slot with a pseudo-random 12-bit value, MSB-aligned.
    let mut state = 0xC0FFEE_u32;
    for slot in &mut buf {
      state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
      let val = ((state >> 4) as u16) & 0xFFF0; // 12-bit, MSB-aligned
      *slot = val;
    }
    let src = Xv36Frame::new(&buf, w as u32, h as u32, (w * 4) as u32);

    let mut rgb_simd = std::vec![0u8; w * h * 3];
    let mut rgb_scalar = std::vec![0u8; w * h * 3];
    let mut sink_simd = MixedSinker::<Xv36>::new(w, h)
      .with_rgb(&mut rgb_simd)
      .unwrap();
    let mut sink_scalar = MixedSinker::<Xv36>::new(w, h)
      .with_rgb(&mut rgb_scalar)
      .unwrap()
      .with_simd(false);
    xv36_to(&src, false, ColorMatrix::Bt709, &mut sink_simd).unwrap();
    xv36_to(&src, false, ColorMatrix::Bt709, &mut sink_scalar).unwrap();
    assert_eq!(rgb_simd, rgb_scalar, "XV36 SIMD≠scalar at width {w}");
  }
}

// ---- Error-path tests --------------------------------------------------

#[test]
#[cfg(all(test, feature = "std"))]
fn xv36_zero_dim_returns_err() {
  // Sink configured for 1 row; passing row index 1 must return
  // RowIndexOutOfRange before any kernel runs.
  let mut rgb = std::vec![0u8; 4 * 3];
  let mut sink = MixedSinker::<Xv36>::new(4, 1).with_rgb(&mut rgb).unwrap();
  let packed = [0u16; 4 * 4]; // width=4, 4 u16 elements per pixel
  let row = Xv36Row::new(&packed, 1, ColorMatrix::Bt601, true);
  let err = sink.process(row).err().unwrap();
  assert!(matches!(
    err,
    MixedSinkerError::RowIndexOutOfRange {
      row: 1,
      configured_height: 1,
    }
  ));
}

#[test]
#[cfg(all(test, feature = "std"))]
fn xv36_process_rejects_short_packed_slice() {
  // 6-pixel-wide sink expects 6×4=24 u16 elements per row; a 23-element
  // slice surfaces as `RowShapeMismatch { which: Xv36Packed, .. }`.
  let mut rgb = std::vec![0u8; 6 * 3];
  let mut sink = MixedSinker::<Xv36>::new(6, 1).with_rgb(&mut rgb).unwrap();
  let packed = [0u16; 23];
  let row = Xv36Row::new(&packed, 0, ColorMatrix::Bt601, true);
  let err = sink.process(row).err().unwrap();
  assert_eq!(
    err,
    MixedSinkerError::RowShapeMismatch {
      which: RowSlice::Xv36Packed,
      row: 0,
      expected: 24,
      actual: 23,
    }
  );
}

#[test]
#[cfg(all(test, feature = "std"))]
fn xv36_buffer_too_short_for_rgba_u16_returns_err() {
  // Buffer holds 6×7 = 42 elements × 4 channels = 168 u16 elements;
  // a 6×8 frame needs 6×8×4 = 192.
  let mut rgba = std::vec![0u16; 6 * 7 * 4];
  let result = MixedSinker::<Xv36>::new(6, 8).with_rgba_u16(&mut rgba);
  let Err(err) = result else {
    panic!("expected RgbaU16BufferTooShort");
  };
  assert!(matches!(
    err,
    MixedSinkerError::RgbaU16BufferTooShort {
      expected: 192,
      actual: 168,
    }
  ));
}
