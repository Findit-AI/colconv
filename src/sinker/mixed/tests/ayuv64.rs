//! Sinker integration tests for `MixedSinker<Ayuv64>` — Ship 12d.
//!
//! Coverage:
//! 1.  `ayuv64_with_rgb_smoke` — limited-range white Y + neutral chroma → near-white u8 RGB.
//! 2.  `ayuv64_with_rgba_passes_source_alpha_depth_converted` — α u16=0xABCD → output α u8=0xAB.
//! 3.  `ayuv64_with_rgb_u16_smoke` — gray + neutral chroma → near-white u16 RGB.
//! 4.  `ayuv64_with_rgba_u16_passes_source_alpha_direct` — α u16=0xABCD → output α u16=0xABCD.
//! 5.  `ayuv64_with_luma_extracts_y_high_byte` — Y u16=0xABCD → luma u8=0xAB.
//! 6.  `ayuv64_with_luma_u16_extracts_y_native` — Y u16=0xABCD → luma u16=0xABCD.
//! 7.  `ayuv64_with_hsv_smoke` — gray → HSV with S=0.
//! 8.  `ayuv64_with_rgb_and_rgba_preserves_source_alpha` — combo u8: RGBA α matches source, RGB
//!     and RGBA's first 3 bytes match.
//! 9.  `ayuv64_with_rgb_u16_and_rgba_u16_preserves_source_alpha` — combo u16: RGBA u16 α
//!     matches source u16 direct, RGB and RGBA's first 3 u16 elements match.
//! 10. `ayuv64_simd_vs_scalar_parity_at_1922_u8` — width 1922 row, SIMD u8 == scalar.
//! 11. `ayuv64_simd_vs_scalar_parity_at_1922_u16` — width 1922 row, SIMD u16 == scalar.
//! 12. `ayuv64_width_mismatch_returns_error` — sinker width=64 receives 128-pixel row → error.
//! 13. `ayuv64_row_index_oor_returns_error` — row_idx >= height → error.
//! 14. `ayuv64_rgb_buffer_too_short_returns_error`.
//! 15. `ayuv64_rgba_buffer_too_short_returns_error`.
//! 16. `ayuv64_rgb_u16_buffer_too_short_returns_error`.
//! 17. `ayuv64_rgba_u16_buffer_too_short_returns_error`.
//! 18. `ayuv64_luma_buffer_too_short_returns_error`.
//! 19. `ayuv64_luma_u16_buffer_too_short_returns_error`.
//! 20. `ayuv64_hsv_buffer_too_short_returns_error`.
//! 21. `ayuv64_planar_parity_with_yuva444p16` — AYUV64 packed ↔ Yuva444p16 planar cross-format
//!     oracle (u8 RGB + u8 RGBA + u16 RGB + u16 RGBA byte-identical at limited range).

#[cfg(all(test, feature = "std"))]
use super::*;

// ---- AYUV64 frame builder ------------------------------------------------

/// Builds a solid-color AYUV64 plane. Each pixel is stored as four u16 words:
///   word 0: A (source alpha)
///   word 1: Y (luma, 16-bit native)
///   word 2: U (Cb chroma, 16-bit native)
///   word 3: V (Cr chroma, 16-bit native)
///
/// Row stride equals `width × 4` u16 elements (no padding).
#[cfg(all(test, feature = "std"))]
pub(super) fn solid_ayuv64_frame(
  width: u32,
  height: u32,
  a: u16,
  y: u16,
  u: u16,
  v: u16,
) -> Vec<u16> {
  let quad = [a, y, u, v];
  (0..(width as usize) * (height as usize))
    .flat_map(|_| quad)
    .collect()
}

// ---- Helper: pack Yuva444p16 planar planes into AYUV64 packed stream ------

/// Converts separate Y / U / V / A planes (16-bit, full-width 4:4:4) into a
/// packed AYUV64 u16 stream. u16 slot order per pixel: `[A, Y, U, V]`.
#[cfg(all(test, feature = "std"))]
fn pack_yuva444p16_to_ayuv64(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  a: &[u16],
  width: usize,
  height: usize,
) -> Vec<u16> {
  let mut out = vec![0u16; width * height * 4];
  for row in 0..height {
    for x in 0..width {
      let i = row * width + x;
      let off = i * 4;
      out[off] = a[i]; // slot 0 = A
      out[off + 1] = y[i]; // slot 1 = Y
      out[off + 2] = u[i]; // slot 2 = U
      out[off + 3] = v[i]; // slot 3 = V
    }
  }
  out
}

// ---- 1: RGB smoke --------------------------------------------------------

#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn ayuv64_with_rgb_smoke() {
  // Limited-range white: Y=60160 (near-white in 16-bit limited range),
  // U=V=32768 (neutral chroma midpoint). BT.709 limited range → near-white RGB.
  let buf = solid_ayuv64_frame(4, 1, 0xFFFF, 60160, 32768, 32768);
  let src = Ayuv64Frame::try_new(&buf, 4, 1, 16).unwrap();
  let mut rgb = std::vec![0u8; 4 * 3];
  let mut sink = MixedSinker::<Ayuv64>::new(4, 1).with_rgb(&mut rgb).unwrap();
  ayuv64_to(&src, false, ColorMatrix::Bt709, &mut sink).unwrap();
  for px in rgb.chunks(3) {
    assert!(
      px[0] >= 220,
      "expected near-white (>=220), got R={}, G={}, B={}",
      px[0],
      px[1],
      px[2],
    );
    // Gray neutral chroma → R == G == B.
    assert_eq!(px[0], px[1], "R != G");
    assert_eq!(px[1], px[2], "G != B");
  }
}

// ---- 2: RGBA source alpha depth-converted (u16 >> 8 → u8) ---------------

#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn ayuv64_with_rgba_passes_source_alpha_depth_converted() {
  // A u16 = 0xABCD → output u8 alpha = 0xAB (depth-converted via >> 8).
  let buf = solid_ayuv64_frame(4, 1, 0xABCD, 32768, 32768, 32768);
  let src = Ayuv64Frame::try_new(&buf, 4, 1, 16).unwrap();
  let mut rgba = std::vec![0u8; 4 * 4];
  let mut sink = MixedSinker::<Ayuv64>::new(4, 1)
    .with_rgba(&mut rgba)
    .unwrap();
  ayuv64_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();
  for px in rgba.chunks(4) {
    assert_eq!(
      px[3], 0xABu8,
      "expected alpha 0xAB (0xABCD >> 8), got {:#X}",
      px[3]
    );
  }
}

// ---- 3: RGB u16 smoke -------------------------------------------------------

#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn ayuv64_with_rgb_u16_smoke() {
  // Full-range gray midpoint: Y=U=V=32768. After YUV→RGB at 16-bit full-range
  // the per-channel u16 value should be near 32768; allow ±0x200 for Q15
  // rounding.
  let buf = solid_ayuv64_frame(4, 1, 0xFFFF, 32768, 32768, 32768);
  let src = Ayuv64Frame::try_new(&buf, 4, 1, 16).unwrap();
  let mut rgb = std::vec![0u16; 4 * 3];
  let mut sink = MixedSinker::<Ayuv64>::new(4, 1)
    .with_rgb_u16(&mut rgb)
    .unwrap();
  ayuv64_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();
  for px in rgb.chunks(3) {
    assert!(
      px[0].abs_diff(32768) <= 0x200,
      "expected ~32768, got {:#X}",
      px[0]
    );
    // Neutral chroma → R == G == B.
    assert_eq!(px[0], px[1], "R u16 != G u16");
    assert_eq!(px[1], px[2], "G u16 != B u16");
  }
}

// ---- 4: RGBA u16 passes source alpha direct (no conversion) ---------------

#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn ayuv64_with_rgba_u16_passes_source_alpha_direct() {
  // A u16 = 0xABCD → output u16 alpha = 0xABCD (direct, no conversion).
  let buf = solid_ayuv64_frame(4, 1, 0xABCD, 32768, 32768, 32768);
  let src = Ayuv64Frame::try_new(&buf, 4, 1, 16).unwrap();
  let mut rgba = std::vec![0u16; 4 * 4];
  let mut sink = MixedSinker::<Ayuv64>::new(4, 1)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  ayuv64_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();
  for px in rgba.chunks(4) {
    assert_eq!(
      px[3], 0xABCDu16,
      "expected alpha 0xABCD (direct pass-through), got {:#X}",
      px[3]
    );
  }
}

// ---- 5: Luma extracts Y high byte ----------------------------------------

#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn ayuv64_with_luma_extracts_y_high_byte() {
  // Y u16 = 0xABCD → luma u8 = 0xAB (>> 8).
  let buf = solid_ayuv64_frame(8, 2, 0xFFFF, 0xABCD, 32768, 32768);
  let src = Ayuv64Frame::try_new(&buf, 8, 2, 32).unwrap();
  let mut luma = std::vec![0u8; 8 * 2];
  let mut sink = MixedSinker::<Ayuv64>::new(8, 2)
    .with_luma(&mut luma)
    .unwrap();
  ayuv64_to(&src, false, ColorMatrix::Bt709, &mut sink).unwrap();
  assert!(
    luma.iter().all(|&y| y == 0xABu8),
    "luma expected 0xAB (0xABCD >> 8), got {:?}",
    &luma[..8]
  );
}

// ---- 6: Luma u16 extracts Y native ----------------------------------------

#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn ayuv64_with_luma_u16_extracts_y_native() {
  // Y u16 = 0xABCD → luma u16 = 0xABCD (direct, no shift).
  let buf = solid_ayuv64_frame(8, 2, 0xFFFF, 0xABCD, 32768, 32768);
  let src = Ayuv64Frame::try_new(&buf, 8, 2, 32).unwrap();
  let mut luma = std::vec![0u16; 8 * 2];
  let mut sink = MixedSinker::<Ayuv64>::new(8, 2)
    .with_luma_u16(&mut luma)
    .unwrap();
  ayuv64_to(&src, false, ColorMatrix::Bt709, &mut sink).unwrap();
  assert!(
    luma.iter().all(|&y| y == 0xABCDu16),
    "luma_u16 expected 0xABCD, got {:?}",
    &luma[..8]
  );
}

// ---- 7: HSV smoke (gray → S = 0) -----------------------------------------

#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn ayuv64_with_hsv_smoke() {
  // Full-range neutral gray → converted RGB R==G==B → HSV saturation S = 0.
  let buf = solid_ayuv64_frame(6, 2, 0xFFFF, 32768, 32768, 32768);
  let src = Ayuv64Frame::try_new(&buf, 6, 2, 24).unwrap();
  let n = 6 * 2;
  let mut h = std::vec![0u8; n];
  let mut s = std::vec![0u8; n];
  let mut v = std::vec![0u8; n];
  let mut sink = MixedSinker::<Ayuv64>::new(6, 2)
    .with_hsv(&mut h, &mut s, &mut v)
    .unwrap();
  ayuv64_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();
  for &sat in &s {
    assert_eq!(sat, 0, "gray must have S=0 in HSV");
  }
}

// ---- 8: RGB + RGBA combined path preserves source alpha (spec § 7.2) ------

#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn ayuv64_with_rgb_and_rgba_preserves_source_alpha() {
  // When BOTH with_rgb AND with_rgba are attached, AYUV64 must run
  // both independent kernel calls — RGB drops α, RGBA passes source α
  // through (depth-converted >> 8). Strategy A fan-out is NEVER used
  // for AYUV64 (per spec § 7.2).

  let width = 8usize;
  let height = 1usize;
  // Build a packed AYUV64 row with distinct source α values.
  let mut packed = std::vec![0u16; width * 4];
  for n in 0..width {
    // Distinct u16 alpha values: 0x0100, 0x0200, ... 0x0800
    let a_val = ((n as u16) + 1) << 8;
    packed[n * 4] = a_val; // A
    packed[n * 4 + 1] = 32768; // Y (mid gray)
    packed[n * 4 + 2] = 32768; // U (neutral chroma)
    packed[n * 4 + 3] = 32768; // V (neutral chroma)
  }
  let frame =
    Ayuv64Frame::try_new(&packed, width as u32, height as u32, (width * 4) as u32).unwrap();
  let mut rgb = std::vec![0u8; width * height * 3];
  let mut rgba = std::vec![0u8; width * height * 4];
  let mut sinker = MixedSinker::<Ayuv64>::new(width, height)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap();
  ayuv64_to(&frame, true, ColorMatrix::Bt709, &mut sinker).unwrap();

  for n in 0..width {
    // RGB and RGBA's first 3 bytes must be bit-identical (same packed input).
    assert_eq!(
      &rgb[n * 3..n * 3 + 3],
      &rgba[n * 4..n * 4 + 3],
      "pixel {n}: RGB and RGBA RGB-channels diverge"
    );
    // RGBA α byte = source α >> 8 (NOT 0xFF — per spec § 7.2 the direct kernel runs).
    let a_u16 = ((n as u16) + 1) << 8;
    let expected_alpha = (a_u16 >> 8) as u8;
    assert_eq!(
      rgba[n * 4 + 3],
      expected_alpha,
      "pixel {n}: source alpha was discarded (got {}, expected {})",
      rgba[n * 4 + 3],
      expected_alpha
    );
  }
}

// ---- 9: RGB u16 + RGBA u16 combined path preserves source alpha (u16 direct) --

#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn ayuv64_with_rgb_u16_and_rgba_u16_preserves_source_alpha() {
  // When both with_rgb_u16 and with_rgba_u16 are attached, AYUV64 must run
  // both independent kernel calls — RGB u16 drops α, RGBA u16 writes source
  // α direct as u16. The first 3 u16 elements of each pixel must match.

  let width = 8usize;
  let height = 1usize;
  let mut packed = std::vec![0u16; width * 4];
  for n in 0..width {
    let a_val = 0x1000u16 + (n as u16 * 0x1111); // distinct u16 alpha values
    packed[n * 4] = a_val; // A
    packed[n * 4 + 1] = 32768; // Y (mid gray)
    packed[n * 4 + 2] = 32768; // U (neutral chroma)
    packed[n * 4 + 3] = 32768; // V (neutral chroma)
  }
  let frame =
    Ayuv64Frame::try_new(&packed, width as u32, height as u32, (width * 4) as u32).unwrap();
  let mut rgb = std::vec![0u16; width * height * 3];
  let mut rgba = std::vec![0u16; width * height * 4];
  let mut sinker = MixedSinker::<Ayuv64>::new(width, height)
    .with_rgb_u16(&mut rgb)
    .unwrap()
    .with_rgba_u16(&mut rgba)
    .unwrap();
  ayuv64_to(&frame, true, ColorMatrix::Bt709, &mut sinker).unwrap();

  for n in 0..width {
    // First 3 u16 elements of RGB and RGBA must be bit-identical.
    assert_eq!(
      &rgb[n * 3..n * 3 + 3],
      &rgba[n * 4..n * 4 + 3],
      "pixel {n}: RGB u16 and RGBA u16 RGB-channels diverge"
    );
    // RGBA u16 alpha = source α written direct (no conversion).
    let expected_alpha = 0x1000u16 + (n as u16 * 0x1111);
    assert_eq!(
      rgba[n * 4 + 3],
      expected_alpha,
      "pixel {n}: source u16 alpha was not preserved (got {:#X}, expected {:#X})",
      rgba[n * 4 + 3],
      expected_alpha
    );
  }
}

// ---- 10: SIMD vs scalar parity at width 1922 (u8) -------------------------

#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn ayuv64_simd_vs_scalar_parity_at_1922_u8() {
  // Width 1922 enters and exits the main loop + scalar tail of every
  // backend block size (NEON 16, SSE4.1 16, AVX2 32, AVX-512 64, wasm 16).
  let w = 1922usize;
  let h = 2usize;
  let mut buf = std::vec![0u16; w * h * 4];
  pseudo_random_u16_low_n_bits(&mut buf, 0xBEEF_DEAD, 16);
  let src = Ayuv64Frame::try_new(&buf, w as u32, h as u32, (w * 4) as u32).unwrap();

  let mut rgb_simd = std::vec![0u8; w * h * 3];
  let mut rgb_scalar = std::vec![0u8; w * h * 3];

  let mut sink_simd = MixedSinker::<Ayuv64>::new(w, h)
    .with_rgb(&mut rgb_simd)
    .unwrap();
  ayuv64_to(&src, false, ColorMatrix::Bt709, &mut sink_simd).unwrap();

  let mut sink_scalar = MixedSinker::<Ayuv64>::new(w, h)
    .with_rgb(&mut rgb_scalar)
    .unwrap()
    .with_simd(false);
  ayuv64_to(&src, false, ColorMatrix::Bt709, &mut sink_scalar).unwrap();

  assert_eq!(
    rgb_simd, rgb_scalar,
    "AYUV64 SIMD != scalar (u8) at width {w}"
  );
}

// ---- 11: SIMD vs scalar parity at width 1922 (u16) ------------------------

#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn ayuv64_simd_vs_scalar_parity_at_1922_u16() {
  // Width 1922 u16 path — ensures i64 chroma kernel SIMD matches scalar
  // across all backend block sizes.
  let w = 1922usize;
  let h = 2usize;
  let mut buf = std::vec![0u16; w * h * 4];
  pseudo_random_u16_low_n_bits(&mut buf, 0xC0DE_FEED, 16);
  let src = Ayuv64Frame::try_new(&buf, w as u32, h as u32, (w * 4) as u32).unwrap();

  let mut rgb_simd = std::vec![0u16; w * h * 3];
  let mut rgb_scalar = std::vec![0u16; w * h * 3];

  let mut sink_simd = MixedSinker::<Ayuv64>::new(w, h)
    .with_rgb_u16(&mut rgb_simd)
    .unwrap();
  ayuv64_to(&src, false, ColorMatrix::Bt709, &mut sink_simd).unwrap();

  let mut sink_scalar = MixedSinker::<Ayuv64>::new(w, h)
    .with_rgb_u16(&mut rgb_scalar)
    .unwrap()
    .with_simd(false);
  ayuv64_to(&src, false, ColorMatrix::Bt709, &mut sink_scalar).unwrap();

  assert_eq!(
    rgb_simd, rgb_scalar,
    "AYUV64 SIMD != scalar (u16) at width {w}"
  );
}

// ---- 12: Width mismatch → RowShapeMismatch --------------------------------

#[test]
#[cfg(all(test, feature = "std"))]
fn ayuv64_width_mismatch_returns_error() {
  // Sinker built for width=64; pass a 128-pixel (512-element) packed row.
  let mut rgb = std::vec![0u8; 64 * 3];
  let mut sink = MixedSinker::<Ayuv64>::new(64, 1)
    .with_rgb(&mut rgb)
    .unwrap();
  // Packed slice for width=128: 128 × 4 = 512 u16 elements.
  let packed = std::vec![0u16; 512];
  let row = Ayuv64Row::new(&packed, 0, ColorMatrix::Bt709, false);
  let err = sink.process(row).err().unwrap();
  assert_eq!(
    err,
    MixedSinkerError::RowShapeMismatch {
      which: RowSlice::Ayuv64Packed,
      row: 0,
      expected: 64 * 4, // 256
      actual: 512,
    }
  );
}

// ---- 13: Row index out of range -------------------------------------------

#[test]
#[cfg(all(test, feature = "std"))]
fn ayuv64_row_index_oor_returns_error() {
  // Sinker built for height=2; pass row index 2 (== height → out of range).
  let mut rgb = std::vec![0u8; 4 * 2 * 3];
  let mut sink = MixedSinker::<Ayuv64>::new(4, 2).with_rgb(&mut rgb).unwrap();
  let packed = std::vec![0u16; 4 * 4]; // width=4, 4 u16 elements per pixel
  let row = Ayuv64Row::new(&packed, 2, ColorMatrix::Bt709, false);
  let err = sink.process(row).err().unwrap();
  assert!(matches!(
    err,
    MixedSinkerError::RowIndexOutOfRange {
      row: 2,
      configured_height: 2,
    }
  ));
}

// ---- 14: RGB buffer too short ----------------------------------------------

#[test]
#[cfg(all(test, feature = "std"))]
fn ayuv64_rgb_buffer_too_short_returns_error() {
  // 8×4 frame needs 8 × 4 × 3 = 96 bytes; supply only 95.
  let mut rgb = std::vec![0u8; 95];
  let result = MixedSinker::<Ayuv64>::new(8, 4).with_rgb(&mut rgb);
  let Err(err) = result else {
    panic!("expected RgbBufferTooShort");
  };
  assert!(matches!(
    err,
    MixedSinkerError::RgbBufferTooShort {
      expected: 96,
      actual: 95,
    }
  ));
}

// ---- 15: RGBA buffer too short ---------------------------------------------

#[test]
#[cfg(all(test, feature = "std"))]
fn ayuv64_rgba_buffer_too_short_returns_error() {
  // 6×4 frame needs 6 × 4 × 4 = 96 bytes; supply only 90.
  let mut rgba = std::vec![0u8; 90];
  let result = MixedSinker::<Ayuv64>::new(6, 4).with_rgba(&mut rgba);
  let Err(err) = result else {
    panic!("expected RgbaBufferTooShort");
  };
  assert!(matches!(
    err,
    MixedSinkerError::RgbaBufferTooShort {
      expected: 96,
      actual: 90,
    }
  ));
}

// ---- 16: RGB u16 buffer too short ------------------------------------------

#[test]
#[cfg(all(test, feature = "std"))]
fn ayuv64_rgb_u16_buffer_too_short_returns_error() {
  // 8×4 frame needs 8 × 4 × 3 = 96 u16 elements; supply only 90.
  let mut rgb = std::vec![0u16; 90];
  let result = MixedSinker::<Ayuv64>::new(8, 4).with_rgb_u16(&mut rgb);
  let Err(err) = result else {
    panic!("expected RgbU16BufferTooShort");
  };
  assert!(matches!(
    err,
    MixedSinkerError::RgbU16BufferTooShort {
      expected: 96,
      actual: 90,
    }
  ));
}

// ---- 17: RGBA u16 buffer too short -----------------------------------------

#[test]
#[cfg(all(test, feature = "std"))]
fn ayuv64_rgba_u16_buffer_too_short_returns_error() {
  // 6×4 frame needs 6 × 4 × 4 = 96 u16 elements; supply only 88.
  let mut rgba = std::vec![0u16; 88];
  let result = MixedSinker::<Ayuv64>::new(6, 4).with_rgba_u16(&mut rgba);
  let Err(err) = result else {
    panic!("expected RgbaU16BufferTooShort");
  };
  assert!(matches!(
    err,
    MixedSinkerError::RgbaU16BufferTooShort {
      expected: 96,
      actual: 88,
    }
  ));
}

// ---- 18: Luma buffer too short ---------------------------------------------

#[test]
#[cfg(all(test, feature = "std"))]
fn ayuv64_luma_buffer_too_short_returns_error() {
  // 8×3 frame needs 8 × 3 = 24 bytes; supply 20.
  let mut luma = std::vec![0u8; 20];
  let result = MixedSinker::<Ayuv64>::new(8, 3).with_luma(&mut luma);
  let Err(err) = result else {
    panic!("expected LumaBufferTooShort");
  };
  assert!(matches!(
    err,
    MixedSinkerError::LumaBufferTooShort {
      expected: 24,
      actual: 20,
    }
  ));
}

// ---- 19: Luma u16 buffer too short -----------------------------------------

#[test]
#[cfg(all(test, feature = "std"))]
fn ayuv64_luma_u16_buffer_too_short_returns_error() {
  // 8×3 frame needs 8 × 3 = 24 u16 elements; supply 20.
  let mut luma = std::vec![0u16; 20];
  let result = MixedSinker::<Ayuv64>::new(8, 3).with_luma_u16(&mut luma);
  let Err(err) = result else {
    panic!("expected LumaU16BufferTooShort");
  };
  assert!(matches!(
    err,
    MixedSinkerError::LumaU16BufferTooShort {
      expected: 24,
      actual: 20,
    }
  ));
}

// ---- 20: HSV buffer too short (H plane) ------------------------------------

#[test]
#[cfg(all(test, feature = "std"))]
fn ayuv64_hsv_buffer_too_short_returns_error() {
  // 4×4 frame needs 16 bytes per HSV plane; supply H with only 15.
  let mut h = std::vec![0u8; 15];
  let mut s = std::vec![0u8; 16];
  let mut v = std::vec![0u8; 16];
  let result = MixedSinker::<Ayuv64>::new(4, 4).with_hsv(&mut h, &mut s, &mut v);
  let Err(err) = result else {
    panic!("expected HsvPlaneTooShort");
  };
  assert!(matches!(
    err,
    MixedSinkerError::HsvPlaneTooShort {
      which: HsvPlane::H,
      expected: 16,
      actual: 15,
    }
  ));
}

// ---- 21: Planar parity with Yuva444p16 (headline cross-format oracle) -----

#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn ayuv64_planar_parity_with_yuva444p16() {
  // Spec § 3.5 cross-format invariant: Yuva444p16 planar and AYUV64 packed
  // carry identical logical YUVA 16-bit samples. Both sinks run with
  // `with_rgb`, `with_rgba`, `with_rgb_u16`, and `with_rgba_u16` attached.
  //
  // Range semantics:
  //   At LIMITED range (full_range=false), both the Yuva444p16 sinker and
  //   the AYUV64 sinker use `range_params_n::<16, BITS_OUT>` — the same
  //   constants for both formats at 16-bit source depth. This means
  //   u8 RGB / RGBA and u16 RGB / RGBA output must be BYTE-IDENTICAL
  //   between the two formats (unlike the VUYA ↔ Yuva444p parity test
  //   which required full_range=true to achieve bit-identity due to
  //   divergent 8-bit constant tables).
  //
  // Alpha semantics:
  //   - AYUV64 with_rgba → ayuv64_to_rgba_row: source α >> 8 → u8.
  //   - Yuva444p16 with_rgba → yuva444p16_to_rgba_row: source α >> 8 → u8.
  //   Both formats depth-convert via >> 8, so RGBA u8 alpha is byte-identical.
  //   - AYUV64 with_rgba_u16 → ayuv64_to_rgba_u16_row: source α direct.
  //   - Yuva444p16 with_rgba_u16 → yuva444p16_to_rgba_u16_row: source α direct.
  //   Both formats write α direct as u16, so RGBA u16 alpha is byte-identical.
  //
  // Use width=64 × height=4 (covers SIMD main loop + scalar tail).

  let width = 64usize;
  let height = 4usize;
  let n = width * height;

  // Build pseudo-random Y / U / V / A 16-bit planes.
  let mut yp = std::vec![0u16; n];
  let mut up = std::vec![0u16; n];
  let mut vp = std::vec![0u16; n];
  let mut ap = std::vec![0u16; n];
  pseudo_random_u16_low_n_bits(&mut yp, 0xC0FFEE_u32, 16);
  pseudo_random_u16_low_n_bits(&mut up, 0xBADF00D_u32, 16);
  pseudo_random_u16_low_n_bits(&mut vp, 0xFEEDFACE_u32, 16);
  pseudo_random_u16_low_n_bits(&mut ap, 0xA1FA5EED_u32, 16);

  // Construct the Yuva444p16 planar frame.
  let planar = Yuva444p16Frame::try_new(
    &yp,
    &up,
    &vp,
    &ap,
    width as u32,
    height as u32,
    width as u32,
    width as u32,
    width as u32,
    width as u32,
  )
  .unwrap();

  // Pack the same samples into an AYUV64 u16 stream.
  let ayuv64_buf = pack_yuva444p16_to_ayuv64(&yp, &up, &vp, &ap, width, height);
  let packed_frame =
    Ayuv64Frame::try_new(&ayuv64_buf, width as u32, height as u32, (width * 4) as u32).unwrap();

  // Use limited range (full_range=false) — both kernels use range_params_n::<16, BITS_OUT>
  // yielding byte-identical output without any full-range workaround.
  let full_range = false;

  // --- Part 1: u8 RGB parity (`with_rgb` only — no RGBA, no alpha divergence) ---
  let mut p_rgb = std::vec![0u8; n * 3];
  let mut a_rgb = std::vec![0u8; n * 3];

  let mut p_sink = MixedSinker::<Yuva444p16>::new(width, height)
    .with_rgb(&mut p_rgb)
    .unwrap();
  yuva444p16_to(&planar, full_range, ColorMatrix::Bt709, &mut p_sink).unwrap();

  let mut a_sink = MixedSinker::<Ayuv64>::new(width, height)
    .with_rgb(&mut a_rgb)
    .unwrap();
  ayuv64_to(&packed_frame, full_range, ColorMatrix::Bt709, &mut a_sink).unwrap();

  assert_eq!(
    p_rgb, a_rgb,
    "AYUV64 <-> Yuva444p16 u8 RGB diverges at limited range"
  );

  // --- Part 2: u8 RGBA source-alpha parity (standalone RGBA path) ---
  // Both formats run standalone RGBA with source-alpha depth-converted >> 8.
  let mut p_rgba = std::vec![0u8; n * 4];
  let mut a_rgba = std::vec![0u8; n * 4];

  let mut p_sink2 = MixedSinker::<Yuva444p16>::new(width, height)
    .with_rgba(&mut p_rgba)
    .unwrap();
  yuva444p16_to(&planar, full_range, ColorMatrix::Bt709, &mut p_sink2).unwrap();

  let mut a_sink2 = MixedSinker::<Ayuv64>::new(width, height)
    .with_rgba(&mut a_rgba)
    .unwrap();
  ayuv64_to(&packed_frame, full_range, ColorMatrix::Bt709, &mut a_sink2).unwrap();

  assert_eq!(
    p_rgba, a_rgba,
    "AYUV64 <-> Yuva444p16 u8 RGBA diverges at limited range (source-alpha path)"
  );

  // Spot-check: RGBA u8 alpha bytes equal source alpha >> 8.
  for (i, &src_a) in ap.iter().enumerate() {
    let expected_alpha = (src_a >> 8) as u8;
    let ayuv64_alpha = a_rgba[i * 4 + 3];
    assert_eq!(
      ayuv64_alpha, expected_alpha,
      "AYUV64 u8 RGBA alpha at pixel {i}: expected {expected_alpha:#X} (src {src_a:#X} >> 8), got {ayuv64_alpha:#X}"
    );
  }

  // --- Part 3: u16 RGB parity ---
  let mut p_rgb_u16 = std::vec![0u16; n * 3];
  let mut a_rgb_u16 = std::vec![0u16; n * 3];

  let mut p_sink3 = MixedSinker::<Yuva444p16>::new(width, height)
    .with_rgb_u16(&mut p_rgb_u16)
    .unwrap();
  yuva444p16_to(&planar, full_range, ColorMatrix::Bt709, &mut p_sink3).unwrap();

  let mut a_sink3 = MixedSinker::<Ayuv64>::new(width, height)
    .with_rgb_u16(&mut a_rgb_u16)
    .unwrap();
  ayuv64_to(&packed_frame, full_range, ColorMatrix::Bt709, &mut a_sink3).unwrap();

  assert_eq!(
    p_rgb_u16, a_rgb_u16,
    "AYUV64 <-> Yuva444p16 u16 RGB diverges at limited range"
  );

  // --- Part 4: u16 RGBA source-alpha parity (headline u16 assertion) ---
  // Both formats write α direct as u16 — no conversion, byte-identical.
  let mut p_rgba_u16 = std::vec![0u16; n * 4];
  let mut a_rgba_u16 = std::vec![0u16; n * 4];

  let mut p_sink4 = MixedSinker::<Yuva444p16>::new(width, height)
    .with_rgba_u16(&mut p_rgba_u16)
    .unwrap();
  yuva444p16_to(&planar, full_range, ColorMatrix::Bt709, &mut p_sink4).unwrap();

  let mut a_sink4 = MixedSinker::<Ayuv64>::new(width, height)
    .with_rgba_u16(&mut a_rgba_u16)
    .unwrap();
  ayuv64_to(&packed_frame, full_range, ColorMatrix::Bt709, &mut a_sink4).unwrap();

  assert_eq!(
    p_rgba_u16, a_rgba_u16,
    "AYUV64 <-> Yuva444p16 u16 RGBA diverges at limited range (headline u16 parity)"
  );

  // Spot-check: RGBA u16 alpha bytes equal source alpha (direct pass-through).
  for (i, &src_a) in ap.iter().enumerate() {
    let ayuv64_alpha = a_rgba_u16[i * 4 + 3];
    assert_eq!(
      ayuv64_alpha, src_a,
      "AYUV64 u16 RGBA alpha at pixel {i}: expected {src_a:#X}, got {ayuv64_alpha:#X}"
    );
  }
}
