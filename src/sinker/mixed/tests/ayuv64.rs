//! Sinker integration tests for `MixedSinker<Ayuv64>` — Ship 12d.
//!
//! Coverage:
//! 1.  `ayuv64_with_rgb_smoke` — limited-range white Y + neutral chroma → near-white u8 RGB.
//! 2.  `ayuv64_with_rgba_passes_source_alpha_depth_converted` — α u16=0xABCD → output α u8=0xAB.
//!     (items 3–21 are existing tests)
//! 22. `ayuv64_strategy_a_plus_matches_independent_kernel` — u8 A+ correctness.
//! 23. `ayuv64_strategy_a_plus_u16_matches_independent_kernel` — u16 A+ correctness.
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
/// Row stride equals `width x 4` u16 elements (no padding).
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
///
/// Gated on `yuva`: the cross-format oracle below feeds the planar
/// `Yuva444p16` source, absent in a `yuv-444-packed`-solo build.
#[cfg(all(test, feature = "std", feature = "yuva"))]
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
  ayuv64_to(&src, false, KernelMatrix::Bt709, &mut sink).unwrap();
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
  ayuv64_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
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
  ayuv64_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
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
  ayuv64_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
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
  ayuv64_to(&src, false, KernelMatrix::Bt709, &mut sink).unwrap();
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
  ayuv64_to(&src, false, KernelMatrix::Bt709, &mut sink).unwrap();
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
  ayuv64_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
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
  ayuv64_to(&frame, true, KernelMatrix::Bt709, &mut sinker).unwrap();

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
  ayuv64_to(&frame, true, KernelMatrix::Bt709, &mut sinker).unwrap();

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
  ayuv64_to(&src, false, KernelMatrix::Bt709, &mut sink_simd).unwrap();

  let mut sink_scalar = MixedSinker::<Ayuv64>::new(w, h)
    .with_rgb(&mut rgb_scalar)
    .unwrap()
    .with_simd(false);
  ayuv64_to(&src, false, KernelMatrix::Bt709, &mut sink_scalar).unwrap();

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
  ayuv64_to(&src, false, KernelMatrix::Bt709, &mut sink_simd).unwrap();

  let mut sink_scalar = MixedSinker::<Ayuv64>::new(w, h)
    .with_rgb_u16(&mut rgb_scalar)
    .unwrap()
    .with_simd(false);
  ayuv64_to(&src, false, KernelMatrix::Bt709, &mut sink_scalar).unwrap();

  assert_eq!(
    rgb_simd, rgb_scalar,
    "AYUV64 SIMD != scalar (u16) at width {w}"
  );
}

// ---- 14: RGB buffer too short ----------------------------------------------

#[test]
#[cfg(all(test, feature = "std"))]
fn ayuv64_rgb_buffer_too_short_returns_error() {
  // 8x4 frame needs 8 x 4 x 3 = 96 bytes; supply only 95.
  let mut rgb = std::vec![0u8; 95];
  let result = MixedSinker::<Ayuv64>::new(8, 4).with_rgb(&mut rgb);
  let Err(err) = result else {
    panic!("expected InsufficientRgbBuffer");
  };
  assert_eq!(
    err,
    MixedSinkerError::InsufficientRgbBuffer(InsufficientBuffer::new(96, 95))
  );
}

// ---- 15: RGBA buffer too short ---------------------------------------------

#[test]
#[cfg(all(test, feature = "std"))]
fn ayuv64_rgba_buffer_too_short_returns_error() {
  // 6x4 frame needs 6 x 4 x 4 = 96 bytes; supply only 90.
  let mut rgba = std::vec![0u8; 90];
  let result = MixedSinker::<Ayuv64>::new(6, 4).with_rgba(&mut rgba);
  let Err(err) = result else {
    panic!("expected InsufficientRgbaBuffer");
  };
  assert_eq!(
    err,
    MixedSinkerError::InsufficientRgbaBuffer(InsufficientBuffer::new(96, 90))
  );
}

// ---- 16: RGB u16 buffer too short ------------------------------------------

#[test]
#[cfg(all(test, feature = "std"))]
fn ayuv64_rgb_u16_buffer_too_short_returns_error() {
  // 8x4 frame needs 8 x 4 x 3 = 96 u16 elements; supply only 90.
  let mut rgb = std::vec![0u16; 90];
  let result = MixedSinker::<Ayuv64>::new(8, 4).with_rgb_u16(&mut rgb);
  let Err(err) = result else {
    panic!("expected InsufficientRgbU16Buffer");
  };
  assert_eq!(
    err,
    MixedSinkerError::InsufficientRgbU16Buffer(InsufficientBuffer::new(96, 90))
  );
}

// ---- 17: RGBA u16 buffer too short -----------------------------------------

#[test]
#[cfg(all(test, feature = "std"))]
fn ayuv64_rgba_u16_buffer_too_short_returns_error() {
  // 6x4 frame needs 6 x 4 x 4 = 96 u16 elements; supply only 88.
  let mut rgba = std::vec![0u16; 88];
  let result = MixedSinker::<Ayuv64>::new(6, 4).with_rgba_u16(&mut rgba);
  let Err(err) = result else {
    panic!("expected InsufficientRgbaU16Buffer");
  };
  assert_eq!(
    err,
    MixedSinkerError::InsufficientRgbaU16Buffer(InsufficientBuffer::new(96, 88))
  );
}

// ---- 18: Luma buffer too short ---------------------------------------------

#[test]
#[cfg(all(test, feature = "std"))]
fn ayuv64_luma_buffer_too_short_returns_error() {
  // 8x3 frame needs 8 x 3 = 24 bytes; supply 20.
  let mut luma = std::vec![0u8; 20];
  let result = MixedSinker::<Ayuv64>::new(8, 3).with_luma(&mut luma);
  let Err(err) = result else {
    panic!("expected InsufficientLumaBuffer");
  };
  assert_eq!(
    err,
    MixedSinkerError::InsufficientLumaBuffer(InsufficientBuffer::new(24, 20))
  );
}

// ---- 19: Luma u16 buffer too short -----------------------------------------

#[test]
#[cfg(all(test, feature = "std"))]
fn ayuv64_luma_u16_buffer_too_short_returns_error() {
  // 8x3 frame needs 8 x 3 = 24 u16 elements; supply 20.
  let mut luma = std::vec![0u16; 20];
  let result = MixedSinker::<Ayuv64>::new(8, 3).with_luma_u16(&mut luma);
  let Err(err) = result else {
    panic!("expected InsufficientLumaU16Buffer");
  };
  assert_eq!(
    err,
    MixedSinkerError::InsufficientLumaU16Buffer(InsufficientBuffer::new(24, 20))
  );
}

// ---- 20: HSV buffer too short (H plane) ------------------------------------

#[test]
#[cfg(all(test, feature = "std"))]
fn ayuv64_hsv_buffer_too_short_returns_error() {
  // 4x4 frame needs 16 bytes per HSV plane; supply H with only 15.
  let mut h = std::vec![0u8; 15];
  let mut s = std::vec![0u8; 16];
  let mut v = std::vec![0u8; 16];
  let result = MixedSinker::<Ayuv64>::new(4, 4).with_hsv(&mut h, &mut s, &mut v);
  let Err(err) = result else {
    panic!("expected InsufficientHsvPlane");
  };
  assert!(matches!(
    &err,
    MixedSinkerError::InsufficientHsvPlane(e)
      if matches!(e.which(), HsvPlane::H) && e.expected() == 16 && e.actual() == 15
  ));
}

// ---- 21: Planar parity with Yuva444p16 (headline cross-format oracle) -----
//
// Gated on `yuva` so a `yuv-444-packed`-solo `--tests` build (which lacks
// the planar `Yuva444p16` frame / walker) still compiles.
#[test]
#[cfg(all(test, feature = "std", feature = "yuva"))]
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
  //   between the two formats. (The analogous VUYA ↔ Yuva444p 8-bit parity
  //   test also uses limited range after the v0.17.0 range_params_n::<8, 8>
  //   migration which unified 8-bit constants.)
  //
  // Alpha semantics:
  //   - AYUV64 with_rgba → ayuv64_to_rgba_row: source α >> 8 → u8.
  //   - Yuva444p16 with_rgba → yuva444p16_to_rgba_row: source α >> 8 → u8.
  //   Both formats depth-convert via >> 8, so RGBA u8 alpha is byte-identical.
  //   - AYUV64 with_rgba_u16 → ayuv64_to_rgba_u16_row: source α direct.
  //   - Yuva444p16 with_rgba_u16 → yuva444p16_to_rgba_u16_row: source α direct.
  //   Both formats write α direct as u16, so RGBA u16 alpha is byte-identical.
  //
  // Use width=64 x height=4 (covers SIMD main loop + scalar tail).

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
  yuva444p16_to(&planar, full_range, KernelMatrix::Bt709, &mut p_sink).unwrap();

  let mut a_sink = MixedSinker::<Ayuv64>::new(width, height)
    .with_rgb(&mut a_rgb)
    .unwrap();
  ayuv64_to(&packed_frame, full_range, KernelMatrix::Bt709, &mut a_sink).unwrap();

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
  yuva444p16_to(&planar, full_range, KernelMatrix::Bt709, &mut p_sink2).unwrap();

  let mut a_sink2 = MixedSinker::<Ayuv64>::new(width, height)
    .with_rgba(&mut a_rgba)
    .unwrap();
  ayuv64_to(&packed_frame, full_range, KernelMatrix::Bt709, &mut a_sink2).unwrap();

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
  yuva444p16_to(&planar, full_range, KernelMatrix::Bt709, &mut p_sink3).unwrap();

  let mut a_sink3 = MixedSinker::<Ayuv64>::new(width, height)
    .with_rgb_u16(&mut a_rgb_u16)
    .unwrap();
  ayuv64_to(&packed_frame, full_range, KernelMatrix::Bt709, &mut a_sink3).unwrap();

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
  yuva444p16_to(&planar, full_range, KernelMatrix::Bt709, &mut p_sink4).unwrap();

  let mut a_sink4 = MixedSinker::<Ayuv64>::new(width, height)
    .with_rgba_u16(&mut a_rgba_u16)
    .unwrap();
  ayuv64_to(&packed_frame, full_range, KernelMatrix::Bt709, &mut a_sink4).unwrap();

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

// ---- 22: Strategy A+ correctness u8 (spec § 6.1) -------------------------

/// Strategy A+ correctness: u8 combo path output == scalar inline-α kernel
/// at all (range, matrix) combinations. See spec § 6.1.
#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn ayuv64_strategy_a_plus_matches_independent_kernel() {
  let width = 128usize;
  let height = 4usize;

  let mut packed = std::vec![0u16; width * height * 4];
  pseudo_random_u16_low_n_bits(&mut packed, 0xC0FFEE, 16);
  let frame =
    Ayuv64Frame::try_new(&packed, width as u32, height as u32, (width * 4) as u32).unwrap();

  for full_range in [true, false] {
    for matrix in [
      KernelMatrix::Bt601,
      KernelMatrix::Bt709,
      KernelMatrix::Bt2020Ncl,
      KernelMatrix::Smpte240m,
      KernelMatrix::Fcc,
      KernelMatrix::YCgCo,
    ] {
      // Sinker combo path (A+).
      let mut sinker_rgb = std::vec![0u8; width * height * 3];
      let mut sinker_rgba = std::vec![0u8; width * height * 4];
      {
        let mut sink = MixedSinker::<Ayuv64>::new(width, height)
          .with_rgb(&mut sinker_rgb)
          .unwrap()
          .with_rgba(&mut sinker_rgba)
          .unwrap();
        ayuv64_to(&frame, full_range, matrix, &mut sink).unwrap();
      }

      // Reference: scalar inline-α kernel per row.
      let mut inline_rgb = std::vec![0u8; width * height * 3];
      let mut inline_rgba = std::vec![0u8; width * height * 4];
      for r in 0..height {
        let row_off_packed = r * width * 4;
        let row_off_rgb = r * width * 3;
        let row_off_rgba = r * width * 4;
        crate::row::scalar::ayuv64_to_rgb_row::<false>(
          &packed[row_off_packed..row_off_packed + width * 4],
          &mut inline_rgb[row_off_rgb..row_off_rgb + width * 3],
          width,
          matrix,
          full_range,
        );
        crate::row::scalar::ayuv64_to_rgba_row::<false>(
          &packed[row_off_packed..row_off_packed + width * 4],
          &mut inline_rgba[row_off_rgba..row_off_rgba + width * 4],
          width,
          matrix,
          full_range,
        );
      }

      assert_eq!(
        sinker_rgb, inline_rgb,
        "AYUV64 A+ u8 RGB diverges (range={full_range}, matrix={matrix:?})"
      );
      assert_eq!(
        sinker_rgba, inline_rgba,
        "AYUV64 A+ u8 RGBA diverges from scalar inline-α (range={full_range}, matrix={matrix:?})"
      );
    }
  }
}

// ---- 23: Strategy A+ correctness u16 (spec § 6.1) ------------------------

/// Strategy A+ correctness: u16 combo path output == scalar inline-α kernel
/// at all (range, matrix) combinations. See spec § 6.1.
#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn ayuv64_strategy_a_plus_u16_matches_independent_kernel() {
  let width = 128usize;
  let height = 4usize;

  let mut packed = std::vec![0u16; width * height * 4];
  pseudo_random_u16_low_n_bits(&mut packed, 0xDEADBEEF, 16);
  let frame =
    Ayuv64Frame::try_new(&packed, width as u32, height as u32, (width * 4) as u32).unwrap();

  for full_range in [true, false] {
    for matrix in [
      KernelMatrix::Bt601,
      KernelMatrix::Bt709,
      KernelMatrix::Bt2020Ncl,
      KernelMatrix::Smpte240m,
      KernelMatrix::Fcc,
      KernelMatrix::YCgCo,
    ] {
      // Sinker combo path (A+): rgb_u16 + rgba_u16.
      let mut sinker_rgb = std::vec![0u16; width * height * 3];
      let mut sinker_rgba = std::vec![0u16; width * height * 4];
      {
        let mut sink = MixedSinker::<Ayuv64>::new(width, height)
          .with_rgb_u16(&mut sinker_rgb)
          .unwrap()
          .with_rgba_u16(&mut sinker_rgba)
          .unwrap();
        ayuv64_to(&frame, full_range, matrix, &mut sink).unwrap();
      }

      // Reference: scalar inline-α kernel per row.
      let mut inline_rgb = std::vec![0u16; width * height * 3];
      let mut inline_rgba = std::vec![0u16; width * height * 4];
      for r in 0..height {
        let row_off_packed = r * width * 4;
        let row_off_rgb = r * width * 3;
        let row_off_rgba = r * width * 4;
        crate::row::scalar::ayuv64_to_rgb_u16_row::<false>(
          &packed[row_off_packed..row_off_packed + width * 4],
          &mut inline_rgb[row_off_rgb..row_off_rgb + width * 3],
          width,
          matrix,
          full_range,
        );
        crate::row::scalar::ayuv64_to_rgba_u16_row::<false>(
          &packed[row_off_packed..row_off_packed + width * 4],
          &mut inline_rgba[row_off_rgba..row_off_rgba + width * 4],
          width,
          matrix,
          full_range,
        );
      }

      assert_eq!(
        sinker_rgb, inline_rgb,
        "AYUV64 A+ u16 RGB diverges (range={full_range}, matrix={matrix:?})"
      );
      assert_eq!(
        sinker_rgba, inline_rgba,
        "AYUV64 A+ u16 RGBA diverges from inline-α (range={full_range}, matrix={matrix:?})"
      );
    }
  }
}
// Phase 4 Tier 5 — Frame BE flag, AYUV64 LE+BE round-trip parity tests.
//
// AYUV64 packs each pixel as four u16 channels (`A, Y, U, V`), 16-bit native.
// The BE wire variant byte-swaps each u16 channel before extraction. Encoding
// the same logical samples as LE bytes vs BE bytes and feeding through
// `MixedSinker<Ayuv64<false>>` vs `MixedSinker<Ayuv64<true>>` must produce
// byte-identical output for both u8 and u16 RGBA paths (also exercising the
// alpha_extract `<BE>` propagation in the Strategy A+ combo path).
#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn ayuv64_le_be_roundtrip_byte_identical() {
  // Build a logical (A, Y, U, V) u16 quadruple stream with non-trivial values.
  let logical: std::vec::Vec<u16> = (0..8 * 4 * 4)
    .map(|i| match i % 4 {
      0 => 0xABCDu16, // A — real source alpha
      1 => 0x8000u16, // Y
      2 => 0x4000u16, // U
      _ => 0xC000u16, // V
    })
    .collect();
  let pix_le: std::vec::Vec<u16> = logical.iter().map(|&v| as_le_u16(v)).collect();
  let pix_be: std::vec::Vec<u16> = logical.iter().map(|&v| as_be_u16(v)).collect();

  // Standalone u8 RGBA path (`ayuv64_to_rgba_row` directly).
  let frame_le = Ayuv64LeFrame::try_new(&pix_le, 8, 4, 8 * 4).unwrap();
  let mut out_le_rgba = std::vec![0u8; 8 * 4 * 4];
  let mut sink_le = MixedSinker::<Ayuv64>::new(8, 4)
    .with_simd(false)
    .with_rgba(&mut out_le_rgba)
    .unwrap();
  ayuv64_to(&frame_le, true, KernelMatrix::Bt709, &mut sink_le).unwrap();

  let frame_be = Ayuv64BeFrame::try_new(&pix_be, 8, 4, 8 * 4).unwrap();
  let mut out_be_rgba = std::vec![0u8; 8 * 4 * 4];
  let mut sink_be = MixedSinker::<Ayuv64<true>>::new(8, 4)
    .with_simd(false)
    .with_rgba(&mut out_be_rgba)
    .unwrap();
  ayuv64_to_endian(&frame_be, true, KernelMatrix::Bt709, &mut sink_be).unwrap();

  assert_eq!(
    out_le_rgba, out_be_rgba,
    "AYUV64 RGBA u8 LE/BE outputs diverge — `<const BE>` propagation broken"
  );

  // Standalone u16 RGBA path (`ayuv64_to_rgba_u16_row` directly).
  let mut out_le_rgba_u16 = std::vec![0u16; 8 * 4 * 4];
  let mut sink_le_u16 = MixedSinker::<Ayuv64>::new(8, 4)
    .with_simd(false)
    .with_rgba_u16(&mut out_le_rgba_u16)
    .unwrap();
  ayuv64_to(&frame_le, true, KernelMatrix::Bt709, &mut sink_le_u16).unwrap();

  let mut out_be_rgba_u16 = std::vec![0u16; 8 * 4 * 4];
  let mut sink_be_u16 = MixedSinker::<Ayuv64<true>>::new(8, 4)
    .with_simd(false)
    .with_rgba_u16(&mut out_be_rgba_u16)
    .unwrap();
  ayuv64_to_endian(&frame_be, true, KernelMatrix::Bt709, &mut sink_be_u16).unwrap();

  assert_eq!(
    out_le_rgba_u16, out_be_rgba_u16,
    "AYUV64 RGBA u16 LE/BE outputs diverge — `<const BE>` propagation broken"
  );
}

#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn ayuv64_le_be_roundtrip_strategy_a_plus_byte_identical() {
  // Strategy A+ combo path: with_rgb + with_rgba (and the u16 analog).
  // Exercises `expand_rgb_to_rgba_row` + `alpha_extract::copy_alpha_*::<BE>`
  // — without `<BE>` propagation through alpha_extract, the BE output would
  // diverge.
  let logical: std::vec::Vec<u16> = (0..8 * 4 * 4)
    .map(|i| match i % 4 {
      0 => 0x55AAu16, // A
      1 => 0x8000u16, // Y
      2 => 0x4000u16, // U
      _ => 0xC000u16, // V
    })
    .collect();
  let pix_le: std::vec::Vec<u16> = logical.iter().map(|&v| as_le_u16(v)).collect();
  let pix_be: std::vec::Vec<u16> = logical.iter().map(|&v| as_be_u16(v)).collect();

  let frame_le = Ayuv64LeFrame::try_new(&pix_le, 8, 4, 8 * 4).unwrap();
  let mut out_le_rgb = std::vec![0u8; 8 * 4 * 3];
  let mut out_le_rgba = std::vec![0u8; 8 * 4 * 4];
  let mut out_le_rgb_u16 = std::vec![0u16; 8 * 4 * 3];
  let mut out_le_rgba_u16 = std::vec![0u16; 8 * 4 * 4];
  let mut sink_le = MixedSinker::<Ayuv64>::new(8, 4)
    .with_simd(false)
    .with_rgb(&mut out_le_rgb)
    .unwrap()
    .with_rgba(&mut out_le_rgba)
    .unwrap()
    .with_rgb_u16(&mut out_le_rgb_u16)
    .unwrap()
    .with_rgba_u16(&mut out_le_rgba_u16)
    .unwrap();
  ayuv64_to(&frame_le, true, KernelMatrix::Bt709, &mut sink_le).unwrap();

  let frame_be = Ayuv64BeFrame::try_new(&pix_be, 8, 4, 8 * 4).unwrap();
  let mut out_be_rgb = std::vec![0u8; 8 * 4 * 3];
  let mut out_be_rgba = std::vec![0u8; 8 * 4 * 4];
  let mut out_be_rgb_u16 = std::vec![0u16; 8 * 4 * 3];
  let mut out_be_rgba_u16 = std::vec![0u16; 8 * 4 * 4];
  let mut sink_be = MixedSinker::<Ayuv64<true>>::new(8, 4)
    .with_simd(false)
    .with_rgb(&mut out_be_rgb)
    .unwrap()
    .with_rgba(&mut out_be_rgba)
    .unwrap()
    .with_rgb_u16(&mut out_be_rgb_u16)
    .unwrap()
    .with_rgba_u16(&mut out_be_rgba_u16)
    .unwrap();
  ayuv64_to_endian(&frame_be, true, KernelMatrix::Bt709, &mut sink_be).unwrap();

  assert_eq!(out_le_rgb, out_be_rgb, "AYUV64 A+ RGB u8 LE/BE diverge");
  assert_eq!(out_le_rgba, out_be_rgba, "AYUV64 A+ RGBA u8 LE/BE diverge");
  assert_eq!(
    out_le_rgb_u16, out_be_rgb_u16,
    "AYUV64 A+ RGB u16 LE/BE diverge"
  );
  assert_eq!(
    out_le_rgba_u16, out_be_rgba_u16,
    "AYUV64 A+ RGBA u16 LE/BE diverge"
  );
}

// ---- Atomicity (#308) --------------------------------------------------
//
// The packed 4:4:4 16-bit identity-path `process` must run the up-front
// RGB-scratch preflight BEFORE any output row (luma / luma_u16) is written, so
// an allocator refusal returns a recoverable `AllocationFailed` leaving the
// output frame untouched rather than partially mutated. Unlike the 8-bit
// packed siblings, `Ayuv64` also exposes u16 colour outputs, so the scratch
// arm (the u8 RGB row staged for HSV) is reached whenever HSV is wanted
// alongside ANY other colour output with no caller RGB buffer —
// `want_hsv && !want_rgb && (want_rgba || want_rgb_u16 || want_rgba_u16)`. Two
// tests pin both halves of that gate: the u8-colour combo and the
// u16-colour-only combo (which the narrow 8-bit gate would have missed).
// Reuses the crate's `yuva`-gated RGB-scratch failpoint, so these tests are
// too (under `--all-features` both `yuv-444-packed` and `yuva` are on).

#[cfg(feature = "yuva")]
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn ayuv64_rgb_scratch_alloc_failure_leaves_outputs_untouched() {
  use crate::resample::ResampleError;

  let n = 16 * 8;
  let buf = solid_ayuv64_frame(16, 8, 0xC000, 32768, 32768, 32768);
  let src = Ayuv64Frame::try_new(&buf, 16, 8, 16 * 4).unwrap();
  let mut luma = std::vec![0xABu8; n];
  let mut rgba = std::vec![0xCDu8; n * 4];
  let (mut hh, mut ss, mut vv) = (
    std::vec![0xCDu8; n],
    std::vec![0xCDu8; n],
    std::vec![0xCDu8; n],
  );
  let mut sink = MixedSinker::<Ayuv64>::new(16, 8)
    .with_luma(&mut luma)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap()
    .with_hsv(&mut hh, &mut ss, &mut vv)
    .unwrap();

  super::super::arm_rgb_scratch_alloc_failure();
  let err = ayuv64_to(&src, false, KernelMatrix::Bt601, &mut sink).unwrap_err();
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

// u16-colour-only variant: `with_hsv` + `with_rgba_u16` (no u8 RGB/RGBA, no
// RGB u16) still stages the u8 RGB scratch for the HSV derive, so it reaches
// `rgb_row_buf_or_scratch`'s scratch arm. This is the case the narrow 8-bit
// gate (`want_hsv && want_rgba && !want_rgb`) would have skipped — proving the
// preflight gate must include the u16 colour outputs. Sentinel is luma_u16.
#[cfg(feature = "yuva")]
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn ayuv64_rgb_scratch_alloc_failure_u16_color_leaves_outputs_untouched() {
  use crate::resample::ResampleError;

  let n = 16 * 8;
  let buf = solid_ayuv64_frame(16, 8, 0xC000, 32768, 32768, 32768);
  let src = Ayuv64Frame::try_new(&buf, 16, 8, 16 * 4).unwrap();
  let mut luma_u16 = std::vec![0xABABu16; n];
  let mut rgba_u16 = std::vec![0xCDCDu16; n * 4];
  let (mut hh, mut ss, mut vv) = (
    std::vec![0xCDu8; n],
    std::vec![0xCDu8; n],
    std::vec![0xCDu8; n],
  );
  let mut sink = MixedSinker::<Ayuv64>::new(16, 8)
    .with_luma_u16(&mut luma_u16)
    .unwrap()
    .with_rgba_u16(&mut rgba_u16)
    .unwrap()
    .with_hsv(&mut hh, &mut ss, &mut vv)
    .unwrap();

  super::super::arm_rgb_scratch_alloc_failure();
  let err = ayuv64_to(&src, false, KernelMatrix::Bt601, &mut sink).unwrap_err();
  drop(sink);

  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
    ),
    "RGB-scratch refusal must surface as a recoverable AllocationFailed, got {err:?}"
  );
  assert!(
    luma_u16.iter().all(|&b| b == 0xABAB),
    "luma_u16 must be untouched on the rgb-scratch alloc-failure path"
  );
}
