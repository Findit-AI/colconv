//! Sinker integration tests for `MixedSinker<Vuyx>` — Ship 12c.
//!
//! Coverage:
//! 1. `vuyx_with_rgb_smoke` — gray row → ~mid-gray RGB.
//! 2. `vuyx_with_rgba_forces_alpha_max_with_zero_source` — source X = 0x00,
//!    output α = 0xFF for every pixel.
//! 3. `vuyx_with_rgba_forces_alpha_max_with_random_source` — source X bytes
//!    are random non-zero values (0x42, 0x99, etc.), output α = 0xFF.
//! 4. `vuyx_with_luma_extracts_y_byte` — luma == source Y byte.
//! 5. `vuyx_with_hsv_smoke` — gray row → HSV S = 0.
//! 6. `vuyx_with_rgb_and_rgba_strategy_a_byte_identical` — both attached;
//!    RGB and RGBA's first 3 bytes match; RGBA α = 0xFF (Strategy A).
//! 7. `vuyx_simd_vs_scalar_parity_at_1922` — SIMD path == scalar.
//! 8. `vuyx_width_mismatch_returns_error` — wrong row width → error.
//! 9. `vuyx_row_index_oor_returns_error` — idx >= height → error.
//! 10. `vuyx_rgb_buffer_too_short_returns_error`.
//! 11. `vuyx_rgba_buffer_too_short_returns_error`.
//! 12. `vuyx_luma_buffer_too_short_returns_error`.
//! 13. `vuyx_hsv_buffer_too_short_returns_error`.
//! 14. `vuyx_force_alpha_max_independent_of_source` — headline VUYX
//!     invariant (spec § 8.3): source X bytes are all distinct per pixel
//!     (0x00, 0x42, 0x99, 0xFF, …); output α = 0xFF for every pixel.

#[cfg(all(test, feature = "std"))]
use super::*;

// ---- VUYX frame builder ---------------------------------------------------

/// Builds a solid-color VUYX plane. Each pixel is `[v_val, u_val, y_val,
/// x_val]`. Row stride equals `width × 4` bytes (no padding).
///
/// The `x_val` is the padding byte — it should be ignored by the sinker.
#[cfg(all(test, feature = "std"))]
pub(super) fn solid_vuyx_frame(width: u32, height: u32, v: u8, u: u8, y: u8, x: u8) -> Vec<u8> {
  let quad = [v, u, y, x];
  (0..(width as usize) * (height as usize))
    .flat_map(|_| quad)
    .collect()
}

// ---- 1: RGB smoke ---------------------------------------------------------

#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn vuyx_with_rgb_smoke() {
  // Gray input: Y=128, U=V=128 (neutral chroma), X=0 (padding ignored).
  // Full-range gray → expect each RGB channel ≈ 128 ± tolerance.
  let buf = solid_vuyx_frame(4, 1, 128, 128, 128, 0);
  let src = VuyxFrame::try_new(&buf, 4, 1, 16).unwrap();
  let mut rgb = std::vec![0u8; 4 * 3];
  let mut sink = MixedSinker::<Vuyx>::new(4, 1).with_rgb(&mut rgb).unwrap();
  vuyx_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();
  for px in rgb.chunks(3) {
    assert!(
      px[0].abs_diff(128) <= 4,
      "expected ~128, got R={}, G={}, B={}",
      px[0],
      px[1],
      px[2],
    );
    // Gray → R == G == B.
    assert_eq!(px[0], px[1], "R ≠ G");
    assert_eq!(px[1], px[2], "G ≠ B");
  }
}

// ---- 2: RGBA forces α=0xFF when source X = 0x00 --------------------------

#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn vuyx_with_rgba_forces_alpha_max_with_zero_source() {
  // Source X bytes = 0x00. Output α must still be 0xFF (padding is ignored).
  let buf = solid_vuyx_frame(8, 1, 128, 128, 128, 0x00);
  let src = VuyxFrame::try_new(&buf, 8, 1, 32).unwrap();
  let mut rgba = std::vec![0u8; 8 * 4];
  let mut sink = MixedSinker::<Vuyx>::new(8, 1).with_rgba(&mut rgba).unwrap();
  vuyx_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();
  for (i, px) in rgba.chunks(4).enumerate() {
    assert_eq!(
      px[3], 0xFF,
      "pixel {i}: α must be 0xFF regardless of source X=0x00, got {:#X}",
      px[3],
    );
  }
}

// ---- 3: RGBA forces α=0xFF when source X bytes are random non-zero --------

#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn vuyx_with_rgba_forces_alpha_max_with_random_source() {
  // Source X bytes = 0x42 (arbitrary non-zero). Output α must be 0xFF.
  let buf = solid_vuyx_frame(6, 2, 128, 128, 128, 0x42);
  let src = VuyxFrame::try_new(&buf, 6, 2, 24).unwrap();
  let n = 6 * 2;
  let mut rgba = std::vec![0u8; n * 4];
  let mut sink = MixedSinker::<Vuyx>::new(6, 2).with_rgba(&mut rgba).unwrap();
  vuyx_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();
  for (i, px) in rgba.chunks(4).enumerate() {
    assert_eq!(
      px[3], 0xFF,
      "pixel {i}: α must be 0xFF, source X=0x42 must be ignored, got {:#X}",
      px[3],
    );
  }
}

// ---- 4: Luma extracts Y byte directly ------------------------------------

#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn vuyx_with_luma_extracts_y_byte() {
  // Y=0xC0 (192). Luma path must copy the Y byte at offset 2 verbatim.
  let buf = solid_vuyx_frame(8, 2, 128, 128, 0xC0, 0xFF);
  let src = VuyxFrame::try_new(&buf, 8, 2, 32).unwrap();
  let mut luma = std::vec![0u8; 8 * 2];
  let mut sink = MixedSinker::<Vuyx>::new(8, 2).with_luma(&mut luma).unwrap();
  vuyx_to(&src, false, ColorMatrix::Bt709, &mut sink).unwrap();
  assert!(
    luma.iter().all(|&y| y == 0xC0),
    "luma expected 0xC0, got {:?}",
    &luma[..8]
  );
}

// ---- 5: HSV smoke (gray → S = 0) ----------------------------------------

#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn vuyx_with_hsv_smoke() {
  // Neutral gray → converted RGB R==G==B → HSV saturation S = 0.
  let buf = solid_vuyx_frame(6, 2, 128, 128, 128, 0);
  let src = VuyxFrame::try_new(&buf, 6, 2, 24).unwrap();
  let n = 6 * 2;
  let mut h = std::vec![0u8; n];
  let mut s = std::vec![0u8; n];
  let mut v = std::vec![0u8; n];
  let mut sink = MixedSinker::<Vuyx>::new(6, 2)
    .with_hsv(&mut h, &mut s, &mut v)
    .unwrap();
  vuyx_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();
  for &sat in &s {
    assert_eq!(sat, 0, "gray must have S=0 in HSV");
  }
}

// ---- 6: RGB + RGBA Strategy A — byte-identical (α = 0xFF) ---------------

#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn vuyx_with_rgb_and_rgba_strategy_a_byte_identical() {
  // When BOTH with_rgb AND with_rgba are attached, VUYX uses Strategy A:
  // RGBA is derived from the RGB row via expand_rgb_to_rgba_row (α=0xFF).
  // Both paths produce α=0xFF — no semantic conflict (spec § 8.4).
  //
  // Verify: RGB and RGBA's first 3 bytes match; RGBA α = 0xFF.
  let width = 8usize;
  let height = 1usize;
  // Use varying X bytes (padding — must be ignored).
  let mut packed = std::vec![0u8; width * 4];
  for n in 0..width {
    packed[n * 4] = 128; // V (neutral chroma)
    packed[n * 4 + 1] = 128; // U (neutral chroma)
    packed[n * 4 + 2] = 200; // Y (bright luma)
    packed[n * 4 + 3] = (n as u8) * 30; // X: padding (0, 30, 60, 90, …)
  }
  let frame =
    VuyxFrame::try_new(&packed, width as u32, height as u32, (width * 4) as u32).unwrap();
  let mut rgb = std::vec![0u8; width * height * 3];
  let mut rgba = std::vec![0u8; width * height * 4];
  let mut sinker = MixedSinker::<Vuyx>::new(width, height)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap();
  vuyx_to(&frame, true, ColorMatrix::Bt709, &mut sinker).unwrap();

  for n in 0..width {
    // RGB and RGBA RGB-channels must be byte-identical (Strategy A).
    assert_eq!(
      &rgb[n * 3..n * 3 + 3],
      &rgba[n * 4..n * 4 + 3],
      "pixel {n}: RGB and RGBA RGB-channels diverge"
    );
    // RGBA α byte = 0xFF (padding X byte must NOT bleed through).
    assert_eq!(
      rgba[n * 4 + 3], 0xFF,
      "pixel {n}: RGBA α must be 0xFF, got {:#X}",
      rgba[n * 4 + 3],
    );
  }
}

// ---- 7: SIMD vs scalar parity at width 1922 ------------------------------

#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn vuyx_simd_vs_scalar_parity_at_1922() {
  // Width 1922 enters and exits the main loop + scalar tail of every
  // backend block size (NEON 16, SSE4.1 16, AVX2 32, AVX-512 64, wasm 16).
  let w = 1922usize;
  let h = 2usize;
  let mut buf = std::vec![0u8; w * h * 4];
  pseudo_random_u8(&mut buf, 0xBEEF_DEAD);
  let src = VuyxFrame::try_new(&buf, w as u32, h as u32, (w * 4) as u32).unwrap();

  let mut rgb_simd = std::vec![0u8; w * h * 3];
  let mut rgb_scalar = std::vec![0u8; w * h * 3];

  let mut sink_simd = MixedSinker::<Vuyx>::new(w, h)
    .with_rgb(&mut rgb_simd)
    .unwrap();
  vuyx_to(&src, false, ColorMatrix::Bt709, &mut sink_simd).unwrap();

  let mut sink_scalar = MixedSinker::<Vuyx>::new(w, h)
    .with_rgb(&mut rgb_scalar)
    .unwrap()
    .with_simd(false);
  vuyx_to(&src, false, ColorMatrix::Bt709, &mut sink_scalar).unwrap();

  assert_eq!(rgb_simd, rgb_scalar, "VUYX SIMD ≠ scalar at width {w}");
}

// ---- 8: Width mismatch → RowShapeMismatch ---------------------------------

#[test]
#[cfg(all(test, feature = "std"))]
fn vuyx_width_mismatch_returns_error() {
  // Sinker built for width=64; pass a 128-pixel (512-byte) packed row.
  let mut rgb = std::vec![0u8; 64 * 3];
  let mut sink = MixedSinker::<Vuyx>::new(64, 1).with_rgb(&mut rgb).unwrap();
  // Packed slice for width=128: 128 × 4 = 512 bytes.
  let packed = std::vec![0u8; 512];
  let row = VuyxRow::new(&packed, 0, ColorMatrix::Bt709, false);
  let err = sink.process(row).err().unwrap();
  assert_eq!(
    err,
    MixedSinkerError::RowShapeMismatch {
      which: RowSlice::VuyxPacked,
      row: 0,
      expected: 64 * 4, // 256
      actual: 512,
    }
  );
}

// ---- 9: Row index out of range --------------------------------------------

#[test]
#[cfg(all(test, feature = "std"))]
fn vuyx_row_index_oor_returns_error() {
  // Sinker built for height=2; pass row index 2 (== height → out of range).
  let mut rgb = std::vec![0u8; 4 * 2 * 3];
  let mut sink = MixedSinker::<Vuyx>::new(4, 2).with_rgb(&mut rgb).unwrap();
  let packed = std::vec![0u8; 4 * 4]; // width=4, 4 bytes per pixel
  let row = VuyxRow::new(&packed, 2, ColorMatrix::Bt709, false);
  let err = sink.process(row).err().unwrap();
  assert!(matches!(
    err,
    MixedSinkerError::RowIndexOutOfRange {
      row: 2,
      configured_height: 2,
    }
  ));
}

// ---- 10: RGB buffer too short ---------------------------------------------

#[test]
#[cfg(all(test, feature = "std"))]
fn vuyx_rgb_buffer_too_short_returns_error() {
  // 8×4 frame needs 8 × 4 × 3 = 96 bytes; supply only 95.
  let mut rgb = std::vec![0u8; 95];
  let result = MixedSinker::<Vuyx>::new(8, 4).with_rgb(&mut rgb);
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

// ---- 11: RGBA buffer too short -------------------------------------------

#[test]
#[cfg(all(test, feature = "std"))]
fn vuyx_rgba_buffer_too_short_returns_error() {
  // 6×4 frame needs 6 × 4 × 4 = 96 bytes; supply only 90.
  let mut rgba = std::vec![0u8; 90];
  let result = MixedSinker::<Vuyx>::new(6, 4).with_rgba(&mut rgba);
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

// ---- 12: Luma buffer too short -------------------------------------------

#[test]
#[cfg(all(test, feature = "std"))]
fn vuyx_luma_buffer_too_short_returns_error() {
  // 8×3 frame needs 8 × 3 = 24 bytes; supply 20.
  let mut luma = std::vec![0u8; 20];
  let result = MixedSinker::<Vuyx>::new(8, 3).with_luma(&mut luma);
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

// ---- 13: HSV buffer too short (H plane) ----------------------------------

#[test]
#[cfg(all(test, feature = "std"))]
fn vuyx_hsv_buffer_too_short_returns_error() {
  // 4×4 frame needs 16 bytes per HSV plane; supply H with only 15.
  let mut h = std::vec![0u8; 15];
  let mut s = std::vec![0u8; 16];
  let mut v = std::vec![0u8; 16];
  let result = MixedSinker::<Vuyx>::new(4, 4).with_hsv(&mut h, &mut s, &mut v);
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

// ---- 14: Force α=0xFF independent of source (headline VUYX invariant) ----

#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn vuyx_force_alpha_max_independent_of_source() {
  // Headline VUYX invariant (spec § 8.3): the X (padding) byte is
  // truly ignored — output α = 0xFF regardless of what is in the source.
  //
  // Fill source X bytes with all distinct values per pixel:
  // 0x00, 0x42, 0x99, 0xFF, 0x01, 0x7F, 0xAB, 0xCD, …
  // Every pixel has a different X byte. ALL output α must be 0xFF.
  let width = 16usize;
  let height = 4usize;
  let n = width * height;
  let x_pattern: [u8; 8] = [0x00, 0x42, 0x99, 0xFF, 0x01, 0x7F, 0xAB, 0xCD];

  let mut packed = std::vec![0u8; n * 4];
  for i in 0..n {
    packed[i * 4] = 128; // V
    packed[i * 4 + 1] = 128; // U
    packed[i * 4 + 2] = 128; // Y (neutral gray)
    packed[i * 4 + 3] = x_pattern[i % x_pattern.len()]; // X: distinct per pixel
  }
  let src =
    VuyxFrame::try_new(&packed, width as u32, height as u32, (width * 4) as u32).unwrap();
  let mut rgba = std::vec![0u8; n * 4];
  let mut sink = MixedSinker::<Vuyx>::new(width, height)
    .with_rgba(&mut rgba)
    .unwrap();
  vuyx_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

  for i in 0..n {
    let src_x = x_pattern[i % x_pattern.len()];
    let out_a = rgba[i * 4 + 3];
    assert_eq!(
      out_a, 0xFF,
      "pixel {i}: output α = {out_a:#X} but must be 0xFF (source X = {src_x:#X} must be ignored)"
    );
  }
}
