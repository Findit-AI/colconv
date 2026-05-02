//! Sinker integration tests for `MixedSinker<Vuya>` — Ship 12c.
//!
//! Coverage:
//! 1. `vuya_with_rgb_smoke` — gray row → ~mid-gray RGB.
//! 2. `vuya_with_rgba_passes_source_alpha` — α byte preserved per pixel.
//! 3. `vuya_with_luma_extracts_y_byte` — luma == source Y byte.
//! 4. `vuya_with_hsv_smoke` — gray row → HSV S = 0.
//! 5. `vuya_with_rgb_and_rgba_strategy_a_byte_identical` — Strategy A
//!    invariant: RGB channels are bit-identical in both outputs; α=0xFF.
//! 6. `vuya_simd_vs_scalar_parity_at_1922` — SIMD path == scalar.
//! 7. `vuya_width_mismatch_returns_error` — wrong row width → error.
//! 8. `vuya_row_index_oor_returns_error` — idx >= height → error.
//! 9. `vuya_rgb_buffer_too_short_returns_error`.
//! 10. `vuya_rgba_buffer_too_short_returns_error`.
//! 11. `vuya_luma_buffer_too_short_returns_error`.
//! 12. `vuya_hsv_buffer_too_short_returns_error`.
//! 13. `vuya_planar_parity_with_yuva444p` — VUYA packed ↔ Yuva444p
//!     planar cross-format oracle (RGB + RGBA byte-identical for the
//!     same logical YUVA samples).

#[cfg(all(test, feature = "std"))]
use super::*;

// ---- VUYA frame builder ---------------------------------------------------

/// Builds a solid-color VUYA plane. Each pixel is `[v_val, u_val, y_val,
/// a_val]`. Row stride equals `width × 4` bytes (no padding).
#[cfg(all(test, feature = "std"))]
pub(super) fn solid_vuya_frame(
  width: u32,
  height: u32,
  v: u8,
  u: u8,
  y: u8,
  a: u8,
) -> Vec<u8> {
  let quad = [v, u, y, a];
  (0..(width as usize) * (height as usize))
    .flat_map(|_| quad)
    .collect()
}

// ---- Helper: pack Yuva444p planar planes into VUYA packed stream ----------

/// Converts separate Y / U / V / A planes (8-bit, full-width 4:4:4) into a
/// packed VUYA byte stream. Byte order per pixel: `[V, U, Y, A]`.
#[cfg(all(test, feature = "std"))]
fn pack_yuva444p_to_vuya(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  a: &[u8],
  width: usize,
  height: usize,
) -> Vec<u8> {
  let mut out = vec![0u8; width * height * 4];
  for row in 0..height {
    for x in 0..width {
      let i = row * width + x;
      let off = i * 4;
      out[off] = v[i];
      out[off + 1] = u[i];
      out[off + 2] = y[i];
      out[off + 3] = a[i];
    }
  }
  out
}

// ---- 1: RGB smoke ---------------------------------------------------------

#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn vuya_with_rgb_smoke() {
  // Gray input: Y=128, U=V=128 (neutral chroma). BT.709.
  // Full-range gray → expect each RGB channel ≈ 128 ± tolerance.
  let buf = solid_vuya_frame(4, 1, 128, 128, 128, 0);
  let src = VuyaFrame::try_new(&buf, 4, 1, 16).unwrap();
  let mut rgb = std::vec![0u8; 4 * 3];
  let mut sink = MixedSinker::<Vuya>::new(4, 1).with_rgb(&mut rgb).unwrap();
  vuya_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();
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

// ---- 2: RGBA source-alpha pass-through ------------------------------------

#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn vuya_with_rgba_passes_source_alpha() {
  // 4 pixels with distinct A bytes. Standalone RGBA path → source α
  // must be preserved verbatim per pixel.
  let alphas: [u8; 4] = [0x00, 0x7F, 0xAB, 0xFF];
  let mut buf = std::vec![0u8; 4 * 4];
  for (i, &a) in alphas.iter().enumerate() {
    // V=128, U=128, Y=128, A=a_val
    buf[i * 4] = 128;
    buf[i * 4 + 1] = 128;
    buf[i * 4 + 2] = 128;
    buf[i * 4 + 3] = a;
  }
  let src = VuyaFrame::try_new(&buf, 4, 1, 16).unwrap();
  let mut rgba = std::vec![0u8; 4 * 4];
  let mut sink = MixedSinker::<Vuya>::new(4, 1)
    .with_rgba(&mut rgba)
    .unwrap();
  vuya_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();
  for (i, &expected_a) in alphas.iter().enumerate() {
    assert_eq!(
      rgba[i * 4 + 3],
      expected_a,
      "alpha mismatch at pixel {i}: expected {expected_a:#X}, got {:#X}",
      rgba[i * 4 + 3],
    );
  }
}

// ---- 3: Luma extracts Y byte directly -------------------------------------

#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn vuya_with_luma_extracts_y_byte() {
  // Y=0xC0 (192). Luma path must copy the Y byte at offset 2 verbatim.
  let buf = solid_vuya_frame(8, 2, 128, 128, 0xC0, 0xFF);
  let src = VuyaFrame::try_new(&buf, 8, 2, 32).unwrap();
  let mut luma = std::vec![0u8; 8 * 2];
  let mut sink = MixedSinker::<Vuya>::new(8, 2)
    .with_luma(&mut luma)
    .unwrap();
  vuya_to(&src, false, ColorMatrix::Bt709, &mut sink).unwrap();
  assert!(
    luma.iter().all(|&y| y == 0xC0),
    "luma expected 0xC0, got {:?}",
    &luma[..8]
  );
}

// ---- 4: HSV smoke (gray → S = 0) -----------------------------------------

#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn vuya_with_hsv_smoke() {
  // Neutral gray → converted RGB R==G==B → HSV saturation S = 0.
  let buf = solid_vuya_frame(6, 2, 128, 128, 128, 0);
  let src = VuyaFrame::try_new(&buf, 6, 2, 24).unwrap();
  let n = 6 * 2;
  let mut h = std::vec![0u8; n];
  let mut s = std::vec![0u8; n];
  let mut v = std::vec![0u8; n];
  let mut sink = MixedSinker::<Vuya>::new(6, 2)
    .with_hsv(&mut h, &mut s, &mut v)
    .unwrap();
  vuya_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();
  for &sat in &s {
    assert_eq!(sat, 0, "gray must have S=0 in HSV");
  }
}

// ---- 5: Strategy A invariant (rgb + rgba byte-identical; α=0xFF) ----------

#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn vuya_with_rgb_and_rgba_strategy_a_byte_identical() {
  // When BOTH with_rgb and with_rgba are attached, Strategy A applies:
  // RGBA RGB channels must be byte-identical to the RGB output, and
  // alpha must be 0xFF (NOT the source alpha byte).
  let w = 8u32;
  let h = 2u32;
  let buf = solid_vuya_frame(w, h, 60, 100, 180, 0x42); // A=0x42 != 0xFF
  let src = VuyaFrame::try_new(&buf, w, h, w * 4).unwrap();
  let mut rgb = std::vec![0u8; (w * h) as usize * 3];
  let mut rgba = std::vec![0u8; (w * h) as usize * 4];
  let mut sink = MixedSinker::<Vuya>::new(w as usize, h as usize)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap();
  vuya_to(&src, false, ColorMatrix::Bt709, &mut sink).unwrap();
  for i in 0..(w * h) as usize {
    assert_eq!(rgba[i * 4], rgb[i * 3], "R mismatch at pixel {i}");
    assert_eq!(rgba[i * 4 + 1], rgb[i * 3 + 1], "G mismatch at pixel {i}");
    assert_eq!(rgba[i * 4 + 2], rgb[i * 3 + 2], "B mismatch at pixel {i}");
    assert_eq!(
      rgba[i * 4 + 3],
      0xFF,
      "Strategy A: alpha must be 0xFF at pixel {i}, not source 0x42"
    );
  }
}

// ---- 6: SIMD vs scalar parity at width 1922 -------------------------------

#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn vuya_simd_vs_scalar_parity_at_1922() {
  // Width 1922 enters and exits the main loop + scalar tail of every
  // backend block size (NEON 16, SSE4.1 16, AVX2 32, AVX-512 64, wasm 16).
  let w = 1922usize;
  let h = 2usize;
  let mut buf = std::vec![0u8; w * h * 4];
  pseudo_random_u8(&mut buf, 0xBEEF_DEAD);
  let src = VuyaFrame::try_new(&buf, w as u32, h as u32, (w * 4) as u32).unwrap();

  let mut rgb_simd = std::vec![0u8; w * h * 3];
  let mut rgb_scalar = std::vec![0u8; w * h * 3];

  let mut sink_simd = MixedSinker::<Vuya>::new(w, h)
    .with_rgb(&mut rgb_simd)
    .unwrap();
  vuya_to(&src, false, ColorMatrix::Bt709, &mut sink_simd).unwrap();

  let mut sink_scalar = MixedSinker::<Vuya>::new(w, h)
    .with_rgb(&mut rgb_scalar)
    .unwrap()
    .with_simd(false);
  vuya_to(&src, false, ColorMatrix::Bt709, &mut sink_scalar).unwrap();

  assert_eq!(rgb_simd, rgb_scalar, "VUYA SIMD ≠ scalar at width {w}");
}

// ---- 7: Width mismatch → RowShapeMismatch ---------------------------------

#[test]
#[cfg(all(test, feature = "std"))]
fn vuya_width_mismatch_returns_error() {
  // Sinker built for width=64; pass a 128-pixel (512-byte) packed row.
  let mut rgb = std::vec![0u8; 64 * 3];
  let mut sink = MixedSinker::<Vuya>::new(64, 1).with_rgb(&mut rgb).unwrap();
  // Packed slice for width=128: 128 × 4 = 512 bytes.
  let packed = std::vec![0u8; 512];
  let row = VuyaRow::new(&packed, 0, ColorMatrix::Bt709, false);
  let err = sink.process(row).err().unwrap();
  assert_eq!(
    err,
    MixedSinkerError::RowShapeMismatch {
      which: RowSlice::VuyaPacked,
      row: 0,
      expected: 64 * 4,  // 256
      actual: 512,
    }
  );
}

// ---- 8: Row index out of range --------------------------------------------

#[test]
#[cfg(all(test, feature = "std"))]
fn vuya_row_index_oor_returns_error() {
  // Sinker built for height=2; pass row index 2 (== height → out of range).
  let mut rgb = std::vec![0u8; 4 * 2 * 3];
  let mut sink = MixedSinker::<Vuya>::new(4, 2).with_rgb(&mut rgb).unwrap();
  let packed = std::vec![0u8; 4 * 4]; // width=4, 4 bytes per pixel
  let row = VuyaRow::new(&packed, 2, ColorMatrix::Bt709, false);
  let err = sink.process(row).err().unwrap();
  assert!(matches!(
    err,
    MixedSinkerError::RowIndexOutOfRange {
      row: 2,
      configured_height: 2,
    }
  ));
}

// ---- 9: RGB buffer too short ----------------------------------------------

#[test]
#[cfg(all(test, feature = "std"))]
fn vuya_rgb_buffer_too_short_returns_error() {
  // 8×4 frame needs 8 × 4 × 3 = 96 bytes; supply only 95.
  let mut rgb = std::vec![0u8; 95];
  let result = MixedSinker::<Vuya>::new(8, 4).with_rgb(&mut rgb);
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

// ---- 10: RGBA buffer too short --------------------------------------------

#[test]
#[cfg(all(test, feature = "std"))]
fn vuya_rgba_buffer_too_short_returns_error() {
  // 6×4 frame needs 6 × 4 × 4 = 96 bytes; supply only 90.
  let mut rgba = std::vec![0u8; 90];
  let result = MixedSinker::<Vuya>::new(6, 4).with_rgba(&mut rgba);
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

// ---- 11: Luma buffer too short --------------------------------------------

#[test]
#[cfg(all(test, feature = "std"))]
fn vuya_luma_buffer_too_short_returns_error() {
  // 8×3 frame needs 8 × 3 = 24 bytes; supply 20.
  let mut luma = std::vec![0u8; 20];
  let result = MixedSinker::<Vuya>::new(8, 3).with_luma(&mut luma);
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

// ---- 12: HSV buffer too short (H plane) -----------------------------------

#[test]
#[cfg(all(test, feature = "std"))]
fn vuya_hsv_buffer_too_short_returns_error() {
  // 4×4 frame needs 16 bytes per HSV plane; supply H with only 15.
  let mut h = std::vec![0u8; 15];
  let mut s = std::vec![0u8; 16];
  let mut v = std::vec![0u8; 16];
  let result = MixedSinker::<Vuya>::new(4, 4).with_hsv(&mut h, &mut s, &mut v);
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

// ---- 13: Planar parity with Yuva444p (headline cross-format oracle) -------

#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn vuya_planar_parity_with_yuva444p() {
  // Spec § 8.3 cross-format invariant: Yuva444p planar and VUYA packed
  // carry identical logical YUVA 8-bit samples. Both sinks run with
  // `with_rgb` AND `with_rgba` attached (plus the standalone-RGBA path).
  //
  // Design note on byte-identity vs 1-LSB tolerance:
  //
  //   The Yuva444p sinker uses `yuv_444_to_rgb_or_rgba_row` (which calls
  //   `range_params` → limited-range scales `38142 / 37306`), while VUYA
  //   uses `vuya_to_rgb_or_rgba_row` (which calls `range_params_n::<8,8>`
  //   → limited-range scales computed from first principles, yielding
  //   slightly different constants). These two implementations are
  //   semantically equivalent but *not* bit-identical under limited range.
  //
  //   Full-range (full_range=true) avoids the issue: at 8-bit full-range
  //   both kernels reduce to scale = 1 << 15 (exact), making the math
  //   path bit-identical. We therefore use full_range=true.
  //
  // Alpha semantics:
  //   - VUYA with_rgb + with_rgba → Strategy A (expand_rgb_to_rgba,
  //     α=0xFF). Source alpha is NOT preserved in the combined path.
  //   - Yuva444p with_rgb + with_rgba → Strategy B fork (runs the alpha-
  //     aware kernel for RGBA regardless of RGB attachment). Source alpha
  //     IS preserved from the source plane.
  //
  //   Because the two formats use different alpha strategies when both
  //   RGB and RGBA are attached, we verify each separately:
  //     * RGB parity: `with_rgb` only on both — validates the shared
  //       YUV→RGB math is bit-identical.
  //     * RGBA parity (source-alpha path): `with_rgba` only on both
  //       (standalone) — invokes the direct RGBA kernel with source-α
  //       pass-through for VUYA and the alpha-aware kernel for Yuva444p.
  //
  // Use width=64 × height=4 (covers SIMD main loop + scalar tail).
  let width = 64usize;
  let height = 4usize;
  let n = width * height;

  // Build pseudo-random Y / U / V / A planes.
  let mut yp = std::vec![0u8; n];
  let mut up = std::vec![0u8; n];
  let mut vp = std::vec![0u8; n];
  let mut ap = std::vec![0u8; n];
  pseudo_random_u8(&mut yp, 0xC0FFEE_u32);
  pseudo_random_u8(&mut up, 0xBADF00D_u32);
  pseudo_random_u8(&mut vp, 0xFEEDFACE_u32);
  pseudo_random_u8(&mut ap, 0xA1FA5EED_u32);

  // Construct the Yuva444p planar frame.
  let planar = Yuva444pFrame::try_new(
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

  // Pack the same samples into a VUYA byte stream.
  let vuya_buf = pack_yuva444p_to_vuya(&yp, &up, &vp, &ap, width, height);
  let packed_frame =
    VuyaFrame::try_new(&vuya_buf, width as u32, height as u32, (width * 4) as u32).unwrap();

  // full_range=true so both kernels use scale = 1 << 15 (bit-identical).
  let full_range = true;

  // --- Part 1: RGB parity (`with_rgb` only — no RGBA, no alpha divergence) ---
  let mut p_rgb = std::vec![0u8; n * 3];
  let mut x_rgb = std::vec![0u8; n * 3];

  let mut p_sink = MixedSinker::<Yuva444p>::new(width, height)
    .with_rgb(&mut p_rgb)
    .unwrap();
  yuva444p_to(&planar, full_range, ColorMatrix::Bt709, &mut p_sink).unwrap();

  let mut x_sink = MixedSinker::<Vuya>::new(width, height)
    .with_rgb(&mut x_rgb)
    .unwrap();
  vuya_to(&packed_frame, full_range, ColorMatrix::Bt709, &mut x_sink).unwrap();

  assert_eq!(p_rgb, x_rgb, "VUYA ↔ Yuva444p u8 RGB diverges");

  // --- Part 2: Standalone RGBA source-alpha pass-through parity ---
  // Both formats run the standalone-RGBA path (no RGB, no HSV attached),
  // which invokes the source-alpha-aware kernel for each. The RGB channels
  // must be bit-identical (same math); the alpha channels must equal the
  // source A bytes (`ap`).
  let mut p_rgba = std::vec![0u8; n * 4];
  let mut x_rgba = std::vec![0u8; n * 4];

  let mut p_sink2 = MixedSinker::<Yuva444p>::new(width, height)
    .with_rgba(&mut p_rgba)
    .unwrap();
  yuva444p_to(&planar, full_range, ColorMatrix::Bt709, &mut p_sink2).unwrap();

  let mut x_sink2 = MixedSinker::<Vuya>::new(width, height)
    .with_rgba(&mut x_rgba)
    .unwrap();
  vuya_to(&packed_frame, full_range, ColorMatrix::Bt709, &mut x_sink2).unwrap();

  assert_eq!(p_rgba, x_rgba, "VUYA ↔ Yuva444p u8 RGBA diverges (source-alpha path)");

  // Spot-check alpha bytes equal the source alpha plane.
  for (i, &src_a) in ap.iter().enumerate() {
    let alpha_out = x_rgba[i * 4 + 3];
    assert_eq!(
      alpha_out,
      src_a,
      "VUYA RGBA alpha at pixel {i}: expected {src_a:#X}, got {alpha_out:#X}"
    );
  }
}
