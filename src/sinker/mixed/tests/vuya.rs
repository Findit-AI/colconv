//! Sinker integration tests for `MixedSinker<Vuya>` — Ship 12c.
//!
//! Coverage:
//! 1. `vuya_with_rgb_smoke` — gray row → ~mid-gray RGB.
//! 2. `vuya_with_rgba_passes_source_alpha` — α byte preserved per pixel.
//! 3. `vuya_with_luma_extracts_y_byte` — luma == source Y byte.
//! 4. `vuya_with_hsv_smoke` — gray row → HSV S = 0.
//! 5. `vuya_with_rgb_and_rgba_preserves_source_alpha` — spec § 7.2:
//!    both kernels run independently; RGBA α equals source α, not 0xFF.
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
//! 14. `vuya_strategy_a_plus_matches_independent_kernel` — Strategy A+
//!     correctness: combo path output == scalar inline-α kernel output.

#[cfg(all(test, feature = "std"))]
use super::*;

// ---- VUYA frame builder ---------------------------------------------------

/// Builds a solid-color VUYA plane. Each pixel is `[v_val, u_val, y_val,
/// a_val]`. Row stride equals `width x 4` bytes (no padding).
#[cfg(all(test, feature = "std"))]
pub(super) fn solid_vuya_frame(width: u32, height: u32, v: u8, u: u8, y: u8, a: u8) -> Vec<u8> {
  let quad = [v, u, y, a];
  (0..(width as usize) * (height as usize))
    .flat_map(|_| quad)
    .collect()
}

// ---- Helper: pack Yuva444p planar planes into VUYA packed stream ----------

/// Converts separate Y / U / V / A planes (8-bit, full-width 4:4:4) into a
/// packed VUYA byte stream. Byte order per pixel: `[V, U, Y, A]`.
///
/// Gated on `yuva`: the cross-format oracle below feeds the planar
/// `Yuva444p` source, absent in a `yuv-444-packed`-solo build.
#[cfg(all(test, feature = "std", feature = "yuva"))]
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
  vuya_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
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
  let mut sink = MixedSinker::<Vuya>::new(4, 1).with_rgba(&mut rgba).unwrap();
  vuya_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
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
  let mut sink = MixedSinker::<Vuya>::new(8, 2).with_luma(&mut luma).unwrap();
  vuya_to(&src, false, KernelMatrix::Bt709, &mut sink).unwrap();
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
  vuya_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
  for &sat in &s {
    assert_eq!(sat, 0, "gray must have S=0 in HSV");
  }
}

// ---- 5: RGB + RGBA combined path preserves source alpha (spec § 7.2) --------

#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn vuya_with_rgb_and_rgba_preserves_source_alpha() {
  // When BOTH with_rgb AND with_rgba are attached, VUYA must run
  // both direct kernels — RGB drops α, RGBA passes source α through.
  // The RGBA path must NOT use Strategy A's α=0xFF fan-out (per spec § 7.2).

  let width = 8usize;
  let height = 1usize;
  // Build a packed VUYA row with distinct source α bytes
  let mut packed = std::vec![0u8; width * 4];
  for n in 0..width {
    packed[n * 4] = 128; // V (neutral chroma)
    packed[n * 4 + 1] = 128; // U (neutral chroma)
    packed[n * 4 + 2] = 128; // Y (mid gray)
    packed[n * 4 + 3] = (n as u8) * 32 + 1; // distinct A: 1, 33, 65, ..., 225
  }
  let frame = VuyaFrame::try_new(&packed, width as u32, height as u32, (width * 4) as u32).unwrap();
  let mut rgb = std::vec![0u8; width * height * 3];
  let mut rgba = std::vec![0u8; width * height * 4];
  let mut sinker = MixedSinker::<Vuya>::new(width, height)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap();
  vuya_to(&frame, true, KernelMatrix::Bt709, &mut sinker).unwrap();

  // Each pixel: RGB matches the YUV→RGB output for gray Y=128
  for n in 0..width {
    // RGB and RGBA's first 3 bytes are bit-identical (both kernels run on same packed input)
    assert_eq!(
      &rgb[n * 3..n * 3 + 3],
      &rgba[n * 4..n * 4 + 3],
      "pixel {n}: RGB and RGBA RGB-channels diverge (RGB={:?} RGBA={:?})",
      &rgb[n * 3..n * 3 + 3],
      &rgba[n * 4..n * 4 + 3]
    );
    // RGBA α byte = source α (NOT 0xFF — per spec § 7.2 the direct kernel runs)
    let expected_alpha = (n as u8) * 32 + 1;
    assert_eq!(
      rgba[n * 4 + 3],
      expected_alpha,
      "pixel {n}: source α was discarded (got {}, expected {})",
      rgba[n * 4 + 3],
      expected_alpha
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
  vuya_to(&src, false, KernelMatrix::Bt709, &mut sink_simd).unwrap();

  let mut sink_scalar = MixedSinker::<Vuya>::new(w, h)
    .with_rgb(&mut rgb_scalar)
    .unwrap()
    .with_simd(false);
  vuya_to(&src, false, KernelMatrix::Bt709, &mut sink_scalar).unwrap();

  assert_eq!(rgb_simd, rgb_scalar, "VUYA SIMD ≠ scalar at width {w}");
}

// ---- 9: RGB buffer too short ----------------------------------------------

#[test]
#[cfg(all(test, feature = "std"))]
fn vuya_rgb_buffer_too_short_returns_error() {
  // 8x4 frame needs 8 x 4 x 3 = 96 bytes; supply only 95.
  let mut rgb = std::vec![0u8; 95];
  let result = MixedSinker::<Vuya>::new(8, 4).with_rgb(&mut rgb);
  let Err(err) = result else {
    panic!("expected InsufficientRgbBuffer");
  };
  assert_eq!(
    err,
    MixedSinkerError::InsufficientRgbBuffer(InsufficientBuffer::new(96, 95))
  );
}

// ---- 10: RGBA buffer too short --------------------------------------------

#[test]
#[cfg(all(test, feature = "std"))]
fn vuya_rgba_buffer_too_short_returns_error() {
  // 6x4 frame needs 6 x 4 x 4 = 96 bytes; supply only 90.
  let mut rgba = std::vec![0u8; 90];
  let result = MixedSinker::<Vuya>::new(6, 4).with_rgba(&mut rgba);
  let Err(err) = result else {
    panic!("expected InsufficientRgbaBuffer");
  };
  assert_eq!(
    err,
    MixedSinkerError::InsufficientRgbaBuffer(InsufficientBuffer::new(96, 90))
  );
}

// ---- 11: Luma buffer too short --------------------------------------------

#[test]
#[cfg(all(test, feature = "std"))]
fn vuya_luma_buffer_too_short_returns_error() {
  // 8x3 frame needs 8 x 3 = 24 bytes; supply 20.
  let mut luma = std::vec![0u8; 20];
  let result = MixedSinker::<Vuya>::new(8, 3).with_luma(&mut luma);
  let Err(err) = result else {
    panic!("expected InsufficientLumaBuffer");
  };
  assert_eq!(
    err,
    MixedSinkerError::InsufficientLumaBuffer(InsufficientBuffer::new(24, 20))
  );
}

// ---- 12: HSV buffer too short (H plane) -----------------------------------

#[test]
#[cfg(all(test, feature = "std"))]
fn vuya_hsv_buffer_too_short_returns_error() {
  // 4x4 frame needs 16 bytes per HSV plane; supply H with only 15.
  let mut h = std::vec![0u8; 15];
  let mut s = std::vec![0u8; 16];
  let mut v = std::vec![0u8; 16];
  let result = MixedSinker::<Vuya>::new(4, 4).with_hsv(&mut h, &mut s, &mut v);
  let Err(err) = result else {
    panic!("expected InsufficientHsvPlane");
  };
  assert!(matches!(
    &err,
    MixedSinkerError::InsufficientHsvPlane(e)
      if matches!(e.which(), HsvPlane::H) && e.expected() == 16 && e.actual() == 15
  ));
}

// ---- 13: Planar parity with Yuva444p (headline cross-format oracle) -------
//
// Gated on `yuva` so a `yuv-444-packed`-solo `--tests` build (which lacks
// the planar `Yuva444p` frame / walker) still compiles.
#[test]
#[cfg(all(test, feature = "std", feature = "yuva"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn vuya_planar_parity_with_yuva444p() {
  // Spec § 8.3 cross-format invariant: Yuva444p planar and VUYA packed
  // carry identical logical YUVA 8-bit samples. Both sinks run with
  // `with_rgb` AND `with_rgba` attached (plus the standalone-RGBA path).
  //
  // Design note on byte-identity:
  //
  //   After the range_params_n::<8, 8> migration (v0.17.0), both the
  //   Yuva444p sinker (`yuv_444_to_rgb_or_rgba_row`) and the VUYA sinker
  //   (`vuya_to_rgb_or_rgba_row`) use `range_params_n::<8, 8>` for scale
  //   constants. Both paths are now byte-identical at full range AND limited
  //   range. The prior full_range=true workaround has been removed.
  //
  // Alpha semantics:
  //   - VUYA with_rgb + with_rgba → each runs its own independent kernel
  //     (spec § 7.2). Source alpha IS preserved via the direct
  //     vuya_to_rgba_row call. Strategy A fan-out is never used for VUYA.
  //   - Yuva444p with_rgb + with_rgba → Strategy B fork (runs the alpha-
  //     aware kernel for RGBA regardless of RGB attachment). Source alpha
  //     IS preserved from the source plane.
  //
  //   Both formats preserve source alpha in all paths. We verify each
  //   format separately for both outputs:
  //     * RGB parity: `with_rgb` only on both — validates the shared
  //       YUV→RGB math is bit-identical.
  //     * RGBA parity (source-alpha path): `with_rgba` only on both
  //       (standalone) — invokes the direct RGBA kernel with source-α
  //       pass-through for both formats.
  //
  // Use width=64 x height=4 (covers SIMD main loop + scalar tail).
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

  // Both kernels now use `range_params_n::<8, 8>` so we exercise BOTH
  // full-range AND limited-range — the prior `full_range=true`-only
  // workaround is gone. Both ranges must be byte-identical between the
  // VUYA and Yuva444p paths.
  for full_range in [true, false] {
    // --- Part 1: RGB parity (`with_rgb` only — no RGBA, no alpha divergence) ---
    let mut p_rgb = std::vec![0u8; n * 3];
    let mut x_rgb = std::vec![0u8; n * 3];

    let mut p_sink = MixedSinker::<Yuva444p>::new(width, height)
      .with_rgb(&mut p_rgb)
      .unwrap();
    yuva444p_to(&planar, full_range, KernelMatrix::Bt709, &mut p_sink).unwrap();

    let mut x_sink = MixedSinker::<Vuya>::new(width, height)
      .with_rgb(&mut x_rgb)
      .unwrap();
    vuya_to(&packed_frame, full_range, KernelMatrix::Bt709, &mut x_sink).unwrap();

    assert_eq!(
      p_rgb, x_rgb,
      "VUYA ↔ Yuva444p u8 RGB diverges (full_range={full_range})"
    );

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
    yuva444p_to(&planar, full_range, KernelMatrix::Bt709, &mut p_sink2).unwrap();

    let mut x_sink2 = MixedSinker::<Vuya>::new(width, height)
      .with_rgba(&mut x_rgba)
      .unwrap();
    vuya_to(&packed_frame, full_range, KernelMatrix::Bt709, &mut x_sink2).unwrap();

    assert_eq!(
      p_rgba, x_rgba,
      "VUYA ↔ Yuva444p u8 RGBA diverges (source-alpha path, full_range={full_range})"
    );

    // Spot-check alpha bytes equal the source alpha plane.
    for (i, &src_a) in ap.iter().enumerate() {
      let alpha_out = x_rgba[i * 4 + 3];
      assert_eq!(
        alpha_out, src_a,
        "VUYA RGBA alpha at pixel {i}: expected {src_a:#X}, got {alpha_out:#X} \
         (full_range={full_range})"
      );
    }
  }
}

// ---- 14: Strategy A+ correctness (spec § 6.1) ----------------------------

/// Strategy A+ correctness: combo path output == inline-α kernel output
/// at all (range, matrix) combinations. See spec § 6.1.
///
/// Validates byte-identity by running the sinker combo path (which uses
/// A+) against the scalar inline-α kernel directly.
#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn vuya_strategy_a_plus_matches_independent_kernel() {
  let width = 128usize;
  let height = 4usize;

  // Pseudo-random source.
  let mut packed = std::vec![0u8; width * height * 4];
  pseudo_random_u8(&mut packed, 0xC0FFEE);
  let frame = VuyaFrame::try_new(&packed, width as u32, height as u32, (width * 4) as u32).unwrap();

  for full_range in [true, false] {
    for matrix in [
      KernelMatrix::Bt601,
      KernelMatrix::Bt709,
      KernelMatrix::Bt2020Ncl,
      KernelMatrix::Smpte240m,
      KernelMatrix::Fcc,
      KernelMatrix::YCgCo,
    ] {
      // Sinker path (uses A+).
      let mut sinker_rgb = std::vec![0u8; width * height * 3];
      let mut sinker_rgba = std::vec![0u8; width * height * 4];
      {
        let mut sink = MixedSinker::<Vuya>::new(width, height)
          .with_rgb(&mut sinker_rgb)
          .unwrap()
          .with_rgba(&mut sinker_rgba)
          .unwrap();
        vuya_to(&frame, full_range, matrix, &mut sink).unwrap();
      }

      // Reference: scalar inline-α kernel directly (per row).
      let mut inline_rgba = std::vec![0u8; width * height * 4];
      let mut inline_rgb = std::vec![0u8; width * height * 3];
      for r in 0..height {
        let row_off_packed = r * width * 4;
        let row_off_rgb = r * width * 3;
        let row_off_rgba = r * width * 4;
        crate::row::scalar::vuya_to_rgb_row(
          &packed[row_off_packed..row_off_packed + width * 4],
          &mut inline_rgb[row_off_rgb..row_off_rgb + width * 3],
          width,
          matrix,
          full_range,
        );
        crate::row::scalar::vuya_to_rgba_row(
          &packed[row_off_packed..row_off_packed + width * 4],
          &mut inline_rgba[row_off_rgba..row_off_rgba + width * 4],
          width,
          matrix,
          full_range,
        );
      }

      assert_eq!(
        sinker_rgb, inline_rgb,
        "VUYA A+ RGB diverges (range={full_range}, matrix={matrix:?})"
      );
      assert_eq!(
        sinker_rgba, inline_rgba,
        "VUYA A+ RGBA diverges from scalar inline-α (range={full_range}, matrix={matrix:?})"
      );
    }
  }
}

// ---- with_luma_u16 tests -----------------------------------------------

#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn vuya_with_luma_u16_extracts_y_zero_extended() {
  let width = 64usize;
  let height = 4usize;
  let n = width * height;

  let mut packed = std::vec![0u8; n * 4];
  pseudo_random_u8(&mut packed, 0xC0FFEE);

  let frame = VuyaFrame::try_new(&packed, width as u32, height as u32, (width * 4) as u32).unwrap();

  let mut luma = std::vec![0u16; n];
  let mut sink = MixedSinker::<Vuya>::new(width, height)
    .with_luma_u16(&mut luma)
    .unwrap();
  vuya_to(&frame, false, KernelMatrix::Bt709, &mut sink).unwrap();

  // Reference: Y bytes at offset 2 of each 4-byte pixel quadruple.
  let expected: std::vec::Vec<u16> = (0..n).map(|i| packed[i * 4 + 2] as u16).collect();
  assert_eq!(luma, expected);
}

#[test]
#[cfg(all(test, feature = "std"))]
fn vuya_luma_u16_buffer_too_short_returns_err() {
  let mut luma = std::vec![0u16; 4 * 4 - 1];
  let result = MixedSinker::<Vuya>::new(4, 4).with_luma_u16(&mut luma);
  let Err(err) = result else {
    panic!("expected InsufficientLumaU16Buffer");
  };
  assert_eq!(
    err,
    MixedSinkerError::InsufficientLumaU16Buffer(InsufficientBuffer::new(16, 15)),
    "unexpected error: {err:?}"
  );
}

// Strategy A+ honors `with_simd(false)`.
//
// `MixedSinker::with_simd(false)` is a documented public knob (used by
// benchmarks, fuzzers, and differential testing). All existing kernel
// calls thread `use_simd = self.simd` to row-level dispatchers; the
// `alpha_extract::*` helpers also accept the flag (otherwise they would
// always select the highest available SIMD backend, silently bypassing
// the knob for the α-extract step of A+).
//
// This test exercises the scalar-fallback path through the dispatcher
// and pins that with_simd(false) still produces correct output when
// combined with the A+ flow. The dispatcher's scalar branch is
// validated against the SIMD-default A+ output (which itself is
// byte-identical to the inline-α reference per test #14).
#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn vuya_strategy_a_plus_with_simd_false_uses_scalar_path() {
  let width = 67usize; // covers SIMD main loop + scalar tail
  let height = 2usize;

  let mut packed = std::vec![0u8; width * height * 4];
  pseudo_random_u8(&mut packed, 0xFEED_BABE);
  let frame = VuyaFrame::try_new(&packed, width as u32, height as u32, (width * 4) as u32).unwrap();

  // Default A+ (SIMD-on, when available).
  let mut simd_rgb = std::vec![0u8; width * height * 3];
  let mut simd_rgba = std::vec![0u8; width * height * 4];
  {
    let mut sink = MixedSinker::<Vuya>::new(width, height)
      .with_rgb(&mut simd_rgb)
      .unwrap()
      .with_rgba(&mut simd_rgba)
      .unwrap();
    vuya_to(&frame, false, KernelMatrix::Bt709, &mut sink).unwrap();
  }

  // A+ with with_simd(false): scalar path through the α-extract dispatcher.
  let mut scalar_rgb = std::vec![0u8; width * height * 3];
  let mut scalar_rgba = std::vec![0u8; width * height * 4];
  {
    let mut sink = MixedSinker::<Vuya>::new(width, height)
      .with_rgb(&mut scalar_rgb)
      .unwrap()
      .with_rgba(&mut scalar_rgba)
      .unwrap()
      .with_simd(false);
    vuya_to(&frame, false, KernelMatrix::Bt709, &mut sink).unwrap();
  }

  assert_eq!(
    simd_rgb, scalar_rgb,
    "VUYA A+ RGB diverges between SIMD and with_simd(false) paths"
  );
  assert_eq!(
    simd_rgba, scalar_rgba,
    "VUYA A+ RGBA diverges between SIMD and with_simd(false) paths"
  );
}

// ---- Atomicity (#308) --------------------------------------------------
//
// The packed 4:4:4 identity-path `process` must run the up-front RGB-scratch
// preflight BEFORE any output row (luma / luma_u16) is written, so an allocator
// refusal returns a recoverable `AllocationFailed` leaving the output frame
// untouched rather than partially mutated. The `luma + RGBA + HSV, no RGB`
// combo is the identity-path shape that reaches `rgb_row_buf_or_scratch`'s
// scratch arm (`want_hsv && want_rgba && !want_rgb`); standalone RGBA uses the
// dedicated `vuya_to_rgba_row` kernel and never grows the scratch. Reuses the
// crate's `yuva`-gated RGB-scratch failpoint, so this test is too (under
// `--all-features` both `yuv-444-packed` and `yuva` are on).

#[cfg(feature = "yuva")]
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn vuya_rgb_scratch_alloc_failure_leaves_outputs_untouched() {
  use crate::resample::ResampleError;

  let buf = solid_vuya_frame(16, 8, 128, 128, 128, 200);
  let src = VuyaFrame::try_new(&buf, 16, 8, 16 * 4).unwrap();
  let n = 16 * 8;
  let mut luma = std::vec![0xABu8; n];
  let mut rgba = std::vec![0xCDu8; n * 4];
  let (mut hh, mut ss, mut vv) = (
    std::vec![0xCDu8; n],
    std::vec![0xCDu8; n],
    std::vec![0xCDu8; n],
  );
  let mut sink = MixedSinker::<Vuya>::new(16, 8)
    .with_luma(&mut luma)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap()
    .with_hsv(&mut hh, &mut ss, &mut vv)
    .unwrap();

  super::super::arm_rgb_scratch_alloc_failure();
  let err = vuya_to(&src, false, KernelMatrix::Bt601, &mut sink).unwrap_err();
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
