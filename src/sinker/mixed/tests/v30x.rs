//! Tier 5 V30X sinker tests — Ship 12a.
//!
//! Coverage matrix:
//! - Single-output paths (luma u8, rgb, rgba, rgb_u16, rgba_u16, hsv)
//!   on solid-gray frames.
//! - Strategy A invariant (`with_rgb` + `with_rgba` byte-identical;
//!   same for the u16 variants).
//! - SIMD-vs-scalar parity across multiple widths covering the main
//!   loop + scalar tail of every backend block size.
//! - Three error-path tests: short packed slice, row index out of
//!   range, and short rgba_u16 buffer.
//!
//! Note: `with_luma_u16` is NOT tested here — V30X does not expose
//! that accessor (deferred per Spec § 11).

#[cfg(all(test, feature = "std"))]
use super::*;

// ---- Solid-color V30X builder -----------------------------------------

/// Builds a solid-color V30X plane with one (U, Y, V) triplet repeated.
/// Each pixel is packed as a u32 word:
///   bits[31:22] = V (10-bit)
///   bits[21:12] = Y (10-bit)
///   bits[11:2]  = U (10-bit)
///   bits[1:0]   = 0 (padding — opposite end from V410)
///
/// Row stride equals width (one u32 per pixel; no padding between rows).
#[cfg(all(test, feature = "std"))]
pub(super) fn solid_v30x_frame(width: u32, height: u32, u: u32, y: u32, v: u32) -> Vec<u32> {
  let word = (v << 22) | (y << 12) | (u << 2);
  std::vec![word; (width as usize) * (height as usize)]
}

// ---- Single-output gray-to-gray tests ---------------------------------

#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn v30x_luma_only_extracts_y_bytes_downshifted() {
  // Y=256 (10-bit) → 8-bit (256 >> 2) = 64.
  let buf = solid_v30x_frame(6, 8, 512, 256, 512);
  let src = V30XFrame::new(&buf, 6, 8, 6);
  let mut luma = std::vec![0u8; 6 * 8];
  let mut sink = MixedSinker::<V30X>::new(6, 8).with_luma(&mut luma).unwrap();
  v30x_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();
  // 10-bit Y=256 → 8-bit (256 >> 2) = 64.
  assert!(luma.iter().all(|&y| y == 64), "luma {luma:?}");
}

#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn v30x_rgb_only_converts_gray_to_gray() {
  // Y=512, U=V=512 (neutral chroma at 10-bit midpoint ≈ 0.5×1023).
  // Mid-gray input should yield mid-gray output at ~128 ± tolerance.
  let buf = solid_v30x_frame(12, 4, 512, 512, 512);
  let src = V30XFrame::new(&buf, 12, 4, 12);
  let mut rgb = std::vec![0u8; 12 * 4 * 3];
  let mut sink = MixedSinker::<V30X>::new(12, 4).with_rgb(&mut rgb).unwrap();
  v30x_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();
  for px in rgb.chunks(3) {
    assert!(px[0].abs_diff(128) <= 4);
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
fn v30x_rgba_only_converts_gray_to_gray_with_opaque_alpha() {
  let buf = solid_v30x_frame(12, 4, 512, 512, 512);
  let src = V30XFrame::new(&buf, 12, 4, 12);
  let mut rgba = std::vec![0u8; 12 * 4 * 4];
  let mut sink = MixedSinker::<V30X>::new(12, 4)
    .with_rgba(&mut rgba)
    .unwrap();
  v30x_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();
  for px in rgba.chunks(4) {
    assert_eq!(px[3], 0xFF);
  }
}

#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn v30x_rgb_u16_only_converts_gray_to_gray_native_depth() {
  // Y=U=V=512 (10-bit midpoint). After YUV→RGB at 10-bit depth the
  // per-channel value should be near 512; allow ±16 for Q15 rounding.
  let buf = solid_v30x_frame(12, 4, 512, 512, 512);
  let src = V30XFrame::new(&buf, 12, 4, 12);
  let mut rgb = std::vec![0u16; 12 * 4 * 3];
  let mut sink = MixedSinker::<V30X>::new(12, 4)
    .with_rgb_u16(&mut rgb)
    .unwrap();
  v30x_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();
  for px in rgb.chunks(3) {
    assert!(px[0].abs_diff(512) <= 16, "expected ~512, got {}", px[0]);
  }
}

#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn v30x_rgba_u16_alpha_is_max() {
  // 10-bit alpha max = 0x3FF = 1023.
  let buf = solid_v30x_frame(12, 4, 512, 512, 512);
  let src = V30XFrame::new(&buf, 12, 4, 12);
  let mut rgba = std::vec![0u16; 12 * 4 * 4];
  let mut sink = MixedSinker::<V30X>::new(12, 4)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  v30x_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();
  for px in rgba.chunks(4) {
    assert_eq!(px[3], 0x3FF);
  }
}

#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn v30x_hsv_only_produces_valid_hue_range() {
  // Solid gray frame → HSV should have H≈0, S≈0, V≈mid-range.
  let buf = solid_v30x_frame(12, 4, 512, 512, 512);
  let src = V30XFrame::new(&buf, 12, 4, 12);
  let n = 12 * 4;
  let mut h = std::vec![0u8; n];
  let mut s = std::vec![0u8; n];
  let mut v_plane = std::vec![0u8; n];
  let mut sink = MixedSinker::<V30X>::new(12, 4)
    .with_hsv(&mut h, &mut s, &mut v_plane)
    .unwrap();
  v30x_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();
  // Gray pixels: hue and saturation must be 0.
  assert!(h.iter().all(|&x| x == 0), "H {h:?}");
  assert!(s.iter().all(|&x| x == 0), "S {s:?}");
}

// ---- Strategy A invariant tests ---------------------------------------

#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn v30x_with_rgb_and_with_rgba_byte_identical_u8() {
  // Strategy A invariant on u8 path — calling both `with_rgb` and
  // `with_rgba` must produce the same RGB bytes in both buffers, with
  // alpha = 0xFF in the RGBA buffer.
  let w = 12u32;
  let h = 4u32;
  let buf = solid_v30x_frame(w, h, 200, 700, 400);
  let src = V30XFrame::new(&buf, w, h, w);
  let mut rgb = std::vec![0u8; (w * h) as usize * 3];
  let mut rgba = std::vec![0u8; (w * h) as usize * 4];
  let mut sink = MixedSinker::<V30X>::new(w as usize, h as usize)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap();
  v30x_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();
  for i in 0..(w * h) as usize {
    assert_eq!(rgba[i * 4], rgb[i * 3]);
    assert_eq!(rgba[i * 4 + 1], rgb[i * 3 + 1]);
    assert_eq!(rgba[i * 4 + 2], rgb[i * 3 + 2]);
    assert_eq!(rgba[i * 4 + 3], 0xFF);
  }
}

#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn v30x_with_rgb_u16_and_with_rgba_u16_byte_identical() {
  // Strategy A invariant on u16 path — alpha must be 0x3FF (10-bit
  // max, low-bit-packed).
  let w = 12u32;
  let h = 4u32;
  let buf = solid_v30x_frame(w, h, 200, 700, 400);
  let src = V30XFrame::new(&buf, w, h, w);
  let mut rgb = std::vec![0u16; (w * h) as usize * 3];
  let mut rgba = std::vec![0u16; (w * h) as usize * 4];
  let mut sink = MixedSinker::<V30X>::new(w as usize, h as usize)
    .with_rgb_u16(&mut rgb)
    .unwrap()
    .with_rgba_u16(&mut rgba)
    .unwrap();
  v30x_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();
  for i in 0..(w * h) as usize {
    assert_eq!(rgba[i * 4], rgb[i * 3]);
    assert_eq!(rgba[i * 4 + 1], rgb[i * 3 + 1]);
    assert_eq!(rgba[i * 4 + 2], rgb[i * 3 + 2]);
    assert_eq!(rgba[i * 4 + 3], 0x3FF);
  }
}

// ---- SIMD-vs-scalar parity --------------------------------------------

#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn v30x_with_simd_false_matches_with_simd_true() {
  // Pseudo-random V30X across multiple widths covering the main loop
  // + scalar tail of every backend block size. V30X samples are 10-bit
  // packed in u32 words as (v << 22) | (y << 12) | (u << 2); pseudo-
  // random fill uses the low 10 bits of each channel slot.
  for w in [1usize, 2, 4, 7, 8, 15, 16, 17, 31, 32, 33, 1920, 1921] {
    let h = 2usize;
    let mut buf = std::vec![0u32; w * h];
    // Fill each word with pseudo-random 10-bit channels.
    let mut state = 0xC0FFEE_u32;
    for word in &mut buf {
      state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
      let u = (state >> 2) & 0x3FF;
      state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
      let y = (state >> 2) & 0x3FF;
      state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
      let v = (state >> 2) & 0x3FF;
      *word = (v << 22) | (y << 12) | (u << 2);
    }
    let src = V30XFrame::new(&buf, w as u32, h as u32, w as u32);

    let mut rgb_simd = std::vec![0u8; w * h * 3];
    let mut rgb_scalar = std::vec![0u8; w * h * 3];
    let mut sink_simd = MixedSinker::<V30X>::new(w, h)
      .with_rgb(&mut rgb_simd)
      .unwrap();
    let mut sink_scalar = MixedSinker::<V30X>::new(w, h)
      .with_rgb(&mut rgb_scalar)
      .unwrap()
      .with_simd(false);
    v30x_to(&src, false, ColorMatrix::Bt709, &mut sink_simd).unwrap();
    v30x_to(&src, false, ColorMatrix::Bt709, &mut sink_scalar).unwrap();
    assert_eq!(rgb_simd, rgb_scalar, "V30X SIMD≠scalar at width {w}");
  }
}

// ---- Planar parity oracle ---------------------------------------------------

/// Pack three 10-bit planes (Y / U / V at 4:4:4, low-bit-packed u16) into
/// V30X word stream layout: bits[31:22] = V, bits[21:12] = Y, bits[11:2] = U,
/// bits[1:0] = 0 (padding at low end).
/// Yuv444p10 stores 10-bit values as low-bit-packed u16 (high 6 bits zero).
fn pack_yuv444p10_to_v30x(
  y_plane: &[u16],
  u_plane: &[u16],
  v_plane: &[u16],
  width: usize,
  height: usize,
) -> Vec<u32> {
  let mut packed = Vec::with_capacity(width * height);
  for r in 0..height {
    for c in 0..width {
      let y = (y_plane[r * width + c] & 0x3FF) as u32;
      let u = (u_plane[r * width + c] & 0x3FF) as u32;
      let v = (v_plane[r * width + c] & 0x3FF) as u32;
      packed.push((v << 22) | (y << 12) | (u << 2));
    }
  }
  packed
}

#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn v30x_planar_parity_with_yuv444p10() {
  // Oracle: Yuv444p10 (separate planes) and V30X (packed u32) carry
  // identical logical 10-bit samples — both paths MUST produce
  // byte-identical RGB output (u8 and u16).
  let width = 16usize;
  let height = 4usize;
  let mut yp = std::vec![0u16; width * height];
  let mut up = std::vec![0u16; width * height];
  let mut vp = std::vec![0u16; width * height];
  pseudo_random_u16_low_n_bits(&mut yp, 0xC0FFEE, 10);
  pseudo_random_u16_low_n_bits(&mut up, 0xBADF00D, 10);
  pseudo_random_u16_low_n_bits(&mut vp, 0xFEEDFACE, 10);

  let planar = Yuv444p10Frame::new(
    &yp,
    &up,
    &vp,
    width as u32,
    height as u32,
    width as u32,
    width as u32,
    width as u32,
  );
  let packed = pack_yuv444p10_to_v30x(&yp, &up, &vp, width, height);
  let v30x = V30XFrame::new(&packed, width as u32, height as u32, width as u32);

  // u8 RGB parity
  let mut p_rgb = std::vec![0u8; width * height * 3];
  let mut v_rgb = std::vec![0u8; width * height * 3];
  let mut p_sink = MixedSinker::<Yuv444p10>::new(width, height)
    .with_rgb(&mut p_rgb)
    .unwrap();
  let mut v_sink = MixedSinker::<V30X>::new(width, height)
    .with_rgb(&mut v_rgb)
    .unwrap();
  yuv444p10_to(&planar, false, ColorMatrix::Bt709, &mut p_sink).unwrap();
  v30x_to(&v30x, false, ColorMatrix::Bt709, &mut v_sink).unwrap();
  assert_eq!(p_rgb, v_rgb, "V30X ↔ Yuv444p10 u8 RGB diverges");

  // u16 RGB parity (validates the low-bit-packed 10-bit path)
  let mut p_rgb_u16 = std::vec![0u16; width * height * 3];
  let mut v_rgb_u16 = std::vec![0u16; width * height * 3];
  let mut p_sink2 = MixedSinker::<Yuv444p10>::new(width, height)
    .with_rgb_u16(&mut p_rgb_u16)
    .unwrap();
  let mut v_sink2 = MixedSinker::<V30X>::new(width, height)
    .with_rgb_u16(&mut v_rgb_u16)
    .unwrap();
  yuv444p10_to(&planar, false, ColorMatrix::Bt709, &mut p_sink2).unwrap();
  v30x_to(&v30x, false, ColorMatrix::Bt709, &mut v_sink2).unwrap();
  assert_eq!(p_rgb_u16, v_rgb_u16, "V30X ↔ Yuv444p10 u16 RGB diverges");
}

// ---- Error-path tests --------------------------------------------------

#[test]
#[cfg(all(test, feature = "std"))]
fn v30x_process_rejects_short_packed_slice() {
  // 6-pixel-wide sink expects 6 u32 elements per row; a 5-element
  // slice surfaces as `RowShapeMismatch { which: V30XPacked, .. }`.
  let mut rgb = std::vec![0u8; 6 * 3];
  let mut sink = MixedSinker::<V30X>::new(6, 1).with_rgb(&mut rgb).unwrap();
  let packed = [0u32; 5];
  let row = V30XRow::new(&packed, 0, ColorMatrix::Bt601, true);
  let err = sink.process(row).err().unwrap();
  assert_eq!(
    err,
    MixedSinkerError::RowShapeMismatch {
      which: RowSlice::V30XPacked,
      row: 0,
      expected: 6,
      actual: 5,
    }
  );
}

#[test]
#[cfg(all(test, feature = "std"))]
fn v30x_process_rejects_row_index_out_of_range() {
  // Sink configured for 1 row; passing row index 1 must return
  // RowIndexOutOfRange before any kernel runs.
  let mut rgb = std::vec![0u8; 4 * 3];
  let mut sink = MixedSinker::<V30X>::new(4, 1).with_rgb(&mut rgb).unwrap();
  let packed = [0u32; 4];
  let row = V30XRow::new(&packed, 1, ColorMatrix::Bt601, true);
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
fn v30x_rgba_u16_buffer_too_short_returns_err() {
  // Buffer holds 6×7 = 42 elements × 4 channels = 168 u16 elements;
  // a 6×8 frame needs 6×8×4 = 192.
  let mut rgba = std::vec![0u16; 6 * 7 * 4];
  let result = MixedSinker::<V30X>::new(6, 8).with_rgba_u16(&mut rgba);
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
