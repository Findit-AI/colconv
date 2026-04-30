//! Tier 4 v210 sinker tests — Ship 11a.
//!
//! Coverage matrix:
//! - Single-output paths (luma u8, luma u16, rgb, rgba, rgb_u16,
//!   rgba_u16) on solid-gray frames.
//! - Strategy A invariant (`with_rgb` + `with_rgba` byte-identical;
//!   same for the u16 variants — first packed-source u16 path test
//!   in the crate).
//! - SIMD-vs-scalar parity across multiple widths covering the main
//!   loop + scalar tail of every backend block size.
//! - Three error-path tests: width-not-multiple-of-6, short packed
//!   slice, short luma_u16 buffer.

use super::*;

// ---- Solid-color v210 builder -----------------------------------------

/// Builds a solid-color v210 plane with one (Y, U, V) repeated. Each
/// 16-byte word holds 12 × 10-bit samples laid out per the spec:
///
/// | Word | bits[9:0] | bits[19:10] | bits[29:20] | bits[31:30] |
/// |------|-----------|-------------|-------------|-------------|
/// | 0    | Cb₀       | Y₀          | Cr₀         | unused      |
/// | 1    | Y₁        | Cb₁         | Y₂          | unused      |
/// | 2    | Cr₁       | Y₃          | Cb₂         | unused      |
/// | 3    | Y₄        | Cr₂         | Y₅          | unused      |
///
/// For a solid color, the same Y goes into all 6 Y slots, the same U
/// into all 3 Cb slots, the same V into all 3 Cr slots.
pub(super) fn solid_v210_frame(width: u32, height: u32, y: u16, u: u16, v: u16) -> Vec<u8> {
  // Round up — partial-word widths (e.g. 1280) need the trailing
  // 16-byte block too, even if only 2 or 4 of its samples are valid.
  let words_per_row = width.div_ceil(6) as usize;
  let mut buf = std::vec![0u8; words_per_row * 16 * height as usize];
  let samples: [u16; 12] = [u, y, v, y, u, y, v, y, u, y, v, y];
  let word = pack_v210_word_for_test(samples);
  for row in 0..height as usize {
    for w in 0..words_per_row {
      let off = (row * words_per_row + w) * 16;
      buf[off..off + 16].copy_from_slice(&word);
    }
  }
  buf
}

/// Pack 12 × 10-bit samples into a 16-byte v210 word per the spec
/// table above. The low 30 bits of each LE u32 hold three 10-bit
/// samples; the top 2 bits are unused.
fn pack_v210_word_for_test(samples: [u16; 12]) -> [u8; 16] {
  let mut out = [0u8; 16];
  let pack = |a: u16, b: u16, c: u16| -> u32 {
    (a as u32 & 0x3FF) | ((b as u32 & 0x3FF) << 10) | ((c as u32 & 0x3FF) << 20)
  };
  out[0..4].copy_from_slice(&pack(samples[0], samples[1], samples[2]).to_le_bytes());
  out[4..8].copy_from_slice(&pack(samples[3], samples[4], samples[5]).to_le_bytes());
  out[8..12].copy_from_slice(&pack(samples[6], samples[7], samples[8]).to_le_bytes());
  out[12..16].copy_from_slice(&pack(samples[9], samples[10], samples[11]).to_le_bytes());
  out
}

// ---- Single-output gray-to-gray tests ---------------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn v210_luma_only_extracts_y_bytes() {
  let buf = solid_v210_frame(6, 8, 200, 512, 512); // Y=200 (10-bit native).
  let src = V210Frame::new(&buf, 6, 8, 16);
  let mut luma = std::vec![0u8; 6 * 8];
  let mut sink = MixedSinker::<V210>::new(6, 8).with_luma(&mut luma).unwrap();
  v210_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();
  // 10-bit Y=200 → 8-bit (200 >> 2) = 50.
  assert!(luma.iter().all(|&y| y == 50), "luma {luma:?}");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn v210_luma_u16_only_extracts_y_native_depth() {
  let buf = solid_v210_frame(6, 8, 200, 512, 512);
  let src = V210Frame::new(&buf, 6, 8, 16);
  let mut luma = std::vec![0u16; 6 * 8];
  let mut sink = MixedSinker::<V210>::new(6, 8)
    .with_luma_u16(&mut luma)
    .unwrap();
  v210_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();
  assert!(luma.iter().all(|&y| y == 200), "luma_u16 {:?}", &luma[..16]);
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn v210_rgb_only_converts_gray_to_gray() {
  let buf = solid_v210_frame(12, 4, 512, 512, 512);
  let src = V210Frame::new(&buf, 12, 4, 32);
  let mut rgb = std::vec![0u8; 12 * 4 * 3];
  let mut sink = MixedSinker::<V210>::new(12, 4).with_rgb(&mut rgb).unwrap();
  v210_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();
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
fn v210_rgba_only_converts_gray_to_gray_with_opaque_alpha() {
  let buf = solid_v210_frame(12, 4, 512, 512, 512);
  let src = V210Frame::new(&buf, 12, 4, 32);
  let mut rgba = std::vec![0u8; 12 * 4 * 4];
  let mut sink = MixedSinker::<V210>::new(12, 4)
    .with_rgba(&mut rgba)
    .unwrap();
  v210_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();
  for px in rgba.chunks(4) {
    assert_eq!(px[3], 0xFF);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn v210_rgb_u16_only_converts_gray_to_gray_native_depth() {
  let buf = solid_v210_frame(12, 4, 512, 512, 512);
  let src = V210Frame::new(&buf, 12, 4, 32);
  let mut rgb = std::vec![0u16; 12 * 4 * 3];
  let mut sink = MixedSinker::<V210>::new(12, 4)
    .with_rgb_u16(&mut rgb)
    .unwrap();
  v210_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();
  for px in rgb.chunks(3) {
    assert!(px[0].abs_diff(512) <= 2, "expected ~512, got {}", px[0]);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn v210_rgba_u16_alpha_is_max() {
  let buf = solid_v210_frame(12, 4, 512, 512, 512);
  let src = V210Frame::new(&buf, 12, 4, 32);
  let mut rgba = std::vec![0u16; 12 * 4 * 4];
  let mut sink = MixedSinker::<V210>::new(12, 4)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  v210_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();
  for px in rgba.chunks(4) {
    assert_eq!(px[3], 1023);
  }
}

// ---- Strategy A invariant tests ---------------------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn v210_with_rgb_and_with_rgba_byte_identical_u8() {
  // Strategy A invariant on u8 path — calling both `with_rgb` and
  // `with_rgba` must produce the same RGB bytes in both buffers, with
  // alpha = 0xFF in the RGBA buffer.
  let w = 12u32;
  let h = 4u32;
  let buf = solid_v210_frame(w, h, 700, 400, 600);
  let src = V210Frame::new(&buf, w, h, (w / 6) * 16);
  let mut rgb = std::vec![0u8; (w * h) as usize * 3];
  let mut rgba = std::vec![0u8; (w * h) as usize * 4];
  let mut sink = MixedSinker::<V210>::new(w as usize, h as usize)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap();
  v210_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();
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
fn v210_with_rgb_u16_and_with_rgba_u16_byte_identical() {
  // Strategy A invariant on u16 path — first packed-source consumer
  // of `expand_rgb_u16_to_rgba_u16_row::<10>` in the sinker family.
  let w = 12u32;
  let h = 4u32;
  let buf = solid_v210_frame(w, h, 700, 400, 600);
  let src = V210Frame::new(&buf, w, h, (w / 6) * 16);
  let mut rgb = std::vec![0u16; (w * h) as usize * 3];
  let mut rgba = std::vec![0u16; (w * h) as usize * 4];
  let mut sink = MixedSinker::<V210>::new(w as usize, h as usize)
    .with_rgb_u16(&mut rgb)
    .unwrap()
    .with_rgba_u16(&mut rgba)
    .unwrap();
  v210_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();
  for i in 0..(w * h) as usize {
    assert_eq!(rgba[i * 4], rgb[i * 3]);
    assert_eq!(rgba[i * 4 + 1], rgb[i * 3 + 1]);
    assert_eq!(rgba[i * 4 + 2], rgb[i * 3 + 2]);
    assert_eq!(rgba[i * 4 + 3], 1023);
  }
}

// ---- SIMD-vs-scalar parity --------------------------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn v210_with_simd_false_matches_with_simd_true() {
  // Pseudo-random v210 across multiple widths covering the main loop
  // + scalar tail of every backend block size, plus partial-word
  // widths (2, 4, 8, 10, 14, 1280) that exercise the partial-tail
  // emitter. Mask packed buffer to keep the low 30 bits of each u32
  // word valid (10-bit fields × 3; the top 2 bits are unused per the
  // v210 spec).
  for w in [
    2usize, 4, 6, 8, 10, 12, 14, 18, 24, 30, 1280, 1920, 1922, 1926,
  ] {
    let h = 2usize;
    let mut buf = std::vec![0u8; w.div_ceil(6) * 16 * h];
    pseudo_random_u8(&mut buf, 0xC0FFEE);
    // Each 4-byte LE u32 has its top 2 bits unused (bits[31:30]).
    // Byte 3 of each word holds bits[31:24]; mask off the top 2 to
    // keep only bits[29:24] (the valid top of the 30-bit payload).
    for i in (0..buf.len()).step_by(4) {
      buf[i + 3] &= 0x3F;
    }
    let src = V210Frame::new(&buf, w as u32, h as u32, (w.div_ceil(6) * 16) as u32);

    let mut rgb_simd = std::vec![0u8; w * h * 3];
    let mut rgb_scalar = std::vec![0u8; w * h * 3];
    let mut sink_simd = MixedSinker::<V210>::new(w, h)
      .with_rgb(&mut rgb_simd)
      .unwrap();
    let mut sink_scalar = MixedSinker::<V210>::new(w, h)
      .with_rgb(&mut rgb_scalar)
      .unwrap()
      .with_simd(false);
    v210_to(&src, false, ColorMatrix::Bt709, &mut sink_simd).unwrap();
    v210_to(&src, false, ColorMatrix::Bt709, &mut sink_scalar).unwrap();
    assert_eq!(rgb_simd, rgb_scalar, "V210 SIMD≠scalar at width {w}");
  }
}

// ---- Error-path tests --------------------------------------------------

#[test]
fn v210_odd_width_returns_err() {
  // Direct `process()` call with a sink configured at width=7 (odd —
  // violates 4:2:2 chroma-pair constraint). Even widths that aren't
  // multiples of 6 are now supported (partial-word handling), so this
  // covers only the genuinely-invalid odd-width case. The width check
  // fires *before* any kernel runs, preserving the no-panic contract
  // — even if the caller bypasses the walker (which would catch this
  // in `begin_frame`).
  let mut rgb = std::vec![0u8; 8 * 8 * 3];
  let mut sink = MixedSinker::<V210>::new(7, 8).with_rgb(&mut rgb).unwrap();
  let buf = std::vec![0u8; 16];
  let row = V210Row::new(&buf, 0, ColorMatrix::Bt601, true);
  let err = sink.process(row).err().unwrap();
  assert!(matches!(err, MixedSinkerError::OddWidth { width: 7 }));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn v210_partial_word_width_works_end_to_end() {
  // 720p-sized solid gray (Y=512, U=V=512) at width = 1280 (partial
  // last word: 213 full + 1 partial holding 2 valid px). The frame +
  // walker + sinker + scalar/SIMD kernels must all agree and produce
  // a uniform mid-gray RGB output across every pixel.
  let buf = solid_v210_frame(1280, 1, 512, 512, 512);
  let stride = 1280u32.div_ceil(6) * 16;
  let src = V210Frame::new(&buf, 1280, 1, stride);
  let mut rgb = std::vec![0u8; 1280 * 3];
  let mut sink = MixedSinker::<V210>::new(1280, 1)
    .with_rgb(&mut rgb)
    .unwrap();
  v210_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();
  for px in rgb.chunks(3) {
    assert!(
      px[0].abs_diff(128) <= 1,
      "partial-word RGB diverged: {px:?}"
    );
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
  }
}

#[test]
fn v210_process_rejects_short_packed_slice() {
  // 6-pixel-wide sink expects 16 bytes per row; a 15-byte slice
  // surfaces as `RowShapeMismatch { which: V210Packed, .. }`.
  let mut rgb = std::vec![0u8; 6 * 3];
  let mut sink = MixedSinker::<V210>::new(6, 1).with_rgb(&mut rgb).unwrap();
  let packed = [0u8; 15];
  let row = V210Row::new(&packed, 0, ColorMatrix::Bt601, true);
  let err = sink.process(row).err().unwrap();
  assert_eq!(
    err,
    MixedSinkerError::RowShapeMismatch {
      which: RowSlice::V210Packed,
      row: 0,
      expected: 16,
      actual: 15,
    }
  );
}

#[test]
fn v210_luma_u16_buffer_too_short_returns_err() {
  // Buffer holds 6×7 = 42 elements; a 6×8 frame needs 48.
  let mut luma = std::vec![0u16; 6 * 7];
  let result = MixedSinker::<V210>::new(6, 8).with_luma_u16(&mut luma);
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

// ---- Planar parity oracle (Task 12) -----------------------------------
//
// Re-pack a Yuv422p10 source into v210 layout and verify both produce
// byte-identical RGB. This is a cross-format invariant: v210 is just a
// packed byte-stream representation of 4:2:2 10-bit planar data, so the
// converted output MUST match Yuv422p10's output for the same samples.

/// Pack three 10-bit planes (Y / U / V at 4:2:2 subsampling) into the
/// v210 byte layout — see `pack_v210_word_for_test` for the per-word
/// bit ordering. Width must be a multiple of 6.
pub(super) fn pack_yuv422p10_to_v210(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  width: usize,
  height: usize,
) -> Vec<u8> {
  let cw = width / 2;
  let words_per_row = width / 6;
  let mut out = std::vec![0u8; words_per_row * 16 * height];
  for row in 0..height {
    for w in 0..words_per_row {
      let px = w * 6;
      let cu = px / 2;
      let samples: [u16; 12] = [
        u[row * cw + cu],
        y[row * width + px],
        v[row * cw + cu],
        y[row * width + px + 1],
        u[row * cw + cu + 1],
        y[row * width + px + 2],
        v[row * cw + cu + 1],
        y[row * width + px + 3],
        u[row * cw + cu + 2],
        y[row * width + px + 4],
        v[row * cw + cu + 2],
        y[row * width + px + 5],
      ];
      let bytes = pack_v210_word_for_test(samples);
      let off = (row * words_per_row + w) * 16;
      out[off..off + 16].copy_from_slice(&bytes);
    }
  }
  out
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn v210_reconstructed_from_yuv422p10_matches_yuv422p10_to_rgb() {
  let w = 12usize;
  let h = 4usize;
  let mut y_plane = std::vec![0u16; w * h];
  let mut u_plane = std::vec![0u16; (w / 2) * h];
  let mut v_plane = std::vec![0u16; (w / 2) * h];
  pseudo_random_u16_low_n_bits(&mut y_plane, 0xC0FFEE, 10);
  pseudo_random_u16_low_n_bits(&mut u_plane, 0xBADF00D, 10);
  pseudo_random_u16_low_n_bits(&mut v_plane, 0xFEEDFACE, 10);

  let planar = Yuv422p10Frame::new(
    &y_plane,
    &u_plane,
    &v_plane,
    w as u32,
    h as u32,
    w as u32,
    (w / 2) as u32,
    (w / 2) as u32,
  );
  let packed = pack_yuv422p10_to_v210(&y_plane, &u_plane, &v_plane, w, h);
  let v210 = V210Frame::new(&packed, w as u32, h as u32, ((w / 6) * 16) as u32);

  let mut rgb_planar = std::vec![0u8; w * h * 3];
  let mut rgb_packed = std::vec![0u8; w * h * 3];
  let mut s_planar = MixedSinker::<Yuv422p10>::new(w, h)
    .with_rgb(&mut rgb_planar)
    .unwrap();
  let mut s_packed = MixedSinker::<V210>::new(w, h)
    .with_rgb(&mut rgb_packed)
    .unwrap();
  yuv422p10_to(&planar, false, ColorMatrix::Bt709, &mut s_planar).unwrap();
  v210_to(&v210, false, ColorMatrix::Bt709, &mut s_packed).unwrap();

  assert_eq!(rgb_planar, rgb_packed);
}
