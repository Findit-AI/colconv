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
/// 16-byte word holds 12 x 10-bit samples laid out per the spec:
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

/// Pack 12 x 10-bit samples into a 16-byte v210 word per the spec
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
  v210_to(&src, true, KernelMatrix::Bt601, &mut sink).unwrap();
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
  v210_to(&src, true, KernelMatrix::Bt601, &mut sink).unwrap();
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
  v210_to(&src, true, KernelMatrix::Bt601, &mut sink).unwrap();
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
  v210_to(&src, true, KernelMatrix::Bt601, &mut sink).unwrap();
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
  v210_to(&src, true, KernelMatrix::Bt601, &mut sink).unwrap();
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
  v210_to(&src, true, KernelMatrix::Bt601, &mut sink).unwrap();
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
  v210_to(&src, true, KernelMatrix::Bt601, &mut sink).unwrap();
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
  v210_to(&src, true, KernelMatrix::Bt601, &mut sink).unwrap();
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
  // word valid (10-bit fields x 3; the top 2 bits are unused per the
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
    v210_to(&src, false, KernelMatrix::Bt709, &mut sink_simd).unwrap();
    v210_to(&src, false, KernelMatrix::Bt709, &mut sink_scalar).unwrap();
    assert_eq!(rgb_simd, rgb_scalar, "V210 SIMD≠scalar at width {w}");
  }
}

// ---- Error-path tests --------------------------------------------------

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
  v210_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
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
fn v210_luma_u16_buffer_too_short_returns_err() {
  // Buffer holds 6x7 = 42 elements; a 6x8 frame needs 48.
  let mut luma = std::vec![0u16; 6 * 7];
  let result = MixedSinker::<V210>::new(6, 8).with_luma_u16(&mut luma);
  let Err(err) = result else {
    panic!("expected InsufficientLumaU16Buffer");
  };
  assert_eq!(
    err,
    MixedSinkerError::InsufficientLumaU16Buffer(InsufficientBuffer::new(48, 42))
  );
}

// ---- Planar parity oracle (Task 12) -----------------------------------
//
// Re-pack a Yuv422p10 source into v210 layout and verify both produce
// byte-identical RGB. This is a cross-format invariant: v210 is just a
// packed byte-stream representation of 4:2:2 10-bit planar data, so the
// converted output MUST match Yuv422p10's output for the same samples.
// Gated `yuv-planar` (the planar source) so the v210-solo `--tests`
// build still compiles.

/// Pack three 10-bit planes (Y / U / V at 4:2:2 subsampling) into the
/// v210 byte layout — see `pack_v210_word_for_test` for the per-word
/// bit ordering. Width must be a multiple of 6.
///
/// Shared by the v210 planar-parity oracle (`yuv-planar`) and the
/// Y210-vs-V210 cross-test (`y2xx`), so it is available under either.
#[cfg(any(feature = "yuv-planar", feature = "y2xx"))]
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

#[cfg(feature = "yuv-planar")]
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
  yuv422p10_to(&planar, false, KernelMatrix::Bt709, &mut s_planar).unwrap();
  v210_to(&v210, false, KernelMatrix::Bt709, &mut s_packed).unwrap();

  assert_eq!(rgb_planar, rgb_packed);
}
// Phase 4 — Frame BE flag, Tier 4 V210 LE/BE round-trip parity test.
//
// V210 packs 12 x 10-bit samples into a 16-byte word as four 32-bit LE u32s.
// `<const BE>` swaps each 32-bit word's bytes; the kernel reads them via
// `load_endian_u32::<BE>` so the sample bits land in the same logical
// positions regardless of wire byte order.
//
// Pattern mirrors the Y210 / Y212 / Y216 LE/BE tests; see
// `sinker/mixed/tests/y210.rs` for the rationale.
/// Re-encode a v210 plane as **BE-encoded** byte storage by byte-swapping
/// each 32-bit word in-place. The kernel's `load_endian_u32::<true>` undoes
/// the swap to recover the same 10-bit samples.
fn v210_as_be(plane_le: &[u8]) -> Vec<u8> {
  assert_eq!(
    plane_le.len() % 4,
    0,
    "v210 plane length must be word-aligned"
  );
  let mut out = Vec::with_capacity(plane_le.len());
  for chunk in plane_le.chunks_exact(4) {
    let w = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    out.extend_from_slice(&w.to_be_bytes());
  }
  out
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn v210_le_be_roundtrip_byte_identical() {
  // Mid-range 10-bit samples (Y=600, U=400, V=700).
  let intended = solid_v210_frame(12, 4, 600, 400, 700);
  let pix_le = intended.clone();
  let pix_be = v210_as_be(&intended);

  // 12-pixel row at canonical 16-byte word size: ceil(12/6)*16 = 32 bytes.
  let stride: u32 = 32;
  let frame_le = V210LeFrame::try_new(&pix_le, 12, 4, stride).unwrap();
  let mut out_le_rgba = std::vec![0u8; 12 * 4 * 4];
  let mut out_le_luma_u16 = std::vec![0u16; 12 * 4];
  let mut sink_le = MixedSinker::<V210>::new(12, 4)
    .with_simd(false)
    .with_rgba(&mut out_le_rgba)
    .unwrap()
    .with_luma_u16(&mut out_le_luma_u16)
    .unwrap();
  v210_to(&frame_le, true, KernelMatrix::Bt709, &mut sink_le).unwrap();

  let frame_be = V210BeFrame::try_new(&pix_be, 12, 4, stride).unwrap();
  let mut out_be_rgba = std::vec![0u8; 12 * 4 * 4];
  let mut out_be_luma_u16 = std::vec![0u16; 12 * 4];
  let mut sink_be = MixedSinker::<V210<true>>::new(12, 4)
    .with_simd(false)
    .with_rgba(&mut out_be_rgba)
    .unwrap()
    .with_luma_u16(&mut out_be_luma_u16)
    .unwrap();
  v210_to_endian(&frame_be, true, KernelMatrix::Bt709, &mut sink_be).unwrap();

  assert_eq!(
    out_le_rgba, out_be_rgba,
    "V210 RGBA u8 LE/BE outputs diverge — `<const BE>` propagation broken"
  );
  assert_eq!(
    out_le_luma_u16, out_be_luma_u16,
    "V210 luma u16 LE/BE outputs diverge — `<const BE>` propagation broken"
  );
}

// ---- #263 direct YUV→HSV ----------------------------------------------

/// Row-kernel parity: the direct `v210_to_hsv_row_endian` dispatcher is
/// byte-identical to `rgb_to_hsv_row(v210_to_rgb_row_endian(...))` within
/// each tier — scalar (`use_simd = false`) AND host SIMD (`use_simd = true`)
/// — across matrices / range / endianness and a width sweep covering
/// partial-word tails (2 / 4 / 8 / 10 / 14) plus the SIMD HSV driver's
/// word-aligned multi-chunk path (62 / 66 straddle the 60-px chunk). Both
/// sides read the same packed buffer through the same `big_endian` decode, so
/// this proves the fused HSV kernel reproduces the via-RGB pipeline exactly
/// (and, via the RGB tier's own SIMD≡scalar parity, that the SIMD HSV kernel
/// matches scalar).
#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn v210_hsv_row_matches_rgb_then_hsv() {
  use crate::row::{rgb_to_hsv_row, v210_to_hsv_row_endian, v210_to_rgb_row_endian};
  for &w in &[2usize, 4, 6, 8, 10, 12, 14, 60, 62, 66] {
    let mut buf = std::vec![0u8; w.div_ceil(6) * 16];
    pseudo_random_u8(&mut buf, 0x51A7_F00D);
    for &matrix in &[
      KernelMatrix::Bt601,
      KernelMatrix::Bt709,
      KernelMatrix::Bt2020Ncl,
    ] {
      for &full in &[true, false] {
        for &use_simd in &[false, true] {
          for &be in &[false, true] {
            let mut rgb = std::vec![0u8; w * 3];
            v210_to_rgb_row_endian(&buf, &mut rgb, w, matrix, full, use_simd, be);
            let mut rh = std::vec![0u8; w];
            let mut rs = std::vec![0u8; w];
            let mut rv = std::vec![0u8; w];
            rgb_to_hsv_row(&rgb, &mut rh, &mut rs, &mut rv, w, use_simd);

            let mut h = std::vec![0u8; w];
            let mut s = std::vec![0u8; w];
            let mut v = std::vec![0u8; w];
            v210_to_hsv_row_endian(&buf, &mut h, &mut s, &mut v, w, matrix, full, use_simd, be);
            assert_eq!(
              (h, s, v),
              (rh, rs, rv),
              "v210 HSV≠RGB→HSV (w={w} {matrix:?} full={full} simd={use_simd} be={be})"
            );
          }
        }
      }
    }
  }
}

/// Structural: an HSV-only V210 sink (no `with_rgb` / `with_rgba`) routes
/// through the direct kernel and must NOT grow the source-width RGB scratch
/// (`rgb_scratch.len() == 0`); its HSV equals the explicit
/// `v210_to_rgb_row` → `rgb_to_hsv_row` reference (row 0).
#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn v210_hsv_only_is_rgb_free_and_matches_reference() {
  use crate::row::{rgb_to_hsv_row, v210_to_rgb_row};
  let (w, h) = (16usize, 8usize);
  let stride = w.div_ceil(6) * 16;
  let mut buf = std::vec![0u8; stride * h];
  pseudo_random_u8(&mut buf, 0xBEEF_2210);
  let src = V210Frame::new(&buf, w as u32, h as u32, stride as u32);
  let mut hh = std::vec![0u8; w * h];
  let mut ss = std::vec![0u8; w * h];
  let mut vv = std::vec![0u8; w * h];
  let scratch_len = {
    let mut sink = MixedSinker::<V210>::new(w, h)
      .with_hsv(&mut hh, &mut ss, &mut vv)
      .unwrap();
    v210_to(&src, true, KernelMatrix::Bt709, &mut sink).unwrap();
    sink.rgb_scratch.len()
  };
  assert_eq!(
    scratch_len, 0,
    "V210 HSV-only must not grow the RGB scratch"
  );

  let mut rgb0 = std::vec![0u8; w * 3];
  v210_to_rgb_row(
    &buf[..stride],
    &mut rgb0,
    w,
    KernelMatrix::Bt709,
    true,
    true,
  );
  let mut rh = std::vec![0u8; w];
  let mut rs = std::vec![0u8; w];
  let mut rv = std::vec![0u8; w];
  rgb_to_hsv_row(&rgb0, &mut rh, &mut rs, &mut rv, w, true);
  assert_eq!(&hh[..w], &rh[..], "row 0 H");
  assert_eq!(&ss[..w], &rs[..], "row 0 S");
  assert_eq!(&vv[..w], &rv[..], "row 0 V");
}

// ---- Atomicity (#308): RGB-scratch preflight before output writes -------
//
// V210's identity `process` hoists the RGB-scratch reservation above every
// output write (luma, luma_u16, then the u16 RGB / RGBA fan-out), so an
// allocator refusal of the scratch grow surfaces as a recoverable
// `AllocationFailed` BEFORE any output plane is touched. The allocating
// (rgb = None) arm of `rgb_row_buf_or_scratch` is reached only at
// `want_hsv && want_rgba && !want_rgb` (HSV-only routes through the direct
// `v210_to_hsv_row_endian` kernel and never allocates): attach luma + luma_u16
// + rgba + hsv (no rgb) and arm the shared failpoint. `yuva`-gated (shares the
// crate's RGB-scratch failpoint).
#[cfg(feature = "yuva")]
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn v210_rgb_scratch_alloc_failure_leaves_outputs_untouched() {
  use crate::resample::ResampleError;

  let buf = solid_v210_frame(16, 8, 512, 512, 512);
  let src = V210Frame::new(&buf, 16, 8, 48);
  let mut luma = std::vec![0xABu8; 16 * 8];
  let mut luma_u16 = std::vec![0xABCDu16; 16 * 8];
  let mut rgba = std::vec![0xCDu8; 16 * 8 * 4];
  let (mut hh, mut ss, mut vv) = (
    std::vec![0xCDu8; 16 * 8],
    std::vec![0xCDu8; 16 * 8],
    std::vec![0xCDu8; 16 * 8],
  );
  let mut sink = MixedSinker::<V210>::new(16, 8)
    .with_luma(&mut luma)
    .unwrap()
    .with_luma_u16(&mut luma_u16)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap()
    .with_hsv(&mut hh, &mut ss, &mut vv)
    .unwrap();

  super::super::arm_rgb_scratch_alloc_failure();
  let err = v210_to(&src, false, KernelMatrix::Bt601, &mut sink).unwrap_err();
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
  assert!(
    luma_u16.iter().all(|&b| b == 0xABCD),
    "luma_u16 must be untouched on the rgb-scratch alloc-failure path"
  );
}
