//! Tier 5 XV48 sinker tests.
//!
//! Coverage matrix:
//! - Single-output paths (rgb, rgba, rgb_u16, rgba_u16, luma u8,
//!   luma u16) on solid-gray frames.
//! - Strategy A invariant (`with_rgb` + `with_rgba` byte-identical;
//!   same for the u16 variants with BITS=16).
//! - SIMD-vs-scalar parity across multiple widths covering the main
//!   loop + scalar tail of every backend block size.
//! - Error-path tests: short rgba_u16 buffer.
//!
//! XV48 specifics vs XV36:
//! - Channels are full 16-bit native (no MSB shift — XV36 is 12-bit
//!   MSB-aligned).
//! - `with_luma_u16` passes Y straight through (no `>> 4`).
//! - u16 alpha max = `0xFFFF` (16-bit max, not 0x0FFF).
//! - The u16 RGB path uses i64 chroma (like AYUV64).

#[cfg(all(test, feature = "std"))]
use super::*;

// ---- Solid-color XV48 builder -----------------------------------------

/// Builds a solid-color XV48 plane with one (U, Y, V, X) quadruple
/// repeated. Each pixel is stored as four u16 words (full 16-bit native,
/// no shift):
///   word 0: U   word 1: Y   word 2: V   word 3: X (padding)
///
/// Row stride equals `width x 4` u16 elements (no padding between rows).
#[cfg(all(test, feature = "std"))]
pub(super) fn solid_xv48_frame(
  width: u32,
  height: u32,
  u: u16,
  y: u16,
  v: u16,
  x: u16,
) -> Vec<u16> {
  let quad = [u, y, v, x];
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
fn xv48_rgb_only_converts_gray_to_gray() {
  // Y=U=V=0x8000 (16-bit midpoint). After full-range YUV→RGB the
  // per-channel u8 value should be near 128 ± tolerance.
  let buf = solid_xv48_frame(12, 4, 0x8000, 0x8000, 0x8000, 0);
  let src = Xv48Frame::new(&buf, 12, 4, 48);
  let mut rgb = std::vec![0u8; 12 * 4 * 3];
  let mut sink = MixedSinker::<Xv48>::new(12, 4).with_rgb(&mut rgb).unwrap();
  xv48_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
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
fn xv48_rgba_only_converts_gray_to_gray_with_opaque_alpha() {
  // Alpha must be forced to 0xFF (XV48 X slot is padding).
  let buf = solid_xv48_frame(12, 4, 0x8000, 0x8000, 0x8000, 0);
  let src = Xv48Frame::new(&buf, 12, 4, 48);
  let mut rgba = std::vec![0u8; 12 * 4 * 4];
  let mut sink = MixedSinker::<Xv48>::new(12, 4)
    .with_rgba(&mut rgba)
    .unwrap();
  xv48_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
  for px in rgba.chunks(4) {
    assert_eq!(px[3], 0xFF, "alpha must be 0xFF for XV48");
  }
}

#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn xv48_rgb_u16_only_converts_gray_to_gray_native_depth() {
  // Y=U=V=0x8000 (16-bit midpoint). After YUV→RGB at 16-bit depth the
  // per-channel u16 value should be near 0x8000; allow ±0x100 for Q15
  // rounding.
  let buf = solid_xv48_frame(12, 4, 0x8000, 0x8000, 0x8000, 0);
  let src = Xv48Frame::new(&buf, 12, 4, 48);
  let mut rgb = std::vec![0u16; 12 * 4 * 3];
  let mut sink = MixedSinker::<Xv48>::new(12, 4)
    .with_rgb_u16(&mut rgb)
    .unwrap();
  xv48_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
  for px in rgb.chunks(3) {
    assert!(
      px[0].abs_diff(0x8000) <= 0x100,
      "expected ~0x8000, got {:#X}",
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
fn xv48_rgba_u16_alpha_is_max() {
  // 16-bit alpha max = 0xFFFF (NOT 0x0FFF or 0x3FF).
  let buf = solid_xv48_frame(12, 4, 0x8000, 0x8000, 0x8000, 0);
  let src = Xv48Frame::new(&buf, 12, 4, 48);
  let mut rgba = std::vec![0u16; 12 * 4 * 4];
  let mut sink = MixedSinker::<Xv48>::new(12, 4)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  xv48_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
  for px in rgba.chunks(4) {
    assert_eq!(px[3], 0xFFFF, "u16 alpha must be 0xFFFF for XV48");
  }
}

#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn xv48_luma_only_extracts_y_bytes_downshifted() {
  // Y=0xF000 (16-bit). u8 luma: 0xF000 >> 8 = 0xF0 = 240.
  let buf = solid_xv48_frame(6, 8, 0x8000, 0xF000, 0x8000, 0);
  let src = Xv48Frame::new(&buf, 6, 8, 24);
  let mut luma = std::vec![0u8; 6 * 8];
  let mut sink = MixedSinker::<Xv48>::new(6, 8).with_luma(&mut luma).unwrap();
  xv48_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
  assert!(luma.iter().all(|&y| y == 0xF0), "luma {luma:?}");
}

#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn xv48_luma_u16_only_extracts_y_native_depth() {
  // Y=0xABCD (16-bit). u16 luma_u16: written direct (no shift) = 0xABCD.
  let buf = solid_xv48_frame(6, 8, 0x8000, 0xABCD, 0x8000, 0);
  let src = Xv48Frame::new(&buf, 6, 8, 24);
  let mut luma = std::vec![0u16; 6 * 8];
  let mut sink = MixedSinker::<Xv48>::new(6, 8)
    .with_luma_u16(&mut luma)
    .unwrap();
  xv48_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
  assert!(
    luma.iter().all(|&y| y == 0xABCD),
    "luma_u16 expected 0xABCD, got {:?}",
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
fn xv48_with_rgb_and_with_rgba_byte_identical_u8() {
  // Strategy A invariant on u8 path — calling both `with_rgb` and
  // `with_rgba` must produce the same RGB bytes in both buffers, with
  // alpha = 0xFF in the RGBA buffer.
  let w = 12u32;
  let h = 4u32;
  let buf = solid_xv48_frame(w, h, 0x3000, 0x7000, 0x5000, 0);
  let src = Xv48Frame::new(&buf, w, h, w * 4);
  let mut rgb = std::vec![0u8; (w * h) as usize * 3];
  let mut rgba = std::vec![0u8; (w * h) as usize * 4];
  let mut sink = MixedSinker::<Xv48>::new(w as usize, h as usize)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap();
  xv48_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
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
fn xv48_with_rgb_u16_and_with_rgba_u16_byte_identical() {
  // Strategy A invariant on u16 path — alpha must be 0xFFFF (16-bit max).
  let w = 12u32;
  let h = 4u32;
  let buf = solid_xv48_frame(w, h, 0x3000, 0x7000, 0x5000, 0);
  let src = Xv48Frame::new(&buf, w, h, w * 4);
  let mut rgb = std::vec![0u16; (w * h) as usize * 3];
  let mut rgba = std::vec![0u16; (w * h) as usize * 4];
  let mut sink = MixedSinker::<Xv48>::new(w as usize, h as usize)
    .with_rgb_u16(&mut rgb)
    .unwrap()
    .with_rgba_u16(&mut rgba)
    .unwrap();
  xv48_to(&src, true, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap();
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
    assert_eq!(rgba[i * 4 + 3], 0xFFFF, "alpha must be 0xFFFF at pixel {i}");
  }
}

// ---- SIMD-vs-scalar parity --------------------------------------------

#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn xv48_with_simd_false_matches_with_simd_true() {
  // Pseudo-random XV48 across multiple widths covering the main loop
  // + scalar tail of every backend block size. XV48 samples are full
  // 16-bit (no shift).
  for w in [
    1usize, 2, 4, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 1920, 1922,
  ] {
    let h = 2usize;
    let mut buf = std::vec![0u16; w * h * 4];
    let mut state = 0xC0FFEE_u32;
    for slot in &mut buf {
      state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
      *slot = (state >> 8) as u16; // full 16-bit value
    }
    let src = Xv48Frame::new(&buf, w as u32, h as u32, (w * 4) as u32);

    // u8 RGB
    let mut rgb_simd = std::vec![0u8; w * h * 3];
    let mut rgb_scalar = std::vec![0u8; w * h * 3];
    let mut sink_simd = MixedSinker::<Xv48>::new(w, h)
      .with_rgb(&mut rgb_simd)
      .unwrap();
    let mut sink_scalar = MixedSinker::<Xv48>::new(w, h)
      .with_rgb(&mut rgb_scalar)
      .unwrap()
      .with_simd(false);
    xv48_to(
      &src,
      false,
      sink_simd.set_kernel_matrix(KernelMatrix::Bt709),
    )
    .unwrap();
    xv48_to(
      &src,
      false,
      sink_scalar.set_kernel_matrix(KernelMatrix::Bt709),
    )
    .unwrap();
    assert_eq!(rgb_simd, rgb_scalar, "XV48 u8 SIMD≠scalar at width {w}");

    // u16 RGB (exercises the i64 chroma path)
    let mut rgb16_simd = std::vec![0u16; w * h * 3];
    let mut rgb16_scalar = std::vec![0u16; w * h * 3];
    let mut s2 = MixedSinker::<Xv48>::new(w, h)
      .with_rgb_u16(&mut rgb16_simd)
      .unwrap();
    let mut s2_scalar = MixedSinker::<Xv48>::new(w, h)
      .with_rgb_u16(&mut rgb16_scalar)
      .unwrap()
      .with_simd(false);
    xv48_to(&src, false, s2.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
    xv48_to(
      &src,
      false,
      s2_scalar.set_kernel_matrix(KernelMatrix::Bt709),
    )
    .unwrap();
    assert_eq!(
      rgb16_simd, rgb16_scalar,
      "XV48 u16 SIMD≠scalar at width {w}"
    );
  }
}

// ---- Planar parity oracle ---------------------------------------------------

/// Pack three 16-bit planes (Y / U / V at 4:4:4) into XV48 quadruple
/// layout: `[u, y, v, 0]` per pixel (full 16-bit, no shift). Yuv444p16
/// and XV48 carry identical logical 16-bit samples.
///
/// Gated on `yuv-planar`: the cross-format oracle below feeds the planar
/// `Yuv444p16` source, absent in a `yuv-444-packed`-solo build.
#[cfg(feature = "yuv-planar")]
fn pack_yuv444p16_to_xv48(
  y_plane: &[u16],
  u_plane: &[u16],
  v_plane: &[u16],
  width: usize,
  height: usize,
) -> Vec<u16> {
  let mut packed = Vec::with_capacity(width * 4 * height);
  for r in 0..height {
    for c in 0..width {
      packed.push(u_plane[r * width + c]);
      packed.push(y_plane[r * width + c]);
      packed.push(v_plane[r * width + c]);
      packed.push(0); // padding X
    }
  }
  packed
}

// Cross-format oracle vs the planar `Yuv444p16` source — gated on
// `yuv-planar` so a `yuv-444-packed`-solo `--tests` build (which lacks the
// planar frame / walker) still compiles.
#[test]
#[cfg(all(test, feature = "std", feature = "yuv-planar"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn xv48_planar_parity_with_yuv444p16() {
  // Oracle: Yuv444p16 (separate planes) and XV48 (packed u16 quadruples)
  // carry identical logical 16-bit samples — both paths MUST produce
  // byte-identical RGB output (u8 and u16).
  let width = 16usize;
  let height = 4usize;
  let mut yp = std::vec![0u16; width * height];
  let mut up = std::vec![0u16; width * height];
  let mut vp = std::vec![0u16; width * height];
  pseudo_random_u16_low_n_bits(&mut yp, 0xC0FFEE, 16);
  pseudo_random_u16_low_n_bits(&mut up, 0xBADF00D, 16);
  pseudo_random_u16_low_n_bits(&mut vp, 0xFEEDFACE, 16);

  let planar = Yuv444p16Frame::new(
    &yp,
    &up,
    &vp,
    width as u32,
    height as u32,
    width as u32,
    width as u32,
    width as u32,
  );
  let packed = pack_yuv444p16_to_xv48(&yp, &up, &vp, width, height);
  let xv48 = Xv48Frame::new(&packed, width as u32, height as u32, (width * 4) as u32);

  // u8 RGB parity
  let mut p_rgb = std::vec![0u8; width * height * 3];
  let mut x_rgb = std::vec![0u8; width * height * 3];
  let mut p_sink = MixedSinker::<Yuv444p16>::new(width, height)
    .with_rgb(&mut p_rgb)
    .unwrap();
  let mut x_sink = MixedSinker::<Xv48>::new(width, height)
    .with_rgb(&mut x_rgb)
    .unwrap();
  yuv444p16_to(
    &planar,
    false,
    p_sink.set_kernel_matrix(KernelMatrix::Bt709),
  )
  .unwrap();
  xv48_to(&xv48, false, x_sink.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
  assert_eq!(p_rgb, x_rgb, "XV48 ↔ Yuv444p16 u8 RGB diverges");

  // u16 RGB parity (validates the BITS=16 i64-chroma path against the
  // established planar reference)
  let mut p_rgb_u16 = std::vec![0u16; width * height * 3];
  let mut x_rgb_u16 = std::vec![0u16; width * height * 3];
  let mut p_sink2 = MixedSinker::<Yuv444p16>::new(width, height)
    .with_rgb_u16(&mut p_rgb_u16)
    .unwrap();
  let mut x_sink2 = MixedSinker::<Xv48>::new(width, height)
    .with_rgb_u16(&mut x_rgb_u16)
    .unwrap();
  yuv444p16_to(
    &planar,
    false,
    p_sink2.set_kernel_matrix(KernelMatrix::Bt709),
  )
  .unwrap();
  xv48_to(&xv48, false, x_sink2.set_kernel_matrix(KernelMatrix::Bt709)).unwrap();
  assert_eq!(p_rgb_u16, x_rgb_u16, "XV48 ↔ Yuv444p16 u16 RGB diverges");
}

// ---- Error-path tests --------------------------------------------------

#[test]
#[cfg(all(test, feature = "std"))]
fn xv48_buffer_too_short_for_rgba_u16_returns_err() {
  // Buffer holds 6x7 = 42 elements x 4 channels = 168 u16 elements;
  // a 6x8 frame needs 6x8x4 = 192.
  let mut rgba = std::vec![0u16; 6 * 7 * 4];
  let result = MixedSinker::<Xv48>::new(6, 8).with_rgba_u16(&mut rgba);
  let Err(err) = result else {
    panic!("expected InsufficientRgbaU16Buffer");
  };
  assert_eq!(
    err,
    MixedSinkerError::InsufficientRgbaU16Buffer(InsufficientBuffer::new(192, 168))
  );
}

// XV48 LE+BE round-trip parity test.
//
// XV48 packs each pixel as four full-16-bit u16 channels (`U, Y, V, X`).
// The BE wire variant byte-swaps each u16 channel before extraction.
// Encoding the same logical samples as LE bytes vs BE bytes and feeding
// through `MixedSinker<Xv48<false>>` vs `MixedSinker<Xv48<true>>` must
// produce byte-identical output.
#[test]
#[cfg(all(test, feature = "std"))]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn xv48_le_be_roundtrip_byte_identical() {
  // Build a logical (U, Y, V, X) full-16-bit u16 quadruple stream.
  let logical: std::vec::Vec<u16> = (0..8 * 4 * 4)
    .map(|i| match i % 4 {
      0 => 0x8000u16, // U
      1 => 0x4000u16, // Y
      2 => 0xA000u16, // V
      _ => 0x0000u16, // X = padding
    })
    .collect();
  let pix_le: std::vec::Vec<u16> = logical.iter().map(|&v| as_le_u16(v)).collect();
  let pix_be: std::vec::Vec<u16> = logical.iter().map(|&v| as_be_u16(v)).collect();

  let frame_le = Xv48LeFrame::try_new(&pix_le, 8, 4, 8 * 4).unwrap();
  let mut out_le = std::vec![0u8; 8 * 4 * 4];
  let mut sink_le = MixedSinker::<Xv48>::new(8, 4)
    .with_simd(false)
    .with_rgba(&mut out_le)
    .unwrap();
  xv48_to(
    &frame_le,
    true,
    sink_le.set_kernel_matrix(KernelMatrix::Bt709),
  )
  .unwrap();

  let frame_be = Xv48BeFrame::try_new(&pix_be, 8, 4, 8 * 4).unwrap();
  let mut out_be = std::vec![0u8; 8 * 4 * 4];
  let mut sink_be = MixedSinker::<Xv48<true>>::new(8, 4)
    .with_simd(false)
    .with_rgba(&mut out_be)
    .unwrap();
  xv48_to_endian(
    &frame_be,
    true,
    sink_be.set_kernel_matrix(KernelMatrix::Bt709),
  )
  .unwrap();

  assert_eq!(
    out_le, out_be,
    "Xv48 LE/BE outputs diverge — `<const BE>` propagation broken"
  );
}

// ---- Atomicity (#308): RGB-scratch preflight before output writes -------
//
// Xv48's identity `process` hoists the RGB-scratch reservation above every
// output write (luma, luma_u16, then the u16 RGB / RGBA fan-out), so an
// allocator refusal of the scratch grow surfaces as a recoverable
// `AllocationFailed` BEFORE any output plane is touched. The allocating
// (rgb = None) arm of `rgb_row_buf_or_scratch` is reached only at
// `want_hsv && want_rgba && !want_rgb` (HSV-only routes through the direct
// `xv48_to_hsv_row` kernel and never allocates): attach luma + luma_u16 + rgba
// + hsv (no rgb) and arm the shared failpoint. `yuva`-gated (shares the crate's
// RGB-scratch failpoint).
#[cfg(all(test, feature = "std", feature = "yuva"))]
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn xv48_rgb_scratch_alloc_failure_leaves_outputs_untouched() {
  use crate::resample::ResampleError;

  let buf = solid_xv48_frame(16, 8, 0x8000, 0x8000, 0x8000, 0);
  let src = Xv48Frame::new(&buf, 16, 8, 64);
  let mut luma = std::vec![0xABu8; 16 * 8];
  let mut luma_u16 = std::vec![0xABCDu16; 16 * 8];
  let mut rgba = std::vec![0xCDu8; 16 * 8 * 4];
  let (mut hh, mut ss, mut vv) = (
    std::vec![0xCDu8; 16 * 8],
    std::vec![0xCDu8; 16 * 8],
    std::vec![0xCDu8; 16 * 8],
  );
  let mut sink = MixedSinker::<Xv48>::new(16, 8)
    .with_luma(&mut luma)
    .unwrap()
    .with_luma_u16(&mut luma_u16)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap()
    .with_hsv(&mut hh, &mut ss, &mut vv)
    .unwrap();

  super::super::arm_rgb_scratch_alloc_failure();
  let err = xv48_to(&src, false, sink.set_kernel_matrix(KernelMatrix::Bt601)).unwrap_err();
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
