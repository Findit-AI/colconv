//! Tier 4 Y212 sinker tests — Ship 11c.
//!
//! Coverage matrix:
//! - Single-output paths (luma u8, luma u16, rgb, rgba, rgb_u16,
//!   rgba_u16) on solid-gray frames.
//! - Strategy A invariant (`with_rgb` + `with_rgba` byte-identical;
//!   same for the u16 variants).
//! - SIMD-vs-scalar parity across multiple widths covering the main
//!   loop + scalar tail of every backend block size.
//! - Three error-path tests: odd width, short packed slice, short
//!   luma_u16 buffer.

use super::*;

// ---- Solid-color Y212 builder -----------------------------------------

/// Builds a solid-color Y212 plane with one (Y, U, V) repeated. Each
/// row is `width x 2` u16 elements (`Y₀, U, Y₁, V` quadruples). All
/// samples are MSB-aligned 12-bit: each `u16` value's active 12 bits
/// occupy bits[15:4] and bits[3:0] are zero.
///
/// Width must be even (4:2:2 chroma pair).
pub(super) fn solid_y212_frame(width: u32, height: u32, y: u16, u: u16, v: u16) -> Vec<u16> {
  assert!(width.is_multiple_of(2), "Y212 requires even width");
  let row_elems = (width as usize) * 2;
  let mut buf = std::vec![0u16; row_elems * height as usize];
  let y_msb = (y & 0xFFF) << 4;
  let u_msb = (u & 0xFFF) << 4;
  let v_msb = (v & 0xFFF) << 4;
  for row in 0..height as usize {
    let off = row * row_elems;
    for q in 0..(width as usize / 2) {
      let base = off + q * 4;
      buf[base] = y_msb;
      buf[base + 1] = u_msb;
      buf[base + 2] = y_msb;
      buf[base + 3] = v_msb;
    }
  }
  buf
}

// ---- Single-output gray-to-gray tests ---------------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn y212_luma_only_extracts_y_bytes_downshifted() {
  let buf = solid_y212_frame(6, 8, 200, 2048, 2048); // Y=200 (12-bit native).
  let src = Y212Frame::new(&buf, 6, 8, 12);
  let mut luma = std::vec![0u8; 6 * 8];
  let mut sink = MixedSinker::<Y212>::new(6, 8).with_luma(&mut luma).unwrap();
  y212_to(&src, true, KernelMatrix::Bt601, &mut sink).unwrap();
  // 12-bit Y=200 → 8-bit (200 >> 4) = 12.
  assert!(luma.iter().all(|&y| y == 12), "luma {luma:?}");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn y212_luma_u16_only_extracts_y_native_depth() {
  let buf = solid_y212_frame(6, 8, 200, 2048, 2048);
  let src = Y212Frame::new(&buf, 6, 8, 12);
  let mut luma = std::vec![0u16; 6 * 8];
  let mut sink = MixedSinker::<Y212>::new(6, 8)
    .with_luma_u16(&mut luma)
    .unwrap();
  y212_to(&src, true, KernelMatrix::Bt601, &mut sink).unwrap();
  assert!(luma.iter().all(|&y| y == 200), "luma_u16 {:?}", &luma[..16]);
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn y212_rgb_only_converts_gray_to_gray() {
  let buf = solid_y212_frame(12, 4, 2048, 2048, 2048);
  let src = Y212Frame::new(&buf, 12, 4, 24);
  let mut rgb = std::vec![0u8; 12 * 4 * 3];
  let mut sink = MixedSinker::<Y212>::new(12, 4).with_rgb(&mut rgb).unwrap();
  y212_to(&src, true, KernelMatrix::Bt601, &mut sink).unwrap();
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
fn y212_rgba_only_converts_gray_to_gray_with_opaque_alpha() {
  let buf = solid_y212_frame(12, 4, 2048, 2048, 2048);
  let src = Y212Frame::new(&buf, 12, 4, 24);
  let mut rgba = std::vec![0u8; 12 * 4 * 4];
  let mut sink = MixedSinker::<Y212>::new(12, 4)
    .with_rgba(&mut rgba)
    .unwrap();
  y212_to(&src, true, KernelMatrix::Bt601, &mut sink).unwrap();
  for px in rgba.chunks(4) {
    assert_eq!(px[3], 0xFF);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn y212_rgb_u16_only_converts_gray_to_gray_native_depth() {
  let buf = solid_y212_frame(12, 4, 2048, 2048, 2048);
  let src = Y212Frame::new(&buf, 12, 4, 24);
  let mut rgb = std::vec![0u16; 12 * 4 * 3];
  let mut sink = MixedSinker::<Y212>::new(12, 4)
    .with_rgb_u16(&mut rgb)
    .unwrap();
  y212_to(&src, true, KernelMatrix::Bt601, &mut sink).unwrap();
  for px in rgb.chunks(3) {
    assert!(px[0].abs_diff(2048) <= 2, "expected ~2048, got {}", px[0]);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn y212_rgba_u16_alpha_is_max() {
  let buf = solid_y212_frame(12, 4, 2048, 2048, 2048);
  let src = Y212Frame::new(&buf, 12, 4, 24);
  let mut rgba = std::vec![0u16; 12 * 4 * 4];
  let mut sink = MixedSinker::<Y212>::new(12, 4)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  y212_to(&src, true, KernelMatrix::Bt601, &mut sink).unwrap();
  for px in rgba.chunks(4) {
    assert_eq!(px[3], 4095);
  }
}

// ---- Strategy A invariant tests ---------------------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn y212_with_rgb_and_with_rgba_byte_identical_u8() {
  // Strategy A invariant on u8 path — calling both `with_rgb` and
  // `with_rgba` must produce the same RGB bytes in both buffers, with
  // alpha = 0xFF in the RGBA buffer.
  let w = 12u32;
  let h = 4u32;
  let buf = solid_y212_frame(w, h, 700, 400, 600);
  let src = Y212Frame::new(&buf, w, h, w * 2);
  let mut rgb = std::vec![0u8; (w * h) as usize * 3];
  let mut rgba = std::vec![0u8; (w * h) as usize * 4];
  let mut sink = MixedSinker::<Y212>::new(w as usize, h as usize)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap();
  y212_to(&src, true, KernelMatrix::Bt601, &mut sink).unwrap();
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
fn y212_with_rgb_u16_and_with_rgba_u16_byte_identical() {
  // Strategy A invariant on u16 path.
  let w = 12u32;
  let h = 4u32;
  let buf = solid_y212_frame(w, h, 700, 400, 600);
  let src = Y212Frame::new(&buf, w, h, w * 2);
  let mut rgb = std::vec![0u16; (w * h) as usize * 3];
  let mut rgba = std::vec![0u16; (w * h) as usize * 4];
  let mut sink = MixedSinker::<Y212>::new(w as usize, h as usize)
    .with_rgb_u16(&mut rgb)
    .unwrap()
    .with_rgba_u16(&mut rgba)
    .unwrap();
  y212_to(&src, true, KernelMatrix::Bt601, &mut sink).unwrap();
  for i in 0..(w * h) as usize {
    assert_eq!(rgba[i * 4], rgb[i * 3]);
    assert_eq!(rgba[i * 4 + 1], rgb[i * 3 + 1]);
    assert_eq!(rgba[i * 4 + 2], rgb[i * 3 + 2]);
    assert_eq!(rgba[i * 4 + 3], 4095);
  }
}

// ---- SIMD-vs-scalar parity --------------------------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn y212_with_simd_false_matches_with_simd_true() {
  // Pseudo-random Y212 across multiple widths covering the main loop
  // + scalar tail of every backend block size. Each u16 sample is
  // generated with 12 active bits, then shifted `<< 4` to become
  // MSB-aligned per the Y212 spec.
  for w in [2usize, 4, 14, 16, 18, 30, 32, 34, 1920, 1922] {
    let h = 2usize;
    let mut buf = std::vec![0u16; w * 2 * h];
    pseudo_random_u16_low_n_bits(&mut buf, 0xC0FFEE, 12);
    // MSB-align: each sample's active 12 bits move to bits[15:4].
    for s in &mut buf {
      *s <<= 4;
    }
    let src = Y212Frame::new(&buf, w as u32, h as u32, (w * 2) as u32);

    let mut rgb_simd = std::vec![0u8; w * h * 3];
    let mut rgb_scalar = std::vec![0u8; w * h * 3];
    let mut sink_simd = MixedSinker::<Y212>::new(w, h)
      .with_rgb(&mut rgb_simd)
      .unwrap();
    let mut sink_scalar = MixedSinker::<Y212>::new(w, h)
      .with_rgb(&mut rgb_scalar)
      .unwrap()
      .with_simd(false);
    y212_to(&src, false, KernelMatrix::Bt709, &mut sink_simd).unwrap();
    y212_to(&src, false, KernelMatrix::Bt709, &mut sink_scalar).unwrap();
    assert_eq!(rgb_simd, rgb_scalar, "Y212 SIMD≠scalar at width {w}");
  }
}

// ---- Error-path tests --------------------------------------------------

#[test]
fn y212_luma_u16_buffer_too_short_returns_err() {
  // Buffer holds 6x7 = 42 elements; a 6x8 frame needs 48.
  let mut luma = std::vec![0u16; 6 * 7];
  let result = MixedSinker::<Y212>::new(6, 8).with_luma_u16(&mut luma);
  let Err(err) = result else {
    panic!("expected InsufficientLumaU16Buffer");
  };
  assert_eq!(
    err,
    MixedSinkerError::InsufficientLumaU16Buffer(InsufficientBuffer::new(48, 42))
  );
}

// ---- Cross-format oracles ---------------------------------------------
//
// Planar parity: re-pack a Yuv422p12 source into Y212 layout and verify
// both produce byte-identical RGB. Y212 is just an MSB-aligned u16
// packing of 4:2:2 12-bit planar data, so the converted output MUST
// match Yuv422p12's output for the same samples.

/// Pack three 12-bit planes (Y / U / V at 4:2:2 subsampling) into the
/// Y212 layout — each row is `width x 2` u16 elements laid out as
/// `(Y₀, U, Y₁, V)` quadruples with each sample MSB-aligned (active 12
/// bits in bits[15:4]). Width must be even.
#[cfg(feature = "yuv-planar")]
fn pack_yuv422p12_to_y212(
  y: &[u16],
  u: &[u16],
  v: &[u16],
  width: usize,
  height: usize,
) -> Vec<u16> {
  let cw = width / 2;
  let mut out = std::vec![0u16; width * 2 * height];
  for row in 0..height {
    for c in 0..cw {
      let off = row * width * 2 + c * 4;
      out[off] = (y[row * width + 2 * c] << 4) & 0xFFF0;
      out[off + 1] = (u[row * cw + c] << 4) & 0xFFF0;
      out[off + 2] = (y[row * width + 2 * c + 1] << 4) & 0xFFF0;
      out[off + 3] = (v[row * cw + c] << 4) & 0xFFF0;
    }
  }
  out
}

// Oracle compares against the planar `Yuv422p12` source, so this case
// only builds when that family is present.
#[cfg(feature = "yuv-planar")]
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn y212_reconstructed_from_yuv422p12_matches_yuv422p12_to_rgb() {
  let w = 16usize;
  let h = 4usize;
  let mut y_plane = std::vec![0u16; w * h];
  let mut u_plane = std::vec![0u16; (w / 2) * h];
  let mut v_plane = std::vec![0u16; (w / 2) * h];
  pseudo_random_u16_low_n_bits(&mut y_plane, 0xC0FFEE, 12);
  pseudo_random_u16_low_n_bits(&mut u_plane, 0xBADF00D, 12);
  pseudo_random_u16_low_n_bits(&mut v_plane, 0xFEEDFACE, 12);

  let planar = Yuv422p12Frame::new(
    &y_plane,
    &u_plane,
    &v_plane,
    w as u32,
    h as u32,
    w as u32,
    (w / 2) as u32,
    (w / 2) as u32,
  );
  let packed = pack_yuv422p12_to_y212(&y_plane, &u_plane, &v_plane, w, h);
  let y212 = Y212Frame::new(&packed, w as u32, h as u32, (w * 2) as u32);

  let mut rgb_planar = std::vec![0u8; w * h * 3];
  let mut rgb_packed = std::vec![0u8; w * h * 3];
  let mut s_planar = MixedSinker::<Yuv422p12>::new(w, h)
    .with_rgb(&mut rgb_planar)
    .unwrap();
  let mut s_packed = MixedSinker::<Y212>::new(w, h)
    .with_rgb(&mut rgb_packed)
    .unwrap();
  yuv422p12_to(&planar, false, KernelMatrix::Bt709, &mut s_planar).unwrap();
  y212_to(&y212, false, KernelMatrix::Bt709, &mut s_packed).unwrap();

  assert_eq!(rgb_planar, rgb_packed);
}
// Phase 4 — Frame BE flag, Tier 4 Y212 LE/BE round-trip parity test.
// Mirrors the Y210 / V210 / Y216 LE/BE tests; see y210.rs for the full pattern.
fn y212_as_be_u16(host: &[u16]) -> Vec<u16> {
  host
    .iter()
    .map(|v| u16::from_ne_bytes(v.to_be_bytes()))
    .collect()
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn y212_le_be_roundtrip_byte_identical() {
  // Mid-range 12-bit Y/U/V samples (MSB-aligned).
  let intended = solid_y212_frame(8, 4, 2400, 1600, 2800);
  let pix_le = intended.clone();
  let pix_be = y212_as_be_u16(&intended);

  let frame_le = Y212LeFrame::try_new(&pix_le, 8, 4, 16).unwrap();
  let mut out_le_rgba = std::vec![0u8; 8 * 4 * 4];
  let mut out_le_luma_u16 = std::vec![0u16; 8 * 4];
  let mut sink_le = MixedSinker::<Y212>::new(8, 4)
    .with_simd(false)
    .with_rgba(&mut out_le_rgba)
    .unwrap()
    .with_luma_u16(&mut out_le_luma_u16)
    .unwrap();
  y212_to(&frame_le, true, KernelMatrix::Bt709, &mut sink_le).unwrap();

  let frame_be = Y212BeFrame::try_new(&pix_be, 8, 4, 16).unwrap();
  let mut out_be_rgba = std::vec![0u8; 8 * 4 * 4];
  let mut out_be_luma_u16 = std::vec![0u16; 8 * 4];
  let mut sink_be = MixedSinker::<Y212<true>>::new(8, 4)
    .with_simd(false)
    .with_rgba(&mut out_be_rgba)
    .unwrap()
    .with_luma_u16(&mut out_be_luma_u16)
    .unwrap();
  y212_to_endian(&frame_be, true, KernelMatrix::Bt709, &mut sink_be).unwrap();

  assert_eq!(
    out_le_rgba, out_be_rgba,
    "Y212 RGBA u8 LE/BE outputs diverge — `<const BE>` propagation broken"
  );
  assert_eq!(
    out_le_luma_u16, out_be_luma_u16,
    "Y212 luma u16 LE/BE outputs diverge — `<const BE>` propagation broken"
  );
}

// ---- Atomicity (#308): RGB-scratch preflight before output writes -------
//
// Y212's identity `process` hoists the RGB-scratch reservation above every
// output write (luma, luma_u16, then the u16 RGB / RGBA fan-out), so an
// allocator refusal of the scratch grow surfaces as a recoverable
// `AllocationFailed` BEFORE any output plane is touched. The allocating
// (rgb = None) arm of `rgb_row_buf_or_scratch` is reached only at
// `want_hsv && want_rgba && !want_rgb` (HSV-only routes through the direct
// `y212_to_hsv_row_endian` kernel and never allocates): attach luma + luma_u16
// + rgba + hsv (no rgb) and arm the shared failpoint. `yuva`-gated (shares the
// crate's RGB-scratch failpoint).
#[cfg(feature = "yuva")]
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn y212_rgb_scratch_alloc_failure_leaves_outputs_untouched() {
  use crate::resample::ResampleError;

  let buf = solid_y212_frame(16, 8, 2048, 2048, 2048);
  let src = Y212Frame::new(&buf, 16, 8, 32);
  let mut luma = std::vec![0xABu8; 16 * 8];
  let mut luma_u16 = std::vec![0xABCDu16; 16 * 8];
  let mut rgba = std::vec![0xCDu8; 16 * 8 * 4];
  let (mut hh, mut ss, mut vv) = (
    std::vec![0xCDu8; 16 * 8],
    std::vec![0xCDu8; 16 * 8],
    std::vec![0xCDu8; 16 * 8],
  );
  let mut sink = MixedSinker::<Y212>::new(16, 8)
    .with_luma(&mut luma)
    .unwrap()
    .with_luma_u16(&mut luma_u16)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap()
    .with_hsv(&mut hh, &mut ss, &mut vv)
    .unwrap();

  super::super::arm_rgb_scratch_alloc_failure();
  let err = y212_to(&src, false, KernelMatrix::Bt601, &mut sink).unwrap_err();
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
