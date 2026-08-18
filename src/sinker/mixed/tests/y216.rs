//! Tier 4 Y216 sinker tests — Ship 11d.
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

// ---- Solid-color Y216 builder -----------------------------------------

/// Builds a solid-color Y216 plane with one (Y, U, V) repeated. Each
/// row is `width x 2` u16 elements (`Y₀, U, Y₁, V` quadruples). All
/// samples are direct 16-bit values — Y216 uses the full u16 range
/// with no MSB-alignment shift (bits[15:0] all active).
///
/// Width must be even (4:2:2 chroma pair).
pub(super) fn solid_y216_frame(width: u32, height: u32, y: u16, u: u16, v: u16) -> Vec<u16> {
  assert!(width.is_multiple_of(2), "Y216 requires even width");
  let row_elems = (width as usize) * 2;
  let mut buf = std::vec![0u16; row_elems * height as usize];
  for row in 0..height as usize {
    let off = row * row_elems;
    for q in 0..(width as usize / 2) {
      let base = off + q * 4;
      buf[base] = y;
      buf[base + 1] = u;
      buf[base + 2] = y;
      buf[base + 3] = v;
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
fn y216_luma_only_extracts_y_bytes_downshifted() {
  // Y=51200 (16-bit direct) → 8-bit (51200 >> 8) = 200.
  let buf = solid_y216_frame(6, 8, 51200, 32768, 32768);
  let src = Y216Frame::new(&buf, 6, 8, 12);
  let mut luma = std::vec![0u8; 6 * 8];
  let mut sink = MixedSinker::<Y216>::new(6, 8).with_luma(&mut luma).unwrap();
  y216_to(&src, true, KernelMatrix::Bt601, &mut sink).unwrap();
  // 16-bit Y=51200 → 8-bit (51200 >> 8) = 200.
  assert!(luma.iter().all(|&y| y == 200), "luma {luma:?}");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn y216_luma_u16_only_extracts_y_native_depth() {
  // Y216 luma_u16 is a direct extract — no shift.
  let buf = solid_y216_frame(6, 8, 51200, 32768, 32768);
  let src = Y216Frame::new(&buf, 6, 8, 12);
  let mut luma = std::vec![0u16; 6 * 8];
  let mut sink = MixedSinker::<Y216>::new(6, 8)
    .with_luma_u16(&mut luma)
    .unwrap();
  y216_to(&src, true, KernelMatrix::Bt601, &mut sink).unwrap();
  assert!(
    luma.iter().all(|&y| y == 51200),
    "luma_u16 {:?}",
    &luma[..16]
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn y216_rgb_only_converts_gray_to_gray() {
  // Y=32896 (≈ 128/255 x 65535), U=V=32768 (neutral chroma, ≈ 0.5x65535).
  let buf = solid_y216_frame(12, 4, 32896, 32768, 32768);
  let src = Y216Frame::new(&buf, 12, 4, 24);
  let mut rgb = std::vec![0u8; 12 * 4 * 3];
  let mut sink = MixedSinker::<Y216>::new(12, 4).with_rgb(&mut rgb).unwrap();
  y216_to(&src, true, KernelMatrix::Bt601, &mut sink).unwrap();
  for px in rgb.chunks(3) {
    assert!(px[0].abs_diff(128) <= 2);
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn y216_rgba_only_converts_gray_to_gray_with_opaque_alpha() {
  let buf = solid_y216_frame(12, 4, 32896, 32768, 32768);
  let src = Y216Frame::new(&buf, 12, 4, 24);
  let mut rgba = std::vec![0u8; 12 * 4 * 4];
  let mut sink = MixedSinker::<Y216>::new(12, 4)
    .with_rgba(&mut rgba)
    .unwrap();
  y216_to(&src, true, KernelMatrix::Bt601, &mut sink).unwrap();
  for px in rgba.chunks(4) {
    assert_eq!(px[3], 0xFF);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn y216_rgb_u16_only_converts_gray_to_gray_native_depth() {
  let buf = solid_y216_frame(12, 4, 32896, 32768, 32768);
  let src = Y216Frame::new(&buf, 12, 4, 24);
  let mut rgb = std::vec![0u16; 12 * 4 * 3];
  let mut sink = MixedSinker::<Y216>::new(12, 4)
    .with_rgb_u16(&mut rgb)
    .unwrap();
  y216_to(&src, true, KernelMatrix::Bt601, &mut sink).unwrap();
  for px in rgb.chunks(3) {
    // Mid-gray at 16-bit depth ≈ 32768; allow ±256 tolerance for
    // Q15 rounding on the 16-bit pipeline.
    assert!(
      px[0].abs_diff(32768) <= 512,
      "expected ~32768, got {}",
      px[0]
    );
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn y216_rgba_u16_alpha_is_max() {
  let buf = solid_y216_frame(12, 4, 32896, 32768, 32768);
  let src = Y216Frame::new(&buf, 12, 4, 24);
  let mut rgba = std::vec![0u16; 12 * 4 * 4];
  let mut sink = MixedSinker::<Y216>::new(12, 4)
    .with_rgba_u16(&mut rgba)
    .unwrap();
  y216_to(&src, true, KernelMatrix::Bt601, &mut sink).unwrap();
  for px in rgba.chunks(4) {
    assert_eq!(px[3], 0xFFFF);
  }
}

// ---- Strategy A invariant tests ---------------------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn y216_with_rgb_and_with_rgba_byte_identical_u8() {
  // Strategy A invariant on u8 path — calling both `with_rgb` and
  // `with_rgba` must produce the same RGB bytes in both buffers, with
  // alpha = 0xFF in the RGBA buffer.
  let w = 12u32;
  let h = 4u32;
  let buf = solid_y216_frame(w, h, 45000, 20000, 50000);
  let src = Y216Frame::new(&buf, w, h, w * 2);
  let mut rgb = std::vec![0u8; (w * h) as usize * 3];
  let mut rgba = std::vec![0u8; (w * h) as usize * 4];
  let mut sink = MixedSinker::<Y216>::new(w as usize, h as usize)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap();
  y216_to(&src, true, KernelMatrix::Bt601, &mut sink).unwrap();
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
fn y216_with_rgb_u16_and_with_rgba_u16_byte_identical() {
  // Strategy A invariant on u16 path — alpha must be 0xFFFF (full
  // 16-bit opaque).
  let w = 12u32;
  let h = 4u32;
  let buf = solid_y216_frame(w, h, 45000, 20000, 50000);
  let src = Y216Frame::new(&buf, w, h, w * 2);
  let mut rgb = std::vec![0u16; (w * h) as usize * 3];
  let mut rgba = std::vec![0u16; (w * h) as usize * 4];
  let mut sink = MixedSinker::<Y216>::new(w as usize, h as usize)
    .with_rgb_u16(&mut rgb)
    .unwrap()
    .with_rgba_u16(&mut rgba)
    .unwrap();
  y216_to(&src, true, KernelMatrix::Bt601, &mut sink).unwrap();
  for i in 0..(w * h) as usize {
    assert_eq!(rgba[i * 4], rgb[i * 3]);
    assert_eq!(rgba[i * 4 + 1], rgb[i * 3 + 1]);
    assert_eq!(rgba[i * 4 + 2], rgb[i * 3 + 2]);
    assert_eq!(rgba[i * 4 + 3], 0xFFFF);
  }
}

// ---- SIMD-vs-scalar parity --------------------------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn y216_with_simd_false_matches_with_simd_true() {
  // Pseudo-random Y216 across multiple widths covering the main loop
  // + scalar tail of every backend block size. Each u16 sample uses
  // the full 16-bit range — no MSB alignment needed for Y216.
  for w in [2usize, 4, 14, 16, 18, 30, 32, 34, 1920, 1922] {
    let h = 2usize;
    let mut buf = std::vec![0u16; w * 2 * h];
    pseudo_random_u16_low_n_bits(&mut buf, 0xC0FFEE, 16);
    let src = Y216Frame::new(&buf, w as u32, h as u32, (w * 2) as u32);

    let mut rgb_simd = std::vec![0u8; w * h * 3];
    let mut rgb_scalar = std::vec![0u8; w * h * 3];
    let mut sink_simd = MixedSinker::<Y216>::new(w, h)
      .with_rgb(&mut rgb_simd)
      .unwrap();
    let mut sink_scalar = MixedSinker::<Y216>::new(w, h)
      .with_rgb(&mut rgb_scalar)
      .unwrap()
      .with_simd(false);
    y216_to(&src, false, KernelMatrix::Bt709, &mut sink_simd).unwrap();
    y216_to(&src, false, KernelMatrix::Bt709, &mut sink_scalar).unwrap();
    assert_eq!(rgb_simd, rgb_scalar, "Y216 SIMD≠scalar at width {w}");
  }
}

// ---- Error-path tests --------------------------------------------------

#[test]
fn y216_luma_u16_buffer_too_short_returns_err() {
  // Buffer holds 6x7 = 42 elements; a 6x8 frame needs 48.
  let mut luma = std::vec![0u16; 6 * 7];
  let result = MixedSinker::<Y216>::new(6, 8).with_luma_u16(&mut luma);
  let Err(err) = result else {
    panic!("expected InsufficientLumaU16Buffer");
  };
  assert_eq!(
    err,
    MixedSinkerError::InsufficientLumaU16Buffer(InsufficientBuffer::new(48, 42))
  );
}

// ---- Planar parity oracle ---------------------------------------------------

/// Pack three 16-bit planes (Y / U / V at 4:2:2 subsampling) into Y216
/// layout — each row is `width x 2` u16 elements laid out as `(Y₀, U,
/// Y₁, V)` quadruples. Y216 uses the full u16 range with no alignment
/// shift; samples are stored direct. Width must be even.
#[cfg(feature = "yuv-planar")]
fn pack_yuv422p16_to_y216(
  y_plane: &[u16],
  u_plane: &[u16],
  v_plane: &[u16],
  width: usize,
  height: usize,
) -> Vec<u16> {
  let mut packed = std::vec::Vec::with_capacity(width * 2 * height);
  for r in 0..height {
    for c in (0..width).step_by(2) {
      packed.push(y_plane[r * width + c]); // Y₀
      packed.push(u_plane[r * (width / 2) + c / 2]); // U
      packed.push(y_plane[r * width + c + 1]); // Y₁
      packed.push(v_plane[r * (width / 2) + c / 2]); // V
    }
  }
  packed
}

// Oracle compares against the planar `Yuv422p16` source, so this case
// only builds when that family is present.
#[cfg(feature = "yuv-planar")]
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn y216_planar_parity_with_yuv422p16() {
  // Oracle: Yuv422p16 (separate planes) and Y216 (interleaved u16 YUYV)
  // carry identical logical samples — both paths MUST produce
  // byte-identical RGB output. This exercises the i64 chroma pipeline
  // introduced for Y216 (the main point of the kernel family).
  let width = 16usize;
  let height = 4usize;
  let yp: Vec<u16> = (0..(width * height)).map(|i| (i * 17) as u16).collect();
  let up: Vec<u16> = (0..((width / 2) * height))
    .map(|i| ((i + 100) * 13) as u16)
    .collect();
  let vp: Vec<u16> = (0..((width / 2) * height))
    .map(|i| ((i + 200) * 11) as u16)
    .collect();

  let planar = Yuv422p16Frame::new(
    &yp,
    &up,
    &vp,
    width as u32,
    height as u32,
    width as u32,
    (width / 2) as u32,
    (width / 2) as u32,
  );

  let packed = pack_yuv422p16_to_y216(&yp, &up, &vp, width, height);
  let y216 = Y216Frame::new(&packed, width as u32, height as u32, (width * 2) as u32);

  // u8 RGB parity
  let mut p_rgb = std::vec![0u8; width * height * 3];
  let mut y_rgb = std::vec![0u8; width * height * 3];
  let mut p_sink = MixedSinker::<Yuv422p16>::new(width, height)
    .with_rgb(&mut p_rgb)
    .unwrap();
  let mut y_sink = MixedSinker::<Y216>::new(width, height)
    .with_rgb(&mut y_rgb)
    .unwrap();
  yuv422p16_to(&planar, false, KernelMatrix::Bt709, &mut p_sink).unwrap();
  y216_to(&y216, false, KernelMatrix::Bt709, &mut y_sink).unwrap();
  assert_eq!(p_rgb, y_rgb, "Y216 vs Yuv422p16 u8 RGB diverges");

  // u16 RGB parity (the i64 chroma path — the whole point of this oracle)
  let mut p_rgb_u16 = std::vec![0u16; width * height * 3];
  let mut y_rgb_u16 = std::vec![0u16; width * height * 3];
  let mut p_sink2 = MixedSinker::<Yuv422p16>::new(width, height)
    .with_rgb_u16(&mut p_rgb_u16)
    .unwrap();
  let mut y_sink2 = MixedSinker::<Y216>::new(width, height)
    .with_rgb_u16(&mut y_rgb_u16)
    .unwrap();
  yuv422p16_to(&planar, false, KernelMatrix::Bt709, &mut p_sink2).unwrap();
  y216_to(&y216, false, KernelMatrix::Bt709, &mut y_sink2).unwrap();
  assert_eq!(p_rgb_u16, y_rgb_u16, "Y216 vs Yuv422p16 u16 RGB diverges");
}

// Frame BE flag — Y216 LE/BE round-trip parity test.
// Mirrors the Y210 / Y212 / V210 LE/BE tests; see y210.rs for the full pattern.

fn y216_as_be_u16(host: &[u16]) -> Vec<u16> {
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
fn y216_le_be_roundtrip_byte_identical() {
  // Mid-range full 16-bit samples (no MSB alignment for Y216).
  let intended = solid_y216_frame(8, 4, 38400, 25600, 44800);
  let pix_le = intended.clone();
  let pix_be = y216_as_be_u16(&intended);

  let frame_le = Y216LeFrame::try_new(&pix_le, 8, 4, 16).unwrap();
  let mut out_le_rgba = std::vec![0u8; 8 * 4 * 4];
  let mut out_le_luma_u16 = std::vec![0u16; 8 * 4];
  let mut sink_le = MixedSinker::<Y216>::new(8, 4)
    .with_simd(false)
    .with_rgba(&mut out_le_rgba)
    .unwrap()
    .with_luma_u16(&mut out_le_luma_u16)
    .unwrap();
  y216_to(&frame_le, true, KernelMatrix::Bt709, &mut sink_le).unwrap();

  let frame_be = Y216BeFrame::try_new(&pix_be, 8, 4, 16).unwrap();
  let mut out_be_rgba = std::vec![0u8; 8 * 4 * 4];
  let mut out_be_luma_u16 = std::vec![0u16; 8 * 4];
  let mut sink_be = MixedSinker::<Y216<true>>::new(8, 4)
    .with_simd(false)
    .with_rgba(&mut out_be_rgba)
    .unwrap()
    .with_luma_u16(&mut out_be_luma_u16)
    .unwrap();
  y216_to_endian(&frame_be, true, KernelMatrix::Bt709, &mut sink_be).unwrap();

  assert_eq!(
    out_le_rgba, out_be_rgba,
    "Y216 RGBA u8 LE/BE outputs diverge — `<const BE>` propagation broken"
  );
  assert_eq!(
    out_le_luma_u16, out_be_luma_u16,
    "Y216 luma u16 LE/BE outputs diverge — `<const BE>` propagation broken"
  );
}

// ---- Atomicity (#308): RGB-scratch preflight before output writes -------
//
// Y216's identity `process` hoists the RGB-scratch reservation above every
// output write (luma, luma_u16, then the u16 RGB / RGBA fan-out), so an
// allocator refusal of the scratch grow surfaces as a recoverable
// `AllocationFailed` BEFORE any output plane is touched. The allocating
// (rgb = None) arm of `rgb_row_buf_or_scratch` is reached only at
// `want_hsv && want_rgba && !want_rgb` (HSV-only routes through the direct
// `y216_to_hsv_row_endian` kernel and never allocates): attach luma + luma_u16
// + rgba + hsv (no rgb) and arm the shared failpoint. `yuva`-gated (shares the
// crate's RGB-scratch failpoint).
#[cfg(feature = "yuva")]
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn y216_rgb_scratch_alloc_failure_leaves_outputs_untouched() {
  use crate::resample::ResampleError;

  let buf = solid_y216_frame(16, 8, 32768, 32768, 32768);
  let src = Y216Frame::new(&buf, 16, 8, 32);
  let mut luma = std::vec![0xABu8; 16 * 8];
  let mut luma_u16 = std::vec![0xABCDu16; 16 * 8];
  let mut rgba = std::vec![0xCDu8; 16 * 8 * 4];
  let (mut hh, mut ss, mut vv) = (
    std::vec![0xCDu8; 16 * 8],
    std::vec![0xCDu8; 16 * 8],
    std::vec![0xCDu8; 16 * 8],
  );
  let mut sink = MixedSinker::<Y216>::new(16, 8)
    .with_luma(&mut luma)
    .unwrap()
    .with_luma_u16(&mut luma_u16)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap()
    .with_hsv(&mut hh, &mut ss, &mut vv)
    .unwrap();

  super::super::arm_rgb_scratch_alloc_failure();
  let err = y216_to(&src, false, KernelMatrix::Bt601, &mut sink).unwrap_err();
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
